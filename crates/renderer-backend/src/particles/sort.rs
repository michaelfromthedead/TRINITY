//! GPU Particle Depth Sorting for TRINITY Engine (T-GPU-5.4).
//!
//! Implements depth-based sorting of particles for correct alpha blending.
//! Uses radix sort with depth keys computed from particle positions relative
//! to the camera view direction.
//!
//! # Overview
//!
//! For correct transparency rendering, particles must be drawn back-to-front.
//! This module provides:
//!
//! 1. **Depth key computation**: GPU shader that computes sort keys from particle
//!    positions, where far particles get low keys (sorted first).
//!
//! 2. **Indirection array**: Instead of reordering particles in memory, we sort
//!    an index array that provides the rendering order.
//!
//! 3. **GpuRadixSort integration**: Leverages the existing radix sort infrastructure
//!    from `gpu_driven::sort` for efficient GPU-based sorting.
//!
//! # Algorithm
//!
//! 1. `compute_sort_keys` shader computes depth for each particle
//! 2. Depth is converted to a sortable 32-bit key (far = low, near = high)
//! 3. Indices are initialized to [0, 1, 2, ...]
//! 4. `GpuRadixSort` sorts keys with indices as payload
//! 5. Sorted indices provide back-to-front rendering order
//!
//! # Performance
//!
//! - Key computation: O(N) with high parallelism
//! - Radix sort: O(N) with 8 passes for 32-bit keys
//! - Target: <1ms for 100K particles on modern GPUs
//!
//! # Usage
//!
//! ```ignore
//! // Create sort pipeline (once)
//! let sort_pipeline = ParticleSortPipeline::new(&device, 100_000);
//!
//! // Each frame:
//! let params = ParticleSortParams::new(
//!     num_particles,
//!     camera_position,
//!     camera_forward,
//! );
//!
//! // Sort particles
//! sort_pipeline.sort(
//!     &device, &queue, &mut encoder,
//!     &particle_buffer, &params, num_particles,
//! );
//!
//! // Use sorted indices for rendering
//! render_pass.set_index_buffer(sort_pipeline.indices_out(), ...);
//! ```

use std::mem;
use wgpu::{Buffer, BufferUsages, ComputePipeline, Device};

use crate::gpu_driven::GpuRadixSort;
use crate::particles::Particle;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Compute shader workgroup size (must match WGSL constant).
pub const WORKGROUP_SIZE: u32 = 256;

/// Size of `ParticleSortParams` in bytes.
pub const SORT_PARAMS_SIZE: usize = 48;

/// Minimum particles for GPU sort (below this, use CPU sort).
pub const MIN_GPU_PARTICLES: u32 = 64;

/// Dead particle key value (sorts to end).
pub const DEAD_PARTICLE_KEY: u32 = 0xFFFFFFFF;

// ---------------------------------------------------------------------------
// ParticleSortParams
// ---------------------------------------------------------------------------

/// GPU uniform buffer for particle sort parameters.
///
/// Matches the WGSL `ParticleSortParams` struct layout.
///
/// # Memory Layout (48 bytes, std140 compatible)
///
/// | Offset | Field           | Size     |
/// |--------|-----------------|----------|
/// | 0      | num_particles   | 4 bytes  |
/// | 4      | _pad0           | 4 bytes  |
/// | 8      | _pad1           | 4 bytes  |
/// | 12     | _pad2           | 4 bytes  |
/// | 16     | camera_position | 12 bytes |
/// | 28     | near_plane      | 4 bytes  |
/// | 32     | view_direction  | 12 bytes |
/// | 44     | far_plane       | 4 bytes  |
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ParticleSortParams {
    /// Number of particles to sort.
    pub num_particles: u32,
    /// Padding for vec3 alignment.
    pub _pad0: u32,
    /// Padding for vec3 alignment.
    pub _pad1: u32,
    /// Padding for vec3 alignment.
    pub _pad2: u32,

    /// Camera world-space position.
    pub camera_position: [f32; 3],
    /// Near plane distance for depth normalization.
    pub near_plane: f32,

    /// Camera view direction (normalized, points into scene).
    pub view_direction: [f32; 3],
    /// Far plane distance for depth normalization.
    pub far_plane: f32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<ParticleSortParams>() == SORT_PARAMS_SIZE);

impl ParticleSortParams {
    /// Create sort parameters from camera data.
    ///
    /// # Arguments
    ///
    /// * `num_particles` - Number of particles to sort.
    /// * `camera_position` - Camera world-space position.
    /// * `view_direction` - Camera forward direction (normalized).
    pub fn new(
        num_particles: u32,
        camera_position: [f32; 3],
        view_direction: [f32; 3],
    ) -> Self {
        Self {
            num_particles,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            camera_position,
            near_plane: 0.1,
            view_direction,
            far_plane: 1000.0,
        }
    }

    /// Create sort parameters with explicit near/far planes.
    pub fn with_planes(
        num_particles: u32,
        camera_position: [f32; 3],
        view_direction: [f32; 3],
        near_plane: f32,
        far_plane: f32,
    ) -> Self {
        Self {
            num_particles,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            camera_position,
            near_plane,
            view_direction,
            far_plane,
        }
    }

    /// Get number of workgroups needed for key computation dispatch.
    #[inline]
    pub fn num_workgroups(&self) -> u32 {
        (self.num_particles + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
    }
}

impl Default for ParticleSortParams {
    fn default() -> Self {
        Self::new(0, [0.0, 0.0, 0.0], [0.0, 0.0, -1.0])
    }
}

// ---------------------------------------------------------------------------
// ParticleSortResources
// ---------------------------------------------------------------------------

/// GPU resources for particle depth sorting.
///
/// Contains buffers for sort keys, indices, and the bind group.
pub struct ParticleSortResources {
    /// Uniform buffer for sort parameters.
    pub params_buffer: Buffer,
    /// Storage buffer for sort keys (computed by shader).
    pub keys_buffer: Buffer,
    /// Storage buffer for sort indices (permutation).
    pub indices_buffer: Buffer,
    /// Maximum capacity in particles.
    pub capacity: u32,
    /// Bind group for key computation shader.
    pub bind_group: wgpu::BindGroup,
}

impl ParticleSortResources {
    /// Create sort resources with the given capacity.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `capacity` - Maximum number of particles.
    /// * `particle_buffer` - Particle buffer to read from.
    /// * `bind_group_layout` - Layout from `ParticleSortPipeline`.
    pub fn new(
        device: &Device,
        capacity: u32,
        particle_buffer: &Buffer,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle_sort_params"),
            size: SORT_PARAMS_SIZE as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let buffer_size = (capacity as u64) * 4; // 4 bytes per u32

        let keys_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle_sort_keys"),
            size: buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let indices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle_sort_indices"),
            size: buffer_size,
            usage: BufferUsages::STORAGE
                | BufferUsages::COPY_DST
                | BufferUsages::COPY_SRC
                | BufferUsages::INDEX,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("particle_sort_bind_group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: particle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: keys_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: indices_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            params_buffer,
            keys_buffer,
            indices_buffer,
            capacity,
            bind_group,
        }
    }

    /// Update sort parameters.
    pub fn update_params(&self, queue: &wgpu::Queue, params: &ParticleSortParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }
}

// ---------------------------------------------------------------------------
// ParticleSortPipeline
// ---------------------------------------------------------------------------

/// GPU compute pipeline for particle depth sorting.
///
/// Combines depth key computation with GpuRadixSort for complete sorting.
pub struct ParticleSortPipeline {
    /// Bind group layout for key computation shader.
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// Compute pipeline for depth key computation.
    pub compute_keys_pipeline: ComputePipeline,
    /// Compute pipeline for normalized depth keys.
    pub compute_keys_normalized_pipeline: ComputePipeline,
    /// Compute pipeline for distance-based keys.
    pub compute_keys_distance_pipeline: ComputePipeline,
    /// GPU radix sorter for key-index pairs.
    pub radix_sort: GpuRadixSort,
    /// Maximum particles this pipeline can handle.
    pub max_particles: u32,
}

impl ParticleSortPipeline {
    /// Create the particle sort pipeline.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `max_particles` - Maximum number of particles to sort.
    pub fn new(device: &Device, max_particles: u32) -> Self {
        // Create bind group layout for key computation
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("particle_sort_bind_group_layout"),
            entries: &[
                // binding 0: ParticleSortParams (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: particles (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: sort_keys (read_write storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 3: sort_indices (read_write storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("particle_sort_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Load shader module
        let shader_source = include_str!("../../shaders/particles/gpu_particle_sort.comp.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("particle_sort_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create compute pipelines
        let compute_keys_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("particle_sort_compute_keys"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "compute_sort_keys",
            compilation_options: Default::default(),
            cache: None,
        });

        let compute_keys_normalized_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("particle_sort_compute_keys_normalized"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: "compute_sort_keys_normalized",
                compilation_options: Default::default(),
                cache: None,
            });

        let compute_keys_distance_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("particle_sort_compute_keys_distance"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: "compute_sort_keys_distance",
                compilation_options: Default::default(),
                cache: None,
            });

        // Create radix sorter
        let radix_sort = GpuRadixSort::new(device, max_particles);

        Self {
            bind_group_layout,
            compute_keys_pipeline,
            compute_keys_normalized_pipeline,
            compute_keys_distance_pipeline,
            radix_sort,
            max_particles,
        }
    }

    /// Dispatch key computation shader.
    ///
    /// Computes depth keys and initializes indices. Does not perform sorting.
    pub fn dispatch_compute_keys(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        resources: &ParticleSortResources,
        num_particles: u32,
    ) {
        if num_particles == 0 {
            return;
        }

        let num_workgroups = (num_particles + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("particle_sort_compute_keys_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.compute_keys_pipeline);
        pass.set_bind_group(0, &resources.bind_group, &[]);
        pass.dispatch_workgroups(num_workgroups, 1, 1);
    }

    /// Full sort: compute keys, copy to radix sort buffers, sort, copy indices back.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `queue` - The wgpu queue.
    /// * `encoder` - Command encoder to record commands.
    /// * `resources` - Sort resources with particle buffer binding.
    /// * `params` - Sort parameters (camera position, direction).
    /// * `num_particles` - Number of particles to sort.
    pub fn sort(
        &self,
        device: &Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        resources: &ParticleSortResources,
        params: &ParticleSortParams,
        num_particles: u32,
    ) {
        if num_particles == 0 {
            return;
        }

        assert!(
            num_particles <= self.max_particles,
            "num_particles ({}) exceeds max_particles ({})",
            num_particles,
            self.max_particles
        );

        // Update parameters
        resources.update_params(queue, params);

        // Step 1: Compute depth keys and initialize indices
        self.dispatch_compute_keys(encoder, resources, num_particles);

        // Step 2: Copy keys and indices to radix sort input buffers
        let buffer_size = (num_particles as u64) * 4;
        encoder.copy_buffer_to_buffer(
            &resources.keys_buffer,
            0,
            self.radix_sort.keys_in(),
            0,
            buffer_size,
        );
        encoder.copy_buffer_to_buffer(
            &resources.indices_buffer,
            0,
            self.radix_sort.values_in(),
            0,
            buffer_size,
        );

        // Step 3: Execute radix sort
        self.radix_sort.sort(device, queue, encoder, num_particles);

        // Step 4: Copy sorted indices back to resources
        // (Keys are discarded - we only need the sorted indices for rendering)
        encoder.copy_buffer_to_buffer(
            self.radix_sort.values_out(),
            0,
            &resources.indices_buffer,
            0,
            buffer_size,
        );
    }

    /// Get the bind group layout for creating resources.
    #[inline]
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Get maximum particles this pipeline can handle.
    #[inline]
    pub fn max_particles(&self) -> u32 {
        self.max_particles
    }
}

// ---------------------------------------------------------------------------
// SortMode
// ---------------------------------------------------------------------------

/// Sorting mode for particle depth sorting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SortMode {
    /// Sort by depth along view direction (default).
    /// Uses float-to-bits conversion for precise ordering.
    #[default]
    ViewDepth,
    /// Sort by normalized depth in [near, far] range.
    /// Useful for predictable depth ranges.
    Normalized,
    /// Sort by radial distance from camera.
    /// Useful for omnidirectional effects.
    Distance,
}

// ---------------------------------------------------------------------------
// CPU Reference Implementation
// ---------------------------------------------------------------------------

/// Compute depth key for a particle position (CPU reference).
///
/// Far particles get low keys, near particles get high keys (back-to-front).
#[inline]
pub fn cpu_compute_depth_key(
    position: [f32; 3],
    camera_position: [f32; 3],
    view_direction: [f32; 3],
) -> u32 {
    // Compute distance along view direction
    let to_particle = [
        position[0] - camera_position[0],
        position[1] - camera_position[1],
        position[2] - camera_position[2],
    ];

    let depth =
        to_particle[0] * view_direction[0] +
        to_particle[1] * view_direction[1] +
        to_particle[2] * view_direction[2];

    // Handle particles behind camera
    if depth < 0.0 {
        return DEAD_PARTICLE_KEY;
    }

    // Convert float to sortable uint
    let bits = depth.to_bits();
    let sortable = bits ^ 0x80000000;

    // Flip for back-to-front (far = low key)
    0xFFFFFFFF - sortable
}

/// Compute distance-based sort key (CPU reference).
#[inline]
pub fn cpu_compute_distance_key(position: [f32; 3], camera_position: [f32; 3]) -> u32 {
    let dx = position[0] - camera_position[0];
    let dy = position[1] - camera_position[1];
    let dz = position[2] - camera_position[2];
    let distance = (dx * dx + dy * dy + dz * dz).sqrt();

    let bits = distance.to_bits();
    let sortable = bits ^ 0x80000000;
    0xFFFFFFFF - sortable
}

/// CPU reference implementation of particle depth sorting.
///
/// Returns sorted indices (indirection array) for back-to-front rendering.
pub fn cpu_sort_particles(
    particles: &[Particle],
    camera_position: [f32; 3],
    view_direction: [f32; 3],
) -> Vec<u32> {
    let n = particles.len();
    if n == 0 {
        return Vec::new();
    }

    // Compute keys and indices
    let mut key_index_pairs: Vec<(u32, u32)> = particles
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let key = if p.is_alive() {
                cpu_compute_depth_key(p.position, camera_position, view_direction)
            } else {
                DEAD_PARTICLE_KEY
            };
            (key, i as u32)
        })
        .collect();

    // Sort by key (stable sort preserves order for equal keys)
    key_index_pairs.sort_by_key(|(k, _)| *k);

    // Extract sorted indices
    key_index_pairs.iter().map(|(_, i)| *i).collect()
}

/// CPU reference implementation with radix sort for consistency.
pub fn cpu_radix_sort_particles(
    particles: &[Particle],
    camera_position: [f32; 3],
    view_direction: [f32; 3],
) -> (Vec<u32>, Vec<u32>) {
    let n = particles.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    // Compute keys
    let mut keys: Vec<u32> = particles
        .iter()
        .map(|p| {
            if p.is_alive() {
                cpu_compute_depth_key(p.position, camera_position, view_direction)
            } else {
                DEAD_PARTICLE_KEY
            }
        })
        .collect();

    // Initialize indices
    let mut indices: Vec<u32> = (0..n as u32).collect();

    // Radix sort (8 passes, 4-bit radix)
    let mut out_keys = vec![0u32; n];
    let mut out_indices = vec![0u32; n];

    for pass in 0..8u32 {
        let shift = pass * 4;
        let mut counts = [0usize; 16];

        // Count occurrences
        for &k in keys.iter() {
            let digit = ((k >> shift) & 0xF) as usize;
            counts[digit] += 1;
        }

        // Exclusive prefix sum
        let mut total = 0usize;
        for c in counts.iter_mut() {
            let old = *c;
            *c = total;
            total += old;
        }

        // Scatter
        for i in 0..n {
            let digit = ((keys[i] >> shift) & 0xF) as usize;
            let dest = counts[digit];
            counts[digit] += 1;
            out_keys[dest] = keys[i];
            out_indices[dest] = indices[i];
        }

        // Swap buffers
        std::mem::swap(&mut keys, &mut out_keys);
        std::mem::swap(&mut indices, &mut out_indices);
    }

    (keys, indices)
}

/// Verify that sorted indices produce back-to-front ordering.
pub fn verify_back_to_front(
    particles: &[Particle],
    sorted_indices: &[u32],
    camera_position: [f32; 3],
    view_direction: [f32; 3],
) -> bool {
    if sorted_indices.len() < 2 {
        return true;
    }

    let mut prev_depth = f32::MAX;

    for &idx in sorted_indices {
        let particle = &particles[idx as usize];

        if !particle.is_alive() {
            // Dead particles should be at the end
            continue;
        }

        let to_particle = [
            particle.position[0] - camera_position[0],
            particle.position[1] - camera_position[1],
            particle.position[2] - camera_position[2],
        ];

        let depth = to_particle[0] * view_direction[0]
            + to_particle[1] * view_direction[1]
            + to_particle[2] * view_direction[2];

        // Back-to-front: depth should decrease (or stay same)
        if depth > prev_depth + 1e-5 {
            return false;
        }
        prev_depth = depth;
    }

    true
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particles::PARTICLE_FLAG_ALIVE;

    // Helper to create test particle
    fn make_particle(position: [f32; 3], alive: bool) -> Particle {
        Particle {
            position,
            age: 0.0,
            velocity: [0.0, 0.0, 0.0],
            lifetime: 1.0,
            color: [1.0, 1.0, 1.0, 1.0],
            size: 0.1,
            rotation: 0.0,
            rotation_speed: 0.0,
            flags: if alive { PARTICLE_FLAG_ALIVE } else { 0 },
        }
    }

    // ── ParticleSortParams ─────────────────────────────────────────────────

    #[test]
    fn test_sort_params_size() {
        assert_eq!(mem::size_of::<ParticleSortParams>(), SORT_PARAMS_SIZE);
        assert_eq!(mem::align_of::<ParticleSortParams>(), 4);
    }

    #[test]
    fn test_sort_params_new() {
        let params = ParticleSortParams::new(
            1000,
            [0.0, 5.0, 10.0],
            [0.0, 0.0, -1.0],
        );
        assert_eq!(params.num_particles, 1000);
        assert_eq!(params.camera_position, [0.0, 5.0, 10.0]);
        assert_eq!(params.view_direction, [0.0, 0.0, -1.0]);
        assert!((params.near_plane - 0.1).abs() < f32::EPSILON);
        assert!((params.far_plane - 1000.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_sort_params_with_planes() {
        let params = ParticleSortParams::with_planes(
            500,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            0.5,
            500.0,
        );
        assert_eq!(params.num_particles, 500);
        assert!((params.near_plane - 0.5).abs() < f32::EPSILON);
        assert!((params.far_plane - 500.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_sort_params_workgroups() {
        assert_eq!(ParticleSortParams::new(1, [0.0; 3], [0.0; 3]).num_workgroups(), 1);
        assert_eq!(ParticleSortParams::new(256, [0.0; 3], [0.0; 3]).num_workgroups(), 1);
        assert_eq!(ParticleSortParams::new(257, [0.0; 3], [0.0; 3]).num_workgroups(), 2);
        assert_eq!(ParticleSortParams::new(512, [0.0; 3], [0.0; 3]).num_workgroups(), 2);
        assert_eq!(ParticleSortParams::new(513, [0.0; 3], [0.0; 3]).num_workgroups(), 3);
    }

    // ── Depth Key Computation ──────────────────────────────────────────────

    #[test]
    fn test_depth_key_far_vs_near() {
        let camera = [0.0, 0.0, 0.0];
        let view_dir = [0.0, 0.0, 1.0]; // Looking +Z

        let near_pos = [0.0, 0.0, 5.0];
        let far_pos = [0.0, 0.0, 100.0];

        let near_key = cpu_compute_depth_key(near_pos, camera, view_dir);
        let far_key = cpu_compute_depth_key(far_pos, camera, view_dir);

        // Far should have lower key (sorts first for back-to-front)
        assert!(far_key < near_key, "far_key {} should be < near_key {}", far_key, near_key);
    }

    #[test]
    fn test_depth_key_behind_camera() {
        let camera = [0.0, 0.0, 0.0];
        let view_dir = [0.0, 0.0, 1.0]; // Looking +Z

        let behind = [0.0, 0.0, -5.0]; // Behind camera
        let key = cpu_compute_depth_key(behind, camera, view_dir);

        assert_eq!(key, DEAD_PARTICLE_KEY, "particles behind camera should get max key");
    }

    #[test]
    fn test_depth_key_same_depth() {
        let camera = [0.0, 0.0, 0.0];
        let view_dir = [0.0, 0.0, 1.0];

        // Both at same Z (depth)
        let pos1 = [5.0, 0.0, 50.0];
        let pos2 = [-5.0, 0.0, 50.0];

        let key1 = cpu_compute_depth_key(pos1, camera, view_dir);
        let key2 = cpu_compute_depth_key(pos2, camera, view_dir);

        // Same depth = same key
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_distance_key() {
        let camera = [0.0, 0.0, 0.0];

        let near = [5.0, 0.0, 0.0];
        let far = [100.0, 0.0, 0.0];

        let near_key = cpu_compute_distance_key(near, camera);
        let far_key = cpu_compute_distance_key(far, camera);

        assert!(far_key < near_key, "far_key {} should be < near_key {}", far_key, near_key);
    }

    // ── CPU Sorting ────────────────────────────────────────────────────────

    #[test]
    fn test_cpu_sort_back_to_front() {
        let camera = [0.0, 0.0, 0.0];
        let view_dir = [0.0, 0.0, 1.0];

        // Particles at different depths
        let particles = vec![
            make_particle([0.0, 0.0, 10.0], true),  // Near
            make_particle([0.0, 0.0, 100.0], true), // Far
            make_particle([0.0, 0.0, 50.0], true),  // Middle
        ];

        let sorted = cpu_sort_particles(&particles, camera, view_dir);

        // Should be: far (1), middle (2), near (0)
        assert_eq!(sorted, vec![1, 2, 0]);
    }

    #[test]
    fn test_cpu_sort_dead_particles_last() {
        let camera = [0.0, 0.0, 0.0];
        let view_dir = [0.0, 0.0, 1.0];

        let particles = vec![
            make_particle([0.0, 0.0, 10.0], false), // Dead (near)
            make_particle([0.0, 0.0, 100.0], true), // Alive (far)
            make_particle([0.0, 0.0, 50.0], true),  // Alive (middle)
        ];

        let sorted = cpu_sort_particles(&particles, camera, view_dir);

        // Dead particle should be last
        assert_eq!(sorted[2], 0, "dead particle should be last");
    }

    #[test]
    fn test_cpu_sort_empty() {
        let sorted = cpu_sort_particles(&[], [0.0; 3], [0.0, 0.0, 1.0]);
        assert!(sorted.is_empty());
    }

    #[test]
    fn test_cpu_sort_single() {
        let particles = vec![make_particle([0.0, 0.0, 10.0], true)];
        let sorted = cpu_sort_particles(&particles, [0.0; 3], [0.0, 0.0, 1.0]);
        assert_eq!(sorted, vec![0]);
    }

    #[test]
    fn test_cpu_radix_sort_consistency() {
        let camera = [0.0, 0.0, 0.0];
        let view_dir = [0.0, 0.0, 1.0];

        let particles: Vec<Particle> = (0..100)
            .map(|i| make_particle([0.0, 0.0, (i * 10) as f32], true))
            .collect();

        let (_, radix_indices) = cpu_radix_sort_particles(&particles, camera, view_dir);
        let std_indices = cpu_sort_particles(&particles, camera, view_dir);

        // Both should produce same ordering
        assert_eq!(radix_indices, std_indices);
    }

    // ── Verification ───────────────────────────────────────────────────────

    #[test]
    fn test_verify_back_to_front_valid() {
        let camera = [0.0, 0.0, 0.0];
        let view_dir = [0.0, 0.0, 1.0];

        let particles = vec![
            make_particle([0.0, 0.0, 10.0], true),
            make_particle([0.0, 0.0, 100.0], true),
            make_particle([0.0, 0.0, 50.0], true),
        ];

        let sorted = cpu_sort_particles(&particles, camera, view_dir);

        assert!(verify_back_to_front(&particles, &sorted, camera, view_dir));
    }

    #[test]
    fn test_verify_back_to_front_invalid() {
        let camera = [0.0, 0.0, 0.0];
        let view_dir = [0.0, 0.0, 1.0];

        let particles = vec![
            make_particle([0.0, 0.0, 10.0], true),
            make_particle([0.0, 0.0, 100.0], true),
            make_particle([0.0, 0.0, 50.0], true),
        ];

        // Wrong order: near, middle, far (front-to-back)
        let wrong_order = vec![0u32, 2, 1];

        assert!(!verify_back_to_front(&particles, &wrong_order, camera, view_dir));
    }

    // ── Large Scale ────────────────────────────────────────────────────────

    #[test]
    fn test_cpu_sort_large() {
        let camera = [0.0, 0.0, 0.0];
        let view_dir = [0.0, 0.0, 1.0];

        // 10K particles at random depths
        let particles: Vec<Particle> = (0..10_000)
            .map(|i| {
                let depth = ((i * 7919 + 104729) % 10_000) as f32;
                make_particle([0.0, 0.0, depth], true)
            })
            .collect();

        let sorted = cpu_sort_particles(&particles, camera, view_dir);

        assert_eq!(sorted.len(), particles.len());
        assert!(verify_back_to_front(&particles, &sorted, camera, view_dir));
    }

    #[test]
    fn test_cpu_radix_sort_large() {
        let camera = [0.0, 0.0, 0.0];
        let view_dir = [0.0, 0.0, 1.0];

        let particles: Vec<Particle> = (0..10_000)
            .map(|i| {
                let depth = ((i * 7919 + 104729) % 10_000) as f32;
                make_particle([0.0, 0.0, depth], true)
            })
            .collect();

        let (_, sorted) = cpu_radix_sort_particles(&particles, camera, view_dir);

        assert!(verify_back_to_front(&particles, &sorted, camera, view_dir));
    }

    // ── Different Camera Directions ────────────────────────────────────────

    #[test]
    fn test_sort_negative_z_view() {
        let camera = [0.0, 0.0, 100.0];
        let view_dir = [0.0, 0.0, -1.0]; // Looking -Z

        let particles = vec![
            make_particle([0.0, 0.0, 90.0], true),  // Near (close to camera)
            make_particle([0.0, 0.0, 10.0], true),  // Far
            make_particle([0.0, 0.0, 50.0], true),  // Middle
        ];

        let sorted = cpu_sort_particles(&particles, camera, view_dir);

        // With -Z view from Z=100:
        // particle 1 at Z=10 is far (depth 90 along -Z)
        // particle 2 at Z=50 is middle (depth 50 along -Z)
        // particle 0 at Z=90 is near (depth 10 along -Z)
        assert_eq!(sorted, vec![1, 2, 0]);
    }

    #[test]
    fn test_sort_x_axis_view() {
        let camera = [0.0, 0.0, 0.0];
        let view_dir = [1.0, 0.0, 0.0]; // Looking +X

        let particles = vec![
            make_particle([10.0, 0.0, 0.0], true),  // Near
            make_particle([100.0, 0.0, 0.0], true), // Far
            make_particle([50.0, 0.0, 0.0], true),  // Middle
        ];

        let sorted = cpu_sort_particles(&particles, camera, view_dir);

        // Far (1), middle (2), near (0)
        assert_eq!(sorted, vec![1, 2, 0]);
    }
}
