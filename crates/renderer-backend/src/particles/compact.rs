//! GPU Particle Compaction for TRINITY Engine (T-GPU-5.3).
//!
//! This module provides GPU-based stream compaction for particle buffers using
//! parallel prefix sum (Blelloch algorithm). It efficiently removes dead particles
//! while preserving the relative order of alive particles (stable compaction).
//!
//! # Overview
//!
//! Particle compaction runs after the particle update pass and before rendering.
//! It takes an input particle buffer and an array of alive flags, producing a
//! compacted output buffer containing only alive particles.
//!
//! The algorithm runs in three phases:
//! 1. **Prefix Sum**: Compute per-block prefix sums of alive flags
//! 2. **Block Scan**: Scan block totals to get global offsets
//! 3. **Scatter**: Write alive particles to their compacted positions
//!
//! # Performance
//!
//! - Work complexity: O(n)
//! - Step complexity: O(log n) per phase
//! - Target: < 0.3ms for 100K particles
//! - Memory: Double-buffered particle storage
//!
//! # Usage
//!
//! ```ignore
//! // Create compaction pipeline
//! let pipeline = ParticleCompactPipeline::new(&device);
//!
//! // Create resources for 100K particles
//! let resources = ParticleCompactResources::new(&device, 100_000);
//!
//! // Each frame: update params and dispatch
//! resources.upload_params(&queue, &ParticleCompactParams::new(active_count, max_particles));
//! pipeline.dispatch(&mut encoder, &resources);
//!
//! // Read compacted count for rendering
//! let alive_count = resources.read_alive_count(&device, &queue);
//! ```

use std::mem;

use super::spawn::{Particle, WORKGROUP_SIZE};

// Re-export PARTICLE_FLAG_ALIVE for convenience
pub use super::spawn::PARTICLE_FLAG_ALIVE;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum particles for single-pass compaction (one workgroup).
pub const SINGLE_PASS_MAX: u32 = 256;

/// Maximum blocks supported by simple block-sum scan.
pub const MAX_BLOCKS_SIMPLE: u32 = 256;

/// Maximum particles with simple block-sum scan.
pub const MAX_PARTICLES_SIMPLE: u32 = MAX_BLOCKS_SIMPLE * WORKGROUP_SIZE; // 65536

// ---------------------------------------------------------------------------
// ParticleCompactParams
// ---------------------------------------------------------------------------

/// GPU uniform buffer for particle compaction parameters.
///
/// Matches the WGSL `ParticleCompactParams` struct layout.
///
/// # Memory Layout
///
/// 16 bytes total, std140/std430 compatible:
///
/// | Offset | Field         | Size    |
/// |--------|---------------|---------|
/// | 0      | num_particles | 4 bytes |
/// | 4      | max_particles | 4 bytes |
/// | 8      | num_blocks    | 4 bytes |
/// | 12     | _padding      | 4 bytes |
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ParticleCompactParams {
    /// Number of particles to process this frame.
    pub num_particles: u32,
    /// Maximum particles the buffer can hold.
    pub max_particles: u32,
    /// Number of workgroups (blocks) in dispatch.
    pub num_blocks: u32,
    /// Padding for 16-byte alignment.
    pub _padding: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<ParticleCompactParams>() == 16);

impl ParticleCompactParams {
    /// Create parameters for the given particle counts.
    pub fn new(num_particles: u32, max_particles: u32) -> Self {
        let num_blocks = (num_particles + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        Self {
            num_particles,
            max_particles,
            num_blocks,
            _padding: 0,
        }
    }

    /// Get the number of workgroups needed for dispatch.
    #[inline]
    pub fn num_blocks(&self) -> u32 {
        self.num_blocks
    }

    /// Check if this can use single-pass compaction.
    #[inline]
    pub fn is_single_pass(&self) -> bool {
        self.num_particles <= SINGLE_PASS_MAX
    }

    /// Check if this can use simple block-sum scan.
    #[inline]
    pub fn is_simple_scan(&self) -> bool {
        self.num_blocks <= MAX_BLOCKS_SIMPLE
    }
}

impl Default for ParticleCompactParams {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

// ---------------------------------------------------------------------------
// DrawIndirectArgs
// ---------------------------------------------------------------------------

/// Indirect draw arguments for DrawIndexedIndirect.
///
/// Used to automatically update instance_count after compaction,
/// enabling GPU-driven particle rendering without CPU readback.
///
/// # Memory Layout
///
/// 20 bytes total:
///
/// | Offset | Field          | Size    |
/// |--------|----------------|---------|
/// | 0      | index_count    | 4 bytes |
/// | 4      | instance_count | 4 bytes |
/// | 8      | first_index    | 4 bytes |
/// | 12     | base_vertex    | 4 bytes |
/// | 16     | first_instance | 4 bytes |
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawIndirectArgs {
    /// Number of indices per instance (6 for particle quad).
    pub index_count: u32,
    /// Number of instances (alive particle count after compaction).
    pub instance_count: u32,
    /// First index offset.
    pub first_index: u32,
    /// Base vertex offset.
    pub base_vertex: i32,
    /// First instance offset.
    pub first_instance: u32,
}

const _: () = assert!(mem::size_of::<DrawIndirectArgs>() == 20);

impl DrawIndirectArgs {
    /// Create default draw args for particle quads (6 indices per quad).
    pub fn particle_quad() -> Self {
        Self {
            index_count: 6, // 2 triangles
            instance_count: 0,
            first_index: 0,
            base_vertex: 0,
            first_instance: 0,
        }
    }
}

impl Default for DrawIndirectArgs {
    fn default() -> Self {
        Self::particle_quad()
    }
}

// ---------------------------------------------------------------------------
// ParticleCompactResources
// ---------------------------------------------------------------------------

/// GPU resources for particle compaction.
///
/// Contains all buffers needed for the compaction algorithm:
/// - `params_buffer`: Uniform buffer for compaction parameters
/// - `alive_flags`: Input alive flags (0 = dead, 1 = alive)
/// - `block_sums`: Intermediate block sum storage
/// - `particles_in`: Input particle buffer
/// - `particles_out`: Output compacted particle buffer
/// - `alive_count`: Final count of alive particles (atomic u32)
/// - `draw_indirect`: Indirect draw buffer for rendering
pub struct ParticleCompactResources {
    /// Uniform buffer for compaction parameters.
    pub params_buffer: wgpu::Buffer,
    /// Input alive flags buffer (0 or 1 per particle).
    pub alive_flags: wgpu::Buffer,
    /// Intermediate block sums buffer for multi-pass compaction.
    pub block_sums: wgpu::Buffer,
    /// Input particle buffer (read-only during compaction).
    pub particles_in: wgpu::Buffer,
    /// Output compacted particle buffer.
    pub particles_out: wgpu::Buffer,
    /// Alive count buffer (single atomic u32).
    pub alive_count: wgpu::Buffer,
    /// Indirect draw buffer for GPU-driven rendering.
    pub draw_indirect: wgpu::Buffer,
    /// Staging buffer for reading alive count back to CPU.
    pub alive_count_staging: wgpu::Buffer,
    /// Maximum particle capacity.
    pub capacity: u32,
}

impl ParticleCompactResources {
    /// Create compaction resources for the given capacity.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `capacity` - Maximum number of particles to support.
    pub fn new(device: &wgpu::Device, capacity: u32) -> Self {
        let num_blocks = (capacity + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        let particle_size = mem::size_of::<Particle>() as u64;

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle_compact_params"),
            size: mem::size_of::<ParticleCompactParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let alive_flags = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle_alive_flags"),
            size: (capacity as u64) * 4, // u32 per particle
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let block_sums = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle_block_sums"),
            size: (num_blocks.max(1) as u64) * 4, // u32 per block
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let particles_in = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particles_in"),
            size: (capacity as u64) * particle_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let particles_out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particles_out"),
            size: (capacity as u64) * particle_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });

        let alive_count = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle_alive_count"),
            size: 4, // single u32
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let draw_indirect = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle_draw_indirect"),
            size: mem::size_of::<DrawIndirectArgs>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let alive_count_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle_alive_count_staging"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            params_buffer,
            alive_flags,
            block_sums,
            particles_in,
            particles_out,
            alive_count,
            draw_indirect,
            alive_count_staging,
            capacity,
        }
    }

    /// Upload compaction parameters to the GPU.
    pub fn upload_params(&self, queue: &wgpu::Queue, params: &ParticleCompactParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Upload alive flags to the GPU.
    pub fn upload_alive_flags(&self, queue: &wgpu::Queue, flags: &[u32]) {
        queue.write_buffer(&self.alive_flags, 0, bytemuck::cast_slice(flags));
    }

    /// Upload input particles to the GPU.
    pub fn upload_particles(&self, queue: &wgpu::Queue, particles: &[Particle]) {
        queue.write_buffer(&self.particles_in, 0, bytemuck::cast_slice(particles));
    }

    /// Initialize the draw indirect buffer with default values.
    pub fn init_draw_indirect(&self, queue: &wgpu::Queue) {
        let args = DrawIndirectArgs::particle_quad();
        queue.write_buffer(&self.draw_indirect, 0, bytemuck::bytes_of(&args));
    }

    /// Read the alive count back to the CPU.
    ///
    /// This is a synchronous operation that waits for GPU completion.
    pub fn read_alive_count(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> u32 {
        // Copy from GPU buffer to staging buffer
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("read_alive_count"),
        });
        encoder.copy_buffer_to_buffer(&self.alive_count, 0, &self.alive_count_staging, 0, 4);
        queue.submit([encoder.finish()]);

        // Map staging buffer and read
        let buffer_slice = self.alive_count_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let count = *bytemuck::from_bytes::<u32>(&data);
        drop(data);
        self.alive_count_staging.unmap();

        count
    }

    /// Swap particle buffers for double-buffering.
    ///
    /// After compaction, the output buffer contains compacted particles.
    /// Call this to make the output the new input for the next frame.
    pub fn swap_buffers(&mut self) {
        std::mem::swap(&mut self.particles_in, &mut self.particles_out);
    }
}

// ---------------------------------------------------------------------------
// ParticleCompactPipeline
// ---------------------------------------------------------------------------

/// Compute pipeline for GPU particle compaction.
///
/// Contains the three compute pipelines for the compaction algorithm:
/// 1. `prefix_sum_pipeline`: Per-block prefix sum of alive flags
/// 2. `block_scan_pipeline`: Scan block sums for global offsets
/// 3. `scatter_pipeline`: Scatter alive particles to output
///
/// Also includes a single-pass pipeline for small particle counts.
pub struct ParticleCompactPipeline {
    /// Pipeline for Phase 1: per-block prefix sum.
    prefix_sum_pipeline: wgpu::ComputePipeline,
    /// Pipeline for Phase 2: block sum scan.
    block_scan_pipeline: wgpu::ComputePipeline,
    /// Pipeline for Phase 3: scatter alive particles.
    scatter_pipeline: wgpu::ComputePipeline,
    /// Pipeline for single-pass compaction (small counts).
    single_pass_pipeline: wgpu::ComputePipeline,
    /// Bind group layout for compaction resources.
    bind_group_layout: wgpu::BindGroupLayout,
}

impl ParticleCompactPipeline {
    /// Create a new particle compaction pipeline.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    pub fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = Self::create_bind_group_layout(device);
        let pipeline_layout = Self::create_pipeline_layout(device, &bind_group_layout);

        let shader_source = include_str!("../../shaders/particles/gpu_particle_compact.comp.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_particle_compact_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let prefix_sum_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("particle_prefix_sum_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "compute_prefix_sum",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let block_scan_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("particle_block_scan_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "scan_block_sums",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let scatter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("particle_scatter_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "compact_particles",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let single_pass_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("particle_compact_single_pass_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: "compact_particles_single_pass",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        Self {
            prefix_sum_pipeline,
            block_scan_pipeline,
            scatter_pipeline,
            single_pass_pipeline,
            bind_group_layout,
        }
    }

    /// Get the bind group layout.
    #[inline]
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Create a bind group for the given resources.
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        resources: &ParticleCompactResources,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("particle_compact_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.alive_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.block_sums.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources.particles_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: resources.particles_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: resources.alive_count.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: resources.draw_indirect.as_entire_binding(),
                },
            ],
        })
    }

    /// Dispatch the compaction algorithm.
    ///
    /// Automatically selects single-pass or multi-pass based on particle count.
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder.
    /// * `bind_group` - The bind group containing all resources.
    /// * `params` - The compaction parameters.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        params: &ParticleCompactParams,
    ) {
        if params.is_single_pass() {
            self.dispatch_single_pass(encoder, bind_group);
        } else {
            self.dispatch_multi_pass(encoder, bind_group, params);
        }
    }

    /// Dispatch single-pass compaction for small particle counts.
    fn dispatch_single_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("particle_compact_single_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.single_pass_pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// Dispatch multi-pass compaction for larger particle counts.
    fn dispatch_multi_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        params: &ParticleCompactParams,
    ) {
        let num_blocks = params.num_blocks();

        // Phase 1: Per-block prefix sum
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("particle_prefix_sum_pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.prefix_sum_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(num_blocks, 1, 1);
        }

        // Phase 2: Block sum scan
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("particle_block_scan_pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.block_scan_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1); // Single workgroup for simple scan
        }

        // Phase 3: Scatter alive particles
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("particle_scatter_pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.scatter_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(num_blocks, 1, 1);
        }
    }

    /// Create the bind group layout.
    fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("particle_compact_bind_group_layout"),
            entries: &[
                // binding 0: params (uniform)
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
                // binding 1: alive_flags (storage, read)
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
                // binding 2: block_sums (storage, read_write)
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
                // binding 3: particles_in (storage, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 4: particles_out (storage, read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 5: alive_count (storage, read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 6: draw_indirect (storage, read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create the pipeline layout.
    fn create_pipeline_layout(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::PipelineLayout {
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("particle_compact_pipeline_layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        })
    }
}

// ---------------------------------------------------------------------------
// CPU Reference Implementation
// ---------------------------------------------------------------------------

/// CPU reference implementation of particle compaction.
///
/// Used for testing GPU results against known-correct values.
/// Performs stable compaction (preserves relative order).
///
/// # Arguments
///
/// * `particles` - Input particle array.
/// * `alive_flags` - Array of alive flags (0 = dead, 1 = alive).
///
/// # Returns
///
/// Tuple of (compacted particles, alive count).
pub fn cpu_compact_particles(particles: &[Particle], alive_flags: &[u32]) -> (Vec<Particle>, u32) {
    let mut output = Vec::with_capacity(particles.len());
    for (i, &flag) in alive_flags.iter().enumerate() {
        if flag != 0 && i < particles.len() {
            output.push(particles[i]);
        }
    }
    let count = output.len() as u32;
    (output, count)
}

/// CPU reference implementation of prefix sum for alive flags.
///
/// Returns exclusive prefix sums.
pub fn cpu_prefix_sum(flags: &[u32]) -> Vec<u32> {
    let mut output = vec![0u32; flags.len()];
    let mut sum = 0u32;
    for (i, &flag) in flags.iter().enumerate() {
        output[i] = sum;
        sum = sum.wrapping_add(flag);
    }
    output
}

/// Extract alive flags from particle buffer.
///
/// Utility function that creates alive flags array from particle flags field.
pub fn extract_alive_flags(particles: &[Particle]) -> Vec<u32> {
    particles
        .iter()
        .map(|p| if p.is_alive() { 1 } else { 0 })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    /// Create a test particle at the given position with specified lifetime and alive state.
    fn make_test_particle(position: [f32; 3], lifetime: f32, alive: bool) -> Particle {
        Particle {
            position,
            age: 0.0,
            velocity: [0.0, 1.0, 0.0],
            lifetime,
            color: [1.0, 1.0, 1.0, 1.0],
            size: 1.0,
            rotation: 0.0,
            rotation_speed: 0.0,
            flags: if alive { PARTICLE_FLAG_ALIVE } else { 0 },
        }
    }

    /// Set the alive flag on a particle.
    fn set_particle_alive(particle: &mut Particle, alive: bool) {
        if alive {
            particle.flags |= PARTICLE_FLAG_ALIVE;
        } else {
            particle.flags &= !PARTICLE_FLAG_ALIVE;
        }
    }

    // -----------------------------------------------------------------------
    // Particle struct tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_particle_size() {
        assert_eq!(mem::size_of::<Particle>(), 64);
    }

    #[test]
    fn test_particle_alignment() {
        assert_eq!(mem::align_of::<Particle>(), 4);
    }

    #[test]
    fn test_particle_pod() {
        let particle = make_test_particle([1.0, 2.0, 3.0], 5.0, true);
        let bytes = bytemuck::bytes_of(&particle);
        assert_eq!(bytes.len(), 64);
    }

    #[test]
    fn test_particle_alive_flag() {
        let mut particle = Particle::dead();
        assert!(!particle.is_alive());

        set_particle_alive(&mut particle, true);
        assert!(particle.is_alive());
        assert_eq!(particle.flags & PARTICLE_FLAG_ALIVE, 1);

        set_particle_alive(&mut particle, false);
        assert!(!particle.is_alive());
        assert_eq!(particle.flags & PARTICLE_FLAG_ALIVE, 0);
    }

    #[test]
    fn test_particle_default() {
        let particle = Particle::default();
        assert!(!particle.is_alive());
        assert_eq!(particle.position, [0.0, 0.0, 0.0]);
        assert_eq!(particle.age, 0.0);
        assert_eq!(particle.lifetime, 0.0);
    }

    // -----------------------------------------------------------------------
    // ParticleCompactParams tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_compact_params_size() {
        assert_eq!(mem::size_of::<ParticleCompactParams>(), 16);
    }

    #[test]
    fn test_compact_params_pod() {
        let params = ParticleCompactParams::new(1000, 4096);
        let bytes = bytemuck::bytes_of(&params);
        assert_eq!(bytes.len(), 16);
    }

    #[test]
    fn test_compact_params_num_blocks() {
        // Exact multiple of workgroup size
        let params = ParticleCompactParams::new(256, 1024);
        assert_eq!(params.num_blocks(), 1);

        let params = ParticleCompactParams::new(512, 1024);
        assert_eq!(params.num_blocks(), 2);

        // Non-exact multiple
        let params = ParticleCompactParams::new(257, 1024);
        assert_eq!(params.num_blocks(), 2);

        let params = ParticleCompactParams::new(1, 1024);
        assert_eq!(params.num_blocks(), 1);

        let params = ParticleCompactParams::new(1000, 4096);
        assert_eq!(params.num_blocks(), 4); // ceil(1000/256) = 4
    }

    #[test]
    fn test_compact_params_single_pass() {
        assert!(ParticleCompactParams::new(1, 256).is_single_pass());
        assert!(ParticleCompactParams::new(256, 256).is_single_pass());
        assert!(!ParticleCompactParams::new(257, 512).is_single_pass());
    }

    #[test]
    fn test_compact_params_simple_scan() {
        assert!(ParticleCompactParams::new(256, 256).is_simple_scan());
        assert!(ParticleCompactParams::new(65536, 65536).is_simple_scan()); // 256 blocks
        assert!(!ParticleCompactParams::new(65537, 100000).is_simple_scan()); // 257 blocks
    }

    // -----------------------------------------------------------------------
    // DrawIndirectArgs tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_draw_indirect_size() {
        assert_eq!(mem::size_of::<DrawIndirectArgs>(), 20);
    }

    #[test]
    fn test_draw_indirect_particle_quad() {
        let args = DrawIndirectArgs::particle_quad();
        assert_eq!(args.index_count, 6);
        assert_eq!(args.instance_count, 0);
        assert_eq!(args.first_index, 0);
        assert_eq!(args.base_vertex, 0);
        assert_eq!(args.first_instance, 0);
    }

    // -----------------------------------------------------------------------
    // CPU reference implementation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cpu_compact_no_dead_particles() {
        let particles: Vec<Particle> = (0..10)
            .map(|i| make_test_particle([i as f32, 0.0, 0.0], 5.0, true))
            .collect();
        let flags = vec![1u32; 10];

        let (output, count) = cpu_compact_particles(&particles, &flags);

        assert_eq!(count, 10);
        assert_eq!(output.len(), 10);
        for (i, p) in output.iter().enumerate() {
            assert_eq!(p.position[0], i as f32);
        }
    }

    #[test]
    fn test_cpu_compact_all_dead_particles() {
        let particles: Vec<Particle> = (0..10)
            .map(|i| make_test_particle([i as f32, 0.0, 0.0], 5.0, false))
            .collect();
        let flags = vec![0u32; 10];

        let (output, count) = cpu_compact_particles(&particles, &flags);

        assert_eq!(count, 0);
        assert!(output.is_empty());
    }

    #[test]
    fn test_cpu_compact_mixed_alive_dead() {
        let particles: Vec<Particle> = (0..8)
            .map(|i| make_test_particle([i as f32, 0.0, 0.0], 5.0, i % 2 == 0))
            .collect();
        // Even indices are alive: 0, 2, 4, 6
        let flags = vec![1, 0, 1, 0, 1, 0, 1, 0];

        let (output, count) = cpu_compact_particles(&particles, &flags);

        assert_eq!(count, 4);
        assert_eq!(output.len(), 4);
        assert_eq!(output[0].position[0], 0.0);
        assert_eq!(output[1].position[0], 2.0);
        assert_eq!(output[2].position[0], 4.0);
        assert_eq!(output[3].position[0], 6.0);
    }

    #[test]
    fn test_cpu_compact_order_preserved() {
        // Verify stable compaction: relative order of alive particles is maintained
        let particles: Vec<Particle> = (0..10)
            .map(|i| make_test_particle([i as f32, 0.0, 0.0], 5.0, true))
            .collect();
        let flags = vec![0, 1, 1, 0, 0, 1, 0, 1, 1, 0]; // Alive: 1, 2, 5, 7, 8

        let (output, count) = cpu_compact_particles(&particles, &flags);

        assert_eq!(count, 5);
        assert_eq!(output[0].position[0], 1.0);
        assert_eq!(output[1].position[0], 2.0);
        assert_eq!(output[2].position[0], 5.0);
        assert_eq!(output[3].position[0], 7.0);
        assert_eq!(output[4].position[0], 8.0);
    }

    #[test]
    fn test_cpu_prefix_sum_basic() {
        let flags = [1, 0, 1, 1, 0, 1, 0, 0];
        let expected = [0, 1, 1, 2, 3, 3, 4, 4];
        let output = cpu_prefix_sum(&flags);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_cpu_prefix_sum_all_alive() {
        let flags = vec![1u32; 8];
        let expected: Vec<u32> = (0..8).collect();
        let output = cpu_prefix_sum(&flags);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_cpu_prefix_sum_all_dead() {
        let flags = vec![0u32; 8];
        let expected = vec![0u32; 8];
        let output = cpu_prefix_sum(&flags);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_extract_alive_flags() {
        let particles = vec![
            make_test_particle([0.0, 0.0, 0.0], 1.0, true),
            make_test_particle([1.0, 0.0, 0.0], 1.0, false),
            make_test_particle([2.0, 0.0, 0.0], 1.0, true),
            make_test_particle([3.0, 0.0, 0.0], 1.0, false),
        ];

        let flags = extract_alive_flags(&particles);
        assert_eq!(flags, vec![1, 0, 1, 0]);
    }

    // -----------------------------------------------------------------------
    // Large array tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cpu_compact_large_all_alive() {
        let size = 10_000;
        let particles: Vec<Particle> = (0..size)
            .map(|i| make_test_particle([i as f32, 0.0, 0.0], 5.0, true))
            .collect();
        let flags = vec![1u32; size];

        let (output, count) = cpu_compact_particles(&particles, &flags);

        assert_eq!(count, size as u32);
        assert_eq!(output.len(), size);
    }

    #[test]
    fn test_cpu_compact_large_half_alive() {
        let size = 10_000;
        let particles: Vec<Particle> = (0..size)
            .map(|i| make_test_particle([i as f32, 0.0, 0.0], 5.0, i % 2 == 0))
            .collect();
        let flags: Vec<u32> = (0..size).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect();

        let (output, count) = cpu_compact_particles(&particles, &flags);

        assert_eq!(count, (size / 2) as u32);
        assert_eq!(output.len(), size / 2);

        // Verify order preserved (even indices)
        for (i, p) in output.iter().enumerate() {
            assert_eq!(p.position[0], (i * 2) as f32);
        }
    }

    #[test]
    fn test_cpu_compact_100k_particles() {
        let size = 100_000usize;
        // Deterministic pattern: every 3rd particle is dead
        let particles: Vec<Particle> = (0..size)
            .map(|i| make_test_particle([i as f32, 0.0, 0.0], 5.0, i % 3 != 0))
            .collect();
        let flags: Vec<u32> = (0..size).map(|i| if i % 3 != 0 { 1 } else { 0 }).collect();

        let (output, count) = cpu_compact_particles(&particles, &flags);

        // 2/3 of particles should be alive
        let expected_alive = (size * 2 / 3) as u32;
        assert!((count as i32 - expected_alive as i32).abs() <= 1);

        // Verify all output particles are from non-divisible-by-3 indices
        let mut expected_idx = 1; // First alive particle
        for p in output.iter().take(10) {
            assert_eq!(p.position[0], expected_idx as f32);
            expected_idx += if expected_idx % 3 == 1 { 1 } else { 2 };
        }
    }

    // -----------------------------------------------------------------------
    // Shader validation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_compact_shader_parses() {
        let shader_source = include_str!("../../shaders/particles/gpu_particle_compact.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("particle compact shader should parse without errors");

        // Verify expected entry points exist
        let entry_names: Vec<_> = module.entry_points.iter().map(|ep| &ep.name).collect();

        assert!(
            entry_names.iter().any(|n| *n == "compute_prefix_sum"),
            "Should have compute_prefix_sum entry point"
        );
        assert!(
            entry_names.iter().any(|n| *n == "scan_block_sums"),
            "Should have scan_block_sums entry point"
        );
        assert!(
            entry_names.iter().any(|n| *n == "compact_particles"),
            "Should have compact_particles entry point"
        );
        assert!(
            entry_names.iter().any(|n| *n == "compact_particles_single_pass"),
            "Should have compact_particles_single_pass entry point"
        );
    }

    #[test]
    fn test_compact_shader_validates() {
        let shader_source = include_str!("../../shaders/particles/gpu_particle_compact.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("particle compact shader should parse without errors");

        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );

        validator
            .validate(&module)
            .expect("particle compact shader should validate without errors");
    }

    #[test]
    fn test_compact_shader_workgroup_size() {
        let shader_source = include_str!("../../shaders/particles/gpu_particle_compact.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("particle compact shader should parse without errors");

        // Verify all compute entry points have 256x1x1 workgroup size
        for ep in &module.entry_points {
            if ep.stage == naga::ShaderStage::Compute {
                assert_eq!(
                    ep.workgroup_size,
                    [256, 1, 1],
                    "Entry point {} should have workgroup size 256x1x1",
                    ep.name
                );
            }
        }
    }

    #[test]
    fn test_compact_shader_entry_points_are_compute() {
        let shader_source = include_str!("../../shaders/particles/gpu_particle_compact.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("particle compact shader should parse without errors");

        // Verify all entry points are compute shaders
        for ep in &module.entry_points {
            assert_eq!(
                ep.stage,
                naga::ShaderStage::Compute,
                "Entry point {} should be a compute shader",
                ep.name
            );
        }
    }

    // -----------------------------------------------------------------------
    // Integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_compact_params_matches_workgroup_size() {
        // Ensure Rust constant matches shader
        assert_eq!(WORKGROUP_SIZE, 256);
    }

    #[test]
    fn test_particle_struct_layout_matches_shader() {
        // Verify particle struct size matches WGSL (64 bytes)
        assert_eq!(mem::size_of::<Particle>(), 64);

        // Verify field offsets
        let particle = Particle::default();
        let base = &particle as *const _ as usize;

        let position_offset = &particle.position as *const _ as usize - base;
        let age_offset = &particle.age as *const _ as usize - base;
        let velocity_offset = &particle.velocity as *const _ as usize - base;
        let lifetime_offset = &particle.lifetime as *const _ as usize - base;
        let color_offset = &particle.color as *const _ as usize - base;
        let size_offset = &particle.size as *const _ as usize - base;
        let rotation_offset = &particle.rotation as *const _ as usize - base;
        let rotation_speed_offset = &particle.rotation_speed as *const _ as usize - base;
        let flags_offset = &particle.flags as *const _ as usize - base;

        assert_eq!(position_offset, 0);
        assert_eq!(age_offset, 12);
        assert_eq!(velocity_offset, 16);
        assert_eq!(lifetime_offset, 28);
        assert_eq!(color_offset, 32);
        assert_eq!(size_offset, 48);
        assert_eq!(rotation_offset, 52);
        assert_eq!(rotation_speed_offset, 56);
        assert_eq!(flags_offset, 60);
    }
}
