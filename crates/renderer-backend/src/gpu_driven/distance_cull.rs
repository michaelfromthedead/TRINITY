//! GPU Distance and LOD Culling for TRINITY Engine (T-GPU-3.2).
//!
//! This module provides GPU-based distance culling and LOD selection using
//! compute shaders. It determines which instances are within the max draw
//! distance and selects the appropriate LOD level based on camera distance.
//!
//! # Overview
//!
//! Distance culling eliminates objects beyond a maximum draw distance,
//! reducing both GPU work and visual clutter at long ranges. LOD selection
//! chooses the appropriate mesh quality based on how far the object is
//! from the camera.
//!
//! # LOD Bias
//!
//! The LOD bias parameter allows global adjustment of quality:
//! - Negative bias (-1 to -2): Higher quality, use for high-end systems
//! - Zero bias (0): Default, balanced quality
//! - Positive bias (+1 to +2): Lower quality, use for performance mode
//!
//! # Performance
//!
//! - Work complexity: O(n), one thread per instance
//! - Target: < 0.05ms for 100K instances
//! - Memory: 64 bytes per instance LOD data
//!
//! # Usage
//!
//! ```ignore
//! // Create pipeline and resources
//! let pipeline = DistanceCullPipeline::new(&device);
//! let resources = DistanceCullResources::new(&device, 100_000);
//!
//! // Each frame: update camera and cull
//! let params = DistanceCullParams::new(
//!     instance_count,
//!     camera_position,
//!     1000.0,  // max_draw_distance
//!     0.0,     // lod_bias
//! );
//! resources.upload_params(&queue, &params);
//! resources.upload_instances(&queue, &instance_lods);
//! pipeline.dispatch(&mut encoder, &resources, instance_count);
//!
//! // Read LOD results
//! let lod_results = resources.read_results(&device, &queue);
//! ```

use std::mem;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Compute shader workgroup size (must match WGSL constant).
pub const WORKGROUP_SIZE: u32 = 256;

/// Maximum number of LOD levels per instance.
pub const MAX_LODS: usize = 8;

/// Marker value indicating instance is culled (distance beyond max).
pub const CULLED_LOD: u32 = 0xFFFFFFFF;

// ---------------------------------------------------------------------------
// DistanceCullParams
// ---------------------------------------------------------------------------

/// GPU uniform buffer for distance culling parameters.
///
/// # Memory Layout
///
/// 48 bytes, std140 compatible (vec3 requires 16-byte alignment):
/// | Offset | Field             | Size |
/// |--------|-------------------|------|
/// | 0      | num_instances     | 4    |
/// | 4      | _pad0             | 4    |
/// | 8      | _pad1             | 4    |
/// | 12     | _pad2             | 4    |
/// | 16     | camera_position   | 12   |
/// | 28     | max_draw_distance | 4    |
/// | 32     | lod_bias          | 4    |
/// | 36     | _pad3             | 4    |
/// | 40     | _pad4             | 4    |
/// | 44     | _pad5             | 4    |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DistanceCullParams {
    /// Number of instances to process.
    pub num_instances: u32,
    /// Padding for vec4 alignment.
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    /// Camera position in world space (xyz).
    pub camera_position: [f32; 3],
    /// Maximum draw distance. Objects beyond this are culled.
    pub max_draw_distance: f32,
    /// Global LOD bias: negative = higher quality, positive = lower quality.
    pub lod_bias: f32,
    /// Padding for 16-byte alignment.
    pub _pad3: f32,
    pub _pad4: f32,
    pub _pad5: f32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<DistanceCullParams>() == 48);

impl DistanceCullParams {
    /// Create parameters for distance culling.
    ///
    /// # Arguments
    ///
    /// * `num_instances` - Number of instances to process.
    /// * `camera_position` - Camera world position.
    /// * `max_draw_distance` - Maximum render distance.
    /// * `lod_bias` - LOD quality bias (-2 to +2 typical).
    pub fn new(
        num_instances: u32,
        camera_position: [f32; 3],
        max_draw_distance: f32,
        lod_bias: f32,
    ) -> Self {
        Self {
            num_instances,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            camera_position,
            max_draw_distance,
            lod_bias,
            _pad3: 0.0,
            _pad4: 0.0,
            _pad5: 0.0,
        }
    }

    /// Get the number of workgroups needed for dispatch.
    #[inline]
    pub fn num_workgroups(&self) -> u32 {
        (self.num_instances + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
    }
}

// ---------------------------------------------------------------------------
// InstanceLOD
// ---------------------------------------------------------------------------

/// Per-instance LOD configuration.
///
/// Contains the instance position, bounding radius, and LOD distance thresholds.
///
/// # Memory Layout
///
/// 64 bytes, vec4 aligned:
/// | Offset | Field         | Size |
/// |--------|---------------|------|
/// | 0      | position      | 12   |
/// | 12     | radius        | 4    |
/// | 16     | lod_distances | 32   |
/// | 48     | num_lods      | 4    |
/// | 52     | _pad0         | 4    |
/// | 56     | _pad1         | 4    |
/// | 60     | _pad2         | 4    |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceLOD {
    /// Instance position in world space (typically bounding sphere center).
    pub position: [f32; 3],
    /// Bounding radius for screen size calculation.
    pub radius: f32,
    /// LOD transition distances: lod_distances[i] = max distance for LOD i.
    pub lod_distances: [f32; 8],
    /// Number of LOD levels (1-8).
    pub num_lods: u32,
    /// Padding for 16-byte alignment.
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<InstanceLOD>() == 64);

impl InstanceLOD {
    /// Create instance LOD configuration.
    ///
    /// # Arguments
    ///
    /// * `position` - Instance world position.
    /// * `radius` - Bounding sphere radius.
    /// * `lod_distances` - Distance thresholds for each LOD level.
    ///
    /// # Panics
    ///
    /// Panics if `lod_distances` is empty or has more than 8 elements.
    pub fn new(position: [f32; 3], radius: f32, lod_distances: &[f32]) -> Self {
        assert!(!lod_distances.is_empty(), "At least one LOD level required");
        assert!(lod_distances.len() <= MAX_LODS, "Maximum {} LOD levels", MAX_LODS);

        let mut distances = [f32::MAX; 8];
        for (i, &d) in lod_distances.iter().enumerate() {
            distances[i] = d;
        }

        Self {
            position,
            radius,
            lod_distances: distances,
            num_lods: lod_distances.len() as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }

    /// Create instance with a single LOD (no LOD transitions).
    pub fn single_lod(position: [f32; 3], radius: f32) -> Self {
        Self::new(position, radius, &[f32::MAX])
    }

    /// Create instance with standard LOD distances.
    ///
    /// Uses sensible defaults: 25, 50, 100, 200 units.
    pub fn standard_lods(position: [f32; 3], radius: f32) -> Self {
        Self::new(position, radius, &[25.0, 50.0, 100.0, 200.0])
    }

    /// Create instance with scaled LOD distances.
    ///
    /// Scales the standard distances by the given factor.
    pub fn scaled_lods(position: [f32; 3], radius: f32, scale: f32) -> Self {
        Self::new(
            position,
            radius,
            &[25.0 * scale, 50.0 * scale, 100.0 * scale, 200.0 * scale],
        )
    }
}

// ---------------------------------------------------------------------------
// LODResult
// ---------------------------------------------------------------------------

/// Output LOD result for each instance.
///
/// # Memory Layout
///
/// 8 bytes:
/// | Offset | Field       | Size |
/// |--------|-------------|------|
/// | 0      | lod_level   | 4    |
/// | 4      | screen_size | 4    |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LODResult {
    /// Selected LOD level (0 = highest quality).
    /// Value of `CULLED_LOD` (0xFFFFFFFF) indicates instance was culled.
    pub lod_level: u32,
    /// Approximate projected screen size (0-1 normalized).
    /// Used for streaming priority decisions.
    pub screen_size: f32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<LODResult>() == 8);

impl LODResult {
    /// Check if this instance was culled (beyond max draw distance).
    #[inline]
    pub fn is_culled(&self) -> bool {
        self.lod_level == CULLED_LOD
    }

    /// Check if this instance is visible.
    #[inline]
    pub fn is_visible(&self) -> bool {
        self.lod_level != CULLED_LOD
    }

    /// Get the LOD level, or `None` if culled.
    #[inline]
    pub fn lod(&self) -> Option<u32> {
        if self.is_culled() {
            None
        } else {
            Some(self.lod_level)
        }
    }
}

// ---------------------------------------------------------------------------
// DistanceCullResources
// ---------------------------------------------------------------------------

/// GPU resources for distance culling.
///
/// Contains all buffers needed for the distance cull compute shader.
pub struct DistanceCullResources {
    /// Uniform buffer for culling parameters.
    pub params_buffer: wgpu::Buffer,
    /// Storage buffer for instance LOD data (input).
    pub instances_buffer: wgpu::Buffer,
    /// Storage buffer for LOD results (output).
    pub results_buffer: wgpu::Buffer,
    /// Staging buffer for reading results back to CPU.
    pub results_staging: wgpu::Buffer,
    /// Maximum number of instances supported.
    pub capacity: u32,
}

impl DistanceCullResources {
    /// Create distance culling resources for the given capacity.
    pub fn new(device: &wgpu::Device, capacity: u32) -> Self {
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("distance_cull_params"),
            size: mem::size_of::<DistanceCullParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let instances_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("distance_cull_instances"),
            size: (capacity as u64) * (mem::size_of::<InstanceLOD>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let results_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("distance_cull_results"),
            size: (capacity as u64) * (mem::size_of::<LODResult>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let results_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("distance_cull_results_staging"),
            size: (capacity as u64) * (mem::size_of::<LODResult>() as u64),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            params_buffer,
            instances_buffer,
            results_buffer,
            results_staging,
            capacity,
        }
    }

    /// Upload culling parameters to GPU.
    pub fn upload_params(&self, queue: &wgpu::Queue, params: &DistanceCullParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Upload instance LOD data to GPU.
    ///
    /// # Panics
    ///
    /// Panics if `instances.len() > self.capacity`.
    pub fn upload_instances(&self, queue: &wgpu::Queue, instances: &[InstanceLOD]) {
        assert!(instances.len() <= self.capacity as usize);
        queue.write_buffer(&self.instances_buffer, 0, bytemuck::cast_slice(instances));
    }
}

// ---------------------------------------------------------------------------
// DistanceCullPipeline
// ---------------------------------------------------------------------------

/// GPU compute pipeline for distance culling and LOD selection.
pub struct DistanceCullPipeline {
    /// Main pipeline: distance cull + LOD selection.
    pub pipeline: wgpu::ComputePipeline,
    /// Distance-only pipeline (no LOD selection).
    pub pipeline_distance_only: wgpu::ComputePipeline,
    /// LOD selection only (no distance culling).
    pub pipeline_lod_only: wgpu::ComputePipeline,
    /// Bind group layout for culling resources.
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl DistanceCullPipeline {
    /// Create the distance culling pipeline.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `shader_source` - WGSL shader source code.
    pub fn new(device: &wgpu::Device, shader_source: &str) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("distance_cull_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("distance_cull_bind_group_layout"),
            entries: &[
                // @binding(0) params: DistanceCullParams
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            std::num::NonZeroU64::new(mem::size_of::<DistanceCullParams>() as u64)
                                .unwrap(),
                        ),
                    },
                    count: None,
                },
                // @binding(1) instances: array<InstanceLOD>
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
                // @binding(2) results: array<LODResult>
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
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("distance_cull_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("distance_cull_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "cull_distance",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let pipeline_distance_only =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("distance_cull_pipeline_distance_only"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: "cull_distance_only",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let pipeline_lod_only = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("distance_cull_pipeline_lod_only"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "lod_select_only",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            pipeline_distance_only,
            pipeline_lod_only,
            bind_group_layout,
        }
    }

    /// Create a bind group for the given resources.
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        resources: &DistanceCullResources,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("distance_cull_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.instances_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.results_buffer.as_entire_binding(),
                },
            ],
        })
    }
}

// ---------------------------------------------------------------------------
// CPU Reference Implementation
// ---------------------------------------------------------------------------

/// CPU reference implementation of distance culling and LOD selection.
///
/// Used for testing and fallback when GPU is not available.
pub fn cpu_distance_cull(
    camera_position: [f32; 3],
    max_draw_distance: f32,
    lod_bias: f32,
    instances: &[InstanceLOD],
) -> Vec<LODResult> {
    instances
        .iter()
        .map(|instance| {
            // Compute distance from camera to instance
            let dx = instance.position[0] - camera_position[0];
            let dy = instance.position[1] - camera_position[1];
            let dz = instance.position[2] - camera_position[2];
            let distance = (dx * dx + dy * dy + dz * dz).sqrt();

            // Distance cull check
            if distance > max_draw_distance {
                return LODResult {
                    lod_level: CULLED_LOD,
                    screen_size: 0.0,
                };
            }

            // LOD selection with bias
            let biased_distance = distance * 2.0_f32.powf(lod_bias);
            let mut lod = instance.num_lods - 1;
            for i in 0..instance.num_lods as usize {
                if biased_distance < instance.lod_distances[i] {
                    lod = i as u32;
                    break;
                }
            }

            // Screen size calculation
            let screen_size = if distance <= 0.001 {
                1.0
            } else {
                (instance.radius / distance).clamp(0.0, 1.0)
            };

            LODResult {
                lod_level: lod,
                screen_size,
            }
        })
        .collect()
}

/// CPU reference implementation for distance-only culling (no LOD selection).
pub fn cpu_distance_cull_only(
    camera_position: [f32; 3],
    max_draw_distance: f32,
    instances: &[InstanceLOD],
) -> Vec<LODResult> {
    instances
        .iter()
        .map(|instance| {
            let dx = instance.position[0] - camera_position[0];
            let dy = instance.position[1] - camera_position[1];
            let dz = instance.position[2] - camera_position[2];
            let distance = (dx * dx + dy * dy + dz * dz).sqrt();

            if distance > max_draw_distance {
                LODResult {
                    lod_level: CULLED_LOD,
                    screen_size: 0.0,
                }
            } else {
                let screen_size = if distance <= 0.001 {
                    1.0
                } else {
                    (instance.radius / distance).clamp(0.0, 1.0)
                };
                LODResult {
                    lod_level: 0,
                    screen_size,
                }
            }
        })
        .collect()
}

/// CPU reference implementation for LOD selection only (no distance culling).
pub fn cpu_lod_select_only(
    camera_position: [f32; 3],
    lod_bias: f32,
    instances: &[InstanceLOD],
) -> Vec<LODResult> {
    instances
        .iter()
        .map(|instance| {
            let dx = instance.position[0] - camera_position[0];
            let dy = instance.position[1] - camera_position[1];
            let dz = instance.position[2] - camera_position[2];
            let distance = (dx * dx + dy * dy + dz * dz).sqrt();

            // LOD selection with bias
            let biased_distance = distance * 2.0_f32.powf(lod_bias);
            let mut lod = instance.num_lods - 1;
            for i in 0..instance.num_lods as usize {
                if biased_distance < instance.lod_distances[i] {
                    lod = i as u32;
                    break;
                }
            }

            let screen_size = if distance <= 0.001 {
                1.0
            } else {
                (instance.radius / distance).clamp(0.0, 1.0)
            };

            LODResult {
                lod_level: lod,
                screen_size,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: Create a simple test instance at given position.
    fn make_test_instance(position: [f32; 3]) -> InstanceLOD {
        InstanceLOD::new(position, 1.0, &[10.0, 25.0, 50.0, 100.0])
    }

    #[test]
    fn test_within_max_distance_visible() {
        let camera_pos = [0.0, 0.0, 0.0];
        let max_dist = 100.0;
        let instances = vec![
            make_test_instance([0.0, 0.0, 50.0]),  // 50 units away
            make_test_instance([30.0, 40.0, 0.0]), // 50 units away (3-4-5 triangle)
            make_test_instance([0.0, 0.0, 99.0]),  // Just within max
        ];

        let results = cpu_distance_cull(camera_pos, max_dist, 0.0, &instances);

        assert!(results[0].is_visible(), "Instance at 50 units should be visible");
        assert!(results[1].is_visible(), "Instance at 50 units should be visible");
        assert!(results[2].is_visible(), "Instance at 99 units should be visible");
    }

    #[test]
    fn test_beyond_max_distance_culled() {
        let camera_pos = [0.0, 0.0, 0.0];
        let max_dist = 100.0;
        let instances = vec![
            make_test_instance([0.0, 0.0, 101.0]),  // Just beyond max
            make_test_instance([0.0, 0.0, 200.0]),  // Way beyond max
            make_test_instance([100.0, 100.0, 0.0]), // ~141 units, beyond max
        ];

        let results = cpu_distance_cull(camera_pos, max_dist, 0.0, &instances);

        assert!(results[0].is_culled(), "Instance at 101 units should be culled");
        assert!(results[1].is_culled(), "Instance at 200 units should be culled");
        assert!(results[2].is_culled(), "Instance at ~141 units should be culled");
    }

    #[test]
    fn test_lod_selection_by_distance() {
        let camera_pos = [0.0, 0.0, 0.0];
        let max_dist = 1000.0;

        // LOD distances: 10, 25, 50, 100
        let instances = vec![
            make_test_instance([0.0, 0.0, 5.0]),   // < 10: LOD 0
            make_test_instance([0.0, 0.0, 15.0]),  // 10-25: LOD 1
            make_test_instance([0.0, 0.0, 35.0]),  // 25-50: LOD 2
            make_test_instance([0.0, 0.0, 75.0]),  // 50-100: LOD 3
            make_test_instance([0.0, 0.0, 150.0]), // > 100: LOD 3 (last)
        ];

        let results = cpu_distance_cull(camera_pos, max_dist, 0.0, &instances);

        assert_eq!(results[0].lod(), Some(0), "Distance 5: should be LOD 0");
        assert_eq!(results[1].lod(), Some(1), "Distance 15: should be LOD 1");
        assert_eq!(results[2].lod(), Some(2), "Distance 35: should be LOD 2");
        assert_eq!(results[3].lod(), Some(3), "Distance 75: should be LOD 3");
        assert_eq!(results[4].lod(), Some(3), "Distance 150: should be LOD 3 (last level)");
    }

    #[test]
    fn test_lod_bias_positive_lowers_quality() {
        let camera_pos = [0.0, 0.0, 0.0];
        let max_dist = 1000.0;

        // Instance at distance 15: normally LOD 1 (10-25 range)
        let instances = vec![make_test_instance([0.0, 0.0, 15.0])];

        // No bias: LOD 1
        let results_no_bias = cpu_distance_cull(camera_pos, max_dist, 0.0, &instances);
        assert_eq!(results_no_bias[0].lod(), Some(1));

        // Positive bias (+1): doubles effective distance (15 -> 30), now LOD 2
        let results_pos_bias = cpu_distance_cull(camera_pos, max_dist, 1.0, &instances);
        assert_eq!(results_pos_bias[0].lod(), Some(2), "Positive bias should lower quality (higher LOD)");
    }

    #[test]
    fn test_lod_bias_negative_raises_quality() {
        let camera_pos = [0.0, 0.0, 0.0];
        let max_dist = 1000.0;

        // Instance at distance 35: normally LOD 2 (25-50 range)
        let instances = vec![make_test_instance([0.0, 0.0, 35.0])];

        // No bias: LOD 2
        let results_no_bias = cpu_distance_cull(camera_pos, max_dist, 0.0, &instances);
        assert_eq!(results_no_bias[0].lod(), Some(2));

        // Negative bias (-1): halves effective distance (35 -> 17.5), now LOD 1
        let results_neg_bias = cpu_distance_cull(camera_pos, max_dist, -1.0, &instances);
        assert_eq!(results_neg_bias[0].lod(), Some(1), "Negative bias should raise quality (lower LOD)");
    }

    #[test]
    fn test_screen_size_calculation() {
        let camera_pos = [0.0, 0.0, 0.0];
        let max_dist = 1000.0;

        // Radius 1.0 at various distances
        let instances = vec![
            InstanceLOD::single_lod([0.0, 0.0, 1.0], 1.0),   // dist=1, size=1.0
            InstanceLOD::single_lod([0.0, 0.0, 10.0], 1.0),  // dist=10, size=0.1
            InstanceLOD::single_lod([0.0, 0.0, 100.0], 1.0), // dist=100, size=0.01
        ];

        let results = cpu_distance_cull(camera_pos, max_dist, 0.0, &instances);

        assert!((results[0].screen_size - 1.0).abs() < 0.01);
        assert!((results[1].screen_size - 0.1).abs() < 0.01);
        assert!((results[2].screen_size - 0.01).abs() < 0.001);
    }

    #[test]
    fn test_zero_distance_returns_lod0() {
        let camera_pos = [10.0, 20.0, 30.0];
        let max_dist = 1000.0;

        // Instance at exact camera position
        let instances = vec![make_test_instance([10.0, 20.0, 30.0])];

        let results = cpu_distance_cull(camera_pos, max_dist, 0.0, &instances);

        assert_eq!(results[0].lod(), Some(0), "Zero distance should return LOD 0");
        assert_eq!(results[0].screen_size, 1.0, "Zero distance should return screen size 1.0");
    }

    #[test]
    fn test_params_struct_size() {
        assert_eq!(
            mem::size_of::<DistanceCullParams>(),
            48,
            "DistanceCullParams must be 48 bytes for GPU alignment"
        );
    }

    #[test]
    fn test_instance_lod_struct_size() {
        assert_eq!(
            mem::size_of::<InstanceLOD>(),
            64,
            "InstanceLOD must be 64 bytes for GPU alignment"
        );
    }

    #[test]
    fn test_lod_result_struct_size() {
        assert_eq!(
            mem::size_of::<LODResult>(),
            8,
            "LODResult must be 8 bytes"
        );
    }

    #[test]
    fn test_instance_lod_constructors() {
        // Single LOD
        let single = InstanceLOD::single_lod([0.0, 0.0, 0.0], 1.0);
        assert_eq!(single.num_lods, 1);

        // Standard LODs
        let standard = InstanceLOD::standard_lods([0.0, 0.0, 0.0], 1.0);
        assert_eq!(standard.num_lods, 4);
        assert_eq!(standard.lod_distances[0], 25.0);

        // Scaled LODs
        let scaled = InstanceLOD::scaled_lods([0.0, 0.0, 0.0], 1.0, 2.0);
        assert_eq!(scaled.num_lods, 4);
        assert_eq!(scaled.lod_distances[0], 50.0);
    }

    #[test]
    fn test_lod_result_helpers() {
        let visible = LODResult {
            lod_level: 2,
            screen_size: 0.5,
        };
        assert!(visible.is_visible());
        assert!(!visible.is_culled());
        assert_eq!(visible.lod(), Some(2));

        let culled = LODResult {
            lod_level: CULLED_LOD,
            screen_size: 0.0,
        };
        assert!(!culled.is_visible());
        assert!(culled.is_culled());
        assert_eq!(culled.lod(), None);
    }

    #[test]
    fn test_distance_only_mode() {
        let camera_pos = [0.0, 0.0, 0.0];
        let max_dist = 100.0;
        let instances = vec![
            make_test_instance([0.0, 0.0, 50.0]),  // Visible
            make_test_instance([0.0, 0.0, 150.0]), // Culled
        ];

        let results = cpu_distance_cull_only(camera_pos, max_dist, &instances);

        assert!(results[0].is_visible());
        assert_eq!(results[0].lod(), Some(0), "Distance-only mode should always use LOD 0");
        assert!(results[1].is_culled());
    }

    #[test]
    fn test_lod_only_mode() {
        let camera_pos = [0.0, 0.0, 0.0];
        let instances = vec![
            make_test_instance([0.0, 0.0, 15.0]),  // LOD 1
            make_test_instance([0.0, 0.0, 9999.0]), // LOD 3 (not culled!)
        ];

        let results = cpu_lod_select_only(camera_pos, 0.0, &instances);

        // LOD-only mode never culls
        assert!(results[0].is_visible());
        assert!(results[1].is_visible(), "LOD-only mode should never cull");
        assert_eq!(results[1].lod(), Some(3));
    }

    #[test]
    fn test_num_workgroups() {
        let params1 = DistanceCullParams::new(1, [0.0; 3], 100.0, 0.0);
        assert_eq!(params1.num_workgroups(), 1);

        let params256 = DistanceCullParams::new(256, [0.0; 3], 100.0, 0.0);
        assert_eq!(params256.num_workgroups(), 1);

        let params257 = DistanceCullParams::new(257, [0.0; 3], 100.0, 0.0);
        assert_eq!(params257.num_workgroups(), 2);

        let params1000 = DistanceCullParams::new(1000, [0.0; 3], 100.0, 0.0);
        assert_eq!(params1000.num_workgroups(), 4);
    }

    #[test]
    fn test_multiple_instances_mixed_visibility() {
        let camera_pos = [0.0, 0.0, 0.0];
        let max_dist = 100.0;

        let instances = vec![
            make_test_instance([0.0, 0.0, 5.0]),    // Visible, LOD 0
            make_test_instance([0.0, 0.0, 150.0]),  // Culled
            make_test_instance([0.0, 0.0, 35.0]),   // Visible, LOD 2
            make_test_instance([200.0, 0.0, 0.0]),  // Culled
            make_test_instance([0.0, 0.0, 75.0]),   // Visible, LOD 3
        ];

        let results = cpu_distance_cull(camera_pos, max_dist, 0.0, &instances);

        assert_eq!(results[0].lod(), Some(0));
        assert!(results[1].is_culled());
        assert_eq!(results[2].lod(), Some(2));
        assert!(results[3].is_culled());
        assert_eq!(results[4].lod(), Some(3));
    }
}
