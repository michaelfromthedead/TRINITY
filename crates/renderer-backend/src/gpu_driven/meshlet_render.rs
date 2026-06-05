//! Meshlet Rendering Pipeline for GPU-driven rendering (T-GPU-4.5).
//!
//! This module implements the meshlet rendering system that renders visible
//! meshlets using indirect draws. It integrates with the meshlet culling system
//! to efficiently draw only visible geometry.
//!
//! # Overview
//!
//! The meshlet render pipeline supports multiple output modes:
//!
//! 1. **Visibility Pass**: Outputs packed instance_id + primitive_id for deferred
//!    material shading
//! 2. **G-Buffer Pass**: Full deferred rendering output (albedo, normal, material)
//! 3. **Shadow Pass**: Depth-only rendering for shadow maps
//!
//! # Data Structures
//!
//! - [`MeshletRenderParams`]: GPU uniform buffer for render configuration
//! - [`MeshletVertex`]: 32-byte vertex with position, normal, UV, tangent
//! - [`MeshletDrawCommand`]: Indirect draw argument generation helper
//! - [`MeshletRenderPipeline`]: Pipeline with multiple entry points
//!
//! # Performance
//!
//! - Target: <0.3ms for 100K visible meshlets
//! - Uses indirect drawing to minimize CPU-GPU synchronization
//! - Vertex data is optimized for cache efficiency (32-byte alignment)
//!
//! # Usage
//!
//! ```ignore
//! // Create pipeline
//! let pipeline = MeshletRenderPipeline::new(&device);
//!
//! // Set up indirect draw commands from culling output
//! let draw_commands = MeshletDrawCommand::generate_from_visibility(&visible_meshlets);
//!
//! // Dispatch visibility pass
//! pipeline.dispatch_visibility_pass(&mut encoder, &resources, &draw_commands);
//!
//! // Or dispatch G-buffer pass
//! pipeline.dispatch_gbuffer_pass(&mut encoder, &resources, &draw_commands);
//! ```

use std::mem;

use bytemuck::{Pod, Zeroable};
use wgpu::{Buffer, BufferUsages, Device, Queue};

use super::indirect_draw::{IndirectDrawIndexedArgs, INDIRECT_DRAW_INDEXED_ARGS_SIZE};
use super::meshlet::Meshlet;

// =============================================================================
// Constants
// =============================================================================

/// Size of MeshletRenderParams in bytes (64 bytes, aligned to 16).
pub const MESHLET_RENDER_PARAMS_SIZE: usize = 64;

/// Size of MeshletVertex in bytes (32 bytes).
pub const MESHLET_VERTEX_SIZE: usize = 32;

/// Size of MeshletInstance in bytes (144 bytes).
pub const MESHLET_INSTANCE_SIZE: usize = 144;

/// Maximum meshlets per indirect multi-draw (limited by GPU buffer size).
pub const MAX_MESHLETS_PER_DRAW: u32 = 65536;

/// Render mode flags.
pub const FLAG_VISIBILITY_BUFFER: u32 = 1;
pub const FLAG_ALPHA_TEST: u32 = 2;
pub const FLAG_DOUBLE_SIDED: u32 = 4;
pub const FLAG_SHADOW_PASS: u32 = 8;

/// Invalid visibility ID sentinel.
pub const INVALID_VISIBILITY_ID: u32 = 0xFFFF_FFFF;

// =============================================================================
// MeshletRenderParams
// =============================================================================

/// GPU uniform buffer for meshlet render parameters.
///
/// # Memory Layout
///
/// 64 bytes, std140 compatible:
/// | Offset | Field         | Size |
/// |--------|---------------|------|
/// | 0      | num_meshlets  | 4    |
/// | 4      | flags         | 4    |
/// | 8      | alpha_cutoff  | 4    |
/// | 12     | _pad0         | 4    |
/// | 16     | viewport_size | 8    |
/// | 24     | near_plane    | 4    |
/// | 28     | far_plane     | 4    |
/// | 32     | _pad1         | 32   |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct MeshletRenderParams {
    /// Number of meshlets to render.
    pub num_meshlets: u32,
    /// Render flags: visibility buffer mode, alpha test, etc.
    pub flags: u32,
    /// Alpha test threshold (when FLAG_ALPHA_TEST is set).
    pub alpha_cutoff: f32,
    /// Padding for alignment.
    pub _pad0: u32,
    /// Viewport dimensions (width, height).
    pub viewport_size: [f32; 2],
    /// Near plane distance.
    pub near_plane: f32,
    /// Far plane distance.
    pub far_plane: f32,
    /// Padding to 64 bytes.
    pub _pad1: [u32; 8],
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<MeshletRenderParams>() == MESHLET_RENDER_PARAMS_SIZE);

impl MeshletRenderParams {
    /// Create default render parameters.
    pub fn new(num_meshlets: u32, viewport_width: f32, viewport_height: f32) -> Self {
        Self {
            num_meshlets,
            flags: 0,
            alpha_cutoff: 0.5,
            _pad0: 0,
            viewport_size: [viewport_width, viewport_height],
            near_plane: 0.1,
            far_plane: 1000.0,
            _pad1: [0; 8],
        }
    }

    /// Create parameters for visibility buffer pass.
    pub fn visibility_pass(
        num_meshlets: u32,
        viewport_width: f32,
        viewport_height: f32,
    ) -> Self {
        let mut params = Self::new(num_meshlets, viewport_width, viewport_height);
        params.flags = FLAG_VISIBILITY_BUFFER;
        params
    }

    /// Create parameters for G-buffer pass.
    pub fn gbuffer_pass(
        num_meshlets: u32,
        viewport_width: f32,
        viewport_height: f32,
    ) -> Self {
        Self::new(num_meshlets, viewport_width, viewport_height)
    }

    /// Create parameters for shadow pass.
    pub fn shadow_pass(num_meshlets: u32) -> Self {
        let mut params = Self::new(num_meshlets, 0.0, 0.0);
        params.flags = FLAG_SHADOW_PASS;
        params
    }

    /// Enable alpha testing with the given cutoff.
    pub fn with_alpha_test(mut self, alpha_cutoff: f32) -> Self {
        self.flags |= FLAG_ALPHA_TEST;
        self.alpha_cutoff = alpha_cutoff;
        self
    }

    /// Enable double-sided rendering.
    pub fn with_double_sided(mut self) -> Self {
        self.flags |= FLAG_DOUBLE_SIDED;
        self
    }

    /// Check if visibility buffer mode is enabled.
    #[inline]
    pub fn is_visibility_mode(&self) -> bool {
        (self.flags & FLAG_VISIBILITY_BUFFER) != 0
    }

    /// Check if alpha testing is enabled.
    #[inline]
    pub fn is_alpha_test(&self) -> bool {
        (self.flags & FLAG_ALPHA_TEST) != 0
    }

    /// Check if shadow pass mode is enabled.
    #[inline]
    pub fn is_shadow_pass(&self) -> bool {
        (self.flags & FLAG_SHADOW_PASS) != 0
    }
}

// =============================================================================
// MeshletVertex
// =============================================================================

/// Vertex data for meshlet rendering (32 bytes).
///
/// Optimized for GPU cache efficiency with 32-byte alignment.
///
/// # Memory Layout
///
/// 32 bytes:
/// | Offset | Field         | Size |
/// |--------|---------------|------|
/// | 0      | position      | 12   |
/// | 12     | normal_packed | 4    |
/// | 16     | texcoord      | 8    |
/// | 24     | tangent_packed| 8    |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct MeshletVertex {
    /// Position in local space.
    pub position: [f32; 3],
    /// Packed normal using octahedral encoding (SNorm16x2 as f32 bits).
    pub normal_packed: f32,
    /// Texture coordinates (UV).
    pub texcoord: [f32; 2],
    /// Packed tangent direction and sign.
    /// x: octahedral-encoded tangent, y: sign bit in LSB.
    pub tangent_packed: [f32; 2],
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<MeshletVertex>() == MESHLET_VERTEX_SIZE);

impl MeshletVertex {
    /// Create a new meshlet vertex.
    pub fn new(
        position: [f32; 3],
        normal: [f32; 3],
        texcoord: [f32; 2],
        tangent: [f32; 4],
    ) -> Self {
        Self {
            position,
            normal_packed: Self::encode_octahedral_normal(normal),
            texcoord,
            tangent_packed: Self::encode_tangent(tangent),
        }
    }

    /// Encode a normal vector using octahedral encoding to SNorm16x2.
    pub fn encode_octahedral_normal(n: [f32; 3]) -> f32 {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        let n = if len > 0.0 {
            [n[0] / len, n[1] / len, n[2] / len]
        } else {
            [0.0, 0.0, 1.0]
        };

        let sum = n[0].abs() + n[1].abs() + n[2].abs();
        let mut oct = [n[0] / sum, n[1] / sum];

        if n[2] < 0.0 {
            let sign_x = if oct[0] >= 0.0 { 1.0 } else { -1.0 };
            let sign_y = if oct[1] >= 0.0 { 1.0 } else { -1.0 };
            oct = [
                (1.0 - oct[1].abs()) * sign_x,
                (1.0 - oct[0].abs()) * sign_y,
            ];
        }

        // Convert to SNorm16x2
        let x_snorm = ((oct[0].clamp(-1.0, 1.0) * 32767.0) as i32 + 32768) as u32;
        let y_snorm = ((oct[1].clamp(-1.0, 1.0) * 32767.0) as i32 + 32768) as u32;
        let packed = (x_snorm & 0xFFFF) | ((y_snorm & 0xFFFF) << 16);

        f32::from_bits(packed)
    }

    /// Encode tangent with sign to packed format.
    pub fn encode_tangent(t: [f32; 4]) -> [f32; 2] {
        let direction_packed = Self::encode_octahedral_normal([t[0], t[1], t[2]]);
        let sign_bit = if t[3] >= 0.0 { 0u32 } else { 1u32 };

        [direction_packed, f32::from_bits(sign_bit)]
    }

    /// Decode the packed normal back to a unit vector.
    pub fn decode_normal(&self) -> [f32; 3] {
        let bits = self.normal_packed.to_bits();
        let x_snorm = (bits & 0xFFFF) as i32 - 32768;
        let y_snorm = ((bits >> 16) & 0xFFFF) as i32 - 32768;

        let oct = [
            x_snorm as f32 / 32767.0,
            y_snorm as f32 / 32767.0,
        ];

        let mut n = [oct[0], oct[1], 1.0 - oct[0].abs() - oct[1].abs()];

        if n[2] < 0.0 {
            let sign_x = if n[0] >= 0.0 { 1.0 } else { -1.0 };
            let sign_y = if n[1] >= 0.0 { 1.0 } else { -1.0 };
            n = [
                (1.0 - oct[1].abs()) * sign_x,
                (1.0 - oct[0].abs()) * sign_y,
                n[2],
            ];
        }

        // Normalize
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len > 0.0 {
            [n[0] / len, n[1] / len, n[2] / len]
        } else {
            [0.0, 0.0, 1.0]
        }
    }
}

// =============================================================================
// MeshletInstance
// =============================================================================

/// Per-meshlet instance data for GPU rendering.
///
/// Contains transform and material references for a single meshlet instance.
///
/// # Memory Layout
///
/// 144 bytes:
/// | Offset | Field         | Size |
/// |--------|---------------|------|
/// | 0      | model         | 64   |
/// | 64     | normal_matrix | 64   |
/// | 128    | meshlet_index | 4    |
/// | 132    | mesh_id       | 4    |
/// | 136    | material_id   | 4    |
/// | 140    | flags         | 4    |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct MeshletInstance {
    /// Model matrix (world transform).
    pub model: [[f32; 4]; 4],
    /// Normal matrix (inverse transpose of model).
    pub normal_matrix: [[f32; 4]; 4],
    /// Index of the meshlet within the mesh.
    pub meshlet_index: u32,
    /// Mesh ID for vertex/index buffer lookup.
    pub mesh_id: u32,
    /// Material ID for shading.
    pub material_id: u32,
    /// Instance flags.
    pub flags: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<MeshletInstance>() == MESHLET_INSTANCE_SIZE);

impl MeshletInstance {
    /// Create a new meshlet instance with identity transform.
    pub fn new(meshlet_index: u32, mesh_id: u32, material_id: u32) -> Self {
        Self {
            model: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            normal_matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            meshlet_index,
            mesh_id,
            material_id,
            flags: 0,
        }
    }

    /// Set the model matrix.
    pub fn with_transform(mut self, model: [[f32; 4]; 4]) -> Self {
        self.model = model;
        // Compute normal matrix (simplified: just use upper-left 3x3)
        // For proper normal transformation, use inverse transpose
        self.normal_matrix = model;
        self
    }
}

// =============================================================================
// MeshletDrawCommand
// =============================================================================

/// Helper for generating indirect draw commands from meshlet visibility data.
#[derive(Clone, Debug, Default)]
pub struct MeshletDrawCommand {
    /// The indirect draw arguments.
    pub args: IndirectDrawIndexedArgs,
    /// First meshlet instance in this draw batch.
    pub first_instance: u32,
    /// Number of meshlet instances.
    pub instance_count: u32,
}

impl MeshletDrawCommand {
    /// Create a draw command for a single meshlet.
    ///
    /// # Arguments
    ///
    /// * `meshlet` - The meshlet descriptor
    /// * `instance_index` - The instance index in the instance buffer
    pub fn from_meshlet(meshlet: &Meshlet, instance_index: u32) -> Self {
        Self {
            args: IndirectDrawIndexedArgs::new(
                meshlet.triangle_count as u32 * 3, // 3 vertices per triangle
                1,                                  // Single instance per meshlet
                0,                                  // first_index (handled by vertex shader)
                0,                                  // base_vertex
                instance_index as u32,             // first_instance
            ),
            first_instance: instance_index,
            instance_count: 1,
        }
    }

    /// Generate draw commands for a batch of visible meshlets.
    ///
    /// # Arguments
    ///
    /// * `meshlets` - Array of meshlet descriptors
    /// * `visibility` - Visibility flags (true = visible)
    pub fn generate_from_visibility(
        meshlets: &[Meshlet],
        visibility: &[bool],
    ) -> Vec<Self> {
        let mut commands = Vec::new();
        let mut instance_index = 0u32;

        for (i, meshlet) in meshlets.iter().enumerate() {
            if i < visibility.len() && visibility[i] && meshlet.triangle_count > 0 {
                commands.push(Self::from_meshlet(meshlet, instance_index));
                instance_index += 1;
            }
        }

        commands
    }

    /// Generate a single multi-draw command for all visible meshlets.
    ///
    /// This batches all visible meshlets into a single indirect draw,
    /// using instance_id to distinguish them.
    pub fn generate_batched(
        meshlets: &[Meshlet],
        visibility: &[bool],
    ) -> (Vec<IndirectDrawIndexedArgs>, u32) {
        let mut commands = Vec::new();
        let mut visible_count = 0u32;

        for (i, meshlet) in meshlets.iter().enumerate() {
            if i < visibility.len() && visibility[i] && meshlet.triangle_count > 0 {
                commands.push(IndirectDrawIndexedArgs::new(
                    meshlet.triangle_count as u32 * 3,
                    1,
                    0,
                    0,
                    visible_count,
                ));
                visible_count += 1;
            }
        }

        (commands, visible_count)
    }

    /// Get the number of indices this draw will process.
    #[inline]
    pub fn index_count(&self) -> u32 {
        self.args.index_count
    }

    /// Check if this draw command is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.args.index_count == 0 || self.instance_count == 0
    }
}

// =============================================================================
// MeshletRenderResources
// =============================================================================

/// GPU resources for meshlet rendering.
pub struct MeshletRenderResources {
    /// Uniform buffer for render parameters.
    pub params_buffer: Buffer,
    /// Instance buffer for visible meshlets.
    pub instance_buffer: Buffer,
    /// Indirect draw command buffer.
    pub draw_commands_buffer: Buffer,
    /// Draw count buffer (for multi-draw-indirect-count).
    pub draw_count_buffer: Buffer,
    /// Maximum supported instances.
    pub max_instances: u32,
    /// Maximum supported draw commands.
    pub max_draws: u32,
}

impl MeshletRenderResources {
    /// Create render resources.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `max_instances` - Maximum meshlet instances to support
    /// * `max_draws` - Maximum indirect draw commands
    pub fn new(device: &Device, max_instances: u32, max_draws: u32) -> Self {
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("meshlet_render_params"),
            size: MESHLET_RENDER_PARAMS_SIZE as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("meshlet_instances"),
            size: (max_instances as u64) * (MESHLET_INSTANCE_SIZE as u64),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let draw_commands_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("meshlet_draw_commands"),
            size: (max_draws as u64) * (INDIRECT_DRAW_INDEXED_ARGS_SIZE as u64),
            usage: BufferUsages::INDIRECT | BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let draw_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("meshlet_draw_count"),
            size: 4,
            usage: BufferUsages::INDIRECT | BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            params_buffer,
            instance_buffer,
            draw_commands_buffer,
            draw_count_buffer,
            max_instances,
            max_draws,
        }
    }

    /// Upload render parameters.
    pub fn upload_params(&self, queue: &Queue, params: &MeshletRenderParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Upload meshlet instances.
    pub fn upload_instances(&self, queue: &Queue, instances: &[MeshletInstance]) {
        assert!(instances.len() <= self.max_instances as usize);
        queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(instances));
    }

    /// Upload draw commands.
    pub fn upload_draw_commands(&self, queue: &Queue, commands: &[IndirectDrawIndexedArgs]) {
        assert!(commands.len() <= self.max_draws as usize);
        queue.write_buffer(
            &self.draw_commands_buffer,
            0,
            bytemuck::cast_slice(commands),
        );
    }

    /// Upload draw count.
    pub fn upload_draw_count(&self, queue: &Queue, count: u32) {
        queue.write_buffer(&self.draw_count_buffer, 0, &count.to_le_bytes());
    }
}

// =============================================================================
// MeshletRenderPipeline
// =============================================================================

/// Pipeline for rendering meshlets with multiple entry points.
pub struct MeshletRenderPipeline {
    /// Visibility pass render pipeline.
    visibility_pipeline: wgpu::RenderPipeline,
    /// G-buffer pass render pipeline.
    gbuffer_pipeline: wgpu::RenderPipeline,
    /// Shadow pass render pipeline.
    shadow_pipeline: wgpu::RenderPipeline,
    /// Bind group layout for camera/params.
    camera_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group layout for meshlet instances.
    instance_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group layout for meshlet geometry.
    geometry_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group layout for textures.
    texture_bind_group_layout: wgpu::BindGroupLayout,
}

impl MeshletRenderPipeline {
    /// Create the meshlet render pipeline.
    pub fn new(
        device: &Device,
        gbuffer_formats: &[wgpu::TextureFormat],
        depth_format: wgpu::TextureFormat,
        visibility_format: wgpu::TextureFormat,
    ) -> Self {
        // Create bind group layouts
        let camera_bind_group_layout = Self::create_camera_bind_group_layout(device);
        let instance_bind_group_layout = Self::create_instance_bind_group_layout(device);
        let geometry_bind_group_layout = Self::create_geometry_bind_group_layout(device);
        let texture_bind_group_layout = Self::create_texture_bind_group_layout(device);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("meshlet_render_pipeline_layout"),
            bind_group_layouts: &[
                &camera_bind_group_layout,
                &instance_bind_group_layout,
                &geometry_bind_group_layout,
                &texture_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        // Load shaders
        let vertex_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("meshlet_render_vertex"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/gpu_driven/meshlet_render.vert.wgsl").into(),
            ),
        });

        let fragment_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("meshlet_render_fragment"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/gpu_driven/meshlet_render.frag.wgsl").into(),
            ),
        });

        // Create visibility pipeline
        let visibility_pipeline = Self::create_visibility_pipeline(
            device,
            &pipeline_layout,
            &vertex_shader,
            &fragment_shader,
            visibility_format,
            depth_format,
        );

        // Create G-buffer pipeline
        let gbuffer_pipeline = Self::create_gbuffer_pipeline(
            device,
            &pipeline_layout,
            &vertex_shader,
            &fragment_shader,
            gbuffer_formats,
            depth_format,
        );

        // Create shadow pipeline
        let shadow_pipeline = Self::create_shadow_pipeline(
            device,
            &pipeline_layout,
            &vertex_shader,
            depth_format,
        );

        Self {
            visibility_pipeline,
            gbuffer_pipeline,
            shadow_pipeline,
            camera_bind_group_layout,
            instance_bind_group_layout,
            geometry_bind_group_layout,
            texture_bind_group_layout,
        }
    }

    /// Get the camera bind group layout.
    pub fn camera_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.camera_bind_group_layout
    }

    /// Get the instance bind group layout.
    pub fn instance_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.instance_bind_group_layout
    }

    /// Get the geometry bind group layout.
    pub fn geometry_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.geometry_bind_group_layout
    }

    /// Get the texture bind group layout.
    pub fn texture_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.texture_bind_group_layout
    }

    fn create_camera_bind_group_layout(device: &Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("meshlet_camera_bind_group_layout"),
            entries: &[
                // Camera uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Render params
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    fn create_instance_bind_group_layout(device: &Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("meshlet_instance_bind_group_layout"),
            entries: &[
                // Meshlet instances
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    fn create_geometry_bind_group_layout(device: &Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("meshlet_geometry_bind_group_layout"),
            entries: &[
                // Meshlet vertices
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Meshlet descriptors
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Vertex indices
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Local indices
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    fn create_texture_bind_group_layout(device: &Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("meshlet_texture_bind_group_layout"),
            entries: &[
                // Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Texture array
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        })
    }

    fn create_visibility_pipeline(
        device: &Device,
        layout: &wgpu::PipelineLayout,
        vertex_shader: &wgpu::ShaderModule,
        fragment_shader: &wgpu::ShaderModule,
        visibility_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("meshlet_visibility_pipeline"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: vertex_shader,
                entry_point: "vs_visibility",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: fragment_shader,
                entry_point: "fs_visibility",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: visibility_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        })
    }

    fn create_gbuffer_pipeline(
        device: &Device,
        layout: &wgpu::PipelineLayout,
        vertex_shader: &wgpu::ShaderModule,
        fragment_shader: &wgpu::ShaderModule,
        gbuffer_formats: &[wgpu::TextureFormat],
        depth_format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        let targets: Vec<_> = gbuffer_formats
            .iter()
            .map(|&format| {
                Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })
            })
            .collect();

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("meshlet_gbuffer_pipeline"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: vertex_shader,
                entry_point: "vs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: fragment_shader,
                entry_point: "fs_gbuffer",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &targets,
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        })
    }

    fn create_shadow_pipeline(
        device: &Device,
        layout: &wgpu::PipelineLayout,
        vertex_shader: &wgpu::ShaderModule,
        depth_format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("meshlet_shadow_pipeline"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: vertex_shader,
                entry_point: "vs_shadow",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: None, // Depth-only
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        })
    }

    /// Get the visibility pipeline.
    pub fn visibility_pipeline(&self) -> &wgpu::RenderPipeline {
        &self.visibility_pipeline
    }

    /// Get the G-buffer pipeline.
    pub fn gbuffer_pipeline(&self) -> &wgpu::RenderPipeline {
        &self.gbuffer_pipeline
    }

    /// Get the shadow pipeline.
    pub fn shadow_pipeline(&self) -> &wgpu::RenderPipeline {
        &self.shadow_pipeline
    }
}

// =============================================================================
// CPU Reference Implementations
// =============================================================================

/// CPU reference implementation for vertex transformation.
///
/// Transforms a vertex position from local space to world space.
pub fn cpu_transform_vertex(
    position: [f32; 3],
    model: &[[f32; 4]; 4],
) -> [f32; 3] {
    let x = model[0][0] * position[0] + model[1][0] * position[1] + model[2][0] * position[2] + model[3][0];
    let y = model[0][1] * position[0] + model[1][1] * position[1] + model[2][1] * position[2] + model[3][1];
    let z = model[0][2] * position[0] + model[1][2] * position[1] + model[2][2] * position[2] + model[3][2];

    [x, y, z]
}

/// CPU reference implementation for visibility ID packing.
///
/// Packs instance_id (20 bits) and primitive_id (12 bits) into a single u32.
pub fn cpu_pack_visibility_id(instance_id: u32, primitive_id: u32) -> u32 {
    ((instance_id & 0xFFFFF) << 12) | (primitive_id & 0xFFF)
}

/// CPU reference implementation for visibility ID unpacking.
///
/// Returns (instance_id, primitive_id) from a packed visibility ID.
pub fn cpu_unpack_visibility_id(visibility_id: u32) -> (u32, u32) {
    let instance_id = visibility_id >> 12;
    let primitive_id = visibility_id & 0xFFF;
    (instance_id, primitive_id)
}

/// CPU reference implementation for normal transformation.
///
/// Transforms a normal vector using the normal matrix.
pub fn cpu_transform_normal(
    normal: [f32; 3],
    normal_matrix: &[[f32; 4]; 4],
) -> [f32; 3] {
    let x = normal_matrix[0][0] * normal[0] + normal_matrix[1][0] * normal[1] + normal_matrix[2][0] * normal[2];
    let y = normal_matrix[0][1] * normal[0] + normal_matrix[1][1] * normal[1] + normal_matrix[2][1] * normal[2];
    let z = normal_matrix[0][2] * normal[0] + normal_matrix[1][2] * normal[1] + normal_matrix[2][2] * normal[2];

    // Normalize
    let len = (x * x + y * y + z * z).sqrt();
    if len > 0.0 {
        [x / len, y / len, z / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Size/Layout Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_meshlet_render_params_size() {
        assert_eq!(mem::size_of::<MeshletRenderParams>(), MESHLET_RENDER_PARAMS_SIZE);
        assert_eq!(mem::size_of::<MeshletRenderParams>(), 64);
    }

    #[test]
    fn test_meshlet_vertex_size() {
        assert_eq!(mem::size_of::<MeshletVertex>(), MESHLET_VERTEX_SIZE);
        assert_eq!(mem::size_of::<MeshletVertex>(), 32);
    }

    #[test]
    fn test_meshlet_instance_size() {
        assert_eq!(mem::size_of::<MeshletInstance>(), MESHLET_INSTANCE_SIZE);
        assert_eq!(mem::size_of::<MeshletInstance>(), 144);
    }

    #[test]
    fn test_meshlet_render_params_alignment() {
        assert_eq!(mem::align_of::<MeshletRenderParams>(), 4);
    }

    #[test]
    fn test_meshlet_vertex_alignment() {
        assert_eq!(mem::align_of::<MeshletVertex>(), 4);
    }

    #[test]
    fn test_meshlet_instance_alignment() {
        assert_eq!(mem::align_of::<MeshletInstance>(), 4);
    }

    // -------------------------------------------------------------------------
    // MeshletRenderParams Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_render_params_new() {
        let params = MeshletRenderParams::new(100, 1920.0, 1080.0);
        assert_eq!(params.num_meshlets, 100);
        assert_eq!(params.viewport_size, [1920.0, 1080.0]);
        assert_eq!(params.flags, 0);
    }

    #[test]
    fn test_render_params_visibility_pass() {
        let params = MeshletRenderParams::visibility_pass(50, 1280.0, 720.0);
        assert!(params.is_visibility_mode());
        assert!(!params.is_shadow_pass());
    }

    #[test]
    fn test_render_params_shadow_pass() {
        let params = MeshletRenderParams::shadow_pass(200);
        assert!(params.is_shadow_pass());
        assert!(!params.is_visibility_mode());
    }

    #[test]
    fn test_render_params_with_alpha_test() {
        let params = MeshletRenderParams::new(10, 800.0, 600.0).with_alpha_test(0.75);
        assert!(params.is_alpha_test());
        assert_eq!(params.alpha_cutoff, 0.75);
    }

    #[test]
    fn test_render_params_with_double_sided() {
        let params = MeshletRenderParams::new(10, 800.0, 600.0).with_double_sided();
        assert_eq!(params.flags & FLAG_DOUBLE_SIDED, FLAG_DOUBLE_SIDED);
    }

    #[test]
    fn test_render_params_pod() {
        let params = MeshletRenderParams::new(100, 1920.0, 1080.0);
        let bytes = bytemuck::bytes_of(&params);
        assert_eq!(bytes.len(), MESHLET_RENDER_PARAMS_SIZE);
    }

    // -------------------------------------------------------------------------
    // MeshletVertex Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_meshlet_vertex_new() {
        let vertex = MeshletVertex::new(
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5],
            [1.0, 0.0, 0.0, 1.0],
        );
        assert_eq!(vertex.position, [1.0, 2.0, 3.0]);
        assert_eq!(vertex.texcoord, [0.5, 0.5]);
    }

    #[test]
    fn test_meshlet_vertex_normal_encoding_z_up() {
        let normal = [0.0, 0.0, 1.0];
        let vertex = MeshletVertex::new([0.0; 3], normal, [0.0; 2], [1.0, 0.0, 0.0, 1.0]);
        let decoded = vertex.decode_normal();

        assert!((decoded[0] - normal[0]).abs() < 0.01);
        assert!((decoded[1] - normal[1]).abs() < 0.01);
        assert!((decoded[2] - normal[2]).abs() < 0.01);
    }

    #[test]
    fn test_meshlet_vertex_normal_encoding_x_axis() {
        let normal = [1.0, 0.0, 0.0];
        let vertex = MeshletVertex::new([0.0; 3], normal, [0.0; 2], [0.0, 1.0, 0.0, 1.0]);
        let decoded = vertex.decode_normal();

        assert!((decoded[0] - normal[0]).abs() < 0.01);
        assert!((decoded[1] - normal[1]).abs() < 0.01);
        assert!((decoded[2] - normal[2]).abs() < 0.01);
    }

    #[test]
    fn test_meshlet_vertex_normal_encoding_diagonal() {
        let normal = [0.577, 0.577, 0.577];
        let vertex = MeshletVertex::new([0.0; 3], normal, [0.0; 2], [1.0, 0.0, 0.0, 1.0]);
        let decoded = vertex.decode_normal();

        // Normalize expected
        let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        let expected = [normal[0] / len, normal[1] / len, normal[2] / len];

        assert!((decoded[0] - expected[0]).abs() < 0.02);
        assert!((decoded[1] - expected[1]).abs() < 0.02);
        assert!((decoded[2] - expected[2]).abs() < 0.02);
    }

    #[test]
    fn test_meshlet_vertex_normal_encoding_z_down() {
        let normal = [0.0, 0.0, -1.0];
        let vertex = MeshletVertex::new([0.0; 3], normal, [0.0; 2], [1.0, 0.0, 0.0, 1.0]);
        let decoded = vertex.decode_normal();

        assert!((decoded[0] - normal[0]).abs() < 0.01);
        assert!((decoded[1] - normal[1]).abs() < 0.01);
        assert!((decoded[2] - normal[2]).abs() < 0.01);
    }

    #[test]
    fn test_meshlet_vertex_pod() {
        let vertex = MeshletVertex::new([1.0, 2.0, 3.0], [0.0, 1.0, 0.0], [0.5, 0.5], [1.0, 0.0, 0.0, 1.0]);
        let bytes = bytemuck::bytes_of(&vertex);
        assert_eq!(bytes.len(), MESHLET_VERTEX_SIZE);
    }

    // -------------------------------------------------------------------------
    // MeshletInstance Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_meshlet_instance_new() {
        let instance = MeshletInstance::new(10, 5, 3);
        assert_eq!(instance.meshlet_index, 10);
        assert_eq!(instance.mesh_id, 5);
        assert_eq!(instance.material_id, 3);
        // Check identity matrix
        assert_eq!(instance.model[0][0], 1.0);
        assert_eq!(instance.model[1][1], 1.0);
        assert_eq!(instance.model[2][2], 1.0);
        assert_eq!(instance.model[3][3], 1.0);
    }

    #[test]
    fn test_meshlet_instance_with_transform() {
        let transform = [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [1.0, 2.0, 3.0, 1.0],
        ];
        let instance = MeshletInstance::new(0, 0, 0).with_transform(transform);
        assert_eq!(instance.model[0][0], 2.0);
        assert_eq!(instance.model[3][0], 1.0);
        assert_eq!(instance.model[3][1], 2.0);
        assert_eq!(instance.model[3][2], 3.0);
    }

    #[test]
    fn test_meshlet_instance_pod() {
        let instance = MeshletInstance::new(10, 5, 3);
        let bytes = bytemuck::bytes_of(&instance);
        assert_eq!(bytes.len(), MESHLET_INSTANCE_SIZE);
    }

    // -------------------------------------------------------------------------
    // MeshletDrawCommand Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_draw_command_from_meshlet() {
        let meshlet = Meshlet::new(0, 0, 32, 40);
        let cmd = MeshletDrawCommand::from_meshlet(&meshlet, 5);

        assert_eq!(cmd.args.index_count, 120); // 40 triangles * 3 vertices
        assert_eq!(cmd.args.instance_count, 1);
        assert_eq!(cmd.args.first_instance, 5);
        assert_eq!(cmd.first_instance, 5);
    }

    #[test]
    fn test_draw_command_is_empty() {
        let empty_meshlet = Meshlet::new(0, 0, 0, 0);
        let cmd = MeshletDrawCommand::from_meshlet(&empty_meshlet, 0);
        assert!(cmd.is_empty());

        let valid_meshlet = Meshlet::new(0, 0, 3, 1);
        let cmd = MeshletDrawCommand::from_meshlet(&valid_meshlet, 0);
        assert!(!cmd.is_empty());
    }

    #[test]
    fn test_generate_from_visibility() {
        let meshlets = vec![
            Meshlet::new(0, 0, 10, 5),
            Meshlet::new(10, 15, 20, 10),
            Meshlet::new(30, 45, 15, 8),
        ];
        let visibility = vec![true, false, true];

        let commands = MeshletDrawCommand::generate_from_visibility(&meshlets, &visibility);

        assert_eq!(commands.len(), 2);
        assert_eq!(commands[0].first_instance, 0);
        assert_eq!(commands[1].first_instance, 1);
    }

    #[test]
    fn test_generate_batched() {
        let meshlets = vec![
            Meshlet::new(0, 0, 10, 5),
            Meshlet::new(10, 15, 20, 10),
            Meshlet::new(30, 45, 15, 8),
        ];
        let visibility = vec![true, true, false];

        let (commands, count) = MeshletDrawCommand::generate_batched(&meshlets, &visibility);

        assert_eq!(count, 2);
        assert_eq!(commands.len(), 2);
        assert_eq!(commands[0].first_instance, 0);
        assert_eq!(commands[1].first_instance, 1);
    }

    // -------------------------------------------------------------------------
    // CPU Reference Implementation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cpu_transform_vertex_identity() {
        let position = [1.0, 2.0, 3.0];
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let result = cpu_transform_vertex(position, &identity);
        assert_eq!(result, position);
    }

    #[test]
    fn test_cpu_transform_vertex_translation() {
        let position = [1.0, 2.0, 3.0];
        let translate = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [10.0, 20.0, 30.0, 1.0],
        ];
        let result = cpu_transform_vertex(position, &translate);
        assert_eq!(result, [11.0, 22.0, 33.0]);
    }

    #[test]
    fn test_cpu_transform_vertex_scale() {
        let position = [1.0, 2.0, 3.0];
        let scale = [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let result = cpu_transform_vertex(position, &scale);
        assert_eq!(result, [2.0, 6.0, 12.0]);
    }

    #[test]
    fn test_cpu_pack_visibility_id() {
        let instance_id = 12345u32;
        let primitive_id = 678u32;
        let packed = cpu_pack_visibility_id(instance_id, primitive_id);

        assert_eq!(packed >> 12, instance_id);
        assert_eq!(packed & 0xFFF, primitive_id);
    }

    #[test]
    fn test_cpu_pack_visibility_id_max_values() {
        // Max instance_id is 20 bits = 0xFFFFF
        let max_instance = 0xFFFFF;
        // Max primitive_id is 12 bits = 0xFFF
        let max_primitive = 0xFFF;

        let packed = cpu_pack_visibility_id(max_instance, max_primitive);
        let (unpacked_instance, unpacked_primitive) = cpu_unpack_visibility_id(packed);

        assert_eq!(unpacked_instance, max_instance);
        assert_eq!(unpacked_primitive, max_primitive);
    }

    #[test]
    fn test_cpu_unpack_visibility_id() {
        let packed = 0x12345678u32;
        let (instance_id, primitive_id) = cpu_unpack_visibility_id(packed);

        assert_eq!(instance_id, packed >> 12);
        assert_eq!(primitive_id, packed & 0xFFF);
    }

    #[test]
    fn test_cpu_visibility_id_roundtrip() {
        for instance in [0, 1, 100, 1000, 0xFFFFF] {
            for primitive in [0, 1, 100, 0xFFF] {
                let packed = cpu_pack_visibility_id(instance, primitive);
                let (unpacked_instance, unpacked_primitive) = cpu_unpack_visibility_id(packed);
                assert_eq!(unpacked_instance, instance);
                assert_eq!(unpacked_primitive, primitive);
            }
        }
    }

    #[test]
    fn test_cpu_transform_normal_identity() {
        let normal = [0.0, 1.0, 0.0];
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let result = cpu_transform_normal(normal, &identity);
        assert!((result[0] - normal[0]).abs() < 0.001);
        assert!((result[1] - normal[1]).abs() < 0.001);
        assert!((result[2] - normal[2]).abs() < 0.001);
    }

    #[test]
    fn test_cpu_transform_normal_preserves_unit_length() {
        let normal = [0.577, 0.577, 0.577];
        let rotation = [
            [0.707, -0.707, 0.0, 0.0],
            [0.707, 0.707, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let result = cpu_transform_normal(normal, &rotation);
        let len = (result[0] * result[0] + result[1] * result[1] + result[2] * result[2]).sqrt();
        assert!((len - 1.0).abs() < 0.01);
    }

    // -------------------------------------------------------------------------
    // Shader Validation Tests (using naga)
    // -------------------------------------------------------------------------

    #[test]
    fn test_vertex_shader_parses() {
        let shader_source =
            include_str!("../../shaders/gpu_driven/meshlet_render.vert.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("vertex shader should parse without errors");

        let entry_names: Vec<_> = module.entry_points.iter().map(|ep| &ep.name).collect();

        assert!(
            entry_names.iter().any(|n| *n == "vs_main"),
            "Should have vs_main entry point"
        );
        assert!(
            entry_names.iter().any(|n| *n == "vs_visibility"),
            "Should have vs_visibility entry point"
        );
        assert!(
            entry_names.iter().any(|n| *n == "vs_shadow"),
            "Should have vs_shadow entry point"
        );
    }

    #[test]
    fn test_vertex_shader_validates() {
        let shader_source =
            include_str!("../../shaders/gpu_driven/meshlet_render.vert.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("vertex shader should parse without errors");

        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );

        validator
            .validate(&module)
            .expect("vertex shader should validate without errors");
    }

    #[test]
    fn test_fragment_shader_parses() {
        let shader_source =
            include_str!("../../shaders/gpu_driven/meshlet_render.frag.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("fragment shader should parse without errors");

        let entry_names: Vec<_> = module.entry_points.iter().map(|ep| &ep.name).collect();

        assert!(
            entry_names.iter().any(|n| *n == "fs_gbuffer"),
            "Should have fs_gbuffer entry point"
        );
        assert!(
            entry_names.iter().any(|n| *n == "fs_visibility"),
            "Should have fs_visibility entry point"
        );
    }

    #[test]
    fn test_fragment_shader_validates() {
        let shader_source =
            include_str!("../../shaders/gpu_driven/meshlet_render.frag.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("fragment shader should parse without errors");

        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );

        validator
            .validate(&module)
            .expect("fragment shader should validate without errors");
    }

    #[test]
    fn test_vertex_shader_entry_points_are_vertex_stage() {
        let shader_source =
            include_str!("../../shaders/gpu_driven/meshlet_render.vert.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("vertex shader should parse without errors");

        for ep in &module.entry_points {
            assert_eq!(
                ep.stage,
                naga::ShaderStage::Vertex,
                "Entry point {} should be a vertex shader",
                ep.name
            );
        }
    }

    #[test]
    fn test_fragment_shader_entry_points_are_fragment_stage() {
        let shader_source =
            include_str!("../../shaders/gpu_driven/meshlet_render.frag.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("fragment shader should parse without errors");

        for ep in &module.entry_points {
            assert_eq!(
                ep.stage,
                naga::ShaderStage::Fragment,
                "Entry point {} should be a fragment shader",
                ep.name
            );
        }
    }

    // -------------------------------------------------------------------------
    // Octahedral Encoding Precision Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_octahedral_encoding_precision() {
        let test_normals = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.577, 0.577, 0.577],
            [-0.577, 0.577, 0.577],
            [0.707, 0.707, 0.0],
        ];

        for normal in test_normals {
            let vertex = MeshletVertex::new([0.0; 3], normal, [0.0; 2], [1.0, 0.0, 0.0, 1.0]);
            let decoded = vertex.decode_normal();

            // Normalize expected
            let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
            let expected = [normal[0] / len, normal[1] / len, normal[2] / len];

            let error =
                (decoded[0] - expected[0]).abs() +
                (decoded[1] - expected[1]).abs() +
                (decoded[2] - expected[2]).abs();

            assert!(
                error < 0.1,
                "Normal {:?} decoded to {:?}, expected {:?}, error {}",
                normal,
                decoded,
                expected,
                error
            );
        }
    }

    // -------------------------------------------------------------------------
    // Draw Command Generation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_generate_from_visibility_empty() {
        let meshlets: Vec<Meshlet> = vec![];
        let visibility: Vec<bool> = vec![];
        let commands = MeshletDrawCommand::generate_from_visibility(&meshlets, &visibility);
        assert!(commands.is_empty());
    }

    #[test]
    fn test_generate_from_visibility_all_invisible() {
        let meshlets = vec![
            Meshlet::new(0, 0, 10, 5),
            Meshlet::new(10, 15, 20, 10),
        ];
        let visibility = vec![false, false];
        let commands = MeshletDrawCommand::generate_from_visibility(&meshlets, &visibility);
        assert!(commands.is_empty());
    }

    #[test]
    fn test_generate_from_visibility_all_visible() {
        let meshlets = vec![
            Meshlet::new(0, 0, 10, 5),
            Meshlet::new(10, 15, 20, 10),
            Meshlet::new(30, 45, 15, 8),
        ];
        let visibility = vec![true, true, true];
        let commands = MeshletDrawCommand::generate_from_visibility(&meshlets, &visibility);
        assert_eq!(commands.len(), 3);
    }

    #[test]
    fn test_generate_from_visibility_skips_empty_meshlets() {
        let meshlets = vec![
            Meshlet::new(0, 0, 10, 5),
            Meshlet::new(10, 15, 20, 0), // Empty
            Meshlet::new(30, 45, 15, 8),
        ];
        let visibility = vec![true, true, true];
        let commands = MeshletDrawCommand::generate_from_visibility(&meshlets, &visibility);
        assert_eq!(commands.len(), 2);
    }
}
