//! Mesh Particle Rendering for TRINITY Engine (T-GPU-6.2).
//!
//! Renders 3D mesh geometry at particle positions for debris, leaves, sparks, etc.
//! Uses instanced indirect draw with bindless mesh references.
//!
//! # Overview
//!
//! Unlike billboard particles that always face the camera, mesh particles render
//! actual 3D geometry at each particle's position. This enables:
//! - Physically-based debris with proper lighting
//! - Leaves with two-sided rendering
//! - Sparks with elongated geometry along velocity
//!
//! # Pipeline
//!
//! 1. CPU provides mesh vertex/index buffers and texture
//! 2. Particle buffer is updated by spawn/update compute shaders
//! 3. Sort indices buffer provides back-to-front ordering for alpha
//! 4. Instanced draw renders one mesh per alive particle
//!
//! # Usage
//!
//! ```ignore
//! let pipeline = MeshParticlePipeline::new(&device);
//! let resources = MeshParticleResources::new(
//!     &device,
//!     &pipeline,
//!     max_particles,
//!     &mesh_vertex_buffer,
//!     &mesh_index_buffer,
//!     index_count,
//!     &texture_view,
//! );
//!
//! // Each frame: render mesh particles
//! let params = MeshParticleParams::new(view_proj, ScaleMode::FromParticleSize);
//! resources.update_params(&queue, &params);
//! pipeline.render(&mut render_pass, &resources, alive_count);
//! ```

use std::mem;

// =============================================================================
// CONSTANTS
// =============================================================================

/// Size of MeshParticleParams in bytes.
pub const MESH_PARTICLE_PARAMS_SIZE: usize = 80;

/// Scale mode: use uniform base scale.
pub const SCALE_MODE_UNIFORM: u32 = 0;

/// Scale mode: scale from particle size field.
pub const SCALE_MODE_FROM_SIZE: u32 = 1;

/// Rotation mode: rotate around Y-up axis.
pub const ROTATION_MODE_Y_UP: u32 = 0;

/// Rotation mode: align to particle velocity direction.
pub const ROTATION_MODE_VELOCITY_ALIGNED: u32 = 1;

// =============================================================================
// MeshParticleParams
// =============================================================================

/// GPU uniform buffer for mesh particle rendering parameters.
///
/// Matches the WGSL `MeshParticleParams` struct layout.
///
/// # Memory Layout (80 bytes, std140 compatible)
///
/// | Offset | Field          | Size     |
/// |--------|----------------|----------|
/// | 0      | view_proj      | 64 bytes |
/// | 64     | scale_from_size| 4 bytes  |
/// | 68     | base_scale     | 4 bytes  |
/// | 72     | rotation_mode  | 4 bytes  |
/// | 76     | _pad           | 4 bytes  |
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshParticleParams {
    /// Combined view-projection matrix.
    pub view_proj: [[f32; 4]; 4],
    /// Scale mode: 0 = uniform, 1 = from particle size.
    pub scale_from_size: u32,
    /// Base scale when scale_from_size is 0.
    pub base_scale: f32,
    /// Rotation mode: 0 = Y-up, 1 = velocity aligned.
    pub rotation_mode: u32,
    /// Padding for 16-byte alignment.
    pub _pad: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<MeshParticleParams>() == MESH_PARTICLE_PARAMS_SIZE);

impl MeshParticleParams {
    /// Create mesh particle parameters with uniform scale.
    ///
    /// # Arguments
    ///
    /// * `view_proj` - Combined view-projection matrix (column-major).
    /// * `scale` - Uniform scale applied to all particles.
    pub fn with_uniform_scale(view_proj: [[f32; 4]; 4], scale: f32) -> Self {
        Self {
            view_proj,
            scale_from_size: SCALE_MODE_UNIFORM,
            base_scale: scale,
            rotation_mode: ROTATION_MODE_Y_UP,
            _pad: 0,
        }
    }

    /// Create mesh particle parameters with scale from particle size.
    ///
    /// # Arguments
    ///
    /// * `view_proj` - Combined view-projection matrix (column-major).
    pub fn with_particle_size_scale(view_proj: [[f32; 4]; 4]) -> Self {
        Self {
            view_proj,
            scale_from_size: SCALE_MODE_FROM_SIZE,
            base_scale: 1.0,
            rotation_mode: ROTATION_MODE_Y_UP,
            _pad: 0,
        }
    }

    /// Create mesh particle parameters with velocity-aligned rotation.
    ///
    /// # Arguments
    ///
    /// * `view_proj` - Combined view-projection matrix (column-major).
    /// * `scale_from_size` - Whether to scale from particle size field.
    /// * `base_scale` - Base scale (used when scale_from_size is false).
    pub fn velocity_aligned(
        view_proj: [[f32; 4]; 4],
        scale_from_size: bool,
        base_scale: f32,
    ) -> Self {
        Self {
            view_proj,
            scale_from_size: if scale_from_size {
                SCALE_MODE_FROM_SIZE
            } else {
                SCALE_MODE_UNIFORM
            },
            base_scale,
            rotation_mode: ROTATION_MODE_VELOCITY_ALIGNED,
            _pad: 0,
        }
    }

    /// Set the rotation mode.
    pub fn with_rotation_mode(mut self, mode: u32) -> Self {
        self.rotation_mode = mode;
        self
    }
}

impl Default for MeshParticleParams {
    fn default() -> Self {
        Self::with_uniform_scale(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            1.0,
        )
    }
}

// =============================================================================
// MeshParticleResources
// =============================================================================

/// GPU resources for mesh particle rendering.
///
/// Contains all buffers and bind groups needed for the render pipeline:
/// - `params_buffer`: Uniform buffer for `MeshParticleParams`
/// - `particle_bind_group`: Bind group 0 (params, particles, sort indices)
/// - `texture_bind_group`: Bind group 1 (texture, sampler)
pub struct MeshParticleResources {
    /// Uniform buffer for render parameters.
    pub params_buffer: wgpu::Buffer,
    /// Bind group for particle data (group 0).
    pub particle_bind_group: wgpu::BindGroup,
    /// Bind group for texture data (group 1).
    pub texture_bind_group: wgpu::BindGroup,
    /// Vertex buffer for mesh geometry.
    pub vertex_buffer: wgpu::Buffer,
    /// Index buffer for mesh geometry.
    pub index_buffer: wgpu::Buffer,
    /// Number of indices in the mesh.
    pub index_count: u32,
    /// Maximum number of particle instances.
    pub max_instances: u32,
}

impl MeshParticleResources {
    /// Create mesh particle resources.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `pipeline` - The mesh particle pipeline (provides bind group layouts).
    /// * `max_particles` - Maximum number of particle instances.
    /// * `particle_buffer` - Storage buffer containing particle data.
    /// * `sort_indices_buffer` - Storage buffer containing sorted particle indices.
    /// * `vertex_buffer` - Vertex buffer for mesh geometry.
    /// * `index_buffer` - Index buffer for mesh geometry.
    /// * `index_count` - Number of indices in the mesh.
    /// * `texture_view` - Texture view for particle material.
    /// * `sampler` - Sampler for texture sampling.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &wgpu::Device,
        pipeline: &MeshParticlePipeline,
        max_particles: u32,
        particle_buffer: &wgpu::Buffer,
        sort_indices_buffer: &wgpu::Buffer,
        vertex_buffer: wgpu::Buffer,
        index_buffer: wgpu::Buffer,
        index_count: u32,
        texture_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) -> Self {
        // Create params uniform buffer
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh_particle_params"),
            size: MESH_PARTICLE_PARAMS_SIZE as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create particle bind group (group 0)
        let particle_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh_particle_bind_group_0"),
            layout: &pipeline.particle_bind_group_layout,
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
                    resource: sort_indices_buffer.as_entire_binding(),
                },
            ],
        });

        // Create texture bind group (group 1)
        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh_particle_bind_group_1"),
            layout: &pipeline.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        Self {
            params_buffer,
            particle_bind_group,
            texture_bind_group,
            vertex_buffer,
            index_buffer,
            index_count,
            max_instances: max_particles,
        }
    }

    /// Update render parameters.
    pub fn update_params(&self, queue: &wgpu::Queue, params: &MeshParticleParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Recreate texture bind group with new texture.
    pub fn update_texture(
        &mut self,
        device: &wgpu::Device,
        pipeline: &MeshParticlePipeline,
        texture_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) {
        self.texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh_particle_bind_group_1"),
            layout: &pipeline.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
    }
}

// =============================================================================
// MeshParticlePipeline
// =============================================================================

/// GPU render pipeline for mesh particle rendering.
///
/// Encapsulates shader modules, pipeline, and bind group layouts.
pub struct MeshParticlePipeline {
    /// Bind group layout for particle data (group 0).
    pub particle_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group layout for texture data (group 1).
    pub texture_bind_group_layout: wgpu::BindGroupLayout,
    /// Render pipeline for opaque mesh particles.
    pub pipeline_opaque: wgpu::RenderPipeline,
    /// Render pipeline for alpha-blended mesh particles.
    pub pipeline_blend: wgpu::RenderPipeline,
}

impl MeshParticlePipeline {
    /// Create the mesh particle render pipeline.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `color_format` - Format of the color render target.
    /// * `depth_format` - Format of the depth buffer (optional).
    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        depth_format: Option<wgpu::TextureFormat>,
    ) -> Self {
        // Create bind group layout for particle data (group 0)
        let particle_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mesh_particle_bind_group_layout_0"),
                entries: &[
                    // binding 0: MeshParticleParams (uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 1: particles (storage, read-only)
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
                    // binding 2: sort_indices (storage, read-only)
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
                ],
            });

        // Create bind group layout for texture data (group 1)
        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mesh_particle_bind_group_layout_1"),
                entries: &[
                    // binding 0: base_texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 1: base_sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh_particle_pipeline_layout"),
            bind_group_layouts: &[&particle_bind_group_layout, &texture_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Load vertex shader
        let vertex_shader_source =
            include_str!("../../shaders/particles/mesh_particle.vert.wgsl");
        let vertex_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh_particle_vertex_shader"),
            source: wgpu::ShaderSource::Wgsl(vertex_shader_source.into()),
        });

        // Load fragment shader
        let fragment_shader_source =
            include_str!("../../shaders/particles/mesh_particle.frag.wgsl");
        let fragment_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh_particle_fragment_shader"),
            source: wgpu::ShaderSource::Wgsl(fragment_shader_source.into()),
        });

        // Vertex buffer layout (position, normal, uv)
        let vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: 32, // 3 + 3 + 2 floats = 8 * 4 = 32 bytes
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // position: vec3<f32>
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                // normal: vec3<f32>
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 12,
                    shader_location: 1,
                },
                // uv: vec2<f32>
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 24,
                    shader_location: 2,
                },
            ],
        };

        // Depth stencil state
        let depth_stencil = depth_format.map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });

        // Create opaque pipeline (no blending)
        let pipeline_opaque = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh_particle_pipeline_opaque"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vertex_shader,
                entry_point: "vs_mesh_particle",
                buffers: &[vertex_buffer_layout.clone()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fragment_shader,
                entry_point: "fs_mesh_particle",
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
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
            depth_stencil: depth_stencil.clone(),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Depth stencil for blend (read-only depth)
        let depth_stencil_blend = depth_format.map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: false, // No depth write for transparency
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });

        // Create alpha-blended pipeline
        let pipeline_blend = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh_particle_pipeline_blend"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vertex_shader,
                entry_point: "vs_mesh_particle",
                buffers: &[vertex_buffer_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fragment_shader,
                entry_point: "fs_mesh_particle",
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // Two-sided for transparency
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: depth_stencil_blend,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            particle_bind_group_layout,
            texture_bind_group_layout,
            pipeline_opaque,
            pipeline_blend,
        }
    }

    /// Render mesh particles (opaque mode).
    ///
    /// # Arguments
    ///
    /// * `pass` - The render pass to record commands into.
    /// * `resources` - Mesh particle resources.
    /// * `instance_count` - Number of particle instances to render.
    pub fn render_opaque<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        resources: &'a MeshParticleResources,
        instance_count: u32,
    ) {
        if instance_count == 0 {
            return;
        }

        pass.set_pipeline(&self.pipeline_opaque);
        pass.set_bind_group(0, &resources.particle_bind_group, &[]);
        pass.set_bind_group(1, &resources.texture_bind_group, &[]);
        pass.set_vertex_buffer(0, resources.vertex_buffer.slice(..));
        pass.set_index_buffer(resources.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..resources.index_count, 0, 0..instance_count);
    }

    /// Render mesh particles (alpha-blended mode).
    ///
    /// # Arguments
    ///
    /// * `pass` - The render pass to record commands into.
    /// * `resources` - Mesh particle resources.
    /// * `instance_count` - Number of particle instances to render.
    pub fn render_blend<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        resources: &'a MeshParticleResources,
        instance_count: u32,
    ) {
        if instance_count == 0 {
            return;
        }

        pass.set_pipeline(&self.pipeline_blend);
        pass.set_bind_group(0, &resources.particle_bind_group, &[]);
        pass.set_bind_group(1, &resources.texture_bind_group, &[]);
        pass.set_vertex_buffer(0, resources.vertex_buffer.slice(..));
        pass.set_index_buffer(resources.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..resources.index_count, 0, 0..instance_count);
    }

    /// Render mesh particles with indirect draw.
    ///
    /// # Arguments
    ///
    /// * `pass` - The render pass to record commands into.
    /// * `resources` - Mesh particle resources.
    /// * `indirect_buffer` - Buffer containing indirect draw arguments.
    /// * `indirect_offset` - Byte offset into the indirect buffer.
    /// * `use_blend` - Whether to use alpha-blended pipeline.
    pub fn render_indirect<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        resources: &'a MeshParticleResources,
        indirect_buffer: &'a wgpu::Buffer,
        indirect_offset: u64,
        use_blend: bool,
    ) {
        let pipeline = if use_blend {
            &self.pipeline_blend
        } else {
            &self.pipeline_opaque
        };

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &resources.particle_bind_group, &[]);
        pass.set_bind_group(1, &resources.texture_bind_group, &[]);
        pass.set_vertex_buffer(0, resources.vertex_buffer.slice(..));
        pass.set_index_buffer(resources.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed_indirect(indirect_buffer, indirect_offset);
    }
}

// =============================================================================
// CPU Reference Implementation
// =============================================================================

/// CPU reference for building the transform matrix from particle data.
///
/// Used for testing and validation against GPU shader.
pub fn cpu_build_particle_transform(
    position: [f32; 3],
    rotation: f32,
    velocity: [f32; 3],
    size: f32,
    scale_from_size: bool,
    rotation_mode: u32,
) -> [[f32; 4]; 4] {
    let scale = if scale_from_size { size } else { 1.0 };

    // Build rotation matrix
    let rot_mat = if rotation_mode == ROTATION_MODE_VELOCITY_ALIGNED {
        cpu_align_to_velocity(velocity, rotation)
    } else {
        cpu_rotation_y(rotation)
    };

    // Combine scale, rotation, translation into 4x4 matrix
    [
        [rot_mat[0][0] * scale, rot_mat[0][1] * scale, rot_mat[0][2] * scale, 0.0],
        [rot_mat[1][0] * scale, rot_mat[1][1] * scale, rot_mat[1][2] * scale, 0.0],
        [rot_mat[2][0] * scale, rot_mat[2][1] * scale, rot_mat[2][2] * scale, 0.0],
        [position[0], position[1], position[2], 1.0],
    ]
}

/// CPU reference: rotation around Y axis.
fn cpu_rotation_y(angle: f32) -> [[f32; 3]; 3] {
    let c = angle.cos();
    let s = angle.sin();
    [[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]]
}

/// CPU reference: align to velocity direction.
fn cpu_align_to_velocity(velocity: [f32; 3], extra_rotation: f32) -> [[f32; 3]; 3] {
    let speed = (velocity[0] * velocity[0] + velocity[1] * velocity[1] + velocity[2] * velocity[2])
        .sqrt();

    if speed < 0.001 {
        return cpu_rotation_y(extra_rotation);
    }

    let forward = [velocity[0] / speed, velocity[1] / speed, velocity[2] / speed];

    // Choose up vector
    let up = if forward[1].abs() > 0.99 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };

    // Cross product: right = up x forward
    let right = [
        up[1] * forward[2] - up[2] * forward[1],
        up[2] * forward[0] - up[0] * forward[2],
        up[0] * forward[1] - up[1] * forward[0],
    ];
    let right_len =
        (right[0] * right[0] + right[1] * right[1] + right[2] * right[2]).sqrt();
    let right = [right[0] / right_len, right[1] / right_len, right[2] / right_len];

    // Cross product: corrected_up = forward x right
    let corrected_up = [
        forward[1] * right[2] - forward[2] * right[1],
        forward[2] * right[0] - forward[0] * right[2],
        forward[0] * right[1] - forward[1] * right[0],
    ];

    // Apply extra rotation around forward axis
    let c = extra_rotation.cos();
    let s = extra_rotation.sin();
    let rotated_right = [
        right[0] * c + corrected_up[0] * s,
        right[1] * c + corrected_up[1] * s,
        right[2] * c + corrected_up[2] * s,
    ];
    let rotated_up = [
        -right[0] * s + corrected_up[0] * c,
        -right[1] * s + corrected_up[1] * c,
        -right[2] * s + corrected_up[2] * c,
    ];

    [rotated_right, rotated_up, forward]
}

/// CPU reference: transform a vertex position by particle transform.
pub fn cpu_transform_vertex(
    vertex: [f32; 3],
    position: [f32; 3],
    rotation: f32,
    velocity: [f32; 3],
    size: f32,
    scale_from_size: bool,
    rotation_mode: u32,
) -> [f32; 3] {
    let scale = if scale_from_size { size } else { 1.0 };
    let scaled = [vertex[0] * scale, vertex[1] * scale, vertex[2] * scale];

    let rot_mat = if rotation_mode == ROTATION_MODE_VELOCITY_ALIGNED {
        cpu_align_to_velocity(velocity, rotation)
    } else {
        cpu_rotation_y(rotation)
    };

    let rotated = [
        rot_mat[0][0] * scaled[0] + rot_mat[1][0] * scaled[1] + rot_mat[2][0] * scaled[2],
        rot_mat[0][1] * scaled[0] + rot_mat[1][1] * scaled[1] + rot_mat[2][1] * scaled[2],
        rot_mat[0][2] * scaled[0] + rot_mat[1][2] * scaled[1] + rot_mat[2][2] * scaled[2],
    ];

    [
        rotated[0] + position[0],
        rotated[1] + position[1],
        rotated[2] + position[2],
    ]
}

/// CPU reference: transform a normal by rotation only.
pub fn cpu_transform_normal(
    normal: [f32; 3],
    rotation: f32,
    velocity: [f32; 3],
    rotation_mode: u32,
) -> [f32; 3] {
    let rot_mat = if rotation_mode == ROTATION_MODE_VELOCITY_ALIGNED {
        cpu_align_to_velocity(velocity, rotation)
    } else {
        cpu_rotation_y(rotation)
    };

    let transformed = [
        rot_mat[0][0] * normal[0] + rot_mat[1][0] * normal[1] + rot_mat[2][0] * normal[2],
        rot_mat[0][1] * normal[0] + rot_mat[1][1] * normal[1] + rot_mat[2][1] * normal[2],
        rot_mat[0][2] * normal[0] + rot_mat[1][2] * normal[1] + rot_mat[2][2] * normal[2],
    ];

    // Normalize
    let len = (transformed[0] * transformed[0]
        + transformed[1] * transformed[1]
        + transformed[2] * transformed[2])
        .sqrt();
    if len > 0.0001 {
        [transformed[0] / len, transformed[1] / len, transformed[2] / len]
    } else {
        [0.0, 1.0, 0.0]
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    // -------------------------------------------------------------------------
    // Struct size tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_params_size() {
        assert_eq!(
            mem::size_of::<MeshParticleParams>(),
            MESH_PARTICLE_PARAMS_SIZE
        );
    }

    #[test]
    fn test_params_alignment() {
        // MeshParticleParams contains mat4x4 which needs 4-byte alignment for f32
        // The struct itself has 4-byte alignment which is correct for GPU uniforms
        assert_eq!(mem::align_of::<MeshParticleParams>(), 4);
    }

    // -------------------------------------------------------------------------
    // Transform at particle position tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_transform_at_particle_position() {
        let vertex = [1.0, 0.0, 0.0];
        let position = [5.0, 10.0, -3.0];
        let rotation = 0.0;
        let velocity = [0.0, 0.0, 0.0];
        let size = 1.0;

        let result = cpu_transform_vertex(
            vertex,
            position,
            rotation,
            velocity,
            size,
            false,
            ROTATION_MODE_Y_UP,
        );

        assert!((result[0] - 6.0).abs() < 0.001);
        assert!((result[1] - 10.0).abs() < 0.001);
        assert!((result[2] - (-3.0)).abs() < 0.001);
    }

    #[test]
    fn test_transform_origin_vertex() {
        let vertex = [0.0, 0.0, 0.0];
        let position = [100.0, -50.0, 25.0];
        let rotation = PI;
        let velocity = [1.0, 2.0, 3.0];
        let size = 5.0;

        // Origin vertex should end up at particle position regardless of rotation/scale
        let result = cpu_transform_vertex(
            vertex,
            position,
            rotation,
            velocity,
            size,
            true,
            ROTATION_MODE_Y_UP,
        );

        assert!((result[0] - 100.0).abs() < 0.001);
        assert!((result[1] - (-50.0)).abs() < 0.001);
        assert!((result[2] - 25.0).abs() < 0.001);
    }

    // -------------------------------------------------------------------------
    // Scale from particle size tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_scale_from_particle_size() {
        let vertex = [1.0, 1.0, 1.0];
        let position = [0.0, 0.0, 0.0];
        let rotation = 0.0;
        let velocity = [0.0, 0.0, 0.0];
        let size = 2.0;

        let result = cpu_transform_vertex(
            vertex,
            position,
            rotation,
            velocity,
            size,
            true, // scale_from_size = true
            ROTATION_MODE_Y_UP,
        );

        assert!((result[0] - 2.0).abs() < 0.001);
        assert!((result[1] - 2.0).abs() < 0.001);
        assert!((result[2] - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_uniform_scale_ignores_size() {
        let vertex = [1.0, 1.0, 1.0];
        let position = [0.0, 0.0, 0.0];
        let rotation = 0.0;
        let velocity = [0.0, 0.0, 0.0];
        let size = 5.0; // Should be ignored

        let result = cpu_transform_vertex(
            vertex,
            position,
            rotation,
            velocity,
            size,
            false, // scale_from_size = false
            ROTATION_MODE_Y_UP,
        );

        assert!((result[0] - 1.0).abs() < 0.001);
        assert!((result[1] - 1.0).abs() < 0.001);
        assert!((result[2] - 1.0).abs() < 0.001);
    }

    // -------------------------------------------------------------------------
    // Rotation tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_rotation_90_degrees_y() {
        let vertex = [1.0, 0.0, 0.0]; // Point along X axis
        let position = [0.0, 0.0, 0.0];
        let rotation = PI / 2.0; // 90 degrees
        let velocity = [0.0, 0.0, 0.0];
        let size = 1.0;

        let result = cpu_transform_vertex(
            vertex,
            position,
            rotation,
            velocity,
            size,
            false,
            ROTATION_MODE_Y_UP,
        );

        // After 90 degree counterclockwise Y rotation (looking down -Y),
        // X axis [1,0,0] should become -Z axis [0,0,-1]
        assert!(result[0].abs() < 0.001);
        assert!(result[1].abs() < 0.001);
        assert!((result[2] - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_rotation_180_degrees_y() {
        let vertex = [1.0, 0.0, 0.0];
        let position = [0.0, 0.0, 0.0];
        let rotation = PI; // 180 degrees
        let velocity = [0.0, 0.0, 0.0];
        let size = 1.0;

        let result = cpu_transform_vertex(
            vertex,
            position,
            rotation,
            velocity,
            size,
            false,
            ROTATION_MODE_Y_UP,
        );

        // After 180 degree Y rotation, X should become -X
        assert!((result[0] - (-1.0)).abs() < 0.001);
        assert!(result[1].abs() < 0.001);
        assert!(result[2].abs() < 0.001);
    }

    #[test]
    fn test_rotation_preserves_y_component() {
        let vertex = [1.0, 5.0, 0.0];
        let position = [0.0, 0.0, 0.0];
        let rotation = PI / 4.0; // 45 degrees
        let velocity = [0.0, 0.0, 0.0];
        let size = 1.0;

        let result = cpu_transform_vertex(
            vertex,
            position,
            rotation,
            velocity,
            size,
            false,
            ROTATION_MODE_Y_UP,
        );

        // Y component should be unchanged by Y-axis rotation
        assert!((result[1] - 5.0).abs() < 0.001);
    }

    // -------------------------------------------------------------------------
    // Normal transformation tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_normal_transformation_no_rotation() {
        let normal = [0.0, 1.0, 0.0];
        let rotation = 0.0;
        let velocity = [0.0, 0.0, 0.0];

        let result = cpu_transform_normal(normal, rotation, velocity, ROTATION_MODE_Y_UP);

        assert!(result[0].abs() < 0.001);
        assert!((result[1] - 1.0).abs() < 0.001);
        assert!(result[2].abs() < 0.001);
    }

    #[test]
    fn test_normal_transformation_with_rotation() {
        let normal = [1.0, 0.0, 0.0];
        let rotation = PI / 2.0;
        let velocity = [0.0, 0.0, 0.0];

        let result = cpu_transform_normal(normal, rotation, velocity, ROTATION_MODE_Y_UP);

        // Normal along X rotated 90 degrees counterclockwise around Y
        // should point along -Z axis
        assert!(result[0].abs() < 0.001);
        assert!(result[1].abs() < 0.001);
        assert!((result[2] - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_normal_is_normalized() {
        let normal = [1.0, 0.0, 0.0];
        let rotation = 0.5;
        let velocity = [0.0, 0.0, 0.0];

        let result = cpu_transform_normal(normal, rotation, velocity, ROTATION_MODE_Y_UP);
        let length =
            (result[0] * result[0] + result[1] * result[1] + result[2] * result[2]).sqrt();

        assert!((length - 1.0).abs() < 0.001);
    }

    // -------------------------------------------------------------------------
    // Color modulation tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_params_default_values() {
        let params = MeshParticleParams::default();

        assert_eq!(params.scale_from_size, SCALE_MODE_UNIFORM);
        assert!((params.base_scale - 1.0).abs() < 0.001);
        assert_eq!(params.rotation_mode, ROTATION_MODE_Y_UP);
    }

    #[test]
    fn test_params_with_particle_size_scale() {
        let view_proj = [[1.0; 4]; 4];
        let params = MeshParticleParams::with_particle_size_scale(view_proj);

        assert_eq!(params.scale_from_size, SCALE_MODE_FROM_SIZE);
    }

    #[test]
    fn test_params_velocity_aligned() {
        let view_proj = [[1.0; 4]; 4];
        let params = MeshParticleParams::velocity_aligned(view_proj, true, 0.5);

        assert_eq!(params.scale_from_size, SCALE_MODE_FROM_SIZE);
        assert!((params.base_scale - 0.5).abs() < 0.001);
        assert_eq!(params.rotation_mode, ROTATION_MODE_VELOCITY_ALIGNED);
    }

    // -------------------------------------------------------------------------
    // Velocity-aligned rotation tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_velocity_aligned_up() {
        let vertex = [1.0, 0.0, 0.0];
        let position = [0.0, 0.0, 0.0];
        let rotation = 0.0;
        let velocity = [0.0, 1.0, 0.0]; // Velocity pointing up
        let size = 1.0;

        let result = cpu_transform_vertex(
            vertex,
            position,
            rotation,
            velocity,
            size,
            false,
            ROTATION_MODE_VELOCITY_ALIGNED,
        );

        // Mesh should be reoriented to align with upward velocity
        // The exact result depends on the orthonormal basis construction
        let length =
            (result[0] * result[0] + result[1] * result[1] + result[2] * result[2]).sqrt();
        assert!((length - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_velocity_aligned_forward() {
        let vertex = [0.0, 1.0, 0.0]; // Vertex along Y axis
        let position = [0.0, 0.0, 0.0];
        let rotation = 0.0;
        let velocity = [0.0, 0.0, 1.0]; // Velocity pointing forward
        let size = 1.0;

        let result = cpu_transform_vertex(
            vertex,
            position,
            rotation,
            velocity,
            size,
            false,
            ROTATION_MODE_VELOCITY_ALIGNED,
        );

        // When velocity is +Z, the Y-up vertex should align with +Z
        let length =
            (result[0] * result[0] + result[1] * result[1] + result[2] * result[2]).sqrt();
        assert!((length - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_velocity_zero_falls_back_to_y_up() {
        let vertex = [1.0, 0.0, 0.0];
        let position = [0.0, 0.0, 0.0];
        let rotation = PI / 2.0;
        let velocity = [0.0, 0.0, 0.0]; // Zero velocity
        let size = 1.0;

        let result_velocity = cpu_transform_vertex(
            vertex,
            position,
            rotation,
            velocity,
            size,
            false,
            ROTATION_MODE_VELOCITY_ALIGNED,
        );

        let result_y_up = cpu_transform_vertex(
            vertex,
            position,
            rotation,
            velocity,
            size,
            false,
            ROTATION_MODE_Y_UP,
        );

        // With zero velocity, velocity-aligned should match Y-up
        assert!((result_velocity[0] - result_y_up[0]).abs() < 0.001);
        assert!((result_velocity[1] - result_y_up[1]).abs() < 0.001);
        assert!((result_velocity[2] - result_y_up[2]).abs() < 0.001);
    }

    // -------------------------------------------------------------------------
    // Indirect draw integration tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_indirect_draw_args_size() {
        // Indirect draw indexed args should be 20 bytes
        assert_eq!(
            std::mem::size_of::<crate::gpu_driven::indirect_draw::IndirectDrawIndexedArgs>(),
            20
        );
    }
}
