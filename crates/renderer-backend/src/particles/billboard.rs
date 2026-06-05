//! Billboard Particle Rendering for TRINITY Engine (T-GPU-6.1).
//!
//! This module implements GPU-based billboard particle rendering. Billboards are
//! camera-facing quads that display particle textures, commonly used for fire,
//! smoke, sparks, and other volumetric effects.
//!
//! # Overview
//!
//! The billboard rendering pipeline:
//! 1. CPU uploads `BillboardParams` with camera matrices and alignment mode
//! 2. GPU reads sorted particle indices (back-to-front order)
//! 3. Vertex shader expands each particle into a 4-vertex quad
//! 4. Fragment shader samples texture and applies color modulation
//!
//! # Alignment Modes
//!
//! - `View`: Billboard always faces camera (standard billboarding)
//! - `Velocity`: Billboard stretches along velocity vector (smoke trails)
//! - `Custom`: Billboard rotates around a custom axis
//!
//! # Blend Modes
//!
//! - `Alpha`: Standard alpha blending for smoke, dust
//! - `Additive`: Add to framebuffer for fire, sparks, glow effects
//!
//! # Usage
//!
//! ```ignore
//! // Create billboard pipeline
//! let pipeline = BillboardPipeline::new(&device, surface_format);
//!
//! // Create resources
//! let resources = BillboardResources::new(
//!     &device,
//!     65536,  // max particles
//!     &pipeline.bind_group_layouts,
//!     particle_buffer,
//!     sort_indices_buffer,
//! );
//!
//! // Each frame: render particles
//! resources.update_params(&queue, &params);
//! pipeline.render(&mut render_pass, &resources, particle_count);
//! ```

use std::mem;

use super::Particle;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Size of BillboardParams in bytes.
pub const BILLBOARD_PARAMS_SIZE: usize = 176;

/// Number of vertices per particle quad (triangle strip).
pub const VERTICES_PER_PARTICLE: u32 = 4;

/// Default white texture size (1x1 pixel).
pub const DEFAULT_TEXTURE_SIZE: u32 = 1;

// ---------------------------------------------------------------------------
// AlignmentMode
// ---------------------------------------------------------------------------

/// Billboard alignment mode.
///
/// Determines how billboards orient relative to the camera.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum AlignmentMode {
    /// Billboard faces camera directly (standard billboarding).
    ///
    /// The billboard plane is always perpendicular to the view direction.
    /// Used for: standard particles, dust, simple effects.
    #[default]
    View = 0,

    /// Billboard stretches along velocity vector.
    ///
    /// The billboard's "up" axis aligns with velocity direction and
    /// stretches based on speed. Creates motion blur effect.
    /// Used for: smoke trails, sparks, rain, snow.
    Velocity = 1,

    /// Billboard rotates around a custom axis.
    ///
    /// The billboard is constrained to rotate around a specified axis
    /// while still trying to face the camera.
    /// Used for: flames (Y-axis), special effects.
    Custom = 2,
}

impl AlignmentMode {
    /// Create from u32 value.
    pub fn from_u32(value: u32) -> Self {
        match value {
            0 => AlignmentMode::View,
            1 => AlignmentMode::Velocity,
            2 => AlignmentMode::Custom,
            _ => AlignmentMode::View,
        }
    }
}

// ---------------------------------------------------------------------------
// BlendMode
// ---------------------------------------------------------------------------

/// Billboard blend mode.
///
/// Determines how particles blend with the existing framebuffer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum BlendMode {
    /// Standard alpha blending: dest = src * alpha + dest * (1 - alpha).
    ///
    /// Used for: smoke, dust, clouds, semi-transparent particles.
    #[default]
    Alpha,

    /// Additive blending: dest = src + dest.
    ///
    /// Used for: fire, sparks, glow, light effects.
    Additive,

    /// Pre-multiplied alpha: dest = src + dest * (1 - alpha).
    ///
    /// Used for: particles with pre-multiplied alpha textures.
    PremultipliedAlpha,
}

impl BlendMode {
    /// Get wgpu blend state for this mode.
    pub fn to_blend_state(&self) -> wgpu::BlendState {
        match self {
            BlendMode::Alpha => wgpu::BlendState::ALPHA_BLENDING,
            BlendMode::Additive => wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::SrcAlpha,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
            },
            BlendMode::PremultipliedAlpha => wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING,
        }
    }
}

// ---------------------------------------------------------------------------
// BillboardParams
// ---------------------------------------------------------------------------

/// GPU uniform buffer for billboard rendering parameters.
///
/// Matches the WGSL `BillboardParams` struct layout.
///
/// # Memory Layout (176 bytes, std140 compatible)
///
/// | Offset | Field           | Size     |
/// |--------|-----------------|----------|
/// | 0      | view_matrix     | 64 bytes |
/// | 64     | proj_matrix     | 64 bytes |
/// | 128    | camera_right    | 12 bytes |
/// | 140    | alignment_mode  | 4 bytes  |
/// | 144    | camera_up       | 12 bytes |
/// | 156    | velocity_stretch| 4 bytes  |
/// | 160    | custom_axis     | 12 bytes |
/// | 172    | time            | 4 bytes  |
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BillboardParams {
    /// View matrix (world to camera space).
    pub view_matrix: [[f32; 4]; 4],
    /// Projection matrix (camera to clip space).
    pub proj_matrix: [[f32; 4]; 4],
    /// Camera right vector in world space.
    pub camera_right: [f32; 3],
    /// Alignment mode: 0=VIEW, 1=VELOCITY, 2=CUSTOM.
    pub alignment_mode: u32,
    /// Camera up vector in world space.
    pub camera_up: [f32; 3],
    /// Velocity stretch factor (1.0 = no stretch).
    pub velocity_stretch: f32,
    /// Custom axis for CUSTOM alignment mode.
    pub custom_axis: [f32; 3],
    /// Time for animated effects.
    pub time: f32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<BillboardParams>() == BILLBOARD_PARAMS_SIZE);

impl BillboardParams {
    /// Create billboard params from camera matrices.
    ///
    /// # Arguments
    ///
    /// * `view` - View matrix (world to camera space).
    /// * `proj` - Projection matrix (camera to clip space).
    pub fn new(view: [[f32; 4]; 4], proj: [[f32; 4]; 4]) -> Self {
        // Extract camera axes from view matrix
        // View matrix columns are: right, up, forward (negated)
        let camera_right = [view[0][0], view[1][0], view[2][0]];
        let camera_up = [view[0][1], view[1][1], view[2][1]];

        Self {
            view_matrix: view,
            proj_matrix: proj,
            camera_right,
            alignment_mode: AlignmentMode::View as u32,
            camera_up,
            velocity_stretch: 0.0,
            custom_axis: [0.0, 1.0, 0.0],
            time: 0.0,
        }
    }

    /// Set alignment mode.
    pub fn with_alignment(mut self, mode: AlignmentMode) -> Self {
        self.alignment_mode = mode as u32;
        self
    }

    /// Set velocity stretch factor (for VELOCITY alignment).
    pub fn with_velocity_stretch(mut self, stretch: f32) -> Self {
        self.velocity_stretch = stretch;
        self
    }

    /// Set custom axis (for CUSTOM alignment).
    pub fn with_custom_axis(mut self, axis: [f32; 3]) -> Self {
        self.custom_axis = axis;
        self
    }

    /// Set time for animated effects.
    pub fn with_time(mut self, time: f32) -> Self {
        self.time = time;
        self
    }

    /// Extract camera position from view matrix.
    pub fn camera_position(&self) -> [f32; 3] {
        let v = &self.view_matrix;
        [
            -(v[0][0] * v[3][0] + v[0][1] * v[3][1] + v[0][2] * v[3][2]),
            -(v[1][0] * v[3][0] + v[1][1] * v[3][1] + v[1][2] * v[3][2]),
            -(v[2][0] * v[3][0] + v[2][1] * v[3][1] + v[2][2] * v[3][2]),
        ]
    }
}

impl Default for BillboardParams {
    fn default() -> Self {
        Self::new(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        )
    }
}

// ---------------------------------------------------------------------------
// BillboardResources
// ---------------------------------------------------------------------------

/// GPU resources for billboard particle rendering.
///
/// Contains all buffers and bind groups needed for the render pipeline.
pub struct BillboardResources {
    /// Uniform buffer for billboard parameters.
    pub params_buffer: wgpu::Buffer,
    /// Bind group for vertex shader (params, particles, sort_indices).
    pub vertex_bind_group: wgpu::BindGroup,
    /// Bind group for fragment shader (texture, sampler).
    pub fragment_bind_group: wgpu::BindGroup,
    /// Default white texture for untextured particles.
    pub default_texture: wgpu::Texture,
    /// Default texture view.
    pub default_texture_view: wgpu::TextureView,
    /// Sampler for particle textures.
    pub sampler: wgpu::Sampler,
    /// Maximum particle capacity.
    pub capacity: u32,
}

impl BillboardResources {
    /// Create billboard resources.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `capacity` - Maximum number of particles.
    /// * `vertex_bind_group_layout` - Layout from `BillboardPipeline`.
    /// * `fragment_bind_group_layout` - Layout from `BillboardPipeline`.
    /// * `particle_buffer` - Existing particle storage buffer.
    /// * `sort_indices_buffer` - Buffer with sorted particle indices.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        capacity: u32,
        vertex_bind_group_layout: &wgpu::BindGroupLayout,
        fragment_bind_group_layout: &wgpu::BindGroupLayout,
        particle_buffer: &wgpu::Buffer,
        sort_indices_buffer: &wgpu::Buffer,
    ) -> Self {
        // Create params uniform buffer
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("billboard_params"),
            size: BILLBOARD_PARAMS_SIZE as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create default white 1x1 texture
        let default_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("billboard_default_texture"),
            size: wgpu::Extent3d {
                width: DEFAULT_TEXTURE_SIZE,
                height: DEFAULT_TEXTURE_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Upload white pixel
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &default_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[255u8, 255, 255, 255],
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        let default_texture_view =
            default_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create sampler
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("billboard_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Create vertex bind group
        let vertex_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("billboard_vertex_bind_group"),
            layout: vertex_bind_group_layout,
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

        // Create fragment bind group
        let fragment_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("billboard_fragment_bind_group"),
            layout: fragment_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&default_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        Self {
            params_buffer,
            vertex_bind_group,
            fragment_bind_group,
            default_texture,
            default_texture_view,
            sampler,
            capacity,
        }
    }

    /// Update billboard parameters.
    pub fn update_params(&self, queue: &wgpu::Queue, params: &BillboardParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Create a new fragment bind group with a custom texture.
    pub fn create_texture_bind_group(
        &self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        texture_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("billboard_custom_texture_bind_group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }
}

// ---------------------------------------------------------------------------
// BillboardPipeline
// ---------------------------------------------------------------------------

/// GPU render pipeline for billboard particles.
///
/// Encapsulates shader modules, pipeline, and bind group layouts.
pub struct BillboardPipeline {
    /// Bind group layout for vertex shader.
    pub vertex_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group layout for fragment shader.
    pub fragment_bind_group_layout: wgpu::BindGroupLayout,
    /// Render pipeline for alpha blending.
    pub pipeline_alpha: wgpu::RenderPipeline,
    /// Render pipeline for additive blending.
    pub pipeline_additive: wgpu::RenderPipeline,
    /// Render pipeline for soft particles (radial falloff).
    pub pipeline_soft: wgpu::RenderPipeline,
}

impl BillboardPipeline {
    /// Create the billboard render pipeline.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `surface_format` - Output texture format.
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        // Create vertex bind group layout
        let vertex_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("billboard_vertex_bind_group_layout"),
                entries: &[
                    // binding 0: BillboardParams (uniform)
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
                    // binding 1: particles (storage, read)
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
                    // binding 2: sort_indices (storage, read)
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

        // Create fragment bind group layout
        let fragment_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("billboard_fragment_bind_group_layout"),
                entries: &[
                    // binding 0: particle_texture
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
                    // binding 1: particle_sampler
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
            label: Some("billboard_pipeline_layout"),
            bind_group_layouts: &[&vertex_bind_group_layout, &fragment_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Load shader modules
        let vertex_shader_source =
            include_str!("../../shaders/particles/billboard.vert.wgsl");
        let fragment_shader_source =
            include_str!("../../shaders/particles/billboard.frag.wgsl");

        let vertex_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("billboard_vertex_shader"),
            source: wgpu::ShaderSource::Wgsl(vertex_shader_source.into()),
        });

        let fragment_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("billboard_fragment_shader"),
            source: wgpu::ShaderSource::Wgsl(fragment_shader_source.into()),
        });

        // Create alpha blend pipeline
        let pipeline_alpha = Self::create_pipeline(
            device,
            &pipeline_layout,
            &vertex_module,
            &fragment_module,
            "fs_billboard",
            BlendMode::Alpha,
            surface_format,
            "billboard_alpha_pipeline",
        );

        // Create additive blend pipeline
        let pipeline_additive = Self::create_pipeline(
            device,
            &pipeline_layout,
            &vertex_module,
            &fragment_module,
            "fs_billboard_additive",
            BlendMode::Additive,
            surface_format,
            "billboard_additive_pipeline",
        );

        // Create soft particle pipeline
        let pipeline_soft = Self::create_pipeline(
            device,
            &pipeline_layout,
            &vertex_module,
            &fragment_module,
            "fs_billboard_soft",
            BlendMode::Alpha,
            surface_format,
            "billboard_soft_pipeline",
        );

        Self {
            vertex_bind_group_layout,
            fragment_bind_group_layout,
            pipeline_alpha,
            pipeline_additive,
            pipeline_soft,
        }
    }

    /// Create a render pipeline with the specified blend mode.
    fn create_pipeline(
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        vertex_module: &wgpu::ShaderModule,
        fragment_module: &wgpu::ShaderModule,
        fragment_entry: &str,
        blend_mode: BlendMode,
        surface_format: wgpu::TextureFormat,
        label: &str,
    ) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: vertex_module,
                entry_point: "vs_billboard",
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // Billboards are double-sided
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false, // Particles don't write depth
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: fragment_module,
                entry_point: fragment_entry,
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(blend_mode.to_blend_state()),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        })
    }

    /// Render particles using alpha blending.
    pub fn render_alpha<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        resources: &'a BillboardResources,
        particle_count: u32,
    ) {
        self.render_with_pipeline(render_pass, &self.pipeline_alpha, resources, particle_count);
    }

    /// Render particles using additive blending.
    pub fn render_additive<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        resources: &'a BillboardResources,
        particle_count: u32,
    ) {
        self.render_with_pipeline(render_pass, &self.pipeline_additive, resources, particle_count);
    }

    /// Render soft particles (radial falloff).
    pub fn render_soft<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        resources: &'a BillboardResources,
        particle_count: u32,
    ) {
        self.render_with_pipeline(render_pass, &self.pipeline_soft, resources, particle_count);
    }

    /// Render with a specific pipeline.
    fn render_with_pipeline<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        pipeline: &'a wgpu::RenderPipeline,
        resources: &'a BillboardResources,
        particle_count: u32,
    ) {
        if particle_count == 0 {
            return;
        }

        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, &resources.vertex_bind_group, &[]);
        render_pass.set_bind_group(1, &resources.fragment_bind_group, &[]);

        // Draw: 4 vertices per particle (triangle strip), instanced
        render_pass.draw(0..VERTICES_PER_PARTICLE, 0..particle_count);
    }
}

// ---------------------------------------------------------------------------
// CPU Reference Implementation
// ---------------------------------------------------------------------------

/// CPU reference implementation of billboard vertex generation.
///
/// Generates quad vertices for a single particle. Used for testing and
/// validation against GPU implementation.
pub fn cpu_generate_billboard_quad(
    particle: &Particle,
    camera_right: [f32; 3],
    camera_up: [f32; 3],
    alignment_mode: AlignmentMode,
    velocity_stretch: f32,
    camera_position: [f32; 3],
    custom_axis: [f32; 3],
) -> [[f32; 3]; 4] {
    // Quad offsets (BL, BR, TL, TR)
    let offsets: [[f32; 2]; 4] = [
        [-0.5, -0.5],
        [0.5, -0.5],
        [-0.5, 0.5],
        [0.5, 0.5],
    ];

    // Calculate billboard axes
    let (right, up) = match alignment_mode {
        AlignmentMode::View => {
            cpu_calculate_view_axes(camera_right, camera_up, particle.rotation)
        }
        AlignmentMode::Velocity => cpu_calculate_velocity_axes(
            particle.velocity,
            camera_position,
            particle.position,
            velocity_stretch,
            particle.rotation,
        ),
        AlignmentMode::Custom => cpu_calculate_custom_axes(
            custom_axis,
            camera_position,
            particle.position,
            particle.rotation,
        ),
    };

    // Generate vertices
    let mut vertices = [[0.0f32; 3]; 4];
    for (i, offset) in offsets.iter().enumerate() {
        vertices[i][0] =
            particle.position[0] + right[0] * offset[0] * particle.size + up[0] * offset[1] * particle.size;
        vertices[i][1] =
            particle.position[1] + right[1] * offset[0] * particle.size + up[1] * offset[1] * particle.size;
        vertices[i][2] =
            particle.position[2] + right[2] * offset[0] * particle.size + up[2] * offset[1] * particle.size;
    }

    vertices
}

/// Calculate view-aligned billboard axes (CPU reference).
fn cpu_calculate_view_axes(
    camera_right: [f32; 3],
    camera_up: [f32; 3],
    rotation: f32,
) -> ([f32; 3], [f32; 3]) {
    let c = rotation.cos();
    let s = rotation.sin();

    let right = [
        camera_right[0] * c + camera_up[0] * s,
        camera_right[1] * c + camera_up[1] * s,
        camera_right[2] * c + camera_up[2] * s,
    ];
    let up = [
        camera_up[0] * c - camera_right[0] * s,
        camera_up[1] * c - camera_right[1] * s,
        camera_up[2] * c - camera_right[2] * s,
    ];

    (right, up)
}

/// Calculate velocity-aligned billboard axes (CPU reference).
fn cpu_calculate_velocity_axes(
    velocity: [f32; 3],
    camera_position: [f32; 3],
    particle_position: [f32; 3],
    stretch_factor: f32,
    rotation: f32,
) -> ([f32; 3], [f32; 3]) {
    let speed = (velocity[0] * velocity[0] + velocity[1] * velocity[1] + velocity[2] * velocity[2]).sqrt();

    if speed < 0.001 {
        // Fallback to simple axes
        return ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    }

    let vel_dir = [
        velocity[0] / speed,
        velocity[1] / speed,
        velocity[2] / speed,
    ];

    // Get view direction
    let to_camera = normalize([
        camera_position[0] - particle_position[0],
        camera_position[1] - particle_position[1],
        camera_position[2] - particle_position[2],
    ]);

    // Right axis perpendicular to velocity and view
    let mut right = cross(vel_dir, to_camera);
    let right_len = (right[0] * right[0] + right[1] * right[1] + right[2] * right[2]).sqrt();

    if right_len < 0.001 {
        // Degenerate case
        right = cross(vel_dir, [0.0, 1.0, 0.0]);
    }
    right = normalize(right);

    // Up axis aligned with velocity, stretched
    let stretch = 1.0 + speed * stretch_factor;
    let up = [
        vel_dir[0] * stretch,
        vel_dir[1] * stretch,
        vel_dir[2] * stretch,
    ];

    // Apply rotation
    let c = rotation.cos();
    let s = rotation.sin();
    let rotated_right = [
        right[0] * c + (vel_dir[1] * right[2] - vel_dir[2] * right[1]) * s,
        right[1] * c + (vel_dir[2] * right[0] - vel_dir[0] * right[2]) * s,
        right[2] * c + (vel_dir[0] * right[1] - vel_dir[1] * right[0]) * s,
    ];

    (rotated_right, up)
}

/// Calculate custom-axis billboard axes (CPU reference).
fn cpu_calculate_custom_axes(
    custom_axis: [f32; 3],
    camera_position: [f32; 3],
    particle_position: [f32; 3],
    rotation: f32,
) -> ([f32; 3], [f32; 3]) {
    let axis = normalize(custom_axis);

    let to_camera = normalize([
        camera_position[0] - particle_position[0],
        camera_position[1] - particle_position[1],
        camera_position[2] - particle_position[2],
    ]);

    let mut right = cross(axis, to_camera);
    let right_len = (right[0] * right[0] + right[1] * right[1] + right[2] * right[2]).sqrt();

    if right_len < 0.001 {
        right = cross(axis, [1.0, 0.0, 0.0]);
    }
    right = normalize(right);

    // Apply rotation
    let c = rotation.cos();
    let s = rotation.sin();
    let rotated_right = [
        right[0] * c + (axis[1] * right[2] - axis[2] * right[1]) * s,
        right[1] * c + (axis[2] * right[0] - axis[0] * right[2]) * s,
        right[2] * c + (axis[0] * right[1] - axis[1] * right[0]) * s,
    ];

    (rotated_right, axis)
}

/// Normalize a 3D vector.
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-10 {
        return [0.0, 0.0, 0.0];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

/// Cross product of two 3D vectors.
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Calculate UV coordinates for a quad vertex.
pub fn cpu_calculate_uv(vertex_index: u32) -> [f32; 2] {
    match vertex_index % 4 {
        0 => [0.0, 1.0], // Bottom-left
        1 => [1.0, 1.0], // Bottom-right
        2 => [0.0, 0.0], // Top-left
        3 => [1.0, 0.0], // Top-right
        _ => [0.0, 0.0],
    }
}

/// Calculate lifetime alpha for a particle.
pub fn cpu_calculate_lifetime_alpha(age_ratio: f32) -> f32 {
    // Fade in during first 10% of lifetime
    let fade_in = smoothstep(0.0, 0.1, age_ratio);
    // Fade out during last 20% of lifetime
    let fade_out = 1.0 - smoothstep(0.8, 1.0, age_ratio);
    fade_in * fade_out
}

/// Smoothstep function.
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    // ── BillboardParams ─────────────────────────────────────────────────

    #[test]
    fn test_billboard_params_size() {
        assert_eq!(mem::size_of::<BillboardParams>(), BILLBOARD_PARAMS_SIZE);
    }

    #[test]
    fn test_billboard_params_default() {
        let params = BillboardParams::default();
        assert_eq!(params.alignment_mode, AlignmentMode::View as u32);
        assert_eq!(params.velocity_stretch, 0.0);
    }

    #[test]
    fn test_billboard_params_with_alignment() {
        let params = BillboardParams::default().with_alignment(AlignmentMode::Velocity);
        assert_eq!(params.alignment_mode, AlignmentMode::Velocity as u32);
    }

    #[test]
    fn test_billboard_params_with_stretch() {
        let params = BillboardParams::default().with_velocity_stretch(0.5);
        assert!((params.velocity_stretch - 0.5).abs() < f32::EPSILON);
    }

    // ── AlignmentMode ───────────────────────────────────────────────────

    #[test]
    fn test_alignment_mode_values() {
        assert_eq!(AlignmentMode::View as u32, 0);
        assert_eq!(AlignmentMode::Velocity as u32, 1);
        assert_eq!(AlignmentMode::Custom as u32, 2);
    }

    #[test]
    fn test_alignment_mode_from_u32() {
        assert_eq!(AlignmentMode::from_u32(0), AlignmentMode::View);
        assert_eq!(AlignmentMode::from_u32(1), AlignmentMode::Velocity);
        assert_eq!(AlignmentMode::from_u32(2), AlignmentMode::Custom);
        assert_eq!(AlignmentMode::from_u32(99), AlignmentMode::View); // Fallback
    }

    // ── BlendMode ───────────────────────────────────────────────────────

    #[test]
    fn test_blend_mode_default() {
        assert_eq!(BlendMode::default(), BlendMode::Alpha);
    }

    // ── CPU Billboard Generation ────────────────────────────────────────

    #[test]
    fn test_view_aligned_billboard_faces_camera() {
        let particle = Particle {
            position: [0.0, 0.0, 0.0],
            age: 0.0,
            velocity: [0.0, 0.0, 0.0],
            lifetime: 1.0,
            color: [1.0, 1.0, 1.0, 1.0],
            size: 1.0,
            rotation: 0.0,
            rotation_speed: 0.0,
            flags: 1,
        };

        let camera_right = [1.0, 0.0, 0.0];
        let camera_up = [0.0, 1.0, 0.0];
        let camera_position = [0.0, 0.0, 5.0];

        let vertices = cpu_generate_billboard_quad(
            &particle,
            camera_right,
            camera_up,
            AlignmentMode::View,
            0.0,
            camera_position,
            [0.0, 1.0, 0.0],
        );

        // Billboard should be in XY plane (perpendicular to camera Z)
        for vertex in &vertices {
            assert!((vertex[2] - 0.0).abs() < 1e-5, "vertex Z should be 0");
        }
    }

    #[test]
    fn test_velocity_aligned_stretches_along_velocity() {
        let particle = Particle {
            position: [0.0, 0.0, 0.0],
            age: 0.0,
            velocity: [0.0, 5.0, 0.0], // Moving up
            lifetime: 1.0,
            color: [1.0, 1.0, 1.0, 1.0],
            size: 1.0,
            rotation: 0.0,
            rotation_speed: 0.0,
            flags: 1,
        };

        let camera_position = [0.0, 0.0, 5.0];
        let stretch_factor = 0.5;

        let vertices = cpu_generate_billboard_quad(
            &particle,
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            AlignmentMode::Velocity,
            stretch_factor,
            camera_position,
            [0.0, 1.0, 0.0],
        );

        // Top vertices should be further from center than bottom
        // due to velocity stretch
        let top_y_avg = (vertices[2][1] + vertices[3][1]) / 2.0;
        let bottom_y_avg = (vertices[0][1] + vertices[1][1]) / 2.0;
        let height = top_y_avg - bottom_y_avg;

        // With velocity stretch, height should be > 1.0 (base size)
        assert!(height > 1.0, "velocity stretch should increase height");
    }

    #[test]
    fn test_rotation_applied_correctly() {
        let particle_no_rotation = Particle {
            position: [0.0, 0.0, 0.0],
            age: 0.0,
            velocity: [0.0, 0.0, 0.0],
            lifetime: 1.0,
            color: [1.0, 1.0, 1.0, 1.0],
            size: 1.0,
            rotation: 0.0,
            rotation_speed: 0.0,
            flags: 1,
        };

        let particle_rotated = Particle {
            rotation: PI / 4.0, // 45 degrees
            ..particle_no_rotation
        };

        let camera_right = [1.0, 0.0, 0.0];
        let camera_up = [0.0, 1.0, 0.0];
        let camera_position = [0.0, 0.0, 5.0];

        let vertices_no_rot = cpu_generate_billboard_quad(
            &particle_no_rotation,
            camera_right,
            camera_up,
            AlignmentMode::View,
            0.0,
            camera_position,
            [0.0, 1.0, 0.0],
        );

        let vertices_rotated = cpu_generate_billboard_quad(
            &particle_rotated,
            camera_right,
            camera_up,
            AlignmentMode::View,
            0.0,
            camera_position,
            [0.0, 1.0, 0.0],
        );

        // Vertices should be different after rotation
        let diff = (vertices_no_rot[0][0] - vertices_rotated[0][0]).abs()
            + (vertices_no_rot[0][1] - vertices_rotated[0][1]).abs();
        assert!(diff > 0.1, "rotation should change vertex positions");
    }

    // ── UV Coordinates ──────────────────────────────────────────────────

    #[test]
    fn test_uv_coordinates_correct() {
        assert_eq!(cpu_calculate_uv(0), [0.0, 1.0]); // Bottom-left
        assert_eq!(cpu_calculate_uv(1), [1.0, 1.0]); // Bottom-right
        assert_eq!(cpu_calculate_uv(2), [0.0, 0.0]); // Top-left
        assert_eq!(cpu_calculate_uv(3), [1.0, 0.0]); // Top-right
    }

    #[test]
    fn test_uv_coordinates_wrap() {
        // Should wrap around for indices >= 4
        assert_eq!(cpu_calculate_uv(4), [0.0, 1.0]);
        assert_eq!(cpu_calculate_uv(5), [1.0, 1.0]);
    }

    // ── Color and Lifetime ──────────────────────────────────────────────

    #[test]
    fn test_lifetime_alpha_fade_in() {
        // At age 0, should be fading in
        let alpha_0 = cpu_calculate_lifetime_alpha(0.0);
        let alpha_005 = cpu_calculate_lifetime_alpha(0.05);
        let alpha_01 = cpu_calculate_lifetime_alpha(0.1);

        assert!(alpha_0 < alpha_005);
        assert!(alpha_005 < alpha_01);
    }

    #[test]
    fn test_lifetime_alpha_peak() {
        // In middle of lifetime, alpha should be ~1.0
        let alpha_mid = cpu_calculate_lifetime_alpha(0.5);
        assert!((alpha_mid - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_lifetime_alpha_fade_out() {
        // At end of life, should be fading out
        let alpha_08 = cpu_calculate_lifetime_alpha(0.8);
        let alpha_09 = cpu_calculate_lifetime_alpha(0.9);
        let alpha_10 = cpu_calculate_lifetime_alpha(1.0);

        assert!(alpha_08 > alpha_09);
        assert!(alpha_09 > alpha_10);
        assert!(alpha_10 < 0.01); // Nearly zero at death
    }

    // ── Size Scaling ────────────────────────────────────────────────────

    #[test]
    fn test_size_scaling() {
        let particle_small = Particle {
            position: [0.0, 0.0, 0.0],
            age: 0.0,
            velocity: [0.0, 0.0, 0.0],
            lifetime: 1.0,
            color: [1.0, 1.0, 1.0, 1.0],
            size: 1.0,
            rotation: 0.0,
            rotation_speed: 0.0,
            flags: 1,
        };

        let particle_large = Particle {
            size: 2.0,
            ..particle_small
        };

        let vertices_small = cpu_generate_billboard_quad(
            &particle_small,
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            AlignmentMode::View,
            0.0,
            [0.0, 0.0, 5.0],
            [0.0, 1.0, 0.0],
        );

        let vertices_large = cpu_generate_billboard_quad(
            &particle_large,
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            AlignmentMode::View,
            0.0,
            [0.0, 0.0, 5.0],
            [0.0, 1.0, 0.0],
        );

        // Large particle should have vertices twice as far from center
        let small_dist = (vertices_small[3][0] - vertices_small[0][0]).abs();
        let large_dist = (vertices_large[3][0] - vertices_large[0][0]).abs();

        assert!(
            (large_dist - small_dist * 2.0).abs() < 1e-5,
            "size 2x should produce 2x larger quad"
        );
    }

    // ── Helper Functions ────────────────────────────────────────────────

    #[test]
    fn test_normalize() {
        let v = normalize([3.0, 4.0, 0.0]);
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cross_product() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        let c = cross(a, b);
        assert!((c[0] - 0.0).abs() < 1e-5);
        assert!((c[1] - 0.0).abs() < 1e-5);
        assert!((c[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_smoothstep_boundaries() {
        assert!((smoothstep(0.0, 1.0, 0.0) - 0.0).abs() < 1e-5);
        assert!((smoothstep(0.0, 1.0, 1.0) - 1.0).abs() < 1e-5);
        assert!((smoothstep(0.0, 1.0, 0.5) - 0.5).abs() < 1e-5);
    }
}
