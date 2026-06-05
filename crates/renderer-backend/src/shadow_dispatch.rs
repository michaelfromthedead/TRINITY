//! Shadow map render dispatch for point lights (cube) and spot lights.
//!
//! This module provides rendering infrastructure for omnidirectional point light
//! shadows using cube shadow maps, and directional spot light shadows using
//! standard 2D shadow maps.
//!
//! # Architecture
//!
//! - [`ShadowRenderPass`]: Common shadow rendering pipeline and bind group layout.
//! - [`CubeShadowDispatch`]: Renders point light shadows to 6 cube faces.
//! - [`SpotShadowDispatch`]: Renders spot light shadows to a single 2D texture.
//!
//! # Cube Shadow Rendering
//!
//! Point lights require omnidirectional shadow maps. We render the scene 6 times,
//! once for each cube face, using different view matrices:
//!
//! ```text
//! Face | Look Direction | Up Vector
//! -----|----------------|----------
//!  +X  | (+1, 0, 0)     | (0, -1, 0)
//!  -X  | (-1, 0, 0)     | (0, -1, 0)
//!  +Y  | (0, +1, 0)     | (0, 0, +1)
//!  -Y  | (0, -1, 0)     | (0, 0, -1)
//!  +Z  | (0, 0, +1)     | (0, -1, 0)
//!  -Z  | (0, 0, -1)     | (0, -1, 0)
//! ```
//!
//! # Usage
//!
//! ```ignore
//! let cube_dispatch = CubeShadowDispatch::new(device);
//! let spot_dispatch = SpotShadowDispatch::new(device);
//!
//! // Render point light shadow
//! cube_dispatch.render_point_light_shadow(
//!     &mut encoder,
//!     light_position,
//!     &cube_array,
//!     light_index,
//!     &scene_meshes,
//! );
//!
//! // Render spot light shadow
//! spot_dispatch.render_spot_light_shadow(
//!     &mut encoder,
//!     light_view_proj,
//!     &shadow_map_view,
//!     &scene_meshes,
//! );
//! ```

use crate::csm::RenderMesh;
use crate::rhi_commands::RhiCommandEncoder;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of faces in a cube shadow map.
pub const CUBE_FACES: u32 = 6;

/// Default shadow map resolution for point/spot lights.
pub const DEFAULT_SHADOW_RESOLUTION: u32 = 1024;

/// Default near plane for shadow projection.
pub const DEFAULT_SHADOW_NEAR: f32 = 0.1;

/// Default far plane for shadow projection (point light radius).
pub const DEFAULT_SHADOW_FAR: f32 = 100.0;

// ---------------------------------------------------------------------------
// Cube Face Direction Data
// ---------------------------------------------------------------------------

/// Cube face directions for omnidirectional shadow mapping.
///
/// Each face has a look direction and an up vector to construct
/// a proper view matrix from the light's position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CubeFaceDirection {
    /// The direction the camera looks for this face.
    pub look: [f32; 3],
    /// The up vector for the camera on this face.
    pub up: [f32; 3],
}

/// Get the cube face directions for building view matrices.
///
/// Returns an array of 6 face directions in the order:
/// +X, -X, +Y, -Y, +Z, -Z
pub const fn cube_face_directions() -> [CubeFaceDirection; 6] {
    [
        // +X face
        CubeFaceDirection {
            look: [1.0, 0.0, 0.0],
            up: [0.0, -1.0, 0.0],
        },
        // -X face
        CubeFaceDirection {
            look: [-1.0, 0.0, 0.0],
            up: [0.0, -1.0, 0.0],
        },
        // +Y face
        CubeFaceDirection {
            look: [0.0, 1.0, 0.0],
            up: [0.0, 0.0, 1.0],
        },
        // -Y face
        CubeFaceDirection {
            look: [0.0, -1.0, 0.0],
            up: [0.0, 0.0, -1.0],
        },
        // +Z face
        CubeFaceDirection {
            look: [0.0, 0.0, 1.0],
            up: [0.0, -1.0, 0.0],
        },
        // -Z face
        CubeFaceDirection {
            look: [0.0, 0.0, -1.0],
            up: [0.0, -1.0, 0.0],
        },
    ]
}

// ---------------------------------------------------------------------------
// Shadow View
// ---------------------------------------------------------------------------

/// View data for a single shadow render pass.
///
/// Contains the view-projection matrix for shadow depth rendering.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowView {
    /// Light view-projection matrix (column-major).
    pub light_view_proj: [[f32; 4]; 4],
}

impl Default for ShadowView {
    fn default() -> Self {
        Self {
            light_view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// Cube Shadow Array
// ---------------------------------------------------------------------------

/// A cube shadow map array for multiple point lights.
///
/// Each point light gets 6 faces in the array. The array is organized as:
/// - Layer 0-5: Light 0 faces (+X, -X, +Y, -Y, +Z, -Z)
/// - Layer 6-11: Light 1 faces
/// - etc.
pub struct CubeShadowArray {
    /// The cube array texture.
    pub texture: wgpu::Texture,
    /// View for binding to shaders (all layers).
    pub array_view: wgpu::TextureView,
    /// Per-face views for rendering (6 per light).
    pub face_views: Vec<wgpu::TextureView>,
    /// Resolution of each cube face.
    pub resolution: u32,
    /// Maximum number of point lights supported.
    pub max_lights: u32,
}

impl CubeShadowArray {
    /// Create a new cube shadow array.
    ///
    /// # Parameters
    ///
    /// * `device` - The wgpu device.
    /// * `resolution` - Resolution of each cube face (e.g., 512, 1024).
    /// * `max_lights` - Maximum number of point lights to support.
    ///
    /// # Returns
    ///
    /// A cube shadow array with `max_lights * 6` layers.
    pub fn new(device: &wgpu::Device, resolution: u32, max_lights: u32) -> Self {
        let total_layers = max_lights * CUBE_FACES;

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Cube Shadow Array"),
            size: wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: total_layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        // Create array view for shader binding
        let array_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Cube Shadow Array View"),
            format: Some(wgpu::TextureFormat::Depth32Float),
            dimension: Some(wgpu::TextureViewDimension::CubeArray),
            aspect: wgpu::TextureAspect::DepthOnly,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(total_layers),
        });

        // Create per-face views for rendering
        let mut face_views = Vec::with_capacity(total_layers as usize);
        for layer in 0..total_layers {
            let view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("Cube Shadow Face {}", layer)),
                format: Some(wgpu::TextureFormat::Depth32Float),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::DepthOnly,
                base_mip_level: 0,
                mip_level_count: Some(1),
                base_array_layer: layer,
                array_layer_count: Some(1),
            });
            face_views.push(view);
        }

        Self {
            texture,
            array_view,
            face_views,
            resolution,
            max_lights,
        }
    }

    /// Get the texture view for a specific light's face.
    ///
    /// # Parameters
    ///
    /// * `light_index` - Index of the point light (0 to max_lights-1).
    /// * `face_index` - Cube face index (0-5: +X, -X, +Y, -Y, +Z, -Z).
    ///
    /// # Returns
    ///
    /// The texture view for rendering to that face.
    pub fn get_face_view(&self, light_index: u32, face_index: u32) -> &wgpu::TextureView {
        debug_assert!(light_index < self.max_lights);
        debug_assert!(face_index < CUBE_FACES);
        let layer = light_index * CUBE_FACES + face_index;
        &self.face_views[layer as usize]
    }
}

// ---------------------------------------------------------------------------
// Shadow Render Pass Configuration
// ---------------------------------------------------------------------------

/// Configuration for shadow rendering.
#[derive(Debug, Clone, Copy)]
pub struct ShadowRenderConfig {
    /// Depth bias constant factor.
    pub depth_bias: i32,
    /// Depth bias slope factor.
    pub depth_bias_slope_scale: f32,
    /// Depth bias clamp value.
    pub depth_bias_clamp: f32,
}

impl Default for ShadowRenderConfig {
    fn default() -> Self {
        Self {
            depth_bias: 2,
            depth_bias_slope_scale: 2.0,
            depth_bias_clamp: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Shadow Render Pass
// ---------------------------------------------------------------------------

/// Common shadow rendering pipeline and bind group layout.
///
/// Provides the shared infrastructure for both cube and spot shadow rendering.
pub struct ShadowRenderPass {
    /// The render pipeline for depth-only shadow rendering.
    pub pipeline: wgpu::RenderPipeline,
    /// Bind group layout for shadow view uniforms (group 0).
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl ShadowRenderPass {
    /// Create a new shadow render pass.
    ///
    /// # Parameters
    ///
    /// * `device` - The wgpu device.
    ///
    /// # Returns
    ///
    /// A shadow render pass with pipeline and bind group layout.
    pub fn new(device: &wgpu::Device) -> Self {
        Self::with_config(device, ShadowRenderConfig::default())
    }

    /// Create a new shadow render pass with custom configuration.
    ///
    /// # Parameters
    ///
    /// * `device` - The wgpu device.
    /// * `config` - Shadow render configuration.
    ///
    /// # Returns
    ///
    /// A shadow render pass with the specified configuration.
    pub fn with_config(device: &wgpu::Device, config: ShadowRenderConfig) -> Self {
        // Create bind group layout for shadow view uniforms
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shadow View Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(
                        std::num::NonZeroU64::new(std::mem::size_of::<ShadowView>() as u64)
                            .unwrap(),
                    ),
                },
                count: None,
            }],
        });

        // Create model bind group layout (group 1)
        let model_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Shadow Model Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shadow Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout, &model_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create shadow vertex shader module
        let vertex_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shadow Vertex Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/shadow.vert.wgsl").into()),
        });

        // Create shadow fragment shader module
        let fragment_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shadow Fragment Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/shadow.frag.wgsl").into()),
        });

        // Create render pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Shadow Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vertex_shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 12, // 3 * f32 for position
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x3,
                    }],
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: config.depth_bias,
                    slope_scale: config.depth_bias_slope_scale,
                    clamp: config.depth_bias_clamp,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &fragment_shader,
                entry_point: "fs_main",
                targets: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
        }
    }

    /// Create a bind group for a shadow view.
    ///
    /// # Parameters
    ///
    /// * `device` - The wgpu device.
    /// * `view_buffer` - Buffer containing the ShadowView uniform.
    ///
    /// # Returns
    ///
    /// A bind group for the shadow view.
    pub fn create_view_bind_group(
        &self,
        device: &wgpu::Device,
        view_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow View Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: view_buffer.as_entire_binding(),
            }],
        })
    }
}

// ---------------------------------------------------------------------------
// Cube Shadow Dispatch
// ---------------------------------------------------------------------------

/// Dispatch for rendering point light cube shadow maps.
///
/// Renders the scene 6 times per light, once for each cube face.
pub struct CubeShadowDispatch {
    /// The shared shadow render pass.
    pub pass: ShadowRenderPass,
}

impl CubeShadowDispatch {
    /// Create a new cube shadow dispatch.
    ///
    /// # Parameters
    ///
    /// * `device` - The wgpu device.
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            pass: ShadowRenderPass::new(device),
        }
    }

    /// Create a new cube shadow dispatch with custom configuration.
    ///
    /// # Parameters
    ///
    /// * `device` - The wgpu device.
    /// * `config` - Shadow render configuration.
    pub fn with_config(device: &wgpu::Device, config: ShadowRenderConfig) -> Self {
        Self {
            pass: ShadowRenderPass::with_config(device, config),
        }
    }

    /// Render point light shadow into a cube shadow array.
    ///
    /// Renders the scene 6 times, once for each cube face, from the
    /// light's position.
    ///
    /// # Parameters
    ///
    /// * `encoder` - Command encoder for recording render commands.
    /// * `light_position` - World position of the point light.
    /// * `cube_array` - Cube shadow array to render into.
    /// * `light_index` - Index of the light in the array (0 to max_lights-1).
    /// * `scene_meshes` - Meshes to render into the shadow map.
    /// * `face_bind_groups` - Per-face bind groups (6 total) with view-proj matrices.
    pub fn render_point_light_shadow<'a>(
        &self,
        encoder: &'a mut RhiCommandEncoder,
        _light_position: [f32; 3],
        cube_array: &CubeShadowArray,
        light_index: u32,
        scene_meshes: &[RenderMesh<'_>],
        face_bind_groups: &[&wgpu::BindGroup; 6],
    ) {
        debug_assert!(light_index < cube_array.max_lights);

        // Render each of the 6 cube faces
        for face_index in 0..CUBE_FACES {
            let face_view = cube_array.get_face_view(light_index, face_index);
            let bind_group = face_bind_groups[face_index as usize];

            // Begin depth-only render pass for this face
            let mut render_pass =
                encoder
                    .inner_mut()
                    .begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some(&format!(
                            "Cube Shadow Light {} Face {}",
                            light_index, face_index
                        )),
                        color_attachments: &[],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: face_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });

            // Set viewport to cover the entire face
            render_pass.set_viewport(
                0.0,
                0.0,
                cube_array.resolution as f32,
                cube_array.resolution as f32,
                0.0,
                1.0,
            );

            // Set scissor rect
            render_pass.set_scissor_rect(0, 0, cube_array.resolution, cube_array.resolution);

            // Bind shadow pipeline
            render_pass.set_pipeline(&self.pass.pipeline);

            // Bind face-specific view-projection matrix
            render_pass.set_bind_group(0, bind_group, &[]);

            // Render all meshes
            for mesh in scene_meshes {
                // Bind per-object data if present
                if let Some(obj_bind_group) = mesh.bind_group {
                    render_pass.set_bind_group(1, obj_bind_group, &[]);
                }

                // Bind vertex buffer
                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));

                // Draw indexed or non-indexed
                if let Some(index_buffer) = mesh.index_buffer {
                    render_pass.set_index_buffer(index_buffer.slice(..), mesh.index_format);
                    render_pass.draw_indexed(
                        mesh.first_index..mesh.first_index + mesh.index_count,
                        0,
                        0..mesh.instance_count,
                    );
                } else {
                    render_pass.draw(
                        mesh.first_index..mesh.first_index + mesh.index_count,
                        0..mesh.instance_count,
                    );
                }
            }

            // Render pass ends when dropped
        }
    }

    /// Get the cube face directions for building view matrices.
    pub fn get_face_directions(&self) -> [CubeFaceDirection; 6] {
        cube_face_directions()
    }
}

// ---------------------------------------------------------------------------
// Spot Shadow Dispatch
// ---------------------------------------------------------------------------

/// Dispatch for rendering spot light shadow maps.
///
/// Renders the scene once per light with a perspective projection.
pub struct SpotShadowDispatch {
    /// The shared shadow render pass.
    pub pass: ShadowRenderPass,
}

impl SpotShadowDispatch {
    /// Create a new spot shadow dispatch.
    ///
    /// # Parameters
    ///
    /// * `device` - The wgpu device.
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            pass: ShadowRenderPass::new(device),
        }
    }

    /// Create a new spot shadow dispatch with custom configuration.
    ///
    /// # Parameters
    ///
    /// * `device` - The wgpu device.
    /// * `config` - Shadow render configuration.
    pub fn with_config(device: &wgpu::Device, config: ShadowRenderConfig) -> Self {
        Self {
            pass: ShadowRenderPass::with_config(device, config),
        }
    }

    /// Render spot light shadow into a shadow map.
    ///
    /// Renders the scene once from the light's perspective.
    ///
    /// # Parameters
    ///
    /// * `encoder` - Command encoder for recording render commands.
    /// * `light_view_proj` - Light's view-projection matrix.
    /// * `shadow_map` - Shadow map texture view to render into.
    /// * `shadow_map_size` - Size of the shadow map in pixels.
    /// * `scene_meshes` - Meshes to render into the shadow map.
    /// * `view_bind_group` - Bind group containing the view-proj matrix.
    pub fn render_spot_light_shadow<'a>(
        &self,
        encoder: &'a mut RhiCommandEncoder,
        _light_view_proj: [[f32; 4]; 4],
        shadow_map: &wgpu::TextureView,
        shadow_map_size: u32,
        scene_meshes: &[RenderMesh<'_>],
        view_bind_group: &wgpu::BindGroup,
    ) {
        // Begin depth-only render pass
        let mut render_pass = encoder
            .inner_mut()
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Spot Shadow"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: shadow_map,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

        // Set viewport
        render_pass.set_viewport(
            0.0,
            0.0,
            shadow_map_size as f32,
            shadow_map_size as f32,
            0.0,
            1.0,
        );

        // Set scissor rect
        render_pass.set_scissor_rect(0, 0, shadow_map_size, shadow_map_size);

        // Bind shadow pipeline
        render_pass.set_pipeline(&self.pass.pipeline);

        // Bind view-projection matrix
        render_pass.set_bind_group(0, view_bind_group, &[]);

        // Render all meshes
        for mesh in scene_meshes {
            // Bind per-object data if present
            if let Some(obj_bind_group) = mesh.bind_group {
                render_pass.set_bind_group(1, obj_bind_group, &[]);
            }

            // Bind vertex buffer
            render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));

            // Draw indexed or non-indexed
            if let Some(index_buffer) = mesh.index_buffer {
                render_pass.set_index_buffer(index_buffer.slice(..), mesh.index_format);
                render_pass.draw_indexed(
                    mesh.first_index..mesh.first_index + mesh.index_count,
                    0,
                    0..mesh.instance_count,
                );
            } else {
                render_pass.draw(
                    mesh.first_index..mesh.first_index + mesh.index_count,
                    0..mesh.instance_count,
                );
            }
        }

        // Render pass ends when dropped
    }
}

// ---------------------------------------------------------------------------
// Helper: Create Spot Shadow Map
// ---------------------------------------------------------------------------

/// Create a 2D depth texture for spot light shadow mapping.
///
/// # Parameters
///
/// * `device` - The wgpu device.
/// * `resolution` - Shadow map resolution (e.g., 1024, 2048).
///
/// # Returns
///
/// A depth texture suitable for spot light shadows.
pub fn create_spot_shadow_map(device: &wgpu::Device, resolution: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Spot Shadow Map"),
        size: wgpu::Extent3d {
            width: resolution,
            height: resolution,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

/// Create an array of 2D depth textures for multiple spot lights.
///
/// # Parameters
///
/// * `device` - The wgpu device.
/// * `resolution` - Shadow map resolution (e.g., 1024, 2048).
/// * `count` - Number of spot lights to support.
///
/// # Returns
///
/// An array texture with `count` layers.
pub fn create_spot_shadow_array(
    device: &wgpu::Device,
    resolution: u32,
    count: u32,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Spot Shadow Array"),
        size: wgpu::Extent3d {
            width: resolution,
            height: resolution,
            depth_or_array_layers: count,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Constants ------------------------------------------------------------

    #[test]
    fn test_cube_faces_constant() {
        assert_eq!(CUBE_FACES, 6);
    }

    #[test]
    fn test_default_shadow_resolution() {
        assert_eq!(DEFAULT_SHADOW_RESOLUTION, 1024);
    }

    // -- Cube Face Directions -------------------------------------------------

    #[test]
    fn test_cube_face_directions_count() {
        let dirs = cube_face_directions();
        assert_eq!(dirs.len(), 6);
    }

    #[test]
    fn test_cube_face_directions_orthogonal() {
        let dirs = cube_face_directions();

        // Each look direction should be a unit vector along one axis
        let expected_looks = [
            [1.0, 0.0, 0.0],   // +X
            [-1.0, 0.0, 0.0],  // -X
            [0.0, 1.0, 0.0],   // +Y
            [0.0, -1.0, 0.0],  // -Y
            [0.0, 0.0, 1.0],   // +Z
            [0.0, 0.0, -1.0],  // -Z
        ];

        for (i, dir) in dirs.iter().enumerate() {
            assert_eq!(dir.look, expected_looks[i], "Face {} look mismatch", i);
        }
    }

    #[test]
    fn test_cube_face_up_vectors_perpendicular_to_look() {
        let dirs = cube_face_directions();

        for (i, dir) in dirs.iter().enumerate() {
            // Dot product of perpendicular vectors should be 0
            let dot = dir.look[0] * dir.up[0] + dir.look[1] * dir.up[1] + dir.look[2] * dir.up[2];
            assert!(
                dot.abs() < 1e-6,
                "Face {} up not perpendicular to look: dot = {}",
                i,
                dot
            );
        }
    }

    // -- ShadowView -----------------------------------------------------------

    #[test]
    fn test_shadow_view_size() {
        // ShadowView should be 64 bytes (4x4 matrix of f32)
        assert_eq!(std::mem::size_of::<ShadowView>(), 64);
    }

    #[test]
    fn test_shadow_view_default_is_identity() {
        let view = ShadowView::default();

        // Check identity matrix
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_eq!(
                    view.light_view_proj[i][j], expected,
                    "Matrix[{}][{}] should be {}",
                    i, j, expected
                );
            }
        }
    }

    // -- ShadowRenderConfig ---------------------------------------------------

    #[test]
    fn test_shadow_render_config_default() {
        let config = ShadowRenderConfig::default();

        assert_eq!(config.depth_bias, 2);
        assert_eq!(config.depth_bias_slope_scale, 2.0);
        assert_eq!(config.depth_bias_clamp, 0.0);
    }

    // -- CubeFaceDirection ----------------------------------------------------

    #[test]
    fn test_cube_face_direction_equality() {
        let dir1 = CubeFaceDirection {
            look: [1.0, 0.0, 0.0],
            up: [0.0, -1.0, 0.0],
        };
        let dir2 = CubeFaceDirection {
            look: [1.0, 0.0, 0.0],
            up: [0.0, -1.0, 0.0],
        };
        let dir3 = CubeFaceDirection {
            look: [-1.0, 0.0, 0.0],
            up: [0.0, -1.0, 0.0],
        };

        assert_eq!(dir1, dir2);
        assert_ne!(dir1, dir3);
    }

    #[test]
    fn test_cube_face_direction_copy() {
        let dir1 = CubeFaceDirection {
            look: [1.0, 2.0, 3.0],
            up: [4.0, 5.0, 6.0],
        };
        let dir2 = dir1; // Copy

        assert_eq!(dir1.look, dir2.look);
        assert_eq!(dir1.up, dir2.up);
    }

    // -- Integration tests (require GPU) --------------------------------------

    // Note: The following tests require a GPU and are marked to skip in CI
    // environments without GPU support.

    #[cfg(feature = "gpu_tests")]
    mod gpu_tests {
        use super::*;
        use crate::rhi_device::{create_instance, request_device, FeatureFlags, QualityTier};

        fn test_device() -> Option<crate::rhi_device::RhiDevice> {
            let instance = create_instance();
            let adapter = pollster::block_on(
                instance
                    .inner()
                    .request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::HighPerformance,
                        compatible_surface: None,
                        force_fallback_adapter: false,
                    }),
            )?;
            Some(request_device(
                &adapter,
                FeatureFlags::empty(),
                QualityTier::Low,
            ))
        }

        #[test]
        fn test_cube_shadow_array_creation() {
            let Some(device) = test_device() else {
                return;
            };

            let array = CubeShadowArray::new(device.device(), 512, 4);

            assert_eq!(array.resolution, 512);
            assert_eq!(array.max_lights, 4);
            assert_eq!(array.face_views.len(), 24); // 4 lights * 6 faces
        }

        #[test]
        fn test_shadow_render_pass_creation() {
            let Some(device) = test_device() else {
                return;
            };

            let _pass = ShadowRenderPass::new(device.device());
            // Pass should be created without panic
        }

        #[test]
        fn test_cube_shadow_dispatch_creation() {
            let Some(device) = test_device() else {
                return;
            };

            let dispatch = CubeShadowDispatch::new(device.device());
            let dirs = dispatch.get_face_directions();
            assert_eq!(dirs.len(), 6);
        }

        #[test]
        fn test_spot_shadow_dispatch_creation() {
            let Some(device) = test_device() else {
                return;
            };

            let _dispatch = SpotShadowDispatch::new(device.device());
            // Dispatch should be created without panic
        }
    }
}
