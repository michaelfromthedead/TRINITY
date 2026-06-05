//! Cascaded Shadow Map (CSM) rendering module.
//!
//! Implements multi-viewport cascade shadow rendering for directional lights.
//! Supports 2-4 cascades arranged in a 2x2 grid layout within a shadow atlas.
//!
//! # Atlas Layout
//!
//! The shadow atlas is divided into a 2x2 grid:
//!
//! ```text
//! +----------+----------+
//! | Cascade0 | Cascade1 |
//! |  (0,0)   |  (1,0)   |
//! +----------+----------+
//! | Cascade2 | Cascade3 |
//! |  (0,1)   |  (1,1)   |
//! +----------+----------+
//! ```
//!
//! Each cascade gets a quarter of the atlas. For a 4096x4096 atlas,
//! each cascade viewport is 2048x2048.
//!
//! # Usage
//!
//! ```ignore
//! let config = CascadeRenderConfig {
//!     cascade_count: 4,
//!     atlas_size: 4096,
//!     cascade_sizes: [2048, 2048, 2048, 2048],
//! };
//!
//! render_shadow_cascades(
//!     encoder,
//!     &config,
//!     &shadow_atlas_view,
//!     &cascade_views,
//!     &scene_meshes,
//!     &shadow_pipeline,
//!     &shadow_bind_group,
//! );
//! ```

use crate::rhi_commands::RhiCommandEncoder;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for cascaded shadow map rendering.
///
/// Defines the number of cascades, atlas dimensions, and per-cascade
/// resolutions for shadow map rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CascadeRenderConfig {
    /// Number of cascades (2-4). Default: 4.
    pub cascade_count: u32,

    /// Shadow atlas size in texels (e.g., 4096 for 4096x4096).
    pub atlas_size: u32,

    /// Per-cascade resolution. For a 2x2 grid layout, each cascade
    /// typically uses `atlas_size / 2` for uniform distribution.
    /// Index 0 = nearest cascade, index 3 = farthest cascade.
    pub cascade_sizes: [u32; 4],
}

impl Default for CascadeRenderConfig {
    fn default() -> Self {
        Self {
            cascade_count: 4,
            atlas_size: 4096,
            cascade_sizes: [2048, 2048, 2048, 2048],
        }
    }
}

impl CascadeRenderConfig {
    /// Create a new configuration with uniform cascade sizes.
    ///
    /// Each cascade will use `atlas_size / 2` resolution for the
    /// 2x2 grid layout.
    pub fn new(cascade_count: u32, atlas_size: u32) -> Self {
        debug_assert!(
            (2..=4).contains(&cascade_count),
            "cascade_count must be 2-4"
        );
        debug_assert!(
            atlas_size.is_power_of_two(),
            "atlas_size should be power of two"
        );

        let cascade_size = atlas_size / 2;
        Self {
            cascade_count,
            atlas_size,
            cascade_sizes: [cascade_size, cascade_size, cascade_size, cascade_size],
        }
    }

    /// Create a configuration with custom per-cascade sizes.
    ///
    /// Useful when different cascades need different resolutions
    /// (e.g., higher resolution for near cascades).
    pub fn with_sizes(cascade_count: u32, atlas_size: u32, sizes: [u32; 4]) -> Self {
        debug_assert!(
            (2..=4).contains(&cascade_count),
            "cascade_count must be 2-4"
        );

        Self {
            cascade_count,
            atlas_size,
            cascade_sizes: sizes,
        }
    }

    /// Validate the configuration.
    ///
    /// Returns `true` if all cascade sizes fit within the atlas grid.
    pub fn is_valid(&self) -> bool {
        let max_cascade_size = self.atlas_size / 2;
        self.cascade_sizes
            .iter()
            .take(self.cascade_count as usize)
            .all(|&size| size <= max_cascade_size && size > 0)
    }
}

// ---------------------------------------------------------------------------
// Cascade View
// ---------------------------------------------------------------------------

/// View data for a single shadow cascade.
///
/// Contains the light-space view-projection matrix and split depth
/// for cascade selection during shadow sampling.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CascadeView {
    /// Light view-projection matrix (column-major).
    ///
    /// Transforms world-space positions to light clip space.
    pub light_view_proj: [[f32; 4]; 4],

    /// View-space split depth for this cascade.
    ///
    /// Used by the fragment shader to select the appropriate cascade.
    pub split_depth: f32,

    /// Index of this cascade's layer in the shadow map array.
    pub shadow_map_index: u32,

    /// Padding for 16-byte alignment.
    pub _pad: [f32; 2],
}

impl Default for CascadeView {
    fn default() -> Self {
        Self {
            light_view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            split_depth: 1000.0,
            shadow_map_index: 0,
            _pad: [0.0; 2],
        }
    }
}

// ---------------------------------------------------------------------------
// Viewport Calculation
// ---------------------------------------------------------------------------

/// Viewport offset and size for a cascade in the shadow atlas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CascadeViewport {
    /// X offset in texels from atlas origin.
    pub x: u32,
    /// Y offset in texels from atlas origin.
    pub y: u32,
    /// Viewport width in texels.
    pub width: u32,
    /// Viewport height in texels.
    pub height: u32,
}

/// Calculate the viewport for a cascade index in a 2x2 grid layout.
///
/// # Grid Layout
///
/// ```text
/// Index | Grid Position | Offset
/// ------|---------------|--------
///   0   |    (0, 0)     | (0, 0)
///   1   |    (1, 0)     | (w, 0)
///   2   |    (0, 1)     | (0, h)
///   3   |    (1, 1)     | (w, h)
/// ```
///
/// # Parameters
///
/// * `cascade_index` - Cascade index (0-3).
/// * `config` - Cascade render configuration.
///
/// # Returns
///
/// The viewport offset and size for the specified cascade.
pub fn calculate_cascade_viewport(
    cascade_index: u32,
    config: &CascadeRenderConfig,
) -> CascadeViewport {
    debug_assert!(cascade_index < 4, "cascade_index must be 0-3");

    let cascade_size = config.cascade_sizes[cascade_index as usize];

    // 2x2 grid: index % 2 = column, index / 2 = row
    let grid_x = cascade_index % 2;
    let grid_y = cascade_index / 2;

    // Calculate offset based on grid position
    // Each cell is atlas_size / 2 in each dimension
    let cell_size = config.atlas_size / 2;
    let x = grid_x * cell_size;
    let y = grid_y * cell_size;

    CascadeViewport {
        x,
        y,
        width: cascade_size,
        height: cascade_size,
    }
}

/// Calculate all cascade viewports for a configuration.
///
/// Returns an array of 4 viewports, even if `cascade_count < 4`.
/// Unused viewports (when `cascade_count < 4`) will have zero size.
pub fn calculate_all_viewports(config: &CascadeRenderConfig) -> [CascadeViewport; 4] {
    let mut viewports = [CascadeViewport {
        x: 0,
        y: 0,
        width: 0,
        height: 0,
    }; 4];

    for i in 0..config.cascade_count {
        viewports[i as usize] = calculate_cascade_viewport(i, config);
    }

    viewports
}

// ---------------------------------------------------------------------------
// Render Mesh
// ---------------------------------------------------------------------------

/// A mesh to be rendered in the shadow pass.
///
/// Contains buffer references and draw parameters for GPU-driven
/// shadow rendering.
pub struct RenderMesh<'a> {
    /// Vertex buffer containing position data.
    pub vertex_buffer: &'a wgpu::Buffer,
    /// Index buffer (optional for non-indexed draws).
    pub index_buffer: Option<&'a wgpu::Buffer>,
    /// Index format (Uint16 or Uint32).
    pub index_format: wgpu::IndexFormat,
    /// Number of indices to draw (or vertices if no index buffer).
    pub index_count: u32,
    /// First index to start drawing from.
    pub first_index: u32,
    /// Instance count for instanced rendering.
    pub instance_count: u32,
    /// Bind group for per-object data (transforms, etc.).
    pub bind_group: Option<&'a wgpu::BindGroup>,
}

// ---------------------------------------------------------------------------
// Shadow Rendering
// ---------------------------------------------------------------------------

/// Render shadow cascades using multiple render passes.
///
/// This function renders the scene from the light's perspective into
/// each cascade's viewport within the shadow atlas. Uses one render
/// pass per cascade (Option 1: sequential rendering).
///
/// # Parameters
///
/// * `encoder` - Command encoder for recording render commands.
/// * `config` - Cascade render configuration.
/// * `shadow_atlas` - Depth texture view for the shadow atlas.
/// * `cascade_views` - Array of 4 cascade view matrices and split depths.
/// * `scene_meshes` - Meshes to render into the shadow map.
/// * `shadow_pipeline` - Render pipeline for shadow pass (depth-only).
/// * `cascade_bind_groups` - Per-cascade bind groups with view-proj matrices.
///
/// # Notes
///
/// The shadow pipeline should be configured for depth-only output with
/// appropriate depth bias to prevent shadow acne.
pub fn render_shadow_cascades<'a>(
    encoder: &'a mut RhiCommandEncoder,
    config: &CascadeRenderConfig,
    shadow_atlas: &wgpu::TextureView,
    cascade_views: &[CascadeView; 4],
    scene_meshes: &[RenderMesh<'_>],
    shadow_pipeline: &wgpu::RenderPipeline,
    cascade_bind_groups: &[&wgpu::BindGroup; 4],
) {
    let viewports = calculate_all_viewports(config);

    for cascade_idx in 0..config.cascade_count as usize {
        let viewport = &viewports[cascade_idx];
        let _cascade_view = &cascade_views[cascade_idx];
        let bind_group = cascade_bind_groups[cascade_idx];

        // Begin depth-only render pass for this cascade
        let mut render_pass = encoder.inner_mut().begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(&format!("Shadow Cascade {}", cascade_idx)),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: shadow_atlas,
                depth_ops: Some(wgpu::Operations {
                    load: if cascade_idx == 0 {
                        wgpu::LoadOp::Clear(1.0)
                    } else {
                        wgpu::LoadOp::Load
                    },
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        // Set viewport for this cascade
        render_pass.set_viewport(
            viewport.x as f32,
            viewport.y as f32,
            viewport.width as f32,
            viewport.height as f32,
            0.0,
            1.0,
        );

        // Set scissor rect to match viewport
        render_pass.set_scissor_rect(
            viewport.x,
            viewport.y,
            viewport.width,
            viewport.height,
        );

        // Bind shadow pipeline
        render_pass.set_pipeline(shadow_pipeline);

        // Bind cascade-specific data (view-projection matrix)
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

/// Render shadow cascades using a single render pass with viewport changes.
///
/// Alternative approach that uses one render pass with dynamic viewport
/// switching. May be more efficient on some hardware due to reduced
/// render pass overhead.
///
/// # Notes
///
/// This approach clears the entire atlas once at the start, then
/// renders each cascade by changing the viewport within the same pass.
pub fn render_shadow_cascades_single_pass<'a>(
    encoder: &'a mut RhiCommandEncoder,
    config: &CascadeRenderConfig,
    shadow_atlas: &wgpu::TextureView,
    cascade_views: &[CascadeView; 4],
    scene_meshes: &[RenderMesh<'_>],
    shadow_pipeline: &wgpu::RenderPipeline,
    cascade_bind_groups: &[&wgpu::BindGroup; 4],
) {
    let viewports = calculate_all_viewports(config);

    // Begin single render pass for all cascades
    let mut render_pass = encoder.inner_mut().begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Shadow Cascades (Single Pass)"),
        color_attachments: &[],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: shadow_atlas,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
    });

    // Bind shadow pipeline once
    render_pass.set_pipeline(shadow_pipeline);

    for cascade_idx in 0..config.cascade_count as usize {
        let viewport = &viewports[cascade_idx];
        let _cascade_view = &cascade_views[cascade_idx];
        let bind_group = cascade_bind_groups[cascade_idx];

        // Set viewport for this cascade
        render_pass.set_viewport(
            viewport.x as f32,
            viewport.y as f32,
            viewport.width as f32,
            viewport.height as f32,
            0.0,
            1.0,
        );

        // Set scissor rect to match viewport
        render_pass.set_scissor_rect(
            viewport.x,
            viewport.y,
            viewport.width,
            viewport.height,
        );

        // Bind cascade-specific data (view-projection matrix)
        render_pass.set_bind_group(0, bind_group, &[]);

        // Render all meshes for this cascade
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
    }

    // Render pass ends when dropped
}

// ---------------------------------------------------------------------------
// Helper: Create Shadow Atlas Texture
// ---------------------------------------------------------------------------

/// Create a depth texture suitable for use as a shadow atlas.
///
/// # Parameters
///
/// * `device` - The wgpu device.
/// * `config` - Cascade render configuration.
///
/// # Returns
///
/// A depth texture sized for the shadow atlas.
pub fn create_shadow_atlas(
    device: &wgpu::Device,
    config: &CascadeRenderConfig,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Shadow Atlas"),
        size: wgpu::Extent3d {
            width: config.atlas_size,
            height: config.atlas_size,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- CascadeRenderConfig --------------------------------------------------

    #[test]
    fn test_config_default() {
        let config = CascadeRenderConfig::default();
        assert_eq!(config.cascade_count, 4);
        assert_eq!(config.atlas_size, 4096);
        assert_eq!(config.cascade_sizes, [2048, 2048, 2048, 2048]);
        assert!(config.is_valid());
    }

    #[test]
    fn test_config_new() {
        let config = CascadeRenderConfig::new(3, 2048);
        assert_eq!(config.cascade_count, 3);
        assert_eq!(config.atlas_size, 2048);
        assert_eq!(config.cascade_sizes, [1024, 1024, 1024, 1024]);
        assert!(config.is_valid());
    }

    #[test]
    fn test_config_with_sizes() {
        let config = CascadeRenderConfig::with_sizes(4, 4096, [2048, 2048, 1024, 1024]);
        assert_eq!(config.cascade_count, 4);
        assert!(config.is_valid());
    }

    #[test]
    fn test_config_invalid_sizes() {
        // Cascade size larger than max (atlas_size / 2)
        let config = CascadeRenderConfig::with_sizes(4, 4096, [4096, 2048, 2048, 2048]);
        assert!(!config.is_valid());
    }

    // -- Viewport Calculation -------------------------------------------------

    #[test]
    fn test_viewport_cascade_0() {
        let config = CascadeRenderConfig::default();
        let viewport = calculate_cascade_viewport(0, &config);

        assert_eq!(viewport.x, 0);
        assert_eq!(viewport.y, 0);
        assert_eq!(viewport.width, 2048);
        assert_eq!(viewport.height, 2048);
    }

    #[test]
    fn test_viewport_cascade_1() {
        let config = CascadeRenderConfig::default();
        let viewport = calculate_cascade_viewport(1, &config);

        assert_eq!(viewport.x, 2048);
        assert_eq!(viewport.y, 0);
        assert_eq!(viewport.width, 2048);
        assert_eq!(viewport.height, 2048);
    }

    #[test]
    fn test_viewport_cascade_2() {
        let config = CascadeRenderConfig::default();
        let viewport = calculate_cascade_viewport(2, &config);

        assert_eq!(viewport.x, 0);
        assert_eq!(viewport.y, 2048);
        assert_eq!(viewport.width, 2048);
        assert_eq!(viewport.height, 2048);
    }

    #[test]
    fn test_viewport_cascade_3() {
        let config = CascadeRenderConfig::default();
        let viewport = calculate_cascade_viewport(3, &config);

        assert_eq!(viewport.x, 2048);
        assert_eq!(viewport.y, 2048);
        assert_eq!(viewport.width, 2048);
        assert_eq!(viewport.height, 2048);
    }

    #[test]
    fn test_all_viewports_cover_atlas() {
        let config = CascadeRenderConfig::default();
        let viewports = calculate_all_viewports(&config);

        // All 4 cascades should cover the entire atlas without overlap
        // when using uniform sizes
        for i in 0..4 {
            let vp = &viewports[i];
            assert!(vp.x + vp.width <= config.atlas_size);
            assert!(vp.y + vp.height <= config.atlas_size);
        }

        // Check grid positions
        assert_eq!((viewports[0].x, viewports[0].y), (0, 0));
        assert_eq!((viewports[1].x, viewports[1].y), (2048, 0));
        assert_eq!((viewports[2].x, viewports[2].y), (0, 2048));
        assert_eq!((viewports[3].x, viewports[3].y), (2048, 2048));
    }

    #[test]
    fn test_viewport_smaller_atlas() {
        let config = CascadeRenderConfig::new(4, 2048);
        let viewports = calculate_all_viewports(&config);

        assert_eq!(viewports[0].width, 1024);
        assert_eq!(viewports[0].height, 1024);
        assert_eq!(viewports[1].x, 1024);
        assert_eq!(viewports[2].y, 1024);
        assert_eq!(viewports[3].x, 1024);
        assert_eq!(viewports[3].y, 1024);
    }

    #[test]
    fn test_viewport_two_cascades() {
        let config = CascadeRenderConfig::new(2, 4096);
        let viewports = calculate_all_viewports(&config);

        // Only first 2 viewports should have size
        assert!(viewports[0].width > 0);
        assert!(viewports[1].width > 0);
        assert_eq!(viewports[2].width, 0);
        assert_eq!(viewports[3].width, 0);
    }

    // -- CascadeView ----------------------------------------------------------

    #[test]
    fn test_cascade_view_size() {
        // CascadeView should be 80 bytes (4x4 matrix + 2 floats + padding)
        assert_eq!(std::mem::size_of::<CascadeView>(), 80);
    }

    #[test]
    fn test_cascade_view_default() {
        let view = CascadeView::default();
        assert_eq!(view.split_depth, 1000.0);
        assert_eq!(view.shadow_map_index, 0);
        // Identity matrix check
        assert_eq!(view.light_view_proj[0][0], 1.0);
        assert_eq!(view.light_view_proj[1][1], 1.0);
        assert_eq!(view.light_view_proj[2][2], 1.0);
        assert_eq!(view.light_view_proj[3][3], 1.0);
    }
}
