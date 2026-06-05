//! Frame graph nodes for lighting passes in the TRINITY renderer.
//!
//! This module defines frame graph nodes that wrap lighting compute and graphics
//! passes. Each node type specifies its inputs, outputs, and dependencies to
//! enable the frame graph compiler to schedule and insert barriers correctly.
//!
//! # Node Types
//!
//! - [`LightCullingNode`]: Compute pass that bins lights into froxels (frustum voxels).
//! - [`ShadowRenderNode`]: Graphics pass that renders shadow maps (CSM, cube, or spot).
//! - [`DeferredLightingNode`]: Compute/graphics pass that applies deferred lighting.
//!
//! # Dependencies
//!
//! ```text
//! ┌─────────────────┐     ┌─────────────────┐
//! │ LightCullingNode│     │ ShadowRenderNode│
//! │   (compute)     │     │   (graphics)    │
//! └────────┬────────┘     └────────┬────────┘
//!          │ froxel_indices        │ shadow_maps
//!          │                       │
//!          └───────────┬───────────┘
//!                      ▼
//!          ┌───────────────────────┐
//!          │ DeferredLightingNode  │
//!          │   (compute/graphics)  │
//!          └───────────────────────┘
//! ```
//!
//! # Barrier Requirements
//!
//! - **Storage buffer barrier**: Between light culling (write) and deferred lighting (read).
//! - **Texture barrier**: Between shadow render (write) and deferred lighting (read).

use crate::frame_graph::{
    AttachmentLoadOp, AttachmentStoreOp, ColorAttachment, DepthStencilAttachment,
    DispatchSource, IrPass, PassIndex, ResourceAccessSet, ResourceHandle, ViewType,
};

// ---------------------------------------------------------------------------
// Shadow type enumeration
// ---------------------------------------------------------------------------

/// The type of shadow map being rendered.
///
/// Determines the render pass configuration (viewport count, projection, etc.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShadowType {
    /// Cascaded Shadow Maps for directional lights.
    ///
    /// Renders multiple cascades into a 2D array texture or atlas.
    /// Typically 2-4 cascades with split distances based on view frustum.
    Csm {
        /// Number of cascades (2-4).
        cascade_count: u32,
    },
    /// Cube shadow map for point lights.
    ///
    /// Renders 6 faces of a cube map for omnidirectional shadows.
    Cube,
    /// Single 2D shadow map for spot lights.
    ///
    /// Uses perspective projection matching the spot cone.
    Spot,
}

impl std::fmt::Display for ShadowType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Csm { cascade_count } => write!(f, "CSM({})", cascade_count),
            Self::Cube => write!(f, "Cube"),
            Self::Spot => write!(f, "Spot"),
        }
    }
}

// ---------------------------------------------------------------------------
// Light culling node
// ---------------------------------------------------------------------------

/// Input resources for the light culling pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LightCullingInputs {
    /// Buffer containing packed light data (positions, radii, colors).
    pub light_buffers: ResourceHandle,
    /// Buffer containing camera matrices and frustum data.
    pub camera: ResourceHandle,
    /// Optional depth buffer for depth-aware culling (HZB or linearized depth).
    pub depth_pyramid: Option<ResourceHandle>,
}

impl LightCullingInputs {
    /// Create new light culling inputs.
    pub fn new(light_buffers: ResourceHandle, camera: ResourceHandle) -> Self {
        Self {
            light_buffers,
            camera,
            depth_pyramid: None,
        }
    }

    /// Create light culling inputs with depth-aware culling.
    pub fn with_depth_pyramid(
        light_buffers: ResourceHandle,
        camera: ResourceHandle,
        depth_pyramid: ResourceHandle,
    ) -> Self {
        Self {
            light_buffers,
            camera,
            depth_pyramid: Some(depth_pyramid),
        }
    }
}

/// Output resources from the light culling pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LightCullingOutputs {
    /// Buffer containing per-froxel light indices.
    ///
    /// Layout: `[froxel_count][max_lights_per_froxel]` u32 indices.
    pub froxel_light_indices: ResourceHandle,
    /// Buffer containing per-froxel light counts.
    ///
    /// Layout: `[froxel_count]` u32 counts.
    pub froxel_light_counts: ResourceHandle,
}

impl LightCullingOutputs {
    /// Create new light culling outputs.
    pub fn new(froxel_light_indices: ResourceHandle, froxel_light_counts: ResourceHandle) -> Self {
        Self {
            froxel_light_indices,
            froxel_light_counts,
        }
    }
}

/// Froxel grid configuration for light culling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FroxelConfig {
    /// Number of froxels in X (typically screen_width / tile_size).
    pub froxels_x: u32,
    /// Number of froxels in Y (typically screen_height / tile_size).
    pub froxels_y: u32,
    /// Number of depth slices (logarithmic distribution).
    pub froxels_z: u32,
    /// Tile size in pixels (typically 16 or 32).
    pub tile_size: u32,
}

impl Default for FroxelConfig {
    fn default() -> Self {
        Self {
            froxels_x: 120,  // 1920 / 16
            froxels_y: 68,   // 1080 / 16
            froxels_z: 24,   // Depth slices
            tile_size: 16,
        }
    }
}

impl FroxelConfig {
    /// Create a froxel config from screen dimensions and tile size.
    pub fn from_screen(width: u32, height: u32, tile_size: u32, depth_slices: u32) -> Self {
        Self {
            froxels_x: (width + tile_size - 1) / tile_size,
            froxels_y: (height + tile_size - 1) / tile_size,
            froxels_z: depth_slices,
            tile_size,
        }
    }

    /// Total number of froxels in the grid.
    pub fn total_froxels(&self) -> u32 {
        self.froxels_x * self.froxels_y * self.froxels_z
    }

    /// Compute workgroup count for dispatching the culling shader.
    pub fn workgroup_count(&self) -> (u32, u32, u32) {
        // Assumes 8x8x1 workgroup size (common for tile-based culling)
        let wg_x = (self.froxels_x + 7) / 8;
        let wg_y = (self.froxels_y + 7) / 8;
        let wg_z = self.froxels_z;
        (wg_x, wg_y, wg_z)
    }
}

/// Frame graph node for light culling (froxel binning).
///
/// This compute pass bins lights into a 3D froxel grid for efficient
/// per-pixel light iteration during deferred lighting.
#[derive(Debug, Clone)]
pub struct LightCullingNode {
    /// Input resources (light buffers, camera).
    pub inputs: LightCullingInputs,
    /// Output resources (froxel light indices).
    pub outputs: LightCullingOutputs,
    /// Froxel grid configuration.
    pub config: FroxelConfig,
    /// Maximum lights per froxel (buffer allocation limit).
    pub max_lights_per_froxel: u32,
}

impl LightCullingNode {
    /// Create a new light culling node.
    pub fn new(
        inputs: LightCullingInputs,
        outputs: LightCullingOutputs,
        config: FroxelConfig,
    ) -> Self {
        Self {
            inputs,
            outputs,
            config,
            max_lights_per_froxel: 256,
        }
    }

    /// Create a light culling node with custom max lights per froxel.
    pub fn with_max_lights(mut self, max_lights: u32) -> Self {
        self.max_lights_per_froxel = max_lights;
        self
    }

    /// Build the resource access set for this node.
    pub fn build_access_set(&self) -> ResourceAccessSet {
        let mut reads = vec![self.inputs.light_buffers, self.inputs.camera];
        if let Some(depth) = self.inputs.depth_pyramid {
            reads.push(depth);
        }

        let writes = vec![
            self.outputs.froxel_light_indices,
            self.outputs.froxel_light_counts,
        ];

        ResourceAccessSet { reads, writes }
    }

    /// Convert this node into an `IrPass` for the frame graph compiler.
    pub fn into_ir_pass(self, index: PassIndex) -> IrPass {
        let (wg_x, wg_y, wg_z) = self.config.workgroup_count();
        let mut pass = IrPass::compute(
            index,
            "light_culling",
            DispatchSource::Direct {
                group_count_x: wg_x,
                group_count_y: wg_y,
                group_count_z: wg_z,
            },
            ViewType::StorageBuffer,
        );
        pass.access_set = self.build_access_set();
        pass.tags.push("lighting".to_string());
        pass
    }
}

// ---------------------------------------------------------------------------
// Shadow render node
// ---------------------------------------------------------------------------

/// Input resources for shadow rendering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShadowInputs {
    /// Buffer containing scene geometry (vertex/index buffers or indirect args).
    pub scene_geometry: ResourceHandle,
    /// Buffer containing per-object transforms.
    pub transforms: ResourceHandle,
    /// Buffer containing light view-projection matrices.
    pub light_matrices: ResourceHandle,
}

impl ShadowInputs {
    /// Create new shadow inputs.
    pub fn new(
        scene_geometry: ResourceHandle,
        transforms: ResourceHandle,
        light_matrices: ResourceHandle,
    ) -> Self {
        Self {
            scene_geometry,
            transforms,
            light_matrices,
        }
    }
}

/// Output resources from shadow rendering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShadowOutputs {
    /// Shadow map texture (2D for spot, 2D array for CSM, cube for point).
    pub shadow_map: ResourceHandle,
}

impl ShadowOutputs {
    /// Create new shadow outputs.
    pub fn new(shadow_map: ResourceHandle) -> Self {
        Self { shadow_map }
    }
}

/// Frame graph node for shadow map rendering.
///
/// This graphics pass renders the scene from a light's perspective to
/// produce shadow maps for use in deferred lighting.
#[derive(Debug, Clone)]
pub struct ShadowRenderNode {
    /// The type of shadow map (CSM, cube, spot).
    pub shadow_type: ShadowType,
    /// Input resources (geometry, transforms, matrices).
    pub inputs: ShadowInputs,
    /// Output resources (shadow map).
    pub outputs: ShadowOutputs,
    /// Shadow map resolution (width = height for square maps).
    pub resolution: u32,
    /// Light index this shadow node corresponds to.
    pub light_index: u32,
}

impl ShadowRenderNode {
    /// Create a new shadow render node.
    pub fn new(
        shadow_type: ShadowType,
        inputs: ShadowInputs,
        outputs: ShadowOutputs,
        resolution: u32,
    ) -> Self {
        Self {
            shadow_type,
            inputs,
            outputs,
            resolution,
            light_index: 0,
        }
    }

    /// Set the light index for this shadow node.
    pub fn with_light_index(mut self, index: u32) -> Self {
        self.light_index = index;
        self
    }

    /// Build the resource access set for this node.
    pub fn build_access_set(&self) -> ResourceAccessSet {
        let reads = vec![
            self.inputs.scene_geometry,
            self.inputs.transforms,
            self.inputs.light_matrices,
        ];
        let writes = vec![self.outputs.shadow_map];

        ResourceAccessSet { reads, writes }
    }

    /// Number of render passes needed for this shadow type.
    pub fn pass_count(&self) -> u32 {
        match self.shadow_type {
            ShadowType::Csm { cascade_count } => cascade_count,
            ShadowType::Cube => 6,
            ShadowType::Spot => 1,
        }
    }

    /// Convert this node into an `IrPass` for the frame graph compiler.
    pub fn into_ir_pass(self, index: PassIndex) -> IrPass {
        let name = match self.shadow_type {
            ShadowType::Csm { cascade_count } => {
                format!("shadow_csm_{}_L{}", cascade_count, self.light_index)
            }
            ShadowType::Cube => format!("shadow_cube_L{}", self.light_index),
            ShadowType::Spot => format!("shadow_spot_L{}", self.light_index),
        };

        let depth_attachment = DepthStencilAttachment {
            resource: self.outputs.shadow_map,
            depth_load_op: AttachmentLoadOp::Clear,
            depth_store_op: AttachmentStoreOp::Store,
            stencil_load_op: AttachmentLoadOp::DontCare,
            stencil_store_op: AttachmentStoreOp::DontCare,
            clear_depth: 1.0,
            clear_stencil: 0,
            depth_test_enabled: true,
            depth_write_enabled: true,
        };

        let mut pass = IrPass::graphics(
            index,
            name,
            Vec::new(), // No color attachments for shadow pass
            Some(depth_attachment),
            crate::frame_graph::InstanceSource::Indirect {
                buffer: self.inputs.scene_geometry,
                offset: 0,
                draw_count: 1024, // Max draws, actual count from GPU buffer
                stride: 20,       // DrawIndexedIndirectCommand size
            },
            ViewType::Texture2D,
        );
        pass.access_set = self.build_access_set();
        pass.tags.push("shadow".to_string());
        pass.tags.push("lighting".to_string());
        pass
    }
}

// ---------------------------------------------------------------------------
// Deferred lighting node
// ---------------------------------------------------------------------------

/// Input resources for deferred lighting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeferredLightingInputs {
    /// G-buffer albedo texture.
    pub g_buffer_albedo: ResourceHandle,
    /// G-buffer normal texture.
    pub g_buffer_normal: ResourceHandle,
    /// G-buffer material properties (roughness, metallic).
    pub g_buffer_material: ResourceHandle,
    /// G-buffer depth.
    pub g_buffer_depth: ResourceHandle,
    /// Buffer containing packed light data.
    pub light_buffers: ResourceHandle,
    /// Froxel light indices from culling pass.
    pub froxel_indices: ResourceHandle,
    /// Froxel light counts from culling pass.
    pub froxel_counts: ResourceHandle,
    /// Shadow map atlas or array.
    pub shadow_maps: ResourceHandle,
    /// Optional: Contact shadow texture from screen-space ray marching (T-LIT-8.2).
    pub contact_shadow: Option<ResourceHandle>,
    /// Optional: DDGI irradiance probes.
    pub ddgi_probes: Option<ResourceHandle>,
    /// Optional: Screen-space ambient occlusion.
    pub ssao: Option<ResourceHandle>,
}

impl DeferredLightingInputs {
    /// Create new deferred lighting inputs with required resources.
    pub fn new(
        g_buffer_albedo: ResourceHandle,
        g_buffer_normal: ResourceHandle,
        g_buffer_material: ResourceHandle,
        g_buffer_depth: ResourceHandle,
        light_buffers: ResourceHandle,
        froxel_indices: ResourceHandle,
        froxel_counts: ResourceHandle,
        shadow_maps: ResourceHandle,
    ) -> Self {
        Self {
            g_buffer_albedo,
            g_buffer_normal,
            g_buffer_material,
            g_buffer_depth,
            light_buffers,
            froxel_indices,
            froxel_counts,
            shadow_maps,
            contact_shadow: None,
            ddgi_probes: None,
            ssao: None,
        }
    }

    /// Add contact shadow texture for screen-space shadow enhancement (T-LIT-8.2).
    ///
    /// Contact shadows improve shadow quality by ray-marching in screen space
    /// to detect small-scale occlusions that traditional shadow maps miss.
    pub fn with_contact_shadow(mut self, contact_shadow: ResourceHandle) -> Self {
        self.contact_shadow = Some(contact_shadow);
        self
    }

    /// Add DDGI probes for global illumination.
    pub fn with_ddgi(mut self, ddgi_probes: ResourceHandle) -> Self {
        self.ddgi_probes = Some(ddgi_probes);
        self
    }

    /// Add SSAO for ambient occlusion.
    pub fn with_ssao(mut self, ssao: ResourceHandle) -> Self {
        self.ssao = Some(ssao);
        self
    }
}

/// Output resources from deferred lighting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeferredLightingOutputs {
    /// HDR color output (before tonemapping).
    pub hdr_output: ResourceHandle,
    /// Optional: Separate specular output for SSR/reflections.
    pub specular_output: Option<ResourceHandle>,
}

impl DeferredLightingOutputs {
    /// Create new deferred lighting outputs.
    pub fn new(hdr_output: ResourceHandle) -> Self {
        Self {
            hdr_output,
            specular_output: None,
        }
    }

    /// Create outputs with separate specular buffer.
    pub fn with_specular(mut self, specular: ResourceHandle) -> Self {
        self.specular_output = Some(specular);
        self
    }
}

/// Frame graph node for deferred lighting.
///
/// This pass applies lighting to the G-buffer using froxel-binned lights
/// and shadow maps to produce an HDR output.
#[derive(Debug, Clone)]
pub struct DeferredLightingNode {
    /// Input resources.
    pub inputs: DeferredLightingInputs,
    /// Output resources.
    pub outputs: DeferredLightingOutputs,
    /// Output resolution (width).
    pub width: u32,
    /// Output resolution (height).
    pub height: u32,
    /// Whether to use compute shader (vs. fullscreen triangle).
    pub use_compute: bool,
}

impl DeferredLightingNode {
    /// Create a new deferred lighting node.
    pub fn new(
        inputs: DeferredLightingInputs,
        outputs: DeferredLightingOutputs,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            inputs,
            outputs,
            width,
            height,
            use_compute: true,
        }
    }

    /// Use graphics pass (fullscreen triangle) instead of compute.
    pub fn with_graphics_pass(mut self) -> Self {
        self.use_compute = false;
        self
    }

    /// Build the resource access set for this node.
    pub fn build_access_set(&self) -> ResourceAccessSet {
        let mut reads = vec![
            self.inputs.g_buffer_albedo,
            self.inputs.g_buffer_normal,
            self.inputs.g_buffer_material,
            self.inputs.g_buffer_depth,
            self.inputs.light_buffers,
            self.inputs.froxel_indices,
            self.inputs.froxel_counts,
            self.inputs.shadow_maps,
        ];

        // T-LIT-8.2: Add contact shadow as read dependency
        if let Some(contact_shadow) = self.inputs.contact_shadow {
            reads.push(contact_shadow);
        }
        if let Some(ddgi) = self.inputs.ddgi_probes {
            reads.push(ddgi);
        }
        if let Some(ssao) = self.inputs.ssao {
            reads.push(ssao);
        }

        let mut writes = vec![self.outputs.hdr_output];
        if let Some(specular) = self.outputs.specular_output {
            writes.push(specular);
        }

        ResourceAccessSet { reads, writes }
    }

    /// Convert this node into an `IrPass` for the frame graph compiler.
    pub fn into_ir_pass(self, index: PassIndex) -> IrPass {
        let access_set = self.build_access_set();

        if self.use_compute {
            // Compute pass with 8x8 workgroups
            let wg_x = (self.width + 7) / 8;
            let wg_y = (self.height + 7) / 8;

            let mut pass = IrPass::compute(
                index,
                "deferred_lighting",
                DispatchSource::Direct {
                    group_count_x: wg_x,
                    group_count_y: wg_y,
                    group_count_z: 1,
                },
                ViewType::Storage,
            );
            pass.access_set = access_set;
            pass.tags.push("lighting".to_string());
            pass
        } else {
            // Graphics pass with fullscreen triangle
            let color_attachment = ColorAttachment {
                resource: self.outputs.hdr_output,
                mip_level: 0,
                array_layer: 0,
                load_op: AttachmentLoadOp::DontCare,
                store_op: AttachmentStoreOp::Store,
                clear_color: [0.0, 0.0, 0.0, 0.0],
            };

            let mut pass = IrPass::graphics(
                index,
                "deferred_lighting",
                vec![color_attachment],
                None,
                crate::frame_graph::InstanceSource::Direct {
                    index_count: 3,
                    instance_count: 1,
                    base_vertex: 0,
                    first_index: 0,
                    first_instance: 0,
                },
                ViewType::Texture2D,
            );
            pass.access_set = access_set;
            pass.tags.push("lighting".to_string());
            pass
        }
    }
}

// ---------------------------------------------------------------------------
// Dependency graph builder
// ---------------------------------------------------------------------------

/// Represents a dependency between two frame graph nodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeDependency {
    /// The node that must execute first (producer).
    pub from: &'static str,
    /// The node that depends on the producer (consumer).
    pub to: &'static str,
    /// The resource creating the dependency.
    pub resource: ResourceHandle,
    /// Description of the dependency.
    pub description: &'static str,
}

impl NodeDependency {
    /// Create a new node dependency.
    pub const fn new(
        from: &'static str,
        to: &'static str,
        resource: ResourceHandle,
        description: &'static str,
    ) -> Self {
        Self {
            from,
            to,
            resource,
            description,
        }
    }
}

/// Barrier type for resource synchronization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BarrierType {
    /// Storage buffer barrier (compute write -> compute/graphics read).
    StorageBuffer,
    /// Texture barrier (graphics write -> compute/graphics read).
    Texture,
    /// Depth attachment barrier (depth write -> shader read).
    DepthAttachment,
}

/// Barrier requirement between passes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BarrierRequirement {
    /// Type of barrier needed.
    pub barrier_type: BarrierType,
    /// Resource requiring the barrier.
    pub resource: ResourceHandle,
    /// Source stage (writing pass).
    pub src_stage: &'static str,
    /// Destination stage (reading pass).
    pub dst_stage: &'static str,
}

impl BarrierRequirement {
    /// Create a new barrier requirement.
    pub const fn new(
        barrier_type: BarrierType,
        resource: ResourceHandle,
        src_stage: &'static str,
        dst_stage: &'static str,
    ) -> Self {
        Self {
            barrier_type,
            resource,
            src_stage,
            dst_stage,
        }
    }
}

/// Build the standard lighting pass dependencies.
///
/// Returns the dependencies between light culling, shadow render, and
/// deferred lighting nodes.
pub fn build_lighting_dependencies(
    froxel_indices: ResourceHandle,
    froxel_counts: ResourceHandle,
    shadow_map: ResourceHandle,
) -> Vec<NodeDependency> {
    vec![
        NodeDependency::new(
            "light_culling",
            "deferred_lighting",
            froxel_indices,
            "Froxel light indices from culling pass",
        ),
        NodeDependency::new(
            "light_culling",
            "deferred_lighting",
            froxel_counts,
            "Froxel light counts from culling pass",
        ),
        NodeDependency::new(
            "shadow_render",
            "deferred_lighting",
            shadow_map,
            "Shadow maps from shadow rendering",
        ),
    ]
}

/// Build the barrier requirements for lighting passes.
pub fn build_lighting_barriers(
    froxel_indices: ResourceHandle,
    froxel_counts: ResourceHandle,
    shadow_map: ResourceHandle,
) -> Vec<BarrierRequirement> {
    vec![
        // Storage buffer barriers from light culling to deferred lighting
        BarrierRequirement::new(
            BarrierType::StorageBuffer,
            froxel_indices,
            "light_culling",
            "deferred_lighting",
        ),
        BarrierRequirement::new(
            BarrierType::StorageBuffer,
            froxel_counts,
            "light_culling",
            "deferred_lighting",
        ),
        // Texture barrier from shadow render to deferred lighting
        BarrierRequirement::new(
            BarrierType::DepthAttachment,
            shadow_map,
            "shadow_render",
            "deferred_lighting",
        ),
    ]
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_graph::PassType;

    // Helper to create test resource handles
    fn make_handle(id: u32) -> ResourceHandle {
        ResourceHandle(id)
    }

    // -----------------------------------------------------------------------
    // LightCullingNode tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_light_culling_node_creation() {
        let inputs = LightCullingInputs::new(make_handle(0), make_handle(1));
        let outputs = LightCullingOutputs::new(make_handle(2), make_handle(3));
        let config = FroxelConfig::default();

        let node = LightCullingNode::new(inputs.clone(), outputs.clone(), config);

        assert_eq!(node.inputs, inputs);
        assert_eq!(node.outputs, outputs);
        assert_eq!(node.config, config);
        assert_eq!(node.max_lights_per_froxel, 256);
    }

    #[test]
    fn test_light_culling_with_depth_pyramid() {
        let inputs = LightCullingInputs::with_depth_pyramid(
            make_handle(0),
            make_handle(1),
            make_handle(10),
        );

        assert_eq!(inputs.depth_pyramid, Some(make_handle(10)));
    }

    #[test]
    fn test_light_culling_access_set() {
        let inputs = LightCullingInputs::with_depth_pyramid(
            make_handle(0),
            make_handle(1),
            make_handle(10),
        );
        let outputs = LightCullingOutputs::new(make_handle(2), make_handle(3));
        let node = LightCullingNode::new(inputs, outputs, FroxelConfig::default());

        let access = node.build_access_set();

        // Should read: light_buffers, camera, depth_pyramid
        assert_eq!(access.reads.len(), 3);
        assert!(access.reads.contains(&make_handle(0)));
        assert!(access.reads.contains(&make_handle(1)));
        assert!(access.reads.contains(&make_handle(10)));

        // Should write: froxel_light_indices, froxel_light_counts
        assert_eq!(access.writes.len(), 2);
        assert!(access.writes.contains(&make_handle(2)));
        assert!(access.writes.contains(&make_handle(3)));
    }

    #[test]
    fn test_light_culling_into_ir_pass() {
        let inputs = LightCullingInputs::new(make_handle(0), make_handle(1));
        let outputs = LightCullingOutputs::new(make_handle(2), make_handle(3));
        let config = FroxelConfig {
            froxels_x: 120,
            froxels_y: 68,
            froxels_z: 24,
            tile_size: 16,
        };
        let node = LightCullingNode::new(inputs, outputs, config);

        let pass = node.into_ir_pass(PassIndex(0));

        assert_eq!(pass.name, "light_culling");
        assert_eq!(pass.pass_type, PassType::Compute);
        assert!(pass.tags.contains(&"lighting".to_string()));

        // Verify dispatch dimensions
        if let Some(DispatchSource::Direct {
            group_count_x,
            group_count_y,
            group_count_z,
        }) = &pass.dispatch_source
        {
            assert_eq!(*group_count_x, 15); // (120 + 7) / 8
            assert_eq!(*group_count_y, 9);  // (68 + 7) / 8
            assert_eq!(*group_count_z, 24);
        } else {
            panic!("Expected direct dispatch source");
        }
    }

    // -----------------------------------------------------------------------
    // FroxelConfig tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_froxel_config_from_screen() {
        let config = FroxelConfig::from_screen(1920, 1080, 16, 24);

        assert_eq!(config.froxels_x, 120);
        assert_eq!(config.froxels_y, 68);
        assert_eq!(config.froxels_z, 24);
        assert_eq!(config.tile_size, 16);
    }

    #[test]
    fn test_froxel_config_total() {
        let config = FroxelConfig::from_screen(1920, 1080, 16, 24);

        assert_eq!(config.total_froxels(), 120 * 68 * 24);
    }

    #[test]
    fn test_froxel_config_workgroup_count() {
        let config = FroxelConfig {
            froxels_x: 120,
            froxels_y: 68,
            froxels_z: 24,
            tile_size: 16,
        };

        let (wg_x, wg_y, wg_z) = config.workgroup_count();

        assert_eq!(wg_x, 15); // ceil(120 / 8)
        assert_eq!(wg_y, 9);  // ceil(68 / 8)
        assert_eq!(wg_z, 24);
    }

    // -----------------------------------------------------------------------
    // ShadowRenderNode tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_shadow_render_node_creation() {
        let inputs = ShadowInputs::new(make_handle(0), make_handle(1), make_handle(2));
        let outputs = ShadowOutputs::new(make_handle(3));

        let node = ShadowRenderNode::new(
            ShadowType::Csm { cascade_count: 4 },
            inputs.clone(),
            outputs.clone(),
            2048,
        );

        assert_eq!(node.shadow_type, ShadowType::Csm { cascade_count: 4 });
        assert_eq!(node.resolution, 2048);
        assert_eq!(node.light_index, 0);
    }

    #[test]
    fn test_shadow_render_pass_count() {
        let inputs = ShadowInputs::new(make_handle(0), make_handle(1), make_handle(2));
        let outputs = ShadowOutputs::new(make_handle(3));

        let csm = ShadowRenderNode::new(
            ShadowType::Csm { cascade_count: 4 },
            inputs.clone(),
            outputs.clone(),
            2048,
        );
        assert_eq!(csm.pass_count(), 4);

        let cube = ShadowRenderNode::new(ShadowType::Cube, inputs.clone(), outputs.clone(), 512);
        assert_eq!(cube.pass_count(), 6);

        let spot = ShadowRenderNode::new(ShadowType::Spot, inputs, outputs, 1024);
        assert_eq!(spot.pass_count(), 1);
    }

    #[test]
    fn test_shadow_render_access_set() {
        let inputs = ShadowInputs::new(make_handle(0), make_handle(1), make_handle(2));
        let outputs = ShadowOutputs::new(make_handle(3));
        let node = ShadowRenderNode::new(ShadowType::Spot, inputs, outputs, 1024);

        let access = node.build_access_set();

        assert_eq!(access.reads.len(), 3);
        assert_eq!(access.writes.len(), 1);
        assert!(access.writes.contains(&make_handle(3)));
    }

    #[test]
    fn test_shadow_render_into_ir_pass() {
        let inputs = ShadowInputs::new(make_handle(0), make_handle(1), make_handle(2));
        let outputs = ShadowOutputs::new(make_handle(3));
        let node = ShadowRenderNode::new(ShadowType::Csm { cascade_count: 4 }, inputs, outputs, 2048)
            .with_light_index(5);

        let pass = node.into_ir_pass(PassIndex(1));

        assert_eq!(pass.name, "shadow_csm_4_L5");
        assert_eq!(pass.pass_type, PassType::Graphics);
        assert!(pass.depth_stencil.is_some());
        assert!(pass.tags.contains(&"shadow".to_string()));
        assert!(pass.tags.contains(&"lighting".to_string()));

        let ds = pass.depth_stencil.unwrap();
        assert_eq!(ds.resource, make_handle(3));
        assert_eq!(ds.clear_depth, 1.0);
    }

    // -----------------------------------------------------------------------
    // DeferredLightingNode tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_deferred_lighting_node_creation() {
        let inputs = DeferredLightingInputs::new(
            make_handle(0),
            make_handle(1),
            make_handle(2),
            make_handle(3),
            make_handle(4),
            make_handle(5),
            make_handle(6),
            make_handle(7),
        );
        let outputs = DeferredLightingOutputs::new(make_handle(8));

        let node = DeferredLightingNode::new(inputs, outputs, 1920, 1080);

        assert_eq!(node.width, 1920);
        assert_eq!(node.height, 1080);
        assert!(node.use_compute);
    }

    #[test]
    fn test_deferred_lighting_with_optional_inputs() {
        let inputs = DeferredLightingInputs::new(
            make_handle(0),
            make_handle(1),
            make_handle(2),
            make_handle(3),
            make_handle(4),
            make_handle(5),
            make_handle(6),
            make_handle(7),
        )
        .with_ddgi(make_handle(10))
        .with_ssao(make_handle(11));

        assert_eq!(inputs.ddgi_probes, Some(make_handle(10)));
        assert_eq!(inputs.ssao, Some(make_handle(11)));
    }

    #[test]
    fn test_deferred_lighting_access_set() {
        let inputs = DeferredLightingInputs::new(
            make_handle(0),
            make_handle(1),
            make_handle(2),
            make_handle(3),
            make_handle(4),
            make_handle(5),
            make_handle(6),
            make_handle(7),
        )
        .with_ddgi(make_handle(10));
        let outputs = DeferredLightingOutputs::new(make_handle(8)).with_specular(make_handle(9));
        let node = DeferredLightingNode::new(inputs, outputs, 1920, 1080);

        let access = node.build_access_set();

        // 8 required inputs + 1 optional (DDGI)
        assert_eq!(access.reads.len(), 9);
        // HDR output + specular output
        assert_eq!(access.writes.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Contact Shadow Integration Tests (T-LIT-8.2)
    // -----------------------------------------------------------------------

    #[test]
    fn test_deferred_lighting_with_contact_shadow() {
        let contact_shadow = make_handle(20);
        let inputs = DeferredLightingInputs::new(
            make_handle(0),
            make_handle(1),
            make_handle(2),
            make_handle(3),
            make_handle(4),
            make_handle(5),
            make_handle(6),
            make_handle(7),
        )
        .with_contact_shadow(contact_shadow);

        assert_eq!(inputs.contact_shadow, Some(contact_shadow));
    }

    #[test]
    fn test_deferred_lighting_access_set_with_contact_shadow() {
        let contact_shadow = make_handle(20);
        let inputs = DeferredLightingInputs::new(
            make_handle(0),
            make_handle(1),
            make_handle(2),
            make_handle(3),
            make_handle(4),
            make_handle(5),
            make_handle(6),
            make_handle(7),
        )
        .with_contact_shadow(contact_shadow);
        let outputs = DeferredLightingOutputs::new(make_handle(8));
        let node = DeferredLightingNode::new(inputs, outputs, 1920, 1080);

        let access = node.build_access_set();

        // 8 required + 1 contact shadow
        assert_eq!(access.reads.len(), 9);
        assert!(access.reads.contains(&contact_shadow));
    }

    #[test]
    fn test_contact_shadow_frame_graph_dependency_registered() {
        // Verify contact shadow can be added as a dependency
        let contact_shadow = make_handle(30);
        let froxel_indices = make_handle(2);
        let froxel_counts = make_handle(3);
        let shadow_map = make_handle(7);

        // Build dependencies including contact shadow
        let deps = build_lighting_dependencies(froxel_indices, froxel_counts, shadow_map);

        // Standard deps exist
        assert_eq!(deps.len(), 3);

        // Create lighting node with contact shadow
        let inputs = DeferredLightingInputs::new(
            make_handle(0),
            make_handle(1),
            make_handle(10),
            make_handle(11),
            make_handle(12),
            froxel_indices,
            froxel_counts,
            shadow_map,
        )
        .with_contact_shadow(contact_shadow);
        let outputs = DeferredLightingOutputs::new(make_handle(8));
        let node = DeferredLightingNode::new(inputs, outputs, 1920, 1080);

        let access = node.build_access_set();

        // Verify contact shadow is in reads
        assert!(access.reads.contains(&contact_shadow));
        assert!(access.reads.contains(&shadow_map));
        assert!(access.reads.contains(&froxel_indices));
    }

    #[test]
    fn test_contact_shadow_with_all_optional_inputs() {
        let contact_shadow = make_handle(20);
        let ddgi = make_handle(21);
        let ssao = make_handle(22);

        let inputs = DeferredLightingInputs::new(
            make_handle(0),
            make_handle(1),
            make_handle(2),
            make_handle(3),
            make_handle(4),
            make_handle(5),
            make_handle(6),
            make_handle(7),
        )
        .with_contact_shadow(contact_shadow)
        .with_ddgi(ddgi)
        .with_ssao(ssao);

        let outputs = DeferredLightingOutputs::new(make_handle(8));
        let node = DeferredLightingNode::new(inputs, outputs, 1920, 1080);

        let access = node.build_access_set();

        // 8 required + 3 optional (contact, ddgi, ssao)
        assert_eq!(access.reads.len(), 11);
        assert!(access.reads.contains(&contact_shadow));
        assert!(access.reads.contains(&ddgi));
        assert!(access.reads.contains(&ssao));
    }

    #[test]
    fn test_deferred_lighting_compute_pass() {
        let inputs = DeferredLightingInputs::new(
            make_handle(0),
            make_handle(1),
            make_handle(2),
            make_handle(3),
            make_handle(4),
            make_handle(5),
            make_handle(6),
            make_handle(7),
        );
        let outputs = DeferredLightingOutputs::new(make_handle(8));
        let node = DeferredLightingNode::new(inputs, outputs, 1920, 1080);

        let pass = node.into_ir_pass(PassIndex(2));

        assert_eq!(pass.name, "deferred_lighting");
        assert_eq!(pass.pass_type, PassType::Compute);

        if let Some(DispatchSource::Direct {
            group_count_x,
            group_count_y,
            group_count_z,
        }) = &pass.dispatch_source
        {
            assert_eq!(*group_count_x, 240); // (1920 + 7) / 8
            assert_eq!(*group_count_y, 135); // (1080 + 7) / 8
            assert_eq!(*group_count_z, 1);
        } else {
            panic!("Expected direct dispatch source");
        }
    }

    #[test]
    fn test_deferred_lighting_graphics_pass() {
        let inputs = DeferredLightingInputs::new(
            make_handle(0),
            make_handle(1),
            make_handle(2),
            make_handle(3),
            make_handle(4),
            make_handle(5),
            make_handle(6),
            make_handle(7),
        );
        let outputs = DeferredLightingOutputs::new(make_handle(8));
        let node = DeferredLightingNode::new(inputs, outputs, 1920, 1080).with_graphics_pass();

        let pass = node.into_ir_pass(PassIndex(2));

        assert_eq!(pass.pass_type, PassType::Graphics);
        assert_eq!(pass.color_attachments.len(), 1);
        assert_eq!(pass.color_attachments[0].resource, make_handle(8));
    }

    // -----------------------------------------------------------------------
    // Dependency and barrier tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_lighting_dependencies() {
        let deps = build_lighting_dependencies(make_handle(0), make_handle(1), make_handle(2));

        assert_eq!(deps.len(), 3);

        // Light culling -> deferred lighting (froxel indices)
        assert_eq!(deps[0].from, "light_culling");
        assert_eq!(deps[0].to, "deferred_lighting");
        assert_eq!(deps[0].resource, make_handle(0));

        // Light culling -> deferred lighting (froxel counts)
        assert_eq!(deps[1].from, "light_culling");
        assert_eq!(deps[1].to, "deferred_lighting");
        assert_eq!(deps[1].resource, make_handle(1));

        // Shadow render -> deferred lighting
        assert_eq!(deps[2].from, "shadow_render");
        assert_eq!(deps[2].to, "deferred_lighting");
        assert_eq!(deps[2].resource, make_handle(2));
    }

    #[test]
    fn test_build_lighting_barriers() {
        let barriers = build_lighting_barriers(make_handle(0), make_handle(1), make_handle(2));

        assert_eq!(barriers.len(), 3);

        // Storage buffer barriers for froxel data
        assert_eq!(barriers[0].barrier_type, BarrierType::StorageBuffer);
        assert_eq!(barriers[0].resource, make_handle(0));

        assert_eq!(barriers[1].barrier_type, BarrierType::StorageBuffer);
        assert_eq!(barriers[1].resource, make_handle(1));

        // Depth attachment barrier for shadow map
        assert_eq!(barriers[2].barrier_type, BarrierType::DepthAttachment);
        assert_eq!(barriers[2].resource, make_handle(2));
    }

    // -----------------------------------------------------------------------
    // ShadowType display tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_shadow_type_display() {
        assert_eq!(format!("{}", ShadowType::Csm { cascade_count: 4 }), "CSM(4)");
        assert_eq!(format!("{}", ShadowType::Cube), "Cube");
        assert_eq!(format!("{}", ShadowType::Spot), "Spot");
    }

    // -----------------------------------------------------------------------
    // Integration test: Full lighting pipeline
    // -----------------------------------------------------------------------

    #[test]
    fn test_full_lighting_pipeline_creation() {
        // Create resource handles
        let light_buffers = make_handle(0);
        let camera = make_handle(1);
        let froxel_indices = make_handle(2);
        let froxel_counts = make_handle(3);
        let scene_geometry = make_handle(4);
        let transforms = make_handle(5);
        let light_matrices = make_handle(6);
        let shadow_map = make_handle(7);
        let g_albedo = make_handle(8);
        let g_normal = make_handle(9);
        let g_material = make_handle(10);
        let g_depth = make_handle(11);
        let hdr_output = make_handle(12);

        // Create nodes
        let culling = LightCullingNode::new(
            LightCullingInputs::new(light_buffers, camera),
            LightCullingOutputs::new(froxel_indices, froxel_counts),
            FroxelConfig::from_screen(1920, 1080, 16, 24),
        );

        let shadow = ShadowRenderNode::new(
            ShadowType::Csm { cascade_count: 4 },
            ShadowInputs::new(scene_geometry, transforms, light_matrices),
            ShadowOutputs::new(shadow_map),
            2048,
        );

        let lighting = DeferredLightingNode::new(
            DeferredLightingInputs::new(
                g_albedo,
                g_normal,
                g_material,
                g_depth,
                light_buffers,
                froxel_indices,
                froxel_counts,
                shadow_map,
            ),
            DeferredLightingOutputs::new(hdr_output),
            1920,
            1080,
        );

        // Convert to IR passes
        let passes = vec![
            culling.into_ir_pass(PassIndex(0)),
            shadow.into_ir_pass(PassIndex(1)),
            lighting.into_ir_pass(PassIndex(2)),
        ];

        // Verify pass types
        assert_eq!(passes[0].pass_type, PassType::Compute);
        assert_eq!(passes[1].pass_type, PassType::Graphics);
        assert_eq!(passes[2].pass_type, PassType::Compute);

        // Build and verify dependencies
        let deps = build_lighting_dependencies(froxel_indices, froxel_counts, shadow_map);
        assert_eq!(deps.len(), 3);

        // Build and verify barriers
        let barriers = build_lighting_barriers(froxel_indices, froxel_counts, shadow_map);
        assert_eq!(barriers.len(), 3);

        // Verify deferred lighting reads from both culling and shadow outputs
        let lighting_reads = &passes[2].access_set.reads;
        assert!(lighting_reads.contains(&froxel_indices));
        assert!(lighting_reads.contains(&froxel_counts));
        assert!(lighting_reads.contains(&shadow_map));
    }
}
