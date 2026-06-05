//! Demoscene Frame Graph Integration (T-DEMO-6.1 / T-DEMO-6.2)
//!
//! This module provides frame graph integration for demoscene rendering:
//! - **DemoscenePass**: Frame graph compute pass for S13 ray marching
//! - **Full-screen mode**: Pure demoscene rendering without rasterization
//!
//! # Overview
//!
//! The demoscene pass is a compute pass that writes to every pixel using ray marching.
//! It can operate in two modes:
//!
//! 1. **Composited mode**: S13 pass is one of many passes in the frame graph
//! 2. **Full-screen mode**: S13 is the only active pass, writes directly to swapchain
//!
//! # Frame Graph Integration
//!
//! ```ignore
//! let mut builder = RenderGraphBuilder::new();
//!
//! // Create demoscene output texture
//! let output = builder.create_texture("demoscene_output", 1920, 1080, "rgba8unorm");
//!
//! // Add demoscene pass
//! let pass = DemoscenePass::new(1920, 1080, output);
//! builder.add_demoscene_pass("s13_raymarch", pass);
//!
//! let (passes, resources) = builder.finalize();
//! ```
//!
//! # Full-Screen Mode
//!
//! When `demoscene_fullscreen` is enabled, the frame graph compiler skips all
//! rasterization passes and only executes the demoscene compute pass:
//!
//! ```ignore
//! let config = FrameGraphConfig {
//!     demoscene_fullscreen: true,
//!     ..Default::default()
//! };
//! let compiled = compiler.compile_with_config(&builder.finalize(), config);
//! ```

use core::fmt;
use std::sync::Arc;

use crate::frame_graph::{
    BindGroup, DispatchSource, IrPass, IrResource, PassFlags, PassIndex, PassType,
    RenderContext, ResourceAccessSet, ResourceDesc, ResourceHandle, ResourceLifetime,
    ResourceState, TextureDesc, View, ViewType,
};

// ---------------------------------------------------------------------------
// Demoscene Pass Configuration
// ---------------------------------------------------------------------------

/// Configuration for the demoscene frame graph pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DemoscenePassConfig {
    /// Output texture width in pixels.
    pub width: u32,

    /// Output texture height in pixels.
    pub height: u32,

    /// Workgroup size X (must match shader @workgroup_size).
    pub workgroup_size_x: u32,

    /// Workgroup size Y (must match shader @workgroup_size).
    pub workgroup_size_y: u32,

    /// Enable full-screen mode (no rasterization passes).
    pub fullscreen: bool,

    /// Pass priority for scheduling (lower = earlier in frame).
    pub priority: i32,

    /// Enable temporal accumulation (for progressive rendering).
    pub temporal_accumulation: bool,

    /// Maximum ray march steps (for quality/performance trade-off).
    pub max_steps: u32,
}

impl Default for DemoscenePassConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            workgroup_size_x: 8,
            workgroup_size_y: 8,
            fullscreen: false,
            priority: 0,
            temporal_accumulation: false,
            max_steps: 64,
        }
    }
}

impl DemoscenePassConfig {
    /// Create a new configuration with the given dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            ..Default::default()
        }
    }

    /// Create a full-screen mode configuration.
    pub fn fullscreen(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            fullscreen: true,
            ..Default::default()
        }
    }

    /// Set custom workgroup size.
    pub fn with_workgroup_size(mut self, x: u32, y: u32) -> Self {
        self.workgroup_size_x = x;
        self.workgroup_size_y = y;
        self
    }

    /// Enable temporal accumulation.
    pub fn with_temporal_accumulation(mut self) -> Self {
        self.temporal_accumulation = true;
        self
    }

    /// Set maximum ray march steps.
    pub fn with_max_steps(mut self, steps: u32) -> Self {
        self.max_steps = steps;
        self
    }

    /// Calculate dispatch workgroup counts.
    pub fn dispatch_size(&self) -> (u32, u32, u32) {
        let x = (self.width + self.workgroup_size_x - 1) / self.workgroup_size_x;
        let y = (self.height + self.workgroup_size_y - 1) / self.workgroup_size_y;
        (x, y, 1)
    }

    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), DemoscenePassError> {
        if self.width == 0 {
            return Err(DemoscenePassError::InvalidWidth(self.width));
        }
        if self.height == 0 {
            return Err(DemoscenePassError::InvalidHeight(self.height));
        }
        if self.workgroup_size_x == 0 || self.workgroup_size_x > 256 {
            return Err(DemoscenePassError::InvalidWorkgroupSize(
                self.workgroup_size_x,
                self.workgroup_size_y,
            ));
        }
        if self.workgroup_size_y == 0 || self.workgroup_size_y > 256 {
            return Err(DemoscenePassError::InvalidWorkgroupSize(
                self.workgroup_size_x,
                self.workgroup_size_y,
            ));
        }
        if self.max_steps == 0 {
            return Err(DemoscenePassError::InvalidMaxSteps(self.max_steps));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Demoscene Pass Error
// ---------------------------------------------------------------------------

/// Errors that can occur during demoscene pass operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DemoscenePassError {
    /// Invalid output width (must be > 0).
    InvalidWidth(u32),

    /// Invalid output height (must be > 0).
    InvalidHeight(u32),

    /// Invalid workgroup size (must be 1-256).
    InvalidWorkgroupSize(u32, u32),

    /// Invalid max steps (must be > 0).
    InvalidMaxSteps(u32),

    /// Output resource not found.
    OutputResourceNotFound(ResourceHandle),

    /// Invalid output resource format.
    InvalidOutputFormat(String),

    /// Pass already registered.
    PassAlreadyRegistered(String),

    /// Full-screen mode conflict with rasterization passes.
    FullscreenConflict(String),
}

impl fmt::Display for DemoscenePassError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidWidth(w) => write!(f, "invalid width: {} (must be > 0)", w),
            Self::InvalidHeight(h) => write!(f, "invalid height: {} (must be > 0)", h),
            Self::InvalidWorkgroupSize(x, y) => {
                write!(f, "invalid workgroup size: {}x{} (must be 1-256)", x, y)
            }
            Self::InvalidMaxSteps(s) => write!(f, "invalid max steps: {} (must be > 0)", s),
            Self::OutputResourceNotFound(h) => write!(f, "output resource not found: {}", h),
            Self::InvalidOutputFormat(fmt) => write!(f, "invalid output format: {}", fmt),
            Self::PassAlreadyRegistered(name) => write!(f, "pass already registered: {}", name),
            Self::FullscreenConflict(msg) => write!(f, "full-screen mode conflict: {}", msg),
        }
    }
}

impl std::error::Error for DemoscenePassError {}

// ---------------------------------------------------------------------------
// Demoscene View
// ---------------------------------------------------------------------------

/// View descriptor for the demoscene output texture.
///
/// This view is used by the frame graph to track the demoscene pass's
/// output resource binding.
#[derive(Clone, Debug)]
pub struct DemosceneView {
    /// Human-readable label.
    pub name: String,

    /// Output texture width.
    pub width: u32,

    /// Output texture height.
    pub height: u32,

    /// Output texture format (e.g., "rgba8unorm").
    pub format: String,

    /// Whether this is a transient (frame-local) resource.
    pub transient: bool,

    /// Animation time for the current frame.
    pub time: f32,
}

impl View for DemosceneView {
    fn view_type(&self) -> ViewType {
        ViewType::Storage
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn is_transient(&self) -> bool {
        self.transient
    }

    fn bind(&self, _ctx: &RenderContext) -> Vec<BindGroup> {
        vec![
            BindGroup(format!("{}_uniforms", self.name)),
            BindGroup(format!("{}_output", self.name)),
        ]
    }
}

impl Default for DemosceneView {
    fn default() -> Self {
        Self {
            name: "demoscene".to_string(),
            width: 1920,
            height: 1080,
            format: "rgba8unorm".to_string(),
            transient: true,
            time: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Demoscene Pass
// ---------------------------------------------------------------------------

/// Demoscene compute pass for the frame graph.
///
/// This pass implements S13 ray marching as a compute pass that writes
/// to every pixel in the output texture. It can operate in full-screen
/// mode where no rasterization passes are executed.
///
/// # Pass Characteristics
///
/// - **Type**: Compute
/// - **Inputs**: None (uses embedded uniforms)
/// - **Outputs**: Color texture (storage image)
/// - **Dispatch**: Full-screen coverage with workgroups
///
/// # Full-Screen Mode
///
/// When `fullscreen` is enabled:
/// - All graphics passes are culled
/// - S13 writes directly to swapchain or final output
/// - No depth testing or blending
/// - Maximum GPU utilization for demoscene effects
#[derive(Debug, Clone)]
pub struct DemoscenePass {
    /// Pass configuration.
    pub config: DemoscenePassConfig,

    /// Output texture handle.
    pub output: ResourceHandle,

    /// Optional input texture for compositing.
    pub input: Option<ResourceHandle>,

    /// Pass name.
    pub name: String,

    /// Pass tags for filtering.
    pub tags: Vec<String>,
}

impl DemoscenePass {
    /// Create a new demoscene pass with default configuration.
    pub fn new(width: u32, height: u32, output: ResourceHandle) -> Self {
        Self {
            config: DemoscenePassConfig::new(width, height),
            output,
            input: None,
            name: "demoscene".to_string(),
            tags: vec!["demoscene".to_string(), "compute".to_string()],
        }
    }

    /// Create a new demoscene pass with custom configuration.
    pub fn with_config(config: DemoscenePassConfig, output: ResourceHandle) -> Self {
        Self {
            config,
            output,
            input: None,
            name: "demoscene".to_string(),
            tags: vec!["demoscene".to_string(), "compute".to_string()],
        }
    }

    /// Create a full-screen demoscene pass.
    pub fn fullscreen(width: u32, height: u32, output: ResourceHandle) -> Self {
        Self {
            config: DemoscenePassConfig::fullscreen(width, height),
            output,
            input: None,
            name: "demoscene_fullscreen".to_string(),
            tags: vec![
                "demoscene".to_string(),
                "compute".to_string(),
                "fullscreen".to_string(),
            ],
        }
    }

    /// Set the pass name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Add a tag to the pass.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Set an input texture for compositing.
    pub fn with_input(mut self, input: ResourceHandle) -> Self {
        self.input = Some(input);
        self
    }

    /// Check if this is a full-screen pass.
    pub fn is_fullscreen(&self) -> bool {
        self.config.fullscreen
    }

    /// Get the dispatch workgroup counts.
    pub fn dispatch_size(&self) -> (u32, u32, u32) {
        self.config.dispatch_size()
    }

    /// Validate the pass configuration.
    pub fn validate(&self) -> Result<(), DemoscenePassError> {
        self.config.validate()
    }

    /// Convert to an IrPass for frame graph compilation.
    pub fn to_ir_pass(&self, index: PassIndex) -> IrPass {
        let (gx, gy, gz) = self.dispatch_size();

        let mut reads = Vec::new();
        let writes = vec![self.output];

        if let Some(input) = self.input {
            reads.push(input);
        }

        let view = Arc::new(DemosceneView {
            name: self.name.clone(),
            width: self.config.width,
            height: self.config.height,
            format: "rgba8unorm".to_string(),
            transient: true,
            time: 0.0,
        });

        let mut flags = PassFlags::SIDE_EFFECTS;
        if self.is_fullscreen() {
            flags = flags | PassFlags::NO_CULL;
        }

        IrPass {
            index,
            name: self.name.clone(),
            pass_type: PassType::Compute,
            access_set: ResourceAccessSet { reads, writes },
            color_attachments: Vec::new(),
            depth_stencil: None,
            instance_source: crate::frame_graph::InstanceSource::Direct {
                index_count: 0,
                instance_count: 1,
                base_vertex: 0,
                first_index: 0,
                first_instance: 0,
            },
            dispatch_source: Some(DispatchSource::Direct {
                group_count_x: gx,
                group_count_y: gy,
                group_count_z: gz,
            }),
            view_type: ViewType::Storage,
            view,
            tags: self.tags.clone(),
            flags,
        }
    }

    /// Create the output texture resource for the pass.
    pub fn create_output_resource(&self, handle: ResourceHandle) -> IrResource {
        IrResource::new(
            handle,
            format!("{}_output", self.name),
            ResourceDesc::Texture2D(TextureDesc {
                width: self.config.width,
                height: self.config.height,
                mip_levels: 1,
                array_layers: 1,
                format: "rgba8unorm".to_string(),
            }),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        )
    }
}

// ---------------------------------------------------------------------------
// Frame Graph Configuration Extension
// ---------------------------------------------------------------------------

/// Configuration for demoscene frame graph compilation.
#[derive(Debug, Clone, Default)]
pub struct DemosceneFrameGraphConfig {
    /// Enable full-screen demoscene mode (skip rasterization passes).
    pub demoscene_fullscreen: bool,

    /// List of demoscene passes to include.
    pub passes: Vec<DemoscenePass>,

    /// Priority offset for demoscene passes.
    pub priority_offset: i32,

    /// Enable debug output.
    pub debug: bool,
}

impl DemosceneFrameGraphConfig {
    /// Create a new demoscene frame graph configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable full-screen mode.
    pub fn with_fullscreen(mut self) -> Self {
        self.demoscene_fullscreen = true;
        self
    }

    /// Add a demoscene pass.
    pub fn with_pass(mut self, pass: DemoscenePass) -> Self {
        self.passes.push(pass);
        self
    }

    /// Set priority offset.
    pub fn with_priority_offset(mut self, offset: i32) -> Self {
        self.priority_offset = offset;
        self
    }

    /// Enable debug output.
    pub fn with_debug(mut self) -> Self {
        self.debug = true;
        self
    }

    /// Check if full-screen mode is enabled.
    pub fn is_fullscreen(&self) -> bool {
        self.demoscene_fullscreen || self.passes.iter().any(|p| p.is_fullscreen())
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), DemoscenePassError> {
        for pass in &self.passes {
            pass.validate()?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Full-Screen Mode Filter
// ---------------------------------------------------------------------------

/// Filter for identifying and handling full-screen demoscene mode.
///
/// When full-screen mode is active, this filter marks all non-demoscene
/// passes for culling, ensuring only the S13 ray march pass executes.
#[derive(Debug, Clone, Default)]
pub struct FullscreenModeFilter {
    /// Whether full-screen mode is active.
    pub active: bool,

    /// Pass names that should be preserved (demoscene passes).
    pub preserved_passes: Vec<String>,
}

impl FullscreenModeFilter {
    /// Create a new full-screen mode filter.
    pub fn new(active: bool) -> Self {
        Self {
            active,
            preserved_passes: Vec::new(),
        }
    }

    /// Add a pass to the preserved list.
    pub fn preserve(&mut self, pass_name: impl Into<String>) {
        self.preserved_passes.push(pass_name.into());
    }

    /// Check if a pass should be culled in full-screen mode.
    pub fn should_cull(&self, pass: &IrPass) -> bool {
        if !self.active {
            return false;
        }

        // Never cull demoscene passes
        if pass.tags.iter().any(|t| t == "demoscene") {
            return false;
        }

        // Never cull explicitly preserved passes
        if self.preserved_passes.contains(&pass.name) {
            return false;
        }

        // Never cull passes with SIDE_EFFECTS or NO_CULL flags
        if pass.flags.is_uncullable() {
            return false;
        }

        // Cull all graphics passes in full-screen mode
        pass.pass_type == PassType::Graphics
    }

    /// Filter passes for full-screen mode.
    pub fn filter_passes(&self, passes: &[IrPass]) -> Vec<IrPass> {
        if !self.active {
            return passes.to_vec();
        }

        passes
            .iter()
            .filter(|p| !self.should_cull(p))
            .cloned()
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Pass Registration
// ---------------------------------------------------------------------------

/// Registry for demoscene passes in the frame graph.
#[derive(Debug, Default)]
pub struct DemoscenePassRegistry {
    /// Registered passes by name.
    passes: std::collections::HashMap<String, DemoscenePass>,

    /// Next pass index to assign.
    next_index: usize,
}

impl DemoscenePassRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a demoscene pass.
    pub fn register(&mut self, pass: DemoscenePass) -> Result<(), DemoscenePassError> {
        if self.passes.contains_key(&pass.name) {
            return Err(DemoscenePassError::PassAlreadyRegistered(pass.name.clone()));
        }
        pass.validate()?;
        self.passes.insert(pass.name.clone(), pass);
        self.next_index += 1;
        Ok(())
    }

    /// Get a registered pass by name.
    pub fn get(&self, name: &str) -> Option<&DemoscenePass> {
        self.passes.get(name)
    }

    /// Get all registered passes.
    pub fn passes(&self) -> impl Iterator<Item = &DemoscenePass> {
        self.passes.values()
    }

    /// Get the number of registered passes.
    pub fn len(&self) -> usize {
        self.passes.len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.passes.is_empty()
    }

    /// Convert all registered passes to IR passes.
    pub fn to_ir_passes(&self, start_index: usize) -> Vec<IrPass> {
        self.passes
            .values()
            .enumerate()
            .map(|(i, p)| p.to_ir_pass(PassIndex(start_index + i)))
            .collect()
    }

    /// Collect output resources from all passes.
    pub fn collect_resources(&self, start_handle: u32) -> Vec<IrResource> {
        self.passes
            .values()
            .enumerate()
            .map(|(i, p)| p.create_output_resource(ResourceHandle(start_handle + i as u32)))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Execution Order
// ---------------------------------------------------------------------------

/// Determines execution order for demoscene passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DemosceneExecutionOrder {
    /// Execute before all other passes.
    First,

    /// Execute after all other passes (default for compositing).
    Last,

    /// Execute at a specific priority level.
    Priority(i32),
}

impl Default for DemosceneExecutionOrder {
    fn default() -> Self {
        Self::Last
    }
}

impl DemosceneExecutionOrder {
    /// Convert to a priority value for sorting.
    pub fn to_priority(&self) -> i32 {
        match self {
            Self::First => i32::MIN,
            Self::Last => i32::MAX,
            Self::Priority(p) => *p,
        }
    }
}

// ---------------------------------------------------------------------------
// Output Binding
// ---------------------------------------------------------------------------

/// Describes how the demoscene output is bound in the frame graph.
#[derive(Debug, Clone)]
pub struct DemosceneOutputBinding {
    /// Resource handle for the output texture.
    pub handle: ResourceHandle,

    /// Binding slot for the storage image.
    pub binding: u32,

    /// Whether to write directly to swapchain.
    pub direct_to_swapchain: bool,

    /// Format of the output texture.
    pub format: String,
}

impl DemosceneOutputBinding {
    /// Create a new output binding.
    pub fn new(handle: ResourceHandle) -> Self {
        Self {
            handle,
            binding: 0,
            direct_to_swapchain: false,
            format: "rgba8unorm".to_string(),
        }
    }

    /// Set the binding slot.
    pub fn with_binding(mut self, binding: u32) -> Self {
        self.binding = binding;
        self
    }

    /// Enable direct swapchain write.
    pub fn with_direct_swapchain(mut self) -> Self {
        self.direct_to_swapchain = true;
        self
    }

    /// Set the output format.
    pub fn with_format(mut self, format: impl Into<String>) -> Self {
        self.format = format.into();
        self
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // DemoscenePassConfig Tests
    // =========================================================================

    #[test]
    fn test_config_default() {
        let config = DemoscenePassConfig::default();
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.workgroup_size_x, 8);
        assert_eq!(config.workgroup_size_y, 8);
        assert!(!config.fullscreen);
        assert_eq!(config.priority, 0);
        assert!(!config.temporal_accumulation);
        assert_eq!(config.max_steps, 64);
    }

    #[test]
    fn test_config_new() {
        let config = DemoscenePassConfig::new(800, 600);
        assert_eq!(config.width, 800);
        assert_eq!(config.height, 600);
    }

    #[test]
    fn test_config_fullscreen() {
        let config = DemoscenePassConfig::fullscreen(1920, 1080);
        assert!(config.fullscreen);
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
    }

    #[test]
    fn test_config_with_workgroup_size() {
        let config = DemoscenePassConfig::new(800, 600).with_workgroup_size(16, 16);
        assert_eq!(config.workgroup_size_x, 16);
        assert_eq!(config.workgroup_size_y, 16);
    }

    #[test]
    fn test_config_with_temporal_accumulation() {
        let config = DemoscenePassConfig::new(800, 600).with_temporal_accumulation();
        assert!(config.temporal_accumulation);
    }

    #[test]
    fn test_config_with_max_steps() {
        let config = DemoscenePassConfig::new(800, 600).with_max_steps(128);
        assert_eq!(config.max_steps, 128);
    }

    #[test]
    fn test_config_dispatch_size() {
        let config = DemoscenePassConfig::new(800, 600);
        let (x, y, z) = config.dispatch_size();
        assert_eq!(x, 100); // 800 / 8
        assert_eq!(y, 75);  // 600 / 8
        assert_eq!(z, 1);
    }

    #[test]
    fn test_config_dispatch_size_non_divisible() {
        let config = DemoscenePassConfig::new(805, 603);
        let (x, y, z) = config.dispatch_size();
        assert_eq!(x, 101); // ceil(805 / 8)
        assert_eq!(y, 76);  // ceil(603 / 8)
        assert_eq!(z, 1);
    }

    #[test]
    fn test_config_validate_success() {
        let config = DemoscenePassConfig::new(800, 600);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validate_zero_width() {
        let mut config = DemoscenePassConfig::new(0, 600);
        config.width = 0;
        let result = config.validate();
        assert!(matches!(result, Err(DemoscenePassError::InvalidWidth(0))));
    }

    #[test]
    fn test_config_validate_zero_height() {
        let mut config = DemoscenePassConfig::new(800, 0);
        config.height = 0;
        let result = config.validate();
        assert!(matches!(result, Err(DemoscenePassError::InvalidHeight(0))));
    }

    #[test]
    fn test_config_validate_invalid_workgroup_x() {
        let mut config = DemoscenePassConfig::new(800, 600);
        config.workgroup_size_x = 0;
        let result = config.validate();
        assert!(matches!(
            result,
            Err(DemoscenePassError::InvalidWorkgroupSize(0, _))
        ));
    }

    #[test]
    fn test_config_validate_invalid_workgroup_y() {
        let mut config = DemoscenePassConfig::new(800, 600);
        config.workgroup_size_y = 0;
        let result = config.validate();
        assert!(matches!(
            result,
            Err(DemoscenePassError::InvalidWorkgroupSize(_, 0))
        ));
    }

    #[test]
    fn test_config_validate_invalid_max_steps() {
        let mut config = DemoscenePassConfig::new(800, 600);
        config.max_steps = 0;
        let result = config.validate();
        assert!(matches!(
            result,
            Err(DemoscenePassError::InvalidMaxSteps(0))
        ));
    }

    // =========================================================================
    // DemoscenePassError Tests
    // =========================================================================

    #[test]
    fn test_error_display() {
        let errors = vec![
            (
                DemoscenePassError::InvalidWidth(0),
                "invalid width: 0 (must be > 0)",
            ),
            (
                DemoscenePassError::InvalidHeight(0),
                "invalid height: 0 (must be > 0)",
            ),
            (
                DemoscenePassError::InvalidWorkgroupSize(300, 8),
                "invalid workgroup size: 300x8 (must be 1-256)",
            ),
            (
                DemoscenePassError::InvalidMaxSteps(0),
                "invalid max steps: 0 (must be > 0)",
            ),
            (
                DemoscenePassError::OutputResourceNotFound(ResourceHandle(42)),
                "output resource not found: ResourceHandle(42)",
            ),
            (
                DemoscenePassError::InvalidOutputFormat("bc7".to_string()),
                "invalid output format: bc7",
            ),
            (
                DemoscenePassError::PassAlreadyRegistered("demo".to_string()),
                "pass already registered: demo",
            ),
            (
                DemoscenePassError::FullscreenConflict("test".to_string()),
                "full-screen mode conflict: test",
            ),
        ];

        for (error, expected) in errors {
            assert_eq!(error.to_string(), expected);
        }
    }

    // =========================================================================
    // DemosceneView Tests
    // =========================================================================

    #[test]
    fn test_view_default() {
        let view = DemosceneView::default();
        assert_eq!(view.name, "demoscene");
        assert_eq!(view.width, 1920);
        assert_eq!(view.height, 1080);
        assert_eq!(view.format, "rgba8unorm");
        assert!(view.transient);
        assert_eq!(view.time, 0.0);
    }

    #[test]
    fn test_view_view_type() {
        let view = DemosceneView::default();
        assert_eq!(view.view_type(), ViewType::Storage);
    }

    #[test]
    fn test_view_name() {
        let view = DemosceneView {
            name: "custom".to_string(),
            ..Default::default()
        };
        assert_eq!(view.name(), "custom");
    }

    #[test]
    fn test_view_is_transient() {
        let mut view = DemosceneView::default();
        assert!(view.is_transient());

        view.transient = false;
        assert!(!view.is_transient());
    }

    #[test]
    fn test_view_bind() {
        let view = DemosceneView {
            name: "demo".to_string(),
            ..Default::default()
        };
        let ctx = RenderContext { frame_index: 0 };
        let bindings = view.bind(&ctx);
        assert_eq!(bindings.len(), 2);
        assert_eq!(bindings[0].0, "demo_uniforms");
        assert_eq!(bindings[1].0, "demo_output");
    }

    // =========================================================================
    // DemoscenePass Tests
    // =========================================================================

    #[test]
    fn test_pass_new() {
        let output = ResourceHandle(0);
        let pass = DemoscenePass::new(800, 600, output);
        assert_eq!(pass.config.width, 800);
        assert_eq!(pass.config.height, 600);
        assert_eq!(pass.output, output);
        assert!(pass.input.is_none());
        assert_eq!(pass.name, "demoscene");
        assert!(pass.tags.contains(&"demoscene".to_string()));
        assert!(pass.tags.contains(&"compute".to_string()));
    }

    #[test]
    fn test_pass_with_config() {
        let config = DemoscenePassConfig::fullscreen(1920, 1080);
        let output = ResourceHandle(0);
        let pass = DemoscenePass::with_config(config, output);
        assert!(pass.is_fullscreen());
    }

    #[test]
    fn test_pass_fullscreen() {
        let output = ResourceHandle(0);
        let pass = DemoscenePass::fullscreen(1920, 1080, output);
        assert!(pass.is_fullscreen());
        assert_eq!(pass.name, "demoscene_fullscreen");
        assert!(pass.tags.contains(&"fullscreen".to_string()));
    }

    #[test]
    fn test_pass_with_name() {
        let output = ResourceHandle(0);
        let pass = DemoscenePass::new(800, 600, output).with_name("custom_pass");
        assert_eq!(pass.name, "custom_pass");
    }

    #[test]
    fn test_pass_with_tag() {
        let output = ResourceHandle(0);
        let pass = DemoscenePass::new(800, 600, output).with_tag("custom_tag");
        assert!(pass.tags.contains(&"custom_tag".to_string()));
    }

    #[test]
    fn test_pass_with_input() {
        let output = ResourceHandle(0);
        let input = ResourceHandle(1);
        let pass = DemoscenePass::new(800, 600, output).with_input(input);
        assert_eq!(pass.input, Some(input));
    }

    #[test]
    fn test_pass_dispatch_size() {
        let output = ResourceHandle(0);
        let pass = DemoscenePass::new(800, 600, output);
        let (x, y, z) = pass.dispatch_size();
        assert_eq!(x, 100);
        assert_eq!(y, 75);
        assert_eq!(z, 1);
    }

    #[test]
    fn test_pass_validate() {
        let output = ResourceHandle(0);
        let pass = DemoscenePass::new(800, 600, output);
        assert!(pass.validate().is_ok());
    }

    #[test]
    fn test_pass_to_ir_pass() {
        let output = ResourceHandle(0);
        let pass = DemoscenePass::new(800, 600, output);
        let ir_pass = pass.to_ir_pass(PassIndex(0));

        assert_eq!(ir_pass.index, PassIndex(0));
        assert_eq!(ir_pass.name, "demoscene");
        assert_eq!(ir_pass.pass_type, PassType::Compute);
        assert!(ir_pass.access_set.writes.contains(&output));
        assert!(ir_pass.dispatch_source.is_some());
        assert_eq!(ir_pass.view_type, ViewType::Storage);
    }

    #[test]
    fn test_pass_to_ir_pass_with_input() {
        let output = ResourceHandle(0);
        let input = ResourceHandle(1);
        let pass = DemoscenePass::new(800, 600, output).with_input(input);
        let ir_pass = pass.to_ir_pass(PassIndex(0));

        assert!(ir_pass.access_set.reads.contains(&input));
        assert!(ir_pass.access_set.writes.contains(&output));
    }

    #[test]
    fn test_pass_to_ir_pass_fullscreen_flags() {
        let output = ResourceHandle(0);
        let pass = DemoscenePass::fullscreen(800, 600, output);
        let ir_pass = pass.to_ir_pass(PassIndex(0));

        assert!(ir_pass.flags.has_no_cull());
        assert!(ir_pass.flags.has_side_effects());
    }

    #[test]
    fn test_pass_create_output_resource() {
        let handle = ResourceHandle(5);
        let pass = DemoscenePass::new(800, 600, ResourceHandle(0));
        let resource = pass.create_output_resource(handle);

        assert_eq!(resource.handle, handle);
        assert_eq!(resource.name, "demoscene_output");
        assert_eq!(resource.lifetime, ResourceLifetime::Transient);
        assert_eq!(resource.initial_state, ResourceState::Uninitialized);

        if let ResourceDesc::Texture2D(desc) = resource.desc {
            assert_eq!(desc.width, 800);
            assert_eq!(desc.height, 600);
            assert_eq!(desc.format, "rgba8unorm");
        } else {
            panic!("Expected Texture2D resource");
        }
    }

    // =========================================================================
    // DemosceneFrameGraphConfig Tests
    // =========================================================================

    #[test]
    fn test_fg_config_default() {
        let config = DemosceneFrameGraphConfig::default();
        assert!(!config.demoscene_fullscreen);
        assert!(config.passes.is_empty());
        assert_eq!(config.priority_offset, 0);
        assert!(!config.debug);
    }

    #[test]
    fn test_fg_config_with_fullscreen() {
        let config = DemosceneFrameGraphConfig::new().with_fullscreen();
        assert!(config.demoscene_fullscreen);
    }

    #[test]
    fn test_fg_config_with_pass() {
        let pass = DemoscenePass::new(800, 600, ResourceHandle(0));
        let config = DemosceneFrameGraphConfig::new().with_pass(pass);
        assert_eq!(config.passes.len(), 1);
    }

    #[test]
    fn test_fg_config_with_priority_offset() {
        let config = DemosceneFrameGraphConfig::new().with_priority_offset(-100);
        assert_eq!(config.priority_offset, -100);
    }

    #[test]
    fn test_fg_config_with_debug() {
        let config = DemosceneFrameGraphConfig::new().with_debug();
        assert!(config.debug);
    }

    #[test]
    fn test_fg_config_is_fullscreen() {
        let config = DemosceneFrameGraphConfig::new().with_fullscreen();
        assert!(config.is_fullscreen());

        let pass = DemoscenePass::fullscreen(800, 600, ResourceHandle(0));
        let config2 = DemosceneFrameGraphConfig::new().with_pass(pass);
        assert!(config2.is_fullscreen());
    }

    #[test]
    fn test_fg_config_validate() {
        let pass = DemoscenePass::new(800, 600, ResourceHandle(0));
        let config = DemosceneFrameGraphConfig::new().with_pass(pass);
        assert!(config.validate().is_ok());
    }

    // =========================================================================
    // FullscreenModeFilter Tests
    // =========================================================================

    #[test]
    fn test_filter_new() {
        let filter = FullscreenModeFilter::new(true);
        assert!(filter.active);
        assert!(filter.preserved_passes.is_empty());
    }

    #[test]
    fn test_filter_preserve() {
        let mut filter = FullscreenModeFilter::new(true);
        filter.preserve("important_pass");
        assert!(filter.preserved_passes.contains(&"important_pass".to_string()));
    }

    #[test]
    fn test_filter_should_cull_inactive() {
        let filter = FullscreenModeFilter::new(false);
        let pass = IrPass::graphics(
            PassIndex(0),
            "test",
            Vec::new(),
            None,
            crate::frame_graph::InstanceSource::Direct {
                index_count: 0,
                instance_count: 1,
                base_vertex: 0,
                first_index: 0,
                first_instance: 0,
            },
            ViewType::ColorAttachment,
        );
        assert!(!filter.should_cull(&pass));
    }

    #[test]
    fn test_filter_should_cull_demoscene_tag() {
        let filter = FullscreenModeFilter::new(true);
        let mut pass = IrPass::compute(
            PassIndex(0),
            "demoscene",
            DispatchSource::Direct {
                group_count_x: 10,
                group_count_y: 10,
                group_count_z: 1,
            },
            ViewType::Storage,
        );
        pass.tags.push("demoscene".to_string());
        assert!(!filter.should_cull(&pass));
    }

    #[test]
    fn test_filter_should_cull_preserved() {
        let mut filter = FullscreenModeFilter::new(true);
        filter.preserve("preserved_pass");
        let pass = IrPass::graphics(
            PassIndex(0),
            "preserved_pass",
            Vec::new(),
            None,
            crate::frame_graph::InstanceSource::Direct {
                index_count: 0,
                instance_count: 1,
                base_vertex: 0,
                first_index: 0,
                first_instance: 0,
            },
            ViewType::ColorAttachment,
        );
        assert!(!filter.should_cull(&pass));
    }

    #[test]
    fn test_filter_should_cull_graphics() {
        let filter = FullscreenModeFilter::new(true);
        let pass = IrPass::graphics(
            PassIndex(0),
            "render_scene",
            Vec::new(),
            None,
            crate::frame_graph::InstanceSource::Direct {
                index_count: 0,
                instance_count: 1,
                base_vertex: 0,
                first_index: 0,
                first_instance: 0,
            },
            ViewType::ColorAttachment,
        );
        assert!(filter.should_cull(&pass));
    }

    #[test]
    fn test_filter_should_cull_uncullable() {
        let filter = FullscreenModeFilter::new(true);
        let mut pass = IrPass::graphics(
            PassIndex(0),
            "side_effect_pass",
            Vec::new(),
            None,
            crate::frame_graph::InstanceSource::Direct {
                index_count: 0,
                instance_count: 1,
                base_vertex: 0,
                first_index: 0,
                first_instance: 0,
            },
            ViewType::ColorAttachment,
        );
        pass.flags = PassFlags::NO_CULL;
        assert!(!filter.should_cull(&pass));
    }

    #[test]
    fn test_filter_filter_passes() {
        let filter = FullscreenModeFilter::new(true);
        let passes = vec![
            {
                let mut p = IrPass::compute(
                    PassIndex(0),
                    "demoscene",
                    DispatchSource::Direct {
                        group_count_x: 10,
                        group_count_y: 10,
                        group_count_z: 1,
                    },
                    ViewType::Storage,
                );
                p.tags.push("demoscene".to_string());
                p
            },
            IrPass::graphics(
                PassIndex(1),
                "render_scene",
                Vec::new(),
                None,
                crate::frame_graph::InstanceSource::Direct {
                    index_count: 0,
                    instance_count: 1,
                    base_vertex: 0,
                    first_index: 0,
                    first_instance: 0,
                },
                ViewType::ColorAttachment,
            ),
        ];

        let filtered = filter.filter_passes(&passes);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].name, "demoscene");
    }

    // =========================================================================
    // DemoscenePassRegistry Tests
    // =========================================================================

    #[test]
    fn test_registry_new() {
        let registry = DemoscenePassRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_registry_register() {
        let mut registry = DemoscenePassRegistry::new();
        let pass = DemoscenePass::new(800, 600, ResourceHandle(0));
        assert!(registry.register(pass).is_ok());
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_registry_register_duplicate() {
        let mut registry = DemoscenePassRegistry::new();
        let pass1 = DemoscenePass::new(800, 600, ResourceHandle(0));
        let pass2 = DemoscenePass::new(1920, 1080, ResourceHandle(1));
        assert!(registry.register(pass1).is_ok());
        let result = registry.register(pass2);
        assert!(matches!(
            result,
            Err(DemoscenePassError::PassAlreadyRegistered(_))
        ));
    }

    #[test]
    fn test_registry_get() {
        let mut registry = DemoscenePassRegistry::new();
        let pass = DemoscenePass::new(800, 600, ResourceHandle(0));
        registry.register(pass).unwrap();

        let retrieved = registry.get("demoscene");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().config.width, 800);
    }

    #[test]
    fn test_registry_passes() {
        let mut registry = DemoscenePassRegistry::new();
        let pass1 = DemoscenePass::new(800, 600, ResourceHandle(0)).with_name("pass1");
        let pass2 = DemoscenePass::new(1920, 1080, ResourceHandle(1)).with_name("pass2");
        registry.register(pass1).unwrap();
        registry.register(pass2).unwrap();

        let passes: Vec<_> = registry.passes().collect();
        assert_eq!(passes.len(), 2);
    }

    #[test]
    fn test_registry_to_ir_passes() {
        let mut registry = DemoscenePassRegistry::new();
        let pass = DemoscenePass::new(800, 600, ResourceHandle(0));
        registry.register(pass).unwrap();

        let ir_passes = registry.to_ir_passes(5);
        assert_eq!(ir_passes.len(), 1);
        assert_eq!(ir_passes[0].index, PassIndex(5));
    }

    #[test]
    fn test_registry_collect_resources() {
        let mut registry = DemoscenePassRegistry::new();
        let pass = DemoscenePass::new(800, 600, ResourceHandle(0));
        registry.register(pass).unwrap();

        let resources = registry.collect_resources(10);
        assert_eq!(resources.len(), 1);
        assert_eq!(resources[0].handle, ResourceHandle(10));
    }

    // =========================================================================
    // DemosceneExecutionOrder Tests
    // =========================================================================

    #[test]
    fn test_execution_order_default() {
        let order = DemosceneExecutionOrder::default();
        assert_eq!(order, DemosceneExecutionOrder::Last);
    }

    #[test]
    fn test_execution_order_to_priority() {
        assert_eq!(DemosceneExecutionOrder::First.to_priority(), i32::MIN);
        assert_eq!(DemosceneExecutionOrder::Last.to_priority(), i32::MAX);
        assert_eq!(DemosceneExecutionOrder::Priority(100).to_priority(), 100);
    }

    // =========================================================================
    // DemosceneOutputBinding Tests
    // =========================================================================

    #[test]
    fn test_output_binding_new() {
        let binding = DemosceneOutputBinding::new(ResourceHandle(5));
        assert_eq!(binding.handle, ResourceHandle(5));
        assert_eq!(binding.binding, 0);
        assert!(!binding.direct_to_swapchain);
        assert_eq!(binding.format, "rgba8unorm");
    }

    #[test]
    fn test_output_binding_with_binding() {
        let binding = DemosceneOutputBinding::new(ResourceHandle(0)).with_binding(2);
        assert_eq!(binding.binding, 2);
    }

    #[test]
    fn test_output_binding_with_direct_swapchain() {
        let binding = DemosceneOutputBinding::new(ResourceHandle(0)).with_direct_swapchain();
        assert!(binding.direct_to_swapchain);
    }

    #[test]
    fn test_output_binding_with_format() {
        let binding = DemosceneOutputBinding::new(ResourceHandle(0)).with_format("rgba16float");
        assert_eq!(binding.format, "rgba16float");
    }

    // =========================================================================
    // Integration Tests
    // =========================================================================

    #[test]
    fn test_integration_full_pipeline() {
        // Create a complete demoscene frame graph setup
        let output = ResourceHandle(0);
        let pass = DemoscenePass::fullscreen(1920, 1080, output)
            .with_name("s13_raymarch")
            .with_tag("intro");

        // Validate
        assert!(pass.validate().is_ok());

        // Convert to IR
        let ir_pass = pass.to_ir_pass(PassIndex(0));
        assert_eq!(ir_pass.name, "s13_raymarch");
        assert!(ir_pass.tags.contains(&"intro".to_string()));

        // Check dispatch
        if let Some(DispatchSource::Direct { group_count_x, group_count_y, group_count_z }) = ir_pass.dispatch_source {
            assert_eq!(group_count_x, 240); // 1920 / 8
            assert_eq!(group_count_y, 135); // 1080 / 8
            assert_eq!(group_count_z, 1);
        } else {
            panic!("Expected Direct dispatch source");
        }
    }

    #[test]
    fn test_integration_fullscreen_filter() {
        // Create mixed passes
        let demoscene_pass = {
            let output = ResourceHandle(0);
            let pass = DemoscenePass::fullscreen(1920, 1080, output);
            pass.to_ir_pass(PassIndex(0))
        };

        let graphics_pass = IrPass::graphics(
            PassIndex(1),
            "render_scene",
            Vec::new(),
            None,
            crate::frame_graph::InstanceSource::Direct {
                index_count: 0,
                instance_count: 1,
                base_vertex: 0,
                first_index: 0,
                first_instance: 0,
            },
            ViewType::ColorAttachment,
        );

        // Apply filter
        let filter = FullscreenModeFilter::new(true);
        let passes = vec![demoscene_pass, graphics_pass];
        let filtered = filter.filter_passes(&passes);

        // Only demoscene pass should remain
        assert_eq!(filtered.len(), 1);
        assert!(filtered[0].tags.contains(&"demoscene".to_string()));
    }

    #[test]
    fn test_integration_registry_workflow() {
        let mut registry = DemoscenePassRegistry::new();

        // Register multiple passes
        let pass1 = DemoscenePass::new(800, 600, ResourceHandle(0)).with_name("pass1");
        let pass2 = DemoscenePass::fullscreen(1920, 1080, ResourceHandle(1)).with_name("pass2");

        registry.register(pass1).unwrap();
        registry.register(pass2).unwrap();

        // Convert to IR
        let ir_passes = registry.to_ir_passes(0);
        assert_eq!(ir_passes.len(), 2);

        // Collect resources
        let resources = registry.collect_resources(0);
        assert_eq!(resources.len(), 2);

        // Verify one pass is fullscreen
        let fullscreen_count = ir_passes.iter().filter(|p| p.flags.has_no_cull()).count();
        assert_eq!(fullscreen_count, 1);
    }
}
