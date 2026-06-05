//! GPU Debug Module for wgpu 25.x
//!
//! This module provides comprehensive GPU debugging utilities for integration
//! with tools like RenderDoc, PIX, Nsight Graphics, and Metal GPU Profiler.
//!
//! # Modules
//!
//! - [`markers`] - Debug marker and label system with stack tracking
//! - [`utils`] - Error handling and diagnostic utilities
//! - [`validation`] - GPU validation layer integration
//! - [`performance`] - GPU performance profiling with timestamp queries
//!
//! # Overview
//!
//! The debug module complements the existing `debug_utils` module by providing:
//!
//! - **DebugLabel**: Labels with optional color for visualization
//! - **DebugGroup**: Debug groups with nesting depth and profiling
//! - **DebugMarkerStack**: Stack-based tracking of debug groups
//! - **Context Wrappers**: Typed wrappers for RenderPass, ComputePass, CommandEncoder
//! - **RAII Guards**: Automatic push/pop with scope guards
//! - **Error Utilities**: Structured GPU error handling and diagnostics
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::debug::markers::*;
//!
//! // Wrap a render pass with debug context
//! let mut ctx = RenderPassDebugContext::new(&mut render_pass);
//!
//! // Push a colored debug group
//! ctx.push_debug_group(DebugLabel::with_color(
//!     "Shadow Pass",
//!     colors::SHADOW,
//! ));
//!
//! // Insert markers
//! ctx.insert_debug_marker(DebugLabel::new("Draw Cascade 0"));
//!
//! // Pop the group
//! ctx.pop_debug_group();
//! ```
//!
//! # Error Handling Example
//!
//! ```ignore
//! use renderer_backend::debug::utils::*;
//!
//! let mut debug = DebugUtils::new();
//!
//! // Set device lost handler
//! debug.set_device_lost_handler(|info| {
//!     eprintln!("Device lost: {} - {}", info.reason, info.message);
//! });
//!
//! // Use error scopes
//! debug.push_error_scope(ErrorFilter::Validation);
//! // ... GPU operations ...
//! if let Some(scope) = debug.pop_error_scope() {
//!     for error in scope.errors() {
//!         eprintln!("GPU Error: {}", error);
//!     }
//! }
//! ```
//!
//! # Relationship to debug_utils
//!
//! This module extends the functionality in `debug_utils`:
//!
//! | debug_utils | debug::markers |
//! |------------|----------------|
//! | Basic RAII DebugScope | DebugScopeGuard with context tracking |
//! | DebugMarker with metadata | DebugLabel with colors |
//! | DebugGroupOps trait | DebugContextOps trait + context wrappers |
//! | Global functions | Stack-based management |
//!
//! Use `debug_utils` for simple, lightweight debug group management.
//! Use `debug::markers` when you need:
//! - Color-coded labels
//! - Profiling timestamps
//! - Stack depth tracking
//! - Maximum depth enforcement
//! - Path-based label hierarchies
//!
//! # Labels Module
//!
//! The [`labels`] module provides categorized debug labels with a registry system:
//!
//! ```ignore
//! use renderer_backend::debug::labels::*;
//!
//! let mut registry = LabelRegistry::new();
//! let shadow_label = registry.register("Shadow Pass", DebugCategory::Pass);
//!
//! // Use scoped debug groups
//! let _guard = DebugScope::new(&mut encoder, shadow_label.clone()).push();
//! // Group auto-pops when _guard drops
//! ```

pub mod labels;
pub mod markers;
pub mod output;
pub mod performance;
pub mod statistics;
pub mod utils;
pub mod validation;
pub mod visualization;

// Re-export commonly used types at the module level
pub use markers::{
    colors, CommandEncoderDebugContext, ComputePassDebugContext, DebugContextOps, DebugGroup,
    DebugLabel, DebugMarkerStack, DebugScopeGuard, RenderPassDebugContext,
};

// Re-export debug utilities
pub use utils::{
    DebugUtils, DeviceLostInfo, DeviceLostReason, ErrorCallbackFn, ErrorCallbackRegistry,
    ErrorCaptureGuard, ErrorFilter, ErrorScope, GpuError, GpuErrorType, Severity, SourceLocation,
};

// Re-export validation types
pub use validation::{
    ValidationCallbackFn, ValidationCallbackRegistry, ValidationFeatures, ValidationLayer,
    ValidationLevel, ValidationMessage, ValidationMessageType, ValidationObject,
    ValidationObjectType, ValidationScope, ValidationScopeResult, ValidationSeverity,
};

// Re-export visualization types
pub use visualization::{
    ChannelMask, DebugShaderData, DebugVisualization, DebugVisualizationManager,
    VisualizationConfig,
};

// Re-export labels types (with prefix to avoid collision with markers::DebugLabel)
pub use labels::{
    DebugCategory, DebugMarker, DebugScope, LabelRegistry,
    DebugLabel as CategorizedLabel,
    DebugScopeGuard as LabelScopeGuard,
};

// Re-export performance profiling types
pub use performance::{
    GpuTimer, MarkerCategory, PerformanceMarker, PerformanceProfiler, TimestampQuery,
};

// Re-export statistics types
pub use statistics::{
    FrameStatistics, GpuStatistic, StatisticType, StatisticsCollector, StatisticsReport,
};

// Re-export output types
pub use output::{
    BufferOutputSink, ConsoleOutputSink, DebugLogger, DebugMessage, DebugOutputLevel,
    DebugOutputSink, DebugSource,
};
