//! Presentation module - window surface and swapchain management.
//!
//! This module provides the presentation infrastructure for TRINITY:
//!
//! - [`TrinitySurface`] - Platform-agnostic surface creation from window handles
//! - [`Swapchain`] - High-level swapchain management with state tracking
//! - [`VSyncMode`] - User-facing VSync preference settings
//! - [`VSyncController`] - Adaptive VSync management with frame time tracking
//! - [`PresentModeSelector`] - Hardware capability queries and present mode selection
//! - Surface configuration and format negotiation
//! - Multi-platform support (Wayland, X11, Windows, macOS, Web)
//!
//! # Architecture
//!
//! The presentation module abstracts platform-specific window system integration:
//!
//! ```text
//! Swapchain (swapchain.rs)
//!     |-- SwapchainConfig (format, size, present mode)
//!     |-- SwapchainState (valid, outdated, lost, minimized)
//!     |-- SwapchainFrame (texture, view, metadata)
//!     `-- SwapchainError (lost, outdated, timeout, oom)
//!
//! TrinitySurface (surface.rs)
//!     |-- raw-window-handle integration (0.6)
//!     |-- Platform detection via cfg
//!     |-- Surface format negotiation
//!     `-- Error handling with detailed diagnostics
//!
//! SurfaceConfig (surface_config.rs)
//!     |-- SurfaceConfigBuilder - Fluent builder for configurations
//!     |-- ConfigPreset - Pre-defined configuration profiles
//!     |-- ConfigValidationError - Detailed validation error types
//!     `-- validate_config - Strict configuration validation
//!
//! VSync (vsync.rs)
//!     |-- VSyncMode - User preference (On/Off/Adaptive/FastSync/HalfRate)
//!     |-- VSyncController - Adaptive mode management with frame tracking
//!     |-- PresentModeSelector - Hardware capability queries
//!     `-- VSyncFramePacer - VSync-aware frame timing
//! ```
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::device::TrinityInstance;
//! use renderer_backend::presentation::{TrinitySurface, Swapchain, SwapchainConfig, SurfaceError};
//!
//! // Create instance
//! let instance = TrinityInstance::new();
//!
//! // Create surface from window (winit, SDL2, etc.)
//! // let surface = TrinitySurface::new(instance.inner(), &window)?;
//!
//! // Create swapchain
//! // let config = SwapchainConfig::new(1920, 1080);
//! // let mut swapchain = Swapchain::new(config);
//! // swapchain.configure(&surface, &device);
//! ```

pub mod frame_acquire;
pub mod frame_pacing;
pub mod multi_window;
pub mod present_mode;
pub mod resize;
pub mod surface;
pub mod surface_config;
pub mod swapchain;
pub mod triple_buffer;
pub mod vsync;

pub use surface::{
    AlphaModePreference,
    BufferingConfig,
    BufferingMode,
    FormatCategory,
    Frame,
    FrameError,
    FrameInFlightTracker,
    FramePacer,
    FrameStatistics,
    FrameTiming,
    HeadlessConfig,
    HeadlessError,
    HeadlessFrame,
    HeadlessRenderer,
    HeadlessTarget,
    MultiWindowError,
    MultiWindowManager,
    MultiWindowStats,
    PlatformTarget,
    PresentModeInfo,
    PresentModePreference,
    ReadbackBuffer,
    ResizeEvent,
    SurfaceCapabilities,
    SurfaceConfiguration,
    SurfaceError,
    SurfaceSize,
    SyncMode,
    TrinitySurface,
    WindowConfig,
    WindowFrame,
    WindowId,
    WindowState,
    are_srgb_companions,
    get_srgb_companion_format,
    DEFAULT_FRAME_HISTORY_SIZE,
};

pub use swapchain::{
    Swapchain,
    SwapchainConfig,
    SwapchainError,
    SwapchainFrame,
    SwapchainState,
};

pub use vsync::{
    PresentModeSelector,
    VSyncController,
    VSyncFramePacer,
    VSyncMode,
    VSyncPresentModeInfo,
};

pub use present_mode::{
    LatencyLevel,
    PresentModeInfo as PresentModeInfoV2,
    PresentModePreference as PresentModePreferenceV2,
    PresentModeSelector as PresentModeSelectorV2,
};

pub use frame_acquire::{
    AcquireConfig,
    AcquireError,
    AcquiredFrame,
    AcquireResult,
    FrameAcquirer,
    FrameAcquireStats,
};

pub use surface_config::{
    ConfigPreset,
    ConfigValidationError,
    SurfaceConfigBuilder,
    SurfaceConfigValidation,
    validate_config,
};

pub use resize::{
    AspectRatioConstraint,
    ResizeEvent as ResizeEventV2,
    ResizeHandler,
    ResizeStrategy,
    ResizeValidation,
};

pub use triple_buffer::{
    BufferSlot,
    BufferState,
    BufferStats,
    BufferStrategy,
    TripleBufferManager,
};

pub use frame_pacing::{
    FrameBudget,
    FramePacerV2,
    FramePacingMode,
    FrameTimingStats,
    SmoothedFrameTime,
    DEFAULT_PACING_HISTORY_SIZE,
};

pub use multi_window::{
    ManagedWindow,
    MultiWindowManager as MultiWindowManagerV2,
    WindowConfig as WindowConfigV2,
    WindowEvent,
    WindowId as WindowIdV2,
    WindowState as WindowStateV2,
};
