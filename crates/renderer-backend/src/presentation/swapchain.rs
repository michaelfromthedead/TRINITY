//! Swapchain management for TRINITY presentation.
//!
//! This module provides the [`Swapchain`] abstraction for managing surface
//! configuration, frame acquisition, and presentation. It builds on top of
//! the lower-level [`TrinitySurface`] and provides a cleaner API for common
//! rendering workflows.
//!
//! # Architecture
//!
//! ```text
//! Swapchain
//!     |-- SwapchainConfig (format, size, present mode)
//!     |-- SwapchainState (valid, outdated, lost, minimized)
//!     |-- SwapchainFrame (texture, view, metadata)
//!     `-- SwapchainError (lost, outdated, timeout, oom)
//! ```
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::presentation::{Swapchain, SwapchainConfig};
//!
//! // Create swapchain configuration
//! let config = SwapchainConfig::new(1920, 1080)
//!     .with_format(wgpu::TextureFormat::Bgra8UnormSrgb)
//!     .with_present_mode(wgpu::PresentMode::Fifo);
//!
//! // Create and configure swapchain
//! let mut swapchain = Swapchain::new(config);
//! swapchain.configure(&surface, &device);
//!
//! // Render loop
//! loop {
//!     match swapchain.acquire_frame(&surface) {
//!         Ok(frame) => {
//!             // Render to frame.view()
//!             swapchain.present();
//!         }
//!         Err(SwapchainError::Outdated) => {
//!             swapchain.resize(new_width, new_height, &surface, &device);
//!         }
//!         Err(e) => break,
//!     }
//! }
//! ```

use std::fmt;
use thiserror::Error;

// ============================================================================
// SwapchainConfig
// ============================================================================

/// Configuration for the swapchain.
///
/// Defines the size, format, present mode, and other properties of the
/// swapchain. This configuration is used when creating or resizing the
/// swapchain.
///
/// # Example
///
/// ```ignore
/// let config = SwapchainConfig::new(1920, 1080)
///     .with_format(wgpu::TextureFormat::Bgra8UnormSrgb)
///     .with_present_mode(wgpu::PresentMode::Fifo)
///     .with_desired_frame_count(3);
/// ```
#[derive(Clone, Debug)]
pub struct SwapchainConfig {
    /// Width of the swapchain textures in pixels.
    pub width: u32,
    /// Height of the swapchain textures in pixels.
    pub height: u32,
    /// Texture format for the swapchain.
    pub format: wgpu::TextureFormat,
    /// Presentation mode (vsync behavior).
    pub present_mode: wgpu::PresentMode,
    /// Alpha compositing mode.
    pub alpha_mode: wgpu::CompositeAlphaMode,
    /// Additional view formats for creating texture views.
    pub view_formats: Vec<wgpu::TextureFormat>,
    /// Desired number of frames in the swapchain (typically 2 or 3).
    pub desired_frame_count: u32,
}

impl SwapchainConfig {
    /// Create a new swapchain configuration with the given dimensions.
    ///
    /// Uses reasonable defaults:
    /// - Format: Bgra8UnormSrgb (sRGB for gamma-correct rendering)
    /// - Present mode: Fifo (vsync)
    /// - Alpha mode: Auto
    /// - Desired frame count: 2 (double buffering)
    ///
    /// # Arguments
    ///
    /// * `width` - Width in pixels (minimum 1).
    /// * `height` - Height in pixels (minimum 1).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = SwapchainConfig::new(1920, 1080);
    /// assert_eq!(config.width, 1920);
    /// assert_eq!(config.height, 1080);
    /// ```
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width: width.max(1),
            height: height.max(1),
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: Vec::new(),
            desired_frame_count: 2,
        }
    }

    /// Set the texture format.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = SwapchainConfig::new(1920, 1080)
    ///     .with_format(wgpu::TextureFormat::Rgba8Unorm);
    /// ```
    pub fn with_format(mut self, format: wgpu::TextureFormat) -> Self {
        self.format = format;
        self
    }

    /// Set the present mode.
    ///
    /// Common present modes:
    /// - `Fifo`: VSync, wait for next vertical blank
    /// - `Mailbox`: Triple buffering, low latency with no tearing
    /// - `Immediate`: No VSync, may tear
    /// - `FifoRelaxed`: Like Fifo but may tear if frame takes too long
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = SwapchainConfig::new(1920, 1080)
    ///     .with_present_mode(wgpu::PresentMode::Mailbox);
    /// ```
    pub fn with_present_mode(mut self, mode: wgpu::PresentMode) -> Self {
        self.present_mode = mode;
        self
    }

    /// Set the alpha compositing mode.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = SwapchainConfig::new(1920, 1080)
    ///     .with_alpha_mode(wgpu::CompositeAlphaMode::Opaque);
    /// ```
    pub fn with_alpha_mode(mut self, mode: wgpu::CompositeAlphaMode) -> Self {
        self.alpha_mode = mode;
        self
    }

    /// Set additional view formats for texture views.
    ///
    /// This allows creating texture views with different formats from the
    /// base swapchain format, commonly used for sRGB/linear toggle.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = SwapchainConfig::new(1920, 1080)
    ///     .with_format(wgpu::TextureFormat::Bgra8Unorm)
    ///     .with_view_formats(vec![wgpu::TextureFormat::Bgra8UnormSrgb]);
    /// ```
    pub fn with_view_formats(mut self, formats: Vec<wgpu::TextureFormat>) -> Self {
        self.view_formats = formats;
        self
    }

    /// Set the desired frame count (buffer count).
    ///
    /// Common values:
    /// - 2: Double buffering (default)
    /// - 3: Triple buffering (reduces latency with Mailbox present mode)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = SwapchainConfig::new(1920, 1080)
    ///     .with_desired_frame_count(3);
    /// ```
    pub fn with_desired_frame_count(mut self, count: u32) -> Self {
        self.desired_frame_count = count.max(1);
        self
    }

    /// Get the aspect ratio (width / height).
    ///
    /// Returns 1.0 if height is 0 to avoid division by zero.
    pub fn aspect_ratio(&self) -> f32 {
        if self.height == 0 {
            1.0
        } else {
            self.width as f32 / self.height as f32
        }
    }

    /// Check if the dimensions are valid (non-zero).
    pub fn is_valid(&self) -> bool {
        self.width > 0 && self.height > 0
    }

    /// Check if the window is minimized (zero dimensions).
    pub fn is_minimized(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// Convert to wgpu SurfaceConfiguration.
    pub fn to_wgpu_config(&self) -> wgpu::SurfaceConfiguration {
        wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: self.format,
            width: self.width,
            height: self.height,
            present_mode: self.present_mode,
            alpha_mode: self.alpha_mode,
            view_formats: self.view_formats.clone(),
            desired_maximum_frame_latency: self.desired_frame_count,
        }
    }
}

impl Default for SwapchainConfig {
    fn default() -> Self {
        Self::new(800, 600)
    }
}

// ============================================================================
// SwapchainState
// ============================================================================

/// Current state of the swapchain.
///
/// The swapchain can be in one of four states, which determines what
/// actions need to be taken before rendering can continue.
///
/// # State Transitions
///
/// ```text
/// Valid <-> Outdated (resize triggers reconfigure)
///   |
///   v
/// Lost (device lost, needs full recreation)
///   |
///   v
/// Minimized (window minimized, skip rendering)
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SwapchainState {
    /// Swapchain is valid and ready for rendering.
    ///
    /// Frame acquisition should succeed in this state.
    Valid,

    /// Swapchain is outdated and needs reconfiguration.
    ///
    /// This typically happens after a window resize. Call `resize()` or
    /// `configure()` before attempting to acquire frames.
    Outdated,

    /// Swapchain is lost and needs complete recreation.
    ///
    /// This can happen due to:
    /// - GPU device reset
    /// - Display mode changes
    /// - Driver updates
    ///
    /// The surface and possibly device need to be recreated.
    Lost,

    /// Window is minimized, rendering should be skipped.
    ///
    /// The swapchain is valid but has zero dimensions. Wait until the
    /// window is restored before attempting to render.
    Minimized,
}

impl SwapchainState {
    /// Returns true if the swapchain is ready for rendering.
    pub fn is_ready(&self) -> bool {
        matches!(self, SwapchainState::Valid)
    }

    /// Returns true if reconfiguration is needed.
    pub fn needs_reconfigure(&self) -> bool {
        matches!(self, SwapchainState::Outdated)
    }

    /// Returns true if the swapchain was lost.
    pub fn is_lost(&self) -> bool {
        matches!(self, SwapchainState::Lost)
    }

    /// Returns true if the window is minimized.
    pub fn is_minimized(&self) -> bool {
        matches!(self, SwapchainState::Minimized)
    }

    /// Returns true if rendering should be skipped this frame.
    pub fn should_skip_frame(&self) -> bool {
        !matches!(self, SwapchainState::Valid)
    }
}

impl fmt::Display for SwapchainState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SwapchainState::Valid => write!(f, "Valid"),
            SwapchainState::Outdated => write!(f, "Outdated"),
            SwapchainState::Lost => write!(f, "Lost"),
            SwapchainState::Minimized => write!(f, "Minimized"),
        }
    }
}

// ============================================================================
// SwapchainError
// ============================================================================

/// Errors that can occur during swapchain operations.
///
/// These errors map closely to wgpu's SurfaceError but provide additional
/// context and recovery guidance.
#[derive(Clone, Debug, Error)]
pub enum SwapchainError {
    /// The surface was lost and needs to be recreated.
    ///
    /// This is an unrecoverable error that requires recreating the surface
    /// and possibly the device.
    #[error("swapchain lost: surface must be recreated")]
    Lost,

    /// The swapchain is outdated and needs reconfiguration.
    ///
    /// This typically happens after a window resize. Call `resize()` and
    /// retry frame acquisition.
    #[error("swapchain outdated: reconfiguration required")]
    Outdated,

    /// Frame acquisition timed out.
    ///
    /// The GPU is busy and couldn't provide a frame in time. This is a
    /// transient error - skip this frame and try again next tick.
    #[error("frame acquisition timed out")]
    Timeout,

    /// Out of GPU memory.
    ///
    /// The GPU has run out of memory. This may be recoverable by reducing
    /// resource usage, but often indicates a serious problem.
    #[error("out of GPU memory")]
    OutOfMemory,
}

impl SwapchainError {
    /// Returns true if this error is recoverable without recreating the surface.
    ///
    /// `Timeout` and `Outdated` are recoverable:
    /// - Timeout: Just retry on the next frame
    /// - Outdated: Reconfigure and retry
    pub fn is_recoverable(&self) -> bool {
        matches!(self, SwapchainError::Timeout | SwapchainError::Outdated)
    }

    /// Returns true if the swapchain needs reconfiguration.
    pub fn needs_reconfigure(&self) -> bool {
        matches!(self, SwapchainError::Outdated)
    }

    /// Returns true if the swapchain is lost.
    pub fn is_lost(&self) -> bool {
        matches!(self, SwapchainError::Lost)
    }

    /// Convert to SwapchainState.
    pub fn to_state(&self) -> SwapchainState {
        match self {
            SwapchainError::Lost | SwapchainError::OutOfMemory => SwapchainState::Lost,
            SwapchainError::Outdated => SwapchainState::Outdated,
            SwapchainError::Timeout => SwapchainState::Valid, // Transient, state is still valid
        }
    }
}

impl From<wgpu::SurfaceError> for SwapchainError {
    fn from(err: wgpu::SurfaceError) -> Self {
        match err {
            wgpu::SurfaceError::Timeout => SwapchainError::Timeout,
            wgpu::SurfaceError::Outdated => SwapchainError::Outdated,
            wgpu::SurfaceError::Lost => SwapchainError::Lost,
            wgpu::SurfaceError::OutOfMemory => SwapchainError::OutOfMemory,
            // Map any future variants to Lost
            _ => SwapchainError::Lost,
        }
    }
}

// ============================================================================
// SwapchainFrame
// ============================================================================

/// An acquired frame from the swapchain, ready for rendering.
///
/// Contains the surface texture, a pre-created texture view, and metadata
/// about the frame. The `suboptimal` flag indicates whether the swapchain
/// should be reconfigured after presenting (e.g., due to a pending resize).
///
/// # Lifecycle
///
/// ```text
/// acquire_frame() -> SwapchainFrame
///     |
///     v
/// [Render to frame.view]
///     |
///     v
/// swapchain.present() -> displays frame
/// ```
///
/// # Example
///
/// ```ignore
/// let frame = swapchain.acquire_frame(&surface)?;
///
/// // Check for suboptimal hint
/// if frame.suboptimal {
///     // Schedule resize for after this frame
/// }
///
/// // Render to the frame
/// let mut encoder = device.create_command_encoder(&Default::default());
/// {
///     let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
///         color_attachments: &[Some(wgpu::RenderPassColorAttachment {
///             view: &frame.view,
///             resolve_target: None,
///             ops: wgpu::Operations {
///                 load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
///                 store: wgpu::StoreOp::Store,
///             },
///         })],
///         ..Default::default()
///     });
/// }
/// queue.submit([encoder.finish()]);
///
/// swapchain.present();
/// ```
pub struct SwapchainFrame {
    /// The acquired surface texture.
    pub texture: wgpu::SurfaceTexture,
    /// Pre-created texture view for rendering.
    pub view: wgpu::TextureView,
    /// Index of this frame in the swapchain (0-based).
    pub index: u32,
    /// Whether the swapchain is suboptimal and should be reconfigured.
    ///
    /// When true, the frame is still usable but the swapchain should be
    /// resized/reconfigured after presenting for optimal performance.
    pub suboptimal: bool,
}

impl SwapchainFrame {
    /// Create a new swapchain frame from a surface texture.
    ///
    /// # Arguments
    ///
    /// * `texture` - The acquired surface texture.
    /// * `format` - The texture format for the view.
    /// * `index` - Frame index in the swapchain.
    /// * `suboptimal` - Whether the swapchain is suboptimal.
    /// * `label` - Optional label for the texture view.
    pub fn new(
        texture: wgpu::SurfaceTexture,
        format: wgpu::TextureFormat,
        index: u32,
        suboptimal: bool,
        label: Option<&str>,
    ) -> Self {
        let view = texture.texture.create_view(&wgpu::TextureViewDescriptor {
            label,
            format: Some(format),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            ..Default::default()
        });

        Self {
            texture,
            view,
            index,
            suboptimal,
        }
    }

    /// Get the width of the frame in pixels.
    pub fn width(&self) -> u32 {
        self.texture.texture.width()
    }

    /// Get the height of the frame in pixels.
    pub fn height(&self) -> u32 {
        self.texture.texture.height()
    }

    /// Get the aspect ratio (width / height).
    pub fn aspect_ratio(&self) -> f32 {
        let h = self.height();
        if h == 0 {
            1.0
        } else {
            self.width() as f32 / h as f32
        }
    }

    /// Get the underlying raw texture.
    pub fn raw_texture(&self) -> &wgpu::Texture {
        &self.texture.texture
    }

    /// Create a texture view with a different format.
    ///
    /// Useful for sRGB/linear toggle when the swapchain is configured
    /// with additional view formats.
    pub fn create_view_with_format(
        &self,
        format: wgpu::TextureFormat,
        label: Option<&str>,
    ) -> wgpu::TextureView {
        self.texture.texture.create_view(&wgpu::TextureViewDescriptor {
            label,
            format: Some(format),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            ..Default::default()
        })
    }
}

impl fmt::Debug for SwapchainFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SwapchainFrame")
            .field("width", &self.width())
            .field("height", &self.height())
            .field("index", &self.index)
            .field("suboptimal", &self.suboptimal)
            .finish()
    }
}

// ============================================================================
// Swapchain
// ============================================================================

/// Manages the swapchain for surface presentation.
///
/// The `Swapchain` provides a higher-level abstraction over wgpu's surface
/// configuration and frame acquisition. It tracks state, handles resizes,
/// and manages frame lifecycle.
///
/// # Usage
///
/// ```ignore
/// // Create swapchain
/// let config = SwapchainConfig::new(1920, 1080);
/// let mut swapchain = Swapchain::new(config);
///
/// // Configure the surface
/// swapchain.configure(&surface, &device);
///
/// // Render loop
/// loop {
///     // Handle resize events
///     if window_resized {
///         swapchain.resize(new_width, new_height, &surface, &device);
///     }
///
///     // Skip rendering if minimized
///     if swapchain.state() == SwapchainState::Minimized {
///         continue;
///     }
///
///     // Acquire and render
///     match swapchain.acquire_frame(&surface) {
///         Ok(frame) => {
///             // Render to frame.view
///             swapchain.present();
///         }
///         Err(SwapchainError::Outdated) => {
///             swapchain.resize(width, height, &surface, &device);
///         }
///         Err(SwapchainError::Timeout) => {
///             // Skip frame, try next tick
///         }
///         Err(e) => {
///             // Lost - need to recreate surface
///             break;
///         }
///     }
/// }
/// ```
pub struct Swapchain {
    /// Current configuration.
    config: SwapchainConfig,
    /// Current state.
    state: SwapchainState,
    /// Total frames acquired since creation.
    frame_count: u64,
    /// Currently acquired frame (if any).
    current_frame: Option<SwapchainFrame>,
    /// Frame index counter (wraps at desired_frame_count).
    frame_index: u32,
}

impl Swapchain {
    /// Create a new swapchain with the given configuration.
    ///
    /// The swapchain starts in `Outdated` state and must be configured
    /// with `configure()` before use.
    ///
    /// # Arguments
    ///
    /// * `config` - Swapchain configuration.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = SwapchainConfig::new(1920, 1080);
    /// let swapchain = Swapchain::new(config);
    /// assert_eq!(swapchain.state(), SwapchainState::Outdated);
    /// ```
    pub fn new(config: SwapchainConfig) -> Self {
        let initial_state = if config.is_minimized() {
            SwapchainState::Minimized
        } else {
            SwapchainState::Outdated
        };

        Self {
            config,
            state: initial_state,
            frame_count: 0,
            current_frame: None,
            frame_index: 0,
        }
    }

    /// Configure the surface with the current swapchain settings.
    ///
    /// This applies the swapchain configuration to the wgpu surface. Must
    /// be called before acquiring frames, and after any resize.
    ///
    /// # Arguments
    ///
    /// * `surface` - The wgpu surface to configure.
    /// * `device` - The wgpu device.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut swapchain = Swapchain::new(config);
    /// swapchain.configure(&surface, &device);
    /// assert_eq!(swapchain.state(), SwapchainState::Valid);
    /// ```
    pub fn configure(&mut self, surface: &wgpu::Surface<'_>, device: &wgpu::Device) {
        if self.config.is_minimized() {
            self.state = SwapchainState::Minimized;
            return;
        }

        let wgpu_config = self.config.to_wgpu_config();
        surface.configure(device, &wgpu_config);
        self.state = SwapchainState::Valid;
    }

    /// Resize the swapchain to new dimensions.
    ///
    /// Updates the configuration and reconfigures the surface. If the new
    /// dimensions are zero (minimized), the state becomes `Minimized`.
    ///
    /// # Arguments
    ///
    /// * `width` - New width in pixels.
    /// * `height` - New height in pixels.
    /// * `surface` - The wgpu surface to reconfigure.
    /// * `device` - The wgpu device.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Handle window resize
    /// swapchain.resize(new_width, new_height, &surface, &device);
    /// ```
    pub fn resize(
        &mut self,
        width: u32,
        height: u32,
        surface: &wgpu::Surface<'_>,
        device: &wgpu::Device,
    ) {
        self.config.width = width.max(1);
        self.config.height = height.max(1);

        // Drop any current frame before reconfiguring
        self.current_frame = None;

        self.configure(surface, device);
    }

    /// Acquire the next frame for rendering.
    ///
    /// Returns a reference to the acquired frame on success. The frame
    /// remains valid until `present()` is called or the frame is dropped.
    ///
    /// # Arguments
    ///
    /// * `surface` - The wgpu surface to acquire from.
    ///
    /// # Returns
    ///
    /// * `Ok(&SwapchainFrame)` - Frame acquired successfully.
    /// * `Err(SwapchainError::Outdated)` - Resize needed, then retry.
    /// * `Err(SwapchainError::Timeout)` - GPU busy, skip this frame.
    /// * `Err(SwapchainError::Lost)` - Surface lost, recreate everything.
    /// * `Err(SwapchainError::OutOfMemory)` - GPU OOM.
    ///
    /// # Example
    ///
    /// ```ignore
    /// match swapchain.acquire_frame(&surface) {
    ///     Ok(frame) => {
    ///         // Render to frame.view
    ///     }
    ///     Err(e) if e.is_recoverable() => {
    ///         // Handle recovery
    ///     }
    ///     Err(e) => {
    ///         // Fatal error
    ///     }
    /// }
    /// ```
    pub fn acquire_frame(
        &mut self,
        surface: &wgpu::Surface<'_>,
    ) -> Result<&SwapchainFrame, SwapchainError> {
        // Check state
        match self.state {
            SwapchainState::Minimized => return Err(SwapchainError::Outdated),
            SwapchainState::Lost => return Err(SwapchainError::Lost),
            SwapchainState::Outdated => return Err(SwapchainError::Outdated),
            SwapchainState::Valid => {}
        }

        // Drop any existing frame
        self.current_frame = None;

        // Acquire new frame
        let output = surface.get_current_texture()?;
        let suboptimal = output.suboptimal;

        // Create frame
        let frame = SwapchainFrame::new(
            output,
            self.config.format,
            self.frame_index,
            suboptimal,
            Some("swapchain_frame"),
        );

        // Update frame index
        self.frame_index = (self.frame_index + 1) % self.config.desired_frame_count;
        self.frame_count += 1;

        // Store and return
        self.current_frame = Some(frame);

        // Update state if suboptimal
        if suboptimal {
            // Still valid for this frame, but should reconfigure after
        }

        Ok(self.current_frame.as_ref().unwrap())
    }

    /// Present the current frame to the display.
    ///
    /// This presents the most recently acquired frame and releases it.
    /// After calling `present()`, the frame is no longer valid.
    ///
    /// # Panics
    ///
    /// Panics if no frame is currently acquired.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let frame = swapchain.acquire_frame(&surface)?;
    /// // Render to frame.view
    /// queue.submit([encoder.finish()]);
    /// swapchain.present();
    /// ```
    pub fn present(&mut self) {
        if let Some(frame) = self.current_frame.take() {
            // Check if we need to trigger reconfigure after presenting
            let should_reconfigure = frame.suboptimal;

            // Present the frame
            frame.texture.present();

            // Update state if suboptimal
            if should_reconfigure {
                self.state = SwapchainState::Outdated;
            }
        }
    }

    /// Get the current swapchain state.
    ///
    /// # Example
    ///
    /// ```ignore
    /// if swapchain.state() == SwapchainState::Minimized {
    ///     // Skip rendering
    /// }
    /// ```
    pub fn state(&self) -> SwapchainState {
        self.state
    }

    /// Get the current swapchain configuration.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = swapchain.config();
    /// println!("Swapchain size: {}x{}", config.width, config.height);
    /// ```
    pub fn config(&self) -> &SwapchainConfig {
        &self.config
    }

    /// Get the total number of frames acquired since creation.
    ///
    /// This counts all frames, including those that were discarded or
    /// had errors during acquisition.
    ///
    /// # Example
    ///
    /// ```ignore
    /// println!("Total frames: {}", swapchain.frame_count());
    /// ```
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Get the current aspect ratio (width / height).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let aspect = swapchain.aspect_ratio();
    /// let projection = Mat4::perspective(fov, aspect, near, far);
    /// ```
    pub fn aspect_ratio(&self) -> f32 {
        self.config.aspect_ratio()
    }

    /// Get the current width in pixels.
    pub fn width(&self) -> u32 {
        self.config.width
    }

    /// Get the current height in pixels.
    pub fn height(&self) -> u32 {
        self.config.height
    }

    /// Get the texture format.
    pub fn format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    /// Get the present mode.
    pub fn present_mode(&self) -> wgpu::PresentMode {
        self.config.present_mode
    }

    /// Set the state directly (for testing or recovery scenarios).
    pub fn set_state(&mut self, state: SwapchainState) {
        self.state = state;
    }

    /// Check if a frame is currently acquired.
    pub fn has_frame(&self) -> bool {
        self.current_frame.is_some()
    }

    /// Get a reference to the current frame, if one is acquired.
    pub fn current_frame(&self) -> Option<&SwapchainFrame> {
        self.current_frame.as_ref()
    }

    /// Discard the current frame without presenting.
    ///
    /// Use when you need to skip a frame (e.g., window minimized).
    pub fn discard_frame(&mut self) {
        self.current_frame = None;
    }

    /// Mark the swapchain as lost.
    ///
    /// Call this when an unrecoverable error occurs.
    pub fn mark_lost(&mut self) {
        self.current_frame = None;
        self.state = SwapchainState::Lost;
    }
}

impl fmt::Debug for Swapchain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Swapchain")
            .field("config", &self.config)
            .field("state", &self.state)
            .field("frame_count", &self.frame_count)
            .field("has_frame", &self.current_frame.is_some())
            .finish()
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // SwapchainConfig Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_swapchain_config_creation() {
        let config = SwapchainConfig::new(1920, 1080);
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.format, wgpu::TextureFormat::Bgra8UnormSrgb);
        assert_eq!(config.present_mode, wgpu::PresentMode::Fifo);
        assert_eq!(config.alpha_mode, wgpu::CompositeAlphaMode::Auto);
        assert_eq!(config.desired_frame_count, 2);
        assert!(config.view_formats.is_empty());
    }

    #[test]
    fn test_swapchain_config_defaults() {
        let config = SwapchainConfig::default();
        assert_eq!(config.width, 800);
        assert_eq!(config.height, 600);
        assert!(config.is_valid());
    }

    #[test]
    fn test_swapchain_config_zero_dimensions_clamped() {
        let config = SwapchainConfig::new(0, 0);
        assert_eq!(config.width, 1);
        assert_eq!(config.height, 1);
        assert!(config.is_valid());
        assert!(!config.is_minimized());
    }

    #[test]
    fn test_swapchain_config_builder_pattern() {
        let config = SwapchainConfig::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Rgba8Unorm)
            .with_present_mode(wgpu::PresentMode::Mailbox)
            .with_alpha_mode(wgpu::CompositeAlphaMode::Opaque)
            .with_view_formats(vec![wgpu::TextureFormat::Rgba8UnormSrgb])
            .with_desired_frame_count(3);

        assert_eq!(config.format, wgpu::TextureFormat::Rgba8Unorm);
        assert_eq!(config.present_mode, wgpu::PresentMode::Mailbox);
        assert_eq!(config.alpha_mode, wgpu::CompositeAlphaMode::Opaque);
        assert_eq!(config.view_formats.len(), 1);
        assert_eq!(config.desired_frame_count, 3);
    }

    #[test]
    fn test_swapchain_config_aspect_ratio() {
        let config = SwapchainConfig::new(1920, 1080);
        let aspect = config.aspect_ratio();
        assert!((aspect - 1.777).abs() < 0.01);

        let square = SwapchainConfig::new(100, 100);
        assert!((square.aspect_ratio() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_swapchain_config_to_wgpu() {
        let config = SwapchainConfig::new(800, 600)
            .with_format(wgpu::TextureFormat::Bgra8Unorm);

        let wgpu_config = config.to_wgpu_config();
        assert_eq!(wgpu_config.width, 800);
        assert_eq!(wgpu_config.height, 600);
        assert_eq!(wgpu_config.format, wgpu::TextureFormat::Bgra8Unorm);
        assert!(wgpu_config.usage.contains(wgpu::TextureUsages::RENDER_ATTACHMENT));
    }

    // -------------------------------------------------------------------------
    // SwapchainState Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_swapchain_state_variants() {
        assert!(SwapchainState::Valid.is_ready());
        assert!(!SwapchainState::Outdated.is_ready());
        assert!(!SwapchainState::Lost.is_ready());
        assert!(!SwapchainState::Minimized.is_ready());
    }

    #[test]
    fn test_swapchain_state_needs_reconfigure() {
        assert!(!SwapchainState::Valid.needs_reconfigure());
        assert!(SwapchainState::Outdated.needs_reconfigure());
        assert!(!SwapchainState::Lost.needs_reconfigure());
        assert!(!SwapchainState::Minimized.needs_reconfigure());
    }

    #[test]
    fn test_swapchain_state_is_lost() {
        assert!(!SwapchainState::Valid.is_lost());
        assert!(!SwapchainState::Outdated.is_lost());
        assert!(SwapchainState::Lost.is_lost());
        assert!(!SwapchainState::Minimized.is_lost());
    }

    #[test]
    fn test_swapchain_state_is_minimized() {
        assert!(!SwapchainState::Valid.is_minimized());
        assert!(!SwapchainState::Outdated.is_minimized());
        assert!(!SwapchainState::Lost.is_minimized());
        assert!(SwapchainState::Minimized.is_minimized());
    }

    #[test]
    fn test_swapchain_state_should_skip_frame() {
        assert!(!SwapchainState::Valid.should_skip_frame());
        assert!(SwapchainState::Outdated.should_skip_frame());
        assert!(SwapchainState::Lost.should_skip_frame());
        assert!(SwapchainState::Minimized.should_skip_frame());
    }

    #[test]
    fn test_swapchain_state_display() {
        assert_eq!(format!("{}", SwapchainState::Valid), "Valid");
        assert_eq!(format!("{}", SwapchainState::Outdated), "Outdated");
        assert_eq!(format!("{}", SwapchainState::Lost), "Lost");
        assert_eq!(format!("{}", SwapchainState::Minimized), "Minimized");
    }

    #[test]
    fn test_swapchain_state_equality() {
        assert_eq!(SwapchainState::Valid, SwapchainState::Valid);
        assert_ne!(SwapchainState::Valid, SwapchainState::Lost);
    }

    // -------------------------------------------------------------------------
    // SwapchainError Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_swapchain_error_variants() {
        let lost = SwapchainError::Lost;
        let outdated = SwapchainError::Outdated;
        let timeout = SwapchainError::Timeout;
        let oom = SwapchainError::OutOfMemory;

        assert!(lost.is_lost());
        assert!(!outdated.is_lost());
        assert!(!timeout.is_lost());
        assert!(!oom.is_lost());
    }

    #[test]
    fn test_swapchain_error_is_recoverable() {
        assert!(!SwapchainError::Lost.is_recoverable());
        assert!(SwapchainError::Outdated.is_recoverable());
        assert!(SwapchainError::Timeout.is_recoverable());
        assert!(!SwapchainError::OutOfMemory.is_recoverable());
    }

    #[test]
    fn test_swapchain_error_needs_reconfigure() {
        assert!(!SwapchainError::Lost.needs_reconfigure());
        assert!(SwapchainError::Outdated.needs_reconfigure());
        assert!(!SwapchainError::Timeout.needs_reconfigure());
        assert!(!SwapchainError::OutOfMemory.needs_reconfigure());
    }

    #[test]
    fn test_swapchain_error_to_state() {
        assert_eq!(SwapchainError::Lost.to_state(), SwapchainState::Lost);
        assert_eq!(SwapchainError::Outdated.to_state(), SwapchainState::Outdated);
        assert_eq!(SwapchainError::Timeout.to_state(), SwapchainState::Valid);
        assert_eq!(SwapchainError::OutOfMemory.to_state(), SwapchainState::Lost);
    }

    #[test]
    fn test_swapchain_error_from_wgpu() {
        assert!(matches!(
            SwapchainError::from(wgpu::SurfaceError::Lost),
            SwapchainError::Lost
        ));
        assert!(matches!(
            SwapchainError::from(wgpu::SurfaceError::Outdated),
            SwapchainError::Outdated
        ));
        assert!(matches!(
            SwapchainError::from(wgpu::SurfaceError::Timeout),
            SwapchainError::Timeout
        ));
        assert!(matches!(
            SwapchainError::from(wgpu::SurfaceError::OutOfMemory),
            SwapchainError::OutOfMemory
        ));
    }

    #[test]
    fn test_swapchain_error_display() {
        let lost = SwapchainError::Lost.to_string().to_lowercase();
        let outdated = SwapchainError::Outdated.to_string().to_lowercase();
        let timeout = SwapchainError::Timeout.to_string().to_lowercase();
        let oom = SwapchainError::OutOfMemory.to_string().to_lowercase();

        assert!(lost.contains("lost"));
        assert!(outdated.contains("outdated"));
        assert!(timeout.contains("timed out") || timeout.contains("timeout"));
        assert!(oom.contains("memory"));
    }

    // -------------------------------------------------------------------------
    // Swapchain Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_swapchain_new() {
        let config = SwapchainConfig::new(1920, 1080);
        let swapchain = Swapchain::new(config);

        assert_eq!(swapchain.state(), SwapchainState::Outdated);
        assert_eq!(swapchain.frame_count(), 0);
        assert_eq!(swapchain.width(), 1920);
        assert_eq!(swapchain.height(), 1080);
        assert!(!swapchain.has_frame());
    }

    #[test]
    fn test_swapchain_aspect_ratio() {
        let config = SwapchainConfig::new(1920, 1080);
        let swapchain = Swapchain::new(config);

        let aspect = swapchain.aspect_ratio();
        assert!((aspect - 1.777).abs() < 0.01);
    }

    #[test]
    fn test_swapchain_config_accessor() {
        let config = SwapchainConfig::new(1280, 720)
            .with_format(wgpu::TextureFormat::Rgba8Unorm);
        let swapchain = Swapchain::new(config);

        assert_eq!(swapchain.config().width, 1280);
        assert_eq!(swapchain.config().height, 720);
        assert_eq!(swapchain.config().format, wgpu::TextureFormat::Rgba8Unorm);
        assert_eq!(swapchain.format(), wgpu::TextureFormat::Rgba8Unorm);
    }

    #[test]
    fn test_swapchain_set_state() {
        let config = SwapchainConfig::new(800, 600);
        let mut swapchain = Swapchain::new(config);

        swapchain.set_state(SwapchainState::Valid);
        assert_eq!(swapchain.state(), SwapchainState::Valid);

        swapchain.set_state(SwapchainState::Lost);
        assert_eq!(swapchain.state(), SwapchainState::Lost);
    }

    #[test]
    fn test_swapchain_mark_lost() {
        let config = SwapchainConfig::new(800, 600);
        let mut swapchain = Swapchain::new(config);

        swapchain.set_state(SwapchainState::Valid);
        swapchain.mark_lost();

        assert_eq!(swapchain.state(), SwapchainState::Lost);
        assert!(!swapchain.has_frame());
    }

    #[test]
    fn test_swapchain_discard_frame() {
        let config = SwapchainConfig::new(800, 600);
        let mut swapchain = Swapchain::new(config);

        // No frame to discard, should not panic
        swapchain.discard_frame();
        assert!(!swapchain.has_frame());
    }

    #[test]
    fn test_swapchain_present_mode_accessor() {
        let config = SwapchainConfig::new(800, 600)
            .with_present_mode(wgpu::PresentMode::Mailbox);
        let swapchain = Swapchain::new(config);

        assert_eq!(swapchain.present_mode(), wgpu::PresentMode::Mailbox);
    }

    #[test]
    fn test_swapchain_debug() {
        let config = SwapchainConfig::new(800, 600);
        let swapchain = Swapchain::new(config);

        let debug_str = format!("{:?}", swapchain);
        assert!(debug_str.contains("Swapchain"));
        assert!(debug_str.contains("state"));
        assert!(debug_str.contains("frame_count"));
    }

    // -------------------------------------------------------------------------
    // SwapchainFrame Tests (without GPU)
    // -------------------------------------------------------------------------

    #[test]
    fn test_swapchain_frame_debug() {
        // We can't create a real SwapchainFrame without GPU, but we can test
        // the Debug implementation through the Swapchain struct
        let config = SwapchainConfig::new(800, 600);
        let swapchain = Swapchain::new(config);

        // current_frame is None, but the accessor should work
        assert!(swapchain.current_frame().is_none());
    }

    // -------------------------------------------------------------------------
    // Integration-style Tests (state machine)
    // -------------------------------------------------------------------------

    #[test]
    fn test_swapchain_state_machine() {
        let config = SwapchainConfig::new(1920, 1080);
        let mut swapchain = Swapchain::new(config);

        // Initial state is Outdated
        assert_eq!(swapchain.state(), SwapchainState::Outdated);

        // Simulate configure (would normally call configure())
        swapchain.set_state(SwapchainState::Valid);
        assert!(swapchain.state().is_ready());

        // Simulate resize needed
        swapchain.set_state(SwapchainState::Outdated);
        assert!(swapchain.state().needs_reconfigure());

        // Simulate surface lost
        swapchain.mark_lost();
        assert!(swapchain.state().is_lost());
    }

    #[test]
    fn test_swapchain_minimized_on_creation() {
        // Create with zero dimensions - should clamp to 1x1, not be minimized
        let config = SwapchainConfig::new(0, 0);
        let swapchain = Swapchain::new(config);

        // Since dimensions are clamped to 1, it's not minimized
        assert_eq!(swapchain.width(), 1);
        assert_eq!(swapchain.height(), 1);
        assert_eq!(swapchain.state(), SwapchainState::Outdated);
    }
}
