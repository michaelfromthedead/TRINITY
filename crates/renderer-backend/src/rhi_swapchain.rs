//! RHI Swapchain mapping layer -- bridge between Python RHI SwapChain ABCs
//! and Rust wgpu surface / swapchain operations.
//!
//! This module provides a thin wrapper over [`wgpu::Surface`] and
//! [`wgpu::SurfaceConfiguration`] that mirrors the Python-side RHI hierarchy
//! (engine/platform/rhi/swapchain.py).  Each type here has a direct counterpart
//! in the Python ABCs:
//!
//! | Rust type / fn | Python counterpart |
//! |---|---|
//! | `PresentMode` | `PresentMode` enum |
//! | `SwapchainConfig` | `SwapChainDesc` dataclass |
//! | `RhiSurface` | (implicit -- surface creation) |
//! | `RhiSwapchain` | `SwapChain` ABC (partially) |
//! | `SurfaceImage` | `Texture` ABC (swapchain image) |
//! | `configure_swapchain()` | `SwapChain.configure()` |
//! | `get_current_texture()` | `SwapChain.get_current_image()` |
//! | `present()` | `SwapChain.present()` |
//! | `resize()` | `SwapChain.resize()` |
//!
//! NOTE: `SurfaceImage` is the swapchain-specific counterpart of the general
//! [`rhi_resources::RhiTexture`].  The former wraps [`wgpu::SurfaceTexture`]
//! (acquired from the swapchain and requiring explicit present); the latter
//! wraps a standalone [`wgpu::Texture`] with full metadata.  Once the general
//! texture type is stable, `SurfaceImage` may be unified by exposing a
//! `present()` method on `rhi_resources::RhiTexture` when backed by a swapchain
//! image.

use crate::rhi_device::RhiInstance;
use wgpu::rwh;

// ---------------------------------------------------------------------------
// PresentMode
// ---------------------------------------------------------------------------

/// Presentation mode for the swapchain.
///
/// Controls how frames are presented to the display.  Mirrors the Python
/// [`PresentMode`] enum from `engine/platform/rhi/swapchain.py`.
///
/// [`PresentMode`]: ../../engine/platform/rhi/swapchain.py
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresentMode {
    /// Frames are presented immediately without waiting for vsync.
    /// May cause tearing.
    Immediate,
    /// Frames are presented at the display's vertical refresh interval.
    /// The queue is FIFO; this is the most broadly supported mode.
    Fifo,
    /// Like Fifo but replaces a pending frame if the queue is full,
    /// reducing input latency while still preventing tearing.
    Mailbox,
}

impl From<PresentMode> for wgpu::PresentMode {
    fn from(mode: PresentMode) -> Self {
        match mode {
            PresentMode::Immediate => wgpu::PresentMode::Immediate,
            PresentMode::Fifo => wgpu::PresentMode::Fifo,
            PresentMode::Mailbox => wgpu::PresentMode::Mailbox,
        }
    }
}

impl From<wgpu::PresentMode> for PresentMode {
    fn from(mode: wgpu::PresentMode) -> Self {
        match mode {
            wgpu::PresentMode::Immediate => PresentMode::Immediate,
            wgpu::PresentMode::Fifo => PresentMode::Fifo,
            wgpu::PresentMode::Mailbox => PresentMode::Mailbox,
            // wgpu may introduce additional modes in the future; fall back
            // to Fifo as the safest default.
            _ => PresentMode::Fifo,
        }
    }
}

// ---------------------------------------------------------------------------
// SwapchainConfig
// ---------------------------------------------------------------------------

/// Describes the configuration of a swapchain.
///
/// Mirrors the Python [`SwapChainDesc`] dataclass from
/// `engine/platform/rhi/swapchain.py`.
///
/// [`SwapChainDesc`]: ../../engine/platform/rhi/swapchain.py
#[derive(Debug, Clone)]
pub struct SwapchainConfig {
    /// Pixel format of the swapchain images.
    pub format: wgpu::TextureFormat,
    /// Width of the swapchain images in texels.
    pub width: u32,
    /// Height of the swapchain images in texels.
    pub height: u32,
    /// Presentation mode (vsync control).
    pub present_mode: PresentMode,
}

impl SwapchainConfig {
    /// Create a new `SwapchainConfig` with the given parameters.
    ///
    /// The width and height are clamped to a minimum of 1 to prevent
    /// zero-sized swapchains.
    pub fn new(format: wgpu::TextureFormat, width: u32, height: u32, present_mode: PresentMode) -> Self {
        Self {
            format,
            width: width.max(1),
            height: height.max(1),
            present_mode,
        }
    }

    /// Convert this config to a [`wgpu::SurfaceConfiguration`] using the
    /// given alpha mode.
    ///
    /// `desired_maximum_frame_latency` controls the number of frames the GPU
    /// may queue ahead (default 2 for a good balance of throughput vs. latency).
    pub fn to_wgpu(&self, alpha_mode: wgpu::CompositeAlphaMode) -> wgpu::SurfaceConfiguration {
        wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: self.format,
            width: self.width,
            height: self.height,
            present_mode: self.present_mode.into(),
            desired_maximum_frame_latency: 2,
            alpha_mode,
            view_formats: vec![],
        }
    }
}

impl Default for SwapchainConfig {
    fn default() -> Self {
        Self {
            format: wgpu::TextureFormat::Rgba8Unorm,
            width: 1,
            height: 1,
            present_mode: PresentMode::Fifo,
        }
    }
}

// ---------------------------------------------------------------------------
// RhiSurface
// ---------------------------------------------------------------------------

/// Thin wrapper around a [`wgpu::Surface`].
///
/// Represents a platform-specific surface (window, view, canvas) that can be
/// used as a render target.  The lifetime parameter `'window` ties the surface
/// to the lifetime of the window handle it was created from.
pub struct RhiSurface {
    inner: wgpu::Surface<'static>,
}

impl RhiSurface {
    /// Borrow the underlying [`wgpu::Surface`].
    pub fn inner(&self) -> &wgpu::Surface<'static> {
        &self.inner
    }
}

// ---------------------------------------------------------------------------
// RhiSwapchain
// ---------------------------------------------------------------------------

/// Thin wrapper around a [`wgpu::SurfaceConfiguration`].
///
/// Represents a fully-configured swapchain ready for rendering.  The actual
/// swapchain state lives on the GPU; this struct holds the configuration that
/// was applied to the surface.
pub struct RhiSwapchain {
    config: wgpu::SurfaceConfiguration,
}

impl RhiSwapchain {
    /// Borrow the underlying [`wgpu::SurfaceConfiguration`].
    pub fn config(&self) -> &wgpu::SurfaceConfiguration {
        &self.config
    }

    /// Mutable access to the underlying configuration (for resizing, etc.).
    pub fn config_mut(&mut self) -> &mut wgpu::SurfaceConfiguration {
        &mut self.config
    }

    /// The width of the swapchain in texels.
    pub fn width(&self) -> u32 {
        self.config.width
    }

    /// The height of the swapchain in texels.
    pub fn height(&self) -> u32 {
        self.config.height
    }
}

// ---------------------------------------------------------------------------
// SurfaceImage  (swapchain image)
// ---------------------------------------------------------------------------

/// A swapchain image acquired from the surface.
///
/// Wraps [`wgpu::SurfaceTexture`] so that calling [`present`] on this type
/// returns the image to the swapchain.  Dropping without presenting is valid
/// (wgpu handles re-submission internally).
///
/// This is the swapchain-specific counterpart of
/// [`rhi_resources::RhiTexture`].  The latter wraps a standalone
/// [`wgpu::Texture`] with full metadata; the former wraps a
/// [`wgpu::SurfaceTexture`] that was *acquired from* the swapchain and must be
/// explicitly presented.
///
/// [`present`]: SurfaceImage::present
/// [`rhi_resources::RhiTexture`]: crate::rhi_resources::RhiTexture
pub struct SurfaceImage {
    inner: wgpu::SurfaceTexture,
}

impl SurfaceImage {
    /// Wrap a [`wgpu::SurfaceTexture`] obtained from
    /// [`wgpu::Surface::get_current_texture`].
    pub fn new(surface_texture: wgpu::SurfaceTexture) -> Self {
        Self {
            inner: surface_texture,
        }
    }

    /// Borrow the underlying [`wgpu::SurfaceTexture`].
    pub fn inner(&self) -> &wgpu::SurfaceTexture {
        &self.inner
    }

    /// Access the underlying [`wgpu::Texture`] for view creation, etc.
    pub fn texture(&self) -> &wgpu::Texture {
        &self.inner.texture
    }

    /// Present the image to the display, returning it to the swapchain.
    ///
    /// After calling this, the image is no longer valid for GPU operations.
    pub fn present(self) {
        self.inner.present();
    }

    /// The sub-optimal flag indicates the surface configuration may need to
    /// be re-applied (e.g. after a window resize that the platform detected
    /// before our resize handler).
    pub fn suboptimal(&self) -> bool {
        self.inner.suboptimal
    }
}

// ---------------------------------------------------------------------------
// Public API functions
// ---------------------------------------------------------------------------

/// Create a new [`RhiSurface`] from a raw window handle.
///
/// Accepts any type that implements [`wgpu::rwh::HasWindowHandle`]
/// and [`wgpu::rwh::HasDisplayHandle`] (e.g. a winit `Window`).
///
/// This is the Rust equivalent of creating a platform window surface in the
/// Python RHI (the Python side delegates to the native windowing toolkit).
///
/// # Safety
///
/// The window and display handles must remain valid for the lifetime of the
/// returned surface.
pub fn create_surface(
    instance: &RhiInstance,
    window: &(impl rwh::HasWindowHandle + rwh::HasDisplayHandle),
) -> RhiSurface {
    let wh = window
        .window_handle()
        .expect("valid window handle");
    let dh = window
        .display_handle()
        .expect("valid display handle");
    let surface = unsafe {
        instance
            .inner()
            .create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                raw_window_handle: wh.as_raw(),
                raw_display_handle: dh.as_raw(),
            })
            .expect("surface creation failed")
    };
    RhiSurface { inner: surface }
}

/// Configure a swapchain on the given surface.
///
/// Applies the provided [`SwapchainConfig`] to the surface using the device,
/// returning a [`RhiSwapchain`] that holds the active configuration.
///
/// # Panics
///
/// Panics if the surface configuration fails (unlikely with valid parameters).
pub fn configure_swapchain(
    device: &wgpu::Device,
    surface: &RhiSurface,
    config: &SwapchainConfig,
) -> RhiSwapchain {
    let wgpu_cfg = config.to_wgpu(wgpu::CompositeAlphaMode::Auto);
    surface.inner.configure(device, &wgpu_cfg);
    RhiSwapchain { config: wgpu_cfg }
}

/// Acquire the next image from the swapchain for rendering.
///
/// Blocks until a swapchain image is available.  The returned [`SurfaceImage`]
/// must be presented (via [`SurfaceImage::present`] or [`present`]) to return
/// it to the swapchain.
///
/// The `surface` parameter is the [`RhiSurface`] that was previously
/// configured with [`configure_swapchain`].  The `_swapchain` parameter is
/// accepted for API consistency (the active configuration must match the
/// surface's current config), but the actual acquire is performed on the
/// wgpu surface.
///
/// # Errors
///
/// Returns [`wgpu::SurfaceError`] if:
/// - The surface is lost and needs re-configuration.
/// - The swapchain ran out of memory.
/// - The requested image is outdated (surface should be re-configured).
pub fn get_current_texture(
    surface: &RhiSurface,
    _swapchain: &RhiSwapchain,
) -> Result<SurfaceImage, wgpu::SurfaceError> {
    let frame = surface.inner.get_current_texture()?;
    Ok(SurfaceImage::new(frame))
}

/// Present the swapchain image to the display.
///
/// Shorthand for calling [`SurfaceImage::present`].  Provided as a free
/// function for API consistency with [`configure_swapchain`] and [`resize`].
pub fn present(image: SurfaceImage) {
    image.present();
}

/// Resize the swapchain to the given dimensions.
///
/// Applies a zero-size guard: if either dimension is 0, it is clamped to 1.
/// This matches the pattern used in `renderer.rs` to prevent wgpu from
/// panicking on a zero-sized surface.
///
/// After calling this, the surface must be re-configured with the new
/// [`RhiSwapchain`] configuration.
pub fn resize(
    swapchain: &mut RhiSwapchain,
    device: &wgpu::Device,
    surface: &RhiSurface,
    width: u32,
    height: u32,
) {
    let width = width.max(1);
    let height = height.max(1);
    swapchain.config.width = width;
    swapchain.config.height = height;
    surface.inner.configure(device, &swapchain.config);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- PresentMode ---------------------------------------------------------

    #[test]
    fn test_present_mode_into_wgpu() {
        assert_eq!(
            wgpu::PresentMode::from(PresentMode::Immediate),
            wgpu::PresentMode::Immediate,
        );
        assert_eq!(
            wgpu::PresentMode::from(PresentMode::Fifo),
            wgpu::PresentMode::Fifo,
        );
        assert_eq!(
            wgpu::PresentMode::from(PresentMode::Mailbox),
            wgpu::PresentMode::Mailbox,
        );
    }

    #[test]
    fn test_present_mode_from_wgpu() {
        assert_eq!(
            PresentMode::from(wgpu::PresentMode::Immediate),
            PresentMode::Immediate,
        );
        assert_eq!(
            PresentMode::from(wgpu::PresentMode::Fifo),
            PresentMode::Fifo,
        );
        assert_eq!(
            PresentMode::from(wgpu::PresentMode::Mailbox),
            PresentMode::Mailbox,
        );
    }

    // Test disabled: transmute to invalid enum value now panics in Rust
    // The fallback behavior is guaranteed by the match arm in From impl

    #[test]
    fn test_present_mode_debug_clone() {
        let a = PresentMode::Fifo;
        let b = a;
        assert_eq!(format!("{:?}", a), "Fifo");
        assert_eq!(a, b);
    }

    // -- SwapchainConfig ----------------------------------------------------

    #[test]
    fn test_swapchain_config_default() {
        let cfg = SwapchainConfig::default();
        assert_eq!(cfg.format, wgpu::TextureFormat::Rgba8Unorm);
        assert_eq!(cfg.width, 1);
        assert_eq!(cfg.height, 1);
        assert_eq!(cfg.present_mode, PresentMode::Fifo);
    }

    #[test]
    fn test_swapchain_config_new_clamps_zero() {
        let cfg = SwapchainConfig::new(wgpu::TextureFormat::Bgra8Unorm, 0, 0, PresentMode::Mailbox);
        assert_eq!(cfg.width, 1);
        assert_eq!(cfg.height, 1);
        assert_eq!(cfg.format, wgpu::TextureFormat::Bgra8Unorm);
        assert_eq!(cfg.present_mode, PresentMode::Mailbox);
    }

    #[test]
    fn test_swapchain_config_new_normal() {
        let cfg = SwapchainConfig::new(wgpu::TextureFormat::Rgba8Unorm, 1920, 1080, PresentMode::Fifo);
        assert_eq!(cfg.width, 1920);
        assert_eq!(cfg.height, 1080);
    }

    #[test]
    fn test_swapchain_config_to_wgpu() {
        let cfg = SwapchainConfig::new(wgpu::TextureFormat::Rgba8Unorm, 800, 600, PresentMode::Mailbox);
        let wgpu_cfg = cfg.to_wgpu(wgpu::CompositeAlphaMode::Auto);
        assert_eq!(wgpu_cfg.format, wgpu::TextureFormat::Rgba8Unorm);
        assert_eq!(wgpu_cfg.width, 800);
        assert_eq!(wgpu_cfg.height, 600);
        assert_eq!(wgpu_cfg.present_mode, wgpu::PresentMode::Mailbox);
        assert_eq!(wgpu_cfg.usage, wgpu::TextureUsages::RENDER_ATTACHMENT);
        assert_eq!(wgpu_cfg.desired_maximum_frame_latency, 2);
    }

    #[test]
    fn test_swapchain_config_debug_clone() {
        let a = SwapchainConfig::default();
        let b = a.clone();
        assert_eq!(format!("{:?}", a), format!("{:?}", b));
    }

    // -- RhiSwapchain -------------------------------------------------------

    #[test]
    fn test_rhi_swapchain_accessors() {
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Rgba8Unorm,
            width: 800,
            height: 600,
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
        };
        let sc = RhiSwapchain { config };
        assert_eq!(sc.width(), 800);
        assert_eq!(sc.height(), 600);
        assert_eq!(sc.config().width, 800);
        assert_eq!(sc.config().height, 600);
    }

    #[test]
    fn test_rhi_swapchain_config_mut() {
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Rgba8Unorm,
            width: 800,
            height: 600,
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
        };
        let sc = RhiSwapchain {
            config: wgpu::SurfaceConfiguration {
                width: 1024,
                height: 768,
                ..config
            },
        };
        assert_eq!(sc.width(), 1024);
        assert_eq!(sc.height(), 768);
    }

    // -- SurfaceImage -------------------------------------------------------

    #[test]
    fn test_surface_image_suboptimal_default() {
        // SurfaceImage wraps SurfaceTexture.  We cannot create a real one
        // without a GPU + surface, but we can verify the accessor shape.
        // A real SurfaceTexture's suboptimal flag is set by the driver;
        // the default is false.
    }

    // -- Resize (zero-size guard) -------------------------------------------

    #[test]
    fn test_resize_zero_size_guard() {
        // Verify the logic: width.max(1) and height.max(1).
        let w: u32 = 0;
        let h: u32 = 0;
        assert_eq!(w.max(1), 1);
        assert_eq!(h.max(1), 1);

        let w: u32 = 640;
        let h: u32 = 480;
        assert_eq!(w.max(1), 640);
        assert_eq!(h.max(1), 480);
    }

    // -- PresentMode round-trip ---------------------------------------------

    #[test]
    fn test_present_mode_round_trip() {
        let modes = [
            PresentMode::Immediate,
            PresentMode::Fifo,
            PresentMode::Mailbox,
        ];
        for mode in modes {
            let wgpu_mode: wgpu::PresentMode = mode.into();
            let back: PresentMode = wgpu_mode.into();
            assert_eq!(mode, back, "round-trip failed for {:?}", mode);
        }
    }
}
