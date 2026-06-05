//! Minimal wgpu bootstrap for demoscene rendering (T-DEMO-5.1 + T-DEMO-5.2).
//!
//! Provides lightweight GPU device and window/presentation setup targeting
//! the 4K mode constraint (<100 lines per component).

use std::sync::Arc;

/// Error type for bootstrap operations.
#[derive(Debug)]
pub enum BootstrapError {
    /// No suitable GPU adapter found.
    NoAdapter,
    /// Device request failed.
    DeviceFailed(wgpu::RequestDeviceError),
    /// Surface configuration failed.
    SurfaceFailed(String),
}

impl std::fmt::Display for BootstrapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoAdapter => write!(f, "no suitable GPU adapter"),
            Self::DeviceFailed(e) => write!(f, "device request failed: {e}"),
            Self::SurfaceFailed(s) => write!(f, "surface failed: {s}"),
        }
    }
}

impl std::error::Error for BootstrapError {}

/// T-DEMO-5.1: Minimal wgpu standalone bootstrap.
///
/// Holds wgpu Device, Queue in a compact struct for demoscene effects.
pub struct DemoBootstrap {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_name: String,
}

impl DemoBootstrap {
    /// Create a headless GPU context (no surface required).
    pub async fn new() -> Result<Self, BootstrapError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .ok_or(BootstrapError::NoAdapter)?;
        let name = adapter.get_info().name;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("DemoBootstrap"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(BootstrapError::DeviceFailed)?;
        Ok(Self { device, queue, adapter_name: name })
    }

    /// Blocking version using pollster.
    pub fn new_blocking() -> Result<Self, BootstrapError> {
        pollster::block_on(Self::new())
    }
}

/// T-DEMO-5.2: Minimal window/presentation layer.
///
/// Manages a wgpu Surface tied to a window for frame presentation.
pub struct DemoWindow {
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
}

impl DemoWindow {
    /// Create from an existing DemoBootstrap and window handle.
    ///
    /// # Safety
    /// Window handle must remain valid for the surface lifetime.
    pub unsafe fn from_bootstrap<W>(
        bootstrap: DemoBootstrap,
        window: &W,
        width: u32,
        height: u32,
    ) -> Result<Self, BootstrapError>
    where
        W: wgpu::rwh::HasWindowHandle + wgpu::rwh::HasDisplayHandle,
    {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let wh = window.window_handle().map_err(|e| BootstrapError::SurfaceFailed(e.to_string()))?;
        let dh = window.display_handle().map_err(|e| BootstrapError::SurfaceFailed(e.to_string()))?;
        let surface = instance
            .create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                raw_window_handle: wh.as_raw(),
                raw_display_handle: dh.as_raw(),
            })
            .map_err(|e| BootstrapError::SurfaceFailed(e.to_string()))?;
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            width: width.max(1),
            height: height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
        };
        let device = Arc::new(bootstrap.device);
        let queue = Arc::new(bootstrap.queue);
        surface.configure(&device, &config);
        Ok(Self { surface, config, device, queue })
    }

    /// Resize the swapchain.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.surface.configure(&self.device, &self.config);
    }

    /// Get current frame for rendering.
    pub fn current_frame(&self) -> Result<wgpu::SurfaceTexture, wgpu::SurfaceError> {
        self.surface.get_current_texture()
    }

    /// Present a rendered frame (consumes the texture).
    pub fn present(frame: wgpu::SurfaceTexture) {
        frame.present();
    }
}

#[cfg(test)]
#[path = "bootstrap_tests.rs"]
mod tests;
