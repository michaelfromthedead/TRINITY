//! RHI resource mapping layer -- buffers, textures, samplers.
//!
//! This module mirrors the Python RHI ABCs in `engine/platform/rhi/resources.py`,
//! providing Rust-native wrappers around wgpu resource types with mapping
//! enums that bridge the logical Python types to wgpu backend types.
//!
//! | Rust type / fn | Python counterpart |
//! |---|---|
//! | `MemoryType` | `MemoryType` enum |
//! | `TextureType` | `TextureType` enum |
//! | `RhiBuffer` | `Buffer` ABC |
//! | `RhiTexture` | `Texture` ABC |
//! | `RhiSampler` | `Sampler` ABC |
//! | `create_buffer()` | `Device.create_buffer()` |
//! | `create_texture()` | `Device.create_texture()` |
//! | `create_sampler()` | `Device.create_sampler()` |

use crate::rhi_device::RhiDevice;

// ---------------------------------------------------------------------------
// MemoryType
// ---------------------------------------------------------------------------

/// Logical memory type for buffer allocation.
///
/// Mirrors the Python [`MemoryType`] enum and maps to [`wgpu::BufferUsages`]
/// for buffer creation.
///
/// [`MemoryType`]: ../../engine/platform/rhi/resources.py
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryType {
    /// Host-visible, host-writable.  Used for uploading data from CPU to GPU.
    Upload,
    /// Host-visible, host-readable.  Used for reading data back from GPU to CPU.
    Download,
    /// Device-local only.  Optimal for GPU access; not host-visible.
    DeviceLocal,
    /// Temporary buffer for GPU-to-GPU transfers (copy source and destination).
    Staging,
}

impl MemoryType {
    /// Returns the corresponding [`wgpu::BufferUsages`] for this memory type.
    pub fn buffer_usages(self) -> wgpu::BufferUsages {
        match self {
            Self::Upload => {
                wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC
            }
            Self::Download => {
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST
            }
            Self::DeviceLocal => {
                wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE
            }
            Self::Staging => {
                wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST
            }
        }
    }
}

impl From<MemoryType> for wgpu::BufferUsages {
    fn from(mt: MemoryType) -> Self {
        mt.buffer_usages()
    }
}

// ---------------------------------------------------------------------------
// TextureType
// ---------------------------------------------------------------------------

/// Texture dimensionality.
///
/// Mirrors the Python [`TextureType`] enum.
///
/// [`TextureType`]: ../../engine/platform/rhi/resources.py
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureType {
    /// 2D texture (width x height).
    D2,
    /// 3D texture (width x height x depth).
    D3,
    /// Cube-map texture (6 faces, each width x height).
    Cube,
    /// 2D array texture (width x height x array layers).
    Array,
}

// ---------------------------------------------------------------------------
// RhiBuffer
// ---------------------------------------------------------------------------

/// A GPU buffer wrapping a [`wgpu::Buffer`] with its allocation size.
///
/// This is the Rust counterpart of the Python [`Buffer`] ABC.
///
/// [`Buffer`]: ../../engine/platform/rhi/resources.py
#[derive(Debug)]
pub struct RhiBuffer {
    inner: wgpu::Buffer,
    size: u64,
}

impl RhiBuffer {
    /// Wrap an existing [`wgpu::Buffer`].
    pub fn new(buffer: wgpu::Buffer, size: u64) -> Self {
        Self {
            inner: buffer,
            size,
        }
    }

    /// Borrow the underlying [`wgpu::Buffer`].
    pub fn inner(&self) -> &wgpu::Buffer {
        &self.inner
    }

    /// Allocation size in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Consume the wrapper and return the underlying [`wgpu::Buffer`].
    pub fn into_inner(self) -> wgpu::Buffer {
        self.inner
    }

    /// Slice covering the full buffer.
    pub fn slice(&self) -> wgpu::BufferSlice<'_> {
        self.inner.slice(..)
    }
}

// ---------------------------------------------------------------------------
// RhiTexture
// ---------------------------------------------------------------------------

/// A GPU texture wrapping a [`wgpu::Texture`] together with its logical type.
///
/// This is the Rust counterpart of the Python [`Texture`] ABC.
///
/// [`Texture`]: ../../engine/platform/rhi/resources.py
#[derive(Debug)]
pub struct RhiTexture {
    inner: wgpu::Texture,
    tex_type: TextureType,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    depth: u32,
}

impl RhiTexture {
    /// Wrap an existing [`wgpu::Texture`].
    pub fn new(
        texture: wgpu::Texture,
        tex_type: TextureType,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        depth: u32,
    ) -> Self {
        Self {
            inner: texture,
            tex_type,
            format,
            width,
            height,
            depth,
        }
    }

    /// Borrow the underlying [`wgpu::Texture`].
    pub fn inner(&self) -> &wgpu::Texture {
        &self.inner
    }

    /// The texture dimension type.
    pub fn tex_type(&self) -> TextureType {
        self.tex_type
    }

    /// The pixel format.
    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    /// Width in texels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height in texels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Depth (or array layer count for Array / Cube textures).
    pub fn depth(&self) -> u32 {
        self.depth
    }

    /// Create a default [`wgpu::TextureView`] covering the full texture.
    pub fn create_view(&self) -> wgpu::TextureView {
        self.inner
            .create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// Consume the wrapper and return the underlying [`wgpu::Texture`].
    pub fn into_inner(self) -> wgpu::Texture {
        self.inner
    }
}

// ---------------------------------------------------------------------------
// RhiSampler
// ---------------------------------------------------------------------------

/// A GPU sampler wrapping a [`wgpu::Sampler`].
///
/// This is the Rust counterpart of the Python [`Sampler`] ABC.
///
/// [`Sampler`]: ../../engine/platform/rhi/resources.py
#[derive(Debug)]
pub struct RhiSampler {
    inner: wgpu::Sampler,
}

impl RhiSampler {
    /// Wrap an existing [`wgpu::Sampler`].
    pub fn new(sampler: wgpu::Sampler) -> Self {
        Self { inner: sampler }
    }

    /// Borrow the underlying [`wgpu::Sampler`].
    pub fn inner(&self) -> &wgpu::Sampler {
        &self.inner
    }

    /// Consume the wrapper and return the underlying [`wgpu::Sampler`].
    pub fn into_inner(self) -> wgpu::Sampler {
        self.inner
    }
}

// ---------------------------------------------------------------------------
// Helper: create_buffer
// ---------------------------------------------------------------------------

/// Create a [`wgpu::Buffer`] with the given [`MemoryType`] and size (in bytes).
///
/// The returned buffer is wrapped in an [`RhiBuffer`].  The usages are derived
/// from the memory type:
///
/// | `MemoryType` | `wgpu::BufferUsages` |
/// |---|---|
/// | `Upload` | `MAP_WRITE | COPY_SRC` |
/// | `Download` | `MAP_READ | COPY_DST` |
/// | `DeviceLocal` | `COPY_DST | STORAGE` |
/// | `Staging` | `COPY_SRC | COPY_DST` |
///
/// Upload and Download buffers are created already mapped so that the CPU can
/// immediately read or write.
///
/// For buffers that need additional usages (e.g. `VERTEX`, `UNIFORM`), use
/// [`create_buffer_with_usages`] instead.
pub fn create_buffer(device: &RhiDevice, mem_type: MemoryType, size: u64) -> RhiBuffer {
    let usages = mem_type.buffer_usages();
    let mapped_at_creation = matches!(mem_type, MemoryType::Upload | MemoryType::Download);
    let buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: usages,
        mapped_at_creation,
    });
    RhiBuffer::new(buffer, size)
}

/// Create a [`wgpu::Buffer`] with explicit [`wgpu::BufferUsages`] combined
/// with the base usages derived from [`MemoryType`].
///
/// This is useful when you need a device-local buffer that doubles as a vertex
/// buffer:
///
/// ```ignore
/// let vb = create_buffer_with_usages(
///     &device,
///     MemoryType::DeviceLocal,
///     1024,
///     wgpu::BufferUsages::VERTEX,
/// );
/// ```
pub fn create_buffer_with_usages(
    device: &RhiDevice,
    mem_type: MemoryType,
    size: u64,
    extra_usages: wgpu::BufferUsages,
) -> RhiBuffer {
    let usages = mem_type.buffer_usages() | extra_usages;
    let mapped_at_creation = matches!(mem_type, MemoryType::Upload | MemoryType::Download);
    let buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: usages,
        mapped_at_creation,
    });
    RhiBuffer::new(buffer, size)
}

// ---------------------------------------------------------------------------
// Helper: create_texture
// ---------------------------------------------------------------------------

/// Create a [`wgpu::Texture`] with the given parameters, wrapped in an
/// [`RhiTexture`].
///
/// The [`TextureType`] controls the [`wgpu::TextureDimension`] and
/// `array_layer_count`:
///
/// | `TextureType` | Dimension | array_layer_count |
/// |---|---|---|
/// | `D2` | D2 | 1 |
/// | `D3` | D3 | 1 |
/// | `Cube` | D2 | 6 |
/// | `Array` | D2 | `depth` |
///
/// For `D2` and `Cube` textures the `depth` parameter should be `1`; for `D3`
/// textures it represents the Z extent; for `Array` textures it is the number
/// of array layers.
///
/// The created texture includes `TEXTURE_BINDING`, `COPY_DST`, and `COPY_SRC`
/// usages.  If additional usages (e.g. `RENDER_ATTACHMENT`) are required, call
/// [`wgpu::Device::create_texture`] directly.
pub fn create_texture(
    device: &RhiDevice,
    tex_type: TextureType,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    depth: u32,
) -> RhiTexture {
    let (dimension, array_layer_count) = match tex_type {
        TextureType::D2 => (wgpu::TextureDimension::D2, 1),
        TextureType::D3 => (wgpu::TextureDimension::D3, 1),
        TextureType::Cube => (wgpu::TextureDimension::D2, 6),
        TextureType::Array => (wgpu::TextureDimension::D2, depth.max(1)),
    };

    let texture = device.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: array_layer_count,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    RhiTexture::new(texture, tex_type, format, width, height, depth)
}

// ---------------------------------------------------------------------------
// Helper: create_sampler
// ---------------------------------------------------------------------------

/// Create a [`wgpu::Sampler`] with the given filter and address modes, wrapped
/// in an [`RhiSampler`].
///
/// * `filter_mode` -- `true` for linear filtering, `false` for nearest.
/// * `address_mode` -- the wrap/clamp/mirror mode for all three axes (U, V, W).
pub fn create_sampler(
    device: &RhiDevice,
    filter_mode: bool,
    address_mode: wgpu::AddressMode,
) -> RhiSampler {
    let filter = if filter_mode {
        wgpu::FilterMode::Linear
    } else {
        wgpu::FilterMode::Nearest
    };

    let sampler = device.device.create_sampler(&wgpu::SamplerDescriptor {
        label: None,
        address_mode_u: address_mode,
        address_mode_v: address_mode,
        address_mode_w: address_mode,
        mag_filter: filter,
        min_filter: filter,
        mipmap_filter: filter,
        lod_min_clamp: 0.0,
        lod_max_clamp: 32.0,
        compare: None,
        anisotropy_clamp: 1,
        border_color: None,
    });

    RhiSampler::new(sampler)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rhi_device::{create_instance, request_device, FeatureFlags, QualityTier};

    // -- MemoryType ---------------------------------------------------------

    #[test]
    fn test_memory_type_upload_usages() {
        let usages = MemoryType::Upload.buffer_usages();
        assert!(usages.contains(wgpu::BufferUsages::MAP_WRITE));
        assert!(usages.contains(wgpu::BufferUsages::COPY_SRC));
    }

    #[test]
    fn test_memory_type_download_usages() {
        let usages = MemoryType::Download.buffer_usages();
        assert!(usages.contains(wgpu::BufferUsages::MAP_READ));
        assert!(usages.contains(wgpu::BufferUsages::COPY_DST));
    }

    #[test]
    fn test_memory_type_device_local_usages() {
        let usages = MemoryType::DeviceLocal.buffer_usages();
        assert!(usages.contains(wgpu::BufferUsages::COPY_DST));
        assert!(usages.contains(wgpu::BufferUsages::STORAGE));
        assert!(!usages.contains(wgpu::BufferUsages::MAP_WRITE));
        assert!(!usages.contains(wgpu::BufferUsages::MAP_READ));
    }

    #[test]
    fn test_memory_type_staging_usages() {
        let usages = MemoryType::Staging.buffer_usages();
        assert!(usages.contains(wgpu::BufferUsages::COPY_SRC));
        assert!(usages.contains(wgpu::BufferUsages::COPY_DST));
    }

    #[test]
    fn test_memory_type_from_trait() {
        let usages: wgpu::BufferUsages = MemoryType::Upload.into();
        assert!(usages.contains(wgpu::BufferUsages::MAP_WRITE));
    }

    #[test]
    fn test_memory_type_debug_clone() {
        let a = MemoryType::DeviceLocal;
        let b = a;
        assert_eq!(format!("{:?}", a), "DeviceLocal");
        assert_eq!(a, b);
    }

    // -- TextureType --------------------------------------------------------

    #[test]
    fn test_texture_type_debug_clone() {
        let t = TextureType::D2;
        assert_eq!(t, TextureType::D2);
        assert_eq!(format!("{:?}", t), "D2");
        assert_ne!(t, TextureType::D3);
    }

    #[test]
    fn test_texture_type_variants() {
        assert_eq!(format!("{:?}", TextureType::D2), "D2");
        assert_eq!(format!("{:?}", TextureType::D3), "D3");
        assert_eq!(format!("{:?}", TextureType::Cube), "Cube");
        assert_eq!(format!("{:?}", TextureType::Array), "Array");
    }

    // -- RhiBuffer construction (requires GPU) ------------------------------

    #[test]
    fn test_create_buffer_upload() {
        let instance = create_instance();
        if let Some(adapter) = pollster::block_on(
            instance
                .inner()
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }),
        ) {
            let dev = request_device(&adapter, FeatureFlags::empty(), QualityTier::Low);
            let buf = create_buffer(&dev, MemoryType::Upload, 256);
            assert_eq!(buf.size(), 256);
            assert!(buf.size() >= 256);
        }
    }

    #[test]
    fn test_create_buffer_download() {
        let instance = create_instance();
        if let Some(adapter) = pollster::block_on(
            instance
                .inner()
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }),
        ) {
            let dev = request_device(&adapter, FeatureFlags::empty(), QualityTier::Low);
            let buf = create_buffer(&dev, MemoryType::Download, 128);
            assert_eq!(buf.size(), 128);
        }
    }

    #[test]
    fn test_create_buffer_device_local() {
        let instance = create_instance();
        if let Some(adapter) = pollster::block_on(
            instance
                .inner()
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }),
        ) {
            let dev = request_device(&adapter, FeatureFlags::empty(), QualityTier::Low);
            let buf = create_buffer(&dev, MemoryType::DeviceLocal, 1024);
            assert_eq!(buf.size(), 1024);
        }
    }

    #[test]
    fn test_create_buffer_staging() {
        let instance = create_instance();
        if let Some(adapter) = pollster::block_on(
            instance
                .inner()
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }),
        ) {
            let dev = request_device(&adapter, FeatureFlags::empty(), QualityTier::Low);
            let buf = create_buffer(&dev, MemoryType::Staging, 512);
            assert_eq!(buf.size(), 512);
        }
    }

    #[test]
    fn test_create_buffer_with_usages_extra() {
        let instance = create_instance();
        if let Some(adapter) = pollster::block_on(
            instance
                .inner()
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }),
        ) {
            let dev = request_device(&adapter, FeatureFlags::empty(), QualityTier::Low);
            // Device-local + VERTEX for use as a vertex buffer.
            let buf = create_buffer_with_usages(
                &dev,
                MemoryType::DeviceLocal,
                256,
                wgpu::BufferUsages::VERTEX,
            );
            assert_eq!(buf.size(), 256);
        }
    }

    // -- RhiTexture creation (requires GPU) ---------------------------------

    #[test]
    fn test_create_texture_d2() {
        let instance = create_instance();
        if let Some(adapter) = pollster::block_on(
            instance
                .inner()
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }),
        ) {
            let dev = request_device(&adapter, FeatureFlags::empty(), QualityTier::Low);
            let tex = create_texture(
                &dev,
                TextureType::D2,
                wgpu::TextureFormat::Rgba8Unorm,
                64,
                64,
                1,
            );
            assert_eq!(tex.width(), 64);
            assert_eq!(tex.height(), 64);
            assert_eq!(tex.depth(), 1);
            assert_eq!(tex.tex_type(), TextureType::D2);
            assert_eq!(tex.format(), wgpu::TextureFormat::Rgba8Unorm);
        }
    }

    #[test]
    fn test_create_texture_d3() {
        let instance = create_instance();
        if let Some(adapter) = pollster::block_on(
            instance
                .inner()
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }),
        ) {
            let dev = request_device(&adapter, FeatureFlags::empty(), QualityTier::Low);
            let tex = create_texture(
                &dev,
                TextureType::D3,
                wgpu::TextureFormat::Rgba8Unorm,
                16,
                16,
                8,
            );
            assert_eq!(tex.width(), 16);
            assert_eq!(tex.height(), 16);
            assert_eq!(tex.depth(), 8);
            assert_eq!(tex.tex_type(), TextureType::D3);
        }
    }

    #[test]
    fn test_create_texture_cube() {
        let instance = create_instance();
        if let Some(adapter) = pollster::block_on(
            instance
                .inner()
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }),
        ) {
            let dev = request_device(&adapter, FeatureFlags::empty(), QualityTier::Low);
            let tex = create_texture(
                &dev,
                TextureType::Cube,
                wgpu::TextureFormat::Rgba8Unorm,
                128,
                128,
                1,
            );
            assert_eq!(tex.width(), 128);
            assert_eq!(tex.tex_type(), TextureType::Cube);
        }
    }

    #[test]
    fn test_create_texture_array() {
        let instance = create_instance();
        if let Some(adapter) = pollster::block_on(
            instance
                .inner()
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }),
        ) {
            let dev = request_device(&adapter, FeatureFlags::empty(), QualityTier::Low);
            let tex = create_texture(
                &dev,
                TextureType::Array,
                wgpu::TextureFormat::Rgba8Unorm,
                32,
                32,
                4,
            );
            assert_eq!(tex.width(), 32);
            assert_eq!(tex.depth(), 4);
            assert_eq!(tex.tex_type(), TextureType::Array);
        }
    }

    #[test]
    fn test_texture_create_view() {
        let instance = create_instance();
        if let Some(adapter) = pollster::block_on(
            instance
                .inner()
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }),
        ) {
            let dev = request_device(&adapter, FeatureFlags::empty(), QualityTier::Low);
            let tex = create_texture(
                &dev,
                TextureType::D2,
                wgpu::TextureFormat::Rgba8Unorm,
                16,
                16,
                1,
            );
            let _view = tex.create_view();
            // Note: wgpu 24 removed the simple dimension() accessor on
            // TextureView; the view is validated by creation succeeding.
        }
    }

    // -- RhiSampler creation (requires GPU) ---------------------------------

    #[test]
    fn test_create_sampler_nearest_clamp() {
        let instance = create_instance();
        if let Some(adapter) = pollster::block_on(
            instance
                .inner()
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }),
        ) {
            let dev = request_device(&adapter, FeatureFlags::empty(), QualityTier::Low);
            let sampler = create_sampler(&dev, false, wgpu::AddressMode::ClampToEdge);
            let _inner = sampler.inner();
        }
    }

    #[test]
    fn test_create_sampler_linear_repeat() {
        let instance = create_instance();
        if let Some(adapter) = pollster::block_on(
            instance
                .inner()
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }),
        ) {
            let dev = request_device(&adapter, FeatureFlags::empty(), QualityTier::Low);
            let sampler = create_sampler(&dev, true, wgpu::AddressMode::Repeat);
            let _inner = sampler.inner();
        }
    }

    #[test]
    fn test_create_sampler_mirror() {
        let instance = create_instance();
        if let Some(adapter) = pollster::block_on(
            instance
                .inner()
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }),
        ) {
            let dev = request_device(&adapter, FeatureFlags::empty(), QualityTier::Low);
            let sampler = create_sampler(&dev, true, wgpu::AddressMode::MirrorRepeat);
            let _inner = sampler.inner();
        }
    }

    // -- RhiBuffer / RhiTexture / RhiSampler accessor tests (with GPU) ------

    #[test]
    fn test_rhi_buffer_into_inner() {
        let instance = create_instance();
        if let Some(adapter) = pollster::block_on(
            instance
                .inner()
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }),
        ) {
            let dev = request_device(&adapter, FeatureFlags::empty(), QualityTier::Low);
            let buf = create_buffer(&dev, MemoryType::Upload, 64);
            let _wgpu_buf = buf.into_inner();
        }
    }

    #[test]
    fn test_rhi_texture_into_inner() {
        let instance = create_instance();
        if let Some(adapter) = pollster::block_on(
            instance
                .inner()
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }),
        ) {
            let dev = request_device(&adapter, FeatureFlags::empty(), QualityTier::Low);
            let tex = create_texture(
                &dev,
                TextureType::D2,
                wgpu::TextureFormat::Rgba8Unorm,
                8,
                8,
                1,
            );
            let _wgpu_tex = tex.into_inner();
        }
    }

    #[test]
    fn test_rhi_sampler_into_inner() {
        let instance = create_instance();
        if let Some(adapter) = pollster::block_on(
            instance
                .inner()
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }),
        ) {
            let dev = request_device(&adapter, FeatureFlags::empty(), QualityTier::Low);
            let sampler = create_sampler(&dev, false, wgpu::AddressMode::ClampToEdge);
            let _wgpu_sampler = sampler.into_inner();
        }
    }

    // -- Integration smoke test ---------------------------------------------

    #[test]
    fn test_create_buffer_texture_sampler_smoke() {
        let instance = create_instance();
        if let Some(adapter) = pollster::block_on(
            instance
                .inner()
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }),
        ) {
            let dev = request_device(&adapter, FeatureFlags::empty(), QualityTier::Low);

            // Create all three resource types in sequence.
            let _buf = create_buffer(&dev, MemoryType::DeviceLocal, 4096);
            let _tex = create_texture(
                &dev,
                TextureType::D2,
                wgpu::TextureFormat::Rgba8Unorm,
                256,
                256,
                1,
            );
            let _smp = create_sampler(&dev, true, wgpu::AddressMode::Repeat);

            // Device should still be responsive after all creations.
            dev.wait_idle();
        }
    }
}
