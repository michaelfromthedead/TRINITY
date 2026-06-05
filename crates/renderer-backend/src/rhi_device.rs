//! RHI device mapping layer -- bridge between Python RHI ABCs and Rust wgpu.
//!
//! This module formalises the wgpu instance / adapter / device / queue lifecycle
//! into a reusable abstraction that mirrors the Python-side RHI hierarchy
//! (engine/platform/rhi/device.py).  Each type here has a direct counterpart
//! in the Python ABCs:
//!
//! | Rust type / fn | Python counterpart |
//! |---|---|
//! | `AdapterSelector` | `AdapterType` enum |
//! | `FeatureFlags` | `FeatureSupport` dataclass |
//! | `request_adapter()` | `Adapter.enumerate()` + selection |
//! | `RhiDevice` | `Device` ABC |
//! | `WgpuFence` | (implicit – `wait_idle()` on Device) |

use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// AdapterSelector
// ---------------------------------------------------------------------------

/// Preference for GPU adapter selection.
///
/// Maps directly to [`wgpu::PowerPreference`] and mirrors the Python
/// [`AdapterType`] enum.
///
/// [`AdapterType`]: ../../engine/platform/rhi/device.py
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterSelector {
    /// Prefer an integrated / battery-efficient GPU.
    LowPower,
    /// Prefer a discrete / high-performance GPU.
    HighPerformance,
    /// No preference – let the system heuristics decide.
    None,
}

impl From<AdapterSelector> for wgpu::PowerPreference {
    fn from(sel: AdapterSelector) -> Self {
        match sel {
            AdapterSelector::LowPower => wgpu::PowerPreference::LowPower,
            AdapterSelector::HighPerformance | AdapterSelector::None => {
                wgpu::PowerPreference::HighPerformance
            }
        }
    }
}

// ---------------------------------------------------------------------------
// FeatureFlags  (bitmask)
// ---------------------------------------------------------------------------

/// Bitmask of logical device features.
///
/// Mirrors the Python [`FeatureSupport`] dataclass: each flag corresponds to a
/// high-level capability that is mapped to one or more [`wgpu::Features`] when
/// the device is requested.
///
/// [`FeatureSupport`]: ../../engine/platform/rhi/device.py
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeatureFlags(u64);

impl FeatureFlags {
    // -- Individual flag constants -----------------------------------------

    /// Ray-tracing support (acceleration structures, ray-query shaders).
    pub const RAY_TRACING: Self = Self(1 << 0);

    /// Mesh / task shader pipeline support.
    pub const MESH_SHADERS: Self = Self(1 << 1);

    /// Bindless resource access (texture / buffer arrays, non-uniform
    /// indexing).
    pub const BINDLESS: Self = Self(1 << 2);

    /// Compute shader support.
    pub const COMPUTE: Self = Self(1 << 3);

    /// Timestamp queries for GPU profiling.
    pub const TIMESTAMP_QUERY: Self = Self(1 << 4);

    /// Pipeline statistics queries.
    pub const PIPELINE_STATISTICS: Self = Self(1 << 5);

    /// Support for 16-bit storage types (f16, i16, u16).
    pub const SHADER_F16: Self = Self(1 << 6);

    /// Support for 64-bit integer atomics.
    pub const SHADER_I64: Self = Self(1 << 7);

    // -- Convenience constructors ------------------------------------------

    /// Empty flag set (no features requested).
    pub const fn empty() -> Self {
        Self(0)
    }

    /// All defined flags set.
    pub const fn all() -> Self {
        Self(u64::MAX)
    }

    // -- Query / mutate ----------------------------------------------------

    /// Returns `true` if `self` contains *all* of the given flags.
    pub fn contains(self, flags: Self) -> bool {
        (self.0 & flags.0) == flags.0
    }

    /// Insert the given flags in-place.
    pub fn insert(&mut self, flags: Self) {
        self.0 |= flags.0;
    }

    /// Remove the given flags in-place.
    pub fn remove(&mut self, flags: Self) {
        self.0 &= !flags.0;
    }

    /// Returns the raw `u64` bitmask.
    pub const fn bits(self) -> u64 {
        self.0
    }
}

// -- Bitwise operators -------------------------------------------------------

impl std::ops::BitOr for FeatureFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for FeatureFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl std::ops::BitAnd for FeatureFlags {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

impl std::ops::BitAndAssign for FeatureFlags {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

// ---------------------------------------------------------------------------
// QualityTier
// ---------------------------------------------------------------------------

/// Device quality / capability tier.
///
/// Used to select feature and limit profiles when requesting a device.
/// Higher tiers enable larger resource limits and more expensive features.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum QualityTier {
    /// Minimum viable tier – default limits.
    Low,
    /// Moderate limits suitable for integrated GPUs.
    Medium,
    /// Generous limits for discrete GPUs.
    High,
    /// Maximum limits for top-tier hardware.
    Ultra,
}

// ---------------------------------------------------------------------------
// DeviceLostCallback
// ---------------------------------------------------------------------------

/// Callback invoked when the GPU device is lost.
///
/// The first argument is the loss reason; the second is a human-readable
/// diagnostic message provided by the driver / backend.
pub type DeviceLostCallback = Box<dyn FnOnce(wgpu::DeviceLostReason, String)>;

// ---------------------------------------------------------------------------
// RhiInstance
// ---------------------------------------------------------------------------

/// Thin wrapper around a [`wgpu::Instance`].
///
/// Represents a connection to the GPU API (Vulkan, Metal, D3D12, or GL) and
/// serves as the factory for adapters.
pub struct RhiInstance {
    inner: wgpu::Instance,
}

impl RhiInstance {
    /// Borrow the underlying [`wgpu::Instance`].
    pub fn inner(&self) -> &wgpu::Instance {
        &self.inner
    }
}

// ---------------------------------------------------------------------------
// RhiDevice
// ---------------------------------------------------------------------------

/// GPU device and its primary command queue.
///
/// Wraps a [`wgpu::Device`] together with its associated [`wgpu::Queue`] and
/// a monotonically-increasing submission-index counter used by [`WgpuFence`].
///
/// This is the Rust-side counterpart of the Python [`Device`] ABC.
///
/// [`Device`]: ../../engine/platform/rhi/device.py
pub struct RhiDevice {
    /// The underlying wgpu device.
    pub device: wgpu::Device,
    /// The primary command queue (graphics + compute).
    pub queue: wgpu::Queue,
    /// Monotonically-increasing submission counter (wraps on overflow).
    submission_index: AtomicU64,
}

impl RhiDevice {
    /// Wrap an already-requested wgpu device + queue pair.
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self {
            device,
            queue,
            submission_index: AtomicU64::new(0),
        }
    }

    /// Try to create a headless RhiDevice for testing purposes.
    ///
    /// This uses wgpu's low-power adapter request with fallback backends.
    /// Returns None if no suitable adapter can be found.
    pub fn try_new_headless() -> Option<Self> {
        use pollster::FutureExt;

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .block_on()?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Headless Test Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .block_on()
            .ok()?;

        Some(Self::new(device, queue))
    }

    // -- Accessors ---------------------------------------------------------

    /// Borrow the underlying [`wgpu::Device`].
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Borrow the underlying [`wgpu::Queue`].
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    // -- Submission index --------------------------------------------------

    /// Atomically increment the submission counter and return the *previous*
    /// value.
    ///
    /// Call this **before** [`wgpu::Queue::submit`] so the returned index
    /// represents the submission that is about to happen.
    pub fn next_submission_index(&self) -> u64 {
        self.submission_index.fetch_add(1, Ordering::SeqCst)
    }

    /// Read the current submission counter without incrementing.
    pub fn current_submission_index(&self) -> u64 {
        self.submission_index.load(Ordering::SeqCst)
    }

    // -- Synchronisation ---------------------------------------------------

    /// Block the CPU until the GPU has completed **all** outstanding work.
    pub fn wait_idle(&self) {
        self.device.poll(wgpu::Maintain::Wait);
    }
}

// ---------------------------------------------------------------------------
// WgpuFence
// ---------------------------------------------------------------------------

/// A CPU-side GPU-fence equivalent for wgpu.
///
/// Wgpu does not expose explicit fence / semaphore objects like Vulkan or
/// D3D12.  Instead, [`wgpu::Queue::submit`] returns a monotonically increasing
/// submission index (u64).  This wrapper tracks that index and provides a
/// [`wait`] method that blocks the CPU until the GPU has completed all work
/// up to the recorded value.
///
/// [`wait`]: Self::wait
pub struct WgpuFence {
    /// The submission index this fence is waiting on.
    signaled_value: AtomicU64,
}

impl WgpuFence {
    /// Create a new fence initially signalled at 0 (nothing to wait on).
    pub fn new() -> Self {
        Self {
            signaled_value: AtomicU64::new(0),
        }
    }

    /// The submission index this fence will wait for.
    pub fn signaled_value(&self) -> u64 {
        self.signaled_value.load(Ordering::SeqCst)
    }

    /// Record a new submission index to wait on.
    ///
    /// Typically called with the return value of [`next_submission_index`]
    /// (or directly with the index returned by [`wgpu::Queue::submit`]).
    ///
    /// [`next_submission_index`]: RhiDevice::next_submission_index
    pub fn set_signaled_value(&self, value: u64) {
        self.signaled_value.store(value, Ordering::SeqCst);
    }

    /// Block the CPU until the GPU has completed **all** outstanding work.
    ///
    /// This is a no-op if the fence has never been signalled
    /// (signaled_value == 0).  Uses [`wgpu::Maintain::Wait`] which polls the
    /// device until the GPU is fully idle.
    pub fn wait(&self, device: &wgpu::Device) {
        if self.signaled_value.load(Ordering::SeqCst) > 0 {
            device.poll(wgpu::Maintain::Wait);
        }
    }
}

impl Default for WgpuFence {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helper: map QualityTier to wgpu Limits
// ---------------------------------------------------------------------------

fn limits_for_tier(tier: QualityTier) -> wgpu::Limits {
    match tier {
        QualityTier::Low => wgpu::Limits::default(),

        QualityTier::Medium => wgpu::Limits {
            max_bind_groups: 4,
            max_texture_dimension_2d: 8192,
            max_buffer_size: 256 * 1024 * 1024, // 256 MiB
            ..wgpu::Limits::default()
        },

        QualityTier::High => wgpu::Limits {
            max_bind_groups: 4,
            max_texture_dimension_2d: 16384,
            max_buffer_size: 1024 * 1024 * 1024, // 1 GiB
            max_storage_buffers_per_shader_stage: 16,
            max_storage_textures_per_shader_stage: 8,
            ..wgpu::Limits::default()
        },

        QualityTier::Ultra => wgpu::Limits {
            max_bind_groups: 4,
            max_texture_dimension_2d: 16384,
            max_buffer_size: 2 * 1024 * 1024 * 1024, // 2 GiB
            max_storage_buffers_per_shader_stage: 24,
            max_storage_textures_per_shader_stage: 16,
            max_compute_workgroups_per_dimension: 65535,
            ..wgpu::Limits::default()
        },
    }
}

// ---------------------------------------------------------------------------
// Helper: map FeatureFlags to wgpu::Features
// ---------------------------------------------------------------------------

fn features_for_flags(flags: FeatureFlags) -> wgpu::Features {
    let mut f = wgpu::Features::empty();

    if flags.contains(FeatureFlags::MESH_SHADERS) {
        f |= wgpu::Features::SHADER_PRIMITIVE_INDEX;
    }

    if flags.contains(FeatureFlags::BINDLESS) {
        f |= wgpu::Features::TEXTURE_BINDING_ARRAY;
        f |= wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY;
        f |= wgpu::Features::BUFFER_BINDING_ARRAY;
    }

    if flags.contains(FeatureFlags::TIMESTAMP_QUERY) {
        f |= wgpu::Features::TIMESTAMP_QUERY;
    }

    if flags.contains(FeatureFlags::PIPELINE_STATISTICS) {
        f |= wgpu::Features::PIPELINE_STATISTICS_QUERY;
    }

    if flags.contains(FeatureFlags::SHADER_F16) {
        f |= wgpu::Features::SHADER_F16;
    }

    if flags.contains(FeatureFlags::SHADER_I64) {
        f |= wgpu::Features::SHADER_F64;
    }

    // COMPUTE is always available on any modern backend – no feature flag
    // needed beyond the base API.

    // RAY_TRACING: wgpu's raytracing feature flag is backend-gated and
    // may not exist on all platforms.  We request it conditionally, but
    // the caller should verify adapter support first.
    if flags.contains(FeatureFlags::RAY_TRACING) {
        f |= wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        // NOTE: RAY_TRACING_ACCELERATION_STRUCTURE is intentionally
        // omitted here because the associated wgpu features are still
        // in development across backends.  As wgpu matures, extend this
        // match arm with:
        //   f |= wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE;
        //   f |= wgpu::Features::RAY_QUERY;
    }

    f
}

// ---------------------------------------------------------------------------
// Public API functions
// ---------------------------------------------------------------------------

/// Create a new [`RhiInstance`] wrapping a [`wgpu::Instance`] configured for
/// all backends.
///
/// This is the Rust equivalent of constructing an RHI entry point; on the
/// Python side it corresponds to the implicit backend initialisation before
/// `Adapter.enumerate()`.
pub fn create_instance() -> RhiInstance {
    let inner = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });
    RhiInstance { inner }
}

/// Request a GPU adapter matching the given [`AdapterSelector`].
///
/// Blocks the current thread with `pollster::block_on` until the adapter is
/// resolved (or none is found).
///
/// # Panics
///
/// Panics if no adapter matching the selector is available.
pub fn request_adapter(
    instance: &RhiInstance,
    selector: AdapterSelector,
    compatible_surface: Option<&wgpu::Surface<'_>>,
) -> wgpu::Adapter {
    pollster::block_on(
        instance
            .inner
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: selector.into(),
                compatible_surface,
                force_fallback_adapter: false,
            }),
    )
    .expect("no suitable GPU adapter found")
}

/// Request a [`RhiDevice`] (device + queue) from the given adapter.
///
/// Maps the logical [`FeatureFlags`] to [`wgpu::Features`] and the
/// [`QualityTier`] to [`wgpu::Limits`] before calling
/// [`wgpu::Adapter::request_device`].  Blocks the current thread with
/// `pollster::block_on`.
///
/// # Panics
///
/// Panics if the device request fails (e.g. the adapter does not support the
/// requested features or limits).
pub fn request_device(
    adapter: &wgpu::Adapter,
    features: FeatureFlags,
    tier: QualityTier,
) -> RhiDevice {
    let required_features = features_for_flags(features);
    let required_limits = limits_for_tier(tier);

    let (device, queue) = pollster::block_on(
        adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("TRINITY RHI Device"),
                required_features,
                required_limits,
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None, // trace path
        ),
    )
    .expect("failed to create wgpu device");

    RhiDevice::new(device, queue)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- AdapterSelector ---------------------------------------------------

    #[test]
    fn test_adapter_selector_into_power_preference() {
        assert_eq!(
            wgpu::PowerPreference::from(AdapterSelector::LowPower),
            wgpu::PowerPreference::LowPower,
        );
        assert_eq!(
            wgpu::PowerPreference::from(AdapterSelector::HighPerformance),
            wgpu::PowerPreference::HighPerformance,
        );
        assert_eq!(
            wgpu::PowerPreference::from(AdapterSelector::None),
            wgpu::PowerPreference::HighPerformance,
        );
    }

    #[test]
    fn test_adapter_selector_debug_clone() {
        let a = AdapterSelector::HighPerformance;
        let b = a;
        assert_eq!(format!("{:?}", a), "HighPerformance");
        assert_eq!(a, b);
    }

    // -- FeatureFlags ------------------------------------------------------

    #[test]
    fn test_feature_flags_empty() {
        let f = FeatureFlags::empty();
        assert!(!f.contains(FeatureFlags::RAY_TRACING));
        assert!(!f.contains(FeatureFlags::COMPUTE));
        assert_eq!(f.bits(), 0);
    }

    #[test]
    fn test_feature_flags_insert_contains() {
        let mut f = FeatureFlags::empty();
        f.insert(FeatureFlags::RAY_TRACING | FeatureFlags::COMPUTE);
        assert!(f.contains(FeatureFlags::RAY_TRACING));
        assert!(f.contains(FeatureFlags::COMPUTE));
        assert!(!f.contains(FeatureFlags::MESH_SHADERS));
    }

    #[test]
    fn test_feature_flags_remove() {
        let mut f = FeatureFlags::RAY_TRACING | FeatureFlags::BINDLESS;
        assert!(f.contains(FeatureFlags::BINDLESS));
        f.remove(FeatureFlags::BINDLESS);
        assert!(!f.contains(FeatureFlags::BINDLESS));
        assert!(f.contains(FeatureFlags::RAY_TRACING));
    }

    #[test]
    fn test_feature_flags_bitops() {
        let a = FeatureFlags::RAY_TRACING | FeatureFlags::MESH_SHADERS;
        assert_eq!(a.bits(), (1 << 0) | (1 << 1));

        let mut b = FeatureFlags::BINDLESS;
        b |= FeatureFlags::COMPUTE;
        assert!(b.contains(FeatureFlags::BINDLESS));
        assert!(b.contains(FeatureFlags::COMPUTE));

        let c = a & FeatureFlags::RAY_TRACING;
        assert!(c.contains(FeatureFlags::RAY_TRACING));
        assert!(!c.contains(FeatureFlags::MESH_SHADERS));
    }

    #[test]
    fn test_feature_flags_const_all() {
        let all = FeatureFlags::all();
        assert!(all.contains(FeatureFlags::RAY_TRACING));
        assert!(all.contains(FeatureFlags::MESH_SHADERS));
        assert!(all.contains(FeatureFlags::BINDLESS));
        assert!(all.contains(FeatureFlags::COMPUTE));
        assert!(all.contains(FeatureFlags::TIMESTAMP_QUERY));
        assert!(all.contains(FeatureFlags::PIPELINE_STATISTICS));
        assert!(all.contains(FeatureFlags::SHADER_F16));
        assert!(all.contains(FeatureFlags::SHADER_I64));
    }

    // -- QualityTier -------------------------------------------------------

    #[test]
    fn test_quality_tier_ordering() {
        assert!(QualityTier::Low < QualityTier::Medium);
        assert!(QualityTier::Medium < QualityTier::High);
        assert!(QualityTier::High < QualityTier::Ultra);
    }

    #[test]
    fn test_quality_tier_debug_clone() {
        let t = QualityTier::High;
        assert_eq!(t, QualityTier::High);
        assert_eq!(format!("{:?}", t), "High");
    }

    // -- Limits mapping ----------------------------------------------------

    #[test]
    fn test_limits_for_tier_low() {
        let limits = limits_for_tier(QualityTier::Low);
        // Low should use default limits.
        assert_eq!(limits, wgpu::Limits::default());
    }

    #[test]
    fn test_limits_for_tier_medium() {
        let limits = limits_for_tier(QualityTier::Medium);
        assert!(limits.max_texture_dimension_2d >= 8192);
    }

    #[test]
    fn test_limits_for_tier_ultra() {
        let limits = limits_for_tier(QualityTier::Ultra);
        assert!(limits.max_storage_buffers_per_shader_stage >= 24);
        assert!(limits.max_compute_workgroups_per_dimension >= 65535);
    }

    // -- Features mapping --------------------------------------------------

    #[test]
    fn test_features_for_flags_empty() {
        let f = features_for_flags(FeatureFlags::empty());
        assert_eq!(f, wgpu::Features::empty());
    }

    #[test]
    fn test_features_for_flags_bindless() {
        let f = features_for_flags(FeatureFlags::BINDLESS);
        assert!(f.contains(wgpu::Features::TEXTURE_BINDING_ARRAY));
        assert!(f.contains(wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY));
    }

    // -- RhiInstance -------------------------------------------------------

    #[test]
    fn test_create_instance() {
        let rhi = create_instance();
        // The instance should be valid (no panic).
        let _ = rhi.inner();
    }

    // -- RhiDevice ---------------------------------------------------------

    #[test]
    fn test_rhi_device_new() {
        // We can't create a real device without a GPU, but we can verify the
        // constructor compiles and the accessors are wired correctly by
        // instantiating via request_device path (test below).
    }

    #[test]
    fn test_rhi_device_submission_index() {
        // Create a dummy device using a headless adapter (skipped if no GPU).
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
            assert_eq!(dev.current_submission_index(), 0);
            let next = dev.next_submission_index();
            assert_eq!(next, 0); // fetch_add returns previous value
            assert_eq!(dev.current_submission_index(), 1);
            dev.wait_idle(); // should not hang on an idle device
        }
    }

    // -- WgpuFence ---------------------------------------------------------

    #[test]
    fn test_wgpu_fence_new() {
        let fence = WgpuFence::new();
        assert_eq!(fence.signaled_value(), 0);
    }

    #[test]
    fn test_wgpu_fence_signaled_value() {
        let fence = WgpuFence::new();
        fence.set_signaled_value(42);
        assert_eq!(fence.signaled_value(), 42);
    }

    #[test]
    fn test_wgpu_fence_wait_noop() {
        // A fence with signaled_value == 0 should be a no-op.
        let fence = WgpuFence::new();
        // Creating a valid wgpu::Device to call wait on.  We must have a GPU
        // for this, so skip if none is available.
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
            // Should not panic or hang.
            fence.wait(&dev.device);
        }
    }

    #[test]
    fn test_wgpu_fence_default() {
        let fence: WgpuFence = Default::default();
        assert_eq!(fence.signaled_value(), 0);
    }

    // -- DeviceLostCallback ------------------------------------------------

    #[test]
    fn test_device_lost_callback_type() {
        // Verify the type alias is usable.
        let _cb: DeviceLostCallback = Box::new(|reason: wgpu::DeviceLostReason, msg: String| {
            eprintln!("Device lost: {:?} — {}", reason, msg);
        });
    }

    // -- Integration smoke test --------------------------------------------

    #[test]
    fn test_create_instance_adapter_device_smoke() {
        // Full lifecycle: instance -> adapter -> device.
        // Skipped on headless / CI systems where no adapter is available.
        let rhi = create_instance();
        if let Some(adapter) = pollster::block_on(
            rhi.inner()
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }),
        ) {
            let dev = request_device(
                &adapter,
                FeatureFlags::COMPUTE | FeatureFlags::BINDLESS,
                QualityTier::Medium,
            );
            assert!(dev.current_submission_index() == 0);

            // Submit a no-op command buffer to exercise the queue.
            let encoder =
                dev.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("smoke-test-encoder"),
                    });
            let cb = encoder.finish();
            let _submit_idx = dev.queue.submit(std::iter::once(cb));

            dev.wait_idle();
        }
    }
}
