//! Device manager with lost callback and recovery logic for TRINITY.
//!
//! This module provides the [`DeviceManager`] struct, which wraps [`TrinityDevice`]
//! with device lost handling, recovery logic, and resource tracking for rebuild.
//!
//! # Overview
//!
//! GPU devices can be lost due to:
//! - Driver crashes
//! - Timeout Detection and Recovery (TDR) events
//! - Power events (sleep/resume)
//! - GPU reset by external applications
//!
//! The `DeviceManager` provides:
//! - Lost callback invocation when device loss is detected
//! - Automatic recovery with exponential backoff
//! - Resource tracking for rebuild after recovery
//! - Maximum retry limit with fatal error handling
//!
//! # Architecture
//!
//! ```text
//! DeviceManager
//!   |
//!   +-- TrinityDevice (optional - None when lost)
//!   |
//!   +-- adapter: Arc<wgpu::Adapter>  (retained for recovery)
//!   |
//!   +-- DeviceState (thread-safe state machine)
//!   |     |
//!   |     +-- Healthy
//!   |     +-- Lost
//!   |     +-- Recovering { retry_count }
//!   |     +-- Fatal
//!   |
//!   +-- RecoveryConfig (backoff parameters)
//!   |
//!   +-- ResourceTracker (what needs rebuilding)
//!   |
//!   +-- lost_callback: Option<Box<dyn Fn(DeviceLostReason)>>
//! ```
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::device::{
//!     DeviceManager, DeviceRequirements, LimitRequirements, TrinityInstance,
//!     AdapterSelector, RecoveryConfig,
//! };
//! use std::sync::Arc;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create instance and select adapter
//! let instance = TrinityInstance::new();
//! let adapters: Vec<_> = instance.inner().enumerate_adapters(instance.backends());
//! let selector = AdapterSelector::new();
//! let result = selector.select(&adapters).expect("No suitable adapter");
//!
//! // Build requirements
//! let requirements = DeviceRequirements::standard();
//! let limits = LimitRequirements::standard();
//!
//! // Create manager with recovery config
//! let config = RecoveryConfig::default();
//! let mut manager = DeviceManager::new(
//!     Arc::new(result.adapter.clone()),
//!     requirements,
//!     limits,
//!     config,
//! ).await?;
//!
//! // Set lost callback
//! manager.set_lost_callback(|reason| {
//!     log::error!("Device lost: {:?}", reason);
//! });
//!
//! // Use the device
//! let device = manager.device().expect("Device should be healthy");
//! // ... render with device ...
//!
//! // Check state
//! if manager.is_healthy() {
//!     println!("Device is healthy");
//! }
//! # Ok(())
//! # }
//! ```

use log::{debug, error, info, warn};
use std::fmt;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use super::{
    negotiate_and_create_device, DeviceRequirements, LimitRequirements, NegotiateAndCreateError,
    TrinityDevice,
};

// ============================================================================
// Device State
// ============================================================================

/// The current state of the managed device.
///
/// This enum represents the device's lifecycle state and is used by
/// `DeviceManager` to track device health and recovery progress.
///
/// # State Transitions
///
/// ```text
/// Healthy ──[device lost]──> Lost
///    ^                         |
///    |                         v
///    |                    Recovering(0)
///    |                         |
///    |        [success]        v
///    +<──────────────── Recovering(n)
///                              |
///                         [max retries]
///                              v
///                            Fatal
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceState {
    /// Device is operational and ready for use.
    Healthy,

    /// Device has been lost and needs recovery.
    Lost,

    /// Device is being recovered, with the current retry attempt count.
    Recovering(u32),

    /// Recovery has failed after maximum retries. Device cannot be used.
    Fatal,
}

impl DeviceState {
    /// Check if the device is usable (Healthy state).
    #[inline]
    pub fn is_healthy(&self) -> bool {
        matches!(self, DeviceState::Healthy)
    }

    /// Check if recovery is needed or in progress.
    #[inline]
    pub fn needs_recovery(&self) -> bool {
        matches!(self, DeviceState::Lost | DeviceState::Recovering(_))
    }

    /// Check if the device is in a fatal, unrecoverable state.
    #[inline]
    pub fn is_fatal(&self) -> bool {
        matches!(self, DeviceState::Fatal)
    }
}

impl fmt::Display for DeviceState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceState::Healthy => write!(f, "Healthy"),
            DeviceState::Lost => write!(f, "Lost"),
            DeviceState::Recovering(n) => write!(f, "Recovering (attempt {})", n),
            DeviceState::Fatal => write!(f, "Fatal"),
        }
    }
}

impl Default for DeviceState {
    fn default() -> Self {
        DeviceState::Healthy
    }
}

// ============================================================================
// Thread-Safe State Holder
// ============================================================================

/// Thread-safe device state holder using atomics.
///
/// This struct provides lock-free state tracking for device health. The state
/// is encoded as a u32 where:
/// - 0 = Healthy
/// - 1 = Lost
/// - 2..=MAX = Recovering(n-2)
/// - u32::MAX = Fatal
#[derive(Debug)]
struct AtomicDeviceState {
    /// Encoded state value.
    state: AtomicU32,
}

impl AtomicDeviceState {
    /// State encoding constants.
    const HEALTHY: u32 = 0;
    const LOST: u32 = 1;
    const RECOVERING_BASE: u32 = 2;
    const FATAL: u32 = u32::MAX;

    /// Create a new atomic state holder initialized to Healthy.
    fn new() -> Self {
        Self {
            state: AtomicU32::new(Self::HEALTHY),
        }
    }

    /// Load the current state.
    fn load(&self) -> DeviceState {
        let val = self.state.load(Ordering::SeqCst);
        match val {
            Self::HEALTHY => DeviceState::Healthy,
            Self::LOST => DeviceState::Lost,
            Self::FATAL => DeviceState::Fatal,
            n => DeviceState::Recovering(n - Self::RECOVERING_BASE),
        }
    }

    /// Store a new state.
    fn store(&self, state: DeviceState) {
        let val = match state {
            DeviceState::Healthy => Self::HEALTHY,
            DeviceState::Lost => Self::LOST,
            DeviceState::Fatal => Self::FATAL,
            DeviceState::Recovering(n) => Self::RECOVERING_BASE.saturating_add(n),
        };
        self.state.store(val, Ordering::SeqCst);
    }

    /// Compare and swap state, returning whether the swap succeeded.
    fn compare_exchange(&self, current: DeviceState, new: DeviceState) -> bool {
        let current_val = match current {
            DeviceState::Healthy => Self::HEALTHY,
            DeviceState::Lost => Self::LOST,
            DeviceState::Fatal => Self::FATAL,
            DeviceState::Recovering(n) => Self::RECOVERING_BASE.saturating_add(n),
        };
        let new_val = match new {
            DeviceState::Healthy => Self::HEALTHY,
            DeviceState::Lost => Self::LOST,
            DeviceState::Fatal => Self::FATAL,
            DeviceState::Recovering(n) => Self::RECOVERING_BASE.saturating_add(n),
        };
        self.state
            .compare_exchange(current_val, new_val, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    }
}

impl Default for AtomicDeviceState {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Recovery Configuration
// ============================================================================

/// Configuration for device recovery behavior.
///
/// This struct controls how the `DeviceManager` attempts to recover from
/// device loss, including retry limits and backoff timing.
///
/// # Exponential Backoff
///
/// Recovery attempts use exponential backoff:
/// - Attempt 1: `initial_backoff_ms`
/// - Attempt 2: `initial_backoff_ms * 2`
/// - Attempt 3: `initial_backoff_ms * 4`
/// - ... capped at `max_backoff_ms`
///
/// # Example
///
/// ```
/// use renderer_backend::device::RecoveryConfig;
///
/// // Custom recovery configuration
/// let config = RecoveryConfig {
///     max_retries: 5,
///     initial_backoff_ms: 100,
///     max_backoff_ms: 10000,
/// };
///
/// // Or use defaults
/// let default_config = RecoveryConfig::default();
/// assert_eq!(default_config.max_retries, 3);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecoveryConfig {
    /// Maximum number of recovery attempts before giving up.
    ///
    /// After this many failed attempts, the device enters the Fatal state.
    pub max_retries: u32,

    /// Initial backoff delay in milliseconds before the first retry.
    ///
    /// This delay doubles with each subsequent attempt (exponential backoff).
    pub initial_backoff_ms: u64,

    /// Maximum backoff delay in milliseconds.
    ///
    /// The backoff will not exceed this value regardless of retry count.
    pub max_backoff_ms: u64,
}

impl RecoveryConfig {
    /// Create a new recovery configuration.
    ///
    /// # Arguments
    ///
    /// * `max_retries` - Maximum recovery attempts
    /// * `initial_backoff_ms` - Initial delay before first retry
    /// * `max_backoff_ms` - Maximum delay cap
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::device::RecoveryConfig;
    ///
    /// let config = RecoveryConfig::new(5, 200, 30000);
    /// ```
    #[must_use]
    pub const fn new(max_retries: u32, initial_backoff_ms: u64, max_backoff_ms: u64) -> Self {
        Self {
            max_retries,
            initial_backoff_ms,
            max_backoff_ms,
        }
    }

    /// Create a configuration for aggressive recovery (more retries, shorter delays).
    ///
    /// This is suitable for interactive applications that need quick recovery.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::device::RecoveryConfig;
    ///
    /// let config = RecoveryConfig::aggressive();
    /// assert!(config.initial_backoff_ms < 100);
    /// ```
    #[must_use]
    pub const fn aggressive() -> Self {
        Self {
            max_retries: 5,
            initial_backoff_ms: 50,
            max_backoff_ms: 2000,
        }
    }

    /// Create a configuration for conservative recovery (fewer retries, longer delays).
    ///
    /// This is suitable for background applications or when device issues
    /// might be hardware-related.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::device::RecoveryConfig;
    ///
    /// let config = RecoveryConfig::conservative();
    /// assert!(config.initial_backoff_ms > 500);
    /// ```
    #[must_use]
    pub const fn conservative() -> Self {
        Self {
            max_retries: 2,
            initial_backoff_ms: 1000,
            max_backoff_ms: 30000,
        }
    }

    /// Calculate the backoff duration for a given retry attempt.
    ///
    /// Uses exponential backoff: `initial * 2^attempt`, capped at max.
    ///
    /// # Arguments
    ///
    /// * `attempt` - The current retry attempt number (0-indexed)
    ///
    /// # Returns
    ///
    /// The duration to wait before the next retry attempt.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::device::RecoveryConfig;
    /// use std::time::Duration;
    ///
    /// let config = RecoveryConfig::new(5, 100, 5000);
    ///
    /// assert_eq!(config.backoff_for_attempt(0), Duration::from_millis(100));
    /// assert_eq!(config.backoff_for_attempt(1), Duration::from_millis(200));
    /// assert_eq!(config.backoff_for_attempt(2), Duration::from_millis(400));
    /// assert_eq!(config.backoff_for_attempt(10), Duration::from_millis(5000)); // capped
    /// ```
    #[must_use]
    pub fn backoff_for_attempt(&self, attempt: u32) -> Duration {
        let backoff = self
            .initial_backoff_ms
            .saturating_mul(1u64 << attempt.min(31));
        Duration::from_millis(backoff.min(self.max_backoff_ms))
    }
}

impl Default for RecoveryConfig {
    /// Default configuration: 3 retries, 200ms initial backoff, 5s max.
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff_ms: 200,
            max_backoff_ms: 5000,
        }
    }
}

impl fmt::Display for RecoveryConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RecoveryConfig(max_retries={}, backoff={}ms..{}ms)",
            self.max_retries, self.initial_backoff_ms, self.max_backoff_ms
        )
    }
}

// ============================================================================
// Device Lost Reason
// ============================================================================

/// Reason why the device was lost.
///
/// This enum provides context about why the GPU device became unavailable,
/// which can help determine the appropriate recovery strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceLostReason {
    /// Device was explicitly destroyed (intentional).
    Destroyed,

    /// Driver crash or internal error.
    DriverError,

    /// Timeout Detection and Recovery (TDR) event.
    ///
    /// This occurs when the GPU takes too long to respond, typically due to
    /// an infinite loop in a shader or excessive workload.
    Timeout,

    /// Power state change (sleep/resume, GPU power management).
    PowerEvent,

    /// Device was reset by an external application.
    ExternalReset,

    /// Unknown or unspecified reason.
    Unknown,
}

impl DeviceLostReason {
    /// Check if recovery is likely to succeed for this reason.
    ///
    /// Some device loss reasons are more recoverable than others. For example,
    /// a power event is usually recoverable, while a hardware failure might not be.
    #[must_use]
    pub fn is_likely_recoverable(&self) -> bool {
        match self {
            DeviceLostReason::PowerEvent => true,
            DeviceLostReason::Timeout => true,
            DeviceLostReason::ExternalReset => true,
            DeviceLostReason::DriverError => true, // worth trying
            DeviceLostReason::Destroyed => false,  // intentional
            DeviceLostReason::Unknown => true,     // worth trying
        }
    }
}

impl fmt::Display for DeviceLostReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceLostReason::Destroyed => write!(f, "Device destroyed"),
            DeviceLostReason::DriverError => write!(f, "Driver error"),
            DeviceLostReason::Timeout => write!(f, "GPU timeout (TDR)"),
            DeviceLostReason::PowerEvent => write!(f, "Power state change"),
            DeviceLostReason::ExternalReset => write!(f, "External reset"),
            DeviceLostReason::Unknown => write!(f, "Unknown"),
        }
    }
}

// ============================================================================
// Resource Tracker
// ============================================================================

/// Tracker for resources that need rebuilding after device recovery.
///
/// When a device is lost, all GPU resources (buffers, textures, pipelines)
/// become invalid. This tracker maintains counts and can be used by the
/// application to know what needs to be recreated.
///
/// # Example
///
/// ```
/// use renderer_backend::device::ResourceTracker;
///
/// let mut tracker = ResourceTracker::new();
///
/// // Track resources as they're created
/// tracker.track_buffer();
/// tracker.track_texture();
/// tracker.track_pipeline();
///
/// println!("Need to rebuild: {}", tracker);
///
/// // After recovery
/// tracker.clear();
/// ```
#[derive(Debug, Default)]
pub struct ResourceTracker {
    /// Number of buffers tracked.
    buffer_count: AtomicU64,
    /// Number of textures tracked.
    texture_count: AtomicU64,
    /// Number of bind groups tracked.
    bind_group_count: AtomicU64,
    /// Number of pipelines (render + compute) tracked.
    pipeline_count: AtomicU64,
    /// Number of samplers tracked.
    sampler_count: AtomicU64,
    /// Number of query sets tracked.
    query_set_count: AtomicU64,
}

impl ResourceTracker {
    /// Create a new empty resource tracker.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Track a buffer being created.
    pub fn track_buffer(&self) {
        self.buffer_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Untrack a buffer being destroyed.
    pub fn untrack_buffer(&self) {
        self.buffer_count.fetch_sub(1, Ordering::Relaxed);
    }

    /// Track a texture being created.
    pub fn track_texture(&self) {
        self.texture_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Untrack a texture being destroyed.
    pub fn untrack_texture(&self) {
        self.texture_count.fetch_sub(1, Ordering::Relaxed);
    }

    /// Track a bind group being created.
    pub fn track_bind_group(&self) {
        self.bind_group_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Untrack a bind group being destroyed.
    pub fn untrack_bind_group(&self) {
        self.bind_group_count.fetch_sub(1, Ordering::Relaxed);
    }

    /// Track a pipeline being created.
    pub fn track_pipeline(&self) {
        self.pipeline_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Untrack a pipeline being destroyed.
    pub fn untrack_pipeline(&self) {
        self.pipeline_count.fetch_sub(1, Ordering::Relaxed);
    }

    /// Track a sampler being created.
    pub fn track_sampler(&self) {
        self.sampler_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Untrack a sampler being destroyed.
    pub fn untrack_sampler(&self) {
        self.sampler_count.fetch_sub(1, Ordering::Relaxed);
    }

    /// Track a query set being created.
    pub fn track_query_set(&self) {
        self.query_set_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Untrack a query set being destroyed.
    pub fn untrack_query_set(&self) {
        self.query_set_count.fetch_sub(1, Ordering::Relaxed);
    }

    /// Get the current buffer count.
    #[inline]
    pub fn buffer_count(&self) -> u64 {
        self.buffer_count.load(Ordering::Relaxed)
    }

    /// Get the current texture count.
    #[inline]
    pub fn texture_count(&self) -> u64 {
        self.texture_count.load(Ordering::Relaxed)
    }

    /// Get the current bind group count.
    #[inline]
    pub fn bind_group_count(&self) -> u64 {
        self.bind_group_count.load(Ordering::Relaxed)
    }

    /// Get the current pipeline count.
    #[inline]
    pub fn pipeline_count(&self) -> u64 {
        self.pipeline_count.load(Ordering::Relaxed)
    }

    /// Get the current sampler count.
    #[inline]
    pub fn sampler_count(&self) -> u64 {
        self.sampler_count.load(Ordering::Relaxed)
    }

    /// Get the current query set count.
    #[inline]
    pub fn query_set_count(&self) -> u64 {
        self.query_set_count.load(Ordering::Relaxed)
    }

    /// Get the total number of tracked resources.
    #[must_use]
    pub fn total_count(&self) -> u64 {
        self.buffer_count()
            + self.texture_count()
            + self.bind_group_count()
            + self.pipeline_count()
            + self.sampler_count()
            + self.query_set_count()
    }

    /// Check if there are any tracked resources.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.total_count() == 0
    }

    /// Clear all tracked resources.
    ///
    /// Call this after successfully rebuilding all resources.
    pub fn clear(&self) {
        self.buffer_count.store(0, Ordering::Relaxed);
        self.texture_count.store(0, Ordering::Relaxed);
        self.bind_group_count.store(0, Ordering::Relaxed);
        self.pipeline_count.store(0, Ordering::Relaxed);
        self.sampler_count.store(0, Ordering::Relaxed);
        self.query_set_count.store(0, Ordering::Relaxed);
    }
}

impl fmt::Display for ResourceTracker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ResourceTracker(buffers={}, textures={}, bind_groups={}, pipelines={}, samplers={}, query_sets={})",
            self.buffer_count(),
            self.texture_count(),
            self.bind_group_count(),
            self.pipeline_count(),
            self.sampler_count(),
            self.query_set_count()
        )
    }
}

// ============================================================================
// Device Manager Error
// ============================================================================

/// Errors that can occur during device management.
#[derive(Debug)]
pub enum DeviceManagerError {
    /// Failed to create the initial device.
    InitialCreation(NegotiateAndCreateError),

    /// Device is in a lost or recovering state.
    DeviceUnavailable(DeviceState),

    /// Recovery failed after maximum retries.
    RecoveryFailed {
        /// Number of recovery attempts made.
        attempts: u32,
        /// Last error encountered.
        last_error: NegotiateAndCreateError,
    },

    /// Device is in fatal state and cannot be recovered.
    FatalState,
}

impl fmt::Display for DeviceManagerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceManagerError::InitialCreation(e) => {
                write!(f, "Failed to create initial device: {}", e)
            }
            DeviceManagerError::DeviceUnavailable(state) => {
                write!(f, "Device unavailable (state: {})", state)
            }
            DeviceManagerError::RecoveryFailed {
                attempts,
                last_error,
            } => {
                write!(
                    f,
                    "Recovery failed after {} attempts: {}",
                    attempts, last_error
                )
            }
            DeviceManagerError::FatalState => {
                write!(f, "Device in fatal state, cannot recover")
            }
        }
    }
}

impl std::error::Error for DeviceManagerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DeviceManagerError::InitialCreation(e) => Some(e),
            DeviceManagerError::RecoveryFailed { last_error, .. } => Some(last_error),
            _ => None,
        }
    }
}

impl From<NegotiateAndCreateError> for DeviceManagerError {
    fn from(err: NegotiateAndCreateError) -> Self {
        DeviceManagerError::InitialCreation(err)
    }
}

// ============================================================================
// Device Manager
// ============================================================================

/// Managed device with lost callback and recovery logic.
///
/// `DeviceManager` wraps a [`TrinityDevice`] and provides:
///
/// - **Lost detection**: Monitors for device loss via wgpu's error callback
/// - **Lost callback**: Invokes a user-provided callback when device is lost
/// - **Automatic recovery**: Attempts to recreate the device with exponential backoff
/// - **Resource tracking**: Tracks what GPU resources need rebuilding
/// - **State management**: Thread-safe state tracking (Healthy/Lost/Recovering/Fatal)
///
/// # Thread Safety
///
/// `DeviceManager` is `Send + Sync`. The internal device state uses atomics for
/// lock-free access. However, recovery operations are not automatically thread-safe
/// and should be coordinated by the caller.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::device::{
///     DeviceManager, DeviceRequirements, LimitRequirements, TrinityInstance,
///     AdapterSelector, RecoveryConfig, DeviceLostReason,
/// };
/// use std::sync::Arc;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// // Setup
/// let instance = TrinityInstance::new();
/// let adapters: Vec<_> = instance.inner().enumerate_adapters(instance.backends());
/// let selector = AdapterSelector::new();
/// let result = selector.select(&adapters).expect("No adapter");
///
/// // Create manager
/// let mut manager = DeviceManager::new(
///     Arc::new(result.adapter.clone()),
///     DeviceRequirements::standard(),
///     LimitRequirements::standard(),
///     RecoveryConfig::default(),
/// ).await?;
///
/// // Set callback for device loss
/// manager.set_lost_callback(|reason| {
///     eprintln!("GPU device lost: {}", reason);
///     // Trigger resource rebuild in your application
/// });
///
/// // Use the device
/// if let Some(device) = manager.device() {
///     // Create buffers, textures, etc.
///     manager.resource_tracker().track_buffer();
/// }
///
/// // Simulate device loss (normally triggered by wgpu error callback)
/// manager.mark_lost(DeviceLostReason::Timeout);
///
/// // Attempt recovery
/// match manager.try_recover().await {
///     Ok(()) => println!("Recovery successful!"),
///     Err(e) => eprintln!("Recovery failed: {}", e),
/// }
/// # Ok(())
/// # }
/// ```
pub struct DeviceManager {
    /// The current device (None when lost or recovering).
    device: Option<TrinityDevice>,

    /// The adapter used for device creation (retained for recovery).
    adapter: Arc<wgpu::Adapter>,

    /// Feature requirements for device creation.
    requirements: DeviceRequirements,

    /// Limit requirements for device creation.
    /// Note: Currently stored for potential future use in device recreation with
    /// different limits. The actual limits are negotiated during device creation.
    #[allow(dead_code)]
    limits: LimitRequirements,

    /// Thread-safe device state.
    state: Arc<AtomicDeviceState>,

    /// Recovery configuration.
    config: RecoveryConfig,

    /// Resource tracker for rebuild.
    resource_tracker: Arc<ResourceTracker>,

    /// Callback invoked when device is lost.
    lost_callback: Option<Box<dyn Fn(DeviceLostReason) + Send + Sync>>,

    /// Total number of recovery attempts made (across all loss events).
    total_recovery_attempts: u64,

    /// Number of successful recoveries.
    successful_recoveries: u64,
}

impl DeviceManager {
    /// Create a new device manager.
    ///
    /// This creates the initial device using the provided adapter and requirements.
    /// If device creation fails, an error is returned immediately.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The adapter to use for device creation (wrapped in Arc for sharing)
    /// * `requirements` - Feature requirements for the device
    /// * `limits` - Limit requirements for the device
    /// * `config` - Recovery configuration
    ///
    /// # Returns
    ///
    /// A `Result` containing the new `DeviceManager` or a `DeviceManagerError` if
    /// initial device creation failed.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::device::{
    ///     DeviceManager, DeviceRequirements, LimitRequirements, RecoveryConfig,
    /// };
    /// use std::sync::Arc;
    ///
    /// # async fn example(adapter: wgpu::Adapter) -> Result<(), Box<dyn std::error::Error>> {
    /// let manager = DeviceManager::new(
    ///     Arc::new(adapter),
    ///     DeviceRequirements::standard(),
    ///     LimitRequirements::standard(),
    ///     RecoveryConfig::default(),
    /// ).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn new(
        adapter: Arc<wgpu::Adapter>,
        requirements: DeviceRequirements,
        limits: LimitRequirements,
        config: RecoveryConfig,
    ) -> Result<Self, DeviceManagerError> {
        let adapter_info = adapter.get_info();
        info!(
            "DeviceManager: Creating managed device from adapter: {} ({:?})",
            adapter_info.name, adapter_info.backend
        );
        info!("DeviceManager: Recovery config: {}", config);

        // Create the initial device
        let (device, negotiation) = negotiate_and_create_device(&requirements, &adapter).await?;

        info!(
            "DeviceManager: Initial device created with {} features ({} degraded)",
            negotiation.enabled_count(),
            negotiation.degraded_count()
        );

        let state = Arc::new(AtomicDeviceState::new());
        let resource_tracker = Arc::new(ResourceTracker::new());

        Ok(Self {
            device: Some(device),
            adapter,
            requirements,
            limits,
            state,
            config,
            resource_tracker,
            lost_callback: None,
            total_recovery_attempts: 0,
            successful_recoveries: 0,
        })
    }

    /// Set the callback to invoke when the device is lost.
    ///
    /// The callback receives the [`DeviceLostReason`] indicating why the device
    /// was lost. Use this to trigger resource rebuilding in your application.
    ///
    /// # Arguments
    ///
    /// * `callback` - A function to call when device loss is detected
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use renderer_backend::device::{DeviceManager, DeviceLostReason};
    /// # fn example(manager: &mut DeviceManager) {
    /// manager.set_lost_callback(|reason| {
    ///     log::error!("Device lost: {}", reason);
    ///     // Signal application to stop rendering and wait for recovery
    /// });
    /// # }
    /// ```
    pub fn set_lost_callback<F>(&mut self, callback: F)
    where
        F: Fn(DeviceLostReason) + Send + Sync + 'static,
    {
        self.lost_callback = Some(Box::new(callback));
    }

    /// Clear the lost callback.
    pub fn clear_lost_callback(&mut self) {
        self.lost_callback = None;
    }

    /// Get a reference to the current device, if healthy.
    ///
    /// Returns `None` if the device is lost, recovering, or in fatal state.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use renderer_backend::device::DeviceManager;
    /// # fn example(manager: &DeviceManager) {
    /// if let Some(device) = manager.device() {
    ///     let encoder = device.create_command_encoder(Some("Frame"));
    ///     // ... record commands ...
    /// } else {
    ///     println!("Device unavailable, state: {}", manager.state());
    /// }
    /// # }
    /// ```
    #[inline]
    pub fn device(&self) -> Option<&TrinityDevice> {
        if self.state.load().is_healthy() {
            self.device.as_ref()
        } else {
            None
        }
    }

    /// Get the current device state.
    #[inline]
    pub fn state(&self) -> DeviceState {
        self.state.load()
    }

    /// Check if the device is healthy and ready for use.
    #[inline]
    pub fn is_healthy(&self) -> bool {
        self.state.load().is_healthy()
    }

    /// Check if the device needs recovery (lost or recovering).
    #[inline]
    pub fn needs_recovery(&self) -> bool {
        self.state.load().needs_recovery()
    }

    /// Check if the device is in fatal state.
    #[inline]
    pub fn is_fatal(&self) -> bool {
        self.state.load().is_fatal()
    }

    /// Get the recovery configuration.
    #[inline]
    pub fn config(&self) -> &RecoveryConfig {
        &self.config
    }

    /// Get the resource tracker.
    ///
    /// Use this to track resources that need rebuilding after recovery.
    #[inline]
    pub fn resource_tracker(&self) -> &Arc<ResourceTracker> {
        &self.resource_tracker
    }

    /// Get the adapter being used.
    #[inline]
    pub fn adapter(&self) -> &Arc<wgpu::Adapter> {
        &self.adapter
    }

    /// Get recovery statistics.
    ///
    /// Returns a tuple of (total_recovery_attempts, successful_recoveries).
    #[inline]
    pub fn recovery_stats(&self) -> (u64, u64) {
        (self.total_recovery_attempts, self.successful_recoveries)
    }

    /// Mark the device as lost.
    ///
    /// This should be called when device loss is detected (e.g., from wgpu's
    /// error callback). It will:
    ///
    /// 1. Transition state to Lost
    /// 2. Clear the current device
    /// 3. Invoke the lost callback (if set)
    ///
    /// # Arguments
    ///
    /// * `reason` - The reason why the device was lost
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use renderer_backend::device::{DeviceManager, DeviceLostReason};
    /// # fn example(manager: &mut DeviceManager) {
    /// // Called from wgpu error callback
    /// manager.mark_lost(DeviceLostReason::Timeout);
    /// # }
    /// ```
    pub fn mark_lost(&mut self, reason: DeviceLostReason) {
        let current_state = self.state.load();

        // Only transition from Healthy to Lost
        if !current_state.is_healthy() {
            warn!(
                "DeviceManager: mark_lost called but state is already {:?}",
                current_state
            );
            return;
        }

        error!("DeviceManager: Device lost - {}", reason);
        self.state.store(DeviceState::Lost);

        // Clear the device
        self.device = None;

        // Invoke callback
        if let Some(ref callback) = self.lost_callback {
            debug!("DeviceManager: Invoking lost callback");
            callback(reason);
        }

        info!(
            "DeviceManager: {} resources need rebuilding",
            self.resource_tracker.total_count()
        );
    }

    /// Attempt to recover the device.
    ///
    /// This method attempts to recreate the device using the original adapter
    /// and requirements. It uses exponential backoff between retries.
    ///
    /// # Returns
    ///
    /// - `Ok(())` if recovery was successful
    /// - `Err(DeviceManagerError::DeviceUnavailable)` if already healthy
    /// - `Err(DeviceManagerError::FatalState)` if in fatal state
    /// - `Err(DeviceManagerError::RecoveryFailed)` if max retries exceeded
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use renderer_backend::device::DeviceManager;
    /// # async fn example(manager: &mut DeviceManager) {
    /// match manager.try_recover().await {
    ///     Ok(()) => {
    ///         println!("Recovery successful!");
    ///         // Rebuild all resources
    ///     }
    ///     Err(e) => {
    ///         eprintln!("Recovery failed: {}", e);
    ///         // Handle fatal error
    ///     }
    /// }
    /// # }
    /// ```
    pub async fn try_recover(&mut self) -> Result<(), DeviceManagerError> {
        let current_state = self.state.load();

        // Check preconditions
        match current_state {
            DeviceState::Healthy => {
                debug!("DeviceManager: try_recover called but device is healthy");
                return Err(DeviceManagerError::DeviceUnavailable(current_state));
            }
            DeviceState::Fatal => {
                warn!("DeviceManager: try_recover called but device is in fatal state");
                return Err(DeviceManagerError::FatalState);
            }
            DeviceState::Lost | DeviceState::Recovering(_) => {
                // Proceed with recovery
            }
        }

        // Handle edge case: if max_retries is 0, go directly to fatal state
        if self.config.max_retries == 0 {
            warn!("DeviceManager: max_retries is 0, entering fatal state immediately");
            self.state.store(DeviceState::Fatal);
            return Err(DeviceManagerError::FatalState);
        }

        info!("DeviceManager: Starting recovery (max {} attempts)", self.config.max_retries);

        let mut last_error: Option<NegotiateAndCreateError> = None;

        for attempt in 0..self.config.max_retries {
            self.state.store(DeviceState::Recovering(attempt));
            self.total_recovery_attempts += 1;

            // Calculate backoff
            let backoff = self.config.backoff_for_attempt(attempt);
            if attempt > 0 {
                info!(
                    "DeviceManager: Recovery attempt {} of {}, waiting {:?}",
                    attempt + 1,
                    self.config.max_retries,
                    backoff
                );

                // Sleep before retry using std::thread::sleep
                // Note: This blocks the current thread. For async contexts, callers
                // should wrap this in a spawn_blocking or similar construct.
                std::thread::sleep(backoff);
            } else {
                info!(
                    "DeviceManager: Recovery attempt {} of {}",
                    attempt + 1,
                    self.config.max_retries
                );
            }

            // Attempt device creation
            match negotiate_and_create_device(&self.requirements, &self.adapter).await {
                Ok((device, negotiation)) => {
                    info!(
                        "DeviceManager: Recovery successful on attempt {} with {} features",
                        attempt + 1,
                        negotiation.enabled_count()
                    );

                    self.device = Some(device);
                    self.state.store(DeviceState::Healthy);
                    self.successful_recoveries += 1;

                    return Ok(());
                }
                Err(e) => {
                    warn!(
                        "DeviceManager: Recovery attempt {} failed: {}",
                        attempt + 1,
                        e
                    );
                    last_error = Some(e);
                }
            }
        }

        // All retries exhausted
        error!(
            "DeviceManager: Recovery failed after {} attempts, entering fatal state",
            self.config.max_retries
        );
        self.state.store(DeviceState::Fatal);

        Err(DeviceManagerError::RecoveryFailed {
            attempts: self.config.max_retries,
            last_error: last_error.expect("should have at least one error"),
        })
    }

    /// Force a recovery attempt with a specific retry count.
    ///
    /// This is useful for testing or when you want to override the normal
    /// recovery flow.
    ///
    /// # Arguments
    ///
    /// * `retry_count` - The retry attempt number (0-indexed)
    ///
    /// # Returns
    ///
    /// Whether the recovery attempt succeeded.
    pub async fn force_recovery_attempt(&mut self, retry_count: u32) -> bool {
        self.state.store(DeviceState::Recovering(retry_count));
        self.total_recovery_attempts += 1;

        match negotiate_and_create_device(&self.requirements, &self.adapter).await {
            Ok((device, _)) => {
                self.device = Some(device);
                self.state.store(DeviceState::Healthy);
                self.successful_recoveries += 1;
                true
            }
            Err(e) => {
                warn!("DeviceManager: Force recovery attempt failed: {}", e);
                if retry_count >= self.config.max_retries.saturating_sub(1) {
                    self.state.store(DeviceState::Fatal);
                } else {
                    self.state.store(DeviceState::Lost);
                }
                false
            }
        }
    }

    /// Reset the device manager from fatal state.
    ///
    /// This allows attempting recovery again after the device has entered
    /// fatal state. Use with caution as repeated failures may indicate
    /// hardware or driver issues.
    ///
    /// # Returns
    ///
    /// `true` if the state was reset from Fatal to Lost, `false` otherwise.
    pub fn reset_from_fatal(&mut self) -> bool {
        if self.state.compare_exchange(DeviceState::Fatal, DeviceState::Lost) {
            warn!("DeviceManager: Reset from fatal state, recovery can be attempted again");
            true
        } else {
            false
        }
    }
}

impl fmt::Debug for DeviceManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceManager")
            .field("state", &self.state.load())
            .field("has_device", &self.device.is_some())
            .field("config", &self.config)
            .field("resource_tracker", &self.resource_tracker)
            .field("total_recovery_attempts", &self.total_recovery_attempts)
            .field("successful_recoveries", &self.successful_recoveries)
            .finish()
    }
}

impl fmt::Display for DeviceManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.state.load();
        write!(
            f,
            "DeviceManager(state={}, resources={}, recoveries={}/{})",
            state,
            self.resource_tracker.total_count(),
            self.successful_recoveries,
            self.total_recovery_attempts
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_state_helpers() {
        assert!(DeviceState::Healthy.is_healthy());
        assert!(!DeviceState::Lost.is_healthy());
        assert!(!DeviceState::Recovering(0).is_healthy());
        assert!(!DeviceState::Fatal.is_healthy());

        assert!(!DeviceState::Healthy.needs_recovery());
        assert!(DeviceState::Lost.needs_recovery());
        assert!(DeviceState::Recovering(1).needs_recovery());
        assert!(!DeviceState::Fatal.needs_recovery());

        assert!(!DeviceState::Healthy.is_fatal());
        assert!(!DeviceState::Lost.is_fatal());
        assert!(!DeviceState::Recovering(0).is_fatal());
        assert!(DeviceState::Fatal.is_fatal());
    }

    #[test]
    fn test_device_state_display() {
        assert_eq!(format!("{}", DeviceState::Healthy), "Healthy");
        assert_eq!(format!("{}", DeviceState::Lost), "Lost");
        assert_eq!(
            format!("{}", DeviceState::Recovering(3)),
            "Recovering (attempt 3)"
        );
        assert_eq!(format!("{}", DeviceState::Fatal), "Fatal");
    }

    #[test]
    fn test_atomic_device_state() {
        let state = AtomicDeviceState::new();
        assert_eq!(state.load(), DeviceState::Healthy);

        state.store(DeviceState::Lost);
        assert_eq!(state.load(), DeviceState::Lost);

        state.store(DeviceState::Recovering(5));
        assert_eq!(state.load(), DeviceState::Recovering(5));

        state.store(DeviceState::Fatal);
        assert_eq!(state.load(), DeviceState::Fatal);
    }

    #[test]
    fn test_atomic_device_state_compare_exchange() {
        let state = AtomicDeviceState::new();

        // Should succeed: Healthy -> Lost
        assert!(state.compare_exchange(DeviceState::Healthy, DeviceState::Lost));
        assert_eq!(state.load(), DeviceState::Lost);

        // Should fail: current is Lost, not Healthy
        assert!(!state.compare_exchange(DeviceState::Healthy, DeviceState::Fatal));
        assert_eq!(state.load(), DeviceState::Lost);

        // Should succeed: Lost -> Recovering(0)
        assert!(state.compare_exchange(DeviceState::Lost, DeviceState::Recovering(0)));
        assert_eq!(state.load(), DeviceState::Recovering(0));
    }

    #[test]
    fn test_recovery_config_default() {
        let config = RecoveryConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.initial_backoff_ms, 200);
        assert_eq!(config.max_backoff_ms, 5000);
    }

    #[test]
    fn test_recovery_config_presets() {
        let aggressive = RecoveryConfig::aggressive();
        assert!(aggressive.max_retries > 3);
        assert!(aggressive.initial_backoff_ms < 100);

        let conservative = RecoveryConfig::conservative();
        assert!(conservative.max_retries < 3);
        assert!(conservative.initial_backoff_ms > 500);
    }

    #[test]
    fn test_recovery_config_backoff() {
        let config = RecoveryConfig::new(5, 100, 5000);

        assert_eq!(config.backoff_for_attempt(0), Duration::from_millis(100));
        assert_eq!(config.backoff_for_attempt(1), Duration::from_millis(200));
        assert_eq!(config.backoff_for_attempt(2), Duration::from_millis(400));
        assert_eq!(config.backoff_for_attempt(3), Duration::from_millis(800));
        assert_eq!(config.backoff_for_attempt(4), Duration::from_millis(1600));
        assert_eq!(config.backoff_for_attempt(5), Duration::from_millis(3200));
        // Capped at max
        assert_eq!(config.backoff_for_attempt(6), Duration::from_millis(5000));
        assert_eq!(config.backoff_for_attempt(10), Duration::from_millis(5000));
    }

    #[test]
    fn test_recovery_config_backoff_overflow_protection() {
        let config = RecoveryConfig::new(100, 1000, u64::MAX);

        // Should not panic with high attempt numbers
        let _backoff = config.backoff_for_attempt(31);
        let _backoff = config.backoff_for_attempt(50);
        let _backoff = config.backoff_for_attempt(u32::MAX);
    }

    #[test]
    fn test_device_lost_reason_recoverable() {
        assert!(DeviceLostReason::PowerEvent.is_likely_recoverable());
        assert!(DeviceLostReason::Timeout.is_likely_recoverable());
        assert!(DeviceLostReason::ExternalReset.is_likely_recoverable());
        assert!(DeviceLostReason::DriverError.is_likely_recoverable());
        assert!(DeviceLostReason::Unknown.is_likely_recoverable());
        assert!(!DeviceLostReason::Destroyed.is_likely_recoverable());
    }

    #[test]
    fn test_device_lost_reason_display() {
        assert_eq!(
            format!("{}", DeviceLostReason::Timeout),
            "GPU timeout (TDR)"
        );
        assert_eq!(
            format!("{}", DeviceLostReason::PowerEvent),
            "Power state change"
        );
    }

    #[test]
    fn test_resource_tracker() {
        let tracker = ResourceTracker::new();
        assert!(tracker.is_empty());
        assert_eq!(tracker.total_count(), 0);

        tracker.track_buffer();
        tracker.track_buffer();
        tracker.track_texture();
        tracker.track_pipeline();

        assert!(!tracker.is_empty());
        assert_eq!(tracker.buffer_count(), 2);
        assert_eq!(tracker.texture_count(), 1);
        assert_eq!(tracker.pipeline_count(), 1);
        assert_eq!(tracker.total_count(), 4);

        tracker.untrack_buffer();
        assert_eq!(tracker.buffer_count(), 1);
        assert_eq!(tracker.total_count(), 3);

        tracker.clear();
        assert!(tracker.is_empty());
    }

    #[test]
    fn test_resource_tracker_display() {
        let tracker = ResourceTracker::new();
        tracker.track_buffer();
        tracker.track_texture();

        let display = format!("{}", tracker);
        assert!(display.contains("buffers=1"));
        assert!(display.contains("textures=1"));
    }

    #[test]
    fn test_device_manager_error_display() {
        let err = DeviceManagerError::DeviceUnavailable(DeviceState::Lost);
        assert!(format!("{}", err).contains("Lost"));

        let err = DeviceManagerError::FatalState;
        assert!(format!("{}", err).contains("fatal"));
    }

    #[test]
    fn test_recovery_config_display() {
        let config = RecoveryConfig::new(5, 100, 10000);
        let display = format!("{}", config);
        assert!(display.contains("max_retries=5"));
        assert!(display.contains("100ms"));
        assert!(display.contains("10000ms"));
    }

    #[test]
    fn test_recovery_config_zero_retries() {
        // Ensure RecoveryConfig with max_retries = 0 can be created without panic
        let config = RecoveryConfig::new(0, 100, 5000);
        assert_eq!(config.max_retries, 0);
        // Backoff calculation should still work
        assert_eq!(config.backoff_for_attempt(0), Duration::from_millis(100));
    }
}
