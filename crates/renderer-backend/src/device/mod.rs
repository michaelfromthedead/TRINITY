//! Device module - wgpu device initialization and management.
//!
//! This module provides the core device infrastructure for TRINITY:
//!
//! - [`TrinityInstance`] - Entry point to wgpu with multi-backend support
//! - [`enumerate_adapters_with_info`] - Enhanced adapter enumeration with logging
//! - [`filter_by_device_type`] / [`filter_by_backend`] - Adapter filtering utilities
//! - [`TrinityDevice`] - Device wrapper with creation helpers and metadata
//! - [`ErrorScope`] - RAII wrapper for fine-grained GPU error handling
//!
//! # Architecture
//!
//! The device module follows a layered architecture:
//!
//! ```text
//! TrinityInstance (instance.rs)
//!     └── Backend selection, instance configuration
//!
//! AdapterEnumerator (adapter.rs)
//!     └── Adapter enumeration, logging, filtering
//!
//! TrinityDevice (device.rs)
//!     └── Device creation, error handling, lost recovery
//!
//! ErrorScope (error_scope.rs)
//!     └── RAII error scopes, validation/OOM error capture
//!
//! TrinityQueue (queue.rs)
//!     └── Command submission, pending work tracking, completion callbacks
//!
//! CapabilityTier (capability.rs)
//!     └── Hardware tier detection, feature-based classification
//! ```
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::device::{TrinityInstance, enumerate_adapters_with_info};
//!
//! // Create instance with platform-appropriate backends
//! let instance = TrinityInstance::new();
//!
//! // Enumerate available adapters with detailed logging
//! let result = enumerate_adapters_with_info(instance.inner(), instance.backends());
//!
//! println!("Found {} adapter(s)", result.len());
//! println!("Backend breakdown: {}", result.backend_counts.summary());
//!
//! // Get the best adapter
//! if let Some(adapter) = result.best_adapter() {
//!     let info = adapter.get_info();
//!     println!("Best adapter: {} ({:?})", info.name, info.backend);
//! }
//! ```

mod adapter;
mod capability;
mod device;
mod error_scope;
mod instance;
mod limits;
mod manager;
mod queue;
mod requirements;

pub use adapter::{
    device_type_description,
    enumerate_adapters_with_info,
    filter_by_backend,
    filter_by_device_type,
    inspect_features,
    inspect_limits,
    AdapterBlacklistEntry,
    AdapterFeatures,
    AdapterLimits,
    AdapterProperties,
    AdapterScore,
    AdapterSelector,
    BackendCounts,
    BindGroupLimits,
    BufferLimits,
    ComputeLimits,
    DeviceTypeWeights,
    EnumerationResult,
    FeaturesSummary,
    FeatureTier,
    LimitsSummary,
    SelectionResult,
    TextureLimits,
    Vendor,
    VertexLimits,
};
pub use capability::{
    can_achieve_tier,
    detect_capability_tier,
    features_for_tier,
    CapabilityManager,
    CapabilityReport,
    CapabilityTier,
    RenderPath,
    TextureCompression,
};
pub use device::{DeviceCreationError, TrinityDevice};
pub use error_scope::{
    with_oom_scope,
    with_validation_scope,
    ErrorFilter,
    ErrorScope,
    ScopedErrorCapture,
};
pub use instance::{
    has_validation_errors,
    make_validation_error_callback,
    reset_validation_errors,
    TrinityInstance,
};
pub use limits::{
    negotiate_limits,
    LimitNegotiationError,
    LimitNegotiationResult,
    LimitRequirements,
    TrinityMinimumLimits,
};
pub use manager::{
    DeviceLostReason,
    DeviceManager,
    DeviceManagerError,
    DeviceState,
    RecoveryConfig,
    ResourceTracker,
};
pub use queue::{
    align_bytes_per_row,
    is_buffer_offset_aligned,
    is_bytes_per_row_aligned,
    BatcherConfig,
    BatcherMetrics,
    QueueWriteError,
    SubmissionBatcher,
    SubmissionTracker,
    TrinityQueue,
    COPY_BUFFER_ALIGNMENT,
    COPY_BYTES_PER_ROW_ALIGNMENT,
    DEFAULT_BATCH_COUNT_THRESHOLD,
    DEFAULT_BATCH_TIME_THRESHOLD_MS,
};
pub use requirements::{
    negotiate_and_create_device,
    negotiate_features,
    DeviceRequirements,
    FeatureNegotiationError,
    NegotiateAndCreateError,
    NegotiationResult,
};
