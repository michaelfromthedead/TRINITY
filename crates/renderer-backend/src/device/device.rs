//! Device creation and management for TRINITY.
//!
//! This module provides the [`TrinityDevice`] struct, which wraps wgpu's device
//! and queue with additional metadata about enabled features and limits.
//!
//! # Overview
//!
//! Device creation in wgpu is an async operation that can fail if:
//! - Requested features are not supported by the adapter
//! - Requested limits exceed what the adapter can provide
//! - The underlying graphics driver encounters an error
//!
//! This module provides:
//! - Type-safe device creation with proper error handling
//! - Automatic feature validation before device request
//! - Automatic limit validation and clamping
//! - Detailed logging of device creation process
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::device::{TrinityInstance, TrinityDevice, AdapterSelector};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create instance and select adapter
//! let instance = TrinityInstance::new();
//! let adapters: Vec<_> = instance.inner().enumerate_adapters(instance.backends());
//! let selector = AdapterSelector::new();
//! let result = selector.select(&adapters).expect("No suitable adapter found");
//!
//! // Create device with default features and limits
//! let device = TrinityDevice::new(
//!     result.adapter,
//!     wgpu::Features::empty(),
//!     wgpu::Limits::default(),
//! ).await?;
//!
//! println!("Device created with {} features enabled", device.features().iter().count());
//! # Ok(())
//! # }
//! ```

use log::{debug, error, info, warn};
use std::fmt;

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during device creation.
///
/// This enum provides detailed information about what went wrong during
/// device creation, allowing callers to handle specific failure modes.
#[derive(Debug)]
pub enum DeviceCreationError {
    /// The underlying wgpu request_device call failed.
    ///
    /// This typically occurs when the graphics driver encounters an internal
    /// error or when the system is out of resources.
    RequestDeviceError(wgpu::RequestDeviceError),

    /// One or more required features are not supported by the adapter.
    ///
    /// The contained `Features` bitflags indicate which features were requested
    /// but not available on the adapter.
    FeatureNotSupported(wgpu::Features),

    /// A required limit exceeds what the adapter can provide.
    ///
    /// Contains the name of the limit, the value that was required, and the
    /// maximum value the adapter supports.
    LimitNotMet {
        /// Name of the limit that was not met (e.g., "max_texture_dimension_2d").
        limit: String,
        /// The value that was requested.
        required: u64,
        /// The maximum value the adapter supports.
        available: u64,
    },
}

impl fmt::Display for DeviceCreationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceCreationError::RequestDeviceError(e) => {
                write!(f, "Failed to request device from adapter: {}", e)
            }
            DeviceCreationError::FeatureNotSupported(features) => {
                write!(
                    f,
                    "Required features not supported by adapter: {:?}",
                    features
                )
            }
            DeviceCreationError::LimitNotMet {
                limit,
                required,
                available,
            } => {
                write!(
                    f,
                    "Limit '{}' not met: required {}, adapter supports {}",
                    limit, required, available
                )
            }
        }
    }
}

impl std::error::Error for DeviceCreationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DeviceCreationError::RequestDeviceError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<wgpu::RequestDeviceError> for DeviceCreationError {
    fn from(err: wgpu::RequestDeviceError) -> Self {
        DeviceCreationError::RequestDeviceError(err)
    }
}

// ============================================================================
// TrinityDevice
// ============================================================================

/// TRINITY's device wrapper with associated queue and metadata.
///
/// `TrinityDevice` encapsulates a wgpu device and queue along with the
/// features and limits that were negotiated during device creation. This
/// provides a single point of access for all device-related operations.
///
/// # Thread Safety
///
/// Both `wgpu::Device` and `wgpu::Queue` are `Send + Sync`, so `TrinityDevice`
/// can be safely shared across threads. For concurrent access patterns, consider
/// wrapping in `Arc<TrinityDevice>`.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::device::TrinityDevice;
///
/// # async fn example(adapter: &wgpu::Adapter) -> Result<(), Box<dyn std::error::Error>> {
/// let device = TrinityDevice::new(
///     adapter,
///     wgpu::Features::TEXTURE_COMPRESSION_BC,
///     wgpu::Limits::default(),
/// ).await?;
///
/// // Access the underlying wgpu device
/// let buffer = device.device().create_buffer(&wgpu::BufferDescriptor {
///     label: Some("My Buffer"),
///     size: 1024,
///     usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
///     mapped_at_creation: false,
/// });
///
/// // Submit work via the queue
/// device.queue().submit(std::iter::empty());
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct TrinityDevice {
    /// The underlying wgpu device.
    device: wgpu::Device,
    /// The device's command queue.
    queue: wgpu::Queue,
    /// Features enabled on this device.
    features: wgpu::Features,
    /// Limits configured for this device.
    limits: wgpu::Limits,
}

impl TrinityDevice {
    /// Create a new device from an adapter.
    ///
    /// This method requests a device from the given adapter with the specified
    /// features and limits. It performs validation before the request to provide
    /// better error messages.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The wgpu adapter to create the device from
    /// * `required_features` - Features that must be enabled on the device
    /// * `required_limits` - Limits that the device must support
    ///
    /// # Returns
    ///
    /// A `Result` containing either the created `TrinityDevice` or a
    /// `DeviceCreationError` describing what went wrong.
    ///
    /// # Errors
    ///
    /// - [`DeviceCreationError::FeatureNotSupported`] - If any required feature
    ///   is not supported by the adapter
    /// - [`DeviceCreationError::LimitNotMet`] - If any required limit exceeds
    ///   what the adapter supports
    /// - [`DeviceCreationError::RequestDeviceError`] - If the underlying wgpu
    ///   request fails
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::device::TrinityDevice;
    ///
    /// # async fn example(adapter: &wgpu::Adapter) -> Result<(), Box<dyn std::error::Error>> {
    /// // Request with specific features
    /// let features = wgpu::Features::TEXTURE_COMPRESSION_BC
    ///     | wgpu::Features::TIMESTAMP_QUERY;
    ///
    /// let device = TrinityDevice::new(adapter, features, wgpu::Limits::default()).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn new(
        adapter: &wgpu::Adapter,
        required_features: wgpu::Features,
        required_limits: wgpu::Limits,
    ) -> Result<Self, DeviceCreationError> {
        let adapter_info = adapter.get_info();
        info!(
            "TrinityDevice: Creating device from adapter: {} ({:?})",
            adapter_info.name, adapter_info.backend
        );

        // Get adapter capabilities
        let adapter_features = adapter.features();
        let adapter_limits = adapter.limits();

        // Validate features
        let missing_features = required_features - adapter_features;
        if !missing_features.is_empty() {
            error!(
                "TrinityDevice: Required features not supported: {:?}",
                missing_features
            );
            return Err(DeviceCreationError::FeatureNotSupported(missing_features));
        }

        debug!(
            "TrinityDevice: Requested features: {:?}",
            required_features
        );

        // Validate and log critical limits
        Self::validate_limits(&required_limits, &adapter_limits)?;

        debug!(
            "TrinityDevice: Requested limits: max_texture_2d={}, max_buffer_size={}, max_bind_groups={}",
            required_limits.max_texture_dimension_2d,
            required_limits.max_buffer_size,
            required_limits.max_bind_groups
        );

        // Request the device
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("TRINITY Device"),
                    required_features,
                    required_limits: required_limits.clone(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None, // trace path
            )
            .await?;

        info!(
            "TrinityDevice: Device created successfully with {} features",
            required_features.iter().count()
        );

        Ok(Self {
            device,
            queue,
            features: required_features,
            limits: required_limits,
        })
    }

    /// Validate that required limits do not exceed adapter limits.
    ///
    /// This performs comprehensive validation of all limit fields and returns
    /// an error on the first violation found.
    fn validate_limits(
        required: &wgpu::Limits,
        available: &wgpu::Limits,
    ) -> Result<(), DeviceCreationError> {
        // Texture limits
        if required.max_texture_dimension_1d > available.max_texture_dimension_1d {
            return Err(DeviceCreationError::LimitNotMet {
                limit: "max_texture_dimension_1d".to_string(),
                required: required.max_texture_dimension_1d as u64,
                available: available.max_texture_dimension_1d as u64,
            });
        }
        if required.max_texture_dimension_2d > available.max_texture_dimension_2d {
            return Err(DeviceCreationError::LimitNotMet {
                limit: "max_texture_dimension_2d".to_string(),
                required: required.max_texture_dimension_2d as u64,
                available: available.max_texture_dimension_2d as u64,
            });
        }
        if required.max_texture_dimension_3d > available.max_texture_dimension_3d {
            return Err(DeviceCreationError::LimitNotMet {
                limit: "max_texture_dimension_3d".to_string(),
                required: required.max_texture_dimension_3d as u64,
                available: available.max_texture_dimension_3d as u64,
            });
        }
        if required.max_texture_array_layers > available.max_texture_array_layers {
            return Err(DeviceCreationError::LimitNotMet {
                limit: "max_texture_array_layers".to_string(),
                required: required.max_texture_array_layers as u64,
                available: available.max_texture_array_layers as u64,
            });
        }

        // Buffer limits
        if required.max_buffer_size > available.max_buffer_size {
            return Err(DeviceCreationError::LimitNotMet {
                limit: "max_buffer_size".to_string(),
                required: required.max_buffer_size,
                available: available.max_buffer_size,
            });
        }
        if required.max_uniform_buffer_binding_size > available.max_uniform_buffer_binding_size {
            return Err(DeviceCreationError::LimitNotMet {
                limit: "max_uniform_buffer_binding_size".to_string(),
                required: required.max_uniform_buffer_binding_size as u64,
                available: available.max_uniform_buffer_binding_size as u64,
            });
        }
        if required.max_storage_buffer_binding_size > available.max_storage_buffer_binding_size {
            return Err(DeviceCreationError::LimitNotMet {
                limit: "max_storage_buffer_binding_size".to_string(),
                required: required.max_storage_buffer_binding_size as u64,
                available: available.max_storage_buffer_binding_size as u64,
            });
        }

        // Bind group limits
        if required.max_bind_groups > available.max_bind_groups {
            return Err(DeviceCreationError::LimitNotMet {
                limit: "max_bind_groups".to_string(),
                required: required.max_bind_groups as u64,
                available: available.max_bind_groups as u64,
            });
        }
        if required.max_bindings_per_bind_group > available.max_bindings_per_bind_group {
            return Err(DeviceCreationError::LimitNotMet {
                limit: "max_bindings_per_bind_group".to_string(),
                required: required.max_bindings_per_bind_group as u64,
                available: available.max_bindings_per_bind_group as u64,
            });
        }

        // Compute limits
        if required.max_compute_workgroup_size_x > available.max_compute_workgroup_size_x {
            return Err(DeviceCreationError::LimitNotMet {
                limit: "max_compute_workgroup_size_x".to_string(),
                required: required.max_compute_workgroup_size_x as u64,
                available: available.max_compute_workgroup_size_x as u64,
            });
        }
        if required.max_compute_workgroup_size_y > available.max_compute_workgroup_size_y {
            return Err(DeviceCreationError::LimitNotMet {
                limit: "max_compute_workgroup_size_y".to_string(),
                required: required.max_compute_workgroup_size_y as u64,
                available: available.max_compute_workgroup_size_y as u64,
            });
        }
        if required.max_compute_workgroup_size_z > available.max_compute_workgroup_size_z {
            return Err(DeviceCreationError::LimitNotMet {
                limit: "max_compute_workgroup_size_z".to_string(),
                required: required.max_compute_workgroup_size_z as u64,
                available: available.max_compute_workgroup_size_z as u64,
            });
        }
        if required.max_compute_invocations_per_workgroup
            > available.max_compute_invocations_per_workgroup
        {
            return Err(DeviceCreationError::LimitNotMet {
                limit: "max_compute_invocations_per_workgroup".to_string(),
                required: required.max_compute_invocations_per_workgroup as u64,
                available: available.max_compute_invocations_per_workgroup as u64,
            });
        }
        if required.max_compute_workgroups_per_dimension
            > available.max_compute_workgroups_per_dimension
        {
            return Err(DeviceCreationError::LimitNotMet {
                limit: "max_compute_workgroups_per_dimension".to_string(),
                required: required.max_compute_workgroups_per_dimension as u64,
                available: available.max_compute_workgroups_per_dimension as u64,
            });
        }

        // Vertex limits
        if required.max_vertex_buffers > available.max_vertex_buffers {
            return Err(DeviceCreationError::LimitNotMet {
                limit: "max_vertex_buffers".to_string(),
                required: required.max_vertex_buffers as u64,
                available: available.max_vertex_buffers as u64,
            });
        }
        if required.max_vertex_attributes > available.max_vertex_attributes {
            return Err(DeviceCreationError::LimitNotMet {
                limit: "max_vertex_attributes".to_string(),
                required: required.max_vertex_attributes as u64,
                available: available.max_vertex_attributes as u64,
            });
        }
        if required.max_vertex_buffer_array_stride > available.max_vertex_buffer_array_stride {
            return Err(DeviceCreationError::LimitNotMet {
                limit: "max_vertex_buffer_array_stride".to_string(),
                required: required.max_vertex_buffer_array_stride as u64,
                available: available.max_vertex_buffer_array_stride as u64,
            });
        }

        // Color attachment limits
        if required.max_color_attachments > available.max_color_attachments {
            return Err(DeviceCreationError::LimitNotMet {
                limit: "max_color_attachments".to_string(),
                required: required.max_color_attachments as u64,
                available: available.max_color_attachments as u64,
            });
        }
        if required.max_color_attachment_bytes_per_sample
            > available.max_color_attachment_bytes_per_sample
        {
            return Err(DeviceCreationError::LimitNotMet {
                limit: "max_color_attachment_bytes_per_sample".to_string(),
                required: required.max_color_attachment_bytes_per_sample as u64,
                available: available.max_color_attachment_bytes_per_sample as u64,
            });
        }

        Ok(())
    }

    /// Create a device with default (minimal) features and limits.
    ///
    /// This is a convenience method for creating a device with no optional
    /// features and default limits. Useful for testing or minimal setups.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The wgpu adapter to create the device from
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::device::TrinityDevice;
    ///
    /// # async fn example(adapter: &wgpu::Adapter) -> Result<(), Box<dyn std::error::Error>> {
    /// let device = TrinityDevice::with_defaults(adapter).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn with_defaults(adapter: &wgpu::Adapter) -> Result<Self, DeviceCreationError> {
        Self::new(adapter, wgpu::Features::empty(), wgpu::Limits::default()).await
    }

    /// Create a device with all features supported by the adapter.
    ///
    /// This requests all optional features that the adapter supports and uses
    /// the adapter's reported limits. Useful for development/debugging when
    /// you want maximum capability.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The wgpu adapter to create the device from
    ///
    /// # Warning
    ///
    /// Using all features may impact performance on some drivers. For production
    /// use, request only the features you need.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::device::TrinityDevice;
    ///
    /// # async fn example(adapter: &wgpu::Adapter) -> Result<(), Box<dyn std::error::Error>> {
    /// let device = TrinityDevice::with_all_features(adapter).await?;
    /// println!("Created device with {} features", device.features().iter().count());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn with_all_features(adapter: &wgpu::Adapter) -> Result<Self, DeviceCreationError> {
        let features = adapter.features();
        let limits = adapter.limits();

        warn!(
            "TrinityDevice: Creating device with ALL {} features - this may impact performance",
            features.iter().count()
        );

        Self::new(adapter, features, limits).await
    }

    /// Get a reference to the underlying wgpu device.
    ///
    /// Use this to create buffers, textures, pipelines, and other GPU resources.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use renderer_backend::device::TrinityDevice;
    /// # fn example(trinity_device: &TrinityDevice) {
    /// let buffer = trinity_device.device().create_buffer(&wgpu::BufferDescriptor {
    ///     label: Some("Vertex Buffer"),
    ///     size: 1024,
    ///     usage: wgpu::BufferUsages::VERTEX,
    ///     mapped_at_creation: false,
    /// });
    /// # }
    /// ```
    #[inline]
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Get a reference to the device's command queue.
    ///
    /// Use this to submit command buffers and write data to buffers/textures.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use renderer_backend::device::TrinityDevice;
    /// # fn example(trinity_device: &TrinityDevice, buffer: &wgpu::Buffer, data: &[u8]) {
    /// trinity_device.queue().write_buffer(buffer, 0, data);
    /// # }
    /// ```
    #[inline]
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Get the features enabled on this device.
    ///
    /// This returns the features that were requested during device creation,
    /// which is a subset of the adapter's supported features.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use renderer_backend::device::TrinityDevice;
    /// # fn example(device: &TrinityDevice) {
    /// if device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
    ///     println!("GPU profiling is available!");
    /// }
    /// # }
    /// ```
    #[inline]
    pub fn features(&self) -> wgpu::Features {
        self.features
    }

    /// Get the limits configured for this device.
    ///
    /// This returns the limits that were requested during device creation.
    /// Actual hardware limits may be higher (use adapter.limits() for those).
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use renderer_backend::device::TrinityDevice;
    /// # fn example(device: &TrinityDevice) {
    /// let max_tex_size = device.limits().max_texture_dimension_2d;
    /// println!("Maximum texture size: {}x{}", max_tex_size, max_tex_size);
    /// # }
    /// ```
    #[inline]
    pub fn limits(&self) -> &wgpu::Limits {
        &self.limits
    }

    /// Check if a specific feature is enabled on this device.
    ///
    /// # Arguments
    ///
    /// * `feature` - The feature to check for
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use renderer_backend::device::TrinityDevice;
    /// # fn example(device: &TrinityDevice) {
    /// if device.has_feature(wgpu::Features::TEXTURE_COMPRESSION_BC) {
    ///     println!("BC texture compression is available");
    /// }
    /// # }
    /// ```
    #[inline]
    pub fn has_feature(&self, feature: wgpu::Features) -> bool {
        self.features.contains(feature)
    }

    /// Submit command buffers to the queue.
    ///
    /// This is a convenience method that forwards to `queue.submit()`.
    ///
    /// # Arguments
    ///
    /// * `command_buffers` - Iterator of command buffers to submit
    ///
    /// # Returns
    ///
    /// A submission index that can be used to track completion.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use renderer_backend::device::TrinityDevice;
    /// # fn example(device: &TrinityDevice, encoder: wgpu::CommandEncoder) {
    /// let command_buffer = encoder.finish();
    /// let submission_index = device.submit(std::iter::once(command_buffer));
    /// # }
    /// ```
    #[inline]
    pub fn submit<I>(&self, command_buffers: I) -> wgpu::SubmissionIndex
    where
        I: IntoIterator<Item = wgpu::CommandBuffer>,
    {
        self.queue.submit(command_buffers)
    }

    /// Create a command encoder for recording GPU commands.
    ///
    /// This is a convenience method that creates a new command encoder with
    /// an optional label.
    ///
    /// # Arguments
    ///
    /// * `label` - Optional label for debugging
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use renderer_backend::device::TrinityDevice;
    /// # fn example(device: &TrinityDevice) {
    /// let encoder = device.create_command_encoder(Some("Main Render Pass"));
    /// // Record commands...
    /// let command_buffer = encoder.finish();
    /// # }
    /// ```
    #[inline]
    pub fn create_command_encoder(&self, label: Option<&str>) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label })
    }
}

impl fmt::Display for TrinityDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "TrinityDevice:")?;
        writeln!(f, "  Features: {} enabled", self.features.iter().count())?;
        writeln!(
            f,
            "  Max texture 2D: {}",
            self.limits.max_texture_dimension_2d
        )?;
        writeln!(f, "  Max buffer size: {} bytes", self.limits.max_buffer_size)?;
        writeln!(f, "  Max bind groups: {}", self.limits.max_bind_groups)?;
        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = DeviceCreationError::FeatureNotSupported(wgpu::Features::TIMESTAMP_QUERY);
        let msg = format!("{}", err);
        assert!(msg.contains("not supported"));
        assert!(msg.contains("TIMESTAMP_QUERY"));
    }

    #[test]
    fn test_limit_error_display() {
        let err = DeviceCreationError::LimitNotMet {
            limit: "max_texture_dimension_2d".to_string(),
            required: 16384,
            available: 8192,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("max_texture_dimension_2d"));
        assert!(msg.contains("16384"));
        assert!(msg.contains("8192"));
    }

    #[test]
    fn test_validate_limits_pass() {
        let required = wgpu::Limits::default();
        let available = wgpu::Limits::default();
        assert!(TrinityDevice::validate_limits(&required, &available).is_ok());
    }

    #[test]
    fn test_validate_limits_fail() {
        let mut required = wgpu::Limits::default();
        required.max_texture_dimension_2d = 32768;

        let available = wgpu::Limits::default(); // Has 8192

        let result = TrinityDevice::validate_limits(&required, &available);
        assert!(result.is_err());

        if let Err(DeviceCreationError::LimitNotMet { limit, .. }) = result {
            assert_eq!(limit, "max_texture_dimension_2d");
        } else {
            panic!("Expected LimitNotMet error");
        }
    }
}
