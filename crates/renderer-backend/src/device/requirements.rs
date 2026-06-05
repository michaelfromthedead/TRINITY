//! Device requirements and feature negotiation for TRINITY.
//!
//! This module provides the [`DeviceRequirements`] struct for specifying required
//! and optional GPU features, and the [`negotiate_features`] function for resolving
//! these requirements against adapter capabilities.
//!
//! # Overview
//!
//! Feature negotiation allows TRINITY to:
//! - Fail fast when required features are unavailable
//! - Gracefully degrade when optional features are missing
//! - Automatically handle feature dependencies
//! - Log the final negotiated feature set
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::device::{DeviceRequirements, negotiate_features, TrinityDevice};
//!
//! # async fn example(adapter: &wgpu::Adapter) -> Result<(), Box<dyn std::error::Error>> {
//! // Build requirements with builder pattern
//! let requirements = DeviceRequirements::new()
//!     .require(wgpu::Features::TEXTURE_COMPRESSION_BC)
//!     .prefer(wgpu::Features::POLYGON_MODE_LINE)
//!     .prefer(wgpu::Features::PUSH_CONSTANTS)
//!     .with_limits(wgpu::Limits::default());
//!
//! // Negotiate features against adapter
//! let result = negotiate_features(&requirements, adapter)?;
//!
//! // Create device with negotiated features
//! let device = TrinityDevice::new(
//!     adapter,
//!     result.enabled_features,
//!     result.limits,
//! ).await?;
//!
//! // Check what optional features were degraded
//! if !result.degraded_features.is_empty() {
//!     println!("Running without: {:?}", result.degraded_features);
//! }
//! # Ok(())
//! # }
//! ```

use log::{debug, info, warn};
use std::fmt;

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during feature negotiation.
///
/// These errors indicate that the adapter cannot meet the device requirements.
/// Required features that are missing will cause negotiation to fail, while
/// optional features are silently degraded.
#[derive(Debug, Clone)]
pub enum FeatureNegotiationError {
    /// One or more required features are not available on the adapter.
    ///
    /// The contained `Features` bitflags indicate which features were required
    /// but not supported by the adapter.
    RequiredFeaturesMissing(wgpu::Features),

    /// A required limit exceeds what the adapter can provide.
    ///
    /// Contains the name of the limit, the value that was required, and the
    /// maximum value the adapter supports.
    LimitsExceedCapabilities {
        /// Name of the limit that was not met (e.g., "max_texture_dimension_2d").
        limit: String,
        /// The value that was requested.
        required: u32,
        /// The maximum value the adapter supports.
        available: u32,
    },
}

impl fmt::Display for FeatureNegotiationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FeatureNegotiationError::RequiredFeaturesMissing(features) => {
                write!(
                    f,
                    "Required features not available on adapter: {:?}",
                    features
                )
            }
            FeatureNegotiationError::LimitsExceedCapabilities {
                limit,
                required,
                available,
            } => {
                write!(
                    f,
                    "Required limit '{}' exceeds adapter capabilities: required {}, available {}",
                    limit, required, available
                )
            }
        }
    }
}

impl std::error::Error for FeatureNegotiationError {}

// ============================================================================
// DeviceRequirements
// ============================================================================

/// Device requirements specifying required and optional features.
///
/// This struct uses a builder pattern to accumulate feature requirements.
/// Required features will cause device creation to fail if unavailable,
/// while optional features will be enabled if available or gracefully
/// degraded if not.
///
/// # Example
///
/// ```
/// use renderer_backend::device::DeviceRequirements;
///
/// let requirements = DeviceRequirements::new()
///     // These MUST be available
///     .require(wgpu::Features::TEXTURE_COMPRESSION_BC)
///     .require(wgpu::Features::DEPTH_CLIP_CONTROL)
///     // These are nice-to-have
///     .prefer(wgpu::Features::POLYGON_MODE_LINE)
///     .prefer(wgpu::Features::PUSH_CONSTANTS)
///     // Set minimum limits
///     .with_limits(wgpu::Limits::downlevel_defaults());
/// ```
#[derive(Debug, Clone)]
pub struct DeviceRequirements {
    /// Features that MUST be available (failure if unavailable).
    pub required_features: wgpu::Features,
    /// Features that are nice-to-have (graceful degradation if unavailable).
    pub optional_features: wgpu::Features,
    /// Required limits (use Limits::default() for minimum).
    pub required_limits: wgpu::Limits,
}

impl Default for DeviceRequirements {
    fn default() -> Self {
        Self::new()
    }
}

impl DeviceRequirements {
    /// Create a new empty requirements specification.
    ///
    /// By default, no features are required or preferred, and the default
    /// wgpu limits are used.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::device::DeviceRequirements;
    ///
    /// let requirements = DeviceRequirements::new();
    /// assert!(requirements.required_features.is_empty());
    /// assert!(requirements.optional_features.is_empty());
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            required_features: wgpu::Features::empty(),
            optional_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        }
    }

    /// Add a required feature (will fail if unavailable).
    ///
    /// Required features are non-negotiable. If the adapter does not support
    /// these features, [`negotiate_features`] will return an error.
    ///
    /// # Arguments
    ///
    /// * `feature` - The feature flag(s) to require
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::device::DeviceRequirements;
    ///
    /// let requirements = DeviceRequirements::new()
    ///     .require(wgpu::Features::TEXTURE_COMPRESSION_BC);
    /// ```
    #[must_use]
    pub fn require(mut self, feature: wgpu::Features) -> Self {
        self.required_features |= feature;
        self
    }

    /// Add an optional feature (will degrade gracefully).
    ///
    /// Optional features are enabled if the adapter supports them, but their
    /// absence does not cause negotiation to fail. The [`NegotiationResult`]
    /// will indicate which optional features were degraded.
    ///
    /// # Arguments
    ///
    /// * `feature` - The feature flag(s) to prefer
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::device::DeviceRequirements;
    ///
    /// let requirements = DeviceRequirements::new()
    ///     .prefer(wgpu::Features::POLYGON_MODE_LINE)
    ///     .prefer(wgpu::Features::TIMESTAMP_QUERY);
    /// ```
    #[must_use]
    pub fn prefer(mut self, feature: wgpu::Features) -> Self {
        self.optional_features |= feature;
        self
    }

    /// Set required limits.
    ///
    /// These limits represent the minimum capabilities the device must provide.
    /// If the adapter cannot meet these limits, negotiation will fail.
    ///
    /// # Arguments
    ///
    /// * `limits` - The minimum limits to require
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::device::DeviceRequirements;
    ///
    /// let mut limits = wgpu::Limits::default();
    /// limits.max_texture_dimension_2d = 4096;
    ///
    /// let requirements = DeviceRequirements::new()
    ///     .with_limits(limits);
    /// ```
    #[must_use]
    pub fn with_limits(mut self, limits: wgpu::Limits) -> Self {
        self.required_limits = limits;
        self
    }

    /// Create requirements for minimal/compatibility mode.
    ///
    /// This preset uses downlevel defaults and no required features,
    /// maximizing compatibility with low-end hardware.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::device::DeviceRequirements;
    ///
    /// let requirements = DeviceRequirements::minimal();
    /// ```
    #[must_use]
    pub fn minimal() -> Self {
        Self {
            required_features: wgpu::Features::empty(),
            optional_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
        }
    }

    /// Create requirements for standard rendering.
    ///
    /// This preset includes common features needed for typical 3D rendering:
    /// - BC texture compression (optional)
    /// - Polygon mode line for wireframe (optional)
    /// - Timestamp queries for profiling (optional)
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::device::DeviceRequirements;
    ///
    /// let requirements = DeviceRequirements::standard();
    /// ```
    #[must_use]
    pub fn standard() -> Self {
        Self {
            required_features: wgpu::Features::empty(),
            optional_features: wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::POLYGON_MODE_LINE
                | wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::DEPTH_CLIP_CONTROL,
            required_limits: wgpu::Limits::default(),
        }
    }

    /// Create requirements for advanced rendering.
    ///
    /// This preset includes features for advanced rendering techniques:
    /// - Required: Push constants for efficient uniform updates
    /// - Optional: Multi-draw indirect, texture arrays, etc.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::device::DeviceRequirements;
    ///
    /// let requirements = DeviceRequirements::advanced();
    /// ```
    #[must_use]
    pub fn advanced() -> Self {
        Self {
            required_features: wgpu::Features::PUSH_CONSTANTS,
            optional_features: wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::POLYGON_MODE_LINE
                | wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::MULTI_DRAW_INDIRECT
                | wgpu::Features::MULTI_DRAW_INDIRECT_COUNT
                | wgpu::Features::TEXTURE_BINDING_ARRAY
                | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
                | wgpu::Features::DEPTH_CLIP_CONTROL,
            required_limits: wgpu::Limits::default(),
        }
    }
}

impl fmt::Display for DeviceRequirements {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "DeviceRequirements:")?;
        writeln!(
            f,
            "  Required features: {} ({} total)",
            if self.required_features.is_empty() {
                "none".to_string()
            } else {
                format!("{:?}", self.required_features)
            },
            self.required_features.iter().count()
        )?;
        writeln!(
            f,
            "  Optional features: {} ({} total)",
            if self.optional_features.is_empty() {
                "none".to_string()
            } else {
                format!("{:?}", self.optional_features)
            },
            self.optional_features.iter().count()
        )?;
        writeln!(
            f,
            "  Required limits: max_texture_2d={}, max_buffer_size={}",
            self.required_limits.max_texture_dimension_2d, self.required_limits.max_buffer_size
        )?;
        Ok(())
    }
}

// ============================================================================
// NegotiationResult
// ============================================================================

/// Result of feature negotiation.
///
/// This struct contains the outcome of negotiating device requirements against
/// adapter capabilities. It includes the final set of features to enable and
/// information about which optional features were unavailable.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::device::{DeviceRequirements, negotiate_features};
///
/// # fn example(adapter: &wgpu::Adapter) -> Result<(), Box<dyn std::error::Error>> {
/// let requirements = DeviceRequirements::standard();
/// let result = negotiate_features(&requirements, adapter)?;
///
/// println!("Enabled {} features", result.enabled_features.iter().count());
/// if !result.degraded_features.is_empty() {
///     println!("Degraded features: {:?}", result.degraded_features);
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct NegotiationResult {
    /// Features that will be enabled on the device.
    ///
    /// This includes all required features plus any optional features that
    /// the adapter supports.
    pub enabled_features: wgpu::Features,

    /// Optional features that were unavailable.
    ///
    /// These features were requested but are not supported by the adapter.
    /// The application should handle their absence gracefully.
    pub degraded_features: wgpu::Features,

    /// Negotiated limits.
    ///
    /// These are the limits that will be requested from the device. They are
    /// at least as large as the required limits, and may be larger if the
    /// adapter supports higher limits.
    pub limits: wgpu::Limits,
}

impl NegotiationResult {
    /// Check if a specific feature was enabled.
    ///
    /// # Arguments
    ///
    /// * `feature` - The feature to check
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use renderer_backend::device::{DeviceRequirements, negotiate_features, NegotiationResult};
    /// # fn example(result: &NegotiationResult) {
    /// if result.has_feature(wgpu::Features::PUSH_CONSTANTS) {
    ///     println!("Push constants available!");
    /// }
    /// # }
    /// ```
    #[inline]
    pub fn has_feature(&self, feature: wgpu::Features) -> bool {
        self.enabled_features.contains(feature)
    }

    /// Check if a specific feature was degraded (unavailable).
    ///
    /// # Arguments
    ///
    /// * `feature` - The feature to check
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use renderer_backend::device::{DeviceRequirements, negotiate_features, NegotiationResult};
    /// # fn example(result: &NegotiationResult) {
    /// if result.was_degraded(wgpu::Features::POLYGON_MODE_LINE) {
    ///     println!("Wireframe rendering unavailable, using fallback");
    /// }
    /// # }
    /// ```
    #[inline]
    pub fn was_degraded(&self, feature: wgpu::Features) -> bool {
        self.degraded_features.contains(feature)
    }

    /// Get the count of enabled features.
    #[inline]
    pub fn enabled_count(&self) -> usize {
        self.enabled_features.iter().count()
    }

    /// Get the count of degraded features.
    #[inline]
    pub fn degraded_count(&self) -> usize {
        self.degraded_features.iter().count()
    }
}

impl fmt::Display for NegotiationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "NegotiationResult:")?;
        writeln!(
            f,
            "  Enabled: {} features",
            self.enabled_features.iter().count()
        )?;
        writeln!(
            f,
            "  Degraded: {} features",
            self.degraded_features.iter().count()
        )?;
        if !self.degraded_features.is_empty() {
            writeln!(f, "  Degraded list: {:?}", self.degraded_features)?;
        }
        Ok(())
    }
}

// ============================================================================
// Feature Dependencies
// ============================================================================

/// Get implicit feature dependencies.
///
/// Some wgpu features imply or require other features. This function returns
/// any features that should be automatically enabled when the given feature
/// is requested.
///
/// Note: wgpu 0.20+ handles most dependencies internally, but we document
/// known relationships here for clarity and future-proofing.
fn get_feature_dependencies(feature: wgpu::Features) -> wgpu::Features {
    let mut deps = wgpu::Features::empty();

    // MULTI_DRAW_INDIRECT_COUNT requires MULTI_DRAW_INDIRECT
    if feature.contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT) {
        deps |= wgpu::Features::MULTI_DRAW_INDIRECT;
    }

    // SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
    // requires TEXTURE_BINDING_ARRAY
    if feature.contains(wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING)
    {
        deps |= wgpu::Features::TEXTURE_BINDING_ARRAY;
    }

    // UNIFORM_BUFFER_AND_STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING
    // requires TEXTURE_BINDING_ARRAY
    if feature.contains(wgpu::Features::UNIFORM_BUFFER_AND_STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING)
    {
        deps |= wgpu::Features::TEXTURE_BINDING_ARRAY;
    }

    // PARTIALLY_BOUND_BINDING_ARRAY requires TEXTURE_BINDING_ARRAY
    if feature.contains(wgpu::Features::PARTIALLY_BOUND_BINDING_ARRAY) {
        deps |= wgpu::Features::TEXTURE_BINDING_ARRAY;
    }

    deps
}

/// Expand features to include all dependencies.
///
/// This function takes a set of features and returns the expanded set including
/// all implicit dependencies.
fn expand_feature_dependencies(features: wgpu::Features) -> wgpu::Features {
    let mut expanded = features;
    let mut prev = wgpu::Features::empty();

    // Iterate until no new dependencies are found
    while expanded != prev {
        prev = expanded;
        for feature in expanded.iter() {
            expanded |= get_feature_dependencies(feature);
        }
    }

    expanded
}

// ============================================================================
// Negotiation Function
// ============================================================================

/// Negotiate features between requirements and adapter capabilities.
///
/// This function resolves the requested device requirements against what the
/// adapter actually supports. It:
///
/// 1. Expands feature dependencies (e.g., if A requires B, B is auto-added)
/// 2. Validates that all required features are available
/// 3. Validates that all required limits are met
/// 4. Determines which optional features can be enabled
/// 5. Reports which optional features were degraded
///
/// # Arguments
///
/// * `requirements` - The device requirements to negotiate
/// * `adapter` - The wgpu adapter to negotiate against
///
/// # Returns
///
/// A `Result` containing either a [`NegotiationResult`] with the negotiated
/// features and limits, or a [`FeatureNegotiationError`] if negotiation failed.
///
/// # Errors
///
/// - [`FeatureNegotiationError::RequiredFeaturesMissing`] - If any required
///   feature (or its dependencies) is not supported
/// - [`FeatureNegotiationError::LimitsExceedCapabilities`] - If any required
///   limit exceeds adapter capabilities
///
/// # Example
///
/// ```no_run
/// use renderer_backend::device::{DeviceRequirements, negotiate_features};
///
/// # fn example(adapter: &wgpu::Adapter) -> Result<(), Box<dyn std::error::Error>> {
/// let requirements = DeviceRequirements::new()
///     .require(wgpu::Features::TEXTURE_COMPRESSION_BC)
///     .prefer(wgpu::Features::POLYGON_MODE_LINE);
///
/// let result = negotiate_features(&requirements, adapter)?;
/// println!("Enabled features: {:?}", result.enabled_features);
/// # Ok(())
/// # }
/// ```
pub fn negotiate_features(
    requirements: &DeviceRequirements,
    adapter: &wgpu::Adapter,
) -> Result<NegotiationResult, FeatureNegotiationError> {
    let adapter_info = adapter.get_info();
    let adapter_features = adapter.features();
    let adapter_limits = adapter.limits();

    info!(
        "Feature negotiation: {} required, {} optional features against adapter: {} ({:?})",
        requirements.required_features.iter().count(),
        requirements.optional_features.iter().count(),
        adapter_info.name,
        adapter_info.backend
    );

    debug!(
        "Required features: {:?}",
        requirements.required_features
    );
    debug!(
        "Optional features: {:?}",
        requirements.optional_features
    );
    debug!("Adapter features: {:?}", adapter_features);

    // Step 1: Expand required features to include dependencies
    let expanded_required = expand_feature_dependencies(requirements.required_features);
    if expanded_required != requirements.required_features {
        let added = expanded_required - requirements.required_features;
        info!(
            "Auto-adding feature dependencies: {:?}",
            added
        );
    }

    // Step 2: Check that all required features are available
    let missing_required = expanded_required - adapter_features;
    if !missing_required.is_empty() {
        warn!(
            "Required features not available: {:?}",
            missing_required
        );
        return Err(FeatureNegotiationError::RequiredFeaturesMissing(
            missing_required,
        ));
    }

    // Step 3: Validate limits
    validate_limits_for_negotiation(&requirements.required_limits, &adapter_limits)?;

    // Step 4: Expand optional features to include dependencies
    let expanded_optional = expand_feature_dependencies(requirements.optional_features);

    // Step 5: Determine which optional features are available
    let available_optional = expanded_optional & adapter_features;
    let degraded_optional = expanded_optional - adapter_features;

    if !degraded_optional.is_empty() {
        warn!(
            "Optional features degraded (unavailable): {:?}",
            degraded_optional
        );
    }

    // Step 6: Build final feature set
    let enabled_features = expanded_required | available_optional;

    // Step 7: Use required limits (could upgrade to adapter limits if desired)
    let limits = requirements.required_limits.clone();

    info!(
        "Negotiation result: {} enabled, {} degraded",
        enabled_features.iter().count(),
        degraded_optional.iter().count()
    );

    Ok(NegotiationResult {
        enabled_features,
        degraded_features: degraded_optional,
        limits,
    })
}

/// Validate that required limits do not exceed adapter limits.
fn validate_limits_for_negotiation(
    required: &wgpu::Limits,
    available: &wgpu::Limits,
) -> Result<(), FeatureNegotiationError> {
    // Texture limits
    if required.max_texture_dimension_1d > available.max_texture_dimension_1d {
        return Err(FeatureNegotiationError::LimitsExceedCapabilities {
            limit: "max_texture_dimension_1d".to_string(),
            required: required.max_texture_dimension_1d,
            available: available.max_texture_dimension_1d,
        });
    }
    if required.max_texture_dimension_2d > available.max_texture_dimension_2d {
        return Err(FeatureNegotiationError::LimitsExceedCapabilities {
            limit: "max_texture_dimension_2d".to_string(),
            required: required.max_texture_dimension_2d,
            available: available.max_texture_dimension_2d,
        });
    }
    if required.max_texture_dimension_3d > available.max_texture_dimension_3d {
        return Err(FeatureNegotiationError::LimitsExceedCapabilities {
            limit: "max_texture_dimension_3d".to_string(),
            required: required.max_texture_dimension_3d,
            available: available.max_texture_dimension_3d,
        });
    }
    if required.max_texture_array_layers > available.max_texture_array_layers {
        return Err(FeatureNegotiationError::LimitsExceedCapabilities {
            limit: "max_texture_array_layers".to_string(),
            required: required.max_texture_array_layers,
            available: available.max_texture_array_layers,
        });
    }

    // Buffer limits
    if required.max_buffer_size > available.max_buffer_size {
        return Err(FeatureNegotiationError::LimitsExceedCapabilities {
            limit: "max_buffer_size".to_string(),
            required: required.max_buffer_size as u32,
            available: available.max_buffer_size as u32,
        });
    }
    if required.max_uniform_buffer_binding_size > available.max_uniform_buffer_binding_size {
        return Err(FeatureNegotiationError::LimitsExceedCapabilities {
            limit: "max_uniform_buffer_binding_size".to_string(),
            required: required.max_uniform_buffer_binding_size,
            available: available.max_uniform_buffer_binding_size,
        });
    }
    if required.max_storage_buffer_binding_size > available.max_storage_buffer_binding_size {
        return Err(FeatureNegotiationError::LimitsExceedCapabilities {
            limit: "max_storage_buffer_binding_size".to_string(),
            required: required.max_storage_buffer_binding_size,
            available: available.max_storage_buffer_binding_size,
        });
    }

    // Bind group limits
    if required.max_bind_groups > available.max_bind_groups {
        return Err(FeatureNegotiationError::LimitsExceedCapabilities {
            limit: "max_bind_groups".to_string(),
            required: required.max_bind_groups,
            available: available.max_bind_groups,
        });
    }
    if required.max_bindings_per_bind_group > available.max_bindings_per_bind_group {
        return Err(FeatureNegotiationError::LimitsExceedCapabilities {
            limit: "max_bindings_per_bind_group".to_string(),
            required: required.max_bindings_per_bind_group,
            available: available.max_bindings_per_bind_group,
        });
    }

    // Compute limits
    if required.max_compute_workgroup_size_x > available.max_compute_workgroup_size_x {
        return Err(FeatureNegotiationError::LimitsExceedCapabilities {
            limit: "max_compute_workgroup_size_x".to_string(),
            required: required.max_compute_workgroup_size_x,
            available: available.max_compute_workgroup_size_x,
        });
    }
    if required.max_compute_workgroup_size_y > available.max_compute_workgroup_size_y {
        return Err(FeatureNegotiationError::LimitsExceedCapabilities {
            limit: "max_compute_workgroup_size_y".to_string(),
            required: required.max_compute_workgroup_size_y,
            available: available.max_compute_workgroup_size_y,
        });
    }
    if required.max_compute_workgroup_size_z > available.max_compute_workgroup_size_z {
        return Err(FeatureNegotiationError::LimitsExceedCapabilities {
            limit: "max_compute_workgroup_size_z".to_string(),
            required: required.max_compute_workgroup_size_z,
            available: available.max_compute_workgroup_size_z,
        });
    }
    if required.max_compute_invocations_per_workgroup
        > available.max_compute_invocations_per_workgroup
    {
        return Err(FeatureNegotiationError::LimitsExceedCapabilities {
            limit: "max_compute_invocations_per_workgroup".to_string(),
            required: required.max_compute_invocations_per_workgroup,
            available: available.max_compute_invocations_per_workgroup,
        });
    }
    if required.max_compute_workgroups_per_dimension
        > available.max_compute_workgroups_per_dimension
    {
        return Err(FeatureNegotiationError::LimitsExceedCapabilities {
            limit: "max_compute_workgroups_per_dimension".to_string(),
            required: required.max_compute_workgroups_per_dimension,
            available: available.max_compute_workgroups_per_dimension,
        });
    }

    // Vertex limits
    if required.max_vertex_buffers > available.max_vertex_buffers {
        return Err(FeatureNegotiationError::LimitsExceedCapabilities {
            limit: "max_vertex_buffers".to_string(),
            required: required.max_vertex_buffers,
            available: available.max_vertex_buffers,
        });
    }
    if required.max_vertex_attributes > available.max_vertex_attributes {
        return Err(FeatureNegotiationError::LimitsExceedCapabilities {
            limit: "max_vertex_attributes".to_string(),
            required: required.max_vertex_attributes,
            available: available.max_vertex_attributes,
        });
    }
    if required.max_vertex_buffer_array_stride > available.max_vertex_buffer_array_stride {
        return Err(FeatureNegotiationError::LimitsExceedCapabilities {
            limit: "max_vertex_buffer_array_stride".to_string(),
            required: required.max_vertex_buffer_array_stride,
            available: available.max_vertex_buffer_array_stride,
        });
    }

    // Color attachment limits
    if required.max_color_attachments > available.max_color_attachments {
        return Err(FeatureNegotiationError::LimitsExceedCapabilities {
            limit: "max_color_attachments".to_string(),
            required: required.max_color_attachments,
            available: available.max_color_attachments,
        });
    }
    if required.max_color_attachment_bytes_per_sample
        > available.max_color_attachment_bytes_per_sample
    {
        return Err(FeatureNegotiationError::LimitsExceedCapabilities {
            limit: "max_color_attachment_bytes_per_sample".to_string(),
            required: required.max_color_attachment_bytes_per_sample,
            available: available.max_color_attachment_bytes_per_sample,
        });
    }

    Ok(())
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Negotiate features and create a device in one step.
///
/// This is a convenience function that combines [`negotiate_features`] with
/// [`TrinityDevice::new`].
///
/// # Arguments
///
/// * `requirements` - The device requirements to negotiate
/// * `adapter` - The wgpu adapter to create the device from
///
/// # Returns
///
/// A `Result` containing either a tuple of (`TrinityDevice`, `NegotiationResult`)
/// or an error if negotiation or device creation failed.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::device::{DeviceRequirements, negotiate_and_create_device};
///
/// # async fn example(adapter: &wgpu::Adapter) -> Result<(), Box<dyn std::error::Error>> {
/// let requirements = DeviceRequirements::standard();
/// let (device, negotiation) = negotiate_and_create_device(&requirements, adapter).await?;
///
/// println!("Device created with {} features", negotiation.enabled_count());
/// # Ok(())
/// # }
/// ```
pub async fn negotiate_and_create_device(
    requirements: &DeviceRequirements,
    adapter: &wgpu::Adapter,
) -> Result<(super::TrinityDevice, NegotiationResult), NegotiateAndCreateError> {
    let negotiation = negotiate_features(requirements, adapter)?;

    let device = super::TrinityDevice::new(
        adapter,
        negotiation.enabled_features,
        negotiation.limits.clone(),
    )
    .await?;

    Ok((device, negotiation))
}

/// Error type for [`negotiate_and_create_device`].
#[derive(Debug)]
pub enum NegotiateAndCreateError {
    /// Feature negotiation failed.
    Negotiation(FeatureNegotiationError),
    /// Device creation failed.
    DeviceCreation(super::DeviceCreationError),
}

impl fmt::Display for NegotiateAndCreateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NegotiateAndCreateError::Negotiation(e) => {
                write!(f, "Feature negotiation failed: {}", e)
            }
            NegotiateAndCreateError::DeviceCreation(e) => {
                write!(f, "Device creation failed: {}", e)
            }
        }
    }
}

impl std::error::Error for NegotiateAndCreateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            NegotiateAndCreateError::Negotiation(e) => Some(e),
            NegotiateAndCreateError::DeviceCreation(e) => Some(e),
        }
    }
}

impl From<FeatureNegotiationError> for NegotiateAndCreateError {
    fn from(err: FeatureNegotiationError) -> Self {
        NegotiateAndCreateError::Negotiation(err)
    }
}

impl From<super::DeviceCreationError> for NegotiateAndCreateError {
    fn from(err: super::DeviceCreationError) -> Self {
        NegotiateAndCreateError::DeviceCreation(err)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_requirements_new() {
        let req = DeviceRequirements::new();
        assert!(req.required_features.is_empty());
        assert!(req.optional_features.is_empty());
    }

    #[test]
    fn test_device_requirements_builder() {
        let req = DeviceRequirements::new()
            .require(wgpu::Features::TEXTURE_COMPRESSION_BC)
            .prefer(wgpu::Features::POLYGON_MODE_LINE)
            .prefer(wgpu::Features::TIMESTAMP_QUERY);

        assert!(req
            .required_features
            .contains(wgpu::Features::TEXTURE_COMPRESSION_BC));
        assert!(req
            .optional_features
            .contains(wgpu::Features::POLYGON_MODE_LINE));
        assert!(req
            .optional_features
            .contains(wgpu::Features::TIMESTAMP_QUERY));
    }

    #[test]
    fn test_device_requirements_default() {
        let req = DeviceRequirements::default();
        assert!(req.required_features.is_empty());
    }

    #[test]
    fn test_feature_dependencies_multi_draw() {
        let feature = wgpu::Features::MULTI_DRAW_INDIRECT_COUNT;
        let deps = get_feature_dependencies(feature);
        assert!(deps.contains(wgpu::Features::MULTI_DRAW_INDIRECT));
    }

    #[test]
    fn test_feature_dependencies_texture_array() {
        let feature = wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
        let deps = get_feature_dependencies(feature);
        assert!(deps.contains(wgpu::Features::TEXTURE_BINDING_ARRAY));
    }

    #[test]
    fn test_expand_feature_dependencies() {
        let features = wgpu::Features::MULTI_DRAW_INDIRECT_COUNT;
        let expanded = expand_feature_dependencies(features);

        assert!(expanded.contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT));
        assert!(expanded.contains(wgpu::Features::MULTI_DRAW_INDIRECT));
    }

    #[test]
    fn test_negotiation_result_helpers() {
        let result = NegotiationResult {
            enabled_features: wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::POLYGON_MODE_LINE,
            degraded_features: wgpu::Features::TIMESTAMP_QUERY,
            limits: wgpu::Limits::default(),
        };

        assert!(result.has_feature(wgpu::Features::TEXTURE_COMPRESSION_BC));
        assert!(result.has_feature(wgpu::Features::POLYGON_MODE_LINE));
        assert!(!result.has_feature(wgpu::Features::TIMESTAMP_QUERY));

        assert!(result.was_degraded(wgpu::Features::TIMESTAMP_QUERY));
        assert!(!result.was_degraded(wgpu::Features::TEXTURE_COMPRESSION_BC));

        assert_eq!(result.enabled_count(), 2);
        assert_eq!(result.degraded_count(), 1);
    }

    #[test]
    fn test_error_display_required_features() {
        let err = FeatureNegotiationError::RequiredFeaturesMissing(
            wgpu::Features::TEXTURE_COMPRESSION_BC,
        );
        let msg = format!("{}", err);
        assert!(msg.contains("Required features"));
        assert!(msg.contains("TEXTURE_COMPRESSION_BC"));
    }

    #[test]
    fn test_error_display_limits() {
        let err = FeatureNegotiationError::LimitsExceedCapabilities {
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
    fn test_limit_validation_pass() {
        let required = wgpu::Limits::default();
        let available = wgpu::Limits::default();
        assert!(validate_limits_for_negotiation(&required, &available).is_ok());
    }

    #[test]
    fn test_limit_validation_fail() {
        let mut required = wgpu::Limits::default();
        required.max_texture_dimension_2d = 32768;

        let available = wgpu::Limits::default();

        let result = validate_limits_for_negotiation(&required, &available);
        assert!(result.is_err());

        if let Err(FeatureNegotiationError::LimitsExceedCapabilities { limit, .. }) = result {
            assert_eq!(limit, "max_texture_dimension_2d");
        } else {
            panic!("Expected LimitsExceedCapabilities error");
        }
    }

    #[test]
    fn test_presets() {
        // Test that presets don't panic
        let _minimal = DeviceRequirements::minimal();
        let _standard = DeviceRequirements::standard();
        let _advanced = DeviceRequirements::advanced();

        // Verify minimal has no required features
        let minimal = DeviceRequirements::minimal();
        assert!(minimal.required_features.is_empty());

        // Verify advanced has push constants required
        let advanced = DeviceRequirements::advanced();
        assert!(advanced
            .required_features
            .contains(wgpu::Features::PUSH_CONSTANTS));
    }

    #[test]
    fn test_requirements_display() {
        let req = DeviceRequirements::new()
            .require(wgpu::Features::TEXTURE_COMPRESSION_BC)
            .prefer(wgpu::Features::POLYGON_MODE_LINE);

        let display = format!("{}", req);
        assert!(display.contains("DeviceRequirements"));
        assert!(display.contains("Required features"));
        assert!(display.contains("Optional features"));
    }

    #[test]
    fn test_negotiation_result_display() {
        let result = NegotiationResult {
            enabled_features: wgpu::Features::TEXTURE_COMPRESSION_BC,
            degraded_features: wgpu::Features::TIMESTAMP_QUERY,
            limits: wgpu::Limits::default(),
        };

        let display = format!("{}", result);
        assert!(display.contains("NegotiationResult"));
        assert!(display.contains("Enabled"));
        assert!(display.contains("Degraded"));
    }
}
