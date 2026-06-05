//! Limit negotiation for TRINITY device requirements.
//!
//! This module provides limit negotiation between TRINITY's baseline requirements
//! and adapter capabilities. Unlike feature negotiation (which is pass/fail),
//! limit negotiation can cap preferred limits to what the adapter provides while
//! enforcing TRINITY's minimum requirements.
//!
//! # Overview
//!
//! Limit negotiation follows a three-tier approach:
//!
//! 1. **TRINITY Minimum** - Baseline limits that MUST be met (fail if not)
//! 2. **Preferred** - Limits we'd like to have (capped to adapter if unavailable)
//! 3. **Adapter** - What the hardware actually supports
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::device::{LimitRequirements, negotiate_limits};
//!
//! # fn example(adapter: &wgpu::Adapter) -> Result<(), Box<dyn std::error::Error>> {
//! let requirements = LimitRequirements::new()
//!     .with_trinity_baseline()
//!     .with_preferred(wgpu::Limits {
//!         max_texture_dimension_2d: 16384,
//!         max_storage_buffer_binding_size: 1 << 30, // 1GB
//!         ..wgpu::Limits::default()
//!     });
//!
//! let result = negotiate_limits(&requirements, adapter)?;
//!
//! if result.had_shortfall {
//!     println!("Some limits were capped: {:?}", result.capped_limits);
//! }
//! println!("Final limits: {:?}", result.limits);
//! # Ok(())
//! # }
//! ```

use log::{debug, info, warn};
use std::fmt;

// ============================================================================
// TRINITY Minimum Limits
// ============================================================================

/// TRINITY's baseline minimum required limits.
///
/// These are the absolute minimum limits that TRINITY requires to function.
/// If an adapter cannot meet these limits, device creation will fail.
///
/// # Limits
///
/// | Limit | Value | Rationale |
/// |-------|-------|-----------|
/// | `max_uniform_buffer_binding_size` | 64KB | Standard UBO size for transforms/materials |
/// | `max_storage_buffer_binding_size` | 128MB | GPU-driven rendering scene data |
/// | `max_texture_dimension_2d` | 8192 | 8K texture support |
/// | `max_bind_groups` | 4 | Frame/Material/Object/Dynamic binding model |
/// | `max_bindings_per_bind_group` | 640 | Bindless textures support |
/// | `max_compute_workgroup_size_x` | 256 | Standard compute workgroup |
/// | `max_buffer_size` | 256MB | Large mesh/texture staging |
///
/// # Example
///
/// ```
/// use renderer_backend::device::TrinityMinimumLimits;
///
/// let baseline = TrinityMinimumLimits::baseline();
/// assert_eq!(baseline.min_uniform_buffer_binding_size, 65536); // 64KB
/// assert_eq!(baseline.min_storage_buffer_max_binding_size, 134217728); // 128MB
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrinityMinimumLimits {
    /// Minimum uniform buffer binding size (64KB).
    pub min_uniform_buffer_binding_size: u32,
    /// Minimum storage buffer max binding size (128MB).
    pub min_storage_buffer_max_binding_size: u32,
    /// Minimum 2D texture dimension (8192 = 8K).
    pub min_max_texture_dimension_2d: u32,
    /// Minimum bind groups (4).
    pub min_max_bind_groups: u32,
    /// Minimum bindings per bind group (640).
    pub min_max_bindings_per_bind_group: u32,
    /// Minimum compute workgroup size X (256).
    pub min_max_compute_workgroup_size_x: u32,
    /// Minimum compute workgroup size Y (256).
    pub min_max_compute_workgroup_size_y: u32,
    /// Minimum compute workgroup size Z (64).
    pub min_max_compute_workgroup_size_z: u32,
    /// Minimum compute invocations per workgroup (256).
    pub min_max_compute_invocations_per_workgroup: u32,
    /// Minimum buffer size (256MB).
    pub min_max_buffer_size: u64,
    /// Minimum 1D texture dimension (8192).
    pub min_max_texture_dimension_1d: u32,
    /// Minimum 3D texture dimension (2048).
    pub min_max_texture_dimension_3d: u32,
    /// Minimum texture array layers (256).
    pub min_max_texture_array_layers: u32,
    /// Minimum vertex buffers (8).
    pub min_max_vertex_buffers: u32,
    /// Minimum vertex attributes (16).
    pub min_max_vertex_attributes: u32,
    /// Minimum color attachments (8).
    pub min_max_color_attachments: u32,
}

impl TrinityMinimumLimits {
    /// Get TRINITY's baseline minimum limits.
    ///
    /// These represent the minimum hardware capabilities required for TRINITY
    /// to function. Any adapter that doesn't meet these limits cannot be used.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::device::TrinityMinimumLimits;
    ///
    /// let baseline = TrinityMinimumLimits::baseline();
    /// println!("TRINITY requires at least {}KB uniform buffers",
    ///          baseline.min_uniform_buffer_binding_size / 1024);
    /// ```
    #[must_use]
    pub const fn baseline() -> Self {
        Self {
            // Core rendering requirements
            min_uniform_buffer_binding_size: 65536,          // 64KB
            min_storage_buffer_max_binding_size: 134217728,  // 128MB
            min_max_texture_dimension_2d: 8192,              // 8K

            // Binding model requirements
            min_max_bind_groups: 4,
            min_max_bindings_per_bind_group: 640,

            // Compute requirements
            min_max_compute_workgroup_size_x: 256,
            min_max_compute_workgroup_size_y: 256,
            min_max_compute_workgroup_size_z: 64,
            min_max_compute_invocations_per_workgroup: 256,

            // Buffer requirements
            min_max_buffer_size: 268435456,  // 256MB

            // Texture requirements
            min_max_texture_dimension_1d: 8192,
            min_max_texture_dimension_3d: 2048,
            min_max_texture_array_layers: 256,

            // Vertex requirements
            min_max_vertex_buffers: 8,
            min_max_vertex_attributes: 16,

            // Attachment requirements
            min_max_color_attachments: 8,
        }
    }

    /// Convert to wgpu::Limits.
    ///
    /// Creates a `wgpu::Limits` instance with TRINITY's minimum values.
    /// Other limits are set to wgpu defaults.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::device::TrinityMinimumLimits;
    ///
    /// let baseline = TrinityMinimumLimits::baseline();
    /// let limits = baseline.to_wgpu_limits();
    /// assert_eq!(limits.max_texture_dimension_2d, 8192);
    /// ```
    #[must_use]
    pub fn to_wgpu_limits(&self) -> wgpu::Limits {
        wgpu::Limits {
            max_uniform_buffer_binding_size: self.min_uniform_buffer_binding_size,
            max_storage_buffer_binding_size: self.min_storage_buffer_max_binding_size,
            max_texture_dimension_2d: self.min_max_texture_dimension_2d,
            max_bind_groups: self.min_max_bind_groups,
            max_bindings_per_bind_group: self.min_max_bindings_per_bind_group,
            max_compute_workgroup_size_x: self.min_max_compute_workgroup_size_x,
            max_compute_workgroup_size_y: self.min_max_compute_workgroup_size_y,
            max_compute_workgroup_size_z: self.min_max_compute_workgroup_size_z,
            max_compute_invocations_per_workgroup: self.min_max_compute_invocations_per_workgroup,
            max_buffer_size: self.min_max_buffer_size,
            max_texture_dimension_1d: self.min_max_texture_dimension_1d,
            max_texture_dimension_3d: self.min_max_texture_dimension_3d,
            max_texture_array_layers: self.min_max_texture_array_layers,
            max_vertex_buffers: self.min_max_vertex_buffers,
            max_vertex_attributes: self.min_max_vertex_attributes,
            max_color_attachments: self.min_max_color_attachments,
            ..wgpu::Limits::default()
        }
    }
}

impl Default for TrinityMinimumLimits {
    fn default() -> Self {
        Self::baseline()
    }
}

impl fmt::Display for TrinityMinimumLimits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "TrinityMinimumLimits:")?;
        writeln!(f, "  Uniform buffer: {}KB", self.min_uniform_buffer_binding_size / 1024)?;
        writeln!(f, "  Storage buffer: {}MB", self.min_storage_buffer_max_binding_size / (1024 * 1024))?;
        writeln!(f, "  2D texture: {}px", self.min_max_texture_dimension_2d)?;
        writeln!(f, "  Bind groups: {}", self.min_max_bind_groups)?;
        writeln!(f, "  Bindings/group: {}", self.min_max_bindings_per_bind_group)?;
        writeln!(f, "  Compute workgroup: {}x{}x{}",
                 self.min_max_compute_workgroup_size_x,
                 self.min_max_compute_workgroup_size_y,
                 self.min_max_compute_workgroup_size_z)?;
        writeln!(f, "  Buffer size: {}MB", self.min_max_buffer_size / (1024 * 1024))?;
        Ok(())
    }
}

// ============================================================================
// Limit Requirements
// ============================================================================

/// Requirements for device limits with minimum/preferred semantics.
///
/// This struct allows specifying both minimum acceptable limits (TRINITY baseline)
/// and preferred limits (what we'd like). During negotiation, preferred limits
/// are capped to adapter capabilities, while minimum limits cause failure if not met.
///
/// # Example
///
/// ```
/// use renderer_backend::device::LimitRequirements;
///
/// let requirements = LimitRequirements::new()
///     .with_trinity_baseline()
///     .with_preferred(wgpu::Limits {
///         max_texture_dimension_2d: 16384,
///         ..wgpu::Limits::default()
///     });
/// ```
#[derive(Debug, Clone)]
pub struct LimitRequirements {
    /// Minimum acceptable limits (TRINITY baseline).
    pub minimum: wgpu::Limits,
    /// Preferred limits (what we'd like to have).
    pub preferred: wgpu::Limits,
}

impl Default for LimitRequirements {
    fn default() -> Self {
        Self::new()
    }
}

impl LimitRequirements {
    /// Create new limit requirements with wgpu defaults.
    ///
    /// Both minimum and preferred are set to `wgpu::Limits::default()`.
    /// Use [`with_trinity_baseline`] to set TRINITY minimums.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::device::LimitRequirements;
    ///
    /// let requirements = LimitRequirements::new();
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            minimum: wgpu::Limits::default(),
            preferred: wgpu::Limits::default(),
        }
    }

    /// Use TRINITY baseline as minimum limits.
    ///
    /// This sets the minimum limits to [`TrinityMinimumLimits::baseline()`],
    /// ensuring that negotiation will fail if the adapter doesn't meet
    /// TRINITY's requirements.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::device::LimitRequirements;
    ///
    /// let requirements = LimitRequirements::new()
    ///     .with_trinity_baseline();
    ///
    /// assert_eq!(requirements.minimum.max_texture_dimension_2d, 8192);
    /// ```
    #[must_use]
    pub fn with_trinity_baseline(mut self) -> Self {
        self.minimum = TrinityMinimumLimits::baseline().to_wgpu_limits();
        self
    }

    /// Set custom minimum limits.
    ///
    /// # Arguments
    ///
    /// * `limits` - The minimum limits to require
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::device::LimitRequirements;
    ///
    /// let mut custom_min = wgpu::Limits::default();
    /// custom_min.max_texture_dimension_2d = 4096;
    ///
    /// let requirements = LimitRequirements::new()
    ///     .with_minimum(custom_min);
    /// ```
    #[must_use]
    pub fn with_minimum(mut self, limits: wgpu::Limits) -> Self {
        self.minimum = limits;
        self
    }

    /// Set preferred limits.
    ///
    /// Preferred limits represent what we'd like to have. If the adapter
    /// supports higher limits, we use our preferred values. If the adapter
    /// supports lower limits (but still above minimum), we cap to the adapter.
    ///
    /// # Arguments
    ///
    /// * `limits` - The preferred limits to request
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::device::LimitRequirements;
    ///
    /// let requirements = LimitRequirements::new()
    ///     .with_trinity_baseline()
    ///     .with_preferred(wgpu::Limits {
    ///         max_texture_dimension_2d: 16384,  // Want 16K textures
    ///         max_storage_buffer_binding_size: 1 << 30,  // Want 1GB storage
    ///         ..wgpu::Limits::default()
    ///     });
    /// ```
    #[must_use]
    pub fn with_preferred(mut self, limits: wgpu::Limits) -> Self {
        self.preferred = limits;
        self
    }

    /// Create requirements for standard TRINITY rendering.
    ///
    /// Uses TRINITY baseline as minimum and moderate preferred limits
    /// suitable for most rendering scenarios.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::device::LimitRequirements;
    ///
    /// let requirements = LimitRequirements::standard();
    /// ```
    #[must_use]
    pub fn standard() -> Self {
        Self {
            minimum: TrinityMinimumLimits::baseline().to_wgpu_limits(),
            preferred: wgpu::Limits {
                // Prefer higher limits for better rendering
                max_texture_dimension_2d: 16384,              // 16K
                max_storage_buffer_binding_size: 268435456,   // 256MB
                max_buffer_size: 536870912,                   // 512MB
                max_compute_workgroups_per_dimension: 65535,
                ..TrinityMinimumLimits::baseline().to_wgpu_limits()
            },
        }
    }

    /// Create requirements for high-end TRINITY rendering.
    ///
    /// Uses TRINITY baseline as minimum and high preferred limits
    /// for demanding rendering scenarios (large scenes, high-res textures).
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::device::LimitRequirements;
    ///
    /// let requirements = LimitRequirements::high_end();
    /// ```
    #[must_use]
    pub fn high_end() -> Self {
        Self {
            minimum: TrinityMinimumLimits::baseline().to_wgpu_limits(),
            preferred: wgpu::Limits {
                // Maximum preferred limits for high-end hardware
                max_texture_dimension_2d: 32768,              // 32K (rare but possible)
                max_storage_buffer_binding_size: 1073741824,  // 1GB
                max_buffer_size: 2147483647,                  // ~2GB
                max_compute_workgroups_per_dimension: 65535,
                max_bindings_per_bind_group: 1000,
                ..TrinityMinimumLimits::baseline().to_wgpu_limits()
            },
        }
    }
}

impl fmt::Display for LimitRequirements {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "LimitRequirements:")?;
        writeln!(f, "  Minimum texture 2D: {}px", self.minimum.max_texture_dimension_2d)?;
        writeln!(f, "  Preferred texture 2D: {}px", self.preferred.max_texture_dimension_2d)?;
        writeln!(f, "  Minimum storage buffer: {}MB",
                 self.minimum.max_storage_buffer_binding_size / (1024 * 1024))?;
        writeln!(f, "  Preferred storage buffer: {}MB",
                 self.preferred.max_storage_buffer_binding_size / (1024 * 1024))?;
        Ok(())
    }
}

// ============================================================================
// Negotiation Result
// ============================================================================

/// Result of limit negotiation.
///
/// Contains the final negotiated limits and information about any limits
/// that were capped to adapter capabilities.
///
/// # Example
///
/// ```no_run
/// # use renderer_backend::device::{LimitRequirements, negotiate_limits, LimitNegotiationResult};
/// # fn example(adapter: &wgpu::Adapter, result: &LimitNegotiationResult) {
/// if result.had_shortfall {
///     println!("Capped limits: {:?}", result.capped_limits);
/// }
/// println!("Final max_texture_dimension_2d: {}", result.limits.max_texture_dimension_2d);
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct LimitNegotiationResult {
    /// Final negotiated limits.
    pub limits: wgpu::Limits,
    /// Whether any limits fell short of preferred.
    pub had_shortfall: bool,
    /// Names of limits that were capped (couldn't meet preferred).
    pub capped_limits: Vec<String>,
}

impl LimitNegotiationResult {
    /// Check if a specific limit was capped.
    ///
    /// # Arguments
    ///
    /// * `limit_name` - The name of the limit to check
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use renderer_backend::device::LimitNegotiationResult;
    /// # fn example(result: &LimitNegotiationResult) {
    /// if result.was_capped("max_texture_dimension_2d") {
    ///     println!("Texture size was limited by adapter");
    /// }
    /// # }
    /// ```
    pub fn was_capped(&self, limit_name: &str) -> bool {
        self.capped_limits.iter().any(|s| s == limit_name)
    }

    /// Get the count of capped limits.
    #[inline]
    pub fn capped_count(&self) -> usize {
        self.capped_limits.len()
    }
}

impl fmt::Display for LimitNegotiationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "LimitNegotiationResult:")?;
        writeln!(f, "  Had shortfall: {}", self.had_shortfall)?;
        if !self.capped_limits.is_empty() {
            writeln!(f, "  Capped limits ({}):", self.capped_limits.len())?;
            for limit in &self.capped_limits {
                writeln!(f, "    - {}", limit)?;
            }
        }
        writeln!(f, "  Final limits:")?;
        writeln!(f, "    max_texture_dimension_2d: {}", self.limits.max_texture_dimension_2d)?;
        writeln!(f, "    max_storage_buffer_binding_size: {}", self.limits.max_storage_buffer_binding_size)?;
        writeln!(f, "    max_buffer_size: {}", self.limits.max_buffer_size)?;
        Ok(())
    }
}

// ============================================================================
// Negotiation Error
// ============================================================================

/// Errors that can occur during limit negotiation.
#[derive(Debug, Clone)]
pub enum LimitNegotiationError {
    /// Adapter doesn't meet TRINITY minimum requirements.
    BelowMinimum {
        /// Name of the limit that failed.
        limit: String,
        /// The minimum value required.
        required: u64,
        /// The value the adapter provides.
        available: u64,
    },
}

impl fmt::Display for LimitNegotiationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LimitNegotiationError::BelowMinimum { limit, required, available } => {
                write!(
                    f,
                    "Adapter limit '{}' below TRINITY minimum: required {}, available {}",
                    limit, required, available
                )
            }
        }
    }
}

impl std::error::Error for LimitNegotiationError {}

// ============================================================================
// Negotiation Function
// ============================================================================

/// Negotiate limits between requirements and adapter capabilities.
///
/// This function performs limit negotiation following TRINITY's three-tier approach:
///
/// 1. Check adapter meets all TRINITY minimum limits
/// 2. For each limit, use min(preferred, adapter) or adapter if no preference
/// 3. Log any limits that were capped
///
/// # Arguments
///
/// * `requirements` - The limit requirements to negotiate
/// * `adapter` - The wgpu adapter to negotiate against
///
/// # Returns
///
/// A `Result` containing either a [`LimitNegotiationResult`] with the negotiated
/// limits, or a [`LimitNegotiationError`] if the adapter doesn't meet minimums.
///
/// # Errors
///
/// - [`LimitNegotiationError::BelowMinimum`] - If the adapter doesn't meet
///   TRINITY's minimum requirements for any limit.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::device::{LimitRequirements, negotiate_limits};
///
/// # fn example(adapter: &wgpu::Adapter) -> Result<(), Box<dyn std::error::Error>> {
/// let requirements = LimitRequirements::new()
///     .with_trinity_baseline()
///     .with_preferred(wgpu::Limits {
///         max_texture_dimension_2d: 16384,
///         ..wgpu::Limits::default()
///     });
///
/// let result = negotiate_limits(&requirements, adapter)?;
/// println!("Negotiated limits: {:?}", result.limits);
/// # Ok(())
/// # }
/// ```
pub fn negotiate_limits(
    requirements: &LimitRequirements,
    adapter: &wgpu::Adapter,
) -> Result<LimitNegotiationResult, LimitNegotiationError> {
    let adapter_info = adapter.get_info();
    let adapter_limits = adapter.limits();

    info!(
        "Limit negotiation with adapter: {} ({:?})",
        adapter_info.name, adapter_info.backend
    );
    debug!("Minimum limits: max_texture_2d={}, max_storage={}",
           requirements.minimum.max_texture_dimension_2d,
           requirements.minimum.max_storage_buffer_binding_size);
    debug!("Preferred limits: max_texture_2d={}, max_storage={}",
           requirements.preferred.max_texture_dimension_2d,
           requirements.preferred.max_storage_buffer_binding_size);

    // Phase 1: Validate adapter meets TRINITY minimums
    validate_minimum_limits(&requirements.minimum, &adapter_limits)?;

    // Phase 2: Negotiate each limit
    let mut capped_limits = Vec::new();
    let mut limits = wgpu::Limits::default();

    // Helper macro to negotiate a single limit
    macro_rules! negotiate_limit {
        ($field:ident, $name:expr) => {{
            let minimum = requirements.minimum.$field;
            let preferred = requirements.preferred.$field;
            let adapter_val = adapter_limits.$field;

            let negotiated = if adapter_val >= preferred {
                // Adapter meets or exceeds preferred, use preferred
                preferred
            } else if adapter_val >= minimum {
                // Adapter meets minimum but not preferred, cap to adapter
                capped_limits.push($name.to_string());
                adapter_val
            } else {
                // Should not reach here - validate_minimum_limits should catch this
                return Err(LimitNegotiationError::BelowMinimum {
                    limit: $name.to_string(),
                    required: minimum as u64,
                    available: adapter_val as u64,
                });
            };

            limits.$field = negotiated;
        }};
    }

    // Negotiate u64 limits separately
    macro_rules! negotiate_limit_u64 {
        ($field:ident, $name:expr) => {{
            let minimum = requirements.minimum.$field;
            let preferred = requirements.preferred.$field;
            let adapter_val = adapter_limits.$field;

            let negotiated = if adapter_val >= preferred {
                preferred
            } else if adapter_val >= minimum {
                capped_limits.push($name.to_string());
                adapter_val
            } else {
                return Err(LimitNegotiationError::BelowMinimum {
                    limit: $name.to_string(),
                    required: minimum,
                    available: adapter_val,
                });
            };

            limits.$field = negotiated;
        }};
    }

    // Texture limits
    negotiate_limit!(max_texture_dimension_1d, "max_texture_dimension_1d");
    negotiate_limit!(max_texture_dimension_2d, "max_texture_dimension_2d");
    negotiate_limit!(max_texture_dimension_3d, "max_texture_dimension_3d");
    negotiate_limit!(max_texture_array_layers, "max_texture_array_layers");

    // Buffer limits
    negotiate_limit_u64!(max_buffer_size, "max_buffer_size");
    negotiate_limit!(max_uniform_buffer_binding_size, "max_uniform_buffer_binding_size");
    negotiate_limit!(max_storage_buffer_binding_size, "max_storage_buffer_binding_size");

    // Bind group limits
    negotiate_limit!(max_bind_groups, "max_bind_groups");
    negotiate_limit!(max_bindings_per_bind_group, "max_bindings_per_bind_group");
    negotiate_limit!(max_dynamic_uniform_buffers_per_pipeline_layout,
                     "max_dynamic_uniform_buffers_per_pipeline_layout");
    negotiate_limit!(max_dynamic_storage_buffers_per_pipeline_layout,
                     "max_dynamic_storage_buffers_per_pipeline_layout");
    negotiate_limit!(max_sampled_textures_per_shader_stage,
                     "max_sampled_textures_per_shader_stage");
    negotiate_limit!(max_samplers_per_shader_stage, "max_samplers_per_shader_stage");
    negotiate_limit!(max_storage_buffers_per_shader_stage,
                     "max_storage_buffers_per_shader_stage");
    negotiate_limit!(max_storage_textures_per_shader_stage,
                     "max_storage_textures_per_shader_stage");
    negotiate_limit!(max_uniform_buffers_per_shader_stage,
                     "max_uniform_buffers_per_shader_stage");

    // Compute limits
    negotiate_limit!(max_compute_workgroup_storage_size, "max_compute_workgroup_storage_size");
    negotiate_limit!(max_compute_invocations_per_workgroup,
                     "max_compute_invocations_per_workgroup");
    negotiate_limit!(max_compute_workgroup_size_x, "max_compute_workgroup_size_x");
    negotiate_limit!(max_compute_workgroup_size_y, "max_compute_workgroup_size_y");
    negotiate_limit!(max_compute_workgroup_size_z, "max_compute_workgroup_size_z");
    negotiate_limit!(max_compute_workgroups_per_dimension,
                     "max_compute_workgroups_per_dimension");

    // Vertex limits
    negotiate_limit!(max_vertex_buffers, "max_vertex_buffers");
    negotiate_limit!(max_vertex_attributes, "max_vertex_attributes");
    negotiate_limit!(max_vertex_buffer_array_stride, "max_vertex_buffer_array_stride");

    // Color attachment limits
    negotiate_limit!(max_color_attachments, "max_color_attachments");
    negotiate_limit!(max_color_attachment_bytes_per_sample,
                     "max_color_attachment_bytes_per_sample");

    // Inter-stage limits
    negotiate_limit!(max_inter_stage_shader_components, "max_inter_stage_shader_components");

    // Push constant limit (if supported)
    negotiate_limit!(max_push_constant_size, "max_push_constant_size");

    // Log results
    let had_shortfall = !capped_limits.is_empty();
    if had_shortfall {
        warn!(
            "Limits capped to adapter capabilities ({}): {:?}",
            capped_limits.len(),
            capped_limits
        );
    }
    info!(
        "Limit negotiation complete: {} capped, final max_texture_2d={}",
        capped_limits.len(),
        limits.max_texture_dimension_2d
    );

    Ok(LimitNegotiationResult {
        limits,
        had_shortfall,
        capped_limits,
    })
}

/// Validate that adapter limits meet TRINITY minimum requirements.
fn validate_minimum_limits(
    minimum: &wgpu::Limits,
    adapter: &wgpu::Limits,
) -> Result<(), LimitNegotiationError> {
    // Helper macro for limit validation
    macro_rules! check_limit {
        ($field:ident, $name:expr) => {
            if adapter.$field < minimum.$field {
                return Err(LimitNegotiationError::BelowMinimum {
                    limit: $name.to_string(),
                    required: minimum.$field as u64,
                    available: adapter.$field as u64,
                });
            }
        };
    }

    macro_rules! check_limit_u64 {
        ($field:ident, $name:expr) => {
            if adapter.$field < minimum.$field {
                return Err(LimitNegotiationError::BelowMinimum {
                    limit: $name.to_string(),
                    required: minimum.$field,
                    available: adapter.$field,
                });
            }
        };
    }

    // Texture limits
    check_limit!(max_texture_dimension_1d, "max_texture_dimension_1d");
    check_limit!(max_texture_dimension_2d, "max_texture_dimension_2d");
    check_limit!(max_texture_dimension_3d, "max_texture_dimension_3d");
    check_limit!(max_texture_array_layers, "max_texture_array_layers");

    // Buffer limits
    check_limit_u64!(max_buffer_size, "max_buffer_size");
    check_limit!(max_uniform_buffer_binding_size, "max_uniform_buffer_binding_size");
    check_limit!(max_storage_buffer_binding_size, "max_storage_buffer_binding_size");

    // Bind group limits
    check_limit!(max_bind_groups, "max_bind_groups");
    check_limit!(max_bindings_per_bind_group, "max_bindings_per_bind_group");

    // Compute limits
    check_limit!(max_compute_workgroup_size_x, "max_compute_workgroup_size_x");
    check_limit!(max_compute_workgroup_size_y, "max_compute_workgroup_size_y");
    check_limit!(max_compute_workgroup_size_z, "max_compute_workgroup_size_z");
    check_limit!(max_compute_invocations_per_workgroup,
                 "max_compute_invocations_per_workgroup");

    // Vertex limits
    check_limit!(max_vertex_buffers, "max_vertex_buffers");
    check_limit!(max_vertex_attributes, "max_vertex_attributes");

    // Color attachment limits
    check_limit!(max_color_attachments, "max_color_attachments");

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trinity_minimum_limits_baseline() {
        let baseline = TrinityMinimumLimits::baseline();

        // Verify key TRINITY requirements
        assert_eq!(baseline.min_uniform_buffer_binding_size, 65536);  // 64KB
        assert_eq!(baseline.min_storage_buffer_max_binding_size, 134217728);  // 128MB
        assert_eq!(baseline.min_max_texture_dimension_2d, 8192);  // 8K
        assert_eq!(baseline.min_max_bind_groups, 4);
        assert_eq!(baseline.min_max_bindings_per_bind_group, 640);
    }

    #[test]
    fn test_trinity_minimum_limits_to_wgpu() {
        let baseline = TrinityMinimumLimits::baseline();
        let limits = baseline.to_wgpu_limits();

        assert_eq!(limits.max_uniform_buffer_binding_size, 65536);
        assert_eq!(limits.max_storage_buffer_binding_size, 134217728);
        assert_eq!(limits.max_texture_dimension_2d, 8192);
    }

    #[test]
    fn test_limit_requirements_new() {
        let req = LimitRequirements::new();
        // Should be wgpu defaults
        assert_eq!(req.minimum.max_texture_dimension_2d, wgpu::Limits::default().max_texture_dimension_2d);
    }

    #[test]
    fn test_limit_requirements_with_trinity_baseline() {
        let req = LimitRequirements::new().with_trinity_baseline();

        // Should have TRINITY minimums
        assert_eq!(req.minimum.max_texture_dimension_2d, 8192);
        assert_eq!(req.minimum.max_uniform_buffer_binding_size, 65536);
        assert_eq!(req.minimum.max_storage_buffer_binding_size, 134217728);
    }

    #[test]
    fn test_limit_requirements_with_preferred() {
        let req = LimitRequirements::new()
            .with_trinity_baseline()
            .with_preferred(wgpu::Limits {
                max_texture_dimension_2d: 16384,
                ..wgpu::Limits::default()
            });

        // Minimum should still be TRINITY baseline
        assert_eq!(req.minimum.max_texture_dimension_2d, 8192);
        // Preferred should be what we set
        assert_eq!(req.preferred.max_texture_dimension_2d, 16384);
    }

    #[test]
    fn test_limit_requirements_standard() {
        let req = LimitRequirements::standard();

        // Minimum should be TRINITY baseline
        assert_eq!(req.minimum.max_texture_dimension_2d, 8192);
        // Preferred should be higher
        assert!(req.preferred.max_texture_dimension_2d >= req.minimum.max_texture_dimension_2d);
    }

    #[test]
    fn test_limit_requirements_high_end() {
        let req = LimitRequirements::high_end();

        // Should have high preferred values
        assert!(req.preferred.max_texture_dimension_2d > req.minimum.max_texture_dimension_2d);
        assert!(req.preferred.max_storage_buffer_binding_size > req.minimum.max_storage_buffer_binding_size);
    }

    #[test]
    fn test_limit_negotiation_result_was_capped() {
        let result = LimitNegotiationResult {
            limits: wgpu::Limits::default(),
            had_shortfall: true,
            capped_limits: vec![
                "max_texture_dimension_2d".to_string(),
                "max_storage_buffer_binding_size".to_string(),
            ],
        };

        assert!(result.was_capped("max_texture_dimension_2d"));
        assert!(result.was_capped("max_storage_buffer_binding_size"));
        assert!(!result.was_capped("max_buffer_size"));
        assert_eq!(result.capped_count(), 2);
    }

    #[test]
    fn test_limit_negotiation_error_display() {
        let err = LimitNegotiationError::BelowMinimum {
            limit: "max_texture_dimension_2d".to_string(),
            required: 8192,
            available: 4096,
        };

        let msg = format!("{}", err);
        assert!(msg.contains("max_texture_dimension_2d"));
        assert!(msg.contains("8192"));
        assert!(msg.contains("4096"));
        assert!(msg.contains("below TRINITY minimum"));
    }

    #[test]
    fn test_validate_minimum_limits_pass() {
        let minimum = TrinityMinimumLimits::baseline().to_wgpu_limits();
        let mut adapter = wgpu::Limits::default();

        // Set adapter limits to meet or exceed minimum
        adapter.max_texture_dimension_2d = 16384;
        adapter.max_uniform_buffer_binding_size = 65536;
        adapter.max_storage_buffer_binding_size = 268435456;
        adapter.max_bind_groups = 8;
        adapter.max_bindings_per_bind_group = 1000;
        adapter.max_buffer_size = 1073741824;
        adapter.max_texture_dimension_1d = 16384;
        adapter.max_texture_dimension_3d = 2048;
        adapter.max_texture_array_layers = 2048;
        adapter.max_compute_workgroup_size_x = 1024;
        adapter.max_compute_workgroup_size_y = 1024;
        adapter.max_compute_workgroup_size_z = 64;
        adapter.max_compute_invocations_per_workgroup = 1024;
        adapter.max_vertex_buffers = 16;
        adapter.max_vertex_attributes = 32;
        adapter.max_color_attachments = 8;

        let result = validate_minimum_limits(&minimum, &adapter);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_minimum_limits_fail_texture() {
        let minimum = TrinityMinimumLimits::baseline().to_wgpu_limits();
        let mut adapter = wgpu::Limits::default();

        // Set texture below minimum
        adapter.max_texture_dimension_2d = 4096;  // Below 8192 minimum

        let result = validate_minimum_limits(&minimum, &adapter);
        assert!(result.is_err());

        if let Err(LimitNegotiationError::BelowMinimum { limit, required, available }) = result {
            assert_eq!(limit, "max_texture_dimension_2d");
            assert_eq!(required, 8192);
            assert_eq!(available, 4096);
        } else {
            panic!("Expected BelowMinimum error");
        }
    }

    #[test]
    fn test_display_implementations() {
        // Test TrinityMinimumLimits display
        let baseline = TrinityMinimumLimits::baseline();
        let display = format!("{}", baseline);
        assert!(display.contains("TrinityMinimumLimits"));
        assert!(display.contains("64KB"));
        assert!(display.contains("128MB"));

        // Test LimitRequirements display
        let req = LimitRequirements::standard();
        let display = format!("{}", req);
        assert!(display.contains("LimitRequirements"));

        // Test LimitNegotiationResult display
        let result = LimitNegotiationResult {
            limits: wgpu::Limits::default(),
            had_shortfall: true,
            capped_limits: vec!["max_texture_dimension_2d".to_string()],
        };
        let display = format!("{}", result);
        assert!(display.contains("LimitNegotiationResult"));
        assert!(display.contains("shortfall: true"));
    }
}
