//! D3D12-specific feature detection for TRINITY.
//!
//! This module provides detection of DirectX 12 features and capabilities that
//! go beyond what wgpu exposes directly. D3D12's feature set varies significantly
//! based on feature level, shader model, and hardware tier, and this module helps
//! identify available functionality for optimal rendering paths.
//!
//! # D3D12 Feature Levels
//!
//! | Level | Hardware | Key Features |
//! |-------|----------|--------------|
//! | FL_11_0 | DX11 hardware | Base D3D12 compatibility |
//! | FL_11_1 | DX11.1 hardware | Logic ops, UAV formats |
//! | FL_12_0 | Modern GPUs | Resource binding, tiled resources |
//! | FL_12_1 | Ray tracing capable | Conservative raster, VRS Tier 1 |
//! | FL_12_2 | Latest GPUs | Mesh shaders, sampler feedback |
//!
//! # Shader Models
//!
//! | SM | Key Features |
//! |----|--------------|
//! | 5.1 | Dynamic indexing |
//! | 6.0 | Wave intrinsics |
//! | 6.1 | Barycentrics, SV_ViewID |
//! | 6.2 | FP16, denorm modes |
//! | 6.3 | Ray tracing intrinsics |
//! | 6.4 | VRS, integer dot products |
//! | 6.5 | Mesh shaders, sampler feedback |
//! | 6.6 | 64-bit atomics, derivatives |
//! | 6.7 | Advanced texture ops |
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::backend::dx12::{D3D12Features, D3D12FeatureLevel};
//!
//! # async fn example() {
//! let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
//!     backends: wgpu::Backends::DX12,
//!     ..Default::default()
//! });
//!
//! let adapter = instance
//!     .request_adapter(&wgpu::RequestAdapterOptions::default())
//!     .await
//!     .unwrap();
//!
//! let dx12_features = D3D12Features::detect(&adapter);
//!
//! if dx12_features.supports_rt() {
//!     println!("DXR ray tracing available!");
//!     println!("  - Ray tracing tier: {:?}", dx12_features.ray_tracing_tier);
//! }
//!
//! if dx12_features.supports_mesh_shaders() {
//!     println!("Mesh shaders available (SM 6.5+)!");
//! }
//! # }
//! ```

use log::debug;
use wgpu::{Adapter, Features};

// ============================================================================
// Type Aliases for API Compatibility
// ============================================================================

/// Alias for D3D12FeatureLevel (matches task specification naming).
pub type D3DFeatureLevel = D3D12FeatureLevel;

/// Alias for D3D12ShaderModel (matches task specification naming).
pub type ShaderModel = D3D12ShaderModel;

/// Alias for D3D12RayTracingTier (matches task specification naming).
pub type RayTracingTier = D3D12RayTracingTier;

/// Alias for D3D12Features (matches task specification naming).
pub type DX12Capabilities = D3D12Features;

// ============================================================================
// D3D12FeatureLevel
// ============================================================================

/// D3D12 feature level.
///
/// Feature levels define the minimum hardware capabilities required. Higher
/// feature levels unlock more advanced GPU features. D3D12 requires at
/// minimum feature level 11.0.
///
/// # Feature Level Requirements
///
/// | Level | Hardware Examples |
/// |-------|-------------------|
/// | FL_11_0 | GTX 400+, HD 5000+, Intel HD 4000+ |
/// | FL_11_1 | GTX 600+, HD 7000+, Intel HD 4400+ |
/// | FL_12_0 | GTX 900+, RX 400+, Intel Xe |
/// | FL_12_1 | RTX 20+, RX 6000+, Intel Arc |
/// | FL_12_2 | RTX 30+, RX 6000+, Intel Arc A |
///
/// # Example
///
/// ```
/// use renderer_backend::backend::dx12::D3D12FeatureLevel;
///
/// let fl = D3D12FeatureLevel::FL_12_1;
/// assert!(fl.supports_ray_tracing());
/// assert!(fl.supports_variable_rate_shading());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum D3D12FeatureLevel {
    /// Feature Level 11.0 - Base D3D12 compatibility.
    ///
    /// Minimum for D3D12. Provides compute shaders, tessellation,
    /// and basic resource binding.
    #[default]
    FL_11_0,

    /// Feature Level 11.1 - Enhanced DX11 features.
    ///
    /// Adds logical blend operations, target-independent rasterization,
    /// and UAV-only rendering.
    FL_11_1,

    /// Feature Level 12.0 - Modern GPU baseline.
    ///
    /// Adds typed UAV loads, resource binding tier 2,
    /// tiled resources tier 2, and conservative rasterization tier 1.
    FL_12_0,

    /// Feature Level 12.1 - Ray tracing capable.
    ///
    /// Adds conservative rasterization tier 2/3, variable rate shading
    /// tier 1, and rasterizer ordered views tier 3.
    FL_12_1,

    /// Feature Level 12.2 - Latest features.
    ///
    /// Adds mesh shaders, sampler feedback, and enhanced ray tracing.
    /// Requires Shader Model 6.5+.
    FL_12_2,
}

impl D3D12FeatureLevel {
    /// Detect feature level from a wgpu adapter.
    ///
    /// Since wgpu doesn't expose feature levels directly, this infers
    /// the level from available features.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The wgpu adapter to query (should be a D3D12 adapter)
    ///
    /// # Returns
    ///
    /// The detected feature level.
    pub fn from_adapter(adapter: &Adapter) -> Self {
        let features = adapter.features();
        Self::from_features(features)
    }

    /// Infer feature level from wgpu features.
    ///
    /// # Arguments
    ///
    /// * `features` - The wgpu features from the adapter
    ///
    /// # Returns
    ///
    /// The inferred feature level.
    pub fn from_features(features: Features) -> Self {
        // Check for FL 12.2 indicators (mesh shaders, sampler feedback)
        // wgpu doesn't expose mesh shaders yet, but we can check for
        // ray tracing + advanced features as a proxy
        let has_rt = features.contains(Features::RAY_TRACING_ACCELERATION_STRUCTURE);
        let has_ray_query = features.contains(Features::RAY_QUERY);

        // FL 12.2 requires mesh shaders which wgpu doesn't expose yet
        // Use ray query + ray tracing as a heuristic for FL 12.2
        if has_rt && has_ray_query {
            return D3D12FeatureLevel::FL_12_2;
        }

        // Check for FL 12.1 indicators (ray tracing tier 1.0)
        if has_rt {
            return D3D12FeatureLevel::FL_12_1;
        }

        // Check for FL 12.0 indicators (bindless, typed UAV loads)
        let has_bindless = features.contains(Features::TEXTURE_BINDING_ARRAY)
            && features.contains(Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING);

        if has_bindless {
            return D3D12FeatureLevel::FL_12_0;
        }

        // Check for FL 11.1 indicators
        let has_timestamp = features.contains(Features::TIMESTAMP_QUERY);
        if has_timestamp {
            return D3D12FeatureLevel::FL_11_1;
        }

        D3D12FeatureLevel::FL_11_0
    }

    /// Check if this feature level supports ray tracing.
    ///
    /// Ray tracing requires Feature Level 12.1 or higher.
    ///
    /// # Returns
    ///
    /// `true` if ray tracing is supported at this feature level.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::dx12::D3D12FeatureLevel;
    ///
    /// assert!(D3D12FeatureLevel::FL_12_1.supports_ray_tracing());
    /// assert!(D3D12FeatureLevel::FL_12_2.supports_ray_tracing());
    /// assert!(!D3D12FeatureLevel::FL_12_0.supports_ray_tracing());
    /// ```
    #[inline]
    pub const fn supports_ray_tracing(&self) -> bool {
        matches!(self, D3D12FeatureLevel::FL_12_1 | D3D12FeatureLevel::FL_12_2)
    }

    /// Check if this feature level supports mesh shaders.
    ///
    /// Mesh shaders require Feature Level 12.1+ with Shader Model 6.5.
    /// In practice, this means FL 12.2 for guaranteed support.
    ///
    /// # Returns
    ///
    /// `true` if mesh shaders are supported at this feature level.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::dx12::D3D12FeatureLevel;
    ///
    /// assert!(D3D12FeatureLevel::FL_12_2.supports_mesh_shaders());
    /// assert!(!D3D12FeatureLevel::FL_12_1.supports_mesh_shaders());
    /// ```
    #[inline]
    pub const fn supports_mesh_shaders(&self) -> bool {
        // Mesh shaders require FL 12.1+ with SM 6.5
        // FL 12.2 guarantees SM 6.5
        matches!(self, D3D12FeatureLevel::FL_12_2)
    }

    /// Check if this feature level supports variable rate shading.
    ///
    /// VRS Tier 1 requires Feature Level 12.1+.
    ///
    /// # Returns
    ///
    /// `true` if VRS is supported at this feature level.
    #[inline]
    pub const fn supports_variable_rate_shading(&self) -> bool {
        matches!(self, D3D12FeatureLevel::FL_12_1 | D3D12FeatureLevel::FL_12_2)
    }

    /// Check if this feature level supports sampler feedback.
    ///
    /// Sampler feedback requires Feature Level 12.2.
    ///
    /// # Returns
    ///
    /// `true` if sampler feedback is supported.
    #[inline]
    pub const fn supports_sampler_feedback(&self) -> bool {
        matches!(self, D3D12FeatureLevel::FL_12_2)
    }

    /// Get the feature level name as a string.
    ///
    /// # Returns
    ///
    /// A static string with the feature level name.
    #[inline]
    pub const fn name(&self) -> &'static str {
        match self {
            D3D12FeatureLevel::FL_11_0 => "11.0",
            D3D12FeatureLevel::FL_11_1 => "11.1",
            D3D12FeatureLevel::FL_12_0 => "12.0",
            D3D12FeatureLevel::FL_12_1 => "12.1",
            D3D12FeatureLevel::FL_12_2 => "12.2",
        }
    }

    /// Get the D3D_FEATURE_LEVEL enum value.
    ///
    /// # Returns
    ///
    /// The numeric value matching D3D_FEATURE_LEVEL.
    #[inline]
    pub const fn d3d_feature_level_value(&self) -> u32 {
        match self {
            D3D12FeatureLevel::FL_11_0 => 0xb000, // D3D_FEATURE_LEVEL_11_0
            D3D12FeatureLevel::FL_11_1 => 0xb100, // D3D_FEATURE_LEVEL_11_1
            D3D12FeatureLevel::FL_12_0 => 0xc000, // D3D_FEATURE_LEVEL_12_0
            D3D12FeatureLevel::FL_12_1 => 0xc100, // D3D_FEATURE_LEVEL_12_1
            D3D12FeatureLevel::FL_12_2 => 0xc200, // D3D_FEATURE_LEVEL_12_2
        }
    }

    /// Get the minimum shader model required for this feature level.
    ///
    /// Higher feature levels require higher shader models for full functionality.
    ///
    /// # Returns
    ///
    /// The minimum shader model for this feature level.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::dx12::D3D12FeatureLevel;
    ///
    /// assert_eq!(D3D12FeatureLevel::FL_11_0.min_shader_model().name(), "5.1");
    /// assert_eq!(D3D12FeatureLevel::FL_12_2.min_shader_model().name(), "6.5");
    /// ```
    #[inline]
    pub const fn min_shader_model(&self) -> D3D12ShaderModel {
        match self {
            D3D12FeatureLevel::FL_11_0 => D3D12ShaderModel::SM_5_1,
            D3D12FeatureLevel::FL_11_1 => D3D12ShaderModel::SM_5_1,
            D3D12FeatureLevel::FL_12_0 => D3D12ShaderModel::SM_6_0,
            D3D12FeatureLevel::FL_12_1 => D3D12ShaderModel::SM_6_3,
            D3D12FeatureLevel::FL_12_2 => D3D12ShaderModel::SM_6_5,
        }
    }
}

impl std::fmt::Display for D3D12FeatureLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "D3D_FEATURE_LEVEL_{}", self.name().replace('.', "_"))
    }
}

// ============================================================================
// D3D12ShaderModel
// ============================================================================

/// D3D12 shader model version.
///
/// Shader models define the capabilities available in HLSL shaders.
/// Higher shader models enable more advanced shader features.
///
/// # Shader Model Features
///
/// | SM | Key HLSL Features |
/// |----|-------------------|
/// | 5.1 | Dynamic indexing, unbounded arrays |
/// | 6.0 | Wave intrinsics, 64-bit integers |
/// | 6.1 | Barycentrics, SV_ViewID, GetAttributeAtVertex |
/// | 6.2 | FP16, explicit denorm mode |
/// | 6.3 | Ray tracing intrinsics (DXR 1.0) |
/// | 6.4 | VRS, library subobjects, integer dot4 |
/// | 6.5 | Mesh/amplification shaders, sampler feedback |
/// | 6.6 | 64-bit atomics, derivatives in compute |
/// | 6.7 | Advanced texture ops, raw buffer load |
///
/// # Example
///
/// ```
/// use renderer_backend::backend::dx12::D3D12ShaderModel;
///
/// let sm = D3D12ShaderModel::SM_6_5;
/// assert!(sm.supports_mesh_shaders());
/// assert!(sm.supports_wave_intrinsics());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum D3D12ShaderModel {
    /// Shader Model 5.0 - Legacy D3D11 shader model.
    ///
    /// Basic shader functionality without D3D12-specific features.
    /// Limited descriptor indexing compared to SM 5.1.
    SM_5_0,

    /// Shader Model 5.1 - Base D3D12 shader model.
    ///
    /// Provides dynamic indexing and unbounded descriptor arrays.
    #[default]
    SM_5_1,

    /// Shader Model 6.0 - Wave intrinsics.
    ///
    /// Adds wave operations (WaveActiveSum, WaveActiveBallot, etc.),
    /// 64-bit integer support, and wave size query.
    SM_6_0,

    /// Shader Model 6.1 - Barycentrics.
    ///
    /// Adds SV_Barycentrics, SV_ViewID, and GetAttributeAtVertex.
    SM_6_1,

    /// Shader Model 6.2 - FP16.
    ///
    /// Adds native 16-bit float and integer types, and denorm mode control.
    SM_6_2,

    /// Shader Model 6.3 - DXR 1.0.
    ///
    /// Adds ray tracing intrinsics: TraceRay, ReportHit, CallShader, etc.
    SM_6_3,

    /// Shader Model 6.4 - VRS.
    ///
    /// Adds variable rate shading, integer dot products, and library subobjects.
    SM_6_4,

    /// Shader Model 6.5 - Mesh shaders.
    ///
    /// Adds mesh and amplification shaders, sampler feedback, and ray query.
    SM_6_5,

    /// Shader Model 6.6 - 64-bit atomics.
    ///
    /// Adds 64-bit atomics, derivatives in compute, and dynamic resources.
    SM_6_6,

    /// Shader Model 6.7 - Advanced texture ops.
    ///
    /// Adds advanced texture operations, raw buffer load, and relaxed limits.
    SM_6_7,
}

impl D3D12ShaderModel {
    /// Detect the highest supported shader model from a wgpu adapter.
    ///
    /// Since wgpu doesn't expose shader model directly, this infers
    /// the model from available features.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The wgpu adapter to query (should be a D3D12 adapter)
    ///
    /// # Returns
    ///
    /// The detected shader model.
    pub fn from_adapter(adapter: &Adapter) -> Self {
        let features = adapter.features();
        Self::from_features(features)
    }

    /// Infer shader model from wgpu features.
    ///
    /// # Arguments
    ///
    /// * `features` - The wgpu features from the adapter
    ///
    /// # Returns
    ///
    /// The inferred shader model.
    pub fn from_features(features: Features) -> Self {
        let has_rt = features.contains(Features::RAY_TRACING_ACCELERATION_STRUCTURE);
        let has_ray_query = features.contains(Features::RAY_QUERY);
        let has_bindless = features.contains(Features::TEXTURE_BINDING_ARRAY)
            && features.contains(Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING);
        let has_subgroups = features.contains(Features::SUBGROUP);

        // SM 6.5+ indicated by ray query support
        if has_ray_query {
            return D3D12ShaderModel::SM_6_5;
        }

        // SM 6.3+ indicated by ray tracing support
        if has_rt {
            return D3D12ShaderModel::SM_6_3;
        }

        // SM 6.0+ indicated by subgroup (wave) support
        if has_subgroups {
            return D3D12ShaderModel::SM_6_0;
        }

        // SM 5.1 baseline for bindless
        if has_bindless {
            return D3D12ShaderModel::SM_5_1;
        }

        D3D12ShaderModel::SM_5_1
    }

    /// Check if this shader model supports wave intrinsics.
    ///
    /// Wave intrinsics require Shader Model 6.0 or higher.
    ///
    /// # Returns
    ///
    /// `true` if wave intrinsics are available.
    #[inline]
    pub const fn supports_wave_intrinsics(&self) -> bool {
        !matches!(self, D3D12ShaderModel::SM_5_0 | D3D12ShaderModel::SM_5_1)
    }

    /// Check if this shader model supports ray tracing intrinsics.
    ///
    /// Ray tracing requires Shader Model 6.3 or higher (DXR 1.0).
    ///
    /// # Returns
    ///
    /// `true` if ray tracing intrinsics are available.
    #[inline]
    pub const fn supports_raytracing_intrinsics(&self) -> bool {
        matches!(
            self,
            D3D12ShaderModel::SM_6_3
                | D3D12ShaderModel::SM_6_4
                | D3D12ShaderModel::SM_6_5
                | D3D12ShaderModel::SM_6_6
                | D3D12ShaderModel::SM_6_7
        )
    }

    /// Check if this shader model supports ray tracing.
    ///
    /// This is an alias for [`supports_raytracing_intrinsics`](Self::supports_raytracing_intrinsics).
    /// Ray tracing requires Shader Model 6.3+ (DXR 1.0).
    ///
    /// # Returns
    ///
    /// `true` if ray tracing intrinsics are available (SM 6.3+).
    #[inline]
    pub const fn supports_ray_tracing(&self) -> bool {
        self.supports_raytracing_intrinsics()
    }

    /// Check if this shader model supports mesh shaders.
    ///
    /// Mesh and amplification shaders require Shader Model 6.5+.
    ///
    /// # Returns
    ///
    /// `true` if mesh shaders are available.
    #[inline]
    pub const fn supports_mesh_shaders(&self) -> bool {
        matches!(
            self,
            D3D12ShaderModel::SM_6_5 | D3D12ShaderModel::SM_6_6 | D3D12ShaderModel::SM_6_7
        )
    }

    /// Check if this shader model supports compute shader derivatives.
    ///
    /// Derivatives in compute require Shader Model 6.6+.
    ///
    /// # Returns
    ///
    /// `true` if compute derivatives are available.
    #[inline]
    pub const fn supports_derivatives(&self) -> bool {
        matches!(
            self,
            D3D12ShaderModel::SM_6_6 | D3D12ShaderModel::SM_6_7
        )
    }

    /// Check if this shader model supports 16-bit types.
    ///
    /// Native FP16 and INT16 require Shader Model 6.2+.
    ///
    /// # Returns
    ///
    /// `true` if 16-bit types are available.
    #[inline]
    pub const fn supports_16bit_types(&self) -> bool {
        matches!(
            self,
            D3D12ShaderModel::SM_6_2
                | D3D12ShaderModel::SM_6_3
                | D3D12ShaderModel::SM_6_4
                | D3D12ShaderModel::SM_6_5
                | D3D12ShaderModel::SM_6_6
                | D3D12ShaderModel::SM_6_7
        )
    }

    /// Get the shader model name as a string.
    ///
    /// # Returns
    ///
    /// A static string with the shader model name.
    #[inline]
    pub const fn name(&self) -> &'static str {
        match self {
            D3D12ShaderModel::SM_5_0 => "5.0",
            D3D12ShaderModel::SM_5_1 => "5.1",
            D3D12ShaderModel::SM_6_0 => "6.0",
            D3D12ShaderModel::SM_6_1 => "6.1",
            D3D12ShaderModel::SM_6_2 => "6.2",
            D3D12ShaderModel::SM_6_3 => "6.3",
            D3D12ShaderModel::SM_6_4 => "6.4",
            D3D12ShaderModel::SM_6_5 => "6.5",
            D3D12ShaderModel::SM_6_6 => "6.6",
            D3D12ShaderModel::SM_6_7 => "6.7",
        }
    }

    /// Get the major.minor version as a tuple.
    ///
    /// # Returns
    ///
    /// A tuple of (major, minor) version numbers.
    #[inline]
    pub const fn version(&self) -> (u8, u8) {
        match self {
            D3D12ShaderModel::SM_5_0 => (5, 0),
            D3D12ShaderModel::SM_5_1 => (5, 1),
            D3D12ShaderModel::SM_6_0 => (6, 0),
            D3D12ShaderModel::SM_6_1 => (6, 1),
            D3D12ShaderModel::SM_6_2 => (6, 2),
            D3D12ShaderModel::SM_6_3 => (6, 3),
            D3D12ShaderModel::SM_6_4 => (6, 4),
            D3D12ShaderModel::SM_6_5 => (6, 5),
            D3D12ShaderModel::SM_6_6 => (6, 6),
            D3D12ShaderModel::SM_6_7 => (6, 7),
        }
    }
}

impl std::fmt::Display for D3D12ShaderModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SM {}", self.name())
    }
}

// ============================================================================
// D3D12RayTracingTier
// ============================================================================

/// D3D12 ray tracing (DXR) tier.
///
/// DXR capabilities are divided into tiers that define what ray tracing
/// features are available.
///
/// # Tier Features
///
/// | Tier | Features |
/// |------|----------|
/// | None | No ray tracing support |
/// | Tier1_0 | Full DXR 1.0: TraceRay, acceleration structures |
/// | Tier1_1 | DXR 1.1: Inline ray tracing (RayQuery), DispatchRaysIndirect |
///
/// # Example
///
/// ```
/// use renderer_backend::backend::dx12::D3D12RayTracingTier;
///
/// let tier = D3D12RayTracingTier::Tier1_1;
/// assert!(tier.supports_inline_raytracing());
/// assert!(tier.supports_rayquery());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum D3D12RayTracingTier {
    /// No ray tracing support.
    #[default]
    None,

    /// DXR 1.0 - Full ray tracing pipeline.
    ///
    /// Includes acceleration structures, TraceRay, hit/miss/closest-hit shaders,
    /// and ray generation shaders.
    Tier1_0,

    /// DXR 1.1 - Inline ray tracing.
    ///
    /// Adds RayQuery for inline tracing in any shader stage,
    /// DispatchRaysIndirect, and GPU-driven ray dispatch.
    Tier1_1,
}

impl D3D12RayTracingTier {
    /// Detect ray tracing tier from a wgpu adapter.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The wgpu adapter to query (should be a D3D12 adapter)
    ///
    /// # Returns
    ///
    /// The detected ray tracing tier.
    pub fn from_adapter(adapter: &Adapter) -> Self {
        let features = adapter.features();
        Self::from_features(features)
    }

    /// Infer ray tracing tier from wgpu features.
    ///
    /// # Arguments
    ///
    /// * `features` - The wgpu features from the adapter
    ///
    /// # Returns
    ///
    /// The inferred ray tracing tier.
    pub fn from_features(features: Features) -> Self {
        let has_rt = features.contains(Features::RAY_TRACING_ACCELERATION_STRUCTURE);
        let has_ray_query = features.contains(Features::RAY_QUERY);

        if has_rt && has_ray_query {
            D3D12RayTracingTier::Tier1_1
        } else if has_rt {
            D3D12RayTracingTier::Tier1_0
        } else {
            D3D12RayTracingTier::None
        }
    }

    /// Check if inline ray tracing (RayQuery) is supported.
    ///
    /// Inline ray tracing requires Tier 1.1.
    ///
    /// # Returns
    ///
    /// `true` if inline ray tracing is available.
    #[inline]
    pub const fn supports_inline_raytracing(&self) -> bool {
        matches!(self, D3D12RayTracingTier::Tier1_1)
    }

    /// Check if RayQuery is supported.
    ///
    /// RayQuery requires Tier 1.1 (DXR 1.1).
    ///
    /// # Returns
    ///
    /// `true` if RayQuery is available.
    #[inline]
    pub const fn supports_rayquery(&self) -> bool {
        matches!(self, D3D12RayTracingTier::Tier1_1)
    }

    /// Check if any ray tracing is available.
    ///
    /// # Returns
    ///
    /// `true` if at least Tier 1.0 is supported.
    #[inline]
    pub const fn is_available(&self) -> bool {
        !matches!(self, D3D12RayTracingTier::None)
    }

    /// Check if any ray tracing is supported.
    ///
    /// This is an alias for [`is_available`](Self::is_available).
    ///
    /// # Returns
    ///
    /// `true` if at least Tier 1.0 is supported.
    #[inline]
    pub const fn is_supported(&self) -> bool {
        self.is_available()
    }

    /// Get the tier name as a string.
    ///
    /// # Returns
    ///
    /// A static string with the tier name.
    #[inline]
    pub const fn name(&self) -> &'static str {
        match self {
            D3D12RayTracingTier::None => "None",
            D3D12RayTracingTier::Tier1_0 => "DXR 1.0",
            D3D12RayTracingTier::Tier1_1 => "DXR 1.1",
        }
    }
}

impl std::fmt::Display for D3D12RayTracingTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// MeshShaderTier
// ============================================================================

/// D3D12 mesh shader tier.
///
/// Mesh shaders are a modern GPU feature that replaces the traditional
/// vertex/geometry/tessellation pipeline with a more flexible compute-like
/// model.
///
/// # Tier Features
///
/// | Tier | Features |
/// |------|----------|
/// | NotSupported | No mesh shader support |
/// | Tier1 | Full mesh and amplification shader support |
///
/// # Example
///
/// ```
/// use renderer_backend::backend::dx12::MeshShaderTier;
///
/// let tier = MeshShaderTier::Tier1;
/// assert!(tier.is_supported());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum MeshShaderTier {
    /// No mesh shader support.
    #[default]
    NotSupported,

    /// Full mesh shader support (amplification + mesh shaders).
    ///
    /// Requires Feature Level 12.1+ with Shader Model 6.5+.
    Tier1,
}

impl MeshShaderTier {
    /// Detect mesh shader tier from a wgpu adapter.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The wgpu adapter to query (should be a D3D12 adapter)
    ///
    /// # Returns
    ///
    /// The detected mesh shader tier.
    pub fn from_adapter(adapter: &Adapter) -> Self {
        let features = adapter.features();
        Self::from_features(features)
    }

    /// Infer mesh shader tier from wgpu features.
    ///
    /// # Arguments
    ///
    /// * `features` - The wgpu features from the adapter
    ///
    /// # Returns
    ///
    /// The inferred mesh shader tier.
    pub fn from_features(features: Features) -> Self {
        // Mesh shaders require ray query (SM 6.5 indicator) on D3D12
        let has_ray_query = features.contains(Features::RAY_QUERY);
        let has_rt = features.contains(Features::RAY_TRACING_ACCELERATION_STRUCTURE);

        // FL 12.2 with SM 6.5 indicates mesh shader support
        if has_rt && has_ray_query {
            MeshShaderTier::Tier1
        } else {
            MeshShaderTier::NotSupported
        }
    }

    /// Check if mesh shaders are supported.
    ///
    /// # Returns
    ///
    /// `true` if at least Tier 1 is supported.
    #[inline]
    pub const fn is_supported(&self) -> bool {
        matches!(self, MeshShaderTier::Tier1)
    }

    /// Get the tier name as a string.
    ///
    /// # Returns
    ///
    /// A static string with the tier name.
    #[inline]
    pub const fn name(&self) -> &'static str {
        match self {
            MeshShaderTier::NotSupported => "Not Supported",
            MeshShaderTier::Tier1 => "Tier 1",
        }
    }
}

impl std::fmt::Display for MeshShaderTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// ShaderCompiler
// ============================================================================

/// HLSL shader compiler type.
///
/// D3D12 supports two shader compilers with different capabilities.
/// FXC is the legacy compiler limited to SM 5.1, while DXC is the modern
/// compiler supporting SM 6.0+.
///
/// # Compiler Comparison
///
/// | Compiler | Shader Models | Features |
/// |----------|---------------|----------|
/// | FXC | SM 5.0, 5.1 | Legacy HLSL compiler |
/// | DXC | SM 6.0 - 6.7 | Modern DXIL compiler, SPIR-V output |
///
/// # Example
///
/// ```
/// use renderer_backend::backend::dx12::{ShaderCompiler, D3D12ShaderModel};
///
/// let compiler = ShaderCompiler::DXC;
/// assert!(compiler.supports_shader_model(D3D12ShaderModel::SM_6_5));
///
/// let recommended = ShaderCompiler::recommended_for(D3D12ShaderModel::SM_6_3);
/// assert_eq!(recommended, ShaderCompiler::DXC);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ShaderCompiler {
    /// FXC - Legacy HLSL compiler.
    ///
    /// The original HLSL compiler (fxc.exe). Limited to Shader Model 5.1
    /// and below. Outputs DXBC bytecode.
    FXC,

    /// DXC - Modern DirectX shader compiler.
    ///
    /// The modern HLSL compiler (dxc.exe). Required for Shader Model 6.0+.
    /// Outputs DXIL bytecode and can also output SPIR-V.
    #[default]
    DXC,
}

impl ShaderCompiler {
    /// Check if this compiler supports a given shader model.
    ///
    /// # Arguments
    ///
    /// * `model` - The shader model to check
    ///
    /// # Returns
    ///
    /// `true` if this compiler can compile shaders for the given model.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::dx12::{ShaderCompiler, D3D12ShaderModel};
    ///
    /// assert!(ShaderCompiler::FXC.supports_shader_model(D3D12ShaderModel::SM_5_1));
    /// assert!(!ShaderCompiler::FXC.supports_shader_model(D3D12ShaderModel::SM_6_0));
    /// assert!(ShaderCompiler::DXC.supports_shader_model(D3D12ShaderModel::SM_6_5));
    /// ```
    #[inline]
    pub const fn supports_shader_model(&self, model: D3D12ShaderModel) -> bool {
        match self {
            ShaderCompiler::FXC => matches!(model, D3D12ShaderModel::SM_5_0 | D3D12ShaderModel::SM_5_1),
            ShaderCompiler::DXC => true, // DXC supports all shader models
        }
    }

    /// Get the recommended compiler for a shader model.
    ///
    /// # Arguments
    ///
    /// * `model` - The target shader model
    ///
    /// # Returns
    ///
    /// The recommended compiler for the given shader model.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::dx12::{ShaderCompiler, D3D12ShaderModel};
    ///
    /// assert_eq!(ShaderCompiler::recommended_for(D3D12ShaderModel::SM_5_1), ShaderCompiler::FXC);
    /// assert_eq!(ShaderCompiler::recommended_for(D3D12ShaderModel::SM_6_0), ShaderCompiler::DXC);
    /// ```
    #[inline]
    pub const fn recommended_for(model: D3D12ShaderModel) -> Self {
        match model {
            D3D12ShaderModel::SM_5_0 | D3D12ShaderModel::SM_5_1 => ShaderCompiler::FXC,
            _ => ShaderCompiler::DXC,
        }
    }

    /// Get the compiler name as a string.
    ///
    /// # Returns
    ///
    /// A static string with the compiler name.
    #[inline]
    pub const fn name(&self) -> &'static str {
        match self {
            ShaderCompiler::FXC => "FXC",
            ShaderCompiler::DXC => "DXC",
        }
    }

    /// Get the executable name for this compiler.
    ///
    /// # Returns
    ///
    /// The executable name (without extension).
    #[inline]
    pub const fn executable(&self) -> &'static str {
        match self {
            ShaderCompiler::FXC => "fxc",
            ShaderCompiler::DXC => "dxc",
        }
    }

    /// Check if this compiler supports SPIR-V output.
    ///
    /// # Returns
    ///
    /// `true` if the compiler can output SPIR-V (for Vulkan cross-compilation).
    #[inline]
    pub const fn supports_spirv(&self) -> bool {
        matches!(self, ShaderCompiler::DXC)
    }
}

impl std::fmt::Display for ShaderCompiler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// DX12Info
// ============================================================================

/// D3D12 adapter information with capabilities.
///
/// This struct combines hardware information (adapter name, vendor, device)
/// with detected capabilities. It provides a complete picture of a D3D12
/// adapter's features and identity.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::backend::dx12::DX12Info;
///
/// # async fn example() {
/// let instance = wgpu::Instance::default();
/// let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
///
/// let info = DX12Info::from_adapter(&adapter);
/// println!("Adapter: {} (Vendor: {:04x})", info.adapter_name, info.vendor_id);
/// println!("Capabilities: {}", info.capabilities.summary());
///
/// if info.supports_ray_tracing() {
///     println!("Ray tracing supported!");
/// }
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct DX12Info {
    /// Detected D3D12 capabilities.
    pub capabilities: D3D12Features,

    /// Human-readable adapter name.
    pub adapter_name: String,

    /// PCI vendor ID.
    ///
    /// Common values:
    /// - 0x10DE: NVIDIA
    /// - 0x1002: AMD
    /// - 0x8086: Intel
    /// - 0x1414: Microsoft (WARP)
    pub vendor_id: u32,

    /// PCI device ID.
    pub device_id: u32,

    /// Driver version string.
    pub driver_version: String,
}

impl DX12Info {
    /// Create DX12Info from a wgpu adapter.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The wgpu adapter to query (should be a D3D12 adapter)
    ///
    /// # Returns
    ///
    /// The detected DX12 info.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::backend::dx12::DX12Info;
    ///
    /// # async fn example() {
    /// let instance = wgpu::Instance::default();
    /// let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
    ///
    /// let info = DX12Info::from_adapter(&adapter);
    /// println!("GPU: {}", info.adapter_name);
    /// # }
    /// ```
    pub fn from_adapter(adapter: &Adapter) -> Self {
        let info = adapter.get_info();
        let capabilities = D3D12Features::detect(adapter);

        Self {
            capabilities,
            adapter_name: info.name.clone(),
            vendor_id: info.vendor,
            device_id: info.device,
            driver_version: info.driver.clone(),
        }
    }

    /// Check if ray tracing (DXR) is supported.
    ///
    /// # Returns
    ///
    /// `true` if any ray tracing tier is available.
    #[inline]
    pub fn supports_ray_tracing(&self) -> bool {
        self.capabilities.supports_rt()
    }

    /// Get the recommended shader compiler for this adapter's capabilities.
    ///
    /// # Returns
    ///
    /// The recommended shader compiler based on the detected shader model.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::backend::dx12::DX12Info;
    ///
    /// # async fn example() {
    /// # let instance = wgpu::Instance::default();
    /// # let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
    /// let info = DX12Info::from_adapter(&adapter);
    /// let compiler = info.recommended_compiler();
    /// println!("Use {} for this GPU", compiler.name());
    /// # }
    /// ```
    #[inline]
    pub fn recommended_compiler(&self) -> ShaderCompiler {
        ShaderCompiler::recommended_for(self.capabilities.shader_model)
    }

    /// Check if this is an NVIDIA GPU.
    #[inline]
    pub fn is_nvidia(&self) -> bool {
        self.vendor_id == 0x10DE
    }

    /// Check if this is an AMD GPU.
    #[inline]
    pub fn is_amd(&self) -> bool {
        self.vendor_id == 0x1002
    }

    /// Check if this is an Intel GPU.
    #[inline]
    pub fn is_intel(&self) -> bool {
        self.vendor_id == 0x8086
    }

    /// Check if this is the Microsoft WARP software renderer.
    #[inline]
    pub fn is_warp(&self) -> bool {
        self.vendor_id == 0x1414
    }

    /// Get a human-readable vendor name.
    pub fn vendor_name(&self) -> &'static str {
        match self.vendor_id {
            0x10DE => "NVIDIA",
            0x1002 => "AMD",
            0x8086 => "Intel",
            0x1414 => "Microsoft",
            _ => "Unknown",
        }
    }
}

impl Default for DX12Info {
    fn default() -> Self {
        Self {
            capabilities: D3D12Features::default(),
            adapter_name: String::new(),
            vendor_id: 0,
            device_id: 0,
            driver_version: String::new(),
        }
    }
}

impl std::fmt::Display for DX12Info {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} ({}) - {}",
            self.adapter_name,
            self.vendor_name(),
            self.capabilities.summary()
        )
    }
}

// ============================================================================
// D3D12Features
// ============================================================================

/// D3D12-specific feature detection.
///
/// This struct captures DirectX 12 capabilities beyond what wgpu exposes
/// directly. Detection is performed by inspecting wgpu's feature flags
/// and adapter info, mapping them to D3D12-specific functionality.
///
/// # Feature Categories
///
/// | Category | Fields |
/// |----------|--------|
/// | Core | `feature_level`, `shader_model` |
/// | Ray Tracing | `ray_tracing_tier` |
/// | Mesh Shading | `mesh_shader_tier` |
/// | VRS | `variable_rate_shading_tier` |
/// | Resources | `bindless_resources`, `resource_binding_tier` |
/// | Rasterization | `conservative_rasterization_tier` |
///
/// # Example
///
/// ```no_run
/// use renderer_backend::backend::dx12::D3D12Features;
///
/// # async fn example() {
/// let instance = wgpu::Instance::default();
/// let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
///
/// let dx12 = D3D12Features::detect(&adapter);
/// println!("Feature Level: {}", dx12.feature_level);
/// println!("Shader Model: {}", dx12.shader_model);
/// println!("Ray Tracing: {}", dx12.ray_tracing_tier);
/// println!("Bindless: {}", dx12.bindless_resources);
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct D3D12Features {
    /// Detected D3D12 feature level.
    pub feature_level: D3D12FeatureLevel,

    /// Detected shader model.
    pub shader_model: D3D12ShaderModel,

    /// Ray tracing (DXR) tier.
    pub ray_tracing_tier: D3D12RayTracingTier,

    /// Mesh shader tier.
    ///
    /// - 0: No mesh shader support
    /// - 1: Full mesh shader support (amplification + mesh)
    pub mesh_shader_tier: u8,

    /// Variable rate shading tier.
    ///
    /// - 0: No VRS support
    /// - 1: Per-draw VRS
    /// - 2: Per-draw + per-primitive VRS with screen-space image
    pub variable_rate_shading_tier: u8,

    /// Sampler feedback tier.
    ///
    /// - 0: No sampler feedback support
    /// - 1: Sampler feedback with MinMip, used for texture streaming
    pub sampler_feedback_tier: u8,

    /// Bindless resources support.
    ///
    /// True if SM 5.1+ dynamic indexing and unbounded arrays are available.
    pub bindless_resources: bool,

    /// Conservative rasterization tier.
    ///
    /// - 0: No conservative rasterization
    /// - 1: Uncertainty region <= 0.5 pixels
    /// - 2: Tier 1 + post-snap degenerate triangles culled
    /// - 3: Tier 2 + inner coverage, SV_InnerCoverage
    pub conservative_rasterization_tier: u8,

    /// Tiled resources tier.
    ///
    /// - 0: No tiled resource support
    /// - 1: Tiled resources with standard tile shapes
    /// - 2: Tier 1 + clamped LOD
    /// - 3: Tier 2 + Texture3D support
    /// - 4: Tier 3 + full support
    pub tiled_resources_tier: u8,

    /// Resource binding tier.
    ///
    /// - 1: Limited descriptor tables
    /// - 2: Full descriptor tables, CBV/SRV/UAV heap
    /// - 3: Tier 2 + full heap indexing
    pub resource_binding_tier: u8,

    /// Root signature version.
    ///
    /// - 1: Original root signatures
    /// - 2: Static samplers, descriptor range flags
    pub root_signature_version: u8,

    /// Wave operation support (subgroups).
    pub wave_ops: bool,

    /// Native 16-bit shader operations support.
    pub native_16bit_ops: bool,

    /// Raytracing pipeline support (vs. inline only).
    pub rt_pipeline: bool,

    /// Rasterizer ordered views tier.
    ///
    /// - 0: No ROV support
    /// - 1: Limited ROV support
    /// - 2: Full ROV support
    /// - 3: Full ROV + atomic ops
    pub rasterizer_ordered_views_tier: u8,
}

impl D3D12Features {
    /// Detect D3D12 features from a wgpu adapter.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The wgpu adapter to query (should be a D3D12 adapter)
    ///
    /// # Returns
    ///
    /// The detected D3D12 features.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::backend::dx12::D3D12Features;
    ///
    /// # async fn example() {
    /// let instance = wgpu::Instance::default();
    /// let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
    ///
    /// let features = D3D12Features::detect(&adapter);
    /// if features.supports_rt() {
    ///     println!("DXR available!");
    /// }
    /// # }
    /// ```
    pub fn detect(adapter: &Adapter) -> Self {
        let wgpu_features = adapter.features();
        Self::from_features(wgpu_features)
    }

    /// Create D3D12 features from wgpu feature flags.
    ///
    /// # Arguments
    ///
    /// * `features` - The wgpu features from the adapter
    ///
    /// # Returns
    ///
    /// The mapped D3D12 features.
    pub fn from_features(features: Features) -> Self {
        let feature_level = D3D12FeatureLevel::from_features(features);
        let shader_model = D3D12ShaderModel::from_features(features);
        let ray_tracing_tier = D3D12RayTracingTier::from_features(features);

        // Bindless resources detection
        let bindless_resources = features.contains(Features::TEXTURE_BINDING_ARRAY)
            && features.contains(Features::BUFFER_BINDING_ARRAY)
            && features.contains(Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING);

        // Wave ops (subgroups)
        let wave_ops = features.contains(Features::SUBGROUP);

        // Native 16-bit operations
        let native_16bit_ops = features.contains(Features::SHADER_F16);

        // RT pipeline (vs inline only)
        let rt_pipeline = features.contains(Features::RAY_TRACING_ACCELERATION_STRUCTURE);

        // Mesh shader tier - inferred from feature level and shader model
        let mesh_shader_tier = if feature_level >= D3D12FeatureLevel::FL_12_1
            && shader_model >= D3D12ShaderModel::SM_6_5
        {
            1
        } else {
            0
        };

        // VRS tier - inferred from feature level
        let variable_rate_shading_tier = if feature_level >= D3D12FeatureLevel::FL_12_2 {
            2 // Assume tier 2 for FL 12.2
        } else if feature_level >= D3D12FeatureLevel::FL_12_1 {
            1 // Tier 1 for FL 12.1
        } else {
            0
        };

        // Sampler feedback tier - FL 12.2 required
        let sampler_feedback_tier = if feature_level >= D3D12FeatureLevel::FL_12_2 {
            1
        } else {
            0
        };

        // Conservative rasterization tier - inferred from feature level
        let conservative_rasterization_tier = if feature_level >= D3D12FeatureLevel::FL_12_1 {
            3 // Assume tier 3 for FL 12.1+
        } else if feature_level >= D3D12FeatureLevel::FL_12_0 {
            1 // Tier 1 for FL 12.0
        } else {
            0
        };

        // Tiled resources tier - inferred from feature level
        let tiled_resources_tier = if feature_level >= D3D12FeatureLevel::FL_12_2 {
            4
        } else if feature_level >= D3D12FeatureLevel::FL_12_1 {
            3
        } else if feature_level >= D3D12FeatureLevel::FL_12_0 {
            2
        } else if feature_level >= D3D12FeatureLevel::FL_11_1 {
            1
        } else {
            0
        };

        // Resource binding tier - inferred from bindless support
        let resource_binding_tier = if bindless_resources {
            3 // Full heap indexing
        } else if feature_level >= D3D12FeatureLevel::FL_12_0 {
            2
        } else {
            1
        };

        // Root signature version - assume v2 for modern hardware
        let root_signature_version = if feature_level >= D3D12FeatureLevel::FL_12_0 {
            2
        } else {
            1
        };

        // ROV tier - inferred from feature level
        let rasterizer_ordered_views_tier = if feature_level >= D3D12FeatureLevel::FL_12_1 {
            3
        } else if feature_level >= D3D12FeatureLevel::FL_12_0 {
            2
        } else if feature_level >= D3D12FeatureLevel::FL_11_1 {
            1
        } else {
            0
        };

        let result = Self {
            feature_level,
            shader_model,
            ray_tracing_tier,
            mesh_shader_tier,
            variable_rate_shading_tier,
            sampler_feedback_tier,
            bindless_resources,
            conservative_rasterization_tier,
            tiled_resources_tier,
            resource_binding_tier,
            root_signature_version,
            wave_ops,
            native_16bit_ops,
            rt_pipeline,
            rasterizer_ordered_views_tier,
        };

        debug!(
            "D3D12Features detected: FL={}, SM={}, DXR={}, bindless={}",
            feature_level.name(),
            shader_model.name(),
            ray_tracing_tier.name(),
            bindless_resources
        );

        result
    }

    /// Check if ray tracing (DXR) is supported.
    ///
    /// # Returns
    ///
    /// `true` if any ray tracing tier is available.
    #[inline]
    pub const fn supports_rt(&self) -> bool {
        self.ray_tracing_tier.is_available()
    }

    /// Check if mesh shaders are supported.
    ///
    /// Mesh shaders require FL 12.1+ with SM 6.5.
    ///
    /// # Returns
    ///
    /// `true` if mesh shaders are available.
    #[inline]
    pub const fn supports_mesh_shaders(&self) -> bool {
        self.mesh_shader_tier >= 1
    }

    /// Check if bindless resources are supported.
    ///
    /// # Returns
    ///
    /// `true` if bindless (SM 5.1+ dynamic indexing) is available.
    #[inline]
    pub const fn supports_bindless(&self) -> bool {
        self.bindless_resources
    }

    /// Check if variable rate shading is supported.
    ///
    /// # Returns
    ///
    /// `true` if any VRS tier is available.
    #[inline]
    pub const fn supports_vrs(&self) -> bool {
        self.variable_rate_shading_tier >= 1
    }

    /// Check if inline ray tracing (RayQuery) is supported.
    ///
    /// # Returns
    ///
    /// `true` if DXR 1.1 inline ray tracing is available.
    #[inline]
    pub const fn supports_inline_rt(&self) -> bool {
        self.ray_tracing_tier.supports_inline_raytracing()
    }

    /// Check if sampler feedback is supported.
    ///
    /// # Returns
    ///
    /// `true` if sampler feedback is available.
    #[inline]
    pub const fn supports_sampler_feedback(&self) -> bool {
        self.sampler_feedback_tier >= 1
    }

    /// Check if conservative rasterization is supported.
    ///
    /// # Returns
    ///
    /// `true` if any conservative rasterization tier is available.
    #[inline]
    pub const fn supports_conservative_raster(&self) -> bool {
        self.conservative_rasterization_tier >= 1
    }

    /// Check if this meets requirements for GPU-driven rendering.
    ///
    /// GPU-driven rendering requires bindless + wave ops.
    ///
    /// # Returns
    ///
    /// `true` if GPU-driven rendering requirements are met.
    #[inline]
    pub const fn supports_gpu_driven(&self) -> bool {
        self.bindless_resources && self.wave_ops
    }

    /// Create a summary string of detected features.
    ///
    /// # Returns
    ///
    /// A human-readable summary of available features.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::backend::dx12::D3D12Features;
    ///
    /// # async fn example() {
    /// # let instance = wgpu::Instance::default();
    /// # let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
    /// let features = D3D12Features::detect(&adapter);
    /// println!("{}", features.summary());
    /// // Output: "FL 12.1, SM 6.5, DXR 1.1, Mesh, VRS T2, Bindless"
    /// # }
    /// ```
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();

        parts.push(format!("FL {}", self.feature_level.name()));
        parts.push(format!("SM {}", self.shader_model.name()));

        if self.ray_tracing_tier.is_available() {
            parts.push(self.ray_tracing_tier.name().to_string());
        }

        if self.mesh_shader_tier >= 1 {
            parts.push("Mesh".to_string());
        }

        if self.variable_rate_shading_tier >= 1 {
            parts.push(format!("VRS T{}", self.variable_rate_shading_tier));
        }

        if self.bindless_resources {
            parts.push("Bindless".to_string());
        }

        if self.wave_ops {
            parts.push("Wave".to_string());
        }

        if self.native_16bit_ops {
            parts.push("FP16".to_string());
        }

        if parts.is_empty() {
            "Basic D3D12".to_string()
        } else {
            parts.join(", ")
        }
    }

    /// Get the minimum Windows version required for detected features.
    ///
    /// # Returns
    ///
    /// A tuple of (major, minor, build) version numbers.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::backend::dx12::D3D12Features;
    ///
    /// let features = D3D12Features::default();
    /// let (major, minor, build) = features.minimum_windows_version();
    /// println!("Requires Windows {}.{} build {}", major, minor, build);
    /// ```
    #[inline]
    pub const fn minimum_windows_version(&self) -> (u32, u32, u32) {
        // DXR 1.1 requires Windows 10 20H1 (build 19041)
        if self.ray_tracing_tier.supports_inline_raytracing() {
            return (10, 0, 19041);
        }

        // DXR 1.0 requires Windows 10 1809 (build 17763)
        if self.ray_tracing_tier.is_available() {
            return (10, 0, 17763);
        }

        // Mesh shaders require Windows 10 20H1
        if self.mesh_shader_tier >= 1 {
            return (10, 0, 19041);
        }

        // VRS requires Windows 10 1903 (build 18362)
        if self.variable_rate_shading_tier >= 1 {
            return (10, 0, 18362);
        }

        // Base D3D12 requires Windows 10
        (10, 0, 10240)
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // D3D12FeatureLevel Tests
    // ========================================================================

    #[test]
    fn test_feature_level_default() {
        let fl = D3D12FeatureLevel::default();
        assert_eq!(fl, D3D12FeatureLevel::FL_11_0);
    }

    #[test]
    fn test_feature_level_ordering() {
        assert!(D3D12FeatureLevel::FL_11_0 < D3D12FeatureLevel::FL_11_1);
        assert!(D3D12FeatureLevel::FL_11_1 < D3D12FeatureLevel::FL_12_0);
        assert!(D3D12FeatureLevel::FL_12_0 < D3D12FeatureLevel::FL_12_1);
        assert!(D3D12FeatureLevel::FL_12_1 < D3D12FeatureLevel::FL_12_2);
    }

    #[test]
    fn test_feature_level_supports_ray_tracing() {
        assert!(!D3D12FeatureLevel::FL_11_0.supports_ray_tracing());
        assert!(!D3D12FeatureLevel::FL_11_1.supports_ray_tracing());
        assert!(!D3D12FeatureLevel::FL_12_0.supports_ray_tracing());
        assert!(D3D12FeatureLevel::FL_12_1.supports_ray_tracing());
        assert!(D3D12FeatureLevel::FL_12_2.supports_ray_tracing());
    }

    #[test]
    fn test_feature_level_supports_mesh_shaders() {
        assert!(!D3D12FeatureLevel::FL_11_0.supports_mesh_shaders());
        assert!(!D3D12FeatureLevel::FL_12_0.supports_mesh_shaders());
        assert!(!D3D12FeatureLevel::FL_12_1.supports_mesh_shaders());
        assert!(D3D12FeatureLevel::FL_12_2.supports_mesh_shaders());
    }

    #[test]
    fn test_feature_level_supports_vrs() {
        assert!(!D3D12FeatureLevel::FL_11_0.supports_variable_rate_shading());
        assert!(!D3D12FeatureLevel::FL_12_0.supports_variable_rate_shading());
        assert!(D3D12FeatureLevel::FL_12_1.supports_variable_rate_shading());
        assert!(D3D12FeatureLevel::FL_12_2.supports_variable_rate_shading());
    }

    #[test]
    fn test_feature_level_supports_sampler_feedback() {
        assert!(!D3D12FeatureLevel::FL_11_0.supports_sampler_feedback());
        assert!(!D3D12FeatureLevel::FL_12_1.supports_sampler_feedback());
        assert!(D3D12FeatureLevel::FL_12_2.supports_sampler_feedback());
    }

    #[test]
    fn test_feature_level_name() {
        assert_eq!(D3D12FeatureLevel::FL_11_0.name(), "11.0");
        assert_eq!(D3D12FeatureLevel::FL_11_1.name(), "11.1");
        assert_eq!(D3D12FeatureLevel::FL_12_0.name(), "12.0");
        assert_eq!(D3D12FeatureLevel::FL_12_1.name(), "12.1");
        assert_eq!(D3D12FeatureLevel::FL_12_2.name(), "12.2");
    }

    #[test]
    fn test_feature_level_d3d_value() {
        assert_eq!(D3D12FeatureLevel::FL_11_0.d3d_feature_level_value(), 0xb000);
        assert_eq!(D3D12FeatureLevel::FL_11_1.d3d_feature_level_value(), 0xb100);
        assert_eq!(D3D12FeatureLevel::FL_12_0.d3d_feature_level_value(), 0xc000);
        assert_eq!(D3D12FeatureLevel::FL_12_1.d3d_feature_level_value(), 0xc100);
        assert_eq!(D3D12FeatureLevel::FL_12_2.d3d_feature_level_value(), 0xc200);
    }

    #[test]
    fn test_feature_level_display() {
        assert_eq!(
            format!("{}", D3D12FeatureLevel::FL_11_0),
            "D3D_FEATURE_LEVEL_11_0"
        );
        assert_eq!(
            format!("{}", D3D12FeatureLevel::FL_12_2),
            "D3D_FEATURE_LEVEL_12_2"
        );
    }

    #[test]
    fn test_feature_level_from_empty_features() {
        let features = Features::empty();
        let fl = D3D12FeatureLevel::from_features(features);
        assert_eq!(fl, D3D12FeatureLevel::FL_11_0);
    }

    #[test]
    fn test_feature_level_from_rt_features() {
        let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE;
        let fl = D3D12FeatureLevel::from_features(features);
        assert_eq!(fl, D3D12FeatureLevel::FL_12_1);
    }

    #[test]
    fn test_feature_level_from_full_rt_features() {
        let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE | Features::RAY_QUERY;
        let fl = D3D12FeatureLevel::from_features(features);
        assert_eq!(fl, D3D12FeatureLevel::FL_12_2);
    }

    // ========================================================================
    // D3D12ShaderModel Tests
    // ========================================================================

    #[test]
    fn test_shader_model_default() {
        let sm = D3D12ShaderModel::default();
        assert_eq!(sm, D3D12ShaderModel::SM_5_1);
    }

    #[test]
    fn test_shader_model_ordering() {
        assert!(D3D12ShaderModel::SM_5_1 < D3D12ShaderModel::SM_6_0);
        assert!(D3D12ShaderModel::SM_6_0 < D3D12ShaderModel::SM_6_3);
        assert!(D3D12ShaderModel::SM_6_3 < D3D12ShaderModel::SM_6_5);
        assert!(D3D12ShaderModel::SM_6_5 < D3D12ShaderModel::SM_6_7);
    }

    #[test]
    fn test_shader_model_supports_wave_intrinsics() {
        assert!(!D3D12ShaderModel::SM_5_1.supports_wave_intrinsics());
        assert!(D3D12ShaderModel::SM_6_0.supports_wave_intrinsics());
        assert!(D3D12ShaderModel::SM_6_5.supports_wave_intrinsics());
    }

    #[test]
    fn test_shader_model_supports_raytracing_intrinsics() {
        assert!(!D3D12ShaderModel::SM_5_1.supports_raytracing_intrinsics());
        assert!(!D3D12ShaderModel::SM_6_0.supports_raytracing_intrinsics());
        assert!(!D3D12ShaderModel::SM_6_2.supports_raytracing_intrinsics());
        assert!(D3D12ShaderModel::SM_6_3.supports_raytracing_intrinsics());
        assert!(D3D12ShaderModel::SM_6_5.supports_raytracing_intrinsics());
    }

    #[test]
    fn test_shader_model_supports_mesh_shaders() {
        assert!(!D3D12ShaderModel::SM_5_1.supports_mesh_shaders());
        assert!(!D3D12ShaderModel::SM_6_3.supports_mesh_shaders());
        assert!(!D3D12ShaderModel::SM_6_4.supports_mesh_shaders());
        assert!(D3D12ShaderModel::SM_6_5.supports_mesh_shaders());
        assert!(D3D12ShaderModel::SM_6_6.supports_mesh_shaders());
        assert!(D3D12ShaderModel::SM_6_7.supports_mesh_shaders());
    }

    #[test]
    fn test_shader_model_supports_derivatives() {
        assert!(!D3D12ShaderModel::SM_5_1.supports_derivatives());
        assert!(!D3D12ShaderModel::SM_6_5.supports_derivatives());
        assert!(D3D12ShaderModel::SM_6_6.supports_derivatives());
        assert!(D3D12ShaderModel::SM_6_7.supports_derivatives());
    }

    #[test]
    fn test_shader_model_supports_16bit_types() {
        assert!(!D3D12ShaderModel::SM_5_1.supports_16bit_types());
        assert!(!D3D12ShaderModel::SM_6_1.supports_16bit_types());
        assert!(D3D12ShaderModel::SM_6_2.supports_16bit_types());
        assert!(D3D12ShaderModel::SM_6_5.supports_16bit_types());
    }

    #[test]
    fn test_shader_model_name() {
        assert_eq!(D3D12ShaderModel::SM_5_1.name(), "5.1");
        assert_eq!(D3D12ShaderModel::SM_6_0.name(), "6.0");
        assert_eq!(D3D12ShaderModel::SM_6_5.name(), "6.5");
        assert_eq!(D3D12ShaderModel::SM_6_7.name(), "6.7");
    }

    #[test]
    fn test_shader_model_version() {
        assert_eq!(D3D12ShaderModel::SM_5_1.version(), (5, 1));
        assert_eq!(D3D12ShaderModel::SM_6_0.version(), (6, 0));
        assert_eq!(D3D12ShaderModel::SM_6_5.version(), (6, 5));
        assert_eq!(D3D12ShaderModel::SM_6_7.version(), (6, 7));
    }

    #[test]
    fn test_shader_model_display() {
        assert_eq!(format!("{}", D3D12ShaderModel::SM_5_1), "SM 5.1");
        assert_eq!(format!("{}", D3D12ShaderModel::SM_6_5), "SM 6.5");
    }

    // ========================================================================
    // D3D12RayTracingTier Tests
    // ========================================================================

    #[test]
    fn test_rt_tier_default() {
        let tier = D3D12RayTracingTier::default();
        assert_eq!(tier, D3D12RayTracingTier::None);
    }

    #[test]
    fn test_rt_tier_ordering() {
        assert!(D3D12RayTracingTier::None < D3D12RayTracingTier::Tier1_0);
        assert!(D3D12RayTracingTier::Tier1_0 < D3D12RayTracingTier::Tier1_1);
    }

    #[test]
    fn test_rt_tier_supports_inline_raytracing() {
        assert!(!D3D12RayTracingTier::None.supports_inline_raytracing());
        assert!(!D3D12RayTracingTier::Tier1_0.supports_inline_raytracing());
        assert!(D3D12RayTracingTier::Tier1_1.supports_inline_raytracing());
    }

    #[test]
    fn test_rt_tier_supports_rayquery() {
        assert!(!D3D12RayTracingTier::None.supports_rayquery());
        assert!(!D3D12RayTracingTier::Tier1_0.supports_rayquery());
        assert!(D3D12RayTracingTier::Tier1_1.supports_rayquery());
    }

    #[test]
    fn test_rt_tier_is_available() {
        assert!(!D3D12RayTracingTier::None.is_available());
        assert!(D3D12RayTracingTier::Tier1_0.is_available());
        assert!(D3D12RayTracingTier::Tier1_1.is_available());
    }

    #[test]
    fn test_rt_tier_name() {
        assert_eq!(D3D12RayTracingTier::None.name(), "None");
        assert_eq!(D3D12RayTracingTier::Tier1_0.name(), "DXR 1.0");
        assert_eq!(D3D12RayTracingTier::Tier1_1.name(), "DXR 1.1");
    }

    #[test]
    fn test_rt_tier_display() {
        assert_eq!(format!("{}", D3D12RayTracingTier::None), "None");
        assert_eq!(format!("{}", D3D12RayTracingTier::Tier1_0), "DXR 1.0");
        assert_eq!(format!("{}", D3D12RayTracingTier::Tier1_1), "DXR 1.1");
    }

    #[test]
    fn test_rt_tier_from_empty_features() {
        let features = Features::empty();
        let tier = D3D12RayTracingTier::from_features(features);
        assert_eq!(tier, D3D12RayTracingTier::None);
    }

    #[test]
    fn test_rt_tier_from_rt_features() {
        let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE;
        let tier = D3D12RayTracingTier::from_features(features);
        assert_eq!(tier, D3D12RayTracingTier::Tier1_0);
    }

    #[test]
    fn test_rt_tier_from_full_rt_features() {
        let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE | Features::RAY_QUERY;
        let tier = D3D12RayTracingTier::from_features(features);
        assert_eq!(tier, D3D12RayTracingTier::Tier1_1);
    }

    // ========================================================================
    // D3D12Features Tests
    // ========================================================================

    #[test]
    fn test_features_default() {
        let features = D3D12Features::default();
        assert_eq!(features.feature_level, D3D12FeatureLevel::FL_11_0);
        assert_eq!(features.shader_model, D3D12ShaderModel::SM_5_1);
        assert_eq!(features.ray_tracing_tier, D3D12RayTracingTier::None);
        assert_eq!(features.mesh_shader_tier, 0);
        assert_eq!(features.variable_rate_shading_tier, 0);
        assert!(!features.bindless_resources);
    }

    #[test]
    fn test_features_supports_rt() {
        let mut features = D3D12Features::default();
        assert!(!features.supports_rt());

        features.ray_tracing_tier = D3D12RayTracingTier::Tier1_0;
        assert!(features.supports_rt());

        features.ray_tracing_tier = D3D12RayTracingTier::Tier1_1;
        assert!(features.supports_rt());
    }

    #[test]
    fn test_features_supports_mesh_shaders() {
        let mut features = D3D12Features::default();
        assert!(!features.supports_mesh_shaders());

        features.mesh_shader_tier = 1;
        assert!(features.supports_mesh_shaders());
    }

    #[test]
    fn test_features_supports_bindless() {
        let mut features = D3D12Features::default();
        assert!(!features.supports_bindless());

        features.bindless_resources = true;
        assert!(features.supports_bindless());
    }

    #[test]
    fn test_features_supports_vrs() {
        let mut features = D3D12Features::default();
        assert!(!features.supports_vrs());

        features.variable_rate_shading_tier = 1;
        assert!(features.supports_vrs());

        features.variable_rate_shading_tier = 2;
        assert!(features.supports_vrs());
    }

    #[test]
    fn test_features_supports_inline_rt() {
        let mut features = D3D12Features::default();
        assert!(!features.supports_inline_rt());

        features.ray_tracing_tier = D3D12RayTracingTier::Tier1_0;
        assert!(!features.supports_inline_rt());

        features.ray_tracing_tier = D3D12RayTracingTier::Tier1_1;
        assert!(features.supports_inline_rt());
    }

    #[test]
    fn test_features_supports_sampler_feedback() {
        let mut features = D3D12Features::default();
        assert!(!features.supports_sampler_feedback());

        features.sampler_feedback_tier = 1;
        assert!(features.supports_sampler_feedback());
    }

    #[test]
    fn test_features_supports_conservative_raster() {
        let mut features = D3D12Features::default();
        assert!(!features.supports_conservative_raster());

        features.conservative_rasterization_tier = 1;
        assert!(features.supports_conservative_raster());
    }

    #[test]
    fn test_features_supports_gpu_driven() {
        let mut features = D3D12Features::default();
        assert!(!features.supports_gpu_driven());

        features.bindless_resources = true;
        assert!(!features.supports_gpu_driven());

        features.wave_ops = true;
        assert!(features.supports_gpu_driven());
    }

    #[test]
    fn test_features_summary_basic() {
        let features = D3D12Features::default();
        let summary = features.summary();
        assert!(summary.contains("FL 11.0"));
        assert!(summary.contains("SM 5.1"));
    }

    #[test]
    fn test_features_summary_full() {
        let features = D3D12Features {
            feature_level: D3D12FeatureLevel::FL_12_2,
            shader_model: D3D12ShaderModel::SM_6_5,
            ray_tracing_tier: D3D12RayTracingTier::Tier1_1,
            mesh_shader_tier: 1,
            variable_rate_shading_tier: 2,
            sampler_feedback_tier: 1,
            bindless_resources: true,
            conservative_rasterization_tier: 3,
            tiled_resources_tier: 4,
            resource_binding_tier: 3,
            root_signature_version: 2,
            wave_ops: true,
            native_16bit_ops: true,
            rt_pipeline: true,
            rasterizer_ordered_views_tier: 3,
        };

        let summary = features.summary();
        assert!(summary.contains("FL 12.2"));
        assert!(summary.contains("SM 6.5"));
        assert!(summary.contains("DXR 1.1"));
        assert!(summary.contains("Mesh"));
        assert!(summary.contains("VRS T2"));
        assert!(summary.contains("Bindless"));
        assert!(summary.contains("Wave"));
        assert!(summary.contains("FP16"));
    }

    #[test]
    fn test_features_minimum_windows_version_basic() {
        let features = D3D12Features::default();
        let (major, minor, build) = features.minimum_windows_version();
        assert_eq!(major, 10);
        assert_eq!(minor, 0);
        assert_eq!(build, 10240);
    }

    #[test]
    fn test_features_minimum_windows_version_vrs() {
        let mut features = D3D12Features::default();
        features.variable_rate_shading_tier = 1;
        let (_, _, build) = features.minimum_windows_version();
        assert_eq!(build, 18362);
    }

    #[test]
    fn test_features_minimum_windows_version_dxr_1_0() {
        let mut features = D3D12Features::default();
        features.ray_tracing_tier = D3D12RayTracingTier::Tier1_0;
        let (_, _, build) = features.minimum_windows_version();
        assert_eq!(build, 17763);
    }

    #[test]
    fn test_features_minimum_windows_version_dxr_1_1() {
        let mut features = D3D12Features::default();
        features.ray_tracing_tier = D3D12RayTracingTier::Tier1_1;
        let (_, _, build) = features.minimum_windows_version();
        assert_eq!(build, 19041);
    }

    #[test]
    fn test_features_from_empty_wgpu_features() {
        let wgpu_features = Features::empty();
        let dx12_features = D3D12Features::from_features(wgpu_features);

        assert_eq!(dx12_features.feature_level, D3D12FeatureLevel::FL_11_0);
        assert_eq!(dx12_features.shader_model, D3D12ShaderModel::SM_5_1);
        assert_eq!(dx12_features.ray_tracing_tier, D3D12RayTracingTier::None);
        assert!(!dx12_features.bindless_resources);
    }

    #[test]
    fn test_features_from_rt_wgpu_features() {
        let wgpu_features = Features::RAY_TRACING_ACCELERATION_STRUCTURE | Features::RAY_QUERY;
        let dx12_features = D3D12Features::from_features(wgpu_features);

        assert_eq!(dx12_features.feature_level, D3D12FeatureLevel::FL_12_2);
        assert_eq!(dx12_features.shader_model, D3D12ShaderModel::SM_6_5);
        assert_eq!(dx12_features.ray_tracing_tier, D3D12RayTracingTier::Tier1_1);
        assert!(dx12_features.supports_rt());
        assert!(dx12_features.supports_inline_rt());
    }

    #[test]
    fn test_features_from_bindless_wgpu_features() {
        let wgpu_features = Features::TEXTURE_BINDING_ARRAY
            | Features::BUFFER_BINDING_ARRAY
            | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
        let dx12_features = D3D12Features::from_features(wgpu_features);

        assert!(dx12_features.bindless_resources);
        assert!(dx12_features.supports_bindless());
    }

    #[test]
    fn test_features_from_subgroup_wgpu_features() {
        let wgpu_features = Features::SUBGROUP;
        let dx12_features = D3D12Features::from_features(wgpu_features);

        assert_eq!(dx12_features.shader_model, D3D12ShaderModel::SM_6_0);
        assert!(dx12_features.wave_ops);
    }

    // ========================================================================
    // Integration-style Tests
    // ========================================================================

    #[test]
    fn test_modern_gpu_has_expected_features() {
        // Simulate a modern GPU with full RT + bindless
        let wgpu_features = Features::RAY_TRACING_ACCELERATION_STRUCTURE
            | Features::RAY_QUERY
            | Features::TEXTURE_BINDING_ARRAY
            | Features::BUFFER_BINDING_ARRAY
            | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
            | Features::SUBGROUP;

        let dx12 = D3D12Features::from_features(wgpu_features);

        assert_eq!(dx12.feature_level, D3D12FeatureLevel::FL_12_2);
        assert_eq!(dx12.shader_model, D3D12ShaderModel::SM_6_5);
        assert_eq!(dx12.ray_tracing_tier, D3D12RayTracingTier::Tier1_1);
        assert!(dx12.supports_rt());
        assert!(dx12.supports_inline_rt());
        assert!(dx12.supports_bindless());
        assert!(dx12.supports_gpu_driven());
        assert_eq!(dx12.mesh_shader_tier, 1);
        assert!(dx12.supports_mesh_shaders());
        assert!(dx12.supports_vrs());
        assert!(dx12.supports_sampler_feedback());
    }

    #[test]
    fn test_feature_level_fl_12_1_capabilities() {
        // Simulate FL 12.1 with RT but no ray query
        let wgpu_features = Features::RAY_TRACING_ACCELERATION_STRUCTURE
            | Features::TEXTURE_BINDING_ARRAY
            | Features::BUFFER_BINDING_ARRAY
            | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
            | Features::SUBGROUP;

        let dx12 = D3D12Features::from_features(wgpu_features);

        assert_eq!(dx12.feature_level, D3D12FeatureLevel::FL_12_1);
        assert!(dx12.supports_rt());
        assert!(!dx12.supports_inline_rt());
        assert!(dx12.supports_vrs());
        assert!(!dx12.supports_sampler_feedback());
    }

    #[test]
    fn test_tiers_consistency() {
        // Test that higher feature levels imply higher tiers
        let fl_11 = D3D12FeatureLevel::FL_11_0;
        let fl_12_0 = D3D12FeatureLevel::FL_12_0;
        let fl_12_1 = D3D12FeatureLevel::FL_12_1;
        let fl_12_2 = D3D12FeatureLevel::FL_12_2;

        // VRS tier progression
        assert!(!fl_11.supports_variable_rate_shading());
        assert!(!fl_12_0.supports_variable_rate_shading());
        assert!(fl_12_1.supports_variable_rate_shading());
        assert!(fl_12_2.supports_variable_rate_shading());

        // RT tier progression
        assert!(!fl_11.supports_ray_tracing());
        assert!(!fl_12_0.supports_ray_tracing());
        assert!(fl_12_1.supports_ray_tracing());
        assert!(fl_12_2.supports_ray_tracing());

        // Mesh shader tier progression
        assert!(!fl_11.supports_mesh_shaders());
        assert!(!fl_12_0.supports_mesh_shaders());
        assert!(!fl_12_1.supports_mesh_shaders());
        assert!(fl_12_2.supports_mesh_shaders());
    }

    #[test]
    fn test_shader_model_feature_progression() {
        let sm_5_1 = D3D12ShaderModel::SM_5_1;
        let sm_6_0 = D3D12ShaderModel::SM_6_0;
        let sm_6_3 = D3D12ShaderModel::SM_6_3;
        let sm_6_5 = D3D12ShaderModel::SM_6_5;
        let sm_6_6 = D3D12ShaderModel::SM_6_6;

        // Wave intrinsics progression
        assert!(!sm_5_1.supports_wave_intrinsics());
        assert!(sm_6_0.supports_wave_intrinsics());
        assert!(sm_6_5.supports_wave_intrinsics());

        // RT intrinsics progression
        assert!(!sm_5_1.supports_raytracing_intrinsics());
        assert!(!sm_6_0.supports_raytracing_intrinsics());
        assert!(sm_6_3.supports_raytracing_intrinsics());
        assert!(sm_6_5.supports_raytracing_intrinsics());

        // Mesh shaders progression
        assert!(!sm_5_1.supports_mesh_shaders());
        assert!(!sm_6_3.supports_mesh_shaders());
        assert!(sm_6_5.supports_mesh_shaders());
        assert!(sm_6_6.supports_mesh_shaders());

        // Derivatives progression
        assert!(!sm_6_5.supports_derivatives());
        assert!(sm_6_6.supports_derivatives());
    }

    // ========================================================================
    // ShaderCompiler Tests
    // ========================================================================

    #[test]
    fn test_shader_compiler_fxc() {
        let fxc = ShaderCompiler::FXC;
        assert_eq!(fxc.name(), "FXC");
        assert_eq!(fxc.executable(), "fxc");
        assert!(!fxc.supports_spirv());
        assert!(fxc.supports_shader_model(D3D12ShaderModel::SM_5_0));
        assert!(fxc.supports_shader_model(D3D12ShaderModel::SM_5_1));
        assert!(!fxc.supports_shader_model(D3D12ShaderModel::SM_6_0));
        assert!(!fxc.supports_shader_model(D3D12ShaderModel::SM_6_5));
    }

    #[test]
    fn test_shader_compiler_dxc() {
        let dxc = ShaderCompiler::DXC;
        assert_eq!(dxc.name(), "DXC");
        assert_eq!(dxc.executable(), "dxc");
        assert!(dxc.supports_spirv());
        assert!(dxc.supports_shader_model(D3D12ShaderModel::SM_5_0));
        assert!(dxc.supports_shader_model(D3D12ShaderModel::SM_5_1));
        assert!(dxc.supports_shader_model(D3D12ShaderModel::SM_6_0));
        assert!(dxc.supports_shader_model(D3D12ShaderModel::SM_6_5));
        assert!(dxc.supports_shader_model(D3D12ShaderModel::SM_6_7));
    }

    #[test]
    fn test_shader_compiler_recommended() {
        assert_eq!(
            ShaderCompiler::recommended_for(D3D12ShaderModel::SM_5_0),
            ShaderCompiler::FXC
        );
        assert_eq!(
            ShaderCompiler::recommended_for(D3D12ShaderModel::SM_5_1),
            ShaderCompiler::FXC
        );
        assert_eq!(
            ShaderCompiler::recommended_for(D3D12ShaderModel::SM_6_0),
            ShaderCompiler::DXC
        );
        assert_eq!(
            ShaderCompiler::recommended_for(D3D12ShaderModel::SM_6_5),
            ShaderCompiler::DXC
        );
    }

    #[test]
    fn test_shader_compiler_default() {
        let compiler = ShaderCompiler::default();
        assert_eq!(compiler, ShaderCompiler::DXC);
    }

    #[test]
    fn test_shader_compiler_display() {
        assert_eq!(format!("{}", ShaderCompiler::FXC), "FXC");
        assert_eq!(format!("{}", ShaderCompiler::DXC), "DXC");
    }

    // ========================================================================
    // MeshShaderTier Tests
    // ========================================================================

    #[test]
    fn test_mesh_shader_tier_default() {
        let tier = MeshShaderTier::default();
        assert_eq!(tier, MeshShaderTier::NotSupported);
    }

    #[test]
    fn test_mesh_shader_tier_is_supported() {
        assert!(!MeshShaderTier::NotSupported.is_supported());
        assert!(MeshShaderTier::Tier1.is_supported());
    }

    #[test]
    fn test_mesh_shader_tier_ordering() {
        assert!(MeshShaderTier::NotSupported < MeshShaderTier::Tier1);
    }

    #[test]
    fn test_mesh_shader_tier_name() {
        assert_eq!(MeshShaderTier::NotSupported.name(), "Not Supported");
        assert_eq!(MeshShaderTier::Tier1.name(), "Tier 1");
    }

    #[test]
    fn test_mesh_shader_tier_display() {
        assert_eq!(format!("{}", MeshShaderTier::NotSupported), "Not Supported");
        assert_eq!(format!("{}", MeshShaderTier::Tier1), "Tier 1");
    }

    #[test]
    fn test_mesh_shader_tier_from_features() {
        // No mesh shader support without ray query
        let features = Features::empty();
        let tier = MeshShaderTier::from_features(features);
        assert_eq!(tier, MeshShaderTier::NotSupported);

        // Mesh shader support with full RT
        let features = Features::RAY_TRACING_ACCELERATION_STRUCTURE | Features::RAY_QUERY;
        let tier = MeshShaderTier::from_features(features);
        assert_eq!(tier, MeshShaderTier::Tier1);
    }

    // ========================================================================
    // DX12Info Tests
    // ========================================================================

    #[test]
    fn test_dx12_info_default() {
        let info = DX12Info::default();
        assert_eq!(info.adapter_name, "");
        assert_eq!(info.vendor_id, 0);
        assert_eq!(info.device_id, 0);
        assert_eq!(info.driver_version, "");
        assert!(!info.supports_ray_tracing());
    }

    #[test]
    fn test_dx12_info_vendor_detection() {
        let mut info = DX12Info::default();

        info.vendor_id = 0x10DE;
        assert!(info.is_nvidia());
        assert!(!info.is_amd());
        assert!(!info.is_intel());
        assert_eq!(info.vendor_name(), "NVIDIA");

        info.vendor_id = 0x1002;
        assert!(info.is_amd());
        assert!(!info.is_nvidia());
        assert_eq!(info.vendor_name(), "AMD");

        info.vendor_id = 0x8086;
        assert!(info.is_intel());
        assert!(!info.is_nvidia());
        assert_eq!(info.vendor_name(), "Intel");

        info.vendor_id = 0x1414;
        assert!(info.is_warp());
        assert_eq!(info.vendor_name(), "Microsoft");

        info.vendor_id = 0x0000;
        assert_eq!(info.vendor_name(), "Unknown");
    }

    #[test]
    fn test_dx12_info_ray_tracing() {
        let mut info = DX12Info::default();
        assert!(!info.supports_ray_tracing());

        info.capabilities.ray_tracing_tier = D3D12RayTracingTier::Tier1_0;
        assert!(info.supports_ray_tracing());

        info.capabilities.ray_tracing_tier = D3D12RayTracingTier::Tier1_1;
        assert!(info.supports_ray_tracing());
    }

    #[test]
    fn test_dx12_info_recommended_compiler() {
        let mut info = DX12Info::default();

        info.capabilities.shader_model = D3D12ShaderModel::SM_5_1;
        assert_eq!(info.recommended_compiler(), ShaderCompiler::FXC);

        info.capabilities.shader_model = D3D12ShaderModel::SM_6_0;
        assert_eq!(info.recommended_compiler(), ShaderCompiler::DXC);

        info.capabilities.shader_model = D3D12ShaderModel::SM_6_5;
        assert_eq!(info.recommended_compiler(), ShaderCompiler::DXC);
    }

    #[test]
    fn test_dx12_info_display() {
        let mut info = DX12Info::default();
        info.adapter_name = "NVIDIA GeForce RTX 4090".to_string();
        info.vendor_id = 0x10DE;

        let display = format!("{}", info);
        assert!(display.contains("NVIDIA GeForce RTX 4090"));
        assert!(display.contains("NVIDIA"));
    }

    // ========================================================================
    // Feature Level min_shader_model Tests
    // ========================================================================

    #[test]
    fn test_feature_level_min_shader_model() {
        assert_eq!(
            D3D12FeatureLevel::FL_11_0.min_shader_model(),
            D3D12ShaderModel::SM_5_1
        );
        assert_eq!(
            D3D12FeatureLevel::FL_11_1.min_shader_model(),
            D3D12ShaderModel::SM_5_1
        );
        assert_eq!(
            D3D12FeatureLevel::FL_12_0.min_shader_model(),
            D3D12ShaderModel::SM_6_0
        );
        assert_eq!(
            D3D12FeatureLevel::FL_12_1.min_shader_model(),
            D3D12ShaderModel::SM_6_3
        );
        assert_eq!(
            D3D12FeatureLevel::FL_12_2.min_shader_model(),
            D3D12ShaderModel::SM_6_5
        );
    }

    // ========================================================================
    // ShaderModel supports_ray_tracing Tests
    // ========================================================================

    #[test]
    fn test_shader_model_supports_ray_tracing() {
        assert!(!D3D12ShaderModel::SM_5_0.supports_ray_tracing());
        assert!(!D3D12ShaderModel::SM_5_1.supports_ray_tracing());
        assert!(!D3D12ShaderModel::SM_6_0.supports_ray_tracing());
        assert!(!D3D12ShaderModel::SM_6_2.supports_ray_tracing());
        assert!(D3D12ShaderModel::SM_6_3.supports_ray_tracing());
        assert!(D3D12ShaderModel::SM_6_5.supports_ray_tracing());
    }

    // ========================================================================
    // RayTracingTier is_supported Tests
    // ========================================================================

    #[test]
    fn test_ray_tracing_tier_is_supported() {
        assert!(!D3D12RayTracingTier::None.is_supported());
        assert!(D3D12RayTracingTier::Tier1_0.is_supported());
        assert!(D3D12RayTracingTier::Tier1_1.is_supported());
    }

    // ========================================================================
    // SM_5_0 Tests
    // ========================================================================

    #[test]
    fn test_shader_model_sm_5_0() {
        let sm = D3D12ShaderModel::SM_5_0;
        assert_eq!(sm.name(), "5.0");
        assert_eq!(sm.version(), (5, 0));
        assert!(!sm.supports_wave_intrinsics());
        assert!(!sm.supports_ray_tracing());
        assert!(!sm.supports_mesh_shaders());
        assert!(!sm.supports_16bit_types());
        assert_eq!(format!("{}", sm), "SM 5.0");
    }

    #[test]
    fn test_shader_model_sm_5_0_ordering() {
        assert!(D3D12ShaderModel::SM_5_0 < D3D12ShaderModel::SM_5_1);
        assert!(D3D12ShaderModel::SM_5_0 < D3D12ShaderModel::SM_6_0);
    }

    // ========================================================================
    // Type Alias Tests
    // ========================================================================

    #[test]
    fn test_type_aliases() {
        // Verify type aliases compile and work correctly
        let fl: D3DFeatureLevel = D3D12FeatureLevel::FL_12_1;
        assert_eq!(fl, D3D12FeatureLevel::FL_12_1);

        let sm: ShaderModel = D3D12ShaderModel::SM_6_5;
        assert_eq!(sm, D3D12ShaderModel::SM_6_5);

        let rt: RayTracingTier = D3D12RayTracingTier::Tier1_1;
        assert_eq!(rt, D3D12RayTracingTier::Tier1_1);

        let caps: DX12Capabilities = D3D12Features::default();
        assert_eq!(caps.feature_level, D3D12FeatureLevel::FL_11_0);
    }
}
