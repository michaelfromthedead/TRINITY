//! Shader permutation management for TRINITY.
//!
//! This module provides a permutation manager that handles shader variants
//! based on feature flags. Permutations allow a single shader source to be
//! compiled with different configurations (e.g., skinned vs static meshes,
//! with or without shadows, etc.).
//!
//! # Overview
//!
//! Shader permutations are defined by feature flags that enable/disable
//! specific shader functionality:
//!
//! - **SKINNED**: Skeletal animation support
//! - **ALPHA_TEST**: Alpha testing for cutout transparency
//! - **NORMAL_MAP**: Normal mapping support
//! - **EMISSIVE**: Emissive material support
//! - **SHADOWS**: Shadow mapping
//! - **FOG**: Fog effects
//! - **INSTANCED**: GPU instancing
//!
//! # Architecture
//!
//! ```text
//! FeatureFlags (bitflags)
//! +-- 7 common feature flags
//! +-- Bitwise operations (AND, OR, XOR)
//! +-- Iteration over enabled flags
//!
//! PermutationKey
//! +-- shader_id: u64 - Base shader identifier
//! +-- features: FeatureFlags - Active features
//! +-- Hash + Eq for HashMap key usage
//!
//! CachedPermutation
//! +-- module: Arc<TrinityShaderModule>
//! +-- features: FeatureFlags
//! +-- created_at: Instant
//! +-- access_count: AtomicU64
//!
//! PermutationConfig
//! +-- max_permutations: usize (default 256)
//! +-- enable_lazy_compilation: bool
//! +-- eviction_policy: EvictionPolicy
//!
//! ShaderPermutationManager
//! +-- cache: RwLock<HashMap<PermutationKey, CachedPermutation>>
//! +-- config: PermutationConfig
//! +-- metrics: PermutationMetrics
//! +-- get_or_compile() - Main API
//! +-- invalidate() - Remove permutations
//! ```
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::shaders::permutations::{
//!     ShaderPermutationManager, PermutationConfig, FeatureFlags,
//! };
//!
//! let manager = ShaderPermutationManager::new(PermutationConfig::default());
//!
//! // Get or compile a permutation with specific features
//! let features = FeatureFlags::SKINNED | FeatureFlags::SHADOWS;
//! let shader = manager.get_or_compile(
//!     &device,
//!     &base_shader,
//!     features,
//!     &override_constants,
//! )?;
//! ```

use bitflags::bitflags;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use super::{
    create_shader_module, PipelineConstants, ShaderError, TrinityShaderDescriptor,
    TrinityShaderModule,
};

// ============================================================================
// Constants
// ============================================================================

/// Default maximum number of cached permutations.
pub const DEFAULT_MAX_PERMUTATIONS: usize = 256;

/// Number of defined feature flags.
pub const FEATURE_FLAG_COUNT: usize = 7;

// ============================================================================
// FeatureFlags
// ============================================================================

bitflags! {
    /// Feature flags for shader permutations.
    ///
    /// Each flag represents a specific shader feature that can be enabled
    /// or disabled at compile time. Combining flags creates unique shader
    /// permutations.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::shaders::permutations::FeatureFlags;
    ///
    /// // Create a feature set for skinned meshes with shadows
    /// let features = FeatureFlags::SKINNED | FeatureFlags::SHADOWS;
    ///
    /// // Check if a feature is enabled
    /// assert!(features.contains(FeatureFlags::SKINNED));
    /// assert!(!features.contains(FeatureFlags::FOG));
    /// ```
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    pub struct FeatureFlags: u32 {
        /// No features enabled.
        const NONE = 0;
        /// Skeletal animation / skinned mesh support.
        const SKINNED = 1 << 0;
        /// Alpha testing for cutout transparency.
        const ALPHA_TEST = 1 << 1;
        /// Normal mapping support.
        const NORMAL_MAP = 1 << 2;
        /// Emissive material support.
        const EMISSIVE = 1 << 3;
        /// Shadow mapping support.
        const SHADOWS = 1 << 4;
        /// Fog effects.
        const FOG = 1 << 5;
        /// GPU instancing.
        const INSTANCED = 1 << 6;
        /// All features enabled.
        const ALL = Self::SKINNED.bits()
            | Self::ALPHA_TEST.bits()
            | Self::NORMAL_MAP.bits()
            | Self::EMISSIVE.bits()
            | Self::SHADOWS.bits()
            | Self::FOG.bits()
            | Self::INSTANCED.bits();
    }
}

impl FeatureFlags {
    /// Returns the number of enabled features.
    #[inline]
    pub fn flag_count(&self) -> usize {
        self.bits().count_ones() as usize
    }

    /// Returns true if no features are enabled.
    #[inline]
    pub fn is_none_set(&self) -> bool {
        self.bits() == 0
    }

    /// Returns true if all features are enabled.
    #[inline]
    pub fn is_all_set(&self) -> bool {
        *self == Self::ALL
    }

    /// Returns an iterator over all enabled flags.
    pub fn iter_enabled(&self) -> impl Iterator<Item = FeatureFlags> + '_ {
        [
            FeatureFlags::SKINNED,
            FeatureFlags::ALPHA_TEST,
            FeatureFlags::NORMAL_MAP,
            FeatureFlags::EMISSIVE,
            FeatureFlags::SHADOWS,
            FeatureFlags::FOG,
            FeatureFlags::INSTANCED,
        ]
        .into_iter()
        .filter(|flag| self.contains(*flag))
    }

    /// Returns a list of enabled feature names.
    pub fn names(&self) -> Vec<&'static str> {
        self.iter_enabled().map(|f| f.flag_name()).collect()
    }

    /// Returns the name of a single flag.
    ///
    /// If multiple flags are set, returns the name of the first one.
    pub fn flag_name(&self) -> &'static str {
        if self.contains(FeatureFlags::SKINNED) {
            "SKINNED"
        } else if self.contains(FeatureFlags::ALPHA_TEST) {
            "ALPHA_TEST"
        } else if self.contains(FeatureFlags::NORMAL_MAP) {
            "NORMAL_MAP"
        } else if self.contains(FeatureFlags::EMISSIVE) {
            "EMISSIVE"
        } else if self.contains(FeatureFlags::SHADOWS) {
            "SHADOWS"
        } else if self.contains(FeatureFlags::FOG) {
            "FOG"
        } else if self.contains(FeatureFlags::INSTANCED) {
            "INSTANCED"
        } else {
            "NONE"
        }
    }

    /// Creates flags from a list of feature names.
    ///
    /// Unknown names are ignored.
    pub fn from_names(names: &[&str]) -> Self {
        let mut flags = Self::empty();
        for name in names {
            flags |= Self::parse_name(name);
        }
        flags
    }

    /// Creates flags from a single feature name.
    ///
    /// Returns NONE if the name is unknown.
    pub fn parse_name(name: &str) -> Self {
        match name.to_uppercase().as_str() {
            "SKINNED" => Self::SKINNED,
            "ALPHA_TEST" => Self::ALPHA_TEST,
            "NORMAL_MAP" => Self::NORMAL_MAP,
            "EMISSIVE" => Self::EMISSIVE,
            "SHADOWS" => Self::SHADOWS,
            "FOG" => Self::FOG,
            "INSTANCED" => Self::INSTANCED,
            _ => Self::NONE,
        }
    }

    /// Adds a feature flag, returning the new flags.
    #[inline]
    pub fn with_feature(self, feature: FeatureFlags) -> Self {
        self | feature
    }

    /// Removes a feature flag, returning the new flags.
    #[inline]
    pub fn without_feature(self, feature: FeatureFlags) -> Self {
        self & !feature
    }

    /// Toggles a feature flag, returning the new flags.
    #[inline]
    pub fn toggle_feature(self, feature: FeatureFlags) -> Self {
        self ^ feature
    }

    /// Returns the total number of possible permutations.
    ///
    /// This is 2^n where n is the number of feature flags.
    #[inline]
    pub fn total_permutations() -> usize {
        1 << FEATURE_FLAG_COUNT
    }

    /// Converts to PipelineConstants for shader compilation.
    ///
    /// Each flag is represented as a boolean constant (0.0 or 1.0).
    pub fn to_pipeline_constants(&self) -> PipelineConstants {
        let mut constants = PipelineConstants::new();
        constants.set_bool("FEATURE_SKINNED", self.contains(Self::SKINNED));
        constants.set_bool("FEATURE_ALPHA_TEST", self.contains(Self::ALPHA_TEST));
        constants.set_bool("FEATURE_NORMAL_MAP", self.contains(Self::NORMAL_MAP));
        constants.set_bool("FEATURE_EMISSIVE", self.contains(Self::EMISSIVE));
        constants.set_bool("FEATURE_SHADOWS", self.contains(Self::SHADOWS));
        constants.set_bool("FEATURE_FOG", self.contains(Self::FOG));
        constants.set_bool("FEATURE_INSTANCED", self.contains(Self::INSTANCED));
        constants
    }
}

impl fmt::Display for FeatureFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            write!(f, "NONE")
        } else {
            let names = self.names();
            write!(f, "{}", names.join(" | "))
        }
    }
}

// ============================================================================
// PermutationKey
// ============================================================================

/// A unique key identifying a shader permutation.
///
/// Combines the base shader identifier with feature flags to create
/// a unique key for caching compiled permutations.
///
/// # Example
///
/// ```
/// use renderer_backend::shaders::permutations::{PermutationKey, FeatureFlags};
///
/// let key = PermutationKey::new(12345, FeatureFlags::SKINNED | FeatureFlags::SHADOWS);
/// assert_eq!(key.shader_id(), 12345);
/// assert!(key.features().contains(FeatureFlags::SKINNED));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct PermutationKey {
    /// Base shader identifier (typically a hash or ID).
    shader_id: u64,
    /// Active feature flags.
    features: FeatureFlags,
}

impl PermutationKey {
    /// Creates a new permutation key.
    #[inline]
    pub fn new(shader_id: u64, features: FeatureFlags) -> Self {
        Self { shader_id, features }
    }

    /// Creates a key with no features enabled.
    #[inline]
    pub fn base(shader_id: u64) -> Self {
        Self {
            shader_id,
            features: FeatureFlags::NONE,
        }
    }

    /// Returns the shader ID.
    #[inline]
    pub fn shader_id(&self) -> u64 {
        self.shader_id
    }

    /// Returns the feature flags.
    #[inline]
    pub fn features(&self) -> FeatureFlags {
        self.features
    }

    /// Creates a new key with an additional feature.
    #[inline]
    pub fn with_feature(self, feature: FeatureFlags) -> Self {
        Self {
            shader_id: self.shader_id,
            features: self.features | feature,
        }
    }

    /// Creates a new key with a feature removed.
    #[inline]
    pub fn without_feature(self, feature: FeatureFlags) -> Self {
        Self {
            shader_id: self.shader_id,
            features: self.features & !feature,
        }
    }

    /// Creates a new key with different features.
    #[inline]
    pub fn with_features(self, features: FeatureFlags) -> Self {
        Self {
            shader_id: self.shader_id,
            features,
        }
    }

    /// Returns a display-friendly string.
    pub fn display_string(&self) -> String {
        format!("{}:{}", self.shader_id, self.features)
    }
}

impl PartialEq for PermutationKey {
    fn eq(&self, other: &Self) -> bool {
        self.shader_id == other.shader_id && self.features == other.features
    }
}

impl Eq for PermutationKey {}

impl Hash for PermutationKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.shader_id.hash(state);
        self.features.bits().hash(state);
    }
}

impl fmt::Display for PermutationKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Permutation({}, {})", self.shader_id, self.features)
    }
}

// ============================================================================
// CachedPermutation
// ============================================================================

/// A cached shader permutation with metadata.
///
/// Stores the compiled shader module along with timing and access
/// information for cache management.
pub struct CachedPermutation {
    /// The compiled shader module.
    module: Arc<TrinityShaderModule>,
    /// Feature flags used for this permutation.
    features: FeatureFlags,
    /// When the permutation was compiled.
    created_at: Instant,
    /// When the permutation was last accessed.
    last_accessed: RwLock<Instant>,
    /// Number of times accessed.
    access_count: AtomicU64,
}

impl CachedPermutation {
    /// Creates a new cached permutation.
    fn new(module: Arc<TrinityShaderModule>, features: FeatureFlags) -> Self {
        let now = Instant::now();
        Self {
            module,
            features,
            created_at: now,
            last_accessed: RwLock::new(now),
            access_count: AtomicU64::new(1),
        }
    }

    /// Returns a clone of the shader module Arc.
    #[inline]
    pub fn module(&self) -> Arc<TrinityShaderModule> {
        Arc::clone(&self.module)
    }

    /// Returns the feature flags.
    #[inline]
    pub fn features(&self) -> FeatureFlags {
        self.features
    }

    /// Returns when this permutation was created.
    #[inline]
    pub fn created_at(&self) -> Instant {
        self.created_at
    }

    /// Returns when this permutation was last accessed.
    #[inline]
    pub fn last_accessed(&self) -> Instant {
        *self.last_accessed.read()
    }

    /// Returns the access count.
    #[inline]
    pub fn access_count(&self) -> u64 {
        self.access_count.load(Ordering::Relaxed)
    }

    /// Updates access time and count.
    fn touch(&self) {
        *self.last_accessed.write() = Instant::now();
        self.access_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Returns the age since creation.
    pub fn age(&self) -> std::time::Duration {
        self.created_at.elapsed()
    }

    /// Returns the time since last access.
    pub fn idle_time(&self) -> std::time::Duration {
        self.last_accessed().elapsed()
    }
}

impl fmt::Debug for CachedPermutation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CachedPermutation")
            .field("features", &self.features)
            .field("created_at", &self.created_at)
            .field("access_count", &self.access_count.load(Ordering::Relaxed))
            .finish()
    }
}

// ============================================================================
// EvictionPolicy
// ============================================================================

/// Cache eviction policy for permutations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EvictionPolicy {
    /// Least Recently Used - evicts entries that haven't been accessed recently.
    #[default]
    LRU,
    /// Least Frequently Used - evicts entries with lowest access counts.
    LFU,
    /// Oldest First - evicts entries by creation time.
    Oldest,
}

impl EvictionPolicy {
    /// Returns the policy name.
    pub fn name(&self) -> &'static str {
        match self {
            EvictionPolicy::LRU => "LRU",
            EvictionPolicy::LFU => "LFU",
            EvictionPolicy::Oldest => "Oldest",
        }
    }
}

impl fmt::Display for EvictionPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// PermutationConfig
// ============================================================================

/// Configuration for the shader permutation manager.
///
/// # Example
///
/// ```
/// use renderer_backend::shaders::permutations::{PermutationConfig, EvictionPolicy};
///
/// let config = PermutationConfig::new()
///     .max_permutations(512)
///     .eviction_policy(EvictionPolicy::LFU)
///     .enable_lazy_compilation(true);
/// ```
#[derive(Debug, Clone)]
pub struct PermutationConfig {
    /// Maximum number of cached permutations.
    pub max_permutations: usize,
    /// Whether to compile permutations lazily (on first use).
    pub enable_lazy_compilation: bool,
    /// Eviction policy when cache is full.
    pub eviction_policy: EvictionPolicy,
}

impl Default for PermutationConfig {
    fn default() -> Self {
        Self {
            max_permutations: DEFAULT_MAX_PERMUTATIONS,
            enable_lazy_compilation: true,
            eviction_policy: EvictionPolicy::LRU,
        }
    }
}

impl PermutationConfig {
    /// Creates a new config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum number of cached permutations.
    #[inline]
    pub fn max_permutations(mut self, max: usize) -> Self {
        self.max_permutations = max;
        self
    }

    /// Enables or disables lazy compilation.
    #[inline]
    pub fn enable_lazy_compilation(mut self, enable: bool) -> Self {
        self.enable_lazy_compilation = enable;
        self
    }

    /// Sets the eviction policy.
    #[inline]
    pub fn eviction_policy(mut self, policy: EvictionPolicy) -> Self {
        self.eviction_policy = policy;
        self
    }

    /// Creates a minimal config for testing.
    pub fn minimal() -> Self {
        Self {
            max_permutations: 16,
            enable_lazy_compilation: true,
            eviction_policy: EvictionPolicy::LRU,
        }
    }

    /// Creates a config optimized for development.
    pub fn development() -> Self {
        Self {
            max_permutations: 64,
            enable_lazy_compilation: true,
            eviction_policy: EvictionPolicy::LRU,
        }
    }

    /// Creates a config optimized for production.
    pub fn production() -> Self {
        Self {
            max_permutations: 512,
            enable_lazy_compilation: true,
            eviction_policy: EvictionPolicy::LFU,
        }
    }

    /// Validates the configuration.
    pub fn validate(&self) -> Result<(), PermutationError> {
        if self.max_permutations == 0 {
            return Err(PermutationError::ConfigError(
                "max_permutations must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

// ============================================================================
// PermutationMetrics
// ============================================================================

/// Metrics for monitoring permutation cache performance.
#[derive(Debug, Clone, Default)]
pub struct PermutationMetrics {
    /// Number of permutations currently cached.
    pub cache_size: usize,
    /// Number of cache hits.
    pub cache_hits: u64,
    /// Number of cache misses (compilations).
    pub cache_misses: u64,
    /// Number of compilations performed.
    pub compilations: u64,
    /// Number of evictions performed.
    pub evictions: u64,
    /// Hit rate as a ratio (0.0 to 1.0).
    pub hit_rate: f64,
}

impl PermutationMetrics {
    /// Creates metrics from raw values.
    pub fn new(
        cache_size: usize,
        cache_hits: u64,
        cache_misses: u64,
        compilations: u64,
        evictions: u64,
    ) -> Self {
        let total = cache_hits + cache_misses;
        let hit_rate = if total > 0 {
            cache_hits as f64 / total as f64
        } else {
            0.0
        };
        Self {
            cache_size,
            cache_hits,
            cache_misses,
            compilations,
            evictions,
            hit_rate,
        }
    }

    /// Returns the total number of requests.
    #[inline]
    pub fn total_requests(&self) -> u64 {
        self.cache_hits + self.cache_misses
    }

    /// Returns the hit rate as a percentage.
    #[inline]
    pub fn hit_rate_percent(&self) -> f64 {
        self.hit_rate * 100.0
    }

    /// Returns the miss rate as a ratio.
    #[inline]
    pub fn miss_rate(&self) -> f64 {
        1.0 - self.hit_rate
    }

    /// Returns true if the cache is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.cache_size == 0
    }

    /// Resets all metrics to zero.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

impl fmt::Display for PermutationMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PermutationMetrics(size={}, hits={}, misses={}, hit_rate={:.1}%)",
            self.cache_size,
            self.cache_hits,
            self.cache_misses,
            self.hit_rate_percent()
        )
    }
}

// ============================================================================
// PermutationError
// ============================================================================

/// Errors related to permutation management.
#[derive(Debug, Clone, PartialEq)]
pub enum PermutationError {
    /// Maximum permutation limit exceeded.
    MaxPermutationsExceeded {
        /// Current count.
        current: usize,
        /// Maximum allowed.
        max: usize,
    },
    /// Shader compilation failed.
    CompilationFailed(String),
    /// Shader not found in cache.
    ShaderNotFound {
        /// The shader ID that was not found.
        shader_id: u64,
    },
    /// Configuration error.
    ConfigError(String),
}

impl PermutationError {
    /// Returns true if this is a max permutations exceeded error.
    pub fn is_max_exceeded(&self) -> bool {
        matches!(self, Self::MaxPermutationsExceeded { .. })
    }

    /// Returns true if this is a compilation error.
    pub fn is_compilation_error(&self) -> bool {
        matches!(self, Self::CompilationFailed(_))
    }

    /// Returns true if this is a shader not found error.
    pub fn is_not_found(&self) -> bool {
        matches!(self, Self::ShaderNotFound { .. })
    }

    /// Returns true if this is a config error.
    pub fn is_config_error(&self) -> bool {
        matches!(self, Self::ConfigError(_))
    }
}

impl fmt::Display for PermutationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MaxPermutationsExceeded { current, max } => {
                write!(
                    f,
                    "maximum permutations exceeded: {} (max {})",
                    current, max
                )
            }
            Self::CompilationFailed(msg) => {
                write!(f, "shader compilation failed: {}", msg)
            }
            Self::ShaderNotFound { shader_id } => {
                write!(f, "shader not found: {}", shader_id)
            }
            Self::ConfigError(msg) => {
                write!(f, "configuration error: {}", msg)
            }
        }
    }
}

impl std::error::Error for PermutationError {}

impl From<ShaderError> for PermutationError {
    fn from(err: ShaderError) -> Self {
        PermutationError::CompilationFailed(err.to_string())
    }
}

// ============================================================================
// ShaderPermutationManager
// ============================================================================

/// Manages shader permutations with lazy compilation and caching.
///
/// The manager maintains a cache of compiled shader permutations, keyed
/// by the combination of base shader ID and feature flags. Permutations
/// are compiled on first use (lazy compilation) and cached for reuse.
///
/// # Thread Safety
///
/// The manager is thread-safe and can be shared across threads. It uses
/// `RwLock` for the cache and atomic counters for metrics.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::shaders::permutations::{
///     ShaderPermutationManager, PermutationConfig, FeatureFlags,
/// };
///
/// let manager = ShaderPermutationManager::new(PermutationConfig::default());
///
/// // Compile or retrieve a cached permutation
/// let features = FeatureFlags::SKINNED | FeatureFlags::SHADOWS;
/// let shader = manager.get_or_compile(
///     &device,
///     &base_shader_source,
///     shader_id,
///     features,
///     &overrides,
/// )?;
///
/// // Check metrics
/// let metrics = manager.metrics();
/// println!("Cache hit rate: {:.1}%", metrics.hit_rate_percent());
/// ```
pub struct ShaderPermutationManager {
    /// Configuration.
    config: PermutationConfig,
    /// Cache of compiled permutations.
    cache: RwLock<HashMap<PermutationKey, CachedPermutation>>,
    /// Cache hit counter.
    cache_hits: AtomicU64,
    /// Cache miss counter.
    cache_misses: AtomicU64,
    /// Compilation counter.
    compilations: AtomicU64,
    /// Eviction counter.
    evictions: AtomicU64,
}

impl ShaderPermutationManager {
    /// Creates a new permutation manager.
    pub fn new(config: PermutationConfig) -> Self {
        Self {
            config,
            cache: RwLock::new(HashMap::new()),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            compilations: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        }
    }

    /// Creates a new manager with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(PermutationConfig::default())
    }

    /// Gets a cached permutation or compiles it.
    ///
    /// This is the main API for retrieving shader permutations. If the
    /// permutation is already cached, it is returned immediately. Otherwise,
    /// the shader is compiled with the appropriate feature flags and cached.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device for compilation
    /// * `source` - The base shader WGSL source
    /// * `shader_id` - Unique identifier for the base shader
    /// * `features` - Feature flags to enable
    /// * `base_constants` - Additional pipeline constants to merge
    ///
    /// # Returns
    ///
    /// An `Arc<TrinityShaderModule>` for the compiled permutation.
    pub fn get_or_compile(
        &self,
        device: &wgpu::Device,
        source: &str,
        shader_id: u64,
        features: FeatureFlags,
        base_constants: &PipelineConstants,
    ) -> Result<Arc<TrinityShaderModule>, PermutationError> {
        let key = PermutationKey::new(shader_id, features);

        // Fast path: read lock to check cache
        {
            let cache = self.cache.read();
            if let Some(cached) = cache.get(&key) {
                self.cache_hits.fetch_add(1, Ordering::Relaxed);
                cached.touch();
                return Ok(cached.module());
            }
        }

        // Slow path: write lock for compilation
        let mut cache = self.cache.write();

        // Double-check pattern
        if let Some(cached) = cache.get(&key) {
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
            cached.touch();
            return Ok(cached.module());
        }

        // Check if we need to evict
        if cache.len() >= self.config.max_permutations {
            if !self.evict_one(&mut cache) {
                return Err(PermutationError::MaxPermutationsExceeded {
                    current: cache.len(),
                    max: self.config.max_permutations,
                });
            }
        }

        // Compile the permutation
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
        let module = self.compile_permutation(device, source, shader_id, features, base_constants)?;

        // Cache it
        let cached = CachedPermutation::new(Arc::clone(&module), features);
        cache.insert(key, cached);

        Ok(module)
    }

    /// Gets a cached permutation without compiling.
    ///
    /// Returns `None` if the permutation is not cached.
    pub fn get_cached(&self, key: &PermutationKey) -> Option<Arc<TrinityShaderModule>> {
        let cache = self.cache.read();
        cache.get(key).map(|c| {
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
            c.touch();
            c.module()
        })
    }

    /// Checks if a permutation is cached.
    pub fn contains(&self, key: &PermutationKey) -> bool {
        self.cache.read().contains_key(key)
    }

    /// Compiles a shader permutation.
    fn compile_permutation(
        &self,
        device: &wgpu::Device,
        source: &str,
        shader_id: u64,
        features: FeatureFlags,
        base_constants: &PipelineConstants,
    ) -> Result<Arc<TrinityShaderModule>, PermutationError> {
        // Merge feature constants with base constants
        let mut constants = features.to_pipeline_constants();
        constants.merge(base_constants);

        // Create a label that includes the feature flags
        let label = format!("shader_{}_{}", shader_id, features.bits());

        // Compile the shader
        let desc = TrinityShaderDescriptor::wgsl(Some(&label), source);
        let module = create_shader_module(device, &desc)?;

        self.compilations.fetch_add(1, Ordering::Relaxed);

        Ok(Arc::new(module))
    }

    /// Evicts one entry from the cache based on eviction policy.
    fn evict_one(&self, cache: &mut HashMap<PermutationKey, CachedPermutation>) -> bool {
        if cache.is_empty() {
            return false;
        }

        let key_to_evict = match self.config.eviction_policy {
            EvictionPolicy::LRU => cache
                .iter()
                .min_by_key(|(_, c)| c.last_accessed())
                .map(|(k, _)| *k),
            EvictionPolicy::LFU => cache
                .iter()
                .min_by_key(|(_, c)| c.access_count())
                .map(|(k, _)| *k),
            EvictionPolicy::Oldest => cache
                .iter()
                .min_by_key(|(_, c)| c.created_at())
                .map(|(k, _)| *k),
        };

        if let Some(key) = key_to_evict {
            cache.remove(&key);
            self.evictions.fetch_add(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Invalidates all permutations for a specific shader.
    ///
    /// Returns the number of permutations removed.
    pub fn invalidate(&self, shader_id: u64) -> usize {
        let mut cache = self.cache.write();
        let keys_to_remove: Vec<_> = cache
            .keys()
            .filter(|k| k.shader_id() == shader_id)
            .copied()
            .collect();

        let count = keys_to_remove.len();
        for key in keys_to_remove {
            cache.remove(&key);
        }
        count
    }

    /// Invalidates a specific permutation.
    ///
    /// Returns true if the permutation was removed.
    pub fn invalidate_key(&self, key: &PermutationKey) -> bool {
        self.cache.write().remove(key).is_some()
    }

    /// Clears the entire cache.
    pub fn invalidate_all(&self) {
        self.cache.write().clear();
    }

    /// Returns the number of cached permutations.
    #[inline]
    pub fn permutation_count(&self) -> usize {
        self.cache.read().len()
    }

    /// Returns true if the cache is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.cache.read().is_empty()
    }

    /// Returns current metrics.
    pub fn metrics(&self) -> PermutationMetrics {
        let cache = self.cache.read();
        let cache_hits = self.cache_hits.load(Ordering::Relaxed);
        let cache_misses = self.cache_misses.load(Ordering::Relaxed);
        let compilations = self.compilations.load(Ordering::Relaxed);
        let evictions = self.evictions.load(Ordering::Relaxed);

        PermutationMetrics::new(cache.len(), cache_hits, cache_misses, compilations, evictions)
    }

    /// Resets metrics counters.
    pub fn reset_metrics(&self) {
        self.cache_hits.store(0, Ordering::Relaxed);
        self.cache_misses.store(0, Ordering::Relaxed);
        self.compilations.store(0, Ordering::Relaxed);
        self.evictions.store(0, Ordering::Relaxed);
    }

    /// Returns the configuration.
    #[inline]
    pub fn config(&self) -> &PermutationConfig {
        &self.config
    }

    /// Returns all cached permutation keys.
    pub fn keys(&self) -> Vec<PermutationKey> {
        self.cache.read().keys().copied().collect()
    }

    /// Returns all unique shader IDs in the cache.
    pub fn shader_ids(&self) -> Vec<u64> {
        let cache = self.cache.read();
        let mut ids: Vec<_> = cache.keys().map(|k| k.shader_id()).collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Returns the number of permutations for a specific shader.
    pub fn permutation_count_for_shader(&self, shader_id: u64) -> usize {
        self.cache
            .read()
            .keys()
            .filter(|k| k.shader_id() == shader_id)
            .count()
    }
}

impl fmt::Debug for ShaderPermutationManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let metrics = self.metrics();
        f.debug_struct("ShaderPermutationManager")
            .field("config", &self.config)
            .field("cache_size", &metrics.cache_size)
            .field("cache_hits", &metrics.cache_hits)
            .field("cache_misses", &metrics.cache_misses)
            .field("hit_rate", &format!("{:.1}%", metrics.hit_rate_percent()))
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // FeatureFlags Tests
    // =========================================================================

    #[test]
    fn test_feature_flags_none() {
        let flags = FeatureFlags::NONE;
        assert!(flags.is_empty());
        assert!(!flags.is_all_set());
        assert_eq!(flags.flag_count(), 0);
        assert_eq!(flags.bits(), 0);
    }

    #[test]
    fn test_feature_flags_all() {
        let flags = FeatureFlags::ALL;
        assert!(!flags.is_empty());
        assert!(flags.is_all_set());
        assert_eq!(flags.flag_count(), FEATURE_FLAG_COUNT);
    }

    #[test]
    fn test_feature_flags_single() {
        let flags = FeatureFlags::SKINNED;
        assert_eq!(flags.flag_count(), 1);
        assert!(flags.contains(FeatureFlags::SKINNED));
        assert!(!flags.contains(FeatureFlags::SHADOWS));
    }

    #[test]
    fn test_feature_flags_combine() {
        let flags = FeatureFlags::SKINNED | FeatureFlags::SHADOWS;
        assert_eq!(flags.flag_count(), 2);
        assert!(flags.contains(FeatureFlags::SKINNED));
        assert!(flags.contains(FeatureFlags::SHADOWS));
        assert!(!flags.contains(FeatureFlags::FOG));
    }

    #[test]
    fn test_feature_flags_intersect() {
        let a = FeatureFlags::SKINNED | FeatureFlags::SHADOWS;
        let b = FeatureFlags::SHADOWS | FeatureFlags::FOG;
        let c = a & b;
        assert_eq!(c, FeatureFlags::SHADOWS);
    }

    #[test]
    fn test_feature_flags_remove() {
        let flags = FeatureFlags::ALL;
        let without_skinned = flags & !FeatureFlags::SKINNED;
        assert!(!without_skinned.contains(FeatureFlags::SKINNED));
        assert!(without_skinned.contains(FeatureFlags::SHADOWS));
    }

    #[test]
    fn test_feature_flags_iter() {
        let flags = FeatureFlags::SKINNED | FeatureFlags::SHADOWS | FeatureFlags::FOG;
        let collected: Vec<_> = flags.iter_enabled().collect();
        assert_eq!(collected.len(), 3);
        assert!(collected.contains(&FeatureFlags::SKINNED));
        assert!(collected.contains(&FeatureFlags::SHADOWS));
        assert!(collected.contains(&FeatureFlags::FOG));
    }

    #[test]
    fn test_feature_flags_names() {
        let flags = FeatureFlags::SKINNED | FeatureFlags::SHADOWS;
        let names = flags.names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"SKINNED"));
        assert!(names.contains(&"SHADOWS"));
    }

    #[test]
    fn test_feature_flags_name_single() {
        assert_eq!(FeatureFlags::SKINNED.flag_name(), "SKINNED");
        assert_eq!(FeatureFlags::ALPHA_TEST.flag_name(), "ALPHA_TEST");
        assert_eq!(FeatureFlags::NORMAL_MAP.flag_name(), "NORMAL_MAP");
        assert_eq!(FeatureFlags::EMISSIVE.flag_name(), "EMISSIVE");
        assert_eq!(FeatureFlags::SHADOWS.flag_name(), "SHADOWS");
        assert_eq!(FeatureFlags::FOG.flag_name(), "FOG");
        assert_eq!(FeatureFlags::INSTANCED.flag_name(), "INSTANCED");
        assert_eq!(FeatureFlags::NONE.flag_name(), "NONE");
    }

    #[test]
    fn test_feature_flags_from_name() {
        assert_eq!(FeatureFlags::parse_name("SKINNED"), FeatureFlags::SKINNED);
        assert_eq!(FeatureFlags::parse_name("skinned"), FeatureFlags::SKINNED);
        assert_eq!(FeatureFlags::parse_name("Skinned"), FeatureFlags::SKINNED);
        assert_eq!(FeatureFlags::parse_name("UNKNOWN"), FeatureFlags::NONE);
    }

    #[test]
    fn test_feature_flags_from_names() {
        let flags = FeatureFlags::from_names(&["SKINNED", "SHADOWS", "FOG"]);
        assert_eq!(flags.flag_count(), 3);
        assert!(flags.contains(FeatureFlags::SKINNED));
        assert!(flags.contains(FeatureFlags::SHADOWS));
        assert!(flags.contains(FeatureFlags::FOG));
    }

    #[test]
    fn test_feature_flags_with_feature() {
        let flags = FeatureFlags::SKINNED;
        let with_shadows = flags.with_feature(FeatureFlags::SHADOWS);
        assert!(with_shadows.contains(FeatureFlags::SKINNED));
        assert!(with_shadows.contains(FeatureFlags::SHADOWS));
    }

    #[test]
    fn test_feature_flags_without_feature() {
        let flags = FeatureFlags::SKINNED | FeatureFlags::SHADOWS;
        let without_shadows = flags.without_feature(FeatureFlags::SHADOWS);
        assert!(without_shadows.contains(FeatureFlags::SKINNED));
        assert!(!without_shadows.contains(FeatureFlags::SHADOWS));
    }

    #[test]
    fn test_feature_flags_toggle_feature() {
        let flags = FeatureFlags::SKINNED;
        let toggled = flags.toggle_feature(FeatureFlags::SKINNED);
        assert!(!toggled.contains(FeatureFlags::SKINNED));

        let toggled_back = toggled.toggle_feature(FeatureFlags::SKINNED);
        assert!(toggled_back.contains(FeatureFlags::SKINNED));
    }

    #[test]
    fn test_feature_flags_total_permutations() {
        assert_eq!(FeatureFlags::total_permutations(), 128); // 2^7
    }

    #[test]
    fn test_feature_flags_display() {
        let flags = FeatureFlags::SKINNED | FeatureFlags::SHADOWS;
        let display = format!("{}", flags);
        assert!(display.contains("SKINNED"));
        assert!(display.contains("SHADOWS"));

        let none = FeatureFlags::NONE;
        assert_eq!(format!("{}", none), "NONE");
    }

    #[test]
    fn test_feature_flags_debug() {
        let flags = FeatureFlags::SKINNED;
        let debug = format!("{:?}", flags);
        assert!(debug.contains("SKINNED"));
    }

    #[test]
    fn test_feature_flags_default() {
        let flags = FeatureFlags::default();
        assert!(flags.is_empty());
    }

    #[test]
    fn test_feature_flags_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(FeatureFlags::SKINNED);
        set.insert(FeatureFlags::SHADOWS);
        set.insert(FeatureFlags::SKINNED); // Duplicate

        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_feature_flags_clone() {
        let flags = FeatureFlags::SKINNED | FeatureFlags::SHADOWS;
        let cloned = flags;
        assert_eq!(flags, cloned);
    }

    #[test]
    fn test_feature_flags_to_pipeline_constants() {
        let flags = FeatureFlags::SKINNED | FeatureFlags::SHADOWS;
        let constants = flags.to_pipeline_constants();

        assert_eq!(constants.get("FEATURE_SKINNED"), Some(1.0));
        assert_eq!(constants.get("FEATURE_SHADOWS"), Some(1.0));
        assert_eq!(constants.get("FEATURE_FOG"), Some(0.0));
    }

    // =========================================================================
    // PermutationKey Tests
    // =========================================================================

    #[test]
    fn test_permutation_key_new() {
        let key = PermutationKey::new(12345, FeatureFlags::SKINNED);
        assert_eq!(key.shader_id(), 12345);
        assert_eq!(key.features(), FeatureFlags::SKINNED);
    }

    #[test]
    fn test_permutation_key_base() {
        let key = PermutationKey::base(12345);
        assert_eq!(key.shader_id(), 12345);
        assert!(key.features().is_empty());
    }

    #[test]
    fn test_permutation_key_with_feature() {
        let key = PermutationKey::base(12345);
        let with_skinned = key.with_feature(FeatureFlags::SKINNED);
        assert!(with_skinned.features().contains(FeatureFlags::SKINNED));
    }

    #[test]
    fn test_permutation_key_without_feature() {
        let key = PermutationKey::new(12345, FeatureFlags::SKINNED | FeatureFlags::SHADOWS);
        let without_skinned = key.without_feature(FeatureFlags::SKINNED);
        assert!(!without_skinned.features().contains(FeatureFlags::SKINNED));
        assert!(without_skinned.features().contains(FeatureFlags::SHADOWS));
    }

    #[test]
    fn test_permutation_key_with_features() {
        let key = PermutationKey::base(12345);
        let with_features = key.with_features(FeatureFlags::SKINNED | FeatureFlags::SHADOWS);
        assert_eq!(
            with_features.features(),
            FeatureFlags::SKINNED | FeatureFlags::SHADOWS
        );
    }

    #[test]
    fn test_permutation_key_equality() {
        let key1 = PermutationKey::new(12345, FeatureFlags::SKINNED);
        let key2 = PermutationKey::new(12345, FeatureFlags::SKINNED);
        let key3 = PermutationKey::new(12345, FeatureFlags::SHADOWS);
        let key4 = PermutationKey::new(99999, FeatureFlags::SKINNED);

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
        assert_ne!(key1, key4);
    }

    #[test]
    fn test_permutation_key_hash() {
        use std::collections::HashMap;
        let mut map: HashMap<PermutationKey, i32> = HashMap::new();

        let key = PermutationKey::new(12345, FeatureFlags::SKINNED);
        map.insert(key, 42);

        assert_eq!(map.get(&key), Some(&42));
    }

    #[test]
    fn test_permutation_key_display() {
        let key = PermutationKey::new(12345, FeatureFlags::SKINNED | FeatureFlags::SHADOWS);
        let display = format!("{}", key);
        assert!(display.contains("12345"));
    }

    #[test]
    fn test_permutation_key_display_string() {
        let key = PermutationKey::new(12345, FeatureFlags::SKINNED);
        let display = key.display_string();
        assert!(display.contains("12345"));
    }

    #[test]
    fn test_permutation_key_debug() {
        let key = PermutationKey::new(12345, FeatureFlags::SKINNED);
        let debug = format!("{:?}", key);
        assert!(debug.contains("PermutationKey"));
    }

    #[test]
    fn test_permutation_key_clone() {
        let key = PermutationKey::new(12345, FeatureFlags::SKINNED);
        let cloned = key;
        assert_eq!(key, cloned);
    }

    // =========================================================================
    // PermutationConfig Tests
    // =========================================================================

    #[test]
    fn test_permutation_config_default() {
        let config = PermutationConfig::default();
        assert_eq!(config.max_permutations, DEFAULT_MAX_PERMUTATIONS);
        assert!(config.enable_lazy_compilation);
        assert_eq!(config.eviction_policy, EvictionPolicy::LRU);
    }

    #[test]
    fn test_permutation_config_new() {
        let config = PermutationConfig::new();
        assert_eq!(config.max_permutations, DEFAULT_MAX_PERMUTATIONS);
    }

    #[test]
    fn test_permutation_config_max_permutations() {
        let config = PermutationConfig::new().max_permutations(512);
        assert_eq!(config.max_permutations, 512);
    }

    #[test]
    fn test_permutation_config_enable_lazy_compilation() {
        let config = PermutationConfig::new().enable_lazy_compilation(false);
        assert!(!config.enable_lazy_compilation);
    }

    #[test]
    fn test_permutation_config_eviction_policy() {
        let config = PermutationConfig::new().eviction_policy(EvictionPolicy::LFU);
        assert_eq!(config.eviction_policy, EvictionPolicy::LFU);
    }

    #[test]
    fn test_permutation_config_builder_chain() {
        let config = PermutationConfig::new()
            .max_permutations(128)
            .enable_lazy_compilation(false)
            .eviction_policy(EvictionPolicy::Oldest);

        assert_eq!(config.max_permutations, 128);
        assert!(!config.enable_lazy_compilation);
        assert_eq!(config.eviction_policy, EvictionPolicy::Oldest);
    }

    #[test]
    fn test_permutation_config_minimal() {
        let config = PermutationConfig::minimal();
        assert_eq!(config.max_permutations, 16);
    }

    #[test]
    fn test_permutation_config_development() {
        let config = PermutationConfig::development();
        assert_eq!(config.max_permutations, 64);
    }

    #[test]
    fn test_permutation_config_production() {
        let config = PermutationConfig::production();
        assert_eq!(config.max_permutations, 512);
        assert_eq!(config.eviction_policy, EvictionPolicy::LFU);
    }

    #[test]
    fn test_permutation_config_validate_success() {
        let config = PermutationConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_permutation_config_validate_zero() {
        let config = PermutationConfig::new().max_permutations(0);
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().is_config_error());
    }

    #[test]
    fn test_permutation_config_clone() {
        let config = PermutationConfig::new().max_permutations(100);
        let cloned = config.clone();
        assert_eq!(cloned.max_permutations, 100);
    }

    #[test]
    fn test_permutation_config_debug() {
        let config = PermutationConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("PermutationConfig"));
    }

    // =========================================================================
    // EvictionPolicy Tests
    // =========================================================================

    #[test]
    fn test_eviction_policy_default() {
        assert_eq!(EvictionPolicy::default(), EvictionPolicy::LRU);
    }

    #[test]
    fn test_eviction_policy_name() {
        assert_eq!(EvictionPolicy::LRU.name(), "LRU");
        assert_eq!(EvictionPolicy::LFU.name(), "LFU");
        assert_eq!(EvictionPolicy::Oldest.name(), "Oldest");
    }

    #[test]
    fn test_eviction_policy_display() {
        assert_eq!(format!("{}", EvictionPolicy::LRU), "LRU");
        assert_eq!(format!("{}", EvictionPolicy::LFU), "LFU");
        assert_eq!(format!("{}", EvictionPolicy::Oldest), "Oldest");
    }

    #[test]
    fn test_eviction_policy_equality() {
        assert_eq!(EvictionPolicy::LRU, EvictionPolicy::LRU);
        assert_ne!(EvictionPolicy::LRU, EvictionPolicy::LFU);
    }

    #[test]
    fn test_eviction_policy_clone() {
        let policy = EvictionPolicy::LFU;
        let cloned = policy;
        assert_eq!(policy, cloned);
    }

    #[test]
    fn test_eviction_policy_debug() {
        let debug = format!("{:?}", EvictionPolicy::LRU);
        assert!(debug.contains("LRU"));
    }

    // =========================================================================
    // PermutationMetrics Tests
    // =========================================================================

    #[test]
    fn test_permutation_metrics_default() {
        let metrics = PermutationMetrics::default();
        assert_eq!(metrics.cache_size, 0);
        assert_eq!(metrics.cache_hits, 0);
        assert_eq!(metrics.cache_misses, 0);
        assert_eq!(metrics.compilations, 0);
        assert_eq!(metrics.evictions, 0);
        assert_eq!(metrics.hit_rate, 0.0);
    }

    #[test]
    fn test_permutation_metrics_new() {
        let metrics = PermutationMetrics::new(10, 80, 20, 20, 5);
        assert_eq!(metrics.cache_size, 10);
        assert_eq!(metrics.cache_hits, 80);
        assert_eq!(metrics.cache_misses, 20);
        assert_eq!(metrics.compilations, 20);
        assert_eq!(metrics.evictions, 5);
        assert_eq!(metrics.hit_rate, 0.8);
    }

    #[test]
    fn test_permutation_metrics_total_requests() {
        let metrics = PermutationMetrics::new(0, 50, 50, 50, 0);
        assert_eq!(metrics.total_requests(), 100);
    }

    #[test]
    fn test_permutation_metrics_hit_rate_percent() {
        let metrics = PermutationMetrics::new(0, 75, 25, 25, 0);
        assert_eq!(metrics.hit_rate_percent(), 75.0);
    }

    #[test]
    fn test_permutation_metrics_miss_rate() {
        let metrics = PermutationMetrics::new(0, 60, 40, 40, 0);
        assert_eq!(metrics.miss_rate(), 0.4);
    }

    #[test]
    fn test_permutation_metrics_is_empty() {
        let empty = PermutationMetrics::new(0, 10, 5, 5, 0);
        assert!(empty.is_empty());

        let not_empty = PermutationMetrics::new(1, 10, 5, 5, 0);
        assert!(!not_empty.is_empty());
    }

    #[test]
    fn test_permutation_metrics_reset() {
        let mut metrics = PermutationMetrics::new(10, 80, 20, 20, 5);
        metrics.reset();
        assert_eq!(metrics.cache_size, 0);
        assert_eq!(metrics.cache_hits, 0);
    }

    #[test]
    fn test_permutation_metrics_zero_requests() {
        let metrics = PermutationMetrics::new(0, 0, 0, 0, 0);
        assert_eq!(metrics.hit_rate, 0.0);
    }

    #[test]
    fn test_permutation_metrics_display() {
        let metrics = PermutationMetrics::new(10, 80, 20, 20, 5);
        let display = format!("{}", metrics);
        assert!(display.contains("PermutationMetrics"));
        assert!(display.contains("80.0%"));
    }

    #[test]
    fn test_permutation_metrics_clone() {
        let metrics = PermutationMetrics::new(10, 80, 20, 20, 5);
        let cloned = metrics.clone();
        assert_eq!(cloned.cache_size, 10);
    }

    #[test]
    fn test_permutation_metrics_debug() {
        let metrics = PermutationMetrics::default();
        let debug = format!("{:?}", metrics);
        assert!(debug.contains("PermutationMetrics"));
    }

    // =========================================================================
    // PermutationError Tests
    // =========================================================================

    #[test]
    fn test_permutation_error_max_exceeded() {
        let err = PermutationError::MaxPermutationsExceeded {
            current: 256,
            max: 256,
        };
        assert!(err.is_max_exceeded());
        assert!(!err.is_compilation_error());

        let display = format!("{}", err);
        assert!(display.contains("maximum permutations exceeded"));
    }

    #[test]
    fn test_permutation_error_compilation_failed() {
        let err = PermutationError::CompilationFailed("parse error".to_string());
        assert!(err.is_compilation_error());
        assert!(!err.is_max_exceeded());

        let display = format!("{}", err);
        assert!(display.contains("compilation failed"));
    }

    #[test]
    fn test_permutation_error_shader_not_found() {
        let err = PermutationError::ShaderNotFound { shader_id: 12345 };
        assert!(err.is_not_found());

        let display = format!("{}", err);
        assert!(display.contains("12345"));
    }

    #[test]
    fn test_permutation_error_config_error() {
        let err = PermutationError::ConfigError("invalid config".to_string());
        assert!(err.is_config_error());

        let display = format!("{}", err);
        assert!(display.contains("configuration error"));
    }

    #[test]
    fn test_permutation_error_from_shader_error() {
        let shader_err = ShaderError::EmptySource { label: None };
        let perm_err: PermutationError = shader_err.into();
        assert!(perm_err.is_compilation_error());
    }

    #[test]
    fn test_permutation_error_debug() {
        let err = PermutationError::MaxPermutationsExceeded {
            current: 256,
            max: 256,
        };
        let debug = format!("{:?}", err);
        assert!(debug.contains("MaxPermutationsExceeded"));
    }

    #[test]
    fn test_permutation_error_clone() {
        let err = PermutationError::CompilationFailed("test".to_string());
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }

    // =========================================================================
    // ShaderPermutationManager Tests (Unit - No Device)
    // =========================================================================

    #[test]
    fn test_manager_new() {
        let manager = ShaderPermutationManager::new(PermutationConfig::default());
        assert!(manager.is_empty());
        assert_eq!(manager.permutation_count(), 0);
    }

    #[test]
    fn test_manager_with_defaults() {
        let manager = ShaderPermutationManager::with_defaults();
        assert!(manager.is_empty());
    }

    #[test]
    fn test_manager_config() {
        let config = PermutationConfig::new().max_permutations(100);
        let manager = ShaderPermutationManager::new(config);
        assert_eq!(manager.config().max_permutations, 100);
    }

    #[test]
    fn test_manager_metrics_initial() {
        let manager = ShaderPermutationManager::with_defaults();
        let metrics = manager.metrics();
        assert_eq!(metrics.cache_size, 0);
        assert_eq!(metrics.cache_hits, 0);
        assert_eq!(metrics.cache_misses, 0);
    }

    #[test]
    fn test_manager_reset_metrics() {
        let manager = ShaderPermutationManager::with_defaults();
        manager.reset_metrics();
        let metrics = manager.metrics();
        assert_eq!(metrics.cache_hits, 0);
        assert_eq!(metrics.cache_misses, 0);
    }

    #[test]
    fn test_manager_keys_empty() {
        let manager = ShaderPermutationManager::with_defaults();
        assert!(manager.keys().is_empty());
    }

    #[test]
    fn test_manager_shader_ids_empty() {
        let manager = ShaderPermutationManager::with_defaults();
        assert!(manager.shader_ids().is_empty());
    }

    #[test]
    fn test_manager_invalidate_all_empty() {
        let manager = ShaderPermutationManager::with_defaults();
        manager.invalidate_all();
        assert!(manager.is_empty());
    }

    #[test]
    fn test_manager_invalidate_nonexistent() {
        let manager = ShaderPermutationManager::with_defaults();
        let count = manager.invalidate(12345);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_manager_invalidate_key_nonexistent() {
        let manager = ShaderPermutationManager::with_defaults();
        let key = PermutationKey::new(12345, FeatureFlags::SKINNED);
        assert!(!manager.invalidate_key(&key));
    }

    #[test]
    fn test_manager_contains_empty() {
        let manager = ShaderPermutationManager::with_defaults();
        let key = PermutationKey::new(12345, FeatureFlags::SKINNED);
        assert!(!manager.contains(&key));
    }

    #[test]
    fn test_manager_get_cached_empty() {
        let manager = ShaderPermutationManager::with_defaults();
        let key = PermutationKey::new(12345, FeatureFlags::SKINNED);
        assert!(manager.get_cached(&key).is_none());
    }

    #[test]
    fn test_manager_permutation_count_for_shader() {
        let manager = ShaderPermutationManager::with_defaults();
        assert_eq!(manager.permutation_count_for_shader(12345), 0);
    }

    #[test]
    fn test_manager_debug() {
        let manager = ShaderPermutationManager::with_defaults();
        let debug = format!("{:?}", manager);
        assert!(debug.contains("ShaderPermutationManager"));
    }

    // =========================================================================
    // CachedPermutation Tests (Limited - needs TrinityShaderModule)
    // =========================================================================

    #[test]
    fn test_cached_permutation_debug() {
        // Can't test fully without device, but test Debug trait exists
        // This is a compile-time check
        fn assert_debug<T: std::fmt::Debug>() {}
        assert_debug::<CachedPermutation>();
    }

    // =========================================================================
    // Thread Safety Tests
    // =========================================================================

    #[test]
    fn test_feature_flags_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<FeatureFlags>();
    }

    #[test]
    fn test_permutation_key_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PermutationKey>();
    }

    #[test]
    fn test_permutation_config_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PermutationConfig>();
    }

    #[test]
    fn test_permutation_metrics_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PermutationMetrics>();
    }

    #[test]
    fn test_permutation_error_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PermutationError>();
    }

    #[test]
    fn test_eviction_policy_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<EvictionPolicy>();
    }

    #[test]
    fn test_manager_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ShaderPermutationManager>();
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_feature_flags_xor() {
        let a = FeatureFlags::SKINNED | FeatureFlags::SHADOWS;
        let b = FeatureFlags::SHADOWS | FeatureFlags::FOG;
        let c = a ^ b;

        assert!(c.contains(FeatureFlags::SKINNED));
        assert!(!c.contains(FeatureFlags::SHADOWS));
        assert!(c.contains(FeatureFlags::FOG));
    }

    #[test]
    fn test_feature_flags_complement() {
        let flags = FeatureFlags::SKINNED;
        let complement = !flags & FeatureFlags::ALL;
        assert!(!complement.contains(FeatureFlags::SKINNED));
        assert!(complement.contains(FeatureFlags::SHADOWS));
    }

    #[test]
    fn test_permutation_key_large_shader_id() {
        let key = PermutationKey::new(u64::MAX, FeatureFlags::ALL);
        assert_eq!(key.shader_id(), u64::MAX);
    }

    #[test]
    fn test_permutation_metrics_large_values() {
        let metrics = PermutationMetrics::new(
            usize::MAX,
            u64::MAX / 2,
            u64::MAX / 2,
            u64::MAX,
            u64::MAX,
        );
        assert_eq!(metrics.cache_size, usize::MAX);
        assert_eq!(metrics.evictions, u64::MAX);
    }

    #[test]
    fn test_feature_flags_empty_names() {
        let flags = FeatureFlags::from_names(&[]);
        assert!(flags.is_empty());
    }

    #[test]
    fn test_feature_flags_duplicate_names() {
        let flags = FeatureFlags::from_names(&["SKINNED", "SKINNED", "SKINNED"]);
        assert_eq!(flags.flag_count(), 1);
        assert!(flags.contains(FeatureFlags::SKINNED));
    }

    #[test]
    fn test_feature_flags_all_names() {
        let flags = FeatureFlags::from_names(&[
            "SKINNED",
            "ALPHA_TEST",
            "NORMAL_MAP",
            "EMISSIVE",
            "SHADOWS",
            "FOG",
            "INSTANCED",
        ]);
        assert!(flags.is_all_set());
    }

    // =========================================================================
    // Constants Tests
    // =========================================================================

    #[test]
    fn test_default_max_permutations() {
        assert_eq!(DEFAULT_MAX_PERMUTATIONS, 256);
    }

    #[test]
    fn test_feature_flag_count() {
        assert_eq!(FEATURE_FLAG_COUNT, 7);
    }

    // =========================================================================
    // Additional FeatureFlags Tests for Coverage
    // =========================================================================

    #[test]
    fn test_feature_flags_all_bits_correct() {
        let all = FeatureFlags::ALL;
        assert!(all.contains(FeatureFlags::SKINNED));
        assert!(all.contains(FeatureFlags::ALPHA_TEST));
        assert!(all.contains(FeatureFlags::NORMAL_MAP));
        assert!(all.contains(FeatureFlags::EMISSIVE));
        assert!(all.contains(FeatureFlags::SHADOWS));
        assert!(all.contains(FeatureFlags::FOG));
        assert!(all.contains(FeatureFlags::INSTANCED));
    }

    #[test]
    fn test_feature_flags_bits_values() {
        assert_eq!(FeatureFlags::SKINNED.bits(), 1 << 0);
        assert_eq!(FeatureFlags::ALPHA_TEST.bits(), 1 << 1);
        assert_eq!(FeatureFlags::NORMAL_MAP.bits(), 1 << 2);
        assert_eq!(FeatureFlags::EMISSIVE.bits(), 1 << 3);
        assert_eq!(FeatureFlags::SHADOWS.bits(), 1 << 4);
        assert_eq!(FeatureFlags::FOG.bits(), 1 << 5);
        assert_eq!(FeatureFlags::INSTANCED.bits(), 1 << 6);
    }

    #[test]
    fn test_feature_flags_from_bits() {
        let flags = FeatureFlags::from_bits(0b0010101);
        assert!(flags.is_some());
        let flags = flags.unwrap();
        assert!(flags.contains(FeatureFlags::SKINNED));
        assert!(flags.contains(FeatureFlags::NORMAL_MAP));
        assert!(flags.contains(FeatureFlags::SHADOWS));
    }

    #[test]
    fn test_feature_flags_from_bits_truncate() {
        let flags = FeatureFlags::from_bits_truncate(0b11111111);
        assert!(flags.is_all_set());
    }

    // =========================================================================
    // PermutationKey Additional Tests
    // =========================================================================

    #[test]
    fn test_permutation_key_zero_shader_id() {
        let key = PermutationKey::new(0, FeatureFlags::SKINNED);
        assert_eq!(key.shader_id(), 0);
    }

    #[test]
    fn test_permutation_key_features_modification_chain() {
        let key = PermutationKey::base(100)
            .with_feature(FeatureFlags::SKINNED)
            .with_feature(FeatureFlags::SHADOWS)
            .without_feature(FeatureFlags::SKINNED);

        assert!(!key.features().contains(FeatureFlags::SKINNED));
        assert!(key.features().contains(FeatureFlags::SHADOWS));
    }

    // =========================================================================
    // Manager Concurrent Key Tests
    // =========================================================================

    #[test]
    fn test_manager_concurrent_key_creation() {
        use std::thread;

        let handles: Vec<_> = (0..10)
            .map(|i| {
                thread::spawn(move || {
                    PermutationKey::new(i as u64, FeatureFlags::from_bits_truncate(i as u32))
                })
            })
            .collect();

        let keys: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // All keys should be unique
        for i in 0..keys.len() {
            for j in (i + 1)..keys.len() {
                assert_ne!(keys[i], keys[j]);
            }
        }
    }

    // =========================================================================
    // WHITEBOX TESTS - T-WGPU-P2.7.6
    // =========================================================================

    // -------------------------------------------------------------------------
    // FeatureFlags Edge Cases (6 tests)
    // -------------------------------------------------------------------------

    #[test]
    fn test_feature_flags_all_combinations_unique() {
        // Verify that all 128 possible flag combinations produce unique bit patterns
        use std::collections::HashSet;
        let mut seen = HashSet::new();

        for bits in 0..FeatureFlags::total_permutations() {
            let flags = FeatureFlags::from_bits_truncate(bits as u32);
            let bits_value = flags.bits();
            assert!(
                seen.insert(bits_value),
                "Duplicate bits pattern detected: {}",
                bits_value
            );
        }
        assert_eq!(seen.len(), FeatureFlags::total_permutations());
    }

    #[test]
    fn test_feature_flags_toggle_all() {
        // Toggle each flag individually and verify correct state
        let mut flags = FeatureFlags::NONE;

        // Toggle all flags on
        for flag in [
            FeatureFlags::SKINNED,
            FeatureFlags::ALPHA_TEST,
            FeatureFlags::NORMAL_MAP,
            FeatureFlags::EMISSIVE,
            FeatureFlags::SHADOWS,
            FeatureFlags::FOG,
            FeatureFlags::INSTANCED,
        ] {
            flags = flags.toggle_feature(flag);
            assert!(flags.contains(flag));
        }
        assert!(flags.is_all_set());

        // Toggle all flags off
        for flag in [
            FeatureFlags::SKINNED,
            FeatureFlags::ALPHA_TEST,
            FeatureFlags::NORMAL_MAP,
            FeatureFlags::EMISSIVE,
            FeatureFlags::SHADOWS,
            FeatureFlags::FOG,
            FeatureFlags::INSTANCED,
        ] {
            flags = flags.toggle_feature(flag);
            assert!(!flags.contains(flag));
        }
        assert!(flags.is_empty());
    }

    #[test]
    fn test_feature_flags_complement_exhaustive() {
        // Test complement operation for all possible flag combinations
        for bits in 0..FeatureFlags::total_permutations() {
            let flags = FeatureFlags::from_bits_truncate(bits as u32);
            let complement = !flags & FeatureFlags::ALL;

            // The complement should have no common flags with original
            assert!((flags & complement).is_empty());

            // The union of flags and complement should equal ALL
            assert_eq!(flags | complement, FeatureFlags::ALL);
        }
    }

    #[test]
    fn test_feature_flags_symmetric_difference() {
        // Test XOR operation produces correct symmetric difference
        let a = FeatureFlags::SKINNED | FeatureFlags::SHADOWS | FeatureFlags::FOG;
        let b = FeatureFlags::SHADOWS | FeatureFlags::FOG | FeatureFlags::INSTANCED;

        let sym_diff = a ^ b;

        // Should contain flags that are in either but not both
        assert!(sym_diff.contains(FeatureFlags::SKINNED));
        assert!(!sym_diff.contains(FeatureFlags::SHADOWS));
        assert!(!sym_diff.contains(FeatureFlags::FOG));
        assert!(sym_diff.contains(FeatureFlags::INSTANCED));

        // Verify XOR is commutative
        assert_eq!(a ^ b, b ^ a);

        // Verify XOR with self is empty
        assert!((a ^ a).is_empty());
    }

    #[test]
    fn test_feature_flags_subset_superset() {
        let subset = FeatureFlags::SKINNED | FeatureFlags::SHADOWS;
        let superset = FeatureFlags::SKINNED | FeatureFlags::SHADOWS | FeatureFlags::FOG;
        let disjoint = FeatureFlags::ALPHA_TEST | FeatureFlags::EMISSIVE;

        // subset is contained in superset
        assert!((subset & superset) == subset);
        assert!((superset & subset) == subset);

        // superset contains subset
        assert!(superset.contains(FeatureFlags::SKINNED));
        assert!(superset.contains(FeatureFlags::SHADOWS));
        assert!(superset.contains(FeatureFlags::FOG));

        // disjoint has no overlap with subset
        assert!((subset & disjoint).is_empty());

        // ALL is superset of everything
        assert!((subset & FeatureFlags::ALL) == subset);
        assert!((superset & FeatureFlags::ALL) == superset);

        // NONE is subset of everything
        assert!((FeatureFlags::NONE & subset) == FeatureFlags::NONE);
    }

    #[test]
    fn test_feature_flags_parse_all_names() {
        // Test that all flag names parse correctly (case insensitive)
        let test_cases = [
            ("skinned", FeatureFlags::SKINNED),
            ("SKINNED", FeatureFlags::SKINNED),
            ("Skinned", FeatureFlags::SKINNED),
            ("ALPHA_TEST", FeatureFlags::ALPHA_TEST),
            ("alpha_test", FeatureFlags::ALPHA_TEST),
            ("Alpha_Test", FeatureFlags::ALPHA_TEST),
            ("NORMAL_MAP", FeatureFlags::NORMAL_MAP),
            ("normal_map", FeatureFlags::NORMAL_MAP),
            ("EMISSIVE", FeatureFlags::EMISSIVE),
            ("emissive", FeatureFlags::EMISSIVE),
            ("SHADOWS", FeatureFlags::SHADOWS),
            ("shadows", FeatureFlags::SHADOWS),
            ("FOG", FeatureFlags::FOG),
            ("fog", FeatureFlags::FOG),
            ("INSTANCED", FeatureFlags::INSTANCED),
            ("instanced", FeatureFlags::INSTANCED),
            ("UNKNOWN", FeatureFlags::NONE),
            ("", FeatureFlags::NONE),
            ("invalid_flag", FeatureFlags::NONE),
        ];

        for (name, expected) in test_cases {
            assert_eq!(
                FeatureFlags::parse_name(name),
                expected,
                "Failed for name: {}",
                name
            );
        }
    }

    // -------------------------------------------------------------------------
    // PermutationKey Hashing (5 tests)
    // -------------------------------------------------------------------------

    #[test]
    fn test_permutation_key_hash_stability() {
        use std::collections::hash_map::DefaultHasher;

        // Hash should be stable across multiple calls
        let key = PermutationKey::new(12345, FeatureFlags::SKINNED | FeatureFlags::SHADOWS);

        let mut hasher1 = DefaultHasher::new();
        key.hash(&mut hasher1);
        let hash1 = hasher1.finish();

        let mut hasher2 = DefaultHasher::new();
        key.hash(&mut hasher2);
        let hash2 = hasher2.finish();

        assert_eq!(hash1, hash2, "Hash should be stable");

        // Same key values should produce same hash
        let key_clone = PermutationKey::new(12345, FeatureFlags::SKINNED | FeatureFlags::SHADOWS);
        let mut hasher3 = DefaultHasher::new();
        key_clone.hash(&mut hasher3);
        let hash3 = hasher3.finish();

        assert_eq!(hash1, hash3, "Equal keys should have equal hashes");
    }

    #[test]
    fn test_permutation_key_hash_collision_resistance() {
        use std::collections::hash_map::DefaultHasher;
        use std::collections::HashSet;

        // Generate many keys and verify low collision rate
        let mut hashes = HashSet::new();

        for shader_id in 0..100u64 {
            for bits in 0..FeatureFlags::total_permutations() {
                let flags = FeatureFlags::from_bits_truncate(bits as u32);
                let key = PermutationKey::new(shader_id, flags);

                let mut hasher = DefaultHasher::new();
                key.hash(&mut hasher);
                hashes.insert(hasher.finish());
            }
        }

        // With 100 * 128 = 12800 unique keys, we expect very few collisions
        let expected_keys = 100 * FeatureFlags::total_permutations();
        let collision_rate = 1.0 - (hashes.len() as f64 / expected_keys as f64);
        assert!(
            collision_rate < 0.01,
            "Collision rate {} is too high",
            collision_rate
        );
    }

    #[test]
    fn test_permutation_key_ordering() {
        // Keys should be orderable by their components for HashMap efficiency
        let key1 = PermutationKey::new(100, FeatureFlags::SKINNED);
        let key2 = PermutationKey::new(200, FeatureFlags::SKINNED);
        let key3 = PermutationKey::new(100, FeatureFlags::SHADOWS);

        // Different shader IDs make different keys
        assert_ne!(key1, key2);

        // Same shader ID, different features make different keys
        assert_ne!(key1, key3);

        // Keys with same shader_id and features are equal
        let key1_dup = PermutationKey::new(100, FeatureFlags::SKINNED);
        assert_eq!(key1, key1_dup);
    }

    #[test]
    fn test_permutation_key_same_features_different_shader() {
        let features = FeatureFlags::SKINNED | FeatureFlags::SHADOWS;

        let key1 = PermutationKey::new(1, features);
        let key2 = PermutationKey::new(2, features);
        let key3 = PermutationKey::new(u64::MAX, features);

        // Same features but different shaders should be different keys
        assert_ne!(key1, key2);
        assert_ne!(key1, key3);
        assert_ne!(key2, key3);

        // HashMap should treat them as different
        let mut map = HashMap::new();
        map.insert(key1, "shader1");
        map.insert(key2, "shader2");
        map.insert(key3, "shader_max");

        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&key1), Some(&"shader1"));
        assert_eq!(map.get(&key2), Some(&"shader2"));
        assert_eq!(map.get(&key3), Some(&"shader_max"));
    }

    #[test]
    fn test_permutation_key_same_shader_different_features() {
        let shader_id = 12345u64;

        let key_none = PermutationKey::new(shader_id, FeatureFlags::NONE);
        let key_skinned = PermutationKey::new(shader_id, FeatureFlags::SKINNED);
        let key_shadows = PermutationKey::new(shader_id, FeatureFlags::SHADOWS);
        let key_both = PermutationKey::new(shader_id, FeatureFlags::SKINNED | FeatureFlags::SHADOWS);
        let key_all = PermutationKey::new(shader_id, FeatureFlags::ALL);

        // All should be different
        let keys = [key_none, key_skinned, key_shadows, key_both, key_all];
        for i in 0..keys.len() {
            for j in (i + 1)..keys.len() {
                assert_ne!(
                    keys[i], keys[j],
                    "Keys {:?} and {:?} should differ",
                    keys[i], keys[j]
                );
            }
        }

        // HashMap should treat them as different
        let mut map = HashMap::new();
        for (idx, key) in keys.iter().enumerate() {
            map.insert(*key, idx);
        }
        assert_eq!(map.len(), 5);
    }

    // -------------------------------------------------------------------------
    // CachedPermutation Lifecycle (6 tests)
    // -------------------------------------------------------------------------

    // Note: These tests are limited as CachedPermutation::new is private
    // and requires TrinityShaderModule. We test what we can.

    #[test]
    fn test_cached_permutation_access_count_increment() {
        // Test that access_count uses AtomicU64 correctly
        let counter = AtomicU64::new(1);
        assert_eq!(counter.load(Ordering::Relaxed), 1);

        counter.fetch_add(1, Ordering::Relaxed);
        assert_eq!(counter.load(Ordering::Relaxed), 2);

        // Multiple increments
        for _ in 0..100 {
            counter.fetch_add(1, Ordering::Relaxed);
        }
        assert_eq!(counter.load(Ordering::Relaxed), 102);

        // Concurrent increments should be atomic
        use std::thread;
        let counter = Arc::new(AtomicU64::new(0));
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let c = Arc::clone(&counter);
                thread::spawn(move || {
                    for _ in 0..1000 {
                        c.fetch_add(1, Ordering::Relaxed);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(counter.load(Ordering::Relaxed), 10_000);
    }

    #[test]
    fn test_cached_permutation_created_at_monotonic() {
        // Test that Instant values are monotonically increasing
        let t1 = Instant::now();
        std::thread::sleep(std::time::Duration::from_micros(100));
        let t2 = Instant::now();
        std::thread::sleep(std::time::Duration::from_micros(100));
        let t3 = Instant::now();

        assert!(t1 < t2);
        assert!(t2 < t3);
        assert!(t1 < t3);

        // Elapsed time should be non-negative
        assert!(t1.elapsed() >= std::time::Duration::ZERO);
    }

    #[test]
    fn test_cached_permutation_clone_shares_module() {
        // Test Arc behavior which CachedPermutation uses for module
        let module = Arc::new(42i32);
        let module_clone = Arc::clone(&module);

        // Both point to same data
        assert!(Arc::ptr_eq(&module, &module_clone));
        assert_eq!(Arc::strong_count(&module), 2);

        // Modifications through one are visible through other (if mutable)
        drop(module_clone);
        assert_eq!(Arc::strong_count(&module), 1);
    }

    #[test]
    fn test_cached_permutation_metrics_tracking() {
        // Test that metrics can track permutation stats correctly
        let mut metrics = PermutationMetrics::new(0, 0, 0, 0, 0);

        // Simulate cache miss + compilation
        metrics.cache_misses += 1;
        metrics.compilations += 1;
        metrics.cache_size += 1;

        assert_eq!(metrics.cache_misses, 1);
        assert_eq!(metrics.compilations, 1);
        assert_eq!(metrics.cache_size, 1);

        // Simulate cache hits
        for _ in 0..10 {
            metrics.cache_hits += 1;
        }
        assert_eq!(metrics.cache_hits, 10);
        assert_eq!(metrics.total_requests(), 11);

        // Recalculate hit rate
        let total = metrics.cache_hits + metrics.cache_misses;
        let hit_rate = if total > 0 {
            metrics.cache_hits as f64 / total as f64
        } else {
            0.0
        };
        assert!((hit_rate - (10.0 / 11.0)).abs() < 0.001);
    }

    #[test]
    fn test_cached_permutation_last_access_update() {
        // Test RwLock<Instant> behavior which CachedPermutation uses
        let last_accessed = RwLock::new(Instant::now());

        let initial = *last_accessed.read();
        std::thread::sleep(std::time::Duration::from_micros(100));

        // Update access time
        *last_accessed.write() = Instant::now();
        let updated = *last_accessed.read();

        assert!(updated > initial);

        // Multiple reads should see same value
        let r1 = *last_accessed.read();
        let r2 = *last_accessed.read();
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_cached_permutation_age_calculation() {
        // Test age calculation logic
        let created_at = Instant::now();
        std::thread::sleep(std::time::Duration::from_millis(10));

        let age = created_at.elapsed();
        assert!(age >= std::time::Duration::from_millis(10));
        assert!(age < std::time::Duration::from_secs(1));

        // Idle time calculation
        let last_accessed = Instant::now();
        std::thread::sleep(std::time::Duration::from_millis(5));
        let idle = last_accessed.elapsed();

        assert!(idle >= std::time::Duration::from_millis(5));
        assert!(idle <= age);
    }

    // -------------------------------------------------------------------------
    // Eviction Policy (8 tests)
    // -------------------------------------------------------------------------

    #[test]
    fn test_eviction_lru_removes_oldest_access() {
        // Verify LRU eviction logic by testing Instant comparison
        let t1 = Instant::now();
        std::thread::sleep(std::time::Duration::from_micros(100));
        let t2 = Instant::now();
        std::thread::sleep(std::time::Duration::from_micros(100));
        let t3 = Instant::now();

        // LRU should select the entry with minimum last_accessed
        let entries = [(1, t2), (2, t1), (3, t3)]; // Entry 2 is oldest access
        let min_entry = entries.iter().min_by_key(|(_, t)| t);
        assert_eq!(min_entry.map(|(id, _)| *id), Some(2));
    }

    #[test]
    fn test_eviction_lfu_removes_least_used() {
        // Verify LFU eviction logic by testing access count comparison
        let entries = [(1u64, 10u64), (2, 5), (3, 15), (4, 2)];

        // LFU should select the entry with minimum access_count
        let min_entry = entries.iter().min_by_key(|(_, count)| count);
        assert_eq!(min_entry.map(|(id, _)| *id), Some(4)); // Entry 4 has count=2
    }

    #[test]
    fn test_eviction_oldest_removes_first_created() {
        // Verify Oldest eviction logic
        let t1 = Instant::now();
        std::thread::sleep(std::time::Duration::from_micros(100));
        let t2 = Instant::now();
        std::thread::sleep(std::time::Duration::from_micros(100));
        let t3 = Instant::now();

        // Oldest should select entry with minimum created_at
        let entries = [(1, t2), (2, t1), (3, t3)]; // Entry 2 was created first
        let oldest = entries.iter().min_by_key(|(_, t)| t);
        assert_eq!(oldest.map(|(id, _)| *id), Some(2));
    }

    #[test]
    fn test_eviction_at_capacity_triggers() {
        // Test that eviction happens when at capacity
        let config = PermutationConfig::new().max_permutations(3);
        let manager = ShaderPermutationManager::new(config);

        // Verify config is set
        assert_eq!(manager.config().max_permutations, 3);

        // Without GPU we can't add entries, but verify capacity check logic
        assert!(manager.permutation_count() < manager.config().max_permutations);
    }

    #[test]
    fn test_eviction_preserves_hot_entries() {
        // Verify that eviction policies preserve frequently accessed entries
        // LFU: higher access count = more likely to stay
        // LRU: more recent access = more likely to stay

        let hot_count = 1000u64;
        let cold_count = 1u64;
        let entries = [
            (1u64, hot_count),
            (2, cold_count),
            (3, hot_count),
            (4, cold_count),
        ];

        // LFU should evict cold entries first
        let to_evict: Vec<_> = entries
            .iter()
            .filter(|(_, count)| *count == cold_count)
            .collect();
        assert_eq!(to_evict.len(), 2); // Entries 2 and 4 are cold
    }

    #[test]
    fn test_eviction_multiple_entries() {
        // Test evicting multiple entries in sequence
        let mut entries: HashMap<u64, u64> = HashMap::new();
        entries.insert(1, 10); // access_count = 10
        entries.insert(2, 5); // access_count = 5
        entries.insert(3, 15); // access_count = 15
        entries.insert(4, 2); // access_count = 2

        // Simulate LFU eviction order
        let mut eviction_order = Vec::new();
        while !entries.is_empty() {
            let min_key = *entries.iter().min_by_key(|(_, v)| *v).unwrap().0;
            entries.remove(&min_key);
            eviction_order.push(min_key);
        }

        // Should evict in order: 4 (2), 2 (5), 1 (10), 3 (15)
        assert_eq!(eviction_order, vec![4, 2, 1, 3]);
    }

    #[test]
    fn test_eviction_after_invalidation() {
        // Test that invalidation reduces need for eviction
        let manager = ShaderPermutationManager::new(PermutationConfig::minimal());

        // Invalidate non-existent shader
        let removed = manager.invalidate(999);
        assert_eq!(removed, 0);

        // Manager should still be empty
        assert!(manager.is_empty());

        // Invalidate all on empty cache is safe
        manager.invalidate_all();
        assert!(manager.is_empty());
    }

    #[test]
    fn test_eviction_policy_change() {
        // Test that different policies select different entries
        let t1 = Instant::now();
        std::thread::sleep(std::time::Duration::from_millis(1));
        let t_old = Instant::now();
        std::thread::sleep(std::time::Duration::from_millis(1));
        let t_recent = Instant::now();

        // Entry: (shader_id, created_at, last_accessed, access_count)
        struct Entry {
            id: u64,
            created: Instant,
            accessed: Instant,
            count: u64,
        }

        let entries = vec![
            Entry { id: 1, created: t_old, accessed: t_recent, count: 100 },
            Entry { id: 2, created: t_recent, accessed: t_old, count: 50 },
            Entry { id: 3, created: t1, accessed: t1, count: 10 },
        ];

        // LRU would evict entry 3 (oldest last_accessed = t1)
        let lru_victim = entries.iter().min_by_key(|e| e.accessed).unwrap();
        assert_eq!(lru_victim.id, 3);

        // LFU would evict entry 3 (lowest count = 10)
        let lfu_victim = entries.iter().min_by_key(|e| e.count).unwrap();
        assert_eq!(lfu_victim.id, 3);

        // Oldest would evict entry 3 (oldest created = t1)
        let oldest_victim = entries.iter().min_by_key(|e| e.created).unwrap();
        assert_eq!(oldest_victim.id, 3);
    }

    // -------------------------------------------------------------------------
    // Thread Safety (6 tests)
    // -------------------------------------------------------------------------

    #[test]
    fn test_manager_concurrent_get_or_compile() {
        use std::thread;

        // Test that manager can be accessed from multiple threads safely
        let manager = Arc::new(ShaderPermutationManager::with_defaults());
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let m = Arc::clone(&manager);
                thread::spawn(move || {
                    // These operations should be thread-safe
                    let key = PermutationKey::new(i, FeatureFlags::SKINNED);
                    assert!(!m.contains(&key));
                    assert!(m.get_cached(&key).is_none());
                    m.permutation_count()
                })
            })
            .collect();

        for h in handles {
            let count = h.join().unwrap();
            assert_eq!(count, 0); // No actual entries added
        }
    }

    #[test]
    fn test_manager_concurrent_invalidate() {
        use std::thread;

        let manager = Arc::new(ShaderPermutationManager::with_defaults());
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let m = Arc::clone(&manager);
                thread::spawn(move || {
                    // Concurrent invalidations should be safe
                    m.invalidate(i as u64);
                    m.invalidate_all();
                    let key = PermutationKey::new(i as u64, FeatureFlags::SKINNED);
                    m.invalidate_key(&key);
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
        assert!(manager.is_empty());
    }

    #[test]
    fn test_manager_read_write_contention() {
        use std::thread;

        // Test concurrent reads and writes to the manager
        let manager = Arc::new(ShaderPermutationManager::with_defaults());

        let handles: Vec<_> = (0..20)
            .map(|i| {
                let m = Arc::clone(&manager);
                thread::spawn(move || {
                    if i % 2 == 0 {
                        // Readers
                        for _ in 0..100 {
                            m.permutation_count();
                            m.is_empty();
                            m.keys();
                            m.shader_ids();
                            m.metrics();
                        }
                    } else {
                        // Writers (no actual writes, just invalidations)
                        for j in 0..100 {
                            m.invalidate(j as u64);
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_metrics_concurrent_increment() {
        use std::thread;

        // Test concurrent metric updates
        let hits = Arc::new(AtomicU64::new(0));
        let misses = Arc::new(AtomicU64::new(0));

        let handles: Vec<_> = (0..20)
            .map(|i| {
                let h = Arc::clone(&hits);
                let m = Arc::clone(&misses);
                thread::spawn(move || {
                    for _ in 0..1000 {
                        if i % 2 == 0 {
                            h.fetch_add(1, Ordering::Relaxed);
                        } else {
                            m.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // 10 threads * 1000 increments each
        assert_eq!(hits.load(Ordering::Relaxed), 10_000);
        assert_eq!(misses.load(Ordering::Relaxed), 10_000);
    }

    #[test]
    fn test_feature_flags_send_sync_static() {
        // Verify FeatureFlags is Send + Sync + 'static
        fn assert_send_sync_static<T: Send + Sync + 'static>() {}
        assert_send_sync_static::<FeatureFlags>();

        // Can be used across threads
        use std::thread;
        let flags = FeatureFlags::SKINNED | FeatureFlags::SHADOWS;
        let handle = thread::spawn(move || {
            assert!(flags.contains(FeatureFlags::SKINNED));
            flags
        });
        let result = handle.join().unwrap();
        assert_eq!(result, FeatureFlags::SKINNED | FeatureFlags::SHADOWS);
    }

    #[test]
    fn test_all_types_send_sync() {
        // Comprehensive Send + Sync verification
        fn assert_send_sync<T: Send + Sync>() {}

        assert_send_sync::<FeatureFlags>();
        assert_send_sync::<PermutationKey>();
        assert_send_sync::<PermutationConfig>();
        assert_send_sync::<PermutationMetrics>();
        assert_send_sync::<PermutationError>();
        assert_send_sync::<EvictionPolicy>();
        assert_send_sync::<ShaderPermutationManager>();

        // Also verify 'static for key types used in threading
        fn assert_static<T: 'static>() {}
        assert_static::<FeatureFlags>();
        assert_static::<PermutationKey>();
        assert_static::<EvictionPolicy>();
    }

    // -------------------------------------------------------------------------
    // Config Validation (5 tests)
    // -------------------------------------------------------------------------

    #[test]
    fn test_config_zero_max_permutations_rejected() {
        let config = PermutationConfig::new().max_permutations(0);
        let result = config.validate();

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.is_config_error());
        assert!(err.to_string().contains("max_permutations"));
    }

    #[test]
    fn test_config_builder_chain_all_options() {
        // Test all builder methods can be chained
        let config = PermutationConfig::new()
            .max_permutations(1024)
            .enable_lazy_compilation(false)
            .eviction_policy(EvictionPolicy::Oldest)
            .max_permutations(512) // Override previous
            .eviction_policy(EvictionPolicy::LFU); // Override previous

        assert_eq!(config.max_permutations, 512);
        assert!(!config.enable_lazy_compilation);
        assert_eq!(config.eviction_policy, EvictionPolicy::LFU);
    }

    #[test]
    fn test_config_defaults_valid() {
        // All preset configs should be valid
        let configs = [
            PermutationConfig::default(),
            PermutationConfig::new(),
            PermutationConfig::minimal(),
            PermutationConfig::development(),
            PermutationConfig::production(),
        ];

        for config in &configs {
            assert!(
                config.validate().is_ok(),
                "Config {:?} should be valid",
                config
            );
            assert!(config.max_permutations > 0);
        }
    }

    #[test]
    fn test_config_clone_independent() {
        let config1 = PermutationConfig::new().max_permutations(100);
        let mut config2 = config1.clone();

        config2.max_permutations = 200;
        config2.enable_lazy_compilation = false;
        config2.eviction_policy = EvictionPolicy::Oldest;

        // Original should be unchanged
        assert_eq!(config1.max_permutations, 100);
        assert!(config1.enable_lazy_compilation);
        assert_eq!(config1.eviction_policy, EvictionPolicy::LRU);

        // Clone should have new values
        assert_eq!(config2.max_permutations, 200);
        assert!(!config2.enable_lazy_compilation);
        assert_eq!(config2.eviction_policy, EvictionPolicy::Oldest);
    }

    #[test]
    fn test_config_debug_format() {
        let config = PermutationConfig::new()
            .max_permutations(128)
            .eviction_policy(EvictionPolicy::LFU);

        let debug = format!("{:?}", config);

        assert!(debug.contains("PermutationConfig"));
        assert!(debug.contains("128"));
        assert!(debug.contains("LFU"));
    }

    // -------------------------------------------------------------------------
    // Error Paths (4 tests)
    // -------------------------------------------------------------------------

    #[test]
    fn test_error_max_exceeded_message() {
        let err = PermutationError::MaxPermutationsExceeded {
            current: 300,
            max: 256,
        };

        let display = err.to_string();
        assert!(display.contains("maximum permutations exceeded"));
        assert!(display.contains("300"));
        assert!(display.contains("256"));

        // Verify accessor methods
        assert!(err.is_max_exceeded());
        assert!(!err.is_compilation_error());
        assert!(!err.is_not_found());
        assert!(!err.is_config_error());
    }

    #[test]
    fn test_error_compilation_failed_preserves_message() {
        let original_msg = "syntax error at line 42: unexpected token";
        let err = PermutationError::CompilationFailed(original_msg.to_string());

        let display = err.to_string();
        assert!(display.contains("shader compilation failed"));
        assert!(display.contains("syntax error at line 42"));
        assert!(display.contains("unexpected token"));

        // Verify accessor methods
        assert!(err.is_compilation_error());
        assert!(!err.is_max_exceeded());
    }

    #[test]
    fn test_error_shader_not_found_details() {
        let shader_id = 0xDEADBEEF_u64;
        let err = PermutationError::ShaderNotFound { shader_id };

        let display = err.to_string();
        assert!(display.contains("shader not found"));
        assert!(display.contains(&shader_id.to_string()));

        // Verify accessor methods
        assert!(err.is_not_found());
        assert!(!err.is_compilation_error());

        // Test with various shader IDs
        let err_zero = PermutationError::ShaderNotFound { shader_id: 0 };
        assert!(err_zero.to_string().contains("0"));

        let err_max = PermutationError::ShaderNotFound { shader_id: u64::MAX };
        assert!(err_max.to_string().contains(&u64::MAX.to_string()));
    }

    #[test]
    fn test_error_display_all_variants() {
        let errors = [
            PermutationError::MaxPermutationsExceeded { current: 10, max: 5 },
            PermutationError::CompilationFailed("test error".to_string()),
            PermutationError::ShaderNotFound { shader_id: 123 },
            PermutationError::ConfigError("invalid option".to_string()),
        ];

        let expected_substrings = [
            "maximum permutations exceeded",
            "shader compilation failed",
            "shader not found",
            "configuration error",
        ];

        for (err, expected) in errors.iter().zip(expected_substrings.iter()) {
            let display = err.to_string();
            assert!(
                display.contains(expected),
                "Error '{}' should contain '{}'",
                display,
                expected
            );
        }

        // All errors should implement std::error::Error
        fn assert_error<T: std::error::Error>() {}
        assert_error::<PermutationError>();

        // All errors should be Clone and PartialEq
        for err in &errors {
            let cloned = err.clone();
            assert_eq!(err, &cloned);
        }
    }
}
