//! Pipeline cache for render pipelines.
//!
//! Provides a thread-safe cache for compiled render pipelines, enabling
//! efficient reuse and hot-reload support via shader invalidation.
//!
//! # Architecture
//!
//! The cache uses a two-level approach:
//! 1. `PipelineKey` identifies unique pipeline configurations
//! 2. `RenderPipelineCache` stores compiled pipelines keyed by configuration
//!
//! # Thread Safety
//!
//! The cache uses `RwLock` for concurrent read access and atomic operations
//! for metrics tracking, minimizing contention in typical rendering scenarios.
//!
//! # Hot Reload
//!
//! When shaders are recompiled, call `invalidate(shader_id)` to remove all
//! pipelines using that shader. The next `get_or_create()` call will rebuild
//! them with the new shader.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use super::vertex_state::VertexBufferLayoutDescriptor;
use super::fragment_state::ColorTargetStateDescriptor;

// ---------------------------------------------------------------------------
// PipelineKey
// ---------------------------------------------------------------------------

/// Unique identifier for a render pipeline configuration.
///
/// All state that affects pipeline compilation is captured in this key.
/// Two pipelines with equal keys are guaranteed to be identical.
///
/// # Shader IDs
///
/// The `vertex_shader_id` and `fragment_shader_id` fields are used for
/// hot-reload invalidation. When a shader is recompiled, all pipelines
/// using that shader can be invalidated efficiently.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PipelineKey {
    /// Vertex shader ID (for hot-reload invalidation).
    pub vertex_shader_id: u64,
    /// Fragment shader ID (for hot-reload invalidation).
    pub fragment_shader_id: Option<u64>,

    /// Hash of vertex buffer layouts.
    pub vertex_layout_hash: u64,

    /// Primitive topology.
    pub topology: wgpu::PrimitiveTopology,
    /// Front face winding order.
    pub front_face: wgpu::FrontFace,
    /// Face culling mode.
    pub cull_mode: Option<wgpu::Face>,
    /// Polygon fill mode.
    pub polygon_mode: wgpu::PolygonMode,

    /// Depth texture format (None = no depth testing).
    pub depth_format: Option<wgpu::TextureFormat>,
    /// Whether depth writes are enabled.
    pub depth_write: bool,
    /// Depth comparison function.
    pub depth_compare: wgpu::CompareFunction,

    /// MSAA sample count.
    pub sample_count: u32,

    /// Hash of color target configurations.
    pub color_targets_hash: u64,
}

impl Hash for PipelineKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.vertex_shader_id.hash(state);
        self.fragment_shader_id.hash(state);
        self.vertex_layout_hash.hash(state);

        // Hash enum discriminants directly
        std::mem::discriminant(&self.topology).hash(state);
        std::mem::discriminant(&self.front_face).hash(state);
        self.cull_mode.map(|f| std::mem::discriminant(&f)).hash(state);
        std::mem::discriminant(&self.polygon_mode).hash(state);

        self.depth_format.map(|f| std::mem::discriminant(&f)).hash(state);
        self.depth_write.hash(state);
        std::mem::discriminant(&self.depth_compare).hash(state);

        self.sample_count.hash(state);
        self.color_targets_hash.hash(state);
    }
}

impl PipelineKey {
    /// Create a minimal key (for testing or simple pipelines).
    pub fn new(vertex_shader_id: u64) -> Self {
        Self {
            vertex_shader_id,
            fragment_shader_id: None,
            vertex_layout_hash: 0,
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            polygon_mode: wgpu::PolygonMode::Fill,
            depth_format: None,
            depth_write: true,
            depth_compare: wgpu::CompareFunction::Less,
            sample_count: 1,
            color_targets_hash: 0,
        }
    }

    /// Set the fragment shader ID.
    pub fn with_fragment_shader(mut self, shader_id: u64) -> Self {
        self.fragment_shader_id = Some(shader_id);
        self
    }

    /// Set the vertex layout hash.
    pub fn with_vertex_layout_hash(mut self, hash: u64) -> Self {
        self.vertex_layout_hash = hash;
        self
    }

    /// Set the primitive topology.
    pub fn with_topology(mut self, topology: wgpu::PrimitiveTopology) -> Self {
        self.topology = topology;
        self
    }

    /// Set the front face winding.
    pub fn with_front_face(mut self, front_face: wgpu::FrontFace) -> Self {
        self.front_face = front_face;
        self
    }

    /// Set the cull mode.
    pub fn with_cull_mode(mut self, cull_mode: Option<wgpu::Face>) -> Self {
        self.cull_mode = cull_mode;
        self
    }

    /// Set the polygon mode.
    pub fn with_polygon_mode(mut self, polygon_mode: wgpu::PolygonMode) -> Self {
        self.polygon_mode = polygon_mode;
        self
    }

    /// Set the depth format.
    pub fn with_depth_format(mut self, format: wgpu::TextureFormat) -> Self {
        self.depth_format = Some(format);
        self
    }

    /// Set depth write enabled.
    pub fn with_depth_write(mut self, enabled: bool) -> Self {
        self.depth_write = enabled;
        self
    }

    /// Set the depth compare function.
    pub fn with_depth_compare(mut self, compare: wgpu::CompareFunction) -> Self {
        self.depth_compare = compare;
        self
    }

    /// Set the sample count.
    pub fn with_sample_count(mut self, count: u32) -> Self {
        self.sample_count = count;
        self
    }

    /// Set the color targets hash.
    pub fn with_color_targets_hash(mut self, hash: u64) -> Self {
        self.color_targets_hash = hash;
        self
    }

    /// Check if this key uses the given shader ID.
    pub fn uses_shader(&self, shader_id: u64) -> bool {
        self.vertex_shader_id == shader_id
            || self.fragment_shader_id == Some(shader_id)
    }
}

// ---------------------------------------------------------------------------
// Hash Helper Functions
// ---------------------------------------------------------------------------

/// Hash a vertex layout for use in `PipelineKey`.
///
/// Produces a consistent 64-bit hash of the vertex buffer configuration,
/// including stride, step mode, and all attribute definitions.
pub fn hash_vertex_layout(buffers: &[VertexBufferLayoutDescriptor]) -> u64 {
    use std::hash::DefaultHasher;
    let mut hasher = DefaultHasher::new();

    buffers.len().hash(&mut hasher);
    for buffer in buffers {
        buffer.array_stride.hash(&mut hasher);
        std::mem::discriminant(&buffer.step_mode).hash(&mut hasher);

        buffer.attributes.len().hash(&mut hasher);
        for attr in &buffer.attributes {
            std::mem::discriminant(&attr.format).hash(&mut hasher);
            attr.offset.hash(&mut hasher);
            attr.shader_location.hash(&mut hasher);
        }
    }

    hasher.finish()
}

/// Hash color targets for use in `PipelineKey`.
///
/// Produces a consistent 64-bit hash of the color target configuration,
/// including format, blend state, and write mask for each target.
pub fn hash_color_targets(targets: &[Option<ColorTargetStateDescriptor>]) -> u64 {
    use std::hash::DefaultHasher;
    let mut hasher = DefaultHasher::new();

    targets.len().hash(&mut hasher);
    for target in targets {
        target.is_some().hash(&mut hasher);
        if let Some(t) = target {
            std::mem::discriminant(&t.format).hash(&mut hasher);
            t.blend.is_some().hash(&mut hasher);
            if let Some(blend) = &t.blend {
                // Hash color blend
                std::mem::discriminant(&blend.color.src_factor).hash(&mut hasher);
                std::mem::discriminant(&blend.color.dst_factor).hash(&mut hasher);
                std::mem::discriminant(&blend.color.operation).hash(&mut hasher);
                // Hash alpha blend
                std::mem::discriminant(&blend.alpha.src_factor).hash(&mut hasher);
                std::mem::discriminant(&blend.alpha.dst_factor).hash(&mut hasher);
                std::mem::discriminant(&blend.alpha.operation).hash(&mut hasher);
            }
            t.write_mask.bits().hash(&mut hasher);
        }
    }

    hasher.finish()
}

// ---------------------------------------------------------------------------
// CacheMetrics
// ---------------------------------------------------------------------------

/// Statistics about cache performance.
#[derive(Debug, Clone, Copy, Default)]
pub struct CacheMetrics {
    /// Number of cache hits.
    pub hits: u64,
    /// Number of cache misses (new pipeline created).
    pub misses: u64,
    /// Number of pipelines evicted.
    pub evictions: u64,
    /// Number of pipelines invalidated (e.g., shader hot-reload).
    pub invalidations: u64,
}

impl CacheMetrics {
    /// Calculate the cache hit rate (0.0 - 1.0).
    ///
    /// Returns 0.0 if no accesses have occurred.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Total number of cache accesses.
    pub fn total_accesses(&self) -> u64 {
        self.hits + self.misses
    }
}

// ---------------------------------------------------------------------------
// AtomicMetrics (internal)
// ---------------------------------------------------------------------------

/// Atomic metrics for lock-free updates.
struct AtomicMetrics {
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    invalidations: AtomicU64,
}

impl AtomicMetrics {
    fn new() -> Self {
        Self {
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            invalidations: AtomicU64::new(0),
        }
    }

    fn record_hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }

    fn record_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }

    fn record_evictions(&self, count: u64) {
        self.evictions.fetch_add(count, Ordering::Relaxed);
    }

    fn record_invalidations(&self, count: u64) {
        self.invalidations.fetch_add(count, Ordering::Relaxed);
    }

    fn snapshot(&self) -> CacheMetrics {
        CacheMetrics {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            invalidations: self.invalidations.load(Ordering::Relaxed),
        }
    }

    fn reset(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.evictions.store(0, Ordering::Relaxed);
        self.invalidations.store(0, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// RenderPipelineCache
// ---------------------------------------------------------------------------

/// Thread-safe cache for compiled render pipelines.
///
/// # Usage
///
/// ```no_run
/// # fn example(device: std::sync::Arc<wgpu::Device>) {
/// use renderer_backend::render_pipeline::pipeline_cache::{
///     RenderPipelineCache, PipelineKey,
/// };
///
/// let cache = RenderPipelineCache::new(device.clone());
///
/// let key = PipelineKey::new(1)
///     .with_fragment_shader(2)
///     .with_sample_count(4);
///
/// let pipeline = cache.get_or_create(&key, || {
///     // Create the actual wgpu pipeline here
///     # panic!("example")
/// });
///
/// // Check cache performance
/// let metrics = cache.metrics();
/// println!("Hit rate: {:.1}%", metrics.hit_rate() * 100.0);
/// # }
/// ```
///
/// # Hot Reload
///
/// When a shader is recompiled:
///
/// ```no_run
/// # fn example(cache: &renderer_backend::render_pipeline::pipeline_cache::RenderPipelineCache, shader_id: u64) {
/// let invalidated = cache.invalidate(shader_id);
/// println!("Invalidated {} pipelines", invalidated);
/// # }
/// ```
pub struct RenderPipelineCache {
    #[allow(dead_code)]
    device: Arc<wgpu::Device>,
    pipelines: RwLock<HashMap<PipelineKey, Arc<wgpu::RenderPipeline>>>,
    metrics: AtomicMetrics,
}

impl RenderPipelineCache {
    /// Create a new pipeline cache.
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        Self {
            device,
            pipelines: RwLock::new(HashMap::new()),
            metrics: AtomicMetrics::new(),
        }
    }

    /// Get an existing pipeline or create a new one.
    ///
    /// If a pipeline with the given key exists, it is returned immediately.
    /// Otherwise, `create_fn` is called to build a new pipeline, which is
    /// then cached and returned.
    ///
    /// # Thread Safety
    ///
    /// This method is safe to call from multiple threads. The cache uses
    /// a read-write lock, so multiple threads can retrieve cached pipelines
    /// concurrently. Pipeline creation is serialized to avoid duplicate work.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn example(
    /// #     cache: &renderer_backend::render_pipeline::pipeline_cache::RenderPipelineCache,
    /// #     key: &renderer_backend::render_pipeline::pipeline_cache::PipelineKey,
    /// #     device: &wgpu::Device,
    /// # ) {
    /// let pipeline = cache.get_or_create(&key, || {
    ///     device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
    ///         label: Some("my_pipeline"),
    ///         // ... pipeline configuration
    ///         # layout: None,
    ///         # vertex: wgpu::VertexState { module: todo!(), entry_point: todo!(), compilation_options: todo!(), buffers: &[] },
    ///         # primitive: wgpu::PrimitiveState::default(),
    ///         # depth_stencil: None,
    ///         # multisample: wgpu::MultisampleState::default(),
    ///         # fragment: None,
    ///         # multiview: None,
    ///         # cache: None,
    ///     })
    /// });
    /// # }
    /// ```
    pub fn get_or_create<F>(&self, key: &PipelineKey, create_fn: F) -> Arc<wgpu::RenderPipeline>
    where
        F: FnOnce() -> wgpu::RenderPipeline,
    {
        // Fast path: try to get from cache with read lock
        {
            let pipelines = self.pipelines.read().unwrap();
            if let Some(pipeline) = pipelines.get(key) {
                self.metrics.record_hit();
                return Arc::clone(pipeline);
            }
        }

        // Slow path: need to create pipeline
        let mut pipelines = self.pipelines.write().unwrap();

        // Double-check after acquiring write lock (another thread may have created it)
        if let Some(pipeline) = pipelines.get(key) {
            self.metrics.record_hit();
            return Arc::clone(pipeline);
        }

        // Create the pipeline
        let pipeline = Arc::new(create_fn());
        pipelines.insert(key.clone(), Arc::clone(&pipeline));
        self.metrics.record_miss();

        pipeline
    }

    /// Invalidate all pipelines using the given shader ID.
    ///
    /// Returns the number of pipelines invalidated.
    ///
    /// This should be called when a shader is recompiled for hot-reload.
    /// The next `get_or_create()` call with an affected key will create
    /// a new pipeline with the updated shader.
    pub fn invalidate(&self, shader_id: u64) -> usize {
        let mut pipelines = self.pipelines.write().unwrap();
        let before = pipelines.len();

        pipelines.retain(|key, _| !key.uses_shader(shader_id));

        let removed = before - pipelines.len();
        self.metrics.record_invalidations(removed as u64);
        removed
    }

    /// Clear the entire cache.
    ///
    /// All cached pipelines are dropped. Returns the number of pipelines removed.
    pub fn clear(&self) -> usize {
        let mut pipelines = self.pipelines.write().unwrap();
        let count = pipelines.len();
        pipelines.clear();
        self.metrics.record_evictions(count as u64);
        count
    }

    /// Get current cache metrics.
    pub fn metrics(&self) -> CacheMetrics {
        self.metrics.snapshot()
    }

    /// Reset cache metrics to zero.
    pub fn reset_metrics(&self) {
        self.metrics.reset();
    }

    /// Get the number of cached pipelines.
    pub fn len(&self) -> usize {
        self.pipelines.read().unwrap().len()
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.pipelines.read().unwrap().is_empty()
    }

    /// Check if a pipeline exists in the cache.
    pub fn contains(&self, key: &PipelineKey) -> bool {
        self.pipelines.read().unwrap().contains_key(key)
    }

    /// Remove a specific pipeline from the cache.
    ///
    /// Returns `true` if the pipeline was present and removed.
    pub fn remove(&self, key: &PipelineKey) -> bool {
        let mut pipelines = self.pipelines.write().unwrap();
        let removed = pipelines.remove(key).is_some();
        if removed {
            self.metrics.record_evictions(1);
        }
        removed
    }
}

// Thread safety markers
unsafe impl Send for RenderPipelineCache {}
unsafe impl Sync for RenderPipelineCache {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;

    // Helper to compute hash of a PipelineKey
    fn hash_key(key: &PipelineKey) -> u64 {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }

    // -------------------------------------------------------------------------
    // PipelineKey Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_pipeline_key_equality() {
        let key1 = PipelineKey::new(1).with_fragment_shader(2);
        let key2 = PipelineKey::new(1).with_fragment_shader(2);
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_pipeline_key_inequality_vertex_shader() {
        let key1 = PipelineKey::new(1);
        let key2 = PipelineKey::new(2);
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_pipeline_key_inequality_fragment_shader() {
        let key1 = PipelineKey::new(1).with_fragment_shader(2);
        let key2 = PipelineKey::new(1).with_fragment_shader(3);
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_pipeline_key_inequality_topology() {
        let key1 = PipelineKey::new(1).with_topology(wgpu::PrimitiveTopology::TriangleList);
        let key2 = PipelineKey::new(1).with_topology(wgpu::PrimitiveTopology::LineList);
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_pipeline_key_inequality_cull_mode() {
        let key1 = PipelineKey::new(1).with_cull_mode(Some(wgpu::Face::Back));
        let key2 = PipelineKey::new(1).with_cull_mode(None);
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_pipeline_key_inequality_sample_count() {
        let key1 = PipelineKey::new(1).with_sample_count(1);
        let key2 = PipelineKey::new(1).with_sample_count(4);
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_pipeline_key_hash_equality() {
        let key1 = PipelineKey::new(1).with_fragment_shader(2).with_sample_count(4);
        let key2 = PipelineKey::new(1).with_fragment_shader(2).with_sample_count(4);
        assert_eq!(hash_key(&key1), hash_key(&key2));
    }

    #[test]
    fn test_pipeline_key_hash_inequality() {
        let key1 = PipelineKey::new(1);
        let key2 = PipelineKey::new(2);
        // Different keys should (very likely) have different hashes
        assert_ne!(hash_key(&key1), hash_key(&key2));
    }

    #[test]
    fn test_pipeline_key_uses_shader() {
        let key = PipelineKey::new(1).with_fragment_shader(2);

        assert!(key.uses_shader(1)); // vertex shader
        assert!(key.uses_shader(2)); // fragment shader
        assert!(!key.uses_shader(3)); // not used
    }

    #[test]
    fn test_pipeline_key_uses_shader_no_fragment() {
        let key = PipelineKey::new(1);

        assert!(key.uses_shader(1));
        assert!(!key.uses_shader(2));
    }

    // -------------------------------------------------------------------------
    // Hash Function Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_hash_vertex_layout_empty() {
        let hash = hash_vertex_layout(&[]);
        // Should produce a consistent hash
        assert_eq!(hash, hash_vertex_layout(&[]));
    }

    #[test]
    fn test_hash_vertex_layout_single_buffer() {
        let buffer = VertexBufferLayoutDescriptor::per_vertex(32)
            .with_attribute(wgpu::VertexFormat::Float32x3, 0, 0)
            .with_attribute(wgpu::VertexFormat::Float32x2, 12, 1);

        let hash1 = hash_vertex_layout(&[buffer.clone()]);
        let hash2 = hash_vertex_layout(&[buffer]);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hash_vertex_layout_different_stride() {
        let buffer1 = VertexBufferLayoutDescriptor::per_vertex(32);
        let buffer2 = VertexBufferLayoutDescriptor::per_vertex(64);

        assert_ne!(hash_vertex_layout(&[buffer1]), hash_vertex_layout(&[buffer2]));
    }

    #[test]
    fn test_hash_vertex_layout_different_step_mode() {
        let buffer1 = VertexBufferLayoutDescriptor::per_vertex(32);
        let buffer2 = VertexBufferLayoutDescriptor::per_instance(32);

        assert_ne!(hash_vertex_layout(&[buffer1]), hash_vertex_layout(&[buffer2]));
    }

    #[test]
    fn test_hash_color_targets_empty() {
        let hash = hash_color_targets(&[]);
        assert_eq!(hash, hash_color_targets(&[]));
    }

    #[test]
    fn test_hash_color_targets_single() {
        let target = ColorTargetStateDescriptor::srgb();

        let hash1 = hash_color_targets(&[Some(target.clone())]);
        let hash2 = hash_color_targets(&[Some(target)]);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hash_color_targets_different_format() {
        let target1 = ColorTargetStateDescriptor::srgb();
        let target2 = ColorTargetStateDescriptor::hdr();

        assert_ne!(
            hash_color_targets(&[Some(target1)]),
            hash_color_targets(&[Some(target2)])
        );
    }

    #[test]
    fn test_hash_color_targets_with_blend() {
        let target1 = ColorTargetStateDescriptor::srgb();
        let target2 = ColorTargetStateDescriptor::srgb().alpha_blend();

        assert_ne!(
            hash_color_targets(&[Some(target1)]),
            hash_color_targets(&[Some(target2)])
        );
    }

    #[test]
    fn test_hash_color_targets_null() {
        let hash1 = hash_color_targets(&[None]);
        let hash2 = hash_color_targets(&[Some(ColorTargetStateDescriptor::srgb())]);

        assert_ne!(hash1, hash2);
    }

    // -------------------------------------------------------------------------
    // CacheMetrics Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cache_metrics_default() {
        let metrics = CacheMetrics::default();
        assert_eq!(metrics.hits, 0);
        assert_eq!(metrics.misses, 0);
        assert_eq!(metrics.evictions, 0);
        assert_eq!(metrics.invalidations, 0);
    }

    #[test]
    fn test_cache_metrics_hit_rate_zero_accesses() {
        let metrics = CacheMetrics::default();
        assert_eq!(metrics.hit_rate(), 0.0);
    }

    #[test]
    fn test_cache_metrics_hit_rate_all_hits() {
        let metrics = CacheMetrics {
            hits: 100,
            misses: 0,
            evictions: 0,
            invalidations: 0,
        };
        assert_eq!(metrics.hit_rate(), 1.0);
    }

    #[test]
    fn test_cache_metrics_hit_rate_all_misses() {
        let metrics = CacheMetrics {
            hits: 0,
            misses: 100,
            evictions: 0,
            invalidations: 0,
        };
        assert_eq!(metrics.hit_rate(), 0.0);
    }

    #[test]
    fn test_cache_metrics_hit_rate_fifty_percent() {
        let metrics = CacheMetrics {
            hits: 50,
            misses: 50,
            evictions: 0,
            invalidations: 0,
        };
        assert!((metrics.hit_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cache_metrics_total_accesses() {
        let metrics = CacheMetrics {
            hits: 30,
            misses: 70,
            evictions: 10,
            invalidations: 5,
        };
        assert_eq!(metrics.total_accesses(), 100);
    }

    // -------------------------------------------------------------------------
    // AtomicMetrics Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_atomic_metrics_new() {
        let metrics = AtomicMetrics::new();
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.hits, 0);
        assert_eq!(snapshot.misses, 0);
    }

    #[test]
    fn test_atomic_metrics_record_hit() {
        let metrics = AtomicMetrics::new();
        metrics.record_hit();
        metrics.record_hit();
        assert_eq!(metrics.snapshot().hits, 2);
    }

    #[test]
    fn test_atomic_metrics_record_miss() {
        let metrics = AtomicMetrics::new();
        metrics.record_miss();
        assert_eq!(metrics.snapshot().misses, 1);
    }

    #[test]
    fn test_atomic_metrics_record_evictions() {
        let metrics = AtomicMetrics::new();
        metrics.record_evictions(5);
        assert_eq!(metrics.snapshot().evictions, 5);
    }

    #[test]
    fn test_atomic_metrics_record_invalidations() {
        let metrics = AtomicMetrics::new();
        metrics.record_invalidations(3);
        assert_eq!(metrics.snapshot().invalidations, 3);
    }

    #[test]
    fn test_atomic_metrics_reset() {
        let metrics = AtomicMetrics::new();
        metrics.record_hit();
        metrics.record_miss();
        metrics.record_evictions(5);
        metrics.reset();

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.hits, 0);
        assert_eq!(snapshot.misses, 0);
        assert_eq!(snapshot.evictions, 0);
        assert_eq!(snapshot.invalidations, 0);
    }

    // -------------------------------------------------------------------------
    // Pipeline Key Builder Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_pipeline_key_builder_chain() {
        let key = PipelineKey::new(1)
            .with_fragment_shader(2)
            .with_vertex_layout_hash(12345)
            .with_topology(wgpu::PrimitiveTopology::TriangleStrip)
            .with_front_face(wgpu::FrontFace::Cw)
            .with_cull_mode(Some(wgpu::Face::Front))
            .with_polygon_mode(wgpu::PolygonMode::Line)
            .with_depth_format(wgpu::TextureFormat::Depth32Float)
            .with_depth_write(false)
            .with_depth_compare(wgpu::CompareFunction::Greater)
            .with_sample_count(4)
            .with_color_targets_hash(67890);

        assert_eq!(key.vertex_shader_id, 1);
        assert_eq!(key.fragment_shader_id, Some(2));
        assert_eq!(key.vertex_layout_hash, 12345);
        assert_eq!(key.topology, wgpu::PrimitiveTopology::TriangleStrip);
        assert_eq!(key.front_face, wgpu::FrontFace::Cw);
        assert_eq!(key.cull_mode, Some(wgpu::Face::Front));
        assert_eq!(key.polygon_mode, wgpu::PolygonMode::Line);
        assert_eq!(key.depth_format, Some(wgpu::TextureFormat::Depth32Float));
        assert!(!key.depth_write);
        assert_eq!(key.depth_compare, wgpu::CompareFunction::Greater);
        assert_eq!(key.sample_count, 4);
        assert_eq!(key.color_targets_hash, 67890);
    }

    #[test]
    fn test_pipeline_key_default_values() {
        let key = PipelineKey::new(1);

        assert_eq!(key.vertex_shader_id, 1);
        assert_eq!(key.fragment_shader_id, None);
        assert_eq!(key.vertex_layout_hash, 0);
        assert_eq!(key.topology, wgpu::PrimitiveTopology::TriangleList);
        assert_eq!(key.front_face, wgpu::FrontFace::Ccw);
        assert_eq!(key.cull_mode, Some(wgpu::Face::Back));
        assert_eq!(key.polygon_mode, wgpu::PolygonMode::Fill);
        assert_eq!(key.depth_format, None);
        assert!(key.depth_write);
        assert_eq!(key.depth_compare, wgpu::CompareFunction::Less);
        assert_eq!(key.sample_count, 1);
        assert_eq!(key.color_targets_hash, 0);
    }

    // -------------------------------------------------------------------------
    // Thread Safety Tests (compile-time checks)
    // -------------------------------------------------------------------------

    #[test]
    fn test_render_pipeline_cache_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<RenderPipelineCache>();
        assert_sync::<RenderPipelineCache>();
    }

    #[test]
    fn test_pipeline_key_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<PipelineKey>();
        assert_sync::<PipelineKey>();
    }

    #[test]
    fn test_cache_metrics_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<CacheMetrics>();
        assert_sync::<CacheMetrics>();
    }

    // -------------------------------------------------------------------------
    // Hash Stability Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_hash_consistency_across_calls() {
        let key = PipelineKey::new(42).with_fragment_shader(99);
        let hash1 = hash_key(&key);
        let hash2 = hash_key(&key);
        let hash3 = hash_key(&key);

        assert_eq!(hash1, hash2);
        assert_eq!(hash2, hash3);
    }

    #[test]
    fn test_hash_vertex_layout_order_matters() {
        let attr1 = VertexBufferLayoutDescriptor::per_vertex(32)
            .with_attribute(wgpu::VertexFormat::Float32x3, 0, 0)
            .with_attribute(wgpu::VertexFormat::Float32x2, 12, 1);

        let attr2 = VertexBufferLayoutDescriptor::per_vertex(32)
            .with_attribute(wgpu::VertexFormat::Float32x2, 12, 1)
            .with_attribute(wgpu::VertexFormat::Float32x3, 0, 0);

        // Different attribute order should produce different hash
        assert_ne!(hash_vertex_layout(&[attr1]), hash_vertex_layout(&[attr2]));
    }

    #[test]
    fn test_hash_color_targets_multiple() {
        let targets1 = vec![
            Some(ColorTargetStateDescriptor::srgb()),
            Some(ColorTargetStateDescriptor::hdr()),
        ];

        let targets2 = vec![
            Some(ColorTargetStateDescriptor::hdr()),
            Some(ColorTargetStateDescriptor::srgb()),
        ];

        // Different order should produce different hash
        assert_ne!(hash_color_targets(&targets1), hash_color_targets(&targets2));
    }

    // -------------------------------------------------------------------------
    // Edge Case Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_pipeline_key_clone() {
        let key = PipelineKey::new(1).with_fragment_shader(2);
        let cloned = key.clone();
        assert_eq!(key, cloned);
        assert_eq!(hash_key(&key), hash_key(&cloned));
    }

    #[test]
    fn test_pipeline_key_all_topologies() {
        let topologies = [
            wgpu::PrimitiveTopology::PointList,
            wgpu::PrimitiveTopology::LineList,
            wgpu::PrimitiveTopology::LineStrip,
            wgpu::PrimitiveTopology::TriangleList,
            wgpu::PrimitiveTopology::TriangleStrip,
        ];

        let hashes: Vec<u64> = topologies
            .iter()
            .map(|t| hash_key(&PipelineKey::new(1).with_topology(*t)))
            .collect();

        // All topologies should produce different hashes
        for i in 0..hashes.len() {
            for j in (i + 1)..hashes.len() {
                assert_ne!(hashes[i], hashes[j], "Topologies {} and {} have same hash", i, j);
            }
        }
    }

    #[test]
    fn test_pipeline_key_all_polygon_modes() {
        let modes = [
            wgpu::PolygonMode::Fill,
            wgpu::PolygonMode::Line,
            wgpu::PolygonMode::Point,
        ];

        let hashes: Vec<u64> = modes
            .iter()
            .map(|m| hash_key(&PipelineKey::new(1).with_polygon_mode(*m)))
            .collect();

        // All modes should produce different hashes
        for i in 0..hashes.len() {
            for j in (i + 1)..hashes.len() {
                assert_ne!(hashes[i], hashes[j]);
            }
        }
    }

    #[test]
    fn test_pipeline_key_depth_format_variations() {
        let key_no_depth = PipelineKey::new(1);
        let key_depth32 = PipelineKey::new(1)
            .with_depth_format(wgpu::TextureFormat::Depth32Float);
        let key_depth24_stencil8 = PipelineKey::new(1)
            .with_depth_format(wgpu::TextureFormat::Depth24PlusStencil8);

        assert_ne!(key_no_depth, key_depth32);
        assert_ne!(key_no_depth, key_depth24_stencil8);
        assert_ne!(key_depth32, key_depth24_stencil8);
    }

    // -------------------------------------------------------------------------
    // Additional Whitebox Tests - PipelineKey Enum Variants
    // -------------------------------------------------------------------------

    #[test]
    fn test_pipeline_key_all_cull_modes() {
        let cull_none = PipelineKey::new(1).with_cull_mode(None);
        let cull_front = PipelineKey::new(1).with_cull_mode(Some(wgpu::Face::Front));
        let cull_back = PipelineKey::new(1).with_cull_mode(Some(wgpu::Face::Back));

        // All should produce different hashes
        let hashes = [hash_key(&cull_none), hash_key(&cull_front), hash_key(&cull_back)];
        assert_ne!(hashes[0], hashes[1]);
        assert_ne!(hashes[0], hashes[2]);
        assert_ne!(hashes[1], hashes[2]);
    }

    #[test]
    fn test_pipeline_key_all_front_faces() {
        let ccw = PipelineKey::new(1).with_front_face(wgpu::FrontFace::Ccw);
        let cw = PipelineKey::new(1).with_front_face(wgpu::FrontFace::Cw);

        assert_ne!(ccw, cw);
        assert_ne!(hash_key(&ccw), hash_key(&cw));
    }

    #[test]
    fn test_pipeline_key_all_compare_functions() {
        let compare_fns = [
            wgpu::CompareFunction::Never,
            wgpu::CompareFunction::Less,
            wgpu::CompareFunction::Equal,
            wgpu::CompareFunction::LessEqual,
            wgpu::CompareFunction::Greater,
            wgpu::CompareFunction::NotEqual,
            wgpu::CompareFunction::GreaterEqual,
            wgpu::CompareFunction::Always,
        ];

        let hashes: Vec<u64> = compare_fns
            .iter()
            .map(|c| hash_key(&PipelineKey::new(1).with_depth_compare(*c)))
            .collect();

        // All compare functions should produce different hashes
        for i in 0..hashes.len() {
            for j in (i + 1)..hashes.len() {
                assert_ne!(
                    hashes[i], hashes[j],
                    "CompareFunction {} and {} have same hash",
                    i, j
                );
            }
        }
    }

    #[test]
    fn test_pipeline_key_all_depth_formats() {
        let depth_formats = [
            None,
            Some(wgpu::TextureFormat::Depth16Unorm),
            Some(wgpu::TextureFormat::Depth24Plus),
            Some(wgpu::TextureFormat::Depth24PlusStencil8),
            Some(wgpu::TextureFormat::Depth32Float),
            Some(wgpu::TextureFormat::Depth32FloatStencil8),
        ];

        let mut keys: Vec<PipelineKey> = Vec::new();
        for df in depth_formats {
            let mut key = PipelineKey::new(1);
            if let Some(format) = df {
                key = key.with_depth_format(format);
            }
            keys.push(key);
        }

        // All depth formats should produce different keys
        for i in 0..keys.len() {
            for j in (i + 1)..keys.len() {
                assert_ne!(keys[i], keys[j], "Depth format {} and {} are equal", i, j);
            }
        }
    }

    #[test]
    fn test_pipeline_key_depth_write_variations() {
        let write_enabled = PipelineKey::new(1).with_depth_write(true);
        let write_disabled = PipelineKey::new(1).with_depth_write(false);

        assert_ne!(write_enabled, write_disabled);
        assert_ne!(hash_key(&write_enabled), hash_key(&write_disabled));
    }

    #[test]
    fn test_pipeline_key_sample_count_variations() {
        let sample_counts = [1, 2, 4, 8, 16];

        let hashes: Vec<u64> = sample_counts
            .iter()
            .map(|s| hash_key(&PipelineKey::new(1).with_sample_count(*s)))
            .collect();

        for i in 0..hashes.len() {
            for j in (i + 1)..hashes.len() {
                assert_ne!(hashes[i], hashes[j]);
            }
        }
    }

    #[test]
    fn test_pipeline_key_uses_shader_with_both_shaders() {
        let key = PipelineKey::new(100).with_fragment_shader(200);

        assert!(key.uses_shader(100));
        assert!(key.uses_shader(200));
        assert!(!key.uses_shader(300));
        assert!(!key.uses_shader(0));
        assert!(!key.uses_shader(99));
        assert!(!key.uses_shader(101));
    }

    #[test]
    fn test_pipeline_key_uses_shader_vertex_only() {
        let key = PipelineKey::new(42);

        assert!(key.uses_shader(42));
        assert!(!key.uses_shader(0));
        assert!(!key.uses_shader(41));
        assert!(!key.uses_shader(43));
    }

    // -------------------------------------------------------------------------
    // Additional Whitebox Tests - Hash Vertex Layout
    // -------------------------------------------------------------------------

    #[test]
    fn test_hash_vertex_layout_two_buffers() {
        let buffer1 = VertexBufferLayoutDescriptor::per_vertex(32)
            .with_attribute(wgpu::VertexFormat::Float32x3, 0, 0);
        let buffer2 = VertexBufferLayoutDescriptor::per_instance(64)
            .with_attribute(wgpu::VertexFormat::Float32x4, 0, 1);

        let hash_two = hash_vertex_layout(&[buffer1.clone(), buffer2.clone()]);
        let hash_two_again = hash_vertex_layout(&[buffer1.clone(), buffer2.clone()]);

        assert_eq!(hash_two, hash_two_again);

        // Different order should produce different hash
        let hash_reversed = hash_vertex_layout(&[buffer2, buffer1]);
        assert_ne!(hash_two, hash_reversed);
    }

    #[test]
    fn test_hash_vertex_layout_three_buffers() {
        let buf1 = VertexBufferLayoutDescriptor::per_vertex(12)
            .with_attribute(wgpu::VertexFormat::Float32x3, 0, 0);
        let buf2 = VertexBufferLayoutDescriptor::per_vertex(20)
            .with_attribute(wgpu::VertexFormat::Float32x3, 0, 1)
            .with_attribute(wgpu::VertexFormat::Float32x2, 12, 2);
        let buf3 = VertexBufferLayoutDescriptor::per_instance(64)
            .with_attribute(wgpu::VertexFormat::Float32x4, 0, 3)
            .with_attribute(wgpu::VertexFormat::Float32x4, 16, 4)
            .with_attribute(wgpu::VertexFormat::Float32x4, 32, 5)
            .with_attribute(wgpu::VertexFormat::Float32x4, 48, 6);

        let hash = hash_vertex_layout(&[buf1.clone(), buf2.clone(), buf3.clone()]);
        let hash_again = hash_vertex_layout(&[buf1, buf2, buf3]);
        assert_eq!(hash, hash_again);
    }

    #[test]
    fn test_hash_vertex_layout_attribute_format_variations() {
        let formats = [
            wgpu::VertexFormat::Float32,
            wgpu::VertexFormat::Float32x2,
            wgpu::VertexFormat::Float32x3,
            wgpu::VertexFormat::Float32x4,
            wgpu::VertexFormat::Sint32,
            wgpu::VertexFormat::Uint32,
            wgpu::VertexFormat::Unorm8x4,
            wgpu::VertexFormat::Float16x2,
        ];

        let hashes: Vec<u64> = formats
            .iter()
            .map(|f| {
                let buffer = VertexBufferLayoutDescriptor::per_vertex(16)
                    .with_attribute(*f, 0, 0);
                hash_vertex_layout(&[buffer])
            })
            .collect();

        for i in 0..hashes.len() {
            for j in (i + 1)..hashes.len() {
                assert_ne!(hashes[i], hashes[j], "Format {} and {} have same hash", i, j);
            }
        }
    }

    #[test]
    fn test_hash_vertex_layout_offset_variations() {
        let offsets = [0u64, 4, 8, 12, 16, 32, 64, 128];

        let hashes: Vec<u64> = offsets
            .iter()
            .map(|o| {
                let buffer = VertexBufferLayoutDescriptor::per_vertex(256)
                    .with_attribute(wgpu::VertexFormat::Float32x4, *o, 0);
                hash_vertex_layout(&[buffer])
            })
            .collect();

        for i in 0..hashes.len() {
            for j in (i + 1)..hashes.len() {
                assert_ne!(hashes[i], hashes[j]);
            }
        }
    }

    #[test]
    fn test_hash_vertex_layout_shader_location_variations() {
        let locations = [0u32, 1, 2, 5, 10, 15];

        let hashes: Vec<u64> = locations
            .iter()
            .map(|l| {
                let buffer = VertexBufferLayoutDescriptor::per_vertex(16)
                    .with_attribute(wgpu::VertexFormat::Float32x4, 0, *l);
                hash_vertex_layout(&[buffer])
            })
            .collect();

        for i in 0..hashes.len() {
            for j in (i + 1)..hashes.len() {
                assert_ne!(hashes[i], hashes[j]);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Additional Whitebox Tests - Hash Color Targets
    // -------------------------------------------------------------------------

    #[test]
    fn test_hash_color_targets_two_targets() {
        let target1 = ColorTargetStateDescriptor::srgb();
        let target2 = ColorTargetStateDescriptor::hdr();

        let hash = hash_color_targets(&[Some(target1.clone()), Some(target2.clone())]);
        let hash_again = hash_color_targets(&[Some(target1), Some(target2)]);
        assert_eq!(hash, hash_again);
    }

    #[test]
    fn test_hash_color_targets_four_targets() {
        let targets = vec![
            Some(ColorTargetStateDescriptor::new(wgpu::TextureFormat::Rgba8Unorm)),
            Some(ColorTargetStateDescriptor::new(wgpu::TextureFormat::Rgba16Float)),
            Some(ColorTargetStateDescriptor::new(wgpu::TextureFormat::Rgba8Unorm)),
            Some(ColorTargetStateDescriptor::new(wgpu::TextureFormat::R32Float)),
        ];

        let hash = hash_color_targets(&targets);
        let hash_again = hash_color_targets(&targets);
        assert_eq!(hash, hash_again);
    }

    #[test]
    fn test_hash_color_targets_eight_targets() {
        let targets: Vec<Option<ColorTargetStateDescriptor>> = (0..8)
            .map(|_| Some(ColorTargetStateDescriptor::srgb()))
            .collect();

        let hash = hash_color_targets(&targets);
        let hash_again = hash_color_targets(&targets);
        assert_eq!(hash, hash_again);
    }

    #[test]
    fn test_hash_color_targets_mixed_null() {
        let targets1 = vec![
            Some(ColorTargetStateDescriptor::srgb()),
            None,
            Some(ColorTargetStateDescriptor::hdr()),
        ];
        let targets2 = vec![
            Some(ColorTargetStateDescriptor::srgb()),
            Some(ColorTargetStateDescriptor::srgb()),
            Some(ColorTargetStateDescriptor::hdr()),
        ];

        assert_ne!(hash_color_targets(&targets1), hash_color_targets(&targets2));
    }

    #[test]
    fn test_hash_color_targets_blend_variations() {
        let no_blend = ColorTargetStateDescriptor::srgb();
        let alpha_blend = ColorTargetStateDescriptor::srgb().alpha_blend();
        let premul_blend = ColorTargetStateDescriptor::srgb().premultiplied_alpha();
        let additive_blend = ColorTargetStateDescriptor::srgb().additive();

        let hashes = [
            hash_color_targets(&[Some(no_blend)]),
            hash_color_targets(&[Some(alpha_blend)]),
            hash_color_targets(&[Some(premul_blend)]),
            hash_color_targets(&[Some(additive_blend)]),
        ];

        for i in 0..hashes.len() {
            for j in (i + 1)..hashes.len() {
                assert_ne!(hashes[i], hashes[j], "Blend {} and {} have same hash", i, j);
            }
        }
    }

    #[test]
    fn test_hash_color_targets_write_mask_variations() {
        let all = ColorTargetStateDescriptor::srgb()
            .write_mask(wgpu::ColorWrites::ALL);
        let none = ColorTargetStateDescriptor::srgb()
            .write_mask(wgpu::ColorWrites::empty());
        let red_only = ColorTargetStateDescriptor::srgb()
            .write_mask(wgpu::ColorWrites::RED);
        let rgb = ColorTargetStateDescriptor::srgb()
            .write_mask(wgpu::ColorWrites::COLOR);

        let hashes = [
            hash_color_targets(&[Some(all)]),
            hash_color_targets(&[Some(none)]),
            hash_color_targets(&[Some(red_only)]),
            hash_color_targets(&[Some(rgb)]),
        ];

        for i in 0..hashes.len() {
            for j in (i + 1)..hashes.len() {
                assert_ne!(hashes[i], hashes[j]);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Additional Whitebox Tests - CacheMetrics
    // -------------------------------------------------------------------------

    #[test]
    fn test_cache_metrics_hit_rate_high_precision() {
        let metrics = CacheMetrics {
            hits: 1,
            misses: 3,
            evictions: 0,
            invalidations: 0,
        };
        assert!((metrics.hit_rate() - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cache_metrics_hit_rate_small_numbers() {
        let metrics = CacheMetrics {
            hits: 1,
            misses: 0,
            evictions: 0,
            invalidations: 0,
        };
        assert_eq!(metrics.hit_rate(), 1.0);
    }

    #[test]
    fn test_cache_metrics_hit_rate_large_numbers() {
        let metrics = CacheMetrics {
            hits: 1_000_000,
            misses: 1_000_000,
            evictions: 0,
            invalidations: 0,
        };
        assert!((metrics.hit_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cache_metrics_total_accesses_zero() {
        let metrics = CacheMetrics::default();
        assert_eq!(metrics.total_accesses(), 0);
    }

    #[test]
    fn test_cache_metrics_total_accesses_overflow_safe() {
        // Test with large but not overflowing values
        let metrics = CacheMetrics {
            hits: u64::MAX / 2,
            misses: u64::MAX / 2,
            evictions: 0,
            invalidations: 0,
        };
        assert!(metrics.total_accesses() > 0);
    }

    #[test]
    fn test_cache_metrics_copy_trait() {
        let metrics = CacheMetrics {
            hits: 10,
            misses: 5,
            evictions: 2,
            invalidations: 1,
        };
        let metrics_copy = metrics;
        assert_eq!(metrics.hits, metrics_copy.hits);
        assert_eq!(metrics.misses, metrics_copy.misses);
    }

    // -------------------------------------------------------------------------
    // Additional Whitebox Tests - AtomicMetrics
    // -------------------------------------------------------------------------

    #[test]
    fn test_atomic_metrics_multiple_increments() {
        let metrics = AtomicMetrics::new();
        for _ in 0..100 {
            metrics.record_hit();
            metrics.record_miss();
        }
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.hits, 100);
        assert_eq!(snapshot.misses, 100);
    }

    #[test]
    fn test_atomic_metrics_mixed_operations() {
        let metrics = AtomicMetrics::new();
        metrics.record_hit();
        metrics.record_hit();
        metrics.record_miss();
        metrics.record_evictions(3);
        metrics.record_invalidations(2);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.hits, 2);
        assert_eq!(snapshot.misses, 1);
        assert_eq!(snapshot.evictions, 3);
        assert_eq!(snapshot.invalidations, 2);
    }

    #[test]
    fn test_atomic_metrics_reset_clears_all() {
        let metrics = AtomicMetrics::new();
        metrics.record_hit();
        metrics.record_miss();
        metrics.record_evictions(5);
        metrics.record_invalidations(3);

        metrics.reset();

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.hits, 0);
        assert_eq!(snapshot.misses, 0);
        assert_eq!(snapshot.evictions, 0);
        assert_eq!(snapshot.invalidations, 0);
    }

    #[test]
    fn test_atomic_metrics_snapshot_is_consistent() {
        let metrics = AtomicMetrics::new();
        metrics.record_hit();
        metrics.record_miss();

        let snapshot1 = metrics.snapshot();
        let snapshot2 = metrics.snapshot();

        assert_eq!(snapshot1.hits, snapshot2.hits);
        assert_eq!(snapshot1.misses, snapshot2.misses);
    }

    // -------------------------------------------------------------------------
    // Additional Whitebox Tests - Hash Stability
    // -------------------------------------------------------------------------

    #[test]
    fn test_hash_stability_complex_key() {
        let key = PipelineKey::new(999)
            .with_fragment_shader(888)
            .with_vertex_layout_hash(12345)
            .with_topology(wgpu::PrimitiveTopology::TriangleStrip)
            .with_front_face(wgpu::FrontFace::Cw)
            .with_cull_mode(Some(wgpu::Face::Front))
            .with_polygon_mode(wgpu::PolygonMode::Line)
            .with_depth_format(wgpu::TextureFormat::Depth32Float)
            .with_depth_write(false)
            .with_depth_compare(wgpu::CompareFunction::Always)
            .with_sample_count(4)
            .with_color_targets_hash(67890);

        let hash1 = hash_key(&key);
        let hash2 = hash_key(&key);
        let hash3 = hash_key(&key);
        let hash4 = hash_key(&key);
        let hash5 = hash_key(&key);

        assert_eq!(hash1, hash2);
        assert_eq!(hash2, hash3);
        assert_eq!(hash3, hash4);
        assert_eq!(hash4, hash5);
    }

    #[test]
    fn test_hash_vertex_layout_stability() {
        let buffer = VertexBufferLayoutDescriptor::per_vertex(48)
            .with_attribute(wgpu::VertexFormat::Float32x3, 0, 0)
            .with_attribute(wgpu::VertexFormat::Float32x3, 12, 1)
            .with_attribute(wgpu::VertexFormat::Float32x2, 24, 2)
            .with_attribute(wgpu::VertexFormat::Float32x4, 32, 3);

        let hash1 = hash_vertex_layout(&[buffer.clone()]);
        let hash2 = hash_vertex_layout(&[buffer.clone()]);
        let hash3 = hash_vertex_layout(&[buffer]);

        assert_eq!(hash1, hash2);
        assert_eq!(hash2, hash3);
    }

    #[test]
    fn test_hash_color_targets_stability() {
        let targets = vec![
            Some(ColorTargetStateDescriptor::srgb().alpha_blend()),
            Some(ColorTargetStateDescriptor::hdr()),
            None,
        ];

        let hash1 = hash_color_targets(&targets);
        let hash2 = hash_color_targets(&targets);
        let hash3 = hash_color_targets(&targets);

        assert_eq!(hash1, hash2);
        assert_eq!(hash2, hash3);
    }

    // -------------------------------------------------------------------------
    // Additional Whitebox Tests - PipelineKey Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_pipeline_key_zero_vertex_shader() {
        let key = PipelineKey::new(0);
        assert_eq!(key.vertex_shader_id, 0);
        assert!(key.uses_shader(0));
    }

    #[test]
    fn test_pipeline_key_max_shader_id() {
        let key = PipelineKey::new(u64::MAX).with_fragment_shader(u64::MAX - 1);
        assert_eq!(key.vertex_shader_id, u64::MAX);
        assert_eq!(key.fragment_shader_id, Some(u64::MAX - 1));
        assert!(key.uses_shader(u64::MAX));
        assert!(key.uses_shader(u64::MAX - 1));
    }

    #[test]
    fn test_pipeline_key_debug_format() {
        let key = PipelineKey::new(1).with_fragment_shader(2);
        let debug_str = format!("{:?}", key);
        assert!(debug_str.contains("PipelineKey"));
        assert!(debug_str.contains("vertex_shader_id"));
    }

    #[test]
    fn test_pipeline_key_partial_eq() {
        let key1 = PipelineKey::new(1)
            .with_topology(wgpu::PrimitiveTopology::TriangleList)
            .with_sample_count(4);
        let key2 = PipelineKey::new(1)
            .with_topology(wgpu::PrimitiveTopology::TriangleList)
            .with_sample_count(4);
        let key3 = PipelineKey::new(1)
            .with_topology(wgpu::PrimitiveTopology::TriangleStrip)
            .with_sample_count(4);

        assert!(key1 == key2);
        assert!(key1 != key3);
    }

    // -------------------------------------------------------------------------
    // Additional Whitebox Tests - Color Target Format Variations
    // -------------------------------------------------------------------------

    #[test]
    fn test_hash_color_targets_various_formats() {
        let formats = [
            wgpu::TextureFormat::R8Unorm,
            wgpu::TextureFormat::Rg8Unorm,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureFormat::Rgba32Float,
        ];

        let hashes: Vec<u64> = formats
            .iter()
            .map(|f| {
                let target = ColorTargetStateDescriptor::new(*f);
                hash_color_targets(&[Some(target)])
            })
            .collect();

        for i in 0..hashes.len() {
            for j in (i + 1)..hashes.len() {
                assert_ne!(hashes[i], hashes[j], "Format {} and {} have same hash", i, j);
            }
        }
    }

    #[test]
    fn test_pipeline_key_vertex_layout_hash_uniqueness() {
        let key1 = PipelineKey::new(1).with_vertex_layout_hash(0);
        let key2 = PipelineKey::new(1).with_vertex_layout_hash(1);
        let key3 = PipelineKey::new(1).with_vertex_layout_hash(u64::MAX);

        assert_ne!(key1, key2);
        assert_ne!(key1, key3);
        assert_ne!(key2, key3);
    }

    #[test]
    fn test_pipeline_key_color_targets_hash_uniqueness() {
        let key1 = PipelineKey::new(1).with_color_targets_hash(0);
        let key2 = PipelineKey::new(1).with_color_targets_hash(1);
        let key3 = PipelineKey::new(1).with_color_targets_hash(u64::MAX);

        assert_ne!(key1, key2);
        assert_ne!(key1, key3);
        assert_ne!(key2, key3);
    }
}
