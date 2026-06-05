//! Pipeline table with LRU eviction for GPU render pipeline caching.
//!
//! This module provides [`LruPipelineTable`], an extension of the basic
//! [`PipelineTable`] that adds LRU (Least Recently Used) eviction to manage
//! GPU memory when the cache exceeds a configurable maximum size.
//!
//! # Features
//!
//! - **LRU Eviction**: Automatically evicts least-recently-used pipelines when
//!   the cache exceeds `max_size`.
//! - **get_or_create_pipeline**: Convenience method that compiles and caches a
//!   pipeline in one call, returning cached pipelines for repeated requests.
//! - **Hot-reload Integration**: Methods to invalidate pipelines by shader hash,
//!   triggering recompilation on next access.
//! - **Statistics**: Hit/miss tracking for performance monitoring.
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::pipeline_table::LruPipelineTable;
//!
//! let mut table = LruPipelineTable::new(64); // max 64 pipelines
//!
//! // Get or create pipeline (compiles on first access)
//! let pipeline_id = table.get_or_create_pipeline(
//!     &device,
//!     wgsl_source,
//!     "vs_main",
//!     "fs_main",
//!     &[],
//!     wgpu::TextureFormat::Rgba8Unorm,
//! )?;
//!
//! // Subsequent calls return cached pipeline
//! let same_id = table.get_or_create_pipeline(...)?;
//! assert_eq!(pipeline_id, same_id);
//! ```

use std::collections::{HashMap, VecDeque};

use crate::pipeline::{CachedPipeline, ContentHash};
use crate::shader_cache::ShaderCacheV2;

// ---------------------------------------------------------------------------
// LruPipelineStats
// ---------------------------------------------------------------------------

/// Statistics for pipeline table performance monitoring.
#[derive(Debug, Clone, Default)]
pub struct LruPipelineStats {
    /// Number of cache hits (existing pipeline returned).
    pub hits: u64,
    /// Number of cache misses (new pipeline compiled).
    pub misses: u64,
    /// Number of pipelines evicted due to LRU policy.
    pub evictions: u64,
    /// Number of pipelines invalidated (hot-reload).
    pub invalidations: u64,
    /// Peak number of cached pipelines.
    pub peak_size: usize,
}

impl LruPipelineStats {
    /// Compute cache hit rate as a percentage [0.0, 100.0].
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 100.0;
        }
        (self.hits as f64 / total as f64) * 100.0
    }
}

// ---------------------------------------------------------------------------
// LruPipelineTable
// ---------------------------------------------------------------------------

/// LRU node tracking access order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct LruEntry {
    /// Pipeline ID.
    id: u32,
    /// Content hash of the shader source.
    shader_hash: ContentHash,
}

/// Pipeline table with LRU eviction and get_or_create semantics.
///
/// Combines a [`ShaderCacheV2`] for shader module deduplication with an
/// LRU-managed pipeline cache. When the number of cached pipelines exceeds
/// `max_size`, the least-recently-used pipelines are evicted.
///
/// # LRU Policy
///
/// - Accessing a pipeline (via `get` or `get_or_create_pipeline`) moves it to
///   the front of the LRU queue.
/// - When inserting a new pipeline would exceed `max_size`, the pipeline at the
///   back of the queue (least recently used) is evicted.
/// - Pipelines can be explicitly invalidated (for hot-reload), which removes
///   them from both the cache and the LRU queue.
///
/// # Thread Safety
///
/// This type is NOT thread-safe. For concurrent access, wrap it in a
/// `RwLock` or use [`ShardedPipelineTable`] for sharded locking.
///
/// [`ShardedPipelineTable`]: crate::pipeline::ShardedPipelineTable
pub struct LruPipelineTable {
    /// Cached pipelines indexed by ID.
    pipelines: HashMap<u32, CachedPipeline>,
    /// Maps content hash to pipeline ID for get_or_create.
    hash_to_id: HashMap<ContentHash, u32>,
    /// LRU queue: front = most recently used, back = least recently used.
    lru_queue: VecDeque<u32>,
    /// Shared shader cache.
    shader_cache: ShaderCacheV2,
    /// Maximum number of pipelines before eviction.
    max_size: usize,
    /// Next available pipeline ID.
    next_id: u32,
    /// Statistics.
    stats: LruPipelineStats,
}

impl LruPipelineTable {
    /// Create a new LRU pipeline table with the given maximum size.
    ///
    /// When the cache exceeds `max_size` pipelines, the least-recently-used
    /// entries are evicted to make room for new ones.
    ///
    /// # Panics
    ///
    /// Panics if `max_size` is 0.
    pub fn new(max_size: usize) -> Self {
        assert!(max_size > 0, "max_size must be greater than 0");
        Self {
            pipelines: HashMap::new(),
            hash_to_id: HashMap::new(),
            lru_queue: VecDeque::new(),
            shader_cache: ShaderCacheV2::new(),
            max_size,
            next_id: 1,
            stats: LruPipelineStats::default(),
        }
    }

    /// Create a pipeline table with unlimited size (no eviction).
    ///
    /// Equivalent to `new(usize::MAX)`.
    pub fn unbounded() -> Self {
        Self::new(usize::MAX)
    }

    /// Get or create a render pipeline for the given WGSL source.
    ///
    /// If a pipeline with the same shader content hash already exists, its ID
    /// is returned (cache hit). Otherwise, a new pipeline is compiled, cached,
    /// and its ID is returned (cache miss).
    ///
    /// # LRU Behavior
    ///
    /// - Cache hit: existing pipeline is moved to front of LRU queue.
    /// - Cache miss: new pipeline is added to front; if cache is full, least-
    ///   recently-used pipeline is evicted.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `wgsl_source` - WGSL shader source (combined vertex + fragment).
    /// * `vertex_entry` - Vertex shader entry point name.
    /// * `fragment_entry` - Fragment shader entry point name.
    /// * `vertex_layouts` - Vertex buffer layouts.
    /// * `color_format` - Output color attachment format.
    ///
    /// # Returns
    ///
    /// `Ok(pipeline_id)` on success, `Err(msg)` if compilation fails.
    pub fn get_or_create_pipeline(
        &mut self,
        device: &wgpu::Device,
        wgsl_source: &str,
        vertex_entry: &str,
        fragment_entry: &str,
        vertex_layouts: &[wgpu::VertexBufferLayout<'_>],
        color_format: wgpu::TextureFormat,
    ) -> Result<u32, String> {
        let content_hash = ContentHash::from_bytes(wgsl_source.as_bytes());

        // Check for existing pipeline with same hash
        if let Some(&existing_id) = self.hash_to_id.get(&content_hash) {
            self.stats.hits += 1;
            self.touch_lru(existing_id);
            return Ok(existing_id);
        }

        // Cache miss: compile new pipeline
        self.stats.misses += 1;

        // Get or compile shader module
        let (module, _shader_hash) = self.shader_cache.cache_shader(device, wgsl_source);

        // Allocate new ID
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);

        // Create bind group layout and pipeline layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("LruPipeline {} BGL", id)),
            entries: &[],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("LruPipeline {} Layout", id)),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create render pipeline
        let render_pipeline =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(&format!("LruPipeline {}", id)),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: vertex_entry,
                        buffers: vertex_layouts,
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &module,
                        entry_point: fragment_entry,
                        targets: &[Some(wgpu::ColorTargetState {
                            format: color_format,
                            blend: Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: Some(wgpu::Face::Back),
                        unclipped_depth: false,
                        polygon_mode: wgpu::PolygonMode::Fill,
                        conservative: false,
                    },
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState {
                        count: 1,
                        mask: !0,
                        alpha_to_coverage_enabled: false,
                    },
                    multiview: None,
                    cache: None,
                })
            }))
            .map_err(|panic_payload| {
                let msg = panic_payload
                    .downcast_ref::<&str>()
                    .copied()
                    .or_else(|| panic_payload.downcast_ref::<String>().map(|s| s.as_str()))
                    .unwrap_or("unknown wgpu panic");
                format!("pipeline compilation panicked: {msg}")
            })?;

        let cached = CachedPipeline {
            id,
            render_pipeline,
            bind_group_layout,
            shader_hash: content_hash.into_bytes(),
        };

        // Insert and evict if necessary
        self.insert_with_eviction(id, content_hash, cached);

        Ok(id)
    }

    /// Insert a pipeline, evicting LRU entries if necessary.
    fn insert_with_eviction(&mut self, id: u32, hash: ContentHash, pipeline: CachedPipeline) {
        // Evict if at capacity
        while self.pipelines.len() >= self.max_size {
            if let Some(evict_id) = self.lru_queue.pop_back() {
                if let Some(evicted) = self.pipelines.remove(&evict_id) {
                    let evicted_hash = ContentHash::from_raw(evicted.shader_hash);
                    self.hash_to_id.remove(&evicted_hash);
                    self.stats.evictions += 1;
                }
            } else {
                break; // Queue is empty, shouldn't happen
            }
        }

        // Insert new pipeline
        self.pipelines.insert(id, pipeline);
        self.hash_to_id.insert(hash, id);
        self.lru_queue.push_front(id);

        // Update peak size
        if self.pipelines.len() > self.stats.peak_size {
            self.stats.peak_size = self.pipelines.len();
        }
    }

    /// Move a pipeline ID to the front of the LRU queue (most recently used).
    fn touch_lru(&mut self, id: u32) {
        // Remove from current position
        self.lru_queue.retain(|&x| x != id);
        // Add to front
        self.lru_queue.push_front(id);
    }

    /// Insert a pre-built pipeline.
    ///
    /// If a pipeline with the same ID exists, it is replaced.
    pub fn insert(&mut self, pipeline: CachedPipeline) {
        let id = pipeline.id;
        let hash = ContentHash::from_raw(pipeline.shader_hash);

        // Remove old entry with same ID if exists
        if self.pipelines.contains_key(&id) {
            self.lru_queue.retain(|&x| x != id);
        }

        self.insert_with_eviction(id, hash, pipeline);
    }

    /// Get a pipeline by ID.
    ///
    /// Returns `None` if no pipeline with that ID exists.
    /// **Does NOT update LRU order** (use `get_or_create_pipeline` for that).
    pub fn get(&self, id: u32) -> Option<&CachedPipeline> {
        self.pipelines.get(&id)
    }

    /// Get a pipeline by ID and update LRU order.
    pub fn get_touch(&mut self, id: u32) -> Option<&CachedPipeline> {
        if self.pipelines.contains_key(&id) {
            self.touch_lru(id);
            self.pipelines.get(&id)
        } else {
            None
        }
    }

    /// Check if a pipeline with the given ID exists.
    pub fn contains(&self, id: u32) -> bool {
        self.pipelines.contains_key(&id)
    }

    /// Check if a pipeline with the given content hash exists.
    pub fn contains_hash(&self, hash: &ContentHash) -> bool {
        self.hash_to_id.contains_key(hash)
    }

    /// Get the pipeline ID for a given content hash.
    pub fn id_for_hash(&self, hash: &ContentHash) -> Option<u32> {
        self.hash_to_id.get(hash).copied()
    }

    /// Remove a pipeline by ID.
    ///
    /// Returns `true` if the pipeline was found and removed.
    pub fn remove(&mut self, id: u32) -> bool {
        if let Some(removed) = self.pipelines.remove(&id) {
            let hash = ContentHash::from_raw(removed.shader_hash);
            self.hash_to_id.remove(&hash);
            self.lru_queue.retain(|&x| x != id);
            true
        } else {
            false
        }
    }

    /// Invalidate all pipelines using the given shader hash.
    ///
    /// Called when a shader source changes and all pipelines using it need
    /// recompilation. Returns the IDs of invalidated pipelines.
    ///
    /// After invalidation, subsequent `get_or_create_pipeline` calls with the
    /// same source will recompile the pipeline.
    pub fn invalidate_by_hash(&mut self, hash: &ContentHash) -> Vec<u32> {
        let mut invalidated = Vec::new();

        if let Some(id) = self.hash_to_id.remove(hash) {
            if self.pipelines.remove(&id).is_some() {
                self.lru_queue.retain(|&x| x != id);
                self.stats.invalidations += 1;
                invalidated.push(id);
            }
        }

        invalidated
    }

    /// Number of cached pipelines.
    pub fn len(&self) -> usize {
        self.pipelines.len()
    }

    /// Returns true if no pipelines are cached.
    pub fn is_empty(&self) -> bool {
        self.pipelines.is_empty()
    }

    /// Maximum pipeline cache size.
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// Set maximum cache size, evicting if necessary.
    pub fn set_max_size(&mut self, max_size: usize) {
        assert!(max_size > 0, "max_size must be greater than 0");
        self.max_size = max_size;

        // Evict until we're under the new limit
        while self.pipelines.len() > self.max_size {
            if let Some(evict_id) = self.lru_queue.pop_back() {
                if let Some(evicted) = self.pipelines.remove(&evict_id) {
                    let evicted_hash = ContentHash::from_raw(evicted.shader_hash);
                    self.hash_to_id.remove(&evicted_hash);
                    self.stats.evictions += 1;
                }
            } else {
                break;
            }
        }
    }

    /// Clear all cached pipelines.
    pub fn clear(&mut self) {
        self.pipelines.clear();
        self.hash_to_id.clear();
        self.lru_queue.clear();
        self.shader_cache.clear();
    }

    /// Get cache statistics.
    pub fn stats(&self) -> &LruPipelineStats {
        &self.stats
    }

    /// Reset statistics (useful after warm-up period).
    pub fn reset_stats(&mut self) {
        self.stats.hits = 0;
        self.stats.misses = 0;
        self.stats.evictions = 0;
        self.stats.invalidations = 0;
    }

    /// Access the underlying shader cache.
    pub fn shader_cache(&self) -> &ShaderCacheV2 {
        &self.shader_cache
    }

    /// Access the underlying shader cache mutably.
    pub fn shader_cache_mut(&mut self) -> &mut ShaderCacheV2 {
        &mut self.shader_cache
    }

    /// List all cached pipeline IDs.
    pub fn cached_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.pipelines.keys().copied()
    }

    /// Get the LRU order (most recent first).
    pub fn lru_order(&self) -> &VecDeque<u32> {
        &self.lru_queue
    }
}

impl Default for LruPipelineTable {
    fn default() -> Self {
        Self::new(64) // Default max 64 pipelines
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create a test device
    fn create_test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;
        Some(
            pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("test device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            ))
            .expect("device creation"),
        )
    }

    #[test]
    fn test_lru_pipeline_table_new() {
        let table = LruPipelineTable::new(10);
        assert!(table.is_empty());
        assert_eq!(table.len(), 0);
        assert_eq!(table.max_size(), 10);
    }

    #[test]
    #[should_panic(expected = "max_size must be greater than 0")]
    fn test_lru_pipeline_table_zero_size_panics() {
        let _ = LruPipelineTable::new(0);
    }

    #[test]
    fn test_lru_pipeline_table_unbounded() {
        let table = LruPipelineTable::unbounded();
        assert_eq!(table.max_size(), usize::MAX);
    }

    #[test]
    fn test_lru_pipeline_table_get_or_create() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let mut table = LruPipelineTable::new(10);

        let src = r#"
            @vertex fn vs_main() -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }
            @fragment fn fs_main() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0, 0.0, 0.0, 1.0);
            }
        "#;

        // First call: cache miss
        let id1 = table
            .get_or_create_pipeline(&device, src, "vs_main", "fs_main", &[], wgpu::TextureFormat::Rgba8Unorm)
            .expect("pipeline creation");

        assert_eq!(table.stats().misses, 1);
        assert_eq!(table.stats().hits, 0);
        assert_eq!(table.len(), 1);

        // Second call: cache hit
        let id2 = table
            .get_or_create_pipeline(&device, src, "vs_main", "fs_main", &[], wgpu::TextureFormat::Rgba8Unorm)
            .expect("pipeline creation");

        assert_eq!(id1, id2);
        assert_eq!(table.stats().misses, 1);
        assert_eq!(table.stats().hits, 1);
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn test_lru_pipeline_table_eviction() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        // Small cache: max 2 pipelines
        let mut table = LruPipelineTable::new(2);

        let make_src = |n: u32| {
            format!(
                r#"
                @vertex fn vs_{n}() -> @builtin(position) vec4<f32> {{
                    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
                }}
                @fragment fn fs_{n}() -> @location(0) vec4<f32> {{
                    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
                }}
                "#,
                n = n
            )
        };

        // Add 3 pipelines to a cache of size 2
        let id1 = table
            .get_or_create_pipeline(
                &device,
                &make_src(1),
                "vs_1",
                "fs_1",
                &[],
                wgpu::TextureFormat::Rgba8Unorm,
            )
            .unwrap();

        let id2 = table
            .get_or_create_pipeline(
                &device,
                &make_src(2),
                "vs_2",
                "fs_2",
                &[],
                wgpu::TextureFormat::Rgba8Unorm,
            )
            .unwrap();

        assert_eq!(table.len(), 2);
        assert_eq!(table.stats().evictions, 0);

        // Adding third pipeline should evict id1 (LRU)
        let _id3 = table
            .get_or_create_pipeline(
                &device,
                &make_src(3),
                "vs_3",
                "fs_3",
                &[],
                wgpu::TextureFormat::Rgba8Unorm,
            )
            .unwrap();

        assert_eq!(table.len(), 2);
        assert_eq!(table.stats().evictions, 1);
        assert!(!table.contains(id1)); // id1 was evicted
        assert!(table.contains(id2));
    }

    #[test]
    fn test_lru_pipeline_table_touch_updates_order() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let mut table = LruPipelineTable::new(2);

        let make_src = |n: u32| {
            format!(
                r#"
                @vertex fn vs_{n}() -> @builtin(position) vec4<f32> {{
                    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
                }}
                @fragment fn fs_{n}() -> @location(0) vec4<f32> {{
                    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
                }}
                "#,
                n = n
            )
        };

        // Add pipelines 1 and 2
        let src1 = make_src(1);
        let src2 = make_src(2);
        let src3 = make_src(3);

        let id1 = table
            .get_or_create_pipeline(&device, &src1, "vs_1", "fs_1", &[], wgpu::TextureFormat::Rgba8Unorm)
            .unwrap();
        let id2 = table
            .get_or_create_pipeline(&device, &src2, "vs_2", "fs_2", &[], wgpu::TextureFormat::Rgba8Unorm)
            .unwrap();

        // Touch id1 (move to front)
        let _ = table.get_or_create_pipeline(&device, &src1, "vs_1", "fs_1", &[], wgpu::TextureFormat::Rgba8Unorm);

        // Now id2 is LRU, adding id3 should evict id2
        let _id3 = table
            .get_or_create_pipeline(&device, &src3, "vs_3", "fs_3", &[], wgpu::TextureFormat::Rgba8Unorm)
            .unwrap();

        assert!(table.contains(id1)); // id1 was touched, not evicted
        assert!(!table.contains(id2)); // id2 was LRU, evicted
    }

    #[test]
    fn test_lru_pipeline_table_invalidate_by_hash() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let mut table = LruPipelineTable::new(10);

        let src = r#"
            @vertex fn vs_main() -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }
            @fragment fn fs_main() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0, 0.0, 0.0, 1.0);
            }
        "#;

        let id = table
            .get_or_create_pipeline(&device, src, "vs_main", "fs_main", &[], wgpu::TextureFormat::Rgba8Unorm)
            .unwrap();

        let hash = ContentHash::from_bytes(src.as_bytes());

        assert!(table.contains(id));
        assert!(table.contains_hash(&hash));

        // Invalidate
        let invalidated = table.invalidate_by_hash(&hash);
        assert_eq!(invalidated, vec![id]);
        assert_eq!(table.stats().invalidations, 1);

        assert!(!table.contains(id));
        assert!(!table.contains_hash(&hash));

        // Next get_or_create should recompile
        let new_id = table
            .get_or_create_pipeline(&device, src, "vs_main", "fs_main", &[], wgpu::TextureFormat::Rgba8Unorm)
            .unwrap();

        assert_ne!(id, new_id);
        assert_eq!(table.stats().misses, 2);
    }

    #[test]
    fn test_lru_pipeline_table_set_max_size() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let mut table = LruPipelineTable::new(5);

        let make_src = |n: u32| {
            format!(
                r#"
                @vertex fn vs_{n}() -> @builtin(position) vec4<f32> {{
                    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
                }}
                @fragment fn fs_{n}() -> @location(0) vec4<f32> {{
                    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
                }}
                "#,
                n = n
            )
        };

        // Add 5 pipelines
        for n in 1..=5 {
            let src = make_src(n);
            let entry_vs = format!("vs_{}", n);
            let entry_fs = format!("fs_{}", n);
            table
                .get_or_create_pipeline(&device, &src, &entry_vs, &entry_fs, &[], wgpu::TextureFormat::Rgba8Unorm)
                .unwrap();
        }

        assert_eq!(table.len(), 5);

        // Reduce max size to 2
        table.set_max_size(2);

        assert_eq!(table.len(), 2);
        assert_eq!(table.stats().evictions, 3);
    }

    #[test]
    fn test_lru_pipeline_table_remove() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let mut table = LruPipelineTable::new(10);

        let src = r#"
            @vertex fn vs_main() -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }
            @fragment fn fs_main() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0, 0.0, 0.0, 1.0);
            }
        "#;

        let id = table
            .get_or_create_pipeline(&device, src, "vs_main", "fs_main", &[], wgpu::TextureFormat::Rgba8Unorm)
            .unwrap();

        assert!(table.contains(id));
        assert!(table.remove(id));
        assert!(!table.contains(id));
        assert!(!table.remove(id)); // Already removed
    }

    #[test]
    fn test_lru_pipeline_table_clear() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let mut table = LruPipelineTable::new(10);

        let src = r#"
            @vertex fn vs_main() -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }
            @fragment fn fs_main() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0, 0.0, 0.0, 1.0);
            }
        "#;

        table
            .get_or_create_pipeline(&device, src, "vs_main", "fs_main", &[], wgpu::TextureFormat::Rgba8Unorm)
            .unwrap();

        assert!(!table.is_empty());
        table.clear();
        assert!(table.is_empty());
    }

    #[test]
    fn test_lru_pipeline_table_stats() {
        let stats = LruPipelineStats::default();
        assert_eq!(stats.hit_rate(), 100.0); // No lookups

        let stats = LruPipelineStats {
            hits: 80,
            misses: 20,
            ..Default::default()
        };
        assert_eq!(stats.hit_rate(), 80.0);
    }

    #[test]
    fn test_lru_pipeline_table_get_touch() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let mut table = LruPipelineTable::new(10);

        let src = r#"
            @vertex fn vs_main() -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }
            @fragment fn fs_main() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0, 0.0, 0.0, 1.0);
            }
        "#;

        let id = table
            .get_or_create_pipeline(&device, src, "vs_main", "fs_main", &[], wgpu::TextureFormat::Rgba8Unorm)
            .unwrap();

        // get doesn't update LRU
        let p1 = table.get(id);
        assert!(p1.is_some());

        // get_touch updates LRU
        let p2 = table.get_touch(id);
        assert!(p2.is_some());

        // Missing ID
        assert!(table.get(9999).is_none());
        assert!(table.get_touch(9999).is_none());
    }

    #[test]
    fn test_lru_pipeline_table_lru_order() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let mut table = LruPipelineTable::new(10);

        let make_src = |n: u32| {
            format!(
                r#"
                @vertex fn vs_{n}() -> @builtin(position) vec4<f32> {{
                    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
                }}
                @fragment fn fs_{n}() -> @location(0) vec4<f32> {{
                    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
                }}
                "#,
                n = n
            )
        };

        let src1 = make_src(1);
        let src2 = make_src(2);
        let src3 = make_src(3);

        let id1 = table.get_or_create_pipeline(&device, &src1, "vs_1", "fs_1", &[], wgpu::TextureFormat::Rgba8Unorm).unwrap();
        let id2 = table.get_or_create_pipeline(&device, &src2, "vs_2", "fs_2", &[], wgpu::TextureFormat::Rgba8Unorm).unwrap();
        let id3 = table.get_or_create_pipeline(&device, &src3, "vs_3", "fs_3", &[], wgpu::TextureFormat::Rgba8Unorm).unwrap();

        // Order: [id3, id2, id1] (most recent first)
        let order: Vec<_> = table.lru_order().iter().copied().collect();
        assert_eq!(order, vec![id3, id2, id1]);

        // Touch id1
        let _ = table.get_or_create_pipeline(&device, &src1, "vs_1", "fs_1", &[], wgpu::TextureFormat::Rgba8Unorm);

        // Order: [id1, id3, id2]
        let order: Vec<_> = table.lru_order().iter().copied().collect();
        assert_eq!(order, vec![id1, id3, id2]);
    }

    #[test]
    fn test_lru_pipeline_table_id_for_hash() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let mut table = LruPipelineTable::new(10);

        let src = r#"
            @vertex fn vs_main() -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }
            @fragment fn fs_main() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0, 0.0, 0.0, 1.0);
            }
        "#;

        let id = table
            .get_or_create_pipeline(&device, src, "vs_main", "fs_main", &[], wgpu::TextureFormat::Rgba8Unorm)
            .unwrap();

        let hash = ContentHash::from_bytes(src.as_bytes());
        assert_eq!(table.id_for_hash(&hash), Some(id));

        let missing_hash = ContentHash::from_bytes(b"other");
        assert_eq!(table.id_for_hash(&missing_hash), None);
    }

    #[test]
    fn test_lru_pipeline_table_cached_ids() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let mut table = LruPipelineTable::new(10);

        let make_src = |n: u32| {
            format!(
                r#"
                @vertex fn vs_{n}() -> @builtin(position) vec4<f32> {{
                    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
                }}
                @fragment fn fs_{n}() -> @location(0) vec4<f32> {{
                    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
                }}
                "#,
                n = n
            )
        };

        let mut ids = Vec::new();
        for n in 1..=3 {
            let src = make_src(n);
            let entry_vs = format!("vs_{}", n);
            let entry_fs = format!("fs_{}", n);
            let id = table
                .get_or_create_pipeline(&device, &src, &entry_vs, &entry_fs, &[], wgpu::TextureFormat::Rgba8Unorm)
                .unwrap();
            ids.push(id);
        }

        let cached: Vec<_> = table.cached_ids().collect();
        assert_eq!(cached.len(), 3);
        for id in &ids {
            assert!(cached.contains(id));
        }
    }
}
