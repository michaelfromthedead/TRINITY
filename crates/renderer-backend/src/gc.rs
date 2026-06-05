//! Tree Store Garbage Collection for ContentTree structures.
//!
//! This module provides mark-and-sweep garbage collection for content-addressed
//! storage with the following features:
//!
//! - **RootSet**: Manages the set of live roots (active materials, assets, pipeline cache)
//! - **TreeGarbageCollector**: Mark-and-sweep GC with BFS traversal from root set
//! - **RefCountedStore**: Wrapper enabling immediate cleanup via reference counting
//! - **BackgroundGC**: Time-sliced GC running in a background thread
//!
//! # Time Budget
//!
//! GC operations are designed to run within a 2ms per-frame budget to avoid
//! impacting render performance. The `GCConfig::max_time_budget_ms` setting
//! controls this limit.
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::gc::{TreeGarbageCollector, RootSet, GCConfig};
//! use renderer_backend::pipeline::FileBackend;
//!
//! let store = FileBackend::new("/tmp/content_store")?;
//! let mut roots = RootSet::new();
//!
//! // Add active roots
//! roots.add(material_tree_hash);
//! roots.add(pipeline_cache_hash);
//!
//! // Run GC
//! let config = GCConfig::default();
//! let mut gc = TreeGarbageCollector::new(&store, roots, config);
//! let stats = gc.run_gc()?;
//!
//! println!("Collected {} orphans in {:?}", stats.deleted_count, stats.elapsed);
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use std::io;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::pipeline::{ChunkedContent, ContentHash, ContentTree, FileBackend};

// ---------------------------------------------------------------------------
// RootSet — manages live roots for GC
// ---------------------------------------------------------------------------

/// A set of live root hashes that should not be collected.
///
/// Roots typically include:
/// - Active material trees
/// - Asset references currently in use
/// - Pipeline cache entries
/// - Shader sources being compiled
///
/// # Thread Safety
///
/// `RootSet` is designed for single-threaded use. For concurrent access,
/// wrap in `Arc<RwLock<RootSet>>`.
#[derive(Debug, Clone, Default)]
pub struct RootSet {
    /// Set of live root hashes.
    roots: HashSet<ContentHash>,
    /// Optional labels for debugging/inspection.
    labels: HashMap<ContentHash, String>,
}

impl RootSet {
    /// Create an empty root set.
    pub fn new() -> Self {
        Self {
            roots: HashSet::new(),
            labels: HashMap::new(),
        }
    }

    /// Create a root set with initial capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            roots: HashSet::with_capacity(capacity),
            labels: HashMap::with_capacity(capacity),
        }
    }

    /// Create a root set from a vector of hashes.
    pub fn from_hashes(hashes: Vec<ContentHash>) -> Self {
        Self {
            roots: hashes.into_iter().collect(),
            labels: HashMap::new(),
        }
    }

    /// Add a root hash.
    pub fn add(&mut self, hash: ContentHash) {
        self.roots.insert(hash);
    }

    /// Add a root hash with a label for debugging.
    pub fn add_labeled(&mut self, hash: ContentHash, label: impl Into<String>) {
        self.roots.insert(hash);
        self.labels.insert(hash, label.into());
    }

    /// Remove a root hash.
    pub fn remove(&mut self, hash: &ContentHash) -> bool {
        self.labels.remove(hash);
        self.roots.remove(hash)
    }

    /// Check if a hash is in the root set.
    pub fn contains(&self, hash: &ContentHash) -> bool {
        self.roots.contains(hash)
    }

    /// Get the label for a hash, if any.
    pub fn label(&self, hash: &ContentHash) -> Option<&str> {
        self.labels.get(hash).map(|s| s.as_str())
    }

    /// Get the number of roots.
    pub fn len(&self) -> usize {
        self.roots.len()
    }

    /// Check if the root set is empty.
    pub fn is_empty(&self) -> bool {
        self.roots.is_empty()
    }

    /// Clear all roots.
    pub fn clear(&mut self) {
        self.roots.clear();
        self.labels.clear();
    }

    /// Get an iterator over all root hashes.
    pub fn iter(&self) -> impl Iterator<Item = &ContentHash> {
        self.roots.iter()
    }

    /// Convert to a vector of hashes.
    pub fn to_vec(&self) -> Vec<ContentHash> {
        self.roots.iter().cloned().collect()
    }

    /// Merge another root set into this one.
    pub fn merge(&mut self, other: &RootSet) {
        for hash in &other.roots {
            self.roots.insert(*hash);
        }
        for (hash, label) in &other.labels {
            self.labels.insert(*hash, label.clone());
        }
    }
}

// ---------------------------------------------------------------------------
// GCConfig — configuration for garbage collection
// ---------------------------------------------------------------------------

/// Configuration for tree garbage collection.
#[derive(Debug, Clone)]
pub struct GCConfig {
    /// Maximum time budget per GC run in milliseconds.
    /// Default: 2ms (to fit within frame budget).
    pub max_time_budget_ms: u64,

    /// Enable reference counting for immediate cleanup.
    /// When enabled, objects with ref count 0 are immediately eligible for deletion.
    /// Default: true
    pub enable_ref_counting: bool,

    /// Whether to actually delete orphans (false = dry run).
    /// Default: true
    pub delete_orphans: bool,

    /// Batch size for mark phase (number of hashes to process per iteration).
    /// Higher values = faster GC but longer pauses.
    /// Default: 100
    pub mark_batch_size: usize,

    /// Batch size for sweep phase (number of deletions per iteration).
    /// Default: 50
    pub sweep_batch_size: usize,

    /// Enable verbose logging.
    /// Default: false
    pub verbose: bool,
}

impl Default for GCConfig {
    fn default() -> Self {
        Self {
            max_time_budget_ms: 2,
            enable_ref_counting: true,
            delete_orphans: true,
            mark_batch_size: 100,
            sweep_batch_size: 50,
            verbose: false,
        }
    }
}

impl GCConfig {
    /// Create a config with the specified time budget in milliseconds.
    pub fn with_time_budget_ms(mut self, ms: u64) -> Self {
        self.max_time_budget_ms = ms;
        self
    }

    /// Create a config for dry-run mode (no deletions).
    pub fn dry_run() -> Self {
        Self {
            delete_orphans: false,
            ..Default::default()
        }
    }

    /// Create an aggressive config for faster GC (larger batches).
    pub fn aggressive() -> Self {
        Self {
            max_time_budget_ms: 10,
            mark_batch_size: 500,
            sweep_batch_size: 200,
            ..Default::default()
        }
    }

    /// Get the time budget as a Duration.
    pub fn time_budget(&self) -> Duration {
        Duration::from_millis(self.max_time_budget_ms)
    }
}

// ---------------------------------------------------------------------------
// GCStats — statistics from a GC run
// ---------------------------------------------------------------------------

/// Statistics from a garbage collection run.
#[derive(Debug, Clone, Default)]
pub struct GCStats {
    /// Number of hashes marked as reachable.
    pub marked_count: usize,

    /// Number of orphan blobs found.
    pub orphan_count: usize,

    /// Number of orphan blobs deleted.
    pub deleted_count: usize,

    /// Total bytes deleted.
    pub bytes_deleted: u64,

    /// Whether GC completed within time budget.
    pub completed: bool,

    /// Time spent on this GC run.
    pub elapsed: Duration,

    /// Time spent in mark phase.
    pub mark_elapsed: Duration,

    /// Time spent in sweep phase.
    pub sweep_elapsed: Duration,

    /// Number of trees traversed.
    pub trees_traversed: usize,

    /// Number of chunked manifests traversed.
    pub manifests_traversed: usize,
}

impl GCStats {
    /// Merge another stats object into this one.
    pub fn merge(&mut self, other: &GCStats) {
        self.marked_count += other.marked_count;
        self.orphan_count += other.orphan_count;
        self.deleted_count += other.deleted_count;
        self.bytes_deleted += other.bytes_deleted;
        self.completed = self.completed && other.completed;
        self.elapsed += other.elapsed;
        self.mark_elapsed += other.mark_elapsed;
        self.sweep_elapsed += other.sweep_elapsed;
        self.trees_traversed += other.trees_traversed;
        self.manifests_traversed += other.manifests_traversed;
    }
}

// ---------------------------------------------------------------------------
// GCPhase — tracks current GC phase for incremental collection
// ---------------------------------------------------------------------------

/// Current phase of garbage collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GCPhase {
    /// Not running.
    Idle,
    /// Marking reachable objects from roots.
    Marking,
    /// Sweeping (deleting) unreachable objects.
    Sweeping,
    /// GC run complete.
    Complete,
}

// ---------------------------------------------------------------------------
// TreeGarbageCollector — mark-and-sweep GC for ContentTree store
// ---------------------------------------------------------------------------

/// Mark-and-sweep garbage collector for ContentTree storage.
///
/// The GC operates in two phases:
///
/// 1. **Mark Phase**: BFS traversal from all roots, marking reachable hashes.
///    Traverses both `ContentTree` and `ChunkedContent` structures.
///
/// 2. **Sweep Phase**: Iterates all stored hashes and deletes those not marked.
///
/// Both phases are time-sliced to stay within the configured frame budget.
///
/// # Example
///
/// ```ignore
/// let mut gc = TreeGarbageCollector::new(&store, roots, GCConfig::default());
/// let stats = gc.run_gc()?;
/// println!("Deleted {} orphans", stats.deleted_count);
/// ```
pub struct TreeGarbageCollector<'a> {
    /// Backend storage.
    backend: &'a FileBackend,
    /// Root set (live objects).
    roots: RootSet,
    /// Marked (reachable) hashes.
    marked: HashSet<ContentHash>,
    /// Queue for BFS traversal.
    mark_queue: VecDeque<ContentHash>,
    /// Configuration.
    config: GCConfig,
    /// Current phase.
    phase: GCPhase,
    /// Statistics.
    stats: GCStats,
    /// All hashes in store (cached for sweep phase).
    all_hashes: Option<Vec<ContentHash>>,
    /// Current index in sweep phase.
    sweep_index: usize,
}

impl<'a> TreeGarbageCollector<'a> {
    /// Create a new garbage collector.
    pub fn new(backend: &'a FileBackend, roots: RootSet, config: GCConfig) -> Self {
        let mark_queue: VecDeque<ContentHash> = roots.iter().cloned().collect();

        Self {
            backend,
            roots,
            marked: HashSet::new(),
            mark_queue,
            config,
            phase: GCPhase::Idle,
            stats: GCStats::default(),
            all_hashes: None,
            sweep_index: 0,
        }
    }

    /// Create a GC with default configuration.
    pub fn with_defaults(backend: &'a FileBackend, roots: RootSet) -> Self {
        Self::new(backend, roots, GCConfig::default())
    }

    /// Get the current GC phase.
    pub fn phase(&self) -> GCPhase {
        self.phase
    }

    /// Get the current statistics.
    pub fn stats(&self) -> &GCStats {
        &self.stats
    }

    /// Get the set of marked hashes.
    pub fn marked(&self) -> &HashSet<ContentHash> {
        &self.marked
    }

    /// Run the mark phase (BFS from roots).
    ///
    /// Returns the number of newly marked hashes in this call.
    /// Call repeatedly until it returns 0 or time budget is exceeded.
    pub fn mark_phase(&mut self) -> io::Result<usize> {
        if self.phase == GCPhase::Idle {
            self.phase = GCPhase::Marking;
        }

        if self.phase != GCPhase::Marking {
            return Ok(0);
        }

        let start = Instant::now();
        let budget = self.config.time_budget();
        let batch_size = self.config.mark_batch_size;
        let mut newly_marked = 0;

        for _ in 0..batch_size {
            if start.elapsed() > budget {
                break;
            }

            let hash = match self.mark_queue.pop_front() {
                Some(h) => h,
                None => {
                    // Mark phase complete
                    self.phase = GCPhase::Sweeping;
                    break;
                }
            };

            if self.marked.contains(&hash) {
                continue;
            }

            self.marked.insert(hash);
            newly_marked += 1;

            // Try to load and traverse children
            if let Some(data) = self.backend.get(&hash)? {
                // Try as ContentTree
                if let Ok(tree) = ContentTree::deserialize(&data) {
                    self.stats.trees_traversed += 1;
                    for entry in tree.entries() {
                        if !self.marked.contains(&entry.hash) {
                            self.mark_queue.push_back(entry.hash);
                        }
                    }
                }

                // Try as ChunkedContent manifest
                if let Ok(manifest) = ChunkedContent::deserialize(&data) {
                    self.stats.manifests_traversed += 1;
                    for chunk_hash in &manifest.chunks {
                        if !self.marked.contains(chunk_hash) {
                            self.mark_queue.push_back(*chunk_hash);
                        }
                    }
                }
            }
        }

        self.stats.mark_elapsed += start.elapsed();
        self.stats.marked_count = self.marked.len();

        Ok(newly_marked)
    }

    /// Run the sweep phase (delete unmarked objects).
    ///
    /// Returns the number of objects deleted in this call.
    /// Call repeatedly until it returns 0 or time budget is exceeded.
    pub fn sweep_phase(&mut self) -> io::Result<usize> {
        if self.phase != GCPhase::Sweeping {
            return Ok(0);
        }

        // Initialize all_hashes on first sweep call
        if self.all_hashes.is_none() {
            self.all_hashes = Some(self.backend.list()?);
        }

        let start = Instant::now();
        let budget = self.config.time_budget();
        let batch_size = self.config.sweep_batch_size;
        let mut deleted_this_batch = 0;

        let all_hashes = self.all_hashes.as_ref().unwrap();

        for _ in 0..batch_size {
            if start.elapsed() > budget {
                break;
            }

            if self.sweep_index >= all_hashes.len() {
                // Sweep complete
                self.phase = GCPhase::Complete;
                self.stats.completed = true;
                break;
            }

            let hash = &all_hashes[self.sweep_index];
            self.sweep_index += 1;

            if !self.marked.contains(hash) {
                self.stats.orphan_count += 1;

                if self.config.delete_orphans {
                    // Get size before deletion for stats
                    if let Ok(Some(size)) = self.backend.size(hash) {
                        self.stats.bytes_deleted += size;
                    }

                    if self.backend.delete(hash)? {
                        deleted_this_batch += 1;
                        self.stats.deleted_count += 1;
                    }
                }
            }
        }

        self.stats.sweep_elapsed += start.elapsed();

        Ok(deleted_this_batch)
    }

    /// Run a complete GC cycle (mark + sweep).
    ///
    /// This runs both phases to completion, respecting the time budget
    /// by yielding periodically. For frame-budget-constrained scenarios,
    /// use `step()` instead.
    pub fn run_gc(&mut self) -> io::Result<GCStats> {
        let start = Instant::now();

        // Reset state
        self.phase = GCPhase::Idle;
        self.marked.clear();
        self.mark_queue = self.roots.iter().cloned().collect();
        self.all_hashes = None;
        self.sweep_index = 0;
        self.stats = GCStats::default();

        // Run mark phase to completion
        while self.phase == GCPhase::Idle || self.phase == GCPhase::Marking {
            self.mark_phase()?;
        }

        // Run sweep phase to completion
        while self.phase == GCPhase::Sweeping {
            self.sweep_phase()?;
        }

        self.stats.elapsed = start.elapsed();
        Ok(self.stats.clone())
    }

    /// Run a single time-sliced step of GC.
    ///
    /// Returns `true` if GC is complete, `false` if more work remains.
    /// Use this for frame-budget-constrained scenarios.
    pub fn step(&mut self) -> io::Result<bool> {
        match self.phase {
            GCPhase::Idle | GCPhase::Marking => {
                self.mark_phase()?;
            }
            GCPhase::Sweeping => {
                self.sweep_phase()?;
            }
            GCPhase::Complete => {
                return Ok(true);
            }
        }

        Ok(self.phase == GCPhase::Complete)
    }

    /// Reset the GC for a new run with updated roots.
    pub fn reset(&mut self, roots: RootSet) {
        self.roots = roots;
        self.marked.clear();
        self.mark_queue = self.roots.iter().cloned().collect();
        self.phase = GCPhase::Idle;
        self.stats = GCStats::default();
        self.all_hashes = None;
        self.sweep_index = 0;
    }
}

// ---------------------------------------------------------------------------
// RefCountedStore — wrapper with reference counting
// ---------------------------------------------------------------------------

/// Reference count entry for a content hash.
#[derive(Debug, Clone, Default)]
struct RefCountEntry {
    /// Current reference count.
    count: usize,
    /// Whether this entry was created with an initial reference.
    tracked: bool,
}

/// A wrapper around FileBackend that provides reference counting for content.
///
/// When `enable_immediate_cleanup` is true, objects with ref count 0 are
/// immediately deleted. Otherwise, they're marked as eligible for GC.
///
/// # Thread Safety
///
/// `RefCountedStore` is thread-safe and can be shared across threads using `Arc`.
///
/// # Example
///
/// ```ignore
/// let store = RefCountedStore::new(FileBackend::new("/tmp/store")?);
///
/// // Put content and get initial reference
/// let hash = store.put_and_ref(b"hello world")?;
///
/// // Create additional references
/// store.increment(&hash);
/// store.increment(&hash);
///
/// // Release references
/// store.decrement(&hash)?; // count: 2
/// store.decrement(&hash)?; // count: 1
/// store.decrement(&hash)?; // count: 0, deleted if enable_immediate_cleanup
/// ```
pub struct RefCountedStore {
    /// Underlying file backend.
    backend: FileBackend,
    /// Reference counts for tracked hashes.
    ref_counts: RwLock<HashMap<ContentHash, RefCountEntry>>,
    /// Whether to immediately delete when ref count reaches 0.
    enable_immediate_cleanup: AtomicBool,
    /// Stats: total increments.
    total_increments: AtomicU64,
    /// Stats: total decrements.
    total_decrements: AtomicU64,
    /// Stats: immediate cleanups performed.
    immediate_cleanups: AtomicU64,
}

impl RefCountedStore {
    /// Create a new reference-counted store wrapping an existing FileBackend.
    pub fn new(backend: FileBackend) -> Self {
        Self {
            backend,
            ref_counts: RwLock::new(HashMap::new()),
            enable_immediate_cleanup: AtomicBool::new(true),
            total_increments: AtomicU64::new(0),
            total_decrements: AtomicU64::new(0),
            immediate_cleanups: AtomicU64::new(0),
        }
    }

    /// Create a store with immediate cleanup disabled.
    pub fn without_immediate_cleanup(backend: FileBackend) -> Self {
        let store = Self::new(backend);
        store.enable_immediate_cleanup.store(false, Ordering::SeqCst);
        store
    }

    /// Set whether immediate cleanup is enabled.
    pub fn set_immediate_cleanup(&self, enabled: bool) {
        self.enable_immediate_cleanup.store(enabled, Ordering::SeqCst);
    }

    /// Check if immediate cleanup is enabled.
    pub fn immediate_cleanup_enabled(&self) -> bool {
        self.enable_immediate_cleanup.load(Ordering::SeqCst)
    }

    /// Get the underlying FileBackend (read-only access).
    pub fn backend(&self) -> &FileBackend {
        &self.backend
    }

    /// Store data and return its hash (no reference tracking).
    pub fn put(&self, data: &[u8]) -> io::Result<ContentHash> {
        self.backend.put(data)
    }

    /// Store data and create an initial reference.
    pub fn put_and_ref(&self, data: &[u8]) -> io::Result<ContentHash> {
        let hash = self.backend.put(data)?;
        self.increment(&hash);
        Ok(hash)
    }

    /// Retrieve data by hash.
    pub fn get(&self, hash: &ContentHash) -> io::Result<Option<Vec<u8>>> {
        self.backend.get(hash)
    }

    /// Check if content exists.
    pub fn has(&self, hash: &ContentHash) -> bool {
        self.backend.has(hash)
    }

    /// Increment the reference count for a hash.
    pub fn increment(&self, hash: &ContentHash) {
        let mut counts = self.ref_counts.write().unwrap();
        let entry = counts.entry(*hash).or_default();
        entry.count += 1;
        entry.tracked = true;
        self.total_increments.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement the reference count for a hash.
    ///
    /// If `enable_immediate_cleanup` is true and count reaches 0,
    /// the content is immediately deleted.
    ///
    /// Returns `true` if content was deleted.
    pub fn decrement(&self, hash: &ContentHash) -> io::Result<bool> {
        let should_delete = {
            let mut counts = self.ref_counts.write().unwrap();
            if let Some(entry) = counts.get_mut(hash) {
                if entry.count > 0 {
                    entry.count -= 1;
                    self.total_decrements.fetch_add(1, Ordering::Relaxed);
                }

                entry.count == 0 && self.enable_immediate_cleanup.load(Ordering::SeqCst)
            } else {
                false
            }
        };

        if should_delete {
            let deleted = self.backend.delete(hash)?;
            if deleted {
                self.immediate_cleanups.fetch_add(1, Ordering::Relaxed);
                let mut counts = self.ref_counts.write().unwrap();
                counts.remove(hash);
            }
            Ok(deleted)
        } else {
            Ok(false)
        }
    }

    /// Get the current reference count for a hash.
    pub fn ref_count(&self, hash: &ContentHash) -> usize {
        let counts = self.ref_counts.read().unwrap();
        counts.get(hash).map(|e| e.count).unwrap_or(0)
    }

    /// Check if a hash is being tracked (has been incremented at least once).
    pub fn is_tracked(&self, hash: &ContentHash) -> bool {
        let counts = self.ref_counts.read().unwrap();
        counts.get(hash).map(|e| e.tracked).unwrap_or(false)
    }

    /// Get all hashes with ref count 0 (eligible for cleanup).
    pub fn orphans(&self) -> Vec<ContentHash> {
        let counts = self.ref_counts.read().unwrap();
        counts
            .iter()
            .filter(|(_, entry)| entry.count == 0)
            .map(|(hash, _)| *hash)
            .collect()
    }

    /// Delete all tracked hashes with ref count 0.
    ///
    /// Returns the number of hashes deleted.
    pub fn cleanup_orphans(&self) -> io::Result<usize> {
        let orphans = self.orphans();
        let mut deleted = 0;

        for hash in orphans {
            if self.backend.delete(&hash)? {
                deleted += 1;
                self.ref_counts.write().unwrap().remove(&hash);
            }
        }

        Ok(deleted)
    }

    /// Get a RootSet of all tracked hashes with ref count > 0.
    pub fn live_roots(&self) -> RootSet {
        let counts = self.ref_counts.read().unwrap();
        let mut roots = RootSet::with_capacity(counts.len());

        for (hash, entry) in counts.iter() {
            if entry.count > 0 {
                roots.add(*hash);
            }
        }

        roots
    }

    /// Get statistics about reference counting.
    pub fn ref_count_stats(&self) -> RefCountStats {
        let counts = self.ref_counts.read().unwrap();
        let total_tracked = counts.len();
        let live = counts.values().filter(|e| e.count > 0).count();
        let orphaned = counts.values().filter(|e| e.count == 0).count();
        let max_refs = counts.values().map(|e| e.count).max().unwrap_or(0);
        let avg_refs = if total_tracked > 0 {
            counts.values().map(|e| e.count).sum::<usize>() as f64 / total_tracked as f64
        } else {
            0.0
        };

        RefCountStats {
            total_tracked,
            live,
            orphaned,
            max_refs,
            avg_refs,
            total_increments: self.total_increments.load(Ordering::Relaxed),
            total_decrements: self.total_decrements.load(Ordering::Relaxed),
            immediate_cleanups: self.immediate_cleanups.load(Ordering::Relaxed),
        }
    }
}

/// Statistics about reference counting in a RefCountedStore.
#[derive(Debug, Clone)]
pub struct RefCountStats {
    /// Total number of tracked hashes.
    pub total_tracked: usize,
    /// Number of hashes with ref count > 0.
    pub live: usize,
    /// Number of hashes with ref count == 0.
    pub orphaned: usize,
    /// Maximum reference count.
    pub max_refs: usize,
    /// Average reference count.
    pub avg_refs: f64,
    /// Total increment operations performed.
    pub total_increments: u64,
    /// Total decrement operations performed.
    pub total_decrements: u64,
    /// Total immediate cleanups performed.
    pub immediate_cleanups: u64,
}

// ---------------------------------------------------------------------------
// BackgroundGC — background thread for time-sliced GC
// ---------------------------------------------------------------------------

/// Handle for controlling background GC.
pub struct BackgroundGCHandle {
    /// Signal to stop the background thread.
    stop_signal: Arc<AtomicBool>,
    /// Signal to trigger immediate GC.
    trigger_gc: Arc<AtomicBool>,
    /// Last GC stats (shared with background thread).
    last_stats: Arc<Mutex<GCStats>>,
    /// Background thread handle.
    thread_handle: Option<JoinHandle<()>>,
    /// Whether GC is currently running.
    is_running: Arc<AtomicBool>,
    /// Total GC runs completed.
    total_runs: Arc<AtomicUsize>,
}

impl BackgroundGCHandle {
    /// Stop the background GC thread.
    pub fn stop(&mut self) {
        self.stop_signal.store(true, Ordering::SeqCst);
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
    }

    /// Trigger an immediate GC run.
    pub fn trigger_gc(&self) {
        self.trigger_gc.store(true, Ordering::SeqCst);
    }

    /// Check if GC is currently running.
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::SeqCst)
    }

    /// Get the number of GC runs completed.
    pub fn total_runs(&self) -> usize {
        self.total_runs.load(Ordering::SeqCst)
    }

    /// Get the last GC statistics.
    pub fn last_stats(&self) -> GCStats {
        self.last_stats.lock().unwrap().clone()
    }
}

impl Drop for BackgroundGCHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Configuration for background GC.
#[derive(Debug, Clone)]
pub struct BackgroundGCConfig {
    /// GC configuration.
    pub gc_config: GCConfig,
    /// Interval between GC runs.
    pub interval: Duration,
    /// Maximum time per GC step (for time-slicing).
    pub step_budget: Duration,
}

impl Default for BackgroundGCConfig {
    fn default() -> Self {
        Self {
            gc_config: GCConfig::default(),
            interval: Duration::from_secs(60),
            step_budget: Duration::from_millis(2),
        }
    }
}

/// Start a background GC thread.
///
/// The GC runs periodically, using time-sliced collection to avoid
/// blocking the main thread for extended periods.
///
/// # Arguments
///
/// * `backend` - The FileBackend to manage (must be Arc-wrapped for sharing)
/// * `roots_provider` - Function that returns the current root set
/// * `config` - Background GC configuration
///
/// # Example
///
/// ```ignore
/// let backend = Arc::new(FileBackend::new("/tmp/store")?);
/// let handle = start_background_gc(
///     Arc::clone(&backend),
///     || get_current_roots(),
///     BackgroundGCConfig::default(),
/// );
///
/// // Later, stop the background GC
/// handle.stop();
/// ```
pub fn start_background_gc<F>(
    backend: Arc<FileBackend>,
    roots_provider: F,
    config: BackgroundGCConfig,
) -> BackgroundGCHandle
where
    F: Fn() -> RootSet + Send + 'static,
{
    let stop_signal = Arc::new(AtomicBool::new(false));
    let trigger_gc = Arc::new(AtomicBool::new(false));
    let last_stats = Arc::new(Mutex::new(GCStats::default()));
    let is_running = Arc::new(AtomicBool::new(false));
    let total_runs = Arc::new(AtomicUsize::new(0));

    let stop_clone = Arc::clone(&stop_signal);
    let trigger_clone = Arc::clone(&trigger_gc);
    let stats_clone = Arc::clone(&last_stats);
    let running_clone = Arc::clone(&is_running);
    let runs_clone = Arc::clone(&total_runs);

    let thread_handle = thread::spawn(move || {
        let mut last_gc = Instant::now();

        while !stop_clone.load(Ordering::SeqCst) {
            let should_gc = trigger_clone.swap(false, Ordering::SeqCst)
                || last_gc.elapsed() >= config.interval;

            if should_gc {
                running_clone.store(true, Ordering::SeqCst);

                let roots = roots_provider();
                let mut gc = TreeGarbageCollector::new(&backend, roots, config.gc_config.clone());

                // Run time-sliced GC
                loop {
                    match gc.step() {
                        Ok(true) => break,
                        Ok(false) => {
                            // Yield to other threads
                            thread::sleep(Duration::from_micros(100));
                        }
                        Err(_) => break,
                    }

                    // Check for stop signal during long GC
                    if stop_clone.load(Ordering::SeqCst) {
                        break;
                    }
                }

                *stats_clone.lock().unwrap() = gc.stats().clone();
                runs_clone.fetch_add(1, Ordering::SeqCst);
                running_clone.store(false, Ordering::SeqCst);
                last_gc = Instant::now();
            }

            // Sleep before next check
            thread::sleep(Duration::from_millis(100));
        }
    });

    BackgroundGCHandle {
        stop_signal,
        trigger_gc,
        last_stats,
        thread_handle: Some(thread_handle),
        is_running,
        total_runs,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: create temp store
    fn create_temp_store() -> (tempfile::TempDir, FileBackend) {
        let dir = tempfile::tempdir().expect("create temp dir");
        let store = FileBackend::new(dir.path()).expect("create store");
        (dir, store)
    }

    // ── RootSet ────────────────────────────────────────────────────────────

    #[test]
    fn test_root_set_new() {
        let roots = RootSet::new();
        assert!(roots.is_empty());
        assert_eq!(roots.len(), 0);
    }

    #[test]
    fn test_root_set_add_remove() {
        let mut roots = RootSet::new();
        let h1 = ContentHash::from_bytes(b"hash1");
        let h2 = ContentHash::from_bytes(b"hash2");

        roots.add(h1);
        assert!(roots.contains(&h1));
        assert!(!roots.contains(&h2));
        assert_eq!(roots.len(), 1);

        roots.add(h2);
        assert_eq!(roots.len(), 2);

        assert!(roots.remove(&h1));
        assert!(!roots.contains(&h1));
        assert_eq!(roots.len(), 1);

        assert!(!roots.remove(&h1)); // Already removed
    }

    #[test]
    fn test_root_set_labeled() {
        let mut roots = RootSet::new();
        let h1 = ContentHash::from_bytes(b"material");

        roots.add_labeled(h1, "active_material");
        assert!(roots.contains(&h1));
        assert_eq!(roots.label(&h1), Some("active_material"));

        roots.remove(&h1);
        assert!(roots.label(&h1).is_none());
    }

    #[test]
    fn test_root_set_merge() {
        let mut roots1 = RootSet::new();
        let mut roots2 = RootSet::new();
        let h1 = ContentHash::from_bytes(b"a");
        let h2 = ContentHash::from_bytes(b"b");

        roots1.add(h1);
        roots2.add(h2);

        roots1.merge(&roots2);
        assert_eq!(roots1.len(), 2);
        assert!(roots1.contains(&h1));
        assert!(roots1.contains(&h2));
    }

    #[test]
    fn test_root_set_from_hashes() {
        let h1 = ContentHash::from_bytes(b"a");
        let h2 = ContentHash::from_bytes(b"b");

        let roots = RootSet::from_hashes(vec![h1, h2]);
        assert_eq!(roots.len(), 2);
        assert!(roots.contains(&h1));
        assert!(roots.contains(&h2));
    }

    // ── GCConfig ───────────────────────────────────────────────────────────

    #[test]
    fn test_gc_config_default() {
        let config = GCConfig::default();
        assert_eq!(config.max_time_budget_ms, 2);
        assert!(config.enable_ref_counting);
        assert!(config.delete_orphans);
    }

    #[test]
    fn test_gc_config_dry_run() {
        let config = GCConfig::dry_run();
        assert!(!config.delete_orphans);
    }

    #[test]
    fn test_gc_config_aggressive() {
        let config = GCConfig::aggressive();
        assert_eq!(config.max_time_budget_ms, 10);
        assert_eq!(config.mark_batch_size, 500);
    }

    #[test]
    fn test_gc_config_time_budget() {
        let config = GCConfig::default().with_time_budget_ms(5);
        assert_eq!(config.time_budget(), Duration::from_millis(5));
    }

    // ── TreeGarbageCollector ───────────────────────────────────────────────

    #[test]
    fn test_gc_marks_roots() {
        let (_dir, store) = create_temp_store();

        let h1 = store.put(b"root1").expect("put");
        let h2 = store.put(b"root2").expect("put");
        let _h3 = store.put(b"orphan").expect("put");

        let mut roots = RootSet::new();
        roots.add(h1);
        roots.add(h2);

        let mut gc = TreeGarbageCollector::new(&store, roots, GCConfig::default());
        let stats = gc.run_gc().expect("gc");

        assert_eq!(stats.marked_count, 2);
        assert!(gc.marked().contains(&h1));
        assert!(gc.marked().contains(&h2));
    }

    #[test]
    fn test_gc_marks_tree_children() {
        let (_dir, store) = create_temp_store();

        let blob1 = store.put(b"blob1").expect("put");
        let blob2 = store.put(b"blob2").expect("put");

        let tree = ContentTree::from_entries(vec![
            crate::pipeline::TreeEntry::blob("a.txt", blob1),
            crate::pipeline::TreeEntry::blob("b.txt", blob2),
        ]);
        let tree_hash = tree.store(&store).expect("store tree");

        let mut roots = RootSet::new();
        roots.add(tree_hash);

        let mut gc = TreeGarbageCollector::new(&store, roots, GCConfig::default());
        let stats = gc.run_gc().expect("gc");

        assert_eq!(stats.marked_count, 3); // tree + 2 blobs
        assert!(gc.marked().contains(&tree_hash));
        assert!(gc.marked().contains(&blob1));
        assert!(gc.marked().contains(&blob2));
    }

    #[test]
    fn test_gc_deletes_orphans() {
        let (_dir, store) = create_temp_store();

        let root = store.put(b"keep me").expect("put");
        let orphan = store.put(b"delete me").expect("put");

        assert!(store.has(&root));
        assert!(store.has(&orphan));

        let mut roots = RootSet::new();
        roots.add(root);

        let mut gc = TreeGarbageCollector::new(&store, roots, GCConfig::default());
        let stats = gc.run_gc().expect("gc");

        assert_eq!(stats.marked_count, 1);
        assert_eq!(stats.orphan_count, 1);
        assert_eq!(stats.deleted_count, 1);
        assert!(stats.completed);

        assert!(store.has(&root));
        assert!(!store.has(&orphan));
    }

    #[test]
    fn test_gc_dry_run() {
        let (_dir, store) = create_temp_store();

        let root = store.put(b"keep me").expect("put");
        let orphan = store.put(b"would be deleted").expect("put");

        let mut roots = RootSet::new();
        roots.add(root);

        let config = GCConfig::dry_run();
        let mut gc = TreeGarbageCollector::new(&store, roots, config);
        let stats = gc.run_gc().expect("gc");

        assert_eq!(stats.orphan_count, 1);
        assert_eq!(stats.deleted_count, 0);

        // Orphan should still exist in dry run
        assert!(store.has(&orphan));
    }

    #[test]
    fn test_gc_step_incremental() {
        let (_dir, store) = create_temp_store();

        // Create several blobs
        let mut hashes = Vec::new();
        for i in 0..50 {
            let hash = store.put(format!("blob{}", i).as_bytes()).expect("put");
            hashes.push(hash);
        }

        // Only keep first 10
        let mut roots = RootSet::new();
        for hash in &hashes[..10] {
            roots.add(*hash);
        }

        let mut gc = TreeGarbageCollector::new(&store, roots, GCConfig::default());

        // Run step by step
        let mut steps = 0;
        while !gc.step().expect("step") {
            steps += 1;
            assert!(steps < 1000, "GC should complete in reasonable steps");
        }

        let stats = gc.stats();
        assert_eq!(stats.marked_count, 10);
        assert_eq!(stats.orphan_count, 40);
        assert_eq!(stats.deleted_count, 40);
    }

    #[test]
    fn test_gc_empty_store() {
        let (_dir, store) = create_temp_store();
        let roots = RootSet::new();

        let mut gc = TreeGarbageCollector::new(&store, roots, GCConfig::default());
        let stats = gc.run_gc().expect("gc");

        assert_eq!(stats.marked_count, 0);
        assert_eq!(stats.orphan_count, 0);
        assert_eq!(stats.deleted_count, 0);
        assert!(stats.completed);
    }

    #[test]
    fn test_gc_reset() {
        let (_dir, store) = create_temp_store();

        let h1 = store.put(b"hash1").expect("put");
        let h2 = store.put(b"hash2").expect("put");

        let mut roots = RootSet::new();
        roots.add(h1);

        let mut gc = TreeGarbageCollector::new(&store, roots, GCConfig::default());
        gc.run_gc().expect("gc");

        // Reset with new roots
        let mut new_roots = RootSet::new();
        new_roots.add(h2);
        gc.reset(new_roots);

        assert_eq!(gc.phase(), GCPhase::Idle);
        assert!(gc.marked().is_empty());
    }

    #[test]
    fn test_gc_time_budget_respected() {
        let (_dir, store) = create_temp_store();

        // Create many blobs
        for i in 0..200 {
            store.put(format!("blob{}", i).as_bytes()).expect("put");
        }

        let roots = RootSet::new(); // No roots - all orphans

        // Very short time budget
        let config = GCConfig::default().with_time_budget_ms(1);
        let mut gc = TreeGarbageCollector::new(&store, roots, config);

        // Single step should respect time budget
        let start = Instant::now();
        gc.step().expect("step");
        let elapsed = start.elapsed();

        // Should complete in a reasonable time (allowing some overhead)
        assert!(elapsed < Duration::from_millis(50));
    }

    // ── RefCountedStore ────────────────────────────────────────────────────

    #[test]
    fn test_ref_counted_store_basic() {
        let (_dir, backend) = create_temp_store();
        let store = RefCountedStore::new(backend);

        let hash = store.put_and_ref(b"hello").expect("put");
        assert_eq!(store.ref_count(&hash), 1);
        assert!(store.is_tracked(&hash));
        assert!(store.has(&hash));
    }

    #[test]
    fn test_ref_counted_store_increment_decrement() {
        let (_dir, backend) = create_temp_store();
        let store = RefCountedStore::new(backend);

        let hash = store.put_and_ref(b"data").expect("put");
        assert_eq!(store.ref_count(&hash), 1);

        store.increment(&hash);
        assert_eq!(store.ref_count(&hash), 2);

        store.increment(&hash);
        assert_eq!(store.ref_count(&hash), 3);

        store.decrement(&hash).expect("decrement");
        assert_eq!(store.ref_count(&hash), 2);

        store.decrement(&hash).expect("decrement");
        assert_eq!(store.ref_count(&hash), 1);
    }

    #[test]
    fn test_ref_counted_store_immediate_cleanup() {
        let (_dir, backend) = create_temp_store();
        let store = RefCountedStore::new(backend);

        let hash = store.put_and_ref(b"to delete").expect("put");
        assert!(store.has(&hash));

        let deleted = store.decrement(&hash).expect("decrement");
        assert!(deleted);
        assert!(!store.has(&hash));
    }

    #[test]
    fn test_ref_counted_store_no_immediate_cleanup() {
        let (_dir, backend) = create_temp_store();
        let store = RefCountedStore::without_immediate_cleanup(backend);

        let hash = store.put_and_ref(b"keep around").expect("put");
        assert!(store.has(&hash));

        let deleted = store.decrement(&hash).expect("decrement");
        assert!(!deleted);
        assert!(store.has(&hash)); // Still exists

        // Should be in orphans list
        let orphans = store.orphans();
        assert!(orphans.contains(&hash));

        // Manual cleanup
        let cleaned = store.cleanup_orphans().expect("cleanup");
        assert_eq!(cleaned, 1);
        assert!(!store.has(&hash));
    }

    #[test]
    fn test_ref_counted_store_live_roots() {
        let (_dir, backend) = create_temp_store();
        let store = RefCountedStore::without_immediate_cleanup(backend);

        let h1 = store.put_and_ref(b"live1").expect("put");
        let h2 = store.put_and_ref(b"live2").expect("put");
        let h3 = store.put_and_ref(b"dead").expect("put");

        store.decrement(&h3).expect("decrement");

        let roots = store.live_roots();
        assert_eq!(roots.len(), 2);
        assert!(roots.contains(&h1));
        assert!(roots.contains(&h2));
        assert!(!roots.contains(&h3));
    }

    #[test]
    fn test_ref_counted_store_stats() {
        let (_dir, backend) = create_temp_store();
        let store = RefCountedStore::new(backend);

        let h1 = store.put_and_ref(b"a").expect("put");
        let h2 = store.put_and_ref(b"b").expect("put");

        store.increment(&h1);
        store.increment(&h1);
        store.decrement(&h2).expect("decrement"); // Triggers cleanup

        let stats = store.ref_count_stats();
        assert_eq!(stats.total_tracked, 1); // h2 was cleaned up
        assert_eq!(stats.live, 1);
        assert_eq!(stats.orphaned, 0);
        assert_eq!(stats.max_refs, 3);
        assert_eq!(stats.total_increments, 4); // 2 put_and_ref + 2 increment
        assert_eq!(stats.total_decrements, 1);
        assert_eq!(stats.immediate_cleanups, 1);
    }

    // ── GCStats ────────────────────────────────────────────────────────────

    #[test]
    fn test_gc_stats_merge() {
        let mut stats1 = GCStats {
            marked_count: 10,
            orphan_count: 5,
            deleted_count: 3,
            bytes_deleted: 1000,
            completed: true,
            elapsed: Duration::from_millis(10),
            mark_elapsed: Duration::from_millis(5),
            sweep_elapsed: Duration::from_millis(5),
            trees_traversed: 2,
            manifests_traversed: 1,
        };

        let stats2 = GCStats {
            marked_count: 5,
            orphan_count: 2,
            deleted_count: 2,
            bytes_deleted: 500,
            completed: true,
            elapsed: Duration::from_millis(5),
            mark_elapsed: Duration::from_millis(3),
            sweep_elapsed: Duration::from_millis(2),
            trees_traversed: 1,
            manifests_traversed: 0,
        };

        stats1.merge(&stats2);

        assert_eq!(stats1.marked_count, 15);
        assert_eq!(stats1.orphan_count, 7);
        assert_eq!(stats1.deleted_count, 5);
        assert_eq!(stats1.bytes_deleted, 1500);
        assert!(stats1.completed);
        assert_eq!(stats1.elapsed, Duration::from_millis(15));
        assert_eq!(stats1.trees_traversed, 3);
        assert_eq!(stats1.manifests_traversed, 1);
    }

    // ── GCPhase ────────────────────────────────────────────────────────────

    #[test]
    fn test_gc_phase_transitions() {
        let (_dir, store) = create_temp_store();
        store.put(b"some data").expect("put");

        let roots = RootSet::new();
        let mut gc = TreeGarbageCollector::new(&store, roots, GCConfig::default());

        assert_eq!(gc.phase(), GCPhase::Idle);

        gc.mark_phase().expect("mark");
        // After processing all roots (empty), moves to sweeping
        assert!(gc.phase() == GCPhase::Marking || gc.phase() == GCPhase::Sweeping);

        // Complete GC
        gc.run_gc().expect("gc");
        assert_eq!(gc.phase(), GCPhase::Complete);
    }

    // ── Integration: GC + RefCountedStore ──────────────────────────────────

    #[test]
    fn test_gc_with_ref_counted_roots() {
        let (_dir, backend) = create_temp_store();
        let store = RefCountedStore::without_immediate_cleanup(backend);

        // Create some content with refs
        let h1 = store.put_and_ref(b"live1").expect("put");
        let h2 = store.put_and_ref(b"live2").expect("put");
        let h3 = store.put(b"orphan").expect("put"); // No ref

        // Get live roots from ref counting
        let roots = store.live_roots();

        // Run GC
        let mut gc = TreeGarbageCollector::new(store.backend(), roots, GCConfig::default());
        let stats = gc.run_gc().expect("gc");

        assert_eq!(stats.marked_count, 2);
        assert_eq!(stats.deleted_count, 1);

        assert!(store.has(&h1));
        assert!(store.has(&h2));
        assert!(!store.has(&h3));
    }

    #[test]
    fn test_gc_marks_chunked_content() {
        let (_dir, store) = create_temp_store();

        // Create chunked content
        let chunk_size = 100;
        let data: Vec<u8> = (0..350).map(|i| (i % 256) as u8).collect();
        let manifest_hash = store
            .put_stream_with_chunk_size(&mut &data[..], chunk_size)
            .expect("put_stream");

        // Create orphan
        let orphan = store.put(b"orphan").expect("put");

        let mut roots = RootSet::new();
        roots.add(manifest_hash);

        let mut gc = TreeGarbageCollector::new(&store, roots, GCConfig::default());
        let stats = gc.run_gc().expect("gc");

        // Should mark manifest + chunks
        assert!(stats.marked_count >= 2);
        assert!(stats.manifests_traversed >= 1);
        assert_eq!(stats.orphan_count, 1);
        assert_eq!(stats.deleted_count, 1);

        // Verify chunked content still works
        assert!(store.has(&manifest_hash));
        assert!(!store.has(&orphan));

        let mut reader = store.get_stream(&manifest_hash).expect("get").expect("exists");
        let mut retrieved = Vec::new();
        std::io::Read::read_to_end(&mut reader, &mut retrieved).expect("read");
        assert_eq!(retrieved, data);
    }

    #[test]
    fn test_gc_bytes_deleted_tracking() {
        let (_dir, store) = create_temp_store();

        // Create blobs of known sizes
        let data1 = vec![0u8; 100];
        let data2 = vec![0u8; 200];
        let data3 = vec![0u8; 50];

        let h1 = store.put(&data1).expect("put");
        let _h2 = store.put(&data2).expect("put"); // orphan
        let _h3 = store.put(&data3).expect("put"); // orphan

        let mut roots = RootSet::new();
        roots.add(h1);

        let mut gc = TreeGarbageCollector::new(&store, roots, GCConfig::default());
        let stats = gc.run_gc().expect("gc");

        assert_eq!(stats.deleted_count, 2);
        assert_eq!(stats.bytes_deleted, 250); // 200 + 50
    }
}
