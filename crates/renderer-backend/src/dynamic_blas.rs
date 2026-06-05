//! Dynamic BLAS Refit for Animated Meshes
//!
//! This module provides BLAS management for skinned and animated meshes that
//! require per-frame or on-demand refitting when vertex positions change.
//!
//! # Architecture
//!
//! - `DynamicBlasConfig`: Configuration for refit behavior and batching
//! - `DynamicBlasEntry`: Per-mesh tracking state including dirty flags
//! - `DynamicBlasManager`: Central manager for queuing and processing refits
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::dynamic_blas::{DynamicBlasConfig, DynamicBlasManager};
//!
//! let config = DynamicBlasConfig::default();
//! let mut manager = DynamicBlasManager::new(config);
//!
//! // Register a skinned mesh
//! manager.register_dynamic_mesh("character_0", 10000, blas_handle);
//!
//! // After GPU skinning pass completes
//! manager.on_skinning_complete(&["character_0", "character_1"]);
//!
//! // Each frame: begin, process refits, then trace
//! manager.begin_frame(current_frame);
//! let to_refit = manager.process_refits(Some(4)); // Max 4 refits this frame
//! for handle in to_refit {
//!     // Issue GPU refit commands for handle
//! }
//! ```

use crate::blas_pool::BlasHandle;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default maximum refits per frame.
const DEFAULT_MAX_REFITS_PER_FRAME: u32 = 8;

/// Default batch threshold in bytes before triggering batch refit.
const DEFAULT_BATCH_THRESHOLD: usize = 4 * 1024 * 1024; // 4 MB

// ---------------------------------------------------------------------------
// DynamicBlasConfig
// ---------------------------------------------------------------------------

/// Configuration for dynamic BLAS refit behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DynamicBlasConfig {
    /// Maximum number of BLAS refits to perform per frame.
    /// Spreading refits across frames reduces frame time spikes.
    pub max_refit_per_frame: u32,

    /// Byte threshold before triggering a batch refit operation.
    /// Smaller meshes can be batched together for efficiency.
    pub batch_threshold: usize,

    /// Prefer refit over rebuild for small geometry changes.
    /// Refit is faster but may result in lower BVH quality.
    pub prefer_refit: bool,
}

impl DynamicBlasConfig {
    /// Create a new config with default values.
    pub const fn new() -> Self {
        Self {
            max_refit_per_frame: DEFAULT_MAX_REFITS_PER_FRAME,
            batch_threshold: DEFAULT_BATCH_THRESHOLD,
            prefer_refit: true,
        }
    }

    /// Create a config optimized for many animated characters.
    /// Higher refit limit, larger batch threshold.
    pub const fn for_crowd() -> Self {
        Self {
            max_refit_per_frame: 16,
            batch_threshold: 8 * 1024 * 1024, // 8 MB
            prefer_refit: true,
        }
    }

    /// Create a config optimized for few high-detail characters.
    /// Lower refit limit, smaller batch threshold, prefer quality.
    pub const fn for_hero() -> Self {
        Self {
            max_refit_per_frame: 4,
            batch_threshold: 2 * 1024 * 1024, // 2 MB
            prefer_refit: false,
        }
    }

    /// Builder: set max refits per frame.
    pub const fn with_max_refit_per_frame(mut self, max: u32) -> Self {
        self.max_refit_per_frame = max;
        self
    }

    /// Builder: set batch threshold.
    pub const fn with_batch_threshold(mut self, threshold: usize) -> Self {
        self.batch_threshold = threshold;
        self
    }

    /// Builder: set prefer refit flag.
    pub const fn with_prefer_refit(mut self, prefer: bool) -> Self {
        self.prefer_refit = prefer;
        self
    }
}

impl Default for DynamicBlasConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// DynamicBlasEntry
// ---------------------------------------------------------------------------

/// Tracking entry for a dynamic mesh's BLAS.
#[derive(Debug, Clone)]
pub struct DynamicBlasEntry {
    /// Handle to the BLAS in the pool.
    pub handle: BlasHandle,
    /// Mesh asset ID.
    pub mesh_id: String,
    /// Frame number when this BLAS was last refitted.
    pub last_refit_frame: u64,
    /// Number of vertices (used for prioritization).
    pub vertex_count: u32,
    /// Whether this BLAS needs refit before next trace.
    pub needs_refit: bool,
}

impl DynamicBlasEntry {
    /// Create a new entry with the given parameters.
    fn new(mesh_id: String, vertex_count: u32, handle: BlasHandle) -> Self {
        Self {
            handle,
            mesh_id,
            last_refit_frame: 0,
            vertex_count,
            needs_refit: false,
        }
    }

    /// Get estimated memory size for prioritization.
    /// Rough estimate: 64 bytes per vertex for BVH nodes.
    pub fn estimated_memory(&self) -> usize {
        self.vertex_count as usize * 64
    }
}

// ---------------------------------------------------------------------------
// DynamicBlasError
// ---------------------------------------------------------------------------

/// Errors that can occur during dynamic BLAS operations.
#[derive(Debug, Clone)]
pub enum DynamicBlasError {
    /// Mesh is not registered with the manager.
    MeshNotRegistered(String),
    /// Refit operation failed.
    RefitFailed(String),
    /// Invalid BLAS handle.
    InvalidHandle,
}

impl std::fmt::Display for DynamicBlasError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MeshNotRegistered(id) => write!(f, "mesh '{}' is not registered", id),
            Self::RefitFailed(msg) => write!(f, "BLAS refit failed: {}", msg),
            Self::InvalidHandle => write!(f, "invalid BLAS handle"),
        }
    }
}

impl std::error::Error for DynamicBlasError {}

// ---------------------------------------------------------------------------
// DynamicBlasManager
// ---------------------------------------------------------------------------

/// Manager for dynamic BLAS refit operations.
///
/// Handles:
/// - Registration of dynamic (skinned/animated) meshes
/// - Dirty flag tracking for meshes that need refit
/// - Per-frame refit budget management
/// - Priority-based refit queue processing
pub struct DynamicBlasManager {
    /// Configuration.
    config: DynamicBlasConfig,
    /// Per-mesh tracking entries.
    entries: HashMap<String, DynamicBlasEntry>,
    /// Queue of mesh IDs that need refit, in priority order.
    refit_queue: Vec<String>,
    /// Current frame number.
    current_frame: u64,
    /// Number of refits processed this frame.
    refits_this_frame: u32,
}

impl DynamicBlasManager {
    /// Create a new dynamic BLAS manager with the given configuration.
    pub fn new(config: DynamicBlasConfig) -> Self {
        Self {
            config,
            entries: HashMap::new(),
            refit_queue: Vec::new(),
            current_frame: 0,
            refits_this_frame: 0,
        }
    }

    /// Get the current configuration.
    pub fn config(&self) -> &DynamicBlasConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: DynamicBlasConfig) {
        self.config = config;
    }

    /// Register a dynamic mesh for refit tracking.
    ///
    /// # Arguments
    ///
    /// * `mesh_id` - Unique identifier for the mesh
    /// * `vertex_count` - Number of vertices (for prioritization)
    /// * `handle` - BLAS handle from the pool
    pub fn register_dynamic_mesh(&mut self, mesh_id: &str, vertex_count: u32, handle: BlasHandle) {
        let entry = DynamicBlasEntry::new(mesh_id.to_string(), vertex_count, handle);
        self.entries.insert(mesh_id.to_string(), entry);
    }

    /// Unregister a dynamic mesh.
    ///
    /// Removes the mesh from tracking and any pending refit queue.
    pub fn unregister(&mut self, mesh_id: &str) {
        self.entries.remove(mesh_id);
        self.refit_queue.retain(|id| id != mesh_id);
    }

    /// Mark a mesh's BLAS as needing refit.
    ///
    /// The mesh will be added to the refit queue if not already present.
    ///
    /// # Arguments
    ///
    /// * `mesh_id` - The mesh to mark dirty
    ///
    /// # Returns
    ///
    /// `Ok(())` if the mesh exists, `Err` if not registered.
    pub fn mark_dirty(&mut self, mesh_id: &str) -> Result<(), DynamicBlasError> {
        if let Some(entry) = self.entries.get_mut(mesh_id) {
            if !entry.needs_refit {
                entry.needs_refit = true;
                // Add to queue if not already present
                if !self.refit_queue.contains(&mesh_id.to_string()) {
                    self.refit_queue.push(mesh_id.to_string());
                }
            }
            Ok(())
        } else {
            Err(DynamicBlasError::MeshNotRegistered(mesh_id.to_string()))
        }
    }

    /// Check if a mesh is marked dirty (needs refit).
    pub fn is_dirty(&self, mesh_id: &str) -> bool {
        self.entries
            .get(mesh_id)
            .map(|e| e.needs_refit)
            .unwrap_or(false)
    }

    /// Called after GPU skinning pass completes to mark affected BLAS dirty.
    ///
    /// This is a convenience method for marking multiple meshes at once.
    ///
    /// # Arguments
    ///
    /// * `mesh_ids` - Slice of mesh IDs that were skinned this frame
    pub fn on_skinning_complete(&mut self, mesh_ids: &[&str]) {
        for id in mesh_ids {
            // Ignore errors for unregistered meshes (might be static)
            let _ = self.mark_dirty(id);
        }
    }

    /// Begin a new frame.
    ///
    /// Resets per-frame state and updates the frame counter.
    /// Does NOT clear dirty flags - those are cleared when refits are processed.
    ///
    /// # Arguments
    ///
    /// * `frame` - The current frame number
    pub fn begin_frame(&mut self, frame: u64) {
        self.current_frame = frame;
        self.refits_this_frame = 0;
    }

    /// Process pending refits up to the budget limit.
    ///
    /// Returns handles that should be refitted. The caller is responsible
    /// for issuing the actual GPU refit commands.
    ///
    /// # Arguments
    ///
    /// * `max_count` - Optional override for max refits (uses config if None)
    ///
    /// # Returns
    ///
    /// Vector of BLAS handles to refit this frame.
    pub fn process_refits(&mut self, max_count: Option<u32>) -> Vec<BlasHandle> {
        let limit = max_count.unwrap_or(self.config.max_refit_per_frame);
        let remaining = limit.saturating_sub(self.refits_this_frame);

        if remaining == 0 || self.refit_queue.is_empty() {
            return Vec::new();
        }

        // Sort queue by priority (smaller vertex count = higher priority for batching)
        // This helps batch small meshes together
        self.refit_queue.sort_by(|a, b| {
            let va = self.entries.get(a).map(|e| e.vertex_count).unwrap_or(0);
            let vb = self.entries.get(b).map(|e| e.vertex_count).unwrap_or(0);
            va.cmp(&vb)
        });

        let count = (remaining as usize).min(self.refit_queue.len());
        let to_process: Vec<String> = self.refit_queue.drain(..count).collect();

        let mut handles = Vec::with_capacity(count);

        for mesh_id in to_process {
            if let Some(entry) = self.entries.get_mut(&mesh_id) {
                entry.needs_refit = false;
                entry.last_refit_frame = self.current_frame;
                handles.push(entry.handle);
                self.refits_this_frame += 1;
            }
        }

        handles
    }

    /// Get an entry by mesh ID.
    pub fn get_entry(&self, mesh_id: &str) -> Option<&DynamicBlasEntry> {
        self.entries.get(mesh_id)
    }

    /// Get a mutable entry by mesh ID.
    pub fn get_entry_mut(&mut self, mesh_id: &str) -> Option<&mut DynamicBlasEntry> {
        self.entries.get_mut(mesh_id)
    }

    /// Get the number of entries pending refit.
    pub fn pending_refit_count(&self) -> usize {
        self.refit_queue.len()
    }

    /// Get the total number of registered dynamic meshes.
    pub fn mesh_count(&self) -> usize {
        self.entries.len()
    }

    /// Check if a mesh is registered.
    pub fn contains(&self, mesh_id: &str) -> bool {
        self.entries.contains_key(mesh_id)
    }

    /// Get the current frame number.
    pub fn current_frame(&self) -> u64 {
        self.current_frame
    }

    /// Get the number of refits processed this frame.
    pub fn refits_this_frame(&self) -> u32 {
        self.refits_this_frame
    }

    /// Clear all entries and reset state.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.refit_queue.clear();
        self.refits_this_frame = 0;
    }

    /// Iterate over all entries.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &DynamicBlasEntry)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Get statistics about the manager.
    pub fn stats(&self) -> DynamicBlasStats {
        let total_vertices: u64 = self.entries.values().map(|e| e.vertex_count as u64).sum();
        let dirty_count = self.entries.values().filter(|e| e.needs_refit).count();
        let estimated_memory: usize = self.entries.values().map(|e| e.estimated_memory()).sum();

        DynamicBlasStats {
            mesh_count: self.entries.len(),
            pending_refits: self.refit_queue.len(),
            dirty_count,
            total_vertices,
            estimated_memory,
            refits_this_frame: self.refits_this_frame,
            current_frame: self.current_frame,
        }
    }
}

impl Default for DynamicBlasManager {
    fn default() -> Self {
        Self::new(DynamicBlasConfig::default())
    }
}

// ---------------------------------------------------------------------------
// DynamicBlasStats
// ---------------------------------------------------------------------------

/// Statistics about the dynamic BLAS manager.
#[derive(Debug, Clone, Default)]
pub struct DynamicBlasStats {
    /// Number of registered dynamic meshes.
    pub mesh_count: usize,
    /// Number of meshes pending refit.
    pub pending_refits: usize,
    /// Number of meshes marked dirty.
    pub dirty_count: usize,
    /// Total vertex count across all meshes.
    pub total_vertices: u64,
    /// Estimated total memory for dynamic BLAS.
    pub estimated_memory: usize,
    /// Number of refits processed this frame.
    pub refits_this_frame: u32,
    /// Current frame number.
    pub current_frame: u64,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Helper functions
    // -------------------------------------------------------------------------

    /// Create a dummy BlasHandle for testing.
    fn make_handle(id: u32) -> BlasHandle {
        // BlasHandle is #[repr(transparent)] over u32
        unsafe { std::mem::transmute(id) }
    }

    // -------------------------------------------------------------------------
    // Test: Config defaults and builder
    // -------------------------------------------------------------------------

    #[test]
    fn test_config_default() {
        let config = DynamicBlasConfig::default();
        assert_eq!(config.max_refit_per_frame, DEFAULT_MAX_REFITS_PER_FRAME);
        assert_eq!(config.batch_threshold, DEFAULT_BATCH_THRESHOLD);
        assert!(config.prefer_refit);
    }

    #[test]
    fn test_config_new() {
        let config = DynamicBlasConfig::new();
        assert_eq!(config.max_refit_per_frame, DEFAULT_MAX_REFITS_PER_FRAME);
        assert_eq!(config.batch_threshold, DEFAULT_BATCH_THRESHOLD);
        assert!(config.prefer_refit);
    }

    #[test]
    fn test_config_for_crowd() {
        let config = DynamicBlasConfig::for_crowd();
        assert_eq!(config.max_refit_per_frame, 16);
        assert_eq!(config.batch_threshold, 8 * 1024 * 1024);
        assert!(config.prefer_refit);
    }

    #[test]
    fn test_config_for_hero() {
        let config = DynamicBlasConfig::for_hero();
        assert_eq!(config.max_refit_per_frame, 4);
        assert_eq!(config.batch_threshold, 2 * 1024 * 1024);
        assert!(!config.prefer_refit);
    }

    #[test]
    fn test_config_builder() {
        let config = DynamicBlasConfig::new()
            .with_max_refit_per_frame(10)
            .with_batch_threshold(1024)
            .with_prefer_refit(false);

        assert_eq!(config.max_refit_per_frame, 10);
        assert_eq!(config.batch_threshold, 1024);
        assert!(!config.prefer_refit);
    }

    // -------------------------------------------------------------------------
    // Test: Manager creation
    // -------------------------------------------------------------------------

    #[test]
    fn test_manager_new() {
        let config = DynamicBlasConfig::default();
        let manager = DynamicBlasManager::new(config);

        assert_eq!(manager.mesh_count(), 0);
        assert_eq!(manager.pending_refit_count(), 0);
        assert_eq!(manager.current_frame(), 0);
    }

    #[test]
    fn test_manager_default() {
        let manager = DynamicBlasManager::default();
        assert_eq!(manager.mesh_count(), 0);
    }

    // -------------------------------------------------------------------------
    // Test: Register dynamic mesh
    // -------------------------------------------------------------------------

    #[test]
    fn test_register_dynamic_mesh() {
        let mut manager = DynamicBlasManager::new(DynamicBlasConfig::default());
        let handle = make_handle(1);

        manager.register_dynamic_mesh("character_0", 5000, handle);

        assert_eq!(manager.mesh_count(), 1);
        assert!(manager.contains("character_0"));
        assert!(!manager.contains("character_1"));
    }

    #[test]
    fn test_register_multiple_meshes() {
        let mut manager = DynamicBlasManager::new(DynamicBlasConfig::default());

        manager.register_dynamic_mesh("mesh_a", 1000, make_handle(1));
        manager.register_dynamic_mesh("mesh_b", 2000, make_handle(2));
        manager.register_dynamic_mesh("mesh_c", 3000, make_handle(3));

        assert_eq!(manager.mesh_count(), 3);
        assert!(manager.contains("mesh_a"));
        assert!(manager.contains("mesh_b"));
        assert!(manager.contains("mesh_c"));
    }

    #[test]
    fn test_register_overwrites_existing() {
        let mut manager = DynamicBlasManager::new(DynamicBlasConfig::default());

        manager.register_dynamic_mesh("mesh", 1000, make_handle(1));
        manager.register_dynamic_mesh("mesh", 2000, make_handle(2));

        assert_eq!(manager.mesh_count(), 1);

        let entry = manager.get_entry("mesh").unwrap();
        assert_eq!(entry.vertex_count, 2000);
        assert_eq!(entry.handle.id(), 2);
    }

    // -------------------------------------------------------------------------
    // Test: Mark dirty / is_dirty
    // -------------------------------------------------------------------------

    #[test]
    fn test_mark_dirty() {
        let mut manager = DynamicBlasManager::new(DynamicBlasConfig::default());
        manager.register_dynamic_mesh("mesh", 1000, make_handle(1));

        assert!(!manager.is_dirty("mesh"));

        manager.mark_dirty("mesh").unwrap();

        assert!(manager.is_dirty("mesh"));
        assert_eq!(manager.pending_refit_count(), 1);
    }

    #[test]
    fn test_mark_dirty_unregistered() {
        let mut manager = DynamicBlasManager::new(DynamicBlasConfig::default());

        let result = manager.mark_dirty("nonexistent");

        assert!(matches!(
            result,
            Err(DynamicBlasError::MeshNotRegistered(_))
        ));
    }

    #[test]
    fn test_mark_dirty_idempotent() {
        let mut manager = DynamicBlasManager::new(DynamicBlasConfig::default());
        manager.register_dynamic_mesh("mesh", 1000, make_handle(1));

        manager.mark_dirty("mesh").unwrap();
        manager.mark_dirty("mesh").unwrap();
        manager.mark_dirty("mesh").unwrap();

        // Should only be in queue once
        assert_eq!(manager.pending_refit_count(), 1);
    }

    #[test]
    fn test_is_dirty_unregistered() {
        let manager = DynamicBlasManager::new(DynamicBlasConfig::default());
        assert!(!manager.is_dirty("nonexistent"));
    }

    // -------------------------------------------------------------------------
    // Test: Process refits respects max_count
    // -------------------------------------------------------------------------

    #[test]
    fn test_process_refits_respects_limit() {
        let config = DynamicBlasConfig::new().with_max_refit_per_frame(2);
        let mut manager = DynamicBlasManager::new(config);

        for i in 0..5 {
            manager.register_dynamic_mesh(&format!("mesh_{}", i), 1000, make_handle(i));
            manager.mark_dirty(&format!("mesh_{}", i)).unwrap();
        }

        assert_eq!(manager.pending_refit_count(), 5);

        let handles = manager.process_refits(None);

        assert_eq!(handles.len(), 2);
        assert_eq!(manager.pending_refit_count(), 3);
        assert_eq!(manager.refits_this_frame(), 2);
    }

    #[test]
    fn test_process_refits_custom_limit() {
        let mut manager = DynamicBlasManager::new(DynamicBlasConfig::default());

        for i in 0..10 {
            manager.register_dynamic_mesh(&format!("mesh_{}", i), 1000, make_handle(i));
            manager.mark_dirty(&format!("mesh_{}", i)).unwrap();
        }

        let handles = manager.process_refits(Some(3));

        assert_eq!(handles.len(), 3);
    }

    #[test]
    fn test_process_refits_clears_dirty() {
        let mut manager = DynamicBlasManager::new(DynamicBlasConfig::default());
        manager.register_dynamic_mesh("mesh", 1000, make_handle(1));
        manager.mark_dirty("mesh").unwrap();

        assert!(manager.is_dirty("mesh"));

        manager.process_refits(None);

        assert!(!manager.is_dirty("mesh"));
    }

    #[test]
    fn test_process_refits_empty_queue() {
        let mut manager = DynamicBlasManager::new(DynamicBlasConfig::default());

        let handles = manager.process_refits(None);

        assert!(handles.is_empty());
    }

    // -------------------------------------------------------------------------
    // Test: Begin frame clears per-frame state
    // -------------------------------------------------------------------------

    #[test]
    fn test_begin_frame_resets_refits_count() {
        let mut manager = DynamicBlasManager::new(DynamicBlasConfig::default());
        manager.register_dynamic_mesh("mesh", 1000, make_handle(1));
        manager.mark_dirty("mesh").unwrap();

        manager.process_refits(None);
        assert_eq!(manager.refits_this_frame(), 1);

        manager.begin_frame(1);

        assert_eq!(manager.refits_this_frame(), 0);
        assert_eq!(manager.current_frame(), 1);
    }

    #[test]
    fn test_begin_frame_preserves_dirty_flags() {
        let mut manager = DynamicBlasManager::new(DynamicBlasConfig::default());
        manager.register_dynamic_mesh("mesh", 1000, make_handle(1));
        manager.mark_dirty("mesh").unwrap();

        manager.begin_frame(1);

        // Dirty flag should NOT be cleared by begin_frame
        assert!(manager.is_dirty("mesh"));
        assert_eq!(manager.pending_refit_count(), 1);
    }

    // -------------------------------------------------------------------------
    // Test: Unregister removes from queue
    // -------------------------------------------------------------------------

    #[test]
    fn test_unregister_removes_entry() {
        let mut manager = DynamicBlasManager::new(DynamicBlasConfig::default());
        manager.register_dynamic_mesh("mesh", 1000, make_handle(1));

        assert!(manager.contains("mesh"));

        manager.unregister("mesh");

        assert!(!manager.contains("mesh"));
        assert_eq!(manager.mesh_count(), 0);
    }

    #[test]
    fn test_unregister_removes_from_queue() {
        let mut manager = DynamicBlasManager::new(DynamicBlasConfig::default());
        manager.register_dynamic_mesh("mesh_a", 1000, make_handle(1));
        manager.register_dynamic_mesh("mesh_b", 2000, make_handle(2));

        manager.mark_dirty("mesh_a").unwrap();
        manager.mark_dirty("mesh_b").unwrap();
        assert_eq!(manager.pending_refit_count(), 2);

        manager.unregister("mesh_a");

        assert_eq!(manager.pending_refit_count(), 1);
    }

    #[test]
    fn test_unregister_nonexistent() {
        let mut manager = DynamicBlasManager::new(DynamicBlasConfig::default());

        // Should not panic
        manager.unregister("nonexistent");
    }

    // -------------------------------------------------------------------------
    // Test: Pending count tracking
    // -------------------------------------------------------------------------

    #[test]
    fn test_pending_count_increases() {
        let mut manager = DynamicBlasManager::new(DynamicBlasConfig::default());

        for i in 0..3 {
            manager.register_dynamic_mesh(&format!("mesh_{}", i), 1000, make_handle(i));
        }

        assert_eq!(manager.pending_refit_count(), 0);

        manager.mark_dirty("mesh_0").unwrap();
        assert_eq!(manager.pending_refit_count(), 1);

        manager.mark_dirty("mesh_1").unwrap();
        assert_eq!(manager.pending_refit_count(), 2);

        manager.mark_dirty("mesh_2").unwrap();
        assert_eq!(manager.pending_refit_count(), 3);
    }

    #[test]
    fn test_pending_count_decreases_on_process() {
        let config = DynamicBlasConfig::new().with_max_refit_per_frame(10);
        let mut manager = DynamicBlasManager::new(config);

        for i in 0..5 {
            manager.register_dynamic_mesh(&format!("mesh_{}", i), 1000, make_handle(i));
            manager.mark_dirty(&format!("mesh_{}", i)).unwrap();
        }

        assert_eq!(manager.pending_refit_count(), 5);

        manager.process_refits(Some(2));
        assert_eq!(manager.pending_refit_count(), 3);

        // Start new frame to reset per-frame budget
        manager.begin_frame(1);
        manager.process_refits(Some(3));
        assert_eq!(manager.pending_refit_count(), 0);
    }

    // -------------------------------------------------------------------------
    // Test: Multiple meshes, priority ordering
    // -------------------------------------------------------------------------

    #[test]
    fn test_priority_ordering_by_vertex_count() {
        let config = DynamicBlasConfig::new().with_max_refit_per_frame(2);
        let mut manager = DynamicBlasManager::new(config);

        // Register with varying vertex counts
        manager.register_dynamic_mesh("large", 10000, make_handle(1));
        manager.register_dynamic_mesh("small", 100, make_handle(2));
        manager.register_dynamic_mesh("medium", 5000, make_handle(3));

        manager.mark_dirty("large").unwrap();
        manager.mark_dirty("small").unwrap();
        manager.mark_dirty("medium").unwrap();

        // Process should prioritize smaller meshes first
        let handles = manager.process_refits(None);

        assert_eq!(handles.len(), 2);
        // small (100) and medium (5000) should be processed before large (10000)
        assert!(handles.iter().any(|h| h.id() == 2)); // small
    }

    // -------------------------------------------------------------------------
    // Test: Error handling for unregistered mesh
    // -------------------------------------------------------------------------

    #[test]
    fn test_error_mesh_not_registered() {
        let err = DynamicBlasError::MeshNotRegistered("test_mesh".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("test_mesh"));
        assert!(msg.contains("not registered"));
    }

    #[test]
    fn test_error_refit_failed() {
        let err = DynamicBlasError::RefitFailed("GPU error".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("refit failed"));
        assert!(msg.contains("GPU error"));
    }

    #[test]
    fn test_error_invalid_handle() {
        let err = DynamicBlasError::InvalidHandle;
        let msg = format!("{}", err);
        assert!(msg.contains("invalid BLAS handle"));
    }

    // -------------------------------------------------------------------------
    // Test: Integration with skinning (on_skinning_complete)
    // -------------------------------------------------------------------------

    #[test]
    fn test_on_skinning_complete() {
        let mut manager = DynamicBlasManager::new(DynamicBlasConfig::default());

        manager.register_dynamic_mesh("char_0", 5000, make_handle(1));
        manager.register_dynamic_mesh("char_1", 6000, make_handle(2));
        manager.register_dynamic_mesh("char_2", 7000, make_handle(3));

        manager.on_skinning_complete(&["char_0", "char_2"]);

        assert!(manager.is_dirty("char_0"));
        assert!(!manager.is_dirty("char_1"));
        assert!(manager.is_dirty("char_2"));
        assert_eq!(manager.pending_refit_count(), 2);
    }

    #[test]
    fn test_on_skinning_complete_ignores_unregistered() {
        let mut manager = DynamicBlasManager::new(DynamicBlasConfig::default());

        manager.register_dynamic_mesh("char_0", 5000, make_handle(1));

        // Should not panic even with unregistered meshes
        manager.on_skinning_complete(&["char_0", "static_mesh", "another_static"]);

        assert!(manager.is_dirty("char_0"));
        assert_eq!(manager.pending_refit_count(), 1);
    }

    // -------------------------------------------------------------------------
    // Test: BlasHandle lifecycle
    // -------------------------------------------------------------------------

    #[test]
    fn test_process_returns_correct_handles() {
        let mut manager = DynamicBlasManager::new(DynamicBlasConfig::default());

        manager.register_dynamic_mesh("mesh_1", 1000, make_handle(100));
        manager.register_dynamic_mesh("mesh_2", 1000, make_handle(200));

        manager.mark_dirty("mesh_1").unwrap();
        manager.mark_dirty("mesh_2").unwrap();

        let handles = manager.process_refits(None);

        assert_eq!(handles.len(), 2);
        let ids: Vec<u32> = handles.iter().map(|h| h.id()).collect();
        assert!(ids.contains(&100));
        assert!(ids.contains(&200));
    }

    #[test]
    fn test_entry_handle_accessible() {
        let mut manager = DynamicBlasManager::new(DynamicBlasConfig::default());
        let handle = make_handle(42);

        manager.register_dynamic_mesh("mesh", 1000, handle);

        let entry = manager.get_entry("mesh").unwrap();
        assert_eq!(entry.handle.id(), 42);
    }

    // -------------------------------------------------------------------------
    // Test: Last refit frame tracking
    // -------------------------------------------------------------------------

    #[test]
    fn test_last_refit_frame_updated() {
        let mut manager = DynamicBlasManager::new(DynamicBlasConfig::default());
        manager.register_dynamic_mesh("mesh", 1000, make_handle(1));

        let entry = manager.get_entry("mesh").unwrap();
        assert_eq!(entry.last_refit_frame, 0);

        manager.begin_frame(10);
        manager.mark_dirty("mesh").unwrap();
        manager.process_refits(None);

        let entry = manager.get_entry("mesh").unwrap();
        assert_eq!(entry.last_refit_frame, 10);
    }

    // -------------------------------------------------------------------------
    // Test: Get entry
    // -------------------------------------------------------------------------

    #[test]
    fn test_get_entry_exists() {
        let mut manager = DynamicBlasManager::new(DynamicBlasConfig::default());
        manager.register_dynamic_mesh("mesh", 5000, make_handle(1));

        let entry = manager.get_entry("mesh");
        assert!(entry.is_some());

        let entry = entry.unwrap();
        assert_eq!(entry.mesh_id, "mesh");
        assert_eq!(entry.vertex_count, 5000);
    }

    #[test]
    fn test_get_entry_not_exists() {
        let manager = DynamicBlasManager::new(DynamicBlasConfig::default());
        assert!(manager.get_entry("nonexistent").is_none());
    }

    #[test]
    fn test_get_entry_mut() {
        let mut manager = DynamicBlasManager::new(DynamicBlasConfig::default());
        manager.register_dynamic_mesh("mesh", 1000, make_handle(1));

        {
            let entry = manager.get_entry_mut("mesh").unwrap();
            entry.vertex_count = 2000;
        }

        let entry = manager.get_entry("mesh").unwrap();
        assert_eq!(entry.vertex_count, 2000);
    }

    // -------------------------------------------------------------------------
    // Test: Statistics
    // -------------------------------------------------------------------------

    #[test]
    fn test_stats() {
        let mut manager = DynamicBlasManager::new(DynamicBlasConfig::default());

        manager.register_dynamic_mesh("mesh_a", 1000, make_handle(1));
        manager.register_dynamic_mesh("mesh_b", 2000, make_handle(2));
        manager.mark_dirty("mesh_a").unwrap();

        manager.begin_frame(5);

        let stats = manager.stats();

        assert_eq!(stats.mesh_count, 2);
        assert_eq!(stats.pending_refits, 1);
        assert_eq!(stats.dirty_count, 1);
        assert_eq!(stats.total_vertices, 3000);
        assert_eq!(stats.current_frame, 5);
    }

    #[test]
    fn test_stats_estimated_memory() {
        let mut manager = DynamicBlasManager::new(DynamicBlasConfig::default());

        manager.register_dynamic_mesh("mesh", 1000, make_handle(1));

        let stats = manager.stats();

        // 1000 vertices * 64 bytes = 64000
        assert_eq!(stats.estimated_memory, 64000);
    }

    // -------------------------------------------------------------------------
    // Test: Clear
    // -------------------------------------------------------------------------

    #[test]
    fn test_clear() {
        let mut manager = DynamicBlasManager::new(DynamicBlasConfig::default());

        manager.register_dynamic_mesh("mesh_a", 1000, make_handle(1));
        manager.register_dynamic_mesh("mesh_b", 2000, make_handle(2));
        manager.mark_dirty("mesh_a").unwrap();
        manager.begin_frame(10);
        manager.process_refits(Some(1));

        assert_eq!(manager.mesh_count(), 2);
        assert_eq!(manager.refits_this_frame(), 1);

        manager.clear();

        assert_eq!(manager.mesh_count(), 0);
        assert_eq!(manager.pending_refit_count(), 0);
        assert_eq!(manager.refits_this_frame(), 0);
    }

    // -------------------------------------------------------------------------
    // Test: Iteration
    // -------------------------------------------------------------------------

    #[test]
    fn test_iter() {
        let mut manager = DynamicBlasManager::new(DynamicBlasConfig::default());

        manager.register_dynamic_mesh("alpha", 1000, make_handle(1));
        manager.register_dynamic_mesh("beta", 2000, make_handle(2));

        let entries: Vec<_> = manager.iter().collect();

        assert_eq!(entries.len(), 2);

        let ids: Vec<&str> = entries.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&"alpha"));
        assert!(ids.contains(&"beta"));
    }

    // -------------------------------------------------------------------------
    // Test: Entry estimated memory
    // -------------------------------------------------------------------------

    #[test]
    fn test_entry_estimated_memory() {
        let entry = DynamicBlasEntry::new("mesh".to_string(), 1000, make_handle(1));
        assert_eq!(entry.estimated_memory(), 1000 * 64);
    }

    // -------------------------------------------------------------------------
    // Test: Config set/get
    // -------------------------------------------------------------------------

    #[test]
    fn test_set_config() {
        let mut manager = DynamicBlasManager::new(DynamicBlasConfig::default());

        let new_config = DynamicBlasConfig::for_crowd();
        manager.set_config(new_config);

        assert_eq!(manager.config().max_refit_per_frame, 16);
    }

    // -------------------------------------------------------------------------
    // Test: Frame budget across multiple process calls
    // -------------------------------------------------------------------------

    #[test]
    fn test_budget_across_process_calls() {
        let config = DynamicBlasConfig::new().with_max_refit_per_frame(5);
        let mut manager = DynamicBlasManager::new(config);

        for i in 0..10 {
            manager.register_dynamic_mesh(&format!("mesh_{}", i), 1000, make_handle(i));
            manager.mark_dirty(&format!("mesh_{}", i)).unwrap();
        }

        // First call: process 3
        let handles1 = manager.process_refits(Some(3));
        assert_eq!(handles1.len(), 3);
        assert_eq!(manager.refits_this_frame(), 3);

        // Second call: only 2 more allowed (5 - 3 = 2)
        let handles2 = manager.process_refits(None);
        assert_eq!(handles2.len(), 2);
        assert_eq!(manager.refits_this_frame(), 5);

        // Third call: budget exhausted
        let handles3 = manager.process_refits(None);
        assert_eq!(handles3.len(), 0);
    }
}
