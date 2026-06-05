//! BLAS Pool with Reference Counting
//!
//! This module provides a resource pool for managing Bottom-Level Acceleration
//! Structures (BLAS) with mesh-asset-ID-keyed reference counting.
//!
//! # Architecture
//!
//! - `BlasHandle`: Lightweight handle for referencing pooled BLAS entries
//! - `BlasPool`: Central pool managing BLAS lifecycle, builds, and compaction
//! - Reference counting ensures shared BLAS instances across multiple users
//! - Deferred build/compact queues for batched GPU operations
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::blas_pool::{BlasPool, BlasHandle};
//! use renderer_backend::blas::BlasConfig;
//!
//! let mut pool = BlasPool::new();
//!
//! // Queue a BLAS build for a mesh
//! pool.queue_build("mesh_cube", BlasConfig::for_static_geometry(), vertices);
//!
//! // Process pending builds
//! let built_handles = pool.process_pending();
//!
//! // Acquire handle (increments ref count if exists)
//! if let Some(handle) = pool.acquire("mesh_cube") {
//!     let blas = pool.get(handle).unwrap();
//!     // Use BLAS for ray tracing...
//!
//!     // Release when done
//!     pool.release(handle);
//! }
//!
//! // Garbage collect zero-ref entries
//! pool.gc(current_frame);
//! ```

use crate::blas::{Blas, BlasBuilder, BlasConfig, BlasError};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Frames to wait before garbage collecting zero-ref BLAS entries.
/// Allows for transient drops without immediate deletion.
const GC_GRACE_FRAMES: u64 = 3;

// ---------------------------------------------------------------------------
// BlasHandle
// ---------------------------------------------------------------------------

/// Lightweight handle for referencing a pooled BLAS entry.
///
/// Handles are copy-able and cheap to pass around. They remain valid
/// as long as the referenced entry exists in the pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlasHandle(u32);

impl BlasHandle {
    /// Create a new handle with the given ID.
    fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw handle ID.
    pub fn id(&self) -> u32 {
        self.0
    }
}

// ---------------------------------------------------------------------------
// BlasEntry
// ---------------------------------------------------------------------------

/// Internal entry in the BLAS pool.
struct BlasEntry {
    /// The actual BLAS data.
    blas: Blas,
    /// Reference count for this entry.
    ref_count: u32,
    /// Mesh asset ID this BLAS was built from.
    mesh_asset_id: String,
    /// Whether the BLAS has been compacted.
    is_compacted: bool,
    /// Frame number when this entry was created.
    created_frame: u64,
    /// Frame number when ref_count became zero (for GC grace period).
    zero_ref_frame: Option<u64>,
}

impl BlasEntry {
    /// Create a new entry with ref_count = 1.
    fn new(blas: Blas, mesh_asset_id: String, created_frame: u64) -> Self {
        Self {
            blas,
            ref_count: 1,
            mesh_asset_id,
            is_compacted: false,
            created_frame,
            zero_ref_frame: None,
        }
    }
}

// ---------------------------------------------------------------------------
// PendingBlas
// ---------------------------------------------------------------------------

/// Pending BLAS build request.
#[derive(Debug, Clone)]
pub struct PendingBlas {
    /// Mesh asset ID for the pending build.
    pub mesh_asset_id: String,
    /// Build configuration.
    pub config: BlasConfig,
    /// Vertex positions for the build.
    pub vertices: Vec<[f32; 3]>,
    /// Optional indices (if None, assumes sequential triangles).
    pub indices: Option<Vec<u32>>,
}

// ---------------------------------------------------------------------------
// BlasPoolError
// ---------------------------------------------------------------------------

/// Errors that can occur during BLAS pool operations.
#[derive(Debug)]
pub enum BlasPoolError {
    /// Handle not found in pool.
    HandleNotFound(BlasHandle),
    /// Mesh asset ID already exists in pool.
    DuplicateMeshId(String),
    /// BLAS build failed.
    BuildFailed(BlasError),
    /// Compact operation failed.
    CompactFailed(BlasError),
}

impl std::fmt::Display for BlasPoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HandleNotFound(h) => write!(f, "BLAS handle {} not found in pool", h.0),
            Self::DuplicateMeshId(id) => write!(f, "mesh asset ID '{}' already in pool", id),
            Self::BuildFailed(e) => write!(f, "BLAS build failed: {}", e),
            Self::CompactFailed(e) => write!(f, "BLAS compact failed: {}", e),
        }
    }
}

impl std::error::Error for BlasPoolError {}

impl From<BlasError> for BlasPoolError {
    fn from(err: BlasError) -> Self {
        BlasPoolError::BuildFailed(err)
    }
}

// ---------------------------------------------------------------------------
// BlasPool
// ---------------------------------------------------------------------------

/// Pool for managing BLAS resources with reference counting.
///
/// The pool maintains:
/// - A map of handles to BLAS entries
/// - A reverse map from mesh asset IDs to handles for deduplication
/// - Queues for pending builds and compactions
/// - Memory usage tracking
pub struct BlasPool {
    /// Active BLAS entries.
    entries: HashMap<BlasHandle, BlasEntry>,
    /// Mesh asset ID to handle mapping for deduplication.
    mesh_id_to_handle: HashMap<String, BlasHandle>,
    /// Next handle ID to allocate.
    next_handle: AtomicU32,
    /// Pending BLAS builds.
    pending_builds: Vec<PendingBlas>,
    /// Pending compaction requests.
    pending_compacts: Vec<BlasHandle>,
    /// Total memory usage across all BLAS entries.
    total_memory: AtomicUsize,
    /// Current frame number (for GC timing).
    current_frame: u64,
}

impl BlasPool {
    /// Create a new empty BLAS pool.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            mesh_id_to_handle: HashMap::new(),
            next_handle: AtomicU32::new(1), // Start at 1, 0 could be "invalid"
            pending_builds: Vec::new(),
            pending_compacts: Vec::new(),
            total_memory: AtomicUsize::new(0),
            current_frame: 0,
        }
    }

    /// Allocate a new unique handle.
    fn alloc_handle(&self) -> BlasHandle {
        BlasHandle::new(self.next_handle.fetch_add(1, Ordering::Relaxed))
    }

    /// Acquire a handle for a mesh asset ID.
    ///
    /// If the mesh already exists in the pool, increments the reference count
    /// and returns the existing handle. Otherwise returns `None`.
    ///
    /// # Arguments
    ///
    /// * `mesh_asset_id` - Unique identifier for the mesh asset
    ///
    /// # Returns
    ///
    /// `Some(handle)` if the mesh exists, `None` otherwise.
    pub fn acquire(&mut self, mesh_asset_id: &str) -> Option<BlasHandle> {
        if let Some(&handle) = self.mesh_id_to_handle.get(mesh_asset_id) {
            if let Some(entry) = self.entries.get_mut(&handle) {
                entry.ref_count += 1;
                entry.zero_ref_frame = None; // Clear GC timer
                return Some(handle);
            }
        }
        None
    }

    /// Release a handle, decrementing its reference count.
    ///
    /// When the reference count reaches zero, the entry is marked for
    /// potential garbage collection (after a grace period).
    ///
    /// # Arguments
    ///
    /// * `handle` - The handle to release
    ///
    /// # Returns
    ///
    /// `true` if the handle was found and released, `false` otherwise.
    pub fn release(&mut self, handle: BlasHandle) -> bool {
        if let Some(entry) = self.entries.get_mut(&handle) {
            if entry.ref_count > 0 {
                entry.ref_count -= 1;
                if entry.ref_count == 0 {
                    entry.zero_ref_frame = Some(self.current_frame);
                }
            }
            true
        } else {
            false
        }
    }

    /// Insert a new BLAS into the pool.
    ///
    /// # Arguments
    ///
    /// * `mesh_asset_id` - Unique identifier for the mesh asset
    /// * `blas` - The built BLAS to insert
    ///
    /// # Returns
    ///
    /// A new handle for the inserted BLAS.
    ///
    /// # Note
    ///
    /// If a BLAS with the same mesh_asset_id already exists, this will
    /// create a new entry with a different handle. Use `acquire` first
    /// to check for existing entries.
    pub fn insert(&mut self, mesh_asset_id: &str, blas: Blas) -> BlasHandle {
        let handle = self.alloc_handle();
        let memory = blas.memory_size;

        let entry = BlasEntry::new(blas, mesh_asset_id.to_string(), self.current_frame);

        self.entries.insert(handle, entry);
        self.mesh_id_to_handle
            .insert(mesh_asset_id.to_string(), handle);
        self.total_memory.fetch_add(memory, Ordering::Relaxed);

        handle
    }

    /// Get a reference to a BLAS by handle.
    ///
    /// # Arguments
    ///
    /// * `handle` - The handle to look up
    ///
    /// # Returns
    ///
    /// `Some(&Blas)` if found, `None` otherwise.
    pub fn get(&self, handle: BlasHandle) -> Option<&Blas> {
        self.entries.get(&handle).map(|e| &e.blas)
    }

    /// Get a mutable reference to a BLAS by handle.
    ///
    /// # Arguments
    ///
    /// * `handle` - The handle to look up
    ///
    /// # Returns
    ///
    /// `Some(&mut Blas)` if found, `None` otherwise.
    pub fn get_mut(&mut self, handle: BlasHandle) -> Option<&mut Blas> {
        self.entries.get_mut(&handle).map(|e| &mut e.blas)
    }

    /// Get the reference count for a handle.
    ///
    /// # Returns
    ///
    /// `Some(count)` if handle exists, `None` otherwise.
    pub fn ref_count(&self, handle: BlasHandle) -> Option<u32> {
        self.entries.get(&handle).map(|e| e.ref_count)
    }

    /// Check if a mesh asset ID exists in the pool.
    pub fn contains_mesh(&self, mesh_asset_id: &str) -> bool {
        self.mesh_id_to_handle.contains_key(mesh_asset_id)
    }

    /// Get the handle for a mesh asset ID if it exists.
    pub fn get_handle(&self, mesh_asset_id: &str) -> Option<BlasHandle> {
        self.mesh_id_to_handle.get(mesh_asset_id).copied()
    }

    /// Queue a BLAS build request.
    ///
    /// The build will be processed on the next call to `process_pending()`.
    ///
    /// # Arguments
    ///
    /// * `mesh_asset_id` - Unique identifier for the mesh
    /// * `config` - BLAS build configuration
    /// * `vertices` - Vertex positions
    pub fn queue_build(&mut self, mesh_asset_id: &str, config: BlasConfig, vertices: Vec<[f32; 3]>) {
        self.pending_builds.push(PendingBlas {
            mesh_asset_id: mesh_asset_id.to_string(),
            config,
            vertices,
            indices: None,
        });
    }

    /// Queue a BLAS build request with indices.
    ///
    /// # Arguments
    ///
    /// * `mesh_asset_id` - Unique identifier for the mesh
    /// * `config` - BLAS build configuration
    /// * `vertices` - Vertex positions
    /// * `indices` - Triangle indices
    pub fn queue_build_indexed(
        &mut self,
        mesh_asset_id: &str,
        config: BlasConfig,
        vertices: Vec<[f32; 3]>,
        indices: Vec<u32>,
    ) {
        self.pending_builds.push(PendingBlas {
            mesh_asset_id: mesh_asset_id.to_string(),
            config,
            vertices,
            indices: Some(indices),
        });
    }

    /// Queue a compaction request for a BLAS.
    ///
    /// The compaction will be processed on the next call to `process_pending()`.
    ///
    /// # Arguments
    ///
    /// * `handle` - Handle of the BLAS to compact
    pub fn queue_compact(&mut self, handle: BlasHandle) {
        if !self.pending_compacts.contains(&handle) {
            self.pending_compacts.push(handle);
        }
    }

    /// Process all pending build and compact requests.
    ///
    /// # Returns
    ///
    /// A vector of handles for successfully built/compacted BLAS entries.
    pub fn process_pending(&mut self) -> Vec<BlasHandle> {
        let mut processed_handles = Vec::new();

        // Process builds
        let builds = std::mem::take(&mut self.pending_builds);
        for pending in builds {
            // Skip if mesh already exists
            if self.contains_mesh(&pending.mesh_asset_id) {
                if let Some(handle) = self.acquire(&pending.mesh_asset_id) {
                    processed_handles.push(handle);
                }
                continue;
            }

            // Build the BLAS
            let builder = BlasBuilder::new().config(pending.config);
            let builder = if let Some(indices) = pending.indices {
                builder.vertices_indices(&pending.vertices, &indices)
            } else {
                builder.vertices(&pending.vertices)
            };

            match builder.build() {
                Ok(blas) => {
                    let handle = self.insert(&pending.mesh_asset_id, blas);
                    processed_handles.push(handle);
                }
                Err(_e) => {
                    // Build failed - could log error here
                }
            }
        }

        // Process compactions
        let compacts = std::mem::take(&mut self.pending_compacts);
        for handle in compacts {
            if let Some(entry) = self.entries.get_mut(&handle) {
                if !entry.is_compacted && entry.blas.supports_compaction() {
                    let old_size = entry.blas.memory_size;
                    if entry.blas.compact().is_ok() {
                        entry.is_compacted = true;
                        let saved = old_size.saturating_sub(entry.blas.memory_size);
                        self.total_memory.fetch_sub(saved, Ordering::Relaxed);
                        processed_handles.push(handle);
                    }
                }
            }
        }

        processed_handles
    }

    /// Garbage collect zero-reference entries after grace period.
    ///
    /// # Arguments
    ///
    /// * `frame` - Current frame number
    ///
    /// # Returns
    ///
    /// Number of entries garbage collected.
    pub fn gc(&mut self, frame: u64) -> usize {
        self.current_frame = frame;

        let mut to_remove = Vec::new();

        for (handle, entry) in &self.entries {
            if entry.ref_count == 0 {
                if let Some(zero_frame) = entry.zero_ref_frame {
                    if frame >= zero_frame + GC_GRACE_FRAMES {
                        to_remove.push(*handle);
                    }
                }
            }
        }

        let count = to_remove.len();
        for handle in to_remove {
            if let Some(entry) = self.entries.remove(&handle) {
                self.mesh_id_to_handle.remove(&entry.mesh_asset_id);
                self.total_memory
                    .fetch_sub(entry.blas.memory_size, Ordering::Relaxed);
            }
        }

        count
    }

    /// Force immediate garbage collection of zero-ref entries.
    ///
    /// Ignores the grace period and removes all zero-ref entries immediately.
    ///
    /// # Returns
    ///
    /// Number of entries removed.
    pub fn gc_immediate(&mut self) -> usize {
        let mut to_remove = Vec::new();

        for (handle, entry) in &self.entries {
            if entry.ref_count == 0 {
                to_remove.push(*handle);
            }
        }

        let count = to_remove.len();
        for handle in to_remove {
            if let Some(entry) = self.entries.remove(&handle) {
                self.mesh_id_to_handle.remove(&entry.mesh_asset_id);
                self.total_memory
                    .fetch_sub(entry.blas.memory_size, Ordering::Relaxed);
            }
        }

        count
    }

    /// Get total memory usage across all BLAS entries.
    pub fn memory_usage(&self) -> usize {
        self.total_memory.load(Ordering::Relaxed)
    }

    /// Get the number of active entries in the pool.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the number of pending builds.
    pub fn pending_build_count(&self) -> usize {
        self.pending_builds.len()
    }

    /// Get the number of pending compactions.
    pub fn pending_compact_count(&self) -> usize {
        self.pending_compacts.len()
    }

    /// Clear all pending builds.
    pub fn clear_pending_builds(&mut self) {
        self.pending_builds.clear();
    }

    /// Clear all pending compactions.
    pub fn clear_pending_compacts(&mut self) {
        self.pending_compacts.clear();
    }

    /// Get statistics about the pool.
    pub fn stats(&self) -> BlasPoolStats {
        let mut total_triangles = 0u64;
        let mut total_vertices = 0u64;
        let mut compacted_count = 0usize;

        for entry in self.entries.values() {
            total_triangles += entry.blas.triangle_count as u64;
            total_vertices += entry.blas.vertex_count as u64;
            if entry.is_compacted {
                compacted_count += 1;
            }
        }

        BlasPoolStats {
            entry_count: self.entries.len(),
            total_memory: self.memory_usage(),
            total_triangles,
            total_vertices,
            pending_builds: self.pending_builds.len(),
            pending_compacts: self.pending_compacts.len(),
            compacted_count,
        }
    }

    /// Iterate over all handles in the pool.
    pub fn handles(&self) -> impl Iterator<Item = BlasHandle> + '_ {
        self.entries.keys().copied()
    }

    /// Iterate over all (handle, mesh_id) pairs.
    pub fn entries_iter(&self) -> impl Iterator<Item = (BlasHandle, &str)> + '_ {
        self.entries
            .iter()
            .map(|(h, e)| (*h, e.mesh_asset_id.as_str()))
    }
}

impl Default for BlasPool {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// BlasPoolStats
// ---------------------------------------------------------------------------

/// Statistics about the BLAS pool.
#[derive(Debug, Clone, Default)]
pub struct BlasPoolStats {
    /// Number of active entries.
    pub entry_count: usize,
    /// Total memory usage in bytes.
    pub total_memory: usize,
    /// Total triangle count across all entries.
    pub total_triangles: u64,
    /// Total vertex count across all entries.
    pub total_vertices: u64,
    /// Number of pending build requests.
    pub pending_builds: usize,
    /// Number of pending compact requests.
    pub pending_compacts: usize,
    /// Number of compacted entries.
    pub compacted_count: usize,
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

    /// Create a simple triangle mesh.
    fn make_triangle() -> Vec<[f32; 3]> {
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]]
    }

    /// Create a quad mesh (6 vertices, 2 triangles).
    fn make_quad() -> (Vec<[f32; 3]>, Vec<u32>) {
        let vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let indices = vec![0, 1, 2, 0, 2, 3];
        (vertices, indices)
    }

    /// Create a cube mesh.
    fn make_cube() -> (Vec<[f32; 3]>, Vec<u32>) {
        let vertices = vec![
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
        ];
        let indices = vec![
            0, 1, 2, 0, 2, 3, // Front
            4, 6, 5, 4, 7, 6, // Back
            0, 3, 7, 0, 7, 4, // Left
            1, 5, 6, 1, 6, 2, // Right
            3, 2, 6, 3, 6, 7, // Top
            0, 4, 5, 0, 5, 1, // Bottom
        ];
        (vertices, indices)
    }

    /// Build a BLAS from vertices.
    fn build_triangle_blas() -> Blas {
        let vertices = make_triangle();
        BlasBuilder::new()
            .vertices(&vertices)
            .build()
            .expect("Failed to build BLAS")
    }

    /// Build a BLAS with compaction enabled.
    fn build_compactable_blas() -> Blas {
        let (vertices, indices) = make_cube();
        BlasBuilder::new()
            .config(BlasConfig::for_static_geometry())
            .vertices_indices(&vertices, &indices)
            .build()
            .expect("Failed to build BLAS")
    }

    // -------------------------------------------------------------------------
    // Test: Pool creation
    // -------------------------------------------------------------------------

    #[test]
    fn test_pool_new() {
        let pool = BlasPool::new();
        assert!(pool.is_empty());
        assert_eq!(pool.len(), 0);
        assert_eq!(pool.memory_usage(), 0);
    }

    #[test]
    fn test_pool_default() {
        let pool = BlasPool::default();
        assert!(pool.is_empty());
    }

    // -------------------------------------------------------------------------
    // Test: Insert and get
    // -------------------------------------------------------------------------

    #[test]
    fn test_insert_returns_handle() {
        let mut pool = BlasPool::new();
        let blas = build_triangle_blas();
        let handle = pool.insert("mesh_triangle", blas);

        assert_eq!(pool.len(), 1);
        assert!(!pool.is_empty());
        assert!(pool.get(handle).is_some());
    }

    #[test]
    fn test_insert_and_get_round_trip() {
        let mut pool = BlasPool::new();
        let blas = build_triangle_blas();
        let expected_triangles = blas.triangle_count;

        let handle = pool.insert("mesh_test", blas);
        let retrieved = pool.get(handle).expect("BLAS not found");

        assert_eq!(retrieved.triangle_count, expected_triangles);
    }

    #[test]
    fn test_get_nonexistent_handle() {
        let pool = BlasPool::new();
        let fake_handle = BlasHandle::new(999);
        assert!(pool.get(fake_handle).is_none());
    }

    #[test]
    fn test_get_mut() {
        let mut pool = BlasPool::new();
        let blas = BlasBuilder::new()
            .config(BlasConfig::new().with_update(true))
            .vertices(&make_triangle())
            .build()
            .unwrap();

        let handle = pool.insert("mesh_mut", blas);

        // Get mutable reference and modify
        let blas_mut = pool.get_mut(handle).expect("BLAS not found");
        let new_vertices = vec![[1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [1.5, 2.0, 0.0]];
        blas_mut.update(&new_vertices).expect("Update failed");

        // Verify modification persisted
        let blas_ref = pool.get(handle).expect("BLAS not found");
        assert!(blas_ref.bounds.min[0] >= 0.9);
    }

    // -------------------------------------------------------------------------
    // Test: Acquire and release
    // -------------------------------------------------------------------------

    #[test]
    fn test_acquire_existing_mesh() {
        let mut pool = BlasPool::new();
        let blas = build_triangle_blas();
        let handle1 = pool.insert("shared_mesh", blas);

        let handle2 = pool.acquire("shared_mesh");
        assert_eq!(handle2, Some(handle1));
        assert_eq!(pool.ref_count(handle1), Some(2));
    }

    #[test]
    fn test_acquire_nonexistent_mesh() {
        let mut pool = BlasPool::new();
        assert!(pool.acquire("nonexistent").is_none());
    }

    #[test]
    fn test_acquire_same_mesh_same_handle() {
        let mut pool = BlasPool::new();
        let blas = build_triangle_blas();
        let handle = pool.insert("mesh_a", blas);

        let acquired1 = pool.acquire("mesh_a").unwrap();
        let acquired2 = pool.acquire("mesh_a").unwrap();

        assert_eq!(handle, acquired1);
        assert_eq!(acquired1, acquired2);
        assert_eq!(pool.ref_count(handle), Some(3)); // 1 (insert) + 2 (acquires)
    }

    #[test]
    fn test_release_decrements_ref_count() {
        let mut pool = BlasPool::new();
        let blas = build_triangle_blas();
        let handle = pool.insert("mesh_release", blas);

        assert_eq!(pool.ref_count(handle), Some(1));

        pool.release(handle);
        assert_eq!(pool.ref_count(handle), Some(0));
    }

    #[test]
    fn test_release_nonexistent_handle() {
        let mut pool = BlasPool::new();
        let fake_handle = BlasHandle::new(999);
        assert!(!pool.release(fake_handle));
    }

    #[test]
    fn test_release_multiple_times() {
        let mut pool = BlasPool::new();
        let blas = build_triangle_blas();
        let handle = pool.insert("mesh_multi", blas);

        pool.acquire("mesh_multi");
        pool.acquire("mesh_multi");
        assert_eq!(pool.ref_count(handle), Some(3));

        pool.release(handle);
        pool.release(handle);
        pool.release(handle);
        assert_eq!(pool.ref_count(handle), Some(0));

        // Extra release should not go negative
        pool.release(handle);
        assert_eq!(pool.ref_count(handle), Some(0));
    }

    // -------------------------------------------------------------------------
    // Test: Garbage collection
    // -------------------------------------------------------------------------

    #[test]
    fn test_gc_zero_ref_after_grace_period() {
        let mut pool = BlasPool::new();
        let blas = build_triangle_blas();
        let handle = pool.insert("mesh_gc", blas);

        pool.release(handle);
        assert_eq!(pool.ref_count(handle), Some(0));

        // GC at same frame should not remove (grace period)
        let removed = pool.gc(0);
        assert_eq!(removed, 0);
        assert_eq!(pool.len(), 1);

        // GC after grace period should remove
        let removed = pool.gc(GC_GRACE_FRAMES + 1);
        assert_eq!(removed, 1);
        assert!(pool.is_empty());
    }

    #[test]
    fn test_gc_preserves_active_refs() {
        let mut pool = BlasPool::new();
        let blas = build_triangle_blas();
        let handle = pool.insert("mesh_active", blas);

        // Don't release, ref_count stays 1
        let removed = pool.gc(100);
        assert_eq!(removed, 0);
        assert_eq!(pool.len(), 1);
        assert!(pool.get(handle).is_some());
    }

    #[test]
    fn test_gc_immediate() {
        let mut pool = BlasPool::new();
        let blas = build_triangle_blas();
        let handle = pool.insert("mesh_immediate", blas);

        pool.release(handle);
        let removed = pool.gc_immediate();

        assert_eq!(removed, 1);
        assert!(pool.is_empty());
    }

    #[test]
    fn test_gc_removes_mesh_id_mapping() {
        let mut pool = BlasPool::new();
        let blas = build_triangle_blas();
        let handle = pool.insert("mesh_mapping", blas);

        assert!(pool.contains_mesh("mesh_mapping"));

        pool.release(handle);
        pool.gc_immediate();

        assert!(!pool.contains_mesh("mesh_mapping"));
        assert!(pool.get_handle("mesh_mapping").is_none());
    }

    // -------------------------------------------------------------------------
    // Test: Pending builds
    // -------------------------------------------------------------------------

    #[test]
    fn test_queue_build() {
        let mut pool = BlasPool::new();
        let vertices = make_triangle();

        pool.queue_build("mesh_queued", BlasConfig::default(), vertices);

        assert_eq!(pool.pending_build_count(), 1);
        assert!(pool.is_empty()); // Not built yet
    }

    #[test]
    fn test_process_pending_builds() {
        let mut pool = BlasPool::new();
        let vertices = make_triangle();

        pool.queue_build("mesh_process", BlasConfig::default(), vertices);
        let handles = pool.process_pending();

        assert_eq!(handles.len(), 1);
        assert_eq!(pool.len(), 1);
        assert_eq!(pool.pending_build_count(), 0);
        assert!(pool.contains_mesh("mesh_process"));
    }

    #[test]
    fn test_queue_build_indexed() {
        let mut pool = BlasPool::new();
        let (vertices, indices) = make_quad();

        pool.queue_build_indexed("mesh_indexed", BlasConfig::default(), vertices, indices);
        let handles = pool.process_pending();

        assert_eq!(handles.len(), 1);
        let blas = pool.get(handles[0]).unwrap();
        assert_eq!(blas.triangle_count, 2);
    }

    #[test]
    fn test_process_pending_skips_duplicate() {
        let mut pool = BlasPool::new();
        let vertices = make_triangle();

        // Insert first
        let blas = build_triangle_blas();
        let handle1 = pool.insert("mesh_dup", blas);

        // Queue build for same mesh
        pool.queue_build("mesh_dup", BlasConfig::default(), vertices);
        let handles = pool.process_pending();

        // Should acquire existing instead of building new
        assert_eq!(handles.len(), 1);
        assert_eq!(handles[0], handle1);
        assert_eq!(pool.len(), 1);
        assert_eq!(pool.ref_count(handle1), Some(2)); // Original + acquire
    }

    #[test]
    fn test_clear_pending_builds() {
        let mut pool = BlasPool::new();
        pool.queue_build("mesh_clear", BlasConfig::default(), make_triangle());

        assert_eq!(pool.pending_build_count(), 1);
        pool.clear_pending_builds();
        assert_eq!(pool.pending_build_count(), 0);
    }

    // -------------------------------------------------------------------------
    // Test: Pending compactions
    // -------------------------------------------------------------------------

    #[test]
    fn test_queue_compact() {
        let mut pool = BlasPool::new();
        let blas = build_compactable_blas();
        let handle = pool.insert("mesh_compact", blas);

        pool.queue_compact(handle);
        assert_eq!(pool.pending_compact_count(), 1);
    }

    #[test]
    fn test_queue_compact_deduplicates() {
        let mut pool = BlasPool::new();
        let blas = build_compactable_blas();
        let handle = pool.insert("mesh_dedup", blas);

        pool.queue_compact(handle);
        pool.queue_compact(handle);
        pool.queue_compact(handle);

        assert_eq!(pool.pending_compact_count(), 1);
    }

    #[test]
    fn test_process_pending_compacts() {
        let mut pool = BlasPool::new();
        let blas = build_compactable_blas();
        let original_memory = blas.memory_size;
        let handle = pool.insert("mesh_do_compact", blas);

        pool.queue_compact(handle);
        let handles = pool.process_pending();

        assert_eq!(handles.len(), 1);
        assert_eq!(pool.pending_compact_count(), 0);

        let blas = pool.get(handle).unwrap();
        assert!(blas.compacted);
        assert!(blas.memory_size < original_memory);
    }

    #[test]
    fn test_clear_pending_compacts() {
        let mut pool = BlasPool::new();
        let blas = build_compactable_blas();
        let handle = pool.insert("mesh_clear_c", blas);

        pool.queue_compact(handle);
        assert_eq!(pool.pending_compact_count(), 1);

        pool.clear_pending_compacts();
        assert_eq!(pool.pending_compact_count(), 0);
    }

    // -------------------------------------------------------------------------
    // Test: Memory tracking
    // -------------------------------------------------------------------------

    #[test]
    fn test_memory_usage_increases_on_insert() {
        let mut pool = BlasPool::new();
        assert_eq!(pool.memory_usage(), 0);

        let blas = build_triangle_blas();
        let expected_memory = blas.memory_size;

        pool.insert("mesh_mem", blas);
        assert_eq!(pool.memory_usage(), expected_memory);
    }

    #[test]
    fn test_memory_usage_decreases_on_gc() {
        let mut pool = BlasPool::new();
        let blas = build_triangle_blas();
        let handle = pool.insert("mesh_gc_mem", blas);

        let memory_before = pool.memory_usage();
        assert!(memory_before > 0);

        pool.release(handle);
        pool.gc_immediate();

        assert_eq!(pool.memory_usage(), 0);
    }

    #[test]
    fn test_memory_usage_decreases_on_compact() {
        let mut pool = BlasPool::new();
        let blas = build_compactable_blas();
        let handle = pool.insert("mesh_compact_mem", blas);

        let memory_before = pool.memory_usage();

        pool.queue_compact(handle);
        pool.process_pending();

        assert!(pool.memory_usage() < memory_before);
    }

    // -------------------------------------------------------------------------
    // Test: Multiple meshes
    // -------------------------------------------------------------------------

    #[test]
    fn test_multiple_meshes_independent() {
        let mut pool = BlasPool::new();

        let blas1 = build_triangle_blas();
        let blas2 = build_compactable_blas();

        let handle1 = pool.insert("mesh_1", blas1);
        let handle2 = pool.insert("mesh_2", blas2);

        assert_ne!(handle1, handle2);
        assert_eq!(pool.len(), 2);
        assert_eq!(pool.ref_count(handle1), Some(1));
        assert_eq!(pool.ref_count(handle2), Some(1));

        // Release one, other unaffected
        pool.release(handle1);
        assert_eq!(pool.ref_count(handle1), Some(0));
        assert_eq!(pool.ref_count(handle2), Some(1));

        // GC removes only zero-ref
        pool.gc_immediate();
        assert_eq!(pool.len(), 1);
        assert!(pool.get(handle2).is_some());
        assert!(pool.get(handle1).is_none());
    }

    #[test]
    fn test_handle_uniqueness() {
        let mut pool = BlasPool::new();

        let mut handles = Vec::new();
        for i in 0..100 {
            let blas = build_triangle_blas();
            let handle = pool.insert(&format!("mesh_{}", i), blas);
            handles.push(handle);
        }

        // All handles should be unique
        let mut seen = std::collections::HashSet::new();
        for handle in &handles {
            assert!(seen.insert(*handle), "Duplicate handle found");
        }
    }

    // -------------------------------------------------------------------------
    // Test: Handle operations
    // -------------------------------------------------------------------------

    #[test]
    fn test_handle_id() {
        let handle = BlasHandle::new(42);
        assert_eq!(handle.id(), 42);
    }

    #[test]
    fn test_handle_equality() {
        let h1 = BlasHandle::new(5);
        let h2 = BlasHandle::new(5);
        let h3 = BlasHandle::new(6);

        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_handle_hash() {
        use std::collections::HashMap;

        let mut map: HashMap<BlasHandle, &str> = HashMap::new();
        let h1 = BlasHandle::new(1);
        let h2 = BlasHandle::new(2);

        map.insert(h1, "first");
        map.insert(h2, "second");

        assert_eq!(map.get(&h1), Some(&"first"));
        assert_eq!(map.get(&h2), Some(&"second"));
    }

    // -------------------------------------------------------------------------
    // Test: Contains and get_handle
    // -------------------------------------------------------------------------

    #[test]
    fn test_contains_mesh() {
        let mut pool = BlasPool::new();
        let blas = build_triangle_blas();

        assert!(!pool.contains_mesh("mesh_check"));
        pool.insert("mesh_check", blas);
        assert!(pool.contains_mesh("mesh_check"));
    }

    #[test]
    fn test_get_handle_by_mesh_id() {
        let mut pool = BlasPool::new();
        let blas = build_triangle_blas();
        let handle = pool.insert("mesh_lookup", blas);

        assert_eq!(pool.get_handle("mesh_lookup"), Some(handle));
        assert_eq!(pool.get_handle("nonexistent"), None);
    }

    // -------------------------------------------------------------------------
    // Test: Statistics
    // -------------------------------------------------------------------------

    #[test]
    fn test_stats() {
        let mut pool = BlasPool::new();

        let (vertices, indices) = make_cube();
        pool.queue_build_indexed("mesh_stats", BlasConfig::for_static_geometry(), vertices, indices);
        pool.process_pending();

        let stats = pool.stats();
        assert_eq!(stats.entry_count, 1);
        assert!(stats.total_memory > 0);
        assert_eq!(stats.total_triangles, 12);
        assert_eq!(stats.total_vertices, 8);
        assert_eq!(stats.pending_builds, 0);
        assert_eq!(stats.pending_compacts, 0);
        assert_eq!(stats.compacted_count, 0);
    }

    #[test]
    fn test_stats_after_compact() {
        let mut pool = BlasPool::new();
        let blas = build_compactable_blas();
        let handle = pool.insert("mesh_stats_compact", blas);

        pool.queue_compact(handle);
        pool.process_pending();

        let stats = pool.stats();
        assert_eq!(stats.compacted_count, 1);
    }

    // -------------------------------------------------------------------------
    // Test: Iterators
    // -------------------------------------------------------------------------

    #[test]
    fn test_handles_iterator() {
        let mut pool = BlasPool::new();

        let h1 = pool.insert("mesh_iter_1", build_triangle_blas());
        let h2 = pool.insert("mesh_iter_2", build_triangle_blas());

        let handles: Vec<_> = pool.handles().collect();
        assert_eq!(handles.len(), 2);
        assert!(handles.contains(&h1));
        assert!(handles.contains(&h2));
    }

    #[test]
    fn test_entries_iter() {
        let mut pool = BlasPool::new();

        pool.insert("alpha", build_triangle_blas());
        pool.insert("beta", build_triangle_blas());

        let entries: Vec<_> = pool.entries_iter().collect();
        assert_eq!(entries.len(), 2);

        let mesh_ids: Vec<_> = entries.iter().map(|(_, id)| *id).collect();
        assert!(mesh_ids.contains(&"alpha"));
        assert!(mesh_ids.contains(&"beta"));
    }

    // -------------------------------------------------------------------------
    // Test: Acquire clears GC timer
    // -------------------------------------------------------------------------

    #[test]
    fn test_acquire_clears_gc_timer() {
        let mut pool = BlasPool::new();
        let blas = build_triangle_blas();
        let handle = pool.insert("mesh_timer", blas);

        // Release and mark for GC
        pool.release(handle);
        pool.gc(0); // Starts grace period

        // Re-acquire before GC completes
        let reacquired = pool.acquire("mesh_timer").unwrap();
        assert_eq!(reacquired, handle);

        // GC should not remove even after grace period
        let removed = pool.gc(100);
        assert_eq!(removed, 0);
        assert!(pool.get(handle).is_some());
    }
}
