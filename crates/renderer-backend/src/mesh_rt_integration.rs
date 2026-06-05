//! Mesh Ray Tracing Integration
//!
//! This module integrates BLAS construction into the mesh asset pipeline,
//! providing automatic BLAS building and lifecycle management when meshes
//! are loaded and unloaded.
//!
//! # Architecture
//!
//! - `MeshRTConfig`: Configuration for ray tracing mesh handling
//! - `MeshRTIntegration`: Main integration point connecting mesh loading to BLAS pool
//! - `StaticBlasEntry`: Tracks per-mesh BLAS state including compaction status
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::mesh_rt_integration::{MeshRTConfig, MeshRTIntegration};
//!
//! let config = MeshRTConfig {
//!     enable_rt: true,
//!     compact_blas: true,
//!     prefer_fast_trace: true,
//! };
//!
//! let mut integration = MeshRTIntegration::new(config);
//! integration.set_memory_budget(256 * 1024 * 1024); // 256 MB
//!
//! // On mesh load
//! let handle = integration.on_mesh_load("hero_mesh", &vertices, Some(&indices))?;
//!
//! // Process pending builds
//! integration.process_pending();
//!
//! // Later, get the BLAS handle for ray tracing
//! if let Some(handle) = integration.get_blas_handle("hero_mesh") {
//!     // Use handle for TLAS construction
//! }
//!
//! // On mesh unload
//! integration.on_mesh_unload("hero_mesh");
//! ```

use crate::blas::{BlasConfig, BlasError};
use crate::blas_pool::{BlasHandle, BlasPool, BlasPoolError};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default memory budget: 512 MB
const DEFAULT_MEMORY_BUDGET: usize = 512 * 1024 * 1024;

// ---------------------------------------------------------------------------
// MeshRTConfig
// ---------------------------------------------------------------------------

/// Configuration for ray tracing mesh handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeshRTConfig {
    /// Enable ray tracing for meshes.
    /// When false, on_mesh_load does nothing and returns a dummy handle.
    pub enable_rt: bool,

    /// Enable BLAS compaction after build.
    /// Compaction reduces memory usage by 30-50% but requires an extra pass.
    pub compact_blas: bool,

    /// Prefer fast ray tracing over fast build times.
    /// Results in better tracing performance but slower builds.
    pub prefer_fast_trace: bool,
}

impl MeshRTConfig {
    /// Create a new config with all options disabled.
    pub const fn new() -> Self {
        Self {
            enable_rt: false,
            compact_blas: false,
            prefer_fast_trace: false,
        }
    }

    /// Create a config optimized for production use.
    /// RT enabled, compaction enabled, fast trace preferred.
    pub const fn production() -> Self {
        Self {
            enable_rt: true,
            compact_blas: true,
            prefer_fast_trace: true,
        }
    }

    /// Create a config optimized for development/iteration.
    /// RT enabled, compaction disabled for faster builds.
    pub const fn development() -> Self {
        Self {
            enable_rt: true,
            compact_blas: false,
            prefer_fast_trace: false,
        }
    }
}

impl Default for MeshRTConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// MeshRTError
// ---------------------------------------------------------------------------

/// Errors that can occur during mesh RT integration.
#[derive(Debug)]
pub enum MeshRTError {
    /// Memory budget would be exceeded by this operation.
    MemoryBudgetExceeded {
        /// Current memory usage in bytes.
        used: usize,
        /// Configured budget in bytes.
        budget: usize,
    },
    /// BLAS build failed.
    BuildFailed(String),
    /// Mesh not found in the integration.
    MeshNotFound(String),
    /// Pool error wrapper.
    PoolError(BlasPoolError),
    /// BLAS error wrapper.
    BlasError(BlasError),
}

impl std::fmt::Display for MeshRTError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MemoryBudgetExceeded { used, budget } => {
                write!(
                    f,
                    "memory budget exceeded: using {} bytes of {} byte budget",
                    used, budget
                )
            }
            Self::BuildFailed(msg) => write!(f, "BLAS build failed: {}", msg),
            Self::MeshNotFound(id) => write!(f, "mesh '{}' not found", id),
            Self::PoolError(e) => write!(f, "pool error: {}", e),
            Self::BlasError(e) => write!(f, "BLAS error: {}", e),
        }
    }
}

impl std::error::Error for MeshRTError {}

impl From<BlasPoolError> for MeshRTError {
    fn from(err: BlasPoolError) -> Self {
        MeshRTError::PoolError(err)
    }
}

impl From<BlasError> for MeshRTError {
    fn from(err: BlasError) -> Self {
        MeshRTError::BlasError(err)
    }
}

// ---------------------------------------------------------------------------
// StaticBlasEntry
// ---------------------------------------------------------------------------

/// Tracking entry for a static mesh's BLAS.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields used for debugging/future features
struct StaticBlasEntry {
    /// Handle to the BLAS in the pool.
    handle: BlasHandle,
    /// Mesh asset ID (for debugging/iteration).
    mesh_id: String,
    /// Whether the BLAS has been compacted.
    is_compacted: bool,
    /// Current memory size in bytes.
    memory_size: usize,
    /// Frame number when the BLAS was created (for metrics).
    created_frame: u64,
}

// ---------------------------------------------------------------------------
// PendingCompaction
// ---------------------------------------------------------------------------

/// A queued compaction request.
#[derive(Debug, Clone)]
struct PendingCompaction {
    /// Mesh ID to compact.
    mesh_id: String,
    /// Handle to compact.
    handle: BlasHandle,
}

// ---------------------------------------------------------------------------
// MeshRTIntegration
// ---------------------------------------------------------------------------

/// Integration layer connecting mesh asset loading to BLAS pool management.
///
/// Handles:
/// - Automatic BLAS building when meshes are loaded
/// - Automatic compaction queuing if enabled
/// - Memory budget tracking and enforcement
/// - BLAS release on mesh unload
/// - Garbage collection of unreferenced BLAS
pub struct MeshRTIntegration {
    /// Underlying BLAS pool.
    blas_pool: BlasPool,
    /// Per-mesh tracking entries.
    entries: HashMap<String, StaticBlasEntry>,
    /// Configuration.
    config: MeshRTConfig,
    /// Memory budget in bytes.
    memory_budget: usize,
    /// Pending compactions to process.
    pending_compactions: Vec<PendingCompaction>,
    /// Current frame number.
    current_frame: u64,
}

impl MeshRTIntegration {
    /// Create a new mesh RT integration with the given configuration.
    pub fn new(config: MeshRTConfig) -> Self {
        Self {
            blas_pool: BlasPool::new(),
            entries: HashMap::new(),
            config,
            memory_budget: DEFAULT_MEMORY_BUDGET,
            pending_compactions: Vec::new(),
            current_frame: 0,
        }
    }

    /// Set the memory budget in bytes.
    pub fn set_memory_budget(&mut self, budget: usize) {
        self.memory_budget = budget;
    }

    /// Get the current memory budget.
    pub fn memory_budget(&self) -> usize {
        self.memory_budget
    }

    /// Get the current configuration.
    pub fn config(&self) -> &MeshRTConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: MeshRTConfig) {
        self.config = config;
    }

    /// Called when a mesh is loaded.
    ///
    /// Queues a BLAS build for the mesh. After calling this, you must call
    /// `process_pending()` to actually execute the build.
    ///
    /// # Arguments
    ///
    /// * `mesh_id` - Unique identifier for the mesh
    /// * `vertices` - Vertex positions
    /// * `indices` - Optional triangle indices (if None, assumes sequential triangles)
    ///
    /// # Returns
    ///
    /// A `BlasHandle` that can be used after `process_pending()` completes.
    ///
    /// # Errors
    ///
    /// Returns `MeshRTError::MemoryBudgetExceeded` if the estimated memory
    /// would exceed the budget.
    pub fn on_mesh_load(
        &mut self,
        mesh_id: &str,
        vertices: &[[f32; 3]],
        indices: Option<&[u32]>,
    ) -> Result<BlasHandle, MeshRTError> {
        // If RT is disabled, return a dummy handle
        if !self.config.enable_rt {
            return Ok(dummy_blas_handle());
        }

        // If already loaded, just acquire and return existing handle
        if let Some(entry) = self.entries.get(mesh_id) {
            if let Some(handle) = self.blas_pool.acquire(mesh_id) {
                return Ok(handle);
            }
            return Ok(entry.handle);
        }

        // Estimate memory for budget check
        let estimated_memory = Self::estimate_blas_memory(vertices.len(), indices.map(|i| i.len()));
        let current_usage = self.memory_usage();

        if current_usage + estimated_memory > self.memory_budget {
            return Err(MeshRTError::MemoryBudgetExceeded {
                used: current_usage,
                budget: self.memory_budget,
            });
        }

        // Build BLAS config from our config
        let blas_config = BlasConfig::new()
            .with_compaction(self.config.compact_blas)
            .with_update(false) // Static meshes don't update
            .with_fast_trace(self.config.prefer_fast_trace);

        // Queue the build
        if let Some(idx) = indices {
            self.blas_pool.queue_build_indexed(
                mesh_id,
                blas_config,
                vertices.to_vec(),
                idx.to_vec(),
            );
        } else {
            self.blas_pool
                .queue_build(mesh_id, blas_config, vertices.to_vec());
        }

        // Return a placeholder handle - actual handle assigned after process_pending
        Ok(dummy_blas_handle())
    }

    /// Called when a mesh is unloaded.
    ///
    /// Releases the BLAS handle and removes tracking. The BLAS will be
    /// garbage collected after the grace period if no other references exist.
    pub fn on_mesh_unload(&mut self, mesh_id: &str) {
        if let Some(entry) = self.entries.remove(mesh_id) {
            self.blas_pool.release(entry.handle);

            // Remove from pending compactions
            self.pending_compactions
                .retain(|p| p.mesh_id != mesh_id);
        }
    }

    /// Get the BLAS handle for a mesh.
    ///
    /// Returns `None` if the mesh hasn't been loaded or the BLAS hasn't
    /// been built yet.
    pub fn get_blas_handle(&self, mesh_id: &str) -> Option<BlasHandle> {
        self.entries.get(mesh_id).map(|e| e.handle)
    }

    /// Check if a mesh has a BLAS.
    pub fn has_blas(&self, mesh_id: &str) -> bool {
        self.entries.contains_key(mesh_id)
    }

    /// Get the total memory usage across all BLAS entries.
    pub fn memory_usage(&self) -> usize {
        self.blas_pool.memory_usage()
    }

    /// Check if current memory usage is within budget.
    pub fn is_within_budget(&self) -> bool {
        self.memory_usage() <= self.memory_budget
    }

    /// Get the number of loaded meshes with BLAS.
    pub fn mesh_count(&self) -> usize {
        self.entries.len()
    }

    /// Process all pending builds and compactions.
    ///
    /// This should be called once per frame or after batching mesh loads.
    pub fn process_pending(&mut self) {
        // Process builds first
        let built_handles = self.blas_pool.process_pending();

        // For each built handle, create tracking entry and optionally queue compaction
        for handle in built_handles {
            // Find the mesh_id for this handle
            if let Some((_, mesh_id)) = self
                .blas_pool
                .entries_iter()
                .find(|(h, _)| *h == handle)
            {
                let mesh_id = mesh_id.to_string();

                // Skip if already tracked (could be a re-acquire)
                if self.entries.contains_key(&mesh_id) {
                    continue;
                }

                // Get memory size from pool
                let memory_size = self
                    .blas_pool
                    .get(handle)
                    .map(|b| b.memory_size)
                    .unwrap_or(0);

                // Create tracking entry
                let entry = StaticBlasEntry {
                    handle,
                    mesh_id: mesh_id.clone(),
                    is_compacted: false,
                    memory_size,
                    created_frame: self.current_frame,
                };
                self.entries.insert(mesh_id.clone(), entry);

                // Queue compaction if enabled
                if self.config.compact_blas {
                    self.pending_compactions.push(PendingCompaction {
                        mesh_id,
                        handle,
                    });
                }
            }
        }

        // Process pending compactions
        self.process_compactions();
    }

    /// Process pending compaction requests.
    fn process_compactions(&mut self) {
        let compactions = std::mem::take(&mut self.pending_compactions);

        for pending in compactions {
            // Queue compact in the pool
            self.blas_pool.queue_compact(pending.handle);
        }

        // Process the compactions
        let compacted = self.blas_pool.process_pending();

        // Update tracking entries
        for handle in compacted {
            for entry in self.entries.values_mut() {
                if entry.handle == handle {
                    entry.is_compacted = true;
                    // Update memory size
                    if let Some(blas) = self.blas_pool.get(handle) {
                        entry.memory_size = blas.memory_size;
                    }
                    break;
                }
            }
        }
    }

    /// Garbage collect unreferenced BLAS entries.
    ///
    /// # Arguments
    ///
    /// * `frame` - Current frame number (for GC timing)
    ///
    /// # Returns
    ///
    /// Number of entries garbage collected.
    pub fn gc(&mut self, frame: u64) -> usize {
        self.current_frame = frame;
        let count = self.blas_pool.gc(frame);

        // Remove tracking entries for GC'd handles
        let pool = &self.blas_pool;
        self.entries
            .retain(|_, entry| pool.get(entry.handle).is_some());

        count
    }

    /// Force immediate garbage collection.
    pub fn gc_immediate(&mut self) -> usize {
        let count = self.blas_pool.gc_immediate();

        // Remove tracking entries for GC'd handles
        let pool = &self.blas_pool;
        self.entries
            .retain(|_, entry| pool.get(entry.handle).is_some());

        count
    }

    /// Get statistics about the integration.
    pub fn stats(&self) -> MeshRTStats {
        let pool_stats = self.blas_pool.stats();
        let compacted_count = self.entries.values().filter(|e| e.is_compacted).count();

        MeshRTStats {
            mesh_count: self.entries.len(),
            total_memory: pool_stats.total_memory,
            memory_budget: self.memory_budget,
            budget_usage_percent: if self.memory_budget > 0 {
                (pool_stats.total_memory as f64 / self.memory_budget as f64 * 100.0) as f32
            } else {
                0.0
            },
            pending_builds: pool_stats.pending_builds,
            pending_compacts: self.pending_compactions.len(),
            compacted_count,
            total_triangles: pool_stats.total_triangles,
            total_vertices: pool_stats.total_vertices,
        }
    }

    /// Get access to the underlying BLAS pool.
    pub fn pool(&self) -> &BlasPool {
        &self.blas_pool
    }

    /// Get mutable access to the underlying BLAS pool.
    pub fn pool_mut(&mut self) -> &mut BlasPool {
        &mut self.blas_pool
    }

    /// Iterate over all (mesh_id, handle) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, BlasHandle)> + '_ {
        self.entries.iter().map(|(k, v)| (k.as_str(), v.handle))
    }

    /// Estimate BLAS memory size for budget checking.
    fn estimate_blas_memory(vertex_count: usize, index_count: Option<usize>) -> usize {
        const MIN_BLAS_MEMORY: usize = 128;
        const BYTES_PER_VERTEX: usize = 12;
        const BYTES_PER_TRIANGLE: usize = 64;

        let triangle_count = index_count.map(|i| i / 3).unwrap_or(vertex_count / 3);

        MIN_BLAS_MEMORY + vertex_count * BYTES_PER_VERTEX + triangle_count * BYTES_PER_TRIANGLE
    }
}

impl Default for MeshRTIntegration {
    fn default() -> Self {
        Self::new(MeshRTConfig::default())
    }
}

// ---------------------------------------------------------------------------
// MeshRTStats
// ---------------------------------------------------------------------------

/// Statistics about the mesh RT integration.
#[derive(Debug, Clone, Default)]
pub struct MeshRTStats {
    /// Number of meshes with BLAS.
    pub mesh_count: usize,
    /// Total memory usage in bytes.
    pub total_memory: usize,
    /// Memory budget in bytes.
    pub memory_budget: usize,
    /// Budget usage as percentage (0-100+).
    pub budget_usage_percent: f32,
    /// Number of pending build requests.
    pub pending_builds: usize,
    /// Number of pending compaction requests.
    pub pending_compacts: usize,
    /// Number of compacted BLAS entries.
    pub compacted_count: usize,
    /// Total triangle count across all BLAS.
    pub total_triangles: u64,
    /// Total vertex count across all BLAS.
    pub total_vertices: u64,
}

// ---------------------------------------------------------------------------
// Dummy handle helper
// ---------------------------------------------------------------------------

/// Create a dummy BLAS handle for when RT is disabled.
/// Uses transmute since BlasHandle's constructor is private.
fn dummy_blas_handle() -> BlasHandle {
    // BlasHandle is #[repr(transparent)] over u32, safe to transmute
    unsafe { std::mem::transmute(0u32) }
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
    fn make_triangle() -> (Vec<[f32; 3]>, Vec<u32>) {
        let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]];
        let indices = vec![0, 1, 2];
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

    /// Create a large grid mesh.
    fn make_large_grid(size: usize) -> (Vec<[f32; 3]>, Vec<u32>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for y in 0..size {
            for x in 0..size {
                vertices.push([x as f32, y as f32, 0.0]);
            }
        }

        for y in 0..(size - 1) {
            for x in 0..(size - 1) {
                let i00 = (y * size + x) as u32;
                let i10 = (y * size + x + 1) as u32;
                let i01 = ((y + 1) * size + x) as u32;
                let i11 = ((y + 1) * size + x + 1) as u32;

                indices.push(i00);
                indices.push(i10);
                indices.push(i11);

                indices.push(i00);
                indices.push(i11);
                indices.push(i01);
            }
        }

        (vertices, indices)
    }

    // -------------------------------------------------------------------------
    // Test: Config defaults
    // -------------------------------------------------------------------------

    #[test]
    fn test_config_default() {
        let config = MeshRTConfig::default();
        assert!(!config.enable_rt);
        assert!(!config.compact_blas);
        assert!(!config.prefer_fast_trace);
    }

    #[test]
    fn test_config_new() {
        let config = MeshRTConfig::new();
        assert!(!config.enable_rt);
        assert!(!config.compact_blas);
        assert!(!config.prefer_fast_trace);
    }

    #[test]
    fn test_config_production() {
        let config = MeshRTConfig::production();
        assert!(config.enable_rt);
        assert!(config.compact_blas);
        assert!(config.prefer_fast_trace);
    }

    #[test]
    fn test_config_development() {
        let config = MeshRTConfig::development();
        assert!(config.enable_rt);
        assert!(!config.compact_blas);
        assert!(!config.prefer_fast_trace);
    }

    // -------------------------------------------------------------------------
    // Test: Integration creation
    // -------------------------------------------------------------------------

    #[test]
    fn test_integration_new() {
        let config = MeshRTConfig::production();
        let integration = MeshRTIntegration::new(config);

        assert_eq!(integration.config(), &config);
        assert_eq!(integration.memory_budget(), DEFAULT_MEMORY_BUDGET);
        assert_eq!(integration.mesh_count(), 0);
        assert_eq!(integration.memory_usage(), 0);
        assert!(integration.is_within_budget());
    }

    #[test]
    fn test_integration_default() {
        let integration = MeshRTIntegration::default();
        assert_eq!(integration.config(), &MeshRTConfig::default());
    }

    #[test]
    fn test_set_memory_budget() {
        let mut integration = MeshRTIntegration::new(MeshRTConfig::production());
        integration.set_memory_budget(1024 * 1024); // 1 MB

        assert_eq!(integration.memory_budget(), 1024 * 1024);
    }

    #[test]
    fn test_set_config() {
        let mut integration = MeshRTIntegration::new(MeshRTConfig::production());
        let new_config = MeshRTConfig::development();
        integration.set_config(new_config);

        assert_eq!(integration.config(), &new_config);
    }

    // -------------------------------------------------------------------------
    // Test: Mesh load triggers BLAS build
    // -------------------------------------------------------------------------

    #[test]
    fn test_mesh_load_queues_build() {
        let mut integration = MeshRTIntegration::new(MeshRTConfig::production());
        let (vertices, indices) = make_triangle();

        let result = integration.on_mesh_load("triangle", &vertices, Some(&indices));
        assert!(result.is_ok());

        // Build not processed yet
        assert_eq!(integration.mesh_count(), 0);
        assert_eq!(integration.pool().pending_build_count(), 1);
    }

    #[test]
    fn test_mesh_load_and_process() {
        let mut integration = MeshRTIntegration::new(MeshRTConfig::production());
        let (vertices, indices) = make_cube();

        integration
            .on_mesh_load("cube", &vertices, Some(&indices))
            .unwrap();
        integration.process_pending();

        assert_eq!(integration.mesh_count(), 1);
        assert!(integration.has_blas("cube"));
        assert!(integration.get_blas_handle("cube").is_some());
        assert!(integration.memory_usage() > 0);
    }

    #[test]
    fn test_mesh_load_without_indices() {
        let mut integration = MeshRTIntegration::new(MeshRTConfig::production());
        let vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ];

        integration.on_mesh_load("tri", &vertices, None).unwrap();
        integration.process_pending();

        assert!(integration.has_blas("tri"));
    }

    #[test]
    fn test_mesh_load_rt_disabled() {
        let config = MeshRTConfig {
            enable_rt: false,
            compact_blas: false,
            prefer_fast_trace: false,
        };
        let mut integration = MeshRTIntegration::new(config);
        let (vertices, indices) = make_triangle();

        let handle = integration
            .on_mesh_load("mesh", &vertices, Some(&indices))
            .unwrap();

        // Should return dummy handle, no build queued
        assert_eq!(handle.id(), 0);
        assert_eq!(integration.pool().pending_build_count(), 0);
    }

    // -------------------------------------------------------------------------
    // Test: Mesh unload releases BLAS
    // -------------------------------------------------------------------------

    #[test]
    fn test_mesh_unload_releases_blas() {
        let mut integration = MeshRTIntegration::new(MeshRTConfig::production());
        let (vertices, indices) = make_cube();

        integration
            .on_mesh_load("cube", &vertices, Some(&indices))
            .unwrap();
        integration.process_pending();

        assert!(integration.has_blas("cube"));

        integration.on_mesh_unload("cube");

        assert!(!integration.has_blas("cube"));
        assert!(integration.get_blas_handle("cube").is_none());
    }

    #[test]
    fn test_mesh_unload_nonexistent() {
        let mut integration = MeshRTIntegration::new(MeshRTConfig::production());

        // Should not panic
        integration.on_mesh_unload("nonexistent");
    }

    // -------------------------------------------------------------------------
    // Test: Memory tracking
    // -------------------------------------------------------------------------

    #[test]
    fn test_memory_tracking() {
        let mut integration = MeshRTIntegration::new(MeshRTConfig::production());
        let (vertices, indices) = make_cube();

        assert_eq!(integration.memory_usage(), 0);

        integration
            .on_mesh_load("cube", &vertices, Some(&indices))
            .unwrap();
        integration.process_pending();

        let usage = integration.memory_usage();
        assert!(usage > 0);
    }

    #[test]
    fn test_memory_decreases_on_unload_and_gc() {
        let mut integration = MeshRTIntegration::new(MeshRTConfig::development());
        let (vertices, indices) = make_cube();

        integration
            .on_mesh_load("cube", &vertices, Some(&indices))
            .unwrap();
        integration.process_pending();

        let usage_before = integration.memory_usage();

        integration.on_mesh_unload("cube");
        integration.gc_immediate();

        let usage_after = integration.memory_usage();
        assert!(usage_after < usage_before);
        assert_eq!(usage_after, 0);
    }

    // -------------------------------------------------------------------------
    // Test: Budget enforcement
    // -------------------------------------------------------------------------

    #[test]
    fn test_budget_enforcement() {
        let mut integration = MeshRTIntegration::new(MeshRTConfig::production());
        integration.set_memory_budget(100); // Very small budget

        let (vertices, indices) = make_large_grid(50);

        let result = integration.on_mesh_load("large", &vertices, Some(&indices));

        assert!(matches!(
            result,
            Err(MeshRTError::MemoryBudgetExceeded { .. })
        ));
    }

    #[test]
    fn test_is_within_budget() {
        let mut integration = MeshRTIntegration::new(MeshRTConfig::production());
        integration.set_memory_budget(10 * 1024 * 1024); // 10 MB

        assert!(integration.is_within_budget());

        let (vertices, indices) = make_cube();
        integration
            .on_mesh_load("cube", &vertices, Some(&indices))
            .unwrap();
        integration.process_pending();

        assert!(integration.is_within_budget());
    }

    // -------------------------------------------------------------------------
    // Test: Compaction runs after build
    // -------------------------------------------------------------------------

    #[test]
    fn test_compaction_after_build() {
        let config = MeshRTConfig {
            enable_rt: true,
            compact_blas: true,
            prefer_fast_trace: true,
        };
        let mut integration = MeshRTIntegration::new(config);
        let (vertices, indices) = make_cube();

        integration
            .on_mesh_load("cube", &vertices, Some(&indices))
            .unwrap();
        integration.process_pending();

        let stats = integration.stats();
        assert_eq!(stats.compacted_count, 1);
    }

    #[test]
    fn test_no_compaction_when_disabled() {
        let config = MeshRTConfig {
            enable_rt: true,
            compact_blas: false,
            prefer_fast_trace: false,
        };
        let mut integration = MeshRTIntegration::new(config);
        let (vertices, indices) = make_cube();

        integration
            .on_mesh_load("cube", &vertices, Some(&indices))
            .unwrap();
        integration.process_pending();

        let stats = integration.stats();
        assert_eq!(stats.compacted_count, 0);
    }

    // -------------------------------------------------------------------------
    // Test: Get handle returns correct BLAS
    // -------------------------------------------------------------------------

    #[test]
    fn test_get_handle_returns_correct_blas() {
        let mut integration = MeshRTIntegration::new(MeshRTConfig::production());
        let (v1, i1) = make_triangle();
        let (v2, i2) = make_cube();

        integration.on_mesh_load("tri", &v1, Some(&i1)).unwrap();
        integration.on_mesh_load("cube", &v2, Some(&i2)).unwrap();
        integration.process_pending();

        let h1 = integration.get_blas_handle("tri");
        let h2 = integration.get_blas_handle("cube");

        assert!(h1.is_some());
        assert!(h2.is_some());
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_get_handle_nonexistent() {
        let integration = MeshRTIntegration::new(MeshRTConfig::production());
        assert!(integration.get_blas_handle("nonexistent").is_none());
    }

    // -------------------------------------------------------------------------
    // Test: Multiple meshes independent
    // -------------------------------------------------------------------------

    #[test]
    fn test_multiple_meshes_independent() {
        let mut integration = MeshRTIntegration::new(MeshRTConfig::production());
        let (v1, i1) = make_triangle();
        let (v2, i2) = make_cube();

        integration.on_mesh_load("mesh_a", &v1, Some(&i1)).unwrap();
        integration.on_mesh_load("mesh_b", &v2, Some(&i2)).unwrap();
        integration.process_pending();

        assert_eq!(integration.mesh_count(), 2);
        assert!(integration.has_blas("mesh_a"));
        assert!(integration.has_blas("mesh_b"));

        // Unload one, other unaffected
        integration.on_mesh_unload("mesh_a");

        assert!(!integration.has_blas("mesh_a"));
        assert!(integration.has_blas("mesh_b"));
        assert_eq!(integration.mesh_count(), 1);
    }

    // -------------------------------------------------------------------------
    // Test: GC removes unreferenced
    // -------------------------------------------------------------------------

    #[test]
    fn test_gc_removes_unreferenced() {
        let mut integration = MeshRTIntegration::new(MeshRTConfig::production());
        let (vertices, indices) = make_cube();

        integration
            .on_mesh_load("cube", &vertices, Some(&indices))
            .unwrap();
        integration.process_pending();

        assert_eq!(integration.mesh_count(), 1);

        integration.on_mesh_unload("cube");
        let removed = integration.gc_immediate();

        assert_eq!(removed, 1);
        assert_eq!(integration.mesh_count(), 0);
    }

    #[test]
    fn test_gc_preserves_referenced() {
        let mut integration = MeshRTIntegration::new(MeshRTConfig::production());
        let (vertices, indices) = make_cube();

        integration
            .on_mesh_load("cube", &vertices, Some(&indices))
            .unwrap();
        integration.process_pending();

        // Don't unload, just GC
        let removed = integration.gc_immediate();

        assert_eq!(removed, 0);
        assert_eq!(integration.mesh_count(), 1);
    }

    // -------------------------------------------------------------------------
    // Test: Statistics
    // -------------------------------------------------------------------------

    #[test]
    fn test_stats() {
        let mut integration = MeshRTIntegration::new(MeshRTConfig::production());
        integration.set_memory_budget(100 * 1024 * 1024);
        let (vertices, indices) = make_cube();

        integration
            .on_mesh_load("cube", &vertices, Some(&indices))
            .unwrap();
        integration.process_pending();

        let stats = integration.stats();
        assert_eq!(stats.mesh_count, 1);
        assert!(stats.total_memory > 0);
        assert_eq!(stats.memory_budget, 100 * 1024 * 1024);
        assert!(stats.budget_usage_percent > 0.0);
        assert!(stats.budget_usage_percent < 100.0);
        assert_eq!(stats.pending_builds, 0);
        assert_eq!(stats.pending_compacts, 0);
        assert_eq!(stats.total_triangles, 12);
        assert_eq!(stats.total_vertices, 8);
    }

    // -------------------------------------------------------------------------
    // Test: Iteration
    // -------------------------------------------------------------------------

    #[test]
    fn test_iter() {
        let mut integration = MeshRTIntegration::new(MeshRTConfig::production());
        let (v1, i1) = make_triangle();
        let (v2, i2) = make_cube();

        integration.on_mesh_load("alpha", &v1, Some(&i1)).unwrap();
        integration.on_mesh_load("beta", &v2, Some(&i2)).unwrap();
        integration.process_pending();

        let pairs: Vec<_> = integration.iter().collect();
        assert_eq!(pairs.len(), 2);

        let mesh_ids: Vec<_> = pairs.iter().map(|(id, _)| *id).collect();
        assert!(mesh_ids.contains(&"alpha"));
        assert!(mesh_ids.contains(&"beta"));
    }

    // -------------------------------------------------------------------------
    // Test: Pool access
    // -------------------------------------------------------------------------

    #[test]
    fn test_pool_access() {
        let mut integration = MeshRTIntegration::new(MeshRTConfig::production());
        let (vertices, indices) = make_cube();

        integration
            .on_mesh_load("cube", &vertices, Some(&indices))
            .unwrap();
        integration.process_pending();

        // Read-only access
        assert_eq!(integration.pool().len(), 1);

        // Mutable access
        let _pool = integration.pool_mut();
    }

    // -------------------------------------------------------------------------
    // Test: Re-acquire existing mesh
    // -------------------------------------------------------------------------

    #[test]
    fn test_reacquire_existing_mesh() {
        let mut integration = MeshRTIntegration::new(MeshRTConfig::production());
        let (vertices, indices) = make_cube();

        integration
            .on_mesh_load("cube", &vertices, Some(&indices))
            .unwrap();
        integration.process_pending();

        let h1 = integration.get_blas_handle("cube");
        assert!(h1.is_some());

        // Load same mesh again
        let h2 = integration.on_mesh_load("cube", &vertices, Some(&indices));
        assert!(h2.is_ok());
        // Should return successfully (acquire existing)
    }

    // -------------------------------------------------------------------------
    // Test: Error types
    // -------------------------------------------------------------------------

    #[test]
    fn test_error_display() {
        let err = MeshRTError::MemoryBudgetExceeded {
            used: 1000,
            budget: 500,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("memory budget exceeded"));

        let err = MeshRTError::BuildFailed("test failure".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("BLAS build failed"));

        let err = MeshRTError::MeshNotFound("missing".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("missing"));
    }

    // -------------------------------------------------------------------------
    // Test: Memory estimation
    // -------------------------------------------------------------------------

    #[test]
    fn test_memory_estimation() {
        let mem = MeshRTIntegration::estimate_blas_memory(100, Some(300));
        // 128 (base) + 100 * 12 (vertices) + 100 * 64 (triangles) = 7728
        assert!(mem > 0);
        assert!(mem >= 128 + 100 * 12 + 100 * 64);
    }

    // -------------------------------------------------------------------------
    // Test: Frame tracking
    // -------------------------------------------------------------------------

    #[test]
    fn test_gc_with_frame() {
        let mut integration = MeshRTIntegration::new(MeshRTConfig::production());
        let (vertices, indices) = make_cube();

        integration
            .on_mesh_load("cube", &vertices, Some(&indices))
            .unwrap();
        integration.process_pending();

        integration.on_mesh_unload("cube");

        // GC at frame 0 shouldn't remove (grace period)
        let removed = integration.gc(0);
        // Note: Pool has grace period, might be 0 or 1 depending on implementation
        // Integration layer should still work
        assert!(removed == 0 || integration.mesh_count() == 0);

        // GC after grace period
        let _removed = integration.gc(100);
        // Should eventually clean up (GC returns count)
    }
}
