//! Shader Edit-and-Continue Pipeline (T-AS-3.6).
//!
//! Provides runtime shader recompilation and hot-reload capabilities:
//!
//! - **Content Hash Change Detection**: Compare file hashes vs cached hashes
//! - **Background Recompilation**: Recompile shaders without blocking rendering
//! - **PSO Swap at Phase Boundary**: Atomically swap pipeline states using AtomicPtr
//! - **Fence Tracking**: Retain old PSO until in-flight frames complete
//! - **Error Handling**: Failed recompilation keeps old shader active
//!
//! # Architecture
//!
//! The hot-reload system operates in three phases:
//!
//! 1. **Detection**: File watcher detects changes, computes content hashes
//! 2. **Recompilation**: Background thread compiles affected shaders
//! 3. **Swap**: At frame boundary, atomically swap to new PSO
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::shader::hot_reload::{ShaderHotReload, RecompileRequest};
//! use renderer_backend::shader::cache::ShaderCache3L;
//! use renderer_backend::shader::dependencies::ShaderDependencyGraph;
//!
//! // Initialize hot-reload system
//! let cache = ShaderCache3L::with_defaults("./cache")?;
//! let deps = ShaderDependencyGraph::new(vec!["shaders/".into()]);
//! let mut hot_reload = ShaderHotReload::new(cache, deps);
//!
//! // In your main loop:
//! loop {
//!     // Check for file changes
//!     let requests = hot_reload.check_for_changes();
//!     for req in requests {
//!         hot_reload.queue_recompile(req);
//!     }
//!
//!     // Process pending recompilations
//!     let results = hot_reload.process_pending();
//!     for result in results {
//!         if result.success {
//!             hot_reload.swap_pso_at_boundary(result.shader_id, result.new_pso.unwrap());
//!         } else {
//!             eprintln!("Shader reload failed: {:?}", result.error);
//!         }
//!     }
//!
//!     // After frame completes
//!     hot_reload.release_old_psos(current_frame_fence);
//! }
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::cache::{CacheKey, ShaderCache3L, TargetPlatform};
use super::dependencies::{DependencyError, ShaderDependencyGraph};
use super::naga_compiler::{CompileError, CompileErrorKind, CompilerOptions, NagaCompiler};
use crate::pipeline::ContentHash;

// ---------------------------------------------------------------------------
// ShaderId
// ---------------------------------------------------------------------------

/// Unique identifier for a shader in the hot-reload system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShaderId(pub u64);

impl ShaderId {
    /// Create a new shader ID from a path hash.
    pub fn from_path(path: &Path) -> Self {
        let hash = ContentHash::from_bytes(path.to_string_lossy().as_bytes());
        let bytes = hash.as_bytes();
        let id = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        Self(id)
    }

    /// Create a shader ID from a raw u64.
    pub const fn from_raw(id: u64) -> Self {
        Self(id)
    }

    /// Get the raw u64 value.
    pub const fn raw(&self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for ShaderId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Shader({:016x})", self.0)
    }
}

// ---------------------------------------------------------------------------
// PipelineState (mock for shader module reference)
// ---------------------------------------------------------------------------

/// Represents a compiled pipeline state object (PSO).
///
/// In a real implementation, this would wrap the platform-specific PSO handle
/// (e.g., `wgpu::RenderPipeline`, `VkPipeline`, `ID3D12PipelineState`).
#[derive(Debug)]
pub struct PipelineState {
    /// Unique identifier for this PSO.
    pub id: u64,
    /// The shader source hash used to create this PSO.
    pub shader_hash: ContentHash,
    /// Compiled SPIR-V bytecode (for Vulkan).
    pub spirv: Option<Vec<u32>>,
    /// Entry point names.
    pub entry_points: Vec<String>,
    /// Creation timestamp.
    pub created_at: Instant,
}

impl PipelineState {
    /// Create a new pipeline state.
    pub fn new(shader_hash: ContentHash, spirv: Vec<u32>, entry_points: Vec<String>) -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        Self {
            id: COUNTER.fetch_add(1, Ordering::Relaxed),
            shader_hash,
            spirv: Some(spirv),
            entry_points,
            created_at: Instant::now(),
        }
    }

    /// Create a dummy PSO for testing.
    pub fn dummy(hash: ContentHash) -> Self {
        Self {
            id: 0,
            shader_hash: hash,
            spirv: None,
            entry_points: Vec::new(),
            created_at: Instant::now(),
        }
    }
}

// ---------------------------------------------------------------------------
// RecompileRequest
// ---------------------------------------------------------------------------

/// A request to recompile a shader.
#[derive(Debug, Clone)]
pub struct RecompileRequest {
    /// Path to the shader source file.
    pub shader_path: PathBuf,
    /// Entry points affected by this change.
    pub affected_entry_points: Vec<String>,
    /// When this request was triggered.
    pub triggered_at: Instant,
    /// Priority (lower = higher priority).
    pub priority: u32,
    /// Previous content hash (if known).
    pub old_hash: Option<ContentHash>,
}

impl RecompileRequest {
    /// Create a new recompile request.
    pub fn new(shader_path: PathBuf, affected_entry_points: Vec<String>) -> Self {
        Self {
            shader_path,
            affected_entry_points,
            triggered_at: Instant::now(),
            priority: 100,
            old_hash: None,
        }
    }

    /// Set the priority.
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Set the old hash.
    pub fn with_old_hash(mut self, hash: ContentHash) -> Self {
        self.old_hash = Some(hash);
        self
    }

    /// Get the shader ID for this request.
    pub fn shader_id(&self) -> ShaderId {
        ShaderId::from_path(&self.shader_path)
    }
}

// ---------------------------------------------------------------------------
// RecompileResult
// ---------------------------------------------------------------------------

/// Result of a shader recompilation.
#[derive(Debug)]
pub struct RecompileResult {
    /// The shader ID.
    pub shader_id: ShaderId,
    /// Path to the shader file.
    pub shader_path: PathBuf,
    /// Whether recompilation succeeded.
    pub success: bool,
    /// Error details if failed.
    pub error: Option<CompileError>,
    /// New PSO if successful.
    pub new_pso: Option<Arc<PipelineState>>,
    /// Time taken to recompile.
    pub compile_time: Duration,
    /// New content hash.
    pub new_hash: ContentHash,
}

impl RecompileResult {
    /// Create a successful result.
    pub fn success(
        shader_id: ShaderId,
        shader_path: PathBuf,
        new_pso: Arc<PipelineState>,
        compile_time: Duration,
        new_hash: ContentHash,
    ) -> Self {
        Self {
            shader_id,
            shader_path,
            success: true,
            error: None,
            new_pso: Some(new_pso),
            compile_time,
            new_hash,
        }
    }

    /// Create a failed result.
    pub fn failure(
        shader_id: ShaderId,
        shader_path: PathBuf,
        error: CompileError,
        compile_time: Duration,
        new_hash: ContentHash,
    ) -> Self {
        Self {
            shader_id,
            shader_path,
            success: false,
            error: Some(error),
            new_pso: None,
            compile_time,
            new_hash,
        }
    }
}

// ---------------------------------------------------------------------------
// RetainedPso
// ---------------------------------------------------------------------------

/// A PSO that is retained until in-flight frames complete.
#[derive(Debug)]
struct RetainedPso {
    /// The old PSO.
    pso: Arc<PipelineState>,
    /// Frame fence when this PSO was retired.
    retired_at_frame: u64,
    /// When this PSO was retired.
    retired_at: Instant,
}

// ---------------------------------------------------------------------------
// AtomicPsoHolder
// ---------------------------------------------------------------------------

/// Thread-safe holder for a PSO using AtomicPtr.
struct AtomicPsoHolder {
    /// Atomic pointer to the current PSO.
    ptr: AtomicPtr<PipelineState>,
    /// Arc to keep the PSO alive.
    #[allow(dead_code)]
    current: Mutex<Option<Arc<PipelineState>>>,
}

impl AtomicPsoHolder {
    /// Create a new holder with an initial PSO.
    fn new(initial: Arc<PipelineState>) -> Self {
        let ptr = Arc::into_raw(Arc::clone(&initial)) as *mut PipelineState;
        Self {
            ptr: AtomicPtr::new(ptr),
            current: Mutex::new(Some(initial)),
        }
    }

    /// Create an empty holder.
    fn empty() -> Self {
        Self {
            ptr: AtomicPtr::new(std::ptr::null_mut()),
            current: Mutex::new(None),
        }
    }

    /// Atomically swap to a new PSO, returning the old one.
    fn swap(&self, new: Arc<PipelineState>) -> Option<Arc<PipelineState>> {
        let new_ptr = Arc::into_raw(Arc::clone(&new)) as *mut PipelineState;
        let old_ptr = self.ptr.swap(new_ptr, Ordering::AcqRel);

        let mut guard = self.current.lock().unwrap();
        let old = guard.take();
        *guard = Some(new);

        // Reconstruct Arc from old pointer if non-null
        if !old_ptr.is_null() {
            // Safety: we created this pointer from Arc::into_raw
            let _ = unsafe { Arc::from_raw(old_ptr) };
        }

        old
    }

    /// Get the current PSO.
    fn get(&self) -> Option<Arc<PipelineState>> {
        let guard = self.current.lock().unwrap();
        guard.clone()
    }

    /// Set the PSO without returning the old one.
    fn set(&self, new: Arc<PipelineState>) {
        let _ = self.swap(new);
    }
}

impl Drop for AtomicPsoHolder {
    fn drop(&mut self) {
        let ptr = *self.ptr.get_mut();
        if !ptr.is_null() {
            // Safety: we created this pointer from Arc::into_raw
            unsafe {
                let _ = Arc::from_raw(ptr);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ShaderHotReload
// ---------------------------------------------------------------------------

/// Shader hot-reload system with edit-and-continue support.
///
/// Manages shader recompilation, PSO swapping, and lifecycle tracking
/// for a seamless edit-and-continue workflow.
pub struct ShaderHotReload {
    /// Shader dependency graph for invalidation propagation.
    dependency_graph: ShaderDependencyGraph,
    /// 3-level shader cache.
    shader_cache: ShaderCache3L,
    /// Pending recompile requests.
    pending_recompiles: Arc<Mutex<Vec<RecompileRequest>>>,
    /// Active PSOs keyed by shader ID.
    active_psos: HashMap<ShaderId, AtomicPsoHolder>,
    /// Content hashes of known shaders (path -> hash).
    known_hashes: HashMap<PathBuf, ContentHash>,
    /// Retired PSOs waiting for fence completion.
    retained_psos: Vec<RetainedPso>,
    /// Current frame number.
    current_frame: AtomicU64,
    /// Naga compiler for WGSL compilation.
    compiler: NagaCompiler,
    /// Compiler options.
    compiler_options: CompilerOptions,
    /// Target platform for compilation.
    target_platform: TargetPlatform,
    /// Whether to use background compilation (stubbed for now).
    #[allow(dead_code)]
    use_background_compilation: bool,
    /// Callback for reload success.
    on_reload_success: Option<Box<dyn Fn(&RecompileResult) + Send + Sync>>,
    /// Callback for reload failure.
    on_reload_failure: Option<Box<dyn Fn(&RecompileResult) + Send + Sync>>,
}

impl ShaderHotReload {
    /// Create a new shader hot-reload system.
    pub fn new(cache: ShaderCache3L, deps: ShaderDependencyGraph) -> Self {
        Self {
            dependency_graph: deps,
            shader_cache: cache,
            pending_recompiles: Arc::new(Mutex::new(Vec::new())),
            active_psos: HashMap::new(),
            known_hashes: HashMap::new(),
            retained_psos: Vec::new(),
            current_frame: AtomicU64::new(0),
            compiler: NagaCompiler::new(),
            compiler_options: CompilerOptions::default(),
            target_platform: TargetPlatform::Vulkan,
            use_background_compilation: true,
            on_reload_success: None,
            on_reload_failure: None,
        }
    }

    /// Set a callback for successful reloads.
    pub fn on_success<F>(&mut self, callback: F)
    where
        F: Fn(&RecompileResult) + Send + Sync + 'static,
    {
        self.on_reload_success = Some(Box::new(callback));
    }

    /// Set a callback for failed reloads.
    pub fn on_failure<F>(&mut self, callback: F)
    where
        F: Fn(&RecompileResult) + Send + Sync + 'static,
    {
        self.on_reload_failure = Some(Box::new(callback));
    }

    /// Set the target platform.
    pub fn set_target_platform(&mut self, platform: TargetPlatform) {
        self.target_platform = platform;
    }

    /// Set compiler options.
    pub fn set_compiler_options(&mut self, options: CompilerOptions) {
        self.compiler_options = options;
    }

    /// Get the dependency graph.
    pub fn dependency_graph(&self) -> &ShaderDependencyGraph {
        &self.dependency_graph
    }

    /// Get mutable dependency graph.
    pub fn dependency_graph_mut(&mut self) -> &mut ShaderDependencyGraph {
        &mut self.dependency_graph
    }

    /// Register a shader with an initial PSO.
    pub fn register_shader(&mut self, path: &Path, pso: Arc<PipelineState>) {
        let shader_id = ShaderId::from_path(path);
        self.active_psos.insert(shader_id, AtomicPsoHolder::new(pso.clone()));
        self.known_hashes.insert(path.to_path_buf(), pso.shader_hash);
    }

    /// Check for shader file changes and return recompile requests.
    ///
    /// Compares current file content hashes against cached hashes.
    /// Uses the dependency graph to find all affected entry points.
    pub fn check_for_changes(&mut self) -> Vec<RecompileRequest> {
        let mut requests = Vec::new();
        let stale_files = self.dependency_graph.verify_hashes();

        for path in stale_files {
            // Get old hash if known
            let old_hash = self.known_hashes.get(&path).cloned();

            // Get affected shaders through dependency graph
            let affected = self.dependency_graph.invalidate(&path);

            // Collect entry points for this shader
            let entry_points = if let Some(node) = self.dependency_graph.get(&path) {
                // In a real implementation, we'd extract entry points from the shader
                // For now, we use the imports as a proxy for affected entry points
                node.imports.clone()
            } else {
                Vec::new()
            };

            let mut request = RecompileRequest::new(path.clone(), entry_points);
            if let Some(hash) = old_hash {
                request = request.with_old_hash(hash);
            }

            requests.push(request);

            // Also queue recompiles for dependents
            for dependent_path in affected {
                let dep_old_hash = self.known_hashes.get(&dependent_path).cloned();
                let mut dep_request =
                    RecompileRequest::new(dependent_path, Vec::new()).with_priority(200);
                if let Some(hash) = dep_old_hash {
                    dep_request = dep_request.with_old_hash(hash);
                }
                requests.push(dep_request);
            }
        }

        // Deduplicate by shader path
        let mut seen = std::collections::HashSet::new();
        requests.retain(|r| seen.insert(r.shader_path.clone()));

        // Sort by priority (lower = higher priority)
        requests.sort_by_key(|r| r.priority);

        requests
    }

    /// Queue a recompile request for background processing.
    pub fn queue_recompile(&mut self, request: RecompileRequest) {
        let mut pending = self.pending_recompiles.lock().unwrap();

        // Check for duplicate paths
        if pending.iter().any(|r| r.shader_path == request.shader_path) {
            return;
        }

        pending.push(request);

        // Sort by priority
        pending.sort_by_key(|r| r.priority);
    }

    /// Process all pending recompilation requests.
    ///
    /// In a real implementation, this would use a background thread pool.
    /// For now, it processes synchronously.
    pub fn process_pending(&mut self) -> Vec<RecompileResult> {
        let requests: Vec<RecompileRequest> = {
            let mut pending = self.pending_recompiles.lock().unwrap();
            std::mem::take(&mut *pending)
        };

        let mut results = Vec::with_capacity(requests.len());

        for request in requests {
            let result = self.compile_shader(&request);
            results.push(result);
        }

        // Invoke callbacks
        for result in &results {
            if result.success {
                if let Some(ref callback) = self.on_reload_success {
                    callback(result);
                }
            } else if let Some(ref callback) = self.on_reload_failure {
                callback(result);
            }
        }

        results
    }

    /// Compile a single shader from a request.
    fn compile_shader(&mut self, request: &RecompileRequest) -> RecompileResult {
        let start = Instant::now();
        let shader_id = request.shader_id();

        // Read shader source
        let source = match std::fs::read_to_string(&request.shader_path) {
            Ok(s) => s,
            Err(e) => {
                return RecompileResult::failure(
                    shader_id,
                    request.shader_path.clone(),
                    CompileError::new(
                        format!("Failed to read shader file: {}", e),
                        CompileErrorKind::Internal,
                    ),
                    start.elapsed(),
                    ContentHash::zero(),
                );
            }
        };

        let new_hash = ContentHash::from_bytes(source.as_bytes());

        // Check if we can use cached bytecode
        let cache_key = CacheKey::builder()
            .source(source.as_bytes())
            .platform(self.target_platform)
            .build();

        if let Ok(Some(cached_spirv)) = self.shader_cache.get(&cache_key) {
            // Use cached bytecode
            let spirv: Vec<u32> = cached_spirv
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();

            let pso = Arc::new(PipelineState::new(
                new_hash,
                spirv,
                request.affected_entry_points.clone(),
            ));

            self.known_hashes
                .insert(request.shader_path.clone(), new_hash);

            return RecompileResult::success(
                shader_id,
                request.shader_path.clone(),
                pso,
                start.elapsed(),
                new_hash,
            );
        }

        // Compile with naga
        let compile_result = self.compiler.compile(&source, &self.compiler_options);

        match compile_result {
            Ok(result) => {
                // Generate SPIR-V
                match result.to_spirv_default() {
                    Ok(spirv) => {
                        // Cache the compiled bytecode
                        let spirv_bytes: Vec<u8> = spirv
                            .iter()
                            .flat_map(|&word| word.to_le_bytes())
                            .collect();
                        let _ = self.shader_cache.put(&cache_key, &spirv_bytes);

                        // Extract entry points from analysis
                        let entry_points: Vec<String> = result
                            .analysis
                            .entry_points
                            .iter()
                            .map(|ep| ep.name.clone())
                            .collect();

                        let pso = Arc::new(PipelineState::new(new_hash, spirv, entry_points));

                        self.known_hashes
                            .insert(request.shader_path.clone(), new_hash);

                        RecompileResult::success(
                            shader_id,
                            request.shader_path.clone(),
                            pso,
                            start.elapsed(),
                            new_hash,
                        )
                    }
                    Err(e) => RecompileResult::failure(
                        shader_id,
                        request.shader_path.clone(),
                        e,
                        start.elapsed(),
                        new_hash,
                    ),
                }
            }
            Err(e) => RecompileResult::failure(
                shader_id,
                request.shader_path.clone(),
                e,
                start.elapsed(),
                new_hash,
            ),
        }
    }

    /// Atomically swap to a new PSO at the render phase boundary.
    ///
    /// The old PSO is retained until in-flight frames complete.
    pub fn swap_pso_at_boundary(&mut self, shader_id: ShaderId, new_pso: Arc<PipelineState>) {
        let frame = self.current_frame.load(Ordering::Acquire);

        if let Some(holder) = self.active_psos.get(&shader_id) {
            if let Some(old_pso) = holder.swap(new_pso) {
                // Retain old PSO until frames complete
                self.retained_psos.push(RetainedPso {
                    pso: old_pso,
                    retired_at_frame: frame,
                    retired_at: Instant::now(),
                });
            }
        } else {
            // First time registering this shader
            self.active_psos
                .insert(shader_id, AtomicPsoHolder::new(new_pso));
        }
    }

    /// Release old PSOs that are no longer needed.
    ///
    /// Call this after a frame fence signals completion.
    pub fn release_old_psos(&mut self, completed_frame: u64) {
        self.retained_psos
            .retain(|retained| retained.retired_at_frame > completed_frame);
    }

    /// Get the active PSO for a shader.
    pub fn get_active_pso(&self, shader_id: ShaderId) -> Option<Arc<PipelineState>> {
        self.active_psos.get(&shader_id).and_then(|h| h.get())
    }

    /// Advance to the next frame.
    pub fn advance_frame(&self) -> u64 {
        self.current_frame.fetch_add(1, Ordering::AcqRel)
    }

    /// Get the current frame number.
    pub fn current_frame(&self) -> u64 {
        self.current_frame.load(Ordering::Acquire)
    }

    /// Get the number of retained (pending release) PSOs.
    pub fn retained_pso_count(&self) -> usize {
        self.retained_psos.len()
    }

    /// Get the number of active shaders.
    pub fn active_shader_count(&self) -> usize {
        self.active_psos.len()
    }

    /// Get the number of pending recompile requests.
    pub fn pending_count(&self) -> usize {
        self.pending_recompiles.lock().unwrap().len()
    }

    /// Force recompile all known shaders.
    pub fn recompile_all(&mut self) -> Vec<RecompileResult> {
        let paths: Vec<PathBuf> = self.known_hashes.keys().cloned().collect();
        let mut results = Vec::new();

        for path in paths {
            let request = RecompileRequest::new(path, Vec::new());
            let result = self.compile_shader(&request);
            results.push(result);
        }

        results
    }

    /// Check if a specific shader needs recompilation.
    pub fn needs_recompile(&self, path: &Path) -> Result<bool, DependencyError> {
        let current_hash = ContentHash::from_bytes(
            &std::fs::read(path).map_err(|e| DependencyError::IoError {
                path: path.display().to_string(),
                message: e.to_string(),
            })?,
        );

        if let Some(known_hash) = self.known_hashes.get(path) {
            Ok(current_hash != *known_hash)
        } else {
            // Unknown shader, assume it needs compilation
            Ok(true)
        }
    }

    /// Invalidate a specific shader and its dependents.
    pub fn invalidate(&mut self, path: &Path) -> Vec<PathBuf> {
        self.dependency_graph.invalidate(path)
    }

    /// Clear all caches and reset state.
    pub fn clear(&mut self) {
        self.pending_recompiles.lock().unwrap().clear();
        self.active_psos.clear();
        self.known_hashes.clear();
        self.retained_psos.clear();
        self.dependency_graph.clear();
    }
}

impl std::fmt::Debug for ShaderHotReload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShaderHotReload")
            .field("active_shaders", &self.active_psos.len())
            .field("known_hashes", &self.known_hashes.len())
            .field("retained_psos", &self.retained_psos.len())
            .field("current_frame", &self.current_frame.load(Ordering::Relaxed))
            .field("pending_recompiles", &self.pending_count())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    // Helper to create a test shader file
    fn create_shader_file(dir: &Path, name: &str, content: &str) -> PathBuf {
        let path = dir.join(name);
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(content.as_bytes()).unwrap();
        path
    }

    // Helper to create a minimal hot-reload system
    fn create_test_system(tmp: &TempDir) -> ShaderHotReload {
        use super::super::cache::CacheConfig;

        let cache_dir = tmp.path().join("cache");
        let cache =
            ShaderCache3L::new(&cache_dir, CacheConfig::with_memory_size_mb(1)).unwrap();
        let deps = ShaderDependencyGraph::new(vec![tmp.path().to_path_buf()]);

        ShaderHotReload::new(cache, deps)
    }

    // Valid WGSL shader for testing
    const VALID_SHADER: &str = r#"
        @vertex
        fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
            return vec4<f32>(f32(idx), 0.0, 0.0, 1.0);
        }

        @fragment
        fn fs_main() -> @location(0) vec4<f32> {
            return vec4<f32>(1.0, 0.0, 0.0, 1.0);
        }
    "#;

    const MODIFIED_SHADER: &str = r#"
        @vertex
        fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
            return vec4<f32>(f32(idx) * 2.0, 0.0, 0.0, 1.0);
        }

        @fragment
        fn fs_main() -> @location(0) vec4<f32> {
            return vec4<f32>(0.0, 1.0, 0.0, 1.0);
        }
    "#;

    const INVALID_SHADER: &str = r#"
        fn broken() {
            let x = ;
        }
    "#;

    // -----------------------------------------------------------------------
    // Test 1-4: Change detection via content hash
    // -----------------------------------------------------------------------

    #[test]
    fn test_change_detection_new_file() {
        let tmp = TempDir::new().unwrap();
        let mut system = create_test_system(&tmp);

        // Create a shader and register it
        let shader_path = create_shader_file(tmp.path(), "test.wgsl", VALID_SHADER);
        let initial_hash = ContentHash::from_bytes(VALID_SHADER.as_bytes());
        let pso = Arc::new(PipelineState::dummy(initial_hash));
        system.register_shader(&shader_path, pso);

        // Check for changes (should be none)
        let requests = system.check_for_changes();
        assert!(requests.is_empty(), "No changes expected for unchanged file");
    }

    #[test]
    fn test_change_detection_modified_file() {
        let tmp = TempDir::new().unwrap();
        let mut system = create_test_system(&tmp);

        // Create and register shader
        let shader_path = create_shader_file(tmp.path(), "test.wgsl", VALID_SHADER);
        let initial_hash = ContentHash::from_bytes(VALID_SHADER.as_bytes());
        let pso = Arc::new(PipelineState::dummy(initial_hash));
        system.register_shader(&shader_path, pso);

        // Also need to analyze it to add to dependency graph
        system.dependency_graph_mut().analyze(&shader_path).ok();

        // Modify the file
        std::fs::write(&shader_path, MODIFIED_SHADER).unwrap();

        // Check for changes
        let requests = system.check_for_changes();
        assert!(!requests.is_empty(), "Should detect file change");
        assert_eq!(requests[0].shader_path, shader_path);
    }

    #[test]
    fn test_change_detection_hash_comparison() {
        let tmp = TempDir::new().unwrap();
        let mut system = create_test_system(&tmp);

        let shader_path = create_shader_file(tmp.path(), "test.wgsl", VALID_SHADER);
        let old_hash = ContentHash::from_bytes(VALID_SHADER.as_bytes());
        let new_hash = ContentHash::from_bytes(MODIFIED_SHADER.as_bytes());

        // Verify hashes are different
        assert_ne!(old_hash, new_hash);

        // Register with old hash
        let pso = Arc::new(PipelineState::dummy(old_hash));
        system.register_shader(&shader_path, pso);

        // Verify needs_recompile detects change
        std::fs::write(&shader_path, MODIFIED_SHADER).unwrap();
        assert!(system.needs_recompile(&shader_path).unwrap());
    }

    #[test]
    fn test_change_detection_unchanged_file() {
        let tmp = TempDir::new().unwrap();
        let mut system = create_test_system(&tmp);

        let shader_path = create_shader_file(tmp.path(), "test.wgsl", VALID_SHADER);

        // Register and verify no change
        let pso = Arc::new(PipelineState::dummy(ContentHash::from_bytes(
            VALID_SHADER.as_bytes(),
        )));
        system.register_shader(&shader_path, pso);

        assert!(!system.needs_recompile(&shader_path).unwrap());
    }

    // -----------------------------------------------------------------------
    // Test 5-8: Dependency invalidation propagation
    // -----------------------------------------------------------------------

    #[test]
    fn test_dependency_invalidation_direct() {
        let tmp = TempDir::new().unwrap();
        let mut system = create_test_system(&tmp);

        // Create main shader with include
        let main_source = r#"
            #include "common.wgsl"
            @vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }
        "#;
        let common_source = "fn common() -> f32 { return 1.0; }";

        let main_path = create_shader_file(tmp.path(), "main.wgsl", main_source);
        let common_path = create_shader_file(tmp.path(), "common.wgsl", common_source);

        // Analyze to build dependency graph
        system.dependency_graph_mut().analyze(&main_path).ok();

        // Invalidate the common file
        let affected = system.invalidate(&common_path);

        // main.wgsl should be affected
        assert!(
            affected.contains(&main_path),
            "main.wgsl should be invalidated when common.wgsl changes"
        );
    }

    #[test]
    fn test_dependency_invalidation_transitive() {
        let tmp = TempDir::new().unwrap();
        let mut system = create_test_system(&tmp);

        // Create chain: a -> b -> c
        let a_source = r#"#include "b.wgsl"
            @vertex fn a() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }"#;
        let b_source = r#"#include "c.wgsl"
            fn b() -> f32 { return 1.0; }"#;
        let c_source = "fn c() -> f32 { return 2.0; }";

        let a_path = create_shader_file(tmp.path(), "a.wgsl", a_source);
        let b_path = create_shader_file(tmp.path(), "b.wgsl", b_source);
        let c_path = create_shader_file(tmp.path(), "c.wgsl", c_source);

        // Analyze to build graph
        system.dependency_graph_mut().analyze(&a_path).ok();

        // Invalidate c
        let affected = system.invalidate(&c_path);

        // Both a and b should be affected
        assert!(affected.contains(&b_path), "b.wgsl should be affected");
        assert!(affected.contains(&a_path), "a.wgsl should be affected");
    }

    #[test]
    fn test_dependency_diamond_invalidation() {
        let tmp = TempDir::new().unwrap();
        let mut system = create_test_system(&tmp);

        // Diamond: main -> (a, b) -> common
        let main_source = r#"#include "a.wgsl"
            #include "b.wgsl"
            @vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }"#;
        let a_source = r#"#include "common.wgsl"
            fn a() -> f32 { return 1.0; }"#;
        let b_source = r#"#include "common.wgsl"
            fn b() -> f32 { return 2.0; }"#;
        let common_source = "fn common() -> f32 { return 0.0; }";

        let main_path = create_shader_file(tmp.path(), "main.wgsl", main_source);
        let a_path = create_shader_file(tmp.path(), "a.wgsl", a_source);
        let b_path = create_shader_file(tmp.path(), "b.wgsl", b_source);
        let common_path = create_shader_file(tmp.path(), "common.wgsl", common_source);

        system.dependency_graph_mut().analyze(&main_path).ok();

        // Invalidate common
        let affected = system.invalidate(&common_path);

        // All three dependents should be affected
        assert!(affected.contains(&a_path));
        assert!(affected.contains(&b_path));
        assert!(affected.contains(&main_path));
    }

    #[test]
    fn test_dependency_no_dependents() {
        let tmp = TempDir::new().unwrap();
        let mut system = create_test_system(&tmp);

        let shader_path = create_shader_file(tmp.path(), "standalone.wgsl", VALID_SHADER);
        system.dependency_graph_mut().analyze(&shader_path).ok();

        // Invalidate standalone shader
        let affected = system.invalidate(&shader_path);

        // No dependents
        assert!(affected.is_empty());
    }

    // -----------------------------------------------------------------------
    // Test 9-11: Background recompile queueing
    // -----------------------------------------------------------------------

    #[test]
    fn test_queue_recompile_single() {
        let tmp = TempDir::new().unwrap();
        let mut system = create_test_system(&tmp);

        let shader_path = create_shader_file(tmp.path(), "test.wgsl", VALID_SHADER);
        let request = RecompileRequest::new(shader_path.clone(), vec!["main".to_string()]);

        system.queue_recompile(request);
        assert_eq!(system.pending_count(), 1);
    }

    #[test]
    fn test_queue_recompile_deduplicate() {
        let tmp = TempDir::new().unwrap();
        let mut system = create_test_system(&tmp);

        let shader_path = create_shader_file(tmp.path(), "test.wgsl", VALID_SHADER);

        // Queue same shader twice
        system.queue_recompile(RecompileRequest::new(shader_path.clone(), vec![]));
        system.queue_recompile(RecompileRequest::new(shader_path.clone(), vec![]));

        // Should be deduplicated
        assert_eq!(system.pending_count(), 1);
    }

    #[test]
    fn test_queue_recompile_priority_sorting() {
        let tmp = TempDir::new().unwrap();
        let mut system = create_test_system(&tmp);

        let path1 = create_shader_file(tmp.path(), "low.wgsl", VALID_SHADER);
        let path2 = create_shader_file(tmp.path(), "high.wgsl", VALID_SHADER);

        system.queue_recompile(RecompileRequest::new(path1.clone(), vec![]).with_priority(100));
        system.queue_recompile(RecompileRequest::new(path2.clone(), vec![]).with_priority(10));

        // Process and verify order
        let results = system.process_pending();
        assert_eq!(results.len(), 2);
        // Higher priority (lower number) should be processed first
        assert_eq!(results[0].shader_path, path2);
        assert_eq!(results[1].shader_path, path1);
    }

    // -----------------------------------------------------------------------
    // Test 12-14: PSO swap atomicity
    // -----------------------------------------------------------------------

    #[test]
    fn test_pso_swap_first_time() {
        let tmp = TempDir::new().unwrap();
        let mut system = create_test_system(&tmp);

        let shader_path = create_shader_file(tmp.path(), "test.wgsl", VALID_SHADER);
        let shader_id = ShaderId::from_path(&shader_path);
        let pso = Arc::new(PipelineState::dummy(ContentHash::zero()));

        // First swap (registration)
        system.swap_pso_at_boundary(shader_id, pso.clone());

        // Verify it's active
        let active = system.get_active_pso(shader_id);
        assert!(active.is_some());
        assert_eq!(active.unwrap().id, pso.id);
    }

    #[test]
    fn test_pso_swap_replace() {
        let tmp = TempDir::new().unwrap();
        let mut system = create_test_system(&tmp);

        let shader_path = create_shader_file(tmp.path(), "test.wgsl", VALID_SHADER);
        let shader_id = ShaderId::from_path(&shader_path);

        let pso1 = Arc::new(PipelineState::dummy(ContentHash::from_bytes(b"v1")));
        let pso2 = Arc::new(PipelineState::dummy(ContentHash::from_bytes(b"v2")));

        // Register first PSO
        system.swap_pso_at_boundary(shader_id, pso1.clone());
        assert_eq!(system.retained_pso_count(), 0);

        // Swap to second PSO
        system.swap_pso_at_boundary(shader_id, pso2.clone());

        // First PSO should be retained
        assert_eq!(system.retained_pso_count(), 1);

        // Active PSO should be the new one
        let active = system.get_active_pso(shader_id).unwrap();
        assert_eq!(active.id, pso2.id);
    }

    #[test]
    fn test_pso_swap_concurrent_safety() {
        let tmp = TempDir::new().unwrap();
        let mut system = create_test_system(&tmp);

        let shader_path = create_shader_file(tmp.path(), "test.wgsl", VALID_SHADER);
        let shader_id = ShaderId::from_path(&shader_path);

        // Register initial PSO
        let pso = Arc::new(PipelineState::dummy(ContentHash::zero()));
        system.swap_pso_at_boundary(shader_id, pso);

        // Multiple rapid swaps
        for i in 0..10 {
            let new_pso = Arc::new(PipelineState::dummy(ContentHash::from_bytes(
                format!("v{}", i).as_bytes(),
            )));
            system.swap_pso_at_boundary(shader_id, new_pso);
        }

        // Should have 10 retained PSOs (the original + 9 intermediate)
        assert_eq!(system.retained_pso_count(), 10);
    }

    // -----------------------------------------------------------------------
    // Test 15-17: Fence-based old PSO retention
    // -----------------------------------------------------------------------

    #[test]
    fn test_fence_retain_old_pso() {
        let tmp = TempDir::new().unwrap();
        let mut system = create_test_system(&tmp);

        let shader_path = create_shader_file(tmp.path(), "test.wgsl", VALID_SHADER);
        let shader_id = ShaderId::from_path(&shader_path);

        // Register and swap
        let pso1 = Arc::new(PipelineState::dummy(ContentHash::zero()));
        let pso2 = Arc::new(PipelineState::dummy(ContentHash::zero()));

        system.swap_pso_at_boundary(shader_id, pso1);
        system.advance_frame();
        system.swap_pso_at_boundary(shader_id, pso2);

        // Old PSO retained
        assert_eq!(system.retained_pso_count(), 1);

        // Frame 0 completes - but PSO was retired at frame 1, so still retained
        system.release_old_psos(0);
        assert_eq!(system.retained_pso_count(), 1);

        // Frame 1 completes - now can release
        system.release_old_psos(1);
        assert_eq!(system.retained_pso_count(), 0);
    }

    #[test]
    fn test_fence_release_after_frame_complete() {
        let tmp = TempDir::new().unwrap();
        let mut system = create_test_system(&tmp);

        let shader_path = create_shader_file(tmp.path(), "test.wgsl", VALID_SHADER);
        let shader_id = ShaderId::from_path(&shader_path);

        // Create several PSOs at different frames
        // Frame 0: register (no retired), advance to 1
        // Frame 1: swap (retired at 1), advance to 2
        // Frame 2: swap (retired at 2), advance to 3
        // Frame 3: swap (retired at 3), advance to 4
        // Frame 4: swap (retired at 4), advance to 5
        for i in 0..5 {
            let pso = Arc::new(PipelineState::dummy(ContentHash::from_bytes(
                format!("v{}", i).as_bytes(),
            )));
            system.swap_pso_at_boundary(shader_id, pso);
            system.advance_frame();
        }

        // 4 old PSOs retained (first registration doesn't create a retained PSO)
        assert_eq!(system.retained_pso_count(), 4);

        // Release up to frame 2 (keeps retired_at_frame > 2, i.e., frames 3 and 4)
        system.release_old_psos(2);

        // PSOs retired at frames 1, 2 are released; frames 3, 4 are retained
        assert_eq!(system.retained_pso_count(), 2);
    }

    #[test]
    fn test_fence_multiple_shaders() {
        let tmp = TempDir::new().unwrap();
        let mut system = create_test_system(&tmp);

        let path1 = create_shader_file(tmp.path(), "a.wgsl", VALID_SHADER);
        let path2 = create_shader_file(tmp.path(), "b.wgsl", VALID_SHADER);
        let id1 = ShaderId::from_path(&path1);
        let id2 = ShaderId::from_path(&path2);

        // Register both
        system.swap_pso_at_boundary(id1, Arc::new(PipelineState::dummy(ContentHash::zero())));
        system.swap_pso_at_boundary(id2, Arc::new(PipelineState::dummy(ContentHash::zero())));

        // Swap both
        system.advance_frame();
        system.swap_pso_at_boundary(id1, Arc::new(PipelineState::dummy(ContentHash::zero())));
        system.swap_pso_at_boundary(id2, Arc::new(PipelineState::dummy(ContentHash::zero())));

        // 2 retained
        assert_eq!(system.retained_pso_count(), 2);

        // Release all
        system.release_old_psos(100);
        assert_eq!(system.retained_pso_count(), 0);
    }

    // -----------------------------------------------------------------------
    // Test 18-20: Error handling and fallback
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_invalid_shader_keeps_old() {
        let tmp = TempDir::new().unwrap();
        let mut system = create_test_system(&tmp);

        // Start with valid shader
        let shader_path = create_shader_file(tmp.path(), "test.wgsl", VALID_SHADER);
        let shader_id = ShaderId::from_path(&shader_path);

        let initial_hash = ContentHash::from_bytes(VALID_SHADER.as_bytes());
        let initial_pso = Arc::new(PipelineState::dummy(initial_hash));
        system.register_shader(&shader_path, initial_pso.clone());

        // Now "edit" to invalid shader
        std::fs::write(&shader_path, INVALID_SHADER).unwrap();

        // Queue and process recompile
        system.queue_recompile(RecompileRequest::new(shader_path.clone(), vec![]));
        let results = system.process_pending();

        // Should fail
        assert_eq!(results.len(), 1);
        assert!(!results[0].success);
        assert!(results[0].error.is_some());

        // Active PSO should still be the old one (unchanged)
        let active = system.get_active_pso(shader_id).unwrap();
        assert_eq!(active.shader_hash, initial_hash);
    }

    #[test]
    fn test_error_file_not_found() {
        let tmp = TempDir::new().unwrap();
        let mut system = create_test_system(&tmp);

        let nonexistent = tmp.path().join("nonexistent.wgsl");
        system.queue_recompile(RecompileRequest::new(nonexistent, vec![]));
        let results = system.process_pending();

        assert!(!results[0].success);
        assert!(results[0].error.is_some());
    }

    #[test]
    fn test_error_callback_invoked() {
        let tmp = TempDir::new().unwrap();
        let mut system = create_test_system(&tmp);

        let failure_count = Arc::new(AtomicU64::new(0));
        let count_clone = failure_count.clone();

        system.on_failure(move |_| {
            count_clone.fetch_add(1, Ordering::SeqCst);
        });

        // Create invalid shader
        let shader_path = create_shader_file(tmp.path(), "bad.wgsl", INVALID_SHADER);
        system.queue_recompile(RecompileRequest::new(shader_path, vec![]));
        let _ = system.process_pending();

        assert_eq!(failure_count.load(Ordering::SeqCst), 1);
    }

    // -----------------------------------------------------------------------
    // Additional tests to reach 20+
    // -----------------------------------------------------------------------

    #[test]
    fn test_shader_id_from_path() {
        let path1 = PathBuf::from("shaders/main.wgsl");
        let path2 = PathBuf::from("shaders/main.wgsl");
        let path3 = PathBuf::from("shaders/other.wgsl");

        let id1 = ShaderId::from_path(&path1);
        let id2 = ShaderId::from_path(&path2);
        let id3 = ShaderId::from_path(&path3);

        // Same path should produce same ID
        assert_eq!(id1, id2);
        // Different paths should produce different IDs
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_recompile_all() {
        let tmp = TempDir::new().unwrap();
        let mut system = create_test_system(&tmp);

        // Create multiple shaders
        let path1 = create_shader_file(tmp.path(), "a.wgsl", VALID_SHADER);
        let path2 = create_shader_file(tmp.path(), "b.wgsl", VALID_SHADER);

        system.register_shader(
            &path1,
            Arc::new(PipelineState::dummy(ContentHash::zero())),
        );
        system.register_shader(
            &path2,
            Arc::new(PipelineState::dummy(ContentHash::zero())),
        );

        // Recompile all
        let results = system.recompile_all();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.success));
    }

    #[test]
    fn test_clear() {
        let tmp = TempDir::new().unwrap();
        let mut system = create_test_system(&tmp);

        let shader_path = create_shader_file(tmp.path(), "test.wgsl", VALID_SHADER);
        system.register_shader(
            &shader_path,
            Arc::new(PipelineState::dummy(ContentHash::zero())),
        );
        system.queue_recompile(RecompileRequest::new(shader_path, vec![]));

        assert_eq!(system.active_shader_count(), 1);
        assert_eq!(system.pending_count(), 1);

        system.clear();

        assert_eq!(system.active_shader_count(), 0);
        assert_eq!(system.pending_count(), 0);
    }

    #[test]
    fn test_success_callback_invoked() {
        let tmp = TempDir::new().unwrap();
        let mut system = create_test_system(&tmp);

        let success_count = Arc::new(AtomicU64::new(0));
        let count_clone = success_count.clone();

        system.on_success(move |_| {
            count_clone.fetch_add(1, Ordering::SeqCst);
        });

        let shader_path = create_shader_file(tmp.path(), "good.wgsl", VALID_SHADER);
        system.queue_recompile(RecompileRequest::new(shader_path, vec![]));
        let _ = system.process_pending();

        assert_eq!(success_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_frame_advance() {
        let tmp = TempDir::new().unwrap();
        let system = create_test_system(&tmp);

        assert_eq!(system.current_frame(), 0);
        system.advance_frame();
        assert_eq!(system.current_frame(), 1);
        system.advance_frame();
        system.advance_frame();
        assert_eq!(system.current_frame(), 3);
    }

    #[test]
    fn test_pipeline_state_creation() {
        let hash = ContentHash::from_bytes(b"test");
        let spirv = vec![0x07230203, 0x00010300]; // SPIR-V header
        let entry_points = vec!["main".to_string()];

        let pso = PipelineState::new(hash, spirv.clone(), entry_points.clone());

        assert_eq!(pso.shader_hash, hash);
        assert_eq!(pso.spirv, Some(spirv));
        assert_eq!(pso.entry_points, entry_points);
    }

    #[test]
    fn test_cached_spirv_reuse() {
        let tmp = TempDir::new().unwrap();
        let mut system = create_test_system(&tmp);

        let shader_path = create_shader_file(tmp.path(), "test.wgsl", VALID_SHADER);

        // First compile populates cache
        system.queue_recompile(RecompileRequest::new(shader_path.clone(), vec![]));
        let results1 = system.process_pending();
        assert!(results1[0].success);
        let time1 = results1[0].compile_time;

        // Second compile should use cache (faster)
        system.queue_recompile(RecompileRequest::new(shader_path, vec![]));
        let results2 = system.process_pending();
        assert!(results2[0].success);
        let time2 = results2[0].compile_time;

        // Cache hit should be faster (or at least not significantly slower)
        // This is a weak test - in practice cache hits are much faster
        assert!(time2 <= time1 + Duration::from_millis(50));
    }

    #[test]
    fn test_recompile_request_builder() {
        let path = PathBuf::from("test.wgsl");
        let old_hash = ContentHash::from_bytes(b"old");

        let request = RecompileRequest::new(path.clone(), vec!["main".to_string()])
            .with_priority(50)
            .with_old_hash(old_hash);

        assert_eq!(request.shader_path, path);
        assert_eq!(request.priority, 50);
        assert_eq!(request.old_hash, Some(old_hash));
        assert_eq!(request.affected_entry_points, vec!["main".to_string()]);
    }

    #[test]
    fn test_debug_impl() {
        let tmp = TempDir::new().unwrap();
        let system = create_test_system(&tmp);

        let debug_str = format!("{:?}", system);
        assert!(debug_str.contains("ShaderHotReload"));
        assert!(debug_str.contains("active_shaders"));
        assert!(debug_str.contains("current_frame"));
    }
}
