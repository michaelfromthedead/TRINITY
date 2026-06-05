//! Material Instance Hot-Reload and Dependency Viewer (T-AS-6.5)
//!
//! Provides material parameter hot-reload for the TRINITY asset pipeline:
//!
//! - **Parameter Updates**: Push updated uniform buffer without full PSO recreation
//! - **DSL File Changes**: Detect via file watcher, trigger WGSL regeneration, follow
//!   shader hot-reload path
//! - **Dependency Graph Visualization**: Structured data from ContentStore provenance
//! - **Graph Queries**: Asset -> dependencies -> dependents -> change propagation path
//! - **Real-time Updates**: Graph refreshes when dependencies change
//!
//! # Architecture
//!
//! ```text
//! +------------------+     +------------------------+     +------------------+
//! | ContentChange    | --> | MaterialReloadManager  | --> | Uniform Updates  |
//! | (parameter/DSL)  |     | (track instances,      |     | (buffer push)    |
//! +------------------+     |  dependency graph)     |     +------------------+
//!                          +------------------------+
//!                                  |
//!                                  v
//!                          +------------------------+
//!                          | ShaderReloadManager    |
//!                          | (for DSL recompilation)|
//!                          +------------------------+
//! ```
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::hot_reload::material_reload::{
//!     MaterialReloadManager, MaterialInstance, MaterialParameter, MaterialValue,
//! };
//!
//! let mut manager = MaterialReloadManager::new();
//!
//! // Register a material instance
//! let instance = MaterialInstance::new(material_id, shader_path);
//! manager.register_material(instance);
//!
//! // When a parameter changes (e.g., color picker in editor)
//! manager.on_parameter_change(material_id, "base_color", MaterialValue::Color([1.0, 0.0, 0.0, 1.0]));
//!
//! // Get pending uniform buffer updates for GPU upload
//! let updates = manager.push_uniform_updates();
//! for (id, buffer) in updates {
//!     upload_uniform_buffer(id, buffer);
//! }
//!
//! // Query dependency graph for visualization
//! let deps = manager.get_dependency_graph(&Path::new("materials/metal.mat"));
//! println!("Dependencies: {:?}", deps.dependencies);
//! println!("Dependents: {:?}", deps.dependents);
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use super::content_change::{ContentChange, ContentChangeKind};
use super::shader_reload::{ReloadStatus, ShaderReloadManager};
use crate::pipeline::ContentHash;

// ---------------------------------------------------------------------------
// MaterialId
// ---------------------------------------------------------------------------

/// Unique identifier for a material instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialId(pub u64);

impl MaterialId {
    /// Create a new material ID from a raw u64.
    pub const fn from_raw(id: u64) -> Self {
        Self(id)
    }

    /// Create a material ID from a path.
    pub fn from_path(path: &Path) -> Self {
        let hash = ContentHash::from_bytes(path.to_string_lossy().as_bytes());
        let bytes = hash.as_bytes();
        let id = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        Self(id)
    }

    /// Get the raw u64 value.
    pub const fn raw(&self) -> u64 {
        self.0
    }

    /// Generate a new unique ID.
    pub fn new_unique() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl std::fmt::Display for MaterialId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Material({:016x})", self.0)
    }
}

// ---------------------------------------------------------------------------
// MaterialValue
// ---------------------------------------------------------------------------

/// Material parameter value types.
#[derive(Debug, Clone, PartialEq)]
pub enum MaterialValue {
    /// Single float value.
    Float(f32),
    /// 2D float vector.
    Vec2([f32; 2]),
    /// 3D float vector.
    Vec3([f32; 3]),
    /// 4D float vector.
    Vec4([f32; 4]),
    /// Single integer value.
    Int(i32),
    /// Boolean value.
    Bool(bool),
    /// RGBA color value (linear space).
    Color([f32; 4]),
}

impl MaterialValue {
    /// Get the size of this value in bytes.
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::Float(_) => 4,
            Self::Vec2(_) => 8,
            Self::Vec3(_) => 12,
            Self::Vec4(_) => 16,
            Self::Int(_) => 4,
            Self::Bool(_) => 4, // Padded to 4 bytes for GPU alignment
            Self::Color(_) => 16,
        }
    }

    /// Write this value to a byte buffer at the given offset.
    pub fn write_to_buffer(&self, buffer: &mut [u8], offset: usize) {
        match self {
            Self::Float(v) => {
                let bytes = v.to_le_bytes();
                buffer[offset..offset + 4].copy_from_slice(&bytes);
            }
            Self::Vec2(v) => {
                for (i, &val) in v.iter().enumerate() {
                    let bytes = val.to_le_bytes();
                    buffer[offset + i * 4..offset + i * 4 + 4].copy_from_slice(&bytes);
                }
            }
            Self::Vec3(v) => {
                for (i, &val) in v.iter().enumerate() {
                    let bytes = val.to_le_bytes();
                    buffer[offset + i * 4..offset + i * 4 + 4].copy_from_slice(&bytes);
                }
            }
            Self::Vec4(v) | Self::Color(v) => {
                for (i, &val) in v.iter().enumerate() {
                    let bytes = val.to_le_bytes();
                    buffer[offset + i * 4..offset + i * 4 + 4].copy_from_slice(&bytes);
                }
            }
            Self::Int(v) => {
                let bytes = v.to_le_bytes();
                buffer[offset..offset + 4].copy_from_slice(&bytes);
            }
            Self::Bool(v) => {
                let int_val: i32 = if *v { 1 } else { 0 };
                let bytes = int_val.to_le_bytes();
                buffer[offset..offset + 4].copy_from_slice(&bytes);
            }
        }
    }

    /// Read a value from a byte buffer at the given offset.
    pub fn read_from_buffer(buffer: &[u8], offset: usize, value_type: &Self) -> Option<Self> {
        if offset + value_type.size_bytes() > buffer.len() {
            return None;
        }

        Some(match value_type {
            Self::Float(_) => {
                let bytes: [u8; 4] = buffer[offset..offset + 4].try_into().ok()?;
                Self::Float(f32::from_le_bytes(bytes))
            }
            Self::Vec2(_) => {
                let mut v = [0.0f32; 2];
                for (i, val) in v.iter_mut().enumerate() {
                    let bytes: [u8; 4] = buffer[offset + i * 4..offset + i * 4 + 4]
                        .try_into()
                        .ok()?;
                    *val = f32::from_le_bytes(bytes);
                }
                Self::Vec2(v)
            }
            Self::Vec3(_) => {
                let mut v = [0.0f32; 3];
                for (i, val) in v.iter_mut().enumerate() {
                    let bytes: [u8; 4] = buffer[offset + i * 4..offset + i * 4 + 4]
                        .try_into()
                        .ok()?;
                    *val = f32::from_le_bytes(bytes);
                }
                Self::Vec3(v)
            }
            Self::Vec4(_) => {
                let mut v = [0.0f32; 4];
                for (i, val) in v.iter_mut().enumerate() {
                    let bytes: [u8; 4] = buffer[offset + i * 4..offset + i * 4 + 4]
                        .try_into()
                        .ok()?;
                    *val = f32::from_le_bytes(bytes);
                }
                Self::Vec4(v)
            }
            Self::Int(_) => {
                let bytes: [u8; 4] = buffer[offset..offset + 4].try_into().ok()?;
                Self::Int(i32::from_le_bytes(bytes))
            }
            Self::Bool(_) => {
                let bytes: [u8; 4] = buffer[offset..offset + 4].try_into().ok()?;
                Self::Bool(i32::from_le_bytes(bytes) != 0)
            }
            Self::Color(_) => {
                let mut v = [0.0f32; 4];
                for (i, val) in v.iter_mut().enumerate() {
                    let bytes: [u8; 4] = buffer[offset + i * 4..offset + i * 4 + 4]
                        .try_into()
                        .ok()?;
                    *val = f32::from_le_bytes(bytes);
                }
                Self::Color(v)
            }
        })
    }
}

// ---------------------------------------------------------------------------
// MaterialParameter
// ---------------------------------------------------------------------------

/// A single material parameter with name, value, and buffer layout info.
#[derive(Debug, Clone)]
pub struct MaterialParameter {
    /// Parameter name (used for lookup).
    pub name: String,
    /// Current parameter value.
    pub value: MaterialValue,
    /// Byte offset in the uniform buffer.
    pub offset: usize,
    /// Size in bytes (matches value type).
    pub size: usize,
}

impl MaterialParameter {
    /// Create a new material parameter.
    pub fn new(name: impl Into<String>, value: MaterialValue, offset: usize) -> Self {
        let size = value.size_bytes();
        Self {
            name: name.into(),
            value,
            offset,
            size,
        }
    }

    /// Update the parameter value.
    pub fn set_value(&mut self, value: MaterialValue) {
        self.value = value;
    }
}

// ---------------------------------------------------------------------------
// MaterialInstance
// ---------------------------------------------------------------------------

/// A material instance with parameters and uniform buffer.
#[derive(Debug, Clone)]
pub struct MaterialInstance {
    /// Unique material ID.
    pub id: MaterialId,
    /// Path to the shader/DSL file.
    pub shader_path: PathBuf,
    /// Material parameters.
    pub parameters: Vec<MaterialParameter>,
    /// Uniform buffer data.
    pub uniform_buffer: Vec<u8>,
    /// Whether this instance has pending updates.
    dirty: bool,
    /// Content hash of the DSL file.
    pub dsl_hash: Option<ContentHash>,
}

impl MaterialInstance {
    /// Create a new material instance.
    pub fn new(id: MaterialId, shader_path: PathBuf) -> Self {
        Self {
            id,
            shader_path,
            parameters: Vec::new(),
            uniform_buffer: Vec::new(),
            dirty: false,
            dsl_hash: None,
        }
    }

    /// Add a parameter to this material instance.
    pub fn add_parameter(&mut self, param: MaterialParameter) {
        // Ensure buffer is large enough
        let required_size = param.offset + param.size;
        if self.uniform_buffer.len() < required_size {
            self.uniform_buffer.resize(required_size, 0);
        }

        // Write initial value to buffer
        param.value.write_to_buffer(&mut self.uniform_buffer, param.offset);

        self.parameters.push(param);
    }

    /// Get a parameter by name.
    pub fn get_parameter(&self, name: &str) -> Option<&MaterialParameter> {
        self.parameters.iter().find(|p| p.name == name)
    }

    /// Get a mutable parameter by name.
    pub fn get_parameter_mut(&mut self, name: &str) -> Option<&mut MaterialParameter> {
        self.parameters.iter_mut().find(|p| p.name == name)
    }

    /// Set a parameter value and update the uniform buffer.
    pub fn set_parameter(&mut self, name: &str, value: MaterialValue) -> bool {
        // Find parameter index to avoid borrow conflicts
        let param_idx = self.parameters.iter().position(|p| p.name == name);

        if let Some(idx) = param_idx {
            let offset = self.parameters[idx].offset;
            self.parameters[idx].value = value.clone();
            value.write_to_buffer(&mut self.uniform_buffer, offset);
            self.dirty = true;
            true
        } else {
            false
        }
    }

    /// Check if this instance has pending updates.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Clear the dirty flag.
    pub fn clear_dirty(&mut self) {
        self.dirty = false;
    }

    /// Mark as dirty for update.
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Get the uniform buffer data.
    pub fn uniform_data(&self) -> &[u8] {
        &self.uniform_buffer
    }

    /// Get total buffer size with 16-byte alignment.
    pub fn aligned_buffer_size(&self) -> usize {
        (self.uniform_buffer.len() + 15) & !15
    }
}

// ---------------------------------------------------------------------------
// MaterialReloadKind
// ---------------------------------------------------------------------------

/// Type of material reload operation.
#[derive(Debug, Clone)]
pub enum MaterialReloadKind {
    /// Only parameters changed - just update uniform buffer.
    ParameterUpdate {
        /// Names of parameters that changed.
        changed: Vec<String>,
    },
    /// DSL file changed - need shader recompilation.
    ShaderRecompile,
    /// Full reload (e.g., layout changed).
    FullReload,
}

impl MaterialReloadKind {
    /// Returns true if this is a parameter-only update.
    pub fn is_parameter_update(&self) -> bool {
        matches!(self, Self::ParameterUpdate { .. })
    }

    /// Returns true if shader recompilation is needed.
    pub fn requires_shader_recompile(&self) -> bool {
        matches!(self, Self::ShaderRecompile | Self::FullReload)
    }
}

// ---------------------------------------------------------------------------
// MaterialReloadEvent
// ---------------------------------------------------------------------------

/// Event emitted during material hot-reload.
#[derive(Debug, Clone)]
pub struct MaterialReloadEvent {
    /// ID of the material being reloaded.
    pub material_id: MaterialId,
    /// Type of reload operation.
    pub kind: MaterialReloadKind,
    /// Current reload status.
    pub status: ReloadStatus,
    /// Error message if failed.
    pub error: Option<String>,
    /// Timestamp when this event was created.
    pub timestamp: Instant,
}

impl MaterialReloadEvent {
    /// Create a parameter update event.
    pub fn parameter_update(material_id: MaterialId, changed: Vec<String>) -> Self {
        Self {
            material_id,
            kind: MaterialReloadKind::ParameterUpdate { changed },
            status: ReloadStatus::Ready,
            timestamp: Instant::now(),
            error: None,
        }
    }

    /// Create a shader recompile event.
    pub fn shader_recompile(material_id: MaterialId) -> Self {
        Self {
            material_id,
            kind: MaterialReloadKind::ShaderRecompile,
            status: ReloadStatus::Pending,
            timestamp: Instant::now(),
            error: None,
        }
    }

    /// Create a failed event.
    pub fn failed(material_id: MaterialId, error: impl Into<String>) -> Self {
        Self {
            material_id,
            kind: MaterialReloadKind::FullReload,
            status: ReloadStatus::Failed,
            error: Some(error.into()),
            timestamp: Instant::now(),
        }
    }
}

// ---------------------------------------------------------------------------
// AssetType
// ---------------------------------------------------------------------------

/// Type of asset in the dependency graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssetType {
    /// Material definition/DSL file.
    Material,
    /// Shader source file (WGSL, GLSL, etc.).
    Shader,
    /// Texture image.
    Texture,
    /// Include/header file.
    Include,
    /// Unknown asset type.
    Unknown,
}

impl AssetType {
    /// Determine asset type from file extension.
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "mat" | "material" | "mtl" => Self::Material,
            "wgsl" | "glsl" | "hlsl" | "vert" | "frag" | "comp" => Self::Shader,
            "png" | "jpg" | "jpeg" | "tga" | "bmp" | "dds" | "ktx" | "ktx2" => Self::Texture,
            "inc" | "h" | "glslinc" | "wgslinc" => Self::Include,
            _ => Self::Unknown,
        }
    }

    /// Determine asset type from path.
    pub fn from_path(path: &Path) -> Self {
        path.extension()
            .and_then(|e| e.to_str())
            .map(Self::from_extension)
            .unwrap_or(Self::Unknown)
    }
}

// ---------------------------------------------------------------------------
// DependencyNode
// ---------------------------------------------------------------------------

/// A node in the asset dependency graph.
#[derive(Debug, Clone)]
pub struct DependencyNode {
    /// Path to this asset.
    pub path: PathBuf,
    /// Type of this asset.
    pub asset_type: AssetType,
    /// Assets that this asset depends on.
    pub dependencies: Vec<PathBuf>,
    /// Assets that depend on this asset.
    pub dependents: Vec<PathBuf>,
    /// Content hash of this asset.
    pub content_hash: Option<ContentHash>,
    /// Last modification time.
    pub last_modified: Option<Instant>,
}

impl DependencyNode {
    /// Create a new dependency node.
    pub fn new(path: PathBuf) -> Self {
        let asset_type = AssetType::from_path(&path);
        Self {
            path,
            asset_type,
            dependencies: Vec::new(),
            dependents: Vec::new(),
            content_hash: None,
            last_modified: None,
        }
    }

    /// Create a node with explicit asset type.
    pub fn with_type(path: PathBuf, asset_type: AssetType) -> Self {
        Self {
            path,
            asset_type,
            dependencies: Vec::new(),
            dependents: Vec::new(),
            content_hash: None,
            last_modified: None,
        }
    }

    /// Add a dependency.
    pub fn add_dependency(&mut self, dep: PathBuf) {
        if !self.dependencies.contains(&dep) {
            self.dependencies.push(dep);
        }
    }

    /// Add a dependent.
    pub fn add_dependent(&mut self, dep: PathBuf) {
        if !self.dependents.contains(&dep) {
            self.dependents.push(dep);
        }
    }

    /// Remove a dependency.
    pub fn remove_dependency(&mut self, dep: &Path) {
        self.dependencies.retain(|d| d != dep);
    }

    /// Remove a dependent.
    pub fn remove_dependent(&mut self, dep: &Path) {
        self.dependents.retain(|d| d != dep);
    }
}

// ---------------------------------------------------------------------------
// MaterialReloadManager
// ---------------------------------------------------------------------------

/// Manager for material instance hot-reload and dependency tracking.
///
/// Coordinates parameter updates, DSL file changes, and dependency graph
/// visualization for the material system.
pub struct MaterialReloadManager {
    /// Registered material instances.
    instances: HashMap<MaterialId, MaterialInstance>,
    /// Pending uniform buffer updates.
    pending_updates: VecDeque<MaterialReloadEvent>,
    /// Asset dependency graph.
    dependency_graph: HashMap<PathBuf, DependencyNode>,
    /// Mapping from shader path to material IDs using that shader.
    shader_to_materials: HashMap<PathBuf, Vec<MaterialId>>,
    /// DSL file extensions.
    dsl_extensions: Vec<String>,
    /// Optional shader reload manager integration.
    shader_manager: Option<Box<ShaderReloadManager>>,
    /// Callback for reload events.
    on_reload: Option<Box<dyn Fn(&MaterialReloadEvent) + Send + Sync>>,
}

impl MaterialReloadManager {
    /// Create a new material reload manager.
    pub fn new() -> Self {
        Self {
            instances: HashMap::new(),
            pending_updates: VecDeque::new(),
            dependency_graph: HashMap::new(),
            shader_to_materials: HashMap::new(),
            dsl_extensions: vec![
                "mat".to_string(),
                "material".to_string(),
                "mtl".to_string(),
            ],
            shader_manager: None,
            on_reload: None,
        }
    }

    /// Set the shader reload manager for DSL file changes.
    pub fn set_shader_manager(&mut self, manager: ShaderReloadManager) {
        self.shader_manager = Some(Box::new(manager));
    }

    /// Set a callback for reload events.
    pub fn on_reload<F>(&mut self, callback: F)
    where
        F: Fn(&MaterialReloadEvent) + Send + Sync + 'static,
    {
        self.on_reload = Some(Box::new(callback));
    }

    /// Register a material instance.
    pub fn register_material(&mut self, instance: MaterialInstance) {
        let shader_path = instance.shader_path.clone();
        let material_id = instance.id;

        // Update shader -> materials mapping
        self.shader_to_materials
            .entry(shader_path.clone())
            .or_default()
            .push(material_id);

        // Add to dependency graph
        let node = DependencyNode::with_type(shader_path.clone(), AssetType::Material);
        self.dependency_graph.insert(shader_path, node);

        // Register the instance
        self.instances.insert(material_id, instance);
    }

    /// Unregister a material instance.
    pub fn unregister_material(&mut self, material_id: MaterialId) {
        if let Some(instance) = self.instances.remove(&material_id) {
            // Remove from shader mapping
            if let Some(materials) = self.shader_to_materials.get_mut(&instance.shader_path) {
                materials.retain(|&id| id != material_id);
            }
        }
    }

    /// Get a material instance by ID.
    pub fn get_material(&self, material_id: MaterialId) -> Option<&MaterialInstance> {
        self.instances.get(&material_id)
    }

    /// Get a mutable material instance by ID.
    pub fn get_material_mut(&mut self, material_id: MaterialId) -> Option<&mut MaterialInstance> {
        self.instances.get_mut(&material_id)
    }

    /// Handle a parameter change.
    ///
    /// Updates the uniform buffer without triggering shader recompilation.
    pub fn on_parameter_change(
        &mut self,
        material_id: MaterialId,
        param_name: &str,
        new_value: MaterialValue,
    ) -> bool {
        if let Some(instance) = self.instances.get_mut(&material_id) {
            if instance.set_parameter(param_name, new_value) {
                // Queue update event
                let event = MaterialReloadEvent::parameter_update(
                    material_id,
                    vec![param_name.to_string()],
                );

                if let Some(ref callback) = self.on_reload {
                    callback(&event);
                }

                self.pending_updates.push_back(event);
                return true;
            }
        }
        false
    }

    /// Handle multiple parameter changes at once.
    pub fn on_parameters_change(
        &mut self,
        material_id: MaterialId,
        params: &[(String, MaterialValue)],
    ) -> bool {
        if let Some(instance) = self.instances.get_mut(&material_id) {
            let mut changed = Vec::new();
            for (name, value) in params {
                if instance.set_parameter(name, value.clone()) {
                    changed.push(name.clone());
                }
            }

            if !changed.is_empty() {
                let event = MaterialReloadEvent::parameter_update(material_id, changed);

                if let Some(ref callback) = self.on_reload {
                    callback(&event);
                }

                self.pending_updates.push_back(event);
                return true;
            }
        }
        false
    }

    /// Handle a DSL file change.
    ///
    /// Triggers shader regeneration through the shader hot-reload path.
    pub fn on_dsl_file_change(&mut self, path: &Path) {
        // Check if this is a DSL file
        if !self.is_dsl_file(path) {
            return;
        }

        // Find all materials using this shader
        let normalized = self.normalize_path(path);
        let affected_materials: Vec<MaterialId> = self
            .shader_to_materials
            .get(&normalized)
            .cloned()
            .unwrap_or_default();

        // Mark each material for shader reload
        for material_id in &affected_materials {
            let event = MaterialReloadEvent::shader_recompile(*material_id);

            if let Some(ref callback) = self.on_reload {
                callback(&event);
            }

            self.pending_updates.push_back(event);
        }

        // Update dependency graph
        if let Some(node) = self.dependency_graph.get_mut(&normalized) {
            node.last_modified = Some(Instant::now());
        }

        // If we have a shader manager, forward the change
        if let Some(ref mut shader_manager) = self.shader_manager {
            let content_change = ContentChange::new(
                normalized,
                ContentChangeKind::Modified,
                None,
                None,
            );
            shader_manager.on_content_change(&content_change);
        }
    }

    /// Get pending uniform buffer updates.
    ///
    /// Returns pairs of (MaterialId, buffer data) for GPU upload.
    /// Clears the dirty flag on each returned instance.
    pub fn push_uniform_updates(&mut self) -> Vec<(MaterialId, Vec<u8>)> {
        let mut updates = Vec::new();

        for (id, instance) in self.instances.iter_mut() {
            if instance.is_dirty() {
                updates.push((*id, instance.uniform_buffer.clone()));
                instance.clear_dirty();
            }
        }

        // Clear processed events
        self.pending_updates.retain(|e| {
            !matches!(e.status, ReloadStatus::Ready | ReloadStatus::Swapped | ReloadStatus::Failed)
        });

        updates
    }

    /// Get the dependency graph for a root asset.
    pub fn get_dependency_graph(&self, root: &Path) -> DependencyNode {
        let normalized = self.normalize_path(root);

        self.dependency_graph
            .get(&normalized)
            .cloned()
            .unwrap_or_else(|| DependencyNode::new(normalized))
    }

    /// Get the propagation path for a change.
    ///
    /// Returns all assets affected by a change to the given path,
    /// in the order they would be processed (BFS from root).
    pub fn get_propagation_path(&self, changed: &Path) -> Vec<PathBuf> {
        let normalized = self.normalize_path(changed);
        let mut visited = HashSet::new();
        let mut result = Vec::new();
        let mut queue = VecDeque::new();

        // Start with the changed file
        queue.push_back(normalized.clone());
        visited.insert(normalized);

        // BFS through dependents
        while let Some(current) = queue.pop_front() {
            result.push(current.clone());

            if let Some(node) = self.dependency_graph.get(&current) {
                for dependent in &node.dependents {
                    if !visited.contains(dependent) {
                        visited.insert(dependent.clone());
                        queue.push_back(dependent.clone());
                    }
                }
            }
        }

        result
    }

    /// Add a dependency edge to the graph.
    pub fn add_dependency(&mut self, from: &Path, to: &Path) {
        let from_normalized = self.normalize_path(from);
        let to_normalized = self.normalize_path(to);

        // Ensure both nodes exist
        if !self.dependency_graph.contains_key(&from_normalized) {
            self.dependency_graph
                .insert(from_normalized.clone(), DependencyNode::new(from_normalized.clone()));
        }
        if !self.dependency_graph.contains_key(&to_normalized) {
            self.dependency_graph
                .insert(to_normalized.clone(), DependencyNode::new(to_normalized.clone()));
        }

        // Add forward edge (from depends on to)
        if let Some(node) = self.dependency_graph.get_mut(&from_normalized) {
            node.add_dependency(to_normalized.clone());
        }

        // Add reverse edge (to has from as dependent)
        if let Some(node) = self.dependency_graph.get_mut(&to_normalized) {
            node.add_dependent(from_normalized);
        }
    }

    /// Remove a dependency edge from the graph.
    pub fn remove_dependency(&mut self, from: &Path, to: &Path) {
        let from_normalized = self.normalize_path(from);
        let to_normalized = self.normalize_path(to);

        if let Some(node) = self.dependency_graph.get_mut(&from_normalized) {
            node.remove_dependency(&to_normalized);
        }

        if let Some(node) = self.dependency_graph.get_mut(&to_normalized) {
            node.remove_dependent(&from_normalized);
        }
    }

    /// Get all materials affected by a change to the given path.
    pub fn get_affected_materials(&self, path: &Path) -> Vec<MaterialId> {
        let propagation = self.get_propagation_path(path);
        let mut affected = Vec::new();

        for path in propagation {
            if let Some(materials) = self.shader_to_materials.get(&path) {
                for &material_id in materials {
                    if !affected.contains(&material_id) {
                        affected.push(material_id);
                    }
                }
            }
        }

        affected
    }

    /// Poll for pending reload events.
    pub fn poll_events(&mut self) -> Vec<MaterialReloadEvent> {
        let events: Vec<_> = self.pending_updates.drain(..).collect();
        events
    }

    /// Get the number of registered materials.
    pub fn material_count(&self) -> usize {
        self.instances.len()
    }

    /// Get the number of pending updates.
    pub fn pending_count(&self) -> usize {
        self.pending_updates.len()
    }

    /// Get the number of nodes in the dependency graph.
    pub fn graph_node_count(&self) -> usize {
        self.dependency_graph.len()
    }

    /// Check if a path is a DSL file.
    fn is_dsl_file(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| self.dsl_extensions.iter().any(|ext| ext == e))
            .unwrap_or(false)
    }

    /// Normalize a path for consistent lookup.
    fn normalize_path(&self, path: &Path) -> PathBuf {
        path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
    }

    /// Clear all state.
    pub fn clear(&mut self) {
        self.instances.clear();
        self.pending_updates.clear();
        self.dependency_graph.clear();
        self.shader_to_materials.clear();
    }
}

impl Default for MaterialReloadManager {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for MaterialReloadManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MaterialReloadManager")
            .field("instances", &self.instances.len())
            .field("pending_updates", &self.pending_updates.len())
            .field("dependency_graph", &self.dependency_graph.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create a test material instance
    fn create_test_material(id: u64, shader_path: &str) -> MaterialInstance {
        let mut instance = MaterialInstance::new(
            MaterialId::from_raw(id),
            PathBuf::from(shader_path),
        );

        // Add some test parameters
        instance.add_parameter(MaterialParameter::new(
            "base_color",
            MaterialValue::Color([1.0, 0.5, 0.25, 1.0]),
            0,
        ));
        instance.add_parameter(MaterialParameter::new(
            "metallic",
            MaterialValue::Float(0.5),
            16,
        ));
        instance.add_parameter(MaterialParameter::new(
            "roughness",
            MaterialValue::Float(0.3),
            20,
        ));
        instance.add_parameter(MaterialParameter::new(
            "uv_scale",
            MaterialValue::Vec2([1.0, 1.0]),
            24,
        ));

        instance
    }

    // ========================================================================
    // Parameter Update Tests (5+)
    // ========================================================================

    #[test]
    fn test_parameter_update_float() {
        let mut manager = MaterialReloadManager::new();
        let material = create_test_material(1, "materials/test.mat");
        let material_id = material.id;

        manager.register_material(material);

        assert!(manager.on_parameter_change(material_id, "metallic", MaterialValue::Float(0.8)));

        let instance = manager.get_material(material_id).unwrap();
        let param = instance.get_parameter("metallic").unwrap();
        assert_eq!(param.value, MaterialValue::Float(0.8));
        assert!(instance.is_dirty());
    }

    #[test]
    fn test_parameter_update_vec4() {
        let mut manager = MaterialReloadManager::new();
        let material = create_test_material(1, "materials/test.mat");
        let material_id = material.id;

        manager.register_material(material);

        let new_color = MaterialValue::Color([0.0, 1.0, 0.0, 1.0]);
        assert!(manager.on_parameter_change(material_id, "base_color", new_color.clone()));

        let instance = manager.get_material(material_id).unwrap();
        let param = instance.get_parameter("base_color").unwrap();
        assert_eq!(param.value, new_color);
    }

    #[test]
    fn test_parameter_update_color() {
        let mut manager = MaterialReloadManager::new();
        let material = create_test_material(1, "materials/test.mat");
        let material_id = material.id;

        manager.register_material(material);

        let new_color = MaterialValue::Color([1.0, 0.0, 0.0, 1.0]);
        assert!(manager.on_parameter_change(material_id, "base_color", new_color.clone()));

        // Verify buffer was updated
        let instance = manager.get_material(material_id).unwrap();
        let buffer = instance.uniform_data();

        // First 4 floats should be the color (16 bytes)
        let r = f32::from_le_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]);
        assert!((r - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_parameter_update_int() {
        let mut manager = MaterialReloadManager::new();
        let mut material = create_test_material(1, "materials/test.mat");
        material.add_parameter(MaterialParameter::new(
            "layer_count",
            MaterialValue::Int(4),
            32,
        ));
        let material_id = material.id;

        manager.register_material(material);

        assert!(manager.on_parameter_change(material_id, "layer_count", MaterialValue::Int(8)));

        let instance = manager.get_material(material_id).unwrap();
        let param = instance.get_parameter("layer_count").unwrap();
        assert_eq!(param.value, MaterialValue::Int(8));
    }

    #[test]
    fn test_parameter_update_multiple_at_once() {
        let mut manager = MaterialReloadManager::new();
        let material = create_test_material(1, "materials/test.mat");
        let material_id = material.id;

        manager.register_material(material);

        let params = vec![
            ("metallic".to_string(), MaterialValue::Float(0.9)),
            ("roughness".to_string(), MaterialValue::Float(0.1)),
        ];

        assert!(manager.on_parameters_change(material_id, &params));

        let instance = manager.get_material(material_id).unwrap();
        assert_eq!(
            instance.get_parameter("metallic").unwrap().value,
            MaterialValue::Float(0.9)
        );
        assert_eq!(
            instance.get_parameter("roughness").unwrap().value,
            MaterialValue::Float(0.1)
        );
    }

    #[test]
    fn test_parameter_update_bool() {
        let mut manager = MaterialReloadManager::new();
        let mut material = create_test_material(1, "materials/test.mat");
        material.add_parameter(MaterialParameter::new(
            "use_normal_map",
            MaterialValue::Bool(false),
            36,
        ));
        let material_id = material.id;

        manager.register_material(material);

        assert!(manager.on_parameter_change(material_id, "use_normal_map", MaterialValue::Bool(true)));

        let instance = manager.get_material(material_id).unwrap();
        let param = instance.get_parameter("use_normal_map").unwrap();
        assert_eq!(param.value, MaterialValue::Bool(true));
    }

    // ========================================================================
    // Uniform Buffer Tests (4+)
    // ========================================================================

    #[test]
    fn test_uniform_buffer_layout() {
        let material = create_test_material(1, "materials/test.mat");

        // Verify buffer layout
        assert!(material.uniform_buffer.len() >= 32); // base_color(16) + metallic(4) + roughness(4) + uv_scale(8)

        // Check base_color at offset 0
        let param = material.get_parameter("base_color").unwrap();
        assert_eq!(param.offset, 0);
        assert_eq!(param.size, 16);

        // Check metallic at offset 16
        let param = material.get_parameter("metallic").unwrap();
        assert_eq!(param.offset, 16);
        assert_eq!(param.size, 4);
    }

    #[test]
    fn test_uniform_buffer_offset_calculation() {
        let mut instance = MaterialInstance::new(MaterialId::from_raw(1), PathBuf::from("test.mat"));

        // Add parameters at specific offsets
        instance.add_parameter(MaterialParameter::new("a", MaterialValue::Float(1.0), 0));
        instance.add_parameter(MaterialParameter::new("b", MaterialValue::Vec2([2.0, 3.0]), 4));
        instance.add_parameter(MaterialParameter::new("c", MaterialValue::Vec4([4.0, 5.0, 6.0, 7.0]), 16));

        // Verify buffer size
        assert!(instance.uniform_buffer.len() >= 32); // 16 offset + 16 size for vec4

        // Verify values at offsets
        let a = f32::from_le_bytes(instance.uniform_buffer[0..4].try_into().unwrap());
        assert!((a - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_uniform_buffer_push_operation() {
        let mut manager = MaterialReloadManager::new();
        let material = create_test_material(1, "materials/test.mat");
        let material_id = material.id;

        manager.register_material(material);

        // Update a parameter
        manager.on_parameter_change(material_id, "metallic", MaterialValue::Float(0.75));

        // Push updates
        let updates = manager.push_uniform_updates();

        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].0, material_id);
        assert!(!updates[0].1.is_empty());

        // Verify dirty flag was cleared
        let instance = manager.get_material(material_id).unwrap();
        assert!(!instance.is_dirty());
    }

    #[test]
    fn test_uniform_buffer_multiple_materials() {
        let mut manager = MaterialReloadManager::new();

        for i in 0..3 {
            let material = create_test_material(i, &format!("materials/test_{}.mat", i));
            manager.register_material(material);
        }

        // Update all materials
        for i in 0..3 {
            manager.on_parameter_change(
                MaterialId::from_raw(i),
                "metallic",
                MaterialValue::Float(i as f32 * 0.1),
            );
        }

        let updates = manager.push_uniform_updates();
        assert_eq!(updates.len(), 3);
    }

    // ========================================================================
    // DSL File Change Tests (3+)
    // ========================================================================

    #[test]
    fn test_dsl_change_detect() {
        let mut manager = MaterialReloadManager::new();
        let material = create_test_material(1, "materials/test.mat");
        let material_id = material.id;

        manager.register_material(material);

        // Simulate DSL file change
        manager.on_dsl_file_change(Path::new("materials/test.mat"));

        // Should have a pending shader recompile event
        let events = manager.poll_events();
        assert_eq!(events.len(), 1);
        assert!(events[0].kind.requires_shader_recompile());
        assert_eq!(events[0].material_id, material_id);
    }

    #[test]
    fn test_dsl_change_trigger_recompile() {
        let mut manager = MaterialReloadManager::new();

        // Register multiple materials with same shader
        for i in 0..3 {
            let mut material = create_test_material(i, "materials/shared.mat");
            material.id = MaterialId::from_raw(i);
            manager.register_material(material);
        }

        manager.on_dsl_file_change(Path::new("materials/shared.mat"));

        // All materials should have pending events
        let events = manager.poll_events();
        assert_eq!(events.len(), 3);
        for event in events {
            assert!(matches!(event.kind, MaterialReloadKind::ShaderRecompile));
        }
    }

    #[test]
    fn test_dsl_change_fallback_on_error() {
        let mut manager = MaterialReloadManager::new();
        let material = create_test_material(1, "materials/test.mat");
        let material_id = material.id;
        let original_buffer = material.uniform_buffer.clone();

        manager.register_material(material);

        // DSL change queues recompile but doesn't modify buffer yet
        manager.on_dsl_file_change(Path::new("materials/test.mat"));

        // Original buffer should be unchanged (fallback behavior)
        let instance = manager.get_material(material_id).unwrap();
        assert_eq!(instance.uniform_buffer, original_buffer);
    }

    // ========================================================================
    // Dependency Graph Tests (4+)
    // ========================================================================

    #[test]
    fn test_dependency_graph_build() {
        let mut manager = MaterialReloadManager::new();

        // Add dependency edges
        manager.add_dependency(
            Path::new("materials/metal.mat"),
            Path::new("shaders/pbr.wgsl"),
        );
        manager.add_dependency(
            Path::new("materials/metal.mat"),
            Path::new("textures/metal_albedo.png"),
        );

        let node = manager.get_dependency_graph(Path::new("materials/metal.mat"));
        assert_eq!(node.dependencies.len(), 2);
    }

    #[test]
    fn test_dependency_graph_query_dependencies() {
        let mut manager = MaterialReloadManager::new();

        manager.add_dependency(Path::new("a"), Path::new("b"));
        manager.add_dependency(Path::new("a"), Path::new("c"));
        manager.add_dependency(Path::new("b"), Path::new("d"));

        let node = manager.get_dependency_graph(Path::new("a"));
        assert_eq!(node.dependencies.len(), 2);
        assert!(node.dependencies.iter().any(|p| p.ends_with("b")));
        assert!(node.dependencies.iter().any(|p| p.ends_with("c")));
    }

    #[test]
    fn test_dependency_graph_query_dependents() {
        let mut manager = MaterialReloadManager::new();

        // a depends on b, c depends on b
        manager.add_dependency(Path::new("a"), Path::new("b"));
        manager.add_dependency(Path::new("c"), Path::new("b"));

        let node = manager.get_dependency_graph(Path::new("b"));
        assert_eq!(node.dependents.len(), 2);
    }

    #[test]
    fn test_dependency_graph_remove_edge() {
        let mut manager = MaterialReloadManager::new();

        manager.add_dependency(Path::new("a"), Path::new("b"));
        manager.add_dependency(Path::new("a"), Path::new("c"));

        // Remove one dependency
        manager.remove_dependency(Path::new("a"), Path::new("b"));

        let node = manager.get_dependency_graph(Path::new("a"));
        assert_eq!(node.dependencies.len(), 1);
        assert!(node.dependencies.iter().any(|p| p.ends_with("c")));
    }

    // ========================================================================
    // Propagation Path Tests (2+)
    // ========================================================================

    #[test]
    fn test_propagation_path_trace_impact() {
        let mut manager = MaterialReloadManager::new();

        // Build a dependency chain: d <- c <- b <- a
        manager.add_dependency(Path::new("a"), Path::new("b"));
        manager.add_dependency(Path::new("b"), Path::new("c"));
        manager.add_dependency(Path::new("c"), Path::new("d"));

        // Changing 'd' should propagate to c, b, a
        let path = manager.get_propagation_path(Path::new("d"));

        // Path should start with d and include dependents
        assert!(!path.is_empty());
        assert!(path[0].ends_with("d"));
    }

    #[test]
    fn test_propagation_path_handle_cycles() {
        let mut manager = MaterialReloadManager::new();

        // Create a cycle: a -> b -> c -> a
        manager.add_dependency(Path::new("a"), Path::new("b"));
        manager.add_dependency(Path::new("b"), Path::new("c"));
        manager.add_dependency(Path::new("c"), Path::new("a"));

        // Should not hang/panic
        let path = manager.get_propagation_path(Path::new("a"));

        // Each node should appear only once
        let mut seen = HashSet::new();
        for p in &path {
            assert!(seen.insert(p.clone()), "Cycle detected: {:?} appeared twice", p);
        }
    }

    // ========================================================================
    // Edge Cases Tests (2+)
    // ========================================================================

    #[test]
    fn test_edge_case_invalid_material() {
        let mut manager = MaterialReloadManager::new();

        // Try to update a non-existent material
        let result = manager.on_parameter_change(
            MaterialId::from_raw(999),
            "metallic",
            MaterialValue::Float(0.5),
        );

        assert!(!result);
    }

    #[test]
    fn test_edge_case_missing_parameter() {
        let mut manager = MaterialReloadManager::new();
        let material = create_test_material(1, "materials/test.mat");
        let material_id = material.id;

        manager.register_material(material);

        // Try to update a non-existent parameter
        let result = manager.on_parameter_change(
            material_id,
            "nonexistent_param",
            MaterialValue::Float(0.5),
        );

        assert!(!result);
    }

    // ========================================================================
    // MaterialValue Tests
    // ========================================================================

    #[test]
    fn test_material_value_size_bytes() {
        assert_eq!(MaterialValue::Float(0.0).size_bytes(), 4);
        assert_eq!(MaterialValue::Vec2([0.0, 0.0]).size_bytes(), 8);
        assert_eq!(MaterialValue::Vec3([0.0, 0.0, 0.0]).size_bytes(), 12);
        assert_eq!(MaterialValue::Vec4([0.0, 0.0, 0.0, 0.0]).size_bytes(), 16);
        assert_eq!(MaterialValue::Int(0).size_bytes(), 4);
        assert_eq!(MaterialValue::Bool(false).size_bytes(), 4);
        assert_eq!(MaterialValue::Color([0.0, 0.0, 0.0, 0.0]).size_bytes(), 16);
    }

    #[test]
    fn test_material_value_write_read_roundtrip() {
        let values = vec![
            MaterialValue::Float(3.14159),
            MaterialValue::Vec2([1.0, 2.0]),
            MaterialValue::Vec3([1.0, 2.0, 3.0]),
            MaterialValue::Vec4([1.0, 2.0, 3.0, 4.0]),
            MaterialValue::Int(-42),
            MaterialValue::Bool(true),
            MaterialValue::Color([0.5, 0.6, 0.7, 0.8]),
        ];

        for original in values {
            let mut buffer = vec![0u8; 16];
            original.write_to_buffer(&mut buffer, 0);

            let read_back = MaterialValue::read_from_buffer(&buffer, 0, &original).unwrap();
            assert_eq!(original, read_back);
        }
    }

    #[test]
    fn test_material_value_vec3_write() {
        let mut buffer = vec![0u8; 16];
        let value = MaterialValue::Vec3([1.0, 2.0, 3.0]);
        value.write_to_buffer(&mut buffer, 0);

        let x = f32::from_le_bytes(buffer[0..4].try_into().unwrap());
        let y = f32::from_le_bytes(buffer[4..8].try_into().unwrap());
        let z = f32::from_le_bytes(buffer[8..12].try_into().unwrap());

        assert!((x - 1.0).abs() < 0.001);
        assert!((y - 2.0).abs() < 0.001);
        assert!((z - 3.0).abs() < 0.001);
    }

    // ========================================================================
    // MaterialInstance Tests
    // ========================================================================

    #[test]
    fn test_material_instance_aligned_buffer_size() {
        let mut instance = MaterialInstance::new(MaterialId::from_raw(1), PathBuf::from("test.mat"));

        // Empty buffer should have aligned size of 0
        assert_eq!(instance.aligned_buffer_size(), 0);

        // Add a single float (4 bytes) - aligned to 16
        instance.add_parameter(MaterialParameter::new("a", MaterialValue::Float(1.0), 0));
        assert_eq!(instance.aligned_buffer_size(), 16);

        // Add more to go beyond 16 bytes
        instance.add_parameter(MaterialParameter::new("b", MaterialValue::Vec4([0.0; 4]), 16));
        assert_eq!(instance.aligned_buffer_size(), 32);
    }

    #[test]
    fn test_material_instance_clear_dirty() {
        let mut instance = MaterialInstance::new(MaterialId::from_raw(1), PathBuf::from("test.mat"));
        instance.add_parameter(MaterialParameter::new("a", MaterialValue::Float(1.0), 0));

        instance.set_parameter("a", MaterialValue::Float(2.0));
        assert!(instance.is_dirty());

        instance.clear_dirty();
        assert!(!instance.is_dirty());
    }

    // ========================================================================
    // AssetType Tests
    // ========================================================================

    #[test]
    fn test_asset_type_from_extension() {
        assert_eq!(AssetType::from_extension("mat"), AssetType::Material);
        assert_eq!(AssetType::from_extension("wgsl"), AssetType::Shader);
        assert_eq!(AssetType::from_extension("png"), AssetType::Texture);
        assert_eq!(AssetType::from_extension("inc"), AssetType::Include);
        assert_eq!(AssetType::from_extension("xyz"), AssetType::Unknown);
    }

    #[test]
    fn test_asset_type_from_path() {
        assert_eq!(
            AssetType::from_path(Path::new("materials/test.mat")),
            AssetType::Material
        );
        assert_eq!(
            AssetType::from_path(Path::new("shaders/pbr.wgsl")),
            AssetType::Shader
        );
        assert_eq!(
            AssetType::from_path(Path::new("textures/albedo.png")),
            AssetType::Texture
        );
    }

    // ========================================================================
    // MaterialReloadKind Tests
    // ========================================================================

    #[test]
    fn test_reload_kind_is_parameter_update() {
        let param_update = MaterialReloadKind::ParameterUpdate {
            changed: vec!["metallic".to_string()],
        };
        assert!(param_update.is_parameter_update());
        assert!(!param_update.requires_shader_recompile());

        let shader_recompile = MaterialReloadKind::ShaderRecompile;
        assert!(!shader_recompile.is_parameter_update());
        assert!(shader_recompile.requires_shader_recompile());
    }

    // ========================================================================
    // Manager State Tests
    // ========================================================================

    #[test]
    fn test_manager_clear() {
        let mut manager = MaterialReloadManager::new();

        let material = create_test_material(1, "materials/test.mat");
        manager.register_material(material);
        manager.add_dependency(Path::new("a"), Path::new("b"));

        assert_eq!(manager.material_count(), 1);
        assert!(manager.graph_node_count() > 0);

        manager.clear();

        assert_eq!(manager.material_count(), 0);
        assert_eq!(manager.graph_node_count(), 0);
    }

    #[test]
    fn test_manager_debug_impl() {
        let manager = MaterialReloadManager::new();
        let debug = format!("{:?}", manager);

        assert!(debug.contains("MaterialReloadManager"));
        assert!(debug.contains("instances"));
    }

    #[test]
    fn test_material_id_display() {
        let id = MaterialId::from_raw(0x1234567890ABCDEF);
        let display = format!("{}", id);
        assert!(display.contains("1234567890abcdef"));
    }

    #[test]
    fn test_material_id_from_path() {
        let id1 = MaterialId::from_path(Path::new("materials/a.mat"));
        let id2 = MaterialId::from_path(Path::new("materials/a.mat"));
        let id3 = MaterialId::from_path(Path::new("materials/b.mat"));

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_unregister_material() {
        let mut manager = MaterialReloadManager::new();

        let material = create_test_material(1, "materials/test.mat");
        let material_id = material.id;

        manager.register_material(material);
        assert_eq!(manager.material_count(), 1);

        manager.unregister_material(material_id);
        assert_eq!(manager.material_count(), 0);
        assert!(manager.get_material(material_id).is_none());
    }

    #[test]
    fn test_get_affected_materials() {
        let mut manager = MaterialReloadManager::new();

        let material = create_test_material(1, "materials/metal.mat");
        manager.register_material(material);

        // Add dependency from material to shader
        manager.add_dependency(
            Path::new("materials/metal.mat"),
            Path::new("shaders/pbr.wgsl"),
        );

        let affected = manager.get_affected_materials(Path::new("materials/metal.mat"));
        assert_eq!(affected.len(), 1);
        assert_eq!(affected[0], MaterialId::from_raw(1));
    }

    #[test]
    fn test_dependency_node_new() {
        let node = DependencyNode::new(PathBuf::from("test.wgsl"));
        assert_eq!(node.asset_type, AssetType::Shader);
        assert!(node.dependencies.is_empty());
        assert!(node.dependents.is_empty());
    }
}
