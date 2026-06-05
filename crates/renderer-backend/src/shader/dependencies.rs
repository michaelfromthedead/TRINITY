//! Shader Dependency Extraction for TRINITY (T-AS-3.5).
//!
//! Provides dependency graph construction and invalidation tracking for shader
//! hot-reload and caching systems:
//!
//! - **Direct #include Extraction**: Parse WGSL source for `#include "path"` directives
//! - **Transitive Dependencies**: Recursively follow includes to build complete dependency tree
//! - **@import Module References**: Extract WGSL `@import module::path` declarations
//! - **Material DSL Dependencies**: Track Python .py material generator files
//! - **Search Path Resolution**: Resolve paths via configurable mount points
//! - **Content Hash Per Dependency**: Compute BLAKE3/SHA-256 hash of each dependency file
//! - **Dependency Tree Storage**: Serialize full tree alongside compiled output
//! - **Invalidation Propagation**: Mark all dependent shaders stale when a dependency changes
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::shader::dependencies::ShaderDependencyGraph;
//!
//! let mut graph = ShaderDependencyGraph::new(vec![
//!     "shaders/".into(),
//!     "shaders/includes/".into(),
//! ]);
//!
//! // Analyze a shader and its dependencies
//! let node = graph.analyze("shaders/pbr.wgsl")?;
//! println!("Direct includes: {:?}", node.includes);
//!
//! // Get all transitive dependencies
//! let deps = graph.get_all_dependencies("shaders/pbr.wgsl");
//! for dep in deps {
//!     println!("{}: {}", dep.path.display(), dep.content_hash);
//! }
//!
//! // When a dependency changes, find affected shaders
//! let affected = graph.invalidate("shaders/includes/common.wgsl");
//! for shader in affected {
//!     println!("Needs recompilation: {}", shader.display());
//! }
//! ```

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::pipeline::ContentHash;

// ---------------------------------------------------------------------------
// ContentHash Serialization Helper
// ---------------------------------------------------------------------------

/// Serialize ContentHash as a hex string.
fn serialize_content_hash<S>(hash: &ContentHash, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_str(&format!("{}", hash))
}

/// Deserialize ContentHash from a hex string.
fn deserialize_content_hash<'de, D>(deserializer: D) -> Result<ContentHash, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    s.parse::<ContentHash>().map_err(serde::de::Error::custom)
}

// ---------------------------------------------------------------------------
// Error Types
// ---------------------------------------------------------------------------

/// Errors that can occur during dependency extraction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DependencyError {
    /// File not found in any search path.
    FileNotFound {
        path: String,
        search_paths: Vec<String>,
    },
    /// Circular dependency detected.
    CircularDependency {
        path: String,
        dependency_chain: Vec<String>,
    },
    /// IO error reading file.
    IoError { path: String, message: String },
    /// Invalid path format.
    InvalidPath { path: String, message: String },
    /// Parse error in source file.
    ParseError {
        path: String,
        line: usize,
        message: String,
    },
}

impl std::fmt::Display for DependencyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileNotFound { path, search_paths } => {
                write!(
                    f,
                    "file not found: '{}' (searched: {:?})",
                    path, search_paths
                )
            }
            Self::CircularDependency {
                path,
                dependency_chain,
            } => {
                write!(
                    f,
                    "circular dependency detected: {} (chain: {})",
                    path,
                    dependency_chain.join(" -> ")
                )
            }
            Self::IoError { path, message } => {
                write!(f, "IO error reading '{}': {}", path, message)
            }
            Self::InvalidPath { path, message } => {
                write!(f, "invalid path '{}': {}", path, message)
            }
            Self::ParseError {
                path,
                line,
                message,
            } => {
                write!(f, "{}:{}: parse error: {}", path, line, message)
            }
        }
    }
}

impl std::error::Error for DependencyError {}

impl From<io::Error> for DependencyError {
    fn from(e: io::Error) -> Self {
        Self::IoError {
            path: String::new(),
            message: e.to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// DependencyNode
// ---------------------------------------------------------------------------

/// A node in the shader dependency graph representing a single file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyNode {
    /// Absolute path to the file.
    pub path: PathBuf,
    /// BLAKE3/SHA-256 content hash of the file.
    #[serde(
        serialize_with = "serialize_content_hash",
        deserialize_with = "deserialize_content_hash"
    )]
    pub content_hash: ContentHash,
    /// Direct #include paths (resolved to absolute paths).
    pub includes: Vec<PathBuf>,
    /// @import module names (e.g., "utils::math", "common::lighting").
    pub imports: Vec<String>,
    /// Python material DSL files that generate this shader.
    pub material_files: Vec<PathBuf>,
    /// Files that depend on this file (reverse dependencies).
    pub dependents: Vec<PathBuf>,
    /// Whether this node has been analyzed (dependencies extracted).
    pub analyzed: bool,
    /// Timestamp of last analysis (Unix epoch milliseconds).
    pub last_analyzed: u64,
}

impl DependencyNode {
    /// Create a new unanalyzed dependency node.
    pub fn new(path: PathBuf, content_hash: ContentHash) -> Self {
        Self {
            path,
            content_hash,
            includes: Vec::new(),
            imports: Vec::new(),
            material_files: Vec::new(),
            dependents: Vec::new(),
            analyzed: false,
            last_analyzed: 0,
        }
    }

    /// Create a fully-analyzed dependency node.
    pub fn analyzed(
        path: PathBuf,
        content_hash: ContentHash,
        includes: Vec<PathBuf>,
        imports: Vec<String>,
        material_files: Vec<PathBuf>,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            path,
            content_hash,
            includes,
            imports,
            material_files,
            dependents: Vec::new(),
            analyzed: true,
            last_analyzed: now,
        }
    }

    /// Get all direct dependencies (includes + material files).
    pub fn direct_dependencies(&self) -> Vec<&PathBuf> {
        let mut deps: Vec<&PathBuf> = self.includes.iter().collect();
        deps.extend(self.material_files.iter());
        deps
    }

    /// Check if this node has any dependencies.
    pub fn has_dependencies(&self) -> bool {
        !self.includes.is_empty() || !self.imports.is_empty() || !self.material_files.is_empty()
    }
}

// ---------------------------------------------------------------------------
// ShaderDependencyGraph
// ---------------------------------------------------------------------------

/// Complete dependency graph for shader hot-reload and caching.
///
/// The graph tracks:
/// - All shader files and their dependencies
/// - Reverse dependency mapping for invalidation
/// - Content hashes for change detection
/// - Search paths for include resolution
pub struct ShaderDependencyGraph {
    /// Root directory for relative path resolution.
    root: PathBuf,
    /// All dependency nodes keyed by absolute path.
    dependencies: HashMap<PathBuf, DependencyNode>,
    /// Search paths for resolving includes (in priority order).
    search_paths: Vec<PathBuf>,
    /// Custom file reader for testing.
    #[allow(clippy::type_complexity)]
    file_reader: Option<Arc<dyn Fn(&Path) -> io::Result<String> + Send + Sync>>,
}

impl std::fmt::Debug for ShaderDependencyGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShaderDependencyGraph")
            .field("root", &self.root)
            .field("dependencies", &self.dependencies)
            .field("search_paths", &self.search_paths)
            .field("file_reader", &self.file_reader.as_ref().map(|_| "<fn>"))
            .finish()
    }
}

impl Clone for ShaderDependencyGraph {
    fn clone(&self) -> Self {
        Self {
            root: self.root.clone(),
            dependencies: self.dependencies.clone(),
            search_paths: self.search_paths.clone(),
            file_reader: self.file_reader.clone(),
        }
    }
}

impl ShaderDependencyGraph {
    /// Create a new dependency graph with the given search paths.
    pub fn new(search_paths: Vec<PathBuf>) -> Self {
        Self {
            root: PathBuf::new(),
            dependencies: HashMap::new(),
            search_paths,
            file_reader: None,
        }
    }

    /// Create a new dependency graph with a root directory.
    pub fn with_root<P: AsRef<Path>>(root: P, search_paths: Vec<PathBuf>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
            dependencies: HashMap::new(),
            search_paths,
            file_reader: None,
        }
    }

    /// Set a custom file reader (for testing).
    pub fn set_file_reader<F>(&mut self, reader: F)
    where
        F: Fn(&Path) -> io::Result<String> + Send + Sync + 'static,
    {
        self.file_reader = Some(Arc::new(reader));
    }

    /// Get the search paths.
    pub fn search_paths(&self) -> &[PathBuf] {
        &self.search_paths
    }

    /// Add a search path.
    pub fn add_search_path<P: AsRef<Path>>(&mut self, path: P) {
        self.search_paths.push(path.as_ref().to_path_buf());
    }

    /// Read file content using the configured reader or filesystem.
    fn read_file(&self, path: &Path) -> io::Result<String> {
        if let Some(ref reader) = self.file_reader {
            reader(path)
        } else {
            fs::read_to_string(path)
        }
    }

    /// Check if a file exists using the configured reader or filesystem.
    fn file_exists(&self, path: &Path) -> bool {
        if let Some(ref reader) = self.file_reader {
            reader(path).is_ok()
        } else {
            path.exists()
        }
    }

    /// Compute content hash for a file.
    fn compute_hash(&self, path: &Path) -> Result<ContentHash, DependencyError> {
        let content = self.read_file(path).map_err(|e| DependencyError::IoError {
            path: path.display().to_string(),
            message: e.to_string(),
        })?;
        Ok(ContentHash::from_bytes(content.as_bytes()))
    }

    /// Resolve an include path using search paths.
    ///
    /// Resolution order:
    /// 1. Relative to the including file's directory
    /// 2. Each search path in order
    /// 3. Relative to root directory
    pub fn resolve_path(
        &self,
        include_path: &str,
        from_file: &Path,
    ) -> Result<PathBuf, DependencyError> {
        // 1. Try relative to current file
        if let Some(parent) = from_file.parent() {
            let relative = parent.join(include_path);
            if self.file_exists(&relative) {
                return Ok(self.canonicalize(&relative));
            }
        }

        // 2. Try each search path
        for search_path in &self.search_paths {
            let full_path = search_path.join(include_path);
            if self.file_exists(&full_path) {
                return Ok(self.canonicalize(&full_path));
            }
        }

        // 3. Try relative to root
        if !self.root.as_os_str().is_empty() {
            let root_relative = self.root.join(include_path);
            if self.file_exists(&root_relative) {
                return Ok(self.canonicalize(&root_relative));
            }
        }

        // 4. Try as absolute path
        let absolute = PathBuf::from(include_path);
        if self.file_exists(&absolute) {
            return Ok(self.canonicalize(&absolute));
        }

        Err(DependencyError::FileNotFound {
            path: include_path.to_string(),
            search_paths: self
                .search_paths
                .iter()
                .map(|p| p.display().to_string())
                .collect(),
        })
    }

    /// Canonicalize a path (or return as-is if canonicalization fails).
    fn canonicalize(&self, path: &Path) -> PathBuf {
        if self.file_reader.is_some() {
            // When using a custom reader, don't try to canonicalize
            path.to_path_buf()
        } else {
            path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
        }
    }

    /// Normalize a path for consistent keys.
    fn normalize_path(&self, path: &Path) -> PathBuf {
        self.canonicalize(path)
    }

    /// Analyze a shader file and extract its dependencies.
    ///
    /// This parses the source for:
    /// - `#include "path"` directives
    /// - `@import module::path` declarations
    ///
    /// Dependencies are recursively analyzed.
    pub fn analyze<P: AsRef<Path>>(
        &mut self,
        shader_path: P,
    ) -> Result<&DependencyNode, DependencyError> {
        let shader_path = shader_path.as_ref();
        let normalized = self.normalize_path(shader_path);

        // Check if already analyzed and up-to-date
        if let Some(node) = self.dependencies.get(&normalized) {
            if node.analyzed {
                // Verify hash hasn't changed
                let current_hash = self.compute_hash(&normalized)?;
                if current_hash == node.content_hash {
                    return Ok(self.dependencies.get(&normalized).unwrap());
                }
            }
        }

        // Perform analysis with cycle detection
        let mut visited = HashSet::new();
        self.analyze_recursive(&normalized, &mut visited)?;

        Ok(self.dependencies.get(&normalized).unwrap())
    }

    /// Recursive analysis with cycle detection.
    fn analyze_recursive(
        &mut self,
        path: &Path,
        visited: &mut HashSet<PathBuf>,
    ) -> Result<(), DependencyError> {
        let normalized = self.normalize_path(path);

        // Cycle detection
        if visited.contains(&normalized) {
            return Err(DependencyError::CircularDependency {
                path: normalized.display().to_string(),
                dependency_chain: visited.iter().map(|p| p.display().to_string()).collect(),
            });
        }
        visited.insert(normalized.clone());

        // Read and parse the file
        let content = self.read_file(&normalized).map_err(|e| DependencyError::IoError {
            path: normalized.display().to_string(),
            message: e.to_string(),
        })?;
        let content_hash = ContentHash::from_bytes(content.as_bytes());

        // Extract dependencies
        let includes = self.extract_includes(&content, &normalized)?;
        let imports = self.extract_imports(&content);
        let material_files = self.extract_material_dependencies(&content, &normalized)?;

        // Create the node
        let node = DependencyNode::analyzed(
            normalized.clone(),
            content_hash,
            includes.clone(),
            imports,
            material_files.clone(),
        );
        self.dependencies.insert(normalized.clone(), node);

        // Recursively analyze includes
        for include_path in &includes {
            // Check for circular dependency via visited set first
            if visited.contains(include_path) {
                return Err(DependencyError::CircularDependency {
                    path: include_path.display().to_string(),
                    dependency_chain: visited.iter().map(|p| p.display().to_string()).collect(),
                });
            }

            // Only analyze if not already fully analyzed
            if !self.dependencies.contains_key(include_path)
                || !self.dependencies.get(include_path).unwrap().analyzed
            {
                self.analyze_recursive(include_path, visited)?;
            }

            // Add reverse dependency
            if let Some(dep_node) = self.dependencies.get_mut(include_path) {
                if !dep_node.dependents.contains(&normalized) {
                    dep_node.dependents.push(normalized.clone());
                }
            }
        }

        // Recursively analyze material files
        for material_path in &material_files {
            if !self.dependencies.contains_key(material_path) {
                // Material files don't have WGSL includes, but we track their hash
                let mat_hash = self.compute_hash(material_path)?;
                let mat_node = DependencyNode::analyzed(
                    material_path.clone(),
                    mat_hash,
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                );
                self.dependencies.insert(material_path.clone(), mat_node);
            }

            // Add reverse dependency
            if let Some(dep_node) = self.dependencies.get_mut(material_path) {
                if !dep_node.dependents.contains(&normalized) {
                    dep_node.dependents.push(normalized.clone());
                }
            }
        }

        visited.remove(&normalized);
        Ok(())
    }

    /// Extract #include directives from source.
    fn extract_includes(
        &self,
        source: &str,
        from_file: &Path,
    ) -> Result<Vec<PathBuf>, DependencyError> {
        let mut includes = Vec::new();

        for (line_num, line) in source.lines().enumerate() {
            let trimmed = line.trim();

            // Match #include "path" pattern
            if trimmed.starts_with("#include") {
                if let Some(path) = Self::parse_include_directive(trimmed) {
                    let resolved = self.resolve_path(&path, from_file)?;
                    if !includes.contains(&resolved) {
                        includes.push(resolved);
                    }
                } else {
                    return Err(DependencyError::ParseError {
                        path: from_file.display().to_string(),
                        line: line_num + 1,
                        message: "invalid #include syntax".to_string(),
                    });
                }
            }
        }

        Ok(includes)
    }

    /// Parse a single #include directive to extract the path.
    fn parse_include_directive(line: &str) -> Option<String> {
        let line = line.trim_start_matches("#include").trim();

        // Match "path" format
        if line.starts_with('"') && line.ends_with('"') && line.len() > 2 {
            return Some(line[1..line.len() - 1].to_string());
        }

        // Match <path> format (angle brackets)
        if line.starts_with('<') && line.ends_with('>') && line.len() > 2 {
            return Some(line[1..line.len() - 1].to_string());
        }

        None
    }

    /// Extract @import module references from source.
    fn extract_imports(&self, source: &str) -> Vec<String> {
        let mut imports = Vec::new();

        for line in source.lines() {
            let trimmed = line.trim();

            // Match @import module::path pattern
            if trimmed.starts_with("@import") {
                if let Some(module) = Self::parse_import_directive(trimmed) {
                    if !imports.contains(&module) {
                        imports.push(module);
                    }
                }
            }

            // Also match // @import comments (for compatibility)
            if trimmed.starts_with("// @import") {
                let comment_content = trimmed.trim_start_matches("//").trim();
                if let Some(module) = Self::parse_import_directive(comment_content) {
                    if !imports.contains(&module) {
                        imports.push(module);
                    }
                }
            }
        }

        imports
    }

    /// Parse a single @import directive to extract the module path.
    fn parse_import_directive(line: &str) -> Option<String> {
        let line = line.trim_start_matches("@import").trim();

        // Remove trailing semicolon if present
        let line = line.trim_end_matches(';').trim();

        if line.is_empty() {
            return None;
        }

        // Validate module path format (alphanumeric, ::, _)
        if line
            .chars()
            .all(|c| c.is_alphanumeric() || c == ':' || c == '_')
        {
            Some(line.to_string())
        } else {
            None
        }
    }

    /// Extract material DSL dependencies from source.
    ///
    /// Looks for patterns like:
    /// - `// @material-generator: path/to/generator.py`
    /// - `#pragma material_dsl "path/to/material.py"`
    fn extract_material_dependencies(
        &self,
        source: &str,
        from_file: &Path,
    ) -> Result<Vec<PathBuf>, DependencyError> {
        let mut materials = Vec::new();

        for line in source.lines() {
            let trimmed = line.trim();

            // Match // @material-generator: path pattern
            if trimmed.starts_with("// @material-generator:") {
                let path = trimmed
                    .trim_start_matches("// @material-generator:")
                    .trim();
                if !path.is_empty() {
                    let resolved = self.resolve_path(path, from_file)?;
                    if !materials.contains(&resolved) {
                        materials.push(resolved);
                    }
                }
            }

            // Match #pragma material_dsl "path" pattern
            if trimmed.starts_with("#pragma material_dsl") {
                let rest = trimmed.trim_start_matches("#pragma material_dsl").trim();
                if rest.starts_with('"') && rest.ends_with('"') && rest.len() > 2 {
                    let path = &rest[1..rest.len() - 1];
                    let resolved = self.resolve_path(path, from_file)?;
                    if !materials.contains(&resolved) {
                        materials.push(resolved);
                    }
                }
            }
        }

        Ok(materials)
    }

    /// Get all dependencies for a shader (transitive closure).
    pub fn get_all_dependencies<P: AsRef<Path>>(&self, shader_path: P) -> Vec<&DependencyNode> {
        let shader_path = shader_path.as_ref();
        let normalized = self.normalize_path(shader_path);

        let mut result = Vec::new();
        let mut visited = HashSet::new();
        self.collect_dependencies(&normalized, &mut result, &mut visited);
        result
    }

    /// Recursively collect all dependencies.
    fn collect_dependencies<'a>(
        &'a self,
        path: &Path,
        result: &mut Vec<&'a DependencyNode>,
        visited: &mut HashSet<PathBuf>,
    ) {
        if visited.contains(path) {
            return;
        }
        visited.insert(path.to_path_buf());

        if let Some(node) = self.dependencies.get(path) {
            // Collect include dependencies
            for include_path in &node.includes {
                self.collect_dependencies(include_path, result, visited);
            }

            // Collect material dependencies
            for material_path in &node.material_files {
                if let Some(mat_node) = self.dependencies.get(material_path) {
                    if !result.iter().any(|n| n.path == mat_node.path) {
                        result.push(mat_node);
                    }
                }
            }

            // Add the node itself
            if !result.iter().any(|n| n.path == node.path) {
                result.push(node);
            }
        }
    }

    /// Invalidate a dependency and return all affected shaders.
    ///
    /// When a file changes, this propagates the invalidation to all files
    /// that depend on it (directly or transitively).
    pub fn invalidate<P: AsRef<Path>>(&mut self, changed_path: P) -> Vec<PathBuf> {
        let changed_path = changed_path.as_ref();
        let normalized = self.normalize_path(changed_path);

        let mut affected = Vec::new();
        let mut visited = HashSet::new();
        self.collect_dependents(&normalized, &mut affected, &mut visited);

        // Update the hash of the changed file
        if let Ok(new_hash) = self.compute_hash(&normalized) {
            if let Some(node) = self.dependencies.get_mut(&normalized) {
                node.content_hash = new_hash;
                node.analyzed = false; // Mark for re-analysis
            }
        }

        affected
    }

    /// Recursively collect all files that depend on a given file.
    fn collect_dependents(
        &self,
        path: &Path,
        affected: &mut Vec<PathBuf>,
        visited: &mut HashSet<PathBuf>,
    ) {
        if visited.contains(path) {
            return;
        }
        visited.insert(path.to_path_buf());

        if let Some(node) = self.dependencies.get(path) {
            for dependent in &node.dependents {
                if !affected.contains(dependent) {
                    affected.push(dependent.clone());
                }
                self.collect_dependents(dependent, affected, visited);
            }
        }
    }

    /// Get a specific dependency node.
    pub fn get<P: AsRef<Path>>(&self, path: P) -> Option<&DependencyNode> {
        let normalized = self.normalize_path(path.as_ref());
        self.dependencies.get(&normalized)
    }

    /// Check if a path is in the graph.
    pub fn contains<P: AsRef<Path>>(&self, path: P) -> bool {
        let normalized = self.normalize_path(path.as_ref());
        self.dependencies.contains_key(&normalized)
    }

    /// Get the number of nodes in the graph.
    pub fn len(&self) -> usize {
        self.dependencies.len()
    }

    /// Check if the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.dependencies.is_empty()
    }

    /// Clear the entire graph.
    pub fn clear(&mut self) {
        self.dependencies.clear();
    }

    /// Remove a node from the graph.
    pub fn remove<P: AsRef<Path>>(&mut self, path: P) -> Option<DependencyNode> {
        let normalized = self.normalize_path(path.as_ref());

        // Remove from dependents lists
        if let Some(node) = self.dependencies.get(&normalized) {
            let includes = node.includes.clone();
            for include_path in includes {
                if let Some(dep_node) = self.dependencies.get_mut(&include_path) {
                    dep_node.dependents.retain(|p| p != &normalized);
                }
            }
        }

        self.dependencies.remove(&normalized)
    }

    /// Serialize the graph to bytes (JSON format).
    pub fn serialize(&self) -> Vec<u8> {
        let data = SerializedGraph {
            root: self.root.clone(),
            search_paths: self.search_paths.clone(),
            dependencies: self.dependencies.clone(),
        };
        serde_json::to_vec(&data).unwrap_or_default()
    }

    /// Deserialize a graph from bytes (JSON format).
    pub fn deserialize(data: &[u8]) -> Result<Self, DependencyError> {
        let serialized: SerializedGraph = serde_json::from_slice(data).map_err(|e| {
            DependencyError::InvalidPath {
                path: String::new(),
                message: format!("deserialization failed: {}", e),
            }
        })?;

        Ok(Self {
            root: serialized.root,
            dependencies: serialized.dependencies,
            search_paths: serialized.search_paths,
            file_reader: None,
        })
    }

    /// Iterate over all nodes in the graph.
    pub fn iter(&self) -> impl Iterator<Item = (&PathBuf, &DependencyNode)> {
        self.dependencies.iter()
    }

    /// Get all root shaders (shaders that no other shader depends on).
    pub fn get_roots(&self) -> Vec<&PathBuf> {
        self.dependencies
            .iter()
            .filter(|(_, node)| node.dependents.is_empty())
            .map(|(path, _)| path)
            .collect()
    }

    /// Get all leaf dependencies (files that have no dependencies of their own).
    pub fn get_leaves(&self) -> Vec<&PathBuf> {
        self.dependencies
            .iter()
            .filter(|(_, node)| !node.has_dependencies())
            .map(|(path, _)| path)
            .collect()
    }

    /// Verify all hashes in the graph against the filesystem.
    ///
    /// Returns a list of paths where the hash no longer matches.
    pub fn verify_hashes(&self) -> Vec<PathBuf> {
        let mut stale = Vec::new();

        for (path, node) in &self.dependencies {
            if let Ok(current_hash) = self.compute_hash(path) {
                if current_hash != node.content_hash {
                    stale.push(path.clone());
                }
            } else {
                // File doesn't exist anymore
                stale.push(path.clone());
            }
        }

        stale
    }
}

/// Serializable representation of the graph.
#[derive(Serialize, Deserialize)]
struct SerializedGraph {
    root: PathBuf,
    search_paths: Vec<PathBuf>,
    dependencies: HashMap<PathBuf, DependencyNode>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // Helper to create a graph with in-memory file system
    fn graph_with_files(files: HashMap<&str, &str>) -> ShaderDependencyGraph {
        let files: HashMap<String, String> = files
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();

        let mut graph = ShaderDependencyGraph::new(vec![PathBuf::from("shaders")]);
        graph.set_file_reader(move |path: &Path| {
            files
                .get(&path.to_string_lossy().to_string())
                .cloned()
                .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "file not found"))
        });
        graph
    }

    // =======================================================================
    // Test 1-5: Direct include extraction
    // =======================================================================

    #[test]
    fn test_extract_single_include() {
        let files = HashMap::from([
            ("main.wgsl", "#include \"common.wgsl\"\nfn main() {}"),
            ("common.wgsl", "fn common() {}"),
        ]);
        let mut graph = graph_with_files(files);

        let node = graph.analyze("main.wgsl").unwrap();
        assert_eq!(node.includes.len(), 1);
        assert!(node.includes[0].to_string_lossy().contains("common.wgsl"));
    }

    #[test]
    fn test_extract_multiple_includes() {
        let files = HashMap::from([
            (
                "main.wgsl",
                "#include \"a.wgsl\"\n#include \"b.wgsl\"\n#include \"c.wgsl\"",
            ),
            ("a.wgsl", "fn a() {}"),
            ("b.wgsl", "fn b() {}"),
            ("c.wgsl", "fn c() {}"),
        ]);
        let mut graph = graph_with_files(files);

        let node = graph.analyze("main.wgsl").unwrap();
        assert_eq!(node.includes.len(), 3);
    }

    #[test]
    fn test_extract_angle_bracket_include() {
        let files = HashMap::from([
            ("main.wgsl", "#include <system/utils.wgsl>"),
            ("shaders/system/utils.wgsl", "fn utils() {}"),
        ]);
        let mut graph = graph_with_files(files);

        let node = graph.analyze("main.wgsl").unwrap();
        assert_eq!(node.includes.len(), 1);
    }

    #[test]
    fn test_deduplicate_includes() {
        let files = HashMap::from([
            (
                "main.wgsl",
                "#include \"common.wgsl\"\n#include \"common.wgsl\"",
            ),
            ("common.wgsl", "fn common() {}"),
        ]);
        let mut graph = graph_with_files(files);

        let node = graph.analyze("main.wgsl").unwrap();
        assert_eq!(node.includes.len(), 1);
    }

    #[test]
    fn test_include_with_whitespace() {
        let files = HashMap::from([
            ("main.wgsl", "  #include   \"common.wgsl\"  "),
            ("common.wgsl", "fn common() {}"),
        ]);
        let mut graph = graph_with_files(files);

        let node = graph.analyze("main.wgsl").unwrap();
        assert_eq!(node.includes.len(), 1);
    }

    // =======================================================================
    // Test 6-10: Transitive dependency resolution
    // =======================================================================

    #[test]
    fn test_transitive_two_level() {
        let files = HashMap::from([
            ("main.wgsl", "#include \"mid.wgsl\""),
            ("mid.wgsl", "#include \"leaf.wgsl\""),
            ("leaf.wgsl", "fn leaf() {}"),
        ]);
        let mut graph = graph_with_files(files);

        graph.analyze("main.wgsl").unwrap();
        let deps = graph.get_all_dependencies("main.wgsl");
        assert_eq!(deps.len(), 3); // main, mid, leaf
    }

    #[test]
    fn test_transitive_diamond_dependency() {
        // main depends on a and b, both depend on common
        let files = HashMap::from([
            ("main.wgsl", "#include \"a.wgsl\"\n#include \"b.wgsl\""),
            ("a.wgsl", "#include \"common.wgsl\""),
            ("b.wgsl", "#include \"common.wgsl\""),
            ("common.wgsl", "fn common() {}"),
        ]);
        let mut graph = graph_with_files(files);

        graph.analyze("main.wgsl").unwrap();
        let deps = graph.get_all_dependencies("main.wgsl");

        // Should have main, a, b, common (no duplicates)
        let paths: HashSet<_> = deps.iter().map(|n| n.path.clone()).collect();
        assert_eq!(paths.len(), 4);
    }

    #[test]
    fn test_transitive_deep_chain() {
        let files = HashMap::from([
            ("l1.wgsl", "#include \"l2.wgsl\""),
            ("l2.wgsl", "#include \"l3.wgsl\""),
            ("l3.wgsl", "#include \"l4.wgsl\""),
            ("l4.wgsl", "#include \"l5.wgsl\""),
            ("l5.wgsl", "fn leaf() {}"),
        ]);
        let mut graph = graph_with_files(files);

        graph.analyze("l1.wgsl").unwrap();
        let deps = graph.get_all_dependencies("l1.wgsl");
        assert_eq!(deps.len(), 5);
    }

    #[test]
    fn test_circular_dependency_detection() {
        let files = HashMap::from([
            ("a.wgsl", "#include \"b.wgsl\""),
            ("b.wgsl", "#include \"a.wgsl\""),
        ]);
        let mut graph = graph_with_files(files);

        let result = graph.analyze("a.wgsl");
        assert!(matches!(result, Err(DependencyError::CircularDependency { .. })));
    }

    #[test]
    fn test_self_referential_dependency() {
        let files = HashMap::from([("self.wgsl", "#include \"self.wgsl\"")]);
        let mut graph = graph_with_files(files);

        let result = graph.analyze("self.wgsl");
        assert!(matches!(result, Err(DependencyError::CircularDependency { .. })));
    }

    // =======================================================================
    // Test 11-14: @import module parsing
    // =======================================================================

    #[test]
    fn test_import_simple_module() {
        let files = HashMap::from([("main.wgsl", "@import utils::math;\nfn main() {}")]);
        let mut graph = graph_with_files(files);

        let node = graph.analyze("main.wgsl").unwrap();
        assert_eq!(node.imports, vec!["utils::math"]);
    }

    #[test]
    fn test_import_multiple_modules() {
        let files = HashMap::from([(
            "main.wgsl",
            "@import utils::math;\n@import common::lighting;\n@import effects::bloom;",
        )]);
        let mut graph = graph_with_files(files);

        let node = graph.analyze("main.wgsl").unwrap();
        assert_eq!(node.imports.len(), 3);
        assert!(node.imports.contains(&"utils::math".to_string()));
        assert!(node.imports.contains(&"common::lighting".to_string()));
        assert!(node.imports.contains(&"effects::bloom".to_string()));
    }

    #[test]
    fn test_import_in_comment() {
        let files = HashMap::from([("main.wgsl", "// @import utils::math\nfn main() {}")]);
        let mut graph = graph_with_files(files);

        let node = graph.analyze("main.wgsl").unwrap();
        assert_eq!(node.imports, vec!["utils::math"]);
    }

    #[test]
    fn test_import_with_underscore() {
        let files = HashMap::from([("main.wgsl", "@import some_module::sub_module;")]);
        let mut graph = graph_with_files(files);

        let node = graph.analyze("main.wgsl").unwrap();
        assert_eq!(node.imports, vec!["some_module::sub_module"]);
    }

    // =======================================================================
    // Test 15-18: Search path resolution
    // =======================================================================

    #[test]
    fn test_search_path_basic() {
        let files = HashMap::from([
            ("main.wgsl", "#include \"utils.wgsl\""),
            ("shaders/utils.wgsl", "fn utils() {}"),
        ]);
        let mut graph = graph_with_files(files);

        let node = graph.analyze("main.wgsl").unwrap();
        assert_eq!(node.includes.len(), 1);
    }

    #[test]
    fn test_search_path_priority() {
        // First search path should win
        let files = HashMap::from([
            ("main.wgsl", "#include \"common.wgsl\""),
            ("shaders/common.wgsl", "// from shaders/"),
            ("other/common.wgsl", "// from other/"),
        ]);

        let mut graph = ShaderDependencyGraph::new(vec![
            PathBuf::from("shaders"),
            PathBuf::from("other"),
        ]);
        let files_clone = files.clone();
        let files_map: HashMap<String, String> = files_clone
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        graph.set_file_reader(move |path: &Path| {
            files_map
                .get(&path.to_string_lossy().to_string())
                .cloned()
                .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "file not found"))
        });

        let node = graph.analyze("main.wgsl").unwrap();
        assert!(node.includes[0].to_string_lossy().contains("shaders"));
    }

    #[test]
    fn test_search_path_not_found() {
        let files = HashMap::from([("main.wgsl", "#include \"nonexistent.wgsl\"")]);
        let mut graph = graph_with_files(files);

        let result = graph.analyze("main.wgsl");
        assert!(matches!(result, Err(DependencyError::FileNotFound { .. })));
    }

    #[test]
    fn test_search_path_relative_to_file() {
        let files = HashMap::from([
            ("dir/main.wgsl", "#include \"sibling.wgsl\""),
            ("dir/sibling.wgsl", "fn sibling() {}"),
        ]);
        let mut graph = graph_with_files(files);

        let node = graph.analyze("dir/main.wgsl").unwrap();
        assert_eq!(node.includes.len(), 1);
    }

    // =======================================================================
    // Test 19-21: Content hash computation
    // =======================================================================

    #[test]
    fn test_content_hash_computed() {
        let files = HashMap::from([("main.wgsl", "fn main() {}")]);
        let mut graph = graph_with_files(files);

        let node = graph.analyze("main.wgsl").unwrap();
        assert!(!node.content_hash.is_zero());
    }

    #[test]
    fn test_content_hash_deterministic() {
        let files = HashMap::from([("main.wgsl", "fn main() {}")]);
        let mut graph1 = graph_with_files(files.clone());
        let mut graph2 = graph_with_files(files);

        let hash1 = graph1.analyze("main.wgsl").unwrap().content_hash;
        let hash2 = graph2.analyze("main.wgsl").unwrap().content_hash;
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_content_hash_changes_with_content() {
        let files1 = HashMap::from([("main.wgsl", "fn main() { version1 }")]);
        let files2 = HashMap::from([("main.wgsl", "fn main() { version2 }")]);

        let mut graph1 = graph_with_files(files1);
        let mut graph2 = graph_with_files(files2);

        let hash1 = graph1.analyze("main.wgsl").unwrap().content_hash;
        let hash2 = graph2.analyze("main.wgsl").unwrap().content_hash;
        assert_ne!(hash1, hash2);
    }

    // =======================================================================
    // Test 22-25: Invalidation propagation
    // =======================================================================

    #[test]
    fn test_invalidate_direct_dependent() {
        let files = HashMap::from([
            ("main.wgsl", "#include \"common.wgsl\""),
            ("common.wgsl", "fn common() {}"),
        ]);
        let mut graph = graph_with_files(files);

        graph.analyze("main.wgsl").unwrap();
        let affected = graph.invalidate("common.wgsl");

        assert_eq!(affected.len(), 1);
        assert!(affected[0].to_string_lossy().contains("main.wgsl"));
    }

    #[test]
    fn test_invalidate_transitive_dependents() {
        let files = HashMap::from([
            ("a.wgsl", "#include \"b.wgsl\""),
            ("b.wgsl", "#include \"c.wgsl\""),
            ("c.wgsl", "fn c() {}"),
        ]);
        let mut graph = graph_with_files(files);

        graph.analyze("a.wgsl").unwrap();
        let affected = graph.invalidate("c.wgsl");

        // Both a.wgsl and b.wgsl depend on c.wgsl
        assert_eq!(affected.len(), 2);
    }

    #[test]
    fn test_invalidate_multiple_dependents() {
        let files = HashMap::from([
            ("a.wgsl", "#include \"common.wgsl\""),
            ("b.wgsl", "#include \"common.wgsl\""),
            ("c.wgsl", "#include \"common.wgsl\""),
            ("common.wgsl", "fn common() {}"),
        ]);
        let mut graph = graph_with_files(files);

        graph.analyze("a.wgsl").unwrap();
        graph.analyze("b.wgsl").unwrap();
        graph.analyze("c.wgsl").unwrap();

        let affected = graph.invalidate("common.wgsl");
        assert_eq!(affected.len(), 3);
    }

    #[test]
    fn test_invalidate_no_dependents() {
        let files = HashMap::from([("standalone.wgsl", "fn standalone() {}")]);
        let mut graph = graph_with_files(files);

        graph.analyze("standalone.wgsl").unwrap();
        let affected = graph.invalidate("standalone.wgsl");

        assert!(affected.is_empty());
    }

    // =======================================================================
    // Additional tests: Serialization, graph operations
    // =======================================================================

    #[test]
    fn test_serialize_deserialize() {
        let files = HashMap::from([
            ("main.wgsl", "#include \"common.wgsl\""),
            ("common.wgsl", "fn common() {}"),
        ]);
        let mut graph = graph_with_files(files);
        graph.analyze("main.wgsl").unwrap();

        let serialized = graph.serialize();
        let restored = ShaderDependencyGraph::deserialize(&serialized).unwrap();

        assert_eq!(restored.len(), graph.len());
    }

    #[test]
    fn test_get_roots() {
        let files = HashMap::from([
            ("root1.wgsl", "#include \"shared.wgsl\""),
            ("root2.wgsl", "#include \"shared.wgsl\""),
            ("shared.wgsl", "fn shared() {}"),
        ]);
        let mut graph = graph_with_files(files);

        graph.analyze("root1.wgsl").unwrap();
        graph.analyze("root2.wgsl").unwrap();

        let roots = graph.get_roots();
        assert_eq!(roots.len(), 2);
    }

    #[test]
    fn test_get_leaves() {
        let files = HashMap::from([
            ("main.wgsl", "#include \"leaf.wgsl\""),
            ("leaf.wgsl", "fn leaf() {}"),
        ]);
        let mut graph = graph_with_files(files);

        graph.analyze("main.wgsl").unwrap();

        let leaves = graph.get_leaves();
        assert_eq!(leaves.len(), 1);
        assert!(leaves[0].to_string_lossy().contains("leaf.wgsl"));
    }

    #[test]
    fn test_material_dsl_dependency() {
        let files = HashMap::from([
            (
                "pbr.wgsl",
                "// @material-generator: materials/pbr_gen.py\nfn pbr() {}",
            ),
            ("materials/pbr_gen.py", "# Python material generator"),
        ]);
        let mut graph = graph_with_files(files);

        let node = graph.analyze("pbr.wgsl").unwrap();
        assert_eq!(node.material_files.len(), 1);
        assert!(node.material_files[0]
            .to_string_lossy()
            .contains("pbr_gen.py"));
    }

    #[test]
    fn test_pragma_material_dsl() {
        let files = HashMap::from([
            (
                "effect.wgsl",
                "#pragma material_dsl \"effects/bloom.py\"\nfn effect() {}",
            ),
            ("effects/bloom.py", "# Bloom effect generator"),
        ]);
        let mut graph = graph_with_files(files);

        let node = graph.analyze("effect.wgsl").unwrap();
        assert_eq!(node.material_files.len(), 1);
    }

    #[test]
    fn test_clear_graph() {
        let files = HashMap::from([("main.wgsl", "fn main() {}")]);
        let mut graph = graph_with_files(files);

        graph.analyze("main.wgsl").unwrap();
        assert!(!graph.is_empty());

        graph.clear();
        assert!(graph.is_empty());
    }

    #[test]
    fn test_remove_node() {
        let files = HashMap::from([
            ("main.wgsl", "#include \"helper.wgsl\""),
            ("helper.wgsl", "fn helper() {}"),
        ]);
        let mut graph = graph_with_files(files);

        graph.analyze("main.wgsl").unwrap();
        assert!(graph.contains("helper.wgsl"));

        graph.remove("helper.wgsl");
        assert!(!graph.contains("helper.wgsl"));
    }

    #[test]
    fn test_analyzed_flag() {
        let files = HashMap::from([("main.wgsl", "fn main() {}")]);
        let mut graph = graph_with_files(files);

        let node = graph.analyze("main.wgsl").unwrap();
        assert!(node.analyzed);
        assert!(node.last_analyzed > 0);
    }

    #[test]
    fn test_direct_dependencies_accessor() {
        let files = HashMap::from([
            ("main.wgsl", "#include \"a.wgsl\"\n// @material-generator: mat.py"),
            ("a.wgsl", "fn a() {}"),
            ("mat.py", "# generator"),
        ]);
        let mut graph = graph_with_files(files);

        let node = graph.analyze("main.wgsl").unwrap();
        let deps = node.direct_dependencies();
        assert_eq!(deps.len(), 2);
    }
}
