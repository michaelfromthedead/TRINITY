//! Graph builder for scanning and parsing codebases.

use std::collections::HashMap;
use std::path::Path;

use walkdir::WalkDir;

use crate::db::HarnessDb;
use crate::parsers::{Language, ParserRegistry, UnitType};
use super::{CodeGraph, CodeNode, NodeId};

/// Statistics from a full scan operation.
#[derive(Debug, Clone, Default)]
pub struct ScanStats {
    /// Total files scanned.
    pub files_scanned: usize,
    /// Files skipped (unsupported extension or read error).
    pub files_skipped: usize,
    /// Number of nodes created per language.
    pub nodes_per_language: HashMap<Language, usize>,
    /// Total nodes created.
    pub total_nodes: usize,
}

impl ScanStats {
    /// Create new empty scan stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a node created for a language.
    pub fn record_node(&mut self, lang: Language) {
        *self.nodes_per_language.entry(lang).or_insert(0) += 1;
        self.total_nodes += 1;
    }

    /// Record a scanned file.
    pub fn record_file(&mut self) {
        self.files_scanned += 1;
    }

    /// Record a skipped file.
    pub fn record_skip(&mut self) {
        self.files_skipped += 1;
    }
}

/// Builder for constructing code graphs from file system scans.
pub struct GraphBuilder<'a> {
    registry: &'a ParserRegistry,
}

impl<'a> GraphBuilder<'a> {
    /// Create a new graph builder with the given parser registry.
    pub fn new(registry: &'a ParserRegistry) -> Self {
        Self { registry }
    }

    /// Perform a full scan of the directory tree starting at `root_path`.
    ///
    /// Walks the directory recursively, parsing files with supported extensions
    /// (.rs, .py, .wgsl) and building a code graph.
    ///
    /// # Arguments
    ///
    /// * `root_path` - The root directory to start scanning from.
    ///
    /// # Returns
    ///
    /// A tuple of (CodeGraph, ScanStats) containing the built graph and statistics.
    pub fn full_scan(&self, root_path: &Path) -> Result<(CodeGraph, ScanStats), ScanError> {
        let mut graph = CodeGraph::new();
        let mut stats = ScanStats::new();

        for entry in WalkDir::new(root_path)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();

            if !path.is_file() {
                continue;
            }

            let Some(lang) = ParserRegistry::detect_language(path) else {
                stats.record_skip();
                continue;
            };

            match self.scan_file(path, lang, &mut graph) {
                Ok(node_count) => {
                    stats.record_file();
                    for _ in 0..node_count {
                        stats.record_node(lang);
                    }
                }
                Err(_) => {
                    stats.record_skip();
                }
            }
        }

        Ok((graph, stats))
    }

    /// Scan a single file and add its nodes to the graph.
    ///
    /// Returns the number of nodes created from this file.
    fn scan_file(
        &self,
        path: &Path,
        lang: Language,
        graph: &mut CodeGraph,
    ) -> Result<usize, ScanError> {
        let source = std::fs::read_to_string(path).map_err(|e| ScanError::IoError {
            path: path.to_path_buf(),
            source: e,
        })?;

        let units = self.registry.parse(&source, lang);
        let path_str = path.to_string_lossy().to_string();

        let mut count = 0;
        for unit in units {
            let node_id = NodeId(graph.nodes().len());
            let node = CodeNode::new(node_id, path_str.clone(), unit);
            graph.add_node(node);
            count += 1;
        }

        Ok(count)
    }

    /// Scan a single file by path, auto-detecting the language.
    ///
    /// Returns `None` if the file extension is not recognized.
    pub fn scan_single_file(&self, path: &Path, graph: &mut CodeGraph) -> Option<usize> {
        let lang = ParserRegistry::detect_language(path)?;
        self.scan_file(path, lang, graph).ok()
    }

    /// Scan only Rust files (.rs) in the directory tree.
    ///
    /// This is a convenience method that filters out non-Rust files,
    /// providing faster scanning when only Rust code is of interest.
    ///
    /// # Arguments
    ///
    /// * `root_path` - The root directory to start scanning from.
    ///
    /// # Returns
    ///
    /// A tuple of (CodeGraph, ScanStats) containing the built graph and statistics.
    pub fn scan_rust(&self, root_path: &Path) -> Result<(CodeGraph, ScanStats), ScanError> {
        self.scan_language(root_path, Language::Rust)
    }

    /// Scan only Python files (.py) in the directory tree.
    ///
    /// This is a convenience method that filters out non-Python files.
    pub fn scan_python(&self, root_path: &Path) -> Result<(CodeGraph, ScanStats), ScanError> {
        self.scan_language(root_path, Language::Python)
    }

    /// Scan only WGSL shader files (.wgsl) in the directory tree.
    ///
    /// This is a convenience method that filters out non-WGSL files.
    pub fn scan_wgsl(&self, root_path: &Path) -> Result<(CodeGraph, ScanStats), ScanError> {
        self.scan_language(root_path, Language::Wgsl)
    }

    /// Scan only files of a specific language in the directory tree.
    ///
    /// # Arguments
    ///
    /// * `root_path` - The root directory to start scanning from.
    /// * `target_lang` - The language to filter for.
    ///
    /// # Returns
    ///
    /// A tuple of (CodeGraph, ScanStats) containing the built graph and statistics.
    pub fn scan_language(
        &self,
        root_path: &Path,
        target_lang: Language,
    ) -> Result<(CodeGraph, ScanStats), ScanError> {
        let mut graph = CodeGraph::new();
        let mut stats = ScanStats::new();

        for entry in WalkDir::new(root_path)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();

            if !path.is_file() {
                continue;
            }

            let Some(lang) = ParserRegistry::detect_language(path) else {
                stats.record_skip();
                continue;
            };

            if lang != target_lang {
                stats.record_skip();
                continue;
            }

            match self.scan_file(path, lang, &mut graph) {
                Ok(node_count) => {
                    stats.record_file();
                    for _ in 0..node_count {
                        stats.record_node(lang);
                    }
                }
                Err(_) => {
                    stats.record_skip();
                }
            }
        }

        Ok((graph, stats))
    }

    /// Analyze dependencies in all files and create edges in the graph.
    ///
    /// This should be called after scanning to populate edges between nodes.
    /// It walks the same directory tree, extracts dependencies from source,
    /// and resolves them to edges.
    pub fn analyze_dependencies(
        &self,
        root_path: &Path,
        graph: &mut CodeGraph,
    ) -> Result<super::DepStats, ScanError> {
        use super::{resolve_deps_to_edges, PythonDepAnalyzer, RustDepAnalyzer};

        let rust_analyzer = RustDepAnalyzer::new();
        let python_analyzer = PythonDepAnalyzer::new();

        let mut all_deps = Vec::new();

        for entry in WalkDir::new(root_path)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();

            if !path.is_file() {
                continue;
            }

            let Some(lang) = ParserRegistry::detect_language(path) else {
                continue;
            };

            let source = match std::fs::read_to_string(path) {
                Ok(s) => s,
                Err(_) => continue,
            };

            let path_str = path.to_string_lossy().to_string();

            let deps = match lang {
                Language::Rust => rust_analyzer.analyze(&source, &path_str),
                Language::Python => python_analyzer.analyze(&source, &path_str),
                Language::Wgsl => Vec::new(), // WGSL deps not implemented yet
            };

            all_deps.extend(deps);
        }

        let stats = resolve_deps_to_edges(graph, &all_deps);
        Ok(stats)
    }

    /// Analyze cross-language bindings and create edges.
    ///
    /// This detects:
    /// - PyO3 bindings (#[pyfunction], #[pyclass], #[pymethods])
    /// - WGSL↔Rust struct mirrors (same name, #[repr(C)])
    pub fn analyze_crosslang(
        &self,
        root_path: &Path,
        graph: &mut CodeGraph,
    ) -> Result<super::CrossLangStats, ScanError> {
        use super::{create_crosslang_edges, Pyo3Analyzer};

        let pyo3_analyzer = Pyo3Analyzer::new();
        let mut all_bindings = Vec::new();

        for entry in WalkDir::new(root_path)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();

            if !path.is_file() {
                continue;
            }

            // Only analyze Rust files for PyO3 bindings
            let Some(lang) = ParserRegistry::detect_language(path) else {
                continue;
            };

            if lang != Language::Rust {
                continue;
            }

            let source = match std::fs::read_to_string(path) {
                Ok(s) => s,
                Err(_) => continue,
            };

            let path_str = path.to_string_lossy().to_string();
            let bindings = pyo3_analyzer.analyze(&source, &path_str);
            all_bindings.extend(bindings);
        }

        let stats = create_crosslang_edges(graph, &all_bindings);
        Ok(stats)
    }

    /// Map Rust tests to their target code nodes.
    ///
    /// This scans `crates/*/tests/*.rs` and other test locations,
    /// applies auto-mapping rules, and creates Tests edges.
    ///
    /// Optionally loads explicit mappings from a TOML config file.
    pub fn map_rust_tests(
        &self,
        root_path: &Path,
        graph: &mut CodeGraph,
        config_path: Option<&Path>,
    ) -> Result<super::MappingStats, ScanError> {
        use super::{create_test_edges, CombinedMapper, MappingConfig};

        let mapper = if let Some(path) = config_path {
            match MappingConfig::load(path) {
                Ok(config) => CombinedMapper::with_explicit(config),
                Err(_) => CombinedMapper::convention_only(),
            }
        } else {
            CombinedMapper::convention_only()
        };

        let (mappings, mut stats) = mapper.map_tests(graph, root_path);
        let edges_created = create_test_edges(graph, &mappings);
        stats.edges_created = edges_created;

        Ok(stats)
    }

    /// Map Python tests to their target code nodes.
    ///
    /// This scans `tests/unit/`, `tests/integration/`, `tests/e2e/`
    /// and other test locations, applies auto-mapping rules, and creates Tests edges.
    ///
    /// Optionally loads explicit mappings from a TOML config file.
    pub fn map_python_tests(
        &self,
        root_path: &Path,
        graph: &mut CodeGraph,
        config_path: Option<&Path>,
    ) -> Result<super::MappingStats, ScanError> {
        // Python test mapping uses the same combined mapper
        self.map_rust_tests(root_path, graph, config_path)
    }

    /// Map all tests (Rust and Python) to their target code nodes.
    ///
    /// This is a convenience method that combines both mappers.
    pub fn map_all_tests(
        &self,
        root_path: &Path,
        graph: &mut CodeGraph,
        config_path: Option<&Path>,
    ) -> Result<super::MappingStats, ScanError> {
        self.map_rust_tests(root_path, graph, config_path)
    }

    /// Map inline tests (#[test] in source files) to their containing module.
    ///
    /// Inline tests are mapped to other code in the same file.
    /// This is separate from blackbox/external test mapping.
    pub fn map_inline_tests(
        &self,
        graph: &mut CodeGraph,
    ) -> Result<super::MappingStats, ScanError> {
        use super::{create_test_edges, InlineTestMapper};

        let mapper = InlineTestMapper::new();
        let (mappings, mut stats) = mapper.map_inline_tests(graph);
        let edges_created = create_test_edges(graph, &mappings);
        stats.edges_created = edges_created;

        Ok(stats)
    }
}

/// Persist a code graph to the database.
///
/// This function inserts all nodes from the graph into the `code_nodes` table.
/// Existing nodes with the same node_id are replaced.
///
/// # Arguments
///
/// * `graph` - The code graph to persist.
/// * `db` - The database connection.
///
/// # Returns
///
/// The number of nodes persisted, or an error.
pub fn persist_graph_to_db(graph: &CodeGraph, db: &HarnessDb) -> Result<usize, PersistError> {
    let conn = db.connection();
    let mut count = 0;

    let mut stmt = conn
        .prepare(
            r#"
            INSERT OR REPLACE INTO code_nodes (
                node_id, file_path, span_start_line, span_end_line,
                language, kind, name, hash_full, current_state
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'unknown')
            "#,
        )
        .map_err(|e| PersistError::DatabaseError(e.to_string()))?;

    for node in graph.nodes() {
        let node_id = format!("{}:{}:{}", node.file_path, node.unit.start_line, node.name());
        let language_str = language_to_str(node.language());
        let kind_str = unit_type_to_str(node.unit_type());
        let hash_full = hex::encode(&node.unit.hashes.full_hash);
        let start_line = node.unit.start_line as i64;
        let end_line = node.unit.end_line as i64;

        stmt.execute(rusqlite::params![
            node_id,
            node.file_path,
            start_line,
            end_line,
            language_str,
            kind_str,
            node.name(),
            hash_full,
        ])
        .map_err(|e| PersistError::DatabaseError(e.to_string()))?;

        count += 1;
    }

    Ok(count)
}

/// Persist all edges from a graph to the database.
///
/// Inserts edges into `code_edges` table. Existing edges are replaced.
pub fn persist_edges_to_db(graph: &CodeGraph, db: &HarnessDb) -> Result<usize, PersistError> {
    use super::EdgeType;

    let conn = db.connection();
    let mut count = 0;

    // Build node_id lookup: NodeId -> string ID used in database
    let mut node_id_map: HashMap<super::NodeId, String> = HashMap::new();
    for node in graph.nodes() {
        let db_id = format!("{}:{}:{}", node.file_path, node.unit.start_line, node.name());
        node_id_map.insert(node.id, db_id);
    }

    let mut stmt = conn
        .prepare(
            r#"
            INSERT OR REPLACE INTO code_edges (
                edge_id, from_node, to_node, kind
            ) VALUES (?1, ?2, ?3, ?4)
            "#,
        )
        .map_err(|e| PersistError::DatabaseError(e.to_string()))?;

    for edge in graph.edges() {
        let Some(from_id) = node_id_map.get(&edge.source) else {
            continue; // Skip edges with missing source
        };
        let Some(to_id) = node_id_map.get(&edge.target) else {
            continue; // Skip edges with missing target
        };

        let kind_str = match edge.edge_type {
            EdgeType::Calls => "calls",
            EdgeType::Uses => "references",
            EdgeType::Extends => "inherits",
            EdgeType::Imports => "imports",
            EdgeType::Binds => "pyo3_call",
            EdgeType::MirrorsLayout => "mirrors_layout",
            EdgeType::Tests => "tests",
        };

        let edge_id = format!("{}->{}:{}", from_id, to_id, kind_str);

        stmt.execute(rusqlite::params![edge_id, from_id, to_id, kind_str])
            .map_err(|e| PersistError::DatabaseError(e.to_string()))?;

        count += 1;
    }

    Ok(count)
}

/// Persist the full graph (nodes + edges) to the database.
///
/// This is the main entry point for persistence. It:
/// 1. Persists all nodes
/// 2. Persists all edges
/// 3. Returns combined statistics
pub fn persist_full_graph(graph: &CodeGraph, db: &HarnessDb) -> Result<PersistStats, PersistError> {
    let nodes = persist_graph_to_db(graph, db)?;
    let edges = persist_edges_to_db(graph, db)?;

    Ok(PersistStats { nodes, edges })
}

/// Statistics from graph persistence.
#[derive(Debug, Clone, Default)]
pub struct PersistStats {
    /// Number of nodes persisted.
    pub nodes: usize,
    /// Number of edges persisted.
    pub edges: usize,
}

/// Convert Language enum to database string.
fn language_to_str(lang: Language) -> &'static str {
    match lang {
        Language::Rust => "rust",
        Language::Python => "python",
        Language::Wgsl => "wgsl",
    }
}

/// Convert UnitType enum to database kind string.
fn unit_type_to_str(unit_type: UnitType) -> &'static str {
    match unit_type {
        UnitType::Function => "rust_function",
        UnitType::Struct => "rust_struct",
        UnitType::Enum => "rust_enum",
        UnitType::Class => "python_class",
        UnitType::Method => "method",
        UnitType::Module => "module",
        UnitType::Impl => "rust_impl",
        UnitType::Trait => "rust_trait",
    }
}

/// Errors that can occur during persistence.
#[derive(Debug)]
pub enum PersistError {
    /// Database error.
    DatabaseError(String),
}

impl std::fmt::Display for PersistError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PersistError::DatabaseError(msg) => write!(f, "database error: {}", msg),
        }
    }
}

impl std::error::Error for PersistError {}

/// Errors that can occur during scanning.
#[derive(Debug)]
pub enum ScanError {
    /// IO error reading a file.
    IoError {
        path: std::path::PathBuf,
        source: std::io::Error,
    },
}

impl std::fmt::Display for ScanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScanError::IoError { path, source } => {
                write!(f, "failed to read {}: {}", path.display(), source)
            }
        }
    }
}

impl std::error::Error for ScanError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ScanError::IoError { source, .. } => Some(source),
        }
    }
}
