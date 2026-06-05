//! Graph builder for scanning and parsing codebases.

use std::collections::HashMap;
use std::path::Path;

use walkdir::WalkDir;

use crate::parsers::{Language, ParserRegistry};
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
}

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
