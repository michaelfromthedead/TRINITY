//! Code dependency graph structures.

use std::collections::{HashMap, HashSet};

mod builder;
mod crosslang;
mod deps;
mod edges;
mod nodes;
mod testmap;

pub use builder::{persist_edges_to_db, persist_full_graph, persist_graph_to_db, GraphBuilder, PersistError, PersistStats, ScanError, ScanStats};
pub use crosslang::{
    create_crosslang_edges, detect_struct_mirrors, BindingType, CrossLangBinding,
    CrossLangStats, Pyo3Analyzer, ReprCAnalyzer,
};
pub use deps::{resolve_deps_to_edges, DepStats, DepType, PythonDepAnalyzer, RawDependency, RustDepAnalyzer};
pub use edges::{CodeEdge, EdgeType};
pub use nodes::{CodeNode, NodeId};
pub use testmap::{
    create_test_edges, extract_unmapped, generate_coverage_report, get_covered_nodes,
    get_orphan_tests, get_uncovered_nodes, mark_as_orphan, validate_mappings,
    verify_test_targets, CombinedMapper, ConventionMapper, CoverageReport, ExplicitMapper,
    ExplicitMapping, FileCoverage, InlineTestMapper, MappingConfig, MappingConfigError,
    MappingSource, MappingStats, PythonTestMapper, RustTestMapper, TestMapping,
    TestValidationResult, UnmappedReview, UnmappedTest,
};
use crate::parsers::{Language, UnitType};

/// A code dependency graph.
pub struct CodeGraph {
    nodes: Vec<CodeNode>,
    edges: Vec<CodeEdge>,
}

impl CodeGraph {
    /// Create a new empty graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, node: CodeNode) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(node);
        id
    }

    /// Add an edge between two nodes.
    pub fn add_edge(&mut self, edge: CodeEdge) {
        self.edges.push(edge);
    }

    /// Get all nodes in the graph.
    pub fn nodes(&self) -> &[CodeNode] {
        &self.nodes
    }

    /// Get all edges in the graph.
    pub fn edges(&self) -> &[CodeEdge] {
        &self.edges
    }

    /// Count nodes by language.
    pub fn node_count_by_language(&self) -> HashMap<Language, usize> {
        let mut counts = HashMap::new();
        for node in &self.nodes {
            *counts.entry(node.language()).or_insert(0) += 1;
        }
        counts
    }

    /// Count edges by type.
    pub fn edge_count_by_type(&self) -> HashMap<EdgeType, usize> {
        let mut counts = HashMap::new();
        for edge in &self.edges {
            *counts.entry(edge.edge_type).or_insert(0) += 1;
        }
        counts
    }

    /// Find orphan nodes (nodes with no incoming or outgoing edges).
    ///
    /// Entry points (functions named "main", entry point shaders, etc.) are
    /// typically orphans and are NOT considered errors.
    pub fn find_orphan_nodes(&self) -> Vec<NodeId> {
        let mut connected: HashSet<NodeId> = HashSet::new();

        for edge in &self.edges {
            connected.insert(edge.source);
            connected.insert(edge.target);
        }

        self.nodes
            .iter()
            .filter(|n| !connected.contains(&n.id))
            .map(|n| n.id)
            .collect()
    }

    /// Check if a node is an entry point (main function, shader entry, etc.).
    pub fn is_entry_point(&self, node_id: NodeId) -> bool {
        if let Some(node) = self.nodes.get(node_id.0) {
            let name = node.name();
            match node.language() {
                Language::Rust => name == "main",
                Language::Python => name == "__main__" || name == "main",
                Language::Wgsl => {
                    // Entry points in WGSL have @vertex, @fragment, @compute
                    // Detected by naming conventions
                    node.unit_type() == UnitType::Function
                        && (name.starts_with("vs_")
                            || name.starts_with("fs_")
                            || name.starts_with("cs_")
                            || name.ends_with("_main")
                            || name == "main")
                }
            }
        } else {
            false
        }
    }

    /// Find orphan nodes that are NOT entry points (potential issues).
    pub fn find_orphan_non_entry_points(&self) -> Vec<NodeId> {
        self.find_orphan_nodes()
            .into_iter()
            .filter(|id| !self.is_entry_point(*id))
            .collect()
    }

    /// Validate the graph and return a summary.
    pub fn validate(&self) -> ValidationResult {
        let nodes_by_language = self.node_count_by_language();
        let edges_by_type = self.edge_count_by_type();
        let orphan_nodes = self.find_orphan_nodes();
        let orphan_entry_points: Vec<_> = orphan_nodes
            .iter()
            .filter(|id| self.is_entry_point(**id))
            .copied()
            .collect();
        let orphan_issues: Vec<_> = orphan_nodes
            .iter()
            .filter(|id| !self.is_entry_point(**id))
            .copied()
            .collect();

        ValidationResult {
            total_nodes: self.nodes.len(),
            total_edges: self.edges.len(),
            nodes_by_language,
            edges_by_type,
            orphan_entry_points: orphan_entry_points.len(),
            orphan_issues: orphan_issues.len(),
            is_valid: orphan_issues.is_empty() || self.edges.is_empty(), // Allow orphans if no edges yet
        }
    }
}

impl Default for CodeGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of graph validation.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Total number of nodes.
    pub total_nodes: usize,
    /// Total number of edges.
    pub total_edges: usize,
    /// Node count by language.
    pub nodes_by_language: HashMap<Language, usize>,
    /// Edge count by type.
    pub edges_by_type: HashMap<EdgeType, usize>,
    /// Number of orphan nodes that are entry points (OK).
    pub orphan_entry_points: usize,
    /// Number of orphan nodes that are NOT entry points (potential issues).
    pub orphan_issues: usize,
    /// Whether the graph passes validation.
    pub is_valid: bool,
}
