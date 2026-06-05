//! Code dependency graph structures.

mod builder;
mod edges;
mod nodes;

pub use builder::{GraphBuilder, ScanError, ScanStats};
pub use edges::{CodeEdge, EdgeType};
pub use nodes::{CodeNode, NodeId};

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
}

impl Default for CodeGraph {
    fn default() -> Self {
        Self::new()
    }
}
