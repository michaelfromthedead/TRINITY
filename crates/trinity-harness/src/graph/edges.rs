//! Graph edge definitions.

use super::NodeId;

/// Types of relationships between code nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeType {
    /// Function calls another function.
    Calls,
    /// Type uses another type.
    Uses,
    /// Type extends/implements another type.
    Extends,
    /// Module imports another module.
    Imports,
    /// Cross-language binding (e.g., Python calling Rust via PyO3).
    Binds,
    /// Struct layout mirror (e.g., WGSL struct mirrors #[repr(C)] Rust struct).
    MirrorsLayout,
}

/// An edge in the code dependency graph.
#[derive(Debug, Clone)]
pub struct CodeEdge {
    pub source: NodeId,
    pub target: NodeId,
    pub edge_type: EdgeType,
}

impl CodeEdge {
    /// Create a new edge.
    pub fn new(source: NodeId, target: NodeId, edge_type: EdgeType) -> Self {
        Self {
            source,
            target,
            edge_type,
        }
    }
}
