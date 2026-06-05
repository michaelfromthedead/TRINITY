//! Graph node definitions.

use crate::parsers::{CodeUnit, Language, UnitType};

/// Unique identifier for a node in the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

/// A node in the code dependency graph.
#[derive(Debug, Clone)]
pub struct CodeNode {
    pub id: NodeId,
    pub file_path: String,
    pub unit: CodeUnit,
}

impl CodeNode {
    /// Create a new code node.
    pub fn new(id: NodeId, file_path: String, unit: CodeUnit) -> Self {
        Self { id, file_path, unit }
    }

    /// Get the language of this node.
    pub fn language(&self) -> Language {
        self.unit.language
    }

    /// Get the unit type of this node.
    pub fn unit_type(&self) -> UnitType {
        self.unit.unit_type
    }

    /// Get the name of this node.
    pub fn name(&self) -> &str {
        &self.unit.name
    }
}
