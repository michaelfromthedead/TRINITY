//! Whitebox tests for trinity-harness graph module.
//!
//! WHITEBOX coverage plan:
//!   - CodeGraph::new() creates empty graph
//!   - CodeGraph::default() delegates to new()
//!   - add_node() returns correct NodeId and stores node
//!   - add_edge() stores edge
//!   - nodes() returns correct slice
//!   - edges() returns correct slice
//!   - CodeNode::new() creates node with correct fields
//!   - CodeNode accessors (language, unit_type, name)
//!   - CodeEdge::new() creates edge with correct fields
//!   - EdgeType variants are distinct
//!   - NodeId equality and hashing

use std::collections::HashSet;
use trinity_harness::graph::{CodeEdge, CodeGraph, CodeNode, EdgeType, NodeId};
use trinity_harness::parsers::{CodeUnit, Language, UnitType};

#[test]
fn test_code_graph_new_is_empty() {
    let graph = CodeGraph::new();
    assert!(graph.nodes().is_empty(), "new graph should have no nodes");
    assert!(graph.edges().is_empty(), "new graph should have no edges");
}

#[test]
fn test_code_graph_default_is_empty() {
    let graph = CodeGraph::default();
    assert!(graph.nodes().is_empty(), "default graph should have no nodes");
    assert!(graph.edges().is_empty(), "default graph should have no edges");
}

#[test]
fn test_add_node_returns_sequential_ids() {
    let mut graph = CodeGraph::new();

    let unit1 = CodeUnit {
        unit_type: UnitType::Function,
        name: "foo".to_string(),
        start_line: 1,
        end_line: 10,
        language: Language::Rust,
    };
    let unit2 = CodeUnit {
        unit_type: UnitType::Struct,
        name: "Bar".to_string(),
        start_line: 20,
        end_line: 30,
        language: Language::Rust,
    };

    let node1 = CodeNode::new(NodeId(0), "src/a.rs".to_string(), unit1);
    let node2 = CodeNode::new(NodeId(1), "src/b.rs".to_string(), unit2);

    let id1 = graph.add_node(node1);
    let id2 = graph.add_node(node2);

    assert_eq!(id1.0, 0, "first node should have id 0");
    assert_eq!(id2.0, 1, "second node should have id 1");
}

#[test]
fn test_add_node_stores_node() {
    let mut graph = CodeGraph::new();

    let unit = CodeUnit {
        unit_type: UnitType::Function,
        name: "test_func".to_string(),
        start_line: 5,
        end_line: 15,
        language: Language::Python,
    };
    let node = CodeNode::new(NodeId(0), "script.py".to_string(), unit);

    graph.add_node(node);

    assert_eq!(graph.nodes().len(), 1);
    assert_eq!(graph.nodes()[0].name(), "test_func");
}

#[test]
fn test_add_edge_stores_edge() {
    let mut graph = CodeGraph::new();

    let edge = CodeEdge::new(NodeId(0), NodeId(1), EdgeType::Calls);
    graph.add_edge(edge);

    assert_eq!(graph.edges().len(), 1);
    assert_eq!(graph.edges()[0].source.0, 0);
    assert_eq!(graph.edges()[0].target.0, 1);
    assert_eq!(graph.edges()[0].edge_type, EdgeType::Calls);
}

#[test]
fn test_nodes_returns_all_nodes() {
    let mut graph = CodeGraph::new();

    for i in 0..5 {
        let unit = CodeUnit {
            unit_type: UnitType::Function,
            name: format!("func_{}", i),
            start_line: i,
            end_line: i + 10,
            language: Language::Rust,
        };
        let node = CodeNode::new(NodeId(i), format!("file_{}.rs", i), unit);
        graph.add_node(node);
    }

    let nodes = graph.nodes();
    assert_eq!(nodes.len(), 5);

    for (i, node) in nodes.iter().enumerate() {
        assert_eq!(node.name(), format!("func_{}", i));
    }
}

#[test]
fn test_edges_returns_all_edges() {
    let mut graph = CodeGraph::new();

    graph.add_edge(CodeEdge::new(NodeId(0), NodeId(1), EdgeType::Calls));
    graph.add_edge(CodeEdge::new(NodeId(1), NodeId(2), EdgeType::Uses));
    graph.add_edge(CodeEdge::new(NodeId(2), NodeId(0), EdgeType::Imports));

    let edges = graph.edges();
    assert_eq!(edges.len(), 3);
}

#[test]
fn test_code_node_new_sets_fields() {
    let unit = CodeUnit {
        unit_type: UnitType::Class,
        name: "MyClass".to_string(),
        start_line: 100,
        end_line: 200,
        language: Language::Python,
    };
    let node = CodeNode::new(NodeId(42), "/path/to/file.py".to_string(), unit);

    assert_eq!(node.id.0, 42);
    assert_eq!(node.file_path, "/path/to/file.py");
    assert_eq!(node.name(), "MyClass");
}

#[test]
fn test_code_node_language_accessor() {
    let unit = CodeUnit {
        unit_type: UnitType::Function,
        name: "shader".to_string(),
        start_line: 0,
        end_line: 50,
        language: Language::Wgsl,
    };
    let node = CodeNode::new(NodeId(0), "shader.wgsl".to_string(), unit);

    assert_eq!(node.language(), Language::Wgsl);
}

#[test]
fn test_code_node_unit_type_accessor() {
    let unit = CodeUnit {
        unit_type: UnitType::Module,
        name: "utils".to_string(),
        start_line: 0,
        end_line: 100,
        language: Language::Rust,
    };
    let node = CodeNode::new(NodeId(0), "utils.rs".to_string(), unit);

    assert_eq!(node.unit_type(), UnitType::Module);
}

#[test]
fn test_code_node_name_accessor() {
    let unit = CodeUnit {
        unit_type: UnitType::Struct,
        name: "Configuration".to_string(),
        start_line: 10,
        end_line: 50,
        language: Language::Rust,
    };
    let node = CodeNode::new(NodeId(0), "config.rs".to_string(), unit);

    assert_eq!(node.name(), "Configuration");
}

#[test]
fn test_code_edge_new_sets_fields() {
    let edge = CodeEdge::new(NodeId(5), NodeId(10), EdgeType::Extends);

    assert_eq!(edge.source.0, 5);
    assert_eq!(edge.target.0, 10);
    assert_eq!(edge.edge_type, EdgeType::Extends);
}

#[test]
fn test_edge_type_variants_distinct() {
    // Verify all EdgeType variants are distinct
    let variants = vec![
        EdgeType::Calls,
        EdgeType::Uses,
        EdgeType::Extends,
        EdgeType::Imports,
        EdgeType::Binds,
    ];

    for i in 0..variants.len() {
        for j in 0..variants.len() {
            if i == j {
                assert_eq!(variants[i], variants[j]);
            } else {
                assert_ne!(variants[i], variants[j], "EdgeType variants should be distinct");
            }
        }
    }
}

#[test]
fn test_node_id_equality() {
    let id1 = NodeId(42);
    let id2 = NodeId(42);
    let id3 = NodeId(43);

    assert_eq!(id1, id2);
    assert_ne!(id1, id3);
}

#[test]
fn test_node_id_hash() {
    let mut set = HashSet::new();
    set.insert(NodeId(1));
    set.insert(NodeId(2));
    set.insert(NodeId(1)); // duplicate

    assert_eq!(set.len(), 2, "HashSet should deduplicate NodeIds");
    assert!(set.contains(&NodeId(1)));
    assert!(set.contains(&NodeId(2)));
}

#[test]
fn test_node_id_copy() {
    let id = NodeId(99);
    let id_copy = id; // Copy
    assert_eq!(id.0, id_copy.0);
}

#[test]
fn test_edge_type_copy() {
    let edge_type = EdgeType::Binds;
    let edge_type_copy = edge_type; // Copy
    assert_eq!(edge_type, edge_type_copy);
}
