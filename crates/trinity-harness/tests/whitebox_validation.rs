//! Whitebox tests for graph validation.

use std::collections::HashMap;
use trinity_harness::graph::{CodeEdge, CodeGraph, CodeNode, EdgeType, NodeId};
use trinity_harness::parsers::{CodeUnit, ContentHashes, Language, UnitType};

fn empty_hashes() -> ContentHashes {
    ContentHashes {
        full_hash: [0u8; 32],
        signature_hash: [0u8; 32],
        body_hash: [0u8; 32],
        layout_hash: [0u8; 32],
    }
}

fn make_node(
    graph: &mut CodeGraph,
    file: &str,
    name: &str,
    unit_type: UnitType,
    lang: Language,
) -> NodeId {
    let id = NodeId(graph.nodes().len());
    let unit = CodeUnit {
        unit_type,
        name: name.to_string(),
        start_line: 1,
        end_line: 10,
        language: lang,
        hashes: empty_hashes(),
    };
    let node = CodeNode::new(id, file.to_string(), unit);
    graph.add_node(node)
}

// ==================== Node Count by Language ====================

#[test]
fn test_node_count_by_language_empty() {
    let graph = CodeGraph::new();
    let counts = graph.node_count_by_language();
    assert!(counts.is_empty());
}

#[test]
fn test_node_count_by_language_single() {
    let mut graph = CodeGraph::new();
    make_node(&mut graph, "a.rs", "func", UnitType::Function, Language::Rust);

    let counts = graph.node_count_by_language();
    assert_eq!(counts.get(&Language::Rust), Some(&1));
}

#[test]
fn test_node_count_by_language_multiple_same() {
    let mut graph = CodeGraph::new();
    make_node(&mut graph, "a.rs", "func1", UnitType::Function, Language::Rust);
    make_node(&mut graph, "b.rs", "func2", UnitType::Function, Language::Rust);
    make_node(&mut graph, "c.rs", "func3", UnitType::Function, Language::Rust);

    let counts = graph.node_count_by_language();
    assert_eq!(counts.get(&Language::Rust), Some(&3));
}

#[test]
fn test_node_count_by_language_mixed() {
    let mut graph = CodeGraph::new();
    make_node(&mut graph, "a.rs", "func", UnitType::Function, Language::Rust);
    make_node(&mut graph, "b.rs", "struct", UnitType::Struct, Language::Rust);
    make_node(&mut graph, "c.py", "func", UnitType::Function, Language::Python);
    make_node(&mut graph, "d.py", "class", UnitType::Class, Language::Python);
    make_node(&mut graph, "e.py", "method", UnitType::Method, Language::Python);
    make_node(&mut graph, "f.wgsl", "vs_main", UnitType::Function, Language::Wgsl);

    let counts = graph.node_count_by_language();
    assert_eq!(counts.get(&Language::Rust), Some(&2));
    assert_eq!(counts.get(&Language::Python), Some(&3));
    assert_eq!(counts.get(&Language::Wgsl), Some(&1));
}

// ==================== Edge Count by Type ====================

#[test]
fn test_edge_count_by_type_empty() {
    let graph = CodeGraph::new();
    let counts = graph.edge_count_by_type();
    assert!(counts.is_empty());
}

#[test]
fn test_edge_count_by_type_single() {
    let mut graph = CodeGraph::new();
    let a = make_node(&mut graph, "a.rs", "a", UnitType::Function, Language::Rust);
    let b = make_node(&mut graph, "b.rs", "b", UnitType::Function, Language::Rust);
    graph.add_edge(CodeEdge::new(a, b, EdgeType::Calls));

    let counts = graph.edge_count_by_type();
    assert_eq!(counts.get(&EdgeType::Calls), Some(&1));
}

#[test]
fn test_edge_count_by_type_multiple_same() {
    let mut graph = CodeGraph::new();
    let a = make_node(&mut graph, "a.rs", "a", UnitType::Function, Language::Rust);
    let b = make_node(&mut graph, "b.rs", "b", UnitType::Function, Language::Rust);
    let c = make_node(&mut graph, "c.rs", "c", UnitType::Function, Language::Rust);
    graph.add_edge(CodeEdge::new(a, b, EdgeType::Calls));
    graph.add_edge(CodeEdge::new(b, c, EdgeType::Calls));
    graph.add_edge(CodeEdge::new(a, c, EdgeType::Calls));

    let counts = graph.edge_count_by_type();
    assert_eq!(counts.get(&EdgeType::Calls), Some(&3));
}

#[test]
fn test_edge_count_by_type_mixed() {
    let mut graph = CodeGraph::new();
    let fn1 = make_node(&mut graph, "a.rs", "fn1", UnitType::Function, Language::Rust);
    let fn2 = make_node(&mut graph, "b.rs", "fn2", UnitType::Function, Language::Rust);
    let struct1 = make_node(&mut graph, "c.rs", "S1", UnitType::Struct, Language::Rust);
    let struct2 = make_node(&mut graph, "d.wgsl", "S1", UnitType::Struct, Language::Wgsl);

    graph.add_edge(CodeEdge::new(fn1, fn2, EdgeType::Calls));
    graph.add_edge(CodeEdge::new(fn1, struct1, EdgeType::Uses));
    graph.add_edge(CodeEdge::new(fn2, struct1, EdgeType::Uses));
    graph.add_edge(CodeEdge::new(struct2, struct1, EdgeType::MirrorsLayout));

    let counts = graph.edge_count_by_type();
    assert_eq!(counts.get(&EdgeType::Calls), Some(&1));
    assert_eq!(counts.get(&EdgeType::Uses), Some(&2));
    assert_eq!(counts.get(&EdgeType::MirrorsLayout), Some(&1));
}

// ==================== Orphan Node Detection ====================

#[test]
fn test_find_orphan_nodes_empty() {
    let graph = CodeGraph::new();
    let orphans = graph.find_orphan_nodes();
    assert!(orphans.is_empty());
}

#[test]
fn test_find_orphan_nodes_all_orphans() {
    let mut graph = CodeGraph::new();
    let a = make_node(&mut graph, "a.rs", "a", UnitType::Function, Language::Rust);
    let b = make_node(&mut graph, "b.rs", "b", UnitType::Function, Language::Rust);

    // No edges, all nodes are orphans
    let orphans = graph.find_orphan_nodes();
    assert_eq!(orphans.len(), 2);
    assert!(orphans.contains(&a));
    assert!(orphans.contains(&b));
}

#[test]
fn test_find_orphan_nodes_none_orphan() {
    let mut graph = CodeGraph::new();
    let a = make_node(&mut graph, "a.rs", "a", UnitType::Function, Language::Rust);
    let b = make_node(&mut graph, "b.rs", "b", UnitType::Function, Language::Rust);
    graph.add_edge(CodeEdge::new(a, b, EdgeType::Calls));

    // Both connected
    let orphans = graph.find_orphan_nodes();
    assert!(orphans.is_empty());
}

#[test]
fn test_find_orphan_nodes_partial() {
    let mut graph = CodeGraph::new();
    let a = make_node(&mut graph, "a.rs", "a", UnitType::Function, Language::Rust);
    let b = make_node(&mut graph, "b.rs", "b", UnitType::Function, Language::Rust);
    let c = make_node(&mut graph, "c.rs", "c", UnitType::Function, Language::Rust);
    graph.add_edge(CodeEdge::new(a, b, EdgeType::Calls));

    // c is orphan
    let orphans = graph.find_orphan_nodes();
    assert_eq!(orphans.len(), 1);
    assert!(orphans.contains(&c));
}

// ==================== Entry Point Detection ====================

#[test]
fn test_is_entry_point_rust_main() {
    let mut graph = CodeGraph::new();
    let main_id = make_node(&mut graph, "main.rs", "main", UnitType::Function, Language::Rust);
    let other_id = make_node(&mut graph, "lib.rs", "helper", UnitType::Function, Language::Rust);

    assert!(graph.is_entry_point(main_id));
    assert!(!graph.is_entry_point(other_id));
}

#[test]
fn test_is_entry_point_python_main() {
    let mut graph = CodeGraph::new();
    let main_id = make_node(&mut graph, "main.py", "main", UnitType::Function, Language::Python);
    let dunder_main = make_node(&mut graph, "app.py", "__main__", UnitType::Function, Language::Python);
    let other_id = make_node(&mut graph, "utils.py", "helper", UnitType::Function, Language::Python);

    assert!(graph.is_entry_point(main_id));
    assert!(graph.is_entry_point(dunder_main));
    assert!(!graph.is_entry_point(other_id));
}

#[test]
fn test_is_entry_point_wgsl_vertex() {
    let mut graph = CodeGraph::new();
    let vs_main = make_node(&mut graph, "shader.wgsl", "vs_main", UnitType::Function, Language::Wgsl);
    let fs_main = make_node(&mut graph, "shader.wgsl", "fs_main", UnitType::Function, Language::Wgsl);
    let cs_compute = make_node(&mut graph, "compute.wgsl", "cs_compute", UnitType::Function, Language::Wgsl);
    let helper = make_node(&mut graph, "shader.wgsl", "calculate", UnitType::Function, Language::Wgsl);

    assert!(graph.is_entry_point(vs_main));
    assert!(graph.is_entry_point(fs_main));
    assert!(graph.is_entry_point(cs_compute));
    assert!(!graph.is_entry_point(helper));
}

#[test]
fn test_is_entry_point_invalid_id() {
    let graph = CodeGraph::new();
    assert!(!graph.is_entry_point(NodeId(999)));
}

#[test]
fn test_find_orphan_non_entry_points() {
    let mut graph = CodeGraph::new();
    let main_id = make_node(&mut graph, "main.rs", "main", UnitType::Function, Language::Rust);
    let helper = make_node(&mut graph, "lib.rs", "helper", UnitType::Function, Language::Rust);
    let connected = make_node(&mut graph, "lib.rs", "connected", UnitType::Function, Language::Rust);
    graph.add_edge(CodeEdge::new(main_id, connected, EdgeType::Calls));

    // main and helper are orphans, but main is entry point
    let non_entry_orphans = graph.find_orphan_non_entry_points();
    assert_eq!(non_entry_orphans.len(), 1);
    assert!(non_entry_orphans.contains(&helper));
    assert!(!non_entry_orphans.contains(&main_id));
}

// ==================== Validation ====================

#[test]
fn test_validate_empty_graph() {
    let graph = CodeGraph::new();
    let result = graph.validate();

    assert_eq!(result.total_nodes, 0);
    assert_eq!(result.total_edges, 0);
    assert!(result.is_valid);
}

#[test]
fn test_validate_nodes_only() {
    let mut graph = CodeGraph::new();
    make_node(&mut graph, "a.rs", "func", UnitType::Function, Language::Rust);
    make_node(&mut graph, "b.py", "func", UnitType::Function, Language::Python);

    let result = graph.validate();

    assert_eq!(result.total_nodes, 2);
    assert_eq!(result.total_edges, 0);
    assert!(result.is_valid); // No edges means orphans are OK
}

#[test]
fn test_validate_fully_connected() {
    let mut graph = CodeGraph::new();
    let a = make_node(&mut graph, "a.rs", "a", UnitType::Function, Language::Rust);
    let b = make_node(&mut graph, "b.rs", "b", UnitType::Function, Language::Rust);
    graph.add_edge(CodeEdge::new(a, b, EdgeType::Calls));

    let result = graph.validate();

    assert_eq!(result.total_nodes, 2);
    assert_eq!(result.total_edges, 1);
    assert_eq!(result.orphan_issues, 0);
    assert!(result.is_valid);
}

#[test]
fn test_validate_with_entry_point_orphan() {
    let mut graph = CodeGraph::new();
    let main_id = make_node(&mut graph, "main.rs", "main", UnitType::Function, Language::Rust);
    let lib_fn = make_node(&mut graph, "lib.rs", "helper", UnitType::Function, Language::Rust);
    let lib_fn2 = make_node(&mut graph, "lib.rs", "other", UnitType::Function, Language::Rust);
    graph.add_edge(CodeEdge::new(lib_fn, lib_fn2, EdgeType::Calls));

    // main is orphan but is entry point
    let result = graph.validate();

    assert_eq!(result.orphan_entry_points, 1);
    assert_eq!(result.orphan_issues, 0);
    assert!(result.is_valid);
}

#[test]
fn test_validate_with_orphan_issue() {
    let mut graph = CodeGraph::new();
    let a = make_node(&mut graph, "a.rs", "connected_a", UnitType::Function, Language::Rust);
    let b = make_node(&mut graph, "b.rs", "connected_b", UnitType::Function, Language::Rust);
    let orphan = make_node(&mut graph, "c.rs", "orphan_func", UnitType::Function, Language::Rust);
    graph.add_edge(CodeEdge::new(a, b, EdgeType::Calls));

    // orphan is not an entry point
    let result = graph.validate();

    assert_eq!(result.orphan_issues, 1);
    assert!(!result.is_valid);
}

#[test]
fn test_validate_summary_counts() {
    let mut graph = CodeGraph::new();
    make_node(&mut graph, "a.rs", "f1", UnitType::Function, Language::Rust);
    make_node(&mut graph, "b.rs", "f2", UnitType::Function, Language::Rust);
    make_node(&mut graph, "c.py", "f3", UnitType::Function, Language::Python);
    make_node(&mut graph, "d.wgsl", "s1", UnitType::Struct, Language::Wgsl);

    let a = graph.nodes()[0].id;
    let b = graph.nodes()[1].id;
    graph.add_edge(CodeEdge::new(a, b, EdgeType::Calls));

    let result = graph.validate();

    assert_eq!(result.total_nodes, 4);
    assert_eq!(result.total_edges, 1);
    assert_eq!(result.nodes_by_language.get(&Language::Rust), Some(&2));
    assert_eq!(result.nodes_by_language.get(&Language::Python), Some(&1));
    assert_eq!(result.nodes_by_language.get(&Language::Wgsl), Some(&1));
    assert_eq!(result.edges_by_type.get(&EdgeType::Calls), Some(&1));
}
