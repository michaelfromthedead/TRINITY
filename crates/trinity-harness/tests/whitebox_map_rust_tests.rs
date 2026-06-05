//! Whitebox tests for Rust test mapping via GraphBuilder.

use trinity_harness::graph::{CodeGraph, CodeNode, EdgeType, MappingStats, NodeId};
use trinity_harness::parsers::{CodeUnit, ContentHashes, Language, UnitType};
use std::path::Path;

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

// ==================== MappingStats ====================

#[test]
fn test_mapping_stats_default() {
    let stats = MappingStats::default();
    assert_eq!(stats.tests_processed, 0);
    assert_eq!(stats.tests_mapped, 0);
    assert_eq!(stats.tests_unmapped, 0);
    assert_eq!(stats.edges_created, 0);
}

#[test]
fn test_mapping_stats_new() {
    let stats = MappingStats::new();
    assert_eq!(stats.tests_processed, 0);
    assert_eq!(stats.tests_mapped, 0);
}

// ==================== Graph with Tests ====================

#[test]
fn test_graph_has_test_edges_after_mapping() {
    use trinity_harness::graph::{create_test_edges, ConventionMapper};

    let mut graph = CodeGraph::new();

    // Source code
    make_node(&mut graph, "src/lib.rs", "compute", UnitType::Function, Language::Rust);

    // Test code
    make_node(&mut graph, "tests/test_compute.rs", "test_compute", UnitType::Function, Language::Rust);

    let mapper = ConventionMapper::new();
    let (mappings, _) = mapper.map_tests(&graph);
    let count = create_test_edges(&mut graph, &mappings);

    assert!(count >= 1, "Should create test edges");

    let test_edges: Vec<_> = graph.edges()
        .iter()
        .filter(|e| e.edge_type == EdgeType::Tests)
        .collect();

    assert!(!test_edges.is_empty(), "Graph should have Tests edges");
}

#[test]
fn test_crates_tests_pattern() {
    use trinity_harness::graph::{create_test_edges, ConventionMapper};

    let mut graph = CodeGraph::new();

    // Simulate crates/*/tests/*.rs pattern
    make_node(&mut graph, "crates/harness/src/lib.rs", "harness_fn", UnitType::Function, Language::Rust);
    make_node(&mut graph, "crates/harness/tests/test_harness.rs", "test_harness_fn", UnitType::Function, Language::Rust);

    let mapper = ConventionMapper::new();
    let (mappings, stats) = mapper.map_tests(&graph);

    assert!(stats.tests_processed >= 1, "Should process crates test");
}

#[test]
fn test_multiple_crates() {
    use trinity_harness::graph::{create_test_edges, ConventionMapper};

    let mut graph = CodeGraph::new();

    // Multiple crates
    make_node(&mut graph, "crates/foo/src/lib.rs", "foo_fn", UnitType::Function, Language::Rust);
    make_node(&mut graph, "crates/bar/src/lib.rs", "bar_fn", UnitType::Function, Language::Rust);
    make_node(&mut graph, "crates/foo/tests/test_foo.rs", "test_foo_fn", UnitType::Function, Language::Rust);
    make_node(&mut graph, "crates/bar/tests/test_bar.rs", "test_bar_fn", UnitType::Function, Language::Rust);

    let mapper = ConventionMapper::new();
    let (mappings, stats) = mapper.map_tests(&graph);
    create_test_edges(&mut graph, &mappings);

    assert!(stats.tests_processed >= 2, "Should process multiple crate tests");
}

#[test]
fn test_blackbox_test_naming() {
    use trinity_harness::graph::{create_test_edges, ConventionMapper};

    let mut graph = CodeGraph::new();

    make_node(&mut graph, "src/parser.rs", "parser", UnitType::Function, Language::Rust);
    make_node(&mut graph, "tests/blackbox_parser.rs", "blackbox_parser", UnitType::Function, Language::Rust);

    let mapper = ConventionMapper::new();
    let (mappings, stats) = mapper.map_tests(&graph);
    let count = create_test_edges(&mut graph, &mappings);

    assert!(stats.tests_mapped >= 1, "blackbox_ prefix should map");
    assert!(count >= 1, "Should create edges for blackbox tests");
}

#[test]
fn test_whitebox_test_naming() {
    use trinity_harness::graph::{create_test_edges, ConventionMapper};

    let mut graph = CodeGraph::new();

    make_node(&mut graph, "src/validator.rs", "validator", UnitType::Function, Language::Rust);
    make_node(&mut graph, "tests/whitebox_validator.rs", "whitebox_validator", UnitType::Function, Language::Rust);

    let mapper = ConventionMapper::new();
    let (mappings, stats) = mapper.map_tests(&graph);
    let count = create_test_edges(&mut graph, &mappings);

    assert!(stats.tests_mapped >= 1, "whitebox_ prefix should map");
    assert!(count >= 1, "Should create edges for whitebox tests");
}

#[test]
fn test_inline_unit_tests() {
    use trinity_harness::graph::{create_test_edges, ConventionMapper, RustTestMapper};

    let mut graph = CodeGraph::new();

    // Inline test in source file
    make_node(&mut graph, "src/lib.rs", "compute", UnitType::Function, Language::Rust);
    make_node(&mut graph, "src/lib.rs", "test_compute", UnitType::Function, Language::Rust);

    let rust_mapper = RustTestMapper::new();
    let unit_tests = rust_mapper.find_unit_tests(&graph);

    assert!(unit_tests.len() >= 1, "Should find inline unit test");
}

#[test]
fn test_stats_edges_created() {
    use trinity_harness::graph::{create_test_edges, ConventionMapper};

    let mut graph = CodeGraph::new();

    make_node(&mut graph, "src/a.rs", "a", UnitType::Function, Language::Rust);
    make_node(&mut graph, "src/b.rs", "b", UnitType::Function, Language::Rust);
    make_node(&mut graph, "tests/test_a.rs", "test_a", UnitType::Function, Language::Rust);
    make_node(&mut graph, "tests/test_b.rs", "test_b", UnitType::Function, Language::Rust);

    let mapper = ConventionMapper::new();
    let (mappings, mut stats) = mapper.map_tests(&graph);
    let count = create_test_edges(&mut graph, &mappings);
    stats.edges_created = count;

    assert_eq!(stats.edges_created, count);
    assert!(count >= 2, "Should create multiple edges");
}
