//! Whitebox tests for result mapping.

use trinity_harness::graph::{
    create_test_edges, CodeGraph, CodeNode, NodeId, TestMapping, MappingSource,
};
use trinity_harness::parsers::{CodeUnit, ContentHashes, Language, UnitType};
use trinity_harness::runners::{
    map_results, get_test_targets, lookup_test_node, NodeResult, MappingResult, TestResult,
};

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

// ==================== NodeResult ====================

#[test]
fn test_node_result_new() {
    let result = NodeResult::new(NodeId(5));
    assert_eq!(result.node_id, NodeId(5));
    assert_eq!(result.passed, 0);
    assert_eq!(result.failed, 0);
    assert_eq!(result.total, 0);
}

#[test]
fn test_node_result_all_passed() {
    let mut result = NodeResult::new(NodeId(0));
    result.total = 3;
    result.passed = 3;

    assert!(result.all_passed());
}

#[test]
fn test_node_result_has_failures() {
    let mut result = NodeResult::new(NodeId(0));
    result.total = 3;
    result.passed = 2;
    result.failed = 1;

    assert!(result.has_failures());
    assert!(!result.all_passed());
}

#[test]
fn test_node_result_is_untested() {
    let result = NodeResult::new(NodeId(0));
    assert!(result.is_untested());
}

#[test]
fn test_node_result_pass_rate() {
    let mut result = NodeResult::new(NodeId(0));
    result.total = 4;
    result.passed = 3;
    result.failed = 1;

    assert_eq!(result.pass_rate(), 75.0);
}

#[test]
fn test_node_result_pass_rate_zero() {
    let result = NodeResult::new(NodeId(0));
    assert_eq!(result.pass_rate(), 0.0);
}

// ==================== MappingResult ====================

#[test]
fn test_mapping_result_new() {
    let result = MappingResult::new();
    assert!(result.by_node.is_empty());
    assert!(result.unmapped_tests.is_empty());
    assert_eq!(result.total_tests, 0);
}

#[test]
fn test_mapping_result_get() {
    let mut result = MappingResult::new();
    result.by_node.insert(NodeId(1), NodeResult::new(NodeId(1)));

    assert!(result.get(NodeId(1)).is_some());
    assert!(result.get(NodeId(99)).is_none());
}

#[test]
fn test_mapping_result_failed_nodes() {
    let mut result = MappingResult::new();

    let mut r1 = NodeResult::new(NodeId(1));
    r1.total = 1;
    r1.passed = 1;

    let mut r2 = NodeResult::new(NodeId(2));
    r2.total = 1;
    r2.failed = 1;

    result.by_node.insert(NodeId(1), r1);
    result.by_node.insert(NodeId(2), r2);

    let failed = result.failed_nodes();
    assert_eq!(failed.len(), 1);
    assert!(failed.contains(&NodeId(2)));
}

#[test]
fn test_mapping_result_passing_nodes() {
    let mut result = MappingResult::new();

    let mut r1 = NodeResult::new(NodeId(1));
    r1.total = 1;
    r1.passed = 1;

    let mut r2 = NodeResult::new(NodeId(2));
    r2.total = 1;
    r2.failed = 1;

    result.by_node.insert(NodeId(1), r1);
    result.by_node.insert(NodeId(2), r2);

    let passing = result.passing_nodes();
    assert_eq!(passing.len(), 1);
    assert!(passing.contains(&NodeId(1)));
}

// ==================== Lookup Test Node ====================

#[test]
fn test_lookup_test_node_exact() {
    let mut graph = CodeGraph::new();
    let id = make_node(&mut graph, "tests/test.rs", "test_foo", UnitType::Function, Language::Rust);

    let found = lookup_test_node(&graph, "test_foo");
    assert_eq!(found, Some(id));
}

#[test]
fn test_lookup_test_node_with_path() {
    let mut graph = CodeGraph::new();
    let id = make_node(&mut graph, "tests/test.rs", "test_bar", UnitType::Function, Language::Rust);

    let found = lookup_test_node(&graph, "module::test_bar");
    assert_eq!(found, Some(id));
}

#[test]
fn test_lookup_test_node_not_found() {
    let graph = CodeGraph::new();

    let found = lookup_test_node(&graph, "test_nonexistent");
    assert!(found.is_none());
}

// ==================== Get Test Targets ====================

#[test]
fn test_get_test_targets_with_edges() {
    let mut graph = CodeGraph::new();

    let code_id = make_node(&mut graph, "src/lib.rs", "compute", UnitType::Function, Language::Rust);
    let test_id = make_node(&mut graph, "tests/test.rs", "test_compute", UnitType::Function, Language::Rust);

    // Create test edge
    let mappings = vec![TestMapping {
        test_id,
        targets: vec![code_id],
        source: MappingSource::Convention,
    }];
    create_test_edges(&mut graph, &mappings);

    let targets = get_test_targets(&graph, test_id);
    assert_eq!(targets.len(), 1);
    assert!(targets.contains(&code_id));
}

#[test]
fn test_get_test_targets_no_edges() {
    let mut graph = CodeGraph::new();

    let test_id = make_node(&mut graph, "tests/test.rs", "test_orphan", UnitType::Function, Language::Rust);

    let targets = get_test_targets(&graph, test_id);
    assert!(targets.is_empty());
}

// ==================== Map Results ====================

#[test]
fn test_map_results_empty() {
    let graph = CodeGraph::new();
    let results: Vec<TestResult> = vec![];

    let mapped = map_results(&graph, &results);

    assert_eq!(mapped.total_tests, 0);
    assert!(mapped.by_node.is_empty());
}

#[test]
fn test_map_results_with_graph() {
    let mut graph = CodeGraph::new();

    let code_id = make_node(&mut graph, "src/lib.rs", "compute", UnitType::Function, Language::Rust);
    let test_id = make_node(&mut graph, "tests/test.rs", "test_compute", UnitType::Function, Language::Rust);

    // Create test edge
    let mappings = vec![TestMapping {
        test_id,
        targets: vec![code_id],
        source: MappingSource::Convention,
    }];
    create_test_edges(&mut graph, &mappings);

    // Run tests
    let test_results = vec![TestResult::passed("test_compute", 100)];

    let mapped = map_results(&graph, &test_results);

    assert_eq!(mapped.total_tests, 1);
    assert_eq!(mapped.mapped_tests, 1);
    
    let node_result = mapped.get(code_id);
    assert!(node_result.is_some());
    assert_eq!(node_result.unwrap().passed, 1);
}
