//! Whitebox tests for test mapping.

use trinity_harness::graph::{
    create_test_edges, CodeGraph, CodeNode, ConventionMapper, EdgeType, MappingSource,
    NodeId, PythonTestMapper, RustTestMapper,
};
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

// ==================== Convention Mapper ====================

#[test]
fn test_convention_mapper_test_prefix() {
    let mut graph = CodeGraph::new();

    // Code node
    let code_id = make_node(&mut graph, "src/lib.rs", "my_function", UnitType::Function, Language::Rust);

    // Test node with test_ prefix
    let test_id = make_node(&mut graph, "tests/test_my_function.rs", "test_my_function", UnitType::Function, Language::Rust);

    let mapper = ConventionMapper::new();
    let (mappings, stats) = mapper.map_tests(&graph);

    assert_eq!(stats.tests_processed, 1);
    assert_eq!(stats.tests_mapped, 1);

    let mapping = &mappings[0];
    assert_eq!(mapping.test_id, test_id);
    assert!(mapping.targets.contains(&code_id));
    assert_eq!(mapping.source, MappingSource::Convention);
}

#[test]
fn test_convention_mapper_test_suffix() {
    let mut graph = CodeGraph::new();

    let code_id = make_node(&mut graph, "src/lib.rs", "parser", UnitType::Function, Language::Rust);
    let test_id = make_node(&mut graph, "tests/parser_test.rs", "parser_test", UnitType::Function, Language::Rust);

    let mapper = ConventionMapper::new();
    let (mappings, _) = mapper.map_tests(&graph);

    let mapping = mappings.iter().find(|m| m.test_id == test_id).unwrap();
    assert!(mapping.targets.contains(&code_id));
}

#[test]
fn test_convention_mapper_multiple_targets() {
    let mut graph = CodeGraph::new();

    // Multiple code nodes with same base name
    let fn_id = make_node(&mut graph, "src/a.rs", "process", UnitType::Function, Language::Rust);
    let struct_id = make_node(&mut graph, "src/b.rs", "process", UnitType::Struct, Language::Rust);

    // Test that maps to both
    let test_id = make_node(&mut graph, "tests/test.rs", "test_process", UnitType::Function, Language::Rust);

    let mapper = ConventionMapper::new();
    let (mappings, _) = mapper.map_tests(&graph);

    let mapping = mappings.iter().find(|m| m.test_id == test_id).unwrap();
    assert!(mapping.targets.contains(&fn_id));
    assert!(mapping.targets.contains(&struct_id));
}

#[test]
fn test_convention_mapper_unmapped_test() {
    let mut graph = CodeGraph::new();

    // Code node
    make_node(&mut graph, "src/lib.rs", "real_function", UnitType::Function, Language::Rust);

    // Test that doesn't match any code
    let test_id = make_node(&mut graph, "tests/test.rs", "test_nonexistent", UnitType::Function, Language::Rust);

    let mapper = ConventionMapper::new();
    let (mappings, stats) = mapper.map_tests(&graph);

    assert_eq!(stats.tests_unmapped, 1);

    let mapping = mappings.iter().find(|m| m.test_id == test_id).unwrap();
    assert!(mapping.targets.is_empty());
    assert_eq!(mapping.source, MappingSource::Unmapped);
}

#[test]
fn test_convention_mapper_blackbox_prefix() {
    let mut graph = CodeGraph::new();

    let code_id = make_node(&mut graph, "src/lib.rs", "parser", UnitType::Function, Language::Rust);
    let test_id = make_node(&mut graph, "tests/blackbox.rs", "blackbox_parser", UnitType::Function, Language::Rust);

    let mapper = ConventionMapper::new();
    let (mappings, _) = mapper.map_tests(&graph);

    let mapping = mappings.iter().find(|m| m.test_id == test_id).unwrap();
    assert!(mapping.targets.contains(&code_id));
}

#[test]
fn test_convention_mapper_whitebox_prefix() {
    let mut graph = CodeGraph::new();

    let code_id = make_node(&mut graph, "src/lib.rs", "validator", UnitType::Function, Language::Rust);
    let test_id = make_node(&mut graph, "tests/whitebox.rs", "whitebox_validator", UnitType::Function, Language::Rust);

    let mapper = ConventionMapper::new();
    let (mappings, _) = mapper.map_tests(&graph);

    let mapping = mappings.iter().find(|m| m.test_id == test_id).unwrap();
    assert!(mapping.targets.contains(&code_id));
}

#[test]
fn test_convention_mapper_python_test_class() {
    let mut graph = CodeGraph::new();

    let class_id = make_node(&mut graph, "src/app.py", "Calculator", UnitType::Class, Language::Python);
    let test_id = make_node(&mut graph, "tests/test_app.py", "TestCalculator", UnitType::Class, Language::Python);

    let mapper = ConventionMapper::new();
    let (mappings, _) = mapper.map_tests(&graph);

    let mapping = mappings.iter().find(|m| m.test_id == test_id).unwrap();
    assert!(mapping.targets.contains(&class_id));
}

#[test]
fn test_convention_mapper_stats() {
    let mut graph = CodeGraph::new();

    // 2 code nodes
    make_node(&mut graph, "src/a.rs", "func_a", UnitType::Function, Language::Rust);
    make_node(&mut graph, "src/b.rs", "func_b", UnitType::Function, Language::Rust);

    // 3 tests: 2 mapped, 1 unmapped
    make_node(&mut graph, "tests/a.rs", "test_func_a", UnitType::Function, Language::Rust);
    make_node(&mut graph, "tests/b.rs", "test_func_b", UnitType::Function, Language::Rust);
    make_node(&mut graph, "tests/c.rs", "test_unknown", UnitType::Function, Language::Rust);

    let mapper = ConventionMapper::new();
    let (_, stats) = mapper.map_tests(&graph);

    assert_eq!(stats.tests_processed, 3);
    assert_eq!(stats.tests_mapped, 2);
    assert_eq!(stats.tests_unmapped, 1);
    assert_eq!(stats.by_source.get(&MappingSource::Convention), Some(&2));
    assert_eq!(stats.by_source.get(&MappingSource::Unmapped), Some(&1));
}

// ==================== Rust Test Mapper ====================

#[test]
fn test_rust_find_blackbox_tests() {
    let mut graph = CodeGraph::new();

    // Blackbox test
    let bb_id = make_node(&mut graph, "crates/foo/tests/test_parser.rs", "test_parse", UnitType::Function, Language::Rust);

    // Not a blackbox test (in src)
    make_node(&mut graph, "crates/foo/src/lib.rs", "test_internal", UnitType::Function, Language::Rust);

    // Python test (wrong lang)
    make_node(&mut graph, "tests/test_app.py", "test_main", UnitType::Function, Language::Python);

    let mapper = RustTestMapper::new();
    let tests = mapper.find_blackbox_tests(&graph);

    assert_eq!(tests.len(), 1);
    assert!(tests.contains(&bb_id));
}

#[test]
fn test_rust_find_unit_tests() {
    let mut graph = CodeGraph::new();

    // Unit test (in src, starts with test_)
    let unit_id = make_node(&mut graph, "crates/foo/src/lib.rs", "test_internal", UnitType::Function, Language::Rust);

    // Blackbox test (not a unit test)
    make_node(&mut graph, "crates/foo/tests/test_parser.rs", "test_parse", UnitType::Function, Language::Rust);

    // Regular function (not a test)
    make_node(&mut graph, "crates/foo/src/lib.rs", "process", UnitType::Function, Language::Rust);

    let mapper = RustTestMapper::new();
    let tests = mapper.find_unit_tests(&graph);

    assert_eq!(tests.len(), 1);
    assert!(tests.contains(&unit_id));
}

// ==================== Python Test Mapper ====================

#[test]
fn test_python_find_tests() {
    let mut graph = CodeGraph::new();

    // Python tests
    let test1 = make_node(&mut graph, "tests/test_app.py", "test_main", UnitType::Function, Language::Python);
    let test2 = make_node(&mut graph, "tests/app_test.py", "test_helper", UnitType::Function, Language::Python);
    let test3 = make_node(&mut graph, "tests/test_suite.py", "TestSuite", UnitType::Class, Language::Python);

    // Not a test
    make_node(&mut graph, "src/app.py", "helper", UnitType::Function, Language::Python);

    // Rust test (wrong lang)
    make_node(&mut graph, "tests/test.rs", "test_rust", UnitType::Function, Language::Rust);

    let mapper = PythonTestMapper::new();
    let tests = mapper.find_python_tests(&graph);

    assert_eq!(tests.len(), 3);
    assert!(tests.contains(&test1));
    assert!(tests.contains(&test2));
    assert!(tests.contains(&test3));
}

// ==================== Edge Creation ====================

#[test]
fn test_create_test_edges() {
    let mut graph = CodeGraph::new();

    let code_id = make_node(&mut graph, "src/lib.rs", "func", UnitType::Function, Language::Rust);
    let test_id = make_node(&mut graph, "tests/test.rs", "test_func", UnitType::Function, Language::Rust);

    let mapper = ConventionMapper::new();
    let (mappings, _) = mapper.map_tests(&graph);

    let count = create_test_edges(&mut graph, &mappings);

    assert_eq!(count, 1);
    assert_eq!(graph.edges().len(), 1);

    let edge = &graph.edges()[0];
    assert_eq!(edge.source, test_id);
    assert_eq!(edge.target, code_id);
    assert_eq!(edge.edge_type, EdgeType::Tests);
}

#[test]
fn test_create_test_edges_skips_unmapped() {
    let mut graph = CodeGraph::new();

    // Only a test, no matching code
    make_node(&mut graph, "tests/test.rs", "test_nonexistent", UnitType::Function, Language::Rust);

    let mapper = ConventionMapper::new();
    let (mappings, _) = mapper.map_tests(&graph);

    let count = create_test_edges(&mut graph, &mappings);

    assert_eq!(count, 0);
    assert!(graph.edges().is_empty());
}

#[test]
fn test_create_test_edges_multiple_targets() {
    let mut graph = CodeGraph::new();

    let code1 = make_node(&mut graph, "src/a.rs", "handler", UnitType::Function, Language::Rust);
    let code2 = make_node(&mut graph, "src/b.rs", "handler", UnitType::Function, Language::Rust);
    make_node(&mut graph, "tests/test.rs", "test_handler", UnitType::Function, Language::Rust);

    let mapper = ConventionMapper::new();
    let (mappings, _) = mapper.map_tests(&graph);

    let count = create_test_edges(&mut graph, &mappings);

    assert_eq!(count, 2);
    assert_eq!(graph.edges().len(), 2);

    let targets: Vec<_> = graph.edges().iter().map(|e| e.target).collect();
    assert!(targets.contains(&code1));
    assert!(targets.contains(&code2));
}

#[test]
fn test_edges_are_tests_type() {
    let mut graph = CodeGraph::new();

    make_node(&mut graph, "src/lib.rs", "func", UnitType::Function, Language::Rust);
    make_node(&mut graph, "tests/test.rs", "test_func", UnitType::Function, Language::Rust);

    let mapper = ConventionMapper::new();
    let (mappings, _) = mapper.map_tests(&graph);
    create_test_edges(&mut graph, &mappings);

    for edge in graph.edges() {
        assert_eq!(edge.edge_type, EdgeType::Tests);
    }
}
