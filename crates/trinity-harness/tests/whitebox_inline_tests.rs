//! Whitebox tests for inline test mapping.

use trinity_harness::graph::{
    create_test_edges, CodeGraph, CodeNode, EdgeType, InlineTestMapper, MappingSource,
    NodeId,
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

// ==================== Find Inline Tests ====================

#[test]
fn test_find_inline_tests_basic() {
    let mut graph = CodeGraph::new();

    // Production code
    make_node(&mut graph, "src/lib.rs", "compute", UnitType::Function, Language::Rust);

    // Inline test in same file
    let test_id = make_node(&mut graph, "src/lib.rs", "test_compute", UnitType::Function, Language::Rust);

    let mapper = InlineTestMapper::new();
    let tests = mapper.find_inline_tests(&graph);

    assert_eq!(tests.len(), 1);
    assert!(tests.contains(&test_id));
}

#[test]
fn test_find_inline_tests_excludes_tests_dir() {
    let mut graph = CodeGraph::new();

    // Blackbox test (in tests/ dir) - should NOT be found
    make_node(&mut graph, "tests/test_lib.rs", "test_compute", UnitType::Function, Language::Rust);

    // Inline test (in src/ dir) - should be found
    let inline_id = make_node(&mut graph, "src/lib.rs", "test_compute", UnitType::Function, Language::Rust);

    let mapper = InlineTestMapper::new();
    let tests = mapper.find_inline_tests(&graph);

    assert_eq!(tests.len(), 1);
    assert!(tests.contains(&inline_id));
}

#[test]
fn test_find_inline_tests_multiple() {
    let mut graph = CodeGraph::new();

    make_node(&mut graph, "src/lib.rs", "func_a", UnitType::Function, Language::Rust);
    let test1 = make_node(&mut graph, "src/lib.rs", "test_func_a", UnitType::Function, Language::Rust);
    let test2 = make_node(&mut graph, "src/lib.rs", "test_func_b", UnitType::Function, Language::Rust);

    let mapper = InlineTestMapper::new();
    let tests = mapper.find_inline_tests(&graph);

    assert_eq!(tests.len(), 2);
    assert!(tests.contains(&test1));
    assert!(tests.contains(&test2));
}

#[test]
fn test_find_inline_tests_excludes_non_functions() {
    let mut graph = CodeGraph::new();

    // Struct starting with test_ (shouldn't be found)
    make_node(&mut graph, "src/lib.rs", "test_struct", UnitType::Struct, Language::Rust);

    // Function (should be found)
    let test_id = make_node(&mut graph, "src/lib.rs", "test_func", UnitType::Function, Language::Rust);

    let mapper = InlineTestMapper::new();
    let tests = mapper.find_inline_tests(&graph);

    assert_eq!(tests.len(), 1);
    assert!(tests.contains(&test_id));
}

// ==================== Map Inline Tests ====================

#[test]
fn test_map_inline_tests_basic() {
    let mut graph = CodeGraph::new();

    // Production code
    let code_id = make_node(&mut graph, "src/lib.rs", "compute", UnitType::Function, Language::Rust);

    // Inline test
    let test_id = make_node(&mut graph, "src/lib.rs", "test_compute", UnitType::Function, Language::Rust);

    let mapper = InlineTestMapper::new();
    let (mappings, stats) = mapper.map_inline_tests(&graph);

    assert_eq!(stats.tests_processed, 1);
    assert_eq!(stats.tests_mapped, 1);

    let mapping = &mappings[0];
    assert_eq!(mapping.test_id, test_id);
    assert!(mapping.targets.contains(&code_id));
}

#[test]
fn test_map_inline_tests_maps_to_all_same_file() {
    let mut graph = CodeGraph::new();

    // Multiple code items in same file
    let code1 = make_node(&mut graph, "src/lib.rs", "func_a", UnitType::Function, Language::Rust);
    let code2 = make_node(&mut graph, "src/lib.rs", "func_b", UnitType::Function, Language::Rust);
    let struct1 = make_node(&mut graph, "src/lib.rs", "MyStruct", UnitType::Struct, Language::Rust);

    // Inline test
    let test_id = make_node(&mut graph, "src/lib.rs", "test_all", UnitType::Function, Language::Rust);

    let mapper = InlineTestMapper::new();
    let (mappings, stats) = mapper.map_inline_tests(&graph);

    let mapping = &mappings[0];
    assert_eq!(mapping.test_id, test_id);
    assert!(mapping.targets.contains(&code1));
    assert!(mapping.targets.contains(&code2));
    assert!(mapping.targets.contains(&struct1));
    assert_eq!(stats.edges_created, 3);
}

#[test]
fn test_map_inline_tests_excludes_self() {
    let mut graph = CodeGraph::new();

    // Only an inline test in file
    let test_id = make_node(&mut graph, "src/lib.rs", "test_something", UnitType::Function, Language::Rust);

    let mapper = InlineTestMapper::new();
    let (mappings, stats) = mapper.map_inline_tests(&graph);

    // Should be unmapped since no non-test code in file
    assert_eq!(stats.tests_unmapped, 1);
    let mapping = mappings.iter().find(|m| m.test_id == test_id).unwrap();
    assert!(mapping.targets.is_empty());
    assert_eq!(mapping.source, MappingSource::Unmapped);
}

#[test]
fn test_map_inline_tests_multiple_files() {
    let mut graph = CodeGraph::new();

    // File 1
    let code1 = make_node(&mut graph, "src/a.rs", "func_a", UnitType::Function, Language::Rust);
    let test1 = make_node(&mut graph, "src/a.rs", "test_func_a", UnitType::Function, Language::Rust);

    // File 2
    let code2 = make_node(&mut graph, "src/b.rs", "func_b", UnitType::Function, Language::Rust);
    let test2 = make_node(&mut graph, "src/b.rs", "test_func_b", UnitType::Function, Language::Rust);

    let mapper = InlineTestMapper::new();
    let (mappings, stats) = mapper.map_inline_tests(&graph);

    assert_eq!(stats.tests_processed, 2);
    assert_eq!(stats.tests_mapped, 2);

    // Each test should only map to code in its own file
    let mapping1 = mappings.iter().find(|m| m.test_id == test1).unwrap();
    assert!(mapping1.targets.contains(&code1));
    assert!(!mapping1.targets.contains(&code2));

    let mapping2 = mappings.iter().find(|m| m.test_id == test2).unwrap();
    assert!(mapping2.targets.contains(&code2));
    assert!(!mapping2.targets.contains(&code1));
}

// ==================== Edge Creation ====================

#[test]
fn test_create_edges_from_inline_mappings() {
    let mut graph = CodeGraph::new();

    make_node(&mut graph, "src/lib.rs", "compute", UnitType::Function, Language::Rust);
    make_node(&mut graph, "src/lib.rs", "test_compute", UnitType::Function, Language::Rust);

    let mapper = InlineTestMapper::new();
    let (mappings, _) = mapper.map_inline_tests(&graph);
    let count = create_test_edges(&mut graph, &mappings);

    assert_eq!(count, 1);

    let test_edges: Vec<_> = graph.edges()
        .iter()
        .filter(|e| e.edge_type == EdgeType::Tests)
        .collect();

    assert_eq!(test_edges.len(), 1);
}

#[test]
fn test_inline_mapping_source() {
    let mut graph = CodeGraph::new();

    make_node(&mut graph, "src/lib.rs", "compute", UnitType::Function, Language::Rust);
    make_node(&mut graph, "src/lib.rs", "test_compute", UnitType::Function, Language::Rust);

    let mapper = InlineTestMapper::new();
    let (mappings, _) = mapper.map_inline_tests(&graph);

    // Inline mappings use Convention source (same-file is a convention)
    assert_eq!(mappings[0].source, MappingSource::Convention);
}
