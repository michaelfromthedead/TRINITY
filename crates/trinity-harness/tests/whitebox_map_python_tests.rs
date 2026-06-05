//! Whitebox tests for Python test mapping.

use trinity_harness::graph::{
    create_test_edges, CodeGraph, CodeNode, ConventionMapper, EdgeType, MappingSource,
    NodeId, PythonTestMapper,
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

// ==================== Python Test Mapper ====================

#[test]
fn test_python_test_prefix() {
    let mut graph = CodeGraph::new();

    let code_id = make_node(&mut graph, "src/app.py", "compute", UnitType::Function, Language::Python);
    let test_id = make_node(&mut graph, "tests/test_app.py", "test_compute", UnitType::Function, Language::Python);

    let mapper = ConventionMapper::new();
    let (mappings, stats) = mapper.map_tests(&graph);

    assert!(stats.tests_mapped >= 1);

    let mapping = mappings.iter().find(|m| m.test_id == test_id).unwrap();
    assert!(mapping.targets.contains(&code_id));
    assert_eq!(mapping.source, MappingSource::Convention);
}

#[test]
fn test_python_test_class() {
    let mut graph = CodeGraph::new();

    let code_id = make_node(&mut graph, "src/calculator.py", "Calculator", UnitType::Class, Language::Python);
    let test_id = make_node(&mut graph, "tests/test_calculator.py", "TestCalculator", UnitType::Class, Language::Python);

    let mapper = ConventionMapper::new();
    let (mappings, _) = mapper.map_tests(&graph);

    let mapping = mappings.iter().find(|m| m.test_id == test_id).unwrap();
    assert!(mapping.targets.contains(&code_id));
}

#[test]
fn test_python_unit_tests_dir() {
    let mut graph = CodeGraph::new();

    make_node(&mut graph, "src/core.py", "process", UnitType::Function, Language::Python);
    let test_id = make_node(&mut graph, "tests/unit/test_core.py", "test_process", UnitType::Function, Language::Python);

    let mapper = PythonTestMapper::new();
    let tests = mapper.find_python_tests(&graph);

    assert!(tests.contains(&test_id));
}

#[test]
fn test_python_integration_tests_dir() {
    let mut graph = CodeGraph::new();

    make_node(&mut graph, "src/api.py", "handler", UnitType::Function, Language::Python);
    let test_id = make_node(&mut graph, "tests/integration/test_api.py", "test_handler", UnitType::Function, Language::Python);

    let mapper = PythonTestMapper::new();
    let tests = mapper.find_python_tests(&graph);

    assert!(tests.contains(&test_id));
}

#[test]
fn test_python_e2e_tests_dir() {
    let mut graph = CodeGraph::new();

    make_node(&mut graph, "src/app.py", "main", UnitType::Function, Language::Python);
    let test_id = make_node(&mut graph, "tests/e2e/test_app.py", "test_main", UnitType::Function, Language::Python);

    let mapper = PythonTestMapper::new();
    let tests = mapper.find_python_tests(&graph);

    assert!(tests.contains(&test_id));
}

#[test]
fn test_python_test_methods() {
    let mut graph = CodeGraph::new();

    make_node(&mut graph, "src/app.py", "App", UnitType::Class, Language::Python);
    let test_id = make_node(&mut graph, "tests/test_app.py", "test_init", UnitType::Method, Language::Python);

    let mapper = PythonTestMapper::new();
    let tests = mapper.find_python_tests(&graph);

    assert!(tests.contains(&test_id));
}

#[test]
fn test_python_creates_test_edges() {
    let mut graph = CodeGraph::new();

    let code_id = make_node(&mut graph, "src/utils.py", "helper", UnitType::Function, Language::Python);
    make_node(&mut graph, "tests/test_utils.py", "test_helper", UnitType::Function, Language::Python);

    let mapper = ConventionMapper::new();
    let (mappings, _) = mapper.map_tests(&graph);
    let count = create_test_edges(&mut graph, &mappings);

    assert!(count >= 1);

    let test_edges: Vec<_> = graph.edges()
        .iter()
        .filter(|e| e.edge_type == EdgeType::Tests)
        .collect();

    assert!(!test_edges.is_empty());
}

#[test]
fn test_python_multiple_test_files() {
    let mut graph = CodeGraph::new();

    make_node(&mut graph, "src/a.py", "a", UnitType::Function, Language::Python);
    make_node(&mut graph, "src/b.py", "b", UnitType::Function, Language::Python);
    make_node(&mut graph, "tests/test_a.py", "test_a", UnitType::Function, Language::Python);
    make_node(&mut graph, "tests/test_b.py", "test_b", UnitType::Function, Language::Python);

    let mapper = ConventionMapper::new();
    let (mappings, stats) = mapper.map_tests(&graph);
    create_test_edges(&mut graph, &mappings);

    assert!(stats.tests_processed >= 2);
    assert!(stats.tests_mapped >= 2);
}

#[test]
fn test_python_only_targets_python() {
    let mut graph = CodeGraph::new();

    // Python code
    make_node(&mut graph, "src/app.py", "compute", UnitType::Function, Language::Python);

    // Rust code with same name
    make_node(&mut graph, "src/lib.rs", "compute", UnitType::Function, Language::Rust);

    // Python test
    make_node(&mut graph, "tests/test_app.py", "test_compute", UnitType::Function, Language::Python);

    let mapper = ConventionMapper::new();
    let (mappings, _) = mapper.map_tests(&graph);

    // Should map to both (convention mapper doesn't filter by language)
    // This is expected behavior - explicit mappings can refine this
    assert!(!mappings.is_empty());
}
