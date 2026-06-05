//! Blackbox tests for test mapping integration.

use std::path::Path;
use tempfile::TempDir;
use trinity_harness::graph::{
    create_test_edges, ConventionMapper, EdgeType, GraphBuilder,
};
use trinity_harness::parsers::ParserRegistry;

fn create_test_dir() -> TempDir {
    TempDir::new().expect("Failed to create temp dir")
}

fn write_file(dir: &Path, name: &str, content: &str) {
    let path = dir.join(name);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(path, content).expect("Failed to write file");
}

#[test]
fn test_rust_convention_mapping() {
    let dir = create_test_dir();
    let root = dir.path();

    // Create source files
    write_file(root, "src/parser.rs", r#"
fn parse() -> i32 {
    42
}

fn validate() {
    // validation
}
"#);

    // Create test files
    write_file(root, "tests/test_parser.rs", r#"
fn test_parse() {
    assert_eq!(parse(), 42);
}

fn test_validate() {
    validate();
}
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, _) = builder.full_scan(root).expect("Scan failed");

    let mapper = ConventionMapper::new();
    let (mappings, stats) = mapper.map_tests(&graph);

    // Should find tests
    assert!(stats.tests_processed >= 2, "Should process at least 2 tests");
    assert!(stats.tests_mapped >= 2, "Should map at least 2 tests");
}

#[test]
fn test_python_convention_mapping() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/calculator.py", r#"
class Calculator:
    def add(self, a, b):
        return a + b

def compute():
    return 42
"#);

    write_file(root, "tests/test_calculator.py", r#"
def test_compute():
    assert compute() == 42

class TestCalculator:
    def test_add(self):
        pass
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, _) = builder.full_scan(root).expect("Scan failed");

    let mapper = ConventionMapper::new();
    let (mappings, stats) = mapper.map_tests(&graph);

    assert!(stats.tests_processed >= 2, "Should process at least 2 tests");
}

#[test]
fn test_mixed_language_mapping() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/core.rs", r#"
fn process() -> i32 {
    42
}
"#);

    write_file(root, "src/app.py", r#"
def process():
    return 42
"#);

    write_file(root, "tests/test_core.rs", r#"
fn test_process() {
    assert_eq!(process(), 42);
}
"#);

    write_file(root, "tests/test_app.py", r#"
def test_process():
    assert process() == 42
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, _) = builder.full_scan(root).expect("Scan failed");

    let mapper = ConventionMapper::new();
    let (_, stats) = mapper.map_tests(&graph);

    assert!(stats.tests_processed >= 2);
}

#[test]
fn test_create_edges_integration() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/lib.rs", r#"
fn helper() -> i32 {
    42
}
"#);

    write_file(root, "tests/test_lib.rs", r#"
fn test_helper() {
    assert_eq!(helper(), 42);
}
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");

    let mapper = ConventionMapper::new();
    let (mappings, _) = mapper.map_tests(&graph);

    let edge_count = create_test_edges(&mut graph, &mappings);

    // Should create at least one Tests edge
    assert!(edge_count >= 1, "Should create at least 1 test edge");

    let test_edges: Vec<_> = graph.edges()
        .iter()
        .filter(|e| e.edge_type == EdgeType::Tests)
        .collect();

    assert!(!test_edges.is_empty(), "Should have Tests edges");
}

#[test]
fn test_unmapped_tests_reported() {
    let dir = create_test_dir();
    let root = dir.path();

    // Only tests, no matching code
    write_file(root, "tests/test_nonexistent.rs", r#"
fn test_something_that_doesnt_exist() {
    // This test has no matching code
}
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, _) = builder.full_scan(root).expect("Scan failed");

    let mapper = ConventionMapper::new();
    let (_, stats) = mapper.map_tests(&graph);

    assert!(stats.tests_unmapped >= 1, "Should have at least 1 unmapped test");
}

#[test]
fn test_blackbox_whitebox_prefixes() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/parser.rs", r#"
fn parser() -> i32 {
    42
}
"#);

    write_file(root, "tests/blackbox_test.rs", r#"
fn blackbox_parser() {
    // Blackbox test
}
"#);

    write_file(root, "tests/whitebox_test.rs", r#"
fn whitebox_parser() {
    // Whitebox test
}
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, _) = builder.full_scan(root).expect("Scan failed");

    let mapper = ConventionMapper::new();
    let (_, stats) = mapper.map_tests(&graph);

    // Both blackbox and whitebox tests should be processed
    assert!(stats.tests_processed >= 2);
}

#[test]
fn test_large_test_suite() {
    let dir = create_test_dir();
    let root = dir.path();

    // Create many code files
    for i in 0..10 {
        write_file(root, &format!("src/module_{}.rs", i), &format!(r#"
fn func_{i}() -> i32 {{
    {i}
}}
"#, i = i));
    }

    // Create matching tests
    for i in 0..10 {
        write_file(root, &format!("tests/test_module_{}.rs", i), &format!(r#"
fn test_func_{i}() {{
    assert_eq!(func_{i}(), {i});
}}
"#, i = i));
    }

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, scan_stats) = builder.full_scan(root).expect("Scan failed");
    assert_eq!(scan_stats.files_scanned, 20);

    let mapper = ConventionMapper::new();
    let (_, stats) = mapper.map_tests(&graph);

    assert_eq!(stats.tests_processed, 10);
    assert!(stats.tests_mapped >= 5, "Should map at least half the tests");
}

#[test]
fn test_empty_project() {
    let dir = create_test_dir();
    let root = dir.path();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, _) = builder.full_scan(root).expect("Scan failed");

    let mapper = ConventionMapper::new();
    let (mappings, stats) = mapper.map_tests(&graph);

    assert_eq!(stats.tests_processed, 0);
    assert!(mappings.is_empty());
}

#[test]
fn test_full_pipeline_with_mapping() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/lib.rs", r#"
fn compute() -> i32 {
    42
}

fn process(x: i32) -> i32 {
    x * 2
}
"#);

    write_file(root, "tests/integration.rs", r#"
fn test_compute() {
    assert_eq!(compute(), 42);
}

fn test_process() {
    assert_eq!(process(21), 42);
}
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    // Full scan
    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");

    // Dependency analysis
    builder.analyze_dependencies(root, &mut graph).expect("Dep analysis failed");

    // Test mapping
    let mapper = ConventionMapper::new();
    let (mappings, stats) = mapper.map_tests(&graph);
    let edge_count = create_test_edges(&mut graph, &mappings);

    // Validate
    let result = graph.validate();

    assert!(result.total_nodes >= 4);
    assert!(edge_count >= 2, "Should create test edges");
    assert!(stats.tests_mapped >= 2);

    // Should have Tests edges
    let test_edge_count = result.edges_by_type.get(&EdgeType::Tests).copied().unwrap_or(0);
    assert!(test_edge_count >= 2, "Should have Tests edges in graph");
}
