//! Blackbox tests for Python test mapping via GraphBuilder.

use std::path::Path;
use tempfile::TempDir;
use trinity_harness::graph::{EdgeType, GraphBuilder};
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
fn test_map_python_tests_basic() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/app.py", r#"
def compute():
    return 42
"#);

    write_file(root, "tests/test_app.py", r#"
def test_compute():
    assert compute() == 42
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");
    let stats = builder.map_python_tests(root, &mut graph, None).expect("Mapping failed");

    assert!(stats.tests_processed >= 1);
    assert!(stats.tests_mapped >= 1);

    let test_edges: Vec<_> = graph.edges()
        .iter()
        .filter(|e| e.edge_type == EdgeType::Tests)
        .collect();

    assert!(!test_edges.is_empty());
}

#[test]
fn test_map_unit_tests_directory() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/core.py", r#"
def process():
    return 42
"#);

    write_file(root, "tests/unit/test_core.py", r#"
def test_process():
    assert process() == 42
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");
    let stats = builder.map_python_tests(root, &mut graph, None).expect("Mapping failed");

    assert!(stats.tests_processed >= 1);
}

#[test]
fn test_map_integration_tests_directory() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/api.py", r#"
def handler():
    return "ok"
"#);

    write_file(root, "tests/integration/test_api.py", r#"
def test_handler():
    assert handler() == "ok"
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");
    let stats = builder.map_python_tests(root, &mut graph, None).expect("Mapping failed");

    assert!(stats.tests_processed >= 1);
}

#[test]
fn test_map_e2e_tests_directory() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/app.py", r#"
def main():
    return 0
"#);

    write_file(root, "tests/e2e/test_app.py", r#"
def test_main():
    assert main() == 0
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");
    let stats = builder.map_python_tests(root, &mut graph, None).expect("Mapping failed");

    assert!(stats.tests_processed >= 1);
}

#[test]
fn test_map_test_classes() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/calculator.py", r#"
class Calculator:
    def add(self, a, b):
        return a + b
"#);

    write_file(root, "tests/test_calculator.py", r#"
class TestCalculator:
    def test_add(self):
        calc = Calculator()
        assert calc.add(1, 2) == 3
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");
    let stats = builder.map_python_tests(root, &mut graph, None).expect("Mapping failed");

    assert!(stats.tests_processed >= 1);
}

#[test]
fn test_map_multiple_test_types() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/core.py", r#"
def helper():
    return 1

class Service:
    def run(self):
        return 2
"#);

    write_file(root, "tests/unit/test_helper.py", r#"
def test_helper():
    assert helper() == 1
"#);

    write_file(root, "tests/integration/test_service.py", r#"
class TestService:
    def test_run(self):
        pass
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, scan_stats) = builder.full_scan(root).expect("Scan failed");
    assert_eq!(scan_stats.files_scanned, 3);

    let stats = builder.map_python_tests(root, &mut graph, None).expect("Mapping failed");

    assert!(stats.tests_processed >= 2);
}

#[test]
fn test_full_pipeline_with_python_mapping() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/lib.py", r#"
def compute():
    return helper()

def helper():
    return 42
"#);

    write_file(root, "tests/test_lib.py", r#"
def test_compute():
    assert compute() == 42

def test_helper():
    assert helper() == 42
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    // Full pipeline
    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");
    builder.analyze_dependencies(root, &mut graph).expect("Dep analysis failed");
    let map_stats = builder.map_python_tests(root, &mut graph, None).expect("Mapping failed");

    // Validate
    let result = graph.validate();

    assert!(result.total_nodes >= 4);
    assert!(map_stats.edges_created >= 2);

    // Should have Tests edges
    let test_edge_count = result.edges_by_type.get(&EdgeType::Tests).copied().unwrap_or(0);
    assert!(test_edge_count >= 2);
}

#[test]
fn test_empty_python_project() {
    let dir = create_test_dir();
    let root = dir.path();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");
    let stats = builder.map_python_tests(root, &mut graph, None).expect("Mapping failed");

    assert_eq!(stats.tests_processed, 0);
    assert_eq!(stats.edges_created, 0);
}
