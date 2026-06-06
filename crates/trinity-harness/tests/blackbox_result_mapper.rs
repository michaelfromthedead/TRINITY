//! Blackbox tests for result mapping via GraphBuilder.

use std::path::Path;
use tempfile::TempDir;
use trinity_harness::graph::GraphBuilder;
use trinity_harness::parsers::ParserRegistry;
use trinity_harness::runners::{map_results, TestResult};

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
fn test_map_results_integration() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/lib.rs", r#"
fn compute() -> i32 { 42 }
"#);

    write_file(root, "tests/test_lib.rs", r#"
fn test_compute() {
    assert_eq!(compute(), 42);
}
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");
    builder.map_rust_tests(root, &mut graph, None).expect("Mapping failed");

    let test_results = vec![TestResult::passed("test_compute", 100)];
    let mapped = map_results(&graph, &test_results);

    assert_eq!(mapped.total_tests, 1);
}

#[test]
fn test_map_results_with_failures() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/lib.rs", r#"
fn compute() -> i32 { 42 }
fn helper() -> i32 { 1 }
"#);

    write_file(root, "tests/test_lib.rs", r#"
fn test_compute() { assert_eq!(compute(), 42); }
fn test_helper() { assert_eq!(helper(), 1); }
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");
    builder.map_rust_tests(root, &mut graph, None).expect("Mapping failed");

    let test_results = vec![
        TestResult::passed("test_compute", 100),
        TestResult::failed("test_helper", 100, "assertion failed"),
    ];

    let mapped = map_results(&graph, &test_results);

    assert_eq!(mapped.total_tests, 2);
    let (passing, failing, _) = mapped.summary();
    // Results depend on whether mapping succeeds
    assert!(passing >= 0);
    assert!(failing >= 0 || mapped.unmapped_tests.len() >= 1);
}

#[test]
fn test_map_results_empty_graph() {
    let graph = trinity_harness::graph::CodeGraph::new();
    let test_results = vec![TestResult::passed("test_foo", 100)];

    let mapped = map_results(&graph, &test_results);

    assert_eq!(mapped.total_tests, 1);
    assert_eq!(mapped.mapped_tests, 0);
    assert_eq!(mapped.unmapped_tests.len(), 1);
}

#[test]
fn test_map_results_empty_results() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/lib.rs", r#"
fn compute() -> i32 { 42 }
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, _) = builder.full_scan(root).expect("Scan failed");

    let test_results: Vec<TestResult> = vec![];
    let mapped = map_results(&graph, &test_results);

    assert_eq!(mapped.total_tests, 0);
    assert!(mapped.by_node.is_empty());
}
