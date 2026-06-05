//! Blackbox tests for inline test mapping via GraphBuilder.

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
fn test_map_inline_tests_basic() {
    let dir = create_test_dir();
    let root = dir.path();

    // File with both production code and inline test
    write_file(root, "src/lib.rs", r#"
fn compute() -> i32 {
    42
}

#[test]
fn test_compute() {
    assert_eq!(compute(), 42);
}
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");
    let stats = builder.map_inline_tests(&mut graph).expect("Mapping failed");

    assert!(stats.tests_processed >= 1);
    assert!(stats.tests_mapped >= 1);
    assert!(stats.edges_created >= 1);

    let test_edges: Vec<_> = graph.edges()
        .iter()
        .filter(|e| e.edge_type == EdgeType::Tests)
        .collect();

    assert!(!test_edges.is_empty());
}

#[test]
fn test_map_inline_tests_multiple_functions() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/lib.rs", r#"
fn func_a() -> i32 { 1 }
fn func_b() -> i32 { 2 }
struct MyStruct { value: i32 }

#[test]
fn test_all() {
    assert_eq!(func_a(), 1);
    assert_eq!(func_b(), 2);
}
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");
    let stats = builder.map_inline_tests(&mut graph).expect("Mapping failed");

    // Should map test to all 3 items (func_a, func_b, MyStruct)
    assert!(stats.edges_created >= 3);
}

#[test]
fn test_map_inline_tests_multiple_tests() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/lib.rs", r#"
fn compute() -> i32 { 42 }

#[test]
fn test_compute_returns_42() {
    assert_eq!(compute(), 42);
}

#[test]
fn test_compute_not_zero() {
    assert_ne!(compute(), 0);
}
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");
    let stats = builder.map_inline_tests(&mut graph).expect("Mapping failed");

    assert!(stats.tests_processed >= 2);
    assert!(stats.tests_mapped >= 2);
}

#[test]
fn test_map_inline_tests_multiple_files() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/a.rs", r#"
fn func_a() -> i32 { 1 }

#[test]
fn test_func_a() {
    assert_eq!(func_a(), 1);
}
"#);

    write_file(root, "src/b.rs", r#"
fn func_b() -> i32 { 2 }

#[test]
fn test_func_b() {
    assert_eq!(func_b(), 2);
}
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, scan_stats) = builder.full_scan(root).expect("Scan failed");
    assert_eq!(scan_stats.files_scanned, 2);

    let stats = builder.map_inline_tests(&mut graph).expect("Mapping failed");

    assert!(stats.tests_processed >= 2);
    assert!(stats.edges_created >= 2);
}

#[test]
fn test_inline_tests_separate_from_blackbox() {
    let dir = create_test_dir();
    let root = dir.path();

    // Inline test
    write_file(root, "src/lib.rs", r#"
fn compute() -> i32 { 42 }

#[test]
fn test_compute() {
    assert_eq!(compute(), 42);
}
"#);

    // Blackbox test (should be handled by map_rust_tests, not map_inline_tests)
    write_file(root, "tests/integration.rs", r#"
fn test_integration() {
    // Integration test
}
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");

    // map_inline_tests should only find the inline test
    let inline_stats = builder.map_inline_tests(&mut graph).expect("Mapping failed");
    assert!(inline_stats.tests_processed >= 1);

    // The blackbox test should be mapped by map_rust_tests
    let rust_stats = builder.map_rust_tests(root, &mut graph, None).expect("Mapping failed");
    assert!(rust_stats.tests_processed >= 1);
}

#[test]
fn test_full_pipeline_with_inline_tests() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/lib.rs", r#"
fn compute() -> i32 {
    helper()
}

fn helper() -> i32 {
    42
}

#[test]
fn test_compute() {
    assert_eq!(compute(), 42);
}

#[test]
fn test_helper() {
    assert_eq!(helper(), 42);
}
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    // Full pipeline
    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");
    builder.analyze_dependencies(root, &mut graph).expect("Dep analysis failed");
    let inline_stats = builder.map_inline_tests(&mut graph).expect("Inline mapping failed");

    // Validate
    let result = graph.validate();

    assert!(result.total_nodes >= 4);
    assert!(inline_stats.edges_created >= 2);

    // Should have Tests edges
    let test_edge_count = result.edges_by_type.get(&EdgeType::Tests).copied().unwrap_or(0);
    assert!(test_edge_count >= 2);
}

#[test]
fn test_empty_project() {
    let dir = create_test_dir();
    let root = dir.path();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");
    let stats = builder.map_inline_tests(&mut graph).expect("Mapping failed");

    assert_eq!(stats.tests_processed, 0);
    assert_eq!(stats.edges_created, 0);
}
