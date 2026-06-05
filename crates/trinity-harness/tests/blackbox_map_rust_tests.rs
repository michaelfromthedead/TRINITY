//! Blackbox tests for Rust test mapping via GraphBuilder.

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
fn test_map_rust_tests_basic() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/lib.rs", r#"
fn compute() -> i32 {
    42
}
"#);

    write_file(root, "tests/test_lib.rs", r#"
fn test_compute() {
    assert_eq!(compute(), 42);
}
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");
    let stats = builder.map_rust_tests(root, &mut graph, None).expect("Mapping failed");

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
fn test_map_crates_tests_pattern() {
    let dir = create_test_dir();
    let root = dir.path();

    // Simulate crates/*/tests/*.rs
    write_file(root, "crates/harness/src/lib.rs", r#"
fn harness_fn() -> i32 {
    42
}
"#);

    write_file(root, "crates/harness/tests/test_harness.rs", r#"
fn test_harness_fn() {
    assert_eq!(harness_fn(), 42);
}
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");
    let stats = builder.map_rust_tests(root, &mut graph, None).expect("Mapping failed");

    assert!(stats.tests_processed >= 1);
}

#[test]
fn test_map_multiple_crates() {
    let dir = create_test_dir();
    let root = dir.path();

    // Multiple crates
    write_file(root, "crates/foo/src/lib.rs", r#"
fn foo_fn() -> i32 { 1 }
"#);

    write_file(root, "crates/bar/src/lib.rs", r#"
fn bar_fn() -> i32 { 2 }
"#);

    write_file(root, "crates/foo/tests/test_foo.rs", r#"
fn test_foo_fn() { assert_eq!(foo_fn(), 1); }
"#);

    write_file(root, "crates/bar/tests/test_bar.rs", r#"
fn test_bar_fn() { assert_eq!(bar_fn(), 2); }
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, scan_stats) = builder.full_scan(root).expect("Scan failed");
    assert_eq!(scan_stats.files_scanned, 4);

    let stats = builder.map_rust_tests(root, &mut graph, None).expect("Mapping failed");

    assert!(stats.tests_processed >= 2);
}

#[test]
fn test_map_with_explicit_config() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/special.rs", r#"
fn special_fn() -> i32 { 42 }
"#);

    write_file(root, "tests/integration.rs", r#"
fn test_special() { assert_eq!(special_fn(), 42); }
"#);

    write_file(root, "test_mappings.toml", r#"
[[mappings]]
test = "tests/integration.rs"
targets = ["src/special.rs"]
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");
    let config_path = root.join("test_mappings.toml");
    let stats = builder.map_rust_tests(root, &mut graph, Some(&config_path)).expect("Mapping failed");

    assert!(stats.tests_mapped >= 1);
}

#[test]
fn test_map_all_tests() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/lib.rs", r#"
fn rust_fn() -> i32 { 42 }
"#);

    write_file(root, "tests/test_lib.rs", r#"
fn test_rust_fn() { assert_eq!(rust_fn(), 42); }
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");
    let stats = builder.map_all_tests(root, &mut graph, None).expect("Mapping failed");

    assert!(stats.tests_processed >= 1);
}

#[test]
fn test_full_pipeline_with_rust_mapping() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/lib.rs", r#"
fn compute() -> i32 {
    helper()
}

fn helper() -> i32 {
    42
}
"#);

    write_file(root, "tests/integration.rs", r#"
fn test_compute() {
    assert_eq!(compute(), 42);
}

fn test_helper() {
    assert_eq!(helper(), 42);
}
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    // Full pipeline
    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");
    builder.analyze_dependencies(root, &mut graph).expect("Dep analysis failed");
    let map_stats = builder.map_rust_tests(root, &mut graph, None).expect("Mapping failed");

    // Validate
    let result = graph.validate();

    assert!(result.total_nodes >= 4);
    assert!(result.total_edges >= 1);
    assert!(map_stats.edges_created >= 2);

    // Should have Tests edges
    let test_edge_count = result.edges_by_type.get(&EdgeType::Tests).copied().unwrap_or(0);
    assert!(test_edge_count >= 2);
}

#[test]
fn test_missing_config_falls_back() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/lib.rs", r#"
fn func() -> i32 { 42 }
"#);

    write_file(root, "tests/test_lib.rs", r#"
fn test_func() { assert_eq!(func(), 42); }
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");

    // Point to non-existent config - should fall back to convention
    let config_path = root.join("nonexistent.toml");
    let stats = builder.map_rust_tests(root, &mut graph, Some(&config_path)).expect("Mapping failed");

    // Should still work via convention
    assert!(stats.tests_processed >= 1);
}

#[test]
fn test_empty_project() {
    let dir = create_test_dir();
    let root = dir.path();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");
    let stats = builder.map_rust_tests(root, &mut graph, None).expect("Mapping failed");

    assert_eq!(stats.tests_processed, 0);
    assert_eq!(stats.edges_created, 0);
}
