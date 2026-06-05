//! Blackbox tests for test mapping validation via GraphBuilder.

use std::path::Path;
use tempfile::TempDir;
use trinity_harness::graph::{validate_mappings, ConventionMapper, GraphBuilder};
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
fn test_validate_mappings_integration() {
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

    let (graph, _) = builder.full_scan(root).expect("Scan failed");

    let mapper = ConventionMapper::new();
    let (mappings, _) = mapper.map_tests(&graph);
    let result = validate_mappings(&graph, &mappings);

    assert!(result.is_valid);
    assert!(result.total_tests >= 1);
}

#[test]
fn test_validate_with_orphan_tests() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/lib.rs", r#"
fn compute() -> i32 { 42 }
"#);

    write_file(root, "tests/test_orphan.rs", r#"
fn test_nonexistent() {
    // No matching code
}
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, _) = builder.full_scan(root).expect("Scan failed");

    let mapper = ConventionMapper::new();
    let (mappings, _) = mapper.map_tests(&graph);
    let result = validate_mappings(&graph, &mappings);

    // Should still be valid (orphans are warnings)
    assert!(result.is_valid);
    assert!(result.orphan_tests.len() >= 1 || mappings.iter().any(|m| m.targets.is_empty()));
}

#[test]
fn test_validate_report_generation() {
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

    let (graph, _) = builder.full_scan(root).expect("Scan failed");

    let mapper = ConventionMapper::new();
    let (mappings, _) = mapper.map_tests(&graph);
    let result = validate_mappings(&graph, &mappings);
    let report = result.generate_report();

    assert!(report.contains("Validation Report"));
    assert!(report.contains("Tests validated"));
}

#[test]
fn test_validate_no_circular_deps() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/lib.rs", r#"
fn compute() -> i32 { 42 }
"#);

    write_file(root, "tests/test_lib.rs", r#"
fn test_compute() { assert_eq!(compute(), 42); }
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, _) = builder.full_scan(root).expect("Scan failed");

    let mapper = ConventionMapper::new();
    let (mappings, _) = mapper.map_tests(&graph);
    let result = validate_mappings(&graph, &mappings);

    assert!(!result.has_circular_deps());
}

#[test]
fn test_validate_empty_project() {
    let dir = create_test_dir();
    let root = dir.path();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, _) = builder.full_scan(root).expect("Scan failed");

    let mappings = vec![];
    let result = validate_mappings(&graph, &mappings);

    assert!(result.is_valid);
    assert_eq!(result.total_tests, 0);
}
