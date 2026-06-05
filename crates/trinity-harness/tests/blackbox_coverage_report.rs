//! Blackbox tests for coverage report generation via GraphBuilder.

use std::path::Path;
use tempfile::TempDir;
use trinity_harness::graph::{generate_coverage_report, ConventionMapper, GraphBuilder};
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
fn test_coverage_report_basic() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/lib.rs", r#"
fn compute() -> i32 { 42 }
fn helper() -> i32 { 1 }
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
    let report = generate_coverage_report(&graph, &mappings);

    assert!(report.total_code_nodes >= 2);
    assert!(report.covered_nodes >= 1);
    assert!(report.coverage_percent > 0.0);
}

#[test]
fn test_coverage_report_all_covered() {
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
    let report = generate_coverage_report(&graph, &mappings);

    // At minimum the function should be covered
    assert!(report.covered_nodes >= 1);
}

#[test]
fn test_coverage_report_zero_coverage() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/lib.rs", r#"
fn func_a() -> i32 { 1 }
fn func_b() -> i32 { 2 }
"#);

    // No tests - empty mappings
    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, _) = builder.full_scan(root).expect("Scan failed");

    let mappings = vec![];
    let report = generate_coverage_report(&graph, &mappings);

    assert!(report.total_code_nodes >= 2);
    assert_eq!(report.covered_nodes, 0);
    assert_eq!(report.coverage_percent, 0.0);
}

#[test]
fn test_coverage_report_by_file() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/a.rs", r#"
fn func_a() -> i32 { 1 }
"#);

    write_file(root, "src/b.rs", r#"
fn func_b() -> i32 { 2 }
"#);

    write_file(root, "tests/test_a.rs", r#"
fn test_func_a() { assert_eq!(func_a(), 1); }
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, _) = builder.full_scan(root).expect("Scan failed");

    let mapper = ConventionMapper::new();
    let (mappings, _) = mapper.map_tests(&graph);
    let report = generate_coverage_report(&graph, &mappings);

    // Should have coverage breakdown by file
    assert!(!report.by_file.is_empty());
}

#[test]
fn test_coverage_report_summary() {
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
    let report = generate_coverage_report(&graph, &mappings);
    let summary = report.generate_summary();

    assert!(summary.contains("Test Coverage Report"));
    assert!(summary.contains("%"));
}

#[test]
fn test_coverage_empty_project() {
    let dir = create_test_dir();
    let root = dir.path();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, _) = builder.full_scan(root).expect("Scan failed");

    let mappings = vec![];
    let report = generate_coverage_report(&graph, &mappings);

    assert_eq!(report.total_code_nodes, 0);
    assert!(!report.has_uncovered());
}
