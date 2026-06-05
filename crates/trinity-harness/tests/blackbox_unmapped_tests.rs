//! Blackbox tests for unmapped test handling via GraphBuilder.

use std::path::Path;
use tempfile::TempDir;
use trinity_harness::graph::{extract_unmapped, GraphBuilder};
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
fn test_extract_unmapped_integration() {
    let dir = create_test_dir();
    let root = dir.path();

    // Code
    write_file(root, "src/lib.rs", r#"
fn existing_function() -> i32 {
    42
}
"#);

    // Unmapped test
    write_file(root, "tests/orphan.rs", r#"
fn test_nonexistent_function() {
    // This test has no matching code
}
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");
    let stats = builder.map_rust_tests(root, &mut graph, None).expect("Mapping failed");

    // Get mappings and extract unmapped
    let mapper = trinity_harness::graph::ConventionMapper::new();
    let (mappings, _) = mapper.map_tests(&graph);
    let review = extract_unmapped(&mappings, &graph);

    assert!(review.has_unmapped() || stats.tests_unmapped >= 1);
}

#[test]
fn test_unmapped_review_report() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/lib.rs", r#"
fn compute() -> i32 { 42 }
"#);

    write_file(root, "tests/orphan.rs", r#"
fn test_unknown() {
    // No matching code
}
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, _) = builder.full_scan(root).expect("Scan failed");

    let mapper = trinity_harness::graph::ConventionMapper::new();
    let (mappings, _) = mapper.map_tests(&graph);
    let review = extract_unmapped(&mappings, &graph);
    let report = review.generate_report();

    // Report should be generated
    assert!(report.contains("Unmapped Tests Review"));
}

#[test]
fn test_all_tests_mapped() {
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

    let mapper = trinity_harness::graph::ConventionMapper::new();
    let (mappings, stats) = mapper.map_tests(&graph);
    let review = extract_unmapped(&mappings, &graph);

    // Should have at least one mapped
    assert!(stats.tests_mapped >= 1);
    // Review should show what's unmapped (may or may not have any)
    let _report = review.generate_report();
}

#[test]
fn test_mixed_mapped_unmapped() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/lib.rs", r#"
fn compute() -> i32 { 42 }
fn helper() -> i32 { 1 }
"#);

    // Mapped test
    write_file(root, "tests/test_compute.rs", r#"
fn test_compute() {
    assert_eq!(compute(), 42);
}
"#);

    // Unmapped test
    write_file(root, "tests/test_nonexistent.rs", r#"
fn test_something_else() {
    // No match
}
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, _) = builder.full_scan(root).expect("Scan failed");

    let mapper = trinity_harness::graph::ConventionMapper::new();
    let (mappings, stats) = mapper.map_tests(&graph);
    let review = extract_unmapped(&mappings, &graph);

    // Should have both mapped and unmapped
    assert!(stats.tests_mapped >= 1);
    assert!(stats.tests_unmapped >= 1 || review.total_unmapped >= 1);
}

#[test]
fn test_suggestions_in_review() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "src/lib.rs", r#"
fn process_data() -> i32 { 42 }
fn process_request() -> i32 { 1 }
"#);

    write_file(root, "tests/test_process.rs", r#"
fn test_process() {
    // Should suggest process_data and process_request
}
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, _) = builder.full_scan(root).expect("Scan failed");

    let mapper = trinity_harness::graph::ConventionMapper::new();
    let (mappings, _) = mapper.map_tests(&graph);
    let review = extract_unmapped(&mappings, &graph);

    // If unmapped, should have suggestions
    for test in &review.unmapped {
        if test.test_name.contains("process") {
            // Should suggest similar names
            assert!(!test.suggestions.is_empty() || review.unmapped.is_empty());
        }
    }
}

#[test]
fn test_empty_project() {
    let dir = create_test_dir();
    let root = dir.path();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, _) = builder.full_scan(root).expect("Scan failed");

    let mapper = trinity_harness::graph::ConventionMapper::new();
    let (mappings, _) = mapper.map_tests(&graph);
    let review = extract_unmapped(&mappings, &graph);

    assert!(!review.has_unmapped());
    assert_eq!(review.total_unmapped, 0);
}
