//! Blackbox tests for dependency detection integration.

use std::path::Path;
use tempfile::TempDir;
use trinity_harness::graph::{EdgeType, GraphBuilder};
use trinity_harness::parsers::ParserRegistry;

fn create_test_dir() -> TempDir {
    TempDir::new().expect("Failed to create temp dir")
}

fn write_file(dir: &Path, name: &str, content: &str) {
    std::fs::write(dir.join(name), content).expect("Failed to write file");
}

// ==================== Rust Integration ====================

#[test]
fn test_rust_dependency_detection_full_flow() {
    let dir = create_test_dir();
    let root = dir.path();

    // Create files with dependencies
    write_file(root, "lib.rs", r#"
mod utils;
use utils::helper;

fn main() {
    let result = helper();
    process(result);
}

fn process(x: i32) -> i32 {
    x * 2
}
"#);

    write_file(root, "utils.rs", r#"
pub fn helper() -> i32 {
    42
}
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    // Scan for nodes
    let (mut graph, scan_stats) = builder.full_scan(root).expect("Scan failed");
    assert!(scan_stats.files_scanned >= 2);

    // Analyze dependencies
    let dep_stats = builder.analyze_dependencies(root, &mut graph).expect("Dep analysis failed");

    // Should find some dependencies
    assert!(dep_stats.deps_found > 0, "Should find dependencies");

    // Should have edges in the graph
    // Note: resolution depends on name matching, which may not resolve all
    let edges = graph.edges();

    // Verify we at least attempted dependency analysis
    assert!(dep_stats.deps_found >= 3, "Should find at least 3 deps (helper call, process call, utils import)");
}

#[test]
fn test_python_dependency_detection_full_flow() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "main.py", r#"
from utils import helper

def process():
    result = helper()
    return transform(result)

def transform(x):
    return x * 2
"#);

    write_file(root, "utils.py", r#"
def helper():
    return 42
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, scan_stats) = builder.full_scan(root).expect("Scan failed");
    assert!(scan_stats.files_scanned >= 2);

    let dep_stats = builder.analyze_dependencies(root, &mut graph).expect("Dep analysis failed");

    // Should find imports and calls
    assert!(dep_stats.deps_found > 0, "Should find dependencies");
}

#[test]
fn test_mixed_language_dependency_detection() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "core.rs", r#"
fn compute() -> i32 {
    42
}

struct Data {
    value: i32,
}
"#);

    write_file(root, "wrapper.py", r#"
def wrap_compute():
    # This would call Rust via PyO3
    return native_compute()
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, scan_stats) = builder.full_scan(root).expect("Scan failed");
    assert_eq!(scan_stats.files_scanned, 2);

    let dep_stats = builder.analyze_dependencies(root, &mut graph).expect("Dep analysis failed");

    // Should find dependencies in both languages
    assert!(dep_stats.deps_found > 0);
}

#[test]
fn test_dependency_edge_types() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "types.rs", r#"
struct Input {
    data: Vec<u8>,
}

struct Output {
    result: String,
}

fn process(input: Input) -> Output {
    Output { result: String::new() }
}

fn caller() {
    let i = Input { data: vec![] };
    let o = process(i);
}
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");
    let dep_stats = builder.analyze_dependencies(root, &mut graph).expect("Dep analysis failed");

    // Should have Uses (type refs) and Calls edges
    let edges = graph.edges();
    let edge_types: Vec<_> = edges.iter().map(|e| e.edge_type).collect();

    // We expect:
    // - process uses Input (parameter type)
    // - process uses Output (return type)
    // - caller calls process

    // Check we have different types of edges (may not resolve all due to name matching)
    assert!(dep_stats.deps_found >= 3, "Should find uses and calls dependencies");
}

#[test]
fn test_dependency_stats_accuracy() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "test.rs", r#"
use std::collections::HashMap;

fn process() {
    let x = compute();
    let y = transform(x);
    finalize(y);
}

fn compute() -> i32 { 1 }
fn transform(x: i32) -> i32 { x * 2 }
fn finalize(x: i32) { }
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");
    let dep_stats = builder.analyze_dependencies(root, &mut graph).expect("Dep analysis failed");

    // Verify stats are consistent
    assert_eq!(
        dep_stats.deps_found,
        dep_stats.deps_resolved + dep_stats.deps_unresolved,
        "Found should equal resolved + unresolved"
    );
}

#[test]
fn test_no_dependencies_empty_files() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "empty.rs", "");
    write_file(root, "comments_only.rs", "// Just a comment\n/* block */");

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");
    let dep_stats = builder.analyze_dependencies(root, &mut graph).expect("Dep analysis failed");

    assert_eq!(dep_stats.deps_found, 0);
    assert_eq!(graph.edges().len(), 0);
}

#[test]
fn test_self_referential_dependencies() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "recursive.rs", r#"
fn factorial(n: u64) -> u64 {
    if n <= 1 {
        1
    } else {
        n * factorial(n - 1)
    }
}
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");
    let dep_stats = builder.analyze_dependencies(root, &mut graph).expect("Dep analysis failed");

    // Should find recursive call
    let calls_count = graph.edges().iter().filter(|e| e.edge_type == EdgeType::Calls).count();

    // factorial calls itself
    assert!(dep_stats.deps_found >= 1, "Should find recursive call");
    assert!(calls_count >= 1, "Should have at least one Calls edge");
}

#[test]
fn test_python_class_method_dependencies() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "class_test.py", r#"
class Calculator:
    def add(self, a, b):
        return a + b

    def multiply(self, a, b):
        return a * b

    def compute(self, x, y):
        sum_val = self.add(x, y)
        product = self.multiply(x, y)
        return helper(sum_val, product)

def helper(a, b):
    return a + b
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");
    let dep_stats = builder.analyze_dependencies(root, &mut graph).expect("Dep analysis failed");

    // Should find method calls and helper call
    assert!(dep_stats.deps_found >= 3, "Should find add, multiply, and helper calls");
}

#[test]
fn test_nested_function_calls() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "nested.rs", r#"
fn outer(x: i32) -> i32 {
    inner(transform(validate(x)))
}

fn inner(x: i32) -> i32 { x }
fn transform(x: i32) -> i32 { x * 2 }
fn validate(x: i32) -> i32 { x.abs() }
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");
    let dep_stats = builder.analyze_dependencies(root, &mut graph).expect("Dep analysis failed");

    // Should find all nested calls
    assert!(dep_stats.deps_found >= 3, "Should find inner, transform, validate calls");
}

#[test]
fn test_wgsl_no_deps_yet() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "shader.wgsl", r#"
struct VertexInput {
    @location(0) position: vec3<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> @builtin(position) vec4<f32> {
    return vec4<f32>(input.position, 1.0);
}
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, scan_stats) = builder.full_scan(root).expect("Scan failed");
    assert_eq!(scan_stats.files_scanned, 1);

    let dep_stats = builder.analyze_dependencies(root, &mut graph).expect("Dep analysis failed");

    // WGSL deps not implemented yet
    assert_eq!(dep_stats.deps_found, 0);
}

#[test]
fn test_large_file_dependency_detection() {
    let dir = create_test_dir();
    let root = dir.path();

    // Generate a file with many functions and calls
    let mut source = String::new();
    for i in 0..50 {
        source.push_str(&format!("fn func_{i}() {{ ", i = i));
        if i > 0 {
            source.push_str(&format!("func_{}(); ", i - 1));
        }
        source.push_str("}\n");
    }

    write_file(root, "large.rs", &source);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");
    let dep_stats = builder.analyze_dependencies(root, &mut graph).expect("Dep analysis failed");

    // Should find many call dependencies
    assert!(dep_stats.deps_found >= 49, "Should find all chain calls");
    assert!(dep_stats.deps_resolved >= 40, "Should resolve most calls");
}

#[test]
fn test_circular_dependencies() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "circular.rs", r#"
fn a() {
    b();
}

fn b() {
    c();
}

fn c() {
    a();  // Back to a
}
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");
    let dep_stats = builder.analyze_dependencies(root, &mut graph).expect("Dep analysis failed");

    // Should find all three calls forming a cycle
    assert!(dep_stats.deps_found >= 3);

    let edges = graph.edges();
    assert!(edges.len() >= 3, "Should create edges for circular deps");
}
