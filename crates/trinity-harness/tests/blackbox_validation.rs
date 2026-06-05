//! Blackbox tests for graph validation integration.

use std::path::Path;
use tempfile::TempDir;
use trinity_harness::graph::{EdgeType, GraphBuilder};
use trinity_harness::parsers::{Language, ParserRegistry};

fn create_test_dir() -> TempDir {
    TempDir::new().expect("Failed to create temp dir")
}

fn write_file(dir: &Path, name: &str, content: &str) {
    std::fs::write(dir.join(name), content).expect("Failed to write file");
}

#[test]
fn test_validate_rust_project() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "main.rs", r#"
fn main() {
    let result = process();
    println!("{}", result);
}

fn process() -> i32 {
    let data = compute();
    transform(data)
}

fn compute() -> i32 {
    42
}

fn transform(x: i32) -> i32 {
    x * 2
}
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");
    builder.analyze_dependencies(root, &mut graph).expect("Dep analysis failed");

    let result = graph.validate();

    assert!(result.total_nodes >= 4, "Should have at least 4 functions");
    assert_eq!(result.nodes_by_language.get(&Language::Rust), Some(&result.total_nodes));
}

#[test]
fn test_validate_python_project() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "main.py", r#"
def main():
    result = process()
    print(result)

def process():
    data = compute()
    return transform(data)

def compute():
    return 42

def transform(x):
    return x * 2

if __name__ == "__main__":
    main()
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");
    builder.analyze_dependencies(root, &mut graph).expect("Dep analysis failed");

    let result = graph.validate();

    assert!(result.total_nodes >= 4, "Should have at least 4 functions");
    assert_eq!(result.nodes_by_language.get(&Language::Python), Some(&result.total_nodes));
}

#[test]
fn test_validate_mixed_language_project() {
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

    write_file(root, "app.py", r#"
def main():
    result = process()
    return result

def process():
    return 42

class Handler:
    def handle(self):
        pass
"#);

    write_file(root, "shader.wgsl", r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vs_main() -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    return out;
}
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, _) = builder.full_scan(root).expect("Scan failed");
    let result = graph.validate();

    // Should have nodes from all three languages
    assert!(result.nodes_by_language.get(&Language::Rust).is_some());
    assert!(result.nodes_by_language.get(&Language::Python).is_some());
    assert!(result.nodes_by_language.get(&Language::Wgsl).is_some());
}

#[test]
fn test_validate_with_dependencies() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "lib.rs", r#"
fn caller() {
    callee();
}

fn callee() {
    helper();
}

fn helper() {
    // leaf
}
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");
    builder.analyze_dependencies(root, &mut graph).expect("Dep analysis failed");

    let result = graph.validate();

    // Should have call edges
    let call_count = result.edges_by_type.get(&EdgeType::Calls).copied().unwrap_or(0);
    assert!(call_count >= 2, "Should have at least 2 call edges");
}

#[test]
fn test_validate_with_struct_mirrors() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "types.rs", r#"
#[repr(C)]
struct Vertex {
    position: [f32; 3],
}

#[repr(C)]
struct Uniform {
    matrix: [[f32; 4]; 4],
}
"#);

    write_file(root, "shader.wgsl", r#"
struct Vertex {
    @location(0) position: vec3<f32>,
}

struct Uniform {
    matrix: mat4x4<f32>,
}

@vertex
fn vs_main() -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0);
}
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");
    builder.analyze_crosslang(root, &mut graph).expect("Crosslang analysis failed");

    let result = graph.validate();

    // Should have MirrorsLayout edges
    let mirror_count = result.edges_by_type.get(&EdgeType::MirrorsLayout).copied().unwrap_or(0);
    assert_eq!(mirror_count, 2, "Should have 2 mirror edges");
}

#[test]
fn test_validate_entry_points_not_flagged() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "main.rs", r#"
fn main() {
    // Entry point - should not be flagged as orphan issue
}
"#);

    write_file(root, "shader.wgsl", r#"
@vertex
fn vs_main() -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, _) = builder.full_scan(root).expect("Scan failed");
    let result = graph.validate();

    // Entry points are OK as orphans
    assert_eq!(result.orphan_issues, 0, "Entry points should not be flagged as issues");
    assert!(result.is_valid);
}

#[test]
fn test_validate_large_project() {
    let dir = create_test_dir();
    let root = dir.path();

    // Generate a project with many files
    for i in 0..10 {
        let content = format!(r#"
fn func_{i}() {{
    func_{next}();
}}
"#, i = i, next = (i + 1) % 10);
        write_file(root, &format!("module_{}.rs", i), &content);
    }

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, scan_stats) = builder.full_scan(root).expect("Scan failed");
    assert_eq!(scan_stats.files_scanned, 10);

    builder.analyze_dependencies(root, &mut graph).expect("Dep analysis failed");

    let result = graph.validate();

    assert_eq!(result.total_nodes, 10, "Should have 10 functions");
    assert!(result.total_edges > 0, "Should have call edges");
}

#[test]
fn test_validate_empty_project() {
    let dir = create_test_dir();
    let root = dir.path();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, _) = builder.full_scan(root).expect("Scan failed");
    let result = graph.validate();

    assert_eq!(result.total_nodes, 0);
    assert_eq!(result.total_edges, 0);
    assert!(result.is_valid);
}

#[test]
fn test_validate_full_pipeline() {
    let dir = create_test_dir();
    let root = dir.path();

    // Create a complete project with all features
    write_file(root, "main.rs", r#"
use pyo3::prelude::*;

fn main() {
    let data = create_data();
    process(data);
}

#[repr(C)]
struct GpuData {
    values: [f32; 4],
}

fn create_data() -> GpuData {
    GpuData { values: [1.0, 2.0, 3.0, 4.0] }
}

fn process(data: GpuData) {
    // Process
}

#[pyfunction]
fn python_entry(x: i32) -> i32 {
    x * 2
}
"#);

    write_file(root, "shader.wgsl", r#"
struct GpuData {
    values: vec4<f32>,
}

@vertex
fn vs_main() -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0);
}

fn helper() -> f32 {
    return 1.0;
}
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    // Full scan
    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");

    // Dependency analysis
    builder.analyze_dependencies(root, &mut graph).expect("Dep analysis failed");

    // Cross-language analysis
    builder.analyze_crosslang(root, &mut graph).expect("Crosslang analysis failed");

    // Validate
    let result = graph.validate();

    // Check comprehensive results
    assert!(result.total_nodes >= 6, "Should have multiple nodes");
    assert!(result.total_edges >= 1, "Should have some edges");
    assert!(result.nodes_by_language.get(&Language::Rust).is_some());
    assert!(result.nodes_by_language.get(&Language::Wgsl).is_some());

    // MirrorsLayout edge for GpuData
    let mirror_count = result.edges_by_type.get(&EdgeType::MirrorsLayout).copied().unwrap_or(0);
    assert_eq!(mirror_count, 1, "Should have 1 GpuData mirror");
}

#[test]
fn test_node_count_by_language_accuracy() {
    let dir = create_test_dir();
    let root = dir.path();

    // Create files with known counts
    write_file(root, "rust.rs", r#"
fn a() {}
fn b() {}
struct C {}
"#);

    write_file(root, "python.py", r#"
def d():
    pass

class E:
    pass
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, _) = builder.full_scan(root).expect("Scan failed");
    let counts = graph.node_count_by_language();

    assert_eq!(counts.get(&Language::Rust), Some(&3));
    assert_eq!(counts.get(&Language::Python), Some(&2));
}

#[test]
fn test_edge_count_by_type_accuracy() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "lib.rs", r#"
fn caller() {
    callee_a();
    callee_b();
}

fn callee_a() {}
fn callee_b() {}
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");
    builder.analyze_dependencies(root, &mut graph).expect("Dep analysis failed");

    let counts = graph.edge_count_by_type();
    let call_count = counts.get(&EdgeType::Calls).copied().unwrap_or(0);

    // caller calls callee_a and callee_b
    assert!(call_count >= 2, "Should have at least 2 call edges");
}
