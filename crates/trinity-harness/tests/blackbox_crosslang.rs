//! Blackbox tests for cross-language edge detection integration.

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

// ==================== PyO3 Integration ====================

#[test]
fn test_pyo3_bindings_full_flow() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "lib.rs", r#"
use pyo3::prelude::*;

#[pyfunction]
fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[pyclass]
struct Calculator {
    value: i32,
}

#[pymethods]
impl Calculator {
    #[new]
    fn new(value: i32) -> Self {
        Calculator { value }
    }

    fn add(&self, x: i32) -> i32 {
        self.value + x
    }
}
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, scan_stats) = builder.full_scan(root).expect("Scan failed");
    assert!(scan_stats.files_scanned >= 1);

    let stats = builder.analyze_crosslang(root, &mut graph).expect("Crosslang analysis failed");

    // Should find pyfunction + pyclass + pymethods
    assert!(stats.bindings_found >= 3, "Should find PyO3 bindings");
    assert!(stats.pyo3_functions >= 1, "Should find pyfunction");
    assert!(stats.pyo3_classes >= 1, "Should find pyclass");
}

#[test]
fn test_pyo3_no_bindings() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "lib.rs", r#"
fn regular_function() -> i32 {
    42
}

struct RegularStruct {
    value: i32,
}
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");
    let stats = builder.analyze_crosslang(root, &mut graph).expect("Crosslang analysis failed");

    assert_eq!(stats.bindings_found, 0);
    assert_eq!(stats.pyo3_functions, 0);
    assert_eq!(stats.pyo3_classes, 0);
}

// ==================== Struct Mirror Integration ====================

#[test]
fn test_struct_mirror_detection() {
    let dir = create_test_dir();
    let root = dir.path();

    // Create matching WGSL and Rust structs
    write_file(root, "shader.wgsl", r#"
struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

struct Uniform {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
}
"#);

    write_file(root, "types.rs", r#"
#[repr(C)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
}

#[repr(C)]
struct Uniform {
    view: [[f32; 4]; 4],
    projection: [[f32; 4]; 4],
}
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, scan_stats) = builder.full_scan(root).expect("Scan failed");
    assert_eq!(scan_stats.files_scanned, 2);

    let stats = builder.analyze_crosslang(root, &mut graph).expect("Crosslang analysis failed");

    // Should find 2 struct mirrors
    assert_eq!(stats.struct_mirrors, 2);
    assert_eq!(stats.edges_created, 2);

    // Verify edges are MirrorsLayout type
    let mirror_edges: Vec<_> = graph.edges()
        .iter()
        .filter(|e| e.edge_type == EdgeType::MirrorsLayout)
        .collect();
    assert_eq!(mirror_edges.len(), 2);
}

#[test]
fn test_struct_mirror_no_match() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "shader.wgsl", r#"
struct WgslOnly {
    value: f32,
}
"#);

    write_file(root, "types.rs", r#"
#[repr(C)]
struct RustOnly {
    value: f32,
}
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");
    let stats = builder.analyze_crosslang(root, &mut graph).expect("Crosslang analysis failed");

    // No matching names
    assert_eq!(stats.struct_mirrors, 0);
    assert_eq!(stats.edges_created, 0);
}

#[test]
fn test_struct_mirror_partial_match() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "shader.wgsl", r#"
struct Vertex {
    position: vec3<f32>,
}

struct WgslOnly {
    value: f32,
}
"#);

    write_file(root, "types.rs", r#"
#[repr(C)]
struct Vertex {
    position: [f32; 3],
}

#[repr(C)]
struct RustOnly {
    value: f32,
}
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");
    let stats = builder.analyze_crosslang(root, &mut graph).expect("Crosslang analysis failed");

    // Only Vertex matches
    assert_eq!(stats.struct_mirrors, 1);
    assert_eq!(stats.edges_created, 1);
}

// ==================== Combined Analysis ====================

#[test]
fn test_combined_pyo3_and_mirrors() {
    let dir = create_test_dir();
    let root = dir.path();

    // PyO3 bindings
    write_file(root, "bindings.rs", r#"
use pyo3::prelude::*;

#[pyfunction]
fn process_data(data: Vec<f32>) -> f32 {
    data.iter().sum()
}

#[pyclass]
struct DataProcessor {
    factor: f32,
}
"#);

    // GPU structs (repr(C))
    write_file(root, "gpu_types.rs", r#"
#[repr(C)]
struct GpuVertex {
    position: [f32; 3],
}
"#);

    // WGSL shader
    write_file(root, "shader.wgsl", r#"
struct GpuVertex {
    @location(0) position: vec3<f32>,
}
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, scan_stats) = builder.full_scan(root).expect("Scan failed");
    assert_eq!(scan_stats.files_scanned, 3);

    let stats = builder.analyze_crosslang(root, &mut graph).expect("Crosslang analysis failed");

    // Should find both PyO3 bindings and struct mirrors
    assert!(stats.pyo3_functions >= 1, "Should find pyfunction");
    assert!(stats.pyo3_classes >= 1, "Should find pyclass");
    assert_eq!(stats.struct_mirrors, 1, "Should find GpuVertex mirror");
}

#[test]
fn test_large_project_crosslang() {
    let dir = create_test_dir();
    let root = dir.path();

    // Create multiple files with bindings
    for i in 0..5 {
        let content = format!(r#"
use pyo3::prelude::*;

#[pyfunction]
fn func_{i}() -> i32 {{ {i} }}

#[pyclass]
struct Class_{i} {{
    value: i32,
}}
"#, i = i);
        write_file(root, &format!("mod_{}.rs", i), &content);
    }

    // Create WGSL/Rust mirror pairs
    for i in 0..3 {
        write_file(root, &format!("shader_{}.wgsl", i), &format!(r#"
struct Mirror_{i} {{
    value: f32,
}}
"#, i = i));

        write_file(root, &format!("types_{}.rs", i), &format!(r#"
#[repr(C)]
struct Mirror_{i} {{
    value: f32,
}}
"#, i = i));
    }

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, scan_stats) = builder.full_scan(root).expect("Scan failed");
    assert!(scan_stats.files_scanned >= 10);

    let stats = builder.analyze_crosslang(root, &mut graph).expect("Crosslang analysis failed");

    // Should find all bindings and mirrors
    assert!(stats.pyo3_functions >= 5);
    assert!(stats.pyo3_classes >= 5);
    assert_eq!(stats.struct_mirrors, 3);
}

#[test]
fn test_empty_directory_crosslang() {
    let dir = create_test_dir();
    let root = dir.path();

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");
    let stats = builder.analyze_crosslang(root, &mut graph).expect("Crosslang analysis failed");

    assert_eq!(stats.bindings_found, 0);
    assert_eq!(stats.struct_mirrors, 0);
    assert_eq!(stats.edges_created, 0);
}

#[test]
fn test_python_files_ignored_for_pyo3() {
    let dir = create_test_dir();
    let root = dir.path();

    // Python files shouldn't contribute to PyO3 bindings
    write_file(root, "main.py", r#"
def pyfunction():
    pass

class PyClass:
    pass
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");
    let stats = builder.analyze_crosslang(root, &mut graph).expect("Crosslang analysis failed");

    // Python code shouldn't register as PyO3 bindings
    assert_eq!(stats.pyo3_functions, 0);
    assert_eq!(stats.pyo3_classes, 0);
}

#[test]
fn test_wgsl_only_no_mirrors() {
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

    let stats = builder.analyze_crosslang(root, &mut graph).expect("Crosslang analysis failed");

    // No Rust files, so no mirrors possible
    assert_eq!(stats.struct_mirrors, 0);
}

#[test]
fn test_full_analysis_pipeline() {
    let dir = create_test_dir();
    let root = dir.path();

    write_file(root, "lib.rs", r#"
use pyo3::prelude::*;

#[repr(C)]
struct GpuData {
    values: [f32; 4],
}

#[pyfunction]
fn process(data: GpuData) -> f32 {
    data.values.iter().sum()
}

fn helper() {
    process(GpuData { values: [1.0, 2.0, 3.0, 4.0] });
}
"#);

    write_file(root, "shader.wgsl", r#"
struct GpuData {
    values: vec4<f32>,
}

@fragment
fn fs_main(data: GpuData) -> @location(0) vec4<f32> {
    return vec4<f32>(data.values);
}
"#);

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    // Full scan
    let (mut graph, _) = builder.full_scan(root).expect("Scan failed");

    // Dependency analysis
    let dep_stats = builder.analyze_dependencies(root, &mut graph).expect("Dep analysis failed");
    assert!(dep_stats.deps_found > 0);

    // Cross-language analysis
    let crosslang_stats = builder.analyze_crosslang(root, &mut graph).expect("Crosslang analysis failed");
    assert!(crosslang_stats.pyo3_functions >= 1);
    assert_eq!(crosslang_stats.struct_mirrors, 1);

    // Verify graph has both regular and cross-lang edges
    let edges = graph.edges();
    let has_calls = edges.iter().any(|e| e.edge_type == EdgeType::Calls);
    let has_mirrors = edges.iter().any(|e| e.edge_type == EdgeType::MirrorsLayout);

    assert!(has_calls || dep_stats.deps_resolved > 0, "Should have call edges or resolved deps");
    assert!(has_mirrors, "Should have mirror edge");
}
