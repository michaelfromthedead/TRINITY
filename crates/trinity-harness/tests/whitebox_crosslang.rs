//! Whitebox tests for cross-language edge detection.

use trinity_harness::graph::{
    create_crosslang_edges, detect_struct_mirrors, BindingType, CodeGraph, CodeNode,
    CrossLangBinding, EdgeType, NodeId, Pyo3Analyzer, ReprCAnalyzer,
};
use trinity_harness::parsers::{CodeUnit, ContentHashes, Language, UnitType};

fn empty_hashes() -> ContentHashes {
    ContentHashes {
        full_hash: [0u8; 32],
        signature_hash: [0u8; 32],
        body_hash: [0u8; 32],
        layout_hash: [0u8; 32],
    }
}

fn make_struct_node(
    graph: &mut CodeGraph,
    file: &str,
    name: &str,
    lang: Language,
) -> NodeId {
    let id = NodeId(graph.nodes().len());
    let unit = CodeUnit {
        unit_type: UnitType::Struct,
        name: name.to_string(),
        start_line: 1,
        end_line: 10,
        language: lang,
        hashes: empty_hashes(),
    };
    let node = CodeNode::new(id, file.to_string(), unit);
    graph.add_node(node)
}

// ==================== PyO3 Analyzer ====================

#[test]
fn test_pyo3_analyzer_detects_pyfunction() {
    let analyzer = Pyo3Analyzer::new();
    let source = r#"
use pyo3::prelude::*;

#[pyfunction]
fn my_function(x: i32) -> i32 {
    x * 2
}
"#;

    let bindings = analyzer.analyze(source, "lib.rs");

    assert_eq!(bindings.len(), 1);
    assert_eq!(bindings[0].name, "my_function");
    assert_eq!(bindings[0].binding_type, BindingType::PyFunction);
    assert_eq!(bindings[0].source_lang, Language::Rust);
    assert_eq!(bindings[0].target_lang, Language::Python);
}

#[test]
fn test_pyo3_analyzer_detects_pyclass() {
    let analyzer = Pyo3Analyzer::new();
    let source = r#"
use pyo3::prelude::*;

#[pyclass]
struct MyClass {
    value: i32,
}
"#;

    let bindings = analyzer.analyze(source, "lib.rs");

    assert_eq!(bindings.len(), 1);
    assert_eq!(bindings[0].name, "MyClass");
    assert_eq!(bindings[0].binding_type, BindingType::PyClass);
}

#[test]
fn test_pyo3_analyzer_detects_pymethods() {
    let analyzer = Pyo3Analyzer::new();
    let source = r#"
use pyo3::prelude::*;

#[pyclass]
struct Calculator {
    value: i32,
}

#[pymethods]
impl Calculator {
    fn add(&self, x: i32) -> i32 {
        self.value + x
    }

    fn multiply(&self, x: i32) -> i32 {
        self.value * x
    }
}
"#;

    let bindings = analyzer.analyze(source, "lib.rs");

    // Should find pyclass + 2 pymethods
    assert_eq!(bindings.len(), 3);

    let types: Vec<_> = bindings.iter().map(|b| b.binding_type).collect();
    assert!(types.contains(&BindingType::PyClass));
    assert_eq!(types.iter().filter(|t| **t == BindingType::PyMethod).count(), 2);

    let names: Vec<_> = bindings.iter().map(|b| b.name.as_str()).collect();
    assert!(names.contains(&"Calculator"));
    assert!(names.contains(&"Calculator::add"));
    assert!(names.contains(&"Calculator::multiply"));
}

#[test]
fn test_pyo3_analyzer_multiple_pyfunctions() {
    let analyzer = Pyo3Analyzer::new();
    let source = r#"
use pyo3::prelude::*;

#[pyfunction]
fn func_a() -> i32 { 1 }

#[pyfunction]
fn func_b() -> i32 { 2 }

#[pyfunction]
fn func_c() -> i32 { 3 }
"#;

    let bindings = analyzer.analyze(source, "lib.rs");

    assert_eq!(bindings.len(), 3);
    let names: Vec<_> = bindings.iter().map(|b| b.name.as_str()).collect();
    assert!(names.contains(&"func_a"));
    assert!(names.contains(&"func_b"));
    assert!(names.contains(&"func_c"));
}

#[test]
fn test_pyo3_analyzer_no_pyo3_attrs() {
    let analyzer = Pyo3Analyzer::new();
    let source = r#"
fn regular_function() -> i32 { 42 }

struct RegularStruct {
    value: i32,
}
"#;

    let bindings = analyzer.analyze(source, "lib.rs");
    assert!(bindings.is_empty());
}

#[test]
fn test_pyo3_analyzer_empty_source() {
    let analyzer = Pyo3Analyzer::new();
    let bindings = analyzer.analyze("", "lib.rs");
    assert!(bindings.is_empty());
}

#[test]
fn test_pyo3_analyzer_invalid_syntax() {
    let analyzer = Pyo3Analyzer::new();
    let source = "fn broken { incomplete";
    let bindings = analyzer.analyze(source, "lib.rs");
    assert!(bindings.is_empty());
}

// ==================== ReprC Analyzer ====================

#[test]
fn test_reprc_analyzer_detects_repr_c() {
    let analyzer = ReprCAnalyzer::new();
    let source = r#"
#[repr(C)]
struct GpuVertex {
    position: [f32; 3],
    normal: [f32; 3],
}
"#;

    let structs = analyzer.analyze(source, "lib.rs");

    assert_eq!(structs.len(), 1);
    assert_eq!(structs[0], "GpuVertex");
}

#[test]
fn test_reprc_analyzer_detects_repr_c_with_other_attrs() {
    let analyzer = ReprCAnalyzer::new();
    let source = r#"
#[repr(C, packed)]
struct PackedData {
    a: u8,
    b: u32,
}

#[derive(Debug)]
#[repr(C)]
struct DebugData {
    x: f32,
}
"#;

    let structs = analyzer.analyze(source, "lib.rs");

    assert_eq!(structs.len(), 2);
    assert!(structs.contains(&"PackedData".to_string()));
    assert!(structs.contains(&"DebugData".to_string()));
}

#[test]
fn test_reprc_analyzer_ignores_non_repr_c() {
    let analyzer = ReprCAnalyzer::new();
    let source = r#"
#[repr(Rust)]
struct RustRepr {
    value: i32,
}

#[derive(Debug)]
struct NoRepr {
    value: i32,
}
"#;

    let structs = analyzer.analyze(source, "lib.rs");
    assert!(structs.is_empty());
}

#[test]
fn test_reprc_analyzer_multiple_structs() {
    let analyzer = ReprCAnalyzer::new();
    let source = r#"
#[repr(C)]
struct Vertex {
    position: [f32; 3],
}

#[repr(C)]
struct Uniform {
    view: [[f32; 4]; 4],
}

#[repr(C)]
struct PushConstants {
    time: f32,
}
"#;

    let structs = analyzer.analyze(source, "lib.rs");

    assert_eq!(structs.len(), 3);
    assert!(structs.contains(&"Vertex".to_string()));
    assert!(structs.contains(&"Uniform".to_string()));
    assert!(structs.contains(&"PushConstants".to_string()));
}

// ==================== Struct Mirror Detection ====================

#[test]
fn test_detect_struct_mirrors_matching_names() {
    let mut graph = CodeGraph::new();

    // Add WGSL struct
    let wgsl_id = make_struct_node(&mut graph, "shader.wgsl", "Vertex", Language::Wgsl);

    // Add matching Rust struct
    let rust_id = make_struct_node(&mut graph, "types.rs", "Vertex", Language::Rust);

    let mirrors = detect_struct_mirrors(&graph);

    assert_eq!(mirrors.len(), 1);
    assert_eq!(mirrors[0], (wgsl_id, rust_id));
}

#[test]
fn test_detect_struct_mirrors_no_match() {
    let mut graph = CodeGraph::new();

    // Add WGSL struct
    make_struct_node(&mut graph, "shader.wgsl", "WgslVertex", Language::Wgsl);

    // Add Rust struct with different name
    make_struct_node(&mut graph, "types.rs", "RustVertex", Language::Rust);

    let mirrors = detect_struct_mirrors(&graph);
    assert!(mirrors.is_empty());
}

#[test]
fn test_detect_struct_mirrors_multiple_matches() {
    let mut graph = CodeGraph::new();

    // Add WGSL structs
    let wgsl_vertex = make_struct_node(&mut graph, "shader.wgsl", "Vertex", Language::Wgsl);
    let wgsl_uniform = make_struct_node(&mut graph, "shader.wgsl", "Uniform", Language::Wgsl);

    // Add matching Rust structs
    let rust_vertex = make_struct_node(&mut graph, "types.rs", "Vertex", Language::Rust);
    let rust_uniform = make_struct_node(&mut graph, "types.rs", "Uniform", Language::Rust);

    // Add non-matching structs
    make_struct_node(&mut graph, "shader.wgsl", "OnlyWgsl", Language::Wgsl);
    make_struct_node(&mut graph, "types.rs", "OnlyRust", Language::Rust);

    let mirrors = detect_struct_mirrors(&graph);

    assert_eq!(mirrors.len(), 2);
    assert!(mirrors.contains(&(wgsl_vertex, rust_vertex)));
    assert!(mirrors.contains(&(wgsl_uniform, rust_uniform)));
}

#[test]
fn test_detect_struct_mirrors_ignores_python() {
    let mut graph = CodeGraph::new();

    // Add Python class (not a struct mirror target)
    let id = NodeId(graph.nodes().len());
    let unit = CodeUnit {
        unit_type: UnitType::Class,
        name: "Vertex".to_string(),
        start_line: 1,
        end_line: 10,
        language: Language::Python,
        hashes: empty_hashes(),
    };
    let node = CodeNode::new(id, "types.py".to_string(), unit);
    graph.add_node(node);

    // Add WGSL struct
    make_struct_node(&mut graph, "shader.wgsl", "Vertex", Language::Wgsl);

    let mirrors = detect_struct_mirrors(&graph);
    assert!(mirrors.is_empty()); // Python class shouldn't match
}

// ==================== Cross-Lang Edge Creation ====================

#[test]
fn test_create_crosslang_edges_struct_mirrors() {
    let mut graph = CodeGraph::new();

    // Add matching structs
    let wgsl_id = make_struct_node(&mut graph, "shader.wgsl", "Vertex", Language::Wgsl);
    let rust_id = make_struct_node(&mut graph, "types.rs", "Vertex", Language::Rust);

    let stats = create_crosslang_edges(&mut graph, &[]);

    assert_eq!(stats.struct_mirrors, 1);
    assert_eq!(stats.edges_created, 1);

    let edges = graph.edges();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].source, wgsl_id);
    assert_eq!(edges[0].target, rust_id);
    assert_eq!(edges[0].edge_type, EdgeType::MirrorsLayout);
}

#[test]
fn test_create_crosslang_edges_pyo3_bindings() {
    let mut graph = CodeGraph::new();

    let bindings = vec![
        CrossLangBinding {
            file: "lib.rs".to_string(),
            name: "my_function".to_string(),
            binding_type: BindingType::PyFunction,
            source_lang: Language::Rust,
            target_lang: Language::Python,
        },
        CrossLangBinding {
            file: "lib.rs".to_string(),
            name: "MyClass".to_string(),
            binding_type: BindingType::PyClass,
            source_lang: Language::Rust,
            target_lang: Language::Python,
        },
    ];

    let stats = create_crosslang_edges(&mut graph, &bindings);

    assert_eq!(stats.bindings_found, 2);
    assert_eq!(stats.pyo3_functions, 1);
    assert_eq!(stats.pyo3_classes, 1);
}

#[test]
fn test_create_crosslang_edges_combined() {
    let mut graph = CodeGraph::new();

    // Add struct mirrors
    make_struct_node(&mut graph, "shader.wgsl", "Vertex", Language::Wgsl);
    make_struct_node(&mut graph, "types.rs", "Vertex", Language::Rust);

    // Add PyO3 binding
    let bindings = vec![
        CrossLangBinding {
            file: "lib.rs".to_string(),
            name: "process".to_string(),
            binding_type: BindingType::PyFunction,
            source_lang: Language::Rust,
            target_lang: Language::Python,
        },
    ];

    let stats = create_crosslang_edges(&mut graph, &bindings);

    assert_eq!(stats.bindings_found, 1);
    assert_eq!(stats.struct_mirrors, 1);
    assert_eq!(stats.edges_created, 1);
}

#[test]
fn test_crosslang_stats_accuracy() {
    let mut graph = CodeGraph::new();

    // Add multiple mirrors
    make_struct_node(&mut graph, "a.wgsl", "A", Language::Wgsl);
    make_struct_node(&mut graph, "a.rs", "A", Language::Rust);
    make_struct_node(&mut graph, "b.wgsl", "B", Language::Wgsl);
    make_struct_node(&mut graph, "b.rs", "B", Language::Rust);

    let bindings = vec![
        CrossLangBinding {
            file: "lib.rs".to_string(),
            name: "f1".to_string(),
            binding_type: BindingType::PyFunction,
            source_lang: Language::Rust,
            target_lang: Language::Python,
        },
        CrossLangBinding {
            file: "lib.rs".to_string(),
            name: "f2".to_string(),
            binding_type: BindingType::PyFunction,
            source_lang: Language::Rust,
            target_lang: Language::Python,
        },
        CrossLangBinding {
            file: "lib.rs".to_string(),
            name: "C".to_string(),
            binding_type: BindingType::PyClass,
            source_lang: Language::Rust,
            target_lang: Language::Python,
        },
    ];

    let stats = create_crosslang_edges(&mut graph, &bindings);

    assert_eq!(stats.bindings_found, 3);
    assert_eq!(stats.pyo3_functions, 2);
    assert_eq!(stats.pyo3_classes, 1);
    assert_eq!(stats.struct_mirrors, 2);
    assert_eq!(stats.edges_created, 2); // Only struct mirrors create edges
}
