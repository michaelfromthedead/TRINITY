//! Whitebox tests for dependency detection.

use trinity_harness::graph::{
    resolve_deps_to_edges, CodeEdge, CodeGraph, CodeNode, DepType, EdgeType, NodeId,
    PythonDepAnalyzer, RawDependency, RustDepAnalyzer,
};
use trinity_harness::parsers::{CodeUnit, ContentHashes, Language, ParserRegistry, UnitType};

fn empty_hashes() -> ContentHashes {
    ContentHashes {
        full_hash: [0u8; 32],
        signature_hash: [0u8; 32],
        body_hash: [0u8; 32],
        layout_hash: [0u8; 32],
    }
}

fn make_node(graph: &mut CodeGraph, file: &str, name: &str, unit_type: UnitType, lang: Language) -> NodeId {
    let id = NodeId(graph.nodes().len());
    let unit = CodeUnit {
        unit_type,
        name: name.to_string(),
        start_line: 1,
        end_line: 10,
        language: lang,
        hashes: empty_hashes(),
    };
    let node = CodeNode::new(id, file.to_string(), unit);
    graph.add_node(node)
}

// ==================== Rust Dependency Analyzer ====================

#[test]
fn test_rust_dep_analyzer_use_statement() {
    let analyzer = RustDepAnalyzer::new();
    let source = r#"
use std::collections::HashMap;
use crate::parser::RustParser;
"#;
    let deps = analyzer.analyze(source, "test.rs");

    // Should find HashMap and RustParser imports
    let import_names: Vec<_> = deps.iter()
        .filter(|d| d.dep_type == DepType::Imports)
        .map(|d| d.to_ref.as_str())
        .collect();

    assert!(import_names.contains(&"HashMap"));
    assert!(import_names.contains(&"RustParser"));
}

#[test]
fn test_rust_dep_analyzer_function_calls() {
    let analyzer = RustDepAnalyzer::new();
    let source = r#"
fn process_data() {
    let result = compute_value();
    transform(result);
}
"#;
    let deps = analyzer.analyze(source, "test.rs");

    let call_names: Vec<_> = deps.iter()
        .filter(|d| d.dep_type == DepType::Calls)
        .map(|d| d.to_ref.as_str())
        .collect();

    assert!(call_names.contains(&"compute_value"));
    assert!(call_names.contains(&"transform"));
}

#[test]
fn test_rust_dep_analyzer_method_calls() {
    let analyzer = RustDepAnalyzer::new();
    let source = r#"
fn process() {
    let x = Vec::new();
    x.push(1);
    x.iter().map(|i| i * 2);
}
"#;
    let deps = analyzer.analyze(source, "test.rs");

    let call_names: Vec<_> = deps.iter()
        .filter(|d| d.dep_type == DepType::Calls)
        .map(|d| d.to_ref.as_str())
        .collect();

    assert!(call_names.contains(&"new"));
    assert!(call_names.contains(&"push"));
    assert!(call_names.contains(&"iter"));
    assert!(call_names.contains(&"map"));
}

#[test]
fn test_rust_dep_analyzer_type_refs_in_signature() {
    let analyzer = RustDepAnalyzer::new();
    let source = r#"
fn process(input: InputData, config: Config) -> OutputResult {
    todo!()
}
"#;
    let deps = analyzer.analyze(source, "test.rs");

    let type_refs: Vec<_> = deps.iter()
        .filter(|d| d.dep_type == DepType::Uses)
        .map(|d| d.to_ref.as_str())
        .collect();

    assert!(type_refs.contains(&"InputData"));
    assert!(type_refs.contains(&"Config"));
    assert!(type_refs.contains(&"OutputResult"));
}

#[test]
fn test_rust_dep_analyzer_struct_field_types() {
    let analyzer = RustDepAnalyzer::new();
    let source = r#"
struct Container {
    data: DataStore,
    cache: CacheManager,
    count: usize,  // primitive, should be skipped
}
"#;
    let deps = analyzer.analyze(source, "test.rs");

    let type_refs: Vec<_> = deps.iter()
        .filter(|d| d.dep_type == DepType::Uses)
        .map(|d| d.to_ref.as_str())
        .collect();

    assert!(type_refs.contains(&"DataStore"));
    assert!(type_refs.contains(&"CacheManager"));
    assert!(!type_refs.contains(&"usize")); // primitive skipped
}

#[test]
fn test_rust_dep_analyzer_impl_trait_reference() {
    let analyzer = RustDepAnalyzer::new();
    let source = r#"
impl Display for MyType {
    fn fmt(&self, f: &mut Formatter) -> Result {
        Ok(())
    }
}
"#;
    let deps = analyzer.analyze(source, "test.rs");

    let uses_refs: Vec<_> = deps.iter()
        .filter(|d| d.dep_type == DepType::Uses)
        .map(|d| d.to_ref.as_str())
        .collect();

    assert!(uses_refs.contains(&"Display"));
}

#[test]
fn test_rust_dep_analyzer_empty_source() {
    let analyzer = RustDepAnalyzer::new();
    let deps = analyzer.analyze("", "test.rs");
    assert!(deps.is_empty());
}

#[test]
fn test_rust_dep_analyzer_invalid_syntax() {
    let analyzer = RustDepAnalyzer::new();
    let source = "fn broken { incomplete";
    let deps = analyzer.analyze(source, "test.rs");
    assert!(deps.is_empty());
}

// ==================== Python Dependency Analyzer ====================

#[test]
fn test_python_dep_analyzer_import_statement() {
    let analyzer = PythonDepAnalyzer::new();
    let source = r#"
import os
import sys
"#;
    let deps = analyzer.analyze(source, "test.py");

    let import_names: Vec<_> = deps.iter()
        .filter(|d| d.dep_type == DepType::Imports)
        .map(|d| d.to_ref.as_str())
        .collect();

    assert!(import_names.contains(&"os"));
    assert!(import_names.contains(&"sys"));
}

#[test]
fn test_python_dep_analyzer_from_import() {
    let analyzer = PythonDepAnalyzer::new();
    let source = r#"
from typing import List, Dict
from collections import defaultdict
"#;
    let deps = analyzer.analyze(source, "test.py");

    let import_names: Vec<_> = deps.iter()
        .filter(|d| d.dep_type == DepType::Imports)
        .map(|d| d.to_ref.as_str())
        .collect();

    assert!(import_names.contains(&"typing"));
    assert!(import_names.contains(&"List"));
    assert!(import_names.contains(&"Dict"));
    assert!(import_names.contains(&"collections"));
    assert!(import_names.contains(&"defaultdict"));
}

#[test]
fn test_python_dep_analyzer_function_calls() {
    let analyzer = PythonDepAnalyzer::new();
    let source = r#"
def process():
    result = compute()
    transform(result)
    return finalize(result)
"#;
    let deps = analyzer.analyze(source, "test.py");

    let call_names: Vec<_> = deps.iter()
        .filter(|d| d.dep_type == DepType::Calls)
        .map(|d| d.to_ref.as_str())
        .collect();

    assert!(call_names.contains(&"compute"));
    assert!(call_names.contains(&"transform"));
    assert!(call_names.contains(&"finalize"));
}

#[test]
fn test_python_dep_analyzer_method_calls() {
    let analyzer = PythonDepAnalyzer::new();
    let source = r#"
def process():
    data = []
    data.append(1)
    result = data.pop()
"#;
    let deps = analyzer.analyze(source, "test.py");

    let call_names: Vec<_> = deps.iter()
        .filter(|d| d.dep_type == DepType::Calls)
        .map(|d| d.to_ref.as_str())
        .collect();

    assert!(call_names.contains(&"append"));
    assert!(call_names.contains(&"pop"));
}

#[test]
fn test_python_dep_analyzer_class_inheritance() {
    let analyzer = PythonDepAnalyzer::new();
    let source = r#"
class MyClass(BaseClass):
    def method(self):
        pass
"#;
    let deps = analyzer.analyze(source, "test.py");

    let uses_refs: Vec<_> = deps.iter()
        .filter(|d| d.dep_type == DepType::Uses)
        .map(|d| d.to_ref.as_str())
        .collect();

    assert!(uses_refs.contains(&"BaseClass"));
}

#[test]
fn test_python_dep_analyzer_calls_in_conditionals() {
    let analyzer = PythonDepAnalyzer::new();
    let source = r#"
def process(x):
    if check_condition(x):
        handle_true()
    else:
        handle_false()
"#;
    let deps = analyzer.analyze(source, "test.py");

    let call_names: Vec<_> = deps.iter()
        .filter(|d| d.dep_type == DepType::Calls)
        .map(|d| d.to_ref.as_str())
        .collect();

    assert!(call_names.contains(&"check_condition"));
    assert!(call_names.contains(&"handle_true"));
    assert!(call_names.contains(&"handle_false"));
}

#[test]
fn test_python_dep_analyzer_calls_in_loops() {
    let analyzer = PythonDepAnalyzer::new();
    let source = r#"
def process():
    for item in get_items():
        process_item(item)

    while should_continue():
        do_work()
"#;
    let deps = analyzer.analyze(source, "test.py");

    let call_names: Vec<_> = deps.iter()
        .filter(|d| d.dep_type == DepType::Calls)
        .map(|d| d.to_ref.as_str())
        .collect();

    assert!(call_names.contains(&"get_items"));
    assert!(call_names.contains(&"process_item"));
    assert!(call_names.contains(&"should_continue"));
    assert!(call_names.contains(&"do_work"));
}

#[test]
fn test_python_dep_analyzer_empty_source() {
    let analyzer = PythonDepAnalyzer::new();
    let deps = analyzer.analyze("", "test.py");
    assert!(deps.is_empty());
}

// ==================== Dependency Resolution ====================

#[test]
fn test_resolve_deps_creates_edges() {
    let mut graph = CodeGraph::new();

    // Create nodes
    let caller_id = make_node(&mut graph, "a.rs", "caller", UnitType::Function, Language::Rust);
    let callee_id = make_node(&mut graph, "b.rs", "callee", UnitType::Function, Language::Rust);

    // Create raw dependency
    let deps = vec![
        RawDependency {
            from_file: "a.rs".to_string(),
            from_name: "caller".to_string(),
            from_line: 1,
            to_ref: "callee".to_string(),
            dep_type: DepType::Calls,
        },
    ];

    let stats = resolve_deps_to_edges(&mut graph, &deps);

    assert_eq!(stats.deps_found, 1);
    assert_eq!(stats.deps_resolved, 1);
    assert_eq!(graph.edges().len(), 1);

    let edge = &graph.edges()[0];
    assert_eq!(edge.source, caller_id);
    assert_eq!(edge.target, callee_id);
    assert_eq!(edge.edge_type, EdgeType::Calls);
}

#[test]
fn test_resolve_deps_unresolved_target() {
    let mut graph = CodeGraph::new();

    // Create only source node
    let _ = make_node(&mut graph, "a.rs", "caller", UnitType::Function, Language::Rust);

    // Create dependency to non-existent target
    let deps = vec![
        RawDependency {
            from_file: "a.rs".to_string(),
            from_name: "caller".to_string(),
            from_line: 1,
            to_ref: "nonexistent".to_string(),
            dep_type: DepType::Calls,
        },
    ];

    let stats = resolve_deps_to_edges(&mut graph, &deps);

    assert_eq!(stats.deps_found, 1);
    assert_eq!(stats.deps_unresolved, 1);
    assert_eq!(graph.edges().len(), 0);
}

#[test]
fn test_resolve_deps_file_level_import_skipped() {
    let mut graph = CodeGraph::new();

    // File-level import (no source node)
    let deps = vec![
        RawDependency {
            from_file: "a.rs".to_string(),
            from_name: "".to_string(), // Empty = file-level
            from_line: 0,
            to_ref: "SomeType".to_string(),
            dep_type: DepType::Imports,
        },
    ];

    let stats = resolve_deps_to_edges(&mut graph, &deps);

    assert_eq!(stats.deps_found, 1);
    assert_eq!(stats.deps_unresolved, 1); // Skipped, counted as unresolved
    assert_eq!(graph.edges().len(), 0);
}

#[test]
fn test_resolve_deps_multiple_edge_types() {
    let mut graph = CodeGraph::new();

    let fn_id = make_node(&mut graph, "a.rs", "process", UnitType::Function, Language::Rust);
    let struct_id = make_node(&mut graph, "b.rs", "Data", UnitType::Struct, Language::Rust);
    let helper_id = make_node(&mut graph, "c.rs", "helper", UnitType::Function, Language::Rust);

    let deps = vec![
        RawDependency {
            from_file: "a.rs".to_string(),
            from_name: "process".to_string(),
            from_line: 1,
            to_ref: "Data".to_string(),
            dep_type: DepType::Uses,
        },
        RawDependency {
            from_file: "a.rs".to_string(),
            from_name: "process".to_string(),
            from_line: 1,
            to_ref: "helper".to_string(),
            dep_type: DepType::Calls,
        },
    ];

    let stats = resolve_deps_to_edges(&mut graph, &deps);

    assert_eq!(stats.deps_found, 2);
    assert_eq!(stats.deps_resolved, 2);
    assert_eq!(graph.edges().len(), 2);

    // Verify edge types
    let edge_types: Vec<_> = graph.edges().iter().map(|e| e.edge_type).collect();
    assert!(edge_types.contains(&EdgeType::Uses));
    assert!(edge_types.contains(&EdgeType::Calls));
}

#[test]
fn test_dep_stats_edges_by_type() {
    let mut graph = CodeGraph::new();

    let a = make_node(&mut graph, "a.rs", "a", UnitType::Function, Language::Rust);
    let b = make_node(&mut graph, "a.rs", "b", UnitType::Function, Language::Rust);
    let c = make_node(&mut graph, "a.rs", "c", UnitType::Function, Language::Rust);
    let d = make_node(&mut graph, "a.rs", "D", UnitType::Struct, Language::Rust);

    let deps = vec![
        RawDependency {
            from_file: "a.rs".to_string(),
            from_name: "a".to_string(),
            from_line: 1,
            to_ref: "b".to_string(),
            dep_type: DepType::Calls,
        },
        RawDependency {
            from_file: "a.rs".to_string(),
            from_name: "a".to_string(),
            from_line: 1,
            to_ref: "c".to_string(),
            dep_type: DepType::Calls,
        },
        RawDependency {
            from_file: "a.rs".to_string(),
            from_name: "a".to_string(),
            from_line: 1,
            to_ref: "D".to_string(),
            dep_type: DepType::Uses,
        },
    ];

    let stats = resolve_deps_to_edges(&mut graph, &deps);

    assert_eq!(stats.edges_by_type.get(&EdgeType::Calls), Some(&2));
    assert_eq!(stats.edges_by_type.get(&EdgeType::Uses), Some(&1));
}
