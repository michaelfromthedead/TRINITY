//! Whitebox tests for WgslParser.
//!
//! WHITEBOX coverage plan:
//!   - Path A: empty input -> empty Vec
//!   - Path B: struct definition -> CodeUnit with Struct type
//!   - Path C: function definition -> CodeUnit with Function type
//!   - Path D: entry point (vertex/fragment/compute) -> CodeUnit with Function type
//!   - Path E: invalid syntax -> empty Vec
//!   - Path F: multiple items -> multiple CodeUnits
//!   - Path G: unnamed types (built-ins) -> not extracted

use std::path::Path;
use trinity_harness::parsers::{Language, ParserRegistry, UnitType, WgslParser};

#[test]
fn test_wgsl_parse_empty_input() {
    // Path A: empty input returns empty Vec
    let registry = ParserRegistry::new();
    let units = registry.parse_file(Path::new("test.wgsl"), "", Language::Wgsl);
    assert!(units.is_empty(), "empty input should produce no units");
}

#[test]
fn test_wgsl_parse_struct() {
    // Path B: struct definition
    let registry = ParserRegistry::new();
    let source = r#"
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let structs: Vec<_> = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Struct)
        .collect();

    assert_eq!(structs.len(), 1, "should find one struct");
    assert_eq!(structs[0].name, "VertexInput");
    assert_eq!(structs[0].language, Language::Wgsl);
}

#[test]
fn test_wgsl_parse_function() {
    // Path C: function definition
    let registry = ParserRegistry::new();
    let source = r#"
fn calculate(x: f32, y: f32) -> f32 {
    return x + y;
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let functions: Vec<_> = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Function)
        .collect();

    assert_eq!(functions.len(), 1, "should find one function");
    assert_eq!(functions[0].name, "calculate");
}

#[test]
fn test_wgsl_parse_vertex_entry_point() {
    // Path D: vertex entry point
    let registry = ParserRegistry::new();
    let source = r#"
@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let functions: Vec<_> = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Function)
        .collect();

    assert_eq!(functions.len(), 1, "should find the vertex entry point");
    assert_eq!(functions[0].name, "vs_main");
}

#[test]
fn test_wgsl_parse_fragment_entry_point() {
    // Path D: fragment entry point
    let registry = ParserRegistry::new();
    let source = r#"
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let functions: Vec<_> = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Function)
        .collect();

    assert_eq!(functions.len(), 1, "should find the fragment entry point");
    assert_eq!(functions[0].name, "fs_main");
}

#[test]
fn test_wgsl_parse_compute_entry_point() {
    // Path D: compute entry point
    let registry = ParserRegistry::new();
    let source = r#"
@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    // compute work
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let functions: Vec<_> = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Function)
        .collect();

    assert_eq!(functions.len(), 1, "should find the compute entry point");
    assert_eq!(functions[0].name, "cs_main");
}

#[test]
fn test_wgsl_parse_invalid_syntax() {
    // Path E: invalid syntax returns empty Vec
    let registry = ParserRegistry::new();
    let source = r#"
fn broken(
    // missing everything
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);
    assert!(units.is_empty(), "invalid syntax should produce no units");
}

#[test]
fn test_wgsl_parse_multiple_items() {
    // Path F: multiple items
    let registry = ParserRegistry::new();
    let source = r#"
struct Input {
    @location(0) pos: vec3<f32>,
}

struct Output {
    @builtin(position) pos: vec4<f32>,
}

fn helper() -> f32 {
    return 1.0;
}

@vertex
fn vs_main(input: Input) -> Output {
    var out: Output;
    out.pos = vec4<f32>(input.pos, 1.0);
    return out;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(helper(), 0.0, 0.0, 1.0);
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let structs = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Struct)
        .count();
    let functions = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Function)
        .count();

    assert_eq!(structs, 2, "should find 2 structs");
    assert_eq!(functions, 3, "should find 3 functions (helper + 2 entry points)");
}

#[test]
fn test_wgsl_parse_with_bindings() {
    // WGSL with various binding attributes
    let registry = ParserRegistry::new();
    let source = r#"
@group(0) @binding(0)
var<uniform> camera: mat4x4<f32>;

@group(0) @binding(1)
var<storage, read_write> data: array<f32>;

struct Uniforms {
    time: f32,
}

fn process() -> f32 {
    return 0.0;
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    // Should find struct and function, not variables
    let structs = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Struct)
        .count();
    let functions = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Function)
        .count();

    assert_eq!(structs, 1, "should find 1 struct");
    assert_eq!(functions, 1, "should find 1 function");
}

#[test]
fn test_wgsl_parser_default_impl() {
    // WgslParser::default() should work
    let _parser = WgslParser::default();
}

#[test]
fn test_wgsl_parse_unnamed_struct_not_extracted() {
    // Path G: unnamed types are not extracted
    // Anonymous structs in return types etc. shouldn't appear as named units
    let registry = ParserRegistry::new();
    let source = r#"
fn foo() -> vec4<f32> {
    return vec4<f32>(0.0);
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    // Only the function should be found, no anonymous types
    let structs = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Struct)
        .count();
    assert_eq!(structs, 0, "should not find anonymous types as structs");
}
