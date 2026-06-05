//! Blackbox tests for WgslParser via ParserRegistry.
//!
//! CLEANROOM: Tests are written based on the public contract only.
//! The implementation file (wgsl.rs) was NOT read.
//!
//! Test coverage plan:
//!   - T1: Basic struct extraction
//!   - T2: Struct with member locations and offsets
//!   - T3: Basic function extraction
//!   - T4: Entry point extraction (vertex/fragment/compute)
//!   - T5: Binding extraction (@group/@binding)
//!   - T6: Multiple items in single file
//!   - T7: Complex shader patterns
//!   - T8: Hash computation validation
//!   - T9: Line number accuracy
//!   - T10: Edge cases (empty, invalid syntax)
//!   - T11: Real-world shader patterns
//!   - T12: Struct alignment (layout critical for GPU)
//!   - T13: Various WGSL types
//!   - T14: Workgroup and storage buffers
//!   - T15: Comprehensive render pipeline shader

use std::path::Path;
use trinity_harness::{Language, ParserRegistry, UnitType};

// ============================================================================
// T1: Basic Struct Extraction
// ============================================================================

#[test]
fn blackbox_wgsl_simple_struct() {
    let registry = ParserRegistry::new();
    let source = r#"
struct Simple {
    value: f32,
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let structs: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Struct).collect();
    assert_eq!(structs.len(), 1);
    assert_eq!(structs[0].name, "Simple");
    assert_eq!(structs[0].language, Language::Wgsl);
}

#[test]
fn blackbox_wgsl_struct_multiple_fields() {
    let registry = ParserRegistry::new();
    let source = r#"
struct MultiField {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let structs: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Struct).collect();
    assert_eq!(structs.len(), 1);
    assert_eq!(structs[0].name, "MultiField");
}

#[test]
fn blackbox_wgsl_multiple_structs() {
    let registry = ParserRegistry::new();
    let source = r#"
struct First {
    a: i32,
}

struct Second {
    b: u32,
}

struct Third {
    c: f32,
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let structs: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Struct).collect();
    assert_eq!(structs.len(), 3);

    let names: Vec<_> = structs.iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"First"));
    assert!(names.contains(&"Second"));
    assert!(names.contains(&"Third"));
}

// ============================================================================
// T2: Struct with Member Locations and Offsets
// ============================================================================

#[test]
fn blackbox_wgsl_vertex_input_struct() {
    let registry = ParserRegistry::new();
    let source = r#"
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let structs: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Struct).collect();
    assert_eq!(structs.len(), 1);
    assert_eq!(structs[0].name, "VertexInput");
}

#[test]
fn blackbox_wgsl_vertex_output_struct() {
    let registry = ParserRegistry::new();
    let source = r#"
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let structs: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Struct).collect();
    assert_eq!(structs.len(), 1);
    assert_eq!(structs[0].name, "VertexOutput");
}

#[test]
fn blackbox_wgsl_uniform_struct() {
    let registry = ParserRegistry::new();
    let source = r#"
struct Uniforms {
    model: mat4x4<f32>,
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    time: f32,
    padding: vec3<f32>,
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let structs: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Struct).collect();
    assert_eq!(structs.len(), 1);
    assert_eq!(structs[0].name, "Uniforms");
}

// ============================================================================
// T3: Basic Function Extraction
// ============================================================================

#[test]
fn blackbox_wgsl_simple_function() {
    let registry = ParserRegistry::new();
    let source = r#"
fn simple() {
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let functions: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Function).collect();
    assert_eq!(functions.len(), 1);
    assert_eq!(functions[0].name, "simple");
    assert_eq!(functions[0].language, Language::Wgsl);
}

#[test]
fn blackbox_wgsl_function_with_params() {
    let registry = ParserRegistry::new();
    let source = r#"
fn add(x: f32, y: f32) -> f32 {
    return x + y;
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let functions: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Function).collect();
    assert_eq!(functions.len(), 1);
    assert_eq!(functions[0].name, "add");
}

#[test]
fn blackbox_wgsl_function_with_vector_types() {
    let registry = ParserRegistry::new();
    let source = r#"
fn normalize_vec3(v: vec3<f32>) -> vec3<f32> {
    let len = length(v);
    return v / len;
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let functions: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Function).collect();
    assert_eq!(functions.len(), 1);
    assert_eq!(functions[0].name, "normalize_vec3");
}

#[test]
fn blackbox_wgsl_multiple_functions() {
    let registry = ParserRegistry::new();
    let source = r#"
fn first() -> f32 {
    return 1.0;
}

fn second() -> f32 {
    return 2.0;
}

fn third() -> f32 {
    return 3.0;
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let functions: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Function).collect();
    assert_eq!(functions.len(), 3);

    let names: Vec<_> = functions.iter().map(|f| f.name.as_str()).collect();
    assert!(names.contains(&"first"));
    assert!(names.contains(&"second"));
    assert!(names.contains(&"third"));
}

// ============================================================================
// T4: Entry Point Extraction
// ============================================================================

#[test]
fn blackbox_wgsl_vertex_entry_point() {
    let registry = ParserRegistry::new();
    let source = r#"
@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let functions: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Function).collect();
    assert_eq!(functions.len(), 1);
    assert_eq!(functions[0].name, "vs_main");
}

#[test]
fn blackbox_wgsl_fragment_entry_point() {
    let registry = ParserRegistry::new();
    let source = r#"
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let functions: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Function).collect();
    assert_eq!(functions.len(), 1);
    assert_eq!(functions[0].name, "fs_main");
}

#[test]
fn blackbox_wgsl_compute_entry_point() {
    let registry = ParserRegistry::new();
    let source = r#"
@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    // compute work
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let functions: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Function).collect();
    assert_eq!(functions.len(), 1);
    assert_eq!(functions[0].name, "cs_main");
}

#[test]
fn blackbox_wgsl_compute_with_3d_workgroup() {
    let registry = ParserRegistry::new();
    let source = r#"
@compute @workgroup_size(8, 8, 1)
fn cs_2d(@builtin(global_invocation_id) id: vec3<u32>) {
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let functions: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Function).collect();
    assert_eq!(functions.len(), 1);
    assert_eq!(functions[0].name, "cs_2d");
}

#[test]
fn blackbox_wgsl_multiple_entry_points() {
    let registry = ParserRegistry::new();
    let source = r#"
@vertex
fn vs_main() -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let functions: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Function).collect();
    assert_eq!(functions.len(), 2);

    let names: Vec<_> = functions.iter().map(|f| f.name.as_str()).collect();
    assert!(names.contains(&"vs_main"));
    assert!(names.contains(&"fs_main"));
}

// ============================================================================
// T5: Binding Extraction
// ============================================================================

#[test]
fn blackbox_wgsl_uniform_binding() {
    let registry = ParserRegistry::new();
    let source = r#"
struct Uniforms {
    mvp: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

fn use_uniform() -> mat4x4<f32> {
    return uniforms.mvp;
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    // Should find struct and function
    let structs = units.iter().filter(|u| u.unit_type == UnitType::Struct).count();
    let functions = units.iter().filter(|u| u.unit_type == UnitType::Function).count();

    assert_eq!(structs, 1, "should find 1 struct");
    assert_eq!(functions, 1, "should find 1 function");
}

#[test]
fn blackbox_wgsl_storage_buffer_binding() {
    let registry = ParserRegistry::new();
    let source = r#"
struct Data {
    values: array<f32>,
}

@group(0) @binding(0)
var<storage, read> input_data: Data;

@group(0) @binding(1)
var<storage, read_write> output_data: Data;

fn process() {
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let structs = units.iter().filter(|u| u.unit_type == UnitType::Struct).count();
    let functions = units.iter().filter(|u| u.unit_type == UnitType::Function).count();

    assert_eq!(structs, 1, "should find 1 struct");
    assert_eq!(functions, 1, "should find 1 function");
}

#[test]
fn blackbox_wgsl_texture_sampler_binding() {
    let registry = ParserRegistry::new();
    let source = r#"
@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;

@group(0) @binding(1)
var s_diffuse: sampler;

fn sample_texture(uv: vec2<f32>) -> vec4<f32> {
    return textureSample(t_diffuse, s_diffuse, uv);
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let functions = units.iter().filter(|u| u.unit_type == UnitType::Function).count();
    assert_eq!(functions, 1, "should find 1 function");
}

#[test]
fn blackbox_wgsl_multiple_binding_groups() {
    let registry = ParserRegistry::new();
    let source = r#"
struct CameraUniforms {
    view_proj: mat4x4<f32>,
}

struct ModelUniforms {
    model: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(1) @binding(0)
var<uniform> model: ModelUniforms;

fn transform() -> mat4x4<f32> {
    return camera.view_proj * model.model;
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let structs = units.iter().filter(|u| u.unit_type == UnitType::Struct).count();
    let functions = units.iter().filter(|u| u.unit_type == UnitType::Function).count();

    assert_eq!(structs, 2, "should find 2 structs");
    assert_eq!(functions, 1, "should find 1 function");
}

// ============================================================================
// T6: Multiple Items in Single File
// ============================================================================

#[test]
fn blackbox_wgsl_mixed_items() {
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

    let structs = units.iter().filter(|u| u.unit_type == UnitType::Struct).count();
    let functions = units.iter().filter(|u| u.unit_type == UnitType::Function).count();

    assert_eq!(structs, 2, "should find 2 structs (Input, Output)");
    assert_eq!(functions, 3, "should find 3 functions (helper, vs_main, fs_main)");
}

// ============================================================================
// T7: Complex Shader Patterns
// ============================================================================

#[test]
fn blackbox_wgsl_lighting_functions() {
    let registry = ParserRegistry::new();
    let source = r#"
fn calculate_diffuse(normal: vec3<f32>, light_dir: vec3<f32>) -> f32 {
    return max(dot(normal, light_dir), 0.0);
}

fn calculate_specular(normal: vec3<f32>, light_dir: vec3<f32>, view_dir: vec3<f32>, shininess: f32) -> f32 {
    let reflect_dir = reflect(-light_dir, normal);
    return pow(max(dot(view_dir, reflect_dir), 0.0), shininess);
}

fn calculate_fresnel(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let functions: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Function).collect();
    assert_eq!(functions.len(), 3);

    let names: Vec<_> = functions.iter().map(|f| f.name.as_str()).collect();
    assert!(names.contains(&"calculate_diffuse"));
    assert!(names.contains(&"calculate_specular"));
    assert!(names.contains(&"calculate_fresnel"));
}

#[test]
fn blackbox_wgsl_matrix_operations() {
    let registry = ParserRegistry::new();
    let source = r#"
fn create_rotation_x(angle: f32) -> mat4x4<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return mat4x4<f32>(
        vec4<f32>(1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, c, -s, 0.0),
        vec4<f32>(0.0, s, c, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0)
    );
}

fn transform_point(m: mat4x4<f32>, p: vec3<f32>) -> vec3<f32> {
    let p4 = m * vec4<f32>(p, 1.0);
    return p4.xyz / p4.w;
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let functions = units.iter().filter(|u| u.unit_type == UnitType::Function).count();
    assert_eq!(functions, 2);
}

// ============================================================================
// T8: Hash Computation Validation
// ============================================================================

#[test]
fn blackbox_wgsl_hash_populated() {
    let registry = ParserRegistry::new();
    let source = r#"
fn compute(x: f32) -> f32 {
    return x * 2.0;
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    assert_eq!(units.len(), 1);

    // Full hash should be non-zero (not default)
    let hashes = &units[0].hashes;
    assert_ne!(hashes.full_hash, [0u8; 32], "full_hash should be computed");
}

#[test]
fn blackbox_wgsl_different_functions_different_hashes() {
    let registry = ParserRegistry::new();

    let source1 = "fn foo() -> f32 { return 1.0; }";
    let source2 = "fn bar() -> f32 { return 2.0; }";

    let units1 = registry.parse_file(Path::new("a.wgsl"), source1, Language::Wgsl);
    let units2 = registry.parse_file(Path::new("b.wgsl"), source2, Language::Wgsl);

    assert_eq!(units1.len(), 1);
    assert_eq!(units2.len(), 1);

    // Different function content should produce different hashes
    assert_ne!(
        units1[0].hashes.full_hash,
        units2[0].hashes.full_hash,
        "different functions should have different full hashes"
    );
}

#[test]
fn blackbox_wgsl_same_function_same_hash() {
    let registry = ParserRegistry::new();

    let source = "fn identical() -> f32 { return 42.0; }";

    let units1 = registry.parse_file(Path::new("a.wgsl"), source, Language::Wgsl);
    let units2 = registry.parse_file(Path::new("b.wgsl"), source, Language::Wgsl);

    assert_eq!(units1.len(), 1);
    assert_eq!(units2.len(), 1);

    // Same function content should produce same hash
    assert_eq!(
        units1[0].hashes.full_hash,
        units2[0].hashes.full_hash,
        "identical functions should have same full hash"
    );
}

#[test]
fn blackbox_wgsl_struct_hash_populated() {
    let registry = ParserRegistry::new();
    let source = r#"
struct Data {
    x: f32,
    y: f32,
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    assert_eq!(units.len(), 1);
    assert_ne!(units[0].hashes.full_hash, [0u8; 32], "struct full_hash should be computed");
}

// ============================================================================
// T9: Line Number Accuracy
// ============================================================================

#[test]
fn blackbox_wgsl_line_numbers_start_line() {
    let registry = ParserRegistry::new();
    let source = r#"
fn first() -> f32 {
    return 1.0;
}

fn second() -> f32 {
    return 2.0;
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let functions: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Function).collect();
    assert_eq!(functions.len(), 2);

    let first = functions.iter().find(|u| u.name == "first").unwrap();
    let second = functions.iter().find(|u| u.name == "second").unwrap();

    // First function should start early in file
    assert!(first.start_line >= 1, "first should start early in file");
    // Second function should start after first
    assert!(second.start_line > first.start_line, "second should start after first");
}

#[test]
fn blackbox_wgsl_line_numbers_end_after_start() {
    let registry = ParserRegistry::new();
    let source = r#"
fn multiline() -> f32 {
    var a: f32 = 1.0;
    var b: f32 = 2.0;
    var c: f32 = a + b;
    return c;
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    assert_eq!(units.len(), 1);
    assert!(
        units[0].end_line >= units[0].start_line,
        "end_line should be >= start_line"
    );
}

#[test]
fn blackbox_wgsl_struct_line_span() {
    let registry = ParserRegistry::new();
    let source = r#"
struct LargeStruct {
    field1: f32,
    field2: f32,
    field3: f32,
    field4: f32,
    field5: f32,
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let structs: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Struct).collect();
    assert_eq!(structs.len(), 1);

    // Struct should span multiple lines
    let struct_span = structs[0].end_line - structs[0].start_line;
    assert!(struct_span >= 3, "struct should span multiple lines");
}

// ============================================================================
// T10: Edge Cases
// ============================================================================

#[test]
fn blackbox_wgsl_empty_source() {
    let registry = ParserRegistry::new();
    let units = registry.parse_file(Path::new("empty.wgsl"), "", Language::Wgsl);
    assert!(units.is_empty(), "empty source should produce no units");
}

#[test]
fn blackbox_wgsl_whitespace_only() {
    let registry = ParserRegistry::new();
    let units = registry.parse_file(Path::new("whitespace.wgsl"), "   \n\n\t\t\n   ", Language::Wgsl);
    assert!(units.is_empty(), "whitespace-only source should produce no units");
}

#[test]
fn blackbox_wgsl_comments_only() {
    let registry = ParserRegistry::new();
    let source = r#"
// This is a comment
// Another comment
/* Block comment */
"#;
    let units = registry.parse_file(Path::new("comments.wgsl"), source, Language::Wgsl);
    assert!(units.is_empty(), "comments-only source should produce no units");
}

#[test]
fn blackbox_wgsl_invalid_syntax() {
    let registry = ParserRegistry::new();
    let source = "fn broken(";
    let units = registry.parse_file(Path::new("broken.wgsl"), source, Language::Wgsl);
    // Invalid syntax should either return empty or gracefully handle
    // The exact behavior depends on implementation, but it should not panic
    let _ = units;
}

#[test]
fn blackbox_wgsl_only_variable_declarations() {
    let registry = ParserRegistry::new();
    let source = r#"
var<private> x: f32 = 1.0;
const PI: f32 = 3.14159;
"#;
    let units = registry.parse_file(Path::new("vars.wgsl"), source, Language::Wgsl);
    // Module-level variables and constants may or may not be extracted
    // depending on implementation - key is no panic
    let _ = units;
}

// ============================================================================
// T11: Real-World Shader Patterns
// ============================================================================

#[test]
fn blackbox_wgsl_pbr_lighting_structures() {
    let registry = ParserRegistry::new();
    let source = r#"
struct Material {
    base_color: vec4<f32>,
    metallic: f32,
    roughness: f32,
    ao: f32,
    _padding: f32,
}

struct Light {
    position: vec3<f32>,
    _pad1: f32,
    color: vec3<f32>,
    intensity: f32,
}

struct LightingParams {
    ambient: vec3<f32>,
    _pad2: f32,
    view_position: vec3<f32>,
    _pad3: f32,
}
"#;
    let units = registry.parse_file(Path::new("pbr.wgsl"), source, Language::Wgsl);

    let structs: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Struct).collect();
    assert_eq!(structs.len(), 3);

    let names: Vec<_> = structs.iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"Material"));
    assert!(names.contains(&"Light"));
    assert!(names.contains(&"LightingParams"));
}

#[test]
fn blackbox_wgsl_particle_system() {
    let registry = ParserRegistry::new();
    let source = r#"
struct Particle {
    position: vec3<f32>,
    velocity: vec3<f32>,
    color: vec4<f32>,
    lifetime: f32,
    size: f32,
}

struct ParticleSystem {
    particles: array<Particle>,
}

@group(0) @binding(0)
var<storage, read_write> particle_buffer: ParticleSystem;

@compute @workgroup_size(256)
fn update_particles(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    // update logic
}
"#;
    let units = registry.parse_file(Path::new("particles.wgsl"), source, Language::Wgsl);

    let structs = units.iter().filter(|u| u.unit_type == UnitType::Struct).count();
    let functions = units.iter().filter(|u| u.unit_type == UnitType::Function).count();

    assert_eq!(structs, 2, "should find Particle and ParticleSystem structs");
    assert_eq!(functions, 1, "should find update_particles compute shader");
}

#[test]
fn blackbox_wgsl_ray_tracing_structures() {
    let registry = ParserRegistry::new();
    let source = r#"
struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

struct HitRecord {
    point: vec3<f32>,
    normal: vec3<f32>,
    t: f32,
    front_face: bool,
}

fn create_ray(origin: vec3<f32>, direction: vec3<f32>) -> Ray {
    var r: Ray;
    r.origin = origin;
    r.direction = normalize(direction);
    return r;
}

fn ray_at(r: Ray, t: f32) -> vec3<f32> {
    return r.origin + t * r.direction;
}
"#;
    let units = registry.parse_file(Path::new("raytracing.wgsl"), source, Language::Wgsl);

    let structs: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Struct).collect();
    let functions: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Function).collect();

    assert_eq!(structs.len(), 2);
    assert_eq!(functions.len(), 2);

    assert!(structs.iter().any(|s| s.name == "Ray"));
    assert!(structs.iter().any(|s| s.name == "HitRecord"));
    assert!(functions.iter().any(|f| f.name == "create_ray"));
    assert!(functions.iter().any(|f| f.name == "ray_at"));
}

// ============================================================================
// T12: Struct Alignment (Layout Critical for GPU)
// ============================================================================

#[test]
fn blackbox_wgsl_aligned_struct_16_byte() {
    let registry = ParserRegistry::new();
    let source = r#"
// 16-byte aligned for GPU uniforms
struct AlignedData {
    value: vec4<f32>,  // 16 bytes, naturally aligned
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let structs: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Struct).collect();
    assert_eq!(structs.len(), 1);
    assert_eq!(structs[0].name, "AlignedData");
}

#[test]
fn blackbox_wgsl_struct_with_padding() {
    let registry = ParserRegistry::new();
    let source = r#"
// Struct with explicit padding for alignment
struct PaddedUniforms {
    time: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
    resolution: vec2<f32>,
    _pad4: vec2<f32>,
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let structs: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Struct).collect();
    assert_eq!(structs.len(), 1);
    assert_eq!(structs[0].name, "PaddedUniforms");
}

#[test]
fn blackbox_wgsl_struct_array_stride() {
    let registry = ParserRegistry::new();
    let source = r#"
// Struct used in arrays must be stride-aligned
struct ArrayElement {
    position: vec3<f32>,
    _pad: f32,  // Ensure 16-byte stride
}

struct ElementBuffer {
    elements: array<ArrayElement>,
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let structs: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Struct).collect();
    assert_eq!(structs.len(), 2);
}

// ============================================================================
// T13: Various WGSL Types
// ============================================================================

#[test]
fn blackbox_wgsl_scalar_types() {
    let registry = ParserRegistry::new();
    // Note: f16 requires specific extensions and may not be universally supported
    // Testing with core WGSL scalar types only
    let source = r#"
struct ScalarTypes {
    a: i32,
    b: u32,
    c: f32,
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let structs: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Struct).collect();
    assert_eq!(structs.len(), 1);
    assert_eq!(structs[0].name, "ScalarTypes");
}

#[test]
fn blackbox_wgsl_vector_types() {
    let registry = ParserRegistry::new();
    let source = r#"
struct VectorTypes {
    v2f: vec2<f32>,
    v3f: vec3<f32>,
    v4f: vec4<f32>,
    v2i: vec2<i32>,
    v3u: vec3<u32>,
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let structs: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Struct).collect();
    assert_eq!(structs.len(), 1);
    assert_eq!(structs[0].name, "VectorTypes");
}

#[test]
fn blackbox_wgsl_matrix_types() {
    let registry = ParserRegistry::new();
    let source = r#"
struct MatrixTypes {
    m2x2: mat2x2<f32>,
    m3x3: mat3x3<f32>,
    m4x4: mat4x4<f32>,
    m2x3: mat2x3<f32>,
    m3x4: mat3x4<f32>,
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let structs: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Struct).collect();
    assert_eq!(structs.len(), 1);
    assert_eq!(structs[0].name, "MatrixTypes");
}

#[test]
fn blackbox_wgsl_array_types() {
    let registry = ParserRegistry::new();
    let source = r#"
struct ArrayTypes {
    fixed_array: array<f32, 16>,
}

struct RuntimeArray {
    data: array<vec4<f32>>,
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let structs: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Struct).collect();
    assert_eq!(structs.len(), 2);
}

// ============================================================================
// T14: Workgroup and Storage Buffers
// ============================================================================

#[test]
fn blackbox_wgsl_workgroup_variable() {
    let registry = ParserRegistry::new();
    let source = r#"
var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn reduce(@builtin(local_invocation_id) lid: vec3<u32>) {
    shared_data[lid.x] = f32(lid.x);
    workgroupBarrier();
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let functions = units.iter().filter(|u| u.unit_type == UnitType::Function).count();
    assert_eq!(functions, 1, "should find reduce compute shader");
}

#[test]
fn blackbox_wgsl_push_constants() {
    let registry = ParserRegistry::new();
    let source = r#"
struct PushConstants {
    offset: vec2<f32>,
    scale: f32,
}

var<push_constant> pc: PushConstants;

fn apply_transform(p: vec2<f32>) -> vec2<f32> {
    return p * pc.scale + pc.offset;
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let structs = units.iter().filter(|u| u.unit_type == UnitType::Struct).count();
    let functions = units.iter().filter(|u| u.unit_type == UnitType::Function).count();

    assert_eq!(structs, 1, "should find PushConstants struct");
    assert_eq!(functions, 1, "should find apply_transform function");
}

// ============================================================================
// T15: Comprehensive Render Pipeline Shader
// ============================================================================

#[test]
fn blackbox_wgsl_complete_render_shader() {
    let registry = ParserRegistry::new();
    let source = r#"
// Vertex input from mesh
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coord: vec2<f32>,
}

// Data passed to fragment shader
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) tex_coord: vec2<f32>,
}

// Camera and transform uniforms
struct CameraUniforms {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    view_position: vec3<f32>,
    _pad: f32,
}

struct ModelUniforms {
    model: mat4x4<f32>,
    normal_matrix: mat3x3<f32>,
}

// Bindings
@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(0) var<uniform> model: ModelUniforms;
@group(2) @binding(0) var t_albedo: texture_2d<f32>;
@group(2) @binding(1) var s_albedo: sampler;

// Helper function for normal transformation
fn transform_normal(n: vec3<f32>) -> vec3<f32> {
    return normalize(model.normal_matrix * n);
}

// Vertex shader
@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    let world_pos = model.model * vec4<f32>(input.position, 1.0);
    output.world_position = world_pos.xyz;
    output.clip_position = camera.projection * camera.view * world_pos;
    output.world_normal = transform_normal(input.normal);
    output.tex_coord = input.tex_coord;

    return output;
}

// Fragment shader
@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let albedo = textureSample(t_albedo, s_albedo, input.tex_coord);

    // Simple directional light
    let light_dir = normalize(vec3<f32>(1.0, 1.0, 1.0));
    let diffuse = max(dot(input.world_normal, light_dir), 0.0);

    return vec4<f32>(albedo.rgb * (0.3 + 0.7 * diffuse), albedo.a);
}
"#;
    let units = registry.parse_file(Path::new("complete.wgsl"), source, Language::Wgsl);

    // Verify structs
    let structs: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Struct).collect();
    assert!(structs.len() >= 4, "should find at least 4 structs");

    let struct_names: Vec<_> = structs.iter().map(|s| s.name.as_str()).collect();
    assert!(struct_names.contains(&"VertexInput"));
    assert!(struct_names.contains(&"VertexOutput"));
    assert!(struct_names.contains(&"CameraUniforms"));
    assert!(struct_names.contains(&"ModelUniforms"));

    // Verify functions
    let functions: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Function).collect();
    assert!(functions.len() >= 3, "should find at least 3 functions");

    let function_names: Vec<_> = functions.iter().map(|f| f.name.as_str()).collect();
    assert!(function_names.contains(&"transform_normal"));
    assert!(function_names.contains(&"vs_main"));
    assert!(function_names.contains(&"fs_main"));
}

// ============================================================================
// Language Detection
// ============================================================================

#[test]
fn blackbox_wgsl_language_detection() {
    assert_eq!(
        ParserRegistry::detect_language(Path::new("shader.wgsl")),
        Some(Language::Wgsl)
    );
    assert_eq!(
        ParserRegistry::detect_language(Path::new("/path/to/compute.wgsl")),
        Some(Language::Wgsl)
    );
}

#[test]
fn blackbox_wgsl_language_returned_correct() {
    let registry = ParserRegistry::new();
    let source = "fn test() {}";
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].language, Language::Wgsl);
}

// ============================================================================
// ParserRegistry API Tests
// ============================================================================

#[test]
fn blackbox_wgsl_registry_default() {
    let registry = ParserRegistry::default();
    let units = registry.parse_file(Path::new("test.wgsl"), "fn x() {}", Language::Wgsl);
    assert_eq!(units.len(), 1);
}

#[test]
fn blackbox_wgsl_parser_reusable() {
    let registry = ParserRegistry::new();

    let units1 = registry.parse_file(Path::new("a.wgsl"), "fn a() {}", Language::Wgsl);
    let units2 = registry.parse_file(Path::new("b.wgsl"), "fn b() {}", Language::Wgsl);
    let units3 = registry.parse_file(Path::new("c.wgsl"), "fn c() {}", Language::Wgsl);

    assert_eq!(units1.len(), 1);
    assert_eq!(units2.len(), 1);
    assert_eq!(units3.len(), 1);

    assert_eq!(units1[0].name, "a");
    assert_eq!(units2[0].name, "b");
    assert_eq!(units3[0].name, "c");
}

// ============================================================================
// Builtin Functions Usage (verify parser handles them)
// ============================================================================

#[test]
fn blackbox_wgsl_builtin_math_functions() {
    let registry = ParserRegistry::new();
    let source = r#"
fn use_builtins(x: f32, y: f32) -> f32 {
    let a = sin(x);
    let b = cos(y);
    let c = pow(x, y);
    let d = sqrt(a * a + b * b);
    let e = clamp(d, 0.0, 1.0);
    let f = mix(a, b, e);
    return f;
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "use_builtins");
}

#[test]
fn blackbox_wgsl_builtin_texture_functions() {
    let registry = ParserRegistry::new();
    let source = r#"
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

fn sample_operations(uv: vec2<f32>) -> vec4<f32> {
    let sample = textureSample(tex, samp, uv);
    let dims = textureDimensions(tex);
    let load = textureLoad(tex, vec2<i32>(0, 0), 0);
    return sample;
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let functions = units.iter().filter(|u| u.unit_type == UnitType::Function).count();
    assert_eq!(functions, 1);
}

// ============================================================================
// Control Flow Patterns
// ============================================================================

#[test]
fn blackbox_wgsl_control_flow() {
    let registry = ParserRegistry::new();
    let source = r#"
fn control_flow_test(x: i32) -> i32 {
    var result: i32 = 0;

    if (x > 0) {
        result = 1;
    } else if (x < 0) {
        result = -1;
    } else {
        result = 0;
    }

    for (var i: i32 = 0; i < 10; i = i + 1) {
        result = result + i;
    }

    var j: i32 = 0;
    while (j < 5) {
        result = result + 1;
        j = j + 1;
    }

    switch (x) {
        case 0: {
            result = 100;
        }
        case 1: {
            result = 200;
        }
        default: {
            result = 300;
        }
    }

    return result;
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "control_flow_test");
}

// ============================================================================
// Nested Structs (Struct containing struct)
// ============================================================================

#[test]
fn blackbox_wgsl_nested_struct_reference() {
    let registry = ParserRegistry::new();
    let source = r#"
struct Inner {
    value: f32,
}

struct Outer {
    inner: Inner,
    extra: f32,
}

fn use_nested(o: Outer) -> f32 {
    return o.inner.value + o.extra;
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let structs = units.iter().filter(|u| u.unit_type == UnitType::Struct).count();
    let functions = units.iter().filter(|u| u.unit_type == UnitType::Function).count();

    assert_eq!(structs, 2, "should find Inner and Outer structs");
    assert_eq!(functions, 1, "should find use_nested function");
}

// ============================================================================
// Anonymous types should NOT be extracted as named units
// ============================================================================

#[test]
fn blackbox_wgsl_anonymous_types_not_extracted() {
    let registry = ParserRegistry::new();
    let source = r#"
fn returns_vec4() -> vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}

fn takes_array(a: array<f32, 4>) -> f32 {
    return a[0];
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    // Only functions should be extracted, not the built-in types
    let structs = units.iter().filter(|u| u.unit_type == UnitType::Struct).count();
    let functions = units.iter().filter(|u| u.unit_type == UnitType::Function).count();

    assert_eq!(structs, 0, "built-in types should not be extracted as structs");
    assert_eq!(functions, 2, "should find both functions");
}
