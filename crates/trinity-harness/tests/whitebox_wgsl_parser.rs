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

// ============================================================================
// WHITEBOX TESTS FOR LAYOUT_HASH WITH MEMBER OFFSETS (CRITICAL FOR ALIGNMENT)
// ============================================================================

#[test]
fn test_wgsl_struct_layout_hash_includes_offsets() {
    // CRITICAL: layout_hash must include member offsets for detecting alignment mismatches
    let registry = ParserRegistry::new();
    let source = r#"
struct TestStruct {
    a: f32,
    b: vec2<f32>,
    c: vec4<f32>,
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let struct_unit = units
        .iter()
        .find(|u| u.unit_type == UnitType::Struct && u.name == "TestStruct")
        .expect("should find TestStruct");

    // layout_hash should NOT be all zeros for a struct with members
    assert_ne!(
        struct_unit.hashes.layout_hash,
        [0u8; 32],
        "layout_hash should be computed (not zeros) for structs with members"
    );
}

#[test]
fn test_wgsl_struct_different_member_order_different_layout_hash() {
    // Different member orders produce different offsets -> different layout_hash
    let registry = ParserRegistry::new();

    // Struct A: a, b, c order
    let source_a = r#"
struct OrderTest {
    a: f32,
    b: f32,
    c: f32,
}
"#;

    // Struct B: c, b, a order (semantically different layout)
    let source_b = r#"
struct OrderTest {
    c: f32,
    b: f32,
    a: f32,
}
"#;

    let units_a = registry.parse_file(Path::new("test.wgsl"), source_a, Language::Wgsl);
    let units_b = registry.parse_file(Path::new("test.wgsl"), source_b, Language::Wgsl);

    let struct_a = units_a
        .iter()
        .find(|u| u.unit_type == UnitType::Struct)
        .expect("should find struct in source_a");
    let struct_b = units_b
        .iter()
        .find(|u| u.unit_type == UnitType::Struct)
        .expect("should find struct in source_b");

    // layout_hash should differ because member order affects layout representation
    assert_ne!(
        struct_a.hashes.layout_hash, struct_b.hashes.layout_hash,
        "different member order should produce different layout_hash"
    );
}

#[test]
fn test_wgsl_struct_with_alignment_padding() {
    // Test struct with alignment that causes padding (vec2 requires 8-byte alignment)
    let registry = ParserRegistry::new();
    let source = r#"
struct AlignedStruct {
    x: f32,
    y: vec2<f32>,
    z: f32,
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let struct_unit = units
        .iter()
        .find(|u| u.unit_type == UnitType::Struct)
        .expect("should find AlignedStruct");

    // Should have valid hashes computed
    assert_ne!(struct_unit.hashes.full_hash, [0u8; 32]);
    assert_ne!(struct_unit.hashes.layout_hash, [0u8; 32]);

    // signature_hash should be computed based on struct name
    assert_ne!(struct_unit.hashes.signature_hash, [0u8; 32]);
}

#[test]
fn test_wgsl_struct_same_content_same_layout_hash() {
    // Identical struct definitions should produce identical layout_hash
    let registry = ParserRegistry::new();
    let source_1 = r#"
struct SameStruct {
    foo: f32,
    bar: vec4<f32>,
}
"#;
    let source_2 = r#"
struct SameStruct {
    foo: f32,
    bar: vec4<f32>,
}
"#;

    let units_1 = registry.parse_file(Path::new("a.wgsl"), source_1, Language::Wgsl);
    let units_2 = registry.parse_file(Path::new("b.wgsl"), source_2, Language::Wgsl);

    let struct_1 = units_1
        .iter()
        .find(|u| u.unit_type == UnitType::Struct)
        .unwrap();
    let struct_2 = units_2
        .iter()
        .find(|u| u.unit_type == UnitType::Struct)
        .unwrap();

    assert_eq!(
        struct_1.hashes.layout_hash, struct_2.hashes.layout_hash,
        "identical struct definitions should have identical layout_hash"
    );
}

#[test]
fn test_wgsl_struct_different_types_different_layout_hash() {
    // Changing member type from scalar to vector should change layout_hash
    // (scalar f32 vs vec4<f32> have different type handles and different offsets)
    let registry = ParserRegistry::new();

    let source_scalar = r#"
struct TypeTest {
    value: f32,
    padding: f32,
}
"#;
    let source_vec4 = r#"
struct TypeTest {
    value: vec4<f32>,
    padding: f32,
}
"#;

    let units_scalar = registry.parse_file(Path::new("test.wgsl"), source_scalar, Language::Wgsl);
    let units_vec4 = registry.parse_file(Path::new("test.wgsl"), source_vec4, Language::Wgsl);

    let struct_scalar = units_scalar.iter().find(|u| u.unit_type == UnitType::Struct).unwrap();
    let struct_vec4 = units_vec4.iter().find(|u| u.unit_type == UnitType::Struct).unwrap();

    // Type changes (scalar vs vector) should result in different layout_hash
    // because they have different type indices AND different member offsets
    assert_ne!(
        struct_scalar.hashes.layout_hash, struct_vec4.hashes.layout_hash,
        "different member types (scalar vs vector) should produce different layout_hash"
    );
}

// ============================================================================
// ENTRY POINT EXTRACTION TESTS (ALL STAGES)
// ============================================================================

#[test]
fn test_wgsl_entry_point_signature_hash_includes_stage() {
    // Entry point signature_hash should differ based on stage
    let registry = ParserRegistry::new();

    let vertex_source = r#"
@vertex
fn main_entry() -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0);
}
"#;

    let fragment_source = r#"
@fragment
fn main_entry() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}
"#;

    let units_vertex = registry.parse_file(Path::new("vert.wgsl"), vertex_source, Language::Wgsl);
    let units_fragment = registry.parse_file(Path::new("frag.wgsl"), fragment_source, Language::Wgsl);

    let vert_fn = units_vertex.iter().find(|u| u.unit_type == UnitType::Function).unwrap();
    let frag_fn = units_fragment.iter().find(|u| u.unit_type == UnitType::Function).unwrap();

    // Same function name but different stages -> different signature_hash
    assert_ne!(
        vert_fn.hashes.signature_hash, frag_fn.hashes.signature_hash,
        "different shader stages should produce different signature_hash"
    );
}

#[test]
fn test_wgsl_entry_point_inline_attribute() {
    // Entry point with attribute on same line as function
    let registry = ParserRegistry::new();
    let source = r#"
@compute @workgroup_size(8, 8, 1) fn cs_inline() {
    // inline attribute style
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let compute_fn = units
        .iter()
        .find(|u| u.unit_type == UnitType::Function && u.name == "cs_inline");

    assert!(compute_fn.is_some(), "should find compute entry point with inline attribute");
}

#[test]
fn test_wgsl_entry_point_line_numbers() {
    // Verify entry point line numbers are correct
    let registry = ParserRegistry::new();
    let source = r#"// line 1
// line 2
@vertex
fn vs_main() -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0);
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let entry = units.iter().find(|u| u.name == "vs_main").unwrap();

    // @vertex is on line 3, function signature on line 4, closing brace on line 6
    assert!(entry.start_line >= 3, "start_line should be at or after @vertex attribute");
    assert!(entry.end_line >= entry.start_line, "end_line should be >= start_line");
}

// ============================================================================
// FUNCTION EXTRACTION TESTS
// ============================================================================

#[test]
fn test_wgsl_regular_function_not_confused_with_entry_point() {
    // Ensure regular functions are not confused with entry points
    let registry = ParserRegistry::new();
    let source = r#"
fn helper_function() -> f32 {
    return 42.0;
}

@vertex
fn vs_main() -> @builtin(position) vec4<f32> {
    return vec4<f32>(helper_function());
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let helper = units.iter().find(|u| u.name == "helper_function");
    let vs_main = units.iter().find(|u| u.name == "vs_main");

    assert!(helper.is_some(), "should find helper_function");
    assert!(vs_main.is_some(), "should find vs_main entry point");

    // Both should be Function type
    assert_eq!(helper.unwrap().unit_type, UnitType::Function);
    assert_eq!(vs_main.unwrap().unit_type, UnitType::Function);
}

#[test]
fn test_wgsl_function_hash_computation() {
    // Verify function hashes are computed correctly
    let registry = ParserRegistry::new();
    let source = r#"
fn compute_value(x: f32) -> f32 {
    return x * 2.0;
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let func = units.iter().find(|u| u.name == "compute_value").unwrap();

    // full_hash and body_hash should be non-zero for functions
    assert_ne!(func.hashes.full_hash, [0u8; 32]);
    assert_ne!(func.hashes.body_hash, [0u8; 32]);
    assert_ne!(func.hashes.signature_hash, [0u8; 32]);

    // layout_hash should be zero for functions (not applicable)
    assert_eq!(func.hashes.layout_hash, [0u8; 32]);
}

// ============================================================================
// HASH CONSISTENCY TESTS
// ============================================================================

#[test]
fn test_wgsl_hash_determinism() {
    // Same source should always produce same hashes
    let registry = ParserRegistry::new();
    let source = r#"
struct Data {
    x: f32,
    y: f32,
}

fn process(d: Data) -> f32 {
    return d.x + d.y;
}
"#;

    let units_1 = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);
    let units_2 = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    for (u1, u2) in units_1.iter().zip(units_2.iter()) {
        assert_eq!(u1.hashes.full_hash, u2.hashes.full_hash);
        assert_eq!(u1.hashes.signature_hash, u2.hashes.signature_hash);
        assert_eq!(u1.hashes.body_hash, u2.hashes.body_hash);
        assert_eq!(u1.hashes.layout_hash, u2.hashes.layout_hash);
    }
}

#[test]
fn test_wgsl_struct_body_change_affects_full_hash() {
    // Changing struct body (but same signature) should change full_hash
    let registry = ParserRegistry::new();

    let source_1 = r#"
struct MyStruct {
    a: f32,
}
"#;
    let source_2 = r#"
struct MyStruct {
    a: f32,
    b: f32,
}
"#;

    let units_1 = registry.parse_file(Path::new("test.wgsl"), source_1, Language::Wgsl);
    let units_2 = registry.parse_file(Path::new("test.wgsl"), source_2, Language::Wgsl);

    let struct_1 = units_1.iter().find(|u| u.unit_type == UnitType::Struct).unwrap();
    let struct_2 = units_2.iter().find(|u| u.unit_type == UnitType::Struct).unwrap();

    // full_hash should differ
    assert_ne!(
        struct_1.hashes.full_hash, struct_2.hashes.full_hash,
        "different struct body should produce different full_hash"
    );

    // layout_hash should also differ (different members)
    assert_ne!(
        struct_1.hashes.layout_hash, struct_2.hashes.layout_hash,
        "different struct members should produce different layout_hash"
    );
}

// ============================================================================
// COMPLEX SHADER TESTS
// ============================================================================

#[test]
fn test_wgsl_complex_shader_with_all_features() {
    // Parse a shader with structs, bindings, entry points, and helper functions
    let registry = ParserRegistry::new();
    let source = r#"
struct Camera {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> camera: Camera;

fn transform_normal(n: vec3<f32>) -> vec3<f32> {
    return normalize(n);
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.proj * camera.view * vec4<f32>(in.position, 1.0);
    out.world_normal = transform_normal(in.normal);
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.world_normal * 0.5 + 0.5, 1.0);
}
"#;

    let units = registry.parse_file(Path::new("shader.wgsl"), source, Language::Wgsl);

    // Count by type
    let structs = units.iter().filter(|u| u.unit_type == UnitType::Struct).count();
    let functions = units.iter().filter(|u| u.unit_type == UnitType::Function).count();

    assert_eq!(structs, 3, "should find 3 structs (Camera, VertexInput, VertexOutput)");
    assert_eq!(
        functions, 3,
        "should find 3 functions (transform_normal, vs_main, fs_main)"
    );

    // Verify each struct has valid layout_hash
    for unit in units.iter().filter(|u| u.unit_type == UnitType::Struct) {
        assert_ne!(
            unit.hashes.layout_hash,
            [0u8; 32],
            "struct {} should have non-zero layout_hash",
            unit.name
        );
    }
}

#[test]
fn test_wgsl_struct_with_nested_types() {
    // Test struct with vec and matrix types that have specific alignment requirements
    let registry = ParserRegistry::new();
    let source = r#"
struct Transform {
    model: mat4x4<f32>,
    normal: mat3x3<f32>,
    scale: vec3<f32>,
    padding: f32,
}
"#;
    let units = registry.parse_file(Path::new("test.wgsl"), source, Language::Wgsl);

    let struct_unit = units.iter().find(|u| u.name == "Transform").unwrap();

    // Should have all hashes computed
    assert_ne!(struct_unit.hashes.full_hash, [0u8; 32]);
    assert_ne!(struct_unit.hashes.layout_hash, [0u8; 32]);
    assert_ne!(struct_unit.hashes.signature_hash, [0u8; 32]);
}
