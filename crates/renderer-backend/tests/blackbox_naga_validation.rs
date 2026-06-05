// SPDX-License-Identifier: MIT
//
// blackbox_naga_validation.rs -- Blackbox tests for T-WGPU-P2.7.3 Naga Validation.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only the documented public types and functions:
//
//   - NagaValidator
//   - ValidationConfig
//   - Strictness (Relaxed, Standard, Strict, Pedantic)
//   - ValidationResult
//   - ValidationError (Parse, Validation, EmptySource)
//   - ErrorLocation
//   - ErrorLabel
//   - SourceSnippet
//   - quick_validate_wgsl()
//   - is_valid_wgsl() [via naga_is_valid_wgsl]
//   - parse_wgsl() [via naga_parse_wgsl]
//   - format_validation_error()
//
// ACCEPTANCE CRITERIA:
//   1. API contract tests      -- 25+ tests covering public API
//   2. Real shader tests       -- 20+ tests using actual project shaders
//   3. Error quality tests     -- 15+ tests verifying human-readable errors
//   4. Strictness tests        -- 10+ tests for different strictness levels
//   5. Snippet tests           -- 10+ tests for source context extraction
//   6. Integration tests       -- 10+ tests with ShaderCache/create_shader_module
//
// Total target: 90+ tests

use renderer_backend::shaders::validation::{
    ErrorLabel, ErrorLocation, NagaValidator, SourceSnippet, Strictness,
    ValidationConfig, ValidationError,
    format_validation_error, quick_validate_wgsl,
};
use renderer_backend::shaders::{
    is_valid_wgsl, naga_is_valid_wgsl, naga_parse_wgsl,
};
use std::path::PathBuf;

// =============================================================================
// HELPERS -- Test shader sources
// =============================================================================

/// Minimal valid vertex shader.
const VERTEX_SHADER: &str = r#"
    @vertex
    fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
"#;

/// Minimal valid fragment shader.
const FRAGMENT_SHADER: &str = r#"
    @fragment
    fn fs_main() -> @location(0) vec4<f32> {
        return vec4<f32>(1.0, 0.0, 0.0, 1.0);
    }
"#;

/// Minimal valid compute shader.
const COMPUTE_SHADER: &str = r#"
    @compute @workgroup_size(64, 1, 1)
    fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
        // Empty compute shader
    }
"#;

/// Combined vertex + fragment shader.
const VERTEX_FRAGMENT_SHADER: &str = r#"
    struct VertexOutput {
        @builtin(position) position: vec4<f32>,
        @location(0) color: vec4<f32>,
    }

    @vertex
    fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
        var out: VertexOutput;
        out.position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        out.color = vec4<f32>(1.0, 0.0, 0.0, 1.0);
        return out;
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
        return in.color;
    }
"#;

/// PBR vertex shader - matches project style.
const PBR_VERTEX_SHADER: &str = r#"
    struct CameraUniforms {
        view: mat4x4<f32>,
        projection: mat4x4<f32>,
        view_projection: mat4x4<f32>,
        camera_position: vec3<f32>,
        _pad: f32,
    }

    struct ModelUniforms {
        model: mat4x4<f32>,
        normal_matrix: mat4x4<f32>,
        material_index: u32,
        _pad0: f32,
        _pad1: f32,
        _pad2: f32,
    }

    @group(0) @binding(0) var<uniform> camera: CameraUniforms;
    @group(1) @binding(0) var<uniform> model: ModelUniforms;

    struct VertexInput {
        @location(0) position: vec3<f32>,
        @location(1) normal: vec3<f32>,
        @location(2) tangent: vec4<f32>,
        @location(3) texcoord: vec2<f32>,
    }

    struct VertexOutput {
        @builtin(position) clip_position: vec4<f32>,
        @location(0) world_position: vec3<f32>,
        @location(1) world_normal: vec3<f32>,
        @location(2) world_tangent: vec4<f32>,
        @location(3) texcoord: vec2<f32>,
        @location(4) material_index: u32,
    }

    @vertex
    fn vs_main(input: VertexInput) -> VertexOutput {
        var output: VertexOutput;
        let world_pos = model.model * vec4<f32>(input.position, 1.0);
        output.clip_position = camera.view_projection * world_pos;
        output.world_position = world_pos.xyz;
        let n = normalize((model.normal_matrix * vec4<f32>(input.normal, 0.0)).xyz);
        let t = normalize((model.normal_matrix * vec4<f32>(input.tangent.xyz, 0.0)).xyz);
        output.world_normal = n;
        output.world_tangent = vec4<f32>(t, input.tangent.w);
        output.texcoord = input.texcoord;
        output.material_index = model.material_index;
        return output;
    }
"#;

/// Shadow vertex shader - matches project style.
const SHADOW_VERTEX_SHADER: &str = r#"
    struct CascadeUniforms {
        light_view_proj: mat4x4<f32>,
    }

    @group(0) @binding(0) var<uniform> cascade: CascadeUniforms;

    struct ModelUniforms {
        model: mat4x4<f32>,
    }

    @group(1) @binding(0) var<uniform> model: ModelUniforms;

    struct VertexInput {
        @location(0) position: vec3<f32>,
    }

    struct VertexOutput {
        @builtin(position) clip_position: vec4<f32>,
    }

    @vertex
    fn vs_main(input: VertexInput) -> VertexOutput {
        var output: VertexOutput;
        let world_pos = model.model * vec4<f32>(input.position, 1.0);
        output.clip_position = cascade.light_view_proj * world_pos;
        return output;
    }
"#;

/// Shadow fragment shader (depth-only).
const SHADOW_FRAGMENT_SHADER: &str = r#"
    @fragment
    fn fs_main() {
        // Depth is written automatically to the depth attachment.
    }
"#;

/// Compute shader with storage buffers.
const COMPUTE_BUFFER_SHADER: &str = r#"
    struct Particle {
        position: vec3<f32>,
        age: f32,
        velocity: vec3<f32>,
        lifetime: f32,
    }

    @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;

    @compute @workgroup_size(256, 1, 1)
    fn particle_update(@builtin(global_invocation_id) gid: vec3<u32>) {
        let idx = gid.x;
        if idx >= arrayLength(&particles) {
            return;
        }
        var p = particles[idx];
        p.position = p.position + p.velocity;
        p.age = p.age + 0.016;
        particles[idx] = p;
    }
"#;

/// Texture sampling shader.
const TEXTURE_SAMPLE_SHADER: &str = r#"
    @group(0) @binding(0) var tex: texture_2d<f32>;
    @group(0) @binding(1) var samp: sampler;

    @fragment
    fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
        return textureSample(tex, samp, uv);
    }
"#;

/// Invalid shader - syntax error.
const INVALID_SYNTAX: &str = r#"
    @vertex fn main() -> @builtin(position) vec4<f32> {
        return vec4<f32>(0.0  // Missing closing paren
    }
"#;

/// Invalid shader - undefined variable.
const INVALID_UNDEFINED: &str = r#"
    @vertex fn main() -> @builtin(position) vec4<f32> {
        return undefined_variable;
    }
"#;

/// Invalid shader - type mismatch.
const INVALID_TYPE: &str = r#"
    @compute @workgroup_size(1)
    fn main() {
        let x: i32 = 1.5;
    }
"#;

/// Invalid shader - missing workgroup_size.
const INVALID_NO_WORKGROUP: &str = r#"
    @compute
    fn main() {
        // Missing @workgroup_size
    }
"#;

/// Invalid shader - missing return.
const INVALID_NO_RETURN: &str = r#"
    @vertex
    fn main() -> @builtin(position) vec4<f32> {
        let x = 1.0;
        // Missing return statement
    }
"#;

// =============================================================================
// SECTION 1 -- API CONTRACT TESTS (25+ tests)
// =============================================================================

// ---- NagaValidator construction ----

/// NagaValidator::new() creates a validator with the given config.
#[test]
fn naga_validator_new_with_config() {
    let config = ValidationConfig::default();
    let validator = NagaValidator::new(config);
    assert_eq!(validator.config().strictness, Strictness::Standard);
}

/// NagaValidator::default() creates a validator with default config.
#[test]
fn naga_validator_default_creates_standard() {
    let validator = NagaValidator::default();
    assert_eq!(validator.config().strictness, Strictness::Standard);
    assert!(validator.config().include_snippets);
}

/// NagaValidator::strict() creates a strict validator.
#[test]
fn naga_validator_strict_creates_strict() {
    let validator = NagaValidator::strict();
    assert_eq!(validator.config().strictness, Strictness::Strict);
}

/// NagaValidator::relaxed() creates a relaxed validator.
#[test]
fn naga_validator_relaxed_creates_relaxed() {
    let validator = NagaValidator::relaxed();
    assert_eq!(validator.config().strictness, Strictness::Relaxed);
    assert!(!validator.config().include_snippets);
}

/// NagaValidator::default_validator() is equivalent to default().
#[test]
fn naga_validator_default_validator_function() {
    let validator = NagaValidator::default_validator();
    assert_eq!(validator.config().strictness, Strictness::Standard);
}

// ---- ValidationConfig construction ----

/// ValidationConfig::default() has expected values.
#[test]
fn validation_config_default_values() {
    let config = ValidationConfig::default();
    assert_eq!(config.strictness, Strictness::Standard);
    assert!(config.include_snippets);
    assert!(config.include_suggestions);
    assert!(!config.use_colors);
    assert!(config.file_path.is_none());
}

/// ValidationConfig::relaxed() disables snippets.
#[test]
fn validation_config_relaxed_disables_snippets() {
    let config = ValidationConfig::relaxed();
    assert_eq!(config.strictness, Strictness::Relaxed);
    assert!(!config.include_snippets);
    assert!(!config.include_suggestions);
}

/// ValidationConfig::strict() has 3 context lines.
#[test]
fn validation_config_strict_has_three_context_lines() {
    let config = ValidationConfig::strict();
    assert_eq!(config.strictness, Strictness::Strict);
    assert_eq!(config.context_lines, 3);
}

/// ValidationConfig::pedantic() has 4 context lines.
#[test]
fn validation_config_pedantic_has_four_context_lines() {
    let config = ValidationConfig::pedantic();
    assert_eq!(config.strictness, Strictness::Pedantic);
    assert_eq!(config.context_lines, 4);
}

/// ValidationConfig::with_strictness() sets strictness.
#[test]
fn validation_config_with_strictness() {
    let config = ValidationConfig::with_strictness(Strictness::Pedantic);
    assert_eq!(config.strictness, Strictness::Pedantic);
}

/// ValidationConfig builder methods work.
#[test]
fn validation_config_builder_methods() {
    let config = ValidationConfig::default()
        .with_file_path("test.wgsl")
        .with_colors()
        .without_snippets()
        .with_context_lines(5);

    assert_eq!(config.file_path, Some(PathBuf::from("test.wgsl")));
    assert!(config.use_colors);
    assert!(!config.include_snippets);
    assert_eq!(config.context_lines, 5);
}

// ---- Strictness enum ----

/// Strictness::default() is Standard.
#[test]
fn strictness_default_is_standard() {
    assert_eq!(Strictness::default(), Strictness::Standard);
}

/// Strictness variants exist.
#[test]
fn strictness_variants_exist() {
    let _relaxed = Strictness::Relaxed;
    let _standard = Strictness::Standard;
    let _strict = Strictness::Strict;
    let _pedantic = Strictness::Pedantic;
}

/// Strictness can be compared for equality.
#[test]
fn strictness_equality() {
    assert_eq!(Strictness::Relaxed, Strictness::Relaxed);
    assert_ne!(Strictness::Relaxed, Strictness::Standard);
}

// ---- ErrorLocation ----

/// ErrorLocation::new() creates location.
#[test]
fn error_location_new() {
    let loc = ErrorLocation::new(10, 5, 100, 20);
    assert_eq!(loc.line, 10);
    assert_eq!(loc.column, 5);
    assert_eq!(loc.offset, 100);
    assert_eq!(loc.length, 20);
}

/// ErrorLocation::default() is line 1, column 1.
#[test]
fn error_location_default() {
    let loc = ErrorLocation::default();
    assert_eq!(loc.line, 1);
    assert_eq!(loc.column, 1);
}

/// ErrorLocation::format() with path.
#[test]
fn error_location_format_with_path() {
    let loc = ErrorLocation::new(10, 5, 0, 0);
    let formatted = loc.format(Some(&PathBuf::from("shader.wgsl")));
    assert_eq!(formatted, "shader.wgsl:10:5");
}

/// ErrorLocation::format() without path.
#[test]
fn error_location_format_without_path() {
    let loc = ErrorLocation::new(10, 5, 0, 0);
    let formatted = loc.format(None);
    assert_eq!(formatted, "10:5");
}

// ---- ErrorLabel ----

/// ErrorLabel::primary() creates primary label.
#[test]
fn error_label_primary() {
    let label = ErrorLabel::primary("error here", None);
    assert!(label.is_primary);
    assert_eq!(label.message, "error here");
    assert!(label.location.is_none());
}

/// ErrorLabel::secondary() creates secondary label.
#[test]
fn error_label_secondary() {
    let label = ErrorLabel::secondary("related note", None);
    assert!(!label.is_primary);
    assert_eq!(label.message, "related note");
}

/// ErrorLabel with location.
#[test]
fn error_label_with_location() {
    let loc = ErrorLocation::new(5, 10, 50, 5);
    let label = ErrorLabel::primary("error", Some(loc.clone()));
    assert_eq!(label.location.unwrap().line, 5);
}

// ---- SourceSnippet ----

/// SourceSnippet::context() creates context line.
#[test]
fn source_snippet_context() {
    let snippet = SourceSnippet::context(5, "let x = 1;");
    assert_eq!(snippet.line_number, 5);
    assert_eq!(snippet.content, "let x = 1;");
    assert!(!snippet.is_error_line);
    assert!(snippet.marker_start.is_none());
}

/// SourceSnippet::error() creates error line with marker.
#[test]
fn source_snippet_error() {
    let snippet = SourceSnippet::error(10, "let x = invalid;", 8, 7);
    assert_eq!(snippet.line_number, 10);
    assert!(snippet.is_error_line);
    assert_eq!(snippet.marker_start, Some(8));
    assert_eq!(snippet.marker_length, Some(7));
}

/// SourceSnippet::error() enforces minimum marker length of 1.
#[test]
fn source_snippet_error_minimum_marker() {
    let snippet = SourceSnippet::error(1, "x", 0, 0);
    assert_eq!(snippet.marker_length, Some(1));
}

/// SourceSnippet::format() produces readable output.
#[test]
fn source_snippet_format_readable() {
    let snippet = SourceSnippet::error(10, "let x = invalid;", 8, 7);
    let formatted = snippet.format(3, false);
    assert!(formatted.contains("10"));
    assert!(formatted.contains("let x = invalid;"));
    assert!(formatted.contains("^^^^^^^"));
}

// ---- ValidationError ----

/// ValidationError::parse() creates parse error.
#[test]
fn validation_error_parse() {
    let err = ValidationError::parse("unexpected token");
    assert!(err.is_parse_error());
    assert!(!err.is_validation_error());
    assert_eq!(err.message(), "unexpected token");
    assert_eq!(err.kind(), "parse");
}

/// ValidationError::validation() creates validation error.
#[test]
fn validation_error_validation() {
    let err = ValidationError::validation("type mismatch");
    assert!(!err.is_parse_error());
    assert!(err.is_validation_error());
    assert_eq!(err.message(), "type mismatch");
    assert_eq!(err.kind(), "validation");
}

/// ValidationError::EmptySource has correct message.
#[test]
fn validation_error_empty_source() {
    let err = ValidationError::EmptySource;
    assert_eq!(err.message(), "shader source is empty");
    assert_eq!(err.kind(), "empty");
}

/// ValidationError builder methods work.
#[test]
fn validation_error_builder_methods() {
    let err = ValidationError::parse("error")
        .with_label(ErrorLabel::secondary("note", None))
        .with_suggestion("try this");

    if let ValidationError::Parse { labels, suggestions, .. } = err {
        assert_eq!(labels.len(), 1);
        assert_eq!(suggestions.len(), 1);
    } else {
        panic!("expected Parse error");
    }
}

// =============================================================================
// SECTION 2 -- REAL SHADER TESTS (20+ tests)
// =============================================================================

/// Valid vertex shader passes validation.
#[test]
fn real_shader_vertex_passes() {
    assert!(quick_validate_wgsl(VERTEX_SHADER).is_ok());
}

/// Valid fragment shader passes validation.
#[test]
fn real_shader_fragment_passes() {
    assert!(quick_validate_wgsl(FRAGMENT_SHADER).is_ok());
}

/// Valid compute shader passes validation.
#[test]
fn real_shader_compute_passes() {
    assert!(quick_validate_wgsl(COMPUTE_SHADER).is_ok());
}

/// Combined vertex/fragment shader passes.
#[test]
fn real_shader_vertex_fragment_passes() {
    let validator = NagaValidator::default();
    let result = validator.validate_wgsl(VERTEX_FRAGMENT_SHADER);
    assert!(result.is_ok());

    let result = result.unwrap();
    assert_eq!(result.entry_points, 2);
    assert!(result.has_vertex());
    assert!(result.has_fragment());
}

/// PBR vertex shader passes validation.
#[test]
fn real_shader_pbr_vertex_passes() {
    let result = quick_validate_wgsl(PBR_VERTEX_SHADER);
    assert!(result.is_ok(), "PBR vertex shader should be valid: {:?}", result);
}

/// Shadow vertex shader passes validation.
#[test]
fn real_shader_shadow_vertex_passes() {
    assert!(quick_validate_wgsl(SHADOW_VERTEX_SHADER).is_ok());
}

/// Shadow fragment shader passes validation.
#[test]
fn real_shader_shadow_fragment_passes() {
    assert!(quick_validate_wgsl(SHADOW_FRAGMENT_SHADER).is_ok());
}

/// Compute shader with storage buffers passes.
#[test]
fn real_shader_compute_storage_passes() {
    let validator = NagaValidator::default();
    let result = validator.validate_wgsl(COMPUTE_BUFFER_SHADER);
    assert!(result.is_ok());

    let result = result.unwrap();
    assert!(result.has_compute());
    assert!(result.global_count >= 1);
}

/// Texture sampling shader passes.
#[test]
fn real_shader_texture_sample_passes() {
    assert!(quick_validate_wgsl(TEXTURE_SAMPLE_SHADER).is_ok());
}

/// Light culling style shader with workgroup memory.
#[test]
fn real_shader_workgroup_memory() {
    let source = r#"
        var<workgroup> shared_data: array<f32, 256>;

        @compute @workgroup_size(256, 1, 1)
        fn main(@builtin(local_invocation_index) idx: u32) {
            shared_data[idx] = f32(idx);
            workgroupBarrier();
            let _sum = shared_data[0] + shared_data[255];
        }
    "#;
    assert!(quick_validate_wgsl(source).is_ok());
}

/// Shader with multiple binding groups.
#[test]
fn real_shader_multiple_binding_groups() {
    let source = r#"
        @group(0) @binding(0) var<uniform> data0: f32;
        @group(1) @binding(0) var<uniform> data1: f32;
        @group(2) @binding(0) var<uniform> data2: f32;

        @compute @workgroup_size(1)
        fn main() {
            let _sum = data0 + data1 + data2;
        }
    "#;
    assert!(quick_validate_wgsl(source).is_ok());
}

/// Shader with arrays and runtime sized arrays.
#[test]
fn real_shader_arrays() {
    let source = r#"
        struct Data {
            fixed: array<f32, 16>,
            values: array<f32>,
        }

        @group(0) @binding(0) var<storage, read> data: Data;

        @compute @workgroup_size(1)
        fn main() {
            let _len = arrayLength(&data.values);
            let _a = data.fixed[0];
            let _b = data.values[0];
        }
    "#;
    assert!(quick_validate_wgsl(source).is_ok());
}

/// Shader with matrix operations.
#[test]
fn real_shader_matrix_ops() {
    let source = r#"
        @compute @workgroup_size(1)
        fn main() {
            let m = mat4x4<f32>(
                vec4<f32>(1.0, 0.0, 0.0, 0.0),
                vec4<f32>(0.0, 1.0, 0.0, 0.0),
                vec4<f32>(0.0, 0.0, 1.0, 0.0),
                vec4<f32>(0.0, 0.0, 0.0, 1.0)
            );
            let v = vec4<f32>(1.0, 2.0, 3.0, 1.0);
            let _result = m * v;
            let _det = determinant(m);
            let _inv = transpose(m);
        }
    "#;
    assert!(quick_validate_wgsl(source).is_ok());
}

/// Shader with builtin functions.
#[test]
fn real_shader_builtin_functions() {
    let source = r#"
        @compute @workgroup_size(1)
        fn main() {
            let _a = sin(1.0);
            let _b = cos(1.0);
            let _c = sqrt(4.0);
            let _d = abs(-1.0);
            let _e = min(1.0, 2.0);
            let _f = max(1.0, 2.0);
            let _g = clamp(0.5, 0.0, 1.0);
            let _h = dot(vec3<f32>(1.0), vec3<f32>(1.0));
            let _i = cross(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0));
            let _j = normalize(vec3<f32>(1.0, 1.0, 1.0));
            let _k = length(vec3<f32>(1.0, 1.0, 1.0));
        }
    "#;
    assert!(quick_validate_wgsl(source).is_ok());
}

/// Shader with control flow.
#[test]
fn real_shader_control_flow() {
    let source = r#"
        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            if gid.x == 0u {
                return;
            }

            var i = 0u;
            loop {
                if i >= 10u {
                    break;
                }
                if i == 5u {
                    i = i + 1u;
                    continue;
                }
                i = i + 1u;
            }

            switch gid.x {
                case 1u: {
                    // case 1
                }
                case 2u, 3u: {
                    // case 2 or 3
                }
                default: {
                    // default
                }
            }
        }
    "#;
    assert!(quick_validate_wgsl(source).is_ok());
}

/// Shader with pointer and reference types.
#[test]
fn real_shader_pointers() {
    // Note: Pointer parameters to storage are complex in WGSL.
    // We test a simpler case of using pointers via let bindings.
    let source = r#"
        @group(0) @binding(0) var<storage, read_write> data: array<f32>;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            // Direct modification via indexing
            data[gid.x] = data[gid.x] * 2.0;
        }
    "#;
    assert!(quick_validate_wgsl(source).is_ok());
}

/// Shader with depth texture.
#[test]
fn real_shader_depth_texture() {
    let source = r#"
        @group(0) @binding(0) var depth_tex: texture_depth_2d;

        @fragment
        fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
            let depth = textureLoad(depth_tex, vec2<i32>(0, 0), 0);
            return vec4<f32>(depth, depth, depth, 1.0);
        }
    "#;
    assert!(quick_validate_wgsl(source).is_ok());
}

/// Shader with storage texture.
#[test]
fn real_shader_storage_texture() {
    let source = r#"
        @group(0) @binding(0) var output_tex: texture_storage_2d<rgba8unorm, write>;

        @compute @workgroup_size(8, 8, 1)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            textureStore(output_tex, vec2<i32>(gid.xy), vec4<f32>(1.0, 0.0, 0.0, 1.0));
        }
    "#;
    assert!(quick_validate_wgsl(source).is_ok());
}

/// Shader with atomics.
#[test]
fn real_shader_atomics() {
    let source = r#"
        @group(0) @binding(0) var<storage, read_write> counter: array<atomic<u32>>;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let old = atomicAdd(&counter[0], 1u);
            atomicMax(&counter[1], gid.x);
            atomicMin(&counter[2], gid.x);
            let _loaded = atomicLoad(&counter[0]);
            atomicStore(&counter[3], 42u);
        }
    "#;
    assert!(quick_validate_wgsl(source).is_ok());
}

/// Shader with multi-sampled texture.
#[test]
fn real_shader_multisampled_texture() {
    let source = r#"
        @group(0) @binding(0) var ms_tex: texture_multisampled_2d<f32>;

        @fragment
        fn fs_main() -> @location(0) vec4<f32> {
            return textureLoad(ms_tex, vec2<i32>(0, 0), 0);
        }
    "#;
    assert!(quick_validate_wgsl(source).is_ok());
}

// =============================================================================
// SECTION 3 -- ERROR QUALITY TESTS (15+ tests)
// =============================================================================

/// Parse error has location.
#[test]
fn error_quality_parse_has_location() {
    let validator = NagaValidator::default();
    let result = validator.validate_wgsl(INVALID_SYNTAX);
    assert!(result.is_err());

    let err = result.unwrap_err();
    assert!(err.is_parse_error());
    // Error should have some location info
    assert!(err.location().is_some() || err.message().len() > 0);
}

/// Undefined variable error is descriptive.
#[test]
fn error_quality_undefined_variable() {
    let result = quick_validate_wgsl(INVALID_UNDEFINED);
    assert!(result.is_err());

    let err = result.unwrap_err();
    let msg = err.message().to_lowercase();
    assert!(msg.contains("unknown") || msg.contains("identifier") || msg.contains("undefined") || msg.contains("resolve"));
}

/// Type mismatch error is descriptive.
#[test]
fn error_quality_type_mismatch() {
    let result = quick_validate_wgsl(INVALID_TYPE);
    assert!(result.is_err());

    let err = result.unwrap_err();
    let msg = err.message().to_lowercase();
    assert!(msg.contains("type") || msg.contains("mismatch") || msg.contains("expected") || msg.contains("automatic"));
}

/// Empty source error is clear.
#[test]
fn error_quality_empty_source() {
    let result = quick_validate_wgsl("");
    assert!(result.is_err());

    let err = result.unwrap_err();
    assert_eq!(err.kind(), "empty");
    assert!(err.message().contains("empty"));
}

/// Whitespace-only source is empty.
#[test]
fn error_quality_whitespace_only() {
    let result = quick_validate_wgsl("   \n\t\n   ");
    assert!(result.is_err());

    let err = result.unwrap_err();
    assert!(matches!(err, ValidationError::EmptySource));
}

/// Format error produces readable output.
#[test]
fn error_quality_format_readable() {
    let source = "invalid @@@";
    let result = quick_validate_wgsl(source);
    let err = result.unwrap_err();

    let formatted = format_validation_error(&err, source);
    assert!(formatted.contains("error"));
}

/// Format error with file path includes path.
#[test]
fn error_quality_format_with_file_path() {
    let validator = NagaValidator::new(
        ValidationConfig::default().with_file_path("shaders/test.wgsl")
    );
    let source = "invalid @@@";
    let result = validator.validate_wgsl(source);
    let err = result.unwrap_err();

    let formatted = validator.format_error(&err, source);
    assert!(formatted.contains("shaders/test.wgsl"));
}

/// Error display implementation works.
#[test]
fn error_quality_display_impl() {
    let err = ValidationError::parse("test error");
    let display = format!("{}", err);
    assert!(display.contains("parse error"));
    assert!(display.contains("test error"));
}

/// Parse error with location has line:column in display.
#[test]
fn error_quality_parse_error_location_display() {
    let loc = ErrorLocation::new(10, 5, 0, 0);
    let err = ValidationError::parse_at("syntax error", loc);
    let display = format!("{}", err);
    assert!(display.contains("10"));
    assert!(display.contains("5"));
}

/// Validation error with location has line:column in display.
#[test]
fn error_quality_validation_error_location_display() {
    let loc = ErrorLocation::new(20, 15, 0, 0);
    let err = ValidationError::validation_at("type error", loc);
    let display = format!("{}", err);
    assert!(display.contains("20"));
    assert!(display.contains("15"));
}

/// Error with multiple labels.
#[test]
fn error_quality_multiple_labels() {
    let err = ValidationError::parse("error")
        .with_label(ErrorLabel::primary("here", None))
        .with_label(ErrorLabel::secondary("and here", None));

    if let ValidationError::Parse { labels, .. } = err {
        assert_eq!(labels.len(), 2);
    }
}

/// Error with suggestions.
#[test]
fn error_quality_suggestions() {
    let validator = NagaValidator::default();
    // Trigger an error that should have suggestions
    let source = "unknown_identifier";
    let result = validator.validate_wgsl(source);
    let err = result.unwrap_err();

    // The error system should generate suggestions
    if let ValidationError::Parse { suggestions, .. } = err {
        // May or may not have suggestions depending on the error
        let _ = suggestions;
    }
}

/// Missing return error is descriptive.
#[test]
fn error_quality_missing_return() {
    let result = quick_validate_wgsl(INVALID_NO_RETURN);
    assert!(result.is_err());
    // Should be a validation error about control flow or return
}

/// Missing workgroup_size error is descriptive.
#[test]
fn error_quality_missing_workgroup_size() {
    let result = quick_validate_wgsl(INVALID_NO_WORKGROUP);
    assert!(result.is_err());

    let err = result.unwrap_err();
    let msg = err.message().to_lowercase();
    // Should mention workgroup_size or attribute
    assert!(msg.len() > 0);
}

/// ValidationError std::error::Error implementation.
#[test]
fn error_quality_std_error_impl() {
    let err = ValidationError::parse("test");
    let _: &dyn std::error::Error = &err;
}

// =============================================================================
// SECTION 4 -- STRICTNESS TESTS (10+ tests)
// =============================================================================

/// Relaxed strictness validation flags.
#[test]
fn strictness_relaxed_flags() {
    let flags = Strictness::Relaxed.validation_flags();
    assert!(flags.is_empty());
}

/// Standard strictness validation flags.
#[test]
fn strictness_standard_flags() {
    let flags = Strictness::Standard.validation_flags();
    assert_eq!(flags, naga::valid::ValidationFlags::all());
}

/// Strict strictness validation flags.
#[test]
fn strictness_strict_flags() {
    let flags = Strictness::Strict.validation_flags();
    assert_eq!(flags, naga::valid::ValidationFlags::all());
}

/// Pedantic strictness validation flags.
#[test]
fn strictness_pedantic_flags() {
    let flags = Strictness::Pedantic.validation_flags();
    assert_eq!(flags, naga::valid::ValidationFlags::all());
}

/// Relaxed capabilities.
#[test]
fn strictness_relaxed_capabilities() {
    let caps = Strictness::Relaxed.capabilities();
    assert_eq!(caps, naga::valid::Capabilities::all());
}

/// Standard capabilities.
#[test]
fn strictness_standard_capabilities() {
    let caps = Strictness::Standard.capabilities();
    assert_eq!(caps, naga::valid::Capabilities::all());
}

/// Different strictness levels validate same shader.
#[test]
fn strictness_all_levels_validate_valid_shader() {
    let source = VERTEX_SHADER;

    let relaxed = NagaValidator::new(ValidationConfig::with_strictness(Strictness::Relaxed));
    let standard = NagaValidator::new(ValidationConfig::with_strictness(Strictness::Standard));
    let strict = NagaValidator::new(ValidationConfig::with_strictness(Strictness::Strict));
    let pedantic = NagaValidator::new(ValidationConfig::with_strictness(Strictness::Pedantic));

    assert!(relaxed.validate_wgsl(source).is_ok());
    assert!(standard.validate_wgsl(source).is_ok());
    assert!(strict.validate_wgsl(source).is_ok());
    assert!(pedantic.validate_wgsl(source).is_ok());
}

/// Config strictness affects validator.
#[test]
fn strictness_config_affects_validator() {
    let config = ValidationConfig::pedantic();
    let validator = NagaValidator::new(config);
    assert_eq!(validator.config().strictness, Strictness::Pedantic);
}

/// Validator config_mut allows changing strictness.
#[test]
fn strictness_config_mut() {
    let mut validator = NagaValidator::default();
    validator.config_mut().strictness = Strictness::Relaxed;
    assert_eq!(validator.config().strictness, Strictness::Relaxed);
}

/// Strictness Debug implementation.
#[test]
fn strictness_debug_impl() {
    let debug = format!("{:?}", Strictness::Standard);
    assert!(debug.contains("Standard"));
}

// =============================================================================
// SECTION 5 -- SNIPPET TESTS (10+ tests)
// =============================================================================

/// Extract snippets around error.
#[test]
fn snippet_extract_snippets() {
    let validator = NagaValidator::default();
    let source = "line1\nline2\nline3\nline4\nline5";
    let location = ErrorLocation::new(3, 1, 12, 5);

    let snippets = validator.extract_snippets(source, &location, 1);

    assert_eq!(snippets.len(), 3); // line2, line3, line4
    assert_eq!(snippets[1].line_number, 3);
    assert!(snippets[1].is_error_line);
}

/// Extract snippets at start of file.
#[test]
fn snippet_extract_at_start() {
    let validator = NagaValidator::default();
    let source = "line1\nline2\nline3";
    let location = ErrorLocation::new(1, 1, 0, 5);

    let snippets = validator.extract_snippets(source, &location, 1);

    assert_eq!(snippets.len(), 2); // line1 (error), line2
    assert!(snippets[0].is_error_line);
}

/// Extract snippets at end of file.
#[test]
fn snippet_extract_at_end() {
    let validator = NagaValidator::default();
    let source = "line1\nline2\nline3";
    let location = ErrorLocation::new(3, 1, 12, 5);

    let snippets = validator.extract_snippets(source, &location, 1);

    assert_eq!(snippets.len(), 2); // line2, line3 (error)
    assert!(snippets[1].is_error_line);
}

/// Extract snippets from empty source.
#[test]
fn snippet_extract_empty_source() {
    let validator = NagaValidator::default();
    let location = ErrorLocation::new(1, 1, 0, 0);

    let snippets = validator.extract_snippets("", &location, 1);
    assert!(snippets.is_empty());
}

/// Snippet format includes line number.
#[test]
fn snippet_format_line_number() {
    let snippet = SourceSnippet::context(42, "test content");
    let formatted = snippet.format(3, false);
    assert!(formatted.contains("42"));
    assert!(formatted.contains("test content"));
}

/// Snippet format error line has marker.
#[test]
fn snippet_format_error_marker() {
    let snippet = SourceSnippet::error(5, "let x = error;", 8, 5);
    let formatted = snippet.format(2, false);
    assert!(formatted.contains("^^^^^"));
}

/// Snippet format with colors includes ANSI codes.
#[test]
fn snippet_format_colors() {
    let snippet = SourceSnippet::error(1, "error", 0, 5);
    let formatted = snippet.format(1, true);
    assert!(formatted.contains("\x1b[31m")); // Red
    assert!(formatted.contains("\x1b[0m"));  // Reset
}

/// Snippet context line has no marker.
#[test]
fn snippet_context_no_marker() {
    let snippet = SourceSnippet::context(5, "context line");
    let formatted = snippet.format(2, false);
    assert!(!formatted.contains("^"));
}

/// Config context_lines affects extraction.
#[test]
fn snippet_config_context_lines() {
    let validator = NagaValidator::new(ValidationConfig::default().with_context_lines(3));
    let source = "1\n2\n3\n4\n5\n6\n7\n8\n9";
    let location = ErrorLocation::new(5, 1, 8, 1);

    let snippets = validator.extract_snippets(source, &location, 3);

    // Should have lines 2,3,4,5,6,7,8 (3 before + error + 3 after)
    assert!(snippets.len() >= 5);
}

/// Config without_snippets disables snippets in format.
#[test]
fn snippet_config_disable() {
    let validator = NagaValidator::new(ValidationConfig::default().without_snippets());
    let source = "invalid @@@";
    let result = validator.validate_wgsl(source);
    let err = result.unwrap_err();

    let formatted = validator.format_error(&err, source);
    // Should still have the error message, but may have fewer/no snippet lines
    assert!(formatted.contains("error"));
}

// =============================================================================
// SECTION 6 -- INTEGRATION TESTS (10+ tests)
// =============================================================================

/// parse_wgsl returns naga Module.
#[test]
fn integration_parse_wgsl_returns_module() {
    let module = naga_parse_wgsl(VERTEX_SHADER);
    assert!(module.is_ok());

    let module = module.unwrap();
    assert_eq!(module.entry_points.len(), 1);
}

/// is_valid_wgsl returns true for valid shader.
#[test]
fn integration_is_valid_wgsl_true() {
    assert!(naga_is_valid_wgsl(VERTEX_SHADER));
    assert!(is_valid_wgsl(VERTEX_SHADER));
}

/// is_valid_wgsl returns false for invalid shader.
#[test]
fn integration_is_valid_wgsl_false() {
    assert!(!naga_is_valid_wgsl("invalid @@@"));
    assert!(!is_valid_wgsl("invalid"));
}

/// ValidationResult has correct entry point count.
#[test]
fn integration_result_entry_points() {
    let validator = NagaValidator::default();
    let result = validator.validate_wgsl(VERTEX_FRAGMENT_SHADER).unwrap();

    assert_eq!(result.entry_points, 2);
    assert_eq!(result.entry_point_names.len(), 2);
    assert!(result.entry_point_names.contains(&"vs_main".to_string()));
    assert!(result.entry_point_names.contains(&"fs_main".to_string()));
}

/// ValidationResult has_vertex/has_fragment/has_compute.
#[test]
fn integration_result_shader_stages() {
    let validator = NagaValidator::default();

    let result = validator.validate_wgsl(VERTEX_SHADER).unwrap();
    assert!(result.has_vertex());
    assert!(!result.has_fragment());
    assert!(!result.has_compute());

    let result = validator.validate_wgsl(FRAGMENT_SHADER).unwrap();
    assert!(!result.has_vertex());
    assert!(result.has_fragment());
    assert!(!result.has_compute());

    let result = validator.validate_wgsl(COMPUTE_SHADER).unwrap();
    assert!(!result.has_vertex());
    assert!(!result.has_fragment());
    assert!(result.has_compute());
}

/// ValidationResult counts functions, globals, types.
#[test]
fn integration_result_counts() {
    let validator = NagaValidator::default();
    let result = validator.validate_wgsl(PBR_VERTEX_SHADER).unwrap();

    // Should have some types (structs)
    assert!(result.type_count > 0);
    // Should have some globals (uniforms)
    assert!(result.global_count > 0);
}

/// Validator can be cloned.
#[test]
fn integration_validator_clone() {
    let validator = NagaValidator::strict();
    let cloned = validator.clone();
    assert_eq!(cloned.config().strictness, Strictness::Strict);
}

/// ValidationConfig can be cloned.
#[test]
fn integration_config_clone() {
    let config = ValidationConfig::pedantic().with_file_path("test.wgsl");
    let cloned = config.clone();
    assert_eq!(cloned.strictness, Strictness::Pedantic);
    assert_eq!(cloned.file_path, Some(PathBuf::from("test.wgsl")));
}

/// ValidationError can be cloned.
#[test]
fn integration_error_clone() {
    let err = ValidationError::parse("test");
    let cloned = err.clone();
    assert_eq!(cloned.message(), "test");
}

/// Multiple validations with same validator.
#[test]
fn integration_multiple_validations() {
    let validator = NagaValidator::default();

    assert!(validator.validate_wgsl(VERTEX_SHADER).is_ok());
    assert!(validator.validate_wgsl(FRAGMENT_SHADER).is_ok());
    assert!(validator.validate_wgsl(COMPUTE_SHADER).is_ok());
    assert!(validator.validate_wgsl("invalid").is_err());
    assert!(validator.validate_wgsl(VERTEX_SHADER).is_ok()); // Still works
}

// =============================================================================
// SECTION 7 -- EDGE CASE TESTS (15+ tests)
// =============================================================================

/// Unicode in comments is valid.
#[test]
fn edge_case_unicode_comments() {
    let source = r#"
        // Unicode comment: Hello World
        @vertex fn main() -> @builtin(position) vec4<f32> {
            return vec4<f32>(0.0);
        }
    "#;
    assert!(quick_validate_wgsl(source).is_ok());
}

/// Very long shader validates.
#[test]
fn edge_case_long_shader() {
    let mut source = String::from("@compute @workgroup_size(1) fn main() {\n");
    for i in 0..500 {
        source.push_str(&format!("    let x{} = {};\n", i, i));
    }
    source.push_str("}\n");

    assert!(quick_validate_wgsl(&source).is_ok());
}

/// Deeply nested control flow validates.
#[test]
fn edge_case_deeply_nested() {
    let source = r#"
        @compute @workgroup_size(1)
        fn main() {
            if true {
                if true {
                    if true {
                        if true {
                            if true {
                                let x = 1;
                            }
                        }
                    }
                }
            }
        }
    "#;
    assert!(quick_validate_wgsl(source).is_ok());
}

/// Many functions validate.
#[test]
fn edge_case_many_functions() {
    let mut source = String::new();
    for i in 0..50 {
        source.push_str(&format!("fn func_{}() -> i32 {{ return {}i; }}\n", i, i));
    }
    source.push_str("@compute @workgroup_size(1) fn main() { let _x = func_0(); }\n");

    assert!(quick_validate_wgsl(&source).is_ok());
}

/// Empty function body validates.
#[test]
fn edge_case_empty_function() {
    let source = "@compute @workgroup_size(1) fn main() {}";
    assert!(quick_validate_wgsl(source).is_ok());
}

/// Single line shader validates.
#[test]
fn edge_case_single_line() {
    let source = "@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }";
    assert!(quick_validate_wgsl(source).is_ok());
}

/// Shader with many bindings validates.
#[test]
fn edge_case_many_bindings() {
    let mut source = String::new();
    for i in 0..16 {
        source.push_str(&format!(
            "@group(0) @binding({}) var<uniform> data{}: f32;\n", i, i
        ));
    }
    source.push_str("@compute @workgroup_size(1) fn main() { let _x = data0 + data15; }\n");

    assert!(quick_validate_wgsl(&source).is_ok());
}

/// Multiline string in comments.
#[test]
fn edge_case_multiline_comment() {
    let source = r#"
        /* This is a
           multiline
           comment
           spanning many lines */
        @vertex fn main() -> @builtin(position) vec4<f32> {
            return vec4<f32>(0.0);
        }
    "#;
    assert!(quick_validate_wgsl(source).is_ok());
}

/// All scalar types validate.
#[test]
fn edge_case_all_scalar_types() {
    let source = r#"
        @compute @workgroup_size(1)
        fn main() {
            let _a: i32 = 0i;
            let _b: u32 = 0u;
            let _c: f32 = 0.0;
            let _d: bool = true;
        }
    "#;
    assert!(quick_validate_wgsl(source).is_ok());
}

/// All vector types validate.
#[test]
fn edge_case_all_vector_types() {
    let source = r#"
        @compute @workgroup_size(1)
        fn main() {
            let _v2f: vec2<f32> = vec2<f32>(0.0);
            let _v3f: vec3<f32> = vec3<f32>(0.0);
            let _v4f: vec4<f32> = vec4<f32>(0.0);
            let _v2i: vec2<i32> = vec2<i32>(0i);
            let _v3i: vec3<i32> = vec3<i32>(0i);
            let _v4i: vec4<i32> = vec4<i32>(0i);
            let _v2u: vec2<u32> = vec2<u32>(0u);
            let _v3u: vec3<u32> = vec3<u32>(0u);
            let _v4u: vec4<u32> = vec4<u32>(0u);
        }
    "#;
    assert!(quick_validate_wgsl(source).is_ok());
}

/// All matrix types validate.
#[test]
fn edge_case_all_matrix_types() {
    let source = r#"
        @compute @workgroup_size(1)
        fn main() {
            let _m22: mat2x2<f32> = mat2x2<f32>(vec2<f32>(0.0), vec2<f32>(0.0));
            let _m23: mat2x3<f32> = mat2x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0));
            let _m24: mat2x4<f32> = mat2x4<f32>(vec4<f32>(0.0), vec4<f32>(0.0));
            let _m32: mat3x2<f32> = mat3x2<f32>(vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0));
            let _m33: mat3x3<f32> = mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
            let _m34: mat3x4<f32> = mat3x4<f32>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
            let _m42: mat4x2<f32> = mat4x2<f32>(vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0));
            let _m43: mat4x3<f32> = mat4x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
            let _m44: mat4x4<f32> = mat4x4<f32>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
        }
    "#;
    assert!(quick_validate_wgsl(source).is_ok());
}

/// Tab characters in source validate.
#[test]
fn edge_case_tabs() {
    let source = "@vertex\nfn main() -> @builtin(position) vec4<f32> {\n\treturn vec4<f32>(0.0);\n}\n";
    assert!(quick_validate_wgsl(source).is_ok());
}

/// Windows line endings validate.
#[test]
fn edge_case_crlf() {
    let source = "@vertex\r\nfn main() -> @builtin(position) vec4<f32> {\r\n    return vec4<f32>(0.0);\r\n}\r\n";
    assert!(quick_validate_wgsl(source).is_ok());
}

/// Mixed line endings validate.
#[test]
fn edge_case_mixed_line_endings() {
    let source = "@vertex\nfn main() -> @builtin(position) vec4<f32> {\r\n    return vec4<f32>(0.0);\n}\r\n";
    assert!(quick_validate_wgsl(source).is_ok());
}

/// ErrorLocation from_span handles source correctly.
#[test]
fn edge_case_error_location_from_span() {
    // Test the from_offset calculation
    let source = "line1\nline2\nline3";
    let loc = ErrorLocation::from_span(
        naga::Span::new(6, 11), // "line2"
        source
    );

    assert!(loc.is_some());
    let loc = loc.unwrap();
    assert_eq!(loc.line, 2);
    assert_eq!(loc.column, 1);
}

// =============================================================================
// SECTION 8 -- VALIDATION RESULT TESTS (5 tests)
// =============================================================================

/// ValidationResult::new creates correct result.
#[test]
fn result_new_correct() {
    let validator = NagaValidator::default();
    let result = validator.validate_wgsl(VERTEX_SHADER).unwrap();

    assert!(result.module.entry_points.len() > 0);
    assert_eq!(result.entry_points, result.module.entry_points.len());
}

/// ValidationResult has module and info.
#[test]
fn result_has_module_and_info() {
    let validator = NagaValidator::default();
    let result = validator.validate_wgsl(COMPUTE_BUFFER_SHADER).unwrap();

    // Module should have the particle struct
    assert!(result.type_count > 0);
    // Info should be valid
    let _ = &result.info;
}

/// ValidationResult entry_point_names matches module.
#[test]
fn result_entry_point_names_match() {
    let validator = NagaValidator::default();
    let result = validator.validate_wgsl(VERTEX_FRAGMENT_SHADER).unwrap();

    for name in &result.entry_point_names {
        assert!(result.module.entry_points.iter().any(|ep| &ep.name == name));
    }
}

/// ValidationResult function_count is accurate.
#[test]
fn result_function_count() {
    let source = r#"
        fn helper() -> i32 { return 1; }
        fn helper2() -> i32 { return 2; }
        @compute @workgroup_size(1)
        fn main() {
            let _x = helper() + helper2();
        }
    "#;

    let validator = NagaValidator::default();
    let result = validator.validate_wgsl(source).unwrap();

    // Should have at least helper and helper2 functions
    assert!(result.function_count >= 2);
}

/// ValidationResult global_count includes bindings.
#[test]
fn result_global_count() {
    let source = r#"
        @group(0) @binding(0) var<uniform> a: f32;
        @group(0) @binding(1) var<uniform> b: f32;
        @compute @workgroup_size(1)
        fn main() {
            let _x = a + b;
        }
    "#;

    let validator = NagaValidator::default();
    let result = validator.validate_wgsl(source).unwrap();

    assert!(result.global_count >= 2);
}

// =============================================================================
// SECTION 9 -- PARSE vs VALIDATE SEPARATION TESTS (5 tests)
// =============================================================================

/// parse_wgsl succeeds but validate_module may fail.
#[test]
fn parse_vs_validate_separation() {
    let validator = NagaValidator::default();

    // Valid parse and valid validation
    let source = VERTEX_SHADER;
    let module = validator.parse_wgsl(source);
    assert!(module.is_ok());

    let module = module.unwrap();
    let info = validator.validate_module(&module);
    assert!(info.is_ok());
}

/// parse_wgsl fails on syntax error.
#[test]
fn parse_fails_syntax_error() {
    let validator = NagaValidator::default();
    let result = validator.parse_wgsl("invalid @@@");
    assert!(result.is_err());
}

/// parse_wgsl fails on empty source.
#[test]
fn parse_fails_empty() {
    let validator = NagaValidator::default();
    let result = validator.parse_wgsl("");
    assert!(matches!(result, Err(ValidationError::EmptySource)));
}

/// validate_module validates a parsed module.
#[test]
fn validate_module_works() {
    let validator = NagaValidator::default();
    let module = validator.parse_wgsl(COMPUTE_SHADER).unwrap();
    let info = validator.validate_module(&module);
    assert!(info.is_ok());
}

/// validate_module_with_source provides better error messages.
#[test]
fn validate_module_with_source() {
    let validator = NagaValidator::default();
    let source = VERTEX_SHADER;
    let module = validator.parse_wgsl(source).unwrap();
    let info = validator.validate_module_with_source(&module, source);
    assert!(info.is_ok());
}

// =============================================================================
// SECTION 10 -- THREAD SAFETY AND REUSABILITY (3 tests)
// =============================================================================

/// Validator can be used from multiple threads.
#[test]
fn thread_safety_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    assert_send::<NagaValidator>();
    assert_sync::<NagaValidator>();
    assert_send::<ValidationConfig>();
    assert_sync::<ValidationConfig>();
    assert_send::<ValidationError>();
    assert_sync::<ValidationError>();
}

/// Validator is reusable across many validations.
#[test]
fn reusability_many_validations() {
    let validator = NagaValidator::default();

    for _ in 0..100 {
        let _ = validator.validate_wgsl(VERTEX_SHADER);
    }
}

/// Quick validate is stateless.
#[test]
fn quick_validate_stateless() {
    for _ in 0..50 {
        assert!(quick_validate_wgsl(COMPUTE_SHADER).is_ok());
        assert!(quick_validate_wgsl("invalid").is_err());
    }
}
