//! Integration tests for Build-Time DSL Compilation Pipeline (T-DEMO-5.5).
//!
//! Tests cover:
//! - Compiled shader embedding
//! - WGSL validation
//! - Shader module creation
//! - Demoscene module integration

use renderer_backend::demoscene;

// =============================================================================
// Static Shader Tests
// =============================================================================

#[test]
fn test_static_demo_shader_not_empty() {
    assert!(!demoscene::DEMO_SCENE_STATIC.is_empty());
}

#[test]
fn test_static_demo_shader_has_compute_entry() {
    assert!(
        demoscene::DEMO_SCENE_STATIC.contains("@compute"),
        "Static shader should have @compute entry point"
    );
}

#[test]
fn test_static_demo_shader_has_main() {
    assert!(
        demoscene::DEMO_SCENE_STATIC.contains("fn main("),
        "Static shader should have main() function"
    );
}

#[test]
fn test_static_demo_shader_has_workgroup_size() {
    assert!(
        demoscene::DEMO_SCENE_STATIC.contains("@workgroup_size"),
        "Static shader should have workgroup_size attribute"
    );
}

// =============================================================================
// Shader Getter Tests
// =============================================================================

#[test]
fn test_get_demo_scene_shader_returns_valid() {
    let shader = demoscene::get_demo_scene_shader();
    assert!(!shader.is_empty());
}

#[test]
fn test_get_demo_scene_shader_has_scene_sdf() {
    let shader = demoscene::get_demo_scene_shader();
    // Should have scene_sdf or sd_scene
    assert!(
        shader.contains("fn scene_sdf(") || shader.contains("fn sd_scene"),
        "Shader should have scene SDF function"
    );
}

#[test]
fn test_get_demo_scene_shader_has_compute() {
    let shader = demoscene::get_demo_scene_shader();
    assert!(
        shader.contains("@compute"),
        "Shader should have @compute entry point"
    );
}

// =============================================================================
// Validation Function Tests
// =============================================================================

#[test]
fn test_validate_valid_shader() {
    let valid = r#"
        @compute @workgroup_size(8, 8, 1)
        fn main() {}
        fn scene_sdf(p: vec3<f32>) -> vec2<f32> { return vec2(0.0); }
        fn scene_material(id: u32) -> Material { return Material(); }
    "#;
    assert!(demoscene::validate_demoscene_shader(valid).is_ok());
}

#[test]
fn test_validate_missing_compute() {
    let invalid = r#"
        fn main() {}
        fn scene_sdf(p: vec3<f32>) -> vec2<f32> { return vec2(0.0); }
        fn scene_material(id: u32) -> Material { return Material(); }
    "#;
    let result = demoscene::validate_demoscene_shader(invalid);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("@compute"));
}

#[test]
fn test_validate_missing_main() {
    let invalid = r#"
        @compute @workgroup_size(8, 8, 1)
        fn scene_sdf(p: vec3<f32>) -> vec2<f32> { return vec2(0.0); }
        fn scene_material(id: u32) -> Material { return Material(); }
    "#;
    let result = demoscene::validate_demoscene_shader(invalid);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("main"));
}

#[test]
fn test_validate_missing_scene_sdf() {
    let invalid = r#"
        @compute @workgroup_size(8, 8, 1)
        fn main() {}
        fn scene_material(id: u32) -> Material { return Material(); }
    "#;
    let result = demoscene::validate_demoscene_shader(invalid);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("scene_sdf"));
}

#[test]
fn test_validate_missing_scene_material() {
    let invalid = r#"
        @compute @workgroup_size(8, 8, 1)
        fn main() {}
        fn scene_sdf(p: vec3<f32>) -> vec2<f32> { return vec2(0.0); }
    "#;
    let result = demoscene::validate_demoscene_shader(invalid);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("scene_material"));
}

#[test]
fn test_validate_unbalanced_braces() {
    let invalid = r#"
        @compute @workgroup_size(8, 8, 1)
        fn main() {
        fn scene_sdf(p: vec3<f32>) -> vec2<f32> { return vec2(0.0); }
        fn scene_material(id: u32) -> Material { return Material(); }
    "#;
    let result = demoscene::validate_demoscene_shader(invalid);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("braces"));
}

#[test]
fn test_validate_unbalanced_parens() {
    let invalid = r#"
        @compute @workgroup_size(8, 8, 1)
        fn main() {}
        fn scene_sdf(p: vec3<f32> -> vec2<f32> { return vec2(0.0); }
        fn scene_material(id: u32) -> Material { return Material(); }
    "#;
    let result = demoscene::validate_demoscene_shader(invalid);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("parentheses"));
}

#[test]
fn test_validate_static_shader() {
    // The static shader should pass validation
    let result = demoscene::validate_demoscene_shader(demoscene::DEMO_SCENE_STATIC);
    assert!(
        result.is_ok(),
        "Static shader should be valid: {:?}",
        result.err()
    );
}

// =============================================================================
// Naga Parsing Tests (using dev-dependency)
// =============================================================================

#[test]
fn test_static_shader_parses_with_naga() {
    use naga::front::wgsl;

    let shader = demoscene::DEMO_SCENE_STATIC;
    let result = wgsl::parse_str(shader);

    assert!(
        result.is_ok(),
        "Static shader should parse with naga: {:?}",
        result.err()
    );
}

#[test]
fn test_static_shader_has_entry_point_in_naga() {
    use naga::front::wgsl;

    let shader = demoscene::DEMO_SCENE_STATIC;
    let module = wgsl::parse_str(shader).expect("Failed to parse shader");

    // Should have at least one entry point
    assert!(
        !module.entry_points.is_empty(),
        "Shader should have entry points"
    );

    // Find the compute entry point
    let compute_entry = module
        .entry_points
        .iter()
        .find(|ep| ep.stage == naga::ShaderStage::Compute);

    assert!(
        compute_entry.is_some(),
        "Shader should have a compute entry point"
    );
}

#[test]
fn test_static_shader_has_scene_sdf_function() {
    use naga::front::wgsl;

    let shader = demoscene::DEMO_SCENE_STATIC;
    let module = wgsl::parse_str(shader).expect("Failed to parse shader");

    // Look for scene_sdf or sd_scene function
    let has_scene_sdf = module.functions.iter().any(|(_, func)| {
        func.name
            .as_ref()
            .map(|n| n.contains("scene") && n.contains("sdf"))
            .unwrap_or(false)
            || func
                .name
                .as_ref()
                .map(|n| n.starts_with("sd_scene"))
                .unwrap_or(false)
    });

    assert!(has_scene_sdf, "Shader should have scene SDF function");
}

// =============================================================================
// Noise Shader Tests (existing shaders)
// =============================================================================

#[test]
fn test_noise_shaders_load() {
    assert!(!demoscene::NOISE_HASH.is_empty());
    assert!(!demoscene::NOISE_VALUE.is_empty());
    assert!(!demoscene::NOISE_PERLIN.is_empty());
    assert!(!demoscene::NOISE_FBM.is_empty());
    assert!(!demoscene::NOISE_RIDGED.is_empty());
    assert!(!demoscene::NOISE_DOMAIN_WARP.is_empty());
}

#[test]
fn test_sdf_shaders_load() {
    assert!(!demoscene::SDF_DOMAIN.is_empty());
    assert!(!demoscene::SDF_PRIMITIVES.is_empty());
    assert!(!demoscene::SDF_COMBINATORS.is_empty());
}

// =============================================================================
// Shader Content Tests
// =============================================================================

#[test]
fn test_static_shader_has_uniforms() {
    let shader = demoscene::DEMO_SCENE_STATIC;
    assert!(
        shader.contains("Uniforms") || shader.contains("uniforms"),
        "Shader should have uniforms"
    );
}

#[test]
fn test_static_shader_has_material_struct() {
    let shader = demoscene::DEMO_SCENE_STATIC;
    assert!(
        shader.contains("struct Material") || shader.contains("Material"),
        "Shader should have Material struct"
    );
}

#[test]
fn test_static_shader_has_ray_marching() {
    let shader = demoscene::DEMO_SCENE_STATIC;
    // Should have ray marching related code
    let has_march = shader.contains("march") || shader.contains("March");
    let has_ray = shader.contains("ray") || shader.contains("Ray");
    assert!(
        has_march || has_ray,
        "Shader should have ray marching code"
    );
}

#[test]
fn test_static_shader_has_lighting() {
    let shader = demoscene::DEMO_SCENE_STATIC;
    let has_lighting =
        shader.contains("lighting") || shader.contains("Lighting") || shader.contains("light");
    assert!(has_lighting, "Shader should have lighting code");
}

#[test]
fn test_static_shader_has_normal_estimation() {
    let shader = demoscene::DEMO_SCENE_STATIC;
    let has_normal =
        shader.contains("normal") || shader.contains("Normal") || shader.contains("estimate");
    assert!(has_normal, "Shader should have normal estimation");
}

// =============================================================================
// Bootstrap Integration Tests
// =============================================================================

#[test]
fn test_bootstrap_exports() {
    // Just verify we can access the bootstrap module types
    fn _check_types() {
        // BootstrapError is accessible
        let _: fn(wgpu::RequestDeviceError) -> demoscene::BootstrapError =
            demoscene::BootstrapError::DeviceFailed;

        // DemoBootstrap and DemoWindow types are accessible
        // (can't construct without GPU, but types compile)
    }
}
