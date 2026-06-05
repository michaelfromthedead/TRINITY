// blackbox_shadow_filter_esm.rs -- Blackbox tests for shadow_filter_esm.wgsl.
//
// This test file verifies the WGSL Exponential Shadow Map filter is structurally valid.
// Cleanroom: tests are based ONLY on the spec definition (T-LIT-6.7:
// "ESM with configurable exponential constant and pre-filter blur").
//
// Coverage:
//   - naga WGSL compilation (full file)
//   - Core ESM functions compilation
//   - Blur compute shader entry points
//   - Function existence verification
//   - Parameter structure validation

use naga::front::wgsl::parse_str;

/// The full WGSL source file, baked in at compile time via include_str!.
static WGSL_SOURCE: &str = include_str!("../shaders/shadow_filter_esm.wgsl");

// =============================================================================
// Section 1: naga compilation
// =============================================================================

/// The full shadow_filter_esm.wgsl file must compile through naga without errors.
/// This is the primary acceptance check for structural WGSL validity.
#[test]
fn compiles_via_naga() {
    let result = parse_str(WGSL_SOURCE);
    match result {
        Ok(_) => {}
        Err(err) => {
            panic!("shadow_filter_esm.wgsl failed to parse via naga:\n{:#?}", err);
        }
    }
}

/// The compute shader entry points must be present in the parsed module.
#[test]
fn compute_entry_points_exist() {
    let result = parse_str(WGSL_SOURCE);
    match result {
        Ok(module) => {
            let entry_names: Vec<_> = module
                .entry_points
                .iter()
                .map(|ep| ep.name.as_str())
                .collect();

            assert!(
                entry_names.contains(&"esm_blur_horizontal"),
                "esm_blur_horizontal compute entry point not found. Found: {:?}",
                entry_names
            );
            assert!(
                entry_names.contains(&"esm_blur_vertical"),
                "esm_blur_vertical compute entry point not found. Found: {:?}",
                entry_names
            );
        }
        Err(err) => {
            panic!("shadow_filter_esm.wgsl failed to parse via naga:\n{:#?}", err);
        }
    }
}

/// The fragment shader entry points must be present in the parsed module.
#[test]
fn fragment_entry_points_exist() {
    let result = parse_str(WGSL_SOURCE);
    match result {
        Ok(module) => {
            let entry_names: Vec<_> = module
                .entry_points
                .iter()
                .map(|ep| ep.name.as_str())
                .collect();

            assert!(
                entry_names.contains(&"esm_generate"),
                "esm_generate fragment entry point not found. Found: {:?}",
                entry_names
            );
            assert!(
                entry_names.contains(&"esm_generate_scaled"),
                "esm_generate_scaled fragment entry point not found. Found: {:?}",
                entry_names
            );
            assert!(
                entry_names.contains(&"esm_generate_warped"),
                "esm_generate_warped fragment entry point not found. Found: {:?}",
                entry_names
            );
        }
        Err(err) => {
            panic!("shadow_filter_esm.wgsl failed to parse via naga:\n{:#?}", err);
        }
    }
}

// =============================================================================
// Section 2: Function existence
// =============================================================================

/// esm_depth_encode must exist in the source.
#[test]
fn esm_depth_encode_exists() {
    assert!(
        WGSL_SOURCE.contains("fn esm_depth_encode("),
        "esm_depth_encode function not found in shadow_filter_esm.wgsl"
    );
}

/// esm_depth_decode must exist in the source.
#[test]
fn esm_depth_decode_exists() {
    assert!(
        WGSL_SOURCE.contains("fn esm_depth_decode("),
        "esm_depth_decode function not found in shadow_filter_esm.wgsl"
    );
}

/// esm_shadow must exist in the source.
#[test]
fn esm_shadow_exists() {
    assert!(
        WGSL_SOURCE.contains("fn esm_shadow("),
        "esm_shadow function not found in shadow_filter_esm.wgsl"
    );
}

/// esm_shadow_hybrid must exist in the source.
#[test]
fn esm_shadow_hybrid_exists() {
    assert!(
        WGSL_SOURCE.contains("fn esm_shadow_hybrid("),
        "esm_shadow_hybrid function not found in shadow_filter_esm.wgsl"
    );
}

/// esm_shadow_hybrid_pcf3x3 must exist in the source.
#[test]
fn esm_shadow_hybrid_pcf3x3_exists() {
    assert!(
        WGSL_SOURCE.contains("fn esm_shadow_hybrid_pcf3x3("),
        "esm_shadow_hybrid_pcf3x3 function not found in shadow_filter_esm.wgsl"
    );
}

/// esm_warp must exist in the source.
#[test]
fn esm_warp_exists() {
    assert!(
        WGSL_SOURCE.contains("fn esm_warp("),
        "esm_warp function not found in shadow_filter_esm.wgsl"
    );
}

/// esm_unwarp must exist in the source.
#[test]
fn esm_unwarp_exists() {
    assert!(
        WGSL_SOURCE.contains("fn esm_unwarp("),
        "esm_unwarp function not found in shadow_filter_esm.wgsl"
    );
}

/// gaussian_weight must exist in the source.
#[test]
fn gaussian_weight_exists() {
    assert!(
        WGSL_SOURCE.contains("fn gaussian_weight("),
        "gaussian_weight function not found in shadow_filter_esm.wgsl"
    );
}

// =============================================================================
// Section 3: Structure definitions
// =============================================================================

/// EsmParams struct must exist in the source.
#[test]
fn esm_params_struct_exists() {
    assert!(
        WGSL_SOURCE.contains("struct EsmParams"),
        "EsmParams struct not found in shadow_filter_esm.wgsl"
    );
}

/// EsmParamsExtended struct must exist in the source.
#[test]
fn esm_params_extended_struct_exists() {
    assert!(
        WGSL_SOURCE.contains("struct EsmParamsExtended"),
        "EsmParamsExtended struct not found in shadow_filter_esm.wgsl"
    );
}

/// BlurConfig struct must exist in the source.
#[test]
fn blur_config_struct_exists() {
    assert!(
        WGSL_SOURCE.contains("struct BlurConfig"),
        "BlurConfig struct not found in shadow_filter_esm.wgsl"
    );
}

/// ShadowMapInfo struct must exist in the source.
#[test]
fn shadow_map_info_struct_exists() {
    assert!(
        WGSL_SOURCE.contains("struct ShadowMapInfo"),
        "ShadowMapInfo struct not found in shadow_filter_esm.wgsl"
    );
}

// =============================================================================
// Section 4: Parameter validation
// =============================================================================

/// EsmParams must contain exponent_c field.
#[test]
fn esm_params_has_exponent_c() {
    assert!(
        WGSL_SOURCE.contains("exponent_c: f32"),
        "exponent_c field not found in EsmParams"
    );
}

/// EsmParams must contain depth_scale field.
#[test]
fn esm_params_has_depth_scale() {
    assert!(
        WGSL_SOURCE.contains("depth_scale: f32"),
        "depth_scale field not found in EsmParams"
    );
}

/// EsmParams must contain filter_radius field.
#[test]
fn esm_params_has_filter_radius() {
    assert!(
        WGSL_SOURCE.contains("filter_radius: f32"),
        "filter_radius field not found in EsmParams"
    );
}

// =============================================================================
// Section 5: Constant definitions
// =============================================================================

/// WORKGROUP_SIZE constant must be defined.
#[test]
fn workgroup_size_constant_exists() {
    assert!(
        WGSL_SOURCE.contains("const WORKGROUP_SIZE"),
        "WORKGROUP_SIZE constant not found in shadow_filter_esm.wgsl"
    );
}

/// ESM_MIN_VALUE constant must be defined.
#[test]
fn esm_min_value_constant_exists() {
    assert!(
        WGSL_SOURCE.contains("const ESM_MIN_VALUE"),
        "ESM_MIN_VALUE constant not found in shadow_filter_esm.wgsl"
    );
}

/// MAX_BLUR_RADIUS constant must be defined.
#[test]
fn max_blur_radius_constant_exists() {
    assert!(
        WGSL_SOURCE.contains("const MAX_BLUR_RADIUS"),
        "MAX_BLUR_RADIUS constant not found in shadow_filter_esm.wgsl"
    );
}

// =============================================================================
// Section 6: Binding declarations
// =============================================================================

/// Uniform bindings must be declared for compute shaders.
#[test]
fn uniform_bindings_declared() {
    assert!(
        WGSL_SOURCE.contains("@group(0) @binding(0)"),
        "Group 0 binding 0 not found in shadow_filter_esm.wgsl"
    );
    assert!(
        WGSL_SOURCE.contains("@group(0) @binding(1)"),
        "Group 0 binding 1 not found in shadow_filter_esm.wgsl"
    );
    assert!(
        WGSL_SOURCE.contains("@group(0) @binding(2)"),
        "Group 0 binding 2 not found in shadow_filter_esm.wgsl"
    );
    assert!(
        WGSL_SOURCE.contains("@group(0) @binding(3)"),
        "Group 0 binding 3 not found in shadow_filter_esm.wgsl"
    );
    assert!(
        WGSL_SOURCE.contains("@group(0) @binding(4)"),
        "Group 0 binding 4 not found in shadow_filter_esm.wgsl"
    );
}

/// Compute shader workgroup_size must be declared.
#[test]
fn compute_workgroup_size_declared() {
    assert!(
        WGSL_SOURCE.contains("@compute @workgroup_size(8, 8, 1)"),
        "Compute workgroup_size(8, 8, 1) attribute not found"
    );
}

// =============================================================================
// Section 7: naga function signature verification
// =============================================================================

/// esm_depth_encode must have the correct naga-level function signature:
/// fn esm_depth_encode(depth: f32, c: f32) -> f32
#[test]
fn esm_depth_encode_naga_signature() {
    let module = parse_str(WGSL_SOURCE)
        .expect("shadow_filter_esm.wgsl must parse via naga for signature verification");

    let func = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some("esm_depth_encode"))
        .map(|(_, f)| f)
        .expect("esm_depth_encode function not found in parsed module");

    // Should have 2 arguments
    assert_eq!(
        func.arguments.len(),
        2,
        "esm_depth_encode should have 2 arguments, found {}",
        func.arguments.len()
    );

    // Should return f32
    assert!(
        func.result.is_some(),
        "esm_depth_encode should have a return type"
    );
}

/// esm_shadow must have texture and sampler parameters.
#[test]
fn esm_shadow_naga_signature() {
    let module = parse_str(WGSL_SOURCE)
        .expect("shadow_filter_esm.wgsl must parse via naga for signature verification");

    let func = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some("esm_shadow"))
        .map(|(_, f)| f)
        .expect("esm_shadow function not found in parsed module");

    // Should have 5 arguments: shadow_map, shadow_sampler, uv, receiver_depth, params
    assert_eq!(
        func.arguments.len(),
        5,
        "esm_shadow should have 5 arguments, found {}",
        func.arguments.len()
    );

    // Should return f32 (shadow factor)
    assert!(
        func.result.is_some(),
        "esm_shadow should have a return type"
    );
}

/// gaussian_weight must have the correct signature.
#[test]
fn gaussian_weight_naga_signature() {
    let module = parse_str(WGSL_SOURCE)
        .expect("shadow_filter_esm.wgsl must parse via naga for signature verification");

    let func = module
        .functions
        .iter()
        .find(|(_, f)| f.name.as_deref() == Some("gaussian_weight"))
        .map(|(_, f)| f)
        .expect("gaussian_weight function not found in parsed module");

    // Should have 2 arguments: offset, sigma
    assert_eq!(
        func.arguments.len(),
        2,
        "gaussian_weight should have 2 arguments, found {}",
        func.arguments.len()
    );
}

// =============================================================================
// Section 8: Documentation validation
// =============================================================================

/// The file must contain quantization notes for R16F/R32F format selection.
#[test]
fn has_quantization_documentation() {
    assert!(
        WGSL_SOURCE.contains("R32F") && WGSL_SOURCE.contains("R16F"),
        "Quantization documentation for R32F/R16F not found"
    );
}

/// The file must document the ESM theory.
#[test]
fn has_esm_theory_documentation() {
    assert!(
        WGSL_SOURCE.contains("ESM Theory"),
        "ESM Theory documentation not found"
    );
}

/// The file must document the recommended exponent range.
#[test]
fn has_exponent_range_documentation() {
    assert!(
        WGSL_SOURCE.contains("32-128") || WGSL_SOURCE.contains("32 to 128"),
        "Exponent range documentation (32-128) not found"
    );
}

// =============================================================================
// Section 9: Debug utilities
// =============================================================================

/// esm_debug_visualize must exist.
#[test]
fn esm_debug_visualize_exists() {
    assert!(
        WGSL_SOURCE.contains("fn esm_debug_visualize("),
        "esm_debug_visualize function not found"
    );
}

/// esm_debug_quantization must exist.
#[test]
fn esm_debug_quantization_exists() {
    assert!(
        WGSL_SOURCE.contains("fn esm_debug_quantization("),
        "esm_debug_quantization function not found"
    );
}

// =============================================================================
// Section 10: Utility functions
// =============================================================================

/// clamp_esm_exponent must exist.
#[test]
fn clamp_esm_exponent_exists() {
    assert!(
        WGSL_SOURCE.contains("fn clamp_esm_exponent("),
        "clamp_esm_exponent function not found"
    );
}

/// estimate_esm_exponent must exist.
#[test]
fn estimate_esm_exponent_exists() {
    assert!(
        WGSL_SOURCE.contains("fn estimate_esm_exponent("),
        "estimate_esm_exponent function not found"
    );
}
