// SPDX-License-Identifier: MIT
//
// blackbox_depth_bias.rs -- Blackbox tests for T-WGPU-P3.4.2 Depth Bias.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only the documented public types and functions:
//
//   - DepthBias -- Core depth bias configuration struct
//   - DepthBiasBuilder -- Fluent builder for depth bias creation
//   - DepthBiasError -- Error type for validation failures
//   - DepthBiasInfo -- Metadata about depth bias presets
//   - DEPTH_BIAS_PRESETS -- Array of preset configurations
//   - get_depth_bias_info -- Lookup preset info by name
//   - get_depth_bias_preset -- Get preset by name
//   - depth_bias_preset_names -- Iterator over preset names
//
// ACCEPTANCE CRITERIA:
//   1. constant: i32 -- Fixed depth offset in depth buffer units
//   2. slope_scale: f32 -- Slope-based scaling factor
//   3. clamp: f32 -- Maximum clamp value
//   4. Shadow map preset -- Verify shadow_map() preset values
//
// Additional test categories:
//   5. Polygon offset preset
//   6. Default (none) preset
//   7. Builder API (fluent interface)
//   8. Real-world scenarios
//   9. wgpu conversion (Into<DepthBiasState>)
//   10. Validation and error handling
//
// Total target: 50+ tests across 10 categories

use renderer_backend::render_pipeline::{
    depth_bias_preset_names, get_depth_bias_info, get_depth_bias_preset, DepthBias,
    DepthBiasBuilder, DepthBiasError, DepthBiasInfo, DEPTH_BIAS_PRESETS,
};

// =============================================================================
// CATEGORY 1: API SURFACE TESTS
// =============================================================================
// Verify all public types and functions are accessible from the blackbox.

#[test]
fn test_api_depth_bias_accessible() {
    let bias = DepthBias::new();
    assert_eq!(bias.constant, 0);
}

#[test]
fn test_api_depth_bias_builder_accessible() {
    let builder = DepthBiasBuilder::new();
    let bias = builder.build().expect("Build should succeed");
    assert!(bias.is_none());
}

#[test]
fn test_api_depth_bias_error_accessible() {
    let bias = DepthBias::new().clamp(-1.0);
    let result = bias.validate();
    assert!(matches!(result, Err(DepthBiasError::NegativeClamp(_))));
}

#[test]
fn test_api_depth_bias_info_accessible() {
    let info = get_depth_bias_info("Shadow Map");
    assert!(info.is_some());
}

#[test]
fn test_api_depth_bias_presets_accessible() {
    assert!(!DEPTH_BIAS_PRESETS.is_empty());
    assert_eq!(DEPTH_BIAS_PRESETS.len(), 7);
}

#[test]
fn test_api_get_preset_accessible() {
    let preset = get_depth_bias_preset("Shadow Map");
    assert!(preset.is_some());
}

#[test]
fn test_api_preset_names_accessible() {
    let names: Vec<_> = depth_bias_preset_names().collect();
    assert!(!names.is_empty());
}

// =============================================================================
// CATEGORY 2: CONSTANT FIELD (i32)
// =============================================================================
// Tests for the constant depth bias field.

#[test]
fn test_constant_default_is_zero() {
    let bias = DepthBias::new();
    assert_eq!(bias.constant, 0);
}

#[test]
fn test_constant_positive_values() {
    let bias = DepthBias::new().constant(5);
    assert_eq!(bias.constant, 5);
}

#[test]
fn test_constant_negative_values() {
    let bias = DepthBias::new().constant(-5);
    assert_eq!(bias.constant, -5);
}

#[test]
fn test_constant_max_i32() {
    let bias = DepthBias::new().constant(i32::MAX);
    assert_eq!(bias.constant, i32::MAX);
    assert!(bias.is_valid());
}

#[test]
fn test_constant_min_i32() {
    let bias = DepthBias::new().constant(i32::MIN);
    assert_eq!(bias.constant, i32::MIN);
    assert!(bias.is_valid());
}

#[test]
fn test_constant_typical_shadow_values() {
    // Common shadow bias constants: 1-8
    for val in [1, 2, 4, 8] {
        let bias = DepthBias::new().constant(val);
        assert_eq!(bias.constant, val);
        assert!(bias.is_valid());
    }
}

#[test]
fn test_constant_affects_is_none() {
    let bias = DepthBias::new().constant(1);
    assert!(!bias.is_none());
    assert!(bias.is_active());
}

// =============================================================================
// CATEGORY 3: SLOPE_SCALE FIELD (f32)
// =============================================================================
// Tests for the slope-scaled depth bias factor.

#[test]
fn test_slope_scale_default_is_zero() {
    let bias = DepthBias::new();
    assert!((bias.slope_scale - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_slope_scale_positive_values() {
    let bias = DepthBias::new().slope_scale(2.5);
    assert!((bias.slope_scale - 2.5).abs() < f32::EPSILON);
}

#[test]
fn test_slope_scale_negative_values() {
    let bias = DepthBias::new().slope_scale(-1.5);
    assert!((bias.slope_scale - (-1.5)).abs() < f32::EPSILON);
    assert!(bias.is_valid()); // Negative slope is allowed
}

#[test]
fn test_slope_scale_small_values() {
    let bias = DepthBias::new().slope_scale(0.001);
    assert!((bias.slope_scale - 0.001).abs() < f32::EPSILON);
}

#[test]
fn test_slope_scale_large_values() {
    let bias = DepthBias::new().slope_scale(100.0);
    assert!((bias.slope_scale - 100.0).abs() < f32::EPSILON);
    assert!(bias.is_valid());
}

#[test]
fn test_slope_scale_f32_max() {
    let bias = DepthBias::new().slope_scale(f32::MAX);
    assert_eq!(bias.slope_scale, f32::MAX);
    assert!(bias.is_valid());
}

#[test]
fn test_slope_scale_infinity_is_valid() {
    // Infinity is technically valid (not NaN)
    let bias = DepthBias::new().slope_scale(f32::INFINITY);
    assert!(bias.is_valid());
}

#[test]
fn test_slope_scale_nan_is_invalid() {
    let bias = DepthBias::new().slope_scale(f32::NAN);
    assert!(!bias.is_valid());
    assert!(matches!(
        bias.validate(),
        Err(DepthBiasError::InvalidSlopeScale(_))
    ));
}

#[test]
fn test_slope_scale_affects_is_none() {
    let bias = DepthBias::new().slope_scale(0.1);
    assert!(!bias.is_none());
}

// =============================================================================
// CATEGORY 4: CLAMP FIELD (f32)
// =============================================================================
// Tests for the maximum clamp value.

#[test]
fn test_clamp_default_is_zero() {
    let bias = DepthBias::new();
    assert!((bias.clamp - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_clamp_positive_values() {
    let bias = DepthBias::new().clamp(0.05);
    assert!((bias.clamp - 0.05).abs() < f32::EPSILON);
    assert!(bias.is_valid());
}

#[test]
fn test_clamp_negative_is_invalid() {
    let bias = DepthBias::new().clamp(-0.1);
    assert!(!bias.is_valid());
    let result = bias.validate();
    assert!(matches!(result, Err(DepthBiasError::NegativeClamp(_))));
}

#[test]
fn test_clamp_zero_means_no_clamping() {
    // Zero clamp = unlimited bias
    let bias = DepthBias::new()
        .constant(100)
        .slope_scale(50.0)
        .clamp(0.0);
    assert!(bias.is_valid());
}

#[test]
fn test_clamp_small_values() {
    let bias = DepthBias::new().clamp(0.001);
    assert!((bias.clamp - 0.001).abs() < f32::EPSILON);
}

#[test]
fn test_clamp_large_values() {
    let bias = DepthBias::new().clamp(100.0);
    assert!((bias.clamp - 100.0).abs() < f32::EPSILON);
    assert!(bias.is_valid());
}

#[test]
fn test_clamp_nan_is_invalid() {
    let bias = DepthBias::new().clamp(f32::NAN);
    assert!(!bias.is_valid());
}

#[test]
fn test_clamp_affects_is_none() {
    let bias = DepthBias::new().clamp(0.01);
    assert!(!bias.is_none());
}

// =============================================================================
// CATEGORY 5: SHADOW MAP PRESET
// =============================================================================
// Verify shadow_map() preset values match documentation.

#[test]
fn test_shadow_map_preset_constant() {
    let bias = DepthBias::shadow_map();
    assert_eq!(bias.constant, 2, "Shadow map constant should be 2");
}

#[test]
fn test_shadow_map_preset_slope_scale() {
    let bias = DepthBias::shadow_map();
    assert!(
        (bias.slope_scale - 2.0).abs() < f32::EPSILON,
        "Shadow map slope_scale should be 2.0"
    );
}

#[test]
fn test_shadow_map_preset_clamp() {
    let bias = DepthBias::shadow_map();
    assert!(
        (bias.clamp - 0.0).abs() < f32::EPSILON,
        "Shadow map clamp should be 0.0 (no clamping)"
    );
}

#[test]
fn test_shadow_map_preset_is_valid() {
    let bias = DepthBias::shadow_map();
    assert!(bias.is_valid());
}

#[test]
fn test_shadow_map_preset_is_active() {
    let bias = DepthBias::shadow_map();
    assert!(bias.is_active());
    assert!(!bias.is_none());
}

#[test]
fn test_shadow_map_preset_info() {
    let info = get_depth_bias_info("Shadow Map").expect("Shadow Map preset should exist");
    assert_eq!(info.preset.constant, 2);
    assert!((info.preset.slope_scale - 2.0).abs() < f32::EPSILON);
    assert!(info.use_cases.iter().any(|u| u.contains("shadow")));
}

#[test]
fn test_cascaded_shadow_map_preset() {
    let bias = DepthBias::cascaded_shadow_map();
    assert_eq!(bias.constant, 4, "CSM needs higher constant");
    assert!(
        (bias.slope_scale - 3.0).abs() < f32::EPSILON,
        "CSM needs higher slope_scale"
    );
    assert!(bias.is_valid());
}

#[test]
fn test_contact_shadow_preset() {
    let bias = DepthBias::contact_shadow();
    assert_eq!(bias.constant, 8, "Contact shadow uses aggressive bias");
    assert!((bias.slope_scale - 4.0).abs() < f32::EPSILON);
}

// =============================================================================
// CATEGORY 6: POLYGON OFFSET PRESET
// =============================================================================

#[test]
fn test_polygon_offset_preset_constant() {
    let bias = DepthBias::polygon_offset();
    assert_eq!(bias.constant, 1);
}

#[test]
fn test_polygon_offset_preset_slope_scale() {
    let bias = DepthBias::polygon_offset();
    assert!((bias.slope_scale - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_polygon_offset_preset_clamp() {
    let bias = DepthBias::polygon_offset();
    assert!((bias.clamp - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_polygon_offset_preset_info() {
    let info = get_depth_bias_info("Polygon Offset").expect("Polygon Offset preset should exist");
    assert_eq!(info.preset.constant, 1);
    assert!(info.use_cases.iter().any(|u| u.contains("wireframe")));
}

#[test]
fn test_decal_preset() {
    let bias = DepthBias::decal();
    assert_eq!(bias.constant, 1);
    assert!((bias.slope_scale - 1.0).abs() < f32::EPSILON);
}

// =============================================================================
// CATEGORY 7: DEFAULT (NONE) PRESET
// =============================================================================

#[test]
fn test_none_preset_all_zeros() {
    let bias = DepthBias::none();
    assert_eq!(bias.constant, 0);
    assert!((bias.slope_scale - 0.0).abs() < f32::EPSILON);
    assert!((bias.clamp - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_none_equals_default() {
    assert_eq!(DepthBias::none(), DepthBias::default());
}

#[test]
fn test_none_equals_new() {
    assert_eq!(DepthBias::none(), DepthBias::new());
}

#[test]
fn test_none_is_none_returns_true() {
    assert!(DepthBias::none().is_none());
}

#[test]
fn test_none_is_active_returns_false() {
    assert!(!DepthBias::none().is_active());
}

#[test]
fn test_outline_preset_negative_values() {
    let bias = DepthBias::outline();
    assert_eq!(bias.constant, -1);
    assert!((bias.slope_scale - (-1.0)).abs() < f32::EPSILON);
    // Negative values are valid for outlines
    assert!(bias.is_valid());
}

// =============================================================================
// CATEGORY 8: BUILDER API
// =============================================================================

#[test]
fn test_builder_new_creates_default() {
    let builder = DepthBiasBuilder::new();
    let bias = builder.build().expect("Default build should succeed");
    assert!(bias.is_none());
}

#[test]
fn test_builder_fluent_api() {
    let bias = DepthBiasBuilder::new()
        .constant(10)
        .slope_scale(5.0)
        .clamp(0.1)
        .build()
        .expect("Valid config should build");

    assert_eq!(bias.constant, 10);
    assert!((bias.slope_scale - 5.0).abs() < f32::EPSILON);
    assert!((bias.clamp - 0.1).abs() < f32::EPSILON);
}

#[test]
fn test_builder_from_preset() {
    let bias = DepthBiasBuilder::from_preset(DepthBias::shadow_map())
        .constant(4) // Override
        .build()
        .expect("Modified preset should build");

    assert_eq!(bias.constant, 4);
    assert!((bias.slope_scale - 2.0).abs() < f32::EPSILON); // From preset
}

#[test]
fn test_builder_shadow_map() {
    let bias = DepthBiasBuilder::shadow_map()
        .build()
        .expect("Shadow map build should succeed");

    assert_eq!(bias, DepthBias::shadow_map());
}

#[test]
fn test_builder_polygon_offset() {
    let bias = DepthBiasBuilder::polygon_offset()
        .build()
        .expect("Polygon offset build should succeed");

    assert_eq!(bias, DepthBias::polygon_offset());
}

#[test]
fn test_builder_scale() {
    let bias = DepthBiasBuilder::shadow_map()
        .scale(2.0)
        .build()
        .expect("Scaled build should succeed");

    assert_eq!(bias.constant, 4); // 2 * 2
    assert!((bias.slope_scale - 4.0).abs() < f32::EPSILON); // 2.0 * 2
}

#[test]
fn test_builder_invert() {
    let bias = DepthBiasBuilder::shadow_map()
        .invert()
        .build()
        .expect("Inverted build should succeed");

    assert_eq!(bias.constant, -2);
    assert!((bias.slope_scale - (-2.0)).abs() < f32::EPSILON);
}

#[test]
fn test_builder_validation_fails_on_invalid() {
    let result = DepthBiasBuilder::new().clamp(-0.5).build();

    assert!(result.is_err());
    assert!(matches!(result, Err(DepthBiasError::NegativeClamp(_))));
}

#[test]
fn test_builder_build_unchecked_allows_invalid() {
    let bias = DepthBiasBuilder::new().clamp(-0.5).build_unchecked();

    assert!((bias.clamp - (-0.5)).abs() < f32::EPSILON);
    // It's constructed, but validation would fail
    assert!(!bias.is_valid());
}

#[test]
fn test_builder_default_trait() {
    let builder: DepthBiasBuilder = Default::default();
    let bias = builder.build().expect("Default builder should build");
    assert!(bias.is_none());
}

#[test]
fn test_builder_chain_multiple_modifications() {
    let bias = DepthBiasBuilder::shadow_map()
        .constant(6) // Override
        .scale(0.5) // Scale: constant 3, slope 1.0
        .invert() // Invert: constant -3, slope -1.0
        .build()
        .expect("Chained modifications should succeed");

    assert_eq!(bias.constant, -3);
    assert!((bias.slope_scale - (-1.0)).abs() < f32::EPSILON);
}

// =============================================================================
// CATEGORY 9: REAL-WORLD SCENARIOS
// =============================================================================

#[test]
fn test_scenario_standard_forward_rendering() {
    // No depth bias needed for standard opaque rendering
    let bias = DepthBias::none();
    assert!(bias.is_valid());
    assert!(bias.is_none());
}

#[test]
fn test_scenario_directional_shadow() {
    let bias = DepthBias::shadow_map();
    assert!(bias.constant >= 1);
    assert!(bias.slope_scale >= 1.0);
}

#[test]
fn test_scenario_csm_far_cascade() {
    // Far cascades need more aggressive bias
    let bias = DepthBias::cascaded_shadow_map()
        .scaled(1.5); // Extra scaling for far cascade

    assert!(bias.constant > DepthBias::shadow_map().constant);
}

#[test]
fn test_scenario_wireframe_over_solid() {
    // Polygon offset for wireframe on top of solid
    let bias = DepthBias::polygon_offset();
    assert_eq!(bias.constant, 1);
    assert!((bias.slope_scale - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_scenario_decal_projection() {
    let bias = DepthBias::decal();
    // Decals need conservative bias to avoid floating
    assert!(bias.constant >= 1);
    assert!(bias.constant <= 2);
}

#[test]
fn test_scenario_cartoon_outline() {
    let bias = DepthBias::outline();
    // Outlines use negative bias for back-face offset
    assert!(bias.constant < 0);
    assert!(bias.slope_scale < 0.0);
}

#[test]
fn test_scenario_custom_shadow_bias() {
    // Scene-specific shadow bias tuning
    let bias = DepthBiasBuilder::shadow_map()
        .constant(6)
        .slope_scale(3.5)
        .clamp(0.02)
        .build()
        .expect("Custom shadow bias should build");

    assert_eq!(bias.constant, 6);
    assert!((bias.slope_scale - 3.5).abs() < f32::EPSILON);
    assert!((bias.clamp - 0.02).abs() < f32::EPSILON);
}

#[test]
fn test_scenario_multi_layer_decals() {
    // Stack multiple decals with increasing bias
    let layer1 = DepthBias::decal();
    let layer2 = DepthBias::decal().scaled(2.0);
    let layer3 = DepthBias::decal().scaled(3.0);

    assert!(layer1.constant < layer2.constant);
    assert!(layer2.constant < layer3.constant);
}

// =============================================================================
// CATEGORY 10: WGPU CONVERSION
// =============================================================================

#[test]
fn test_into_wgpu_depth_bias_state() {
    let bias = DepthBias::new()
        .constant(5)
        .slope_scale(3.0)
        .clamp(0.02);

    let wgpu_bias: wgpu::DepthBiasState = bias.into();

    assert_eq!(wgpu_bias.constant, 5);
    assert!((wgpu_bias.slope_scale - 3.0).abs() < f32::EPSILON);
    assert!((wgpu_bias.clamp - 0.02).abs() < f32::EPSILON);
}

#[test]
fn test_from_wgpu_depth_bias_state() {
    let wgpu_bias = wgpu::DepthBiasState {
        constant: 7,
        slope_scale: 2.5,
        clamp: 0.03,
    };

    let bias: DepthBias = wgpu_bias.into();

    assert_eq!(bias.constant, 7);
    assert!((bias.slope_scale - 2.5).abs() < f32::EPSILON);
    assert!((bias.clamp - 0.03).abs() < f32::EPSILON);
}

#[test]
fn test_roundtrip_wgpu_conversion() {
    let original = DepthBias::shadow_map();
    let wgpu_bias: wgpu::DepthBiasState = original.into();
    let converted: DepthBias = wgpu_bias.into();

    assert_eq!(original, converted);
}

#[test]
fn test_from_tuple() {
    let bias: DepthBias = (3, 1.5, 0.01).into();
    assert_eq!(bias.constant, 3);
    assert!((bias.slope_scale - 1.5).abs() < f32::EPSILON);
    assert!((bias.clamp - 0.01).abs() < f32::EPSILON);
}

#[test]
fn test_preset_wgpu_conversion() {
    // All presets should convert cleanly
    for preset in [
        DepthBias::none(),
        DepthBias::shadow_map(),
        DepthBias::cascaded_shadow_map(),
        DepthBias::polygon_offset(),
        DepthBias::decal(),
        DepthBias::outline(),
        DepthBias::contact_shadow(),
    ] {
        let wgpu_bias: wgpu::DepthBiasState = preset.into();
        let back: DepthBias = wgpu_bias.into();
        assert_eq!(preset, back);
    }
}

// =============================================================================
// CATEGORY 11: VALIDATION AND ERROR HANDLING
// =============================================================================

#[test]
fn test_validate_returns_ok_for_valid() {
    let bias = DepthBias::shadow_map();
    assert!(bias.validate().is_ok());
}

#[test]
fn test_validate_negative_clamp_error() {
    let bias = DepthBias::new().clamp(-0.1);
    let result = bias.validate();
    match result {
        Err(DepthBiasError::NegativeClamp(v)) => {
            assert!((v - (-0.1)).abs() < f32::EPSILON);
        }
        _ => panic!("Expected NegativeClamp error"),
    }
}

#[test]
fn test_validate_nan_slope_scale_error() {
    let bias = DepthBias::new().slope_scale(f32::NAN);
    let result = bias.validate();
    assert!(matches!(result, Err(DepthBiasError::InvalidSlopeScale(_))));
}

#[test]
fn test_validate_nan_clamp_error() {
    let bias = DepthBias::new().clamp(f32::NAN);
    let result = bias.validate();
    assert!(result.is_err());
}

#[test]
fn test_is_valid_helper() {
    assert!(DepthBias::shadow_map().is_valid());
    assert!(!DepthBias::new().clamp(-0.1).is_valid());
}

#[test]
fn test_error_display_negative_clamp() {
    let err = DepthBiasError::NegativeClamp(-0.5);
    let msg = format!("{}", err);
    assert!(msg.contains("-0.5"));
    assert!(msg.contains("clamp"));
}

#[test]
fn test_error_display_invalid_slope() {
    let err = DepthBiasError::InvalidSlopeScale(f32::NAN);
    let msg = format!("{}", err);
    assert!(msg.contains("slope_scale"));
}

#[test]
fn test_error_is_std_error() {
    fn assert_error<E: std::error::Error>() {}
    assert_error::<DepthBiasError>();
}

// =============================================================================
// CATEGORY 12: TRANSFORMATIONS
// =============================================================================

#[test]
fn test_scaled_multiplies_constant_and_slope() {
    let bias = DepthBias::new()
        .constant(2)
        .slope_scale(2.0)
        .scaled(3.0);

    assert_eq!(bias.constant, 6); // 2 * 3
    assert!((bias.slope_scale - 6.0).abs() < f32::EPSILON); // 2.0 * 3
}

#[test]
fn test_scaled_does_not_affect_clamp() {
    let bias = DepthBias::new()
        .clamp(0.5)
        .scaled(2.0);

    assert!((bias.clamp - 0.5).abs() < f32::EPSILON);
}

#[test]
fn test_scaled_by_zero() {
    let bias = DepthBias::shadow_map().scaled(0.0);
    assert_eq!(bias.constant, 0);
    assert!((bias.slope_scale - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_scaled_by_negative() {
    let bias = DepthBias::new()
        .constant(2)
        .slope_scale(2.0)
        .scaled(-1.0);

    assert_eq!(bias.constant, -2);
    assert!((bias.slope_scale - (-2.0)).abs() < f32::EPSILON);
}

#[test]
fn test_inverted_negates_constant_and_slope() {
    let bias = DepthBias::new()
        .constant(5)
        .slope_scale(3.0)
        .inverted();

    assert_eq!(bias.constant, -5);
    assert!((bias.slope_scale - (-3.0)).abs() < f32::EPSILON);
}

#[test]
fn test_inverted_preserves_clamp() {
    let bias = DepthBias::new()
        .clamp(0.5)
        .inverted();

    assert!((bias.clamp - 0.5).abs() < f32::EPSILON);
}

#[test]
fn test_double_invert_returns_original() {
    let original = DepthBias::shadow_map();
    let inverted_twice = original.inverted().inverted();

    assert_eq!(original.constant, inverted_twice.constant);
    assert!((original.slope_scale - inverted_twice.slope_scale).abs() < f32::EPSILON);
}

// =============================================================================
// CATEGORY 13: PRESET REGISTRY
// =============================================================================

#[test]
fn test_depth_bias_presets_count() {
    assert_eq!(DEPTH_BIAS_PRESETS.len(), 7);
}

#[test]
fn test_all_presets_are_valid() {
    for info in &DEPTH_BIAS_PRESETS {
        assert!(
            info.preset.is_valid(),
            "Preset '{}' is invalid",
            info.name
        );
    }
}

#[test]
fn test_preset_names_match_info() {
    let names: Vec<_> = depth_bias_preset_names().collect();
    assert_eq!(names.len(), DEPTH_BIAS_PRESETS.len());

    for name in names {
        let info = get_depth_bias_info(name);
        assert!(info.is_some(), "Preset name '{}' not found in info", name);
    }
}

#[test]
fn test_get_preset_returns_correct_values() {
    let preset = get_depth_bias_preset("Shadow Map").unwrap();
    assert_eq!(preset.constant, 2);
    assert!((preset.slope_scale - 2.0).abs() < f32::EPSILON);
}

#[test]
fn test_get_preset_none() {
    let preset = get_depth_bias_preset("None").unwrap();
    assert!(preset.is_none());
}

#[test]
fn test_get_nonexistent_preset() {
    assert!(get_depth_bias_preset("NonExistent").is_none());
    assert!(get_depth_bias_info("NonExistent").is_none());
}

#[test]
fn test_preset_info_has_use_cases() {
    for info in &DEPTH_BIAS_PRESETS {
        assert!(
            !info.use_cases.is_empty(),
            "Preset '{}' has no use cases",
            info.name
        );
    }
}

#[test]
fn test_preset_info_has_description() {
    for info in &DEPTH_BIAS_PRESETS {
        assert!(
            !info.description.is_empty(),
            "Preset '{}' has no description",
            info.name
        );
    }
}

// =============================================================================
// CATEGORY 14: TRAITS AND ATTRIBUTES
// =============================================================================

#[test]
fn test_depth_bias_debug() {
    let bias = DepthBias::shadow_map();
    let debug = format!("{:?}", bias);
    assert!(debug.contains("DepthBias"));
    assert!(debug.contains("constant"));
}

#[test]
fn test_depth_bias_clone() {
    let original = DepthBias::shadow_map();
    let cloned = original.clone();
    assert_eq!(original, cloned);
}

#[test]
fn test_depth_bias_copy() {
    let original = DepthBias::shadow_map();
    let copied = original;
    assert_eq!(original, copied);
}

#[test]
fn test_depth_bias_partial_eq() {
    let bias1 = DepthBias::shadow_map();
    let bias2 = DepthBias::shadow_map();
    let bias3 = DepthBias::polygon_offset();

    assert_eq!(bias1, bias2);
    assert_ne!(bias1, bias3);
}

#[test]
fn test_depth_bias_send() {
    fn assert_send<T: Send>() {}
    assert_send::<DepthBias>();
}

#[test]
fn test_depth_bias_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<DepthBias>();
}

#[test]
fn test_builder_debug() {
    let builder = DepthBiasBuilder::shadow_map();
    let debug = format!("{:?}", builder);
    assert!(debug.contains("DepthBiasBuilder"));
}

#[test]
fn test_builder_clone() {
    let builder = DepthBiasBuilder::shadow_map();
    let cloned = builder.clone();
    assert_eq!(
        builder.build().unwrap(),
        cloned.build().unwrap()
    );
}

// =============================================================================
// CATEGORY 15: EDGE CASES
// =============================================================================

#[test]
fn test_epsilon_slope_scale() {
    let bias = DepthBias::new().slope_scale(f32::EPSILON);
    assert!(bias.is_valid());
    assert!(bias.slope_scale > 0.0);
}

#[test]
fn test_negative_infinity_slope_scale() {
    let bias = DepthBias::new().slope_scale(f32::NEG_INFINITY);
    // Negative infinity is valid (not NaN)
    assert!(bias.is_valid());
}

#[test]
fn test_min_positive_clamp() {
    let bias = DepthBias::new().clamp(f32::MIN_POSITIVE);
    assert!(bias.is_valid());
    assert!(bias.clamp > 0.0);
}

#[test]
fn test_zero_clamp_exactly() {
    let bias = DepthBias::new().clamp(0.0);
    assert!(bias.is_valid());
    assert_eq!(bias.clamp, 0.0);
}

#[test]
fn test_negative_zero_clamp() {
    // -0.0 should be treated as 0.0 for validation
    let bias = DepthBias::new().clamp(-0.0);
    // -0.0 equals 0.0 in IEEE 754
    assert!(bias.is_valid());
}

#[test]
fn test_chained_builder_methods_order_independence() {
    let bias1 = DepthBiasBuilder::new()
        .constant(5)
        .slope_scale(2.0)
        .clamp(0.1)
        .build()
        .unwrap();

    let bias2 = DepthBiasBuilder::new()
        .clamp(0.1)
        .slope_scale(2.0)
        .constant(5)
        .build()
        .unwrap();

    assert_eq!(bias1, bias2);
}

#[test]
fn test_overwrite_builder_value() {
    let bias = DepthBiasBuilder::new()
        .constant(1)
        .constant(2)
        .constant(3)
        .build()
        .unwrap();

    assert_eq!(bias.constant, 3); // Last value wins
}

// =============================================================================
// SUMMARY TESTS
// =============================================================================

#[test]
fn test_acceptance_criteria_constant() {
    // AC: constant: i32
    let bias = DepthBias::new();
    let _: i32 = bias.constant;

    let bias = DepthBias::new().constant(i32::MAX);
    assert_eq!(bias.constant, i32::MAX);

    let bias = DepthBias::new().constant(i32::MIN);
    assert_eq!(bias.constant, i32::MIN);
}

#[test]
fn test_acceptance_criteria_slope_scale() {
    // AC: slope_scale: f32
    let bias = DepthBias::new();
    let _: f32 = bias.slope_scale;

    let bias = DepthBias::new().slope_scale(f32::MAX);
    assert_eq!(bias.slope_scale, f32::MAX);
}

#[test]
fn test_acceptance_criteria_clamp() {
    // AC: clamp: f32
    let bias = DepthBias::new();
    let _: f32 = bias.clamp;

    let bias = DepthBias::new().clamp(0.5);
    assert!((bias.clamp - 0.5).abs() < f32::EPSILON);
}

#[test]
fn test_acceptance_criteria_shadow_map_preset() {
    // AC: Shadow map preset
    let bias = DepthBias::shadow_map();
    assert_eq!(bias.constant, 2);
    assert!((bias.slope_scale - 2.0).abs() < f32::EPSILON);
    assert!((bias.clamp - 0.0).abs() < f32::EPSILON);
    assert!(bias.is_valid());
}
