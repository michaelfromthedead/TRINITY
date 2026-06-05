// SPDX-License-Identifier: MIT
//
// blackbox_blend_mode.rs -- Blackbox tests for T-WGPU-P3.5.2 Blend Modes.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only the documented public types and functions:
//
//   - BlendMode -- Core struct for blend mode configuration
//   - BlendModeBuilder -- Fluent builder API
//   - BlendModeInfo -- Preset metadata
//   - BlendFactorInfo -- Blend factor metadata
//   - BlendOperationInfo -- Blend operation metadata
//
// PUBLIC FUNCTIONS:
//   - get_blend_mode_info, get_blend_mode_preset
//   - get_blend_factor_info, get_blend_operation_info
//   - alpha_presets, blend_hdr_presets, constant_presets
//   - blend_mode_preset_names, blend_factor_names, blend_operation_names
//
// CONSTANTS:
//   - BLEND_MODE_PRESETS, BLEND_FACTORS, BLEND_OPERATIONS
//
// ACCEPTANCE CRITERIA (T-WGPU-P3.5.2):
//   1. Alpha blending - src*srcA + dst*(1-srcA)
//   2. Premultiplied alpha - Correct factors
//   3. Additive - src + dst
//   4. Multiply - src * dst
//   5. BlendFactor enum - 13 values
//   6. BlendOperation enum - 5 values
//   7. Color and alpha separate
//
// TEST CATEGORIES:
//   1. API Tests - Public interface, constructors, methods
//   2. Alpha blending - src*srcA + dst*(1-srcA)
//   3. Premultiplied alpha - Correct factors
//   4. Additive - src + dst
//   5. Multiply - src * dst
//   6. BlendFactor enum - All 13 values
//   7. BlendOperation enum - All 5 values
//   8. Color/alpha separate - Independent configuration
//   9. Builder API - BlendModeBuilder fluent interface
//   10. Real-world scenarios - Common blending configurations
//
// Total target: 60+ tests

use renderer_backend::render_pipeline::{
    alpha_presets, blend_factor_names, blend_hdr_presets, blend_mode_preset_names,
    blend_operation_names, constant_presets, get_blend_factor_info, get_blend_mode_info,
    get_blend_mode_preset, get_blend_operation_info, BlendMode, BlendModeBuilder,
    BLEND_FACTORS, BLEND_MODE_PRESETS, BLEND_OPERATIONS,
};

// =============================================================================
// CATEGORY 1: API TESTS - Public Interface
// =============================================================================

mod api_tests {
    use super::*;

    #[test]
    fn test_blend_mode_is_public() {
        // Verify BlendMode struct is accessible
        let mode = BlendMode::default();
        assert_eq!(mode.color.src_factor, wgpu::BlendFactor::SrcAlpha);
    }

    #[test]
    fn test_blend_mode_builder_is_public() {
        // Verify BlendModeBuilder is accessible
        let builder = BlendModeBuilder::new();
        let mode = builder.build();
        assert_eq!(mode.color.operation, wgpu::BlendOperation::Add);
    }

    #[test]
    fn test_blend_mode_info_is_public() {
        // Verify BlendModeInfo is accessible
        let info = &BLEND_MODE_PRESETS[0];
        assert!(!info.name.is_empty());
    }

    #[test]
    fn test_blend_factor_info_is_public() {
        // Verify BlendFactorInfo is accessible
        let info = &BLEND_FACTORS[0];
        assert!(!info.name.is_empty());
    }

    #[test]
    fn test_blend_operation_info_is_public() {
        // Verify BlendOperationInfo is accessible
        let info = &BLEND_OPERATIONS[0];
        assert!(!info.name.is_empty());
    }

    #[test]
    fn test_blend_mode_presets_array() {
        // Verify BLEND_MODE_PRESETS array is accessible
        assert_eq!(BLEND_MODE_PRESETS.len(), 12);
    }

    #[test]
    fn test_blend_factors_array() {
        // Verify BLEND_FACTORS array has all 13 factors
        assert_eq!(BLEND_FACTORS.len(), 13);
    }

    #[test]
    fn test_blend_operations_array() {
        // Verify BLEND_OPERATIONS array has all 5 operations
        assert_eq!(BLEND_OPERATIONS.len(), 5);
    }

    #[test]
    fn test_public_field_access() {
        // Verify public fields are accessible
        let mode = BlendMode::alpha();
        let _color = mode.color;
        let _alpha = mode.alpha;
    }

    #[test]
    fn test_default_derives() {
        // Verify BlendMode has Debug, Clone, Copy, PartialEq
        let mode = BlendMode::alpha();
        let cloned = mode;
        assert_eq!(mode, cloned);
        let debug_str = format!("{:?}", mode);
        assert!(debug_str.contains("BlendMode"));
    }
}

// =============================================================================
// CATEGORY 2: ALPHA BLENDING - src*srcA + dst*(1-srcA)
// =============================================================================

mod alpha_blending_tests {
    use super::*;

    #[test]
    fn test_alpha_preset_color_src_factor() {
        let mode = BlendMode::alpha();
        // src * srcA
        assert_eq!(mode.color.src_factor, wgpu::BlendFactor::SrcAlpha);
    }

    #[test]
    fn test_alpha_preset_color_dst_factor() {
        let mode = BlendMode::alpha();
        // dst * (1 - srcA)
        assert_eq!(mode.color.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
    }

    #[test]
    fn test_alpha_preset_color_operation() {
        let mode = BlendMode::alpha();
        // src*srcA + dst*(1-srcA)
        assert_eq!(mode.color.operation, wgpu::BlendOperation::Add);
    }

    #[test]
    fn test_alpha_preset_alpha_channel() {
        let mode = BlendMode::alpha();
        // Alpha: one * src + (1 - src_alpha) * dst
        assert_eq!(mode.alpha.src_factor, wgpu::BlendFactor::One);
        assert_eq!(mode.alpha.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
        assert_eq!(mode.alpha.operation, wgpu::BlendOperation::Add);
    }

    #[test]
    fn test_alpha_preset_uses_alpha() {
        let mode = BlendMode::alpha();
        assert!(mode.uses_alpha());
    }

    #[test]
    fn test_alpha_preset_not_replace() {
        let mode = BlendMode::alpha();
        assert!(!mode.is_replace());
    }

    #[test]
    fn test_alpha_preset_not_additive() {
        let mode = BlendMode::alpha();
        assert!(!mode.is_additive());
    }

    #[test]
    fn test_alpha_preset_matches_wgpu() {
        let mode = BlendMode::alpha();
        let wgpu_state = wgpu::BlendState::ALPHA_BLENDING;
        assert_eq!(mode.color, wgpu_state.color);
        assert_eq!(mode.alpha, wgpu_state.alpha);
    }

    #[test]
    fn test_default_is_alpha() {
        let default_mode = BlendMode::default();
        let alpha_mode = BlendMode::alpha();
        assert_eq!(default_mode, alpha_mode);
    }

    #[test]
    fn test_alpha_cannot_exceed_one() {
        let mode = BlendMode::alpha();
        assert!(!mode.can_exceed_one());
    }
}

// =============================================================================
// CATEGORY 3: PREMULTIPLIED ALPHA - Correct factors
// =============================================================================

mod premultiplied_alpha_tests {
    use super::*;

    #[test]
    fn test_premultiplied_color_src_factor() {
        let mode = BlendMode::premultiplied_alpha();
        // Premultiplied: src factor is One (RGB already multiplied by alpha)
        assert_eq!(mode.color.src_factor, wgpu::BlendFactor::One);
    }

    #[test]
    fn test_premultiplied_color_dst_factor() {
        let mode = BlendMode::premultiplied_alpha();
        // dst * (1 - srcA)
        assert_eq!(mode.color.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
    }

    #[test]
    fn test_premultiplied_color_operation() {
        let mode = BlendMode::premultiplied_alpha();
        // src + dst*(1-srcA)
        assert_eq!(mode.color.operation, wgpu::BlendOperation::Add);
    }

    #[test]
    fn test_premultiplied_alpha_channel() {
        let mode = BlendMode::premultiplied_alpha();
        // Alpha: one * src + (1 - src_alpha) * dst
        assert_eq!(mode.alpha.src_factor, wgpu::BlendFactor::One);
        assert_eq!(mode.alpha.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
        assert_eq!(mode.alpha.operation, wgpu::BlendOperation::Add);
    }

    #[test]
    fn test_premultiplied_uses_alpha() {
        let mode = BlendMode::premultiplied_alpha();
        assert!(mode.uses_alpha());
    }

    #[test]
    fn test_premultiplied_is_uniform() {
        let mode = BlendMode::premultiplied_alpha();
        // Premultiplied has same config for color and alpha
        assert!(mode.is_uniform());
    }

    #[test]
    fn test_premultiplied_matches_wgpu() {
        let mode = BlendMode::premultiplied_alpha();
        let wgpu_state = wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING;
        assert_eq!(mode.color, wgpu_state.color);
        assert_eq!(mode.alpha, wgpu_state.alpha);
    }

    #[test]
    fn test_premultiplied_differs_from_alpha() {
        let premultiplied = BlendMode::premultiplied_alpha();
        let alpha = BlendMode::alpha();
        // They differ in color src factor
        assert_ne!(premultiplied.color.src_factor, alpha.color.src_factor);
    }

    #[test]
    fn test_premultiplied_cannot_exceed_one() {
        let mode = BlendMode::premultiplied_alpha();
        assert!(!mode.can_exceed_one());
    }
}

// =============================================================================
// CATEGORY 4: ADDITIVE BLENDING - src + dst
// =============================================================================

mod additive_blending_tests {
    use super::*;

    #[test]
    fn test_additive_color_factors() {
        let mode = BlendMode::additive();
        // src + dst: both factors are One
        assert_eq!(mode.color.src_factor, wgpu::BlendFactor::One);
        assert_eq!(mode.color.dst_factor, wgpu::BlendFactor::One);
    }

    #[test]
    fn test_additive_color_operation() {
        let mode = BlendMode::additive();
        assert_eq!(mode.color.operation, wgpu::BlendOperation::Add);
    }

    #[test]
    fn test_additive_alpha_factors() {
        let mode = BlendMode::additive();
        // Same for alpha
        assert_eq!(mode.alpha.src_factor, wgpu::BlendFactor::One);
        assert_eq!(mode.alpha.dst_factor, wgpu::BlendFactor::One);
    }

    #[test]
    fn test_additive_is_additive() {
        let mode = BlendMode::additive();
        assert!(mode.is_additive());
    }

    #[test]
    fn test_additive_is_uniform() {
        let mode = BlendMode::additive();
        assert!(mode.is_uniform());
    }

    #[test]
    fn test_additive_does_not_use_alpha() {
        let mode = BlendMode::additive();
        assert!(!mode.uses_alpha());
    }

    #[test]
    fn test_additive_can_exceed_one() {
        let mode = BlendMode::additive();
        // Additive blending can produce values > 1.0 (HDR)
        assert!(mode.can_exceed_one());
    }

    #[test]
    fn test_additive_alpha_variant() {
        let mode = BlendMode::additive_alpha();
        // Color: src_alpha * src + dst
        assert_eq!(mode.color.src_factor, wgpu::BlendFactor::SrcAlpha);
        assert_eq!(mode.color.dst_factor, wgpu::BlendFactor::One);
    }

    #[test]
    fn test_additive_alpha_uses_alpha() {
        let mode = BlendMode::additive_alpha();
        assert!(mode.uses_alpha());
    }

    #[test]
    fn test_additive_not_replace() {
        let mode = BlendMode::additive();
        assert!(!mode.is_replace());
    }
}

// =============================================================================
// CATEGORY 5: MULTIPLY BLENDING - src * dst
// =============================================================================

mod multiply_blending_tests {
    use super::*;

    #[test]
    fn test_multiply_color_factors() {
        let mode = BlendMode::multiply();
        // src * dst achieved by: dst * src + zero * dst
        assert_eq!(mode.color.src_factor, wgpu::BlendFactor::Dst);
        assert_eq!(mode.color.dst_factor, wgpu::BlendFactor::Zero);
    }

    #[test]
    fn test_multiply_color_operation() {
        let mode = BlendMode::multiply();
        assert_eq!(mode.color.operation, wgpu::BlendOperation::Add);
    }

    #[test]
    fn test_multiply_alpha_factors() {
        let mode = BlendMode::multiply();
        // Alpha uses DstAlpha
        assert_eq!(mode.alpha.src_factor, wgpu::BlendFactor::DstAlpha);
        assert_eq!(mode.alpha.dst_factor, wgpu::BlendFactor::Zero);
    }

    #[test]
    fn test_multiply_only_darkens() {
        let mode = BlendMode::multiply();
        assert!(mode.only_darkens());
    }

    #[test]
    fn test_multiply_not_additive() {
        let mode = BlendMode::multiply();
        assert!(!mode.is_additive());
    }

    #[test]
    fn test_multiply_cannot_exceed_one() {
        let mode = BlendMode::multiply();
        assert!(!mode.can_exceed_one());
    }

    #[test]
    fn test_multiply_does_not_use_alpha_factor() {
        let mode = BlendMode::multiply();
        // Multiply uses Dst factor, not alpha-related factors for color
        assert!(!mode.uses_alpha());
    }

    #[test]
    fn test_screen_is_inverse_of_multiply() {
        let screen = BlendMode::screen();
        // Screen: 1 - (1-src)*(1-dst) = src + (1-src)*dst
        assert_eq!(screen.color.src_factor, wgpu::BlendFactor::One);
        assert_eq!(screen.color.dst_factor, wgpu::BlendFactor::OneMinusSrc);
    }

    #[test]
    fn test_screen_only_lightens() {
        let mode = BlendMode::screen();
        assert!(mode.only_lightens());
    }

    #[test]
    fn test_soft_light_preset() {
        let mode = BlendMode::soft_light();
        assert_eq!(mode.color.src_factor, wgpu::BlendFactor::Dst);
        assert_eq!(mode.color.dst_factor, wgpu::BlendFactor::OneMinusSrc);
    }
}

// =============================================================================
// CATEGORY 6: BLEND FACTOR ENUM - All 13 values
// =============================================================================

mod blend_factor_tests {
    use super::*;

    #[test]
    fn test_blend_factor_zero() {
        let info = get_blend_factor_info(wgpu::BlendFactor::Zero);
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.name, "Zero");
        assert_eq!(info.value, "0");
    }

    #[test]
    fn test_blend_factor_one() {
        let info = get_blend_factor_info(wgpu::BlendFactor::One);
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.name, "One");
        assert_eq!(info.value, "1");
    }

    #[test]
    fn test_blend_factor_src() {
        let info = get_blend_factor_info(wgpu::BlendFactor::Src);
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.name, "Src");
        assert_eq!(info.value, "src");
    }

    #[test]
    fn test_blend_factor_one_minus_src() {
        let info = get_blend_factor_info(wgpu::BlendFactor::OneMinusSrc);
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.name, "OneMinusSrc");
        assert_eq!(info.value, "1 - src");
    }

    #[test]
    fn test_blend_factor_src_alpha() {
        let info = get_blend_factor_info(wgpu::BlendFactor::SrcAlpha);
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.name, "SrcAlpha");
        assert_eq!(info.value, "src.a");
    }

    #[test]
    fn test_blend_factor_one_minus_src_alpha() {
        let info = get_blend_factor_info(wgpu::BlendFactor::OneMinusSrcAlpha);
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.name, "OneMinusSrcAlpha");
        assert_eq!(info.value, "1 - src.a");
    }

    #[test]
    fn test_blend_factor_dst() {
        let info = get_blend_factor_info(wgpu::BlendFactor::Dst);
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.name, "Dst");
        assert_eq!(info.value, "dst");
    }

    #[test]
    fn test_blend_factor_one_minus_dst() {
        let info = get_blend_factor_info(wgpu::BlendFactor::OneMinusDst);
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.name, "OneMinusDst");
        assert_eq!(info.value, "1 - dst");
    }

    #[test]
    fn test_blend_factor_dst_alpha() {
        let info = get_blend_factor_info(wgpu::BlendFactor::DstAlpha);
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.name, "DstAlpha");
        assert_eq!(info.value, "dst.a");
    }

    #[test]
    fn test_blend_factor_one_minus_dst_alpha() {
        let info = get_blend_factor_info(wgpu::BlendFactor::OneMinusDstAlpha);
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.name, "OneMinusDstAlpha");
        assert_eq!(info.value, "1 - dst.a");
    }

    #[test]
    fn test_blend_factor_src_alpha_saturated() {
        let info = get_blend_factor_info(wgpu::BlendFactor::SrcAlphaSaturated);
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.name, "SrcAlphaSaturated");
        assert_eq!(info.value, "min(src.a, 1 - dst.a)");
    }

    #[test]
    fn test_blend_factor_constant() {
        let info = get_blend_factor_info(wgpu::BlendFactor::Constant);
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.name, "Constant");
        assert_eq!(info.value, "const");
    }

    #[test]
    fn test_blend_factor_one_minus_constant() {
        let info = get_blend_factor_info(wgpu::BlendFactor::OneMinusConstant);
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.name, "OneMinusConstant");
        assert_eq!(info.value, "1 - const");
    }

    #[test]
    fn test_all_13_blend_factors_have_info() {
        let factors = [
            wgpu::BlendFactor::Zero,
            wgpu::BlendFactor::One,
            wgpu::BlendFactor::Src,
            wgpu::BlendFactor::OneMinusSrc,
            wgpu::BlendFactor::SrcAlpha,
            wgpu::BlendFactor::OneMinusSrcAlpha,
            wgpu::BlendFactor::Dst,
            wgpu::BlendFactor::OneMinusDst,
            wgpu::BlendFactor::DstAlpha,
            wgpu::BlendFactor::OneMinusDstAlpha,
            wgpu::BlendFactor::SrcAlphaSaturated,
            wgpu::BlendFactor::Constant,
            wgpu::BlendFactor::OneMinusConstant,
        ];

        for factor in factors {
            assert!(
                get_blend_factor_info(factor).is_some(),
                "Missing info for {:?}",
                factor
            );
        }
    }

    #[test]
    fn test_blend_factor_names_count() {
        let names: Vec<_> = blend_factor_names().collect();
        assert_eq!(names.len(), 13);
    }

    #[test]
    fn test_all_factors_can_be_set() {
        let factors = [
            wgpu::BlendFactor::Zero,
            wgpu::BlendFactor::One,
            wgpu::BlendFactor::Src,
            wgpu::BlendFactor::OneMinusSrc,
            wgpu::BlendFactor::SrcAlpha,
            wgpu::BlendFactor::OneMinusSrcAlpha,
            wgpu::BlendFactor::Dst,
            wgpu::BlendFactor::OneMinusDst,
            wgpu::BlendFactor::DstAlpha,
            wgpu::BlendFactor::OneMinusDstAlpha,
            wgpu::BlendFactor::SrcAlphaSaturated,
            wgpu::BlendFactor::Constant,
            wgpu::BlendFactor::OneMinusConstant,
        ];

        for factor in factors {
            let mode = BlendMode::alpha().with_color_src_factor(factor);
            assert_eq!(mode.color.src_factor, factor);
        }
    }
}

// =============================================================================
// CATEGORY 7: BLEND OPERATION ENUM - All 5 values
// =============================================================================

mod blend_operation_tests {
    use super::*;

    #[test]
    fn test_blend_operation_add() {
        let info = get_blend_operation_info(wgpu::BlendOperation::Add);
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.name, "Add");
        assert_eq!(info.formula, "src + dst");
    }

    #[test]
    fn test_blend_operation_subtract() {
        let info = get_blend_operation_info(wgpu::BlendOperation::Subtract);
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.name, "Subtract");
        assert_eq!(info.formula, "src - dst");
    }

    #[test]
    fn test_blend_operation_reverse_subtract() {
        let info = get_blend_operation_info(wgpu::BlendOperation::ReverseSubtract);
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.name, "ReverseSubtract");
        assert_eq!(info.formula, "dst - src");
    }

    #[test]
    fn test_blend_operation_min() {
        let info = get_blend_operation_info(wgpu::BlendOperation::Min);
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.name, "Min");
        assert_eq!(info.formula, "min(src, dst)");
    }

    #[test]
    fn test_blend_operation_max() {
        let info = get_blend_operation_info(wgpu::BlendOperation::Max);
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.name, "Max");
        assert_eq!(info.formula, "max(src, dst)");
    }

    #[test]
    fn test_all_5_blend_operations_have_info() {
        let operations = [
            wgpu::BlendOperation::Add,
            wgpu::BlendOperation::Subtract,
            wgpu::BlendOperation::ReverseSubtract,
            wgpu::BlendOperation::Min,
            wgpu::BlendOperation::Max,
        ];

        for op in operations {
            assert!(
                get_blend_operation_info(op).is_some(),
                "Missing info for {:?}",
                op
            );
        }
    }

    #[test]
    fn test_blend_operation_names_count() {
        let names: Vec<_> = blend_operation_names().collect();
        assert_eq!(names.len(), 5);
    }

    #[test]
    fn test_subtract_preset() {
        let mode = BlendMode::subtract();
        assert_eq!(mode.color.operation, wgpu::BlendOperation::ReverseSubtract);
    }

    #[test]
    fn test_min_preset() {
        let mode = BlendMode::min();
        assert_eq!(mode.color.operation, wgpu::BlendOperation::Min);
        assert_eq!(mode.alpha.operation, wgpu::BlendOperation::Min);
    }

    #[test]
    fn test_max_preset() {
        let mode = BlendMode::max();
        assert_eq!(mode.color.operation, wgpu::BlendOperation::Max);
        assert_eq!(mode.alpha.operation, wgpu::BlendOperation::Max);
    }

    #[test]
    fn test_all_operations_can_be_set() {
        let operations = [
            wgpu::BlendOperation::Add,
            wgpu::BlendOperation::Subtract,
            wgpu::BlendOperation::ReverseSubtract,
            wgpu::BlendOperation::Min,
            wgpu::BlendOperation::Max,
        ];

        for op in operations {
            let mode = BlendMode::alpha().with_color_operation(op);
            assert_eq!(mode.color.operation, op);
        }
    }
}

// =============================================================================
// CATEGORY 8: COLOR/ALPHA SEPARATE - Independent Configuration
// =============================================================================

mod color_alpha_separate_tests {
    use super::*;

    #[test]
    fn test_alpha_preset_has_different_color_and_alpha() {
        let mode = BlendMode::alpha();
        // Color uses SrcAlpha, alpha uses One for src_factor
        assert_ne!(mode.color.src_factor, mode.alpha.src_factor);
        assert!(!mode.is_uniform());
    }

    #[test]
    fn test_additive_has_same_color_and_alpha() {
        let mode = BlendMode::additive();
        assert_eq!(mode.color, mode.alpha);
        assert!(mode.is_uniform());
    }

    #[test]
    fn test_with_color_independent_of_alpha() {
        let component = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::Constant,
            dst_factor: wgpu::BlendFactor::OneMinusConstant,
            operation: wgpu::BlendOperation::Subtract,
        };
        let mode = BlendMode::alpha().with_color(component);

        // Color changed
        assert_eq!(mode.color, component);
        // Alpha unchanged
        assert_eq!(mode.alpha.src_factor, wgpu::BlendFactor::One);
    }

    #[test]
    fn test_with_alpha_independent_of_color() {
        let component = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::DstAlpha,
            dst_factor: wgpu::BlendFactor::OneMinusDstAlpha,
            operation: wgpu::BlendOperation::Max,
        };
        let mode = BlendMode::alpha().with_alpha(component);

        // Alpha changed
        assert_eq!(mode.alpha, component);
        // Color unchanged (SrcAlpha from alpha preset)
        assert_eq!(mode.color.src_factor, wgpu::BlendFactor::SrcAlpha);
    }

    #[test]
    fn test_builder_separate_color_alpha() {
        let mode = BlendModeBuilder::new()
            .color_src_factor(wgpu::BlendFactor::One)
            .color_dst_factor(wgpu::BlendFactor::One)
            .alpha_src_factor(wgpu::BlendFactor::Zero)
            .alpha_dst_factor(wgpu::BlendFactor::One)
            .build();

        // Color is additive
        assert_eq!(mode.color.src_factor, wgpu::BlendFactor::One);
        assert_eq!(mode.color.dst_factor, wgpu::BlendFactor::One);

        // Alpha preserves destination
        assert_eq!(mode.alpha.src_factor, wgpu::BlendFactor::Zero);
        assert_eq!(mode.alpha.dst_factor, wgpu::BlendFactor::One);

        assert!(!mode.is_uniform());
    }

    #[test]
    fn test_color_and_alpha_operations_independent() {
        let mode = BlendModeBuilder::new()
            .color_operation(wgpu::BlendOperation::Add)
            .alpha_operation(wgpu::BlendOperation::Max)
            .build();

        assert_eq!(mode.color.operation, wgpu::BlendOperation::Add);
        assert_eq!(mode.alpha.operation, wgpu::BlendOperation::Max);
    }

    #[test]
    fn test_color_operation_accessor() {
        let mode = BlendMode::min();
        assert_eq!(mode.color_operation(), wgpu::BlendOperation::Min);
    }

    #[test]
    fn test_alpha_operation_accessor() {
        let mode = BlendMode::max();
        assert_eq!(mode.alpha_operation(), wgpu::BlendOperation::Max);
    }

    #[test]
    fn test_uniform_constructor() {
        let component = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        let mode = BlendMode::uniform(component);
        assert_eq!(mode.color, component);
        assert_eq!(mode.alpha, component);
        assert!(mode.is_uniform());
    }

    #[test]
    fn test_new_constructor_separate_components() {
        let color = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::Zero,
            operation: wgpu::BlendOperation::Add,
        };
        let alpha = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Max,
        };
        let mode = BlendMode::new(color, alpha);
        assert_eq!(mode.color, color);
        assert_eq!(mode.alpha, alpha);
    }
}

// =============================================================================
// CATEGORY 9: BUILDER API - BlendModeBuilder Fluent Interface
// =============================================================================

mod builder_tests {
    use super::*;

    #[test]
    fn test_builder_new_defaults_to_alpha() {
        let mode = BlendModeBuilder::new().build();
        assert_eq!(mode, BlendMode::alpha());
    }

    #[test]
    fn test_builder_from_preset() {
        let mode = BlendModeBuilder::from_preset(BlendMode::additive())
            .color_dst_factor(wgpu::BlendFactor::SrcAlpha)
            .build();
        assert_eq!(mode.color.src_factor, wgpu::BlendFactor::One);
        assert_eq!(mode.color.dst_factor, wgpu::BlendFactor::SrcAlpha);
    }

    #[test]
    fn test_builder_from_replace() {
        let mode = BlendModeBuilder::from_replace().build();
        assert!(mode.is_replace());
    }

    #[test]
    fn test_builder_color_component() {
        let component = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::Dst,
            dst_factor: wgpu::BlendFactor::Src,
            operation: wgpu::BlendOperation::Subtract,
        };
        let mode = BlendModeBuilder::new().color(component).build();
        assert_eq!(mode.color, component);
    }

    #[test]
    fn test_builder_alpha_component() {
        let component = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::DstAlpha,
            dst_factor: wgpu::BlendFactor::SrcAlpha,
            operation: wgpu::BlendOperation::Max,
        };
        let mode = BlendModeBuilder::new().alpha(component).build();
        assert_eq!(mode.alpha, component);
    }

    #[test]
    fn test_builder_uniform() {
        let component = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        let mode = BlendModeBuilder::new().uniform(component).build();
        assert_eq!(mode.color, component);
        assert_eq!(mode.alpha, component);
    }

    #[test]
    fn test_builder_src_factor_sets_both() {
        let mode = BlendModeBuilder::new()
            .src_factor(wgpu::BlendFactor::Constant)
            .build();
        assert_eq!(mode.color.src_factor, wgpu::BlendFactor::Constant);
        assert_eq!(mode.alpha.src_factor, wgpu::BlendFactor::Constant);
    }

    #[test]
    fn test_builder_dst_factor_sets_both() {
        let mode = BlendModeBuilder::new()
            .dst_factor(wgpu::BlendFactor::Zero)
            .build();
        assert_eq!(mode.color.dst_factor, wgpu::BlendFactor::Zero);
        assert_eq!(mode.alpha.dst_factor, wgpu::BlendFactor::Zero);
    }

    #[test]
    fn test_builder_operation_sets_both() {
        let mode = BlendModeBuilder::new()
            .operation(wgpu::BlendOperation::Min)
            .build();
        assert_eq!(mode.color.operation, wgpu::BlendOperation::Min);
        assert_eq!(mode.alpha.operation, wgpu::BlendOperation::Min);
    }

    #[test]
    fn test_builder_build_wgpu() {
        let state = BlendModeBuilder::new()
            .src_factor(wgpu::BlendFactor::One)
            .dst_factor(wgpu::BlendFactor::One)
            .build_wgpu();
        assert_eq!(state.color.src_factor, wgpu::BlendFactor::One);
        assert_eq!(state.color.dst_factor, wgpu::BlendFactor::One);
    }

    #[test]
    fn test_builder_fluent_chain() {
        let mode = BlendModeBuilder::new()
            .color_src_factor(wgpu::BlendFactor::SrcAlpha)
            .color_dst_factor(wgpu::BlendFactor::OneMinusSrcAlpha)
            .color_operation(wgpu::BlendOperation::Add)
            .alpha_src_factor(wgpu::BlendFactor::One)
            .alpha_dst_factor(wgpu::BlendFactor::Zero)
            .alpha_operation(wgpu::BlendOperation::Add)
            .build();

        assert_eq!(mode.color.src_factor, wgpu::BlendFactor::SrcAlpha);
        assert_eq!(mode.alpha.src_factor, wgpu::BlendFactor::One);
        assert_eq!(mode.alpha.dst_factor, wgpu::BlendFactor::Zero);
    }

    #[test]
    fn test_builder_default_impl() {
        let builder = BlendModeBuilder::default();
        let mode = builder.build();
        assert_eq!(mode, BlendMode::alpha());
    }

    #[test]
    fn test_builder_clone() {
        let builder = BlendModeBuilder::new()
            .color_src_factor(wgpu::BlendFactor::One);
        let cloned = builder.clone();
        assert_eq!(builder.build(), cloned.build());
    }
}

// =============================================================================
// CATEGORY 10: REAL-WORLD SCENARIOS - Common Blending Configurations
// =============================================================================

mod real_world_tests {
    use super::*;

    #[test]
    fn test_particle_system_glow() {
        // Particles typically use additive for glow
        let mode = BlendMode::additive();
        assert!(mode.is_additive());
        assert!(mode.can_exceed_one());
    }

    #[test]
    fn test_ui_transparency() {
        // UI uses standard alpha blending
        let mode = BlendMode::alpha();
        assert!(mode.uses_alpha());
        assert!(!mode.is_replace());
    }

    #[test]
    fn test_opaque_geometry() {
        // Opaque geometry uses replace
        let mode = BlendMode::replace();
        assert!(mode.is_replace());
        assert!(!mode.uses_alpha());
    }

    #[test]
    fn test_shadow_overlay() {
        // Shadows use multiply to darken
        let mode = BlendMode::multiply();
        assert!(mode.only_darkens());
    }

    #[test]
    fn test_highlight_overlay() {
        // Highlights use screen to lighten
        let mode = BlendMode::screen();
        assert!(mode.only_lightens());
    }

    #[test]
    fn test_premultiplied_web_textures() {
        // WebGL textures often use premultiplied alpha
        let mode = BlendMode::premultiplied_alpha();
        assert!(mode.is_uniform());
    }

    #[test]
    fn test_light_accumulation_max() {
        // Light accumulation can use max to find brightest
        let mode = BlendMode::max();
        assert_eq!(mode.color.operation, wgpu::BlendOperation::Max);
    }

    #[test]
    fn test_uses_constant_detection() {
        // Fade effects may use blend constants
        let mode = BlendModeBuilder::new()
            .color_src_factor(wgpu::BlendFactor::Constant)
            .build();
        assert!(mode.uses_constant());
    }

    #[test]
    fn test_constant_not_used_by_standard_presets() {
        assert!(!BlendMode::alpha().uses_constant());
        assert!(!BlendMode::additive().uses_constant());
        assert!(!BlendMode::multiply().uses_constant());
    }

    #[test]
    fn test_preset_by_name() {
        let mode = get_blend_mode_preset("Additive");
        assert!(mode.is_some());
        assert!(mode.unwrap().is_additive());
    }

    #[test]
    fn test_preset_names_iteration() {
        let names: Vec<_> = blend_mode_preset_names().collect();
        assert!(names.contains(&"Alpha"));
        assert!(names.contains(&"Additive"));
        assert!(names.contains(&"Multiply"));
        assert!(names.contains(&"Screen"));
    }

    #[test]
    fn test_hdr_presets_can_exceed_one() {
        let hdr: Vec<_> = blend_hdr_presets().collect();
        assert!(!hdr.is_empty());
        for info in &hdr {
            assert!(info.can_exceed_one);
        }
    }

    #[test]
    fn test_alpha_presets_use_alpha() {
        let alpha: Vec<_> = alpha_presets().collect();
        assert!(!alpha.is_empty());
        for info in &alpha {
            assert!(info.uses_alpha);
        }
    }

    #[test]
    fn test_constant_presets_use_constant() {
        let constant: Vec<_> = constant_presets().collect();
        assert!(!constant.is_empty());
        for info in &constant {
            assert!(info.uses_constant);
        }
    }
}

// =============================================================================
// CATEGORY 11: WGPU CONVERSION - To/From wgpu Types
// =============================================================================

mod wgpu_conversion_tests {
    use super::*;

    #[test]
    fn test_into_wgpu_blend_state() {
        let mode = BlendMode::alpha();
        let state: wgpu::BlendState = mode.into();
        assert_eq!(state.color, mode.color);
        assert_eq!(state.alpha, mode.alpha);
    }

    #[test]
    fn test_ref_into_wgpu_blend_state() {
        let mode = BlendMode::additive();
        let state: wgpu::BlendState = (&mode).into();
        assert_eq!(state.color, mode.color);
        assert_eq!(state.alpha, mode.alpha);
    }

    #[test]
    fn test_from_wgpu_blend_state() {
        let state = wgpu::BlendState::ALPHA_BLENDING;
        let mode: BlendMode = state.into();
        assert_eq!(mode.color, state.color);
        assert_eq!(mode.alpha, state.alpha);
    }

    #[test]
    fn test_roundtrip_conversion() {
        let original = BlendMode::premultiplied_alpha();
        let wgpu_state: wgpu::BlendState = original.into();
        let back: BlendMode = wgpu_state.into();
        assert_eq!(original, back);
    }

    #[test]
    fn test_wgpu_replace_matches() {
        let mode = BlendMode::replace();
        let wgpu_state = wgpu::BlendState::REPLACE;
        assert_eq!(mode.color, wgpu_state.color);
        assert_eq!(mode.alpha, wgpu_state.alpha);
    }

    #[test]
    fn test_display_impl() {
        let mode = BlendMode::alpha();
        let display_str = format!("{}", mode);
        assert!(display_str.contains("BlendMode"));
        assert!(display_str.contains("src"));
        assert!(display_str.contains("dst"));
    }
}

// =============================================================================
// CATEGORY 12: MODIFIER METHODS - with_* API
// =============================================================================

mod modifier_tests {
    use super::*;

    #[test]
    fn test_with_color_src_factor() {
        let mode = BlendMode::alpha().with_color_src_factor(wgpu::BlendFactor::One);
        assert_eq!(mode.color.src_factor, wgpu::BlendFactor::One);
    }

    #[test]
    fn test_with_color_dst_factor() {
        let mode = BlendMode::alpha().with_color_dst_factor(wgpu::BlendFactor::Zero);
        assert_eq!(mode.color.dst_factor, wgpu::BlendFactor::Zero);
    }

    #[test]
    fn test_with_color_operation() {
        let mode = BlendMode::alpha().with_color_operation(wgpu::BlendOperation::Max);
        assert_eq!(mode.color.operation, wgpu::BlendOperation::Max);
    }

    #[test]
    fn test_with_alpha_src_factor() {
        let mode = BlendMode::alpha().with_alpha_src_factor(wgpu::BlendFactor::DstAlpha);
        assert_eq!(mode.alpha.src_factor, wgpu::BlendFactor::DstAlpha);
    }

    #[test]
    fn test_with_alpha_dst_factor() {
        let mode = BlendMode::alpha().with_alpha_dst_factor(wgpu::BlendFactor::SrcAlpha);
        assert_eq!(mode.alpha.dst_factor, wgpu::BlendFactor::SrcAlpha);
    }

    #[test]
    fn test_with_alpha_operation() {
        let mode = BlendMode::alpha().with_alpha_operation(wgpu::BlendOperation::Min);
        assert_eq!(mode.alpha.operation, wgpu::BlendOperation::Min);
    }

    #[test]
    fn test_chained_modifiers() {
        let mode = BlendMode::replace()
            .with_color_src_factor(wgpu::BlendFactor::SrcAlpha)
            .with_color_dst_factor(wgpu::BlendFactor::OneMinusSrcAlpha)
            .with_alpha_src_factor(wgpu::BlendFactor::One)
            .with_alpha_dst_factor(wgpu::BlendFactor::OneMinusSrcAlpha);

        assert_eq!(mode.color.src_factor, wgpu::BlendFactor::SrcAlpha);
        assert_eq!(mode.color.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
        assert_eq!(mode.alpha.src_factor, wgpu::BlendFactor::One);
        assert_eq!(mode.alpha.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
    }
}

// =============================================================================
// CATEGORY 13: THREAD SAFETY - Send + Sync
// =============================================================================

mod thread_safety_tests {
    use super::*;

    #[test]
    fn test_blend_mode_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<BlendMode>();
    }

    #[test]
    fn test_blend_mode_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<BlendMode>();
    }

    #[test]
    fn test_blend_mode_copy() {
        let original = BlendMode::alpha();
        let copied = original;
        // Both are valid after copy
        assert_eq!(original, copied);
    }
}

// =============================================================================
// CATEGORY 14: PRESET INFO - BlendModeInfo Metadata
// =============================================================================

mod preset_info_tests {
    use super::*;

    #[test]
    fn test_get_blend_mode_info_alpha() {
        let info = get_blend_mode_info("Alpha");
        assert!(info.is_some());
        let info = info.unwrap();
        assert!(info.uses_alpha);
        assert!(!info.can_exceed_one);
        assert!(info.description.contains("alpha"));
    }

    #[test]
    fn test_get_blend_mode_info_additive() {
        let info = get_blend_mode_info("Additive");
        assert!(info.is_some());
        let info = info.unwrap();
        assert!(info.can_exceed_one);
        assert!(!info.uses_alpha);
    }

    #[test]
    fn test_get_blend_mode_info_multiply() {
        let info = get_blend_mode_info("Multiply");
        assert!(info.is_some());
        let info = info.unwrap();
        assert!(!info.can_exceed_one);
    }

    #[test]
    fn test_get_blend_mode_info_nonexistent() {
        let info = get_blend_mode_info("NonExistent");
        assert!(info.is_none());
    }

    #[test]
    fn test_preset_info_use_cases_not_empty() {
        for preset in BLEND_MODE_PRESETS.iter() {
            assert!(
                !preset.use_cases.is_empty(),
                "{} should have use cases",
                preset.name
            );
        }
    }

    #[test]
    fn test_preset_info_has_equation() {
        for preset in BLEND_MODE_PRESETS.iter() {
            assert!(
                !preset.color_equation.is_empty(),
                "{} should have equation",
                preset.name
            );
        }
    }

    #[test]
    fn test_blend_factor_info_has_description() {
        for factor_info in BLEND_FACTORS.iter() {
            assert!(
                !factor_info.description.is_empty(),
                "{} should have description",
                factor_info.name
            );
        }
    }

    #[test]
    fn test_blend_operation_info_has_formula() {
        for op_info in BLEND_OPERATIONS.iter() {
            assert!(
                !op_info.formula.is_empty(),
                "{} should have formula",
                op_info.name
            );
        }
    }
}
