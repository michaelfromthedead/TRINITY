// SPDX-License-Identifier: MIT
//
// blackbox_depth_test.rs -- Blackbox tests for T-WGPU-P3.6.1 Depth Test Config.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only the documented public types and functions:
//
//   - DepthStencilStateDescriptor -- Core struct for depth/stencil configuration
//   - StencilFaceStateDescriptor -- Stencil operations for one face
//   - DepthBiasStateDescriptor -- Depth bias (polygon offset) settings
//   - CompareFunctionInfo -- Compare function metadata
//   - DepthFormatInfo -- Depth format metadata
//   - DepthPresetInfo -- Preset metadata
//
// PUBLIC FUNCTIONS:
//   - get_compare_function_info, get_depth_format_info, get_depth_preset_info
//   - is_depth_format, has_stencil
//
// CONSTANTS:
//   - COMPARE_FUNCTIONS, DEPTH_FORMATS, DEPTH_PRESETS
//
// ACCEPTANCE CRITERIA (T-WGPU-P3.6.1):
//   1. depth_write_enabled - boolean flag
//   2. 8 compare functions - Never, Less, Equal, LessEqual, Greater, NotEqual, GreaterEqual, Always
//   3. Common presets - depth_less(), depth_less_equal(), depth_always(), etc.
//   4. Depth format selection - Depth32Float, Depth24PlusStencil8, etc.
//
// TEST CATEGORIES:
//   1. API Tests - Public interface, constructors, methods
//   2. Compare Function Tests - All 8 functions
//   3. Preset Tests - Each preset method
//   4. Format Tests - Each depth format
//   5. Write Enable Tests - Enable/disable, presets
//   6. Info Helper Tests - All info functions, iteration
//   7. Builder Chain Tests - Method chaining, fluent API
//   8. wgpu Conversion Tests - Into<wgpu::*>
//   9. Real-world Scenario Tests - Shadow maps, transparent, reverse-Z
//   10. Thread Safety Tests - Send + Sync
//
// Total target: 60+ tests

use renderer_backend::render_pipeline::{
    get_compare_function_info, get_depth_format_info, get_depth_preset_info, has_stencil,
    is_depth_format, CompareFunctionInfo, DepthBiasStateDescriptor, DepthFormatInfo,
    DepthPresetInfo, DepthStencilStateDescriptor, StencilFaceStateDescriptor, COMPARE_FUNCTIONS,
    DEPTH_FORMATS, DEPTH_PRESETS,
};

// =============================================================================
// CATEGORY 1: API TESTS - Public Interface
// =============================================================================

mod api_tests {
    use super::*;

    #[test]
    fn test_depth_stencil_state_descriptor_is_public() {
        // Verify DepthStencilStateDescriptor struct is accessible
        let state = DepthStencilStateDescriptor::new()
            .format(wgpu::TextureFormat::Depth32Float);
        assert_eq!(state.format, wgpu::TextureFormat::Depth32Float);
    }

    #[test]
    fn test_stencil_face_state_descriptor_is_public() {
        // Verify StencilFaceStateDescriptor struct is accessible
        let state = StencilFaceStateDescriptor::new();
        assert_eq!(state.compare, wgpu::CompareFunction::Always);
    }

    #[test]
    fn test_depth_bias_state_descriptor_is_public() {
        // Verify DepthBiasStateDescriptor struct is accessible
        let bias = DepthBiasStateDescriptor::new();
        assert_eq!(bias.constant, 0);
    }

    #[test]
    fn test_compare_function_info_is_public() {
        // Verify CompareFunctionInfo struct is accessible
        let info = &COMPARE_FUNCTIONS[0];
        assert!(!info.name.is_empty());
    }

    #[test]
    fn test_depth_format_info_is_public() {
        // Verify DepthFormatInfo struct is accessible
        let info = &DEPTH_FORMATS[0];
        assert!(!info.name.is_empty());
    }

    #[test]
    fn test_depth_preset_info_is_public() {
        // Verify DepthPresetInfo struct is accessible
        let info = &DEPTH_PRESETS[0];
        assert!(!info.name.is_empty());
    }

    #[test]
    fn test_compare_functions_array() {
        // Verify COMPARE_FUNCTIONS array has all 8 functions
        assert_eq!(COMPARE_FUNCTIONS.len(), 8);
    }

    #[test]
    fn test_depth_formats_array() {
        // Verify DEPTH_FORMATS array has supported formats
        assert_eq!(DEPTH_FORMATS.len(), 4);
    }

    #[test]
    fn test_depth_presets_array() {
        // Verify DEPTH_PRESETS array has all presets
        assert_eq!(DEPTH_PRESETS.len(), 11);
    }

    #[test]
    fn test_default_derives() {
        // Verify DepthStencilStateDescriptor has Debug, Clone, PartialEq
        let state = DepthStencilStateDescriptor::depth_less();
        let cloned = state.clone();
        assert_eq!(state, cloned);
        let debug_str = format!("{:?}", state);
        assert!(debug_str.contains("DepthStencilStateDescriptor"));
    }
}

// =============================================================================
// CATEGORY 2: COMPARE FUNCTION TESTS - All 8 Functions
// =============================================================================

mod compare_function_tests {
    use super::*;

    #[test]
    fn test_compare_function_never() {
        let state = DepthStencilStateDescriptor::new().format(wgpu::TextureFormat::Depth32Float)
            .depth_compare(wgpu::CompareFunction::Never);
        assert_eq!(state.depth_compare, wgpu::CompareFunction::Never);
    }

    #[test]
    fn test_compare_function_less() {
        let state = DepthStencilStateDescriptor::new().format(wgpu::TextureFormat::Depth32Float)
            .depth_compare(wgpu::CompareFunction::Less);
        assert_eq!(state.depth_compare, wgpu::CompareFunction::Less);
    }

    #[test]
    fn test_compare_function_equal() {
        let state = DepthStencilStateDescriptor::new().format(wgpu::TextureFormat::Depth32Float)
            .depth_compare(wgpu::CompareFunction::Equal);
        assert_eq!(state.depth_compare, wgpu::CompareFunction::Equal);
    }

    #[test]
    fn test_compare_function_less_equal() {
        let state = DepthStencilStateDescriptor::new().format(wgpu::TextureFormat::Depth32Float)
            .depth_compare(wgpu::CompareFunction::LessEqual);
        assert_eq!(state.depth_compare, wgpu::CompareFunction::LessEqual);
    }

    #[test]
    fn test_compare_function_greater() {
        let state = DepthStencilStateDescriptor::new().format(wgpu::TextureFormat::Depth32Float)
            .depth_compare(wgpu::CompareFunction::Greater);
        assert_eq!(state.depth_compare, wgpu::CompareFunction::Greater);
    }

    #[test]
    fn test_compare_function_not_equal() {
        let state = DepthStencilStateDescriptor::new().format(wgpu::TextureFormat::Depth32Float)
            .depth_compare(wgpu::CompareFunction::NotEqual);
        assert_eq!(state.depth_compare, wgpu::CompareFunction::NotEqual);
    }

    #[test]
    fn test_compare_function_greater_equal() {
        let state = DepthStencilStateDescriptor::new().format(wgpu::TextureFormat::Depth32Float)
            .depth_compare(wgpu::CompareFunction::GreaterEqual);
        assert_eq!(state.depth_compare, wgpu::CompareFunction::GreaterEqual);
    }

    #[test]
    fn test_compare_function_always() {
        let state = DepthStencilStateDescriptor::new().format(wgpu::TextureFormat::Depth32Float)
            .depth_compare(wgpu::CompareFunction::Always);
        assert_eq!(state.depth_compare, wgpu::CompareFunction::Always);
    }

    #[test]
    fn test_compare_function_info_never() {
        let info = get_compare_function_info(wgpu::CompareFunction::Never);
        assert!(info.is_some());
        assert_eq!(info.unwrap().name, "Never");
    }

    #[test]
    fn test_compare_function_info_less() {
        let info = get_compare_function_info(wgpu::CompareFunction::Less);
        assert!(info.is_some());
        assert_eq!(info.unwrap().name, "Less");
    }

    #[test]
    fn test_compare_function_info_equal() {
        let info = get_compare_function_info(wgpu::CompareFunction::Equal);
        assert!(info.is_some());
        assert_eq!(info.unwrap().name, "Equal");
    }

    #[test]
    fn test_compare_function_info_less_equal() {
        let info = get_compare_function_info(wgpu::CompareFunction::LessEqual);
        assert!(info.is_some());
        assert_eq!(info.unwrap().name, "LessEqual");
    }

    #[test]
    fn test_compare_function_info_greater() {
        let info = get_compare_function_info(wgpu::CompareFunction::Greater);
        assert!(info.is_some());
        assert_eq!(info.unwrap().name, "Greater");
    }

    #[test]
    fn test_compare_function_info_not_equal() {
        let info = get_compare_function_info(wgpu::CompareFunction::NotEqual);
        assert!(info.is_some());
        assert_eq!(info.unwrap().name, "NotEqual");
    }

    #[test]
    fn test_compare_function_info_greater_equal() {
        let info = get_compare_function_info(wgpu::CompareFunction::GreaterEqual);
        assert!(info.is_some());
        assert_eq!(info.unwrap().name, "GreaterEqual");
    }

    #[test]
    fn test_compare_function_info_always() {
        let info = get_compare_function_info(wgpu::CompareFunction::Always);
        assert!(info.is_some());
        assert_eq!(info.unwrap().name, "Always");
    }
}

// =============================================================================
// CATEGORY 3: PRESET TESTS - Each Preset Method
// =============================================================================

mod preset_tests {
    use super::*;

    #[test]
    fn test_preset_depth_less() {
        let state = DepthStencilStateDescriptor::depth_less();
        // New API defaults to Depth24PlusStencil8
        assert_eq!(state.format, wgpu::TextureFormat::Depth24PlusStencil8);
        assert_eq!(state.depth_compare, wgpu::CompareFunction::Less);
        assert!(state.depth_write_enabled);
    }

    #[test]
    fn test_preset_depth_less_equal() {
        let state = DepthStencilStateDescriptor::depth_less_equal();
        assert_eq!(state.format, wgpu::TextureFormat::Depth24PlusStencil8);
        assert_eq!(state.depth_compare, wgpu::CompareFunction::LessEqual);
        assert!(state.depth_write_enabled);
    }

    #[test]
    fn test_preset_depth_always() {
        let state = DepthStencilStateDescriptor::depth_always();
        assert_eq!(state.format, wgpu::TextureFormat::Depth24PlusStencil8);
        assert_eq!(state.depth_compare, wgpu::CompareFunction::Always);
        // New API: depth_always() disables depth write
        assert!(!state.depth_write_enabled);
    }

    #[test]
    fn test_preset_depth_greater() {
        let state = DepthStencilStateDescriptor::depth_greater();
        assert_eq!(state.format, wgpu::TextureFormat::Depth24PlusStencil8);
        assert_eq!(state.depth_compare, wgpu::CompareFunction::Greater);
        assert!(state.depth_write_enabled);
    }

    #[test]
    fn test_preset_depth_greater_equal() {
        let state = DepthStencilStateDescriptor::depth_greater_equal();
        assert_eq!(state.format, wgpu::TextureFormat::Depth24PlusStencil8);
        assert_eq!(state.depth_compare, wgpu::CompareFunction::GreaterEqual);
        assert!(state.depth_write_enabled);
    }

    #[test]
    fn test_preset_depth_never() {
        let state = DepthStencilStateDescriptor::depth_never();
        assert_eq!(state.format, wgpu::TextureFormat::Depth24PlusStencil8);
        assert_eq!(state.depth_compare, wgpu::CompareFunction::Never);
        // New API: depth_never() disables depth write
        assert!(!state.depth_write_enabled);
    }

    #[test]
    fn test_preset_depth_read_only() {
        let state = DepthStencilStateDescriptor::depth_read_only();
        assert_eq!(state.format, wgpu::TextureFormat::Depth24PlusStencil8);
        assert_eq!(state.depth_compare, wgpu::CompareFunction::Less);
        assert!(!state.depth_write_enabled);
    }

    #[test]
    fn test_preset_depth_read_only_reverse_z() {
        let state = DepthStencilStateDescriptor::depth_read_only_reverse_z();
        assert_eq!(state.format, wgpu::TextureFormat::Depth24PlusStencil8);
        assert_eq!(state.depth_compare, wgpu::CompareFunction::Greater);
        assert!(!state.depth_write_enabled);
    }

    #[test]
    fn test_preset_transparent() {
        let state = DepthStencilStateDescriptor::transparent();
        assert_eq!(state.depth_compare, wgpu::CompareFunction::LessEqual);
        assert!(!state.depth_write_enabled);
    }

    #[test]
    fn test_preset_shadow_map() {
        let state = DepthStencilStateDescriptor::shadow_map();
        assert_eq!(state.format, wgpu::TextureFormat::Depth32Float);
        assert!(state.depth_write_enabled);
        assert!(state.bias.constant > 0); // Shadow map has depth bias
    }

    #[test]
    fn test_preset_depth_prepass() {
        let state = DepthStencilStateDescriptor::depth_prepass();
        assert_eq!(state.format, wgpu::TextureFormat::Depth32Float);
        assert!(state.depth_write_enabled);
        assert_eq!(state.depth_compare, wgpu::CompareFunction::Less);
    }
}

// =============================================================================
// CATEGORY 4: FORMAT TESTS - Each Depth Format
// =============================================================================

mod format_tests {
    use super::*;

    #[test]
    fn test_format_depth32float() {
        let state = DepthStencilStateDescriptor::new().format(wgpu::TextureFormat::Depth32Float);
        assert_eq!(state.format, wgpu::TextureFormat::Depth32Float);
    }

    #[test]
    fn test_format_depth24_stencil8() {
        let state = DepthStencilStateDescriptor::new();
        assert_eq!(state.format, wgpu::TextureFormat::Depth24PlusStencil8);
    }

    #[test]
    fn test_format_new_with_custom() {
        let state = DepthStencilStateDescriptor::new()
            .format(wgpu::TextureFormat::Depth24Plus);
        assert_eq!(state.format, wgpu::TextureFormat::Depth24Plus);
    }

    #[test]
    fn test_format_info_depth32float() {
        let info = get_depth_format_info(wgpu::TextureFormat::Depth32Float);
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.depth_bits, 32);
        assert_eq!(info.stencil_bits, 0);
    }

    #[test]
    fn test_format_info_depth24plus() {
        let info = get_depth_format_info(wgpu::TextureFormat::Depth24Plus);
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.depth_bits, 24);
        assert_eq!(info.stencil_bits, 0);
    }

    #[test]
    fn test_format_info_depth24plus_stencil8() {
        let info = get_depth_format_info(wgpu::TextureFormat::Depth24PlusStencil8);
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.depth_bits, 24);
        assert_eq!(info.stencil_bits, 8);
    }

    #[test]
    fn test_format_info_depth32float_stencil8() {
        let info = get_depth_format_info(wgpu::TextureFormat::Depth32FloatStencil8);
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.depth_bits, 32);
        assert_eq!(info.stencil_bits, 8);
    }

    #[test]
    fn test_is_depth_format_valid() {
        assert!(is_depth_format(wgpu::TextureFormat::Depth32Float));
        assert!(is_depth_format(wgpu::TextureFormat::Depth24Plus));
        assert!(is_depth_format(wgpu::TextureFormat::Depth24PlusStencil8));
        assert!(is_depth_format(wgpu::TextureFormat::Depth32FloatStencil8));
    }

    #[test]
    fn test_is_depth_format_invalid() {
        assert!(!is_depth_format(wgpu::TextureFormat::Rgba8Unorm));
        assert!(!is_depth_format(wgpu::TextureFormat::Bgra8Unorm));
        assert!(!is_depth_format(wgpu::TextureFormat::R32Float));
    }

    #[test]
    fn test_has_stencil_true() {
        assert!(has_stencil(wgpu::TextureFormat::Depth24PlusStencil8));
        assert!(has_stencil(wgpu::TextureFormat::Depth32FloatStencil8));
    }

    #[test]
    fn test_has_stencil_false() {
        assert!(!has_stencil(wgpu::TextureFormat::Depth32Float));
        assert!(!has_stencil(wgpu::TextureFormat::Depth24Plus));
        assert!(!has_stencil(wgpu::TextureFormat::Rgba8Unorm));
    }
}

// =============================================================================
// CATEGORY 5: WRITE ENABLE TESTS - Enable/Disable, Presets
// =============================================================================

mod write_enable_tests {
    use super::*;

    #[test]
    fn test_depth_write_enabled_default() {
        let state = DepthStencilStateDescriptor::new().format(wgpu::TextureFormat::Depth32Float);
        assert!(state.depth_write_enabled);
    }

    #[test]
    fn test_depth_write_enabled_explicit_true() {
        let state = DepthStencilStateDescriptor::new().format(wgpu::TextureFormat::Depth32Float).depth_write_enabled(true);
        assert!(state.depth_write_enabled);
    }

    #[test]
    fn test_depth_write_enabled_explicit_false() {
        let state = DepthStencilStateDescriptor::new().format(wgpu::TextureFormat::Depth32Float).depth_write_enabled(false);
        assert!(!state.depth_write_enabled);
    }

    #[test]
    fn test_no_depth_write_method() {
        let state = DepthStencilStateDescriptor::new().format(wgpu::TextureFormat::Depth32Float).depth_write_enabled(false);
        assert!(!state.depth_write_enabled);
    }

    #[test]
    fn test_depth_write_in_transparent_preset() {
        let state = DepthStencilStateDescriptor::transparent();
        assert!(!state.depth_write_enabled);
    }

    #[test]
    fn test_depth_write_in_shadow_map_preset() {
        let state = DepthStencilStateDescriptor::shadow_map();
        assert!(state.depth_write_enabled);
    }
}

// =============================================================================
// CATEGORY 6: INFO HELPER TESTS - All Info Functions, Iteration
// =============================================================================

mod info_helper_tests {
    use super::*;

    #[test]
    fn test_get_compare_function_info_all() {
        // All 8 compare functions should have info
        let functions = [
            wgpu::CompareFunction::Never,
            wgpu::CompareFunction::Less,
            wgpu::CompareFunction::Equal,
            wgpu::CompareFunction::LessEqual,
            wgpu::CompareFunction::Greater,
            wgpu::CompareFunction::NotEqual,
            wgpu::CompareFunction::GreaterEqual,
            wgpu::CompareFunction::Always,
        ];

        for func in functions {
            let info = get_compare_function_info(func);
            assert!(info.is_some(), "Missing info for {:?}", func);
            assert_eq!(info.unwrap().function, func);
        }
    }

    #[test]
    fn test_compare_function_info_has_description() {
        for info in &COMPARE_FUNCTIONS {
            assert!(!info.description.is_empty(), "Missing description for {}", info.name);
        }
    }

    #[test]
    fn test_compare_function_info_has_use_cases() {
        for info in &COMPARE_FUNCTIONS {
            assert!(!info.use_cases.is_empty(), "Missing use cases for {}", info.name);
        }
    }

    #[test]
    fn test_get_depth_format_info_all() {
        let formats = [
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureFormat::Depth24Plus,
            wgpu::TextureFormat::Depth24PlusStencil8,
            wgpu::TextureFormat::Depth32FloatStencil8,
        ];

        for format in formats {
            let info = get_depth_format_info(format);
            assert!(info.is_some(), "Missing info for {:?}", format);
            assert_eq!(info.unwrap().format, format);
        }
    }

    #[test]
    fn test_depth_format_info_has_description() {
        for info in &DEPTH_FORMATS {
            assert!(!info.description.is_empty(), "Missing description for {}", info.name);
        }
    }

    #[test]
    fn test_depth_format_info_valid_depth_bits() {
        for info in &DEPTH_FORMATS {
            assert!(info.depth_bits >= 24, "Depth bits should be at least 24 for {}", info.name);
        }
    }

    #[test]
    fn test_get_depth_preset_info_all() {
        let preset_names = [
            "depth_less",
            "depth_less_equal",
            "depth_always",
            "depth_greater",
            "depth_greater_equal",
            "depth_never",
            "depth_read_only",
            "depth_read_only_reverse_z",
            "transparent",
            "shadow_map",
            "depth_prepass",
        ];

        for name in preset_names {
            let info = get_depth_preset_info(name);
            assert!(info.is_some(), "Missing info for preset: {}", name);
            assert_eq!(info.unwrap().name, name);
        }
    }

    #[test]
    fn test_get_depth_preset_info_not_found() {
        let info = get_depth_preset_info("nonexistent_preset");
        assert!(info.is_none());
    }

    #[test]
    fn test_depth_preset_info_has_description() {
        for info in &DEPTH_PRESETS {
            assert!(!info.description.is_empty(), "Missing description for {}", info.name);
        }
    }

    #[test]
    fn test_depth_preset_info_has_use_cases() {
        for info in &DEPTH_PRESETS {
            assert!(!info.use_cases.is_empty(), "Missing use cases for {}", info.name);
        }
    }

    #[test]
    fn test_iterate_compare_functions() {
        let mut count = 0;
        for info in &COMPARE_FUNCTIONS {
            assert!(!info.name.is_empty());
            count += 1;
        }
        assert_eq!(count, 8);
    }

    #[test]
    fn test_iterate_depth_formats() {
        let mut count = 0;
        for info in &DEPTH_FORMATS {
            assert!(info.depth_bits > 0);
            count += 1;
        }
        assert_eq!(count, 4);
    }
}

// =============================================================================
// CATEGORY 7: BUILDER CHAIN TESTS - Method Chaining, Fluent API
// =============================================================================

mod builder_chain_tests {
    use super::*;

    #[test]
    fn test_builder_chain_depth_compare() {
        let state = DepthStencilStateDescriptor::new().format(wgpu::TextureFormat::Depth32Float)
            .depth_compare(wgpu::CompareFunction::Greater);
        assert_eq!(state.depth_compare, wgpu::CompareFunction::Greater);
    }

    #[test]
    fn test_builder_chain_depth_write() {
        let state = DepthStencilStateDescriptor::new().format(wgpu::TextureFormat::Depth32Float)
            .depth_write_enabled(false);
        assert!(!state.depth_write_enabled);
    }

    #[test]
    fn test_builder_chain_reverse_z() {
        let state = DepthStencilStateDescriptor::new().format(wgpu::TextureFormat::Depth32Float).depth_compare(wgpu::CompareFunction::Greater);
        assert_eq!(state.depth_compare, wgpu::CompareFunction::Greater);
    }

    #[test]
    fn test_builder_chain_stencil() {
        let stencil = StencilFaceStateDescriptor::new()
            .compare(wgpu::CompareFunction::Equal)
            .pass_op(wgpu::StencilOperation::Replace);
        let state = DepthStencilStateDescriptor::new().stencil_both(stencil);
        assert_eq!(state.stencil_front.compare, wgpu::CompareFunction::Equal);
        assert_eq!(state.stencil_back.compare, wgpu::CompareFunction::Equal);
    }

    #[test]
    fn test_builder_chain_stencil_masks() {
        let state = DepthStencilStateDescriptor::new()
            .stencil_read_mask(0x0F).stencil_write_mask(0xF0);
        assert_eq!(state.stencil_read_mask, 0x0F);
        assert_eq!(state.stencil_write_mask, 0xF0);
    }

    #[test]
    fn test_builder_chain_bias() {
        let bias = DepthBiasStateDescriptor::new()
            .constant(10)
            .slope_scale(1.5);
        let state = DepthStencilStateDescriptor::new().format(wgpu::TextureFormat::Depth32Float).bias(bias);
        assert_eq!(state.bias.constant, 10);
    }

    #[test]
    fn test_builder_chain_multiple() {
        let state = DepthStencilStateDescriptor::new().format(wgpu::TextureFormat::Depth32Float)
            .depth_compare(wgpu::CompareFunction::LessEqual)
            .depth_write_enabled(true)
            .stencil_read_mask(0xFF).stencil_write_mask(0xFF);
        assert_eq!(state.depth_compare, wgpu::CompareFunction::LessEqual);
        assert!(state.depth_write_enabled);
        assert_eq!(state.stencil_read_mask, 0xFF);
    }

    #[test]
    fn test_stencil_face_builder_chain() {
        let state = StencilFaceStateDescriptor::new()
            .compare(wgpu::CompareFunction::NotEqual)
            .fail_op(wgpu::StencilOperation::Zero)
            .depth_fail_op(wgpu::StencilOperation::Invert)
            .pass_op(wgpu::StencilOperation::Replace);

        assert_eq!(state.compare, wgpu::CompareFunction::NotEqual);
        assert_eq!(state.fail_op, wgpu::StencilOperation::Zero);
        assert_eq!(state.depth_fail_op, wgpu::StencilOperation::Invert);
        assert_eq!(state.pass_op, wgpu::StencilOperation::Replace);
    }

    #[test]
    fn test_depth_bias_builder_chain() {
        let bias = DepthBiasStateDescriptor::new()
            .constant(100)
            .slope_scale(2.5)
            .clamp(0.01);

        assert_eq!(bias.constant, 100);
        assert!((bias.slope_scale - 2.5).abs() < f32::EPSILON);
        assert!((bias.clamp - 0.01).abs() < f32::EPSILON);
    }
}

// =============================================================================
// CATEGORY 8: WGPU CONVERSION TESTS - Into<wgpu::*>
// =============================================================================

mod wgpu_conversion_tests {
    use super::*;

    #[test]
    fn test_stencil_face_into_wgpu() {
        let face = StencilFaceStateDescriptor::new()
            .compare(wgpu::CompareFunction::Equal)
            .fail_op(wgpu::StencilOperation::Zero)
            .depth_fail_op(wgpu::StencilOperation::Invert)
            .pass_op(wgpu::StencilOperation::Replace);

        let wgpu_face: wgpu::StencilFaceState = face.into();

        assert_eq!(wgpu_face.compare, wgpu::CompareFunction::Equal);
        assert_eq!(wgpu_face.fail_op, wgpu::StencilOperation::Zero);
        assert_eq!(wgpu_face.depth_fail_op, wgpu::StencilOperation::Invert);
        assert_eq!(wgpu_face.pass_op, wgpu::StencilOperation::Replace);
    }

    #[test]
    fn test_depth_bias_into_wgpu() {
        let bias = DepthBiasStateDescriptor::new()
            .constant(50)
            .slope_scale(3.0)
            .clamp(0.05);

        let wgpu_bias: wgpu::DepthBiasState = bias.into();

        assert_eq!(wgpu_bias.constant, 50);
        assert!((wgpu_bias.slope_scale - 3.0).abs() < f32::EPSILON);
        assert!((wgpu_bias.clamp - 0.05).abs() < f32::EPSILON);
    }

    #[test]
    fn test_stencil_face_default_into_wgpu() {
        let face = StencilFaceStateDescriptor::default();
        let wgpu_face: wgpu::StencilFaceState = face.into();

        assert_eq!(wgpu_face.compare, wgpu::CompareFunction::Always);
        assert_eq!(wgpu_face.fail_op, wgpu::StencilOperation::Keep);
        assert_eq!(wgpu_face.depth_fail_op, wgpu::StencilOperation::Keep);
        assert_eq!(wgpu_face.pass_op, wgpu::StencilOperation::Keep);
    }

    #[test]
    fn test_depth_bias_default_into_wgpu() {
        let bias = DepthBiasStateDescriptor::default();
        let wgpu_bias: wgpu::DepthBiasState = bias.into();

        assert_eq!(wgpu_bias.constant, 0);
        assert!((wgpu_bias.slope_scale - 0.0).abs() < f32::EPSILON);
        assert!((wgpu_bias.clamp - 0.0).abs() < f32::EPSILON);
    }
}

// =============================================================================
// CATEGORY 9: REAL-WORLD SCENARIO TESTS
// =============================================================================

mod real_world_tests {
    use super::*;

    #[test]
    fn test_scenario_shadow_map_rendering() {
        // Shadow maps need depth bias to prevent acne
        let state = DepthStencilStateDescriptor::shadow_map();
        assert!(state.depth_write_enabled);
        assert_eq!(state.depth_compare, wgpu::CompareFunction::Less);
        assert!(state.bias.constant > 0);
        assert!(state.bias.slope_scale > 0.0);
    }

    #[test]
    fn test_scenario_transparent_objects() {
        // Transparent objects should read but not write depth
        let state = DepthStencilStateDescriptor::transparent();
        assert!(!state.depth_write_enabled);
        assert_eq!(state.depth_compare, wgpu::CompareFunction::LessEqual);
    }

    #[test]
    fn test_scenario_reverse_z_pipeline() {
        // Reverse-Z for better precision in large scenes
        let state = DepthStencilStateDescriptor::depth_greater();
        assert_eq!(state.depth_compare, wgpu::CompareFunction::Greater);
        assert!(state.depth_write_enabled);
    }

    #[test]
    fn test_scenario_depth_prepass() {
        // Early depth pass for occlusion culling
        let state = DepthStencilStateDescriptor::depth_prepass();
        assert!(state.depth_write_enabled);
        assert_eq!(state.depth_compare, wgpu::CompareFunction::Less);
    }

    #[test]
    fn test_scenario_decal_rendering() {
        // Decals use LessEqual to render at exact depth
        let state = DepthStencilStateDescriptor::depth_less_equal();
        assert_eq!(state.depth_compare, wgpu::CompareFunction::LessEqual);
    }

    #[test]
    fn test_scenario_stencil_masking() {
        // Stencil buffer for masking/portal effects
        let stencil = StencilFaceStateDescriptor::new()
            .compare(wgpu::CompareFunction::Always)
            .pass_op(wgpu::StencilOperation::Replace);

        let state = DepthStencilStateDescriptor::new()
            .stencil_both(stencil)
            .stencil_read_mask(0xFF).stencil_write_mask(0xFF);

        assert_eq!(state.format, wgpu::TextureFormat::Depth24PlusStencil8);
        assert_eq!(state.stencil_front.compare, wgpu::CompareFunction::Always);
        assert_eq!(state.stencil_front.pass_op, wgpu::StencilOperation::Replace);
    }

    #[test]
    fn test_scenario_particles() {
        // Particles typically read depth but don't write
        let state = DepthStencilStateDescriptor::depth_read_only();
        assert!(!state.depth_write_enabled);
        assert_eq!(state.depth_compare, wgpu::CompareFunction::Less);
    }

    #[test]
    fn test_scenario_skybox() {
        // Skybox renders at max depth, uses LessEqual
        let state = DepthStencilStateDescriptor::new().format(wgpu::TextureFormat::Depth32Float)
            .depth_compare(wgpu::CompareFunction::LessEqual)
            .depth_write_enabled(false);
        assert!(!state.depth_write_enabled);
        assert_eq!(state.depth_compare, wgpu::CompareFunction::LessEqual);
    }
}

// =============================================================================
// CATEGORY 10: THREAD SAFETY TESTS - Send + Sync
// =============================================================================

mod thread_safety_tests {
    use super::*;

    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    #[test]
    fn test_depth_stencil_state_descriptor_send() {
        assert_send::<DepthStencilStateDescriptor>();
    }

    #[test]
    fn test_depth_stencil_state_descriptor_sync() {
        assert_sync::<DepthStencilStateDescriptor>();
    }

    #[test]
    fn test_stencil_face_state_descriptor_send_sync() {
        assert_send::<StencilFaceStateDescriptor>();
        assert_sync::<StencilFaceStateDescriptor>();
    }

    #[test]
    fn test_depth_bias_state_descriptor_send_sync() {
        assert_send::<DepthBiasStateDescriptor>();
        assert_sync::<DepthBiasStateDescriptor>();
    }

    #[test]
    fn test_compare_function_info_send_sync() {
        assert_send::<CompareFunctionInfo>();
        assert_sync::<CompareFunctionInfo>();
    }

    #[test]
    fn test_depth_format_info_send_sync() {
        assert_send::<DepthFormatInfo>();
        assert_sync::<DepthFormatInfo>();
    }

    #[test]
    fn test_depth_preset_info_send_sync() {
        assert_send::<DepthPresetInfo>();
        assert_sync::<DepthPresetInfo>();
    }
}

// =============================================================================
// ADDITIONAL EDGE CASE TESTS
// =============================================================================

mod edge_case_tests {
    use super::*;

    #[test]
    fn test_stencil_masks_zero() {
        // Zero masks effectively disable stencil
        let state = DepthStencilStateDescriptor::new()
            .stencil_read_mask(0x00).stencil_write_mask(0x00);
        assert_eq!(state.stencil_read_mask, 0x00);
        assert_eq!(state.stencil_write_mask, 0x00);
    }

    #[test]
    fn test_stencil_masks_full() {
        let state = DepthStencilStateDescriptor::new()
            .stencil_read_mask(0xFF).stencil_write_mask(0xFF);
        assert_eq!(state.stencil_read_mask, 0xFF);
        assert_eq!(state.stencil_write_mask, 0xFF);
    }

    #[test]
    fn test_stencil_masks_partial() {
        let state = DepthStencilStateDescriptor::new()
            .stencil_read_mask(0x0F).stencil_write_mask(0xF0);
        assert_eq!(state.stencil_read_mask, 0x0F);
        assert_eq!(state.stencil_write_mask, 0xF0);
    }

    #[test]
    fn test_depth_bias_negative() {
        let bias = DepthBiasStateDescriptor::new()
            .constant(-10)
            .slope_scale(-1.5);
        assert_eq!(bias.constant, -10);
        assert!((bias.slope_scale - (-1.5)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_depth_bias_shadow_map_preset() {
        let bias = DepthBiasStateDescriptor::shadow_map();
        assert_eq!(bias.constant, 2);
        assert!((bias.slope_scale - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_separate_front_back_stencil() {
        let front = StencilFaceStateDescriptor::new()
            .compare(wgpu::CompareFunction::Always)
            .pass_op(wgpu::StencilOperation::IncrementClamp);

        let back = StencilFaceStateDescriptor::new()
            .compare(wgpu::CompareFunction::Never)
            .pass_op(wgpu::StencilOperation::DecrementClamp);

        let state = DepthStencilStateDescriptor::new()
            .stencil_front(front)
            .stencil_back(back);

        assert_eq!(state.stencil_front.compare, wgpu::CompareFunction::Always);
        assert_eq!(state.stencil_back.compare, wgpu::CompareFunction::Never);
    }

    #[test]
    fn test_equality_same_config() {
        let state1 = DepthStencilStateDescriptor::new().format(wgpu::TextureFormat::Depth32Float);
        let state2 = DepthStencilStateDescriptor::new().format(wgpu::TextureFormat::Depth32Float);
        assert_eq!(state1, state2);
    }

    #[test]
    fn test_equality_different_format() {
        let state1 = DepthStencilStateDescriptor::new().format(wgpu::TextureFormat::Depth32Float);
        let state2 = DepthStencilStateDescriptor::new();
        assert_ne!(state1, state2);
    }

    #[test]
    fn test_equality_different_compare() {
        let state1 = DepthStencilStateDescriptor::depth_less();
        let state2 = DepthStencilStateDescriptor::depth_greater();
        assert_ne!(state1, state2);
    }

    #[test]
    fn test_all_stencil_operations() {
        let ops = [
            wgpu::StencilOperation::Keep,
            wgpu::StencilOperation::Zero,
            wgpu::StencilOperation::Replace,
            wgpu::StencilOperation::Invert,
            wgpu::StencilOperation::IncrementClamp,
            wgpu::StencilOperation::DecrementClamp,
            wgpu::StencilOperation::IncrementWrap,
            wgpu::StencilOperation::DecrementWrap,
        ];

        for op in ops {
            let state = StencilFaceStateDescriptor::new()
                .fail_op(op)
                .depth_fail_op(op)
                .pass_op(op);
            assert_eq!(state.fail_op, op);
            assert_eq!(state.depth_fail_op, op);
            assert_eq!(state.pass_op, op);
        }
    }
}
