// SPDX-License-Identifier: MIT
//
// blackbox_color_target.rs -- Blackbox tests for T-WGPU-P3.5.1 Color Target State.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only the documented public types and functions:
//
//   - ColorTarget -- Core struct for color target configuration
//   - ColorTargetBuilder -- Fluent builder API
//   - ColorTargetArray -- Multiple render target (MRT) configuration
//   - ColorTargetInfo -- Preset metadata
//   - ColorTargetError -- Validation errors
//
// PUBLIC FUNCTIONS:
//   - get_color_target_info, get_color_target_preset
//   - hdr_presets, srgb_presets, color_target_preset_names
//
// CONSTANTS:
//   - COLOR_TARGET_PRESETS, MAX_COLOR_ATTACHMENTS
//
// ACCEPTANCE CRITERIA (T-WGPU-P3.5.1):
//   1. Format selection - TextureFormat configuration
//   2. Blend state - Optional BlendState configuration
//   3. Write mask - ColorWrites flags (R, G, B, A, ALL)
//   4. Per-target configuration - Multiple render targets
//
// TEST CATEGORIES:
//   1. API Tests - Public interface, constructors, methods
//   2. Format selection - TextureFormat configuration
//   3. Blend state - Optional BlendState configuration
//   4. Write mask - ColorWrites flags (R, G, B, A, ALL)
//   5. Per-target configuration - Multiple render targets
//   6. Builder API - ColorTargetBuilder fluent interface
//   7. Presets - rgba8_unorm, rgba16_float, bgra8_unorm
//   8. HDR formats - High dynamic range presets
//   9. Real-world scenarios - Common rendering configurations
//   10. wgpu conversion - To ColorTargetState
//
// Total target: 40+ tests

use renderer_backend::render_pipeline::{
    color_target_preset_names, get_color_target_info, get_color_target_preset, hdr_presets,
    srgb_presets, ColorTarget, ColorTargetArray, ColorTargetBuilder, ColorTargetError,
    COLOR_TARGET_PRESETS, MAX_COLOR_ATTACHMENTS,
};

// =============================================================================
// CATEGORY 1: API TESTS - Public Interface
// =============================================================================

mod api_tests {
    use super::*;

    #[test]
    fn test_color_target_is_public() {
        // Verify ColorTarget struct is accessible
        let target = ColorTarget::default();
        assert!(target.format == wgpu::TextureFormat::Rgba8UnormSrgb);
    }

    #[test]
    fn test_color_target_builder_is_public() {
        // Verify ColorTargetBuilder is accessible
        let builder = ColorTargetBuilder::new();
        let target = builder.build_unchecked();
        assert!(target.format == wgpu::TextureFormat::Rgba8UnormSrgb);
    }

    #[test]
    fn test_color_target_array_is_public() {
        // Verify ColorTargetArray is accessible
        let array = ColorTargetArray::new();
        assert!(array.is_empty());
    }

    #[test]
    fn test_color_target_error_is_public() {
        // Verify ColorTargetError is accessible
        let err = ColorTargetError::TooManyTargets { count: 10, max: 8 };
        let msg = format!("{}", err);
        assert!(msg.contains("Too many"));
    }

    #[test]
    fn test_color_target_info_is_public() {
        // Verify ColorTargetInfo is accessible
        let info = &COLOR_TARGET_PRESETS[0];
        assert!(!info.name.is_empty());
    }

    #[test]
    fn test_max_color_attachments_constant() {
        // Verify MAX_COLOR_ATTACHMENTS constant is accessible
        assert_eq!(MAX_COLOR_ATTACHMENTS, 8);
    }

    #[test]
    fn test_color_target_presets_array() {
        // Verify COLOR_TARGET_PRESETS array is accessible
        assert!(COLOR_TARGET_PRESETS.len() >= 10);
    }

    #[test]
    fn test_public_field_access() {
        // Verify public fields are accessible
        let target = ColorTarget::rgba8_unorm();
        let _format = target.format;
        let _blend = target.blend;
        let _write_mask = target.write_mask;
    }
}

// =============================================================================
// CATEGORY 2: FORMAT SELECTION - TextureFormat Configuration
// =============================================================================

mod format_selection_tests {
    use super::*;

    #[test]
    fn test_new_with_format() {
        let target = ColorTarget::new(wgpu::TextureFormat::Rgba16Float);
        assert_eq!(target.format, wgpu::TextureFormat::Rgba16Float);
        assert!(target.blend.is_none());
        assert_eq!(target.write_mask, wgpu::ColorWrites::ALL);
    }

    #[test]
    fn test_default_format() {
        let target = ColorTarget::default();
        assert_eq!(target.format, wgpu::TextureFormat::Rgba8UnormSrgb);
    }

    #[test]
    fn test_from_texture_format() {
        let target: ColorTarget = wgpu::TextureFormat::Bgra8Unorm.into();
        assert_eq!(target.format, wgpu::TextureFormat::Bgra8Unorm);
    }

    #[test]
    fn test_builder_format() {
        let target = ColorTargetBuilder::new()
            .format(wgpu::TextureFormat::Rgba32Float)
            .build_unchecked();
        assert_eq!(target.format, wgpu::TextureFormat::Rgba32Float);
    }

    #[test]
    fn test_various_formats() {
        let formats = [
            wgpu::TextureFormat::R8Unorm,
            wgpu::TextureFormat::Rg8Unorm,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureFormat::R16Float,
            wgpu::TextureFormat::Rg16Float,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureFormat::R32Float,
            wgpu::TextureFormat::Rg32Float,
            wgpu::TextureFormat::Rgba32Float,
            wgpu::TextureFormat::Rgb10a2Unorm,
            wgpu::TextureFormat::Rg11b10Float,
        ];

        for format in formats {
            let target = ColorTarget::new(format);
            assert_eq!(target.format, format);
        }
    }

    #[test]
    fn test_srgb_format_detection() {
        let srgb_target = ColorTarget::rgba8_unorm_srgb();
        assert!(srgb_target.is_srgb());

        let linear_target = ColorTarget::rgba8_unorm();
        assert!(!linear_target.is_srgb());
    }

    #[test]
    fn test_hdr_format_detection() {
        let hdr_target = ColorTarget::rgba16_float();
        assert!(hdr_target.is_hdr());

        let ldr_target = ColorTarget::rgba8_unorm();
        assert!(!ldr_target.is_hdr());
    }

    #[test]
    fn test_bytes_per_pixel() {
        assert_eq!(ColorTarget::r8_unorm().bytes_per_pixel(), Some(1));
        assert_eq!(ColorTarget::rg8_unorm().bytes_per_pixel(), Some(2));
        assert_eq!(ColorTarget::rgba8_unorm().bytes_per_pixel(), Some(4));
        assert_eq!(ColorTarget::rgba16_float().bytes_per_pixel(), Some(8));
        assert_eq!(ColorTarget::rgba32_float().bytes_per_pixel(), Some(16));
    }
}

// =============================================================================
// CATEGORY 3: BLEND STATE - Optional BlendState Configuration
// =============================================================================

mod blend_state_tests {
    use super::*;

    #[test]
    fn test_no_blend_by_default() {
        let target = ColorTarget::rgba8_unorm();
        assert!(!target.has_blend());
        assert!(target.blend.is_none());
    }

    #[test]
    fn test_alpha_blend() {
        let target = ColorTarget::rgba8_unorm().alpha_blend();
        assert!(target.has_blend());

        let blend = target.blend.unwrap();
        assert_eq!(blend.color.src_factor, wgpu::BlendFactor::SrcAlpha);
        assert_eq!(blend.color.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
        assert_eq!(blend.color.operation, wgpu::BlendOperation::Add);
    }

    #[test]
    fn test_premultiplied_alpha_blend() {
        let target = ColorTarget::rgba8_unorm().premultiplied_alpha();
        assert!(target.has_blend());

        let blend = target.blend.unwrap();
        assert_eq!(blend.color.src_factor, wgpu::BlendFactor::One);
        assert_eq!(blend.color.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
    }

    #[test]
    fn test_additive_blend() {
        let target = ColorTarget::rgba8_unorm().additive();
        assert!(target.has_blend());

        let blend = target.blend.unwrap();
        assert_eq!(blend.color.src_factor, wgpu::BlendFactor::One);
        assert_eq!(blend.color.dst_factor, wgpu::BlendFactor::One);
        assert_eq!(blend.color.operation, wgpu::BlendOperation::Add);
    }

    #[test]
    fn test_multiply_blend() {
        let target = ColorTarget::rgba8_unorm().multiply();
        assert!(target.has_blend());

        let blend = target.blend.unwrap();
        assert_eq!(blend.color.src_factor, wgpu::BlendFactor::Dst);
        assert_eq!(blend.color.dst_factor, wgpu::BlendFactor::Zero);
    }

    #[test]
    fn test_replace_blend() {
        let target = ColorTarget::rgba8_unorm().replace();
        assert!(target.has_blend());
        // Replace blend just overwrites
    }

    #[test]
    fn test_no_blend_method() {
        let target = ColorTarget::rgba8_unorm()
            .alpha_blend()
            .no_blend();
        assert!(!target.has_blend());
        assert!(target.blend.is_none());
    }

    #[test]
    fn test_custom_blend_state() {
        let custom = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Src,
                dst_factor: wgpu::BlendFactor::Dst,
                operation: wgpu::BlendOperation::Max,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::Zero,
                operation: wgpu::BlendOperation::Add,
            },
        };

        let target = ColorTarget::rgba8_unorm().blend(custom);
        assert!(target.has_blend());

        let blend = target.blend.unwrap();
        assert_eq!(blend.color.operation, wgpu::BlendOperation::Max);
    }

    #[test]
    fn test_blend_opt_some() {
        let target = ColorTarget::rgba8_unorm()
            .blend_opt(Some(wgpu::BlendState::ALPHA_BLENDING));
        assert!(target.has_blend());
    }

    #[test]
    fn test_blend_opt_none() {
        let target = ColorTarget::rgba8_unorm()
            .alpha_blend()
            .blend_opt(None);
        assert!(!target.has_blend());
    }

    #[test]
    fn test_builder_blend_methods() {
        let alpha = ColorTargetBuilder::new().alpha_blend().build_unchecked();
        assert!(alpha.has_blend());

        let premul = ColorTargetBuilder::new().premultiplied_alpha().build_unchecked();
        assert!(premul.has_blend());

        let additive = ColorTargetBuilder::new().additive().build_unchecked();
        assert!(additive.has_blend());

        let multiply = ColorTargetBuilder::new().multiply().build_unchecked();
        assert!(multiply.has_blend());

        let replace = ColorTargetBuilder::new().replace().build_unchecked();
        assert!(replace.has_blend());

        let no_blend = ColorTargetBuilder::new().alpha_blend().no_blend().build_unchecked();
        assert!(!no_blend.has_blend());
    }

    #[test]
    fn test_format_supports_blend() {
        // Float/unorm formats support blending
        assert!(ColorTarget::rgba8_unorm().format_supports_blend());
        assert!(ColorTarget::rgba16_float().format_supports_blend());
        assert!(ColorTarget::bgra8_unorm().format_supports_blend());

        // Integer formats do not support blending
        let uint_target = ColorTarget::new(wgpu::TextureFormat::Rgba8Uint);
        assert!(!uint_target.format_supports_blend());

        let sint_target = ColorTarget::new(wgpu::TextureFormat::Rgba16Sint);
        assert!(!sint_target.format_supports_blend());
    }
}

// =============================================================================
// CATEGORY 4: WRITE MASK - ColorWrites Flags (R, G, B, A, ALL)
// =============================================================================

mod write_mask_tests {
    use super::*;

    #[test]
    fn test_default_write_mask_all() {
        let target = ColorTarget::rgba8_unorm();
        assert_eq!(target.write_mask, wgpu::ColorWrites::ALL);
        assert!(target.writes_all());
        assert!(target.writes_any());
    }

    #[test]
    fn test_write_all() {
        let target = ColorTarget::rgba8_unorm()
            .write_none()
            .write_all();
        assert_eq!(target.write_mask, wgpu::ColorWrites::ALL);
        assert!(target.writes_all());
    }

    #[test]
    fn test_write_color_rgb_only() {
        let target = ColorTarget::rgba8_unorm().write_color();
        assert_eq!(target.write_mask, wgpu::ColorWrites::COLOR);
        assert!(target.write_mask.contains(wgpu::ColorWrites::RED));
        assert!(target.write_mask.contains(wgpu::ColorWrites::GREEN));
        assert!(target.write_mask.contains(wgpu::ColorWrites::BLUE));
        assert!(!target.write_mask.contains(wgpu::ColorWrites::ALPHA));
        assert!(!target.writes_all());
        assert!(target.writes_any());
    }

    #[test]
    fn test_write_red_only() {
        let target = ColorTarget::rgba8_unorm().write_red();
        assert_eq!(target.write_mask, wgpu::ColorWrites::RED);
        assert!(target.write_mask.contains(wgpu::ColorWrites::RED));
        assert!(!target.write_mask.contains(wgpu::ColorWrites::GREEN));
        assert!(!target.writes_all());
    }

    #[test]
    fn test_write_green_only() {
        let target = ColorTarget::rgba8_unorm().write_green();
        assert_eq!(target.write_mask, wgpu::ColorWrites::GREEN);
        assert!(target.write_mask.contains(wgpu::ColorWrites::GREEN));
        assert!(!target.write_mask.contains(wgpu::ColorWrites::RED));
    }

    #[test]
    fn test_write_blue_only() {
        let target = ColorTarget::rgba8_unorm().write_blue();
        assert_eq!(target.write_mask, wgpu::ColorWrites::BLUE);
        assert!(target.write_mask.contains(wgpu::ColorWrites::BLUE));
    }

    #[test]
    fn test_write_alpha_only() {
        let target = ColorTarget::rgba8_unorm().write_alpha();
        assert_eq!(target.write_mask, wgpu::ColorWrites::ALPHA);
        assert!(target.write_mask.contains(wgpu::ColorWrites::ALPHA));
        assert!(!target.write_mask.contains(wgpu::ColorWrites::RED));
    }

    #[test]
    fn test_write_none() {
        let target = ColorTarget::rgba8_unorm().write_none();
        assert_eq!(target.write_mask, wgpu::ColorWrites::empty());
        assert!(!target.writes_all());
        assert!(!target.writes_any());
    }

    #[test]
    fn test_custom_write_mask() {
        let target = ColorTarget::rgba8_unorm()
            .write_mask(wgpu::ColorWrites::RED | wgpu::ColorWrites::BLUE);
        assert!(target.write_mask.contains(wgpu::ColorWrites::RED));
        assert!(!target.write_mask.contains(wgpu::ColorWrites::GREEN));
        assert!(target.write_mask.contains(wgpu::ColorWrites::BLUE));
        assert!(!target.write_mask.contains(wgpu::ColorWrites::ALPHA));
    }

    #[test]
    fn test_builder_write_mask_methods() {
        let all = ColorTargetBuilder::new().write_all().build_unchecked();
        assert_eq!(all.write_mask, wgpu::ColorWrites::ALL);

        let color = ColorTargetBuilder::new().write_color().build_unchecked();
        assert_eq!(color.write_mask, wgpu::ColorWrites::COLOR);

        let red = ColorTargetBuilder::new().write_red().build_unchecked();
        assert_eq!(red.write_mask, wgpu::ColorWrites::RED);

        let alpha = ColorTargetBuilder::new().write_alpha().build_unchecked();
        assert_eq!(alpha.write_mask, wgpu::ColorWrites::ALPHA);

        let none = ColorTargetBuilder::new().write_none().build_unchecked();
        assert_eq!(none.write_mask, wgpu::ColorWrites::empty());
    }

    #[test]
    fn test_writes_any_vs_writes_all() {
        // All channels
        let all = ColorTarget::rgba8_unorm();
        assert!(all.writes_all());
        assert!(all.writes_any());

        // Some channels
        let some = ColorTarget::rgba8_unorm().write_red();
        assert!(!some.writes_all());
        assert!(some.writes_any());

        // No channels
        let none = ColorTarget::rgba8_unorm().write_none();
        assert!(!none.writes_all());
        assert!(!none.writes_any());
    }
}

// =============================================================================
// CATEGORY 5: PER-TARGET CONFIGURATION - Multiple Render Targets
// =============================================================================

mod mrt_tests {
    use super::*;

    #[test]
    fn test_array_new() {
        let array = ColorTargetArray::new();
        assert!(array.is_empty());
        assert_eq!(array.len(), 0);
    }

    #[test]
    fn test_array_single_target() {
        let array = ColorTargetArray::new()
            .target(ColorTarget::rgba8_unorm());
        assert!(!array.is_empty());
        assert_eq!(array.len(), 1);
    }

    #[test]
    fn test_array_multiple_targets() {
        let array = ColorTargetArray::new()
            .target(ColorTarget::rgba8_unorm())
            .target(ColorTarget::rgba16_float())
            .target(ColorTarget::rg16_float());
        assert_eq!(array.len(), 3);
    }

    #[test]
    fn test_array_target_format() {
        let array = ColorTargetArray::new()
            .target_format(wgpu::TextureFormat::Rgba8Unorm)
            .target_format(wgpu::TextureFormat::Rgba16Float);
        assert_eq!(array.len(), 2);
    }

    #[test]
    fn test_array_null_target() {
        let targets = ColorTargetArray::new()
            .target(ColorTarget::rgba8_unorm())
            .null_target()
            .target(ColorTarget::rgba16_float())
            .build();

        assert_eq!(targets.len(), 3);
        assert!(targets[0].is_some());
        assert!(targets[1].is_none());
        assert!(targets[2].is_some());
    }

    #[test]
    fn test_array_gbuffer() {
        let array = ColorTargetArray::gbuffer()
            .albedo(wgpu::TextureFormat::Rgba8Unorm)
            .normal(wgpu::TextureFormat::Rgba16Float)
            .material(wgpu::TextureFormat::Rgba8Unorm)
            .position(wgpu::TextureFormat::Rgba32Float);
        assert_eq!(array.len(), 4);
    }

    #[test]
    fn test_array_velocity_target() {
        let array = ColorTargetArray::new()
            .target(ColorTarget::rgba8_unorm())
            .velocity(wgpu::TextureFormat::Rg16Float);
        assert_eq!(array.len(), 2);
    }

    #[test]
    fn test_array_bloom_target() {
        let array = ColorTargetArray::new()
            .target(ColorTarget::rgba16_float())
            .bloom(wgpu::TextureFormat::Rg11b10Float);
        assert_eq!(array.len(), 2);
    }

    #[test]
    fn test_array_build() {
        let targets = ColorTargetArray::new()
            .target(ColorTarget::rgba8_unorm())
            .target(ColorTarget::rgba16_float())
            .build();

        assert_eq!(targets.len(), 2);
        assert_eq!(
            targets[0].as_ref().unwrap().format,
            wgpu::TextureFormat::Rgba8Unorm
        );
        assert_eq!(
            targets[1].as_ref().unwrap().format,
            wgpu::TextureFormat::Rgba16Float
        );
    }

    #[test]
    fn test_array_build_wgpu() {
        let targets = ColorTargetArray::new()
            .target(ColorTarget::rgba8_unorm().alpha_blend())
            .target(ColorTarget::rgba16_float())
            .build_wgpu();

        assert_eq!(targets.len(), 2);
        assert!(targets[0].as_ref().unwrap().blend.is_some());
        assert!(targets[1].as_ref().unwrap().blend.is_none());
    }

    #[test]
    fn test_array_validation_success() {
        let array = ColorTargetArray::new()
            .target(ColorTarget::rgba8_unorm())
            .target(ColorTarget::rgba16_float());
        assert!(array.validate().is_ok());
    }

    #[test]
    fn test_array_validation_too_many_targets() {
        let mut array = ColorTargetArray::new();
        for _ in 0..(MAX_COLOR_ATTACHMENTS + 2) {
            array = array.target(ColorTarget::rgba8_unorm());
        }
        let result = array.validate();
        assert!(matches!(
            result,
            Err(ColorTargetError::TooManyTargets { .. })
        ));
    }

    #[test]
    fn test_max_allowed_targets() {
        let mut array = ColorTargetArray::new();
        for _ in 0..MAX_COLOR_ATTACHMENTS {
            array = array.target(ColorTarget::rgba8_unorm());
        }
        assert!(array.validate().is_ok());
        assert_eq!(array.len(), MAX_COLOR_ATTACHMENTS);
    }
}

// =============================================================================
// CATEGORY 6: BUILDER API - ColorTargetBuilder Fluent Interface
// =============================================================================

mod builder_tests {
    use super::*;

    #[test]
    fn test_builder_new() {
        let target = ColorTargetBuilder::new().build_unchecked();
        assert_eq!(target, ColorTarget::default());
    }

    #[test]
    fn test_builder_default() {
        let target = ColorTargetBuilder::default().build_unchecked();
        assert_eq!(target, ColorTarget::default());
    }

    #[test]
    fn test_builder_from_target() {
        let original = ColorTarget::rgba16_float().alpha_blend();
        let target = ColorTargetBuilder::from_target(original.clone())
            .write_color()
            .build_unchecked();

        assert_eq!(target.format, wgpu::TextureFormat::Rgba16Float);
        assert!(target.has_blend());
        assert_eq!(target.write_mask, wgpu::ColorWrites::COLOR);
    }

    #[test]
    fn test_builder_from_preset() {
        let target = ColorTargetBuilder::from_preset(ColorTarget::rgba16_float())
            .alpha_blend()
            .build_unchecked();

        assert_eq!(target.format, wgpu::TextureFormat::Rgba16Float);
        assert!(target.has_blend());
    }

    #[test]
    fn test_builder_chained() {
        let target = ColorTargetBuilder::new()
            .format(wgpu::TextureFormat::Rgba16Float)
            .premultiplied_alpha()
            .write_color()
            .build()
            .expect("Valid configuration should build successfully");

        assert_eq!(target.format, wgpu::TextureFormat::Rgba16Float);
        assert!(target.has_blend());
        assert_eq!(target.write_mask, wgpu::ColorWrites::COLOR);
    }

    #[test]
    fn test_builder_validation_success() {
        let result = ColorTargetBuilder::new()
            .format(wgpu::TextureFormat::Rgba8Unorm)
            .alpha_blend()
            .build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_builder_validation_failure() {
        let result = ColorTargetBuilder::new()
            .format(wgpu::TextureFormat::Rgba8Uint)
            .alpha_blend()
            .build();
        assert!(matches!(
            result,
            Err(ColorTargetError::BlendNotSupported(_))
        ));
    }

    #[test]
    fn test_builder_unchecked_bypasses_validation() {
        // This would fail validation but build_unchecked bypasses it
        let target = ColorTargetBuilder::new()
            .format(wgpu::TextureFormat::Rgba8Uint)
            .alpha_blend()
            .build_unchecked();

        assert_eq!(target.format, wgpu::TextureFormat::Rgba8Uint);
        assert!(target.has_blend());
    }

    #[test]
    fn test_builder_custom_blend() {
        let custom = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Src,
                dst_factor: wgpu::BlendFactor::OneMinusDst,
                operation: wgpu::BlendOperation::Subtract,
            },
            alpha: wgpu::BlendComponent::REPLACE,
        };

        let target = ColorTargetBuilder::new()
            .format(wgpu::TextureFormat::Rgba16Float)
            .blend(custom)
            .build_unchecked();

        let blend = target.blend.unwrap();
        assert_eq!(blend.color.operation, wgpu::BlendOperation::Subtract);
    }
}

// =============================================================================
// CATEGORY 7: PRESETS - rgba8_unorm, rgba16_float, bgra8_unorm
// =============================================================================

mod preset_tests {
    use super::*;

    #[test]
    fn test_rgba8_unorm_preset() {
        let target = ColorTarget::rgba8_unorm();
        assert_eq!(target.format, wgpu::TextureFormat::Rgba8Unorm);
        assert!(!target.is_srgb());
        assert!(!target.is_hdr());
    }

    #[test]
    fn test_rgba8_unorm_srgb_preset() {
        let target = ColorTarget::rgba8_unorm_srgb();
        assert_eq!(target.format, wgpu::TextureFormat::Rgba8UnormSrgb);
        assert!(target.is_srgb());
        assert!(!target.is_hdr());
    }

    #[test]
    fn test_bgra8_unorm_preset() {
        let target = ColorTarget::bgra8_unorm();
        assert_eq!(target.format, wgpu::TextureFormat::Bgra8Unorm);
        assert!(!target.is_srgb());
    }

    #[test]
    fn test_bgra8_unorm_srgb_preset() {
        let target = ColorTarget::bgra8_unorm_srgb();
        assert_eq!(target.format, wgpu::TextureFormat::Bgra8UnormSrgb);
        assert!(target.is_srgb());
    }

    #[test]
    fn test_rgba16_float_preset() {
        let target = ColorTarget::rgba16_float();
        assert_eq!(target.format, wgpu::TextureFormat::Rgba16Float);
        assert!(target.is_hdr());
        assert!(!target.is_srgb());
    }

    #[test]
    fn test_rgba32_float_preset() {
        let target = ColorTarget::rgba32_float();
        assert_eq!(target.format, wgpu::TextureFormat::Rgba32Float);
        assert!(target.is_hdr());
    }

    #[test]
    fn test_rgb10a2_unorm_preset() {
        let target = ColorTarget::rgb10a2_unorm();
        assert_eq!(target.format, wgpu::TextureFormat::Rgb10a2Unorm);
    }

    #[test]
    fn test_rg11b10_float_preset() {
        let target = ColorTarget::rg11b10_float();
        assert_eq!(target.format, wgpu::TextureFormat::Rg11b10Float);
        assert!(target.is_hdr());
    }

    #[test]
    fn test_r8_unorm_preset() {
        let target = ColorTarget::r8_unorm();
        assert_eq!(target.format, wgpu::TextureFormat::R8Unorm);
        assert_eq!(target.bytes_per_pixel(), Some(1));
    }

    #[test]
    fn test_rg8_unorm_preset() {
        let target = ColorTarget::rg8_unorm();
        assert_eq!(target.format, wgpu::TextureFormat::Rg8Unorm);
        assert_eq!(target.bytes_per_pixel(), Some(2));
    }

    #[test]
    fn test_r16_float_preset() {
        let target = ColorTarget::r16_float();
        assert_eq!(target.format, wgpu::TextureFormat::R16Float);
        assert!(target.is_hdr());
        assert_eq!(target.bytes_per_pixel(), Some(2));
    }

    #[test]
    fn test_rg16_float_preset() {
        let target = ColorTarget::rg16_float();
        assert_eq!(target.format, wgpu::TextureFormat::Rg16Float);
        assert!(target.is_hdr());
    }

    #[test]
    fn test_r32_float_preset() {
        let target = ColorTarget::r32_float();
        assert_eq!(target.format, wgpu::TextureFormat::R32Float);
        assert!(target.is_hdr());
    }

    #[test]
    fn test_rg32_float_preset() {
        let target = ColorTarget::rg32_float();
        assert_eq!(target.format, wgpu::TextureFormat::Rg32Float);
        assert!(target.is_hdr());
    }

    #[test]
    fn test_get_preset_by_name() {
        let target = get_color_target_preset("RGBA16 Float");
        assert!(target.is_some());
        assert_eq!(target.unwrap().format, wgpu::TextureFormat::Rgba16Float);
    }

    #[test]
    fn test_get_preset_not_found() {
        let target = get_color_target_preset("NonExistent");
        assert!(target.is_none());
    }
}

// =============================================================================
// CATEGORY 8: HDR FORMATS - High Dynamic Range Presets
// =============================================================================

mod hdr_format_tests {
    use super::*;

    #[test]
    fn test_hdr_presets_iterator() {
        let hdr: Vec<_> = hdr_presets().collect();
        assert!(!hdr.is_empty());
        for info in &hdr {
            assert!(info.is_hdr);
        }
    }

    #[test]
    fn test_srgb_presets_iterator() {
        let srgb: Vec<_> = srgb_presets().collect();
        assert!(!srgb.is_empty());
        for info in &srgb {
            assert!(info.is_srgb);
        }
    }

    #[test]
    fn test_preset_names() {
        let names: Vec<_> = color_target_preset_names().collect();
        assert!(names.contains(&"RGBA8 Unorm"));
        assert!(names.contains(&"RGBA16 Float"));
        assert!(names.contains(&"BGRA8 Unorm sRGB"));
        assert!(names.contains(&"RG11B10 Float"));
    }

    #[test]
    fn test_hdr_format_characteristics() {
        // RGBA16 Float - main HDR format
        let rgba16f = ColorTarget::rgba16_float();
        assert!(rgba16f.is_hdr());
        assert_eq!(rgba16f.bytes_per_pixel(), Some(8));
        assert!(rgba16f.format_supports_blend());

        // RGBA32 Float - high precision HDR
        let rgba32f = ColorTarget::rgba32_float();
        assert!(rgba32f.is_hdr());
        assert_eq!(rgba32f.bytes_per_pixel(), Some(16));

        // RG11B10 Float - compact HDR (no alpha)
        let rg11b10f = ColorTarget::rg11b10_float();
        assert!(rg11b10f.is_hdr());
        assert_eq!(rg11b10f.bytes_per_pixel(), Some(4));
    }

    #[test]
    fn test_get_color_target_info() {
        let info = get_color_target_info("RGBA16 Float");
        assert!(info.is_some());

        let info = info.unwrap();
        assert_eq!(info.name, "RGBA16 Float");
        assert_eq!(info.format, wgpu::TextureFormat::Rgba16Float);
        assert!(info.is_hdr);
        assert!(!info.is_srgb);
        assert_eq!(info.bytes_per_pixel, 8);
        assert!(!info.use_cases.is_empty());
    }

    #[test]
    fn test_color_target_presets_array() {
        assert!(COLOR_TARGET_PRESETS.len() >= 12);

        for preset in &COLOR_TARGET_PRESETS {
            assert!(!preset.name.is_empty());
            assert!(!preset.description.is_empty());
            assert!(preset.bytes_per_pixel > 0);
        }
    }
}

// =============================================================================
// CATEGORY 9: REAL-WORLD SCENARIOS - Common Rendering Configurations
// =============================================================================

mod real_world_tests {
    use super::*;

    /// Test typical swapchain configuration.
    #[test]
    fn test_swapchain_configuration() {
        // Common swapchain formats
        let _bgra_srgb = ColorTarget::bgra8_unorm_srgb();
        let _rgba_srgb = ColorTarget::rgba8_unorm_srgb();
        let _bgra_linear = ColorTarget::bgra8_unorm();

        // Verify they're valid
        assert!(_bgra_srgb.is_valid());
        assert!(_rgba_srgb.is_valid());
        assert!(_bgra_linear.is_valid());
    }

    /// Test deferred rendering G-buffer setup.
    #[test]
    fn test_gbuffer_configuration() {
        let targets = ColorTargetArray::gbuffer()
            .albedo(wgpu::TextureFormat::Rgba8Unorm)        // Color + roughness
            .normal(wgpu::TextureFormat::Rgba16Float)       // World-space normals
            .material(wgpu::TextureFormat::Rgba8Unorm)      // Metallic + AO
            .position(wgpu::TextureFormat::Rgba32Float)     // World position
            .build();

        assert_eq!(targets.len(), 4);
        // G-buffer targets typically don't use blending
        for target in targets.iter().flatten() {
            assert!(!target.has_blend());
        }
    }

    /// Test HDR rendering with bloom extraction.
    #[test]
    fn test_hdr_bloom_configuration() {
        let targets = ColorTargetArray::new()
            .target(ColorTarget::rgba16_float())                // Main HDR buffer
            .target(ColorTarget::rg11b10_float().write_color()) // Bloom brightness
            .build();

        assert_eq!(targets.len(), 2);
        assert!(targets[0].as_ref().unwrap().is_hdr());
        assert!(targets[1].as_ref().unwrap().is_hdr());
    }

    /// Test particle system with additive blending.
    #[test]
    fn test_particle_configuration() {
        let target = ColorTarget::rgba16_float().additive();

        assert!(target.is_hdr());
        assert!(target.has_blend());
        let blend = target.blend.unwrap();
        assert_eq!(blend.color.src_factor, wgpu::BlendFactor::One);
        assert_eq!(blend.color.dst_factor, wgpu::BlendFactor::One);
    }

    /// Test UI rendering with alpha blending.
    #[test]
    fn test_ui_configuration() {
        let target = ColorTarget::bgra8_unorm_srgb()
            .premultiplied_alpha();

        assert!(target.is_srgb());
        assert!(target.has_blend());
    }

    /// Test motion blur velocity buffer.
    #[test]
    fn test_velocity_buffer_configuration() {
        let targets = ColorTargetArray::new()
            .target(ColorTarget::rgba8_unorm())
            .velocity(wgpu::TextureFormat::Rg16Float)
            .build();

        assert_eq!(targets.len(), 2);
        assert_eq!(
            targets[1].as_ref().unwrap().format,
            wgpu::TextureFormat::Rg16Float
        );
    }

    /// Test shadow map (depth only, no color attachment).
    #[test]
    fn test_shadow_map_configuration() {
        // For shadow maps, we might use write_none with a dummy target
        let target = ColorTarget::r8_unorm().write_none();
        assert!(!target.writes_any());
    }

    /// Test tone mapping output.
    #[test]
    fn test_tone_mapping_configuration() {
        // HDR to LDR conversion
        let target = ColorTarget::rgb10a2_unorm();
        assert_eq!(target.format, wgpu::TextureFormat::Rgb10a2Unorm);
        assert_eq!(target.bytes_per_pixel(), Some(4));
    }

    /// Test depth linearization pass.
    #[test]
    fn test_depth_linearization_configuration() {
        let target = ColorTarget::r32_float();
        assert!(target.is_hdr());
        assert_eq!(target.bytes_per_pixel(), Some(4));
    }

    /// Test decal rendering with multiply blend.
    #[test]
    fn test_decal_configuration() {
        let target = ColorTarget::rgba8_unorm().multiply();

        assert!(target.has_blend());
        let blend = target.blend.unwrap();
        assert_eq!(blend.color.src_factor, wgpu::BlendFactor::Dst);
    }
}

// =============================================================================
// CATEGORY 10: WGPU CONVERSION - To ColorTargetState
// =============================================================================

mod wgpu_conversion_tests {
    use super::*;

    #[test]
    fn test_into_wgpu_color_target_state() {
        let target = ColorTarget::rgba16_float()
            .alpha_blend()
            .write_color();

        let wgpu_state: wgpu::ColorTargetState = target.into();

        assert_eq!(wgpu_state.format, wgpu::TextureFormat::Rgba16Float);
        assert!(wgpu_state.blend.is_some());
        assert_eq!(wgpu_state.write_mask, wgpu::ColorWrites::COLOR);
    }

    #[test]
    fn test_from_ref_into_wgpu() {
        let target = ColorTarget::bgra8_unorm_srgb();
        let wgpu_state: wgpu::ColorTargetState = (&target).into();

        assert_eq!(wgpu_state.format, wgpu::TextureFormat::Bgra8UnormSrgb);
    }

    #[test]
    fn test_from_wgpu_color_target_state() {
        let wgpu_state = wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba16Float,
            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::COLOR,
        };

        let target: ColorTarget = wgpu_state.into();

        assert_eq!(target.format, wgpu::TextureFormat::Rgba16Float);
        assert!(target.has_blend());
        assert_eq!(target.write_mask, wgpu::ColorWrites::COLOR);
    }

    #[test]
    fn test_roundtrip_conversion() {
        let original = ColorTarget::rgba16_float()
            .premultiplied_alpha()
            .write_mask(wgpu::ColorWrites::RED | wgpu::ColorWrites::GREEN);

        let wgpu_state: wgpu::ColorTargetState = original.clone().into();
        let roundtrip: ColorTarget = wgpu_state.into();

        assert_eq!(original.format, roundtrip.format);
        assert_eq!(original.blend, roundtrip.blend);
        assert_eq!(original.write_mask, roundtrip.write_mask);
    }

    #[test]
    fn test_array_build_wgpu_conversion() {
        let targets = ColorTargetArray::new()
            .target(ColorTarget::rgba8_unorm().alpha_blend())
            .target(ColorTarget::rgba16_float())
            .null_target()
            .target(ColorTarget::rg16_float())
            .build_wgpu();

        assert_eq!(targets.len(), 4);

        // First target has blend
        let t0 = targets[0].as_ref().unwrap();
        assert!(t0.blend.is_some());
        assert_eq!(t0.format, wgpu::TextureFormat::Rgba8Unorm);

        // Second target no blend
        let t1 = targets[1].as_ref().unwrap();
        assert!(t1.blend.is_none());
        assert_eq!(t1.format, wgpu::TextureFormat::Rgba16Float);

        // Third is null
        assert!(targets[2].is_none());

        // Fourth target
        let t3 = targets[3].as_ref().unwrap();
        assert_eq!(t3.format, wgpu::TextureFormat::Rg16Float);
    }

    #[test]
    fn test_wgpu_state_with_all_fields() {
        let target = ColorTarget::rgba32_float()
            .blend(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Src,
                    dst_factor: wgpu::BlendFactor::OneMinusDst,
                    operation: wgpu::BlendOperation::Max,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::Zero,
                    operation: wgpu::BlendOperation::Add,
                },
            })
            .write_mask(wgpu::ColorWrites::RED | wgpu::ColorWrites::ALPHA);

        let wgpu_state: wgpu::ColorTargetState = target.into();

        assert_eq!(wgpu_state.format, wgpu::TextureFormat::Rgba32Float);

        let blend = wgpu_state.blend.unwrap();
        assert_eq!(blend.color.operation, wgpu::BlendOperation::Max);
        assert_eq!(blend.alpha.operation, wgpu::BlendOperation::Add);

        assert!(wgpu_state.write_mask.contains(wgpu::ColorWrites::RED));
        assert!(wgpu_state.write_mask.contains(wgpu::ColorWrites::ALPHA));
        assert!(!wgpu_state.write_mask.contains(wgpu::ColorWrites::GREEN));
        assert!(!wgpu_state.write_mask.contains(wgpu::ColorWrites::BLUE));
    }
}

// =============================================================================
// ADDITIONAL TESTS: Validation, Error Handling, Traits
// =============================================================================

mod validation_tests {
    use super::*;

    #[test]
    fn test_validate_valid_config() {
        let target = ColorTarget::rgba8_unorm().alpha_blend();
        assert!(target.validate().is_ok());
        assert!(target.is_valid());
    }

    #[test]
    fn test_validate_blend_not_supported() {
        let target = ColorTarget::new(wgpu::TextureFormat::Rgba8Uint).alpha_blend();
        let result = target.validate();

        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(ColorTargetError::BlendNotSupported(wgpu::TextureFormat::Rgba8Uint))
        ));
        assert!(!target.is_valid());
    }

    #[test]
    fn test_validate_no_blend_with_uint_is_ok() {
        let target = ColorTarget::new(wgpu::TextureFormat::Rgba8Uint);
        assert!(target.validate().is_ok());
        assert!(target.is_valid());
    }

    #[test]
    fn test_error_display() {
        let err = ColorTargetError::BlendNotSupported(wgpu::TextureFormat::Rgba8Uint);
        let msg = format!("{}", err);
        assert!(msg.contains("Blend state not supported"));
        assert!(msg.contains("Rgba8Uint"));
    }

    #[test]
    fn test_error_display_too_many() {
        let err = ColorTargetError::TooManyTargets { count: 10, max: 8 };
        let msg = format!("{}", err);
        assert!(msg.contains("Too many color targets"));
        assert!(msg.contains("10"));
        assert!(msg.contains("8"));
    }

    #[test]
    fn test_array_validation_with_invalid_target() {
        let array = ColorTargetArray::new()
            .target(ColorTarget::new(wgpu::TextureFormat::Rgba8Uint).alpha_blend());
        let result = array.validate();

        assert!(matches!(
            result,
            Err(ColorTargetError::ArrayError { index: 0, .. })
        ));
    }
}

mod trait_tests {
    use super::*;

    #[test]
    fn test_clone() {
        let original = ColorTarget::rgba16_float().alpha_blend().write_color();
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn test_partial_eq() {
        let a = ColorTarget::rgba8_unorm();
        let b = ColorTarget::rgba8_unorm();
        let c = ColorTarget::rgba16_float();

        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_debug() {
        let target = ColorTarget::rgba8_unorm().alpha_blend();
        let debug_str = format!("{:?}", target);
        assert!(debug_str.contains("Rgba8Unorm"));
        assert!(debug_str.contains("blend"));
    }

    #[test]
    fn test_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<ColorTarget>();
        assert_sync::<ColorTarget>();
    }
}
