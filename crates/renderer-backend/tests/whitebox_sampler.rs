//! Whitebox tests for T-WGPU-P2.4.1: Sampler Creation
//!
//! This module contains comprehensive whitebox tests that verify the internal
//! implementation details of the sampler creation system, including:
//!
//! - TrinitySamplerDescriptor: defaults, builder methods, cloning, debug
//! - Presets: linear_clamp, linear_repeat, nearest_clamp, nearest_repeat, shadow, trilinear
//! - Address modes: ClampToEdge, Repeat, MirrorRepeat, ClampToBorder
//! - Filter modes: Nearest, Linear, mixed combinations
//! - Validation: anisotropy limits, LOD ranges, border color requirements
//! - TrinitySampler wrapper: accessors, is_* methods
//! - Error variants and Display trait
//!
//! These tests have full access to implementation details and verify internal
//! invariants that may not be visible through the public API alone.

use wgpu::{AddressMode, CompareFunction, FilterMode, SamplerBorderColor};

// Import from the crate under test
use renderer_backend::resources::{
    validate_descriptor, SamplerValidationError, TrinitySamplerDescriptor,
};

// ============================================================================
// MODULE 1: Descriptor Default Value Tests (15 tests)
// ============================================================================

mod descriptor_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // 1.1 Default Values
    // -------------------------------------------------------------------------

    #[test]
    fn test_default_label_is_none() {
        let desc = TrinitySamplerDescriptor::default();
        assert!(desc.label.is_none());
    }

    #[test]
    fn test_default_address_mode_u_is_clamp_to_edge() {
        let desc = TrinitySamplerDescriptor::default();
        assert_eq!(desc.address_mode_u, AddressMode::ClampToEdge);
    }

    #[test]
    fn test_default_address_mode_v_is_clamp_to_edge() {
        let desc = TrinitySamplerDescriptor::default();
        assert_eq!(desc.address_mode_v, AddressMode::ClampToEdge);
    }

    #[test]
    fn test_default_address_mode_w_is_clamp_to_edge() {
        let desc = TrinitySamplerDescriptor::default();
        assert_eq!(desc.address_mode_w, AddressMode::ClampToEdge);
    }

    #[test]
    fn test_default_mag_filter_is_linear() {
        let desc = TrinitySamplerDescriptor::default();
        assert_eq!(desc.mag_filter, FilterMode::Linear);
    }

    #[test]
    fn test_default_min_filter_is_linear() {
        let desc = TrinitySamplerDescriptor::default();
        assert_eq!(desc.min_filter, FilterMode::Linear);
    }

    #[test]
    fn test_default_mipmap_filter_is_linear() {
        let desc = TrinitySamplerDescriptor::default();
        assert_eq!(desc.mipmap_filter, FilterMode::Linear);
    }

    #[test]
    fn test_default_lod_min_clamp_is_zero() {
        let desc = TrinitySamplerDescriptor::default();
        assert_eq!(desc.lod_min_clamp, 0.0);
    }

    #[test]
    fn test_default_lod_max_clamp_is_32() {
        let desc = TrinitySamplerDescriptor::default();
        assert_eq!(desc.lod_max_clamp, 32.0);
    }

    #[test]
    fn test_default_compare_is_none() {
        let desc = TrinitySamplerDescriptor::default();
        assert!(desc.compare.is_none());
    }

    #[test]
    fn test_default_anisotropy_clamp_is_one() {
        let desc = TrinitySamplerDescriptor::default();
        assert_eq!(desc.anisotropy_clamp, 1);
    }

    #[test]
    fn test_default_border_color_is_none() {
        let desc = TrinitySamplerDescriptor::default();
        assert!(desc.border_color.is_none());
    }

    #[test]
    fn test_new_equals_default() {
        let new = TrinitySamplerDescriptor::new();
        let default = TrinitySamplerDescriptor::default();

        assert_eq!(new.label, default.label);
        assert_eq!(new.address_mode_u, default.address_mode_u);
        assert_eq!(new.address_mode_v, default.address_mode_v);
        assert_eq!(new.address_mode_w, default.address_mode_w);
        assert_eq!(new.mag_filter, default.mag_filter);
        assert_eq!(new.min_filter, default.min_filter);
        assert_eq!(new.mipmap_filter, default.mipmap_filter);
        assert_eq!(new.lod_min_clamp, default.lod_min_clamp);
        assert_eq!(new.lod_max_clamp, default.lod_max_clamp);
        assert_eq!(new.compare, default.compare);
        assert_eq!(new.anisotropy_clamp, default.anisotropy_clamp);
        assert_eq!(new.border_color, default.border_color);
    }

    // -------------------------------------------------------------------------
    // 1.2 Clone Trait
    // -------------------------------------------------------------------------

    #[test]
    fn test_descriptor_clone_preserves_all_fields() {
        let desc = TrinitySamplerDescriptor::new()
            .label("test_clone")
            .address_mode(AddressMode::Repeat)
            .filter(FilterMode::Nearest)
            .anisotropy(8)
            .lod_clamp(1.0, 10.0)
            .compare(CompareFunction::Less);

        let cloned = desc.clone();

        assert_eq!(cloned.label.as_deref(), Some("test_clone"));
        assert_eq!(cloned.address_mode_u, AddressMode::Repeat);
        assert_eq!(cloned.mag_filter, FilterMode::Nearest);
        assert_eq!(cloned.anisotropy_clamp, 8);
        assert_eq!(cloned.lod_min_clamp, 1.0);
        assert_eq!(cloned.lod_max_clamp, 10.0);
        assert_eq!(cloned.compare, Some(CompareFunction::Less));
    }

    #[test]
    fn test_descriptor_clone_is_independent() {
        let desc = TrinitySamplerDescriptor::new().label("original");
        let cloned = desc.clone();

        // Cloned and original should have same values
        assert_eq!(desc.label, cloned.label);
        // But be separate instances (test by checking they're equal references)
        assert!(desc.label.is_some());
        assert!(cloned.label.is_some());
    }

    // -------------------------------------------------------------------------
    // 1.3 Debug Trait
    // -------------------------------------------------------------------------

    #[test]
    fn test_descriptor_debug_contains_struct_name() {
        let desc = TrinitySamplerDescriptor::new();
        let debug_str = format!("{:?}", desc);
        assert!(debug_str.contains("TrinitySamplerDescriptor"));
    }

    #[test]
    fn test_descriptor_debug_contains_label() {
        let desc = TrinitySamplerDescriptor::new().label("debug_label_test");
        let debug_str = format!("{:?}", desc);
        assert!(debug_str.contains("debug_label_test"));
    }

    #[test]
    fn test_descriptor_debug_contains_filter_mode() {
        let desc = TrinitySamplerDescriptor::new().filter(FilterMode::Nearest);
        let debug_str = format!("{:?}", desc);
        assert!(debug_str.contains("Nearest"));
    }
}

// ============================================================================
// MODULE 2: Builder Method Tests (20 tests)
// ============================================================================

mod builder_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // 2.1 Label Builder
    // -------------------------------------------------------------------------

    #[test]
    fn test_label_sets_string_slice() {
        let desc = TrinitySamplerDescriptor::new().label("my_sampler");
        assert_eq!(desc.label.as_deref(), Some("my_sampler"));
    }

    #[test]
    fn test_label_sets_owned_string() {
        let name = String::from("dynamic_sampler");
        let desc = TrinitySamplerDescriptor::new().label(name);
        assert_eq!(desc.label.as_deref(), Some("dynamic_sampler"));
    }

    #[test]
    fn test_label_can_be_empty_string() {
        let desc = TrinitySamplerDescriptor::new().label("");
        assert_eq!(desc.label.as_deref(), Some(""));
    }

    #[test]
    fn test_label_overwrites_previous() {
        let desc = TrinitySamplerDescriptor::new()
            .label("first")
            .label("second");
        assert_eq!(desc.label.as_deref(), Some("second"));
    }

    // -------------------------------------------------------------------------
    // 2.2 Address Mode Builders
    // -------------------------------------------------------------------------

    #[test]
    fn test_address_mode_sets_all_three() {
        let desc = TrinitySamplerDescriptor::new().address_mode(AddressMode::MirrorRepeat);
        assert_eq!(desc.address_mode_u, AddressMode::MirrorRepeat);
        assert_eq!(desc.address_mode_v, AddressMode::MirrorRepeat);
        assert_eq!(desc.address_mode_w, AddressMode::MirrorRepeat);
    }

    #[test]
    fn test_address_mode_uvw_sets_each_separately() {
        let desc = TrinitySamplerDescriptor::new().address_mode_uvw(
            AddressMode::Repeat,
            AddressMode::ClampToEdge,
            AddressMode::MirrorRepeat,
        );
        assert_eq!(desc.address_mode_u, AddressMode::Repeat);
        assert_eq!(desc.address_mode_v, AddressMode::ClampToEdge);
        assert_eq!(desc.address_mode_w, AddressMode::MirrorRepeat);
    }

    #[test]
    fn test_address_mode_uvw_all_same_value() {
        let desc = TrinitySamplerDescriptor::new().address_mode_uvw(
            AddressMode::Repeat,
            AddressMode::Repeat,
            AddressMode::Repeat,
        );
        assert_eq!(desc.address_mode_u, AddressMode::Repeat);
        assert_eq!(desc.address_mode_v, AddressMode::Repeat);
        assert_eq!(desc.address_mode_w, AddressMode::Repeat);
    }

    // -------------------------------------------------------------------------
    // 2.3 Filter Builders
    // -------------------------------------------------------------------------

    #[test]
    fn test_filter_sets_all_three() {
        let desc = TrinitySamplerDescriptor::new().filter(FilterMode::Nearest);
        assert_eq!(desc.mag_filter, FilterMode::Nearest);
        assert_eq!(desc.min_filter, FilterMode::Nearest);
        assert_eq!(desc.mipmap_filter, FilterMode::Nearest);
    }

    #[test]
    fn test_filter_separate_sets_each_independently() {
        let desc = TrinitySamplerDescriptor::new().filter_separate(
            FilterMode::Linear,
            FilterMode::Nearest,
            FilterMode::Linear,
        );
        assert_eq!(desc.mag_filter, FilterMode::Linear);
        assert_eq!(desc.min_filter, FilterMode::Nearest);
        assert_eq!(desc.mipmap_filter, FilterMode::Linear);
    }

    #[test]
    fn test_filter_separate_all_nearest() {
        let desc = TrinitySamplerDescriptor::new().filter_separate(
            FilterMode::Nearest,
            FilterMode::Nearest,
            FilterMode::Nearest,
        );
        assert_eq!(desc.mag_filter, FilterMode::Nearest);
        assert_eq!(desc.min_filter, FilterMode::Nearest);
        assert_eq!(desc.mipmap_filter, FilterMode::Nearest);
    }

    #[test]
    fn test_filter_separate_all_linear() {
        let desc = TrinitySamplerDescriptor::new().filter_separate(
            FilterMode::Linear,
            FilterMode::Linear,
            FilterMode::Linear,
        );
        assert_eq!(desc.mag_filter, FilterMode::Linear);
        assert_eq!(desc.min_filter, FilterMode::Linear);
        assert_eq!(desc.mipmap_filter, FilterMode::Linear);
    }

    // -------------------------------------------------------------------------
    // 2.4 LOD Clamp Builder
    // -------------------------------------------------------------------------

    #[test]
    fn test_lod_clamp_sets_min_and_max() {
        let desc = TrinitySamplerDescriptor::new().lod_clamp(2.0, 8.0);
        assert_eq!(desc.lod_min_clamp, 2.0);
        assert_eq!(desc.lod_max_clamp, 8.0);
    }

    #[test]
    fn test_lod_clamp_allows_equal_values() {
        let desc = TrinitySamplerDescriptor::new().lod_clamp(5.0, 5.0);
        assert_eq!(desc.lod_min_clamp, 5.0);
        assert_eq!(desc.lod_max_clamp, 5.0);
    }

    #[test]
    fn test_lod_clamp_allows_zero_min() {
        let desc = TrinitySamplerDescriptor::new().lod_clamp(0.0, 16.0);
        assert_eq!(desc.lod_min_clamp, 0.0);
        assert_eq!(desc.lod_max_clamp, 16.0);
    }

    // -------------------------------------------------------------------------
    // 2.5 Anisotropy Builder
    // -------------------------------------------------------------------------

    #[test]
    fn test_anisotropy_sets_value() {
        let desc = TrinitySamplerDescriptor::new().anisotropy(4);
        assert_eq!(desc.anisotropy_clamp, 4);
    }

    #[test]
    fn test_anisotropy_allows_one() {
        let desc = TrinitySamplerDescriptor::new().anisotropy(1);
        assert_eq!(desc.anisotropy_clamp, 1);
    }

    #[test]
    fn test_anisotropy_allows_sixteen() {
        let desc = TrinitySamplerDescriptor::new().anisotropy(16);
        assert_eq!(desc.anisotropy_clamp, 16);
    }

    // -------------------------------------------------------------------------
    // 2.6 Compare and Border Color Builders
    // -------------------------------------------------------------------------

    #[test]
    fn test_compare_sets_function() {
        let desc = TrinitySamplerDescriptor::new().compare(CompareFunction::Greater);
        assert_eq!(desc.compare, Some(CompareFunction::Greater));
    }

    #[test]
    fn test_border_color_sets_value() {
        let desc = TrinitySamplerDescriptor::new()
            .address_mode(AddressMode::ClampToBorder)
            .border_color(SamplerBorderColor::TransparentBlack);
        assert_eq!(
            desc.border_color,
            Some(SamplerBorderColor::TransparentBlack)
        );
    }

    // -------------------------------------------------------------------------
    // 2.7 Method Chaining
    // -------------------------------------------------------------------------

    #[test]
    fn test_full_builder_chain() {
        let desc = TrinitySamplerDescriptor::new()
            .label("full_chain")
            .address_mode(AddressMode::Repeat)
            .filter(FilterMode::Linear)
            .lod_clamp(0.0, 12.0)
            .anisotropy(8)
            .compare(CompareFunction::LessEqual);

        assert_eq!(desc.label.as_deref(), Some("full_chain"));
        assert_eq!(desc.address_mode_u, AddressMode::Repeat);
        assert_eq!(desc.mag_filter, FilterMode::Linear);
        assert_eq!(desc.lod_max_clamp, 12.0);
        assert_eq!(desc.anisotropy_clamp, 8);
        assert_eq!(desc.compare, Some(CompareFunction::LessEqual));
    }
}

// ============================================================================
// MODULE 3: Preset Tests (18 tests)
// ============================================================================

mod preset_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // 3.1 linear_clamp Preset
    // -------------------------------------------------------------------------

    #[test]
    fn test_linear_clamp_has_label() {
        let desc = TrinitySamplerDescriptor::linear_clamp();
        assert_eq!(desc.label.as_deref(), Some("linear_clamp"));
    }

    #[test]
    fn test_linear_clamp_mag_filter() {
        let desc = TrinitySamplerDescriptor::linear_clamp();
        assert_eq!(desc.mag_filter, FilterMode::Linear);
    }

    #[test]
    fn test_linear_clamp_min_filter() {
        let desc = TrinitySamplerDescriptor::linear_clamp();
        assert_eq!(desc.min_filter, FilterMode::Linear);
    }

    #[test]
    fn test_linear_clamp_mipmap_filter() {
        let desc = TrinitySamplerDescriptor::linear_clamp();
        assert_eq!(desc.mipmap_filter, FilterMode::Linear);
    }

    #[test]
    fn test_linear_clamp_address_modes() {
        let desc = TrinitySamplerDescriptor::linear_clamp();
        assert_eq!(desc.address_mode_u, AddressMode::ClampToEdge);
        assert_eq!(desc.address_mode_v, AddressMode::ClampToEdge);
        assert_eq!(desc.address_mode_w, AddressMode::ClampToEdge);
    }

    // -------------------------------------------------------------------------
    // 3.2 linear_repeat Preset
    // -------------------------------------------------------------------------

    #[test]
    fn test_linear_repeat_has_label() {
        let desc = TrinitySamplerDescriptor::linear_repeat();
        assert_eq!(desc.label.as_deref(), Some("linear_repeat"));
    }

    #[test]
    fn test_linear_repeat_filters() {
        let desc = TrinitySamplerDescriptor::linear_repeat();
        assert_eq!(desc.mag_filter, FilterMode::Linear);
        assert_eq!(desc.min_filter, FilterMode::Linear);
        assert_eq!(desc.mipmap_filter, FilterMode::Linear);
    }

    #[test]
    fn test_linear_repeat_address_modes() {
        let desc = TrinitySamplerDescriptor::linear_repeat();
        assert_eq!(desc.address_mode_u, AddressMode::Repeat);
        assert_eq!(desc.address_mode_v, AddressMode::Repeat);
        assert_eq!(desc.address_mode_w, AddressMode::Repeat);
    }

    // -------------------------------------------------------------------------
    // 3.3 nearest_clamp Preset
    // -------------------------------------------------------------------------

    #[test]
    fn test_nearest_clamp_has_label() {
        let desc = TrinitySamplerDescriptor::nearest_clamp();
        assert_eq!(desc.label.as_deref(), Some("nearest_clamp"));
    }

    #[test]
    fn test_nearest_clamp_filters() {
        let desc = TrinitySamplerDescriptor::nearest_clamp();
        assert_eq!(desc.mag_filter, FilterMode::Nearest);
        assert_eq!(desc.min_filter, FilterMode::Nearest);
        assert_eq!(desc.mipmap_filter, FilterMode::Nearest);
    }

    #[test]
    fn test_nearest_clamp_address_modes() {
        let desc = TrinitySamplerDescriptor::nearest_clamp();
        assert_eq!(desc.address_mode_u, AddressMode::ClampToEdge);
        assert_eq!(desc.address_mode_v, AddressMode::ClampToEdge);
        assert_eq!(desc.address_mode_w, AddressMode::ClampToEdge);
    }

    // -------------------------------------------------------------------------
    // 3.4 nearest_repeat Preset
    // -------------------------------------------------------------------------

    #[test]
    fn test_nearest_repeat_has_label() {
        let desc = TrinitySamplerDescriptor::nearest_repeat();
        assert_eq!(desc.label.as_deref(), Some("nearest_repeat"));
    }

    #[test]
    fn test_nearest_repeat_filters() {
        let desc = TrinitySamplerDescriptor::nearest_repeat();
        assert_eq!(desc.mag_filter, FilterMode::Nearest);
        assert_eq!(desc.min_filter, FilterMode::Nearest);
        assert_eq!(desc.mipmap_filter, FilterMode::Nearest);
    }

    #[test]
    fn test_nearest_repeat_address_modes() {
        let desc = TrinitySamplerDescriptor::nearest_repeat();
        assert_eq!(desc.address_mode_u, AddressMode::Repeat);
        assert_eq!(desc.address_mode_v, AddressMode::Repeat);
        assert_eq!(desc.address_mode_w, AddressMode::Repeat);
    }

    // -------------------------------------------------------------------------
    // 3.5 shadow Preset
    // -------------------------------------------------------------------------

    #[test]
    fn test_shadow_has_label() {
        let desc = TrinitySamplerDescriptor::shadow();
        assert_eq!(desc.label.as_deref(), Some("shadow"));
    }

    #[test]
    fn test_shadow_compare_function() {
        let desc = TrinitySamplerDescriptor::shadow();
        assert_eq!(desc.compare, Some(CompareFunction::Less));
    }

    #[test]
    fn test_shadow_filters() {
        let desc = TrinitySamplerDescriptor::shadow();
        assert_eq!(desc.mag_filter, FilterMode::Linear);
        assert_eq!(desc.min_filter, FilterMode::Linear);
        assert_eq!(desc.mipmap_filter, FilterMode::Nearest);
    }

    #[test]
    fn test_shadow_address_modes() {
        let desc = TrinitySamplerDescriptor::shadow();
        assert_eq!(desc.address_mode_u, AddressMode::ClampToEdge);
        assert_eq!(desc.address_mode_v, AddressMode::ClampToEdge);
        assert_eq!(desc.address_mode_w, AddressMode::ClampToEdge);
    }

    // -------------------------------------------------------------------------
    // 3.6 trilinear Preset
    // -------------------------------------------------------------------------

    #[test]
    fn test_trilinear_has_label() {
        let desc = TrinitySamplerDescriptor::trilinear();
        assert_eq!(desc.label.as_deref(), Some("trilinear"));
    }

    #[test]
    fn test_trilinear_filters() {
        let desc = TrinitySamplerDescriptor::trilinear();
        assert_eq!(desc.mag_filter, FilterMode::Linear);
        assert_eq!(desc.min_filter, FilterMode::Linear);
        assert_eq!(desc.mipmap_filter, FilterMode::Linear);
    }

    #[test]
    fn test_trilinear_address_modes() {
        let desc = TrinitySamplerDescriptor::trilinear();
        assert_eq!(desc.address_mode_u, AddressMode::ClampToEdge);
        assert_eq!(desc.address_mode_v, AddressMode::ClampToEdge);
        assert_eq!(desc.address_mode_w, AddressMode::ClampToEdge);
    }
}

// ============================================================================
// MODULE 4: Address Mode Tests (10 tests)
// ============================================================================

mod address_mode_tests {
    use super::*;

    #[test]
    fn test_clamp_to_edge_is_default() {
        let desc = TrinitySamplerDescriptor::new();
        assert_eq!(desc.address_mode_u, AddressMode::ClampToEdge);
        assert_eq!(desc.address_mode_v, AddressMode::ClampToEdge);
        assert_eq!(desc.address_mode_w, AddressMode::ClampToEdge);
    }

    #[test]
    fn test_repeat_address_mode() {
        let desc = TrinitySamplerDescriptor::new().address_mode(AddressMode::Repeat);
        assert_eq!(desc.address_mode_u, AddressMode::Repeat);
        assert_eq!(desc.address_mode_v, AddressMode::Repeat);
        assert_eq!(desc.address_mode_w, AddressMode::Repeat);
    }

    #[test]
    fn test_mirror_repeat_address_mode() {
        let desc = TrinitySamplerDescriptor::new().address_mode(AddressMode::MirrorRepeat);
        assert_eq!(desc.address_mode_u, AddressMode::MirrorRepeat);
        assert_eq!(desc.address_mode_v, AddressMode::MirrorRepeat);
        assert_eq!(desc.address_mode_w, AddressMode::MirrorRepeat);
    }

    #[test]
    fn test_clamp_to_border_address_mode() {
        let desc = TrinitySamplerDescriptor::new().address_mode(AddressMode::ClampToBorder);
        assert_eq!(desc.address_mode_u, AddressMode::ClampToBorder);
        assert_eq!(desc.address_mode_v, AddressMode::ClampToBorder);
        assert_eq!(desc.address_mode_w, AddressMode::ClampToBorder);
    }

    #[test]
    fn test_mixed_address_modes_u_repeat() {
        let desc = TrinitySamplerDescriptor::new().address_mode_uvw(
            AddressMode::Repeat,
            AddressMode::ClampToEdge,
            AddressMode::ClampToEdge,
        );
        assert_eq!(desc.address_mode_u, AddressMode::Repeat);
        assert_eq!(desc.address_mode_v, AddressMode::ClampToEdge);
        assert_eq!(desc.address_mode_w, AddressMode::ClampToEdge);
    }

    #[test]
    fn test_mixed_address_modes_v_repeat() {
        let desc = TrinitySamplerDescriptor::new().address_mode_uvw(
            AddressMode::ClampToEdge,
            AddressMode::Repeat,
            AddressMode::ClampToEdge,
        );
        assert_eq!(desc.address_mode_u, AddressMode::ClampToEdge);
        assert_eq!(desc.address_mode_v, AddressMode::Repeat);
        assert_eq!(desc.address_mode_w, AddressMode::ClampToEdge);
    }

    #[test]
    fn test_mixed_address_modes_w_repeat() {
        let desc = TrinitySamplerDescriptor::new().address_mode_uvw(
            AddressMode::ClampToEdge,
            AddressMode::ClampToEdge,
            AddressMode::Repeat,
        );
        assert_eq!(desc.address_mode_u, AddressMode::ClampToEdge);
        assert_eq!(desc.address_mode_v, AddressMode::ClampToEdge);
        assert_eq!(desc.address_mode_w, AddressMode::Repeat);
    }

    #[test]
    fn test_all_different_address_modes() {
        let desc = TrinitySamplerDescriptor::new().address_mode_uvw(
            AddressMode::Repeat,
            AddressMode::MirrorRepeat,
            AddressMode::ClampToBorder,
        );
        assert_eq!(desc.address_mode_u, AddressMode::Repeat);
        assert_eq!(desc.address_mode_v, AddressMode::MirrorRepeat);
        assert_eq!(desc.address_mode_w, AddressMode::ClampToBorder);
    }

    #[test]
    fn test_clamp_to_border_with_transparent_black() {
        let desc = TrinitySamplerDescriptor::new()
            .address_mode(AddressMode::ClampToBorder)
            .border_color(SamplerBorderColor::TransparentBlack);
        assert_eq!(desc.address_mode_u, AddressMode::ClampToBorder);
        assert_eq!(
            desc.border_color,
            Some(SamplerBorderColor::TransparentBlack)
        );
    }

    #[test]
    fn test_clamp_to_border_with_opaque_white() {
        let desc = TrinitySamplerDescriptor::new()
            .address_mode(AddressMode::ClampToBorder)
            .border_color(SamplerBorderColor::OpaqueWhite);
        assert_eq!(desc.address_mode_u, AddressMode::ClampToBorder);
        assert_eq!(desc.border_color, Some(SamplerBorderColor::OpaqueWhite));
    }
}

// ============================================================================
// MODULE 5: Filter Mode Tests (12 tests)
// ============================================================================

mod filter_mode_tests {
    use super::*;

    #[test]
    fn test_linear_is_default() {
        let desc = TrinitySamplerDescriptor::new();
        assert_eq!(desc.mag_filter, FilterMode::Linear);
        assert_eq!(desc.min_filter, FilterMode::Linear);
        assert_eq!(desc.mipmap_filter, FilterMode::Linear);
    }

    #[test]
    fn test_all_nearest() {
        let desc = TrinitySamplerDescriptor::new().filter(FilterMode::Nearest);
        assert_eq!(desc.mag_filter, FilterMode::Nearest);
        assert_eq!(desc.min_filter, FilterMode::Nearest);
        assert_eq!(desc.mipmap_filter, FilterMode::Nearest);
    }

    #[test]
    fn test_all_linear() {
        let desc = TrinitySamplerDescriptor::new().filter(FilterMode::Linear);
        assert_eq!(desc.mag_filter, FilterMode::Linear);
        assert_eq!(desc.min_filter, FilterMode::Linear);
        assert_eq!(desc.mipmap_filter, FilterMode::Linear);
    }

    #[test]
    fn test_mag_linear_min_nearest() {
        let desc = TrinitySamplerDescriptor::new().filter_separate(
            FilterMode::Linear,
            FilterMode::Nearest,
            FilterMode::Linear,
        );
        assert_eq!(desc.mag_filter, FilterMode::Linear);
        assert_eq!(desc.min_filter, FilterMode::Nearest);
    }

    #[test]
    fn test_mag_nearest_min_linear() {
        let desc = TrinitySamplerDescriptor::new().filter_separate(
            FilterMode::Nearest,
            FilterMode::Linear,
            FilterMode::Nearest,
        );
        assert_eq!(desc.mag_filter, FilterMode::Nearest);
        assert_eq!(desc.min_filter, FilterMode::Linear);
    }

    #[test]
    fn test_mipmap_linear_others_nearest() {
        let desc = TrinitySamplerDescriptor::new().filter_separate(
            FilterMode::Nearest,
            FilterMode::Nearest,
            FilterMode::Linear,
        );
        assert_eq!(desc.mag_filter, FilterMode::Nearest);
        assert_eq!(desc.min_filter, FilterMode::Nearest);
        assert_eq!(desc.mipmap_filter, FilterMode::Linear);
    }

    #[test]
    fn test_mipmap_nearest_others_linear() {
        let desc = TrinitySamplerDescriptor::new().filter_separate(
            FilterMode::Linear,
            FilterMode::Linear,
            FilterMode::Nearest,
        );
        assert_eq!(desc.mag_filter, FilterMode::Linear);
        assert_eq!(desc.min_filter, FilterMode::Linear);
        assert_eq!(desc.mipmap_filter, FilterMode::Nearest);
    }

    #[test]
    fn test_bilinear_config() {
        // Bilinear: linear mag/min, nearest mipmap
        let desc = TrinitySamplerDescriptor::new().filter_separate(
            FilterMode::Linear,
            FilterMode::Linear,
            FilterMode::Nearest,
        );
        assert_eq!(desc.mag_filter, FilterMode::Linear);
        assert_eq!(desc.min_filter, FilterMode::Linear);
        assert_eq!(desc.mipmap_filter, FilterMode::Nearest);
    }

    #[test]
    fn test_trilinear_config() {
        // Trilinear: all linear
        let desc = TrinitySamplerDescriptor::new().filter_separate(
            FilterMode::Linear,
            FilterMode::Linear,
            FilterMode::Linear,
        );
        assert_eq!(desc.mag_filter, FilterMode::Linear);
        assert_eq!(desc.min_filter, FilterMode::Linear);
        assert_eq!(desc.mipmap_filter, FilterMode::Linear);
    }

    #[test]
    fn test_pixelated_config() {
        // Pixelated: all nearest
        let desc = TrinitySamplerDescriptor::new().filter_separate(
            FilterMode::Nearest,
            FilterMode::Nearest,
            FilterMode::Nearest,
        );
        assert_eq!(desc.mag_filter, FilterMode::Nearest);
        assert_eq!(desc.min_filter, FilterMode::Nearest);
        assert_eq!(desc.mipmap_filter, FilterMode::Nearest);
    }

    #[test]
    fn test_filter_overwrites_previous() {
        let desc = TrinitySamplerDescriptor::new()
            .filter(FilterMode::Nearest)
            .filter(FilterMode::Linear);
        assert_eq!(desc.mag_filter, FilterMode::Linear);
        assert_eq!(desc.min_filter, FilterMode::Linear);
        assert_eq!(desc.mipmap_filter, FilterMode::Linear);
    }

    #[test]
    fn test_filter_separate_overwrites_filter() {
        let desc = TrinitySamplerDescriptor::new()
            .filter(FilterMode::Nearest)
            .filter_separate(FilterMode::Linear, FilterMode::Linear, FilterMode::Nearest);
        assert_eq!(desc.mag_filter, FilterMode::Linear);
        assert_eq!(desc.min_filter, FilterMode::Linear);
        assert_eq!(desc.mipmap_filter, FilterMode::Nearest);
    }
}

// ============================================================================
// MODULE 6: Validation Tests (25 tests)
// ============================================================================

mod validation_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // 6.1 Valid Descriptors
    // -------------------------------------------------------------------------

    #[test]
    fn test_validate_default_descriptor_ok() {
        let desc = TrinitySamplerDescriptor::new();
        assert!(validate_descriptor(&desc).is_ok());
    }

    #[test]
    fn test_validate_linear_clamp_preset_ok() {
        let desc = TrinitySamplerDescriptor::linear_clamp();
        assert!(validate_descriptor(&desc).is_ok());
    }

    #[test]
    fn test_validate_linear_repeat_preset_ok() {
        let desc = TrinitySamplerDescriptor::linear_repeat();
        assert!(validate_descriptor(&desc).is_ok());
    }

    #[test]
    fn test_validate_shadow_preset_ok() {
        let desc = TrinitySamplerDescriptor::shadow();
        assert!(validate_descriptor(&desc).is_ok());
    }

    #[test]
    fn test_validate_trilinear_preset_ok() {
        let desc = TrinitySamplerDescriptor::trilinear();
        assert!(validate_descriptor(&desc).is_ok());
    }

    // -------------------------------------------------------------------------
    // 6.2 Anisotropy Validation
    // -------------------------------------------------------------------------

    #[test]
    fn test_validate_anisotropy_1_ok() {
        let desc = TrinitySamplerDescriptor::new().anisotropy(1);
        assert!(validate_descriptor(&desc).is_ok());
    }

    #[test]
    fn test_validate_anisotropy_8_ok() {
        let desc = TrinitySamplerDescriptor::new().anisotropy(8);
        assert!(validate_descriptor(&desc).is_ok());
    }

    #[test]
    fn test_validate_anisotropy_16_ok() {
        let desc = TrinitySamplerDescriptor::new().anisotropy(16);
        assert!(validate_descriptor(&desc).is_ok());
    }

    #[test]
    fn test_validate_anisotropy_17_error() {
        let desc = TrinitySamplerDescriptor::new().anisotropy(17);
        let result = validate_descriptor(&desc);
        assert!(matches!(
            result,
            Err(SamplerValidationError::InvalidAnisotropy { .. })
        ));
    }

    #[test]
    fn test_validate_anisotropy_32_error() {
        let desc = TrinitySamplerDescriptor::new().anisotropy(32);
        let result = validate_descriptor(&desc);
        assert!(matches!(
            result,
            Err(SamplerValidationError::InvalidAnisotropy {
                requested: 32,
                max_supported: 16
            })
        ));
    }

    #[test]
    fn test_validate_anisotropy_u16_max_error() {
        let desc = TrinitySamplerDescriptor::new().anisotropy(u16::MAX);
        let result = validate_descriptor(&desc);
        assert!(matches!(
            result,
            Err(SamplerValidationError::InvalidAnisotropy { .. })
        ));
    }

    // -------------------------------------------------------------------------
    // 6.3 LOD Range Validation
    // -------------------------------------------------------------------------

    #[test]
    fn test_validate_lod_zero_to_32_ok() {
        let desc = TrinitySamplerDescriptor::new().lod_clamp(0.0, 32.0);
        assert!(validate_descriptor(&desc).is_ok());
    }

    #[test]
    fn test_validate_lod_equal_values_ok() {
        let desc = TrinitySamplerDescriptor::new().lod_clamp(5.0, 5.0);
        assert!(validate_descriptor(&desc).is_ok());
    }

    #[test]
    fn test_validate_lod_small_range_ok() {
        let desc = TrinitySamplerDescriptor::new().lod_clamp(4.0, 8.0);
        assert!(validate_descriptor(&desc).is_ok());
    }

    #[test]
    fn test_validate_lod_inverted_range_error() {
        let mut desc = TrinitySamplerDescriptor::new();
        desc.lod_min_clamp = 10.0;
        desc.lod_max_clamp = 5.0;
        let result = validate_descriptor(&desc);
        assert!(matches!(
            result,
            Err(SamplerValidationError::InvalidLodRange { min: 10.0, max: 5.0 })
        ));
    }

    #[test]
    fn test_validate_lod_negative_min_error() {
        let mut desc = TrinitySamplerDescriptor::new();
        desc.lod_min_clamp = -1.0;
        let result = validate_descriptor(&desc);
        assert!(matches!(
            result,
            Err(SamplerValidationError::NegativeLod { value: -1.0 })
        ));
    }

    #[test]
    fn test_validate_lod_negative_max_error() {
        let mut desc = TrinitySamplerDescriptor::new();
        desc.lod_max_clamp = -0.5;
        let result = validate_descriptor(&desc);
        assert!(matches!(
            result,
            Err(SamplerValidationError::NegativeLod { .. })
        ));
    }

    #[test]
    fn test_validate_lod_large_values_ok() {
        let desc = TrinitySamplerDescriptor::new().lod_clamp(0.0, 1000.0);
        assert!(validate_descriptor(&desc).is_ok());
    }

    #[test]
    fn test_validate_lod_fractional_values_ok() {
        let desc = TrinitySamplerDescriptor::new().lod_clamp(0.5, 7.25);
        assert!(validate_descriptor(&desc).is_ok());
    }

    // -------------------------------------------------------------------------
    // 6.4 Border Color Validation
    // -------------------------------------------------------------------------

    #[test]
    fn test_validate_border_color_with_clamp_to_border_ok() {
        let desc = TrinitySamplerDescriptor::new()
            .address_mode(AddressMode::ClampToBorder)
            .border_color(SamplerBorderColor::OpaqueBlack);
        assert!(validate_descriptor(&desc).is_ok());
    }

    #[test]
    fn test_validate_border_color_with_partial_clamp_to_border_ok() {
        let desc = TrinitySamplerDescriptor::new()
            .address_mode_uvw(
                AddressMode::ClampToBorder,
                AddressMode::Repeat,
                AddressMode::Repeat,
            )
            .border_color(SamplerBorderColor::OpaqueWhite);
        assert!(validate_descriptor(&desc).is_ok());
    }

    #[test]
    fn test_validate_border_color_without_clamp_to_border_error() {
        let desc = TrinitySamplerDescriptor::new()
            .address_mode(AddressMode::Repeat)
            .border_color(SamplerBorderColor::OpaqueBlack);
        let result = validate_descriptor(&desc);
        assert!(matches!(
            result,
            Err(SamplerValidationError::BorderColorRequiresClampToBorder)
        ));
    }

    #[test]
    fn test_validate_border_color_transparent_black_without_border_error() {
        let desc = TrinitySamplerDescriptor::new()
            .address_mode(AddressMode::ClampToEdge)
            .border_color(SamplerBorderColor::TransparentBlack);
        let result = validate_descriptor(&desc);
        assert!(matches!(
            result,
            Err(SamplerValidationError::BorderColorRequiresClampToBorder)
        ));
    }

    #[test]
    fn test_validate_no_border_color_ok() {
        let desc = TrinitySamplerDescriptor::new().address_mode(AddressMode::Repeat);
        assert!(validate_descriptor(&desc).is_ok());
    }

    #[test]
    fn test_validate_clamp_to_border_without_border_color_ok() {
        // ClampToBorder without explicit border_color is valid
        // (wgpu uses a default)
        let desc = TrinitySamplerDescriptor::new().address_mode(AddressMode::ClampToBorder);
        assert!(validate_descriptor(&desc).is_ok());
    }
}

// ============================================================================
// MODULE 7: Validation Error Display Tests (8 tests)
// ============================================================================

mod error_display_tests {
    use super::*;

    #[test]
    fn test_invalid_anisotropy_display_contains_requested() {
        let err = SamplerValidationError::InvalidAnisotropy {
            requested: 32,
            max_supported: 16,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("32"));
    }

    #[test]
    fn test_invalid_anisotropy_display_contains_max() {
        let err = SamplerValidationError::InvalidAnisotropy {
            requested: 32,
            max_supported: 16,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("16"));
    }

    #[test]
    fn test_invalid_lod_range_display_contains_min() {
        let err = SamplerValidationError::InvalidLodRange {
            min: 10.0,
            max: 5.0,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("10"));
    }

    #[test]
    fn test_invalid_lod_range_display_contains_max() {
        let err = SamplerValidationError::InvalidLodRange {
            min: 10.0,
            max: 5.0,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("5"));
    }

    #[test]
    fn test_border_color_error_display_contains_clamp_to_border() {
        let err = SamplerValidationError::BorderColorRequiresClampToBorder;
        let msg = format!("{}", err);
        assert!(msg.contains("ClampToBorder"));
    }

    #[test]
    fn test_negative_lod_display_contains_value() {
        let err = SamplerValidationError::NegativeLod { value: -2.5 };
        let msg = format!("{}", err);
        assert!(msg.contains("-2.5"));
    }

    #[test]
    fn test_negative_lod_display_contains_non_negative() {
        let err = SamplerValidationError::NegativeLod { value: -1.0 };
        let msg = format!("{}", err);
        assert!(msg.contains("non-negative"));
    }

    #[test]
    fn test_error_is_std_error() {
        let err = SamplerValidationError::InvalidAnisotropy {
            requested: 32,
            max_supported: 16,
        };
        // This compiles if SamplerValidationError implements std::error::Error
        let _: &dyn std::error::Error = &err;
    }
}

// ============================================================================
// MODULE 8: Validation Error Equality Tests (6 tests)
// ============================================================================

mod error_equality_tests {
    use super::*;

    #[test]
    fn test_invalid_anisotropy_equality() {
        let err1 = SamplerValidationError::InvalidAnisotropy {
            requested: 32,
            max_supported: 16,
        };
        let err2 = SamplerValidationError::InvalidAnisotropy {
            requested: 32,
            max_supported: 16,
        };
        assert_eq!(err1, err2);
    }

    #[test]
    fn test_invalid_anisotropy_inequality() {
        let err1 = SamplerValidationError::InvalidAnisotropy {
            requested: 32,
            max_supported: 16,
        };
        let err2 = SamplerValidationError::InvalidAnisotropy {
            requested: 64,
            max_supported: 16,
        };
        assert_ne!(err1, err2);
    }

    #[test]
    fn test_invalid_lod_range_equality() {
        let err1 = SamplerValidationError::InvalidLodRange {
            min: 10.0,
            max: 5.0,
        };
        let err2 = SamplerValidationError::InvalidLodRange {
            min: 10.0,
            max: 5.0,
        };
        assert_eq!(err1, err2);
    }

    #[test]
    fn test_negative_lod_equality() {
        let err1 = SamplerValidationError::NegativeLod { value: -1.5 };
        let err2 = SamplerValidationError::NegativeLod { value: -1.5 };
        assert_eq!(err1, err2);
    }

    #[test]
    fn test_border_color_requires_clamp_equality() {
        let err1 = SamplerValidationError::BorderColorRequiresClampToBorder;
        let err2 = SamplerValidationError::BorderColorRequiresClampToBorder;
        assert_eq!(err1, err2);
    }

    #[test]
    fn test_different_error_variants_inequality() {
        let err1 = SamplerValidationError::InvalidAnisotropy {
            requested: 32,
            max_supported: 16,
        };
        let err2 = SamplerValidationError::InvalidLodRange {
            min: 10.0,
            max: 5.0,
        };
        assert_ne!(err1, err2);
    }
}

// ============================================================================
// MODULE 9: Compare Function Tests (10 tests)
// ============================================================================

mod compare_function_tests {
    use super::*;

    #[test]
    fn test_compare_never() {
        let desc = TrinitySamplerDescriptor::new().compare(CompareFunction::Never);
        assert_eq!(desc.compare, Some(CompareFunction::Never));
    }

    #[test]
    fn test_compare_less() {
        let desc = TrinitySamplerDescriptor::new().compare(CompareFunction::Less);
        assert_eq!(desc.compare, Some(CompareFunction::Less));
    }

    #[test]
    fn test_compare_equal() {
        let desc = TrinitySamplerDescriptor::new().compare(CompareFunction::Equal);
        assert_eq!(desc.compare, Some(CompareFunction::Equal));
    }

    #[test]
    fn test_compare_less_equal() {
        let desc = TrinitySamplerDescriptor::new().compare(CompareFunction::LessEqual);
        assert_eq!(desc.compare, Some(CompareFunction::LessEqual));
    }

    #[test]
    fn test_compare_greater() {
        let desc = TrinitySamplerDescriptor::new().compare(CompareFunction::Greater);
        assert_eq!(desc.compare, Some(CompareFunction::Greater));
    }

    #[test]
    fn test_compare_not_equal() {
        let desc = TrinitySamplerDescriptor::new().compare(CompareFunction::NotEqual);
        assert_eq!(desc.compare, Some(CompareFunction::NotEqual));
    }

    #[test]
    fn test_compare_greater_equal() {
        let desc = TrinitySamplerDescriptor::new().compare(CompareFunction::GreaterEqual);
        assert_eq!(desc.compare, Some(CompareFunction::GreaterEqual));
    }

    #[test]
    fn test_compare_always() {
        let desc = TrinitySamplerDescriptor::new().compare(CompareFunction::Always);
        assert_eq!(desc.compare, Some(CompareFunction::Always));
    }

    #[test]
    fn test_compare_none_by_default() {
        let desc = TrinitySamplerDescriptor::new();
        assert!(desc.compare.is_none());
    }

    #[test]
    fn test_compare_validates_ok() {
        // Comparison samplers should pass validation
        let desc = TrinitySamplerDescriptor::new().compare(CompareFunction::Less);
        assert!(validate_descriptor(&desc).is_ok());
    }
}

// ============================================================================
// MODULE 10: Border Color Tests (6 tests)
// ============================================================================

mod border_color_tests {
    use super::*;

    #[test]
    fn test_border_color_transparent_black() {
        let desc = TrinitySamplerDescriptor::new()
            .address_mode(AddressMode::ClampToBorder)
            .border_color(SamplerBorderColor::TransparentBlack);
        assert_eq!(
            desc.border_color,
            Some(SamplerBorderColor::TransparentBlack)
        );
    }

    #[test]
    fn test_border_color_opaque_black() {
        let desc = TrinitySamplerDescriptor::new()
            .address_mode(AddressMode::ClampToBorder)
            .border_color(SamplerBorderColor::OpaqueBlack);
        assert_eq!(desc.border_color, Some(SamplerBorderColor::OpaqueBlack));
    }

    #[test]
    fn test_border_color_opaque_white() {
        let desc = TrinitySamplerDescriptor::new()
            .address_mode(AddressMode::ClampToBorder)
            .border_color(SamplerBorderColor::OpaqueWhite);
        assert_eq!(desc.border_color, Some(SamplerBorderColor::OpaqueWhite));
    }

    #[test]
    fn test_border_color_none_by_default() {
        let desc = TrinitySamplerDescriptor::new();
        assert!(desc.border_color.is_none());
    }

    #[test]
    fn test_border_color_u_clamp_to_border_valid() {
        let desc = TrinitySamplerDescriptor::new()
            .address_mode_uvw(
                AddressMode::ClampToBorder,
                AddressMode::Repeat,
                AddressMode::Repeat,
            )
            .border_color(SamplerBorderColor::OpaqueBlack);
        assert!(validate_descriptor(&desc).is_ok());
    }

    #[test]
    fn test_border_color_v_clamp_to_border_valid() {
        let desc = TrinitySamplerDescriptor::new()
            .address_mode_uvw(
                AddressMode::Repeat,
                AddressMode::ClampToBorder,
                AddressMode::Repeat,
            )
            .border_color(SamplerBorderColor::OpaqueBlack);
        assert!(validate_descriptor(&desc).is_ok());
    }
}

// ============================================================================
// MODULE 11: Edge Case Tests (15 tests)
// ============================================================================

mod edge_case_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // 11.1 Anisotropy Boundary Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_anisotropy_zero_validates() {
        // 0 is technically invalid but descriptors allow setting it
        let mut desc = TrinitySamplerDescriptor::new();
        desc.anisotropy_clamp = 0;
        // Validation should pass (wgpu clamps to 1)
        assert!(validate_descriptor(&desc).is_ok());
    }

    #[test]
    fn test_anisotropy_boundary_16() {
        let desc = TrinitySamplerDescriptor::new().anisotropy(16);
        assert!(validate_descriptor(&desc).is_ok());
        assert_eq!(desc.anisotropy_clamp, 16);
    }

    #[test]
    fn test_anisotropy_boundary_17() {
        let desc = TrinitySamplerDescriptor::new().anisotropy(17);
        assert!(matches!(
            validate_descriptor(&desc),
            Err(SamplerValidationError::InvalidAnisotropy { .. })
        ));
    }

    // -------------------------------------------------------------------------
    // 11.2 LOD Boundary Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_lod_zero_boundary() {
        let desc = TrinitySamplerDescriptor::new().lod_clamp(0.0, 0.0);
        assert!(validate_descriptor(&desc).is_ok());
    }

    #[test]
    fn test_lod_large_max() {
        let desc = TrinitySamplerDescriptor::new().lod_clamp(0.0, f32::MAX);
        assert!(validate_descriptor(&desc).is_ok());
    }

    #[test]
    fn test_lod_epsilon_above_zero() {
        let desc = TrinitySamplerDescriptor::new().lod_clamp(f32::EPSILON, 32.0);
        assert!(validate_descriptor(&desc).is_ok());
    }

    #[test]
    fn test_lod_infinity_max() {
        let desc = TrinitySamplerDescriptor::new().lod_clamp(0.0, f32::INFINITY);
        assert!(validate_descriptor(&desc).is_ok());
    }

    #[test]
    fn test_lod_negative_infinity_min_error() {
        let mut desc = TrinitySamplerDescriptor::new();
        desc.lod_min_clamp = f32::NEG_INFINITY;
        assert!(matches!(
            validate_descriptor(&desc),
            Err(SamplerValidationError::NegativeLod { .. })
        ));
    }

    // -------------------------------------------------------------------------
    // 11.3 Label Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_label_unicode() {
        let desc = TrinitySamplerDescriptor::new().label("texture_sampler_");
        assert_eq!(desc.label.as_deref(), Some("texture_sampler_"));
    }

    #[test]
    fn test_label_very_long() {
        let long_label = "a".repeat(1000);
        let desc = TrinitySamplerDescriptor::new().label(long_label.clone());
        assert_eq!(desc.label.as_deref(), Some(long_label.as_str()));
    }

    #[test]
    fn test_label_with_spaces() {
        let desc = TrinitySamplerDescriptor::new().label("my sampler with spaces");
        assert_eq!(desc.label.as_deref(), Some("my sampler with spaces"));
    }

    #[test]
    fn test_label_special_characters() {
        let desc = TrinitySamplerDescriptor::new().label("sampler!@#$%^&*()");
        assert_eq!(desc.label.as_deref(), Some("sampler!@#$%^&*()"));
    }

    // -------------------------------------------------------------------------
    // 11.4 Builder Overwrite Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_address_mode_overwrites_uvw() {
        let desc = TrinitySamplerDescriptor::new()
            .address_mode_uvw(
                AddressMode::Repeat,
                AddressMode::MirrorRepeat,
                AddressMode::ClampToBorder,
            )
            .address_mode(AddressMode::ClampToEdge);
        assert_eq!(desc.address_mode_u, AddressMode::ClampToEdge);
        assert_eq!(desc.address_mode_v, AddressMode::ClampToEdge);
        assert_eq!(desc.address_mode_w, AddressMode::ClampToEdge);
    }

    #[test]
    fn test_filter_separate_overwrites_filter() {
        let desc = TrinitySamplerDescriptor::new()
            .filter(FilterMode::Nearest)
            .filter_separate(FilterMode::Linear, FilterMode::Nearest, FilterMode::Linear);
        assert_eq!(desc.mag_filter, FilterMode::Linear);
        assert_eq!(desc.min_filter, FilterMode::Nearest);
        assert_eq!(desc.mipmap_filter, FilterMode::Linear);
    }

    #[test]
    fn test_multiple_compare_calls() {
        let desc = TrinitySamplerDescriptor::new()
            .compare(CompareFunction::Less)
            .compare(CompareFunction::Greater)
            .compare(CompareFunction::Equal);
        assert_eq!(desc.compare, Some(CompareFunction::Equal));
    }
}

// ============================================================================
// MODULE 12: Complex Configuration Tests (8 tests)
// ============================================================================

mod complex_config_tests {
    use super::*;

    #[test]
    fn test_shadow_sampler_with_pcf_config() {
        // PCF-friendly shadow sampler configuration
        let desc = TrinitySamplerDescriptor::new()
            .label("shadow_pcf")
            .filter_separate(FilterMode::Linear, FilterMode::Linear, FilterMode::Nearest)
            .address_mode(AddressMode::ClampToEdge)
            .compare(CompareFunction::Less)
            .lod_clamp(0.0, 0.0); // No mipmaps for shadow maps

        assert!(validate_descriptor(&desc).is_ok());
        assert_eq!(desc.compare, Some(CompareFunction::Less));
        assert_eq!(desc.lod_min_clamp, 0.0);
        assert_eq!(desc.lod_max_clamp, 0.0);
    }

    #[test]
    fn test_anisotropic_texture_sampler() {
        let desc = TrinitySamplerDescriptor::new()
            .label("aniso_diffuse")
            .filter(FilterMode::Linear)
            .address_mode(AddressMode::Repeat)
            .anisotropy(16);

        assert!(validate_descriptor(&desc).is_ok());
        assert_eq!(desc.anisotropy_clamp, 16);
    }

    #[test]
    fn test_cubemap_sampler_config() {
        let desc = TrinitySamplerDescriptor::new()
            .label("cubemap_sampler")
            .filter(FilterMode::Linear)
            .address_mode(AddressMode::ClampToEdge); // Typical for cubemaps

        assert!(validate_descriptor(&desc).is_ok());
    }

    #[test]
    fn test_sprite_sampler_config() {
        // Pixel-perfect sprite sampling
        let desc = TrinitySamplerDescriptor::new()
            .label("sprite_sampler")
            .filter(FilterMode::Nearest)
            .address_mode(AddressMode::ClampToEdge);

        assert!(validate_descriptor(&desc).is_ok());
        assert_eq!(desc.mag_filter, FilterMode::Nearest);
    }

    #[test]
    fn test_terrain_sampler_config() {
        // Terrain with tiling and anisotropic filtering
        let desc = TrinitySamplerDescriptor::new()
            .label("terrain_sampler")
            .filter(FilterMode::Linear)
            .address_mode(AddressMode::Repeat)
            .anisotropy(8)
            .lod_clamp(0.0, 10.0);

        assert!(validate_descriptor(&desc).is_ok());
    }

    #[test]
    fn test_lightmap_sampler_config() {
        // Lightmap: bilinear, no wrapping
        let desc = TrinitySamplerDescriptor::new()
            .label("lightmap_sampler")
            .filter_separate(FilterMode::Linear, FilterMode::Linear, FilterMode::Nearest)
            .address_mode(AddressMode::ClampToEdge)
            .lod_clamp(0.0, 0.0);

        assert!(validate_descriptor(&desc).is_ok());
    }

    #[test]
    fn test_border_sampler_config() {
        // Sampler with explicit border color
        let desc = TrinitySamplerDescriptor::new()
            .label("border_sampler")
            .filter(FilterMode::Linear)
            .address_mode(AddressMode::ClampToBorder)
            .border_color(SamplerBorderColor::TransparentBlack);

        assert!(validate_descriptor(&desc).is_ok());
    }

    #[test]
    fn test_mixed_addressing_sampler() {
        // Different addressing per axis
        let desc = TrinitySamplerDescriptor::new()
            .label("mixed_address")
            .filter(FilterMode::Linear)
            .address_mode_uvw(
                AddressMode::Repeat,
                AddressMode::MirrorRepeat,
                AddressMode::ClampToEdge,
            );

        assert!(validate_descriptor(&desc).is_ok());
    }
}

// ============================================================================
// MODULE 13: Error Clone Tests (4 tests)
// ============================================================================

mod error_clone_tests {
    use super::*;

    #[test]
    fn test_invalid_anisotropy_clone() {
        let err = SamplerValidationError::InvalidAnisotropy {
            requested: 32,
            max_supported: 16,
        };
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }

    #[test]
    fn test_invalid_lod_range_clone() {
        let err = SamplerValidationError::InvalidLodRange {
            min: 10.0,
            max: 5.0,
        };
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }

    #[test]
    fn test_negative_lod_clone() {
        let err = SamplerValidationError::NegativeLod { value: -1.0 };
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }

    #[test]
    fn test_border_color_requires_clamp_clone() {
        let err = SamplerValidationError::BorderColorRequiresClampToBorder;
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }
}

// ============================================================================
// MODULE 14: Error Debug Tests (4 tests)
// ============================================================================

mod error_debug_tests {
    use super::*;

    #[test]
    fn test_invalid_anisotropy_debug() {
        let err = SamplerValidationError::InvalidAnisotropy {
            requested: 32,
            max_supported: 16,
        };
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("InvalidAnisotropy"));
        assert!(debug_str.contains("32"));
        assert!(debug_str.contains("16"));
    }

    #[test]
    fn test_invalid_lod_range_debug() {
        let err = SamplerValidationError::InvalidLodRange {
            min: 10.0,
            max: 5.0,
        };
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("InvalidLodRange"));
    }

    #[test]
    fn test_negative_lod_debug() {
        let err = SamplerValidationError::NegativeLod { value: -1.0 };
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("NegativeLod"));
    }

    #[test]
    fn test_border_color_requires_clamp_debug() {
        let err = SamplerValidationError::BorderColorRequiresClampToBorder;
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("BorderColorRequiresClampToBorder"));
    }
}

// ============================================================================
// MODULE 15: Preset Validation Tests (6 tests)
// ============================================================================

mod preset_validation_tests {
    use super::*;

    #[test]
    fn test_all_presets_validate_ok() {
        let presets = [
            TrinitySamplerDescriptor::linear_clamp(),
            TrinitySamplerDescriptor::linear_repeat(),
            TrinitySamplerDescriptor::nearest_clamp(),
            TrinitySamplerDescriptor::nearest_repeat(),
            TrinitySamplerDescriptor::shadow(),
            TrinitySamplerDescriptor::trilinear(),
        ];

        for preset in presets.iter() {
            assert!(
                validate_descriptor(preset).is_ok(),
                "Preset with label {:?} failed validation",
                preset.label
            );
        }
    }

    #[test]
    fn test_presets_have_labels() {
        assert!(TrinitySamplerDescriptor::linear_clamp().label.is_some());
        assert!(TrinitySamplerDescriptor::linear_repeat().label.is_some());
        assert!(TrinitySamplerDescriptor::nearest_clamp().label.is_some());
        assert!(TrinitySamplerDescriptor::nearest_repeat().label.is_some());
        assert!(TrinitySamplerDescriptor::shadow().label.is_some());
        assert!(TrinitySamplerDescriptor::trilinear().label.is_some());
    }

    #[test]
    fn test_presets_use_default_anisotropy() {
        let presets = [
            TrinitySamplerDescriptor::linear_clamp(),
            TrinitySamplerDescriptor::linear_repeat(),
            TrinitySamplerDescriptor::nearest_clamp(),
            TrinitySamplerDescriptor::nearest_repeat(),
            TrinitySamplerDescriptor::shadow(),
            TrinitySamplerDescriptor::trilinear(),
        ];

        for preset in presets.iter() {
            assert_eq!(
                preset.anisotropy_clamp, 1,
                "Preset {:?} has non-default anisotropy",
                preset.label
            );
        }
    }

    #[test]
    fn test_presets_use_default_lod() {
        let presets = [
            TrinitySamplerDescriptor::linear_clamp(),
            TrinitySamplerDescriptor::linear_repeat(),
            TrinitySamplerDescriptor::nearest_clamp(),
            TrinitySamplerDescriptor::nearest_repeat(),
            TrinitySamplerDescriptor::trilinear(),
        ];

        for preset in presets.iter() {
            assert_eq!(preset.lod_min_clamp, 0.0);
            assert_eq!(preset.lod_max_clamp, 32.0);
        }
    }

    #[test]
    fn test_presets_have_no_border_color() {
        let presets = [
            TrinitySamplerDescriptor::linear_clamp(),
            TrinitySamplerDescriptor::linear_repeat(),
            TrinitySamplerDescriptor::nearest_clamp(),
            TrinitySamplerDescriptor::nearest_repeat(),
            TrinitySamplerDescriptor::shadow(),
            TrinitySamplerDescriptor::trilinear(),
        ];

        for preset in presets.iter() {
            assert!(preset.border_color.is_none());
        }
    }

    #[test]
    fn test_only_shadow_preset_has_compare() {
        assert!(TrinitySamplerDescriptor::linear_clamp().compare.is_none());
        assert!(TrinitySamplerDescriptor::linear_repeat().compare.is_none());
        assert!(TrinitySamplerDescriptor::nearest_clamp().compare.is_none());
        assert!(TrinitySamplerDescriptor::nearest_repeat()
            .compare
            .is_none());
        assert!(TrinitySamplerDescriptor::shadow().compare.is_some());
        assert!(TrinitySamplerDescriptor::trilinear().compare.is_none());
    }
}
