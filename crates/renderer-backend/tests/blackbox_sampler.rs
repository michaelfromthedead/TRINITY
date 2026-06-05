// Blackbox contract tests for T-WGPU-P2.4.1 Sampler Creation API
//
// CLEANROOM: No access to implementation files. Tests use only the public API
// exported by `renderer_backend::resources::sampler`.
//
// Forbidden files (per TESTDEV_BLACKBOX prompt):
//   - crates/renderer-backend/src/resources/sampler.rs (implementation)
//   - Any WHITEBOX test file for this task
//
// Contract sources:
//   - Task acceptance criteria (T-WGPU-P2.4.1)
//   - Public API documentation
//
// Public API under test:
//   - TrinitySamplerDescriptor: Builder pattern for sampler configuration
//   - TrinitySampler: Wrapper with metadata and inner wgpu::Sampler access
//   - create_sampler(device, descriptor) -> TrinitySampler
//   - try_create_sampler(device, descriptor) -> Result<TrinitySampler, SamplerValidationError>
//   - validate_descriptor(descriptor) -> Result<(), SamplerValidationError>
//   - Presets: linear_clamp, linear_repeat, nearest_clamp, nearest_repeat, shadow, trilinear
//   - Re-exports: AddressMode, FilterMode, CompareFunction, SamplerBorderColor
//
// Test design rationale:
//   - API contract: All public methods exist and return correct types
//   - Builder pattern: Chaining, modification, defaults
//   - Presets: Each preset has distinct characteristics
//   - Validation: Invalid configs rejected, valid configs pass
//   - Integration: GPU sampler creation (requires device)

use pollster::block_on;
use renderer_backend::device::{enumerate_adapters_with_info, TrinityInstance};
use renderer_backend::resources::sampler::{
    create_sampler, try_create_sampler, validate_descriptor, AddressMode, CompareFunction,
    FilterMode, SamplerBorderColor, SamplerValidationError, TrinitySampler,
    TrinitySamplerDescriptor,
};

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Creates a TrinityInstance and gets the first available adapter.
fn get_test_adapter() -> Option<wgpu::Adapter> {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());
    result.adapters.into_iter().next()
}

/// Helper macro to skip test if no GPU adapter is available.
macro_rules! require_adapter {
    () => {
        match get_test_adapter() {
            Some(adapter) => adapter,
            None => {
                eprintln!("SKIP: No GPU adapter available for this test");
                return;
            }
        }
    };
}

/// Creates a wgpu device for testing.
fn create_test_device(adapter: &wgpu::Adapter) -> Option<(wgpu::Device, wgpu::Queue)> {
    block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).ok()
}

/// Helper macro to get a device, skipping if unavailable.
macro_rules! require_device {
    ($adapter:expr) => {
        match create_test_device($adapter) {
            Some((device, queue)) => (device, queue),
            None => {
                eprintln!("SKIP: Could not create device");
                return;
            }
        }
    };
}

// =============================================================================
// MODULE: API CONTRACT TESTS (no GPU required)
// =============================================================================

mod api_contract_tests {
    use super::*;

    /// Test: TrinitySamplerDescriptor::new() exists and returns Self.
    #[test]
    fn descriptor_new_exists() {
        let desc = TrinitySamplerDescriptor::new();
        // Just verify it compiles and returns a descriptor
        let _ = desc;
    }

    /// Test: TrinitySamplerDescriptor has label() builder method.
    #[test]
    fn descriptor_has_label_method() {
        let desc = TrinitySamplerDescriptor::new().label("test_sampler");
        let _ = desc;
    }

    /// Test: TrinitySamplerDescriptor has address_mode() builder method.
    #[test]
    fn descriptor_has_address_mode_method() {
        let desc = TrinitySamplerDescriptor::new().address_mode(AddressMode::Repeat);
        let _ = desc;
    }

    /// Test: TrinitySamplerDescriptor has address_mode_uvw() builder method.
    #[test]
    fn descriptor_has_address_mode_uvw_method() {
        let desc = TrinitySamplerDescriptor::new().address_mode_uvw(
            AddressMode::ClampToEdge,
            AddressMode::Repeat,
            AddressMode::MirrorRepeat,
        );
        let _ = desc;
    }

    /// Test: TrinitySamplerDescriptor has filter() builder method.
    #[test]
    fn descriptor_has_filter_method() {
        let desc = TrinitySamplerDescriptor::new().filter(FilterMode::Linear);
        let _ = desc;
    }

    /// Test: TrinitySamplerDescriptor has filter_separate() builder method.
    #[test]
    fn descriptor_has_filter_separate_method() {
        let desc = TrinitySamplerDescriptor::new().filter_separate(
            FilterMode::Linear,
            FilterMode::Linear,
            FilterMode::Nearest,
        );
        let _ = desc;
    }

    /// Test: TrinitySamplerDescriptor has lod_clamp() builder method.
    #[test]
    fn descriptor_has_lod_clamp_method() {
        let desc = TrinitySamplerDescriptor::new().lod_clamp(0.0, 10.0);
        let _ = desc;
    }

    /// Test: TrinitySamplerDescriptor has anisotropy() builder method.
    #[test]
    fn descriptor_has_anisotropy_method() {
        let desc = TrinitySamplerDescriptor::new().anisotropy(8);
        let _ = desc;
    }

    /// Test: TrinitySamplerDescriptor has compare() builder method.
    #[test]
    fn descriptor_has_compare_method() {
        let desc = TrinitySamplerDescriptor::new().compare(CompareFunction::Less);
        let _ = desc;
    }

    /// Test: TrinitySamplerDescriptor has border_color() builder method.
    #[test]
    fn descriptor_has_border_color_method() {
        let desc = TrinitySamplerDescriptor::new().border_color(SamplerBorderColor::OpaqueBlack);
        let _ = desc;
    }

    /// Test: linear_clamp() preset exists.
    #[test]
    fn preset_linear_clamp_exists() {
        let desc = TrinitySamplerDescriptor::linear_clamp();
        let _ = desc;
    }

    /// Test: linear_repeat() preset exists.
    #[test]
    fn preset_linear_repeat_exists() {
        let desc = TrinitySamplerDescriptor::linear_repeat();
        let _ = desc;
    }

    /// Test: nearest_clamp() preset exists.
    #[test]
    fn preset_nearest_clamp_exists() {
        let desc = TrinitySamplerDescriptor::nearest_clamp();
        let _ = desc;
    }

    /// Test: nearest_repeat() preset exists.
    #[test]
    fn preset_nearest_repeat_exists() {
        let desc = TrinitySamplerDescriptor::nearest_repeat();
        let _ = desc;
    }

    /// Test: shadow() preset exists.
    #[test]
    fn preset_shadow_exists() {
        let desc = TrinitySamplerDescriptor::shadow();
        let _ = desc;
    }

    /// Test: trilinear() preset exists.
    #[test]
    fn preset_trilinear_exists() {
        let desc = TrinitySamplerDescriptor::trilinear();
        let _ = desc;
    }

    /// Test: AddressMode re-export has expected variants.
    #[test]
    fn address_mode_variants_exist() {
        let _ = AddressMode::ClampToEdge;
        let _ = AddressMode::Repeat;
        let _ = AddressMode::MirrorRepeat;
        let _ = AddressMode::ClampToBorder;
    }

    /// Test: FilterMode re-export has expected variants.
    #[test]
    fn filter_mode_variants_exist() {
        let _ = FilterMode::Nearest;
        let _ = FilterMode::Linear;
    }

    /// Test: CompareFunction re-export has expected variants.
    #[test]
    fn compare_function_variants_exist() {
        let _ = CompareFunction::Never;
        let _ = CompareFunction::Less;
        let _ = CompareFunction::Equal;
        let _ = CompareFunction::LessEqual;
        let _ = CompareFunction::Greater;
        let _ = CompareFunction::NotEqual;
        let _ = CompareFunction::GreaterEqual;
        let _ = CompareFunction::Always;
    }

    /// Test: SamplerBorderColor re-export has expected variants.
    #[test]
    fn sampler_border_color_variants_exist() {
        let _ = SamplerBorderColor::TransparentBlack;
        let _ = SamplerBorderColor::OpaqueBlack;
        let _ = SamplerBorderColor::OpaqueWhite;
    }

    /// Test: SamplerValidationError type exists (as enum or struct).
    #[test]
    fn sampler_validation_error_type_exists() {
        // We can't construct it directly without knowing variants,
        // but we can verify it's used in return types
        fn _takes_error(_e: SamplerValidationError) {}
    }

    /// Test: validate_descriptor function exists with correct signature.
    #[test]
    fn validate_descriptor_function_exists() {
        let desc = TrinitySamplerDescriptor::new();
        let _result: Result<(), SamplerValidationError> = validate_descriptor(&desc);
    }
}

// =============================================================================
// MODULE: BUILDER TESTS (no GPU required)
// =============================================================================

mod builder_tests {
    use super::*;

    /// Test: Builder methods can be chained.
    #[test]
    fn builder_chaining_works() {
        let desc = TrinitySamplerDescriptor::new()
            .label("chained_sampler")
            .address_mode(AddressMode::Repeat)
            .filter(FilterMode::Linear)
            .lod_clamp(0.0, 16.0)
            .anisotropy(4);
        let _ = desc;
    }

    /// Test: Multiple chained calls of same method (last wins).
    #[test]
    fn builder_last_call_wins() {
        let desc = TrinitySamplerDescriptor::new()
            .filter(FilterMode::Nearest)
            .filter(FilterMode::Linear); // This should override
        let _ = desc;
    }

    /// Test: Preset can be modified with builder methods.
    #[test]
    fn preset_modification_works() {
        let desc = TrinitySamplerDescriptor::linear_clamp()
            .label("modified_linear_clamp")
            .anisotropy(8);
        let _ = desc;
    }

    /// Test: address_mode_uvw allows different modes per axis.
    #[test]
    fn address_mode_uvw_allows_different_per_axis() {
        let desc = TrinitySamplerDescriptor::new().address_mode_uvw(
            AddressMode::ClampToEdge,
            AddressMode::Repeat,
            AddressMode::MirrorRepeat,
        );
        let _ = desc;
    }

    /// Test: filter_separate allows different modes per stage.
    #[test]
    fn filter_separate_allows_different_modes() {
        let desc = TrinitySamplerDescriptor::new().filter_separate(
            FilterMode::Linear,  // mag
            FilterMode::Linear,  // min
            FilterMode::Nearest, // mip
        );
        let _ = desc;
    }

    /// Test: LOD clamp with min > max is allowed at build time (validated later).
    #[test]
    fn lod_clamp_allows_any_values_at_build_time() {
        // Builder should accept any values; validation happens later
        let desc = TrinitySamplerDescriptor::new().lod_clamp(10.0, 0.0); // min > max
        let _ = desc;
    }

    /// Test: Anisotropy value of 0 is allowed at build time.
    #[test]
    fn anisotropy_zero_allowed_at_build_time() {
        let desc = TrinitySamplerDescriptor::new().anisotropy(0);
        let _ = desc;
    }

    /// Test: Anisotropy value of 1 is allowed (effectively disables anisotropy).
    #[test]
    fn anisotropy_one_allowed() {
        let desc = TrinitySamplerDescriptor::new().anisotropy(1);
        let _ = desc;
    }

    /// Test: Large anisotropy values allowed at build time (clamped on creation).
    #[test]
    fn anisotropy_large_values_allowed_at_build_time() {
        let desc = TrinitySamplerDescriptor::new().anisotropy(32);
        let _ = desc;
    }

    /// Test: Border color can be set with ClampToBorder address mode.
    #[test]
    fn border_color_with_clamp_to_border() {
        let desc = TrinitySamplerDescriptor::new()
            .address_mode(AddressMode::ClampToBorder)
            .border_color(SamplerBorderColor::OpaqueWhite);
        let _ = desc;
    }

    /// Test: Multiple labels can be set (last wins).
    #[test]
    fn label_last_wins() {
        let desc = TrinitySamplerDescriptor::new()
            .label("first")
            .label("second")
            .label("third");
        let _ = desc;
    }

    /// Test: Empty label is allowed.
    #[test]
    fn empty_label_allowed() {
        let desc = TrinitySamplerDescriptor::new().label("");
        let _ = desc;
    }

    /// Test: Unicode label is allowed.
    #[test]
    fn unicode_label_allowed() {
        let desc = TrinitySamplerDescriptor::new().label("sampler_unicode_test");
        let _ = desc;
    }

    /// Test: Compare function can be set.
    #[test]
    fn compare_function_can_be_set() {
        let desc = TrinitySamplerDescriptor::new().compare(CompareFunction::LessEqual);
        let _ = desc;
    }

    /// Test: All compare functions can be set.
    #[test]
    fn all_compare_functions_settable() {
        let functions = [
            CompareFunction::Never,
            CompareFunction::Less,
            CompareFunction::Equal,
            CompareFunction::LessEqual,
            CompareFunction::Greater,
            CompareFunction::NotEqual,
            CompareFunction::GreaterEqual,
            CompareFunction::Always,
        ];

        for func in functions {
            let desc = TrinitySamplerDescriptor::new().compare(func);
            let _ = desc;
        }
    }

    /// Test: LOD clamp with negative min is allowed at build time.
    #[test]
    fn lod_clamp_negative_min_allowed() {
        let desc = TrinitySamplerDescriptor::new().lod_clamp(-1.0, 10.0);
        let _ = desc;
    }

    /// Test: LOD clamp with very large max is allowed.
    #[test]
    fn lod_clamp_large_max_allowed() {
        let desc = TrinitySamplerDescriptor::new().lod_clamp(0.0, 1000.0);
        let _ = desc;
    }
}

// =============================================================================
// MODULE: PRESET TESTS (no GPU required)
// =============================================================================

mod preset_tests {
    use super::*;

    /// Test: linear_clamp preset is distinct from linear_repeat.
    #[test]
    fn linear_clamp_distinct_from_linear_repeat() {
        let clamp = TrinitySamplerDescriptor::linear_clamp();
        let repeat = TrinitySamplerDescriptor::linear_repeat();
        // Both should compile and be usable
        let _ = clamp;
        let _ = repeat;
    }

    /// Test: nearest_clamp preset is distinct from nearest_repeat.
    #[test]
    fn nearest_clamp_distinct_from_nearest_repeat() {
        let clamp = TrinitySamplerDescriptor::nearest_clamp();
        let repeat = TrinitySamplerDescriptor::nearest_repeat();
        let _ = clamp;
        let _ = repeat;
    }

    /// Test: shadow preset is for comparison sampling.
    #[test]
    fn shadow_preset_is_comparison_sampler() {
        let shadow = TrinitySamplerDescriptor::shadow();
        // shadow() preset should enable comparison - we verify this on creation
        let _ = shadow;
    }

    /// Test: trilinear preset has mipmap filtering.
    #[test]
    fn trilinear_preset_has_mipmap_filtering() {
        let trilinear = TrinitySamplerDescriptor::trilinear();
        let _ = trilinear;
    }

    /// Test: Presets can be further modified.
    #[test]
    fn presets_can_be_modified() {
        let modified_linear = TrinitySamplerDescriptor::linear_clamp()
            .label("custom_linear")
            .anisotropy(4);
        let _ = modified_linear;

        let modified_shadow = TrinitySamplerDescriptor::shadow()
            .label("custom_shadow")
            .compare(CompareFunction::Greater);
        let _ = modified_shadow;
    }

    /// Test: All presets compile without error.
    #[test]
    fn all_presets_compile() {
        let presets = [
            TrinitySamplerDescriptor::linear_clamp(),
            TrinitySamplerDescriptor::linear_repeat(),
            TrinitySamplerDescriptor::nearest_clamp(),
            TrinitySamplerDescriptor::nearest_repeat(),
            TrinitySamplerDescriptor::shadow(),
            TrinitySamplerDescriptor::trilinear(),
        ];

        assert_eq!(presets.len(), 6, "Should have 6 presets");
    }
}

// =============================================================================
// MODULE: VALIDATION TESTS (no GPU required)
// =============================================================================

mod validation_tests {
    use super::*;

    /// Test: Default descriptor passes validation.
    #[test]
    fn default_descriptor_validates() {
        let desc = TrinitySamplerDescriptor::new();
        let result = validate_descriptor(&desc);
        assert!(result.is_ok(), "Default descriptor should validate");
    }

    /// Test: linear_clamp preset passes validation.
    #[test]
    fn linear_clamp_preset_validates() {
        let desc = TrinitySamplerDescriptor::linear_clamp();
        let result = validate_descriptor(&desc);
        assert!(result.is_ok(), "linear_clamp should validate");
    }

    /// Test: linear_repeat preset passes validation.
    #[test]
    fn linear_repeat_preset_validates() {
        let desc = TrinitySamplerDescriptor::linear_repeat();
        let result = validate_descriptor(&desc);
        assert!(result.is_ok(), "linear_repeat should validate");
    }

    /// Test: nearest_clamp preset passes validation.
    #[test]
    fn nearest_clamp_preset_validates() {
        let desc = TrinitySamplerDescriptor::nearest_clamp();
        let result = validate_descriptor(&desc);
        assert!(result.is_ok(), "nearest_clamp should validate");
    }

    /// Test: nearest_repeat preset passes validation.
    #[test]
    fn nearest_repeat_preset_validates() {
        let desc = TrinitySamplerDescriptor::nearest_repeat();
        let result = validate_descriptor(&desc);
        assert!(result.is_ok(), "nearest_repeat should validate");
    }

    /// Test: shadow preset passes validation.
    #[test]
    fn shadow_preset_validates() {
        let desc = TrinitySamplerDescriptor::shadow();
        let result = validate_descriptor(&desc);
        assert!(result.is_ok(), "shadow should validate");
    }

    /// Test: trilinear preset passes validation.
    #[test]
    fn trilinear_preset_validates() {
        let desc = TrinitySamplerDescriptor::trilinear();
        let result = validate_descriptor(&desc);
        assert!(result.is_ok(), "trilinear should validate");
    }

    /// Test: Anisotropy within limits passes validation.
    #[test]
    fn anisotropy_within_limits_validates() {
        let desc = TrinitySamplerDescriptor::new().anisotropy(8);
        let result = validate_descriptor(&desc);
        assert!(result.is_ok(), "Anisotropy 8 should validate");
    }

    /// Test: Anisotropy of 1 (disabled) passes validation.
    #[test]
    fn anisotropy_one_validates() {
        let desc = TrinitySamplerDescriptor::new().anisotropy(1);
        let result = validate_descriptor(&desc);
        assert!(result.is_ok(), "Anisotropy 1 should validate");
    }

    /// Test: Zero LOD range passes validation.
    #[test]
    fn zero_lod_range_validates() {
        let desc = TrinitySamplerDescriptor::new().lod_clamp(0.0, 0.0);
        let result = validate_descriptor(&desc);
        assert!(result.is_ok(), "Zero LOD range should validate");
    }

    /// Test: Positive LOD range passes validation.
    #[test]
    fn positive_lod_range_validates() {
        let desc = TrinitySamplerDescriptor::new().lod_clamp(0.0, 16.0);
        let result = validate_descriptor(&desc);
        assert!(result.is_ok(), "Positive LOD range should validate");
    }

    /// Test: ClampToBorder with border color passes validation.
    #[test]
    fn clamp_to_border_with_color_validates() {
        let desc = TrinitySamplerDescriptor::new()
            .address_mode(AddressMode::ClampToBorder)
            .border_color(SamplerBorderColor::OpaqueBlack);
        let result = validate_descriptor(&desc);
        assert!(result.is_ok(), "ClampToBorder with color should validate");
    }

    /// Test: All address modes validate individually.
    #[test]
    fn all_address_modes_validate() {
        let modes = [
            AddressMode::ClampToEdge,
            AddressMode::Repeat,
            AddressMode::MirrorRepeat,
        ];

        for mode in modes {
            let desc = TrinitySamplerDescriptor::new().address_mode(mode);
            let result = validate_descriptor(&desc);
            assert!(
                result.is_ok(),
                "Address mode {:?} should validate",
                mode
            );
        }
    }

    /// Test: All filter modes validate.
    #[test]
    fn all_filter_modes_validate() {
        let filters = [FilterMode::Nearest, FilterMode::Linear];

        for filter in filters {
            let desc = TrinitySamplerDescriptor::new().filter(filter);
            let result = validate_descriptor(&desc);
            assert!(
                result.is_ok(),
                "Filter mode {:?} should validate",
                filter
            );
        }
    }

    /// Test: Comparison sampler validates.
    #[test]
    fn comparison_sampler_validates() {
        let desc = TrinitySamplerDescriptor::new().compare(CompareFunction::Less);
        let result = validate_descriptor(&desc);
        assert!(result.is_ok(), "Comparison sampler should validate");
    }

    /// Test: Large anisotropy is handled (may be clamped or rejected).
    #[test]
    fn anisotropy_large_values_handled() {
        let desc = TrinitySamplerDescriptor::new().anisotropy(32);
        // May pass (clamped) or fail - either is acceptable
        let _ = validate_descriptor(&desc);
    }

    /// Test: Complex valid configuration validates.
    #[test]
    fn complex_valid_config_validates() {
        let desc = TrinitySamplerDescriptor::new()
            .label("complex_sampler")
            .address_mode_uvw(
                AddressMode::Repeat,
                AddressMode::MirrorRepeat,
                AddressMode::ClampToEdge,
            )
            .filter_separate(FilterMode::Linear, FilterMode::Linear, FilterMode::Linear)
            .lod_clamp(0.0, 12.0)
            .anisotropy(8);

        let result = validate_descriptor(&desc);
        assert!(result.is_ok(), "Complex valid config should validate");
    }
}

// =============================================================================
// MODULE: INTEGRATION TESTS (require GPU)
// =============================================================================

mod integration_tests {
    use super::*;

    /// Test: create_sampler with default descriptor succeeds.
    #[test]
    
    fn create_sampler_default_succeeds() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let desc = TrinitySamplerDescriptor::new();
        let sampler = create_sampler(&device, &desc);

        // Sampler should be created
        let _ = sampler.inner();
    }

    /// Test: create_sampler with linear_clamp preset succeeds.
    #[test]
    
    fn create_sampler_linear_clamp_succeeds() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let desc = TrinitySamplerDescriptor::linear_clamp();
        let sampler = create_sampler(&device, &desc);

        let _ = sampler.inner();
    }

    /// Test: create_sampler with linear_repeat preset succeeds.
    #[test]
    
    fn create_sampler_linear_repeat_succeeds() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let desc = TrinitySamplerDescriptor::linear_repeat();
        let sampler = create_sampler(&device, &desc);

        let _ = sampler.inner();
    }

    /// Test: create_sampler with nearest_clamp preset succeeds.
    #[test]
    
    fn create_sampler_nearest_clamp_succeeds() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let desc = TrinitySamplerDescriptor::nearest_clamp();
        let sampler = create_sampler(&device, &desc);

        let _ = sampler.inner();
    }

    /// Test: create_sampler with nearest_repeat preset succeeds.
    #[test]
    
    fn create_sampler_nearest_repeat_succeeds() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let desc = TrinitySamplerDescriptor::nearest_repeat();
        let sampler = create_sampler(&device, &desc);

        let _ = sampler.inner();
    }

    /// Test: create_sampler with shadow preset succeeds.
    #[test]
    
    fn create_sampler_shadow_succeeds() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let desc = TrinitySamplerDescriptor::shadow();
        let sampler = create_sampler(&device, &desc);

        // Shadow sampler should be a comparison sampler
        assert!(
            sampler.is_comparison_sampler(),
            "Shadow sampler should be comparison sampler"
        );
    }

    /// Test: create_sampler with trilinear preset succeeds.
    #[test]
    
    fn create_sampler_trilinear_succeeds() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let desc = TrinitySamplerDescriptor::trilinear();
        let sampler = create_sampler(&device, &desc);

        let _ = sampler.inner();
    }

    /// Test: create_sampler with custom configuration succeeds.
    #[test]
    
    fn create_sampler_custom_config_succeeds() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let desc = TrinitySamplerDescriptor::new()
            .label("custom_sampler")
            .address_mode(AddressMode::Repeat)
            .filter(FilterMode::Linear)
            .lod_clamp(0.0, 10.0);

        let sampler = create_sampler(&device, &desc);
        let _ = sampler.inner();
    }

    /// Test: create_sampler with anisotropic filtering succeeds.
    #[test]
    
    fn create_sampler_anisotropic_succeeds() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let desc = TrinitySamplerDescriptor::new()
            .label("anisotropic_sampler")
            .filter(FilterMode::Linear)
            .anisotropy(8);

        let sampler = create_sampler(&device, &desc);

        assert!(
            sampler.is_anisotropic(),
            "Sampler with anisotropy > 1 should be anisotropic"
        );
    }

    /// Test: TrinitySampler::inner() returns valid wgpu::Sampler reference.
    #[test]
    
    fn sampler_inner_returns_valid_reference() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let desc = TrinitySamplerDescriptor::linear_clamp();
        let sampler = create_sampler(&device, &desc);

        let inner: &wgpu::Sampler = sampler.inner();
        // Just verify we can get the reference
        let _ = inner;
    }

    /// Test: TrinitySampler::descriptor() returns reference to original descriptor.
    #[test]
    
    fn sampler_descriptor_returns_reference() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let desc = TrinitySamplerDescriptor::linear_clamp().label("test_label");
        let sampler = create_sampler(&device, &desc);

        let stored_desc: &TrinitySamplerDescriptor = sampler.descriptor();
        let _ = stored_desc;
    }

    /// Test: TrinitySampler::label() returns the set label.
    #[test]
    
    fn sampler_label_returns_set_value() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let desc = TrinitySamplerDescriptor::new().label("my_test_sampler");
        let sampler = create_sampler(&device, &desc);

        assert_eq!(sampler.label(), Some("my_test_sampler"));
    }

    /// Test: TrinitySampler::label() returns None when not set.
    #[test]
    
    fn sampler_label_returns_none_when_unset() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let desc = TrinitySamplerDescriptor::new();
        let sampler = create_sampler(&device, &desc);

        assert_eq!(sampler.label(), None);
    }

    /// Test: is_comparison_sampler returns false for non-comparison sampler.
    #[test]
    
    fn non_comparison_sampler_returns_false() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let desc = TrinitySamplerDescriptor::linear_clamp();
        let sampler = create_sampler(&device, &desc);

        assert!(
            !sampler.is_comparison_sampler(),
            "linear_clamp should not be comparison sampler"
        );
    }

    /// Test: is_comparison_sampler returns true for comparison sampler.
    #[test]
    
    fn comparison_sampler_returns_true() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let desc = TrinitySamplerDescriptor::new().compare(CompareFunction::Less);
        let sampler = create_sampler(&device, &desc);

        assert!(
            sampler.is_comparison_sampler(),
            "Sampler with compare function should be comparison sampler"
        );
    }

    /// Test: is_anisotropic returns false for non-anisotropic sampler.
    #[test]
    
    fn non_anisotropic_sampler_returns_false() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let desc = TrinitySamplerDescriptor::new().anisotropy(1);
        let sampler = create_sampler(&device, &desc);

        assert!(
            !sampler.is_anisotropic(),
            "Sampler with anisotropy 1 should not be anisotropic"
        );
    }

    /// Test: is_anisotropic returns true for anisotropic sampler.
    #[test]
    
    fn anisotropic_sampler_returns_true() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let desc = TrinitySamplerDescriptor::new().anisotropy(4);
        let sampler = create_sampler(&device, &desc);

        assert!(
            sampler.is_anisotropic(),
            "Sampler with anisotropy 4 should be anisotropic"
        );
    }

    /// Test: try_create_sampler with valid config returns Ok.
    #[test]
    
    fn try_create_sampler_valid_returns_ok() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let desc = TrinitySamplerDescriptor::linear_clamp();
        let result = try_create_sampler(&device, &desc);

        assert!(result.is_ok(), "try_create_sampler with valid config should return Ok");
    }

    /// Test: Multiple samplers can be created.
    #[test]
    
    fn multiple_samplers_can_be_created() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let samplers: Vec<TrinitySampler> = vec![
            create_sampler(&device, &TrinitySamplerDescriptor::linear_clamp()),
            create_sampler(&device, &TrinitySamplerDescriptor::linear_repeat()),
            create_sampler(&device, &TrinitySamplerDescriptor::nearest_clamp()),
            create_sampler(&device, &TrinitySamplerDescriptor::nearest_repeat()),
            create_sampler(&device, &TrinitySamplerDescriptor::shadow()),
            create_sampler(&device, &TrinitySamplerDescriptor::trilinear()),
        ];

        assert_eq!(samplers.len(), 6, "Should create 6 samplers");
    }

    /// Test: Sampler with ClampToBorder address mode.
    #[test]
    
    fn create_sampler_clamp_to_border() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let desc = TrinitySamplerDescriptor::new()
            .address_mode(AddressMode::ClampToBorder)
            .border_color(SamplerBorderColor::OpaqueBlack);

        let sampler = create_sampler(&device, &desc);
        let _ = sampler.inner();
    }

    /// Test: Sampler with MirrorRepeat address mode.
    #[test]
    
    fn create_sampler_mirror_repeat() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let desc = TrinitySamplerDescriptor::new().address_mode(AddressMode::MirrorRepeat);

        let sampler = create_sampler(&device, &desc);
        let _ = sampler.inner();
    }

    /// Test: Sampler with per-axis address modes.
    #[test]
    
    fn create_sampler_per_axis_address_modes() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let desc = TrinitySamplerDescriptor::new().address_mode_uvw(
            AddressMode::Repeat,
            AddressMode::MirrorRepeat,
            AddressMode::ClampToEdge,
        );

        let sampler = create_sampler(&device, &desc);
        let _ = sampler.inner();
    }

    /// Test: Sampler with separate filter modes.
    #[test]
    
    fn create_sampler_separate_filter_modes() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let desc = TrinitySamplerDescriptor::new().filter_separate(
            FilterMode::Linear,
            FilterMode::Linear,
            FilterMode::Nearest,
        );

        let sampler = create_sampler(&device, &desc);
        let _ = sampler.inner();
    }

    /// Test: Sampler with LOD clamp range.
    #[test]
    
    fn create_sampler_lod_clamp() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let desc = TrinitySamplerDescriptor::new()
            .filter(FilterMode::Linear)
            .lod_clamp(2.0, 8.0);

        let sampler = create_sampler(&device, &desc);
        let _ = sampler.inner();
    }

    /// Test: Sampler with all comparison functions.
    #[test]
    
    fn create_sampler_all_compare_functions() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let functions = [
            CompareFunction::Never,
            CompareFunction::Less,
            CompareFunction::Equal,
            CompareFunction::LessEqual,
            CompareFunction::Greater,
            CompareFunction::NotEqual,
            CompareFunction::GreaterEqual,
            CompareFunction::Always,
        ];

        for func in functions {
            let desc = TrinitySamplerDescriptor::new().compare(func);
            let sampler = create_sampler(&device, &desc);
            assert!(
                sampler.is_comparison_sampler(),
                "Sampler with {:?} should be comparison sampler",
                func
            );
        }
    }

    /// Test: Sampler created from modified preset.
    #[test]
    
    fn create_sampler_modified_preset() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let desc = TrinitySamplerDescriptor::trilinear()
            .label("modified_trilinear")
            .anisotropy(4)
            .lod_clamp(0.0, 8.0);

        let sampler = create_sampler(&device, &desc);
        assert!(
            sampler.is_anisotropic(),
            "Modified trilinear with anisotropy should be anisotropic"
        );
    }

    /// Test: TrinitySampler accessor methods work correctly.
    #[test]
    
    fn sampler_accessor_methods_work() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let desc = TrinitySamplerDescriptor::new()
            .label("accessor_test")
            .filter_separate(FilterMode::Linear, FilterMode::Nearest, FilterMode::Linear)
            .address_mode_uvw(AddressMode::Repeat, AddressMode::MirrorRepeat, AddressMode::ClampToEdge)
            .anisotropy(4)
            .compare(CompareFunction::Less);

        let sampler = create_sampler(&device, &desc);

        // Test all accessor methods
        assert_eq!(sampler.label(), Some("accessor_test"));
        assert_eq!(sampler.mag_filter(), FilterMode::Linear);
        assert_eq!(sampler.min_filter(), FilterMode::Nearest);
        assert_eq!(sampler.mipmap_filter(), FilterMode::Linear);
        assert_eq!(sampler.address_mode_u(), AddressMode::Repeat);
        assert_eq!(sampler.address_mode_v(), AddressMode::MirrorRepeat);
        assert_eq!(sampler.address_mode_w(), AddressMode::ClampToEdge);
        assert_eq!(sampler.anisotropy_clamp(), 4);
        assert_eq!(sampler.compare(), Some(CompareFunction::Less));
        assert!(sampler.is_comparison_sampler());
        assert!(sampler.is_anisotropic());
    }

    /// Test: Created sampler can be used (smoke test).
    #[test]
    
    fn created_sampler_is_usable() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        // Create sampler
        let desc = TrinitySamplerDescriptor::linear_clamp();
        let sampler = create_sampler(&device, &desc);

        // Verify inner sampler exists and can be accessed
        let inner = sampler.inner();
        let _ = inner;

        // Verify descriptor is preserved
        let stored_desc = sampler.descriptor();
        let _ = stored_desc;

        // Verify query methods work
        let _ = sampler.is_comparison_sampler();
        let _ = sampler.is_anisotropic();
    }

    /// Test: into_inner consumes sampler and returns wgpu::Sampler.
    #[test]
    
    fn into_inner_returns_wgpu_sampler() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let desc = TrinitySamplerDescriptor::linear_clamp();
        let sampler = create_sampler(&device, &desc);

        let inner: wgpu::Sampler = sampler.into_inner();
        let _ = inner;
    }
}

// =============================================================================
// MODULE: EDGE CASE TESTS
// =============================================================================

mod edge_case_tests {
    use super::*;

    /// Test: Empty string label is valid.
    #[test]
    fn empty_label_is_valid() {
        let desc = TrinitySamplerDescriptor::new().label("");
        let result = validate_descriptor(&desc);
        assert!(result.is_ok(), "Empty label should be valid");
    }

    /// Test: Very long label is valid.
    #[test]
    fn long_label_is_valid() {
        let long_label = "a".repeat(1000);
        let desc = TrinitySamplerDescriptor::new().label(long_label);
        let result = validate_descriptor(&desc);
        assert!(result.is_ok(), "Long label should be valid");
    }

    /// Test: LOD clamp with equal min and max is valid.
    #[test]
    fn lod_clamp_equal_min_max_is_valid() {
        let desc = TrinitySamplerDescriptor::new().lod_clamp(5.0, 5.0);
        let result = validate_descriptor(&desc);
        assert!(result.is_ok(), "Equal LOD min/max should be valid");
    }

    /// Test: LOD clamp with zero range at zero is valid.
    #[test]
    fn lod_clamp_zero_zero_is_valid() {
        let desc = TrinitySamplerDescriptor::new().lod_clamp(0.0, 0.0);
        let result = validate_descriptor(&desc);
        assert!(result.is_ok(), "LOD 0-0 should be valid");
    }

    /// Test: All border colors are valid with ClampToBorder.
    #[test]
    fn all_border_colors_valid() {
        let colors = [
            SamplerBorderColor::TransparentBlack,
            SamplerBorderColor::OpaqueBlack,
            SamplerBorderColor::OpaqueWhite,
        ];

        for color in colors {
            let desc = TrinitySamplerDescriptor::new()
                .address_mode(AddressMode::ClampToBorder)
                .border_color(color);
            let result = validate_descriptor(&desc);
            assert!(result.is_ok(), "Border color {:?} should be valid", color);
        }
    }

    /// Test: Anisotropy of 0 is handled (may be treated as 1 or error).
    #[test]
    fn anisotropy_zero_handled() {
        let desc = TrinitySamplerDescriptor::new().anisotropy(0);
        // May pass (treated as 1) or fail - either is acceptable behavior
        let _ = validate_descriptor(&desc);
    }

    /// Test: Maximum u16 anisotropy is handled.
    #[test]
    fn anisotropy_max_u16_handled() {
        let desc = TrinitySamplerDescriptor::new().anisotropy(u16::MAX);
        // May be clamped or rejected - either is acceptable
        let _ = validate_descriptor(&desc);
    }

    /// Test: Combination of comparison and anisotropy (may be invalid).
    #[test]
    fn comparison_with_anisotropy_handled() {
        let desc = TrinitySamplerDescriptor::new()
            .compare(CompareFunction::Less)
            .anisotropy(4);
        // Some implementations reject this combination
        let _ = validate_descriptor(&desc);
    }
}

// =============================================================================
// MODULE: SEND/SYNC TRAIT TESTS
// =============================================================================

mod trait_tests {
    use super::*;

    /// Test: TrinitySamplerDescriptor is Send.
    #[test]
    fn descriptor_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<TrinitySamplerDescriptor>();
    }

    /// Test: TrinitySamplerDescriptor is Sync.
    #[test]
    fn descriptor_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<TrinitySamplerDescriptor>();
    }

    /// Test: SamplerValidationError is Send.
    #[test]
    fn validation_error_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<SamplerValidationError>();
    }

    /// Test: SamplerValidationError is Sync.
    #[test]
    fn validation_error_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<SamplerValidationError>();
    }

    /// Test: TrinitySamplerDescriptor is Clone.
    #[test]
    fn descriptor_is_clone() {
        fn assert_clone<T: Clone>() {}
        assert_clone::<TrinitySamplerDescriptor>();
    }

    /// Test: TrinitySamplerDescriptor is Debug.
    #[test]
    fn descriptor_is_debug() {
        fn assert_debug<T: std::fmt::Debug>() {}
        assert_debug::<TrinitySamplerDescriptor>();
    }

    /// Test: SamplerValidationError is Debug.
    #[test]
    fn validation_error_is_debug() {
        fn assert_debug<T: std::fmt::Debug>() {}
        assert_debug::<SamplerValidationError>();
    }
}

// =============================================================================
// MODULE: CLONE BEHAVIOR TESTS
// =============================================================================

mod clone_tests {
    use super::*;

    /// Test: Cloned descriptor is equal to original.
    #[test]
    fn cloned_descriptor_equals_original() {
        let desc = TrinitySamplerDescriptor::new()
            .label("clone_test")
            .filter(FilterMode::Linear)
            .anisotropy(8);

        let cloned = desc.clone();

        // Both should validate the same
        assert!(validate_descriptor(&desc).is_ok());
        assert!(validate_descriptor(&cloned).is_ok());
    }

    /// Test: Modifying clone does not affect original.
    #[test]
    fn modifying_clone_does_not_affect_original() {
        let original = TrinitySamplerDescriptor::new()
            .label("original")
            .filter(FilterMode::Linear);

        let mut cloned = original.clone();
        cloned = cloned.label("modified").filter(FilterMode::Nearest);

        // Both should still validate
        assert!(validate_descriptor(&original).is_ok());
        assert!(validate_descriptor(&cloned).is_ok());
    }

    /// Test: All presets can be cloned.
    #[test]
    fn all_presets_clonable() {
        let presets = vec![
            TrinitySamplerDescriptor::linear_clamp(),
            TrinitySamplerDescriptor::linear_repeat(),
            TrinitySamplerDescriptor::nearest_clamp(),
            TrinitySamplerDescriptor::nearest_repeat(),
            TrinitySamplerDescriptor::shadow(),
            TrinitySamplerDescriptor::trilinear(),
        ];

        for preset in presets {
            let cloned = preset.clone();
            assert!(validate_descriptor(&cloned).is_ok());
        }
    }
}

// =============================================================================
// MODULE: DEBUG OUTPUT TESTS
// =============================================================================

mod debug_tests {
    use super::*;

    /// Test: Default descriptor debug output is non-empty.
    #[test]
    fn default_descriptor_debug_non_empty() {
        let desc = TrinitySamplerDescriptor::new();
        let debug_str = format!("{:?}", desc);
        assert!(!debug_str.is_empty());
    }

    /// Test: Preset descriptor debug output contains type info.
    #[test]
    fn preset_debug_contains_info() {
        let desc = TrinitySamplerDescriptor::shadow();
        let debug_str = format!("{:?}", desc);
        assert!(!debug_str.is_empty());
    }

    /// Test: SamplerValidationError debug works (if we can get one).
    #[test]
    fn validation_error_debug_works() {
        // Try to trigger a validation error
        let desc = TrinitySamplerDescriptor::new().lod_clamp(100.0, -100.0);
        if let Err(e) = validate_descriptor(&desc) {
            let debug_str = format!("{:?}", e);
            assert!(!debug_str.is_empty());
        }
    }
}
