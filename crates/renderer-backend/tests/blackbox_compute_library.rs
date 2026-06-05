// SPDX-License-Identifier: MIT
//
// blackbox_compute_library.rs -- Blackbox tests for T-WGPU-P3.10.6 ComputeLibrary Integration.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only the documented public types and functions:
//
//   - ComputeLibrary -- Unified compute shader library
//   - DispatchHelper -- Workgroup calculation utility
//   - PipelineStats -- Pipeline count statistics
//   - ComputeLibraryError -- Error type for compute operations
//
// PUBLIC API METHODS:
//   ComputeLibrary:
//     - new(device) -> Self
//     - new_lazy(device) -> Self
//     - reduce_sum(device, queue, buffer, count) -> Result<f32, ReductionError>
//     - reduce_min(device, queue, buffer, count) -> Result<f32, ReductionError>
//     - reduce_max(device, queue, buffer, count) -> Result<f32, ReductionError>
//     - reduce_minmax(device, queue, buffer, count) -> Result<(f32, f32), ReductionError>
//     - prefix_scan_exclusive(device, queue, buffer, count)
//     - prefix_scan_with_total(device, queue, buffer, count) -> Buffer
//     - stream_compact_nonzero(device, queue, input, output, count) -> Buffer
//     - stream_compact_with_predicates(device, queue, input, pred, output, count) -> Buffer
//     - stream_compact_vec4(device, queue, input, pred, output, count) -> Buffer
//     - radix_sort_keys(device, queue, keys, count)
//     - radix_sort_pairs(device, queue, keys, values, count)
//     - image_processor() -> &ImageProcessor
//     - blur_gaussian(device, encoder, src, temp, dst, uniforms, w, h)
//     - dispatch_reduction() -> DispatchHelper
//     - dispatch_prefix_scan() -> DispatchHelper
//     - dispatch_stream_compact() -> DispatchHelper
//     - dispatch_radix_sort() -> DispatchHelper
//     - dispatch_image() -> DispatchHelper
//     - stats() -> PipelineStats
//
//   DispatchHelper:
//     - new(workgroup_size, elements_per_thread) -> Self
//     - for_reduction() -> Self
//     - for_prefix_scan() -> Self
//     - for_stream_compact() -> Self
//     - for_radix_sort() -> Self
//     - for_image() -> Self
//     - elements_per_workgroup() -> u32
//     - num_workgroups(element_count) -> u32
//     - num_workgroups_2d(width, height) -> (u32, u32)
//
//   PipelineStats:
//     - reduction_pipelines: u32
//     - prefix_scan_pipelines: u32
//     - stream_compact_pipelines: u32
//     - radix_sort_pipelines: u32
//     - image_pipelines: u32
//     - total_pipelines: u32
//     - Display
//
// ACCEPTANCE CRITERIA (T-WGPU-P3.10.6):
//   1. All 25 pipelines created (3 reduction + 4 scan + 6 compact + 6 sort + 6 image)
//   2. Initialization at startup via ComputeLibrary::new(device)
//   3. Easy dispatch helpers (reduce_sum, prefix_scan, etc.)
//   4. DispatchHelper utility for workgroup calculations
//
// TEST CATEGORIES:
//   1. API Tests - Public interface, types exist
//   2. DispatchHelper - Workgroup calculations
//   3. PipelineStats - Statistics tracking
//   4. Error Types - Error variants exist
//   5. Re-exports - All sub-module types accessible
//   6. Thread Safety - Send + Sync bounds
//
// Total target: 40+ tests

use renderer_backend::compute_library::{
    // Main types
    ComputeLibrary,
    ComputeLibraryError,
    DispatchHelper,
    PipelineStats,
    // Re-exported types
    BlurUniforms,
    CompactParams,
    DownsampleUniforms,
    FilterMode,
    HistogramUniforms,
    ImageProcessor,
    PrefixScanError,
    PrefixScanPipeline,
    PredicateType,
    RadixSortError,
    RadixSortParams,
    RadixSortPipeline,
    ReductionError,
    ReductionOperation,
    ReductionParams,
    ReductionPipeline,
    ScanParams,
    StreamCompactError,
    StreamCompactPipeline,
    TonemapMode,
    TonemapUniforms,
};

// ============================================================================
// CATEGORY 1: API TESTS - Public interface verification
// ============================================================================

mod api_tests {
    use super::*;

    #[test]
    fn compute_library_type_exists() {
        fn _assert_type_exists(_: &ComputeLibrary) {}
    }

    #[test]
    fn dispatch_helper_type_exists() {
        fn _assert_type_exists(_: DispatchHelper) {}
    }

    #[test]
    fn pipeline_stats_type_exists() {
        fn _assert_type_exists(_: PipelineStats) {}
    }

    #[test]
    fn compute_library_error_type_exists() {
        fn _assert_type_exists(_: ComputeLibraryError) {}
    }

    #[test]
    fn reduction_pipeline_type_exists() {
        fn _assert_type_exists(_: &ReductionPipeline) {}
    }

    #[test]
    fn prefix_scan_pipeline_type_exists() {
        fn _assert_type_exists(_: &PrefixScanPipeline) {}
    }

    #[test]
    fn stream_compact_pipeline_type_exists() {
        fn _assert_type_exists(_: &StreamCompactPipeline) {}
    }

    #[test]
    fn radix_sort_pipeline_type_exists() {
        fn _assert_type_exists(_: &RadixSortPipeline) {}
    }

    #[test]
    fn image_processor_type_exists() {
        fn _assert_type_exists(_: &ImageProcessor) {}
    }
}

// ============================================================================
// CATEGORY 2: DISPATCH HELPER TESTS
// ============================================================================

mod dispatch_helper_tests {
    use super::*;

    #[test]
    fn new_creates_helper_with_settings() {
        let helper = DispatchHelper::new(128, 4);
        assert_eq!(helper.workgroup_size, 128);
        assert_eq!(helper.elements_per_thread, 4);
    }

    #[test]
    fn for_reduction_returns_correct_settings() {
        let helper = DispatchHelper::for_reduction();
        assert_eq!(helper.workgroup_size, 256);
        assert_eq!(helper.elements_per_thread, 2);
    }

    #[test]
    fn for_prefix_scan_returns_correct_settings() {
        let helper = DispatchHelper::for_prefix_scan();
        assert_eq!(helper.workgroup_size, 256);
        assert_eq!(helper.elements_per_thread, 2);
    }

    #[test]
    fn for_stream_compact_returns_correct_settings() {
        let helper = DispatchHelper::for_stream_compact();
        assert_eq!(helper.workgroup_size, 256);
        assert_eq!(helper.elements_per_thread, 1);
    }

    #[test]
    fn for_radix_sort_returns_correct_settings() {
        let helper = DispatchHelper::for_radix_sort();
        assert_eq!(helper.workgroup_size, 256);
        assert_eq!(helper.elements_per_thread, 4);
    }

    #[test]
    fn for_image_returns_correct_settings() {
        let helper = DispatchHelper::for_image();
        assert_eq!(helper.workgroup_size, 8);
        assert_eq!(helper.elements_per_thread, 1);
    }

    #[test]
    fn elements_per_workgroup_calculates_correctly() {
        let helper = DispatchHelper::new(256, 2);
        assert_eq!(helper.elements_per_workgroup(), 512);

        let helper2 = DispatchHelper::new(256, 4);
        assert_eq!(helper2.elements_per_workgroup(), 1024);
    }

    #[test]
    fn num_workgroups_single_workgroup() {
        let helper = DispatchHelper::for_reduction(); // 512 elements per workgroup
        assert_eq!(helper.num_workgroups(1), 1);
        assert_eq!(helper.num_workgroups(256), 1);
        assert_eq!(helper.num_workgroups(512), 1);
    }

    #[test]
    fn num_workgroups_multiple_workgroups() {
        let helper = DispatchHelper::for_reduction(); // 512 elements per workgroup
        assert_eq!(helper.num_workgroups(513), 2);
        assert_eq!(helper.num_workgroups(1024), 2);
        assert_eq!(helper.num_workgroups(1025), 3);
    }

    #[test]
    fn num_workgroups_large_counts() {
        let helper = DispatchHelper::for_reduction();
        assert_eq!(helper.num_workgroups(1_000_000), 1954); // ceil(1M / 512)
    }

    #[test]
    fn num_workgroups_2d_exact_fit() {
        let helper = DispatchHelper::for_image(); // 8x8 workgroups
        let (x, y) = helper.num_workgroups_2d(64, 64);
        assert_eq!(x, 8);
        assert_eq!(y, 8);
    }

    #[test]
    fn num_workgroups_2d_with_remainder() {
        let helper = DispatchHelper::for_image(); // 8x8 workgroups
        let (x, y) = helper.num_workgroups_2d(800, 600);
        assert_eq!(x, 100); // ceil(800 / 8)
        assert_eq!(y, 75);  // ceil(600 / 8)
    }

    #[test]
    fn num_workgroups_2d_small() {
        let helper = DispatchHelper::for_image();
        let (x, y) = helper.num_workgroups_2d(1, 1);
        assert_eq!(x, 1);
        assert_eq!(y, 1);
    }

    #[test]
    fn default_helper() {
        let helper = DispatchHelper::default();
        assert_eq!(helper.workgroup_size, 256);
        assert_eq!(helper.elements_per_thread, 1);
    }
}

// ============================================================================
// CATEGORY 3: PIPELINE STATS TESTS
// ============================================================================

mod pipeline_stats_tests {
    use super::*;

    #[test]
    fn stats_has_expected_fields() {
        let stats = PipelineStats {
            reduction_pipelines: 3,
            prefix_scan_pipelines: 4,
            stream_compact_pipelines: 6,
            radix_sort_pipelines: 6,
            image_pipelines: 6,
            total_pipelines: 25,
        };

        assert_eq!(stats.reduction_pipelines, 3);
        assert_eq!(stats.prefix_scan_pipelines, 4);
        assert_eq!(stats.stream_compact_pipelines, 6);
        assert_eq!(stats.radix_sort_pipelines, 6);
        assert_eq!(stats.image_pipelines, 6);
        assert_eq!(stats.total_pipelines, 25);
    }

    #[test]
    fn stats_display_contains_total() {
        let stats = PipelineStats {
            reduction_pipelines: 3,
            prefix_scan_pipelines: 4,
            stream_compact_pipelines: 6,
            radix_sort_pipelines: 6,
            image_pipelines: 6,
            total_pipelines: 25,
        };

        let s = format!("{}", stats);
        assert!(s.contains("25 pipelines"), "Display should contain '25 pipelines': {}", s);
    }

    #[test]
    fn stats_display_contains_categories() {
        let stats = PipelineStats {
            reduction_pipelines: 3,
            prefix_scan_pipelines: 4,
            stream_compact_pipelines: 6,
            radix_sort_pipelines: 6,
            image_pipelines: 6,
            total_pipelines: 25,
        };

        let s = format!("{}", stats);
        assert!(s.contains("reduction=3"), "Display should contain 'reduction=3': {}", s);
        assert!(s.contains("scan=4"), "Display should contain 'scan=4': {}", s);
        assert!(s.contains("compact=6"), "Display should contain 'compact=6': {}", s);
        assert!(s.contains("sort=6"), "Display should contain 'sort=6': {}", s);
        assert!(s.contains("image=6"), "Display should contain 'image=6': {}", s);
    }

    #[test]
    fn stats_is_copy() {
        let stats = PipelineStats {
            reduction_pipelines: 3,
            prefix_scan_pipelines: 4,
            stream_compact_pipelines: 6,
            radix_sort_pipelines: 6,
            image_pipelines: 6,
            total_pipelines: 25,
        };

        let stats2 = stats; // Copy
        assert_eq!(stats.total_pipelines, stats2.total_pipelines);
    }

    #[test]
    fn stats_is_clone() {
        let stats = PipelineStats {
            reduction_pipelines: 3,
            prefix_scan_pipelines: 4,
            stream_compact_pipelines: 6,
            radix_sort_pipelines: 6,
            image_pipelines: 6,
            total_pipelines: 25,
        };

        let stats2 = stats.clone();
        assert_eq!(stats.total_pipelines, stats2.total_pipelines);
    }

    #[test]
    fn stats_is_debug() {
        let stats = PipelineStats {
            reduction_pipelines: 3,
            prefix_scan_pipelines: 4,
            stream_compact_pipelines: 6,
            radix_sort_pipelines: 6,
            image_pipelines: 6,
            total_pipelines: 25,
        };

        let s = format!("{:?}", stats);
        assert!(s.contains("PipelineStats"), "Debug should contain struct name: {}", s);
    }
}

// ============================================================================
// CATEGORY 4: ERROR TYPE TESTS
// ============================================================================

mod error_tests {
    use super::*;

    #[test]
    fn reduction_error_empty_input_variant() {
        let err = ReductionError::EmptyInput;
        let s = format!("{}", err);
        assert!(s.to_lowercase().contains("empty"), "EmptyInput error: {}", s);
    }

    #[test]
    fn reduction_error_timeout_variant() {
        let err = ReductionError::Timeout(1000);
        let s = format!("{}", err);
        assert!(s.contains("1000"), "Timeout error: {}", s);
    }

    #[test]
    fn prefix_scan_error_empty_input_variant() {
        let err = PrefixScanError::EmptyInput;
        let s = format!("{}", err);
        assert!(s.to_lowercase().contains("zero") || s.to_lowercase().contains("empty"),
                "EmptyInput error: {}", s);
    }

    #[test]
    fn prefix_scan_error_too_large_variant() {
        let err = PrefixScanError::InputTooLarge { size: 1000, max: 500 };
        let s = format!("{}", err);
        assert!(s.contains("1000"), "InputTooLarge error: {}", s);
    }

    #[test]
    fn stream_compact_error_empty_input_variant() {
        let err = StreamCompactError::EmptyInput;
        let s = format!("{}", err);
        assert!(s.to_lowercase().contains("zero") || s.to_lowercase().contains("empty"),
                "EmptyInput error: {}", s);
    }

    #[test]
    fn radix_sort_error_empty_input_variant() {
        let err = RadixSortError::EmptyInput;
        let s = format!("{}", err);
        assert!(s.to_lowercase().contains("zero") || s.to_lowercase().contains("empty"),
                "EmptyInput error: {}", s);
    }

    #[test]
    fn compute_library_error_from_reduction() {
        let err: ComputeLibraryError = ReductionError::EmptyInput.into();
        let s = format!("{}", err);
        assert!(s.to_lowercase().contains("reduction"), "ComputeLibraryError: {}", s);
    }

    #[test]
    fn compute_library_error_from_prefix_scan() {
        let err: ComputeLibraryError = PrefixScanError::EmptyInput.into();
        let s = format!("{}", err);
        assert!(s.to_lowercase().contains("scan"), "ComputeLibraryError: {}", s);
    }

    #[test]
    fn compute_library_error_from_stream_compact() {
        let err: ComputeLibraryError = StreamCompactError::EmptyInput.into();
        let s = format!("{}", err);
        assert!(s.to_lowercase().contains("compact"), "ComputeLibraryError: {}", s);
    }

    #[test]
    fn compute_library_error_from_radix_sort() {
        let err: ComputeLibraryError = RadixSortError::EmptyInput.into();
        let s = format!("{}", err);
        assert!(s.to_lowercase().contains("sort"), "ComputeLibraryError: {}", s);
    }

    #[test]
    fn compute_library_error_is_std_error() {
        fn _assert_error<E: std::error::Error>() {}
        _assert_error::<ComputeLibraryError>();
    }

    #[test]
    fn reduction_error_is_std_error() {
        fn _assert_error<E: std::error::Error>() {}
        _assert_error::<ReductionError>();
    }

    #[test]
    fn prefix_scan_error_is_std_error() {
        fn _assert_error<E: std::error::Error>() {}
        _assert_error::<PrefixScanError>();
    }

    #[test]
    fn stream_compact_error_is_std_error() {
        fn _assert_error<E: std::error::Error>() {}
        _assert_error::<StreamCompactError>();
    }

    #[test]
    fn radix_sort_error_is_std_error() {
        fn _assert_error<E: std::error::Error>() {}
        _assert_error::<RadixSortError>();
    }
}

// ============================================================================
// CATEGORY 5: RE-EXPORT TESTS
// ============================================================================

mod reexport_tests {
    use super::*;

    #[test]
    fn reduction_params_constructable() {
        let params = ReductionParams::new(1024, 0);
        assert_eq!(params.input_size, 1024);
        assert_eq!(params.output_offset, 0);
    }

    #[test]
    fn reduction_operation_variants() {
        assert_eq!(ReductionOperation::Sum.identity(), 0.0);
        assert_eq!(ReductionOperation::Min.identity(), f32::MAX);
        assert_eq!(ReductionOperation::Max.identity(), f32::MIN);
    }

    #[test]
    fn scan_params_exclusive() {
        let params = ScanParams::exclusive(512);
        assert_eq!(params.input_size, 512);
        assert_eq!(params.is_inclusive, 0);
    }

    #[test]
    fn scan_params_inclusive() {
        let params = ScanParams::inclusive(512);
        assert_eq!(params.input_size, 512);
        assert_eq!(params.is_inclusive, 1);
    }

    #[test]
    fn compact_params_constructable() {
        let params = CompactParams::new(256);
        assert_eq!(params.input_size, 256);
    }

    #[test]
    fn predicate_type_variants() {
        let _ = PredicateType::NonZero;
        let _ = PredicateType::GreaterThan(10);
        let _ = PredicateType::LessThan(10);
        let _ = PredicateType::Equal(10);
        let _ = PredicateType::NotEqual(10);
    }

    #[test]
    fn radix_sort_params_constructable() {
        let params = RadixSortParams::new(1024, 0, 4);
        assert_eq!(params.input_size, 1024);
        assert_eq!(params.pass_number, 0);
        assert_eq!(params.num_workgroups, 4);
    }

    #[test]
    fn blur_uniforms_constructable() {
        let uniforms = BlurUniforms::new(1920, 1080);
        assert_eq!(uniforms.src_dims, [1920, 1080]);
        assert_eq!(uniforms.dst_dims, [1920, 1080]);
    }

    #[test]
    fn blur_uniforms_with_scale() {
        let uniforms = BlurUniforms::with_scale(1920, 1080, 2.0);
        assert_eq!(uniforms.blur_scale, 2.0);
    }

    #[test]
    fn downsample_uniforms_constructable() {
        let uniforms = DownsampleUniforms::new(1920, 1080, FilterMode::Box, 0);
        assert_eq!(uniforms.src_dims, [1920, 1080]);
        assert_eq!(uniforms.dst_dims, [960, 540]);
    }

    #[test]
    fn filter_mode_variants() {
        assert_eq!(FilterMode::Box.as_u32(), 0);
        assert_eq!(FilterMode::Bilinear.as_u32(), 1);
        assert_eq!(FilterMode::Karis.as_u32(), 2);
    }

    #[test]
    fn histogram_uniforms_constructable() {
        let uniforms = HistogramUniforms::new(1920, 1080);
        assert_eq!(uniforms.src_dims, [1920, 1080]);
        assert_eq!(uniforms.num_pixels, 1920 * 1080);
    }

    #[test]
    fn tonemap_uniforms_constructable() {
        let uniforms = TonemapUniforms::new(1920, 1080);
        assert_eq!(uniforms.src_dims, [1920, 1080]);
        assert_eq!(uniforms.gamma, 2.2);
    }

    #[test]
    fn tonemap_mode_variants() {
        assert_eq!(TonemapMode::Aces.as_u32(), 0);
        assert_eq!(TonemapMode::Reinhard.as_u32(), 1);
        assert_eq!(TonemapMode::Uncharted2.as_u32(), 2);
        assert_eq!(TonemapMode::AcesFitted.as_u32(), 3);
    }
}

// ============================================================================
// CATEGORY 6: STATIC DISPATCH HELPERS
// ============================================================================

mod static_dispatch_tests {
    use super::*;

    #[test]
    fn dispatch_reduction_is_const() {
        const HELPER: DispatchHelper = ComputeLibrary::dispatch_reduction();
        assert_eq!(HELPER.workgroup_size, 256);
        assert_eq!(HELPER.elements_per_thread, 2);
    }

    #[test]
    fn dispatch_prefix_scan_is_const() {
        const HELPER: DispatchHelper = ComputeLibrary::dispatch_prefix_scan();
        assert_eq!(HELPER.workgroup_size, 256);
        assert_eq!(HELPER.elements_per_thread, 2);
    }

    #[test]
    fn dispatch_stream_compact_is_const() {
        const HELPER: DispatchHelper = ComputeLibrary::dispatch_stream_compact();
        assert_eq!(HELPER.workgroup_size, 256);
        assert_eq!(HELPER.elements_per_thread, 1);
    }

    #[test]
    fn dispatch_radix_sort_is_const() {
        const HELPER: DispatchHelper = ComputeLibrary::dispatch_radix_sort();
        assert_eq!(HELPER.workgroup_size, 256);
        assert_eq!(HELPER.elements_per_thread, 4);
    }

    #[test]
    fn dispatch_image_is_const() {
        const HELPER: DispatchHelper = ComputeLibrary::dispatch_image();
        assert_eq!(HELPER.workgroup_size, 8);
        assert_eq!(HELPER.elements_per_thread, 1);
    }
}

// ============================================================================
// CATEGORY 7: BYTEMUCK POD/ZEROABLE TESTS
// ============================================================================

mod pod_tests {
    use super::*;

    #[test]
    fn reduction_params_is_pod() {
        fn _assert_pod<T: bytemuck::Pod>() {}
        _assert_pod::<ReductionParams>();
    }

    #[test]
    fn scan_params_is_pod() {
        fn _assert_pod<T: bytemuck::Pod>() {}
        _assert_pod::<ScanParams>();
    }

    #[test]
    fn compact_params_is_pod() {
        fn _assert_pod<T: bytemuck::Pod>() {}
        _assert_pod::<CompactParams>();
    }

    #[test]
    fn radix_sort_params_is_pod() {
        fn _assert_pod<T: bytemuck::Pod>() {}
        _assert_pod::<RadixSortParams>();
    }

    #[test]
    fn blur_uniforms_is_pod() {
        fn _assert_pod<T: bytemuck::Pod>() {}
        _assert_pod::<BlurUniforms>();
    }

    #[test]
    fn downsample_uniforms_is_pod() {
        fn _assert_pod<T: bytemuck::Pod>() {}
        _assert_pod::<DownsampleUniforms>();
    }

    #[test]
    fn histogram_uniforms_is_pod() {
        fn _assert_pod<T: bytemuck::Pod>() {}
        _assert_pod::<HistogramUniforms>();
    }

    #[test]
    fn tonemap_uniforms_is_pod() {
        fn _assert_pod<T: bytemuck::Pod>() {}
        _assert_pod::<TonemapUniforms>();
    }
}

// ============================================================================
// CATEGORY 8: EDGE CASES
// ============================================================================

mod edge_case_tests {
    use super::*;

    #[test]
    fn dispatch_helper_zero_elements() {
        let helper = DispatchHelper::for_reduction();
        assert_eq!(helper.num_workgroups(0), 0);
    }

    #[test]
    fn dispatch_helper_max_u32() {
        let helper = DispatchHelper::for_reduction();
        // Should not panic
        let _ = helper.num_workgroups(u32::MAX);
    }

    #[test]
    fn scan_params_with_offset() {
        let params = ScanParams::exclusive(256).with_offset(128);
        assert_eq!(params.block_offset, 128);
    }

    #[test]
    fn compact_params_with_stride() {
        let params = CompactParams::new(256)
            .with_input_stride(4)
            .with_output_stride(4);
        assert_eq!(params.input_stride, 4);
        assert_eq!(params.output_stride, 4);
    }

    #[test]
    fn downsample_uniforms_custom_dst() {
        let uniforms = DownsampleUniforms::with_dst_size(
            1024, 768, 512, 384, FilterMode::Karis, 1
        );
        assert_eq!(uniforms.src_dims, [1024, 768]);
        assert_eq!(uniforms.dst_dims, [512, 384]);
        assert_eq!(uniforms.mip_level, 1);
    }

    #[test]
    fn histogram_uniforms_custom_range() {
        let uniforms = HistogramUniforms::with_range(1920, 1080, -8.0, 8.0);
        assert_eq!(uniforms.min_luminance, -8.0);
        assert_eq!(uniforms.max_luminance, 8.0);
    }

    #[test]
    fn tonemap_uniforms_full_config() {
        let uniforms = TonemapUniforms::full(
            1920, 1080, 1.5, 2.4, TonemapMode::Uncharted2, 8.0
        );
        assert_eq!(uniforms.exposure, 1.5);
        assert_eq!(uniforms.gamma, 2.4);
        assert_eq!(uniforms.mode, TonemapMode::Uncharted2.as_u32());
        assert_eq!(uniforms.white_point, 8.0);
    }
}
