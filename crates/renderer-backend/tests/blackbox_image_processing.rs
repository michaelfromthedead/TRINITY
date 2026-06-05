// Blackbox contract tests for T-WGPU-P3.10.5 Image Processing.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::compute_library::*` -- no internal fields,
// no private methods, no implementation details.
//
// Acceptance criteria (T-WGPU-P3.10.5):
//   1. blur_horizontal.wgsl - Horizontal Gaussian blur with shared memory
//   2. blur_vertical.wgsl - Vertical Gaussian blur with shared memory
//   3. downsample.wgsl - 2x downsampling with Box/Bilinear/Karis filters
//   4. histogram.wgsl - 256-bin luminance histogram
//   5. tonemapping.wgsl - HDR tonemapping (ACES, Reinhard, Uncharted2, AcesFitted)
//   6. Shared memory tile optimization
//
// Coverage:
//   1.  FilterMode enum variants and u32 conversion
//   2.  TonemapMode enum variants and u32 conversion
//   3.  BlurUniforms construction and field access
//   4.  DownsampleUniforms construction and field access
//   5.  HistogramUniforms construction and field access
//   6.  TonemapUniforms construction and field access
//   7.  Bytemuck compatibility for all uniform structs
//   8.  16-byte alignment for GPU uniform buffers
//   9.  Workgroup size helpers
//  10.  Default implementations
//  11.  Clone/Copy/Debug traits
//  12.  ImageProcessor type existence
//  13.  Post-processing pipeline integration
//  14.  Mip chain generation scenarios
//  15.  HDR to LDR conversion scenarios

use renderer_backend::compute_library::{
    BlurUniforms, DownsampleUniforms, FilterMode, HistogramUniforms, ImageProcessor,
    TonemapMode, TonemapUniforms,
};

use renderer_backend::compute_library::image_processing::{
    compute_workgroups_8x8, compute_workgroups_blur_h, compute_workgroups_blur_v,
    BLUR_KERNEL_RADIUS, BLUR_WORKGROUP_SIZE, HISTOGRAM_BINS, IMAGE_WORKGROUP_SIZE,
};

// =============================================================================
// FilterMode Tests
// =============================================================================

#[test]
fn test_filter_mode_box_variant() {
    let mode = FilterMode::Box;
    assert_eq!(mode.as_u32(), 0);
}

#[test]
fn test_filter_mode_bilinear_variant() {
    let mode = FilterMode::Bilinear;
    assert_eq!(mode.as_u32(), 1);
}

#[test]
fn test_filter_mode_karis_variant() {
    let mode = FilterMode::Karis;
    assert_eq!(mode.as_u32(), 2);
}

#[test]
fn test_filter_mode_default_is_box() {
    let mode = FilterMode::default();
    assert_eq!(mode, FilterMode::Box);
}

#[test]
fn test_filter_mode_clone() {
    let mode = FilterMode::Karis;
    let cloned = mode.clone();
    assert_eq!(mode, cloned);
}

#[test]
fn test_filter_mode_copy() {
    let mode = FilterMode::Bilinear;
    let copied: FilterMode = mode; // Copy
    assert_eq!(mode, copied);
}

#[test]
fn test_filter_mode_debug() {
    let mode = FilterMode::Box;
    let debug_str = format!("{:?}", mode);
    assert!(debug_str.contains("Box"));
}

#[test]
fn test_filter_mode_hash_consistency() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(FilterMode::Box);
    set.insert(FilterMode::Bilinear);
    set.insert(FilterMode::Karis);
    assert_eq!(set.len(), 3);
    // Inserting duplicate should not increase size
    set.insert(FilterMode::Box);
    assert_eq!(set.len(), 3);
}

// =============================================================================
// TonemapMode Tests
// =============================================================================

#[test]
fn test_tonemap_mode_aces_variant() {
    let mode = TonemapMode::Aces;
    assert_eq!(mode.as_u32(), 0);
}

#[test]
fn test_tonemap_mode_reinhard_variant() {
    let mode = TonemapMode::Reinhard;
    assert_eq!(mode.as_u32(), 1);
}

#[test]
fn test_tonemap_mode_uncharted2_variant() {
    let mode = TonemapMode::Uncharted2;
    assert_eq!(mode.as_u32(), 2);
}

#[test]
fn test_tonemap_mode_aces_fitted_variant() {
    let mode = TonemapMode::AcesFitted;
    assert_eq!(mode.as_u32(), 3);
}

#[test]
fn test_tonemap_mode_default_is_aces() {
    let mode = TonemapMode::default();
    assert_eq!(mode, TonemapMode::Aces);
}

#[test]
fn test_tonemap_mode_clone() {
    let mode = TonemapMode::Uncharted2;
    let cloned = mode.clone();
    assert_eq!(mode, cloned);
}

#[test]
fn test_tonemap_mode_copy() {
    let mode = TonemapMode::AcesFitted;
    let copied: TonemapMode = mode; // Copy
    assert_eq!(mode, copied);
}

#[test]
fn test_tonemap_mode_debug() {
    let mode = TonemapMode::Reinhard;
    let debug_str = format!("{:?}", mode);
    assert!(debug_str.contains("Reinhard"));
}

#[test]
fn test_tonemap_mode_hash_consistency() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(TonemapMode::Aces);
    set.insert(TonemapMode::Reinhard);
    set.insert(TonemapMode::Uncharted2);
    set.insert(TonemapMode::AcesFitted);
    assert_eq!(set.len(), 4);
}

// =============================================================================
// BlurUniforms Tests
// =============================================================================

#[test]
fn test_blur_uniforms_new() {
    let uniforms = BlurUniforms::new(1920, 1080);
    assert_eq!(uniforms.src_dims, [1920, 1080]);
    assert_eq!(uniforms.dst_dims, [1920, 1080]);
    assert_eq!(uniforms.blur_scale, 1.0);
}

#[test]
fn test_blur_uniforms_with_scale() {
    let uniforms = BlurUniforms::with_scale(1920, 1080, 2.5);
    assert_eq!(uniforms.src_dims, [1920, 1080]);
    assert_eq!(uniforms.dst_dims, [1920, 1080]);
    assert_eq!(uniforms.blur_scale, 2.5);
}

#[test]
fn test_blur_uniforms_small_dimensions() {
    let uniforms = BlurUniforms::new(1, 1);
    assert_eq!(uniforms.src_dims, [1, 1]);
    assert_eq!(uniforms.dst_dims, [1, 1]);
}

#[test]
fn test_blur_uniforms_large_dimensions() {
    let uniforms = BlurUniforms::new(8192, 4096);
    assert_eq!(uniforms.src_dims, [8192, 4096]);
    assert_eq!(uniforms.dst_dims, [8192, 4096]);
}

#[test]
fn test_blur_uniforms_zero_scale() {
    let uniforms = BlurUniforms::with_scale(1920, 1080, 0.0);
    assert_eq!(uniforms.blur_scale, 0.0);
}

#[test]
fn test_blur_uniforms_clone() {
    let uniforms = BlurUniforms::new(1920, 1080);
    let cloned = uniforms.clone();
    assert_eq!(uniforms.src_dims, cloned.src_dims);
    assert_eq!(uniforms.dst_dims, cloned.dst_dims);
    assert_eq!(uniforms.blur_scale, cloned.blur_scale);
}

#[test]
fn test_blur_uniforms_copy() {
    let uniforms = BlurUniforms::new(1920, 1080);
    let copied: BlurUniforms = uniforms; // Copy
    assert_eq!(uniforms.src_dims, copied.src_dims);
}

#[test]
fn test_blur_uniforms_debug() {
    let uniforms = BlurUniforms::new(1920, 1080);
    let debug_str = format!("{:?}", uniforms);
    assert!(debug_str.contains("BlurUniforms"));
}

#[test]
fn test_blur_uniforms_16byte_alignment() {
    assert_eq!(std::mem::size_of::<BlurUniforms>() % 16, 0);
}

#[test]
fn test_blur_uniforms_bytemuck_pod() {
    let uniforms = BlurUniforms::new(1920, 1080);
    let bytes: &[u8] = bytemuck::bytes_of(&uniforms);
    assert_eq!(bytes.len(), std::mem::size_of::<BlurUniforms>());
}

#[test]
fn test_blur_uniforms_bytemuck_zeroed() {
    let zeroed: BlurUniforms = bytemuck::Zeroable::zeroed();
    assert_eq!(zeroed.src_dims, [0, 0]);
    assert_eq!(zeroed.dst_dims, [0, 0]);
    assert_eq!(zeroed.blur_scale, 0.0);
}

// =============================================================================
// DownsampleUniforms Tests
// =============================================================================

#[test]
fn test_downsample_uniforms_new_box() {
    let uniforms = DownsampleUniforms::new(1920, 1080, FilterMode::Box, 0);
    assert_eq!(uniforms.src_dims, [1920, 1080]);
    assert_eq!(uniforms.dst_dims, [960, 540]);
    assert_eq!(uniforms.filter_mode, 0);
    assert_eq!(uniforms.mip_level, 0);
}

#[test]
fn test_downsample_uniforms_new_bilinear() {
    let uniforms = DownsampleUniforms::new(1920, 1080, FilterMode::Bilinear, 1);
    assert_eq!(uniforms.filter_mode, 1);
    assert_eq!(uniforms.mip_level, 1);
}

#[test]
fn test_downsample_uniforms_new_karis() {
    let uniforms = DownsampleUniforms::new(1920, 1080, FilterMode::Karis, 0);
    assert_eq!(uniforms.filter_mode, 2);
}

#[test]
fn test_downsample_uniforms_with_dst_size() {
    let uniforms = DownsampleUniforms::with_dst_size(
        1920, 1080,
        480, 270,
        FilterMode::Box,
        2,
    );
    assert_eq!(uniforms.src_dims, [1920, 1080]);
    assert_eq!(uniforms.dst_dims, [480, 270]);
    assert_eq!(uniforms.mip_level, 2);
}

#[test]
fn test_downsample_uniforms_2x_reduction() {
    // Verify standard 2x downsampling
    let uniforms = DownsampleUniforms::new(1920, 1080, FilterMode::Box, 0);
    assert_eq!(uniforms.dst_dims[0], uniforms.src_dims[0] / 2);
    assert_eq!(uniforms.dst_dims[1], uniforms.src_dims[1] / 2);
}

#[test]
fn test_downsample_uniforms_small_texture() {
    let uniforms = DownsampleUniforms::new(2, 2, FilterMode::Box, 0);
    assert_eq!(uniforms.dst_dims, [1, 1]);
}

#[test]
fn test_downsample_uniforms_odd_dimensions() {
    // Odd dimensions: 1921/2 = 960, 1081/2 = 540 (integer division)
    let uniforms = DownsampleUniforms::new(1921, 1081, FilterMode::Box, 0);
    assert_eq!(uniforms.dst_dims, [960, 540]);
}

#[test]
fn test_downsample_uniforms_clone() {
    let uniforms = DownsampleUniforms::new(1920, 1080, FilterMode::Karis, 0);
    let cloned = uniforms.clone();
    assert_eq!(uniforms.src_dims, cloned.src_dims);
    assert_eq!(uniforms.filter_mode, cloned.filter_mode);
}

#[test]
fn test_downsample_uniforms_copy() {
    let uniforms = DownsampleUniforms::new(1920, 1080, FilterMode::Box, 0);
    let copied: DownsampleUniforms = uniforms; // Copy
    assert_eq!(uniforms.dst_dims, copied.dst_dims);
}

#[test]
fn test_downsample_uniforms_debug() {
    let uniforms = DownsampleUniforms::new(1920, 1080, FilterMode::Box, 0);
    let debug_str = format!("{:?}", uniforms);
    assert!(debug_str.contains("DownsampleUniforms"));
}

#[test]
fn test_downsample_uniforms_16byte_alignment() {
    assert_eq!(std::mem::size_of::<DownsampleUniforms>() % 16, 0);
}

#[test]
fn test_downsample_uniforms_bytemuck_pod() {
    let uniforms = DownsampleUniforms::new(1920, 1080, FilterMode::Box, 0);
    let bytes: &[u8] = bytemuck::bytes_of(&uniforms);
    assert_eq!(bytes.len(), std::mem::size_of::<DownsampleUniforms>());
}

#[test]
fn test_downsample_uniforms_bytemuck_zeroed() {
    let zeroed: DownsampleUniforms = bytemuck::Zeroable::zeroed();
    assert_eq!(zeroed.src_dims, [0, 0]);
    assert_eq!(zeroed.dst_dims, [0, 0]);
    assert_eq!(zeroed.filter_mode, 0);
    assert_eq!(zeroed.mip_level, 0);
}

// =============================================================================
// HistogramUniforms Tests
// =============================================================================

#[test]
fn test_histogram_uniforms_new() {
    let uniforms = HistogramUniforms::new(1920, 1080);
    assert_eq!(uniforms.src_dims, [1920, 1080]);
    assert_eq!(uniforms.num_pixels, 1920 * 1080);
    assert_eq!(uniforms.min_luminance, -10.0);
    assert_eq!(uniforms.max_luminance, 4.0);
}

#[test]
fn test_histogram_uniforms_with_range() {
    let uniforms = HistogramUniforms::with_range(1920, 1080, -5.0, 10.0);
    assert_eq!(uniforms.min_luminance, -5.0);
    assert_eq!(uniforms.max_luminance, 10.0);
}

#[test]
fn test_histogram_uniforms_pixel_count() {
    let uniforms = HistogramUniforms::new(100, 100);
    assert_eq!(uniforms.num_pixels, 10000);
}

#[test]
fn test_histogram_uniforms_large_image() {
    let uniforms = HistogramUniforms::new(4096, 4096);
    assert_eq!(uniforms.num_pixels, 4096 * 4096);
}

#[test]
fn test_histogram_uniforms_clone() {
    let uniforms = HistogramUniforms::new(1920, 1080);
    let cloned = uniforms.clone();
    assert_eq!(uniforms.num_pixels, cloned.num_pixels);
}

#[test]
fn test_histogram_uniforms_copy() {
    let uniforms = HistogramUniforms::new(1920, 1080);
    let copied: HistogramUniforms = uniforms; // Copy
    assert_eq!(uniforms.min_luminance, copied.min_luminance);
}

#[test]
fn test_histogram_uniforms_debug() {
    let uniforms = HistogramUniforms::new(1920, 1080);
    let debug_str = format!("{:?}", uniforms);
    assert!(debug_str.contains("HistogramUniforms"));
}

#[test]
fn test_histogram_uniforms_16byte_alignment() {
    assert_eq!(std::mem::size_of::<HistogramUniforms>() % 16, 0);
}

#[test]
fn test_histogram_uniforms_bytemuck_pod() {
    let uniforms = HistogramUniforms::new(1920, 1080);
    let bytes: &[u8] = bytemuck::bytes_of(&uniforms);
    assert_eq!(bytes.len(), std::mem::size_of::<HistogramUniforms>());
}

#[test]
fn test_histogram_uniforms_bytemuck_zeroed() {
    let zeroed: HistogramUniforms = bytemuck::Zeroable::zeroed();
    assert_eq!(zeroed.src_dims, [0, 0]);
    assert_eq!(zeroed.num_pixels, 0);
    assert_eq!(zeroed.min_luminance, 0.0);
    assert_eq!(zeroed.max_luminance, 0.0);
}

// =============================================================================
// TonemapUniforms Tests
// =============================================================================

#[test]
fn test_tonemap_uniforms_new() {
    let uniforms = TonemapUniforms::new(1920, 1080);
    assert_eq!(uniforms.src_dims, [1920, 1080]);
    assert_eq!(uniforms.dst_dims, [1920, 1080]);
    assert_eq!(uniforms.exposure, 0.0);
    assert_eq!(uniforms.gamma, 2.2);
    assert_eq!(uniforms.mode, 0); // ACES
    assert_eq!(uniforms.white_point, 4.0);
}

#[test]
fn test_tonemap_uniforms_with_exposure() {
    let uniforms = TonemapUniforms::with_exposure(1920, 1080, 1.5);
    assert_eq!(uniforms.exposure, 1.5);
    assert_eq!(uniforms.gamma, 2.2); // Default gamma
    assert_eq!(uniforms.mode, 0);     // Default ACES
}

#[test]
fn test_tonemap_uniforms_full_aces() {
    let uniforms = TonemapUniforms::full(1920, 1080, 0.0, 2.2, TonemapMode::Aces, 4.0);
    assert_eq!(uniforms.mode, 0);
}

#[test]
fn test_tonemap_uniforms_full_reinhard() {
    let uniforms = TonemapUniforms::full(1920, 1080, 1.0, 2.2, TonemapMode::Reinhard, 6.0);
    assert_eq!(uniforms.mode, 1);
    assert_eq!(uniforms.white_point, 6.0);
}

#[test]
fn test_tonemap_uniforms_full_uncharted2() {
    let uniforms = TonemapUniforms::full(1920, 1080, 2.0, 2.4, TonemapMode::Uncharted2, 4.0);
    assert_eq!(uniforms.mode, 2);
    assert_eq!(uniforms.exposure, 2.0);
    assert_eq!(uniforms.gamma, 2.4);
}

#[test]
fn test_tonemap_uniforms_full_aces_fitted() {
    let uniforms = TonemapUniforms::full(1920, 1080, 0.5, 2.2, TonemapMode::AcesFitted, 4.0);
    assert_eq!(uniforms.mode, 3);
}

#[test]
fn test_tonemap_uniforms_negative_exposure() {
    // Negative exposure darkens the image
    let uniforms = TonemapUniforms::with_exposure(1920, 1080, -2.0);
    assert_eq!(uniforms.exposure, -2.0);
}

#[test]
fn test_tonemap_uniforms_clone() {
    let uniforms = TonemapUniforms::new(1920, 1080);
    let cloned = uniforms.clone();
    assert_eq!(uniforms.exposure, cloned.exposure);
    assert_eq!(uniforms.mode, cloned.mode);
}

#[test]
fn test_tonemap_uniforms_copy() {
    let uniforms = TonemapUniforms::new(1920, 1080);
    let copied: TonemapUniforms = uniforms; // Copy
    assert_eq!(uniforms.gamma, copied.gamma);
}

#[test]
fn test_tonemap_uniforms_debug() {
    let uniforms = TonemapUniforms::new(1920, 1080);
    let debug_str = format!("{:?}", uniforms);
    assert!(debug_str.contains("TonemapUniforms"));
}

#[test]
fn test_tonemap_uniforms_16byte_alignment() {
    assert_eq!(std::mem::size_of::<TonemapUniforms>() % 16, 0);
}

#[test]
fn test_tonemap_uniforms_bytemuck_pod() {
    let uniforms = TonemapUniforms::new(1920, 1080);
    let bytes: &[u8] = bytemuck::bytes_of(&uniforms);
    assert_eq!(bytes.len(), std::mem::size_of::<TonemapUniforms>());
}

#[test]
fn test_tonemap_uniforms_bytemuck_zeroed() {
    let zeroed: TonemapUniforms = bytemuck::Zeroable::zeroed();
    assert_eq!(zeroed.src_dims, [0, 0]);
    assert_eq!(zeroed.dst_dims, [0, 0]);
    assert_eq!(zeroed.exposure, 0.0);
    assert_eq!(zeroed.gamma, 0.0);
    assert_eq!(zeroed.mode, 0);
    assert_eq!(zeroed.white_point, 0.0);
}

// =============================================================================
// Constants Tests
// =============================================================================

#[test]
fn test_blur_workgroup_size() {
    assert_eq!(BLUR_WORKGROUP_SIZE, 128);
}

#[test]
fn test_image_workgroup_size() {
    assert_eq!(IMAGE_WORKGROUP_SIZE, 8);
}

#[test]
fn test_histogram_bins() {
    assert_eq!(HISTOGRAM_BINS, 256);
}

#[test]
fn test_blur_kernel_radius() {
    // 9-tap kernel = radius 4
    assert_eq!(BLUR_KERNEL_RADIUS, 4);
}

// =============================================================================
// Workgroup Helper Tests
// =============================================================================

#[test]
fn test_compute_workgroups_8x8_standard() {
    let (x, y) = compute_workgroups_8x8(1920, 1080);
    assert_eq!(x, 240);
    assert_eq!(y, 135);
}

#[test]
fn test_compute_workgroups_8x8_exact() {
    let (x, y) = compute_workgroups_8x8(8, 8);
    assert_eq!(x, 1);
    assert_eq!(y, 1);
}

#[test]
fn test_compute_workgroups_8x8_one_pixel() {
    let (x, y) = compute_workgroups_8x8(1, 1);
    assert_eq!(x, 1);
    assert_eq!(y, 1);
}

#[test]
fn test_compute_workgroups_8x8_odd() {
    let (x, y) = compute_workgroups_8x8(9, 9);
    assert_eq!(x, 2);
    assert_eq!(y, 2);
}

#[test]
fn test_compute_workgroups_blur_h_standard() {
    let (x, y) = compute_workgroups_blur_h(1920, 1080);
    assert_eq!(x, 15); // ceil(1920 / 128) = 15
    assert_eq!(y, 1080);
}

#[test]
fn test_compute_workgroups_blur_h_exact() {
    let (x, y) = compute_workgroups_blur_h(128, 100);
    assert_eq!(x, 1);
    assert_eq!(y, 100);
}

#[test]
fn test_compute_workgroups_blur_h_one_over() {
    let (x, y) = compute_workgroups_blur_h(129, 100);
    assert_eq!(x, 2);
    assert_eq!(y, 100);
}

#[test]
fn test_compute_workgroups_blur_v_standard() {
    let (x, y) = compute_workgroups_blur_v(1920, 1080);
    assert_eq!(x, 1920);
    assert_eq!(y, 9); // ceil(1080 / 128) = 9
}

#[test]
fn test_compute_workgroups_blur_v_exact() {
    let (x, y) = compute_workgroups_blur_v(100, 128);
    assert_eq!(x, 100);
    assert_eq!(y, 1);
}

#[test]
fn test_compute_workgroups_blur_v_one_over() {
    let (x, y) = compute_workgroups_blur_v(100, 129);
    assert_eq!(x, 100);
    assert_eq!(y, 2);
}

// =============================================================================
// ImageProcessor Type Existence Tests
// =============================================================================

#[test]
fn test_image_processor_type_exists() {
    // Verify the ImageProcessor type is accessible from the public API
    fn _check_type_exists<T>() {}
    _check_type_exists::<ImageProcessor>();
}

// =============================================================================
// Real-World Scenario Tests
// =============================================================================

#[test]
fn test_scenario_post_process_pipeline() {
    // Scenario: Set up uniforms for a post-processing pipeline
    // Input: 1920x1080 HDR -> Blur -> Tonemap -> Output

    let width = 1920u32;
    let height = 1080u32;

    // Step 1: Blur uniforms (2 passes: horizontal + vertical)
    let blur_h = BlurUniforms::new(width, height);
    let blur_v = BlurUniforms::new(width, height);

    // Verify blur dimensions match
    assert_eq!(blur_h.src_dims, blur_v.src_dims);
    assert_eq!(blur_h.dst_dims, blur_v.dst_dims);

    // Step 2: Tonemap uniforms (ACES with slight exposure boost)
    let tonemap = TonemapUniforms::with_exposure(width, height, 0.5);
    assert_eq!(tonemap.src_dims, [width, height]);
    assert_eq!(tonemap.exposure, 0.5);
    assert_eq!(tonemap.mode, TonemapMode::Aces.as_u32());
}

#[test]
fn test_scenario_mip_chain_generation() {
    // Scenario: Generate mip chain from 1024x1024 down to 1x1

    let mut mips = Vec::new();
    let mut w = 1024u32;
    let mut h = 1024u32;
    let mut level = 0u32;

    while w > 1 || h > 1 {
        // Use Karis filter for first mip (reduces fireflies), Box for rest
        let filter = if level == 0 { FilterMode::Karis } else { FilterMode::Box };
        let uniforms = DownsampleUniforms::new(w, h, filter, level);

        mips.push(uniforms);

        w = (w / 2).max(1);
        h = (h / 2).max(1);
        level += 1;
    }

    // 1024 -> 512 -> 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1 = 10 levels
    assert_eq!(mips.len(), 10);

    // First mip uses Karis
    assert_eq!(mips[0].filter_mode, FilterMode::Karis.as_u32());

    // Subsequent mips use Box
    for mip in &mips[1..] {
        assert_eq!(mip.filter_mode, FilterMode::Box.as_u32());
    }

    // Last mip goes to 1x1
    assert_eq!(mips.last().unwrap().dst_dims, [1, 1]);
}

#[test]
fn test_scenario_auto_exposure_histogram() {
    // Scenario: Compute histogram for auto-exposure

    let width = 1920u32;
    let height = 1080u32;

    // Set up histogram with HDR luminance range
    // log2(0.001) ~ -10, log2(100) ~ 6.6
    let uniforms = HistogramUniforms::with_range(width, height, -10.0, 7.0);

    assert_eq!(uniforms.num_pixels, 1920 * 1080);
    assert_eq!(uniforms.min_luminance, -10.0);
    assert_eq!(uniforms.max_luminance, 7.0);

    // Verify luminance range covers typical HDR content
    let range = uniforms.max_luminance - uniforms.min_luminance;
    assert!(range > 10.0, "HDR range should be substantial");
}

#[test]
fn test_scenario_hdr_to_sdr_conversion() {
    // Scenario: Convert HDR content to SDR display

    let width = 3840u32;  // 4K
    let height = 2160u32;

    // Different tonemapping options for different content
    let film_content = TonemapUniforms::full(
        width, height,
        0.0,  // No exposure adjustment
        2.2,  // Standard gamma
        TonemapMode::Aces,
        4.0,
    );

    let game_content = TonemapUniforms::full(
        width, height,
        0.5,  // Slight boost
        2.2,
        TonemapMode::Uncharted2,
        4.0,
    );

    let photo_content = TonemapUniforms::full(
        width, height,
        0.0,
        2.2,
        TonemapMode::AcesFitted,  // Most accurate
        4.0,
    );

    // Verify all have correct dimensions
    assert_eq!(film_content.src_dims, [width, height]);
    assert_eq!(game_content.src_dims, [width, height]);
    assert_eq!(photo_content.src_dims, [width, height]);

    // Verify different modes
    assert_ne!(film_content.mode, game_content.mode);
    assert_ne!(game_content.mode, photo_content.mode);
}

#[test]
fn test_scenario_bloom_with_karis_downsample() {
    // Scenario: Bloom pipeline with Karis filter for firefly reduction

    let width = 1920u32;
    let height = 1080u32;

    // Bloom downsample chain (5 levels for typical bloom)
    let mip0 = DownsampleUniforms::new(width, height, FilterMode::Karis, 0);
    let mip1 = DownsampleUniforms::new(mip0.dst_dims[0], mip0.dst_dims[1], FilterMode::Box, 1);
    let mip2 = DownsampleUniforms::new(mip1.dst_dims[0], mip1.dst_dims[1], FilterMode::Box, 2);
    let mip3 = DownsampleUniforms::new(mip2.dst_dims[0], mip2.dst_dims[1], FilterMode::Box, 3);
    let mip4 = DownsampleUniforms::new(mip3.dst_dims[0], mip3.dst_dims[1], FilterMode::Box, 4);

    // Verify Karis on first level
    assert_eq!(mip0.filter_mode, FilterMode::Karis.as_u32());

    // Verify progressive downsampling
    assert_eq!(mip0.dst_dims, [960, 540]);
    assert_eq!(mip1.dst_dims, [480, 270]);
    assert_eq!(mip2.dst_dims, [240, 135]);
    assert_eq!(mip3.dst_dims, [120, 67]);
    assert_eq!(mip4.dst_dims, [60, 33]);
}

// =============================================================================
// Edge Case Tests
// =============================================================================

#[test]
fn test_edge_case_minimum_texture_size() {
    let blur = BlurUniforms::new(1, 1);
    let downsample = DownsampleUniforms::new(2, 2, FilterMode::Box, 0);
    let histogram = HistogramUniforms::new(1, 1);
    let tonemap = TonemapUniforms::new(1, 1);

    assert_eq!(blur.src_dims, [1, 1]);
    assert_eq!(downsample.dst_dims, [1, 1]);
    assert_eq!(histogram.num_pixels, 1);
    assert_eq!(tonemap.src_dims, [1, 1]);
}

#[test]
fn test_edge_case_very_large_texture() {
    let width = 16384u32; // 16K
    let height = 8192u32;

    let blur = BlurUniforms::new(width, height);
    let histogram = HistogramUniforms::new(width, height);

    assert_eq!(blur.src_dims, [width, height]);
    assert_eq!(histogram.num_pixels, width * height);
}

#[test]
fn test_edge_case_non_power_of_two() {
    // Odd dimensions
    let uniforms = DownsampleUniforms::new(1000, 500, FilterMode::Box, 0);
    assert_eq!(uniforms.dst_dims, [500, 250]);

    // Prime dimensions
    let uniforms2 = DownsampleUniforms::new(1009, 503, FilterMode::Box, 0);
    assert_eq!(uniforms2.dst_dims, [504, 251]);
}

#[test]
fn test_edge_case_extreme_exposure() {
    // Very bright
    let bright = TonemapUniforms::with_exposure(1920, 1080, 10.0);
    assert_eq!(bright.exposure, 10.0);

    // Very dark
    let dark = TonemapUniforms::with_exposure(1920, 1080, -10.0);
    assert_eq!(dark.exposure, -10.0);
}

#[test]
fn test_edge_case_extreme_luminance_range() {
    // Very wide HDR range
    let wide = HistogramUniforms::with_range(1920, 1080, -20.0, 20.0);
    assert_eq!(wide.min_luminance, -20.0);
    assert_eq!(wide.max_luminance, 20.0);

    // Narrow range (for specific analysis)
    let narrow = HistogramUniforms::with_range(1920, 1080, 0.0, 1.0);
    assert_eq!(narrow.min_luminance, 0.0);
    assert_eq!(narrow.max_luminance, 1.0);
}

#[test]
fn test_edge_case_all_filter_modes_for_same_input() {
    let width = 1920u32;
    let height = 1080u32;

    let box_filter = DownsampleUniforms::new(width, height, FilterMode::Box, 0);
    let bilinear_filter = DownsampleUniforms::new(width, height, FilterMode::Bilinear, 0);
    let karis_filter = DownsampleUniforms::new(width, height, FilterMode::Karis, 0);

    // All produce same output dimensions
    assert_eq!(box_filter.dst_dims, bilinear_filter.dst_dims);
    assert_eq!(bilinear_filter.dst_dims, karis_filter.dst_dims);

    // But different filter modes
    assert_ne!(box_filter.filter_mode, bilinear_filter.filter_mode);
    assert_ne!(bilinear_filter.filter_mode, karis_filter.filter_mode);
}

#[test]
fn test_edge_case_all_tonemap_modes_for_same_input() {
    let width = 1920u32;
    let height = 1080u32;

    let aces = TonemapUniforms::full(width, height, 0.0, 2.2, TonemapMode::Aces, 4.0);
    let reinhard = TonemapUniforms::full(width, height, 0.0, 2.2, TonemapMode::Reinhard, 4.0);
    let uncharted = TonemapUniforms::full(width, height, 0.0, 2.2, TonemapMode::Uncharted2, 4.0);
    let fitted = TonemapUniforms::full(width, height, 0.0, 2.2, TonemapMode::AcesFitted, 4.0);

    // Same dimensions for all
    assert_eq!(aces.src_dims, reinhard.src_dims);
    assert_eq!(reinhard.src_dims, uncharted.src_dims);
    assert_eq!(uncharted.src_dims, fitted.src_dims);

    // Different modes
    let modes: Vec<u32> = vec![aces.mode, reinhard.mode, uncharted.mode, fitted.mode];
    let unique_modes: std::collections::HashSet<_> = modes.iter().collect();
    assert_eq!(unique_modes.len(), 4, "All tonemap modes should be unique");
}
