//! Blackbox tests for T-DEMO-6.7 (Multi-Pass SDF) and T-DEMO-6.8 (Post-Processing Integration)
//!
//! These tests verify:
//! - Opaque SDF pass functionality
//! - Transparent SDF pass functionality
//! - Blend mode correctness
//! - Pass ordering
//! - Tone mapping integration
//! - Bloom extraction
//! - TAA compatibility

use renderer_backend::demoscene::multipass::{
    MultiPassUniforms, SdfBlendMode, SdfDepthMode, SdfPassConfig, SdfPassType,
    RenderPassOrder, MULTIPASS_OPAQUE_SHADER, MULTIPASS_TRANSPARENT_SHADER,
    MULTIPASS_WORKGROUP_SIZE, MAX_TRANSPARENT_OBJECTS, OPAQUE_ALPHA_THRESHOLD,
    TRANSPARENT_ALPHA_MIN,
};
use renderer_backend::demoscene::post_integration::{
    SdfPostProcessConfig, PostProcessUniforms, create_sdf_post_process_chain,
    create_sdf_bloom_bright_pass, create_bloom_blur_pass, create_sdf_motion_vector_pass,
    SDF_BLOOM_THRESHOLD, SDF_BLOOM_INTENSITY, SDF_TAA_FEEDBACK,
    TONEMAP_SHADER, BLOOM_BRIGHT_SHADER, BLOOM_BLUR_SHADER, TAA_SHADER,
};
use renderer_backend::frame_graph::{PassIndex, ResourceHandle, PassType};

// =============================================================================
// T-DEMO-6.7: OPAQUE PASS TESTS
// =============================================================================

#[test]
fn test_opaque_pass_config_depth_write_enabled() {
    let config = SdfPassConfig::opaque();
    assert!(config.depth_mode.writes_depth());
}

#[test]
fn test_opaque_pass_config_depth_test_enabled() {
    let config = SdfPassConfig::opaque();
    assert!(config.depth_mode.tests_depth());
}

#[test]
fn test_opaque_pass_config_no_blending() {
    let config = SdfPassConfig::opaque();
    assert_eq!(config.blend_mode, SdfBlendMode::Opaque);
}

#[test]
fn test_opaque_pass_config_alpha_threshold() {
    let config = SdfPassConfig::opaque();
    assert!(config.alpha_threshold > 0.9);
}

#[test]
fn test_opaque_pass_config_no_sorting() {
    let config = SdfPassConfig::opaque();
    assert!(!config.sort_transparent);
}

#[test]
fn test_opaque_uniforms_pass_type() {
    let uniforms = MultiPassUniforms::opaque(1920, 1080, 0.0);
    assert_eq!(uniforms.pass_type, 0);
}

#[test]
fn test_opaque_uniforms_alpha_threshold_set() {
    let uniforms = MultiPassUniforms::opaque(1920, 1080, 0.0);
    assert_eq!(uniforms.alpha_threshold, OPAQUE_ALPHA_THRESHOLD);
}

#[test]
fn test_opaque_shader_writes_depth() {
    assert!(MULTIPASS_OPAQUE_SHADER.contains("textureStore"));
    assert!(MULTIPASS_OPAQUE_SHADER.contains("depth_texture"));
}

#[test]
fn test_opaque_shader_discards_transparent() {
    assert!(MULTIPASS_OPAQUE_SHADER.contains("alpha_threshold"));
}

#[test]
fn test_opaque_shader_has_sdf_primitives() {
    assert!(MULTIPASS_OPAQUE_SHADER.contains("sdf_sphere"));
    assert!(MULTIPASS_OPAQUE_SHADER.contains("sdf_box"));
}

// =============================================================================
// T-DEMO-6.7: TRANSPARENT PASS TESTS
// =============================================================================

#[test]
fn test_transparent_pass_config_depth_read_only() {
    let config = SdfPassConfig::transparent();
    assert!(!config.depth_mode.writes_depth());
}

#[test]
fn test_transparent_pass_config_depth_test_enabled() {
    let config = SdfPassConfig::transparent();
    assert!(config.depth_mode.tests_depth());
}

#[test]
fn test_transparent_pass_config_alpha_blend() {
    let config = SdfPassConfig::transparent();
    assert_eq!(config.blend_mode, SdfBlendMode::AlphaBlend);
}

#[test]
fn test_transparent_pass_config_alpha_min() {
    let config = SdfPassConfig::transparent();
    assert!(config.alpha_min > 0.0);
    assert!(config.alpha_min < 0.1);
}

#[test]
fn test_transparent_pass_config_sorting_enabled() {
    let config = SdfPassConfig::transparent();
    assert!(config.sort_transparent);
}

#[test]
fn test_transparent_uniforms_pass_type() {
    let uniforms = MultiPassUniforms::transparent(1920, 1080, 0.0);
    assert_eq!(uniforms.pass_type, 1);
}

#[test]
fn test_transparent_uniforms_depth_bias() {
    let uniforms = MultiPassUniforms::transparent(1920, 1080, 0.0);
    assert!(uniforms.depth_bias > 0.0);
}

#[test]
fn test_transparent_shader_reads_depth() {
    assert!(MULTIPASS_TRANSPARENT_SHADER.contains("textureLoad"));
    assert!(MULTIPASS_TRANSPARENT_SHADER.contains("depth_texture"));
}

#[test]
fn test_transparent_shader_has_hdr_output() {
    assert!(MULTIPASS_TRANSPARENT_SHADER.contains("rgba16float"));
}

#[test]
fn test_transparent_shader_has_compositing() {
    assert!(MULTIPASS_TRANSPARENT_SHADER.contains("accumulated_color"));
    assert!(MULTIPASS_TRANSPARENT_SHADER.contains("accumulated_alpha"));
}

// =============================================================================
// T-DEMO-6.7: BLEND CORRECTNESS TESTS
// =============================================================================

#[test]
fn test_blend_mode_opaque_no_wgpu_state() {
    let mode = SdfBlendMode::Opaque;
    assert!(mode.to_wgpu_blend_state().is_none());
}

#[test]
fn test_blend_mode_alpha_src_factor() {
    let mode = SdfBlendMode::AlphaBlend;
    let state = mode.to_wgpu_blend_state().unwrap();
    assert_eq!(state.color.src_factor, wgpu::BlendFactor::SrcAlpha);
}

#[test]
fn test_blend_mode_alpha_dst_factor() {
    let mode = SdfBlendMode::AlphaBlend;
    let state = mode.to_wgpu_blend_state().unwrap();
    assert_eq!(state.color.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
}

#[test]
fn test_blend_mode_alpha_operation() {
    let mode = SdfBlendMode::AlphaBlend;
    let state = mode.to_wgpu_blend_state().unwrap();
    assert_eq!(state.color.operation, wgpu::BlendOperation::Add);
}

#[test]
fn test_blend_mode_additive_factors() {
    let mode = SdfBlendMode::Additive;
    let state = mode.to_wgpu_blend_state().unwrap();
    assert_eq!(state.color.src_factor, wgpu::BlendFactor::One);
    assert_eq!(state.color.dst_factor, wgpu::BlendFactor::One);
}

#[test]
fn test_blend_mode_premultiplied_src() {
    let mode = SdfBlendMode::Premultiplied;
    let state = mode.to_wgpu_blend_state().unwrap();
    assert_eq!(state.color.src_factor, wgpu::BlendFactor::One);
}

#[test]
fn test_blend_mode_premultiplied_dst() {
    let mode = SdfBlendMode::Premultiplied;
    let state = mode.to_wgpu_blend_state().unwrap();
    assert_eq!(state.color.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
}

// =============================================================================
// T-DEMO-6.7: PASS ORDERING TESTS
// =============================================================================

#[test]
fn test_pass_order_raster_opaque_first() {
    let order = RenderPassOrder::all_ordered();
    assert_eq!(order[0], RenderPassOrder::RasterOpaque);
}

#[test]
fn test_pass_order_sdf_opaque_second() {
    let order = RenderPassOrder::all_ordered();
    assert_eq!(order[1], RenderPassOrder::SdfOpaque);
}

#[test]
fn test_pass_order_sdf_transparent_third() {
    let order = RenderPassOrder::all_ordered();
    assert_eq!(order[2], RenderPassOrder::SdfTransparent);
}

#[test]
fn test_pass_order_raster_transparent_last() {
    let order = RenderPassOrder::all_ordered();
    assert_eq!(order[3], RenderPassOrder::RasterTransparent);
}

#[test]
fn test_pass_order_opaque_before_transparent() {
    assert!(RenderPassOrder::RasterOpaque < RenderPassOrder::SdfTransparent);
    assert!(RenderPassOrder::SdfOpaque < RenderPassOrder::SdfTransparent);
    assert!(RenderPassOrder::SdfOpaque < RenderPassOrder::RasterTransparent);
}

#[test]
fn test_pass_order_opaque_writes_depth() {
    assert!(RenderPassOrder::RasterOpaque.writes_depth());
    assert!(RenderPassOrder::SdfOpaque.writes_depth());
}

#[test]
fn test_pass_order_transparent_no_depth_write() {
    assert!(!RenderPassOrder::SdfTransparent.writes_depth());
    assert!(!RenderPassOrder::RasterTransparent.writes_depth());
}

#[test]
fn test_pass_order_sdf_identification() {
    assert!(RenderPassOrder::SdfOpaque.is_sdf());
    assert!(RenderPassOrder::SdfTransparent.is_sdf());
    assert!(!RenderPassOrder::RasterOpaque.is_sdf());
    assert!(!RenderPassOrder::RasterTransparent.is_sdf());
}

#[test]
fn test_pass_order_transparent_identification() {
    assert!(RenderPassOrder::SdfTransparent.is_transparent());
    assert!(RenderPassOrder::RasterTransparent.is_transparent());
    assert!(!RenderPassOrder::RasterOpaque.is_transparent());
    assert!(!RenderPassOrder::SdfOpaque.is_transparent());
}

// =============================================================================
// T-DEMO-6.8: TONE MAPPING INTEGRATION TESTS
// =============================================================================

#[test]
fn test_tonemap_shader_has_aces() {
    assert!(TONEMAP_SHADER.contains("aces_tonemap"));
}

#[test]
fn test_tonemap_shader_has_exposure() {
    assert!(TONEMAP_SHADER.contains("exposure"));
}

#[test]
fn test_tonemap_shader_compute_entry() {
    assert!(TONEMAP_SHADER.contains("@compute"));
    assert!(TONEMAP_SHADER.contains("@workgroup_size(8, 8, 1)"));
}

#[test]
fn test_tonemap_shader_reads_hdr() {
    assert!(TONEMAP_SHADER.contains("hdr_input"));
}

#[test]
fn test_tonemap_shader_writes_output() {
    assert!(TONEMAP_SHADER.contains("textureStore"));
}

#[test]
fn test_tonemap_config_enabled_by_default() {
    let config = SdfPostProcessConfig::default();
    assert!(config.enable_tonemap);
}

// =============================================================================
// T-DEMO-6.8: BLOOM EXTRACTION TESTS
// =============================================================================

#[test]
fn test_bloom_bright_shader_has_threshold() {
    assert!(BLOOM_BRIGHT_SHADER.contains("bloom_threshold"));
}

#[test]
fn test_bloom_bright_shader_has_luminance() {
    assert!(BLOOM_BRIGHT_SHADER.contains("luminance"));
}

#[test]
fn test_bloom_bright_shader_has_intensity() {
    assert!(BLOOM_BRIGHT_SHADER.contains("bloom_intensity"));
}

#[test]
fn test_bloom_blur_shader_has_kernel() {
    assert!(BLOOM_BLUR_SHADER.contains("KERNEL_WEIGHTS"));
}

#[test]
fn test_bloom_blur_shader_separable() {
    assert!(BLOOM_BLUR_SHADER.contains("is_horizontal"));
}

#[test]
fn test_bloom_config_enabled_by_default() {
    let config = SdfPostProcessConfig::default();
    assert!(config.enable_bloom);
}

#[test]
fn test_bloom_threshold_constant() {
    assert!(SDF_BLOOM_THRESHOLD >= 0.0);
    assert!(SDF_BLOOM_THRESHOLD <= 5.0);
}

#[test]
fn test_bloom_intensity_constant() {
    assert!(SDF_BLOOM_INTENSITY >= 0.0);
    assert!(SDF_BLOOM_INTENSITY <= 2.0);
}

#[test]
fn test_bloom_bright_pass_tags() {
    let pass = create_sdf_bloom_bright_pass(
        PassIndex(0),
        ResourceHandle(1),
        ResourceHandle(2),
        1920,
        1080,
    );
    assert!(pass.tags.contains(&"bloom".into()));
    assert!(pass.tags.contains(&"sdf-integration".into()));
}

#[test]
fn test_bloom_blur_pass_horizontal_name() {
    let pass = create_bloom_blur_pass(
        PassIndex(0),
        ResourceHandle(1),
        ResourceHandle(2),
        1920,
        1080,
        true,
    );
    assert_eq!(pass.name, "bloom_blur_h");
}

#[test]
fn test_bloom_blur_pass_vertical_name() {
    let pass = create_bloom_blur_pass(
        PassIndex(0),
        ResourceHandle(1),
        ResourceHandle(2),
        1920,
        1080,
        false,
    );
    assert_eq!(pass.name, "bloom_blur_v");
}

// =============================================================================
// T-DEMO-6.8: TAA COMPATIBILITY TESTS
// =============================================================================

#[test]
fn test_taa_shader_has_feedback() {
    assert!(TAA_SHADER.contains("taa_feedback"));
}

#[test]
fn test_taa_shader_has_neighborhood_sampling() {
    assert!(TAA_SHADER.contains("sample_coords") || TAA_SHADER.contains("neighborhood"));
}

#[test]
fn test_taa_config_enabled_by_default() {
    let config = SdfPostProcessConfig::default();
    assert!(config.enable_taa);
}

#[test]
fn test_taa_feedback_constant() {
    assert!(SDF_TAA_FEEDBACK >= 0.0);
    assert!(SDF_TAA_FEEDBACK <= 1.0);
}

#[test]
fn test_taa_config_with_custom_feedback() {
    let config = SdfPostProcessConfig::default().with_taa(0.85);
    assert_eq!(config.taa_feedback, 0.85);
}

// =============================================================================
// POST-PROCESS CHAIN TESTS
// =============================================================================

#[test]
fn test_chain_minimal_single_pass() {
    let config = SdfPostProcessConfig::minimal(1920, 1080);
    let (passes, _) = create_sdf_post_process_chain(
        PassIndex(0),
        ResourceHandle(1),
        ResourceHandle(2),
        &config,
    );
    assert_eq!(passes.len(), 1);
    assert_eq!(passes[0].name, "tonemap");
}

#[test]
fn test_chain_full_has_bloom() {
    let config = SdfPostProcessConfig::full(1920, 1080);
    let (passes, _) = create_sdf_post_process_chain(
        PassIndex(0),
        ResourceHandle(1),
        ResourceHandle(2),
        &config,
    );
    let names: Vec<_> = passes.iter().map(|p| p.name.as_str()).collect();
    assert!(names.contains(&"sdf_bloom_bright"));
}

#[test]
fn test_chain_full_has_tonemap() {
    let config = SdfPostProcessConfig::full(1920, 1080);
    let (passes, _) = create_sdf_post_process_chain(
        PassIndex(0),
        ResourceHandle(1),
        ResourceHandle(2),
        &config,
    );
    let names: Vec<_> = passes.iter().map(|p| p.name.as_str()).collect();
    assert!(names.contains(&"tonemap"));
}

#[test]
fn test_chain_full_has_taa() {
    let config = SdfPostProcessConfig::full(1920, 1080);
    let (passes, _) = create_sdf_post_process_chain(
        PassIndex(0),
        ResourceHandle(1),
        ResourceHandle(2),
        &config,
    );
    let names: Vec<_> = passes.iter().map(|p| p.name.as_str()).collect();
    assert!(names.contains(&"taa"));
}

#[test]
fn test_chain_consecutive_indices() {
    let config = SdfPostProcessConfig::full(1920, 1080);
    let (passes, _) = create_sdf_post_process_chain(
        PassIndex(5),
        ResourceHandle(1),
        ResourceHandle(2),
        &config,
    );
    for (i, pass) in passes.iter().enumerate() {
        assert_eq!(pass.index.0, 5 + i);
    }
}

#[test]
fn test_chain_resources_reserved_range() {
    let config = SdfPostProcessConfig::full(1920, 1080);
    let (_, resources) = create_sdf_post_process_chain(
        PassIndex(0),
        ResourceHandle(1),
        ResourceHandle(2),
        &config,
    );
    for r in &resources {
        assert!(r.handle.0 >= 0xFE00);
    }
}

#[test]
fn test_chain_all_passes_are_compute() {
    let config = SdfPostProcessConfig::full(1920, 1080);
    let (passes, _) = create_sdf_post_process_chain(
        PassIndex(0),
        ResourceHandle(1),
        ResourceHandle(2),
        &config,
    );
    for pass in &passes {
        assert_eq!(pass.pass_type, PassType::Compute);
    }
}

// =============================================================================
// UNIFORM DATA TESTS
// =============================================================================

#[test]
fn test_multipass_uniforms_size() {
    assert_eq!(std::mem::size_of::<MultiPassUniforms>(), 32);
}

#[test]
fn test_postprocess_uniforms_size() {
    assert_eq!(std::mem::size_of::<PostProcessUniforms>(), 32);
}

#[test]
fn test_multipass_uniforms_alignment() {
    assert_eq!(std::mem::align_of::<MultiPassUniforms>(), 4);
}

#[test]
fn test_postprocess_uniforms_from_config() {
    let config = SdfPostProcessConfig::full(800, 600).with_bloom(0.5, 0.8);
    let uniforms = PostProcessUniforms::from_config(&config, 1.0);
    assert_eq!(uniforms.resolution, [800.0, 600.0]);
    assert_eq!(uniforms.bloom_threshold, 0.5);
    assert_eq!(uniforms.bloom_intensity, 0.8);
}

// =============================================================================
// CONFIG VALIDATION TESTS
// =============================================================================

#[test]
fn test_config_validate_ok() {
    let config = SdfPostProcessConfig::default();
    assert!(config.validate().is_ok());
}

#[test]
fn test_config_validate_zero_width() {
    let mut config = SdfPostProcessConfig::default();
    config.width = 0;
    assert!(config.validate().is_err());
}

#[test]
fn test_config_validate_zero_height() {
    let mut config = SdfPostProcessConfig::default();
    config.height = 0;
    assert!(config.validate().is_err());
}

#[test]
fn test_config_validate_negative_bloom() {
    let mut config = SdfPostProcessConfig::default();
    config.bloom_threshold = -1.0;
    assert!(config.validate().is_err());
}

#[test]
fn test_config_validate_invalid_taa() {
    let mut config = SdfPostProcessConfig::default();
    config.taa_feedback = 1.5;
    assert!(config.validate().is_err());
}

// =============================================================================
// SHADER WGSL PARSE TESTS
// =============================================================================

#[test]
fn test_opaque_shader_parses_naga() {
    use naga::front::wgsl;
    let result = wgsl::parse_str(MULTIPASS_OPAQUE_SHADER);
    assert!(result.is_ok(), "Opaque shader error: {:?}", result.err());
}

#[test]
fn test_transparent_shader_parses_naga() {
    use naga::front::wgsl;
    let result = wgsl::parse_str(MULTIPASS_TRANSPARENT_SHADER);
    assert!(result.is_ok(), "Transparent shader error: {:?}", result.err());
}

#[test]
fn test_tonemap_shader_parses_naga() {
    use naga::front::wgsl;
    let result = wgsl::parse_str(TONEMAP_SHADER);
    assert!(result.is_ok(), "Tonemap shader error: {:?}", result.err());
}

#[test]
fn test_bloom_bright_shader_parses_naga() {
    use naga::front::wgsl;
    let result = wgsl::parse_str(BLOOM_BRIGHT_SHADER);
    assert!(result.is_ok(), "Bloom bright shader error: {:?}", result.err());
}

#[test]
fn test_bloom_blur_shader_parses_naga() {
    use naga::front::wgsl;
    let result = wgsl::parse_str(BLOOM_BLUR_SHADER);
    assert!(result.is_ok(), "Bloom blur shader error: {:?}", result.err());
}

#[test]
fn test_taa_shader_parses_naga() {
    use naga::front::wgsl;
    let result = wgsl::parse_str(TAA_SHADER);
    assert!(result.is_ok(), "TAA shader error: {:?}", result.err());
}

// =============================================================================
// WORKGROUP SIZE TESTS
// =============================================================================

#[test]
fn test_workgroup_size_matches_shader() {
    assert_eq!(MULTIPASS_WORKGROUP_SIZE, 8);
    assert!(MULTIPASS_OPAQUE_SHADER.contains("@workgroup_size(8, 8, 1)"));
    assert!(MULTIPASS_TRANSPARENT_SHADER.contains("@workgroup_size(8, 8, 1)"));
}

// =============================================================================
// CONSTANTS TESTS
// =============================================================================

#[test]
fn test_max_transparent_objects() {
    assert!(MAX_TRANSPARENT_OBJECTS >= 64);
}

#[test]
fn test_alpha_threshold_valid_range() {
    assert!(OPAQUE_ALPHA_THRESHOLD > 0.0);
    assert!(OPAQUE_ALPHA_THRESHOLD <= 1.0);
}

#[test]
fn test_alpha_min_valid_range() {
    assert!(TRANSPARENT_ALPHA_MIN > 0.0);
    assert!(TRANSPARENT_ALPHA_MIN < 1.0);
}

// =============================================================================
// MOTION VECTOR TESTS
// =============================================================================

#[test]
fn test_motion_vector_pass_creation() {
    let pass = create_sdf_motion_vector_pass(
        PassIndex(0),
        ResourceHandle(1),
        ResourceHandle(2),
        1920,
        1080,
    );
    assert_eq!(pass.name, "sdf_motion_vectors");
    assert!(pass.tags.contains(&"motion-vectors".into()));
}

#[test]
fn test_motion_vector_pass_compute_type() {
    let pass = create_sdf_motion_vector_pass(
        PassIndex(0),
        ResourceHandle(1),
        ResourceHandle(2),
        1920,
        1080,
    );
    assert_eq!(pass.pass_type, PassType::Compute);
}

#[test]
fn test_motion_vector_config_option() {
    let config = SdfPostProcessConfig::full(1920, 1080);
    assert!(config.enable_motion_vectors);
}
