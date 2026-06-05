//! T-DEMO-7.7 Integration Tests for Demoscene Renderer
//!
//! Comprehensive integration tests covering:
//! 1. S13 writes to main framebuffer
//! 2. Hybrid compositing (SDF + raster)
//! 3. Depth consistency across pipelines
//! 4. Post-processing integration (tone map, bloom, TAA)
//! 5. Full-screen mode
//! 6. Frame graph integration
//! 7. Resource transitions/barriers
//! 8. Multi-pass rendering
//! 9. Python-to-Rust pipeline
//! 10. End-to-end integration
//!
//! Run: cargo test demoscene_integration

use renderer_backend::demoscene::{
    // Hybrid depth compositing
    HybridDepthConfig, HybridDepthRenderer, HybridUniforms, DepthBufferBinding,
    DepthCompareResult, HYBRID_DEPTH_SHADER, DEPTH_BUFFER_FORMAT,
    DEFAULT_NEAR_PLANE, DEFAULT_FAR_PLANE, MAX_RAY_MARCH_DIST, DEPTH_EPSILON,
    ndc_to_linear, linear_to_ndc, reverse_z_to_linear, linear_to_reverse_z,
    // Multi-pass
    MultiPassSdfRenderer, MultiPassUniforms, SdfBlendMode, SdfDepthMode,
    SdfPassConfig, SdfPassType, RenderPassOrder,
    MULTIPASS_OPAQUE_SHADER, MULTIPASS_TRANSPARENT_SHADER,
    MULTIPASS_WORKGROUP_SIZE, MAX_TRANSPARENT_OBJECTS,
    OPAQUE_ALPHA_THRESHOLD, TRANSPARENT_ALPHA_MIN,
    // Post-processing integration
    SdfPostProcessConfig, SdfPostProcessor, PostProcessUniforms,
    create_sdf_post_process_chain, create_sdf_bloom_bright_pass,
    create_bloom_blur_pass, create_sdf_motion_vector_pass,
    SDF_BLOOM_THRESHOLD, SDF_BLOOM_INTENSITY, SDF_TAA_FEEDBACK, SDF_MOTION_SCALE,
    TONEMAP_SHADER, BLOOM_BRIGHT_SHADER, BLOOM_BLUR_SHADER, TAA_SHADER,
    // Minimal renderer
    MinimalRenderer, MinimalUniforms, MINIMAL_SHADER,
    // Barriers
    DepthProjection, DemoResourceState, DemoResourceTransition,
    DemoBarrierScheduler, DemoFrameBarriers, resource_ids,
    // Shader validation
    validate_demoscene_shader, get_demo_scene_shader,
    // Static shaders
    DEMO_SCENE_STATIC, NOISE_HASH, NOISE_VALUE, NOISE_PERLIN, NOISE_FBM,
    SDF_DOMAIN, SDF_PRIMITIVES, SDF_COMBINATORS,
};
use renderer_backend::rhi_device::RhiDevice;
use renderer_backend::frame_graph::{PassIndex, ResourceHandle, PassType};

// =============================================================================
// 1. S13 Framebuffer Output Tests
// =============================================================================

mod s13_framebuffer_tests {
    use super::*;

    #[test]
    fn test_framebuffer_output_texture_creation() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let renderer = HybridDepthRenderer::new(&device.device, 1920, 1080);
            let texture = renderer.output_texture();

            assert_eq!(texture.width(), 1920);
            assert_eq!(texture.height(), 1080);
        }
    }

    #[test]
    fn test_framebuffer_output_view_accessible() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let renderer = HybridDepthRenderer::new(&device.device, 800, 600);
            // Should not panic
            let _view = renderer.output_view();
        }
    }

    #[test]
    fn test_framebuffer_write_shader_present() {
        // Verify shader has textureStore for output
        assert!(HYBRID_DEPTH_SHADER.contains("textureStore") ||
                HYBRID_DEPTH_SHADER.contains("output_texture"));
    }

    #[test]
    fn test_framebuffer_dimensions_match_config() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let config = HybridDepthConfig::new(3840, 2160);
            let renderer = HybridDepthRenderer::with_config(&device.device, config);

            assert_eq!(renderer.size(), (3840, 2160));
        }
    }

    #[test]
    fn test_framebuffer_resize_works() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let mut renderer = HybridDepthRenderer::new(&device.device, 800, 600);
            renderer.resize(&device.device, 1920, 1080);

            assert_eq!(renderer.size(), (1920, 1080));
        }
    }

    #[test]
    fn test_framebuffer_dispatch_produces_output() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let mut renderer = HybridDepthRenderer::new(&device.device, 256, 256);
            renderer.update_and_upload(&device.queue, 0.0);

            let mut encoder = device.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("Test") }
            );
            renderer.dispatch(&mut encoder);
            device.queue.submit(std::iter::once(encoder.finish()));
            device.wait_idle();
            // No panic = success
        }
    }

    #[test]
    fn test_minimal_renderer_framebuffer_output() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let renderer = MinimalRenderer::new(&device.device, 800, 600);
            let texture = renderer.output_texture();

            assert_eq!(texture.width(), 800);
            assert_eq!(texture.height(), 600);
        }
    }

    #[test]
    fn test_multipass_framebuffer_integration() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let renderer = MultiPassSdfRenderer::new(&device.device, 1024, 768);

            assert_eq!(renderer.size(), (1024, 768));
        }
    }
}

// =============================================================================
// 2. Hybrid Compositing Tests
// =============================================================================

mod hybrid_compositing_tests {
    use super::*;

    #[test]
    fn test_hybrid_config_creation() {
        let config = HybridDepthConfig::new(1920, 1080)
            .with_depth_planes(0.1, 500.0)
            .with_reverse_z(true);

        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.near_plane, 0.1);
        assert_eq!(config.far_plane, 500.0);
        assert!(config.reverse_z);
    }

    #[test]
    fn test_hybrid_depth_binding() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let mut renderer = HybridDepthRenderer::new(&device.device, 800, 600);

            let depth_texture = device.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Test Depth"),
                size: wgpu::Extent3d { width: 800, height: 600, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH_BUFFER_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

            renderer.bind_depth_buffer(&device.device, &depth_view, 800, 600, false);
            assert!(renderer.has_depth_buffer());
        }
    }

    #[test]
    fn test_hybrid_sdf_raster_comparison() {
        // SDF at 5.0, raster at 10.0 -> SDF closer
        let result = DepthCompareResult::compare(5.0, 10.0, DEPTH_EPSILON);
        assert_eq!(result, DepthCompareResult::RayMarchCloser);

        // SDF at 10.0, raster at 5.0 -> raster closer
        let result = DepthCompareResult::compare(10.0, 5.0, DEPTH_EPSILON);
        assert_eq!(result, DepthCompareResult::RasterCloser);
    }

    #[test]
    fn test_hybrid_depth_shader_has_compositing() {
        assert!(HYBRID_DEPTH_SHADER.contains("sample_depth") ||
                HYBRID_DEPTH_SHADER.contains("depth_texture"));
        assert!(HYBRID_DEPTH_SHADER.contains("ray_march"));
    }

    #[test]
    fn test_hybrid_blend_preserve_raster() {
        let result = DepthCompareResult::RasterCloser;
        assert!(result.should_preserve_raster());
        assert!(!result.should_write_ray_march());
    }

    #[test]
    fn test_hybrid_blend_write_sdf() {
        let result = DepthCompareResult::RayMarchCloser;
        assert!(result.should_write_ray_march());
        assert!(!result.should_preserve_raster());
    }

    #[test]
    fn test_hybrid_dispatch_with_depth() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let mut renderer = HybridDepthRenderer::new(&device.device, 256, 256);

            let depth_texture = device.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Test Depth"),
                size: wgpu::Extent3d { width: 256, height: 256, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH_BUFFER_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

            renderer.bind_depth_buffer(&device.device, &depth_view, 256, 256, false);
            renderer.update_and_upload(&device.queue, 1.0);

            let mut encoder = device.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("Test") }
            );
            renderer.dispatch(&mut encoder);
            device.queue.submit(std::iter::once(encoder.finish()));
            device.wait_idle();
        }
    }
}

// =============================================================================
// 3. Depth Consistency Tests
// =============================================================================

mod depth_consistency_tests {
    use super::*;

    #[test]
    fn test_ndc_linear_conversion_near() {
        let linear = ndc_to_linear(0.0, 0.1, 100.0);
        assert!((linear - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_ndc_linear_conversion_far() {
        let linear = ndc_to_linear(1.0, 0.1, 100.0);
        assert!((linear - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_linear_ndc_roundtrip() {
        let near = 0.1;
        let far = 100.0;

        for ndc in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let linear = ndc_to_linear(ndc, near, far);
            let back = linear_to_ndc(linear, near, far);
            assert!((ndc - back).abs() < 0.001, "Roundtrip failed for ndc={}", ndc);
        }
    }

    #[test]
    fn test_reverse_z_consistency() {
        let near = 0.1;
        let far = 100.0;

        // Reverse-Z: near plane is 1.0, far is 0.0
        let at_near = reverse_z_to_linear(1.0, near, far);
        let at_far = reverse_z_to_linear(0.0, near, far);

        assert!((at_near - near).abs() < 0.001);
        assert!((at_far - far).abs() < 0.001);
    }

    #[test]
    fn test_reverse_z_roundtrip() {
        let near = 0.5;
        let far = 50.0;

        for rz in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let linear = reverse_z_to_linear(rz, near, far);
            let back = linear_to_reverse_z(linear, near, far);
            assert!((rz - back).abs() < 0.001);
        }
    }

    #[test]
    fn test_depth_format_is_float32() {
        assert_eq!(DEPTH_BUFFER_FORMAT, wgpu::TextureFormat::Depth32Float);
    }

    #[test]
    fn test_depth_epsilon_reasonable() {
        assert!(DEPTH_EPSILON > 0.0);
        assert!(DEPTH_EPSILON < 0.1);
    }

    #[test]
    fn test_max_ray_march_distance() {
        assert!(MAX_RAY_MARCH_DIST > 0.0);
        assert!(MAX_RAY_MARCH_DIST >= 10.0);
    }
}

// =============================================================================
// 4. Post-Processing Integration Tests
// =============================================================================

mod post_processing_tests {
    use super::*;

    #[test]
    fn test_tonemap_shader_has_aces() {
        assert!(TONEMAP_SHADER.contains("aces") || TONEMAP_SHADER.contains("ACES"));
    }

    #[test]
    fn test_tonemap_shader_is_compute() {
        assert!(TONEMAP_SHADER.contains("@compute"));
        assert!(TONEMAP_SHADER.contains("@workgroup_size"));
    }

    #[test]
    fn test_bloom_bright_pass_shader() {
        assert!(BLOOM_BRIGHT_SHADER.contains("threshold") ||
                BLOOM_BRIGHT_SHADER.contains("bright"));
    }

    #[test]
    fn test_bloom_blur_shader_separable() {
        // Bloom blur should be separable (horizontal/vertical)
        assert!(BLOOM_BLUR_SHADER.contains("blur") ||
                BLOOM_BLUR_SHADER.contains("gaussian") ||
                BLOOM_BLUR_SHADER.contains("weight"));
    }

    #[test]
    fn test_taa_shader_has_history() {
        assert!(TAA_SHADER.contains("history") ||
                TAA_SHADER.contains("feedback") ||
                TAA_SHADER.contains("temporal"));
    }

    #[test]
    fn test_post_process_config_default() {
        let config = SdfPostProcessConfig::default();

        assert!(config.enable_tonemap);
        assert!(config.enable_bloom);
        assert!(config.enable_taa);
    }

    #[test]
    fn test_post_process_config_minimal() {
        let config = SdfPostProcessConfig::minimal(1920, 1080);

        assert!(config.enable_tonemap);
        assert!(!config.enable_bloom);
        assert!(!config.enable_taa);
    }

    #[test]
    fn test_post_process_config_full() {
        let config = SdfPostProcessConfig::full(1920, 1080);

        assert!(config.enable_tonemap);
        assert!(config.enable_bloom);
        assert!(config.enable_taa);
        assert!(config.enable_motion_vectors);
    }

    #[test]
    fn test_post_process_uniforms_from_config() {
        let config = SdfPostProcessConfig::full(1920, 1080);
        let uniforms = PostProcessUniforms::from_config(&config, 0.5);

        assert_eq!(uniforms.resolution[0], 1920.0);
        assert_eq!(uniforms.resolution[1], 1080.0);
        assert_eq!(uniforms.time, 0.5);
    }

    #[test]
    fn test_bloom_constants() {
        assert!(SDF_BLOOM_THRESHOLD > 0.0);
        assert!(SDF_BLOOM_INTENSITY > 0.0);
        assert!(SDF_BLOOM_INTENSITY <= 2.0);
    }

    #[test]
    fn test_taa_feedback_in_range() {
        assert!(SDF_TAA_FEEDBACK >= 0.0);
        assert!(SDF_TAA_FEEDBACK <= 1.0);
    }
}

// =============================================================================
// 5. Full-Screen Mode Tests
// =============================================================================

mod full_screen_tests {
    use super::*;

    #[test]
    fn test_minimal_shader_is_compute() {
        assert!(MINIMAL_SHADER.contains("@compute"));
        assert!(!MINIMAL_SHADER.contains("@vertex"));
        assert!(!MINIMAL_SHADER.contains("@fragment"));
    }

    #[test]
    fn test_minimal_shader_has_workgroup_size() {
        assert!(MINIMAL_SHADER.contains("@workgroup_size(8, 8"));
    }

    #[test]
    fn test_minimal_shader_has_output() {
        assert!(MINIMAL_SHADER.contains("textureStore") ||
                MINIMAL_SHADER.contains("output"));
    }

    #[test]
    fn test_minimal_renderer_dispatch_size() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let renderer = MinimalRenderer::new(&device.device, 800, 600);
            let (dx, dy, dz) = renderer.dispatch_size();

            // 800 / 8 = 100, 600 / 8 = 75
            assert_eq!(dx, 100);
            assert_eq!(dy, 75);
            assert_eq!(dz, 1);
        }
    }

    #[test]
    fn test_minimal_renderer_dispatch_size_unaligned() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let renderer = MinimalRenderer::new(&device.device, 801, 601);
            let (dx, dy, dz) = renderer.dispatch_size();

            // (801 + 7) / 8 = 101, (601 + 7) / 8 = 76
            assert_eq!(dx, 101);
            assert_eq!(dy, 76);
            assert_eq!(dz, 1);
        }
    }

    #[test]
    fn test_minimal_uniforms_default() {
        let uniforms = MinimalUniforms::default();

        assert_eq!(uniforms.time, 0.0);
        assert_eq!(uniforms.resolution_x, 800.0);
        assert_eq!(uniforms.resolution_y, 600.0);
    }

    #[test]
    fn test_minimal_uniforms_new() {
        let uniforms = MinimalUniforms::new(1920, 1080);

        assert_eq!(uniforms.resolution_x, 1920.0);
        assert_eq!(uniforms.resolution_y, 1080.0);
    }

    #[test]
    fn test_minimal_render_full_frame() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let renderer = MinimalRenderer::new(&device.device, 320, 240);
            renderer.render(&device.device, &device.queue, 0.0);
            device.wait_idle();
        }
    }
}

// =============================================================================
// 6. Frame Graph Integration Tests
// =============================================================================

mod frame_graph_tests {
    use super::*;

    #[test]
    fn test_demo_shader_validation_passes() {
        let shader = get_demo_scene_shader();
        let result = validate_demoscene_shader(shader);

        assert!(result.is_ok(), "Shader validation failed: {:?}", result.err());
    }

    #[test]
    fn test_demo_shader_has_scene_sdf() {
        let shader = get_demo_scene_shader();
        assert!(shader.contains("scene_sdf"));
    }

    #[test]
    fn test_demo_shader_has_scene_material() {
        let shader = get_demo_scene_shader();
        assert!(shader.contains("scene_material"));
    }

    #[test]
    fn test_hybrid_config_dispatch_size() {
        let config = HybridDepthConfig::new(800, 600);
        let (x, y, z) = config.dispatch_size();

        assert_eq!(x, 100); // 800 / 8
        assert_eq!(y, 75);  // 600 / 8
        assert_eq!(z, 1);
    }

    #[test]
    fn test_multipass_pass_indices() {
        let order = RenderPassOrder::all_ordered();

        // Verify ordering is correct
        assert!(order.iter().position(|&x| x == RenderPassOrder::RasterOpaque) <
                order.iter().position(|&x| x == RenderPassOrder::SdfOpaque));
        assert!(order.iter().position(|&x| x == RenderPassOrder::SdfOpaque) <
                order.iter().position(|&x| x == RenderPassOrder::SdfTransparent));
    }

    #[test]
    fn test_resource_handle_validity() {
        // ResourceHandle should be valid non-zero
        let handle = ResourceHandle::from_raw(1);
        assert!(!handle.is_null());
    }

    #[test]
    fn test_post_process_chain_creation() {
        let config = SdfPostProcessConfig::full(1920, 1080);
        let result = config.validate();

        assert!(result.is_ok());
    }
}

// =============================================================================
// 7. Resource Transition / Barrier Tests
// =============================================================================

mod resource_transition_tests {
    use super::*;

    #[test]
    fn test_resource_state_output() {
        let state = DemoResourceState::Output;
        assert!(state.is_writable());
    }

    #[test]
    fn test_resource_state_sampled() {
        let state = DemoResourceState::Sampled;
        assert!(state.is_readable());
    }

    #[test]
    fn test_resource_transition_creation() {
        let transition = DemoResourceTransition::new(
            resource_ids::COLOR_OUTPUT,
            DemoResourceState::Undefined,
            DemoResourceState::Output,
        );

        assert_eq!(transition.resource, resource_ids::COLOR_OUTPUT);
        assert_eq!(transition.from, DemoResourceState::Undefined);
        assert_eq!(transition.to, DemoResourceState::Output);
    }

    #[test]
    fn test_barrier_scheduler_creation() {
        let scheduler = DemoBarrierScheduler::new();
        assert!(scheduler.pending_transitions().is_empty());
    }

    #[test]
    fn test_barrier_scheduler_add_transition() {
        let mut scheduler = DemoBarrierScheduler::new();

        scheduler.transition(
            resource_ids::DEPTH_BUFFER,
            DemoResourceState::Undefined,
            DemoResourceState::DepthWrite,
        );

        assert!(!scheduler.pending_transitions().is_empty());
    }

    #[test]
    fn test_frame_barriers_structure() {
        let barriers = DemoFrameBarriers::default();

        // Should have slots for each pass type
        assert!(barriers.pre_sdf_opaque.is_empty());
        assert!(barriers.post_sdf_opaque.is_empty());
    }

    #[test]
    fn test_depth_projection_ndc() {
        let proj = DepthProjection::NDC;
        assert_eq!(proj.near_value(), 0.0);
        assert_eq!(proj.far_value(), 1.0);
    }

    #[test]
    fn test_depth_projection_reverse_z() {
        let proj = DepthProjection::ReverseZ;
        assert_eq!(proj.near_value(), 1.0);
        assert_eq!(proj.far_value(), 0.0);
    }
}

// =============================================================================
// 8. Multi-Pass Rendering Tests
// =============================================================================

mod multipass_tests {
    use super::*;

    #[test]
    fn test_opaque_pass_config() {
        let config = SdfPassConfig::opaque();

        assert!(config.depth_mode.writes_depth());
        assert!(config.depth_mode.tests_depth());
        assert_eq!(config.blend_mode, SdfBlendMode::Opaque);
    }

    #[test]
    fn test_transparent_pass_config() {
        let config = SdfPassConfig::transparent();

        assert!(!config.depth_mode.writes_depth());
        assert!(config.depth_mode.tests_depth());
        assert_eq!(config.blend_mode, SdfBlendMode::AlphaBlend);
    }

    #[test]
    fn test_blend_mode_alpha() {
        let blend = SdfBlendMode::AlphaBlend.to_wgpu_blend_state().unwrap();

        assert_eq!(blend.color.src_factor, wgpu::BlendFactor::SrcAlpha);
        assert_eq!(blend.color.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
    }

    #[test]
    fn test_blend_mode_additive() {
        let blend = SdfBlendMode::Additive.to_wgpu_blend_state().unwrap();

        assert_eq!(blend.color.src_factor, wgpu::BlendFactor::One);
        assert_eq!(blend.color.dst_factor, wgpu::BlendFactor::One);
    }

    #[test]
    fn test_pass_order_sdf_identification() {
        assert!(RenderPassOrder::SdfOpaque.is_sdf());
        assert!(RenderPassOrder::SdfTransparent.is_sdf());
        assert!(!RenderPassOrder::RasterOpaque.is_sdf());
    }

    #[test]
    fn test_pass_order_transparent_identification() {
        assert!(RenderPassOrder::SdfTransparent.is_transparent());
        assert!(RenderPassOrder::RasterTransparent.is_transparent());
        assert!(!RenderPassOrder::SdfOpaque.is_transparent());
    }

    #[test]
    fn test_multipass_uniforms_opaque() {
        let uniforms = MultiPassUniforms::opaque(1920, 1080, 0.5);

        assert_eq!(uniforms.pass_type, 0);
        assert_eq!(uniforms.resolution_x, 1920.0);
        assert_eq!(uniforms.resolution_y, 1080.0);
        assert_eq!(uniforms.time, 0.5);
    }

    #[test]
    fn test_multipass_uniforms_transparent() {
        let uniforms = MultiPassUniforms::transparent(1920, 1080, 0.5);

        assert_eq!(uniforms.pass_type, 1);
        assert!(uniforms.depth_bias > 0.0);
    }

    #[test]
    fn test_opaque_shader_has_depth_write() {
        assert!(MULTIPASS_OPAQUE_SHADER.contains("depth"));
    }

    #[test]
    fn test_transparent_shader_has_alpha_blend() {
        assert!(MULTIPASS_TRANSPARENT_SHADER.contains("alpha") ||
                MULTIPASS_TRANSPARENT_SHADER.contains("accumulated"));
    }
}

// =============================================================================
// 9. Python-to-Rust Pipeline Tests
// =============================================================================

mod python_pipeline_tests {
    use super::*;

    #[test]
    fn test_demo_shader_syntax_validation() {
        let shader = get_demo_scene_shader();

        // Check for balanced braces
        let open = shader.matches('{').count();
        let close = shader.matches('}').count();
        assert_eq!(open, close, "Unbalanced braces");

        // Check for balanced parentheses
        let open = shader.matches('(').count();
        let close = shader.matches(')').count();
        assert_eq!(open, close, "Unbalanced parentheses");
    }

    #[test]
    fn test_static_shader_non_empty() {
        assert!(!DEMO_SCENE_STATIC.is_empty());
    }

    #[test]
    fn test_sdf_primitives_shader() {
        assert!(SDF_PRIMITIVES.contains("sdf_sphere") ||
                SDF_PRIMITIVES.contains("sdSphere"));
        assert!(SDF_PRIMITIVES.contains("sdf_box") ||
                SDF_PRIMITIVES.contains("sdBox"));
    }

    #[test]
    fn test_sdf_combinators_shader() {
        assert!(SDF_COMBINATORS.contains("sdf_union") ||
                SDF_COMBINATORS.contains("opUnion") ||
                SDF_COMBINATORS.contains("min("));
    }

    #[test]
    fn test_sdf_domain_shader() {
        assert!(SDF_DOMAIN.contains("repeat") ||
                SDF_DOMAIN.contains("mirror") ||
                SDF_DOMAIN.contains("twist") ||
                SDF_DOMAIN.contains("mod("));
    }

    #[test]
    fn test_noise_shaders_present() {
        assert!(!NOISE_HASH.is_empty());
        assert!(!NOISE_VALUE.is_empty());
        assert!(!NOISE_PERLIN.is_empty());
        assert!(!NOISE_FBM.is_empty());
    }

    #[test]
    fn test_noise_fbm_has_octaves() {
        assert!(NOISE_FBM.contains("octave") || NOISE_FBM.contains("lacunarity"));
    }

    #[test]
    fn test_validate_shader_detects_missing_compute() {
        let bad_shader = "fn main() { }";
        let result = validate_demoscene_shader(bad_shader);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_shader_detects_missing_scene_sdf() {
        let bad_shader = "@compute fn main() { }";
        let result = validate_demoscene_shader(bad_shader);
        assert!(result.is_err());
    }
}

// =============================================================================
// 10. End-to-End Integration Tests
// =============================================================================

mod end_to_end_tests {
    use super::*;

    #[test]
    fn test_full_pipeline_minimal_render() {
        if let Some(device) = RhiDevice::try_new_headless() {
            // Create minimal renderer
            let renderer = MinimalRenderer::new(&device.device, 256, 256);

            // Render frame
            renderer.render(&device.device, &device.queue, 0.0);
            device.wait_idle();

            // Verify output exists
            let texture = renderer.output_texture();
            assert_eq!(texture.width(), 256);
            assert_eq!(texture.height(), 256);
        }
    }

    #[test]
    fn test_full_pipeline_hybrid_render() {
        if let Some(device) = RhiDevice::try_new_headless() {
            // Create hybrid renderer with depth buffer
            let mut renderer = HybridDepthRenderer::new(&device.device, 256, 256);

            // Create depth buffer
            let depth_texture = device.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Test Depth"),
                size: wgpu::Extent3d { width: 256, height: 256, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH_BUFFER_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Bind and render
            renderer.bind_depth_buffer(&device.device, &depth_view, 256, 256, false);
            renderer.update_and_upload(&device.queue, 0.5);

            let mut encoder = device.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("Test") }
            );
            renderer.dispatch(&mut encoder);
            device.queue.submit(std::iter::once(encoder.finish()));
            device.wait_idle();
        }
    }

    #[test]
    fn test_full_pipeline_multipass_render() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let mut renderer = MultiPassSdfRenderer::new(&device.device, 256, 256);

            // Update and dispatch all passes
            renderer.update(&device.queue, 0.5);

            let mut encoder = device.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("Test") }
            );
            renderer.dispatch_all(&mut encoder);
            device.queue.submit(std::iter::once(encoder.finish()));
            device.wait_idle();
        }
    }

    #[test]
    fn test_multiple_frame_rendering() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let mut renderer = HybridDepthRenderer::new(&device.device, 128, 128);

            // Render 10 frames
            for i in 0..10 {
                renderer.update_and_upload(&device.queue, i as f32 * 0.016);

                let mut encoder = device.device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("Frame") }
                );
                renderer.dispatch(&mut encoder);
                device.queue.submit(std::iter::once(encoder.finish()));
            }
            device.wait_idle();
        }
    }

    #[test]
    fn test_resize_and_rerender() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let mut renderer = HybridDepthRenderer::new(&device.device, 256, 256);

            // First render
            renderer.update_and_upload(&device.queue, 0.0);
            {
                let mut encoder = device.device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("Test") }
                );
                renderer.dispatch(&mut encoder);
                device.queue.submit(std::iter::once(encoder.finish()));
            }

            // Resize
            renderer.resize(&device.device, 512, 512);
            assert_eq!(renderer.size(), (512, 512));

            // Render again
            renderer.update_and_upload(&device.queue, 0.5);
            {
                let mut encoder = device.device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("Test") }
                );
                renderer.dispatch(&mut encoder);
                device.queue.submit(std::iter::once(encoder.finish()));
            }
            device.wait_idle();
        }
    }

    #[test]
    fn test_shader_compilation_on_device() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let shader_source = get_demo_scene_shader();

            // This should not panic if shader is valid
            let _module = device.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Demo Scene"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });
        }
    }

    #[test]
    fn test_minimal_shader_compilation_on_device() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let _module = device.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Minimal"),
                source: wgpu::ShaderSource::Wgsl(MINIMAL_SHADER.into()),
            });
        }
    }

    #[test]
    fn test_hybrid_shader_compilation_on_device() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let _module = device.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Hybrid Depth"),
                source: wgpu::ShaderSource::Wgsl(HYBRID_DEPTH_SHADER.into()),
            });
        }
    }
}

// =============================================================================
// Performance and Edge Case Tests
// =============================================================================

mod edge_case_tests {
    use super::*;

    #[test]
    fn test_zero_size_handling() {
        let config = HybridDepthConfig::new(0, 0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_very_large_resolution() {
        let config = HybridDepthConfig::new(7680, 4320); // 8K
        assert!(config.validate().is_ok());

        let (dx, dy, _) = config.dispatch_size();
        assert_eq!(dx, 960);  // 7680 / 8
        assert_eq!(dy, 540);  // 4320 / 8
    }

    #[test]
    fn test_negative_near_plane() {
        let mut config = HybridDepthConfig::default();
        config.near_plane = -1.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_far_less_than_near() {
        let mut config = HybridDepthConfig::default();
        config.near_plane = 100.0;
        config.far_plane = 10.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_uniforms_memory_layout() {
        assert_eq!(std::mem::size_of::<HybridUniforms>(), 32);
        assert_eq!(std::mem::align_of::<HybridUniforms>(), 4);
    }

    #[test]
    fn test_multipass_uniforms_memory_layout() {
        assert_eq!(std::mem::size_of::<MultiPassUniforms>(), 32);
    }

    #[test]
    fn test_post_process_uniforms_memory_layout() {
        assert_eq!(std::mem::size_of::<PostProcessUniforms>(), 32);
    }

    #[test]
    fn test_workgroup_size_constant() {
        assert_eq!(MULTIPASS_WORKGROUP_SIZE, 8);
    }

    #[test]
    fn test_max_transparent_objects() {
        assert!(MAX_TRANSPARENT_OBJECTS >= 64);
    }

    #[test]
    fn test_alpha_thresholds() {
        assert!(OPAQUE_ALPHA_THRESHOLD > 0.9);
        assert!(OPAQUE_ALPHA_THRESHOLD <= 1.0);
        assert!(TRANSPARENT_ALPHA_MIN > 0.0);
        assert!(TRANSPARENT_ALPHA_MIN < 0.1);
    }
}
