//! T-DEMO-6.3 / T-DEMO-6.4 Hybrid Mode Depth Buffer Tests
//!
//! This test module provides comprehensive coverage for the hybrid depth buffer
//! functionality that composites ray-marched SDF content with rasterized geometry.
//!
//! Run with: `cargo test demoscene_hybrid --features test-utils`

use renderer_backend::demoscene::{
    HybridDepthConfig, HybridDepthRenderer, HybridUniforms, DepthBufferBinding,
    DepthCompareResult, HYBRID_DEPTH_SHADER, DEPTH_BUFFER_FORMAT,
    DEFAULT_NEAR_PLANE, DEFAULT_FAR_PLANE, MAX_RAY_MARCH_DIST, DEPTH_EPSILON,
    ndc_to_linear, linear_to_ndc, reverse_z_to_linear, linear_to_reverse_z,
};
use renderer_backend::rhi_device::RhiDevice;

// =============================================================================
// T-DEMO-6.3: Depth Buffer Binding Tests
// =============================================================================

mod depth_buffer_binding_tests {
    use super::*;

    #[test]
    fn test_binding_default_state() {
        let binding = DepthBufferBinding::default();
        assert!(!binding.is_bound);
        assert_eq!(binding.width, 0);
        assert_eq!(binding.height, 0);
        assert!(!binding.is_reverse_z);
    }

    #[test]
    fn test_binding_creation() {
        let binding = DepthBufferBinding::new(1920, 1080, false);
        assert!(binding.is_bound);
        assert_eq!(binding.width, 1920);
        assert_eq!(binding.height, 1080);
    }

    #[test]
    fn test_binding_reverse_z_flag() {
        let binding = DepthBufferBinding::new(800, 600, true);
        assert!(binding.is_reverse_z);
    }

    #[test]
    fn test_binding_dimension_match() {
        let binding = DepthBufferBinding::new(1024, 768, false);
        assert!(binding.matches_dimensions(1024, 768));
        assert!(!binding.matches_dimensions(1920, 1080));
        assert!(!binding.matches_dimensions(1024, 1024));
    }

    #[test]
    fn test_binding_clear() {
        let mut binding = DepthBufferBinding::new(800, 600, true);
        assert!(binding.is_bound);
        binding.clear();
        assert!(!binding.is_bound);
        assert_eq!(binding.width, 0);
    }
}

// =============================================================================
// T-DEMO-6.3: Depth Sampling Tests
// =============================================================================

mod depth_sampling_tests {
    use super::*;

    #[test]
    fn test_ndc_to_linear_near_plane_exact() {
        let linear = ndc_to_linear(0.0, 0.1, 100.0);
        assert!((linear - 0.1).abs() < 0.001, "Near plane should be 0.1, got {}", linear);
    }

    #[test]
    fn test_ndc_to_linear_far_plane_exact() {
        let linear = ndc_to_linear(1.0, 0.1, 100.0);
        assert!((linear - 100.0).abs() < 0.001, "Far plane should be 100.0, got {}", linear);
    }

    #[test]
    fn test_ndc_to_linear_midpoint() {
        let linear = ndc_to_linear(0.5, 0.1, 100.0);
        assert!(linear > 0.1 && linear < 100.0, "Midpoint should be between near and far");
    }

    #[test]
    fn test_ndc_to_linear_quarter_point() {
        let linear = ndc_to_linear(0.25, 0.1, 100.0);
        assert!(linear > 0.1 && linear < 50.0, "Quarter point check");
    }

    #[test]
    fn test_ndc_to_linear_three_quarter_point() {
        let linear = ndc_to_linear(0.75, 0.1, 100.0);
        // Perspective projection concentrates depth near the near plane
        // At NDC 0.75, linear depth is still relatively close to near plane
        assert!(linear > 0.1 && linear < 100.0, "Three-quarter point: {}", linear);
    }

    #[test]
    fn test_linear_to_ndc_near_plane() {
        let ndc = linear_to_ndc(0.1, 0.1, 100.0);
        assert!(ndc.abs() < 0.001, "Near plane NDC should be 0.0, got {}", ndc);
    }

    #[test]
    fn test_linear_to_ndc_far_plane() {
        let ndc = linear_to_ndc(100.0, 0.1, 100.0);
        assert!((ndc - 1.0).abs() < 0.001, "Far plane NDC should be 1.0, got {}", ndc);
    }

    #[test]
    fn test_roundtrip_ndc_linear_multiple() {
        let near = 0.1;
        let far = 100.0;
        for ndc_val in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] {
            let linear = ndc_to_linear(ndc_val, near, far);
            let ndc_back = linear_to_ndc(linear, near, far);
            assert!(
                (ndc_val - ndc_back).abs() < 0.001,
                "Roundtrip failed for ndc={}: {} -> {} -> {}",
                ndc_val, ndc_val, linear, ndc_back
            );
        }
    }

    #[test]
    fn test_reverse_z_near_plane() {
        // In reverse-Z, 1.0 maps to near plane
        let linear = reverse_z_to_linear(1.0, 0.1, 100.0);
        assert!((linear - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_reverse_z_far_plane() {
        // In reverse-Z, 0.0 maps to far plane
        let linear = reverse_z_to_linear(0.0, 0.1, 100.0);
        assert!((linear - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_linear_to_reverse_z_roundtrip() {
        let near = 0.5;
        let far = 50.0;
        for rz_val in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let linear = reverse_z_to_linear(rz_val, near, far);
            let rz_back = linear_to_reverse_z(linear, near, far);
            assert!(
                (rz_val - rz_back).abs() < 0.001,
                "Reverse-Z roundtrip failed: {} -> {} -> {}",
                rz_val, linear, rz_back
            );
        }
    }
}

// =============================================================================
// T-DEMO-6.4: Depth Comparison Tests
// =============================================================================

mod depth_comparison_tests {
    use super::*;

    #[test]
    fn test_ray_march_closer() {
        let result = DepthCompareResult::compare(5.0, 10.0, DEPTH_EPSILON);
        assert_eq!(result, DepthCompareResult::RayMarchCloser);
    }

    #[test]
    fn test_raster_closer() {
        let result = DepthCompareResult::compare(15.0, 10.0, DEPTH_EPSILON);
        assert_eq!(result, DepthCompareResult::RasterCloser);
    }

    #[test]
    fn test_ray_march_miss() {
        let result = DepthCompareResult::compare(MAX_RAY_MARCH_DIST, 10.0, DEPTH_EPSILON);
        assert_eq!(result, DepthCompareResult::RayMarchMiss);
    }

    #[test]
    fn test_equal_depths() {
        let result = DepthCompareResult::compare(5.0, 5.0, 0.1);
        assert_eq!(result, DepthCompareResult::Equal);
    }

    #[test]
    fn test_near_equal_within_epsilon() {
        let result = DepthCompareResult::compare(5.0, 5.05, 0.1);
        assert_eq!(result, DepthCompareResult::Equal);
    }

    #[test]
    fn test_near_equal_outside_epsilon() {
        let result = DepthCompareResult::compare(5.0, 5.2, 0.1);
        assert_eq!(result, DepthCompareResult::RayMarchCloser);
    }

    #[test]
    fn test_should_write_ray_march_closer() {
        assert!(DepthCompareResult::RayMarchCloser.should_write_ray_march());
    }

    #[test]
    fn test_should_write_equal() {
        assert!(DepthCompareResult::Equal.should_write_ray_march());
    }

    #[test]
    fn test_should_not_write_raster_closer() {
        assert!(!DepthCompareResult::RasterCloser.should_write_ray_march());
    }

    #[test]
    fn test_should_not_write_miss() {
        assert!(!DepthCompareResult::RayMarchMiss.should_write_ray_march());
    }

    #[test]
    fn test_should_preserve_raster_closer() {
        assert!(DepthCompareResult::RasterCloser.should_preserve_raster());
    }

    #[test]
    fn test_should_preserve_miss() {
        assert!(DepthCompareResult::RayMarchMiss.should_preserve_raster());
    }

    #[test]
    fn test_boundary_near_max_dist() {
        // Just below max distance should not be a miss
        let result = DepthCompareResult::compare(MAX_RAY_MARCH_DIST - 0.1, 5.0, DEPTH_EPSILON);
        assert_eq!(result, DepthCompareResult::RasterCloser);
    }

    #[test]
    fn test_boundary_at_max_dist() {
        // Exactly at max distance (within epsilon) should be miss
        let result = DepthCompareResult::compare(MAX_RAY_MARCH_DIST - DEPTH_EPSILON / 2.0, 5.0, DEPTH_EPSILON);
        assert_eq!(result, DepthCompareResult::RayMarchMiss);
    }
}

// =============================================================================
// T-DEMO-6.3: HybridUniforms Tests
// =============================================================================

mod hybrid_uniforms_tests {
    use super::*;

    #[test]
    fn test_default_uniforms() {
        let uniforms = HybridUniforms::default();
        assert_eq!(uniforms.time, 0.0);
        assert_eq!(uniforms.resolution_x, 800.0);
        assert_eq!(uniforms.resolution_y, 600.0);
        assert_eq!(uniforms.near_plane, DEFAULT_NEAR_PLANE);
        assert_eq!(uniforms.far_plane, DEFAULT_FAR_PLANE);
        assert!(!uniforms.is_depth_enabled());
    }

    #[test]
    fn test_new_uniforms() {
        let uniforms = HybridUniforms::new(1920, 1080);
        assert_eq!(uniforms.resolution_x, 1920.0);
        assert_eq!(uniforms.resolution_y, 1080.0);
    }

    #[test]
    fn test_with_depth_uniforms() {
        let uniforms = HybridUniforms::with_depth(800, 600, 0.5, 200.0);
        assert_eq!(uniforms.near_plane, 0.5);
        assert_eq!(uniforms.far_plane, 200.0);
        assert!(uniforms.is_depth_enabled());
    }

    #[test]
    fn test_set_time() {
        let mut uniforms = HybridUniforms::default();
        uniforms.set_time(5.5);
        assert_eq!(uniforms.time, 5.5);
    }

    #[test]
    fn test_set_resolution() {
        let mut uniforms = HybridUniforms::default();
        uniforms.set_resolution(3840, 2160);
        assert_eq!(uniforms.resolution_x, 3840.0);
        assert_eq!(uniforms.resolution_y, 2160.0);
    }

    #[test]
    fn test_set_depth_planes() {
        let mut uniforms = HybridUniforms::default();
        uniforms.set_depth_planes(0.01, 1000.0);
        assert_eq!(uniforms.near_plane, 0.01);
        assert_eq!(uniforms.far_plane, 1000.0);
    }

    #[test]
    fn test_set_depth_planes_clamp_near() {
        let mut uniforms = HybridUniforms::default();
        uniforms.set_depth_planes(-1.0, 100.0);
        assert!(uniforms.near_plane > 0.0, "Near plane should be positive");
    }

    #[test]
    fn test_set_depth_planes_clamp_far() {
        let mut uniforms = HybridUniforms::default();
        uniforms.set_depth_planes(10.0, 5.0);
        assert!(uniforms.far_plane > uniforms.near_plane, "Far > near");
    }

    #[test]
    fn test_depth_enabled_toggle() {
        let mut uniforms = HybridUniforms::default();
        assert!(!uniforms.is_depth_enabled());
        uniforms.set_depth_enabled(true);
        assert!(uniforms.is_depth_enabled());
        uniforms.set_depth_enabled(false);
        assert!(!uniforms.is_depth_enabled());
    }

    #[test]
    fn test_resolution_getter() {
        let uniforms = HybridUniforms::new(1280, 720);
        let (w, h) = uniforms.resolution();
        assert_eq!(w, 1280);
        assert_eq!(h, 720);
    }

    #[test]
    fn test_as_bytes_size() {
        let uniforms = HybridUniforms::default();
        let bytes = uniforms.as_bytes();
        assert_eq!(bytes.len(), 32, "HybridUniforms should be 32 bytes");
    }

    #[test]
    fn test_memory_layout() {
        assert_eq!(std::mem::size_of::<HybridUniforms>(), 32);
        assert_eq!(std::mem::align_of::<HybridUniforms>(), 4);
    }

    #[test]
    fn test_ndc_to_linear_via_uniforms() {
        let uniforms = HybridUniforms::with_depth(800, 600, 0.1, 100.0);
        let linear = uniforms.ndc_to_linear_depth(0.0);
        assert!((linear - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_linear_to_ndc_via_uniforms() {
        let uniforms = HybridUniforms::with_depth(800, 600, 0.1, 100.0);
        let ndc = uniforms.linear_to_ndc_depth(100.0);
        assert!((ndc - 1.0).abs() < 0.001);
    }
}

// =============================================================================
// T-DEMO-6.3/6.4: HybridDepthConfig Tests
// =============================================================================

mod hybrid_config_tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = HybridDepthConfig::default();
        assert_eq!(config.width, 800);
        assert_eq!(config.height, 600);
        assert_eq!(config.near_plane, DEFAULT_NEAR_PLANE);
        assert_eq!(config.far_plane, DEFAULT_FAR_PLANE);
        assert!(!config.reverse_z);
    }

    #[test]
    fn test_new_config() {
        let config = HybridDepthConfig::new(1920, 1080);
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
    }

    #[test]
    fn test_with_depth_planes() {
        let config = HybridDepthConfig::new(800, 600).with_depth_planes(0.5, 500.0);
        assert_eq!(config.near_plane, 0.5);
        assert_eq!(config.far_plane, 500.0);
    }

    #[test]
    fn test_with_reverse_z() {
        let config = HybridDepthConfig::new(800, 600).with_reverse_z(true);
        assert!(config.reverse_z);
    }

    #[test]
    fn test_with_epsilon() {
        let config = HybridDepthConfig::new(800, 600).with_epsilon(0.01);
        assert_eq!(config.depth_epsilon, 0.01);
    }

    #[test]
    fn test_dispatch_size_aligned() {
        let config = HybridDepthConfig::new(800, 600);
        let (x, y, z) = config.dispatch_size();
        assert_eq!(x, 100); // 800 / 8
        assert_eq!(y, 75);  // 600 / 8
        assert_eq!(z, 1);
    }

    #[test]
    fn test_dispatch_size_unaligned() {
        let config = HybridDepthConfig::new(801, 601);
        let (x, y, z) = config.dispatch_size();
        assert_eq!(x, 101); // (801 + 7) / 8
        assert_eq!(y, 76);  // (601 + 7) / 8
        assert_eq!(z, 1);
    }

    #[test]
    fn test_dispatch_size_4k() {
        let config = HybridDepthConfig::new(3840, 2160);
        let (x, y, z) = config.dispatch_size();
        assert_eq!(x, 480); // 3840 / 8
        assert_eq!(y, 270); // 2160 / 8
        assert_eq!(z, 1);
    }

    #[test]
    fn test_validate_valid() {
        let config = HybridDepthConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_zero_width() {
        let mut config = HybridDepthConfig::default();
        config.width = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_zero_height() {
        let mut config = HybridDepthConfig::default();
        config.height = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_negative_near() {
        let mut config = HybridDepthConfig::default();
        config.near_plane = -1.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_far_less_than_near() {
        let mut config = HybridDepthConfig::default();
        config.near_plane = 10.0;
        config.far_plane = 5.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_to_uniforms() {
        let config = HybridDepthConfig::new(1024, 768).with_depth_planes(0.5, 200.0);
        let uniforms = config.to_uniforms();
        assert_eq!(uniforms.resolution_x, 1024.0);
        assert_eq!(uniforms.resolution_y, 768.0);
        assert_eq!(uniforms.near_plane, 0.5);
        assert_eq!(uniforms.far_plane, 200.0);
    }
}

// =============================================================================
// Shader Validation Tests
// =============================================================================

mod shader_tests {
    use super::*;

    #[test]
    fn test_shader_non_empty() {
        assert!(!HYBRID_DEPTH_SHADER.is_empty());
    }

    #[test]
    fn test_shader_has_entry_point() {
        assert!(HYBRID_DEPTH_SHADER.contains("fn main"));
    }

    #[test]
    fn test_shader_has_compute_attribute() {
        assert!(HYBRID_DEPTH_SHADER.contains("@compute"));
    }

    #[test]
    fn test_shader_has_workgroup_size() {
        assert!(HYBRID_DEPTH_SHADER.contains("@workgroup_size"));
    }

    #[test]
    fn test_shader_has_hybrid_uniforms() {
        assert!(HYBRID_DEPTH_SHADER.contains("HybridUniforms"));
    }

    #[test]
    fn test_shader_has_near_plane() {
        assert!(HYBRID_DEPTH_SHADER.contains("near_plane"));
    }

    #[test]
    fn test_shader_has_far_plane() {
        assert!(HYBRID_DEPTH_SHADER.contains("far_plane"));
    }

    #[test]
    fn test_shader_has_depth_enabled() {
        assert!(HYBRID_DEPTH_SHADER.contains("depth_enabled"));
    }

    #[test]
    fn test_shader_has_depth_texture_binding() {
        assert!(HYBRID_DEPTH_SHADER.contains("depth_texture"));
    }

    #[test]
    fn test_shader_has_depth_sampler_binding() {
        assert!(HYBRID_DEPTH_SHADER.contains("depth_sampler"));
    }

    #[test]
    fn test_shader_has_ndc_to_linear() {
        assert!(HYBRID_DEPTH_SHADER.contains("ndc_to_linear"));
    }

    #[test]
    fn test_shader_has_sample_depth() {
        assert!(HYBRID_DEPTH_SHADER.contains("sample_depth"));
    }

    #[test]
    fn test_shader_has_ray_march() {
        assert!(HYBRID_DEPTH_SHADER.contains("ray_march"));
    }

    #[test]
    fn test_shader_has_max_steps() {
        assert!(HYBRID_DEPTH_SHADER.contains("MAX_STEPS"));
    }

    #[test]
    fn test_shader_has_scene_sdf() {
        assert!(HYBRID_DEPTH_SHADER.contains("scene_sdf"));
    }
}

// =============================================================================
// GPU Integration Tests
// =============================================================================

mod gpu_tests {
    use super::*;

    #[test]
    fn test_renderer_creation() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let renderer = HybridDepthRenderer::new(&device.device, 800, 600);
            assert_eq!(renderer.size(), (800, 600));
            assert!(!renderer.has_depth_buffer());
        }
    }

    #[test]
    fn test_renderer_with_config() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let config = HybridDepthConfig::new(1280, 720).with_depth_planes(0.5, 200.0);
            let renderer = HybridDepthRenderer::with_config(&device.device, config);
            assert_eq!(renderer.size(), (1280, 720));
        }
    }

    #[test]
    fn test_renderer_update_time() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let mut renderer = HybridDepthRenderer::new(&device.device, 800, 600);
            renderer.update(1.5);
            assert_eq!(renderer.uniforms().time, 1.5);
        }
    }

    #[test]
    fn test_renderer_update_and_upload() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let mut renderer = HybridDepthRenderer::new(&device.device, 800, 600);
            renderer.update_and_upload(&device.queue, 2.5);
            assert_eq!(renderer.uniforms().time, 2.5);
        }
    }

    #[test]
    fn test_renderer_dispatch_no_depth() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let renderer = HybridDepthRenderer::new(&device.device, 800, 600);
            let mut encoder = device.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("Test") }
            );
            renderer.dispatch(&mut encoder);
            device.queue.submit(std::iter::once(encoder.finish()));
            device.wait_idle();
        }
    }

    #[test]
    fn test_renderer_resize() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let mut renderer = HybridDepthRenderer::new(&device.device, 800, 600);
            renderer.resize(&device.device, 1920, 1080);
            assert_eq!(renderer.size(), (1920, 1080));
        }
    }

    #[test]
    fn test_renderer_resize_no_change() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let mut renderer = HybridDepthRenderer::new(&device.device, 800, 600);
            renderer.resize(&device.device, 800, 600);
            assert_eq!(renderer.size(), (800, 600));
        }
    }

    #[test]
    fn test_renderer_set_depth_planes() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let mut renderer = HybridDepthRenderer::new(&device.device, 800, 600);
            renderer.set_depth_planes(0.01, 500.0);
            assert_eq!(renderer.uniforms().near_plane, 0.01);
            assert_eq!(renderer.uniforms().far_plane, 500.0);
        }
    }

    #[test]
    fn test_renderer_bind_depth_buffer() {
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
            assert!(renderer.uniforms().is_depth_enabled());
        }
    }

    #[test]
    fn test_renderer_unbind_depth_buffer() {
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
            renderer.unbind_depth_buffer(&device.device);
            assert!(!renderer.has_depth_buffer());
            assert!(!renderer.uniforms().is_depth_enabled());
        }
    }

    #[test]
    fn test_renderer_dispatch_with_depth() {
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
            renderer.update_and_upload(&device.queue, 1.0);

            let mut encoder = device.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("Test") }
            );
            renderer.dispatch(&mut encoder);
            device.queue.submit(std::iter::once(encoder.finish()));
            device.wait_idle();
        }
    }

    #[test]
    fn test_renderer_output_view() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let renderer = HybridDepthRenderer::new(&device.device, 800, 600);
            let _view = renderer.output_view();
        }
    }

    #[test]
    fn test_renderer_output_texture() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let renderer = HybridDepthRenderer::new(&device.device, 800, 600);
            let texture = renderer.output_texture();
            assert_eq!(texture.width(), 800);
            assert_eq!(texture.height(), 600);
        }
    }

    #[test]
    fn test_renderer_config_accessor() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let config = HybridDepthConfig::new(1024, 768);
            let renderer = HybridDepthRenderer::with_config(&device.device, config);
            assert_eq!(renderer.config().width, 1024);
            assert_eq!(renderer.config().height, 768);
        }
    }

    #[test]
    fn test_renderer_multiple_dispatches() {
        if let Some(device) = RhiDevice::try_new_headless() {
            let mut renderer = HybridDepthRenderer::new(&device.device, 800, 600);

            for i in 0..5 {
                renderer.update_and_upload(&device.queue, i as f32 * 0.1);
                let mut encoder = device.device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("Test") }
                );
                renderer.dispatch(&mut encoder);
                device.queue.submit(std::iter::once(encoder.finish()));
            }
            device.wait_idle();
        }
    }
}

// =============================================================================
// Edge Case Tests
// =============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn test_extreme_near_plane() {
        let linear = ndc_to_linear(0.001, 0.001, 1000.0);
        assert!(linear >= 0.001 && linear < 0.01);
    }

    #[test]
    fn test_extreme_far_plane() {
        // Perspective projection: depth is non-linear, so 0.999 NDC
        // does not map to 9990 linear with large far plane
        let linear = ndc_to_linear(0.999, 0.1, 10000.0);
        assert!(linear > 50.0 && linear < 200.0, "Got {}", linear);

        // NDC 1.0 should give the far plane
        let far = ndc_to_linear(1.0, 0.1, 10000.0);
        assert!((far - 10000.0).abs() < 1.0);
    }

    #[test]
    fn test_very_small_depth_range() {
        let linear = ndc_to_linear(0.5, 0.1, 0.2);
        assert!(linear > 0.1 && linear < 0.2);
    }

    #[test]
    fn test_large_depth_range() {
        let linear = ndc_to_linear(0.5, 0.001, 100000.0);
        assert!(linear > 0.001 && linear < 100000.0);
    }

    #[test]
    fn test_depth_compare_zero_epsilon() {
        // With zero epsilon, equal depths result in diff.abs() < 0 being false
        // so we get RasterCloser (diff = 0.0, which is not < 0.0)
        // Actually, diff = 5.0 - 5.0 = 0.0, and 0.0.abs() = 0.0 < 0.0 is false
        // So diff < 0.0 is also false, meaning we get RasterCloser
        let result = DepthCompareResult::compare(5.0, 5.0, 0.0);
        // With exactly zero epsilon, the result depends on the comparison order
        // diff = 0, diff.abs() = 0, 0 < 0 is false, diff < 0 is false
        // So we get RasterCloser (the else branch)
        assert_eq!(result, DepthCompareResult::RasterCloser);
    }

    #[test]
    fn test_depth_compare_large_epsilon() {
        let result = DepthCompareResult::compare(5.0, 10.0, 10.0);
        assert_eq!(result, DepthCompareResult::Equal);
    }

    #[test]
    fn test_config_chain_all_methods() {
        let config = HybridDepthConfig::new(1920, 1080)
            .with_depth_planes(0.01, 1000.0)
            .with_reverse_z(true)
            .with_epsilon(0.001);

        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.near_plane, 0.01);
        assert_eq!(config.far_plane, 1000.0);
        assert!(config.reverse_z);
        assert_eq!(config.depth_epsilon, 0.001);
    }

    #[test]
    fn test_uniforms_copy() {
        let u1 = HybridUniforms::with_depth(800, 600, 0.1, 100.0);
        let u2 = u1;
        assert_eq!(u1, u2);
    }

    #[test]
    fn test_config_copy() {
        let c1 = HybridDepthConfig::new(800, 600);
        let c2 = c1;
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_depth_compare_result_debug() {
        let variants = [
            DepthCompareResult::RayMarchCloser,
            DepthCompareResult::RasterCloser,
            DepthCompareResult::RayMarchMiss,
            DepthCompareResult::Equal,
        ];
        for v in variants {
            let _ = format!("{:?}", v);
        }
    }
}

// =============================================================================
// Constants Tests
// =============================================================================

mod constants_tests {
    use super::*;

    #[test]
    fn test_default_near_plane() {
        assert_eq!(DEFAULT_NEAR_PLANE, 0.1);
    }

    #[test]
    fn test_default_far_plane() {
        assert_eq!(DEFAULT_FAR_PLANE, 100.0);
    }

    #[test]
    fn test_max_ray_march_dist() {
        assert_eq!(MAX_RAY_MARCH_DIST, 20.0);
    }

    #[test]
    fn test_depth_buffer_format() {
        assert_eq!(DEPTH_BUFFER_FORMAT, wgpu::TextureFormat::Depth32Float);
    }

    #[test]
    fn test_depth_epsilon_positive() {
        assert!(DEPTH_EPSILON > 0.0);
    }

    #[test]
    fn test_depth_epsilon_small() {
        assert!(DEPTH_EPSILON < 0.01);
    }
}

// =============================================================================
// Compositing Logic Tests
// =============================================================================

mod compositing_tests {
    use super::*;

    #[test]
    fn test_ray_closer_than_raster_writes() {
        // Ray at 5.0, raster at 10.0 -> ray should write
        let result = DepthCompareResult::compare(5.0, 10.0, DEPTH_EPSILON);
        assert!(result.should_write_ray_march());
        assert!(!result.should_preserve_raster());
    }

    #[test]
    fn test_raster_closer_preserves() {
        // Ray at 10.0, raster at 5.0 -> preserve raster
        let result = DepthCompareResult::compare(10.0, 5.0, DEPTH_EPSILON);
        assert!(!result.should_write_ray_march());
        assert!(result.should_preserve_raster());
    }

    #[test]
    fn test_miss_preserves_raster() {
        // Ray missed -> preserve raster
        let result = DepthCompareResult::compare(MAX_RAY_MARCH_DIST, 5.0, DEPTH_EPSILON);
        assert!(!result.should_write_ray_march());
        assert!(result.should_preserve_raster());
    }

    #[test]
    fn test_equal_prefers_ray() {
        // Equal depths -> prefer ray march for consistency
        let result = DepthCompareResult::compare(5.0, 5.0, 0.1);
        assert!(result.should_write_ray_march());
    }

    #[test]
    fn test_compositing_sequence() {
        // Simulate a sequence of depth comparisons
        let test_cases = [
            (1.0, 5.0, true),   // ray closer
            (5.0, 1.0, false),  // raster closer
            (3.0, 3.0, true),   // equal
            (MAX_RAY_MARCH_DIST, 2.0, false), // miss
        ];

        for (ray_dist, raster_depth, should_write) in test_cases {
            let result = DepthCompareResult::compare(ray_dist, raster_depth, 0.1);
            assert_eq!(
                result.should_write_ray_march(), should_write,
                "Failed for ray={}, raster={}", ray_dist, raster_depth
            );
        }
    }
}
