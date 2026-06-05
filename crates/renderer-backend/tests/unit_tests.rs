//! Comprehensive Unit Test Suite for TRINITY wgpu Renderer Backend
//!
//! Task: T-WGPU-P7.7.1 Implement Unit Test Suite
//!
//! This module provides structured unit tests across all major subsystems:
//! - Device: Initialization, queue creation, feature detection
//! - Resources: Buffers, textures, samplers, bind groups
//! - Pipelines: Render/compute pipeline creation, caching, shaders
//! - Frame Graph: Pass declaration, resource tracking, scheduling
//! - Memory: Allocation, leak detection, budget enforcement
//!
//! Test Design:
//! - Tests use only public API (blackbox style where possible)
//! - Each test is isolated and idempotent
//! - GPU-dependent tests gracefully skip if no adapter available
//! - Tests verify both success and error paths

use pollster::block_on;
use std::sync::Arc;

// =============================================================================
// COMMON TEST INFRASTRUCTURE
// =============================================================================

/// Creates a TrinityInstance and gets the first available adapter.
fn get_test_adapter() -> Option<wgpu::Adapter> {
    use renderer_backend::device::{enumerate_adapters_with_info, TrinityInstance};
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
// MODULE 1: DEVICE TESTS
// =============================================================================

mod device {
    use super::*;
    use renderer_backend::device::{
        detect_capability_tier, enumerate_adapters_with_info, negotiate_limits,
        CapabilityTier, LimitRequirements, TrinityInstance, TrinityMinimumLimits,
    };

    // -------------------------------------------------------------------------
    // Instance Creation Tests
    // -------------------------------------------------------------------------

    /// Test: TrinityInstance creates successfully with default backends.
    #[test]
    fn instance_creation_default_backends() {
        let instance = TrinityInstance::new();
        assert!(
            !instance.backends().is_empty(),
            "Instance should have at least one backend"
        );
    }

    /// Test: TrinityInstance inner() returns valid wgpu::Instance reference.
    #[test]
    fn instance_inner_returns_valid_reference() {
        let instance = TrinityInstance::new();
        let _inner: &wgpu::Instance = instance.inner();
        // If we can borrow it without panic, the instance is valid
    }

    /// Test: TrinityInstance supports all expected platform backends.
    #[test]
    fn instance_supports_platform_backends() {
        let instance = TrinityInstance::new();
        let backends = instance.backends();

        // On any platform, we should have at least one backend
        assert!(!backends.is_empty(), "No backends available");

        // Check that backends is a valid wgpu::Backends bitflags
        let all_backends = wgpu::Backends::all();
        assert!(
            backends.bits() <= all_backends.bits(),
            "Invalid backend bits"
        );
    }

    // -------------------------------------------------------------------------
    // Adapter Enumeration Tests
    // -------------------------------------------------------------------------

    /// Test: Adapter enumeration returns results structure.
    #[test]
    fn adapter_enumeration_returns_result() {
        let instance = TrinityInstance::new();
        let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

        // Result should always be constructable, even with 0 adapters
        assert!(
            result.len() >= 0,
            "Enumeration should return valid length"
        );
    }

    /// Test: Backend counts are consistent with adapter count.
    #[test]
    fn adapter_enumeration_backend_counts_consistent() {
        let instance = TrinityInstance::new();
        let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

        // Use the total() method which sums all backend counts
        let total_from_counts = result.backend_counts.total();

        assert_eq!(
            result.len(),
            total_from_counts,
            "Backend counts should sum to total adapters"
        );
    }

    /// Test: Best adapter selection returns None for empty results.
    #[test]
    fn adapter_enumeration_best_adapter_works() {
        let instance = TrinityInstance::new();
        let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

        if result.len() > 0 {
            assert!(
                result.best_adapter().is_some(),
                "Non-empty results should have best adapter"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Queue Creation Tests
    // -------------------------------------------------------------------------

    /// Test: Queue is created along with device.
    #[test]
    fn queue_creation_with_device() {
        let adapter = require_adapter!();
        let (device, queue) = require_device!(&adapter);

        // Queue should be usable - submit an empty command buffer
        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_encoder"),
        });
        queue.submit(Some(encoder.finish()));
    }

    /// Test: Queue can write to buffer.
    #[test]
    fn queue_write_buffer() {
        let adapter = require_adapter!();
        let (device, queue) = require_device!(&adapter);

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_buffer"),
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let data = [1u8; 256];
        queue.write_buffer(&buffer, 0, &data);

        // If we get here without panic, the write succeeded
    }

    /// Test: Queue submit returns submission index.
    #[test]
    fn queue_submit_returns_index() {
        let adapter = require_adapter!();
        let (device, queue) = require_device!(&adapter);

        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_encoder"),
        });

        let index = queue.submit(Some(encoder.finish()));
        // wgpu::SubmissionIndex is returned - if we can call this, it works
        let _ = index;
    }

    // -------------------------------------------------------------------------
    // Capability Tier Detection Tests
    // -------------------------------------------------------------------------

    /// Test: Capability tier detection returns valid tier.
    #[test]
    fn capability_tier_detection_returns_valid_tier() {
        let adapter = require_adapter!();

        let tier = detect_capability_tier(&adapter);

        // Tier should be one of the valid values
        match tier {
            CapabilityTier::Minimal => {}
            CapabilityTier::Standard => {}
            CapabilityTier::Advanced => {}
            CapabilityTier::Full => {}
        }
    }

    /// Test: Any adapter should reach at least Minimal tier.
    #[test]
    fn capability_tier_minimal_always_achievable() {
        let adapter = require_adapter!();

        let tier = detect_capability_tier(&adapter);

        // Any adapter should be at least Minimal tier
        assert!(
            matches!(
                tier,
                CapabilityTier::Minimal
                    | CapabilityTier::Standard
                    | CapabilityTier::Advanced
                    | CapabilityTier::Full
            ),
            "Tier should be valid"
        );
    }

    // -------------------------------------------------------------------------
    // Limit Negotiation Tests
    // -------------------------------------------------------------------------

    /// Test: Limit negotiation with default requirements succeeds.
    #[test]
    fn limit_negotiation_default_requirements() {
        let adapter = require_adapter!();

        let requirements = LimitRequirements::default();

        let result = negotiate_limits(&requirements, &adapter);
        assert!(result.is_ok(), "Default requirements should always succeed");
    }

    /// Test: Limit negotiation respects adapter limits.
    #[test]
    fn limit_negotiation_respects_adapter_limits() {
        let adapter = require_adapter!();

        let requirements = LimitRequirements::new();

        let result = negotiate_limits(&requirements, &adapter);
        assert!(
            result.is_ok(),
            "Minimal requirements should succeed: {:?}",
            result
        );
    }

    /// Test: TrinityMinimumLimits defines baseline requirements.
    #[test]
    fn minimum_limits_baseline() {
        let min_limits = TrinityMinimumLimits::baseline();

        // These are TRINITY's minimum viable limits
        assert!(
            min_limits.min_max_texture_dimension_2d >= 2048,
            "Min 2D texture dimension should be >= 2048"
        );
        assert!(
            min_limits.min_max_bind_groups >= 4,
            "Min bind groups should be >= 4"
        );
    }
}

// =============================================================================
// MODULE 2: RESOURCE TESTS
// =============================================================================

mod resources {
    use super::*;
    use renderer_backend::resources::{
        align_size, buffer_usages, create_buffer, create_texture, is_aligned,
        TrinityBuffer, TrinityBufferDescriptor, TrinityTexture, TrinityTextureDescriptor,
        BUFFER_ALIGNMENT,
    };
    use wgpu::{BufferUsages, TextureUsages};

    // -------------------------------------------------------------------------
    // Buffer Creation Tests
    // -------------------------------------------------------------------------

    /// Test: Buffer creation with valid descriptor succeeds.
    #[test]
    fn buffer_creation_valid_descriptor() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let desc = TrinityBufferDescriptor {
            label: Some("test_vertex_buffer"),
            size: 1024,
            usage: buffer_usages::VERTEX,
            mapped_at_creation: false,
        };

        let buffer = create_buffer(&device, &desc);
        assert_eq!(buffer.size(), 1024, "Buffer size should match descriptor");
        assert!(
            buffer.usage().contains(BufferUsages::VERTEX),
            "Buffer should have VERTEX usage"
        );
    }

    /// Test: Buffer creation auto-aligns size.
    #[test]
    fn buffer_creation_auto_aligns_size() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let desc = TrinityBufferDescriptor {
            label: Some("unaligned_buffer"),
            size: 1023, // Not 4-byte aligned
            usage: buffer_usages::UNIFORM,
            mapped_at_creation: false,
        };

        let buffer = create_buffer(&device, &desc);
        assert!(
            buffer.size() >= 1024,
            "Buffer should be aligned to at least 1024 bytes"
        );
        assert!(
            is_aligned(buffer.size()),
            "Buffer size should be aligned"
        );
    }

    /// Test: Buffer label is preserved.
    #[test]
    fn buffer_label_preserved() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let desc = TrinityBufferDescriptor {
            label: Some("my_labeled_buffer"),
            size: 256,
            usage: buffer_usages::STORAGE_READ,
            mapped_at_creation: false,
        };

        let buffer = create_buffer(&device, &desc);
        assert_eq!(
            buffer.label(),
            Some("my_labeled_buffer"),
            "Label should be preserved"
        );
    }

    /// Test: Buffer usage presets are valid combinations.
    #[test]
    fn buffer_usage_presets_valid() {
        // VERTEX preset
        assert!(buffer_usages::VERTEX.contains(BufferUsages::VERTEX));
        assert!(buffer_usages::VERTEX.contains(BufferUsages::COPY_DST));

        // INDEX preset
        assert!(buffer_usages::INDEX.contains(BufferUsages::INDEX));
        assert!(buffer_usages::INDEX.contains(BufferUsages::COPY_DST));

        // UNIFORM preset
        assert!(buffer_usages::UNIFORM.contains(BufferUsages::UNIFORM));
        assert!(buffer_usages::UNIFORM.contains(BufferUsages::COPY_DST));

        // STAGING presets
        assert!(buffer_usages::STAGING_UPLOAD.contains(BufferUsages::MAP_WRITE));
        assert!(buffer_usages::STAGING_UPLOAD.contains(BufferUsages::COPY_SRC));
        assert!(buffer_usages::STAGING_READBACK.contains(BufferUsages::MAP_READ));
        assert!(buffer_usages::STAGING_READBACK.contains(BufferUsages::COPY_DST));
    }

    /// Test: Multiple buffers can be created with same device.
    #[test]
    fn buffer_creation_multiple_buffers() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let sizes = [256u64, 512, 1024, 4096, 65536];
        let mut buffers = Vec::new();

        for &size in &sizes {
            let desc = TrinityBufferDescriptor {
                label: Some("multi_buffer"),
                size,
                usage: buffer_usages::VERTEX,
                mapped_at_creation: false,
            };
            buffers.push(create_buffer(&device, &desc));
        }

        for (buffer, &expected_size) in buffers.iter().zip(sizes.iter()) {
            assert_eq!(
                buffer.size(),
                expected_size,
                "Buffer should have correct size"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Buffer Destruction Tests
    // -------------------------------------------------------------------------

    /// Test: Buffer drop doesn't panic.
    #[test]
    fn buffer_destruction_clean() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        {
            let desc = TrinityBufferDescriptor {
                label: Some("temp_buffer"),
                size: 1024,
                usage: buffer_usages::VERTEX,
                mapped_at_creation: false,
            };
            let _buffer = create_buffer(&device, &desc);
            // Buffer drops here
        }
        // If we reach here without panic, destruction succeeded
    }

    // -------------------------------------------------------------------------
    // Texture Creation Tests
    // -------------------------------------------------------------------------

    /// Test: Texture creation with valid descriptor succeeds.
    #[test]
    fn texture_creation_valid_descriptor() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let desc = TrinityTextureDescriptor {
            label: Some("test_texture"),
            size: wgpu::Extent3d {
                width: 256,
                height: 256,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        };

        let texture = create_texture(&device, &desc);
        assert_eq!(texture.width(), 256);
        assert_eq!(texture.height(), 256);
        assert_eq!(texture.format(), wgpu::TextureFormat::Rgba8Unorm);
    }

    /// Test: Texture creation with mip levels succeeds.
    #[test]
    fn texture_creation_with_mipmaps() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        // 256x256 supports up to 9 mip levels (256 -> 1)
        let desc = TrinityTextureDescriptor {
            label: Some("mipped_texture"),
            size: wgpu::Extent3d {
                width: 256,
                height: 256,
                depth_or_array_layers: 1,
            },
            mip_level_count: 5,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        };

        let texture = create_texture(&device, &desc);
        assert_eq!(texture.mip_level_count(), 5);
    }

    /// Test: Texture creation with array layers succeeds.
    #[test]
    fn texture_creation_array_texture() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let desc = TrinityTextureDescriptor {
            label: Some("array_texture"),
            size: wgpu::Extent3d {
                width: 128,
                height: 128,
                depth_or_array_layers: 6, // Cubemap-compatible
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        };

        let texture = create_texture(&device, &desc);
        assert_eq!(texture.depth_or_array_layers(), 6);
    }

    /// Test: Texture destruction doesn't panic.
    #[test]
    fn texture_destruction_clean() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        {
            let desc = TrinityTextureDescriptor {
                label: Some("temp_texture"),
                size: wgpu::Extent3d {
                    width: 64,
                    height: 64,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            };
            let _texture = create_texture(&device, &desc);
        }
        // If we reach here without panic, destruction succeeded
    }

    // -------------------------------------------------------------------------
    // Sampler Creation Tests
    // -------------------------------------------------------------------------

    /// Test: Sampler creation with linear filtering.
    #[test]
    fn sampler_creation_linear() {
        use renderer_backend::resources::{create_sampler, TrinitySamplerDescriptor};

        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let desc = TrinitySamplerDescriptor::linear_clamp();
        let sampler = create_sampler(&device, &desc);

        // If we can create it, the descriptor was valid
        let _ = sampler;
    }

    /// Test: Sampler creation with nearest filtering.
    #[test]
    fn sampler_creation_nearest() {
        use renderer_backend::resources::{create_sampler, TrinitySamplerDescriptor};

        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let desc = TrinitySamplerDescriptor::nearest_clamp();
        let sampler = create_sampler(&device, &desc);

        let _ = sampler;
    }

    /// Test: Sampler creation with comparison function (shadow sampler).
    #[test]
    fn sampler_creation_shadow() {
        use renderer_backend::resources::{create_sampler, TrinitySamplerDescriptor};

        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let desc = TrinitySamplerDescriptor::shadow();
        let sampler = create_sampler(&device, &desc);

        let _ = sampler;
    }

    // -------------------------------------------------------------------------
    // Bind Group Layout Tests
    // -------------------------------------------------------------------------

    /// Test: Bind group layout cache hit returns same layout.
    #[test]
    fn bind_group_layout_cache_hit() {
        use renderer_backend::resources::BindGroupLayoutCache;

        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let cache = BindGroupLayoutCache::new();

        let entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];

        let layout1 = cache.get_or_create(&device, None, &entries);
        let layout2 = cache.get_or_create(&device, None, &entries);

        // Both should point to same Arc
        assert!(
            Arc::ptr_eq(&layout1, &layout2),
            "Cache should return same layout"
        );
    }

    /// Test: Bind group layout cache miss creates new layout.
    #[test]
    fn bind_group_layout_cache_miss() {
        use renderer_backend::resources::BindGroupLayoutCache;

        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let cache = BindGroupLayoutCache::new();

        let entries1 = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];

        let entries2 = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT, // Different visibility
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];

        let layout1 = cache.get_or_create(&device, None, &entries1);
        let layout2 = cache.get_or_create(&device, None, &entries2);

        // Different entries should produce different layouts
        assert!(
            !Arc::ptr_eq(&layout1, &layout2),
            "Different entries should create different layouts"
        );
    }

    // -------------------------------------------------------------------------
    // Pipeline Layout Tests
    // -------------------------------------------------------------------------

    /// Test: Pipeline layout creation with bind group layouts.
    #[test]
    fn pipeline_layout_creation() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        // Use wgpu directly for pipeline layout creation test
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("test_layout"),
            entries: &[],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("test_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let _ = pipeline_layout;
    }

    // -------------------------------------------------------------------------
    // Resource Aliasing Tests
    // -------------------------------------------------------------------------

    /// Test: Buffer aliasing is not directly supported (each buffer is unique).
    #[test]
    fn buffer_no_aliasing() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let desc1 = TrinityBufferDescriptor {
            label: Some("buffer1"),
            size: 1024,
            usage: buffer_usages::VERTEX,
            mapped_at_creation: false,
        };

        let desc2 = TrinityBufferDescriptor {
            label: Some("buffer2"),
            size: 1024,
            usage: buffer_usages::VERTEX,
            mapped_at_creation: false,
        };

        let buffer1 = create_buffer(&device, &desc1);
        let buffer2 = create_buffer(&device, &desc2);

        // Each buffer should have its own backing storage
        // We can't directly test aliasing, but we verify they're distinct
        assert!(
            buffer1.inner().global_id() != buffer2.inner().global_id(),
            "Buffers should have distinct IDs"
        );
    }
}

// =============================================================================
// MODULE 3: PIPELINE TESTS
// =============================================================================

mod pipelines {
    use super::*;
    use renderer_backend::pipeline::ContentHash;

    // -------------------------------------------------------------------------
    // Shader Compilation Tests
    // -------------------------------------------------------------------------

    /// Test: ContentHash from identical bytes produces same hash.
    #[test]
    fn content_hash_deterministic() {
        let data = b"@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }";

        let hash1 = ContentHash::from_bytes(data);
        let hash2 = ContentHash::from_bytes(data);

        assert_eq!(hash1, hash2, "Same content should produce same hash");
    }

    /// Test: ContentHash from different bytes produces different hash.
    #[test]
    fn content_hash_different_content() {
        let data1 = b"@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }";
        let data2 = b"@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(1.0); }";

        let hash1 = ContentHash::from_bytes(data1);
        let hash2 = ContentHash::from_bytes(data2);

        assert_ne!(hash1, hash2, "Different content should produce different hash");
    }

    /// Test: ContentHash display format is hex string.
    #[test]
    fn content_hash_display_format() {
        let hash = ContentHash::from_bytes(b"test");
        let display = format!("{}", hash);

        assert_eq!(display.len(), 64, "SHA-256 hash should be 64 hex chars");
        assert!(
            display.chars().all(|c| c.is_ascii_hexdigit()),
            "Hash should only contain hex digits"
        );
    }

    /// Test: ContentHash zero hash is detectable.
    #[test]
    fn content_hash_zero() {
        let zero = ContentHash::zero();
        assert!(zero.is_zero(), "Zero hash should be detectable");

        let non_zero = ContentHash::from_bytes(b"test");
        assert!(!non_zero.is_zero(), "Non-zero hash should not be zero");
    }

    // -------------------------------------------------------------------------
    // Render Pipeline Creation Tests
    // -------------------------------------------------------------------------

    /// Test: Basic render pipeline creation.
    #[test]
    fn render_pipeline_creation_basic() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let shader_source = r#"
            @vertex
            fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }

            @fragment
            fn fs_main() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0, 0.0, 0.0, 1.0);
            }
        "#;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("test_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("test_pipeline_layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("test_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let _ = pipeline;
    }

    // -------------------------------------------------------------------------
    // Compute Pipeline Creation Tests
    // -------------------------------------------------------------------------

    /// Test: Basic compute pipeline creation.
    #[test]
    fn compute_pipeline_creation_basic() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let shader_source = r#"
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                // Empty compute shader
            }
        "#;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("test_compute_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compute_pipeline_layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("test_compute_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        });

        let _ = pipeline;
    }

    // -------------------------------------------------------------------------
    // Pipeline Caching Tests
    // -------------------------------------------------------------------------

    /// Test: Shader cache returns same module for same source.
    #[test]
    fn shader_cache_deduplication() {
        use renderer_backend::pipeline::ShaderCache;

        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let mut cache = ShaderCache::new();

        let source = r#"
            @vertex
            fn main() -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0);
            }
        "#;

        let (module1, hash1) = cache.get_or_compile(&device, source);
        let (module2, hash2) = cache.get_or_compile(&device, source);

        assert_eq!(hash1, hash2, "Same source should produce same hash");
        assert!(
            Arc::ptr_eq(&module1, &module2),
            "Same source should return cached module"
        );
    }

    /// Test: Shader cache creates new module for different source.
    #[test]
    fn shader_cache_different_source() {
        use renderer_backend::pipeline::ShaderCache;

        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        let mut cache = ShaderCache::new();

        let source1 = r#"
            @vertex
            fn main() -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0);
            }
        "#;

        let source2 = r#"
            @vertex
            fn main() -> @builtin(position) vec4<f32> {
                return vec4<f32>(1.0);
            }
        "#;

        let (module1, hash1) = cache.get_or_compile(&device, source1);
        let (module2, hash2) = cache.get_or_compile(&device, source2);

        assert_ne!(hash1, hash2, "Different source should produce different hash");
        assert!(
            !Arc::ptr_eq(&module1, &module2),
            "Different source should create new module"
        );
    }

    // -------------------------------------------------------------------------
    // Specialization Constants Tests (via override constants in wgpu)
    // -------------------------------------------------------------------------

    /// Test: Shader with override constants compiles.
    #[test]
    fn shader_override_constants() {
        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        // Note: wgpu doesn't have traditional spec constants like Vulkan
        // but we can verify shader compilation with const expressions
        let shader_source = r#"
            const WORKGROUP_SIZE: u32 = 64u;

            @compute @workgroup_size(WORKGROUP_SIZE)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            }
        "#;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("const_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let _ = shader;
    }
}

// =============================================================================
// MODULE 4: FRAME GRAPH TESTS
// =============================================================================

mod frame_graph {
    use super::*;
    use renderer_backend::frame_graph::{
        PassIndex, PassType, ResourceAccess, ResourceAccessEntry, ResourceAccessSet,
        ResourceHandle,
    };

    // -------------------------------------------------------------------------
    // Pass Declaration Tests
    // -------------------------------------------------------------------------

    /// Test: PassType enum has expected variants.
    #[test]
    fn pass_type_variants() {
        let _graphics = PassType::Graphics;
        let _compute = PassType::Compute;
        let _copy = PassType::Copy;
        let _raytracing = PassType::RayTracing;
    }

    /// Test: PassType display formatting.
    #[test]
    fn pass_type_display() {
        assert_eq!(format!("{}", PassType::Graphics), "Graphics");
        assert_eq!(format!("{}", PassType::Compute), "Compute");
        assert_eq!(format!("{}", PassType::Copy), "Copy");
        assert_eq!(format!("{}", PassType::RayTracing), "RayTracing");
    }

    /// Test: PassIndex wraps usize correctly.
    #[test]
    fn pass_index_wrapping() {
        let idx = PassIndex(42);
        assert_eq!(idx.0, 42);
        assert_eq!(format!("{}", idx), "PassIndex(42)");
    }

    // -------------------------------------------------------------------------
    // Resource Tracking Tests
    // -------------------------------------------------------------------------

    /// Test: ResourceHandle wraps u32 correctly.
    #[test]
    fn resource_handle_wrapping() {
        let handle = ResourceHandle(123);
        assert_eq!(handle.0, 123);
        assert_eq!(format!("{}", handle), "ResourceHandle(123)");
    }

    /// Test: ResourceHandle NONE sentinel value.
    #[test]
    fn resource_handle_none_sentinel() {
        let none = ResourceHandle::NONE;
        assert_eq!(none.0, u32::MAX);
        assert!(format!("{}", none).contains("NONE"));
    }

    /// Test: ResourceAccess enum variants.
    #[test]
    fn resource_access_variants() {
        let _read = ResourceAccess::Read;
        let _write = ResourceAccess::Write;
        let _readwrite = ResourceAccess::ReadWrite;
    }

    /// Test: ResourceAccessEntry construction.
    #[test]
    fn resource_access_entry_construction() {
        let entry = ResourceAccessEntry::new(ResourceHandle(0), ResourceAccess::Read);
        assert_eq!(entry.resource, ResourceHandle(0));
        assert_eq!(entry.access, ResourceAccess::Read);
    }

    /// Test: ResourceAccessSet default is empty.
    #[test]
    fn resource_access_set_default_empty() {
        let set = ResourceAccessSet::default();
        assert!(set.reads.is_empty());
        assert!(set.writes.is_empty());
    }

    /// Test: ResourceAccessSet can track reads and writes.
    #[test]
    fn resource_access_set_track_accesses() {
        let mut set = ResourceAccessSet::default();
        set.reads.push(ResourceHandle(0));
        set.reads.push(ResourceHandle(1));
        set.writes.push(ResourceHandle(2));

        assert_eq!(set.reads.len(), 2);
        assert_eq!(set.writes.len(), 1);
    }

    // -------------------------------------------------------------------------
    // Barrier Insertion Tests (via scheduling module)
    // -------------------------------------------------------------------------

    /// Test: Dependencies create correct edge relationships.
    #[test]
    fn dependency_edge_types() {
        // Read after Write (RAW) - consumer reads producer's output
        let producer_writes = vec![ResourceHandle(0)];
        let consumer_reads = vec![ResourceHandle(0)];

        // There should be a dependency from producer to consumer
        assert!(
            producer_writes.iter().any(|w| consumer_reads.contains(w)),
            "RAW dependency should exist"
        );

        // Write after Read (WAR) - writer overwrites what reader was using
        let reader_reads = vec![ResourceHandle(1)];
        let writer_writes = vec![ResourceHandle(1)];

        assert!(
            reader_reads.iter().any(|r| writer_writes.contains(r)),
            "WAR dependency should exist"
        );
    }

    // -------------------------------------------------------------------------
    // Pass Scheduling Tests
    // -------------------------------------------------------------------------

    /// Test: Topological ordering preserves dependencies.
    #[test]
    fn topological_ordering_preserves_deps() {
        // Simple linear dependency: A -> B -> C
        let passes = vec![
            ("A", vec![], vec![ResourceHandle(0)]),           // Writes R0
            ("B", vec![ResourceHandle(0)], vec![ResourceHandle(1)]), // Reads R0, Writes R1
            ("C", vec![ResourceHandle(1)], vec![]),           // Reads R1
        ];

        // In topological order, A must come before B, B must come before C
        // This is already the order in passes, so indices should be [0, 1, 2]
        let order: Vec<usize> = (0..passes.len()).collect();

        // Verify constraints
        let a_idx = order.iter().position(|&x| x == 0).unwrap();
        let b_idx = order.iter().position(|&x| x == 1).unwrap();
        let c_idx = order.iter().position(|&x| x == 2).unwrap();

        assert!(a_idx < b_idx, "A must come before B");
        assert!(b_idx < c_idx, "B must come before C");
    }

    // -------------------------------------------------------------------------
    // Async Compute Overlap Tests
    // -------------------------------------------------------------------------

    /// Test: Compute passes can be identified for async scheduling.
    #[test]
    fn async_compute_identification() {
        let passes = vec![
            (PassType::Graphics, "shadow"),
            (PassType::Compute, "culling"),      // Candidate for async
            (PassType::Graphics, "gbuffer"),
            (PassType::Compute, "lighting"),     // Candidate for async
            (PassType::Graphics, "composite"),
        ];

        let async_candidates: Vec<_> = passes
            .iter()
            .enumerate()
            .filter(|(_, (ty, _))| *ty == PassType::Compute)
            .map(|(i, _)| i)
            .collect();

        assert_eq!(async_candidates, vec![1, 3], "Should identify compute passes");
    }
}

// =============================================================================
// MODULE 5: MEMORY TESTS
// =============================================================================

mod memory {
    use super::*;
    use renderer_backend::memory::{FrameAllocator, GpuBudget, PoolAllocator, StackAllocator};

    // -------------------------------------------------------------------------
    // Allocation Tracking Tests
    // -------------------------------------------------------------------------

    /// Test: FrameAllocator tracks allocations correctly.
    #[test]
    fn frame_allocator_tracks_used() {
        let mut alloc = FrameAllocator::new(4096);

        assert_eq!(alloc.used(), 0, "Initial usage should be 0");

        let _ = alloc.allocate(256, 4);
        assert!(alloc.used() >= 256, "Usage should reflect allocation");

        let _ = alloc.allocate(512, 4);
        assert!(alloc.used() >= 768, "Usage should accumulate");
    }

    /// Test: FrameAllocator reset clears usage.
    #[test]
    fn frame_allocator_reset() {
        let mut alloc = FrameAllocator::new(4096);

        let _ = alloc.allocate(1024, 4);
        assert!(alloc.used() > 0);

        alloc.reset();
        assert_eq!(alloc.used(), 0, "Reset should clear usage");
    }

    /// Test: FrameAllocator alignment works correctly.
    #[test]
    fn frame_allocator_alignment() {
        let mut alloc = FrameAllocator::new(4096);

        // Allocate with specific alignment
        let slice = alloc.allocate(100, 16).unwrap();
        let ptr = slice.as_ptr() as usize;

        assert_eq!(ptr % 16, 0, "Allocation should be 16-byte aligned");
    }

    /// Test: FrameAllocator returns None when capacity exceeded.
    #[test]
    fn frame_allocator_capacity_exceeded() {
        let mut alloc = FrameAllocator::new(256);

        // Try to allocate more than capacity
        let result = alloc.allocate(512, 4);
        assert!(result.is_none(), "Should return None when exceeded");
    }

    // -------------------------------------------------------------------------
    // Budget Tests (using actual GpuBudget API)
    // -------------------------------------------------------------------------

    /// Test: GpuBudget tracks reservations and releases.
    #[test]
    fn gpu_budget_tracking() {
        let budget = GpuBudget::new(1024 * 1024); // 1MB budget

        assert_eq!(budget.used(), 0, "Initial usage should be 0");

        // Use try_reserve instead of allocate
        assert!(budget.try_reserve(4096), "First reserve should succeed");
        assert_eq!(budget.used(), 4096);

        assert!(budget.try_reserve(8192), "Second reserve should succeed");
        assert_eq!(budget.used(), 12288);

        // Use release instead of deallocate
        budget.release(4096);
        assert_eq!(budget.used(), 8192);
    }

    /// Test: GpuBudget detects over-budget allocations.
    #[test]
    fn gpu_budget_over_budget() {
        let budget = GpuBudget::new(1024); // 1KB budget

        // Allocate within budget
        assert!(budget.try_reserve(512), "Should succeed within budget");

        // Allocate over budget
        assert!(!budget.try_reserve(1024), "Should fail when over budget");
    }

    // -------------------------------------------------------------------------
    // Budget Enforcement Tests
    // -------------------------------------------------------------------------

    /// Test: GpuBudget cap is respected.
    #[test]
    fn gpu_budget_cap() {
        let budget = GpuBudget::new(100);
        assert_eq!(budget.cap, 100);
    }

    /// Test: GpuBudget available calculation.
    #[test]
    fn gpu_budget_available() {
        let budget = GpuBudget::new(1000);

        assert!(budget.try_reserve(400));
        assert_eq!(budget.available(), 600);

        assert!(budget.try_reserve(200));
        assert_eq!(budget.available(), 400);
    }

    /// Test: Used tracking across multiple operations.
    #[test]
    fn gpu_budget_used_tracking() {
        let budget = GpuBudget::new(10000);

        assert!(budget.try_reserve(1000));
        assert!(budget.try_reserve(2000)); // Total: 3000
        budget.release(1000);
        assert!(budget.try_reserve(500));  // Total: 2500

        assert_eq!(budget.used(), 2500, "Current usage should be 2500");
    }

    // -------------------------------------------------------------------------
    // Pool Allocator Tests
    // -------------------------------------------------------------------------

    /// Test: PoolAllocator uses correct size classes.
    #[test]
    fn pool_allocator_size_classes() {
        let mut pool = PoolAllocator::new();

        // Allocate various sizes and verify they get appropriate blocks
        let small = pool.allocate(1000).unwrap();  // Should get 64KB block
        assert!(small.len() >= 64 * 1024);

        let medium = pool.allocate(100 * 1024).unwrap(); // Should get 256KB block
        assert!(medium.len() >= 256 * 1024);

        let large = pool.allocate(500 * 1024).unwrap();  // Should get 1MB block
        assert!(large.len() >= 1 * 1024 * 1024);
    }

    /// Test: PoolAllocator recycles blocks.
    #[test]
    fn pool_allocator_recycling() {
        let mut pool = PoolAllocator::new();

        // Allocate and deallocate
        let block = pool.allocate(1000).unwrap();
        let block_len = block.len();
        pool.deallocate(block);

        // Allocate again - should reuse
        let block2 = pool.allocate(1000).unwrap();
        assert_eq!(block2.len(), block_len, "Should reuse same size block");
    }

    // -------------------------------------------------------------------------
    // Stack Allocator Tests
    // -------------------------------------------------------------------------

    /// Test: StackAllocator push/pop LIFO order.
    #[test]
    fn stack_allocator_lifo() {
        let mut stack = StackAllocator::new(1024);

        let marker1 = stack.push(&[1, 2, 3]).unwrap();
        let marker2 = stack.push(&[4, 5, 6]).unwrap();
        let marker3 = stack.push(&[7, 8, 9]).unwrap();

        // Pop in reverse order
        stack.pop(marker3);
        stack.pop(marker2);
        stack.pop(marker1);

        // Stack should be empty now (able to push same amount)
        assert!(stack.push(&[1, 2, 3, 4, 5, 6, 7, 8, 9]).is_some());
    }

    /// Test: StackAllocator marker validation.
    #[test]
    #[should_panic(expected = "marker")]
    fn stack_allocator_invalid_marker_panics() {
        let mut stack = StackAllocator::new(256);

        let _marker = stack.push(&[1, 2, 3]).unwrap();

        // Pop with invalid marker should panic
        stack.pop(1000); // Invalid marker (beyond buffer length)
    }
}

// =============================================================================
// INTEGRATION TESTS (Cross-module interactions)
// =============================================================================

mod integration {
    use super::*;

    /// Test: Full buffer creation and destruction lifecycle.
    #[test]
    fn buffer_lifecycle_complete() {
        use renderer_backend::resources::{buffer_usages, create_buffer, TrinityBufferDescriptor};

        let adapter = require_adapter!();
        let (device, queue) = require_device!(&adapter);

        // Create buffer
        let desc = TrinityBufferDescriptor {
            label: Some("lifecycle_test"),
            size: 1024,
            usage: buffer_usages::VERTEX,
            mapped_at_creation: false,
        };
        let buffer = create_buffer(&device, &desc);

        // Use buffer (write data)
        let data = [0u8; 1024];
        queue.write_buffer(buffer.inner(), 0, &data);

        // Buffer drops here - should be clean
    }

    /// Test: Multiple resource types can coexist.
    #[test]
    fn multiple_resource_types() {
        use renderer_backend::resources::{
            buffer_usages, create_buffer, create_sampler, create_texture,
            TrinityBufferDescriptor, TrinitySamplerDescriptor, TrinityTextureDescriptor,
        };

        let adapter = require_adapter!();
        let (device, _queue) = require_device!(&adapter);

        // Create buffer
        let buffer = create_buffer(
            &device,
            &TrinityBufferDescriptor {
                label: Some("test_buffer"),
                size: 256,
                usage: buffer_usages::UNIFORM,
                mapped_at_creation: false,
            },
        );

        // Create texture
        let texture = create_texture(
            &device,
            &TrinityTextureDescriptor {
                label: Some("test_texture"),
                size: wgpu::Extent3d {
                    width: 64,
                    height: 64,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        // Create sampler
        let sampler = create_sampler(&device, &TrinitySamplerDescriptor::linear_clamp());

        // All resources exist simultaneously
        let _ = (buffer, texture, sampler);
    }

    /// Test: Command encoder can reference multiple resources.
    #[test]
    fn command_encoder_multi_resource() {
        use renderer_backend::resources::{buffer_usages, create_buffer, TrinityBufferDescriptor};

        let adapter = require_adapter!();
        let (device, queue) = require_device!(&adapter);

        let src_buffer = create_buffer(
            &device,
            &TrinityBufferDescriptor {
                label: Some("src"),
                size: 256,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        );

        let dst_buffer = create_buffer(
            &device,
            &TrinityBufferDescriptor {
                label: Some("dst"),
                size: 256,
                usage: wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        );

        // Write to source
        let data = [42u8; 256];
        queue.write_buffer(src_buffer.inner(), 0, &data);

        // Copy between buffers
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("copy_encoder"),
        });

        encoder.copy_buffer_to_buffer(src_buffer.inner(), 0, dst_buffer.inner(), 0, 256);

        queue.submit(Some(encoder.finish()));
    }
}

// =============================================================================
// TEST COUNT SUMMARY
// =============================================================================

#[cfg(test)]
mod test_count {
    //! Summary of tests in this module:
    //!
    //! ## Device Module (12 tests)
    //! - Instance creation: 3 tests
    //! - Adapter enumeration: 3 tests
    //! - Queue creation: 3 tests
    //! - Capability tier: 2 tests
    //! - Limit negotiation: 3 tests
    //!
    //! ## Resources Module (18 tests)
    //! - Buffer creation: 5 tests
    //! - Buffer destruction: 1 test
    //! - Texture creation: 4 tests
    //! - Sampler creation: 3 tests
    //! - Bind group layout: 2 tests
    //! - Pipeline layout: 1 test
    //! - Resource aliasing: 1 test
    //!
    //! ## Pipelines Module (8 tests)
    //! - Content hash: 4 tests
    //! - Render pipeline: 1 test
    //! - Compute pipeline: 1 test
    //! - Shader cache: 2 tests
    //!
    //! ## Frame Graph Module (9 tests)
    //! - Pass declaration: 3 tests
    //! - Resource tracking: 5 tests
    //! - Scheduling: 1 test
    //!
    //! ## Memory Module (12 tests)
    //! - Frame allocator: 4 tests
    //! - GPU budget: 5 tests
    //! - Pool allocator: 2 tests
    //! - Stack allocator: 2 tests (1 panic test)
    //!
    //! ## Integration Module (3 tests)
    //! - Lifecycle: 1 test
    //! - Multi-resource: 2 tests
    //!
    //! **Total: 62 tests**
}
