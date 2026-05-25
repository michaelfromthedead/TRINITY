//! Multi-frame stress test (1000 frames) and device lost recovery tests.
//!
//! These are Rust-native integration tests beyond the existing Python suite.

mod common;

use common::*;

// =========================================================================
// Multi-frame stress test
//
// Simulates a 1000-frame render loop cycling through:
//   - Buffer and texture creation/usage
//   - Swapchain acquire + present cycle
//   - Command list recording + queue submission
//   - Resource tracking to verify no leaks
// =========================================================================

/// Number of frames for the stress test.
const STRESS_FRAME_COUNT: u32 = 1000;

/// Verify that a later handle exceeds an earlier one (monotonic).
/// Uses a relaxed delta because tests run in parallel within the same binary
/// and share global atomic handle counters.
fn assert_handle_advanced(initial: u64, final_h: u64) {
    assert!(
        final_h > initial,
        "Handle did not advance: initial={} final={}",
        initial,
        final_h
    );
}

#[test]
fn test_multi_frame_stress_swapchain_present_cycle() {
    // This test focuses on the swapchain present+acquire cycle
    let desc = SwapchainDesc {
        width: 1280,
        height: 720,
        format: Format::RGBA8Unorm,
        buffer_count: 3,
        present_mode: PresentMode::Vsync,
        color_space: ColorSpace::Srgb,
    };
    let sc = MockSwapchain::new(desc);

    for frame in 0..STRESS_FRAME_COUNT {
        let _texture = sc.current_texture();
        assert!(_texture.is_valid(),
            "Frame {}: current texture invalid", frame);

        sc.present();
    }

    // After 1000 frames, the index should be (1000 % 3) = 1
    assert_eq!(sc.current_index(), STRESS_FRAME_COUNT % 3,
        "Final swapchain index after {} presents", STRESS_FRAME_COUNT);
}

#[test]
fn test_multi_frame_stress_buffer_create_destroy() {
    let device = create_test_device();
    let base_desc = BufferDesc {
        size: 1024,
        usage: BufferUsage::VERTEX | BufferUsage::COPY_DST,
        memory_type: MemoryType::Default,
        stride: 32,
    };

    let first_handle;
    let last_handle;

    {
        let first = device.create_buffer(base_desc.clone());
        first_handle = first.handle();
    }

    // Create and destroy buffers across 1000 frames
    for frame in 0..STRESS_FRAME_COUNT {
        let mut buffer = device.create_buffer(BufferDesc {
            size: 64 + (frame as u64) * 16,
            usage: BufferUsage::STORAGE,
            memory_type: MemoryType::Default,
            stride: 0,
        });

        // Verify each frame's buffer is initially valid
        assert!(buffer.is_valid(),
            "Frame {}: buffer invalid on creation", frame);
        assert_eq!(buffer.desc().size, 64 + (frame as u64) * 16,
            "Frame {}: size mismatch", frame);

        // Explicitly destroy to simulate per-frame resource cleanup
        buffer.destroy();
        assert!(!buffer.is_valid(),
            "Frame {}: buffer still valid after destroy", frame);
    }

    // Create one more to get the final handle
    {
        let last = device.create_buffer(base_desc);
        last_handle = last.handle();
    }

    assert_handle_advanced(first_handle, last_handle);
}

#[test]
fn test_multi_frame_stress_texture_create_destroy() {
    let device = create_test_device();
    let base_desc = TextureDesc {
        ty: TextureType::Texture2D,
        format: Format::RGBA8Unorm,
        width: 256,
        height: 256,
        depth: 1,
        mip_levels: 1,
        array_size: 1,
        sample_count: SampleCount::X1,
        usage: TextureUsage::SHADER_RESOURCE,
    };

    let first_handle = {
        let first = device.create_texture(base_desc.clone());
        first.handle()
    };

    for frame in 0..STRESS_FRAME_COUNT {
        let mut tex = device.create_texture(TextureDesc {
            width: 64 + (frame % 128),
            height: 64 + (frame % 128),
            ..base_desc.clone()
        });

        assert!(tex.is_valid(),
            "Frame {}: texture invalid on creation", frame);
        tex.destroy();
        assert!(!tex.is_valid(),
            "Frame {}: texture still valid after destroy", frame);
    }

    let last_handle = {
        let last = device.create_texture(base_desc);
        last.handle()
    };

    assert_handle_advanced(first_handle, last_handle);
}

#[test]
fn test_multi_frame_stress_command_recording() {
    let device = create_test_device();
    let pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
        vertex_shader: Some(ShaderDesc {
            stage: ShaderStage::Vertex,
            source: b"vs".to_vec(),
            entry_point: "main".into(),
        }),
        pixel_shader: Some(ShaderDesc {
            stage: ShaderStage::Pixel,
            source: b"ps".to_vec(),
            entry_point: "main".into(),
        }),
        ..Default::default()
    });

    // Simulate 1000 frames of command recording and queue submission
    let queue = device.get_queue(QueueType::Graphics);
    let fence = MockFence::new(0);

    for _ in 0..STRESS_FRAME_COUNT {
        let rt = device.create_texture(TextureDesc {
            ty: TextureType::Texture2D,
            format: Format::RGBA8Unorm,
            width: 1280,
            height: 720,
            depth: 1,
            mip_levels: 1,
            array_size: 1,
            sample_count: SampleCount::X1,
            usage: TextureUsage::RENDER_TARGET | TextureUsage::SHADER_RESOURCE,
        });

        let vb = device.create_buffer(BufferDesc {
            size: 4096,
            usage: BufferUsage::VERTEX,
            memory_type: MemoryType::Default,
            stride: 32,
        });

        let mut cmd = MockCommandList::new();
        cmd.begin();
        cmd.barrier(rt.handle(), ResourceState::Undefined, ResourceState::RenderTarget);
        cmd.begin_render_pass(&[rt.handle()], None, Some((0.0, 0.0, 0.0, 1.0)), None);
        cmd.set_pipeline(pipeline.handle());
        cmd.set_viewport(0.0, 0.0, 1280.0, 720.0, 0.0, 1.0);
        cmd.set_scissor(0, 0, 1280, 720);
        cmd.set_vertex_buffer(0, vb.handle(), 0, 32);
        cmd.draw(3, 1, 0, 0);
        cmd.end_render_pass();
        cmd.barrier(rt.handle(), ResourceState::RenderTarget, ResourceState::Present);
        cmd.end();

        let cmds = cmd.recorded_commands();
        assert_eq!(cmds.len(), 9,
            "Expected 9 commands per frame (got {}): {:?}", cmds.len(),
            cmds.iter().map(|c| c.cmd_type.clone()).collect::<Vec<_>>());
    }

    queue.submit_with_fence(&[], &fence);
    assert!(fence.value() > 0,
        "Fence should have been signaled after stress submission");
}

#[test]
fn test_multi_frame_stress_full_pipeline() {
    // Combined stress: swapchain + queue + command recording
    let device = create_test_device();
    let sc_desc = SwapchainDesc {
        width: 1280,
        height: 720,
        format: Format::RGBA8Unorm,
        buffer_count: 3,
        present_mode: PresentMode::Vsync,
        color_space: ColorSpace::Srgb,
    };
    let sc = MockSwapchain::new(sc_desc);
    let queue = device.get_queue(QueueType::Graphics);

    // Pre-create a pipeline to reuse across frames
    let pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
        vertex_shader: Some(ShaderDesc {
            stage: ShaderStage::Vertex,
            source: b"vs".to_vec(),
            entry_point: "main".into(),
        }),
        pixel_shader: Some(ShaderDesc {
            stage: ShaderStage::Pixel,
            source: b"ps".to_vec(),
            entry_point: "main".into(),
        }),
        ..Default::default()
    });

    for frame in 0..STRESS_FRAME_COUNT {
        // Acquire
        let backbuffer = sc.current_texture();
        assert!(backbuffer.is_valid());

        // Record
        let mut cmd = MockCommandList::new();
        cmd.begin();
        cmd.barrier(
            backbuffer.handle(),
            ResourceState::Present,
            ResourceState::RenderTarget,
        );
        cmd.begin_render_pass(&[backbuffer.handle()], None, Some((0.0, 0.0, 0.0, 1.0)), None);
        cmd.set_pipeline(pipeline.handle());
        cmd.set_viewport(0.0, 0.0, 1280.0, 720.0, 0.0, 1.0);
        cmd.set_scissor(0, 0, 1280, 720);
        cmd.draw(3, 1, 0, 0);
        cmd.end_render_pass();
        cmd.barrier(
            backbuffer.handle(),
            ResourceState::RenderTarget,
            ResourceState::Present,
        );
        cmd.end();

        // Submit
        queue.submit(&[cmd]);

        // Present
        sc.present();

        // Verify frame invariants
        if frame % 100 == 0 {
            // Every 100 frames, verify queue accumulated properly
            assert_eq!(queue.submitted_count(), (frame + 1) as usize,
                "Queue submission count mismatch at frame {}", frame);
        }
    }

    // Verify final state
    assert_eq!(queue.submitted_count(), STRESS_FRAME_COUNT as usize,
        "All {} frames should have been submitted", STRESS_FRAME_COUNT);
    assert_eq!(sc.current_index(), STRESS_FRAME_COUNT % 3,
        "Final swapchain index");
}

// =========================================================================
// Device lost recovery tests
// =========================================================================

#[test]
fn test_device_shutdown_recovery() {
    // Simulate: create device -> work -> shutdown -> recreate
    let adapter = MockAdapter::new(AdapterType::Discrete);

    let mut device = MockDevice::create_with_config(adapter.clone(), DeviceConfig {
        enable_debug: true,
        enable_validation: false,
    });

    // Do some work
    let _buf = device.create_buffer(BufferDesc {
        size: 1024,
        usage: BufferUsage::VERTEX,
        memory_type: MemoryType::Default,
        stride: 32,
    });

    // Shutdown
    device.shutdown();
    assert!(device.is_shutdown());

    // Recreate (recovery path)
    let device2 = MockDevice::create_with_config(adapter, DeviceConfig {
        enable_debug: false,
        enable_validation: true,
    });
    assert!(!device2.is_shutdown());

    // New device should be functional
    let _buf2 = device2.create_buffer(BufferDesc {
        size: 2048,
        usage: BufferUsage::STORAGE,
        memory_type: MemoryType::Default,
        stride: 0,
    });
    let queue = device2.get_queue(QueueType::Graphics);
    assert_eq!(queue.queue_type(), QueueType::Graphics);
}

#[test]
fn test_device_recovery_after_resource_leak() {
    // Simulate device lost state and verify recovery:
    // create many resources -> simulate device lost -> create new device -> verify new device works
    let adapter = MockAdapter::new(AdapterType::Software);
    let mut device = MockDevice::create(adapter.clone());

    // Create resources
    let _buffers: Vec<_> = (0..100)
        .map(|i| {
            device.create_buffer(BufferDesc {
                size: 1024 + i,
                usage: BufferUsage::STORAGE,
                memory_type: MemoryType::Default,
                stride: 0,
            })
        })
        .collect();

    assert!(!device.is_shutdown());

    // Device lost: shutdown
    device.shutdown();
    assert!(device.is_shutdown());

    // Recovery: create new device
    let new_device = MockDevice::create(adapter);
    assert!(!new_device.is_shutdown());

    // New device should allocate fresh resources
    let new_buf = new_device.create_buffer(BufferDesc {
        size: 512,
        usage: BufferUsage::VERTEX | BufferUsage::INDEX,
        memory_type: MemoryType::Default,
        stride: 16,
    });
    assert!(new_buf.is_valid());
    assert_eq!(new_buf.desc().size, 512);
}

#[test]
fn test_device_graceful_shutdown_with_active_queues() {
    let mut device = create_test_device();

    // Get queues and submit work
    let gfx = device.get_queue(QueueType::Graphics);
    let compute = device.get_queue(QueueType::Compute);

    let mut cmd = MockCommandList::new();
    cmd.begin();
    cmd.dispatch(1, 1, 1);
    cmd.end();
    gfx.submit(&[cmd]);

    assert!(!gfx.is_shutdown());
    assert!(!compute.is_shutdown());

    // Shutdown should clean up all queues
    device.shutdown();

    assert!(device.is_shutdown());

    // Verify queues were shut down through device
    assert!(gfx.is_shutdown());
    assert!(compute.is_shutdown());
}

#[test]
fn test_device_lost_during_frame() {
    // Simulate: begin frame -> device lost -> recover -> continue
    let adapter = MockAdapter::new(AdapterType::Discrete);
    let mut device = MockDevice::create(adapter.clone());

    // Frame N: normal operation
    {
        let _rt = device.create_texture(TextureDesc {
            ty: TextureType::Texture2D,
            format: Format::RGBA8Unorm,
            width: 1280,
            height: 720,
            depth: 1,
            mip_levels: 1,
            array_size: 1,
            sample_count: SampleCount::X1,
            usage: TextureUsage::RENDER_TARGET,
        });
        let _ = device.get_queue(QueueType::Graphics);
    }

    // Device lost
    device.shutdown();
    assert!(device.is_shutdown());

    // Recover
    let device2 = MockDevice::create_with_config(adapter, DeviceConfig {
        enable_debug: false,
        enable_validation: false,
    });

    // Frame N+1: should work on new device
    let rt2 = device2.create_texture(TextureDesc {
        ty: TextureType::Texture2D,
        format: Format::RGBA8Unorm,
        width: 1920,
        height: 1080,
        depth: 1,
        mip_levels: 1,
        array_size: 1,
        sample_count: SampleCount::X1,
        usage: TextureUsage::RENDER_TARGET | TextureUsage::SHADER_RESOURCE,
    });
    assert!(rt2.is_valid());
    assert_eq!(rt2.desc().width, 1920);

    let queue2 = device2.get_queue(QueueType::Graphics);
    let mut cmd = MockCommandList::new();
    cmd.begin();
    cmd.barrier(rt2.handle(), ResourceState::Undefined, ResourceState::RenderTarget);
    cmd.begin_render_pass(&[rt2.handle()], None, Some((0.0, 0.0, 0.0, 1.0)), None);
    cmd.end_render_pass();
    cmd.end();
    queue2.submit(&[cmd]);

    assert_eq!(queue2.submitted_count(), 1,
        "Recovered device should process commands");
}

#[test]
fn test_device_recovery_preserves_adapter_properties() {
    // Verify that after recovery, the new device reports same adapter properties
    let adapter = MockAdapter::new(AdapterType::Discrete);
    let features_before = adapter.query_features();

    let mut device = MockDevice::create(adapter.clone());
    device.shutdown();

    // Recreate with same adapter
    let _device2 = MockDevice::create(adapter.clone());

    // Adapter properties should be preserved (adapter is separate from device)
    let features_after = adapter.query_features();
    assert_eq!(features_before.ray_tracing, features_after.ray_tracing);
    assert_eq!(features_before.compute, features_after.compute);
    assert_eq!(features_before.max_texture_size, features_after.max_texture_size);
}

#[test]
fn test_multiple_device_create_destroy_cycles() {
    for cycle in 0..10 {
        let mut device = create_test_device();

        // Do work
        let _buf = device.create_buffer(BufferDesc {
            size: 1024,
            usage: BufferUsage::VERTEX,
            memory_type: MemoryType::Default,
            stride: 32,
        });
        let _tex = device.create_texture(TextureDesc {
            ty: TextureType::Texture2D,
            format: Format::RGBA8Unorm,
            width: 256,
            height: 256,
            depth: 1,
            mip_levels: 1,
            array_size: 1,
            sample_count: SampleCount::X1,
            usage: TextureUsage::SHADER_RESOURCE,
        });

        assert!(!device.is_shutdown());

        // Destroy
        device.shutdown();
        assert!(device.is_shutdown(),
            "Device should be shut down after cycle {}", cycle);
    }
}
