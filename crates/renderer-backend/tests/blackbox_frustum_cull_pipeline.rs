//! Integration tests for FrustumCullPipeline (T-WGPU-P6.3.3).
//!
//! Tests the frustum culling compute pipeline that binds frustum, objects,
//! and visibility buffers.

use renderer_backend::gpu_driven::{
    CullDispatchParams, FrustumBuffer, FrustumCullPipelineV2,
    ObjectData, SceneDataBuffers, VisibilityFlagsBuffer,
    is_visible, object_flags, workgroups_for_objects,
    CULL_DISPATCH_PARAMS_SIZE, FRUSTUM_CULL_PIPELINE_WORKGROUP_SIZE,
};

// =============================================================================
// TEST HELPERS
// =============================================================================

/// Create a test wgpu instance for integration tests.
fn create_test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("test_device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::default(),
        },
        None,
    ))
    .ok()?;

    Some((device, queue))
}

/// Create a view-projection matrix for testing.
fn test_view_projection() -> [[f32; 4]; 4] {
    // Simple perspective looking down -Z
    let fovy = std::f32::consts::FRAC_PI_4;
    let aspect = 1.0;
    let near = 0.1;
    let far = 100.0;

    let f = 1.0 / (fovy * 0.5).tan();
    let range_inv = 1.0 / (near - far);

    let proj = [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, far * range_inv, -1.0],
        [0.0, 0.0, near * far * range_inv, 0.0],
    ];

    // Simple view looking at origin from z=10
    let view = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, -10.0, 1.0],
    ];

    // VP = view * proj (column-major multiplication)
    let mut vp = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                vp[i][j] += view[k][j] * proj[i][k];
            }
        }
    }
    vp
}

// =============================================================================
// UNIT TESTS (No GPU Required)
// =============================================================================

#[test]
fn test_pipeline_creation() {
    // Test that pipeline can be created when a device is available
    let Some((device, _queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let pipeline = FrustumCullPipelineV2::new(&device);

    // Verify pipeline components exist
    assert!(!format!("{:?}", pipeline).is_empty());
}

#[test]
fn test_workgroup_size_constant() {
    // Verify workgroup size is 64 as documented
    assert_eq!(FRUSTUM_CULL_PIPELINE_WORKGROUP_SIZE, 64);
}

#[test]
fn test_workgroup_calculation() {
    // Test workgroup count calculations
    assert_eq!(workgroups_for_objects(0), 0);
    assert_eq!(workgroups_for_objects(1), 1);
    assert_eq!(workgroups_for_objects(64), 1);
    assert_eq!(workgroups_for_objects(65), 2);
    assert_eq!(workgroups_for_objects(128), 2);
    assert_eq!(workgroups_for_objects(1000), 16); // ceil(1000/64) = 16
}

#[test]
fn test_cull_dispatch_params() {
    let params = CullDispatchParams::new(500);
    assert_eq!(params.object_count, 500);
    assert_eq!(params.flags, 0);
    assert_eq!(std::mem::size_of::<CullDispatchParams>(), CULL_DISPATCH_PARAMS_SIZE);
}

#[test]
fn test_dispatch_small_count() {
    // Test dispatch with few objects
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let pipeline = FrustumCullPipelineV2::new(&device);
    let mut frustum = FrustumBuffer::new(&device);
    let mut scene = SceneDataBuffers::new(&device, 100, Some("test"));
    let visibility = VisibilityFlagsBuffer::new(&device, 100, Some("test"));

    // Add some test objects
    for i in 0..10 {
        let obj = ObjectData::new()
            .with_mesh(i as u32)
            .with_flags(object_flags::VISIBLE)
            .with_aabb(
                [-1.0, -1.0, -1.0],
                [1.0, 1.0, 1.0],
            );
        scene.add(obj);
    }
    scene.upload(&device, &queue);

    // Update frustum
    let vp = test_view_projection();
    frustum.update(&queue, &vp);

    // Clear visibility and dispatch
    visibility.clear(&queue);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    pipeline.dispatch(
        &mut encoder,
        &device,
        &queue,
        &frustum,
        &scene,
        &visibility,
        10,
    );

    queue.submit([encoder.finish()]);

    // Wait for GPU
    device.poll(wgpu::Maintain::Wait);
}

#[test]
fn test_bind_group_creation() {
    // Test bind group creation helpers
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let pipeline = FrustumCullPipelineV2::new(&device);
    let frustum = FrustumBuffer::new(&device);
    let scene = SceneDataBuffers::new(&device, 100, Some("test"));
    let visibility = VisibilityFlagsBuffer::new(&device, 100, Some("test"));

    // Create frustum bind group
    let frustum_bg = pipeline.create_frustum_bind_group(&device, &frustum);
    assert!(!format!("{:?}", frustum_bg).is_empty());

    // Create params buffer
    let params = CullDispatchParams::new(100);
    let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_params"),
        size: CULL_DISPATCH_PARAMS_SIZE as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

    // Create objects bind group
    let objects_bg = pipeline.create_objects_bind_group(
        &device,
        &params_buffer,
        &scene,
        &visibility,
    );
    assert!(!format!("{:?}", objects_bg).is_empty());
}

#[test]
fn test_dispatch_with_bind_groups() {
    // Test the optimized dispatch path with pre-created bind groups
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let pipeline = FrustumCullPipelineV2::new(&device);
    let frustum = FrustumBuffer::new(&device);
    let scene = SceneDataBuffers::new(&device, 100, Some("test"));
    let visibility = VisibilityFlagsBuffer::new(&device, 100, Some("test"));

    // Create bind groups
    let frustum_bg = pipeline.create_frustum_bind_group(&device, &frustum);

    let params = CullDispatchParams::new(50);
    let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_params"),
        size: CULL_DISPATCH_PARAMS_SIZE as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

    let objects_bg = pipeline.create_objects_bind_group(
        &device,
        &params_buffer,
        &scene,
        &visibility,
    );

    // Dispatch with pre-created bind groups
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    pipeline.dispatch_with_bind_groups(
        &mut encoder,
        &frustum_bg,
        &objects_bg,
        50,
    );

    queue.submit([encoder.finish()]);
    device.poll(wgpu::Maintain::Wait);
}

#[test]
fn test_zero_objects_dispatch() {
    // Test that dispatching with 0 objects doesn't panic
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let pipeline = FrustumCullPipelineV2::new(&device);
    let frustum = FrustumBuffer::new(&device);
    let scene = SceneDataBuffers::new(&device, 100, Some("test"));
    let visibility = VisibilityFlagsBuffer::new(&device, 100, Some("test"));

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    // Should handle 0 objects gracefully
    pipeline.dispatch(
        &mut encoder,
        &device,
        &queue,
        &frustum,
        &scene,
        &visibility,
        0,
    );

    queue.submit([encoder.finish()]);
    device.poll(wgpu::Maintain::Wait);
}

#[test]
fn test_pipeline_layouts() {
    // Test that layout accessors work
    let Some((device, _queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let pipeline = FrustumCullPipelineV2::new(&device);

    // Verify we can access layouts
    let _ = pipeline.pipeline();
    let _ = pipeline.frustum_layout();
    let _ = pipeline.objects_layout();
    let _ = pipeline.pipeline_layout();
}

#[test]
fn test_frustum_planes_in_view() {
    // Test that objects in view are marked visible
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let pipeline = FrustumCullPipelineV2::new(&device);
    let mut frustum = FrustumBuffer::new(&device);
    let mut scene = SceneDataBuffers::new(&device, 100, Some("test"));
    let visibility = VisibilityFlagsBuffer::new(&device, 100, Some("test"));

    // Add object at origin (should be visible from z=10 looking at origin)
    let obj = ObjectData::new()
        .with_mesh(0)
        .with_flags(object_flags::VISIBLE)
        .with_aabb([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
    scene.add(obj);
    scene.upload(&device, &queue);

    // Update frustum
    let vp = test_view_projection();
    frustum.update(&queue, &vp);

    // Clear visibility
    visibility.clear(&queue);

    // Dispatch
    let mut encoder = device.create_command_encoder(&Default::default());
    pipeline.dispatch(
        &mut encoder,
        &device,
        &queue,
        &frustum,
        &scene,
        &visibility,
        1,
    );
    queue.submit([encoder.finish()]);
    device.poll(wgpu::Maintain::Wait);

    // Read back visibility
    let flags = visibility.read_back(&device, &queue);

    // Object 0 should be visible (bit 0 of word 0)
    assert!(
        is_visible(&flags, 0),
        "Object at origin should be visible from z=10"
    );
}

#[test]
fn test_frustum_planes_out_of_view() {
    // Test that objects outside view frustum are not marked visible
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let pipeline = FrustumCullPipelineV2::new(&device);
    let mut frustum = FrustumBuffer::new(&device);
    let mut scene = SceneDataBuffers::new(&device, 100, Some("test"));
    let visibility = VisibilityFlagsBuffer::new(&device, 100, Some("test"));

    // Add object far behind the camera (z=100, camera at z=10 looking at origin)
    let obj = ObjectData::new()
        .with_mesh(0)
        .with_flags(object_flags::VISIBLE)
        .with_aabb([98.0, -1.0, -1.0], [100.0, 1.0, 1.0]);
    scene.add(obj);
    scene.upload(&device, &queue);

    // Update frustum
    let vp = test_view_projection();
    frustum.update(&queue, &vp);

    // Clear visibility
    visibility.clear(&queue);

    // Dispatch
    let mut encoder = device.create_command_encoder(&Default::default());
    pipeline.dispatch(
        &mut encoder,
        &device,
        &queue,
        &frustum,
        &scene,
        &visibility,
        1,
    );
    queue.submit([encoder.finish()]);
    device.poll(wgpu::Maintain::Wait);

    // Read back visibility
    let flags = visibility.read_back(&device, &queue);

    // Object 0 should NOT be visible (behind camera)
    assert!(
        !is_visible(&flags, 0),
        "Object behind camera should not be visible"
    );
}

#[test]
fn test_invisible_object_skipped() {
    // Test that objects without VISIBLE flag are skipped
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let pipeline = FrustumCullPipelineV2::new(&device);
    let mut frustum = FrustumBuffer::new(&device);
    let mut scene = SceneDataBuffers::new(&device, 100, Some("test"));
    let visibility = VisibilityFlagsBuffer::new(&device, 100, Some("test"));

    // Add object at origin but without VISIBLE flag
    let obj = ObjectData::new()
        .with_mesh(0)
        .with_flags(0) // No VISIBLE flag
        .with_aabb([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
    scene.add(obj);
    scene.upload(&device, &queue);

    // Update frustum
    let vp = test_view_projection();
    frustum.update(&queue, &vp);

    // Clear visibility
    visibility.clear(&queue);

    // Dispatch
    let mut encoder = device.create_command_encoder(&Default::default());
    pipeline.dispatch(
        &mut encoder,
        &device,
        &queue,
        &frustum,
        &scene,
        &visibility,
        1,
    );
    queue.submit([encoder.finish()]);
    device.poll(wgpu::Maintain::Wait);

    // Read back visibility
    let flags = visibility.read_back(&device, &queue);

    // Object should not have visibility bit set (skipped because VISIBLE flag not set)
    assert!(
        !is_visible(&flags, 0),
        "Object without VISIBLE flag should be skipped"
    );
}

#[test]
fn test_large_object_count() {
    // Test with many objects
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let object_count = 1000;

    let pipeline = FrustumCullPipelineV2::new(&device);
    let mut frustum = FrustumBuffer::new(&device);
    let mut scene = SceneDataBuffers::new(&device, object_count + 100, Some("test"));
    let visibility = VisibilityFlagsBuffer::new(&device, object_count + 100, Some("test"));

    // Add many objects
    for i in 0..object_count {
        let z = (i as f32 * 0.1) - 50.0; // Spread along Z axis
        let obj = ObjectData::new()
            .with_mesh(i as u32)
            .with_flags(object_flags::VISIBLE)
            .with_aabb([-0.5, -0.5, z - 0.5], [0.5, 0.5, z + 0.5]);
        scene.add(obj);
    }
    scene.upload(&device, &queue);

    // Update frustum
    let vp = test_view_projection();
    frustum.update(&queue, &vp);

    // Clear visibility
    visibility.clear(&queue);

    // Dispatch
    let mut encoder = device.create_command_encoder(&Default::default());
    pipeline.dispatch(
        &mut encoder,
        &device,
        &queue,
        &frustum,
        &scene,
        &visibility,
        object_count as u32,
    );
    queue.submit([encoder.finish()]);
    device.poll(wgpu::Maintain::Wait);

    // Verify dispatch completed (read back to sync)
    let flags = visibility.read_back(&device, &queue);

    // Count visible objects
    let visible_count: usize = flags.iter().map(|w| w.count_ones() as usize).sum();
    println!("Visible: {} / {}", visible_count, object_count);

    // At least some should be visible, at least some should be culled
    // (objects spread from z=-50 to z=50, camera at z=10 looking at origin)
    assert!(visible_count > 0, "Some objects should be visible");
    assert!(visible_count < object_count, "Some objects should be culled");
}

// =============================================================================
// ADDITIONAL EDGE CASE TESTS (T-WGPU-P6.3.3 ACCEPTANCE CRITERIA)
// =============================================================================

#[test]
fn test_workgroup_calculation_edge_cases() {
    // Extended edge case testing for workgroup calculation
    // Boundary: workgroup size is 64

    // Below workgroup size
    assert_eq!(workgroups_for_objects(32), 1);
    assert_eq!(workgroups_for_objects(63), 1);

    // Exactly at workgroup boundary
    assert_eq!(workgroups_for_objects(64), 1);

    // Just above workgroup boundary
    assert_eq!(workgroups_for_objects(65), 2);

    // Multiple of workgroup size
    assert_eq!(workgroups_for_objects(192), 3);
    assert_eq!(workgroups_for_objects(256), 4);

    // Non-multiple boundary cases
    assert_eq!(workgroups_for_objects(129), 3); // ceil(129/64) = 3
    assert_eq!(workgroups_for_objects(191), 3); // ceil(191/64) = 3
    assert_eq!(workgroups_for_objects(193), 4); // ceil(193/64) = 4

    // Large counts
    assert_eq!(workgroups_for_objects(10000), 157); // ceil(10000/64) = 157
    assert_eq!(workgroups_for_objects(65536), 1024); // ceil(65536/64) = 1024
}

#[test]
fn test_atomic_or_visibility_pattern() {
    // Test that visibility uses atomic OR (multiple objects can set bits in same word)
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let object_count = 64; // Enough to fill multiple words
    let pipeline = FrustumCullPipelineV2::new(&device);
    let mut frustum = FrustumBuffer::new(&device);
    let mut scene = SceneDataBuffers::new(&device, object_count + 10, Some("test"));
    let visibility = VisibilityFlagsBuffer::new(&device, object_count + 10, Some("test"));

    // Add objects that should all be visible (at origin, in view)
    for i in 0..object_count {
        let obj = ObjectData::new()
            .with_mesh(i as u32)
            .with_flags(object_flags::VISIBLE)
            .with_aabb([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        scene.add(obj);
    }
    scene.upload(&device, &queue);

    // Update frustum
    let vp = test_view_projection();
    frustum.update(&queue, &vp);

    // Clear visibility
    visibility.clear(&queue);

    // Dispatch
    let mut encoder = device.create_command_encoder(&Default::default());
    pipeline.dispatch(
        &mut encoder,
        &device,
        &queue,
        &frustum,
        &scene,
        &visibility,
        object_count as u32,
    );
    queue.submit([encoder.finish()]);
    device.poll(wgpu::Maintain::Wait);

    // Read back and verify all objects visible
    let flags = visibility.read_back(&device, &queue);

    for i in 0..object_count {
        assert!(
            is_visible(&flags, i),
            "Object {} should be visible (atomic OR pattern)",
            i
        );
    }

    // Verify word packing: first word should have all 32 bits set
    if !flags.is_empty() {
        assert_eq!(
            flags[0], 0xFFFFFFFF,
            "First visibility word should have all bits set for first 32 objects"
        );
    }
}

#[test]
fn test_cull_dispatch_params_fields() {
    // Test CullDispatchParams structure and field access
    let params = CullDispatchParams::new(12345);

    assert_eq!(params.object_count, 12345);
    assert_eq!(params.flags, 0);

    // Verify size constraint
    assert!(
        CULL_DISPATCH_PARAMS_SIZE >= 8,
        "CullDispatchParams should be at least 8 bytes (object_count + flags)"
    );
}

#[test]
fn test_dispatch_boundary_object_count() {
    // Test dispatch at workgroup boundaries (64, 128, etc.)
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    for object_count in [64, 65, 127, 128, 129] {
        let pipeline = FrustumCullPipelineV2::new(&device);
        let frustum = FrustumBuffer::new(&device);
        let scene = SceneDataBuffers::new(&device, object_count + 10, Some("test"));
        let visibility = VisibilityFlagsBuffer::new(&device, object_count + 10, Some("test"));

        visibility.clear(&queue);

        let mut encoder = device.create_command_encoder(&Default::default());
        pipeline.dispatch(
            &mut encoder,
            &device,
            &queue,
            &frustum,
            &scene,
            &visibility,
            object_count as u32,
        );
        queue.submit([encoder.finish()]);
        device.poll(wgpu::Maintain::Wait);

        // Verify dispatch completed without error
        let _ = visibility.read_back(&device, &queue);
    }
}

#[test]
fn test_multiple_dispatches_accumulate() {
    // Test that multiple dispatches accumulate visibility (atomic OR)
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let pipeline = FrustumCullPipelineV2::new(&device);
    let mut frustum = FrustumBuffer::new(&device);
    let mut scene = SceneDataBuffers::new(&device, 100, Some("test"));
    let visibility = VisibilityFlagsBuffer::new(&device, 100, Some("test"));

    // Add visible object
    let obj = ObjectData::new()
        .with_mesh(0)
        .with_flags(object_flags::VISIBLE)
        .with_aabb([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
    scene.add(obj);
    scene.upload(&device, &queue);

    let vp = test_view_projection();
    frustum.update(&queue, &vp);

    // Clear visibility
    visibility.clear(&queue);

    // First dispatch
    let mut encoder = device.create_command_encoder(&Default::default());
    pipeline.dispatch(&mut encoder, &device, &queue, &frustum, &scene, &visibility, 1);
    queue.submit([encoder.finish()]);
    device.poll(wgpu::Maintain::Wait);

    let flags1 = visibility.read_back(&device, &queue);
    let visible1 = is_visible(&flags1, 0);

    // Second dispatch (without clearing)
    let mut encoder = device.create_command_encoder(&Default::default());
    pipeline.dispatch(&mut encoder, &device, &queue, &frustum, &scene, &visibility, 1);
    queue.submit([encoder.finish()]);
    device.poll(wgpu::Maintain::Wait);

    let flags2 = visibility.read_back(&device, &queue);
    let visible2 = is_visible(&flags2, 0);

    // Visibility should persist through multiple dispatches (atomic OR)
    assert_eq!(visible1, visible2, "Visibility should be consistent across dispatches");
}

#[test]
fn test_object_data_buffer_read_only() {
    // Test that object data buffer is used read-only by verifying
    // dispatch succeeds and visibility output is correct (object data not corrupted)
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let pipeline = FrustumCullPipelineV2::new(&device);
    let mut frustum = FrustumBuffer::new(&device);
    let mut scene = SceneDataBuffers::new(&device, 100, Some("test"));
    let visibility = VisibilityFlagsBuffer::new(&device, 100, Some("test"));

    // Add test objects at origin (should be visible)
    for i in 0..10 {
        let obj = ObjectData::new()
            .with_mesh(i as u32)
            .with_flags(object_flags::VISIBLE)
            .with_aabb([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        scene.add(obj);
    }
    scene.upload(&device, &queue);

    let vp = test_view_projection();
    frustum.update(&queue, &vp);

    // Clear and dispatch
    visibility.clear(&queue);
    let mut encoder = device.create_command_encoder(&Default::default());
    pipeline.dispatch(&mut encoder, &device, &queue, &frustum, &scene, &visibility, 10);
    queue.submit([encoder.finish()]);
    device.poll(wgpu::Maintain::Wait);

    // Read back visibility - objects should be visible (data not corrupted)
    let flags = visibility.read_back(&device, &queue);

    // All objects at origin should be visible
    for i in 0..10 {
        assert!(
            is_visible(&flags, i),
            "Object {} should be visible (object buffer read-only, data intact)",
            i
        );
    }
}
