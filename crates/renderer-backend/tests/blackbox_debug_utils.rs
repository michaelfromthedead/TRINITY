// Blackbox contract tests for T-WGPU-P4.5.1 Debug Group RAII and T-WGPU-P4.5.2 Debug Markers.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::debug_utils::*` -- no internal fields,
// no private methods, no implementation details.
//
// Acceptance criteria (T-WGPU-P4.5.1):
//   1. push_debug_group() on creation -- DebugScope pushes on new()
//   2. pop_debug_group() on Drop -- DebugScope pops when dropped
//   3. Works with encoder and passes -- CommandEncoder, RenderPass, ComputePass
//   4. Nested scopes supported -- multiple levels work
//
// Acceptance criteria (T-WGPU-P4.5.2):
//   1. insert_debug_marker() wrapper -- insert_marker function
//   2. Timestamp markers -- insert_marker_timed, DebugMarker::with_timestamp
//   3. Conditional markers -- insert_marker_if, debug_marker_if! macro
//   4. Formatted markers -- insert_marker_fmt, debug_marker! macro
//
// Coverage (T-WGPU-P4.5.1):
//   01. DebugScope::new() creates scope and pushes debug group
//   02. DebugScope::empty() creates no-op scope (no push/pop)
//   03. DebugScope drops automatically and pops debug group
//   04. DebugScope::insert_marker() inserts debug marker
//   05. DebugScope::target() returns immutable reference
//   06. DebugScope::target_mut() returns mutable reference
//   07. DebugGroupOps trait works with CommandEncoder
//   08. DebugGroupOps trait works with RenderPass
//   09. DebugGroupOps trait works with ComputePass
//   10. DebugScopeBuilder::new() creates builder
//   11. DebugScopeBuilder::with_start_marker() adds start marker
//   12. DebugScopeBuilder::build() produces DebugScope
//   13. debug_scope() convenience function
//   14. debug_scope_render() convenience function
//   15. debug_scope_compute() convenience function
//   16. Nested scopes -- multiple levels deep
//   17. Sequential scopes -- back-to-back scopes
//   18. Empty label handling
//   19. insert_marker_if() conditional marker insertion
//   20. insert_marker_fmt() formatted marker insertion
//
// Coverage (T-WGPU-P4.5.2 Debug Markers):
//   26. insert_marker() function -- basic marker insertion wrapper
//   27. insert_marker_timed() function -- marker with timestamp
//   28. DebugMarker::new() -- basic construction
//   29. DebugMarker::with_timestamp() -- construction with timestamp
//   30. DebugMarker::with_metadata() -- construction with metadata
//   31. DebugMarker::with_timestamp_and_metadata() -- full construction
//   32. DebugMarker::full_label() -- formatted label generation
//   33. DebugMarker::insert() -- marker insertion on target
//   34. DebugMarkerBuilder::new() -- builder creation
//   35. DebugMarkerBuilder::with_timestamp() -- builder timestamp option
//   36. DebugMarkerBuilder::with_metadata() -- builder metadata option
//   37. DebugMarkerBuilder::build() -- builder produces DebugMarker
//   38. DebugMarkerBuilder::insert() -- builder direct insert
//   39. DebugMarkerBuilder::insert_if() -- builder conditional insert
//   40. insert_marker_if() with true condition
//   41. insert_marker_if() with false condition
//   42. insert_marker_fmt() with format arguments
//   43. debug_marker! macro basic usage
//   44. debug_marker! macro with format args
//   45. debug_marker_if! macro with true condition
//   46. debug_marker_if! macro with false condition
//   47. debug_marker_timed! macro basic usage
//   48. insert_marker_with_metadata() function
//   49. DebugMarkerBuilder chained methods
//   50. Multiple markers in sequence

use pollster::block_on;
use std::time::Instant;
// DebugGroupOps trait must be imported for trait methods to be available on wgpu types
#[allow(unused_imports)]
use renderer_backend::debug_utils::DebugGroupOps;
use renderer_backend::debug_utils::{
    debug_scope, debug_scope_compute, debug_scope_render, insert_marker, insert_marker_fmt,
    insert_marker_if, insert_marker_timed, insert_marker_with_metadata, DebugMarker,
    DebugMarkerBuilder, DebugScope, DebugScopeBuilder,
};
// Import macros
use renderer_backend::{debug_marker, debug_marker_if, debug_marker_timed};

// =========================================================================
// TEST INFRASTRUCTURE -- Headless wgpu device/queue
// =========================================================================

/// Creates a headless wgpu device for testing.
/// Returns None if no adapter is available (CI without GPU).
fn create_test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });

    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;

    Some(
        block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("blackbox_debug_utils test device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .ok()?,
    )
}

/// Creates a simple render target texture for render pass tests.
fn create_render_target(device: &wgpu::Device) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("test_render_target"),
        size: wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

// =========================================================================
// SECTION 1 -- DebugScope Creation and Basic Operations
// =========================================================================

/// Test 01: DebugScope::new() creates scope and pushes debug group
#[test]
fn debug_scope_new_creates_scope() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    // Creating a DebugScope should push a debug group
    {
        let _scope = DebugScope::new(&mut encoder, "test_group");
        // Scope is active here
    }
    // Scope dropped, debug group should be popped

    // If we get here without panic, the push/pop balanced correctly
    let _commands = encoder.finish();
}

/// Test 02: DebugScope::empty() creates no-op scope (no push/pop)
#[test]
fn debug_scope_empty_creates_noop_scope() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    // Empty scope should not push/pop anything
    {
        let _scope = DebugScope::empty(&mut encoder);
        // No debug group pushed
    }

    let _commands = encoder.finish();
}

/// Test 03: DebugScope drops automatically and pops debug group
#[test]
fn debug_scope_drop_pops_debug_group() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    // Explicit scope block to test Drop
    {
        let mut outer = DebugScope::new(&mut encoder, "outer");
        {
            let _inner = DebugScope::new(outer.target_mut(), "inner");
            // inner scope active
        }
        // inner dropped, but outer still active
    }
    // outer dropped

    // Encoder should complete without panic (balanced push/pop)
    let _commands = encoder.finish();
}

/// Test 04: DebugScope::insert_marker() inserts debug marker
#[test]
fn debug_scope_insert_marker() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    {
        let mut scope = DebugScope::new(&mut encoder, "test_group");
        scope.insert_marker("checkpoint_1");
        scope.insert_marker("checkpoint_2");
    }

    let _commands = encoder.finish();
}

/// Test 05: DebugScope::target() returns immutable reference
#[test]
fn debug_scope_target_returns_immutable_ref() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    {
        let scope = DebugScope::new(&mut encoder, "test_group");
        let _target: &wgpu::CommandEncoder = scope.target();
        // Can access target through scope
    }

    let _commands = encoder.finish();
}

/// Test 06: DebugScope::target_mut() returns mutable reference
#[test]
fn debug_scope_target_mut_returns_mutable_ref() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    {
        let mut scope = DebugScope::new(&mut encoder, "test_group");
        let target: &mut wgpu::CommandEncoder = scope.target_mut();
        // Can mutably access target through scope
        target.insert_debug_marker("via_target_mut");
    }

    let _commands = encoder.finish();
}

// =========================================================================
// SECTION 2 -- DebugGroupOps Trait with Different Target Types
// =========================================================================

/// Test 07: DebugGroupOps trait works with CommandEncoder
#[test]
fn debug_group_ops_command_encoder() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    // Use trait methods directly
    encoder.push_debug_group("direct_push");
    encoder.insert_debug_marker("marker");
    encoder.pop_debug_group();

    let _commands = encoder.finish();
}

/// Test 08: DebugGroupOps trait works with RenderPass
#[test]
fn debug_group_ops_render_pass() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let render_target = create_render_target(&device);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("test_render_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &render_target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        // Use trait methods on RenderPass
        render_pass.push_debug_group("render_group");
        render_pass.insert_debug_marker("render_marker");
        render_pass.pop_debug_group();
    }

    let _commands = encoder.finish();
}

/// Test 09: DebugGroupOps trait works with ComputePass
#[test]
fn debug_group_ops_compute_pass() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("test_compute_pass"),
            timestamp_writes: None,
        });

        // Use trait methods on ComputePass
        compute_pass.push_debug_group("compute_group");
        compute_pass.insert_debug_marker("compute_marker");
        compute_pass.pop_debug_group();
    }

    let _commands = encoder.finish();
}

// =========================================================================
// SECTION 3 -- DebugScopeBuilder Pattern
// =========================================================================

/// Test 10: DebugScopeBuilder::new() creates builder
#[test]
fn debug_scope_builder_new() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    {
        let builder = DebugScopeBuilder::new(&mut encoder, "builder_test");
        let _scope = builder.build();
    }

    let _commands = encoder.finish();
}

/// Test 11: DebugScopeBuilder::with_start_marker() adds start marker
#[test]
fn debug_scope_builder_with_start_marker() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    {
        let _scope = DebugScopeBuilder::new(&mut encoder, "builder_test")
            .with_start_marker("start_marker")
            .build();
    }

    let _commands = encoder.finish();
}

/// Test 12: DebugScopeBuilder::build() produces DebugScope
#[test]
fn debug_scope_builder_build_produces_scope() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    {
        let scope: DebugScope<'_, wgpu::CommandEncoder> =
            DebugScopeBuilder::new(&mut encoder, "typed_test").build();

        // Verify we have a proper scope
        let _target = scope.target();
    }

    let _commands = encoder.finish();
}

// =========================================================================
// SECTION 4 -- Convenience Functions
// =========================================================================

/// Test 13: debug_scope() convenience function
#[test]
fn debug_scope_convenience_function() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    {
        let _scope = debug_scope(&mut encoder, "convenience_test");
    }

    let _commands = encoder.finish();
}

/// Test 14: debug_scope_render() convenience function
#[test]
fn debug_scope_render_convenience_function() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let render_target = create_render_target(&device);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("test_render_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &render_target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        {
            let _scope = debug_scope_render(&mut render_pass, "render_convenience");
        }
    }

    let _commands = encoder.finish();
}

/// Test 15: debug_scope_compute() convenience function
#[test]
fn debug_scope_compute_convenience_function() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("test_compute_pass"),
            timestamp_writes: None,
        });

        {
            let _scope = debug_scope_compute(&mut compute_pass, "compute_convenience");
        }
    }

    let _commands = encoder.finish();
}

// =========================================================================
// SECTION 5 -- Nested and Sequential Scopes
// =========================================================================

/// Test 16: Nested scopes -- multiple levels deep
#[test]
fn nested_scopes_multiple_levels() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    {
        let mut level1 = DebugScope::new(&mut encoder, "level_1");
        {
            let mut level2 = DebugScope::new(level1.target_mut(), "level_2");
            {
                let mut level3 = DebugScope::new(level2.target_mut(), "level_3");
                {
                    let _level4 = DebugScope::new(level3.target_mut(), "level_4");
                    // Four levels deep
                }
            }
        }
    }

    let _commands = encoder.finish();
}

/// Test 17: Sequential scopes -- back-to-back scopes
#[test]
fn sequential_scopes_back_to_back() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    // First scope
    {
        let _scope1 = DebugScope::new(&mut encoder, "scope_1");
    }

    // Second scope (after first is dropped)
    {
        let _scope2 = DebugScope::new(&mut encoder, "scope_2");
    }

    // Third scope
    {
        let _scope3 = DebugScope::new(&mut encoder, "scope_3");
    }

    let _commands = encoder.finish();
}

/// Test 18: Empty label handling
#[test]
fn empty_label_handling() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    // Empty label should still work
    {
        let _scope = DebugScope::new(&mut encoder, "");
    }

    let _commands = encoder.finish();
}

// =========================================================================
// SECTION 6 -- Marker Insertion Functions
// =========================================================================

/// Test 19: insert_marker_if() conditional marker insertion
#[test]
fn insert_marker_if_conditional() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    // Should insert marker when condition is true
    insert_marker_if(&mut encoder, true, "conditional_true");

    // Should not insert marker when condition is false
    insert_marker_if(&mut encoder, false, "conditional_false");

    let _commands = encoder.finish();
}

/// Test 20: insert_marker_fmt() formatted marker insertion
#[test]
fn insert_marker_fmt_formatted() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    let iteration = 42;
    insert_marker_fmt(&mut encoder, format_args!("iteration_{}", iteration));

    let _commands = encoder.finish();
}

// =========================================================================
// SECTION 7 -- Combined Complex Scenarios
// =========================================================================

/// Test 21: Complex scenario with builder, nesting, and markers
#[test]
fn complex_scenario_builder_nesting_markers() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    {
        let mut outer = DebugScopeBuilder::new(&mut encoder, "frame_render")
            .with_start_marker("frame_begin")
            .build();

        outer.insert_marker("pre_geometry");

        {
            let mut inner = DebugScope::new(outer.target_mut(), "geometry_pass");
            inner.insert_marker("drawing_meshes");
        }

        outer.insert_marker("post_geometry");
    }

    let _commands = encoder.finish();
}

/// Test 22: Debug scope with render pass operations
#[test]
fn debug_scope_with_render_operations() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let render_target = create_render_target(&device);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("test_render_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &render_target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        {
            let mut scope = debug_scope_render(&mut render_pass, "opaque_pass");
            scope.insert_marker("opaque_start");
            // Would draw opaque geometry here
            scope.insert_marker("opaque_end");
        }

        {
            let mut scope = debug_scope_render(&mut render_pass, "transparent_pass");
            scope.insert_marker("transparent_start");
            // Would draw transparent geometry here
            scope.insert_marker("transparent_end");
        }
    }

    let _commands = encoder.finish();
}

/// Test 23: Multiple encoders with independent scopes
#[test]
fn multiple_encoders_independent_scopes() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder1 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("encoder_1"),
    });

    let mut encoder2 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("encoder_2"),
    });

    // Scopes on different encoders should be independent
    {
        let _scope1 = DebugScope::new(&mut encoder1, "encoder1_scope");
        {
            let _scope2 = DebugScope::new(&mut encoder2, "encoder2_scope");
            // Both scopes active simultaneously on different encoders
        }
    }

    let _commands1 = encoder1.finish();
    let _commands2 = encoder2.finish();
}

/// Test 24: Scope with String label (owned)
#[test]
fn debug_scope_with_owned_string_label() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    let label = String::from("owned_label");
    {
        let _scope = DebugScopeBuilder::new(&mut encoder, label).build();
    }

    let _commands = encoder.finish();
}

/// Test 25: Deeply nested with markers at each level
#[test]
fn deeply_nested_with_markers() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    {
        let mut l1 = DebugScope::new(&mut encoder, "level_1");
        l1.insert_marker("l1_marker");
        {
            let mut l2 = DebugScope::new(l1.target_mut(), "level_2");
            l2.insert_marker("l2_marker");
            {
                let mut l3 = DebugScope::new(l2.target_mut(), "level_3");
                l3.insert_marker("l3_marker");
            }
            l2.insert_marker("l2_after_l3");
        }
        l1.insert_marker("l1_after_l2");
    }

    let _commands = encoder.finish();
}

// =========================================================================
// SECTION 8 -- T-WGPU-P4.5.2 Debug Markers
// =========================================================================

/// Test 26: insert_marker() function -- basic marker insertion wrapper
#[test]
fn insert_marker_basic_wrapper() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    // insert_marker is a wrapper for insert_debug_marker
    insert_marker(&mut encoder, "basic_marker");
    insert_marker(&mut encoder, "another_marker");

    let _commands = encoder.finish();
}

/// Test 27: insert_marker_timed() function -- marker with timestamp
#[test]
fn insert_marker_timed_with_timestamp() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    let reference_time = Instant::now();

    // Insert timed marker with reference (wrapped in Some)
    insert_marker_timed(&mut encoder, "timed_marker", Some(reference_time));

    let _commands = encoder.finish();
}

/// Test 28: DebugMarker::new() -- basic construction
#[test]
fn debug_marker_new_basic_construction() {
    let marker = DebugMarker::new("test_label");

    // Verify basic properties through full_label (no internal access)
    let label = marker.full_label(None);
    assert!(label.contains("test_label"));
}

/// Test 29: DebugMarker::with_timestamp() -- construction with timestamp
#[test]
fn debug_marker_with_timestamp_construction() {
    let marker = DebugMarker::with_timestamp("timed_label");

    // When reference time is provided, should include timing info
    let reference = Instant::now();
    // Small delay to ensure some time has passed
    std::thread::sleep(std::time::Duration::from_millis(1));

    let label = marker.full_label(Some(reference));
    // Label should contain the base label
    assert!(label.contains("timed_label"));
}

/// Test 30: DebugMarker::with_metadata() -- construction with metadata
#[test]
fn debug_marker_with_metadata_construction() {
    let marker = DebugMarker::with_metadata("label_with_meta", "frame=42");

    let label = marker.full_label(None);
    // Should contain both label and metadata
    assert!(label.contains("label_with_meta"));
    assert!(label.contains("frame=42"));
}

/// Test 31: DebugMarker::with_timestamp_and_metadata() -- full construction
#[test]
fn debug_marker_with_timestamp_and_metadata() {
    let marker = DebugMarker::with_timestamp_and_metadata("full_marker", "pass=shadow");

    let reference = Instant::now();
    let label = marker.full_label(Some(reference));

    assert!(label.contains("full_marker"));
    assert!(label.contains("pass=shadow"));
}

/// Test 32: DebugMarker::full_label() -- formatted label generation
#[test]
fn debug_marker_full_label_formatting() {
    // Without timestamp or metadata
    let simple = DebugMarker::new("simple");
    assert_eq!(simple.full_label(None), "simple");

    // With metadata only
    let with_meta = DebugMarker::with_metadata("labelled", "info");
    let label = with_meta.full_label(None);
    assert!(label.contains("labelled"));
    assert!(label.contains("info"));
}

/// Test 33: DebugMarker::insert() -- marker insertion on target
#[test]
fn debug_marker_insert_on_target() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    let marker = DebugMarker::new("inserted_marker");
    marker.insert(&mut encoder);

    let _commands = encoder.finish();
}

/// Test 34: DebugMarkerBuilder::new() -- builder creation
#[test]
fn debug_marker_builder_new() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    // Builder can be created
    let _builder = DebugMarkerBuilder::new(&mut encoder, "builder_marker");

    let _commands = encoder.finish();
}

/// Test 35: DebugMarkerBuilder::with_timestamp() -- builder timestamp option
#[test]
fn debug_marker_builder_with_timestamp() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    DebugMarkerBuilder::new(&mut encoder, "timestamp_builder")
        .with_timestamp()
        .insert();

    let _commands = encoder.finish();
}

/// Test 36: DebugMarkerBuilder::with_metadata() -- builder metadata option
#[test]
fn debug_marker_builder_with_metadata() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    DebugMarkerBuilder::new(&mut encoder, "metadata_builder")
        .with_metadata("draw_call=100")
        .insert();

    let _commands = encoder.finish();
}

/// Test 37: DebugMarkerBuilder::build() -- builder produces DebugMarker
#[test]
fn debug_marker_builder_build() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    let marker: DebugMarker = DebugMarkerBuilder::new(&mut encoder, "built_marker")
        .with_metadata("built")
        .build();

    // Marker was built successfully
    let label = marker.full_label(None);
    assert!(label.contains("built_marker"));

    let _commands = encoder.finish();
}

/// Test 38: DebugMarkerBuilder::insert() -- builder direct insert
#[test]
fn debug_marker_builder_insert_direct() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    // Direct insert without calling build()
    DebugMarkerBuilder::new(&mut encoder, "direct_insert").insert();

    let _commands = encoder.finish();
}

/// Test 39: DebugMarkerBuilder::insert_if() -- builder conditional insert
#[test]
fn debug_marker_builder_insert_if() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    // Should insert (condition true)
    DebugMarkerBuilder::new(&mut encoder, "conditional_true").insert_if(true);

    // Should not insert (condition false)
    DebugMarkerBuilder::new(&mut encoder, "conditional_false").insert_if(false);

    let _commands = encoder.finish();
}

/// Test 40: insert_marker_if() with true condition
#[test]
fn insert_marker_if_true_condition() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    // Marker should be inserted when condition is true
    insert_marker_if(&mut encoder, true, "should_insert");

    let _commands = encoder.finish();
}

/// Test 41: insert_marker_if() with false condition
#[test]
fn insert_marker_if_false_condition() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    // Marker should NOT be inserted when condition is false
    insert_marker_if(&mut encoder, false, "should_not_insert");

    let _commands = encoder.finish();
}

/// Test 42: insert_marker_fmt() with format arguments
#[test]
fn insert_marker_fmt_with_format_args() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    let frame_id = 42;
    let pass_name = "shadow";
    insert_marker_fmt(&mut encoder, format_args!("frame_{}_pass_{}", frame_id, pass_name));

    let _commands = encoder.finish();
}

/// Test 43: debug_marker! macro basic usage
#[test]
fn debug_marker_macro_basic() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    debug_marker!(&mut encoder, "macro_marker");

    let _commands = encoder.finish();
}

/// Test 44: debug_marker! macro with format args
#[test]
fn debug_marker_macro_with_format() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    let iteration = 5;
    debug_marker!(&mut encoder, "iteration_{}", iteration);

    let _commands = encoder.finish();
}

/// Test 45: debug_marker_if! macro with true condition
#[test]
fn debug_marker_if_macro_true() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    let should_mark = true;
    debug_marker_if!(&mut encoder, should_mark, "conditional_macro_true");

    let _commands = encoder.finish();
}

/// Test 46: debug_marker_if! macro with false condition
#[test]
fn debug_marker_if_macro_false() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    let should_mark = false;
    debug_marker_if!(&mut encoder, should_mark, "conditional_macro_false");

    let _commands = encoder.finish();
}

/// Test 47: debug_marker_timed! macro basic usage
#[test]
fn debug_marker_timed_macro_basic() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    let frame_start = Instant::now();
    debug_marker_timed!(&mut encoder, frame_start, "timed_macro_marker");

    let _commands = encoder.finish();
}

/// Test 48: insert_marker_with_metadata() function
#[test]
fn insert_marker_with_metadata_function() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    insert_marker_with_metadata(&mut encoder, "metadata_marker", "objects=1000");

    let _commands = encoder.finish();
}

/// Test 49: DebugMarkerBuilder chained methods
#[test]
fn debug_marker_builder_chained() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    let reference = Instant::now();

    DebugMarkerBuilder::new(&mut encoder, "fully_chained")
        .with_timestamp()
        .with_metadata("draw_calls=50")
        .with_reference_time(reference)
        .insert();

    let _commands = encoder.finish();
}

/// Test 50: Multiple markers in sequence
#[test]
fn multiple_markers_in_sequence() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    let frame_start = Instant::now();

    // Mix of different marker types in sequence
    insert_marker(&mut encoder, "frame_begin");
    insert_marker_timed(&mut encoder, "geometry_start", Some(frame_start));
    insert_marker_with_metadata(&mut encoder, "draw_batch", "meshes=100");
    insert_marker_if(&mut encoder, true, "checkpoint_1");
    insert_marker_fmt(&mut encoder, format_args!("pass_{}_complete", 0));

    debug_marker!(&mut encoder, "shadow_pass");
    debug_marker_if!(&mut encoder, true, "lighting_pass");
    debug_marker_timed!(&mut encoder, frame_start, "post_process");

    insert_marker(&mut encoder, "frame_end");

    let _commands = encoder.finish();
}

// =========================================================================
// SECTION 9 -- Debug Marker Edge Cases
// =========================================================================

/// Test 51: DebugMarker with empty label
#[test]
fn debug_marker_empty_label() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    // Empty label should still work
    let marker = DebugMarker::new("");
    marker.insert(&mut encoder);

    let _commands = encoder.finish();
}

/// Test 52: DebugMarker with unicode label
#[test]
fn debug_marker_unicode_label() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    let marker = DebugMarker::new("unicode_test_\u{1F680}_rocket");
    marker.insert(&mut encoder);

    let label = marker.full_label(None);
    assert!(label.contains("\u{1F680}"));

    let _commands = encoder.finish();
}

/// Test 53: DebugMarker insert_full with reference time
#[test]
fn debug_marker_insert_full_with_reference() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    let reference = Instant::now();
    std::thread::sleep(std::time::Duration::from_millis(1));

    let marker = DebugMarker::with_timestamp("full_insert_test");
    marker.insert_full(&mut encoder, Some(reference));

    let _commands = encoder.finish();
}

/// Test 54: DebugMarkerBuilder with_reference_time
#[test]
fn debug_marker_builder_with_reference_time() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    let reference = Instant::now();

    DebugMarkerBuilder::new(&mut encoder, "reference_time_test")
        .with_reference_time(reference)
        .insert();

    let _commands = encoder.finish();
}

/// Test 55: Markers on RenderPass
#[test]
fn debug_markers_on_render_pass() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let render_target = create_render_target(&device);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("test_render_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &render_target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        // Use various marker functions on RenderPass
        insert_marker(&mut render_pass, "render_marker_1");
        insert_marker_if(&mut render_pass, true, "render_conditional");
        insert_marker_fmt(&mut render_pass, format_args!("render_formatted_{}", 42));

        let marker = DebugMarker::new("render_struct_marker");
        marker.insert(&mut render_pass);
    }

    let _commands = encoder.finish();
}

/// Test 56: Markers on ComputePass
#[test]
fn debug_markers_on_compute_pass() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("test_compute_pass"),
            timestamp_writes: None,
        });

        // Use various marker functions on ComputePass
        insert_marker(&mut compute_pass, "compute_marker_1");
        insert_marker_if(&mut compute_pass, true, "compute_conditional");
        insert_marker_fmt(&mut compute_pass, format_args!("compute_formatted_{}", 99));

        let marker = DebugMarker::new("compute_struct_marker");
        marker.insert(&mut compute_pass);
    }

    let _commands = encoder.finish();
}

/// Test 57: Long marker label handling
#[test]
fn debug_marker_long_label() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    // Create a very long label
    let long_label = "a".repeat(1000);
    insert_marker(&mut encoder, &long_label);

    let marker = DebugMarker::new(&long_label);
    assert_eq!(marker.full_label(None).len(), 1000);

    let _commands = encoder.finish();
}

/// Test 58: Marker with special characters
#[test]
fn debug_marker_special_characters() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    insert_marker(&mut encoder, "label:with:colons");
    insert_marker(&mut encoder, "label/with/slashes");
    insert_marker(&mut encoder, "label[with]brackets");
    insert_marker(&mut encoder, "label{with}braces");
    insert_marker(&mut encoder, "label<with>angles");
    insert_marker(&mut encoder, "label|with|pipes");

    let _commands = encoder.finish();
}

/// Test 59: debug_marker_if! macro with format args
#[test]
fn debug_marker_if_macro_with_format() {
    let Some((device, _queue)) = create_test_device() else {
        eprintln!("SKIP: no GPU adapter available");
        return;
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    let frame = 100;
    let should_mark = true;
    debug_marker_if!(&mut encoder, should_mark, "frame_{}_conditional", frame);

    let _commands = encoder.finish();
}

/// Test 60: DebugMarker timestamp relative to reference
#[test]
fn debug_marker_timestamp_relative_formatting() {
    let reference = Instant::now();
    std::thread::sleep(std::time::Duration::from_millis(10));

    let marker = DebugMarker::with_timestamp("relative_test");

    // With reference, label should include timing info
    let label_with_ref = marker.full_label(Some(reference));
    assert!(label_with_ref.contains("relative_test"));

    // Without reference, just the label
    let label_without_ref = marker.full_label(None);
    assert!(label_without_ref.contains("relative_test"));
}
