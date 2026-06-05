// Blackbox integration tests for Indirect Buffer Generation (T-WGPU-P6.6.2).
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only observable behavior from user perspective.
//
// Component: GPU compute shader that builds DrawIndexedIndirectArgs from
// visible object list with LOD-aware mesh selection.
//
// Test Scope (Integration/Behavior):
//   1. Pipeline lifecycle -- create, dispatch, read results
//   2. Visible object processing -- input list to output indirect args
//   3. LOD mesh selection -- correct mesh data lookup per LOD level
//   4. Atomic count -- draw_count matches visible object count
//   5. Batched mode -- correct behavior for >10K objects
//   6. CPU/GPU parity -- CPU reference matches expected output
//   7. Edge cases -- 0 objects, max objects, all same LOD, mixed LODs

use bytemuck;
use pollster::FutureExt;
use std::mem;

// Import public types from the crate
use renderer_backend::gpu_driven::build_indirect::{
    BuildIndirectParams, BuildIndirectPipeline, BuildIndirectResources,
    IndirectDrawIndexedArgs, MeshData, cpu_build_indirect,
    BATCH_SIZE, WORKGROUP_SIZE, MAX_LOD_LEVELS,
    BUILD_INDIRECT_PARAMS_SIZE, MESH_DATA_SIZE, DRAW_INDEXED_INDIRECT_ARGS_SIZE,
};

use renderer_backend::gpu_driven::object_data::{ObjectData, object_flags, OBJECT_DATA_SIZE};
use renderer_backend::gpu_driven::lod_buffer::{LodEntry, LodBuffer, LOD_ENTRY_SIZE};
use renderer_backend::gpu_driven::scene_data::SceneDataBuffers;

// =============================================================================
// TEST FIXTURES
// =============================================================================

/// Test harness for build indirect pipeline testing.
struct BuildIndirectTestHarness {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: BuildIndirectPipeline,
}

impl BuildIndirectTestHarness {
    /// Create a new test harness with GPU resources.
    fn new() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .block_on()?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("test_device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .block_on()
            .ok()?;

        let pipeline = BuildIndirectPipeline::new(&device);

        Some(Self {
            device,
            queue,
            pipeline,
        })
    }

    /// Create test buffers for a specific configuration using proper buffer managers.
    fn create_test_buffers(
        &self,
        compacted_indices: &[u32],
        object_data: &[ObjectData],
        mesh_data: &[MeshData],
        lod_entries: &[LodEntry],
        max_draws: u32,
    ) -> TestBuffers {
        // Use a minimum capacity to ensure valid buffer sizes
        let min_capacity = object_data.len().max(1).max(compacted_indices.len().max(1));

        // Compacted indices buffer
        let compacted_indices_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_compacted_indices"),
            size: (min_capacity * mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !compacted_indices.is_empty() {
            self.queue.write_buffer(
                &compacted_indices_buffer,
                0,
                bytemuck::cast_slice(compacted_indices),
            );
        }

        // Use SceneDataBuffers for proper ObjectData buffer management
        let mut scene_data = SceneDataBuffers::new(&self.device, min_capacity, Some("test"));
        for obj in object_data {
            scene_data.add(*obj);
        }
        scene_data.upload(&self.device, &self.queue);

        // Mesh data buffer
        let mesh_count = mesh_data.len().max(1);
        let mesh_data_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_mesh_data"),
            size: (mesh_count * MESH_DATA_SIZE) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !mesh_data.is_empty() {
            self.queue.write_buffer(
                &mesh_data_buffer,
                0,
                bytemuck::cast_slice(mesh_data),
            );
        }

        // Use LodBuffer for proper LOD entry management
        let lod_buffer = LodBuffer::new(&self.device, min_capacity as u32, Some("test"));
        if !lod_entries.is_empty() {
            lod_buffer.upload(&self.queue, lod_entries);
        }

        // Build indirect resources (output buffers)
        let resources = BuildIndirectResources::new(
            &self.device,
            min_capacity as u32,
            max_draws,
        );

        TestBuffers {
            compacted_indices_buffer,
            scene_data,
            mesh_data_buffer,
            lod_buffer,
            resources,
        }
    }

    /// Execute build indirect dispatch and read back results.
    fn dispatch_and_read(
        &self,
        buffers: &TestBuffers,
        visible_count: u32,
        max_draws: u32,
        use_batched: bool,
    ) -> (u32, Vec<IndirectDrawIndexedArgs>) {
        let params = BuildIndirectParams::new(visible_count, max_draws);

        // Upload params
        buffers.resources.upload_params(&self.queue, &params);

        // Clear draw count
        buffers.resources.clear_draw_count(&self.queue);

        // Create bind groups
        let input_bind_group = self.pipeline.create_input_bind_group(
            &self.device,
            &buffers.compacted_indices_buffer,
            buffers.scene_data.object_buffer(),
            &buffers.mesh_data_buffer,
            buffers.lod_buffer.buffer(),
        );

        let output_bind_group = self.pipeline.create_output_bind_group(
            &self.device,
            &buffers.resources,
        );

        let params_bind_group = self.pipeline.create_params_bind_group(
            &self.device,
            &buffers.resources,
        );

        // Dispatch
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_build_indirect_encoder"),
        });

        self.pipeline.dispatch(
            &mut encoder,
            &input_bind_group,
            &output_bind_group,
            &params_bind_group,
            &params,
            use_batched,
        );

        self.queue.submit([encoder.finish()]);

        // Read back draw count
        let draw_count = buffers.resources.read_draw_count(&self.device, &self.queue);

        // Read back indirect commands
        let commands = self.read_indirect_commands(
            &buffers.resources.indirect_commands_buffer,
            draw_count.min(max_draws) as usize,
        );

        (draw_count, commands)
    }

    /// Read indirect commands from GPU buffer.
    fn read_indirect_commands(
        &self,
        commands_buffer: &wgpu::Buffer,
        count: usize,
    ) -> Vec<IndirectDrawIndexedArgs> {
        if count == 0 {
            return Vec::new();
        }

        let size = count * DRAW_INDEXED_INDIRECT_ARGS_SIZE;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_commands_staging"),
            size: size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_read_commands_encoder"),
        });
        encoder.copy_buffer_to_buffer(commands_buffer, 0, &staging, 0, size as u64);
        self.queue.submit([encoder.finish()]);

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let data = slice.get_mapped_range();
        let commands: Vec<IndirectDrawIndexedArgs> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        commands
    }
}

struct TestBuffers {
    compacted_indices_buffer: wgpu::Buffer,
    scene_data: SceneDataBuffers,
    mesh_data_buffer: wgpu::Buffer,
    lod_buffer: LodBuffer,
    resources: BuildIndirectResources,
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Create an ObjectData with specified mesh index.
fn make_object(mesh_index: u32) -> ObjectData {
    let mut obj = ObjectData::default();
    obj.mesh_index = mesh_index;
    obj.flags = object_flags::VISIBLE;
    obj
}

/// Create a simple MeshData without LOD.
fn make_simple_mesh(index_count: u32, first_index: u32, base_vertex: i32) -> MeshData {
    MeshData::new(index_count, first_index, base_vertex)
}

/// Create a MeshData with LOD support.
fn make_lod_mesh(
    index_count: u32,
    first_index: u32,
    base_vertex: i32,
    lod_counts: &[u32],
    lod_offsets: &[u32],
) -> MeshData {
    MeshData::with_lods(index_count, first_index, base_vertex, lod_counts, lod_offsets)
}

/// Create a LodEntry with specified level.
fn make_lod_entry(level: u32) -> LodEntry {
    LodEntry {
        level,
        blend_factor: 0.0,
    }
}

// =============================================================================
// TEST 1: PIPELINE LIFECYCLE
// =============================================================================

#[test]
fn test_pipeline_creation_succeeds() {
    let harness = match BuildIndirectTestHarness::new() {
        Some(h) => h,
        None => {
            eprintln!("Skipping GPU test: no adapter available");
            return;
        }
    };

    // Pipeline should be created successfully
    assert!(harness.pipeline.input_bind_group_layout().global_id().inner() > 0);
    assert!(harness.pipeline.output_bind_group_layout().global_id().inner() > 0);
    assert!(harness.pipeline.params_bind_group_layout().global_id().inner() > 0);
}

#[test]
fn test_resources_creation_succeeds() {
    let harness = match BuildIndirectTestHarness::new() {
        Some(h) => h,
        None => {
            eprintln!("Skipping GPU test: no adapter available");
            return;
        }
    };

    let resources = BuildIndirectResources::new(&harness.device, 1000, 4096);

    // Resources should be created with correct capacity
    assert_eq!(resources.max_visible, 1000);
    assert_eq!(resources.max_draws, 4096);
}

// =============================================================================
// TEST 2: ZERO OBJECTS (EMPTY INPUT)
// =============================================================================

#[test]
fn test_zero_objects_produces_zero_draw_count() {
    let harness = match BuildIndirectTestHarness::new() {
        Some(h) => h,
        None => {
            eprintln!("Skipping GPU test: no adapter available");
            return;
        }
    };

    let compacted_indices: Vec<u32> = vec![];
    let object_data: Vec<ObjectData> = vec![];
    let mesh_data = vec![make_simple_mesh(100, 0, 0)];
    let lod_entries: Vec<LodEntry> = vec![];

    let buffers = harness.create_test_buffers(
        &compacted_indices,
        &object_data,
        &mesh_data,
        &lod_entries,
        1024,
    );

    let (draw_count, commands) = harness.dispatch_and_read(&buffers, 0, 1024, false);

    assert_eq!(draw_count, 0, "Draw count should be 0 for empty input");
    assert_eq!(commands.len(), 0, "No commands should be generated");
}

// =============================================================================
// TEST 3: SINGLE OBJECT
// =============================================================================

#[test]
fn test_single_object_produces_single_draw() {
    let harness = match BuildIndirectTestHarness::new() {
        Some(h) => h,
        None => {
            eprintln!("Skipping GPU test: no adapter available");
            return;
        }
    };

    let compacted_indices = vec![0u32];
    let object_data = vec![make_object(0)];
    let mesh_data = vec![make_simple_mesh(300, 0, 0)];
    let lod_entries = vec![make_lod_entry(0)];

    let buffers = harness.create_test_buffers(
        &compacted_indices,
        &object_data,
        &mesh_data,
        &lod_entries,
        1024,
    );

    let (draw_count, commands) = harness.dispatch_and_read(&buffers, 1, 1024, false);

    assert_eq!(draw_count, 1, "Draw count should be 1 for single object");
    assert_eq!(commands.len(), 1, "One command should be generated");

    let cmd = &commands[0];
    assert_eq!(cmd.index_count, 300, "Index count should match mesh");
    assert_eq!(cmd.instance_count, 1, "Instance count should be 1");
    assert_eq!(cmd.first_index, 0, "First index should be 0");
    assert_eq!(cmd.base_vertex, 0, "Base vertex should be 0");
}

// =============================================================================
// TEST 4: MULTIPLE OBJECTS - ALL SAME LOD
// =============================================================================

#[test]
fn test_multiple_objects_same_lod() {
    let harness = match BuildIndirectTestHarness::new() {
        Some(h) => h,
        None => {
            eprintln!("Skipping GPU test: no adapter available");
            return;
        }
    };

    // 5 objects, all using mesh 0 at LOD 0
    let compacted_indices: Vec<u32> = (0..5).collect();
    let object_data: Vec<ObjectData> = (0..5).map(|_| make_object(0)).collect();
    let mesh_data = vec![make_lod_mesh(
        1000, 0, 0,
        &[1000, 500, 250, 125],
        &[0, 1000, 1500, 1750],
    )];
    let lod_entries: Vec<LodEntry> = (0..5).map(|_| make_lod_entry(0)).collect();

    let buffers = harness.create_test_buffers(
        &compacted_indices,
        &object_data,
        &mesh_data,
        &lod_entries,
        1024,
    );

    let (draw_count, commands) = harness.dispatch_and_read(&buffers, 5, 1024, false);

    assert_eq!(draw_count, 5, "Draw count should be 5");
    assert_eq!(commands.len(), 5, "Five commands should be generated");

    // All commands should use LOD 0 (1000 indices)
    for (i, cmd) in commands.iter().enumerate() {
        assert_eq!(
            cmd.index_count, 1000,
            "Command {} should have 1000 indices (LOD 0)", i
        );
        assert_eq!(cmd.instance_count, 1);
        assert_eq!(cmd.first_index, 0, "LOD 0 starts at offset 0");
    }
}

// =============================================================================
// TEST 5: MIXED LOD LEVELS
// =============================================================================

#[test]
fn test_mixed_lod_levels() {
    let harness = match BuildIndirectTestHarness::new() {
        Some(h) => h,
        None => {
            eprintln!("Skipping GPU test: no adapter available");
            return;
        }
    };

    // 4 objects, one at each LOD level
    let compacted_indices: Vec<u32> = (0..4).collect();
    let object_data: Vec<ObjectData> = (0..4).map(|_| make_object(0)).collect();
    let mesh_data = vec![make_lod_mesh(
        1000, 0, 0,
        &[1000, 500, 250, 125],  // LOD 0-3 index counts
        &[0, 1000, 1500, 1750],  // LOD 0-3 first index offsets
    )];
    let lod_entries: Vec<LodEntry> = (0..4)
        .map(|i| make_lod_entry(i as u32))
        .collect();

    let buffers = harness.create_test_buffers(
        &compacted_indices,
        &object_data,
        &mesh_data,
        &lod_entries,
        1024,
    );

    let (draw_count, commands) = harness.dispatch_and_read(&buffers, 4, 1024, false);

    assert_eq!(draw_count, 4, "Draw count should be 4");
    assert_eq!(commands.len(), 4, "Four commands should be generated");

    // Verify each LOD level is correctly applied
    // Note: GPU output order may differ due to atomics, so we check presence
    let expected: [(u32, u32); 4] = [
        (1000, 0),     // LOD 0: 1000 indices at offset 0
        (500, 1000),   // LOD 1: 500 indices at offset 1000
        (250, 1500),   // LOD 2: 250 indices at offset 1500
        (125, 1750),   // LOD 3: 125 indices at offset 1750
    ];

    // All expected values should appear in the output
    for (count, offset) in expected.iter() {
        let found = commands.iter().any(|c| c.index_count == *count && c.first_index == *offset);
        assert!(
            found,
            "Expected command with index_count={} first_index={} not found. Got: {:?}",
            count, offset, commands
        );
    }
}

// =============================================================================
// TEST 6: MULTIPLE MESHES
// =============================================================================

#[test]
fn test_multiple_meshes() {
    let harness = match BuildIndirectTestHarness::new() {
        Some(h) => h,
        None => {
            eprintln!("Skipping GPU test: no adapter available");
            return;
        }
    };

    // 3 objects using different meshes
    let compacted_indices: Vec<u32> = (0..3).collect();
    let object_data: Vec<ObjectData> = vec![
        make_object(0),  // Uses mesh 0
        make_object(1),  // Uses mesh 1
        make_object(2),  // Uses mesh 2
    ];
    let mesh_data = vec![
        make_simple_mesh(100, 0, 0),
        make_simple_mesh(200, 100, 50),
        make_simple_mesh(300, 300, 100),
    ];
    let lod_entries: Vec<LodEntry> = (0..3).map(|_| make_lod_entry(0)).collect();

    let buffers = harness.create_test_buffers(
        &compacted_indices,
        &object_data,
        &mesh_data,
        &lod_entries,
        1024,
    );

    let (draw_count, commands) = harness.dispatch_and_read(&buffers, 3, 1024, false);

    assert_eq!(draw_count, 3, "Draw count should be 3");
    assert_eq!(commands.len(), 3, "Three commands should be generated");

    // Verify all mesh configurations are present
    let expected: Vec<(u32, u32, i32)> = vec![
        (100, 0, 0),
        (200, 100, 50),
        (300, 300, 100),
    ];

    for (idx_count, first_idx, base_vertex) in expected {
        let found = commands.iter().any(|c| {
            c.index_count == idx_count
                && c.first_index == first_idx
                && c.base_vertex == base_vertex
        });
        assert!(
            found,
            "Expected mesh config (idx_count={}, first_idx={}, base_vertex={}) not found",
            idx_count, first_idx, base_vertex
        );
    }
}

// =============================================================================
// TEST 7: LOD FALLBACK (NO EXPLICIT LOD DATA)
// =============================================================================

#[test]
fn test_lod_fallback_to_base_mesh() {
    let harness = match BuildIndirectTestHarness::new() {
        Some(h) => h,
        None => {
            eprintln!("Skipping GPU test: no adapter available");
            return;
        }
    };

    // Mesh with no LOD-specific counts (zeros)
    let compacted_indices = vec![0u32];
    let object_data = vec![make_object(0)];
    let mesh_data = vec![MeshData::new(500, 0, 0)]; // No LOD data, uses base index_count
    let lod_entries = vec![make_lod_entry(2)]; // Request LOD 2, but should fallback

    let buffers = harness.create_test_buffers(
        &compacted_indices,
        &object_data,
        &mesh_data,
        &lod_entries,
        1024,
    );

    let (draw_count, commands) = harness.dispatch_and_read(&buffers, 1, 1024, false);

    assert_eq!(draw_count, 1);
    assert_eq!(commands.len(), 1);

    // Should use base index_count since LOD counts are 0
    assert_eq!(
        commands[0].index_count, 500,
        "Should fall back to base index_count when LOD count is 0"
    );
}

// =============================================================================
// TEST 8: MAX DRAWS LIMIT
// =============================================================================

#[test]
fn test_max_draws_limit_enforced() {
    let harness = match BuildIndirectTestHarness::new() {
        Some(h) => h,
        None => {
            eprintln!("Skipping GPU test: no adapter available");
            return;
        }
    };

    // 10 objects, but max_draws = 5
    let compacted_indices: Vec<u32> = (0..10).collect();
    let object_data: Vec<ObjectData> = (0..10).map(|_| make_object(0)).collect();
    let mesh_data = vec![make_simple_mesh(100, 0, 0)];
    let lod_entries: Vec<LodEntry> = (0..10).map(|_| make_lod_entry(0)).collect();

    let buffers = harness.create_test_buffers(
        &compacted_indices,
        &object_data,
        &mesh_data,
        &lod_entries,
        5, // Limit to 5 draws
    );

    let (draw_count, commands) = harness.dispatch_and_read(&buffers, 10, 5, false);

    // Draw count should be clamped to max_draws
    assert!(
        draw_count <= 5,
        "Draw count {} should be <= max_draws (5)",
        draw_count
    );
    assert!(commands.len() <= 5);
}

// =============================================================================
// TEST 9: NON-SEQUENTIAL COMPACTED INDICES
// =============================================================================

#[test]
fn test_non_sequential_compacted_indices() {
    let harness = match BuildIndirectTestHarness::new() {
        Some(h) => h,
        None => {
            eprintln!("Skipping GPU test: no adapter available");
            return;
        }
    };

    // Compacted indices are not sequential (sparse visibility)
    let compacted_indices = vec![2u32, 5, 7];
    let object_data: Vec<ObjectData> = (0..10).map(|i| make_object((i % 3) as u32)).collect();
    let mesh_data = vec![
        make_simple_mesh(100, 0, 0),
        make_simple_mesh(200, 100, 0),
        make_simple_mesh(300, 300, 0),
    ];
    let lod_entries: Vec<LodEntry> = (0..10).map(|_| make_lod_entry(0)).collect();

    let buffers = harness.create_test_buffers(
        &compacted_indices,
        &object_data,
        &mesh_data,
        &lod_entries,
        1024,
    );

    let (draw_count, commands) = harness.dispatch_and_read(&buffers, 3, 1024, false);

    assert_eq!(draw_count, 3, "Draw count should be 3");
    assert_eq!(commands.len(), 3);

    // Object 2 uses mesh 2%3=2, Object 5 uses mesh 5%3=2, Object 7 uses mesh 7%3=1
    // Mesh 1: 200 indices, Mesh 2: 300 indices
    let mut found_mesh1 = false;
    let mut found_mesh2_count = 0;

    for cmd in &commands {
        if cmd.index_count == 200 {
            found_mesh1 = true;
        } else if cmd.index_count == 300 {
            found_mesh2_count += 1;
        }
    }

    assert!(found_mesh1, "Should have a draw for mesh 1 (200 indices)");
    assert_eq!(found_mesh2_count, 2, "Should have 2 draws for mesh 2 (300 indices)");
}

// =============================================================================
// TEST 10: NEGATIVE BASE VERTEX
// =============================================================================

#[test]
fn test_negative_base_vertex() {
    let harness = match BuildIndirectTestHarness::new() {
        Some(h) => h,
        None => {
            eprintln!("Skipping GPU test: no adapter available");
            return;
        }
    };

    let compacted_indices = vec![0u32];
    let object_data = vec![make_object(0)];
    let mesh_data = vec![make_simple_mesh(100, 0, -50)]; // Negative base vertex
    let lod_entries = vec![make_lod_entry(0)];

    let buffers = harness.create_test_buffers(
        &compacted_indices,
        &object_data,
        &mesh_data,
        &lod_entries,
        1024,
    );

    let (draw_count, commands) = harness.dispatch_and_read(&buffers, 1, 1024, false);

    assert_eq!(draw_count, 1);
    assert_eq!(commands.len(), 1);
    assert_eq!(commands[0].base_vertex, -50, "Base vertex should be -50");
}

// =============================================================================
// TEST 11: LARGE OBJECT COUNT (STANDARD MODE)
// =============================================================================

#[test]
fn test_large_object_count_standard_mode() {
    let harness = match BuildIndirectTestHarness::new() {
        Some(h) => h,
        None => {
            eprintln!("Skipping GPU test: no adapter available");
            return;
        }
    };

    // 1000 objects (below batched mode threshold)
    let count = 1000u32;
    let compacted_indices: Vec<u32> = (0..count).collect();
    let object_data: Vec<ObjectData> = (0..count)
        .map(|i| make_object((i % 4) as u32))
        .collect();
    let mesh_data = vec![
        make_simple_mesh(100, 0, 0),
        make_simple_mesh(200, 100, 0),
        make_simple_mesh(300, 300, 0),
        make_simple_mesh(400, 600, 0),
    ];
    let lod_entries: Vec<LodEntry> = (0..count)
        .map(|_| make_lod_entry(0))
        .collect();

    let buffers = harness.create_test_buffers(
        &compacted_indices,
        &object_data,
        &mesh_data,
        &lod_entries,
        count,
    );

    let params = BuildIndirectParams::new(count, count);
    assert!(!params.use_batched_mode(), "1000 objects should use standard mode");

    let (draw_count, commands) = harness.dispatch_and_read(&buffers, count, count, false);

    assert_eq!(draw_count, count, "Draw count should match object count");
    assert_eq!(commands.len() as u32, count);

    // Verify we have draws for all mesh types
    let mesh_counts: Vec<usize> = (0..4)
        .map(|mesh_id| {
            let expected_idx_count = (mesh_id + 1) * 100;
            commands.iter().filter(|c| c.index_count == expected_idx_count).count()
        })
        .collect();

    assert_eq!(mesh_counts[0], 250, "Should have 250 draws for mesh 0");
    assert_eq!(mesh_counts[1], 250, "Should have 250 draws for mesh 1");
    assert_eq!(mesh_counts[2], 250, "Should have 250 draws for mesh 2");
    assert_eq!(mesh_counts[3], 250, "Should have 250 draws for mesh 3");
}

// =============================================================================
// TEST 12: BATCHED MODE (>10K OBJECTS)
// =============================================================================

#[test]
fn test_batched_mode_large_count() {
    let harness = match BuildIndirectTestHarness::new() {
        Some(h) => h,
        None => {
            eprintln!("Skipping GPU test: no adapter available");
            return;
        }
    };

    // 15000 objects (above batched mode threshold)
    let count = 15000u32;
    let compacted_indices: Vec<u32> = (0..count).collect();
    let object_data: Vec<ObjectData> = (0..count)
        .map(|_| make_object(0))
        .collect();
    let mesh_data = vec![make_simple_mesh(100, 0, 0)];
    let lod_entries: Vec<LodEntry> = (0..count)
        .map(|_| make_lod_entry(0))
        .collect();

    let buffers = harness.create_test_buffers(
        &compacted_indices,
        &object_data,
        &mesh_data,
        &lod_entries,
        count,
    );

    let params = BuildIndirectParams::new(count, count);
    assert!(params.use_batched_mode(), "15000 objects should use batched mode");

    let (draw_count, commands) = harness.dispatch_and_read(&buffers, count, count, true);

    assert_eq!(draw_count, count, "Draw count should match object count");
    assert_eq!(commands.len() as u32, count);

    // Verify all commands have correct index count
    for cmd in &commands {
        assert_eq!(cmd.index_count, 100);
        assert_eq!(cmd.instance_count, 1);
    }
}

// =============================================================================
// TEST 13: CPU/GPU PARITY
// =============================================================================

#[test]
fn test_cpu_gpu_parity() {
    let harness = match BuildIndirectTestHarness::new() {
        Some(h) => h,
        None => {
            eprintln!("Skipping GPU test: no adapter available");
            return;
        }
    };

    // Setup: 5 objects with mixed LODs and meshes
    let compacted_indices = vec![0u32, 2, 4];
    let object_mesh_indices = vec![0u32, 1, 0, 2, 1]; // Objects 0,2,4 use meshes 0,0,1
    let lod_levels = vec![0u32, 1, 2, 0, 1]; // Objects 0,2,4 use LODs 0,2,1

    let object_data: Vec<ObjectData> = object_mesh_indices
        .iter()
        .map(|&mesh_id| make_object(mesh_id))
        .collect();

    let mesh_data = vec![
        make_lod_mesh(1000, 0, 0, &[1000, 500, 250, 125], &[0, 1000, 1500, 1750]),
        make_lod_mesh(800, 2000, 100, &[800, 400, 200, 100], &[0, 800, 1200, 1400]),
        make_simple_mesh(600, 3000, 200),
    ];

    let lod_entries: Vec<LodEntry> = lod_levels
        .iter()
        .map(|&level| make_lod_entry(level))
        .collect();

    // CPU reference
    let cpu_commands = cpu_build_indirect(
        &compacted_indices,
        &object_mesh_indices,
        &lod_levels,
        &mesh_data,
    );

    // GPU execution
    let buffers = harness.create_test_buffers(
        &compacted_indices,
        &object_data,
        &mesh_data,
        &lod_entries,
        1024,
    );

    let (draw_count, gpu_commands) = harness.dispatch_and_read(&buffers, 3, 1024, false);

    assert_eq!(draw_count, 3);
    assert_eq!(gpu_commands.len(), cpu_commands.len());

    // Compare CPU and GPU results (order may differ due to atomics)
    for cpu_cmd in &cpu_commands {
        let matching_gpu = gpu_commands.iter().find(|gpu_cmd| {
            gpu_cmd.index_count == cpu_cmd.index_count
                && gpu_cmd.first_index == cpu_cmd.first_index
                && gpu_cmd.base_vertex == cpu_cmd.base_vertex
                && gpu_cmd.instance_count == cpu_cmd.instance_count
        });

        assert!(
            matching_gpu.is_some(),
            "CPU command {:?} not found in GPU output. GPU commands: {:?}",
            cpu_cmd,
            gpu_commands
        );
    }
}

// =============================================================================
// TEST 14: WORKGROUP CALCULATION
// =============================================================================

#[test]
fn test_workgroup_calculation() {
    // Standard mode: ceil(visible_count / 64)
    let params = BuildIndirectParams::new(1, 1024);
    assert_eq!(params.workgroups(), 1);

    let params = BuildIndirectParams::new(64, 1024);
    assert_eq!(params.workgroups(), 1);

    let params = BuildIndirectParams::new(65, 1024);
    assert_eq!(params.workgroups(), 2);

    let params = BuildIndirectParams::new(1000, 1024);
    assert_eq!(params.workgroups(), 16);

    // Batched mode: ceil(visible_count / (64 * 4))
    let params = BuildIndirectParams::new(256, 1024);
    assert_eq!(params.workgroups_batched(), 1);

    let params = BuildIndirectParams::new(257, 1024);
    assert_eq!(params.workgroups_batched(), 2);

    let params = BuildIndirectParams::new(1000, 1024);
    assert_eq!(params.workgroups_batched(), 4);
}

// =============================================================================
// TEST 15: BATCHED MODE THRESHOLD
// =============================================================================

#[test]
fn test_batched_mode_threshold() {
    assert!(!BuildIndirectParams::new(5000, 4096).use_batched_mode());
    assert!(!BuildIndirectParams::new(10000, 4096).use_batched_mode());
    assert!(BuildIndirectParams::new(10001, 4096).use_batched_mode());
    assert!(BuildIndirectParams::new(50000, 65536).use_batched_mode());
}

// =============================================================================
// TEST 16: MEMORY LAYOUT VERIFICATION
// =============================================================================

#[test]
fn test_struct_sizes_match_gpu_layout() {
    // BuildIndirectParams: 16 bytes
    assert_eq!(
        mem::size_of::<BuildIndirectParams>(),
        BUILD_INDIRECT_PARAMS_SIZE,
        "BuildIndirectParams size mismatch"
    );
    assert_eq!(mem::size_of::<BuildIndirectParams>(), 16);

    // MeshData: 48 bytes
    assert_eq!(
        mem::size_of::<MeshData>(),
        MESH_DATA_SIZE,
        "MeshData size mismatch"
    );
    assert_eq!(mem::size_of::<MeshData>(), 48);

    // IndirectDrawIndexedArgs: 20 bytes
    assert_eq!(
        mem::size_of::<IndirectDrawIndexedArgs>(),
        DRAW_INDEXED_INDIRECT_ARGS_SIZE,
        "IndirectDrawIndexedArgs size mismatch"
    );
    assert_eq!(mem::size_of::<IndirectDrawIndexedArgs>(), 20);

    // ObjectData: 144 bytes
    assert_eq!(
        mem::size_of::<ObjectData>(),
        OBJECT_DATA_SIZE,
        "ObjectData size mismatch"
    );
    assert_eq!(mem::size_of::<ObjectData>(), 144);

    // LodEntry: 8 bytes
    assert_eq!(
        mem::size_of::<LodEntry>(),
        LOD_ENTRY_SIZE,
        "LodEntry size mismatch"
    );
    assert_eq!(mem::size_of::<LodEntry>(), 8);
}

// =============================================================================
// TEST 17: INDIRECT DRAW ARGS FIELD LAYOUT
// =============================================================================

#[test]
fn test_indirect_draw_args_field_layout() {
    let args = IndirectDrawIndexedArgs::new(100, 1, 200, -50, 42);
    let bytes: &[u8] = bytemuck::bytes_of(&args);

    // Verify field offsets match wgpu DrawIndexedIndirectArgs
    // index_count at offset 0
    assert_eq!(
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        100,
        "index_count at wrong offset"
    );

    // instance_count at offset 4
    assert_eq!(
        u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        1,
        "instance_count at wrong offset"
    );

    // first_index at offset 8
    assert_eq!(
        u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
        200,
        "first_index at wrong offset"
    );

    // base_vertex at offset 12 (signed)
    assert_eq!(
        i32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]),
        -50,
        "base_vertex at wrong offset"
    );

    // first_instance at offset 16
    assert_eq!(
        u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]),
        42,
        "first_instance at wrong offset"
    );
}

// =============================================================================
// TEST 18: LOD CLAMPING
// =============================================================================

#[test]
fn test_lod_level_clamping() {
    let harness = match BuildIndirectTestHarness::new() {
        Some(h) => h,
        None => {
            eprintln!("Skipping GPU test: no adapter available");
            return;
        }
    };

    // Request LOD 10 (out of range), should be clamped to MAX_LOD_LEVEL (3)
    let compacted_indices = vec![0u32];
    let object_data = vec![make_object(0)];
    let mesh_data = vec![make_lod_mesh(
        1000, 0, 0,
        &[1000, 500, 250, 125],
        &[0, 1000, 1500, 1750],
    )];
    let lod_entries = vec![LodEntry { level: 10, blend_factor: 0.0 }];

    let buffers = harness.create_test_buffers(
        &compacted_indices,
        &object_data,
        &mesh_data,
        &lod_entries,
        1024,
    );

    let (draw_count, commands) = harness.dispatch_and_read(&buffers, 1, 1024, false);

    assert_eq!(draw_count, 1);
    assert_eq!(commands.len(), 1);

    // Should use LOD 3 (clamped): 125 indices at offset 1750
    assert_eq!(commands[0].index_count, 125, "Should clamp to LOD 3");
    assert_eq!(commands[0].first_index, 1750, "Should use LOD 3 offset");
}

// =============================================================================
// TEST 19: OBJECT INDEX PASSED VIA FIRST_INSTANCE
// =============================================================================

#[test]
fn test_first_instance_contains_object_index() {
    let harness = match BuildIndirectTestHarness::new() {
        Some(h) => h,
        None => {
            eprintln!("Skipping GPU test: no adapter available");
            return;
        }
    };

    // Use non-sequential object indices to verify first_instance mapping
    let compacted_indices = vec![5u32, 10, 15];
    let object_data: Vec<ObjectData> = (0..20).map(|_| make_object(0)).collect();
    let mesh_data = vec![make_simple_mesh(100, 0, 0)];
    let lod_entries: Vec<LodEntry> = (0..20).map(|_| make_lod_entry(0)).collect();

    let buffers = harness.create_test_buffers(
        &compacted_indices,
        &object_data,
        &mesh_data,
        &lod_entries,
        1024,
    );

    let (draw_count, commands) = harness.dispatch_and_read(&buffers, 3, 1024, false);

    assert_eq!(draw_count, 3);
    assert_eq!(commands.len(), 3);

    // Verify first_instance contains original object indices (5, 10, 15)
    let first_instances: Vec<u32> = commands.iter().map(|c| c.first_instance).collect();

    assert!(
        first_instances.contains(&5),
        "first_instance should contain object index 5"
    );
    assert!(
        first_instances.contains(&10),
        "first_instance should contain object index 10"
    );
    assert!(
        first_instances.contains(&15),
        "first_instance should contain object index 15"
    );
}

// =============================================================================
// TEST 20: IS_VISIBLE CHECK
// =============================================================================

#[test]
fn test_indirect_draw_args_is_visible() {
    // Visible draw
    let visible = IndirectDrawIndexedArgs::new(100, 1, 0, 0, 0);
    assert!(visible.is_visible());

    // Zero index count
    let zero_indices = IndirectDrawIndexedArgs::new(0, 1, 0, 0, 0);
    assert!(!zero_indices.is_visible());

    // Zero instance count
    let zero_instances = IndirectDrawIndexedArgs::new(100, 0, 0, 0, 0);
    assert!(!zero_instances.is_visible());

    // Both zero
    let both_zero = IndirectDrawIndexedArgs::new(0, 0, 0, 0, 0);
    assert!(!both_zero.is_visible());
}

// =============================================================================
// TEST 21: MESH DATA LOD HELPERS
// =============================================================================

#[test]
fn test_mesh_data_lod_helpers() {
    let mesh = MeshData::with_lods(
        1000, 100, 50,
        &[1000, 500, 250, 125],
        &[0, 1000, 1500, 1750],
    );

    // LOD 0 (highest detail)
    assert_eq!(mesh.index_count_for_lod(0), 1000);
    assert_eq!(mesh.first_index_for_lod(0), 100); // first_index + offset[0] = 100 + 0

    // LOD 1
    assert_eq!(mesh.index_count_for_lod(1), 500);
    assert_eq!(mesh.first_index_for_lod(1), 1100); // 100 + 1000

    // LOD 2
    assert_eq!(mesh.index_count_for_lod(2), 250);
    assert_eq!(mesh.first_index_for_lod(2), 1600); // 100 + 1500

    // LOD 3 (lowest detail)
    assert_eq!(mesh.index_count_for_lod(3), 125);
    assert_eq!(mesh.first_index_for_lod(3), 1850); // 100 + 1750

    // Out of range LOD should clamp to MAX
    assert_eq!(mesh.index_count_for_lod(10), 125);
    assert_eq!(mesh.first_index_for_lod(10), 1850);
}

// =============================================================================
// TEST 22: CPU REFERENCE IMPLEMENTATION
// =============================================================================

#[test]
fn test_cpu_build_indirect_basic() {
    let compacted_indices = vec![0u32, 1, 2];
    let object_mesh_indices = vec![0u32, 0, 1];
    let lod_levels = vec![0u32, 1, 0];
    let mesh_data = vec![
        MeshData::with_lods(1000, 0, 0, &[1000, 500, 0, 0], &[0, 1000, 0, 0]),
        MeshData::new(600, 2000, 100),
    ];

    let commands = cpu_build_indirect(
        &compacted_indices,
        &object_mesh_indices,
        &lod_levels,
        &mesh_data,
    );

    assert_eq!(commands.len(), 3);

    // Object 0: mesh 0, LOD 0 -> 1000 indices at offset 0
    assert_eq!(commands[0].index_count, 1000);
    assert_eq!(commands[0].first_index, 0);
    assert_eq!(commands[0].first_instance, 0); // object_idx from compacted[0]

    // Object 1: mesh 0, LOD 1 -> 500 indices at offset 1000
    assert_eq!(commands[1].index_count, 500);
    assert_eq!(commands[1].first_index, 1000);
    assert_eq!(commands[1].first_instance, 1); // object_idx from compacted[1]

    // Object 2: mesh 1, LOD 0 -> 600 indices at offset 2000
    assert_eq!(commands[2].index_count, 600);
    assert_eq!(commands[2].first_index, 2000);
    assert_eq!(commands[2].base_vertex, 100);
    assert_eq!(commands[2].first_instance, 2); // object_idx from compacted[2]
}

// =============================================================================
// TEST 23: BOUNDARY VALUES
// =============================================================================

#[test]
fn test_boundary_values() {
    // Test with maximum u32 index values
    let mesh = MeshData::with_lods(
        u32::MAX,
        u32::MAX - 1000,
        i32::MIN,
        &[u32::MAX, u32::MAX / 2, u32::MAX / 4, u32::MAX / 8],
        &[0, 100, 200, 300],
    );

    assert_eq!(mesh.index_count_for_lod(0), u32::MAX);
    assert_eq!(mesh.first_index_for_lod(0), u32::MAX - 1000);

    // Verify no overflow in offset calculation
    // first_index + lod_first_index[3] should not overflow
    assert_eq!(mesh.first_index_for_lod(3), u32::MAX - 1000 + 300);

    // Test with maximum LOD level
    let lod = LodEntry { level: u32::MAX, blend_factor: 1.0 };
    assert_eq!(lod.level, u32::MAX);
}

// =============================================================================
// TEST 24: DEFAULT IMPLEMENTATIONS
// =============================================================================

#[test]
fn test_default_implementations() {
    let default_params = BuildIndirectParams::default();
    assert_eq!(default_params.visible_count, 0);
    assert_eq!(default_params.max_draws, 0);

    let default_mesh = MeshData::default();
    assert_eq!(default_mesh.index_count, 0);
    assert_eq!(default_mesh.first_index, 0);
    assert_eq!(default_mesh.base_vertex, 0);
    assert_eq!(default_mesh.lod_index_counts, [0; MAX_LOD_LEVELS]);
    assert_eq!(default_mesh.lod_first_index, [0; MAX_LOD_LEVELS]);

    let default_lod = LodEntry::default();
    assert_eq!(default_lod.level, 0);
    assert_eq!(default_lod.blend_factor, 0.0);

    let default_args = IndirectDrawIndexedArgs::default();
    assert_eq!(default_args.index_count, 0);
    assert_eq!(default_args.instance_count, 0);
    assert!(!default_args.is_visible());
}

// =============================================================================
// TEST 25: CONSTANTS VERIFICATION
// =============================================================================

#[test]
fn test_constants() {
    assert_eq!(WORKGROUP_SIZE, 64);
    assert_eq!(BATCH_SIZE, 4);
    assert_eq!(MAX_LOD_LEVELS, 4);
    assert_eq!(BUILD_INDIRECT_PARAMS_SIZE, 16);
    assert_eq!(MESH_DATA_SIZE, 48);
    assert_eq!(DRAW_INDEXED_INDIRECT_ARGS_SIZE, 20);
}
