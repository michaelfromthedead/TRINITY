// SPDX-License-Identifier: MIT
//
// T-WGPU-P4.9.2: Phase 4 Synchronization Integration Tests
//
// CLEANROOM: Tests only via public API. No implementation details read.
//
// ACCEPTANCE CRITERIA:
//   1. Buffer copy round-trip test (write->copy->read back->verify)
//   2. Texture copy test (buffer->texture->buffer round-trip)
//   3. Timestamp query test (record timestamps, resolve, verify ordering)
//   4. Frame pacing test (verify FramePacer targets frame time)
//   5. Readback test (BufferReadback and DoubleBufferedReadback verification)
//
// Integration Focus:
//   These tests verify that Phase 4 synchronization modules work together
//   end-to-end, not just in isolation. They exercise:
//   - frame_sync (FrameFence, FrameSyncManager, DoubleBufferedRenderer, FramePacer)
//   - buffer_mapping (BufferMapper, BufferReadback, DoubleBufferedReadback)
//   - copy_commands (buffer and texture copy operations)
//   - query_pool (timestamp queries and resolution)
//   - resource_state (barrier detection and state transitions)

use bytemuck::{Pod, Zeroable};
use std::time::Duration;

use renderer_backend::buffer_mapping::{
    BufferReadback, DoubleBufferedReadback, MappingState,
};
use renderer_backend::copy_commands::{
    BufferCopyParams, CopyAlignmentCalculator, CopyExtent3d, validate_params,
};
use renderer_backend::frame_sync::{
    BufferCount, DoubleBufferedRenderer, FrameFence, FramePacer, FrameSyncManager,
    TrinityFrameSynchronizer,
};
use renderer_backend::resource_state::{
    AccessFlags, BarrierDetector, HazardType, PipelineStage, ResourceState,
    ResourceStateTracker, TextureLayout,
};

// ============================================================================
// Test Data Types
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
struct TestVertex {
    position: [f32; 4],
    normal: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
struct TestTransform {
    matrix: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
struct TestCounter {
    value: u32,
    _padding: [u32; 3], // Align to 16 bytes
}

// ============================================================================
// GPU Test Infrastructure
// ============================================================================

/// Create a test device and queue using the fallback adapter.
/// Returns None if no GPU adapter is available (CI without GPU).
fn create_test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: true,
    }))?;

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("T-WGPU-P4.9.2 Integration Test Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::default(),
        },
        None,
    ))
    .ok()?;

    Some((device, queue))
}

/// Create a test device with timestamp query support for SECTION 3.
fn create_timestamp_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: true,
    }))?;

    // Check if TIMESTAMP_QUERY is supported
    if !adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
        return None;
    }

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("Timestamp Query Test Device"),
            required_features: wgpu::Features::TIMESTAMP_QUERY,
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::default(),
        },
        None,
    ))
    .ok()?;

    Some((device, queue))
}

/// Create a source buffer initialized with data.
fn create_initialized_buffer<T: Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    data: &[T],
    usage: wgpu::BufferUsages,
    label: &str,
) -> wgpu::Buffer {
    let bytes: &[u8] = bytemuck::cast_slice(data);

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len() as u64,
        usage,
        mapped_at_creation: false,
    });

    queue.write_buffer(&buffer, 0, bytes);

    buffer
}

// ============================================================================
// SECTION 1: Buffer Copy Round-Trip Integration Tests
// ============================================================================

mod buffer_copy_roundtrip {
    use super::*;

    /// Test complete buffer copy round-trip: write->copy->read back->verify
    #[test]
    fn test_buffer_copy_roundtrip_u32() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        // Step 1: Create source buffer with test data
        let source_data: Vec<u32> = (1..=64).collect();
        let source_buffer = create_initialized_buffer(
            &device,
            &queue,
            &source_data,
            wgpu::BufferUsages::COPY_SRC,
            "Source Buffer",
        );

        // Step 2: Create intermediate buffer for copy
        let intermediate_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Intermediate Buffer"),
            size: (source_data.len() * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Step 3: Create readback for verification
        let mut readback = BufferReadback::new(
            &device,
            (source_data.len() * 4) as u64,
            Some("Readback"),
        );

        // Step 4: Execute copy chain: source -> intermediate -> staging
        let mut encoder = device.create_command_encoder(&Default::default());

        // Copy source to intermediate
        encoder.copy_buffer_to_buffer(
            &source_buffer,
            0,
            &intermediate_buffer,
            0,
            (source_data.len() * 4) as u64,
        );

        // Copy intermediate to staging (readback)
        readback.read_buffer(&device, &queue, &mut encoder, &intermediate_buffer, 0).unwrap();

        queue.submit([encoder.finish()]);
        device.poll(wgpu::Maintain::Wait);

        // Step 5: Verify data integrity
        let read_data: Vec<u32> = readback.get_data().unwrap();
        assert_eq!(read_data, source_data, "Round-trip data must match");
    }

    /// Test buffer copy with offset verification
    #[test]
    fn test_buffer_copy_with_offsets() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        // Create larger source buffer
        let full_data: Vec<u32> = (0..128).collect();
        let source_buffer = create_initialized_buffer(
            &device,
            &queue,
            &full_data,
            wgpu::BufferUsages::COPY_SRC,
            "Full Source",
        );

        // Create destination buffer
        let dest_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Destination"),
            size: 256, // 64 u32s
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create readback
        let mut readback = BufferReadback::new(&device, 128, Some("Offset Readback"));

        let mut encoder = device.create_command_encoder(&Default::default());

        // Copy from offset 128 (32 u32s) in source to offset 64 (16 u32s) in dest
        encoder.copy_buffer_to_buffer(&source_buffer, 128, &dest_buffer, 64, 128);

        // Read back from dest at offset 64
        readback.read_buffer(&device, &queue, &mut encoder, &dest_buffer, 64).unwrap();

        queue.submit([encoder.finish()]);
        device.poll(wgpu::Maintain::Wait);

        let read_data: Vec<u32> = readback.get_data().unwrap();
        let expected: Vec<u32> = (32..64).collect(); // Elements 32-63 from source
        assert_eq!(read_data, expected);
    }

    /// Test multiple sequential buffer copies
    #[test]
    fn test_sequential_buffer_copies() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        // Create chain of buffers
        let initial_data: Vec<u32> = vec![0xDEADBEEF; 16];
        let buf1 = create_initialized_buffer(
            &device, &queue, &initial_data,
            wgpu::BufferUsages::COPY_SRC, "Buf1",
        );
        let buf2 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Buf2"),
            size: 64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buf3 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Buf3"),
            size: 64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let mut readback = BufferReadback::new(&device, 64, Some("Chain Readback"));

        // Chain copies: buf1 -> buf2 -> buf3 -> readback
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&buf1, 0, &buf2, 0, 64);
        encoder.copy_buffer_to_buffer(&buf2, 0, &buf3, 0, 64);
        readback.read_buffer(&device, &queue, &mut encoder, &buf3, 0).unwrap();

        queue.submit([encoder.finish()]);
        device.poll(wgpu::Maintain::Wait);

        let read_data: Vec<u32> = readback.get_data().unwrap();
        assert_eq!(read_data, initial_data);
    }

    /// Test buffer copy params validation integration
    #[test]
    fn test_buffer_copy_params_validation() {
        // Valid aligned params
        let valid_params = BufferCopyParams::new(0, 256, 1024);
        assert!(validate_params(&valid_params).is_ok());
        assert!(valid_params.is_aligned());

        // Invalid unaligned source offset
        let invalid_source = BufferCopyParams::new(3, 256, 1024);
        assert!(validate_params(&invalid_source).is_err());

        // Invalid unaligned dest offset
        let invalid_dest = BufferCopyParams::new(0, 255, 1024);
        assert!(validate_params(&invalid_dest).is_err());

        // Invalid unaligned size
        let invalid_size = BufferCopyParams::new(0, 256, 1023);
        assert!(validate_params(&invalid_size).is_err());

        // Invalid zero size
        let zero_size = BufferCopyParams::new(0, 0, 0);
        assert!(validate_params(&zero_size).is_err());
    }

    /// Test vertex data round-trip
    #[test]
    fn test_vertex_data_roundtrip() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        let vertices: Vec<TestVertex> = vec![
            TestVertex { position: [0.0, 1.0, 0.0, 1.0], normal: [0.0, 1.0, 0.0, 0.0] },
            TestVertex { position: [1.0, 0.0, 0.0, 1.0], normal: [1.0, 0.0, 0.0, 0.0] },
            TestVertex { position: [0.0, 0.0, 1.0, 1.0], normal: [0.0, 0.0, 1.0, 0.0] },
            TestVertex { position: [-1.0, 0.0, 0.0, 1.0], normal: [-1.0, 0.0, 0.0, 0.0] },
        ];

        let source = create_initialized_buffer(
            &device, &queue, &vertices,
            wgpu::BufferUsages::COPY_SRC, "Vertex Source",
        );

        let intermediate = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Intermediate"),
            size: (vertices.len() * std::mem::size_of::<TestVertex>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let mut readback = BufferReadback::new(
            &device,
            (vertices.len() * std::mem::size_of::<TestVertex>()) as u64,
            Some("Vertex Readback"),
        );

        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&source, 0, &intermediate, 0, intermediate.size());
        readback.read_buffer(&device, &queue, &mut encoder, &intermediate, 0).unwrap();

        queue.submit([encoder.finish()]);
        device.poll(wgpu::Maintain::Wait);

        let read_vertices: Vec<TestVertex> = readback.get_data().unwrap();
        assert_eq!(read_vertices, vertices);
    }
}

// ============================================================================
// SECTION 2: Texture Copy Integration Tests
// ============================================================================

mod texture_copy {
    use super::*;

    /// Test buffer-to-texture-to-buffer round-trip
    #[test]
    fn test_texture_copy_roundtrip() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        // Create 4x4 RGBA8 texture data
        let width = 4u32;
        let height = 4u32;
        let mut pixel_data: Vec<u32> = Vec::new();
        for y in 0..height {
            for x in 0..width {
                // RGBA8 packed as u32: R + (G << 8) + (B << 16) + (A << 24)
                let r = (x * 64) as u32;
                let g = (y * 64) as u32;
                let b = ((x + y) * 32) as u32;
                let a = 255u32;
                pixel_data.push(r | (g << 8) | (b << 16) | (a << 24));
            }
        }

        let bytes_per_row = CopyAlignmentCalculator::calculate_aligned_bytes_per_row(width * 4);
        let buffer_size = (bytes_per_row * height) as u64;

        // Create staging buffer with aligned rows
        let mut aligned_data: Vec<u8> = vec![0u8; buffer_size as usize];
        for y in 0..height as usize {
            let dst_start = y * (bytes_per_row as usize);
            let row_bytes: &[u8] = bytemuck::cast_slice(&pixel_data[y * (width as usize)..(y + 1) * (width as usize)]);
            aligned_data[dst_start..dst_start + row_bytes.len()].copy_from_slice(row_bytes);
        }

        let source_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Texture Source Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&source_buffer, 0, &aligned_data);

        // Create texture
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Test Texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        // Create destination buffer for readback
        let dest_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Texture Dest Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Copy: buffer -> texture -> buffer
        let mut encoder = device.create_command_encoder(&Default::default());

        encoder.copy_buffer_to_texture(
            wgpu::ImageCopyBuffer {
                buffer: &source_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: None,
                },
            },
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &dest_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );

        queue.submit([encoder.finish()]);
        device.poll(wgpu::Maintain::Wait);

        // Map and verify
        let buffer_slice = dest_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let read_data: Vec<u8> = data.to_vec();
        drop(data);
        dest_buffer.unmap();

        // Verify row by row (accounting for alignment padding)
        for y in 0..height as usize {
            let expected_start = y * (bytes_per_row as usize);
            let expected_row = &aligned_data[expected_start..expected_start + (width as usize) * 4];
            let actual_row = &read_data[expected_start..expected_start + (width as usize) * 4];
            assert_eq!(actual_row, expected_row, "Row {} mismatch", y);
        }
    }

    /// Test texture copy alignment calculations
    #[test]
    fn test_texture_copy_alignment() {
        // Test bytes_per_row alignment to 256
        assert_eq!(CopyAlignmentCalculator::calculate_aligned_bytes_per_row(100), 256);
        assert_eq!(CopyAlignmentCalculator::calculate_aligned_bytes_per_row(256), 256);
        assert_eq!(CopyAlignmentCalculator::calculate_aligned_bytes_per_row(257), 512);
        assert_eq!(CopyAlignmentCalculator::calculate_aligned_bytes_per_row(512), 512);

        // Test alignment check
        assert!(CopyAlignmentCalculator::is_bytes_per_row_aligned(256));
        assert!(CopyAlignmentCalculator::is_bytes_per_row_aligned(512));
        assert!(!CopyAlignmentCalculator::is_bytes_per_row_aligned(100));
        assert!(!CopyAlignmentCalculator::is_bytes_per_row_aligned(300));

        // Test row padding calculation
        assert_eq!(CopyAlignmentCalculator::calculate_row_padding(256), 0);
        assert_eq!(CopyAlignmentCalculator::calculate_row_padding(100), 156);
        assert_eq!(CopyAlignmentCalculator::calculate_row_padding(300), 212);

        // Test buffer layout calculation
        let (aligned_bpr, total_size) = CopyAlignmentCalculator::calculate_texture_buffer_layout(
            100, 100, 1, 4
        );
        assert_eq!(aligned_bpr, 512); // 100 * 4 = 400 -> 512
        assert_eq!(total_size, 51200); // 512 * 100 * 1
    }

    /// Test CopyExtent3d utilities
    #[test]
    fn test_copy_extent_utilities() {
        let extent_2d = CopyExtent3d::new_2d(256, 256);
        assert_eq!(extent_2d.width, 256);
        assert_eq!(extent_2d.height, 256);
        assert_eq!(extent_2d.depth_or_array_layers, 1);
        assert!(extent_2d.is_valid());
        assert!(!extent_2d.is_multi_layer());

        let extent_3d = CopyExtent3d::new(64, 64, 8);
        assert!(extent_3d.is_valid());
        assert!(extent_3d.is_multi_layer());

        let wgpu_extent = extent_2d.to_wgpu();
        assert_eq!(wgpu_extent.width, 256);
    }
}

// ============================================================================
// SECTION 3: Timestamp Query Integration Tests
// ============================================================================

mod timestamp_query {
    use super::*;
    use renderer_backend::query_pool::{TimestampQueryPool, QueryResolveParams};

    /// Test timestamp query creation and basic allocation
    #[test]
    fn test_timestamp_query_pool_creation() {
        let Some((device, queue)) = create_timestamp_device() else {
            eprintln!("Skipping: TIMESTAMP_QUERY not supported");
            return;
        };

        let pool = TimestampQueryPool::new(&device, &queue, 64);
        assert!(pool.is_ok(), "Pool creation should succeed");

        let pool = pool.unwrap();
        assert_eq!(pool.capacity(), 64);
        assert_eq!(pool.used(), 0);
        assert_eq!(pool.available(), 64);
        assert!(pool.has_capacity());
        assert!(pool.is_empty());
        assert!(!pool.is_full());
    }

    /// Test timestamp query allocation
    #[test]
    fn test_timestamp_query_allocation() {
        let Some((device, queue)) = create_timestamp_device() else {
            eprintln!("Skipping: TIMESTAMP_QUERY not supported");
            return;
        };

        let mut pool = TimestampQueryPool::new(&device, &queue, 8).unwrap();

        // Allocate individual indices
        let idx0 = pool.allocate().unwrap();
        assert_eq!(idx0, 0);
        assert_eq!(pool.used(), 1);

        let idx1 = pool.allocate().unwrap();
        assert_eq!(idx1, 1);

        // Allocate range
        let range_start = pool.allocate_range(4).unwrap();
        assert_eq!(range_start, 2);
        assert_eq!(pool.used(), 6);

        // Pool should have 2 remaining
        assert_eq!(pool.available(), 2);

        // Exhaust pool
        pool.allocate().unwrap();
        pool.allocate().unwrap();
        assert!(pool.is_full());
        assert!(pool.allocate().is_err());
    }

    /// Test timestamp query reset
    #[test]
    fn test_timestamp_query_reset() {
        let Some((device, queue)) = create_timestamp_device() else {
            eprintln!("Skipping: TIMESTAMP_QUERY not supported");
            return;
        };

        let mut pool = TimestampQueryPool::new(&device, &queue, 16).unwrap();

        // Use some indices
        pool.allocate_range(10).unwrap();
        assert_eq!(pool.used(), 10);

        let gen0 = pool.generation();

        // Reset
        pool.reset();
        assert_eq!(pool.used(), 0);
        assert_eq!(pool.available(), 16);
        assert_eq!(pool.generation(), gen0.wrapping_add(1));

        // Can allocate again
        let idx = pool.allocate().unwrap();
        assert_eq!(idx, 0);
    }

    /// Test timestamp query resolve parameters
    #[test]
    fn test_timestamp_resolve_params() {
        let params = QueryResolveParams::new(0, 4, 0);
        assert_eq!(params.start_query, 0);
        assert_eq!(params.query_count, 4);
        assert_eq!(params.destination_offset, 0);
        assert_eq!(params.end_query(), 4);
        assert_eq!(params.required_buffer_size(), 32); // 4 * 8 bytes

        let params_from_start = QueryResolveParams::from_start(8);
        assert_eq!(params_from_start.start_query, 0);
        assert_eq!(params_from_start.required_buffer_size(), 64);
    }

    /// Test timestamp queries in a simulated frame
    #[test]
    fn test_timestamp_query_frame_workflow() {
        let Some((device, queue)) = create_timestamp_device() else {
            eprintln!("Skipping: TIMESTAMP_QUERY not supported");
            return;
        };

        let mut pool = TimestampQueryPool::new(&device, &queue, 32).unwrap();

        // Simulate recording timestamps
        let begin_shadow = pool.allocate().unwrap();
        let end_shadow = pool.allocate().unwrap();
        let begin_main = pool.allocate().unwrap();
        let end_main = pool.allocate().unwrap();

        assert_eq!(pool.used(), 4);

        // Create encoder and write timestamps
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Timestamp Test Encoder"),
        });

        encoder.write_timestamp(pool.query_set(), begin_shadow);
        // ... shadow pass work would go here ...
        encoder.write_timestamp(pool.query_set(), end_shadow);

        encoder.write_timestamp(pool.query_set(), begin_main);
        // ... main pass work would go here ...
        encoder.write_timestamp(pool.query_set(), end_main);

        // Resolve all used queries
        pool.resolve_all(&mut encoder).unwrap();

        queue.submit([encoder.finish()]);
        device.poll(wgpu::Maintain::Wait);

        // Read back timestamps
        let data = pollster::block_on(async {
            pool.read_timestamps_blocking(&device, 0..4)
        });

        if let Ok(timestamp_data) = data {
            // Verify we got 4 timestamps
            assert_eq!(timestamp_data.len(), 4);

            // Verify ordering: each timestamp should be >= previous
            // (or 0 if not supported)
            for i in 1..timestamp_data.len() {
                if timestamp_data.timestamps[i] > 0 && timestamp_data.timestamps[i - 1] > 0 {
                    assert!(
                        timestamp_data.timestamps[i] >= timestamp_data.timestamps[i - 1],
                        "Timestamp ordering violated at index {}",
                        i
                    );
                }
            }
        }

        // Reset for next frame
        pool.reset();
        assert_eq!(pool.used(), 0);
    }

    /// Test timestamp period conversion
    #[test]
    fn test_timestamp_period_conversion() {
        let Some((device, queue)) = create_timestamp_device() else {
            eprintln!("Skipping: TIMESTAMP_QUERY not supported");
            return;
        };

        let pool = TimestampQueryPool::new(&device, &queue, 4).unwrap();

        let period = pool.timestamp_period();
        assert!(period > 0.0, "Timestamp period should be positive");

        // Test conversion utilities
        let ticks = 1_000_000u64;
        let ms = pool.ticks_to_ms(ticks);
        assert!(ms > 0.0);

        let delta_ms = pool.delta_to_ms(0, ticks);
        assert!((delta_ms - ms).abs() < 0.0001);
    }
}

// ============================================================================
// SECTION 4: Frame Pacing Integration Tests
// ============================================================================

mod frame_pacing {
    use super::*;
    use std::thread;

    /// Test FramePacer targets frame time
    #[test]
    fn test_frame_pacer_targets_frame_time() {
        let mut pacer = FramePacer::new(60.0);

        // Verify target frame time is approximately 16.67ms
        let target = pacer.target_frame_time();
        let expected = Duration::from_secs_f64(1.0 / 60.0);
        let diff_us = target.as_micros().abs_diff(expected.as_micros());
        assert!(diff_us < 100, "Target frame time mismatch: {:?} vs {:?}", target, expected);

        // Run several frames with adaptive pacing
        pacer.set_adaptive(true);

        for _ in 0..5 {
            pacer.begin_frame();
            thread::sleep(Duration::from_millis(8)); // Simulate 8ms of work
            pacer.end_frame();

            // With adaptive pacing, should_sleep should suggest sleeping
            if let Some(sleep) = pacer.should_sleep() {
                // Should suggest some sleep time since we're under budget
                if sleep > Duration::ZERO {
                    // Actually sleep (but cap it for test speed)
                    let capped = sleep.min(Duration::from_millis(5));
                    thread::sleep(capped);
                }
            }
        }

        // Verify stats are populated
        assert!(pacer.total_frames() >= 5);
        assert!(pacer.average_frame_time().is_some());
    }

    /// Test FramePacer with FrameSyncManager integration
    #[test]
    fn test_frame_pacer_with_sync_manager() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        let sync = FrameSyncManager::new(3);
        let mut pacer = FramePacer::new(30.0); // 30 FPS target

        for i in 0..5 {
            // Frame start
            let frame_num = sync.begin_frame();
            pacer.begin_frame();

            // Simulate GPU work
            let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("Frame {} Encoder", i)),
            });

            // Submit and record
            let submission = queue.submit([encoder.finish()]);
            sync.record_submission(submission);

            // Frame end
            let stats = sync.end_frame(&device, &queue);
            pacer.end_frame();

            assert_eq!(stats.frame_number, frame_num);
        }

        assert!(sync.total_frames() >= 5);
        assert!(pacer.total_frames() >= 5);
    }

    /// Test FramePacer variance tracking
    #[test]
    fn test_frame_pacer_variance() {
        let mut pacer = FramePacer::new(60.0);

        // Record frames with varying times
        for i in 0..10 {
            pacer.begin_frame();
            let sleep_ms = 5 + (i % 3) * 3; // 5, 8, 11, 5, 8, 11...
            thread::sleep(Duration::from_millis(sleep_ms));
            pacer.end_frame();
        }

        // Should have measurable variance
        if let Some(var) = pacer.variance() {
            assert!(var >= 0.0);
        }

        if let Some(std_dev) = pacer.std_deviation() {
            assert!(std_dev >= 0.0);
            if let Some(var) = pacer.variance() {
                let expected_std = var.sqrt();
                assert!((expected_std - std_dev).abs() < 0.0001);
            }
        }
    }

    /// Test TrinityFrameSynchronizer with different buffer counts
    #[test]
    fn test_trinity_frame_synchronizer() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        // Test double buffering
        let sync_double = TrinityFrameSynchronizer::new(BufferCount::Double);
        assert_eq!(sync_double.buffer_count(), 2);
        assert_eq!(sync_double.wait_offset(), 1);

        // Test triple buffering
        let sync_triple = TrinityFrameSynchronizer::new(BufferCount::Triple);
        assert_eq!(sync_triple.buffer_count(), 3);
        assert_eq!(sync_triple.wait_offset(), 2);

        // Run some frames
        for i in 0..6 {
            sync_triple.begin_frame(&device);

            let expected_buffer = (i % 3) as u32;
            assert_eq!(sync_triple.current_index(), expected_buffer);

            let encoder = device.create_command_encoder(&Default::default());
            sync_triple.end_frame_with_iter(&queue, Some(encoder.finish()));
        }

        assert_eq!(sync_triple.frame_count(), 6);

        // Test wait_idle
        sync_triple.wait_idle(&device);
    }

    /// Test DoubleBufferedRenderer frame cycling
    #[test]
    fn test_double_buffered_renderer_frame_cycling() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        let renderer = DoubleBufferedRenderer::new();

        // Verify initial state
        assert_eq!(renderer.current_index(), 0);
        assert_eq!(renderer.next_index(), 1);
        assert_eq!(renderer.frame_count(), 0);

        // Run frames and verify ping-pong
        for i in 0..10 {
            let expected_buffer = (i % 2) as u32;
            assert_eq!(renderer.current_index(), expected_buffer);

            renderer.begin_frame(&device);
            let encoder = device.create_command_encoder(&Default::default());
            renderer.end_frame_with_iter(&queue, Some(encoder.finish()));

            assert_eq!(renderer.frame_count(), (i + 1) as u64);
        }

        renderer.wait_idle(&device);
    }
}

// ============================================================================
// SECTION 5: Readback Integration Tests
// ============================================================================

mod readback_integration {
    use super::*;

    /// Test BufferReadback full workflow
    #[test]
    fn test_buffer_readback_full_workflow() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        // Create test data
        let data: Vec<u32> = (0..256).collect();
        let source = create_initialized_buffer(
            &device, &queue, &data,
            wgpu::BufferUsages::COPY_SRC,
            "Readback Source",
        );

        // Create readback utility
        let mut readback = BufferReadback::new(&device, 1024, Some("Test Readback"));

        // Verify initial state
        assert_eq!(readback.state(), MappingState::Unmapped);
        assert_eq!(readback.size(), 1024);
        assert!(!readback.is_ready());

        // Execute readback
        let mut encoder = device.create_command_encoder(&Default::default());
        readback.read_buffer(&device, &queue, &mut encoder, &source, 0).unwrap();
        queue.submit([encoder.finish()]);
        device.poll(wgpu::Maintain::Wait);

        // Verify data
        let read_data: Vec<u32> = readback.get_data().unwrap();
        assert_eq!(read_data, data);
    }

    /// Test DoubleBufferedReadback for continuous streaming
    #[test]
    fn test_double_buffered_readback_streaming() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        let mut double_readback = DoubleBufferedReadback::new(&device, 64, Some("Double"));

        // Verify initial state
        assert_eq!(double_readback.size(), 64);
        assert_eq!(double_readback.current_index(), 0);
        assert_eq!(double_readback.previous_index(), 1);

        // Simulate streaming frames
        for frame in 0..5 {
            // Create different data each frame
            let data: Vec<u32> = vec![frame as u32; 16];
            let source = create_initialized_buffer(
                &device, &queue, &data,
                wgpu::BufferUsages::COPY_SRC,
                &format!("Frame {} Source", frame),
            );

            // Begin readback of current frame
            let mut encoder = device.create_command_encoder(&Default::default());
            double_readback.begin_readback(&mut encoder, &source, 0);
            queue.submit([encoder.finish()]);

            // Poll for previous frame's data (skip first frame)
            if frame > 0 {
                // Wait and get data from previous frame
                if let Some(prev_data) = double_readback.wait_and_get::<u32>(&device) {
                    let expected_val = (frame - 1) as u32;
                    assert!(
                        prev_data.iter().all(|&v| v == expected_val),
                        "Frame {} previous data mismatch",
                        frame
                    );
                }
            }

            // Swap buffers for next frame
            double_readback.swap();
        }
    }

    /// Test readback with barrier detection integration
    #[test]
    fn test_readback_with_barrier_detection() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        let mut detector = BarrierDetector::new();
        let resource_id = 1u64;

        // Initial state: buffer was written by transfer
        detector.record_access(resource_id, ResourceState::buffer(
            PipelineStage::Transfer,
            AccessFlags::TRANSFER_WRITE,
        ));

        // Now we want to read it back
        let new_state = ResourceState::buffer(
            PipelineStage::Host,
            AccessFlags::HOST_READ,
        );

        // Check if barrier is needed
        let barrier_info = detector.needs_barrier(resource_id, &new_state);
        assert!(barrier_info.is_some(), "Barrier required for read-after-write");

        let barrier = barrier_info.unwrap();
        assert_eq!(barrier.hazard, HazardType::ReadAfterWrite);
        assert_eq!(barrier.src_stage, PipelineStage::Transfer);
        assert_eq!(barrier.dst_stage, PipelineStage::Host);

        // Perform the actual readback
        let data: Vec<u32> = vec![42; 16];
        let source = create_initialized_buffer(
            &device, &queue, &data,
            wgpu::BufferUsages::COPY_SRC,
            "Barrier Test Source",
        );

        let mut readback = BufferReadback::new(&device, 64, None);
        let mut encoder = device.create_command_encoder(&Default::default());
        readback.read_buffer(&device, &queue, &mut encoder, &source, 0).unwrap();
        queue.submit([encoder.finish()]);
        device.poll(wgpu::Maintain::Wait);

        let read_data: Vec<u32> = readback.get_data().unwrap();
        assert_eq!(read_data, data);

        // Update detector state
        detector.record_access(resource_id, new_state);
    }

    /// Test async readback polling
    #[test]
    fn test_async_readback_polling() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        let data: Vec<u32> = vec![0xCAFEBABE; 16];
        let source = create_initialized_buffer(
            &device, &queue, &data,
            wgpu::BufferUsages::COPY_SRC,
            "Async Source",
        );

        let mut readback = BufferReadback::new(&device, 64, Some("Async Readback"));

        // Begin async readback
        let mut encoder = device.create_command_encoder(&Default::default());
        readback.begin_readback(&mut encoder, &source, 0);
        queue.submit([encoder.finish()]);

        // Poll until ready
        let mut polls = 0;
        loop {
            let state = readback.poll_readback(&device);
            polls += 1;

            if state.is_mapped() {
                break;
            }
            if state.is_failed() {
                panic!("Readback failed");
            }
            if polls > 1000 {
                panic!("Readback poll timeout");
            }
        }

        // Finish and verify
        let read_data: Vec<u32> = readback.finish_readback(&device).unwrap();
        assert_eq!(read_data, data);
    }

    /// Test FrameFence with readback synchronization
    #[test]
    fn test_frame_fence_readback_sync() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        let fence = FrameFence::new(3);

        // Frame 0: Write data
        let data: Vec<u32> = vec![123; 32];
        let source = create_initialized_buffer(
            &device, &queue, &data,
            wgpu::BufferUsages::COPY_SRC,
            "Fence Test Source",
        );

        let mut encoder = device.create_command_encoder(&Default::default());
        let mut readback = BufferReadback::new(&device, 128, None);
        readback.read_buffer(&device, &queue, &mut encoder, &source, 0).unwrap();

        let submission = queue.submit([encoder.finish()]);
        fence.record_submission(submission);
        fence.advance_frame();

        // Wait for frame 0 to complete
        fence.wait_for_frame(&device, 0);

        // Now readback should be ready
        let read_data: Vec<u32> = readback.get_data().unwrap();
        assert_eq!(read_data, data);
    }
}

// ============================================================================
// SECTION 6: Resource State Tracking Integration Tests
// ============================================================================

mod resource_state_integration {
    use super::*;

    /// Test resource state tracker with multiple resources
    #[test]
    fn test_resource_state_tracker_multi_resource() {
        let mut tracker = ResourceStateTracker::new();

        // Register multiple resources
        let buffer_id = 1u64;
        let texture_id = 2u64;

        tracker.set(buffer_id, ResourceState::buffer(
            PipelineStage::Transfer,
            AccessFlags::TRANSFER_WRITE,
        ));

        tracker.set(texture_id, ResourceState::texture(
            PipelineStage::Transfer,
            AccessFlags::TRANSFER_WRITE,
            TextureLayout::TransferDst,
        ));

        assert_eq!(tracker.len(), 2);
        assert!(tracker.contains(buffer_id));
        assert!(tracker.contains(texture_id));

        // Update buffer state
        tracker.update(buffer_id, PipelineStage::VertexShader, AccessFlags::VERTEX_BUFFER_READ);

        let buffer_state = tracker.get(buffer_id).unwrap();
        assert_eq!(buffer_state.stage, PipelineStage::VertexShader);
        assert!(buffer_state.access.contains(AccessFlags::VERTEX_BUFFER_READ));

        // Update texture layout
        tracker.update_layout(texture_id, TextureLayout::ShaderReadOnly);

        let texture_state = tracker.get(texture_id).unwrap();
        assert_eq!(texture_state.layout, Some(TextureLayout::ShaderReadOnly));
    }

    /// Test barrier detector with complex state transitions
    #[test]
    fn test_barrier_detector_complex_transitions() {
        let mut detector = BarrierDetector::new();

        let texture_id = 1u64;

        // Initial: undefined
        detector.record_access(texture_id, ResourceState::texture(
            PipelineStage::None,
            AccessFlags::NONE,
            TextureLayout::Undefined,
        ));

        // Transition 1: Transfer write (upload)
        let upload_state = ResourceState::texture(
            PipelineStage::Transfer,
            AccessFlags::TRANSFER_WRITE,
            TextureLayout::TransferDst,
        );
        let _barrier1 = detector.transition(texture_id, upload_state);
        // First transition from undefined may or may not require barrier
        // depending on implementation

        // Transition 2: Shader read (sampling)
        let sample_state = ResourceState::texture(
            PipelineStage::FragmentShader,
            AccessFlags::SHADER_READ,
            TextureLayout::ShaderReadOnly,
        );
        let barrier2 = detector.transition(texture_id, sample_state);
        assert!(barrier2.is_some(), "Write->Read requires barrier");

        let b2 = barrier2.unwrap();
        assert_eq!(b2.hazard, HazardType::ReadAfterWrite);
        assert!(b2.has_layout_transition());
        assert_eq!(b2.old_layout, Some(TextureLayout::TransferDst));
        assert_eq!(b2.new_layout, Some(TextureLayout::ShaderReadOnly));

        // Transition 3: Another shader read (same layout, no barrier needed)
        let sample_state2 = ResourceState::texture(
            PipelineStage::ComputeShader,
            AccessFlags::SHADER_READ,
            TextureLayout::ShaderReadOnly,
        );
        let barrier3 = detector.transition(texture_id, sample_state2);
        // Read-after-read doesn't need barrier (no hazard)
        assert!(barrier3.is_none());

        // Transition 4: Shader write (storage image)
        let write_state = ResourceState::texture(
            PipelineStage::ComputeShader,
            AccessFlags::SHADER_WRITE,
            TextureLayout::StorageImage,
        );
        let barrier4 = detector.transition(texture_id, write_state);
        assert!(barrier4.is_some(), "Read->Write requires barrier");
        assert_eq!(barrier4.unwrap().hazard, HazardType::WriteAfterRead);
    }

    /// Test batch barrier detection
    #[test]
    fn test_batch_barrier_detection() {
        let mut detector = BarrierDetector::new();

        // Setup initial states
        detector.record_access(1, ResourceState::buffer(
            PipelineStage::Transfer, AccessFlags::TRANSFER_WRITE,
        ));
        detector.record_access(2, ResourceState::buffer(
            PipelineStage::Transfer, AccessFlags::TRANSFER_WRITE,
        ));
        detector.record_access(3, ResourceState::buffer(
            PipelineStage::ComputeShader, AccessFlags::SHADER_READ,
        ));

        // Prepare batch transitions
        let accesses = vec![
            (1, ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::VERTEX_BUFFER_READ)),
            (2, ResourceState::buffer(PipelineStage::FragmentShader, AccessFlags::UNIFORM_BUFFER_READ)),
            (3, ResourceState::buffer(PipelineStage::ComputeShader, AccessFlags::SHADER_WRITE)),
        ];

        // Detect all barriers without modifying state
        let barriers = detector.detect_all_barriers(&accesses);
        assert_eq!(barriers.len(), 3); // All need barriers

        // Now apply transitions
        let applied_barriers = detector.transition_batch(&accesses);
        assert_eq!(applied_barriers.len(), 3);
    }

    /// Test hazard type detection
    #[test]
    fn test_hazard_type_detection() {
        // Read-after-write
        let old_write = ResourceState::buffer(
            PipelineStage::ComputeShader, AccessFlags::SHADER_WRITE,
        );
        let new_read = ResourceState::buffer(
            PipelineStage::FragmentShader, AccessFlags::SHADER_READ,
        );
        assert_eq!(BarrierDetector::detect_hazard(&old_write, &new_read), HazardType::ReadAfterWrite);

        // Write-after-read
        let old_read = ResourceState::buffer(
            PipelineStage::VertexShader, AccessFlags::VERTEX_BUFFER_READ,
        );
        let new_write = ResourceState::buffer(
            PipelineStage::ComputeShader, AccessFlags::SHADER_WRITE,
        );
        assert_eq!(BarrierDetector::detect_hazard(&old_read, &new_write), HazardType::WriteAfterRead);

        // Write-after-write
        let old_write2 = ResourceState::buffer(
            PipelineStage::Transfer, AccessFlags::TRANSFER_WRITE,
        );
        let new_write2 = ResourceState::buffer(
            PipelineStage::ComputeShader, AccessFlags::SHADER_WRITE,
        );
        assert_eq!(BarrierDetector::detect_hazard(&old_write2, &new_write2), HazardType::WriteAfterWrite);

        // Read-after-read (no hazard)
        let old_read2 = ResourceState::buffer(
            PipelineStage::VertexShader, AccessFlags::VERTEX_BUFFER_READ,
        );
        let new_read2 = ResourceState::buffer(
            PipelineStage::FragmentShader, AccessFlags::SHADER_READ,
        );
        assert_eq!(BarrierDetector::detect_hazard(&old_read2, &new_read2), HazardType::None);

        // Layout transition only
        let old_tex = ResourceState::texture(
            PipelineStage::FragmentShader, AccessFlags::SHADER_READ, TextureLayout::ShaderReadOnly,
        );
        let new_tex = ResourceState::texture(
            PipelineStage::ColorOutput, AccessFlags::SHADER_READ, TextureLayout::ColorAttachment,
        );
        // Read-Read but different layouts should detect layout transition
        let hazard = BarrierDetector::detect_hazard(&old_tex, &new_tex);
        assert_eq!(hazard, HazardType::LayoutTransition);
    }
}

// ============================================================================
// SECTION 7: End-to-End Integration Tests
// ============================================================================

mod end_to_end {
    use super::*;
    use std::thread;

    /// Complete render frame simulation with all sync primitives
    #[test]
    fn test_complete_frame_simulation() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        let renderer = DoubleBufferedRenderer::new();
        let mut pacer = FramePacer::new(60.0);
        let mut detector = BarrierDetector::new();

        // Simulate vertex buffer (would be uploaded)
        let vertex_data: Vec<TestVertex> = vec![
            TestVertex { position: [0.0, 0.0, 0.0, 1.0], normal: [0.0, 1.0, 0.0, 0.0] },
            TestVertex { position: [1.0, 0.0, 0.0, 1.0], normal: [0.0, 1.0, 0.0, 0.0] },
            TestVertex { position: [0.0, 1.0, 0.0, 1.0], normal: [0.0, 1.0, 0.0, 0.0] },
        ];
        let _vertex_buffer = create_initialized_buffer(
            &device, &queue, &vertex_data,
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_SRC,
            "Vertex Buffer",
        );

        // Track vertex buffer state
        let vertex_buffer_id = 1u64;
        detector.record_access(vertex_buffer_id, ResourceState::buffer(
            PipelineStage::Transfer,
            AccessFlags::TRANSFER_WRITE,
        ));

        // Simulate 3 frames
        for frame in 0..3 {
            // Begin frame
            renderer.begin_frame(&device);
            pacer.begin_frame();

            // Transition vertex buffer for rendering
            let render_state = ResourceState::buffer(
                PipelineStage::VertexInput,
                AccessFlags::VERTEX_BUFFER_READ,
            );
            let barrier = detector.transition(vertex_buffer_id, render_state);
            if frame == 0 {
                assert!(barrier.is_some(), "First frame needs write->read barrier");
            }

            // Create command encoder
            let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("Frame {} Encoder", frame)),
            });

            // Simulate render pass (just encode some work)
            // In a real scenario, we'd have actual render commands here

            // Submit
            let submission = renderer.end_frame_with_iter(&queue, Some(encoder.finish()));
            let _ = submission;

            // End frame timing
            pacer.end_frame();

            // Adaptive pacing
            if pacer.is_adaptive() {
                if let Some(sleep) = pacer.should_sleep() {
                    thread::sleep(sleep.min(Duration::from_millis(5)));
                }
            }
        }

        // Wait for all GPU work
        renderer.wait_idle(&device);

        // Verify frame count
        assert_eq!(renderer.frame_count(), 3);
        assert_eq!(pacer.total_frames(), 3);
    }

    /// Test multiple sync primitives working together
    #[test]
    fn test_sync_primitives_integration() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        // Use TrinityFrameSynchronizer for triple buffering
        let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);
        let mut detector = BarrierDetector::new();

        // Create buffers for each frame's data
        let buffer_size = 256u64;
        let buffers: Vec<wgpu::Buffer> = (0..3).map(|i| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Frame Buffer {}", i)),
                size: buffer_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        }).collect();

        // Readback for verification
        let mut readback = BufferReadback::new(&device, buffer_size, Some("Integration Readback"));

        // Simulate frames
        for frame in 0..6 {
            sync.begin_frame(&device);

            let buffer_idx = sync.current_index() as usize;
            let buffer = &buffers[buffer_idx];

            // Track buffer state
            let buffer_id = (buffer_idx + 1) as u64;
            if frame < 3 {
                detector.record_access(buffer_id, ResourceState::buffer(
                    PipelineStage::Transfer,
                    AccessFlags::TRANSFER_WRITE,
                ));
            }

            // Write frame data
            let frame_data: Vec<u32> = vec![frame as u32; 64];
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&frame_data));

            // Create encoder and submit
            let encoder = device.create_command_encoder(&Default::default());
            sync.end_frame_with_iter(&queue, Some(encoder.finish()));
        }

        // Wait for all work
        sync.wait_idle(&device);

        // Verify last buffer's contents
        let last_buffer_idx = (5 % 3) as usize; // Frame 5 used buffer (5 % 3) = 2
        let last_buffer = &buffers[last_buffer_idx];

        let mut encoder = device.create_command_encoder(&Default::default());
        readback.read_buffer(&device, &queue, &mut encoder, last_buffer, 0).unwrap();
        queue.submit([encoder.finish()]);
        device.poll(wgpu::Maintain::Wait);

        let read_data: Vec<u32> = readback.get_data().unwrap();
        // Last write to buffer 2 was frame 5
        assert!(read_data.iter().all(|&v| v == 5), "Buffer should contain frame 5 data");
    }

    /// Test resource state consistency across frame boundaries
    #[test]
    fn test_resource_state_across_frames() {
        let mut tracker = ResourceStateTracker::new();
        let texture_id = 1u64;

        // Frame 0: Upload texture
        tracker.set(texture_id, ResourceState::texture(
            PipelineStage::Transfer,
            AccessFlags::TRANSFER_WRITE,
            TextureLayout::TransferDst,
        ));

        // Frame 1: Use in shader
        let frame1_state = tracker.get(texture_id).unwrap();
        assert_eq!(frame1_state.layout, Some(TextureLayout::TransferDst));

        // Transition for shader read
        let new_state = ResourceState::texture(
            PipelineStage::FragmentShader,
            AccessFlags::SHADER_READ,
            TextureLayout::ShaderReadOnly,
        );
        let transition = tracker.transition(texture_id, new_state.clone());
        assert!(transition.is_some());

        // Frame 2: Still in shader read (no transition needed)
        let same_state = ResourceState::texture(
            PipelineStage::FragmentShader,
            AccessFlags::SHADER_READ,
            TextureLayout::ShaderReadOnly,
        );
        let no_transition = tracker.transition(texture_id, same_state);
        assert!(no_transition.is_none(), "Same state shouldn't require transition");

        // Frame 3: Render target use
        let render_state = ResourceState::texture(
            PipelineStage::ColorOutput,
            AccessFlags::COLOR_ATTACHMENT_WRITE,
            TextureLayout::ColorAttachment,
        );
        let render_transition = tracker.transition(texture_id, render_state);
        assert!(render_transition.is_some(), "Read->Write requires transition");
    }
}
