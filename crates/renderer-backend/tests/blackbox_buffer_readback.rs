//! Blackbox tests for buffer readback utility (T-WGPU-P4.8.2)
//!
//! CLEANROOM: Tests only via public API. No implementation details read.
//!
//! ACCEPTANCE CRITERIA:
//! 1. Staging buffer creation for GPU->CPU readback
//! 2. Copy command encoding (GPU buffer -> staging buffer)
//! 3. Async readback with callback
//! 4. Double-buffered readback for continuous streaming
//!
//! PUBLIC API UNDER TEST:
//! - BufferReadback::new(device, size, label) -> Self
//! - BufferReadback::read_buffer(&mut self, device, queue, encoder, source, offset) -> Result
//! - BufferReadback::get_data<T: Pod>(&self) -> Option<Vec<T>>
//! - BufferReadback::begin_readback(&mut self, encoder, source, offset)
//! - BufferReadback::poll_readback(&mut self, device) -> MappingState
//! - BufferReadback::finish_readback<T: Pod>(&mut self, device) -> Option<Vec<T>>
//! - BufferReadback::staging_buffer(&self) -> &Buffer
//! - BufferReadback::size(&self) -> u64
//! - DoubleBufferedReadback::new(device, size, label) -> Self
//! - DoubleBufferedReadback::begin_readback(&mut self, encoder, source, offset)
//! - DoubleBufferedReadback::poll_and_get<T: Pod>(&mut self, device) -> Option<Vec<T>>
//! - DoubleBufferedReadback::swap(&mut self)
//! - DoubleBufferedReadback::wait_and_get<T: Pod>(&mut self, device) -> Option<Vec<T>>
//! - DoubleBufferedReadback::size(&self) -> u64
//! - DoubleBufferedReadback::current_index(&self) -> usize
//! - DoubleBufferedReadback::previous_index(&self) -> usize

use bytemuck::{Pod, Zeroable};
use renderer_backend::buffer_mapping::{
    BufferMapError, BufferReadback, DoubleBufferedReadback, MappingState,
};

// ============================================================================
// Test Data Types
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
struct TestVec4 {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
struct TestParticle {
    position: [f32; 4],
    velocity: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
struct TestQueryResult {
    value: u64,
}

// ============================================================================
// GPU Test Helpers
// ============================================================================

/// Create a test device and queue.
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
            label: Some("BufferReadback Test Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::default(),
        },
        None,
    )).ok()?;

    Some((device, queue))
}

/// Create a source buffer and initialize with data.
fn create_initialized_source_buffer<T: Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    data: &[T],
    label: &str,
) -> wgpu::Buffer {
    let bytes: &[u8] = bytemuck::cast_slice(data);

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    queue.write_buffer(&buffer, 0, bytes);

    buffer
}

// ============================================================================
// SECTION 1: BufferReadback Creation Tests
// ============================================================================

mod buffer_readback_creation {
    use super::*;

    #[test]
    fn test_create_small_readback() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        let readback = BufferReadback::new(&device, 64, Some("Small Readback"));
        assert_eq!(readback.size(), 64);
    }

    #[test]
    fn test_create_medium_readback() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        let readback = BufferReadback::new(&device, 4096, Some("Medium Readback"));
        assert_eq!(readback.size(), 4096);
    }

    #[test]
    fn test_create_large_readback() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        let readback = BufferReadback::new(&device, 1024 * 1024, Some("Large Readback"));
        assert_eq!(readback.size(), 1024 * 1024);
    }

    #[test]
    fn test_create_readback_without_label() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        let readback = BufferReadback::new(&device, 256, None);
        assert_eq!(readback.size(), 256);
    }

    #[test]
    fn test_readback_initial_state() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        let readback = BufferReadback::new(&device, 256, None);
        assert_eq!(readback.state(), MappingState::Unmapped);
        assert!(!readback.is_ready());
    }

    #[test]
    fn test_readback_staging_buffer_accessible() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        let readback = BufferReadback::new(&device, 512, Some("Access Test"));
        let staging = readback.staging_buffer();
        assert_eq!(staging.size(), 512);
    }

    #[test]
    #[should_panic(expected = "multiple of")]
    fn test_create_readback_unaligned_size_panics() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            panic!("multiple of"); // Trigger expected panic message
        };

        // Size 5 is not a multiple of MAP_SIZE_ALIGNMENT (4)
        let _readback = BufferReadback::new(&device, 5, None);
    }

    #[test]
    fn test_readback_debug_format() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        let readback = BufferReadback::new(&device, 256, Some("Debug Test"));
        let debug_str = format!("{:?}", readback);
        assert!(debug_str.contains("BufferReadback"));
        assert!(debug_str.contains("256"));
    }
}

// ============================================================================
// SECTION 2: Single-Shot Readback Tests
// ============================================================================

mod single_shot_readback {
    use super::*;

    #[test]
    fn test_read_buffer_u32() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        let data: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let source = create_initialized_source_buffer(&device, &queue, &data, "Source");

        let mut readback = BufferReadback::new(&device, (data.len() * 4) as u64, None);

        let mut encoder = device.create_command_encoder(&Default::default());
        let result = readback.read_buffer(&device, &queue, &mut encoder, &source, 0);
        assert!(result.is_ok());

        queue.submit([encoder.finish()]);
        device.poll(wgpu::Maintain::Wait);

        // Use wait_and_get_data to properly start deferred mapping
        let read_data: Vec<u32> = readback.wait_and_get_data(&device).unwrap();
        assert_eq!(read_data, data);
    }

    #[test]
    fn test_read_buffer_f32() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        let data: Vec<f32> = vec![1.0, 2.5, 3.14159, 4.0, -5.5, 0.0, f32::MAX, f32::MIN];
        let source = create_initialized_source_buffer(&device, &queue, &data, "F32 Source");

        let mut readback = BufferReadback::new(&device, (data.len() * 4) as u64, None);

        let mut encoder = device.create_command_encoder(&Default::default());
        readback.read_buffer(&device, &queue, &mut encoder, &source, 0).unwrap();
        queue.submit([encoder.finish()]);
        device.poll(wgpu::Maintain::Wait);

        let read_data: Vec<f32> = readback.wait_and_get_data(&device).unwrap();
        assert_eq!(read_data.len(), data.len());
        for (a, b) in read_data.iter().zip(data.iter()) {
            if a.is_nan() {
                assert!(b.is_nan());
            } else {
                assert!((a - b).abs() < f32::EPSILON || (*a == *b));
            }
        }
    }

    #[test]
    fn test_read_buffer_with_offset() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        // Create source with 32 u32s (128 bytes)
        let data: Vec<u32> = (0..32).collect();
        let source = create_initialized_source_buffer(&device, &queue, &data, "Offset Source");

        // Read only 8 u32s (32 bytes) starting at offset 64 (16 u32s)
        let mut readback = BufferReadback::new(&device, 32, None);

        let mut encoder = device.create_command_encoder(&Default::default());
        readback.read_buffer(&device, &queue, &mut encoder, &source, 64).unwrap();
        queue.submit([encoder.finish()]);
        device.poll(wgpu::Maintain::Wait);

        let read_data: Vec<u32> = readback.wait_and_get_data(&device).unwrap();
        let expected: Vec<u32> = (16..24).collect();
        assert_eq!(read_data, expected);
    }

    #[test]
    fn test_read_buffer_unaligned_offset_fails() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        let data: Vec<u32> = vec![1, 2, 3, 4];
        let source = create_initialized_source_buffer(&device, &queue, &data, "Source");

        let mut readback = BufferReadback::new(&device, 16, None);

        let mut encoder = device.create_command_encoder(&Default::default());
        // Offset 4 is not aligned to 8
        let result = readback.read_buffer(&device, &queue, &mut encoder, &source, 4);

        assert!(matches!(result, Err(BufferMapError::UnalignedOffset { .. })));
    }

    #[test]
    fn test_read_buffer_vec4() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        let data: Vec<TestVec4> = vec![
            TestVec4 { x: 1.0, y: 2.0, z: 3.0, w: 4.0 },
            TestVec4 { x: 5.0, y: 6.0, z: 7.0, w: 8.0 },
            TestVec4 { x: 9.0, y: 10.0, z: 11.0, w: 12.0 },
            TestVec4 { x: 13.0, y: 14.0, z: 15.0, w: 16.0 },
        ];
        let source = create_initialized_source_buffer(&device, &queue, &data, "Vec4 Source");

        let mut readback = BufferReadback::new(&device, (data.len() * 16) as u64, None);

        let mut encoder = device.create_command_encoder(&Default::default());
        readback.read_buffer(&device, &queue, &mut encoder, &source, 0).unwrap();
        queue.submit([encoder.finish()]);
        device.poll(wgpu::Maintain::Wait);

        let read_data: Vec<TestVec4> = readback.wait_and_get_data(&device).unwrap();
        assert_eq!(read_data, data);
    }

    #[test]
    fn test_read_buffer_particle() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        let data: Vec<TestParticle> = vec![
            TestParticle {
                position: [1.0, 2.0, 3.0, 1.0],
                velocity: [0.1, 0.2, 0.3, 0.0],
            },
            TestParticle {
                position: [4.0, 5.0, 6.0, 1.0],
                velocity: [0.4, 0.5, 0.6, 0.0],
            },
        ];
        let source = create_initialized_source_buffer(&device, &queue, &data, "Particle Source");

        let mut readback = BufferReadback::new(&device, (data.len() * 32) as u64, None);

        let mut encoder = device.create_command_encoder(&Default::default());
        readback.read_buffer(&device, &queue, &mut encoder, &source, 0).unwrap();
        queue.submit([encoder.finish()]);
        device.poll(wgpu::Maintain::Wait);

        let read_data: Vec<TestParticle> = readback.wait_and_get_data(&device).unwrap();
        assert_eq!(read_data, data);
    }

    #[test]
    fn test_wait_and_get_data() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        let data: Vec<u32> = vec![100, 200, 300, 400];
        let source = create_initialized_source_buffer(&device, &queue, &data, "Source");

        let mut readback = BufferReadback::new(&device, 16, None);

        let mut encoder = device.create_command_encoder(&Default::default());
        readback.read_buffer(&device, &queue, &mut encoder, &source, 0).unwrap();
        queue.submit([encoder.finish()]);

        // Use wait_and_get_data instead of separate poll/get
        let read_data: Vec<u32> = readback.wait_and_get_data(&device).unwrap();
        assert_eq!(read_data, data);
    }
}

// ============================================================================
// SECTION 3: Async Readback Tests
// ============================================================================

mod async_readback {
    use super::*;

    #[test]
    fn test_async_readback_workflow() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        let data: Vec<u32> = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let source = create_initialized_source_buffer(&device, &queue, &data, "Async Source");

        let mut readback = BufferReadback::new(&device, 32, None);

        // Step 1: Begin readback
        let mut encoder = device.create_command_encoder(&Default::default());
        readback.begin_readback(&mut encoder, &source, 0);
        queue.submit([encoder.finish()]);

        // Step 2: Poll until ready
        loop {
            let state = readback.poll_readback(&device);
            if state.is_mapped() {
                break;
            }
            if state.is_failed() {
                panic!("Readback failed");
            }
        }

        // Step 3: Finish and get data
        let read_data: Vec<u32> = readback.finish_readback(&device).unwrap();
        assert_eq!(read_data, data);
    }

    #[test]
    fn test_async_readback_poll_states() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        let data: Vec<u32> = vec![1, 2, 3, 4];
        let source = create_initialized_source_buffer(&device, &queue, &data, "Poll Source");

        let mut readback = BufferReadback::new(&device, 16, None);

        // Initial state
        assert_eq!(readback.state(), MappingState::Unmapped);

        // Begin readback (just encodes copy, doesn't start mapping)
        let mut encoder = device.create_command_encoder(&Default::default());
        readback.begin_readback(&mut encoder, &source, 0);
        queue.submit([encoder.finish()]);

        // First poll starts the mapping
        let state = readback.poll_readback(&device);
        assert!(state.is_pending() || state.is_mapped());

        // Poll until complete
        while !readback.poll_readback(&device).is_mapped() {
            // Keep polling
        }

        assert!(readback.is_ready());
    }

    #[test]
    fn test_async_readback_finish_without_prior_poll() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        let data: Vec<u32> = vec![42, 43, 44, 45];
        let source = create_initialized_source_buffer(&device, &queue, &data, "Direct Source");

        let mut readback = BufferReadback::new(&device, 16, None);

        let mut encoder = device.create_command_encoder(&Default::default());
        readback.begin_readback(&mut encoder, &source, 0);
        queue.submit([encoder.finish()]);

        // Directly call finish_readback without polling first
        let read_data: Vec<u32> = readback.finish_readback(&device).unwrap();
        assert_eq!(read_data, data);
    }

    #[test]
    fn test_async_readback_multiple_sequential() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        let mut readback = BufferReadback::new(&device, 16, None);

        for i in 0..5 {
            let data: Vec<u32> = vec![i, i + 1, i + 2, i + 3];
            let source = create_initialized_source_buffer(&device, &queue, &data, "Seq Source");

            let mut encoder = device.create_command_encoder(&Default::default());
            readback.begin_readback(&mut encoder, &source, 0);
            queue.submit([encoder.finish()]);

            let read_data: Vec<u32> = readback.finish_readback(&device).unwrap();
            assert_eq!(read_data, data);
        }
    }

    #[test]
    fn test_async_readback_reset() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        let data: Vec<u32> = vec![1, 2, 3, 4];
        let source = create_initialized_source_buffer(&device, &queue, &data, "Reset Source");

        let mut readback = BufferReadback::new(&device, 16, None);

        // Start a readback
        let mut encoder = device.create_command_encoder(&Default::default());
        readback.begin_readback(&mut encoder, &source, 0);
        queue.submit([encoder.finish()]);

        // Reset before completing
        readback.reset();
        assert_eq!(readback.state(), MappingState::Unmapped);

        // Can start a new readback
        let mut encoder = device.create_command_encoder(&Default::default());
        readback.begin_readback(&mut encoder, &source, 0);
        queue.submit([encoder.finish()]);

        let read_data: Vec<u32> = readback.finish_readback(&device).unwrap();
        assert_eq!(read_data, data);
    }
}

// ============================================================================
// SECTION 4: DoubleBufferedReadback Tests
// ============================================================================

mod double_buffered_readback {
    use super::*;

    #[test]
    fn test_create_double_buffered() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        let double = DoubleBufferedReadback::new(&device, 256, Some("Double"));
        assert_eq!(double.size(), 256);
        assert_eq!(double.current_index(), 0);
        assert_eq!(double.previous_index(), 1);
    }

    #[test]
    fn test_double_buffered_swap() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        let mut double = DoubleBufferedReadback::new(&device, 256, None);

        assert_eq!(double.current_index(), 0);
        assert_eq!(double.previous_index(), 1);

        double.swap();
        assert_eq!(double.current_index(), 1);
        assert_eq!(double.previous_index(), 0);

        double.swap();
        assert_eq!(double.current_index(), 0);
        assert_eq!(double.previous_index(), 1);
    }

    #[test]
    fn test_double_buffered_buffer_access() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        let double = DoubleBufferedReadback::new(&device, 512, Some("Access"));

        let buffer_0 = double.staging_buffer(0);
        let buffer_1 = double.staging_buffer(1);

        assert_eq!(buffer_0.size(), 512);
        assert_eq!(buffer_1.size(), 512);
    }

    #[test]
    fn test_double_buffered_streaming_pattern() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        let mut double = DoubleBufferedReadback::new(&device, 16, Some("Streaming"));

        // Simulate several frames
        for frame in 0..5 {
            let data: Vec<u32> = vec![frame, frame + 1, frame + 2, frame + 3];
            let source = create_initialized_source_buffer(&device, &queue, &data, "Frame Source");

            // Begin readback for current frame
            let mut encoder = device.create_command_encoder(&Default::default());
            double.begin_readback(&mut encoder, &source, 0);
            queue.submit([encoder.finish()]);
            device.poll(wgpu::Maintain::Wait);

            // On frame 1+, previous frame's data should be available
            if frame > 0 {
                let prev_data: Option<Vec<u32>> = double.wait_and_get(&device);
                assert!(prev_data.is_some());
            }

            double.swap();
        }
    }

    #[test]
    fn test_double_buffered_poll_and_get() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        let mut double = DoubleBufferedReadback::new(&device, 16, None);

        // First frame: start readback, no previous data
        let data_0: Vec<u32> = vec![10, 20, 30, 40];
        let source_0 = create_initialized_source_buffer(&device, &queue, &data_0, "Source 0");

        let mut encoder = device.create_command_encoder(&Default::default());
        double.begin_readback(&mut encoder, &source_0, 0);
        queue.submit([encoder.finish()]);
        device.poll(wgpu::Maintain::Wait);

        double.swap();

        // Second frame: start new readback, get previous data
        let data_1: Vec<u32> = vec![50, 60, 70, 80];
        let source_1 = create_initialized_source_buffer(&device, &queue, &data_1, "Source 1");

        let mut encoder = device.create_command_encoder(&Default::default());
        double.begin_readback(&mut encoder, &source_1, 0);
        queue.submit([encoder.finish()]);
        device.poll(wgpu::Maintain::Wait);

        // Now we can get the data from frame 0
        let read_0: Vec<u32> = double.wait_and_get(&device).unwrap();
        assert_eq!(read_0, data_0);
    }

    #[test]
    fn test_double_buffered_reset() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        let mut double = DoubleBufferedReadback::new(&device, 16, None);

        // Do some work
        let data: Vec<u32> = vec![1, 2, 3, 4];
        let source = create_initialized_source_buffer(&device, &queue, &data, "Source");

        let mut encoder = device.create_command_encoder(&Default::default());
        double.begin_readback(&mut encoder, &source, 0);
        queue.submit([encoder.finish()]);

        double.swap();

        // Reset
        double.reset();
        assert_eq!(double.current_index(), 0);
    }

    #[test]
    fn test_double_buffered_debug_format() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        let double = DoubleBufferedReadback::new(&device, 256, Some("Debug"));
        let debug_str = format!("{:?}", double);
        assert!(debug_str.contains("DoubleBufferedReadback"));
        assert!(debug_str.contains("256"));
    }

    #[test]
    fn test_double_buffered_current_previous_buffer_refs() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        let mut double = DoubleBufferedReadback::new(&device, 128, None);

        // Check current buffer
        let current = double.current_buffer();
        assert_eq!(current.size(), 128);

        // Check previous buffer
        let previous = double.previous_buffer();
        assert_eq!(previous.size(), 128);

        // After swap, they should be different
        let current_addr = current.staging_buffer() as *const _;
        double.swap();
        let new_current_addr = double.current_buffer().staging_buffer() as *const _;

        // The addresses should be different after swap
        assert_ne!(current_addr, new_current_addr);
    }

    #[test]
    #[should_panic(expected = "Buffer index must be 0 or 1")]
    fn test_double_buffered_invalid_index_panics() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            panic!("Buffer index must be 0 or 1");
        };

        let double = DoubleBufferedReadback::new(&device, 128, None);
        let _ = double.staging_buffer(2); // Should panic
    }
}

// ============================================================================
// SECTION 5: Edge Cases and Error Handling
// ============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn test_minimum_aligned_size() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        // Minimum aligned size is 4 (MAP_SIZE_ALIGNMENT)
        let readback = BufferReadback::new(&device, 4, None);
        assert_eq!(readback.size(), 4);
    }

    #[test]
    fn test_read_buffer_already_mapped_fails() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        let data: Vec<u32> = vec![1, 2, 3, 4];
        let source = create_initialized_source_buffer(&device, &queue, &data, "Source");

        let mut readback = BufferReadback::new(&device, 16, None);

        // First read
        let mut encoder = device.create_command_encoder(&Default::default());
        readback.read_buffer(&device, &queue, &mut encoder, &source, 0).unwrap();
        queue.submit([encoder.finish()]);

        // Try to read again without finishing
        let mut encoder2 = device.create_command_encoder(&Default::default());
        let result = readback.read_buffer(&device, &queue, &mut encoder2, &source, 0);

        assert!(matches!(result, Err(BufferMapError::AlreadyMapped)));
    }

    #[test]
    fn test_get_data_before_ready_returns_none() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        let data: Vec<u32> = vec![1, 2, 3, 4];
        let source = create_initialized_source_buffer(&device, &queue, &data, "Source");

        let mut readback = BufferReadback::new(&device, 16, None);

        // Get data before any readback
        let result: Option<Vec<u32>> = readback.get_data();
        assert!(result.is_none());

        // Start readback
        let mut encoder = device.create_command_encoder(&Default::default());
        readback.begin_readback(&mut encoder, &source, 0);
        queue.submit([encoder.finish()]);

        // Get data before mapping completes (might be None if not ready)
        // This is timing dependent, so we just verify it doesn't crash
        let _result: Option<Vec<u32>> = readback.get_data();
    }

    #[test]
    fn test_large_buffer_readback() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        // 64KB buffer
        let size = 65536;
        let element_count = size / 4;
        let data: Vec<u32> = (0..element_count).map(|i| i as u32).collect();
        let source = create_initialized_source_buffer(&device, &queue, &data, "Large Source");

        let mut readback = BufferReadback::new(&device, size as u64, None);

        let mut encoder = device.create_command_encoder(&Default::default());
        readback.read_buffer(&device, &queue, &mut encoder, &source, 0).unwrap();
        queue.submit([encoder.finish()]);

        let read_data: Vec<u32> = readback.wait_and_get_data(&device).unwrap();
        assert_eq!(read_data, data);
    }

    #[test]
    fn test_zero_offset_readback() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        let data: Vec<u32> = vec![111, 222, 333, 444];
        let source = create_initialized_source_buffer(&device, &queue, &data, "Source");

        let mut readback = BufferReadback::new(&device, 16, None);

        let mut encoder = device.create_command_encoder(&Default::default());
        // Explicit zero offset
        readback.read_buffer(&device, &queue, &mut encoder, &source, 0).unwrap();
        queue.submit([encoder.finish()]);

        let read_data: Vec<u32> = readback.wait_and_get_data(&device).unwrap();
        assert_eq!(read_data, data);
    }

    #[test]
    fn test_readback_u64_query_results() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        let data: Vec<u64> = vec![1000000, 2000000, 3000000, 4000000];
        let source = create_initialized_source_buffer(&device, &queue, &data, "Query Source");

        let mut readback = BufferReadback::new(&device, (data.len() * 8) as u64, None);

        let mut encoder = device.create_command_encoder(&Default::default());
        readback.read_buffer(&device, &queue, &mut encoder, &source, 0).unwrap();
        queue.submit([encoder.finish()]);

        let read_data: Vec<u64> = readback.wait_and_get_data(&device).unwrap();
        assert_eq!(read_data, data);
    }

    #[test]
    #[should_panic(expected = "aligned to")]
    fn test_begin_readback_unaligned_offset_panics() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            panic!("aligned to");
        };

        let data: Vec<u32> = vec![1, 2, 3, 4];
        let source = create_initialized_source_buffer(&device, &queue, &data, "Source");

        let mut readback = BufferReadback::new(&device, 16, None);
        let mut encoder = device.create_command_encoder(&Default::default());

        // Offset 7 is not aligned to 8
        readback.begin_readback(&mut encoder, &source, 7);
    }
}

// ============================================================================
// SECTION 6: Real-World Usage Patterns
// ============================================================================

mod usage_patterns {
    use super::*;

    #[test]
    fn test_compute_shader_output_readback() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        // Simulate compute shader output (256 floats)
        let output_count = 256;
        let data: Vec<f32> = (0..output_count).map(|i| i as f32 * 0.1).collect();
        let source = create_initialized_source_buffer(&device, &queue, &data, "Compute Output");

        let mut readback = BufferReadback::new(&device, (output_count * 4) as u64, Some("Compute Readback"));

        let mut encoder = device.create_command_encoder(&Default::default());
        readback.read_buffer(&device, &queue, &mut encoder, &source, 0).unwrap();
        queue.submit([encoder.finish()]);

        let read_data: Vec<f32> = readback.wait_and_get_data(&device).unwrap();
        assert_eq!(read_data.len(), output_count);
    }

    #[test]
    fn test_occlusion_query_readback() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        // Simulate 16 occlusion query results
        let query_count = 16;
        let data: Vec<TestQueryResult> = (0..query_count)
            .map(|i| TestQueryResult { value: i as u64 * 1000 })
            .collect();
        let source = create_initialized_source_buffer(&device, &queue, &data, "Query Results");

        let mut readback = BufferReadback::new(&device, (query_count * 8) as u64, Some("Query Readback"));

        let mut encoder = device.create_command_encoder(&Default::default());
        readback.read_buffer(&device, &queue, &mut encoder, &source, 0).unwrap();
        queue.submit([encoder.finish()]);

        let read_data: Vec<TestQueryResult> = readback.wait_and_get_data(&device).unwrap();
        assert_eq!(read_data, data);
    }

    #[test]
    fn test_particle_system_streaming_readback() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        let particle_count = 64;
        let particle_size = std::mem::size_of::<TestParticle>() as u64;
        let total_size = particle_count as u64 * particle_size;

        let mut double = DoubleBufferedReadback::new(&device, total_size, Some("Particles"));

        // Simulate 3 frames of particle updates
        for frame in 0..3 {
            let particles: Vec<TestParticle> = (0..particle_count)
                .map(|i| TestParticle {
                    position: [i as f32, frame as f32, 0.0, 1.0],
                    velocity: [0.1, 0.2, 0.3, 0.0],
                })
                .collect();
            let source = create_initialized_source_buffer(&device, &queue, &particles, "Particle Buffer");

            let mut encoder = device.create_command_encoder(&Default::default());
            double.begin_readback(&mut encoder, &source, 0);
            queue.submit([encoder.finish()]);
            device.poll(wgpu::Maintain::Wait);

            if frame > 0 {
                let prev_particles: Option<Vec<TestParticle>> = double.wait_and_get(&device);
                assert!(prev_particles.is_some());
            }

            double.swap();
        }
    }

    #[test]
    fn test_indirect_draw_args_readback() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        // Indirect draw args: vertex_count, instance_count, first_vertex, first_instance
        let draw_count = 16;
        let args: Vec<[u32; 4]> = (0..draw_count)
            .map(|i| [i as u32 * 3, 1, 0, i as u32])
            .collect();
        let source = create_initialized_source_buffer(&device, &queue, &args, "Draw Args");

        let mut readback = BufferReadback::new(&device, (draw_count * 16) as u64, Some("Draw Args Readback"));

        let mut encoder = device.create_command_encoder(&Default::default());
        readback.read_buffer(&device, &queue, &mut encoder, &source, 0).unwrap();
        queue.submit([encoder.finish()]);

        let read_data: Vec<[u32; 4]> = readback.wait_and_get_data(&device).unwrap();
        assert_eq!(read_data, args);
    }

    #[test]
    fn test_matrix_buffer_readback() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping: no GPU available");
            return;
        };

        // 16 4x4 matrices (1024 bytes)
        let matrix_count = 16;
        let matrices: Vec<[[f32; 4]; 4]> = (0..matrix_count)
            .map(|i| {
                let v = i as f32;
                [[v, 0.0, 0.0, 0.0],
                 [0.0, v, 0.0, 0.0],
                 [0.0, 0.0, v, 0.0],
                 [0.0, 0.0, 0.0, 1.0]]
            })
            .collect();
        let source = create_initialized_source_buffer(&device, &queue, &matrices, "Matrices");

        let mut readback = BufferReadback::new(&device, (matrix_count * 64) as u64, Some("Matrix Readback"));

        let mut encoder = device.create_command_encoder(&Default::default());
        readback.read_buffer(&device, &queue, &mut encoder, &source, 0).unwrap();
        queue.submit([encoder.finish()]);

        let read_data: Vec<[[f32; 4]; 4]> = readback.wait_and_get_data(&device).unwrap();
        assert_eq!(read_data, matrices);
    }
}
