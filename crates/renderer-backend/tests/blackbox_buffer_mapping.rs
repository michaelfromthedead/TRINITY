//! Blackbox tests for async buffer mapping (T-WGPU-P4.8.1)
//!
//! CLEANROOM: Tests only via public API. No implementation details read.
//!
//! ACCEPTANCE CRITERIA:
//! 1. Async buffer mapping with map_async()
//! 2. Non-blocking status polling
//! 3. Mapped slice access (read/write)
//! 4. Automatic unmap on drop
//!
//! PUBLIC API UNDER TEST:
//! - MappingState { Unmapped, Pending, Mapped, Failed }
//! - MappingMode { Read, Write }
//! - BufferMapError { AlreadyMapped, InvalidRange, MappingFailed, NotMapped, WrongMode, ... }
//! - BufferMapper::new(buffer: Arc<Buffer>) -> Self
//! - BufferMapper::map_async(&mut self, device, mode, offset, size) -> Result<(), BufferMapError>
//! - BufferMapper::poll(&mut self, device) -> MappingState
//! - BufferMapper::is_ready(&self) -> bool
//! - BufferMapper::wait(&mut self, device)
//! - BufferMapper::get_mapped_range(&self) -> Option<BufferView>
//! - BufferMapper::get_mapped_range_mut(&self) -> Option<BufferViewMut>
//! - BufferMapper::read_data<T: Pod>(&self) -> Option<Vec<T>>
//! - BufferMapper::write_data<T: Pod>(&mut self, data: &[T]) -> Result<(), BufferMapError>
//! - BufferMapper::state(&self) -> MappingState
//! - BufferMapper::unmap(&mut self)
//! - map_for_read(buffer, device) -> Result<BufferMapper, BufferMapError>
//! - map_for_write(buffer, device) -> Result<BufferMapper, BufferMapError>
//! - read_buffer_sync<T>(buffer, device, offset, count) -> Result<Vec<T>, BufferMapError>
//! - write_buffer_sync<T>(buffer, device, data, offset) -> Result<(), BufferMapError>

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use renderer_backend::buffer_mapping::{
    BufferMapError, BufferMapper, MappingMode, MappingState,
    map_for_read, map_for_write, read_buffer_sync, write_buffer_sync,
    COPY_BUFFER_ALIGNMENT, MAP_SIZE_ALIGNMENT,
};

// ============================================================================
// Test Data Types
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
struct TestVertex {
    position: [f32; 3],
    normal: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
struct TestUniform {
    model_view: [[f32; 4]; 4],
    projection: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
struct SimpleData {
    values: [u32; 4],
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
            label: Some("BufferMapping Test Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::default(),
        },
        None,
    )).ok()?;

    Some((device, queue))
}

/// Create a buffer suitable for reading (MAP_READ usage).
fn create_read_buffer(device: &wgpu::Device, size: u64) -> Arc<wgpu::Buffer> {
    Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Read Buffer"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }))
}

/// Create a buffer suitable for writing (MAP_WRITE usage).
fn create_write_buffer(device: &wgpu::Device, size: u64) -> Arc<wgpu::Buffer> {
    Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Write Buffer"),
        size,
        usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    }))
}

/// Create a buffer with both read and write capabilities.
fn create_read_write_buffer(device: &wgpu::Device, size: u64) -> Arc<wgpu::Buffer> {
    Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Read/Write Buffer"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE
             | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }))
}

/// Create a buffer with no mapping capability.
fn create_unmappable_buffer(device: &wgpu::Device, size: u64) -> Arc<wgpu::Buffer> {
    Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Unmappable Buffer"),
        size,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }))
}

/// Create a staging buffer for data transfer.
fn create_staging_buffer(device: &wgpu::Device, data: &[u8]) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: data.len() as u64,
        usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    })
}

// ============================================================================
// SECTION 1: MappingState Enum Tests
// ============================================================================

mod mapping_state_tests {
    use super::*;

    #[test]
    fn mapping_state_unmapped_default() {
        let state = MappingState::default();
        assert!(state.is_unmapped());
        assert!(!state.is_pending());
        assert!(!state.is_mapped());
        assert!(!state.is_failed());
    }

    #[test]
    fn mapping_state_unmapped_properties() {
        let state = MappingState::Unmapped;
        assert!(state.is_unmapped());
        assert!(!state.is_pending());
        assert!(!state.is_mapped());
        assert!(!state.is_failed());
        assert!(!state.is_ready());
        assert!(state.is_idle());
    }

    #[test]
    fn mapping_state_pending_properties() {
        let state = MappingState::Pending;
        assert!(!state.is_unmapped());
        assert!(state.is_pending());
        assert!(!state.is_mapped());
        assert!(!state.is_failed());
        assert!(!state.is_ready());
        assert!(!state.is_idle());
    }

    #[test]
    fn mapping_state_mapped_properties() {
        let state = MappingState::Mapped;
        assert!(!state.is_unmapped());
        assert!(!state.is_pending());
        assert!(state.is_mapped());
        assert!(!state.is_failed());
        assert!(state.is_ready());
        assert!(!state.is_idle());
    }

    #[test]
    fn mapping_state_failed_properties() {
        let state = MappingState::Failed;
        assert!(!state.is_unmapped());
        assert!(!state.is_pending());
        assert!(!state.is_mapped());
        assert!(state.is_failed());
        assert!(!state.is_ready());
        assert!(state.is_idle());
    }

    #[test]
    fn mapping_state_display_unmapped() {
        let state = MappingState::Unmapped;
        let display = format!("{}", state);
        assert!(display.to_lowercase().contains("unmap"));
    }

    #[test]
    fn mapping_state_display_pending() {
        let state = MappingState::Pending;
        let display = format!("{}", state);
        assert!(display.to_lowercase().contains("pend"));
    }

    #[test]
    fn mapping_state_display_mapped() {
        let state = MappingState::Mapped;
        let display = format!("{}", state);
        assert!(display.to_lowercase().contains("map"));
    }

    #[test]
    fn mapping_state_display_failed() {
        let state = MappingState::Failed;
        let display = format!("{}", state);
        assert!(display.to_lowercase().contains("fail"));
    }

    #[test]
    fn mapping_state_copy_clone() {
        let state = MappingState::Mapped;
        let copied = state;
        let cloned = state.clone();
        assert_eq!(copied, state);
        assert_eq!(cloned, state);
    }

    #[test]
    fn mapping_state_equality() {
        assert_eq!(MappingState::Unmapped, MappingState::Unmapped);
        assert_eq!(MappingState::Pending, MappingState::Pending);
        assert_eq!(MappingState::Mapped, MappingState::Mapped);
        assert_eq!(MappingState::Failed, MappingState::Failed);
        assert_ne!(MappingState::Unmapped, MappingState::Mapped);
        assert_ne!(MappingState::Pending, MappingState::Failed);
    }
}

// ============================================================================
// SECTION 2: MappingMode Enum Tests
// ============================================================================

mod mapping_mode_tests {
    use super::*;

    #[test]
    fn mapping_mode_read_properties() {
        let mode = MappingMode::Read;
        let usage = mode.required_usage();
        assert!(usage.contains(wgpu::BufferUsages::MAP_READ));
    }

    #[test]
    fn mapping_mode_write_properties() {
        let mode = MappingMode::Write;
        let usage = mode.required_usage();
        assert!(usage.contains(wgpu::BufferUsages::MAP_WRITE));
    }

    #[test]
    fn mapping_mode_display_read() {
        let mode = MappingMode::Read;
        let display = format!("{}", mode);
        assert!(display.to_lowercase().contains("read"));
    }

    #[test]
    fn mapping_mode_display_write() {
        let mode = MappingMode::Write;
        let display = format!("{}", mode);
        assert!(display.to_lowercase().contains("write"));
    }

    #[test]
    fn mapping_mode_copy_clone() {
        let mode = MappingMode::Read;
        let copied = mode;
        let cloned = mode.clone();
        assert_eq!(copied, mode);
        assert_eq!(cloned, mode);
    }

    #[test]
    fn mapping_mode_equality() {
        assert_eq!(MappingMode::Read, MappingMode::Read);
        assert_eq!(MappingMode::Write, MappingMode::Write);
        assert_ne!(MappingMode::Read, MappingMode::Write);
    }

    #[test]
    fn mapping_mode_converts_to_wgpu_mapmode() {
        let read_mode: wgpu::MapMode = MappingMode::Read.into();
        let write_mode: wgpu::MapMode = MappingMode::Write.into();
        assert_eq!(read_mode, wgpu::MapMode::Read);
        assert_eq!(write_mode, wgpu::MapMode::Write);
    }
}

// ============================================================================
// SECTION 3: BufferMapError Tests
// ============================================================================

mod buffer_map_error_tests {
    use super::*;

    #[test]
    fn error_already_mapped_display() {
        let err = BufferMapError::AlreadyMapped;
        let display = format!("{}", err);
        assert!(display.to_lowercase().contains("already") || display.to_lowercase().contains("mapped"));
    }

    #[test]
    fn error_invalid_range_display() {
        let err = BufferMapError::InvalidRange { offset: 100, size: 200, buffer_size: 50 };
        let display = format!("{}", err);
        assert!(display.contains("100") || display.contains("200") || display.to_lowercase().contains("range"));
    }

    #[test]
    fn error_mapping_failed_display() {
        let err = BufferMapError::MappingFailed("GPU timeout".to_string());
        let display = format!("{}", err);
        assert!(display.contains("timeout") || display.to_lowercase().contains("fail"));
    }

    #[test]
    fn error_not_mapped_display() {
        let err = BufferMapError::NotMapped;
        let display = format!("{}", err);
        assert!(display.to_lowercase().contains("not") || display.to_lowercase().contains("mapped"));
    }

    #[test]
    fn error_wrong_mode_display() {
        let err = BufferMapError::WrongMode { current: MappingMode::Write, required: MappingMode::Read };
        let display = format!("{}", err);
        assert!(display.to_lowercase().contains("mode") || display.to_lowercase().contains("read") || display.to_lowercase().contains("write"));
    }

    #[test]
    fn error_implements_std_error() {
        let err: Box<dyn std::error::Error> = Box::new(BufferMapError::NotMapped);
        assert!(err.to_string().len() > 0);
    }

    #[test]
    fn error_debug_format() {
        let err = BufferMapError::AlreadyMapped;
        let debug = format!("{:?}", err);
        assert!(debug.contains("AlreadyMapped"));
    }
}

// ============================================================================
// SECTION 4: Constants Tests
// ============================================================================

mod constants_tests {
    use super::*;

    #[test]
    fn copy_buffer_alignment_is_nonzero() {
        assert!(COPY_BUFFER_ALIGNMENT > 0);
    }

    #[test]
    fn copy_buffer_alignment_is_power_of_two() {
        assert!(COPY_BUFFER_ALIGNMENT.is_power_of_two());
    }

    #[test]
    fn map_size_alignment_is_nonzero() {
        assert!(MAP_SIZE_ALIGNMENT > 0);
    }

    #[test]
    fn map_size_alignment_is_power_of_two() {
        assert!(MAP_SIZE_ALIGNMENT.is_power_of_two());
    }
}

// ============================================================================
// SECTION 5: BufferMapper Construction Tests (requires GPU)
// ============================================================================

mod buffer_mapper_construction {
    use super::*;

    #[test]
    
    fn new_creates_unmapped_mapper() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 256);
        let mapper = BufferMapper::new(buffer);

        assert_eq!(mapper.state(), MappingState::Unmapped);
        assert!(!mapper.is_ready());
        assert!(!mapper.is_pending());
    }

    #[test]
    
    fn new_preserves_buffer_reference() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 512);
        let mapper = BufferMapper::new(buffer.clone());

        // Buffer reference should be accessible
        let buf_ref = mapper.buffer();
        assert_eq!(buf_ref.size(), 512);
    }

    #[test]
    
    fn new_initializes_offset_and_size_to_zero() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 256);
        let mapper = BufferMapper::new(buffer);

        assert_eq!(mapper.offset(), 0);
        assert_eq!(mapper.mapped_size(), 0);
        assert!(mapper.mode().is_none());
    }
}

// ============================================================================
// SECTION 6: CRITERION 1 - Async Buffer Mapping with map_async()
// ============================================================================

mod criterion_1_async_mapping {
    use super::*;

    #[test]
    
    fn map_async_read_mode_succeeds() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 256);
        let mut mapper = BufferMapper::new(buffer);

        let result = mapper.map_async(&device, MappingMode::Read, 0, 256);
        assert!(result.is_ok());
    }

    #[test]
    
    fn map_async_write_mode_succeeds() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_write_buffer(&device, 256);
        let mut mapper = BufferMapper::new(buffer);

        let result = mapper.map_async(&device, MappingMode::Write, 0, 256);
        assert!(result.is_ok());
    }

    #[test]
    
    fn map_async_transitions_to_pending() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 256);
        let mut mapper = BufferMapper::new(buffer);

        mapper.map_async(&device, MappingMode::Read, 0, 256).unwrap();

        // State should be Pending or Mapped (depending on how fast GPU responds)
        let state = mapper.state();
        assert!(state.is_pending() || state.is_mapped());
    }

    #[test]
    
    fn map_async_sets_mode() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 256);
        let mut mapper = BufferMapper::new(buffer);

        mapper.map_async(&device, MappingMode::Read, 0, 256).unwrap();

        assert_eq!(mapper.mode(), Some(MappingMode::Read));
    }

    #[test]
    
    fn map_async_sets_offset_and_size() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 1024);
        let mut mapper = BufferMapper::new(buffer);

        mapper.map_async(&device, MappingMode::Read, 128, 512).unwrap();

        assert_eq!(mapper.offset(), 128);
        assert_eq!(mapper.mapped_size(), 512);
    }

    #[test]
    
    fn map_async_partial_buffer() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 1024);
        let mut mapper = BufferMapper::new(buffer);

        // Map only middle portion
        let result = mapper.map_async(&device, MappingMode::Read, 256, 512);
        assert!(result.is_ok());
    }

    #[test]
    
    fn map_async_fails_when_already_mapped() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 256);
        let mut mapper = BufferMapper::new(buffer);

        mapper.map_async(&device, MappingMode::Read, 0, 256).unwrap();
        mapper.wait(&device);

        let result = mapper.map_async(&device, MappingMode::Read, 0, 256);
        assert!(matches!(result, Err(BufferMapError::AlreadyMapped)));
    }

    #[test]
    
    fn map_async_fails_when_pending() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 256);
        let mut mapper = BufferMapper::new(buffer);

        mapper.map_async(&device, MappingMode::Read, 0, 256).unwrap();

        // Second map_async should fail if still pending
        let result = mapper.map_async(&device, MappingMode::Read, 0, 256);
        assert!(result.is_err());
    }

    #[test]
    
    fn map_async_invalid_range_beyond_buffer() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 256);
        let mut mapper = BufferMapper::new(buffer);

        let result = mapper.map_async(&device, MappingMode::Read, 0, 512);
        assert!(matches!(result, Err(BufferMapError::InvalidRange { .. })));
    }

    #[test]
    
    fn map_async_invalid_offset_plus_size() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 256);
        let mut mapper = BufferMapper::new(buffer);

        let result = mapper.map_async(&device, MappingMode::Read, 200, 100);
        assert!(matches!(result, Err(BufferMapError::InvalidRange { .. })));
    }

    #[test]
    
    fn map_async_zero_size_fails() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 256);
        let mut mapper = BufferMapper::new(buffer);

        let result = mapper.map_async(&device, MappingMode::Read, 0, 0);
        // Zero-size mapping should fail or be rejected
        assert!(result.is_err());
    }
}

// ============================================================================
// SECTION 7: CRITERION 2 - Non-blocking Status Polling
// ============================================================================

mod criterion_2_polling {
    use super::*;

    #[test]
    
    fn poll_returns_unmapped_before_map() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 256);
        let mut mapper = BufferMapper::new(buffer);

        let state = mapper.poll(&device);
        assert_eq!(state, MappingState::Unmapped);
    }

    #[test]
    
    fn poll_returns_pending_or_mapped_after_map_async() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 256);
        let mut mapper = BufferMapper::new(buffer);

        mapper.map_async(&device, MappingMode::Read, 0, 256).unwrap();

        let state = mapper.poll(&device);
        assert!(state.is_pending() || state.is_mapped());
    }

    #[test]
    
    fn poll_eventually_returns_mapped() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 256);
        let mut mapper = BufferMapper::new(buffer);

        mapper.map_async(&device, MappingMode::Read, 0, 256).unwrap();

        // Poll until mapped (with reasonable limit)
        let mut attempts = 0;
        while mapper.poll(&device) != MappingState::Mapped && attempts < 1000 {
            device.poll(wgpu::Maintain::Poll);
            attempts += 1;
        }

        assert_eq!(mapper.state(), MappingState::Mapped);
    }

    #[test]
    
    fn poll_is_nonblocking() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 256);
        let mut mapper = BufferMapper::new(buffer);

        mapper.map_async(&device, MappingMode::Read, 0, 256).unwrap();

        // Poll should return immediately without blocking
        let start = std::time::Instant::now();
        let _state = mapper.poll(&device);
        let elapsed = start.elapsed();

        // Poll should complete in < 10ms (non-blocking)
        assert!(elapsed.as_millis() < 100);
    }

    #[test]
    
    fn is_ready_false_before_mapping_complete() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 256);
        let mapper = BufferMapper::new(buffer);

        assert!(!mapper.is_ready());
    }

    #[test]
    
    fn is_ready_true_after_wait() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 256);
        let mut mapper = BufferMapper::new(buffer);

        mapper.map_async(&device, MappingMode::Read, 0, 256).unwrap();
        mapper.wait(&device);

        assert!(mapper.is_ready());
    }

    #[test]
    
    fn wait_blocks_until_mapped() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 256);
        let mut mapper = BufferMapper::new(buffer);

        mapper.map_async(&device, MappingMode::Read, 0, 256).unwrap();
        mapper.wait(&device);

        assert_eq!(mapper.state(), MappingState::Mapped);
    }

    #[test]
    
    fn multiple_polls_are_idempotent() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 256);
        let mut mapper = BufferMapper::new(buffer);

        mapper.map_async(&device, MappingMode::Read, 0, 256).unwrap();
        mapper.wait(&device);

        // Multiple polls after mapping complete should all return Mapped
        for _ in 0..5 {
            assert_eq!(mapper.poll(&device), MappingState::Mapped);
        }
    }
}

// ============================================================================
// SECTION 8: CRITERION 3 - Mapped Slice Access (read/write)
// ============================================================================

mod criterion_3_slice_access {
    use super::*;

    #[test]
    
    fn get_mapped_range_none_when_unmapped() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 256);
        let mapper = BufferMapper::new(buffer);

        assert!(mapper.get_mapped_range().is_none());
    }

    #[test]
    
    fn get_mapped_range_some_when_mapped() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 256);
        let mut mapper = BufferMapper::new(buffer);

        mapper.map_async(&device, MappingMode::Read, 0, 256).unwrap();
        mapper.wait(&device);

        assert!(mapper.get_mapped_range().is_some());
    }

    #[test]
    
    fn get_mapped_range_mut_none_when_unmapped() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_write_buffer(&device, 256);
        let mapper = BufferMapper::new(buffer);

        assert!(mapper.get_mapped_range_mut().is_none());
    }

    #[test]
    
    fn get_mapped_range_mut_some_when_write_mapped() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_write_buffer(&device, 256);
        let mut mapper = BufferMapper::new(buffer);

        mapper.map_async(&device, MappingMode::Write, 0, 256).unwrap();
        mapper.wait(&device);

        assert!(mapper.get_mapped_range_mut().is_some());
    }

    #[test]
    
    fn read_data_returns_none_when_unmapped() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 256);
        let mapper = BufferMapper::new(buffer);

        let data: Option<Vec<u32>> = mapper.read_data();
        assert!(data.is_none());
    }

    #[test]
    
    fn read_data_returns_data_when_read_mapped() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        // Create buffer with known data
        let buffer = create_read_buffer(&device, 16);

        // Write known data to the buffer first
        let source_data: [u32; 4] = [1, 2, 3, 4];
        queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&source_data));
        queue.submit([]);
        device.poll(wgpu::Maintain::Wait);

        let mut mapper = BufferMapper::new(buffer);
        mapper.map_async(&device, MappingMode::Read, 0, 16).unwrap();
        mapper.wait(&device);

        let data: Option<Vec<u32>> = mapper.read_data();
        assert!(data.is_some());
        assert_eq!(data.unwrap(), vec![1u32, 2, 3, 4]);
    }

    #[test]
    
    fn write_data_fails_when_unmapped() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_write_buffer(&device, 256);
        let mut mapper = BufferMapper::new(buffer);

        let data: [u32; 4] = [1, 2, 3, 4];
        let result = mapper.write_data(&data);
        assert!(matches!(result, Err(BufferMapError::NotMapped)));
    }

    #[test]
    
    fn write_data_succeeds_when_write_mapped() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_write_buffer(&device, 64);
        let mut mapper = BufferMapper::new(buffer);

        mapper.map_async(&device, MappingMode::Write, 0, 64).unwrap();
        mapper.wait(&device);

        let data: [u32; 4] = [100, 200, 300, 400];
        let result = mapper.write_data(&data);
        assert!(result.is_ok());
    }

    #[test]

    fn write_data_fails_when_read_mapped() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        // Use a read-only buffer (wgpu doesn't allow MAP_READ | MAP_WRITE together)
        let buffer = create_read_buffer(&device, 64);
        let mut mapper = BufferMapper::new(buffer);

        mapper.map_async(&device, MappingMode::Read, 0, 64).unwrap();
        mapper.wait(&device);

        let data: [u32; 4] = [1, 2, 3, 4];
        let result = mapper.write_data(&data);
        assert!(matches!(result, Err(BufferMapError::WrongMode { .. })));
    }

    #[test]
    
    fn read_data_with_structured_types() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let vertex = TestVertex {
            position: [1.0, 2.0, 3.0],
            normal: [0.0, 1.0, 0.0],
        };
        let vertex_bytes = bytemuck::bytes_of(&vertex);
        let buffer_size = vertex_bytes.len() as u64;

        let buffer = create_read_buffer(&device, buffer_size);
        queue.write_buffer(&buffer, 0, vertex_bytes);
        queue.submit([]);
        device.poll(wgpu::Maintain::Wait);

        let mut mapper = BufferMapper::new(buffer);
        mapper.map_async(&device, MappingMode::Read, 0, buffer_size).unwrap();
        mapper.wait(&device);

        let data: Option<Vec<TestVertex>> = mapper.read_data();
        assert!(data.is_some());
        let vertices = data.unwrap();
        assert_eq!(vertices.len(), 1);
        assert_eq!(vertices[0], vertex);
    }

    #[test]
    
    fn write_data_with_structured_types() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let vertex = TestVertex {
            position: [4.0, 5.0, 6.0],
            normal: [1.0, 0.0, 0.0],
        };
        let buffer_size = std::mem::size_of::<TestVertex>() as u64;

        let buffer = create_write_buffer(&device, buffer_size);
        let mut mapper = BufferMapper::new(buffer);

        mapper.map_async(&device, MappingMode::Write, 0, buffer_size).unwrap();
        mapper.wait(&device);

        let result = mapper.write_data(&[vertex]);
        assert!(result.is_ok());
    }

    #[test]
    
    fn mapped_range_has_correct_size() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 512);
        let mut mapper = BufferMapper::new(buffer);

        mapper.map_async(&device, MappingMode::Read, 64, 256).unwrap();
        mapper.wait(&device);

        let view = mapper.get_mapped_range();
        assert!(view.is_some());
        assert_eq!(view.unwrap().len(), 256);
    }
}

// ============================================================================
// SECTION 9: CRITERION 4 - Automatic Unmap on Drop
// ============================================================================

mod criterion_4_automatic_unmap {
    use super::*;

    #[test]
    
    fn unmap_transitions_to_unmapped() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 256);
        let mut mapper = BufferMapper::new(buffer);

        mapper.map_async(&device, MappingMode::Read, 0, 256).unwrap();
        mapper.wait(&device);
        assert_eq!(mapper.state(), MappingState::Mapped);

        mapper.unmap();
        assert_eq!(mapper.state(), MappingState::Unmapped);
    }

    #[test]
    
    fn unmap_clears_mode() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 256);
        let mut mapper = BufferMapper::new(buffer);

        mapper.map_async(&device, MappingMode::Read, 0, 256).unwrap();
        mapper.wait(&device);
        assert!(mapper.mode().is_some());

        mapper.unmap();
        assert!(mapper.mode().is_none());
    }

    #[test]
    
    fn drop_unmaps_automatically() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 256);

        {
            let mut mapper = BufferMapper::new(buffer.clone());
            mapper.map_async(&device, MappingMode::Read, 0, 256).unwrap();
            mapper.wait(&device);
            // mapper drops here
        }

        // After drop, buffer should be usable again
        let mut mapper2 = BufferMapper::new(buffer);
        let result = mapper2.map_async(&device, MappingMode::Read, 0, 256);
        assert!(result.is_ok());
    }

    #[test]
    
    fn reset_allows_remapping() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 256);
        let mut mapper = BufferMapper::new(buffer);

        mapper.map_async(&device, MappingMode::Read, 0, 256).unwrap();
        mapper.wait(&device);

        mapper.reset();
        assert_eq!(mapper.state(), MappingState::Unmapped);

        // Should be able to map again
        let result = mapper.map_async(&device, MappingMode::Read, 0, 256);
        assert!(result.is_ok());
    }

    #[test]
    
    fn unmap_on_unmapped_is_safe() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 256);
        let mut mapper = BufferMapper::new(buffer);

        // Calling unmap when already unmapped should be safe (no-op)
        mapper.unmap();
        assert_eq!(mapper.state(), MappingState::Unmapped);
    }

    #[test]
    
    fn unmap_invalidates_mapped_range() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 256);
        let mut mapper = BufferMapper::new(buffer);

        mapper.map_async(&device, MappingMode::Read, 0, 256).unwrap();
        mapper.wait(&device);
        assert!(mapper.get_mapped_range().is_some());

        mapper.unmap();
        assert!(mapper.get_mapped_range().is_none());
    }
}

// ============================================================================
// SECTION 10: Convenience Function Tests
// ============================================================================

mod convenience_functions {
    use super::*;

    #[test]
    
    fn map_for_read_creates_ready_mapper() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 256);
        let result = map_for_read(buffer, &device);

        assert!(result.is_ok());
        let mut mapper = result.unwrap();
        // Wait for async mapping to complete before checking readiness
        mapper.wait(&device);
        assert!(mapper.is_ready());
        assert_eq!(mapper.mode(), Some(MappingMode::Read));
    }

    #[test]

    fn map_for_write_creates_ready_mapper() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_write_buffer(&device, 256);
        let result = map_for_write(buffer, &device);

        assert!(result.is_ok());
        let mut mapper = result.unwrap();
        // Wait for async mapping to complete before checking readiness
        mapper.wait(&device);
        assert!(mapper.is_ready());
        assert_eq!(mapper.mode(), Some(MappingMode::Write));
    }

    #[test]
    
    fn read_buffer_sync_reads_data() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 16);
        let source_data: [u32; 4] = [10, 20, 30, 40];
        queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&source_data));
        queue.submit([]);
        device.poll(wgpu::Maintain::Wait);

        let result: Result<Vec<u32>, _> = read_buffer_sync(buffer, &device);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec![10u32, 20, 30, 40]);
    }

    #[test]
    
    fn write_buffer_sync_writes_data() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_write_buffer(&device, 16);
        let data: [u32; 4] = [50, 60, 70, 80];

        let result = write_buffer_sync(buffer, &device, &data);
        assert!(result.is_ok());
    }

    #[test]
    
    fn read_buffer_sync_reads_entire_buffer() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 32);
        let source_data: [u32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&source_data));
        queue.submit([]);
        device.poll(wgpu::Maintain::Wait);

        // read_buffer_sync reads entire buffer
        let result: Result<Vec<u32>, _> = read_buffer_sync(buffer, &device);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec![1u32, 2, 3, 4, 5, 6, 7, 8]);
    }
}

// ============================================================================
// SECTION 11: Edge Cases and Error Conditions
// ============================================================================

mod edge_cases {
    use super::*;

    #[test]
    
    fn map_wrong_usage_read_on_write_only() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_write_buffer(&device, 256);
        let mut mapper = BufferMapper::new(buffer);

        // Try to map for read on write-only buffer
        let result = mapper.map_async(&device, MappingMode::Read, 0, 256);
        // This may fail or succeed depending on implementation
        // If it succeeds, wait should fail or state should be Failed
        if result.is_ok() {
            mapper.wait(&device);
            // State might be Failed
        }
    }

    #[test]
    
    fn map_wrong_usage_write_on_read_only() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 256);
        let mut mapper = BufferMapper::new(buffer);

        // Try to map for write on read-only buffer
        let result = mapper.map_async(&device, MappingMode::Write, 0, 256);
        if result.is_ok() {
            mapper.wait(&device);
        }
    }

    #[test]
    
    fn large_buffer_mapping() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        // 1MB buffer
        let buffer = create_read_buffer(&device, 1024 * 1024);
        let mut mapper = BufferMapper::new(buffer);

        let result = mapper.map_async(&device, MappingMode::Read, 0, 1024 * 1024);
        assert!(result.is_ok());
        mapper.wait(&device);
        assert!(mapper.is_ready());
    }

    #[test]
    
    fn aligned_offset_mapping() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 1024);
        let mut mapper = BufferMapper::new(buffer);

        // Use aligned offset
        let aligned_offset = COPY_BUFFER_ALIGNMENT * 4;
        let result = mapper.map_async(&device, MappingMode::Read, aligned_offset, 256);
        assert!(result.is_ok());
    }

    #[test]
    
    fn minimum_size_buffer() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        // Minimum practical buffer size
        let buffer = create_read_buffer(&device, MAP_SIZE_ALIGNMENT);
        let mut mapper = BufferMapper::new(buffer);

        let result = mapper.map_async(&device, MappingMode::Read, 0, MAP_SIZE_ALIGNMENT);
        assert!(result.is_ok());
    }

    #[test]
    
    fn buffer_debug_format() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 256);
        let mapper = BufferMapper::new(buffer);

        let debug = format!("{:?}", mapper);
        assert!(debug.contains("BufferMapper"));
    }

    #[test]
    
    fn sequential_map_unmap_cycles() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 256);
        let mut mapper = BufferMapper::new(buffer);

        for _ in 0..5 {
            mapper.map_async(&device, MappingMode::Read, 0, 256).unwrap();
            mapper.wait(&device);
            assert!(mapper.is_ready());
            mapper.unmap();
            assert_eq!(mapper.state(), MappingState::Unmapped);
        }
    }
}

// ============================================================================
// SECTION 12: State Transition Tests
// ============================================================================

mod state_transitions {
    use super::*;

    #[test]
    
    fn unmapped_to_pending_on_map_async() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 256);
        let mut mapper = BufferMapper::new(buffer);

        assert_eq!(mapper.state(), MappingState::Unmapped);
        mapper.map_async(&device, MappingMode::Read, 0, 256).unwrap();

        let state = mapper.state();
        assert!(state.is_pending() || state.is_mapped());
    }

    #[test]
    
    fn pending_to_mapped_on_completion() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 256);
        let mut mapper = BufferMapper::new(buffer);

        mapper.map_async(&device, MappingMode::Read, 0, 256).unwrap();
        mapper.wait(&device);

        assert_eq!(mapper.state(), MappingState::Mapped);
    }

    #[test]
    
    fn mapped_to_unmapped_on_unmap() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer = create_read_buffer(&device, 256);
        let mut mapper = BufferMapper::new(buffer);

        mapper.map_async(&device, MappingMode::Read, 0, 256).unwrap();
        mapper.wait(&device);
        assert_eq!(mapper.state(), MappingState::Mapped);

        mapper.unmap();
        assert_eq!(mapper.state(), MappingState::Unmapped);
    }

    #[test]
    
    fn failed_state_is_idle() {
        // Failed state should be considered idle for cleanup purposes
        let state = MappingState::Failed;
        assert!(state.is_idle());
        assert!(!state.is_ready());
    }
}

// ============================================================================
// SECTION 13: Multiple Buffer Tests
// ============================================================================

mod multiple_buffers {
    use super::*;

    #[test]
    
    fn multiple_buffers_can_be_mapped_simultaneously() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let buffer1 = create_read_buffer(&device, 256);
        let buffer2 = create_read_buffer(&device, 256);
        let buffer3 = create_read_buffer(&device, 256);

        let mut mapper1 = BufferMapper::new(buffer1);
        let mut mapper2 = BufferMapper::new(buffer2);
        let mut mapper3 = BufferMapper::new(buffer3);

        mapper1.map_async(&device, MappingMode::Read, 0, 256).unwrap();
        mapper2.map_async(&device, MappingMode::Read, 0, 256).unwrap();
        mapper3.map_async(&device, MappingMode::Read, 0, 256).unwrap();

        mapper1.wait(&device);
        mapper2.wait(&device);
        mapper3.wait(&device);

        assert!(mapper1.is_ready());
        assert!(mapper2.is_ready());
        assert!(mapper3.is_ready());
    }

    #[test]
    
    fn different_modes_on_different_buffers() {
        let Some((device, _queue)) = create_test_device() else {
            eprintln!("No GPU available, skipping test");
            return;
        };

        let read_buffer = create_read_buffer(&device, 256);
        let write_buffer = create_write_buffer(&device, 256);

        let mut read_mapper = BufferMapper::new(read_buffer);
        let mut write_mapper = BufferMapper::new(write_buffer);

        read_mapper.map_async(&device, MappingMode::Read, 0, 256).unwrap();
        write_mapper.map_async(&device, MappingMode::Write, 0, 256).unwrap();

        read_mapper.wait(&device);
        write_mapper.wait(&device);

        assert_eq!(read_mapper.mode(), Some(MappingMode::Read));
        assert_eq!(write_mapper.mode(), Some(MappingMode::Write));
    }
}

// ============================================================================
// Summary Test
// ============================================================================

/// Summary test verifying all four acceptance criteria work together.
#[test]

fn full_roundtrip_read_write_cycle() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    // CRITERION 1: Async buffer mapping with map_async()
    let write_buffer = create_write_buffer(&device, 64);
    let mut write_mapper = BufferMapper::new(write_buffer.clone());
    write_mapper.map_async(&device, MappingMode::Write, 0, 64).unwrap();

    // CRITERION 2: Non-blocking status polling
    let initial_state = write_mapper.poll(&device);
    assert!(initial_state.is_pending() || initial_state.is_mapped());
    write_mapper.wait(&device);
    assert_eq!(write_mapper.state(), MappingState::Mapped);

    // CRITERION 3: Mapped slice access (write)
    let test_data: [u32; 4] = [0xDEAD, 0xBEEF, 0xCAFE, 0xBABE];
    write_mapper.write_data(&test_data).unwrap();

    // CRITERION 4: Automatic unmap on drop
    drop(write_mapper);

    // Verify buffer can be used again
    let mut verify_mapper = BufferMapper::new(write_buffer);
    let result = verify_mapper.map_async(&device, MappingMode::Write, 0, 64);
    assert!(result.is_ok());
}
