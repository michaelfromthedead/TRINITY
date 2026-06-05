// Blackbox contract tests for T-WGPU-P2.1.6 Storage Buffers.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::resources::*` -- no internal fields,
// no private methods, no implementation details.
//
// Acceptance criteria (T-WGPU-P2.1.6):
//   - STORAGE_ALIGNMENT constant (16 bytes for std430 vec4)
//   - STORAGE_DYNAMIC_ALIGNMENT constant (256 bytes for dynamic offset)
//   - align_storage_size(size: u64) -> u64
//   - align_storage_dynamic_offset(offset: u64) -> u64
//   - storage_buffer_size(element_count: u32, element_size: u64) -> u64
//   - storage_binding_type_readonly() -> BindingType
//   - storage_binding_type_readwrite() -> BindingType
//   - storage_binding_type_readonly_sized(size: u64) -> BindingType
//   - storage_binding_type_readwrite_sized(size: u64) -> BindingType
//   - storage_binding_type_dynamic_readonly() -> BindingType
//   - storage_binding_type_dynamic_readwrite() -> BindingType
//   - storage_binding_type_dynamic_readonly_sized(size: u64) -> BindingType
//   - storage_binding_type_dynamic_readwrite_sized(size: u64) -> BindingType
//   - StorageHeader struct (GPU indirect header)
//   - InstanceData struct (per-instance GPU data)
//   - DrawIndirectArgs struct (indirect draw command)
//   - DrawIndexedIndirectArgs struct (indexed indirect draw command)
//   - DispatchIndirectArgs struct (compute dispatch command)
//
// Coverage:
//   1. Alignment constant value verification
//   2. Size alignment for various inputs
//   3. Dynamic offset alignment
//   4. Buffer sizing for element arrays
//   5. Binding type helpers (readonly vs readwrite)
//   6. Binding type helpers (static vs dynamic)
//   7. Binding type helpers (sized vs unsized)
//   8. GPU struct construction and bytemuck compatibility
//   9. Integration tests (ignored - require GPU)

use renderer_backend::resources::{
    align_storage_dynamic_offset, align_storage_size, storage_binding_type_dynamic_readonly,
    storage_binding_type_dynamic_readonly_sized, storage_binding_type_dynamic_readwrite,
    storage_binding_type_dynamic_readwrite_sized, storage_binding_type_readonly,
    storage_binding_type_readonly_sized, storage_binding_type_readwrite,
    storage_binding_type_readwrite_sized, storage_buffer_size, DispatchIndirectArgs,
    DrawIndexedIndirectArgs, DrawIndirectArgs, InstanceData, StorageHeader, STORAGE_ALIGNMENT,
    STORAGE_DYNAMIC_ALIGNMENT,
};
use wgpu::BindingType;

// =============================================================================
// 1. Alignment Constant Tests
// =============================================================================

#[test]
fn test_storage_alignment_constant_is_16() {
    // Storage alignment is 16 bytes per std430 layout (vec4 alignment)
    assert_eq!(STORAGE_ALIGNMENT, 16);
}

#[test]
fn test_storage_alignment_is_power_of_two() {
    // Alignment must be a power of two for bitwise operations
    assert!(STORAGE_ALIGNMENT.is_power_of_two());
}

#[test]
fn test_storage_dynamic_alignment_constant_is_256() {
    // Dynamic offset alignment is 256 bytes per WebGPU spec
    assert_eq!(STORAGE_DYNAMIC_ALIGNMENT, 256);
}

#[test]
fn test_storage_dynamic_alignment_is_power_of_two() {
    // Dynamic alignment must be a power of two
    assert!(STORAGE_DYNAMIC_ALIGNMENT.is_power_of_two());
}

#[test]
fn test_storage_alignments_relationship() {
    // Dynamic alignment should be greater than or equal to base alignment
    assert!(STORAGE_DYNAMIC_ALIGNMENT >= STORAGE_ALIGNMENT);
    // Dynamic alignment should be a multiple of base alignment
    assert_eq!(STORAGE_DYNAMIC_ALIGNMENT % STORAGE_ALIGNMENT, 0);
}

// =============================================================================
// 2. Size Alignment Tests
// =============================================================================

#[test]
fn test_align_storage_size_zero() {
    // Zero size should remain zero (already aligned)
    let aligned = align_storage_size(0);
    assert_eq!(aligned, 0, "Zero size should align to 0");
}

#[test]
fn test_align_storage_size_small_value() {
    // 10 bytes should round up to 16
    let aligned = align_storage_size(10);
    assert_eq!(aligned, 16, "10 should align to 16");
}

#[test]
fn test_align_storage_size_exact_boundary() {
    // 16 bytes should stay at 16
    let aligned = align_storage_size(16);
    assert_eq!(aligned, 16, "16 should remain 16");
}

#[test]
fn test_align_storage_size_just_over_boundary() {
    // 17 bytes should round up to 32
    let aligned = align_storage_size(17);
    assert_eq!(aligned, 32, "17 should align to 32");
}

#[test]
fn test_align_storage_size_multiple_boundaries() {
    // Test various values across multiple alignment boundaries
    assert_eq!(align_storage_size(1), 16);
    assert_eq!(align_storage_size(15), 16);
    assert_eq!(align_storage_size(32), 32);
    assert_eq!(align_storage_size(33), 48);
    assert_eq!(align_storage_size(64), 64);
    assert_eq!(align_storage_size(100), 112); // rounds to 112 (7 * 16)
}

#[test]
fn test_align_storage_size_large_value() {
    // Large value should still align correctly
    let size = 1_000_000u64;
    let aligned = align_storage_size(size);
    assert_eq!(aligned % STORAGE_ALIGNMENT, 0);
    assert!(aligned >= size);
    assert!(aligned < size + STORAGE_ALIGNMENT);
}

// =============================================================================
// 3. Dynamic Offset Alignment Tests
// =============================================================================

#[test]
fn test_align_storage_dynamic_offset_zero() {
    // Zero offset should remain zero
    let aligned = align_storage_dynamic_offset(0);
    assert_eq!(aligned, 0, "Zero offset should align to 0");
}

#[test]
fn test_align_storage_dynamic_offset_small_value() {
    // 100 bytes should round up to 256
    let aligned = align_storage_dynamic_offset(100);
    assert_eq!(aligned, 256, "100 should align to 256");
}

#[test]
fn test_align_storage_dynamic_offset_exact_boundary() {
    // 256 bytes should stay at 256
    let aligned = align_storage_dynamic_offset(256);
    assert_eq!(aligned, 256, "256 should remain 256");
}

#[test]
fn test_align_storage_dynamic_offset_just_over_boundary() {
    // 257 bytes should round up to 512
    let aligned = align_storage_dynamic_offset(257);
    assert_eq!(aligned, 512, "257 should align to 512");
}

#[test]
fn test_align_storage_dynamic_offset_multiple_boundaries() {
    // Test various values across multiple alignment boundaries
    assert_eq!(align_storage_dynamic_offset(1), 256);
    assert_eq!(align_storage_dynamic_offset(255), 256);
    assert_eq!(align_storage_dynamic_offset(512), 512);
    assert_eq!(align_storage_dynamic_offset(513), 768);
    assert_eq!(align_storage_dynamic_offset(1024), 1024);
}

#[test]
fn test_align_storage_dynamic_offset_large_value() {
    // Large offset should still align correctly
    let offset = 1_000_000u64;
    let aligned = align_storage_dynamic_offset(offset);
    assert_eq!(aligned % STORAGE_DYNAMIC_ALIGNMENT, 0);
    assert!(aligned >= offset);
    assert!(aligned < offset + STORAGE_DYNAMIC_ALIGNMENT);
}

// =============================================================================
// 4. Buffer Sizing Tests
// =============================================================================

#[test]
fn test_storage_buffer_size_single_element() {
    // 1 element of 32 bytes needs 32 bytes (aligned to 16, so 32)
    let size = storage_buffer_size(1, 32);
    assert_eq!(size, 32, "1 element of 32 bytes should need 32 bytes");
}

#[test]
fn test_storage_buffer_size_multiple_elements() {
    // 10 elements of 32 bytes needs 320 bytes
    let size = storage_buffer_size(10, 32);
    assert_eq!(size, 320, "10 elements of 32 bytes should need 320 bytes");
}

#[test]
fn test_storage_buffer_size_zero_elements() {
    // 0 elements should need 0 bytes
    let size = storage_buffer_size(0, 32);
    assert_eq!(size, 0, "0 elements should need 0 bytes");
}

#[test]
fn test_storage_buffer_size_unaligned_element() {
    // Elements should be aligned to STORAGE_ALIGNMENT
    // 10 bytes per element -> aligned to 16 bytes
    // 5 elements = 5 * 16 = 80 bytes
    let size = storage_buffer_size(5, 10);
    assert_eq!(size, 80, "5 elements of 10 bytes should need 80 bytes (aligned)");
}

#[test]
fn test_storage_buffer_size_large_count() {
    // Large element count should work correctly
    let count = 10_000u32;
    let element_size = 64u64;
    let expected = count as u64 * element_size;
    let size = storage_buffer_size(count, element_size);
    assert_eq!(size, expected);
}

// =============================================================================
// 5. Binding Type Tests - Readonly vs Readwrite
// =============================================================================

#[test]
fn test_storage_binding_type_readonly_is_buffer() {
    let binding_type = storage_binding_type_readonly();
    match binding_type {
        BindingType::Buffer { .. } => {}
        _ => panic!("Expected Buffer binding type for readonly storage"),
    }
}

#[test]
fn test_storage_binding_type_readonly_is_storage() {
    let binding_type = storage_binding_type_readonly();
    match binding_type {
        BindingType::Buffer { ty, .. } => {
            match ty {
                wgpu::BufferBindingType::Storage { read_only } => {
                    assert!(read_only, "Readonly binding should have read_only=true");
                }
                _ => panic!("Expected Storage buffer type"),
            }
        }
        _ => panic!("Expected Buffer binding type"),
    }
}

#[test]
fn test_storage_binding_type_readwrite_is_buffer() {
    let binding_type = storage_binding_type_readwrite();
    match binding_type {
        BindingType::Buffer { .. } => {}
        _ => panic!("Expected Buffer binding type for readwrite storage"),
    }
}

#[test]
fn test_storage_binding_type_readwrite_is_storage() {
    let binding_type = storage_binding_type_readwrite();
    match binding_type {
        BindingType::Buffer { ty, .. } => {
            match ty {
                wgpu::BufferBindingType::Storage { read_only } => {
                    assert!(!read_only, "Readwrite binding should have read_only=false");
                }
                _ => panic!("Expected Storage buffer type"),
            }
        }
        _ => panic!("Expected Buffer binding type"),
    }
}

// =============================================================================
// 6. Binding Type Tests - Static vs Dynamic
// =============================================================================

#[test]
fn test_storage_binding_type_readonly_no_dynamic_offset() {
    let binding_type = storage_binding_type_readonly();
    match binding_type {
        BindingType::Buffer { has_dynamic_offset, .. } => {
            assert!(!has_dynamic_offset, "Static binding should not have dynamic offset");
        }
        _ => panic!("Expected Buffer binding type"),
    }
}

#[test]
fn test_storage_binding_type_readwrite_no_dynamic_offset() {
    let binding_type = storage_binding_type_readwrite();
    match binding_type {
        BindingType::Buffer { has_dynamic_offset, .. } => {
            assert!(!has_dynamic_offset, "Static binding should not have dynamic offset");
        }
        _ => panic!("Expected Buffer binding type"),
    }
}

#[test]
fn test_storage_binding_type_dynamic_readonly_has_dynamic_offset() {
    let binding_type = storage_binding_type_dynamic_readonly();
    match binding_type {
        BindingType::Buffer { has_dynamic_offset, .. } => {
            assert!(has_dynamic_offset, "Dynamic binding should have dynamic offset");
        }
        _ => panic!("Expected Buffer binding type"),
    }
}

#[test]
fn test_storage_binding_type_dynamic_readwrite_has_dynamic_offset() {
    let binding_type = storage_binding_type_dynamic_readwrite();
    match binding_type {
        BindingType::Buffer { has_dynamic_offset, .. } => {
            assert!(has_dynamic_offset, "Dynamic binding should have dynamic offset");
        }
        _ => panic!("Expected Buffer binding type"),
    }
}

#[test]
fn test_storage_binding_type_dynamic_readonly_is_readonly() {
    let binding_type = storage_binding_type_dynamic_readonly();
    match binding_type {
        BindingType::Buffer { ty, .. } => {
            match ty {
                wgpu::BufferBindingType::Storage { read_only } => {
                    assert!(read_only, "Dynamic readonly should be read_only=true");
                }
                _ => panic!("Expected Storage buffer type"),
            }
        }
        _ => panic!("Expected Buffer binding type"),
    }
}

#[test]
fn test_storage_binding_type_dynamic_readwrite_is_readwrite() {
    let binding_type = storage_binding_type_dynamic_readwrite();
    match binding_type {
        BindingType::Buffer { ty, .. } => {
            match ty {
                wgpu::BufferBindingType::Storage { read_only } => {
                    assert!(!read_only, "Dynamic readwrite should be read_only=false");
                }
                _ => panic!("Expected Storage buffer type"),
            }
        }
        _ => panic!("Expected Buffer binding type"),
    }
}

// =============================================================================
// 7. Binding Type Tests - Sized Variants
// =============================================================================

#[test]
fn test_storage_binding_type_readonly_sized_has_min_binding_size() {
    let size = 128u64;
    let binding_type = storage_binding_type_readonly_sized(size);
    match binding_type {
        BindingType::Buffer { min_binding_size, .. } => {
            assert!(min_binding_size.is_some(), "Sized binding should have min_binding_size");
            assert_eq!(
                min_binding_size.unwrap().get(),
                size,
                "min_binding_size should match requested size"
            );
        }
        _ => panic!("Expected Buffer binding type"),
    }
}

#[test]
fn test_storage_binding_type_readwrite_sized_has_min_binding_size() {
    let size = 256u64;
    let binding_type = storage_binding_type_readwrite_sized(size);
    match binding_type {
        BindingType::Buffer { min_binding_size, .. } => {
            assert!(min_binding_size.is_some(), "Sized binding should have min_binding_size");
            assert_eq!(
                min_binding_size.unwrap().get(),
                size,
                "min_binding_size should match requested size"
            );
        }
        _ => panic!("Expected Buffer binding type"),
    }
}

#[test]
fn test_storage_binding_type_dynamic_readonly_sized_has_both() {
    let size = 512u64;
    let binding_type = storage_binding_type_dynamic_readonly_sized(size);
    match binding_type {
        BindingType::Buffer { has_dynamic_offset, min_binding_size, ty, .. } => {
            assert!(has_dynamic_offset, "Should have dynamic offset");
            assert!(min_binding_size.is_some(), "Should have min_binding_size");
            assert_eq!(min_binding_size.unwrap().get(), size);
            match ty {
                wgpu::BufferBindingType::Storage { read_only } => {
                    assert!(read_only, "Should be read_only");
                }
                _ => panic!("Expected Storage buffer type"),
            }
        }
        _ => panic!("Expected Buffer binding type"),
    }
}

#[test]
fn test_storage_binding_type_dynamic_readwrite_sized_has_both() {
    let size = 1024u64;
    let binding_type = storage_binding_type_dynamic_readwrite_sized(size);
    match binding_type {
        BindingType::Buffer { has_dynamic_offset, min_binding_size, ty, .. } => {
            assert!(has_dynamic_offset, "Should have dynamic offset");
            assert!(min_binding_size.is_some(), "Should have min_binding_size");
            assert_eq!(min_binding_size.unwrap().get(), size);
            match ty {
                wgpu::BufferBindingType::Storage { read_only } => {
                    assert!(!read_only, "Should be read_write");
                }
                _ => panic!("Expected Storage buffer type"),
            }
        }
        _ => panic!("Expected Buffer binding type"),
    }
}

#[test]
fn test_storage_binding_type_unsized_has_no_min_binding_size() {
    // Unsized variants should have min_binding_size = None
    let readonly = storage_binding_type_readonly();
    let readwrite = storage_binding_type_readwrite();
    let dynamic_readonly = storage_binding_type_dynamic_readonly();
    let dynamic_readwrite = storage_binding_type_dynamic_readwrite();

    for (name, binding_type) in [
        ("readonly", readonly),
        ("readwrite", readwrite),
        ("dynamic_readonly", dynamic_readonly),
        ("dynamic_readwrite", dynamic_readwrite),
    ] {
        match binding_type {
            BindingType::Buffer { min_binding_size, .. } => {
                assert!(
                    min_binding_size.is_none(),
                    "{} should have no min_binding_size",
                    name
                );
            }
            _ => panic!("Expected Buffer binding type for {}", name),
        }
    }
}

// =============================================================================
// 8. GPU Struct Tests - StorageHeader
// =============================================================================

#[test]
fn test_storage_header_default() {
    // StorageHeader should have Default
    let header = StorageHeader::default();
    // Default should have zero counts
    assert_eq!(header.count, 0, "Default count should be 0");
}

#[test]
fn test_storage_header_fields() {
    // StorageHeader should have accessible fields via constructor
    let header = StorageHeader::new(100, 200);
    assert_eq!(header.count, 100);
    assert_eq!(header.capacity, 200);
}

#[test]
fn test_storage_header_size() {
    // StorageHeader should be properly sized for GPU
    let size = std::mem::size_of::<StorageHeader>();
    // Should be aligned to 16 bytes (vec4)
    assert_eq!(size % 16, 0, "StorageHeader size {} not aligned to 16", size);
}

#[test]
fn test_storage_header_is_pod() {
    // StorageHeader should be Pod (plain old data) for bytemuck
    let header = StorageHeader::new(42, 100);
    // If this compiles, the type is Pod-compatible
    let bytes: &[u8] = bytemuck::bytes_of(&header);
    assert!(!bytes.is_empty());
}

#[test]
fn test_storage_header_is_zeroable() {
    // StorageHeader should be Zeroable for bytemuck
    let zeroed: StorageHeader = bytemuck::Zeroable::zeroed();
    assert_eq!(zeroed.count, 0);
}

// =============================================================================
// 9. GPU Struct Tests - InstanceData
// =============================================================================

#[test]
fn test_instance_data_default() {
    // InstanceData should have Default
    let _instance = InstanceData::default();
}

#[test]
fn test_instance_data_size() {
    // InstanceData should be properly sized for GPU
    let size = std::mem::size_of::<InstanceData>();
    // Should be aligned to 16 bytes minimum
    assert_eq!(size % 16, 0, "InstanceData size {} not aligned to 16", size);
}

#[test]
fn test_instance_data_is_pod() {
    // InstanceData should be Pod for bytemuck
    let instance = InstanceData::default();
    let bytes: &[u8] = bytemuck::bytes_of(&instance);
    assert!(!bytes.is_empty());
}

#[test]
fn test_instance_data_is_zeroable() {
    // InstanceData should be Zeroable for bytemuck
    let _zeroed: InstanceData = bytemuck::Zeroable::zeroed();
}

// =============================================================================
// 10. GPU Struct Tests - DrawIndirectArgs
// =============================================================================

#[test]
fn test_draw_indirect_args_default() {
    let args = DrawIndirectArgs::default();
    // Default should have zero values
    assert_eq!(args.vertex_count, 0);
    assert_eq!(args.instance_count, 0);
    assert_eq!(args.first_vertex, 0);
    assert_eq!(args.first_instance, 0);
}

#[test]
fn test_draw_indirect_args_construction() {
    // Use constructor instead of field-level initialization (has padding fields)
    let args = DrawIndirectArgs::new(36, 100, 0, 0);
    assert_eq!(args.vertex_count, 36);
    assert_eq!(args.instance_count, 100);
}

#[test]
fn test_draw_indirect_args_size() {
    // DrawIndirectArgs should be exactly 16 bytes (4 x u32)
    let size = std::mem::size_of::<DrawIndirectArgs>();
    assert_eq!(size, 16, "DrawIndirectArgs should be 16 bytes, got {}", size);
}

#[test]
fn test_draw_indirect_args_is_pod() {
    let args = DrawIndirectArgs::new(6, 1, 0, 0);
    let bytes: &[u8] = bytemuck::bytes_of(&args);
    assert_eq!(bytes.len(), 16);
}

#[test]
fn test_draw_indirect_args_is_zeroable() {
    let zeroed: DrawIndirectArgs = bytemuck::Zeroable::zeroed();
    assert_eq!(zeroed.vertex_count, 0);
    assert_eq!(zeroed.instance_count, 0);
}

// =============================================================================
// 11. GPU Struct Tests - DrawIndexedIndirectArgs
// =============================================================================

#[test]
fn test_draw_indexed_indirect_args_default() {
    let args = DrawIndexedIndirectArgs::default();
    assert_eq!(args.index_count, 0);
    assert_eq!(args.instance_count, 0);
    assert_eq!(args.first_index, 0);
    assert_eq!(args.base_vertex, 0);
    assert_eq!(args.first_instance, 0);
}

#[test]
fn test_draw_indexed_indirect_args_construction() {
    // Use constructor instead of field-level initialization (has padding fields)
    let args = DrawIndexedIndirectArgs::new(36, 100, 0, 0, 50);
    assert_eq!(args.index_count, 36);
    assert_eq!(args.instance_count, 100);
    assert_eq!(args.first_instance, 50);
}

#[test]
fn test_draw_indexed_indirect_args_size() {
    // DrawIndexedIndirectArgs should be 24 bytes (5 x u32 + 1 padding for alignment)
    let size = std::mem::size_of::<DrawIndexedIndirectArgs>();
    assert_eq!(size, 24, "DrawIndexedIndirectArgs should be 24 bytes, got {}", size);
}

#[test]
fn test_draw_indexed_indirect_args_is_pod() {
    let args = DrawIndexedIndirectArgs::default();
    let bytes: &[u8] = bytemuck::bytes_of(&args);
    // 24 bytes due to alignment padding
    assert_eq!(bytes.len(), 24);
}

#[test]
fn test_draw_indexed_indirect_args_is_zeroable() {
    let zeroed: DrawIndexedIndirectArgs = bytemuck::Zeroable::zeroed();
    assert_eq!(zeroed.index_count, 0);
}

// =============================================================================
// 12. GPU Struct Tests - DispatchIndirectArgs
// =============================================================================

#[test]
fn test_dispatch_indirect_args_default() {
    let args = DispatchIndirectArgs::default();
    assert_eq!(args.x, 0);
    assert_eq!(args.y, 0);
    assert_eq!(args.z, 0);
}

#[test]
fn test_dispatch_indirect_args_construction() {
    // Use constructor instead of field-level initialization (has padding)
    let args = DispatchIndirectArgs::new(64, 64, 1);
    assert_eq!(args.x, 64);
    assert_eq!(args.y, 64);
    assert_eq!(args.z, 1);
}

#[test]
fn test_dispatch_indirect_args_size() {
    // DispatchIndirectArgs should be 16 bytes (3 x u32 + 1 padding for alignment)
    let size = std::mem::size_of::<DispatchIndirectArgs>();
    assert_eq!(size, 16, "DispatchIndirectArgs should be 16 bytes, got {}", size);
}

#[test]
fn test_dispatch_indirect_args_is_pod() {
    let args = DispatchIndirectArgs::new(1, 1, 1);
    let bytes: &[u8] = bytemuck::bytes_of(&args);
    // 16 bytes due to padding for alignment (not 12)
    assert_eq!(bytes.len(), 16);
}

#[test]
fn test_dispatch_indirect_args_is_zeroable() {
    let zeroed: DispatchIndirectArgs = bytemuck::Zeroable::zeroed();
    assert_eq!(zeroed.x, 0);
    assert_eq!(zeroed.y, 0);
    assert_eq!(zeroed.z, 0);
}

// =============================================================================
// 13. Edge Cases and Stress Tests
// =============================================================================

#[test]
fn test_alignment_boundary_values() {
    // Test exact boundary values
    for boundary in [16u64, 32, 48, 64, 128, 256, 512, 1024] {
        let aligned = align_storage_size(boundary);
        assert_eq!(aligned, boundary, "Exact boundary {} should remain unchanged", boundary);
    }
}

#[test]
fn test_alignment_near_boundary_values() {
    // Test values just below boundaries
    for boundary in [16u64, 32, 48, 64, 128, 256] {
        let below = boundary - 1;
        let aligned = align_storage_size(below);
        assert_eq!(aligned, boundary, "{} should align to {}", below, boundary);
    }
}

#[test]
fn test_dynamic_offset_sequential_correctness() {
    // Dynamic offsets should be sequential at STORAGE_DYNAMIC_ALIGNMENT intervals
    let offsets: Vec<u64> = (0..10)
        .map(|i| align_storage_dynamic_offset(i * 100))
        .collect();

    // Each offset should be at a valid dynamic alignment boundary
    for (i, &offset) in offsets.iter().enumerate() {
        assert_eq!(
            offset % STORAGE_DYNAMIC_ALIGNMENT,
            0,
            "Offset {} at index {} not aligned to {}",
            offset,
            i,
            STORAGE_DYNAMIC_ALIGNMENT
        );
    }
}

#[test]
fn test_buffer_size_array_consistency() {
    // Buffer size for N elements should be N * aligned_element_size
    let element_size = 48u64;
    let aligned_element = align_storage_size(element_size);

    for count in [1u32, 5, 10, 100, 1000] {
        let size = storage_buffer_size(count, element_size);
        let expected = count as u64 * aligned_element;
        assert_eq!(
            size, expected,
            "{} elements of {} bytes should need {} bytes",
            count, element_size, expected
        );
    }
}

#[test]
fn test_binding_type_combinations_complete() {
    // All 8 binding type combinations should be valid
    let bindings = [
        ("readonly", storage_binding_type_readonly()),
        ("readwrite", storage_binding_type_readwrite()),
        ("readonly_sized", storage_binding_type_readonly_sized(64)),
        ("readwrite_sized", storage_binding_type_readwrite_sized(64)),
        ("dynamic_readonly", storage_binding_type_dynamic_readonly()),
        ("dynamic_readwrite", storage_binding_type_dynamic_readwrite()),
        ("dynamic_readonly_sized", storage_binding_type_dynamic_readonly_sized(64)),
        ("dynamic_readwrite_sized", storage_binding_type_dynamic_readwrite_sized(64)),
    ];

    for (name, binding) in bindings {
        match binding {
            BindingType::Buffer { ty, .. } => {
                match ty {
                    wgpu::BufferBindingType::Storage { .. } => {}
                    _ => panic!("{} should be Storage type", name),
                }
            }
            _ => panic!("{} should be Buffer type", name),
        }
    }
}

#[test]
fn test_gpu_structs_slice_conversion() {
    // GPU structs should be convertible to byte slices for buffer upload
    let draws = [
        DrawIndirectArgs::new(6, 1, 0, 0),
        DrawIndirectArgs::new(36, 10, 6, 1),
    ];

    let bytes: &[u8] = bytemuck::cast_slice(&draws);
    assert_eq!(bytes.len(), 32); // 2 * 16 bytes
}

#[test]
fn test_draw_indexed_indirect_slice_conversion() {
    let draws = [
        DrawIndexedIndirectArgs::default(),
        DrawIndexedIndirectArgs::new(36, 1, 0, 0, 0),
    ];

    let bytes: &[u8] = bytemuck::cast_slice(&draws);
    // 2 * 24 bytes due to alignment padding
    assert_eq!(bytes.len(), 48);
}

#[test]
fn test_dispatch_indirect_slice_conversion() {
    let dispatches = [
        DispatchIndirectArgs::new(64, 64, 1),
        DispatchIndirectArgs::new(128, 1, 1),
    ];

    let bytes: &[u8] = bytemuck::cast_slice(&dispatches);
    // 2 * 16 bytes due to alignment padding
    assert_eq!(bytes.len(), 32);
}

// =============================================================================
// 14. Integration Tests (require GPU - ignored by default)
// =============================================================================

#[test]

fn test_integration_create_storage_buffer() {
    // This test would create an actual storage buffer with wgpu
    // Requires a wgpu::Device which needs adapter initialization
    //
    // Documented behavior to test:
    // 1. Create buffer with BufferUsages::STORAGE
    // 2. Buffer size must be aligned to STORAGE_ALIGNMENT
    // 3. Buffer can be bound to storage binding in bind group

    // Placeholder - would use pollster::block_on to create device
    panic!("GPU integration test not implemented - placeholder for manual testing");
}

#[test]

fn test_integration_bind_storage_buffer_readonly() {
    // Test binding a storage buffer as readonly
    //
    // 1. Create bind group layout with storage_binding_type_readonly()
    // 2. Create storage buffer
    // 3. Create bind group with buffer binding
    // 4. Verify bind group creation succeeds

    panic!("GPU integration test not implemented - placeholder for manual testing");
}

#[test]

fn test_integration_bind_storage_buffer_readwrite() {
    // Test binding a storage buffer as read-write
    //
    // 1. Create bind group layout with storage_binding_type_readwrite()
    // 2. Create storage buffer
    // 3. Create bind group with buffer binding
    // 4. Verify bind group creation succeeds

    panic!("GPU integration test not implemented - placeholder for manual testing");
}

#[test]

fn test_integration_dynamic_storage_offset() {
    // Test dynamic offset binding for storage buffers
    //
    // 1. Create bind group layout with storage_binding_type_dynamic_readonly()
    // 2. Create large storage buffer for multiple objects
    // 3. Create bind group
    // 4. Set dynamic offsets at aligned boundaries
    // 5. Verify offsets work in render/compute pass

    panic!("GPU integration test not implemented - placeholder for manual testing");
}

#[test]

fn test_integration_indirect_draw_buffer() {
    // Test using DrawIndirectArgs in actual indirect draw
    //
    // 1. Create storage buffer with DrawIndirectArgs data
    // 2. Execute draw_indirect or draw_indexed_indirect
    // 3. Verify correct number of vertices/instances rendered

    panic!("GPU integration test not implemented - placeholder for manual testing");
}

#[test]

fn test_integration_indirect_dispatch_buffer() {
    // Test using DispatchIndirectArgs in compute dispatch
    //
    // 1. Create storage buffer with DispatchIndirectArgs data
    // 2. Execute dispatch_workgroups_indirect
    // 3. Verify correct workgroup counts

    panic!("GPU integration test not implemented - placeholder for manual testing");
}
