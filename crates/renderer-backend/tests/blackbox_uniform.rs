// Blackbox contract tests for T-WGPU-P2.1.5 Dynamic Uniform Buffers.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::resources::*` -- no internal fields,
// no private methods, no implementation details.
//
// Acceptance criteria (T-WGPU-P2.1.5):
//   - UNIFORM_ALIGNMENT constant (256 bytes)
//   - align_uniform_offset(offset: u64) -> u64
//   - dynamic_offset_for_object(index: u32, size: u64) -> DynamicOffset
//   - uniform_buffer_size_for_objects(count: u32, size: u64) -> u64
//   - ObjectTransform struct with model/normal matrices
//   - uniform_binding_type_dynamic() -> BindingType
//   - uniform_binding_type_static() -> BindingType
//
// Coverage:
//   1. Alignment constant value verification
//   2. Offset alignment for various inputs
//   3. Dynamic offset calculation for indexed objects
//   4. Buffer sizing for multiple objects
//   5. ObjectTransform struct construction and fields
//   6. Binding type helpers (dynamic vs static)

use renderer_backend::resources::{
    align_uniform_offset, aligned_uniform_size, dynamic_offset_for_object,
    uniform_binding_type_dynamic, uniform_binding_type_static,
    uniform_buffer_size_for_objects, CameraUniform, ObjectTransform, UNIFORM_ALIGNMENT,
};
use wgpu::BindingType;

// =============================================================================
// 1. Alignment Constant Tests
// =============================================================================

#[test]
fn test_uniform_alignment_constant_is_256() {
    // The uniform alignment constant must be 256 bytes per wgpu/WebGPU spec
    assert_eq!(UNIFORM_ALIGNMENT, 256);
}

#[test]
fn test_uniform_alignment_is_power_of_two() {
    // Alignment must be a power of two for bitwise operations
    assert!(UNIFORM_ALIGNMENT.is_power_of_two());
}

// =============================================================================
// 2. Offset Alignment Tests
// =============================================================================

#[test]
fn test_align_uniform_offset_zero() {
    // Zero offset should remain zero (already aligned)
    let aligned = align_uniform_offset(0);
    assert_eq!(aligned, 0, "Zero offset should align to 0");
}

#[test]
fn test_align_uniform_offset_small_value() {
    // 100 bytes should round up to 256
    let aligned = align_uniform_offset(100);
    assert_eq!(aligned, 256, "100 should align to 256");
}

#[test]
fn test_align_uniform_offset_exact_boundary() {
    // 256 bytes should stay at 256
    let aligned = align_uniform_offset(256);
    assert_eq!(aligned, 256, "256 should remain 256");
}

#[test]
fn test_align_uniform_offset_just_over_boundary() {
    // 257 bytes should round up to 512
    let aligned = align_uniform_offset(257);
    assert_eq!(aligned, 512, "257 should align to 512");
}

#[test]
fn test_align_uniform_offset_multiple_boundaries() {
    // Test various values across multiple alignment boundaries
    assert_eq!(align_uniform_offset(1), 256);
    assert_eq!(align_uniform_offset(255), 256);
    assert_eq!(align_uniform_offset(512), 512);
    assert_eq!(align_uniform_offset(513), 768);
    assert_eq!(align_uniform_offset(1024), 1024);
}

#[test]
fn test_aligned_uniform_size_zero() {
    // Zero size input - implementation rounds up zero to the next alignment boundary
    // which is also 0 (since 0 % 256 == 0)
    let size = aligned_uniform_size(0);
    assert_eq!(size, 0, "Zero size should remain zero (already aligned)");
}

#[test]
fn test_aligned_uniform_size_small() {
    // Small sizes should round up to alignment
    let size = aligned_uniform_size(64);
    assert_eq!(size, 256, "64 bytes should align to 256");
}

#[test]
fn test_aligned_uniform_size_exact() {
    // Exact alignment size should stay the same
    let size = aligned_uniform_size(256);
    assert_eq!(size, 256, "256 bytes should remain 256");
}

// =============================================================================
// 3. Dynamic Offset Tests
// =============================================================================

#[test]
fn test_dynamic_offset_object_zero() {
    // Object 0 with size 128 -> offset 0
    let offset = dynamic_offset_for_object(0, 128);
    assert_eq!(offset, 0, "Object 0 should have offset 0");
}

#[test]
fn test_dynamic_offset_object_one() {
    // Object 1 with size 128 -> offset 256 (aligned)
    let offset = dynamic_offset_for_object(1, 128);
    assert_eq!(offset, 256, "Object 1 with size 128 should have offset 256");
}

#[test]
fn test_dynamic_offset_object_two() {
    // Object 2 with size 128 -> offset 512
    let offset = dynamic_offset_for_object(2, 128);
    assert_eq!(offset, 512, "Object 2 with size 128 should have offset 512");
}

#[test]
fn test_dynamic_offset_sequential_objects() {
    // Sequential objects should have increasing offsets by alignment stride
    let size = 64u64;
    for i in 0..10u32 {
        let offset = dynamic_offset_for_object(i, size);
        let expected = (i as u32) * (UNIFORM_ALIGNMENT as u32);
        assert_eq!(offset, expected, "Object {} offset mismatch", i);
    }
}

#[test]
fn test_dynamic_offset_large_data_size() {
    // Large data size (300 bytes) should use 512-byte stride
    let offset_0 = dynamic_offset_for_object(0, 300);
    let offset_1 = dynamic_offset_for_object(1, 300);
    let offset_2 = dynamic_offset_for_object(2, 300);

    assert_eq!(offset_0, 0);
    assert_eq!(offset_1, 512); // 300 rounds up to 512
    assert_eq!(offset_2, 1024);
}

// =============================================================================
// 4. Buffer Sizing Tests
// =============================================================================

#[test]
fn test_buffer_size_single_object_small() {
    // 1 object of 64 bytes needs 256 bytes (minimum alignment)
    let size = uniform_buffer_size_for_objects(1, 64);
    assert_eq!(size, 256, "1 object of 64 bytes should need 256 bytes");
}

#[test]
fn test_buffer_size_ten_objects_small() {
    // 10 objects of 64 bytes needs 10 * 256 = 2560 bytes
    let size = uniform_buffer_size_for_objects(10, 64);
    assert_eq!(size, 2560, "10 objects of 64 bytes should need 2560 bytes");
}

#[test]
fn test_buffer_size_zero_objects() {
    // 0 objects should need 0 bytes
    let size = uniform_buffer_size_for_objects(0, 64);
    assert_eq!(size, 0, "0 objects should need 0 bytes");
}

#[test]
fn test_buffer_size_large_data() {
    // 5 objects of 300 bytes: each needs 512 (aligned), total 2560
    let size = uniform_buffer_size_for_objects(5, 300);
    assert_eq!(size, 2560, "5 objects of 300 bytes should need 2560 bytes");
}

#[test]
fn test_buffer_size_exact_alignment() {
    // 3 objects of exactly 256 bytes: 3 * 256 = 768
    let size = uniform_buffer_size_for_objects(3, 256);
    assert_eq!(size, 768, "3 objects of 256 bytes should need 768 bytes");
}

// =============================================================================
// 5. ObjectTransform Tests
// =============================================================================

#[test]
fn test_object_transform_identity() {
    // Can create ObjectTransform with identity()
    let transform = ObjectTransform::identity();

    // Verify it has model and normal fields (identity matrices)
    // Model matrix should be 4x4 identity
    let model = transform.model;
    assert_eq!(model[0][0], 1.0);
    assert_eq!(model[1][1], 1.0);
    assert_eq!(model[2][2], 1.0);
    assert_eq!(model[3][3], 1.0);

    // Off-diagonal should be zero
    assert_eq!(model[0][1], 0.0);
    assert_eq!(model[1][0], 0.0);
}

#[test]
fn test_object_transform_has_normal_matrix() {
    // ObjectTransform should have normal field
    let transform = ObjectTransform::identity();

    // Normal matrix for identity should also be identity (3x3 portion)
    let normal = transform.normal;
    assert_eq!(normal[0][0], 1.0);
    assert_eq!(normal[1][1], 1.0);
    assert_eq!(normal[2][2], 1.0);
}

#[test]
fn test_object_transform_default() {
    // Default should be same as identity
    let default_transform = ObjectTransform::default();
    let identity_transform = ObjectTransform::identity();

    assert_eq!(default_transform.model, identity_transform.model);
    assert_eq!(default_transform.normal, identity_transform.normal);
}

#[test]
fn test_object_transform_size_for_alignment() {
    // ObjectTransform should fit within reasonable alignment bounds
    let size = std::mem::size_of::<ObjectTransform>();
    // Model (4x4 f32 = 64 bytes) + Normal (3x4 f32 = 48 bytes) + padding
    // Should be <= 256 bytes for single-slot alignment
    assert!(
        size <= 256,
        "ObjectTransform size {} exceeds single alignment slot",
        size
    );
}

// =============================================================================
// 6. Binding Type Tests
// =============================================================================

#[test]
fn test_binding_type_dynamic_exists() {
    // Dynamic binding type should be constructable
    let binding_type = uniform_binding_type_dynamic();

    // Should be a Buffer type with dynamic offset
    match binding_type {
        BindingType::Buffer {
            has_dynamic_offset, ..
        } => {
            assert!(has_dynamic_offset, "Dynamic binding should have dynamic offset");
        }
        _ => panic!("Expected Buffer binding type for dynamic uniform"),
    }
}

#[test]
fn test_binding_type_static_exists() {
    // Static binding type should be constructable
    let binding_type = uniform_binding_type_static();

    // Should be a Buffer type without dynamic offset
    match binding_type {
        BindingType::Buffer {
            has_dynamic_offset, ..
        } => {
            assert!(
                !has_dynamic_offset,
                "Static binding should not have dynamic offset"
            );
        }
        _ => panic!("Expected Buffer binding type for static uniform"),
    }
}

#[test]
fn test_binding_types_are_uniform() {
    // Both binding types should be uniform buffers
    let dynamic = uniform_binding_type_dynamic();
    let static_type = uniform_binding_type_static();

    // Verify both are uniform buffer types
    match dynamic {
        BindingType::Buffer { ty, .. } => {
            assert_eq!(ty, wgpu::BufferBindingType::Uniform);
        }
        _ => panic!("Dynamic should be Buffer type"),
    }

    match static_type {
        BindingType::Buffer { ty, .. } => {
            assert_eq!(ty, wgpu::BufferBindingType::Uniform);
        }
        _ => panic!("Static should be Buffer type"),
    }
}

// =============================================================================
// 7. CameraUniform Tests (bonus coverage)
// =============================================================================

#[test]
fn test_camera_uniform_default() {
    // CameraUniform should have a default
    let _camera = CameraUniform::default();
    // Existence test passes if it compiles
}

#[test]
fn test_camera_uniform_size() {
    // CameraUniform should fit within alignment
    let size = std::mem::size_of::<CameraUniform>();
    assert!(
        size <= 512, // Allow for view + projection + position
        "CameraUniform size {} exceeds expected bounds",
        size
    );
}

// =============================================================================
// 8. Edge Cases and Stress Tests
// =============================================================================

#[test]
fn test_alignment_large_offset() {
    // Large offset should still align correctly
    let offset = 1_000_000u64;
    let aligned = align_uniform_offset(offset);
    assert_eq!(aligned % UNIFORM_ALIGNMENT, 0);
    assert!(aligned >= offset);
    assert!(aligned < offset + UNIFORM_ALIGNMENT);
}

#[test]
fn test_dynamic_offset_many_objects() {
    // Verify offsets for many objects
    let count = 1000u32;
    let size = 128u64;

    for i in 0..count {
        let offset = dynamic_offset_for_object(i, size);
        // Each offset should be aligned
        assert_eq!(
            offset as u64 % UNIFORM_ALIGNMENT,
            0,
            "Offset for object {} not aligned",
            i
        );
        // Offsets should be sequential
        if i > 0 {
            let prev_offset = dynamic_offset_for_object(i - 1, size);
            assert!(
                offset > prev_offset,
                "Offset {} not greater than previous {}",
                offset,
                prev_offset
            );
        }
    }
}

#[test]
fn test_buffer_size_consistency() {
    // Buffer size should equal last object offset + aligned size
    let count = 5u32;
    let data_size = 100u64;

    let total_size = uniform_buffer_size_for_objects(count, data_size);
    let last_offset = dynamic_offset_for_object(count - 1, data_size);
    let aligned_data = aligned_uniform_size(data_size);

    assert_eq!(
        total_size,
        last_offset as u64 + aligned_data,
        "Buffer size should equal last offset + aligned data size"
    );
}
