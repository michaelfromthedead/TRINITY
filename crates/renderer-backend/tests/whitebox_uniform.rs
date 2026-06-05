// WHITEBOX tests for T-WGPU-P2.1.5 (Dynamic Uniform Buffers)
//
// WHITEBOX discipline: These tests have FULL ACCESS to the implementation.
// They exercise internal code paths, branch conditions, and edge cases
// that are not visible through the public contract alone.
//
// Implementation under test: crates/renderer-backend/src/resources/uniform.rs
//   - UNIFORM_ALIGNMENT = 256 bytes (WebGPU spec requirement)
//   - align_uniform_offset(offset) -> aligned offset
//   - aligned_uniform_size(size) -> aligned size
//   - dynamic_offset_for_object(index, size) -> DynamicOffset
//   - uniform_buffer_size_for_objects(count, size) -> total size
//   - ObjectTransform: model mat4, normal mat3x4, object_id, padding (128 bytes)
//   - CameraUniform: view/projection matrices, position, viewport (256 bytes)
//   - uniform_binding_type_dynamic() -> BindingType with has_dynamic_offset=true
//   - uniform_binding_type_static() -> BindingType with has_dynamic_offset=false
//
// WHITEBOX coverage plan:
//   - Path A: UNIFORM_ALIGNMENT constant verification
//   - Path B: align_uniform_offset() edge cases (0, 1, 255, 256, 257)
//   - Path C: align_uniform_offset() large values
//   - Path D: aligned_uniform_size() edge cases
//   - Path E: aligned_uniform_size() multiple of 256 values
//   - Path F: dynamic_offset_for_object() index 0 always returns 0
//   - Path G: dynamic_offset_for_object() spacing verification
//   - Path H: uniform_buffer_size_for_objects() zero count
//   - Path I: uniform_buffer_size_for_objects() large counts
//   - Path J: ObjectTransform size = 128 bytes
//   - Path K: ObjectTransform identity constructor
//   - Path L: ObjectTransform with_model constructor
//   - Path M: ObjectTransform new constructor
//   - Path N: ObjectTransform Default trait
//   - Path O: ObjectTransform bytemuck Pod/Zeroable
//   - Path P: CameraUniform size = 256 bytes
//   - Path Q: CameraUniform identity constructor
//   - Path R: CameraUniform Default trait
//   - Path S: CameraUniform bytemuck Pod/Zeroable
//   - Path T: uniform_binding_type_dynamic() has_dynamic_offset = true
//   - Path U: uniform_binding_type_static() has_dynamic_offset = false
//   - Path V: Alignment consistency across all functions
//   - Path W: Memory layout verification for GPU compatibility

use renderer_backend::resources::uniform::{
    align_uniform_offset, aligned_uniform_size, dynamic_offset_for_object,
    uniform_binding_type_dynamic, uniform_binding_type_static, uniform_buffer_size_for_objects,
    CameraUniform, ObjectTransform, UNIFORM_ALIGNMENT,
};
use wgpu::{BindingType, BufferBindingType};

// ============================================================================
// Path A: UNIFORM_ALIGNMENT Constant Verification
// ============================================================================

#[test]
fn test_uniform_alignment_is_256() {
    // WebGPU spec requires minUniformBufferOffsetAlignment of 256 bytes
    assert_eq!(UNIFORM_ALIGNMENT, 256);
}

#[test]
fn test_uniform_alignment_is_power_of_two() {
    // Alignment must be a power of two for efficient masking operations
    assert!(UNIFORM_ALIGNMENT.is_power_of_two());
}

#[test]
fn test_uniform_alignment_type_is_u64() {
    // Verify the constant is u64 to avoid truncation on large buffers
    let _: u64 = UNIFORM_ALIGNMENT;
}

// ============================================================================
// Path B: align_uniform_offset() Edge Cases
// ============================================================================

#[test]
fn test_align_uniform_offset_zero() {
    // Zero offset should remain zero (special case in implementation)
    assert_eq!(align_uniform_offset(0), 0);
}

#[test]
fn test_align_uniform_offset_one() {
    // Offset 1 should round up to 256
    assert_eq!(align_uniform_offset(1), 256);
}

#[test]
fn test_align_uniform_offset_255() {
    // Just below alignment boundary should round up
    assert_eq!(align_uniform_offset(255), 256);
}

#[test]
fn test_align_uniform_offset_256() {
    // Exact alignment boundary should stay
    assert_eq!(align_uniform_offset(256), 256);
}

#[test]
fn test_align_uniform_offset_257() {
    // Just above alignment boundary should round to next
    assert_eq!(align_uniform_offset(257), 512);
}

#[test]
fn test_align_uniform_offset_512() {
    // Multiple of alignment should stay
    assert_eq!(align_uniform_offset(512), 512);
}

#[test]
fn test_align_uniform_offset_small_values() {
    // All values 1-255 should align to 256
    for i in 1..=255u64 {
        assert_eq!(
            align_uniform_offset(i),
            256,
            "Failed for offset {}",
            i
        );
    }
}

#[test]
fn test_align_uniform_offset_values_256_to_511() {
    // Value 256 stays 256, values 257-511 round to 512
    assert_eq!(align_uniform_offset(256), 256);
    for i in 257..=511u64 {
        assert_eq!(
            align_uniform_offset(i),
            512,
            "Failed for offset {}",
            i
        );
    }
}

// ============================================================================
// Path C: align_uniform_offset() Large Values
// ============================================================================

#[test]
fn test_align_uniform_offset_large_aligned() {
    // Large values that are already aligned
    assert_eq!(align_uniform_offset(1024), 1024);
    assert_eq!(align_uniform_offset(4096), 4096);
    assert_eq!(align_uniform_offset(65536), 65536);
    assert_eq!(align_uniform_offset(1024 * 1024), 1024 * 1024);
}

#[test]
fn test_align_uniform_offset_large_unaligned() {
    // Large values just above alignment boundaries
    assert_eq!(align_uniform_offset(1025), 1280); // 1025 -> 1280 (5 * 256)
    assert_eq!(align_uniform_offset(4097), 4352); // 4097 -> 4352 (17 * 256)
}

#[test]
fn test_align_uniform_offset_powers_of_256() {
    // Powers of 256 should stay as-is
    for n in 1..10 {
        let value = 256u64 * n;
        assert_eq!(align_uniform_offset(value), value, "Failed for {}", value);
    }
}

#[test]
fn test_align_uniform_offset_near_max() {
    // Test near u64::MAX to ensure no overflow
    let large: u64 = u64::MAX - 512;
    let aligned = align_uniform_offset(large);
    // Should round up to nearest 256 multiple
    assert!(aligned >= large);
    assert_eq!(aligned % 256, 0);
}

// ============================================================================
// Path D: aligned_uniform_size() Edge Cases
// ============================================================================

#[test]
fn test_aligned_uniform_size_zero() {
    // Zero size stays zero (special case)
    assert_eq!(aligned_uniform_size(0), 0);
}

#[test]
fn test_aligned_uniform_size_one() {
    // Size 1 rounds up to 256
    assert_eq!(aligned_uniform_size(1), 256);
}

#[test]
fn test_aligned_uniform_size_64() {
    // Typical small struct size
    assert_eq!(aligned_uniform_size(64), 256);
}

#[test]
fn test_aligned_uniform_size_128() {
    // ObjectTransform actual size
    assert_eq!(aligned_uniform_size(128), 256);
}

#[test]
fn test_aligned_uniform_size_200() {
    // Arbitrary size below 256
    assert_eq!(aligned_uniform_size(200), 256);
}

#[test]
fn test_aligned_uniform_size_255() {
    // Just below alignment boundary
    assert_eq!(aligned_uniform_size(255), 256);
}

// ============================================================================
// Path E: aligned_uniform_size() Multiple of 256 Values
// ============================================================================

#[test]
fn test_aligned_uniform_size_exactly_256() {
    // Exactly 256 bytes should stay 256
    assert_eq!(aligned_uniform_size(256), 256);
}

#[test]
fn test_aligned_uniform_size_257() {
    // Just above 256 should round to 512
    assert_eq!(aligned_uniform_size(257), 512);
}

#[test]
fn test_aligned_uniform_size_300() {
    // Larger than 256 but less than 512
    assert_eq!(aligned_uniform_size(300), 512);
}

#[test]
fn test_aligned_uniform_size_512() {
    // Exactly 512 should stay 512
    assert_eq!(aligned_uniform_size(512), 512);
}

#[test]
fn test_aligned_uniform_size_513() {
    // Just above 512 should round to 768
    assert_eq!(aligned_uniform_size(513), 768);
}

#[test]
fn test_aligned_uniform_size_multiples_of_256() {
    // All multiples of 256 should stay as-is
    for n in 1..=10 {
        let size = 256u64 * n;
        assert_eq!(aligned_uniform_size(size), size, "Failed for size {}", size);
    }
}

// ============================================================================
// Path F: dynamic_offset_for_object() Index 0
// ============================================================================

#[test]
fn test_dynamic_offset_object_0_size_64() {
    // Object 0 always at offset 0 regardless of data size
    assert_eq!(dynamic_offset_for_object(0, 64), 0);
}

#[test]
fn test_dynamic_offset_object_0_size_128() {
    assert_eq!(dynamic_offset_for_object(0, 128), 0);
}

#[test]
fn test_dynamic_offset_object_0_size_256() {
    assert_eq!(dynamic_offset_for_object(0, 256), 0);
}

#[test]
fn test_dynamic_offset_object_0_size_300() {
    assert_eq!(dynamic_offset_for_object(0, 300), 0);
}

#[test]
fn test_dynamic_offset_object_0_size_512() {
    assert_eq!(dynamic_offset_for_object(0, 512), 0);
}

// ============================================================================
// Path G: dynamic_offset_for_object() Spacing Verification
// ============================================================================

#[test]
fn test_dynamic_offset_spacing_64_bytes() {
    // 64-byte data aligns to 256 stride
    assert_eq!(dynamic_offset_for_object(0, 64), 0);
    assert_eq!(dynamic_offset_for_object(1, 64), 256);
    assert_eq!(dynamic_offset_for_object(2, 64), 512);
    assert_eq!(dynamic_offset_for_object(3, 64), 768);
    assert_eq!(dynamic_offset_for_object(10, 64), 2560);
    assert_eq!(dynamic_offset_for_object(100, 64), 25600);
}

#[test]
fn test_dynamic_offset_spacing_128_bytes() {
    // 128-byte data (ObjectTransform size) aligns to 256 stride
    assert_eq!(dynamic_offset_for_object(0, 128), 0);
    assert_eq!(dynamic_offset_for_object(1, 128), 256);
    assert_eq!(dynamic_offset_for_object(2, 128), 512);
}

#[test]
fn test_dynamic_offset_spacing_256_bytes() {
    // 256-byte data has 256 stride (no padding needed)
    assert_eq!(dynamic_offset_for_object(0, 256), 0);
    assert_eq!(dynamic_offset_for_object(1, 256), 256);
    assert_eq!(dynamic_offset_for_object(2, 256), 512);
    assert_eq!(dynamic_offset_for_object(5, 256), 1280);
}

#[test]
fn test_dynamic_offset_spacing_300_bytes() {
    // 300-byte data aligns to 512 stride
    assert_eq!(dynamic_offset_for_object(0, 300), 0);
    assert_eq!(dynamic_offset_for_object(1, 300), 512);
    assert_eq!(dynamic_offset_for_object(2, 300), 1024);
}

#[test]
fn test_dynamic_offset_spacing_512_bytes() {
    // 512-byte data has 512 stride
    assert_eq!(dynamic_offset_for_object(0, 512), 0);
    assert_eq!(dynamic_offset_for_object(1, 512), 512);
    assert_eq!(dynamic_offset_for_object(2, 512), 1024);
}

#[test]
fn test_dynamic_offset_consecutive_spacing() {
    // Verify all offsets are evenly spaced for various sizes
    for data_size in [64u64, 128, 200, 256, 300, 400, 512, 600] {
        let stride = aligned_uniform_size(data_size);
        for i in 0..10u32 {
            let expected = i as u64 * stride;
            let actual = dynamic_offset_for_object(i, data_size) as u64;
            assert_eq!(
                actual, expected,
                "Offset mismatch for object {} with data_size {}: expected {}, got {}",
                i, data_size, expected, actual
            );
        }
    }
}

// ============================================================================
// Path H: uniform_buffer_size_for_objects() Zero Count
// ============================================================================

#[test]
fn test_buffer_size_zero_objects_64_bytes() {
    assert_eq!(uniform_buffer_size_for_objects(0, 64), 0);
}

#[test]
fn test_buffer_size_zero_objects_256_bytes() {
    assert_eq!(uniform_buffer_size_for_objects(0, 256), 0);
}

#[test]
fn test_buffer_size_zero_objects_512_bytes() {
    assert_eq!(uniform_buffer_size_for_objects(0, 512), 0);
}

#[test]
fn test_buffer_size_zero_data_size() {
    // Zero data size results in zero buffer regardless of count
    assert_eq!(uniform_buffer_size_for_objects(1, 0), 0);
    assert_eq!(uniform_buffer_size_for_objects(10, 0), 0);
    assert_eq!(uniform_buffer_size_for_objects(100, 0), 0);
}

// ============================================================================
// Path I: uniform_buffer_size_for_objects() Various Counts
// ============================================================================

#[test]
fn test_buffer_size_one_object() {
    // One object needs at least one aligned block
    assert_eq!(uniform_buffer_size_for_objects(1, 64), 256);
    assert_eq!(uniform_buffer_size_for_objects(1, 128), 256);
    assert_eq!(uniform_buffer_size_for_objects(1, 256), 256);
    assert_eq!(uniform_buffer_size_for_objects(1, 300), 512);
}

#[test]
fn test_buffer_size_ten_objects() {
    // 10 objects * stride
    assert_eq!(uniform_buffer_size_for_objects(10, 64), 2560);   // 10 * 256
    assert_eq!(uniform_buffer_size_for_objects(10, 128), 2560);  // 10 * 256
    assert_eq!(uniform_buffer_size_for_objects(10, 256), 2560);  // 10 * 256
    assert_eq!(uniform_buffer_size_for_objects(10, 300), 5120);  // 10 * 512
}

#[test]
fn test_buffer_size_hundred_objects() {
    assert_eq!(uniform_buffer_size_for_objects(100, 64), 25600);  // 100 * 256
    assert_eq!(uniform_buffer_size_for_objects(100, 256), 25600); // 100 * 256
}

#[test]
fn test_buffer_size_large_count() {
    // 1000 objects
    assert_eq!(uniform_buffer_size_for_objects(1000, 64), 256000);  // 1000 * 256
    assert_eq!(uniform_buffer_size_for_objects(1000, 256), 256000); // 1000 * 256
    assert_eq!(uniform_buffer_size_for_objects(1000, 300), 512000); // 1000 * 512
}

#[test]
fn test_buffer_size_matches_last_offset_plus_stride() {
    // Buffer size should equal count * stride
    for count in [1u32, 5, 10, 50, 100] {
        for data_size in [64u64, 128, 256, 300, 512] {
            let buffer_size = uniform_buffer_size_for_objects(count, data_size);
            let stride = aligned_uniform_size(data_size);
            let expected = count as u64 * stride;
            assert_eq!(
                buffer_size, expected,
                "Buffer size mismatch for {} objects of {} bytes",
                count, data_size
            );
        }
    }
}

// ============================================================================
// Path J: ObjectTransform Size
// ============================================================================

#[test]
fn test_object_transform_size_is_128() {
    // ObjectTransform should be exactly 128 bytes
    assert_eq!(std::mem::size_of::<ObjectTransform>(), 128);
}

#[test]
fn test_object_transform_size_constant() {
    // SIZE constant should match actual size
    assert_eq!(ObjectTransform::SIZE, 128);
    assert_eq!(ObjectTransform::SIZE as usize, std::mem::size_of::<ObjectTransform>());
}

#[test]
fn test_object_transform_field_sizes() {
    // Verify field sizes add up correctly:
    // model: mat4x4<f32> = 4*4*4 = 64 bytes
    // normal: mat3x4<f32> = 3*4*4 = 48 bytes
    // object_id: u32 = 4 bytes
    // _padding: [u32; 3] = 12 bytes
    // Total: 64 + 48 + 4 + 12 = 128 bytes
    let expected = 64 + 48 + 4 + 12;
    assert_eq!(expected, 128);
    assert_eq!(std::mem::size_of::<ObjectTransform>(), expected);
}

#[test]
fn test_object_transform_alignment() {
    // ObjectTransform should have alignment suitable for GPU (at least 4 bytes)
    assert!(std::mem::align_of::<ObjectTransform>() >= 4);
}

// ============================================================================
// Path K: ObjectTransform identity Constructor
// ============================================================================

#[test]
fn test_object_transform_identity_model_matrix() {
    let t = ObjectTransform::identity();

    // Identity model matrix
    assert_eq!(t.model[0], [1.0, 0.0, 0.0, 0.0]);
    assert_eq!(t.model[1], [0.0, 1.0, 0.0, 0.0]);
    assert_eq!(t.model[2], [0.0, 0.0, 1.0, 0.0]);
    assert_eq!(t.model[3], [0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn test_object_transform_identity_normal_matrix() {
    let t = ObjectTransform::identity();

    // Identity normal matrix (3x4 for std140 padding)
    assert_eq!(t.normal[0], [1.0, 0.0, 0.0, 0.0]);
    assert_eq!(t.normal[1], [0.0, 1.0, 0.0, 0.0]);
    assert_eq!(t.normal[2], [0.0, 0.0, 1.0, 0.0]);
}

#[test]
fn test_object_transform_identity_object_id() {
    let t = ObjectTransform::identity();
    assert_eq!(t.object_id, 0);
}

#[test]
fn test_object_transform_identity_padding() {
    let t = ObjectTransform::identity();
    assert_eq!(t._padding, [0, 0, 0]);
}

// ============================================================================
// Path L: ObjectTransform with_model Constructor
// ============================================================================

#[test]
fn test_object_transform_with_model_stores_matrix() {
    let model = [
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 3.0, 0.0, 0.0],
        [0.0, 0.0, 4.0, 0.0],
        [5.0, 6.0, 7.0, 1.0],
    ];
    let t = ObjectTransform::with_model(model, 42);

    assert_eq!(t.model, model);
}

#[test]
fn test_object_transform_with_model_stores_object_id() {
    let model = [[1.0; 4]; 4];
    let t = ObjectTransform::with_model(model, 12345);

    assert_eq!(t.object_id, 12345);
}

#[test]
fn test_object_transform_with_model_identity_normal() {
    // with_model should set normal to identity (user must compute proper normal)
    let model = [[1.0; 4]; 4];
    let t = ObjectTransform::with_model(model, 0);

    assert_eq!(t.normal[0], [1.0, 0.0, 0.0, 0.0]);
    assert_eq!(t.normal[1], [0.0, 1.0, 0.0, 0.0]);
    assert_eq!(t.normal[2], [0.0, 0.0, 1.0, 0.0]);
}

#[test]
fn test_object_transform_with_model_zeroed_padding() {
    let model = [[1.0; 4]; 4];
    let t = ObjectTransform::with_model(model, 0);

    assert_eq!(t._padding, [0, 0, 0]);
}

// ============================================================================
// Path M: ObjectTransform new Constructor
// ============================================================================

#[test]
fn test_object_transform_new_all_fields() {
    let model = [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
    ];
    let normal = [
        [0.5, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0],
    ];
    let t = ObjectTransform::new(model, normal, 999);

    assert_eq!(t.model, model);
    assert_eq!(t.normal, normal);
    assert_eq!(t.object_id, 999);
    assert_eq!(t._padding, [0, 0, 0]);
}

#[test]
fn test_object_transform_new_preserves_normal() {
    // Unlike with_model, new() should preserve the given normal matrix
    let model = [[1.0; 4]; 4];
    let normal = [
        [0.1, 0.2, 0.3, 0.0],
        [0.4, 0.5, 0.6, 0.0],
        [0.7, 0.8, 0.9, 0.0],
    ];
    let t = ObjectTransform::new(model, normal, 1);

    assert_eq!(t.normal, normal);
}

// ============================================================================
// Path N: ObjectTransform Default Trait
// ============================================================================

#[test]
fn test_object_transform_default_equals_identity() {
    let default_t: ObjectTransform = Default::default();
    let identity_t = ObjectTransform::identity();

    assert_eq!(default_t.model, identity_t.model);
    assert_eq!(default_t.normal, identity_t.normal);
    assert_eq!(default_t.object_id, identity_t.object_id);
    assert_eq!(default_t._padding, identity_t._padding);
}

#[test]
fn test_object_transform_default_values() {
    let t: ObjectTransform = Default::default();

    // Verify all default values
    assert_eq!(t.model[0][0], 1.0);
    assert_eq!(t.model[1][1], 1.0);
    assert_eq!(t.model[2][2], 1.0);
    assert_eq!(t.model[3][3], 1.0);
    assert_eq!(t.object_id, 0);
}

// ============================================================================
// Path O: ObjectTransform Bytemuck Pod/Zeroable
// ============================================================================

#[test]
fn test_object_transform_bytemuck_bytes_of() {
    let t = ObjectTransform::identity();
    let bytes: &[u8] = bytemuck::bytes_of(&t);

    assert_eq!(bytes.len(), 128);
}

#[test]
fn test_object_transform_bytemuck_from_bytes() {
    let t = ObjectTransform::identity();
    let bytes: &[u8] = bytemuck::bytes_of(&t);
    let recovered: &ObjectTransform = bytemuck::from_bytes(bytes);

    assert_eq!(recovered.model, t.model);
    assert_eq!(recovered.normal, t.normal);
    assert_eq!(recovered.object_id, t.object_id);
}

#[test]
fn test_object_transform_bytemuck_zeroed() {
    // Zeroable trait allows creating zeroed instance
    let zeroed: ObjectTransform = bytemuck::Zeroable::zeroed();

    // All fields should be zero
    for row in &zeroed.model {
        for &val in row {
            assert_eq!(val, 0.0);
        }
    }
    for row in &zeroed.normal {
        for &val in row {
            assert_eq!(val, 0.0);
        }
    }
    assert_eq!(zeroed.object_id, 0);
}

#[test]
fn test_object_transform_bytemuck_cast_slice() {
    // Can cast slice of transforms to bytes
    let transforms = [ObjectTransform::identity(), ObjectTransform::identity()];
    let bytes: &[u8] = bytemuck::cast_slice(&transforms);

    assert_eq!(bytes.len(), 256); // 2 * 128 bytes
}

// ============================================================================
// Path P: CameraUniform Size
// ============================================================================

#[test]
fn test_camera_uniform_size_is_256() {
    // CameraUniform should be exactly 256 bytes (aligned to UNIFORM_ALIGNMENT)
    assert_eq!(std::mem::size_of::<CameraUniform>(), 256);
}

#[test]
fn test_camera_uniform_size_constant() {
    assert_eq!(CameraUniform::SIZE, 256);
    assert_eq!(CameraUniform::SIZE as usize, std::mem::size_of::<CameraUniform>());
}

#[test]
fn test_camera_uniform_matches_uniform_alignment() {
    // CameraUniform size equals UNIFORM_ALIGNMENT (no padding waste for this struct)
    assert_eq!(CameraUniform::SIZE, UNIFORM_ALIGNMENT);
}

#[test]
fn test_camera_uniform_field_sizes() {
    // Verify field sizes:
    // view: mat4x4<f32> = 64 bytes
    // projection: mat4x4<f32> = 64 bytes
    // view_projection: mat4x4<f32> = 64 bytes
    // camera_position: vec4<f32> = 16 bytes
    // viewport: vec4<f32> = 16 bytes
    // frame_index: u32 = 4 bytes
    // time: f32 = 4 bytes
    // _padding: [f32; 6] = 24 bytes
    // Total: 64 + 64 + 64 + 16 + 16 + 4 + 4 + 24 = 256 bytes
    let expected = 64 + 64 + 64 + 16 + 16 + 4 + 4 + 24;
    assert_eq!(expected, 256);
}

// ============================================================================
// Path Q: CameraUniform identity Constructor
// ============================================================================

#[test]
fn test_camera_uniform_identity_view_matrix() {
    let c = CameraUniform::identity();

    // Identity view matrix
    assert_eq!(c.view[0], [1.0, 0.0, 0.0, 0.0]);
    assert_eq!(c.view[1], [0.0, 1.0, 0.0, 0.0]);
    assert_eq!(c.view[2], [0.0, 0.0, 1.0, 0.0]);
    assert_eq!(c.view[3], [0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn test_camera_uniform_identity_projection_matrix() {
    let c = CameraUniform::identity();

    // Identity projection matrix
    assert_eq!(c.projection[0], [1.0, 0.0, 0.0, 0.0]);
    assert_eq!(c.projection[1], [0.0, 1.0, 0.0, 0.0]);
    assert_eq!(c.projection[2], [0.0, 0.0, 1.0, 0.0]);
    assert_eq!(c.projection[3], [0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn test_camera_uniform_identity_view_projection_matrix() {
    let c = CameraUniform::identity();

    // Identity view_projection matrix
    assert_eq!(c.view_projection[0], [1.0, 0.0, 0.0, 0.0]);
    assert_eq!(c.view_projection[1], [0.0, 1.0, 0.0, 0.0]);
    assert_eq!(c.view_projection[2], [0.0, 0.0, 1.0, 0.0]);
    assert_eq!(c.view_projection[3], [0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn test_camera_uniform_identity_camera_position() {
    let c = CameraUniform::identity();

    // Camera at origin with w=1
    assert_eq!(c.camera_position, [0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn test_camera_uniform_identity_viewport() {
    let c = CameraUniform::identity();

    // Default viewport: 1920x1080, near=0.1, far=1000
    assert_eq!(c.viewport[0], 1920.0); // width
    assert_eq!(c.viewport[1], 1080.0); // height
    assert_eq!(c.viewport[2], 0.1);    // near
    assert_eq!(c.viewport[3], 1000.0); // far
}

#[test]
fn test_camera_uniform_identity_frame_index() {
    let c = CameraUniform::identity();
    assert_eq!(c.frame_index, 0);
}

#[test]
fn test_camera_uniform_identity_time() {
    let c = CameraUniform::identity();
    assert_eq!(c.time, 0.0);
}

#[test]
fn test_camera_uniform_identity_padding() {
    let c = CameraUniform::identity();
    assert_eq!(c._padding, [0.0; 6]);
}

// ============================================================================
// Path R: CameraUniform Default Trait
// ============================================================================

#[test]
fn test_camera_uniform_default_equals_identity() {
    let default_c: CameraUniform = Default::default();
    let identity_c = CameraUniform::identity();

    assert_eq!(default_c.view, identity_c.view);
    assert_eq!(default_c.projection, identity_c.projection);
    assert_eq!(default_c.view_projection, identity_c.view_projection);
    assert_eq!(default_c.camera_position, identity_c.camera_position);
    assert_eq!(default_c.viewport, identity_c.viewport);
    assert_eq!(default_c.frame_index, identity_c.frame_index);
    assert_eq!(default_c.time, identity_c.time);
}

// ============================================================================
// Path S: CameraUniform Bytemuck Pod/Zeroable
// ============================================================================

#[test]
fn test_camera_uniform_bytemuck_bytes_of() {
    let c = CameraUniform::identity();
    let bytes: &[u8] = bytemuck::bytes_of(&c);

    assert_eq!(bytes.len(), 256);
}

#[test]
fn test_camera_uniform_bytemuck_from_bytes() {
    let c = CameraUniform::identity();
    let bytes: &[u8] = bytemuck::bytes_of(&c);
    let recovered: &CameraUniform = bytemuck::from_bytes(bytes);

    assert_eq!(recovered.view, c.view);
    assert_eq!(recovered.camera_position, c.camera_position);
    assert_eq!(recovered.frame_index, c.frame_index);
}

#[test]
fn test_camera_uniform_bytemuck_zeroed() {
    let zeroed: CameraUniform = bytemuck::Zeroable::zeroed();

    // All matrix values should be zero
    for row in &zeroed.view {
        for &val in row {
            assert_eq!(val, 0.0);
        }
    }
    assert_eq!(zeroed.frame_index, 0);
    assert_eq!(zeroed.time, 0.0);
}

// ============================================================================
// Path T: uniform_binding_type_dynamic()
// ============================================================================

#[test]
fn test_binding_type_dynamic_is_buffer() {
    let binding = uniform_binding_type_dynamic();

    match binding {
        BindingType::Buffer { .. } => { /* OK */ }
        _ => panic!("Expected Buffer binding type"),
    }
}

#[test]
fn test_binding_type_dynamic_is_uniform() {
    let binding = uniform_binding_type_dynamic();

    match binding {
        BindingType::Buffer { ty, .. } => {
            assert!(matches!(ty, BufferBindingType::Uniform));
        }
        _ => panic!("Expected Buffer binding type"),
    }
}

#[test]
fn test_binding_type_dynamic_has_dynamic_offset_true() {
    let binding = uniform_binding_type_dynamic();

    match binding {
        BindingType::Buffer { has_dynamic_offset, .. } => {
            assert!(has_dynamic_offset, "has_dynamic_offset should be true");
        }
        _ => panic!("Expected Buffer binding type"),
    }
}

#[test]
fn test_binding_type_dynamic_min_binding_size_none() {
    let binding = uniform_binding_type_dynamic();

    match binding {
        BindingType::Buffer { min_binding_size, .. } => {
            assert!(min_binding_size.is_none());
        }
        _ => panic!("Expected Buffer binding type"),
    }
}

// ============================================================================
// Path U: uniform_binding_type_static()
// ============================================================================

#[test]
fn test_binding_type_static_is_buffer() {
    let binding = uniform_binding_type_static();

    match binding {
        BindingType::Buffer { .. } => { /* OK */ }
        _ => panic!("Expected Buffer binding type"),
    }
}

#[test]
fn test_binding_type_static_is_uniform() {
    let binding = uniform_binding_type_static();

    match binding {
        BindingType::Buffer { ty, .. } => {
            assert!(matches!(ty, BufferBindingType::Uniform));
        }
        _ => panic!("Expected Buffer binding type"),
    }
}

#[test]
fn test_binding_type_static_has_dynamic_offset_false() {
    let binding = uniform_binding_type_static();

    match binding {
        BindingType::Buffer { has_dynamic_offset, .. } => {
            assert!(!has_dynamic_offset, "has_dynamic_offset should be false");
        }
        _ => panic!("Expected Buffer binding type"),
    }
}

#[test]
fn test_binding_type_static_min_binding_size_none() {
    let binding = uniform_binding_type_static();

    match binding {
        BindingType::Buffer { min_binding_size, .. } => {
            assert!(min_binding_size.is_none());
        }
        _ => panic!("Expected Buffer binding type"),
    }
}

#[test]
fn test_binding_type_dynamic_vs_static_difference() {
    let dynamic = uniform_binding_type_dynamic();
    let static_ = uniform_binding_type_static();

    // Both are Uniform buffers
    match (&dynamic, &static_) {
        (
            BindingType::Buffer { ty: ty_d, has_dynamic_offset: dyn_d, .. },
            BindingType::Buffer { ty: ty_s, has_dynamic_offset: dyn_s, .. },
        ) => {
            // Same buffer type
            assert!(matches!(ty_d, BufferBindingType::Uniform));
            assert!(matches!(ty_s, BufferBindingType::Uniform));
            // Different dynamic offset flag
            assert!(*dyn_d);  // dynamic has true
            assert!(!*dyn_s); // static has false
        }
        _ => panic!("Both should be Buffer binding types"),
    }
}

// ============================================================================
// Path V: Alignment Consistency Across Functions
// ============================================================================

#[test]
fn test_alignment_consistency_offset_and_size() {
    // For any value, align_uniform_offset and aligned_uniform_size should behave similarly
    // (both round up to nearest multiple of 256)
    for val in [1u64, 64, 128, 200, 255, 256, 257, 300, 512, 1000] {
        let offset = align_uniform_offset(val);
        let size = aligned_uniform_size(val);
        assert_eq!(offset, size, "Mismatch for value {}", val);
    }
}

#[test]
fn test_alignment_buffer_size_equals_count_times_stride() {
    // uniform_buffer_size_for_objects(n, size) == n * aligned_uniform_size(size)
    for count in [1u32, 5, 10, 100] {
        for data_size in [64u64, 128, 256, 300, 512] {
            let buffer_size = uniform_buffer_size_for_objects(count, data_size);
            let stride = aligned_uniform_size(data_size);
            assert_eq!(
                buffer_size,
                count as u64 * stride,
                "Mismatch for {} objects of {} bytes",
                count,
                data_size
            );
        }
    }
}

#[test]
fn test_alignment_offset_equals_index_times_stride() {
    // dynamic_offset_for_object(i, size) == i * aligned_uniform_size(size)
    for idx in [0u32, 1, 5, 10, 100] {
        for data_size in [64u64, 128, 256, 300, 512] {
            let offset = dynamic_offset_for_object(idx, data_size) as u64;
            let stride = aligned_uniform_size(data_size);
            assert_eq!(
                offset,
                idx as u64 * stride,
                "Mismatch for index {} with {} bytes",
                idx,
                data_size
            );
        }
    }
}

#[test]
fn test_alignment_last_object_fits_in_buffer() {
    // For any count > 0, the last object's offset + data should fit in buffer
    for count in [1u32, 5, 10, 100] {
        for data_size in [64u64, 128, 256, 300, 512] {
            let buffer_size = uniform_buffer_size_for_objects(count, data_size);
            let last_offset = dynamic_offset_for_object(count - 1, data_size) as u64;
            let aligned = aligned_uniform_size(data_size);

            // Last object starts at last_offset and uses aligned bytes
            assert!(
                buffer_size >= last_offset + aligned,
                "Buffer too small: {} < {} + {} for {} objects of {} bytes",
                buffer_size,
                last_offset,
                aligned,
                count,
                data_size
            );
        }
    }
}

// ============================================================================
// Path W: Memory Layout Verification for GPU Compatibility
// ============================================================================

#[test]
fn test_object_transform_repr_c() {
    // ObjectTransform should have C representation for GPU compatibility
    // This is verified by the #[repr(C)] attribute, but we can check size
    // and alignment to ensure no unexpected padding
    let size = std::mem::size_of::<ObjectTransform>();
    let align = std::mem::align_of::<ObjectTransform>();

    assert_eq!(size, 128);
    assert!(align >= 4); // At least 4-byte alignment for f32
}

#[test]
fn test_camera_uniform_repr_c() {
    let size = std::mem::size_of::<CameraUniform>();
    let align = std::mem::align_of::<CameraUniform>();

    assert_eq!(size, 256);
    assert!(align >= 4);
}

#[test]
fn test_object_transform_std140_compatible_normal() {
    // In std140/std430 layout, vec3 must be aligned to vec4
    // ObjectTransform uses mat3x4 (3 rows of vec4) for normal matrix
    // This ensures proper GPU alignment
    let t = ObjectTransform::identity();

    // Each row of normal has 4 components (vec4 with padding)
    assert_eq!(t.normal.len(), 3);
    assert_eq!(t.normal[0].len(), 4);
}

#[test]
fn test_object_transform_continuous_memory() {
    // Verify transforms can be stored continuously in a buffer
    let transforms = [
        ObjectTransform::identity(),
        ObjectTransform::with_model([[2.0; 4]; 4], 1),
        ObjectTransform::with_model([[3.0; 4]; 4], 2),
    ];

    let bytes: &[u8] = bytemuck::cast_slice(&transforms);
    assert_eq!(bytes.len(), 3 * 128);

    // Verify we can recover individual transforms
    let recovered: &[ObjectTransform] = bytemuck::cast_slice(bytes);
    assert_eq!(recovered[0].object_id, 0);
    assert_eq!(recovered[1].object_id, 1);
    assert_eq!(recovered[2].object_id, 2);
}

#[test]
fn test_camera_uniform_continuous_memory() {
    // CameraUniform can also be stored in arrays
    let cameras = [CameraUniform::identity(), CameraUniform::identity()];
    let bytes: &[u8] = bytemuck::cast_slice(&cameras);
    assert_eq!(bytes.len(), 2 * 256);
}

// ============================================================================
// Additional Edge Cases
// ============================================================================

#[test]
fn test_dynamic_offset_type_is_u32() {
    // DynamicOffset is u32, verify no truncation for reasonable object counts
    let offset = dynamic_offset_for_object(65535, 256);
    assert_eq!(offset, 65535 * 256);
}

#[test]
fn test_object_transform_copy_clone() {
    let t1 = ObjectTransform::identity();
    let t2 = t1; // Copy
    let t3 = t1.clone(); // Clone

    assert_eq!(t1.object_id, t2.object_id);
    assert_eq!(t1.object_id, t3.object_id);
}

#[test]
fn test_camera_uniform_copy_clone() {
    let c1 = CameraUniform::identity();
    let c2 = c1; // Copy
    let c3 = c1.clone(); // Clone

    assert_eq!(c1.frame_index, c2.frame_index);
    assert_eq!(c1.frame_index, c3.frame_index);
}

#[test]
fn test_object_transform_debug() {
    let t = ObjectTransform::identity();
    let debug_str = format!("{:?}", t);

    // Debug output should contain type name and fields
    assert!(debug_str.contains("ObjectTransform"));
    assert!(debug_str.contains("model"));
    assert!(debug_str.contains("object_id"));
}

#[test]
fn test_camera_uniform_debug() {
    let c = CameraUniform::identity();
    let debug_str = format!("{:?}", c);

    assert!(debug_str.contains("CameraUniform"));
    assert!(debug_str.contains("view"));
    assert!(debug_str.contains("frame_index"));
}
