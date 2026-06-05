//! Blackbox tests for T-WGPU-P6.2.1: ObjectData Struct
//!
//! CLEANROOM RULES:
//! - DO NOT read implementation details of `crates/renderer-backend/src/gpu_driven/object_data.rs`
//! - Only test PUBLIC API through imports
//! - Focus on behavioral contract
//!
//! Tests verify:
//! 1. Public API contract - struct existence, constructors
//! 2. Size and alignment requirements for GPU compatibility
//! 3. Field access patterns
//! 4. Bytemuck compatibility for GPU data transfer
//! 5. Builder pattern methods
//! 6. ObjectDataBuffer collection type
//! 7. object_flags module constants

use renderer_backend::gpu_driven::{
    ObjectData, ObjectDataBuffer, object_flags,
    DEFAULT_LOD_DISTANCES, INVALID_MATERIAL_INDEX, INVALID_MESH_INDEX,
    OBJECT_DATA_SIZE, OBJECT_MAX_LOD_LEVELS,
};

// ============================================================================
// SECTION 1: Public API Contract Tests
// ============================================================================

/// Verify ObjectData struct exists and is constructible
#[test]
fn blackbox_struct_exists() {
    // Constructor must work
    let obj = ObjectData::new();
    // Struct must be usable (not just a ZST)
    let _size = std::mem::size_of_val(&obj);
}

/// Verify OBJECT_DATA_SIZE constant is exported and reasonable
#[test]
fn blackbox_size_constant_exported() {
    // Size constant must be exported
    let size = OBJECT_DATA_SIZE;

    // Must be at least 144 bytes for typical GPU object data:
    // - transform: 64 bytes (4x4 f32 matrix)
    // - aabb_min/max: 24 bytes
    // - mesh/material indices: 8 bytes
    // - lod_distances: 16 bytes
    // - flags/padding: remaining
    assert!(
        size >= 144,
        "OBJECT_DATA_SIZE ({}) must be at least 144 bytes for GPU object data",
        size
    );

    // Must be 16-byte aligned for GPU uniform buffers
    assert_eq!(
        size % 16,
        0,
        "OBJECT_DATA_SIZE ({}) must be 16-byte aligned",
        size
    );
}

/// Verify ObjectData struct size matches the constant
#[test]
fn blackbox_size_aligned() {
    let obj = ObjectData::new();
    let actual_size = std::mem::size_of_val(&obj);

    // Struct size must match the exported constant
    assert_eq!(
        actual_size,
        OBJECT_DATA_SIZE,
        "ObjectData size ({}) must match OBJECT_DATA_SIZE ({})",
        actual_size,
        OBJECT_DATA_SIZE
    );

    // Must be 16-byte aligned
    assert_eq!(
        actual_size % 16,
        0,
        "ObjectData must be 16-byte aligned, got {} bytes",
        actual_size
    );

    // Must be at least 4-byte aligned (minimum for GPU)
    assert_eq!(
        std::mem::align_of::<ObjectData>() % 4,
        0,
        "ObjectData must have at least 4-byte alignment"
    );
}

/// Verify ObjectData::SIZE associated constant
#[test]
fn blackbox_size_associated_const() {
    assert_eq!(ObjectData::SIZE, OBJECT_DATA_SIZE);
}

// ============================================================================
// SECTION 2: Transform Field Tests
// ============================================================================

/// Verify transform field is accessible with identity matrix default
#[test]
fn blackbox_transform_accessible() {
    let obj = ObjectData::new();

    // Transform field must be accessible
    let transform = obj.transform;

    // Default should be identity matrix (diagonal of 1s)
    assert_eq!(transform[0][0], 1.0, "Identity matrix [0][0] should be 1.0");
    assert_eq!(transform[1][1], 1.0, "Identity matrix [1][1] should be 1.0");
    assert_eq!(transform[2][2], 1.0, "Identity matrix [2][2] should be 1.0");
    assert_eq!(transform[3][3], 1.0, "Identity matrix [3][3] should be 1.0");

    // Off-diagonal should be 0
    assert_eq!(transform[0][1], 0.0, "Identity matrix off-diagonal should be 0");
    assert_eq!(transform[1][0], 0.0, "Identity matrix off-diagonal should be 0");
}

/// Verify transform field can be modified
#[test]
fn blackbox_transform_mutable() {
    let mut obj = ObjectData::new();

    // Translation in transform matrix
    obj.transform[3][0] = 10.0; // X translation
    obj.transform[3][1] = 20.0; // Y translation
    obj.transform[3][2] = 30.0; // Z translation

    assert_eq!(obj.transform[3][0], 10.0);
    assert_eq!(obj.transform[3][1], 20.0);
    assert_eq!(obj.transform[3][2], 30.0);
}

/// Verify with_transform builder method
#[test]
fn blackbox_with_transform_builder() {
    let custom_transform = [
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
        [5.0, 10.0, 15.0, 1.0],
    ];

    let obj = ObjectData::new().with_transform(custom_transform);

    // Verify transform was set
    assert_eq!(obj.transform[0][0], 2.0);
    assert_eq!(obj.transform[3][0], 5.0);
    assert_eq!(obj.transform[3][1], 10.0);
    assert_eq!(obj.transform[3][2], 15.0);
}

/// Verify translation helper method
#[test]
fn blackbox_translation_helper() {
    let obj = ObjectData::new().with_transform([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [100.0, 200.0, 300.0, 1.0],
    ]);

    let translation = obj.translation();
    assert_eq!(translation[0], 100.0);
    assert_eq!(translation[1], 200.0);
    assert_eq!(translation[2], 300.0);
}

// ============================================================================
// SECTION 3: AABB Field Tests
// ============================================================================

/// Verify AABB fields are accessible
#[test]
fn blackbox_aabb_accessible() {
    let obj = ObjectData::new();

    // AABB fields must be accessible
    let _min = obj.aabb_min;
    let _max = obj.aabb_max;
}

/// Verify with_aabb builder method
#[test]
fn blackbox_with_aabb_builder() {
    let obj = ObjectData::new()
        .with_aabb([-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]);

    assert_eq!(obj.aabb_min, [-1.0, -2.0, -3.0]);
    assert_eq!(obj.aabb_max, [1.0, 2.0, 3.0]);
}

/// Verify AABB center calculation
#[test]
fn blackbox_aabb_center() {
    let obj = ObjectData::new()
        .with_aabb([0.0, 0.0, 0.0], [10.0, 20.0, 30.0]);

    let center = obj.aabb_center();
    assert_eq!(center[0], 5.0);
    assert_eq!(center[1], 10.0);
    assert_eq!(center[2], 15.0);
}

/// Verify AABB extents calculation
#[test]
fn blackbox_aabb_extents() {
    let obj = ObjectData::new()
        .with_aabb([0.0, 0.0, 0.0], [10.0, 20.0, 30.0]);

    let extents = obj.aabb_extents();
    assert_eq!(extents[0], 5.0);  // half-width
    assert_eq!(extents[1], 10.0); // half-height
    assert_eq!(extents[2], 15.0); // half-depth
}

/// Verify bounding sphere radius calculation
#[test]
fn blackbox_bounding_sphere_radius() {
    let obj = ObjectData::new()
        .with_aabb([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);

    let radius = obj.bounding_sphere_radius();
    // For a unit cube, extents are (1,1,1), so radius = sqrt(1+1+1) = sqrt(3) ~ 1.732
    assert!((radius - 1.732f32).abs() < 0.01);
}

// ============================================================================
// SECTION 4: Mesh and Material Index Tests
// ============================================================================

/// Verify mesh and material index fields are accessible
#[test]
fn blackbox_mesh_material_index() {
    let mut obj = ObjectData::new();

    // Set mesh and material indices
    obj.mesh_index = 5;
    obj.material_index = 10;

    assert_eq!(obj.mesh_index, 5, "mesh_index should be settable");
    assert_eq!(obj.material_index, 10, "material_index should be settable");
}

/// Verify invalid index constants
#[test]
fn blackbox_invalid_indices() {
    // Invalid indices should be 0xFFFFFFFF (u32::MAX)
    assert_eq!(
        INVALID_MESH_INDEX,
        u32::MAX,
        "INVALID_MESH_INDEX should be u32::MAX"
    );
    assert_eq!(
        INVALID_MATERIAL_INDEX,
        u32::MAX,
        "INVALID_MATERIAL_INDEX should be u32::MAX"
    );

    // Default should use invalid indices
    let obj = ObjectData::new();
    assert_eq!(
        obj.mesh_index,
        INVALID_MESH_INDEX,
        "Default mesh_index should be INVALID_MESH_INDEX"
    );
    assert_eq!(
        obj.material_index,
        INVALID_MATERIAL_INDEX,
        "Default material_index should be INVALID_MATERIAL_INDEX"
    );
}

/// Verify with_mesh builder method
#[test]
fn blackbox_with_mesh_builder() {
    let obj = ObjectData::new().with_mesh(42);
    assert_eq!(obj.mesh_index, 42);
}

/// Verify with_material builder method
#[test]
fn blackbox_with_material_builder() {
    let obj = ObjectData::new().with_material(99);
    assert_eq!(obj.material_index, 99);
}

/// Verify has_valid_resources helper
#[test]
fn blackbox_has_valid_resources() {
    let obj_invalid = ObjectData::new();
    assert!(!obj_invalid.has_valid_resources());

    let obj_valid = ObjectData::new()
        .with_mesh(0)
        .with_material(0);
    assert!(obj_valid.has_valid_resources());
}

// ============================================================================
// SECTION 5: LOD Tests
// ============================================================================

/// Verify LOD-related constants are exported
#[test]
fn blackbox_lod_constants() {
    // Max LOD levels must be reasonable (typically 4-8)
    assert!(
        OBJECT_MAX_LOD_LEVELS >= 2,
        "OBJECT_MAX_LOD_LEVELS must be at least 2"
    );
    assert!(
        OBJECT_MAX_LOD_LEVELS <= 16,
        "OBJECT_MAX_LOD_LEVELS should not exceed 16"
    );

    // Default LOD distances array must exist
    let distances = DEFAULT_LOD_DISTANCES;
    assert_eq!(
        distances.len(),
        OBJECT_MAX_LOD_LEVELS,
        "DEFAULT_LOD_DISTANCES length must match OBJECT_MAX_LOD_LEVELS"
    );

    // Distances should be monotonically increasing
    for i in 1..distances.len() {
        assert!(
            distances[i] >= distances[i - 1],
            "LOD distances must be monotonically increasing"
        );
    }
}

/// Verify lod_distances field is accessible
#[test]
fn blackbox_lod_distances_field() {
    let obj = ObjectData::new();

    // LOD distances should match default
    assert_eq!(obj.lod_distances, DEFAULT_LOD_DISTANCES);
}

/// Verify with_lod_distances builder method
#[test]
fn blackbox_with_lod_distances_builder() {
    let custom_distances = [50.0, 200.0, 500.0, 1000.0];
    let obj = ObjectData::new().with_lod_distances(custom_distances);
    assert_eq!(obj.lod_distances, custom_distances);
}

/// Verify select_lod method
#[test]
fn blackbox_select_lod() {
    // LOD distances represent squared distances for efficient comparison
    let obj = ObjectData::new().with_lod_distances([100.0, 400.0, 900.0, 1600.0]);

    // Test LOD selection based on distance squared
    // LOD 0: distance_sq < 100
    assert_eq!(obj.select_lod(50.0), 0);
    assert_eq!(obj.select_lod(99.0), 0);

    // LOD 1: 100 <= distance_sq < 400
    assert_eq!(obj.select_lod(100.0), 1);
    assert_eq!(obj.select_lod(150.0), 1);
    assert_eq!(obj.select_lod(399.0), 1);

    // LOD 2: 400 <= distance_sq < 900
    assert_eq!(obj.select_lod(400.0), 2);
    assert_eq!(obj.select_lod(500.0), 2);
    assert_eq!(obj.select_lod(899.0), 2);

    // LOD 3: 900 <= distance_sq < 1600
    assert_eq!(obj.select_lod(900.0), 3);
    assert_eq!(obj.select_lod(1000.0), 3);
    assert_eq!(obj.select_lod(1599.0), 3);

    // Beyond max: distance_sq >= 1600 returns MAX_LOD_LEVELS (beyond visible range)
    // This is valid behavior - indicates object is beyond all LOD ranges
    let beyond_max = obj.select_lod(2000.0);
    assert!(
        beyond_max >= 3,
        "Beyond max LOD should return >= 3 (got {})",
        beyond_max
    );
}

// ============================================================================
// SECTION 6: Flags Tests
// ============================================================================

/// Verify object_flags module constants
#[test]
fn blackbox_flags_module() {
    // Standard flags that should exist
    let visible = object_flags::VISIBLE;
    let casts_shadow = object_flags::CASTS_SHADOW;
    let receives_shadow = object_flags::RECEIVES_SHADOW;

    // Flags should be non-zero bit patterns
    assert_ne!(visible, 0, "VISIBLE flag must be non-zero");
    assert_ne!(casts_shadow, 0, "CASTS_SHADOW flag must be non-zero");
    assert_ne!(receives_shadow, 0, "RECEIVES_SHADOW flag must be non-zero");

    // Flags should be distinct (no overlapping bits for primary flags)
    assert_ne!(visible, casts_shadow, "VISIBLE and CASTS_SHADOW must be distinct");
    assert_ne!(visible, receives_shadow, "VISIBLE and RECEIVES_SHADOW must be distinct");
    assert_ne!(casts_shadow, receives_shadow, "CASTS_SHADOW and RECEIVES_SHADOW must be distinct");
}

/// Verify all object_flags constants
#[test]
fn blackbox_all_flags() {
    // Test all flag constants exist and are powers of 2
    let flags = [
        object_flags::VISIBLE,
        object_flags::CASTS_SHADOW,
        object_flags::STATIC,
        object_flags::RECEIVES_DECALS,
        object_flags::RECEIVES_SHADOW,
        object_flags::TWO_SIDED,
        object_flags::ALPHA_TEST,
        object_flags::ALPHA_BLEND,
        object_flags::SELECTED,
        object_flags::DIRTY,
        object_flags::SKINNED,
        object_flags::MOTION_BLUR,
    ];

    // Each flag should be a single bit
    for (i, &flag) in flags.iter().enumerate() {
        assert!(
            flag.count_ones() == 1,
            "Flag {} should be a single bit, got {:b}",
            i,
            flag
        );
    }

    // DEFAULT flag should combine multiple flags
    let default = object_flags::DEFAULT;
    assert_ne!(default.count_ones(), 1, "DEFAULT should combine multiple flags");
    assert_ne!(default & object_flags::VISIBLE, 0, "DEFAULT should include VISIBLE");
}

/// Verify flags field is accessible and combinable
#[test]
fn blackbox_flags_field() {
    let mut obj = ObjectData::new();

    // Should be able to set flags
    obj.flags = object_flags::VISIBLE;
    assert_eq!(obj.flags, object_flags::VISIBLE);

    // Should be able to combine flags
    obj.flags = object_flags::VISIBLE | object_flags::CASTS_SHADOW;
    assert_eq!(obj.flags & object_flags::VISIBLE, object_flags::VISIBLE);
    assert_eq!(obj.flags & object_flags::CASTS_SHADOW, object_flags::CASTS_SHADOW);

    // Should be able to check flag presence
    assert_ne!(obj.flags & object_flags::VISIBLE, 0);
    assert_eq!(obj.flags & object_flags::STATIC, 0);
}

/// Verify with_flags builder methods
#[test]
fn blackbox_with_flags_builder() {
    let obj = ObjectData::new()
        .with_flags(object_flags::VISIBLE | object_flags::STATIC);

    assert!(obj.is_visible());
    assert!(obj.is_static());
    assert!(!obj.casts_shadow()); // Not set

    // Test with_flags_added
    let obj2 = obj.with_flags_added(object_flags::CASTS_SHADOW);
    assert!(obj2.is_visible());
    assert!(obj2.is_static());
    assert!(obj2.casts_shadow());

    // Test with_flags_removed
    let obj3 = obj2.with_flags_removed(object_flags::STATIC);
    assert!(obj3.is_visible());
    assert!(!obj3.is_static());
    assert!(obj3.casts_shadow());
}

/// Verify flag query methods
#[test]
fn blackbox_flag_queries() {
    let obj = ObjectData::new()
        .with_flags(object_flags::VISIBLE | object_flags::CASTS_SHADOW | object_flags::RECEIVES_SHADOW);

    assert!(obj.is_visible());
    assert!(obj.casts_shadow());
    assert!(obj.receives_shadow());
    assert!(!obj.is_static());
    assert!(!obj.receives_decals());
    assert!(!obj.is_two_sided());
    assert!(!obj.has_alpha_test());
    assert!(!obj.has_alpha_blend());
    assert!(!obj.is_selected());
    assert!(!obj.is_dirty());
    assert!(!obj.is_skinned());
    assert!(!obj.has_motion_blur());
}

/// Verify flag setter methods
#[test]
fn blackbox_flag_setters() {
    let mut obj = ObjectData::new().with_flags(0);

    obj.set_visible(true);
    assert!(obj.is_visible());

    obj.set_casts_shadow(true);
    assert!(obj.casts_shadow());

    obj.set_static(true);
    assert!(obj.is_static());

    obj.set_dirty(true);
    assert!(obj.is_dirty());

    // Turn off
    obj.set_visible(false);
    assert!(!obj.is_visible());
}

// ============================================================================
// SECTION 7: Bytemuck Compatibility Tests
// ============================================================================

/// Verify ObjectData implements Pod and Zeroable via bytemuck
#[test]
fn blackbox_bytemuck() {
    let obj = ObjectData::new();

    // Must be convertible to bytes
    let bytes: &[u8] = bytemuck::bytes_of(&obj);
    assert_eq!(
        bytes.len(),
        OBJECT_DATA_SIZE,
        "bytemuck::bytes_of size must match OBJECT_DATA_SIZE"
    );

    // Must be reconstructible from bytes
    let bytes_copy = bytes.to_vec();
    let reconstructed: &ObjectData = bytemuck::from_bytes(&bytes_copy);

    // Verify round-trip preserves data
    assert_eq!(reconstructed.transform[0][0], obj.transform[0][0]);
    assert_eq!(reconstructed.mesh_index, obj.mesh_index);
    assert_eq!(reconstructed.material_index, obj.material_index);
}

/// Verify ObjectData can be zeroed via bytemuck
#[test]
fn blackbox_bytemuck_zeroed() {
    let zeroed: ObjectData = bytemuck::Zeroable::zeroed();

    // All fields should be zero
    assert_eq!(zeroed.transform[0][0], 0.0);
    assert_eq!(zeroed.mesh_index, 0);
    assert_eq!(zeroed.material_index, 0);
    assert_eq!(zeroed.flags, 0);
}

/// Verify ObjectData::zeroed() constructor
#[test]
fn blackbox_zeroed_constructor() {
    let zeroed = ObjectData::zeroed();

    assert_eq!(zeroed.transform[0][0], 0.0);
    assert_eq!(zeroed.transform[3][3], 0.0);
    assert_eq!(zeroed.mesh_index, 0);
    assert_eq!(zeroed.material_index, 0);
    assert_eq!(zeroed.flags, 0);
}

/// Verify ObjectData can be cast to/from slice
#[test]
fn blackbox_bytemuck_slice() {
    let objects = vec![ObjectData::new(); 10];

    // Cast slice to bytes
    let bytes: &[u8] = bytemuck::cast_slice(&objects);
    assert_eq!(bytes.len(), OBJECT_DATA_SIZE * 10);

    // Cast bytes back to slice
    let reconstructed: &[ObjectData] = bytemuck::cast_slice(bytes);
    assert_eq!(reconstructed.len(), 10);
}

// ============================================================================
// SECTION 8: ObjectDataBuffer Tests
// ============================================================================

/// Verify ObjectDataBuffer exists and is usable
#[test]
fn blackbox_buffer_exists() {
    // ObjectDataBuffer should be a collection type
    let buffer = ObjectDataBuffer::new(10);
    let _len = buffer.len();
}

/// Verify ObjectDataBuffer can store and retrieve objects
#[test]
fn blackbox_buffer_storage() {
    let mut buffer = ObjectDataBuffer::new(10);

    // Should be empty initially
    assert!(buffer.is_empty());

    // Add an object
    let mut obj = ObjectData::new();
    obj.mesh_index = 42;
    let index = buffer.add(obj);

    // Should have one element
    assert_eq!(buffer.len(), 1);
    assert_eq!(index, 0);

    // Should be retrievable
    assert_eq!(buffer.get(0).unwrap().mesh_index, 42);
}

/// Verify ObjectDataBuffer can be converted to bytes for GPU upload
#[test]
fn blackbox_buffer_as_bytes() {
    let mut buffer = ObjectDataBuffer::new(5);

    for i in 0..5 {
        let obj = ObjectData::new().with_mesh(i);
        buffer.add(obj);
    }

    // Should be able to get raw byte slice for GPU upload
    let bytes = buffer.as_bytes();
    assert_eq!(bytes.len(), OBJECT_DATA_SIZE * 5);
}

/// Verify ObjectDataBuffer capacity management
#[test]
fn blackbox_buffer_capacity() {
    let buffer = ObjectDataBuffer::new(100);

    // Should have reserved capacity
    assert!(buffer.capacity() >= 100);
}

/// Verify ObjectDataBuffer with_objects constructor
#[test]
fn blackbox_buffer_with_objects() {
    let objects: Vec<ObjectData> = (0..5)
        .map(|i| ObjectData::new().with_mesh(i))
        .collect();

    let buffer = ObjectDataBuffer::with_objects(objects);
    assert_eq!(buffer.len(), 5);
}

/// Verify ObjectDataBuffer dirty tracking
#[test]
fn blackbox_buffer_dirty_tracking() {
    let mut buffer = ObjectDataBuffer::new(10);

    // Adding should mark dirty
    buffer.add(ObjectData::new());
    assert!(buffer.is_dirty());

    // Clear dirty flag
    buffer.clear_dirty();
    assert!(!buffer.is_dirty());

    // Modifying should mark dirty again
    buffer.update_transform(0, [[1.0; 4]; 4]);
    assert!(buffer.is_dirty());
}

/// Verify ObjectDataBuffer get/get_mut
#[test]
fn blackbox_buffer_get() {
    let mut buffer = ObjectDataBuffer::new(10);
    buffer.add(ObjectData::new().with_mesh(100));

    // get should return reference
    assert_eq!(buffer.get(0).unwrap().mesh_index, 100);
    assert!(buffer.get(1).is_none());

    // get_mut should allow modification
    if let Some(obj) = buffer.get_mut(0) {
        obj.mesh_index = 200;
    }
    assert_eq!(buffer.get(0).unwrap().mesh_index, 200);
}

/// Verify ObjectDataBuffer as_slice
#[test]
fn blackbox_buffer_as_slice() {
    let mut buffer = ObjectDataBuffer::new(5);

    for i in 0..5 {
        buffer.add(ObjectData::new().with_mesh(i));
    }

    let slice = buffer.as_slice();
    assert_eq!(slice.len(), 5);
    assert_eq!(slice[0].mesh_index, 0);
    assert_eq!(slice[4].mesh_index, 4);
}

/// Verify ObjectDataBuffer byte_size
#[test]
fn blackbox_buffer_byte_size() {
    let mut buffer = ObjectDataBuffer::new(10);

    for _ in 0..3 {
        buffer.add(ObjectData::new());
    }

    assert_eq!(buffer.byte_size(), OBJECT_DATA_SIZE * 3);
}

/// Verify ObjectDataBuffer update helpers
#[test]
fn blackbox_buffer_update_helpers() {
    let mut buffer = ObjectDataBuffer::new(10);
    buffer.add(ObjectData::new());

    // update_transform
    let new_transform = [
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    buffer.update_transform(0, new_transform);
    assert_eq!(buffer.get(0).unwrap().transform[0][0], 2.0);

    // update_aabb
    buffer.update_aabb(0, [-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]);
    assert_eq!(buffer.get(0).unwrap().aabb_min, [-5.0, -5.0, -5.0]);

    // set_visible
    buffer.set_visible(0, false);
    assert!(!buffer.get(0).unwrap().is_visible());
    buffer.set_visible(0, true);
    assert!(buffer.get(0).unwrap().is_visible());
}

// ============================================================================
// SECTION 9: Builder Pattern Chaining Tests
// ============================================================================

/// Test complete builder pattern chaining
#[test]
fn blackbox_builder_pattern() {
    let obj = ObjectData::new()
        .with_transform([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [10.0, 20.0, 30.0, 1.0],
        ])
        .with_aabb([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
        .with_mesh(5)
        .with_material(10)
        .with_lod_distances([50.0, 100.0, 200.0, 400.0])
        .with_flags(object_flags::VISIBLE | object_flags::CASTS_SHADOW);

    // Verify all fields
    assert_eq!(obj.transform[3][0], 10.0);
    assert_eq!(obj.aabb_min, [-1.0, -1.0, -1.0]);
    assert_eq!(obj.mesh_index, 5);
    assert_eq!(obj.material_index, 10);
    assert_eq!(obj.lod_distances[0], 50.0);
    assert!(obj.is_visible());
    assert!(obj.casts_shadow());
}

// ============================================================================
// SECTION 10: Clone and Copy Tests
// ============================================================================

/// Verify ObjectData implements Clone
#[test]
fn blackbox_clone() {
    let obj = ObjectData::new()
        .with_mesh(99)
        .with_transform([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [123.0, 0.0, 0.0, 1.0],
        ]);

    let cloned = obj.clone();

    assert_eq!(cloned.mesh_index, 99);
    assert_eq!(cloned.transform[3][0], 123.0);
}

/// Verify ObjectData implements Copy (required for Pod)
#[test]
fn blackbox_copy() {
    let obj = ObjectData::new();
    let _copied = obj; // Move
    let _still_valid = obj; // Copy - obj is still valid because Copy is implemented
}

// ============================================================================
// SECTION 11: Debug Trait Tests
// ============================================================================

/// Verify ObjectData implements Debug for logging
#[test]
fn blackbox_debug() {
    let obj = ObjectData::new();
    let debug_str = format!("{:?}", obj);

    // Debug output should be non-empty
    assert!(!debug_str.is_empty());

    // Should contain struct name
    assert!(
        debug_str.contains("ObjectData"),
        "Debug output should contain 'ObjectData'"
    );
}

// ============================================================================
// SECTION 12: Edge Cases and Stress Tests
// ============================================================================

/// Test extreme values don't cause panics
#[test]
fn blackbox_extreme_values() {
    let obj = ObjectData::new()
        .with_mesh(u32::MAX)
        .with_material(u32::MAX)
        .with_transform([
            [f32::MAX, 0.0, 0.0, 0.0],
            [0.0, f32::MIN, 0.0, 0.0],
            [0.0, 0.0, f32::INFINITY, 0.0],
            [0.0, 0.0, 0.0, f32::NEG_INFINITY],
        ]);

    // Should still be convertible to bytes without panic
    let bytes = bytemuck::bytes_of(&obj);
    assert_eq!(bytes.len(), OBJECT_DATA_SIZE);
}

/// Test large buffer allocation
#[test]
fn blackbox_large_buffer() {
    const LARGE_COUNT: usize = 10_000;

    let mut buffer = ObjectDataBuffer::new(LARGE_COUNT);

    for i in 0..LARGE_COUNT {
        buffer.add(ObjectData::new().with_mesh(i as u32));
    }

    assert_eq!(buffer.len(), LARGE_COUNT);

    // Verify random access
    assert_eq!(buffer.get(5000).unwrap().mesh_index, 5000);
    assert_eq!(buffer.get(9999).unwrap().mesh_index, 9999);
}

/// Test buffer clear and reuse
#[test]
fn blackbox_buffer_clear() {
    let mut buffer = ObjectDataBuffer::new(100);

    // Fill buffer
    for _ in 0..100 {
        buffer.add(ObjectData::new());
    }
    assert_eq!(buffer.len(), 100);

    // Clear buffer
    buffer.clear();
    assert!(buffer.is_empty());
    assert_eq!(buffer.len(), 0);

    // Should be reusable
    buffer.add(ObjectData::new());
    assert_eq!(buffer.len(), 1);
}

// ============================================================================
// SECTION 13: Default Flags Tests
// ============================================================================

/// Verify default ObjectData has DEFAULT flags
#[test]
fn blackbox_default_flags() {
    let obj = ObjectData::new();

    // Default should include standard rendering flags
    assert!(obj.is_visible(), "Default should be visible");
    assert!(obj.casts_shadow(), "Default should cast shadow");
    assert!(obj.receives_shadow(), "Default should receive shadow");
    assert!(obj.receives_decals(), "Default should receive decals");
}
