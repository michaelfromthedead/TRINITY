// SPDX-License-Identifier: MIT
//
// blackbox_vertex_format_registry.rs -- Blackbox contract tests for T-WGPU-P3.2.1 (Vertex Format Registry).
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::render_pipeline::*` -- no internal fields,
// no private methods, no implementation details.
//
// Public API under test:
//   VertexFormatId          -- StaticMesh, SkinnedMesh, Terrain, Particle, Ui, Custom(u32)
//   VertexFormat            -- id, name, stride, attributes, to_buffer_layout()
//   VertexFormatRegistry    -- new(), register(), get(), get_buffer_layout(), contains(), iter(), len(), is_empty()
//   static_mesh()           -- Creates StaticMesh format (48 bytes)
//   skinned_mesh()          -- Creates SkinnedMesh format (72 bytes)
//   terrain()               -- Creates Terrain format (32 bytes)
//   particle()              -- Creates Particle format (36 bytes)
//   ui()                    -- Creates UI format (20 bytes)
//
// Test Categories:
//   1. API Surface Tests (verify imports and types compile)
//   2. Standard Format Constructor Tests
//   3. Registry Creation and Pre-registration Tests
//   4. Registry Lookup Pattern Tests
//   5. Custom Format Registration Tests
//   6. Buffer Layout Conversion Tests
//   7. Registry Iteration Tests
//   8. Edge Case and Error Handling Tests
//   9. Thread Safety Tests
//  10. Integration with VertexStateDescriptor Tests

use std::collections::HashSet;

use renderer_backend::render_pipeline::{
    particle, skinned_mesh, static_mesh, terrain, ui, VertexAttributeDescriptor,
    VertexBufferLayoutDescriptor, VertexFormat, VertexFormatId, VertexFormatRegistry,
};

// ===========================================================================
// 1. API Surface Tests
// ===========================================================================

/// Verify all required types are exported from the render_pipeline module.
#[test]
fn test_api_surface_imports_compile() {
    // This test passes if it compiles - verifies the API surface exists
    let _: fn() -> VertexFormat = static_mesh;
    let _: fn() -> VertexFormat = skinned_mesh;
    let _: fn() -> VertexFormat = terrain;
    let _: fn() -> VertexFormat = particle;
    let _: fn() -> VertexFormat = ui;

    // Verify registry type exists
    let registry = VertexFormatRegistry::new();
    let _ = registry.len();
}

/// Verify VertexFormatId enum has all expected variants.
#[test]
fn test_api_surface_vertex_format_id_variants() {
    let _static_mesh = VertexFormatId::StaticMesh;
    let _skinned_mesh = VertexFormatId::SkinnedMesh;
    let _terrain = VertexFormatId::Terrain;
    let _particle = VertexFormatId::Particle;
    let _ui = VertexFormatId::Ui;
    let _custom = VertexFormatId::Custom(42);
}

/// Verify VertexFormat struct fields are accessible.
#[test]
fn test_api_surface_vertex_format_fields() {
    let format = static_mesh();

    // All public fields should be accessible
    let _id: VertexFormatId = format.id;
    let _name: &'static str = format.name;
    let _stride: u64 = format.stride;
    let _attrs: &Vec<VertexAttributeDescriptor> = &format.attributes;
}

// ===========================================================================
// 2. Standard Format Constructor Tests
// ===========================================================================

#[test]
fn test_static_mesh_format_properties() {
    let format = static_mesh();

    assert_eq!(format.id, VertexFormatId::StaticMesh);
    assert_eq!(format.name, "StaticMesh");
    assert_eq!(format.stride, 48);
    assert_eq!(format.attributes.len(), 4);
}

#[test]
fn test_static_mesh_format_attributes() {
    let format = static_mesh();

    // Verify attribute layout:
    // position: Float32x3 @ offset 0, location 0
    // normal: Float32x3 @ offset 12, location 1
    // tangent: Float32x4 @ offset 24, location 2
    // uv: Float32x2 @ offset 40, location 3
    assert_eq!(format.attributes[0].offset, 0);
    assert_eq!(format.attributes[0].shader_location, 0);
    assert_eq!(format.attributes[0].format, wgpu::VertexFormat::Float32x3);

    assert_eq!(format.attributes[1].offset, 12);
    assert_eq!(format.attributes[1].shader_location, 1);
    assert_eq!(format.attributes[1].format, wgpu::VertexFormat::Float32x3);

    assert_eq!(format.attributes[2].offset, 24);
    assert_eq!(format.attributes[2].shader_location, 2);
    assert_eq!(format.attributes[2].format, wgpu::VertexFormat::Float32x4);

    assert_eq!(format.attributes[3].offset, 40);
    assert_eq!(format.attributes[3].shader_location, 3);
    assert_eq!(format.attributes[3].format, wgpu::VertexFormat::Float32x2);
}

#[test]
fn test_skinned_mesh_format_properties() {
    let format = skinned_mesh();

    assert_eq!(format.id, VertexFormatId::SkinnedMesh);
    assert_eq!(format.name, "SkinnedMesh");
    assert_eq!(format.stride, 72);
    assert_eq!(format.attributes.len(), 6);
}

#[test]
fn test_skinned_mesh_format_attributes() {
    let format = skinned_mesh();

    // Verify bone_indices and bone_weights are included
    // bone_indices: Uint16x4 @ offset 48, location 4
    // bone_weights: Float32x4 @ offset 56, location 5
    assert_eq!(format.attributes[4].offset, 48);
    assert_eq!(format.attributes[4].shader_location, 4);
    assert_eq!(format.attributes[4].format, wgpu::VertexFormat::Uint16x4);

    assert_eq!(format.attributes[5].offset, 56);
    assert_eq!(format.attributes[5].shader_location, 5);
    assert_eq!(format.attributes[5].format, wgpu::VertexFormat::Float32x4);
}

#[test]
fn test_terrain_format_properties() {
    let format = terrain();

    assert_eq!(format.id, VertexFormatId::Terrain);
    assert_eq!(format.name, "Terrain");
    assert_eq!(format.stride, 32);
    assert_eq!(format.attributes.len(), 3);
}

#[test]
fn test_terrain_format_attributes() {
    let format = terrain();

    // position: Float32x3 @ offset 0, location 0
    // normal: Float32x3 @ offset 12, location 1
    // uv: Float32x2 @ offset 24, location 2
    assert_eq!(format.attributes[0].offset, 0);
    assert_eq!(format.attributes[1].offset, 12);
    assert_eq!(format.attributes[2].offset, 24);
}

#[test]
fn test_particle_format_properties() {
    let format = particle();

    assert_eq!(format.id, VertexFormatId::Particle);
    assert_eq!(format.name, "Particle");
    assert_eq!(format.stride, 36);
    assert_eq!(format.attributes.len(), 3);
}

#[test]
fn test_particle_format_attributes() {
    let format = particle();

    // position: Float32x3 @ offset 0, location 0
    // color: Float32x4 @ offset 12, location 1
    // size_rotation: Float32x2 @ offset 28, location 2
    assert_eq!(format.attributes[0].offset, 0);
    assert_eq!(format.attributes[0].format, wgpu::VertexFormat::Float32x3);

    assert_eq!(format.attributes[1].offset, 12);
    assert_eq!(format.attributes[1].format, wgpu::VertexFormat::Float32x4);

    assert_eq!(format.attributes[2].offset, 28);
    assert_eq!(format.attributes[2].format, wgpu::VertexFormat::Float32x2);
}

#[test]
fn test_ui_format_properties() {
    let format = ui();

    assert_eq!(format.id, VertexFormatId::Ui);
    assert_eq!(format.name, "UI");
    assert_eq!(format.stride, 20);
    assert_eq!(format.attributes.len(), 3);
}

#[test]
fn test_ui_format_attributes() {
    let format = ui();

    // position: Float32x2 @ offset 0, location 0
    // uv: Float32x2 @ offset 8, location 1
    // color: Unorm8x4 @ offset 16, location 2
    assert_eq!(format.attributes[0].offset, 0);
    assert_eq!(format.attributes[0].format, wgpu::VertexFormat::Float32x2);

    assert_eq!(format.attributes[1].offset, 8);
    assert_eq!(format.attributes[1].format, wgpu::VertexFormat::Float32x2);

    assert_eq!(format.attributes[2].offset, 16);
    assert_eq!(format.attributes[2].format, wgpu::VertexFormat::Unorm8x4);
}

// ===========================================================================
// 3. Registry Creation and Pre-registration Tests
// ===========================================================================

#[test]
fn test_registry_new_creates_nonempty_registry() {
    let registry = VertexFormatRegistry::new();

    assert!(!registry.is_empty());
    assert!(registry.len() >= 5, "Registry should have at least 5 standard formats");
}

#[test]
fn test_registry_default_same_as_new() {
    let registry1 = VertexFormatRegistry::new();
    let registry2 = VertexFormatRegistry::default();

    assert_eq!(registry1.len(), registry2.len());
}

#[test]
fn test_registry_preregisters_all_standard_formats() {
    let registry = VertexFormatRegistry::new();

    assert!(registry.contains(VertexFormatId::StaticMesh));
    assert!(registry.contains(VertexFormatId::SkinnedMesh));
    assert!(registry.contains(VertexFormatId::Terrain));
    assert!(registry.contains(VertexFormatId::Particle));
    assert!(registry.contains(VertexFormatId::Ui));
}

#[test]
fn test_registry_preregistered_count_is_five() {
    let registry = VertexFormatRegistry::new();

    // Should have exactly 5 standard formats
    assert_eq!(registry.len(), 5);
}

// ===========================================================================
// 4. Registry Lookup Pattern Tests
// ===========================================================================

#[test]
fn test_registry_get_returns_some_for_registered() {
    let registry = VertexFormatRegistry::new();

    assert!(registry.get(VertexFormatId::StaticMesh).is_some());
    assert!(registry.get(VertexFormatId::SkinnedMesh).is_some());
    assert!(registry.get(VertexFormatId::Terrain).is_some());
    assert!(registry.get(VertexFormatId::Particle).is_some());
    assert!(registry.get(VertexFormatId::Ui).is_some());
}

#[test]
fn test_registry_get_returns_none_for_unregistered_custom() {
    let registry = VertexFormatRegistry::new();

    assert!(registry.get(VertexFormatId::Custom(0)).is_none());
    assert!(registry.get(VertexFormatId::Custom(100)).is_none());
    assert!(registry.get(VertexFormatId::Custom(u32::MAX)).is_none());
}

#[test]
fn test_registry_get_returns_correct_format_data() {
    let registry = VertexFormatRegistry::new();

    let format = registry.get(VertexFormatId::StaticMesh).unwrap();
    assert_eq!(format.id, VertexFormatId::StaticMesh);
    assert_eq!(format.stride, 48);
    assert_eq!(format.name, "StaticMesh");
}

#[test]
fn test_registry_contains_matches_get() {
    let registry = VertexFormatRegistry::new();

    // contains() should return true if and only if get() returns Some
    let ids = [
        VertexFormatId::StaticMesh,
        VertexFormatId::SkinnedMesh,
        VertexFormatId::Terrain,
        VertexFormatId::Particle,
        VertexFormatId::Ui,
        VertexFormatId::Custom(0),
        VertexFormatId::Custom(999),
    ];

    for id in ids {
        assert_eq!(
            registry.contains(id),
            registry.get(id).is_some(),
            "contains() and get() should be consistent for {:?}",
            id
        );
    }
}

// ===========================================================================
// 5. Custom Format Registration Tests
// ===========================================================================

#[test]
fn test_registry_register_custom_format() {
    let mut registry = VertexFormatRegistry::new();

    let custom = VertexFormat {
        id: VertexFormatId::Custom(42),
        name: "CustomVertex",
        stride: 16,
        attributes: vec![VertexAttributeDescriptor::new(
            wgpu::VertexFormat::Float32x4,
            0,
            0,
        )],
    };

    registry.register(custom);

    assert!(registry.contains(VertexFormatId::Custom(42)));
    assert_eq!(registry.len(), 6);
}

#[test]
fn test_registry_register_multiple_custom_formats() {
    let mut registry = VertexFormatRegistry::new();

    for i in 0..10 {
        let custom = VertexFormat {
            id: VertexFormatId::Custom(i),
            name: "Custom",
            stride: (i as u64 + 1) * 4,
            attributes: vec![],
        };
        registry.register(custom);
    }

    assert_eq!(registry.len(), 15); // 5 standard + 10 custom
}

#[test]
fn test_registry_register_overwrites_existing() {
    let mut registry = VertexFormatRegistry::new();

    // Register a custom format
    let original = VertexFormat {
        id: VertexFormatId::Custom(1),
        name: "Original",
        stride: 32,
        attributes: vec![],
    };
    registry.register(original);

    // Overwrite with a different format using the same ID
    let replacement = VertexFormat {
        id: VertexFormatId::Custom(1),
        name: "Replacement",
        stride: 64,
        attributes: vec![],
    };
    registry.register(replacement);

    // Should still be 6 total (5 standard + 1 custom)
    assert_eq!(registry.len(), 6);

    // Should have replacement data
    let format = registry.get(VertexFormatId::Custom(1)).unwrap();
    assert_eq!(format.name, "Replacement");
    assert_eq!(format.stride, 64);
}

#[test]
fn test_registry_register_can_overwrite_standard_formats() {
    let mut registry = VertexFormatRegistry::new();

    let modified_static = VertexFormat {
        id: VertexFormatId::StaticMesh,
        name: "ModifiedStaticMesh",
        stride: 64,
        attributes: vec![],
    };
    registry.register(modified_static);

    let format = registry.get(VertexFormatId::StaticMesh).unwrap();
    assert_eq!(format.name, "ModifiedStaticMesh");
    assert_eq!(format.stride, 64);
}

#[test]
fn test_registry_custom_format_with_complex_attributes() {
    let mut registry = VertexFormatRegistry::new();

    let complex = VertexFormat {
        id: VertexFormatId::Custom(100),
        name: "ComplexVertex",
        stride: 96,
        attributes: vec![
            VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x3, 0, 0),
            VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x3, 12, 1),
            VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x4, 24, 2),
            VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x2, 40, 3),
            VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x4, 48, 4),
            VertexAttributeDescriptor::new(wgpu::VertexFormat::Uint32x4, 64, 5),
            VertexAttributeDescriptor::new(wgpu::VertexFormat::Snorm16x4, 80, 6),
            VertexAttributeDescriptor::new(wgpu::VertexFormat::Unorm8x4, 88, 7),
        ],
    };

    registry.register(complex);

    let format = registry.get(VertexFormatId::Custom(100)).unwrap();
    assert_eq!(format.attributes.len(), 8);
    assert_eq!(format.stride, 96);
}

// ===========================================================================
// 6. Buffer Layout Conversion Tests
// ===========================================================================

#[test]
fn test_format_to_buffer_layout_stride() {
    let format = static_mesh();
    let layout = format.to_buffer_layout();

    assert_eq!(layout.array_stride, format.stride);
}

#[test]
fn test_format_to_buffer_layout_step_mode() {
    let format = static_mesh();
    let layout = format.to_buffer_layout();

    assert_eq!(layout.step_mode, wgpu::VertexStepMode::Vertex);
}

#[test]
fn test_format_to_buffer_layout_attributes() {
    let format = terrain();
    let layout = format.to_buffer_layout();

    assert_eq!(layout.attributes.len(), format.attributes.len());

    for (i, attr) in layout.attributes.iter().enumerate() {
        assert_eq!(attr.offset, format.attributes[i].offset);
        assert_eq!(attr.shader_location, format.attributes[i].shader_location);
        assert_eq!(attr.format, format.attributes[i].format);
    }
}

#[test]
fn test_registry_get_buffer_layout() {
    let registry = VertexFormatRegistry::new();

    let layout = registry.get_buffer_layout(VertexFormatId::Terrain).unwrap();

    assert_eq!(layout.array_stride, 32);
    assert_eq!(layout.attributes.len(), 3);
    assert_eq!(layout.step_mode, wgpu::VertexStepMode::Vertex);
}

#[test]
fn test_registry_get_buffer_layout_returns_none_for_unregistered() {
    let registry = VertexFormatRegistry::new();

    assert!(registry.get_buffer_layout(VertexFormatId::Custom(999)).is_none());
}

#[test]
fn test_buffer_layout_attribute_offsets_ascending() {
    let registry = VertexFormatRegistry::new();

    // For all standard formats, attribute offsets should be ascending
    let ids = [
        VertexFormatId::StaticMesh,
        VertexFormatId::SkinnedMesh,
        VertexFormatId::Terrain,
        VertexFormatId::Particle,
        VertexFormatId::Ui,
    ];

    for id in ids {
        let layout = registry.get_buffer_layout(id).unwrap();
        for i in 1..layout.attributes.len() {
            assert!(
                layout.attributes[i].offset >= layout.attributes[i - 1].offset,
                "Offsets should be ascending for {:?}",
                id
            );
        }
    }
}

#[test]
fn test_buffer_layout_shader_locations_unique() {
    let registry = VertexFormatRegistry::new();

    let ids = [
        VertexFormatId::StaticMesh,
        VertexFormatId::SkinnedMesh,
        VertexFormatId::Terrain,
        VertexFormatId::Particle,
        VertexFormatId::Ui,
    ];

    for id in ids {
        let layout = registry.get_buffer_layout(id).unwrap();
        let locations: HashSet<u32> = layout.attributes.iter().map(|a| a.shader_location).collect();

        assert_eq!(
            locations.len(),
            layout.attributes.len(),
            "Shader locations should be unique for {:?}",
            id
        );
    }
}

// ===========================================================================
// 7. Registry Iteration Tests
// ===========================================================================

#[test]
fn test_registry_iter_returns_all_formats() {
    let registry = VertexFormatRegistry::new();

    let formats: Vec<_> = registry.iter().collect();

    assert_eq!(formats.len(), 5);
}

#[test]
fn test_registry_iter_includes_custom_formats() {
    let mut registry = VertexFormatRegistry::new();

    registry.register(VertexFormat {
        id: VertexFormatId::Custom(1),
        name: "Custom1",
        stride: 16,
        attributes: vec![],
    });

    let formats: Vec<_> = registry.iter().collect();

    assert_eq!(formats.len(), 6);
}

#[test]
fn test_registry_iter_contains_all_standard_ids() {
    let registry = VertexFormatRegistry::new();

    let ids: HashSet<VertexFormatId> = registry.iter().map(|f| f.id).collect();

    assert!(ids.contains(&VertexFormatId::StaticMesh));
    assert!(ids.contains(&VertexFormatId::SkinnedMesh));
    assert!(ids.contains(&VertexFormatId::Terrain));
    assert!(ids.contains(&VertexFormatId::Particle));
    assert!(ids.contains(&VertexFormatId::Ui));
}

#[test]
fn test_registry_iter_format_data_consistent() {
    let registry = VertexFormatRegistry::new();

    for format in registry.iter() {
        // Verify each format from iter matches what get() returns
        let by_get = registry.get(format.id).unwrap();
        assert_eq!(format.id, by_get.id);
        assert_eq!(format.name, by_get.name);
        assert_eq!(format.stride, by_get.stride);
        assert_eq!(format.attributes.len(), by_get.attributes.len());
    }
}

// ===========================================================================
// 8. Edge Case and Error Handling Tests
// ===========================================================================

#[test]
fn test_vertex_format_id_equality() {
    assert_eq!(VertexFormatId::StaticMesh, VertexFormatId::StaticMesh);
    assert_ne!(VertexFormatId::StaticMesh, VertexFormatId::SkinnedMesh);
    assert_ne!(VertexFormatId::StaticMesh, VertexFormatId::Custom(0));
}

#[test]
fn test_vertex_format_id_custom_equality() {
    assert_eq!(VertexFormatId::Custom(0), VertexFormatId::Custom(0));
    assert_ne!(VertexFormatId::Custom(0), VertexFormatId::Custom(1));
    assert_eq!(VertexFormatId::Custom(u32::MAX), VertexFormatId::Custom(u32::MAX));
}

#[test]
fn test_vertex_format_id_hash() {
    let mut set = HashSet::new();
    set.insert(VertexFormatId::StaticMesh);
    set.insert(VertexFormatId::Custom(42));

    assert!(set.contains(&VertexFormatId::StaticMesh));
    assert!(set.contains(&VertexFormatId::Custom(42)));
    assert!(!set.contains(&VertexFormatId::Terrain));
    assert!(!set.contains(&VertexFormatId::Custom(0)));
}

#[test]
fn test_vertex_format_id_debug() {
    let id = VertexFormatId::StaticMesh;
    let debug_str = format!("{:?}", id);
    assert!(!debug_str.is_empty());
}

#[test]
fn test_vertex_format_id_clone_copy() {
    let original = VertexFormatId::Custom(42);
    let cloned = original.clone();
    let copied = original; // Would fail without Copy

    assert_eq!(original, cloned);
    assert_eq!(original, copied);
}

#[test]
fn test_vertex_format_clone() {
    let format = static_mesh();
    let cloned = format.clone();

    assert_eq!(format.id, cloned.id);
    assert_eq!(format.name, cloned.name);
    assert_eq!(format.stride, cloned.stride);
    assert_eq!(format.attributes.len(), cloned.attributes.len());
}

#[test]
fn test_empty_attributes_format() {
    let mut registry = VertexFormatRegistry::new();

    let empty = VertexFormat {
        id: VertexFormatId::Custom(0),
        name: "EmptyFormat",
        stride: 0,
        attributes: vec![],
    };

    registry.register(empty);

    let format = registry.get(VertexFormatId::Custom(0)).unwrap();
    assert!(format.attributes.is_empty());
    assert_eq!(format.stride, 0);

    let layout = format.to_buffer_layout();
    assert!(layout.attributes.is_empty());
    assert_eq!(layout.array_stride, 0);
}

#[test]
fn test_format_stride_consistency() {
    // Verify documented stride values match
    assert_eq!(static_mesh().stride, 48);
    assert_eq!(skinned_mesh().stride, 72);
    assert_eq!(terrain().stride, 32);
    assert_eq!(particle().stride, 36);
    assert_eq!(ui().stride, 20);
}

#[test]
fn test_format_constructors_idempotent() {
    // Calling constructor multiple times should produce identical results
    let s1 = static_mesh();
    let s2 = static_mesh();

    assert_eq!(s1.id, s2.id);
    assert_eq!(s1.name, s2.name);
    assert_eq!(s1.stride, s2.stride);
    assert_eq!(s1.attributes.len(), s2.attributes.len());
}

// ===========================================================================
// 9. Thread Safety Tests
// ===========================================================================

#[test]
fn test_vertex_format_registry_send_sync() {
    // Compile-time check for Send + Sync bounds
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<VertexFormatRegistry>();
}

#[test]
fn test_vertex_format_id_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<VertexFormatId>();
}

#[test]
fn test_concurrent_registry_read_access() {
    use std::sync::Arc;
    use std::thread;

    let registry = Arc::new(VertexFormatRegistry::new());
    let mut handles = vec![];

    for _ in 0..4 {
        let registry_clone = Arc::clone(&registry);
        handles.push(thread::spawn(move || {
            // Multiple threads can safely read from the registry
            for _ in 0..100 {
                let _ = registry_clone.get(VertexFormatId::StaticMesh);
                let _ = registry_clone.contains(VertexFormatId::Terrain);
                let _ = registry_clone.len();
                let count: usize = registry_clone.iter().count();
                assert_eq!(count, 5);
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

// ===========================================================================
// 10. Integration with VertexStateDescriptor Tests
// ===========================================================================

#[test]
fn test_buffer_layout_compatible_with_vertex_state_descriptor() {
    // This test verifies that buffer layouts from the registry can be used
    // with VertexStateDescriptor's buffer() method (compile-time check)

    // Note: We cannot create an actual VertexStateDescriptor without a ShaderModule,
    // but we can verify the type compatibility by checking the method signature.
    // The layout should be of type VertexBufferLayoutDescriptor which is what
    // VertexStateDescriptor::buffer() accepts.

    let registry = VertexFormatRegistry::new();
    let layout = registry.get_buffer_layout(VertexFormatId::StaticMesh).unwrap();

    // Verify the type is VertexBufferLayoutDescriptor
    let _: VertexBufferLayoutDescriptor = layout;
}

#[test]
fn test_format_to_buffer_layout_returns_correct_type() {
    let format = static_mesh();
    let layout: VertexBufferLayoutDescriptor = format.to_buffer_layout();

    // Verify all expected fields are accessible
    let _stride: u64 = layout.array_stride;
    let _mode: wgpu::VertexStepMode = layout.step_mode;
    let _attrs: &Vec<VertexAttributeDescriptor> = &layout.attributes;
}

#[test]
fn test_multiple_buffer_layouts_can_be_collected() {
    let registry = VertexFormatRegistry::new();

    // Simulate preparing multiple buffer layouts for a complex vertex setup
    // (e.g., interleaved vertex buffer + instance buffer)
    let layouts: Vec<VertexBufferLayoutDescriptor> = vec![
        registry.get_buffer_layout(VertexFormatId::StaticMesh).unwrap(),
        VertexBufferLayoutDescriptor::per_instance(64)
            .with_attribute(wgpu::VertexFormat::Float32x4, 0, 10)
            .with_attribute(wgpu::VertexFormat::Float32x4, 16, 11)
            .with_attribute(wgpu::VertexFormat::Float32x4, 32, 12)
            .with_attribute(wgpu::VertexFormat::Float32x4, 48, 13),
    ];

    assert_eq!(layouts.len(), 2);
    assert_eq!(layouts[0].step_mode, wgpu::VertexStepMode::Vertex);
    assert_eq!(layouts[1].step_mode, wgpu::VertexStepMode::Instance);
}

// ===========================================================================
// Additional Integration Tests
// ===========================================================================

#[test]
fn test_skinned_mesh_extends_static_mesh_layout() {
    let static_fmt = static_mesh();
    let skinned_fmt = skinned_mesh();

    // Skinned mesh should have all static mesh attributes plus bone data
    assert!(skinned_fmt.attributes.len() > static_fmt.attributes.len());
    assert!(skinned_fmt.stride > static_fmt.stride);

    // First 4 attributes should have same offsets (position, normal, tangent, uv)
    for i in 0..4 {
        assert_eq!(
            static_fmt.attributes[i].offset, skinned_fmt.attributes[i].offset,
            "Base attribute {} offset should match",
            i
        );
    }
}

#[test]
fn test_all_standard_formats_have_position_at_offset_zero() {
    let formats = [
        static_mesh(),
        skinned_mesh(),
        terrain(),
        particle(),
        ui(),
    ];

    for format in formats {
        assert_eq!(
            format.attributes[0].offset,
            0,
            "{} should have position at offset 0",
            format.name
        );
        assert_eq!(
            format.attributes[0].shader_location,
            0,
            "{} should have position at shader location 0",
            format.name
        );
    }
}

#[test]
fn test_registry_is_not_empty_by_default() {
    let registry = VertexFormatRegistry::new();

    // is_empty() should be the inverse of having any formats
    assert!(!registry.is_empty());
    assert!(registry.len() > 0);
}

#[test]
fn test_attribute_format_sizes_match_stride() {
    // For each standard format, verify that attribute sizes add up correctly
    // (accounting for potential padding which we accept)

    let format = static_mesh();
    // Float32x3 (12) + Float32x3 (12) + Float32x4 (16) + Float32x2 (8) = 48
    let expected_min: u64 = 12 + 12 + 16 + 8;
    assert!(format.stride >= expected_min);

    let format = terrain();
    // Float32x3 (12) + Float32x3 (12) + Float32x2 (8) = 32
    let expected_min: u64 = 12 + 12 + 8;
    assert!(format.stride >= expected_min);

    let format = ui();
    // Float32x2 (8) + Float32x2 (8) + Unorm8x4 (4) = 20
    let expected_min: u64 = 8 + 8 + 4;
    assert!(format.stride >= expected_min);
}

#[test]
fn test_format_constructors_return_owned_values() {
    // Verify that each constructor returns an owned VertexFormat
    // (not a reference), allowing independent modification
    let mut format1 = static_mesh();
    let format2 = static_mesh();

    // Modifying format1 should not affect format2
    format1.attributes.clear();

    assert!(format1.attributes.is_empty());
    assert!(!format2.attributes.is_empty());
}
