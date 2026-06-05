// SPDX-License-Identifier: MIT
//
// blackbox_vertex.rs -- Blackbox contract tests for T-WGPU-P2.1.4 (Vertex Buffer Registry).
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::resources::*` -- no internal fields,
// no private methods, no implementation details.
//
// Public API under test:
//   VertexLayoutId       -- Pbr, Skinned, Terrain, Particle, Ui, Custom(u32)
//   PbrVertex            -- 48 bytes, position/normal/tangent/uv
//   SkinnedVertex        -- 72 bytes, position/normal/uv/tangent/joints/weights
//   TerrainVertex        -- 32 bytes, position/normal/uv
//   ParticleVertex       -- 32 bytes, position/color/size/life/rotation/_padding
//   UiVertex             -- 20 bytes, position/uv/color
//   VertexFormatRegistry -- new() with standard layouts
//   registry.get(id)     -- Option<&VertexBufferLayout>
//
// Acceptance criteria (T-WGPU-P2.1.4):
//   1.  PbrVertex construction and size verification (48 bytes)
//   2.  SkinnedVertex construction and size verification (72 bytes)
//   3.  TerrainVertex construction and size verification (32 bytes)
//   4.  ParticleVertex construction and size verification (32 bytes)
//   5.  UiVertex construction and size verification (20 bytes)
//   6.  VertexFormatRegistry creation with standard layouts
//   7.  Registry.get(VertexLayoutId::Pbr) returns layout
//   8.  Registry.get(VertexLayoutId::Skinned) returns layout
//   9.  Registry.get(VertexLayoutId::Terrain) returns layout
//  10.  Registry.get(VertexLayoutId::Particle) returns layout
//  11.  Registry.get(VertexLayoutId::Ui) returns layout
//  12.  Layouts have correct array_stride matching vertex sizes
//  13.  Layouts have non-empty attribute arrays
//  14.  VertexLayoutId enum variants are distinct
//  15.  VertexLayoutId::Custom(0) != VertexLayoutId::Custom(1)
//  16.  VertexLayoutId implements Debug, Clone, Copy traits
//  17.  Registry.get with unregistered Custom id returns None

use renderer_backend::resources::{
    ParticleVertex, PbrVertex, SkinnedVertex, TerrainVertex, UiVertex,
    VertexFormatRegistry, VertexLayoutId,
};

// ---------------------------------------------------------------------------
// Test 1: PbrVertex Construction and Size
// ---------------------------------------------------------------------------

#[test]
fn test_pbr_vertex_construction() {
    let vertex = PbrVertex {
        position: [1.0, 2.0, 3.0],
        normal: [0.0, 1.0, 0.0],
        tangent: [1.0, 0.0, 0.0, 1.0],
        uv: [0.5, 0.5],
    };

    assert_eq!(vertex.position, [1.0, 2.0, 3.0]);
    assert_eq!(vertex.normal, [0.0, 1.0, 0.0]);
    assert_eq!(vertex.tangent, [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(vertex.uv, [0.5, 0.5]);
}

#[test]
fn test_pbr_vertex_size() {
    // PbrVertex should be 48 bytes:
    // position: [f32; 3] = 12 bytes
    // normal:   [f32; 3] = 12 bytes
    // tangent:  [f32; 4] = 16 bytes
    // uv:       [f32; 2] =  8 bytes
    // Total: 48 bytes
    assert_eq!(std::mem::size_of::<PbrVertex>(), 48);
}

// ---------------------------------------------------------------------------
// Test 2: SkinnedVertex Construction and Size
// ---------------------------------------------------------------------------

#[test]
fn test_skinned_vertex_construction() {
    let vertex = SkinnedVertex {
        position: [1.0, 2.0, 3.0],
        normal: [0.0, 1.0, 0.0],
        uv: [0.5, 0.5],
        tangent: [1.0, 0.0, 0.0, 1.0],
        joints: [0, 1, 2, 3],
        weights: [0.5, 0.3, 0.15, 0.05],
    };

    assert_eq!(vertex.position, [1.0, 2.0, 3.0]);
    assert_eq!(vertex.normal, [0.0, 1.0, 0.0]);
    assert_eq!(vertex.tangent, [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(vertex.uv, [0.5, 0.5]);
    assert_eq!(vertex.joints, [0, 1, 2, 3]);
    assert_eq!(vertex.weights, [0.5, 0.3, 0.15, 0.05]);
}

#[test]
fn test_skinned_vertex_size() {
    // SkinnedVertex layout:
    // position: [f32; 3] = 12 bytes
    // normal:   [f32; 3] = 12 bytes
    // uv:       [f32; 2] =  8 bytes
    // tangent:  [f32; 4] = 16 bytes
    // joints:   [u16; 4] =  8 bytes
    // weights:  [f32; 4] = 16 bytes
    // Total: 72 bytes
    let size = std::mem::size_of::<SkinnedVertex>();
    assert_eq!(size, 72, "SkinnedVertex size should be 72 bytes, got {}", size);
}

// ---------------------------------------------------------------------------
// Test 3: TerrainVertex Construction and Size
// ---------------------------------------------------------------------------

#[test]
fn test_terrain_vertex_construction() {
    let vertex = TerrainVertex {
        position: [100.0, 50.0, 200.0],
        normal: [0.0, 1.0, 0.0],
        uv: [0.25, 0.75],
    };

    assert_eq!(vertex.position, [100.0, 50.0, 200.0]);
    assert_eq!(vertex.normal, [0.0, 1.0, 0.0]);
    assert_eq!(vertex.uv, [0.25, 0.75]);
}

#[test]
fn test_terrain_vertex_size() {
    // TerrainVertex should be 32 bytes:
    // position: [f32; 3] = 12 bytes
    // normal:   [f32; 3] = 12 bytes
    // uv:       [f32; 2] =  8 bytes
    // Total: 32 bytes
    assert_eq!(std::mem::size_of::<TerrainVertex>(), 32);
}

// ---------------------------------------------------------------------------
// Test 4: ParticleVertex Construction and Size
// ---------------------------------------------------------------------------

#[test]
fn test_particle_vertex_construction() {
    let vertex = ParticleVertex {
        position: [10.0, 20.0, 30.0],
        color: 0xFF8000FF, // Packed RGBA (orange with alpha)
        size: 2.5,
        life: 0.5, // Normalized lifetime (0.0-1.0)
        rotation: 1.57, // Rotation in radians
        _padding: 0,
    };

    assert_eq!(vertex.position, [10.0, 20.0, 30.0]);
    assert_eq!(vertex.color, 0xFF8000FF);
    assert_eq!(vertex.size, 2.5);
    assert_eq!(vertex.life, 0.5);
    assert_eq!(vertex.rotation, 1.57);
}

#[test]
fn test_particle_vertex_size() {
    // ParticleVertex layout:
    // position: [f32; 3] = 12 bytes
    // color:    u32      =  4 bytes
    // size:     f32      =  4 bytes
    // life:     f32      =  4 bytes
    // rotation: f32      =  4 bytes
    // _padding: u32      =  4 bytes
    // Total: 32 bytes
    let size = std::mem::size_of::<ParticleVertex>();
    assert_eq!(size, 32, "ParticleVertex size should be 32 bytes, got {}", size);
}

// ---------------------------------------------------------------------------
// Test 5: UiVertex Construction and Size
// ---------------------------------------------------------------------------

#[test]
fn test_ui_vertex_construction() {
    let vertex = UiVertex {
        position: [100.0, 200.0],
        uv: [0.0, 1.0],
        color: 0xFF00FF00, // ARGB green
    };

    assert_eq!(vertex.position, [100.0, 200.0]);
    assert_eq!(vertex.uv, [0.0, 1.0]);
    assert_eq!(vertex.color, 0xFF00FF00);
}

#[test]
fn test_ui_vertex_size() {
    // UiVertex should be 20 bytes:
    // position: [f32; 2] =  8 bytes
    // uv:       [f32; 2] =  8 bytes
    // color:    u32      =  4 bytes
    // Total: 20 bytes
    assert_eq!(std::mem::size_of::<UiVertex>(), 20);
}

// ---------------------------------------------------------------------------
// Test 6: VertexFormatRegistry Creation
// ---------------------------------------------------------------------------

#[test]
fn test_registry_creation() {
    let registry = VertexFormatRegistry::new();
    // Registry should be created successfully with standard layouts
    // We verify this by checking that we can get standard layouts
    assert!(registry.get(VertexLayoutId::Pbr).is_some());
}

// ---------------------------------------------------------------------------
// Test 7-11: Registry Layout Retrieval
// ---------------------------------------------------------------------------

#[test]
fn test_registry_get_pbr_layout() {
    let registry = VertexFormatRegistry::new();
    let layout = registry.get(VertexLayoutId::Pbr);
    assert!(layout.is_some(), "PBR layout should be registered");
}

#[test]
fn test_registry_get_skinned_layout() {
    let registry = VertexFormatRegistry::new();
    let layout = registry.get(VertexLayoutId::Skinned);
    assert!(layout.is_some(), "Skinned layout should be registered");
}

#[test]
fn test_registry_get_terrain_layout() {
    let registry = VertexFormatRegistry::new();
    let layout = registry.get(VertexLayoutId::Terrain);
    assert!(layout.is_some(), "Terrain layout should be registered");
}

#[test]
fn test_registry_get_particle_layout() {
    let registry = VertexFormatRegistry::new();
    let layout = registry.get(VertexLayoutId::Particle);
    assert!(layout.is_some(), "Particle layout should be registered");
}

#[test]
fn test_registry_get_ui_layout() {
    let registry = VertexFormatRegistry::new();
    let layout = registry.get(VertexLayoutId::Ui);
    assert!(layout.is_some(), "UI layout should be registered");
}

// ---------------------------------------------------------------------------
// Test 12: Layout Array Stride Verification
// ---------------------------------------------------------------------------

#[test]
fn test_layout_array_stride_pbr() {
    let registry = VertexFormatRegistry::new();
    let layout = registry.get(VertexLayoutId::Pbr).expect("PBR layout exists");
    // array_stride should match PbrVertex size
    assert_eq!(
        layout.array_stride as usize,
        std::mem::size_of::<PbrVertex>(),
        "PBR layout stride should match PbrVertex size"
    );
}

#[test]
fn test_layout_array_stride_terrain() {
    let registry = VertexFormatRegistry::new();
    let layout = registry
        .get(VertexLayoutId::Terrain)
        .expect("Terrain layout exists");
    // array_stride should match TerrainVertex size
    assert_eq!(
        layout.array_stride as usize,
        std::mem::size_of::<TerrainVertex>(),
        "Terrain layout stride should match TerrainVertex size"
    );
}

#[test]
fn test_layout_array_stride_ui() {
    let registry = VertexFormatRegistry::new();
    let layout = registry.get(VertexLayoutId::Ui).expect("UI layout exists");
    // array_stride should match UiVertex size
    assert_eq!(
        layout.array_stride as usize,
        std::mem::size_of::<UiVertex>(),
        "UI layout stride should match UiVertex size"
    );
}

// ---------------------------------------------------------------------------
// Test 13: Layouts Have Attributes
// ---------------------------------------------------------------------------

#[test]
fn test_layout_has_attributes_pbr() {
    let registry = VertexFormatRegistry::new();
    let layout = registry.get(VertexLayoutId::Pbr).expect("PBR layout exists");
    assert!(
        !layout.attributes.is_empty(),
        "PBR layout should have attributes"
    );
}

#[test]
fn test_layout_has_attributes_skinned() {
    let registry = VertexFormatRegistry::new();
    let layout = registry
        .get(VertexLayoutId::Skinned)
        .expect("Skinned layout exists");
    assert!(
        !layout.attributes.is_empty(),
        "Skinned layout should have attributes"
    );
    // Skinned should have more attributes than PBR (joint indices + weights)
    let pbr_layout = registry.get(VertexLayoutId::Pbr).expect("PBR layout exists");
    assert!(
        layout.attributes.len() >= pbr_layout.attributes.len(),
        "Skinned layout should have at least as many attributes as PBR"
    );
}

#[test]
fn test_layout_has_attributes_terrain() {
    let registry = VertexFormatRegistry::new();
    let layout = registry
        .get(VertexLayoutId::Terrain)
        .expect("Terrain layout exists");
    assert!(
        !layout.attributes.is_empty(),
        "Terrain layout should have attributes"
    );
}

#[test]
fn test_layout_has_attributes_particle() {
    let registry = VertexFormatRegistry::new();
    let layout = registry
        .get(VertexLayoutId::Particle)
        .expect("Particle layout exists");
    assert!(
        !layout.attributes.is_empty(),
        "Particle layout should have attributes"
    );
}

#[test]
fn test_layout_has_attributes_ui() {
    let registry = VertexFormatRegistry::new();
    let layout = registry.get(VertexLayoutId::Ui).expect("UI layout exists");
    assert!(
        !layout.attributes.is_empty(),
        "UI layout should have attributes"
    );
}

// ---------------------------------------------------------------------------
// Test 14: VertexLayoutId Enum Variants Are Distinct
// ---------------------------------------------------------------------------

#[test]
fn test_vertex_layout_id_variants_distinct() {
    // Each standard variant should be distinct
    assert_ne!(
        std::mem::discriminant(&VertexLayoutId::Pbr),
        std::mem::discriminant(&VertexLayoutId::Skinned)
    );
    assert_ne!(
        std::mem::discriminant(&VertexLayoutId::Pbr),
        std::mem::discriminant(&VertexLayoutId::Terrain)
    );
    assert_ne!(
        std::mem::discriminant(&VertexLayoutId::Pbr),
        std::mem::discriminant(&VertexLayoutId::Particle)
    );
    assert_ne!(
        std::mem::discriminant(&VertexLayoutId::Pbr),
        std::mem::discriminant(&VertexLayoutId::Ui)
    );
    assert_ne!(
        std::mem::discriminant(&VertexLayoutId::Skinned),
        std::mem::discriminant(&VertexLayoutId::Terrain)
    );
}

// ---------------------------------------------------------------------------
// Test 15: Custom Variants Are Distinct
// ---------------------------------------------------------------------------

#[test]
fn test_vertex_layout_id_custom_distinct() {
    let custom_0 = VertexLayoutId::Custom(0);
    let custom_1 = VertexLayoutId::Custom(1);
    let custom_100 = VertexLayoutId::Custom(100);

    // Custom variants with different values should not be equal
    assert_ne!(custom_0, custom_1);
    assert_ne!(custom_0, custom_100);
    assert_ne!(custom_1, custom_100);

    // Same custom value should be equal
    assert_eq!(VertexLayoutId::Custom(42), VertexLayoutId::Custom(42));
}

// ---------------------------------------------------------------------------
// Test 16: VertexLayoutId Trait Implementations
// ---------------------------------------------------------------------------

#[test]
fn test_vertex_layout_id_debug() {
    // Debug trait should be implemented
    let id = VertexLayoutId::Pbr;
    let debug_str = format!("{:?}", id);
    assert!(!debug_str.is_empty(), "Debug output should not be empty");
}

#[test]
fn test_vertex_layout_id_clone() {
    // Clone trait should be implemented
    let id = VertexLayoutId::Skinned;
    let cloned = id.clone();
    assert_eq!(id, cloned);
}

#[test]
fn test_vertex_layout_id_copy() {
    // Copy trait should be implemented (no move semantics)
    let id = VertexLayoutId::Terrain;
    let copied = id; // This would fail to compile without Copy
    assert_eq!(id, copied); // Both should still be usable
}

// ---------------------------------------------------------------------------
// Test 17: Unregistered Custom Layout Returns None
// ---------------------------------------------------------------------------

#[test]
fn test_registry_get_unregistered_custom() {
    let registry = VertexFormatRegistry::new();
    // An arbitrary custom ID that was never registered should return None
    let layout = registry.get(VertexLayoutId::Custom(999999));
    assert!(
        layout.is_none(),
        "Unregistered custom layout should return None"
    );
}

// ---------------------------------------------------------------------------
// Additional Edge Case Tests
// ---------------------------------------------------------------------------

#[test]
fn test_vertex_default_values() {
    // Ensure vertices can be created with zero/default values
    let pbr = PbrVertex {
        position: [0.0, 0.0, 0.0],
        normal: [0.0, 0.0, 0.0],
        tangent: [0.0, 0.0, 0.0, 0.0],
        uv: [0.0, 0.0],
    };
    assert_eq!(pbr.position[0], 0.0);

    let terrain = TerrainVertex {
        position: [0.0, 0.0, 0.0],
        normal: [0.0, 0.0, 0.0],
        uv: [0.0, 0.0],
    };
    assert_eq!(terrain.position[0], 0.0);
}

#[test]
fn test_vertex_extreme_values() {
    // Ensure vertices handle extreme float values
    let pbr = PbrVertex {
        position: [f32::MAX, f32::MIN, f32::INFINITY],
        normal: [f32::NEG_INFINITY, f32::NAN, 0.0],
        tangent: [1e38, -1e38, 1e-38, -1e-38],
        uv: [f32::EPSILON, 1.0 - f32::EPSILON],
    };
    assert_eq!(pbr.position[0], f32::MAX);
    assert_eq!(pbr.position[2], f32::INFINITY);
}

#[test]
fn test_all_standard_layouts_registered() {
    let registry = VertexFormatRegistry::new();

    let standard_ids = [
        VertexLayoutId::Pbr,
        VertexLayoutId::Skinned,
        VertexLayoutId::Terrain,
        VertexLayoutId::Particle,
        VertexLayoutId::Ui,
    ];

    for id in standard_ids {
        assert!(
            registry.get(id).is_some(),
            "Standard layout {:?} should be registered",
            id
        );
    }
}

#[test]
fn test_layout_step_mode() {
    let registry = VertexFormatRegistry::new();
    let layout = registry.get(VertexLayoutId::Pbr).expect("PBR layout exists");
    // Vertex layouts typically use Vertex step mode (per-vertex data)
    // This verifies the step_mode field exists and is accessible
    let _ = layout.step_mode; // Just verify field exists
}

#[test]
fn test_attribute_shader_locations_unique() {
    let registry = VertexFormatRegistry::new();
    let layout = registry.get(VertexLayoutId::Pbr).expect("PBR layout exists");

    // Collect all shader locations
    let locations: Vec<u32> = layout
        .attributes
        .iter()
        .map(|attr| attr.shader_location)
        .collect();

    // Verify no duplicates
    let mut sorted = locations.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(
        locations.len(),
        sorted.len(),
        "Shader locations should be unique within a layout"
    );
}
