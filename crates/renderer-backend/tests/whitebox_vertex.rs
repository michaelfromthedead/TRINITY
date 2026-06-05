//! Whitebox tests for vertex buffer registry (T-WGPU-P2.1.4).
//!
//! These tests verify implementation details of the vertex format system:
//! - Exact vertex struct sizes and memory layouts
//! - Attribute locations and formats
//! - Registry lookup and registration
//! - Bytemuck compatibility for GPU buffer uploads
//! - Color packing utilities

use renderer_backend::resources::vertex::{
    ParticleVertex, PbrVertex, SkinnedVertex, TerrainVertex, UiVertex,
    VertexFormatRegistry, VertexLayoutId,
};
use std::mem::{align_of, size_of};
use wgpu::{vertex_attr_array, VertexAttribute, VertexBufferLayout, VertexFormat, VertexStepMode};

// ============================================================================
// SIZE TESTS - Verify exact byte sizes for each vertex type
// ============================================================================

mod size_tests {
    use super::*;

    #[test]
    fn pbr_vertex_is_48_bytes() {
        assert_eq!(size_of::<PbrVertex>(), 48, "PbrVertex must be exactly 48 bytes");
    }

    #[test]
    fn skinned_vertex_is_72_bytes() {
        assert_eq!(size_of::<SkinnedVertex>(), 72, "SkinnedVertex must be exactly 72 bytes");
    }

    #[test]
    fn terrain_vertex_is_32_bytes() {
        assert_eq!(size_of::<TerrainVertex>(), 32, "TerrainVertex must be exactly 32 bytes");
    }

    #[test]
    fn particle_vertex_is_32_bytes() {
        assert_eq!(size_of::<ParticleVertex>(), 32, "ParticleVertex must be exactly 32 bytes");
    }

    #[test]
    fn ui_vertex_is_20_bytes() {
        assert_eq!(size_of::<UiVertex>(), 20, "UiVertex must be exactly 20 bytes");
    }

    #[test]
    fn skinned_is_pbr_plus_skinning_data() {
        // Skinned = PBR (48) + joints (8) + weights (16) = 72
        let pbr_size = size_of::<PbrVertex>();
        let joints_size = size_of::<[u16; 4]>();
        let weights_size = size_of::<[f32; 4]>();
        assert_eq!(
            size_of::<SkinnedVertex>(),
            pbr_size + joints_size + weights_size,
            "SkinnedVertex should be PbrVertex + joints + weights"
        );
    }

    #[test]
    fn terrain_is_pbr_minus_tangent() {
        // Terrain = pos (12) + normal (12) + uv (8) = 32
        // PBR = pos (12) + normal (12) + uv (8) + tangent (16) = 48
        let tangent_size = size_of::<[f32; 4]>();
        assert_eq!(
            size_of::<TerrainVertex>(),
            size_of::<PbrVertex>() - tangent_size,
            "TerrainVertex should be PbrVertex minus tangent"
        );
    }

    #[test]
    fn vertex_alignment_is_4_bytes() {
        // All vertex types should have 4-byte alignment for GPU compatibility
        assert_eq!(align_of::<PbrVertex>(), 4);
        assert_eq!(align_of::<SkinnedVertex>(), 4);
        assert_eq!(align_of::<TerrainVertex>(), 4);
        assert_eq!(align_of::<ParticleVertex>(), 4);
        assert_eq!(align_of::<UiVertex>(), 4);
    }
}

// ============================================================================
// LAYOUT TESTS - Verify buffer layouts match struct sizes
// ============================================================================

mod layout_tests {
    use super::*;

    #[test]
    fn pbr_layout_stride_matches_size() {
        assert_eq!(
            PbrVertex::LAYOUT.array_stride,
            size_of::<PbrVertex>() as u64,
            "PBR layout stride must match struct size"
        );
    }

    #[test]
    fn skinned_layout_stride_matches_size() {
        assert_eq!(
            SkinnedVertex::LAYOUT.array_stride,
            size_of::<SkinnedVertex>() as u64,
            "Skinned layout stride must match struct size"
        );
    }

    #[test]
    fn terrain_layout_stride_matches_size() {
        assert_eq!(
            TerrainVertex::LAYOUT.array_stride,
            size_of::<TerrainVertex>() as u64,
            "Terrain layout stride must match struct size"
        );
    }

    #[test]
    fn particle_layout_stride_matches_size() {
        assert_eq!(
            ParticleVertex::LAYOUT.array_stride,
            size_of::<ParticleVertex>() as u64,
            "Particle layout stride must match struct size"
        );
    }

    #[test]
    fn ui_layout_stride_matches_size() {
        assert_eq!(
            UiVertex::LAYOUT.array_stride,
            size_of::<UiVertex>() as u64,
            "UI layout stride must match struct size"
        );
    }

    #[test]
    fn pbr_step_mode_is_vertex() {
        assert_eq!(
            PbrVertex::LAYOUT.step_mode,
            VertexStepMode::Vertex,
            "PBR should use per-vertex stepping"
        );
    }

    #[test]
    fn skinned_step_mode_is_vertex() {
        assert_eq!(
            SkinnedVertex::LAYOUT.step_mode,
            VertexStepMode::Vertex,
            "Skinned should use per-vertex stepping"
        );
    }

    #[test]
    fn terrain_step_mode_is_vertex() {
        assert_eq!(
            TerrainVertex::LAYOUT.step_mode,
            VertexStepMode::Vertex,
            "Terrain should use per-vertex stepping"
        );
    }

    #[test]
    fn particle_step_mode_is_instance() {
        assert_eq!(
            ParticleVertex::LAYOUT.step_mode,
            VertexStepMode::Instance,
            "Particle default layout should use per-instance stepping for GPU particles"
        );
    }

    #[test]
    fn particle_vertex_layout_is_vertex() {
        assert_eq!(
            ParticleVertex::LAYOUT_VERTEX.step_mode,
            VertexStepMode::Vertex,
            "Particle LAYOUT_VERTEX should use per-vertex stepping"
        );
    }

    #[test]
    fn ui_step_mode_is_vertex() {
        assert_eq!(
            UiVertex::LAYOUT.step_mode,
            VertexStepMode::Vertex,
            "UI should use per-vertex stepping"
        );
    }
}

// ============================================================================
// ATTRIBUTE TESTS - Verify shader locations are sequential
// ============================================================================

mod attribute_tests {
    use super::*;

    fn verify_sequential_locations(attrs: &[VertexAttribute], name: &str) {
        for (i, attr) in attrs.iter().enumerate() {
            assert_eq!(
                attr.shader_location,
                i as u32,
                "{} attribute {} should have location {}",
                name,
                i,
                i
            );
        }
    }

    #[test]
    fn pbr_attributes_sequential() {
        verify_sequential_locations(&PbrVertex::ATTRIBS, "PBR");
        assert_eq!(PbrVertex::ATTRIBS.len(), 4, "PBR should have 4 attributes");
    }

    #[test]
    fn skinned_attributes_sequential() {
        verify_sequential_locations(&SkinnedVertex::ATTRIBS, "Skinned");
        assert_eq!(SkinnedVertex::ATTRIBS.len(), 6, "Skinned should have 6 attributes");
    }

    #[test]
    fn terrain_attributes_sequential() {
        verify_sequential_locations(&TerrainVertex::ATTRIBS, "Terrain");
        assert_eq!(TerrainVertex::ATTRIBS.len(), 3, "Terrain should have 3 attributes");
    }

    #[test]
    fn particle_attributes_sequential() {
        verify_sequential_locations(&ParticleVertex::ATTRIBS, "Particle");
        assert_eq!(ParticleVertex::ATTRIBS.len(), 5, "Particle should have 5 attributes");
    }

    #[test]
    fn ui_attributes_sequential() {
        verify_sequential_locations(&UiVertex::ATTRIBS, "UI");
        assert_eq!(UiVertex::ATTRIBS.len(), 3, "UI should have 3 attributes");
    }

    #[test]
    fn pbr_attribute_formats() {
        let attrs = &PbrVertex::ATTRIBS;
        assert_eq!(attrs[0].format, VertexFormat::Float32x3, "position should be Float32x3");
        assert_eq!(attrs[1].format, VertexFormat::Float32x3, "normal should be Float32x3");
        assert_eq!(attrs[2].format, VertexFormat::Float32x2, "uv should be Float32x2");
        assert_eq!(attrs[3].format, VertexFormat::Float32x4, "tangent should be Float32x4");
    }

    #[test]
    fn skinned_attribute_formats() {
        let attrs = &SkinnedVertex::ATTRIBS;
        // First 4 same as PBR
        assert_eq!(attrs[0].format, VertexFormat::Float32x3, "position should be Float32x3");
        assert_eq!(attrs[1].format, VertexFormat::Float32x3, "normal should be Float32x3");
        assert_eq!(attrs[2].format, VertexFormat::Float32x2, "uv should be Float32x2");
        assert_eq!(attrs[3].format, VertexFormat::Float32x4, "tangent should be Float32x4");
        // Skinning data
        assert_eq!(attrs[4].format, VertexFormat::Uint16x4, "joints should be Uint16x4");
        assert_eq!(attrs[5].format, VertexFormat::Float32x4, "weights should be Float32x4");
    }

    #[test]
    fn terrain_attribute_formats() {
        let attrs = &TerrainVertex::ATTRIBS;
        assert_eq!(attrs[0].format, VertexFormat::Float32x3, "position should be Float32x3");
        assert_eq!(attrs[1].format, VertexFormat::Float32x3, "normal should be Float32x3");
        assert_eq!(attrs[2].format, VertexFormat::Float32x2, "uv should be Float32x2");
    }

    #[test]
    fn particle_attribute_formats() {
        let attrs = &ParticleVertex::ATTRIBS;
        assert_eq!(attrs[0].format, VertexFormat::Float32x3, "position should be Float32x3");
        assert_eq!(attrs[1].format, VertexFormat::Uint32, "color should be Uint32 (packed RGBA)");
        assert_eq!(attrs[2].format, VertexFormat::Float32, "size should be Float32");
        assert_eq!(attrs[3].format, VertexFormat::Float32, "life should be Float32");
        assert_eq!(attrs[4].format, VertexFormat::Float32, "rotation should be Float32");
    }

    #[test]
    fn ui_attribute_formats() {
        let attrs = &UiVertex::ATTRIBS;
        assert_eq!(attrs[0].format, VertexFormat::Float32x2, "position should be Float32x2 (2D)");
        assert_eq!(attrs[1].format, VertexFormat::Float32x2, "uv should be Float32x2");
        assert_eq!(attrs[2].format, VertexFormat::Uint32, "color should be Uint32 (packed RGBA)");
    }

    #[test]
    fn pbr_attribute_offsets_correct() {
        let attrs = &PbrVertex::ATTRIBS;
        assert_eq!(attrs[0].offset, 0, "position at offset 0");
        assert_eq!(attrs[1].offset, 12, "normal at offset 12 (after 3 floats)");
        assert_eq!(attrs[2].offset, 24, "uv at offset 24 (after 6 floats)");
        assert_eq!(attrs[3].offset, 32, "tangent at offset 32 (after 8 floats)");
    }

    #[test]
    fn skinned_attribute_offsets_correct() {
        let attrs = &SkinnedVertex::ATTRIBS;
        assert_eq!(attrs[0].offset, 0, "position at offset 0");
        assert_eq!(attrs[1].offset, 12, "normal at offset 12");
        assert_eq!(attrs[2].offset, 24, "uv at offset 24");
        assert_eq!(attrs[3].offset, 32, "tangent at offset 32");
        assert_eq!(attrs[4].offset, 48, "joints at offset 48 (after tangent)");
        assert_eq!(attrs[5].offset, 56, "weights at offset 56 (after joints[u16;4]=8 bytes)");
    }

    #[test]
    fn terrain_attribute_offsets_correct() {
        let attrs = &TerrainVertex::ATTRIBS;
        assert_eq!(attrs[0].offset, 0, "position at offset 0");
        assert_eq!(attrs[1].offset, 12, "normal at offset 12");
        assert_eq!(attrs[2].offset, 24, "uv at offset 24");
    }

    #[test]
    fn particle_attribute_offsets_correct() {
        let attrs = &ParticleVertex::ATTRIBS;
        assert_eq!(attrs[0].offset, 0, "position at offset 0");
        assert_eq!(attrs[1].offset, 12, "color at offset 12 (after 3 floats)");
        assert_eq!(attrs[2].offset, 16, "size at offset 16 (after color u32)");
        assert_eq!(attrs[3].offset, 20, "life at offset 20");
        assert_eq!(attrs[4].offset, 24, "rotation at offset 24");
        // Note: _padding is at 28, total 32 bytes
    }

    #[test]
    fn ui_attribute_offsets_correct() {
        let attrs = &UiVertex::ATTRIBS;
        assert_eq!(attrs[0].offset, 0, "position at offset 0");
        assert_eq!(attrs[1].offset, 8, "uv at offset 8 (after 2 floats)");
        assert_eq!(attrs[2].offset, 16, "color at offset 16 (after 4 floats)");
    }
}

// ============================================================================
// REGISTRY TESTS - Verify registry lookup and registration
// ============================================================================

mod registry_tests {
    use super::*;

    #[test]
    fn new_registry_has_all_standard_layouts() {
        let registry = VertexFormatRegistry::new();

        assert!(registry.get(VertexLayoutId::Pbr).is_some(), "PBR should be registered");
        assert!(registry.get(VertexLayoutId::Skinned).is_some(), "Skinned should be registered");
        assert!(registry.get(VertexLayoutId::Terrain).is_some(), "Terrain should be registered");
        assert!(registry.get(VertexLayoutId::Particle).is_some(), "Particle should be registered");
        assert!(registry.get(VertexLayoutId::Ui).is_some(), "UI should be registered");
    }

    #[test]
    fn registry_has_five_standard_layouts() {
        let registry = VertexFormatRegistry::new();
        assert_eq!(registry.len(), 5, "Registry should have exactly 5 standard layouts");
    }

    #[test]
    fn registry_get_returns_none_for_unknown_custom() {
        let registry = VertexFormatRegistry::new();

        assert!(registry.get(VertexLayoutId::Custom(0)).is_none());
        assert!(registry.get(VertexLayoutId::Custom(1)).is_none());
        assert!(registry.get(VertexLayoutId::Custom(999)).is_none());
        assert!(registry.get(VertexLayoutId::Custom(u32::MAX)).is_none());
    }

    #[test]
    fn registry_register_adds_custom_layout() {
        static CUSTOM_ATTRS: [VertexAttribute; 2] = vertex_attr_array![
            0 => Float32x3,
            1 => Float32x3
        ];

        let mut registry = VertexFormatRegistry::new();
        let custom_id = VertexLayoutId::Custom(42);

        assert!(registry.get(custom_id).is_none(), "Custom layout should not exist yet");

        let custom_layout = VertexBufferLayout {
            array_stride: 24,
            step_mode: VertexStepMode::Vertex,
            attributes: &CUSTOM_ATTRS,
        };
        registry.register(custom_id, custom_layout);

        assert!(registry.get(custom_id).is_some(), "Custom layout should exist after registration");
        assert!(registry.contains(custom_id), "contains() should return true");
        assert_eq!(registry.stride(custom_id), Some(24), "stride should return correct value");
    }

    #[test]
    fn registry_register_overwrites_existing() {
        static CUSTOM_ATTRS_A: [VertexAttribute; 1] = vertex_attr_array![0 => Float32x3];
        static CUSTOM_ATTRS_B: [VertexAttribute; 2] = vertex_attr_array![
            0 => Float32x3,
            1 => Float32x3
        ];

        let mut registry = VertexFormatRegistry::new();
        let custom_id = VertexLayoutId::Custom(1);

        let layout_a = VertexBufferLayout {
            array_stride: 12,
            step_mode: VertexStepMode::Vertex,
            attributes: &CUSTOM_ATTRS_A,
        };
        registry.register(custom_id, layout_a);
        assert_eq!(registry.stride(custom_id), Some(12));

        let layout_b = VertexBufferLayout {
            array_stride: 24,
            step_mode: VertexStepMode::Vertex,
            attributes: &CUSTOM_ATTRS_B,
        };
        registry.register(custom_id, layout_b);
        assert_eq!(registry.stride(custom_id), Some(24), "Should overwrite with new stride");
    }

    #[test]
    fn registry_unregister_removes_layout() {
        static CUSTOM_ATTRS: [VertexAttribute; 1] = vertex_attr_array![0 => Float32x3];

        let mut registry = VertexFormatRegistry::new();
        let custom_id = VertexLayoutId::Custom(100);

        let layout = VertexBufferLayout {
            array_stride: 12,
            step_mode: VertexStepMode::Vertex,
            attributes: &CUSTOM_ATTRS,
        };
        registry.register(custom_id, layout);
        assert!(registry.contains(custom_id));

        let removed = registry.unregister(custom_id);
        assert!(removed, "unregister should return true when removing existing layout");
        assert!(!registry.contains(custom_id), "Layout should be gone after unregister");
    }

    #[test]
    fn registry_unregister_returns_false_for_nonexistent() {
        let mut registry = VertexFormatRegistry::new();
        let removed = registry.unregister(VertexLayoutId::Custom(999));
        assert!(!removed, "unregister should return false for non-existent layout");
    }

    #[test]
    fn registry_default_equals_new() {
        let reg_new = VertexFormatRegistry::new();
        let reg_default = VertexFormatRegistry::default();

        assert_eq!(reg_new.len(), reg_default.len());
        assert!(reg_default.contains(VertexLayoutId::Pbr));
        assert!(reg_default.contains(VertexLayoutId::Skinned));
    }

    #[test]
    fn registry_layout_ids_iterator() {
        let registry = VertexFormatRegistry::new();
        let ids: Vec<_> = registry.layout_ids().collect();

        assert_eq!(ids.len(), 5);
        assert!(ids.contains(&&VertexLayoutId::Pbr));
        assert!(ids.contains(&&VertexLayoutId::Skinned));
        assert!(ids.contains(&&VertexLayoutId::Terrain));
        assert!(ids.contains(&&VertexLayoutId::Particle));
        assert!(ids.contains(&&VertexLayoutId::Ui));
    }

    #[test]
    fn registry_is_empty_false_for_new() {
        let registry = VertexFormatRegistry::new();
        assert!(!registry.is_empty());
    }

    #[test]
    fn registry_stride_returns_correct_values() {
        let registry = VertexFormatRegistry::new();

        assert_eq!(registry.stride(VertexLayoutId::Pbr), Some(48));
        assert_eq!(registry.stride(VertexLayoutId::Skinned), Some(72));
        assert_eq!(registry.stride(VertexLayoutId::Terrain), Some(32));
        assert_eq!(registry.stride(VertexLayoutId::Particle), Some(32));
        assert_eq!(registry.stride(VertexLayoutId::Ui), Some(20));
        assert_eq!(registry.stride(VertexLayoutId::Custom(999)), None);
    }
}

// ============================================================================
// BYTEMUCK TESTS - Verify Pod/Zeroable traits work correctly
// ============================================================================

mod bytemuck_tests {
    use super::*;

    #[test]
    fn pbr_vertex_to_bytes() {
        let vertex = PbrVertex::new(
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5],
            [1.0, 0.0, 0.0, 1.0],
        );
        let bytes: &[u8] = bytemuck::bytes_of(&vertex);
        assert_eq!(bytes.len(), 48);
    }

    #[test]
    fn skinned_vertex_to_bytes() {
        let vertex = SkinnedVertex::new(
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5],
            [1.0, 0.0, 0.0, 1.0],
            [0, 1, 2, 3],
            [0.5, 0.3, 0.15, 0.05],
        );
        let bytes: &[u8] = bytemuck::bytes_of(&vertex);
        assert_eq!(bytes.len(), 72);
    }

    #[test]
    fn terrain_vertex_to_bytes() {
        let vertex = TerrainVertex::new(
            [10.0, 5.0, 10.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0],
        );
        let bytes: &[u8] = bytemuck::bytes_of(&vertex);
        assert_eq!(bytes.len(), 32);
    }

    #[test]
    fn particle_vertex_to_bytes() {
        let vertex = ParticleVertex::new(
            [0.0, 5.0, 0.0],
            0xFFFFFFFF,
            1.0,
            0.5,
            0.0,
        );
        let bytes: &[u8] = bytemuck::bytes_of(&vertex);
        assert_eq!(bytes.len(), 32);
    }

    #[test]
    fn ui_vertex_to_bytes() {
        let vertex = UiVertex::new([100.0, 200.0], [0.0, 1.0], 0xFFFFFFFF);
        let bytes: &[u8] = bytemuck::bytes_of(&vertex);
        assert_eq!(bytes.len(), 20);
    }

    #[test]
    fn pbr_vertex_roundtrip() {
        let original = PbrVertex::new(
            [1.5, 2.5, 3.5],
            [0.0, 1.0, 0.0],
            [0.25, 0.75],
            [1.0, 0.0, 0.0, -1.0],
        );
        let bytes: &[u8] = bytemuck::bytes_of(&original);
        let recovered: &PbrVertex = bytemuck::from_bytes(bytes);

        assert_eq!(recovered.position, original.position);
        assert_eq!(recovered.normal, original.normal);
        assert_eq!(recovered.uv, original.uv);
        assert_eq!(recovered.tangent, original.tangent);
    }

    #[test]
    fn skinned_vertex_roundtrip() {
        let original = SkinnedVertex::new(
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5],
            [1.0, 0.0, 0.0, 1.0],
            [10, 20, 30, 40],
            [0.4, 0.3, 0.2, 0.1],
        );
        let bytes: &[u8] = bytemuck::bytes_of(&original);
        let recovered: &SkinnedVertex = bytemuck::from_bytes(bytes);

        assert_eq!(recovered.joints, original.joints);
        assert_eq!(recovered.weights, original.weights);
    }

    #[test]
    fn vertex_slice_to_bytes() {
        let vertices = [
            PbrVertex::new([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0], [1.0, 0.0, 0.0, 1.0]),
            PbrVertex::new([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0], [1.0, 0.0, 0.0, 1.0]),
            PbrVertex::new([0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0], [1.0, 0.0, 0.0, 1.0]),
        ];
        let bytes: &[u8] = bytemuck::cast_slice(&vertices);
        assert_eq!(bytes.len(), 48 * 3, "3 vertices = 144 bytes");
    }

    #[test]
    fn zeroed_pbr_vertex_is_valid() {
        let zeroed: PbrVertex = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.position, [0.0, 0.0, 0.0]);
        assert_eq!(zeroed.normal, [0.0, 0.0, 0.0]);
        assert_eq!(zeroed.uv, [0.0, 0.0]);
        assert_eq!(zeroed.tangent, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn zeroed_skinned_vertex_is_valid() {
        let zeroed: SkinnedVertex = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.joints, [0, 0, 0, 0]);
        assert_eq!(zeroed.weights, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn zeroed_particle_vertex_is_valid() {
        let zeroed: ParticleVertex = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.position, [0.0, 0.0, 0.0]);
        assert_eq!(zeroed.color, 0);
        assert_eq!(zeroed.size, 0.0);
        assert_eq!(zeroed.life, 0.0);
        assert_eq!(zeroed.rotation, 0.0);
        assert_eq!(zeroed._padding, 0);
    }

    #[test]
    fn zeroed_ui_vertex_is_valid() {
        let zeroed: UiVertex = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.position, [0.0, 0.0]);
        assert_eq!(zeroed.uv, [0.0, 0.0]);
        assert_eq!(zeroed.color, 0);
    }

    #[test]
    fn bytes_to_vertex_slice() {
        let original = [
            PbrVertex::new([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0], [1.0, 0.0, 0.0, 1.0]),
            PbrVertex::new([0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5], [1.0, 0.0, 0.0, 1.0]),
        ];
        let bytes: &[u8] = bytemuck::cast_slice(&original);
        let recovered: &[PbrVertex] = bytemuck::cast_slice(bytes);

        assert_eq!(recovered.len(), 2);
        assert_eq!(recovered[0].position, original[0].position);
        assert_eq!(recovered[1].position, original[1].position);
    }
}

// ============================================================================
// COLOR PACKING TESTS - Verify pack_color utilities
// ============================================================================

mod color_packing_tests {
    use super::*;

    #[test]
    fn pack_color_white() {
        let color = ParticleVertex::pack_color(255, 255, 255, 255);
        assert_eq!(color, 0xFFFFFFFF);
    }

    #[test]
    fn pack_color_black_opaque() {
        let color = ParticleVertex::pack_color(0, 0, 0, 255);
        assert_eq!(color, 0xFF000000);
    }

    #[test]
    fn pack_color_red() {
        let color = ParticleVertex::pack_color(255, 0, 0, 255);
        assert_eq!(color & 0xFF, 255, "R channel");
        assert_eq!((color >> 8) & 0xFF, 0, "G channel");
        assert_eq!((color >> 16) & 0xFF, 0, "B channel");
        assert_eq!((color >> 24) & 0xFF, 255, "A channel");
    }

    #[test]
    fn pack_color_green() {
        let color = ParticleVertex::pack_color(0, 255, 0, 255);
        assert_eq!(color & 0xFF, 0, "R channel");
        assert_eq!((color >> 8) & 0xFF, 255, "G channel");
        assert_eq!((color >> 16) & 0xFF, 0, "B channel");
        assert_eq!((color >> 24) & 0xFF, 255, "A channel");
    }

    #[test]
    fn pack_color_blue() {
        let color = ParticleVertex::pack_color(0, 0, 255, 255);
        assert_eq!(color & 0xFF, 0, "R channel");
        assert_eq!((color >> 8) & 0xFF, 0, "G channel");
        assert_eq!((color >> 16) & 0xFF, 255, "B channel");
        assert_eq!((color >> 24) & 0xFF, 255, "A channel");
    }

    #[test]
    fn pack_color_transparent() {
        let color = ParticleVertex::pack_color(255, 255, 255, 0);
        assert_eq!(color, 0x00FFFFFF);
    }

    #[test]
    fn pack_color_half_alpha() {
        let color = ParticleVertex::pack_color(255, 255, 255, 128);
        assert_eq!((color >> 24) & 0xFF, 128);
    }

    #[test]
    fn pack_color_roundtrip() {
        let r: u8 = 123;
        let g: u8 = 45;
        let b: u8 = 67;
        let a: u8 = 200;

        let color = ParticleVertex::pack_color(r, g, b, a);

        let unpacked_r = (color & 0xFF) as u8;
        let unpacked_g = ((color >> 8) & 0xFF) as u8;
        let unpacked_b = ((color >> 16) & 0xFF) as u8;
        let unpacked_a = ((color >> 24) & 0xFF) as u8;

        assert_eq!(unpacked_r, r);
        assert_eq!(unpacked_g, g);
        assert_eq!(unpacked_b, b);
        assert_eq!(unpacked_a, a);
    }

    #[test]
    fn pack_color_f32_white() {
        let color = ParticleVertex::pack_color_f32(1.0, 1.0, 1.0, 1.0);
        assert_eq!(color, 0xFFFFFFFF);
    }

    #[test]
    fn pack_color_f32_black() {
        let color = ParticleVertex::pack_color_f32(0.0, 0.0, 0.0, 1.0);
        assert_eq!(color, 0xFF000000);
    }

    #[test]
    fn pack_color_f32_clamps_negative() {
        let color = ParticleVertex::pack_color_f32(-1.0, -0.5, -10.0, 1.0);
        assert_eq!(color & 0xFF, 0, "negative R clamped to 0");
        assert_eq!((color >> 8) & 0xFF, 0, "negative G clamped to 0");
        assert_eq!((color >> 16) & 0xFF, 0, "negative B clamped to 0");
    }

    #[test]
    fn pack_color_f32_clamps_overflow() {
        let color = ParticleVertex::pack_color_f32(2.0, 1.5, 10.0, 1.0);
        assert_eq!(color & 0xFF, 255, "R > 1.0 clamped to 255");
        assert_eq!((color >> 8) & 0xFF, 255, "G > 1.0 clamped to 255");
        assert_eq!((color >> 16) & 0xFF, 255, "B > 1.0 clamped to 255");
    }

    #[test]
    fn pack_color_f32_half_values() {
        let color = ParticleVertex::pack_color_f32(0.5, 0.5, 0.5, 0.5);
        let r = (color & 0xFF) as u8;
        let g = ((color >> 8) & 0xFF) as u8;
        let b = ((color >> 16) & 0xFF) as u8;
        let a = ((color >> 24) & 0xFF) as u8;

        // 0.5 * 255 = 127.5, truncated to 127
        assert_eq!(r, 127);
        assert_eq!(g, 127);
        assert_eq!(b, 127);
        assert_eq!(a, 127);
    }

    #[test]
    fn ui_pack_color_same_as_particle() {
        let r = 100u8;
        let g = 150u8;
        let b = 200u8;
        let a = 255u8;

        let particle_color = ParticleVertex::pack_color(r, g, b, a);
        let ui_color = UiVertex::pack_color(r, g, b, a);

        assert_eq!(particle_color, ui_color, "UI and Particle should use same packing");
    }

    #[test]
    fn default_white_particle_is_white() {
        let particle = ParticleVertex::default_white();
        assert_eq!(particle.color, 0xFFFFFFFF);
        assert_eq!(particle.position, [0.0, 0.0, 0.0]);
        assert_eq!(particle.size, 1.0);
        assert_eq!(particle.life, 0.0);
        assert_eq!(particle.rotation, 0.0);
    }

    #[test]
    fn ui_white_vertex_is_white() {
        let vertex = UiVertex::white([50.0, 100.0], [0.5, 0.5]);
        assert_eq!(vertex.color, 0xFFFFFFFF);
        assert_eq!(vertex.position, [50.0, 100.0]);
        assert_eq!(vertex.uv, [0.5, 0.5]);
    }
}

// ============================================================================
// VERTEX CONSTRUCTION TESTS - Verify constructors work correctly
// ============================================================================

mod construction_tests {
    use super::*;

    #[test]
    fn pbr_new_sets_all_fields() {
        let v = PbrVertex::new(
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 0.0],
            [0.25, 0.75],
            [1.0, 0.0, 0.0, -1.0],
        );
        assert_eq!(v.position, [1.0, 2.0, 3.0]);
        assert_eq!(v.normal, [0.0, 1.0, 0.0]);
        assert_eq!(v.uv, [0.25, 0.75]);
        assert_eq!(v.tangent, [1.0, 0.0, 0.0, -1.0]);
    }

    #[test]
    fn pbr_with_default_tangent() {
        let v = PbrVertex::with_default_tangent(
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5],
        );
        assert_eq!(v.position, [1.0, 2.0, 3.0]);
        assert_eq!(v.tangent, [1.0, 0.0, 0.0, 1.0], "Default tangent should be +X with w=1");
    }

    #[test]
    fn skinned_new_sets_all_fields() {
        let v = SkinnedVertex::new(
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5],
            [1.0, 0.0, 0.0, 1.0],
            [0, 1, 2, 3],
            [0.5, 0.3, 0.15, 0.05],
        );
        assert_eq!(v.joints, [0, 1, 2, 3]);
        assert_eq!(v.weights, [0.5, 0.3, 0.15, 0.05]);
    }

    #[test]
    fn skinned_from_pbr_preserves_pbr_data() {
        let pbr = PbrVertex::new(
            [10.0, 20.0, 30.0],
            [0.0, 0.0, 1.0],
            [0.1, 0.9],
            [0.0, 1.0, 0.0, -1.0],
        );
        let skinned = SkinnedVertex::from_pbr(pbr, 42);

        assert_eq!(skinned.position, pbr.position);
        assert_eq!(skinned.normal, pbr.normal);
        assert_eq!(skinned.uv, pbr.uv);
        assert_eq!(skinned.tangent, pbr.tangent);
    }

    #[test]
    fn skinned_from_pbr_sets_single_bone() {
        let pbr = PbrVertex::new(
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
        );
        let skinned = SkinnedVertex::from_pbr(pbr, 7);

        assert_eq!(skinned.joints, [7, 0, 0, 0], "First joint should be bone index");
        assert_eq!(skinned.weights, [1.0, 0.0, 0.0, 0.0], "First weight should be 1.0");
    }

    #[test]
    fn terrain_new_sets_all_fields() {
        let v = TerrainVertex::new(
            [100.0, 50.0, 200.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5],
        );
        assert_eq!(v.position, [100.0, 50.0, 200.0]);
        assert_eq!(v.normal, [0.0, 1.0, 0.0]);
        assert_eq!(v.uv, [0.5, 0.5]);
    }

    #[test]
    fn terrain_flat_has_up_normal() {
        let v = TerrainVertex::flat([10.0, 0.0, 10.0], [0.5, 0.5]);
        assert_eq!(v.normal, [0.0, 1.0, 0.0], "Flat terrain should have up-facing normal");
    }

    #[test]
    fn particle_new_sets_all_fields() {
        let v = ParticleVertex::new(
            [1.0, 2.0, 3.0],
            0xFF00FF00,
            2.5,
            0.75,
            1.5708,
        );
        assert_eq!(v.position, [1.0, 2.0, 3.0]);
        assert_eq!(v.color, 0xFF00FF00);
        assert_eq!(v.size, 2.5);
        assert_eq!(v.life, 0.75);
        assert_eq!(v.rotation, 1.5708);
        assert_eq!(v._padding, 0);
    }

    #[test]
    fn ui_new_sets_all_fields() {
        let v = UiVertex::new([100.0, 200.0], [0.25, 0.75], 0xAABBCCDD);
        assert_eq!(v.position, [100.0, 200.0]);
        assert_eq!(v.uv, [0.25, 0.75]);
        assert_eq!(v.color, 0xAABBCCDD);
    }

    #[test]
    fn ui_quad_generates_four_vertices() {
        let quad = UiVertex::quad(0.0, 0.0, 100.0, 50.0, 0xFFFFFFFF);
        assert_eq!(quad.len(), 4);
    }

    #[test]
    fn ui_quad_positions_correct() {
        let x = 10.0;
        let y = 20.0;
        let w = 100.0;
        let h = 50.0;
        let quad = UiVertex::quad(x, y, w, h, 0xFFFFFFFF);

        // Counter-clockwise from bottom-left
        assert_eq!(quad[0].position, [x, y + h], "bottom-left");
        assert_eq!(quad[1].position, [x + w, y + h], "bottom-right");
        assert_eq!(quad[2].position, [x + w, y], "top-right");
        assert_eq!(quad[3].position, [x, y], "top-left");
    }

    #[test]
    fn ui_quad_uvs_correct() {
        let quad = UiVertex::quad(0.0, 0.0, 100.0, 50.0, 0xFFFFFFFF);

        // UVs should be standard 0-1 mapping
        assert_eq!(quad[0].uv, [0.0, 1.0], "bottom-left UV");
        assert_eq!(quad[1].uv, [1.0, 1.0], "bottom-right UV");
        assert_eq!(quad[2].uv, [1.0, 0.0], "top-right UV");
        assert_eq!(quad[3].uv, [0.0, 0.0], "top-left UV");
    }

    #[test]
    fn ui_quad_all_same_color() {
        let color = 0xAABBCCDD;
        let quad = UiVertex::quad(0.0, 0.0, 100.0, 50.0, color);

        for (i, v) in quad.iter().enumerate() {
            assert_eq!(v.color, color, "vertex {} should have same color", i);
        }
    }
}

// ============================================================================
// LAYOUT ID TESTS - Verify VertexLayoutId enum behavior
// ============================================================================

mod layout_id_tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn layout_id_equality() {
        assert_eq!(VertexLayoutId::Pbr, VertexLayoutId::Pbr);
        assert_eq!(VertexLayoutId::Custom(1), VertexLayoutId::Custom(1));
        assert_ne!(VertexLayoutId::Custom(1), VertexLayoutId::Custom(2));
        assert_ne!(VertexLayoutId::Pbr, VertexLayoutId::Skinned);
    }

    #[test]
    fn layout_id_hashable() {
        let mut set = HashSet::new();
        set.insert(VertexLayoutId::Pbr);
        set.insert(VertexLayoutId::Skinned);
        set.insert(VertexLayoutId::Custom(1));
        set.insert(VertexLayoutId::Custom(2));

        assert_eq!(set.len(), 4);
        assert!(set.contains(&VertexLayoutId::Pbr));
        assert!(set.contains(&VertexLayoutId::Custom(1)));
    }

    #[test]
    fn layout_id_clone() {
        let id = VertexLayoutId::Custom(42);
        let cloned = id;
        assert_eq!(id, cloned);
    }

    #[test]
    fn layout_id_debug_format() {
        let pbr = format!("{:?}", VertexLayoutId::Pbr);
        assert!(pbr.contains("Pbr"));

        let custom = format!("{:?}", VertexLayoutId::Custom(123));
        assert!(custom.contains("Custom"));
        assert!(custom.contains("123"));
    }
}
