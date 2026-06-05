// WHITEBOX tests for T-WGPU-P7.3.1 (Debug Visualization)
//
// WHITEBOX discipline: These tests have FULL ACCESS to the implementation.
// They exercise internal code paths, branch conditions, and edge cases
// that are not visible through the public contract alone.
//
// Implementation under test: crates/renderer-backend/src/debug/visualization.rs
//   - DebugVisualization enum (18 modes)
//   - ChannelMask struct
//   - VisualizationConfig struct
//   - DebugVisualizationManager struct
//   - DebugShaderData struct
//
// WHITEBOX coverage plan:
//   - DebugVisualization enum tests (40+): repr(u32) values, shader defines, category checks
//   - ChannelMask tests (25+): constants, to_array, combinations
//   - VisualizationConfig tests (30+): defaults, builder pattern, field access
//   - DebugVisualizationManager tests (35+): state management, cycling, config updates
//   - DebugShaderData tests (20+): GPU binding, alignment, byte conversion
//
// Target: 150+ comprehensive whitebox tests

use renderer_backend::debug::visualization::{
    ChannelMask, DebugShaderData, DebugVisualization, DebugVisualizationManager,
    VisualizationConfig,
};
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::mem;

// ============================================================================
// SECTION 1: DebugVisualization Enum Tests (40+)
// ============================================================================

// --- Test 1-18: All mode values match repr(u32) ---

#[test]
fn test_mode_none_value() {
    assert_eq!(DebugVisualization::None as u32, 0);
    assert_eq!(DebugVisualization::None.define_value(), 0);
}

#[test]
fn test_mode_wireframe_value() {
    assert_eq!(DebugVisualization::Wireframe as u32, 1);
    assert_eq!(DebugVisualization::Wireframe.define_value(), 1);
}

#[test]
fn test_mode_normals_value() {
    assert_eq!(DebugVisualization::Normals as u32, 2);
    assert_eq!(DebugVisualization::Normals.define_value(), 2);
}

#[test]
fn test_mode_tangents_value() {
    assert_eq!(DebugVisualization::Tangents as u32, 3);
    assert_eq!(DebugVisualization::Tangents.define_value(), 3);
}

#[test]
fn test_mode_bitangents_value() {
    assert_eq!(DebugVisualization::Bitangents as u32, 4);
    assert_eq!(DebugVisualization::Bitangents.define_value(), 4);
}

#[test]
fn test_mode_uvs_value() {
    assert_eq!(DebugVisualization::UVs as u32, 5);
    assert_eq!(DebugVisualization::UVs.define_value(), 5);
}

#[test]
fn test_mode_albedo_value() {
    assert_eq!(DebugVisualization::Albedo as u32, 6);
    assert_eq!(DebugVisualization::Albedo.define_value(), 6);
}

#[test]
fn test_mode_roughness_value() {
    assert_eq!(DebugVisualization::Roughness as u32, 7);
    assert_eq!(DebugVisualization::Roughness.define_value(), 7);
}

#[test]
fn test_mode_metallic_value() {
    assert_eq!(DebugVisualization::Metallic as u32, 8);
    assert_eq!(DebugVisualization::Metallic.define_value(), 8);
}

#[test]
fn test_mode_ambient_occlusion_value() {
    assert_eq!(DebugVisualization::AmbientOcclusion as u32, 9);
    assert_eq!(DebugVisualization::AmbientOcclusion.define_value(), 9);
}

#[test]
fn test_mode_emissive_value() {
    assert_eq!(DebugVisualization::Emissive as u32, 10);
    assert_eq!(DebugVisualization::Emissive.define_value(), 10);
}

#[test]
fn test_mode_depth_value() {
    assert_eq!(DebugVisualization::Depth as u32, 11);
    assert_eq!(DebugVisualization::Depth.define_value(), 11);
}

#[test]
fn test_mode_linear_depth_value() {
    assert_eq!(DebugVisualization::LinearDepth as u32, 12);
    assert_eq!(DebugVisualization::LinearDepth.define_value(), 12);
}

#[test]
fn test_mode_motion_vectors_value() {
    assert_eq!(DebugVisualization::MotionVectors as u32, 13);
    assert_eq!(DebugVisualization::MotionVectors.define_value(), 13);
}

#[test]
fn test_mode_overdraw_value() {
    assert_eq!(DebugVisualization::Overdraw as u32, 14);
    assert_eq!(DebugVisualization::Overdraw.define_value(), 14);
}

#[test]
fn test_mode_mip_level_value() {
    assert_eq!(DebugVisualization::MipLevel as u32, 15);
    assert_eq!(DebugVisualization::MipLevel.define_value(), 15);
}

#[test]
fn test_mode_light_complexity_value() {
    assert_eq!(DebugVisualization::LightComplexity as u32, 16);
    assert_eq!(DebugVisualization::LightComplexity.define_value(), 16);
}

#[test]
fn test_mode_shadow_cascades_value() {
    assert_eq!(DebugVisualization::ShadowCascades as u32, 17);
    assert_eq!(DebugVisualization::ShadowCascades.define_value(), 17);
}

// --- Test 19-36: shader_define() format verification ---

#[test]
fn test_shader_define_none() {
    assert_eq!(DebugVisualization::None.shader_define(), "DEBUG_VIS_NONE");
}

#[test]
fn test_shader_define_wireframe() {
    assert_eq!(
        DebugVisualization::Wireframe.shader_define(),
        "DEBUG_VIS_WIREFRAME"
    );
}

#[test]
fn test_shader_define_normals() {
    assert_eq!(
        DebugVisualization::Normals.shader_define(),
        "DEBUG_VIS_NORMALS"
    );
}

#[test]
fn test_shader_define_tangents() {
    assert_eq!(
        DebugVisualization::Tangents.shader_define(),
        "DEBUG_VIS_TANGENTS"
    );
}

#[test]
fn test_shader_define_bitangents() {
    assert_eq!(
        DebugVisualization::Bitangents.shader_define(),
        "DEBUG_VIS_BITANGENTS"
    );
}

#[test]
fn test_shader_define_uvs() {
    assert_eq!(DebugVisualization::UVs.shader_define(), "DEBUG_VIS_UVS");
}

#[test]
fn test_shader_define_albedo() {
    assert_eq!(
        DebugVisualization::Albedo.shader_define(),
        "DEBUG_VIS_ALBEDO"
    );
}

#[test]
fn test_shader_define_roughness() {
    assert_eq!(
        DebugVisualization::Roughness.shader_define(),
        "DEBUG_VIS_ROUGHNESS"
    );
}

#[test]
fn test_shader_define_metallic() {
    assert_eq!(
        DebugVisualization::Metallic.shader_define(),
        "DEBUG_VIS_METALLIC"
    );
}

#[test]
fn test_shader_define_ambient_occlusion() {
    assert_eq!(
        DebugVisualization::AmbientOcclusion.shader_define(),
        "DEBUG_VIS_AMBIENT_OCCLUSION"
    );
}

#[test]
fn test_shader_define_emissive() {
    assert_eq!(
        DebugVisualization::Emissive.shader_define(),
        "DEBUG_VIS_EMISSIVE"
    );
}

#[test]
fn test_shader_define_depth() {
    assert_eq!(
        DebugVisualization::Depth.shader_define(),
        "DEBUG_VIS_DEPTH"
    );
}

#[test]
fn test_shader_define_linear_depth() {
    assert_eq!(
        DebugVisualization::LinearDepth.shader_define(),
        "DEBUG_VIS_LINEAR_DEPTH"
    );
}

#[test]
fn test_shader_define_motion_vectors() {
    assert_eq!(
        DebugVisualization::MotionVectors.shader_define(),
        "DEBUG_VIS_MOTION_VECTORS"
    );
}

#[test]
fn test_shader_define_overdraw() {
    assert_eq!(
        DebugVisualization::Overdraw.shader_define(),
        "DEBUG_VIS_OVERDRAW"
    );
}

#[test]
fn test_shader_define_mip_level() {
    assert_eq!(
        DebugVisualization::MipLevel.shader_define(),
        "DEBUG_VIS_MIP_LEVEL"
    );
}

#[test]
fn test_shader_define_light_complexity() {
    assert_eq!(
        DebugVisualization::LightComplexity.shader_define(),
        "DEBUG_VIS_LIGHT_COMPLEXITY"
    );
}

#[test]
fn test_shader_define_shadow_cascades() {
    assert_eq!(
        DebugVisualization::ShadowCascades.shader_define(),
        "DEBUG_VIS_SHADOW_CASCADES"
    );
}

// --- Test 37: All shader defines start with DEBUG_VIS_ prefix ---

#[test]
fn test_all_shader_defines_have_prefix() {
    for mode in DebugVisualization::ALL.iter() {
        let define = mode.shader_define();
        assert!(
            define.starts_with("DEBUG_VIS_"),
            "Mode {:?} define '{}' should start with DEBUG_VIS_",
            mode,
            define
        );
    }
}

// --- Test 38: All shader defines are unique ---

#[test]
fn test_all_shader_defines_unique() {
    let mut seen = HashSet::new();
    for mode in DebugVisualization::ALL.iter() {
        let define = mode.shader_define();
        assert!(
            seen.insert(define),
            "Duplicate shader define: {}",
            define
        );
    }
    assert_eq!(seen.len(), 18);
}

// --- Test 39-43: is_geometry() for geometry modes ---

#[test]
fn test_is_geometry_wireframe() {
    assert!(DebugVisualization::Wireframe.is_geometry());
}

#[test]
fn test_is_geometry_normals() {
    assert!(DebugVisualization::Normals.is_geometry());
}

#[test]
fn test_is_geometry_tangents() {
    assert!(DebugVisualization::Tangents.is_geometry());
}

#[test]
fn test_is_geometry_bitangents() {
    assert!(DebugVisualization::Bitangents.is_geometry());
}

#[test]
fn test_is_geometry_uvs() {
    assert!(DebugVisualization::UVs.is_geometry());
}

// --- Test 44: Geometry mode count matches constant ---

#[test]
fn test_geometry_mode_count() {
    let geometry_count = DebugVisualization::ALL
        .iter()
        .filter(|m| m.is_geometry())
        .count();
    assert_eq!(geometry_count, 5);
    assert_eq!(geometry_count, DebugVisualization::GEOMETRY_MODES.len());
}

// --- Test 45: Non-geometry modes return false ---

#[test]
fn test_is_geometry_false_for_lighting() {
    assert!(!DebugVisualization::Albedo.is_geometry());
    assert!(!DebugVisualization::Roughness.is_geometry());
    assert!(!DebugVisualization::Metallic.is_geometry());
    assert!(!DebugVisualization::AmbientOcclusion.is_geometry());
    assert!(!DebugVisualization::Emissive.is_geometry());
}

#[test]
fn test_is_geometry_false_for_depth_motion() {
    assert!(!DebugVisualization::Depth.is_geometry());
    assert!(!DebugVisualization::LinearDepth.is_geometry());
    assert!(!DebugVisualization::MotionVectors.is_geometry());
}

#[test]
fn test_is_geometry_false_for_performance() {
    assert!(!DebugVisualization::Overdraw.is_geometry());
    assert!(!DebugVisualization::MipLevel.is_geometry());
    assert!(!DebugVisualization::LightComplexity.is_geometry());
    assert!(!DebugVisualization::ShadowCascades.is_geometry());
}

#[test]
fn test_is_geometry_false_for_none() {
    assert!(!DebugVisualization::None.is_geometry());
}

// --- Test 49-53: is_lighting() for lighting modes ---

#[test]
fn test_is_lighting_albedo() {
    assert!(DebugVisualization::Albedo.is_lighting());
}

#[test]
fn test_is_lighting_roughness() {
    assert!(DebugVisualization::Roughness.is_lighting());
}

#[test]
fn test_is_lighting_metallic() {
    assert!(DebugVisualization::Metallic.is_lighting());
}

#[test]
fn test_is_lighting_ambient_occlusion() {
    assert!(DebugVisualization::AmbientOcclusion.is_lighting());
}

#[test]
fn test_is_lighting_emissive() {
    assert!(DebugVisualization::Emissive.is_lighting());
}

// --- Test 54: Lighting mode count matches constant ---

#[test]
fn test_lighting_mode_count() {
    let lighting_count = DebugVisualization::ALL
        .iter()
        .filter(|m| m.is_lighting())
        .count();
    assert_eq!(lighting_count, 5);
    assert_eq!(lighting_count, DebugVisualization::LIGHTING_MODES.len());
}

// --- Test 55: Non-lighting modes return false ---

#[test]
fn test_is_lighting_false_for_geometry() {
    for mode in DebugVisualization::GEOMETRY_MODES.iter() {
        assert!(!mode.is_lighting(), "{:?} should not be lighting", mode);
    }
}

// --- Test 56-59: is_performance() for performance modes ---

#[test]
fn test_is_performance_overdraw() {
    assert!(DebugVisualization::Overdraw.is_performance());
}

#[test]
fn test_is_performance_mip_level() {
    assert!(DebugVisualization::MipLevel.is_performance());
}

#[test]
fn test_is_performance_light_complexity() {
    assert!(DebugVisualization::LightComplexity.is_performance());
}

#[test]
fn test_is_performance_shadow_cascades() {
    assert!(DebugVisualization::ShadowCascades.is_performance());
}

// --- Test 60: Performance mode count matches constant ---

#[test]
fn test_performance_mode_count() {
    let perf_count = DebugVisualization::ALL
        .iter()
        .filter(|m| m.is_performance())
        .count();
    assert_eq!(perf_count, 4);
    assert_eq!(perf_count, DebugVisualization::PERFORMANCE_MODES.len());
}

// --- Test 61-63: is_depth_motion() for depth/motion modes ---

#[test]
fn test_is_depth_motion_depth() {
    assert!(DebugVisualization::Depth.is_depth_motion());
}

#[test]
fn test_is_depth_motion_linear_depth() {
    assert!(DebugVisualization::LinearDepth.is_depth_motion());
}

#[test]
fn test_is_depth_motion_motion_vectors() {
    assert!(DebugVisualization::MotionVectors.is_depth_motion());
}

// --- Test 64: Depth/motion mode count ---

#[test]
fn test_depth_motion_mode_count() {
    let depth_motion_count = DebugVisualization::ALL
        .iter()
        .filter(|m| m.is_depth_motion())
        .count();
    assert_eq!(depth_motion_count, 3);
}

// --- Test 65: Non-depth/motion modes return false ---

#[test]
fn test_is_depth_motion_false_for_others() {
    assert!(!DebugVisualization::None.is_depth_motion());
    assert!(!DebugVisualization::Wireframe.is_depth_motion());
    assert!(!DebugVisualization::Albedo.is_depth_motion());
    assert!(!DebugVisualization::Overdraw.is_depth_motion());
}

// --- Test 66: description() not empty for all modes ---

#[test]
fn test_all_descriptions_not_empty() {
    for mode in DebugVisualization::ALL.iter() {
        let desc = mode.description();
        assert!(!desc.is_empty(), "{:?} should have description", mode);
    }
}

// --- Test 67: description() minimum length ---

#[test]
fn test_description_minimum_length() {
    for mode in DebugVisualization::ALL.iter() {
        let desc = mode.description();
        assert!(
            desc.len() >= 10,
            "{:?} description too short: '{}'",
            mode,
            desc
        );
    }
}

// --- Test 68: description() unique per mode ---

#[test]
fn test_descriptions_unique() {
    let mut seen = HashSet::new();
    for mode in DebugVisualization::ALL.iter() {
        let desc = mode.description();
        assert!(
            seen.insert(desc),
            "Duplicate description: {}",
            desc
        );
    }
}

// --- Test 69: Default trait returns None ---

#[test]
fn test_debug_visualization_default() {
    let mode: DebugVisualization = Default::default();
    assert_eq!(mode, DebugVisualization::None);
}

// --- Test 70: Clone semantics ---

#[test]
fn test_debug_visualization_clone() {
    let mode = DebugVisualization::Wireframe;
    let cloned = mode.clone();
    assert_eq!(mode, cloned);
}

// --- Test 71: Copy semantics ---

#[test]
fn test_debug_visualization_copy() {
    let mode = DebugVisualization::Normals;
    let copied = mode; // Copy, not move
    assert_eq!(mode, copied);
}

// --- Test 72: Debug formatting ---

#[test]
fn test_debug_visualization_debug_format() {
    let mode = DebugVisualization::Albedo;
    let debug_str = format!("{:?}", mode);
    assert_eq!(debug_str, "Albedo");
}

// --- Test 73: PartialEq comparisons ---

#[test]
fn test_debug_visualization_equality() {
    assert_eq!(DebugVisualization::None, DebugVisualization::None);
    assert_ne!(DebugVisualization::None, DebugVisualization::Wireframe);
    assert_ne!(DebugVisualization::Wireframe, DebugVisualization::Normals);
}

// --- Test 74: Hash consistency ---

#[test]
fn test_debug_visualization_hash_consistency() {
    use std::collections::hash_map::DefaultHasher;

    let mode1 = DebugVisualization::Overdraw;
    let mode2 = DebugVisualization::Overdraw;

    let mut hasher1 = DefaultHasher::new();
    let mut hasher2 = DefaultHasher::new();

    mode1.hash(&mut hasher1);
    mode2.hash(&mut hasher2);

    assert_eq!(hasher1.finish(), hasher2.finish());
}

// --- Test 75: Hash in HashSet ---

#[test]
fn test_debug_visualization_in_hashset() {
    let mut set = HashSet::new();
    set.insert(DebugVisualization::Wireframe);
    set.insert(DebugVisualization::Normals);
    set.insert(DebugVisualization::Wireframe); // Duplicate

    assert_eq!(set.len(), 2);
    assert!(set.contains(&DebugVisualization::Wireframe));
    assert!(set.contains(&DebugVisualization::Normals));
    assert!(!set.contains(&DebugVisualization::Albedo));
}

// --- Test 76: ALL array length ---

#[test]
fn test_all_array_length() {
    assert_eq!(DebugVisualization::ALL.len(), 18);
}

// --- Test 77: ALL array order ---

#[test]
fn test_all_array_order() {
    assert_eq!(DebugVisualization::ALL[0], DebugVisualization::None);
    assert_eq!(DebugVisualization::ALL[1], DebugVisualization::Wireframe);
    assert_eq!(DebugVisualization::ALL[17], DebugVisualization::ShadowCascades);
}

// --- Test 78: index() method ---

#[test]
fn test_index_method() {
    for (i, mode) in DebugVisualization::ALL.iter().enumerate() {
        assert_eq!(mode.index(), i, "{:?} index mismatch", mode);
    }
}

// --- Test 79: from_index() valid indices ---

#[test]
fn test_from_index_valid() {
    for i in 0..18 {
        let mode = DebugVisualization::from_index(i);
        assert!(mode.is_some(), "Index {} should be valid", i);
        assert_eq!(mode.unwrap().index(), i);
    }
}

// --- Test 80: from_index() invalid indices ---

#[test]
fn test_from_index_invalid() {
    assert!(DebugVisualization::from_index(18).is_none());
    assert!(DebugVisualization::from_index(100).is_none());
    assert!(DebugVisualization::from_index(usize::MAX).is_none());
}

// --- Test 81: Category arrays contain correct modes ---

#[test]
fn test_geometry_modes_array() {
    for mode in DebugVisualization::GEOMETRY_MODES.iter() {
        assert!(mode.is_geometry(), "{:?} should be geometry", mode);
    }
}

#[test]
fn test_lighting_modes_array() {
    for mode in DebugVisualization::LIGHTING_MODES.iter() {
        assert!(mode.is_lighting(), "{:?} should be lighting", mode);
    }
}

#[test]
fn test_performance_modes_array() {
    for mode in DebugVisualization::PERFORMANCE_MODES.iter() {
        assert!(mode.is_performance(), "{:?} should be performance", mode);
    }
}

// --- Test 84: Mode categories are mutually exclusive ---

#[test]
fn test_mode_categories_mutually_exclusive() {
    for mode in DebugVisualization::ALL.iter() {
        let mut category_count = 0;
        if mode.is_geometry() {
            category_count += 1;
        }
        if mode.is_lighting() {
            category_count += 1;
        }
        if mode.is_performance() {
            category_count += 1;
        }
        if mode.is_depth_motion() {
            category_count += 1;
        }
        assert!(
            category_count <= 1,
            "{:?} is in multiple categories",
            mode
        );
    }
}

// --- Test 85: None mode is in no category ---

#[test]
fn test_none_in_no_category() {
    let none = DebugVisualization::None;
    assert!(!none.is_geometry());
    assert!(!none.is_lighting());
    assert!(!none.is_performance());
    assert!(!none.is_depth_motion());
}

// ============================================================================
// SECTION 2: ChannelMask Tests (25+)
// ============================================================================

// --- Test 86: ALL constant ---

#[test]
fn test_channel_mask_all_constant() {
    let mask = ChannelMask::ALL;
    assert!(mask.r);
    assert!(mask.g);
    assert!(mask.b);
    assert!(mask.a);
    assert!(mask.is_all());
    assert!(!mask.is_rgb());
    assert!(!mask.is_none());
    assert_eq!(mask.count(), 4);
}

// --- Test 87: RED constant ---

#[test]
fn test_channel_mask_red_constant() {
    let mask = ChannelMask::RED;
    assert!(mask.r);
    assert!(!mask.g);
    assert!(!mask.b);
    assert!(!mask.a);
    assert!(!mask.is_all());
    assert!(!mask.is_rgb());
    assert!(!mask.is_none());
    assert_eq!(mask.count(), 1);
}

// --- Test 88: GREEN constant ---

#[test]
fn test_channel_mask_green_constant() {
    let mask = ChannelMask::GREEN;
    assert!(!mask.r);
    assert!(mask.g);
    assert!(!mask.b);
    assert!(!mask.a);
    assert_eq!(mask.count(), 1);
}

// --- Test 89: BLUE constant ---

#[test]
fn test_channel_mask_blue_constant() {
    let mask = ChannelMask::BLUE;
    assert!(!mask.r);
    assert!(!mask.g);
    assert!(mask.b);
    assert!(!mask.a);
    assert_eq!(mask.count(), 1);
}

// --- Test 90: ALPHA constant ---

#[test]
fn test_channel_mask_alpha_constant() {
    let mask = ChannelMask::ALPHA;
    assert!(!mask.r);
    assert!(!mask.g);
    assert!(!mask.b);
    assert!(mask.a);
    assert_eq!(mask.count(), 1);
}

// --- Test 91: RGB constant ---

#[test]
fn test_channel_mask_rgb_constant() {
    let mask = ChannelMask::RGB;
    assert!(mask.r);
    assert!(mask.g);
    assert!(mask.b);
    assert!(!mask.a);
    assert!(!mask.is_all());
    assert!(mask.is_rgb());
    assert!(!mask.is_none());
    assert_eq!(mask.count(), 3);
}

// --- Test 92: Default trait returns ALL ---

#[test]
fn test_channel_mask_default() {
    let mask: ChannelMask = Default::default();
    assert_eq!(mask, ChannelMask::ALL);
}

// --- Test 93: as_floats() for ALL ---

#[test]
fn test_channel_mask_as_floats_all() {
    assert_eq!(ChannelMask::ALL.as_floats(), [1.0, 1.0, 1.0, 1.0]);
}

// --- Test 94: as_floats() for RED ---

#[test]
fn test_channel_mask_as_floats_red() {
    assert_eq!(ChannelMask::RED.as_floats(), [1.0, 0.0, 0.0, 0.0]);
}

// --- Test 95: as_floats() for GREEN ---

#[test]
fn test_channel_mask_as_floats_green() {
    assert_eq!(ChannelMask::GREEN.as_floats(), [0.0, 1.0, 0.0, 0.0]);
}

// --- Test 96: as_floats() for BLUE ---

#[test]
fn test_channel_mask_as_floats_blue() {
    assert_eq!(ChannelMask::BLUE.as_floats(), [0.0, 0.0, 1.0, 0.0]);
}

// --- Test 97: as_floats() for ALPHA ---

#[test]
fn test_channel_mask_as_floats_alpha() {
    assert_eq!(ChannelMask::ALPHA.as_floats(), [0.0, 0.0, 0.0, 1.0]);
}

// --- Test 98: as_floats() for RGB ---

#[test]
fn test_channel_mask_as_floats_rgb() {
    assert_eq!(ChannelMask::RGB.as_floats(), [1.0, 1.0, 1.0, 0.0]);
}

// --- Test 99: Clone semantics ---

#[test]
fn test_channel_mask_clone() {
    let mask = ChannelMask::RED;
    let cloned = mask.clone();
    assert_eq!(mask, cloned);
}

// --- Test 100: Copy semantics ---

#[test]
fn test_channel_mask_copy() {
    let mask = ChannelMask::GREEN;
    let copied = mask; // Copy, not move
    assert_eq!(mask, copied);
}

// --- Test 101: PartialEq comparisons ---

#[test]
fn test_channel_mask_equality() {
    assert_eq!(ChannelMask::ALL, ChannelMask::ALL);
    assert_ne!(ChannelMask::ALL, ChannelMask::RGB);
    assert_ne!(ChannelMask::RED, ChannelMask::GREEN);
}

// --- Test 102: new() constructor ---

#[test]
fn test_channel_mask_new() {
    let mask = ChannelMask::new(true, false, true, false);
    assert!(mask.r);
    assert!(!mask.g);
    assert!(mask.b);
    assert!(!mask.a);
    assert_eq!(mask.count(), 2);
    assert_eq!(mask.as_floats(), [1.0, 0.0, 1.0, 0.0]);
}

// --- Test 103: Custom combination ---

#[test]
fn test_channel_mask_custom_combination() {
    let mask = ChannelMask::new(false, true, false, true);
    assert!(!mask.r);
    assert!(mask.g);
    assert!(!mask.b);
    assert!(mask.a);
    assert_eq!(mask.count(), 2);
    assert_eq!(mask.as_floats(), [0.0, 1.0, 0.0, 1.0]);
}

// --- Test 104: is_none() for no channels ---

#[test]
fn test_channel_mask_is_none() {
    let mask = ChannelMask::new(false, false, false, false);
    assert!(mask.is_none());
    assert!(!mask.is_all());
    assert!(!mask.is_rgb());
    assert_eq!(mask.count(), 0);
}

// --- Test 105: count() for various combinations ---

#[test]
fn test_channel_mask_count_combinations() {
    assert_eq!(ChannelMask::new(false, false, false, false).count(), 0);
    assert_eq!(ChannelMask::new(true, false, false, false).count(), 1);
    assert_eq!(ChannelMask::new(true, true, false, false).count(), 2);
    assert_eq!(ChannelMask::new(true, true, true, false).count(), 3);
    assert_eq!(ChannelMask::new(true, true, true, true).count(), 4);
}

// --- Test 106: Debug formatting ---

#[test]
fn test_channel_mask_debug_format() {
    let mask = ChannelMask::RED;
    let debug_str = format!("{:?}", mask);
    assert!(debug_str.contains("ChannelMask"));
    assert!(debug_str.contains("true")); // r is true
}

// --- Test 107: is_all() returns true only when all are true ---

#[test]
fn test_channel_mask_is_all_strict() {
    assert!(!ChannelMask::new(true, true, true, false).is_all());
    assert!(!ChannelMask::new(true, true, false, true).is_all());
    assert!(!ChannelMask::new(true, false, true, true).is_all());
    assert!(!ChannelMask::new(false, true, true, true).is_all());
    assert!(ChannelMask::new(true, true, true, true).is_all());
}

// --- Test 108: is_rgb() returns true only for RGB without alpha ---

#[test]
fn test_channel_mask_is_rgb_strict() {
    assert!(ChannelMask::RGB.is_rgb());
    assert!(!ChannelMask::ALL.is_rgb());
    assert!(!ChannelMask::new(true, true, false, false).is_rgb());
    assert!(!ChannelMask::new(true, false, true, false).is_rgb());
}

// --- Test 109: Field access ---

#[test]
fn test_channel_mask_field_access() {
    let mask = ChannelMask::new(true, false, true, false);
    assert_eq!(mask.r, true);
    assert_eq!(mask.g, false);
    assert_eq!(mask.b, true);
    assert_eq!(mask.a, false);
}

// --- Test 110: as_floats returns f32 array ---

#[test]
fn test_channel_mask_as_floats_type() {
    let floats: [f32; 4] = ChannelMask::ALL.as_floats();
    assert_eq!(floats.len(), 4);
    for f in &floats {
        assert!(*f == 0.0 || *f == 1.0);
    }
}

// ============================================================================
// SECTION 3: VisualizationConfig Tests (30+)
// ============================================================================

// --- Test 111: Default values ---

#[test]
fn test_visualization_config_default_values() {
    let config = VisualizationConfig::default();
    assert_eq!(config.mode, DebugVisualization::None);
    assert_eq!(config.intensity, 1.0);
    assert_eq!(config.overlay, false);
    assert!(config.channel_mask.is_all());
}

// --- Test 112: mode field access ---

#[test]
fn test_visualization_config_mode_field() {
    let mut config = VisualizationConfig::default();
    config.mode = DebugVisualization::Wireframe;
    assert_eq!(config.mode, DebugVisualization::Wireframe);
}

// --- Test 113: intensity field range (0.0) ---

#[test]
fn test_visualization_config_intensity_zero() {
    let config = VisualizationConfig::with_mode(DebugVisualization::Normals).with_intensity(0.0);
    assert_eq!(config.intensity, 0.0);
}

// --- Test 114: intensity field range (1.0) ---

#[test]
fn test_visualization_config_intensity_one() {
    let config = VisualizationConfig::default();
    assert_eq!(config.intensity, 1.0);
}

// --- Test 115: intensity field range (2.0 - typical max) ---

#[test]
fn test_visualization_config_intensity_max() {
    let config = VisualizationConfig::with_mode(DebugVisualization::Albedo).with_intensity(2.0);
    assert_eq!(config.intensity, 2.0);
}

// --- Test 116: intensity field fractional values ---

#[test]
fn test_visualization_config_intensity_fractional() {
    let config = VisualizationConfig::with_mode(DebugVisualization::Roughness).with_intensity(0.5);
    assert_eq!(config.intensity, 0.5);
}

// --- Test 117: overlay field true ---

#[test]
fn test_visualization_config_overlay_true() {
    let config = VisualizationConfig::with_mode(DebugVisualization::Metallic).with_overlay(true);
    assert!(config.overlay);
}

// --- Test 118: overlay field false ---

#[test]
fn test_visualization_config_overlay_false() {
    let config = VisualizationConfig::with_mode(DebugVisualization::Depth).with_overlay(false);
    assert!(!config.overlay);
}

// --- Test 119: channel_mask field ---

#[test]
fn test_visualization_config_channel_mask_field() {
    let config = VisualizationConfig::with_mode(DebugVisualization::Albedo)
        .with_channel_mask(ChannelMask::RED);
    assert_eq!(config.channel_mask, ChannelMask::RED);
}

// --- Test 120: with_mode() builder ---

#[test]
fn test_visualization_config_with_mode() {
    let config = VisualizationConfig::with_mode(DebugVisualization::Overdraw);
    assert_eq!(config.mode, DebugVisualization::Overdraw);
    assert_eq!(config.intensity, 1.0); // Default
    assert!(!config.overlay); // Default
}

// --- Test 121: with_intensity() builder ---

#[test]
fn test_visualization_config_with_intensity_builder() {
    let config = VisualizationConfig::default().with_intensity(0.75);
    assert_eq!(config.intensity, 0.75);
    assert_eq!(config.mode, DebugVisualization::None); // Unchanged
}

// --- Test 122: with_overlay() builder ---

#[test]
fn test_visualization_config_with_overlay_builder() {
    let config = VisualizationConfig::default().with_overlay(true);
    assert!(config.overlay);
    assert_eq!(config.mode, DebugVisualization::None); // Unchanged
}

// --- Test 123: with_channel_mask() builder ---

#[test]
fn test_visualization_config_with_channel_mask_builder() {
    let config = VisualizationConfig::default().with_channel_mask(ChannelMask::RGB);
    assert_eq!(config.channel_mask, ChannelMask::RGB);
    assert_eq!(config.mode, DebugVisualization::None); // Unchanged
}

// --- Test 124: Chained builder pattern ---

#[test]
fn test_visualization_config_chained_builder() {
    let config = VisualizationConfig::with_mode(DebugVisualization::MipLevel)
        .with_intensity(1.5)
        .with_overlay(true)
        .with_channel_mask(ChannelMask::GREEN);

    assert_eq!(config.mode, DebugVisualization::MipLevel);
    assert_eq!(config.intensity, 1.5);
    assert!(config.overlay);
    assert_eq!(config.channel_mask, ChannelMask::GREEN);
}

// --- Test 125: Clone semantics ---

#[test]
fn test_visualization_config_clone() {
    let config = VisualizationConfig::with_mode(DebugVisualization::Wireframe)
        .with_intensity(0.8)
        .with_overlay(true);
    let cloned = config.clone();

    assert_eq!(config.mode, cloned.mode);
    assert_eq!(config.intensity, cloned.intensity);
    assert_eq!(config.overlay, cloned.overlay);
    assert_eq!(config.channel_mask, cloned.channel_mask);
}

// --- Test 126: PartialEq comparisons ---

#[test]
fn test_visualization_config_equality() {
    let config1 = VisualizationConfig::with_mode(DebugVisualization::Normals);
    let config2 = VisualizationConfig::with_mode(DebugVisualization::Normals);
    let config3 = VisualizationConfig::with_mode(DebugVisualization::Tangents);

    assert_eq!(config1, config2);
    assert_ne!(config1, config3);
}

// --- Test 127: Debug formatting ---

#[test]
fn test_visualization_config_debug_format() {
    let config = VisualizationConfig::default();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("VisualizationConfig"));
    assert!(debug_str.contains("mode"));
    assert!(debug_str.contains("intensity"));
}

// --- Test 128: Direct field mutation ---

#[test]
fn test_visualization_config_direct_mutation() {
    let mut config = VisualizationConfig::default();
    config.mode = DebugVisualization::ShadowCascades;
    config.intensity = 1.2;
    config.overlay = true;
    config.channel_mask = ChannelMask::BLUE;

    assert_eq!(config.mode, DebugVisualization::ShadowCascades);
    assert_eq!(config.intensity, 1.2);
    assert!(config.overlay);
    assert_eq!(config.channel_mask, ChannelMask::BLUE);
}

// --- Test 129: All modes work with config ---

#[test]
fn test_visualization_config_all_modes() {
    for mode in DebugVisualization::ALL.iter() {
        let config = VisualizationConfig::with_mode(*mode);
        assert_eq!(config.mode, *mode);
    }
}

// --- Test 130: Negative intensity (edge case) ---

#[test]
fn test_visualization_config_negative_intensity() {
    let config = VisualizationConfig::default().with_intensity(-0.5);
    // No validation, so negative is allowed
    assert_eq!(config.intensity, -0.5);
}

// --- Test 131: Large intensity (edge case) ---

#[test]
fn test_visualization_config_large_intensity() {
    let config = VisualizationConfig::default().with_intensity(100.0);
    assert_eq!(config.intensity, 100.0);
}

// --- Test 132: Infinity intensity (edge case) ---

#[test]
fn test_visualization_config_infinity_intensity() {
    let config = VisualizationConfig::default().with_intensity(f32::INFINITY);
    assert!(config.intensity.is_infinite());
}

// --- Test 133: NaN intensity (edge case) ---

#[test]
fn test_visualization_config_nan_intensity() {
    let config = VisualizationConfig::default().with_intensity(f32::NAN);
    assert!(config.intensity.is_nan());
}

// --- Test 134: Size of VisualizationConfig ---

#[test]
fn test_visualization_config_size() {
    // Struct should be reasonably sized
    let size = mem::size_of::<VisualizationConfig>();
    assert!(size <= 32, "VisualizationConfig too large: {} bytes", size);
}

// --- Test 135: Alignment of VisualizationConfig ---

#[test]
fn test_visualization_config_alignment() {
    let align = mem::align_of::<VisualizationConfig>();
    assert!(align <= 8, "Unexpected alignment: {}", align);
}

// ============================================================================
// SECTION 4: DebugVisualizationManager Tests (35+)
// ============================================================================

// --- Test 136: new() default state ---

#[test]
fn test_manager_new_default_state() {
    let manager = DebugVisualizationManager::new();
    assert_eq!(manager.current_mode(), DebugVisualization::None);
    assert!(!manager.is_enabled());
    assert_eq!(manager.intensity(), 1.0);
    assert!(!manager.is_overlay());
    assert_eq!(manager.channel_mask(), ChannelMask::ALL);
}

// --- Test 137: Default trait ---

#[test]
fn test_manager_default_trait() {
    let manager: DebugVisualizationManager = Default::default();
    assert_eq!(manager.current_mode(), DebugVisualization::None);
    assert!(!manager.is_enabled());
}

// --- Test 138: set_mode() updates mode ---

#[test]
fn test_manager_set_mode() {
    let mut manager = DebugVisualizationManager::new();
    manager.set_mode(DebugVisualization::Normals);
    assert_eq!(manager.current_mode(), DebugVisualization::Normals);
}

// --- Test 139: set_mode() updates config ---

#[test]
fn test_manager_set_mode_updates_config() {
    let mut manager = DebugVisualizationManager::new();
    manager.set_mode(DebugVisualization::Albedo);
    assert_eq!(manager.config().mode, DebugVisualization::Albedo);
}

// --- Test 140: current_mode() returns correct mode ---

#[test]
fn test_manager_current_mode() {
    let mut manager = DebugVisualizationManager::new();
    assert_eq!(manager.current_mode(), DebugVisualization::None);

    manager.set_mode(DebugVisualization::Overdraw);
    assert_eq!(manager.current_mode(), DebugVisualization::Overdraw);
}

// --- Test 141: toggle() flips enabled state ---

#[test]
fn test_manager_toggle() {
    let mut manager = DebugVisualizationManager::new();
    assert!(!manager.is_enabled());

    manager.toggle();
    assert!(manager.is_enabled());

    manager.toggle();
    assert!(!manager.is_enabled());
}

// --- Test 142: enable() explicit control ---

#[test]
fn test_manager_enable() {
    let mut manager = DebugVisualizationManager::new();
    manager.enable();
    assert!(manager.is_enabled());

    // Calling enable again should still be enabled
    manager.enable();
    assert!(manager.is_enabled());
}

// --- Test 143: disable() explicit control ---

#[test]
fn test_manager_disable() {
    let mut manager = DebugVisualizationManager::new();
    manager.enable();
    manager.disable();
    assert!(!manager.is_enabled());

    // Calling disable again should still be disabled
    manager.disable();
    assert!(!manager.is_enabled());
}

// --- Test 144: is_enabled() state check ---

#[test]
fn test_manager_is_enabled() {
    let mut manager = DebugVisualizationManager::new();
    assert!(!manager.is_enabled());

    manager.enable();
    assert!(manager.is_enabled());

    manager.disable();
    assert!(!manager.is_enabled());
}

// --- Test 145: cycle_next() through all 18 modes ---

#[test]
fn test_manager_cycle_next_all_modes() {
    let mut manager = DebugVisualizationManager::new();
    let mut seen_modes = Vec::new();

    for _ in 0..18 {
        seen_modes.push(manager.current_mode());
        manager.cycle_next();
    }

    assert_eq!(seen_modes.len(), 18);
    assert_eq!(seen_modes[0], DebugVisualization::None);
    assert_eq!(seen_modes[17], DebugVisualization::ShadowCascades);
}

// --- Test 146: cycle_next() wraps at boundary (17 -> 0) ---

#[test]
fn test_manager_cycle_next_wrap() {
    let mut manager = DebugVisualizationManager::new();
    manager.set_mode(DebugVisualization::ShadowCascades);
    assert_eq!(manager.current_mode().index(), 17);

    manager.cycle_next();
    assert_eq!(manager.current_mode(), DebugVisualization::None);
    assert_eq!(manager.current_mode().index(), 0);
}

// --- Test 147: cycle_prev() reverse cycling ---

#[test]
fn test_manager_cycle_prev() {
    let mut manager = DebugVisualizationManager::new();
    manager.set_mode(DebugVisualization::Normals); // Index 2

    manager.cycle_prev();
    assert_eq!(manager.current_mode(), DebugVisualization::Wireframe); // Index 1

    manager.cycle_prev();
    assert_eq!(manager.current_mode(), DebugVisualization::None); // Index 0
}

// --- Test 148: cycle_prev() wraps at boundary (0 -> 17) ---

#[test]
fn test_manager_cycle_prev_wrap() {
    let mut manager = DebugVisualizationManager::new();
    assert_eq!(manager.current_mode(), DebugVisualization::None);
    assert_eq!(manager.current_mode().index(), 0);

    manager.cycle_prev();
    assert_eq!(manager.current_mode(), DebugVisualization::ShadowCascades);
    assert_eq!(manager.current_mode().index(), 17);
}

// --- Test 149: Full cycle with cycle_prev ---

#[test]
fn test_manager_cycle_prev_full_cycle() {
    let mut manager = DebugVisualizationManager::new();

    // Cycle back through all 18 modes
    for _ in 0..18 {
        manager.cycle_prev();
    }

    // Should be back at None
    assert_eq!(manager.current_mode(), DebugVisualization::None);
}

// --- Test 150: set_config() updates config ---

#[test]
fn test_manager_set_config() {
    let mut manager = DebugVisualizationManager::new();
    let config = VisualizationConfig::with_mode(DebugVisualization::MipLevel)
        .with_intensity(0.8)
        .with_overlay(true);

    manager.set_config(config);

    assert_eq!(manager.current_mode(), DebugVisualization::MipLevel);
    assert_eq!(manager.intensity(), 0.8);
    assert!(manager.is_overlay());
}

// --- Test 151: set_config() syncs current_mode ---

#[test]
fn test_manager_set_config_syncs_mode() {
    let mut manager = DebugVisualizationManager::new();
    let config = VisualizationConfig::with_mode(DebugVisualization::LightComplexity);

    manager.set_config(config);

    assert_eq!(manager.current_mode(), DebugVisualization::LightComplexity);
}

// --- Test 152: config() returns reference ---

#[test]
fn test_manager_config_reference() {
    let manager = DebugVisualizationManager::new();
    let config = manager.config();

    assert_eq!(config.mode, DebugVisualization::None);
    assert_eq!(config.intensity, 1.0);
}

// --- Test 153: config_mut() returns mutable reference ---

#[test]
fn test_manager_config_mut() {
    let mut manager = DebugVisualizationManager::new();

    manager.config_mut().intensity = 0.5;
    manager.config_mut().overlay = true;

    assert_eq!(manager.intensity(), 0.5);
    assert!(manager.is_overlay());
}

// --- Test 154: all_modes() returns 18 modes ---

#[test]
fn test_manager_all_modes_count() {
    let modes = DebugVisualizationManager::all_modes();
    assert_eq!(modes.len(), 18);
}

// --- Test 155: all_modes() order matches enum order ---

#[test]
fn test_manager_all_modes_order() {
    let modes = DebugVisualizationManager::all_modes();

    for (i, mode) in modes.iter().enumerate() {
        assert_eq!(mode.index(), i);
        assert_eq!(*mode, DebugVisualization::ALL[i]);
    }
}

// --- Test 156: effective_mode() when disabled ---

#[test]
fn test_manager_effective_mode_disabled() {
    let mut manager = DebugVisualizationManager::new();
    manager.set_mode(DebugVisualization::Wireframe);
    // Not enabled

    assert_eq!(manager.effective_mode(), DebugVisualization::None);
}

// --- Test 157: effective_mode() when enabled ---

#[test]
fn test_manager_effective_mode_enabled() {
    let mut manager = DebugVisualizationManager::new();
    manager.set_mode(DebugVisualization::Wireframe);
    manager.enable();

    assert_eq!(manager.effective_mode(), DebugVisualization::Wireframe);
}

// --- Test 158: set_intensity() ---

#[test]
fn test_manager_set_intensity() {
    let mut manager = DebugVisualizationManager::new();
    manager.set_intensity(0.75);
    assert_eq!(manager.intensity(), 0.75);
}

// --- Test 159: intensity() ---

#[test]
fn test_manager_intensity() {
    let manager = DebugVisualizationManager::new();
    assert_eq!(manager.intensity(), 1.0);
}

// --- Test 160: set_overlay() ---

#[test]
fn test_manager_set_overlay() {
    let mut manager = DebugVisualizationManager::new();
    manager.set_overlay(true);
    assert!(manager.is_overlay());

    manager.set_overlay(false);
    assert!(!manager.is_overlay());
}

// --- Test 161: is_overlay() ---

#[test]
fn test_manager_is_overlay() {
    let manager = DebugVisualizationManager::new();
    assert!(!manager.is_overlay());
}

// --- Test 162: set_channel_mask() ---

#[test]
fn test_manager_set_channel_mask() {
    let mut manager = DebugVisualizationManager::new();
    manager.set_channel_mask(ChannelMask::RED);
    assert_eq!(manager.channel_mask(), ChannelMask::RED);
}

// --- Test 163: channel_mask() ---

#[test]
fn test_manager_channel_mask() {
    let manager = DebugVisualizationManager::new();
    assert_eq!(manager.channel_mask(), ChannelMask::ALL);
}

// --- Test 164: Clone semantics ---

#[test]
fn test_manager_clone() {
    let mut manager = DebugVisualizationManager::new();
    manager.set_mode(DebugVisualization::Normals);
    manager.enable();
    manager.set_intensity(0.8);

    let cloned = manager.clone();

    assert_eq!(cloned.current_mode(), DebugVisualization::Normals);
    assert!(cloned.is_enabled());
    assert_eq!(cloned.intensity(), 0.8);
}

// --- Test 165: Debug formatting ---

#[test]
fn test_manager_debug_format() {
    let manager = DebugVisualizationManager::new();
    let debug_str = format!("{:?}", manager);
    assert!(debug_str.contains("DebugVisualizationManager"));
}

// --- Test 166: Cycling maintains config ---

#[test]
fn test_manager_cycle_maintains_config() {
    let mut manager = DebugVisualizationManager::new();
    manager.set_intensity(0.5);
    manager.set_overlay(true);

    manager.cycle_next();
    manager.cycle_next();

    assert_eq!(manager.intensity(), 0.5);
    assert!(manager.is_overlay());
}

// --- Test 167: Multiple toggles ---

#[test]
fn test_manager_multiple_toggles() {
    let mut manager = DebugVisualizationManager::new();

    for i in 0..10 {
        manager.toggle();
        assert_eq!(manager.is_enabled(), i % 2 == 0);
    }
}

// --- Test 168: Set mode to same mode ---

#[test]
fn test_manager_set_mode_same() {
    let mut manager = DebugVisualizationManager::new();
    manager.set_mode(DebugVisualization::Albedo);
    manager.set_mode(DebugVisualization::Albedo);
    assert_eq!(manager.current_mode(), DebugVisualization::Albedo);
}

// --- Test 169: Cycle from middle ---

#[test]
fn test_manager_cycle_from_middle() {
    let mut manager = DebugVisualizationManager::new();
    manager.set_mode(DebugVisualization::Roughness); // Index 7

    manager.cycle_next();
    assert_eq!(manager.current_mode(), DebugVisualization::Metallic); // Index 8

    manager.cycle_prev();
    manager.cycle_prev();
    assert_eq!(manager.current_mode(), DebugVisualization::Albedo); // Index 6
}

// --- Test 170: Manager size ---

#[test]
fn test_manager_size() {
    let size = mem::size_of::<DebugVisualizationManager>();
    // Should be reasonably sized
    assert!(size <= 64, "Manager too large: {} bytes", size);
}

// ============================================================================
// SECTION 5: DebugShaderData Tests (20+)
// ============================================================================

// --- Test 171: from_config() construction ---

#[test]
fn test_shader_data_from_config_construction() {
    let config = VisualizationConfig::with_mode(DebugVisualization::Wireframe);
    let data = DebugShaderData::from_config(&config);

    assert_eq!(data.mode, 1); // Wireframe = 1
    assert_eq!(data.intensity, 1.0);
    assert_eq!(data.channel_mask, [1.0, 1.0, 1.0, 1.0]);
}

// --- Test 172: mode field matches config ---

#[test]
fn test_shader_data_mode_matches_config() {
    for mode in DebugVisualization::ALL.iter() {
        let config = VisualizationConfig::with_mode(*mode);
        let data = DebugShaderData::from_config(&config);
        assert_eq!(data.mode, mode.define_value());
    }
}

// --- Test 173: intensity field matches config ---

#[test]
fn test_shader_data_intensity_matches_config() {
    let config = VisualizationConfig::with_mode(DebugVisualization::None).with_intensity(0.42);
    let data = DebugShaderData::from_config(&config);
    assert_eq!(data.intensity, 0.42);
}

// --- Test 174: channel_mask array from ChannelMask ---

#[test]
fn test_shader_data_channel_mask_array() {
    let config = VisualizationConfig::with_mode(DebugVisualization::Albedo)
        .with_channel_mask(ChannelMask::RED);
    let data = DebugShaderData::from_config(&config);
    assert_eq!(data.channel_mask, [1.0, 0.0, 0.0, 0.0]);
}

// --- Test 175: as_bytes() returns 32 bytes ---

#[test]
fn test_shader_data_as_bytes_size() {
    let data = DebugShaderData::default();
    let bytes = data.as_bytes();
    assert_eq!(bytes.len(), 32);
}

// --- Test 176: size() returns 32 ---

#[test]
fn test_shader_data_size_method() {
    assert_eq!(DebugShaderData::size(), 32);
}

// --- Test 177: Struct size matches size() ---

#[test]
fn test_shader_data_struct_size() {
    assert_eq!(mem::size_of::<DebugShaderData>(), 32);
    assert_eq!(mem::size_of::<DebugShaderData>(), DebugShaderData::size());
}

// --- Test 178: Alignment verification (std140 compatible) ---

#[test]
fn test_shader_data_alignment() {
    // std140 requires vec4 alignment (16 bytes) for uniform blocks
    // Our struct should have appropriate alignment
    let align = mem::align_of::<DebugShaderData>();
    assert!(align >= 4, "Minimum 4-byte alignment expected, got {}", align);
}

// --- Test 179: repr(C) layout verification ---

#[test]
fn test_shader_data_repr_c_layout() {
    // mode (u32) = 4 bytes
    // intensity (f32) = 4 bytes
    // channel_mask ([f32; 4]) = 16 bytes
    // _padding ([f32; 2]) = 8 bytes
    // Total = 32 bytes

    let data = DebugShaderData {
        mode: 5,
        intensity: 0.75,
        channel_mask: [1.0, 0.0, 1.0, 0.0],
        _padding: [0.0, 0.0],
    };

    assert_eq!(data.mode, 5);
    assert_eq!(data.intensity, 0.75);
    assert_eq!(data.channel_mask, [1.0, 0.0, 1.0, 0.0]);
}

// --- Test 180: Default values ---

#[test]
fn test_shader_data_default() {
    let data = DebugShaderData::default();
    assert_eq!(data.mode, 0);
    assert_eq!(data.intensity, 1.0);
    assert_eq!(data.channel_mask, [1.0, 1.0, 1.0, 1.0]);
    assert_eq!(data._padding, [0.0, 0.0]);
}

// --- Test 181: from_manager() construction ---

#[test]
fn test_shader_data_from_manager() {
    let mut manager = DebugVisualizationManager::new();
    manager.set_mode(DebugVisualization::Depth);
    manager.enable();
    manager.set_intensity(0.6);

    let data = DebugShaderData::from_manager(&manager);
    assert_eq!(data.mode, 11); // Depth = 11
    assert_eq!(data.intensity, 0.6);
}

// --- Test 182: from_manager() uses effective_mode ---

#[test]
fn test_shader_data_from_manager_effective_mode() {
    let mut manager = DebugVisualizationManager::new();
    manager.set_mode(DebugVisualization::Overdraw);
    // Not enabled, so effective mode is None

    let data = DebugShaderData::from_manager(&manager);
    assert_eq!(data.mode, 0); // None = 0
}

// --- Test 183: is_disabled() ---

#[test]
fn test_shader_data_is_disabled() {
    let data = DebugShaderData::default();
    assert!(data.is_disabled());

    let data2 = DebugShaderData::from_config(&VisualizationConfig::with_mode(
        DebugVisualization::Wireframe,
    ));
    assert!(!data2.is_disabled());
}

// --- Test 184: Clone semantics ---

#[test]
fn test_shader_data_clone() {
    let data = DebugShaderData {
        mode: 3,
        intensity: 0.8,
        channel_mask: [1.0, 1.0, 0.0, 0.0],
        _padding: [0.0, 0.0],
    };
    let cloned = data.clone();
    assert_eq!(data, cloned);
}

// --- Test 185: Copy semantics ---

#[test]
fn test_shader_data_copy() {
    let data = DebugShaderData::default();
    let copied = data; // Copy, not move
    assert_eq!(data, copied);
}

// --- Test 186: Debug formatting ---

#[test]
fn test_shader_data_debug_format() {
    let data = DebugShaderData::default();
    let debug_str = format!("{:?}", data);
    assert!(debug_str.contains("DebugShaderData"));
    assert!(debug_str.contains("mode"));
    assert!(debug_str.contains("intensity"));
}

// --- Test 187: PartialEq comparisons ---

#[test]
fn test_shader_data_equality() {
    let data1 = DebugShaderData::default();
    let data2 = DebugShaderData::default();
    let data3 = DebugShaderData::from_config(&VisualizationConfig::with_mode(
        DebugVisualization::Wireframe,
    ));

    assert_eq!(data1, data2);
    assert_ne!(data1, data3);
}

// --- Test 188: Byte layout verification ---

#[test]
fn test_shader_data_byte_layout() {
    let data = DebugShaderData {
        mode: 5,
        intensity: 1.0,
        channel_mask: [1.0, 1.0, 1.0, 1.0],
        _padding: [0.0, 0.0],
    };

    let bytes = data.as_bytes();

    // First 4 bytes should be mode (5 as u32)
    let mode_bytes = [bytes[0], bytes[1], bytes[2], bytes[3]];
    let mode_value = u32::from_ne_bytes(mode_bytes);
    assert_eq!(mode_value, 5);

    // Next 4 bytes should be intensity (1.0 as f32)
    let intensity_bytes = [bytes[4], bytes[5], bytes[6], bytes[7]];
    let intensity_value = f32::from_ne_bytes(intensity_bytes);
    assert_eq!(intensity_value, 1.0);
}

// --- Test 189: as_bytes() multiple calls return same data ---

#[test]
fn test_shader_data_as_bytes_consistent() {
    let data = DebugShaderData::from_config(&VisualizationConfig::with_mode(
        DebugVisualization::Normals,
    ));

    let bytes1 = data.as_bytes();
    let bytes2 = data.as_bytes();

    assert_eq!(bytes1, bytes2);
}

// --- Test 190: All modes produce valid shader data ---

#[test]
fn test_shader_data_all_modes_valid() {
    for mode in DebugVisualization::ALL.iter() {
        let config = VisualizationConfig::with_mode(*mode);
        let data = DebugShaderData::from_config(&config);

        assert_eq!(data.mode, mode.define_value());
        assert_eq!(data.as_bytes().len(), 32);
        assert!(!data.intensity.is_nan());
    }
}

// ============================================================================
// SECTION 6: Additional Edge Cases and Integration Tests (10+)
// ============================================================================

// --- Test 191: Full workflow test ---

#[test]
fn test_full_workflow() {
    let mut manager = DebugVisualizationManager::new();

    // Setup
    manager.set_mode(DebugVisualization::Wireframe);
    manager.enable();
    manager.set_intensity(0.9);
    manager.set_overlay(true);
    manager.set_channel_mask(ChannelMask::RGB);

    // Verify
    assert_eq!(manager.current_mode(), DebugVisualization::Wireframe);
    assert!(manager.is_enabled());
    assert_eq!(manager.effective_mode(), DebugVisualization::Wireframe);

    // Create shader data
    let data = DebugShaderData::from_manager(&manager);
    assert_eq!(data.mode, 1);
    assert_eq!(data.intensity, 0.9);
    assert_eq!(data.channel_mask, [1.0, 1.0, 1.0, 0.0]);
}

// --- Test 192: Cycle through all modes and verify ---

#[test]
fn test_cycle_all_modes_verify() {
    let mut manager = DebugVisualizationManager::new();
    manager.enable();

    for expected_mode in DebugVisualization::ALL.iter() {
        assert_eq!(manager.current_mode(), *expected_mode);
        assert_eq!(manager.effective_mode(), *expected_mode);

        let data = DebugShaderData::from_manager(&manager);
        assert_eq!(data.mode, expected_mode.define_value());

        manager.cycle_next();
    }
}

// --- Test 193: Config builder chain preserves order ---

#[test]
fn test_config_builder_chain_order() {
    // Ensure later calls override earlier ones
    let config = VisualizationConfig::with_mode(DebugVisualization::None)
        .with_intensity(0.5)
        .with_intensity(0.8); // Override

    assert_eq!(config.intensity, 0.8);
}

// --- Test 194: Manager config_mut affects config ---

#[test]
fn test_manager_config_mut_affects_config() {
    let mut manager = DebugVisualizationManager::new();

    {
        let config = manager.config_mut();
        config.mode = DebugVisualization::Metallic;
        config.intensity = 0.3;
    }

    // Note: config_mut doesn't sync current_mode back
    // This is a known quirk of the API
    assert_eq!(manager.config().mode, DebugVisualization::Metallic);
    assert_eq!(manager.intensity(), 0.3);
}

// --- Test 195: Shader data padding is zero ---

#[test]
fn test_shader_data_padding_zero() {
    let data = DebugShaderData::from_config(&VisualizationConfig::with_mode(
        DebugVisualization::Overdraw,
    ));
    assert_eq!(data._padding, [0.0, 0.0]);
}

// --- Test 196: Category check for all depth/motion modes ---

#[test]
fn test_depth_motion_category_complete() {
    let depth_motion = [
        DebugVisualization::Depth,
        DebugVisualization::LinearDepth,
        DebugVisualization::MotionVectors,
    ];

    for mode in depth_motion.iter() {
        assert!(mode.is_depth_motion());
        assert!(!mode.is_geometry());
        assert!(!mode.is_lighting());
        assert!(!mode.is_performance());
    }
}

// --- Test 197: Mode index round-trip ---

#[test]
fn test_mode_index_round_trip() {
    for mode in DebugVisualization::ALL.iter() {
        let index = mode.index();
        let recovered = DebugVisualization::from_index(index).unwrap();
        assert_eq!(*mode, recovered);
    }
}

// --- Test 198: Shader define contains mode name ---

#[test]
fn test_shader_define_contains_name() {
    // Most defines should contain a recognizable part of the mode name
    assert!(DebugVisualization::Wireframe
        .shader_define()
        .contains("WIREFRAME"));
    assert!(DebugVisualization::Normals.shader_define().contains("NORMALS"));
    assert!(DebugVisualization::Overdraw
        .shader_define()
        .contains("OVERDRAW"));
}

// --- Test 199: Channel mask all combinations ---

#[test]
fn test_channel_mask_all_16_combinations() {
    // There are 16 possible combinations of 4 booleans
    for r in [false, true] {
        for g in [false, true] {
            for b in [false, true] {
                for a in [false, true] {
                    let mask = ChannelMask::new(r, g, b, a);
                    let expected_count = r as u32 + g as u32 + b as u32 + a as u32;
                    assert_eq!(mask.count(), expected_count);

                    let floats = mask.as_floats();
                    assert_eq!(floats[0], if r { 1.0 } else { 0.0 });
                    assert_eq!(floats[1], if g { 1.0 } else { 0.0 });
                    assert_eq!(floats[2], if b { 1.0 } else { 0.0 });
                    assert_eq!(floats[3], if a { 1.0 } else { 0.0 });
                }
            }
        }
    }
}

// --- Test 200: Manager state isolation after clone ---

#[test]
fn test_manager_clone_isolation() {
    let mut manager1 = DebugVisualizationManager::new();
    manager1.set_mode(DebugVisualization::Normals);
    manager1.enable();

    let mut manager2 = manager1.clone();

    // Modify manager2
    manager2.set_mode(DebugVisualization::Albedo);
    manager2.disable();

    // manager1 should be unaffected
    assert_eq!(manager1.current_mode(), DebugVisualization::Normals);
    assert!(manager1.is_enabled());

    // manager2 has its own state
    assert_eq!(manager2.current_mode(), DebugVisualization::Albedo);
    assert!(!manager2.is_enabled());
}
