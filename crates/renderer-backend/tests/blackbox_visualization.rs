// Blackbox contract tests for T-WGPU-P7.3.1: Debug Visualization
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::debug::*` -- no internal fields,
// no private methods, no implementation details.
//
// Contract:
//   DebugVisualization enum provides 18 visualization modes for GPU debugging.
//   ChannelMask controls RGBA channel selection.
//   VisualizationConfig holds complete visualization settings.
//   DebugVisualizationManager manages runtime mode switching and state.
//   DebugShaderData provides GPU-bindable uniform data.
//
// Scenarios:
//   API Contract Tests (25+)
//   Visualization Mode Scenarios (30+)
//   Manager Lifecycle (20+)
//   Shader Integration (15+)
//   Channel Mask Operations (15+)
//
// Total: 105+ blackbox tests

use renderer_backend::debug::{
    ChannelMask, DebugShaderData, DebugVisualization, DebugVisualizationManager,
    VisualizationConfig,
};

// =============================================================================
// SECTION 1: API Contract Tests (25+ tests)
// =============================================================================

// Test 1: DebugVisualization::None is accessible
#[test]
fn test_debug_visualization_none_accessible() {
    let mode = DebugVisualization::None;
    assert_eq!(mode.define_value(), 0);
}

// Test 2: DebugVisualization::Wireframe is accessible
#[test]
fn test_debug_visualization_wireframe_accessible() {
    let mode = DebugVisualization::Wireframe;
    assert_eq!(mode.define_value(), 1);
}

// Test 3: DebugVisualization::Normals is accessible
#[test]
fn test_debug_visualization_normals_accessible() {
    let mode = DebugVisualization::Normals;
    assert_eq!(mode.define_value(), 2);
}

// Test 4: DebugVisualization::Tangents is accessible
#[test]
fn test_debug_visualization_tangents_accessible() {
    let mode = DebugVisualization::Tangents;
    assert_eq!(mode.define_value(), 3);
}

// Test 5: DebugVisualization::Bitangents is accessible
#[test]
fn test_debug_visualization_bitangents_accessible() {
    let mode = DebugVisualization::Bitangents;
    assert_eq!(mode.define_value(), 4);
}

// Test 6: DebugVisualization::UVs is accessible
#[test]
fn test_debug_visualization_uvs_accessible() {
    let mode = DebugVisualization::UVs;
    assert_eq!(mode.define_value(), 5);
}

// Test 7: DebugVisualization::Albedo is accessible
#[test]
fn test_debug_visualization_albedo_accessible() {
    let mode = DebugVisualization::Albedo;
    assert_eq!(mode.define_value(), 6);
}

// Test 8: DebugVisualization::Roughness is accessible
#[test]
fn test_debug_visualization_roughness_accessible() {
    let mode = DebugVisualization::Roughness;
    assert_eq!(mode.define_value(), 7);
}

// Test 9: DebugVisualization::Metallic is accessible
#[test]
fn test_debug_visualization_metallic_accessible() {
    let mode = DebugVisualization::Metallic;
    assert_eq!(mode.define_value(), 8);
}

// Test 10: DebugVisualization::AmbientOcclusion is accessible
#[test]
fn test_debug_visualization_ambient_occlusion_accessible() {
    let mode = DebugVisualization::AmbientOcclusion;
    assert_eq!(mode.define_value(), 9);
}

// Test 11: DebugVisualization::Emissive is accessible
#[test]
fn test_debug_visualization_emissive_accessible() {
    let mode = DebugVisualization::Emissive;
    assert_eq!(mode.define_value(), 10);
}

// Test 12: DebugVisualization::Depth is accessible
#[test]
fn test_debug_visualization_depth_accessible() {
    let mode = DebugVisualization::Depth;
    assert_eq!(mode.define_value(), 11);
}

// Test 13: DebugVisualization::LinearDepth is accessible
#[test]
fn test_debug_visualization_linear_depth_accessible() {
    let mode = DebugVisualization::LinearDepth;
    assert_eq!(mode.define_value(), 12);
}

// Test 14: DebugVisualization::MotionVectors is accessible
#[test]
fn test_debug_visualization_motion_vectors_accessible() {
    let mode = DebugVisualization::MotionVectors;
    assert_eq!(mode.define_value(), 13);
}

// Test 15: DebugVisualization::Overdraw is accessible
#[test]
fn test_debug_visualization_overdraw_accessible() {
    let mode = DebugVisualization::Overdraw;
    assert_eq!(mode.define_value(), 14);
}

// Test 16: DebugVisualization::MipLevel is accessible
#[test]
fn test_debug_visualization_mip_level_accessible() {
    let mode = DebugVisualization::MipLevel;
    assert_eq!(mode.define_value(), 15);
}

// Test 17: DebugVisualization::LightComplexity is accessible
#[test]
fn test_debug_visualization_light_complexity_accessible() {
    let mode = DebugVisualization::LightComplexity;
    assert_eq!(mode.define_value(), 16);
}

// Test 18: DebugVisualization::ShadowCascades is accessible
#[test]
fn test_debug_visualization_shadow_cascades_accessible() {
    let mode = DebugVisualization::ShadowCascades;
    assert_eq!(mode.define_value(), 17);
}

// Test 19: ChannelMask::ALL constant is accessible
#[test]
fn test_channel_mask_all_accessible() {
    let mask = ChannelMask::ALL;
    assert!(mask.is_all());
}

// Test 20: ChannelMask::RED constant is accessible
#[test]
fn test_channel_mask_red_accessible() {
    let mask = ChannelMask::RED;
    assert!(mask.r);
    assert!(!mask.g);
    assert!(!mask.b);
    assert!(!mask.a);
}

// Test 21: ChannelMask::GREEN constant is accessible
#[test]
fn test_channel_mask_green_accessible() {
    let mask = ChannelMask::GREEN;
    assert!(!mask.r);
    assert!(mask.g);
    assert!(!mask.b);
    assert!(!mask.a);
}

// Test 22: ChannelMask::BLUE constant is accessible
#[test]
fn test_channel_mask_blue_accessible() {
    let mask = ChannelMask::BLUE;
    assert!(!mask.r);
    assert!(!mask.g);
    assert!(mask.b);
    assert!(!mask.a);
}

// Test 23: ChannelMask::ALPHA constant is accessible
#[test]
fn test_channel_mask_alpha_accessible() {
    let mask = ChannelMask::ALPHA;
    assert!(!mask.r);
    assert!(!mask.g);
    assert!(!mask.b);
    assert!(mask.a);
}

// Test 24: ChannelMask::RGB constant is accessible
#[test]
fn test_channel_mask_rgb_accessible() {
    let mask = ChannelMask::RGB;
    assert!(mask.r);
    assert!(mask.g);
    assert!(mask.b);
    assert!(!mask.a);
}

// Test 25: VisualizationConfig default construction
#[test]
fn test_visualization_config_default_construction() {
    let config = VisualizationConfig::default();
    assert_eq!(config.mode, DebugVisualization::None);
    assert_eq!(config.intensity, 1.0);
    assert!(!config.overlay);
}

// Test 26: DebugVisualizationManager public construction
#[test]
fn test_manager_public_construction() {
    let manager = DebugVisualizationManager::new();
    assert_eq!(manager.current_mode(), DebugVisualization::None);
    assert!(!manager.is_enabled());
}

// Test 27: DebugShaderData public construction
#[test]
fn test_shader_data_public_construction() {
    let data = DebugShaderData::default();
    assert_eq!(data.mode, 0);
    assert_eq!(data.intensity, 1.0);
}

// Test 28: DebugVisualization::ALL array is accessible
#[test]
fn test_debug_visualization_all_array_accessible() {
    let all = DebugVisualization::ALL;
    assert_eq!(all.len(), 18);
}

// Test 29: DebugVisualization derives Default
#[test]
fn test_debug_visualization_derives_default() {
    let mode: DebugVisualization = Default::default();
    assert_eq!(mode, DebugVisualization::None);
}

// =============================================================================
// SECTION 2: Visualization Mode Scenarios (30+ tests)
// =============================================================================

// Test 30: Geometry debugging workflow - enable wireframe
#[test]
fn test_geometry_workflow_wireframe() {
    let mode = DebugVisualization::Wireframe;
    assert!(mode.is_geometry());
    assert!(!mode.is_lighting());
    assert!(!mode.is_performance());
    assert!(!mode.is_depth_motion());
    assert_eq!(mode.shader_define(), "DEBUG_VIS_WIREFRAME");
}

// Test 31: Geometry debugging workflow - enable normals
#[test]
fn test_geometry_workflow_normals() {
    let mode = DebugVisualization::Normals;
    assert!(mode.is_geometry());
    assert_eq!(mode.shader_define(), "DEBUG_VIS_NORMALS");
}

// Test 32: Geometry debugging workflow - enable tangents
#[test]
fn test_geometry_workflow_tangents() {
    let mode = DebugVisualization::Tangents;
    assert!(mode.is_geometry());
    assert_eq!(mode.shader_define(), "DEBUG_VIS_TANGENTS");
}

// Test 33: Geometry debugging workflow - enable bitangents
#[test]
fn test_geometry_workflow_bitangents() {
    let mode = DebugVisualization::Bitangents;
    assert!(mode.is_geometry());
    assert_eq!(mode.shader_define(), "DEBUG_VIS_BITANGENTS");
}

// Test 34: Geometry debugging workflow - enable UVs
#[test]
fn test_geometry_workflow_uvs() {
    let mode = DebugVisualization::UVs;
    assert!(mode.is_geometry());
    assert_eq!(mode.shader_define(), "DEBUG_VIS_UVS");
}

// Test 35: Lighting debugging workflow - enable albedo
#[test]
fn test_lighting_workflow_albedo() {
    let mode = DebugVisualization::Albedo;
    assert!(mode.is_lighting());
    assert!(!mode.is_geometry());
    assert!(!mode.is_performance());
    assert_eq!(mode.shader_define(), "DEBUG_VIS_ALBEDO");
}

// Test 36: Lighting debugging workflow - enable roughness
#[test]
fn test_lighting_workflow_roughness() {
    let mode = DebugVisualization::Roughness;
    assert!(mode.is_lighting());
    assert_eq!(mode.shader_define(), "DEBUG_VIS_ROUGHNESS");
}

// Test 37: Lighting debugging workflow - enable metallic
#[test]
fn test_lighting_workflow_metallic() {
    let mode = DebugVisualization::Metallic;
    assert!(mode.is_lighting());
    assert_eq!(mode.shader_define(), "DEBUG_VIS_METALLIC");
}

// Test 38: Lighting debugging workflow - enable ambient occlusion
#[test]
fn test_lighting_workflow_ambient_occlusion() {
    let mode = DebugVisualization::AmbientOcclusion;
    assert!(mode.is_lighting());
    assert_eq!(mode.shader_define(), "DEBUG_VIS_AMBIENT_OCCLUSION");
}

// Test 39: Lighting debugging workflow - enable emissive
#[test]
fn test_lighting_workflow_emissive() {
    let mode = DebugVisualization::Emissive;
    assert!(mode.is_lighting());
    assert_eq!(mode.shader_define(), "DEBUG_VIS_EMISSIVE");
}

// Test 40: Performance analysis workflow - enable overdraw
#[test]
fn test_performance_workflow_overdraw() {
    let mode = DebugVisualization::Overdraw;
    assert!(mode.is_performance());
    assert!(!mode.is_geometry());
    assert!(!mode.is_lighting());
    assert_eq!(mode.shader_define(), "DEBUG_VIS_OVERDRAW");
}

// Test 41: Performance analysis workflow - enable mip level
#[test]
fn test_performance_workflow_mip_level() {
    let mode = DebugVisualization::MipLevel;
    assert!(mode.is_performance());
    assert_eq!(mode.shader_define(), "DEBUG_VIS_MIP_LEVEL");
}

// Test 42: Performance analysis workflow - enable light complexity
#[test]
fn test_performance_workflow_light_complexity() {
    let mode = DebugVisualization::LightComplexity;
    assert!(mode.is_performance());
    assert_eq!(mode.shader_define(), "DEBUG_VIS_LIGHT_COMPLEXITY");
}

// Test 43: Performance analysis workflow - enable shadow cascades
#[test]
fn test_performance_workflow_shadow_cascades() {
    let mode = DebugVisualization::ShadowCascades;
    assert!(mode.is_performance());
    assert_eq!(mode.shader_define(), "DEBUG_VIS_SHADOW_CASCADES");
}

// Test 44: Depth visualization workflow - enable depth
#[test]
fn test_depth_workflow_depth() {
    let mode = DebugVisualization::Depth;
    assert!(mode.is_depth_motion());
    assert!(!mode.is_geometry());
    assert!(!mode.is_lighting());
    assert!(!mode.is_performance());
    assert_eq!(mode.shader_define(), "DEBUG_VIS_DEPTH");
}

// Test 45: Depth visualization workflow - switch to linear depth
#[test]
fn test_depth_workflow_linear_depth() {
    let mode = DebugVisualization::LinearDepth;
    assert!(mode.is_depth_motion());
    assert_eq!(mode.shader_define(), "DEBUG_VIS_LINEAR_DEPTH");
}

// Test 46: Motion vectors visualization workflow
#[test]
fn test_motion_workflow_motion_vectors() {
    let mode = DebugVisualization::MotionVectors;
    assert!(mode.is_depth_motion());
    assert_eq!(mode.shader_define(), "DEBUG_VIS_MOTION_VECTORS");
}

// Test 47: Cycle through all geometry modes
#[test]
fn test_cycle_geometry_modes() {
    let geometry_modes = DebugVisualization::GEOMETRY_MODES;
    assert_eq!(geometry_modes.len(), 5);
    for mode in geometry_modes.iter() {
        assert!(mode.is_geometry());
    }
}

// Test 48: Cycle through all lighting modes
#[test]
fn test_cycle_lighting_modes() {
    let lighting_modes = DebugVisualization::LIGHTING_MODES;
    assert_eq!(lighting_modes.len(), 5);
    for mode in lighting_modes.iter() {
        assert!(mode.is_lighting());
    }
}

// Test 49: Cycle through all performance modes
#[test]
fn test_cycle_performance_modes() {
    let perf_modes = DebugVisualization::PERFORMANCE_MODES;
    assert_eq!(perf_modes.len(), 4);
    for mode in perf_modes.iter() {
        assert!(mode.is_performance());
    }
}

// Test 50: None mode is not in any category
#[test]
fn test_none_mode_no_category() {
    let mode = DebugVisualization::None;
    assert!(!mode.is_geometry());
    assert!(!mode.is_lighting());
    assert!(!mode.is_performance());
    assert!(!mode.is_depth_motion());
}

// Test 51: Get shader data for geometry mode
#[test]
fn test_shader_data_geometry_mode() {
    let config = VisualizationConfig::with_mode(DebugVisualization::Wireframe);
    let data = DebugShaderData::from_config(&config);
    assert_eq!(data.mode, 1);
}

// Test 52: Get shader data for lighting mode
#[test]
fn test_shader_data_lighting_mode() {
    let config = VisualizationConfig::with_mode(DebugVisualization::Albedo);
    let data = DebugShaderData::from_config(&config);
    assert_eq!(data.mode, 6);
}

// Test 53: Get shader data for performance mode
#[test]
fn test_shader_data_performance_mode() {
    let config = VisualizationConfig::with_mode(DebugVisualization::Overdraw);
    let data = DebugShaderData::from_config(&config);
    assert_eq!(data.mode, 14);
}

// Test 54: Get shader data for depth mode
#[test]
fn test_shader_data_depth_mode() {
    let config = VisualizationConfig::with_mode(DebugVisualization::Depth);
    let data = DebugShaderData::from_config(&config);
    assert_eq!(data.mode, 11);
}

// Test 55: Mode index is sequential
#[test]
fn test_mode_index_sequential() {
    for (i, mode) in DebugVisualization::ALL.iter().enumerate() {
        assert_eq!(mode.index(), i);
    }
}

// Test 56: from_index recovers mode
#[test]
fn test_from_index_round_trip() {
    for mode in DebugVisualization::ALL.iter() {
        let index = mode.index();
        let recovered = DebugVisualization::from_index(index);
        assert_eq!(recovered, Some(*mode));
    }
}

// Test 57: from_index out of bounds returns None
#[test]
fn test_from_index_out_of_bounds() {
    assert_eq!(DebugVisualization::from_index(18), None);
    assert_eq!(DebugVisualization::from_index(100), None);
    assert_eq!(DebugVisualization::from_index(usize::MAX), None);
}

// Test 58: Descriptions are non-empty
#[test]
fn test_descriptions_non_empty() {
    for mode in DebugVisualization::ALL.iter() {
        let desc = mode.description();
        assert!(!desc.is_empty(), "{:?} should have description", mode);
    }
}

// Test 59: Shader defines start with DEBUG_VIS_
#[test]
fn test_shader_defines_prefix() {
    for mode in DebugVisualization::ALL.iter() {
        let define = mode.shader_define();
        assert!(
            define.starts_with("DEBUG_VIS_"),
            "{:?} shader define should start with DEBUG_VIS_",
            mode
        );
    }
}

// =============================================================================
// SECTION 3: Manager Lifecycle (20+ tests)
// =============================================================================

// Test 60: Manager initializes disabled
#[test]
fn test_manager_initializes_disabled() {
    let manager = DebugVisualizationManager::new();
    assert!(!manager.is_enabled());
}

// Test 61: Manager initializes with None mode
#[test]
fn test_manager_initializes_none_mode() {
    let manager = DebugVisualizationManager::new();
    assert_eq!(manager.current_mode(), DebugVisualization::None);
}

// Test 62: Manager enable sets enabled state
#[test]
fn test_manager_enable() {
    let mut manager = DebugVisualizationManager::new();
    manager.enable();
    assert!(manager.is_enabled());
}

// Test 63: Manager disable clears enabled state
#[test]
fn test_manager_disable() {
    let mut manager = DebugVisualizationManager::new();
    manager.enable();
    manager.disable();
    assert!(!manager.is_enabled());
}

// Test 64: Manager set_mode updates current mode
#[test]
fn test_manager_set_mode() {
    let mut manager = DebugVisualizationManager::new();
    manager.set_mode(DebugVisualization::Normals);
    assert_eq!(manager.current_mode(), DebugVisualization::Normals);
}

// Test 65: Manager set_mode updates config mode
#[test]
fn test_manager_set_mode_updates_config() {
    let mut manager = DebugVisualizationManager::new();
    manager.set_mode(DebugVisualization::Wireframe);
    assert_eq!(manager.config().mode, DebugVisualization::Wireframe);
}

// Test 66: Manager toggle changes state
#[test]
fn test_manager_toggle() {
    let mut manager = DebugVisualizationManager::new();
    assert!(!manager.is_enabled());
    manager.toggle();
    assert!(manager.is_enabled());
    manager.toggle();
    assert!(!manager.is_enabled());
}

// Test 67: Manager cycle_next moves to next mode
#[test]
fn test_manager_cycle_next() {
    let mut manager = DebugVisualizationManager::new();
    assert_eq!(manager.current_mode(), DebugVisualization::None);
    manager.cycle_next();
    assert_eq!(manager.current_mode(), DebugVisualization::Wireframe);
    manager.cycle_next();
    assert_eq!(manager.current_mode(), DebugVisualization::Normals);
}

// Test 68: Manager cycle_next wraps around
#[test]
fn test_manager_cycle_next_wraps() {
    let mut manager = DebugVisualizationManager::new();
    manager.set_mode(DebugVisualization::ShadowCascades);
    manager.cycle_next();
    assert_eq!(manager.current_mode(), DebugVisualization::None);
}

// Test 69: Manager cycle_prev moves to previous mode
#[test]
fn test_manager_cycle_prev() {
    let mut manager = DebugVisualizationManager::new();
    manager.set_mode(DebugVisualization::Normals);
    manager.cycle_prev();
    assert_eq!(manager.current_mode(), DebugVisualization::Wireframe);
}

// Test 70: Manager cycle_prev wraps around
#[test]
fn test_manager_cycle_prev_wraps() {
    let mut manager = DebugVisualizationManager::new();
    assert_eq!(manager.current_mode(), DebugVisualization::None);
    manager.cycle_prev();
    assert_eq!(manager.current_mode(), DebugVisualization::ShadowCascades);
}

// Test 71: Manager set_intensity configures intensity
#[test]
fn test_manager_set_intensity() {
    let mut manager = DebugVisualizationManager::new();
    manager.set_intensity(0.5);
    assert_eq!(manager.intensity(), 0.5);
}

// Test 72: Manager intensity affects config
#[test]
fn test_manager_intensity_affects_config() {
    let mut manager = DebugVisualizationManager::new();
    manager.set_intensity(2.0);
    assert_eq!(manager.config().intensity, 2.0);
}

// Test 73: Manager set_channel_mask configures mask
#[test]
fn test_manager_set_channel_mask() {
    let mut manager = DebugVisualizationManager::new();
    manager.set_channel_mask(ChannelMask::RED);
    assert_eq!(manager.channel_mask(), ChannelMask::RED);
}

// Test 74: Manager channel_mask affects config
#[test]
fn test_manager_channel_mask_affects_config() {
    let mut manager = DebugVisualizationManager::new();
    manager.set_channel_mask(ChannelMask::RGB);
    assert!(manager.config().channel_mask.is_rgb());
}

// Test 75: Manager set_overlay configures overlay
#[test]
fn test_manager_set_overlay() {
    let mut manager = DebugVisualizationManager::new();
    manager.set_overlay(true);
    assert!(manager.is_overlay());
}

// Test 76: Manager overlay affects config
#[test]
fn test_manager_overlay_affects_config() {
    let mut manager = DebugVisualizationManager::new();
    manager.set_overlay(true);
    assert!(manager.config().overlay);
}

// Test 77: Manager effective_mode when disabled
#[test]
fn test_manager_effective_mode_disabled() {
    let mut manager = DebugVisualizationManager::new();
    manager.set_mode(DebugVisualization::Overdraw);
    assert!(!manager.is_enabled());
    assert_eq!(manager.effective_mode(), DebugVisualization::None);
}

// Test 78: Manager effective_mode when enabled
#[test]
fn test_manager_effective_mode_enabled() {
    let mut manager = DebugVisualizationManager::new();
    manager.set_mode(DebugVisualization::Overdraw);
    manager.enable();
    assert_eq!(manager.effective_mode(), DebugVisualization::Overdraw);
}

// Test 79: Manager set_config updates everything
#[test]
fn test_manager_set_config() {
    let mut manager = DebugVisualizationManager::new();
    let config = VisualizationConfig::with_mode(DebugVisualization::Albedo)
        .with_intensity(0.75)
        .with_overlay(true)
        .with_channel_mask(ChannelMask::BLUE);

    manager.set_config(config);
    assert_eq!(manager.current_mode(), DebugVisualization::Albedo);
    assert_eq!(manager.intensity(), 0.75);
    assert!(manager.is_overlay());
    assert_eq!(manager.channel_mask(), ChannelMask::BLUE);
}

// Test 80: Manager config_mut allows modification
#[test]
fn test_manager_config_mut() {
    let mut manager = DebugVisualizationManager::new();
    manager.config_mut().intensity = 1.5;
    assert_eq!(manager.intensity(), 1.5);
}

// Test 81: Manager all_modes returns all 18 modes
#[test]
fn test_manager_all_modes() {
    let modes = DebugVisualizationManager::all_modes();
    assert_eq!(modes.len(), 18);
}

// Test 82: Manager Default trait works
#[test]
fn test_manager_default_trait() {
    let manager: DebugVisualizationManager = Default::default();
    assert_eq!(manager.current_mode(), DebugVisualization::None);
    assert!(!manager.is_enabled());
}

// Test 83: Full manager workflow simulation
#[test]
fn test_manager_full_workflow() {
    let mut manager = DebugVisualizationManager::new();

    // Start disabled
    assert!(!manager.is_enabled());

    // Enable and set wireframe
    manager.enable();
    manager.set_mode(DebugVisualization::Wireframe);
    assert!(manager.is_enabled());
    assert_eq!(manager.effective_mode(), DebugVisualization::Wireframe);

    // Cycle through geometry modes
    for _ in 0..4 {
        manager.cycle_next();
        assert!(manager.current_mode().is_geometry() || manager.current_mode() == DebugVisualization::Albedo);
    }

    // Adjust intensity
    manager.set_intensity(0.5);
    assert_eq!(manager.intensity(), 0.5);

    // Toggle off
    manager.toggle();
    assert!(!manager.is_enabled());
    assert_eq!(manager.effective_mode(), DebugVisualization::None);
}

// =============================================================================
// SECTION 4: Shader Integration (15+ tests)
// =============================================================================

// Test 84: shader_define returns valid define string
#[test]
fn test_shader_define_valid_string() {
    for mode in DebugVisualization::ALL.iter() {
        let define = mode.shader_define();
        assert!(!define.is_empty());
        assert!(define.chars().all(|c| c.is_ascii_uppercase() || c == '_'));
    }
}

// Test 85: define_value matches enum index
#[test]
fn test_define_value_matches_index() {
    for mode in DebugVisualization::ALL.iter() {
        assert_eq!(mode.define_value() as usize, mode.index());
    }
}

// Test 86: DebugShaderData as_bytes returns correct size
#[test]
fn test_shader_data_as_bytes_size() {
    let data = DebugShaderData::default();
    let bytes = data.as_bytes();
    assert_eq!(bytes.len(), DebugShaderData::size());
    // u32 + f32 + [f32;4] + [f32;2] = 4 + 4 + 16 + 8 = 32 bytes
    assert_eq!(bytes.len(), 32);
}

// Test 87: DebugShaderData size is constant
#[test]
fn test_shader_data_size_constant() {
    assert_eq!(DebugShaderData::size(), 32);
}

// Test 88: Shader data from config matches mode
#[test]
fn test_shader_data_from_config_mode() {
    for mode in DebugVisualization::ALL.iter() {
        let config = VisualizationConfig::with_mode(*mode);
        let data = DebugShaderData::from_config(&config);
        assert_eq!(data.mode, mode.define_value());
    }
}

// Test 89: Shader data from config matches intensity
#[test]
fn test_shader_data_from_config_intensity() {
    let config = VisualizationConfig::with_mode(DebugVisualization::Wireframe).with_intensity(0.3);
    let data = DebugShaderData::from_config(&config);
    assert_eq!(data.intensity, 0.3);
}

// Test 90: Shader data from config matches channel mask
#[test]
fn test_shader_data_from_config_channel_mask() {
    let config =
        VisualizationConfig::with_mode(DebugVisualization::Normals).with_channel_mask(ChannelMask::GREEN);
    let data = DebugShaderData::from_config(&config);
    assert_eq!(data.channel_mask, [0.0, 1.0, 0.0, 0.0]);
}

// Test 91: Shader data from manager uses effective mode
#[test]
fn test_shader_data_from_manager_effective_mode() {
    let mut manager = DebugVisualizationManager::new();
    manager.set_mode(DebugVisualization::Albedo);

    // Disabled: effective mode is None
    let data_disabled = DebugShaderData::from_manager(&manager);
    assert_eq!(data_disabled.mode, 0);

    // Enabled: effective mode is Albedo
    manager.enable();
    let data_enabled = DebugShaderData::from_manager(&manager);
    assert_eq!(data_enabled.mode, 6);
}

// Test 92: Shader data from manager uses intensity
#[test]
fn test_shader_data_from_manager_intensity() {
    let mut manager = DebugVisualizationManager::new();
    manager.set_intensity(1.8);
    let data = DebugShaderData::from_manager(&manager);
    assert_eq!(data.intensity, 1.8);
}

// Test 93: Shader data from manager uses channel mask
#[test]
fn test_shader_data_from_manager_channel_mask() {
    let mut manager = DebugVisualizationManager::new();
    manager.set_channel_mask(ChannelMask::ALPHA);
    let data = DebugShaderData::from_manager(&manager);
    assert_eq!(data.channel_mask, [0.0, 0.0, 0.0, 1.0]);
}

// Test 94: Shader data is_disabled for None mode
#[test]
fn test_shader_data_is_disabled_none() {
    let data = DebugShaderData::default();
    assert!(data.is_disabled());
}

// Test 95: Shader data is not disabled for active mode
#[test]
fn test_shader_data_not_disabled_active() {
    let config = VisualizationConfig::with_mode(DebugVisualization::Wireframe);
    let data = DebugShaderData::from_config(&config);
    assert!(!data.is_disabled());
}

// Test 96: Mode change updates shader data
#[test]
fn test_mode_change_updates_shader_data() {
    let mut manager = DebugVisualizationManager::new();
    manager.enable();
    manager.set_mode(DebugVisualization::Depth);

    let data1 = DebugShaderData::from_manager(&manager);
    assert_eq!(data1.mode, 11);

    manager.set_mode(DebugVisualization::LinearDepth);
    let data2 = DebugShaderData::from_manager(&manager);
    assert_eq!(data2.mode, 12);
}

// Test 97: Shader data default has all channels enabled
#[test]
fn test_shader_data_default_all_channels() {
    let data = DebugShaderData::default();
    assert_eq!(data.channel_mask, [1.0, 1.0, 1.0, 1.0]);
}

// Test 98: Shader data derives Clone
#[test]
fn test_shader_data_derives_clone() {
    let data = DebugShaderData::default();
    let cloned = data.clone();
    assert_eq!(data, cloned);
}

// =============================================================================
// SECTION 5: Channel Mask Operations (15+ tests)
// =============================================================================

// Test 99: RED mask isolates red channel
#[test]
fn test_channel_mask_red_isolates() {
    let mask = ChannelMask::RED;
    let floats = mask.as_floats();
    assert_eq!(floats, [1.0, 0.0, 0.0, 0.0]);
}

// Test 100: GREEN mask isolates green channel
#[test]
fn test_channel_mask_green_isolates() {
    let mask = ChannelMask::GREEN;
    let floats = mask.as_floats();
    assert_eq!(floats, [0.0, 1.0, 0.0, 0.0]);
}

// Test 101: BLUE mask isolates blue channel
#[test]
fn test_channel_mask_blue_isolates() {
    let mask = ChannelMask::BLUE;
    let floats = mask.as_floats();
    assert_eq!(floats, [0.0, 0.0, 1.0, 0.0]);
}

// Test 102: ALPHA mask isolates alpha channel
#[test]
fn test_channel_mask_alpha_isolates() {
    let mask = ChannelMask::ALPHA;
    let floats = mask.as_floats();
    assert_eq!(floats, [0.0, 0.0, 0.0, 1.0]);
}

// Test 103: RGB mask excludes alpha
#[test]
fn test_channel_mask_rgb_excludes_alpha() {
    let mask = ChannelMask::RGB;
    assert!(mask.r && mask.g && mask.b);
    assert!(!mask.a);
    assert!(mask.is_rgb());
}

// Test 104: ALL mask includes all channels
#[test]
fn test_channel_mask_all_includes_all() {
    let mask = ChannelMask::ALL;
    assert!(mask.r && mask.g && mask.b && mask.a);
    assert!(mask.is_all());
}

// Test 105: Custom mask combination RG
#[test]
fn test_channel_mask_custom_rg() {
    let mask = ChannelMask::new(true, true, false, false);
    assert!(mask.r && mask.g);
    assert!(!mask.b && !mask.a);
    assert_eq!(mask.count(), 2);
}

// Test 106: Custom mask combination BA
#[test]
fn test_channel_mask_custom_ba() {
    let mask = ChannelMask::new(false, false, true, true);
    assert!(!mask.r && !mask.g);
    assert!(mask.b && mask.a);
    assert_eq!(mask.count(), 2);
}

// Test 107: Empty mask is_none
#[test]
fn test_channel_mask_empty_is_none() {
    let mask = ChannelMask::new(false, false, false, false);
    assert!(mask.is_none());
    assert_eq!(mask.count(), 0);
}

// Test 108: Channel mask count accuracy
#[test]
fn test_channel_mask_count_accuracy() {
    assert_eq!(ChannelMask::ALL.count(), 4);
    assert_eq!(ChannelMask::RGB.count(), 3);
    assert_eq!(ChannelMask::RED.count(), 1);
    assert_eq!(ChannelMask::new(false, false, false, false).count(), 0);
    assert_eq!(ChannelMask::new(true, true, false, true).count(), 3);
}

// Test 109: Channel mask Default is ALL
#[test]
fn test_channel_mask_default_is_all() {
    let mask: ChannelMask = Default::default();
    assert!(mask.is_all());
}

// Test 110: Channel mask derives PartialEq
#[test]
fn test_channel_mask_derives_partial_eq() {
    assert_eq!(ChannelMask::ALL, ChannelMask::ALL);
    assert_ne!(ChannelMask::RED, ChannelMask::GREEN);
}

// Test 111: Channel mask derives Clone
#[test]
fn test_channel_mask_derives_clone() {
    let mask = ChannelMask::RGB;
    let cloned = mask.clone();
    assert_eq!(mask, cloned);
}

// Test 112: Channel mask derives Copy
#[test]
fn test_channel_mask_derives_copy() {
    let mask = ChannelMask::BLUE;
    let copied = mask; // Copy, not move
    assert_eq!(mask, copied);
}

// Test 113: as_floats consistency
#[test]
fn test_channel_mask_as_floats_consistency() {
    for r in [true, false] {
        for g in [true, false] {
            for b in [true, false] {
                for a in [true, false] {
                    let mask = ChannelMask::new(r, g, b, a);
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

// =============================================================================
// SECTION 6: Additional Scenario Tests (to reach 105+ tests)
// =============================================================================

// Test 114: VisualizationConfig with_mode builder
#[test]
fn test_config_with_mode_builder() {
    let config = VisualizationConfig::with_mode(DebugVisualization::Tangents);
    assert_eq!(config.mode, DebugVisualization::Tangents);
    // Defaults
    assert_eq!(config.intensity, 1.0);
    assert!(!config.overlay);
    assert!(config.channel_mask.is_all());
}

// Test 115: VisualizationConfig full builder chain
#[test]
fn test_config_full_builder_chain() {
    let config = VisualizationConfig::with_mode(DebugVisualization::Overdraw)
        .with_intensity(0.25)
        .with_overlay(true)
        .with_channel_mask(ChannelMask::RED);

    assert_eq!(config.mode, DebugVisualization::Overdraw);
    assert_eq!(config.intensity, 0.25);
    assert!(config.overlay);
    assert_eq!(config.channel_mask, ChannelMask::RED);
}

// Test 116: VisualizationConfig derives Clone
#[test]
fn test_config_derives_clone() {
    let config = VisualizationConfig::with_mode(DebugVisualization::Albedo);
    let cloned = config.clone();
    assert_eq!(config, cloned);
}

// Test 117: VisualizationConfig derives PartialEq
#[test]
fn test_config_derives_partial_eq() {
    let config1 = VisualizationConfig::with_mode(DebugVisualization::Albedo);
    let config2 = VisualizationConfig::with_mode(DebugVisualization::Albedo);
    let config3 = VisualizationConfig::with_mode(DebugVisualization::Roughness);
    assert_eq!(config1, config2);
    assert_ne!(config1, config3);
}

// Test 118: DebugVisualization derives Debug
#[test]
fn test_debug_visualization_derives_debug() {
    let mode = DebugVisualization::Wireframe;
    let debug_str = format!("{:?}", mode);
    assert!(debug_str.contains("Wireframe"));
}

// Test 119: DebugVisualization derives Clone
#[test]
fn test_debug_visualization_derives_clone() {
    let mode = DebugVisualization::Normals;
    let cloned = mode.clone();
    assert_eq!(mode, cloned);
}

// Test 120: DebugVisualization derives Copy
#[test]
fn test_debug_visualization_derives_copy() {
    let mode = DebugVisualization::Tangents;
    let copied = mode; // Copy, not move
    assert_eq!(mode, copied);
}

// Test 121: DebugVisualization derives PartialEq
#[test]
fn test_debug_visualization_derives_partial_eq() {
    assert_eq!(DebugVisualization::None, DebugVisualization::None);
    assert_ne!(DebugVisualization::None, DebugVisualization::Wireframe);
}

// Test 122: DebugVisualization derives Eq
#[test]
fn test_debug_visualization_derives_eq() {
    let mode1 = DebugVisualization::Albedo;
    let mode2 = DebugVisualization::Albedo;
    // Eq trait has no methods, but we can verify it by using it in HashMap key
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(mode1);
    assert!(set.contains(&mode2));
}

// Test 123: DebugVisualization derives Hash
#[test]
fn test_debug_visualization_derives_hash() {
    use std::collections::HashMap;
    let mut map = HashMap::new();
    map.insert(DebugVisualization::Wireframe, "wireframe");
    map.insert(DebugVisualization::Normals, "normals");
    assert_eq!(map.get(&DebugVisualization::Wireframe), Some(&"wireframe"));
}

// Test 124: Manager cycle through all modes
#[test]
fn test_manager_cycle_all_modes() {
    let mut manager = DebugVisualizationManager::new();
    let mut seen = Vec::new();

    for _ in 0..18 {
        seen.push(manager.current_mode());
        manager.cycle_next();
    }

    // Should see all modes exactly once
    assert_eq!(seen.len(), 18);
    for mode in DebugVisualization::ALL.iter() {
        assert!(seen.contains(mode), "Should have seen {:?}", mode);
    }
}

// Test 125: Shader data bytes are valid
#[test]
fn test_shader_data_bytes_valid() {
    let data = DebugShaderData {
        mode: 5,
        intensity: 0.5,
        channel_mask: [1.0, 0.0, 1.0, 0.0],
        _padding: [0.0, 0.0],
    };

    let bytes = data.as_bytes();
    assert_eq!(bytes.len(), 32);

    // First 4 bytes should be mode (5 in little endian)
    let mode_bytes = &bytes[0..4];
    assert_eq!(u32::from_le_bytes(mode_bytes.try_into().unwrap()), 5);
}
