//! Blackbox tests for override constants module.
//!
//! Tests the public API of the override_constants module from an external
//! user perspective, verifying behavior without access to internal state.

use std::collections::HashMap;

use renderer_backend::shaders::{
    extract_overrides_from_wgsl, OverrideConstantInfo, OverrideConstantType, OverrideConstants,
    OverrideError, PipelineConstants,
};

// ============================================================================
// API Surface Tests - OverrideConstantType
// ============================================================================

#[test]
fn test_api_override_constant_type_all_variants_exist() {
    // Verify all four variants are accessible
    let types = [
        OverrideConstantType::Bool,
        OverrideConstantType::I32,
        OverrideConstantType::U32,
        OverrideConstantType::F32,
    ];
    assert_eq!(types.len(), 4);
}

#[test]
fn test_api_override_constant_type_wgsl_name_returns_non_empty() {
    let types = [
        OverrideConstantType::Bool,
        OverrideConstantType::I32,
        OverrideConstantType::U32,
        OverrideConstantType::F32,
    ];
    for ty in types {
        let name = ty.wgsl_name();
        assert!(!name.is_empty(), "wgsl_name() should not be empty for {:?}", ty);
    }
}

#[test]
fn test_api_override_constant_type_wgsl_name_matches_wgsl_syntax() {
    assert_eq!(OverrideConstantType::Bool.wgsl_name(), "bool");
    assert_eq!(OverrideConstantType::I32.wgsl_name(), "i32");
    assert_eq!(OverrideConstantType::U32.wgsl_name(), "u32");
    assert_eq!(OverrideConstantType::F32.wgsl_name(), "f32");
}

#[test]
fn test_api_override_constant_type_display_matches_wgsl_name() {
    let types = [
        OverrideConstantType::Bool,
        OverrideConstantType::I32,
        OverrideConstantType::U32,
        OverrideConstantType::F32,
    ];
    for ty in types {
        assert_eq!(format!("{}", ty), ty.wgsl_name());
    }
}

#[test]
fn test_api_override_constant_type_default_is_f32() {
    assert_eq!(OverrideConstantType::default(), OverrideConstantType::F32);
}

#[test]
fn test_api_override_constant_type_default_value_all_zero() {
    let types = [
        OverrideConstantType::Bool,
        OverrideConstantType::I32,
        OverrideConstantType::U32,
        OverrideConstantType::F32,
    ];
    for ty in types {
        assert_eq!(ty.default_value(), 0.0, "{:?} default should be 0.0", ty);
    }
}

#[test]
fn test_api_override_constant_type_value_range_non_empty() {
    let types = [
        OverrideConstantType::Bool,
        OverrideConstantType::I32,
        OverrideConstantType::U32,
        OverrideConstantType::F32,
    ];
    for ty in types {
        let range = ty.value_range();
        assert!(!range.is_empty(), "value_range() should not be empty for {:?}", ty);
    }
}

#[test]
fn test_api_override_constant_type_debug_and_clone() {
    let ty = OverrideConstantType::Bool;
    let cloned = ty.clone();
    assert_eq!(ty, cloned);
    let debug = format!("{:?}", ty);
    assert!(debug.contains("Bool"));
}

#[test]
fn test_api_override_constant_type_hash_eq() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(OverrideConstantType::Bool);
    set.insert(OverrideConstantType::I32);
    set.insert(OverrideConstantType::U32);
    set.insert(OverrideConstantType::F32);
    assert_eq!(set.len(), 4);
    // Inserting duplicate should not increase size
    set.insert(OverrideConstantType::Bool);
    assert_eq!(set.len(), 4);
}

#[test]
fn test_api_override_constant_type_copy() {
    let ty = OverrideConstantType::Bool;
    let copied = ty; // Copy trait
    assert_eq!(ty, copied);
}

// ============================================================================
// API Surface Tests - OverrideConstantInfo
// ============================================================================

#[test]
fn test_api_override_constant_info_new() {
    let info = OverrideConstantInfo::new(
        Some("TEST".to_string()),
        Some(0),
        OverrideConstantType::F32,
        Some(1.0),
    );
    assert_eq!(info.name, Some("TEST".to_string()));
    assert_eq!(info.id, Some(0));
    assert_eq!(info.ty, OverrideConstantType::F32);
    assert_eq!(info.default_value, Some(1.0));
    assert!(!info.required);
}

#[test]
fn test_api_override_constant_info_required() {
    let info = OverrideConstantInfo::required(
        Some("REQUIRED_CONST".to_string()),
        Some(1),
        OverrideConstantType::U32,
    );
    assert!(info.required);
    assert!(info.default_value.is_none());
}

#[test]
fn test_api_override_constant_info_with_default() {
    let info = OverrideConstantInfo::with_default(
        Some("WITH_DEFAULT".to_string()),
        None,
        OverrideConstantType::I32,
        42.0,
    );
    assert!(!info.required);
    assert_eq!(info.default_value, Some(42.0));
}

#[test]
fn test_api_override_constant_info_key_prefers_name() {
    let info = OverrideConstantInfo::new(
        Some("MY_CONST".to_string()),
        Some(5),
        OverrideConstantType::F32,
        None,
    );
    assert_eq!(info.key(), Some("MY_CONST".to_string()));
}

#[test]
fn test_api_override_constant_info_key_falls_back_to_id() {
    let info = OverrideConstantInfo::new(None, Some(42), OverrideConstantType::F32, None);
    assert_eq!(info.key(), Some("42".to_string()));
}

#[test]
fn test_api_override_constant_info_key_none_when_neither() {
    let info = OverrideConstantInfo::new(None, None, OverrideConstantType::F32, None);
    assert_eq!(info.key(), None);
}

#[test]
fn test_api_override_constant_info_has_name() {
    let with_name = OverrideConstantInfo::new(Some("X".to_string()), None, OverrideConstantType::F32, None);
    let without_name = OverrideConstantInfo::new(None, Some(0), OverrideConstantType::F32, None);
    assert!(with_name.has_name());
    assert!(!without_name.has_name());
}

#[test]
fn test_api_override_constant_info_has_id() {
    let with_id = OverrideConstantInfo::new(None, Some(0), OverrideConstantType::F32, None);
    let without_id = OverrideConstantInfo::new(Some("X".to_string()), None, OverrideConstantType::F32, None);
    assert!(with_id.has_id());
    assert!(!without_id.has_id());
}

#[test]
fn test_api_override_constant_info_has_default() {
    let with_default = OverrideConstantInfo::with_default(Some("X".to_string()), None, OverrideConstantType::F32, 1.0);
    let without_default = OverrideConstantInfo::required(Some("Y".to_string()), None, OverrideConstantType::F32);
    assert!(with_default.has_default());
    assert!(!without_default.has_default());
}

#[test]
fn test_api_override_constant_info_display() {
    let info = OverrideConstantInfo::new(
        Some("WIDTH".to_string()),
        Some(0),
        OverrideConstantType::F32,
        Some(1920.0),
    );
    let display = format!("{}", info);
    assert!(display.contains("@id(0)"));
    assert!(display.contains("WIDTH"));
    assert!(display.contains("f32"));
    assert!(display.contains("1920"));
}

// ============================================================================
// API Surface Tests - OverrideConstants
// ============================================================================

#[test]
fn test_api_override_constants_new_is_empty() {
    let overrides = OverrideConstants::new();
    assert!(overrides.is_empty());
    assert_eq!(overrides.len(), 0);
}

#[test]
fn test_api_override_constants_default_is_empty() {
    let overrides = OverrideConstants::default();
    assert!(overrides.is_empty());
}

#[test]
fn test_api_override_constants_add_increments_len() {
    let mut overrides = OverrideConstants::new();
    overrides.add(OverrideConstantInfo::new(
        Some("A".to_string()),
        Some(0),
        OverrideConstantType::F32,
        Some(1.0),
    ));
    assert_eq!(overrides.len(), 1);
    overrides.add(OverrideConstantInfo::new(
        Some("B".to_string()),
        Some(1),
        OverrideConstantType::U32,
        Some(2.0),
    ));
    assert_eq!(overrides.len(), 2);
}

#[test]
fn test_api_override_constants_from_infos() {
    let infos = vec![
        OverrideConstantInfo::new(Some("A".to_string()), Some(0), OverrideConstantType::F32, Some(1.0)),
        OverrideConstantInfo::new(Some("B".to_string()), Some(1), OverrideConstantType::U32, Some(2.0)),
        OverrideConstantInfo::new(Some("C".to_string()), Some(2), OverrideConstantType::Bool, Some(1.0)),
    ];
    let overrides = OverrideConstants::from_infos(infos);
    assert_eq!(overrides.len(), 3);
}

#[test]
fn test_api_override_constants_get_by_name() {
    let mut overrides = OverrideConstants::new();
    overrides.add(OverrideConstantInfo::new(
        Some("SCREEN_WIDTH".to_string()),
        None,
        OverrideConstantType::F32,
        Some(1920.0),
    ));

    let result = overrides.get_by_name("SCREEN_WIDTH");
    assert!(result.is_some());
    assert_eq!(result.unwrap().default_value, Some(1920.0));

    assert!(overrides.get_by_name("NONEXISTENT").is_none());
}

#[test]
fn test_api_override_constants_get_by_id() {
    let mut overrides = OverrideConstants::new();
    overrides.add(OverrideConstantInfo::new(
        Some("TILE_SIZE".to_string()),
        Some(42),
        OverrideConstantType::U32,
        Some(16.0),
    ));

    let result = overrides.get_by_id(42);
    assert!(result.is_some());
    assert_eq!(result.unwrap().name, Some("TILE_SIZE".to_string()));

    assert!(overrides.get_by_id(99).is_none());
}

#[test]
fn test_api_override_constants_get_by_key_uses_name_first() {
    let mut overrides = OverrideConstants::new();
    overrides.add(OverrideConstantInfo::new(
        Some("MY_CONST".to_string()),
        Some(10),
        OverrideConstantType::F32,
        Some(1.0),
    ));

    assert!(overrides.get_by_key("MY_CONST").is_some());
    assert!(overrides.get_by_key("10").is_some()); // ID as string also works
}

#[test]
fn test_api_override_constants_get_by_key_parses_numeric_as_id() {
    let mut overrides = OverrideConstants::new();
    overrides.add(OverrideConstantInfo::new(
        None,
        Some(7),
        OverrideConstantType::F32,
        Some(1.0),
    ));

    assert!(overrides.get_by_key("7").is_some());
    assert!(overrides.get_by_key("8").is_none());
}

#[test]
fn test_api_override_constants_iter_yields_all() {
    let infos = vec![
        OverrideConstantInfo::new(Some("A".to_string()), None, OverrideConstantType::F32, Some(1.0)),
        OverrideConstantInfo::new(Some("B".to_string()), None, OverrideConstantType::U32, Some(2.0)),
    ];
    let overrides = OverrideConstants::from_infos(infos);

    let collected: Vec<_> = overrides.iter().collect();
    assert_eq!(collected.len(), 2);
}

#[test]
fn test_api_override_constants_as_slice() {
    let infos = vec![
        OverrideConstantInfo::new(Some("A".to_string()), None, OverrideConstantType::F32, Some(1.0)),
    ];
    let overrides = OverrideConstants::from_infos(infos);

    let slice = overrides.as_slice();
    assert_eq!(slice.len(), 1);
    assert_eq!(slice[0].name, Some("A".to_string()));
}

#[test]
fn test_api_override_constants_required_filter() {
    let infos = vec![
        OverrideConstantInfo::with_default(Some("OPTIONAL".to_string()), None, OverrideConstantType::F32, 1.0),
        OverrideConstantInfo::required(Some("REQUIRED".to_string()), None, OverrideConstantType::U32),
    ];
    let overrides = OverrideConstants::from_infos(infos);

    let required: Vec<_> = overrides.required().collect();
    assert_eq!(required.len(), 1);
    assert_eq!(required[0].name, Some("REQUIRED".to_string()));
}

#[test]
fn test_api_override_constants_optional_filter() {
    let infos = vec![
        OverrideConstantInfo::with_default(Some("OPTIONAL".to_string()), None, OverrideConstantType::F32, 1.0),
        OverrideConstantInfo::required(Some("REQUIRED".to_string()), None, OverrideConstantType::U32),
    ];
    let overrides = OverrideConstants::from_infos(infos);

    let optional: Vec<_> = overrides.optional().collect();
    assert_eq!(optional.len(), 1);
    assert_eq!(optional[0].name, Some("OPTIONAL".to_string()));
}

#[test]
fn test_api_override_constants_names() {
    let infos = vec![
        OverrideConstantInfo::new(Some("A".to_string()), None, OverrideConstantType::F32, Some(1.0)),
        OverrideConstantInfo::new(None, Some(0), OverrideConstantType::U32, Some(2.0)), // No name
        OverrideConstantInfo::new(Some("C".to_string()), None, OverrideConstantType::Bool, Some(1.0)),
    ];
    let overrides = OverrideConstants::from_infos(infos);

    let names: Vec<_> = overrides.names().collect();
    assert_eq!(names.len(), 2);
    assert!(names.contains(&"A"));
    assert!(names.contains(&"C"));
}

#[test]
fn test_api_override_constants_ids() {
    let infos = vec![
        OverrideConstantInfo::new(Some("A".to_string()), Some(0), OverrideConstantType::F32, Some(1.0)),
        OverrideConstantInfo::new(Some("B".to_string()), None, OverrideConstantType::U32, Some(2.0)), // No ID
        OverrideConstantInfo::new(Some("C".to_string()), Some(5), OverrideConstantType::Bool, Some(1.0)),
    ];
    let overrides = OverrideConstants::from_infos(infos);

    let ids: Vec<_> = overrides.ids().collect();
    assert_eq!(ids.len(), 2);
    assert!(ids.contains(&0));
    assert!(ids.contains(&5));
}

#[test]
fn test_api_override_constants_into_iter() {
    let infos = vec![
        OverrideConstantInfo::new(Some("A".to_string()), None, OverrideConstantType::F32, Some(1.0)),
        OverrideConstantInfo::new(Some("B".to_string()), None, OverrideConstantType::U32, Some(2.0)),
    ];
    let overrides = OverrideConstants::from_infos(infos);

    let collected: Vec<OverrideConstantInfo> = overrides.into_iter().collect();
    assert_eq!(collected.len(), 2);
}

#[test]
fn test_api_override_constants_ref_into_iter() {
    let infos = vec![
        OverrideConstantInfo::new(Some("A".to_string()), None, OverrideConstantType::F32, Some(1.0)),
    ];
    let overrides = OverrideConstants::from_infos(infos);

    for info in &overrides {
        assert!(info.name.is_some());
    }
}

#[test]
fn test_api_override_constants_display() {
    let infos = vec![
        OverrideConstantInfo::new(Some("A".to_string()), None, OverrideConstantType::F32, Some(1.0)),
    ];
    let overrides = OverrideConstants::from_infos(infos);

    let display = format!("{}", overrides);
    assert!(display.contains("OverrideConstants"));
    assert!(display.contains("1 constants"));
}

// ============================================================================
// API Surface Tests - PipelineConstants
// ============================================================================

#[test]
fn test_api_pipeline_constants_new_is_empty() {
    let constants = PipelineConstants::new();
    assert!(constants.is_empty());
    assert_eq!(constants.len(), 0);
}

#[test]
fn test_api_pipeline_constants_default_is_empty() {
    let constants = PipelineConstants::default();
    assert!(constants.is_empty());
}

#[test]
fn test_api_pipeline_constants_set_and_get() {
    let mut constants = PipelineConstants::new();
    constants.set("WIDTH", 1920.0);

    assert_eq!(constants.get("WIDTH"), Some(1920.0));
    assert_eq!(constants.len(), 1);
}

#[test]
fn test_api_pipeline_constants_with_chaining() {
    let constants = PipelineConstants::new()
        .with("A", 1.0)
        .with("B", 2.0)
        .with("C", 3.0);

    assert_eq!(constants.len(), 3);
    assert_eq!(constants.get("A"), Some(1.0));
    assert_eq!(constants.get("B"), Some(2.0));
    assert_eq!(constants.get("C"), Some(3.0));
}

#[test]
fn test_api_pipeline_constants_set_bool() {
    let mut constants = PipelineConstants::new();
    constants.set_bool("ENABLED", true);
    constants.set_bool("DISABLED", false);

    assert_eq!(constants.get("ENABLED"), Some(1.0));
    assert_eq!(constants.get("DISABLED"), Some(0.0));
}

#[test]
fn test_api_pipeline_constants_with_bool() {
    let constants = PipelineConstants::new()
        .with_bool("FEATURE_A", true)
        .with_bool("FEATURE_B", false);

    assert_eq!(constants.get("FEATURE_A"), Some(1.0));
    assert_eq!(constants.get("FEATURE_B"), Some(0.0));
}

#[test]
fn test_api_pipeline_constants_set_i32() {
    let mut constants = PipelineConstants::new();
    constants.set_i32("POSITIVE", 100);
    constants.set_i32("NEGATIVE", -100);
    constants.set_i32("ZERO", 0);

    assert_eq!(constants.get("POSITIVE"), Some(100.0));
    assert_eq!(constants.get("NEGATIVE"), Some(-100.0));
    assert_eq!(constants.get("ZERO"), Some(0.0));
}

#[test]
fn test_api_pipeline_constants_with_i32() {
    let constants = PipelineConstants::new().with_i32("OFFSET", -50);
    assert_eq!(constants.get("OFFSET"), Some(-50.0));
}

#[test]
fn test_api_pipeline_constants_set_u32() {
    let mut constants = PipelineConstants::new();
    constants.set_u32("COUNT", 42);

    assert_eq!(constants.get("COUNT"), Some(42.0));
}

#[test]
fn test_api_pipeline_constants_with_u32() {
    let constants = PipelineConstants::new().with_u32("SIZE", 256);
    assert_eq!(constants.get("SIZE"), Some(256.0));
}

#[test]
fn test_api_pipeline_constants_set_f32() {
    let mut constants = PipelineConstants::new();
    constants.set_f32("SCALE", 1.5);

    assert_eq!(constants.get("SCALE"), Some(1.5_f32 as f64));
}

#[test]
fn test_api_pipeline_constants_with_f32() {
    let constants = PipelineConstants::new().with_f32("FACTOR", 2.5);
    assert_eq!(constants.get("FACTOR"), Some(2.5_f32 as f64));
}

#[test]
fn test_api_pipeline_constants_remove() {
    let mut constants = PipelineConstants::new().with("A", 1.0);

    let removed = constants.remove("A");
    assert_eq!(removed, Some(1.0));
    assert!(constants.is_empty());

    let removed_again = constants.remove("A");
    assert_eq!(removed_again, None);
}

#[test]
fn test_api_pipeline_constants_contains() {
    let constants = PipelineConstants::new().with("EXISTS", 1.0);

    assert!(constants.contains("EXISTS"));
    assert!(!constants.contains("MISSING"));
}

#[test]
fn test_api_pipeline_constants_clear() {
    let mut constants = PipelineConstants::new()
        .with("A", 1.0)
        .with("B", 2.0);

    constants.clear();
    assert!(constants.is_empty());
}

#[test]
fn test_api_pipeline_constants_iter() {
    let constants = PipelineConstants::new()
        .with("A", 1.0)
        .with("B", 2.0);

    let collected: Vec<_> = constants.iter().collect();
    assert_eq!(collected.len(), 2);
}

#[test]
fn test_api_pipeline_constants_keys() {
    let constants = PipelineConstants::new()
        .with("X", 1.0)
        .with("Y", 2.0)
        .with("Z", 3.0);

    let keys: Vec<_> = constants.keys().collect();
    assert_eq!(keys.len(), 3);
}

#[test]
fn test_api_pipeline_constants_to_wgpu() {
    let constants = PipelineConstants::new()
        .with("WIDTH", 1920.0)
        .with("HEIGHT", 1080.0);

    let wgpu_map = constants.to_wgpu();
    assert_eq!(wgpu_map.get("WIDTH"), Some(&1920.0));
    assert_eq!(wgpu_map.get("HEIGHT"), Some(&1080.0));
}

#[test]
fn test_api_pipeline_constants_as_map() {
    let constants = PipelineConstants::new().with("KEY", 42.0);

    let map = constants.as_map();
    assert_eq!(map.get("KEY"), Some(&42.0));
}

#[test]
fn test_api_pipeline_constants_from_map() {
    let mut map = HashMap::new();
    map.insert("A".to_string(), 1.0);
    map.insert("B".to_string(), 2.0);

    let constants = PipelineConstants::from_map(map);
    assert_eq!(constants.len(), 2);
    assert_eq!(constants.get("A"), Some(1.0));
}

#[test]
fn test_api_pipeline_constants_merge() {
    let mut a = PipelineConstants::new()
        .with("X", 1.0)
        .with("Y", 2.0);

    let b = PipelineConstants::new()
        .with("Y", 20.0) // Override
        .with("Z", 3.0); // New

    a.merge(&b);

    assert_eq!(a.get("X"), Some(1.0));
    assert_eq!(a.get("Y"), Some(20.0));
    assert_eq!(a.get("Z"), Some(3.0));
}

#[test]
fn test_api_pipeline_constants_merged_with() {
    let a = PipelineConstants::new().with("A", 1.0);
    let b = PipelineConstants::new().with("B", 2.0);

    let c = a.merged_with(&b);

    assert_eq!(c.get("A"), Some(1.0));
    assert_eq!(c.get("B"), Some(2.0));
}

#[test]
fn test_api_pipeline_constants_from_iter_string() {
    let items = vec![
        ("A".to_string(), 1.0),
        ("B".to_string(), 2.0),
    ];

    let constants: PipelineConstants = items.into_iter().collect();
    assert_eq!(constants.get("A"), Some(1.0));
    assert_eq!(constants.get("B"), Some(2.0));
}

#[test]
fn test_api_pipeline_constants_from_iter_str() {
    let items = vec![("X", 10.0), ("Y", 20.0)];

    let constants: PipelineConstants = items.into_iter().collect();
    assert_eq!(constants.get("X"), Some(10.0));
    assert_eq!(constants.get("Y"), Some(20.0));
}

#[test]
fn test_api_pipeline_constants_display() {
    let constants = PipelineConstants::new().with("TEST", 42.0);

    let display = format!("{}", constants);
    assert!(display.contains("PipelineConstants"));
    assert!(display.contains("TEST"));
    assert!(display.contains("42"));
}

// ============================================================================
// Real Shader Integration Tests
// ============================================================================

#[test]
fn test_shader_simple_override() {
    let source = r#"
        override SCALE: f32 = 1.0;

        var<private> sink: f32;

        @compute @workgroup_size(1)
        fn main() {
            sink = SCALE;
        }
    "#;

    let result = extract_overrides_from_wgsl(source);
    assert!(result.is_ok());

    let overrides = result.unwrap();
    assert_eq!(overrides.len(), 1);

    let scale = overrides.get_by_name("SCALE");
    assert!(scale.is_some());
    assert_eq!(scale.unwrap().ty, OverrideConstantType::F32);
    assert_eq!(scale.unwrap().default_value, Some(1.0));
    assert!(!scale.unwrap().required);
}

#[test]
fn test_shader_screen_dimensions_pattern() {
    let source = r#"
        @id(0) override SCREEN_WIDTH: f32 = 1920.0;
        @id(1) override SCREEN_HEIGHT: f32 = 1080.0;

        var<private> sink: f32;

        @compute @workgroup_size(1)
        fn main() {
            sink = SCREEN_WIDTH * SCREEN_HEIGHT;
        }
    "#;

    let overrides = extract_overrides_from_wgsl(source).unwrap();
    assert_eq!(overrides.len(), 2);

    let width = overrides.get_by_id(0);
    assert!(width.is_some());
    assert_eq!(width.unwrap().name, Some("SCREEN_WIDTH".to_string()));

    let height = overrides.get_by_id(1);
    assert!(height.is_some());
    assert_eq!(height.unwrap().name, Some("SCREEN_HEIGHT".to_string()));
}

#[test]
fn test_shader_tile_size_pattern() {
    let source = r#"
        override TILE_SIZE: u32 = 16u;
        override TILES_X: u32 = 8u;
        override TILES_Y: u32 = 8u;

        var<private> sink: u32;

        @compute @workgroup_size(1)
        fn main() {
            sink = TILE_SIZE * TILES_X * TILES_Y;
        }
    "#;

    let overrides = extract_overrides_from_wgsl(source).unwrap();
    assert_eq!(overrides.len(), 3);

    for name in ["TILE_SIZE", "TILES_X", "TILES_Y"] {
        let info = overrides.get_by_name(name);
        assert!(info.is_some(), "Missing override: {}", name);
        assert_eq!(info.unwrap().ty, OverrideConstantType::U32);
    }
}

#[test]
fn test_shader_feature_flags_pattern() {
    let source = r#"
        override ENABLE_SHADOWS: bool = true;
        override ENABLE_AO: bool = false;
        override ENABLE_SSR: bool = false;

        var<private> sink_f: f32;

        @compute @workgroup_size(1)
        fn main() {
            if ENABLE_SHADOWS {
                sink_f = 1.0;
            }
        }
    "#;

    let overrides = extract_overrides_from_wgsl(source).unwrap();
    assert_eq!(overrides.len(), 3);

    let shadows = overrides.get_by_name("ENABLE_SHADOWS").unwrap();
    assert_eq!(shadows.ty, OverrideConstantType::Bool);
    assert_eq!(shadows.default_value, Some(1.0)); // true = 1.0

    let ao = overrides.get_by_name("ENABLE_AO").unwrap();
    assert_eq!(ao.default_value, Some(0.0)); // false = 0.0
}

#[test]
fn test_shader_pbr_defaults_pattern() {
    let source = r#"
        override DEFAULT_METALLIC: f32 = 0.0;
        override DEFAULT_ROUGHNESS: f32 = 0.5;
        override DEFAULT_AO: f32 = 1.0;

        var<private> sink: f32;

        @compute @workgroup_size(1)
        fn main() {
            sink = DEFAULT_METALLIC + DEFAULT_ROUGHNESS + DEFAULT_AO;
        }
    "#;

    let overrides = extract_overrides_from_wgsl(source).unwrap();

    let metallic = overrides.get_by_name("DEFAULT_METALLIC").unwrap();
    assert_eq!(metallic.default_value, Some(0.0));

    let roughness = overrides.get_by_name("DEFAULT_ROUGHNESS").unwrap();
    assert_eq!(roughness.default_value, Some(0.5));

    let ao = overrides.get_by_name("DEFAULT_AO").unwrap();
    assert_eq!(ao.default_value, Some(1.0));
}

#[test]
fn test_shader_compute_workgroup_counts() {
    let source = r#"
        override DISPATCH_X: u32 = 256u;
        override DISPATCH_Y: u32 = 256u;
        override DISPATCH_Z: u32 = 1u;

        var<private> sink: u32;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            sink = id.x;
        }
    "#;

    let overrides = extract_overrides_from_wgsl(source).unwrap();
    assert_eq!(overrides.len(), 3);

    let dispatch_x = overrides.get_by_name("DISPATCH_X").unwrap();
    assert_eq!(dispatch_x.ty, OverrideConstantType::U32);
    assert_eq!(dispatch_x.default_value, Some(256.0));
}

#[test]
fn test_shader_required_constant() {
    let source = r#"
        override MAX_LIGHTS: u32;

        var<private> sink: u32;

        @compute @workgroup_size(1)
        fn main() {
            sink = MAX_LIGHTS;
        }
    "#;

    let overrides = extract_overrides_from_wgsl(source).unwrap();

    let max_lights = overrides.get_by_name("MAX_LIGHTS").unwrap();
    assert!(max_lights.required);
    assert!(max_lights.default_value.is_none());
}

#[test]
fn test_shader_mixed_types() {
    let source = r#"
        override FLAG: bool = false;
        override COUNT: u32 = 10u;
        override OFFSET: i32 = -5i;
        override SCALE: f32 = 1.5;

        var<private> sink_f: f32;

        @compute @workgroup_size(1)
        fn main() {
            if FLAG {
                sink_f = f32(COUNT) * SCALE + f32(OFFSET);
            }
        }
    "#;

    let overrides = extract_overrides_from_wgsl(source).unwrap();
    assert_eq!(overrides.len(), 4);

    assert_eq!(overrides.get_by_name("FLAG").unwrap().ty, OverrideConstantType::Bool);
    assert_eq!(overrides.get_by_name("COUNT").unwrap().ty, OverrideConstantType::U32);
    assert_eq!(overrides.get_by_name("OFFSET").unwrap().ty, OverrideConstantType::I32);
    assert_eq!(overrides.get_by_name("SCALE").unwrap().ty, OverrideConstantType::F32);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_error_invalid_shader_source() {
    let source = "this is not valid wgsl @@@";
    let result = extract_overrides_from_wgsl(source);
    assert!(result.is_err());
}

#[test]
fn test_error_missing_required_constant() {
    let overrides = OverrideConstants::from_infos(vec![
        OverrideConstantInfo::required(Some("REQUIRED".to_string()), None, OverrideConstantType::U32),
    ]);

    let constants = PipelineConstants::new(); // Empty - missing required

    let result = constants.validate(&overrides);
    assert!(result.is_err());

    let err = result.unwrap_err();
    assert!(err.is_missing_required());
}

#[test]
fn test_error_unknown_constant() {
    let overrides = OverrideConstants::from_infos(vec![
        OverrideConstantInfo::with_default(Some("KNOWN".to_string()), None, OverrideConstantType::F32, 1.0),
    ]);

    let constants = PipelineConstants::new().with("UNKNOWN", 42.0);

    let result = constants.validate(&overrides);
    assert!(result.is_err());

    let err = result.unwrap_err();
    assert!(err.is_unknown());
    assert_eq!(err.key(), Some("UNKNOWN"));
}

#[test]
fn test_error_value_out_of_range_bool() {
    let info = OverrideConstantInfo::new(
        Some("FLAG".to_string()),
        None,
        OverrideConstantType::Bool,
        Some(0.0),
    );

    // Valid values for bool: 0.0 and 1.0
    assert!(info.validate_value(0.0).is_ok());
    assert!(info.validate_value(1.0).is_ok());

    // Invalid values
    assert!(info.validate_value(0.5).is_err());
    assert!(info.validate_value(2.0).is_err());
    assert!(info.validate_value(-1.0).is_err());
}

#[test]
fn test_error_value_out_of_range_u32() {
    let info = OverrideConstantInfo::new(
        Some("COUNT".to_string()),
        None,
        OverrideConstantType::U32,
        Some(0.0),
    );

    // Valid values
    assert!(info.validate_value(0.0).is_ok());
    assert!(info.validate_value(100.0).is_ok());
    assert!(info.validate_value(u32::MAX as f64).is_ok());

    // Invalid: negative
    assert!(info.validate_value(-1.0).is_err());
    // Invalid: fractional
    assert!(info.validate_value(1.5).is_err());
}

#[test]
fn test_error_value_out_of_range_i32() {
    let info = OverrideConstantInfo::new(
        Some("OFFSET".to_string()),
        None,
        OverrideConstantType::I32,
        Some(0.0),
    );

    // Valid values
    assert!(info.validate_value(0.0).is_ok());
    assert!(info.validate_value(-100.0).is_ok());
    assert!(info.validate_value(100.0).is_ok());
    assert!(info.validate_value(i32::MIN as f64).is_ok());
    assert!(info.validate_value(i32::MAX as f64).is_ok());

    // Invalid: fractional
    assert!(info.validate_value(1.5).is_err());
}

#[test]
fn test_error_value_out_of_range_f32() {
    let info = OverrideConstantInfo::new(
        Some("SCALE".to_string()),
        None,
        OverrideConstantType::F32,
        Some(1.0),
    );

    // Valid values
    assert!(info.validate_value(0.0).is_ok());
    assert!(info.validate_value(0.5).is_ok());
    assert!(info.validate_value(-1.5).is_ok());
    assert!(info.validate_value(1e30).is_ok());

    // NaN is valid for f32
    assert!(info.validate_value(f64::NAN).is_ok());

    // Infinity is NOT valid
    assert!(info.validate_value(f64::INFINITY).is_err());
    assert!(info.validate_value(f64::NEG_INFINITY).is_err());
}

#[test]
fn test_error_override_error_key_extraction() {
    let err = OverrideError::UnknownConstant { key: "TEST".to_string() };
    assert_eq!(err.key(), Some("TEST"));

    let err = OverrideError::MissingRequired { name: Some("MISSING".to_string()), id: None };
    assert_eq!(err.key(), Some("MISSING"));

    let err = OverrideError::MissingRequired { name: None, id: Some(5) };
    assert_eq!(err.key(), None); // ID doesn't return a key
}

#[test]
fn test_error_override_error_display() {
    let err = OverrideError::UnknownConstant { key: "UNKNOWN".to_string() };
    let display = format!("{}", err);
    assert!(display.contains("unknown"));
    assert!(display.contains("UNKNOWN"));

    let err = OverrideError::MissingRequired { name: Some("REQ".to_string()), id: Some(0) };
    let display = format!("{}", err);
    assert!(display.contains("missing required"));
    assert!(display.contains("REQ"));
    assert!(display.contains("@id(0)"));

    let err = OverrideError::ValueOutOfRange { key: "VAL".to_string(), value: -5.0, expected: "0 to 100" };
    let display = format!("{}", err);
    assert!(display.contains("out of range"));
    assert!(display.contains("-5"));
}

// ============================================================================
// Pipeline Integration Tests
// ============================================================================

#[test]
fn test_pipeline_to_wgpu_format() {
    let constants = PipelineConstants::new()
        .with_f32("SCREEN_WIDTH", 2560.0)
        .with_f32("SCREEN_HEIGHT", 1440.0)
        .with_u32("TILE_SIZE", 32)
        .with_bool("ENABLE_SHADOWS", true);

    let wgpu_map = constants.to_wgpu();

    // Verify it returns HashMap<String, f64>
    assert_eq!(wgpu_map.len(), 4);
    assert_eq!(wgpu_map.get("SCREEN_WIDTH"), Some(&(2560.0_f32 as f64)));
    assert_eq!(wgpu_map.get("ENABLE_SHADOWS"), Some(&1.0));
}

#[test]
fn test_pipeline_validation_success() {
    let source = r#"
        override WIDTH: f32 = 1920.0;
        override HEIGHT: f32 = 1080.0;
        override ENABLED: bool = true;

        var<private> sink: f32;

        @compute @workgroup_size(1)
        fn main() {
            if ENABLED {
                sink = WIDTH * HEIGHT;
            }
        }
    "#;

    let overrides = extract_overrides_from_wgsl(source).unwrap();

    let constants = PipelineConstants::new()
        .with_f32("WIDTH", 2560.0)
        .with_f32("HEIGHT", 1440.0)
        .with_bool("ENABLED", false);

    assert!(constants.validate(&overrides).is_ok());
}

#[test]
fn test_pipeline_validation_partial_override() {
    let source = r#"
        override WIDTH: f32 = 1920.0;
        override HEIGHT: f32 = 1080.0;

        var<private> sink: f32;

        @compute @workgroup_size(1)
        fn main() {
            sink = WIDTH * HEIGHT;
        }
    "#;

    let overrides = extract_overrides_from_wgsl(source).unwrap();

    // Only override WIDTH, HEIGHT uses default
    let constants = PipelineConstants::new().with_f32("WIDTH", 2560.0);

    assert!(constants.validate(&overrides).is_ok());
}

#[test]
fn test_pipeline_validation_with_required() {
    let source = r#"
        override REQUIRED_VALUE: u32;
        override OPTIONAL_VALUE: f32 = 1.0;

        var<private> sink: f32;

        @compute @workgroup_size(1)
        fn main() {
            sink = f32(REQUIRED_VALUE) + OPTIONAL_VALUE;
        }
    "#;

    let overrides = extract_overrides_from_wgsl(source).unwrap();

    // Must provide required constant
    let constants = PipelineConstants::new().with_u32("REQUIRED_VALUE", 100);

    assert!(constants.validate(&overrides).is_ok());
}

#[test]
fn test_pipeline_builder_fluent_api() {
    let constants = PipelineConstants::new()
        .with_f32("A", 1.0)
        .with_u32("B", 2)
        .with_i32("C", -3)
        .with_bool("D", true)
        .with("E", 5.0);

    assert_eq!(constants.len(), 5);
}

#[test]
fn test_pipeline_validate_against_extracted_overrides() {
    // Full integration: parse shader -> extract overrides -> set constants -> validate
    let source = r#"
        @id(0) override SCREEN_WIDTH: f32 = 1920.0;
        @id(1) override SCREEN_HEIGHT: f32 = 1080.0;
        @id(2) override MAX_ITERATIONS: u32;

        var<private> sink: f32;

        @compute @workgroup_size(1)
        fn main() {
            sink = SCREEN_WIDTH * f32(MAX_ITERATIONS);
        }
    "#;

    let overrides = extract_overrides_from_wgsl(source).expect("Parse should succeed");

    let constants = PipelineConstants::new()
        .with_f32("SCREEN_WIDTH", 3840.0)
        .with_f32("SCREEN_HEIGHT", 2160.0)
        .with_u32("MAX_ITERATIONS", 1000);

    let result = constants.validate(&overrides);
    assert!(result.is_ok(), "Validation failed: {:?}", result.err());
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn test_edge_empty_shader_no_overrides() {
    let source = r#"
        @compute @workgroup_size(1)
        fn main() {}
    "#;

    let overrides = extract_overrides_from_wgsl(source).unwrap();
    assert!(overrides.is_empty());
}

#[test]
fn test_edge_many_overrides() {
    // Shader with 10+ override constants
    let source = r#"
        override A: f32 = 1.0;
        override B: f32 = 2.0;
        override C: f32 = 3.0;
        override D: f32 = 4.0;
        override E: f32 = 5.0;
        override F: f32 = 6.0;
        override G: f32 = 7.0;
        override H: f32 = 8.0;
        override I: f32 = 9.0;
        override J: f32 = 10.0;
        override K: f32 = 11.0;
        override L: f32 = 12.0;

        var<private> sink: f32;

        @compute @workgroup_size(1)
        fn main() {
            sink = A + B + C + D + E + F + G + H + I + J + K + L;
        }
    "#;

    let overrides = extract_overrides_from_wgsl(source).unwrap();
    assert_eq!(overrides.len(), 12);
}

#[test]
fn test_edge_long_constant_names() {
    let source = r#"
        override THIS_IS_A_VERY_LONG_CONSTANT_NAME_THAT_SHOULD_STILL_WORK: f32 = 1.0;

        var<private> sink: f32;

        @compute @workgroup_size(1)
        fn main() {
            sink = THIS_IS_A_VERY_LONG_CONSTANT_NAME_THAT_SHOULD_STILL_WORK;
        }
    "#;

    let overrides = extract_overrides_from_wgsl(source).unwrap();
    assert_eq!(overrides.len(), 1);
    assert!(overrides.get_by_name("THIS_IS_A_VERY_LONG_CONSTANT_NAME_THAT_SHOULD_STILL_WORK").is_some());
}

#[test]
fn test_edge_underscore_names() {
    // Note: WGSL reserves identifiers starting with __ (double underscore)
    // Single underscore prefix is allowed
    let source = r#"
        override _PRIVATE_LIKE: f32 = 1.0;
        override NAME_WITH_UNDERSCORES: f32 = 2.0;
        override TRAILING_: f32 = 3.0;

        var<private> sink: f32;

        @compute @workgroup_size(1)
        fn main() {
            sink = _PRIVATE_LIKE + NAME_WITH_UNDERSCORES + TRAILING_;
        }
    "#;

    let overrides = extract_overrides_from_wgsl(source).unwrap();
    assert_eq!(overrides.len(), 3);
}

#[test]
fn test_edge_double_underscore_reserved() {
    // WGSL reserves identifiers starting with __ (double underscore)
    let source = r#"
        override __RESERVED: f32 = 1.0;

        var<private> sink: f32;

        @compute @workgroup_size(1)
        fn main() {
            sink = __RESERVED;
        }
    "#;

    let result = extract_overrides_from_wgsl(source);
    assert!(result.is_err(), "Identifiers starting with __ should be rejected");
}

#[test]
fn test_edge_numeric_id_only() {
    let source = r#"
        @id(100) override _unnamed: f32 = 1.0;

        var<private> sink: f32;

        @compute @workgroup_size(1)
        fn main() {
            sink = _unnamed;
        }
    "#;

    let overrides = extract_overrides_from_wgsl(source).unwrap();
    let info = overrides.get_by_id(100);
    assert!(info.is_some());
}

#[test]
fn test_edge_i32_boundary_values() {
    let overrides = OverrideConstants::from_infos(vec![
        OverrideConstantInfo::with_default(Some("I".to_string()), None, OverrideConstantType::I32, 0.0),
    ]);

    let mut constants = PipelineConstants::new();

    // Test i32::MIN
    constants.set_i32("I", i32::MIN);
    assert!(constants.validate(&overrides).is_ok());

    // Test i32::MAX
    constants.set_i32("I", i32::MAX);
    assert!(constants.validate(&overrides).is_ok());
}

#[test]
fn test_edge_u32_boundary_values() {
    let overrides = OverrideConstants::from_infos(vec![
        OverrideConstantInfo::with_default(Some("U".to_string()), None, OverrideConstantType::U32, 0.0),
    ]);

    let mut constants = PipelineConstants::new();

    // Test 0
    constants.set_u32("U", 0);
    assert!(constants.validate(&overrides).is_ok());

    // Test u32::MAX
    constants.set_u32("U", u32::MAX);
    assert!(constants.validate(&overrides).is_ok());
}

#[test]
fn test_edge_constant_overwrite() {
    let mut constants = PipelineConstants::new();
    constants.set("VALUE", 1.0);
    constants.set("VALUE", 2.0);

    assert_eq!(constants.get("VALUE"), Some(2.0));
    assert_eq!(constants.len(), 1); // Still only one entry
}

#[test]
fn test_edge_empty_validation() {
    let overrides = OverrideConstants::new();
    let constants = PipelineConstants::new();

    // No required constants, no values - should pass
    assert!(constants.validate(&overrides).is_ok());
}

#[test]
fn test_edge_all_optional_no_values() {
    let overrides = OverrideConstants::from_infos(vec![
        OverrideConstantInfo::with_default(Some("A".to_string()), None, OverrideConstantType::F32, 1.0),
        OverrideConstantInfo::with_default(Some("B".to_string()), None, OverrideConstantType::F32, 2.0),
    ]);

    let constants = PipelineConstants::new(); // Empty

    // All have defaults, so validation should pass
    assert!(constants.validate(&overrides).is_ok());
}

#[test]
fn test_edge_validate_required_by_id() {
    let overrides = OverrideConstants::from_infos(vec![
        OverrideConstantInfo::required(None, Some(42), OverrideConstantType::U32), // ID-only required
    ]);

    // Provide value using stringified ID
    let constants = PipelineConstants::new().with_u32("42", 100);

    assert!(constants.validate(&overrides).is_ok());
}

// ============================================================================
// Performance Characteristics Tests
// ============================================================================

#[test]
fn test_perf_large_shader_parsing() {
    // Generate a shader with many constants
    let mut source = String::new();
    for i in 0..50 {
        source.push_str(&format!("override CONST_{}: f32 = {}.0;\n", i, i));
    }
    source.push_str("var<private> sink: f32;\n");
    source.push_str("@compute @workgroup_size(1) fn main() { sink = CONST_0; }\n");

    let start = std::time::Instant::now();
    let result = extract_overrides_from_wgsl(&source);
    let duration = start.elapsed();

    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 50);

    // Should complete in reasonable time (under 1 second for 50 constants)
    assert!(duration.as_secs() < 1, "Parsing took too long: {:?}", duration);
}

#[test]
fn test_perf_lookup_performance() {
    let mut overrides = OverrideConstants::new();
    for i in 0..100 {
        overrides.add(OverrideConstantInfo::new(
            Some(format!("CONST_{}", i)),
            Some(i as u32),
            OverrideConstantType::F32,
            Some(i as f64),
        ));
    }

    let start = std::time::Instant::now();
    for i in 0..1000 {
        let name = format!("CONST_{}", i % 100);
        let _ = overrides.get_by_name(&name);
        let _ = overrides.get_by_id((i % 100) as u32);
    }
    let duration = start.elapsed();

    // 1000 lookups should be fast (under 10ms)
    assert!(duration.as_millis() < 10, "Lookups took too long: {:?}", duration);
}

#[test]
fn test_perf_to_wgpu_allocation() {
    let constants = PipelineConstants::new()
        .with("A", 1.0)
        .with("B", 2.0)
        .with("C", 3.0)
        .with("D", 4.0)
        .with("E", 5.0);

    // Call to_wgpu multiple times - should not cause issues
    for _ in 0..100 {
        let map = constants.to_wgpu();
        assert_eq!(map.len(), 5);
    }
}

#[test]
fn test_perf_repeated_validation() {
    let overrides = OverrideConstants::from_infos(vec![
        OverrideConstantInfo::with_default(Some("A".to_string()), None, OverrideConstantType::F32, 1.0),
        OverrideConstantInfo::with_default(Some("B".to_string()), None, OverrideConstantType::U32, 2.0),
        OverrideConstantInfo::required(Some("C".to_string()), None, OverrideConstantType::Bool),
    ]);

    let constants = PipelineConstants::new()
        .with_f32("A", 10.0)
        .with_u32("B", 20)
        .with_bool("C", true);

    let start = std::time::Instant::now();
    for _ in 0..1000 {
        let result = constants.validate(&overrides);
        assert!(result.is_ok());
    }
    let duration = start.elapsed();

    // 1000 validations should be fast
    assert!(duration.as_millis() < 50, "Validations took too long: {:?}", duration);
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

#[test]
fn test_thread_safety_override_constant_type_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<OverrideConstantType>();
}

#[test]
fn test_thread_safety_override_constant_info_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<OverrideConstantInfo>();
}

#[test]
fn test_thread_safety_override_constants_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<OverrideConstants>();
}

#[test]
fn test_thread_safety_pipeline_constants_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<PipelineConstants>();
}

#[test]
fn test_thread_safety_override_error_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<OverrideError>();
}

// ============================================================================
// Clone and Debug Tests
// ============================================================================

#[test]
fn test_traits_override_constant_info_clone() {
    let info = OverrideConstantInfo::new(
        Some("TEST".to_string()),
        Some(0),
        OverrideConstantType::F32,
        Some(1.0),
    );
    let cloned = info.clone();
    assert_eq!(info, cloned);
}

#[test]
fn test_traits_override_constants_clone() {
    let infos = vec![
        OverrideConstantInfo::new(Some("A".to_string()), None, OverrideConstantType::F32, Some(1.0)),
    ];
    let overrides = OverrideConstants::from_infos(infos);
    let cloned = overrides.clone();
    assert_eq!(cloned.len(), overrides.len());
}

#[test]
fn test_traits_pipeline_constants_clone() {
    let constants = PipelineConstants::new().with("A", 1.0);
    let cloned = constants.clone();
    assert_eq!(cloned.get("A"), constants.get("A"));
}

#[test]
fn test_traits_override_error_clone_eq() {
    let err1 = OverrideError::UnknownConstant { key: "X".to_string() };
    let err2 = err1.clone();
    assert_eq!(err1, err2);
}

#[test]
fn test_traits_debug_implementations() {
    let ty = OverrideConstantType::Bool;
    assert!(!format!("{:?}", ty).is_empty());

    let info = OverrideConstantInfo::new(Some("X".to_string()), None, OverrideConstantType::F32, None);
    assert!(!format!("{:?}", info).is_empty());

    let overrides = OverrideConstants::new();
    assert!(!format!("{:?}", overrides).is_empty());

    let constants = PipelineConstants::new();
    assert!(!format!("{:?}", constants).is_empty());

    let err = OverrideError::UnknownConstant { key: "X".to_string() };
    assert!(!format!("{:?}", err).is_empty());
}

// ============================================================================
// Error Trait Implementation Tests
// ============================================================================

#[test]
fn test_error_trait_implementation() {
    let err: Box<dyn std::error::Error> = Box::new(OverrideError::UnknownConstant {
        key: "TEST".to_string(),
    });

    // Should implement Error trait
    let _ = err.to_string();
}

// ============================================================================
// Cross-Module Integration Patterns
// ============================================================================

#[test]
fn test_integration_typical_render_pipeline_flow() {
    // Simulate typical engine usage: parse shader, extract overrides, set values, validate

    // 1. Shader source (would normally be loaded from file)
    let shader_source = r#"
        // Screen resolution overrides
        @id(0) override SCREEN_WIDTH: f32 = 1920.0;
        @id(1) override SCREEN_HEIGHT: f32 = 1080.0;

        // Feature toggles
        @id(10) override ENABLE_MSAA: bool = true;
        @id(11) override MSAA_SAMPLES: u32 = 4u;

        // Required: must be set by application
        @id(20) override MAX_DRAW_CALLS: u32;

        var<private> sink: f32;

        @compute @workgroup_size(1)
        fn main() {
            if ENABLE_MSAA {
                sink = SCREEN_WIDTH * f32(MSAA_SAMPLES);
            }
        }
    "#;

    // 2. Extract overrides from shader
    let overrides = extract_overrides_from_wgsl(shader_source)
        .expect("Shader should parse successfully");

    // 3. Verify expected overrides
    assert_eq!(overrides.len(), 5);
    assert!(overrides.get_by_id(0).is_some());
    assert!(overrides.get_by_id(20).unwrap().required);

    // 4. Set pipeline constants based on application state
    let constants = PipelineConstants::new()
        .with_f32("SCREEN_WIDTH", 3840.0)   // 4K resolution
        .with_f32("SCREEN_HEIGHT", 2160.0)
        .with_bool("ENABLE_MSAA", true)
        .with_u32("MSAA_SAMPLES", 8)        // Override default 4x to 8x
        .with_u32("MAX_DRAW_CALLS", 10000); // Required value

    // 5. Validate before creating pipeline
    let validation = constants.validate(&overrides);
    assert!(validation.is_ok(), "Validation failed: {:?}", validation.err());

    // 6. Convert to wgpu format
    let wgpu_constants = constants.to_wgpu();
    assert_eq!(wgpu_constants.len(), 5);
}

#[test]
fn test_integration_compute_shader_dispatch_pattern() {
    let shader_source = r#"
        override WORKGROUP_SIZE_X: u32 = 64u;
        override WORKGROUP_SIZE_Y: u32 = 1u;
        override WORKGROUP_SIZE_Z: u32 = 1u;

        var<private> sink: u32;

        @compute @workgroup_size(1)  // Would use overrides in real scenario
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            sink = id.x;
        }
    "#;

    let overrides = extract_overrides_from_wgsl(shader_source).unwrap();

    // Configure for different hardware capabilities
    let nvidia_constants = PipelineConstants::new()
        .with_u32("WORKGROUP_SIZE_X", 256);

    let mobile_constants = PipelineConstants::new()
        .with_u32("WORKGROUP_SIZE_X", 32);

    assert!(nvidia_constants.validate(&overrides).is_ok());
    assert!(mobile_constants.validate(&overrides).is_ok());
}

#[test]
fn test_integration_specialization_variant_pattern() {
    let shader_source = r#"
        override QUALITY_LEVEL: u32 = 1u;  // 0=low, 1=medium, 2=high, 3=ultra
        override ENABLE_RAYTRACING: bool = false;
        override MAX_SHADOW_CASCADES: u32 = 4u;

        var<private> sink: u32;

        @compute @workgroup_size(1)
        fn main() {
            sink = QUALITY_LEVEL + MAX_SHADOW_CASCADES;
        }
    "#;

    let overrides = extract_overrides_from_wgsl(shader_source).unwrap();

    // Create different quality presets
    let low_quality = PipelineConstants::new()
        .with_u32("QUALITY_LEVEL", 0)
        .with_bool("ENABLE_RAYTRACING", false)
        .with_u32("MAX_SHADOW_CASCADES", 2);

    let ultra_quality = PipelineConstants::new()
        .with_u32("QUALITY_LEVEL", 3)
        .with_bool("ENABLE_RAYTRACING", true)
        .with_u32("MAX_SHADOW_CASCADES", 8);

    assert!(low_quality.validate(&overrides).is_ok());
    assert!(ultra_quality.validate(&overrides).is_ok());
}
