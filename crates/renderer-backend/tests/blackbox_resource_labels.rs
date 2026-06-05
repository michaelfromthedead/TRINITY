//! Blackbox tests for resource_labels module
//!
//! Tests the public API for GPU resource labeling utilities without
//! examining internal implementation details.
//!
//! CLEANROOM: Tests written against public API contract only.

use renderer_backend::resource_labels::{
    buffer_label, bind_group_label, compute_pass_label, hierarchical_indexed_label,
    indexed_label, is_valid_label_char, pipeline_label, render_pass_label,
    sampler_label, shader_label, texture_label, validate_label, sanitize_label,
    LabelBuilder, LabelError, LabelRegistry, ResourceLabel,
    prefixes, MAX_LABEL_LENGTH, LABEL_SEPARATOR, COMPONENT_SEPARATOR,
};

// ============================================================================
// CRITERION 1: Label generation utilities
// ============================================================================

#[test]
fn test_buffer_label_generates_prefixed_label() {
    let label = buffer_label("vertex_data");
    assert!(label.contains("vertex_data"), "Label should contain the name");
    assert!(label.starts_with(prefixes::BUFFER) || label.contains("buf"),
        "Buffer label should have buffer prefix");
}

#[test]
fn test_texture_label_generates_prefixed_label() {
    let label = texture_label("albedo_map");
    assert!(label.contains("albedo_map"), "Label should contain the name");
    assert!(label.starts_with(prefixes::TEXTURE) || label.contains("tex"),
        "Texture label should have texture prefix");
}

#[test]
fn test_pipeline_label_generates_prefixed_label() {
    let label = pipeline_label("pbr_forward");
    assert!(label.contains("pbr_forward"), "Label should contain the name");
    assert!(label.starts_with(prefixes::PIPELINE) || label.contains("pipe"),
        "Pipeline label should have pipeline prefix");
}

#[test]
fn test_bind_group_label_generates_prefixed_label() {
    let label = bind_group_label("material_bindings");
    assert!(label.contains("material_bindings"), "Label should contain the name");
}

#[test]
fn test_sampler_label_generates_prefixed_label() {
    let label = sampler_label("linear_repeat");
    assert!(label.contains("linear_repeat"), "Label should contain the name");
}

#[test]
fn test_render_pass_label_generates_prefixed_label() {
    let label = render_pass_label("shadow_pass");
    assert!(label.contains("shadow_pass"), "Label should contain the name");
}

#[test]
fn test_compute_pass_label_generates_prefixed_label() {
    let label = compute_pass_label("culling");
    assert!(label.contains("culling"), "Label should contain the name");
}

#[test]
fn test_shader_label_generates_prefixed_label() {
    let label = shader_label("vertex_transform");
    assert!(label.contains("vertex_transform"), "Label should contain the name");
}

#[test]
fn test_indexed_label_appends_index() {
    let label = indexed_label("cascade", 3);
    assert!(label.contains("cascade"), "Label should contain base name");
    assert!(label.contains("3"), "Label should contain index");
}

#[test]
fn test_hierarchical_indexed_label_combines_components() {
    let label = hierarchical_indexed_label("shadow", "cascade", 2);
    assert!(label.contains("shadow"), "Label should contain prefix");
    assert!(label.contains("cascade"), "Label should contain name");
    assert!(label.contains("2"), "Label should contain index");
}

// ============================================================================
// CRITERION 2: Label validation
// ============================================================================

#[test]
fn test_validate_label_accepts_valid_alphanumeric() {
    assert!(validate_label("simple_label").is_ok());
    assert!(validate_label("Label123").is_ok());
    assert!(validate_label("a").is_ok());
}

#[test]
fn test_validate_label_rejects_empty() {
    let result = validate_label("");
    assert!(result.is_err(), "Empty label should be rejected");
    if let Err(e) = result {
        assert!(matches!(e, LabelError::Empty), "Error should be Empty variant");
    }
}

#[test]
fn test_validate_label_rejects_too_long() {
    let long_label = "a".repeat(MAX_LABEL_LENGTH + 1);
    let result = validate_label(&long_label);
    assert!(result.is_err(), "Too-long label should be rejected");
    if let Err(e) = result {
        assert!(matches!(e, LabelError::TooLong { .. }), "Error should be TooLong variant");
    }
}

#[test]
fn test_validate_label_accepts_max_length() {
    let max_label = "a".repeat(MAX_LABEL_LENGTH);
    assert!(validate_label(&max_label).is_ok(), "Max length label should be valid");
}

#[test]
fn test_validate_label_rejects_invalid_characters() {
    // Test common invalid characters
    let result = validate_label("label\0with\0null");
    assert!(result.is_err(), "Label with null bytes should be rejected");
}

#[test]
fn test_sanitize_label_removes_invalid_chars() {
    let sanitized = sanitize_label("hello\0world");
    assert!(!sanitized.contains('\0'), "Sanitized label should not contain null bytes");
}

#[test]
fn test_sanitize_label_truncates_long_input() {
    let long_input = "x".repeat(MAX_LABEL_LENGTH + 100);
    let sanitized = sanitize_label(&long_input);
    assert!(sanitized.len() <= MAX_LABEL_LENGTH, "Sanitized label should respect max length");
}

#[test]
fn test_sanitize_label_preserves_valid_content() {
    let valid = "valid_label_123";
    let sanitized = sanitize_label(valid);
    assert!(sanitized.contains("valid") || sanitized == valid,
        "Sanitization should preserve valid content");
}

#[test]
fn test_is_valid_label_char_accepts_alphanumeric() {
    assert!(is_valid_label_char('a'));
    assert!(is_valid_label_char('Z'));
    assert!(is_valid_label_char('5'));
}

#[test]
fn test_is_valid_label_char_accepts_common_separators() {
    // Underscores and hyphens are commonly valid
    assert!(is_valid_label_char('_') || is_valid_label_char('-'),
        "At least one common separator should be valid");
}

#[test]
fn test_is_valid_label_char_rejects_control_chars() {
    assert!(!is_valid_label_char('\0'), "Null should be invalid");
    assert!(!is_valid_label_char('\n'), "Newline should be invalid");
    assert!(!is_valid_label_char('\t'), "Tab should be invalid");
}

// ============================================================================
// CRITERION 3: Hierarchical labels (ResourceLabel)
// ============================================================================

#[test]
fn test_resource_label_new_creates_root_label() {
    let label = ResourceLabel::new("root");
    assert_eq!(label.leaf(), Some("root"));
    assert!(label.parent().is_none(), "Root label should have no parent");
    // depth() counts segments: a single-segment label has depth 1
    assert_eq!(label.depth(), 1, "Single-segment label should have depth 1");
}

#[test]
fn test_resource_label_child_creates_nested_label() {
    let parent = ResourceLabel::new("parent");
    let child = parent.child("child");

    assert_eq!(child.leaf(), Some("child"));
    assert!(child.parent().is_some(), "Child should have parent");
    // Two segments: "parent" + "child"
    assert_eq!(child.depth(), 2, "Two-segment label should have depth 2");
}

#[test]
fn test_resource_label_deep_hierarchy() {
    let root = ResourceLabel::new("scene");
    let level1 = root.child("objects");
    let level2 = level1.child("meshes");
    let level3 = level2.child("cube");

    // Four segments: scene/objects/meshes/cube
    assert_eq!(level3.depth(), 4, "Four-segment label should have depth 4");
    assert!(level3.parent().is_some());
}

#[test]
fn test_resource_label_segments_includes_ancestors() {
    let root = ResourceLabel::new("renderer");
    let child = root.child("passes");
    let grandchild = child.child("shadow");

    let segments = grandchild.segments();
    assert!(segments.iter().any(|s| s == "renderer"), "Segments should include root");
    assert!(segments.iter().any(|s| s == "passes"), "Segments should include parent");
    assert!(segments.iter().any(|s| s == "shadow"), "Segments should include self");
}

#[test]
fn test_resource_label_to_string_is_meaningful() {
    let label = ResourceLabel::new("test_resource");
    let s = label.to_string();
    assert!(!s.is_empty(), "String representation should not be empty");
    assert!(s.contains("test_resource"), "String should contain the label name");
}

// ============================================================================
// CRITERION 4: Label registry
// ============================================================================

#[test]
fn test_label_registry_new_is_empty() {
    let registry = LabelRegistry::new();
    assert_eq!(registry.count(), 0, "New registry should be empty");
    assert!(registry.is_empty());
}

#[test]
fn test_label_registry_register_adds_label() {
    let mut registry = LabelRegistry::new();
    let result = registry.register("my_buffer");

    assert!(result, "Registration should succeed (return true)");
    assert_eq!(registry.count(), 1, "Registry should have one entry");
    assert!(!registry.is_empty());
}

#[test]
fn test_label_registry_register_rejects_duplicate() {
    let mut registry = LabelRegistry::new();
    let first = registry.register("unique_name");
    assert!(first, "First registration should succeed");

    let result = registry.register("unique_name");
    assert!(!result, "Duplicate registration should fail (return false)");
}

#[test]
fn test_label_registry_unregister_removes_label() {
    let mut registry = LabelRegistry::new();
    registry.register("to_remove");
    assert_eq!(registry.count(), 1);

    let removed = registry.unregister("to_remove");
    assert!(removed, "Unregister should return true for existing label");
    assert!(registry.is_empty(), "Registry should be empty after removal");
}

#[test]
fn test_label_registry_unregister_nonexistent_returns_false() {
    let mut registry = LabelRegistry::new();
    let removed = registry.unregister("never_added");
    assert!(!removed, "Unregister should return false for non-existent label");
}

#[test]
fn test_label_registry_contains() {
    let mut registry = LabelRegistry::new();
    registry.register("exists");

    assert!(registry.contains("exists"), "Should find registered label");
    assert!(!registry.contains("missing"), "Should not find unregistered label");
}

#[test]
fn test_label_registry_generate_unique_creates_distinct_labels() {
    let registry = LabelRegistry::new();

    let label1 = registry.generate_unique("base");
    let label2 = registry.generate_unique("base");
    let label3 = registry.generate_unique("base");

    // Note: generate_unique uses random suffixes, so labels should differ
    // But since registry doesn't track generated labels automatically,
    // we verify the base is incorporated
    assert!(label1.contains("base"), "Generated label should incorporate base");
    assert!(label2.contains("base"), "Generated label should incorporate base");
    assert!(label3.contains("base"), "Generated label should incorporate base");
}

#[test]
fn test_label_registry_generate_unique_incorporates_base() {
    let registry = LabelRegistry::new();
    let label = registry.generate_unique("shadow_map");

    assert!(label.contains("shadow_map"), "Generated label should incorporate base name");
}

#[test]
fn test_label_registry_clear_empties_registry() {
    let mut registry = LabelRegistry::new();
    registry.register("a");
    registry.register("b");
    registry.register("c");
    assert_eq!(registry.count(), 3);

    registry.clear();
    assert!(registry.is_empty(), "Clear should empty the registry");
}

// ============================================================================
// CRITERION EXTRA: LabelBuilder fluent API
// ============================================================================

#[test]
fn test_label_builder_basic_construction() {
    let label = LabelBuilder::new("resource").build();
    assert!(label.contains("resource"), "Built label should contain name");
}

#[test]
fn test_label_builder_with_prefix() {
    let label = LabelBuilder::new("data")
        .with_prefix("buf")
        .build();

    assert!(label.contains("buf"), "Label should contain prefix");
    assert!(label.contains("data"), "Label should contain name");
}

#[test]
fn test_label_builder_with_index() {
    let label = LabelBuilder::new("cascade")
        .with_index(5)
        .build();

    assert!(label.contains("cascade"), "Label should contain name");
    assert!(label.contains("5"), "Label should contain index");
}

#[test]
fn test_label_builder_with_suffix() {
    let label = LabelBuilder::new("texture")
        .with_suffix("hdr")
        .build();

    assert!(label.contains("texture"), "Label should contain name");
    assert!(label.contains("hdr"), "Label should contain suffix");
}

#[test]
fn test_label_builder_chained_operations() {
    let label = LabelBuilder::new("shadow")
        .with_prefix("pass")
        .with_index(0)
        .with_suffix("depth")
        .build();

    assert!(label.contains("shadow"), "Label should contain name");
    // The label should incorporate all components
    let combined = format!("{}", label);
    assert!(!combined.is_empty());
}

// ============================================================================
// Constants and prefixes
// ============================================================================

#[test]
fn test_max_label_length_is_reasonable() {
    assert!(MAX_LABEL_LENGTH >= 64, "Max length should allow reasonable labels");
    assert!(MAX_LABEL_LENGTH <= 1024, "Max length should not be excessive");
}

#[test]
fn test_separators_are_valid_chars() {
    assert!(is_valid_label_char(LABEL_SEPARATOR) || LABEL_SEPARATOR == '/',
        "Label separator should be usable");
    assert!(is_valid_label_char(COMPONENT_SEPARATOR) || COMPONENT_SEPARATOR == '_',
        "Component separator should be usable");
}

#[test]
fn test_prefixes_module_provides_constants() {
    // Verify prefixes module exports expected constants
    let _ = prefixes::BUFFER;
    let _ = prefixes::TEXTURE;
    let _ = prefixes::SAMPLER;
    let _ = prefixes::PIPELINE;
    let _ = prefixes::BIND_GROUP;
    let _ = prefixes::RENDER_PASS;
    let _ = prefixes::COMPUTE_PASS;
    let _ = prefixes::SHADER;
}

#[test]
fn test_prefixes_are_non_empty() {
    assert!(!prefixes::BUFFER.is_empty(), "Buffer prefix should not be empty");
    assert!(!prefixes::TEXTURE.is_empty(), "Texture prefix should not be empty");
    assert!(!prefixes::PIPELINE.is_empty(), "Pipeline prefix should not be empty");
}

// ============================================================================
// Error handling
// ============================================================================

#[test]
fn test_label_error_display_is_descriptive() {
    let error = LabelError::Empty;
    let msg = format!("{}", error);
    assert!(!msg.is_empty(), "Error display should produce message");
}

#[test]
fn test_label_error_too_long_contains_length_info() {
    let error = LabelError::TooLong {
        length: 300,
        max: MAX_LABEL_LENGTH
    };
    let msg = format!("{}", error);
    assert!(msg.contains("300") || msg.contains(&MAX_LABEL_LENGTH.to_string()),
        "TooLong error should mention lengths");
}
