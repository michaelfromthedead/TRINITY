//! Blackbox Integration Tests for Instance Layout (T-WGPU-P3.2.3)
//!
//! Tests the instance_layout module's public API for GPU instancing support.
//! Validates:
//! - InstanceLayoutBuilder fluent API
//! - Transform matrix layouts (4x4 and 3x4)
//! - Color attribute layouts (HDR Float32x4, LDR Unorm8x4)
//! - Instance ID support
//! - Preset functions
//! - Stride and offset calculations
//! - Shader location assignments
//! - VertexStepMode::Instance enforcement

use renderer_backend::render_pipeline::{
    create_instance_layout, is_valid_instance_layout,
    instance_presets as presets, InstanceLayoutBuilder,
    VertexAttributeDescriptor, VertexBufferLayoutDescriptor,
    INSTANCE_COLOR_LOCATION, INSTANCE_CUSTOM_START_LOCATION,
    INSTANCE_ID_LOCATION, INSTANCE_TRANSFORM_LOCATIONS,
    STRIDE_TRANSFORM, STRIDE_TRANSFORM_COLOR_CUSTOM,
    STRIDE_TRANSFORM_COLOR_FLOAT, STRIDE_TRANSFORM_COLOR_PACKED,
};
use wgpu::VertexStepMode;

// =============================================================================
// CATEGORY 1: API Surface Tests
// =============================================================================

#[test]
fn test_api_constants_are_accessible() {
    // Verify all public constants are accessible
    assert_eq!(INSTANCE_TRANSFORM_LOCATIONS.start, 4);
    assert_eq!(INSTANCE_TRANSFORM_LOCATIONS.end, 8);
    assert_eq!(INSTANCE_COLOR_LOCATION, 8);
    assert_eq!(INSTANCE_ID_LOCATION, 9);
    assert_eq!(INSTANCE_CUSTOM_START_LOCATION, 10);
}

#[test]
fn test_api_stride_constants_are_accessible() {
    // Verify stride constants match expected values
    assert_eq!(STRIDE_TRANSFORM, 64);
    assert_eq!(STRIDE_TRANSFORM_COLOR_PACKED, 68);
    assert_eq!(STRIDE_TRANSFORM_COLOR_FLOAT, 80);
    assert_eq!(STRIDE_TRANSFORM_COLOR_CUSTOM, 96);
}

#[test]
fn test_api_instance_layout_builder_is_constructible() {
    // Verify builder can be constructed
    let builder = InstanceLayoutBuilder::new(4);
    assert_eq!(builder.current_stride(), 0);
    assert_eq!(builder.next_location(), 4);
}

#[test]
fn test_api_instance_layout_builder_default_is_constructible() {
    // Verify default builder uses standard transform start
    let builder = InstanceLayoutBuilder::default();
    assert_eq!(builder.next_location(), INSTANCE_TRANSFORM_LOCATIONS.start);
}

#[test]
fn test_api_all_preset_functions_callable() {
    // Verify all preset functions are accessible and return valid layouts
    let _ = presets::transform_only();
    let _ = presets::transform_color_float();
    let _ = presets::transform_color_packed();
    let _ = presets::transform_color_custom();
    let _ = presets::transform_3x4_color_packed();
    let _ = presets::instance_id_only();
    let _ = presets::transform_id();
}

#[test]
fn test_api_helper_functions_callable() {
    // Verify helper functions are accessible
    let attrs = vec![
        VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x4, 0, 4),
    ];
    let layout = create_instance_layout(&attrs);
    assert!(is_valid_instance_layout(&layout));
}

// =============================================================================
// CATEGORY 2: Real Instance Layout Tests
// =============================================================================

#[test]
fn test_instance_layout_builds_valid_descriptor() {
    let layout = InstanceLayoutBuilder::default()
        .with_transform()
        .build();

    // Verify it's a valid VertexBufferLayoutDescriptor
    assert_eq!(layout.array_stride, 64);
    assert_eq!(layout.step_mode, VertexStepMode::Instance);
    assert!(!layout.attributes.is_empty());
}

#[test]
fn test_instance_layout_attributes_are_vertex_attribute_descriptors() {
    let layout = InstanceLayoutBuilder::default()
        .with_transform()
        .build();

    for attr in &layout.attributes {
        // Verify each attribute has valid format, offset, and location
        assert!(attr.offset < layout.array_stride || layout.array_stride == 0);
        assert!(attr.shader_location >= INSTANCE_TRANSFORM_LOCATIONS.start);
    }
}

#[test]
fn test_instance_layout_can_be_cloned() {
    let layout = InstanceLayoutBuilder::default()
        .with_transform()
        .build();

    let cloned = layout.clone();
    assert_eq!(layout.array_stride, cloned.array_stride);
    assert_eq!(layout.step_mode, cloned.step_mode);
    assert_eq!(layout.attributes.len(), cloned.attributes.len());
}

// =============================================================================
// CATEGORY 3: Transform Matrix Tests
// =============================================================================

#[test]
fn test_transform_4x4_layout_stride() {
    let layout = InstanceLayoutBuilder::default()
        .with_transform()
        .build();

    // 4x4 matrix = 4 rows * 16 bytes = 64 bytes
    assert_eq!(layout.array_stride, 64);
}

#[test]
fn test_transform_4x4_layout_attribute_count() {
    let layout = InstanceLayoutBuilder::default()
        .with_transform()
        .build();

    // 4 rows of vec4
    assert_eq!(layout.attributes.len(), 4);
}

#[test]
fn test_transform_4x4_layout_attribute_formats() {
    let layout = InstanceLayoutBuilder::default()
        .with_transform()
        .build();

    for attr in &layout.attributes {
        assert_eq!(attr.format, wgpu::VertexFormat::Float32x4);
    }
}

#[test]
fn test_transform_4x4_layout_attribute_offsets() {
    let layout = InstanceLayoutBuilder::default()
        .with_transform()
        .build();

    // Each row is 16 bytes apart
    assert_eq!(layout.attributes[0].offset, 0);
    assert_eq!(layout.attributes[1].offset, 16);
    assert_eq!(layout.attributes[2].offset, 32);
    assert_eq!(layout.attributes[3].offset, 48);
}

#[test]
fn test_transform_4x4_layout_shader_locations() {
    let layout = InstanceLayoutBuilder::default()
        .with_transform()
        .build();

    // Locations 4, 5, 6, 7 for the 4 rows
    assert_eq!(layout.attributes[0].shader_location, 4);
    assert_eq!(layout.attributes[1].shader_location, 5);
    assert_eq!(layout.attributes[2].shader_location, 6);
    assert_eq!(layout.attributes[3].shader_location, 7);
}

#[test]
fn test_transform_3x4_layout_stride() {
    let layout = InstanceLayoutBuilder::default()
        .with_transform_3x4()
        .build();

    // 3x4 matrix = 3 rows * 16 bytes = 48 bytes
    assert_eq!(layout.array_stride, 48);
}

#[test]
fn test_transform_3x4_layout_attribute_count() {
    let layout = InstanceLayoutBuilder::default()
        .with_transform_3x4()
        .build();

    // 3 rows of vec4
    assert_eq!(layout.attributes.len(), 3);
}

#[test]
fn test_transform_3x4_layout_attribute_offsets() {
    let layout = InstanceLayoutBuilder::default()
        .with_transform_3x4()
        .build();

    assert_eq!(layout.attributes[0].offset, 0);
    assert_eq!(layout.attributes[1].offset, 16);
    assert_eq!(layout.attributes[2].offset, 32);
}

#[test]
fn test_transform_3x4_layout_shader_locations() {
    let layout = InstanceLayoutBuilder::default()
        .with_transform_3x4()
        .build();

    // Locations 4, 5, 6 for the 3 rows
    assert_eq!(layout.attributes[0].shader_location, 4);
    assert_eq!(layout.attributes[1].shader_location, 5);
    assert_eq!(layout.attributes[2].shader_location, 6);
}

#[test]
fn test_transform_3x4_is_more_compact_than_4x4() {
    let layout_4x4 = InstanceLayoutBuilder::default()
        .with_transform()
        .build();
    let layout_3x4 = InstanceLayoutBuilder::default()
        .with_transform_3x4()
        .build();

    assert!(layout_3x4.array_stride < layout_4x4.array_stride);
    assert_eq!(layout_4x4.array_stride - layout_3x4.array_stride, 16);
}

// =============================================================================
// CATEGORY 4: Color Attribute Tests
// =============================================================================

#[test]
fn test_color_float_hdr_size() {
    let layout = InstanceLayoutBuilder::default()
        .with_color_float()
        .build();

    // Float32x4 = 16 bytes
    assert_eq!(layout.array_stride, 16);
}

#[test]
fn test_color_float_hdr_format() {
    let layout = InstanceLayoutBuilder::default()
        .with_color_float()
        .build();

    assert_eq!(layout.attributes[0].format, wgpu::VertexFormat::Float32x4);
}

#[test]
fn test_color_packed_ldr_size() {
    let layout = InstanceLayoutBuilder::default()
        .with_color_packed()
        .build();

    // Unorm8x4 = 4 bytes
    assert_eq!(layout.array_stride, 4);
}

#[test]
fn test_color_packed_ldr_format() {
    let layout = InstanceLayoutBuilder::default()
        .with_color_packed()
        .build();

    assert_eq!(layout.attributes[0].format, wgpu::VertexFormat::Unorm8x4);
}

#[test]
fn test_color_after_transform_offset() {
    let layout = InstanceLayoutBuilder::default()
        .with_transform()
        .with_color_float()
        .build();

    // Color should start after transform (64 bytes)
    assert_eq!(layout.attributes[4].offset, 64);
}

#[test]
fn test_color_after_transform_location() {
    let layout = InstanceLayoutBuilder::default()
        .with_transform()
        .with_color_float()
        .build();

    // Color location should be 8 (after transform locations 4-7)
    assert_eq!(layout.attributes[4].shader_location, INSTANCE_COLOR_LOCATION);
}

#[test]
fn test_packed_color_is_more_compact_than_float_color() {
    let layout_float = InstanceLayoutBuilder::default()
        .with_transform()
        .with_color_float()
        .build();
    let layout_packed = InstanceLayoutBuilder::default()
        .with_transform()
        .with_color_packed()
        .build();

    assert!(layout_packed.array_stride < layout_float.array_stride);
    // Float: 80, Packed: 68 => difference of 12 bytes
    assert_eq!(layout_float.array_stride - layout_packed.array_stride, 12);
}

// =============================================================================
// CATEGORY 5: Builder Chain Tests
// =============================================================================

#[test]
fn test_builder_chain_transform_color_float() {
    let layout = InstanceLayoutBuilder::default()
        .with_transform()
        .with_color_float()
        .build();

    // 64 + 16 = 80 bytes
    assert_eq!(layout.array_stride, 80);
    assert_eq!(layout.attributes.len(), 5);
}

#[test]
fn test_builder_chain_transform_color_packed() {
    let layout = InstanceLayoutBuilder::default()
        .with_transform()
        .with_color_packed()
        .build();

    // 64 + 4 = 68 bytes
    assert_eq!(layout.array_stride, 68);
    assert_eq!(layout.attributes.len(), 5);
}

#[test]
fn test_builder_chain_transform_color_custom() {
    let layout = InstanceLayoutBuilder::default()
        .with_transform()
        .with_color_float()
        .with_attribute(wgpu::VertexFormat::Float32x4)
        .build();

    // 64 + 16 + 16 = 96 bytes
    assert_eq!(layout.array_stride, 96);
    assert_eq!(layout.attributes.len(), 6);
}

#[test]
fn test_builder_chain_transform_instance_id() {
    let layout = InstanceLayoutBuilder::default()
        .with_transform()
        .with_instance_id()
        .build();

    // 64 + 4 = 68 bytes
    assert_eq!(layout.array_stride, 68);
    assert_eq!(layout.attributes.len(), 5);
}

#[test]
fn test_builder_chain_all_options() {
    let layout = InstanceLayoutBuilder::default()
        .with_transform()
        .with_color_packed()
        .with_instance_id()
        .build();

    // 64 + 4 + 4 = 72 bytes
    assert_eq!(layout.array_stride, 72);
    assert_eq!(layout.attributes.len(), 6);
}

#[test]
fn test_builder_chain_compact_transform_with_color() {
    let layout = InstanceLayoutBuilder::default()
        .with_transform_3x4()
        .with_color_packed()
        .build();

    // 48 + 4 = 52 bytes
    assert_eq!(layout.array_stride, 52);
    assert_eq!(layout.attributes.len(), 4);
}

#[test]
fn test_builder_chain_multiple_custom_attributes() {
    let layout = InstanceLayoutBuilder::default()
        .with_transform()
        .with_attribute(wgpu::VertexFormat::Float32x2)
        .with_attribute(wgpu::VertexFormat::Float32)
        .build();

    // 64 + 8 + 4 = 76 bytes
    assert_eq!(layout.array_stride, 76);
    assert_eq!(layout.attributes.len(), 6);
}

#[test]
fn test_builder_chain_with_attribute_at() {
    let layout = InstanceLayoutBuilder::new(0)
        .with_attribute_at(wgpu::VertexFormat::Float32x4, 0, 0)
        .with_attribute_at(wgpu::VertexFormat::Float32x4, 32, 1)
        .build();

    // Stride is max(0+16, 32+16) = 48
    assert_eq!(layout.array_stride, 48);
    assert_eq!(layout.attributes.len(), 2);
}

#[test]
fn test_builder_chain_with_explicit_stride() {
    let layout = InstanceLayoutBuilder::default()
        .with_transform()
        .with_stride(128)
        .build();

    // Explicit stride overrides calculated
    assert_eq!(layout.array_stride, 128);
}

// =============================================================================
// CATEGORY 6: Stride Calculations
// =============================================================================

#[test]
fn test_stride_transform_only() {
    let layout = presets::transform_only();
    assert_eq!(layout.array_stride, STRIDE_TRANSFORM);
    assert_eq!(layout.array_stride, 64);
}

#[test]
fn test_stride_transform_color_packed() {
    let layout = presets::transform_color_packed();
    assert_eq!(layout.array_stride, STRIDE_TRANSFORM_COLOR_PACKED);
    assert_eq!(layout.array_stride, 68);
}

#[test]
fn test_stride_transform_color_float() {
    let layout = presets::transform_color_float();
    assert_eq!(layout.array_stride, STRIDE_TRANSFORM_COLOR_FLOAT);
    assert_eq!(layout.array_stride, 80);
}

#[test]
fn test_stride_transform_color_custom() {
    let layout = presets::transform_color_custom();
    assert_eq!(layout.array_stride, STRIDE_TRANSFORM_COLOR_CUSTOM);
    assert_eq!(layout.array_stride, 96);
}

#[test]
fn test_stride_instance_id_only() {
    let layout = presets::instance_id_only();
    assert_eq!(layout.array_stride, 4);
}

#[test]
fn test_stride_compact_transform_color() {
    let layout = presets::transform_3x4_color_packed();
    assert_eq!(layout.array_stride, 52);
}

#[test]
fn test_stride_accumulation_is_correct() {
    let builder = InstanceLayoutBuilder::default();
    assert_eq!(builder.current_stride(), 0);

    let builder = builder.with_transform();
    assert_eq!(builder.current_stride(), 64);

    let builder = builder.with_color_float();
    assert_eq!(builder.current_stride(), 80);

    let builder = builder.with_instance_id();
    assert_eq!(builder.current_stride(), 84);
}

// =============================================================================
// CATEGORY 7: Offset Calculations
// =============================================================================

#[test]
fn test_offset_transform_rows_sequential() {
    let layout = presets::transform_only();

    for (i, attr) in layout.attributes.iter().enumerate() {
        assert_eq!(attr.offset, (i * 16) as u64);
    }
}

#[test]
fn test_offset_color_after_transform() {
    let layout = presets::transform_color_float();

    // First 4 attributes are transform
    for i in 0..4 {
        assert_eq!(layout.attributes[i].offset, (i * 16) as u64);
    }
    // Color is at offset 64
    assert_eq!(layout.attributes[4].offset, 64);
}

#[test]
fn test_offset_custom_attributes_sequential() {
    let layout = InstanceLayoutBuilder::default()
        .with_transform()
        .with_color_packed()
        .with_instance_id()
        .build();

    // Transform: 0, 16, 32, 48
    // Color: 64
    // ID: 68
    assert_eq!(layout.attributes[4].offset, 64);
    assert_eq!(layout.attributes[5].offset, 68);
}

#[test]
fn test_offset_gaps_with_attribute_at() {
    let layout = InstanceLayoutBuilder::new(0)
        .with_attribute_at(wgpu::VertexFormat::Float32x4, 0, 0)
        .with_attribute_at(wgpu::VertexFormat::Float32x4, 64, 1)
        .build();

    assert_eq!(layout.attributes[0].offset, 0);
    assert_eq!(layout.attributes[1].offset, 64);
    // Stride should be 64 + 16 = 80
    assert_eq!(layout.array_stride, 80);
}

// =============================================================================
// CATEGORY 8: Shader Location Tests
// =============================================================================

#[test]
fn test_shader_location_transform_uses_4_to_7() {
    let layout = presets::transform_only();

    assert_eq!(layout.attributes[0].shader_location, 4);
    assert_eq!(layout.attributes[1].shader_location, 5);
    assert_eq!(layout.attributes[2].shader_location, 6);
    assert_eq!(layout.attributes[3].shader_location, 7);
}

#[test]
fn test_shader_location_color_uses_8() {
    let layout = presets::transform_color_float();
    assert_eq!(layout.attributes[4].shader_location, 8);
}

#[test]
fn test_shader_location_id_uses_9_after_color() {
    let layout = InstanceLayoutBuilder::default()
        .with_transform()
        .with_color_float()
        .with_instance_id()
        .build();

    assert_eq!(layout.attributes[5].shader_location, 9);
}

#[test]
fn test_shader_location_custom_start() {
    let layout = InstanceLayoutBuilder::default()
        .with_transform()
        .with_color_float()
        .with_attribute(wgpu::VertexFormat::Float32x4)
        .build();

    // Custom attribute should be at location 9
    assert_eq!(layout.attributes[5].shader_location, 9);
}

#[test]
fn test_shader_locations_no_overlap() {
    let layout = InstanceLayoutBuilder::default()
        .with_transform()
        .with_color_packed()
        .with_instance_id()
        .with_attribute(wgpu::VertexFormat::Float32x2)
        .build();

    let mut locations: Vec<u32> = layout.attributes.iter()
        .map(|a| a.shader_location)
        .collect();
    locations.sort();

    // Check for duplicates
    for i in 1..locations.len() {
        assert_ne!(locations[i], locations[i-1], "Duplicate shader location found");
    }
}

#[test]
fn test_shader_location_custom_start_location() {
    let layout = InstanceLayoutBuilder::new(10)
        .with_color_float()
        .build();

    assert_eq!(layout.attributes[0].shader_location, 10);
}

#[test]
fn test_shader_location_tracking() {
    let builder = InstanceLayoutBuilder::default();
    assert_eq!(builder.next_location(), 4);

    let builder = builder.with_transform();
    assert_eq!(builder.next_location(), 8);

    let builder = builder.with_color_float();
    assert_eq!(builder.next_location(), 9);
}

// =============================================================================
// CATEGORY 9: Step Mode Tests
// =============================================================================

#[test]
fn test_step_mode_preset_transform_only() {
    let layout = presets::transform_only();
    assert_eq!(layout.step_mode, VertexStepMode::Instance);
}

#[test]
fn test_step_mode_preset_transform_color_float() {
    let layout = presets::transform_color_float();
    assert_eq!(layout.step_mode, VertexStepMode::Instance);
}

#[test]
fn test_step_mode_preset_transform_color_packed() {
    let layout = presets::transform_color_packed();
    assert_eq!(layout.step_mode, VertexStepMode::Instance);
}

#[test]
fn test_step_mode_preset_transform_color_custom() {
    let layout = presets::transform_color_custom();
    assert_eq!(layout.step_mode, VertexStepMode::Instance);
}

#[test]
fn test_step_mode_preset_transform_3x4_color() {
    let layout = presets::transform_3x4_color_packed();
    assert_eq!(layout.step_mode, VertexStepMode::Instance);
}

#[test]
fn test_step_mode_preset_instance_id_only() {
    let layout = presets::instance_id_only();
    assert_eq!(layout.step_mode, VertexStepMode::Instance);
}

#[test]
fn test_step_mode_preset_transform_id() {
    let layout = presets::transform_id();
    assert_eq!(layout.step_mode, VertexStepMode::Instance);
}

#[test]
fn test_step_mode_builder_output() {
    let layout = InstanceLayoutBuilder::default()
        .with_transform()
        .with_color_float()
        .build();
    assert_eq!(layout.step_mode, VertexStepMode::Instance);
}

#[test]
fn test_step_mode_create_instance_layout() {
    let attrs = vec![
        VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x4, 0, 4),
    ];
    let layout = create_instance_layout(&attrs);
    assert_eq!(layout.step_mode, VertexStepMode::Instance);
}

#[test]
fn test_is_valid_instance_layout_returns_true_for_instance() {
    let layout = presets::transform_only();
    assert!(is_valid_instance_layout(&layout));
}

#[test]
fn test_is_valid_instance_layout_returns_false_for_vertex() {
    let layout = VertexBufferLayoutDescriptor::per_vertex(32);
    assert!(!is_valid_instance_layout(&layout));
}

// =============================================================================
// CATEGORY 10: Preset Tests
// =============================================================================

#[test]
fn test_preset_transform_only_complete() {
    let layout = presets::transform_only();

    assert_eq!(layout.array_stride, 64);
    assert_eq!(layout.step_mode, VertexStepMode::Instance);
    assert_eq!(layout.attributes.len(), 4);

    for (i, attr) in layout.attributes.iter().enumerate() {
        assert_eq!(attr.format, wgpu::VertexFormat::Float32x4);
        assert_eq!(attr.offset, (i * 16) as u64);
        assert_eq!(attr.shader_location, 4 + i as u32);
    }
}

#[test]
fn test_preset_transform_color_float_complete() {
    let layout = presets::transform_color_float();

    assert_eq!(layout.array_stride, 80);
    assert_eq!(layout.attributes.len(), 5);

    // Color attribute
    assert_eq!(layout.attributes[4].format, wgpu::VertexFormat::Float32x4);
    assert_eq!(layout.attributes[4].offset, 64);
    assert_eq!(layout.attributes[4].shader_location, 8);
}

#[test]
fn test_preset_transform_color_packed_complete() {
    let layout = presets::transform_color_packed();

    assert_eq!(layout.array_stride, 68);
    assert_eq!(layout.attributes.len(), 5);

    // Color attribute
    assert_eq!(layout.attributes[4].format, wgpu::VertexFormat::Unorm8x4);
    assert_eq!(layout.attributes[4].offset, 64);
    assert_eq!(layout.attributes[4].shader_location, 8);
}

#[test]
fn test_preset_transform_color_custom_complete() {
    let layout = presets::transform_color_custom();

    assert_eq!(layout.array_stride, 96);
    assert_eq!(layout.attributes.len(), 6);

    // Color and custom
    assert_eq!(layout.attributes[4].format, wgpu::VertexFormat::Float32x4);
    assert_eq!(layout.attributes[4].offset, 64);
    assert_eq!(layout.attributes[5].format, wgpu::VertexFormat::Float32x4);
    assert_eq!(layout.attributes[5].offset, 80);
}

#[test]
fn test_preset_transform_3x4_color_packed_complete() {
    let layout = presets::transform_3x4_color_packed();

    assert_eq!(layout.array_stride, 52);
    assert_eq!(layout.attributes.len(), 4);

    // 3 transform rows
    for i in 0..3 {
        assert_eq!(layout.attributes[i].format, wgpu::VertexFormat::Float32x4);
        assert_eq!(layout.attributes[i].offset, (i * 16) as u64);
        assert_eq!(layout.attributes[i].shader_location, 4 + i as u32);
    }

    // Color
    assert_eq!(layout.attributes[3].format, wgpu::VertexFormat::Unorm8x4);
    assert_eq!(layout.attributes[3].offset, 48);
    assert_eq!(layout.attributes[3].shader_location, 7);
}

#[test]
fn test_preset_instance_id_only_complete() {
    let layout = presets::instance_id_only();

    assert_eq!(layout.array_stride, 4);
    assert_eq!(layout.attributes.len(), 1);

    assert_eq!(layout.attributes[0].format, wgpu::VertexFormat::Uint32);
    assert_eq!(layout.attributes[0].offset, 0);
    assert_eq!(layout.attributes[0].shader_location, INSTANCE_ID_LOCATION);
}

#[test]
fn test_preset_transform_id_complete() {
    let layout = presets::transform_id();

    assert_eq!(layout.array_stride, 68);
    assert_eq!(layout.attributes.len(), 5);

    // Instance ID attribute
    assert_eq!(layout.attributes[4].format, wgpu::VertexFormat::Uint32);
    assert_eq!(layout.attributes[4].offset, 64);
    assert_eq!(layout.attributes[4].shader_location, 8);
}

// =============================================================================
// CATEGORY 11: Edge Case Tests
// =============================================================================

#[test]
fn test_empty_builder_produces_valid_layout() {
    let layout = InstanceLayoutBuilder::default().build();

    assert_eq!(layout.array_stride, 0);
    assert!(layout.attributes.is_empty());
    assert_eq!(layout.step_mode, VertexStepMode::Instance);
}

#[test]
fn test_create_instance_layout_empty_attributes() {
    let layout = create_instance_layout(&[]);

    assert_eq!(layout.array_stride, 0);
    assert!(layout.attributes.is_empty());
    assert_eq!(layout.step_mode, VertexStepMode::Instance);
}

#[test]
fn test_builder_with_zero_start_location() {
    let layout = InstanceLayoutBuilder::new(0)
        .with_transform()
        .build();

    assert_eq!(layout.attributes[0].shader_location, 0);
    assert_eq!(layout.attributes[1].shader_location, 1);
    assert_eq!(layout.attributes[2].shader_location, 2);
    assert_eq!(layout.attributes[3].shader_location, 3);
}

#[test]
fn test_builder_with_high_start_location() {
    let layout = InstanceLayoutBuilder::new(12)
        .with_color_float()
        .build();

    assert_eq!(layout.attributes[0].shader_location, 12);
}

#[test]
fn test_single_attribute_layout() {
    let layout = InstanceLayoutBuilder::default()
        .with_instance_id()
        .build();

    assert_eq!(layout.array_stride, 4);
    assert_eq!(layout.attributes.len(), 1);
}

#[test]
fn test_stride_override_smaller_than_content() {
    // This is an edge case - stride smaller than data is unusual but allowed
    let layout = InstanceLayoutBuilder::default()
        .with_transform()
        .with_stride(32)
        .build();

    assert_eq!(layout.array_stride, 32);
}

#[test]
fn test_stride_override_larger_for_alignment() {
    let layout = InstanceLayoutBuilder::default()
        .with_transform()
        .with_stride(256)
        .build();

    assert_eq!(layout.array_stride, 256);
}

// =============================================================================
// CATEGORY 12: Complex Layout Integration Tests
// =============================================================================

#[test]
fn test_complex_layout_transform_color_uv_id() {
    let layout = InstanceLayoutBuilder::default()
        .with_transform()                                    // 64 bytes, loc 4-7
        .with_color_packed()                                 // 4 bytes, loc 8
        .with_attribute(wgpu::VertexFormat::Float32x2)       // 8 bytes, loc 9 (UV offset)
        .with_instance_id()                                  // 4 bytes, loc 10
        .build();

    assert_eq!(layout.array_stride, 80);
    assert_eq!(layout.attributes.len(), 7);

    // Verify all shader locations are sequential and unique
    let expected_locations = [4, 5, 6, 7, 8, 9, 10];
    for (i, loc) in expected_locations.iter().enumerate() {
        assert_eq!(layout.attributes[i].shader_location, *loc);
    }
}

#[test]
fn test_multiple_custom_vec4_attributes() {
    let layout = InstanceLayoutBuilder::default()
        .with_transform()
        .with_attribute(wgpu::VertexFormat::Float32x4)
        .with_attribute(wgpu::VertexFormat::Float32x4)
        .with_attribute(wgpu::VertexFormat::Float32x4)
        .build();

    // 64 + 16 + 16 + 16 = 112 bytes
    assert_eq!(layout.array_stride, 112);
    assert_eq!(layout.attributes.len(), 7);
}

#[test]
fn test_layout_with_mixed_attribute_sizes() {
    let layout = InstanceLayoutBuilder::default()
        .with_transform()                                    // 64 bytes
        .with_attribute(wgpu::VertexFormat::Uint8x4)         // 4 bytes
        .with_attribute(wgpu::VertexFormat::Float16x2)       // 4 bytes
        .with_attribute(wgpu::VertexFormat::Float32)         // 4 bytes
        .with_attribute(wgpu::VertexFormat::Float64x2)       // 16 bytes
        .build();

    assert_eq!(layout.array_stride, 92);
    assert_eq!(layout.attributes.len(), 8);
}

#[test]
fn test_create_instance_layout_calculates_stride() {
    let attrs = vec![
        VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x4, 0, 4),
        VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x4, 16, 5),
        VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x4, 32, 6),
        VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x4, 48, 7),
    ];
    let layout = create_instance_layout(&attrs);

    // Stride is max(48 + 16) = 64
    assert_eq!(layout.array_stride, 64);
    assert_eq!(layout.attributes.len(), 4);
}

#[test]
fn test_create_instance_layout_non_contiguous() {
    let attrs = vec![
        VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x4, 0, 4),
        VertexAttributeDescriptor::new(wgpu::VertexFormat::Uint32, 64, 8),
    ];
    let layout = create_instance_layout(&attrs);

    // Stride is max(0 + 16, 64 + 4) = 68
    assert_eq!(layout.array_stride, 68);
}

// =============================================================================
// CATEGORY 13: Format Verification Tests
// =============================================================================

#[test]
fn test_instance_id_format_is_uint32() {
    let layout = InstanceLayoutBuilder::default()
        .with_instance_id()
        .build();

    assert_eq!(layout.attributes[0].format, wgpu::VertexFormat::Uint32);
}

#[test]
fn test_transform_row_format_is_float32x4() {
    let layout = presets::transform_only();

    for attr in &layout.attributes {
        assert_eq!(attr.format, wgpu::VertexFormat::Float32x4);
    }
}

#[test]
fn test_custom_attribute_preserves_format() {
    let formats = [
        wgpu::VertexFormat::Float32,
        wgpu::VertexFormat::Float32x2,
        wgpu::VertexFormat::Float32x3,
        wgpu::VertexFormat::Uint32x4,
        wgpu::VertexFormat::Sint32x2,
        wgpu::VertexFormat::Unorm8x4,
    ];

    for format in formats {
        let layout = InstanceLayoutBuilder::new(0)
            .with_attribute(format)
            .build();

        assert_eq!(layout.attributes[0].format, format);
    }
}
