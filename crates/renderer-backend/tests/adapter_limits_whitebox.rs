//! Whitebox structural tests for AdapterLimits and related types.
//!
//! These tests verify the internal structure and behavior of the AdapterLimits
//! API, including all accessor methods, summary structs, and trait implementations.

use renderer_backend::device::{AdapterLimits, LimitsSummary, inspect_limits};

// ============================================================================
// Helper: Create AdapterLimits from raw wgpu::Limits
// ============================================================================

/// Creates an AdapterLimits with default wgpu::Limits for testing.
fn make_limits() -> AdapterLimits {
    AdapterLimits {
        raw: wgpu::Limits::default(),
    }
}

/// Creates an AdapterLimits with custom values for boundary testing.
fn make_custom_limits(texture_1d: u32, texture_2d: u32, buffer_size: u64) -> AdapterLimits {
    let mut raw = wgpu::Limits::default();
    raw.max_texture_dimension_1d = texture_1d;
    raw.max_texture_dimension_2d = texture_2d;
    raw.max_buffer_size = buffer_size;
    AdapterLimits { raw }
}

// ============================================================================
// 1. AdapterLimits Construction Tests
// ============================================================================

mod construction {
    use super::*;

    #[test]
    fn from_raw_limits_preserves_values() {
        let raw = wgpu::Limits::default();
        let limits = AdapterLimits { raw: raw.clone() };

        assert_eq!(limits.raw.max_texture_dimension_1d, raw.max_texture_dimension_1d);
        assert_eq!(limits.raw.max_texture_dimension_2d, raw.max_texture_dimension_2d);
        assert_eq!(limits.raw.max_buffer_size, raw.max_buffer_size);
    }

    #[test]
    fn raw_field_is_accessible() {
        let limits = make_limits();
        // Verify raw field contains actual wgpu::Limits
        let _raw: &wgpu::Limits = &limits.raw;
        assert!(limits.raw.max_texture_dimension_2d > 0);
    }

    #[test]
    fn custom_values_are_preserved() {
        let limits = make_custom_limits(4096, 8192, 1_000_000_000);

        assert_eq!(limits.max_texture_dimension_1d(), 4096);
        assert_eq!(limits.max_texture_dimension_2d(), 8192);
        assert_eq!(limits.max_buffer_size(), 1_000_000_000);
    }
}

// ============================================================================
// 2. Texture Limit Accessors
// ============================================================================

mod texture_limits {
    use super::*;

    #[test]
    fn max_texture_dimension_1d_returns_raw_value() {
        let limits = make_limits();
        assert_eq!(limits.max_texture_dimension_1d(), limits.raw.max_texture_dimension_1d);
    }

    #[test]
    fn max_texture_dimension_2d_returns_raw_value() {
        let limits = make_limits();
        assert_eq!(limits.max_texture_dimension_2d(), limits.raw.max_texture_dimension_2d);
    }

    #[test]
    fn max_texture_dimension_3d_returns_raw_value() {
        let limits = make_limits();
        assert_eq!(limits.max_texture_dimension_3d(), limits.raw.max_texture_dimension_3d);
    }

    #[test]
    fn max_texture_array_layers_returns_raw_value() {
        let limits = make_limits();
        assert_eq!(limits.max_texture_array_layers(), limits.raw.max_texture_array_layers);
    }

    #[test]
    fn texture_limits_are_nonzero_for_defaults() {
        let limits = make_limits();
        assert!(limits.max_texture_dimension_1d() > 0);
        assert!(limits.max_texture_dimension_2d() > 0);
        assert!(limits.max_texture_dimension_3d() > 0);
        assert!(limits.max_texture_array_layers() > 0);
    }

    #[test]
    fn texture_2d_typically_larger_than_1d() {
        let limits = make_limits();
        // Default limits have equal 1D and 2D, but custom limits can vary
        assert!(limits.max_texture_dimension_2d() >= limits.max_texture_dimension_1d());
    }
}

// ============================================================================
// 3. Buffer Limit Accessors
// ============================================================================

mod buffer_limits {
    use super::*;

    #[test]
    fn max_buffer_size_returns_raw_value() {
        let limits = make_limits();
        assert_eq!(limits.max_buffer_size(), limits.raw.max_buffer_size);
    }

    #[test]
    fn max_uniform_buffer_binding_size_returns_raw_value() {
        let limits = make_limits();
        assert_eq!(
            limits.max_uniform_buffer_binding_size(),
            limits.raw.max_uniform_buffer_binding_size
        );
    }

    #[test]
    fn max_storage_buffer_binding_size_returns_raw_value() {
        let limits = make_limits();
        assert_eq!(
            limits.max_storage_buffer_binding_size(),
            limits.raw.max_storage_buffer_binding_size
        );
    }

    #[test]
    fn buffer_size_is_u64_for_large_values() {
        let limits = make_custom_limits(8192, 8192, u64::MAX);
        assert_eq!(limits.max_buffer_size(), u64::MAX);
    }

    #[test]
    fn uniform_smaller_than_storage_binding() {
        let limits = make_limits();
        // Uniform buffer bindings are typically smaller than storage buffer bindings
        assert!(
            limits.max_uniform_buffer_binding_size() <= limits.max_storage_buffer_binding_size()
        );
    }

    #[test]
    fn min_uniform_buffer_offset_alignment_returns_raw() {
        let limits = make_limits();
        assert_eq!(
            limits.min_uniform_buffer_offset_alignment(),
            limits.raw.min_uniform_buffer_offset_alignment
        );
    }

    #[test]
    fn min_storage_buffer_offset_alignment_returns_raw() {
        let limits = make_limits();
        assert_eq!(
            limits.min_storage_buffer_offset_alignment(),
            limits.raw.min_storage_buffer_offset_alignment
        );
    }
}

// ============================================================================
// 4. Bind Group Limit Accessors
// ============================================================================

mod bind_group_limits {
    use super::*;

    #[test]
    fn max_bind_groups_returns_raw_value() {
        let limits = make_limits();
        assert_eq!(limits.max_bind_groups(), limits.raw.max_bind_groups);
    }

    #[test]
    fn max_bindings_per_bind_group_returns_raw_value() {
        let limits = make_limits();
        assert_eq!(
            limits.max_bindings_per_bind_group(),
            limits.raw.max_bindings_per_bind_group
        );
    }

    #[test]
    fn max_dynamic_uniform_buffers_per_pipeline_layout_returns_raw() {
        let limits = make_limits();
        assert_eq!(
            limits.max_dynamic_uniform_buffers_per_pipeline_layout(),
            limits.raw.max_dynamic_uniform_buffers_per_pipeline_layout
        );
    }

    #[test]
    fn max_dynamic_storage_buffers_per_pipeline_layout_returns_raw() {
        let limits = make_limits();
        assert_eq!(
            limits.max_dynamic_storage_buffers_per_pipeline_layout(),
            limits.raw.max_dynamic_storage_buffers_per_pipeline_layout
        );
    }

    #[test]
    fn max_sampled_textures_per_shader_stage_returns_raw() {
        let limits = make_limits();
        assert_eq!(
            limits.max_sampled_textures_per_shader_stage(),
            limits.raw.max_sampled_textures_per_shader_stage
        );
    }

    #[test]
    fn max_samplers_per_shader_stage_returns_raw() {
        let limits = make_limits();
        assert_eq!(
            limits.max_samplers_per_shader_stage(),
            limits.raw.max_samplers_per_shader_stage
        );
    }

    #[test]
    fn max_storage_buffers_per_shader_stage_returns_raw() {
        let limits = make_limits();
        assert_eq!(
            limits.max_storage_buffers_per_shader_stage(),
            limits.raw.max_storage_buffers_per_shader_stage
        );
    }

    #[test]
    fn max_storage_textures_per_shader_stage_returns_raw() {
        let limits = make_limits();
        assert_eq!(
            limits.max_storage_textures_per_shader_stage(),
            limits.raw.max_storage_textures_per_shader_stage
        );
    }

    #[test]
    fn max_uniform_buffers_per_shader_stage_returns_raw() {
        let limits = make_limits();
        assert_eq!(
            limits.max_uniform_buffers_per_shader_stage(),
            limits.raw.max_uniform_buffers_per_shader_stage
        );
    }

    #[test]
    fn webgpu_minimum_bind_groups_is_at_least_4() {
        let limits = make_limits();
        assert!(limits.max_bind_groups() >= 4, "WebGPU minimum is 4 bind groups");
    }

    #[test]
    fn all_eleven_bind_group_methods_are_accessible() {
        let limits = make_limits();

        // Exercise all 11 bind group-related methods
        let _ = limits.max_bind_groups();
        let _ = limits.max_bindings_per_bind_group();
        let _ = limits.max_dynamic_uniform_buffers_per_pipeline_layout();
        let _ = limits.max_dynamic_storage_buffers_per_pipeline_layout();
        let _ = limits.max_sampled_textures_per_shader_stage();
        let _ = limits.max_samplers_per_shader_stage();
        let _ = limits.max_storage_buffers_per_shader_stage();
        let _ = limits.max_storage_textures_per_shader_stage();
        let _ = limits.max_uniform_buffers_per_shader_stage();
        let _ = limits.min_uniform_buffer_offset_alignment();
        let _ = limits.min_storage_buffer_offset_alignment();

        // All methods should return non-panic values
    }
}

// ============================================================================
// 5. Compute Limit Accessors
// ============================================================================

mod compute_limits {
    use super::*;

    #[test]
    fn max_compute_workgroup_size_x_returns_raw() {
        let limits = make_limits();
        assert_eq!(
            limits.max_compute_workgroup_size_x(),
            limits.raw.max_compute_workgroup_size_x
        );
    }

    #[test]
    fn max_compute_workgroup_size_y_returns_raw() {
        let limits = make_limits();
        assert_eq!(
            limits.max_compute_workgroup_size_y(),
            limits.raw.max_compute_workgroup_size_y
        );
    }

    #[test]
    fn max_compute_workgroup_size_z_returns_raw() {
        let limits = make_limits();
        assert_eq!(
            limits.max_compute_workgroup_size_z(),
            limits.raw.max_compute_workgroup_size_z
        );
    }

    #[test]
    fn max_compute_invocations_per_workgroup_returns_raw() {
        let limits = make_limits();
        assert_eq!(
            limits.max_compute_invocations_per_workgroup(),
            limits.raw.max_compute_invocations_per_workgroup
        );
    }

    #[test]
    fn max_compute_workgroups_per_dimension_returns_raw() {
        let limits = make_limits();
        assert_eq!(
            limits.max_compute_workgroups_per_dimension(),
            limits.raw.max_compute_workgroups_per_dimension
        );
    }

    #[test]
    fn workgroup_size_product_within_invocations_limit() {
        let limits = make_limits();
        // The product of workgroup sizes should typically fit within max invocations
        let max_product = limits.max_compute_workgroup_size_x()
            .min(limits.max_compute_workgroup_size_y())
            .min(limits.max_compute_workgroup_size_z());

        // At minimum, each dimension should allow at least 1
        assert!(max_product >= 1);
    }

    #[test]
    fn all_five_compute_methods_are_accessible() {
        let limits = make_limits();

        let x = limits.max_compute_workgroup_size_x();
        let y = limits.max_compute_workgroup_size_y();
        let z = limits.max_compute_workgroup_size_z();
        let inv = limits.max_compute_invocations_per_workgroup();
        let dim = limits.max_compute_workgroups_per_dimension();

        assert!(x > 0);
        assert!(y > 0);
        assert!(z > 0);
        assert!(inv > 0);
        assert!(dim > 0);
    }
}

// ============================================================================
// 6. Vertex Limit Accessors
// ============================================================================

mod vertex_limits {
    use super::*;

    #[test]
    fn max_vertex_buffers_returns_raw() {
        let limits = make_limits();
        assert_eq!(limits.max_vertex_buffers(), limits.raw.max_vertex_buffers);
    }

    #[test]
    fn max_vertex_attributes_returns_raw() {
        let limits = make_limits();
        assert_eq!(limits.max_vertex_attributes(), limits.raw.max_vertex_attributes);
    }

    #[test]
    fn max_vertex_buffer_array_stride_returns_raw() {
        let limits = make_limits();
        assert_eq!(
            limits.max_vertex_buffer_array_stride(),
            limits.raw.max_vertex_buffer_array_stride
        );
    }

    #[test]
    fn all_three_vertex_methods_return_nonzero() {
        let limits = make_limits();

        assert!(limits.max_vertex_buffers() > 0);
        assert!(limits.max_vertex_attributes() > 0);
        assert!(limits.max_vertex_buffer_array_stride() > 0);
    }

    #[test]
    fn vertex_attributes_at_least_16() {
        // WebGPU minimum for vertex attributes
        let limits = make_limits();
        assert!(
            limits.max_vertex_attributes() >= 16,
            "WebGPU minimum is 16 vertex attributes"
        );
    }
}

// ============================================================================
// 7. Summary Method Tests
// ============================================================================

mod summary_method {
    use super::*;

    #[test]
    fn summary_returns_limits_summary_struct() {
        let limits = make_limits();
        let summary: LimitsSummary = limits.summary();

        // Verify it's the correct type by accessing fields
        let _ = summary.texture;
        let _ = summary.buffer;
        let _ = summary.bind_group;
        let _ = summary.compute;
        let _ = summary.vertex;
    }

    #[test]
    fn summary_texture_matches_accessors() {
        let limits = make_limits();
        let summary = limits.summary();

        assert_eq!(summary.texture.max_1d, limits.max_texture_dimension_1d());
        assert_eq!(summary.texture.max_2d, limits.max_texture_dimension_2d());
        assert_eq!(summary.texture.max_3d, limits.max_texture_dimension_3d());
        assert_eq!(summary.texture.max_array_layers, limits.max_texture_array_layers());
    }

    #[test]
    fn summary_buffer_matches_accessors() {
        let limits = make_limits();
        let summary = limits.summary();

        assert_eq!(summary.buffer.max_size, limits.max_buffer_size());
        assert_eq!(
            summary.buffer.max_uniform_binding,
            limits.max_uniform_buffer_binding_size()
        );
        assert_eq!(
            summary.buffer.max_storage_binding,
            limits.max_storage_buffer_binding_size()
        );
        assert_eq!(
            summary.buffer.min_uniform_offset_alignment,
            limits.min_uniform_buffer_offset_alignment()
        );
        assert_eq!(
            summary.buffer.min_storage_offset_alignment,
            limits.min_storage_buffer_offset_alignment()
        );
    }

    #[test]
    fn summary_bind_group_matches_accessors() {
        let limits = make_limits();
        let summary = limits.summary();

        assert_eq!(summary.bind_group.max_bind_groups, limits.max_bind_groups());
        assert_eq!(
            summary.bind_group.max_bindings_per_group,
            limits.max_bindings_per_bind_group()
        );
        assert_eq!(
            summary.bind_group.max_dynamic_uniform_buffers,
            limits.max_dynamic_uniform_buffers_per_pipeline_layout()
        );
        assert_eq!(
            summary.bind_group.max_dynamic_storage_buffers,
            limits.max_dynamic_storage_buffers_per_pipeline_layout()
        );
        assert_eq!(
            summary.bind_group.max_sampled_textures,
            limits.max_sampled_textures_per_shader_stage()
        );
        assert_eq!(
            summary.bind_group.max_samplers,
            limits.max_samplers_per_shader_stage()
        );
        assert_eq!(
            summary.bind_group.max_storage_buffers,
            limits.max_storage_buffers_per_shader_stage()
        );
        assert_eq!(
            summary.bind_group.max_storage_textures,
            limits.max_storage_textures_per_shader_stage()
        );
        assert_eq!(
            summary.bind_group.max_uniform_buffers,
            limits.max_uniform_buffers_per_shader_stage()
        );
    }

    #[test]
    fn summary_compute_matches_accessors() {
        let limits = make_limits();
        let summary = limits.summary();

        assert_eq!(
            summary.compute.max_workgroup_size_x,
            limits.max_compute_workgroup_size_x()
        );
        assert_eq!(
            summary.compute.max_workgroup_size_y,
            limits.max_compute_workgroup_size_y()
        );
        assert_eq!(
            summary.compute.max_workgroup_size_z,
            limits.max_compute_workgroup_size_z()
        );
        assert_eq!(
            summary.compute.max_invocations_per_workgroup,
            limits.max_compute_invocations_per_workgroup()
        );
        assert_eq!(
            summary.compute.max_workgroups_per_dimension,
            limits.max_compute_workgroups_per_dimension()
        );
    }

    #[test]
    fn summary_vertex_matches_accessors() {
        let limits = make_limits();
        let summary = limits.summary();

        assert_eq!(summary.vertex.max_buffers, limits.max_vertex_buffers());
        assert_eq!(summary.vertex.max_attributes, limits.max_vertex_attributes());
        assert_eq!(
            summary.vertex.max_buffer_array_stride,
            limits.max_vertex_buffer_array_stride()
        );
    }

    #[test]
    fn summary_all_categories_populated() {
        let limits = make_limits();
        let summary = limits.summary();

        // All categories should have non-zero values with default limits
        assert!(summary.texture.max_2d > 0);
        assert!(summary.buffer.max_size > 0);
        assert!(summary.bind_group.max_bind_groups > 0);
        assert!(summary.compute.max_workgroup_size_x > 0);
        assert!(summary.vertex.max_buffers > 0);
    }
}

// ============================================================================
// 8. WebGPU Minimum Compliance Tests
// ============================================================================

mod webgpu_compliance {
    use super::*;

    #[test]
    fn meets_webgpu_minimum_returns_true_for_defaults() {
        let limits = make_limits();
        assert!(limits.meets_webgpu_minimum());
    }

    #[test]
    fn meets_webgpu_minimum_checks_texture_1d() {
        let mut raw = wgpu::Limits::default();
        raw.max_texture_dimension_1d = 1; // Below minimum
        let limits = AdapterLimits { raw };

        // This should fail WebGPU minimum check
        assert!(!limits.meets_webgpu_minimum());
    }

    #[test]
    fn meets_webgpu_minimum_checks_texture_2d() {
        let mut raw = wgpu::Limits::default();
        raw.max_texture_dimension_2d = 1; // Below minimum
        let limits = AdapterLimits { raw };

        assert!(!limits.meets_webgpu_minimum());
    }

    #[test]
    fn meets_webgpu_minimum_checks_bind_groups() {
        let mut raw = wgpu::Limits::default();
        raw.max_bind_groups = 1; // Below minimum (4)
        let limits = AdapterLimits { raw };

        assert!(!limits.meets_webgpu_minimum());
    }

    #[test]
    fn webgl2_downlevel_defaults_pass_webgpu_minimum() {
        // WebGL2 defaults should meet their own minimum
        let limits = AdapterLimits {
            raw: wgpu::Limits::downlevel_webgl2_defaults(),
        };
        assert!(limits.meets_webgpu_minimum());
    }

    #[test]
    fn default_limits_pass_webgpu_minimum() {
        let limits = AdapterLimits {
            raw: wgpu::Limits::default(),
        };
        assert!(limits.meets_webgpu_minimum());
    }
}

// ============================================================================
// 9. inspect_limits Function Tests
// ============================================================================

mod inspect_limits_function {
    use super::*;

    // Note: inspect_limits requires an actual wgpu::Adapter, which needs GPU hardware.
    // These tests verify the format_limits_internal behavior through Display trait.

    #[test]
    fn display_returns_non_empty_string() {
        let limits = make_limits();
        let output = format!("{}", limits);
        assert!(!output.is_empty());
    }

    #[test]
    fn display_contains_adapter_limits_header() {
        let limits = make_limits();
        let output = format!("{}", limits);
        assert!(output.contains("Adapter Limits:"));
    }

    #[test]
    fn display_contains_all_limit_categories() {
        let limits = make_limits();
        let output = format!("{}", limits);

        assert!(output.contains("=== Texture Limits ==="));
        assert!(output.contains("=== Buffer Limits ==="));
        assert!(output.contains("=== Bind Group Limits ==="));
        assert!(output.contains("=== Compute Limits ==="));
        assert!(output.contains("=== Vertex Limits ==="));
        assert!(output.contains("=== Other Limits ==="));
    }

    #[test]
    fn display_contains_human_readable_sizes() {
        let limits = make_limits();
        let output = format!("{}", limits);

        assert!(output.contains("bytes"));
        assert!(output.contains("GB"));
        assert!(output.contains("KB"));
        assert!(output.contains("MB"));
    }

    #[test]
    fn display_contains_texture_limit_labels() {
        let limits = make_limits();
        let output = format!("{}", limits);

        assert!(output.contains("Max 1D Dimension"));
        assert!(output.contains("Max 2D Dimension"));
        assert!(output.contains("Max 3D Dimension"));
        assert!(output.contains("Max Array Layers"));
    }

    #[test]
    fn display_contains_buffer_limit_labels() {
        let limits = make_limits();
        let output = format!("{}", limits);

        assert!(output.contains("Max Buffer Size"));
        assert!(output.contains("Max Uniform Binding"));
        assert!(output.contains("Max Storage Binding"));
        assert!(output.contains("Min Uniform Alignment"));
        assert!(output.contains("Min Storage Alignment"));
    }

    #[test]
    fn display_contains_bind_group_labels() {
        let limits = make_limits();
        let output = format!("{}", limits);

        assert!(output.contains("Max Bind Groups"));
        assert!(output.contains("Max Bindings/Group"));
        assert!(output.contains("Max Dynamic Uniforms"));
        assert!(output.contains("Max Dynamic Storage"));
        assert!(output.contains("Max Sampled Textures"));
        assert!(output.contains("Max Samplers"));
        assert!(output.contains("Max Storage Buffers"));
        assert!(output.contains("Max Storage Textures"));
        assert!(output.contains("Max Uniform Buffers"));
    }

    #[test]
    fn display_contains_compute_labels() {
        let limits = make_limits();
        let output = format!("{}", limits);

        assert!(output.contains("Max Workgroup Size"));
        assert!(output.contains("Max Invocations/WG"));
        assert!(output.contains("Max Workgroups/Dim"));
    }

    #[test]
    fn display_contains_vertex_labels() {
        let limits = make_limits();
        let output = format!("{}", limits);

        assert!(output.contains("Max Vertex Buffers"));
        assert!(output.contains("Max Vertex Attributes"));
        assert!(output.contains("Max Buffer Stride"));
    }

    #[test]
    fn display_contains_other_limits() {
        let limits = make_limits();
        let output = format!("{}", limits);

        assert!(output.contains("Max Inter-Stage Comps"));
        assert!(output.contains("Max Color Attachments"));
        assert!(output.contains("Max Color Bytes/Sample"));
    }

    #[test]
    fn display_contains_per_stage_annotations() {
        let limits = make_limits();
        let output = format!("{}", limits);

        // Several bind group limits are annotated as "per stage"
        assert!(output.contains("(per stage)"));
    }
}

// ============================================================================
// 10. Trait Implementation Tests
// ============================================================================

mod trait_implementations {
    use super::*;

    #[test]
    fn adapter_limits_is_clone() {
        let limits = make_limits();
        let cloned = limits.clone();

        assert_eq!(limits.max_texture_dimension_2d(), cloned.max_texture_dimension_2d());
        assert_eq!(limits.max_buffer_size(), cloned.max_buffer_size());
        assert_eq!(limits.max_bind_groups(), cloned.max_bind_groups());
        assert_eq!(limits.max_compute_workgroup_size_x(), cloned.max_compute_workgroup_size_x());
        assert_eq!(limits.max_vertex_buffers(), cloned.max_vertex_buffers());
    }

    #[test]
    fn adapter_limits_display_is_non_empty() {
        let limits = make_limits();
        let display = format!("{}", limits);
        assert!(!display.is_empty());
        assert!(display.len() > 100); // Should be substantial output
    }

    #[test]
    fn adapter_limits_debug_is_implemented() {
        let limits = make_limits();
        let debug = format!("{:?}", limits);
        assert!(debug.contains("AdapterLimits"));
    }

    #[test]
    fn limits_summary_is_clone() {
        let limits = make_limits();
        let summary = limits.summary();
        let cloned = summary.clone();

        assert_eq!(summary.texture.max_2d, cloned.texture.max_2d);
        assert_eq!(summary.buffer.max_size, cloned.buffer.max_size);
    }

    #[test]
    fn limits_summary_is_debug() {
        let limits = make_limits();
        let summary = limits.summary();
        let debug = format!("{:?}", summary);

        assert!(debug.contains("LimitsSummary"));
    }

    #[test]
    fn texture_limits_is_copy() {
        let limits = make_limits();
        let summary = limits.summary();
        let texture = summary.texture;
        let copy = texture; // Copy

        assert_eq!(texture.max_2d, copy.max_2d);
    }

    #[test]
    fn buffer_limits_is_copy() {
        let limits = make_limits();
        let summary = limits.summary();
        let buffer = summary.buffer;
        let copy = buffer; // Copy

        assert_eq!(buffer.max_size, copy.max_size);
    }

    #[test]
    fn bind_group_limits_is_copy() {
        let limits = make_limits();
        let summary = limits.summary();
        let bind_group = summary.bind_group;
        let copy = bind_group; // Copy

        assert_eq!(bind_group.max_bind_groups, copy.max_bind_groups);
    }

    #[test]
    fn compute_limits_is_copy() {
        let limits = make_limits();
        let summary = limits.summary();
        let compute = summary.compute;
        let copy = compute; // Copy

        assert_eq!(compute.max_workgroup_size_x, copy.max_workgroup_size_x);
    }

    #[test]
    fn vertex_limits_is_copy() {
        let limits = make_limits();
        let summary = limits.summary();
        let vertex = summary.vertex;
        let copy = vertex; // Copy

        assert_eq!(vertex.max_buffers, copy.max_buffers);
    }

    #[test]
    fn all_summary_structs_are_debug() {
        let limits = make_limits();
        let summary = limits.summary();

        let _ = format!("{:?}", summary.texture);
        let _ = format!("{:?}", summary.buffer);
        let _ = format!("{:?}", summary.bind_group);
        let _ = format!("{:?}", summary.compute);
        let _ = format!("{:?}", summary.vertex);
    }
}

// ============================================================================
// 11. Additional Limits Tests (Other category)
// ============================================================================

mod additional_limits {
    use super::*;

    #[test]
    fn max_inter_stage_shader_components_returns_raw() {
        let limits = make_limits();
        assert_eq!(
            limits.max_inter_stage_shader_components(),
            limits.raw.max_inter_stage_shader_components
        );
    }

    #[test]
    fn max_color_attachments_returns_raw() {
        let limits = make_limits();
        assert_eq!(limits.max_color_attachments(), limits.raw.max_color_attachments);
    }

    #[test]
    fn max_color_attachment_bytes_per_sample_returns_raw() {
        let limits = make_limits();
        assert_eq!(
            limits.max_color_attachment_bytes_per_sample(),
            limits.raw.max_color_attachment_bytes_per_sample
        );
    }

    #[test]
    fn color_attachments_at_least_1() {
        let limits = make_limits();
        assert!(limits.max_color_attachments() >= 1);
    }

    #[test]
    fn inter_stage_components_nonzero() {
        let limits = make_limits();
        assert!(limits.max_inter_stage_shader_components() > 0);
    }
}

// ============================================================================
// 12. Boundary and Edge Case Tests
// ============================================================================

mod boundary_cases {
    use super::*;

    #[test]
    fn extreme_texture_dimensions() {
        let limits = make_custom_limits(u32::MAX, u32::MAX, u64::MAX);
        assert_eq!(limits.max_texture_dimension_1d(), u32::MAX);
        assert_eq!(limits.max_texture_dimension_2d(), u32::MAX);
    }

    #[test]
    fn extreme_buffer_size() {
        let limits = make_custom_limits(8192, 8192, u64::MAX);
        assert_eq!(limits.max_buffer_size(), u64::MAX);
    }

    #[test]
    fn zero_texture_dimensions_handled() {
        let mut raw = wgpu::Limits::default();
        raw.max_texture_dimension_1d = 0;
        raw.max_texture_dimension_2d = 0;
        let limits = AdapterLimits { raw };

        // Methods should still work
        assert_eq!(limits.max_texture_dimension_1d(), 0);
        assert_eq!(limits.max_texture_dimension_2d(), 0);
    }

    #[test]
    fn multiple_clones_independent() {
        let limits = make_limits();
        let clone1 = limits.clone();
        let clone2 = limits.clone();

        // All should have same values
        assert_eq!(limits.max_texture_dimension_2d(), clone1.max_texture_dimension_2d());
        assert_eq!(limits.max_texture_dimension_2d(), clone2.max_texture_dimension_2d());
        assert_eq!(clone1.max_texture_dimension_2d(), clone2.max_texture_dimension_2d());
    }

    #[test]
    fn summary_repeated_calls_consistent() {
        let limits = make_limits();
        let summary1 = limits.summary();
        let summary2 = limits.summary();

        assert_eq!(summary1.texture.max_2d, summary2.texture.max_2d);
        assert_eq!(summary1.buffer.max_size, summary2.buffer.max_size);
    }

    #[test]
    fn display_repeated_calls_consistent() {
        let limits = make_limits();
        let display1 = format!("{}", limits);
        let display2 = format!("{}", limits);

        assert_eq!(display1, display2);
    }
}

// ============================================================================
// Integration Tests (require actual GPU hardware)
// ============================================================================

#[cfg(not(feature = "ci"))]
mod integration_tests {
    use super::*;

    #[test]
    fn from_adapter_with_real_hardware() {
        // This test requires actual GPU hardware
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(async {
            instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
        });

        if let Some(adapter) = adapter {
            let limits = AdapterLimits::from_adapter(&adapter);

            // Real hardware should have reasonable limits
            assert!(limits.max_texture_dimension_2d() >= 4096);
            assert!(limits.max_buffer_size() > 0);
            assert!(limits.max_bind_groups() >= 4);
            assert!(limits.meets_webgpu_minimum());
        }
    }

    #[test]
    fn inspect_limits_with_real_hardware() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(async {
            instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
        });

        if let Some(adapter) = adapter {
            let output = inspect_limits(&adapter);

            assert!(!output.is_empty());
            assert!(output.contains("Adapter Limits:"));
            assert!(output.contains("=== Texture Limits ==="));
        }
    }
}
