// Blackbox contract tests for T-WGPU-P1.2.3 Adapter Limits Query
//
// CLEANROOM: No access to implementation files. Tests use only the public API
// exported by `renderer_backend::device`.
//
// Forbidden files (per TESTDEV_BLACKBOX prompt):
//   - crates/renderer-backend/src/device/adapter.rs
//   - crates/renderer-backend/src/device/instance.rs
//   - Any WHITEBOX test file for this task
//
// Contract sources:
//   - docs/WGPU_DOCS/RDC_OUTPUT/PHASE_1_CORE_TODO.md (T-WGPU-P1.2.3)
//
// Acceptance criteria (T-WGPU-P1.2.3):
//   - All texture limits exposed
//   - All buffer limits exposed
//   - All bind group limits exposed
//   - All compute limits exposed
//   - Formatted output for debugging
//
// Test design rationale:
//   Equivalence partitioning:
//     - Valid adapters from enumeration
//     - Different adapter types (discrete, integrated)
//   Boundary cases:
//     - Zero adapters (no limits to query)
//     - Minimum reasonable values (> 0 for most limits)
//   Contract verification:
//     - AdapterLimits struct and methods
//     - LimitsSummary struct and sub-structs
//     - inspect_limits() function

use renderer_backend::device::{
    enumerate_adapters_with_info, inspect_limits, AdapterLimits, BindGroupLimits, BufferLimits,
    ComputeLimits, LimitsSummary, TextureLimits, TrinityInstance, VertexLimits,
};

// =============================================================================
// 1. Texture Limits Contract Tests
// =============================================================================

/// Verifies that AdapterLimits can be created from a wgpu::Adapter.
///
/// Contract: AdapterLimits::from_adapter() returns valid limits struct.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_limits_from_adapter_returns_valid_limits() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let limits = AdapterLimits::from_adapter(adapter);

        // Should have reasonable texture dimension limits (> 0)
        assert!(
            limits.max_texture_dimension_1d() > 0,
            "max_texture_dimension_1d should be > 0"
        );
    }
}

/// Verifies that max_texture_dimension_1d() returns u32.
///
/// Contract: max_texture_dimension_1d() returns u32 value.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_limits_max_texture_dimension_1d_returns_u32() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let limits = AdapterLimits::from_adapter(adapter);

        let dim: u32 = limits.max_texture_dimension_1d();
        assert!(
            dim >= 1024,
            "1D texture dimension should be at least 1024, got: {}",
            dim
        );
    }
}

/// Verifies that max_texture_dimension_2d() returns u32.
///
/// Contract: max_texture_dimension_2d() returns u32 value within typical GPU ranges.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_limits_max_texture_dimension_2d_returns_u32() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let limits = AdapterLimits::from_adapter(adapter);

        let dim: u32 = limits.max_texture_dimension_2d();
        // WebGPU minimum is 8192
        assert!(
            dim >= 8192,
            "2D texture dimension should be at least 8192, got: {}",
            dim
        );
    }
}

/// Verifies that max_texture_dimension_3d() returns u32.
///
/// Contract: max_texture_dimension_3d() returns u32 value within typical GPU ranges.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_limits_max_texture_dimension_3d_returns_u32() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let limits = AdapterLimits::from_adapter(adapter);

        let dim: u32 = limits.max_texture_dimension_3d();
        // WebGPU minimum is 2048
        assert!(
            dim >= 256,
            "3D texture dimension should be at least 256, got: {}",
            dim
        );
    }
}

/// Verifies that max_texture_array_layers() returns u32.
///
/// Contract: max_texture_array_layers() returns u32 value.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_limits_max_texture_array_layers_returns_u32() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let limits = AdapterLimits::from_adapter(adapter);

        let layers: u32 = limits.max_texture_array_layers();
        // WebGPU minimum is 256
        assert!(
            layers >= 256,
            "Texture array layers should be at least 256, got: {}",
            layers
        );
    }
}

/// Verifies that all texture limits are reasonable (> 0).
///
/// Contract: All texture dimension limits should be positive.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_limits_all_texture_limits_positive() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    for adapter in &result.adapters {
        let limits = AdapterLimits::from_adapter(adapter);
        let info = adapter.get_info();

        assert!(
            limits.max_texture_dimension_1d() > 0,
            "max_texture_dimension_1d should be > 0 for adapter: {}",
            info.name
        );
        assert!(
            limits.max_texture_dimension_2d() > 0,
            "max_texture_dimension_2d should be > 0 for adapter: {}",
            info.name
        );
        assert!(
            limits.max_texture_dimension_3d() > 0,
            "max_texture_dimension_3d should be > 0 for adapter: {}",
            info.name
        );
        assert!(
            limits.max_texture_array_layers() > 0,
            "max_texture_array_layers should be > 0 for adapter: {}",
            info.name
        );
    }
}

// =============================================================================
// 2. Buffer Limits Contract Tests
// =============================================================================

/// Verifies that max_buffer_size() returns u64.
///
/// Contract: max_buffer_size() returns u64 value.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_limits_max_buffer_size_returns_u64() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let limits = AdapterLimits::from_adapter(adapter);

        let size: u64 = limits.max_buffer_size();
        // Should be at least 256MB (268435456 bytes) for reasonable GPUs
        assert!(
            size >= 256 * 1024 * 1024,
            "max_buffer_size should be at least 256MB, got: {} bytes",
            size
        );
    }
}

/// Verifies that max_uniform_buffer_binding_size() returns u32.
///
/// Contract: max_uniform_buffer_binding_size() returns u32 value.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_limits_max_uniform_buffer_binding_size_returns_u32() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let limits = AdapterLimits::from_adapter(adapter);

        let size: u32 = limits.max_uniform_buffer_binding_size();
        // WebGPU minimum is 65536 (64KB)
        assert!(
            size >= 65536,
            "max_uniform_buffer_binding_size should be at least 64KB, got: {} bytes",
            size
        );
    }
}

/// Verifies that max_storage_buffer_binding_size() returns u32.
///
/// Contract: max_storage_buffer_binding_size() returns u32 value.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_limits_max_storage_buffer_binding_size_returns_u32() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let limits = AdapterLimits::from_adapter(adapter);

        let size: u32 = limits.max_storage_buffer_binding_size();
        // WebGPU minimum is 128MB
        assert!(
            size >= 128 * 1024 * 1024,
            "max_storage_buffer_binding_size should be at least 128MB, got: {} bytes",
            size
        );
    }
}

/// Verifies that all buffer limits are reasonable (> 0).
///
/// Contract: All buffer limits should be positive.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_limits_all_buffer_limits_positive() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    for adapter in &result.adapters {
        let limits = AdapterLimits::from_adapter(adapter);
        let info = adapter.get_info();

        assert!(
            limits.max_buffer_size() > 0,
            "max_buffer_size should be > 0 for adapter: {}",
            info.name
        );
        assert!(
            limits.max_uniform_buffer_binding_size() > 0,
            "max_uniform_buffer_binding_size should be > 0 for adapter: {}",
            info.name
        );
        assert!(
            limits.max_storage_buffer_binding_size() > 0,
            "max_storage_buffer_binding_size should be > 0 for adapter: {}",
            info.name
        );
    }
}

// =============================================================================
// 3. Bind Group Limits Contract Tests
// =============================================================================

/// Verifies that max_bind_groups() returns u32.
///
/// Contract: max_bind_groups() returns u32 value.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_limits_max_bind_groups_returns_u32() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let limits = AdapterLimits::from_adapter(adapter);

        let groups: u32 = limits.max_bind_groups();
        // WebGPU minimum is 4
        assert!(
            groups >= 4,
            "max_bind_groups should be at least 4, got: {}",
            groups
        );
    }
}

/// Verifies that max_bindings_per_bind_group() returns u32.
///
/// Contract: max_bindings_per_bind_group() returns u32 value.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_limits_max_bindings_per_bind_group_returns_u32() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let limits = AdapterLimits::from_adapter(adapter);

        let bindings: u32 = limits.max_bindings_per_bind_group();
        // WebGPU minimum is 1000
        assert!(
            bindings >= 100,
            "max_bindings_per_bind_group should be at least 100, got: {}",
            bindings
        );
    }
}

/// Verifies that all bind group limits are positive.
///
/// Contract: All bind group limits should be > 0.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_limits_all_bind_group_limits_positive() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    for adapter in &result.adapters {
        let limits = AdapterLimits::from_adapter(adapter);
        let info = adapter.get_info();

        assert!(
            limits.max_bind_groups() > 0,
            "max_bind_groups should be > 0 for adapter: {}",
            info.name
        );
        assert!(
            limits.max_bindings_per_bind_group() > 0,
            "max_bindings_per_bind_group should be > 0 for adapter: {}",
            info.name
        );
    }
}

// =============================================================================
// 4. Compute Limits Contract Tests
// =============================================================================

/// Verifies that max_compute_workgroup_size_x() returns u32.
///
/// Contract: max_compute_workgroup_size_x() returns u32 value.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_limits_max_compute_workgroup_size_x_returns_u32() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let limits = AdapterLimits::from_adapter(adapter);

        let size: u32 = limits.max_compute_workgroup_size_x();
        // WebGPU minimum is 256
        assert!(
            size >= 64,
            "max_compute_workgroup_size_x should be at least 64, got: {}",
            size
        );
    }
}

/// Verifies that max_compute_workgroup_size_y() returns u32.
///
/// Contract: max_compute_workgroup_size_y() returns u32 value.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_limits_max_compute_workgroup_size_y_returns_u32() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let limits = AdapterLimits::from_adapter(adapter);

        let size: u32 = limits.max_compute_workgroup_size_y();
        // WebGPU minimum is 256
        assert!(
            size >= 64,
            "max_compute_workgroup_size_y should be at least 64, got: {}",
            size
        );
    }
}

/// Verifies that max_compute_workgroup_size_z() returns u32.
///
/// Contract: max_compute_workgroup_size_z() returns u32 value.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_limits_max_compute_workgroup_size_z_returns_u32() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let limits = AdapterLimits::from_adapter(adapter);

        let size: u32 = limits.max_compute_workgroup_size_z();
        // WebGPU minimum is 64
        assert!(
            size >= 64,
            "max_compute_workgroup_size_z should be at least 64, got: {}",
            size
        );
    }
}

/// Verifies that max_compute_invocations_per_workgroup() returns u32.
///
/// Contract: max_compute_invocations_per_workgroup() returns u32 value.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_limits_max_compute_invocations_per_workgroup_returns_u32() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let limits = AdapterLimits::from_adapter(adapter);

        let invocations: u32 = limits.max_compute_invocations_per_workgroup();
        // WebGPU minimum is 256
        assert!(
            invocations >= 256,
            "max_compute_invocations_per_workgroup should be at least 256, got: {}",
            invocations
        );
    }
}

/// Verifies that all compute limits are positive.
///
/// Contract: All compute limits should be > 0.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_limits_all_compute_limits_positive() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    for adapter in &result.adapters {
        let limits = AdapterLimits::from_adapter(adapter);
        let info = adapter.get_info();

        assert!(
            limits.max_compute_workgroup_size_x() > 0,
            "max_compute_workgroup_size_x should be > 0 for adapter: {}",
            info.name
        );
        assert!(
            limits.max_compute_workgroup_size_y() > 0,
            "max_compute_workgroup_size_y should be > 0 for adapter: {}",
            info.name
        );
        assert!(
            limits.max_compute_workgroup_size_z() > 0,
            "max_compute_workgroup_size_z should be > 0 for adapter: {}",
            info.name
        );
        assert!(
            limits.max_compute_invocations_per_workgroup() > 0,
            "max_compute_invocations_per_workgroup should be > 0 for adapter: {}",
            info.name
        );
    }
}

/// Verifies that compute workgroup size limits are consistent.
///
/// Contract: max_compute_invocations should be >= x * y * z (at minimum sizes).
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_limits_compute_workgroup_sizes_consistent() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    for adapter in &result.adapters {
        let limits = AdapterLimits::from_adapter(adapter);
        let info = adapter.get_info();

        let x = limits.max_compute_workgroup_size_x();
        let y = limits.max_compute_workgroup_size_y();
        let z = limits.max_compute_workgroup_size_z();
        let total = limits.max_compute_invocations_per_workgroup();

        // x * y * z could exceed total (you can't use max of all dimensions at once)
        // But total should be at least as large as any single dimension
        assert!(
            total >= x,
            "max_compute_invocations ({}) should be >= max_x ({}) for adapter: {}",
            total,
            x,
            info.name
        );
        assert!(
            total >= y,
            "max_compute_invocations ({}) should be >= max_y ({}) for adapter: {}",
            total,
            y,
            info.name
        );
        assert!(
            total >= z,
            "max_compute_invocations ({}) should be >= max_z ({}) for adapter: {}",
            total,
            z,
            info.name
        );
    }
}

// =============================================================================
// 5. Summary Contract Tests
// =============================================================================

/// Verifies that summary() returns LimitsSummary.
///
/// Contract: summary() method returns LimitsSummary struct.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_limits_summary_returns_limits_summary() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let limits = AdapterLimits::from_adapter(adapter);

        let summary: LimitsSummary = limits.summary();
        let _ = summary;
    }
}

/// Verifies that LimitsSummary has texture field.
///
/// Contract: LimitsSummary.texture is accessible and is TextureLimits.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_limits_summary_has_texture_field() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let limits = AdapterLimits::from_adapter(adapter);
        let summary = limits.summary();

        let _texture: &TextureLimits = &summary.texture;
    }
}

/// Verifies that LimitsSummary has buffer field.
///
/// Contract: LimitsSummary.buffer is accessible and is BufferLimits.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_limits_summary_has_buffer_field() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let limits = AdapterLimits::from_adapter(adapter);
        let summary = limits.summary();

        let _buffer: &BufferLimits = &summary.buffer;
    }
}

/// Verifies that LimitsSummary has bind_group field.
///
/// Contract: LimitsSummary.bind_group is accessible and is BindGroupLimits.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_limits_summary_has_bind_group_field() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let limits = AdapterLimits::from_adapter(adapter);
        let summary = limits.summary();

        let _bind_group: &BindGroupLimits = &summary.bind_group;
    }
}

/// Verifies that LimitsSummary has compute field.
///
/// Contract: LimitsSummary.compute is accessible and is ComputeLimits.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_limits_summary_has_compute_field() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let limits = AdapterLimits::from_adapter(adapter);
        let summary = limits.summary();

        let _compute: &ComputeLimits = &summary.compute;
    }
}

/// Verifies that LimitsSummary has vertex field.
///
/// Contract: LimitsSummary.vertex is accessible and is VertexLimits.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_limits_summary_has_vertex_field() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let limits = AdapterLimits::from_adapter(adapter);
        let summary = limits.summary();

        let _vertex: &VertexLimits = &summary.vertex;
    }
}

/// Verifies that TextureLimits sub-struct has expected fields.
///
/// Contract: TextureLimits has max_1d, max_2d, max_3d, max_array_layers.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_texture_limits_has_expected_fields() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let limits = AdapterLimits::from_adapter(adapter);
        let summary = limits.summary();

        let _d1: u32 = summary.texture.max_1d;
        let _d2: u32 = summary.texture.max_2d;
        let _d3: u32 = summary.texture.max_3d;
        let _layers: u32 = summary.texture.max_array_layers;
    }
}

/// Verifies that BufferLimits sub-struct has expected fields.
///
/// Contract: BufferLimits has max_size, max_uniform_binding, max_storage_binding.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_buffer_limits_has_expected_fields() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let limits = AdapterLimits::from_adapter(adapter);
        let summary = limits.summary();

        let _size: u64 = summary.buffer.max_size;
        let _uniform: u32 = summary.buffer.max_uniform_binding;
        let _storage: u32 = summary.buffer.max_storage_binding;
    }
}

/// Verifies that BindGroupLimits sub-struct has expected fields.
///
/// Contract: BindGroupLimits has max_bind_groups, max_bindings_per_group.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_bind_group_limits_has_expected_fields() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let limits = AdapterLimits::from_adapter(adapter);
        let summary = limits.summary();

        let _groups: u32 = summary.bind_group.max_bind_groups;
        let _bindings: u32 = summary.bind_group.max_bindings_per_group;
    }
}

/// Verifies that ComputeLimits sub-struct has expected fields.
///
/// Contract: ComputeLimits has workgroup size x/y/z and invocations.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_compute_limits_has_expected_fields() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let limits = AdapterLimits::from_adapter(adapter);
        let summary = limits.summary();

        let _x: u32 = summary.compute.max_workgroup_size_x;
        let _y: u32 = summary.compute.max_workgroup_size_y;
        let _z: u32 = summary.compute.max_workgroup_size_z;
        let _inv: u32 = summary.compute.max_invocations_per_workgroup;
    }
}

/// Verifies that VertexLimits sub-struct has expected fields.
///
/// Contract: VertexLimits has max_buffers, max_attributes, max_buffer_array_stride.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_vertex_limits_has_expected_fields() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let limits = AdapterLimits::from_adapter(adapter);
        let summary = limits.summary();

        let _buffers: u32 = summary.vertex.max_buffers;
        let _attributes: u32 = summary.vertex.max_attributes;
        let _stride: u32 = summary.vertex.max_buffer_array_stride;
    }
}

// =============================================================================
// 6. inspect_limits() Contract Tests
// =============================================================================

/// Verifies that inspect_limits() returns non-empty String.
///
/// Contract: inspect_limits(&Adapter) returns non-empty formatted String.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_inspect_limits_returns_nonempty_string() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let output = inspect_limits(adapter);

        assert!(
            !output.is_empty(),
            "inspect_limits() should return non-empty string"
        );
    }
}

/// Verifies that inspect_limits() contains texture-related text.
///
/// Contract: Output should contain "Texture" or texture-related info.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_inspect_limits_contains_texture_info() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let output = inspect_limits(adapter);
        let lower = output.to_lowercase();

        let has_texture = lower.contains("texture")
            || lower.contains("dimension")
            || lower.contains("array_layers");

        assert!(
            has_texture,
            "inspect_limits() output should contain texture-related info, got:\n{}",
            output
        );
    }
}

/// Verifies that inspect_limits() contains buffer-related text.
///
/// Contract: Output should contain "Buffer" or buffer-related info.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_inspect_limits_contains_buffer_info() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let output = inspect_limits(adapter);
        let lower = output.to_lowercase();

        let has_buffer =
            lower.contains("buffer") || lower.contains("uniform") || lower.contains("storage");

        assert!(
            has_buffer,
            "inspect_limits() output should contain buffer-related info, got:\n{}",
            output
        );
    }
}

/// Verifies that inspect_limits() contains compute-related text.
///
/// Contract: Output should contain "Compute" or compute-related info.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_inspect_limits_contains_compute_info() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let output = inspect_limits(adapter);
        let lower = output.to_lowercase();

        let has_compute =
            lower.contains("compute") || lower.contains("workgroup") || lower.contains("invocation");

        assert!(
            has_compute,
            "inspect_limits() output should contain compute-related info, got:\n{}",
            output
        );
    }
}

/// Verifies that inspect_limits() output is multi-line formatted.
///
/// Contract: Output should be formatted with newlines for readability.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_inspect_limits_is_multiline_formatted() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let output = inspect_limits(adapter);

        let line_count = output.lines().count();
        assert!(
            line_count > 3,
            "inspect_limits() output should be multi-line formatted, got {} lines",
            line_count
        );
    }
}

// =============================================================================
// 7. WebGPU Minimum Contract Tests
// =============================================================================

/// Verifies that meets_webgpu_minimum() returns bool.
///
/// Contract: meets_webgpu_minimum() returns boolean.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_limits_meets_webgpu_minimum_returns_bool() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let limits = AdapterLimits::from_adapter(adapter);

        let _meets: bool = limits.meets_webgpu_minimum();
    }
}

/// Verifies that modern GPUs meet WebGPU minimum requirements.
///
/// Contract: Modern discrete and integrated GPUs should meet WebGPU minimum.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_limits_modern_gpu_meets_webgpu_minimum() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    for adapter in &result.adapters {
        let info = adapter.get_info();

        // Skip software renderers (they may not meet minimums)
        if matches!(info.device_type, wgpu::DeviceType::Cpu) {
            continue;
        }

        let limits = AdapterLimits::from_adapter(adapter);

        assert!(
            limits.meets_webgpu_minimum(),
            "Modern GPU '{}' ({:?}) should meet WebGPU minimum requirements",
            info.name,
            info.device_type
        );
    }
}

/// Verifies that WebGPU minimum check includes texture limits.
///
/// Contract: Adapter with low texture limits should fail WebGPU minimum check.
/// (This is a behavioral test - if the adapter reports adequate limits, it should pass)
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_limits_webgpu_minimum_validates_textures() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    for adapter in &result.adapters {
        let limits = AdapterLimits::from_adapter(adapter);

        // If adapter meets minimum, texture limits should meet WebGPU spec
        if limits.meets_webgpu_minimum() {
            assert!(
                limits.max_texture_dimension_2d() >= 8192,
                "If meets_webgpu_minimum is true, max_texture_dimension_2d should be >= 8192"
            );
        }
    }
}

// =============================================================================
// 8. Graceful Handling Tests
// =============================================================================

/// Verifies that AdapterLimits works with all real adapters from enumeration.
///
/// Contract: from_adapter() should not panic for any adapter from enumeration.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_limits_works_with_all_real_adapters() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    for adapter in &result.adapters {
        // This should not panic
        let limits = AdapterLimits::from_adapter(adapter);
        let _summary = limits.summary();
    }
}

/// Verifies that no panics occur when accessing any limit.
///
/// Contract: All limit accessor methods should not panic.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_limits_no_panics_on_any_accessor() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    for adapter in &result.adapters {
        let limits = AdapterLimits::from_adapter(adapter);

        // Texture limits
        let _ = limits.max_texture_dimension_1d();
        let _ = limits.max_texture_dimension_2d();
        let _ = limits.max_texture_dimension_3d();
        let _ = limits.max_texture_array_layers();

        // Buffer limits
        let _ = limits.max_buffer_size();
        let _ = limits.max_uniform_buffer_binding_size();
        let _ = limits.max_storage_buffer_binding_size();

        // Bind group limits
        let _ = limits.max_bind_groups();
        let _ = limits.max_bindings_per_bind_group();

        // Compute limits
        let _ = limits.max_compute_workgroup_size_x();
        let _ = limits.max_compute_workgroup_size_y();
        let _ = limits.max_compute_workgroup_size_z();
        let _ = limits.max_compute_invocations_per_workgroup();

        // Summary and WebGPU check
        let _ = limits.summary();
        let _ = limits.meets_webgpu_minimum();
    }
}

/// Verifies that inspect_limits() works with all real adapters.
///
/// Contract: inspect_limits() should not panic for any adapter.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_inspect_limits_works_with_all_real_adapters() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    for adapter in &result.adapters {
        // This should not panic
        let output = inspect_limits(adapter);

        // Should return non-empty output
        assert!(
            !output.is_empty(),
            "inspect_limits() should return non-empty output for all adapters"
        );
    }
}

// =============================================================================
// 9. Consistency and Determinism Tests
// =============================================================================

/// Verifies that multiple calls to from_adapter produce consistent results.
///
/// Contract: from_adapter is deterministic for the same adapter.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_limits_from_adapter_is_deterministic() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];

        let limits1 = AdapterLimits::from_adapter(adapter);
        let limits2 = AdapterLimits::from_adapter(adapter);

        // All limits should be identical
        assert_eq!(
            limits1.max_texture_dimension_2d(),
            limits2.max_texture_dimension_2d(),
            "Multiple from_adapter calls should produce same texture limits"
        );
        assert_eq!(
            limits1.max_buffer_size(),
            limits2.max_buffer_size(),
            "Multiple from_adapter calls should produce same buffer limits"
        );
        assert_eq!(
            limits1.max_bind_groups(),
            limits2.max_bind_groups(),
            "Multiple from_adapter calls should produce same bind group limits"
        );
        assert_eq!(
            limits1.max_compute_invocations_per_workgroup(),
            limits2.max_compute_invocations_per_workgroup(),
            "Multiple from_adapter calls should produce same compute limits"
        );
    }
}

/// Verifies that summary() is deterministic.
///
/// Contract: summary() returns identical results for the same limits.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_adapter_limits_summary_is_deterministic() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];
        let limits = AdapterLimits::from_adapter(adapter);

        let summary1 = limits.summary();
        let summary2 = limits.summary();

        assert_eq!(
            summary1.texture.max_2d, summary2.texture.max_2d,
            "Multiple summary() calls should produce same texture limits"
        );
        assert_eq!(
            summary1.buffer.max_size, summary2.buffer.max_size,
            "Multiple summary() calls should produce same buffer limits"
        );
    }
}

/// Verifies that inspect_limits() is deterministic.
///
/// Contract: inspect_limits() returns identical strings for the same adapter.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_inspect_limits_is_deterministic() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    if !result.adapters.is_empty() {
        let adapter = &result.adapters[0];

        let output1 = inspect_limits(adapter);
        let output2 = inspect_limits(adapter);
        let output3 = inspect_limits(adapter);

        assert_eq!(
            output1, output2,
            "Multiple inspect_limits() calls should produce same output"
        );
        assert_eq!(
            output2, output3,
            "Multiple inspect_limits() calls should produce same output"
        );
    }
}

// =============================================================================
// 10. Summary Field Value Consistency Tests
// =============================================================================

/// Verifies that summary texture limits match direct accessor values.
///
/// Contract: summary().texture values should match individual accessor methods.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_summary_texture_limits_match_accessors() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    for adapter in &result.adapters {
        let limits = AdapterLimits::from_adapter(adapter);
        let summary = limits.summary();

        assert_eq!(
            summary.texture.max_1d,
            limits.max_texture_dimension_1d(),
            "summary().texture.max_1d should match accessor"
        );
        assert_eq!(
            summary.texture.max_2d,
            limits.max_texture_dimension_2d(),
            "summary().texture.max_2d should match accessor"
        );
        assert_eq!(
            summary.texture.max_3d,
            limits.max_texture_dimension_3d(),
            "summary().texture.max_3d should match accessor"
        );
        assert_eq!(
            summary.texture.max_array_layers,
            limits.max_texture_array_layers(),
            "summary().texture.max_array_layers should match accessor"
        );
    }
}

/// Verifies that summary buffer limits match direct accessor values.
///
/// Contract: summary().buffer values should match individual accessor methods.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_summary_buffer_limits_match_accessors() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    for adapter in &result.adapters {
        let limits = AdapterLimits::from_adapter(adapter);
        let summary = limits.summary();

        assert_eq!(
            summary.buffer.max_size,
            limits.max_buffer_size(),
            "summary().buffer.max_size should match accessor"
        );
        assert_eq!(
            summary.buffer.max_uniform_binding,
            limits.max_uniform_buffer_binding_size(),
            "summary().buffer.max_uniform_binding should match accessor"
        );
        assert_eq!(
            summary.buffer.max_storage_binding,
            limits.max_storage_buffer_binding_size(),
            "summary().buffer.max_storage_binding should match accessor"
        );
    }
}

/// Verifies that summary bind group limits match direct accessor values.
///
/// Contract: summary().bind_group values should match individual accessor methods.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_summary_bind_group_limits_match_accessors() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    for adapter in &result.adapters {
        let limits = AdapterLimits::from_adapter(adapter);
        let summary = limits.summary();

        assert_eq!(
            summary.bind_group.max_bind_groups,
            limits.max_bind_groups(),
            "summary().bind_group.max_bind_groups should match accessor"
        );
        assert_eq!(
            summary.bind_group.max_bindings_per_group,
            limits.max_bindings_per_bind_group(),
            "summary().bind_group.max_bindings_per_group should match accessor"
        );
    }
}

/// Verifies that summary compute limits match direct accessor values.
///
/// Contract: summary().compute values should match individual accessor methods.
#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_summary_compute_limits_match_accessors() {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());

    for adapter in &result.adapters {
        let limits = AdapterLimits::from_adapter(adapter);
        let summary = limits.summary();

        assert_eq!(
            summary.compute.max_workgroup_size_x,
            limits.max_compute_workgroup_size_x(),
            "summary().compute.max_workgroup_size_x should match accessor"
        );
        assert_eq!(
            summary.compute.max_workgroup_size_y,
            limits.max_compute_workgroup_size_y(),
            "summary().compute.max_workgroup_size_y should match accessor"
        );
        assert_eq!(
            summary.compute.max_workgroup_size_z,
            limits.max_compute_workgroup_size_z(),
            "summary().compute.max_workgroup_size_z should match accessor"
        );
        assert_eq!(
            summary.compute.max_invocations_per_workgroup,
            limits.max_compute_invocations_per_workgroup(),
            "summary().compute.max_invocations_per_workgroup should match accessor"
        );
    }
}
