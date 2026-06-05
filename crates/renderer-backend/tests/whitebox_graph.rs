//! Whitebox tests for T-WGPU-P7.5.1: Frame Graph Data Structure
//!
//! Comprehensive whitebox testing of the core frame graph data structures:
//! - ResourceId: unique identifier for GPU resources
//! - PassId: unique identifier for render passes
//! - ResourceAccess: read/write access patterns and hazard detection
//! - ResourceUsage: resource + access + pipeline stage
//! - PassType: render/compute/transfer/raytracing classification
//! - ResourceType: buffer/texture physical types
//! - GraphResourceLifetime: transient/persistent/imported
//! - PassNode: render pass with inputs/outputs/callbacks
//! - ResourceNode: GPU resource with producer/consumer tracking
//! - FrameGraph: main orchestration structure
//! - FrameGraphError: error types for compilation/execution
//! - FrameGraphBuilder: fluent API for graph construction

use renderer_backend::frame_graph::graph::{
    FrameGraph, FrameGraphBuilder, FrameGraphError, GraphResourceLifetime, PassId, PassNode,
    PassType, RenderContext, ResourceAccess, ResourceId, ResourceNode, ResourceType,
    ResourceUsage,
};
use renderer_backend::resource_state::PipelineStage;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

// ===========================================================================
// Section 1: ResourceId Tests (15+)
// ===========================================================================

#[test]
fn test_resource_id_new_zero() {
    let id = ResourceId::new(0);
    assert_eq!(id.raw(), 0);
    assert!(!id.is_invalid());
}

#[test]
fn test_resource_id_new_arbitrary() {
    let id = ResourceId::new(42);
    assert_eq!(id.raw(), 42);
}

#[test]
fn test_resource_id_new_large_value() {
    let id = ResourceId::new(u64::MAX - 1);
    assert_eq!(id.raw(), u64::MAX - 1);
    assert!(!id.is_invalid());
}

#[test]
fn test_resource_id_invalid_sentinel() {
    let id = ResourceId::INVALID;
    assert!(id.is_invalid());
    assert_eq!(id.raw(), u64::MAX);
}

#[test]
fn test_resource_id_none_sentinel_via_default() {
    let id = ResourceId::default();
    assert!(id.is_invalid());
    assert_eq!(id, ResourceId::INVALID);
}

#[test]
fn test_resource_id_raw_accessor_const() {
    const ID: ResourceId = ResourceId::new(100);
    assert_eq!(ID.raw(), 100);
}

#[test]
fn test_resource_id_hash_collection() {
    let mut set = HashSet::new();
    set.insert(ResourceId::new(1));
    set.insert(ResourceId::new(2));
    set.insert(ResourceId::new(3));
    assert_eq!(set.len(), 3);
    assert!(set.contains(&ResourceId::new(2)));
    assert!(!set.contains(&ResourceId::new(4)));
}

#[test]
fn test_resource_id_hash_invalid_insertable() {
    let mut set = HashSet::new();
    set.insert(ResourceId::INVALID);
    assert!(set.contains(&ResourceId::INVALID));
}

#[test]
fn test_resource_id_eq_same_values() {
    let a = ResourceId::new(50);
    let b = ResourceId::new(50);
    assert_eq!(a, b);
}

#[test]
fn test_resource_id_eq_different_values() {
    let a = ResourceId::new(50);
    let b = ResourceId::new(51);
    assert_ne!(a, b);
}

#[test]
fn test_resource_id_clone() {
    let original = ResourceId::new(999);
    let cloned = original.clone();
    assert_eq!(original, cloned);
}

#[test]
fn test_resource_id_copy() {
    let original = ResourceId::new(123);
    let copied: ResourceId = original; // Copy trait
    assert_eq!(original.raw(), copied.raw());
}

#[test]
fn test_resource_id_debug_format() {
    let id = ResourceId::new(7);
    let debug = format!("{:?}", id);
    assert!(debug.contains("ResourceId"));
    assert!(debug.contains("7"));
}

#[test]
fn test_resource_id_display_normal() {
    let id = ResourceId::new(5);
    assert_eq!(format!("{}", id), "ResourceId(5)");
}

#[test]
fn test_resource_id_display_invalid() {
    assert_eq!(format!("{}", ResourceId::INVALID), "ResourceId::INVALID");
}

#[test]
fn test_resource_id_ordering() {
    let a = ResourceId::new(10);
    let b = ResourceId::new(20);
    assert!(a < b);
    assert!(b > a);
}

// ===========================================================================
// Section 2: PassId Tests (15+)
// ===========================================================================

#[test]
fn test_pass_id_new_zero() {
    let id = PassId::new(0);
    assert_eq!(id.raw(), 0);
    assert!(!id.is_invalid());
}

#[test]
fn test_pass_id_new_arbitrary() {
    let id = PassId::new(100);
    assert_eq!(id.raw(), 100);
}

#[test]
fn test_pass_id_new_large_value() {
    let id = PassId::new(u64::MAX - 1);
    assert_eq!(id.raw(), u64::MAX - 1);
    assert!(!id.is_invalid());
}

#[test]
fn test_pass_id_invalid_sentinel() {
    let id = PassId::INVALID;
    assert!(id.is_invalid());
    assert_eq!(id.raw(), u64::MAX);
}

#[test]
fn test_pass_id_default_is_invalid() {
    let id = PassId::default();
    assert!(id.is_invalid());
    assert_eq!(id, PassId::INVALID);
}

#[test]
fn test_pass_id_raw_accessor() {
    let id = PassId::new(77);
    assert_eq!(id.raw(), 77);
}

#[test]
fn test_pass_id_hash_collection() {
    let mut set = HashSet::new();
    set.insert(PassId::new(1));
    set.insert(PassId::new(2));
    set.insert(PassId::new(1)); // Duplicate
    assert_eq!(set.len(), 2);
}

#[test]
fn test_pass_id_eq_same_values() {
    let a = PassId::new(33);
    let b = PassId::new(33);
    assert_eq!(a, b);
}

#[test]
fn test_pass_id_eq_different_values() {
    let a = PassId::new(33);
    let b = PassId::new(34);
    assert_ne!(a, b);
}

#[test]
fn test_pass_id_clone() {
    let original = PassId::new(888);
    let cloned = original.clone();
    assert_eq!(original, cloned);
}

#[test]
fn test_pass_id_copy() {
    let original = PassId::new(456);
    let copied: PassId = original; // Copy trait
    assert_eq!(original.raw(), copied.raw());
}

#[test]
fn test_pass_id_debug_format() {
    let id = PassId::new(11);
    let debug = format!("{:?}", id);
    assert!(debug.contains("PassId"));
    assert!(debug.contains("11"));
}

#[test]
fn test_pass_id_display_normal() {
    let id = PassId::new(15);
    assert_eq!(format!("{}", id), "PassId(15)");
}

#[test]
fn test_pass_id_display_invalid() {
    assert_eq!(format!("{}", PassId::INVALID), "PassId::INVALID");
}

#[test]
fn test_pass_id_ordering_less_than() {
    let a = PassId::new(5);
    let b = PassId::new(10);
    assert!(a < b);
}

#[test]
fn test_pass_id_ordering_greater_than() {
    let a = PassId::new(100);
    let b = PassId::new(50);
    assert!(a > b);
}

#[test]
fn test_pass_id_ordering_equal() {
    let a = PassId::new(25);
    let b = PassId::new(25);
    assert!(a <= b);
    assert!(a >= b);
}

// ===========================================================================
// Section 3: ResourceAccess Tests (20+)
// ===========================================================================

#[test]
fn test_resource_access_read_is_read() {
    assert!(ResourceAccess::Read.is_read());
}

#[test]
fn test_resource_access_read_is_not_write() {
    assert!(!ResourceAccess::Read.is_write());
}

#[test]
fn test_resource_access_write_is_not_read() {
    assert!(!ResourceAccess::Write.is_read());
}

#[test]
fn test_resource_access_write_is_write() {
    assert!(ResourceAccess::Write.is_write());
}

#[test]
fn test_resource_access_readwrite_is_read() {
    assert!(ResourceAccess::ReadWrite.is_read());
}

#[test]
fn test_resource_access_readwrite_is_write() {
    assert!(ResourceAccess::ReadWrite.is_write());
}

#[test]
fn test_resource_access_conflicts_raw_hazard() {
    // RAW: Read After Write
    assert!(ResourceAccess::Read.conflicts_with(&ResourceAccess::Write));
}

#[test]
fn test_resource_access_conflicts_war_hazard() {
    // WAR: Write After Read
    assert!(ResourceAccess::Write.conflicts_with(&ResourceAccess::Read));
}

#[test]
fn test_resource_access_conflicts_waw_hazard() {
    // WAW: Write After Write
    assert!(ResourceAccess::Write.conflicts_with(&ResourceAccess::Write));
}

#[test]
fn test_resource_access_no_conflict_rar() {
    // RAR: Read After Read (no conflict)
    assert!(!ResourceAccess::Read.conflicts_with(&ResourceAccess::Read));
}

#[test]
fn test_resource_access_readwrite_conflicts_with_read() {
    assert!(ResourceAccess::ReadWrite.conflicts_with(&ResourceAccess::Read));
}

#[test]
fn test_resource_access_readwrite_conflicts_with_write() {
    assert!(ResourceAccess::ReadWrite.conflicts_with(&ResourceAccess::Write));
}

#[test]
fn test_resource_access_readwrite_conflicts_with_readwrite() {
    assert!(ResourceAccess::ReadWrite.conflicts_with(&ResourceAccess::ReadWrite));
}

#[test]
fn test_resource_access_read_conflicts_with_readwrite() {
    // ReadWrite includes write, so Read (RAW) conflicts
    assert!(ResourceAccess::Read.conflicts_with(&ResourceAccess::ReadWrite));
}

#[test]
fn test_resource_access_write_conflicts_with_readwrite() {
    // WAW (write portion) or WAR (read portion)
    assert!(ResourceAccess::Write.conflicts_with(&ResourceAccess::ReadWrite));
}

#[test]
fn test_resource_access_display_read() {
    assert_eq!(format!("{}", ResourceAccess::Read), "Read");
}

#[test]
fn test_resource_access_display_write() {
    assert_eq!(format!("{}", ResourceAccess::Write), "Write");
}

#[test]
fn test_resource_access_display_readwrite() {
    assert_eq!(format!("{}", ResourceAccess::ReadWrite), "ReadWrite");
}

#[test]
fn test_resource_access_default() {
    assert_eq!(ResourceAccess::default(), ResourceAccess::Read);
}

#[test]
fn test_resource_access_hash() {
    let mut set = HashSet::new();
    set.insert(ResourceAccess::Read);
    set.insert(ResourceAccess::Write);
    set.insert(ResourceAccess::ReadWrite);
    assert_eq!(set.len(), 3);
}

#[test]
fn test_resource_access_clone() {
    let original = ResourceAccess::ReadWrite;
    let cloned = original.clone();
    assert_eq!(original, cloned);
}

#[test]
fn test_resource_access_copy() {
    let original = ResourceAccess::Write;
    let copied: ResourceAccess = original;
    assert_eq!(original, copied);
}

// ===========================================================================
// Section 4: ResourceUsage Tests (15+)
// ===========================================================================

#[test]
fn test_resource_usage_new() {
    let res = ResourceId::new(1);
    let usage = ResourceUsage::new(res, ResourceAccess::Read, PipelineStage::FragmentShader);
    assert_eq!(usage.resource, res);
    assert_eq!(usage.access, ResourceAccess::Read);
    assert_eq!(usage.stage, PipelineStage::FragmentShader);
}

#[test]
fn test_resource_usage_read_helper() {
    let res = ResourceId::new(2);
    let usage = ResourceUsage::read(res);
    assert_eq!(usage.resource, res);
    assert!(usage.access.is_read());
    assert!(!usage.access.is_write());
}

#[test]
fn test_resource_usage_write_helper() {
    let res = ResourceId::new(3);
    let usage = ResourceUsage::write(res);
    assert_eq!(usage.resource, res);
    assert!(usage.access.is_write());
    assert!(!usage.access.is_read());
}

#[test]
fn test_resource_usage_read_write_helper() {
    let res = ResourceId::new(4);
    let usage = ResourceUsage::read_write(res);
    assert!(usage.access.is_read());
    assert!(usage.access.is_write());
}

#[test]
fn test_resource_usage_vertex_input_stage() {
    let res = ResourceId::new(5);
    let usage = ResourceUsage::new(res, ResourceAccess::Read, PipelineStage::VertexInput);
    assert_eq!(usage.stage, PipelineStage::VertexInput);
}

#[test]
fn test_resource_usage_vertex_shader_stage() {
    let res = ResourceId::new(6);
    let usage = ResourceUsage::new(res, ResourceAccess::Read, PipelineStage::VertexShader);
    assert_eq!(usage.stage, PipelineStage::VertexShader);
}

#[test]
fn test_resource_usage_compute_shader_stage() {
    let res = ResourceId::new(7);
    let usage = ResourceUsage::new(res, ResourceAccess::ReadWrite, PipelineStage::ComputeShader);
    assert_eq!(usage.stage, PipelineStage::ComputeShader);
}

#[test]
fn test_resource_usage_transfer_stage() {
    let res = ResourceId::new(8);
    let usage = ResourceUsage::new(res, ResourceAccess::Write, PipelineStage::Transfer);
    assert_eq!(usage.stage, PipelineStage::Transfer);
}

#[test]
fn test_resource_usage_color_output_stage() {
    let res = ResourceId::new(9);
    let usage = ResourceUsage::new(res, ResourceAccess::Write, PipelineStage::ColorOutput);
    assert_eq!(usage.stage, PipelineStage::ColorOutput);
}

#[test]
fn test_resource_usage_early_depth_stage() {
    let res = ResourceId::new(10);
    let usage = ResourceUsage::new(res, ResourceAccess::Read, PipelineStage::EarlyDepth);
    assert_eq!(usage.stage, PipelineStage::EarlyDepth);
}

#[test]
fn test_resource_usage_late_depth_stage() {
    let res = ResourceId::new(11);
    let usage = ResourceUsage::new(res, ResourceAccess::Write, PipelineStage::LateDepth);
    assert_eq!(usage.stage, PipelineStage::LateDepth);
}

#[test]
fn test_resource_usage_display() {
    let res = ResourceId::new(1);
    let usage = ResourceUsage::new(res, ResourceAccess::Read, PipelineStage::FragmentShader);
    let display = format!("{}", usage);
    assert!(display.contains("ResourceUsage"));
    assert!(display.contains("Read"));
}

#[test]
fn test_resource_usage_eq() {
    let res = ResourceId::new(1);
    let a = ResourceUsage::new(res, ResourceAccess::Read, PipelineStage::FragmentShader);
    let b = ResourceUsage::new(res, ResourceAccess::Read, PipelineStage::FragmentShader);
    assert_eq!(a, b);
}

#[test]
fn test_resource_usage_ne_different_access() {
    let res = ResourceId::new(1);
    let a = ResourceUsage::new(res, ResourceAccess::Read, PipelineStage::FragmentShader);
    let b = ResourceUsage::new(res, ResourceAccess::Write, PipelineStage::FragmentShader);
    assert_ne!(a, b);
}

#[test]
fn test_resource_usage_clone() {
    let res = ResourceId::new(1);
    let original = ResourceUsage::new(res, ResourceAccess::Read, PipelineStage::ComputeShader);
    let cloned = original.clone();
    assert_eq!(original, cloned);
}

// ===========================================================================
// Section 5: PassType Tests (15+)
// ===========================================================================

#[test]
fn test_pass_type_render_is_graphics() {
    assert!(PassType::Render.is_graphics());
}

#[test]
fn test_pass_type_compute_is_not_graphics() {
    assert!(!PassType::Compute.is_graphics());
}

#[test]
fn test_pass_type_transfer_is_not_graphics() {
    assert!(!PassType::Transfer.is_graphics());
}

#[test]
fn test_pass_type_raytracing_is_not_graphics() {
    assert!(!PassType::RayTracing.is_graphics());
}

#[test]
fn test_pass_type_compute_is_compute() {
    assert!(PassType::Compute.is_compute());
}

#[test]
fn test_pass_type_render_is_not_compute() {
    assert!(!PassType::Render.is_compute());
}

#[test]
fn test_pass_type_transfer_is_not_compute() {
    assert!(!PassType::Transfer.is_compute());
}

#[test]
fn test_pass_type_transfer_is_transfer() {
    assert!(PassType::Transfer.is_transfer());
}

#[test]
fn test_pass_type_render_is_not_transfer() {
    assert!(!PassType::Render.is_transfer());
}

#[test]
fn test_pass_type_raytracing_is_raytracing() {
    assert!(PassType::RayTracing.is_raytracing());
}

#[test]
fn test_pass_type_compute_is_not_raytracing() {
    assert!(!PassType::Compute.is_raytracing());
}

#[test]
fn test_pass_type_display_render() {
    assert_eq!(format!("{}", PassType::Render), "Render");
}

#[test]
fn test_pass_type_display_compute() {
    assert_eq!(format!("{}", PassType::Compute), "Compute");
}

#[test]
fn test_pass_type_display_transfer() {
    assert_eq!(format!("{}", PassType::Transfer), "Transfer");
}

#[test]
fn test_pass_type_display_raytracing() {
    assert_eq!(format!("{}", PassType::RayTracing), "RayTracing");
}

#[test]
fn test_pass_type_default() {
    assert_eq!(PassType::default(), PassType::Render);
}

#[test]
fn test_pass_type_hash() {
    let mut set = HashSet::new();
    set.insert(PassType::Render);
    set.insert(PassType::Compute);
    set.insert(PassType::Transfer);
    set.insert(PassType::RayTracing);
    assert_eq!(set.len(), 4);
}

// ===========================================================================
// Section 6: ResourceType Tests (10+)
// ===========================================================================

#[test]
fn test_resource_type_buffer_is_buffer() {
    assert!(ResourceType::Buffer.is_buffer());
}

#[test]
fn test_resource_type_texture2d_is_not_buffer() {
    assert!(!ResourceType::Texture2D.is_buffer());
}

#[test]
fn test_resource_type_texture2d_is_texture() {
    assert!(ResourceType::Texture2D.is_texture());
}

#[test]
fn test_resource_type_texture3d_is_texture() {
    assert!(ResourceType::Texture3D.is_texture());
}

#[test]
fn test_resource_type_texture_cube_is_texture() {
    assert!(ResourceType::TextureCube.is_texture());
}

#[test]
fn test_resource_type_texture2d_array_is_texture() {
    assert!(ResourceType::Texture2DArray.is_texture());
}

#[test]
fn test_resource_type_buffer_is_not_texture() {
    assert!(!ResourceType::Buffer.is_texture());
}

#[test]
fn test_resource_type_acceleration_structure_is_accel() {
    assert!(ResourceType::AccelerationStructure.is_acceleration_structure());
}

#[test]
fn test_resource_type_buffer_is_not_accel() {
    assert!(!ResourceType::Buffer.is_acceleration_structure());
}

#[test]
fn test_resource_type_display_buffer() {
    assert_eq!(format!("{}", ResourceType::Buffer), "Buffer");
}

#[test]
fn test_resource_type_display_texture2d() {
    assert_eq!(format!("{}", ResourceType::Texture2D), "Texture2D");
}

#[test]
fn test_resource_type_display_texture3d() {
    assert_eq!(format!("{}", ResourceType::Texture3D), "Texture3D");
}

#[test]
fn test_resource_type_display_cube() {
    assert_eq!(format!("{}", ResourceType::TextureCube), "TextureCube");
}

#[test]
fn test_resource_type_display_array() {
    assert_eq!(format!("{}", ResourceType::Texture2DArray), "Texture2DArray");
}

#[test]
fn test_resource_type_display_accel() {
    assert_eq!(
        format!("{}", ResourceType::AccelerationStructure),
        "AccelerationStructure"
    );
}

#[test]
fn test_resource_type_default() {
    assert_eq!(ResourceType::default(), ResourceType::Texture2D);
}

// ===========================================================================
// Section 7: GraphResourceLifetime Tests (10+)
// ===========================================================================

#[test]
fn test_lifetime_transient_is_transient() {
    assert!(GraphResourceLifetime::Transient.is_transient());
}

#[test]
fn test_lifetime_persistent_is_not_transient() {
    assert!(!GraphResourceLifetime::Persistent.is_transient());
}

#[test]
fn test_lifetime_imported_is_not_transient() {
    assert!(!GraphResourceLifetime::Imported.is_transient());
}

#[test]
fn test_lifetime_persistent_is_persistent() {
    assert!(GraphResourceLifetime::Persistent.is_persistent());
}

#[test]
fn test_lifetime_transient_is_not_persistent() {
    assert!(!GraphResourceLifetime::Transient.is_persistent());
}

#[test]
fn test_lifetime_imported_is_imported() {
    assert!(GraphResourceLifetime::Imported.is_imported());
}

#[test]
fn test_lifetime_transient_is_not_imported() {
    assert!(!GraphResourceLifetime::Transient.is_imported());
}

#[test]
fn test_lifetime_transient_can_alias() {
    assert!(GraphResourceLifetime::Transient.can_alias());
}

#[test]
fn test_lifetime_persistent_cannot_alias() {
    assert!(!GraphResourceLifetime::Persistent.can_alias());
}

#[test]
fn test_lifetime_imported_cannot_alias() {
    assert!(!GraphResourceLifetime::Imported.can_alias());
}

#[test]
fn test_lifetime_display_transient() {
    assert_eq!(
        format!("{}", GraphResourceLifetime::Transient),
        "Transient"
    );
}

#[test]
fn test_lifetime_display_persistent() {
    assert_eq!(
        format!("{}", GraphResourceLifetime::Persistent),
        "Persistent"
    );
}

#[test]
fn test_lifetime_display_imported() {
    assert_eq!(format!("{}", GraphResourceLifetime::Imported), "Imported");
}

#[test]
fn test_lifetime_default() {
    assert_eq!(
        GraphResourceLifetime::default(),
        GraphResourceLifetime::Transient
    );
}

// ===========================================================================
// Section 8: PassNode Tests (20+)
// ===========================================================================

#[test]
fn test_pass_node_new_with_id() {
    let pass = PassNode::new(PassId::new(0), "test_pass", PassType::Render);
    assert_eq!(pass.id, PassId::new(0));
}

#[test]
fn test_pass_node_new_with_name() {
    let pass = PassNode::new(PassId::new(0), "my_render_pass", PassType::Render);
    assert_eq!(pass.name, "my_render_pass");
}

#[test]
fn test_pass_node_new_with_pass_type() {
    let pass = PassNode::new(PassId::new(0), "compute", PassType::Compute);
    assert_eq!(pass.pass_type, PassType::Compute);
}

#[test]
fn test_pass_node_new_empty_inputs() {
    let pass = PassNode::new(PassId::new(0), "test", PassType::Render);
    assert!(pass.inputs.is_empty());
}

#[test]
fn test_pass_node_new_empty_outputs() {
    let pass = PassNode::new(PassId::new(0), "test", PassType::Render);
    assert!(pass.outputs.is_empty());
}

#[test]
fn test_pass_node_new_enabled_by_default() {
    let pass = PassNode::new(PassId::new(0), "test", PassType::Render);
    assert!(pass.enabled);
}

#[test]
fn test_pass_node_add_input_increases_count() {
    let mut pass = PassNode::new(PassId::new(0), "test", PassType::Compute);
    let res = ResourceId::new(1);
    pass.add_input(ResourceUsage::read(res));
    assert_eq!(pass.inputs.len(), 1);
}

#[test]
fn test_pass_node_add_input_stores_correct_resource() {
    let mut pass = PassNode::new(PassId::new(0), "test", PassType::Compute);
    let res = ResourceId::new(42);
    pass.add_input(ResourceUsage::read(res));
    assert_eq!(pass.inputs[0].resource, res);
}

#[test]
fn test_pass_node_add_input_resource() {
    let mut pass = PassNode::new(PassId::new(0), "test", PassType::Compute);
    let res = ResourceId::new(1);
    pass.add_input_resource(res, PipelineStage::ComputeShader);
    assert_eq!(pass.inputs.len(), 1);
    assert_eq!(pass.inputs[0].access, ResourceAccess::Read);
}

#[test]
fn test_pass_node_add_output_increases_count() {
    let mut pass = PassNode::new(PassId::new(0), "test", PassType::Render);
    let res = ResourceId::new(1);
    pass.add_output(ResourceUsage::write(res));
    assert_eq!(pass.outputs.len(), 1);
}

#[test]
fn test_pass_node_add_output_stores_correct_resource() {
    let mut pass = PassNode::new(PassId::new(0), "test", PassType::Render);
    let res = ResourceId::new(99);
    pass.add_output(ResourceUsage::write(res));
    assert_eq!(pass.outputs[0].resource, res);
}

#[test]
fn test_pass_node_add_output_resource() {
    let mut pass = PassNode::new(PassId::new(0), "test", PassType::Render);
    let res = ResourceId::new(1);
    pass.add_output_resource(res, PipelineStage::ColorOutput);
    assert_eq!(pass.outputs.len(), 1);
    assert_eq!(pass.outputs[0].access, ResourceAccess::Write);
}

#[test]
fn test_pass_node_no_callback_initially() {
    let pass = PassNode::new(PassId::new(0), "test", PassType::Render);
    assert!(!pass.has_callback());
}

#[test]
fn test_pass_node_set_callback() {
    let mut pass = PassNode::new(PassId::new(0), "test", PassType::Compute);
    pass.set_callback(|_ctx| {});
    assert!(pass.has_callback());
}

#[test]
fn test_pass_node_take_callback() {
    let mut pass = PassNode::new(PassId::new(0), "test", PassType::Compute);
    pass.set_callback(|_ctx| {});
    let cb = pass.take_callback();
    assert!(cb.is_some());
    assert!(!pass.has_callback());
}

#[test]
fn test_pass_node_take_callback_none_when_empty() {
    let mut pass = PassNode::new(PassId::new(0), "test", PassType::Compute);
    let cb = pass.take_callback();
    assert!(cb.is_none());
}

#[test]
fn test_pass_node_enabled_flag_toggle() {
    let mut pass = PassNode::new(PassId::new(0), "test", PassType::Render);
    assert!(pass.enabled);
    pass.enabled = false;
    assert!(!pass.enabled);
}

#[test]
fn test_pass_node_all_resources() {
    let mut pass = PassNode::new(PassId::new(0), "test", PassType::Compute);
    let res1 = ResourceId::new(1);
    let res2 = ResourceId::new(2);
    pass.add_input_resource(res1, PipelineStage::ComputeShader);
    pass.add_output_resource(res2, PipelineStage::ComputeShader);

    let all: Vec<_> = pass.all_resources().collect();
    assert_eq!(all.len(), 2);
    assert!(all.contains(&res1));
    assert!(all.contains(&res2));
}

#[test]
fn test_pass_node_read_resources() {
    let mut pass = PassNode::new(PassId::new(0), "test", PassType::Compute);
    let res1 = ResourceId::new(1);
    let res2 = ResourceId::new(2);
    pass.add_input_resource(res1, PipelineStage::ComputeShader);
    pass.add_output_resource(res2, PipelineStage::ComputeShader);

    let reads = pass.read_resources();
    assert_eq!(reads.len(), 1);
    assert!(reads.contains(&res1));
}

#[test]
fn test_pass_node_write_resources() {
    let mut pass = PassNode::new(PassId::new(0), "test", PassType::Render);
    let res1 = ResourceId::new(1);
    let res2 = ResourceId::new(2);
    pass.add_input_resource(res1, PipelineStage::FragmentShader);
    pass.add_output_resource(res2, PipelineStage::ColorOutput);

    let writes = pass.write_resources();
    assert_eq!(writes.len(), 1);
    assert!(writes.contains(&res2));
}

#[test]
fn test_pass_node_debug_format() {
    let pass = PassNode::new(PassId::new(5), "debug_test", PassType::Transfer);
    let debug = format!("{:?}", pass);
    assert!(debug.contains("PassNode"));
    assert!(debug.contains("debug_test"));
    assert!(debug.contains("Transfer"));
}

#[test]
fn test_pass_node_display_format() {
    let pass = PassNode::new(PassId::new(5), "display_test", PassType::Compute);
    let display = format!("{}", pass);
    assert!(display.contains("PassNode"));
    assert!(display.contains("display_test"));
    assert!(display.contains("Compute"));
}

// ===========================================================================
// Section 9: ResourceNode Tests (15+)
// ===========================================================================

#[test]
fn test_resource_node_new_with_id() {
    let node = ResourceNode::new(
        ResourceId::new(0),
        "texture",
        ResourceType::Texture2D,
        GraphResourceLifetime::Transient,
    );
    assert_eq!(node.id, ResourceId::new(0));
}

#[test]
fn test_resource_node_new_with_name() {
    let node = ResourceNode::new(
        ResourceId::new(0),
        "my_buffer",
        ResourceType::Buffer,
        GraphResourceLifetime::Persistent,
    );
    assert_eq!(node.name, "my_buffer");
}

#[test]
fn test_resource_node_new_with_type() {
    let node = ResourceNode::new(
        ResourceId::new(0),
        "tex",
        ResourceType::TextureCube,
        GraphResourceLifetime::Transient,
    );
    assert_eq!(node.resource_type, ResourceType::TextureCube);
}

#[test]
fn test_resource_node_new_with_lifetime() {
    let node = ResourceNode::new(
        ResourceId::new(0),
        "swapchain",
        ResourceType::Texture2D,
        GraphResourceLifetime::Imported,
    );
    assert_eq!(node.lifetime, GraphResourceLifetime::Imported);
}

#[test]
fn test_resource_node_no_producer_initially() {
    let node = ResourceNode::new(
        ResourceId::new(0),
        "tex",
        ResourceType::Texture2D,
        GraphResourceLifetime::Transient,
    );
    assert!(node.producer.is_none());
}

#[test]
fn test_resource_node_empty_consumers_initially() {
    let node = ResourceNode::new(
        ResourceId::new(0),
        "tex",
        ResourceType::Texture2D,
        GraphResourceLifetime::Transient,
    );
    assert!(node.consumers.is_empty());
}

#[test]
fn test_resource_node_set_producer() {
    let mut node = ResourceNode::new(
        ResourceId::new(0),
        "tex",
        ResourceType::Texture2D,
        GraphResourceLifetime::Transient,
    );
    node.set_producer(PassId::new(5));
    assert_eq!(node.producer, Some(PassId::new(5)));
}

#[test]
fn test_resource_node_add_consumer() {
    let mut node = ResourceNode::new(
        ResourceId::new(0),
        "buffer",
        ResourceType::Buffer,
        GraphResourceLifetime::Transient,
    );
    node.add_consumer(PassId::new(1));
    assert_eq!(node.consumers.len(), 1);
    assert!(node.consumers.contains(&PassId::new(1)));
}

#[test]
fn test_resource_node_no_duplicate_consumers() {
    let mut node = ResourceNode::new(
        ResourceId::new(0),
        "buffer",
        ResourceType::Buffer,
        GraphResourceLifetime::Transient,
    );
    node.add_consumer(PassId::new(1));
    node.add_consumer(PassId::new(1)); // duplicate
    node.add_consumer(PassId::new(2));
    assert_eq!(node.consumers.len(), 2);
}

#[test]
fn test_resource_node_is_transient() {
    let node = ResourceNode::new(
        ResourceId::new(0),
        "tex",
        ResourceType::Texture2D,
        GraphResourceLifetime::Transient,
    );
    assert!(node.is_transient());
}

#[test]
fn test_resource_node_is_persistent() {
    let node = ResourceNode::new(
        ResourceId::new(0),
        "persistent_buf",
        ResourceType::Buffer,
        GraphResourceLifetime::Persistent,
    );
    assert!(node.is_persistent());
}

#[test]
fn test_resource_node_is_imported() {
    let node = ResourceNode::new(
        ResourceId::new(0),
        "swapchain",
        ResourceType::Texture2D,
        GraphResourceLifetime::Imported,
    );
    assert!(node.is_imported());
}

#[test]
fn test_resource_node_reference_count_zero() {
    let node = ResourceNode::new(
        ResourceId::new(0),
        "tex",
        ResourceType::Texture2D,
        GraphResourceLifetime::Transient,
    );
    assert_eq!(node.reference_count(), 0);
}

#[test]
fn test_resource_node_reference_count_with_producer() {
    let mut node = ResourceNode::new(
        ResourceId::new(0),
        "tex",
        ResourceType::Texture2D,
        GraphResourceLifetime::Transient,
    );
    node.set_producer(PassId::new(1));
    assert_eq!(node.reference_count(), 1);
}

#[test]
fn test_resource_node_reference_count_with_consumers() {
    let mut node = ResourceNode::new(
        ResourceId::new(0),
        "tex",
        ResourceType::Texture2D,
        GraphResourceLifetime::Transient,
    );
    node.add_consumer(PassId::new(1));
    node.add_consumer(PassId::new(2));
    assert_eq!(node.reference_count(), 2);
}

#[test]
fn test_resource_node_reference_count_mixed() {
    let mut node = ResourceNode::new(
        ResourceId::new(0),
        "tex",
        ResourceType::Texture2D,
        GraphResourceLifetime::Transient,
    );
    node.set_producer(PassId::new(0));
    node.add_consumer(PassId::new(1));
    node.add_consumer(PassId::new(2));
    node.add_consumer(PassId::new(3));
    assert_eq!(node.reference_count(), 4);
}

#[test]
fn test_resource_node_display() {
    let node = ResourceNode::new(
        ResourceId::new(5),
        "display_test",
        ResourceType::Buffer,
        GraphResourceLifetime::Persistent,
    );
    let display = format!("{}", node);
    assert!(display.contains("ResourceNode"));
    assert!(display.contains("display_test"));
    assert!(display.contains("Buffer"));
}

// ===========================================================================
// Section 10: FrameGraph Tests (40+)
// ===========================================================================

#[test]
fn test_frame_graph_new_empty() {
    let graph = FrameGraph::new();
    assert_eq!(graph.pass_count(), 0);
    assert_eq!(graph.resource_count(), 0);
}

#[test]
fn test_frame_graph_new_not_compiled() {
    let graph = FrameGraph::new();
    assert!(!graph.is_compiled());
}

#[test]
fn test_frame_graph_default_same_as_new() {
    let a = FrameGraph::new();
    let b = FrameGraph::default();
    assert_eq!(a.pass_count(), b.pass_count());
    assert_eq!(a.resource_count(), b.resource_count());
}

#[test]
fn test_frame_graph_add_pass_returns_unique_id() {
    let mut graph = FrameGraph::new();
    let id1 = graph.add_pass("pass1", PassType::Render);
    let id2 = graph.add_pass("pass2", PassType::Compute);
    assert_ne!(id1, id2);
}

#[test]
fn test_frame_graph_add_pass_increments_count() {
    let mut graph = FrameGraph::new();
    graph.add_pass("pass1", PassType::Render);
    assert_eq!(graph.pass_count(), 1);
    graph.add_pass("pass2", PassType::Render);
    assert_eq!(graph.pass_count(), 2);
}

#[test]
fn test_frame_graph_add_pass_sequential_ids() {
    let mut graph = FrameGraph::new();
    let id0 = graph.add_pass("p0", PassType::Render);
    let id1 = graph.add_pass("p1", PassType::Render);
    let id2 = graph.add_pass("p2", PassType::Render);
    assert_eq!(id0.raw(), 0);
    assert_eq!(id1.raw(), 1);
    assert_eq!(id2.raw(), 2);
}

#[test]
fn test_frame_graph_add_resource_returns_unique_id() {
    let mut graph = FrameGraph::new();
    let id1 = graph.add_resource("res1", ResourceType::Buffer, GraphResourceLifetime::Transient);
    let id2 = graph.add_resource(
        "res2",
        ResourceType::Texture2D,
        GraphResourceLifetime::Persistent,
    );
    assert_ne!(id1, id2);
}

#[test]
fn test_frame_graph_add_resource_increments_count() {
    let mut graph = FrameGraph::new();
    graph.add_resource("r1", ResourceType::Buffer, GraphResourceLifetime::Transient);
    assert_eq!(graph.resource_count(), 1);
    graph.add_resource(
        "r2",
        ResourceType::Texture2D,
        GraphResourceLifetime::Transient,
    );
    assert_eq!(graph.resource_count(), 2);
}

#[test]
fn test_frame_graph_add_resource_sequential_ids() {
    let mut graph = FrameGraph::new();
    let r0 = graph.add_resource("r0", ResourceType::Buffer, GraphResourceLifetime::Transient);
    let r1 = graph.add_resource(
        "r1",
        ResourceType::Texture2D,
        GraphResourceLifetime::Transient,
    );
    assert_eq!(r0.raw(), 0);
    assert_eq!(r1.raw(), 1);
}

#[test]
fn test_frame_graph_get_pass_existing() {
    let mut graph = FrameGraph::new();
    let id = graph.add_pass("test_pass", PassType::Compute);
    let pass = graph.get_pass(id);
    assert!(pass.is_some());
    assert_eq!(pass.unwrap().name, "test_pass");
}

#[test]
fn test_frame_graph_get_pass_nonexistent() {
    let graph = FrameGraph::new();
    let pass = graph.get_pass(PassId::new(999));
    assert!(pass.is_none());
}

#[test]
fn test_frame_graph_get_pass_mut_existing() {
    let mut graph = FrameGraph::new();
    let id = graph.add_pass("mutable_pass", PassType::Render);
    let pass = graph.get_pass_mut(id);
    assert!(pass.is_some());
    pass.unwrap().enabled = false;
    assert!(!graph.get_pass(id).unwrap().enabled);
}

#[test]
fn test_frame_graph_get_pass_mut_invalidates_compile() {
    let mut graph = FrameGraph::new();
    let id = graph.add_pass("pass", PassType::Render);
    graph.compile().unwrap();
    assert!(graph.is_compiled());
    let _ = graph.get_pass_mut(id);
    assert!(!graph.is_compiled());
}

#[test]
fn test_frame_graph_get_resource_existing() {
    let mut graph = FrameGraph::new();
    let id = graph.add_resource("tex", ResourceType::Texture2D, GraphResourceLifetime::Transient);
    let res = graph.get_resource(id);
    assert!(res.is_some());
    assert_eq!(res.unwrap().name, "tex");
}

#[test]
fn test_frame_graph_get_resource_nonexistent() {
    let graph = FrameGraph::new();
    let res = graph.get_resource(ResourceId::new(999));
    assert!(res.is_none());
}

#[test]
fn test_frame_graph_get_resource_mut_invalidates_compile() {
    let mut graph = FrameGraph::new();
    let id = graph.add_resource("buf", ResourceType::Buffer, GraphResourceLifetime::Transient);
    graph.compile().unwrap();
    assert!(graph.is_compiled());
    let _ = graph.get_resource_mut(id);
    assert!(!graph.is_compiled());
}

#[test]
fn test_frame_graph_connect_write() {
    let mut graph = FrameGraph::new();
    let pass = graph.add_pass("writer", PassType::Render);
    let res = graph.add_resource("tex", ResourceType::Texture2D, GraphResourceLifetime::Transient);
    graph.connect(pass, res, ResourceAccess::Write);

    let pass_node = graph.get_pass(pass).unwrap();
    assert_eq!(pass_node.outputs.len(), 1);

    let res_node = graph.get_resource(res).unwrap();
    assert_eq!(res_node.producer, Some(pass));
}

#[test]
fn test_frame_graph_connect_read() {
    let mut graph = FrameGraph::new();
    let pass = graph.add_pass("reader", PassType::Compute);
    let res = graph.add_resource("buf", ResourceType::Buffer, GraphResourceLifetime::Transient);
    graph.connect(pass, res, ResourceAccess::Read);

    let pass_node = graph.get_pass(pass).unwrap();
    assert_eq!(pass_node.inputs.len(), 1);

    let res_node = graph.get_resource(res).unwrap();
    assert!(res_node.consumers.contains(&pass));
}

#[test]
fn test_frame_graph_connect_readwrite() {
    let mut graph = FrameGraph::new();
    let pass = graph.add_pass("modifier", PassType::Compute);
    let res = graph.add_resource("buf", ResourceType::Buffer, GraphResourceLifetime::Transient);
    graph.connect(pass, res, ResourceAccess::ReadWrite);

    let pass_node = graph.get_pass(pass).unwrap();
    assert!(!pass_node.inputs.is_empty()); // read portion
    assert!(!pass_node.outputs.is_empty()); // write portion
}

#[test]
fn test_frame_graph_connect_invalidates_compile() {
    let mut graph = FrameGraph::new();
    let pass = graph.add_pass("p", PassType::Render);
    let res = graph.add_resource("r", ResourceType::Texture2D, GraphResourceLifetime::Transient);
    graph.compile().unwrap();
    graph.connect(pass, res, ResourceAccess::Write);
    assert!(!graph.is_compiled());
}

#[test]
fn test_frame_graph_compile_empty_succeeds() {
    let mut graph = FrameGraph::new();
    assert!(graph.compile().is_ok());
    assert!(graph.is_compiled());
}

#[test]
fn test_frame_graph_compile_single_pass() {
    let mut graph = FrameGraph::new();
    let pass = graph.add_pass("single", PassType::Render);
    let res = graph.add_resource("out", ResourceType::Texture2D, GraphResourceLifetime::Transient);
    graph.connect(pass, res, ResourceAccess::Write);
    assert!(graph.compile().is_ok());
    assert!(graph.execution_order().contains(&pass));
}

#[test]
fn test_frame_graph_compile_topological_sort() {
    let mut graph = FrameGraph::new();

    let a = graph.add_pass("A", PassType::Render);
    let b = graph.add_pass("B", PassType::Render);
    let res = graph.add_resource("r", ResourceType::Texture2D, GraphResourceLifetime::Transient);

    graph.connect(a, res, ResourceAccess::Write);
    graph.connect(b, res, ResourceAccess::Read);

    assert!(graph.compile().is_ok());

    let order = graph.execution_order();
    let pos_a = order.iter().position(|&id| id == a).unwrap();
    let pos_b = order.iter().position(|&id| id == b).unwrap();
    assert!(pos_a < pos_b);
}

#[test]
fn test_frame_graph_compile_cycle_detection() {
    let mut graph = FrameGraph::new();

    let a = graph.add_pass("A", PassType::Render);
    let b = graph.add_pass("B", PassType::Render);

    let r1 = graph.add_resource("r1", ResourceType::Texture2D, GraphResourceLifetime::Transient);
    let r2 = graph.add_resource("r2", ResourceType::Texture2D, GraphResourceLifetime::Transient);

    // A -> B via r1
    graph.connect(a, r1, ResourceAccess::Write);
    graph.connect(b, r1, ResourceAccess::Read);

    // B -> A via r2 (creates cycle)
    graph.connect(b, r2, ResourceAccess::Write);
    graph.connect(a, r2, ResourceAccess::Read);

    let result = graph.compile();
    assert!(matches!(result, Err(FrameGraphError::CyclicDependency)));
}

#[test]
fn test_frame_graph_compile_missing_resource_error() {
    let mut graph = FrameGraph::new();
    let pass = graph.add_pass("broken", PassType::Render);

    // Manually insert a reference to non-existent resource
    if let Some(p) = graph.get_pass_mut(pass) {
        p.add_input(ResourceUsage::read(ResourceId::new(999)));
    }

    let result = graph.compile();
    assert!(matches!(result, Err(FrameGraphError::MissingResource(_))));
}

#[test]
fn test_frame_graph_execute_not_compiled_error() {
    let mut graph = FrameGraph::new();
    graph.add_pass("test", PassType::Render);
    let mut ctx = RenderContext::new(0);
    let result = graph.execute(&mut ctx);
    assert!(matches!(result, Err(FrameGraphError::NotCompiled)));
}

#[test]
fn test_frame_graph_execute_runs_passes_in_order() {
    let counter = Arc::new(AtomicU32::new(0));

    let mut graph = FrameGraph::new();

    let a = graph.add_pass("A", PassType::Render);
    let b = graph.add_pass("B", PassType::Render);
    let res = graph.add_resource("r", ResourceType::Texture2D, GraphResourceLifetime::Transient);

    graph.connect(a, res, ResourceAccess::Write);
    graph.connect(b, res, ResourceAccess::Read);

    // A sets counter to 1
    let counter_a = Arc::clone(&counter);
    graph.get_pass_mut(a).unwrap().set_callback(move |_ctx| {
        counter_a.store(1, Ordering::SeqCst);
    });

    // B multiplies by 10 (should get 10 if A ran first)
    let counter_b = Arc::clone(&counter);
    graph.get_pass_mut(b).unwrap().set_callback(move |_ctx| {
        let val = counter_b.load(Ordering::SeqCst);
        counter_b.store(val * 10, Ordering::SeqCst);
    });

    graph.compile().unwrap();

    let mut ctx = RenderContext::new(0);
    graph.execute(&mut ctx).unwrap();

    assert_eq!(counter.load(Ordering::SeqCst), 10);
}

#[test]
fn test_frame_graph_execute_skips_disabled_passes() {
    let executed = Arc::new(AtomicU32::new(0));

    let mut graph = FrameGraph::new();
    let pass = graph.add_pass("disabled", PassType::Render);

    let executed_clone = Arc::clone(&executed);
    if let Some(p) = graph.get_pass_mut(pass) {
        p.enabled = false;
        p.set_callback(move |_ctx| {
            executed_clone.fetch_add(1, Ordering::SeqCst);
        });
    }

    graph.compile().unwrap();

    let mut ctx = RenderContext::new(0);
    graph.execute(&mut ctx).unwrap();

    assert_eq!(executed.load(Ordering::SeqCst), 0);
}

#[test]
fn test_frame_graph_reset_clears_compiled() {
    let mut graph = FrameGraph::new();
    let pass = graph.add_pass("p", PassType::Render);
    let res = graph.add_resource("r", ResourceType::Texture2D, GraphResourceLifetime::Transient);
    graph.connect(pass, res, ResourceAccess::Write);
    graph.compile().unwrap();

    graph.reset();

    assert!(!graph.is_compiled());
}

#[test]
fn test_frame_graph_reset_preserves_passes() {
    let mut graph = FrameGraph::new();
    graph.add_pass("pass1", PassType::Render);
    graph.add_pass("pass2", PassType::Compute);

    graph.reset();

    assert_eq!(graph.pass_count(), 2);
}

#[test]
fn test_frame_graph_reset_preserves_resources() {
    let mut graph = FrameGraph::new();
    graph.add_resource("r1", ResourceType::Buffer, GraphResourceLifetime::Transient);
    graph.add_resource("r2", ResourceType::Texture2D, GraphResourceLifetime::Transient);

    graph.reset();

    assert_eq!(graph.resource_count(), 2);
}

#[test]
fn test_frame_graph_reset_clears_connections() {
    let mut graph = FrameGraph::new();
    let pass = graph.add_pass("p", PassType::Render);
    let res = graph.add_resource("r", ResourceType::Texture2D, GraphResourceLifetime::Transient);
    graph.connect(pass, res, ResourceAccess::Write);

    graph.reset();

    assert!(graph.get_pass(pass).unwrap().inputs.is_empty());
    assert!(graph.get_pass(pass).unwrap().outputs.is_empty());
    assert!(graph.get_resource(res).unwrap().producer.is_none());
    assert!(graph.get_resource(res).unwrap().consumers.is_empty());
}

#[test]
fn test_frame_graph_clear() {
    let mut graph = FrameGraph::new();
    graph.add_pass("p1", PassType::Render);
    graph.add_pass("p2", PassType::Compute);
    graph.add_resource("r1", ResourceType::Buffer, GraphResourceLifetime::Transient);

    graph.clear();

    assert_eq!(graph.pass_count(), 0);
    assert_eq!(graph.resource_count(), 0);
}

#[test]
fn test_frame_graph_clear_resets_ids() {
    let mut graph = FrameGraph::new();
    graph.add_pass("p", PassType::Render);
    graph.add_resource("r", ResourceType::Buffer, GraphResourceLifetime::Transient);

    graph.clear();

    // After clear, IDs should restart from 0
    let new_pass = graph.add_pass("new_p", PassType::Render);
    let new_res = graph.add_resource("new_r", ResourceType::Buffer, GraphResourceLifetime::Transient);

    assert_eq!(new_pass.raw(), 0);
    assert_eq!(new_res.raw(), 0);
}

#[test]
fn test_frame_graph_pass_count() {
    let mut graph = FrameGraph::new();
    assert_eq!(graph.pass_count(), 0);
    graph.add_pass("p1", PassType::Render);
    assert_eq!(graph.pass_count(), 1);
    graph.add_pass("p2", PassType::Compute);
    graph.add_pass("p3", PassType::Transfer);
    assert_eq!(graph.pass_count(), 3);
}

#[test]
fn test_frame_graph_resource_count() {
    let mut graph = FrameGraph::new();
    assert_eq!(graph.resource_count(), 0);
    graph.add_resource("r1", ResourceType::Buffer, GraphResourceLifetime::Transient);
    assert_eq!(graph.resource_count(), 1);
    graph.add_resource("r2", ResourceType::Texture2D, GraphResourceLifetime::Persistent);
    assert_eq!(graph.resource_count(), 2);
}

#[test]
fn test_frame_graph_diamond_pattern() {
    //     A
    //    / \
    //   B   C
    //    \ /
    //     D
    let mut graph = FrameGraph::new();

    let a = graph.add_pass("A", PassType::Render);
    let b = graph.add_pass("B", PassType::Compute);
    let c = graph.add_pass("C", PassType::Compute);
    let d = graph.add_pass("D", PassType::Render);

    let r_ab = graph.add_resource("r_ab", ResourceType::Buffer, GraphResourceLifetime::Transient);
    let r_ac = graph.add_resource("r_ac", ResourceType::Buffer, GraphResourceLifetime::Transient);
    let r_bd = graph.add_resource("r_bd", ResourceType::Buffer, GraphResourceLifetime::Transient);
    let r_cd = graph.add_resource("r_cd", ResourceType::Buffer, GraphResourceLifetime::Transient);

    graph.connect(a, r_ab, ResourceAccess::Write);
    graph.connect(b, r_ab, ResourceAccess::Read);
    graph.connect(a, r_ac, ResourceAccess::Write);
    graph.connect(c, r_ac, ResourceAccess::Read);
    graph.connect(b, r_bd, ResourceAccess::Write);
    graph.connect(d, r_bd, ResourceAccess::Read);
    graph.connect(c, r_cd, ResourceAccess::Write);
    graph.connect(d, r_cd, ResourceAccess::Read);

    assert!(graph.compile().is_ok());

    let order = graph.execution_order();
    let pos_a = order.iter().position(|&id| id == a).unwrap();
    let pos_b = order.iter().position(|&id| id == b).unwrap();
    let pos_c = order.iter().position(|&id| id == c).unwrap();
    let pos_d = order.iter().position(|&id| id == d).unwrap();

    assert!(pos_a < pos_b);
    assert!(pos_a < pos_c);
    assert!(pos_b < pos_d);
    assert!(pos_c < pos_d);
}

#[test]
fn test_frame_graph_linear_chain() {
    let mut graph = FrameGraph::new();

    let passes: Vec<_> = (0..5)
        .map(|i| graph.add_pass(format!("pass_{}", i), PassType::Compute))
        .collect();

    for i in 0..4 {
        let res = graph.add_resource(
            format!("res_{}", i),
            ResourceType::Buffer,
            GraphResourceLifetime::Transient,
        );
        graph.connect(passes[i], res, ResourceAccess::Write);
        graph.connect(passes[i + 1], res, ResourceAccess::Read);
    }

    assert!(graph.compile().is_ok());

    let order = graph.execution_order();
    for i in 0..4 {
        let pos_i = order.iter().position(|&id| id == passes[i]).unwrap();
        let pos_next = order.iter().position(|&id| id == passes[i + 1]).unwrap();
        assert!(pos_i < pos_next);
    }
}

#[test]
fn test_frame_graph_find_writers() {
    let mut graph = FrameGraph::new();

    let p1 = graph.add_pass("writer", PassType::Render);
    let p2 = graph.add_pass("reader", PassType::Render);

    let res = graph.add_resource("tex", ResourceType::Texture2D, GraphResourceLifetime::Transient);

    graph.connect(p1, res, ResourceAccess::Write);
    graph.connect(p2, res, ResourceAccess::Read);

    let writers = graph.find_writers(res);
    assert_eq!(writers.len(), 1);
    assert!(writers.contains(&p1));
}

#[test]
fn test_frame_graph_find_readers() {
    let mut graph = FrameGraph::new();

    let writer = graph.add_pass("writer", PassType::Render);
    let reader1 = graph.add_pass("reader1", PassType::Render);
    let reader2 = graph.add_pass("reader2", PassType::Compute);

    let res = graph.add_resource("shared", ResourceType::Texture2D, GraphResourceLifetime::Transient);

    graph.connect(writer, res, ResourceAccess::Write);
    graph.connect(reader1, res, ResourceAccess::Read);
    graph.connect(reader2, res, ResourceAccess::Read);

    let readers = graph.find_readers(res);
    assert_eq!(readers.len(), 2);
    assert!(readers.contains(&reader1));
    assert!(readers.contains(&reader2));
}

#[test]
fn test_frame_graph_validate() {
    let mut graph = FrameGraph::new();
    let pass = graph.add_pass("valid", PassType::Render);
    let res = graph.add_resource("tex", ResourceType::Texture2D, GraphResourceLifetime::Transient);
    graph.connect(pass, res, ResourceAccess::Write);

    assert!(graph.validate().is_ok());
}

#[test]
fn test_frame_graph_validate_missing_resource() {
    let mut graph = FrameGraph::new();
    let pass = graph.add_pass("broken", PassType::Render);

    if let Some(p) = graph.get_pass_mut(pass) {
        p.add_input(ResourceUsage::read(ResourceId::new(999)));
    }

    let result = graph.validate();
    assert!(matches!(result, Err(FrameGraphError::MissingResource(_))));
}

#[test]
fn test_frame_graph_execution_order_accessor() {
    let mut graph = FrameGraph::new();
    let p = graph.add_pass("p", PassType::Render);
    graph.compile().unwrap();

    let order = graph.execution_order();
    assert!(order.contains(&p));
}

#[test]
fn test_frame_graph_passes_iterator() {
    let mut graph = FrameGraph::new();
    graph.add_pass("p1", PassType::Render);
    graph.add_pass("p2", PassType::Compute);
    graph.add_pass("p3", PassType::Transfer);

    let passes: Vec<_> = graph.passes().collect();
    assert_eq!(passes.len(), 3);
}

#[test]
fn test_frame_graph_resources_iterator() {
    let mut graph = FrameGraph::new();
    graph.add_resource("r1", ResourceType::Buffer, GraphResourceLifetime::Transient);
    graph.add_resource("r2", ResourceType::Texture2D, GraphResourceLifetime::Persistent);

    let resources: Vec<_> = graph.resources().collect();
    assert_eq!(resources.len(), 2);
}

#[test]
fn test_frame_graph_debug_format() {
    let mut graph = FrameGraph::new();
    graph.add_pass("p", PassType::Render);
    graph.add_resource("r", ResourceType::Buffer, GraphResourceLifetime::Transient);

    let debug = format!("{:?}", graph);
    assert!(debug.contains("FrameGraph"));
    assert!(debug.contains("passes"));
    assert!(debug.contains("resources"));
}

#[test]
fn test_frame_graph_display_format() {
    let mut graph = FrameGraph::new();
    graph.add_pass("p1", PassType::Render);
    graph.add_pass("p2", PassType::Compute);
    graph.add_resource("r", ResourceType::Buffer, GraphResourceLifetime::Transient);

    let display = format!("{}", graph);
    assert!(display.contains("FrameGraph"));
    assert!(display.contains("passes=2"));
    assert!(display.contains("resources=1"));
}

// ===========================================================================
// Section 11: FrameGraphError Tests (10+)
// ===========================================================================

#[test]
fn test_error_cyclic_dependency_display() {
    let err = FrameGraphError::CyclicDependency;
    let msg = format!("{}", err);
    assert!(msg.contains("Cyclic"));
}

#[test]
fn test_error_missing_resource_display() {
    let err = FrameGraphError::MissingResource(ResourceId::new(42));
    let msg = format!("{}", err);
    assert!(msg.contains("Missing"));
    assert!(msg.contains("42"));
}

#[test]
fn test_error_missing_pass_display() {
    let err = FrameGraphError::MissingPass(PassId::new(99));
    let msg = format!("{}", err);
    assert!(msg.contains("Missing pass"));
    assert!(msg.contains("99"));
}

#[test]
fn test_error_invalid_access_display() {
    let err = FrameGraphError::InvalidAccess("test message".to_string());
    let msg = format!("{}", err);
    assert!(msg.contains("Invalid"));
    assert!(msg.contains("test message"));
}

#[test]
fn test_error_not_compiled_display() {
    let err = FrameGraphError::NotCompiled;
    let msg = format!("{}", err);
    assert!(msg.contains("not been compiled"));
}

#[test]
fn test_error_execution_failed_display() {
    let err = FrameGraphError::ExecutionFailed("shader error".to_string());
    let msg = format!("{}", err);
    assert!(msg.contains("execution failed"));
    assert!(msg.contains("shader error"));
}

#[test]
fn test_error_debug_format() {
    let err = FrameGraphError::CyclicDependency;
    let debug = format!("{:?}", err);
    assert!(debug.contains("CyclicDependency"));
}

#[test]
fn test_error_eq() {
    let a = FrameGraphError::NotCompiled;
    let b = FrameGraphError::NotCompiled;
    assert_eq!(a, b);
}

#[test]
fn test_error_ne() {
    let a = FrameGraphError::NotCompiled;
    let b = FrameGraphError::CyclicDependency;
    assert_ne!(a, b);
}

#[test]
fn test_error_is_std_error() {
    let err = FrameGraphError::NotCompiled;
    let _: &dyn std::error::Error = &err;
}

#[test]
fn test_error_clone() {
    let err = FrameGraphError::InvalidAccess("test".to_string());
    let cloned = err.clone();
    assert_eq!(err, cloned);
}

// ===========================================================================
// Section 12: FrameGraphBuilder Tests (15+)
// ===========================================================================

#[test]
fn test_builder_new() {
    let builder = FrameGraphBuilder::new();
    let graph = builder.build_unchecked();
    assert_eq!(graph.pass_count(), 0);
    assert_eq!(graph.resource_count(), 0);
}

#[test]
fn test_builder_default() {
    let builder = FrameGraphBuilder::default();
    let graph = builder.build_unchecked();
    assert_eq!(graph.pass_count(), 0);
}

#[test]
fn test_builder_add_resource() {
    let mut builder = FrameGraphBuilder::new();
    let res = builder.add_resource("tex", ResourceType::Texture2D, GraphResourceLifetime::Transient);
    assert!(!res.is_invalid());
}

#[test]
fn test_builder_add_pass_returns_pass_builder() {
    let mut builder = FrameGraphBuilder::new();
    let _pass_builder = builder.add_pass("render", PassType::Render);
    // Just verify it compiles and returns something
}

#[test]
fn test_builder_pass_builder_read() {
    let mut builder = FrameGraphBuilder::new();
    let res = builder.add_resource("input", ResourceType::Texture2D, GraphResourceLifetime::Transient);
    let pass_id = builder.add_pass("reader", PassType::Render).read(res).build();

    let graph = builder.build_unchecked();
    let pass = graph.get_pass(pass_id).unwrap();
    assert_eq!(pass.inputs.len(), 1);
}

#[test]
fn test_builder_pass_builder_write() {
    let mut builder = FrameGraphBuilder::new();
    let res = builder.add_resource("output", ResourceType::Texture2D, GraphResourceLifetime::Transient);
    let pass_id = builder.add_pass("writer", PassType::Render).write(res).build();

    let graph = builder.build_unchecked();
    let pass = graph.get_pass(pass_id).unwrap();
    assert_eq!(pass.outputs.len(), 1);
}

#[test]
fn test_builder_pass_builder_read_write() {
    let mut builder = FrameGraphBuilder::new();
    let res = builder.add_resource("buf", ResourceType::Buffer, GraphResourceLifetime::Transient);
    let pass_id = builder
        .add_pass("modifier", PassType::Compute)
        .read_write(res)
        .build();

    let graph = builder.build_unchecked();
    let pass = graph.get_pass(pass_id).unwrap();
    assert!(!pass.inputs.is_empty());
    assert!(!pass.outputs.is_empty());
}

#[test]
fn test_builder_pass_builder_callback() {
    let executed = Arc::new(AtomicU64::new(0));
    let executed_clone = Arc::clone(&executed);

    let mut builder = FrameGraphBuilder::new();
    let res = builder.add_resource("tex", ResourceType::Texture2D, GraphResourceLifetime::Transient);

    builder
        .add_pass("with_callback", PassType::Compute)
        .write(res)
        .callback(move |_ctx| {
            executed_clone.store(42, Ordering::SeqCst);
        })
        .build();

    let mut graph = builder.build().unwrap();
    let mut ctx = RenderContext::new(0);
    graph.execute(&mut ctx).unwrap();

    assert_eq!(executed.load(Ordering::SeqCst), 42);
}

#[test]
fn test_builder_pass_builder_disable() {
    let mut builder = FrameGraphBuilder::new();
    let pass_id = builder
        .add_pass("disabled", PassType::Render)
        .disable()
        .build();

    let graph = builder.build_unchecked();
    let pass = graph.get_pass(pass_id).unwrap();
    assert!(!pass.enabled);
}

#[test]
fn test_builder_pass_builder_id() {
    let mut builder = FrameGraphBuilder::new();
    let pass_builder = builder.add_pass("test", PassType::Render);
    let id = pass_builder.id();
    let built_id = pass_builder.build();
    assert_eq!(id, built_id);
}

#[test]
fn test_builder_pass_builder_chaining() {
    let mut builder = FrameGraphBuilder::new();
    let r1 = builder.add_resource("r1", ResourceType::Texture2D, GraphResourceLifetime::Transient);
    let r2 = builder.add_resource("r2", ResourceType::Texture2D, GraphResourceLifetime::Transient);
    let r3 = builder.add_resource("r3", ResourceType::Buffer, GraphResourceLifetime::Transient);

    let pass_id = builder
        .add_pass("multi", PassType::Compute)
        .read(r1)
        .read(r2)
        .write(r3)
        .build();

    let graph = builder.build_unchecked();
    let pass = graph.get_pass(pass_id).unwrap();
    assert_eq!(pass.inputs.len(), 2);
    assert_eq!(pass.outputs.len(), 1);
}

#[test]
fn test_builder_build_compiles() {
    let mut builder = FrameGraphBuilder::new();
    let res = builder.add_resource("tex", ResourceType::Texture2D, GraphResourceLifetime::Transient);
    builder.add_pass("p", PassType::Render).write(res).build();

    let graph = builder.build().unwrap();
    assert!(graph.is_compiled());
}

#[test]
fn test_builder_build_unchecked_does_not_compile() {
    let mut builder = FrameGraphBuilder::new();
    builder.add_pass("p", PassType::Render).build();

    let graph = builder.build_unchecked();
    assert!(!graph.is_compiled());
}

#[test]
fn test_builder_build_returns_error_on_cycle() {
    let mut builder = FrameGraphBuilder::new();

    let r1 = builder.add_resource("r1", ResourceType::Buffer, GraphResourceLifetime::Transient);
    let r2 = builder.add_resource("r2", ResourceType::Buffer, GraphResourceLifetime::Transient);

    let p1 = builder.add_pass("p1", PassType::Render).write(r1).read(r2).build();
    let p2 = builder.add_pass("p2", PassType::Render).write(r2).read(r1).build();

    let result = builder.build();
    assert!(result.is_err());
}

#[test]
fn test_builder_complex_deferred_rendering() {
    let mut builder = FrameGraphBuilder::new();

    // G-Buffer resources
    let gbuf_albedo =
        builder.add_resource("gbuf_albedo", ResourceType::Texture2D, GraphResourceLifetime::Transient);
    let gbuf_normal =
        builder.add_resource("gbuf_normal", ResourceType::Texture2D, GraphResourceLifetime::Transient);
    let gbuf_depth =
        builder.add_resource("gbuf_depth", ResourceType::Texture2D, GraphResourceLifetime::Transient);
    let lighting =
        builder.add_resource("lighting", ResourceType::Texture2D, GraphResourceLifetime::Transient);
    let final_output =
        builder.add_resource("final", ResourceType::Texture2D, GraphResourceLifetime::Imported);

    // G-Buffer pass
    builder
        .add_pass("gbuffer", PassType::Render)
        .write(gbuf_albedo)
        .write(gbuf_normal)
        .write(gbuf_depth)
        .build();

    // Lighting pass
    builder
        .add_pass("lighting", PassType::Compute)
        .read(gbuf_albedo)
        .read(gbuf_normal)
        .read(gbuf_depth)
        .write(lighting)
        .build();

    // Composite pass
    builder
        .add_pass("composite", PassType::Render)
        .read(lighting)
        .write(final_output)
        .build();

    let graph = builder.build().unwrap();
    assert_eq!(graph.pass_count(), 3);
    assert_eq!(graph.resource_count(), 5);
    assert!(graph.is_compiled());

    // Verify execution order
    let order = graph.execution_order();
    assert_eq!(order.len(), 3);
}

// ===========================================================================
// Section 13: RenderContext Tests
// ===========================================================================

#[test]
fn test_render_context_new() {
    let ctx = RenderContext::new(42);
    assert_eq!(ctx.frame_index, 42);
    assert!(ctx.current_pass_label.is_empty());
}

#[test]
fn test_render_context_default() {
    let ctx = RenderContext::default();
    assert_eq!(ctx.frame_index, 0);
}

#[test]
fn test_render_context_pass_label_updated_during_execute() {
    let label_captured = Arc::new(std::sync::Mutex::new(String::new()));
    let label_clone = Arc::clone(&label_captured);

    let mut graph = FrameGraph::new();
    let pass = graph.add_pass("my_labeled_pass", PassType::Render);
    let res = graph.add_resource("r", ResourceType::Texture2D, GraphResourceLifetime::Transient);
    graph.connect(pass, res, ResourceAccess::Write);

    if let Some(p) = graph.get_pass_mut(pass) {
        p.set_callback(move |ctx| {
            *label_clone.lock().unwrap() = ctx.current_pass_label.clone();
        });
    }

    graph.compile().unwrap();

    let mut ctx = RenderContext::new(0);
    graph.execute(&mut ctx).unwrap();

    assert_eq!(*label_captured.lock().unwrap(), "my_labeled_pass");
}

// ===========================================================================
// Section 14: Edge Cases and Stress Tests
// ===========================================================================

#[test]
fn test_large_graph_100_passes() {
    let mut graph = FrameGraph::new();

    for i in 0..100 {
        graph.add_pass(format!("pass_{}", i), PassType::Compute);
    }

    assert_eq!(graph.pass_count(), 100);
    assert!(graph.compile().is_ok());
}

#[test]
fn test_large_graph_100_resources() {
    let mut graph = FrameGraph::new();

    for i in 0..100 {
        graph.add_resource(
            format!("res_{}", i),
            ResourceType::Buffer,
            GraphResourceLifetime::Transient,
        );
    }

    assert_eq!(graph.resource_count(), 100);
}

#[test]
fn test_many_readers_single_writer() {
    let mut graph = FrameGraph::new();

    let writer = graph.add_pass("writer", PassType::Render);
    let res = graph.add_resource("shared", ResourceType::Texture2D, GraphResourceLifetime::Transient);
    graph.connect(writer, res, ResourceAccess::Write);

    for i in 0..20 {
        let reader = graph.add_pass(format!("reader_{}", i), PassType::Compute);
        graph.connect(reader, res, ResourceAccess::Read);
    }

    assert!(graph.compile().is_ok());

    // Writer should come first
    let order = graph.execution_order();
    let writer_pos = order.iter().position(|&id| id == writer).unwrap();

    for i in 0..20 {
        let reader_id = PassId::new(i + 1); // reader IDs start at 1
        let reader_pos = order.iter().position(|&id| id == reader_id).unwrap();
        assert!(writer_pos < reader_pos);
    }
}

#[test]
fn test_isolated_passes_no_resources() {
    let mut graph = FrameGraph::new();

    graph.add_pass("isolated1", PassType::Render);
    graph.add_pass("isolated2", PassType::Compute);
    graph.add_pass("isolated3", PassType::Transfer);

    assert!(graph.compile().is_ok());
    assert_eq!(graph.execution_order().len(), 3);
}

#[test]
fn test_resource_with_no_users() {
    let mut graph = FrameGraph::new();

    graph.add_resource(
        "unused",
        ResourceType::Buffer,
        GraphResourceLifetime::Transient,
    );
    graph.add_pass("pass", PassType::Render);

    assert!(graph.compile().is_ok());
}

#[test]
fn test_pass_with_multiple_outputs() {
    let mut graph = FrameGraph::new();

    let pass = graph.add_pass("multi_out", PassType::Render);
    let r1 = graph.add_resource("r1", ResourceType::Texture2D, GraphResourceLifetime::Transient);
    let r2 = graph.add_resource("r2", ResourceType::Texture2D, GraphResourceLifetime::Transient);
    let r3 = graph.add_resource("r3", ResourceType::Texture2D, GraphResourceLifetime::Transient);

    graph.connect(pass, r1, ResourceAccess::Write);
    graph.connect(pass, r2, ResourceAccess::Write);
    graph.connect(pass, r3, ResourceAccess::Write);

    assert!(graph.compile().is_ok());
    assert_eq!(graph.get_pass(pass).unwrap().outputs.len(), 3);
}

#[test]
fn test_pass_with_multiple_inputs() {
    let mut graph = FrameGraph::new();

    let writer1 = graph.add_pass("w1", PassType::Render);
    let writer2 = graph.add_pass("w2", PassType::Render);
    let writer3 = graph.add_pass("w3", PassType::Render);
    let reader = graph.add_pass("reader", PassType::Compute);

    let r1 = graph.add_resource("r1", ResourceType::Texture2D, GraphResourceLifetime::Transient);
    let r2 = graph.add_resource("r2", ResourceType::Texture2D, GraphResourceLifetime::Transient);
    let r3 = graph.add_resource("r3", ResourceType::Texture2D, GraphResourceLifetime::Transient);

    graph.connect(writer1, r1, ResourceAccess::Write);
    graph.connect(writer2, r2, ResourceAccess::Write);
    graph.connect(writer3, r3, ResourceAccess::Write);

    graph.connect(reader, r1, ResourceAccess::Read);
    graph.connect(reader, r2, ResourceAccess::Read);
    graph.connect(reader, r3, ResourceAccess::Read);

    assert!(graph.compile().is_ok());
    assert_eq!(graph.get_pass(reader).unwrap().inputs.len(), 3);
}

#[test]
fn test_self_loop_prevention() {
    // A pass that reads and writes the same resource shouldn't create a self-cycle
    let mut graph = FrameGraph::new();

    let pass = graph.add_pass("self_modify", PassType::Compute);
    let res = graph.add_resource("buf", ResourceType::Buffer, GraphResourceLifetime::Persistent);

    graph.connect(pass, res, ResourceAccess::ReadWrite);

    // This should compile fine - a pass can read and write the same resource
    assert!(graph.compile().is_ok());
}

#[test]
fn test_connect_with_stage() {
    let mut graph = FrameGraph::new();

    let pass = graph.add_pass("test", PassType::Render);
    let res = graph.add_resource("tex", ResourceType::Texture2D, GraphResourceLifetime::Transient);

    graph.connect_with_stage(pass, res, ResourceAccess::Read, PipelineStage::VertexShader);

    let pass_node = graph.get_pass(pass).unwrap();
    assert_eq!(pass_node.inputs.len(), 1);
    assert_eq!(pass_node.inputs[0].stage, PipelineStage::VertexShader);
}

#[test]
fn test_multiple_compiles() {
    let mut graph = FrameGraph::new();

    let a = graph.add_pass("A", PassType::Render);
    let b = graph.add_pass("B", PassType::Render);
    let res = graph.add_resource("r", ResourceType::Texture2D, GraphResourceLifetime::Transient);

    graph.connect(a, res, ResourceAccess::Write);
    graph.connect(b, res, ResourceAccess::Read);

    // First compile
    assert!(graph.compile().is_ok());
    let order1 = graph.execution_order().to_vec();

    // Modify and recompile
    graph.reset();
    graph.connect(a, res, ResourceAccess::Write);
    graph.connect(b, res, ResourceAccess::Read);
    assert!(graph.compile().is_ok());
    let order2 = graph.execution_order().to_vec();

    // Order should be same (deterministic)
    assert_eq!(order1, order2);
}
