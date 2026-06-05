//! Blackbox integration tests for primitive topologies (T-WGPU-P3.3.1)
//!
//! Tests the public API of primitive state without relying on implementation details.
//! Validates all 5 topologies, vertex/primitive calculations, and edge cases.

use renderer_backend::render_pipeline::{
    get_topology_info, is_list_topology, is_strip_topology, minimum_vertex_count,
    topology_primitive_count, topology_vertex_count, PrimitiveStateDescriptor, TopologyInfo,
    TOPOLOGIES,
};
use wgpu::{Face, FrontFace, IndexFormat, PolygonMode, PrimitiveState, PrimitiveTopology};

// =============================================================================
// Category 1: API Surface Tests
// =============================================================================

#[test]
fn test_topologies_array_accessible() {
    // TOPOLOGIES constant should be publicly accessible
    let topologies: &[TopologyInfo; 5] = &TOPOLOGIES;
    assert_eq!(topologies.len(), 5);
}

#[test]
fn test_all_topologies_documented() {
    assert_eq!(TOPOLOGIES.len(), 5);
    for info in &TOPOLOGIES {
        assert!(!info.name.is_empty(), "Every topology must have a name");
        assert!(
            !info.description.is_empty(),
            "Every topology must have a description"
        );
        assert!(
            !info.use_cases.is_empty(),
            "Every topology must have use cases"
        );
    }
}

#[test]
fn test_topology_info_struct_fields() {
    // Verify TopologyInfo has all required fields via pattern matching
    let info = &TOPOLOGIES[0];
    let TopologyInfo {
        topology,
        name,
        description,
        use_cases,
    } = info;
    assert!(!name.is_empty());
    assert!(!description.is_empty());
    assert!(!use_cases.is_empty());
    let _ = topology; // Use topology to avoid warning
}

#[test]
fn test_primitive_state_descriptor_default_constructor() {
    let state = PrimitiveStateDescriptor::default();
    assert_eq!(state.topology, PrimitiveTopology::TriangleList);
}

#[test]
fn test_primitive_state_descriptor_new_constructor() {
    let state = PrimitiveStateDescriptor::new();
    assert_eq!(state.topology, PrimitiveTopology::TriangleList);
}

// =============================================================================
// Category 2: All Topology Tests
// =============================================================================

#[test]
fn test_point_list_topology_exists() {
    let info = get_topology_info(PrimitiveTopology::PointList);
    assert_eq!(info.name, "PointList");
    assert!(info.description.contains("point"));
}

#[test]
fn test_line_list_topology_exists() {
    let info = get_topology_info(PrimitiveTopology::LineList);
    assert_eq!(info.name, "LineList");
    assert!(info.description.contains("line"));
}

#[test]
fn test_line_strip_topology_exists() {
    let info = get_topology_info(PrimitiveTopology::LineStrip);
    assert_eq!(info.name, "LineStrip");
    assert!(info.description.contains("line"));
}

#[test]
fn test_triangle_list_topology_exists() {
    let info = get_topology_info(PrimitiveTopology::TriangleList);
    assert_eq!(info.name, "TriangleList");
    assert!(info.description.contains("triangle"));
}

#[test]
fn test_triangle_strip_topology_exists() {
    let info = get_topology_info(PrimitiveTopology::TriangleStrip);
    assert_eq!(info.name, "TriangleStrip");
    assert!(info.description.contains("triangle"));
}

#[test]
fn test_topologies_cover_all_wgpu_variants() {
    // Verify TOPOLOGIES includes exactly all 5 wgpu PrimitiveTopology variants
    let expected = [
        PrimitiveTopology::PointList,
        PrimitiveTopology::LineList,
        PrimitiveTopology::LineStrip,
        PrimitiveTopology::TriangleList,
        PrimitiveTopology::TriangleStrip,
    ];
    for expected_topo in &expected {
        let found = TOPOLOGIES.iter().any(|t| t.topology == *expected_topo);
        assert!(found, "Missing topology: {:?}", expected_topo);
    }
}

// =============================================================================
// Category 3: Vertex Count Tests
// =============================================================================

#[test]
fn test_point_list_vertex_formula() {
    // PointList: vertices = primitives (1 vertex per point)
    for n in 0..100 {
        assert_eq!(
            topology_vertex_count(PrimitiveTopology::PointList, n),
            n,
            "PointList: {} primitives should need {} vertices",
            n,
            n
        );
    }
}

#[test]
fn test_line_list_vertex_formula() {
    // LineList: vertices = primitives * 2 (2 vertices per line)
    for n in 0..100 {
        assert_eq!(
            topology_vertex_count(PrimitiveTopology::LineList, n),
            n * 2,
            "LineList: {} primitives should need {} vertices",
            n,
            n * 2
        );
    }
}

#[test]
fn test_line_strip_vertex_formula() {
    // LineStrip: vertices = primitives + 1 (shared vertices)
    assert_eq!(topology_vertex_count(PrimitiveTopology::LineStrip, 0), 0);
    for n in 1..100 {
        assert_eq!(
            topology_vertex_count(PrimitiveTopology::LineStrip, n),
            n + 1,
            "LineStrip: {} primitives should need {} vertices",
            n,
            n + 1
        );
    }
}

#[test]
fn test_triangle_list_vertex_formula() {
    // TriangleList: vertices = triangles * 3
    for n in 0..100 {
        assert_eq!(
            topology_vertex_count(PrimitiveTopology::TriangleList, n),
            n * 3,
            "TriangleList: {} primitives should need {} vertices",
            n,
            n * 3
        );
    }
}

#[test]
fn test_triangle_strip_vertex_formula() {
    // TriangleStrip: vertices = triangles + 2 (for triangles >= 1)
    assert_eq!(topology_vertex_count(PrimitiveTopology::TriangleStrip, 0), 0);
    for n in 1..100 {
        assert_eq!(
            topology_vertex_count(PrimitiveTopology::TriangleStrip, n),
            n + 2,
            "TriangleStrip: {} primitives should need {} vertices",
            n,
            n + 2
        );
    }
}

// =============================================================================
// Category 4: Primitive Count Tests
// =============================================================================

#[test]
fn test_point_list_primitive_formula() {
    // PointList: primitives = vertices
    for v in 0..100 {
        assert_eq!(
            topology_primitive_count(PrimitiveTopology::PointList, v),
            v,
            "PointList: {} vertices should give {} primitives",
            v,
            v
        );
    }
}

#[test]
fn test_line_list_primitive_formula() {
    // LineList: primitives = vertices / 2 (integer division)
    for v in 0..100 {
        assert_eq!(
            topology_primitive_count(PrimitiveTopology::LineList, v),
            v / 2,
            "LineList: {} vertices should give {} primitives",
            v,
            v / 2
        );
    }
}

#[test]
fn test_line_strip_primitive_formula() {
    // LineStrip: primitives = vertices - 1 (for vertices >= 2)
    assert_eq!(topology_primitive_count(PrimitiveTopology::LineStrip, 0), 0);
    assert_eq!(topology_primitive_count(PrimitiveTopology::LineStrip, 1), 0);
    for v in 2..100 {
        assert_eq!(
            topology_primitive_count(PrimitiveTopology::LineStrip, v),
            v - 1,
            "LineStrip: {} vertices should give {} primitives",
            v,
            v - 1
        );
    }
}

#[test]
fn test_triangle_list_primitive_formula() {
    // TriangleList: primitives = vertices / 3 (integer division)
    for v in 0..100 {
        assert_eq!(
            topology_primitive_count(PrimitiveTopology::TriangleList, v),
            v / 3,
            "TriangleList: {} vertices should give {} primitives",
            v,
            v / 3
        );
    }
}

#[test]
fn test_triangle_strip_primitive_formula() {
    // TriangleStrip: triangles = vertices - 2 (for vertices >= 3)
    assert_eq!(
        topology_primitive_count(PrimitiveTopology::TriangleStrip, 0),
        0
    );
    assert_eq!(
        topology_primitive_count(PrimitiveTopology::TriangleStrip, 1),
        0
    );
    assert_eq!(
        topology_primitive_count(PrimitiveTopology::TriangleStrip, 2),
        0
    );
    for v in 3..100 {
        assert_eq!(
            topology_primitive_count(PrimitiveTopology::TriangleStrip, v),
            v - 2,
            "TriangleStrip: {} vertices should give {} primitives",
            v,
            v - 2
        );
    }
}

// =============================================================================
// Category 5: Classification Tests
// =============================================================================

#[test]
fn test_is_strip_point_list() {
    assert!(!is_strip_topology(PrimitiveTopology::PointList));
}

#[test]
fn test_is_strip_line_list() {
    assert!(!is_strip_topology(PrimitiveTopology::LineList));
}

#[test]
fn test_is_strip_line_strip() {
    assert!(is_strip_topology(PrimitiveTopology::LineStrip));
}

#[test]
fn test_is_strip_triangle_list() {
    assert!(!is_strip_topology(PrimitiveTopology::TriangleList));
}

#[test]
fn test_is_strip_triangle_strip() {
    assert!(is_strip_topology(PrimitiveTopology::TriangleStrip));
}

#[test]
fn test_is_list_point_list() {
    assert!(is_list_topology(PrimitiveTopology::PointList));
}

#[test]
fn test_is_list_line_list() {
    assert!(is_list_topology(PrimitiveTopology::LineList));
}

#[test]
fn test_is_list_line_strip() {
    assert!(!is_list_topology(PrimitiveTopology::LineStrip));
}

#[test]
fn test_is_list_triangle_list() {
    assert!(is_list_topology(PrimitiveTopology::TriangleList));
}

#[test]
fn test_is_list_triangle_strip() {
    assert!(!is_list_topology(PrimitiveTopology::TriangleStrip));
}

#[test]
fn test_list_and_strip_mutually_exclusive() {
    // Every topology should be either list or strip, not both or neither
    let all_topologies = [
        PrimitiveTopology::PointList,
        PrimitiveTopology::LineList,
        PrimitiveTopology::LineStrip,
        PrimitiveTopology::TriangleList,
        PrimitiveTopology::TriangleStrip,
    ];
    for topo in &all_topologies {
        let is_list = is_list_topology(*topo);
        let is_strip = is_strip_topology(*topo);
        assert!(
            is_list ^ is_strip,
            "{:?} should be exactly one of list or strip",
            topo
        );
    }
}

// =============================================================================
// Category 6: TopologyInfo Lookup Tests
// =============================================================================

#[test]
fn test_get_topology_info_point_list() {
    let info = get_topology_info(PrimitiveTopology::PointList);
    assert_eq!(info.topology, PrimitiveTopology::PointList);
    assert_eq!(info.name, "PointList");
}

#[test]
fn test_get_topology_info_line_list() {
    let info = get_topology_info(PrimitiveTopology::LineList);
    assert_eq!(info.topology, PrimitiveTopology::LineList);
    assert_eq!(info.name, "LineList");
}

#[test]
fn test_get_topology_info_line_strip() {
    let info = get_topology_info(PrimitiveTopology::LineStrip);
    assert_eq!(info.topology, PrimitiveTopology::LineStrip);
    assert_eq!(info.name, "LineStrip");
}

#[test]
fn test_get_topology_info_triangle_list() {
    let info = get_topology_info(PrimitiveTopology::TriangleList);
    assert_eq!(info.topology, PrimitiveTopology::TriangleList);
    assert_eq!(info.name, "TriangleList");
}

#[test]
fn test_get_topology_info_triangle_strip() {
    let info = get_topology_info(PrimitiveTopology::TriangleStrip);
    assert_eq!(info.topology, PrimitiveTopology::TriangleStrip);
    assert_eq!(info.name, "TriangleStrip");
}

// =============================================================================
// Category 7: Use Case Documentation Tests
// =============================================================================

#[test]
fn test_point_list_use_cases() {
    let info = get_topology_info(PrimitiveTopology::PointList);
    assert!(
        info.use_cases.contains(&"particle systems"),
        "PointList should document particle systems use case"
    );
}

#[test]
fn test_line_list_use_cases() {
    let info = get_topology_info(PrimitiveTopology::LineList);
    assert!(
        info.use_cases.contains(&"wireframe"),
        "LineList should document wireframe use case"
    );
}

#[test]
fn test_line_strip_use_cases() {
    let info = get_topology_info(PrimitiveTopology::LineStrip);
    assert!(
        info.use_cases.contains(&"path visualization"),
        "LineStrip should document path visualization use case"
    );
}

#[test]
fn test_triangle_list_use_cases() {
    let info = get_topology_info(PrimitiveTopology::TriangleList);
    assert!(
        info.use_cases.contains(&"general mesh rendering"),
        "TriangleList should document mesh rendering use case"
    );
}

#[test]
fn test_triangle_strip_use_cases() {
    let info = get_topology_info(PrimitiveTopology::TriangleStrip);
    assert!(
        info.use_cases.contains(&"terrain"),
        "TriangleStrip should document terrain use case"
    );
}

// =============================================================================
// Category 8: Real PrimitiveState Build Tests
// =============================================================================

#[test]
fn test_build_primitive_state_point_list() {
    let desc = PrimitiveStateDescriptor::point_list();
    let state: PrimitiveState = desc.into();
    assert_eq!(state.topology, PrimitiveTopology::PointList);
    assert_eq!(state.cull_mode, None); // Points don't need culling
}

#[test]
fn test_build_primitive_state_line_list() {
    let desc = PrimitiveStateDescriptor::line_list();
    let state: PrimitiveState = desc.into();
    assert_eq!(state.topology, PrimitiveTopology::LineList);
    assert_eq!(state.cull_mode, None); // Lines don't need culling
}

#[test]
fn test_build_primitive_state_line_strip() {
    let desc = PrimitiveStateDescriptor::line_strip(IndexFormat::Uint16);
    let state: PrimitiveState = desc.into();
    assert_eq!(state.topology, PrimitiveTopology::LineStrip);
    assert_eq!(state.strip_index_format, Some(IndexFormat::Uint16));
}

#[test]
fn test_build_primitive_state_triangle_list() {
    let desc = PrimitiveStateDescriptor::triangle_list();
    let state: PrimitiveState = desc.into();
    assert_eq!(state.topology, PrimitiveTopology::TriangleList);
    assert_eq!(state.cull_mode, Some(Face::Back)); // Default backface culling
}

#[test]
fn test_build_primitive_state_triangle_strip() {
    let desc = PrimitiveStateDescriptor::triangle_strip(IndexFormat::Uint32);
    let state: PrimitiveState = desc.into();
    assert_eq!(state.topology, PrimitiveTopology::TriangleStrip);
    assert_eq!(state.strip_index_format, Some(IndexFormat::Uint32));
}

#[test]
fn test_build_primitive_state_with_all_options() {
    let desc = PrimitiveStateDescriptor::new()
        .topology(PrimitiveTopology::TriangleList)
        .front_face(FrontFace::Cw)
        .cull_mode(Some(Face::Front))
        .polygon_mode(PolygonMode::Line)
        .unclipped_depth(true)
        .conservative(true);

    let state: PrimitiveState = desc.into();
    assert_eq!(state.topology, PrimitiveTopology::TriangleList);
    assert_eq!(state.front_face, FrontFace::Cw);
    assert_eq!(state.cull_mode, Some(Face::Front));
    assert_eq!(state.polygon_mode, PolygonMode::Line);
    assert!(state.unclipped_depth);
    assert!(state.conservative);
}

// =============================================================================
// Category 9: Edge Cases
// =============================================================================

#[test]
fn test_zero_primitives_all_topologies() {
    assert_eq!(topology_vertex_count(PrimitiveTopology::PointList, 0), 0);
    assert_eq!(topology_vertex_count(PrimitiveTopology::LineList, 0), 0);
    assert_eq!(topology_vertex_count(PrimitiveTopology::LineStrip, 0), 0);
    assert_eq!(topology_vertex_count(PrimitiveTopology::TriangleList, 0), 0);
    assert_eq!(
        topology_vertex_count(PrimitiveTopology::TriangleStrip, 0),
        0
    );
}

#[test]
fn test_zero_vertices_all_topologies() {
    assert_eq!(topology_primitive_count(PrimitiveTopology::PointList, 0), 0);
    assert_eq!(topology_primitive_count(PrimitiveTopology::LineList, 0), 0);
    assert_eq!(topology_primitive_count(PrimitiveTopology::LineStrip, 0), 0);
    assert_eq!(
        topology_primitive_count(PrimitiveTopology::TriangleList, 0),
        0
    );
    assert_eq!(
        topology_primitive_count(PrimitiveTopology::TriangleStrip, 0),
        0
    );
}

#[test]
fn test_single_primitive_all_topologies() {
    assert_eq!(topology_vertex_count(PrimitiveTopology::PointList, 1), 1);
    assert_eq!(topology_vertex_count(PrimitiveTopology::LineList, 1), 2);
    assert_eq!(topology_vertex_count(PrimitiveTopology::LineStrip, 1), 2);
    assert_eq!(topology_vertex_count(PrimitiveTopology::TriangleList, 1), 3);
    assert_eq!(
        topology_vertex_count(PrimitiveTopology::TriangleStrip, 1),
        3
    );
}

#[test]
fn test_minimum_vertex_count_all_topologies() {
    assert_eq!(minimum_vertex_count(PrimitiveTopology::PointList), 1);
    assert_eq!(minimum_vertex_count(PrimitiveTopology::LineList), 2);
    assert_eq!(minimum_vertex_count(PrimitiveTopology::LineStrip), 2);
    assert_eq!(minimum_vertex_count(PrimitiveTopology::TriangleList), 3);
    assert_eq!(minimum_vertex_count(PrimitiveTopology::TriangleStrip), 3);
}

#[test]
fn test_insufficient_vertices_line_list() {
    // 1 vertex is not enough for a line
    assert_eq!(topology_primitive_count(PrimitiveTopology::LineList, 1), 0);
}

#[test]
fn test_insufficient_vertices_triangle_list() {
    // 1 or 2 vertices are not enough for a triangle
    assert_eq!(
        topology_primitive_count(PrimitiveTopology::TriangleList, 1),
        0
    );
    assert_eq!(
        topology_primitive_count(PrimitiveTopology::TriangleList, 2),
        0
    );
}

#[test]
fn test_insufficient_vertices_line_strip() {
    assert_eq!(topology_primitive_count(PrimitiveTopology::LineStrip, 1), 0);
}

#[test]
fn test_insufficient_vertices_triangle_strip() {
    assert_eq!(
        topology_primitive_count(PrimitiveTopology::TriangleStrip, 1),
        0
    );
    assert_eq!(
        topology_primitive_count(PrimitiveTopology::TriangleStrip, 2),
        0
    );
}

#[test]
fn test_large_vertex_count_point_list() {
    let large = 1_000_000;
    assert_eq!(
        topology_vertex_count(PrimitiveTopology::PointList, large),
        large
    );
    assert_eq!(
        topology_primitive_count(PrimitiveTopology::PointList, large),
        large
    );
}

#[test]
fn test_large_vertex_count_triangle_list() {
    let large = 1_000_000u32;
    assert_eq!(
        topology_vertex_count(PrimitiveTopology::TriangleList, large),
        large * 3
    );
    assert_eq!(
        topology_primitive_count(PrimitiveTopology::TriangleList, large * 3),
        large
    );
}

#[test]
fn test_large_vertex_count_triangle_strip() {
    let large = 1_000_000u32;
    assert_eq!(
        topology_vertex_count(PrimitiveTopology::TriangleStrip, large),
        large + 2
    );
    assert_eq!(
        topology_primitive_count(PrimitiveTopology::TriangleStrip, large + 2),
        large
    );
}

#[test]
fn test_vertex_primitive_roundtrip_list_topologies() {
    // For list topologies, primitives -> vertices -> primitives should be identity
    for n in 0..50 {
        // PointList
        let v = topology_vertex_count(PrimitiveTopology::PointList, n);
        assert_eq!(topology_primitive_count(PrimitiveTopology::PointList, v), n);

        // LineList
        let v = topology_vertex_count(PrimitiveTopology::LineList, n);
        assert_eq!(topology_primitive_count(PrimitiveTopology::LineList, v), n);

        // TriangleList
        let v = topology_vertex_count(PrimitiveTopology::TriangleList, n);
        assert_eq!(
            topology_primitive_count(PrimitiveTopology::TriangleList, v),
            n
        );
    }
}

#[test]
fn test_vertex_primitive_roundtrip_strip_topologies() {
    // For strip topologies with n >= 1
    for n in 1..50 {
        // LineStrip
        let v = topology_vertex_count(PrimitiveTopology::LineStrip, n);
        assert_eq!(
            topology_primitive_count(PrimitiveTopology::LineStrip, v),
            n
        );

        // TriangleStrip
        let v = topology_vertex_count(PrimitiveTopology::TriangleStrip, n);
        assert_eq!(
            topology_primitive_count(PrimitiveTopology::TriangleStrip, v),
            n
        );
    }
}

#[test]
fn test_odd_vertex_counts_line_list() {
    // LineList with odd vertices should truncate
    assert_eq!(topology_primitive_count(PrimitiveTopology::LineList, 3), 1);
    assert_eq!(topology_primitive_count(PrimitiveTopology::LineList, 5), 2);
    assert_eq!(topology_primitive_count(PrimitiveTopology::LineList, 7), 3);
}

#[test]
fn test_non_divisible_vertex_counts_triangle_list() {
    // TriangleList with non-3-divisible vertices should truncate
    assert_eq!(
        topology_primitive_count(PrimitiveTopology::TriangleList, 4),
        1
    );
    assert_eq!(
        topology_primitive_count(PrimitiveTopology::TriangleList, 5),
        1
    );
    assert_eq!(
        topology_primitive_count(PrimitiveTopology::TriangleList, 7),
        2
    );
    assert_eq!(
        topology_primitive_count(PrimitiveTopology::TriangleList, 8),
        2
    );
}

// =============================================================================
// Category 10: Builder Pattern Tests
// =============================================================================

#[test]
fn test_builder_chaining() {
    let state = PrimitiveStateDescriptor::new()
        .topology(PrimitiveTopology::TriangleStrip)
        .strip_index_format(Some(IndexFormat::Uint16))
        .front_face(FrontFace::Cw)
        .cull_mode(Some(Face::Back))
        .unclipped_depth(false)
        .polygon_mode(PolygonMode::Fill)
        .conservative(false);

    assert_eq!(state.topology, PrimitiveTopology::TriangleStrip);
    assert_eq!(state.strip_index_format, Some(IndexFormat::Uint16));
    assert_eq!(state.front_face, FrontFace::Cw);
}

#[test]
fn test_wireframe_helper() {
    let state = PrimitiveStateDescriptor::new().wireframe();
    let wgpu_state: PrimitiveState = state.into();
    assert_eq!(wgpu_state.polygon_mode, PolygonMode::Line);
}

#[test]
fn test_point_helper() {
    let state = PrimitiveStateDescriptor::new().point();
    let wgpu_state: PrimitiveState = state.into();
    assert_eq!(wgpu_state.polygon_mode, PolygonMode::Point);
}

#[test]
fn test_no_culling_helper() {
    let state = PrimitiveStateDescriptor::new().no_culling();
    let wgpu_state: PrimitiveState = state.into();
    assert_eq!(wgpu_state.cull_mode, None);
}

#[test]
fn test_cull_front_helper() {
    let state = PrimitiveStateDescriptor::new().cull_front();
    let wgpu_state: PrimitiveState = state.into();
    assert_eq!(wgpu_state.cull_mode, Some(Face::Front));
}

#[test]
fn test_cull_back_helper() {
    let state = PrimitiveStateDescriptor::new().cull_back();
    let wgpu_state: PrimitiveState = state.into();
    assert_eq!(wgpu_state.cull_mode, Some(Face::Back));
}

// =============================================================================
// Category 11: Strip Index Format Tests
// =============================================================================

#[test]
fn test_strip_topologies_need_index_format() {
    // Strip topologies should have strip_index_format for primitive restart
    let line_strip = PrimitiveStateDescriptor::line_strip(IndexFormat::Uint16);
    assert!(line_strip.strip_index_format.is_some());

    let tri_strip = PrimitiveStateDescriptor::triangle_strip(IndexFormat::Uint32);
    assert!(tri_strip.strip_index_format.is_some());
}

#[test]
fn test_list_topologies_no_index_format() {
    // List topologies should not have strip_index_format
    let point_list = PrimitiveStateDescriptor::point_list();
    assert!(point_list.strip_index_format.is_none());

    let line_list = PrimitiveStateDescriptor::line_list();
    assert!(line_list.strip_index_format.is_none());

    let tri_list = PrimitiveStateDescriptor::triangle_list();
    assert!(tri_list.strip_index_format.is_none());
}

#[test]
fn test_uint16_index_format_for_strips() {
    let state = PrimitiveStateDescriptor::triangle_strip(IndexFormat::Uint16);
    assert_eq!(state.strip_index_format, Some(IndexFormat::Uint16));
}

#[test]
fn test_uint32_index_format_for_strips() {
    let state = PrimitiveStateDescriptor::triangle_strip(IndexFormat::Uint32);
    assert_eq!(state.strip_index_format, Some(IndexFormat::Uint32));
}

// =============================================================================
// Category 12: Trait Implementation Tests
// =============================================================================

#[test]
fn test_primitive_state_descriptor_copy() {
    let original = PrimitiveStateDescriptor::default();
    let copied = original;
    assert_eq!(original, copied);
}

#[test]
fn test_primitive_state_descriptor_clone() {
    let original = PrimitiveStateDescriptor::new().wireframe();
    let cloned = original.clone();
    assert_eq!(original, cloned);
}

#[test]
fn test_primitive_state_descriptor_partial_eq() {
    let a = PrimitiveStateDescriptor::default();
    let b = PrimitiveStateDescriptor::default();
    let c = PrimitiveStateDescriptor::new().wireframe();

    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn test_primitive_state_descriptor_debug() {
    let state = PrimitiveStateDescriptor::default();
    let debug_str = format!("{:?}", state);
    assert!(debug_str.contains("PrimitiveStateDescriptor"));
}

#[test]
fn test_topology_info_copy() {
    let info = TOPOLOGIES[0];
    let copied = info;
    assert_eq!(info, copied);
}

#[test]
fn test_topology_info_clone() {
    let info = &TOPOLOGIES[0];
    let cloned = info.clone();
    assert_eq!(*info, cloned);
}

#[test]
fn test_topology_info_partial_eq() {
    assert_eq!(TOPOLOGIES[0], TOPOLOGIES[0]);
    assert_ne!(TOPOLOGIES[0], TOPOLOGIES[1]);
}

#[test]
fn test_topology_info_debug() {
    let debug_str = format!("{:?}", TOPOLOGIES[0]);
    assert!(debug_str.contains("TopologyInfo"));
}

// =============================================================================
// BLACKBOX - T-WGPU-P3.3.2 Culling and Front Face Tests
// =============================================================================

use renderer_backend::render_pipeline::{
    get_cull_mode_info, get_front_face_info, CullModeInfo, FrontFaceInfo, CULL_MODES, FRONT_FACES,
};

// =============================================================================
// Category 1: API Surface Tests - FrontFaceInfo, CullModeInfo accessible
// =============================================================================

#[test]
fn test_front_faces_constant_accessible() {
    // FRONT_FACES constant should be publicly accessible
    let front_faces: &[FrontFaceInfo; 2] = &FRONT_FACES;
    assert_eq!(front_faces.len(), 2);
}

#[test]
fn test_cull_modes_constant_accessible() {
    // CULL_MODES constant should be publicly accessible
    let cull_modes: &[CullModeInfo; 3] = &CULL_MODES;
    assert_eq!(cull_modes.len(), 3);
}

#[test]
fn test_front_face_info_struct_fields() {
    // Verify FrontFaceInfo has all required fields via pattern matching
    let info = &FRONT_FACES[0];
    let FrontFaceInfo {
        front_face,
        name,
        description,
    } = info;
    assert!(!name.is_empty());
    assert!(!description.is_empty());
    let _ = front_face; // Use to avoid warning
}

#[test]
fn test_cull_mode_info_struct_fields() {
    // Verify CullModeInfo has all required fields via pattern matching
    let info = &CULL_MODES[0];
    let CullModeInfo {
        cull_mode,
        name,
        description,
        use_cases,
    } = info;
    assert!(!name.is_empty());
    assert!(!description.is_empty());
    assert!(!use_cases.is_empty());
    let _ = cull_mode; // Use to avoid warning
}

// =============================================================================
// Category 2: FrontFace Tests - Ccw and Cw configurations
// =============================================================================

#[test]
fn test_front_faces_constant_complete() {
    // Must contain exactly Ccw and Cw
    assert_eq!(FRONT_FACES.len(), 2);
    assert!(FRONT_FACES.iter().any(|f| f.front_face == FrontFace::Ccw));
    assert!(FRONT_FACES.iter().any(|f| f.front_face == FrontFace::Cw));
}

#[test]
fn test_front_face_ccw_info() {
    let info = get_front_face_info(FrontFace::Ccw);
    assert_eq!(info.front_face, FrontFace::Ccw);
    assert_eq!(info.name, "Counter-Clockwise");
    assert!(info.description.contains("counter-clockwise"));
}

#[test]
fn test_front_face_cw_info() {
    let info = get_front_face_info(FrontFace::Cw);
    assert_eq!(info.front_face, FrontFace::Cw);
    assert_eq!(info.name, "Clockwise");
    assert!(info.description.contains("clockwise"));
}

#[test]
fn test_front_face_ccw_is_default() {
    // CCW is the wgpu/OpenGL/Vulkan default
    let state = PrimitiveStateDescriptor::default();
    assert_eq!(state.front_face, FrontFace::Ccw);
}

#[test]
fn test_front_face_ccw_builder() {
    let state = PrimitiveStateDescriptor::new().ccw();
    assert_eq!(state.front_face, FrontFace::Ccw);
}

#[test]
fn test_front_face_cw_builder() {
    let state = PrimitiveStateDescriptor::new().cw();
    assert_eq!(state.front_face, FrontFace::Cw);
}

#[test]
fn test_front_face_method() {
    let state_ccw = PrimitiveStateDescriptor::new().front_face(FrontFace::Ccw);
    assert_eq!(state_ccw.front_face, FrontFace::Ccw);

    let state_cw = PrimitiveStateDescriptor::new().front_face(FrontFace::Cw);
    assert_eq!(state_cw.front_face, FrontFace::Cw);
}

#[test]
fn test_with_front_face_method() {
    let state_ccw = PrimitiveStateDescriptor::new().with_front_face(FrontFace::Ccw);
    assert_eq!(state_ccw.front_face, FrontFace::Ccw);

    let state_cw = PrimitiveStateDescriptor::new().with_front_face(FrontFace::Cw);
    assert_eq!(state_cw.front_face, FrontFace::Cw);
}

// =============================================================================
// Category 3: CullMode Tests - None, Front, Back configurations
// =============================================================================

#[test]
fn test_cull_modes_constant_complete() {
    // Must contain exactly None, Front, Back
    assert_eq!(CULL_MODES.len(), 3);
    assert!(CULL_MODES.iter().any(|c| c.cull_mode == None));
    assert!(CULL_MODES.iter().any(|c| c.cull_mode == Some(Face::Front)));
    assert!(CULL_MODES.iter().any(|c| c.cull_mode == Some(Face::Back)));
}

#[test]
fn test_cull_mode_none_info() {
    let info = get_cull_mode_info(None);
    assert_eq!(info.cull_mode, None);
    assert_eq!(info.name, "None");
    assert!(info.description.contains("No culling"));
}

#[test]
fn test_cull_mode_front_info() {
    let info = get_cull_mode_info(Some(Face::Front));
    assert_eq!(info.cull_mode, Some(Face::Front));
    assert_eq!(info.name, "Front");
    assert!(info.description.contains("Front faces culled"));
}

#[test]
fn test_cull_mode_back_info() {
    let info = get_cull_mode_info(Some(Face::Back));
    assert_eq!(info.cull_mode, Some(Face::Back));
    assert_eq!(info.name, "Back");
    assert!(info.description.contains("Back faces culled"));
}

#[test]
fn test_cull_mode_back_is_default() {
    // Back-face culling is the default for performance
    let state = PrimitiveStateDescriptor::default();
    assert_eq!(state.cull_mode, Some(Face::Back));
}

#[test]
fn test_cull_none_builder() {
    let state = PrimitiveStateDescriptor::new().cull_none();
    assert_eq!(state.cull_mode, None);
}

#[test]
fn test_cull_front_builder() {
    let state = PrimitiveStateDescriptor::new().cull_front();
    assert_eq!(state.cull_mode, Some(Face::Front));
}

#[test]
fn test_cull_back_builder() {
    let state = PrimitiveStateDescriptor::new().cull_back();
    assert_eq!(state.cull_mode, Some(Face::Back));
}

#[test]
fn test_no_culling_alias() {
    // no_culling() should be an alias for cull_none()
    let state1 = PrimitiveStateDescriptor::new().no_culling();
    let state2 = PrimitiveStateDescriptor::new().cull_none();
    assert_eq!(state1.cull_mode, state2.cull_mode);
    assert_eq!(state1.cull_mode, None);
}

#[test]
fn test_cull_mode_method() {
    let state_none = PrimitiveStateDescriptor::new().cull_mode(None);
    assert_eq!(state_none.cull_mode, None);

    let state_front = PrimitiveStateDescriptor::new().cull_mode(Some(Face::Front));
    assert_eq!(state_front.cull_mode, Some(Face::Front));

    let state_back = PrimitiveStateDescriptor::new().cull_mode(Some(Face::Back));
    assert_eq!(state_back.cull_mode, Some(Face::Back));
}

#[test]
fn test_with_cull_mode_method() {
    let state_none = PrimitiveStateDescriptor::new().with_cull_mode(None);
    assert_eq!(state_none.cull_mode, None);

    let state_front = PrimitiveStateDescriptor::new().with_cull_mode(Some(Face::Front));
    assert_eq!(state_front.cull_mode, Some(Face::Front));

    let state_back = PrimitiveStateDescriptor::new().with_cull_mode(Some(Face::Back));
    assert_eq!(state_back.cull_mode, Some(Face::Back));
}

// =============================================================================
// Category 4: Builder Chain Tests - Multiple culling options chained
// =============================================================================

#[test]
fn test_chain_front_face_methods() {
    // Later calls should override earlier ones
    let state = PrimitiveStateDescriptor::new().ccw().cw().ccw();
    assert_eq!(state.front_face, FrontFace::Ccw);
}

#[test]
fn test_chain_cull_mode_methods() {
    // Later calls should override earlier ones
    let state = PrimitiveStateDescriptor::new()
        .cull_back()
        .cull_front()
        .cull_none()
        .cull_back();
    assert_eq!(state.cull_mode, Some(Face::Back));
}

#[test]
fn test_chain_culling_with_topology() {
    let state = PrimitiveStateDescriptor::new()
        .topology(PrimitiveTopology::TriangleList)
        .cw()
        .cull_front();
    assert_eq!(state.topology, PrimitiveTopology::TriangleList);
    assert_eq!(state.front_face, FrontFace::Cw);
    assert_eq!(state.cull_mode, Some(Face::Front));
}

#[test]
fn test_chain_culling_with_polygon_mode() {
    let state = PrimitiveStateDescriptor::new()
        .wireframe()
        .cull_none()
        .ccw();
    assert_eq!(state.polygon_mode, PolygonMode::Line);
    assert_eq!(state.cull_mode, None);
    assert_eq!(state.front_face, FrontFace::Ccw);
}

#[test]
fn test_chain_full_culling_configuration() {
    let state = PrimitiveStateDescriptor::new()
        .topology(PrimitiveTopology::TriangleStrip)
        .strip_index_format(Some(IndexFormat::Uint16))
        .front_face(FrontFace::Cw)
        .cull_mode(Some(Face::Back))
        .polygon_mode(PolygonMode::Fill)
        .conservative(false)
        .unclipped_depth(false);

    assert_eq!(state.topology, PrimitiveTopology::TriangleStrip);
    assert_eq!(state.strip_index_format, Some(IndexFormat::Uint16));
    assert_eq!(state.front_face, FrontFace::Cw);
    assert_eq!(state.cull_mode, Some(Face::Back));
}

// =============================================================================
// Category 5: Real PrimitiveState Tests - Build wgpu::PrimitiveState with culling
// =============================================================================

#[test]
fn test_build_wgpu_state_ccw_cull_back() {
    // Standard OpenGL configuration
    let desc = PrimitiveStateDescriptor::new().ccw().cull_back();
    let state: PrimitiveState = desc.into();
    assert_eq!(state.front_face, FrontFace::Ccw);
    assert_eq!(state.cull_mode, Some(Face::Back));
}

#[test]
fn test_build_wgpu_state_cw_cull_back() {
    // DirectX-style configuration
    let desc = PrimitiveStateDescriptor::new().cw().cull_back();
    let state: PrimitiveState = desc.into();
    assert_eq!(state.front_face, FrontFace::Cw);
    assert_eq!(state.cull_mode, Some(Face::Back));
}

#[test]
fn test_build_wgpu_state_cw_cull_front() {
    // Interior rendering with DirectX winding
    let desc = PrimitiveStateDescriptor::new().cw().cull_front();
    let state: PrimitiveState = desc.into();
    assert_eq!(state.front_face, FrontFace::Cw);
    assert_eq!(state.cull_mode, Some(Face::Front));
}

#[test]
fn test_build_wgpu_state_no_culling() {
    // Two-sided material
    let desc = PrimitiveStateDescriptor::new().cull_none();
    let state: PrimitiveState = desc.into();
    assert_eq!(state.cull_mode, None);
}

#[test]
fn test_build_two_sided_material() {
    // Foliage, transparency, etc.
    let state = PrimitiveStateDescriptor::new().cull_none();
    let wgpu_state: PrimitiveState = state.into();
    assert_eq!(wgpu_state.cull_mode, None);
}

#[test]
fn test_build_interior_rendering() {
    // Shadow volumes, inside-out objects
    let state = PrimitiveStateDescriptor::new().cull_front();
    let wgpu_state: PrimitiveState = state.into();
    assert_eq!(wgpu_state.cull_mode, Some(Face::Front));
}

#[test]
fn test_build_standard_opaque_mesh() {
    // Default backface culling for performance
    let state = PrimitiveStateDescriptor::triangle_list();
    let wgpu_state: PrimitiveState = state.into();
    assert_eq!(wgpu_state.cull_mode, Some(Face::Back));
    assert_eq!(wgpu_state.topology, PrimitiveTopology::TriangleList);
}

// =============================================================================
// Category 6: Info Lookup Tests - get_front_face_info, get_cull_mode_info
// =============================================================================

#[test]
fn test_get_front_face_info_returns_static_ref() {
    let info: &'static FrontFaceInfo = get_front_face_info(FrontFace::Ccw);
    assert_eq!(info.front_face, FrontFace::Ccw);
}

#[test]
fn test_get_cull_mode_info_returns_static_ref() {
    let info: &'static CullModeInfo = get_cull_mode_info(None);
    assert_eq!(info.cull_mode, None);
}

#[test]
fn test_get_front_face_info_all_variants() {
    // Test all FrontFace variants return correct info
    let ccw = get_front_face_info(FrontFace::Ccw);
    let cw = get_front_face_info(FrontFace::Cw);

    assert_eq!(ccw.front_face, FrontFace::Ccw);
    assert_eq!(cw.front_face, FrontFace::Cw);
    assert_ne!(ccw.name, cw.name);
}

#[test]
fn test_get_cull_mode_info_all_variants() {
    // Test all cull mode variants return correct info
    let none = get_cull_mode_info(None);
    let front = get_cull_mode_info(Some(Face::Front));
    let back = get_cull_mode_info(Some(Face::Back));

    assert_eq!(none.cull_mode, None);
    assert_eq!(front.cull_mode, Some(Face::Front));
    assert_eq!(back.cull_mode, Some(Face::Back));
    assert_ne!(none.name, front.name);
    assert_ne!(front.name, back.name);
}

// =============================================================================
// Category 7: Constant Array Tests - FRONT_FACES, CULL_MODES contents
// =============================================================================

#[test]
fn test_front_faces_array_contents() {
    // Verify FRONT_FACES[0] is CCW (conventional ordering)
    assert_eq!(FRONT_FACES[0].front_face, FrontFace::Ccw);
    assert_eq!(FRONT_FACES[1].front_face, FrontFace::Cw);
}

#[test]
fn test_cull_modes_array_contents() {
    // Verify CULL_MODES ordering: None, Front, Back
    assert_eq!(CULL_MODES[0].cull_mode, None);
    assert_eq!(CULL_MODES[1].cull_mode, Some(Face::Front));
    assert_eq!(CULL_MODES[2].cull_mode, Some(Face::Back));
}

#[test]
fn test_front_faces_all_documented() {
    for info in &FRONT_FACES {
        assert!(!info.name.is_empty(), "Front face must have a name");
        assert!(
            !info.description.is_empty(),
            "Front face must have a description"
        );
    }
}

#[test]
fn test_cull_modes_all_documented() {
    for info in &CULL_MODES {
        assert!(!info.name.is_empty(), "Cull mode must have a name");
        assert!(
            !info.description.is_empty(),
            "Cull mode must have a description"
        );
    }
}

// =============================================================================
// Category 8: Use Cases Tests - Verify use_cases documented
// =============================================================================

#[test]
fn test_cull_modes_have_use_cases() {
    for mode in &CULL_MODES {
        assert!(
            !mode.use_cases.is_empty(),
            "{} should have use cases",
            mode.name
        );
    }
}

#[test]
fn test_cull_mode_none_use_cases() {
    let info = get_cull_mode_info(None);
    assert!(info.use_cases.contains(&"two-sided materials"));
    assert!(info.use_cases.contains(&"foliage"));
}

#[test]
fn test_cull_mode_front_use_cases() {
    let info = get_cull_mode_info(Some(Face::Front));
    assert!(info.use_cases.contains(&"interior rendering"));
    assert!(info.use_cases.contains(&"shadow volumes"));
}

#[test]
fn test_cull_mode_back_use_cases() {
    let info = get_cull_mode_info(Some(Face::Back));
    assert!(info.use_cases.contains(&"standard mesh rendering"));
    assert!(info.use_cases.contains(&"performance optimization"));
}

#[test]
fn test_cull_modes_minimum_use_cases() {
    // Each cull mode should have at least 2 use cases
    for mode in &CULL_MODES {
        assert!(
            mode.use_cases.len() >= 2,
            "{} should have at least 2 use cases, found {}",
            mode.name,
            mode.use_cases.len()
        );
    }
}

// =============================================================================
// Category 9: Default Values Tests - Default is Ccw + cull Back
// =============================================================================

#[test]
fn test_default_front_face_is_ccw() {
    let state = PrimitiveStateDescriptor::default();
    assert_eq!(state.front_face, FrontFace::Ccw);
}

#[test]
fn test_default_cull_mode_is_back() {
    let state = PrimitiveStateDescriptor::default();
    assert_eq!(state.cull_mode, Some(Face::Back));
}

#[test]
fn test_new_has_same_defaults_as_default() {
    let state1 = PrimitiveStateDescriptor::new();
    let state2 = PrimitiveStateDescriptor::default();
    assert_eq!(state1.front_face, state2.front_face);
    assert_eq!(state1.cull_mode, state2.cull_mode);
}

#[test]
fn test_triangle_list_preset_defaults() {
    let state = PrimitiveStateDescriptor::triangle_list();
    assert_eq!(state.front_face, FrontFace::Ccw);
    assert_eq!(state.cull_mode, Some(Face::Back));
}

#[test]
fn test_wgpu_conversion_preserves_defaults() {
    let desc = PrimitiveStateDescriptor::default();
    let wgpu_state: PrimitiveState = desc.into();
    assert_eq!(wgpu_state.front_face, FrontFace::Ccw);
    assert_eq!(wgpu_state.cull_mode, Some(Face::Back));
}

// =============================================================================
// Category 10: Trait Implementation Tests for Culling Types
// =============================================================================

#[test]
fn test_front_face_info_copy() {
    let info = FRONT_FACES[0];
    let copied = info;
    assert_eq!(info, copied);
}

#[test]
fn test_front_face_info_clone() {
    let info = &FRONT_FACES[0];
    let cloned = info.clone();
    assert_eq!(*info, cloned);
}

#[test]
fn test_front_face_info_partial_eq() {
    assert_eq!(FRONT_FACES[0], FRONT_FACES[0]);
    assert_ne!(FRONT_FACES[0], FRONT_FACES[1]);
}

#[test]
fn test_front_face_info_debug() {
    let debug_str = format!("{:?}", FRONT_FACES[0]);
    assert!(debug_str.contains("FrontFaceInfo"));
    assert!(debug_str.contains("Ccw"));
}

#[test]
fn test_cull_mode_info_copy() {
    let info = CULL_MODES[0];
    let copied = info;
    assert_eq!(info, copied);
}

#[test]
fn test_cull_mode_info_clone() {
    let info = &CULL_MODES[0];
    let cloned = info.clone();
    assert_eq!(*info, cloned);
}

#[test]
fn test_cull_mode_info_partial_eq() {
    assert_eq!(CULL_MODES[0], CULL_MODES[0]);
    assert_ne!(CULL_MODES[0], CULL_MODES[1]);
    assert_ne!(CULL_MODES[1], CULL_MODES[2]);
}

#[test]
fn test_cull_mode_info_debug() {
    let debug_str = format!("{:?}", CULL_MODES[0]);
    assert!(debug_str.contains("CullModeInfo"));
    assert!(debug_str.contains("None"));
}

// =============================================================================
// Category 11: Presets with Culling Tests
// =============================================================================

#[test]
fn test_point_list_preset_no_culling() {
    // Points don't benefit from culling
    let state = PrimitiveStateDescriptor::point_list();
    assert_eq!(state.cull_mode, None);
}

#[test]
fn test_line_list_preset_no_culling() {
    // Lines don't benefit from culling
    let state = PrimitiveStateDescriptor::line_list();
    assert_eq!(state.cull_mode, None);
}

#[test]
fn test_line_strip_preset_no_culling() {
    // Line strips don't benefit from culling
    let state = PrimitiveStateDescriptor::line_strip(IndexFormat::Uint16);
    assert_eq!(state.cull_mode, None);
}

#[test]
fn test_triangle_list_preset_cull_back() {
    // Triangle lists benefit from backface culling
    let state = PrimitiveStateDescriptor::triangle_list();
    assert_eq!(state.cull_mode, Some(Face::Back));
}

#[test]
fn test_triangle_strip_preset_cull_back() {
    // Triangle strips benefit from backface culling
    let state = PrimitiveStateDescriptor::triangle_strip(IndexFormat::Uint16);
    assert_eq!(state.cull_mode, Some(Face::Back));
}

// =============================================================================
// Category 12: Real-World Configuration Tests
// =============================================================================

#[test]
fn test_opengl_style_configuration() {
    // OpenGL/Vulkan/wgpu default: CCW front, cull back
    let state = PrimitiveStateDescriptor::new().ccw().cull_back();
    let wgpu_state: PrimitiveState = state.into();
    assert_eq!(wgpu_state.front_face, FrontFace::Ccw);
    assert_eq!(wgpu_state.cull_mode, Some(Face::Back));
}

#[test]
fn test_directx_style_configuration() {
    // DirectX style: CW front, cull back
    let state = PrimitiveStateDescriptor::new().cw().cull_back();
    let wgpu_state: PrimitiveState = state.into();
    assert_eq!(wgpu_state.front_face, FrontFace::Cw);
    assert_eq!(wgpu_state.cull_mode, Some(Face::Back));
}

#[test]
fn test_gltf_configuration() {
    // glTF specifies CCW front faces
    let state = PrimitiveStateDescriptor::new().ccw().cull_back();
    let wgpu_state: PrimitiveState = state.into();
    assert_eq!(wgpu_state.front_face, FrontFace::Ccw);
}

#[test]
fn test_foliage_configuration() {
    // Foliage: no culling, default winding
    let state = PrimitiveStateDescriptor::new().cull_none();
    let wgpu_state: PrimitiveState = state.into();
    assert_eq!(wgpu_state.cull_mode, None);
    assert_eq!(wgpu_state.front_face, FrontFace::Ccw); // Default
}

#[test]
fn test_transparent_material_configuration() {
    // Transparent materials often need two-sided rendering
    let state = PrimitiveStateDescriptor::new().no_culling();
    let wgpu_state: PrimitiveState = state.into();
    assert_eq!(wgpu_state.cull_mode, None);
}

#[test]
fn test_shadow_volume_configuration() {
    // Shadow volumes: cull front to render back faces
    let state = PrimitiveStateDescriptor::new().cull_front();
    let wgpu_state: PrimitiveState = state.into();
    assert_eq!(wgpu_state.cull_mode, Some(Face::Front));
}

#[test]
fn test_debug_wireframe_configuration() {
    // Debug wireframe: no culling, wireframe mode
    let state = PrimitiveStateDescriptor::new().wireframe().cull_none();
    let wgpu_state: PrimitiveState = state.into();
    assert_eq!(wgpu_state.polygon_mode, PolygonMode::Line);
    assert_eq!(wgpu_state.cull_mode, None);
}

// =============================================================================
// BLACKBOX - T-WGPU-P3.3.3 Polygon Mode Tests
// =============================================================================

use renderer_backend::render_pipeline::{
    get_polygon_mode_info, required_feature_for_polygon_mode, requires_non_fill_feature,
    PolygonModeInfo, POLYGON_MODES,
};

// =============================================================================
// Category 1: API Surface Tests - PolygonModeInfo, POLYGON_MODES accessible
// =============================================================================

#[test]
fn test_polygon_modes_constant_accessible() {
    // POLYGON_MODES constant should be publicly accessible
    let polygon_modes: &[PolygonModeInfo; 3] = &POLYGON_MODES;
    assert_eq!(polygon_modes.len(), 3);
}

#[test]
fn test_polygon_mode_info_struct_fields() {
    // Verify PolygonModeInfo has all required fields via pattern matching
    let info = &POLYGON_MODES[0];
    let PolygonModeInfo {
        mode,
        name,
        description,
        use_cases,
        requires_feature,
    } = info;
    assert!(!name.is_empty());
    assert!(!description.is_empty());
    assert!(!use_cases.is_empty());
    let _ = mode; // Use to avoid warning
    let _ = requires_feature; // Use to avoid warning
}

#[test]
fn test_polygon_modes_constant_complete() {
    // Must contain exactly Fill, Line, Point
    assert_eq!(POLYGON_MODES.len(), 3);
    assert!(POLYGON_MODES.iter().any(|m| m.mode == PolygonMode::Fill));
    assert!(POLYGON_MODES.iter().any(|m| m.mode == PolygonMode::Line));
    assert!(POLYGON_MODES.iter().any(|m| m.mode == PolygonMode::Point));
}

// =============================================================================
// Category 2: All Mode Tests - Fill, Line, Point configurations
// =============================================================================

#[test]
fn test_polygon_mode_fill_info() {
    let info = get_polygon_mode_info(PolygonMode::Fill);
    assert_eq!(info.mode, PolygonMode::Fill);
    assert_eq!(info.name, "Fill");
    assert!(info.description.contains("Fill") || info.description.to_lowercase().contains("default"));
}

#[test]
fn test_polygon_mode_line_info() {
    let info = get_polygon_mode_info(PolygonMode::Line);
    assert_eq!(info.mode, PolygonMode::Line);
    assert_eq!(info.name, "Line");
    assert!(
        info.description.contains("Wireframe") || info.description.contains("edge"),
        "Line mode description should mention wireframe or edges"
    );
}

#[test]
fn test_polygon_mode_point_info() {
    let info = get_polygon_mode_info(PolygonMode::Point);
    assert_eq!(info.mode, PolygonMode::Point);
    assert_eq!(info.name, "Point");
    assert!(
        info.description.contains("Point") || info.description.contains("vert"),
        "Point mode description should mention point or vertices"
    );
}

#[test]
fn test_polygon_mode_fill_is_default() {
    // Fill is the wgpu default
    let state = PrimitiveStateDescriptor::default();
    assert_eq!(state.polygon_mode, PolygonMode::Fill);
}

// =============================================================================
// Category 3: Builder Tests - polygon_fill(), polygon_line(), polygon_point(), wireframe()
// =============================================================================

#[test]
fn test_polygon_fill_builder() {
    let state = PrimitiveStateDescriptor::new().polygon_fill();
    assert_eq!(state.polygon_mode, PolygonMode::Fill);
}

#[test]
fn test_polygon_line_builder() {
    let state = PrimitiveStateDescriptor::new().polygon_line();
    assert_eq!(state.polygon_mode, PolygonMode::Line);
}

#[test]
fn test_polygon_point_builder() {
    let state = PrimitiveStateDescriptor::new().polygon_point();
    assert_eq!(state.polygon_mode, PolygonMode::Point);
}

#[test]
fn test_wireframe_builder() {
    let state = PrimitiveStateDescriptor::new().wireframe();
    assert_eq!(state.polygon_mode, PolygonMode::Line);
}

#[test]
fn test_point_builder() {
    let state = PrimitiveStateDescriptor::new().point();
    assert_eq!(state.polygon_mode, PolygonMode::Point);
}

#[test]
fn test_wireframe_equals_polygon_line() {
    // wireframe() should be equivalent to polygon_line()
    let wireframe_state = PrimitiveStateDescriptor::new().wireframe();
    let polygon_line_state = PrimitiveStateDescriptor::new().polygon_line();
    assert_eq!(wireframe_state.polygon_mode, polygon_line_state.polygon_mode);
}

#[test]
fn test_point_equals_polygon_point() {
    // point() should be equivalent to polygon_point()
    let point_state = PrimitiveStateDescriptor::new().point();
    let polygon_point_state = PrimitiveStateDescriptor::new().polygon_point();
    assert_eq!(point_state.polygon_mode, polygon_point_state.polygon_mode);
}

#[test]
fn test_polygon_mode_method() {
    // Test the generic polygon_mode() method
    let fill = PrimitiveStateDescriptor::new().polygon_mode(PolygonMode::Fill);
    assert_eq!(fill.polygon_mode, PolygonMode::Fill);

    let line = PrimitiveStateDescriptor::new().polygon_mode(PolygonMode::Line);
    assert_eq!(line.polygon_mode, PolygonMode::Line);

    let point = PrimitiveStateDescriptor::new().polygon_mode(PolygonMode::Point);
    assert_eq!(point.polygon_mode, PolygonMode::Point);
}

// =============================================================================
// Category 4: Feature Flag Tests - requires_non_fill_feature(), required_feature_for_polygon_mode()
// =============================================================================

#[test]
fn test_fill_requires_no_feature() {
    assert!(!requires_non_fill_feature(PolygonMode::Fill));
    assert_eq!(required_feature_for_polygon_mode(PolygonMode::Fill), None);
}

#[test]
fn test_line_requires_feature() {
    assert!(requires_non_fill_feature(PolygonMode::Line));
    assert_eq!(
        required_feature_for_polygon_mode(PolygonMode::Line),
        Some(wgpu::Features::POLYGON_MODE_LINE)
    );
}

#[test]
fn test_point_requires_feature() {
    assert!(requires_non_fill_feature(PolygonMode::Point));
    assert_eq!(
        required_feature_for_polygon_mode(PolygonMode::Point),
        Some(wgpu::Features::POLYGON_MODE_POINT)
    );
}

#[test]
fn test_feature_requirement_documentation() {
    // Verify POLYGON_MODES.requires_feature matches function result
    for mode_info in &POLYGON_MODES {
        if mode_info.requires_feature {
            assert!(
                matches!(mode_info.mode, PolygonMode::Line | PolygonMode::Point),
                "{} requires_feature=true but is not Line or Point",
                mode_info.name
            );
        } else {
            assert_eq!(
                mode_info.mode,
                PolygonMode::Fill,
                "Only Fill should have requires_feature=false"
            );
        }
    }
}

#[test]
fn test_requires_non_fill_feature_matches_info() {
    // Cross-validate requires_non_fill_feature() with POLYGON_MODES
    for mode_info in &POLYGON_MODES {
        assert_eq!(
            requires_non_fill_feature(mode_info.mode),
            mode_info.requires_feature,
            "Mismatch for {}",
            mode_info.name
        );
    }
}

#[test]
fn test_required_feature_consistency() {
    // If requires_non_fill_feature is true, required_feature_for_polygon_mode should return Some
    for mode in [PolygonMode::Fill, PolygonMode::Line, PolygonMode::Point] {
        let requires = requires_non_fill_feature(mode);
        let feature = required_feature_for_polygon_mode(mode);
        if requires {
            assert!(
                feature.is_some(),
                "Mode {:?} requires feature but required_feature_for_polygon_mode returned None",
                mode
            );
        } else {
            assert!(
                feature.is_none(),
                "Mode {:?} does not require feature but required_feature_for_polygon_mode returned Some",
                mode
            );
        }
    }
}

// =============================================================================
// Category 5: Info Lookup Tests - get_polygon_mode_info() for each mode
// =============================================================================

#[test]
fn test_get_polygon_mode_info_returns_static_ref() {
    let info: &'static PolygonModeInfo = get_polygon_mode_info(PolygonMode::Fill);
    assert_eq!(info.mode, PolygonMode::Fill);
}

#[test]
fn test_get_polygon_mode_info_all_modes() {
    // Test all PolygonMode variants return correct info
    let fill = get_polygon_mode_info(PolygonMode::Fill);
    let line = get_polygon_mode_info(PolygonMode::Line);
    let point = get_polygon_mode_info(PolygonMode::Point);

    assert_eq!(fill.mode, PolygonMode::Fill);
    assert_eq!(line.mode, PolygonMode::Line);
    assert_eq!(point.mode, PolygonMode::Point);
    assert_ne!(fill.name, line.name);
    assert_ne!(line.name, point.name);
}

#[test]
fn test_get_polygon_mode_info_consistent_with_constant() {
    // get_polygon_mode_info should return values from POLYGON_MODES
    assert_eq!(get_polygon_mode_info(PolygonMode::Fill), &POLYGON_MODES[0]);
    assert_eq!(get_polygon_mode_info(PolygonMode::Line), &POLYGON_MODES[1]);
    assert_eq!(get_polygon_mode_info(PolygonMode::Point), &POLYGON_MODES[2]);
}

// =============================================================================
// Category 6: Use Cases Tests - Verify use_cases documented
// =============================================================================

#[test]
fn test_polygon_modes_have_use_cases() {
    for mode in &POLYGON_MODES {
        assert!(
            !mode.use_cases.is_empty(),
            "{} should have use cases",
            mode.name
        );
    }
}

#[test]
fn test_polygon_mode_fill_use_cases() {
    let info = get_polygon_mode_info(PolygonMode::Fill);
    assert!(info.use_cases.contains(&"standard rendering"));
}

#[test]
fn test_polygon_mode_line_use_cases() {
    let info = get_polygon_mode_info(PolygonMode::Line);
    assert!(info.use_cases.contains(&"debug visualization"));
}

#[test]
fn test_polygon_mode_point_use_cases() {
    let info = get_polygon_mode_info(PolygonMode::Point);
    assert!(info.use_cases.contains(&"vertex visualization"));
}

#[test]
fn test_polygon_modes_minimum_use_cases() {
    // Each polygon mode should have at least 2 use cases
    for mode in &POLYGON_MODES {
        assert!(
            mode.use_cases.len() >= 2,
            "{} should have at least 2 use cases, found {}",
            mode.name,
            mode.use_cases.len()
        );
    }
}

// =============================================================================
// Category 7: Real PrimitiveState Tests - Build wgpu::PrimitiveState with each mode
// =============================================================================

#[test]
fn test_build_wgpu_state_fill() {
    let desc = PrimitiveStateDescriptor::new().polygon_fill();
    let state: PrimitiveState = desc.into();
    assert_eq!(state.polygon_mode, PolygonMode::Fill);
}

#[test]
fn test_build_wgpu_state_line() {
    let desc = PrimitiveStateDescriptor::new().polygon_line();
    let state: PrimitiveState = desc.into();
    assert_eq!(state.polygon_mode, PolygonMode::Line);
}

#[test]
fn test_build_wgpu_state_point() {
    let desc = PrimitiveStateDescriptor::new().polygon_point();
    let state: PrimitiveState = desc.into();
    assert_eq!(state.polygon_mode, PolygonMode::Point);
}

#[test]
fn test_build_wgpu_state_wireframe() {
    let desc = PrimitiveStateDescriptor::new().wireframe();
    let state: PrimitiveState = desc.into();
    assert_eq!(state.polygon_mode, PolygonMode::Line);
}

#[test]
fn test_build_all_modes_into_wgpu() {
    // Verify all polygon modes convert correctly to wgpu::PrimitiveState
    for mode in [PolygonMode::Fill, PolygonMode::Line, PolygonMode::Point] {
        let state = PrimitiveStateDescriptor::new().polygon_mode(mode);
        let wgpu_state: PrimitiveState = state.into();
        assert_eq!(wgpu_state.polygon_mode, mode);
    }
}

// =============================================================================
// Category 8: Combination Tests - polygon mode + topology + culling
// =============================================================================

#[test]
fn test_polygon_mode_with_topology_triangle_list() {
    let state = PrimitiveStateDescriptor::new()
        .topology(PrimitiveTopology::TriangleList)
        .polygon_line();
    let wgpu_state: PrimitiveState = state.into();
    assert_eq!(wgpu_state.topology, PrimitiveTopology::TriangleList);
    assert_eq!(wgpu_state.polygon_mode, PolygonMode::Line);
}

#[test]
fn test_polygon_mode_with_topology_triangle_strip() {
    let state = PrimitiveStateDescriptor::new()
        .topology(PrimitiveTopology::TriangleStrip)
        .strip_index_format(Some(IndexFormat::Uint16))
        .polygon_point();
    let wgpu_state: PrimitiveState = state.into();
    assert_eq!(wgpu_state.topology, PrimitiveTopology::TriangleStrip);
    assert_eq!(wgpu_state.polygon_mode, PolygonMode::Point);
    assert_eq!(wgpu_state.strip_index_format, Some(IndexFormat::Uint16));
}

#[test]
fn test_polygon_mode_with_culling() {
    // Wireframe with no culling (common debug configuration)
    let state = PrimitiveStateDescriptor::new().wireframe().no_culling();
    let wgpu_state: PrimitiveState = state.into();
    assert_eq!(wgpu_state.polygon_mode, PolygonMode::Line);
    assert_eq!(wgpu_state.cull_mode, None);
}

#[test]
fn test_polygon_mode_with_front_face() {
    let state = PrimitiveStateDescriptor::new()
        .polygon_line()
        .front_face(FrontFace::Cw);
    let wgpu_state: PrimitiveState = state.into();
    assert_eq!(wgpu_state.polygon_mode, PolygonMode::Line);
    assert_eq!(wgpu_state.front_face, FrontFace::Cw);
}

#[test]
fn test_full_configuration_with_polygon_mode() {
    let state = PrimitiveStateDescriptor::new()
        .topology(PrimitiveTopology::TriangleList)
        .front_face(FrontFace::Ccw)
        .cull_mode(Some(Face::Back))
        .polygon_mode(PolygonMode::Fill)
        .unclipped_depth(false)
        .conservative(false);
    let wgpu_state: PrimitiveState = state.into();
    assert_eq!(wgpu_state.topology, PrimitiveTopology::TriangleList);
    assert_eq!(wgpu_state.front_face, FrontFace::Ccw);
    assert_eq!(wgpu_state.cull_mode, Some(Face::Back));
    assert_eq!(wgpu_state.polygon_mode, PolygonMode::Fill);
    assert!(!wgpu_state.unclipped_depth);
    assert!(!wgpu_state.conservative);
}

#[test]
fn test_wireframe_debug_configuration() {
    // Common debug configuration: wireframe, no culling, CCW
    let state = PrimitiveStateDescriptor::new()
        .wireframe()
        .cull_none()
        .ccw();
    let wgpu_state: PrimitiveState = state.into();
    assert_eq!(wgpu_state.polygon_mode, PolygonMode::Line);
    assert_eq!(wgpu_state.cull_mode, None);
    assert_eq!(wgpu_state.front_face, FrontFace::Ccw);
}

#[test]
fn test_point_cloud_configuration() {
    // Point cloud: point topology + point polygon mode
    let state = PrimitiveStateDescriptor::point_list().point();
    let wgpu_state: PrimitiveState = state.into();
    assert_eq!(wgpu_state.topology, PrimitiveTopology::PointList);
    assert_eq!(wgpu_state.polygon_mode, PolygonMode::Point);
}

// =============================================================================
// Category 9: Builder Chain Override Tests
// =============================================================================

#[test]
fn test_polygon_mode_chain_override() {
    // Later calls should override earlier ones
    let state = PrimitiveStateDescriptor::new()
        .polygon_fill()
        .polygon_line()
        .polygon_point();
    assert_eq!(state.polygon_mode, PolygonMode::Point);
}

#[test]
fn test_wireframe_then_fill() {
    let state = PrimitiveStateDescriptor::new().wireframe().polygon_fill();
    assert_eq!(state.polygon_mode, PolygonMode::Fill);
}

#[test]
fn test_point_then_wireframe() {
    let state = PrimitiveStateDescriptor::new().point().wireframe();
    assert_eq!(state.polygon_mode, PolygonMode::Line);
}

// =============================================================================
// Category 10: Constant Array Contents Tests
// =============================================================================

#[test]
fn test_polygon_modes_array_order() {
    // Verify POLYGON_MODES ordering: Fill, Line, Point
    assert_eq!(POLYGON_MODES[0].mode, PolygonMode::Fill);
    assert_eq!(POLYGON_MODES[1].mode, PolygonMode::Line);
    assert_eq!(POLYGON_MODES[2].mode, PolygonMode::Point);
}

#[test]
fn test_polygon_modes_names() {
    assert_eq!(POLYGON_MODES[0].name, "Fill");
    assert_eq!(POLYGON_MODES[1].name, "Line");
    assert_eq!(POLYGON_MODES[2].name, "Point");
}

#[test]
fn test_polygon_modes_all_documented() {
    for mode in &POLYGON_MODES {
        assert!(!mode.name.is_empty(), "Polygon mode must have a name");
        assert!(
            !mode.description.is_empty(),
            "Polygon mode must have a description"
        );
    }
}

#[test]
fn test_polygon_modes_descriptions_non_empty() {
    for mode in &POLYGON_MODES {
        assert!(
            mode.description.len() >= 10,
            "{} description too short: '{}'",
            mode.name,
            mode.description
        );
    }
}

// =============================================================================
// Category 11: Trait Implementation Tests for Polygon Mode Types
// =============================================================================

#[test]
fn test_polygon_mode_info_copy() {
    let info = POLYGON_MODES[0];
    let copied = info;
    assert_eq!(info, copied);
}

#[test]
fn test_polygon_mode_info_clone() {
    let info = &POLYGON_MODES[0];
    let cloned = info.clone();
    assert_eq!(*info, cloned);
}

#[test]
fn test_polygon_mode_info_partial_eq() {
    assert_eq!(POLYGON_MODES[0], POLYGON_MODES[0]);
    assert_ne!(POLYGON_MODES[0], POLYGON_MODES[1]);
    assert_ne!(POLYGON_MODES[1], POLYGON_MODES[2]);
}

#[test]
fn test_polygon_mode_info_debug() {
    let debug_str = format!("{:?}", POLYGON_MODES[0]);
    assert!(debug_str.contains("PolygonModeInfo"));
    assert!(debug_str.contains("Fill"));
}

// =============================================================================
// Category 12: Default Values Tests
// =============================================================================

#[test]
fn test_default_polygon_mode_is_fill() {
    let state = PrimitiveStateDescriptor::default();
    assert_eq!(state.polygon_mode, PolygonMode::Fill);
}

#[test]
fn test_new_has_fill_polygon_mode() {
    let state = PrimitiveStateDescriptor::new();
    assert_eq!(state.polygon_mode, PolygonMode::Fill);
}

#[test]
fn test_presets_have_fill_polygon_mode() {
    // All presets should default to Fill polygon mode
    let tri_list = PrimitiveStateDescriptor::triangle_list();
    assert_eq!(tri_list.polygon_mode, PolygonMode::Fill);

    let tri_strip = PrimitiveStateDescriptor::triangle_strip(IndexFormat::Uint16);
    assert_eq!(tri_strip.polygon_mode, PolygonMode::Fill);

    let line_list = PrimitiveStateDescriptor::line_list();
    assert_eq!(line_list.polygon_mode, PolygonMode::Fill);

    let line_strip = PrimitiveStateDescriptor::line_strip(IndexFormat::Uint16);
    assert_eq!(line_strip.polygon_mode, PolygonMode::Fill);

    let point_list = PrimitiveStateDescriptor::point_list();
    assert_eq!(point_list.polygon_mode, PolygonMode::Fill);
}

#[test]
fn test_wgpu_conversion_preserves_polygon_mode() {
    for mode in [PolygonMode::Fill, PolygonMode::Line, PolygonMode::Point] {
        let desc = PrimitiveStateDescriptor::new().polygon_mode(mode);
        let wgpu_state: PrimitiveState = desc.into();
        assert_eq!(wgpu_state.polygon_mode, mode);
    }
}

// =============================================================================
// Category 13: Real-World Configuration Tests
// =============================================================================

#[test]
fn test_standard_opaque_rendering() {
    // Standard opaque: fill mode, backface culling
    let state = PrimitiveStateDescriptor::new().polygon_fill().cull_back();
    let wgpu_state: PrimitiveState = state.into();
    assert_eq!(wgpu_state.polygon_mode, PolygonMode::Fill);
    assert_eq!(wgpu_state.cull_mode, Some(Face::Back));
}

#[test]
fn test_mesh_inspection_wireframe() {
    // Mesh inspection: wireframe, no culling
    let state = PrimitiveStateDescriptor::new().wireframe().no_culling();
    let wgpu_state: PrimitiveState = state.into();
    assert_eq!(wgpu_state.polygon_mode, PolygonMode::Line);
    assert_eq!(wgpu_state.cull_mode, None);
}

#[test]
fn test_vertex_debug_visualization() {
    // Vertex debug: point mode
    let state = PrimitiveStateDescriptor::new().polygon_point();
    let wgpu_state: PrimitiveState = state.into();
    assert_eq!(wgpu_state.polygon_mode, PolygonMode::Point);
}

#[test]
fn test_cad_style_display() {
    // CAD-style: wireframe with specific winding
    let state = PrimitiveStateDescriptor::new().polygon_line().cw();
    let wgpu_state: PrimitiveState = state.into();
    assert_eq!(wgpu_state.polygon_mode, PolygonMode::Line);
    assert_eq!(wgpu_state.front_face, FrontFace::Cw);
}

#[test]
fn test_sparse_geometry_point_rendering() {
    // Sparse geometry: point mode for sparse point cloud
    let state = PrimitiveStateDescriptor::point_list().polygon_point();
    let wgpu_state: PrimitiveState = state.into();
    assert_eq!(wgpu_state.topology, PrimitiveTopology::PointList);
    assert_eq!(wgpu_state.polygon_mode, PolygonMode::Point);
}
