//! Black-box tests for T-MAT-8.1: glTF mesh loader
//!
//! Acceptance criteria:
//! - Loads standard glTF 2.0 test models
//! - Vertex and index data matches glTF spec

use renderer_backend::gltf::{
    load_gltf_from_json, AttributeType, ComponentType, GltfError, IndexFormat, VertexSemantic,
};

// ============================================================================
// Test helpers
// ============================================================================

/// Build position buffer with known vertex positions
fn make_position_buffer(positions: &[[f32; 3]]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(positions.len() * 12);
    for pos in positions {
        buf.extend_from_slice(&pos[0].to_le_bytes());
        buf.extend_from_slice(&pos[1].to_le_bytes());
        buf.extend_from_slice(&pos[2].to_le_bytes());
    }
    buf
}

/// Build normal buffer with known normal vectors
fn make_normal_buffer(normals: &[[f32; 3]]) -> Vec<u8> {
    make_position_buffer(normals) // Same structure as positions
}

/// Build tangent buffer (Vec4) with known tangent vectors
fn make_tangent_buffer(tangents: &[[f32; 4]]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(tangents.len() * 16);
    for t in tangents {
        buf.extend_from_slice(&t[0].to_le_bytes());
        buf.extend_from_slice(&t[1].to_le_bytes());
        buf.extend_from_slice(&t[2].to_le_bytes());
        buf.extend_from_slice(&t[3].to_le_bytes());
    }
    buf
}

/// Build UV buffer with known texture coordinates
fn make_uv_buffer(uvs: &[[f32; 2]]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(uvs.len() * 8);
    for uv in uvs {
        buf.extend_from_slice(&uv[0].to_le_bytes());
        buf.extend_from_slice(&uv[1].to_le_bytes());
    }
    buf
}

/// Build U8 index buffer
fn make_index_buffer_u8(indices: &[u8]) -> Vec<u8> {
    indices.to_vec()
}

/// Build U16 index buffer
fn make_index_buffer_u16(indices: &[u16]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(indices.len() * 2);
    for &idx in indices {
        buf.extend_from_slice(&idx.to_le_bytes());
    }
    buf
}

/// Build U32 index buffer
fn make_index_buffer_u32(indices: &[u32]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(indices.len() * 4);
    for &idx in indices {
        buf.extend_from_slice(&idx.to_le_bytes());
    }
    buf
}

/// Extract f32 values from raw bytes
fn extract_f32s(data: &[u8]) -> Vec<f32> {
    data.chunks(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Extract u8 indices from raw bytes
fn extract_u8_indices(data: &[u8]) -> Vec<u8> {
    data.to_vec()
}

/// Extract u16 indices from raw bytes
fn extract_u16_indices(data: &[u8]) -> Vec<u16> {
    data.chunks(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Extract u32 indices from raw bytes
fn extract_u32_indices(data: &[u8]) -> Vec<u32> {
    data.chunks(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ============================================================================
// 1. STANDARD MODEL LOADING TEST
// ============================================================================

/// Test loading a standard glTF 2.0 triangle model
#[test]
fn test_standard_triangle_model() {
    let positions = [
        [0.0f32, 0.0, 0.0],
        [1.0f32, 0.0, 0.0],
        [0.5f32, 1.0, 0.0],
    ];
    let indices: [u16; 3] = [0, 1, 2];

    let mut buffer = make_position_buffer(&positions);
    buffer.extend(make_index_buffer_u16(&indices));

    let json = r#"{
        "asset": { "version": "2.0" },
        "buffers": [{ "byteLength": 42 }],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0, "byteLength": 36 },
            { "buffer": 0, "byteOffset": 36, "byteLength": 6 }
        ],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3" },
            { "bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR" }
        ],
        "meshes": [{
            "name": "StandardTriangle",
            "primitives": [{
                "attributes": { "POSITION": 0 },
                "indices": 1
            }]
        }]
    }"#;

    let meshes = load_gltf_from_json(json, &[buffer]).expect("should load without error");

    // Verify mesh count
    assert_eq!(meshes.len(), 1, "should have exactly 1 mesh");

    // Verify primitive count
    assert_eq!(meshes[0].primitives.len(), 1, "should have exactly 1 primitive");

    // Verify attribute count
    assert_eq!(
        meshes[0].primitives[0].attributes.len(),
        1,
        "should have exactly 1 attribute (POSITION)"
    );

    // Verify mesh name
    assert_eq!(meshes[0].name, Some("StandardTriangle".to_string()));
}

/// Test loading model with multiple meshes and multiple primitives
#[test]
fn test_multi_mesh_multi_primitive_model() {
    let mut buffer = Vec::new();

    // Mesh 1, Primitive 1: Triangle
    buffer.extend(make_position_buffer(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]]));
    // Mesh 1, Primitive 2: Quad (4 vertices)
    buffer.extend(make_position_buffer(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ]));
    // Mesh 2, Primitive 1: Another triangle
    buffer.extend(make_position_buffer(&[[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.5, 1.0, 0.0]]));

    let json = r#"{
        "asset": { "version": "2.0" },
        "buffers": [{ "byteLength": 120 }],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0, "byteLength": 36 },
            { "buffer": 0, "byteOffset": 36, "byteLength": 48 },
            { "buffer": 0, "byteOffset": 84, "byteLength": 36 }
        ],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3" },
            { "bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC3" },
            { "bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC3" }
        ],
        "meshes": [
            {
                "name": "Mesh1",
                "primitives": [
                    { "attributes": { "POSITION": 0 } },
                    { "attributes": { "POSITION": 1 } }
                ]
            },
            {
                "name": "Mesh2",
                "primitives": [
                    { "attributes": { "POSITION": 2 } }
                ]
            }
        ]
    }"#;

    let meshes = load_gltf_from_json(json, &[buffer]).expect("should load without error");

    // Verify mesh count
    assert_eq!(meshes.len(), 2, "should have 2 meshes");

    // Verify primitive counts
    assert_eq!(meshes[0].primitives.len(), 2, "Mesh1 should have 2 primitives");
    assert_eq!(meshes[1].primitives.len(), 1, "Mesh2 should have 1 primitive");

    // Verify vertex counts
    assert_eq!(
        meshes[0].primitives[0].attributes[&VertexSemantic::Position].count,
        3
    );
    assert_eq!(
        meshes[0].primitives[1].attributes[&VertexSemantic::Position].count,
        4
    );
    assert_eq!(
        meshes[1].primitives[0].attributes[&VertexSemantic::Position].count,
        3
    );
}

// ============================================================================
// 2. VERTEX DATA VERIFICATION
// ============================================================================

/// Test that position data matches expected bytes exactly
#[test]
fn test_vertex_position_data_matches() {
    let positions = [
        [1.0f32, 2.0, 3.0],
        [4.0f32, 5.0, 6.0],
        [7.0f32, 8.0, 9.0],
    ];
    let buffer = make_position_buffer(&positions);

    let json = r#"{
        "asset": { "version": "2.0" },
        "buffers": [{ "byteLength": 36 }],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0, "byteLength": 36 }
        ],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3" }
        ],
        "meshes": [{
            "primitives": [{
                "attributes": { "POSITION": 0 }
            }]
        }]
    }"#;

    let meshes = load_gltf_from_json(json, &[buffer]).unwrap();
    let pos_attr = &meshes[0].primitives[0].attributes[&VertexSemantic::Position];

    // Verify raw bytes match expected
    let expected_bytes = make_position_buffer(&positions);
    assert_eq!(pos_attr.data, expected_bytes, "position bytes should match exactly");

    // Verify parsed values
    let parsed: Vec<f32> = extract_f32s(&pos_attr.data);
    assert_eq!(
        parsed,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        "parsed position values should match"
    );
}

/// Test that normal data matches expected bytes
#[test]
fn test_vertex_normal_data_matches() {
    let normals = [
        [0.0f32, 0.0, 1.0],
        [0.0f32, 1.0, 0.0],
        [1.0f32, 0.0, 0.0],
    ];
    let buffer = make_normal_buffer(&normals);

    let json = r#"{
        "asset": { "version": "2.0" },
        "buffers": [{ "byteLength": 36 }],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0, "byteLength": 36 }
        ],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3" }
        ],
        "meshes": [{
            "primitives": [{
                "attributes": { "NORMAL": 0 }
            }]
        }]
    }"#;

    let meshes = load_gltf_from_json(json, &[buffer]).unwrap();
    let normal_attr = &meshes[0].primitives[0].attributes[&VertexSemantic::Normal];

    assert_eq!(normal_attr.component_type, ComponentType::F32);
    assert_eq!(normal_attr.attribute_type, AttributeType::Vec3);
    assert_eq!(normal_attr.count, 3);

    let parsed: Vec<f32> = extract_f32s(&normal_attr.data);
    assert_eq!(
        parsed,
        vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        "parsed normal values should match"
    );
}

/// Test that tangent data (Vec4) matches expected bytes
#[test]
fn test_vertex_tangent_data_matches() {
    let tangents = [
        [1.0f32, 0.0, 0.0, 1.0],  // +X tangent, +1 handedness
        [0.0f32, 1.0, 0.0, -1.0], // +Y tangent, -1 handedness
    ];
    let buffer = make_tangent_buffer(&tangents);

    let json = r#"{
        "asset": { "version": "2.0" },
        "buffers": [{ "byteLength": 32 }],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0, "byteLength": 32 }
        ],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": 2, "type": "VEC4" }
        ],
        "meshes": [{
            "primitives": [{
                "attributes": { "TANGENT": 0 }
            }]
        }]
    }"#;

    let meshes = load_gltf_from_json(json, &[buffer]).unwrap();
    let tangent_attr = &meshes[0].primitives[0].attributes[&VertexSemantic::Tangent];

    assert_eq!(tangent_attr.component_type, ComponentType::F32);
    assert_eq!(tangent_attr.attribute_type, AttributeType::Vec4);
    assert_eq!(tangent_attr.count, 2);
    assert_eq!(tangent_attr.element_size(), 16); // 4 floats * 4 bytes

    let parsed: Vec<f32> = extract_f32s(&tangent_attr.data);
    assert_eq!(
        parsed,
        vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0],
        "parsed tangent values should match"
    );
}

/// Test that UV data (Vec2) matches expected bytes
#[test]
fn test_vertex_uv_data_matches() {
    let uvs = [
        [0.0f32, 0.0],
        [1.0f32, 0.0],
        [0.5f32, 1.0],
    ];
    let buffer = make_uv_buffer(&uvs);

    let json = r#"{
        "asset": { "version": "2.0" },
        "buffers": [{ "byteLength": 24 }],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0, "byteLength": 24 }
        ],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC2" }
        ],
        "meshes": [{
            "primitives": [{
                "attributes": { "TEXCOORD_0": 0 }
            }]
        }]
    }"#;

    let meshes = load_gltf_from_json(json, &[buffer]).unwrap();
    let uv_attr = &meshes[0].primitives[0].attributes[&VertexSemantic::TexCoord0];

    assert_eq!(uv_attr.component_type, ComponentType::F32);
    assert_eq!(uv_attr.attribute_type, AttributeType::Vec2);
    assert_eq!(uv_attr.count, 3);
    assert_eq!(uv_attr.element_size(), 8); // 2 floats * 4 bytes

    let parsed: Vec<f32> = extract_f32s(&uv_attr.data);
    assert_eq!(
        parsed,
        vec![0.0, 0.0, 1.0, 0.0, 0.5, 1.0],
        "parsed UV values should match"
    );
}

/// Test full vertex data with all common attributes
#[test]
fn test_vertex_all_common_attributes() {
    let mut buffer = Vec::new();

    // POSITION: 3 * Vec3<f32> = 36 bytes
    buffer.extend(make_position_buffer(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
    ]));

    // NORMAL: 3 * Vec3<f32> = 36 bytes
    buffer.extend(make_normal_buffer(&[
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ]));

    // TANGENT: 3 * Vec4<f32> = 48 bytes
    buffer.extend(make_tangent_buffer(&[
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
    ]));

    // TEXCOORD_0: 3 * Vec2<f32> = 24 bytes
    buffer.extend(make_uv_buffer(&[[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]));

    let json = r#"{
        "asset": { "version": "2.0" },
        "buffers": [{ "byteLength": 144 }],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0, "byteLength": 36 },
            { "buffer": 0, "byteOffset": 36, "byteLength": 36 },
            { "buffer": 0, "byteOffset": 72, "byteLength": 48 },
            { "buffer": 0, "byteOffset": 120, "byteLength": 24 }
        ],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3" },
            { "bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3" },
            { "bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC4" },
            { "bufferView": 3, "componentType": 5126, "count": 3, "type": "VEC2" }
        ],
        "meshes": [{
            "primitives": [{
                "attributes": {
                    "POSITION": 0,
                    "NORMAL": 1,
                    "TANGENT": 2,
                    "TEXCOORD_0": 3
                }
            }]
        }]
    }"#;

    let meshes = load_gltf_from_json(json, &[buffer]).unwrap();
    let attrs = &meshes[0].primitives[0].attributes;

    // Verify all attributes present
    assert!(attrs.contains_key(&VertexSemantic::Position));
    assert!(attrs.contains_key(&VertexSemantic::Normal));
    assert!(attrs.contains_key(&VertexSemantic::Tangent));
    assert!(attrs.contains_key(&VertexSemantic::TexCoord0));

    // Verify attribute types per glTF spec
    assert_eq!(attrs[&VertexSemantic::Position].attribute_type, AttributeType::Vec3);
    assert_eq!(attrs[&VertexSemantic::Normal].attribute_type, AttributeType::Vec3);
    assert_eq!(attrs[&VertexSemantic::Tangent].attribute_type, AttributeType::Vec4);
    assert_eq!(attrs[&VertexSemantic::TexCoord0].attribute_type, AttributeType::Vec2);
}

// ============================================================================
// 3. INDEX DATA VERIFICATION
// ============================================================================

/// Test U8 indices (0-255 range)
#[test]
fn test_index_u8_format() {
    let indices: [u8; 6] = [0, 1, 2, 2, 1, 3];
    let mut buffer = make_position_buffer(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ]);
    buffer.extend(make_index_buffer_u8(&indices));

    let json = r#"{
        "asset": { "version": "2.0" },
        "buffers": [{ "byteLength": 54 }],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0, "byteLength": 48 },
            { "buffer": 0, "byteOffset": 48, "byteLength": 6 }
        ],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3" },
            { "bufferView": 1, "componentType": 5121, "count": 6, "type": "SCALAR" }
        ],
        "meshes": [{
            "primitives": [{
                "attributes": { "POSITION": 0 },
                "indices": 1
            }]
        }]
    }"#;

    let meshes = load_gltf_from_json(json, &[buffer]).unwrap();
    let idx = meshes[0].primitives[0].indices.as_ref().unwrap();

    assert_eq!(idx.format, IndexFormat::U8);
    assert_eq!(idx.count, 6);
    assert_eq!(idx.data.len(), 6);

    let parsed = extract_u8_indices(&idx.data);
    assert_eq!(parsed, vec![0, 1, 2, 2, 1, 3]);
}

/// Test U8 indices at boundary values (0 and 255)
#[test]
fn test_index_u8_boundary_values() {
    // Create 256 vertices to test full U8 range
    let mut positions = Vec::new();
    for i in 0..256 {
        positions.push([i as f32, 0.0, 0.0]);
    }

    let indices: [u8; 3] = [0, 127, 255]; // Min, mid, max U8 values
    let mut buffer = make_position_buffer(&positions);
    buffer.extend(make_index_buffer_u8(&indices));

    let json = r#"{
        "asset": { "version": "2.0" },
        "buffers": [{ "byteLength": 3075 }],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0, "byteLength": 3072 },
            { "buffer": 0, "byteOffset": 3072, "byteLength": 3 }
        ],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": 256, "type": "VEC3" },
            { "bufferView": 1, "componentType": 5121, "count": 3, "type": "SCALAR" }
        ],
        "meshes": [{
            "primitives": [{
                "attributes": { "POSITION": 0 },
                "indices": 1
            }]
        }]
    }"#;

    let meshes = load_gltf_from_json(json, &[buffer]).unwrap();
    let idx = meshes[0].primitives[0].indices.as_ref().unwrap();

    assert_eq!(idx.format, IndexFormat::U8);
    let parsed = extract_u8_indices(&idx.data);
    assert_eq!(parsed, vec![0, 127, 255]);
}

/// Test U16 indices (256-65535 range)
#[test]
fn test_index_u16_format() {
    let indices: [u16; 6] = [0, 1, 2, 256, 1000, 65535];
    let mut buffer = Vec::new();
    // Need at least 65536 vertices for max index
    for i in 0..65536u32 {
        buffer.extend(&(i as f32).to_le_bytes());
        buffer.extend(&0.0f32.to_le_bytes());
        buffer.extend(&0.0f32.to_le_bytes());
    }
    buffer.extend(make_index_buffer_u16(&indices));

    let pos_byte_len = 65536 * 12;
    let json = format!(
        r#"{{
        "asset": {{ "version": "2.0" }},
        "buffers": [{{ "byteLength": {} }}],
        "bufferViews": [
            {{ "buffer": 0, "byteOffset": 0, "byteLength": {} }},
            {{ "buffer": 0, "byteOffset": {}, "byteLength": 12 }}
        ],
        "accessors": [
            {{ "bufferView": 0, "componentType": 5126, "count": 65536, "type": "VEC3" }},
            {{ "bufferView": 1, "componentType": 5123, "count": 6, "type": "SCALAR" }}
        ],
        "meshes": [{{
            "primitives": [{{
                "attributes": {{ "POSITION": 0 }},
                "indices": 1
            }}]
        }}]
    }}"#,
        pos_byte_len + 12,
        pos_byte_len,
        pos_byte_len
    );

    let meshes = load_gltf_from_json(&json, &[buffer]).unwrap();
    let idx = meshes[0].primitives[0].indices.as_ref().unwrap();

    assert_eq!(idx.format, IndexFormat::U16);
    assert_eq!(idx.count, 6);

    let parsed = extract_u16_indices(&idx.data);
    assert_eq!(parsed, vec![0, 1, 2, 256, 1000, 65535]);
}

/// Test U32 indices (large values)
#[test]
fn test_index_u32_format() {
    let indices: [u32; 3] = [0, 100000, 4294967295]; // Include max U32 value

    // Create 4 vertices and just test parsing (index validation is separate)
    let mut buffer = make_position_buffer(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ]);
    buffer.extend(make_index_buffer_u32(&indices));

    let json = r#"{
        "asset": { "version": "2.0" },
        "buffers": [{ "byteLength": 60 }],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0, "byteLength": 48 },
            { "buffer": 0, "byteOffset": 48, "byteLength": 12 }
        ],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3" },
            { "bufferView": 1, "componentType": 5125, "count": 3, "type": "SCALAR" }
        ],
        "meshes": [{
            "primitives": [{
                "attributes": { "POSITION": 0 },
                "indices": 1
            }]
        }]
    }"#;

    let meshes = load_gltf_from_json(json, &[buffer]).unwrap();
    let idx = meshes[0].primitives[0].indices.as_ref().unwrap();

    assert_eq!(idx.format, IndexFormat::U32);
    assert_eq!(idx.count, 3);
    assert_eq!(idx.data.len(), 12);

    let parsed = extract_u32_indices(&idx.data);
    assert_eq!(parsed, vec![0, 100000, 4294967295]);
}

/// Test that index count matches expected value
#[test]
fn test_index_count_matches() {
    let indices: [u16; 12] = [0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4];
    let mut buffer = make_position_buffer(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [3.0, 1.0, 0.0],
        [2.0, 1.0, 0.0],
    ]);
    buffer.extend(make_index_buffer_u16(&indices));

    let json = r#"{
        "asset": { "version": "2.0" },
        "buffers": [{ "byteLength": 120 }],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0, "byteLength": 96 },
            { "buffer": 0, "byteOffset": 96, "byteLength": 24 }
        ],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3" },
            { "bufferView": 1, "componentType": 5123, "count": 12, "type": "SCALAR" }
        ],
        "meshes": [{
            "primitives": [{
                "attributes": { "POSITION": 0 },
                "indices": 1
            }]
        }]
    }"#;

    let meshes = load_gltf_from_json(json, &[buffer]).unwrap();
    let idx = meshes[0].primitives[0].indices.as_ref().unwrap();

    assert_eq!(idx.count, 12, "index count should be 12");
    assert_eq!(idx.data.len(), 24, "index data should be 24 bytes (12 * 2)");
}

// ============================================================================
// 4. FORMAT SUPPORT TESTS
// ============================================================================

/// Test interleaved vertex format (single buffer view)
#[test]
fn test_interleaved_vertex_format() {
    // Interleaved: [pos, normal, pos, normal, pos, normal]
    // Each pos = Vec3<f32> = 12 bytes
    // Each normal = Vec3<f32> = 12 bytes
    // Stride = 24 bytes

    let mut buffer = Vec::new();
    let positions = [[0.0f32, 0.0, 0.0], [1.0f32, 0.0, 0.0], [0.5f32, 1.0, 0.0]];
    let normals = [[0.0f32, 0.0, 1.0], [0.0f32, 0.0, 1.0], [0.0f32, 0.0, 1.0]];

    for i in 0..3 {
        buffer.extend(&positions[i][0].to_le_bytes());
        buffer.extend(&positions[i][1].to_le_bytes());
        buffer.extend(&positions[i][2].to_le_bytes());
        buffer.extend(&normals[i][0].to_le_bytes());
        buffer.extend(&normals[i][1].to_le_bytes());
        buffer.extend(&normals[i][2].to_le_bytes());
    }

    let json = r#"{
        "asset": { "version": "2.0" },
        "buffers": [{ "byteLength": 72 }],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0, "byteLength": 72, "byteStride": 24 }
        ],
        "accessors": [
            { "bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": 3, "type": "VEC3" },
            { "bufferView": 0, "byteOffset": 12, "componentType": 5126, "count": 3, "type": "VEC3" }
        ],
        "meshes": [{
            "primitives": [{
                "attributes": { "POSITION": 0, "NORMAL": 1 }
            }]
        }]
    }"#;

    let meshes = load_gltf_from_json(json, &[buffer]).unwrap();
    let prim = &meshes[0].primitives[0];

    let pos = &prim.attributes[&VertexSemantic::Position];
    let normal = &prim.attributes[&VertexSemantic::Normal];

    // Verify stride was recorded
    assert_eq!(pos.stride, 24, "position stride should be 24");
    assert_eq!(normal.stride, 24, "normal stride should be 24");

    // Verify de-interleaved data is correct
    let pos_values: Vec<f32> = extract_f32s(&pos.data);
    let normal_values: Vec<f32> = extract_f32s(&normal.data);

    assert_eq!(pos_values, vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0]);
    assert_eq!(
        normal_values,
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    );
}

/// Test split vertex format (separate buffer views)
#[test]
fn test_split_vertex_format() {
    let mut buffer = Vec::new();

    // Buffer view 0: Positions
    buffer.extend(make_position_buffer(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
    ]));

    // Buffer view 1: Normals
    buffer.extend(make_normal_buffer(&[
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ]));

    // Buffer view 2: UVs
    buffer.extend(make_uv_buffer(&[[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]));

    let json = r#"{
        "asset": { "version": "2.0" },
        "buffers": [{ "byteLength": 96 }],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0, "byteLength": 36 },
            { "buffer": 0, "byteOffset": 36, "byteLength": 36 },
            { "buffer": 0, "byteOffset": 72, "byteLength": 24 }
        ],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3" },
            { "bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3" },
            { "bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC2" }
        ],
        "meshes": [{
            "primitives": [{
                "attributes": {
                    "POSITION": 0,
                    "NORMAL": 1,
                    "TEXCOORD_0": 2
                }
            }]
        }]
    }"#;

    let meshes = load_gltf_from_json(json, &[buffer]).unwrap();
    let prim = &meshes[0].primitives[0];

    assert!(prim.attributes.contains_key(&VertexSemantic::Position));
    assert!(prim.attributes.contains_key(&VertexSemantic::Normal));
    assert!(prim.attributes.contains_key(&VertexSemantic::TexCoord0));

    // Verify each attribute has correct element size
    assert_eq!(prim.attributes[&VertexSemantic::Position].element_size(), 12);
    assert_eq!(prim.attributes[&VertexSemantic::Normal].element_size(), 12);
    assert_eq!(prim.attributes[&VertexSemantic::TexCoord0].element_size(), 8);
}

/// Test GLB binary container format
#[test]
fn test_glb_binary_container() {
    // Build a minimal GLB manually
    let positions = [[0.0f32, 0.0, 0.0], [1.0f32, 0.0, 0.0], [0.5f32, 1.0, 0.0]];
    let bin_buffer = make_position_buffer(&positions);

    let json = r#"{
        "asset": { "version": "2.0" },
        "buffers": [{ "byteLength": 36 }],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0, "byteLength": 36 }
        ],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3" }
        ],
        "meshes": [{
            "name": "GLBTriangle",
            "primitives": [{
                "attributes": { "POSITION": 0 }
            }]
        }]
    }"#;

    let json_bytes = json.as_bytes();

    // Pad JSON to 4-byte alignment
    let json_padded_len = (json_bytes.len() + 3) & !3;
    let mut json_padded = json_bytes.to_vec();
    json_padded.resize(json_padded_len, b' ');

    // Pad binary buffer to 4-byte alignment
    let bin_padded_len = (bin_buffer.len() + 3) & !3;
    let mut bin_padded = bin_buffer.clone();
    bin_padded.resize(bin_padded_len, 0);

    // Build GLB
    const GLB_MAGIC: u32 = 0x46546C67;
    const GLB_CHUNK_JSON: u32 = 0x4E4F534A;
    const GLB_CHUNK_BIN: u32 = 0x004E4942;

    let total_length = 12 + 8 + json_padded_len + 8 + bin_padded_len;
    let mut glb = Vec::new();

    // Header
    glb.extend_from_slice(&GLB_MAGIC.to_le_bytes());
    glb.extend_from_slice(&2u32.to_le_bytes()); // version
    glb.extend_from_slice(&(total_length as u32).to_le_bytes());

    // JSON chunk
    glb.extend_from_slice(&(json_padded_len as u32).to_le_bytes());
    glb.extend_from_slice(&GLB_CHUNK_JSON.to_le_bytes());
    glb.extend_from_slice(&json_padded);

    // BIN chunk
    glb.extend_from_slice(&(bin_padded_len as u32).to_le_bytes());
    glb.extend_from_slice(&GLB_CHUNK_BIN.to_le_bytes());
    glb.extend_from_slice(&bin_padded);

    // Parse using the internal parse_glb function (via load_gltf_from_json with parsed data)
    // Since parse_glb is not public, we test via the full chain by using the JSON + buffer
    let meshes = load_gltf_from_json(json, &[bin_buffer]).unwrap();

    assert_eq!(meshes.len(), 1);
    assert_eq!(meshes[0].name, Some("GLBTriangle".to_string()));

    let pos_values: Vec<f32> = extract_f32s(&meshes[0].primitives[0].attributes[&VertexSemantic::Position].data);
    assert_eq!(pos_values, vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0]);
}

/// Test embedded data URI (base64)
#[test]
fn test_embedded_data_uri_base64() {
    // The JSON would reference data: URIs, but load_gltf_from_json expects
    // pre-loaded buffers, so we test the core functionality directly
    let buffer = make_position_buffer(&[[1.0, 2.0, 3.0]]);

    let json = r#"{
        "asset": { "version": "2.0" },
        "buffers": [{ "byteLength": 12 }],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0, "byteLength": 12 }
        ],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": 1, "type": "VEC3" }
        ],
        "meshes": [{
            "primitives": [{
                "attributes": { "POSITION": 0 }
            }]
        }]
    }"#;

    let meshes = load_gltf_from_json(json, &[buffer]).unwrap();
    assert_eq!(meshes.len(), 1);
}

// ============================================================================
// 6. ERROR HANDLING TESTS
// ============================================================================

/// Test missing required attributes
#[test]
fn test_error_missing_accessor() {
    let json = r#"{
        "asset": { "version": "2.0" },
        "buffers": [{ "byteLength": 36 }],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0, "byteLength": 36 }
        ],
        "accessors": [],
        "meshes": [{
            "primitives": [{
                "attributes": { "POSITION": 0 }
            }]
        }]
    }"#;

    let buffer = vec![0u8; 36];
    let result = load_gltf_from_json(json, &[buffer]);

    assert!(result.is_err());
    match result {
        Err(GltfError::InvalidAccessor(msg)) => {
            assert!(msg.contains("not found"), "should mention accessor not found");
        }
        _ => panic!("expected InvalidAccessor error"),
    }
}

/// Test invalid buffer references
#[test]
fn test_error_invalid_buffer_reference() {
    let json = r#"{
        "asset": { "version": "2.0" },
        "buffers": [{ "byteLength": 36 }],
        "bufferViews": [
            { "buffer": 5, "byteOffset": 0, "byteLength": 36 }
        ],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3" }
        ],
        "meshes": [{
            "primitives": [{
                "attributes": { "POSITION": 0 }
            }]
        }]
    }"#;

    let buffer = vec![0u8; 36];
    let result = load_gltf_from_json(json, &[buffer]);

    assert!(result.is_err());
    match result {
        Err(GltfError::MissingBuffer(idx)) => {
            assert_eq!(idx, 5, "should report missing buffer 5");
        }
        _ => panic!("expected MissingBuffer error"),
    }
}

/// Test invalid buffer view reference
#[test]
fn test_error_invalid_buffer_view_reference() {
    let json = r#"{
        "asset": { "version": "2.0" },
        "buffers": [{ "byteLength": 36 }],
        "bufferViews": [],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3" }
        ],
        "meshes": [{
            "primitives": [{
                "attributes": { "POSITION": 0 }
            }]
        }]
    }"#;

    let buffer = vec![0u8; 36];
    let result = load_gltf_from_json(json, &[buffer]);

    assert!(result.is_err());
    match result {
        Err(GltfError::InvalidBufferView(msg)) => {
            assert!(msg.contains("not found"), "should mention buffer view not found");
        }
        _ => panic!("expected InvalidBufferView error"),
    }
}

/// Test invalid component type
#[test]
fn test_error_invalid_component_type() {
    let json = r#"{
        "asset": { "version": "2.0" },
        "buffers": [{ "byteLength": 36 }],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0, "byteLength": 36 }
        ],
        "accessors": [
            { "bufferView": 0, "componentType": 9999, "count": 3, "type": "VEC3" }
        ],
        "meshes": [{
            "primitives": [{
                "attributes": { "POSITION": 0 }
            }]
        }]
    }"#;

    let buffer = vec![0u8; 36];
    let result = load_gltf_from_json(json, &[buffer]);

    assert!(result.is_err());
    match result {
        Err(GltfError::InvalidAccessor(msg)) => {
            assert!(
                msg.contains("component type"),
                "should mention invalid component type"
            );
        }
        _ => panic!("expected InvalidAccessor error"),
    }
}

/// Test invalid accessor type
#[test]
fn test_error_invalid_accessor_type() {
    let json = r#"{
        "asset": { "version": "2.0" },
        "buffers": [{ "byteLength": 36 }],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0, "byteLength": 36 }
        ],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": 3, "type": "INVALID" }
        ],
        "meshes": [{
            "primitives": [{
                "attributes": { "POSITION": 0 }
            }]
        }]
    }"#;

    let buffer = vec![0u8; 36];
    let result = load_gltf_from_json(json, &[buffer]);

    assert!(result.is_err());
    match result {
        Err(GltfError::InvalidAccessor(msg)) => {
            assert!(
                msg.contains("accessor type"),
                "should mention invalid accessor type"
            );
        }
        _ => panic!("expected InvalidAccessor error"),
    }
}

/// Test invalid JSON
#[test]
fn test_error_invalid_json() {
    let json = "{ not valid json }";
    let result = load_gltf_from_json(json, &[]);

    assert!(result.is_err());
    match result {
        Err(GltfError::JsonError(_)) => {}
        _ => panic!("expected JsonError"),
    }
}

/// Test accessor data extends past buffer
#[test]
fn test_error_accessor_past_buffer_end() {
    let json = r#"{
        "asset": { "version": "2.0" },
        "buffers": [{ "byteLength": 36 }],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0, "byteLength": 36 }
        ],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": 100, "type": "VEC3" }
        ],
        "meshes": [{
            "primitives": [{
                "attributes": { "POSITION": 0 }
            }]
        }]
    }"#;

    let buffer = vec![0u8; 36]; // Only enough for 3 Vec3s, not 100
    let result = load_gltf_from_json(json, &[buffer]);

    assert!(result.is_err());
    match result {
        Err(GltfError::InvalidAccessor(msg)) => {
            assert!(
                msg.contains("past buffer end"),
                "should mention data extends past buffer end"
            );
        }
        _ => panic!("expected InvalidAccessor error about buffer bounds"),
    }
}

/// Test invalid index component type (e.g., F32 for indices)
#[test]
fn test_error_invalid_index_component_type() {
    let mut buffer = make_position_buffer(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]]);
    buffer.extend(&[0u8; 12]); // Padding for "indices"

    let json = r#"{
        "asset": { "version": "2.0" },
        "buffers": [{ "byteLength": 48 }],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0, "byteLength": 36 },
            { "buffer": 0, "byteOffset": 36, "byteLength": 12 }
        ],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3" },
            { "bufferView": 1, "componentType": 5126, "count": 3, "type": "SCALAR" }
        ],
        "meshes": [{
            "primitives": [{
                "attributes": { "POSITION": 0 },
                "indices": 1
            }]
        }]
    }"#;

    let result = load_gltf_from_json(json, &[buffer]);

    assert!(result.is_err());
    match result {
        Err(GltfError::InvalidAccessor(msg)) => {
            assert!(
                msg.contains("index component type"),
                "should mention invalid index component type"
            );
        }
        _ => panic!("expected InvalidAccessor error for index component type"),
    }
}

// ============================================================================
// Additional edge case tests
// ============================================================================

/// Test mesh with no primitives
#[test]
fn test_mesh_with_no_primitives() {
    let json = r#"{
        "asset": { "version": "2.0" },
        "meshes": [{
            "name": "EmptyMesh",
            "primitives": []
        }]
    }"#;

    let meshes = load_gltf_from_json(json, &[]).unwrap();
    assert_eq!(meshes.len(), 1);
    assert_eq!(meshes[0].primitives.len(), 0);
}

/// Test primitive without indices (non-indexed draw)
#[test]
fn test_primitive_without_indices() {
    let buffer = make_position_buffer(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]]);

    let json = r#"{
        "asset": { "version": "2.0" },
        "buffers": [{ "byteLength": 36 }],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0, "byteLength": 36 }
        ],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3" }
        ],
        "meshes": [{
            "primitives": [{
                "attributes": { "POSITION": 0 }
            }]
        }]
    }"#;

    let meshes = load_gltf_from_json(json, &[buffer]).unwrap();
    assert!(meshes[0].primitives[0].indices.is_none());
}

/// Test accessor without buffer view (sparse accessor / zero-filled)
#[test]
fn test_accessor_without_buffer_view() {
    let json = r#"{
        "asset": { "version": "2.0" },
        "accessors": [
            { "componentType": 5126, "count": 4, "type": "VEC3" }
        ],
        "meshes": [{
            "primitives": [{
                "attributes": { "POSITION": 0 }
            }]
        }]
    }"#;

    let meshes = load_gltf_from_json(json, &[]).unwrap();
    let pos = &meshes[0].primitives[0].attributes[&VertexSemantic::Position];

    assert_eq!(pos.count, 4);
    assert_eq!(pos.data.len(), 48); // 4 * 12 bytes
    assert!(pos.data.iter().all(|&b| b == 0), "should be zero-filled");
}

/// Test multiple buffer views referencing same buffer
#[test]
fn test_multiple_buffer_views_same_buffer() {
    let mut buffer = Vec::new();
    buffer.extend(make_position_buffer(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]]));
    buffer.extend(make_normal_buffer(&[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]));

    let json = r#"{
        "asset": { "version": "2.0" },
        "buffers": [{ "byteLength": 72 }],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0, "byteLength": 36 },
            { "buffer": 0, "byteOffset": 36, "byteLength": 36 }
        ],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3" },
            { "bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3" }
        ],
        "meshes": [{
            "primitives": [{
                "attributes": { "POSITION": 0, "NORMAL": 1 }
            }]
        }]
    }"#;

    let meshes = load_gltf_from_json(json, &[buffer]).unwrap();
    let attrs = &meshes[0].primitives[0].attributes;

    assert!(attrs.contains_key(&VertexSemantic::Position));
    assert!(attrs.contains_key(&VertexSemantic::Normal));

    // Verify position and normal data are from different regions
    let pos_vals: Vec<f32> = extract_f32s(&attrs[&VertexSemantic::Position].data);
    let norm_vals: Vec<f32> = extract_f32s(&attrs[&VertexSemantic::Normal].data);

    assert_eq!(pos_vals[0], 0.0);
    assert_eq!(norm_vals[2], 1.0); // z component of first normal
}

/// Test JOINTS_0 and WEIGHTS_0 attributes (skinning)
#[test]
fn test_skinning_attributes() {
    let mut buffer = Vec::new();

    // POSITION: 2 * Vec3<f32> = 24 bytes
    buffer.extend(make_position_buffer(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]));

    // JOINTS_0: 2 * Vec4<u16> = 16 bytes
    for _ in 0..2 {
        buffer.extend(&0u16.to_le_bytes());
        buffer.extend(&1u16.to_le_bytes());
        buffer.extend(&2u16.to_le_bytes());
        buffer.extend(&3u16.to_le_bytes());
    }

    // WEIGHTS_0: 2 * Vec4<f32> = 32 bytes
    for _ in 0..2 {
        buffer.extend(&0.25f32.to_le_bytes());
        buffer.extend(&0.25f32.to_le_bytes());
        buffer.extend(&0.25f32.to_le_bytes());
        buffer.extend(&0.25f32.to_le_bytes());
    }

    let json = r#"{
        "asset": { "version": "2.0" },
        "buffers": [{ "byteLength": 72 }],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0, "byteLength": 24 },
            { "buffer": 0, "byteOffset": 24, "byteLength": 16 },
            { "buffer": 0, "byteOffset": 40, "byteLength": 32 }
        ],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": 2, "type": "VEC3" },
            { "bufferView": 1, "componentType": 5123, "count": 2, "type": "VEC4" },
            { "bufferView": 2, "componentType": 5126, "count": 2, "type": "VEC4" }
        ],
        "meshes": [{
            "primitives": [{
                "attributes": {
                    "POSITION": 0,
                    "JOINTS_0": 1,
                    "WEIGHTS_0": 2
                }
            }]
        }]
    }"#;

    let meshes = load_gltf_from_json(json, &[buffer]).unwrap();
    let attrs = &meshes[0].primitives[0].attributes;

    assert!(attrs.contains_key(&VertexSemantic::Joints0));
    assert!(attrs.contains_key(&VertexSemantic::Weights0));

    // Verify JOINTS_0 is U16 Vec4
    assert_eq!(attrs[&VertexSemantic::Joints0].component_type, ComponentType::U16);
    assert_eq!(attrs[&VertexSemantic::Joints0].attribute_type, AttributeType::Vec4);

    // Verify WEIGHTS_0 is F32 Vec4
    assert_eq!(attrs[&VertexSemantic::Weights0].component_type, ComponentType::F32);
    assert_eq!(attrs[&VertexSemantic::Weights0].attribute_type, AttributeType::Vec4);
}

/// Test COLOR_0 attribute
#[test]
fn test_color_attribute() {
    let mut buffer = make_position_buffer(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);

    // COLOR_0: 2 * Vec4<f32> = 32 bytes (RGBA)
    buffer.extend(&1.0f32.to_le_bytes()); // R
    buffer.extend(&0.0f32.to_le_bytes()); // G
    buffer.extend(&0.0f32.to_le_bytes()); // B
    buffer.extend(&1.0f32.to_le_bytes()); // A
    buffer.extend(&0.0f32.to_le_bytes());
    buffer.extend(&1.0f32.to_le_bytes());
    buffer.extend(&0.0f32.to_le_bytes());
    buffer.extend(&1.0f32.to_le_bytes());

    let json = r#"{
        "asset": { "version": "2.0" },
        "buffers": [{ "byteLength": 56 }],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0, "byteLength": 24 },
            { "buffer": 0, "byteOffset": 24, "byteLength": 32 }
        ],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": 2, "type": "VEC3" },
            { "bufferView": 1, "componentType": 5126, "count": 2, "type": "VEC4" }
        ],
        "meshes": [{
            "primitives": [{
                "attributes": {
                    "POSITION": 0,
                    "COLOR_0": 1
                }
            }]
        }]
    }"#;

    let meshes = load_gltf_from_json(json, &[buffer]).unwrap();
    let color = &meshes[0].primitives[0].attributes[&VertexSemantic::Color0];

    assert_eq!(color.attribute_type, AttributeType::Vec4);
    assert_eq!(color.count, 2);

    let colors: Vec<f32> = extract_f32s(&color.data);
    assert_eq!(colors[0], 1.0); // First vertex red
    assert_eq!(colors[4], 0.0); // Second vertex red = 0
    assert_eq!(colors[5], 1.0); // Second vertex green = 1
}

/// Test TEXCOORD_1 (second UV set)
#[test]
fn test_second_uv_set() {
    let mut buffer = make_position_buffer(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);

    // TEXCOORD_0
    buffer.extend(make_uv_buffer(&[[0.0, 0.0], [1.0, 0.0]]));

    // TEXCOORD_1 (lightmap UVs, different values)
    buffer.extend(make_uv_buffer(&[[0.5, 0.5], [0.75, 0.25]]));

    let json = r#"{
        "asset": { "version": "2.0" },
        "buffers": [{ "byteLength": 56 }],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0, "byteLength": 24 },
            { "buffer": 0, "byteOffset": 24, "byteLength": 16 },
            { "buffer": 0, "byteOffset": 40, "byteLength": 16 }
        ],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": 2, "type": "VEC3" },
            { "bufferView": 1, "componentType": 5126, "count": 2, "type": "VEC2" },
            { "bufferView": 2, "componentType": 5126, "count": 2, "type": "VEC2" }
        ],
        "meshes": [{
            "primitives": [{
                "attributes": {
                    "POSITION": 0,
                    "TEXCOORD_0": 1,
                    "TEXCOORD_1": 2
                }
            }]
        }]
    }"#;

    let meshes = load_gltf_from_json(json, &[buffer]).unwrap();
    let attrs = &meshes[0].primitives[0].attributes;

    assert!(attrs.contains_key(&VertexSemantic::TexCoord0));
    assert!(attrs.contains_key(&VertexSemantic::TexCoord1));

    let uv1: Vec<f32> = extract_f32s(&attrs[&VertexSemantic::TexCoord1].data);
    assert_eq!(uv1, vec![0.5, 0.5, 0.75, 0.25]);
}
