//! Enhanced glTF 2.0 parser with progressive loading and streaming support.
//!
//! This module re-exports the glTF parser from the main `gltf` module, providing
//! a convenient access point through the asset pipeline.
//!
//! # Features
//!
//! - Schema validation with detailed error reporting
//! - Progressive loading stages (bounds -> geometry -> attributes -> skinning)
//! - Streaming support for large files (>2GB)
//! - Thread-safe parsing via `Send + Sync` types
//! - Node hierarchy extraction with world transforms
//! - Skin/skeleton support
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::asset::gltf_parser::{GltfParser, LoadStage};
//!
//! // Create parser from JSON
//! let parser = GltfParser::new(&json)?;
//!
//! // Progressive loading
//! let bounds = parser.load_bounds();  // Fast, no buffer reads
//! let full = parser.load_full(&buffers)?;
//! ```

// Re-export all types from the main gltf module
pub use crate::gltf::{
    // Error types
    GltfError,
    ValidationError,
    ValidationResult,
    ValidationSeverity,

    // Math types
    Mat4,
    Aabb,

    // Core mesh types
    GltfMesh,
    GltfPrimitive,
    VertexAttribute,
    VertexSemantic,
    IndexBuffer,
    IndexFormat,
    ComponentType,
    AttributeType,

    // Node and scene types
    GltfNode,
    GltfSkin,
    GltfScene,

    // Progressive loading
    LoadStage,
    BoundsResult,
    GltfDocument,

    // Parser
    GltfParser,

    // Streaming
    StreamingBufferReader,
    load_gltf_streaming,

    // Simple loading functions
    load_gltf,
    load_gltf_from_json,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reexports_available() {
        // Verify all re-exports are accessible
        fn _assert_types() {
            let _: fn(&str) -> Result<GltfParser, GltfError> = GltfParser::new;
            let _: LoadStage = LoadStage::Bounds;
            let _: Mat4 = Mat4::IDENTITY;
            let _: Aabb = Aabb::EMPTY;
            let _: ValidationSeverity = ValidationSeverity::Error;
        }
    }

    #[test]
    fn test_parser_creation_via_asset_module() {
        let json = r#"{
            "asset": { "version": "2.0" },
            "buffers": [{ "byteLength": 42 }],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 36 },
                { "buffer": 0, "byteOffset": 36, "byteLength": 6 }
            ],
            "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                  "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0] },
                { "bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR" }
            ],
            "meshes": [{
                "name": "Triangle",
                "primitives": [{ "attributes": { "POSITION": 0 }, "indices": 1 }]
            }]
        }"#;

        let parser = GltfParser::new(json).unwrap();
        assert!(parser.validation().is_valid);
        assert_eq!(parser.mesh_count(), 1);

        let bounds = parser.load_bounds();
        assert_eq!(bounds.mesh_count, 1);
        assert!(bounds.scene_bounds.is_valid());
    }
}
