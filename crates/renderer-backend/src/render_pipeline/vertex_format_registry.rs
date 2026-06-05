//! Standard vertex format registry for TRINITY engine.
//!
//! Provides pre-defined vertex formats for common mesh types (static, skinned,
//! terrain, particle, UI) with ID-based lookup and custom format registration.

use std::collections::HashMap;

use crate::render_pipeline::{VertexAttributeDescriptor, VertexBufferLayoutDescriptor};

// ---------------------------------------------------------------------------
// VertexFormatId
// ---------------------------------------------------------------------------

/// Identifier for a vertex format in the registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VertexFormatId {
    /// Standard static mesh format (48 bytes).
    StaticMesh,
    /// Skinned mesh with bone data (72 bytes).
    SkinnedMesh,
    /// Terrain vertex format (32 bytes).
    Terrain,
    /// Particle vertex format (36 bytes).
    Particle,
    /// UI vertex format (20 bytes).
    Ui,
    /// Custom user-defined format.
    Custom(u32),
}

// ---------------------------------------------------------------------------
// VertexFormat
// ---------------------------------------------------------------------------

/// Describes a complete vertex format with metadata.
#[derive(Debug, Clone)]
pub struct VertexFormat {
    /// Unique identifier for this format.
    pub id: VertexFormatId,
    /// Human-readable name.
    pub name: &'static str,
    /// Total byte stride per vertex.
    pub stride: u64,
    /// Ordered list of vertex attributes.
    pub attributes: Vec<VertexAttributeDescriptor>,
}

impl VertexFormat {
    /// Convert to a wgpu-compatible buffer layout descriptor.
    pub fn to_buffer_layout(&self) -> VertexBufferLayoutDescriptor {
        let mut layout = VertexBufferLayoutDescriptor::per_vertex(self.stride);
        for attr in &self.attributes {
            layout = layout.attribute(*attr);
        }
        layout
    }
}

// ---------------------------------------------------------------------------
// Standard Format Constructors
// ---------------------------------------------------------------------------

/// Creates the standard static mesh vertex format (48 bytes).
///
/// Layout:
/// - position: Float32x3 (12 bytes) @ location 0
/// - normal: Float32x3 (12 bytes) @ location 1
/// - tangent: Float32x4 (16 bytes) @ location 2
/// - uv: Float32x2 (8 bytes) @ location 3
pub fn static_mesh() -> VertexFormat {
    VertexFormat {
        id: VertexFormatId::StaticMesh,
        name: "StaticMesh",
        stride: 48,
        attributes: vec![
            VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x3, 0, 0),  // position
            VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x3, 12, 1), // normal
            VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x4, 24, 2), // tangent
            VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x2, 40, 3), // uv
        ],
    }
}

/// Creates the skinned mesh vertex format (72 bytes).
///
/// Layout:
/// - position: Float32x3 (12 bytes) @ location 0
/// - normal: Float32x3 (12 bytes) @ location 1
/// - tangent: Float32x4 (16 bytes) @ location 2
/// - uv: Float32x2 (8 bytes) @ location 3
/// - bone_indices: Uint16x4 (8 bytes) @ location 4
/// - bone_weights: Float32x4 (16 bytes) @ location 5
pub fn skinned_mesh() -> VertexFormat {
    VertexFormat {
        id: VertexFormatId::SkinnedMesh,
        name: "SkinnedMesh",
        stride: 72,
        attributes: vec![
            VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x3, 0, 0),  // position
            VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x3, 12, 1), // normal
            VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x4, 24, 2), // tangent
            VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x2, 40, 3), // uv
            VertexAttributeDescriptor::new(wgpu::VertexFormat::Uint16x4, 48, 4),  // bone_indices
            VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x4, 56, 5), // bone_weights
        ],
    }
}

/// Creates the terrain vertex format (32 bytes).
///
/// Layout:
/// - position: Float32x3 (12 bytes) @ location 0
/// - normal: Float32x3 (12 bytes) @ location 1
/// - uv: Float32x2 (8 bytes) @ location 2
pub fn terrain() -> VertexFormat {
    VertexFormat {
        id: VertexFormatId::Terrain,
        name: "Terrain",
        stride: 32,
        attributes: vec![
            VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x3, 0, 0),  // position
            VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x3, 12, 1), // normal
            VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x2, 24, 2), // uv
        ],
    }
}

/// Creates the particle vertex format (36 bytes).
///
/// Layout:
/// - position: Float32x3 (12 bytes) @ location 0
/// - color: Float32x4 (16 bytes) @ location 1
/// - size_rotation: Float32x2 (8 bytes) @ location 2 [size, rotation_angle]
pub fn particle() -> VertexFormat {
    VertexFormat {
        id: VertexFormatId::Particle,
        name: "Particle",
        stride: 36,
        attributes: vec![
            VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x3, 0, 0),  // position
            VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x4, 12, 1), // color
            VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x2, 28, 2), // size_rotation
        ],
    }
}

/// Creates the UI vertex format (20 bytes).
///
/// Layout:
/// - position: Float32x2 (8 bytes) @ location 0
/// - uv: Float32x2 (8 bytes) @ location 1
/// - color: Unorm8x4 (4 bytes) @ location 2
pub fn ui() -> VertexFormat {
    VertexFormat {
        id: VertexFormatId::Ui,
        name: "UI",
        stride: 20,
        attributes: vec![
            VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x2, 0, 0),  // position
            VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x2, 8, 1),  // uv
            VertexAttributeDescriptor::new(wgpu::VertexFormat::Unorm8x4, 16, 2),  // color
        ],
    }
}

// ---------------------------------------------------------------------------
// VertexFormatRegistry
// ---------------------------------------------------------------------------

/// Registry of vertex formats with ID-based lookup.
///
/// Pre-registers all standard formats on construction. Custom formats can be
/// added via [`register`](Self::register).
///
/// # Thread Safety
///
/// The registry is `Send + Sync` and can be shared across threads.
#[derive(Debug)]
pub struct VertexFormatRegistry {
    formats: HashMap<VertexFormatId, VertexFormat>,
}

impl Default for VertexFormatRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl VertexFormatRegistry {
    /// Creates a new registry with all standard formats pre-registered.
    pub fn new() -> Self {
        let mut registry = Self {
            formats: HashMap::new(),
        };
        registry.register(static_mesh());
        registry.register(skinned_mesh());
        registry.register(terrain());
        registry.register(particle());
        registry.register(ui());
        registry
    }

    /// Registers a vertex format. Replaces any existing format with the same ID.
    pub fn register(&mut self, format: VertexFormat) {
        self.formats.insert(format.id, format);
    }

    /// Retrieves a format by ID.
    pub fn get(&self, id: VertexFormatId) -> Option<&VertexFormat> {
        self.formats.get(&id)
    }

    /// Returns a buffer layout descriptor for the given format ID.
    pub fn get_buffer_layout(&self, id: VertexFormatId) -> Option<VertexBufferLayoutDescriptor> {
        self.get(id).map(|f| f.to_buffer_layout())
    }

    /// Checks if a format is registered.
    pub fn contains(&self, id: VertexFormatId) -> bool {
        self.formats.contains_key(&id)
    }

    /// Returns an iterator over all registered formats.
    pub fn iter(&self) -> impl Iterator<Item = &VertexFormat> {
        self.formats.values()
    }

    /// Returns the number of registered formats.
    pub fn len(&self) -> usize {
        self.formats.len()
    }

    /// Returns true if no formats are registered.
    pub fn is_empty(&self) -> bool {
        self.formats.is_empty()
    }
}

// Ensure thread safety
static_assertions::assert_impl_all!(VertexFormatRegistry: Send, Sync);

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_static_mesh_format() {
        let format = static_mesh();
        assert_eq!(format.id, VertexFormatId::StaticMesh);
        assert_eq!(format.stride, 48);
        assert_eq!(format.attributes.len(), 4);
        assert_eq!(format.name, "StaticMesh");
    }

    #[test]
    fn test_skinned_mesh_format() {
        let format = skinned_mesh();
        assert_eq!(format.id, VertexFormatId::SkinnedMesh);
        assert_eq!(format.stride, 72);
        assert_eq!(format.attributes.len(), 6);
        assert_eq!(format.name, "SkinnedMesh");
    }

    #[test]
    fn test_terrain_format() {
        let format = terrain();
        assert_eq!(format.id, VertexFormatId::Terrain);
        assert_eq!(format.stride, 32);
        assert_eq!(format.attributes.len(), 3);
        assert_eq!(format.name, "Terrain");
    }

    #[test]
    fn test_particle_format() {
        let format = particle();
        assert_eq!(format.id, VertexFormatId::Particle);
        assert_eq!(format.stride, 36);
        assert_eq!(format.attributes.len(), 3);
        assert_eq!(format.name, "Particle");
    }

    #[test]
    fn test_ui_format() {
        let format = ui();
        assert_eq!(format.id, VertexFormatId::Ui);
        assert_eq!(format.stride, 20);
        assert_eq!(format.attributes.len(), 3);
        assert_eq!(format.name, "UI");
    }

    #[test]
    fn test_registry_new_preregisters_standard_formats() {
        let registry = VertexFormatRegistry::new();
        assert_eq!(registry.len(), 5);
        assert!(registry.contains(VertexFormatId::StaticMesh));
        assert!(registry.contains(VertexFormatId::SkinnedMesh));
        assert!(registry.contains(VertexFormatId::Terrain));
        assert!(registry.contains(VertexFormatId::Particle));
        assert!(registry.contains(VertexFormatId::Ui));
    }

    #[test]
    fn test_registry_get() {
        let registry = VertexFormatRegistry::new();
        let format = registry.get(VertexFormatId::StaticMesh).unwrap();
        assert_eq!(format.stride, 48);
        assert!(registry.get(VertexFormatId::Custom(999)).is_none());
    }

    #[test]
    fn test_registry_register_custom() {
        let mut registry = VertexFormatRegistry::new();
        let custom = VertexFormat {
            id: VertexFormatId::Custom(42),
            name: "CustomFormat",
            stride: 16,
            attributes: vec![
                VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x4, 0, 0),
            ],
        };
        registry.register(custom);
        assert_eq!(registry.len(), 6);
        assert!(registry.contains(VertexFormatId::Custom(42)));
        let retrieved = registry.get(VertexFormatId::Custom(42)).unwrap();
        assert_eq!(retrieved.stride, 16);
    }

    #[test]
    fn test_registry_get_buffer_layout() {
        let registry = VertexFormatRegistry::new();
        let layout = registry.get_buffer_layout(VertexFormatId::Terrain).unwrap();
        assert_eq!(layout.array_stride, 32);
        assert_eq!(layout.attributes.len(), 3);
        assert_eq!(layout.step_mode, wgpu::VertexStepMode::Vertex);
    }

    #[test]
    fn test_buffer_layout_conversion() {
        let format = static_mesh();
        let layout = format.to_buffer_layout();
        assert_eq!(layout.array_stride, 48);
        assert_eq!(layout.attributes.len(), 4);
        assert_eq!(layout.attributes[0].offset, 0);
        assert_eq!(layout.attributes[1].offset, 12);
        assert_eq!(layout.attributes[2].offset, 24);
        assert_eq!(layout.attributes[3].offset, 40);
    }

    #[test]
    fn test_registry_iter() {
        let registry = VertexFormatRegistry::new();
        let formats: Vec<_> = registry.iter().collect();
        assert_eq!(formats.len(), 5);
    }

    #[test]
    fn test_registry_is_empty() {
        let registry = VertexFormatRegistry::new();
        assert!(!registry.is_empty());
    }

    #[test]
    fn test_vertex_format_id_equality() {
        assert_eq!(VertexFormatId::StaticMesh, VertexFormatId::StaticMesh);
        assert_ne!(VertexFormatId::StaticMesh, VertexFormatId::SkinnedMesh);
        assert_eq!(VertexFormatId::Custom(1), VertexFormatId::Custom(1));
        assert_ne!(VertexFormatId::Custom(1), VertexFormatId::Custom(2));
    }

    #[test]
    fn test_vertex_format_id_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(VertexFormatId::StaticMesh);
        set.insert(VertexFormatId::Custom(42));
        assert!(set.contains(&VertexFormatId::StaticMesh));
        assert!(set.contains(&VertexFormatId::Custom(42)));
        assert!(!set.contains(&VertexFormatId::Terrain));
    }

    #[test]
    fn test_static_mesh_attribute_offsets() {
        let format = static_mesh();
        assert_eq!(format.attributes[0].offset, 0);   // position
        assert_eq!(format.attributes[1].offset, 12);  // normal
        assert_eq!(format.attributes[2].offset, 24);  // tangent
        assert_eq!(format.attributes[3].offset, 40);  // uv
    }

    #[test]
    fn test_skinned_mesh_attribute_offsets() {
        let format = skinned_mesh();
        assert_eq!(format.attributes[0].offset, 0);   // position
        assert_eq!(format.attributes[1].offset, 12);  // normal
        assert_eq!(format.attributes[2].offset, 24);  // tangent
        assert_eq!(format.attributes[3].offset, 40);  // uv
        assert_eq!(format.attributes[4].offset, 48);  // bone_indices
        assert_eq!(format.attributes[5].offset, 56);  // bone_weights
    }

    #[test]
    fn test_register_overwrites_existing() {
        let mut registry = VertexFormatRegistry::new();
        let modified = VertexFormat {
            id: VertexFormatId::StaticMesh,
            name: "ModifiedStaticMesh",
            stride: 64,
            attributes: vec![],
        };
        registry.register(modified);
        let format = registry.get(VertexFormatId::StaticMesh).unwrap();
        assert_eq!(format.stride, 64);
        assert_eq!(format.name, "ModifiedStaticMesh");
    }

    // -------------------------------------------------------------------------
    // VertexFormatId Whitebox Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_vertex_format_id_custom_zero() {
        let id = VertexFormatId::Custom(0);
        assert_eq!(id, VertexFormatId::Custom(0));
        assert_ne!(id, VertexFormatId::Custom(1));
    }

    #[test]
    fn test_vertex_format_id_custom_one() {
        let id = VertexFormatId::Custom(1);
        assert_eq!(id, VertexFormatId::Custom(1));
        assert_ne!(id, VertexFormatId::Custom(0));
    }

    #[test]
    fn test_vertex_format_id_custom_max() {
        let id = VertexFormatId::Custom(u32::MAX);
        assert_eq!(id, VertexFormatId::Custom(u32::MAX));
        assert_ne!(id, VertexFormatId::Custom(u32::MAX - 1));
    }

    #[test]
    fn test_vertex_format_id_partial_eq_all_variants() {
        // Each variant equals itself
        assert_eq!(VertexFormatId::StaticMesh, VertexFormatId::StaticMesh);
        assert_eq!(VertexFormatId::SkinnedMesh, VertexFormatId::SkinnedMesh);
        assert_eq!(VertexFormatId::Terrain, VertexFormatId::Terrain);
        assert_eq!(VertexFormatId::Particle, VertexFormatId::Particle);
        assert_eq!(VertexFormatId::Ui, VertexFormatId::Ui);

        // Different variants are not equal
        assert_ne!(VertexFormatId::StaticMesh, VertexFormatId::SkinnedMesh);
        assert_ne!(VertexFormatId::Terrain, VertexFormatId::Particle);
        assert_ne!(VertexFormatId::Ui, VertexFormatId::StaticMesh);

        // Custom with different values
        assert_ne!(VertexFormatId::Custom(0), VertexFormatId::Custom(1));

        // Custom vs standard
        assert_ne!(VertexFormatId::Custom(0), VertexFormatId::StaticMesh);
        assert_ne!(VertexFormatId::Custom(1), VertexFormatId::Terrain);
    }

    #[test]
    fn test_vertex_format_id_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        fn compute_hash<T: Hash>(value: &T) -> u64 {
            let mut hasher = DefaultHasher::new();
            value.hash(&mut hasher);
            hasher.finish()
        }

        // Same value produces same hash
        let hash1 = compute_hash(&VertexFormatId::StaticMesh);
        let hash2 = compute_hash(&VertexFormatId::StaticMesh);
        assert_eq!(hash1, hash2);

        let hash3 = compute_hash(&VertexFormatId::Custom(42));
        let hash4 = compute_hash(&VertexFormatId::Custom(42));
        assert_eq!(hash3, hash4);

        // Different values produce different hashes (with high probability)
        let hash_static = compute_hash(&VertexFormatId::StaticMesh);
        let hash_skinned = compute_hash(&VertexFormatId::SkinnedMesh);
        assert_ne!(hash_static, hash_skinned);
    }

    #[test]
    fn test_vertex_format_id_debug() {
        // Test Debug formatting
        let debug_str = format!("{:?}", VertexFormatId::StaticMesh);
        assert!(debug_str.contains("StaticMesh"));

        let custom_debug = format!("{:?}", VertexFormatId::Custom(42));
        assert!(custom_debug.contains("Custom"));
        assert!(custom_debug.contains("42"));
    }

    #[test]
    fn test_vertex_format_id_clone() {
        let id = VertexFormatId::Custom(100);
        let cloned = id.clone();
        assert_eq!(id, cloned);
    }

    #[test]
    fn test_vertex_format_id_copy() {
        let id = VertexFormatId::Terrain;
        let copied = id; // Copy
        assert_eq!(id, copied);
    }

    // -------------------------------------------------------------------------
    // VertexFormat Whitebox Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_static_mesh_attribute_count() {
        let format = static_mesh();
        assert_eq!(format.attributes.len(), 4);
    }

    #[test]
    fn test_skinned_mesh_attribute_count() {
        let format = skinned_mesh();
        assert_eq!(format.attributes.len(), 6);
    }

    #[test]
    fn test_terrain_attribute_count() {
        let format = terrain();
        assert_eq!(format.attributes.len(), 3);
    }

    #[test]
    fn test_particle_attribute_count() {
        let format = particle();
        assert_eq!(format.attributes.len(), 3);
    }

    #[test]
    fn test_ui_attribute_count() {
        let format = ui();
        assert_eq!(format.attributes.len(), 3);
    }

    #[test]
    fn test_static_mesh_shader_locations() {
        let format = static_mesh();
        assert_eq!(format.attributes[0].shader_location, 0); // position
        assert_eq!(format.attributes[1].shader_location, 1); // normal
        assert_eq!(format.attributes[2].shader_location, 2); // tangent
        assert_eq!(format.attributes[3].shader_location, 3); // uv
    }

    #[test]
    fn test_skinned_mesh_shader_locations() {
        let format = skinned_mesh();
        assert_eq!(format.attributes[0].shader_location, 0); // position
        assert_eq!(format.attributes[1].shader_location, 1); // normal
        assert_eq!(format.attributes[2].shader_location, 2); // tangent
        assert_eq!(format.attributes[3].shader_location, 3); // uv
        assert_eq!(format.attributes[4].shader_location, 4); // bone_indices
        assert_eq!(format.attributes[5].shader_location, 5); // bone_weights
    }

    #[test]
    fn test_terrain_shader_locations() {
        let format = terrain();
        assert_eq!(format.attributes[0].shader_location, 0); // position
        assert_eq!(format.attributes[1].shader_location, 1); // normal
        assert_eq!(format.attributes[2].shader_location, 2); // uv
    }

    #[test]
    fn test_particle_shader_locations() {
        let format = particle();
        assert_eq!(format.attributes[0].shader_location, 0); // position
        assert_eq!(format.attributes[1].shader_location, 1); // color
        assert_eq!(format.attributes[2].shader_location, 2); // size_rotation
    }

    #[test]
    fn test_ui_shader_locations() {
        let format = ui();
        assert_eq!(format.attributes[0].shader_location, 0); // position
        assert_eq!(format.attributes[1].shader_location, 1); // uv
        assert_eq!(format.attributes[2].shader_location, 2); // color
    }

    #[test]
    fn test_static_mesh_attribute_formats() {
        let format = static_mesh();
        assert_eq!(format.attributes[0].format, wgpu::VertexFormat::Float32x3); // position
        assert_eq!(format.attributes[1].format, wgpu::VertexFormat::Float32x3); // normal
        assert_eq!(format.attributes[2].format, wgpu::VertexFormat::Float32x4); // tangent
        assert_eq!(format.attributes[3].format, wgpu::VertexFormat::Float32x2); // uv
    }

    #[test]
    fn test_skinned_mesh_attribute_formats() {
        let format = skinned_mesh();
        assert_eq!(format.attributes[0].format, wgpu::VertexFormat::Float32x3); // position
        assert_eq!(format.attributes[1].format, wgpu::VertexFormat::Float32x3); // normal
        assert_eq!(format.attributes[2].format, wgpu::VertexFormat::Float32x4); // tangent
        assert_eq!(format.attributes[3].format, wgpu::VertexFormat::Float32x2); // uv
        assert_eq!(format.attributes[4].format, wgpu::VertexFormat::Uint16x4);  // bone_indices
        assert_eq!(format.attributes[5].format, wgpu::VertexFormat::Float32x4); // bone_weights
    }

    #[test]
    fn test_terrain_attribute_formats() {
        let format = terrain();
        assert_eq!(format.attributes[0].format, wgpu::VertexFormat::Float32x3); // position
        assert_eq!(format.attributes[1].format, wgpu::VertexFormat::Float32x3); // normal
        assert_eq!(format.attributes[2].format, wgpu::VertexFormat::Float32x2); // uv
    }

    #[test]
    fn test_particle_attribute_formats() {
        let format = particle();
        assert_eq!(format.attributes[0].format, wgpu::VertexFormat::Float32x3); // position
        assert_eq!(format.attributes[1].format, wgpu::VertexFormat::Float32x4); // color
        assert_eq!(format.attributes[2].format, wgpu::VertexFormat::Float32x2); // size_rotation
    }

    #[test]
    fn test_ui_attribute_formats() {
        let format = ui();
        assert_eq!(format.attributes[0].format, wgpu::VertexFormat::Float32x2); // position
        assert_eq!(format.attributes[1].format, wgpu::VertexFormat::Float32x2); // uv
        assert_eq!(format.attributes[2].format, wgpu::VertexFormat::Unorm8x4);  // color
    }

    #[test]
    fn test_to_buffer_layout_preserves_stride() {
        let format = static_mesh();
        let layout = format.to_buffer_layout();
        assert_eq!(layout.array_stride, format.stride);
    }

    #[test]
    fn test_to_buffer_layout_preserves_attributes() {
        let format = terrain();
        let layout = format.to_buffer_layout();
        assert_eq!(layout.attributes.len(), format.attributes.len());

        for (i, attr) in layout.attributes.iter().enumerate() {
            assert_eq!(attr.offset, format.attributes[i].offset);
            assert_eq!(attr.format, format.attributes[i].format);
            assert_eq!(attr.shader_location, format.attributes[i].shader_location);
        }
    }

    #[test]
    fn test_to_buffer_layout_is_per_vertex() {
        let format = skinned_mesh();
        let layout = format.to_buffer_layout();
        assert_eq!(layout.step_mode, wgpu::VertexStepMode::Vertex);
    }

    // -------------------------------------------------------------------------
    // VertexFormatRegistry Whitebox Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_registry_default_same_as_new() {
        let registry1 = VertexFormatRegistry::new();
        let registry2 = VertexFormatRegistry::default();
        assert_eq!(registry1.len(), registry2.len());
    }

    #[test]
    fn test_registry_duplicate_registration_updates() {
        let mut registry = VertexFormatRegistry::new();
        let original = registry.get(VertexFormatId::Terrain).unwrap().stride;
        assert_eq!(original, 32);

        let modified = VertexFormat {
            id: VertexFormatId::Terrain,
            name: "ModifiedTerrain",
            stride: 64,
            attributes: vec![],
        };
        registry.register(modified);

        // Should still have 5 formats (not 6)
        assert_eq!(registry.len(), 5);

        // Should have updated stride
        let retrieved = registry.get(VertexFormatId::Terrain).unwrap();
        assert_eq!(retrieved.stride, 64);
        assert_eq!(retrieved.name, "ModifiedTerrain");
    }

    #[test]
    fn test_registry_get_buffer_layout_none() {
        let registry = VertexFormatRegistry::new();
        let layout = registry.get_buffer_layout(VertexFormatId::Custom(999));
        assert!(layout.is_none());
    }

    #[test]
    fn test_registry_get_buffer_layout_for_all_standard() {
        let registry = VertexFormatRegistry::new();

        // StaticMesh
        let layout = registry.get_buffer_layout(VertexFormatId::StaticMesh);
        assert!(layout.is_some());
        assert_eq!(layout.unwrap().array_stride, 48);

        // SkinnedMesh
        let layout = registry.get_buffer_layout(VertexFormatId::SkinnedMesh);
        assert!(layout.is_some());
        assert_eq!(layout.unwrap().array_stride, 72);

        // Terrain
        let layout = registry.get_buffer_layout(VertexFormatId::Terrain);
        assert!(layout.is_some());
        assert_eq!(layout.unwrap().array_stride, 32);

        // Particle
        let layout = registry.get_buffer_layout(VertexFormatId::Particle);
        assert!(layout.is_some());
        assert_eq!(layout.unwrap().array_stride, 36);

        // UI
        let layout = registry.get_buffer_layout(VertexFormatId::Ui);
        assert!(layout.is_some());
        assert_eq!(layout.unwrap().array_stride, 20);
    }

    #[test]
    fn test_registry_len_after_custom_adds() {
        let mut registry = VertexFormatRegistry::new();
        assert_eq!(registry.len(), 5);

        registry.register(VertexFormat {
            id: VertexFormatId::Custom(1),
            name: "Custom1",
            stride: 16,
            attributes: vec![],
        });
        assert_eq!(registry.len(), 6);

        registry.register(VertexFormat {
            id: VertexFormatId::Custom(2),
            name: "Custom2",
            stride: 32,
            attributes: vec![],
        });
        assert_eq!(registry.len(), 7);
    }

    #[test]
    fn test_registry_is_empty_never_true_after_new() {
        let registry = VertexFormatRegistry::new();
        assert!(!registry.is_empty());
    }

    #[test]
    fn test_registry_contains_all_standard_formats() {
        let registry = VertexFormatRegistry::new();
        assert!(registry.contains(VertexFormatId::StaticMesh));
        assert!(registry.contains(VertexFormatId::SkinnedMesh));
        assert!(registry.contains(VertexFormatId::Terrain));
        assert!(registry.contains(VertexFormatId::Particle));
        assert!(registry.contains(VertexFormatId::Ui));
    }

    #[test]
    fn test_registry_contains_not_custom() {
        let registry = VertexFormatRegistry::new();
        assert!(!registry.contains(VertexFormatId::Custom(0)));
        assert!(!registry.contains(VertexFormatId::Custom(1)));
        assert!(!registry.contains(VertexFormatId::Custom(u32::MAX)));
    }

    #[test]
    fn test_registry_iter_contains_all_standard() {
        let registry = VertexFormatRegistry::new();
        let names: Vec<&str> = registry.iter().map(|f| f.name).collect();

        assert!(names.contains(&"StaticMesh"));
        assert!(names.contains(&"SkinnedMesh"));
        assert!(names.contains(&"Terrain"));
        assert!(names.contains(&"Particle"));
        assert!(names.contains(&"UI"));
    }

    #[test]
    fn test_registry_iter_count_matches_len() {
        let registry = VertexFormatRegistry::new();
        assert_eq!(registry.iter().count(), registry.len());
    }

    // -------------------------------------------------------------------------
    // Edge Cases and Attribute Offset/Stride Validation
    // -------------------------------------------------------------------------

    #[test]
    fn test_static_mesh_stride_matches_attribute_sum() {
        // Float32x3(12) + Float32x3(12) + Float32x4(16) + Float32x2(8) = 48
        let format = static_mesh();
        let expected_stride = 12 + 12 + 16 + 8;
        assert_eq!(format.stride, expected_stride);
    }

    #[test]
    fn test_skinned_mesh_stride_matches_attribute_sum() {
        // Float32x3(12) + Float32x3(12) + Float32x4(16) + Float32x2(8) + Uint16x4(8) + Float32x4(16) = 72
        let format = skinned_mesh();
        let expected_stride = 12 + 12 + 16 + 8 + 8 + 16;
        assert_eq!(format.stride, expected_stride);
    }

    #[test]
    fn test_terrain_stride_matches_attribute_sum() {
        // Float32x3(12) + Float32x3(12) + Float32x2(8) = 32
        let format = terrain();
        let expected_stride = 12 + 12 + 8;
        assert_eq!(format.stride, expected_stride);
    }

    #[test]
    fn test_particle_stride_matches_attribute_sum() {
        // Float32x3(12) + Float32x4(16) + Float32x2(8) = 36
        let format = particle();
        let expected_stride = 12 + 16 + 8;
        assert_eq!(format.stride, expected_stride);
    }

    #[test]
    fn test_ui_stride_matches_attribute_sum() {
        // Float32x2(8) + Float32x2(8) + Unorm8x4(4) = 20
        let format = ui();
        let expected_stride = 8 + 8 + 4;
        assert_eq!(format.stride, expected_stride);
    }

    #[test]
    fn test_terrain_attribute_offsets() {
        let format = terrain();
        assert_eq!(format.attributes[0].offset, 0);   // position
        assert_eq!(format.attributes[1].offset, 12);  // normal (0 + 12)
        assert_eq!(format.attributes[2].offset, 24);  // uv (12 + 12)
    }

    #[test]
    fn test_particle_attribute_offsets() {
        let format = particle();
        assert_eq!(format.attributes[0].offset, 0);   // position
        assert_eq!(format.attributes[1].offset, 12);  // color (0 + 12)
        assert_eq!(format.attributes[2].offset, 28);  // size_rotation (12 + 16)
    }

    #[test]
    fn test_ui_attribute_offsets() {
        let format = ui();
        assert_eq!(format.attributes[0].offset, 0);   // position
        assert_eq!(format.attributes[1].offset, 8);   // uv (0 + 8)
        assert_eq!(format.attributes[2].offset, 16);  // color (8 + 8)
    }

    #[test]
    fn test_particle_is_compact_format() {
        // Particle format is designed for efficiency with just 3 attributes
        let format = particle();
        assert_eq!(format.attributes.len(), 3);
        assert_eq!(format.stride, 36); // Relatively compact
    }

    #[test]
    fn test_ui_is_compact_format() {
        // UI format is the most compact at 20 bytes
        let format = ui();
        assert_eq!(format.attributes.len(), 3);
        assert_eq!(format.stride, 20); // Smallest stride

        // Uses Unorm8x4 for colors (4 bytes instead of 16 for Float32x4)
        assert_eq!(format.attributes[2].format, wgpu::VertexFormat::Unorm8x4);
    }

    #[test]
    fn test_vertex_format_clone() {
        let format = static_mesh();
        let cloned = format.clone();
        assert_eq!(cloned.id, format.id);
        assert_eq!(cloned.name, format.name);
        assert_eq!(cloned.stride, format.stride);
        assert_eq!(cloned.attributes.len(), format.attributes.len());
    }

    #[test]
    fn test_vertex_format_debug() {
        let format = terrain();
        let debug_str = format!("{:?}", format);
        assert!(debug_str.contains("Terrain"));
        assert!(debug_str.contains("32")); // stride
    }

    #[test]
    fn test_registry_debug() {
        let registry = VertexFormatRegistry::new();
        let debug_str = format!("{:?}", registry);
        assert!(debug_str.contains("VertexFormatRegistry"));
    }

    #[test]
    fn test_multiple_custom_formats() {
        let mut registry = VertexFormatRegistry::new();

        for i in 0..10 {
            registry.register(VertexFormat {
                id: VertexFormatId::Custom(i),
                name: "Custom",
                stride: (i * 16) as u64,
                attributes: vec![],
            });
        }

        assert_eq!(registry.len(), 15); // 5 standard + 10 custom

        for i in 0..10 {
            assert!(registry.contains(VertexFormatId::Custom(i)));
            let format = registry.get(VertexFormatId::Custom(i)).unwrap();
            assert_eq!(format.stride, (i * 16) as u64);
        }
    }

    #[test]
    fn test_get_returns_none_for_unregistered() {
        let registry = VertexFormatRegistry::new();
        assert!(registry.get(VertexFormatId::Custom(12345)).is_none());
    }

    #[test]
    fn test_custom_format_with_attributes() {
        let mut registry = VertexFormatRegistry::new();

        let custom = VertexFormat {
            id: VertexFormatId::Custom(100),
            name: "CustomWithAttrs",
            stride: 28,
            attributes: vec![
                VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x3, 0, 0),
                VertexAttributeDescriptor::new(wgpu::VertexFormat::Float32x4, 12, 1),
            ],
        };
        registry.register(custom);

        let layout = registry.get_buffer_layout(VertexFormatId::Custom(100)).unwrap();
        assert_eq!(layout.array_stride, 28);
        assert_eq!(layout.attributes.len(), 2);
        assert_eq!(layout.attributes[0].offset, 0);
        assert_eq!(layout.attributes[1].offset, 12);
    }
}
