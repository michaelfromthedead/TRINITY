use parking_lot::RwLock;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct FieldLayout {
    pub name: String,
    pub type_code: String,
    pub offset: usize,
}


/// A deterministic archetype identifier derived from a set of component type IDs.
///
/// The same set of component IDs (regardless of order) always produces the same
/// `ArchetypeId`.  Different sets produce (with high probability) different IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArchetypeId(u32);

impl ArchetypeId {
    /// Derive an archetype ID from a set of component type IDs.
    ///
    /// Sorts the IDs first, then hashes them with `DefaultHasher` so that the
    /// result is order-independent and deterministic.
    pub fn from_component_ids(ids: &[u32]) -> Self {
        let mut sorted: Vec<u32> = ids.to_vec();
        sorted.sort();
        let mut hasher = DefaultHasher::new();
        for id in &sorted {
            id.hash(&mut hasher);
        }
        let hash = hasher.finish();
        // XOR-fold the 64-bit hash down to 32 bits for better distribution
        // than a simple truncation.
        ArchetypeId(((hash >> 32) as u32) ^ (hash as u32))
    }
}

impl From<ArchetypeId> for u32 {
    fn from(id: ArchetypeId) -> Self {
        id.0
    }
}

/// Metadata describing a single component type in the ECS.
#[derive(Debug, Clone)]
pub struct ComponentTypeInfo {
    pub id: u32,
    pub name: String,
    pub size: usize,
    pub fields: Vec<FieldLayout>,
    /// Bitfield for component properties (reserved, zeroed for now).
    pub flags: u32,
    /// The archetype this component is assigned to, if any.
    pub archetype_id: Option<ArchetypeId>,
}

/// A thread-safe registry of component types.
///
/// All access is mediated through a `parking_lot::RwLock` so that readers never
/// block each other and writers get exclusive access.
#[derive(Debug)]
pub struct TypeRegistry {
    types: RwLock<std::collections::HashMap<u32, ComponentTypeInfo>>,
}

impl TypeRegistry {
    pub fn new() -> Self {
        Self {
            types: RwLock::new(std::collections::HashMap::new()),
        }
    }

    /// Register (or overwrite) a component type.
    pub fn register(&self, info: ComponentTypeInfo) {
        self.types.write().insert(info.id, info);
    }

    /// Look up a component type by ID.  Returns a clone of the stored info.
    pub fn get(&self, id: u32) -> Option<ComponentTypeInfo> {
        self.types.read().get(&id).cloned()
    }

    /// Number of registered component types.
    pub fn len(&self) -> usize {
        self.types.read().len()
    }

    /// Whether the registry contains no component types.
    pub fn is_empty(&self) -> bool {
        self.types.read().is_empty()
    }

    /// Returns the IDs of all registered component types.
    pub fn ids(&self) -> Vec<u32> {
        self.types.read().keys().copied().collect()
    }

    /// Returns `true` if a component type with the given ID is registered.
    pub fn contains(&self, id: u32) -> bool {
        self.types.read().contains_key(&id)
    }

    /// Returns a debug-friendly list of `(id, name, size)` tuples for every
    /// registered component type.
    pub fn type_list(&self) -> Vec<(u32, String, usize)> {
        self.types
            .read()
            .values()
            .map(|info| (info.id, info.name.clone(), info.size))
            .collect()
    }

    /// Derive an `ArchetypeId` for the given set of component IDs.
    ///
    /// The result is deterministic: the same set (any order) always produces
    /// the same archetype ID.
    pub fn archetype_for(&self, component_ids: &[u32]) -> ArchetypeId {
        ArchetypeId::from_component_ids(component_ids)
    }

    /// Assign a registered component type to an archetype.
    ///
    /// Returns `None` if no component with the given `component_id` is
    /// registered.
    pub fn assign_to_archetype(
        &self,
        component_id: u32,
        archetype_id: ArchetypeId,
    ) -> Option<()> {
        let mut types = self.types.write();
        if let Some(info) = types.get_mut(&component_id) {
            info.archetype_id = Some(archetype_id);
            Some(())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_type(id: u32, name: &str, size: usize) -> ComponentTypeInfo {
        ComponentTypeInfo {
            id,
            name: name.to_string(),
            size,
            fields: vec![],
            flags: 0,
            archetype_id: None,
        }
    }

    #[test]
    fn test_new_registry_empty() {
        let registry = TypeRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_register_get_roundtrip() {
        let registry = TypeRegistry::new();
        registry.register(make_type(1, "Position", 12));

        let retrieved = registry.get(1).expect("should find id 1");
        assert_eq!(retrieved.id, 1);
        assert_eq!(retrieved.name, "Position");
        assert_eq!(retrieved.size, 12);
        assert_eq!(retrieved.flags, 0);
        assert!(retrieved.archetype_id.is_none());
    }

    #[test]
    fn test_get_missing_returns_none() {
        let registry = TypeRegistry::new();
        assert!(registry.get(999).is_none());
    }

    #[test]
    fn test_register_overwrites() {
        let registry = TypeRegistry::new();
        registry.register(make_type(1, "Original", 8));
        registry.register(ComponentTypeInfo {
            id: 1,
            name: "Overwritten".to_string(),
            size: 16,
            fields: vec![],
            flags: 0,
            archetype_id: None,
        });

        let retrieved = registry.get(1).expect("should find id 1 after overwrite");
        assert_eq!(retrieved.name, "Overwritten");
        assert_eq!(retrieved.size, 16);
    }

    #[test]
    fn test_type_list() {
        let registry = TypeRegistry::new();
        registry.register(make_type(1, "Position", 12));
        registry.register(make_type(2, "Velocity", 12));
        registry.register(make_type(3, "Health", 4));

        let list = registry.type_list();
        assert_eq!(list.len(), 3);

        let mut names: Vec<&str> = list.iter().map(|(_, n, _)| n.as_str()).collect();
        names.sort();
        assert_eq!(names, vec!["Health", "Position", "Velocity"]);
    }

    #[test]
    fn test_archetype_for_same_set() {
        let registry = TypeRegistry::new();
        let id_a = registry.archetype_for(&[1, 2, 3]);
        let id_b = registry.archetype_for(&[3, 2, 1]); // same set, reversed
        assert_eq!(id_a, id_b, "same set should produce same ArchetypeId");
    }

    #[test]
    fn test_archetype_for_different_sets() {
        let registry = TypeRegistry::new();
        let id_a = registry.archetype_for(&[1, 2, 3]);
        let id_b = registry.archetype_for(&[4, 5, 6]);
        assert_ne!(id_a, id_b, "different sets should produce different ArchetypeIds");
    }

    #[test]
    fn test_len_and_is_empty() {
        let registry = TypeRegistry::new();
        assert_eq!(registry.len(), 0);
        assert!(registry.is_empty());

        registry.register(make_type(1, "A", 4));
        assert_eq!(registry.len(), 1);
        assert!(!registry.is_empty());

        registry.register(make_type(2, "B", 8));
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn test_contains() {
        let registry = TypeRegistry::new();
        registry.register(make_type(10, "Transform", 64));
        assert!(registry.contains(10));
        assert!(!registry.contains(20));
    }

    #[test]
    fn test_ids_returns_registered_ids() {
        let registry = TypeRegistry::new();
        registry.register(make_type(5, "A", 1));
        registry.register(make_type(3, "B", 2));
        registry.register(make_type(7, "C", 3));

        let mut ids = registry.ids();
        ids.sort();
        assert_eq!(ids, vec![3, 5, 7]);
    }

    #[test]
    fn test_archetype_id_from_u32() {
        let id = ArchetypeId::from_component_ids(&[42]);
        let raw: u32 = id.into();
        // The raw value is deterministic — just verify the round-trip compiles
        // and produces a non-zero ID for a non-empty set.
        assert_ne!(raw, 0, "ArchetypeId for a non-empty set should not be zero");
    }

    // -- assign_to_archetype -------------------------------------------------

    #[test]
    fn test_assign_to_archetype_updates_component() {
        let registry = TypeRegistry::new();
        registry.register(make_type(1, "Position", 12));

        let arch_id = ArchetypeId::from_component_ids(&[1, 2]);
        let result = registry.assign_to_archetype(1, arch_id);
        assert!(result.is_some(), "should succeed for registered component");

        let info = registry.get(1).unwrap();
        assert_eq!(info.archetype_id, Some(arch_id));
    }

    #[test]
    fn test_assign_to_archetype_missing_component_returns_none() {
        let registry = TypeRegistry::new();
        let arch_id = ArchetypeId::from_component_ids(&[99]);
        let result = registry.assign_to_archetype(99, arch_id);
        assert!(result.is_none(), "unregistered component should return None");
    }

    #[test]
    fn test_assign_to_archetype_overwrites_previous() {
        let registry = TypeRegistry::new();
        registry.register(make_type(1, "Position", 12));

        let arch_a = ArchetypeId::from_component_ids(&[1]);
        let arch_b = ArchetypeId::from_component_ids(&[1, 2]);

        registry.assign_to_archetype(1, arch_a).unwrap();
        registry.assign_to_archetype(1, arch_b).unwrap();

        let info = registry.get(1).unwrap();
        assert_eq!(info.archetype_id, Some(arch_b), "should use latest assignment");
    }
}
