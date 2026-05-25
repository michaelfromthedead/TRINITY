// SPDX-License-Identifier: MIT
//
// blackbox_type_registry_v2.rs -- Blackbox contract tests for TypeRegistry v2.
//
// T-BRG-1.1: TypeRegistry now has ArchetypeId, flags/archetype_id on
// ComponentTypeInfo, and a parking_lot::RwLock interior.  These tests validate
// the public API from outside the crate, using only re-exported items.

use renderer_backend::type_registry::{
    ArchetypeId, ComponentTypeInfo, FieldLayout, TypeRegistry,
};

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

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

// ===========================================================================
// Test 1 -- TypeRegistry::new() produces an empty registry
// ===========================================================================

#[test]
fn new_registry_is_empty() {
    let registry = TypeRegistry::new();
    assert_eq!(registry.len(), 0, "fresh registry must report len = 0");
    assert!(registry.is_empty(), "fresh registry must report is_empty = true");
}

// ===========================================================================
// Test 2 -- register() + get() round-trip preserves all fields
// ===========================================================================

#[test]
fn register_and_get_roundtrip_preserves_all_fields() {
    let registry = TypeRegistry::new();

    let info = ComponentTypeInfo {
        id: 42,
        name: "Transform".to_string(),
        size: 64,
        fields: vec![
            FieldLayout {
                name: "tx".to_string(),
                type_code: "F32".to_string(),
                offset: 0,
            },
            FieldLayout {
                name: "ty".to_string(),
                type_code: "F32".to_string(),
                offset: 4,
            },
        ],
        flags: 0b1010,
        archetype_id: Some(ArchetypeId::from_component_ids(&[1, 2])),
    };

    registry.register(info);

    let retrieved = registry.get(42).expect("should retrieve registered id");
    assert_eq!(retrieved.id, 42);
    assert_eq!(retrieved.name, "Transform");
    assert_eq!(retrieved.size, 64);
    assert_eq!(retrieved.fields.len(), 2);
    assert_eq!(retrieved.fields[0].name, "tx");
    assert_eq!(retrieved.fields[1].name, "ty");
    assert_eq!(retrieved.flags, 0b1010, "flags must survive round-trip");
    assert_eq!(
        retrieved.archetype_id,
        Some(ArchetypeId::from_component_ids(&[1, 2])),
        "archetype_id must survive round-trip"
    );
}

// ===========================================================================
// Test 3 -- get() on an unregistered id returns None
// ===========================================================================

#[test]
fn get_unregistered_returns_none() {
    let registry = TypeRegistry::new();
    registry.register(make_type(1, "Position", 12));

    assert!(registry.get(1).is_some(), "registered id must be found");
    assert!(registry.get(999).is_none(), "unregistered id must return None");
    assert!(registry.get(0).is_none(), "id 0 is unregistered, must be None");
}

// ===========================================================================
// Test 4 -- contains() works for registered and unregistered IDs
// ===========================================================================

#[test]
fn contains_registered_and_unregistered() {
    let registry = TypeRegistry::new();
    registry.register(make_type(10, "Health", 4));
    registry.register(make_type(20, "Mana", 4));

    assert!(registry.contains(10), "contains must return true for registered id");
    assert!(registry.contains(20), "contains must return true for registered id");
    assert!(
        !registry.contains(30),
        "contains must return false for unregistered id"
    );
    assert!(
        !registry.contains(0),
        "contains must return false for never-registered id 0"
    );
}

// ===========================================================================
// Test 5 -- ids() returns all registered IDs
// ===========================================================================

#[test]
fn ids_returns_all_registered_ids() {
    let registry = TypeRegistry::new();
    registry.register(make_type(5, "A", 1));
    registry.register(make_type(3, "B", 2));
    registry.register(make_type(7, "C", 3));

    let mut ids = registry.ids();
    ids.sort();
    assert_eq!(ids, vec![3, 5, 7], "ids() must contain every registered id");

    // An empty registry must return an empty vector.
    let empty = TypeRegistry::new();
    assert!(
        empty.ids().is_empty(),
        "ids() on empty registry must return empty vec"
    );
}

// ===========================================================================
// Test 6 -- type_list() returns (id, name, size) tuples for all types
// ===========================================================================

#[test]
fn type_list_returns_all_registered_types() {
    let registry = TypeRegistry::new();
    registry.register(make_type(1, "Position", 12));
    registry.register(make_type(2, "Velocity", 12));
    registry.register(make_type(3, "Health", 4));

    let mut list = registry.type_list();
    // Sort by name for a stable assertion.
    list.sort_by(|a, b| a.1.cmp(&b.1));

    assert_eq!(list.len(), 3);
    assert_eq!(list[0], (3, "Health".to_string(), 4));
    assert_eq!(list[1], (1, "Position".to_string(), 12));
    assert_eq!(list[2], (2, "Velocity".to_string(), 12));
}

// ===========================================================================
// Test 7 -- archetype_for() is deterministic (same input, same result)
// ===========================================================================

#[test]
fn archetype_for_is_deterministic() {
    let registry = TypeRegistry::new();

    let first = registry.archetype_for(&[1, 2, 3]);
    let second = registry.archetype_for(&[1, 2, 3]);
    assert_eq!(
        first, second,
        "calling archetype_for twice with the same set must produce the same ArchetypeId"
    );

    // Also verify that order does not matter.
    let reversed = registry.archetype_for(&[3, 2, 1]);
    assert_eq!(
        first, reversed,
        "archetype_for must be order-independent"
    );

    // Single-element sets.
    let a = registry.archetype_for(&[99]);
    let b = registry.archetype_for(&[99]);
    assert_eq!(a, b, "single-element set must be deterministic");
}

// ===========================================================================
// Test 8 -- archetype_for() with different sets returns different IDs
// ===========================================================================

#[test]
fn archetype_for_different_sets_different_ids() {
    let registry = TypeRegistry::new();

    let id_a = registry.archetype_for(&[1, 2, 3]);
    let id_b = registry.archetype_for(&[4, 5, 6]);
    assert_ne!(
        id_a, id_b,
        "different component sets should produce different ArchetypeIds"
    );

    // Overlapping but not identical.
    let id_c = registry.archetype_for(&[1, 2, 4]);
    assert_ne!(id_a, id_c, "overlapping-but-different sets must differ");
    assert_ne!(id_b, id_c, "overlapping-but-different sets must differ");

    // Empty set vs. non-empty set.
    let id_empty = registry.archetype_for(&[]);
    let id_nonempty = registry.archetype_for(&[1]);
    assert_ne!(
        id_empty, id_nonempty,
        "empty and non-empty sets must produce different ArchetypeIds"
    );
}

// ===========================================================================
// Test 9 -- ArchetypeId can be converted to u32
// ===========================================================================

#[test]
fn archetype_id_converts_to_u32() {
    let id = ArchetypeId::from_component_ids(&[42]);
    let raw: u32 = id.into();

    // The hash is deterministic.  Just check round-trip compiles.
    assert_eq!(
        raw,
        u32::from(ArchetypeId::from_component_ids(&[42])),
        "ArchetypeId -> u32 conversion must be deterministic"
    );

    // From<ArchetypeId> for u32 via Into trait.
    let id2 = ArchetypeId::from_component_ids(&[1, 2, 3]);
    let raw2: u32 = id2.into();
    assert_ne!(raw2, 0, "non-empty component set should not hash to zero");

    // Explicit from() call.
    let raw3 = u32::from(ArchetypeId::from_component_ids(&[10]));
    assert_eq!(raw3, u32::from(ArchetypeId::from_component_ids(&[10])));
}
