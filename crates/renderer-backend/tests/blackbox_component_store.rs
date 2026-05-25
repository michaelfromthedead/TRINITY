// SPDX-License-Identifier: MIT
//
// blackbox_component_store.rs -- Blackbox contract tests for ComponentStore.
//
// T-BRG-2.1: ComponentStore exposes Archetype, ComponentStore, and the global
// singleton (initialize_component_store / global_component_store).  These tests
// validate the public API from outside the crate, using only re-exported items.
//
// CLEANROOM: No implementation source files were read during authoring.

use renderer_backend::component_store::{
ComponentStore, initialize_component_store, global_component_store,
};
use renderer_backend::type_registry::{ArchetypeId, ComponentTypeInfo, TypeRegistry};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a registry with four test component types:
///   id=1  "Position"  12 bytes (3 × f32)
///   id=2  "Velocity"   8 bytes (2 × f32)
///   id=3  "Health"     4 bytes (1 × f32)
///   id=4  "Color"     16 bytes (4 × f32)
fn make_registry() -> Arc<TypeRegistry> {
    let registry = TypeRegistry::new();
    for (id, name, size) in &[(1, "Position", 12), (2, "Velocity", 8), (3, "Health", 4), (4, "Color", 16)] {
        registry.register(ComponentTypeInfo {
            id: *id,
            name: name.to_string(),
            size: *size,
            fields: vec![],
            flags: 0,
            archetype_id: None,
        });
    }
    Arc::new(registry)
}

// ===========================================================================
// Test 1 -- ComponentStore::new creates an empty store
// ===========================================================================

#[test]
fn new_store_is_empty() {
    let registry = make_registry();
    let store = ComponentStore::new(registry);

    assert_eq!(
        store.entity_count(),
        0,
        "fresh store must report entity_count = 0"
    );
    assert_eq!(
        store.archetype_count(),
        0,
        "fresh store must report archetype_count = 0"
    );
    assert!(
        store.archetypes.is_empty(),
        "archetypes map must be empty"
    );
    assert!(
        store.entity_index.is_empty(),
        "entity_index must be empty"
    );

    // Query for any component set must return nothing.
    let result = store.query(&[1]);
    assert!(result.is_empty(), "query on empty store must return empty vec");

    let result = store.query(&[1, 2, 3]);
    assert!(result.is_empty(), "query on empty store must return empty vec");
}

// ===========================================================================
// Test 2 -- Spawn a single entity, verify query and read_field
// ===========================================================================

#[test]
fn spawn_single_entity_query_and_read_back() {
    let registry = make_registry();
    let mut store = ComponentStore::new(registry);

    let pos_data: Vec<u8> = (0..12).collect();
    store.spawn(100, &[1], &[(1, pos_data.clone())]);

    // Query for component 1 must return the entity.
    let results = store.query(&[1]);
    assert_eq!(results.len(), 1, "query for [1] must return exactly one entity");
    assert_eq!(results[0], 100, "query must return entity id 100");

    // read_field must return the written data.
    let read_back = store
        .read_field(100, 1, 0, 12)
        .expect("read_field for existing entity must return Some");
    assert_eq!(
        read_back, pos_data,
        "read_field must return the data passed to spawn"
    );

    // Singleton archetype.
    assert_eq!(store.archetype_count(), 1);
    assert_eq!(store.entity_count(), 1);
}

// ===========================================================================
// Test 3 -- Spawn two entities with the same components, query returns both
// ===========================================================================

#[test]
fn spawn_two_entities_same_archetype_query_returns_both() {
    let registry = make_registry();
    let mut store = ComponentStore::new(registry);

    store.spawn(100, &[1, 2], &[(1, vec![0xAA; 12]), (2, vec![0xBB; 8])]);
    store.spawn(200, &[1, 2], &[(1, vec![0xCC; 12]), (2, vec![0xDD; 8])]);

    // Only one archetype should exist.
    assert_eq!(
        store.archetype_count(),
        1,
        "two entities with same components share one archetype"
    );
    assert_eq!(store.entity_count(), 2);

    let results = store.query(&[1, 2]);
    assert_eq!(results.len(), 2, "query for [1, 2] must return both entities");
    assert!(results.contains(&100), "results must contain entity 100");
    assert!(results.contains(&200), "results must contain entity 200");

    // Each entity's data is independent.
    assert_eq!(
        store.read_field(100, 1, 0, 12),
        Some(vec![0xAA; 12]),
    );
    assert_eq!(
        store.read_field(200, 1, 0, 12),
        Some(vec![0xCC; 12]),
    );
}

// ===========================================================================
// Test 4 -- Despawn removes entity from query and read_field returns None
// ===========================================================================

#[test]
fn despawn_removes_entity_from_store() {
    let registry = make_registry();
    let mut store = ComponentStore::new(registry);

    store.spawn(100, &[1], &[(1, vec![0; 12])]);
    store.spawn(200, &[1], &[(1, vec![0; 12])]);

    assert_eq!(store.entity_count(), 2);
    assert_eq!(store.query(&[1]).len(), 2);

    store.despawn(100);

    // Entity 100 should be gone.
    assert_eq!(store.entity_count(), 1, "only one entity remains after despawn");
    assert!(
        store.read_field(100, 1, 0, 1).is_none(),
        "read_field for despawned entity must return None"
    );

    let results = store.query(&[1]);
    assert_eq!(results.len(), 1, "query after despawn must return one entity");
    assert_eq!(results[0], 200, "remaining entity must be 200");

    // Entity 200 is still readable.
    assert!(
        store.read_field(200, 1, 0, 1).is_some(),
        "read_field for surviving entity must still return data"
    );
}

// ===========================================================================
// Test 5 -- Despawn twice on same entity is idempotent (no panic)
// ===========================================================================

#[test]
fn despawn_twice_is_idempotent() {
    let registry = make_registry();
    let mut store = ComponentStore::new(registry);

    store.spawn(100, &[1], &[(1, vec![0; 12])]);
    store.spawn(200, &[1], &[(1, vec![0; 12])]);

    // Despawn once.
    store.despawn(100);
    assert_eq!(store.entity_count(), 1);

    // Despawn again -- must not panic and must not change state.
    store.despawn(100);
    assert_eq!(
        store.entity_count(),
        1,
        "idempotent despawn must not change entity_count"
    );

    // Despawn a never-spawned entity -- also must not panic.
    store.despawn(999);
    assert_eq!(store.entity_count(), 1, "despawning never-spawned entity is a no-op");

    // Despawn the remaining entity.
    store.despawn(200);
    assert_eq!(
        store.entity_count(),
        0,
        "all entities removed"
    );

    // Despawn again on now-empty store.
    store.despawn(200);
    assert_eq!(store.entity_count(), 0, "still empty after redundant despawn");
}

// ===========================================================================
// Test 6 -- write_field changes stored data, read_field returns new data
// ===========================================================================

#[test]
fn write_field_updates_stored_data() {
    let registry = make_registry();
    let mut store = ComponentStore::new(registry);

    // Spawn with zeroed-out Position.
    store.spawn(100, &[1], &[(1, vec![0; 12])]);

    // Verify initial state.
    assert_eq!(
        store.read_field(100, 1, 0, 12),
        Some(vec![0; 12]),
        "initial data must be zero-filled"
    );

    // Write full field.
    let new_data: Vec<u8> = (0x10..0x1C).collect();
    store.write_field(100, 1, 0, &new_data);

    let read_back = store
        .read_field(100, 1, 0, 12)
        .expect("read_field must succeed after write_field");
    assert_eq!(
        read_back, new_data,
        "write_field must persist the data"
    );

    // Partial overwrite at offset 4 (e.g. the "y" component of a position).
    let partial = vec![0xFF, 0xFE, 0xFD, 0xFC];
    store.write_field(100, 1, 4, &partial);

    let full = store.read_field(100, 1, 0, 12).unwrap();
    assert_eq!(
        &full[0..4],
        &new_data[0..4],
        "bytes before offset must be unchanged"
    );
    assert_eq!(
        &full[4..8],
        &partial[..],
        "bytes at offset must be updated"
    );
    assert_eq!(
        &full[8..12],
        &new_data[8..12],
        "bytes after the write range must be unchanged"
    );
}

// ===========================================================================
// Test 7 -- Query cross-archetype: entities with overlapping components
// ===========================================================================

#[test]
fn query_cross_archetype_returns_matching_entities() {
    let registry = make_registry();
    let mut store = ComponentStore::new(registry);

    // Entity A has [Position, Velocity]   = arch {1, 2}
    // Entity B has [Velocity, Health]     = arch {2, 3}
    // Entity C has [Position, Health]     = arch {1, 3}
    store.spawn(100, &[1, 2], &[(1, vec![0; 12]), (2, vec![0; 8])]);
    store.spawn(200, &[2, 3], &[(2, vec![0; 8]), (3, vec![0; 4])]);
    store.spawn(300, &[1, 3], &[(1, vec![0; 12]), (3, vec![0; 4])]);

    // Three different archetypes.
    assert_eq!(store.archetype_count(), 3);

    // Query for [Velocity] (component 2) must return A and B (but not C).
    let with_velocity = store.query(&[2]);
    assert_eq!(with_velocity.len(), 2);
    assert!(with_velocity.contains(&100));
    assert!(with_velocity.contains(&200));
    assert!(!with_velocity.contains(&300));

    // Query for [Position] (component 1) must return A and C.
    let with_position = store.query(&[1]);
    assert_eq!(with_position.len(), 2);
    assert!(with_position.contains(&100));
    assert!(with_position.contains(&300));
    assert!(!with_position.contains(&200));

    // Query for [Health] (component 3) must return B and C.
    let with_health = store.query(&[3]);
    assert_eq!(with_health.len(), 2);
    assert!(with_health.contains(&200));
    assert!(with_health.contains(&300));
    assert!(!with_health.contains(&100));
}

// ===========================================================================
// Test 8 -- Query for components an entity doesn't have returns empty
// ===========================================================================

#[test]
fn query_components_entity_does_not_have_returns_empty() {
    let registry = make_registry();
    let mut store = ComponentStore::new(registry);

    // Entity A has only [Position] (component 1).
    store.spawn(100, &[1], &[(1, vec![0; 12])]);

    // Query for a strict superset [1, 2] -- entity has 1 but not 2.
    let result = store.query(&[1, 2]);
    assert!(
        result.is_empty(),
        "entity with only [1] must not match query for [1, 2]"
    );

    // Query for a completely unrelated component.
    let result = store.query(&[4]);
    assert!(
        result.is_empty(),
        "entity with [1] must not match query for [4]"
    );

    // Entity A should still be found by query for [1] alone.
    let result = store.query(&[1]);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], 100);

    // Add entity B with [1, 2].
    store.spawn(200, &[1, 2], &[(1, vec![0; 12]), (2, vec![0; 8])]);

    // Query for [1, 2] now returns entity B, but still not A.
    let result = store.query(&[1, 2]);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], 200);
}

// ===========================================================================
// Test 9 -- Global singleton: initialize then access returns the same store
// ===========================================================================

#[test]
fn global_singleton_returns_initialized_store() {
    let registry = make_registry();

    // Initialise the global singleton.
    initialize_component_store(registry);

    // Access the global store.
    let store_arc = global_component_store();
    let store = store_arc.read();

    // The store must be valid, with zero entities.
    assert_eq!(
        store.entity_count(),
        0,
        "global store must start empty"
    );
    assert_eq!(
        store.archetype_count(),
        0,
        "global store must have zero archetypes"
    );
    assert!(
        store.query(&[1]).is_empty(),
        "global store query must return empty"
    );
    drop(store);

    // Verify we can mutate through the singleton.
    {
        let mut store = store_arc.write();
        store.spawn(100, &[1], &[(1, vec![0x42; 12])]);
    }

    // Read back through the singleton.
    let store = store_arc.read();
    assert_eq!(store.entity_count(), 1);
    assert_eq!(
        store.read_field(100, 1, 0, 12),
        Some(vec![0x42; 12]),
        "global singleton must persist mutations"
    );
}

// ===========================================================================
// Test 10 -- Alive count matches expectation after spawn/despawn
// ===========================================================================

#[test]
fn alive_count_tracks_spawn_and_despawn() {
    let registry = make_registry();
    let mut store = ComponentStore::new(registry);

    // Archetype::alive_count should reflect free rows.
    store.spawn(100, &[1], &[(1, vec![0; 12])]);
    store.spawn(200, &[1], &[(1, vec![0; 12])]);
    store.spawn(300, &[1], &[(1, vec![0; 12])]);

    {
        let arch_id = ArchetypeId::from_component_ids(&[1]);
        let arch = store.archetypes.get(&arch_id).unwrap();
        assert_eq!(arch.alive_count(), 3, "three alive entities");
        assert_eq!(arch.entities.len(), 3);
        assert!(arch.free_rows.is_empty());
    }

    store.despawn(200);

    {
        let arch_id = ArchetypeId::from_component_ids(&[1]);
        let arch = store.archetypes.get(&arch_id).unwrap();
        assert_eq!(arch.alive_count(), 2, "two alive after despawn");
        assert_eq!(arch.free_rows.len(), 1, "one free row after despawn");
    }

    // Re-spawn should reuse the free row.
    store.spawn(400, &[1], &[(1, vec![0xFF; 12])]);

    {
        let arch_id = ArchetypeId::from_component_ids(&[1]);
        let arch = store.archetypes.get(&arch_id).unwrap();
        assert_eq!(arch.alive_count(), 3, "back to three alive after re-spawn");
        assert!(arch.free_rows.is_empty(), "free list consumed by re-spawn");
    }
}

// ===========================================================================
// Test 11 -- column_slice returns raw bytes for a component column
// ===========================================================================

#[test]
fn column_slice_returns_raw_column_bytes() {
    let registry = make_registry();
    let mut store = ComponentStore::new(registry);

    store.spawn(100, &[1], &[(1, (0..12).collect())]);
    store.spawn(200, &[1], &[(1, (12..24).collect())]);

    let arch_id = ArchetypeId::from_component_ids(&[1]);
    let slice = store
        .column_slice(arch_id, 0)
        .expect("column_slice for existing archetype must return Some");

    // Two rows of 12-byte components = 24 bytes.
    assert_eq!(slice.len(), 24);
    assert_eq!(&slice[0..12], &(0..12).collect::<Vec<u8>>()[..]);
    assert_eq!(&slice[12..24], &(12..24).collect::<Vec<u8>>()[..]);

    // Nonexistent column index returns None.
    assert!(store.column_slice(arch_id, 99).is_none());

    // Nonexistent archetype returns None.
    let bogus_id = ArchetypeId::from_component_ids(&[99]);
    assert!(store.column_slice(bogus_id, 0).is_none());
}
