//! White-box tests for [`ComponentStore`].
//!
//! These tests exercise concurrency, free-list internals, empty-archetype
//! edge cases, order-independent archetype deduplication, partial-match
//! query semantics, and column-slice consistency after writes -- all of
//! which require direct access to internal fields (`archetypes`,
//! `free_rows`, `columns`, etc.) that black-box tests cannot reach.

use parking_lot::RwLock;
use renderer_backend::component_store::ComponentStore;
use renderer_backend::type_registry::{ArchetypeId, ComponentTypeInfo, TypeRegistry};
use std::sync::{Arc, Barrier};
use std::thread;

// ── Helpers ───────────────────────────────────────────────────────────────

fn make_registry() -> Arc<TypeRegistry> {
    let registry = TypeRegistry::new();
    registry.register(ComponentTypeInfo {
        id: 1,
        name: "Position".into(),
        size: 12,
        fields: vec![],
        flags: 0,
        archetype_id: None,
    });
    registry.register(ComponentTypeInfo {
        id: 2,
        name: "Velocity".into(),
        size: 8,
        fields: vec![],
        flags: 0,
        archetype_id: None,
    });
    registry.register(ComponentTypeInfo {
        id: 3,
        name: "Health".into(),
        size: 4,
        fields: vec![],
        flags: 0,
        archetype_id: None,
    });
    registry.register(ComponentTypeInfo {
        id: 4,
        name: "Color".into(),
        size: 16,
        fields: vec![],
        flags: 0,
        archetype_id: None,
    });
    Arc::new(registry)
}

// ── Test 1: Concurrent spawn ──────────────────────────────────────────────

#[test]
fn test_whitebox_concurrent_spawn() {
    let registry = make_registry();
    let store = Arc::new(RwLock::new(ComponentStore::new(registry)));
    let barrier = Arc::new(Barrier::new(4));
    let mut handles = Vec::with_capacity(4);

    for i in 0..4 {
        let store = Arc::clone(&store);
        let barrier = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            barrier.wait();
            let entity_id = 100 + i as u64;
            let data = vec![i as u8; 12];
            store.write().spawn(entity_id, &[1], &[(1, data)]);
        }));
    }

    for h in handles {
        h.join().expect("thread panicked");
    }

    // Verify all four entities exist through the store.
    let guard = store.read();
    assert_eq!(guard.entity_count(), 4, "all 4 entities should exist");

    for i in 0..4 {
        let entity_id = 100 + i as u64;
        let data = guard
            .read_field(entity_id, 1, 0, 12)
            .unwrap_or_else(|| panic!("entity {} should exist after concurrent spawn", entity_id));
        assert_eq!(
            data,
            vec![i as u8; 12],
            "entity {} data should match its thread's value",
            entity_id,
        );
    }

    // Query by component 1 should return all four.
    let results = guard.query(&[1]);
    assert_eq!(results.len(), 4, "query should return all 4 entities");
}

// ── Test 2: Concurrent read_field ─────────────────────────────────────────

#[test]
fn test_whitebox_concurrent_read_field() {
    let registry = make_registry();
    let store = Arc::new(RwLock::new(ComponentStore::new(registry)));

    // Prepare one entity with known data.
    let pos_data: Vec<u8> = (0..12).collect();
    store.write().spawn(100, &[1], &[(1, pos_data.clone())]);

    let barrier = Arc::new(Barrier::new(4));
    let mut handles = Vec::with_capacity(4);

    for _ in 0..4 {
        let store = Arc::clone(&store);
        let barrier = Arc::clone(&barrier);
        let expected = pos_data.clone();
        handles.push(thread::spawn(move || {
            barrier.wait();
            let guard = store.read();
            let result = guard
                .read_field(100, 1, 0, 12)
                .expect("concurrent read should succeed");
            assert_eq!(
                result, expected,
                "concurrent read returned incorrect data"
            );
        }));
    }

    for h in handles {
        h.join().expect("thread panicked");
    }
}

// ── Test 3: Concurrent write_field ────────────────────────────────────────

#[test]
fn test_whitebox_concurrent_write_field() {
    let registry = make_registry();
    let store = Arc::new(RwLock::new(ComponentStore::new(registry)));

    // Component 4 (Color) = 16 bytes, zero-initialised.
    store.write().spawn(100, &[4], &[(4, vec![0; 16])]);

    let barrier = Arc::new(Barrier::new(4));
    let mut handles = Vec::with_capacity(4);

    for i in 0..4 {
        let store = Arc::clone(&store);
        let barrier = Arc::clone(&barrier);
        let offset = i * 4;
        let data = vec![0x10 + i as u8; 4];
        handles.push(thread::spawn(move || {
            barrier.wait();
            store.write().write_field(100, 4, offset, &data);
        }));
    }

    for h in handles {
        h.join().expect("thread panicked");
    }

    // Verify each 4-byte field contains the data its thread wrote.
    let guard = store.read();
    for i in 0..4 {
        let field_data = guard
            .read_field(100, 4, i * 4, 4)
            .unwrap_or_else(|| panic!("should read field {} back", i));
        assert_eq!(
            field_data,
            vec![0x10 + i as u8; 4],
            "field {} should contain correct data after concurrent write",
            i,
        );
    }
}

// ── Test 4: Free list LIFO ordering ───────────────────────────────────────

#[test]
fn test_whitebox_free_list_lifo_ordering() {
    let registry = make_registry();
    let mut store = ComponentStore::new(registry);

    // Spawn 3 entities: rows 0, 1, 2.
    store.spawn(100, &[1], &[(1, vec![0; 12])]);
    store.spawn(200, &[1], &[(1, vec![0; 12])]);
    store.spawn(300, &[1], &[(1, vec![0; 12])]);

    // Despawn entity 100 (row 0), then entity 300 (row 2).
    store.despawn(100); // free_rows = [0]
    store.despawn(300); // free_rows = [0, 2]

    let arch_id = ArchetypeId::from_component_ids(&[1]);
    {
        let arch = store.archetypes.get(&arch_id).unwrap();
        assert_eq!(
            arch.free_rows,
            vec![0, 2],
            "after despawning rows 0 and 2, free_rows should be [0, 2] in that order"
        );
    }

    // Spawn entity 400: LIFO means row 2 (last freed) is reused first.
    store.spawn(400, &[1], &[(1, vec![0xDD; 12])]);

    let arch = store.archetypes.get(&arch_id).unwrap();
    assert_eq!(
        arch.entities[2], 400,
        "entity 400 should reuse row 2 (LIFO: row 2 was freed last)"
    );
    assert_eq!(
        arch.free_rows,
        vec![0],
        "free_rows should have only row 0 remaining after LIFO reuse"
    );
    assert_eq!(
        arch.entities.len(),
        3,
        "entities vec should not grow; reused existing slot"
    );

    // Data at row 2 must belong to entity 400, not stale entity 300 data.
    let data = store.read_field(400, 1, 0, 12).unwrap();
    assert_eq!(data, vec![0xDD; 12], "reused row should contain new entity's data");
}

// ── Test 5: Spawn with empty component_ids ────────────────────────────────

#[test]
fn test_whitebox_spawn_empty_component_ids() {
    let registry = make_registry();
    let mut store = ComponentStore::new(registry);

    // An entity with NO components is a valid (degenerate) case.
    store.spawn(100, &[], &[]);

    assert_eq!(
        store.entity_count(),
        1,
        "entity with no components should be tracked in the index"
    );
    assert_eq!(
        store.archetype_count(),
        1,
        "one archetype should be created for the empty component set"
    );

    let arch_id = ArchetypeId::from_component_ids(&[]);
    let arch = store
        .archetypes
        .get(&arch_id)
        .expect("archetype for empty component set should exist");
    assert!(arch.component_ids.is_empty(), "archetype should have no component types");
    assert!(arch.columns.is_empty(), "archetype should have no data columns");
    assert!(
        arch.entities.contains(&100),
        "archetype should track the entity"
    );

    // Query with empty component_ids returns all entities (vacuous truth).
    let all = store.query(&[]);
    assert_eq!(all, vec![100], "query with empty component IDs should return entity 100");
}

// ── Test 6: Order-independent archetype deduplication ─────────────────────

#[test]
fn test_whitebox_order_independent_archetype() {
    let registry = make_registry();
    let mut store = ComponentStore::new(registry);

    // Same component set, different declaration order.
    store.spawn(100, &[1, 2], &[(1, vec![0xAA; 12]), (2, vec![0xBB; 8])]);
    store.spawn(200, &[2, 1], &[(1, vec![0xCC; 12]), (2, vec![0xDD; 8])]);

    assert_eq!(
        store.archetypes.len(),
        1,
        "[1,2] and [2,1] should map to the same archetype"
    );
    assert_eq!(store.entity_count(), 2);

    let arch_id = ArchetypeId::from_component_ids(&[1, 2]);
    let arch = store.archetypes.get(&arch_id).expect("archetype should exist");
    assert_eq!(
        arch.component_ids,
        vec![1, 2],
        "archetype component_ids should be sorted"
    );
    assert_eq!(arch.entities.len(), 2);

    // Both entities must be queryable.
    let results = store.query(&[1, 2]);
    assert_eq!(results.len(), 2);
    assert!(results.contains(&100));
    assert!(results.contains(&200));
}

// ── Test 7: Query with partial (superset) match ──────────────────────────

#[test]
fn test_whitebox_query_partial_superset_match() {
    let registry = make_registry();
    let mut store = ComponentStore::new(registry);

    // Entity has [1, 2, 3] -- three components.
    store.spawn(100, &[1, 2, 3], &[
        (1, vec![0xAA; 12]),
        (2, vec![0xBB; 8]),
        (3, vec![0xCC; 4]),
    ]);

    // Query for a subset: entity has [1,2,3], query is [1,2] -> match (superset).
    let subset = store.query(&[1, 2]);
    assert_eq!(subset, vec![100], "query [1,2] should match entity [1,2,3]");

    // Query for a component the entity does NOT have: no match.
    let missing = store.query(&[1, 4]);
    assert!(
        missing.is_empty(),
        "query [1,4] should NOT match entity [1,2,3]"
    );

    // Single component match.
    let single = store.query(&[2]);
    assert_eq!(single, vec![100], "query [2] should match entity [1,2,3]");

    // Unrelated component.
    let none = store.query(&[4]);
    assert!(none.is_empty(), "query [4] should not match entity [1,2,3]");
}

// ── Test 8: column_slice after write ──────────────────────────────────────

#[test]
fn test_whitebox_column_slice_after_write() {
    let registry = make_registry();
    let mut store = ComponentStore::new(registry);

    // Spawn 3 entities in the same archetype [1].
    store.spawn(100, &[1], &[(1, vec![0xAA; 12])]);
    store.spawn(200, &[1], &[(1, vec![0xBB; 12])]);
    store.spawn(300, &[1], &[(1, vec![0xCC; 12])]);

    // Write into a different byte range of each entity.
    store.write_field(100, 1, 0, &[0x10, 0x11, 0x12, 0x13]);
    store.write_field(200, 1, 4, &[0x20, 0x21, 0x22, 0x23]);
    store.write_field(300, 1, 8, &[0x30, 0x31, 0x32, 0x33]);

    let arch_id = ArchetypeId::from_component_ids(&[1]);
    let slice = store
        .column_slice(arch_id, 0)
        .expect("column_slice should exist for the archetype");

    // 3 rows x 12 bytes = 36 total.
    assert_eq!(slice.len(), 36, "3 rows of 12-byte components = 36 bytes");

    // Row 0 (entity 100): first 4 bytes overwritten, rest 0xAA.
    assert_eq!(
        &slice[0..4],
        &[0x10, 0x11, 0x12, 0x13],
        "row 0's first 4 bytes should reflect write_field"
    );
    assert_eq!(&slice[4..12], &[0xAA; 8], "row 0's tail unchanged");

    // Row 1 (entity 200): bytes 4-7 overwritten, rest 0xBB.
    assert_eq!(&slice[12..16], &[0xBB; 4], "row 1's head unchanged");
    assert_eq!(
        &slice[16..20],
        &[0x20, 0x21, 0x22, 0x23],
        "row 1's middle bytes should reflect write_field"
    );
    assert_eq!(&slice[20..24], &[0xBB; 4], "row 1's tail unchanged");

    // Row 2 (entity 300): bytes 8-11 overwritten, rest 0xCC.
    assert_eq!(&slice[24..32], &[0xCC; 8], "row 2's head unchanged");
    assert_eq!(
        &slice[32..36],
        &[0x30, 0x31, 0x32, 0x33],
        "row 2's last 4 bytes should reflect write_field"
    );
}
