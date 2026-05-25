// Whitebox tests for type_registry.rs V2 -- internals exercise.
//
// T-BRG-1.1: DEV rewrote type_registry.rs from a skeleton into a production
// implementation (ArchetypeId XOR-fold hashing, parking_lot::RwLock interior,
// flags/archetype_id on ComponentTypeInfo).  These tests validate internal
// invariants that blackbox tests cannot reach: hash collision resistance,
// concurrent mutation, concurrent reads, structural defaults.
//
// NOTE: `TypeRegistry.types` is private, so all verification goes through
// the public API (get, contains, len, is_empty, ids, type_list).

use renderer_backend::type_registry::{
    ArchetypeId, ComponentTypeInfo, FieldLayout, TypeRegistry,
};
use std::sync::Arc;
use std::collections::HashSet;
use std::thread;

// ---------------------------------------------------------------------------
// Helper -- construct a ComponentTypeInfo with canonical defaults.
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
// 1.  ArchetypeId collision resistance
//
//     Verify that four distinct component sets produce four distinct IDs.
//     This catches XOR-fold collisions introduced by the DefaultHasher path.
// ===========================================================================

#[test]
fn archetype_id_collision_resistance() {
    let sets: [&[u32]; 4] = [&[1, 2], &[2, 3], &[1, 2, 3], &[100, 200, 300]];

    let unique: HashSet<ArchetypeId> =
        sets.iter().map(|s| ArchetypeId::from_component_ids(s)).collect();

    assert_eq!(
        unique.len(),
        sets.len(),
        "collision detected: {} distinct sets produced only {} unique ArchetypeIds",
        sets.len(),
        unique.len(),
    );
}

// ===========================================================================
// 2.  Concurrent register -- no panic under 4-way write contention
//
//     Each thread registers a *different* component type id so there is no
//     data race on the key itself, but all 4 go through the same
//     parking_lot::RwLock write lock.
// ===========================================================================

#[test]
fn concurrent_register_no_panic() {
    let registry = Arc::new(TypeRegistry::new());
    let mut handles = Vec::new();

    for i in 0..4u32 {
        let reg = Arc::clone(&registry);
        handles.push(thread::spawn(move || {
            reg.register(make_type(i, &format!("Type_{}", i), (i as usize + 1) * 4));
        }));
    }

    for h in handles {
        h.join().expect("a thread panicked during concurrent register");
    }

    assert_eq!(registry.len(), 4, "all 4 types must survive concurrent register");
    for i in 0..4 {
        assert!(registry.contains(i), "type id {} must be present in registry", i);
    }
}

// ===========================================================================
// 3.  Concurrent read (get) -- no panic under 4-way read contention
//
//     Pre-register 4 types, then spawn 4 threads each reading one of them
//     simultaneously.  parking_lot::RwLock permits concurrent readers.
// ===========================================================================

#[test]
fn concurrent_get_no_panic() {
    let registry = Arc::new(TypeRegistry::new());

    // Pre-register
    for i in 0..4u32 {
        registry.register(make_type(i, &format!("Type_{}", i), 4));
    }

    let mut handles = Vec::new();
    for i in 0..4u32 {
        let reg = Arc::clone(&registry);
        handles.push(thread::spawn(move || {
            let retrieved = reg.get(i);
            assert!(
                retrieved.is_some(),
                "thread-{} should retrieve its type from the registry",
                i,
            );
            assert_eq!(retrieved.unwrap().id, i);
        }));
    }

    for h in handles {
        h.join().expect("a thread panicked during concurrent get");
    }

    // Registry state must be unchanged after concurrent reads.
    assert_eq!(registry.len(), 4);
    assert!(!registry.is_empty());
}

// ===========================================================================
// 4.  type_list after registering 10 types -- count + name presence
//
//     This is a stress-lite for the HashMap iteration path: 10 entries are
//     enough to surface issues with the collect + map pipeline in type_list().
// ===========================================================================

#[test]
fn type_list_reflects_ten_registrations() {
    let registry = TypeRegistry::new();
    let mut expected: Vec<String> = (0..10)
        .map(|i| format!("Component_{}", i))
        .collect();

    for (i, name) in expected.iter().enumerate() {
        registry.register(make_type(i as u32, name, (i + 1) * 4));
    }

    let list = registry.type_list();
    assert_eq!(
        list.len(),
        10,
        "type_list must report exactly 10 entries after 10 registrations",
    );

    // Collect names, sort both, assert full match.
    let mut got: Vec<String> = list.into_iter().map(|(_, n, _)| n).collect();
    got.sort();
    expected.sort();
    assert_eq!(got, expected, "all 10 type names must appear in type_list");

    // Also verify idempotency -- calling type_list a second time returns the
    // same count.
    assert_eq!(registry.type_list().len(), 10);
    assert_eq!(registry.len(), 10);
}

// ===========================================================================
// 5.  archetype_for with empty slice -- deterministic
//
//     The XOR-fold of a DefaultHasher over an empty iterator is still a
//     deterministic 32-bit value.  Verify that two calls return the same id.
// ===========================================================================

#[test]
fn archetype_for_empty_slice_deterministic() {
    let registry = TypeRegistry::new();

    let a = registry.archetype_for(&[]);
    let b = registry.archetype_for(&[]);

    assert_eq!(
        a, b,
        "archetype_for(&[]) must be deterministic: two calls must produce the same ArchetypeId",
    );

    // Also verify that the value round-trips through u32.
    let raw: u32 = a.into();
    let raw_b: u32 = b.into();
    assert_eq!(raw, raw_b, "u32 representation of empty-set archetype must match");
}

// ===========================================================================
// 6.  ComponentTypeInfo.flags defaults to 0
//
//     The new `flags: u32` field was added by DEV.  Until the bitflags are
//     assigned, every freshly-constructed ComponentTypeInfo should carry
//     flags == 0, and that value must survive a register+get round-trip.
// ===========================================================================

#[test]
fn flags_defaults_to_zero() {
    // Direct construction.
    let info = make_type(1, "Direct", 4);
    assert_eq!(
        info.flags, 0,
        "ComponentTypeInfo constructed with make_type must have flags == 0",
    );

    // register + get round-trip.
    let registry = TypeRegistry::new();
    registry.register(info);
    let retrieved = registry.get(1).expect("get must find registered type");
    assert_eq!(
        retrieved.flags, 0,
        "flags must survive register/get round-trip as 0",
    );
}

// ===========================================================================
// 7.  ComponentTypeInfo.archetype_id defaults to None
//
//     The new `archetype_id: Option<ArchetypeId>` field was added by DEV.
//     Until a component is explicitly assigned to an archetype it should be
//     None, and that value must survive a register+get round-trip.
// ===========================================================================

#[test]
fn archetype_id_defaults_to_none() {
    // Direct construction.
    let info = make_type(1, "Direct", 4);
    assert!(
        info.archetype_id.is_none(),
        "ComponentTypeInfo constructed with make_type must have archetype_id == None",
    );

    // register + get round-trip.
    let registry = TypeRegistry::new();
    registry.register(info);
    let retrieved = registry.get(1).expect("get must find registered type");
    assert!(
        retrieved.archetype_id.is_none(),
        "archetype_id must survive register/get round-trip as None",
    );
}

// ===========================================================================
// 8.  register() overwrite with full V2 fields
//
//     Verify that the overwrite semantics (last-write-wins) still work when
//     the new V2 fields (flags, archetype_id) carry non-default values.
// ===========================================================================

#[test]
fn register_overwrite_with_v2_fields() {
    let registry = TypeRegistry::new();

    // First registration -- defaults for V2 fields.
    registry.register(ComponentTypeInfo {
        id: 1,
        name: "Original".to_string(),
        size: 8,
        fields: vec![],
        flags: 0,
        archetype_id: None,
    });

    // Overwrite -- set V2 fields to non-default values.
    registry.register(ComponentTypeInfo {
        id: 1,
        name: "Overwritten".to_string(),
        size: 16,
        fields: vec![FieldLayout {
            name: "val".to_string(),
            type_code: "U32".to_string(),
            offset: 0,
        }],
        flags: 0b0011,
        archetype_id: Some(ArchetypeId::from_component_ids(&[10, 20])),
    });

    let retrieved = registry.get(1).expect("get must find overwritten id");
    assert_eq!(retrieved.name, "Overwritten");
    assert_eq!(retrieved.size, 16);
    assert_eq!(retrieved.fields.len(), 1);
    assert_eq!(retrieved.flags, 0b0011, "overwritten flags must survive round-trip");
    assert_eq!(
        retrieved.archetype_id,
        Some(ArchetypeId::from_component_ids(&[10, 20])),
        "overwritten archetype_id must survive round-trip",
    );

    // Registry must still have exactly 1 entry.
    assert_eq!(registry.len(), 1);
}
