//! ECS storage backend -- raw byte-component data in SoA (Struct of Arrays)
//! columns per archetype.
//!
//! This module is **not** a full ECS -- entity lifecycle (World, ArchetypeGraph,
//! Query) stays in Python (`engine/core/ecs/`).  The Rust side stores raw bytes
//! and provides read / write / query primitives that the Python layer calls via
//! PyO3 through the [`bridge`](crate::bridge) module.
//!
//! # Layout
//!
//! Each [`Archetype`] holds one contiguous `Vec<u8>` per component type (a
//! "column").  Row `i` of a component occupies `stride` bytes starting at
//! byte `i * stride`, where `stride` is that component's total byte size as
//! recorded in the [`TypeRegistry`](crate::type_registry::TypeRegistry).
//!
//! Field offsets (from [`FieldLayout`](crate::type_registry::FieldLayout)) are
//! interpreted by the caller -- the store itself is offset-and-byte-range
//! agnostic.

use crate::type_registry::{ArchetypeId, TypeRegistry};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::OnceLock;

// ── Archetype ──────────────────────────────────────────────────────────

/// A set of entities that all share the same component type signature.
///
/// Data is stored in **Struct of Arrays** layout:
/// - `columns[ci]` holds the raw bytes for *every* entity's component of type
///   `component_ids[ci]`, concatenated row-by-row.
/// - Row `r` of column `ci` begins at byte `r * stride`, where `stride` is
///   the component's total size from the type registry.
/// - A row that has been freed (via [`ComponentStore::despawn`]) is pushed
///   onto `free_rows` so the next spawn can reuse it without reallocating.
#[derive(Debug)]
pub struct Archetype {
    /// Stable identifier derived from the set of component type IDs.
    pub id: ArchetypeId,
    /// Sorted component type IDs that define this archetype.
    pub component_ids: Vec<u32>,
    /// SoA columns -- one `Vec<u8>` per component type.
    ///
    /// Indexed by position in [`Self::component_ids`]:
    /// `columns[i]` corresponds to `component_ids[i]`.
    pub columns: Vec<Vec<u8>>,
    /// Entity IDs indexed by row position.  `entities[row] == entity_id`.
    pub entities: Vec<u64>,
    /// Free list of row indices available for reuse.
    pub free_rows: Vec<usize>,
}

impl Archetype {
    /// Number of currently-alive entities in this archetype.
    pub fn alive_count(&self) -> usize {
        self.entities.len() - self.free_rows.len()
    }
}

// ── ComponentStore ─────────────────────────────────────────────────────

/// Thread-safe ECS storage backend.
///
/// Manages a collection of archetypes and provides low-level byte
/// read / write / query operations.  All public mutation is mediated through
/// `&mut self` -- callers are expected to wrap this in `Arc<RwLock<...>>` for
/// shared access (see [`initialize_component_store`]).
#[derive(Debug)]
pub struct ComponentStore {
    /// All archetypes, keyed by [`ArchetypeId`].
    pub archetypes: HashMap<ArchetypeId, Archetype>,
    /// Entity index -- maps `entity_id -> (archetype_id, row)`.
    pub entity_index: HashMap<u64, (ArchetypeId, usize)>,
    /// Reference to the shared type registry.
    pub registry: Arc<TypeRegistry>,
}

impl ComponentStore {
    /// Create a new, empty store.
    pub fn new(registry: Arc<TypeRegistry>) -> Self {
        Self {
            archetypes: HashMap::new(),
            entity_index: HashMap::new(),
            registry,
        }
    }

    /// Spawn an entity with the given component types and initial data.
    ///
    /// `component_ids` defines the archetype the entity will belong to;
    /// the archetype is created on demand.  `component_data` supplies
    /// initial bytes for each component -- the caller is expected to provide
    /// data for every component in the set, but any missing components are
    /// zero-initialised.
    ///
    /// The entity's row is allocated from the free list when possible,
    /// otherwise appended.
    pub fn spawn(
        &mut self,
        entity_id: u64,
        component_ids: &[u32],
        component_data: &[(u32, Vec<u8>)],
    ) {
        let arch_id = ArchetypeId::from_component_ids(component_ids);

        // Sorted copy for deterministic column ordering.
        let sorted_ids = {
            let mut ids = component_ids.to_vec();
            ids.sort();
            ids
        };

        let archetype = self.archetypes.entry(arch_id).or_insert_with(|| {
            let col_count = sorted_ids.len();
            Archetype {
                id: arch_id,
                component_ids: sorted_ids,
                columns: (0..col_count).map(|_| Vec::new()).collect(),
                entities: Vec::new(),
                free_rows: Vec::new(),
            }
        });

        // ── Row allocation ──────────────────────────────────────────
        let row = if let Some(free) = archetype.free_rows.pop() {
            archetype.entities[free] = entity_id;
            free
        } else {
            let row = archetype.entities.len();
            archetype.entities.push(entity_id);
            row
        };

        // ── Write initial component data ────────────────────────────
        for (comp_id, data) in component_data {
            if let Some(comp_idx) = archetype.component_ids.iter().position(|c| *c == *comp_id) {
                if let Some(info) = self.registry.get(*comp_id) {
                    let stride = info.size;
                    let col = &mut archetype.columns[comp_idx];
                    let needed = (row + 1) * stride;
                    if col.len() < needed {
                        col.resize(needed, 0);
                    }
                    let write_len = data.len().min(stride);
                    let start = row * stride;
                    col[start..start + write_len].copy_from_slice(&data[..write_len]);
                }
            }
        }

        self.entity_index.insert(entity_id, (arch_id, row));
    }

    /// Despawn (remove) an entity, freeing its storage row for reuse.
    ///
    /// This is idempotent: calling `despawn` a second time for the same
    /// entity has no effect (the entity is already absent from the index).
    pub fn despawn(&mut self, entity_id: u64) {
        if let Some((arch_id, row)) = self.entity_index.remove(&entity_id) {
            if let Some(archetype) = self.archetypes.get_mut(&arch_id) {
                // Guard against double-free (should not happen in normal
                // usage since entity_index.remove would return None for an
                // already-despawned entity, but covering the direct path
                // is cheap insurance).
                if !archetype.free_rows.contains(&row) {
                    archetype.free_rows.push(row);
                }
            }
        }
    }

    /// Read raw bytes for a specific field of a component on an entity.
    ///
    /// Returns `None` when the entity, component, or byte range is invalid.
    pub fn read_field(
        &self,
        entity_id: u64,
        component_id: u32,
        offset: usize,
        size: usize,
    ) -> Option<Vec<u8>> {
        let (arch_id, row) = self.entity_index.get(&entity_id)?;
        let archetype = self.archetypes.get(arch_id)?;
        let comp_idx = archetype
            .component_ids
            .iter()
            .position(|c| *c == component_id)?;
        let col = &archetype.columns[comp_idx];
        let info = self.registry.get(component_id)?;
        let stride = info.size;
        let start = row * stride + offset;
        if start + size > col.len() {
            return None;
        }
        Some(col[start..start + size].to_vec())
    }

    /// Write raw bytes into a specific field of a component on an entity.
    ///
    /// If the column is not yet long enough to hold the data at the given
    /// row and offset, it is silently extended with zeros.
    ///
    /// Does nothing when the entity or component is unknown.
    pub fn write_field(
        &mut self,
        entity_id: u64,
        component_id: u32,
        offset: usize,
        data: &[u8],
    ) {
        // We need (arch_id, row) by value to avoid the borrow checker
        // fighting with the mutable self access below.
        let entry = self
            .entity_index
            .get(&entity_id)
            .map(|&(arch_id, row)| (arch_id, row));
        let (arch_id, row) = match entry {
            Some(e) => e,
            None => return,
        };

        let archetype = match self.archetypes.get_mut(&arch_id) {
            Some(a) => a,
            None => return,
        };

        let comp_idx = match archetype.component_ids.iter().position(|c| *c == component_id) {
            Some(idx) => idx,
            None => return,
        };

        let info = match self.registry.get(component_id) {
            Some(i) => i,
            None => return,
        };

        let stride = info.size;
        let start = row * stride + offset;
        let col = &mut archetype.columns[comp_idx];
        let write_end = start + data.len();
        if write_end > col.len() {
            col.resize(write_end, 0);
        }
        col[start..write_end].copy_from_slice(data);
    }

    /// Find all entities that possess **every** component type listed in
    /// `component_ids` (superset match).
    ///
    /// An entity matches if its archetype contains all of the queried
    /// component types, even if it also contains additional types.
    /// Despawned (freed) entities are excluded from results.
    pub fn query(&self, component_ids: &[u32]) -> Vec<u64> {
        let mut results = Vec::new();
        for archetype in self.archetypes.values() {
            if component_ids
                .iter()
                .all(|cid| archetype.component_ids.contains(cid))
            {
                for (row, &entity_id) in archetype.entities.iter().enumerate() {
                    if !archetype.free_rows.contains(&row) {
                        results.push(entity_id);
                    }
                }
            }
        }
        results
    }

    /// Return a raw byte slice for an entire component column in an archetype.
    ///
    /// `component_index` is the index into the archetype's
    /// [`component_ids`](Archetype::component_ids) / [`columns`](Archetype::columns)
    /// vectors (not a type ID).  Returns `None` when the archetype or column
    /// does not exist.
    pub fn column_slice(
        &self,
        archetype_id: ArchetypeId,
        component_index: usize,
    ) -> Option<&[u8]> {
        self.archetypes
            .get(&archetype_id)
            .and_then(|a| a.columns.get(component_index))
            .map(|c| c.as_slice())
    }

    // ── Convenience accessors ───────────────────────────────────────

    /// Number of alive (spawned, not despawned) entities.
    pub fn entity_count(&self) -> usize {
        self.entity_index.len()
    }

    /// Number of archetypes in the store.
    pub fn archetype_count(&self) -> usize {
        self.archetypes.len()
    }
}

// ── Global singleton ───────────────────────────────────────────────────

static COMPONENT_STORE: OnceLock<Arc<RwLock<ComponentStore>>> = OnceLock::new();

/// Initialise the global component store singleton.
///
/// Must be called exactly once before [`global_component_store`] is used.
/// Subsequent calls are silently ignored (the first registration wins).
pub fn initialize_component_store(registry: Arc<TypeRegistry>) {
    COMPONENT_STORE.get_or_init(|| Arc::new(RwLock::new(ComponentStore::new(registry))));
}

/// Access the global component store.
///
/// # Panics
/// Panics if [`initialize_component_store`] has not been called.
pub fn global_component_store() -> &'static Arc<RwLock<ComponentStore>> {
    COMPONENT_STORE
        .get()
        .expect("ComponentStore not initialised. Call initialize_component_store first.")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::type_registry::ComponentTypeInfo;

    // ── Test helpers ────────────────────────────────────────────────

    fn make_registry() -> Arc<TypeRegistry> {
        let registry = TypeRegistry::new();
        // Component 1: Position (3 f32s = 12 bytes)
        registry.register(ComponentTypeInfo {
            id: 1,
            name: "Position".into(),
            size: 12,
            fields: vec![],
            flags: 0,
            archetype_id: None,
        });
        // Component 2: Velocity (2 f32s = 8 bytes)
        registry.register(ComponentTypeInfo {
            id: 2,
            name: "Velocity".into(),
            size: 8,
            fields: vec![],
            flags: 0,
            archetype_id: None,
        });
        // Component 3: Health (1 f32 = 4 bytes)
        registry.register(ComponentTypeInfo {
            id: 3,
            name: "Health".into(),
            size: 4,
            fields: vec![],
            flags: 0,
            archetype_id: None,
        });
        // Component 4: Color (4 f32s = 16 bytes)
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

    // ── Tests ───────────────────────────────────────────────────────

    #[test]
    fn test_spawn_and_read_field_round_trip() {
        let registry = make_registry();
        let mut store = ComponentStore::new(registry);

        let pos_data: Vec<u8> = (0..12).collect(); // 0, 1, 2, ..., 11
        let vel_data: Vec<u8> = (10..18).collect(); // 10, 11, ..., 17

        store.spawn(100, &[1, 2], &[(1, pos_data.clone()), (2, vel_data.clone())]);

        let read_pos = store
            .read_field(100, 1, 0, 12)
            .expect("should read Position back");
        assert_eq!(read_pos, pos_data, "Position data should round-trip");

        let read_vel = store
            .read_field(100, 2, 0, 8)
            .expect("should read Velocity back");
        assert_eq!(read_vel, vel_data, "Velocity data should round-trip");
    }

    #[test]
    fn test_write_field_updates_data() {
        let registry = make_registry();
        let mut store = ComponentStore::new(registry);

        store.spawn(100, &[1], &[(1, vec![0; 12])]);

        // Full-field overwrite
        let new_pos: Vec<u8> = (0x10..0x1C).collect();
        store.write_field(100, 1, 0, &new_pos);

        let read_back = store
            .read_field(100, 1, 0, 12)
            .expect("should read updated Position");
        assert_eq!(read_back, new_pos, "Full-field write should update all bytes");

        // Partial overwrite at offset 4 (sub-field, e.g. "y" in x/y/z).
        let partial = vec![0xFF, 0xFE, 0xFD, 0xFC];
        store.write_field(100, 1, 4, &partial);

        let full = store.read_field(100, 1, 0, 12).unwrap();
        assert_eq!(&full[0..4], &new_pos[0..4], "First 4 bytes (x) unchanged");
        assert_eq!(&full[4..8], &partial[..], "Bytes 4-7 (y) updated");
        assert_eq!(&full[8..12], &new_pos[8..12], "Last 4 bytes (z) unchanged");
    }

    #[test]
    fn test_despawn_removes_entity() {
        let registry = make_registry();
        let mut store = ComponentStore::new(registry);

        store.spawn(100, &[1], &[(1, vec![0; 12])]);
        assert!(
            store.read_field(100, 1, 0, 1).is_some(),
            "Entity exists after spawn"
        );

        store.despawn(100);
        assert!(
            store.read_field(100, 1, 0, 1).is_none(),
            "Entity gone after despawn"
        );
        assert_eq!(store.entity_count(), 0, "Entity count is zero");
    }

    #[test]
    fn test_despawn_twice_is_idempotent() {
        let registry = make_registry();
        let mut store = ComponentStore::new(registry);

        store.spawn(100, &[1], &[(1, vec![0; 12])]);
        store.despawn(100);
        store.despawn(100); // must not panic
    }

    #[test]
    fn test_query_returns_matching_entities() {
        let registry = make_registry();
        let mut store = ComponentStore::new(registry);

        // A: Position + Velocity
        store.spawn(100, &[1, 2], &[(1, vec![0; 12]), (2, vec![0; 8])]);
        // B: Position + Health
        store.spawn(200, &[1, 3], &[(1, vec![0; 12]), (3, vec![0; 4])]);
        // C: Position + Velocity + Health
        store.spawn(
            300,
            &[1, 2, 3],
            &[(1, vec![0; 12]), (2, vec![0; 8]), (3, vec![0; 4])],
        );

        // Only Position: all 3 match.
        let only_pos = store.query(&[1]);
        assert_eq!(only_pos.len(), 3);
        assert!(only_pos.contains(&100));
        assert!(only_pos.contains(&200));
        assert!(only_pos.contains(&300));

        // Position + Velocity: A and C.
        let pos_vel = store.query(&[1, 2]);
        assert_eq!(pos_vel.len(), 2);
        assert!(pos_vel.contains(&100));
        assert!(pos_vel.contains(&300));

        // Velocity + Health: only C.
        let vel_health = store.query(&[2, 3]);
        assert_eq!(vel_health.len(), 1);
        assert_eq!(vel_health[0], 300);

        // Color alone: none.
        let color = store.query(&[4]);
        assert!(color.is_empty());
    }

    #[test]
    fn test_query_excludes_despawned_entities() {
        let registry = make_registry();
        let mut store = ComponentStore::new(registry);

        store.spawn(100, &[1], &[(1, vec![0; 12])]);
        store.spawn(200, &[1], &[(1, vec![0; 12])]);
        store.despawn(100);

        let entities = store.query(&[1]);
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0], 200);
    }

    #[test]
    fn test_query_empty_store() {
        let registry = make_registry();
        let store = ComponentStore::new(registry);
        assert!(store.query(&[1]).is_empty());
    }

    #[test]
    fn test_column_slice_returns_correct_bytes() {
        let registry = make_registry();
        let mut store = ComponentStore::new(registry);

        store.spawn(100, &[1], &[(1, (0..12).collect())]);
        store.spawn(200, &[1], &[(1, (12..24).collect())]);

        let arch_id = ArchetypeId::from_component_ids(&[1]);
        let slice = store
            .column_slice(arch_id, 0)
            .expect("column should exist");
        assert_eq!(slice.len(), 24, "2 rows of 12-byte components = 24 bytes");
        assert_eq!(&slice[0..12], &(0..12).collect::<Vec<u8>>()[..]);
        assert_eq!(&slice[12..24], &(12..24).collect::<Vec<u8>>()[..]);
    }

    #[test]
    fn test_column_slice_nonexistent() {
        let registry = make_registry();
        let store = ComponentStore::new(registry);

        // Archetype that has never been created.
        let arch_id = ArchetypeId::from_component_ids(&[1]);
        assert!(store.column_slice(arch_id, 0).is_none());
    }

    #[test]
    fn test_spawn_multiple_entities_same_archetype() {
        let registry = make_registry();
        let mut store = ComponentStore::new(registry);

        store.spawn(100, &[1, 2], &[(1, vec![0; 12]), (2, vec![0; 8])]);
        store.spawn(200, &[1, 2], &[(1, vec![0; 12]), (2, vec![0; 8])]);

        // Only one archetype should have been created.
        assert_eq!(store.archetypes.len(), 1);

        let arch_id = ArchetypeId::from_component_ids(&[1, 2]);
        let arch = store.archetypes.get(&arch_id).unwrap();
        assert_eq!(arch.entities.len(), 2);
        assert!(!arch.free_rows.contains(&0));
        assert!(!arch.free_rows.contains(&1));

        // Each column: 2 rows × component stride.
        assert_eq!(arch.columns[0].len(), 24); // Position: 2 × 12
        assert_eq!(arch.columns[1].len(), 16); // Velocity: 2 × 8
    }

    #[test]
    fn test_archetype_deduplication() {
        let registry = make_registry();
        let mut store = ComponentStore::new(registry);

        // Same component set, different input order.
        store.spawn(100, &[1, 2], &[(1, vec![0; 12]), (2, vec![0; 8])]);
        store.spawn(200, &[2, 1], &[(1, vec![0; 12]), (2, vec![0; 8])]);

        assert_eq!(
            store.archetypes.len(),
            1,
            "same component set => single archetype"
        );
        assert_eq!(store.entity_count(), 2);
    }

    #[test]
    fn test_despawn_reuses_row() {
        let registry = make_registry();
        let mut store = ComponentStore::new(registry);

        store.spawn(100, &[1], &[(1, vec![0xAA; 12])]);
        store.spawn(200, &[1], &[(1, vec![0xBB; 12])]);

        store.despawn(100); // frees row 0

        // Spawn a third entity -- should reuse row 0.
        store.spawn(300, &[1], &[(1, vec![0xCC; 12])]);

        let arch_id = ArchetypeId::from_component_ids(&[1]);
        let arch = store.archetypes.get(&arch_id).unwrap();

        // entities vec did not grow past 2.
        assert_eq!(arch.entities.len(), 2);
        assert_eq!(arch.entities[0], 300, "row 0 reused by entity 300");
        assert_eq!(arch.entities[1], 200, "row 1 still holds entity 200");
        assert!(arch.free_rows.is_empty(), "free list consumed");

        // Data at row 0 belongs to entity 300, not a stale copy.
        let data = store.read_field(300, 1, 0, 12).unwrap();
        assert_eq!(data, vec![0xCC; 12]);
    }

    #[test]
    fn test_read_field_nonexistent_entity() {
        let registry = make_registry();
        let store = ComponentStore::new(registry);
        assert!(store.read_field(999, 1, 0, 4).is_none());
    }

    #[test]
    fn test_read_field_out_of_range() {
        let registry = make_registry();
        let mut store = ComponentStore::new(registry);

        store.spawn(100, &[1], &[(1, vec![0; 12])]);

        // Reading beyond the component's size should return None.
        assert!(
            store.read_field(100, 1, 0, 13).is_none(),
            "read beyond component size should be None"
        );
        assert!(
            store.read_field(100, 1, 12, 1).is_none(),
            "read exactly at end should be None"
        );
    }

    #[test]
    fn test_write_field_nonexistent_does_nothing() {
        let registry = make_registry();
        let mut store = ComponentStore::new(registry);

        // Must not panic.
        store.write_field(999, 1, 0, &[0; 4]);
    }

    #[test]
    fn test_write_field_extends_column() {
        let registry = make_registry();
        let mut store = ComponentStore::new(registry);

        // Spawn with no data (will create empty column? No, spawn requires
        // at least empty data, but let's use zero-length data).
        store.spawn(100, &[1], &[(1, vec![0; 12])]);

        // Write beyond the current row's stride -- only the row boundary
        // matters, but offset 12 would be *row 1's* data.  Since we only
        // have row 0, writing at offset 12 would extend the column by 12
        // bytes (row 1's slot).  This is an edge case: writing into "row 1
        // territory" for an entity that lives at row 0.
        //
        // This exercises the resize path.
        store.write_field(100, 1, 8, &[0xFF; 8]);

        // Read back the last 4 bytes (written into what is now row 0's end).
        let tail = store.read_field(100, 1, 8, 4).unwrap();
        assert_eq!(tail, vec![0xFF; 4]);
    }
}
