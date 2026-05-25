//! Editor state that connects the renderer to the ECS for inspection.
//!
//! Provides a lightweight [`Editor`] struct that wraps a shared
//! [`ComponentStore`](crate::component_store::ComponentStore) and exposes
//! entity selection, hierarchy/inspector/viewport visibility flags, and
//! methods for retrieving component data about the currently-selected entity.

use crate::component_store::{ComponentStore, global_component_store};
use std::sync::Arc;
use parking_lot::RwLock;

/// Visual-state flags and selection for the editor UI.
#[derive(Debug, Clone)]
pub struct EditorState {
    /// Entity currently selected in the editor (None = nothing selected).
    pub selected_entity: Option<u64>,
    /// Whether the entity hierarchy panel is visible.
    pub show_hierarchy: bool,
    /// Whether the component inspector panel is visible.
    pub show_inspector: bool,
    /// Whether the 3D viewport is visible.
    pub show_viewport: bool,
}

impl Default for EditorState {
    fn default() -> Self {
        Self {
            selected_entity: None,
            show_hierarchy: true,
            show_inspector: true,
            show_viewport: true,
        }
    }
}

/// Editor front-end sitting on top of the global ECS component store.
///
/// The editor does **not** own the store -- it holds a shared reference so
/// that the renderer and the Python bridge layer can mutate entities while
/// the editor remains in sync.
#[derive(Debug)]
pub struct Editor {
    pub state: EditorState,
    pub component_store: Arc<RwLock<ComponentStore>>,
}

impl Editor {
    /// Create a new editor wrapping the given component store.
    pub fn new(store: Arc<RwLock<ComponentStore>>) -> Self {
        Self {
            state: EditorState::default(),
            component_store: store,
        }
    }

    /// Set the selected entity to `entity_id`.
    pub fn select_entity(&mut self, entity_id: u64) {
        self.state.selected_entity = Some(entity_id);
    }

    /// Deselect the currently-selected entity.
    pub fn deselect(&mut self) {
        self.state.selected_entity = None;
    }

    /// Return component data for the selected entity.
    ///
    /// Each entry is `(component_id, component_name, raw_bytes)`.
    /// Returns an empty `Vec` when nothing is selected or the selected entity
    /// no longer exists.
    pub fn selected_components(&self) -> Vec<(u32, String, Vec<u8>)> {
        let entity_id = match self.state.selected_entity {
            Some(id) => id,
            None => return Vec::new(),
        };

        let store = self.component_store.read();
        let (arch_id, row) = match store.entity_index.get(&entity_id) {
            Some(entry) => *entry,
            None => return Vec::new(),
        };
        let archetype = match store.archetypes.get(&arch_id) {
            Some(a) => a,
            None => return Vec::new(),
        };

        let mut components = Vec::with_capacity(archetype.component_ids.len());
        for (col_idx, comp_id) in archetype.component_ids.iter().enumerate() {
            let name = store
                .registry
                .get(*comp_id)
                .map(|info| info.name)
                .unwrap_or_else(|| format!("unknown_{}", comp_id));
            let stride = store
                .registry
                .get(*comp_id)
                .map(|info| info.size)
                .unwrap_or(0);

            let col = &archetype.columns[col_idx];
            let start = row * stride;
            let end = (start + stride).min(col.len());
            let bytes = if start < end {
                col[start..end].to_vec()
            } else {
                Vec::new()
            };
            components.push((*comp_id, name, bytes));
        }

        components
    }

    /// Number of alive entities in the component store.
    pub fn entity_count(&self) -> usize {
        let store = self.component_store.read();
        store.entity_index.len()
    }

    /// Return the IDs of all alive entities in the component store.
    pub fn entity_ids(&self) -> Vec<u64> {
        let store = self.component_store.read();
        store.entity_index.keys().copied().collect()
    }
}

/// Convenience constructor using the **global** component store singleton.
///
/// # Panics
/// Panics if [`global_component_store()`] has not been initialised (call
/// [`initialize_component_store`](crate::component_store::initialize_component_store)
/// first).
pub fn open_editor() -> Editor {
    Editor::new(global_component_store().clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component_store::ComponentStore;
    use crate::type_registry::{ComponentTypeInfo, TypeRegistry};

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
        Arc::new(registry)
    }

    fn make_store() -> Arc<RwLock<ComponentStore>> {
        let registry = make_registry();
        Arc::new(RwLock::new(ComponentStore::new(registry)))
    }

    fn populate_store(store: &Arc<RwLock<ComponentStore>>) {
        let mut guard = store.write();
        guard.spawn(100, &[1, 2], &[(1, vec![0xAA; 12]), (2, vec![0xBB; 8])]);
        guard.spawn(200, &[1], &[(1, vec![0xCC; 12])]);
        // 200 is Position-only
    }

    // ── Tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_new_editor_has_default_state() {
        let store = make_store();
        let editor = Editor::new(store);
        assert!(editor.state.selected_entity.is_none());
        assert!(editor.state.show_hierarchy);
        assert!(editor.state.show_inspector);
        assert!(editor.state.show_viewport);
    }

    #[test]
    fn test_select_and_deselect_entity() {
        let store = make_store();
        let mut editor = Editor::new(store);

        editor.select_entity(42);
        assert_eq!(editor.state.selected_entity, Some(42));

        editor.deselect();
        assert!(editor.state.selected_entity.is_none());
    }

    #[test]
    fn test_entity_count_and_ids() {
        let store = make_store();
        populate_store(&store);
        let editor = Editor::new(store);

        assert_eq!(editor.entity_count(), 2);

        let mut ids = editor.entity_ids();
        ids.sort();
        assert_eq!(ids, vec![100, 200]);
    }

    #[test]
    fn test_selected_components_empty_when_nothing_selected() {
        let store = make_store();
        populate_store(&store);
        let editor = Editor::new(store);

        assert!(editor.selected_components().is_empty());
    }

    #[test]
    fn test_selected_components_returns_data_for_selected_entity() {
        let store = make_store();
        populate_store(&store);
        let mut editor = Editor::new(store);

        editor.select_entity(100);
        let components = editor.selected_components();

        // Entity 100 has Position (id=1) and Velocity (id=2).
        assert_eq!(components.len(), 2);

        // Position should be first (sorted component_ids: [1, 2]).
        assert_eq!(components[0].0, 1, "first component is Position");
        assert_eq!(components[0].1, "Position");
        assert_eq!(components[0].2.len(), 12);

        // Velocity second.
        assert_eq!(components[1].0, 2, "second component is Velocity");
        assert_eq!(components[1].1, "Velocity");
        assert_eq!(components[1].2.len(), 8);

        // Verify actual bytes.
        assert_eq!(components[0].2, vec![0xAA; 12]);
        assert_eq!(components[1].2, vec![0xBB; 8]);
    }

    #[test]
    fn test_selected_components_empty_for_bogus_entity() {
        let store = make_store();
        let mut editor = Editor::new(store);

        editor.select_entity(999);
        assert!(editor.selected_components().is_empty());
    }

    #[test]
    fn test_entity_count_empty_store() {
        let store = make_store();
        let editor = Editor::new(store);
        assert_eq!(editor.entity_count(), 0);
        assert!(editor.entity_ids().is_empty());
    }
}
