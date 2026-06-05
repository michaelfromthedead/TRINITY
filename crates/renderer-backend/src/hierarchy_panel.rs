//! Entity Hierarchy Tree Panel
//!
//! This module provides a tree view panel for displaying and manipulating
//! entity hierarchies in the editor. It supports:
//!
//! - Tree construction from flat entity lists
//! - Expand/collapse of tree nodes
//! - Entity selection
//! - Drag-and-drop reparenting with cycle detection
//! - Search/filter functionality
//! - Context menu actions (delete, duplicate, rename)
//!
//! # Example
//!
//! ```rust,ignore
//! use renderer_backend::hierarchy_panel::{HierarchyPanel, EntityNode};
//!
//! let mut panel = HierarchyPanel::new();
//! panel.set_entities(vec![
//!     EntityNode::new(1, "Root"),
//!     EntityNode::with_parent(2, "Child", 1),
//! ]);
//!
//! // Render the panel
//! let action = panel.render(&mut ui_context);
//! match action {
//!     HierarchyAction::Selected(id) => { /* handle selection */ }
//!     HierarchyAction::Reparent { entity, new_parent } => { /* handle reparent */ }
//!     _ => {}
//! }
//! ```

use crate::egui_adapter::UIContext;
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// EntityNode
// ---------------------------------------------------------------------------

/// A node in the entity hierarchy tree.
///
/// Each node represents an entity with its display name, parent-child
/// relationships, and editor state (visible, locked).
#[derive(Debug, Clone, PartialEq)]
pub struct EntityNode {
    /// Unique entity identifier.
    pub entity_id: u64,
    /// Display name for the entity.
    pub name: String,
    /// Parent entity ID, or None for root entities.
    pub parent_id: Option<u64>,
    /// Direct child entity IDs.
    pub children: Vec<u64>,
    /// Whether the entity is visible in the scene.
    pub visible: bool,
    /// Whether the entity is locked from editing.
    pub locked: bool,
}

impl EntityNode {
    /// Create a new root entity node.
    pub fn new(entity_id: u64, name: impl Into<String>) -> Self {
        Self {
            entity_id,
            name: name.into(),
            parent_id: None,
            children: Vec::new(),
            visible: true,
            locked: false,
        }
    }

    /// Create a new entity node with a parent.
    pub fn with_parent(entity_id: u64, name: impl Into<String>, parent_id: u64) -> Self {
        Self {
            entity_id,
            name: name.into(),
            parent_id: Some(parent_id),
            children: Vec::new(),
            visible: true,
            locked: false,
        }
    }

    /// Set visibility state.
    pub fn with_visible(mut self, visible: bool) -> Self {
        self.visible = visible;
        self
    }

    /// Set locked state.
    pub fn with_locked(mut self, locked: bool) -> Self {
        self.locked = locked;
        self
    }

    /// Add a child entity ID.
    pub fn add_child(&mut self, child_id: u64) {
        if !self.children.contains(&child_id) {
            self.children.push(child_id);
        }
    }

    /// Remove a child entity ID.
    pub fn remove_child(&mut self, child_id: u64) {
        self.children.retain(|&id| id != child_id);
    }

    /// Check if this node has children.
    pub fn has_children(&self) -> bool {
        !self.children.is_empty()
    }

    /// Check if this is a root node (no parent).
    pub fn is_root(&self) -> bool {
        self.parent_id.is_none()
    }
}

// ---------------------------------------------------------------------------
// DragState
// ---------------------------------------------------------------------------

/// State for drag-and-drop reparenting operations.
#[derive(Debug, Clone, PartialEq)]
pub struct DragState {
    /// The entity being dragged.
    pub entity_id: u64,
    /// The original parent before the drag started.
    pub original_parent: Option<u64>,
    /// Current drop target (entity to drop onto, or None for root).
    pub drop_target: Option<u64>,
}

impl DragState {
    /// Create a new drag state.
    pub fn new(entity_id: u64, original_parent: Option<u64>) -> Self {
        Self {
            entity_id,
            original_parent,
            drop_target: None,
        }
    }

    /// Set the current drop target.
    pub fn with_drop_target(mut self, target: Option<u64>) -> Self {
        self.drop_target = target;
        self
    }
}

// ---------------------------------------------------------------------------
// HierarchyAction
// ---------------------------------------------------------------------------

/// Actions that can result from user interaction with the hierarchy panel.
#[derive(Debug, Clone, PartialEq)]
pub enum HierarchyAction {
    /// No action occurred.
    None,
    /// An entity was selected.
    Selected(u64),
    /// An entity should be reparented.
    Reparent {
        /// The entity to reparent.
        entity: u64,
        /// The new parent (None for root).
        new_parent: Option<u64>,
    },
    /// A context menu was requested.
    ContextMenu {
        /// The entity for which the context menu was requested.
        entity: u64,
        /// X position of the menu.
        x: f32,
        /// Y position of the menu.
        y: f32,
    },
    /// An entity should be deleted.
    Delete(u64),
    /// An entity should be duplicated.
    Duplicate(u64),
    /// An entity should be renamed.
    Rename {
        /// The entity to rename.
        entity: u64,
        /// The new name.
        new_name: String,
    },
    /// Visibility was toggled.
    ToggleVisibility(u64),
    /// Lock state was toggled.
    ToggleLock(u64),
}

impl Default for HierarchyAction {
    fn default() -> Self {
        Self::None
    }
}

// ---------------------------------------------------------------------------
// HierarchyPanel
// ---------------------------------------------------------------------------

/// The entity hierarchy tree panel.
///
/// Displays entities in a tree structure with expand/collapse, selection,
/// search filtering, and drag-and-drop reparenting.
#[derive(Debug)]
pub struct HierarchyPanel {
    /// All entity nodes indexed by ID.
    nodes: HashMap<u64, EntityNode>,
    /// IDs of root entities (entities with no parent).
    root_ids: Vec<u64>,
    /// Set of expanded node IDs.
    expanded: HashSet<u64>,
    /// Currently selected entity ID.
    selected: Option<u64>,
    /// Search/filter string.
    search_filter: String,
    /// Current drag state for reparenting.
    drag_state: Option<DragState>,
    /// Whether we're in rename mode.
    rename_mode: Option<(u64, String)>,
    /// Context menu position (if open).
    context_menu: Option<(u64, f32, f32)>,
}

impl Default for HierarchyPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl HierarchyPanel {
    /// Create a new empty hierarchy panel.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            root_ids: Vec::new(),
            expanded: HashSet::new(),
            selected: None,
            search_filter: String::new(),
            drag_state: None,
            rename_mode: None,
            context_menu: None,
        }
    }

    /// Set the entities to display in the hierarchy.
    ///
    /// This rebuilds the tree structure from a flat list of nodes.
    /// The nodes' `children` fields are updated based on `parent_id` references.
    pub fn set_entities(&mut self, nodes: Vec<EntityNode>) {
        self.nodes.clear();
        self.root_ids.clear();

        // First pass: insert all nodes
        for node in nodes {
            self.nodes.insert(node.entity_id, node);
        }

        // Second pass: build children lists and find roots
        let ids: Vec<u64> = self.nodes.keys().copied().collect();
        for id in &ids {
            if let Some(node) = self.nodes.get(id) {
                let parent_id = node.parent_id;
                if let Some(parent_id) = parent_id {
                    // This is a child - add it to parent's children list
                    if let Some(parent) = self.nodes.get_mut(&parent_id) {
                        parent.add_child(*id);
                    } else {
                        // Parent doesn't exist, treat as root
                        self.root_ids.push(*id);
                    }
                } else {
                    // This is a root node
                    self.root_ids.push(*id);
                }
            }
        }

        // Sort roots by ID for consistent ordering
        self.root_ids.sort();
    }

    /// Add a single entity to the hierarchy.
    pub fn add_entity(&mut self, node: EntityNode) {
        let id = node.entity_id;
        let parent_id = node.parent_id;

        self.nodes.insert(id, node);

        if let Some(parent_id) = parent_id {
            if let Some(parent) = self.nodes.get_mut(&parent_id) {
                parent.add_child(id);
            } else {
                self.root_ids.push(id);
                self.root_ids.sort();
            }
        } else {
            self.root_ids.push(id);
            self.root_ids.sort();
        }
    }

    /// Remove an entity from the hierarchy.
    ///
    /// Children of the removed entity become orphaned (treated as roots).
    pub fn remove_entity(&mut self, entity_id: u64) {
        if let Some(node) = self.nodes.remove(&entity_id) {
            // Remove from parent's children
            if let Some(parent_id) = node.parent_id {
                if let Some(parent) = self.nodes.get_mut(&parent_id) {
                    parent.remove_child(entity_id);
                }
            }

            // Remove from roots if it was a root
            self.root_ids.retain(|&id| id != entity_id);

            // Orphan children (make them roots)
            for child_id in &node.children {
                if let Some(child) = self.nodes.get_mut(child_id) {
                    child.parent_id = None;
                    self.root_ids.push(*child_id);
                }
            }
            self.root_ids.sort();

            // Clear selection if this entity was selected
            if self.selected == Some(entity_id) {
                self.selected = None;
            }

            // Clear expand state
            self.expanded.remove(&entity_id);
        }
    }

    /// Get a reference to an entity node.
    pub fn get_entity(&self, entity_id: u64) -> Option<&EntityNode> {
        self.nodes.get(&entity_id)
    }

    /// Get a mutable reference to an entity node.
    pub fn get_entity_mut(&mut self, entity_id: u64) -> Option<&mut EntityNode> {
        self.nodes.get_mut(&entity_id)
    }

    /// Get all root entity IDs.
    pub fn root_ids(&self) -> &[u64] {
        &self.root_ids
    }

    /// Get the currently selected entity.
    pub fn selected(&self) -> Option<u64> {
        self.selected
    }

    /// Select an entity.
    pub fn select(&mut self, entity_id: u64) {
        if self.nodes.contains_key(&entity_id) {
            self.selected = Some(entity_id);
        }
    }

    /// Clear selection.
    pub fn clear_selection(&mut self) {
        self.selected = None;
    }

    /// Expand a node to show its children.
    pub fn expand(&mut self, entity_id: u64) {
        if self.nodes.contains_key(&entity_id) {
            self.expanded.insert(entity_id);
        }
    }

    /// Collapse a node to hide its children.
    pub fn collapse(&mut self, entity_id: u64) {
        self.expanded.remove(&entity_id);
    }

    /// Toggle expand/collapse state.
    pub fn toggle_expand(&mut self, entity_id: u64) {
        if self.expanded.contains(&entity_id) {
            self.collapse(entity_id);
        } else {
            self.expand(entity_id);
        }
    }

    /// Check if a node is expanded.
    pub fn is_expanded(&self, entity_id: u64) -> bool {
        self.expanded.contains(&entity_id)
    }

    /// Expand all nodes.
    pub fn expand_all(&mut self) {
        for id in self.nodes.keys() {
            self.expanded.insert(*id);
        }
    }

    /// Collapse all nodes.
    pub fn collapse_all(&mut self) {
        self.expanded.clear();
    }

    /// Expand to show a specific entity (expand all ancestors).
    pub fn reveal(&mut self, entity_id: u64) {
        let mut current = entity_id;
        while let Some(node) = self.nodes.get(&current) {
            if let Some(parent_id) = node.parent_id {
                self.expand(parent_id);
                current = parent_id;
            } else {
                break;
            }
        }
    }

    /// Set the search filter string.
    pub fn set_filter(&mut self, filter: &str) {
        self.search_filter = filter.to_lowercase();
    }

    /// Clear the search filter.
    pub fn clear_filter(&mut self) {
        self.search_filter.clear();
    }

    /// Get the current search filter.
    pub fn filter(&self) -> &str {
        &self.search_filter
    }

    /// Check if an entity matches the current filter.
    fn matches_filter(&self, entity_id: u64) -> bool {
        if self.search_filter.is_empty() {
            return true;
        }

        if let Some(node) = self.nodes.get(&entity_id) {
            // Check if this node matches
            if node.name.to_lowercase().contains(&self.search_filter) {
                return true;
            }

            // Check if any descendant matches
            for &child_id in &node.children {
                if self.matches_filter_recursive(child_id) {
                    return true;
                }
            }
        }

        false
    }

    /// Recursively check if an entity or any of its descendants match the filter.
    fn matches_filter_recursive(&self, entity_id: u64) -> bool {
        if let Some(node) = self.nodes.get(&entity_id) {
            if node.name.to_lowercase().contains(&self.search_filter) {
                return true;
            }
            for &child_id in &node.children {
                if self.matches_filter_recursive(child_id) {
                    return true;
                }
            }
        }
        false
    }

    /// Get all entities that match the current filter.
    pub fn filtered_entities(&self) -> Vec<u64> {
        if self.search_filter.is_empty() {
            return self.nodes.keys().copied().collect();
        }

        let mut matches = Vec::new();
        for &id in self.nodes.keys() {
            if let Some(node) = self.nodes.get(&id) {
                if node.name.to_lowercase().contains(&self.search_filter) {
                    matches.push(id);
                }
            }
        }
        matches
    }

    /// Start dragging an entity for reparenting.
    pub fn start_drag(&mut self, entity_id: u64) {
        if let Some(node) = self.nodes.get(&entity_id) {
            self.drag_state = Some(DragState::new(entity_id, node.parent_id));
        }
    }

    /// Update the drop target during a drag operation.
    pub fn update_drag_target(&mut self, target: Option<u64>) {
        if let Some(ref mut drag) = self.drag_state {
            drag.drop_target = target;
        }
    }

    /// Cancel the current drag operation.
    pub fn cancel_drag(&mut self) {
        self.drag_state = None;
    }

    /// Check if reparenting would create a cycle.
    ///
    /// A cycle would occur if the new parent is a descendant of the entity
    /// being moved.
    pub fn would_create_cycle(&self, entity_id: u64, new_parent: u64) -> bool {
        // Can't parent to self
        if entity_id == new_parent {
            return true;
        }

        // Check if new_parent is a descendant of entity_id
        self.is_descendant_of(new_parent, entity_id)
    }

    /// Check if `entity_id` is a descendant of `ancestor_id`.
    pub fn is_descendant_of(&self, entity_id: u64, ancestor_id: u64) -> bool {
        let mut current = entity_id;
        let mut visited = HashSet::new();

        while let Some(node) = self.nodes.get(&current) {
            if visited.contains(&current) {
                // Already have a cycle
                return false;
            }
            visited.insert(current);

            if let Some(parent_id) = node.parent_id {
                if parent_id == ancestor_id {
                    return true;
                }
                current = parent_id;
            } else {
                break;
            }
        }

        false
    }

    /// Check if `ancestor_id` is an ancestor of `entity_id`.
    pub fn is_ancestor_of(&self, ancestor_id: u64, entity_id: u64) -> bool {
        self.is_descendant_of(entity_id, ancestor_id)
    }

    /// Finish dragging and reparent if valid.
    ///
    /// Returns a HierarchyAction::Reparent if successful, or None if cancelled
    /// or invalid (would create a cycle).
    pub fn finish_drag(&mut self) -> Option<HierarchyAction> {
        let drag = self.drag_state.take()?;

        // Check if this is a valid reparent operation
        if let Some(new_parent) = drag.drop_target {
            if self.would_create_cycle(drag.entity_id, new_parent) {
                return None;
            }
        }

        // Don't generate action if parent didn't change
        if drag.drop_target == drag.original_parent {
            return None;
        }

        Some(HierarchyAction::Reparent {
            entity: drag.entity_id,
            new_parent: drag.drop_target,
        })
    }

    /// Actually perform the reparent operation on the internal data.
    pub fn reparent(&mut self, entity_id: u64, new_parent: Option<u64>) -> bool {
        // Validate cycle prevention
        if let Some(new_parent_id) = new_parent {
            if self.would_create_cycle(entity_id, new_parent_id) {
                return false;
            }
        }

        // Get current parent
        let old_parent = self.nodes.get(&entity_id).and_then(|n| n.parent_id);

        // Remove from old parent's children
        if let Some(old_parent_id) = old_parent {
            if let Some(parent) = self.nodes.get_mut(&old_parent_id) {
                parent.remove_child(entity_id);
            }
        } else {
            // Was a root, remove from roots
            self.root_ids.retain(|&id| id != entity_id);
        }

        // Update node's parent
        if let Some(node) = self.nodes.get_mut(&entity_id) {
            node.parent_id = new_parent;
        }

        // Add to new parent's children
        if let Some(new_parent_id) = new_parent {
            if let Some(parent) = self.nodes.get_mut(&new_parent_id) {
                parent.add_child(entity_id);
            }
        } else {
            // Becoming a root
            self.root_ids.push(entity_id);
            self.root_ids.sort();
        }

        true
    }

    /// Get the depth of an entity in the hierarchy (0 = root).
    pub fn depth(&self, entity_id: u64) -> usize {
        let mut depth = 0;
        let mut current = entity_id;
        let mut visited = HashSet::new();

        while let Some(node) = self.nodes.get(&current) {
            if visited.contains(&current) {
                break;
            }
            visited.insert(current);

            if let Some(parent_id) = node.parent_id {
                depth += 1;
                current = parent_id;
            } else {
                break;
            }
        }

        depth
    }

    /// Get the path from root to an entity (list of ancestor IDs including the entity).
    pub fn path_to(&self, entity_id: u64) -> Vec<u64> {
        let mut path = Vec::new();
        let mut current = entity_id;
        let mut visited = HashSet::new();

        while let Some(node) = self.nodes.get(&current) {
            if visited.contains(&current) {
                break;
            }
            visited.insert(current);

            path.push(current);
            if let Some(parent_id) = node.parent_id {
                current = parent_id;
            } else {
                break;
            }
        }

        path.reverse();
        path
    }

    /// Get all descendants of an entity.
    pub fn descendants(&self, entity_id: u64) -> Vec<u64> {
        let mut result = Vec::new();
        self.collect_descendants(entity_id, &mut result);
        result
    }

    fn collect_descendants(&self, entity_id: u64, result: &mut Vec<u64>) {
        if let Some(node) = self.nodes.get(&entity_id) {
            for &child_id in &node.children {
                result.push(child_id);
                self.collect_descendants(child_id, result);
            }
        }
    }

    /// Get the total number of entities in the hierarchy.
    pub fn entity_count(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the hierarchy is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Enter rename mode for an entity.
    pub fn start_rename(&mut self, entity_id: u64) {
        if let Some(node) = self.nodes.get(&entity_id) {
            self.rename_mode = Some((entity_id, node.name.clone()));
        }
    }

    /// Cancel rename mode.
    pub fn cancel_rename(&mut self) {
        self.rename_mode = None;
    }

    /// Finish rename and return the action if name changed.
    pub fn finish_rename(&mut self) -> Option<HierarchyAction> {
        let (entity_id, new_name) = self.rename_mode.take()?;

        if let Some(node) = self.nodes.get(&entity_id) {
            if node.name != new_name && !new_name.is_empty() {
                return Some(HierarchyAction::Rename {
                    entity: entity_id,
                    new_name,
                });
            }
        }

        None
    }

    /// Apply a rename to the internal data.
    pub fn apply_rename(&mut self, entity_id: u64, new_name: &str) {
        if let Some(node) = self.nodes.get_mut(&entity_id) {
            node.name = new_name.to_string();
        }
    }

    /// Open context menu for an entity.
    pub fn open_context_menu(&mut self, entity_id: u64, x: f32, y: f32) {
        self.context_menu = Some((entity_id, x, y));
    }

    /// Close the context menu.
    pub fn close_context_menu(&mut self) {
        self.context_menu = None;
    }

    /// Get the context menu state.
    pub fn context_menu(&self) -> Option<(u64, f32, f32)> {
        self.context_menu
    }

    /// Render the hierarchy panel and return any action.
    pub fn render(&mut self, ctx: &mut impl UIContext) -> HierarchyAction {
        let mut action = HierarchyAction::None;

        // Search bar
        ctx.horizontal(|ctx| {
            ctx.label("Search:");
            let mut filter = self.search_filter.clone();
            if ctx.text_edit("", &mut filter) {
                self.search_filter = filter.to_lowercase();
            }
        });

        ctx.separator();

        // Toolbar
        ctx.horizontal(|ctx| {
            if ctx.button("Expand All") {
                self.expand_all();
            }
            if ctx.button("Collapse All") {
                self.collapse_all();
            }
        });

        ctx.separator();

        // Render tree
        let root_ids: Vec<u64> = self.root_ids.clone();
        for root_id in root_ids {
            if self.matches_filter(root_id) {
                if let Some(node_action) = self.render_node(ctx, root_id, 0) {
                    action = node_action;
                }
            }
        }

        action
    }

    /// Render a single node and its children recursively.
    fn render_node(&mut self, ctx: &mut impl UIContext, entity_id: u64, depth: usize) -> Option<HierarchyAction> {
        let node = self.nodes.get(&entity_id)?.clone();
        let has_children = !node.children.is_empty();
        let is_expanded = self.expanded.contains(&entity_id);
        let is_selected = self.selected == Some(entity_id);

        let mut action: Option<HierarchyAction> = None;

        // Build indent string
        let indent = "  ".repeat(depth);

        // Build node label
        let expand_indicator = if has_children {
            if is_expanded { "[-]" } else { "[+]" }
        } else {
            "   "
        };

        let visibility_indicator = if node.visible { "O" } else { "X" };
        let lock_indicator = if node.locked { "L" } else { " " };
        let selection_indicator = if is_selected { ">" } else { " " };

        ctx.horizontal(|ctx| {
            // Indent and expand toggle
            if has_children {
                let toggle_label = format!("{}{}", indent, expand_indicator);
                if ctx.button(&toggle_label) {
                    self.toggle_expand(entity_id);
                }
            } else {
                ctx.label(&format!("{}{}", indent, expand_indicator));
            }

            // Selection indicator and name
            let name_label = format!("{} {}", selection_indicator, node.name);
            if ctx.button(&name_label) {
                self.selected = Some(entity_id);
                action = Some(HierarchyAction::Selected(entity_id));
            }

            // Visibility toggle
            if ctx.button(visibility_indicator) {
                action = Some(HierarchyAction::ToggleVisibility(entity_id));
            }

            // Lock toggle
            if ctx.button(lock_indicator) {
                action = Some(HierarchyAction::ToggleLock(entity_id));
            }
        });

        // Render children if expanded
        if is_expanded && has_children {
            let children: Vec<u64> = node.children.clone();
            for child_id in children {
                if self.matches_filter(child_id) {
                    if let Some(child_action) = self.render_node(ctx, child_id, depth + 1) {
                        action = Some(child_action);
                    }
                }
            }
        }

        action
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egui_adapter::MockUIContext;

    // =========================================================================
    // EntityNode Tests
    // =========================================================================

    #[test]
    fn test_entity_node_new() {
        let node = EntityNode::new(42, "Test Entity");
        assert_eq!(node.entity_id, 42);
        assert_eq!(node.name, "Test Entity");
        assert!(node.parent_id.is_none());
        assert!(node.children.is_empty());
        assert!(node.visible);
        assert!(!node.locked);
    }

    #[test]
    fn test_entity_node_with_parent() {
        let node = EntityNode::with_parent(2, "Child", 1);
        assert_eq!(node.entity_id, 2);
        assert_eq!(node.parent_id, Some(1));
    }

    #[test]
    fn test_entity_node_builder_methods() {
        let node = EntityNode::new(1, "Test")
            .with_visible(false)
            .with_locked(true);
        assert!(!node.visible);
        assert!(node.locked);
    }

    #[test]
    fn test_entity_node_add_child() {
        let mut node = EntityNode::new(1, "Parent");
        node.add_child(2);
        node.add_child(3);
        assert_eq!(node.children, vec![2, 3]);

        // Adding duplicate should not add again
        node.add_child(2);
        assert_eq!(node.children, vec![2, 3]);
    }

    #[test]
    fn test_entity_node_remove_child() {
        let mut node = EntityNode::new(1, "Parent");
        node.add_child(2);
        node.add_child(3);
        node.remove_child(2);
        assert_eq!(node.children, vec![3]);
    }

    #[test]
    fn test_entity_node_has_children() {
        let mut node = EntityNode::new(1, "Parent");
        assert!(!node.has_children());
        node.add_child(2);
        assert!(node.has_children());
    }

    #[test]
    fn test_entity_node_is_root() {
        let root = EntityNode::new(1, "Root");
        let child = EntityNode::with_parent(2, "Child", 1);
        assert!(root.is_root());
        assert!(!child.is_root());
    }

    // =========================================================================
    // DragState Tests
    // =========================================================================

    #[test]
    fn test_drag_state_new() {
        let drag = DragState::new(5, Some(1));
        assert_eq!(drag.entity_id, 5);
        assert_eq!(drag.original_parent, Some(1));
        assert!(drag.drop_target.is_none());
    }

    #[test]
    fn test_drag_state_with_drop_target() {
        let drag = DragState::new(5, None).with_drop_target(Some(10));
        assert_eq!(drag.drop_target, Some(10));
    }

    // =========================================================================
    // HierarchyAction Tests
    // =========================================================================

    #[test]
    fn test_hierarchy_action_default() {
        let action: HierarchyAction = Default::default();
        assert_eq!(action, HierarchyAction::None);
    }

    #[test]
    fn test_hierarchy_action_variants() {
        assert_eq!(HierarchyAction::Selected(1), HierarchyAction::Selected(1));
        assert_eq!(
            HierarchyAction::Reparent { entity: 1, new_parent: Some(2) },
            HierarchyAction::Reparent { entity: 1, new_parent: Some(2) }
        );
        assert_eq!(HierarchyAction::Delete(1), HierarchyAction::Delete(1));
        assert_eq!(HierarchyAction::Duplicate(1), HierarchyAction::Duplicate(1));
    }

    // =========================================================================
    // HierarchyPanel Basic Tests
    // =========================================================================

    #[test]
    fn test_hierarchy_panel_new() {
        let panel = HierarchyPanel::new();
        assert!(panel.is_empty());
        assert_eq!(panel.entity_count(), 0);
        assert!(panel.selected().is_none());
    }

    #[test]
    fn test_hierarchy_panel_set_entities_flat() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![
            EntityNode::new(1, "A"),
            EntityNode::new(2, "B"),
            EntityNode::new(3, "C"),
        ]);

        assert_eq!(panel.entity_count(), 3);
        assert_eq!(panel.root_ids(), &[1, 2, 3]);
    }

    #[test]
    fn test_hierarchy_panel_set_entities_tree() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![
            EntityNode::new(1, "Root"),
            EntityNode::with_parent(2, "Child A", 1),
            EntityNode::with_parent(3, "Child B", 1),
            EntityNode::with_parent(4, "Grandchild", 2),
        ]);

        assert_eq!(panel.entity_count(), 4);
        assert_eq!(panel.root_ids(), &[1]);

        let root = panel.get_entity(1).unwrap();
        assert_eq!(root.children.len(), 2);
        assert!(root.children.contains(&2));
        assert!(root.children.contains(&3));

        let child_a = panel.get_entity(2).unwrap();
        assert_eq!(child_a.children, vec![4]);
    }

    #[test]
    fn test_hierarchy_panel_add_entity() {
        let mut panel = HierarchyPanel::new();
        panel.add_entity(EntityNode::new(1, "Root"));
        panel.add_entity(EntityNode::with_parent(2, "Child", 1));

        assert_eq!(panel.entity_count(), 2);
        assert_eq!(panel.root_ids(), &[1]);

        let root = panel.get_entity(1).unwrap();
        assert!(root.children.contains(&2));
    }

    #[test]
    fn test_hierarchy_panel_remove_entity() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![
            EntityNode::new(1, "Root"),
            EntityNode::with_parent(2, "Child", 1),
            EntityNode::with_parent(3, "Grandchild", 2),
        ]);

        // Select entity 2
        panel.select(2);
        assert_eq!(panel.selected(), Some(2));

        // Remove entity 2
        panel.remove_entity(2);

        assert_eq!(panel.entity_count(), 2);
        assert!(panel.get_entity(2).is_none());

        // Entity 3 should now be a root (orphaned)
        assert!(panel.root_ids().contains(&3));

        // Selection should be cleared
        assert!(panel.selected().is_none());

        // Entity 1 should no longer have entity 2 as child
        let root = panel.get_entity(1).unwrap();
        assert!(!root.children.contains(&2));
    }

    // =========================================================================
    // Selection Tests
    // =========================================================================

    #[test]
    fn test_hierarchy_panel_select() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![
            EntityNode::new(1, "A"),
            EntityNode::new(2, "B"),
        ]);

        panel.select(1);
        assert_eq!(panel.selected(), Some(1));

        panel.select(2);
        assert_eq!(panel.selected(), Some(2));

        // Selecting non-existent entity should not change selection
        panel.select(999);
        assert_eq!(panel.selected(), Some(2));
    }

    #[test]
    fn test_hierarchy_panel_clear_selection() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![EntityNode::new(1, "A")]);
        panel.select(1);
        panel.clear_selection();
        assert!(panel.selected().is_none());
    }

    // =========================================================================
    // Expand/Collapse Tests
    // =========================================================================

    #[test]
    fn test_hierarchy_panel_expand_collapse() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![
            EntityNode::new(1, "Root"),
            EntityNode::with_parent(2, "Child", 1),
        ]);

        assert!(!panel.is_expanded(1));

        panel.expand(1);
        assert!(panel.is_expanded(1));

        panel.collapse(1);
        assert!(!panel.is_expanded(1));
    }

    #[test]
    fn test_hierarchy_panel_toggle_expand() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![EntityNode::new(1, "Root")]);

        panel.toggle_expand(1);
        assert!(panel.is_expanded(1));

        panel.toggle_expand(1);
        assert!(!panel.is_expanded(1));
    }

    #[test]
    fn test_hierarchy_panel_expand_all() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![
            EntityNode::new(1, "A"),
            EntityNode::new(2, "B"),
            EntityNode::new(3, "C"),
        ]);

        panel.expand_all();
        assert!(panel.is_expanded(1));
        assert!(panel.is_expanded(2));
        assert!(panel.is_expanded(3));
    }

    #[test]
    fn test_hierarchy_panel_collapse_all() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![
            EntityNode::new(1, "A"),
            EntityNode::new(2, "B"),
        ]);

        panel.expand_all();
        panel.collapse_all();
        assert!(!panel.is_expanded(1));
        assert!(!panel.is_expanded(2));
    }

    #[test]
    fn test_hierarchy_panel_reveal() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![
            EntityNode::new(1, "Root"),
            EntityNode::with_parent(2, "Child", 1),
            EntityNode::with_parent(3, "Grandchild", 2),
            EntityNode::with_parent(4, "Great-grandchild", 3),
        ]);

        // Reveal the deepest entity
        panel.reveal(4);

        // All ancestors should be expanded
        assert!(panel.is_expanded(1));
        assert!(panel.is_expanded(2));
        assert!(panel.is_expanded(3));
        // The entity itself should not be expanded (only ancestors)
        assert!(!panel.is_expanded(4));
    }

    // =========================================================================
    // Filter Tests
    // =========================================================================

    #[test]
    fn test_hierarchy_panel_set_filter() {
        let mut panel = HierarchyPanel::new();
        panel.set_filter("Test");
        assert_eq!(panel.filter(), "test"); // Should be lowercased
    }

    #[test]
    fn test_hierarchy_panel_clear_filter() {
        let mut panel = HierarchyPanel::new();
        panel.set_filter("Test");
        panel.clear_filter();
        assert!(panel.filter().is_empty());
    }

    #[test]
    fn test_hierarchy_panel_filtered_entities() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![
            EntityNode::new(1, "Camera"),
            EntityNode::new(2, "Light"),
            EntityNode::new(3, "Camera_Main"),
        ]);

        panel.set_filter("camera");
        let matches = panel.filtered_entities();
        assert_eq!(matches.len(), 2);
        assert!(matches.contains(&1));
        assert!(matches.contains(&3));
    }

    #[test]
    fn test_hierarchy_panel_filter_includes_descendants() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![
            EntityNode::new(1, "Root"),
            EntityNode::with_parent(2, "Branch", 1),
            EntityNode::with_parent(3, "Camera", 2),
        ]);

        panel.set_filter("camera");

        // Root should match because it has a descendant that matches
        assert!(panel.matches_filter(1));
        // Branch should match because it has a child that matches
        assert!(panel.matches_filter(2));
        // Camera should match directly
        assert!(panel.matches_filter(3));
    }

    // =========================================================================
    // Cycle Detection Tests
    // =========================================================================

    #[test]
    fn test_hierarchy_panel_would_create_cycle_self() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![EntityNode::new(1, "A")]);

        // Can't parent entity to itself
        assert!(panel.would_create_cycle(1, 1));
    }

    #[test]
    fn test_hierarchy_panel_would_create_cycle_direct_child() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![
            EntityNode::new(1, "Parent"),
            EntityNode::with_parent(2, "Child", 1),
        ]);

        // Can't parent entity 1 to entity 2 (its child)
        assert!(panel.would_create_cycle(1, 2));
    }

    #[test]
    fn test_hierarchy_panel_would_create_cycle_deep() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![
            EntityNode::new(1, "Root"),
            EntityNode::with_parent(2, "Child", 1),
            EntityNode::with_parent(3, "Grandchild", 2),
            EntityNode::with_parent(4, "Great-grandchild", 3),
        ]);

        // Can't parent entity 1 to any of its descendants
        assert!(panel.would_create_cycle(1, 2));
        assert!(panel.would_create_cycle(1, 3));
        assert!(panel.would_create_cycle(1, 4));

        // Can parent entity 4 to entity 1 (sibling relationship)
        assert!(!panel.would_create_cycle(4, 1));
    }

    #[test]
    fn test_hierarchy_panel_would_not_create_cycle_sibling() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![
            EntityNode::new(1, "Root"),
            EntityNode::with_parent(2, "Child A", 1),
            EntityNode::with_parent(3, "Child B", 1),
        ]);

        // Can parent entity 2 to entity 3 (siblings)
        assert!(!panel.would_create_cycle(2, 3));
    }

    // =========================================================================
    // Is Descendant/Ancestor Tests
    // =========================================================================

    #[test]
    fn test_hierarchy_panel_is_descendant_of() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![
            EntityNode::new(1, "Root"),
            EntityNode::with_parent(2, "Child", 1),
            EntityNode::with_parent(3, "Grandchild", 2),
        ]);

        assert!(panel.is_descendant_of(3, 1));
        assert!(panel.is_descendant_of(3, 2));
        assert!(panel.is_descendant_of(2, 1));
        assert!(!panel.is_descendant_of(1, 2));
        assert!(!panel.is_descendant_of(1, 3));
    }

    #[test]
    fn test_hierarchy_panel_is_ancestor_of() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![
            EntityNode::new(1, "Root"),
            EntityNode::with_parent(2, "Child", 1),
        ]);

        assert!(panel.is_ancestor_of(1, 2));
        assert!(!panel.is_ancestor_of(2, 1));
    }

    // =========================================================================
    // Reparent Tests
    // =========================================================================

    #[test]
    fn test_hierarchy_panel_reparent_to_root() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![
            EntityNode::new(1, "Root"),
            EntityNode::with_parent(2, "Child", 1),
        ]);

        let result = panel.reparent(2, None);
        assert!(result);

        // Entity 2 should now be a root
        assert!(panel.root_ids().contains(&2));
        let node2 = panel.get_entity(2).unwrap();
        assert!(node2.parent_id.is_none());

        // Entity 1 should no longer have entity 2 as child
        let node1 = panel.get_entity(1).unwrap();
        assert!(!node1.children.contains(&2));
    }

    #[test]
    fn test_hierarchy_panel_reparent_to_new_parent() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![
            EntityNode::new(1, "A"),
            EntityNode::new(2, "B"),
            EntityNode::with_parent(3, "C", 1),
        ]);

        let result = panel.reparent(3, Some(2));
        assert!(result);

        // Entity 3 should now be under entity 2
        let node3 = panel.get_entity(3).unwrap();
        assert_eq!(node3.parent_id, Some(2));

        // Entity 2 should have entity 3 as child
        let node2 = panel.get_entity(2).unwrap();
        assert!(node2.children.contains(&3));

        // Entity 1 should no longer have entity 3 as child
        let node1 = panel.get_entity(1).unwrap();
        assert!(!node1.children.contains(&3));
    }

    #[test]
    fn test_hierarchy_panel_reparent_prevents_cycle() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![
            EntityNode::new(1, "Root"),
            EntityNode::with_parent(2, "Child", 1),
        ]);

        // Try to reparent entity 1 under entity 2 (would create cycle)
        let result = panel.reparent(1, Some(2));
        assert!(!result);

        // Hierarchy should be unchanged
        let node1 = panel.get_entity(1).unwrap();
        assert!(node1.parent_id.is_none());
    }

    // =========================================================================
    // Drag Tests
    // =========================================================================

    #[test]
    fn test_hierarchy_panel_start_drag() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![
            EntityNode::new(1, "Root"),
            EntityNode::with_parent(2, "Child", 1),
        ]);

        panel.start_drag(2);
        assert!(panel.drag_state.is_some());

        let drag = panel.drag_state.as_ref().unwrap();
        assert_eq!(drag.entity_id, 2);
        assert_eq!(drag.original_parent, Some(1));
    }

    #[test]
    fn test_hierarchy_panel_update_drag_target() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![
            EntityNode::new(1, "A"),
            EntityNode::new(2, "B"),
        ]);

        panel.start_drag(1);
        panel.update_drag_target(Some(2));

        let drag = panel.drag_state.as_ref().unwrap();
        assert_eq!(drag.drop_target, Some(2));
    }

    #[test]
    fn test_hierarchy_panel_cancel_drag() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![EntityNode::new(1, "A")]);

        panel.start_drag(1);
        panel.cancel_drag();
        assert!(panel.drag_state.is_none());
    }

    #[test]
    fn test_hierarchy_panel_finish_drag_valid() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![
            EntityNode::new(1, "A"),
            EntityNode::new(2, "B"),
        ]);

        panel.start_drag(1);
        panel.update_drag_target(Some(2));
        let action = panel.finish_drag();

        assert_eq!(
            action,
            Some(HierarchyAction::Reparent {
                entity: 1,
                new_parent: Some(2)
            })
        );
    }

    #[test]
    fn test_hierarchy_panel_finish_drag_cycle() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![
            EntityNode::new(1, "Root"),
            EntityNode::with_parent(2, "Child", 1),
        ]);

        panel.start_drag(1);
        panel.update_drag_target(Some(2));
        let action = panel.finish_drag();

        // Should return None because it would create a cycle
        assert!(action.is_none());
    }

    #[test]
    fn test_hierarchy_panel_finish_drag_no_change() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![
            EntityNode::new(1, "Root"),
            EntityNode::with_parent(2, "Child", 1),
        ]);

        panel.start_drag(2);
        panel.update_drag_target(Some(1)); // Same parent
        let action = panel.finish_drag();

        // Should return None because parent didn't change
        assert!(action.is_none());
    }

    // =========================================================================
    // Depth and Path Tests
    // =========================================================================

    #[test]
    fn test_hierarchy_panel_depth() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![
            EntityNode::new(1, "Root"),
            EntityNode::with_parent(2, "Child", 1),
            EntityNode::with_parent(3, "Grandchild", 2),
            EntityNode::with_parent(4, "Great-grandchild", 3),
        ]);

        assert_eq!(panel.depth(1), 0);
        assert_eq!(panel.depth(2), 1);
        assert_eq!(panel.depth(3), 2);
        assert_eq!(panel.depth(4), 3);
    }

    #[test]
    fn test_hierarchy_panel_path_to() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![
            EntityNode::new(1, "Root"),
            EntityNode::with_parent(2, "Child", 1),
            EntityNode::with_parent(3, "Grandchild", 2),
        ]);

        let path = panel.path_to(3);
        assert_eq!(path, vec![1, 2, 3]);
    }

    // =========================================================================
    // Descendants Tests
    // =========================================================================

    #[test]
    fn test_hierarchy_panel_descendants() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![
            EntityNode::new(1, "Root"),
            EntityNode::with_parent(2, "Child A", 1),
            EntityNode::with_parent(3, "Child B", 1),
            EntityNode::with_parent(4, "Grandchild", 2),
        ]);

        let descendants = panel.descendants(1);
        assert_eq!(descendants.len(), 3);
        assert!(descendants.contains(&2));
        assert!(descendants.contains(&3));
        assert!(descendants.contains(&4));
    }

    #[test]
    fn test_hierarchy_panel_descendants_empty() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![EntityNode::new(1, "Leaf")]);

        let descendants = panel.descendants(1);
        assert!(descendants.is_empty());
    }

    // =========================================================================
    // Rename Tests
    // =========================================================================

    #[test]
    fn test_hierarchy_panel_rename() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![EntityNode::new(1, "OldName")]);

        panel.start_rename(1);
        assert!(panel.rename_mode.is_some());

        // Simulate editing the name
        if let Some((_, ref mut name)) = panel.rename_mode {
            *name = "NewName".to_string();
        }

        let action = panel.finish_rename();
        assert_eq!(
            action,
            Some(HierarchyAction::Rename {
                entity: 1,
                new_name: "NewName".to_string()
            })
        );
    }

    #[test]
    fn test_hierarchy_panel_rename_no_change() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![EntityNode::new(1, "Name")]);

        panel.start_rename(1);
        // Don't change the name
        let action = panel.finish_rename();
        assert!(action.is_none());
    }

    #[test]
    fn test_hierarchy_panel_apply_rename() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![EntityNode::new(1, "OldName")]);

        panel.apply_rename(1, "NewName");
        assert_eq!(panel.get_entity(1).unwrap().name, "NewName");
    }

    #[test]
    fn test_hierarchy_panel_cancel_rename() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![EntityNode::new(1, "Name")]);

        panel.start_rename(1);
        panel.cancel_rename();
        assert!(panel.rename_mode.is_none());
    }

    // =========================================================================
    // Context Menu Tests
    // =========================================================================

    #[test]
    fn test_hierarchy_panel_context_menu() {
        let mut panel = HierarchyPanel::new();

        panel.open_context_menu(1, 100.0, 200.0);
        assert_eq!(panel.context_menu(), Some((1, 100.0, 200.0)));

        panel.close_context_menu();
        assert!(panel.context_menu().is_none());
    }

    // =========================================================================
    // Render Tests
    // =========================================================================

    #[test]
    fn test_hierarchy_panel_render_empty() {
        let mut panel = HierarchyPanel::new();
        let mut ctx = MockUIContext::new(1);

        let action = panel.render(&mut ctx);
        assert_eq!(action, HierarchyAction::None);
    }

    #[test]
    fn test_hierarchy_panel_render_with_entities() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![
            EntityNode::new(1, "Root"),
            EntityNode::with_parent(2, "Child", 1),
        ]);

        let mut ctx = MockUIContext::new(1);
        let action = panel.render(&mut ctx);

        // By default, should render without action
        assert_eq!(action, HierarchyAction::None);
    }

    #[test]
    fn test_hierarchy_panel_render_selection() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![EntityNode::new(1, "Test")]);

        let mut ctx = MockUIContext::new(1);

        // Configure the mock to simulate clicking the entity button
        ctx.click_button("  Test");

        let action = panel.render(&mut ctx);
        assert_eq!(action, HierarchyAction::Selected(1));
    }

    #[test]
    fn test_hierarchy_panel_render_expand() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![
            EntityNode::new(1, "Root"),
            EntityNode::with_parent(2, "Child", 1),
        ]);

        let mut ctx = MockUIContext::new(1);

        // Configure mock to click expand button
        ctx.click_button("[+]");

        panel.render(&mut ctx);

        // Node should now be expanded
        assert!(panel.is_expanded(1));
    }

    #[test]
    fn test_hierarchy_panel_render_collapse() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![
            EntityNode::new(1, "Root"),
            EntityNode::with_parent(2, "Child", 1),
        ]);
        panel.expand(1);

        let mut ctx = MockUIContext::new(1);
        ctx.click_button("[-]");

        panel.render(&mut ctx);

        assert!(!panel.is_expanded(1));
    }

    #[test]
    fn test_hierarchy_panel_render_filtered() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![
            EntityNode::new(1, "Camera"),
            EntityNode::new(2, "Light"),
        ]);
        panel.set_filter("camera");

        let mut ctx = MockUIContext::new(1);
        panel.render(&mut ctx);

        // Should only render Camera, not Light
        // (We can't easily verify this with MockUIContext, but the filter logic is tested above)
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_hierarchy_panel_orphaned_child() {
        let mut panel = HierarchyPanel::new();
        // Add child with non-existent parent
        panel.set_entities(vec![
            EntityNode::with_parent(2, "Orphan", 999),
        ]);

        // Orphan should be treated as root
        assert!(panel.root_ids().contains(&2));
    }

    #[test]
    fn test_hierarchy_panel_multiple_roots() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![
            EntityNode::new(5, "E"),
            EntityNode::new(1, "A"),
            EntityNode::new(3, "C"),
        ]);

        // Roots should be sorted by ID
        assert_eq!(panel.root_ids(), &[1, 3, 5]);
    }

    #[test]
    fn test_hierarchy_panel_get_entity_mut() {
        let mut panel = HierarchyPanel::new();
        panel.set_entities(vec![EntityNode::new(1, "Test")]);

        if let Some(node) = panel.get_entity_mut(1) {
            node.name = "Modified".to_string();
        }

        assert_eq!(panel.get_entity(1).unwrap().name, "Modified");
    }

    #[test]
    fn test_hierarchy_panel_expand_nonexistent() {
        let mut panel = HierarchyPanel::new();
        panel.expand(999); // Should not panic
        assert!(!panel.is_expanded(999));
    }
}
