//! Inspector Multi-Select and Comparison (T-TL-2.5)
//!
//! Provides multi-entity selection with side-by-side component diffing.
//! Supports comparing components across two selected entities, highlighting
//! differences, and copying field values between entities.
//!
//! # Architecture
//!
//! ```text
//! MultiSelection              DiffPanel                 UIContext
//! ==============              =========                 =========
//!     |                           |                         |
//!     +-- primary (entity A)      |                         |
//!     +-- secondary (entity B)    |                         |
//!     |                           |                         |
//!     +--- set_entities() ------->+                         |
//!                                 +--- compute_diff() ----->|
//!                                 |                         |
//!                                 +--- render() ----------->| (side-by-side view)
//!                                 |                         |
//!                                 +--- copy_field() ------->| (copy values)
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use renderer_backend::inspector_diff::{MultiSelection, DiffPanel, SelectionMode};
//!
//! let mut selection = MultiSelection::new();
//! selection.set_primary(Some(100));
//! selection.set_secondary(Some(200));
//! selection.mode = SelectionMode::Compare;
//!
//! let mut diff_panel = DiffPanel::new();
//! diff_panel.set_entities(100, 200);
//!
//! let left_components = vec![(1, "Transform".to_string(), transform_bytes_a)];
//! let right_components = vec![(1, "Transform".to_string(), transform_bytes_b)];
//!
//! diff_panel.compute_diff(&left_components, &right_components);
//! diff_panel.render(&mut ctx);
//! ```

use crate::egui_adapter::UIContext;
use crate::inspector_panel::{TypeDecoder, Value};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Selection Types
// ---------------------------------------------------------------------------

/// Mode of entity selection in the inspector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum SelectionMode {
    /// Single entity selection (default).
    #[default]
    Single,
    /// Compare mode: two entities side-by-side.
    Compare,
    /// Multi-select mode: multiple entities selected.
    Multi,
}

/// Tracks multi-entity selection state.
#[derive(Debug, Clone, Default)]
pub struct MultiSelection {
    /// Primary selected entity (left side in compare mode).
    pub primary: Option<u64>,
    /// Secondary selected entity (right side in compare mode).
    pub secondary: Option<u64>,
    /// Additional selected entities (for Multi mode).
    pub additional: Vec<u64>,
    /// Current selection mode.
    pub mode: SelectionMode,
}

impl MultiSelection {
    /// Create a new empty selection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the primary selected entity.
    pub fn set_primary(&mut self, entity_id: Option<u64>) {
        self.primary = entity_id;
        // If we have a primary and secondary, ensure compare mode
        if self.primary.is_some() && self.secondary.is_some() {
            self.mode = SelectionMode::Compare;
        } else if self.primary.is_none() && self.secondary.is_none() {
            self.mode = SelectionMode::Single;
        }
    }

    /// Set the secondary selected entity.
    pub fn set_secondary(&mut self, entity_id: Option<u64>) {
        self.secondary = entity_id;
        if self.primary.is_some() && self.secondary.is_some() {
            self.mode = SelectionMode::Compare;
        }
    }

    /// Add an entity to the selection (for Multi mode).
    pub fn add_entity(&mut self, entity_id: u64) {
        if !self.additional.contains(&entity_id)
            && self.primary != Some(entity_id)
            && self.secondary != Some(entity_id)
        {
            self.additional.push(entity_id);
            if !self.additional.is_empty() {
                self.mode = SelectionMode::Multi;
            }
        }
    }

    /// Remove an entity from the selection.
    pub fn remove_entity(&mut self, entity_id: u64) {
        if self.primary == Some(entity_id) {
            self.primary = None;
        } else if self.secondary == Some(entity_id) {
            self.secondary = None;
        } else {
            self.additional.retain(|&id| id != entity_id);
        }
        self.update_mode();
    }

    /// Toggle an entity's selection state.
    pub fn toggle_entity(&mut self, entity_id: u64) {
        if self.contains(entity_id) {
            self.remove_entity(entity_id);
        } else {
            // Add as secondary if primary exists, otherwise as primary
            if self.primary.is_none() {
                self.set_primary(Some(entity_id));
            } else if self.secondary.is_none() {
                self.set_secondary(Some(entity_id));
            } else {
                self.add_entity(entity_id);
            }
        }
    }

    /// Check if an entity is selected.
    pub fn contains(&self, entity_id: u64) -> bool {
        self.primary == Some(entity_id)
            || self.secondary == Some(entity_id)
            || self.additional.contains(&entity_id)
    }

    /// Get all selected entity IDs.
    pub fn all_entities(&self) -> Vec<u64> {
        let mut entities = Vec::new();
        if let Some(id) = self.primary {
            entities.push(id);
        }
        if let Some(id) = self.secondary {
            entities.push(id);
        }
        entities.extend_from_slice(&self.additional);
        entities
    }

    /// Get the count of selected entities.
    pub fn count(&self) -> usize {
        let mut count = 0;
        if self.primary.is_some() {
            count += 1;
        }
        if self.secondary.is_some() {
            count += 1;
        }
        count += self.additional.len();
        count
    }

    /// Clear all selections.
    pub fn clear(&mut self) {
        self.primary = None;
        self.secondary = None;
        self.additional.clear();
        self.mode = SelectionMode::Single;
    }

    /// Check if in compare mode with both entities set.
    pub fn is_comparing(&self) -> bool {
        self.mode == SelectionMode::Compare && self.primary.is_some() && self.secondary.is_some()
    }

    /// Swap primary and secondary entities.
    pub fn swap(&mut self) {
        std::mem::swap(&mut self.primary, &mut self.secondary);
    }

    /// Update mode based on current selection state.
    fn update_mode(&mut self) {
        if !self.additional.is_empty() {
            self.mode = SelectionMode::Multi;
        } else if self.primary.is_some() && self.secondary.is_some() {
            self.mode = SelectionMode::Compare;
        } else {
            self.mode = SelectionMode::Single;
        }
    }
}

// ---------------------------------------------------------------------------
// Diff Types
// ---------------------------------------------------------------------------

/// Status of a diff comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum DiffStatus {
    /// Values are the same.
    #[default]
    Same,
    /// Values are different.
    Modified,
    /// Only present on the left side.
    OnlyLeft,
    /// Only present on the right side.
    OnlyRight,
}

impl DiffStatus {
    /// Get the display color for this status (RGBA).
    pub fn color(&self) -> [f32; 4] {
        match self {
            DiffStatus::Same => [0.7, 0.7, 0.7, 1.0],       // Gray
            DiffStatus::Modified => [1.0, 0.8, 0.2, 1.0],   // Yellow
            DiffStatus::OnlyLeft => [0.2, 0.6, 1.0, 1.0],   // Blue
            DiffStatus::OnlyRight => [0.2, 1.0, 0.4, 1.0],  // Green
        }
    }

    /// Get a status indicator character.
    pub fn indicator(&self) -> char {
        match self {
            DiffStatus::Same => '=',
            DiffStatus::Modified => '!',
            DiffStatus::OnlyLeft => '<',
            DiffStatus::OnlyRight => '>',
        }
    }

    /// Check if this status represents a difference.
    pub fn is_different(&self) -> bool {
        !matches!(self, DiffStatus::Same)
    }
}

/// Which side of a diff comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Side {
    /// Left side (primary entity).
    Left,
    /// Right side (secondary entity).
    Right,
}

impl Side {
    /// Get the opposite side.
    pub fn opposite(&self) -> Side {
        match self {
            Side::Left => Side::Right,
            Side::Right => Side::Left,
        }
    }
}

/// Represents a field-level difference between two values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldDiff {
    /// Path to the field (e.g., "position.x" or "rotation.w").
    pub field_path: String,
    /// Value on the left side (primary entity), if present.
    pub left_value: Option<Value>,
    /// Value on the right side (secondary entity), if present.
    pub right_value: Option<Value>,
    /// Status of this field comparison.
    pub status: DiffStatus,
}

impl FieldDiff {
    /// Create a new field diff.
    pub fn new(
        field_path: impl Into<String>,
        left_value: Option<Value>,
        right_value: Option<Value>,
    ) -> Self {
        let status = Self::compute_status(&left_value, &right_value);
        Self {
            field_path: field_path.into(),
            left_value,
            right_value,
            status,
        }
    }

    /// Compute the diff status from two optional values.
    fn compute_status(left: &Option<Value>, right: &Option<Value>) -> DiffStatus {
        match (left, right) {
            (None, None) => DiffStatus::Same,
            (Some(_), None) => DiffStatus::OnlyLeft,
            (None, Some(_)) => DiffStatus::OnlyRight,
            (Some(l), Some(r)) => {
                if l.approx_eq(r, 0.0001) {
                    DiffStatus::Same
                } else {
                    DiffStatus::Modified
                }
            }
        }
    }

    /// Get the value for a specific side.
    pub fn value(&self, side: Side) -> Option<&Value> {
        match side {
            Side::Left => self.left_value.as_ref(),
            Side::Right => self.right_value.as_ref(),
        }
    }

    /// Check if both sides have values.
    pub fn has_both(&self) -> bool {
        self.left_value.is_some() && self.right_value.is_some()
    }

    /// Get a display string for a value, handling None.
    pub fn display_value(&self, side: Side) -> String {
        match self.value(side) {
            Some(v) => format_value(v),
            None => "(absent)".to_string(),
        }
    }
}

/// Represents a component-level difference between two entities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComponentDiff {
    /// Name of the component type.
    pub component_name: String,
    /// Component type ID.
    pub component_id: u32,
    /// Overall status of this component comparison.
    pub status: DiffStatus,
    /// Field-level diffs within this component.
    pub field_diffs: Vec<FieldDiff>,
}

impl ComponentDiff {
    /// Create a new component diff.
    pub fn new(component_name: impl Into<String>, component_id: u32) -> Self {
        Self {
            component_name: component_name.into(),
            component_id,
            status: DiffStatus::Same,
            field_diffs: Vec::new(),
        }
    }

    /// Add a field diff and update overall status.
    pub fn add_field(&mut self, field: FieldDiff) {
        // Update overall status based on field status
        match (&self.status, &field.status) {
            (DiffStatus::Same, other) => self.status = *other,
            (_, DiffStatus::Modified) => self.status = DiffStatus::Modified,
            _ => {}
        }
        self.field_diffs.push(field);
    }

    /// Get all fields that have differences.
    pub fn different_fields(&self) -> impl Iterator<Item = &FieldDiff> {
        self.field_diffs.iter().filter(|f| f.status.is_different())
    }

    /// Count fields with differences.
    pub fn difference_count(&self) -> usize {
        self.field_diffs
            .iter()
            .filter(|f| f.status.is_different())
            .count()
    }

    /// Check if this component has any differences.
    pub fn has_differences(&self) -> bool {
        self.status.is_different()
    }
}

// ---------------------------------------------------------------------------
// Field Copy
// ---------------------------------------------------------------------------

/// Represents a field copy operation from one entity to another.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldCopy {
    /// The source entity ID.
    pub source_entity: u64,
    /// The target entity ID.
    pub target_entity: u64,
    /// The component ID.
    pub component_id: u32,
    /// The field path to copy.
    pub field_path: String,
    /// The value being copied.
    pub value: Value,
}

impl FieldCopy {
    /// Create a new field copy operation.
    pub fn new(
        source_entity: u64,
        target_entity: u64,
        component_id: u32,
        field_path: impl Into<String>,
        value: Value,
    ) -> Self {
        Self {
            source_entity,
            target_entity,
            component_id,
            field_path: field_path.into(),
            value,
        }
    }
}

// ---------------------------------------------------------------------------
// Diff Panel
// ---------------------------------------------------------------------------

/// Panel for rendering side-by-side entity comparison.
pub struct DiffPanel {
    /// Left entity ID.
    left_entity: Option<u64>,
    /// Right entity ID.
    right_entity: Option<u64>,
    /// Computed component diffs.
    diffs: Vec<ComponentDiff>,
    /// Whether to only show differences.
    filter_different: bool,
    /// Expanded component IDs.
    expanded: std::collections::HashSet<u32>,
    /// Pending copy operations.
    pending_copies: Vec<FieldCopy>,
}

impl DiffPanel {
    /// Create a new diff panel.
    pub fn new() -> Self {
        Self {
            left_entity: None,
            right_entity: None,
            diffs: Vec::new(),
            filter_different: false,
            expanded: std::collections::HashSet::new(),
            pending_copies: Vec::new(),
        }
    }

    /// Set the entities to compare.
    pub fn set_entities(&mut self, left: u64, right: u64) {
        if self.left_entity != Some(left) || self.right_entity != Some(right) {
            self.left_entity = Some(left);
            self.right_entity = Some(right);
            self.diffs.clear();
            self.pending_copies.clear();
        }
    }

    /// Clear the comparison.
    pub fn clear(&mut self) {
        self.left_entity = None;
        self.right_entity = None;
        self.diffs.clear();
        self.pending_copies.clear();
    }

    /// Get the left entity ID.
    pub fn left_entity(&self) -> Option<u64> {
        self.left_entity
    }

    /// Get the right entity ID.
    pub fn right_entity(&self) -> Option<u64> {
        self.right_entity
    }

    /// Set whether to filter to only show differences.
    pub fn set_filter_different(&mut self, filter: bool) {
        self.filter_different = filter;
    }

    /// Check if filtering to differences only.
    pub fn is_filtering_different(&self) -> bool {
        self.filter_different
    }

    /// Toggle filtering to differences only.
    pub fn toggle_filter_different(&mut self) {
        self.filter_different = !self.filter_different;
    }

    /// Get pending field copy operations.
    pub fn pending_copies(&self) -> &[FieldCopy] {
        &self.pending_copies
    }

    /// Clear pending copies.
    pub fn clear_pending_copies(&mut self) {
        self.pending_copies.clear();
    }

    /// Take pending copies, clearing the internal list.
    pub fn take_pending_copies(&mut self) -> Vec<FieldCopy> {
        std::mem::take(&mut self.pending_copies)
    }

    /// Get computed diffs.
    pub fn diffs(&self) -> &[ComponentDiff] {
        &self.diffs
    }

    /// Get the total number of differences.
    pub fn total_differences(&self) -> usize {
        self.diffs.iter().map(|d| d.difference_count()).sum()
    }

    /// Check if a component is expanded.
    pub fn is_expanded(&self, component_id: u32) -> bool {
        self.expanded.contains(&component_id)
    }

    /// Set whether a component is expanded.
    pub fn set_expanded(&mut self, component_id: u32, expanded: bool) {
        if expanded {
            self.expanded.insert(component_id);
        } else {
            self.expanded.remove(&component_id);
        }
    }

    /// Toggle expansion for a component.
    pub fn toggle_expanded(&mut self, component_id: u32) {
        if self.expanded.contains(&component_id) {
            self.expanded.remove(&component_id);
        } else {
            self.expanded.insert(component_id);
        }
    }

    /// Expand all components with differences.
    pub fn expand_all_different(&mut self) {
        for diff in &self.diffs {
            if diff.has_differences() {
                self.expanded.insert(diff.component_id);
            }
        }
    }

    /// Collapse all components.
    pub fn collapse_all(&mut self) {
        self.expanded.clear();
    }

    /// Compute differences between two sets of components.
    ///
    /// # Arguments
    ///
    /// * `left_components` - Components from the left/primary entity.
    /// * `right_components` - Components from the right/secondary entity.
    ///
    /// Each component is a tuple of (component_id, component_name, raw_bytes).
    pub fn compute_diff(
        &mut self,
        left_components: &[(u32, String, Vec<u8>)],
        right_components: &[(u32, String, Vec<u8>)],
    ) {
        self.diffs.clear();

        // Build maps for lookup
        let left_map: HashMap<u32, (&String, &Vec<u8>)> = left_components
            .iter()
            .map(|(id, name, bytes)| (*id, (name, bytes)))
            .collect();
        let right_map: HashMap<u32, (&String, &Vec<u8>)> = right_components
            .iter()
            .map(|(id, name, bytes)| (*id, (name, bytes)))
            .collect();

        // Find all unique component IDs
        let mut all_ids: Vec<u32> = left_map.keys().chain(right_map.keys()).copied().collect();
        all_ids.sort();
        all_ids.dedup();

        for component_id in all_ids {
            let left = left_map.get(&component_id);
            let right = right_map.get(&component_id);

            let mut diff = match (left, right) {
                (Some((name, _)), _) => ComponentDiff::new(*name, component_id),
                (_, Some((name, _))) => ComponentDiff::new(*name, component_id),
                _ => continue,
            };

            match (left, right) {
                (Some((name, left_bytes)), Some((_, right_bytes))) => {
                    // Both sides have this component - compare fields
                    self.compare_component_bytes(&mut diff, name, left_bytes, right_bytes);
                }
                (Some((name, bytes)), None) => {
                    // Only on left
                    diff.status = DiffStatus::OnlyLeft;
                    self.add_component_fields(&mut diff, name, bytes, Side::Left);
                }
                (None, Some((name, bytes))) => {
                    // Only on right
                    diff.status = DiffStatus::OnlyRight;
                    self.add_component_fields(&mut diff, name, bytes, Side::Right);
                }
                (None, None) => continue,
            }

            self.diffs.push(diff);
        }
    }

    /// Compare component bytes and add field diffs.
    fn compare_component_bytes(
        &self,
        diff: &mut ComponentDiff,
        name: &str,
        left_bytes: &[u8],
        right_bytes: &[u8],
    ) {
        let left_value = decode_component(name, left_bytes);
        let right_value = decode_component(name, right_bytes);

        self.compare_values(diff, "", &left_value, &right_value);
    }

    /// Recursively compare two values and add field diffs.
    fn compare_values(
        &self,
        diff: &mut ComponentDiff,
        path_prefix: &str,
        left: &Value,
        right: &Value,
    ) {
        match (left, right) {
            (Value::Struct(left_fields), Value::Struct(right_fields)) => {
                // Compare struct fields
                let left_map: HashMap<&str, &Value> = left_fields
                    .iter()
                    .map(|(k, v)| (k.as_str(), v))
                    .collect();
                let right_map: HashMap<&str, &Value> = right_fields
                    .iter()
                    .map(|(k, v)| (k.as_str(), v))
                    .collect();

                let mut all_keys: Vec<&str> =
                    left_map.keys().chain(right_map.keys()).copied().collect();
                all_keys.sort();
                all_keys.dedup();

                for key in all_keys {
                    let path = if path_prefix.is_empty() {
                        key.to_string()
                    } else {
                        format!("{}.{}", path_prefix, key)
                    };

                    match (left_map.get(key), right_map.get(key)) {
                        (Some(lv), Some(rv)) => {
                            self.compare_values(diff, &path, lv, rv);
                        }
                        (Some(lv), None) => {
                            diff.add_field(FieldDiff::new(&path, Some((*lv).clone()), None));
                        }
                        (None, Some(rv)) => {
                            diff.add_field(FieldDiff::new(&path, None, Some((*rv).clone())));
                        }
                        (None, None) => {}
                    }
                }
            }
            _ => {
                // Leaf values - add as single field diff
                let path = if path_prefix.is_empty() {
                    "value".to_string()
                } else {
                    path_prefix.to_string()
                };
                diff.add_field(FieldDiff::new(
                    &path,
                    Some(left.clone()),
                    Some(right.clone()),
                ));
            }
        }
    }

    /// Add fields from a single component (for OnlyLeft/OnlyRight status).
    fn add_component_fields(&self, diff: &mut ComponentDiff, name: &str, bytes: &[u8], side: Side) {
        let value = decode_component(name, bytes);
        self.add_value_fields(diff, "", &value, side);
    }

    /// Recursively add fields from a value.
    fn add_value_fields(&self, diff: &mut ComponentDiff, path_prefix: &str, value: &Value, side: Side) {
        match value {
            Value::Struct(fields) => {
                for (key, val) in fields {
                    let path = if path_prefix.is_empty() {
                        key.clone()
                    } else {
                        format!("{}.{}", path_prefix, key)
                    };
                    self.add_value_fields(diff, &path, val, side);
                }
            }
            _ => {
                let path = if path_prefix.is_empty() {
                    "value".to_string()
                } else {
                    path_prefix.to_string()
                };
                let field = match side {
                    Side::Left => FieldDiff::new(&path, Some(value.clone()), None),
                    Side::Right => FieldDiff::new(&path, None, Some(value.clone())),
                };
                diff.add_field(field);
            }
        }
    }

    /// Copy a field value from one side to the other.
    ///
    /// Returns a FieldCopy if the operation is valid, None otherwise.
    pub fn copy_field(&mut self, from: Side, to: Side, field_path: &str) -> Option<FieldCopy> {
        let (source_entity, target_entity) = match (from, to, self.left_entity, self.right_entity) {
            (Side::Left, Side::Right, Some(left), Some(right)) => (left, right),
            (Side::Right, Side::Left, Some(left), Some(right)) => (right, left),
            _ => return None,
        };

        // Find the field diff
        for diff in &self.diffs {
            for field in &diff.field_diffs {
                if field.field_path == field_path {
                    if let Some(value) = field.value(from) {
                        let copy = FieldCopy::new(
                            source_entity,
                            target_entity,
                            diff.component_id,
                            field_path,
                            value.clone(),
                        );
                        self.pending_copies.push(copy.clone());
                        return Some(copy);
                    }
                }
            }
        }
        None
    }

    /// Copy all different fields from one side to the other.
    pub fn copy_all_from(&mut self, from: Side) -> Vec<FieldCopy> {
        let mut copies = Vec::new();
        let field_paths: Vec<(u32, String)> = self
            .diffs
            .iter()
            .flat_map(|d| {
                d.field_diffs
                    .iter()
                    .filter(|f| f.status.is_different() && f.value(from).is_some())
                    .map(|f| (d.component_id, f.field_path.clone()))
            })
            .collect();

        for (_, path) in field_paths {
            if let Some(copy) = self.copy_field(from, from.opposite(), &path) {
                copies.push(copy);
            }
        }
        copies
    }

    /// Render the diff panel.
    pub fn render<T: UIContext>(&mut self, ctx: &mut T) {
        // Header
        let (left_entity, right_entity) = (self.left_entity, self.right_entity);
        match (left_entity, right_entity) {
            (Some(left), Some(right)) => {
                ctx.label(&format!("Comparing Entity {} vs {}", left, right));
            }
            _ => {
                ctx.label("Select two entities to compare");
                return;
            }
        }

        ctx.separator();

        // Toolbar - collect state before rendering
        let filter_different = self.filter_different;
        let mut new_filter = filter_different;
        let mut expand_different = false;
        let mut collapse_all = false;

        ctx.horizontal(|h| {
            if h.checkbox("Show only differences", &mut new_filter) {
                // State updated after
            }
            if h.button("Expand Different") {
                expand_different = true;
            }
            if h.button("Collapse All") {
                collapse_all = true;
            }
        });

        // Apply toolbar state changes
        self.filter_different = new_filter;
        if expand_different {
            self.expand_all_different();
        }
        if collapse_all {
            self.collapse_all();
        }

        // Summary
        let total_diffs = self.total_differences();
        if total_diffs > 0 {
            ctx.label(&format!("{} field difference(s) found", total_diffs));
        } else {
            ctx.label("No differences found");
        }

        ctx.spacing();

        // Components
        if self.diffs.is_empty() {
            ctx.label("No components to compare");
            return;
        }

        // Collect indices of diffs to render
        let filter = self.filter_different;
        let indices_to_render: Vec<usize> = self
            .diffs
            .iter()
            .enumerate()
            .filter(|(_, d)| !filter || d.has_differences())
            .map(|(i, _)| i)
            .collect();

        if indices_to_render.is_empty() {
            ctx.label("No differences (all components match)");
            return;
        }

        // Render each component diff
        for idx in indices_to_render {
            self.render_component_diff_at(ctx, idx);
        }

        // Pending copies footer
        let pending_count = self.pending_copies.len();
        if pending_count > 0 {
            ctx.separator();
            let mut clear_pending = false;
            ctx.horizontal(|h| {
                h.label(&format!("{} pending copy operation(s)", pending_count));
                if h.button("Apply Copies") {
                    // Application handled by caller
                }
                if h.button("Clear") {
                    clear_pending = true;
                }
            });
            if clear_pending {
                self.pending_copies.clear();
            }
        }
    }

    /// Render a single component diff by index.
    fn render_component_diff_at<T: UIContext>(&self, ctx: &mut T, idx: usize) {
        let diff = &self.diffs[idx];
        let status_indicator = diff.status.indicator();
        let header = format!(
            "[{}] {} ({} diff(s))",
            status_indicator,
            diff.component_name,
            diff.difference_count()
        );

        // Clone data needed for rendering inside closure
        let filter_different = self.filter_different;
        let field_diffs: Vec<_> = diff.field_diffs.iter().cloned().collect();

        ctx.collapsing(&header, |inner| {
            // Column headers
            inner.horizontal(|h| {
                h.label("Field");
                h.label("Left");
                h.label("");  // Action column
                h.label("Right");
            });

            inner.separator();

            // Field rows
            for field in &field_diffs {
                if filter_different && !field.status.is_different() {
                    continue;
                }

                Self::render_field_diff_static(inner, field);
            }
        });
    }

    /// Render a single field diff row (static method to avoid borrow issues).
    fn render_field_diff_static<T: UIContext>(ctx: &mut T, field: &FieldDiff) {
        // Status indicator and field path
        let status_char = field.status.indicator();
        ctx.label(&format!("[{}] {} | {} | {} | {}",
            status_char,
            field.field_path,
            field.display_value(Side::Left),
            match field.status {
                DiffStatus::Modified => "<-> ",
                DiffStatus::OnlyLeft => " -> ",
                DiffStatus::OnlyRight => " <- ",
                DiffStatus::Same => " = ",
            },
            field.display_value(Side::Right)
        ));
    }
}

impl Default for DiffPanel {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Format a Value for display.
fn format_value(value: &Value) -> String {
    match value {
        Value::Bool(v) => format!("{}", v),
        Value::I32(v) => format!("{}", v),
        Value::U32(v) => format!("{}", v),
        Value::F32(v) => format!("{:.3}", v),
        Value::F64(v) => format!("{:.3}", v),
        Value::String(s) => format!("\"{}\"", s),
        Value::Vec2(v) => format!("[{:.3}, {:.3}]", v[0], v[1]),
        Value::Vec3(v) => format!("[{:.3}, {:.3}, {:.3}]", v[0], v[1], v[2]),
        Value::Vec4(v) => format!("[{:.3}, {:.3}, {:.3}, {:.3}]", v[0], v[1], v[2], v[3]),
        Value::Quat(v) => format!("q({:.3}, {:.3}, {:.3}, {:.3})", v[0], v[1], v[2], v[3]),
        Value::Rgba(v) => format!("rgba({:.3}, {:.3}, {:.3}, {:.3})", v[0], v[1], v[2], v[3]),
        Value::Raw(bytes) => format!("<{} bytes>", bytes.len()),
        Value::Struct(fields) => {
            let field_strs: Vec<String> = fields
                .iter()
                .map(|(k, v)| format!("{}: {}", k, format_value(v)))
                .collect();
            format!("{{ {} }}", field_strs.join(", "))
        }
    }
}

/// Decode component bytes into a Value based on component name.
fn decode_component(name: &str, bytes: &[u8]) -> Value {
    let name_lower = name.to_lowercase();

    // Handle common component patterns
    if name_lower.contains("transform") {
        return decode_transform(bytes);
    }
    if name_lower.contains("position") || name_lower.contains("pos") {
        return decode_position(bytes);
    }
    if name_lower.contains("rotation") || name_lower.contains("rot") {
        return decode_rotation(bytes);
    }
    if name_lower.contains("scale") {
        return decode_scale(bytes);
    }
    if name_lower.contains("color") || name_lower.contains("colour") {
        return decode_color(bytes);
    }
    if name_lower.contains("velocity") || name_lower.contains("vel") {
        return decode_velocity(bytes);
    }

    // Fall back to size-based detection
    decode_by_size(bytes)
}

fn decode_transform(bytes: &[u8]) -> Value {
    if bytes.len() >= 40 {
        let mut fields = Vec::new();
        if let Some(pos) = TypeDecoder::decode_vec3(&bytes[0..12]) {
            fields.push(("position".to_string(), Value::Vec3(pos)));
        }
        if let Some(rot) = TypeDecoder::decode_vec4(&bytes[12..28]) {
            fields.push(("rotation".to_string(), Value::Quat(rot)));
        }
        if let Some(scale) = TypeDecoder::decode_vec3(&bytes[28..40]) {
            fields.push(("scale".to_string(), Value::Vec3(scale)));
        }
        if !fields.is_empty() {
            return Value::Struct(fields);
        }
    }
    Value::Raw(bytes.to_vec())
}

fn decode_position(bytes: &[u8]) -> Value {
    if bytes.len() >= 12 {
        if let Some(v) = TypeDecoder::decode_vec3(bytes) {
            return Value::Vec3(v);
        }
    }
    if bytes.len() >= 8 {
        if let Some(v) = TypeDecoder::decode_vec2(bytes) {
            return Value::Vec2(v);
        }
    }
    Value::Raw(bytes.to_vec())
}

fn decode_rotation(bytes: &[u8]) -> Value {
    if bytes.len() >= 16 {
        if let Some(v) = TypeDecoder::decode_vec4(bytes) {
            return Value::Quat(v);
        }
    }
    if bytes.len() >= 12 {
        if let Some(v) = TypeDecoder::decode_vec3(bytes) {
            return Value::Vec3(v);
        }
    }
    Value::Raw(bytes.to_vec())
}

fn decode_scale(bytes: &[u8]) -> Value {
    if bytes.len() >= 12 {
        if let Some(v) = TypeDecoder::decode_vec3(bytes) {
            return Value::Vec3(v);
        }
    }
    if bytes.len() >= 4 {
        if let Some(v) = TypeDecoder::decode_f32(bytes) {
            return Value::F32(v);
        }
    }
    Value::Raw(bytes.to_vec())
}

fn decode_color(bytes: &[u8]) -> Value {
    if bytes.len() >= 16 {
        if let Some(v) = TypeDecoder::decode_vec4(bytes) {
            return Value::Rgba(v);
        }
    }
    if bytes.len() >= 12 {
        if let Some(v) = TypeDecoder::decode_vec3(bytes) {
            return Value::Rgba([v[0], v[1], v[2], 1.0]);
        }
    }
    Value::Raw(bytes.to_vec())
}

fn decode_velocity(bytes: &[u8]) -> Value {
    if bytes.len() >= 12 {
        if let Some(v) = TypeDecoder::decode_vec3(bytes) {
            return Value::Vec3(v);
        }
    }
    if bytes.len() >= 8 {
        if let Some(v) = TypeDecoder::decode_vec2(bytes) {
            return Value::Vec2(v);
        }
    }
    Value::Raw(bytes.to_vec())
}

fn decode_by_size(bytes: &[u8]) -> Value {
    match bytes.len() {
        1 => TypeDecoder::decode_bool(bytes)
            .map(Value::Bool)
            .unwrap_or_else(|| Value::Raw(bytes.to_vec())),
        4 => TypeDecoder::decode_f32(bytes)
            .map(Value::F32)
            .unwrap_or_else(|| Value::Raw(bytes.to_vec())),
        8 => TypeDecoder::decode_vec2(bytes)
            .map(Value::Vec2)
            .unwrap_or_else(|| Value::Raw(bytes.to_vec())),
        12 => TypeDecoder::decode_vec3(bytes)
            .map(Value::Vec3)
            .unwrap_or_else(|| Value::Raw(bytes.to_vec())),
        16 => TypeDecoder::decode_vec4(bytes)
            .map(Value::Vec4)
            .unwrap_or_else(|| Value::Raw(bytes.to_vec())),
        _ => Value::Raw(bytes.to_vec()),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egui_adapter::MockUIContext;

    // -------------------------------------------------------------------------
    // SelectionMode Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_selection_mode_default() {
        let mode: SelectionMode = Default::default();
        assert_eq!(mode, SelectionMode::Single);
    }

    // -------------------------------------------------------------------------
    // MultiSelection Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_multi_selection_new() {
        let sel = MultiSelection::new();
        assert!(sel.primary.is_none());
        assert!(sel.secondary.is_none());
        assert!(sel.additional.is_empty());
        assert_eq!(sel.mode, SelectionMode::Single);
    }

    #[test]
    fn test_multi_selection_set_primary() {
        let mut sel = MultiSelection::new();
        sel.set_primary(Some(100));
        assert_eq!(sel.primary, Some(100));
        assert_eq!(sel.mode, SelectionMode::Single);
    }

    #[test]
    fn test_multi_selection_set_secondary() {
        let mut sel = MultiSelection::new();
        sel.set_primary(Some(100));
        sel.set_secondary(Some(200));
        assert_eq!(sel.secondary, Some(200));
        assert_eq!(sel.mode, SelectionMode::Compare);
    }

    #[test]
    fn test_multi_selection_add_entity() {
        let mut sel = MultiSelection::new();
        sel.set_primary(Some(100));
        sel.set_secondary(Some(200));
        sel.add_entity(300);
        assert!(sel.additional.contains(&300));
        assert_eq!(sel.mode, SelectionMode::Multi);
    }

    #[test]
    fn test_multi_selection_add_duplicate_ignored() {
        let mut sel = MultiSelection::new();
        sel.set_primary(Some(100));
        sel.add_entity(100); // Should be ignored - same as primary
        assert!(sel.additional.is_empty());
    }

    #[test]
    fn test_multi_selection_remove_entity() {
        let mut sel = MultiSelection::new();
        sel.set_primary(Some(100));
        sel.set_secondary(Some(200));
        sel.remove_entity(200);
        assert!(sel.secondary.is_none());
    }

    #[test]
    fn test_multi_selection_toggle_entity() {
        let mut sel = MultiSelection::new();
        sel.toggle_entity(100);
        assert_eq!(sel.primary, Some(100));
        sel.toggle_entity(200);
        assert_eq!(sel.secondary, Some(200));
        sel.toggle_entity(100);
        assert!(sel.primary.is_none());
    }

    #[test]
    fn test_multi_selection_contains() {
        let mut sel = MultiSelection::new();
        sel.set_primary(Some(100));
        sel.set_secondary(Some(200));
        sel.add_entity(300);
        assert!(sel.contains(100));
        assert!(sel.contains(200));
        assert!(sel.contains(300));
        assert!(!sel.contains(400));
    }

    #[test]
    fn test_multi_selection_all_entities() {
        let mut sel = MultiSelection::new();
        sel.set_primary(Some(100));
        sel.set_secondary(Some(200));
        sel.add_entity(300);
        let all = sel.all_entities();
        assert_eq!(all.len(), 3);
        assert!(all.contains(&100));
        assert!(all.contains(&200));
        assert!(all.contains(&300));
    }

    #[test]
    fn test_multi_selection_count() {
        let mut sel = MultiSelection::new();
        assert_eq!(sel.count(), 0);
        sel.set_primary(Some(100));
        assert_eq!(sel.count(), 1);
        sel.set_secondary(Some(200));
        assert_eq!(sel.count(), 2);
        sel.add_entity(300);
        assert_eq!(sel.count(), 3);
    }

    #[test]
    fn test_multi_selection_clear() {
        let mut sel = MultiSelection::new();
        sel.set_primary(Some(100));
        sel.set_secondary(Some(200));
        sel.add_entity(300);
        sel.clear();
        assert_eq!(sel.count(), 0);
        assert_eq!(sel.mode, SelectionMode::Single);
    }

    #[test]
    fn test_multi_selection_is_comparing() {
        let mut sel = MultiSelection::new();
        assert!(!sel.is_comparing());
        sel.set_primary(Some(100));
        assert!(!sel.is_comparing());
        sel.set_secondary(Some(200));
        assert!(sel.is_comparing());
    }

    #[test]
    fn test_multi_selection_swap() {
        let mut sel = MultiSelection::new();
        sel.set_primary(Some(100));
        sel.set_secondary(Some(200));
        sel.swap();
        assert_eq!(sel.primary, Some(200));
        assert_eq!(sel.secondary, Some(100));
    }

    // -------------------------------------------------------------------------
    // DiffStatus Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_diff_status_default() {
        let status: DiffStatus = Default::default();
        assert_eq!(status, DiffStatus::Same);
    }

    #[test]
    fn test_diff_status_color() {
        assert_eq!(DiffStatus::Same.color()[0], 0.7);
        assert_eq!(DiffStatus::Modified.color()[0], 1.0);
        assert_eq!(DiffStatus::OnlyLeft.color()[0], 0.2);
        assert_eq!(DiffStatus::OnlyRight.color()[0], 0.2);
    }

    #[test]
    fn test_diff_status_indicator() {
        assert_eq!(DiffStatus::Same.indicator(), '=');
        assert_eq!(DiffStatus::Modified.indicator(), '!');
        assert_eq!(DiffStatus::OnlyLeft.indicator(), '<');
        assert_eq!(DiffStatus::OnlyRight.indicator(), '>');
    }

    #[test]
    fn test_diff_status_is_different() {
        assert!(!DiffStatus::Same.is_different());
        assert!(DiffStatus::Modified.is_different());
        assert!(DiffStatus::OnlyLeft.is_different());
        assert!(DiffStatus::OnlyRight.is_different());
    }

    // -------------------------------------------------------------------------
    // Side Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_side_opposite() {
        assert_eq!(Side::Left.opposite(), Side::Right);
        assert_eq!(Side::Right.opposite(), Side::Left);
    }

    // -------------------------------------------------------------------------
    // FieldDiff Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_field_diff_new_same() {
        let diff = FieldDiff::new(
            "position.x",
            Some(Value::F32(1.0)),
            Some(Value::F32(1.0)),
        );
        assert_eq!(diff.status, DiffStatus::Same);
    }

    #[test]
    fn test_field_diff_new_modified() {
        let diff = FieldDiff::new(
            "position.x",
            Some(Value::F32(1.0)),
            Some(Value::F32(2.0)),
        );
        assert_eq!(diff.status, DiffStatus::Modified);
    }

    #[test]
    fn test_field_diff_new_only_left() {
        let diff = FieldDiff::new("position.x", Some(Value::F32(1.0)), None);
        assert_eq!(diff.status, DiffStatus::OnlyLeft);
    }

    #[test]
    fn test_field_diff_new_only_right() {
        let diff = FieldDiff::new("position.x", None, Some(Value::F32(1.0)));
        assert_eq!(diff.status, DiffStatus::OnlyRight);
    }

    #[test]
    fn test_field_diff_value() {
        let diff = FieldDiff::new(
            "x",
            Some(Value::F32(1.0)),
            Some(Value::F32(2.0)),
        );
        assert!(diff.value(Side::Left).is_some());
        assert!(diff.value(Side::Right).is_some());
    }

    #[test]
    fn test_field_diff_has_both() {
        let both = FieldDiff::new("x", Some(Value::F32(1.0)), Some(Value::F32(2.0)));
        assert!(both.has_both());

        let left_only = FieldDiff::new("x", Some(Value::F32(1.0)), None);
        assert!(!left_only.has_both());
    }

    #[test]
    fn test_field_diff_display_value() {
        let diff = FieldDiff::new("x", Some(Value::F32(1.5)), None);
        assert!(diff.display_value(Side::Left).contains("1.5"));
        assert_eq!(diff.display_value(Side::Right), "(absent)");
    }

    // -------------------------------------------------------------------------
    // ComponentDiff Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_component_diff_new() {
        let diff = ComponentDiff::new("Transform", 1);
        assert_eq!(diff.component_name, "Transform");
        assert_eq!(diff.component_id, 1);
        assert_eq!(diff.status, DiffStatus::Same);
        assert!(diff.field_diffs.is_empty());
    }

    #[test]
    fn test_component_diff_add_field() {
        let mut diff = ComponentDiff::new("Transform", 1);
        diff.add_field(FieldDiff::new("x", Some(Value::F32(1.0)), Some(Value::F32(2.0))));
        assert_eq!(diff.field_diffs.len(), 1);
        assert_eq!(diff.status, DiffStatus::Modified);
    }

    #[test]
    fn test_component_diff_different_fields() {
        let mut diff = ComponentDiff::new("Transform", 1);
        diff.add_field(FieldDiff::new("x", Some(Value::F32(1.0)), Some(Value::F32(1.0))));
        diff.add_field(FieldDiff::new("y", Some(Value::F32(1.0)), Some(Value::F32(2.0))));

        let different: Vec<_> = diff.different_fields().collect();
        assert_eq!(different.len(), 1);
        assert_eq!(different[0].field_path, "y");
    }

    #[test]
    fn test_component_diff_difference_count() {
        let mut diff = ComponentDiff::new("Transform", 1);
        diff.add_field(FieldDiff::new("x", Some(Value::F32(1.0)), Some(Value::F32(1.0))));
        diff.add_field(FieldDiff::new("y", Some(Value::F32(1.0)), Some(Value::F32(2.0))));
        diff.add_field(FieldDiff::new("z", Some(Value::F32(1.0)), None));

        assert_eq!(diff.difference_count(), 2);
    }

    #[test]
    fn test_component_diff_has_differences() {
        let mut diff = ComponentDiff::new("Transform", 1);
        assert!(!diff.has_differences());

        diff.add_field(FieldDiff::new("x", Some(Value::F32(1.0)), Some(Value::F32(2.0))));
        assert!(diff.has_differences());
    }

    // -------------------------------------------------------------------------
    // FieldCopy Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_field_copy_new() {
        let copy = FieldCopy::new(100, 200, 1, "position.x", Value::F32(5.0));
        assert_eq!(copy.source_entity, 100);
        assert_eq!(copy.target_entity, 200);
        assert_eq!(copy.component_id, 1);
        assert_eq!(copy.field_path, "position.x");
    }

    // -------------------------------------------------------------------------
    // DiffPanel Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_diff_panel_new() {
        let panel = DiffPanel::new();
        assert!(panel.left_entity().is_none());
        assert!(panel.right_entity().is_none());
        assert!(panel.diffs().is_empty());
        assert!(!panel.is_filtering_different());
    }

    #[test]
    fn test_diff_panel_set_entities() {
        let mut panel = DiffPanel::new();
        panel.set_entities(100, 200);
        assert_eq!(panel.left_entity(), Some(100));
        assert_eq!(panel.right_entity(), Some(200));
    }

    #[test]
    fn test_diff_panel_clear() {
        let mut panel = DiffPanel::new();
        panel.set_entities(100, 200);
        panel.clear();
        assert!(panel.left_entity().is_none());
        assert!(panel.right_entity().is_none());
    }

    #[test]
    fn test_diff_panel_filter_different() {
        let mut panel = DiffPanel::new();
        assert!(!panel.is_filtering_different());
        panel.set_filter_different(true);
        assert!(panel.is_filtering_different());
        panel.toggle_filter_different();
        assert!(!panel.is_filtering_different());
    }

    #[test]
    fn test_diff_panel_expansion() {
        let mut panel = DiffPanel::new();
        assert!(!panel.is_expanded(1));
        panel.set_expanded(1, true);
        assert!(panel.is_expanded(1));
        panel.toggle_expanded(1);
        assert!(!panel.is_expanded(1));
    }

    #[test]
    fn test_diff_panel_collapse_all() {
        let mut panel = DiffPanel::new();
        panel.set_expanded(1, true);
        panel.set_expanded(2, true);
        panel.collapse_all();
        assert!(!panel.is_expanded(1));
        assert!(!panel.is_expanded(2));
    }

    #[test]
    fn test_diff_panel_pending_copies() {
        let mut panel = DiffPanel::new();
        assert!(panel.pending_copies().is_empty());
        panel.pending_copies.push(FieldCopy::new(1, 2, 1, "x", Value::F32(1.0)));
        assert_eq!(panel.pending_copies().len(), 1);
        panel.clear_pending_copies();
        assert!(panel.pending_copies().is_empty());
    }

    #[test]
    fn test_diff_panel_take_pending_copies() {
        let mut panel = DiffPanel::new();
        panel.pending_copies.push(FieldCopy::new(1, 2, 1, "x", Value::F32(1.0)));
        let copies = panel.take_pending_copies();
        assert_eq!(copies.len(), 1);
        assert!(panel.pending_copies().is_empty());
    }

    #[test]
    fn test_diff_panel_compute_diff_same_components() {
        let mut panel = DiffPanel::new();
        panel.set_entities(100, 200);

        let pos_bytes: Vec<u8> = make_vec3_bytes(1.0, 2.0, 3.0);
        let left = vec![(1, "Position".to_string(), pos_bytes.clone())];
        let right = vec![(1, "Position".to_string(), pos_bytes)];

        panel.compute_diff(&left, &right);
        assert_eq!(panel.diffs().len(), 1);
        assert_eq!(panel.diffs()[0].status, DiffStatus::Same);
    }

    #[test]
    fn test_diff_panel_compute_diff_modified_components() {
        let mut panel = DiffPanel::new();
        panel.set_entities(100, 200);

        let left_bytes = make_vec3_bytes(1.0, 2.0, 3.0);
        let right_bytes = make_vec3_bytes(4.0, 5.0, 6.0);

        let left = vec![(1, "Position".to_string(), left_bytes)];
        let right = vec![(1, "Position".to_string(), right_bytes)];

        panel.compute_diff(&left, &right);
        assert_eq!(panel.diffs().len(), 1);
        assert_eq!(panel.diffs()[0].status, DiffStatus::Modified);
    }

    #[test]
    fn test_diff_panel_compute_diff_only_left() {
        let mut panel = DiffPanel::new();
        panel.set_entities(100, 200);

        let pos_bytes = make_vec3_bytes(1.0, 2.0, 3.0);
        let left = vec![(1, "Position".to_string(), pos_bytes)];
        let right: Vec<(u32, String, Vec<u8>)> = vec![];

        panel.compute_diff(&left, &right);
        assert_eq!(panel.diffs().len(), 1);
        assert_eq!(panel.diffs()[0].status, DiffStatus::OnlyLeft);
    }

    #[test]
    fn test_diff_panel_compute_diff_only_right() {
        let mut panel = DiffPanel::new();
        panel.set_entities(100, 200);

        let pos_bytes = make_vec3_bytes(1.0, 2.0, 3.0);
        let left: Vec<(u32, String, Vec<u8>)> = vec![];
        let right = vec![(1, "Position".to_string(), pos_bytes)];

        panel.compute_diff(&left, &right);
        assert_eq!(panel.diffs().len(), 1);
        assert_eq!(panel.diffs()[0].status, DiffStatus::OnlyRight);
    }

    #[test]
    fn test_diff_panel_compute_diff_multiple_components() {
        let mut panel = DiffPanel::new();
        panel.set_entities(100, 200);

        let pos_bytes = make_vec3_bytes(1.0, 2.0, 3.0);
        let color_bytes = make_vec4_bytes(1.0, 0.0, 0.0, 1.0);

        let left = vec![
            (1, "Position".to_string(), pos_bytes.clone()),
            (2, "Color".to_string(), color_bytes.clone()),
        ];
        let right = vec![
            (1, "Position".to_string(), pos_bytes),
            (2, "Color".to_string(), make_vec4_bytes(0.0, 1.0, 0.0, 1.0)),
        ];

        panel.compute_diff(&left, &right);
        assert_eq!(panel.diffs().len(), 2);
    }

    #[test]
    fn test_diff_panel_total_differences() {
        let mut panel = DiffPanel::new();
        panel.set_entities(100, 200);

        let left_bytes = make_vec3_bytes(1.0, 2.0, 3.0);
        let right_bytes = make_vec3_bytes(4.0, 5.0, 6.0);

        let left = vec![(1, "Position".to_string(), left_bytes)];
        let right = vec![(1, "Position".to_string(), right_bytes)];

        panel.compute_diff(&left, &right);
        assert!(panel.total_differences() > 0);
    }

    #[test]
    fn test_diff_panel_expand_all_different() {
        let mut panel = DiffPanel::new();
        panel.set_entities(100, 200);

        let left_bytes = make_vec3_bytes(1.0, 2.0, 3.0);
        let right_bytes = make_vec3_bytes(4.0, 5.0, 6.0);

        let left = vec![(1, "Position".to_string(), left_bytes)];
        let right = vec![(1, "Position".to_string(), right_bytes)];

        panel.compute_diff(&left, &right);
        panel.expand_all_different();
        assert!(panel.is_expanded(1));
    }

    #[test]
    fn test_diff_panel_copy_field() {
        let mut panel = DiffPanel::new();
        panel.set_entities(100, 200);

        let left_bytes = make_vec3_bytes(1.0, 2.0, 3.0);
        let right_bytes = make_vec3_bytes(4.0, 5.0, 6.0);

        let left = vec![(1, "Position".to_string(), left_bytes)];
        let right = vec![(1, "Position".to_string(), right_bytes)];

        panel.compute_diff(&left, &right);

        // Copy from left to right
        let copy = panel.copy_field(Side::Left, Side::Right, "value");
        assert!(copy.is_some());
        let copy = copy.unwrap();
        assert_eq!(copy.source_entity, 100);
        assert_eq!(copy.target_entity, 200);
    }

    #[test]
    fn test_diff_panel_copy_field_not_found() {
        let mut panel = DiffPanel::new();
        panel.set_entities(100, 200);
        panel.compute_diff(&[], &[]);

        let copy = panel.copy_field(Side::Left, Side::Right, "nonexistent");
        assert!(copy.is_none());
    }

    #[test]
    fn test_diff_panel_render_no_entities() {
        let mut panel = DiffPanel::new();
        let mut ctx = MockUIContext::new(1);
        panel.render(&mut ctx);
        // Should not panic
    }

    #[test]
    fn test_diff_panel_render_with_entities() {
        let mut panel = DiffPanel::new();
        panel.set_entities(100, 200);
        let mut ctx = MockUIContext::new(1);
        panel.render(&mut ctx);
        // Should not panic
    }

    #[test]
    fn test_diff_panel_render_with_diffs() {
        let mut panel = DiffPanel::new();
        panel.set_entities(100, 200);

        let left_bytes = make_vec3_bytes(1.0, 2.0, 3.0);
        let right_bytes = make_vec3_bytes(4.0, 5.0, 6.0);

        let left = vec![(1, "Position".to_string(), left_bytes)];
        let right = vec![(1, "Position".to_string(), right_bytes)];

        panel.compute_diff(&left, &right);

        let mut ctx = MockUIContext::new(1);
        panel.render(&mut ctx);
        // Should not panic
    }

    #[test]
    fn test_diff_panel_default() {
        let panel: DiffPanel = Default::default();
        assert!(panel.left_entity().is_none());
    }

    // -------------------------------------------------------------------------
    // Helper Function Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_format_value_bool() {
        assert_eq!(format_value(&Value::Bool(true)), "true");
        assert_eq!(format_value(&Value::Bool(false)), "false");
    }

    #[test]
    fn test_format_value_int() {
        assert_eq!(format_value(&Value::I32(42)), "42");
        assert_eq!(format_value(&Value::U32(100)), "100");
    }

    #[test]
    fn test_format_value_float() {
        let formatted = format_value(&Value::F32(3.14159));
        assert!(formatted.starts_with("3.14"));
    }

    #[test]
    fn test_format_value_string() {
        assert_eq!(format_value(&Value::String("hello".into())), "\"hello\"");
    }

    #[test]
    fn test_format_value_vec2() {
        let formatted = format_value(&Value::Vec2([1.0, 2.0]));
        assert!(formatted.contains("1.0"));
        assert!(formatted.contains("2.0"));
    }

    #[test]
    fn test_format_value_vec3() {
        let formatted = format_value(&Value::Vec3([1.0, 2.0, 3.0]));
        assert!(formatted.contains("1.0"));
        assert!(formatted.contains("2.0"));
        assert!(formatted.contains("3.0"));
    }

    #[test]
    fn test_format_value_vec4() {
        let formatted = format_value(&Value::Vec4([1.0, 2.0, 3.0, 4.0]));
        assert!(formatted.contains("1.0"));
        assert!(formatted.contains("4.0"));
    }

    #[test]
    fn test_format_value_quat() {
        let formatted = format_value(&Value::Quat([0.0, 0.0, 0.0, 1.0]));
        assert!(formatted.starts_with("q("));
    }

    #[test]
    fn test_format_value_rgba() {
        let formatted = format_value(&Value::Rgba([1.0, 0.5, 0.0, 1.0]));
        assert!(formatted.starts_with("rgba("));
    }

    #[test]
    fn test_format_value_raw() {
        let formatted = format_value(&Value::Raw(vec![1, 2, 3, 4, 5]));
        assert!(formatted.contains("5 bytes"));
    }

    #[test]
    fn test_format_value_struct() {
        let formatted = format_value(&Value::Struct(vec![
            ("x".to_string(), Value::F32(1.0)),
        ]));
        assert!(formatted.contains("x:"));
    }

    #[test]
    fn test_decode_component_position() {
        let bytes = make_vec3_bytes(1.0, 2.0, 3.0);
        let value = decode_component("Position", &bytes);
        assert!(matches!(value, Value::Vec3(_)));
    }

    #[test]
    fn test_decode_component_rotation() {
        let bytes = make_vec4_bytes(0.0, 0.0, 0.0, 1.0);
        let value = decode_component("Rotation", &bytes);
        assert!(matches!(value, Value::Quat(_)));
    }

    #[test]
    fn test_decode_component_color() {
        let bytes = make_vec4_bytes(1.0, 0.0, 0.0, 1.0);
        let value = decode_component("Color", &bytes);
        assert!(matches!(value, Value::Rgba(_)));
    }

    #[test]
    fn test_decode_component_transform() {
        let mut bytes = Vec::new();
        // Position (12 bytes)
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        bytes.extend_from_slice(&2.0f32.to_le_bytes());
        bytes.extend_from_slice(&3.0f32.to_le_bytes());
        // Rotation (16 bytes)
        bytes.extend_from_slice(&0.0f32.to_le_bytes());
        bytes.extend_from_slice(&0.0f32.to_le_bytes());
        bytes.extend_from_slice(&0.0f32.to_le_bytes());
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        // Scale (12 bytes)
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        bytes.extend_from_slice(&1.0f32.to_le_bytes());

        let value = decode_component("Transform", &bytes);
        assert!(matches!(value, Value::Struct(_)));
    }

    #[test]
    fn test_decode_by_size_bool() {
        let value = decode_by_size(&[1]);
        assert!(matches!(value, Value::Bool(true)));
    }

    #[test]
    fn test_decode_by_size_f32() {
        let value = decode_by_size(&1.0f32.to_le_bytes());
        assert!(matches!(value, Value::F32(_)));
    }

    #[test]
    fn test_decode_by_size_vec2() {
        let value = decode_by_size(&make_vec2_bytes(1.0, 2.0));
        assert!(matches!(value, Value::Vec2(_)));
    }

    #[test]
    fn test_decode_by_size_vec3() {
        let value = decode_by_size(&make_vec3_bytes(1.0, 2.0, 3.0));
        assert!(matches!(value, Value::Vec3(_)));
    }

    #[test]
    fn test_decode_by_size_vec4() {
        let value = decode_by_size(&make_vec4_bytes(1.0, 2.0, 3.0, 4.0));
        assert!(matches!(value, Value::Vec4(_)));
    }

    #[test]
    fn test_decode_by_size_raw() {
        let value = decode_by_size(&[1, 2, 3, 4, 5, 6, 7]);
        assert!(matches!(value, Value::Raw(_)));
    }

    // -------------------------------------------------------------------------
    // Helper functions for tests
    // -------------------------------------------------------------------------

    fn make_vec2_bytes(x: f32, y: f32) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(8);
        bytes.extend_from_slice(&x.to_le_bytes());
        bytes.extend_from_slice(&y.to_le_bytes());
        bytes
    }

    fn make_vec3_bytes(x: f32, y: f32, z: f32) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(12);
        bytes.extend_from_slice(&x.to_le_bytes());
        bytes.extend_from_slice(&y.to_le_bytes());
        bytes.extend_from_slice(&z.to_le_bytes());
        bytes
    }

    fn make_vec4_bytes(x: f32, y: f32, z: f32, w: f32) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(16);
        bytes.extend_from_slice(&x.to_le_bytes());
        bytes.extend_from_slice(&y.to_le_bytes());
        bytes.extend_from_slice(&z.to_le_bytes());
        bytes.extend_from_slice(&w.to_le_bytes());
        bytes
    }
}
