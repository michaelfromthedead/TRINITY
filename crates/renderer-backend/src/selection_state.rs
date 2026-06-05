//! Shared Selection State for Editor Panels (T-TL-2.7)
//!
//! Provides thread-safe shared selection state that connects all editor panels.
//! Supports single and multi-selection, gizmo mode/space management, snap settings,
//! and listener notifications for selection changes.
//!
//! # Architecture
//!
//! ```text
//! SharedSelectionState (Arc<RwLock<SelectionState>>)
//!          │
//!          ├── HierarchyPanel ─────────► select(), toggle_selection()
//!          │
//!          ├── InspectorPanel ─────────► read().selected_entity_id
//!          │
//!          ├── ViewportPanel ──────────► hover(), set_gizmo_mode()
//!          │
//!          └── SelectionListener[] ────► on_selection_changed(), on_hover_changed()
//! ```
//!
//! # Example
//!
//! ```rust
//! use renderer_backend::selection_state::{SharedSelectionState, GizmoMode, GizmoSpace};
//!
//! let state = SharedSelectionState::new();
//!
//! // Select an entity
//! state.select(Some(42));
//! assert_eq!(state.read().selected_entity_id, Some(42));
//!
//! // Multi-select
//! state.add_to_selection(43);
//! state.add_to_selection(44);
//! assert_eq!(state.read().multi_selection.len(), 3);
//!
//! // Change gizmo mode
//! state.set_gizmo_mode(GizmoMode::Rotate);
//! assert_eq!(state.read().gizmo_mode, GizmoMode::Rotate);
//! ```

use std::ops::Deref;
use std::sync::{Arc, RwLock, RwLockReadGuard};

// ---------------------------------------------------------------------------
// Gizmo Mode
// ---------------------------------------------------------------------------

/// Transform gizmo interaction mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GizmoMode {
    /// Translate (move) entities - keyboard shortcut: W.
    #[default]
    Translate,
    /// Rotate entities - keyboard shortcut: E.
    Rotate,
    /// Scale entities - keyboard shortcut: R.
    Scale,
    /// No gizmo displayed.
    None,
}

impl GizmoMode {
    /// Get the next mode in the cycle (Translate -> Rotate -> Scale -> Translate).
    #[must_use]
    pub fn next(self) -> Self {
        match self {
            GizmoMode::Translate => GizmoMode::Rotate,
            GizmoMode::Rotate => GizmoMode::Scale,
            GizmoMode::Scale => GizmoMode::Translate,
            GizmoMode::None => GizmoMode::Translate,
        }
    }

    /// Get the keyboard shortcut for this mode.
    #[must_use]
    pub fn shortcut(self) -> Option<char> {
        match self {
            GizmoMode::Translate => Some('W'),
            GizmoMode::Rotate => Some('E'),
            GizmoMode::Scale => Some('R'),
            GizmoMode::None => None,
        }
    }

    /// Get mode from keyboard shortcut.
    #[must_use]
    pub fn from_shortcut(key: char) -> Option<Self> {
        match key.to_ascii_uppercase() {
            'W' => Some(GizmoMode::Translate),
            'E' => Some(GizmoMode::Rotate),
            'R' => Some(GizmoMode::Scale),
            _ => None,
        }
    }

    /// Get display name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            GizmoMode::Translate => "Translate",
            GizmoMode::Rotate => "Rotate",
            GizmoMode::Scale => "Scale",
            GizmoMode::None => "None",
        }
    }
}

// ---------------------------------------------------------------------------
// Gizmo Space
// ---------------------------------------------------------------------------

/// Coordinate space for gizmo transformations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GizmoSpace {
    /// Transform relative to object's local axes.
    #[default]
    Local,
    /// Transform relative to world axes.
    World,
}

impl GizmoSpace {
    /// Toggle between Local and World space.
    #[must_use]
    pub fn toggle(self) -> Self {
        match self {
            GizmoSpace::Local => GizmoSpace::World,
            GizmoSpace::World => GizmoSpace::Local,
        }
    }

    /// Get display name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            GizmoSpace::Local => "Local",
            GizmoSpace::World => "World",
        }
    }
}

// ---------------------------------------------------------------------------
// Snap Settings
// ---------------------------------------------------------------------------

/// Snap settings for transform operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SnapSettings {
    /// Whether snapping is enabled.
    pub enabled: bool,
    /// Translation snap increment (in world units).
    pub translate: f32,
    /// Rotation snap increment (in degrees).
    pub rotate: f32,
    /// Scale snap increment (multiplier).
    pub scale: f32,
}

impl Default for SnapSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            translate: 1.0,
            rotate: 15.0,
            scale: 0.1,
        }
    }
}

impl SnapSettings {
    /// Create snap settings with custom increments.
    #[must_use]
    pub fn new(translate: f32, rotate: f32, scale: f32) -> Self {
        Self {
            enabled: false,
            translate,
            rotate,
            scale,
        }
    }

    /// Enable snapping.
    #[must_use]
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Apply translation snap to a value.
    #[must_use]
    pub fn snap_translate(&self, value: f32) -> f32 {
        if self.enabled && self.translate > 0.0 {
            (value / self.translate).round() * self.translate
        } else {
            value
        }
    }

    /// Apply rotation snap to a value (in degrees).
    #[must_use]
    pub fn snap_rotate(&self, value: f32) -> f32 {
        if self.enabled && self.rotate > 0.0 {
            (value / self.rotate).round() * self.rotate
        } else {
            value
        }
    }

    /// Apply scale snap to a value.
    #[must_use]
    pub fn snap_scale(&self, value: f32) -> f32 {
        if self.enabled && self.scale > 0.0 {
            (value / self.scale).round() * self.scale
        } else {
            value
        }
    }
}

// ---------------------------------------------------------------------------
// Selection Listener
// ---------------------------------------------------------------------------

/// Trait for receiving selection state change notifications.
///
/// Implement this trait to receive callbacks when selection state changes.
/// Listeners must be Send + Sync for thread safety.
pub trait SelectionListener: Send + Sync {
    /// Called when the primary selection changes.
    fn on_selection_changed(&mut self, old: Option<u64>, new: Option<u64>);

    /// Called when the hovered entity changes.
    fn on_hover_changed(&mut self, entity_id: Option<u64>);

    /// Called when the gizmo mode changes.
    fn on_gizmo_mode_changed(&mut self, mode: GizmoMode);

    /// Called when the gizmo space changes.
    fn on_gizmo_space_changed(&mut self, space: GizmoSpace) {
        let _ = space; // Default implementation does nothing
    }

    /// Called when multi-selection changes.
    fn on_multi_selection_changed(&mut self, selection: &[u64]) {
        let _ = selection; // Default implementation does nothing
    }

    /// Called when snap settings change.
    fn on_snap_changed(&mut self, settings: &SnapSettings) {
        let _ = settings; // Default implementation does nothing
    }
}

// ---------------------------------------------------------------------------
// Selection State
// ---------------------------------------------------------------------------

/// Internal selection state data.
///
/// This struct holds the actual selection data and is wrapped by
/// `SharedSelectionState` for thread-safe access.
#[derive(Debug, Clone)]
pub struct SelectionState {
    /// Currently selected entity ID (primary selection).
    pub selected_entity_id: Option<u64>,
    /// Currently hovered entity ID.
    pub hovered_entity_id: Option<u64>,
    /// Multi-selection list (includes primary selection).
    pub multi_selection: Vec<u64>,
    /// Current gizmo transformation mode.
    pub gizmo_mode: GizmoMode,
    /// Current gizmo coordinate space.
    pub gizmo_space: GizmoSpace,
    /// Whether snap is enabled.
    pub snap_enabled: bool,
    /// Translation snap increment.
    pub snap_translate: f32,
    /// Rotation snap increment (degrees).
    pub snap_rotate: f32,
    /// Scale snap increment.
    pub snap_scale: f32,
}

impl Default for SelectionState {
    fn default() -> Self {
        Self {
            selected_entity_id: None,
            hovered_entity_id: None,
            multi_selection: Vec::new(),
            gizmo_mode: GizmoMode::default(),
            gizmo_space: GizmoSpace::default(),
            snap_enabled: false,
            snap_translate: 1.0,
            snap_rotate: 15.0,
            snap_scale: 0.1,
        }
    }
}

impl SelectionState {
    /// Create a new selection state with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Get snap settings as a struct.
    #[must_use]
    pub fn snap_settings(&self) -> SnapSettings {
        SnapSettings {
            enabled: self.snap_enabled,
            translate: self.snap_translate,
            rotate: self.snap_rotate,
            scale: self.snap_scale,
        }
    }

    /// Check if an entity is selected (in multi-selection).
    #[must_use]
    pub fn is_selected(&self, entity_id: u64) -> bool {
        self.multi_selection.contains(&entity_id)
    }

    /// Check if an entity is the primary selection.
    #[must_use]
    pub fn is_primary_selection(&self, entity_id: u64) -> bool {
        self.selected_entity_id == Some(entity_id)
    }

    /// Check if multi-selection is active (more than one entity selected).
    #[must_use]
    pub fn has_multi_selection(&self) -> bool {
        self.multi_selection.len() > 1
    }

    /// Get the number of selected entities.
    #[must_use]
    pub fn selection_count(&self) -> usize {
        self.multi_selection.len()
    }
}

// ---------------------------------------------------------------------------
// Shared Selection State
// ---------------------------------------------------------------------------

/// Thread-safe wrapper for `SelectionState`.
///
/// Provides atomic access to selection state from multiple editor panels.
/// Uses `Arc<RwLock<...>>` for interior mutability and thread safety.
#[derive(Clone)]
pub struct SharedSelectionState {
    inner: Arc<RwLock<SelectionState>>,
    listeners: Arc<RwLock<Vec<Box<dyn SelectionListener>>>>,
}

impl std::fmt::Debug for SharedSelectionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.inner.read().unwrap();
        f.debug_struct("SharedSelectionState")
            .field("selected_entity_id", &state.selected_entity_id)
            .field("hovered_entity_id", &state.hovered_entity_id)
            .field("multi_selection", &state.multi_selection)
            .field("gizmo_mode", &state.gizmo_mode)
            .field("gizmo_space", &state.gizmo_space)
            .finish()
    }
}

impl Default for SharedSelectionState {
    fn default() -> Self {
        Self::new()
    }
}

/// A read guard for the selection state.
pub struct SelectionStateGuard<'a> {
    guard: RwLockReadGuard<'a, SelectionState>,
}

impl<'a> Deref for SelectionStateGuard<'a> {
    type Target = SelectionState;

    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

impl SharedSelectionState {
    /// Create a new shared selection state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(SelectionState::new())),
            listeners: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Register a listener for selection change events.
    pub fn add_listener(&self, listener: Box<dyn SelectionListener>) {
        let mut listeners = self.listeners.write().unwrap();
        listeners.push(listener);
    }

    /// Remove all listeners.
    pub fn clear_listeners(&self) {
        let mut listeners = self.listeners.write().unwrap();
        listeners.clear();
    }

    /// Get the number of registered listeners.
    #[must_use]
    pub fn listener_count(&self) -> usize {
        let listeners = self.listeners.read().unwrap();
        listeners.len()
    }

    /// Get a read-only view of the selection state.
    #[must_use]
    pub fn read(&self) -> SelectionStateGuard<'_> {
        SelectionStateGuard {
            guard: self.inner.read().unwrap(),
        }
    }

    /// Select an entity (replaces current selection).
    pub fn select(&self, entity_id: Option<u64>) {
        let old;
        {
            let mut state = self.inner.write().unwrap();
            old = state.selected_entity_id;
            state.selected_entity_id = entity_id;
            state.multi_selection.clear();
            if let Some(id) = entity_id {
                state.multi_selection.push(id);
            }
        }

        // Notify listeners
        if old != entity_id {
            let mut listeners = self.listeners.write().unwrap();
            for listener in listeners.iter_mut() {
                listener.on_selection_changed(old, entity_id);
            }
        }
    }

    /// Set hover state.
    pub fn hover(&self, entity_id: Option<u64>) {
        let changed;
        {
            let mut state = self.inner.write().unwrap();
            changed = state.hovered_entity_id != entity_id;
            state.hovered_entity_id = entity_id;
        }

        if changed {
            let mut listeners = self.listeners.write().unwrap();
            for listener in listeners.iter_mut() {
                listener.on_hover_changed(entity_id);
            }
        }
    }

    /// Toggle entity selection (add if not selected, remove if selected).
    pub fn toggle_selection(&self, entity_id: u64) {
        let (was_selected, new_primary, old_selection);
        {
            let mut state = self.inner.write().unwrap();
            old_selection = state.multi_selection.clone();
            was_selected = state.multi_selection.contains(&entity_id);

            if was_selected {
                state.multi_selection.retain(|&id| id != entity_id);
                // Update primary selection
                if state.selected_entity_id == Some(entity_id) {
                    state.selected_entity_id = state.multi_selection.first().copied();
                }
            } else {
                state.multi_selection.push(entity_id);
                // First entity becomes primary
                if state.selected_entity_id.is_none() {
                    state.selected_entity_id = Some(entity_id);
                }
            }
            new_primary = state.selected_entity_id;
        }

        // Notify listeners about selection change
        let state = self.inner.read().unwrap();
        if state.multi_selection != old_selection {
            drop(state);
            let mut listeners = self.listeners.write().unwrap();
            let state = self.inner.read().unwrap();
            for listener in listeners.iter_mut() {
                listener.on_multi_selection_changed(&state.multi_selection);
            }
            drop(state);
            drop(listeners);

            // Notify primary selection change if needed
            let old_primary = if was_selected {
                Some(entity_id)
            } else {
                None
            };
            if old_primary.is_some() || new_primary != old_selection.first().copied() {
                let mut listeners = self.listeners.write().unwrap();
                for listener in listeners.iter_mut() {
                    if was_selected && old_selection.first() == Some(&entity_id) {
                        listener.on_selection_changed(Some(entity_id), new_primary);
                    }
                }
            }
        }
    }

    /// Add entity to multi-selection without removing others.
    pub fn add_to_selection(&self, entity_id: u64) {
        let added;
        {
            let mut state = self.inner.write().unwrap();
            added = !state.multi_selection.contains(&entity_id);
            if added {
                state.multi_selection.push(entity_id);
                // Set as primary if first selection
                if state.selected_entity_id.is_none() {
                    state.selected_entity_id = Some(entity_id);
                }
            }
        }

        if added {
            let state = self.inner.read().unwrap();
            let selection = state.multi_selection.clone();
            drop(state);
            let mut listeners = self.listeners.write().unwrap();
            for listener in listeners.iter_mut() {
                listener.on_multi_selection_changed(&selection);
            }
        }
    }

    /// Remove entity from multi-selection.
    pub fn remove_from_selection(&self, entity_id: u64) {
        let (removed, new_primary);
        {
            let mut state = self.inner.write().unwrap();
            let old_len = state.multi_selection.len();
            state.multi_selection.retain(|&id| id != entity_id);
            removed = state.multi_selection.len() < old_len;

            if removed && state.selected_entity_id == Some(entity_id) {
                state.selected_entity_id = state.multi_selection.first().copied();
            }
            new_primary = state.selected_entity_id;
        }

        if removed {
            let state = self.inner.read().unwrap();
            let selection = state.multi_selection.clone();
            drop(state);

            let mut listeners = self.listeners.write().unwrap();
            for listener in listeners.iter_mut() {
                listener.on_multi_selection_changed(&selection);
                if new_primary.is_none() || new_primary == Some(entity_id) {
                    listener.on_selection_changed(Some(entity_id), new_primary);
                }
            }
        }
    }

    /// Clear all selection.
    pub fn clear_selection(&self) {
        let had_selection;
        let old_primary;
        {
            let mut state = self.inner.write().unwrap();
            had_selection = !state.multi_selection.is_empty();
            old_primary = state.selected_entity_id;
            state.selected_entity_id = None;
            state.multi_selection.clear();
        }

        if had_selection {
            let mut listeners = self.listeners.write().unwrap();
            for listener in listeners.iter_mut() {
                listener.on_selection_changed(old_primary, None);
                listener.on_multi_selection_changed(&[]);
            }
        }
    }

    /// Set gizmo mode.
    pub fn set_gizmo_mode(&self, mode: GizmoMode) {
        let changed;
        {
            let mut state = self.inner.write().unwrap();
            changed = state.gizmo_mode != mode;
            state.gizmo_mode = mode;
        }

        if changed {
            let mut listeners = self.listeners.write().unwrap();
            for listener in listeners.iter_mut() {
                listener.on_gizmo_mode_changed(mode);
            }
        }
    }

    /// Cycle through gizmo modes (Translate -> Rotate -> Scale -> Translate).
    pub fn cycle_gizmo_mode(&self) {
        let new_mode;
        {
            let mut state = self.inner.write().unwrap();
            new_mode = state.gizmo_mode.next();
            state.gizmo_mode = new_mode;
        }

        let mut listeners = self.listeners.write().unwrap();
        for listener in listeners.iter_mut() {
            listener.on_gizmo_mode_changed(new_mode);
        }
    }

    /// Toggle gizmo space between Local and World.
    pub fn toggle_gizmo_space(&self) {
        let new_space;
        {
            let mut state = self.inner.write().unwrap();
            new_space = state.gizmo_space.toggle();
            state.gizmo_space = new_space;
        }

        let mut listeners = self.listeners.write().unwrap();
        for listener in listeners.iter_mut() {
            listener.on_gizmo_space_changed(new_space);
        }
    }

    /// Set gizmo space.
    pub fn set_gizmo_space(&self, space: GizmoSpace) {
        let changed;
        {
            let mut state = self.inner.write().unwrap();
            changed = state.gizmo_space != space;
            state.gizmo_space = space;
        }

        if changed {
            let mut listeners = self.listeners.write().unwrap();
            for listener in listeners.iter_mut() {
                listener.on_gizmo_space_changed(space);
            }
        }
    }

    /// Enable or disable snap.
    pub fn set_snap_enabled(&self, enabled: bool) {
        let settings;
        {
            let mut state = self.inner.write().unwrap();
            state.snap_enabled = enabled;
            settings = state.snap_settings();
        }

        let mut listeners = self.listeners.write().unwrap();
        for listener in listeners.iter_mut() {
            listener.on_snap_changed(&settings);
        }
    }

    /// Set snap settings.
    pub fn set_snap_settings(&self, settings: SnapSettings) {
        {
            let mut state = self.inner.write().unwrap();
            state.snap_enabled = settings.enabled;
            state.snap_translate = settings.translate;
            state.snap_rotate = settings.rotate;
            state.snap_scale = settings.scale;
        }

        let mut listeners = self.listeners.write().unwrap();
        for listener in listeners.iter_mut() {
            listener.on_snap_changed(&settings);
        }
    }

    /// Handle keyboard shortcut for gizmo mode.
    ///
    /// Returns true if the key was handled.
    pub fn handle_gizmo_shortcut(&self, key: char) -> bool {
        if let Some(mode) = GizmoMode::from_shortcut(key) {
            self.set_gizmo_mode(mode);
            true
        } else {
            false
        }
    }

    /// Select a range of entities (for shift-click selection).
    pub fn select_range(&self, entity_ids: &[u64]) {
        if entity_ids.is_empty() {
            return;
        }

        let old_primary;
        {
            let mut state = self.inner.write().unwrap();
            old_primary = state.selected_entity_id;
            state.multi_selection.clear();
            state.multi_selection.extend_from_slice(entity_ids);
            state.selected_entity_id = Some(entity_ids[0]);
        }

        let state = self.inner.read().unwrap();
        let selection = state.multi_selection.clone();
        let new_primary = state.selected_entity_id;
        drop(state);

        let mut listeners = self.listeners.write().unwrap();
        for listener in listeners.iter_mut() {
            if old_primary != new_primary {
                listener.on_selection_changed(old_primary, new_primary);
            }
            listener.on_multi_selection_changed(&selection);
        }
    }

    /// Get currently selected entity IDs.
    #[must_use]
    pub fn selected_entities(&self) -> Vec<u64> {
        let state = self.inner.read().unwrap();
        state.multi_selection.clone()
    }

    /// Get primary selected entity ID.
    #[must_use]
    pub fn primary_selection(&self) -> Option<u64> {
        let state = self.inner.read().unwrap();
        state.selected_entity_id
    }

    /// Get current gizmo mode.
    #[must_use]
    pub fn gizmo_mode(&self) -> GizmoMode {
        let state = self.inner.read().unwrap();
        state.gizmo_mode
    }

    /// Get current gizmo space.
    #[must_use]
    pub fn gizmo_space(&self) -> GizmoSpace {
        let state = self.inner.read().unwrap();
        state.gizmo_space
    }

    /// Get current snap settings.
    #[must_use]
    pub fn snap_settings(&self) -> SnapSettings {
        let state = self.inner.read().unwrap();
        state.snap_settings()
    }

    /// Check if an entity is selected.
    #[must_use]
    pub fn is_selected(&self, entity_id: u64) -> bool {
        let state = self.inner.read().unwrap();
        state.is_selected(entity_id)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::thread;

    // Test listener for tracking notifications
    struct TestListener {
        selection_changes: Arc<AtomicU32>,
        hover_changes: Arc<AtomicU32>,
        gizmo_changes: Arc<AtomicU32>,
        space_changes: Arc<AtomicU32>,
        multi_changes: Arc<AtomicU32>,
        snap_changes: Arc<AtomicU32>,
    }

    impl TestListener {
        fn new() -> Self {
            Self {
                selection_changes: Arc::new(AtomicU32::new(0)),
                hover_changes: Arc::new(AtomicU32::new(0)),
                gizmo_changes: Arc::new(AtomicU32::new(0)),
                space_changes: Arc::new(AtomicU32::new(0)),
                multi_changes: Arc::new(AtomicU32::new(0)),
                snap_changes: Arc::new(AtomicU32::new(0)),
            }
        }

        fn selection_count(&self) -> u32 {
            self.selection_changes.load(Ordering::SeqCst)
        }

        fn hover_count(&self) -> u32 {
            self.hover_changes.load(Ordering::SeqCst)
        }

        fn gizmo_count(&self) -> u32 {
            self.gizmo_changes.load(Ordering::SeqCst)
        }

        fn space_count(&self) -> u32 {
            self.space_changes.load(Ordering::SeqCst)
        }

        fn multi_count(&self) -> u32 {
            self.multi_changes.load(Ordering::SeqCst)
        }

        fn snap_count(&self) -> u32 {
            self.snap_changes.load(Ordering::SeqCst)
        }
    }

    impl SelectionListener for TestListener {
        fn on_selection_changed(&mut self, _old: Option<u64>, _new: Option<u64>) {
            self.selection_changes.fetch_add(1, Ordering::SeqCst);
        }

        fn on_hover_changed(&mut self, _entity_id: Option<u64>) {
            self.hover_changes.fetch_add(1, Ordering::SeqCst);
        }

        fn on_gizmo_mode_changed(&mut self, _mode: GizmoMode) {
            self.gizmo_changes.fetch_add(1, Ordering::SeqCst);
        }

        fn on_gizmo_space_changed(&mut self, _space: GizmoSpace) {
            self.space_changes.fetch_add(1, Ordering::SeqCst);
        }

        fn on_multi_selection_changed(&mut self, _selection: &[u64]) {
            self.multi_changes.fetch_add(1, Ordering::SeqCst);
        }

        fn on_snap_changed(&mut self, _settings: &SnapSettings) {
            self.snap_changes.fetch_add(1, Ordering::SeqCst);
        }
    }

    // ---------------------------------------------------------------------------
    // GizmoMode Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_gizmo_mode_default() {
        assert_eq!(GizmoMode::default(), GizmoMode::Translate);
    }

    #[test]
    fn test_gizmo_mode_next() {
        assert_eq!(GizmoMode::Translate.next(), GizmoMode::Rotate);
        assert_eq!(GizmoMode::Rotate.next(), GizmoMode::Scale);
        assert_eq!(GizmoMode::Scale.next(), GizmoMode::Translate);
        assert_eq!(GizmoMode::None.next(), GizmoMode::Translate);
    }

    #[test]
    fn test_gizmo_mode_shortcut() {
        assert_eq!(GizmoMode::Translate.shortcut(), Some('W'));
        assert_eq!(GizmoMode::Rotate.shortcut(), Some('E'));
        assert_eq!(GizmoMode::Scale.shortcut(), Some('R'));
        assert_eq!(GizmoMode::None.shortcut(), None);
    }

    #[test]
    fn test_gizmo_mode_from_shortcut() {
        assert_eq!(GizmoMode::from_shortcut('W'), Some(GizmoMode::Translate));
        assert_eq!(GizmoMode::from_shortcut('w'), Some(GizmoMode::Translate));
        assert_eq!(GizmoMode::from_shortcut('E'), Some(GizmoMode::Rotate));
        assert_eq!(GizmoMode::from_shortcut('R'), Some(GizmoMode::Scale));
        assert_eq!(GizmoMode::from_shortcut('X'), None);
    }

    #[test]
    fn test_gizmo_mode_name() {
        assert_eq!(GizmoMode::Translate.name(), "Translate");
        assert_eq!(GizmoMode::Rotate.name(), "Rotate");
        assert_eq!(GizmoMode::Scale.name(), "Scale");
        assert_eq!(GizmoMode::None.name(), "None");
    }

    // ---------------------------------------------------------------------------
    // GizmoSpace Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_gizmo_space_default() {
        assert_eq!(GizmoSpace::default(), GizmoSpace::Local);
    }

    #[test]
    fn test_gizmo_space_toggle() {
        assert_eq!(GizmoSpace::Local.toggle(), GizmoSpace::World);
        assert_eq!(GizmoSpace::World.toggle(), GizmoSpace::Local);
    }

    #[test]
    fn test_gizmo_space_name() {
        assert_eq!(GizmoSpace::Local.name(), "Local");
        assert_eq!(GizmoSpace::World.name(), "World");
    }

    // ---------------------------------------------------------------------------
    // SnapSettings Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_snap_settings_default() {
        let snap = SnapSettings::default();
        assert!(!snap.enabled);
        assert_eq!(snap.translate, 1.0);
        assert_eq!(snap.rotate, 15.0);
        assert_eq!(snap.scale, 0.1);
    }

    #[test]
    fn test_snap_settings_new() {
        let snap = SnapSettings::new(2.0, 30.0, 0.25);
        assert!(!snap.enabled);
        assert_eq!(snap.translate, 2.0);
        assert_eq!(snap.rotate, 30.0);
        assert_eq!(snap.scale, 0.25);
    }

    #[test]
    fn test_snap_settings_with_enabled() {
        let snap = SnapSettings::default().with_enabled(true);
        assert!(snap.enabled);
    }

    #[test]
    fn test_snap_translate_disabled() {
        let snap = SnapSettings::default();
        assert_eq!(snap.snap_translate(1.7), 1.7);
    }

    #[test]
    fn test_snap_translate_enabled() {
        let snap = SnapSettings::default().with_enabled(true);
        assert_eq!(snap.snap_translate(1.7), 2.0);
        assert_eq!(snap.snap_translate(1.2), 1.0);
        assert_eq!(snap.snap_translate(0.4), 0.0); // rounds to nearest (0.5 rounds to 1.0 due to banker's rounding)
        assert_eq!(snap.snap_translate(2.5), 3.0); // 2.5 rounds up
    }

    #[test]
    fn test_snap_rotate_enabled() {
        let snap = SnapSettings::default().with_enabled(true);
        assert_eq!(snap.snap_rotate(20.0), 15.0); // closer to 15 than 30
        assert_eq!(snap.snap_rotate(38.0), 45.0);
        assert_eq!(snap.snap_rotate(8.0), 15.0);  // closer to 15 than 0
    }

    #[test]
    fn test_snap_scale_enabled() {
        let snap = SnapSettings::default().with_enabled(true);
        assert_eq!(snap.snap_scale(0.17), 0.2);
        assert_eq!(snap.snap_scale(0.12), 0.1);
    }

    // ---------------------------------------------------------------------------
    // SelectionState Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_selection_state_default() {
        let state = SelectionState::default();
        assert_eq!(state.selected_entity_id, None);
        assert_eq!(state.hovered_entity_id, None);
        assert!(state.multi_selection.is_empty());
        assert_eq!(state.gizmo_mode, GizmoMode::Translate);
        assert_eq!(state.gizmo_space, GizmoSpace::Local);
        assert!(!state.snap_enabled);
    }

    #[test]
    fn test_selection_state_is_selected() {
        let mut state = SelectionState::new();
        state.multi_selection = vec![1, 2, 3];
        assert!(state.is_selected(1));
        assert!(state.is_selected(2));
        assert!(state.is_selected(3));
        assert!(!state.is_selected(4));
    }

    #[test]
    fn test_selection_state_is_primary_selection() {
        let mut state = SelectionState::new();
        state.selected_entity_id = Some(42);
        assert!(state.is_primary_selection(42));
        assert!(!state.is_primary_selection(43));
    }

    #[test]
    fn test_selection_state_has_multi_selection() {
        let mut state = SelectionState::new();
        assert!(!state.has_multi_selection());
        state.multi_selection = vec![1];
        assert!(!state.has_multi_selection());
        state.multi_selection = vec![1, 2];
        assert!(state.has_multi_selection());
    }

    #[test]
    fn test_selection_state_selection_count() {
        let mut state = SelectionState::new();
        assert_eq!(state.selection_count(), 0);
        state.multi_selection = vec![1, 2, 3];
        assert_eq!(state.selection_count(), 3);
    }

    // ---------------------------------------------------------------------------
    // SharedSelectionState Basic Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_shared_selection_state_new() {
        let state = SharedSelectionState::new();
        assert_eq!(state.primary_selection(), None);
        assert!(state.selected_entities().is_empty());
    }

    #[test]
    fn test_shared_selection_state_default() {
        let state = SharedSelectionState::default();
        assert_eq!(state.primary_selection(), None);
    }

    #[test]
    fn test_shared_selection_state_select() {
        let state = SharedSelectionState::new();
        state.select(Some(42));
        assert_eq!(state.primary_selection(), Some(42));
        assert_eq!(state.selected_entities(), vec![42]);
    }

    #[test]
    fn test_shared_selection_state_select_none() {
        let state = SharedSelectionState::new();
        state.select(Some(42));
        state.select(None);
        assert_eq!(state.primary_selection(), None);
        assert!(state.selected_entities().is_empty());
    }

    #[test]
    fn test_shared_selection_state_hover() {
        let state = SharedSelectionState::new();
        state.hover(Some(42));
        assert_eq!(state.read().hovered_entity_id, Some(42));
        state.hover(None);
        assert_eq!(state.read().hovered_entity_id, None);
    }

    // ---------------------------------------------------------------------------
    // Multi-Selection Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_toggle_selection_add() {
        let state = SharedSelectionState::new();
        state.toggle_selection(1);
        assert!(state.is_selected(1));
        assert_eq!(state.primary_selection(), Some(1));
    }

    #[test]
    fn test_toggle_selection_remove() {
        let state = SharedSelectionState::new();
        state.select(Some(1));
        state.toggle_selection(1);
        assert!(!state.is_selected(1));
        assert_eq!(state.primary_selection(), None);
    }

    #[test]
    fn test_add_to_selection() {
        let state = SharedSelectionState::new();
        state.select(Some(1));
        state.add_to_selection(2);
        state.add_to_selection(3);
        assert_eq!(state.selected_entities(), vec![1, 2, 3]);
        assert_eq!(state.primary_selection(), Some(1)); // Primary unchanged
    }

    #[test]
    fn test_add_to_selection_duplicate() {
        let state = SharedSelectionState::new();
        state.select(Some(1));
        state.add_to_selection(1);
        assert_eq!(state.selected_entities(), vec![1]); // No duplicate
    }

    #[test]
    fn test_remove_from_selection() {
        let state = SharedSelectionState::new();
        state.select(Some(1));
        state.add_to_selection(2);
        state.add_to_selection(3);
        state.remove_from_selection(2);
        assert_eq!(state.selected_entities(), vec![1, 3]);
    }

    #[test]
    fn test_remove_from_selection_primary() {
        let state = SharedSelectionState::new();
        state.select(Some(1));
        state.add_to_selection(2);
        state.remove_from_selection(1);
        assert_eq!(state.primary_selection(), Some(2)); // Primary updated
    }

    #[test]
    fn test_clear_selection() {
        let state = SharedSelectionState::new();
        state.select(Some(1));
        state.add_to_selection(2);
        state.clear_selection();
        assert!(state.selected_entities().is_empty());
        assert_eq!(state.primary_selection(), None);
    }

    #[test]
    fn test_select_range() {
        let state = SharedSelectionState::new();
        state.select_range(&[10, 20, 30]);
        assert_eq!(state.selected_entities(), vec![10, 20, 30]);
        assert_eq!(state.primary_selection(), Some(10));
    }

    #[test]
    fn test_select_range_empty() {
        let state = SharedSelectionState::new();
        state.select(Some(1));
        state.select_range(&[]);
        // Empty range should not change selection
        assert_eq!(state.primary_selection(), Some(1));
    }

    // ---------------------------------------------------------------------------
    // Gizmo Mode Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_set_gizmo_mode() {
        let state = SharedSelectionState::new();
        state.set_gizmo_mode(GizmoMode::Rotate);
        assert_eq!(state.gizmo_mode(), GizmoMode::Rotate);
    }

    #[test]
    fn test_cycle_gizmo_mode() {
        let state = SharedSelectionState::new();
        assert_eq!(state.gizmo_mode(), GizmoMode::Translate);
        state.cycle_gizmo_mode();
        assert_eq!(state.gizmo_mode(), GizmoMode::Rotate);
        state.cycle_gizmo_mode();
        assert_eq!(state.gizmo_mode(), GizmoMode::Scale);
        state.cycle_gizmo_mode();
        assert_eq!(state.gizmo_mode(), GizmoMode::Translate);
    }

    #[test]
    fn test_handle_gizmo_shortcut() {
        let state = SharedSelectionState::new();
        assert!(state.handle_gizmo_shortcut('E'));
        assert_eq!(state.gizmo_mode(), GizmoMode::Rotate);
        assert!(state.handle_gizmo_shortcut('R'));
        assert_eq!(state.gizmo_mode(), GizmoMode::Scale);
        assert!(state.handle_gizmo_shortcut('W'));
        assert_eq!(state.gizmo_mode(), GizmoMode::Translate);
        assert!(!state.handle_gizmo_shortcut('X')); // Invalid key
    }

    // ---------------------------------------------------------------------------
    // Gizmo Space Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_set_gizmo_space() {
        let state = SharedSelectionState::new();
        state.set_gizmo_space(GizmoSpace::World);
        assert_eq!(state.gizmo_space(), GizmoSpace::World);
    }

    #[test]
    fn test_toggle_gizmo_space() {
        let state = SharedSelectionState::new();
        assert_eq!(state.gizmo_space(), GizmoSpace::Local);
        state.toggle_gizmo_space();
        assert_eq!(state.gizmo_space(), GizmoSpace::World);
        state.toggle_gizmo_space();
        assert_eq!(state.gizmo_space(), GizmoSpace::Local);
    }

    // ---------------------------------------------------------------------------
    // Snap Settings Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_set_snap_enabled() {
        let state = SharedSelectionState::new();
        state.set_snap_enabled(true);
        assert!(state.snap_settings().enabled);
        state.set_snap_enabled(false);
        assert!(!state.snap_settings().enabled);
    }

    #[test]
    fn test_set_snap_settings() {
        let state = SharedSelectionState::new();
        let settings = SnapSettings::new(2.0, 45.0, 0.5).with_enabled(true);
        state.set_snap_settings(settings);
        let s = state.snap_settings();
        assert!(s.enabled);
        assert_eq!(s.translate, 2.0);
        assert_eq!(s.rotate, 45.0);
        assert_eq!(s.scale, 0.5);
    }

    // ---------------------------------------------------------------------------
    // Listener Notification Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_listener_selection_notification() {
        let state = SharedSelectionState::new();
        let listener = TestListener::new();
        let selection_counter = listener.selection_changes.clone();
        state.add_listener(Box::new(listener));

        state.select(Some(1));
        assert_eq!(selection_counter.load(Ordering::SeqCst), 1);

        state.select(Some(2));
        assert_eq!(selection_counter.load(Ordering::SeqCst), 2);

        // Same selection should not notify
        state.select(Some(2));
        assert_eq!(selection_counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_listener_hover_notification() {
        let state = SharedSelectionState::new();
        let listener = TestListener::new();
        let hover_counter = listener.hover_changes.clone();
        state.add_listener(Box::new(listener));

        state.hover(Some(1));
        assert_eq!(hover_counter.load(Ordering::SeqCst), 1);

        state.hover(Some(2));
        assert_eq!(hover_counter.load(Ordering::SeqCst), 2);

        // Same hover should not notify
        state.hover(Some(2));
        assert_eq!(hover_counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_listener_gizmo_mode_notification() {
        let state = SharedSelectionState::new();
        let listener = TestListener::new();
        let gizmo_counter = listener.gizmo_changes.clone();
        state.add_listener(Box::new(listener));

        state.set_gizmo_mode(GizmoMode::Rotate);
        assert_eq!(gizmo_counter.load(Ordering::SeqCst), 1);

        state.cycle_gizmo_mode(); // Rotate -> Scale
        assert_eq!(gizmo_counter.load(Ordering::SeqCst), 2);

        // Same mode should not notify
        state.set_gizmo_mode(GizmoMode::Scale);
        assert_eq!(gizmo_counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_listener_gizmo_space_notification() {
        let state = SharedSelectionState::new();
        let listener = TestListener::new();
        let space_counter = listener.space_changes.clone();
        state.add_listener(Box::new(listener));

        state.toggle_gizmo_space();
        assert_eq!(space_counter.load(Ordering::SeqCst), 1);

        state.set_gizmo_space(GizmoSpace::Local);
        assert_eq!(space_counter.load(Ordering::SeqCst), 2);

        // Same space should not notify
        state.set_gizmo_space(GizmoSpace::Local);
        assert_eq!(space_counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_listener_multi_selection_notification() {
        let state = SharedSelectionState::new();
        let listener = TestListener::new();
        let multi_counter = listener.multi_changes.clone();
        state.add_listener(Box::new(listener));

        state.add_to_selection(1);
        assert_eq!(multi_counter.load(Ordering::SeqCst), 1);

        state.add_to_selection(2);
        assert_eq!(multi_counter.load(Ordering::SeqCst), 2);

        state.remove_from_selection(1);
        assert_eq!(multi_counter.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_listener_snap_notification() {
        let state = SharedSelectionState::new();
        let listener = TestListener::new();
        let snap_counter = listener.snap_changes.clone();
        state.add_listener(Box::new(listener));

        state.set_snap_enabled(true);
        assert_eq!(snap_counter.load(Ordering::SeqCst), 1);

        state.set_snap_settings(SnapSettings::new(2.0, 30.0, 0.2));
        assert_eq!(snap_counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_clear_listeners() {
        let state = SharedSelectionState::new();
        let listener = TestListener::new();
        let selection_counter = listener.selection_changes.clone();
        state.add_listener(Box::new(listener));
        assert_eq!(state.listener_count(), 1);

        state.clear_listeners();
        assert_eq!(state.listener_count(), 0);

        // No notifications after clearing
        state.select(Some(1));
        assert_eq!(selection_counter.load(Ordering::SeqCst), 0);
    }

    // ---------------------------------------------------------------------------
    // Thread Safety Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_concurrent_reads() {
        let state = Arc::new(SharedSelectionState::new());
        state.select(Some(42));
        state.add_to_selection(43);
        state.add_to_selection(44);

        let handles: Vec<_> = (0..10)
            .map(|_| {
                let s = state.clone();
                thread::spawn(move || {
                    for _ in 0..100 {
                        let _ = s.primary_selection();
                        let _ = s.selected_entities();
                        let _ = s.gizmo_mode();
                        let _ = s.is_selected(42);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // State should be unchanged
        assert_eq!(state.primary_selection(), Some(42));
        assert_eq!(state.selected_entities().len(), 3);
    }

    #[test]
    fn test_concurrent_writes() {
        let state = Arc::new(SharedSelectionState::new());

        let handles: Vec<_> = (0..4)
            .map(|i| {
                let s = state.clone();
                thread::spawn(move || {
                    for j in 0..25 {
                        let entity = (i * 100 + j) as u64;
                        s.add_to_selection(entity);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // All entities should be added (100 total)
        assert_eq!(state.selected_entities().len(), 100);
    }

    #[test]
    fn test_concurrent_select_clear() {
        let state = Arc::new(SharedSelectionState::new());

        let handles: Vec<_> = (0..4)
            .map(|i| {
                let s = state.clone();
                thread::spawn(move || {
                    for _ in 0..50 {
                        if i % 2 == 0 {
                            s.select(Some(i as u64));
                        } else {
                            s.clear_selection();
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // Final state should be consistent (either some selection or none)
        let selection = state.primary_selection();
        let entities = state.selected_entities();
        if selection.is_some() {
            assert!(!entities.is_empty());
        } else {
            assert!(entities.is_empty());
        }
    }

    #[test]
    fn test_concurrent_gizmo_cycling() {
        let state = Arc::new(SharedSelectionState::new());

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let s = state.clone();
                thread::spawn(move || {
                    for _ in 0..100 {
                        s.cycle_gizmo_mode();
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // Mode should be valid
        let mode = state.gizmo_mode();
        assert!(matches!(
            mode,
            GizmoMode::Translate | GizmoMode::Rotate | GizmoMode::Scale
        ));
    }

    // ---------------------------------------------------------------------------
    // Read Guard Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_read_guard_deref() {
        let state = SharedSelectionState::new();
        state.select(Some(42));
        state.set_gizmo_mode(GizmoMode::Scale);

        let guard = state.read();
        assert_eq!(guard.selected_entity_id, Some(42));
        assert_eq!(guard.gizmo_mode, GizmoMode::Scale);
        assert!(guard.is_selected(42));
    }

    #[test]
    fn test_read_guard_snap_settings() {
        let state = SharedSelectionState::new();
        state.set_snap_settings(SnapSettings::new(5.0, 90.0, 1.0).with_enabled(true));

        let guard = state.read();
        let snap = guard.snap_settings();
        assert!(snap.enabled);
        assert_eq!(snap.translate, 5.0);
        assert_eq!(snap.rotate, 90.0);
        assert_eq!(snap.scale, 1.0);
    }

    // ---------------------------------------------------------------------------
    // Debug Trait Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_debug_shared_selection_state() {
        let state = SharedSelectionState::new();
        state.select(Some(42));
        let debug_str = format!("{:?}", state);
        assert!(debug_str.contains("SharedSelectionState"));
        assert!(debug_str.contains("42"));
    }

    // ---------------------------------------------------------------------------
    // Clone Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_shared_selection_state_clone() {
        let state1 = SharedSelectionState::new();
        state1.select(Some(42));

        let state2 = state1.clone();
        assert_eq!(state2.primary_selection(), Some(42));

        // Both share the same underlying state
        state2.select(Some(100));
        assert_eq!(state1.primary_selection(), Some(100));
    }

    // ---------------------------------------------------------------------------
    // Edge Case Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_remove_from_empty_selection() {
        let state = SharedSelectionState::new();
        state.remove_from_selection(42); // Should not panic
        assert!(state.selected_entities().is_empty());
    }

    #[test]
    fn test_clear_empty_selection() {
        let state = SharedSelectionState::new();
        state.clear_selection(); // Should not panic or notify
        assert!(state.selected_entities().is_empty());
    }

    #[test]
    fn test_add_then_select_clears_multi() {
        let state = SharedSelectionState::new();
        state.add_to_selection(1);
        state.add_to_selection(2);
        state.add_to_selection(3);
        assert_eq!(state.selected_entities().len(), 3);

        // select() should clear multi-selection and set only the new entity
        state.select(Some(42));
        assert_eq!(state.selected_entities(), vec![42]);
        assert_eq!(state.primary_selection(), Some(42));
    }

    #[test]
    fn test_snap_with_zero_increment() {
        let snap = SnapSettings {
            enabled: true,
            translate: 0.0, // Zero increment
            rotate: 0.0,
            scale: 0.0,
        };
        // Should return value unchanged when increment is zero
        assert_eq!(snap.snap_translate(1.5), 1.5);
        assert_eq!(snap.snap_rotate(45.0), 45.0);
        assert_eq!(snap.snap_scale(0.75), 0.75);
    }
}
