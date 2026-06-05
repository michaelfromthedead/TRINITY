//! Component Inspector Panel for the Editor
//!
//! Provides a UI panel for inspecting and editing ECS components.
//! Supports type-aware rendering for common types (primitives, vectors,
//! colors, strings, nested structs) and tracks pending edits.
//!
//! # Architecture
//!
//! ```text
//! InspectorPanel                     UIContext
//! ==============                     =========
//!     │                                  │
//!     ├── set_selection(entity_id) ─────►│
//!     │                                  │
//!     ├── render(ctx, components) ──────►│ (renders UI)
//!     │                                  │
//!     ├── pending_edits() ──────────────►│ (returns edits)
//!     │                                  │
//!     └── clear_edits() ────────────────►│
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use renderer_backend::inspector_panel::InspectorPanel;
//! use renderer_backend::egui_adapter::MockUIContext;
//!
//! let mut panel = InspectorPanel::new();
//! panel.set_selection(Some(42));
//!
//! let components = vec![
//!     (1, "Transform".to_string(), vec![0u8; 48]),
//! ];
//!
//! let mut ctx = MockUIContext::new(1);
//! panel.render(&mut ctx, &components);
//!
//! let edits = panel.pending_edits();
//! ```

use crate::egui_adapter::UIContext;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Value Types
// ---------------------------------------------------------------------------

/// A dynamic value that can represent various component field types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    /// Boolean value.
    Bool(bool),
    /// 32-bit signed integer.
    I32(i32),
    /// 32-bit unsigned integer.
    U32(u32),
    /// 32-bit floating point.
    F32(f32),
    /// 64-bit floating point.
    F64(f64),
    /// String value.
    String(String),
    /// 2D vector (x, y).
    Vec2([f32; 2]),
    /// 3D vector (x, y, z).
    Vec3([f32; 3]),
    /// 4D vector (x, y, z, w).
    Vec4([f32; 4]),
    /// Quaternion (x, y, z, w).
    Quat([f32; 4]),
    /// RGBA color (r, g, b, a).
    Rgba([f32; 4]),
    /// Raw bytes for unknown types.
    Raw(Vec<u8>),
    /// Nested struct with named fields.
    Struct(Vec<(String, Value)>),
}

impl Value {
    /// Get the type name of this value.
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Bool(_) => "bool",
            Value::I32(_) => "i32",
            Value::U32(_) => "u32",
            Value::F32(_) => "f32",
            Value::F64(_) => "f64",
            Value::String(_) => "String",
            Value::Vec2(_) => "Vec2",
            Value::Vec3(_) => "Vec3",
            Value::Vec4(_) => "Vec4",
            Value::Quat(_) => "Quat",
            Value::Rgba(_) => "Rgba",
            Value::Raw(_) => "Raw",
            Value::Struct(_) => "Struct",
        }
    }

    /// Check if this value is equal to another, with floating point tolerance.
    pub fn approx_eq(&self, other: &Value, epsilon: f32) -> bool {
        match (self, other) {
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::I32(a), Value::I32(b)) => a == b,
            (Value::U32(a), Value::U32(b)) => a == b,
            (Value::F32(a), Value::F32(b)) => (a - b).abs() < epsilon,
            (Value::F64(a), Value::F64(b)) => (a - b).abs() < epsilon as f64,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Vec2(a), Value::Vec2(b)) => {
                (a[0] - b[0]).abs() < epsilon && (a[1] - b[1]).abs() < epsilon
            }
            (Value::Vec3(a), Value::Vec3(b)) => {
                (a[0] - b[0]).abs() < epsilon
                    && (a[1] - b[1]).abs() < epsilon
                    && (a[2] - b[2]).abs() < epsilon
            }
            (Value::Vec4(a), Value::Vec4(b))
            | (Value::Quat(a), Value::Quat(b))
            | (Value::Rgba(a), Value::Rgba(b)) => {
                (a[0] - b[0]).abs() < epsilon
                    && (a[1] - b[1]).abs() < epsilon
                    && (a[2] - b[2]).abs() < epsilon
                    && (a[3] - b[3]).abs() < epsilon
            }
            (Value::Raw(a), Value::Raw(b)) => a == b,
            (Value::Struct(a), Value::Struct(b)) => {
                if a.len() != b.len() {
                    return false;
                }
                a.iter()
                    .zip(b.iter())
                    .all(|((name_a, val_a), (name_b, val_b))| {
                        name_a == name_b && val_a.approx_eq(val_b, epsilon)
                    })
            }
            _ => false,
        }
    }
}

impl Default for Value {
    fn default() -> Self {
        Value::Raw(Vec::new())
    }
}

// ---------------------------------------------------------------------------
// Component Edit
// ---------------------------------------------------------------------------

/// Represents a pending edit to a component field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComponentEdit {
    /// The entity being edited.
    pub entity_id: u64,
    /// The component type ID.
    pub component_id: u32,
    /// Path to the field being edited (e.g., "position.x" or "transform.rotation.y").
    pub field_path: String,
    /// The original value before editing.
    pub old_value: Value,
    /// The new value after editing.
    pub new_value: Value,
}

impl ComponentEdit {
    /// Create a new component edit.
    pub fn new(
        entity_id: u64,
        component_id: u32,
        field_path: impl Into<String>,
        old_value: Value,
        new_value: Value,
    ) -> Self {
        Self {
            entity_id,
            component_id,
            field_path: field_path.into(),
            old_value,
            new_value,
        }
    }

    /// Check if this edit is a no-op (old == new).
    pub fn is_noop(&self) -> bool {
        self.old_value.approx_eq(&self.new_value, f32::EPSILON)
    }
}

// ---------------------------------------------------------------------------
// Edit State
// ---------------------------------------------------------------------------

/// Tracks the state of pending edits in the inspector panel.
#[derive(Debug, Clone, Default)]
pub struct EditState {
    /// Pending edits that have not been applied.
    pending: Vec<ComponentEdit>,
    /// Whether an edit is in progress (for undo grouping).
    editing: bool,
    /// The component currently being edited (for focus tracking).
    active_component: Option<u32>,
    /// The field path currently being edited.
    active_field: Option<String>,
}

impl EditState {
    /// Create a new empty edit state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a pending edit.
    pub fn push_edit(&mut self, edit: ComponentEdit) {
        // Don't add no-op edits
        if !edit.is_noop() {
            self.pending.push(edit);
        }
    }

    /// Get all pending edits.
    pub fn pending_edits(&self) -> &[ComponentEdit] {
        &self.pending
    }

    /// Clear all pending edits.
    pub fn clear(&mut self) {
        self.pending.clear();
        self.editing = false;
        self.active_component = None;
        self.active_field = None;
    }

    /// Check if there are any pending edits.
    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Get the number of pending edits.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Begin editing a field.
    pub fn begin_edit(&mut self, component_id: u32, field_path: &str) {
        self.editing = true;
        self.active_component = Some(component_id);
        self.active_field = Some(field_path.to_string());
    }

    /// End the current edit.
    pub fn end_edit(&mut self) {
        self.editing = false;
    }

    /// Check if currently editing.
    pub fn is_editing(&self) -> bool {
        self.editing
    }

    /// Get the active component being edited.
    pub fn active_component(&self) -> Option<u32> {
        self.active_component
    }

    /// Get the active field path being edited.
    pub fn active_field(&self) -> Option<&str> {
        self.active_field.as_deref()
    }
}

// ---------------------------------------------------------------------------
// Type Decoder
// ---------------------------------------------------------------------------

/// Decodes raw bytes into typed values based on type codes.
pub struct TypeDecoder;

impl TypeDecoder {
    /// Decode a boolean from bytes.
    pub fn decode_bool(bytes: &[u8]) -> Option<bool> {
        bytes.first().map(|&b| b != 0)
    }

    /// Decode an i32 from bytes (little-endian).
    pub fn decode_i32(bytes: &[u8]) -> Option<i32> {
        if bytes.len() >= 4 {
            Some(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        } else {
            None
        }
    }

    /// Decode a u32 from bytes (little-endian).
    pub fn decode_u32(bytes: &[u8]) -> Option<u32> {
        if bytes.len() >= 4 {
            Some(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        } else {
            None
        }
    }

    /// Decode an f32 from bytes (little-endian).
    pub fn decode_f32(bytes: &[u8]) -> Option<f32> {
        if bytes.len() >= 4 {
            Some(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        } else {
            None
        }
    }

    /// Decode an f64 from bytes (little-endian).
    pub fn decode_f64(bytes: &[u8]) -> Option<f64> {
        if bytes.len() >= 8 {
            Some(f64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ]))
        } else {
            None
        }
    }

    /// Decode a Vec2 from bytes (two f32s).
    pub fn decode_vec2(bytes: &[u8]) -> Option<[f32; 2]> {
        if bytes.len() >= 8 {
            let x = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            let y = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
            Some([x, y])
        } else {
            None
        }
    }

    /// Decode a Vec3 from bytes (three f32s).
    pub fn decode_vec3(bytes: &[u8]) -> Option<[f32; 3]> {
        if bytes.len() >= 12 {
            let x = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            let y = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
            let z = f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
            Some([x, y, z])
        } else {
            None
        }
    }

    /// Decode a Vec4/Quat/Rgba from bytes (four f32s).
    pub fn decode_vec4(bytes: &[u8]) -> Option<[f32; 4]> {
        if bytes.len() >= 16 {
            let x = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            let y = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
            let z = f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
            let w = f32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
            Some([x, y, z, w])
        } else {
            None
        }
    }

    /// Decode based on type code string.
    pub fn decode(type_code: &str, bytes: &[u8]) -> Value {
        match type_code.to_lowercase().as_str() {
            "bool" | "b" => TypeDecoder::decode_bool(bytes)
                .map(Value::Bool)
                .unwrap_or_else(|| Value::Raw(bytes.to_vec())),
            "i32" | "int" | "i" => TypeDecoder::decode_i32(bytes)
                .map(Value::I32)
                .unwrap_or_else(|| Value::Raw(bytes.to_vec())),
            "u32" | "uint" | "u" => TypeDecoder::decode_u32(bytes)
                .map(Value::U32)
                .unwrap_or_else(|| Value::Raw(bytes.to_vec())),
            "f32" | "float" | "f" => TypeDecoder::decode_f32(bytes)
                .map(Value::F32)
                .unwrap_or_else(|| Value::Raw(bytes.to_vec())),
            "f64" | "double" | "d" => TypeDecoder::decode_f64(bytes)
                .map(Value::F64)
                .unwrap_or_else(|| Value::Raw(bytes.to_vec())),
            "vec2" | "vector2" => TypeDecoder::decode_vec2(bytes)
                .map(Value::Vec2)
                .unwrap_or_else(|| Value::Raw(bytes.to_vec())),
            "vec3" | "vector3" => TypeDecoder::decode_vec3(bytes)
                .map(Value::Vec3)
                .unwrap_or_else(|| Value::Raw(bytes.to_vec())),
            "vec4" | "vector4" => TypeDecoder::decode_vec4(bytes)
                .map(Value::Vec4)
                .unwrap_or_else(|| Value::Raw(bytes.to_vec())),
            "quat" | "quaternion" => TypeDecoder::decode_vec4(bytes)
                .map(Value::Quat)
                .unwrap_or_else(|| Value::Raw(bytes.to_vec())),
            "rgba" | "color" | "color4" => TypeDecoder::decode_vec4(bytes)
                .map(Value::Rgba)
                .unwrap_or_else(|| Value::Raw(bytes.to_vec())),
            _ => Value::Raw(bytes.to_vec()),
        }
    }

    /// Encode a value back to bytes.
    pub fn encode(value: &Value) -> Vec<u8> {
        match value {
            Value::Bool(b) => vec![if *b { 1 } else { 0 }],
            Value::I32(v) => v.to_le_bytes().to_vec(),
            Value::U32(v) => v.to_le_bytes().to_vec(),
            Value::F32(v) => v.to_le_bytes().to_vec(),
            Value::F64(v) => v.to_le_bytes().to_vec(),
            Value::String(s) => s.as_bytes().to_vec(),
            Value::Vec2(v) => {
                let mut bytes = Vec::with_capacity(8);
                bytes.extend_from_slice(&v[0].to_le_bytes());
                bytes.extend_from_slice(&v[1].to_le_bytes());
                bytes
            }
            Value::Vec3(v) => {
                let mut bytes = Vec::with_capacity(12);
                bytes.extend_from_slice(&v[0].to_le_bytes());
                bytes.extend_from_slice(&v[1].to_le_bytes());
                bytes.extend_from_slice(&v[2].to_le_bytes());
                bytes
            }
            Value::Vec4(v) | Value::Quat(v) | Value::Rgba(v) => {
                let mut bytes = Vec::with_capacity(16);
                bytes.extend_from_slice(&v[0].to_le_bytes());
                bytes.extend_from_slice(&v[1].to_le_bytes());
                bytes.extend_from_slice(&v[2].to_le_bytes());
                bytes.extend_from_slice(&v[3].to_le_bytes());
                bytes
            }
            Value::Raw(bytes) => bytes.clone(),
            Value::Struct(fields) => {
                let mut bytes = Vec::new();
                for (_, value) in fields {
                    bytes.extend(TypeDecoder::encode(value));
                }
                bytes
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Inspector Panel
// ---------------------------------------------------------------------------

/// Component inspector panel for the editor.
///
/// Displays component data for the selected entity and tracks edits.
pub struct InspectorPanel {
    /// Currently selected entity ID.
    selection: Option<u64>,
    /// Set of expanded component IDs (collapsible sections).
    expanded: HashSet<u32>,
    /// Pending edit state.
    edit_state: EditState,
    /// Whether to show raw bytes for unknown types.
    show_raw: bool,
    /// Whether the panel is locked (won't follow selection changes).
    locked: bool,
}

impl InspectorPanel {
    /// Create a new inspector panel.
    pub fn new() -> Self {
        Self {
            selection: None,
            expanded: HashSet::new(),
            edit_state: EditState::new(),
            show_raw: false,
            locked: false,
        }
    }

    /// Set the selected entity ID.
    pub fn set_selection(&mut self, entity_id: Option<u64>) {
        if !self.locked {
            if self.selection != entity_id {
                // Clear edits when selection changes
                self.edit_state.clear();
            }
            self.selection = entity_id;
        }
    }

    /// Get the current selection.
    pub fn selection(&self) -> Option<u64> {
        self.selection
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

    /// Toggle expansion state for a component.
    pub fn toggle_expanded(&mut self, component_id: u32) {
        if self.expanded.contains(&component_id) {
            self.expanded.remove(&component_id);
        } else {
            self.expanded.insert(component_id);
        }
    }

    /// Expand all components.
    pub fn expand_all(&mut self, component_ids: &[u32]) {
        for &id in component_ids {
            self.expanded.insert(id);
        }
    }

    /// Collapse all components.
    pub fn collapse_all(&mut self) {
        self.expanded.clear();
    }

    /// Lock the panel to the current selection.
    pub fn lock(&mut self) {
        self.locked = true;
    }

    /// Unlock the panel to follow selection changes.
    pub fn unlock(&mut self) {
        self.locked = false;
    }

    /// Check if the panel is locked.
    pub fn is_locked(&self) -> bool {
        self.locked
    }

    /// Toggle show raw bytes mode.
    pub fn toggle_show_raw(&mut self) {
        self.show_raw = !self.show_raw;
    }

    /// Set show raw bytes mode.
    pub fn set_show_raw(&mut self, show: bool) {
        self.show_raw = show;
    }

    /// Check if showing raw bytes.
    pub fn is_showing_raw(&self) -> bool {
        self.show_raw
    }

    /// Get pending edits.
    pub fn pending_edits(&self) -> &[ComponentEdit] {
        self.edit_state.pending_edits()
    }

    /// Clear all pending edits.
    pub fn clear_edits(&mut self) {
        self.edit_state.clear();
    }

    /// Check if there are pending edits.
    pub fn has_pending_edits(&self) -> bool {
        self.edit_state.has_pending()
    }

    /// Get the number of pending edits.
    pub fn pending_edit_count(&self) -> usize {
        self.edit_state.pending_count()
    }

    /// Render the inspector panel.
    ///
    /// # Arguments
    ///
    /// * `ctx` - The UI context to render to.
    /// * `components` - List of (component_id, component_name, raw_bytes) tuples.
    pub fn render<T: UIContext>(&mut self, ctx: &mut T, components: &[(u32, String, Vec<u8>)]) {
        // Header
        if let Some(entity_id) = self.selection {
            ctx.horizontal(|h| {
                h.label(&format!("Entity: {}", entity_id));
                if self.locked {
                    if h.button("Unlock") {
                        self.locked = false;
                    }
                } else if h.button("Lock") {
                    self.locked = true;
                }
            });
        } else {
            ctx.label("No entity selected");
            return;
        }

        ctx.separator();

        // Toolbar
        ctx.horizontal(|h| {
            if h.button("Expand All") {
                let ids: Vec<u32> = components.iter().map(|(id, _, _)| *id).collect();
                self.expand_all(&ids);
            }
            if h.button("Collapse All") {
                self.collapse_all();
            }
            let mut show_raw = self.show_raw;
            if h.checkbox("Show Raw", &mut show_raw) {
                self.show_raw = show_raw;
            }
        });

        ctx.spacing();

        // Components
        if components.is_empty() {
            ctx.label("No components");
            return;
        }

        for (component_id, name, bytes) in components {
            self.render_component(ctx, *component_id, name, bytes);
        }

        // Pending edits footer
        if self.has_pending_edits() {
            ctx.separator();
            ctx.horizontal(|h| {
                h.label(&format!("{} pending edit(s)", self.pending_edit_count()));
                if h.button("Apply") {
                    // Application is handled by the caller
                }
                if h.button("Discard") {
                    self.clear_edits();
                }
            });
        }
    }

    /// Render a single component section.
    fn render_component<T: UIContext>(
        &mut self,
        ctx: &mut T,
        component_id: u32,
        name: &str,
        bytes: &[u8],
    ) {
        let is_expanded = self.is_expanded(component_id);

        ctx.collapsing(&format!("{} (id: {})", name, component_id), |inner| {
            // Mark as expanded when collapsing is used
            if !self.expanded.contains(&component_id) {
                self.expanded.insert(component_id);
            }

            if self.show_raw {
                self.render_raw_bytes(inner, bytes);
            } else {
                self.render_component_fields(inner, component_id, name, bytes);
            }
        });

        // Update expanded state based on whether the collapsing header was used
        if is_expanded != self.is_expanded(component_id) {
            // State changed via UI interaction
        }
    }

    /// Render raw bytes view.
    fn render_raw_bytes<T: UIContext>(&self, ctx: &mut T, bytes: &[u8]) {
        if bytes.is_empty() {
            ctx.label("(empty)");
            return;
        }

        // Format bytes in hex, 16 per line
        let mut offset = 0;
        while offset < bytes.len() {
            let end = (offset + 16).min(bytes.len());
            let hex: Vec<String> = bytes[offset..end]
                .iter()
                .map(|b| format!("{:02X}", b))
                .collect();
            ctx.label(&format!("{:04X}: {}", offset, hex.join(" ")));
            offset = end;
        }
    }

    /// Render component fields based on type detection.
    fn render_component_fields<T: UIContext>(
        &mut self,
        ctx: &mut T,
        component_id: u32,
        name: &str,
        bytes: &[u8],
    ) {
        let entity_id = match self.selection {
            Some(id) => id,
            None => return,
        };

        // Attempt to auto-detect field types based on component name and size
        let value = self.detect_value(name, bytes);

        match value {
            Value::Struct(fields) => {
                for (field_name, field_value) in fields {
                    self.render_field(ctx, entity_id, component_id, &field_name, field_value);
                }
            }
            other => {
                self.render_field(ctx, entity_id, component_id, name, other);
            }
        }
    }

    /// Attempt to detect the value type from component name and size.
    fn detect_value(&self, name: &str, bytes: &[u8]) -> Value {
        let name_lower = name.to_lowercase();

        // Handle common component patterns
        if name_lower.contains("transform") {
            return self.detect_transform(bytes);
        }
        if name_lower.contains("position") || name_lower.contains("pos") {
            return self.detect_position(bytes);
        }
        if name_lower.contains("rotation") || name_lower.contains("rot") {
            return self.detect_rotation(bytes);
        }
        if name_lower.contains("scale") {
            return self.detect_scale(bytes);
        }
        if name_lower.contains("color") || name_lower.contains("colour") {
            return self.detect_color(bytes);
        }
        if name_lower.contains("velocity") || name_lower.contains("vel") {
            return self.detect_velocity(bytes);
        }

        // Fall back to size-based detection
        self.detect_by_size(bytes)
    }

    /// Detect transform component (position + rotation + scale).
    fn detect_transform(&self, bytes: &[u8]) -> Value {
        if bytes.len() >= 40 {
            // 12 (pos) + 16 (quat) + 12 (scale) = 40 bytes
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

    /// Detect position component (Vec3).
    fn detect_position(&self, bytes: &[u8]) -> Value {
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

    /// Detect rotation component (Quat or Euler).
    fn detect_rotation(&self, bytes: &[u8]) -> Value {
        if bytes.len() >= 16 {
            if let Some(v) = TypeDecoder::decode_vec4(bytes) {
                return Value::Quat(v);
            }
        }
        if bytes.len() >= 12 {
            if let Some(v) = TypeDecoder::decode_vec3(bytes) {
                // Could be Euler angles
                return Value::Vec3(v);
            }
        }
        Value::Raw(bytes.to_vec())
    }

    /// Detect scale component (Vec3 or uniform f32).
    fn detect_scale(&self, bytes: &[u8]) -> Value {
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

    /// Detect color component (RGBA).
    fn detect_color(&self, bytes: &[u8]) -> Value {
        if bytes.len() >= 16 {
            if let Some(v) = TypeDecoder::decode_vec4(bytes) {
                return Value::Rgba(v);
            }
        }
        if bytes.len() >= 12 {
            if let Some(v) = TypeDecoder::decode_vec3(bytes) {
                // RGB, add alpha = 1.0
                return Value::Rgba([v[0], v[1], v[2], 1.0]);
            }
        }
        Value::Raw(bytes.to_vec())
    }

    /// Detect velocity component (Vec3 or Vec2).
    fn detect_velocity(&self, bytes: &[u8]) -> Value {
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

    /// Detect value type by size alone.
    fn detect_by_size(&self, bytes: &[u8]) -> Value {
        match bytes.len() {
            1 => TypeDecoder::decode_bool(bytes)
                .map(Value::Bool)
                .unwrap_or_else(|| Value::Raw(bytes.to_vec())),
            4 => {
                // Could be i32, u32, or f32 - default to f32 for editor
                TypeDecoder::decode_f32(bytes)
                    .map(Value::F32)
                    .unwrap_or_else(|| Value::Raw(bytes.to_vec()))
            }
            8 => {
                // Could be Vec2 or f64
                TypeDecoder::decode_vec2(bytes)
                    .map(Value::Vec2)
                    .unwrap_or_else(|| Value::Raw(bytes.to_vec()))
            }
            12 => TypeDecoder::decode_vec3(bytes)
                .map(Value::Vec3)
                .unwrap_or_else(|| Value::Raw(bytes.to_vec())),
            16 => TypeDecoder::decode_vec4(bytes)
                .map(Value::Vec4)
                .unwrap_or_else(|| Value::Raw(bytes.to_vec())),
            _ => Value::Raw(bytes.to_vec()),
        }
    }

    /// Render a single field with type-aware UI.
    fn render_field<T: UIContext>(
        &mut self,
        ctx: &mut T,
        entity_id: u64,
        component_id: u32,
        field_path: &str,
        value: Value,
    ) {
        match value {
            Value::Bool(mut v) => {
                let old = v;
                if ctx.checkbox(field_path, &mut v) {
                    self.edit_state.push_edit(ComponentEdit::new(
                        entity_id,
                        component_id,
                        field_path,
                        Value::Bool(old),
                        Value::Bool(v),
                    ));
                }
            }
            Value::I32(v) => {
                let mut int_val = v;
                let old = int_val;
                if ctx.slider_int(field_path, &mut int_val, i32::MIN / 2, i32::MAX / 2) {
                    self.edit_state.push_edit(ComponentEdit::new(
                        entity_id,
                        component_id,
                        field_path,
                        Value::I32(old),
                        Value::I32(int_val),
                    ));
                }
            }
            Value::U32(v) => {
                // Render as i32 slider (UI limitation)
                let mut int_val = v as i32;
                let old = int_val;
                if ctx.slider_int(field_path, &mut int_val, 0, i32::MAX) {
                    self.edit_state.push_edit(ComponentEdit::new(
                        entity_id,
                        component_id,
                        field_path,
                        Value::U32(old as u32),
                        Value::U32(int_val as u32),
                    ));
                }
            }
            Value::F32(v) => {
                let mut float_val = v;
                let old = float_val;
                if ctx.slider(field_path, &mut float_val, -1000.0, 1000.0) {
                    self.edit_state.push_edit(ComponentEdit::new(
                        entity_id,
                        component_id,
                        field_path,
                        Value::F32(old),
                        Value::F32(float_val),
                    ));
                }
            }
            Value::F64(v) => {
                let mut float_val = v as f32;
                let old = float_val;
                if ctx.slider(field_path, &mut float_val, -1000.0, 1000.0) {
                    self.edit_state.push_edit(ComponentEdit::new(
                        entity_id,
                        component_id,
                        field_path,
                        Value::F64(old as f64),
                        Value::F64(float_val as f64),
                    ));
                }
            }
            Value::String(mut s) => {
                let old = s.clone();
                if ctx.text_edit(field_path, &mut s) {
                    self.edit_state.push_edit(ComponentEdit::new(
                        entity_id,
                        component_id,
                        field_path,
                        Value::String(old),
                        Value::String(s),
                    ));
                }
            }
            Value::Vec2(v) => {
                self.render_vec2(ctx, entity_id, component_id, field_path, v);
            }
            Value::Vec3(v) => {
                self.render_vec3(ctx, entity_id, component_id, field_path, v);
            }
            Value::Vec4(v) => {
                self.render_vec4(ctx, entity_id, component_id, field_path, v);
            }
            Value::Quat(v) => {
                self.render_quat(ctx, entity_id, component_id, field_path, v);
            }
            Value::Rgba(v) => {
                self.render_color(ctx, entity_id, component_id, field_path, v);
            }
            Value::Raw(bytes) => {
                ctx.label(&format!("{}: {} bytes", field_path, bytes.len()));
            }
            Value::Struct(fields) => {
                ctx.collapsing(field_path, |inner| {
                    for (name, field_value) in fields {
                        let nested_path = format!("{}.{}", field_path, name);
                        self.render_field(inner, entity_id, component_id, &nested_path, field_value);
                    }
                });
            }
        }
    }

    /// Render a Vec2 field.
    fn render_vec2<T: UIContext>(
        &mut self,
        ctx: &mut T,
        entity_id: u64,
        component_id: u32,
        field_path: &str,
        v: [f32; 2],
    ) {
        ctx.horizontal(|h| {
            h.label(field_path);
            let mut x = v[0];
            let mut y = v[1];
            let old = [x, y];

            let mut changed = false;
            if h.slider("x", &mut x, -1000.0, 1000.0) {
                changed = true;
            }
            if h.slider("y", &mut y, -1000.0, 1000.0) {
                changed = true;
            }

            if changed {
                self.edit_state.push_edit(ComponentEdit::new(
                    entity_id,
                    component_id,
                    field_path,
                    Value::Vec2(old),
                    Value::Vec2([x, y]),
                ));
            }
        });
    }

    /// Render a Vec3 field.
    fn render_vec3<T: UIContext>(
        &mut self,
        ctx: &mut T,
        entity_id: u64,
        component_id: u32,
        field_path: &str,
        v: [f32; 3],
    ) {
        ctx.horizontal(|h| {
            h.label(field_path);
            let mut x = v[0];
            let mut y = v[1];
            let mut z = v[2];
            let old = [x, y, z];

            let mut changed = false;
            if h.slider("x", &mut x, -1000.0, 1000.0) {
                changed = true;
            }
            if h.slider("y", &mut y, -1000.0, 1000.0) {
                changed = true;
            }
            if h.slider("z", &mut z, -1000.0, 1000.0) {
                changed = true;
            }

            if changed {
                self.edit_state.push_edit(ComponentEdit::new(
                    entity_id,
                    component_id,
                    field_path,
                    Value::Vec3(old),
                    Value::Vec3([x, y, z]),
                ));
            }
        });
    }

    /// Render a Vec4 field.
    fn render_vec4<T: UIContext>(
        &mut self,
        ctx: &mut T,
        entity_id: u64,
        component_id: u32,
        field_path: &str,
        v: [f32; 4],
    ) {
        ctx.horizontal(|h| {
            h.label(field_path);
            let mut x = v[0];
            let mut y = v[1];
            let mut z = v[2];
            let mut w = v[3];
            let old = [x, y, z, w];

            let mut changed = false;
            if h.slider("x", &mut x, -1000.0, 1000.0) {
                changed = true;
            }
            if h.slider("y", &mut y, -1000.0, 1000.0) {
                changed = true;
            }
            if h.slider("z", &mut z, -1000.0, 1000.0) {
                changed = true;
            }
            if h.slider("w", &mut w, -1000.0, 1000.0) {
                changed = true;
            }

            if changed {
                self.edit_state.push_edit(ComponentEdit::new(
                    entity_id,
                    component_id,
                    field_path,
                    Value::Vec4(old),
                    Value::Vec4([x, y, z, w]),
                ));
            }
        });
    }

    /// Render a quaternion field.
    fn render_quat<T: UIContext>(
        &mut self,
        ctx: &mut T,
        entity_id: u64,
        component_id: u32,
        field_path: &str,
        v: [f32; 4],
    ) {
        ctx.collapsing(&format!("{} (Quat)", field_path), |inner| {
            let mut x = v[0];
            let mut y = v[1];
            let mut z = v[2];
            let mut w = v[3];
            let old = [x, y, z, w];

            let mut changed = false;
            if inner.slider("x", &mut x, -1.0, 1.0) {
                changed = true;
            }
            if inner.slider("y", &mut y, -1.0, 1.0) {
                changed = true;
            }
            if inner.slider("z", &mut z, -1.0, 1.0) {
                changed = true;
            }
            if inner.slider("w", &mut w, -1.0, 1.0) {
                changed = true;
            }

            if changed {
                self.edit_state.push_edit(ComponentEdit::new(
                    entity_id,
                    component_id,
                    field_path,
                    Value::Quat(old),
                    Value::Quat([x, y, z, w]),
                ));
            }
        });
    }

    /// Render a color field with color picker.
    fn render_color<T: UIContext>(
        &mut self,
        ctx: &mut T,
        entity_id: u64,
        component_id: u32,
        field_path: &str,
        v: [f32; 4],
    ) {
        let mut color = v;
        let old = color;

        if ctx.color_edit(field_path, &mut color) {
            self.edit_state.push_edit(ComponentEdit::new(
                entity_id,
                component_id,
                field_path,
                Value::Rgba(old),
                Value::Rgba(color),
            ));
        }
    }
}

impl Default for InspectorPanel {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egui_adapter::{MockOperation, MockUIContext};

    // -------------------------------------------------------------------------
    // Value Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_value_type_name() {
        assert_eq!(Value::Bool(true).type_name(), "bool");
        assert_eq!(Value::I32(42).type_name(), "i32");
        assert_eq!(Value::U32(42).type_name(), "u32");
        assert_eq!(Value::F32(1.0).type_name(), "f32");
        assert_eq!(Value::F64(1.0).type_name(), "f64");
        assert_eq!(Value::String("test".into()).type_name(), "String");
        assert_eq!(Value::Vec2([0.0; 2]).type_name(), "Vec2");
        assert_eq!(Value::Vec3([0.0; 3]).type_name(), "Vec3");
        assert_eq!(Value::Vec4([0.0; 4]).type_name(), "Vec4");
        assert_eq!(Value::Quat([0.0; 4]).type_name(), "Quat");
        assert_eq!(Value::Rgba([0.0; 4]).type_name(), "Rgba");
        assert_eq!(Value::Raw(vec![]).type_name(), "Raw");
        assert_eq!(Value::Struct(vec![]).type_name(), "Struct");
    }

    #[test]
    fn test_value_approx_eq_bool() {
        assert!(Value::Bool(true).approx_eq(&Value::Bool(true), 0.001));
        assert!(!Value::Bool(true).approx_eq(&Value::Bool(false), 0.001));
    }

    #[test]
    fn test_value_approx_eq_int() {
        assert!(Value::I32(42).approx_eq(&Value::I32(42), 0.001));
        assert!(!Value::I32(42).approx_eq(&Value::I32(43), 0.001));
        assert!(Value::U32(42).approx_eq(&Value::U32(42), 0.001));
    }

    #[test]
    fn test_value_approx_eq_float() {
        assert!(Value::F32(1.0).approx_eq(&Value::F32(1.0001), 0.01));
        assert!(!Value::F32(1.0).approx_eq(&Value::F32(2.0), 0.01));
        assert!(Value::F64(1.0).approx_eq(&Value::F64(1.0001), 0.01));
    }

    #[test]
    fn test_value_approx_eq_vec() {
        assert!(Value::Vec2([1.0, 2.0]).approx_eq(&Value::Vec2([1.0, 2.0]), 0.001));
        assert!(Value::Vec3([1.0, 2.0, 3.0]).approx_eq(&Value::Vec3([1.0, 2.0, 3.0]), 0.001));
        assert!(Value::Vec4([1.0, 2.0, 3.0, 4.0]).approx_eq(&Value::Vec4([1.0, 2.0, 3.0, 4.0]), 0.001));
    }

    #[test]
    fn test_value_approx_eq_different_types() {
        assert!(!Value::I32(1).approx_eq(&Value::F32(1.0), 0.001));
        assert!(!Value::Vec2([1.0, 2.0]).approx_eq(&Value::Vec3([1.0, 2.0, 0.0]), 0.001));
    }

    #[test]
    fn test_value_default() {
        let v: Value = Default::default();
        assert!(matches!(v, Value::Raw(bytes) if bytes.is_empty()));
    }

    // -------------------------------------------------------------------------
    // ComponentEdit Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_component_edit_new() {
        let edit = ComponentEdit::new(
            100,
            1,
            "position.x",
            Value::F32(0.0),
            Value::F32(1.0),
        );
        assert_eq!(edit.entity_id, 100);
        assert_eq!(edit.component_id, 1);
        assert_eq!(edit.field_path, "position.x");
    }

    #[test]
    fn test_component_edit_is_noop() {
        let noop = ComponentEdit::new(
            100,
            1,
            "x",
            Value::F32(1.0),
            Value::F32(1.0),
        );
        assert!(noop.is_noop());

        let real = ComponentEdit::new(
            100,
            1,
            "x",
            Value::F32(1.0),
            Value::F32(2.0),
        );
        assert!(!real.is_noop());
    }

    // -------------------------------------------------------------------------
    // EditState Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_edit_state_new() {
        let state = EditState::new();
        assert!(!state.has_pending());
        assert_eq!(state.pending_count(), 0);
        assert!(!state.is_editing());
    }

    #[test]
    fn test_edit_state_push_edit() {
        let mut state = EditState::new();
        state.push_edit(ComponentEdit::new(
            1,
            1,
            "x",
            Value::F32(0.0),
            Value::F32(1.0),
        ));
        assert!(state.has_pending());
        assert_eq!(state.pending_count(), 1);
    }

    #[test]
    fn test_edit_state_push_noop_ignored() {
        let mut state = EditState::new();
        state.push_edit(ComponentEdit::new(
            1,
            1,
            "x",
            Value::F32(1.0),
            Value::F32(1.0),
        ));
        assert!(!state.has_pending());
    }

    #[test]
    fn test_edit_state_clear() {
        let mut state = EditState::new();
        state.push_edit(ComponentEdit::new(
            1,
            1,
            "x",
            Value::F32(0.0),
            Value::F32(1.0),
        ));
        state.begin_edit(1, "x");
        state.clear();
        assert!(!state.has_pending());
        assert!(!state.is_editing());
    }

    #[test]
    fn test_edit_state_begin_end_edit() {
        let mut state = EditState::new();
        state.begin_edit(5, "test.field");
        assert!(state.is_editing());
        assert_eq!(state.active_component(), Some(5));
        assert_eq!(state.active_field(), Some("test.field"));

        state.end_edit();
        assert!(!state.is_editing());
    }

    // -------------------------------------------------------------------------
    // TypeDecoder Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_decode_bool() {
        assert_eq!(TypeDecoder::decode_bool(&[0]), Some(false));
        assert_eq!(TypeDecoder::decode_bool(&[1]), Some(true));
        assert_eq!(TypeDecoder::decode_bool(&[255]), Some(true));
        assert_eq!(TypeDecoder::decode_bool(&[]), None);
    }

    #[test]
    fn test_decode_i32() {
        let bytes = 42i32.to_le_bytes();
        assert_eq!(TypeDecoder::decode_i32(&bytes), Some(42));
        assert_eq!(TypeDecoder::decode_i32(&[1, 2, 3]), None);
    }

    #[test]
    fn test_decode_u32() {
        let bytes = 42u32.to_le_bytes();
        assert_eq!(TypeDecoder::decode_u32(&bytes), Some(42));
    }

    #[test]
    fn test_decode_f32() {
        let bytes = 3.14f32.to_le_bytes();
        let decoded = TypeDecoder::decode_f32(&bytes).unwrap();
        assert!((decoded - 3.14).abs() < 0.001);
    }

    #[test]
    fn test_decode_f64() {
        let bytes = 3.14159265359f64.to_le_bytes();
        let decoded = TypeDecoder::decode_f64(&bytes).unwrap();
        assert!((decoded - 3.14159265359).abs() < 0.0001);
    }

    #[test]
    fn test_decode_vec2() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        bytes.extend_from_slice(&2.0f32.to_le_bytes());
        let decoded = TypeDecoder::decode_vec2(&bytes).unwrap();
        assert!((decoded[0] - 1.0).abs() < 0.001);
        assert!((decoded[1] - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_decode_vec3() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        bytes.extend_from_slice(&2.0f32.to_le_bytes());
        bytes.extend_from_slice(&3.0f32.to_le_bytes());
        let decoded = TypeDecoder::decode_vec3(&bytes).unwrap();
        assert!((decoded[0] - 1.0).abs() < 0.001);
        assert!((decoded[1] - 2.0).abs() < 0.001);
        assert!((decoded[2] - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_decode_vec4() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        bytes.extend_from_slice(&2.0f32.to_le_bytes());
        bytes.extend_from_slice(&3.0f32.to_le_bytes());
        bytes.extend_from_slice(&4.0f32.to_le_bytes());
        let decoded = TypeDecoder::decode_vec4(&bytes).unwrap();
        assert!((decoded[0] - 1.0).abs() < 0.001);
        assert!((decoded[1] - 2.0).abs() < 0.001);
        assert!((decoded[2] - 3.0).abs() < 0.001);
        assert!((decoded[3] - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_decode_by_type_code() {
        assert!(matches!(TypeDecoder::decode("bool", &[1]), Value::Bool(true)));
        assert!(matches!(TypeDecoder::decode("f32", &3.14f32.to_le_bytes()), Value::F32(_)));
        assert!(matches!(TypeDecoder::decode("vec3", &[0; 12]), Value::Vec3(_)));
        assert!(matches!(TypeDecoder::decode("quat", &[0; 16]), Value::Quat(_)));
        assert!(matches!(TypeDecoder::decode("rgba", &[0; 16]), Value::Rgba(_)));
        assert!(matches!(TypeDecoder::decode("unknown", &[1, 2, 3]), Value::Raw(_)));
    }

    #[test]
    fn test_encode_bool() {
        assert_eq!(TypeDecoder::encode(&Value::Bool(true)), vec![1]);
        assert_eq!(TypeDecoder::encode(&Value::Bool(false)), vec![0]);
    }

    #[test]
    fn test_encode_i32() {
        let encoded = TypeDecoder::encode(&Value::I32(42));
        assert_eq!(encoded, 42i32.to_le_bytes().to_vec());
    }

    #[test]
    fn test_encode_f32() {
        let encoded = TypeDecoder::encode(&Value::F32(3.14));
        let decoded = f32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]);
        assert!((decoded - 3.14).abs() < 0.001);
    }

    #[test]
    fn test_encode_vec3() {
        let encoded = TypeDecoder::encode(&Value::Vec3([1.0, 2.0, 3.0]));
        assert_eq!(encoded.len(), 12);
    }

    #[test]
    fn test_encode_roundtrip() {
        let original = Value::Vec4([1.0, 2.0, 3.0, 4.0]);
        let bytes = TypeDecoder::encode(&original);
        let decoded = TypeDecoder::decode("vec4", &bytes);
        assert!(original.approx_eq(&decoded, 0.001));
    }

    // -------------------------------------------------------------------------
    // InspectorPanel Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_inspector_panel_new() {
        let panel = InspectorPanel::new();
        assert!(panel.selection().is_none());
        assert!(!panel.has_pending_edits());
        assert!(!panel.is_locked());
        assert!(!panel.is_showing_raw());
    }

    #[test]
    fn test_inspector_panel_selection() {
        let mut panel = InspectorPanel::new();
        panel.set_selection(Some(42));
        assert_eq!(panel.selection(), Some(42));

        panel.set_selection(None);
        assert_eq!(panel.selection(), None);
    }

    #[test]
    fn test_inspector_panel_selection_clears_edits() {
        let mut panel = InspectorPanel::new();
        panel.set_selection(Some(1));
        panel.edit_state.push_edit(ComponentEdit::new(
            1,
            1,
            "x",
            Value::F32(0.0),
            Value::F32(1.0),
        ));
        assert!(panel.has_pending_edits());

        panel.set_selection(Some(2));
        assert!(!panel.has_pending_edits());
    }

    #[test]
    fn test_inspector_panel_lock() {
        let mut panel = InspectorPanel::new();
        panel.set_selection(Some(42));
        panel.lock();
        assert!(panel.is_locked());

        panel.set_selection(Some(100));
        assert_eq!(panel.selection(), Some(42)); // Still 42 because locked

        panel.unlock();
        panel.set_selection(Some(100));
        assert_eq!(panel.selection(), Some(100));
    }

    #[test]
    fn test_inspector_panel_expansion() {
        let mut panel = InspectorPanel::new();
        assert!(!panel.is_expanded(1));

        panel.set_expanded(1, true);
        assert!(panel.is_expanded(1));

        panel.toggle_expanded(1);
        assert!(!panel.is_expanded(1));

        panel.expand_all(&[1, 2, 3]);
        assert!(panel.is_expanded(1));
        assert!(panel.is_expanded(2));
        assert!(panel.is_expanded(3));

        panel.collapse_all();
        assert!(!panel.is_expanded(1));
        assert!(!panel.is_expanded(2));
        assert!(!panel.is_expanded(3));
    }

    #[test]
    fn test_inspector_panel_show_raw() {
        let mut panel = InspectorPanel::new();
        assert!(!panel.is_showing_raw());

        panel.toggle_show_raw();
        assert!(panel.is_showing_raw());

        panel.set_show_raw(false);
        assert!(!panel.is_showing_raw());
    }

    #[test]
    fn test_inspector_panel_pending_edits() {
        let mut panel = InspectorPanel::new();
        panel.set_selection(Some(1));
        panel.edit_state.push_edit(ComponentEdit::new(
            1,
            1,
            "x",
            Value::F32(0.0),
            Value::F32(1.0),
        ));
        assert_eq!(panel.pending_edit_count(), 1);
        assert!(panel.has_pending_edits());

        let edits = panel.pending_edits();
        assert_eq!(edits.len(), 1);
        assert_eq!(edits[0].field_path, "x");

        panel.clear_edits();
        assert!(!panel.has_pending_edits());
    }

    #[test]
    fn test_inspector_panel_default() {
        let panel: InspectorPanel = Default::default();
        assert!(panel.selection().is_none());
    }

    // -------------------------------------------------------------------------
    // Render Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_render_no_selection() {
        let mut panel = InspectorPanel::new();
        let mut ctx = MockUIContext::new(1);
        let components: Vec<(u32, String, Vec<u8>)> = vec![];

        panel.render(&mut ctx, &components);

        assert!(ctx.has_operation(&MockOperation::Label("No entity selected".to_string())));
    }

    #[test]
    fn test_render_with_selection_no_components() {
        let mut panel = InspectorPanel::new();
        panel.set_selection(Some(42));
        let mut ctx = MockUIContext::new(1);
        let components: Vec<(u32, String, Vec<u8>)> = vec![];

        panel.render(&mut ctx, &components);

        assert!(ctx.has_operation(&MockOperation::Label("No components".to_string())));
    }

    #[test]
    fn test_render_with_components() {
        let mut panel = InspectorPanel::new();
        panel.set_selection(Some(42));
        let mut ctx = MockUIContext::new(1);

        // Position component with Vec3 data
        let mut pos_bytes = Vec::new();
        pos_bytes.extend_from_slice(&1.0f32.to_le_bytes());
        pos_bytes.extend_from_slice(&2.0f32.to_le_bytes());
        pos_bytes.extend_from_slice(&3.0f32.to_le_bytes());

        let components = vec![(1, "Position".to_string(), pos_bytes)];

        panel.render(&mut ctx, &components);

        // Should have rendered the entity header
        let ops = ctx.operations();
        assert!(ops.iter().any(|op| matches!(op, MockOperation::HorizontalBegin)));
    }

    #[test]
    fn test_render_toolbar_buttons() {
        let mut panel = InspectorPanel::new();
        panel.set_selection(Some(42));
        let mut ctx = MockUIContext::new(1);
        let components = vec![(1, "Test".to_string(), vec![])];

        panel.render(&mut ctx, &components);

        // Check toolbar buttons exist
        assert!(ctx.has_operation(&MockOperation::Button("Expand All".to_string())));
        assert!(ctx.has_operation(&MockOperation::Button("Collapse All".to_string())));
    }

    // -------------------------------------------------------------------------
    // Type Detection Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_detect_position() {
        let panel = InspectorPanel::new();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        bytes.extend_from_slice(&2.0f32.to_le_bytes());
        bytes.extend_from_slice(&3.0f32.to_le_bytes());

        let value = panel.detect_value("Position", &bytes);
        assert!(matches!(value, Value::Vec3(_)));
    }

    #[test]
    fn test_detect_rotation() {
        let panel = InspectorPanel::new();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0.0f32.to_le_bytes());
        bytes.extend_from_slice(&0.0f32.to_le_bytes());
        bytes.extend_from_slice(&0.0f32.to_le_bytes());
        bytes.extend_from_slice(&1.0f32.to_le_bytes());

        let value = panel.detect_value("Rotation", &bytes);
        assert!(matches!(value, Value::Quat(_)));
    }

    #[test]
    fn test_detect_color() {
        let panel = InspectorPanel::new();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        bytes.extend_from_slice(&0.0f32.to_le_bytes());
        bytes.extend_from_slice(&0.0f32.to_le_bytes());
        bytes.extend_from_slice(&1.0f32.to_le_bytes());

        let value = panel.detect_value("Color", &bytes);
        assert!(matches!(value, Value::Rgba(_)));
    }

    #[test]
    fn test_detect_transform() {
        let panel = InspectorPanel::new();
        let mut bytes = Vec::new();
        // Position (12 bytes)
        bytes.extend_from_slice(&0.0f32.to_le_bytes());
        bytes.extend_from_slice(&0.0f32.to_le_bytes());
        bytes.extend_from_slice(&0.0f32.to_le_bytes());
        // Rotation (16 bytes)
        bytes.extend_from_slice(&0.0f32.to_le_bytes());
        bytes.extend_from_slice(&0.0f32.to_le_bytes());
        bytes.extend_from_slice(&0.0f32.to_le_bytes());
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        // Scale (12 bytes)
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        bytes.extend_from_slice(&1.0f32.to_le_bytes());

        let value = panel.detect_value("Transform", &bytes);
        assert!(matches!(value, Value::Struct(_)));
    }

    #[test]
    fn test_detect_by_size_bool() {
        let panel = InspectorPanel::new();
        let value = panel.detect_by_size(&[1]);
        assert!(matches!(value, Value::Bool(true)));
    }

    #[test]
    fn test_detect_by_size_f32() {
        let panel = InspectorPanel::new();
        let value = panel.detect_by_size(&3.14f32.to_le_bytes());
        assert!(matches!(value, Value::F32(_)));
    }

    #[test]
    fn test_detect_by_size_vec2() {
        let panel = InspectorPanel::new();
        let value = panel.detect_by_size(&[0; 8]);
        assert!(matches!(value, Value::Vec2(_)));
    }

    #[test]
    fn test_detect_by_size_vec3() {
        let panel = InspectorPanel::new();
        let value = panel.detect_by_size(&[0; 12]);
        assert!(matches!(value, Value::Vec3(_)));
    }

    #[test]
    fn test_detect_by_size_vec4() {
        let panel = InspectorPanel::new();
        let value = panel.detect_by_size(&[0; 16]);
        assert!(matches!(value, Value::Vec4(_)));
    }

    #[test]
    fn test_detect_by_size_raw() {
        let panel = InspectorPanel::new();
        let value = panel.detect_by_size(&[0; 7]);
        assert!(matches!(value, Value::Raw(_)));
    }

    // -------------------------------------------------------------------------
    // Serialization Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_value_serialization() {
        let values = vec![
            Value::Bool(true),
            Value::I32(42),
            Value::U32(42),
            Value::F32(3.14),
            Value::F64(3.14159),
            Value::String("test".to_string()),
            Value::Vec2([1.0, 2.0]),
            Value::Vec3([1.0, 2.0, 3.0]),
            Value::Vec4([1.0, 2.0, 3.0, 4.0]),
            Value::Quat([0.0, 0.0, 0.0, 1.0]),
            Value::Rgba([1.0, 0.0, 0.0, 1.0]),
            Value::Raw(vec![1, 2, 3]),
            Value::Struct(vec![("x".to_string(), Value::F32(1.0))]),
        ];

        for value in values {
            let json = serde_json::to_string(&value).unwrap();
            let parsed: Value = serde_json::from_str(&json).unwrap();
            assert!(value.approx_eq(&parsed, 0.0001), "Failed for {:?}", value);
        }
    }

    #[test]
    fn test_component_edit_serialization() {
        let edit = ComponentEdit::new(
            100,
            1,
            "position.x",
            Value::F32(0.0),
            Value::F32(1.0),
        );
        let json = serde_json::to_string(&edit).unwrap();
        let parsed: ComponentEdit = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.entity_id, 100);
        assert_eq!(parsed.component_id, 1);
        assert_eq!(parsed.field_path, "position.x");
    }
}
