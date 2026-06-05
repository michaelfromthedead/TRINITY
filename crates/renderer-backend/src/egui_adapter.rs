//! EguiUIContext Adapter for Python-Rust UI Bridge
//!
//! This module provides a protocol adapter that maps Python's UIContext protocol
//! to egui::Ui in Rust. It enables Python code to drive egui UI through the bridge.
//!
//! # Architecture
//!
//! ```text
//! Python UI Code           Bridge              Rust (egui)
//! ==============           ======              ===========
//!     │                       │                     │
//!     ├── ui.button("Ok") ───►├── UICommand ───────►│ egui::Ui::button()
//!     │                       │                     │
//!     │◄── clicked: true ────┤◄── UIResponse ──────┤
//!     │                       │                     │
//! ```
//!
//! # Remote UI Mode
//!
//! When running in remote mode (Python on host, Rust on GPU server), commands
//! are batched into a `CommandBuffer` and sent over the wire. Responses are
//! collected and returned to Python after the frame.
//!
//! # Example
//!
//! ```rust,ignore
//! use renderer_backend::egui_adapter::{UIContext, EguiUIContext};
//!
//! fn build_ui(ui: &mut egui::Ui) {
//!     let mut ctx = EguiUIContext::new(ui);
//!     ctx.label("Settings");
//!     if ctx.button("Apply") {
//!         // Button was clicked
//!     }
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// UIContext Trait
// ---------------------------------------------------------------------------

/// A UI context protocol that mirrors Python's UI abstraction.
///
/// This trait defines the interface for building immediate-mode UIs.
/// It can be implemented by different backends (egui, remote, mock).
pub trait UIContext {
    /// Display a text label.
    fn label(&mut self, text: &str);

    /// Display a clickable button. Returns true if clicked.
    fn button(&mut self, text: &str) -> bool;

    /// Display a checkbox with label. Returns true if the value changed.
    fn checkbox(&mut self, text: &str, value: &mut bool) -> bool;

    /// Display a slider for a float value. Returns true if the value changed.
    fn slider(&mut self, text: &str, value: &mut f32, min: f32, max: f32) -> bool;

    /// Display a single-line text edit. Returns true if the text changed.
    fn text_edit(&mut self, text: &str, buffer: &mut String) -> bool;

    /// Display a collapsible section with nested content.
    fn collapsing<F>(&mut self, text: &str, add_contents: F)
    where
        F: FnOnce(&mut Self);

    /// Lay out contents horizontally.
    fn horizontal<F>(&mut self, add_contents: F)
    where
        F: FnOnce(&mut Self);

    /// Lay out contents vertically.
    fn vertical<F>(&mut self, add_contents: F)
    where
        F: FnOnce(&mut Self);

    /// Add a horizontal separator line.
    fn separator(&mut self);

    /// Add spacing between elements.
    fn spacing(&mut self);

    // Extended widget set

    /// Display a slider for an integer value. Returns true if the value changed.
    fn slider_int(&mut self, text: &str, value: &mut i32, min: i32, max: i32) -> bool;

    /// Display a multi-line text area. Returns true if the text changed.
    fn text_area(&mut self, text: &str, buffer: &mut String, lines: u32) -> bool;

    /// Display a color picker. Returns true if the color changed.
    fn color_edit(&mut self, text: &str, color: &mut [f32; 4]) -> bool;

    /// Display a dropdown combo box. Returns true if the selection changed.
    fn combo(&mut self, text: &str, current: &mut usize, options: &[&str]) -> bool;

    /// Display a progress bar (0.0 to 1.0).
    fn progress(&mut self, fraction: f32, text: Option<&str>);

    /// Display a tooltip for the previous widget.
    fn tooltip(&mut self, text: &str);

    /// Begin a disabled section (widgets will be grayed out).
    fn disabled<F>(&mut self, disabled: bool, add_contents: F)
    where
        F: FnOnce(&mut Self);

    /// Get the unique ID for this context (for widget state tracking).
    fn id(&self) -> u64;
}

// ---------------------------------------------------------------------------
// UI Commands (for remote/serialized UI)
// ---------------------------------------------------------------------------

/// A UI command that can be serialized for remote rendering.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum UICommand {
    /// Display a label.
    Label { text: String },

    /// Display a button.
    Button { id: u64, text: String },

    /// Display a checkbox.
    Checkbox {
        id: u64,
        text: String,
        value: bool,
    },

    /// Display a float slider.
    Slider {
        id: u64,
        text: String,
        value: f32,
        min: f32,
        max: f32,
    },

    /// Display an integer slider.
    SliderInt {
        id: u64,
        text: String,
        value: i32,
        min: i32,
        max: i32,
    },

    /// Display a single-line text edit.
    TextEdit {
        id: u64,
        text: String,
        value: String,
    },

    /// Display a multi-line text area.
    TextArea {
        id: u64,
        text: String,
        value: String,
        lines: u32,
    },

    /// Display a color picker.
    ColorEdit {
        id: u64,
        text: String,
        color: [f32; 4],
    },

    /// Display a combo box.
    Combo {
        id: u64,
        text: String,
        current: usize,
        options: Vec<String>,
    },

    /// Display a progress bar.
    Progress {
        fraction: f32,
        text: Option<String>,
    },

    /// Add a tooltip to the previous widget.
    Tooltip { text: String },

    /// Begin a collapsing section.
    CollapsingBegin { id: u64, text: String },

    /// End a collapsing section.
    CollapsingEnd,

    /// Begin horizontal layout.
    HorizontalBegin,

    /// End horizontal layout.
    HorizontalEnd,

    /// Begin vertical layout.
    VerticalBegin,

    /// End vertical layout.
    VerticalEnd,

    /// Begin disabled section.
    DisabledBegin { disabled: bool },

    /// End disabled section.
    DisabledEnd,

    /// Add a separator.
    Separator,

    /// Add spacing.
    Spacing,
}

impl UICommand {
    /// Get the widget ID if this command has one.
    pub fn widget_id(&self) -> Option<u64> {
        match self {
            UICommand::Button { id, .. }
            | UICommand::Checkbox { id, .. }
            | UICommand::Slider { id, .. }
            | UICommand::SliderInt { id, .. }
            | UICommand::TextEdit { id, .. }
            | UICommand::TextArea { id, .. }
            | UICommand::ColorEdit { id, .. }
            | UICommand::Combo { id, .. }
            | UICommand::CollapsingBegin { id, .. } => Some(*id),
            _ => None,
        }
    }

    /// Check if this command is interactive (can produce a response).
    pub fn is_interactive(&self) -> bool {
        self.widget_id().is_some()
    }
}

// ---------------------------------------------------------------------------
// UI Response
// ---------------------------------------------------------------------------

/// A response from an interactive UI widget.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum UIResponse {
    /// Button was clicked.
    ButtonClicked { id: u64 },

    /// Checkbox value changed.
    CheckboxChanged { id: u64, value: bool },

    /// Slider value changed.
    SliderChanged { id: u64, value: f32 },

    /// Integer slider value changed.
    SliderIntChanged { id: u64, value: i32 },

    /// Text edit value changed.
    TextChanged { id: u64, value: String },

    /// Color value changed.
    ColorChanged { id: u64, color: [f32; 4] },

    /// Combo selection changed.
    ComboChanged { id: u64, index: usize },

    /// Collapsing header was toggled.
    CollapsingToggled { id: u64, open: bool },
}

impl UIResponse {
    /// Get the widget ID for this response.
    pub fn widget_id(&self) -> u64 {
        match self {
            UIResponse::ButtonClicked { id }
            | UIResponse::CheckboxChanged { id, .. }
            | UIResponse::SliderChanged { id, .. }
            | UIResponse::SliderIntChanged { id, .. }
            | UIResponse::TextChanged { id, .. }
            | UIResponse::ColorChanged { id, .. }
            | UIResponse::ComboChanged { id, .. }
            | UIResponse::CollapsingToggled { id, .. } => *id,
        }
    }
}

// ---------------------------------------------------------------------------
// Command Buffer
// ---------------------------------------------------------------------------

/// A buffer for batching UI commands for remote rendering.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CommandBuffer {
    /// Buffered commands in order.
    commands: Vec<UICommand>,
    /// Responses for interactive widgets (keyed by widget ID).
    responses: HashMap<u64, UIResponse>,
    /// Frame number this buffer is for.
    frame: u64,
}

impl CommandBuffer {
    /// Create a new empty command buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a command buffer for a specific frame.
    pub fn for_frame(frame: u64) -> Self {
        Self {
            commands: Vec::new(),
            responses: HashMap::new(),
            frame,
        }
    }

    /// Add a command to the buffer.
    pub fn push(&mut self, command: UICommand) {
        self.commands.push(command);
    }

    /// Add a response for a widget.
    pub fn add_response(&mut self, response: UIResponse) {
        let id = response.widget_id();
        self.responses.insert(id, response);
    }

    /// Get the response for a widget ID.
    pub fn get_response(&self, id: u64) -> Option<&UIResponse> {
        self.responses.get(&id)
    }

    /// Check if a button was clicked.
    pub fn button_clicked(&self, id: u64) -> bool {
        matches!(self.responses.get(&id), Some(UIResponse::ButtonClicked { .. }))
    }

    /// Get all commands.
    pub fn commands(&self) -> &[UICommand] {
        &self.commands
    }

    /// Get all responses.
    pub fn responses(&self) -> &HashMap<u64, UIResponse> {
        &self.responses
    }

    /// Get the frame number.
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// Clear the buffer for reuse.
    pub fn clear(&mut self) {
        self.commands.clear();
        self.responses.clear();
    }

    /// Get the number of commands.
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }

    /// Deserialize from JSON.
    pub fn from_json(data: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(data)
    }

    /// Merge another buffer into this one.
    pub fn merge(&mut self, other: &CommandBuffer) {
        self.commands.extend(other.commands.iter().cloned());
        for (id, response) in &other.responses {
            self.responses.insert(*id, response.clone());
        }
    }
}

// ---------------------------------------------------------------------------
// Remote UI Context (records commands for later execution)
// ---------------------------------------------------------------------------

/// A UI context that records commands for remote rendering.
///
/// This is used when Python is running on a different machine from the
/// Rust renderer. Commands are batched and sent over the network.
pub struct RemoteUIContext {
    /// Command buffer for recording.
    buffer: CommandBuffer,
    /// Next widget ID.
    next_id: AtomicU64,
    /// Context ID for state tracking.
    context_id: u64,
    /// Nesting depth for layout tracking.
    nesting_depth: u32,
}

impl RemoteUIContext {
    /// Create a new remote UI context.
    pub fn new(context_id: u64) -> Self {
        Self {
            buffer: CommandBuffer::new(),
            next_id: AtomicU64::new(1),
            context_id,
            nesting_depth: 0,
        }
    }

    /// Create a remote UI context for a specific frame.
    pub fn for_frame(context_id: u64, frame: u64) -> Self {
        Self {
            buffer: CommandBuffer::for_frame(frame),
            next_id: AtomicU64::new(1),
            context_id,
            nesting_depth: 0,
        }
    }

    /// Get the next widget ID.
    fn next_widget_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Get the command buffer.
    pub fn buffer(&self) -> &CommandBuffer {
        &self.buffer
    }

    /// Take the command buffer, replacing it with an empty one.
    pub fn take_buffer(&mut self) -> CommandBuffer {
        std::mem::take(&mut self.buffer)
    }

    /// Set responses from the remote renderer.
    pub fn set_responses(&mut self, responses: HashMap<u64, UIResponse>) {
        self.buffer.responses = responses;
    }

    /// Get the nesting depth.
    pub fn nesting_depth(&self) -> u32 {
        self.nesting_depth
    }
}

impl UIContext for RemoteUIContext {
    fn label(&mut self, text: &str) {
        self.buffer.push(UICommand::Label {
            text: text.to_string(),
        });
    }

    fn button(&mut self, text: &str) -> bool {
        let id = self.next_widget_id();
        self.buffer.push(UICommand::Button {
            id,
            text: text.to_string(),
        });
        self.buffer.button_clicked(id)
    }

    fn checkbox(&mut self, text: &str, value: &mut bool) -> bool {
        let id = self.next_widget_id();
        self.buffer.push(UICommand::Checkbox {
            id,
            text: text.to_string(),
            value: *value,
        });
        if let Some(UIResponse::CheckboxChanged { value: new_value, .. }) =
            self.buffer.get_response(id)
        {
            *value = *new_value;
            return true;
        }
        false
    }

    fn slider(&mut self, text: &str, value: &mut f32, min: f32, max: f32) -> bool {
        let id = self.next_widget_id();
        self.buffer.push(UICommand::Slider {
            id,
            text: text.to_string(),
            value: *value,
            min,
            max,
        });
        if let Some(UIResponse::SliderChanged { value: new_value, .. }) =
            self.buffer.get_response(id)
        {
            *value = *new_value;
            return true;
        }
        false
    }

    fn text_edit(&mut self, text: &str, buffer: &mut String) -> bool {
        let id = self.next_widget_id();
        self.buffer.push(UICommand::TextEdit {
            id,
            text: text.to_string(),
            value: buffer.clone(),
        });
        if let Some(UIResponse::TextChanged { value, .. }) = self.buffer.get_response(id) {
            *buffer = value.clone();
            return true;
        }
        false
    }

    fn collapsing<F>(&mut self, text: &str, add_contents: F)
    where
        F: FnOnce(&mut Self),
    {
        let id = self.next_widget_id();
        self.buffer.push(UICommand::CollapsingBegin {
            id,
            text: text.to_string(),
        });
        self.nesting_depth += 1;
        add_contents(self);
        self.nesting_depth -= 1;
        self.buffer.push(UICommand::CollapsingEnd);
    }

    fn horizontal<F>(&mut self, add_contents: F)
    where
        F: FnOnce(&mut Self),
    {
        self.buffer.push(UICommand::HorizontalBegin);
        self.nesting_depth += 1;
        add_contents(self);
        self.nesting_depth -= 1;
        self.buffer.push(UICommand::HorizontalEnd);
    }

    fn vertical<F>(&mut self, add_contents: F)
    where
        F: FnOnce(&mut Self),
    {
        self.buffer.push(UICommand::VerticalBegin);
        self.nesting_depth += 1;
        add_contents(self);
        self.nesting_depth -= 1;
        self.buffer.push(UICommand::VerticalEnd);
    }

    fn separator(&mut self) {
        self.buffer.push(UICommand::Separator);
    }

    fn spacing(&mut self) {
        self.buffer.push(UICommand::Spacing);
    }

    fn slider_int(&mut self, text: &str, value: &mut i32, min: i32, max: i32) -> bool {
        let id = self.next_widget_id();
        self.buffer.push(UICommand::SliderInt {
            id,
            text: text.to_string(),
            value: *value,
            min,
            max,
        });
        if let Some(UIResponse::SliderIntChanged { value: new_value, .. }) =
            self.buffer.get_response(id)
        {
            *value = *new_value;
            return true;
        }
        false
    }

    fn text_area(&mut self, text: &str, buffer: &mut String, lines: u32) -> bool {
        let id = self.next_widget_id();
        self.buffer.push(UICommand::TextArea {
            id,
            text: text.to_string(),
            value: buffer.clone(),
            lines,
        });
        if let Some(UIResponse::TextChanged { value, .. }) = self.buffer.get_response(id) {
            *buffer = value.clone();
            return true;
        }
        false
    }

    fn color_edit(&mut self, text: &str, color: &mut [f32; 4]) -> bool {
        let id = self.next_widget_id();
        self.buffer.push(UICommand::ColorEdit {
            id,
            text: text.to_string(),
            color: *color,
        });
        if let Some(UIResponse::ColorChanged { color: new_color, .. }) =
            self.buffer.get_response(id)
        {
            *color = *new_color;
            return true;
        }
        false
    }

    fn combo(&mut self, text: &str, current: &mut usize, options: &[&str]) -> bool {
        let id = self.next_widget_id();
        self.buffer.push(UICommand::Combo {
            id,
            text: text.to_string(),
            current: *current,
            options: options.iter().map(|s| s.to_string()).collect(),
        });
        if let Some(UIResponse::ComboChanged { index, .. }) = self.buffer.get_response(id) {
            *current = *index;
            return true;
        }
        false
    }

    fn progress(&mut self, fraction: f32, text: Option<&str>) {
        self.buffer.push(UICommand::Progress {
            fraction,
            text: text.map(|s| s.to_string()),
        });
    }

    fn tooltip(&mut self, text: &str) {
        self.buffer.push(UICommand::Tooltip {
            text: text.to_string(),
        });
    }

    fn disabled<F>(&mut self, disabled: bool, add_contents: F)
    where
        F: FnOnce(&mut Self),
    {
        self.buffer.push(UICommand::DisabledBegin { disabled });
        self.nesting_depth += 1;
        add_contents(self);
        self.nesting_depth -= 1;
        self.buffer.push(UICommand::DisabledEnd);
    }

    fn id(&self) -> u64 {
        self.context_id
    }
}

// ---------------------------------------------------------------------------
// Mock UI Context (for testing)
// ---------------------------------------------------------------------------

/// A mock UI context for testing without egui.
///
/// Records all operations and allows setting up expected responses.
pub struct MockUIContext {
    /// Recorded operations (for verification).
    operations: Vec<MockOperation>,
    /// Pre-configured responses.
    responses: HashMap<String, MockResponse>,
    /// Context ID.
    context_id: u64,
    /// Mutable state for checkboxes, sliders, etc.
    state: HashMap<String, MockState>,
}

/// A recorded UI operation.
#[derive(Debug, Clone, PartialEq)]
pub enum MockOperation {
    Label(String),
    Button(String),
    Checkbox(String, bool),
    Slider(String, f32, f32, f32),
    SliderInt(String, i32, i32, i32),
    TextEdit(String, String),
    TextArea(String, String, u32),
    ColorEdit(String, [f32; 4]),
    Combo(String, usize, Vec<String>),
    Progress(f32, Option<String>),
    Tooltip(String),
    CollapsingBegin(String),
    CollapsingEnd,
    HorizontalBegin,
    HorizontalEnd,
    VerticalBegin,
    VerticalEnd,
    DisabledBegin(bool),
    DisabledEnd,
    Separator,
    Spacing,
}

/// Pre-configured response for a mock widget.
#[derive(Debug, Clone)]
pub enum MockResponse {
    Clicked,
    Changed,
}

/// State for a mock widget.
#[derive(Debug, Clone)]
pub enum MockState {
    Bool(bool),
    F32(f32),
    I32(i32),
    String(String),
    Color([f32; 4]),
    Index(usize),
}

impl MockUIContext {
    /// Create a new mock UI context.
    pub fn new(context_id: u64) -> Self {
        Self {
            operations: Vec::new(),
            responses: HashMap::new(),
            context_id,
            state: HashMap::new(),
        }
    }

    /// Configure a button to be clicked.
    pub fn click_button(&mut self, text: &str) {
        self.responses
            .insert(format!("button:{}", text), MockResponse::Clicked);
    }

    /// Configure a widget to report a change.
    pub fn set_changed(&mut self, widget_type: &str, text: &str) {
        self.responses
            .insert(format!("{}:{}", widget_type, text), MockResponse::Changed);
    }

    /// Set the state for a checkbox.
    pub fn set_checkbox_state(&mut self, text: &str, value: bool) {
        self.state
            .insert(format!("checkbox:{}", text), MockState::Bool(value));
    }

    /// Set the state for a slider.
    pub fn set_slider_state(&mut self, text: &str, value: f32) {
        self.state
            .insert(format!("slider:{}", text), MockState::F32(value));
    }

    /// Set the state for an integer slider.
    pub fn set_slider_int_state(&mut self, text: &str, value: i32) {
        self.state
            .insert(format!("slider_int:{}", text), MockState::I32(value));
    }

    /// Set the state for a text edit.
    pub fn set_text_state(&mut self, text: &str, value: &str) {
        self.state.insert(
            format!("text:{}", text),
            MockState::String(value.to_string()),
        );
    }

    /// Set the state for a color edit.
    pub fn set_color_state(&mut self, text: &str, color: [f32; 4]) {
        self.state
            .insert(format!("color:{}", text), MockState::Color(color));
    }

    /// Set the state for a combo box.
    pub fn set_combo_state(&mut self, text: &str, index: usize) {
        self.state
            .insert(format!("combo:{}", text), MockState::Index(index));
    }

    /// Get all recorded operations.
    pub fn operations(&self) -> &[MockOperation] {
        &self.operations
    }

    /// Clear recorded operations.
    pub fn clear_operations(&mut self) {
        self.operations.clear();
    }

    /// Check if a specific operation was recorded.
    pub fn has_operation(&self, op: &MockOperation) -> bool {
        self.operations.contains(op)
    }
}

impl UIContext for MockUIContext {
    fn label(&mut self, text: &str) {
        self.operations.push(MockOperation::Label(text.to_string()));
    }

    fn button(&mut self, text: &str) -> bool {
        self.operations.push(MockOperation::Button(text.to_string()));
        self.responses
            .contains_key(&format!("button:{}", text))
    }

    fn checkbox(&mut self, text: &str, value: &mut bool) -> bool {
        self.operations
            .push(MockOperation::Checkbox(text.to_string(), *value));
        if let Some(MockState::Bool(new_value)) = self.state.get(&format!("checkbox:{}", text)) {
            if *new_value != *value {
                *value = *new_value;
                return true;
            }
        }
        false
    }

    fn slider(&mut self, text: &str, value: &mut f32, min: f32, max: f32) -> bool {
        self.operations
            .push(MockOperation::Slider(text.to_string(), *value, min, max));
        if let Some(MockState::F32(new_value)) = self.state.get(&format!("slider:{}", text)) {
            if (*new_value - *value).abs() > f32::EPSILON {
                *value = *new_value;
                return true;
            }
        }
        false
    }

    fn text_edit(&mut self, text: &str, buffer: &mut String) -> bool {
        self.operations
            .push(MockOperation::TextEdit(text.to_string(), buffer.clone()));
        if let Some(MockState::String(new_value)) = self.state.get(&format!("text:{}", text)) {
            if new_value != buffer {
                *buffer = new_value.clone();
                return true;
            }
        }
        false
    }

    fn collapsing<F>(&mut self, text: &str, add_contents: F)
    where
        F: FnOnce(&mut Self),
    {
        self.operations
            .push(MockOperation::CollapsingBegin(text.to_string()));
        add_contents(self);
        self.operations.push(MockOperation::CollapsingEnd);
    }

    fn horizontal<F>(&mut self, add_contents: F)
    where
        F: FnOnce(&mut Self),
    {
        self.operations.push(MockOperation::HorizontalBegin);
        add_contents(self);
        self.operations.push(MockOperation::HorizontalEnd);
    }

    fn vertical<F>(&mut self, add_contents: F)
    where
        F: FnOnce(&mut Self),
    {
        self.operations.push(MockOperation::VerticalBegin);
        add_contents(self);
        self.operations.push(MockOperation::VerticalEnd);
    }

    fn separator(&mut self) {
        self.operations.push(MockOperation::Separator);
    }

    fn spacing(&mut self) {
        self.operations.push(MockOperation::Spacing);
    }

    fn slider_int(&mut self, text: &str, value: &mut i32, min: i32, max: i32) -> bool {
        self.operations
            .push(MockOperation::SliderInt(text.to_string(), *value, min, max));
        if let Some(MockState::I32(new_value)) = self.state.get(&format!("slider_int:{}", text)) {
            if *new_value != *value {
                *value = *new_value;
                return true;
            }
        }
        false
    }

    fn text_area(&mut self, text: &str, buffer: &mut String, lines: u32) -> bool {
        self.operations
            .push(MockOperation::TextArea(text.to_string(), buffer.clone(), lines));
        if let Some(MockState::String(new_value)) = self.state.get(&format!("text:{}", text)) {
            if new_value != buffer {
                *buffer = new_value.clone();
                return true;
            }
        }
        false
    }

    fn color_edit(&mut self, text: &str, color: &mut [f32; 4]) -> bool {
        self.operations
            .push(MockOperation::ColorEdit(text.to_string(), *color));
        if let Some(MockState::Color(new_color)) = self.state.get(&format!("color:{}", text)) {
            if new_color != color {
                *color = *new_color;
                return true;
            }
        }
        false
    }

    fn combo(&mut self, text: &str, current: &mut usize, options: &[&str]) -> bool {
        self.operations.push(MockOperation::Combo(
            text.to_string(),
            *current,
            options.iter().map(|s| s.to_string()).collect(),
        ));
        if let Some(MockState::Index(new_index)) = self.state.get(&format!("combo:{}", text)) {
            if *new_index != *current {
                *current = *new_index;
                return true;
            }
        }
        false
    }

    fn progress(&mut self, fraction: f32, text: Option<&str>) {
        self.operations
            .push(MockOperation::Progress(fraction, text.map(|s| s.to_string())));
    }

    fn tooltip(&mut self, text: &str) {
        self.operations
            .push(MockOperation::Tooltip(text.to_string()));
    }

    fn disabled<F>(&mut self, disabled: bool, add_contents: F)
    where
        F: FnOnce(&mut Self),
    {
        self.operations.push(MockOperation::DisabledBegin(disabled));
        add_contents(self);
        self.operations.push(MockOperation::DisabledEnd);
    }

    fn id(&self) -> u64 {
        self.context_id
    }
}

// ---------------------------------------------------------------------------
// Command Executor (executes commands against a UIContext)
// ---------------------------------------------------------------------------

/// State for executing UI commands.
///
/// This is used on the renderer side to replay commands from Python.
#[derive(Debug, Default)]
pub struct CommandExecutorState {
    /// Widget values by ID.
    values: HashMap<u64, WidgetValue>,
}

/// A widget's current value.
#[derive(Debug, Clone)]
pub enum WidgetValue {
    Bool(bool),
    F32(f32),
    I32(i32),
    String(String),
    Color([f32; 4]),
    Index(usize),
}

impl CommandExecutorState {
    /// Create a new executor state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get a boolean value.
    pub fn get_bool(&self, id: u64) -> Option<bool> {
        match self.values.get(&id) {
            Some(WidgetValue::Bool(v)) => Some(*v),
            _ => None,
        }
    }

    /// Set a boolean value.
    pub fn set_bool(&mut self, id: u64, value: bool) {
        self.values.insert(id, WidgetValue::Bool(value));
    }

    /// Get a float value.
    pub fn get_f32(&self, id: u64) -> Option<f32> {
        match self.values.get(&id) {
            Some(WidgetValue::F32(v)) => Some(*v),
            _ => None,
        }
    }

    /// Set a float value.
    pub fn set_f32(&mut self, id: u64, value: f32) {
        self.values.insert(id, WidgetValue::F32(value));
    }

    /// Get an integer value.
    pub fn get_i32(&self, id: u64) -> Option<i32> {
        match self.values.get(&id) {
            Some(WidgetValue::I32(v)) => Some(*v),
            _ => None,
        }
    }

    /// Set an integer value.
    pub fn set_i32(&mut self, id: u64, value: i32) {
        self.values.insert(id, WidgetValue::I32(value));
    }

    /// Get a string value.
    pub fn get_string(&self, id: u64) -> Option<&str> {
        match self.values.get(&id) {
            Some(WidgetValue::String(v)) => Some(v.as_str()),
            _ => None,
        }
    }

    /// Set a string value.
    pub fn set_string(&mut self, id: u64, value: String) {
        self.values.insert(id, WidgetValue::String(value));
    }

    /// Get a color value.
    pub fn get_color(&self, id: u64) -> Option<[f32; 4]> {
        match self.values.get(&id) {
            Some(WidgetValue::Color(v)) => Some(*v),
            _ => None,
        }
    }

    /// Set a color value.
    pub fn set_color(&mut self, id: u64, color: [f32; 4]) {
        self.values.insert(id, WidgetValue::Color(color));
    }

    /// Get an index value.
    pub fn get_index(&self, id: u64) -> Option<usize> {
        match self.values.get(&id) {
            Some(WidgetValue::Index(v)) => Some(*v),
            _ => None,
        }
    }

    /// Set an index value.
    pub fn set_index(&mut self, id: u64, index: usize) {
        self.values.insert(id, WidgetValue::Index(index));
    }

    /// Clear all state.
    pub fn clear(&mut self) {
        self.values.clear();
    }
}

// ---------------------------------------------------------------------------
// Bridge Protocol Integration
// ---------------------------------------------------------------------------

/// Namespace for UI commands in the bridge protocol.
pub mod ui_ns {
    use super::*;
    use serde::{Deserialize, Serialize};

    /// Request to render a batch of UI commands.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct RenderRequest {
        /// Frame number.
        pub frame: u64,
        /// Commands to render.
        pub commands: Vec<UICommand>,
    }

    /// Response from rendering UI commands.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct RenderResponse {
        /// Frame number.
        pub frame: u64,
        /// Responses from interactive widgets.
        pub responses: Vec<UIResponse>,
    }

    /// Request to get current widget state.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GetStateRequest {
        /// Widget IDs to query.
        pub widget_ids: Vec<u64>,
    }

    /// Response with widget state.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GetStateResponse {
        /// Widget values by ID.
        pub values: HashMap<u64, serde_json::Value>,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // UICommand Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_ui_command_label() {
        let cmd = UICommand::Label {
            text: "Hello".to_string(),
        };
        assert!(!cmd.is_interactive());
        assert!(cmd.widget_id().is_none());
    }

    #[test]
    fn test_ui_command_button() {
        let cmd = UICommand::Button {
            id: 42,
            text: "Click me".to_string(),
        };
        assert!(cmd.is_interactive());
        assert_eq!(cmd.widget_id(), Some(42));
    }

    #[test]
    fn test_ui_command_checkbox() {
        let cmd = UICommand::Checkbox {
            id: 1,
            text: "Enable".to_string(),
            value: true,
        };
        assert!(cmd.is_interactive());
        assert_eq!(cmd.widget_id(), Some(1));
    }

    #[test]
    fn test_ui_command_slider() {
        let cmd = UICommand::Slider {
            id: 2,
            text: "Volume".to_string(),
            value: 0.5,
            min: 0.0,
            max: 1.0,
        };
        assert!(cmd.is_interactive());
        assert_eq!(cmd.widget_id(), Some(2));
    }

    #[test]
    fn test_ui_command_slider_int() {
        let cmd = UICommand::SliderInt {
            id: 3,
            text: "Count".to_string(),
            value: 10,
            min: 0,
            max: 100,
        };
        assert!(cmd.is_interactive());
        assert_eq!(cmd.widget_id(), Some(3));
    }

    #[test]
    fn test_ui_command_text_edit() {
        let cmd = UICommand::TextEdit {
            id: 4,
            text: "Name".to_string(),
            value: "John".to_string(),
        };
        assert!(cmd.is_interactive());
        assert_eq!(cmd.widget_id(), Some(4));
    }

    #[test]
    fn test_ui_command_color_edit() {
        let cmd = UICommand::ColorEdit {
            id: 5,
            text: "Color".to_string(),
            color: [1.0, 0.0, 0.0, 1.0],
        };
        assert!(cmd.is_interactive());
        assert_eq!(cmd.widget_id(), Some(5));
    }

    #[test]
    fn test_ui_command_combo() {
        let cmd = UICommand::Combo {
            id: 6,
            text: "Mode".to_string(),
            current: 0,
            options: vec!["Option A".to_string(), "Option B".to_string()],
        };
        assert!(cmd.is_interactive());
        assert_eq!(cmd.widget_id(), Some(6));
    }

    #[test]
    fn test_ui_command_separator() {
        let cmd = UICommand::Separator;
        assert!(!cmd.is_interactive());
        assert!(cmd.widget_id().is_none());
    }

    #[test]
    fn test_ui_command_spacing() {
        let cmd = UICommand::Spacing;
        assert!(!cmd.is_interactive());
    }

    // -------------------------------------------------------------------------
    // UIResponse Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_ui_response_button_clicked() {
        let resp = UIResponse::ButtonClicked { id: 42 };
        assert_eq!(resp.widget_id(), 42);
    }

    #[test]
    fn test_ui_response_checkbox_changed() {
        let resp = UIResponse::CheckboxChanged { id: 1, value: true };
        assert_eq!(resp.widget_id(), 1);
    }

    #[test]
    fn test_ui_response_slider_changed() {
        let resp = UIResponse::SliderChanged { id: 2, value: 0.75 };
        assert_eq!(resp.widget_id(), 2);
    }

    #[test]
    fn test_ui_response_text_changed() {
        let resp = UIResponse::TextChanged {
            id: 3,
            value: "New text".to_string(),
        };
        assert_eq!(resp.widget_id(), 3);
    }

    #[test]
    fn test_ui_response_color_changed() {
        let resp = UIResponse::ColorChanged {
            id: 4,
            color: [0.0, 1.0, 0.0, 1.0],
        };
        assert_eq!(resp.widget_id(), 4);
    }

    #[test]
    fn test_ui_response_combo_changed() {
        let resp = UIResponse::ComboChanged { id: 5, index: 2 };
        assert_eq!(resp.widget_id(), 5);
    }

    // -------------------------------------------------------------------------
    // Command Serialization Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_command_serialization_label() {
        let cmd = UICommand::Label {
            text: "Test".to_string(),
        };
        let json = serde_json::to_string(&cmd).unwrap();
        let parsed: UICommand = serde_json::from_str(&json).unwrap();
        assert_eq!(cmd, parsed);
    }

    #[test]
    fn test_command_serialization_button() {
        let cmd = UICommand::Button {
            id: 123,
            text: "OK".to_string(),
        };
        let json = serde_json::to_string(&cmd).unwrap();
        let parsed: UICommand = serde_json::from_str(&json).unwrap();
        assert_eq!(cmd, parsed);
    }

    #[test]
    fn test_command_serialization_slider() {
        let cmd = UICommand::Slider {
            id: 1,
            text: "Value".to_string(),
            value: 0.5,
            min: 0.0,
            max: 1.0,
        };
        let json = serde_json::to_string(&cmd).unwrap();
        let parsed: UICommand = serde_json::from_str(&json).unwrap();
        assert_eq!(cmd, parsed);
    }

    #[test]
    fn test_response_serialization() {
        let resp = UIResponse::SliderChanged { id: 42, value: 0.8 };
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: UIResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(resp, parsed);
    }

    // -------------------------------------------------------------------------
    // CommandBuffer Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_command_buffer_new() {
        let buf = CommandBuffer::new();
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn test_command_buffer_push() {
        let mut buf = CommandBuffer::new();
        buf.push(UICommand::Label {
            text: "Test".to_string(),
        });
        assert_eq!(buf.len(), 1);
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_command_buffer_responses() {
        let mut buf = CommandBuffer::new();
        buf.add_response(UIResponse::ButtonClicked { id: 42 });
        assert!(buf.button_clicked(42));
        assert!(!buf.button_clicked(43));
    }

    #[test]
    fn test_command_buffer_serialization() {
        let mut buf = CommandBuffer::for_frame(10);
        buf.push(UICommand::Label {
            text: "Hello".to_string(),
        });
        buf.push(UICommand::Button {
            id: 1,
            text: "OK".to_string(),
        });
        buf.add_response(UIResponse::ButtonClicked { id: 1 });

        let json = buf.to_json().unwrap();
        let parsed = CommandBuffer::from_json(&json).unwrap();

        assert_eq!(parsed.frame(), 10);
        assert_eq!(parsed.len(), 2);
        assert!(parsed.button_clicked(1));
    }

    #[test]
    fn test_command_buffer_clear() {
        let mut buf = CommandBuffer::new();
        buf.push(UICommand::Separator);
        buf.add_response(UIResponse::ButtonClicked { id: 1 });
        buf.clear();
        assert!(buf.is_empty());
        assert!(!buf.button_clicked(1));
    }

    #[test]
    fn test_command_buffer_merge() {
        let mut buf1 = CommandBuffer::new();
        buf1.push(UICommand::Label {
            text: "A".to_string(),
        });

        let mut buf2 = CommandBuffer::new();
        buf2.push(UICommand::Label {
            text: "B".to_string(),
        });
        buf2.add_response(UIResponse::ButtonClicked { id: 5 });

        buf1.merge(&buf2);
        assert_eq!(buf1.len(), 2);
        assert!(buf1.button_clicked(5));
    }

    // -------------------------------------------------------------------------
    // RemoteUIContext Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_remote_context_label() {
        let mut ctx = RemoteUIContext::new(1);
        ctx.label("Hello");
        assert_eq!(ctx.buffer().len(), 1);
        assert!(matches!(
            &ctx.buffer().commands()[0],
            UICommand::Label { text } if text == "Hello"
        ));
    }

    #[test]
    fn test_remote_context_button() {
        let mut ctx = RemoteUIContext::new(1);
        let clicked = ctx.button("OK");
        assert!(!clicked); // No response configured
        assert_eq!(ctx.buffer().len(), 1);
    }

    #[test]
    fn test_remote_context_checkbox() {
        let mut ctx = RemoteUIContext::new(1);
        let mut value = false;
        let changed = ctx.checkbox("Enable", &mut value);
        assert!(!changed);
        assert!(!value);
    }

    #[test]
    fn test_remote_context_slider() {
        let mut ctx = RemoteUIContext::new(1);
        let mut value = 0.5;
        let changed = ctx.slider("Volume", &mut value, 0.0, 1.0);
        assert!(!changed);
        assert_eq!(value, 0.5);
    }

    #[test]
    fn test_remote_context_text_edit() {
        let mut ctx = RemoteUIContext::new(1);
        let mut text = "Hello".to_string();
        let changed = ctx.text_edit("Name", &mut text);
        assert!(!changed);
        assert_eq!(text, "Hello");
    }

    #[test]
    fn test_remote_context_collapsing() {
        let mut ctx = RemoteUIContext::new(1);
        ctx.collapsing("Settings", |inner| {
            inner.label("Inside");
        });
        assert_eq!(ctx.buffer().len(), 3);
        assert!(matches!(
            &ctx.buffer().commands()[0],
            UICommand::CollapsingBegin { text, .. } if text == "Settings"
        ));
        assert!(matches!(
            &ctx.buffer().commands()[2],
            UICommand::CollapsingEnd
        ));
    }

    #[test]
    fn test_remote_context_horizontal() {
        let mut ctx = RemoteUIContext::new(1);
        ctx.horizontal(|inner| {
            inner.label("A");
            inner.label("B");
        });
        assert_eq!(ctx.buffer().len(), 4);
        assert!(matches!(
            &ctx.buffer().commands()[0],
            UICommand::HorizontalBegin
        ));
        assert!(matches!(
            &ctx.buffer().commands()[3],
            UICommand::HorizontalEnd
        ));
    }

    #[test]
    fn test_remote_context_vertical() {
        let mut ctx = RemoteUIContext::new(1);
        ctx.vertical(|inner| {
            inner.label("A");
        });
        assert_eq!(ctx.buffer().len(), 3);
    }

    #[test]
    fn test_remote_context_nesting() {
        let mut ctx = RemoteUIContext::new(1);
        assert_eq!(ctx.nesting_depth(), 0);
        ctx.horizontal(|inner| {
            assert_eq!(inner.nesting_depth(), 1);
            inner.vertical(|inner2| {
                assert_eq!(inner2.nesting_depth(), 2);
            });
        });
        assert_eq!(ctx.nesting_depth(), 0);
    }

    #[test]
    fn test_remote_context_take_buffer() {
        let mut ctx = RemoteUIContext::new(1);
        ctx.label("Test");
        let buf = ctx.take_buffer();
        assert_eq!(buf.len(), 1);
        assert!(ctx.buffer().is_empty());
    }

    #[test]
    fn test_remote_context_id() {
        let ctx = RemoteUIContext::new(42);
        assert_eq!(ctx.id(), 42);
    }

    // -------------------------------------------------------------------------
    // MockUIContext Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_mock_context_label() {
        let mut ctx = MockUIContext::new(1);
        ctx.label("Hello");
        assert!(ctx.has_operation(&MockOperation::Label("Hello".to_string())));
    }

    #[test]
    fn test_mock_context_button_click() {
        let mut ctx = MockUIContext::new(1);
        ctx.click_button("OK");
        let clicked = ctx.button("OK");
        assert!(clicked);
    }

    #[test]
    fn test_mock_context_button_no_click() {
        let mut ctx = MockUIContext::new(1);
        let clicked = ctx.button("Cancel");
        assert!(!clicked);
    }

    #[test]
    fn test_mock_context_checkbox_state() {
        let mut ctx = MockUIContext::new(1);
        ctx.set_checkbox_state("Enable", true);
        let mut value = false;
        let changed = ctx.checkbox("Enable", &mut value);
        assert!(changed);
        assert!(value);
    }

    #[test]
    fn test_mock_context_slider_state() {
        let mut ctx = MockUIContext::new(1);
        ctx.set_slider_state("Volume", 0.75);
        let mut value = 0.5;
        let changed = ctx.slider("Volume", &mut value, 0.0, 1.0);
        assert!(changed);
        assert!((value - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn test_mock_context_text_state() {
        let mut ctx = MockUIContext::new(1);
        ctx.set_text_state("Name", "Alice");
        let mut text = "Bob".to_string();
        let changed = ctx.text_edit("Name", &mut text);
        assert!(changed);
        assert_eq!(text, "Alice");
    }

    #[test]
    fn test_mock_context_collapsing() {
        let mut ctx = MockUIContext::new(1);
        ctx.collapsing("Settings", |inner| {
            inner.label("Inside");
        });
        assert!(ctx.has_operation(&MockOperation::CollapsingBegin("Settings".to_string())));
        assert!(ctx.has_operation(&MockOperation::Label("Inside".to_string())));
        assert!(ctx.has_operation(&MockOperation::CollapsingEnd));
    }

    #[test]
    fn test_mock_context_clear() {
        let mut ctx = MockUIContext::new(1);
        ctx.label("Test");
        ctx.clear_operations();
        assert!(ctx.operations().is_empty());
    }

    // -------------------------------------------------------------------------
    // CommandExecutorState Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_executor_state_bool() {
        let mut state = CommandExecutorState::new();
        state.set_bool(1, true);
        assert_eq!(state.get_bool(1), Some(true));
        assert_eq!(state.get_bool(2), None);
    }

    #[test]
    fn test_executor_state_f32() {
        let mut state = CommandExecutorState::new();
        state.set_f32(1, 0.5);
        assert_eq!(state.get_f32(1), Some(0.5));
    }

    #[test]
    fn test_executor_state_i32() {
        let mut state = CommandExecutorState::new();
        state.set_i32(1, 42);
        assert_eq!(state.get_i32(1), Some(42));
    }

    #[test]
    fn test_executor_state_string() {
        let mut state = CommandExecutorState::new();
        state.set_string(1, "hello".to_string());
        assert_eq!(state.get_string(1), Some("hello"));
    }

    #[test]
    fn test_executor_state_color() {
        let mut state = CommandExecutorState::new();
        state.set_color(1, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(state.get_color(1), Some([1.0, 0.0, 0.0, 1.0]));
    }

    #[test]
    fn test_executor_state_index() {
        let mut state = CommandExecutorState::new();
        state.set_index(1, 5);
        assert_eq!(state.get_index(1), Some(5));
    }

    #[test]
    fn test_executor_state_clear() {
        let mut state = CommandExecutorState::new();
        state.set_bool(1, true);
        state.clear();
        assert_eq!(state.get_bool(1), None);
    }

    // -------------------------------------------------------------------------
    // Protocol Integration Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_render_request_serialization() {
        let req = ui_ns::RenderRequest {
            frame: 100,
            commands: vec![
                UICommand::Label {
                    text: "Test".to_string(),
                },
                UICommand::Button {
                    id: 1,
                    text: "OK".to_string(),
                },
            ],
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: ui_ns::RenderRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.frame, 100);
        assert_eq!(parsed.commands.len(), 2);
    }

    #[test]
    fn test_render_response_serialization() {
        let resp = ui_ns::RenderResponse {
            frame: 100,
            responses: vec![
                UIResponse::ButtonClicked { id: 1 },
                UIResponse::SliderChanged { id: 2, value: 0.5 },
            ],
        };
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: ui_ns::RenderResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.frame, 100);
        assert_eq!(parsed.responses.len(), 2);
    }

    // -------------------------------------------------------------------------
    // Extended Widget Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_mock_context_slider_int() {
        let mut ctx = MockUIContext::new(1);
        ctx.set_slider_int_state("Count", 50);
        let mut value = 10;
        let changed = ctx.slider_int("Count", &mut value, 0, 100);
        assert!(changed);
        assert_eq!(value, 50);
    }

    #[test]
    fn test_mock_context_color_edit() {
        let mut ctx = MockUIContext::new(1);
        ctx.set_color_state("Color", [0.0, 1.0, 0.0, 1.0]);
        let mut color = [1.0, 0.0, 0.0, 1.0];
        let changed = ctx.color_edit("Color", &mut color);
        assert!(changed);
        assert_eq!(color, [0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_mock_context_combo() {
        let mut ctx = MockUIContext::new(1);
        ctx.set_combo_state("Mode", 2);
        let mut current = 0;
        let options = ["A", "B", "C"];
        let changed = ctx.combo("Mode", &mut current, &options);
        assert!(changed);
        assert_eq!(current, 2);
    }

    #[test]
    fn test_mock_context_progress() {
        let mut ctx = MockUIContext::new(1);
        ctx.progress(0.5, Some("Loading..."));
        assert!(ctx.has_operation(&MockOperation::Progress(0.5, Some("Loading...".to_string()))));
    }

    #[test]
    fn test_mock_context_tooltip() {
        let mut ctx = MockUIContext::new(1);
        ctx.tooltip("Help text");
        assert!(ctx.has_operation(&MockOperation::Tooltip("Help text".to_string())));
    }

    #[test]
    fn test_mock_context_disabled() {
        let mut ctx = MockUIContext::new(1);
        ctx.disabled(true, |inner| {
            inner.button("Disabled Button");
        });
        assert!(ctx.has_operation(&MockOperation::DisabledBegin(true)));
        assert!(ctx.has_operation(&MockOperation::Button("Disabled Button".to_string())));
        assert!(ctx.has_operation(&MockOperation::DisabledEnd));
    }

    #[test]
    fn test_remote_context_combo() {
        let mut ctx = RemoteUIContext::new(1);
        let mut current = 0;
        let options = ["A", "B", "C"];
        ctx.combo("Mode", &mut current, &options);
        assert!(matches!(
            &ctx.buffer().commands()[0],
            UICommand::Combo { text, current: 0, options, .. }
            if text == "Mode" && options.len() == 3
        ));
    }

    #[test]
    fn test_remote_context_progress() {
        let mut ctx = RemoteUIContext::new(1);
        ctx.progress(0.75, None);
        assert!(matches!(
            &ctx.buffer().commands()[0],
            UICommand::Progress { fraction, text: None } if (*fraction - 0.75).abs() < f32::EPSILON
        ));
    }

    #[test]
    fn test_remote_context_disabled() {
        let mut ctx = RemoteUIContext::new(1);
        ctx.disabled(true, |inner| {
            inner.label("Disabled");
        });
        assert_eq!(ctx.buffer().len(), 3);
        assert!(matches!(
            &ctx.buffer().commands()[0],
            UICommand::DisabledBegin { disabled: true }
        ));
    }
}
