//! EguiUIContext Input Handling for Python-Rust UI Bridge
//!
//! This module provides input event mapping from Python to egui's input system.
//! It enables Python code to send keyboard, mouse, and window events to the
//! Rust egui renderer through the bridge protocol.
//!
//! # Architecture
//!
//! ```text
//! Python UI Code           Bridge              Rust (egui)
//! ==============           ======              ===========
//!     |                       |                     |
//!     +-- KeyPressed --------+-- InputEvent ------>| egui::RawInput
//!     +-- MouseMoved --------+-- InputEvent ------>|
//!     +-- WindowResize ------+-- InputEvent ------>|
//!     |                       |                     |
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use renderer_backend::egui_input::{InputMapper, InputEvent, KeyCode, Modifiers};
//!
//! let mut mapper = InputMapper::new();
//!
//! // Push events from Python
//! mapper.push_event(InputEvent::MouseMoved { x: 100.0, y: 200.0 });
//! mapper.push_event(InputEvent::KeyPressed {
//!     key: KeyCode::A,
//!     modifiers: Modifiers::CTRL,
//! });
//! mapper.push_event(InputEvent::CharTyped { char: 'a' });
//!
//! // Get egui-compatible RawInput
//! let raw_input = mapper.take_raw_input();
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Key Codes
// ---------------------------------------------------------------------------

/// Key codes for keyboard input, compatible with egui::Key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KeyCode {
    // Letters
    A, B, C, D, E, F, G, H, I, J, K, L, M,
    N, O, P, Q, R, S, T, U, V, W, X, Y, Z,

    // Numbers (top row)
    Num0, Num1, Num2, Num3, Num4,
    Num5, Num6, Num7, Num8, Num9,

    // Function keys
    F1, F2, F3, F4, F5, F6,
    F7, F8, F9, F10, F11, F12,

    // Navigation
    ArrowDown, ArrowLeft, ArrowRight, ArrowUp,
    Home, End, PageUp, PageDown,

    // Editing
    Backspace, Delete, Insert,
    Enter, Tab,

    // Modifiers (as keys, not as modifier state)
    LeftShift, RightShift,
    LeftCtrl, RightCtrl,
    LeftAlt, RightAlt,
    LeftMeta, RightMeta,

    // Special
    Space, Escape, CapsLock,
    PrintScreen, ScrollLock, Pause,
    NumLock,

    // Punctuation
    Minus, Equals, LeftBracket, RightBracket,
    Backslash, Semicolon, Apostrophe,
    Comma, Period, Slash, Grave,

    // Numpad
    Numpad0, Numpad1, Numpad2, Numpad3, Numpad4,
    Numpad5, Numpad6, Numpad7, Numpad8, Numpad9,
    NumpadAdd, NumpadSubtract, NumpadMultiply, NumpadDivide,
    NumpadDecimal, NumpadEnter,

    // Media keys
    MediaPlayPause, MediaStop, MediaNext, MediaPrevious,
    VolumeUp, VolumeDown, VolumeMute,

    // Unknown key (for unmapped keys)
    Unknown,
}

impl KeyCode {
    /// Check if this is a letter key (A-Z).
    pub fn is_letter(&self) -> bool {
        matches!(
            self,
            KeyCode::A | KeyCode::B | KeyCode::C | KeyCode::D | KeyCode::E |
            KeyCode::F | KeyCode::G | KeyCode::H | KeyCode::I | KeyCode::J |
            KeyCode::K | KeyCode::L | KeyCode::M | KeyCode::N | KeyCode::O |
            KeyCode::P | KeyCode::Q | KeyCode::R | KeyCode::S | KeyCode::T |
            KeyCode::U | KeyCode::V | KeyCode::W | KeyCode::X | KeyCode::Y |
            KeyCode::Z
        )
    }

    /// Check if this is a number key (0-9).
    pub fn is_number(&self) -> bool {
        matches!(
            self,
            KeyCode::Num0 | KeyCode::Num1 | KeyCode::Num2 | KeyCode::Num3 |
            KeyCode::Num4 | KeyCode::Num5 | KeyCode::Num6 | KeyCode::Num7 |
            KeyCode::Num8 | KeyCode::Num9
        )
    }

    /// Check if this is a function key (F1-F12).
    pub fn is_function_key(&self) -> bool {
        matches!(
            self,
            KeyCode::F1 | KeyCode::F2 | KeyCode::F3 | KeyCode::F4 |
            KeyCode::F5 | KeyCode::F6 | KeyCode::F7 | KeyCode::F8 |
            KeyCode::F9 | KeyCode::F10 | KeyCode::F11 | KeyCode::F12
        )
    }

    /// Check if this is a numpad key.
    pub fn is_numpad(&self) -> bool {
        matches!(
            self,
            KeyCode::Numpad0 | KeyCode::Numpad1 | KeyCode::Numpad2 |
            KeyCode::Numpad3 | KeyCode::Numpad4 | KeyCode::Numpad5 |
            KeyCode::Numpad6 | KeyCode::Numpad7 | KeyCode::Numpad8 |
            KeyCode::Numpad9 | KeyCode::NumpadAdd | KeyCode::NumpadSubtract |
            KeyCode::NumpadMultiply | KeyCode::NumpadDivide |
            KeyCode::NumpadDecimal | KeyCode::NumpadEnter
        )
    }

    /// Check if this is a modifier key.
    pub fn is_modifier(&self) -> bool {
        matches!(
            self,
            KeyCode::LeftShift | KeyCode::RightShift |
            KeyCode::LeftCtrl | KeyCode::RightCtrl |
            KeyCode::LeftAlt | KeyCode::RightAlt |
            KeyCode::LeftMeta | KeyCode::RightMeta
        )
    }

    /// Check if this is an arrow key.
    pub fn is_arrow(&self) -> bool {
        matches!(
            self,
            KeyCode::ArrowDown | KeyCode::ArrowLeft |
            KeyCode::ArrowRight | KeyCode::ArrowUp
        )
    }

    /// Get the character representation of a letter or number key.
    /// Returns None for non-character keys.
    pub fn to_char(&self, shift: bool) -> Option<char> {
        let c = match self {
            KeyCode::A => 'a', KeyCode::B => 'b', KeyCode::C => 'c',
            KeyCode::D => 'd', KeyCode::E => 'e', KeyCode::F => 'f',
            KeyCode::G => 'g', KeyCode::H => 'h', KeyCode::I => 'i',
            KeyCode::J => 'j', KeyCode::K => 'k', KeyCode::L => 'l',
            KeyCode::M => 'm', KeyCode::N => 'n', KeyCode::O => 'o',
            KeyCode::P => 'p', KeyCode::Q => 'q', KeyCode::R => 'r',
            KeyCode::S => 's', KeyCode::T => 't', KeyCode::U => 'u',
            KeyCode::V => 'v', KeyCode::W => 'w', KeyCode::X => 'x',
            KeyCode::Y => 'y', KeyCode::Z => 'z',

            KeyCode::Num0 => if shift { ')' } else { '0' },
            KeyCode::Num1 => if shift { '!' } else { '1' },
            KeyCode::Num2 => if shift { '@' } else { '2' },
            KeyCode::Num3 => if shift { '#' } else { '3' },
            KeyCode::Num4 => if shift { '$' } else { '4' },
            KeyCode::Num5 => if shift { '%' } else { '5' },
            KeyCode::Num6 => if shift { '^' } else { '6' },
            KeyCode::Num7 => if shift { '&' } else { '7' },
            KeyCode::Num8 => if shift { '*' } else { '8' },
            KeyCode::Num9 => if shift { '(' } else { '9' },

            KeyCode::Space => ' ',
            KeyCode::Minus => if shift { '_' } else { '-' },
            KeyCode::Equals => if shift { '+' } else { '=' },
            KeyCode::LeftBracket => if shift { '{' } else { '[' },
            KeyCode::RightBracket => if shift { '}' } else { ']' },
            KeyCode::Backslash => if shift { '|' } else { '\\' },
            KeyCode::Semicolon => if shift { ':' } else { ';' },
            KeyCode::Apostrophe => if shift { '"' } else { '\'' },
            KeyCode::Comma => if shift { '<' } else { ',' },
            KeyCode::Period => if shift { '>' } else { '.' },
            KeyCode::Slash => if shift { '?' } else { '/' },
            KeyCode::Grave => if shift { '~' } else { '`' },

            KeyCode::Numpad0 => '0', KeyCode::Numpad1 => '1',
            KeyCode::Numpad2 => '2', KeyCode::Numpad3 => '3',
            KeyCode::Numpad4 => '4', KeyCode::Numpad5 => '5',
            KeyCode::Numpad6 => '6', KeyCode::Numpad7 => '7',
            KeyCode::Numpad8 => '8', KeyCode::Numpad9 => '9',
            KeyCode::NumpadAdd => '+', KeyCode::NumpadSubtract => '-',
            KeyCode::NumpadMultiply => '*', KeyCode::NumpadDivide => '/',
            KeyCode::NumpadDecimal => '.',

            _ => return None,
        };

        if self.is_letter() && shift {
            Some(c.to_ascii_uppercase())
        } else {
            Some(c)
        }
    }

    /// Parse a key code from a string name.
    pub fn from_name(name: &str) -> Option<Self> {
        let name = name.to_lowercase();
        Some(match name.as_str() {
            "a" => KeyCode::A, "b" => KeyCode::B, "c" => KeyCode::C,
            "d" => KeyCode::D, "e" => KeyCode::E, "f" => KeyCode::F,
            "g" => KeyCode::G, "h" => KeyCode::H, "i" => KeyCode::I,
            "j" => KeyCode::J, "k" => KeyCode::K, "l" => KeyCode::L,
            "m" => KeyCode::M, "n" => KeyCode::N, "o" => KeyCode::O,
            "p" => KeyCode::P, "q" => KeyCode::Q, "r" => KeyCode::R,
            "s" => KeyCode::S, "t" => KeyCode::T, "u" => KeyCode::U,
            "v" => KeyCode::V, "w" => KeyCode::W, "x" => KeyCode::X,
            "y" => KeyCode::Y, "z" => KeyCode::Z,

            "0" | "num0" => KeyCode::Num0,
            "1" | "num1" => KeyCode::Num1,
            "2" | "num2" => KeyCode::Num2,
            "3" | "num3" => KeyCode::Num3,
            "4" | "num4" => KeyCode::Num4,
            "5" | "num5" => KeyCode::Num5,
            "6" | "num6" => KeyCode::Num6,
            "7" | "num7" => KeyCode::Num7,
            "8" | "num8" => KeyCode::Num8,
            "9" | "num9" => KeyCode::Num9,

            "f1" => KeyCode::F1, "f2" => KeyCode::F2, "f3" => KeyCode::F3,
            "f4" => KeyCode::F4, "f5" => KeyCode::F5, "f6" => KeyCode::F6,
            "f7" => KeyCode::F7, "f8" => KeyCode::F8, "f9" => KeyCode::F9,
            "f10" => KeyCode::F10, "f11" => KeyCode::F11, "f12" => KeyCode::F12,

            "down" | "arrowdown" => KeyCode::ArrowDown,
            "left" | "arrowleft" => KeyCode::ArrowLeft,
            "right" | "arrowright" => KeyCode::ArrowRight,
            "up" | "arrowup" => KeyCode::ArrowUp,

            "home" => KeyCode::Home, "end" => KeyCode::End,
            "pageup" => KeyCode::PageUp, "pagedown" => KeyCode::PageDown,

            "backspace" => KeyCode::Backspace,
            "delete" => KeyCode::Delete,
            "insert" => KeyCode::Insert,
            "enter" | "return" => KeyCode::Enter,
            "tab" => KeyCode::Tab,
            "space" => KeyCode::Space,
            "escape" | "esc" => KeyCode::Escape,

            "shift" | "leftshift" => KeyCode::LeftShift,
            "rightshift" => KeyCode::RightShift,
            "ctrl" | "control" | "leftctrl" => KeyCode::LeftCtrl,
            "rightctrl" => KeyCode::RightCtrl,
            "alt" | "leftalt" => KeyCode::LeftAlt,
            "rightalt" => KeyCode::RightAlt,
            "meta" | "super" | "win" | "cmd" | "leftmeta" => KeyCode::LeftMeta,
            "rightmeta" => KeyCode::RightMeta,

            _ => return None,
        })
    }
}

impl Default for KeyCode {
    fn default() -> Self {
        KeyCode::Unknown
    }
}

// ---------------------------------------------------------------------------
// Mouse Buttons
// ---------------------------------------------------------------------------

/// Mouse button identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MouseButton {
    #[default]
    Left,
    Right,
    Middle,
    /// Extra button 1 (typically "back" button).
    Extra1,
    /// Extra button 2 (typically "forward" button).
    Extra2,
}

impl MouseButton {
    /// Parse a mouse button from a string name.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "left" | "primary" | "0" => Some(MouseButton::Left),
            "right" | "secondary" | "1" => Some(MouseButton::Right),
            "middle" | "2" => Some(MouseButton::Middle),
            "extra1" | "back" | "3" => Some(MouseButton::Extra1),
            "extra2" | "forward" | "4" => Some(MouseButton::Extra2),
            _ => None,
        }
    }

    /// Get the button index (0 = Left, 1 = Right, etc.).
    pub fn index(&self) -> u8 {
        match self {
            MouseButton::Left => 0,
            MouseButton::Right => 1,
            MouseButton::Middle => 2,
            MouseButton::Extra1 => 3,
            MouseButton::Extra2 => 4,
        }
    }
}

// ---------------------------------------------------------------------------
// Modifiers
// ---------------------------------------------------------------------------

/// Keyboard modifier state.
///
/// Uses bitflags for efficient storage and combination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct Modifiers {
    /// Ctrl key is held.
    pub ctrl: bool,
    /// Shift key is held.
    pub shift: bool,
    /// Alt key is held (Option on macOS).
    pub alt: bool,
    /// Meta key is held (Cmd on macOS, Win on Windows).
    pub meta: bool,
}

impl Modifiers {
    /// No modifiers held.
    pub const NONE: Self = Self {
        ctrl: false,
        shift: false,
        alt: false,
        meta: false,
    };

    /// Ctrl modifier only.
    pub const CTRL: Self = Self {
        ctrl: true,
        shift: false,
        alt: false,
        meta: false,
    };

    /// Shift modifier only.
    pub const SHIFT: Self = Self {
        ctrl: false,
        shift: true,
        alt: false,
        meta: false,
    };

    /// Alt modifier only.
    pub const ALT: Self = Self {
        ctrl: false,
        shift: false,
        alt: true,
        meta: false,
    };

    /// Meta modifier only.
    pub const META: Self = Self {
        ctrl: false,
        shift: false,
        alt: false,
        meta: true,
    };

    /// Create new modifiers with specified state.
    pub fn new(ctrl: bool, shift: bool, alt: bool, meta: bool) -> Self {
        Self { ctrl, shift, alt, meta }
    }

    /// Check if any modifier is held.
    pub fn any(&self) -> bool {
        self.ctrl || self.shift || self.alt || self.meta
    }

    /// Check if no modifiers are held.
    pub fn none(&self) -> bool {
        !self.any()
    }

    /// Check if this represents Ctrl+C (copy) on the current platform.
    pub fn is_copy_shortcut(&self) -> bool {
        // Ctrl+C on Windows/Linux, Cmd+C on macOS
        (self.ctrl && !self.shift && !self.alt && !self.meta) ||
        (!self.ctrl && !self.shift && !self.alt && self.meta)
    }

    /// Check if this represents Ctrl+V (paste) on the current platform.
    pub fn is_paste_shortcut(&self) -> bool {
        self.is_copy_shortcut() // Same modifier combination
    }

    /// Check if this represents Ctrl+Z (undo) on the current platform.
    pub fn is_undo_shortcut(&self) -> bool {
        self.is_copy_shortcut() // Same modifier combination
    }

    /// Combine with another modifier set (OR).
    pub fn union(&self, other: &Self) -> Self {
        Self {
            ctrl: self.ctrl || other.ctrl,
            shift: self.shift || other.shift,
            alt: self.alt || other.alt,
            meta: self.meta || other.meta,
        }
    }

    /// Convert to a bitmask (ctrl=1, shift=2, alt=4, meta=8).
    pub fn to_bits(&self) -> u8 {
        let mut bits = 0;
        if self.ctrl { bits |= 1; }
        if self.shift { bits |= 2; }
        if self.alt { bits |= 4; }
        if self.meta { bits |= 8; }
        bits
    }

    /// Create from a bitmask (ctrl=1, shift=2, alt=4, meta=8).
    pub fn from_bits(bits: u8) -> Self {
        Self {
            ctrl: bits & 1 != 0,
            shift: bits & 2 != 0,
            alt: bits & 4 != 0,
            meta: bits & 8 != 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Input Events
// ---------------------------------------------------------------------------

/// Input events from Python to be mapped to egui's input system.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputEvent {
    /// Key was pressed.
    KeyPressed {
        key: KeyCode,
        modifiers: Modifiers,
    },

    /// Key was released.
    KeyReleased {
        key: KeyCode,
        modifiers: Modifiers,
    },

    /// Character was typed (after keyboard layout processing).
    CharTyped {
        #[serde(rename = "char")]
        character: char,
    },

    /// Mouse cursor moved.
    MouseMoved {
        x: f32,
        y: f32,
    },

    /// Mouse button was pressed.
    MousePressed {
        button: MouseButton,
        x: f32,
        y: f32,
    },

    /// Mouse button was released.
    MouseReleased {
        button: MouseButton,
        x: f32,
        y: f32,
    },

    /// Mouse wheel was scrolled.
    MouseWheel {
        delta_x: f32,
        delta_y: f32,
    },

    /// Window was resized.
    WindowResize {
        width: u32,
        height: u32,
    },

    /// Window gained or lost focus.
    WindowFocus {
        focused: bool,
    },

    /// Touch event started.
    TouchStart {
        id: u64,
        x: f32,
        y: f32,
    },

    /// Touch event moved.
    TouchMove {
        id: u64,
        x: f32,
        y: f32,
    },

    /// Touch event ended.
    TouchEnd {
        id: u64,
        x: f32,
        y: f32,
    },

    /// Touch event was cancelled.
    TouchCancel {
        id: u64,
    },

    /// IME composition started.
    ImeCompositionStart,

    /// IME composition updated.
    ImeCompositionUpdate {
        text: String,
        cursor: Option<usize>,
    },

    /// IME composition ended.
    ImeCompositionEnd {
        text: String,
    },

    /// Copy command (Ctrl+C / Cmd+C).
    Copy,

    /// Cut command (Ctrl+X / Cmd+X).
    Cut,

    /// Paste command with text (Ctrl+V / Cmd+V).
    Paste {
        text: String,
    },

    /// File was dropped on the window.
    FileDrop {
        path: String,
        x: f32,
        y: f32,
    },

    /// Files are being hovered over the window.
    FileHover {
        paths: Vec<String>,
        x: f32,
        y: f32,
    },

    /// File hover ended (files dragged away).
    FileHoverCancel,
}

impl InputEvent {
    /// Check if this event is a keyboard event.
    pub fn is_keyboard(&self) -> bool {
        matches!(
            self,
            InputEvent::KeyPressed { .. } |
            InputEvent::KeyReleased { .. } |
            InputEvent::CharTyped { .. }
        )
    }

    /// Check if this event is a mouse event.
    pub fn is_mouse(&self) -> bool {
        matches!(
            self,
            InputEvent::MouseMoved { .. } |
            InputEvent::MousePressed { .. } |
            InputEvent::MouseReleased { .. } |
            InputEvent::MouseWheel { .. }
        )
    }

    /// Check if this event is a touch event.
    pub fn is_touch(&self) -> bool {
        matches!(
            self,
            InputEvent::TouchStart { .. } |
            InputEvent::TouchMove { .. } |
            InputEvent::TouchEnd { .. } |
            InputEvent::TouchCancel { .. }
        )
    }

    /// Check if this event is a window event.
    pub fn is_window(&self) -> bool {
        matches!(
            self,
            InputEvent::WindowResize { .. } |
            InputEvent::WindowFocus { .. }
        )
    }

    /// Get the position for mouse/touch events.
    pub fn position(&self) -> Option<(f32, f32)> {
        match self {
            InputEvent::MouseMoved { x, y } |
            InputEvent::MousePressed { x, y, .. } |
            InputEvent::MouseReleased { x, y, .. } |
            InputEvent::TouchStart { x, y, .. } |
            InputEvent::TouchMove { x, y, .. } |
            InputEvent::TouchEnd { x, y, .. } |
            InputEvent::FileDrop { x, y, .. } |
            InputEvent::FileHover { x, y, .. } => Some((*x, *y)),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// RawInput (egui-compatible structure)
// ---------------------------------------------------------------------------

/// egui-compatible raw input structure.
///
/// This mirrors egui::RawInput for serialization over the bridge.
/// It can be converted to egui::RawInput when egui is available.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RawInput {
    /// Keyboard events.
    pub events: Vec<RawInputEvent>,

    /// Current screen rect (pixels).
    pub screen_rect: Option<Rect>,

    /// Pixels per point (DPI scale factor).
    pub pixels_per_point: Option<f32>,

    /// Maximum texture side length.
    pub max_texture_side: Option<usize>,

    /// Time since last frame (seconds).
    pub predicted_dt: f32,

    /// Current modifier state.
    pub modifiers: Modifiers,

    /// Focused widget (if any).
    pub focused: bool,
}

/// A rectangle in screen coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Rect {
    pub min_x: f32,
    pub min_y: f32,
    pub max_x: f32,
    pub max_y: f32,
}

impl Rect {
    pub fn new(min_x: f32, min_y: f32, max_x: f32, max_y: f32) -> Self {
        Self { min_x, min_y, max_x, max_y }
    }

    pub fn from_size(width: f32, height: f32) -> Self {
        Self { min_x: 0.0, min_y: 0.0, max_x: width, max_y: height }
    }

    pub fn width(&self) -> f32 {
        self.max_x - self.min_x
    }

    pub fn height(&self) -> f32 {
        self.max_y - self.min_y
    }

    pub fn contains(&self, x: f32, y: f32) -> bool {
        x >= self.min_x && x <= self.max_x && y >= self.min_y && y <= self.max_y
    }
}

impl Default for Rect {
    fn default() -> Self {
        Self::from_size(1920.0, 1080.0)
    }
}

/// Raw input event types (matches egui::Event structure).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RawInputEvent {
    /// Pointer (mouse/touch) moved.
    PointerMoved {
        x: f32,
        y: f32,
    },

    /// Pointer button state changed.
    PointerButton {
        x: f32,
        y: f32,
        button: MouseButton,
        pressed: bool,
        modifiers: Modifiers,
    },

    /// Pointer left the window.
    PointerGone,

    /// Mouse wheel scrolled.
    Scroll {
        delta_x: f32,
        delta_y: f32,
    },

    /// Key pressed or released.
    Key {
        key: KeyCode,
        pressed: bool,
        repeat: bool,
        modifiers: Modifiers,
    },

    /// Text typed.
    Text {
        text: String,
    },

    /// Copy command.
    Copy,

    /// Cut command.
    Cut,

    /// Paste command.
    Paste {
        text: String,
    },

    /// Window focused.
    WindowFocused {
        focused: bool,
    },

    /// Touch event.
    Touch {
        id: u64,
        phase: TouchPhase,
        x: f32,
        y: f32,
    },
}

/// Touch event phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TouchPhase {
    Start,
    Move,
    End,
    Cancel,
}

// ---------------------------------------------------------------------------
// Input Mapper
// ---------------------------------------------------------------------------

/// Maps input events from Python to egui's RawInput format.
pub struct InputMapper {
    /// Current screen size.
    screen_size: (u32, u32),

    /// Pixels per point (DPI scale).
    pixels_per_point: f32,

    /// Current mouse position.
    mouse_position: Option<(f32, f32)>,

    /// Currently pressed mouse buttons.
    pressed_buttons: HashSet<MouseButton>,

    /// Currently pressed keys.
    pressed_keys: HashSet<KeyCode>,

    /// Current modifier state.
    modifiers: Modifiers,

    /// Whether the window is focused.
    focused: bool,

    /// Accumulated events for the current frame.
    events: Vec<RawInputEvent>,

    /// Pending characters for repeat detection.
    last_key_event: Option<(KeyCode, std::time::Instant)>,

    /// Frame delta time.
    dt: f32,
}

impl InputMapper {
    /// Create a new input mapper.
    pub fn new() -> Self {
        Self {
            screen_size: (1920, 1080),
            pixels_per_point: 1.0,
            mouse_position: None,
            pressed_buttons: HashSet::new(),
            pressed_keys: HashSet::new(),
            modifiers: Modifiers::NONE,
            focused: true,
            events: Vec::new(),
            last_key_event: None,
            dt: 1.0 / 60.0,
        }
    }

    /// Create an input mapper with a specific screen size.
    pub fn with_screen_size(width: u32, height: u32) -> Self {
        let mut mapper = Self::new();
        mapper.screen_size = (width, height);
        mapper
    }

    /// Set the screen size.
    pub fn set_screen_size(&mut self, width: u32, height: u32) {
        self.screen_size = (width, height);
    }

    /// Get the current screen size.
    pub fn screen_size(&self) -> (u32, u32) {
        self.screen_size
    }

    /// Set the DPI scale factor.
    pub fn set_pixels_per_point(&mut self, ppp: f32) {
        self.pixels_per_point = ppp;
    }

    /// Get the DPI scale factor.
    pub fn pixels_per_point(&self) -> f32 {
        self.pixels_per_point
    }

    /// Set the frame delta time.
    pub fn set_dt(&mut self, dt: f32) {
        self.dt = dt;
    }

    /// Get the current modifier state.
    pub fn modifiers(&self) -> Modifiers {
        self.modifiers
    }

    /// Get the current mouse position.
    pub fn mouse_position(&self) -> Option<(f32, f32)> {
        self.mouse_position
    }

    /// Check if a mouse button is pressed.
    pub fn is_button_pressed(&self, button: MouseButton) -> bool {
        self.pressed_buttons.contains(&button)
    }

    /// Check if a key is pressed.
    pub fn is_key_pressed(&self, key: KeyCode) -> bool {
        self.pressed_keys.contains(&key)
    }

    /// Check if the window is focused.
    pub fn is_focused(&self) -> bool {
        self.focused
    }

    /// Push an input event.
    pub fn push_event(&mut self, event: InputEvent) {
        match event {
            InputEvent::KeyPressed { key, modifiers } => {
                // First apply the explicit modifiers from the event
                self.modifiers = modifiers;
                // Then update based on the key being pressed (modifier keys update state)
                self.update_modifiers_from_key(&key, true);

                let repeat = self.pressed_keys.contains(&key);
                self.pressed_keys.insert(key);

                self.events.push(RawInputEvent::Key {
                    key,
                    pressed: true,
                    repeat,
                    modifiers: self.modifiers,
                });

                self.last_key_event = Some((key, std::time::Instant::now()));
            }

            InputEvent::KeyReleased { key, modifiers } => {
                // First apply the explicit modifiers from the event
                self.modifiers = modifiers;
                // Then update based on the key being released
                self.update_modifiers_from_key(&key, false);
                self.pressed_keys.remove(&key);

                self.events.push(RawInputEvent::Key {
                    key,
                    pressed: false,
                    repeat: false,
                    modifiers: self.modifiers,
                });
            }

            InputEvent::CharTyped { character } => {
                self.events.push(RawInputEvent::Text {
                    text: character.to_string(),
                });
            }

            InputEvent::MouseMoved { x, y } => {
                self.mouse_position = Some((x, y));
                self.events.push(RawInputEvent::PointerMoved { x, y });
            }

            InputEvent::MousePressed { button, x, y } => {
                self.mouse_position = Some((x, y));
                self.pressed_buttons.insert(button);

                self.events.push(RawInputEvent::PointerButton {
                    x,
                    y,
                    button,
                    pressed: true,
                    modifiers: self.modifiers,
                });
            }

            InputEvent::MouseReleased { button, x, y } => {
                self.mouse_position = Some((x, y));
                self.pressed_buttons.remove(&button);

                self.events.push(RawInputEvent::PointerButton {
                    x,
                    y,
                    button,
                    pressed: false,
                    modifiers: self.modifiers,
                });
            }

            InputEvent::MouseWheel { delta_x, delta_y } => {
                self.events.push(RawInputEvent::Scroll { delta_x, delta_y });
            }

            InputEvent::WindowResize { width, height } => {
                self.screen_size = (width, height);
            }

            InputEvent::WindowFocus { focused } => {
                self.focused = focused;
                self.events.push(RawInputEvent::WindowFocused { focused });

                // Clear pressed state on focus loss
                if !focused {
                    self.pressed_buttons.clear();
                    self.pressed_keys.clear();
                    self.modifiers = Modifiers::NONE;
                }
            }

            InputEvent::TouchStart { id, x, y } => {
                self.events.push(RawInputEvent::Touch {
                    id,
                    phase: TouchPhase::Start,
                    x,
                    y,
                });
            }

            InputEvent::TouchMove { id, x, y } => {
                self.events.push(RawInputEvent::Touch {
                    id,
                    phase: TouchPhase::Move,
                    x,
                    y,
                });
            }

            InputEvent::TouchEnd { id, x, y } => {
                self.events.push(RawInputEvent::Touch {
                    id,
                    phase: TouchPhase::End,
                    x,
                    y,
                });
            }

            InputEvent::TouchCancel { id } => {
                self.events.push(RawInputEvent::Touch {
                    id,
                    phase: TouchPhase::Cancel,
                    x: 0.0,
                    y: 0.0,
                });
            }

            InputEvent::Copy => {
                self.events.push(RawInputEvent::Copy);
            }

            InputEvent::Cut => {
                self.events.push(RawInputEvent::Cut);
            }

            InputEvent::Paste { text } => {
                self.events.push(RawInputEvent::Paste { text });
            }

            InputEvent::ImeCompositionStart |
            InputEvent::ImeCompositionUpdate { .. } |
            InputEvent::ImeCompositionEnd { .. } => {
                // IME events are handled via CharTyped in this simple implementation
                if let InputEvent::ImeCompositionEnd { text } = event {
                    self.events.push(RawInputEvent::Text { text });
                }
            }

            InputEvent::FileDrop { .. } |
            InputEvent::FileHover { .. } |
            InputEvent::FileHoverCancel => {
                // File events are handled separately by the application
            }
        }
    }

    /// Push multiple events at once.
    pub fn push_events(&mut self, events: impl IntoIterator<Item = InputEvent>) {
        for event in events {
            self.push_event(event);
        }
    }

    /// Take the accumulated RawInput and reset for the next frame.
    pub fn take_raw_input(&mut self) -> RawInput {
        let events = std::mem::take(&mut self.events);

        RawInput {
            events,
            screen_rect: Some(Rect::from_size(
                self.screen_size.0 as f32,
                self.screen_size.1 as f32,
            )),
            pixels_per_point: Some(self.pixels_per_point),
            max_texture_side: Some(8192),
            predicted_dt: self.dt,
            modifiers: self.modifiers,
            focused: self.focused,
        }
    }

    /// Clear all accumulated events without taking them.
    pub fn clear_events(&mut self) {
        self.events.clear();
    }

    /// Get the number of pending events.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Check if there are any pending events.
    pub fn has_events(&self) -> bool {
        !self.events.is_empty()
    }

    /// Reset all state (useful when window loses focus).
    pub fn reset(&mut self) {
        self.mouse_position = None;
        self.pressed_buttons.clear();
        self.pressed_keys.clear();
        self.modifiers = Modifiers::NONE;
        self.events.clear();
        self.last_key_event = None;
    }

    /// Update modifiers based on key press/release.
    fn update_modifiers_from_key(&mut self, key: &KeyCode, pressed: bool) {
        match key {
            KeyCode::LeftCtrl | KeyCode::RightCtrl => self.modifiers.ctrl = pressed,
            KeyCode::LeftShift | KeyCode::RightShift => self.modifiers.shift = pressed,
            KeyCode::LeftAlt | KeyCode::RightAlt => self.modifiers.alt = pressed,
            KeyCode::LeftMeta | KeyCode::RightMeta => self.modifiers.meta = pressed,
            _ => {}
        }
    }
}

impl Default for InputMapper {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Bridge Protocol Integration
// ---------------------------------------------------------------------------

/// Namespace for input events in the bridge protocol.
pub mod input_ns {
    use super::*;
    use serde::{Deserialize, Serialize};

    /// Batch of input events from Python.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct InputBatch {
        /// Frame number these events are for.
        pub frame: u64,
        /// Input events.
        pub events: Vec<InputEvent>,
        /// Timestamp (milliseconds since epoch).
        pub timestamp_ms: u64,
    }

    impl InputBatch {
        /// Create a new input batch.
        pub fn new(frame: u64, events: Vec<InputEvent>) -> Self {
            Self {
                frame,
                events,
                timestamp_ms: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0),
            }
        }

        /// Check if the batch is empty.
        pub fn is_empty(&self) -> bool {
            self.events.is_empty()
        }

        /// Get the number of events.
        pub fn len(&self) -> usize {
            self.events.len()
        }

        /// Serialize to JSON.
        pub fn to_json(&self) -> Result<Vec<u8>, serde_json::Error> {
            serde_json::to_vec(self)
        }

        /// Deserialize from JSON.
        pub fn from_json(data: &[u8]) -> Result<Self, serde_json::Error> {
            serde_json::from_slice(data)
        }
    }

    /// Request to update input state.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct UpdateInputRequest {
        /// Input batch.
        pub batch: InputBatch,
    }

    /// Response to input update.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct UpdateInputResponse {
        /// Whether input was processed.
        pub processed: bool,
        /// Number of events processed.
        pub event_count: usize,
        /// Any error message.
        pub error: Option<String>,
    }

    /// Request to get current input state.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GetInputStateRequest {
        /// Whether to include modifier state.
        pub include_modifiers: bool,
        /// Whether to include mouse position.
        pub include_mouse: bool,
        /// Whether to include pressed keys.
        pub include_keys: bool,
    }

    /// Response with current input state.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GetInputStateResponse {
        /// Current modifier state.
        pub modifiers: Option<Modifiers>,
        /// Current mouse position.
        pub mouse_position: Option<(f32, f32)>,
        /// Currently pressed keys.
        pub pressed_keys: Option<Vec<KeyCode>>,
        /// Currently pressed mouse buttons.
        pub pressed_buttons: Option<Vec<MouseButton>>,
        /// Whether window is focused.
        pub focused: bool,
        /// Current screen size.
        pub screen_size: (u32, u32),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // KeyCode Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_keycode_is_letter() {
        assert!(KeyCode::A.is_letter());
        assert!(KeyCode::Z.is_letter());
        assert!(!KeyCode::Num0.is_letter());
        assert!(!KeyCode::Space.is_letter());
    }

    #[test]
    fn test_keycode_is_number() {
        assert!(KeyCode::Num0.is_number());
        assert!(KeyCode::Num9.is_number());
        assert!(!KeyCode::A.is_number());
        assert!(!KeyCode::Numpad0.is_number());
    }

    #[test]
    fn test_keycode_is_function_key() {
        assert!(KeyCode::F1.is_function_key());
        assert!(KeyCode::F12.is_function_key());
        assert!(!KeyCode::A.is_function_key());
    }

    #[test]
    fn test_keycode_is_numpad() {
        assert!(KeyCode::Numpad0.is_numpad());
        assert!(KeyCode::NumpadAdd.is_numpad());
        assert!(!KeyCode::Num0.is_numpad());
    }

    #[test]
    fn test_keycode_is_modifier() {
        assert!(KeyCode::LeftCtrl.is_modifier());
        assert!(KeyCode::RightShift.is_modifier());
        assert!(!KeyCode::A.is_modifier());
    }

    #[test]
    fn test_keycode_is_arrow() {
        assert!(KeyCode::ArrowUp.is_arrow());
        assert!(KeyCode::ArrowDown.is_arrow());
        assert!(KeyCode::ArrowLeft.is_arrow());
        assert!(KeyCode::ArrowRight.is_arrow());
        assert!(!KeyCode::A.is_arrow());
    }

    #[test]
    fn test_keycode_to_char_letters() {
        assert_eq!(KeyCode::A.to_char(false), Some('a'));
        assert_eq!(KeyCode::A.to_char(true), Some('A'));
        assert_eq!(KeyCode::Z.to_char(false), Some('z'));
        assert_eq!(KeyCode::Z.to_char(true), Some('Z'));
    }

    #[test]
    fn test_keycode_to_char_numbers() {
        assert_eq!(KeyCode::Num1.to_char(false), Some('1'));
        assert_eq!(KeyCode::Num1.to_char(true), Some('!'));
        assert_eq!(KeyCode::Num0.to_char(true), Some(')'));
    }

    #[test]
    fn test_keycode_to_char_punctuation() {
        assert_eq!(KeyCode::Minus.to_char(false), Some('-'));
        assert_eq!(KeyCode::Minus.to_char(true), Some('_'));
        assert_eq!(KeyCode::Equals.to_char(false), Some('='));
        assert_eq!(KeyCode::Equals.to_char(true), Some('+'));
    }

    #[test]
    fn test_keycode_to_char_none() {
        assert_eq!(KeyCode::F1.to_char(false), None);
        assert_eq!(KeyCode::Enter.to_char(false), None);
        assert_eq!(KeyCode::Escape.to_char(false), None);
    }

    #[test]
    fn test_keycode_from_name() {
        assert_eq!(KeyCode::from_name("a"), Some(KeyCode::A));
        assert_eq!(KeyCode::from_name("A"), Some(KeyCode::A));
        assert_eq!(KeyCode::from_name("enter"), Some(KeyCode::Enter));
        assert_eq!(KeyCode::from_name("return"), Some(KeyCode::Enter));
        assert_eq!(KeyCode::from_name("escape"), Some(KeyCode::Escape));
        assert_eq!(KeyCode::from_name("esc"), Some(KeyCode::Escape));
        assert_eq!(KeyCode::from_name("f1"), Some(KeyCode::F1));
        assert_eq!(KeyCode::from_name("unknown_key"), None);
    }

    #[test]
    fn test_keycode_from_name_arrows() {
        assert_eq!(KeyCode::from_name("up"), Some(KeyCode::ArrowUp));
        assert_eq!(KeyCode::from_name("arrowup"), Some(KeyCode::ArrowUp));
        assert_eq!(KeyCode::from_name("down"), Some(KeyCode::ArrowDown));
        assert_eq!(KeyCode::from_name("left"), Some(KeyCode::ArrowLeft));
        assert_eq!(KeyCode::from_name("right"), Some(KeyCode::ArrowRight));
    }

    #[test]
    fn test_keycode_serialization() {
        let key = KeyCode::A;
        let json = serde_json::to_string(&key).unwrap();
        assert_eq!(json, "\"a\"");
        let parsed: KeyCode = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, KeyCode::A);
    }

    // -------------------------------------------------------------------------
    // MouseButton Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_mouse_button_from_name() {
        assert_eq!(MouseButton::from_name("left"), Some(MouseButton::Left));
        assert_eq!(MouseButton::from_name("primary"), Some(MouseButton::Left));
        assert_eq!(MouseButton::from_name("right"), Some(MouseButton::Right));
        assert_eq!(MouseButton::from_name("middle"), Some(MouseButton::Middle));
        assert_eq!(MouseButton::from_name("extra1"), Some(MouseButton::Extra1));
        assert_eq!(MouseButton::from_name("back"), Some(MouseButton::Extra1));
        assert_eq!(MouseButton::from_name("extra2"), Some(MouseButton::Extra2));
        assert_eq!(MouseButton::from_name("forward"), Some(MouseButton::Extra2));
    }

    #[test]
    fn test_mouse_button_index() {
        assert_eq!(MouseButton::Left.index(), 0);
        assert_eq!(MouseButton::Right.index(), 1);
        assert_eq!(MouseButton::Middle.index(), 2);
        assert_eq!(MouseButton::Extra1.index(), 3);
        assert_eq!(MouseButton::Extra2.index(), 4);
    }

    #[test]
    fn test_mouse_button_serialization() {
        let button = MouseButton::Left;
        let json = serde_json::to_string(&button).unwrap();
        assert_eq!(json, "\"left\"");
        let parsed: MouseButton = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, MouseButton::Left);
    }

    // -------------------------------------------------------------------------
    // Modifiers Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_modifiers_none() {
        let mods = Modifiers::NONE;
        assert!(!mods.ctrl);
        assert!(!mods.shift);
        assert!(!mods.alt);
        assert!(!mods.meta);
        assert!(mods.none());
        assert!(!mods.any());
    }

    #[test]
    fn test_modifiers_ctrl() {
        let mods = Modifiers::CTRL;
        assert!(mods.ctrl);
        assert!(!mods.shift);
        assert!(mods.any());
        assert!(!mods.none());
    }

    #[test]
    fn test_modifiers_union() {
        let mods = Modifiers::CTRL.union(&Modifiers::SHIFT);
        assert!(mods.ctrl);
        assert!(mods.shift);
        assert!(!mods.alt);
        assert!(!mods.meta);
    }

    #[test]
    fn test_modifiers_to_bits() {
        assert_eq!(Modifiers::NONE.to_bits(), 0);
        assert_eq!(Modifiers::CTRL.to_bits(), 1);
        assert_eq!(Modifiers::SHIFT.to_bits(), 2);
        assert_eq!(Modifiers::ALT.to_bits(), 4);
        assert_eq!(Modifiers::META.to_bits(), 8);

        let mods = Modifiers::new(true, true, false, false);
        assert_eq!(mods.to_bits(), 3);
    }

    #[test]
    fn test_modifiers_from_bits() {
        assert_eq!(Modifiers::from_bits(0), Modifiers::NONE);
        assert_eq!(Modifiers::from_bits(1), Modifiers::CTRL);
        assert_eq!(Modifiers::from_bits(2), Modifiers::SHIFT);
        assert_eq!(Modifiers::from_bits(15), Modifiers::new(true, true, true, true));
    }

    #[test]
    fn test_modifiers_serialization() {
        let mods = Modifiers::new(true, false, true, false);
        let json = serde_json::to_string(&mods).unwrap();
        let parsed: Modifiers = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, mods);
    }

    // -------------------------------------------------------------------------
    // InputEvent Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_input_event_key_pressed() {
        let event = InputEvent::KeyPressed {
            key: KeyCode::A,
            modifiers: Modifiers::CTRL,
        };
        assert!(event.is_keyboard());
        assert!(!event.is_mouse());
        assert!(!event.is_window());
    }

    #[test]
    fn test_input_event_mouse_moved() {
        let event = InputEvent::MouseMoved { x: 100.0, y: 200.0 };
        assert!(!event.is_keyboard());
        assert!(event.is_mouse());
        assert_eq!(event.position(), Some((100.0, 200.0)));
    }

    #[test]
    fn test_input_event_mouse_pressed() {
        let event = InputEvent::MousePressed {
            button: MouseButton::Left,
            x: 50.0,
            y: 75.0,
        };
        assert!(event.is_mouse());
        assert_eq!(event.position(), Some((50.0, 75.0)));
    }

    #[test]
    fn test_input_event_window_resize() {
        let event = InputEvent::WindowResize { width: 1920, height: 1080 };
        assert!(event.is_window());
        assert!(!event.is_mouse());
        assert!(!event.is_keyboard());
    }

    #[test]
    fn test_input_event_touch() {
        let event = InputEvent::TouchStart { id: 1, x: 100.0, y: 100.0 };
        assert!(event.is_touch());
        assert!(!event.is_mouse());
    }

    #[test]
    fn test_input_event_serialization_key() {
        let event = InputEvent::KeyPressed {
            key: KeyCode::Enter,
            modifiers: Modifiers::SHIFT,
        };
        let json = serde_json::to_string(&event).unwrap();
        let parsed: InputEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, event);
    }

    #[test]
    fn test_input_event_serialization_mouse() {
        let event = InputEvent::MousePressed {
            button: MouseButton::Right,
            x: 100.0,
            y: 200.0,
        };
        let json = serde_json::to_string(&event).unwrap();
        let parsed: InputEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, event);
    }

    #[test]
    fn test_input_event_serialization_char() {
        let event = InputEvent::CharTyped { character: 'x' };
        let json = serde_json::to_string(&event).unwrap();
        let parsed: InputEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, event);
    }

    #[test]
    fn test_input_event_serialization_paste() {
        let event = InputEvent::Paste { text: "Hello".to_string() };
        let json = serde_json::to_string(&event).unwrap();
        let parsed: InputEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, event);
    }

    // -------------------------------------------------------------------------
    // InputMapper Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_input_mapper_new() {
        let mapper = InputMapper::new();
        assert_eq!(mapper.screen_size(), (1920, 1080));
        assert_eq!(mapper.pixels_per_point(), 1.0);
        assert!(mapper.mouse_position().is_none());
        assert!(mapper.is_focused());
    }

    #[test]
    fn test_input_mapper_screen_size() {
        let mut mapper = InputMapper::with_screen_size(800, 600);
        assert_eq!(mapper.screen_size(), (800, 600));

        mapper.set_screen_size(1024, 768);
        assert_eq!(mapper.screen_size(), (1024, 768));
    }

    #[test]
    fn test_input_mapper_mouse_move() {
        let mut mapper = InputMapper::new();

        mapper.push_event(InputEvent::MouseMoved { x: 100.0, y: 200.0 });

        assert_eq!(mapper.mouse_position(), Some((100.0, 200.0)));
        assert!(mapper.has_events());
        assert_eq!(mapper.event_count(), 1);
    }

    #[test]
    fn test_input_mapper_mouse_button() {
        let mut mapper = InputMapper::new();

        mapper.push_event(InputEvent::MousePressed {
            button: MouseButton::Left,
            x: 50.0,
            y: 50.0,
        });

        assert!(mapper.is_button_pressed(MouseButton::Left));
        assert!(!mapper.is_button_pressed(MouseButton::Right));

        mapper.push_event(InputEvent::MouseReleased {
            button: MouseButton::Left,
            x: 50.0,
            y: 50.0,
        });

        assert!(!mapper.is_button_pressed(MouseButton::Left));
    }

    #[test]
    fn test_input_mapper_key_press() {
        let mut mapper = InputMapper::new();

        mapper.push_event(InputEvent::KeyPressed {
            key: KeyCode::A,
            modifiers: Modifiers::NONE,
        });

        assert!(mapper.is_key_pressed(KeyCode::A));
        assert!(!mapper.is_key_pressed(KeyCode::B));

        mapper.push_event(InputEvent::KeyReleased {
            key: KeyCode::A,
            modifiers: Modifiers::NONE,
        });

        assert!(!mapper.is_key_pressed(KeyCode::A));
    }

    #[test]
    fn test_input_mapper_modifiers() {
        let mut mapper = InputMapper::new();

        mapper.push_event(InputEvent::KeyPressed {
            key: KeyCode::A,
            modifiers: Modifiers::CTRL,
        });

        assert_eq!(mapper.modifiers(), Modifiers::CTRL);
    }

    #[test]
    fn test_input_mapper_modifiers_from_key() {
        let mut mapper = InputMapper::new();

        mapper.push_event(InputEvent::KeyPressed {
            key: KeyCode::LeftCtrl,
            modifiers: Modifiers::NONE,
        });

        // Modifiers should be updated from the key press
        assert!(mapper.modifiers().ctrl);
    }

    #[test]
    fn test_input_mapper_window_focus() {
        let mut mapper = InputMapper::new();

        // Simulate pressed keys and buttons
        mapper.push_event(InputEvent::KeyPressed {
            key: KeyCode::A,
            modifiers: Modifiers::CTRL,
        });
        mapper.push_event(InputEvent::MousePressed {
            button: MouseButton::Left,
            x: 0.0,
            y: 0.0,
        });

        // Lose focus
        mapper.push_event(InputEvent::WindowFocus { focused: false });

        // State should be cleared
        assert!(!mapper.is_focused());
        assert!(!mapper.is_key_pressed(KeyCode::A));
        assert!(!mapper.is_button_pressed(MouseButton::Left));
        assert_eq!(mapper.modifiers(), Modifiers::NONE);
    }

    #[test]
    fn test_input_mapper_take_raw_input() {
        let mut mapper = InputMapper::new();

        mapper.push_event(InputEvent::MouseMoved { x: 100.0, y: 100.0 });
        mapper.push_event(InputEvent::KeyPressed {
            key: KeyCode::A,
            modifiers: Modifiers::CTRL,
        });

        let raw_input = mapper.take_raw_input();

        assert_eq!(raw_input.events.len(), 2);
        assert!(raw_input.screen_rect.is_some());
        assert!(raw_input.focused);

        // Events should be cleared
        assert!(!mapper.has_events());
    }

    #[test]
    fn test_input_mapper_clear_events() {
        let mut mapper = InputMapper::new();

        mapper.push_event(InputEvent::MouseMoved { x: 100.0, y: 100.0 });
        assert!(mapper.has_events());

        mapper.clear_events();
        assert!(!mapper.has_events());
    }

    #[test]
    fn test_input_mapper_reset() {
        let mut mapper = InputMapper::new();

        mapper.push_event(InputEvent::MouseMoved { x: 100.0, y: 100.0 });
        mapper.push_event(InputEvent::KeyPressed {
            key: KeyCode::A,
            modifiers: Modifiers::CTRL,
        });

        mapper.reset();

        assert!(mapper.mouse_position().is_none());
        assert!(!mapper.is_key_pressed(KeyCode::A));
        assert_eq!(mapper.modifiers(), Modifiers::NONE);
        assert!(!mapper.has_events());
    }

    #[test]
    fn test_input_mapper_push_events() {
        let mut mapper = InputMapper::new();

        let events = vec![
            InputEvent::MouseMoved { x: 10.0, y: 20.0 },
            InputEvent::MouseMoved { x: 30.0, y: 40.0 },
            InputEvent::MouseMoved { x: 50.0, y: 60.0 },
        ];

        mapper.push_events(events);

        assert_eq!(mapper.event_count(), 3);
        assert_eq!(mapper.mouse_position(), Some((50.0, 60.0)));
    }

    #[test]
    fn test_input_mapper_touch_events() {
        let mut mapper = InputMapper::new();

        mapper.push_event(InputEvent::TouchStart { id: 1, x: 100.0, y: 100.0 });
        mapper.push_event(InputEvent::TouchMove { id: 1, x: 150.0, y: 150.0 });
        mapper.push_event(InputEvent::TouchEnd { id: 1, x: 200.0, y: 200.0 });

        let raw_input = mapper.take_raw_input();
        assert_eq!(raw_input.events.len(), 3);
    }

    #[test]
    fn test_input_mapper_clipboard_events() {
        let mut mapper = InputMapper::new();

        mapper.push_event(InputEvent::Copy);
        mapper.push_event(InputEvent::Cut);
        mapper.push_event(InputEvent::Paste { text: "test".to_string() });

        let raw_input = mapper.take_raw_input();
        assert_eq!(raw_input.events.len(), 3);

        assert!(matches!(raw_input.events[0], RawInputEvent::Copy));
        assert!(matches!(raw_input.events[1], RawInputEvent::Cut));
        assert!(matches!(raw_input.events[2], RawInputEvent::Paste { .. }));
    }

    #[test]
    fn test_input_mapper_window_resize() {
        let mut mapper = InputMapper::new();

        mapper.push_event(InputEvent::WindowResize { width: 800, height: 600 });

        assert_eq!(mapper.screen_size(), (800, 600));

        let raw_input = mapper.take_raw_input();
        let rect = raw_input.screen_rect.unwrap();
        assert_eq!(rect.max_x, 800.0);
        assert_eq!(rect.max_y, 600.0);
    }

    #[test]
    fn test_input_mapper_key_repeat() {
        let mut mapper = InputMapper::new();

        // First press is not a repeat
        mapper.push_event(InputEvent::KeyPressed {
            key: KeyCode::A,
            modifiers: Modifiers::NONE,
        });

        // Second press (while still held) is a repeat
        mapper.push_event(InputEvent::KeyPressed {
            key: KeyCode::A,
            modifiers: Modifiers::NONE,
        });

        let raw_input = mapper.take_raw_input();

        // Check that the second event has repeat=true
        if let RawInputEvent::Key { repeat, .. } = &raw_input.events[0] {
            assert!(!repeat);
        }
        if let RawInputEvent::Key { repeat, .. } = &raw_input.events[1] {
            assert!(*repeat);
        }
    }

    // -------------------------------------------------------------------------
    // RawInput Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_raw_input_default() {
        let input = RawInput::default();
        assert!(input.events.is_empty());
        assert!(input.screen_rect.is_none());
    }

    #[test]
    fn test_raw_input_serialization() {
        let mut input = RawInput::default();
        input.events.push(RawInputEvent::PointerMoved { x: 100.0, y: 200.0 });
        input.screen_rect = Some(Rect::from_size(800.0, 600.0));
        input.focused = true;

        let json = serde_json::to_string(&input).unwrap();
        let parsed: RawInput = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.events.len(), 1);
        assert!(parsed.screen_rect.is_some());
        assert!(parsed.focused);
    }

    // -------------------------------------------------------------------------
    // Rect Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_rect_new() {
        let rect = Rect::new(10.0, 20.0, 110.0, 120.0);
        assert_eq!(rect.width(), 100.0);
        assert_eq!(rect.height(), 100.0);
    }

    #[test]
    fn test_rect_from_size() {
        let rect = Rect::from_size(800.0, 600.0);
        assert_eq!(rect.min_x, 0.0);
        assert_eq!(rect.min_y, 0.0);
        assert_eq!(rect.max_x, 800.0);
        assert_eq!(rect.max_y, 600.0);
    }

    #[test]
    fn test_rect_contains() {
        let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
        assert!(rect.contains(50.0, 50.0));
        assert!(rect.contains(0.0, 0.0));
        assert!(rect.contains(100.0, 100.0));
        assert!(!rect.contains(101.0, 50.0));
        assert!(!rect.contains(-1.0, 50.0));
    }

    // -------------------------------------------------------------------------
    // RawInputEvent Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_raw_input_event_pointer_moved() {
        let event = RawInputEvent::PointerMoved { x: 100.0, y: 200.0 };
        let json = serde_json::to_string(&event).unwrap();
        let parsed: RawInputEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, event);
    }

    #[test]
    fn test_raw_input_event_pointer_button() {
        let event = RawInputEvent::PointerButton {
            x: 50.0,
            y: 50.0,
            button: MouseButton::Left,
            pressed: true,
            modifiers: Modifiers::SHIFT,
        };
        let json = serde_json::to_string(&event).unwrap();
        let parsed: RawInputEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, event);
    }

    #[test]
    fn test_raw_input_event_scroll() {
        let event = RawInputEvent::Scroll { delta_x: 0.0, delta_y: -100.0 };
        let json = serde_json::to_string(&event).unwrap();
        let parsed: RawInputEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, event);
    }

    #[test]
    fn test_raw_input_event_key() {
        let event = RawInputEvent::Key {
            key: KeyCode::Enter,
            pressed: true,
            repeat: false,
            modifiers: Modifiers::CTRL,
        };
        let json = serde_json::to_string(&event).unwrap();
        let parsed: RawInputEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, event);
    }

    #[test]
    fn test_raw_input_event_text() {
        let event = RawInputEvent::Text { text: "Hello World".to_string() };
        let json = serde_json::to_string(&event).unwrap();
        let parsed: RawInputEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, event);
    }

    #[test]
    fn test_raw_input_event_touch() {
        let event = RawInputEvent::Touch {
            id: 42,
            phase: TouchPhase::Start,
            x: 100.0,
            y: 200.0,
        };
        let json = serde_json::to_string(&event).unwrap();
        let parsed: RawInputEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, event);
    }

    // -------------------------------------------------------------------------
    // input_ns Protocol Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_input_batch_new() {
        let events = vec![
            InputEvent::MouseMoved { x: 100.0, y: 100.0 },
            InputEvent::KeyPressed { key: KeyCode::A, modifiers: Modifiers::NONE },
        ];
        let batch = input_ns::InputBatch::new(1, events);

        assert_eq!(batch.frame, 1);
        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_input_batch_serialization() {
        let events = vec![
            InputEvent::MouseMoved { x: 100.0, y: 100.0 },
        ];
        let batch = input_ns::InputBatch::new(42, events);

        let json = batch.to_json().unwrap();
        let parsed = input_ns::InputBatch::from_json(&json).unwrap();

        assert_eq!(parsed.frame, 42);
        assert_eq!(parsed.len(), 1);
    }

    #[test]
    fn test_update_input_request() {
        let batch = input_ns::InputBatch::new(1, vec![]);
        let request = input_ns::UpdateInputRequest { batch };

        let json = serde_json::to_string(&request).unwrap();
        let parsed: input_ns::UpdateInputRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.batch.frame, 1);
    }

    #[test]
    fn test_update_input_response() {
        let response = input_ns::UpdateInputResponse {
            processed: true,
            event_count: 5,
            error: None,
        };

        let json = serde_json::to_string(&response).unwrap();
        let parsed: input_ns::UpdateInputResponse = serde_json::from_str(&json).unwrap();
        assert!(parsed.processed);
        assert_eq!(parsed.event_count, 5);
    }

    #[test]
    fn test_get_input_state_request() {
        let request = input_ns::GetInputStateRequest {
            include_modifiers: true,
            include_mouse: true,
            include_keys: false,
        };

        let json = serde_json::to_string(&request).unwrap();
        let parsed: input_ns::GetInputStateRequest = serde_json::from_str(&json).unwrap();
        assert!(parsed.include_modifiers);
        assert!(parsed.include_mouse);
        assert!(!parsed.include_keys);
    }

    #[test]
    fn test_get_input_state_response() {
        let response = input_ns::GetInputStateResponse {
            modifiers: Some(Modifiers::CTRL),
            mouse_position: Some((100.0, 200.0)),
            pressed_keys: Some(vec![KeyCode::A, KeyCode::B]),
            pressed_buttons: Some(vec![MouseButton::Left]),
            focused: true,
            screen_size: (1920, 1080),
        };

        let json = serde_json::to_string(&response).unwrap();
        let parsed: input_ns::GetInputStateResponse = serde_json::from_str(&json).unwrap();
        assert!(parsed.focused);
        assert_eq!(parsed.screen_size, (1920, 1080));
    }

    // -------------------------------------------------------------------------
    // Edge Case Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_keycode_numpad_chars() {
        assert_eq!(KeyCode::Numpad0.to_char(false), Some('0'));
        assert_eq!(KeyCode::NumpadAdd.to_char(false), Some('+'));
        assert_eq!(KeyCode::NumpadDecimal.to_char(false), Some('.'));
    }

    #[test]
    fn test_modifiers_copy_shortcut() {
        let ctrl = Modifiers::CTRL;
        let meta = Modifiers::META;
        let ctrl_shift = Modifiers::new(true, true, false, false);

        assert!(ctrl.is_copy_shortcut());
        assert!(meta.is_copy_shortcut());
        assert!(!ctrl_shift.is_copy_shortcut());
    }

    #[test]
    fn test_multiple_mouse_buttons() {
        let mut mapper = InputMapper::new();

        mapper.push_event(InputEvent::MousePressed { button: MouseButton::Left, x: 0.0, y: 0.0 });
        mapper.push_event(InputEvent::MousePressed { button: MouseButton::Right, x: 0.0, y: 0.0 });

        assert!(mapper.is_button_pressed(MouseButton::Left));
        assert!(mapper.is_button_pressed(MouseButton::Right));

        mapper.push_event(InputEvent::MouseReleased { button: MouseButton::Left, x: 0.0, y: 0.0 });

        assert!(!mapper.is_button_pressed(MouseButton::Left));
        assert!(mapper.is_button_pressed(MouseButton::Right));
    }

    #[test]
    fn test_ime_composition_end() {
        let mut mapper = InputMapper::new();

        mapper.push_event(InputEvent::ImeCompositionEnd { text: "test".to_string() });

        let raw_input = mapper.take_raw_input();
        assert_eq!(raw_input.events.len(), 1);

        if let RawInputEvent::Text { text } = &raw_input.events[0] {
            assert_eq!(text, "test");
        } else {
            panic!("Expected Text event");
        }
    }

    #[test]
    fn test_scroll_event() {
        let mut mapper = InputMapper::new();

        mapper.push_event(InputEvent::MouseWheel { delta_x: 0.0, delta_y: -120.0 });

        let raw_input = mapper.take_raw_input();
        assert_eq!(raw_input.events.len(), 1);

        if let RawInputEvent::Scroll { delta_x, delta_y } = raw_input.events[0] {
            assert_eq!(delta_x, 0.0);
            assert_eq!(delta_y, -120.0);
        } else {
            panic!("Expected Scroll event");
        }
    }
}
