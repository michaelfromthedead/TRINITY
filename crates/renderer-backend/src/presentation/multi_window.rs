//! Multi-window rendering support for simultaneous render targets.
//!
//! This module provides types and utilities for managing multiple rendering windows
//! in TRINITY. It enables applications to render to multiple displays or viewports
//! simultaneously.
//!
//! # Architecture
//!
//! ```text
//! MultiWindowManager
//!     |-- windows: HashMap<WindowId, ManagedWindow>
//!     |-- primary_window: Option<WindowId>
//!     `-- next_id: u64
//!
//! ManagedWindow
//!     |-- id: WindowId
//!     |-- config: WindowConfig
//!     |-- state: WindowState
//!     `-- frame_count: u64
//!
//! WindowEvent
//!     |-- Created, Resized, Focused
//!     |-- Unfocused, Minimized, Restored
//!     `-- Closed
//! ```
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::presentation::multi_window::*;
//!
//! let mut manager = MultiWindowManager::new();
//!
//! // Create primary window
//! let primary_config = WindowConfig::new("Main Window", 1920, 1080)
//!     .with_vsync(true)
//!     .as_primary();
//! let primary_id = manager.create_window(primary_config);
//!
//! // Create secondary window
//! let secondary_config = WindowConfig::new("Tools", 800, 600);
//! let secondary_id = manager.create_window(secondary_config);
//!
//! // Handle events
//! manager.handle_event(WindowEvent::Focused(primary_id));
//!
//! // Query windows
//! let active = manager.active_windows();
//! let visible = manager.visible_windows();
//! ```

use std::collections::HashMap;
use std::fmt;

// ============================================================================
// Window Identifier
// ============================================================================

/// Unique identifier for a window in the multi-window system.
///
/// `WindowId` provides type-safe window identification for tracking multiple
/// render targets. Each window has a unique ID that remains stable for the
/// window's lifetime.
///
/// # Example
///
/// ```ignore
/// let id = WindowId::new(1);
/// assert_eq!(id.as_u64(), 1);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WindowId(u64);

impl WindowId {
    /// Create a new window ID with the specified value.
    ///
    /// # Arguments
    ///
    /// * `id` - The unique identifier value.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let id = WindowId::new(42);
    /// assert_eq!(id.as_u64(), 42);
    /// ```
    #[inline]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Get the raw u64 value of this window ID.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let id = WindowId::new(123);
    /// assert_eq!(id.as_u64(), 123);
    /// ```
    #[inline]
    pub const fn as_u64(&self) -> u64 {
        self.0
    }

    /// Create a window ID representing the primary window (ID 0).
    ///
    /// By convention, WindowId(0) is reserved for the primary/main window.
    #[inline]
    pub const fn primary() -> Self {
        Self(0)
    }

    /// Check if this is the primary window ID.
    #[inline]
    pub const fn is_primary(&self) -> bool {
        self.0 == 0
    }
}

impl Default for WindowId {
    fn default() -> Self {
        Self(0)
    }
}

impl fmt::Display for WindowId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Window({})", self.0)
    }
}

impl From<u64> for WindowId {
    fn from(id: u64) -> Self {
        Self::new(id)
    }
}

impl From<WindowId> for u64 {
    fn from(id: WindowId) -> Self {
        id.0
    }
}

// ============================================================================
// Window Configuration
// ============================================================================

/// Configuration for a window in the multi-window system.
///
/// `WindowConfig` contains all the parameters needed to create and configure
/// a rendering window.
///
/// # Example
///
/// ```ignore
/// let config = WindowConfig::new("Main Window", 1920, 1080)
///     .with_vsync(true)
///     .as_primary();
/// ```
#[derive(Clone, Debug)]
pub struct WindowConfig {
    /// Unique identifier for this window.
    pub id: WindowId,
    /// Human-readable window title.
    pub title: String,
    /// Window width in pixels.
    pub width: u32,
    /// Window height in pixels.
    pub height: u32,
    /// Whether VSync is enabled for this window.
    pub vsync: bool,
    /// Whether this is the primary/main window.
    pub primary: bool,
}

impl WindowConfig {
    /// Create a new window configuration.
    ///
    /// # Arguments
    ///
    /// * `title` - Human-readable window title.
    /// * `width` - Window width in pixels.
    /// * `height` - Window height in pixels.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = WindowConfig::new("My Window", 800, 600);
    /// assert_eq!(config.width, 800);
    /// assert_eq!(config.height, 600);
    /// assert!(!config.vsync);
    /// assert!(!config.primary);
    /// ```
    pub fn new(title: &str, width: u32, height: u32) -> Self {
        Self {
            id: WindowId::new(0), // Will be assigned by manager
            title: title.to_string(),
            width,
            height,
            vsync: false,
            primary: false,
        }
    }

    /// Enable or disable VSync for this window.
    ///
    /// # Arguments
    ///
    /// * `vsync` - Whether to enable VSync.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = WindowConfig::new("Window", 800, 600)
    ///     .with_vsync(true);
    /// assert!(config.vsync);
    /// ```
    #[must_use]
    pub fn with_vsync(mut self, vsync: bool) -> Self {
        self.vsync = vsync;
        self
    }

    /// Mark this window as the primary window.
    ///
    /// The primary window typically receives input focus by default and
    /// may have special handling in the application.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = WindowConfig::new("Main", 1920, 1080)
    ///     .as_primary();
    /// assert!(config.primary);
    /// ```
    #[must_use]
    pub fn as_primary(mut self) -> Self {
        self.primary = true;
        self
    }

    /// Get the window dimensions as a tuple.
    #[inline]
    pub const fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Get the aspect ratio (width / height).
    pub fn aspect_ratio(&self) -> f32 {
        if self.height > 0 {
            self.width as f32 / self.height as f32
        } else {
            1.0
        }
    }

    /// Calculate the total pixel count.
    #[inline]
    pub const fn pixel_count(&self) -> u64 {
        self.width as u64 * self.height as u64
    }
}

impl fmt::Display for WindowConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} [{}x{}, vsync={}, primary={}]",
            self.title, self.width, self.height, self.vsync, self.primary
        )
    }
}

// ============================================================================
// Window State
// ============================================================================

/// State of a managed window.
///
/// `WindowState` represents the current lifecycle state of a window in the
/// multi-window system.
///
/// # State Transitions
///
/// ```text
/// Active <-> Minimized
/// Active <-> Hidden
/// Active -> Closed
/// Minimized -> Closed
/// Hidden -> Closed
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WindowState {
    /// Window is active and renderable.
    Active,
    /// Window is minimized (iconified).
    Minimized,
    /// Window is hidden but not minimized.
    Hidden,
    /// Window has been closed and is no longer usable.
    Closed,
}

impl WindowState {
    /// Check if this window state allows rendering.
    ///
    /// A window is renderable only when it is active (not minimized, hidden,
    /// or closed).
    ///
    /// # Example
    ///
    /// ```ignore
    /// assert!(WindowState::Active.is_renderable());
    /// assert!(!WindowState::Minimized.is_renderable());
    /// assert!(!WindowState::Hidden.is_renderable());
    /// assert!(!WindowState::Closed.is_renderable());
    /// ```
    #[inline]
    pub const fn is_renderable(&self) -> bool {
        matches!(self, WindowState::Active)
    }

    /// Check if this window state indicates the window is visible.
    ///
    /// A window is visible when it is active. Minimized, hidden, and closed
    /// windows are not visible.
    ///
    /// # Example
    ///
    /// ```ignore
    /// assert!(WindowState::Active.is_visible());
    /// assert!(!WindowState::Minimized.is_visible());
    /// ```
    #[inline]
    pub const fn is_visible(&self) -> bool {
        matches!(self, WindowState::Active)
    }

    /// Check if the window can be restored to active state.
    ///
    /// Minimized and hidden windows can be restored; closed windows cannot.
    #[inline]
    pub const fn can_restore(&self) -> bool {
        matches!(self, WindowState::Minimized | WindowState::Hidden)
    }

    /// Check if the window is in a terminal state.
    ///
    /// Only closed windows are in a terminal state.
    #[inline]
    pub const fn is_terminal(&self) -> bool {
        matches!(self, WindowState::Closed)
    }
}

impl Default for WindowState {
    fn default() -> Self {
        WindowState::Active
    }
}

impl fmt::Display for WindowState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WindowState::Active => write!(f, "Active"),
            WindowState::Minimized => write!(f, "Minimized"),
            WindowState::Hidden => write!(f, "Hidden"),
            WindowState::Closed => write!(f, "Closed"),
        }
    }
}

// ============================================================================
// Managed Window
// ============================================================================

/// A managed window with its render target.
///
/// `ManagedWindow` combines the window configuration, current state, and
/// frame statistics for a single rendering window.
///
/// # Example
///
/// ```ignore
/// let config = WindowConfig::new("Window", 800, 600);
/// let mut window = ManagedWindow::new(config);
///
/// assert!(window.is_active());
/// assert!(window.is_visible());
///
/// window.set_state(WindowState::Minimized);
/// assert!(!window.is_active());
/// ```
#[derive(Clone, Debug)]
pub struct ManagedWindow {
    /// Unique window identifier.
    id: WindowId,
    /// Window configuration.
    config: WindowConfig,
    /// Current window state.
    state: WindowState,
    /// Number of frames rendered to this window.
    frame_count: u64,
}

impl ManagedWindow {
    /// Create a new managed window.
    ///
    /// The window is created in the Active state.
    ///
    /// # Arguments
    ///
    /// * `config` - Window configuration.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = WindowConfig::new("My Window", 1024, 768);
    /// let window = ManagedWindow::new(config);
    /// assert!(window.is_active());
    /// ```
    pub fn new(config: WindowConfig) -> Self {
        Self {
            id: config.id,
            config,
            state: WindowState::Active,
            frame_count: 0,
        }
    }

    /// Get the window ID.
    #[inline]
    pub const fn id(&self) -> WindowId {
        self.id
    }

    /// Get the current window state.
    #[inline]
    pub const fn state(&self) -> &WindowState {
        &self.state
    }

    /// Check if the window is in the Active state.
    #[inline]
    pub const fn is_active(&self) -> bool {
        matches!(self.state, WindowState::Active)
    }

    /// Check if the window is visible (not minimized, hidden, or closed).
    #[inline]
    pub const fn is_visible(&self) -> bool {
        self.state.is_visible()
    }

    /// Get the window dimensions.
    #[inline]
    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// Set the window state.
    ///
    /// # Arguments
    ///
    /// * `state` - The new window state.
    ///
    /// # Example
    ///
    /// ```ignore
    /// window.set_state(WindowState::Minimized);
    /// assert!(!window.is_active());
    /// ```
    #[inline]
    pub fn set_state(&mut self, state: WindowState) {
        self.state = state;
    }

    /// Get the window configuration.
    #[inline]
    pub const fn config(&self) -> &WindowConfig {
        &self.config
    }

    /// Get the window title.
    #[inline]
    pub fn title(&self) -> &str {
        &self.config.title
    }

    /// Check if VSync is enabled.
    #[inline]
    pub const fn vsync(&self) -> bool {
        self.config.vsync
    }

    /// Check if this is the primary window.
    #[inline]
    pub const fn is_primary(&self) -> bool {
        self.config.primary
    }

    /// Get the total number of frames rendered to this window.
    #[inline]
    pub const fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Increment the frame counter.
    ///
    /// Called after successfully presenting a frame.
    #[inline]
    pub fn increment_frame_count(&mut self) {
        self.frame_count = self.frame_count.saturating_add(1);
    }

    /// Update the window size.
    ///
    /// # Arguments
    ///
    /// * `width` - New width in pixels.
    /// * `height` - New height in pixels.
    #[inline]
    pub fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width;
        self.config.height = height;
    }
}

impl fmt::Display for ManagedWindow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({}, {}x{}, {} frames)",
            self.config.title,
            self.state,
            self.config.width,
            self.config.height,
            self.frame_count
        )
    }
}

// ============================================================================
// Window Events
// ============================================================================

/// Window event for state changes.
///
/// `WindowEvent` represents events that can occur on windows in the multi-window
/// system. The manager processes these events to update window states.
///
/// # Example
///
/// ```ignore
/// let event = WindowEvent::Created(window_id);
/// manager.handle_event(event);
///
/// let event = WindowEvent::Resized { id: window_id, width: 1920, height: 1080 };
/// manager.handle_event(event);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WindowEvent {
    /// A new window was created.
    Created(WindowId),
    /// A window was resized.
    Resized {
        /// The window that was resized.
        id: WindowId,
        /// New width in pixels.
        width: u32,
        /// New height in pixels.
        height: u32,
    },
    /// A window gained input focus.
    Focused(WindowId),
    /// A window lost input focus.
    Unfocused(WindowId),
    /// A window was minimized.
    Minimized(WindowId),
    /// A window was restored from minimized state.
    Restored(WindowId),
    /// A window was closed.
    Closed(WindowId),
}

impl WindowEvent {
    /// Get the window ID associated with this event.
    #[inline]
    pub const fn window_id(&self) -> WindowId {
        match self {
            WindowEvent::Created(id)
            | WindowEvent::Focused(id)
            | WindowEvent::Unfocused(id)
            | WindowEvent::Minimized(id)
            | WindowEvent::Restored(id)
            | WindowEvent::Closed(id) => *id,
            WindowEvent::Resized { id, .. } => *id,
        }
    }

    /// Check if this event represents a state change.
    #[inline]
    pub const fn is_state_change(&self) -> bool {
        matches!(
            self,
            WindowEvent::Created(_)
                | WindowEvent::Minimized(_)
                | WindowEvent::Restored(_)
                | WindowEvent::Closed(_)
        )
    }

    /// Check if this event represents a focus change.
    #[inline]
    pub const fn is_focus_change(&self) -> bool {
        matches!(self, WindowEvent::Focused(_) | WindowEvent::Unfocused(_))
    }

    /// Check if this event represents a size change.
    #[inline]
    pub const fn is_size_change(&self) -> bool {
        matches!(self, WindowEvent::Resized { .. })
    }
}

impl fmt::Display for WindowEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WindowEvent::Created(id) => write!(f, "Created({})", id),
            WindowEvent::Resized { id, width, height } => {
                write!(f, "Resized({}, {}x{})", id, width, height)
            }
            WindowEvent::Focused(id) => write!(f, "Focused({})", id),
            WindowEvent::Unfocused(id) => write!(f, "Unfocused({})", id),
            WindowEvent::Minimized(id) => write!(f, "Minimized({})", id),
            WindowEvent::Restored(id) => write!(f, "Restored({})", id),
            WindowEvent::Closed(id) => write!(f, "Closed({})", id),
        }
    }
}

// ============================================================================
// Multi-Window Manager
// ============================================================================

/// Manager for multiple rendering windows.
///
/// `MultiWindowManager` provides centralized management of multiple windows,
/// including creation, destruction, focus tracking, and event handling.
///
/// # Example
///
/// ```ignore
/// let mut manager = MultiWindowManager::new();
///
/// // Create primary window
/// let main_config = WindowConfig::new("Main", 1920, 1080).as_primary();
/// let main_id = manager.create_window(main_config);
///
/// // Create secondary window
/// let tools_config = WindowConfig::new("Tools", 400, 600);
/// let tools_id = manager.create_window(tools_config);
///
/// // Query windows
/// assert_eq!(manager.window_count(), 2);
/// assert_eq!(manager.primary_window(), Some(main_id));
///
/// // Handle events
/// manager.handle_event(WindowEvent::Minimized(tools_id));
///
/// // Get active windows for rendering
/// let active = manager.active_windows();
/// ```
#[derive(Debug)]
pub struct MultiWindowManager {
    /// All managed windows.
    windows: HashMap<WindowId, ManagedWindow>,
    /// Primary window ID (if set).
    primary_window: Option<WindowId>,
    /// Counter for generating unique IDs.
    next_id: u64,
    /// Currently focused window.
    focused_window: Option<WindowId>,
}

impl MultiWindowManager {
    /// Create a new multi-window manager.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let manager = MultiWindowManager::new();
    /// assert_eq!(manager.window_count(), 0);
    /// assert!(manager.primary_window().is_none());
    /// ```
    pub fn new() -> Self {
        Self {
            windows: HashMap::new(),
            primary_window: None,
            next_id: 1,
            focused_window: None,
        }
    }

    /// Create a new window with the given configuration.
    ///
    /// Returns the assigned `WindowId` for the new window.
    ///
    /// # Arguments
    ///
    /// * `config` - Window configuration.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = WindowConfig::new("My Window", 800, 600);
    /// let id = manager.create_window(config);
    /// assert!(manager.get_window(id).is_some());
    /// ```
    pub fn create_window(&mut self, mut config: WindowConfig) -> WindowId {
        // Assign a unique ID
        let id = WindowId::new(self.next_id);
        self.next_id += 1;
        config.id = id;

        let is_primary = config.primary;
        let window = ManagedWindow::new(config);

        self.windows.insert(id, window);

        // Set as primary if marked or if first window
        if is_primary || self.primary_window.is_none() {
            self.primary_window = Some(id);
        }

        // Set focus to new window if none focused
        if self.focused_window.is_none() {
            self.focused_window = Some(id);
        }

        id
    }

    /// Close and remove a window.
    ///
    /// Returns `true` if the window was found and closed, `false` otherwise.
    ///
    /// # Arguments
    ///
    /// * `id` - The window to close.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let id = manager.create_window(config);
    /// assert!(manager.close_window(id));
    /// assert!(manager.get_window(id).is_none());
    /// ```
    pub fn close_window(&mut self, id: WindowId) -> bool {
        if self.windows.remove(&id).is_some() {
            // Update primary if this was the primary window
            if self.primary_window == Some(id) {
                self.primary_window = self.windows.keys().next().copied();
            }

            // Update focus if this was the focused window
            if self.focused_window == Some(id) {
                self.focused_window = self.primary_window.or_else(|| self.windows.keys().next().copied());
            }

            true
        } else {
            false
        }
    }

    /// Get a reference to a window by ID.
    ///
    /// # Arguments
    ///
    /// * `id` - The window ID to look up.
    ///
    /// # Example
    ///
    /// ```ignore
    /// if let Some(window) = manager.get_window(id) {
    ///     println!("Window size: {:?}", window.size());
    /// }
    /// ```
    #[inline]
    pub fn get_window(&self, id: WindowId) -> Option<&ManagedWindow> {
        self.windows.get(&id)
    }

    /// Get a mutable reference to a window by ID.
    ///
    /// # Arguments
    ///
    /// * `id` - The window ID to look up.
    ///
    /// # Example
    ///
    /// ```ignore
    /// if let Some(window) = manager.get_window_mut(id) {
    ///     window.set_state(WindowState::Minimized);
    /// }
    /// ```
    #[inline]
    pub fn get_window_mut(&mut self, id: WindowId) -> Option<&mut ManagedWindow> {
        self.windows.get_mut(&id)
    }

    /// Get the primary window ID.
    ///
    /// Returns `None` if no primary window is set.
    ///
    /// # Example
    ///
    /// ```ignore
    /// if let Some(primary_id) = manager.primary_window() {
    ///     println!("Primary window: {}", primary_id);
    /// }
    /// ```
    #[inline]
    pub fn primary_window(&self) -> Option<WindowId> {
        self.primary_window
    }

    /// Set the primary window.
    ///
    /// Returns `true` if the window exists and was set as primary,
    /// `false` if the window was not found.
    ///
    /// # Arguments
    ///
    /// * `id` - The window to set as primary.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let id = manager.create_window(config);
    /// assert!(manager.set_primary(id));
    /// assert_eq!(manager.primary_window(), Some(id));
    /// ```
    pub fn set_primary(&mut self, id: WindowId) -> bool {
        if self.windows.contains_key(&id) {
            // Update old primary
            if let Some(old_id) = self.primary_window {
                if let Some(old_window) = self.windows.get_mut(&old_id) {
                    old_window.config.primary = false;
                }
            }

            // Set new primary
            if let Some(window) = self.windows.get_mut(&id) {
                window.config.primary = true;
            }
            self.primary_window = Some(id);
            true
        } else {
            false
        }
    }

    /// Get all active (renderable) window IDs.
    ///
    /// Returns windows that are in the `Active` state.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let active_ids = manager.active_windows();
    /// for id in active_ids {
    ///     // Render to window
    /// }
    /// ```
    pub fn active_windows(&self) -> Vec<WindowId> {
        self.windows
            .iter()
            .filter(|(_, w)| w.is_active())
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get all visible window IDs.
    ///
    /// Returns windows that are visible (same as active in this implementation).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let visible_ids = manager.visible_windows();
    /// ```
    pub fn visible_windows(&self) -> Vec<WindowId> {
        self.windows
            .iter()
            .filter(|(_, w)| w.is_visible())
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get the total number of managed windows.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let count = manager.window_count();
    /// println!("Managing {} windows", count);
    /// ```
    #[inline]
    pub fn window_count(&self) -> usize {
        self.windows.len()
    }

    /// Handle a window event.
    ///
    /// Updates window states based on the event type.
    ///
    /// # Arguments
    ///
    /// * `event` - The window event to handle.
    ///
    /// # Example
    ///
    /// ```ignore
    /// manager.handle_event(WindowEvent::Minimized(window_id));
    /// manager.handle_event(WindowEvent::Restored(window_id));
    /// ```
    pub fn handle_event(&mut self, event: WindowEvent) {
        match event {
            WindowEvent::Created(_id) => {
                // Window creation is handled by create_window()
            }
            WindowEvent::Resized { id, width, height } => {
                if let Some(window) = self.windows.get_mut(&id) {
                    window.resize(width, height);
                }
            }
            WindowEvent::Focused(id) => {
                // Unfocus previous window
                if let Some(old_id) = self.focused_window {
                    if old_id != id {
                        // Just update tracking, no state change needed
                    }
                }
                self.focused_window = Some(id);
            }
            WindowEvent::Unfocused(id) => {
                if self.focused_window == Some(id) {
                    self.focused_window = None;
                }
            }
            WindowEvent::Minimized(id) => {
                if let Some(window) = self.windows.get_mut(&id) {
                    window.set_state(WindowState::Minimized);
                }
            }
            WindowEvent::Restored(id) => {
                if let Some(window) = self.windows.get_mut(&id) {
                    window.set_state(WindowState::Active);
                }
            }
            WindowEvent::Closed(id) => {
                if let Some(window) = self.windows.get_mut(&id) {
                    window.set_state(WindowState::Closed);
                }
                // Optionally remove the window
                self.close_window(id);
            }
        }
    }

    /// Resize a window.
    ///
    /// Returns `true` if the window was found and resized, `false` otherwise.
    ///
    /// # Arguments
    ///
    /// * `id` - The window to resize.
    /// * `width` - New width in pixels.
    /// * `height` - New height in pixels.
    ///
    /// # Example
    ///
    /// ```ignore
    /// manager.resize_window(window_id, 1920, 1080);
    /// ```
    pub fn resize_window(&mut self, id: WindowId, width: u32, height: u32) -> bool {
        if let Some(window) = self.windows.get_mut(&id) {
            window.resize(width, height);
            true
        } else {
            false
        }
    }

    /// Get the currently focused window ID.
    #[inline]
    pub fn focused_window(&self) -> Option<WindowId> {
        self.focused_window
    }

    /// Set focus to a window.
    ///
    /// Returns `true` if the window exists and was focused.
    pub fn set_focus(&mut self, id: WindowId) -> bool {
        if self.windows.contains_key(&id) {
            self.focused_window = Some(id);
            true
        } else {
            false
        }
    }

    /// Iterate over all windows.
    pub fn iter(&self) -> impl Iterator<Item = (&WindowId, &ManagedWindow)> {
        self.windows.iter()
    }

    /// Iterate mutably over all windows.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&WindowId, &mut ManagedWindow)> {
        self.windows.iter_mut()
    }

    /// Get all window IDs.
    pub fn window_ids(&self) -> Vec<WindowId> {
        self.windows.keys().copied().collect()
    }

    /// Check if a window exists.
    #[inline]
    pub fn contains(&self, id: WindowId) -> bool {
        self.windows.contains_key(&id)
    }

    /// Check if the manager has any windows.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }

    /// Hide a window.
    pub fn hide_window(&mut self, id: WindowId) -> bool {
        if let Some(window) = self.windows.get_mut(&id) {
            window.set_state(WindowState::Hidden);
            true
        } else {
            false
        }
    }

    /// Show a hidden window.
    pub fn show_window(&mut self, id: WindowId) -> bool {
        if let Some(window) = self.windows.get_mut(&id) {
            if matches!(window.state(), WindowState::Hidden) {
                window.set_state(WindowState::Active);
            }
            true
        } else {
            false
        }
    }

    /// Minimize a window.
    pub fn minimize_window(&mut self, id: WindowId) -> bool {
        if let Some(window) = self.windows.get_mut(&id) {
            window.set_state(WindowState::Minimized);
            true
        } else {
            false
        }
    }

    /// Restore a minimized window.
    pub fn restore_window(&mut self, id: WindowId) -> bool {
        if let Some(window) = self.windows.get_mut(&id) {
            if window.state().can_restore() {
                window.set_state(WindowState::Active);
            }
            true
        } else {
            false
        }
    }
}

impl Default for MultiWindowManager {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- WindowId Tests ---

    #[test]
    fn test_window_id_creation() {
        let id = WindowId::new(42);
        assert_eq!(id.as_u64(), 42);
    }

    #[test]
    fn test_window_id_comparison() {
        let id1 = WindowId::new(1);
        let id2 = WindowId::new(1);
        let id3 = WindowId::new(2);

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_window_id_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(WindowId::new(1));
        set.insert(WindowId::new(2));
        set.insert(WindowId::new(1)); // Duplicate

        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_window_id_primary() {
        let primary = WindowId::primary();
        assert_eq!(primary.as_u64(), 0);
        assert!(primary.is_primary());

        let non_primary = WindowId::new(5);
        assert!(!non_primary.is_primary());
    }

    #[test]
    fn test_window_id_from_u64() {
        let id: WindowId = 123u64.into();
        assert_eq!(id.as_u64(), 123);
    }

    #[test]
    fn test_window_id_into_u64() {
        let id = WindowId::new(456);
        let val: u64 = id.into();
        assert_eq!(val, 456);
    }

    #[test]
    fn test_window_id_display() {
        let id = WindowId::new(7);
        assert_eq!(format!("{}", id), "Window(7)");
    }

    #[test]
    fn test_window_id_default() {
        let id = WindowId::default();
        assert_eq!(id.as_u64(), 0);
    }

    // --- WindowConfig Tests ---

    #[test]
    fn test_window_config_new() {
        let config = WindowConfig::new("Test Window", 800, 600);
        assert_eq!(config.title, "Test Window");
        assert_eq!(config.width, 800);
        assert_eq!(config.height, 600);
        assert!(!config.vsync);
        assert!(!config.primary);
    }

    #[test]
    fn test_window_config_with_vsync() {
        let config = WindowConfig::new("Window", 640, 480).with_vsync(true);
        assert!(config.vsync);
    }

    #[test]
    fn test_window_config_as_primary() {
        let config = WindowConfig::new("Main", 1920, 1080).as_primary();
        assert!(config.primary);
    }

    #[test]
    fn test_window_config_builder_chain() {
        let config = WindowConfig::new("Game", 1280, 720)
            .with_vsync(true)
            .as_primary();

        assert_eq!(config.title, "Game");
        assert_eq!(config.width, 1280);
        assert_eq!(config.height, 720);
        assert!(config.vsync);
        assert!(config.primary);
    }

    #[test]
    fn test_window_config_dimensions() {
        let config = WindowConfig::new("Win", 1024, 768);
        assert_eq!(config.dimensions(), (1024, 768));
    }

    #[test]
    fn test_window_config_aspect_ratio() {
        let config = WindowConfig::new("Wide", 1920, 1080);
        let ratio = config.aspect_ratio();
        assert!((ratio - 1.777).abs() < 0.01);

        let square = WindowConfig::new("Square", 100, 100);
        assert!((square.aspect_ratio() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_window_config_aspect_ratio_zero_height() {
        let config = WindowConfig::new("Zero", 100, 0);
        assert!((config.aspect_ratio() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_window_config_pixel_count() {
        let config = WindowConfig::new("Win", 1920, 1080);
        assert_eq!(config.pixel_count(), 1920 * 1080);
    }

    #[test]
    fn test_window_config_display() {
        let config = WindowConfig::new("Test", 800, 600).with_vsync(true);
        let display = format!("{}", config);
        assert!(display.contains("Test"));
        assert!(display.contains("800x600"));
        assert!(display.contains("vsync=true"));
    }

    // --- WindowState Tests ---

    #[test]
    fn test_window_state_is_renderable() {
        assert!(WindowState::Active.is_renderable());
        assert!(!WindowState::Minimized.is_renderable());
        assert!(!WindowState::Hidden.is_renderable());
        assert!(!WindowState::Closed.is_renderable());
    }

    #[test]
    fn test_window_state_is_visible() {
        assert!(WindowState::Active.is_visible());
        assert!(!WindowState::Minimized.is_visible());
        assert!(!WindowState::Hidden.is_visible());
        assert!(!WindowState::Closed.is_visible());
    }

    #[test]
    fn test_window_state_can_restore() {
        assert!(!WindowState::Active.can_restore());
        assert!(WindowState::Minimized.can_restore());
        assert!(WindowState::Hidden.can_restore());
        assert!(!WindowState::Closed.can_restore());
    }

    #[test]
    fn test_window_state_is_terminal() {
        assert!(!WindowState::Active.is_terminal());
        assert!(!WindowState::Minimized.is_terminal());
        assert!(!WindowState::Hidden.is_terminal());
        assert!(WindowState::Closed.is_terminal());
    }

    #[test]
    fn test_window_state_default() {
        assert_eq!(WindowState::default(), WindowState::Active);
    }

    #[test]
    fn test_window_state_display() {
        assert_eq!(format!("{}", WindowState::Active), "Active");
        assert_eq!(format!("{}", WindowState::Minimized), "Minimized");
        assert_eq!(format!("{}", WindowState::Hidden), "Hidden");
        assert_eq!(format!("{}", WindowState::Closed), "Closed");
    }

    // --- ManagedWindow Tests ---

    #[test]
    fn test_managed_window_new() {
        let config = WindowConfig::new("Test", 800, 600);
        let window = ManagedWindow::new(config);

        assert!(window.is_active());
        assert!(window.is_visible());
        assert_eq!(window.size(), (800, 600));
        assert_eq!(window.frame_count(), 0);
    }

    #[test]
    fn test_managed_window_state_management() {
        let config = WindowConfig::new("Test", 800, 600);
        let mut window = ManagedWindow::new(config);

        assert!(window.is_active());

        window.set_state(WindowState::Minimized);
        assert!(!window.is_active());
        assert!(!window.is_visible());

        window.set_state(WindowState::Active);
        assert!(window.is_active());
    }

    #[test]
    fn test_managed_window_frame_count() {
        let config = WindowConfig::new("Test", 800, 600);
        let mut window = ManagedWindow::new(config);

        assert_eq!(window.frame_count(), 0);
        window.increment_frame_count();
        assert_eq!(window.frame_count(), 1);
        window.increment_frame_count();
        assert_eq!(window.frame_count(), 2);
    }

    #[test]
    fn test_managed_window_resize() {
        let config = WindowConfig::new("Test", 800, 600);
        let mut window = ManagedWindow::new(config);

        window.resize(1920, 1080);
        assert_eq!(window.size(), (1920, 1080));
    }

    #[test]
    fn test_managed_window_properties() {
        let config = WindowConfig::new("My Window", 1024, 768)
            .with_vsync(true)
            .as_primary();
        let window = ManagedWindow::new(config);

        assert_eq!(window.title(), "My Window");
        assert!(window.vsync());
        assert!(window.is_primary());
    }

    #[test]
    fn test_managed_window_display() {
        let config = WindowConfig::new("Test", 800, 600);
        let window = ManagedWindow::new(config);
        let display = format!("{}", window);
        assert!(display.contains("Test"));
        assert!(display.contains("Active"));
        assert!(display.contains("800x600"));
    }

    // --- WindowEvent Tests ---

    #[test]
    fn test_window_event_window_id() {
        let id = WindowId::new(5);
        assert_eq!(WindowEvent::Created(id).window_id(), id);
        assert_eq!(WindowEvent::Focused(id).window_id(), id);
        assert_eq!(WindowEvent::Unfocused(id).window_id(), id);
        assert_eq!(WindowEvent::Minimized(id).window_id(), id);
        assert_eq!(WindowEvent::Restored(id).window_id(), id);
        assert_eq!(WindowEvent::Closed(id).window_id(), id);
        assert_eq!(WindowEvent::Resized { id, width: 100, height: 100 }.window_id(), id);
    }

    #[test]
    fn test_window_event_is_state_change() {
        let id = WindowId::new(1);
        assert!(WindowEvent::Created(id).is_state_change());
        assert!(WindowEvent::Minimized(id).is_state_change());
        assert!(WindowEvent::Restored(id).is_state_change());
        assert!(WindowEvent::Closed(id).is_state_change());
        assert!(!WindowEvent::Focused(id).is_state_change());
        assert!(!WindowEvent::Resized { id, width: 100, height: 100 }.is_state_change());
    }

    #[test]
    fn test_window_event_is_focus_change() {
        let id = WindowId::new(1);
        assert!(WindowEvent::Focused(id).is_focus_change());
        assert!(WindowEvent::Unfocused(id).is_focus_change());
        assert!(!WindowEvent::Created(id).is_focus_change());
    }

    #[test]
    fn test_window_event_is_size_change() {
        let id = WindowId::new(1);
        assert!(WindowEvent::Resized { id, width: 100, height: 100 }.is_size_change());
        assert!(!WindowEvent::Created(id).is_size_change());
    }

    #[test]
    fn test_window_event_display() {
        let id = WindowId::new(3);
        assert_eq!(format!("{}", WindowEvent::Created(id)), "Created(Window(3))");
        assert_eq!(format!("{}", WindowEvent::Resized { id, width: 800, height: 600 }), "Resized(Window(3), 800x600)");
    }

    // --- MultiWindowManager Tests ---

    #[test]
    fn test_manager_new() {
        let manager = MultiWindowManager::new();
        assert_eq!(manager.window_count(), 0);
        assert!(manager.primary_window().is_none());
        assert!(manager.is_empty());
    }

    #[test]
    fn test_manager_create_window() {
        let mut manager = MultiWindowManager::new();
        let config = WindowConfig::new("Test", 800, 600);
        let id = manager.create_window(config);

        assert_eq!(manager.window_count(), 1);
        assert!(!manager.is_empty());
        assert!(manager.get_window(id).is_some());
    }

    #[test]
    fn test_manager_close_window() {
        let mut manager = MultiWindowManager::new();
        let config = WindowConfig::new("Test", 800, 600);
        let id = manager.create_window(config);

        assert!(manager.close_window(id));
        assert!(manager.get_window(id).is_none());
        assert_eq!(manager.window_count(), 0);
    }

    #[test]
    fn test_manager_close_nonexistent() {
        let mut manager = MultiWindowManager::new();
        assert!(!manager.close_window(WindowId::new(999)));
    }

    #[test]
    fn test_manager_primary_window() {
        let mut manager = MultiWindowManager::new();

        // First window becomes primary
        let id1 = manager.create_window(WindowConfig::new("Win1", 800, 600));
        assert_eq!(manager.primary_window(), Some(id1));

        // Explicitly marked primary takes over
        let id2 = manager.create_window(WindowConfig::new("Win2", 800, 600).as_primary());
        assert_eq!(manager.primary_window(), Some(id2));
    }

    #[test]
    fn test_manager_set_primary() {
        let mut manager = MultiWindowManager::new();
        let id1 = manager.create_window(WindowConfig::new("Win1", 800, 600));
        let id2 = manager.create_window(WindowConfig::new("Win2", 800, 600));

        assert!(manager.set_primary(id2));
        assert_eq!(manager.primary_window(), Some(id2));

        // Setting non-existent as primary fails
        assert!(!manager.set_primary(WindowId::new(999)));
    }

    #[test]
    fn test_manager_active_windows() {
        let mut manager = MultiWindowManager::new();
        let id1 = manager.create_window(WindowConfig::new("Win1", 800, 600));
        let id2 = manager.create_window(WindowConfig::new("Win2", 800, 600));

        let active = manager.active_windows();
        assert_eq!(active.len(), 2);
        assert!(active.contains(&id1));
        assert!(active.contains(&id2));
    }

    #[test]
    fn test_manager_visible_windows() {
        let mut manager = MultiWindowManager::new();
        let id1 = manager.create_window(WindowConfig::new("Win1", 800, 600));
        let id2 = manager.create_window(WindowConfig::new("Win2", 800, 600));

        manager.minimize_window(id2);

        let visible = manager.visible_windows();
        assert_eq!(visible.len(), 1);
        assert!(visible.contains(&id1));
    }

    #[test]
    fn test_manager_handle_resize_event() {
        let mut manager = MultiWindowManager::new();
        let id = manager.create_window(WindowConfig::new("Test", 800, 600));

        manager.handle_event(WindowEvent::Resized { id, width: 1920, height: 1080 });

        let window = manager.get_window(id).unwrap();
        assert_eq!(window.size(), (1920, 1080));
    }

    #[test]
    fn test_manager_handle_minimize_restore() {
        let mut manager = MultiWindowManager::new();
        let id = manager.create_window(WindowConfig::new("Test", 800, 600));

        manager.handle_event(WindowEvent::Minimized(id));
        assert!(!manager.get_window(id).unwrap().is_active());

        manager.handle_event(WindowEvent::Restored(id));
        assert!(manager.get_window(id).unwrap().is_active());
    }

    #[test]
    fn test_manager_handle_focus() {
        let mut manager = MultiWindowManager::new();
        let id1 = manager.create_window(WindowConfig::new("Win1", 800, 600));
        let id2 = manager.create_window(WindowConfig::new("Win2", 800, 600));

        manager.handle_event(WindowEvent::Focused(id2));
        assert_eq!(manager.focused_window(), Some(id2));

        manager.handle_event(WindowEvent::Unfocused(id2));
        assert_eq!(manager.focused_window(), None);

        manager.handle_event(WindowEvent::Focused(id1));
        assert_eq!(manager.focused_window(), Some(id1));
    }

    #[test]
    fn test_manager_handle_closed() {
        let mut manager = MultiWindowManager::new();
        let id = manager.create_window(WindowConfig::new("Test", 800, 600));

        manager.handle_event(WindowEvent::Closed(id));
        assert!(manager.get_window(id).is_none());
    }

    #[test]
    fn test_manager_resize_window() {
        let mut manager = MultiWindowManager::new();
        let id = manager.create_window(WindowConfig::new("Test", 800, 600));

        assert!(manager.resize_window(id, 1024, 768));
        assert_eq!(manager.get_window(id).unwrap().size(), (1024, 768));

        assert!(!manager.resize_window(WindowId::new(999), 100, 100));
    }

    #[test]
    fn test_manager_window_state_transitions() {
        let mut manager = MultiWindowManager::new();
        let id = manager.create_window(WindowConfig::new("Test", 800, 600));

        // Active -> Hidden -> Active
        manager.hide_window(id);
        assert!(!manager.get_window(id).unwrap().is_visible());

        manager.show_window(id);
        assert!(manager.get_window(id).unwrap().is_visible());

        // Active -> Minimized -> Active
        manager.minimize_window(id);
        assert!(!manager.get_window(id).unwrap().is_active());

        manager.restore_window(id);
        assert!(manager.get_window(id).unwrap().is_active());
    }

    #[test]
    fn test_manager_closing_primary_updates_primary() {
        let mut manager = MultiWindowManager::new();
        let id1 = manager.create_window(WindowConfig::new("Win1", 800, 600));
        let id2 = manager.create_window(WindowConfig::new("Win2", 800, 600));

        // id1 is primary (first created)
        assert_eq!(manager.primary_window(), Some(id1));

        // Close primary
        manager.close_window(id1);

        // id2 should become primary
        assert!(manager.primary_window().is_some());
    }

    #[test]
    fn test_manager_contains() {
        let mut manager = MultiWindowManager::new();
        let id = manager.create_window(WindowConfig::new("Test", 800, 600));

        assert!(manager.contains(id));
        assert!(!manager.contains(WindowId::new(999)));
    }

    #[test]
    fn test_manager_window_ids() {
        let mut manager = MultiWindowManager::new();
        let id1 = manager.create_window(WindowConfig::new("Win1", 800, 600));
        let id2 = manager.create_window(WindowConfig::new("Win2", 800, 600));

        let ids = manager.window_ids();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&id1));
        assert!(ids.contains(&id2));
    }

    #[test]
    fn test_manager_iter() {
        let mut manager = MultiWindowManager::new();
        manager.create_window(WindowConfig::new("Win1", 800, 600));
        manager.create_window(WindowConfig::new("Win2", 800, 600));

        let count = manager.iter().count();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_manager_iter_mut() {
        let mut manager = MultiWindowManager::new();
        manager.create_window(WindowConfig::new("Win1", 800, 600));
        manager.create_window(WindowConfig::new("Win2", 800, 600));

        for (_, window) in manager.iter_mut() {
            window.increment_frame_count();
        }

        for (_, window) in manager.iter() {
            assert_eq!(window.frame_count(), 1);
        }
    }

    #[test]
    fn test_manager_set_focus() {
        let mut manager = MultiWindowManager::new();
        let id = manager.create_window(WindowConfig::new("Test", 800, 600));

        assert!(manager.set_focus(id));
        assert_eq!(manager.focused_window(), Some(id));

        assert!(!manager.set_focus(WindowId::new(999)));
    }

    #[test]
    fn test_manager_default() {
        let manager = MultiWindowManager::default();
        assert_eq!(manager.window_count(), 0);
    }

    #[test]
    fn test_duplicate_window_ids_not_possible() {
        let mut manager = MultiWindowManager::new();
        let id1 = manager.create_window(WindowConfig::new("Win1", 800, 600));
        let id2 = manager.create_window(WindowConfig::new("Win2", 800, 600));

        // Each window gets a unique ID
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_closing_focused_updates_focus() {
        let mut manager = MultiWindowManager::new();
        let id1 = manager.create_window(WindowConfig::new("Win1", 800, 600));
        let id2 = manager.create_window(WindowConfig::new("Win2", 800, 600));

        manager.set_focus(id1);
        assert_eq!(manager.focused_window(), Some(id1));

        manager.close_window(id1);
        // Focus should move to another window or primary
        assert!(manager.focused_window().is_some() || manager.is_empty());
    }

    #[test]
    fn test_empty_manager_queries() {
        let manager = MultiWindowManager::new();

        assert!(manager.active_windows().is_empty());
        assert!(manager.visible_windows().is_empty());
        assert!(manager.window_ids().is_empty());
        assert!(manager.primary_window().is_none());
        assert!(manager.focused_window().is_none());
    }

    #[test]
    fn test_get_window_mut() {
        let mut manager = MultiWindowManager::new();
        let id = manager.create_window(WindowConfig::new("Test", 800, 600));

        if let Some(window) = manager.get_window_mut(id) {
            window.resize(1024, 768);
        }

        assert_eq!(manager.get_window(id).unwrap().size(), (1024, 768));
    }
}
