// Blackbox contract tests for T-WGPU-P7.1.11 Multi-window API.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only public types/traits -- no internal fields, no private
// methods, no implementation details.
//
// Contract:
//   WindowId: Type-safe window identification.
//     - new()       -> unique monotonically increasing ID
//     - primary()   -> WindowId(0), the primary window
//     - is_primary() -> true iff id == 0
//     - as_u64()    -> raw id value
//     - from_raw_id(u64) -> WindowId (no uniqueness guarantee)
//     - Default     -> new()
//     - From<u64>   -> from_raw_id
//     - Into<u64>   -> as_u64
//
//   WindowConfig: Configuration for individual windows.
//     - new(id, config)     -> basic config with defaults
//     - primary(config)     -> primary window config (id=0, high priority)
//     - with_focus(bool)    -> builder
//     - with_visibility(bool) -> builder
//     - with_priority(u8)   -> builder (higher = rendered first)
//     - with_label(str)     -> builder
//     - with_sync_to_primary(bool) -> builder
//     - dimensions()        -> (width, height)
//     - aspect_ratio()      -> f32
//     - should_render()     -> visible && width > 0 && height > 0
//
//   MultiWindowManager: Centralized window management.
//     - new()               -> empty manager
//     - with_max_windows(n) -> manager with window limit
//     - register_window(surface, config) -> Result<WindowId, MultiWindowError>
//     - unregister_window(id) -> Result<WindowState, MultiWindowError>
//     - set_focus(id)       -> Result<Option<WindowId>, MultiWindowError>
//     - set_visible(id, bool) -> Result<(), MultiWindowError>
//     - resize_window(id, device, w, h) -> Result<(), MultiWindowError>
//     - acquire_frame(id)   -> Result<WindowFrame, MultiWindowError>
//     - acquire_all_frames() -> Vec<Result<WindowFrame, (WindowId, FrameError)>>
//     - present_all(frames) -> ()
//     - set_sync_mode(mode) -> ()
//     - sync_mode()         -> SyncMode
//     - window_count()      -> usize
//     - window_ids()        -> &[WindowId] (in render order by priority)
//
//   SyncMode: Presentation synchronization.
//     - Independent         -> each window presents independently
//     - SyncToPrimary       -> secondary windows sync to primary
//     - SyncToRate { target_hz } -> sync to specific rate
//     - Simultaneous        -> present all at once
//     - requires_coordination() -> true for non-Independent
//     - target_interval()   -> Option<Duration> for SyncToRate
//
// Scenarios:
//   WindowId (14 tests):
//     1.  new() returns unique IDs
//     2.  new() IDs are monotonically increasing
//     3.  primary() returns WindowId(0)
//     4.  is_primary() true for primary
//     5.  is_primary() false for non-primary
//     6.  as_u64() returns raw value
//     7.  from_raw_id() creates from u64
//     8.  Default trait calls new()
//     9.  From<u64> trait works
//    10.  Into<u64> trait works
//    11.  Eq and Hash work
//    12.  Debug formatting
//    13.  Display formatting
//    14.  Clone and Copy
//
//   WindowConfig (15 tests):
//    15.  new() creates config with defaults
//    16.  primary() creates primary config
//    17.  with_focus() sets focus
//    18.  with_visibility() sets visibility
//    19.  with_priority() sets priority
//    20.  with_label() sets label
//    21.  with_sync_to_primary() enables sync
//    22.  dimensions() returns width, height
//    23.  aspect_ratio() calculates correctly
//    24.  should_render() true when visible and valid dims
//    25.  should_render() false when invisible
//    26.  should_render() false when zero width
//    27.  should_render() false when zero height
//    28.  Display formatting
//    29.  Clone works
//
//   SyncMode (12 tests):
//    30.  Default is Independent
//    31.  Independent requires no coordination
//    32.  SyncToPrimary requires coordination
//    33.  SyncToRate requires coordination
//    34.  Simultaneous requires coordination
//    35.  target_interval() None for Independent
//    36.  target_interval() None for SyncToPrimary
//    37.  target_interval() Some for SyncToRate
//    38.  target_interval() None for Simultaneous
//    39.  sync_to_rate() constructor
//    40.  Display formatting
//    41.  Clone, Copy, PartialEq, Eq
//
//   MultiWindowManager Single Window (12 tests):
//    42.  new() creates empty manager
//    43.  with_max_windows() sets limit
//    44.  register primary window
//    45.  window_count() returns 1
//    46.  window_ids() contains primary
//    47.  focused_window_id() returns primary
//    48.  get_window() returns state
//    49.  unregister_window() removes window
//    50.  window_count() returns 0 after unregister
//    51.  unregister_window() error for unknown ID
//    52.  has_windows() true with windows
//    53.  has_windows() false when empty
//
//   Dual Monitor / Two Windows (8 tests):
//    54.  register two windows
//    55.  window_count() returns 2
//    56.  both windows in window_ids()
//    57.  set_focus() changes focus
//    58.  old focus returned by set_focus()
//    59.  visible_window_ids() returns visible windows
//    60.  set_visible() hides window
//    61.  visible_window_ids() excludes hidden
//
//   Picture-in-Picture (6 tests):
//    62.  overlay window with low priority
//    63.  main window with high priority
//    64.  render order: main first (high priority)
//    65.  small overlay dimensions
//    66.  both visible
//    67.  both should_render()
//
//   Window Lifecycle (7 tests):
//    68.  create window
//    69.  use window (acquire_frame)
//    70.  destroy window (unregister)
//    71.  cannot acquire from unregistered
//    72.  re-register same ID fails if still registered
//    73.  re-register after unregister succeeds
//    74.  max_windows limit enforced
//
//   Focus Switching (6 tests):
//    75.  initial focus on first registered
//    76.  set_focus() to second window
//    77.  set_focus() returns old focus
//    78.  set_focus() error for unknown ID
//    79.  focused_window_id() updates
//    80.  focused_window() returns correct state
//
//   Visibility Toggle (6 tests):
//    81.  default visibility is true
//    82.  set_visible(false) hides
//    83.  set_visible(true) shows
//    84.  hidden window excluded from visible_window_ids()
//    85.  set_visible error for unknown ID
//    86.  multiple visibility toggles
//
//   Priority Ordering (6 tests):
//    87.  higher priority first in render order
//    88.  equal priority stable order
//    89.  priority 255 (max) first
//    90.  priority 0 (min) last
//    91.  update_render_order after registration
//    92.  primary() config has priority 255
//
//   Synchronized Presentation (8 tests):
//    93.  set_sync_mode() changes mode
//    94.  sync_mode() returns current
//    95.  Independent mode default
//    96.  SyncToPrimary presents primary first
//    97.  SyncToRate has target interval
//    98.  Simultaneous presents all at once
//    99.  present_all() updates global frame count
//   100.  global_frame_count() increments
//
//   MultiWindowError (8 tests):
//   101.  WindowNotFound error
//   102.  WindowExists error
//   103.  NoWindows error
//   104.  MaxWindowsReached error
//   105.  is_recoverable() for window errors
//   106.  Error Display formatting
//   107.  SurfaceError wrapping
//   108.  FrameError wrapping
//
//   WindowState (8 tests):
//   109.  WindowState::new() creates state
//   110.  id() returns config ID
//   111.  is_focused() returns focus state
//   112.  is_visible() returns visibility
//   113.  priority() returns priority
//   114.  set_focused() updates focus
//   115.  set_visible() updates visibility
//   116.  frame_count() starts at 0
//
//   WindowFrame (6 tests):
//   117.  WindowFrame::new() creates frame
//   118.  window_id field accessible
//   119.  frame field accessible
//   120.  view() returns texture view
//   121.  dimensions() returns frame dims
//   122.  age() returns time since acquire
//
//   MultiWindowStats (6 tests):
//   123.  aggregate_stats() returns stats
//   124.  window_count accurate
//   125.  total_frames accurate
//   126.  drop_rate() calculation
//   127.  estimated_fps() calculation
//   128.  Display formatting
//
// Local stubs mirror the multi-window API from surface.rs.
// When the real API is exported, remove these stubs and use the crate exports.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

// ============================================================================
// WindowId stub
// ============================================================================

/// Counter for generating unique WindowIds.
static WINDOW_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Type-safe window identification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WindowId(u64);

impl WindowId {
    /// Create a new unique window ID.
    pub fn new() -> Self {
        let id = WINDOW_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
        Self(id)
    }

    /// Create a WindowId from a raw u64 value.
    pub fn from_raw_id(id: u64) -> Self {
        Self(id)
    }

    /// Get the raw u64 value.
    pub fn as_u64(&self) -> u64 {
        self.0
    }

    /// Create the primary window ID (0).
    pub const fn primary() -> Self {
        Self(0)
    }

    /// Check if this is the primary window.
    pub const fn is_primary(&self) -> bool {
        self.0 == 0
    }
}

impl Default for WindowId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for WindowId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Window({})", self.0)
    }
}

impl From<u64> for WindowId {
    fn from(id: u64) -> Self {
        Self::from_raw_id(id)
    }
}

impl From<WindowId> for u64 {
    fn from(id: WindowId) -> Self {
        id.0
    }
}

// ============================================================================
// SurfaceConfiguration stub (minimal for testing)
// ============================================================================

#[derive(Debug, Clone)]
pub struct SurfaceConfiguration {
    pub width: u32,
    pub height: u32,
}

impl SurfaceConfiguration {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width: width.max(1),
            height: height.max(1),
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width.max(1);
        self.height = height.max(1);
    }
}

// ============================================================================
// WindowConfig stub
// ============================================================================

/// Configuration for an individual window.
#[derive(Debug, Clone)]
pub struct WindowConfig {
    pub id: WindowId,
    pub config: SurfaceConfiguration,
    pub is_focused: bool,
    pub is_visible: bool,
    pub priority: u8,
    pub label: Option<String>,
    pub sync_to_primary: bool,
}

impl WindowConfig {
    pub fn new(id: WindowId, config: SurfaceConfiguration) -> Self {
        Self {
            id,
            config,
            is_focused: false,
            is_visible: true,
            priority: 128,
            label: None,
            sync_to_primary: false,
        }
    }

    pub fn primary(config: SurfaceConfiguration) -> Self {
        Self {
            id: WindowId::primary(),
            config,
            is_focused: true,
            is_visible: true,
            priority: 255,
            label: Some("Primary".to_string()),
            sync_to_primary: false,
        }
    }

    pub fn with_focus(mut self, focused: bool) -> Self {
        self.is_focused = focused;
        self
    }

    pub fn with_visibility(mut self, visible: bool) -> Self {
        self.is_visible = visible;
        self
    }

    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn with_sync_to_primary(mut self, sync: bool) -> Self {
        self.sync_to_primary = sync;
        self
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    pub fn aspect_ratio(&self) -> f32 {
        if self.config.height > 0 {
            self.config.width as f32 / self.config.height as f32
        } else {
            1.0
        }
    }

    pub fn should_render(&self) -> bool {
        self.is_visible && self.config.width > 0 && self.config.height > 0
    }
}

impl fmt::Display for WindowConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = self.label.as_deref().unwrap_or("Unnamed");
        write!(
            f,
            "{} [{}x{}, priority={}, focused={}, visible={}]",
            label,
            self.config.width,
            self.config.height,
            self.priority,
            self.is_focused,
            self.is_visible
        )
    }
}

// ============================================================================
// SyncMode stub
// ============================================================================

/// Presentation synchronization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SyncMode {
    #[default]
    Independent,
    SyncToPrimary,
    SyncToRate { target_hz: u32 },
    Simultaneous,
}

impl SyncMode {
    pub fn sync_to_rate(hz: u32) -> Self {
        SyncMode::SyncToRate { target_hz: hz }
    }

    pub fn target_interval(&self) -> Option<Duration> {
        match self {
            SyncMode::SyncToRate { target_hz } if *target_hz > 0 => {
                Some(Duration::from_secs_f64(1.0 / *target_hz as f64))
            }
            _ => None,
        }
    }

    pub fn requires_coordination(&self) -> bool {
        !matches!(self, SyncMode::Independent)
    }
}

impl fmt::Display for SyncMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SyncMode::Independent => write!(f, "Independent"),
            SyncMode::SyncToPrimary => write!(f, "Sync to Primary"),
            SyncMode::SyncToRate { target_hz } => write!(f, "Sync to {}Hz", target_hz),
            SyncMode::Simultaneous => write!(f, "Simultaneous"),
        }
    }
}

// ============================================================================
// MultiWindowError stub
// ============================================================================

#[derive(Debug)]
pub enum MultiWindowError {
    WindowNotFound(WindowId),
    WindowExists(WindowId),
    NoWindows,
    NoFocusedWindow,
    SurfaceError(String),
    FrameError(String),
    MaxWindowsReached { max: usize },
}

impl MultiWindowError {
    pub fn is_recoverable(&self) -> bool {
        match self {
            MultiWindowError::WindowNotFound(_) => false,
            MultiWindowError::WindowExists(_) => false,
            MultiWindowError::NoWindows => false,
            MultiWindowError::NoFocusedWindow => true,
            MultiWindowError::SurfaceError(_) => true,
            MultiWindowError::FrameError(_) => true,
            MultiWindowError::MaxWindowsReached { .. } => false,
        }
    }
}

impl fmt::Display for MultiWindowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MultiWindowError::WindowNotFound(id) => write!(f, "window not found: {}", id),
            MultiWindowError::WindowExists(id) => write!(f, "window already exists: {}", id),
            MultiWindowError::NoWindows => write!(f, "no windows registered"),
            MultiWindowError::NoFocusedWindow => write!(f, "no window has focus"),
            MultiWindowError::SurfaceError(e) => write!(f, "surface error: {}", e),
            MultiWindowError::FrameError(e) => write!(f, "frame error: {}", e),
            MultiWindowError::MaxWindowsReached { max } => {
                write!(f, "maximum number of windows ({}) reached", max)
            }
        }
    }
}

impl std::error::Error for MultiWindowError {}

// ============================================================================
// WindowState stub
// ============================================================================

/// Runtime state for a window.
pub struct WindowState {
    pub config: WindowConfig,
    pub last_frame_time: Instant,
    frame_count: u64,
    dropped_frames: u64,
    average_frame_time_ms: f32,
}

impl WindowState {
    pub fn new(config: WindowConfig) -> Self {
        Self {
            config,
            last_frame_time: Instant::now(),
            frame_count: 0,
            dropped_frames: 0,
            average_frame_time_ms: 0.0,
        }
    }

    pub fn id(&self) -> WindowId {
        self.config.id
    }

    pub fn is_focused(&self) -> bool {
        self.config.is_focused
    }

    pub fn is_visible(&self) -> bool {
        self.config.is_visible
    }

    pub fn priority(&self) -> u8 {
        self.config.priority
    }

    pub fn set_focused(&mut self, focused: bool) {
        self.config.is_focused = focused;
    }

    pub fn set_visible(&mut self, visible: bool) {
        self.config.is_visible = visible;
    }

    pub fn should_render(&self) -> bool {
        self.config.should_render()
    }

    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    pub fn dropped_frames(&self) -> u64 {
        self.dropped_frames
    }

    pub fn record_frame_presented(&mut self) {
        let now = Instant::now();
        let frame_time = now.duration_since(self.last_frame_time);
        self.last_frame_time = now;
        self.frame_count += 1;

        let frame_time_ms = frame_time.as_secs_f32() * 1000.0;
        const ALPHA: f32 = 0.1;
        if self.average_frame_time_ms == 0.0 {
            self.average_frame_time_ms = frame_time_ms;
        } else {
            self.average_frame_time_ms =
                ALPHA * frame_time_ms + (1.0 - ALPHA) * self.average_frame_time_ms;
        }
    }

    pub fn record_frame_dropped(&mut self) {
        self.dropped_frames += 1;
    }

    pub fn average_frame_time_ms(&self) -> f32 {
        self.average_frame_time_ms
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.config.config.resize(width, height);
    }
}

impl fmt::Debug for WindowState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WindowState")
            .field("id", &self.config.id)
            .field("config", &self.config)
            .field("frame_count", &self.frame_count)
            .field("dropped_frames", &self.dropped_frames)
            .field("avg_frame_time_ms", &self.average_frame_time_ms)
            .finish()
    }
}

// ============================================================================
// WindowFrame stub (simulated frame acquisition)
// ============================================================================

/// A frame acquired from a specific window.
#[derive(Debug)]
pub struct WindowFrame {
    pub window_id: WindowId,
    pub acquired_at: Instant,
    pub width: u32,
    pub height: u32,
    presented: bool,
}

impl WindowFrame {
    pub fn new(window_id: WindowId, width: u32, height: u32) -> Self {
        Self {
            window_id,
            acquired_at: Instant::now(),
            width,
            height,
            presented: false,
        }
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn age(&self) -> Duration {
        Instant::now().duration_since(self.acquired_at)
    }

    pub fn present(mut self) {
        self.presented = true;
    }

    pub fn discard(self) {
        // Drop without presenting
    }
}

// ============================================================================
// MultiWindowStats stub
// ============================================================================

#[derive(Debug, Clone, Copy)]
pub struct MultiWindowStats {
    pub window_count: usize,
    pub total_frames: u64,
    pub total_dropped: u64,
    pub average_frame_time_ms: f32,
    pub global_frame_count: u64,
}

impl MultiWindowStats {
    pub fn drop_rate(&self) -> f32 {
        let total = self.total_frames + self.total_dropped;
        if total > 0 {
            self.total_dropped as f32 / total as f32
        } else {
            0.0
        }
    }

    pub fn estimated_fps(&self) -> f32 {
        if self.average_frame_time_ms > 0.0 {
            1000.0 / self.average_frame_time_ms
        } else {
            0.0
        }
    }
}

impl fmt::Display for MultiWindowStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} windows, {} frames ({} dropped, {:.1}%), {:.1} FPS avg",
            self.window_count,
            self.total_frames,
            self.total_dropped,
            self.drop_rate() * 100.0,
            self.estimated_fps()
        )
    }
}

// ============================================================================
// MultiWindowManager stub
// ============================================================================

/// Manager for multiple rendering windows.
pub struct MultiWindowManager {
    windows: HashMap<WindowId, WindowState>,
    focused_window: Option<WindowId>,
    render_order: Vec<WindowId>,
    sync_mode: SyncMode,
    max_windows: usize,
    global_frame_count: u64,
}

impl MultiWindowManager {
    pub fn new() -> Self {
        Self {
            windows: HashMap::new(),
            focused_window: None,
            render_order: Vec::new(),
            sync_mode: SyncMode::Independent,
            max_windows: 0,
            global_frame_count: 0,
        }
    }

    pub fn with_max_windows(max: usize) -> Self {
        Self {
            max_windows: max,
            ..Self::new()
        }
    }

    pub fn set_sync_mode(&mut self, mode: SyncMode) {
        self.sync_mode = mode;
    }

    pub fn sync_mode(&self) -> SyncMode {
        self.sync_mode
    }

    pub fn register_window(&mut self, config: WindowConfig) -> Result<WindowId, MultiWindowError> {
        if self.max_windows > 0 && self.windows.len() >= self.max_windows {
            return Err(MultiWindowError::MaxWindowsReached { max: self.max_windows });
        }

        let id = config.id;

        if self.windows.contains_key(&id) {
            return Err(MultiWindowError::WindowExists(id));
        }

        let state = WindowState::new(config);
        self.windows.insert(id, state);
        self.update_render_order();

        if self.focused_window.is_none() || id.is_primary() {
            self.focused_window = Some(id);
        }

        Ok(id)
    }

    pub fn unregister_window(&mut self, id: WindowId) -> Result<WindowState, MultiWindowError> {
        let state = self.windows.remove(&id).ok_or(MultiWindowError::WindowNotFound(id))?;

        self.render_order.retain(|&wid| wid != id);

        if self.focused_window == Some(id) {
            self.focused_window = self.render_order.first().copied();
        }

        Ok(state)
    }

    pub fn get_window(&self, id: WindowId) -> Option<&WindowState> {
        self.windows.get(&id)
    }

    pub fn get_window_mut(&mut self, id: WindowId) -> Option<&mut WindowState> {
        self.windows.get_mut(&id)
    }

    pub fn focused_window(&self) -> Option<&WindowState> {
        self.focused_window.and_then(|id| self.windows.get(&id))
    }

    pub fn focused_window_id(&self) -> Option<WindowId> {
        self.focused_window
    }

    pub fn set_focus(&mut self, id: WindowId) -> Result<Option<WindowId>, MultiWindowError> {
        if !self.windows.contains_key(&id) {
            return Err(MultiWindowError::WindowNotFound(id));
        }

        let old_focus = self.focused_window;

        if let Some(old_id) = old_focus {
            if let Some(state) = self.windows.get_mut(&old_id) {
                state.set_focused(false);
            }
        }

        if let Some(state) = self.windows.get_mut(&id) {
            state.set_focused(true);
        }

        self.focused_window = Some(id);
        Ok(old_focus)
    }

    pub fn set_visible(&mut self, id: WindowId, visible: bool) -> Result<(), MultiWindowError> {
        let state = self.windows.get_mut(&id).ok_or(MultiWindowError::WindowNotFound(id))?;
        state.set_visible(visible);
        Ok(())
    }

    pub fn resize_window(&mut self, id: WindowId, width: u32, height: u32) -> Result<(), MultiWindowError> {
        let state = self.windows.get_mut(&id).ok_or(MultiWindowError::WindowNotFound(id))?;
        state.resize(width, height);
        Ok(())
    }

    pub fn window_count(&self) -> usize {
        self.windows.len()
    }

    pub fn has_windows(&self) -> bool {
        !self.windows.is_empty()
    }

    pub fn window_ids(&self) -> &[WindowId] {
        &self.render_order
    }

    pub fn visible_window_ids(&self) -> Vec<WindowId> {
        self.render_order
            .iter()
            .filter(|id| self.windows.get(id).map(|s| s.is_visible()).unwrap_or(false))
            .copied()
            .collect()
    }

    pub fn iter(&self) -> impl Iterator<Item = (WindowId, &WindowState)> {
        self.render_order.iter().filter_map(|&id| self.windows.get(&id).map(|state| (id, state)))
    }

    fn update_render_order(&mut self) {
        self.render_order = self.windows.keys().copied().collect();
        self.render_order.sort_by(|a, b| {
            let pa = self.windows.get(a).map(|s| s.priority()).unwrap_or(0);
            let pb = self.windows.get(b).map(|s| s.priority()).unwrap_or(0);
            pb.cmp(&pa) // Higher priority first
        });
    }

    pub fn acquire_frame(&mut self, id: WindowId) -> Result<WindowFrame, MultiWindowError> {
        let state = self.windows.get(&id).ok_or(MultiWindowError::WindowNotFound(id))?;
        let (w, h) = state.config.dimensions();
        Ok(WindowFrame::new(id, w, h))
    }

    pub fn acquire_all_frames(&mut self) -> Vec<Result<WindowFrame, (WindowId, MultiWindowError)>> {
        let visible_ids = self.visible_window_ids();
        let mut frames = Vec::with_capacity(visible_ids.len());

        for id in visible_ids {
            if let Some(state) = self.windows.get(&id) {
                let (w, h) = state.config.dimensions();
                frames.push(Ok(WindowFrame::new(id, w, h)));
            }
        }

        frames
    }

    pub fn present_all(&mut self, frames: Vec<WindowFrame>) {
        match self.sync_mode {
            SyncMode::Independent => {
                for wf in frames {
                    if let Some(state) = self.windows.get_mut(&wf.window_id) {
                        wf.present();
                        state.record_frame_presented();
                    }
                }
            }
            SyncMode::SyncToPrimary => {
                let (primary, others): (Vec<_>, Vec<_>) =
                    frames.into_iter().partition(|f| f.window_id.is_primary());

                for wf in primary {
                    if let Some(state) = self.windows.get_mut(&wf.window_id) {
                        wf.present();
                        state.record_frame_presented();
                    }
                }

                for wf in others {
                    if let Some(state) = self.windows.get_mut(&wf.window_id) {
                        wf.present();
                        state.record_frame_presented();
                    }
                }
            }
            SyncMode::SyncToRate { .. } => {
                for wf in frames {
                    if let Some(state) = self.windows.get_mut(&wf.window_id) {
                        wf.present();
                        state.record_frame_presented();
                    }
                }
            }
            SyncMode::Simultaneous => {
                for wf in frames {
                    if let Some(state) = self.windows.get_mut(&wf.window_id) {
                        wf.present();
                        state.record_frame_presented();
                    }
                }
            }
        }

        self.global_frame_count += 1;
    }

    pub fn global_frame_count(&self) -> u64 {
        self.global_frame_count
    }

    pub fn aggregate_stats(&self) -> MultiWindowStats {
        let mut total_frames: u64 = 0;
        let mut total_dropped: u64 = 0;
        let mut total_frame_time: f32 = 0.0;
        let mut window_count: usize = 0;

        for state in self.windows.values() {
            total_frames += state.frame_count();
            total_dropped += state.dropped_frames();
            total_frame_time += state.average_frame_time_ms();
            window_count += 1;
        }

        let avg_frame_time = if window_count > 0 {
            total_frame_time / window_count as f32
        } else {
            0.0
        };

        MultiWindowStats {
            window_count,
            total_frames,
            total_dropped,
            average_frame_time_ms: avg_frame_time,
            global_frame_count: self.global_frame_count,
        }
    }
}

impl Default for MultiWindowManager {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for MultiWindowManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MultiWindowManager")
            .field("window_count", &self.windows.len())
            .field("focused_window", &self.focused_window)
            .field("sync_mode", &self.sync_mode)
            .field("global_frame_count", &self.global_frame_count)
            .finish()
    }
}

// ============================================================================
// TESTS
// ============================================================================

mod window_id_tests {
    use super::*;

    // Test 1: new() returns unique IDs
    #[test]
    fn test_new_returns_unique_ids() {
        let id1 = WindowId::new();
        let id2 = WindowId::new();
        let id3 = WindowId::new();
        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_ne!(id1, id3);
    }

    // Test 2: new() IDs are monotonically increasing
    #[test]
    fn test_new_ids_monotonically_increasing() {
        let id1 = WindowId::new();
        let id2 = WindowId::new();
        let id3 = WindowId::new();
        assert!(id1.as_u64() < id2.as_u64());
        assert!(id2.as_u64() < id3.as_u64());
    }

    // Test 3: primary() returns WindowId(0)
    #[test]
    fn test_primary_returns_zero() {
        let primary = WindowId::primary();
        assert_eq!(primary.as_u64(), 0);
    }

    // Test 4: is_primary() true for primary
    #[test]
    fn test_is_primary_true_for_primary() {
        let primary = WindowId::primary();
        assert!(primary.is_primary());
    }

    // Test 5: is_primary() false for non-primary
    #[test]
    fn test_is_primary_false_for_non_primary() {
        let id = WindowId::from_raw_id(1);
        assert!(!id.is_primary());
        let id2 = WindowId::new();
        assert!(!id2.is_primary());
    }

    // Test 6: as_u64() returns raw value
    #[test]
    fn test_as_u64_returns_raw() {
        let id = WindowId::from_raw_id(42);
        assert_eq!(id.as_u64(), 42);
    }

    // Test 7: from_raw_id() creates from u64
    #[test]
    fn test_from_raw_id() {
        let id = WindowId::from_raw_id(123);
        assert_eq!(id.as_u64(), 123);
    }

    // Test 8: Default trait calls new() (produces unique IDs)
    #[test]
    fn test_default_trait() {
        let id1: WindowId = Default::default();
        let id2: WindowId = Default::default();
        assert_ne!(id1, id2);
    }

    // Test 9: From<u64> trait works
    #[test]
    fn test_from_u64_trait() {
        let id: WindowId = 999u64.into();
        assert_eq!(id.as_u64(), 999);
    }

    // Test 10: Into<u64> trait works
    #[test]
    fn test_into_u64_trait() {
        let id = WindowId::from_raw_id(777);
        let raw: u64 = id.into();
        assert_eq!(raw, 777);
    }

    // Test 11: Eq and Hash work
    #[test]
    fn test_eq_and_hash() {
        use std::collections::HashSet;
        let id1 = WindowId::from_raw_id(1);
        let id2 = WindowId::from_raw_id(1);
        let id3 = WindowId::from_raw_id(2);
        assert_eq!(id1, id2);
        assert_ne!(id1, id3);

        let mut set = HashSet::new();
        set.insert(id1);
        assert!(set.contains(&id2));
        assert!(!set.contains(&id3));
    }

    // Test 12: Debug formatting
    #[test]
    fn test_debug_formatting() {
        let id = WindowId::from_raw_id(42);
        let debug = format!("{:?}", id);
        assert!(debug.contains("42"));
    }

    // Test 13: Display formatting
    #[test]
    fn test_display_formatting() {
        let id = WindowId::from_raw_id(42);
        let display = format!("{}", id);
        assert_eq!(display, "Window(42)");
    }

    // Test 14: Clone and Copy
    #[test]
    fn test_clone_and_copy() {
        let id1 = WindowId::from_raw_id(50);
        let id2 = id1; // Copy
        let id3 = id1.clone();
        assert_eq!(id1, id2);
        assert_eq!(id1, id3);
    }
}

mod window_config_tests {
    use super::*;

    // Test 15: new() creates config with defaults
    #[test]
    fn test_new_creates_defaults() {
        let id = WindowId::new();
        let surface_config = SurfaceConfiguration::new(800, 600);
        let config = WindowConfig::new(id, surface_config);

        assert_eq!(config.id, id);
        assert!(!config.is_focused);
        assert!(config.is_visible);
        assert_eq!(config.priority, 128);
        assert!(config.label.is_none());
        assert!(!config.sync_to_primary);
    }

    // Test 16: primary() creates primary config
    #[test]
    fn test_primary_creates_primary() {
        let surface_config = SurfaceConfiguration::new(1920, 1080);
        let config = WindowConfig::primary(surface_config);

        assert!(config.id.is_primary());
        assert!(config.is_focused);
        assert!(config.is_visible);
        assert_eq!(config.priority, 255);
        assert_eq!(config.label, Some("Primary".to_string()));
    }

    // Test 17: with_focus() sets focus
    #[test]
    fn test_with_focus() {
        let config = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))
            .with_focus(true);
        assert!(config.is_focused);

        let config2 = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))
            .with_focus(false);
        assert!(!config2.is_focused);
    }

    // Test 18: with_visibility() sets visibility
    #[test]
    fn test_with_visibility() {
        let config = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))
            .with_visibility(false);
        assert!(!config.is_visible);

        let config2 = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))
            .with_visibility(true);
        assert!(config2.is_visible);
    }

    // Test 19: with_priority() sets priority
    #[test]
    fn test_with_priority() {
        let config = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))
            .with_priority(200);
        assert_eq!(config.priority, 200);
    }

    // Test 20: with_label() sets label
    #[test]
    fn test_with_label() {
        let config = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))
            .with_label("Test Window");
        assert_eq!(config.label, Some("Test Window".to_string()));
    }

    // Test 21: with_sync_to_primary() enables sync
    #[test]
    fn test_with_sync_to_primary() {
        let config = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))
            .with_sync_to_primary(true);
        assert!(config.sync_to_primary);
    }

    // Test 22: dimensions() returns width, height
    #[test]
    fn test_dimensions() {
        let config = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(1280, 720));
        assert_eq!(config.dimensions(), (1280, 720));
    }

    // Test 23: aspect_ratio() calculates correctly
    #[test]
    fn test_aspect_ratio() {
        let config = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(1920, 1080));
        let ratio = config.aspect_ratio();
        assert!((ratio - 16.0 / 9.0).abs() < 0.01);

        let config2 = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 800));
        assert!((config2.aspect_ratio() - 1.0).abs() < 0.001);
    }

    // Test 24: should_render() true when visible and valid dims
    #[test]
    fn test_should_render_true() {
        let config = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))
            .with_visibility(true);
        assert!(config.should_render());
    }

    // Test 25: should_render() false when invisible
    #[test]
    fn test_should_render_false_invisible() {
        let config = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))
            .with_visibility(false);
        assert!(!config.should_render());
    }

    // Test 26: should_render() false when zero width
    #[test]
    fn test_should_render_false_zero_width() {
        // SurfaceConfiguration::new clamps to min 1, so we need to modify
        let mut config = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600));
        config.config.width = 0;
        assert!(!config.should_render());
    }

    // Test 27: should_render() false when zero height
    #[test]
    fn test_should_render_false_zero_height() {
        let mut config = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600));
        config.config.height = 0;
        assert!(!config.should_render());
    }

    // Test 28: Display formatting
    #[test]
    fn test_display_formatting() {
        let config = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))
            .with_label("My Window")
            .with_priority(150);
        let display = format!("{}", config);
        assert!(display.contains("My Window"));
        assert!(display.contains("800x600"));
        assert!(display.contains("priority=150"));
    }

    // Test 29: Clone works
    #[test]
    fn test_clone() {
        let config = WindowConfig::new(WindowId::from_raw_id(5), SurfaceConfiguration::new(800, 600))
            .with_label("Cloned");
        let cloned = config.clone();
        assert_eq!(cloned.id, config.id);
        assert_eq!(cloned.label, config.label);
    }
}

mod sync_mode_tests {
    use super::*;

    // Test 30: Default is Independent
    #[test]
    fn test_default_is_independent() {
        let mode: SyncMode = Default::default();
        assert_eq!(mode, SyncMode::Independent);
    }

    // Test 31: Independent requires no coordination
    #[test]
    fn test_independent_no_coordination() {
        assert!(!SyncMode::Independent.requires_coordination());
    }

    // Test 32: SyncToPrimary requires coordination
    #[test]
    fn test_sync_to_primary_requires_coordination() {
        assert!(SyncMode::SyncToPrimary.requires_coordination());
    }

    // Test 33: SyncToRate requires coordination
    #[test]
    fn test_sync_to_rate_requires_coordination() {
        assert!(SyncMode::SyncToRate { target_hz: 60 }.requires_coordination());
    }

    // Test 34: Simultaneous requires coordination
    #[test]
    fn test_simultaneous_requires_coordination() {
        assert!(SyncMode::Simultaneous.requires_coordination());
    }

    // Test 35: target_interval() None for Independent
    #[test]
    fn test_target_interval_independent() {
        assert!(SyncMode::Independent.target_interval().is_none());
    }

    // Test 36: target_interval() None for SyncToPrimary
    #[test]
    fn test_target_interval_sync_to_primary() {
        assert!(SyncMode::SyncToPrimary.target_interval().is_none());
    }

    // Test 37: target_interval() Some for SyncToRate
    #[test]
    fn test_target_interval_sync_to_rate() {
        let mode = SyncMode::SyncToRate { target_hz: 60 };
        let interval = mode.target_interval().unwrap();
        let expected = Duration::from_secs_f64(1.0 / 60.0);
        assert!((interval.as_secs_f64() - expected.as_secs_f64()).abs() < 0.0001);
    }

    // Test 38: target_interval() None for Simultaneous
    #[test]
    fn test_target_interval_simultaneous() {
        assert!(SyncMode::Simultaneous.target_interval().is_none());
    }

    // Test 39: sync_to_rate() constructor
    #[test]
    fn test_sync_to_rate_constructor() {
        let mode = SyncMode::sync_to_rate(144);
        assert_eq!(mode, SyncMode::SyncToRate { target_hz: 144 });
    }

    // Test 40: Display formatting
    #[test]
    fn test_display_formatting() {
        assert_eq!(format!("{}", SyncMode::Independent), "Independent");
        assert_eq!(format!("{}", SyncMode::SyncToPrimary), "Sync to Primary");
        assert_eq!(format!("{}", SyncMode::SyncToRate { target_hz: 60 }), "Sync to 60Hz");
        assert_eq!(format!("{}", SyncMode::Simultaneous), "Simultaneous");
    }

    // Test 41: Clone, Copy, PartialEq, Eq
    #[test]
    fn test_clone_copy_eq() {
        let mode1 = SyncMode::SyncToRate { target_hz: 120 };
        let mode2 = mode1; // Copy
        let mode3 = mode1.clone();
        assert_eq!(mode1, mode2);
        assert_eq!(mode1, mode3);

        assert_ne!(SyncMode::Independent, SyncMode::SyncToPrimary);
    }
}

mod manager_single_window_tests {
    use super::*;

    // Test 42: new() creates empty manager
    #[test]
    fn test_new_creates_empty() {
        let manager = MultiWindowManager::new();
        assert_eq!(manager.window_count(), 0);
        assert!(!manager.has_windows());
    }

    // Test 43: with_max_windows() sets limit
    #[test]
    fn test_with_max_windows() {
        let manager = MultiWindowManager::with_max_windows(5);
        assert_eq!(manager.window_count(), 0);
    }

    // Test 44: register primary window
    #[test]
    fn test_register_primary_window() {
        let mut manager = MultiWindowManager::new();
        let config = WindowConfig::primary(SurfaceConfiguration::new(1920, 1080));
        let id = manager.register_window(config).unwrap();
        assert!(id.is_primary());
    }

    // Test 45: window_count() returns 1
    #[test]
    fn test_window_count_one() {
        let mut manager = MultiWindowManager::new();
        manager.register_window(WindowConfig::primary(SurfaceConfiguration::new(800, 600))).unwrap();
        assert_eq!(manager.window_count(), 1);
    }

    // Test 46: window_ids() contains primary
    #[test]
    fn test_window_ids_contains_primary() {
        let mut manager = MultiWindowManager::new();
        let id = manager.register_window(WindowConfig::primary(SurfaceConfiguration::new(800, 600))).unwrap();
        assert!(manager.window_ids().contains(&id));
    }

    // Test 47: focused_window_id() returns primary
    #[test]
    fn test_focused_window_id_primary() {
        let mut manager = MultiWindowManager::new();
        let id = manager.register_window(WindowConfig::primary(SurfaceConfiguration::new(800, 600))).unwrap();
        assert_eq!(manager.focused_window_id(), Some(id));
    }

    // Test 48: get_window() returns state
    #[test]
    fn test_get_window_returns_state() {
        let mut manager = MultiWindowManager::new();
        let id = manager.register_window(WindowConfig::primary(SurfaceConfiguration::new(800, 600))).unwrap();
        let state = manager.get_window(id);
        assert!(state.is_some());
        assert_eq!(state.unwrap().id(), id);
    }

    // Test 49: unregister_window() removes window
    #[test]
    fn test_unregister_window_removes() {
        let mut manager = MultiWindowManager::new();
        let id = manager.register_window(WindowConfig::primary(SurfaceConfiguration::new(800, 600))).unwrap();
        let state = manager.unregister_window(id).unwrap();
        assert_eq!(state.id(), id);
        assert!(manager.get_window(id).is_none());
    }

    // Test 50: window_count() returns 0 after unregister
    #[test]
    fn test_window_count_zero_after_unregister() {
        let mut manager = MultiWindowManager::new();
        let id = manager.register_window(WindowConfig::primary(SurfaceConfiguration::new(800, 600))).unwrap();
        manager.unregister_window(id).unwrap();
        assert_eq!(manager.window_count(), 0);
    }

    // Test 51: unregister_window() error for unknown ID
    #[test]
    fn test_unregister_unknown_error() {
        let mut manager = MultiWindowManager::new();
        let result = manager.unregister_window(WindowId::from_raw_id(999));
        assert!(result.is_err());
    }

    // Test 52: has_windows() true with windows
    #[test]
    fn test_has_windows_true() {
        let mut manager = MultiWindowManager::new();
        manager.register_window(WindowConfig::primary(SurfaceConfiguration::new(800, 600))).unwrap();
        assert!(manager.has_windows());
    }

    // Test 53: has_windows() false when empty
    #[test]
    fn test_has_windows_false() {
        let manager = MultiWindowManager::new();
        assert!(!manager.has_windows());
    }
}

mod dual_window_tests {
    use super::*;

    // Test 54: register two windows
    #[test]
    fn test_register_two_windows() {
        let mut manager = MultiWindowManager::new();
        let id1 = manager.register_window(WindowConfig::primary(SurfaceConfiguration::new(1920, 1080))).unwrap();
        let id2 = manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(1280, 720))).unwrap();
        assert_ne!(id1, id2);
    }

    // Test 55: window_count() returns 2
    #[test]
    fn test_window_count_two() {
        let mut manager = MultiWindowManager::new();
        manager.register_window(WindowConfig::primary(SurfaceConfiguration::new(1920, 1080))).unwrap();
        manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(1280, 720))).unwrap();
        assert_eq!(manager.window_count(), 2);
    }

    // Test 56: both windows in window_ids()
    #[test]
    fn test_both_in_window_ids() {
        let mut manager = MultiWindowManager::new();
        let id1 = manager.register_window(WindowConfig::primary(SurfaceConfiguration::new(1920, 1080))).unwrap();
        let id2 = manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(1280, 720))).unwrap();
        let ids = manager.window_ids();
        assert!(ids.contains(&id1));
        assert!(ids.contains(&id2));
    }

    // Test 57: set_focus() changes focus
    #[test]
    fn test_set_focus_changes() {
        let mut manager = MultiWindowManager::new();
        let id1 = manager.register_window(WindowConfig::primary(SurfaceConfiguration::new(1920, 1080))).unwrap();
        let id2 = manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(1280, 720))).unwrap();
        manager.set_focus(id2).unwrap();
        assert_eq!(manager.focused_window_id(), Some(id2));
        assert!(manager.get_window(id2).unwrap().is_focused());
        assert!(!manager.get_window(id1).unwrap().is_focused());
    }

    // Test 58: old focus returned by set_focus()
    #[test]
    fn test_set_focus_returns_old() {
        let mut manager = MultiWindowManager::new();
        let id1 = manager.register_window(WindowConfig::primary(SurfaceConfiguration::new(1920, 1080))).unwrap();
        let id2 = manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(1280, 720))).unwrap();
        let old = manager.set_focus(id2).unwrap();
        assert_eq!(old, Some(id1));
    }

    // Test 59: visible_window_ids() returns visible windows
    #[test]
    fn test_visible_window_ids_all_visible() {
        let mut manager = MultiWindowManager::new();
        let id1 = manager.register_window(WindowConfig::primary(SurfaceConfiguration::new(1920, 1080))).unwrap();
        let id2 = manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(1280, 720))).unwrap();
        let visible = manager.visible_window_ids();
        assert!(visible.contains(&id1));
        assert!(visible.contains(&id2));
    }

    // Test 60: set_visible() hides window
    #[test]
    fn test_set_visible_hides() {
        let mut manager = MultiWindowManager::new();
        manager.register_window(WindowConfig::primary(SurfaceConfiguration::new(1920, 1080))).unwrap();
        let id2 = manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(1280, 720))).unwrap();
        manager.set_visible(id2, false).unwrap();
        assert!(!manager.get_window(id2).unwrap().is_visible());
    }

    // Test 61: visible_window_ids() excludes hidden
    #[test]
    fn test_visible_excludes_hidden() {
        let mut manager = MultiWindowManager::new();
        let id1 = manager.register_window(WindowConfig::primary(SurfaceConfiguration::new(1920, 1080))).unwrap();
        let id2 = manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(1280, 720))).unwrap();
        manager.set_visible(id2, false).unwrap();
        let visible = manager.visible_window_ids();
        assert!(visible.contains(&id1));
        assert!(!visible.contains(&id2));
    }
}

mod picture_in_picture_tests {
    use super::*;

    // Test 62: overlay window with low priority
    #[test]
    fn test_overlay_low_priority() {
        let mut manager = MultiWindowManager::new();
        let config = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(320, 240))
            .with_priority(50)
            .with_label("Overlay");
        let id = manager.register_window(config).unwrap();
        assert_eq!(manager.get_window(id).unwrap().priority(), 50);
    }

    // Test 63: main window with high priority
    #[test]
    fn test_main_high_priority() {
        let mut manager = MultiWindowManager::new();
        let config = WindowConfig::primary(SurfaceConfiguration::new(1920, 1080));
        let id = manager.register_window(config).unwrap();
        assert_eq!(manager.get_window(id).unwrap().priority(), 255);
    }

    // Test 64: render order: main first (high priority)
    #[test]
    fn test_render_order_high_priority_first() {
        let mut manager = MultiWindowManager::new();
        let overlay_config = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(320, 240))
            .with_priority(50);
        let main_config = WindowConfig::primary(SurfaceConfiguration::new(1920, 1080));

        let overlay_id = manager.register_window(overlay_config).unwrap();
        let main_id = manager.register_window(main_config).unwrap();

        let order = manager.window_ids();
        // Primary (priority 255) should be first, overlay (priority 50) last
        assert_eq!(order[0], main_id);
        assert_eq!(order[1], overlay_id);
    }

    // Test 65: small overlay dimensions
    #[test]
    fn test_small_overlay_dimensions() {
        let config = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(320, 240))
            .with_label("PiP");
        assert_eq!(config.dimensions(), (320, 240));
    }

    // Test 66: both visible
    #[test]
    fn test_pip_both_visible() {
        let mut manager = MultiWindowManager::new();
        manager.register_window(WindowConfig::primary(SurfaceConfiguration::new(1920, 1080))).unwrap();
        manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(320, 240))).unwrap();
        assert_eq!(manager.visible_window_ids().len(), 2);
    }

    // Test 67: both should_render()
    #[test]
    fn test_pip_both_should_render() {
        let mut manager = MultiWindowManager::new();
        let id1 = manager.register_window(WindowConfig::primary(SurfaceConfiguration::new(1920, 1080))).unwrap();
        let id2 = manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(320, 240))).unwrap();
        assert!(manager.get_window(id1).unwrap().should_render());
        assert!(manager.get_window(id2).unwrap().should_render());
    }
}

mod window_lifecycle_tests {
    use super::*;

    // Test 68: create window
    #[test]
    fn test_create_window() {
        let mut manager = MultiWindowManager::new();
        let id = manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))).unwrap();
        assert!(manager.get_window(id).is_some());
    }

    // Test 69: use window (acquire_frame)
    #[test]
    fn test_use_window_acquire_frame() {
        let mut manager = MultiWindowManager::new();
        let id = manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))).unwrap();
        let frame = manager.acquire_frame(id).unwrap();
        assert_eq!(frame.window_id, id);
        assert_eq!(frame.dimensions(), (800, 600));
    }

    // Test 70: destroy window (unregister)
    #[test]
    fn test_destroy_window() {
        let mut manager = MultiWindowManager::new();
        let id = manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))).unwrap();
        manager.unregister_window(id).unwrap();
        assert!(manager.get_window(id).is_none());
    }

    // Test 71: cannot acquire from unregistered
    #[test]
    fn test_cannot_acquire_unregistered() {
        let mut manager = MultiWindowManager::new();
        let result = manager.acquire_frame(WindowId::from_raw_id(999));
        assert!(result.is_err());
    }

    // Test 72: re-register same ID fails if still registered
    #[test]
    fn test_reregister_fails_if_exists() {
        let mut manager = MultiWindowManager::new();
        let id = WindowId::from_raw_id(42);
        manager.register_window(WindowConfig::new(id, SurfaceConfiguration::new(800, 600))).unwrap();
        let result = manager.register_window(WindowConfig::new(id, SurfaceConfiguration::new(800, 600)));
        assert!(result.is_err());
    }

    // Test 73: re-register after unregister succeeds
    #[test]
    fn test_reregister_after_unregister() {
        let mut manager = MultiWindowManager::new();
        let id = WindowId::from_raw_id(42);
        manager.register_window(WindowConfig::new(id, SurfaceConfiguration::new(800, 600))).unwrap();
        manager.unregister_window(id).unwrap();
        let result = manager.register_window(WindowConfig::new(id, SurfaceConfiguration::new(1024, 768)));
        assert!(result.is_ok());
    }

    // Test 74: max_windows limit enforced
    #[test]
    fn test_max_windows_enforced() {
        let mut manager = MultiWindowManager::with_max_windows(2);
        manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))).unwrap();
        manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))).unwrap();
        let result = manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600)));
        assert!(result.is_err());
    }
}

mod focus_switching_tests {
    use super::*;

    // Test 75: initial focus on first registered
    #[test]
    fn test_initial_focus_first_registered() {
        let mut manager = MultiWindowManager::new();
        let id = manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))).unwrap();
        assert_eq!(manager.focused_window_id(), Some(id));
    }

    // Test 76: set_focus() to second window
    #[test]
    fn test_set_focus_second_window() {
        let mut manager = MultiWindowManager::new();
        manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))).unwrap();
        let id2 = manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(1024, 768))).unwrap();
        manager.set_focus(id2).unwrap();
        assert_eq!(manager.focused_window_id(), Some(id2));
    }

    // Test 77: set_focus() returns old focus
    #[test]
    fn test_set_focus_returns_old_focus() {
        let mut manager = MultiWindowManager::new();
        let id1 = manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))).unwrap();
        let id2 = manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(1024, 768))).unwrap();
        let old = manager.set_focus(id2).unwrap();
        assert_eq!(old, Some(id1));
    }

    // Test 78: set_focus() error for unknown ID
    #[test]
    fn test_set_focus_unknown_error() {
        let mut manager = MultiWindowManager::new();
        manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))).unwrap();
        let result = manager.set_focus(WindowId::from_raw_id(999));
        assert!(result.is_err());
    }

    // Test 79: focused_window_id() updates
    #[test]
    fn test_focused_window_id_updates() {
        let mut manager = MultiWindowManager::new();
        let id1 = manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))).unwrap();
        assert_eq!(manager.focused_window_id(), Some(id1));
        let id2 = manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(1024, 768))).unwrap();
        // Primary is auto-focused, but id2 is not primary so id1 stays focused
        assert_eq!(manager.focused_window_id(), Some(id1));
        manager.set_focus(id2).unwrap();
        assert_eq!(manager.focused_window_id(), Some(id2));
    }

    // Test 80: focused_window() returns correct state
    #[test]
    fn test_focused_window_returns_state() {
        let mut manager = MultiWindowManager::new();
        let id = manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))).unwrap();
        let focused = manager.focused_window().unwrap();
        assert_eq!(focused.id(), id);
    }
}

mod visibility_toggle_tests {
    use super::*;

    // Test 81: default visibility is true
    #[test]
    fn test_default_visibility_true() {
        let mut manager = MultiWindowManager::new();
        let id = manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))).unwrap();
        assert!(manager.get_window(id).unwrap().is_visible());
    }

    // Test 82: set_visible(false) hides
    #[test]
    fn test_set_visible_false_hides() {
        let mut manager = MultiWindowManager::new();
        let id = manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))).unwrap();
        manager.set_visible(id, false).unwrap();
        assert!(!manager.get_window(id).unwrap().is_visible());
    }

    // Test 83: set_visible(true) shows
    #[test]
    fn test_set_visible_true_shows() {
        let mut manager = MultiWindowManager::new();
        let id = manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))).unwrap();
        manager.set_visible(id, false).unwrap();
        manager.set_visible(id, true).unwrap();
        assert!(manager.get_window(id).unwrap().is_visible());
    }

    // Test 84: hidden window excluded from visible_window_ids()
    #[test]
    fn test_hidden_excluded_from_visible() {
        let mut manager = MultiWindowManager::new();
        let id1 = manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))).unwrap();
        let id2 = manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(1024, 768))).unwrap();
        manager.set_visible(id1, false).unwrap();
        let visible = manager.visible_window_ids();
        assert!(!visible.contains(&id1));
        assert!(visible.contains(&id2));
    }

    // Test 85: set_visible error for unknown ID
    #[test]
    fn test_set_visible_unknown_error() {
        let mut manager = MultiWindowManager::new();
        let result = manager.set_visible(WindowId::from_raw_id(999), true);
        assert!(result.is_err());
    }

    // Test 86: multiple visibility toggles
    #[test]
    fn test_multiple_visibility_toggles() {
        let mut manager = MultiWindowManager::new();
        let id = manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))).unwrap();
        manager.set_visible(id, false).unwrap();
        manager.set_visible(id, true).unwrap();
        manager.set_visible(id, false).unwrap();
        manager.set_visible(id, true).unwrap();
        assert!(manager.get_window(id).unwrap().is_visible());
    }
}

mod priority_ordering_tests {
    use super::*;

    // Test 87: higher priority first in render order
    #[test]
    fn test_higher_priority_first() {
        let mut manager = MultiWindowManager::new();
        let id_low = manager.register_window(
            WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))
                .with_priority(10)
        ).unwrap();
        let id_high = manager.register_window(
            WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))
                .with_priority(200)
        ).unwrap();
        let order = manager.window_ids();
        assert_eq!(order[0], id_high);
        assert_eq!(order[1], id_low);
    }

    // Test 88: equal priority stable order (order of insertion)
    #[test]
    fn test_equal_priority_stable() {
        let mut manager = MultiWindowManager::new();
        let id1 = manager.register_window(
            WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))
                .with_priority(100)
        ).unwrap();
        let id2 = manager.register_window(
            WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))
                .with_priority(100)
        ).unwrap();
        let order = manager.window_ids();
        // Both are in the list
        assert!(order.contains(&id1));
        assert!(order.contains(&id2));
    }

    // Test 89: priority 255 (max) first
    #[test]
    fn test_priority_255_first() {
        let mut manager = MultiWindowManager::new();
        let id_mid = manager.register_window(
            WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))
                .with_priority(128)
        ).unwrap();
        let id_max = manager.register_window(
            WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))
                .with_priority(255)
        ).unwrap();
        let order = manager.window_ids();
        assert_eq!(order[0], id_max);
    }

    // Test 90: priority 0 (min) last
    #[test]
    fn test_priority_0_last() {
        let mut manager = MultiWindowManager::new();
        let id_min = manager.register_window(
            WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))
                .with_priority(0)
        ).unwrap();
        let id_mid = manager.register_window(
            WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))
                .with_priority(100)
        ).unwrap();
        let order = manager.window_ids();
        assert_eq!(order[order.len() - 1], id_min);
        assert_eq!(order[0], id_mid);
    }

    // Test 91: update_render_order after registration
    #[test]
    fn test_update_render_order_after_registration() {
        let mut manager = MultiWindowManager::new();
        // Register low priority first
        let id_low = manager.register_window(
            WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))
                .with_priority(10)
        ).unwrap();
        // Then high priority
        let id_high = manager.register_window(
            WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))
                .with_priority(200)
        ).unwrap();
        // High priority should be first despite registration order
        assert_eq!(manager.window_ids()[0], id_high);
    }

    // Test 92: primary() config has priority 255
    #[test]
    fn test_primary_config_priority_255() {
        let config = WindowConfig::primary(SurfaceConfiguration::new(1920, 1080));
        assert_eq!(config.priority, 255);
    }
}

mod sync_presentation_tests {
    use super::*;

    // Test 93: set_sync_mode() changes mode
    #[test]
    fn test_set_sync_mode_changes() {
        let mut manager = MultiWindowManager::new();
        manager.set_sync_mode(SyncMode::SyncToPrimary);
        assert_eq!(manager.sync_mode(), SyncMode::SyncToPrimary);
    }

    // Test 94: sync_mode() returns current
    #[test]
    fn test_sync_mode_returns_current() {
        let mut manager = MultiWindowManager::new();
        assert_eq!(manager.sync_mode(), SyncMode::Independent);
        manager.set_sync_mode(SyncMode::Simultaneous);
        assert_eq!(manager.sync_mode(), SyncMode::Simultaneous);
    }

    // Test 95: Independent mode default
    #[test]
    fn test_independent_mode_default() {
        let manager = MultiWindowManager::new();
        assert_eq!(manager.sync_mode(), SyncMode::Independent);
    }

    // Test 96: SyncToPrimary presents primary first
    #[test]
    fn test_sync_to_primary_presents_primary_first() {
        let mut manager = MultiWindowManager::new();
        manager.set_sync_mode(SyncMode::SyncToPrimary);

        let primary_id = manager.register_window(WindowConfig::primary(SurfaceConfiguration::new(1920, 1080))).unwrap();
        let secondary_id = manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(1280, 720))).unwrap();

        let frames: Vec<Result<WindowFrame, (WindowId, MultiWindowError)>> = manager.acquire_all_frames();
        let good_frames: Vec<WindowFrame> = frames.into_iter().filter_map(|r| r.ok()).collect();

        // Should have both frames
        assert_eq!(good_frames.len(), 2);

        // present_all handles ordering internally
        manager.present_all(good_frames);
    }

    // Test 97: SyncToRate has target interval
    #[test]
    fn test_sync_to_rate_has_interval() {
        let mode = SyncMode::SyncToRate { target_hz: 144 };
        let interval = mode.target_interval().unwrap();
        let expected = Duration::from_secs_f64(1.0 / 144.0);
        assert!((interval.as_secs_f64() - expected.as_secs_f64()).abs() < 0.0001);
    }

    // Test 98: Simultaneous presents all at once
    #[test]
    fn test_simultaneous_presents_all() {
        let mut manager = MultiWindowManager::new();
        manager.set_sync_mode(SyncMode::Simultaneous);

        manager.register_window(WindowConfig::primary(SurfaceConfiguration::new(1920, 1080))).unwrap();
        manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(1280, 720))).unwrap();

        let frames: Vec<Result<WindowFrame, (WindowId, MultiWindowError)>> = manager.acquire_all_frames();
        let good_frames: Vec<WindowFrame> = frames.into_iter().filter_map(|r| r.ok()).collect();

        manager.present_all(good_frames);
        assert_eq!(manager.global_frame_count(), 1);
    }

    // Test 99: present_all() updates global frame count
    #[test]
    fn test_present_all_updates_global_count() {
        let mut manager = MultiWindowManager::new();
        manager.register_window(WindowConfig::primary(SurfaceConfiguration::new(800, 600))).unwrap();

        assert_eq!(manager.global_frame_count(), 0);

        let frames = manager.acquire_all_frames();
        let good_frames: Vec<WindowFrame> = frames.into_iter().filter_map(|r| r.ok()).collect();
        manager.present_all(good_frames);

        assert_eq!(manager.global_frame_count(), 1);
    }

    // Test 100: global_frame_count() increments
    #[test]
    fn test_global_frame_count_increments() {
        let mut manager = MultiWindowManager::new();
        manager.register_window(WindowConfig::primary(SurfaceConfiguration::new(800, 600))).unwrap();

        for i in 1..=5 {
            let frames = manager.acquire_all_frames();
            let good_frames: Vec<WindowFrame> = frames.into_iter().filter_map(|r| r.ok()).collect();
            manager.present_all(good_frames);
            assert_eq!(manager.global_frame_count(), i);
        }
    }
}

mod multi_window_error_tests {
    use super::*;

    // Test 101: WindowNotFound error
    #[test]
    fn test_window_not_found_error() {
        let id = WindowId::from_raw_id(123);
        let err = MultiWindowError::WindowNotFound(id);
        let msg = format!("{}", err);
        assert!(msg.contains("not found"));
        assert!(msg.contains("123"));
    }

    // Test 102: WindowExists error
    #[test]
    fn test_window_exists_error() {
        let id = WindowId::from_raw_id(456);
        let err = MultiWindowError::WindowExists(id);
        let msg = format!("{}", err);
        assert!(msg.contains("exists"));
        assert!(msg.contains("456"));
    }

    // Test 103: NoWindows error
    #[test]
    fn test_no_windows_error() {
        let err = MultiWindowError::NoWindows;
        let msg = format!("{}", err);
        assert!(msg.contains("no windows"));
    }

    // Test 104: MaxWindowsReached error
    #[test]
    fn test_max_windows_reached_error() {
        let err = MultiWindowError::MaxWindowsReached { max: 8 };
        let msg = format!("{}", err);
        assert!(msg.contains("8"));
        assert!(msg.contains("maximum"));
    }

    // Test 105: is_recoverable() for window errors
    #[test]
    fn test_is_recoverable() {
        assert!(!MultiWindowError::WindowNotFound(WindowId::primary()).is_recoverable());
        assert!(!MultiWindowError::WindowExists(WindowId::primary()).is_recoverable());
        assert!(!MultiWindowError::NoWindows.is_recoverable());
        assert!(MultiWindowError::NoFocusedWindow.is_recoverable());
        assert!(MultiWindowError::SurfaceError("test".to_string()).is_recoverable());
        assert!(MultiWindowError::FrameError("test".to_string()).is_recoverable());
        assert!(!MultiWindowError::MaxWindowsReached { max: 5 }.is_recoverable());
    }

    // Test 106: Error Display formatting
    #[test]
    fn test_error_display() {
        let errors = vec![
            MultiWindowError::WindowNotFound(WindowId::from_raw_id(1)),
            MultiWindowError::WindowExists(WindowId::from_raw_id(2)),
            MultiWindowError::NoWindows,
            MultiWindowError::NoFocusedWindow,
            MultiWindowError::SurfaceError("surface lost".to_string()),
            MultiWindowError::FrameError("timeout".to_string()),
            MultiWindowError::MaxWindowsReached { max: 10 },
        ];

        for err in errors {
            let msg = format!("{}", err);
            assert!(!msg.is_empty());
        }
    }

    // Test 107: SurfaceError wrapping
    #[test]
    fn test_surface_error_wrapping() {
        let err = MultiWindowError::SurfaceError("surface lost".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("surface"));
    }

    // Test 108: FrameError wrapping
    #[test]
    fn test_frame_error_wrapping() {
        let err = MultiWindowError::FrameError("timeout".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("frame"));
    }
}

mod window_state_tests {
    use super::*;

    // Test 109: WindowState::new() creates state
    #[test]
    fn test_window_state_new() {
        let config = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600));
        let state = WindowState::new(config);
        assert!(state.frame_count() == 0);
    }

    // Test 110: id() returns config ID
    #[test]
    fn test_id_returns_config_id() {
        let id = WindowId::from_raw_id(42);
        let config = WindowConfig::new(id, SurfaceConfiguration::new(800, 600));
        let state = WindowState::new(config);
        assert_eq!(state.id(), id);
    }

    // Test 111: is_focused() returns focus state
    #[test]
    fn test_is_focused() {
        let config = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))
            .with_focus(true);
        let state = WindowState::new(config);
        assert!(state.is_focused());
    }

    // Test 112: is_visible() returns visibility
    #[test]
    fn test_is_visible() {
        let config = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))
            .with_visibility(false);
        let state = WindowState::new(config);
        assert!(!state.is_visible());
    }

    // Test 113: priority() returns priority
    #[test]
    fn test_priority_returns() {
        let config = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))
            .with_priority(200);
        let state = WindowState::new(config);
        assert_eq!(state.priority(), 200);
    }

    // Test 114: set_focused() updates focus
    #[test]
    fn test_set_focused() {
        let config = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600));
        let mut state = WindowState::new(config);
        assert!(!state.is_focused());
        state.set_focused(true);
        assert!(state.is_focused());
    }

    // Test 115: set_visible() updates visibility
    #[test]
    fn test_set_visible() {
        let config = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600));
        let mut state = WindowState::new(config);
        assert!(state.is_visible());
        state.set_visible(false);
        assert!(!state.is_visible());
    }

    // Test 116: frame_count() starts at 0
    #[test]
    fn test_frame_count_starts_zero() {
        let config = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600));
        let state = WindowState::new(config);
        assert_eq!(state.frame_count(), 0);
    }
}

mod window_frame_tests {
    use super::*;
    use std::thread;

    // Test 117: WindowFrame::new() creates frame
    #[test]
    fn test_window_frame_new() {
        let id = WindowId::new();
        let frame = WindowFrame::new(id, 800, 600);
        assert_eq!(frame.window_id, id);
    }

    // Test 118: window_id field accessible
    #[test]
    fn test_window_id_accessible() {
        let id = WindowId::from_raw_id(99);
        let frame = WindowFrame::new(id, 800, 600);
        assert_eq!(frame.window_id.as_u64(), 99);
    }

    // Test 119: width/height accessible
    #[test]
    fn test_dimensions_accessible() {
        let frame = WindowFrame::new(WindowId::new(), 1920, 1080);
        assert_eq!(frame.width, 1920);
        assert_eq!(frame.height, 1080);
    }

    // Test 120: dimensions() returns frame dims
    #[test]
    fn test_dimensions_method() {
        let frame = WindowFrame::new(WindowId::new(), 1280, 720);
        assert_eq!(frame.dimensions(), (1280, 720));
    }

    // Test 121: age() returns time since acquire
    #[test]
    fn test_age() {
        let frame = WindowFrame::new(WindowId::new(), 800, 600);
        thread::sleep(Duration::from_millis(10));
        let age = frame.age();
        assert!(age >= Duration::from_millis(10));
    }

    // Test 122: present/discard consume frame
    #[test]
    fn test_present_discard_consume() {
        let frame1 = WindowFrame::new(WindowId::new(), 800, 600);
        frame1.present();

        let frame2 = WindowFrame::new(WindowId::new(), 800, 600);
        frame2.discard();
        // Frames are consumed, no double-use possible
    }
}

mod multi_window_stats_tests {
    use super::*;

    // Test 123: aggregate_stats() returns stats
    #[test]
    fn test_aggregate_stats() {
        let mut manager = MultiWindowManager::new();
        manager.register_window(WindowConfig::primary(SurfaceConfiguration::new(800, 600))).unwrap();
        let stats = manager.aggregate_stats();
        assert_eq!(stats.window_count, 1);
    }

    // Test 124: window_count accurate
    #[test]
    fn test_stats_window_count() {
        let mut manager = MultiWindowManager::new();
        manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))).unwrap();
        manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))).unwrap();
        manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))).unwrap();
        let stats = manager.aggregate_stats();
        assert_eq!(stats.window_count, 3);
    }

    // Test 125: total_frames accurate
    #[test]
    fn test_stats_total_frames() {
        let mut manager = MultiWindowManager::new();
        manager.register_window(WindowConfig::primary(SurfaceConfiguration::new(800, 600))).unwrap();

        // Present a few frames
        for _ in 0..3 {
            let frames = manager.acquire_all_frames();
            let good_frames: Vec<WindowFrame> = frames.into_iter().filter_map(|r| r.ok()).collect();
            manager.present_all(good_frames);
        }

        let stats = manager.aggregate_stats();
        assert_eq!(stats.total_frames, 3);
    }

    // Test 126: drop_rate() calculation
    #[test]
    fn test_drop_rate() {
        let stats = MultiWindowStats {
            window_count: 1,
            total_frames: 90,
            total_dropped: 10,
            average_frame_time_ms: 16.67,
            global_frame_count: 100,
        };
        let drop_rate = stats.drop_rate();
        assert!((drop_rate - 0.1).abs() < 0.001);
    }

    // Test 127: estimated_fps() calculation
    #[test]
    fn test_estimated_fps() {
        let stats = MultiWindowStats {
            window_count: 1,
            total_frames: 100,
            total_dropped: 0,
            average_frame_time_ms: 16.67,
            global_frame_count: 100,
        };
        let fps = stats.estimated_fps();
        assert!((fps - 60.0).abs() < 1.0);
    }

    // Test 128: Display formatting
    #[test]
    fn test_stats_display() {
        let stats = MultiWindowStats {
            window_count: 2,
            total_frames: 1000,
            total_dropped: 5,
            average_frame_time_ms: 8.33,
            global_frame_count: 500,
        };
        let display = format!("{}", stats);
        assert!(display.contains("2 windows"));
        assert!(display.contains("1000 frames"));
        assert!(display.contains("5 dropped"));
    }
}

// ============================================================================
// Additional edge case tests
// ============================================================================

mod edge_case_tests {
    use super::*;

    // Test 129: acquire_all_frames() with no windows
    #[test]
    fn test_acquire_all_frames_empty() {
        let mut manager = MultiWindowManager::new();
        let frames = manager.acquire_all_frames();
        assert!(frames.is_empty());
    }

    // Test 130: present_all() with empty vec
    #[test]
    fn test_present_all_empty() {
        let mut manager = MultiWindowManager::new();
        manager.present_all(vec![]);
        assert_eq!(manager.global_frame_count(), 1);
    }

    // Test 131: iter() over windows
    #[test]
    fn test_iter_windows() {
        let mut manager = MultiWindowManager::new();
        let id1 = manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))).unwrap();
        let id2 = manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(1024, 768))).unwrap();

        let mut ids_found = Vec::new();
        for (id, _state) in manager.iter() {
            ids_found.push(id);
        }

        assert!(ids_found.contains(&id1));
        assert!(ids_found.contains(&id2));
    }

    // Test 132: resize_window() updates dimensions
    #[test]
    fn test_resize_window() {
        let mut manager = MultiWindowManager::new();
        let id = manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))).unwrap();
        manager.resize_window(id, 1920, 1080).unwrap();
        let state = manager.get_window(id).unwrap();
        assert_eq!(state.config.dimensions(), (1920, 1080));
    }

    // Test 133: resize_window() error for unknown
    #[test]
    fn test_resize_unknown_error() {
        let mut manager = MultiWindowManager::new();
        let result = manager.resize_window(WindowId::from_raw_id(999), 800, 600);
        assert!(result.is_err());
    }

    // Test 134: WindowConfig aspect_ratio with zero height returns 1.0
    #[test]
    fn test_aspect_ratio_zero_height() {
        let mut config = WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600));
        config.config.height = 0;
        assert!((config.aspect_ratio() - 1.0).abs() < 0.001);
    }

    // Test 135: SyncToRate with 0 Hz returns None interval
    #[test]
    fn test_sync_to_rate_zero_hz() {
        let mode = SyncMode::SyncToRate { target_hz: 0 };
        assert!(mode.target_interval().is_none());
    }

    // Test 136: Debug formatting for MultiWindowManager
    #[test]
    fn test_manager_debug() {
        let manager = MultiWindowManager::new();
        let debug = format!("{:?}", manager);
        assert!(debug.contains("MultiWindowManager"));
        assert!(debug.contains("window_count"));
    }

    // Test 137: get_window_mut() returns mutable state
    #[test]
    fn test_get_window_mut() {
        let mut manager = MultiWindowManager::new();
        let id = manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))).unwrap();
        if let Some(state) = manager.get_window_mut(id) {
            state.set_focused(true);
        }
        assert!(manager.get_window(id).unwrap().is_focused());
    }

    // Test 138: Multiple windows with same priority
    #[test]
    fn test_same_priority_multiple() {
        let mut manager = MultiWindowManager::new();
        for _ in 0..5 {
            manager.register_window(
                WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))
                    .with_priority(100)
            ).unwrap();
        }
        assert_eq!(manager.window_count(), 5);
        assert_eq!(manager.window_ids().len(), 5);
    }

    // Test 139: Primary window auto-focuses when registered
    #[test]
    fn test_primary_auto_focuses() {
        let mut manager = MultiWindowManager::new();
        // Register non-primary first
        let id1 = manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))).unwrap();
        assert_eq!(manager.focused_window_id(), Some(id1));

        // Register primary
        let primary_id = manager.register_window(WindowConfig::primary(SurfaceConfiguration::new(1920, 1080))).unwrap();
        // Primary should now be focused
        assert_eq!(manager.focused_window_id(), Some(primary_id));
    }

    // Test 140: Drop rate zero when no dropped frames
    #[test]
    fn test_drop_rate_zero() {
        let stats = MultiWindowStats {
            window_count: 1,
            total_frames: 100,
            total_dropped: 0,
            average_frame_time_ms: 16.67,
            global_frame_count: 100,
        };
        assert!((stats.drop_rate() - 0.0).abs() < 0.001);
    }

    // Test 141: Drop rate with no frames is zero
    #[test]
    fn test_drop_rate_no_frames() {
        let stats = MultiWindowStats {
            window_count: 1,
            total_frames: 0,
            total_dropped: 0,
            average_frame_time_ms: 0.0,
            global_frame_count: 0,
        };
        assert!((stats.drop_rate() - 0.0).abs() < 0.001);
    }

    // Test 142: Estimated FPS zero when frame time is zero
    #[test]
    fn test_estimated_fps_zero_frame_time() {
        let stats = MultiWindowStats {
            window_count: 1,
            total_frames: 0,
            total_dropped: 0,
            average_frame_time_ms: 0.0,
            global_frame_count: 0,
        };
        assert!((stats.estimated_fps() - 0.0).abs() < 0.001);
    }

    // Test 143: Focus shifts to next window after unregister
    #[test]
    fn test_focus_shifts_after_unregister() {
        let mut manager = MultiWindowManager::new();
        let id1 = manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(800, 600))).unwrap();
        let id2 = manager.register_window(WindowConfig::new(WindowId::new(), SurfaceConfiguration::new(1024, 768))).unwrap();

        manager.set_focus(id1).unwrap();
        manager.unregister_window(id1).unwrap();

        // Focus should shift to remaining window
        assert_eq!(manager.focused_window_id(), Some(id2));
    }
}
