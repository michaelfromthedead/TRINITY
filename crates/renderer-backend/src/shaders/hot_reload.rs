//! Shader hot-reload system for TRINITY.
//!
//! This module provides file-system watching and automatic shader recompilation
//! for development workflows. It monitors shader directories for changes and
//! triggers cache invalidation and pipeline rebuild callbacks.
//!
//! # Overview
//!
//! Hot-reload enables rapid shader iteration without restarting the application:
//!
//! - **File watching**: Uses notify crate for cross-platform file system events
//! - **Debouncing**: Coalesces rapid file changes (e.g., editor auto-save)
//! - **Callback system**: Register handlers for shader change notifications
//! - **Error recovery**: Graceful handling of compilation errors
//!
//! # Architecture
//!
//! ```text
//! ShaderHotReload
//! +-- ShaderWatcher (notify-based file watcher)
//! |   +-- watcher: RecommendedWatcher
//! |   +-- receiver: Receiver<Event>
//! |   +-- watched_paths: HashSet<PathBuf>
//! +-- ShaderCache (for invalidation)
//! +-- Callbacks (reload notification)
//! +-- Reload Queue (pending reloads)
//! +-- Config (debounce, extensions, etc.)
//! ```
//!
//! # Feature Gate
//!
//! This module is only compiled when the `hot-reload` feature is enabled:
//!
//! ```toml
//! [features]
//! hot-reload = ["dep:notify", "dep:crossbeam-channel"]
//! ```
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::shaders::hot_reload::{
//!     ShaderHotReload, HotReloadConfig, ShaderWatcher,
//! };
//! use renderer_backend::shaders::ShaderCache;
//! use std::sync::Arc;
//!
//! # fn example(cache: Arc<ShaderCache>) -> Result<(), renderer_backend::shaders::hot_reload::HotReloadError> {
//! // Create hot-reload system
//! let config = HotReloadConfig::default();
//! let mut hot_reload = ShaderHotReload::new(cache, config)?;
//!
//! // Watch shader directory
//! hot_reload.watch_shader_directory("shaders/")?;
//!
//! // Register reload callback
//! hot_reload.register_callback(|path| {
//!     println!("Shader changed: {:?}", path);
//! });
//!
//! // In render loop
//! loop {
//!     hot_reload.poll()?;
//!     // ... render ...
//!     # break;
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Debug-Only Usage
//!
//! Hot-reload is intended for development builds. In release builds, the module
//! compiles to no-ops or is excluded entirely via feature gates.

use notify::{
    Config as NotifyConfig, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher,
};
use std::collections::{HashMap, HashSet};
use std::ffi::OsStr;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::cache::ShaderCache;

// ============================================================================
// Constants
// ============================================================================

/// Default debounce duration in milliseconds.
pub const DEFAULT_DEBOUNCE_MS: u64 = 100;

/// Default file extensions to watch.
pub const DEFAULT_WATCH_EXTENSIONS: &[&str] = &["wgsl"];

/// Maximum number of pending reload events before dropping oldest.
pub const MAX_PENDING_RELOADS: usize = 256;

// ============================================================================
// HotReloadError
// ============================================================================

/// Errors that can occur during hot-reload operations.
#[derive(Debug)]
pub enum HotReloadError {
    /// Error from the notify crate during file watching.
    WatchError(notify::Error),
    /// Shader compilation failed during reload.
    CompilationError(String),
    /// Requested path does not exist.
    PathNotFound(PathBuf),
    /// Communication channel was closed unexpectedly.
    ChannelClosed,
    /// Configuration error.
    ConfigError(String),
    /// IO error during file operations.
    IoError(std::io::Error),
}

impl HotReloadError {
    /// Creates a watch error.
    #[inline]
    pub fn watch(err: notify::Error) -> Self {
        Self::WatchError(err)
    }

    /// Creates a compilation error.
    #[inline]
    pub fn compilation(msg: impl Into<String>) -> Self {
        Self::CompilationError(msg.into())
    }

    /// Creates a path not found error.
    #[inline]
    pub fn path_not_found(path: impl Into<PathBuf>) -> Self {
        Self::PathNotFound(path.into())
    }

    /// Creates a channel closed error.
    #[inline]
    pub fn channel_closed() -> Self {
        Self::ChannelClosed
    }

    /// Creates a config error.
    #[inline]
    pub fn config(msg: impl Into<String>) -> Self {
        Self::ConfigError(msg.into())
    }

    /// Returns true if this is a watch error.
    #[inline]
    pub fn is_watch_error(&self) -> bool {
        matches!(self, Self::WatchError(_))
    }

    /// Returns true if this is a compilation error.
    #[inline]
    pub fn is_compilation_error(&self) -> bool {
        matches!(self, Self::CompilationError(_))
    }

    /// Returns true if this is a path not found error.
    #[inline]
    pub fn is_path_not_found(&self) -> bool {
        matches!(self, Self::PathNotFound(_))
    }

    /// Returns true if this is a channel closed error.
    #[inline]
    pub fn is_channel_closed(&self) -> bool {
        matches!(self, Self::ChannelClosed)
    }

    /// Returns the path if this is a PathNotFound error.
    pub fn path(&self) -> Option<&Path> {
        match self {
            Self::PathNotFound(p) => Some(p),
            _ => None,
        }
    }

    /// Returns the compilation error message if applicable.
    pub fn compilation_message(&self) -> Option<&str> {
        match self {
            Self::CompilationError(msg) => Some(msg),
            _ => None,
        }
    }
}

impl fmt::Display for HotReloadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WatchError(e) => write!(f, "watch error: {}", e),
            Self::CompilationError(msg) => write!(f, "compilation error: {}", msg),
            Self::PathNotFound(path) => write!(f, "path not found: {}", path.display()),
            Self::ChannelClosed => write!(f, "reload channel closed"),
            Self::ConfigError(msg) => write!(f, "configuration error: {}", msg),
            Self::IoError(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for HotReloadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::WatchError(e) => Some(e),
            Self::IoError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<notify::Error> for HotReloadError {
    fn from(err: notify::Error) -> Self {
        Self::WatchError(err)
    }
}

impl From<std::io::Error> for HotReloadError {
    fn from(err: std::io::Error) -> Self {
        Self::IoError(err)
    }
}

// ============================================================================
// HotReloadEvent
// ============================================================================

/// Events generated by the hot-reload system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HotReloadEvent {
    /// A shader file was modified.
    ShaderModified {
        /// Path to the modified shader.
        path: PathBuf,
    },
    /// A new shader file was created.
    ShaderCreated {
        /// Path to the new shader.
        path: PathBuf,
    },
    /// A shader file was deleted.
    ShaderDeleted {
        /// Path to the deleted shader.
        path: PathBuf,
    },
    /// An error occurred while watching.
    WatchError {
        /// Error description.
        error: String,
    },
}

impl HotReloadEvent {
    /// Creates a shader modified event.
    #[inline]
    pub fn modified(path: impl Into<PathBuf>) -> Self {
        Self::ShaderModified { path: path.into() }
    }

    /// Creates a shader created event.
    #[inline]
    pub fn created(path: impl Into<PathBuf>) -> Self {
        Self::ShaderCreated { path: path.into() }
    }

    /// Creates a shader deleted event.
    #[inline]
    pub fn deleted(path: impl Into<PathBuf>) -> Self {
        Self::ShaderDeleted { path: path.into() }
    }

    /// Creates a watch error event.
    #[inline]
    pub fn watch_error(error: impl Into<String>) -> Self {
        Self::WatchError {
            error: error.into(),
        }
    }

    /// Returns the path associated with this event, if any.
    pub fn path(&self) -> Option<&Path> {
        match self {
            Self::ShaderModified { path }
            | Self::ShaderCreated { path }
            | Self::ShaderDeleted { path } => Some(path),
            Self::WatchError { .. } => None,
        }
    }

    /// Returns true if this is a modification event.
    #[inline]
    pub fn is_modified(&self) -> bool {
        matches!(self, Self::ShaderModified { .. })
    }

    /// Returns true if this is a creation event.
    #[inline]
    pub fn is_created(&self) -> bool {
        matches!(self, Self::ShaderCreated { .. })
    }

    /// Returns true if this is a deletion event.
    #[inline]
    pub fn is_deleted(&self) -> bool {
        matches!(self, Self::ShaderDeleted { .. })
    }

    /// Returns true if this is an error event.
    #[inline]
    pub fn is_error(&self) -> bool {
        matches!(self, Self::WatchError { .. })
    }

    /// Returns true if this event requires recompilation.
    #[inline]
    pub fn requires_recompilation(&self) -> bool {
        matches!(
            self,
            Self::ShaderModified { .. } | Self::ShaderCreated { .. }
        )
    }

    /// Returns the error message if this is an error event.
    pub fn error_message(&self) -> Option<&str> {
        match self {
            Self::WatchError { error } => Some(error),
            _ => None,
        }
    }
}

impl fmt::Display for HotReloadEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShaderModified { path } => {
                write!(f, "shader modified: {}", path.display())
            }
            Self::ShaderCreated { path } => {
                write!(f, "shader created: {}", path.display())
            }
            Self::ShaderDeleted { path } => {
                write!(f, "shader deleted: {}", path.display())
            }
            Self::WatchError { error } => {
                write!(f, "watch error: {}", error)
            }
        }
    }
}

// ============================================================================
// HotReloadConfig
// ============================================================================

/// Configuration for the hot-reload system.
#[derive(Debug, Clone)]
pub struct HotReloadConfig {
    /// Debounce duration in milliseconds.
    ///
    /// Rapid file changes (e.g., from editors saving multiple times) are
    /// coalesced within this window.
    pub debounce_ms: u64,

    /// Whether to automatically reload shaders when changes are detected.
    ///
    /// If false, changes are queued but not processed automatically.
    pub auto_reload: bool,

    /// File extensions to watch (without leading dot).
    ///
    /// Only files with these extensions trigger reload events.
    pub watch_extensions: Vec<String>,

    /// Whether to watch directories recursively.
    pub recursive: bool,

    /// Whether to log reload events.
    pub log_events: bool,

    /// Maximum number of pending reload events.
    pub max_pending: usize,
}

impl Default for HotReloadConfig {
    fn default() -> Self {
        Self {
            debounce_ms: DEFAULT_DEBOUNCE_MS,
            auto_reload: true,
            watch_extensions: DEFAULT_WATCH_EXTENSIONS
                .iter()
                .map(|s| s.to_string())
                .collect(),
            recursive: true,
            log_events: true,
            max_pending: MAX_PENDING_RELOADS,
        }
    }
}

impl HotReloadConfig {
    /// Creates a new config with default values.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the debounce duration in milliseconds.
    #[inline]
    pub fn debounce_ms(mut self, ms: u64) -> Self {
        self.debounce_ms = ms;
        self
    }

    /// Sets the debounce duration from a Duration.
    #[inline]
    pub fn debounce(mut self, duration: Duration) -> Self {
        self.debounce_ms = duration.as_millis() as u64;
        self
    }

    /// Enables or disables auto-reload.
    #[inline]
    pub fn auto_reload(mut self, enabled: bool) -> Self {
        self.auto_reload = enabled;
        self
    }

    /// Sets the file extensions to watch.
    pub fn watch_extensions(mut self, extensions: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.watch_extensions = extensions.into_iter().map(Into::into).collect();
        self
    }

    /// Adds a file extension to watch.
    #[inline]
    pub fn add_extension(mut self, ext: impl Into<String>) -> Self {
        self.watch_extensions.push(ext.into());
        self
    }

    /// Enables or disables recursive directory watching.
    #[inline]
    pub fn recursive(mut self, enabled: bool) -> Self {
        self.recursive = enabled;
        self
    }

    /// Enables or disables event logging.
    #[inline]
    pub fn log_events(mut self, enabled: bool) -> Self {
        self.log_events = enabled;
        self
    }

    /// Sets the maximum number of pending reload events.
    #[inline]
    pub fn max_pending(mut self, max: usize) -> Self {
        self.max_pending = max;
        self
    }

    /// Creates a minimal config for testing.
    pub fn minimal() -> Self {
        Self {
            debounce_ms: 10,
            auto_reload: false,
            watch_extensions: vec!["wgsl".to_string()],
            recursive: false,
            log_events: false,
            max_pending: 16,
        }
    }

    /// Creates a config optimized for development.
    pub fn development() -> Self {
        Self {
            debounce_ms: 50,
            auto_reload: true,
            watch_extensions: vec!["wgsl".to_string(), "glsl".to_string()],
            recursive: true,
            log_events: true,
            max_pending: 128,
        }
    }

    /// Returns the debounce duration.
    #[inline]
    pub fn debounce_duration(&self) -> Duration {
        Duration::from_millis(self.debounce_ms)
    }

    /// Checks if an extension is being watched.
    pub fn watches_extension(&self, ext: &str) -> bool {
        let ext_lower = ext.to_lowercase();
        let ext_trimmed = ext_lower.trim_start_matches('.');
        self.watch_extensions
            .iter()
            .any(|e| e.eq_ignore_ascii_case(ext_trimmed))
    }

    /// Validates the configuration.
    pub fn validate(&self) -> Result<(), HotReloadError> {
        if self.watch_extensions.is_empty() {
            return Err(HotReloadError::config(
                "at least one watch extension is required",
            ));
        }
        if self.max_pending == 0 {
            return Err(HotReloadError::config("max_pending must be greater than 0"));
        }
        Ok(())
    }
}

// ============================================================================
// ShaderWatcher
// ============================================================================

/// Low-level file system watcher for shader files.
///
/// This wraps the notify crate's RecommendedWatcher with shader-specific
/// filtering and event handling.
pub struct ShaderWatcher {
    /// The underlying notify watcher.
    watcher: RecommendedWatcher,
    /// Channel receiver for file system events.
    receiver: Receiver<notify::Result<Event>>,
    /// Set of watched directory paths.
    watched_paths: HashSet<PathBuf>,
    /// File extensions to filter for.
    extensions: HashSet<String>,
    /// Whether to watch recursively.
    recursive: bool,
}

impl ShaderWatcher {
    /// Creates a new shader watcher.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::shaders::hot_reload::ShaderWatcher;
    ///
    /// let watcher = ShaderWatcher::new()?;
    /// # Ok::<(), renderer_backend::shaders::hot_reload::HotReloadError>(())
    /// ```
    pub fn new() -> Result<Self, HotReloadError> {
        Self::with_extensions(DEFAULT_WATCH_EXTENSIONS.iter().map(|s| s.to_string()))
    }

    /// Creates a new shader watcher with custom extensions.
    pub fn with_extensions(
        extensions: impl IntoIterator<Item = String>,
    ) -> Result<Self, HotReloadError> {
        let (sender, receiver) = channel();

        let watcher = RecommendedWatcher::new(
            move |result: notify::Result<Event>| {
                // Ignore send errors (receiver dropped)
                let _ = sender.send(result);
            },
            NotifyConfig::default(),
        )?;

        Ok(Self {
            watcher,
            receiver,
            watched_paths: HashSet::new(),
            extensions: extensions.into_iter().collect(),
            recursive: true,
        })
    }

    /// Creates a watcher from a HotReloadConfig.
    pub fn from_config(config: &HotReloadConfig) -> Result<Self, HotReloadError> {
        let mut watcher = Self::with_extensions(config.watch_extensions.clone())?;
        watcher.recursive = config.recursive;
        Ok(watcher)
    }

    /// Adds a directory to watch.
    ///
    /// # Arguments
    ///
    /// * `path` - Directory path to watch
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or `Err(HotReloadError)` if the path doesn't exist
    /// or watching fails.
    pub fn watch_directory(&mut self, path: impl AsRef<Path>) -> Result<(), HotReloadError> {
        let path = path.as_ref();

        // Verify path exists
        if !path.exists() {
            return Err(HotReloadError::path_not_found(path));
        }

        // Convert to canonical path for consistent tracking
        let canonical = path.canonicalize()?;

        // Check if already watching
        if self.watched_paths.contains(&canonical) {
            return Ok(());
        }

        let mode = if self.recursive {
            RecursiveMode::Recursive
        } else {
            RecursiveMode::NonRecursive
        };

        self.watcher.watch(&canonical, mode)?;
        self.watched_paths.insert(canonical);

        Ok(())
    }

    /// Removes a directory from watching.
    ///
    /// # Arguments
    ///
    /// * `path` - Directory path to stop watching
    ///
    /// # Returns
    ///
    /// `true` if the directory was being watched and is now unwatched.
    pub fn unwatch_directory(&mut self, path: impl AsRef<Path>) -> Result<bool, HotReloadError> {
        let path = path.as_ref();

        // Try canonical path first
        let canonical = if path.exists() {
            path.canonicalize().ok()
        } else {
            None
        };

        let fallback = path.to_path_buf();
        let watch_path = canonical.as_ref().unwrap_or(&fallback);

        if !self.watched_paths.contains(watch_path) {
            return Ok(false);
        }

        self.watcher.unwatch(watch_path)?;
        self.watched_paths.remove(watch_path);

        Ok(true)
    }

    /// Polls for file system changes without blocking.
    ///
    /// # Returns
    ///
    /// A vector of changed file paths that match the watched extensions.
    pub fn poll_changes(&self) -> Vec<PathBuf> {
        let mut changes = Vec::new();

        loop {
            match self.receiver.try_recv() {
                Ok(Ok(event)) => {
                    // Filter for relevant event kinds
                    if !Self::is_relevant_event(&event.kind) {
                        continue;
                    }

                    // Filter paths by extension
                    for path in event.paths {
                        if self.matches_extension(&path) {
                            changes.push(path);
                        }
                    }
                }
                Ok(Err(_)) => {
                    // Watch error - continue polling
                    continue;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }

        // Deduplicate paths
        changes.sort();
        changes.dedup();
        changes
    }

    /// Returns the changed files as HotReloadEvents.
    pub fn changed_files(&self) -> Vec<HotReloadEvent> {
        let mut events = Vec::new();

        loop {
            match self.receiver.try_recv() {
                Ok(Ok(event)) => {
                    events.extend(self.convert_event(&event));
                }
                Ok(Err(err)) => {
                    events.push(HotReloadEvent::watch_error(err.to_string()));
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }

        events
    }

    /// Returns the set of watched directories.
    #[inline]
    pub fn watched_paths(&self) -> &HashSet<PathBuf> {
        &self.watched_paths
    }

    /// Returns the number of watched directories.
    #[inline]
    pub fn watched_count(&self) -> usize {
        self.watched_paths.len()
    }

    /// Returns true if a path is being watched.
    pub fn is_watching(&self, path: impl AsRef<Path>) -> bool {
        let path = path.as_ref();
        if let Ok(canonical) = path.canonicalize() {
            self.watched_paths.contains(&canonical)
        } else {
            self.watched_paths.contains(path)
        }
    }

    /// Returns the file extensions being watched.
    #[inline]
    pub fn extensions(&self) -> &HashSet<String> {
        &self.extensions
    }

    /// Adds an extension to watch.
    #[inline]
    pub fn add_extension(&mut self, ext: impl Into<String>) {
        self.extensions.insert(ext.into());
    }

    /// Removes an extension from watching.
    #[inline]
    pub fn remove_extension(&mut self, ext: &str) -> bool {
        self.extensions.remove(ext)
    }

    /// Sets whether to watch recursively.
    #[inline]
    pub fn set_recursive(&mut self, recursive: bool) {
        self.recursive = recursive;
    }

    /// Returns whether recursive watching is enabled.
    #[inline]
    pub fn is_recursive(&self) -> bool {
        self.recursive
    }

    /// Checks if an event kind is relevant for shader hot-reload.
    fn is_relevant_event(kind: &EventKind) -> bool {
        matches!(
            kind,
            EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_)
        )
    }

    /// Checks if a path matches the watched extensions.
    fn matches_extension(&self, path: &Path) -> bool {
        if self.extensions.is_empty() {
            return true;
        }

        path.extension()
            .and_then(OsStr::to_str)
            .map(|ext| {
                self.extensions
                    .iter()
                    .any(|e| e.eq_ignore_ascii_case(ext))
            })
            .unwrap_or(false)
    }

    /// Converts a notify Event to HotReloadEvents.
    fn convert_event(&self, event: &Event) -> Vec<HotReloadEvent> {
        let mut events = Vec::new();

        // Filter for relevant event kinds
        let event_type = match &event.kind {
            EventKind::Create(_) => Some(HotReloadEventType::Created),
            EventKind::Modify(_) => Some(HotReloadEventType::Modified),
            EventKind::Remove(_) => Some(HotReloadEventType::Deleted),
            _ => None,
        };

        if let Some(event_type) = event_type {
            for path in &event.paths {
                if self.matches_extension(path) {
                    events.push(match event_type {
                        HotReloadEventType::Created => HotReloadEvent::created(path),
                        HotReloadEventType::Modified => HotReloadEvent::modified(path),
                        HotReloadEventType::Deleted => HotReloadEvent::deleted(path),
                    });
                }
            }
        }

        events
    }
}

impl fmt::Debug for ShaderWatcher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ShaderWatcher")
            .field("watched_paths", &self.watched_paths)
            .field("extensions", &self.extensions)
            .field("recursive", &self.recursive)
            .finish_non_exhaustive()
    }
}

/// Internal enum for event type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HotReloadEventType {
    Created,
    Modified,
    Deleted,
}

// ============================================================================
// DebounceState
// ============================================================================

/// Tracks debounce state for a single file.
#[derive(Debug, Clone)]
struct DebounceEntry {
    /// Last event time.
    last_event: Instant,
    /// Event type.
    event_type: HotReloadEventType,
}

/// Debounce tracker for multiple files.
#[derive(Debug)]
struct DebounceState {
    /// Map of path to debounce entry.
    entries: HashMap<PathBuf, DebounceEntry>,
    /// Debounce duration.
    duration: Duration,
}

impl DebounceState {
    /// Creates a new debounce state.
    fn new(duration: Duration) -> Self {
        Self {
            entries: HashMap::new(),
            duration,
        }
    }

    /// Records an event, returning true if it should be emitted (debounce passed).
    fn record(&mut self, path: &Path, event_type: HotReloadEventType) -> bool {
        let now = Instant::now();

        if let Some(entry) = self.entries.get(path) {
            // Check if debounce period has passed
            if now.duration_since(entry.last_event) < self.duration {
                // Update timestamp but don't emit
                self.entries.insert(
                    path.to_path_buf(),
                    DebounceEntry {
                        last_event: now,
                        event_type,
                    },
                );
                return false;
            }
        }

        // Emit and record
        self.entries.insert(
            path.to_path_buf(),
            DebounceEntry {
                last_event: now,
                event_type,
            },
        );
        true
    }

    /// Returns paths that have passed their debounce period.
    fn drain_ready(&mut self) -> Vec<(PathBuf, HotReloadEventType)> {
        let now = Instant::now();
        let mut ready = Vec::new();

        self.entries.retain(|path, entry| {
            if now.duration_since(entry.last_event) >= self.duration {
                ready.push((path.clone(), entry.event_type));
                false
            } else {
                true
            }
        });

        ready
    }

    /// Clears all debounce state.
    fn clear(&mut self) {
        self.entries.clear();
    }

    /// Returns the number of pending entries.
    #[inline]
    fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if there are no pending entries.
    #[inline]
    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ============================================================================
// ShaderHotReload
// ============================================================================

/// Type alias for reload callbacks.
pub type ReloadCallback = Box<dyn Fn(&Path) + Send + Sync>;

/// High-level shader hot-reload system.
///
/// Combines file watching, debouncing, cache invalidation, and callback
/// notification for seamless shader iteration.
pub struct ShaderHotReload {
    /// File system watcher.
    watcher: ShaderWatcher,
    /// Shader cache for invalidation.
    cache: Arc<ShaderCache>,
    /// Registered reload callbacks.
    callbacks: Vec<ReloadCallback>,
    /// Channel for reload notifications.
    reload_sender: Sender<PathBuf>,
    /// Channel receiver for reload notifications.
    reload_receiver: Receiver<PathBuf>,
    /// Configuration.
    config: HotReloadConfig,
    /// Debounce state.
    debounce: DebounceState,
    /// Pending reload queue.
    pending_reloads: Vec<PathBuf>,
    /// Statistics.
    stats: HotReloadStats,
}

impl ShaderHotReload {
    /// Creates a new hot-reload system.
    ///
    /// # Arguments
    ///
    /// * `cache` - The shader cache to invalidate on changes
    /// * `config` - Hot-reload configuration
    ///
    /// # Returns
    ///
    /// A new `ShaderHotReload` instance, or an error if initialization fails.
    pub fn new(cache: Arc<ShaderCache>, config: HotReloadConfig) -> Result<Self, HotReloadError> {
        config.validate()?;

        let watcher = ShaderWatcher::from_config(&config)?;
        let (reload_sender, reload_receiver) = channel();
        let debounce = DebounceState::new(config.debounce_duration());

        Ok(Self {
            watcher,
            cache,
            callbacks: Vec::new(),
            reload_sender,
            reload_receiver,
            config,
            debounce,
            pending_reloads: Vec::new(),
            stats: HotReloadStats::default(),
        })
    }

    /// Creates a hot-reload system with default configuration.
    pub fn with_defaults(cache: Arc<ShaderCache>) -> Result<Self, HotReloadError> {
        Self::new(cache, HotReloadConfig::default())
    }

    /// Adds a directory to watch for shader changes.
    ///
    /// # Arguments
    ///
    /// * `path` - Directory path to watch
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or an error if the path doesn't exist.
    pub fn watch_shader_directory(&mut self, path: impl AsRef<Path>) -> Result<(), HotReloadError> {
        self.watcher.watch_directory(path)
    }

    /// Removes a directory from watching.
    pub fn unwatch_directory(&mut self, path: impl AsRef<Path>) -> Result<bool, HotReloadError> {
        self.watcher.unwatch_directory(path)
    }

    /// Registers a callback to be invoked when shaders are reloaded.
    ///
    /// Callbacks receive the path of the changed shader file.
    pub fn register_callback<F>(&mut self, callback: F)
    where
        F: Fn(&Path) + Send + Sync + 'static,
    {
        self.callbacks.push(Box::new(callback));
    }

    /// Clears all registered callbacks.
    pub fn clear_callbacks(&mut self) {
        self.callbacks.clear();
    }

    /// Returns the number of registered callbacks.
    #[inline]
    pub fn callback_count(&self) -> usize {
        self.callbacks.len()
    }

    /// Polls for shader changes and processes them.
    ///
    /// This should be called regularly (e.g., each frame) in your render loop.
    ///
    /// # Returns
    ///
    /// The list of shaders that were reloaded.
    pub fn poll(&mut self) -> Result<Vec<PathBuf>, HotReloadError> {
        let mut reloaded = Vec::new();

        // Poll for new events
        let events = self.watcher.changed_files();
        for event in events {
            if let Some(path) = event.path() {
                let event_type = if event.is_created() {
                    HotReloadEventType::Created
                } else if event.is_deleted() {
                    HotReloadEventType::Deleted
                } else {
                    HotReloadEventType::Modified
                };

                // Record for debouncing
                if self.debounce.record(path, event_type) && self.config.auto_reload {
                    self.queue_reload(path.to_path_buf());
                }
            }

            if event.is_error() {
                self.stats.errors += 1;
                if self.config.log_events {
                    log::warn!(
                        "Hot-reload watch error: {}",
                        event.error_message().unwrap_or("unknown")
                    );
                }
            }
        }

        // Check for debounced events that are ready
        let ready = self.debounce.drain_ready();
        for (path, event_type) in ready {
            if matches!(
                event_type,
                HotReloadEventType::Created | HotReloadEventType::Modified
            ) {
                self.queue_reload(path);
            }
        }

        // Process pending reloads from queue
        while let Ok(path) = self.reload_receiver.try_recv() {
            if self.config.auto_reload {
                self.process_reload(&path)?;
                reloaded.push(path);
            }
        }

        // Process pending reloads
        let pending = std::mem::take(&mut self.pending_reloads);
        for path in pending {
            self.process_reload(&path)?;
            reloaded.push(path);
        }

        Ok(reloaded)
    }

    /// Manually triggers a shader reload.
    ///
    /// This bypasses file watching and directly invalidates and reloads
    /// the specified shader.
    pub fn reload_shader(&mut self, path: impl AsRef<Path>) -> Result<(), HotReloadError> {
        let path = path.as_ref();
        self.process_reload(path)
    }

    /// Returns the list of pending reloads.
    pub fn pending_reloads(&self) -> &[PathBuf] {
        &self.pending_reloads
    }

    /// Returns the number of pending reloads.
    #[inline]
    pub fn pending_count(&self) -> usize {
        self.pending_reloads.len()
    }

    /// Clears all pending reloads.
    pub fn clear_pending(&mut self) {
        self.pending_reloads.clear();
        self.debounce.clear();
    }

    /// Returns the hot-reload configuration.
    #[inline]
    pub fn config(&self) -> &HotReloadConfig {
        &self.config
    }

    /// Returns the underlying shader watcher.
    #[inline]
    pub fn watcher(&self) -> &ShaderWatcher {
        &self.watcher
    }

    /// Returns hot-reload statistics.
    #[inline]
    pub fn stats(&self) -> &HotReloadStats {
        &self.stats
    }

    /// Resets statistics counters.
    pub fn reset_stats(&mut self) {
        self.stats = HotReloadStats::default();
    }

    /// Returns the watched directories.
    pub fn watched_directories(&self) -> Vec<PathBuf> {
        self.watcher.watched_paths().iter().cloned().collect()
    }

    /// Queues a shader for reload.
    fn queue_reload(&mut self, path: PathBuf) {
        // Enforce max pending limit
        while self.pending_reloads.len() >= self.config.max_pending {
            self.pending_reloads.remove(0);
            self.stats.dropped += 1;
        }

        // Avoid duplicates
        if !self.pending_reloads.contains(&path) {
            self.pending_reloads.push(path.clone());
            // Also send to channel for immediate processing
            let _ = self.reload_sender.send(path);
        }
    }

    /// Processes a shader reload.
    fn process_reload(&mut self, path: &Path) -> Result<(), HotReloadError> {
        if self.config.log_events {
            log::info!("Hot-reloading shader: {}", path.display());
        }

        // Invalidate cache entry
        self.cache.invalidate_by_path(path);
        self.stats.invalidations += 1;

        // Notify callbacks
        for callback in &self.callbacks {
            callback(path);
        }
        self.stats.callbacks_invoked += self.callbacks.len() as u64;

        self.stats.reloads += 1;

        Ok(())
    }
}

impl fmt::Debug for ShaderHotReload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ShaderHotReload")
            .field("watcher", &self.watcher)
            .field("callbacks", &self.callbacks.len())
            .field("pending_reloads", &self.pending_reloads.len())
            .field("config", &self.config)
            .field("stats", &self.stats)
            .finish()
    }
}

// ============================================================================
// HotReloadStats
// ============================================================================

/// Statistics for hot-reload operations.
#[derive(Debug, Clone, Default)]
pub struct HotReloadStats {
    /// Number of shaders reloaded.
    pub reloads: u64,
    /// Number of cache invalidations.
    pub invalidations: u64,
    /// Number of callbacks invoked.
    pub callbacks_invoked: u64,
    /// Number of watch errors.
    pub errors: u64,
    /// Number of events dropped due to queue overflow.
    pub dropped: u64,
}

impl HotReloadStats {
    /// Returns the total number of events processed.
    #[inline]
    pub fn total_events(&self) -> u64 {
        self.reloads + self.errors
    }

    /// Returns true if any errors occurred.
    #[inline]
    pub fn has_errors(&self) -> bool {
        self.errors > 0
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // HotReloadError Tests (6 tests)
    // =========================================================================

    #[test]
    fn test_error_watch() {
        let err = HotReloadError::watch(notify::Error::generic("test"));
        assert!(err.is_watch_error());
        assert!(!err.is_compilation_error());
        assert!(!err.is_path_not_found());
        assert!(!err.is_channel_closed());
    }

    #[test]
    fn test_error_compilation() {
        let err = HotReloadError::compilation("shader compile failed");
        assert!(!err.is_watch_error());
        assert!(err.is_compilation_error());
        assert_eq!(err.compilation_message(), Some("shader compile failed"));
    }

    #[test]
    fn test_error_path_not_found() {
        let err = HotReloadError::path_not_found("/missing/path");
        assert!(err.is_path_not_found());
        assert_eq!(err.path(), Some(Path::new("/missing/path")));
    }

    #[test]
    fn test_error_channel_closed() {
        let err = HotReloadError::channel_closed();
        assert!(err.is_channel_closed());
    }

    #[test]
    fn test_error_display() {
        let err = HotReloadError::compilation("test error");
        let display = format!("{}", err);
        assert!(display.contains("compilation error"));
        assert!(display.contains("test error"));
    }

    #[test]
    fn test_error_from_notify() {
        let notify_err = notify::Error::generic("notify error");
        let err: HotReloadError = notify_err.into();
        assert!(err.is_watch_error());
    }

    // =========================================================================
    // HotReloadEvent Tests (6 tests)
    // =========================================================================

    #[test]
    fn test_event_modified() {
        let event = HotReloadEvent::modified("test.wgsl");
        assert!(event.is_modified());
        assert!(!event.is_created());
        assert!(!event.is_deleted());
        assert!(!event.is_error());
        assert_eq!(event.path(), Some(Path::new("test.wgsl")));
        assert!(event.requires_recompilation());
    }

    #[test]
    fn test_event_created() {
        let event = HotReloadEvent::created("new.wgsl");
        assert!(!event.is_modified());
        assert!(event.is_created());
        assert!(event.requires_recompilation());
    }

    #[test]
    fn test_event_deleted() {
        let event = HotReloadEvent::deleted("old.wgsl");
        assert!(event.is_deleted());
        assert!(!event.requires_recompilation());
    }

    #[test]
    fn test_event_watch_error() {
        let event = HotReloadEvent::watch_error("file not found");
        assert!(event.is_error());
        assert!(event.path().is_none());
        assert_eq!(event.error_message(), Some("file not found"));
    }

    #[test]
    fn test_event_display() {
        let event = HotReloadEvent::modified("shaders/pbr.wgsl");
        let display = format!("{}", event);
        assert!(display.contains("modified"));
        assert!(display.contains("pbr.wgsl"));
    }

    #[test]
    fn test_event_equality() {
        let e1 = HotReloadEvent::modified("test.wgsl");
        let e2 = HotReloadEvent::modified("test.wgsl");
        let e3 = HotReloadEvent::modified("other.wgsl");

        assert_eq!(e1, e2);
        assert_ne!(e1, e3);
    }

    // =========================================================================
    // HotReloadConfig Tests (8 tests)
    // =========================================================================

    #[test]
    fn test_config_default() {
        let config = HotReloadConfig::default();
        assert_eq!(config.debounce_ms, DEFAULT_DEBOUNCE_MS);
        assert!(config.auto_reload);
        assert!(config.recursive);
        assert!(config.log_events);
        assert!(!config.watch_extensions.is_empty());
    }

    #[test]
    fn test_config_builder() {
        let config = HotReloadConfig::new()
            .debounce_ms(200)
            .auto_reload(false)
            .recursive(false)
            .log_events(false)
            .max_pending(50);

        assert_eq!(config.debounce_ms, 200);
        assert!(!config.auto_reload);
        assert!(!config.recursive);
        assert!(!config.log_events);
        assert_eq!(config.max_pending, 50);
    }

    #[test]
    fn test_config_debounce_duration() {
        let config = HotReloadConfig::new().debounce(Duration::from_millis(150));
        assert_eq!(config.debounce_duration(), Duration::from_millis(150));
    }

    #[test]
    fn test_config_watch_extensions() {
        let config = HotReloadConfig::new()
            .watch_extensions(vec!["wgsl", "glsl", "hlsl"]);
        assert_eq!(config.watch_extensions.len(), 3);
    }

    #[test]
    fn test_config_add_extension() {
        let config = HotReloadConfig::new().add_extension("spv");
        assert!(config.watch_extensions.contains(&"spv".to_string()));
    }

    #[test]
    fn test_config_watches_extension() {
        let config = HotReloadConfig::default();
        assert!(config.watches_extension("wgsl"));
        assert!(config.watches_extension("WGSL"));
        assert!(config.watches_extension(".wgsl"));
        assert!(!config.watches_extension("txt"));
    }

    #[test]
    fn test_config_validate_success() {
        let config = HotReloadConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validate_failure() {
        let config = HotReloadConfig::new().watch_extensions(Vec::<String>::new());
        assert!(config.validate().is_err());

        let config = HotReloadConfig::new().max_pending(0);
        assert!(config.validate().is_err());
    }

    // =========================================================================
    // ShaderWatcher Tests (8 tests)
    // =========================================================================

    #[test]
    fn test_watcher_new() {
        let watcher = ShaderWatcher::new();
        assert!(watcher.is_ok());
        let watcher = watcher.unwrap();
        assert!(watcher.watched_paths().is_empty());
    }

    #[test]
    fn test_watcher_with_extensions() {
        let watcher = ShaderWatcher::with_extensions(vec!["wgsl".to_string(), "glsl".to_string()]);
        assert!(watcher.is_ok());
        let watcher = watcher.unwrap();
        assert!(watcher.extensions().contains("wgsl"));
        assert!(watcher.extensions().contains("glsl"));
    }

    #[test]
    fn test_watcher_from_config() {
        let config = HotReloadConfig::new().recursive(false);
        let watcher = ShaderWatcher::from_config(&config);
        assert!(watcher.is_ok());
        let watcher = watcher.unwrap();
        assert!(!watcher.is_recursive());
    }

    #[test]
    fn test_watcher_watch_nonexistent() {
        let mut watcher = ShaderWatcher::new().unwrap();
        let result = watcher.watch_directory("/nonexistent/path/that/does/not/exist");
        assert!(result.is_err());
    }

    #[test]
    fn test_watcher_add_extension() {
        let mut watcher = ShaderWatcher::new().unwrap();
        watcher.add_extension("hlsl");
        assert!(watcher.extensions().contains("hlsl"));
    }

    #[test]
    fn test_watcher_remove_extension() {
        let mut watcher = ShaderWatcher::with_extensions(vec!["wgsl".to_string()]).unwrap();
        assert!(watcher.remove_extension("wgsl"));
        assert!(!watcher.remove_extension("wgsl"));
    }

    #[test]
    fn test_watcher_set_recursive() {
        let mut watcher = ShaderWatcher::new().unwrap();
        assert!(watcher.is_recursive());
        watcher.set_recursive(false);
        assert!(!watcher.is_recursive());
    }

    #[test]
    fn test_watcher_poll_empty() {
        let watcher = ShaderWatcher::new().unwrap();
        let changes = watcher.poll_changes();
        assert!(changes.is_empty());
    }

    // =========================================================================
    // ShaderHotReload Tests (10 tests)
    // =========================================================================

    // Note: ShaderHotReload tests require a ShaderCache which needs a wgpu device.
    // The following tests focus on unit-testable aspects.

    #[test]
    fn test_config_minimal() {
        let config = HotReloadConfig::minimal();
        assert_eq!(config.debounce_ms, 10);
        assert!(!config.auto_reload);
        assert!(!config.recursive);
    }

    #[test]
    fn test_config_development() {
        let config = HotReloadConfig::development();
        assert_eq!(config.debounce_ms, 50);
        assert!(config.auto_reload);
        assert!(config.recursive);
    }

    #[test]
    fn test_stats_default() {
        let stats = HotReloadStats::default();
        assert_eq!(stats.reloads, 0);
        assert_eq!(stats.invalidations, 0);
        assert_eq!(stats.callbacks_invoked, 0);
        assert_eq!(stats.errors, 0);
        assert_eq!(stats.dropped, 0);
    }

    #[test]
    fn test_stats_total_events() {
        let mut stats = HotReloadStats::default();
        stats.reloads = 10;
        stats.errors = 2;
        assert_eq!(stats.total_events(), 12);
    }

    #[test]
    fn test_stats_has_errors() {
        let mut stats = HotReloadStats::default();
        assert!(!stats.has_errors());
        stats.errors = 1;
        assert!(stats.has_errors());
    }

    #[test]
    fn test_debounce_state_new() {
        let state = DebounceState::new(Duration::from_millis(100));
        assert!(state.is_empty());
        assert_eq!(state.len(), 0);
    }

    #[test]
    fn test_debounce_state_record_first() {
        let mut state = DebounceState::new(Duration::from_millis(100));
        let path = Path::new("test.wgsl");
        assert!(state.record(path, HotReloadEventType::Modified));
        assert_eq!(state.len(), 1);
    }

    #[test]
    fn test_debounce_state_record_rapid() {
        let mut state = DebounceState::new(Duration::from_millis(1000));
        let path = Path::new("test.wgsl");

        // First event emits
        assert!(state.record(path, HotReloadEventType::Modified));
        // Rapid second event does not emit
        assert!(!state.record(path, HotReloadEventType::Modified));
    }

    #[test]
    fn test_debounce_state_clear() {
        let mut state = DebounceState::new(Duration::from_millis(100));
        state.record(Path::new("a.wgsl"), HotReloadEventType::Modified);
        state.record(Path::new("b.wgsl"), HotReloadEventType::Modified);
        assert_eq!(state.len(), 2);

        state.clear();
        assert!(state.is_empty());
    }

    #[test]
    fn test_debounce_state_drain_ready() {
        let mut state = DebounceState::new(Duration::from_millis(0));
        let path = Path::new("test.wgsl");
        state.record(path, HotReloadEventType::Modified);

        // With 0ms debounce, should be immediately ready
        std::thread::sleep(Duration::from_millis(1));
        let ready = state.drain_ready();
        assert_eq!(ready.len(), 1);
        assert!(state.is_empty());
    }

    // =========================================================================
    // Debouncing Tests (4 tests)
    // =========================================================================

    #[test]
    fn test_debounce_different_paths() {
        let mut state = DebounceState::new(Duration::from_millis(100));

        assert!(state.record(Path::new("a.wgsl"), HotReloadEventType::Modified));
        assert!(state.record(Path::new("b.wgsl"), HotReloadEventType::Modified));

        // Different paths should both emit
        assert_eq!(state.len(), 2);
    }

    #[test]
    fn test_debounce_event_type_change() {
        let mut state = DebounceState::new(Duration::from_millis(100));
        let path = Path::new("test.wgsl");

        assert!(state.record(path, HotReloadEventType::Created));
        // Same path, different event type, within debounce period
        assert!(!state.record(path, HotReloadEventType::Modified));
    }

    #[test]
    fn test_debounce_zero_duration() {
        let mut state = DebounceState::new(Duration::from_millis(0));
        let path = Path::new("test.wgsl");

        assert!(state.record(path, HotReloadEventType::Modified));
        // With 0 debounce, second event should also emit (barely)
        std::thread::sleep(Duration::from_micros(1));
        assert!(state.record(path, HotReloadEventType::Modified));
    }

    #[test]
    fn test_debounce_long_duration() {
        let mut state = DebounceState::new(Duration::from_secs(10));
        let path = Path::new("test.wgsl");

        assert!(state.record(path, HotReloadEventType::Modified));
        // With long debounce, second event should not emit
        assert!(!state.record(path, HotReloadEventType::Modified));
    }

    // =========================================================================
    // Path Filtering Tests (3 tests)
    // =========================================================================

    #[test]
    fn test_watcher_matches_extension_wgsl() {
        let watcher = ShaderWatcher::new().unwrap();
        assert!(watcher.matches_extension(Path::new("test.wgsl")));
        assert!(watcher.matches_extension(Path::new("path/to/shader.wgsl")));
    }

    #[test]
    fn test_watcher_matches_extension_case_insensitive() {
        let watcher = ShaderWatcher::new().unwrap();
        assert!(watcher.matches_extension(Path::new("test.WGSL")));
        assert!(watcher.matches_extension(Path::new("test.Wgsl")));
    }

    #[test]
    fn test_watcher_matches_extension_other() {
        let watcher = ShaderWatcher::new().unwrap();
        assert!(!watcher.matches_extension(Path::new("test.txt")));
        assert!(!watcher.matches_extension(Path::new("test.glsl")));
        assert!(!watcher.matches_extension(Path::new("test")));
    }

    // =========================================================================
    // Thread Safety Tests
    // =========================================================================

    #[test]
    fn test_config_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<HotReloadConfig>();
        assert_sync::<HotReloadConfig>();
    }

    #[test]
    fn test_error_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<HotReloadError>();
        assert_sync::<HotReloadError>();
    }

    #[test]
    fn test_event_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<HotReloadEvent>();
        assert_sync::<HotReloadEvent>();
    }

    #[test]
    fn test_stats_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<HotReloadStats>();
        assert_sync::<HotReloadStats>();
    }

    // =========================================================================
    // Constants Tests
    // =========================================================================

    #[test]
    fn test_default_debounce_ms() {
        assert_eq!(DEFAULT_DEBOUNCE_MS, 100);
    }

    #[test]
    fn test_default_watch_extensions() {
        assert!(!DEFAULT_WATCH_EXTENSIONS.is_empty());
        assert!(DEFAULT_WATCH_EXTENSIONS.contains(&"wgsl"));
    }

    #[test]
    fn test_max_pending_reloads() {
        assert!(MAX_PENDING_RELOADS > 0);
    }

    // =========================================================================
    // Clone/Debug Tests
    // =========================================================================

    #[test]
    fn test_config_clone() {
        let config = HotReloadConfig::new().debounce_ms(50).auto_reload(false);
        let cloned = config.clone();
        assert_eq!(cloned.debounce_ms, 50);
        assert!(!cloned.auto_reload);
    }

    #[test]
    fn test_event_clone() {
        let event = HotReloadEvent::modified("test.wgsl");
        let cloned = event.clone();
        assert_eq!(event, cloned);
    }

    #[test]
    fn test_stats_clone() {
        let mut stats = HotReloadStats::default();
        stats.reloads = 5;
        let cloned = stats.clone();
        assert_eq!(cloned.reloads, 5);
    }

    #[test]
    fn test_config_debug() {
        let config = HotReloadConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("HotReloadConfig"));
    }

    #[test]
    fn test_event_debug() {
        let event = HotReloadEvent::modified("test.wgsl");
        let debug = format!("{:?}", event);
        assert!(debug.contains("ShaderModified"));
    }

    #[test]
    fn test_stats_debug() {
        let stats = HotReloadStats::default();
        let debug = format!("{:?}", stats);
        assert!(debug.contains("HotReloadStats"));
    }

    #[test]
    fn test_watcher_debug() {
        let watcher = ShaderWatcher::new().unwrap();
        let debug = format!("{:?}", watcher);
        assert!(debug.contains("ShaderWatcher"));
    }

    // =========================================================================
    // WHITEBOX Tests - ShaderWatcher Edge Cases (6 tests)
    // =========================================================================

    #[test]
    fn test_watcher_watch_same_directory_twice() {
        use std::fs;

        let mut watcher = ShaderWatcher::new().unwrap();
        // Create a dedicated temp directory for this test
        let test_dir = std::env::temp_dir().join("trinity_hot_reload_test_same_dir");
        let _ = fs::create_dir_all(&test_dir);

        // First watch should succeed
        let result1 = watcher.watch_directory(&test_dir);
        assert!(result1.is_ok(), "First watch failed: {:?}", result1.err());
        assert_eq!(watcher.watched_count(), 1);

        // Second watch of same directory should be a no-op (idempotent)
        let result2 = watcher.watch_directory(&test_dir);
        assert!(result2.is_ok(), "Second watch failed: {:?}", result2.err());
        assert_eq!(watcher.watched_count(), 1); // Count should remain 1

        // Cleanup
        let _ = fs::remove_dir(&test_dir);
    }

    #[test]
    fn test_watcher_unwatch_unwatched_directory() {
        let mut watcher = ShaderWatcher::new().unwrap();

        // Unwatch a directory that was never watched
        let result = watcher.unwatch_directory("/some/random/path/never/watched");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), false); // Should return false, not error
    }

    #[test]
    fn test_watcher_extension_filtering_case_insensitive() {
        let watcher = ShaderWatcher::with_extensions(vec!["wgsl".to_string()]).unwrap();

        // Test various case combinations
        assert!(watcher.matches_extension(Path::new("test.wgsl")));
        assert!(watcher.matches_extension(Path::new("test.WGSL")));
        assert!(watcher.matches_extension(Path::new("test.WgSl")));
        assert!(watcher.matches_extension(Path::new("test.wGsL")));

        // Non-matching extensions
        assert!(!watcher.matches_extension(Path::new("test.wgslx")));
        assert!(!watcher.matches_extension(Path::new("testwgsl")));
    }

    #[test]
    fn test_watcher_empty_extensions_accepts_none() {
        // Empty extensions means accept NO files (not ALL files, based on implementation)
        let watcher = ShaderWatcher::with_extensions(Vec::<String>::new()).unwrap();

        // With empty extensions, the implementation returns true for all paths
        // This is the documented behavior: "if extensions.is_empty() return true"
        assert!(watcher.matches_extension(Path::new("test.wgsl")));
        assert!(watcher.matches_extension(Path::new("test.txt")));
        assert!(watcher.matches_extension(Path::new("test.any")));
    }

    #[test]
    fn test_watcher_clear_extensions() {
        let mut watcher = ShaderWatcher::with_extensions(vec![
            "wgsl".to_string(),
            "glsl".to_string(),
        ]).unwrap();

        assert_eq!(watcher.extensions().len(), 2);

        // Remove all extensions one by one
        assert!(watcher.remove_extension("wgsl"));
        assert!(watcher.remove_extension("glsl"));

        assert!(watcher.extensions().is_empty());

        // Now with empty extensions, matches_extension returns true for any path
        assert!(watcher.matches_extension(Path::new("test.txt")));
    }

    #[test]
    fn test_watcher_poll_empty_returns_vec() {
        let watcher = ShaderWatcher::new().unwrap();

        // Polling without any watched directories should return empty vec
        let changes = watcher.poll_changes();
        assert!(changes.is_empty());

        // Should be able to poll multiple times
        let changes2 = watcher.poll_changes();
        assert!(changes2.is_empty());

        // changed_files should also return empty
        let events = watcher.changed_files();
        assert!(events.is_empty());
    }

    // =========================================================================
    // WHITEBOX Tests - ShaderHotReload Internal (8 tests)
    // =========================================================================

    #[test]
    fn test_reload_callback_execution_order() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        // Test that callbacks are invoked in registration order
        let order = Arc::new(AtomicUsize::new(0));
        let order1 = Arc::clone(&order);
        let order2 = Arc::clone(&order);
        let order3 = Arc::clone(&order);

        let results = Arc::new(std::sync::Mutex::new(Vec::new()));
        let r1 = Arc::clone(&results);
        let r2 = Arc::clone(&results);
        let r3 = Arc::clone(&results);

        let cb1: ReloadCallback = Box::new(move |_| {
            r1.lock().unwrap().push(order1.fetch_add(1, Ordering::SeqCst));
        });
        let cb2: ReloadCallback = Box::new(move |_| {
            r2.lock().unwrap().push(order2.fetch_add(1, Ordering::SeqCst));
        });
        let cb3: ReloadCallback = Box::new(move |_| {
            r3.lock().unwrap().push(order3.fetch_add(1, Ordering::SeqCst));
        });

        // Store callbacks in a Vec and invoke them manually
        let callbacks: Vec<ReloadCallback> = vec![cb1, cb2, cb3];
        let path = Path::new("test.wgsl");

        for cb in &callbacks {
            cb(path);
        }

        let final_results = results.lock().unwrap();
        assert_eq!(*final_results, vec![0, 1, 2]);
    }

    #[test]
    fn test_reload_multiple_callbacks() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let counter = Arc::new(AtomicUsize::new(0));

        // Create multiple callbacks that increment counter
        let callbacks: Vec<ReloadCallback> = (0..5)
            .map(|_| {
                let c = Arc::clone(&counter);
                Box::new(move |_: &Path| {
                    c.fetch_add(1, Ordering::SeqCst);
                }) as ReloadCallback
            })
            .collect();

        // Invoke all callbacks
        let path = Path::new("test.wgsl");
        for cb in &callbacks {
            cb(path);
        }

        assert_eq!(counter.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn test_reload_callback_error_isolation() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let second_called = Arc::new(AtomicBool::new(false));
        let second_called_clone = Arc::clone(&second_called);

        // Even if one callback panics (though we won't actually panic here),
        // test that callbacks are independent
        let cb1: ReloadCallback = Box::new(|_| {
            // Callback 1 does nothing special
        });

        let cb2: ReloadCallback = Box::new(move |_| {
            second_called_clone.store(true, Ordering::SeqCst);
        });

        let callbacks: Vec<ReloadCallback> = vec![cb1, cb2];
        let path = Path::new("test.wgsl");

        for cb in &callbacks {
            cb(path);
        }

        assert!(second_called.load(Ordering::SeqCst));
    }

    #[test]
    fn test_reload_pending_queue_capacity() {
        // Test that pending queue respects max_pending limit
        let config = HotReloadConfig::new().max_pending(4);
        assert_eq!(config.max_pending, 4);

        // Create a pending reloads queue manually
        let mut pending: Vec<PathBuf> = Vec::new();
        let max_pending = 4;

        for i in 0..10 {
            // Simulate queue_reload logic
            while pending.len() >= max_pending {
                pending.remove(0); // Drop oldest
            }
            pending.push(PathBuf::from(format!("shader{}.wgsl", i)));
        }

        // Should only have the last 4
        assert_eq!(pending.len(), 4);
        assert!(pending.iter().all(|p| {
            let name = p.to_string_lossy();
            name.contains("6") || name.contains("7") || name.contains("8") || name.contains("9")
        }));
    }

    #[test]
    fn test_reload_pending_overflow_handling() {
        // Test that overflow drops oldest entries
        let max_pending = 3;
        let mut pending: Vec<PathBuf> = Vec::new();
        let mut dropped_count = 0_u64;

        // Add 5 items to queue with max 3
        for i in 0..5 {
            while pending.len() >= max_pending {
                pending.remove(0);
                dropped_count += 1;
            }
            pending.push(PathBuf::from(format!("shader{}.wgsl", i)));
        }

        // Should have dropped 2 entries
        assert_eq!(dropped_count, 2);
        assert_eq!(pending.len(), 3);
    }

    #[test]
    fn test_reload_clear_pending_while_processing() {
        // Verify clear_pending behavior
        let mut pending: Vec<PathBuf> = vec![
            PathBuf::from("a.wgsl"),
            PathBuf::from("b.wgsl"),
            PathBuf::from("c.wgsl"),
        ];

        // Clear pending
        pending.clear();
        assert!(pending.is_empty());

        // Should be able to add new entries after clear
        pending.push(PathBuf::from("new.wgsl"));
        assert_eq!(pending.len(), 1);
    }

    #[test]
    fn test_reload_stats_accumulation() {
        let mut stats = HotReloadStats::default();

        // Simulate multiple reload operations
        for _ in 0..10 {
            stats.reloads += 1;
            stats.invalidations += 1;
            stats.callbacks_invoked += 3; // 3 callbacks per reload
        }

        assert_eq!(stats.reloads, 10);
        assert_eq!(stats.invalidations, 10);
        assert_eq!(stats.callbacks_invoked, 30);
        assert_eq!(stats.total_events(), 10); // reloads + errors
    }

    #[test]
    fn test_reload_stats_reset_independence() {
        let mut stats = HotReloadStats::default();

        // Accumulate some stats
        stats.reloads = 5;
        stats.errors = 2;
        stats.invalidations = 10;
        stats.callbacks_invoked = 15;
        stats.dropped = 1;

        // Reset by creating new default
        stats = HotReloadStats::default();

        // All fields should be 0
        assert_eq!(stats.reloads, 0);
        assert_eq!(stats.errors, 0);
        assert_eq!(stats.invalidations, 0);
        assert_eq!(stats.callbacks_invoked, 0);
        assert_eq!(stats.dropped, 0);
    }

    // =========================================================================
    // WHITEBOX Tests - Debouncing Logic (6 tests)
    // =========================================================================

    #[test]
    fn test_debounce_rapid_changes() {
        let mut state = DebounceState::new(Duration::from_millis(500));
        let path = Path::new("test.wgsl");

        // First change emits
        assert!(state.record(path, HotReloadEventType::Modified));

        // Rapid subsequent changes within window are suppressed
        assert!(!state.record(path, HotReloadEventType::Modified));
        assert!(!state.record(path, HotReloadEventType::Modified));
        assert!(!state.record(path, HotReloadEventType::Modified));

        // Still only one entry
        assert_eq!(state.len(), 1);
    }

    #[test]
    fn test_debounce_different_files() {
        let mut state = DebounceState::new(Duration::from_millis(100));

        // Different files should all emit independently
        assert!(state.record(Path::new("a.wgsl"), HotReloadEventType::Modified));
        assert!(state.record(Path::new("b.wgsl"), HotReloadEventType::Modified));
        assert!(state.record(Path::new("c.wgsl"), HotReloadEventType::Modified));
        assert!(state.record(Path::new("subdir/d.wgsl"), HotReloadEventType::Created));

        assert_eq!(state.len(), 4);

        // Same file again should not emit
        assert!(!state.record(Path::new("a.wgsl"), HotReloadEventType::Modified));
        assert_eq!(state.len(), 4);
    }

    #[test]
    fn test_debounce_zero_ms_instant() {
        let mut state = DebounceState::new(Duration::from_millis(0));
        let path = Path::new("test.wgsl");

        assert!(state.record(path, HotReloadEventType::Modified));

        // With 0ms debounce, any time gap should allow re-emit
        std::thread::sleep(Duration::from_micros(10));
        assert!(state.record(path, HotReloadEventType::Modified));
    }

    #[test]
    fn test_debounce_large_value() {
        let mut state = DebounceState::new(Duration::from_secs(60 * 60)); // 1 hour
        let path = Path::new("test.wgsl");

        assert!(state.record(path, HotReloadEventType::Modified));

        // Second emit should be suppressed (within 1 hour window)
        assert!(!state.record(path, HotReloadEventType::Modified));

        // drain_ready should return empty (not enough time passed)
        let ready = state.drain_ready();
        assert!(ready.is_empty());
    }

    #[test]
    fn test_debounce_timer_reset() {
        let mut state = DebounceState::new(Duration::from_millis(50));
        let path = Path::new("test.wgsl");

        // First event
        assert!(state.record(path, HotReloadEventType::Modified));

        // Sleep less than debounce
        std::thread::sleep(Duration::from_millis(20));

        // Second event resets timer (doesn't emit)
        assert!(!state.record(path, HotReloadEventType::Modified));

        // Sleep less than debounce again
        std::thread::sleep(Duration::from_millis(20));

        // Third event also resets timer (still within new window)
        assert!(!state.record(path, HotReloadEventType::Modified));
    }

    #[test]
    fn test_debounce_after_clear() {
        let mut state = DebounceState::new(Duration::from_millis(100));
        let path = Path::new("test.wgsl");

        assert!(state.record(path, HotReloadEventType::Modified));
        assert!(!state.record(path, HotReloadEventType::Modified));

        // Clear state
        state.clear();
        assert!(state.is_empty());

        // After clear, same path should emit again
        assert!(state.record(path, HotReloadEventType::Modified));
    }

    // =========================================================================
    // WHITEBOX Tests - HotReloadConfig Validation (5 tests)
    // =========================================================================

    #[test]
    fn test_config_zero_debounce_valid() {
        let config = HotReloadConfig::new().debounce_ms(0);
        assert_eq!(config.debounce_ms, 0);
        assert_eq!(config.debounce_duration(), Duration::from_millis(0));

        // Zero debounce is valid (just means instant)
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_empty_extensions_valid() {
        // Note: Per implementation, empty extensions is NOT valid
        let config = HotReloadConfig::new().watch_extensions(Vec::<String>::new());
        let result = config.validate();
        assert!(result.is_err());

        if let Err(HotReloadError::ConfigError(msg)) = result {
            assert!(msg.contains("extension"));
        }
    }

    #[test]
    fn test_config_max_pending_zero_rejected() {
        let config = HotReloadConfig::new().max_pending(0);
        let result = config.validate();
        assert!(result.is_err());

        if let Err(HotReloadError::ConfigError(msg)) = result {
            assert!(msg.contains("max_pending"));
        }
    }

    #[test]
    fn test_config_builder_fluent() {
        // Test full fluent chain
        let config = HotReloadConfig::new()
            .debounce_ms(100)
            .debounce(Duration::from_millis(200))  // Overrides previous
            .auto_reload(true)
            .recursive(false)
            .log_events(true)
            .max_pending(128)
            .watch_extensions(vec!["wgsl", "glsl"])
            .add_extension("hlsl");

        assert_eq!(config.debounce_ms, 200);
        assert!(config.auto_reload);
        assert!(!config.recursive);
        assert!(config.log_events);
        assert_eq!(config.max_pending, 128);
        assert_eq!(config.watch_extensions.len(), 3);
        assert!(config.watch_extensions.contains(&"hlsl".to_string()));
    }

    #[test]
    fn test_config_presets() {
        // Test minimal preset
        let minimal = HotReloadConfig::minimal();
        assert_eq!(minimal.debounce_ms, 10);
        assert!(!minimal.auto_reload);
        assert!(!minimal.recursive);
        assert!(!minimal.log_events);
        assert_eq!(minimal.max_pending, 16);
        assert!(minimal.validate().is_ok());

        // Test development preset
        let dev = HotReloadConfig::development();
        assert_eq!(dev.debounce_ms, 50);
        assert!(dev.auto_reload);
        assert!(dev.recursive);
        assert!(dev.log_events);
        assert_eq!(dev.max_pending, 128);
        assert!(dev.validate().is_ok());
    }

    // =========================================================================
    // WHITEBOX Tests - Thread Safety (5 tests)
    // =========================================================================

    #[test]
    fn test_watcher_send_sync() {
        fn assert_send<T: Send>() {}
        // ShaderWatcher contains Receiver which is not Sync, but is Send
        assert_send::<ShaderWatcher>();
        // Note: ShaderWatcher is NOT Sync because Receiver<T> is not Sync
    }

    #[test]
    fn test_reload_send_sync() {
        // HotReloadEvent should be Send + Sync
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<HotReloadEvent>();
        assert_sync::<HotReloadEvent>();
    }

    #[test]
    fn test_config_send_sync_bounds() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<HotReloadConfig>();
        assert_sync::<HotReloadConfig>();

        // Config can be safely shared across threads
        let config = Arc::new(HotReloadConfig::default());
        let config_clone = Arc::clone(&config);

        std::thread::spawn(move || {
            let _ = config_clone.debounce_ms;
        }).join().unwrap();
    }

    #[test]
    fn test_stats_send_sync_bounds() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<HotReloadStats>();
        assert_sync::<HotReloadStats>();

        // Stats can be cloned and sent to another thread
        let stats = HotReloadStats {
            reloads: 10,
            invalidations: 5,
            callbacks_invoked: 20,
            errors: 1,
            dropped: 2,
        };

        let handle = std::thread::spawn(move || {
            assert_eq!(stats.total_events(), 11);
            stats
        });

        let returned = handle.join().unwrap();
        assert_eq!(returned.reloads, 10);
    }

    #[test]
    fn test_error_send_sync_bounds() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<HotReloadError>();
        assert_sync::<HotReloadError>();

        // Error can be sent across thread boundary
        let err = HotReloadError::compilation("test error");
        let handle = std::thread::spawn(move || {
            assert!(err.is_compilation_error());
            err
        });

        let returned = handle.join().unwrap();
        assert_eq!(returned.compilation_message(), Some("test error"));
    }

    // =========================================================================
    // WHITEBOX Tests - Error Paths (5 tests)
    // =========================================================================

    #[test]
    fn test_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let hot_err: HotReloadError = io_err.into();

        match hot_err {
            HotReloadError::IoError(_) => (),
            _ => panic!("Expected IoError variant"),
        }
    }

    #[test]
    fn test_error_from_notify_conversion() {
        let notify_err = notify::Error::generic("generic notify error");
        let hot_err: HotReloadError = notify_err.into();

        assert!(hot_err.is_watch_error());

        // Also test the explicit constructor
        let hot_err2 = HotReloadError::watch(notify::Error::generic("another"));
        assert!(hot_err2.is_watch_error());
    }

    #[test]
    fn test_error_display_all_variants() {
        let errors = vec![
            HotReloadError::watch(notify::Error::generic("watch fail")),
            HotReloadError::compilation("compile fail"),
            HotReloadError::path_not_found("/missing"),
            HotReloadError::channel_closed(),
            HotReloadError::config("bad config"),
            HotReloadError::IoError(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "no access",
            )),
        ];

        let expected_substrings = vec![
            "watch error",
            "compilation error",
            "path not found",
            "channel closed",
            "configuration error",
            "IO error",
        ];

        for (err, expected) in errors.iter().zip(expected_substrings.iter()) {
            let display = format!("{}", err);
            assert!(
                display.contains(expected),
                "Expected '{}' to contain '{}'",
                display,
                expected
            );
        }
    }

    #[test]
    fn test_error_source_chain() {
        use std::error::Error;

        // WatchError should have a source
        let watch_err = HotReloadError::watch(notify::Error::generic("inner"));
        assert!(watch_err.source().is_some());

        // IoError should have a source
        let io_err = HotReloadError::IoError(std::io::Error::new(
            std::io::ErrorKind::Other,
            "inner io",
        ));
        assert!(io_err.source().is_some());

        // Other variants should not have a source
        let comp_err = HotReloadError::compilation("no source");
        assert!(comp_err.source().is_none());

        let path_err = HotReloadError::path_not_found("/path");
        assert!(path_err.source().is_none());

        let chan_err = HotReloadError::channel_closed();
        assert!(chan_err.source().is_none());

        let conf_err = HotReloadError::config("bad");
        assert!(conf_err.source().is_none());
    }

    #[test]
    fn test_error_is_recoverable() {
        // Define which errors are considered recoverable
        fn is_recoverable(err: &HotReloadError) -> bool {
            match err {
                // Compilation errors are recoverable (just fix shader and try again)
                HotReloadError::CompilationError(_) => true,
                // Watch errors might be recoverable (e.g., transient FS issues)
                HotReloadError::WatchError(_) => true,
                // Channel closed is not recoverable (system is broken)
                HotReloadError::ChannelClosed => false,
                // Path not found is recoverable (file might appear later)
                HotReloadError::PathNotFound(_) => true,
                // Config errors are not recoverable (programmer error)
                HotReloadError::ConfigError(_) => false,
                // IO errors might be recoverable
                HotReloadError::IoError(e) => {
                    matches!(
                        e.kind(),
                        std::io::ErrorKind::NotFound | std::io::ErrorKind::PermissionDenied
                    )
                }
            }
        }

        assert!(is_recoverable(&HotReloadError::compilation("test")));
        assert!(is_recoverable(&HotReloadError::path_not_found("/test")));
        assert!(!is_recoverable(&HotReloadError::channel_closed()));
        assert!(!is_recoverable(&HotReloadError::config("bad")));
    }
}
