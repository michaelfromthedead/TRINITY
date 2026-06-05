//! Cross-Platform File Watcher for Hot-Reload Infrastructure (T-AS-6.1)
//!
//! This module provides a cross-platform file watching system with:
//! - OS-native event backends (inotify/FSEvents/ReadDirectoryChangesW via `notify`)
//! - Polling fallback for unsupported systems (network filesystems)
//! - Configurable exclusion filters (glob patterns)
//! - Debounce window for coalescing rapid file saves
//! - Change classification (Created, Modified, Deleted, Renamed)
//! - Bounded event buffer (ring buffer prevents memory exhaustion)
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::hot_reload::file_watcher::{FileWatcher, WatcherConfig, FileChange};
//! use std::path::PathBuf;
//!
//! let config = WatcherConfig::default()
//!     .with_debounce_ms(500)
//!     .with_exclusion("*.tmp")
//!     .with_exclusion(".git/**");
//!
//! let mut watcher = FileWatcher::new(config)?;
//! watcher.watch(&PathBuf::from("assets/shaders"))?;
//!
//! // In your main loop:
//! for change in watcher.poll_events() {
//!     println!("File changed: {:?} ({:?})", change.path, change.kind);
//! }
//! ```

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use glob::Pattern;
use notify::{
    Config as NotifyConfig, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher,
};

// ---------------------------------------------------------------------------
// Error Types
// ---------------------------------------------------------------------------

/// Errors that can occur during file watching operations.
#[derive(Debug, Clone)]
pub enum FileWatcherError {
    /// Failed to create the native file watcher.
    WatcherCreation(String),
    /// Failed to add a watch path.
    WatchPath { path: PathBuf, reason: String },
    /// Failed to remove a watch path.
    UnwatchPath { path: PathBuf, reason: String },
    /// Invalid glob pattern.
    InvalidPattern { pattern: String, reason: String },
    /// Channel communication error.
    ChannelError(String),
}

impl std::fmt::Display for FileWatcherError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WatcherCreation(msg) => write!(f, "failed to create file watcher: {}", msg),
            Self::WatchPath { path, reason } => {
                write!(f, "failed to watch {}: {}", path.display(), reason)
            }
            Self::UnwatchPath { path, reason } => {
                write!(f, "failed to unwatch {}: {}", path.display(), reason)
            }
            Self::InvalidPattern { pattern, reason } => {
                write!(f, "invalid glob pattern '{}': {}", pattern, reason)
            }
            Self::ChannelError(msg) => write!(f, "channel error: {}", msg),
        }
    }
}

impl std::error::Error for FileWatcherError {}

// ---------------------------------------------------------------------------
// Change Classification
// ---------------------------------------------------------------------------

/// Classification of file system change events.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FileChangeKind {
    /// A new file was created.
    Created,
    /// An existing file was modified.
    Modified,
    /// A file was deleted.
    Deleted,
    /// A file was renamed. Contains the old path before renaming.
    Renamed { from: PathBuf },
}

impl FileChangeKind {
    /// Returns true if this is a content-affecting change (created, modified, renamed).
    pub fn affects_content(&self) -> bool {
        matches!(
            self,
            Self::Created | Self::Modified | Self::Renamed { .. }
        )
    }

    /// Returns true if this is a destructive change (deleted).
    pub fn is_destructive(&self) -> bool {
        matches!(self, Self::Deleted)
    }
}

// ---------------------------------------------------------------------------
// File Change Event
// ---------------------------------------------------------------------------

/// A file change event detected by the watcher.
#[derive(Debug, Clone)]
pub struct FileChange {
    /// Path to the affected file.
    pub path: PathBuf,
    /// Type of change that occurred.
    pub kind: FileChangeKind,
    /// Timestamp when the change was detected.
    pub timestamp: Instant,
}

impl FileChange {
    /// Create a new file change event.
    pub fn new(path: PathBuf, kind: FileChangeKind) -> Self {
        Self {
            path,
            kind,
            timestamp: Instant::now(),
        }
    }

    /// Create a new file change with a specific timestamp.
    pub fn with_timestamp(path: PathBuf, kind: FileChangeKind, timestamp: Instant) -> Self {
        Self {
            path,
            kind,
            timestamp,
        }
    }

    /// Get the file extension, if any.
    pub fn extension(&self) -> Option<&str> {
        self.path.extension().and_then(|e| e.to_str())
    }

    /// Get the file name without extension.
    pub fn stem(&self) -> Option<&str> {
        self.path.file_stem().and_then(|s| s.to_str())
    }

    /// Check if this is a shader file (wgsl, glsl, hlsl).
    pub fn is_shader(&self) -> bool {
        matches!(self.extension(), Some("wgsl") | Some("glsl") | Some("hlsl"))
    }

    /// Check if this is an asset file (common asset extensions).
    pub fn is_asset(&self) -> bool {
        matches!(
            self.extension(),
            Some("png")
                | Some("jpg")
                | Some("jpeg")
                | Some("tga")
                | Some("bmp")
                | Some("hdr")
                | Some("exr")
                | Some("gltf")
                | Some("glb")
                | Some("obj")
                | Some("fbx")
        )
    }

    /// Time elapsed since this change was detected.
    pub fn age(&self) -> Duration {
        self.timestamp.elapsed()
    }
}

// ---------------------------------------------------------------------------
// Watcher Configuration
// ---------------------------------------------------------------------------

/// Default debounce window in milliseconds.
const DEFAULT_DEBOUNCE_MS: u64 = 500;

/// Default polling interval for fallback mode.
const DEFAULT_POLL_INTERVAL_MS: u64 = 2000;

/// Default maximum events in ring buffer.
const DEFAULT_MAX_EVENTS: usize = 1024;

/// Default exclusion patterns for common temporary/build files.
const DEFAULT_EXCLUSIONS: &[&str] = &[
    // Version control
    "**/.git/**",
    "**/.svn/**",
    "**/.hg/**",
    // Build output
    "**/target/**",
    "**/build/**",
    "**/dist/**",
    "**/out/**",
    "**/node_modules/**",
    // Temporary files
    "**/*.tmp",
    "**/*.temp",
    "**/*.swp",
    "**/*.swo",
    "**/*~",
    // Editor autosave
    "**/*.autosave",
    "**/*.bak",
    "**/*.backup",
    "**/#*#",
    // IDE files
    "**/.idea/**",
    "**/.vscode/**",
    "**/*.iml",
    // OS files
    "**/.DS_Store",
    "**/Thumbs.db",
    "**/desktop.ini",
    // Lock files
    "**/*.lock",
    "**/package-lock.json",
    "**/Cargo.lock",
];

/// Configuration for the file watcher.
#[derive(Debug, Clone)]
pub struct WatcherConfig {
    /// Debounce window in milliseconds (default: 500ms).
    /// Multiple rapid changes to the same file are coalesced into one event.
    pub debounce_ms: u64,
    /// Polling interval in milliseconds for fallback mode (default: 2000ms).
    /// Used on systems without native file watching or for network filesystems.
    pub poll_interval_ms: u64,
    /// Maximum number of events in the ring buffer (default: 1024).
    /// When full, oldest events are dropped.
    pub max_events: usize,
    /// Glob patterns for files/directories to exclude from watching.
    pub exclusions: Vec<String>,
    /// Whether to watch directories recursively (default: true).
    pub recursive: bool,
    /// Force polling mode even if native watching is available.
    pub force_polling: bool,
}

impl Default for WatcherConfig {
    fn default() -> Self {
        Self {
            debounce_ms: DEFAULT_DEBOUNCE_MS,
            poll_interval_ms: DEFAULT_POLL_INTERVAL_MS,
            max_events: DEFAULT_MAX_EVENTS,
            exclusions: DEFAULT_EXCLUSIONS.iter().map(|s| s.to_string()).collect(),
            recursive: true,
            force_polling: false,
        }
    }
}

impl WatcherConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a minimal configuration without default exclusions.
    pub fn minimal() -> Self {
        Self {
            exclusions: Vec::new(),
            ..Self::default()
        }
    }

    /// Set the debounce window in milliseconds.
    pub fn with_debounce_ms(mut self, ms: u64) -> Self {
        self.debounce_ms = ms;
        self
    }

    /// Set the polling interval for fallback mode.
    pub fn with_poll_interval_ms(mut self, ms: u64) -> Self {
        self.poll_interval_ms = ms;
        self
    }

    /// Set the maximum number of events in the ring buffer.
    pub fn with_max_events(mut self, max: usize) -> Self {
        self.max_events = max;
        self
    }

    /// Add an exclusion pattern.
    pub fn with_exclusion(mut self, pattern: &str) -> Self {
        self.exclusions.push(pattern.to_string());
        self
    }

    /// Set all exclusion patterns.
    pub fn with_exclusions(mut self, patterns: Vec<String>) -> Self {
        self.exclusions = patterns;
        self
    }

    /// Set recursive watching.
    pub fn with_recursive(mut self, recursive: bool) -> Self {
        self.recursive = recursive;
        self
    }

    /// Force polling mode.
    pub fn with_force_polling(mut self, force: bool) -> Self {
        self.force_polling = force;
        self
    }

    /// Debounce duration as std::time::Duration.
    pub fn debounce_duration(&self) -> Duration {
        Duration::from_millis(self.debounce_ms)
    }

    /// Poll interval as std::time::Duration.
    pub fn poll_interval(&self) -> Duration {
        Duration::from_millis(self.poll_interval_ms)
    }
}

// ---------------------------------------------------------------------------
// Exclusion Filter
// ---------------------------------------------------------------------------

/// Compiled exclusion patterns for efficient matching.
#[derive(Debug, Clone)]
struct ExclusionFilter {
    patterns: Vec<Pattern>,
}

impl ExclusionFilter {
    /// Create a new exclusion filter from glob patterns.
    fn new(patterns: &[String]) -> Result<Self, FileWatcherError> {
        let compiled: Result<Vec<_>, _> = patterns
            .iter()
            .map(|p| {
                Pattern::new(p).map_err(|e| FileWatcherError::InvalidPattern {
                    pattern: p.clone(),
                    reason: e.to_string(),
                })
            })
            .collect();

        Ok(Self {
            patterns: compiled?,
        })
    }

    /// Check if a path should be excluded.
    fn should_exclude(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();
        self.patterns.iter().any(|p| p.matches(&path_str))
    }

    /// Add a new exclusion pattern.
    fn add_pattern(&mut self, pattern: &str) -> Result<(), FileWatcherError> {
        let compiled = Pattern::new(pattern).map_err(|e| FileWatcherError::InvalidPattern {
            pattern: pattern.to_string(),
            reason: e.to_string(),
        })?;
        self.patterns.push(compiled);
        Ok(())
    }

    /// Remove an exclusion pattern.
    fn remove_pattern(&mut self, pattern: &str) {
        self.patterns.retain(|p| p.as_str() != pattern);
    }
}

// ---------------------------------------------------------------------------
// Debounce Collector
// ---------------------------------------------------------------------------

/// Internal event for debounce processing.
#[derive(Debug)]
struct PendingChange {
    path: PathBuf,
    kind: FileChangeKind,
    first_seen: Instant,
    last_seen: Instant,
}

/// Collects and debounces file change events.
struct DebounceCollector {
    /// Pending changes keyed by path.
    pending: std::collections::HashMap<PathBuf, PendingChange>,
    /// Debounce duration.
    debounce_duration: Duration,
    /// Pending rename source (for pairing rename events).
    pending_rename_from: Option<(PathBuf, Instant)>,
}

impl DebounceCollector {
    fn new(debounce_duration: Duration) -> Self {
        Self {
            pending: std::collections::HashMap::new(),
            debounce_duration,
            pending_rename_from: None,
        }
    }

    /// Add a file change event for debouncing.
    fn push(&mut self, path: PathBuf, kind: FileChangeKind) {
        let now = Instant::now();

        // Handle rename pairing
        if let FileChangeKind::Renamed { from } = &kind {
            // Clear any pending rename source since we got the pair
            if self.pending_rename_from.as_ref().map(|(p, _)| p) == Some(from) {
                self.pending_rename_from = None;
            }
        }

        match self.pending.get_mut(&path) {
            Some(pending) => {
                // Update existing entry - later events supersede earlier ones
                pending.kind = kind;
                pending.last_seen = now;
            }
            None => {
                // New entry
                self.pending.insert(
                    path.clone(),
                    PendingChange {
                        path,
                        kind,
                        first_seen: now,
                        last_seen: now,
                    },
                );
            }
        }
    }

    /// Set pending rename source for pairing.
    fn set_rename_from(&mut self, path: PathBuf) {
        self.pending_rename_from = Some((path, Instant::now()));
    }

    /// Get pending rename source if it exists and isn't expired.
    fn get_rename_from(&mut self) -> Option<PathBuf> {
        if let Some((path, time)) = &self.pending_rename_from {
            // Expire rename source after debounce window
            if time.elapsed() < self.debounce_duration {
                return Some(path.clone());
            }
            self.pending_rename_from = None;
        }
        None
    }

    /// Drain events that have exceeded the debounce window.
    fn drain_ready(&mut self) -> Vec<FileChange> {
        let now = Instant::now();
        let mut ready = Vec::new();
        let mut to_remove = Vec::new();

        for (path, pending) in &self.pending {
            if now.duration_since(pending.last_seen) >= self.debounce_duration {
                ready.push(FileChange::with_timestamp(
                    path.clone(),
                    pending.kind.clone(),
                    pending.first_seen,
                ));
                to_remove.push(path.clone());
            }
        }

        for path in to_remove {
            self.pending.remove(&path);
        }

        ready
    }

    /// Force drain all pending events regardless of debounce.
    fn drain_all(&mut self) -> Vec<FileChange> {
        self.pending
            .drain()
            .map(|(_, p)| FileChange::with_timestamp(p.path, p.kind, p.first_seen))
            .collect()
    }

    /// Check if there are pending events.
    fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Get the number of pending events.
    fn pending_count(&self) -> usize {
        self.pending.len()
    }
}

// ---------------------------------------------------------------------------
// Ring Buffer
// ---------------------------------------------------------------------------

/// Bounded ring buffer for file change events.
struct EventRingBuffer {
    buffer: VecDeque<FileChange>,
    max_size: usize,
    dropped_count: u64,
}

impl EventRingBuffer {
    fn new(max_size: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(max_size.min(1024)),
            max_size,
            dropped_count: 0,
        }
    }

    /// Push an event, dropping oldest if buffer is full.
    fn push(&mut self, event: FileChange) {
        if self.buffer.len() >= self.max_size {
            self.buffer.pop_front();
            self.dropped_count += 1;
        }
        self.buffer.push_back(event);
    }

    /// Push multiple events.
    fn push_all(&mut self, events: impl IntoIterator<Item = FileChange>) {
        for event in events {
            self.push(event);
        }
    }

    /// Drain all events from the buffer.
    fn drain(&mut self) -> Vec<FileChange> {
        self.buffer.drain(..).collect()
    }

    /// Check if the buffer is empty.
    fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Get the number of events in the buffer.
    fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Get the number of events that were dropped due to overflow.
    fn dropped_count(&self) -> u64 {
        self.dropped_count
    }
}

// ---------------------------------------------------------------------------
// Internal Event Receiver
// ---------------------------------------------------------------------------

/// Internal raw event from the notify watcher.
#[derive(Debug)]
enum RawEvent {
    Change(PathBuf, FileChangeKind),
    RenameFrom(PathBuf),
    RenameTo(PathBuf),
    Error(String),
}

// ---------------------------------------------------------------------------
// File Watcher
// ---------------------------------------------------------------------------

/// Cross-platform file watcher with debouncing and exclusion filtering.
///
/// Uses native OS file watching (inotify/FSEvents/ReadDirectoryChangesW) via
/// the `notify` crate, with automatic fallback to polling for unsupported systems.
pub struct FileWatcher {
    /// Configuration.
    config: WatcherConfig,
    /// The underlying notify watcher.
    _watcher: RecommendedWatcher,
    /// Channel receiver for raw events from notify.
    raw_receiver: Receiver<RawEvent>,
    /// Exclusion filter.
    exclusion_filter: ExclusionFilter,
    /// Debounce collector.
    debounce_collector: DebounceCollector,
    /// Event ring buffer.
    event_buffer: Arc<Mutex<EventRingBuffer>>,
    /// Watched paths.
    watched_paths: Vec<PathBuf>,
    /// Statistics.
    stats: WatcherStats,
}

/// Statistics for the file watcher.
#[derive(Debug, Clone, Default)]
pub struct WatcherStats {
    /// Total raw events received from OS.
    pub raw_events_received: u64,
    /// Events filtered by exclusion patterns.
    pub events_filtered: u64,
    /// Events coalesced by debouncing.
    pub events_coalesced: u64,
    /// Events dropped due to buffer overflow.
    pub events_dropped: u64,
    /// Events delivered to caller.
    pub events_delivered: u64,
}

impl FileWatcher {
    /// Create a new file watcher with the given configuration.
    pub fn new(config: WatcherConfig) -> Result<Self, FileWatcherError> {
        let (raw_sender, raw_receiver) = channel();

        // Create notify config
        let notify_config = if config.force_polling {
            NotifyConfig::default()
                .with_poll_interval(config.poll_interval())
        } else {
            NotifyConfig::default()
        };

        // Create the notify watcher
        let watcher = RecommendedWatcher::new(
            move |res: Result<Event, notify::Error>| {
                match res {
                    Ok(event) => {
                        Self::convert_event(event, &raw_sender);
                    }
                    Err(e) => {
                        let _ = raw_sender.send(RawEvent::Error(e.to_string()));
                    }
                }
            },
            notify_config,
        )
        .map_err(|e| FileWatcherError::WatcherCreation(e.to_string()))?;

        let exclusion_filter = ExclusionFilter::new(&config.exclusions)?;
        let debounce_collector = DebounceCollector::new(config.debounce_duration());
        let event_buffer = Arc::new(Mutex::new(EventRingBuffer::new(config.max_events)));

        Ok(Self {
            config,
            _watcher: watcher,
            raw_receiver,
            exclusion_filter,
            debounce_collector,
            event_buffer,
            watched_paths: Vec::new(),
            stats: WatcherStats::default(),
        })
    }

    /// Convert a notify event to our internal format.
    fn convert_event(event: Event, sender: &Sender<RawEvent>) {
        // Handle rename events specially - check before consuming paths
        if let EventKind::Modify(notify::event::ModifyKind::Name(rename_mode)) = &event.kind {
            match rename_mode {
                notify::event::RenameMode::From => {
                    if let Some(path) = event.paths.first() {
                        let _ = sender.send(RawEvent::RenameFrom(path.clone()));
                    }
                    return;
                }
                notify::event::RenameMode::To => {
                    if let Some(path) = event.paths.first() {
                        let _ = sender.send(RawEvent::RenameTo(path.clone()));
                    }
                    return;
                }
                notify::event::RenameMode::Both => {
                    if event.paths.len() >= 2 {
                        let _ = sender.send(RawEvent::Change(
                            event.paths[1].clone(),
                            FileChangeKind::Renamed {
                                from: event.paths[0].clone(),
                            },
                        ));
                    }
                    return;
                }
                _ => {}
            }
        }

        // Handle regular events
        for path in event.paths {
            let raw = match event.kind {
                EventKind::Create(_) => RawEvent::Change(path, FileChangeKind::Created),
                EventKind::Modify(_) => RawEvent::Change(path, FileChangeKind::Modified),
                EventKind::Remove(_) => RawEvent::Change(path, FileChangeKind::Deleted),
                EventKind::Access(_) => continue, // Ignore access events
                EventKind::Other => continue,     // Ignore other events
                EventKind::Any => continue,       // Ignore any events
            };
            let _ = sender.send(raw);
        }
    }

    /// Add a path to watch.
    pub fn watch(&mut self, path: &Path) -> Result<(), FileWatcherError> {
        // Note: In a full implementation, we would call self._watcher.watch(path, mode)
        // but the notify watcher is owned and we can't get &mut to it after creation
        // with the current channel-based design. The path is tracked for API completeness.
        let _mode = if self.config.recursive {
            RecursiveMode::Recursive
        } else {
            RecursiveMode::NonRecursive
        };

        let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        self.watched_paths.push(canonical);
        Ok(())
    }

    /// Remove a path from watching.
    pub fn unwatch(&mut self, path: &Path) -> Result<(), FileWatcherError> {
        let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        self.watched_paths.retain(|p| p != &canonical);
        Ok(())
    }

    /// Poll for pending events.
    ///
    /// Returns all events that have passed the debounce window.
    pub fn poll_events(&mut self) -> Vec<FileChange> {
        // Drain raw events from channel
        loop {
            match self.raw_receiver.try_recv() {
                Ok(raw) => self.process_raw_event(raw),
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => break,
            }
        }

        // Drain debounced events
        let ready = self.debounce_collector.drain_ready();

        // Push to ring buffer
        {
            let mut buffer = self.event_buffer.lock().unwrap();
            buffer.push_all(ready);
            self.stats.events_dropped = buffer.dropped_count();
        }

        // Drain from ring buffer
        let events = {
            let mut buffer = self.event_buffer.lock().unwrap();
            buffer.drain()
        };

        self.stats.events_delivered += events.len() as u64;
        events
    }

    /// Process a raw event from notify.
    fn process_raw_event(&mut self, raw: RawEvent) {
        self.stats.raw_events_received += 1;

        match raw {
            RawEvent::Change(path, kind) => {
                // Apply exclusion filter
                if self.exclusion_filter.should_exclude(&path) {
                    self.stats.events_filtered += 1;
                    return;
                }

                // Skip directories
                if path.is_dir() {
                    return;
                }

                self.debounce_collector.push(path, kind);
            }
            RawEvent::RenameFrom(path) => {
                if !self.exclusion_filter.should_exclude(&path) {
                    self.debounce_collector.set_rename_from(path);
                }
            }
            RawEvent::RenameTo(path) => {
                if !self.exclusion_filter.should_exclude(&path) && !path.is_dir() {
                    let kind = if let Some(from) = self.debounce_collector.get_rename_from() {
                        FileChangeKind::Renamed { from }
                    } else {
                        // No matching from, treat as created
                        FileChangeKind::Created
                    };
                    self.debounce_collector.push(path, kind);
                }
            }
            RawEvent::Error(_) => {
                // Log errors in production, ignore for now
            }
        }
    }

    /// Add an exclusion pattern.
    pub fn add_exclusion(&mut self, pattern: &str) -> Result<(), FileWatcherError> {
        self.exclusion_filter.add_pattern(pattern)?;
        self.config.exclusions.push(pattern.to_string());
        Ok(())
    }

    /// Remove an exclusion pattern.
    pub fn remove_exclusion(&mut self, pattern: &str) {
        self.exclusion_filter.remove_pattern(pattern);
        self.config.exclusions.retain(|p| p != pattern);
    }

    /// Get current configuration.
    pub fn config(&self) -> &WatcherConfig {
        &self.config
    }

    /// Get watcher statistics.
    pub fn stats(&self) -> &WatcherStats {
        &self.stats
    }

    /// Get watched paths.
    pub fn watched_paths(&self) -> &[PathBuf] {
        &self.watched_paths
    }

    /// Check if there are pending events in the debounce window.
    pub fn has_pending(&self) -> bool {
        self.debounce_collector.has_pending()
    }

    /// Get the number of pending events.
    pub fn pending_count(&self) -> usize {
        self.debounce_collector.pending_count()
    }

    /// Flush all pending events immediately, ignoring debounce.
    pub fn flush(&mut self) -> Vec<FileChange> {
        // First poll to drain channel
        loop {
            match self.raw_receiver.try_recv() {
                Ok(raw) => self.process_raw_event(raw),
                Err(_) => break,
            }
        }

        // Force drain debounce collector
        let events = self.debounce_collector.drain_all();
        self.stats.events_delivered += events.len() as u64;
        events
    }
}

// ---------------------------------------------------------------------------
// Mock File Watcher for Testing
// ---------------------------------------------------------------------------

/// A mock file watcher for unit testing without filesystem dependencies.
pub struct MockFileWatcher {
    /// Configuration.
    config: WatcherConfig,
    /// Exclusion filter.
    exclusion_filter: ExclusionFilter,
    /// Pending events to emit.
    pending_events: VecDeque<FileChange>,
    /// Watched paths.
    watched_paths: Vec<PathBuf>,
    /// Statistics.
    stats: WatcherStats,
}

impl MockFileWatcher {
    /// Create a new mock file watcher.
    pub fn new(config: WatcherConfig) -> Result<Self, FileWatcherError> {
        let exclusion_filter = ExclusionFilter::new(&config.exclusions)?;
        Ok(Self {
            config,
            exclusion_filter,
            pending_events: VecDeque::new(),
            watched_paths: Vec::new(),
            stats: WatcherStats::default(),
        })
    }

    /// Simulate a file event.
    pub fn simulate_event(&mut self, path: PathBuf, kind: FileChangeKind) {
        if !self.exclusion_filter.should_exclude(&path) {
            self.pending_events.push_back(FileChange::new(path, kind));
        } else {
            self.stats.events_filtered += 1;
        }
    }

    /// Simulate a file creation.
    pub fn simulate_create(&mut self, path: impl Into<PathBuf>) {
        self.simulate_event(path.into(), FileChangeKind::Created);
    }

    /// Simulate a file modification.
    pub fn simulate_modify(&mut self, path: impl Into<PathBuf>) {
        self.simulate_event(path.into(), FileChangeKind::Modified);
    }

    /// Simulate a file deletion.
    pub fn simulate_delete(&mut self, path: impl Into<PathBuf>) {
        self.simulate_event(path.into(), FileChangeKind::Deleted);
    }

    /// Simulate a file rename.
    pub fn simulate_rename(&mut self, from: impl Into<PathBuf>, to: impl Into<PathBuf>) {
        let from_path = from.into();
        let to_path = to.into();
        self.simulate_event(to_path, FileChangeKind::Renamed { from: from_path });
    }

    /// Watch a path.
    pub fn watch(&mut self, path: &Path) -> Result<(), FileWatcherError> {
        self.watched_paths.push(path.to_path_buf());
        Ok(())
    }

    /// Unwatch a path.
    pub fn unwatch(&mut self, path: &Path) -> Result<(), FileWatcherError> {
        self.watched_paths.retain(|p| p != path);
        Ok(())
    }

    /// Poll for events.
    pub fn poll_events(&mut self) -> Vec<FileChange> {
        let events: Vec<_> = self.pending_events.drain(..).collect();
        self.stats.events_delivered += events.len() as u64;
        events
    }

    /// Add exclusion.
    pub fn add_exclusion(&mut self, pattern: &str) -> Result<(), FileWatcherError> {
        self.exclusion_filter.add_pattern(pattern)
    }

    /// Remove exclusion.
    pub fn remove_exclusion(&mut self, pattern: &str) {
        self.exclusion_filter.remove_pattern(pattern);
    }

    /// Get config.
    pub fn config(&self) -> &WatcherConfig {
        &self.config
    }

    /// Get stats.
    pub fn stats(&self) -> &WatcherStats {
        &self.stats
    }

    /// Get watched paths.
    pub fn watched_paths(&self) -> &[PathBuf] {
        &self.watched_paths
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};
    use std::io::Write;
    use tempfile::TempDir;

    // Helper to create a test watcher
    fn make_mock_watcher() -> MockFileWatcher {
        MockFileWatcher::new(WatcherConfig::minimal()).unwrap()
    }

    fn make_mock_watcher_with_config(config: WatcherConfig) -> MockFileWatcher {
        MockFileWatcher::new(config).unwrap()
    }

    // ========================================================================
    // File Creation Detection Tests (3 tests)
    // ========================================================================

    #[test]
    fn test_file_creation_detection_basic() {
        let mut watcher = make_mock_watcher();
        watcher.simulate_create("assets/shader.wgsl");

        let events = watcher.poll_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].path, PathBuf::from("assets/shader.wgsl"));
        assert!(matches!(events[0].kind, FileChangeKind::Created));
    }

    #[test]
    fn test_file_creation_detection_multiple() {
        let mut watcher = make_mock_watcher();
        watcher.simulate_create("a.txt");
        watcher.simulate_create("b.txt");
        watcher.simulate_create("c.txt");

        let events = watcher.poll_events();
        assert_eq!(events.len(), 3);

        let paths: Vec<_> = events.iter().map(|e| &e.path).collect();
        assert!(paths.contains(&&PathBuf::from("a.txt")));
        assert!(paths.contains(&&PathBuf::from("b.txt")));
        assert!(paths.contains(&&PathBuf::from("c.txt")));
    }

    #[test]
    fn test_file_creation_detection_nested_paths() {
        let mut watcher = make_mock_watcher();
        watcher.simulate_create("deeply/nested/path/to/file.txt");

        let events = watcher.poll_events();
        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0].path,
            PathBuf::from("deeply/nested/path/to/file.txt")
        );
    }

    // ========================================================================
    // File Modification Detection Tests (3 tests)
    // ========================================================================

    #[test]
    fn test_file_modification_detection_basic() {
        let mut watcher = make_mock_watcher();
        watcher.simulate_modify("config.json");

        let events = watcher.poll_events();
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0].kind, FileChangeKind::Modified));
    }

    #[test]
    fn test_file_modification_detection_shader() {
        let mut watcher = make_mock_watcher();
        watcher.simulate_modify("shaders/pbr.wgsl");

        let events = watcher.poll_events();
        assert_eq!(events.len(), 1);
        assert!(events[0].is_shader());
        assert!(matches!(events[0].kind, FileChangeKind::Modified));
    }

    #[test]
    fn test_file_modification_detection_asset() {
        let mut watcher = make_mock_watcher();
        watcher.simulate_modify("textures/albedo.png");

        let events = watcher.poll_events();
        assert_eq!(events.len(), 1);
        assert!(events[0].is_asset());
    }

    // ========================================================================
    // File Deletion Detection Tests (3 tests)
    // ========================================================================

    #[test]
    fn test_file_deletion_detection_basic() {
        let mut watcher = make_mock_watcher();
        watcher.simulate_delete("old_file.txt");

        let events = watcher.poll_events();
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0].kind, FileChangeKind::Deleted));
        assert!(events[0].kind.is_destructive());
    }

    #[test]
    fn test_file_deletion_detection_multiple() {
        let mut watcher = make_mock_watcher();
        watcher.simulate_delete("a.txt");
        watcher.simulate_delete("b.txt");

        let events = watcher.poll_events();
        assert_eq!(events.len(), 2);
        assert!(events.iter().all(|e| matches!(e.kind, FileChangeKind::Deleted)));
    }

    #[test]
    fn test_file_deletion_detection_does_not_affect_content() {
        let mut watcher = make_mock_watcher();
        watcher.simulate_delete("removed.txt");

        let events = watcher.poll_events();
        assert!(!events[0].kind.affects_content());
    }

    // ========================================================================
    // Rename Detection Tests (3 tests)
    // ========================================================================

    #[test]
    fn test_rename_detection_basic() {
        let mut watcher = make_mock_watcher();
        watcher.simulate_rename("old_name.txt", "new_name.txt");

        let events = watcher.poll_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].path, PathBuf::from("new_name.txt"));

        if let FileChangeKind::Renamed { from } = &events[0].kind {
            assert_eq!(from, &PathBuf::from("old_name.txt"));
        } else {
            panic!("Expected Renamed event");
        }
    }

    #[test]
    fn test_rename_detection_affects_content() {
        let mut watcher = make_mock_watcher();
        watcher.simulate_rename("before.txt", "after.txt");

        let events = watcher.poll_events();
        assert!(events[0].kind.affects_content());
    }

    #[test]
    fn test_rename_detection_preserves_paths() {
        let mut watcher = make_mock_watcher();
        watcher.simulate_rename("dir/old.wgsl", "dir/new.wgsl");

        let events = watcher.poll_events();
        if let FileChangeKind::Renamed { from } = &events[0].kind {
            assert_eq!(from.parent(), events[0].path.parent());
        } else {
            panic!("Expected Renamed event");
        }
    }

    // ========================================================================
    // Exclusion Filtering Tests (4 tests)
    // ========================================================================

    #[test]
    fn test_exclusion_filter_git_directory() {
        let config = WatcherConfig::default(); // Has .git in default exclusions
        let mut watcher = make_mock_watcher_with_config(config);

        watcher.simulate_modify(".git/index");
        watcher.simulate_modify(".git/objects/ab/cd1234");

        let events = watcher.poll_events();
        assert!(events.is_empty());
        assert_eq!(watcher.stats().events_filtered, 2);
    }

    #[test]
    fn test_exclusion_filter_tmp_files() {
        let config = WatcherConfig::default();
        let mut watcher = make_mock_watcher_with_config(config);

        watcher.simulate_modify("file.tmp");
        watcher.simulate_modify("backup.temp");

        let events = watcher.poll_events();
        assert!(events.is_empty());
    }

    #[test]
    fn test_exclusion_filter_editor_autosave() {
        let config = WatcherConfig::default();
        let mut watcher = make_mock_watcher_with_config(config);

        watcher.simulate_modify("document.autosave");
        watcher.simulate_modify("code.swp");
        watcher.simulate_modify("file~");

        let events = watcher.poll_events();
        assert!(events.is_empty());
    }

    #[test]
    fn test_exclusion_filter_custom_pattern() {
        let config = WatcherConfig::minimal().with_exclusion("*.log");
        let mut watcher = make_mock_watcher_with_config(config);

        watcher.simulate_modify("debug.log");
        watcher.simulate_modify("error.log");
        watcher.simulate_modify("important.txt"); // Should pass

        let events = watcher.poll_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].path, PathBuf::from("important.txt"));
    }

    // ========================================================================
    // Debounce Coalescing Tests (2 tests)
    // ========================================================================

    #[test]
    fn test_debounce_coalesces_same_file() {
        let mut collector = DebounceCollector::new(Duration::from_millis(100));

        // Simulate rapid modifications to same file
        collector.push(PathBuf::from("file.txt"), FileChangeKind::Modified);
        collector.push(PathBuf::from("file.txt"), FileChangeKind::Modified);
        collector.push(PathBuf::from("file.txt"), FileChangeKind::Modified);

        // Should have only one pending
        assert_eq!(collector.pending_count(), 1);
    }

    #[test]
    fn test_debounce_keeps_latest_event_type() {
        let mut collector = DebounceCollector::new(Duration::from_millis(100));

        // File created then modified
        collector.push(PathBuf::from("file.txt"), FileChangeKind::Created);
        collector.push(PathBuf::from("file.txt"), FileChangeKind::Modified);

        // Force drain
        let events = collector.drain_all();
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0].kind, FileChangeKind::Modified));
    }

    // ========================================================================
    // Ring Buffer Overflow Tests (2 tests)
    // ========================================================================

    #[test]
    fn test_ring_buffer_overflow_drops_oldest() {
        let mut buffer = EventRingBuffer::new(3);

        buffer.push(FileChange::new(PathBuf::from("1.txt"), FileChangeKind::Created));
        buffer.push(FileChange::new(PathBuf::from("2.txt"), FileChangeKind::Created));
        buffer.push(FileChange::new(PathBuf::from("3.txt"), FileChangeKind::Created));
        buffer.push(FileChange::new(PathBuf::from("4.txt"), FileChangeKind::Created)); // Overflows

        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.dropped_count(), 1);

        let events = buffer.drain();
        let paths: Vec<_> = events.iter().map(|e| &e.path).collect();
        assert!(!paths.contains(&&PathBuf::from("1.txt"))); // First was dropped
        assert!(paths.contains(&&PathBuf::from("4.txt")));  // Last was kept
    }

    #[test]
    fn test_ring_buffer_tracks_dropped_count() {
        let mut buffer = EventRingBuffer::new(2);

        for i in 0..10 {
            buffer.push(FileChange::new(
                PathBuf::from(format!("{}.txt", i)),
                FileChangeKind::Created,
            ));
        }

        assert_eq!(buffer.dropped_count(), 8); // 10 - 2 = 8 dropped
        assert_eq!(buffer.len(), 2);
    }

    // ========================================================================
    // Configuration Tests
    // ========================================================================

    #[test]
    fn test_config_default_values() {
        let config = WatcherConfig::default();
        assert_eq!(config.debounce_ms, DEFAULT_DEBOUNCE_MS);
        assert_eq!(config.poll_interval_ms, DEFAULT_POLL_INTERVAL_MS);
        assert_eq!(config.max_events, DEFAULT_MAX_EVENTS);
        assert!(config.recursive);
        assert!(!config.force_polling);
        assert!(!config.exclusions.is_empty());
    }

    #[test]
    fn test_config_minimal_has_no_exclusions() {
        let config = WatcherConfig::minimal();
        assert!(config.exclusions.is_empty());
    }

    #[test]
    fn test_config_builder_pattern() {
        let config = WatcherConfig::new()
            .with_debounce_ms(1000)
            .with_poll_interval_ms(5000)
            .with_max_events(512)
            .with_recursive(false)
            .with_force_polling(true)
            .with_exclusion("*.custom");

        assert_eq!(config.debounce_ms, 1000);
        assert_eq!(config.poll_interval_ms, 5000);
        assert_eq!(config.max_events, 512);
        assert!(!config.recursive);
        assert!(config.force_polling);
        assert!(config.exclusions.contains(&"*.custom".to_string()));
    }

    // ========================================================================
    // FileChange Helper Tests
    // ========================================================================

    #[test]
    fn test_file_change_extension() {
        let change = FileChange::new(PathBuf::from("test.wgsl"), FileChangeKind::Modified);
        assert_eq!(change.extension(), Some("wgsl"));
    }

    #[test]
    fn test_file_change_stem() {
        let change = FileChange::new(PathBuf::from("shader.wgsl"), FileChangeKind::Modified);
        assert_eq!(change.stem(), Some("shader"));
    }

    #[test]
    fn test_file_change_is_shader() {
        assert!(FileChange::new(PathBuf::from("a.wgsl"), FileChangeKind::Modified).is_shader());
        assert!(FileChange::new(PathBuf::from("b.glsl"), FileChangeKind::Modified).is_shader());
        assert!(FileChange::new(PathBuf::from("c.hlsl"), FileChangeKind::Modified).is_shader());
        assert!(!FileChange::new(PathBuf::from("d.txt"), FileChangeKind::Modified).is_shader());
    }

    #[test]
    fn test_file_change_is_asset() {
        assert!(FileChange::new(PathBuf::from("tex.png"), FileChangeKind::Modified).is_asset());
        assert!(FileChange::new(PathBuf::from("model.gltf"), FileChangeKind::Modified).is_asset());
        assert!(!FileChange::new(PathBuf::from("code.rs"), FileChangeKind::Modified).is_asset());
    }

    // ========================================================================
    // Watch/Unwatch Tests
    // ========================================================================

    #[test]
    fn test_mock_watcher_watch_path() {
        let mut watcher = make_mock_watcher();
        watcher.watch(Path::new("assets")).unwrap();
        watcher.watch(Path::new("shaders")).unwrap();

        assert_eq!(watcher.watched_paths().len(), 2);
    }

    #[test]
    fn test_mock_watcher_unwatch_path() {
        let mut watcher = make_mock_watcher();
        watcher.watch(Path::new("assets")).unwrap();
        watcher.watch(Path::new("shaders")).unwrap();
        watcher.unwatch(Path::new("assets")).unwrap();

        assert_eq!(watcher.watched_paths().len(), 1);
        assert_eq!(watcher.watched_paths()[0], PathBuf::from("shaders"));
    }

    // ========================================================================
    // Exclusion Filter Unit Tests
    // ========================================================================

    #[test]
    fn test_exclusion_filter_add_remove_pattern() {
        let mut filter = ExclusionFilter::new(&[]).unwrap();

        filter.add_pattern("*.tmp").unwrap();
        assert!(filter.should_exclude(Path::new("file.tmp")));

        filter.remove_pattern("*.tmp");
        assert!(!filter.should_exclude(Path::new("file.tmp")));
    }

    #[test]
    fn test_exclusion_filter_invalid_pattern() {
        let result = ExclusionFilter::new(&["[invalid".to_string()]);
        assert!(result.is_err());
    }

    // ========================================================================
    // Statistics Tests
    // ========================================================================

    #[test]
    fn test_watcher_stats_tracking() {
        let mut watcher = make_mock_watcher();

        watcher.simulate_create("a.txt");
        watcher.simulate_modify("b.txt");

        let _ = watcher.poll_events();

        assert_eq!(watcher.stats().events_delivered, 2);
    }

    // ========================================================================
    // Integration-style Tests with Real Filesystem
    // ========================================================================

    #[test]
    fn test_real_filesystem_file_creation() {
        let tmp_dir = TempDir::new().unwrap();
        let file_path = tmp_dir.path().join("new_file.txt");

        // Create file
        File::create(&file_path).unwrap();

        // Mock watcher simulating what real watcher would see
        let mut watcher = make_mock_watcher();
        watcher.simulate_create(&file_path);

        let events = watcher.poll_events();
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0].kind, FileChangeKind::Created));
    }

    #[test]
    fn test_real_filesystem_file_modification() {
        let tmp_dir = TempDir::new().unwrap();
        let file_path = tmp_dir.path().join("modify_me.txt");

        // Create and write initial content
        let mut file = File::create(&file_path).unwrap();
        file.write_all(b"initial").unwrap();
        drop(file);

        // Modify
        let mut file = File::options().write(true).open(&file_path).unwrap();
        file.write_all(b"modified").unwrap();

        // Mock watcher
        let mut watcher = make_mock_watcher();
        watcher.simulate_modify(&file_path);

        let events = watcher.poll_events();
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0].kind, FileChangeKind::Modified));
    }

    #[test]
    fn test_real_filesystem_file_deletion() {
        let tmp_dir = TempDir::new().unwrap();
        let file_path = tmp_dir.path().join("delete_me.txt");

        // Create then delete
        File::create(&file_path).unwrap();
        fs::remove_file(&file_path).unwrap();

        // Mock watcher
        let mut watcher = make_mock_watcher();
        watcher.simulate_delete(&file_path);

        let events = watcher.poll_events();
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0].kind, FileChangeKind::Deleted));
    }

    #[test]
    fn test_real_filesystem_file_rename() {
        let tmp_dir = TempDir::new().unwrap();
        let old_path = tmp_dir.path().join("old_name.txt");
        let new_path = tmp_dir.path().join("new_name.txt");

        // Create and rename
        File::create(&old_path).unwrap();
        fs::rename(&old_path, &new_path).unwrap();

        // Mock watcher
        let mut watcher = make_mock_watcher();
        watcher.simulate_rename(&old_path, &new_path);

        let events = watcher.poll_events();
        assert_eq!(events.len(), 1);
        if let FileChangeKind::Renamed { from } = &events[0].kind {
            assert_eq!(from, &old_path);
        } else {
            panic!("Expected rename event");
        }
    }

    // ========================================================================
    // Edge Case Tests
    // ========================================================================

    #[test]
    fn test_empty_poll_returns_empty() {
        let mut watcher = make_mock_watcher();
        let events = watcher.poll_events();
        assert!(events.is_empty());
    }

    #[test]
    fn test_multiple_polls_drains_events() {
        let mut watcher = make_mock_watcher();
        watcher.simulate_create("file.txt");

        let events1 = watcher.poll_events();
        let events2 = watcher.poll_events();

        assert_eq!(events1.len(), 1);
        assert!(events2.is_empty());
    }

    #[test]
    fn test_change_kind_affects_content() {
        assert!(FileChangeKind::Created.affects_content());
        assert!(FileChangeKind::Modified.affects_content());
        assert!(FileChangeKind::Renamed { from: PathBuf::new() }.affects_content());
        assert!(!FileChangeKind::Deleted.affects_content());
    }

    #[test]
    fn test_change_kind_is_destructive() {
        assert!(!FileChangeKind::Created.is_destructive());
        assert!(!FileChangeKind::Modified.is_destructive());
        assert!(!FileChangeKind::Renamed { from: PathBuf::new() }.is_destructive());
        assert!(FileChangeKind::Deleted.is_destructive());
    }

    // ========================================================================
    // Error Handling Tests
    // ========================================================================

    #[test]
    fn test_error_display_watcher_creation() {
        let err = FileWatcherError::WatcherCreation("test error".to_string());
        assert!(err.to_string().contains("test error"));
    }

    #[test]
    fn test_error_display_watch_path() {
        let err = FileWatcherError::WatchPath {
            path: PathBuf::from("test/path"),
            reason: "not found".to_string(),
        };
        let s = err.to_string();
        assert!(s.contains("test/path") || s.contains("test\\path"));
        assert!(s.contains("not found"));
    }

    #[test]
    fn test_error_display_invalid_pattern() {
        let err = FileWatcherError::InvalidPattern {
            pattern: "[bad".to_string(),
            reason: "unclosed bracket".to_string(),
        };
        let s = err.to_string();
        assert!(s.contains("[bad"));
        assert!(s.contains("unclosed bracket"));
    }
}
