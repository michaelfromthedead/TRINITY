//! File watcher for detecting code changes.
//!
//! Watches for file changes and sends events to the daemon.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use super::DaemonEvent;

/// Configuration for the file watcher.
#[derive(Debug, Clone)]
pub struct WatcherConfig {
    /// Root directory to watch.
    pub root: PathBuf,
    /// File extensions to watch.
    pub extensions: Vec<String>,
    /// Directories to ignore.
    pub ignore_dirs: Vec<String>,
    /// Debounce time in milliseconds.
    pub debounce_ms: u64,
    /// Poll interval in milliseconds.
    pub poll_interval_ms: u64,
}

impl Default for WatcherConfig {
    fn default() -> Self {
        Self {
            root: PathBuf::from("."),
            extensions: vec![
                "rs".to_string(),
                "py".to_string(),
                "wgsl".to_string(),
            ],
            ignore_dirs: vec![
                "target".to_string(),
                "node_modules".to_string(),
                ".git".to_string(),
                "__pycache__".to_string(),
                ".venv".to_string(),
            ],
            debounce_ms: 100,
            poll_interval_ms: 500,
        }
    }
}

impl WatcherConfig {
    /// Create a config for a specific root.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            ..Default::default()
        }
    }

    /// Add extensions to watch.
    pub fn extensions(mut self, exts: Vec<String>) -> Self {
        self.extensions = exts;
        self
    }

    /// Set debounce time.
    pub fn debounce(mut self, ms: u64) -> Self {
        self.debounce_ms = ms;
        self
    }

    /// Add directories to ignore.
    pub fn ignore(mut self, dirs: Vec<String>) -> Self {
        self.ignore_dirs = dirs;
        self
    }
}

/// A file change event.
#[derive(Debug, Clone)]
pub struct FileChange {
    /// Path to the file.
    pub path: PathBuf,
    /// Type of change.
    pub kind: ChangeKind,
    /// When the change was detected.
    pub timestamp: Instant,
}

/// Type of file change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeKind {
    /// File was created.
    Created,
    /// File was modified.
    Modified,
    /// File was deleted.
    Deleted,
}

/// File watcher that runs in a separate thread.
pub struct FileWatcher {
    /// Configuration.
    config: WatcherConfig,
    /// Channel to send events.
    sender: Sender<FileChange>,
    /// Handle to the watcher thread.
    handle: Option<JoinHandle<()>>,
    /// Running flag.
    running: Arc<Mutex<bool>>,
    /// File modification times for change detection.
    mtimes: Arc<Mutex<HashMap<PathBuf, std::time::SystemTime>>>,
}

impl FileWatcher {
    /// Create a new file watcher.
    pub fn new(config: WatcherConfig) -> (Self, Receiver<FileChange>) {
        let (sender, receiver) = mpsc::channel();

        let watcher = Self {
            config,
            sender,
            handle: None,
            running: Arc::new(Mutex::new(false)),
            mtimes: Arc::new(Mutex::new(HashMap::new())),
        };

        (watcher, receiver)
    }

    /// Start watching in a background thread.
    pub fn start(&mut self) {
        if self.handle.is_some() {
            return;
        }

        *self.running.lock().unwrap() = true;

        let config = self.config.clone();
        let sender = self.sender.clone();
        let running = self.running.clone();
        let mtimes = self.mtimes.clone();

        let handle = thread::spawn(move || {
            watch_loop(&config, &sender, &running, &mtimes);
        });

        self.handle = Some(handle);
    }

    /// Stop the watcher.
    pub fn stop(&mut self) {
        *self.running.lock().unwrap() = false;

        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }

    /// Check if the watcher is running.
    pub fn is_running(&self) -> bool {
        *self.running.lock().unwrap()
    }

    /// Get the number of tracked files.
    pub fn tracked_files(&self) -> usize {
        self.mtimes.lock().unwrap().len()
    }
}

/// Main watch loop.
fn watch_loop(
    config: &WatcherConfig,
    sender: &Sender<FileChange>,
    running: &Arc<Mutex<bool>>,
    mtimes: &Arc<Mutex<HashMap<PathBuf, std::time::SystemTime>>>,
) {
    // Initial scan
    scan_directory(&config.root, config, mtimes);

    while *running.lock().unwrap() {
        // Check for changes
        let changes = detect_changes(&config.root, config, mtimes);

        for change in changes {
            let _ = sender.send(change);
        }

        thread::sleep(Duration::from_millis(config.poll_interval_ms));
    }
}

/// Scan a directory for files to track.
fn scan_directory(
    dir: &Path,
    config: &WatcherConfig,
    mtimes: &Arc<Mutex<HashMap<PathBuf, std::time::SystemTime>>>,
) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();

            if path.is_dir() {
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                if !config.ignore_dirs.iter().any(|d| d == name.as_ref()) {
                    scan_directory(&path, config, mtimes);
                }
            } else if should_watch(&path, config) {
                if let Ok(meta) = std::fs::metadata(&path) {
                    if let Ok(mtime) = meta.modified() {
                        mtimes.lock().unwrap().insert(path, mtime);
                    }
                }
            }
        }
    }
}

/// Check if a file should be watched.
fn should_watch(path: &Path, config: &WatcherConfig) -> bool {
    if let Some(ext) = path.extension() {
        let ext_str = ext.to_string_lossy();
        config.extensions.iter().any(|e| e == ext_str.as_ref())
    } else {
        false
    }
}

/// Detect changes in watched files.
fn detect_changes(
    dir: &Path,
    config: &WatcherConfig,
    mtimes: &Arc<Mutex<HashMap<PathBuf, std::time::SystemTime>>>,
) -> Vec<FileChange> {
    let mut changes = Vec::new();
    let mut current_files: HashMap<PathBuf, std::time::SystemTime> = HashMap::new();

    // Scan current state
    collect_files(dir, config, &mut current_files);

    let mut stored = mtimes.lock().unwrap();

    // Check for modifications and deletions
    for (path, old_mtime) in stored.iter() {
        if let Some(&new_mtime) = current_files.get(path) {
            if new_mtime != *old_mtime {
                changes.push(FileChange {
                    path: path.clone(),
                    kind: ChangeKind::Modified,
                    timestamp: Instant::now(),
                });
            }
        } else {
            changes.push(FileChange {
                path: path.clone(),
                kind: ChangeKind::Deleted,
                timestamp: Instant::now(),
            });
        }
    }

    // Check for new files
    for (path, _) in &current_files {
        if !stored.contains_key(path) {
            changes.push(FileChange {
                path: path.clone(),
                kind: ChangeKind::Created,
                timestamp: Instant::now(),
            });
        }
    }

    // Update stored mtimes
    *stored = current_files;

    changes
}

/// Collect all watchable files.
fn collect_files(
    dir: &Path,
    config: &WatcherConfig,
    files: &mut HashMap<PathBuf, std::time::SystemTime>,
) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();

            if path.is_dir() {
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                if !config.ignore_dirs.iter().any(|d| d == name.as_ref()) {
                    collect_files(&path, config, files);
                }
            } else if should_watch(&path, config) {
                if let Ok(meta) = std::fs::metadata(&path) {
                    if let Ok(mtime) = meta.modified() {
                        files.insert(path, mtime);
                    }
                }
            }
        }
    }
}

/// Debouncer for rapid file changes.
pub struct Debouncer {
    /// Last event time per file.
    last_events: HashMap<PathBuf, Instant>,
    /// Debounce duration.
    debounce: Duration,
}

impl Debouncer {
    /// Create a new debouncer.
    pub fn new(debounce_ms: u64) -> Self {
        Self {
            last_events: HashMap::new(),
            debounce: Duration::from_millis(debounce_ms),
        }
    }

    /// Check if an event should be processed (not debounced).
    pub fn should_process(&mut self, change: &FileChange) -> bool {
        let now = Instant::now();

        if let Some(&last) = self.last_events.get(&change.path) {
            if now.duration_since(last) < self.debounce {
                return false;
            }
        }

        self.last_events.insert(change.path.clone(), now);
        true
    }

    /// Clean up old entries.
    pub fn cleanup(&mut self) {
        let now = Instant::now();
        let timeout = self.debounce * 10;

        self.last_events
            .retain(|_, &mut last| now.duration_since(last) < timeout);
    }
}

/// Convert FileChange to DaemonEvent.
impl From<FileChange> for DaemonEvent {
    fn from(change: FileChange) -> Self {
        let path = change.path.to_string_lossy().to_string();
        match change.kind {
            ChangeKind::Created => DaemonEvent::FileCreated { path },
            ChangeKind::Modified => DaemonEvent::FileModified { path },
            ChangeKind::Deleted => DaemonEvent::FileDeleted { path },
        }
    }
}
