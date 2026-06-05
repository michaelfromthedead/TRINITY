//! Session Persistence Foundation (T-TL-7.1)
//!
//! Provides editor state serialization and crash recovery for the TRINITY editor.
//! Uses SQLite for snapshot storage with automatic versioning and journal-based
//! crash recovery.
//!
//! # Architecture
//!
//! ```text
//! EditorSnapshot ─────┬────► Foundation (SQLite)
//!                     │          │
//!                     │          ├── save_snapshot()
//!                     │          ├── load_latest()
//!                     │          ├── list_snapshots()
//!                     │          └── delete_snapshot()
//!                     │
//!                     └────► CrashRecovery (Journal)
//!                                │
//!                                ├── record_change()
//!                                ├── has_pending_changes()
//!                                ├── recover()
//!                                └── clear_journal()
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use renderer_backend::foundation::{Foundation, EditorSnapshot, CrashRecovery};
//! use std::path::Path;
//! use std::time::Duration;
//!
//! // Initialize foundation database
//! let mut foundation = Foundation::new(Path::new("editor.db"))?;
//!
//! // Save editor state
//! let snapshot = EditorSnapshot::default();
//! let snapshot_id = foundation.save_snapshot(&snapshot)?;
//!
//! // Load latest on restart
//! if let Some(restored) = foundation.load_latest()? {
//!     println!("Restored {} panels", restored.panels.len());
//! }
//!
//! // List all saved snapshots
//! for info in foundation.list_snapshots()? {
//!     println!("Snapshot {} from {}", info.id, info.timestamp);
//! }
//! ```

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use glam::Vec3;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

use crate::editor_camera::CameraPose;

// ---------------------------------------------------------------------------
// Error Types
// ---------------------------------------------------------------------------

/// Errors that can occur during foundation operations.
#[derive(Debug)]
pub enum FoundationError {
    /// Database operation failed.
    Database(rusqlite::Error),
    /// Serialization/deserialization failed.
    Serialization(String),
    /// IO operation failed.
    Io(io::Error),
    /// Snapshot not found.
    NotFound(u64),
    /// Journal is corrupted.
    CorruptedJournal(String),
    /// Schema version mismatch.
    SchemaMismatch { expected: u32, found: u32 },
}

impl std::fmt::Display for FoundationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Database(e) => write!(f, "database error: {}", e),
            Self::Serialization(msg) => write!(f, "serialization error: {}", msg),
            Self::Io(e) => write!(f, "IO error: {}", e),
            Self::NotFound(id) => write!(f, "snapshot not found: {}", id),
            Self::CorruptedJournal(msg) => write!(f, "corrupted journal: {}", msg),
            Self::SchemaMismatch { expected, found } => {
                write!(f, "schema version mismatch: expected {}, found {}", expected, found)
            }
        }
    }
}

impl std::error::Error for FoundationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Database(e) => Some(e),
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<rusqlite::Error> for FoundationError {
    fn from(e: rusqlite::Error) -> Self {
        Self::Database(e)
    }
}

impl From<io::Error> for FoundationError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for FoundationError {
    fn from(e: serde_json::Error) -> Self {
        Self::Serialization(e.to_string())
    }
}

// ---------------------------------------------------------------------------
// Selection Snapshot
// ---------------------------------------------------------------------------

/// Serializable snapshot of selection state.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SelectionSnapshot {
    /// Primary selected entity ID.
    pub selected_entity_id: Option<u64>,
    /// All selected entity IDs (for multi-selection).
    pub multi_selection: Vec<u64>,
    /// Current gizmo mode (0=Translate, 1=Rotate, 2=Scale, 3=None).
    pub gizmo_mode: u8,
    /// Gizmo space (0=Local, 1=World).
    pub gizmo_space: u8,
    /// Position snap value.
    pub snap_position: f32,
    /// Rotation snap value in degrees.
    pub snap_rotation: f32,
    /// Scale snap value.
    pub snap_scale: f32,
    /// Whether snapping is enabled.
    pub snap_enabled: bool,
}

// ---------------------------------------------------------------------------
// Dock Position
// ---------------------------------------------------------------------------

/// Panel docking position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DockPosition {
    /// Docked to the left side.
    Left,
    /// Docked to the right side.
    Right,
    /// Docked to the top.
    Top,
    /// Docked to the bottom.
    Bottom,
    /// Docked in the center (main area).
    Center,
    /// Floating (not docked).
    Floating,
}

impl Default for DockPosition {
    fn default() -> Self {
        Self::Floating
    }
}

// ---------------------------------------------------------------------------
// Panel Snapshot
// ---------------------------------------------------------------------------

/// Serializable snapshot of a single panel's state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PanelSnapshot {
    /// Unique panel identifier.
    pub id: String,
    /// Whether the panel is currently visible.
    pub visible: bool,
    /// Panel position (x, y) in screen coordinates.
    pub position: (f32, f32),
    /// Panel size (width, height).
    pub size: (f32, f32),
    /// Docking position, if docked.
    pub docked: Option<DockPosition>,
    /// Panel-specific serialized state.
    pub state: Vec<u8>,
}

impl Default for PanelSnapshot {
    fn default() -> Self {
        Self {
            id: String::new(),
            visible: true,
            position: (0.0, 0.0),
            size: (300.0, 400.0),
            docked: None,
            state: Vec::new(),
        }
    }
}

impl PanelSnapshot {
    /// Create a new panel snapshot with the given ID.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            ..Default::default()
        }
    }

    /// Set the panel position.
    pub fn with_position(mut self, x: f32, y: f32) -> Self {
        self.position = (x, y);
        self
    }

    /// Set the panel size.
    pub fn with_size(mut self, width: f32, height: f32) -> Self {
        self.size = (width, height);
        self
    }

    /// Set the docking position.
    pub fn with_dock(mut self, dock: DockPosition) -> Self {
        self.docked = Some(dock);
        self
    }

    /// Set the panel visibility.
    pub fn with_visible(mut self, visible: bool) -> Self {
        self.visible = visible;
        self
    }

    /// Set the panel-specific state.
    pub fn with_state(mut self, state: Vec<u8>) -> Self {
        self.state = state;
        self
    }

    /// Check if this panel is docked.
    pub fn is_docked(&self) -> bool {
        self.docked.is_some()
    }

    /// Get the panel area (width * height).
    pub fn area(&self) -> f32 {
        self.size.0 * self.size.1
    }
}

// ---------------------------------------------------------------------------
// Editor Snapshot
// ---------------------------------------------------------------------------

/// Current schema version for editor snapshots.
pub const SNAPSHOT_SCHEMA_VERSION: u32 = 1;

/// Complete snapshot of editor state for session persistence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EditorSnapshot {
    /// Schema version for forward compatibility.
    pub version: u32,
    /// Unix timestamp when the snapshot was created.
    pub timestamp: u64,
    /// Selection state snapshot.
    pub selection: SelectionSnapshot,
    /// All panel snapshots.
    pub panels: Vec<PanelSnapshot>,
    /// Camera pose.
    pub camera: CameraPose,
    /// REPL history (most recent last).
    pub repl_history: Vec<String>,
    /// Expanded entity IDs in the hierarchy panel.
    pub expanded_entities: Vec<u64>,
    /// Custom data for extensibility (plugin state, etc.).
    pub custom_data: HashMap<String, Vec<u8>>,
}

impl Default for EditorSnapshot {
    fn default() -> Self {
        Self {
            version: SNAPSHOT_SCHEMA_VERSION,
            timestamp: current_timestamp(),
            selection: SelectionSnapshot::default(),
            panels: Vec::new(),
            camera: CameraPose::default(),
            repl_history: Vec::new(),
            expanded_entities: Vec::new(),
            custom_data: HashMap::new(),
        }
    }
}

impl EditorSnapshot {
    /// Create a new snapshot with the current timestamp.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a snapshot with a specific timestamp.
    pub fn with_timestamp(timestamp: u64) -> Self {
        Self {
            timestamp,
            ..Default::default()
        }
    }

    /// Add a panel snapshot.
    pub fn add_panel(&mut self, panel: PanelSnapshot) {
        self.panels.push(panel);
    }

    /// Get a panel by ID.
    pub fn get_panel(&self, id: &str) -> Option<&PanelSnapshot> {
        self.panels.iter().find(|p| p.id == id)
    }

    /// Get a mutable panel by ID.
    pub fn get_panel_mut(&mut self, id: &str) -> Option<&mut PanelSnapshot> {
        self.panels.iter_mut().find(|p| p.id == id)
    }

    /// Remove a panel by ID.
    pub fn remove_panel(&mut self, id: &str) -> Option<PanelSnapshot> {
        if let Some(pos) = self.panels.iter().position(|p| p.id == id) {
            Some(self.panels.remove(pos))
        } else {
            None
        }
    }

    /// Add REPL history entry.
    pub fn add_repl_history(&mut self, entry: String) {
        self.repl_history.push(entry);
    }

    /// Set custom data.
    pub fn set_custom_data(&mut self, key: impl Into<String>, data: Vec<u8>) {
        self.custom_data.insert(key.into(), data);
    }

    /// Get custom data.
    pub fn get_custom_data(&self, key: &str) -> Option<&[u8]> {
        self.custom_data.get(key).map(|v| v.as_slice())
    }

    /// Toggle entity expansion in hierarchy.
    pub fn toggle_expanded(&mut self, entity_id: u64) -> bool {
        if let Some(pos) = self.expanded_entities.iter().position(|&id| id == entity_id) {
            self.expanded_entities.remove(pos);
            false
        } else {
            self.expanded_entities.push(entity_id);
            true
        }
    }

    /// Check if an entity is expanded.
    pub fn is_expanded(&self, entity_id: u64) -> bool {
        self.expanded_entities.contains(&entity_id)
    }

    /// Serialize to JSON bytes.
    pub fn to_json(&self) -> Result<Vec<u8>, FoundationError> {
        serde_json::to_vec(self).map_err(FoundationError::from)
    }

    /// Deserialize from JSON bytes.
    pub fn from_json(data: &[u8]) -> Result<Self, FoundationError> {
        serde_json::from_slice(data).map_err(FoundationError::from)
    }

    /// Estimate the serialized size in bytes.
    pub fn estimated_size(&self) -> usize {
        // Base overhead
        let mut size = 256;
        // Panels
        size += self.panels.iter().map(|p| p.id.len() + p.state.len() + 64).sum::<usize>();
        // REPL history
        size += self.repl_history.iter().map(|s| s.len()).sum::<usize>();
        // Expanded entities
        size += self.expanded_entities.len() * 8;
        // Custom data
        size += self.custom_data.iter().map(|(k, v)| k.len() + v.len()).sum::<usize>();
        size
    }
}

// ---------------------------------------------------------------------------
// Snapshot Info
// ---------------------------------------------------------------------------

/// Lightweight metadata about a saved snapshot (for listing).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotInfo {
    /// Unique snapshot ID.
    pub id: u64,
    /// Unix timestamp when created.
    pub timestamp: u64,
    /// Schema version.
    pub version: u32,
    /// Serialized size in bytes.
    pub size_bytes: usize,
    /// Number of panels in the snapshot.
    pub panel_count: usize,
    /// Optional description/name.
    pub description: Option<String>,
    /// Whether this is an auto-save or manual save.
    pub is_auto_save: bool,
}

impl SnapshotInfo {
    /// Format the timestamp as a human-readable string.
    pub fn formatted_time(&self) -> String {
        // Simple ISO-8601 style formatting
        let secs = self.timestamp;
        format!("{}", secs)
    }
}

// ---------------------------------------------------------------------------
// Foundation Database
// ---------------------------------------------------------------------------

/// Current database schema version.
const DB_SCHEMA_VERSION: u32 = 1;

/// SQLite-backed foundation for editor session persistence.
///
/// Stores editor snapshots with automatic versioning and provides
/// efficient queries for listing and loading snapshots.
pub struct Foundation {
    /// SQLite connection.
    db: Connection,
    /// Auto-save interval.
    auto_save_interval: Duration,
    /// Last save timestamp.
    last_save: Instant,
    /// Maximum number of auto-saves to keep.
    max_auto_saves: usize,
    /// Database path (for debugging).
    path: PathBuf,
}

impl Foundation {
    /// Create a new foundation database at the given path.
    ///
    /// Creates the database file and schema if they don't exist.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the SQLite database file.
    ///
    /// # Returns
    ///
    /// A new `Foundation` instance or an error if initialization fails.
    pub fn new(path: &Path) -> Result<Self, FoundationError> {
        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let db = Connection::open(path)?;
        let mut foundation = Self {
            db,
            auto_save_interval: Duration::from_secs(300), // 5 minutes default
            last_save: Instant::now(),
            max_auto_saves: 10,
            path: path.to_path_buf(),
        };

        foundation.init_schema()?;
        Ok(foundation)
    }

    /// Create an in-memory foundation (for testing).
    pub fn in_memory() -> Result<Self, FoundationError> {
        let db = Connection::open_in_memory()?;
        let mut foundation = Self {
            db,
            auto_save_interval: Duration::from_secs(300),
            last_save: Instant::now(),
            max_auto_saves: 10,
            path: PathBuf::from(":memory:"),
        };

        foundation.init_schema()?;
        Ok(foundation)
    }

    /// Initialize the database schema.
    fn init_schema(&mut self) -> Result<(), FoundationError> {
        self.db.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                version INTEGER NOT NULL,
                size_bytes INTEGER NOT NULL,
                panel_count INTEGER NOT NULL,
                description TEXT,
                is_auto_save INTEGER NOT NULL DEFAULT 0,
                data BLOB NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON snapshots(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_snapshots_auto_save ON snapshots(is_auto_save, timestamp DESC);
            "#,
        )?;

        // Check/set schema version
        let existing_version: Option<u32> = self
            .db
            .query_row("SELECT version FROM schema_version LIMIT 1", [], |row| row.get(0))
            .optional()?;

        match existing_version {
            Some(v) if v != DB_SCHEMA_VERSION => {
                return Err(FoundationError::SchemaMismatch {
                    expected: DB_SCHEMA_VERSION,
                    found: v,
                });
            }
            None => {
                self.db.execute(
                    "INSERT INTO schema_version (version) VALUES (?)",
                    params![DB_SCHEMA_VERSION],
                )?;
            }
            _ => {}
        }

        Ok(())
    }

    /// Set the auto-save interval.
    pub fn set_auto_save_interval(&mut self, interval: Duration) {
        self.auto_save_interval = interval;
    }

    /// Get the auto-save interval.
    pub fn auto_save_interval(&self) -> Duration {
        self.auto_save_interval
    }

    /// Set the maximum number of auto-saves to keep.
    pub fn set_max_auto_saves(&mut self, max: usize) {
        self.max_auto_saves = max;
    }

    /// Save an editor snapshot to the database.
    ///
    /// # Arguments
    ///
    /// * `snapshot` - The editor snapshot to save.
    ///
    /// # Returns
    ///
    /// The unique ID of the saved snapshot.
    pub fn save_snapshot(&mut self, snapshot: &EditorSnapshot) -> Result<u64, FoundationError> {
        self.save_snapshot_with_options(snapshot, None, false)
    }

    /// Save a snapshot with optional description and auto-save flag.
    pub fn save_snapshot_with_options(
        &mut self,
        snapshot: &EditorSnapshot,
        description: Option<&str>,
        is_auto_save: bool,
    ) -> Result<u64, FoundationError> {
        let data = snapshot.to_json()?;
        let size_bytes = data.len();
        let panel_count = snapshot.panels.len();

        self.db.execute(
            "INSERT INTO snapshots (timestamp, version, size_bytes, panel_count, description, is_auto_save, data)
             VALUES (?, ?, ?, ?, ?, ?, ?)",
            params![
                snapshot.timestamp as i64,
                snapshot.version,
                size_bytes,
                panel_count,
                description,
                is_auto_save as i32,
                data
            ],
        )?;

        let id = self.db.last_insert_rowid() as u64;

        // Clean up old auto-saves if needed
        if is_auto_save {
            self.cleanup_auto_saves()?;
        }

        self.last_save = Instant::now();
        Ok(id)
    }

    /// Load the most recent snapshot.
    pub fn load_latest(&self) -> Result<Option<EditorSnapshot>, FoundationError> {
        let result: Option<Vec<u8>> = self
            .db
            .query_row(
                "SELECT data FROM snapshots ORDER BY timestamp DESC LIMIT 1",
                [],
                |row| row.get(0),
            )
            .optional()?;

        match result {
            Some(data) => Ok(Some(EditorSnapshot::from_json(&data)?)),
            None => Ok(None),
        }
    }

    /// Load a specific snapshot by ID.
    pub fn load_snapshot(&self, id: u64) -> Result<EditorSnapshot, FoundationError> {
        let data: Vec<u8> = self
            .db
            .query_row("SELECT data FROM snapshots WHERE id = ?", params![id as i64], |row| {
                row.get(0)
            })
            .optional()?
            .ok_or(FoundationError::NotFound(id))?;

        EditorSnapshot::from_json(&data)
    }

    /// List all saved snapshots (metadata only).
    pub fn list_snapshots(&self) -> Result<Vec<SnapshotInfo>, FoundationError> {
        let mut stmt = self.db.prepare(
            "SELECT id, timestamp, version, size_bytes, panel_count, description, is_auto_save
             FROM snapshots ORDER BY timestamp DESC",
        )?;

        let rows = stmt.query_map([], |row| {
            Ok(SnapshotInfo {
                id: row.get::<_, i64>(0)? as u64,
                timestamp: row.get::<_, i64>(1)? as u64,
                version: row.get::<_, u32>(2)?,
                size_bytes: row.get::<_, usize>(3)?,
                panel_count: row.get::<_, usize>(4)?,
                description: row.get(5)?,
                is_auto_save: row.get::<_, i32>(6)? != 0,
            })
        })?;

        let mut snapshots = Vec::new();
        for row in rows {
            snapshots.push(row?);
        }

        Ok(snapshots)
    }

    /// Delete a snapshot by ID.
    pub fn delete_snapshot(&mut self, id: u64) -> Result<(), FoundationError> {
        let deleted = self
            .db
            .execute("DELETE FROM snapshots WHERE id = ?", params![id as i64])?;

        if deleted == 0 {
            return Err(FoundationError::NotFound(id));
        }

        Ok(())
    }

    /// Check if an auto-save should be performed.
    pub fn should_auto_save(&self) -> bool {
        self.last_save.elapsed() >= self.auto_save_interval
    }

    /// Mark that a save just occurred (resets auto-save timer).
    pub fn mark_saved(&mut self) {
        self.last_save = Instant::now();
    }

    /// Get the time since last save.
    pub fn time_since_save(&self) -> Duration {
        self.last_save.elapsed()
    }

    /// Get the number of saved snapshots.
    pub fn snapshot_count(&self) -> Result<usize, FoundationError> {
        let count: i64 = self
            .db
            .query_row("SELECT COUNT(*) FROM snapshots", [], |row| row.get(0))?;
        Ok(count as usize)
    }

    /// Get total database size estimate.
    pub fn total_size(&self) -> Result<usize, FoundationError> {
        let size: i64 = self
            .db
            .query_row("SELECT COALESCE(SUM(size_bytes), 0) FROM snapshots", [], |row| {
                row.get(0)
            })?;
        Ok(size as usize)
    }

    /// Clean up old auto-saves, keeping only the most recent `max_auto_saves`.
    fn cleanup_auto_saves(&mut self) -> Result<usize, FoundationError> {
        // Get IDs of auto-saves to delete
        let mut stmt = self.db.prepare(
            "SELECT id FROM snapshots WHERE is_auto_save = 1
             ORDER BY timestamp DESC
             LIMIT -1 OFFSET ?",
        )?;

        let ids_to_delete: Vec<i64> = stmt
            .query_map(params![self.max_auto_saves], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();

        let count = ids_to_delete.len();

        for id in ids_to_delete {
            self.db.execute("DELETE FROM snapshots WHERE id = ?", params![id])?;
        }

        Ok(count)
    }

    /// Delete all snapshots.
    pub fn clear_all(&mut self) -> Result<usize, FoundationError> {
        let count = self.db.execute("DELETE FROM snapshots", [])?;
        Ok(count)
    }

    /// Get database path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Vacuum the database to reclaim space.
    pub fn vacuum(&mut self) -> Result<(), FoundationError> {
        self.db.execute_batch("VACUUM")?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Change Types for Crash Recovery
// ---------------------------------------------------------------------------

/// Types of changes that can be recorded in the crash recovery journal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ChangeType {
    /// Entity selection changed.
    Selection,
    /// Camera moved.
    Camera,
    /// Panel layout changed.
    PanelLayout,
    /// Component value modified.
    ComponentEdit { entity_id: u64, component: String },
    /// Entity created.
    EntityCreated(u64),
    /// Entity deleted.
    EntityDeleted(u64),
    /// Entity hierarchy changed.
    Hierarchy,
    /// REPL command executed.
    ReplCommand,
    /// Custom change type.
    Custom(String),
}

/// A single change recorded in the crash recovery journal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Change {
    /// Sequence number (monotonically increasing).
    pub sequence: u64,
    /// Unix timestamp when the change occurred.
    pub timestamp: u64,
    /// Type of change.
    pub change_type: ChangeType,
    /// Serialized change data.
    pub data: Vec<u8>,
    /// Optional description.
    pub description: Option<String>,
}

impl Change {
    /// Create a new change with the given type and data.
    pub fn new(change_type: ChangeType, data: Vec<u8>) -> Self {
        Self {
            sequence: 0, // Set by CrashRecovery
            timestamp: current_timestamp(),
            change_type,
            data,
            description: None,
        }
    }

    /// Create a change with a description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Serialize to a journal line (newline-delimited JSON).
    pub fn to_journal_line(&self) -> Result<String, FoundationError> {
        serde_json::to_string(self).map_err(FoundationError::from)
    }

    /// Deserialize from a journal line.
    pub fn from_journal_line(line: &str) -> Result<Self, FoundationError> {
        serde_json::from_str(line).map_err(FoundationError::from)
    }
}

// ---------------------------------------------------------------------------
// Crash Recovery
// ---------------------------------------------------------------------------

/// Journal-based crash recovery for editor changes.
///
/// Records incremental changes to a write-ahead journal file that can be
/// replayed to recover state after a crash. The journal is cleared after
/// each successful snapshot save.
pub struct CrashRecovery {
    /// Path to the journal file.
    journal_path: PathBuf,
    /// Pending changes not yet saved to a snapshot.
    pending_changes: Vec<Change>,
    /// Next sequence number.
    next_sequence: u64,
    /// Maximum pending changes before forcing a flush.
    max_pending: usize,
}

impl CrashRecovery {
    /// Create a new crash recovery instance.
    ///
    /// # Arguments
    ///
    /// * `journal_path` - Path to the journal file.
    pub fn new(journal_path: impl Into<PathBuf>) -> Self {
        Self {
            journal_path: journal_path.into(),
            pending_changes: Vec::new(),
            next_sequence: 0,
            max_pending: 1000,
        }
    }

    /// Initialize crash recovery, loading any existing journal.
    pub fn init(&mut self) -> Result<(), FoundationError> {
        if self.journal_path.exists() {
            // Load existing journal to get sequence number
            let changes = self.read_journal()?;
            if let Some(last) = changes.last() {
                self.next_sequence = last.sequence + 1;
            }
            self.pending_changes = changes;
        }
        Ok(())
    }

    /// Record a change to the journal.
    ///
    /// The change is appended to the journal file immediately for durability.
    pub fn record_change(&mut self, mut change: Change) -> Result<(), FoundationError> {
        change.sequence = self.next_sequence;
        self.next_sequence += 1;

        // Append to journal file
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.journal_path)?;

        let mut writer = BufWriter::new(file);
        let line = change.to_journal_line()?;
        writeln!(writer, "{}", line)?;
        writer.flush()?;

        self.pending_changes.push(change);

        // Force flush if we have too many pending changes
        if self.pending_changes.len() > self.max_pending {
            self.flush_to_disk()?;
        }

        Ok(())
    }

    /// Check if there are pending changes.
    pub fn has_pending_changes(&self) -> bool {
        !self.pending_changes.is_empty()
    }

    /// Get the number of pending changes.
    pub fn pending_count(&self) -> usize {
        self.pending_changes.len()
    }

    /// Get pending changes (for inspection).
    pub fn pending_changes(&self) -> &[Change] {
        &self.pending_changes
    }

    /// Recover changes from the journal file.
    ///
    /// Call this on startup to recover from a crash.
    pub fn recover(&self) -> Result<Vec<Change>, FoundationError> {
        self.read_journal()
    }

    /// Read the journal file.
    fn read_journal(&self) -> Result<Vec<Change>, FoundationError> {
        if !self.journal_path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(&self.journal_path)?;
        let reader = BufReader::new(file);

        let mut changes = Vec::new();
        let mut line_num = 0;

        for line_result in reader.lines() {
            line_num += 1;
            let line = line_result?;

            if line.trim().is_empty() {
                continue;
            }

            match Change::from_journal_line(&line) {
                Ok(change) => changes.push(change),
                Err(e) => {
                    return Err(FoundationError::CorruptedJournal(format!(
                        "line {}: {}",
                        line_num, e
                    )));
                }
            }
        }

        Ok(changes)
    }

    /// Clear the journal after a successful save.
    pub fn clear_journal(&mut self) -> Result<(), FoundationError> {
        self.pending_changes.clear();
        self.next_sequence = 0;

        if self.journal_path.exists() {
            fs::remove_file(&self.journal_path)?;
        }

        Ok(())
    }

    /// Flush pending changes to disk.
    fn flush_to_disk(&mut self) -> Result<(), FoundationError> {
        // Journal is already appended per-change, nothing extra needed here
        // This is a placeholder for potential batching optimization
        Ok(())
    }

    /// Get the journal file path.
    pub fn journal_path(&self) -> &Path {
        &self.journal_path
    }

    /// Check if the journal file exists.
    pub fn journal_exists(&self) -> bool {
        self.journal_path.exists()
    }

    /// Get the journal file size.
    pub fn journal_size(&self) -> Result<u64, FoundationError> {
        if self.journal_path.exists() {
            let metadata = fs::metadata(&self.journal_path)?;
            Ok(metadata.len())
        } else {
            Ok(0)
        }
    }

    /// Set maximum pending changes before auto-flush.
    pub fn set_max_pending(&mut self, max: usize) {
        self.max_pending = max;
    }

    /// Get changes of a specific type.
    pub fn changes_of_type(&self, change_type: &ChangeType) -> Vec<&Change> {
        self.pending_changes
            .iter()
            .filter(|c| std::mem::discriminant(&c.change_type) == std::mem::discriminant(change_type))
            .collect()
    }

    /// Get the most recent change.
    pub fn latest_change(&self) -> Option<&Change> {
        self.pending_changes.last()
    }

    /// Compact the journal by removing redundant changes.
    ///
    /// For example, multiple selection changes can be collapsed to just the latest.
    pub fn compact(&mut self) -> Result<usize, FoundationError> {
        if self.pending_changes.is_empty() {
            return Ok(0);
        }

        let original_count = self.pending_changes.len();

        // Keep only the latest of certain change types
        let mut seen_types = std::collections::HashSet::new();
        let mut compacted = Vec::new();

        for change in self.pending_changes.iter().rev() {
            let type_key = match &change.change_type {
                ChangeType::Selection => "selection".to_string(),
                ChangeType::Camera => "camera".to_string(),
                ChangeType::PanelLayout => "panel_layout".to_string(),
                // Keep all component edits, entity operations
                _ => format!("{:?}_{}", change.change_type, change.sequence),
            };

            if !seen_types.contains(&type_key) {
                seen_types.insert(type_key);
                compacted.push(change.clone());
            }
        }

        compacted.reverse();
        let removed = original_count - compacted.len();

        if removed > 0 {
            self.pending_changes = compacted;

            // Rewrite journal file
            if self.journal_path.exists() {
                let file = File::create(&self.journal_path)?;
                let mut writer = BufWriter::new(file);

                for change in &self.pending_changes {
                    let line = change.to_journal_line()?;
                    writeln!(writer, "{}", line)?;
                }

                writer.flush()?;
            }
        }

        Ok(removed)
    }
}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Returns the current Unix timestamp in seconds.
pub fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // --- Helper functions ---

    fn create_test_foundation() -> (Foundation, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let foundation = Foundation::new(&db_path).unwrap();
        (foundation, temp_dir)
    }

    fn create_test_snapshot() -> EditorSnapshot {
        let mut snapshot = EditorSnapshot::new();
        snapshot.selection.selected_entity_id = Some(42);
        snapshot.selection.multi_selection = vec![42, 43, 44];
        snapshot.add_panel(PanelSnapshot::new("hierarchy").with_dock(DockPosition::Left));
        snapshot.add_panel(PanelSnapshot::new("inspector").with_dock(DockPosition::Right));
        snapshot.repl_history = vec!["print(x)".to_string(), "help()".to_string()];
        snapshot.expanded_entities = vec![1, 2, 3];
        snapshot.set_custom_data("plugin_state", vec![1, 2, 3, 4]);
        snapshot
    }

    // --- EditorSnapshot Tests ---

    #[test]
    fn test_snapshot_default() {
        let snapshot = EditorSnapshot::default();
        assert_eq!(snapshot.version, SNAPSHOT_SCHEMA_VERSION);
        assert!(snapshot.panels.is_empty());
        assert!(snapshot.repl_history.is_empty());
        assert!(snapshot.expanded_entities.is_empty());
        assert!(snapshot.custom_data.is_empty());
    }

    #[test]
    fn test_snapshot_add_panel() {
        let mut snapshot = EditorSnapshot::new();
        snapshot.add_panel(PanelSnapshot::new("test"));
        assert_eq!(snapshot.panels.len(), 1);
        assert_eq!(snapshot.panels[0].id, "test");
    }

    #[test]
    fn test_snapshot_get_panel() {
        let mut snapshot = EditorSnapshot::new();
        snapshot.add_panel(PanelSnapshot::new("panel1"));
        snapshot.add_panel(PanelSnapshot::new("panel2"));

        assert!(snapshot.get_panel("panel1").is_some());
        assert!(snapshot.get_panel("panel2").is_some());
        assert!(snapshot.get_panel("panel3").is_none());
    }

    #[test]
    fn test_snapshot_remove_panel() {
        let mut snapshot = EditorSnapshot::new();
        snapshot.add_panel(PanelSnapshot::new("panel1"));
        snapshot.add_panel(PanelSnapshot::new("panel2"));

        let removed = snapshot.remove_panel("panel1");
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().id, "panel1");
        assert_eq!(snapshot.panels.len(), 1);
        assert!(snapshot.remove_panel("nonexistent").is_none());
    }

    #[test]
    fn test_snapshot_toggle_expanded() {
        let mut snapshot = EditorSnapshot::new();

        // First toggle adds entity
        assert!(snapshot.toggle_expanded(42));
        assert!(snapshot.is_expanded(42));

        // Second toggle removes entity
        assert!(!snapshot.toggle_expanded(42));
        assert!(!snapshot.is_expanded(42));
    }

    #[test]
    fn test_snapshot_custom_data() {
        let mut snapshot = EditorSnapshot::new();
        snapshot.set_custom_data("key1", vec![1, 2, 3]);

        assert_eq!(snapshot.get_custom_data("key1"), Some(&[1, 2, 3][..]));
        assert_eq!(snapshot.get_custom_data("key2"), None);
    }

    #[test]
    fn test_snapshot_json_roundtrip() {
        let snapshot = create_test_snapshot();
        let json = snapshot.to_json().unwrap();
        let restored = EditorSnapshot::from_json(&json).unwrap();

        assert_eq!(snapshot.version, restored.version);
        assert_eq!(snapshot.selection, restored.selection);
        assert_eq!(snapshot.panels.len(), restored.panels.len());
        assert_eq!(snapshot.repl_history, restored.repl_history);
        assert_eq!(snapshot.expanded_entities, restored.expanded_entities);
    }

    #[test]
    fn test_snapshot_estimated_size() {
        let snapshot = create_test_snapshot();
        let size = snapshot.estimated_size();
        assert!(size > 0);

        // Empty snapshot should have smaller estimate
        let empty = EditorSnapshot::default();
        assert!(empty.estimated_size() < size);
    }

    // --- PanelSnapshot Tests ---

    #[test]
    fn test_panel_snapshot_builder() {
        let panel = PanelSnapshot::new("test")
            .with_position(100.0, 200.0)
            .with_size(300.0, 400.0)
            .with_dock(DockPosition::Left)
            .with_visible(false)
            .with_state(vec![1, 2, 3]);

        assert_eq!(panel.id, "test");
        assert_eq!(panel.position, (100.0, 200.0));
        assert_eq!(panel.size, (300.0, 400.0));
        assert_eq!(panel.docked, Some(DockPosition::Left));
        assert!(!panel.visible);
        assert_eq!(panel.state, vec![1, 2, 3]);
    }

    #[test]
    fn test_panel_is_docked() {
        let docked = PanelSnapshot::new("test").with_dock(DockPosition::Left);
        let floating = PanelSnapshot::new("test");

        assert!(docked.is_docked());
        assert!(!floating.is_docked());
    }

    #[test]
    fn test_panel_area() {
        let panel = PanelSnapshot::new("test").with_size(100.0, 200.0);
        assert!((panel.area() - 20000.0).abs() < 0.001);
    }

    // --- Foundation Database Tests ---

    #[test]
    fn test_foundation_new() {
        let (foundation, _temp) = create_test_foundation();
        assert_eq!(foundation.snapshot_count().unwrap(), 0);
    }

    #[test]
    fn test_foundation_in_memory() {
        let foundation = Foundation::in_memory().unwrap();
        assert_eq!(foundation.snapshot_count().unwrap(), 0);
    }

    #[test]
    fn test_foundation_save_and_load_snapshot() {
        let (mut foundation, _temp) = create_test_foundation();
        let snapshot = create_test_snapshot();

        let id = foundation.save_snapshot(&snapshot).unwrap();
        assert!(id > 0);

        let loaded = foundation.load_snapshot(id).unwrap();
        assert_eq!(loaded.selection, snapshot.selection);
        assert_eq!(loaded.panels.len(), snapshot.panels.len());
    }

    #[test]
    fn test_foundation_load_latest() {
        let (mut foundation, _temp) = create_test_foundation();

        // No snapshots yet
        assert!(foundation.load_latest().unwrap().is_none());

        // Save first snapshot
        let mut snapshot1 = EditorSnapshot::with_timestamp(1000);
        snapshot1.selection.selected_entity_id = Some(1);
        foundation.save_snapshot(&snapshot1).unwrap();

        // Save second snapshot
        let mut snapshot2 = EditorSnapshot::with_timestamp(2000);
        snapshot2.selection.selected_entity_id = Some(2);
        foundation.save_snapshot(&snapshot2).unwrap();

        // Load latest should return snapshot2
        let latest = foundation.load_latest().unwrap().unwrap();
        assert_eq!(latest.selection.selected_entity_id, Some(2));
    }

    #[test]
    fn test_foundation_list_snapshots() {
        let (mut foundation, _temp) = create_test_foundation();

        // Save multiple snapshots
        for i in 0..5 {
            let mut snapshot = EditorSnapshot::with_timestamp(1000 + i);
            snapshot.add_panel(PanelSnapshot::new(format!("panel_{}", i)));
            foundation
                .save_snapshot_with_options(&snapshot, Some(&format!("Snapshot {}", i)), false)
                .unwrap();
        }

        let list = foundation.list_snapshots().unwrap();
        assert_eq!(list.len(), 5);

        // Should be sorted by timestamp descending
        assert!(list[0].timestamp >= list[1].timestamp);
    }

    #[test]
    fn test_foundation_delete_snapshot() {
        let (mut foundation, _temp) = create_test_foundation();

        let id = foundation.save_snapshot(&create_test_snapshot()).unwrap();
        assert_eq!(foundation.snapshot_count().unwrap(), 1);

        foundation.delete_snapshot(id).unwrap();
        assert_eq!(foundation.snapshot_count().unwrap(), 0);

        // Deleting non-existent should error
        assert!(matches!(
            foundation.delete_snapshot(9999),
            Err(FoundationError::NotFound(9999))
        ));
    }

    #[test]
    fn test_foundation_auto_save_timing() {
        let (mut foundation, _temp) = create_test_foundation();
        foundation.set_auto_save_interval(Duration::from_millis(100));

        // Just saved, shouldn't need auto-save
        foundation.mark_saved();
        assert!(!foundation.should_auto_save());

        // After interval passes, should need auto-save
        std::thread::sleep(Duration::from_millis(150));
        assert!(foundation.should_auto_save());
    }

    #[test]
    fn test_foundation_auto_save_cleanup() {
        let (mut foundation, _temp) = create_test_foundation();
        foundation.set_max_auto_saves(3);

        // Save 5 auto-saves
        for i in 0..5 {
            let snapshot = EditorSnapshot::with_timestamp(1000 + i);
            foundation.save_snapshot_with_options(&snapshot, None, true).unwrap();
        }

        // Should have cleaned up to 3
        let list: Vec<_> = foundation
            .list_snapshots()
            .unwrap()
            .into_iter()
            .filter(|s| s.is_auto_save)
            .collect();
        assert_eq!(list.len(), 3);
    }

    #[test]
    fn test_foundation_total_size() {
        let (mut foundation, _temp) = create_test_foundation();

        assert_eq!(foundation.total_size().unwrap(), 0);

        foundation.save_snapshot(&create_test_snapshot()).unwrap();

        let size = foundation.total_size().unwrap();
        assert!(size > 0);
    }

    #[test]
    fn test_foundation_clear_all() {
        let (mut foundation, _temp) = create_test_foundation();

        for _ in 0..5 {
            foundation.save_snapshot(&create_test_snapshot()).unwrap();
        }

        assert_eq!(foundation.snapshot_count().unwrap(), 5);

        let cleared = foundation.clear_all().unwrap();
        assert_eq!(cleared, 5);
        assert_eq!(foundation.snapshot_count().unwrap(), 0);
    }

    #[test]
    fn test_foundation_snapshot_not_found() {
        let (foundation, _temp) = create_test_foundation();

        assert!(matches!(
            foundation.load_snapshot(9999),
            Err(FoundationError::NotFound(9999))
        ));
    }

    // --- CrashRecovery Tests ---

    #[test]
    fn test_crash_recovery_new() {
        let temp_dir = TempDir::new().unwrap();
        let journal_path = temp_dir.path().join("journal.log");
        let recovery = CrashRecovery::new(&journal_path);

        assert!(!recovery.has_pending_changes());
        assert_eq!(recovery.pending_count(), 0);
    }

    #[test]
    fn test_crash_recovery_record_change() {
        let temp_dir = TempDir::new().unwrap();
        let journal_path = temp_dir.path().join("journal.log");
        let mut recovery = CrashRecovery::new(&journal_path);

        let change = Change::new(ChangeType::Selection, vec![1, 2, 3]);
        recovery.record_change(change).unwrap();

        assert!(recovery.has_pending_changes());
        assert_eq!(recovery.pending_count(), 1);
        assert!(recovery.journal_exists());
    }

    #[test]
    fn test_crash_recovery_multiple_changes() {
        let temp_dir = TempDir::new().unwrap();
        let journal_path = temp_dir.path().join("journal.log");
        let mut recovery = CrashRecovery::new(&journal_path);

        recovery.record_change(Change::new(ChangeType::Selection, vec![])).unwrap();
        recovery.record_change(Change::new(ChangeType::Camera, vec![])).unwrap();
        recovery.record_change(Change::new(ChangeType::PanelLayout, vec![])).unwrap();

        assert_eq!(recovery.pending_count(), 3);

        // Check sequence numbers
        let changes = recovery.pending_changes();
        assert_eq!(changes[0].sequence, 0);
        assert_eq!(changes[1].sequence, 1);
        assert_eq!(changes[2].sequence, 2);
    }

    #[test]
    fn test_crash_recovery_recover() {
        let temp_dir = TempDir::new().unwrap();
        let journal_path = temp_dir.path().join("journal.log");

        // Write some changes
        {
            let mut recovery = CrashRecovery::new(&journal_path);
            recovery.record_change(Change::new(ChangeType::Selection, vec![1])).unwrap();
            recovery.record_change(Change::new(ChangeType::Camera, vec![2])).unwrap();
        }

        // Create new instance and recover
        let recovery2 = CrashRecovery::new(&journal_path);
        let recovered = recovery2.recover().unwrap();

        assert_eq!(recovered.len(), 2);
        assert_eq!(recovered[0].data, vec![1]);
        assert_eq!(recovered[1].data, vec![2]);
    }

    #[test]
    fn test_crash_recovery_clear_journal() {
        let temp_dir = TempDir::new().unwrap();
        let journal_path = temp_dir.path().join("journal.log");
        let mut recovery = CrashRecovery::new(&journal_path);

        recovery.record_change(Change::new(ChangeType::Selection, vec![])).unwrap();
        assert!(recovery.journal_exists());

        recovery.clear_journal().unwrap();

        assert!(!recovery.has_pending_changes());
        assert!(!recovery.journal_exists());
    }

    #[test]
    fn test_crash_recovery_init() {
        let temp_dir = TempDir::new().unwrap();
        let journal_path = temp_dir.path().join("journal.log");

        // Write some changes
        {
            let mut recovery = CrashRecovery::new(&journal_path);
            recovery.record_change(Change::new(ChangeType::Selection, vec![])).unwrap();
            recovery.record_change(Change::new(ChangeType::Camera, vec![])).unwrap();
        }

        // Init should load existing journal
        let mut recovery2 = CrashRecovery::new(&journal_path);
        recovery2.init().unwrap();

        assert_eq!(recovery2.pending_count(), 2);
        // Next sequence should continue from where we left off
        recovery2.record_change(Change::new(ChangeType::Selection, vec![])).unwrap();
        assert_eq!(recovery2.pending_changes().last().unwrap().sequence, 2);
    }

    #[test]
    fn test_crash_recovery_journal_size() {
        let temp_dir = TempDir::new().unwrap();
        let journal_path = temp_dir.path().join("journal.log");
        let mut recovery = CrashRecovery::new(&journal_path);

        assert_eq!(recovery.journal_size().unwrap(), 0);

        recovery.record_change(Change::new(ChangeType::Selection, vec![1, 2, 3, 4, 5])).unwrap();

        let size = recovery.journal_size().unwrap();
        assert!(size > 0);
    }

    #[test]
    fn test_crash_recovery_changes_of_type() {
        let temp_dir = TempDir::new().unwrap();
        let journal_path = temp_dir.path().join("journal.log");
        let mut recovery = CrashRecovery::new(&journal_path);

        recovery.record_change(Change::new(ChangeType::Selection, vec![])).unwrap();
        recovery.record_change(Change::new(ChangeType::Camera, vec![])).unwrap();
        recovery.record_change(Change::new(ChangeType::Selection, vec![])).unwrap();

        let selection_changes = recovery.changes_of_type(&ChangeType::Selection);
        assert_eq!(selection_changes.len(), 2);

        let camera_changes = recovery.changes_of_type(&ChangeType::Camera);
        assert_eq!(camera_changes.len(), 1);
    }

    #[test]
    fn test_crash_recovery_compact() {
        let temp_dir = TempDir::new().unwrap();
        let journal_path = temp_dir.path().join("journal.log");
        let mut recovery = CrashRecovery::new(&journal_path);

        // Record multiple selection changes
        recovery.record_change(Change::new(ChangeType::Selection, vec![1])).unwrap();
        recovery.record_change(Change::new(ChangeType::Selection, vec![2])).unwrap();
        recovery.record_change(Change::new(ChangeType::Selection, vec![3])).unwrap();
        recovery.record_change(Change::new(ChangeType::Camera, vec![4])).unwrap();

        let removed = recovery.compact().unwrap();
        assert!(removed > 0);

        // Should keep only latest selection and camera
        let selection_changes = recovery.changes_of_type(&ChangeType::Selection);
        assert_eq!(selection_changes.len(), 1);
        assert_eq!(selection_changes[0].data, vec![3]); // Latest
    }

    #[test]
    fn test_crash_recovery_latest_change() {
        let temp_dir = TempDir::new().unwrap();
        let journal_path = temp_dir.path().join("journal.log");
        let mut recovery = CrashRecovery::new(&journal_path);

        assert!(recovery.latest_change().is_none());

        recovery.record_change(Change::new(ChangeType::Selection, vec![1])).unwrap();
        recovery.record_change(Change::new(ChangeType::Camera, vec![2])).unwrap();

        let latest = recovery.latest_change().unwrap();
        assert_eq!(latest.data, vec![2]);
    }

    // --- Change Tests ---

    #[test]
    fn test_change_new() {
        let change = Change::new(ChangeType::Selection, vec![1, 2, 3]);
        assert_eq!(change.sequence, 0);
        assert_eq!(change.data, vec![1, 2, 3]);
        assert!(change.description.is_none());
    }

    #[test]
    fn test_change_with_description() {
        let change = Change::new(ChangeType::Selection, vec![]).with_description("Test change");
        assert_eq!(change.description, Some("Test change".to_string()));
    }

    #[test]
    fn test_change_journal_roundtrip() {
        let change = Change {
            sequence: 42,
            timestamp: 1234567890,
            change_type: ChangeType::ComponentEdit {
                entity_id: 100,
                component: "Transform".to_string(),
            },
            data: vec![1, 2, 3, 4],
            description: Some("Moved entity".to_string()),
        };

        let line = change.to_journal_line().unwrap();
        let restored = Change::from_journal_line(&line).unwrap();

        assert_eq!(change.sequence, restored.sequence);
        assert_eq!(change.timestamp, restored.timestamp);
        assert_eq!(change.data, restored.data);
        assert_eq!(change.description, restored.description);
    }

    // --- SelectionSnapshot Tests ---

    #[test]
    fn test_selection_snapshot_default() {
        let selection = SelectionSnapshot::default();
        assert!(selection.selected_entity_id.is_none());
        assert!(selection.multi_selection.is_empty());
        assert_eq!(selection.gizmo_mode, 0);
        assert!(!selection.snap_enabled);
    }

    // --- DockPosition Tests ---

    #[test]
    fn test_dock_position_default() {
        let dock = DockPosition::default();
        assert_eq!(dock, DockPosition::Floating);
    }

    // --- SnapshotInfo Tests ---

    #[test]
    fn test_snapshot_info_formatted_time() {
        let info = SnapshotInfo {
            id: 1,
            timestamp: 1234567890,
            version: 1,
            size_bytes: 100,
            panel_count: 2,
            description: None,
            is_auto_save: false,
        };

        let formatted = info.formatted_time();
        assert!(!formatted.is_empty());
    }

    // --- Error Tests ---

    #[test]
    fn test_foundation_error_display() {
        let err = FoundationError::NotFound(42);
        assert!(err.to_string().contains("42"));

        let err = FoundationError::SchemaMismatch { expected: 1, found: 2 };
        assert!(err.to_string().contains("1"));
        assert!(err.to_string().contains("2"));
    }

    // --- Integration Tests ---

    #[test]
    fn test_full_session_workflow() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("session.db");
        let journal_path = temp_dir.path().join("journal.log");

        // Create foundation and recovery
        let mut foundation = Foundation::new(&db_path).unwrap();
        let mut recovery = CrashRecovery::new(&journal_path);

        // Simulate editor session with changes
        recovery.record_change(Change::new(ChangeType::Selection, vec![1])).unwrap();
        recovery.record_change(Change::new(ChangeType::Camera, vec![2])).unwrap();

        // Create snapshot
        let mut snapshot = EditorSnapshot::new();
        snapshot.selection.selected_entity_id = Some(42);
        snapshot.add_panel(PanelSnapshot::new("hierarchy").with_dock(DockPosition::Left));

        // Save snapshot
        let id = foundation.save_snapshot(&snapshot).unwrap();

        // Clear recovery journal after successful save
        recovery.clear_journal().unwrap();

        // Simulate restart
        drop(foundation);
        drop(recovery);

        // Restore session
        let foundation2 = Foundation::new(&db_path).unwrap();
        let mut recovery2 = CrashRecovery::new(&journal_path);

        // No pending recovery changes (journal was cleared)
        recovery2.init().unwrap();
        assert!(!recovery2.has_pending_changes());

        // Load latest snapshot
        let restored = foundation2.load_latest().unwrap().unwrap();
        assert_eq!(restored.selection.selected_entity_id, Some(42));
        assert_eq!(restored.panels.len(), 1);
    }
}
