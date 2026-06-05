//! Content-Level Change Detection for TRINITY Asset Pipeline (T-AS-6.2)
//!
//! This module provides content hash-based change detection for the hot-reload system:
//!
//! - **Content Hash Comparison**: Detects changes even when timestamps match (e.g., git checkout)
//! - **Editor Startup Scan**: Detects changes made while editor was closed
//! - **Periodic Integrity Check**: Optional full hash scan for content verification
//! - **File Watcher Integration**: Hash verification after filesystem events
//! - **Asset Cache Invalidation**: On content hash mismatch
//!
//! # Architecture
//!
//! ```text
//! +----------------+     +------------------------+     +------------------+
//! | FileWatcher    | --> | ContentChangeDetector  | --> | Asset Invalidation|
//! | (FileChange)   |     | (hash verification)    |     | (cache clear)    |
//! +----------------+     +------------------------+     +------------------+
//!                               |
//!                               v
//!                        +------------------------+
//!                        | Hash Cache             |
//!                        | (path -> mtime, hash)  |
//!                        +------------------------+
//! ```
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::hot_reload::content_change::{
//!     ContentChangeDetector, ChangeDetectorConfig
//! };
//! use std::path::PathBuf;
//!
//! let config = ChangeDetectorConfig::default();
//! let mut detector = ContentChangeDetector::new(config);
//!
//! // On editor startup, scan for changes
//! let changes = detector.startup_scan(&[PathBuf::from("assets")]);
//! for change in changes {
//!     println!("Changed: {:?} ({:?})", change.path, change.kind);
//! }
//!
//! // After file watcher event, verify with hash
//! // if let Some(content_change) = detector.verify_file_change(&file_change) {
//! //     invalidate_asset(&content_change.path);
//! // }
//! ```

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime};

use crate::hot_reload::file_watcher::{FileChange, FileChangeKind};
use crate::pipeline::ContentHash;

// ---------------------------------------------------------------------------
// Content Change Types
// ---------------------------------------------------------------------------

/// Classification of content-level changes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentChangeKind {
    /// A new file was added.
    Added,
    /// An existing file was modified (content changed).
    Modified,
    /// A file was deleted.
    Deleted,
    /// Timestamp unchanged but hash differs (e.g., git checkout, restore).
    HashMismatch,
}

impl ContentChangeKind {
    /// Returns true if this change requires content reload.
    pub fn requires_reload(&self) -> bool {
        matches!(
            self,
            Self::Added | Self::Modified | Self::HashMismatch
        )
    }

    /// Returns true if the file no longer exists.
    pub fn is_deletion(&self) -> bool {
        matches!(self, Self::Deleted)
    }
}

/// A content change event detected by the change detector.
#[derive(Debug, Clone)]
pub struct ContentChange {
    /// Path to the affected file.
    pub path: PathBuf,
    /// Type of content change.
    pub kind: ContentChangeKind,
    /// Hash of the content before the change (None for new files).
    pub old_hash: Option<ContentHash>,
    /// Hash of the content after the change (None for deleted files).
    pub new_hash: Option<ContentHash>,
    /// When the change was detected.
    pub detected_at: Instant,
}

impl ContentChange {
    /// Create a new content change event.
    pub fn new(
        path: PathBuf,
        kind: ContentChangeKind,
        old_hash: Option<ContentHash>,
        new_hash: Option<ContentHash>,
    ) -> Self {
        Self {
            path,
            kind,
            old_hash,
            new_hash,
            detected_at: Instant::now(),
        }
    }

    /// Check if this change requires asset invalidation.
    pub fn requires_invalidation(&self) -> bool {
        self.kind.requires_reload() || self.kind.is_deletion()
    }

    /// Get the file extension, if any.
    pub fn extension(&self) -> Option<&str> {
        self.path.extension().and_then(|e| e.to_str())
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Default scan interval for periodic integrity checks (5 minutes).
const DEFAULT_SCAN_INTERVAL_SECS: u64 = 300;

/// Default buffer size for reading files during hashing.
const HASH_BUFFER_SIZE: usize = 64 * 1024; // 64KB

/// Configuration for the content change detector.
#[derive(Debug, Clone)]
pub struct ChangeDetectorConfig {
    /// Enable periodic full integrity scans.
    pub enable_periodic_scan: bool,
    /// Interval between periodic scans in seconds.
    pub scan_interval_secs: u64,
    /// Hash all cached files on startup.
    pub hash_on_startup: bool,
    /// Automatically invalidate cache on hash mismatch.
    pub invalidate_on_mismatch: bool,
    /// Log warnings for deleted files.
    pub warn_on_delete: bool,
    /// Extensions to include in scans (empty means all files).
    pub include_extensions: Vec<String>,
    /// Extensions to exclude from scans.
    pub exclude_extensions: Vec<String>,
}

impl Default for ChangeDetectorConfig {
    fn default() -> Self {
        Self {
            enable_periodic_scan: false,
            scan_interval_secs: DEFAULT_SCAN_INTERVAL_SECS,
            hash_on_startup: true,
            invalidate_on_mismatch: true,
            warn_on_delete: true,
            include_extensions: Vec::new(),
            exclude_extensions: vec![
                "tmp".to_string(),
                "swp".to_string(),
                "lock".to_string(),
            ],
        }
    }
}

impl ChangeDetectorConfig {
    /// Create a new configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable periodic integrity scans.
    pub fn with_periodic_scan(mut self, enabled: bool) -> Self {
        self.enable_periodic_scan = enabled;
        self
    }

    /// Set the scan interval in seconds.
    pub fn with_scan_interval_secs(mut self, secs: u64) -> Self {
        self.scan_interval_secs = secs;
        self
    }

    /// Enable hashing on startup.
    pub fn with_hash_on_startup(mut self, enabled: bool) -> Self {
        self.hash_on_startup = enabled;
        self
    }

    /// Enable automatic invalidation on hash mismatch.
    pub fn with_invalidate_on_mismatch(mut self, enabled: bool) -> Self {
        self.invalidate_on_mismatch = enabled;
        self
    }

    /// Set extensions to include in scans.
    pub fn with_include_extensions(mut self, exts: Vec<String>) -> Self {
        self.include_extensions = exts;
        self
    }

    /// Set extensions to exclude from scans.
    pub fn with_exclude_extensions(mut self, exts: Vec<String>) -> Self {
        self.exclude_extensions = exts;
        self
    }

    /// Enable or disable warnings for deleted files.
    pub fn with_warn_on_delete(mut self, enabled: bool) -> Self {
        self.warn_on_delete = enabled;
        self
    }
}

// ---------------------------------------------------------------------------
// Hash Cache Entry
// ---------------------------------------------------------------------------

/// Cached hash entry with modification time for fast-path comparison.
#[derive(Debug, Clone)]
struct HashCacheEntry {
    /// Last known modification time.
    mtime: SystemTime,
    /// Content hash at that modification time.
    hash: ContentHash,
    /// Size of the file in bytes.
    size: u64,
}

// ---------------------------------------------------------------------------
// Content Change Detector
// ---------------------------------------------------------------------------

/// Statistics for the content change detector.
#[derive(Debug, Clone, Default)]
pub struct ChangeDetectorStats {
    /// Total files scanned.
    pub files_scanned: u64,
    /// Files where hash matched cache.
    pub hash_hits: u64,
    /// Files where hash computation was needed.
    pub hash_computed: u64,
    /// Hash mismatches detected (content changed without mtime change).
    pub hash_mismatches: u64,
    /// Files added since last scan.
    pub files_added: u64,
    /// Files modified since last scan.
    pub files_modified: u64,
    /// Files deleted since last scan.
    pub files_deleted: u64,
    /// Errors encountered during scanning.
    pub errors: u64,
}

/// Content-level change detector with hash caching.
///
/// Maintains a cache of `path -> (mtime, hash)` to enable fast-path detection
/// when modification times change, and full hash verification for detecting
/// changes when timestamps haven't been updated (e.g., git operations).
pub struct ContentChangeDetector {
    /// Hash cache: path -> (mtime, hash, size).
    hash_cache: HashMap<PathBuf, HashCacheEntry>,
    /// Configuration.
    config: ChangeDetectorConfig,
    /// Statistics.
    stats: ChangeDetectorStats,
    /// Last periodic scan time.
    last_periodic_scan: Option<Instant>,
}

impl ContentChangeDetector {
    /// Create a new content change detector with the given configuration.
    pub fn new(config: ChangeDetectorConfig) -> Self {
        Self {
            hash_cache: HashMap::new(),
            config,
            stats: ChangeDetectorStats::default(),
            last_periodic_scan: None,
        }
    }

    /// Scan directories for changes on editor startup.
    ///
    /// Walks all files in the given roots and compares against the cached hashes.
    /// Returns all changes detected (new files, modified files, deleted files).
    pub fn startup_scan(&mut self, roots: &[PathBuf]) -> Vec<ContentChange> {
        let mut changes = Vec::new();
        let mut seen_paths = std::collections::HashSet::new();

        // Scan all files in roots
        for root in roots {
            if root.is_file() {
                // Single file
                if self.should_scan_file(root) {
                    seen_paths.insert(root.clone());
                    if let Some(change) = self.scan_file(root) {
                        changes.push(change);
                    }
                }
            } else if root.is_dir() {
                // Directory - walk recursively
                self.scan_directory(root, &mut seen_paths, &mut changes);
            }
        }

        // Check for deleted files (files in cache but not seen during scan)
        let cached_paths: Vec<PathBuf> = self.hash_cache.keys().cloned().collect();
        for path in cached_paths {
            // Only check deletion if the path was under one of our roots
            let under_root = roots.iter().any(|r| path.starts_with(r));
            if under_root && !seen_paths.contains(&path) {
                // File was in cache but no longer exists
                let old_hash = self.hash_cache.get(&path).map(|e| e.hash);
                self.hash_cache.remove(&path);
                self.stats.files_deleted += 1;

                if self.config.warn_on_delete {
                    // In production, this would use a proper logging framework
                    eprintln!("[WARN] File deleted: {}", path.display());
                }

                changes.push(ContentChange::new(
                    path,
                    ContentChangeKind::Deleted,
                    old_hash,
                    None,
                ));
            }
        }

        changes
    }

    /// Scan a directory recursively for changes.
    fn scan_directory(
        &mut self,
        dir: &Path,
        seen_paths: &mut std::collections::HashSet<PathBuf>,
        changes: &mut Vec<ContentChange>,
    ) {
        let entries = match fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => {
                self.stats.errors += 1;
                return;
            }
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                self.scan_directory(&path, seen_paths, changes);
            } else if path.is_file() && self.should_scan_file(&path) {
                seen_paths.insert(path.clone());
                if let Some(change) = self.scan_file(&path) {
                    changes.push(change);
                }
            }
        }
    }

    /// Scan a single file for changes.
    fn scan_file(&mut self, path: &Path) -> Option<ContentChange> {
        self.stats.files_scanned += 1;

        // Get current file metadata
        let metadata = match fs::metadata(path) {
            Ok(m) => m,
            Err(_) => {
                self.stats.errors += 1;
                return None;
            }
        };

        let current_mtime = match metadata.modified() {
            Ok(t) => t,
            Err(_) => {
                self.stats.errors += 1;
                return None;
            }
        };
        let current_size = metadata.len();

        // Check if we have a cached entry
        if let Some(cached) = self.hash_cache.get(path) {
            // Fast path: if mtime and size match, assume content is the same
            if cached.mtime == current_mtime && cached.size == current_size {
                self.stats.hash_hits += 1;

                // If hash_on_startup is enabled, verify the hash anyway
                if self.config.hash_on_startup {
                    if let Some(current_hash) = self.compute_file_hash(path) {
                        if current_hash != cached.hash {
                            // Hash mismatch! Content changed without mtime change
                            self.stats.hash_mismatches += 1;
                            let old_hash = cached.hash;

                            // Update cache
                            self.hash_cache.insert(
                                path.to_path_buf(),
                                HashCacheEntry {
                                    mtime: current_mtime,
                                    hash: current_hash,
                                    size: current_size,
                                },
                            );

                            return Some(ContentChange::new(
                                path.to_path_buf(),
                                ContentChangeKind::HashMismatch,
                                Some(old_hash),
                                Some(current_hash),
                            ));
                        }
                    }
                }

                return None;
            }

            // Mtime or size changed - compute new hash
            self.stats.hash_computed += 1;
            if let Some(current_hash) = self.compute_file_hash(path) {
                let old_hash = cached.hash;

                // Update cache
                self.hash_cache.insert(
                    path.to_path_buf(),
                    HashCacheEntry {
                        mtime: current_mtime,
                        hash: current_hash,
                        size: current_size,
                    },
                );

                if current_hash != old_hash {
                    self.stats.files_modified += 1;
                    return Some(ContentChange::new(
                        path.to_path_buf(),
                        ContentChangeKind::Modified,
                        Some(old_hash),
                        Some(current_hash),
                    ));
                }
            }

            return None;
        }

        // No cached entry - this is a new file
        self.stats.hash_computed += 1;
        if let Some(hash) = self.compute_file_hash(path) {
            self.hash_cache.insert(
                path.to_path_buf(),
                HashCacheEntry {
                    mtime: current_mtime,
                    hash,
                    size: current_size,
                },
            );

            self.stats.files_added += 1;
            return Some(ContentChange::new(
                path.to_path_buf(),
                ContentChangeKind::Added,
                None,
                Some(hash),
            ));
        }

        None
    }

    /// Verify a file change event with hash comparison.
    ///
    /// Given a `FileChange` from the file watcher, compute the content hash
    /// and determine if the content actually changed.
    pub fn verify_file_change(&mut self, file_change: &FileChange) -> Option<ContentChange> {
        let path = &file_change.path;

        match &file_change.kind {
            FileChangeKind::Created => {
                // New file - compute hash and add to cache
                if !self.should_scan_file(path) {
                    return None;
                }

                let metadata = fs::metadata(path).ok()?;
                let mtime = metadata.modified().ok()?;
                let size = metadata.len();

                self.stats.hash_computed += 1;
                let hash = self.compute_file_hash(path)?;

                self.hash_cache.insert(
                    path.clone(),
                    HashCacheEntry { mtime, hash, size },
                );

                self.stats.files_added += 1;
                Some(ContentChange::new(
                    path.clone(),
                    ContentChangeKind::Added,
                    None,
                    Some(hash),
                ))
            }

            FileChangeKind::Modified => {
                if !self.should_scan_file(path) {
                    return None;
                }

                let metadata = fs::metadata(path).ok()?;
                let mtime = metadata.modified().ok()?;
                let size = metadata.len();

                self.stats.hash_computed += 1;
                let new_hash = self.compute_file_hash(path)?;

                let old_hash = self.hash_cache.get(path).map(|e| e.hash);

                // Update cache
                self.hash_cache.insert(
                    path.clone(),
                    HashCacheEntry {
                        mtime,
                        hash: new_hash,
                        size,
                    },
                );

                // Check if content actually changed
                if old_hash.map(|h| h != new_hash).unwrap_or(true) {
                    self.stats.files_modified += 1;
                    Some(ContentChange::new(
                        path.clone(),
                        ContentChangeKind::Modified,
                        old_hash,
                        Some(new_hash),
                    ))
                } else {
                    None
                }
            }

            FileChangeKind::Deleted => {
                let old_hash = self.hash_cache.get(path).map(|e| e.hash);
                self.hash_cache.remove(path);
                self.stats.files_deleted += 1;

                if self.config.warn_on_delete {
                    eprintln!("[WARN] File deleted: {}", path.display());
                }

                Some(ContentChange::new(
                    path.clone(),
                    ContentChangeKind::Deleted,
                    old_hash,
                    None,
                ))
            }

            FileChangeKind::Renamed { from } => {
                // Handle rename as delete + create
                let old_hash = self.hash_cache.get(from).map(|e| e.hash);
                self.hash_cache.remove(from);

                if !self.should_scan_file(path) {
                    // Just the deletion part
                    if old_hash.is_some() {
                        self.stats.files_deleted += 1;
                        return Some(ContentChange::new(
                            from.clone(),
                            ContentChangeKind::Deleted,
                            old_hash,
                            None,
                        ));
                    }
                    return None;
                }

                // Get new file info
                let metadata = fs::metadata(path).ok()?;
                let mtime = metadata.modified().ok()?;
                let size = metadata.len();

                self.stats.hash_computed += 1;
                let new_hash = self.compute_file_hash(path)?;

                self.hash_cache.insert(
                    path.clone(),
                    HashCacheEntry {
                        mtime,
                        hash: new_hash,
                        size,
                    },
                );

                // If hash differs from old location, it's a modification
                // Otherwise just treat it as added at new location
                if old_hash.map(|h| h != new_hash).unwrap_or(true) {
                    self.stats.files_modified += 1;
                    Some(ContentChange::new(
                        path.clone(),
                        ContentChangeKind::Modified,
                        old_hash,
                        Some(new_hash),
                    ))
                } else {
                    self.stats.files_added += 1;
                    Some(ContentChange::new(
                        path.clone(),
                        ContentChangeKind::Added,
                        None,
                        Some(new_hash),
                    ))
                }
            }
        }
    }

    /// Perform a periodic integrity check of all cached files.
    ///
    /// Re-hashes all cached files and returns any that have changed.
    pub fn periodic_integrity_check(&mut self) -> Vec<ContentChange> {
        if !self.config.enable_periodic_scan {
            return Vec::new();
        }

        // Check if enough time has passed
        if let Some(last_scan) = self.last_periodic_scan {
            let elapsed = last_scan.elapsed().as_secs();
            if elapsed < self.config.scan_interval_secs {
                return Vec::new();
            }
        }

        self.last_periodic_scan = Some(Instant::now());

        let mut changes = Vec::new();
        let paths: Vec<PathBuf> = self.hash_cache.keys().cloned().collect();

        for path in paths {
            // Check if file still exists
            if !path.exists() {
                let old_hash = self.hash_cache.get(&path).map(|e| e.hash);
                self.hash_cache.remove(&path);
                self.stats.files_deleted += 1;

                if self.config.warn_on_delete {
                    eprintln!("[WARN] File deleted: {}", path.display());
                }

                changes.push(ContentChange::new(
                    path,
                    ContentChangeKind::Deleted,
                    old_hash,
                    None,
                ));
                continue;
            }

            // Verify hash
            let cached = match self.hash_cache.get(&path) {
                Some(c) => c.clone(),
                None => continue,
            };

            self.stats.hash_computed += 1;
            if let Some(current_hash) = self.compute_file_hash(&path) {
                if current_hash != cached.hash {
                    // Content corrupted or changed
                    self.stats.hash_mismatches += 1;

                    // Update cache with new hash
                    if let Ok(metadata) = fs::metadata(&path) {
                        if let Ok(mtime) = metadata.modified() {
                            self.hash_cache.insert(
                                path.clone(),
                                HashCacheEntry {
                                    mtime,
                                    hash: current_hash,
                                    size: metadata.len(),
                                },
                            );
                        }
                    }

                    changes.push(ContentChange::new(
                        path,
                        ContentChangeKind::HashMismatch,
                        Some(cached.hash),
                        Some(current_hash),
                    ));
                }
            }
        }

        changes
    }

    /// Force a periodic integrity check regardless of timing.
    pub fn force_integrity_check(&mut self) -> Vec<ContentChange> {
        // Reset the timer to force check
        self.last_periodic_scan = None;
        // Temporarily enable periodic scan if disabled
        let was_enabled = self.config.enable_periodic_scan;
        self.config.enable_periodic_scan = true;

        let changes = self.periodic_integrity_check();

        // Restore original setting
        self.config.enable_periodic_scan = was_enabled;

        changes
    }

    /// Invalidate the cache entry for a specific path.
    pub fn invalidate_path(&mut self, path: &Path) {
        self.hash_cache.remove(path);
    }

    /// Invalidate all cache entries under a directory.
    pub fn invalidate_subtree(&mut self, root: &Path) {
        let paths_to_remove: Vec<PathBuf> = self
            .hash_cache
            .keys()
            .filter(|p| p.starts_with(root))
            .cloned()
            .collect();

        for path in paths_to_remove {
            self.hash_cache.remove(&path);
        }
    }

    /// Get the cached hash for a path, if available.
    pub fn get_cached_hash(&self, path: &Path) -> Option<&ContentHash> {
        self.hash_cache.get(path).map(|e| &e.hash)
    }

    /// Get the cached entry for a path (mtime and hash).
    pub fn get_cached_entry(&self, path: &Path) -> Option<(SystemTime, ContentHash)> {
        self.hash_cache.get(path).map(|e| (e.mtime, e.hash))
    }

    /// Get the current configuration.
    pub fn config(&self) -> &ChangeDetectorConfig {
        &self.config
    }

    /// Get the current statistics.
    pub fn stats(&self) -> &ChangeDetectorStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = ChangeDetectorStats::default();
    }

    /// Get the number of cached entries.
    pub fn cached_count(&self) -> usize {
        self.hash_cache.len()
    }

    /// Clear the entire cache.
    pub fn clear_cache(&mut self) {
        self.hash_cache.clear();
    }

    /// Manually add or update a cache entry.
    ///
    /// Useful for seeding the cache from a manifest or previous session.
    pub fn seed_cache(&mut self, path: PathBuf, hash: ContentHash, mtime: SystemTime, size: u64) {
        self.hash_cache.insert(path, HashCacheEntry { mtime, hash, size });
    }

    /// Check if a file should be scanned based on extension filters.
    fn should_scan_file(&self, path: &Path) -> bool {
        let ext = match path.extension().and_then(|e| e.to_str()) {
            Some(e) => e.to_lowercase(),
            None => return self.config.include_extensions.is_empty(),
        };

        // Check exclusions first
        if self.config.exclude_extensions.iter().any(|e| e.to_lowercase() == ext) {
            return false;
        }

        // If include list is empty, include all (that aren't excluded)
        if self.config.include_extensions.is_empty() {
            return true;
        }

        // Check if in include list
        self.config.include_extensions.iter().any(|e| e.to_lowercase() == ext)
    }

    /// Compute the content hash of a file.
    fn compute_file_hash(&self, path: &Path) -> Option<ContentHash> {
        let mut file = File::open(path).ok()?;
        let mut buffer = vec![0u8; HASH_BUFFER_SIZE];
        let mut all_data = Vec::new();

        loop {
            let n = file.read(&mut buffer).ok()?;
            if n == 0 {
                break;
            }
            all_data.extend_from_slice(&buffer[..n]);
        }

        Some(ContentHash::from_bytes(&all_data))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use std::thread;
    use std::time::Duration;
    use tempfile::TempDir;

    // Helper to create a test file with content
    fn create_test_file(dir: &Path, name: &str, content: &[u8]) -> PathBuf {
        let path = dir.join(name);
        let mut file = File::create(&path).unwrap();
        file.write_all(content).unwrap();
        file.sync_all().unwrap();
        path
    }

    // Helper to modify a test file
    fn modify_test_file(path: &Path, content: &[u8]) {
        let mut file = File::create(path).unwrap();
        file.write_all(content).unwrap();
        file.sync_all().unwrap();
    }

    // ========================================================================
    // Hash Comparison Tests (5+ tests)
    // ========================================================================

    #[test]
    fn test_hash_same_content_same_hash() {
        let tmp = TempDir::new().unwrap();
        let config = ChangeDetectorConfig::default();
        let mut detector = ContentChangeDetector::new(config);

        let path1 = create_test_file(tmp.path(), "file1.txt", b"hello world");
        let path2 = create_test_file(tmp.path(), "file2.txt", b"hello world");

        let changes = detector.startup_scan(&[tmp.path().to_path_buf()]);

        let hash1 = detector.get_cached_hash(&path1).unwrap();
        let hash2 = detector.get_cached_hash(&path2).unwrap();

        assert_eq!(hash1, hash2, "Same content should produce same hash");
        assert_eq!(changes.len(), 2); // Both files are new
    }

    #[test]
    fn test_hash_different_content_different_hash() {
        let tmp = TempDir::new().unwrap();
        let config = ChangeDetectorConfig::default();
        let mut detector = ContentChangeDetector::new(config);

        let path1 = create_test_file(tmp.path(), "file1.txt", b"hello world");
        let path2 = create_test_file(tmp.path(), "file2.txt", b"goodbye world");

        detector.startup_scan(&[tmp.path().to_path_buf()]);

        let hash1 = detector.get_cached_hash(&path1).unwrap();
        let hash2 = detector.get_cached_hash(&path2).unwrap();

        assert_ne!(hash1, hash2, "Different content should produce different hash");
    }

    #[test]
    fn test_hash_timestamp_unchanged_but_content_changed() {
        let tmp = TempDir::new().unwrap();
        let config = ChangeDetectorConfig::default().with_hash_on_startup(true);
        let mut detector = ContentChangeDetector::new(config);

        let path = create_test_file(tmp.path(), "file.txt", b"original content");

        // First scan - file is added
        let changes = detector.startup_scan(&[tmp.path().to_path_buf()]);
        assert_eq!(changes.len(), 1);
        assert!(matches!(changes[0].kind, ContentChangeKind::Added));

        let original_hash = *detector.get_cached_hash(&path).unwrap();

        // Manually modify cache entry to simulate unchanged mtime but different content on disk
        // This simulates scenarios like git checkout
        modify_test_file(&path, b"modified content");

        // Get current mtime to simulate "unchanged" mtime scenario
        let metadata = fs::metadata(&path).unwrap();
        let mtime = metadata.modified().unwrap();

        // Manually set cache with old hash but current mtime (simulating git checkout scenario)
        detector.seed_cache(path.clone(), original_hash, mtime, metadata.len());

        // Second scan with hash_on_startup should detect the mismatch
        let changes = detector.startup_scan(&[tmp.path().to_path_buf()]);

        // Should detect hash mismatch
        assert_eq!(changes.len(), 1);
        assert!(matches!(changes[0].kind, ContentChangeKind::HashMismatch));
        assert_eq!(changes[0].old_hash, Some(original_hash));
        assert!(changes[0].new_hash.is_some());
        assert_ne!(changes[0].new_hash, Some(original_hash));
    }

    #[test]
    fn test_hash_verify_modification() {
        let tmp = TempDir::new().unwrap();
        let config = ChangeDetectorConfig::default();
        let mut detector = ContentChangeDetector::new(config);

        let path = create_test_file(tmp.path(), "file.txt", b"original");

        detector.startup_scan(&[tmp.path().to_path_buf()]);
        let original_hash = *detector.get_cached_hash(&path).unwrap();

        // Wait a bit to ensure mtime changes
        thread::sleep(Duration::from_millis(50));

        // Modify file
        modify_test_file(&path, b"modified");

        // Scan again
        let changes = detector.startup_scan(&[tmp.path().to_path_buf()]);

        assert_eq!(changes.len(), 1);
        assert!(matches!(changes[0].kind, ContentChangeKind::Modified));
        assert_eq!(changes[0].old_hash, Some(original_hash));

        let new_hash = detector.get_cached_hash(&path).unwrap();
        assert_ne!(*new_hash, original_hash);
    }

    #[test]
    fn test_hash_empty_file() {
        let tmp = TempDir::new().unwrap();
        let config = ChangeDetectorConfig::default();
        let mut detector = ContentChangeDetector::new(config);

        let path = create_test_file(tmp.path(), "empty.txt", b"");

        let changes = detector.startup_scan(&[tmp.path().to_path_buf()]);

        assert_eq!(changes.len(), 1);
        assert!(detector.get_cached_hash(&path).is_some());

        // Empty file should have a consistent hash
        let hash = *detector.get_cached_hash(&path).unwrap();
        let expected_hash = ContentHash::from_bytes(b"");
        assert_eq!(hash, expected_hash);
    }

    // ========================================================================
    // Startup Scan Tests (4+ tests)
    // ========================================================================

    #[test]
    fn test_startup_scan_detect_new_files() {
        let tmp = TempDir::new().unwrap();
        let config = ChangeDetectorConfig::default();
        let mut detector = ContentChangeDetector::new(config);

        create_test_file(tmp.path(), "file1.txt", b"content1");
        create_test_file(tmp.path(), "file2.txt", b"content2");

        let changes = detector.startup_scan(&[tmp.path().to_path_buf()]);

        assert_eq!(changes.len(), 2);
        assert!(changes.iter().all(|c| matches!(c.kind, ContentChangeKind::Added)));
        assert!(changes.iter().all(|c| c.old_hash.is_none()));
        assert!(changes.iter().all(|c| c.new_hash.is_some()));
    }

    #[test]
    fn test_startup_scan_detect_modified_files() {
        let tmp = TempDir::new().unwrap();
        let config = ChangeDetectorConfig::default();
        let mut detector = ContentChangeDetector::new(config);

        let path = create_test_file(tmp.path(), "file.txt", b"original");

        // Initial scan
        detector.startup_scan(&[tmp.path().to_path_buf()]);

        // Modify the file
        thread::sleep(Duration::from_millis(50));
        modify_test_file(&path, b"modified content");

        // Second scan
        let changes = detector.startup_scan(&[tmp.path().to_path_buf()]);

        assert_eq!(changes.len(), 1);
        assert!(matches!(changes[0].kind, ContentChangeKind::Modified));
    }

    #[test]
    fn test_startup_scan_detect_deleted_files() {
        let tmp = TempDir::new().unwrap();
        let config = ChangeDetectorConfig::default().with_warn_on_delete(false);
        let mut detector = ContentChangeDetector::new(config);

        let path = create_test_file(tmp.path(), "file.txt", b"content");

        // Initial scan
        detector.startup_scan(&[tmp.path().to_path_buf()]);
        assert!(detector.get_cached_hash(&path).is_some());

        // Delete the file
        fs::remove_file(&path).unwrap();

        // Second scan
        let changes = detector.startup_scan(&[tmp.path().to_path_buf()]);

        assert_eq!(changes.len(), 1);
        assert!(matches!(changes[0].kind, ContentChangeKind::Deleted));
        assert!(changes[0].old_hash.is_some());
        assert!(changes[0].new_hash.is_none());
        assert!(detector.get_cached_hash(&path).is_none());
    }

    #[test]
    fn test_startup_scan_nested_directories() {
        let tmp = TempDir::new().unwrap();
        let config = ChangeDetectorConfig::default();
        let mut detector = ContentChangeDetector::new(config);

        // Create nested structure
        let subdir = tmp.path().join("subdir");
        fs::create_dir_all(&subdir).unwrap();

        create_test_file(tmp.path(), "root.txt", b"root");
        create_test_file(&subdir, "nested.txt", b"nested");

        let changes = detector.startup_scan(&[tmp.path().to_path_buf()]);

        assert_eq!(changes.len(), 2);
        assert!(changes.iter().all(|c| matches!(c.kind, ContentChangeKind::Added)));
    }

    // ========================================================================
    // File Change Verification Tests (4+ tests)
    // ========================================================================

    #[test]
    fn test_verify_file_change_create() {
        let tmp = TempDir::new().unwrap();
        let config = ChangeDetectorConfig::default();
        let mut detector = ContentChangeDetector::new(config);

        let path = create_test_file(tmp.path(), "new_file.txt", b"new content");

        let file_change = FileChange::new(path.clone(), FileChangeKind::Created);
        let content_change = detector.verify_file_change(&file_change);

        assert!(content_change.is_some());
        let change = content_change.unwrap();
        assert!(matches!(change.kind, ContentChangeKind::Added));
        assert!(change.old_hash.is_none());
        assert!(change.new_hash.is_some());
    }

    #[test]
    fn test_verify_file_change_modify() {
        let tmp = TempDir::new().unwrap();
        let config = ChangeDetectorConfig::default();
        let mut detector = ContentChangeDetector::new(config);

        let path = create_test_file(tmp.path(), "file.txt", b"original");

        // Seed cache
        let file_change = FileChange::new(path.clone(), FileChangeKind::Created);
        detector.verify_file_change(&file_change);

        // Modify file
        modify_test_file(&path, b"modified");

        let file_change = FileChange::new(path.clone(), FileChangeKind::Modified);
        let content_change = detector.verify_file_change(&file_change);

        assert!(content_change.is_some());
        let change = content_change.unwrap();
        assert!(matches!(change.kind, ContentChangeKind::Modified));
        assert!(change.old_hash.is_some());
        assert!(change.new_hash.is_some());
        assert_ne!(change.old_hash, change.new_hash);
    }

    #[test]
    fn test_verify_file_change_delete() {
        let tmp = TempDir::new().unwrap();
        let config = ChangeDetectorConfig::default().with_warn_on_delete(false);
        let mut detector = ContentChangeDetector::new(config);

        let path = create_test_file(tmp.path(), "file.txt", b"content");

        // Seed cache
        let file_change = FileChange::new(path.clone(), FileChangeKind::Created);
        detector.verify_file_change(&file_change);

        let original_hash = *detector.get_cached_hash(&path).unwrap();

        // Delete the file
        fs::remove_file(&path).unwrap();

        let file_change = FileChange::new(path.clone(), FileChangeKind::Deleted);
        let content_change = detector.verify_file_change(&file_change);

        assert!(content_change.is_some());
        let change = content_change.unwrap();
        assert!(matches!(change.kind, ContentChangeKind::Deleted));
        assert_eq!(change.old_hash, Some(original_hash));
        assert!(change.new_hash.is_none());
    }

    #[test]
    fn test_verify_file_change_rename() {
        let tmp = TempDir::new().unwrap();
        let config = ChangeDetectorConfig::default();
        let mut detector = ContentChangeDetector::new(config);

        let old_path = create_test_file(tmp.path(), "old.txt", b"content");
        let new_path = tmp.path().join("new.txt");

        // Seed cache for old path
        let file_change = FileChange::new(old_path.clone(), FileChangeKind::Created);
        detector.verify_file_change(&file_change);

        // Rename the file
        fs::rename(&old_path, &new_path).unwrap();

        let file_change = FileChange::new(
            new_path.clone(),
            FileChangeKind::Renamed { from: old_path.clone() },
        );
        let content_change = detector.verify_file_change(&file_change);

        assert!(content_change.is_some());
        // Old path should be removed from cache
        assert!(detector.get_cached_hash(&old_path).is_none());
        // New path should be in cache
        assert!(detector.get_cached_hash(&new_path).is_some());
    }

    // ========================================================================
    // Periodic Scan Tests (3+ tests)
    // ========================================================================

    #[test]
    fn test_periodic_scan_finds_corrupted_files() {
        let tmp = TempDir::new().unwrap();
        let config = ChangeDetectorConfig::default()
            .with_periodic_scan(true)
            .with_scan_interval_secs(0); // Immediate
        let mut detector = ContentChangeDetector::new(config);

        let path = create_test_file(tmp.path(), "file.txt", b"original");

        // Initial scan
        detector.startup_scan(&[tmp.path().to_path_buf()]);
        let original_hash = *detector.get_cached_hash(&path).unwrap();

        // Manually corrupt the cache entry (simulate hash corruption)
        let metadata = fs::metadata(&path).unwrap();
        let mtime = metadata.modified().unwrap();
        detector.seed_cache(
            path.clone(),
            ContentHash::from_bytes(b"wrong hash source"),
            mtime,
            metadata.len(),
        );

        // Periodic check should detect mismatch
        let changes = detector.periodic_integrity_check();

        assert_eq!(changes.len(), 1);
        assert!(matches!(changes[0].kind, ContentChangeKind::HashMismatch));
        assert_ne!(changes[0].old_hash, changes[0].new_hash);
        assert_eq!(changes[0].new_hash, Some(original_hash));
    }

    #[test]
    fn test_periodic_scan_verifies_all_cached_paths() {
        let tmp = TempDir::new().unwrap();
        let config = ChangeDetectorConfig::default()
            .with_periodic_scan(true)
            .with_scan_interval_secs(0);
        let mut detector = ContentChangeDetector::new(config);

        create_test_file(tmp.path(), "file1.txt", b"content1");
        create_test_file(tmp.path(), "file2.txt", b"content2");
        create_test_file(tmp.path(), "file3.txt", b"content3");

        detector.startup_scan(&[tmp.path().to_path_buf()]);
        assert_eq!(detector.cached_count(), 3);

        // Periodic check with no changes
        let changes = detector.periodic_integrity_check();
        assert!(changes.is_empty());
    }

    #[test]
    fn test_periodic_scan_respects_config() {
        let tmp = TempDir::new().unwrap();
        let config = ChangeDetectorConfig::default()
            .with_periodic_scan(true)
            .with_scan_interval_secs(3600); // 1 hour
        let mut detector = ContentChangeDetector::new(config);

        create_test_file(tmp.path(), "file.txt", b"content");
        detector.startup_scan(&[tmp.path().to_path_buf()]);

        // First check should run
        let changes1 = detector.periodic_integrity_check();
        assert!(changes1.is_empty());

        // Second immediate check should be skipped due to interval
        let changes2 = detector.periodic_integrity_check();
        assert!(changes2.is_empty());
    }

    #[test]
    fn test_periodic_scan_disabled() {
        let tmp = TempDir::new().unwrap();
        let config = ChangeDetectorConfig::default()
            .with_periodic_scan(false);
        let mut detector = ContentChangeDetector::new(config);

        create_test_file(tmp.path(), "file.txt", b"content");
        detector.startup_scan(&[tmp.path().to_path_buf()]);

        let changes = detector.periodic_integrity_check();
        assert!(changes.is_empty());
    }

    // ========================================================================
    // Cache Invalidation Tests (2+ tests)
    // ========================================================================

    #[test]
    fn test_invalidate_single_path() {
        let tmp = TempDir::new().unwrap();
        let config = ChangeDetectorConfig::default();
        let mut detector = ContentChangeDetector::new(config);

        let path1 = create_test_file(tmp.path(), "file1.txt", b"content1");
        let path2 = create_test_file(tmp.path(), "file2.txt", b"content2");

        detector.startup_scan(&[tmp.path().to_path_buf()]);
        assert_eq!(detector.cached_count(), 2);

        detector.invalidate_path(&path1);

        assert!(detector.get_cached_hash(&path1).is_none());
        assert!(detector.get_cached_hash(&path2).is_some());
        assert_eq!(detector.cached_count(), 1);
    }

    #[test]
    fn test_invalidate_subtree() {
        let tmp = TempDir::new().unwrap();
        let config = ChangeDetectorConfig::default();
        let mut detector = ContentChangeDetector::new(config);

        // Create nested structure
        let subdir = tmp.path().join("subdir");
        fs::create_dir_all(&subdir).unwrap();

        let root_file = create_test_file(tmp.path(), "root.txt", b"root");
        let nested_file1 = create_test_file(&subdir, "nested1.txt", b"nested1");
        let nested_file2 = create_test_file(&subdir, "nested2.txt", b"nested2");

        detector.startup_scan(&[tmp.path().to_path_buf()]);
        assert_eq!(detector.cached_count(), 3);

        // Invalidate only the subdir
        detector.invalidate_subtree(&subdir);

        assert!(detector.get_cached_hash(&root_file).is_some());
        assert!(detector.get_cached_hash(&nested_file1).is_none());
        assert!(detector.get_cached_hash(&nested_file2).is_none());
        assert_eq!(detector.cached_count(), 1);
    }

    // ========================================================================
    // Edge Case Tests (2+ tests)
    // ========================================================================

    #[test]
    fn test_binary_file_hashing() {
        let tmp = TempDir::new().unwrap();
        let config = ChangeDetectorConfig::default();
        let mut detector = ContentChangeDetector::new(config);

        // Binary content with null bytes and various byte values
        let binary_content: Vec<u8> = (0..=255).collect();
        let path = create_test_file(tmp.path(), "binary.bin", &binary_content);

        let changes = detector.startup_scan(&[tmp.path().to_path_buf()]);

        assert_eq!(changes.len(), 1);
        let hash = detector.get_cached_hash(&path).unwrap();
        assert_eq!(*hash, ContentHash::from_bytes(&binary_content));
    }

    #[test]
    fn test_symlink_handling() {
        // Skip on Windows where symlinks require special permissions
        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;

            let tmp = TempDir::new().unwrap();
            let config = ChangeDetectorConfig::default();
            let mut detector = ContentChangeDetector::new(config);

            let real_file = create_test_file(tmp.path(), "real.txt", b"real content");
            let link_path = tmp.path().join("link.txt");
            symlink(&real_file, &link_path).unwrap();

            let changes = detector.startup_scan(&[tmp.path().to_path_buf()]);

            // Both files should be detected (symlink resolves to file)
            assert_eq!(changes.len(), 2);

            // Both should have the same hash (same content)
            let real_hash = detector.get_cached_hash(&real_file).unwrap();
            let link_hash = detector.get_cached_hash(&link_path).unwrap();
            assert_eq!(real_hash, link_hash);
        }
    }

    #[test]
    fn test_large_file_hashing() {
        let tmp = TempDir::new().unwrap();
        let config = ChangeDetectorConfig::default();
        let mut detector = ContentChangeDetector::new(config);

        // Create a file larger than the buffer size (>64KB)
        let large_content: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();
        let path = create_test_file(tmp.path(), "large.bin", &large_content);

        let changes = detector.startup_scan(&[tmp.path().to_path_buf()]);

        assert_eq!(changes.len(), 1);
        let hash = detector.get_cached_hash(&path).unwrap();
        assert_eq!(*hash, ContentHash::from_bytes(&large_content));
    }

    // ========================================================================
    // Configuration Tests
    // ========================================================================

    #[test]
    fn test_config_extension_filtering() {
        let tmp = TempDir::new().unwrap();
        let config = ChangeDetectorConfig::default()
            .with_include_extensions(vec!["txt".to_string()]);
        let mut detector = ContentChangeDetector::new(config);

        create_test_file(tmp.path(), "included.txt", b"included");
        create_test_file(tmp.path(), "excluded.bin", b"excluded");

        let changes = detector.startup_scan(&[tmp.path().to_path_buf()]);

        assert_eq!(changes.len(), 1);
        assert!(changes[0].path.to_string_lossy().contains("included.txt"));
    }

    #[test]
    fn test_config_exclude_extensions() {
        let tmp = TempDir::new().unwrap();
        let config = ChangeDetectorConfig::default()
            .with_exclude_extensions(vec!["tmp".to_string(), "swp".to_string()]);
        let mut detector = ContentChangeDetector::new(config);

        create_test_file(tmp.path(), "keep.txt", b"keep");
        create_test_file(tmp.path(), "ignore.tmp", b"ignore");
        create_test_file(tmp.path(), "also_ignore.swp", b"also ignore");

        let changes = detector.startup_scan(&[tmp.path().to_path_buf()]);

        assert_eq!(changes.len(), 1);
        assert!(changes[0].path.to_string_lossy().contains("keep.txt"));
    }

    // ========================================================================
    // Statistics Tests
    // ========================================================================

    #[test]
    fn test_statistics_tracking() {
        let tmp = TempDir::new().unwrap();
        let config = ChangeDetectorConfig::default().with_warn_on_delete(false);
        let mut detector = ContentChangeDetector::new(config);

        create_test_file(tmp.path(), "file1.txt", b"content1");
        create_test_file(tmp.path(), "file2.txt", b"content2");

        detector.startup_scan(&[tmp.path().to_path_buf()]);

        let stats = detector.stats();
        assert_eq!(stats.files_scanned, 2);
        assert_eq!(stats.files_added, 2);
        assert_eq!(stats.hash_computed, 2);
    }

    #[test]
    fn test_statistics_reset() {
        let tmp = TempDir::new().unwrap();
        let config = ChangeDetectorConfig::default();
        let mut detector = ContentChangeDetector::new(config);

        create_test_file(tmp.path(), "file.txt", b"content");
        detector.startup_scan(&[tmp.path().to_path_buf()]);

        assert!(detector.stats().files_scanned > 0);

        detector.reset_stats();

        assert_eq!(detector.stats().files_scanned, 0);
        assert_eq!(detector.stats().files_added, 0);
    }

    // ========================================================================
    // ContentChange Helper Tests
    // ========================================================================

    #[test]
    fn test_content_change_requires_invalidation() {
        let added = ContentChange::new(
            PathBuf::from("test.txt"),
            ContentChangeKind::Added,
            None,
            Some(ContentHash::from_bytes(b"test")),
        );
        assert!(added.requires_invalidation());

        let modified = ContentChange::new(
            PathBuf::from("test.txt"),
            ContentChangeKind::Modified,
            Some(ContentHash::from_bytes(b"old")),
            Some(ContentHash::from_bytes(b"new")),
        );
        assert!(modified.requires_invalidation());

        let deleted = ContentChange::new(
            PathBuf::from("test.txt"),
            ContentChangeKind::Deleted,
            Some(ContentHash::from_bytes(b"old")),
            None,
        );
        assert!(deleted.requires_invalidation());

        let mismatch = ContentChange::new(
            PathBuf::from("test.txt"),
            ContentChangeKind::HashMismatch,
            Some(ContentHash::from_bytes(b"cached")),
            Some(ContentHash::from_bytes(b"actual")),
        );
        assert!(mismatch.requires_invalidation());
    }

    #[test]
    fn test_content_change_kind_properties() {
        assert!(ContentChangeKind::Added.requires_reload());
        assert!(ContentChangeKind::Modified.requires_reload());
        assert!(ContentChangeKind::HashMismatch.requires_reload());
        assert!(!ContentChangeKind::Deleted.requires_reload());

        assert!(!ContentChangeKind::Added.is_deletion());
        assert!(!ContentChangeKind::Modified.is_deletion());
        assert!(!ContentChangeKind::HashMismatch.is_deletion());
        assert!(ContentChangeKind::Deleted.is_deletion());
    }

    // ========================================================================
    // Force Integrity Check Tests
    // ========================================================================

    #[test]
    fn test_force_integrity_check() {
        let tmp = TempDir::new().unwrap();
        let config = ChangeDetectorConfig::default()
            .with_periodic_scan(false); // Disabled by default
        let mut detector = ContentChangeDetector::new(config);

        let path = create_test_file(tmp.path(), "file.txt", b"content");
        detector.startup_scan(&[tmp.path().to_path_buf()]);

        // Normal periodic check should return empty (disabled)
        let changes = detector.periodic_integrity_check();
        assert!(changes.is_empty());

        // Force check should work regardless of config
        let changes = detector.force_integrity_check();
        assert!(changes.is_empty()); // No actual changes

        // Verify the config was restored
        assert!(!detector.config().enable_periodic_scan);
    }

    // ========================================================================
    // Seed Cache Tests
    // ========================================================================

    #[test]
    fn test_seed_cache() {
        let tmp = TempDir::new().unwrap();
        let config = ChangeDetectorConfig::default();
        let mut detector = ContentChangeDetector::new(config);

        let path = tmp.path().join("seeded.txt");
        let hash = ContentHash::from_bytes(b"test content");
        let mtime = SystemTime::now();

        detector.seed_cache(path.clone(), hash, mtime, 12);

        let (cached_mtime, cached_hash) = detector.get_cached_entry(&path).unwrap();
        assert_eq!(cached_hash, hash);
        assert_eq!(cached_mtime, mtime);
    }

    // ========================================================================
    // Clear Cache Tests
    // ========================================================================

    #[test]
    fn test_clear_cache() {
        let tmp = TempDir::new().unwrap();
        let config = ChangeDetectorConfig::default();
        let mut detector = ContentChangeDetector::new(config);

        create_test_file(tmp.path(), "file1.txt", b"content1");
        create_test_file(tmp.path(), "file2.txt", b"content2");

        detector.startup_scan(&[tmp.path().to_path_buf()]);
        assert_eq!(detector.cached_count(), 2);

        detector.clear_cache();
        assert_eq!(detector.cached_count(), 0);
    }
}
