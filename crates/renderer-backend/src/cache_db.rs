//! Database-backed cache with TTL expiry and LRU eviction.
//!
//! This module provides a SQLite-backed metadata store for content-addressed
//! caching with per-entry TTL expiry and configurable eviction policies.
//!
//! # Architecture
//!
//! ```text
//! +----------------+     +-------------+
//! |   TtlCache     |---->| FileBackend | (binary data)
//! |                |     +-------------+
//! |                |     +-------------+
//! |                |---->|   CacheDb   | (metadata in SQLite)
//! +----------------+     +-------------+
//! ```
//!
//! - Binary blobs remain in the content-addressed [`FileBackend`](crate::pipeline::FileBackend)
//! - Metadata (timestamps, TTL, access counts) lives in SQLite for fast queries
//! - Expired entries are auto-evicted on access or via batch cleanup
//!
//! # Example
//!
//! ```ignore
//! let cache = TtlCache::new("/tmp/cache", "/tmp/cache.db")?;
//!
//! // Store with 60-second TTL
//! let hash = cache.put(b"hello world", Some(60))?;
//!
//! // Retrieve (returns None if expired)
//! let data = cache.get(&hash)?;
//!
//! // Batch cleanup of expired entries
//! let removed = cache.evict_expired()?;
//! ```

use std::io;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection, OptionalExtension};

use crate::pipeline::{ContentHash, FileBackend};

// ---------------------------------------------------------------------------
// CacheEntry — metadata for a cached item
// ---------------------------------------------------------------------------

/// Metadata for a single cache entry.
///
/// Tracks creation time, last access, access count, and optional TTL.
/// The actual binary data is stored separately in a [`FileBackend`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheEntry {
    /// Content hash (SHA-256 or BLAKE3 depending on feature flags).
    pub hash: ContentHash,
    /// Size of the cached data in bytes.
    pub size: usize,
    /// Unix timestamp when the entry was created.
    pub created_at: u64,
    /// Unix timestamp when the entry was last accessed.
    pub last_access: u64,
    /// Number of times the entry has been accessed.
    pub access_count: u64,
    /// Optional TTL in seconds from creation time.
    /// If `None`, the entry never expires based on time (but may still be evicted by LRU).
    pub ttl_seconds: Option<u64>,
    /// Bitflags for entry state (reserved for future use).
    /// Bit 0: pinned (never evicted)
    /// Bit 1: compressed
    /// Bits 2-31: reserved
    pub flags: u32,
}

impl CacheEntry {
    /// Check if this entry has expired based on current time.
    pub fn is_expired(&self) -> bool {
        self.is_expired_at(current_timestamp())
    }

    /// Check if this entry has expired at a specific timestamp.
    pub fn is_expired_at(&self, now: u64) -> bool {
        match self.ttl_seconds {
            Some(ttl) => self.created_at.saturating_add(ttl) <= now,
            None => false,
        }
    }

    /// Check if this entry is pinned (never evicted).
    pub fn is_pinned(&self) -> bool {
        self.flags & 1 != 0
    }

    /// Calculate the expiry timestamp, or `None` if the entry never expires.
    pub fn expires_at(&self) -> Option<u64> {
        self.ttl_seconds.map(|ttl| self.created_at.saturating_add(ttl))
    }
}

/// Flags for cache entries.
pub mod entry_flags {
    /// Entry is pinned and should never be evicted.
    pub const PINNED: u32 = 1 << 0;
    /// Entry data is compressed.
    pub const COMPRESSED: u32 = 1 << 1;
}

// ---------------------------------------------------------------------------
// Helper: current Unix timestamp
// ---------------------------------------------------------------------------

/// Returns the current Unix timestamp in seconds.
pub fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// CacheDb — SQLite-backed metadata store
// ---------------------------------------------------------------------------

/// SQLite-backed metadata store for cache entries.
///
/// Stores entry metadata (timestamps, TTL, access counts) separately from
/// the binary data. Provides fast queries for eviction candidates.
///
/// # Schema
///
/// ```sql
/// CREATE TABLE cache_entries (
///     hash_hex TEXT PRIMARY KEY,
///     size INTEGER NOT NULL,
///     created_at INTEGER NOT NULL,
///     last_access INTEGER NOT NULL,
///     access_count INTEGER NOT NULL,
///     ttl_seconds INTEGER,
///     flags INTEGER NOT NULL DEFAULT 0
/// );
///
/// CREATE INDEX idx_last_access ON cache_entries(last_access);
/// CREATE INDEX idx_expires_at ON cache_entries(created_at + ttl_seconds)
///     WHERE ttl_seconds IS NOT NULL;
/// ```
pub struct CacheDb {
    conn: Connection,
}

impl CacheDb {
    /// Open or create a cache database at the given path.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, CacheDbError> {
        let conn = Connection::open(path)?;
        let db = Self { conn };
        db.initialize_schema()?;
        Ok(db)
    }

    /// Create an in-memory cache database (useful for testing).
    pub fn in_memory() -> Result<Self, CacheDbError> {
        let conn = Connection::open_in_memory()?;
        let db = Self { conn };
        db.initialize_schema()?;
        Ok(db)
    }

    /// Initialize the database schema.
    fn initialize_schema(&self) -> Result<(), CacheDbError> {
        self.conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS cache_entries (
                hash_hex TEXT PRIMARY KEY,
                size INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                last_access INTEGER NOT NULL,
                access_count INTEGER NOT NULL,
                ttl_seconds INTEGER,
                flags INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_last_access
                ON cache_entries(last_access);

            CREATE INDEX IF NOT EXISTS idx_created_at
                ON cache_entries(created_at);

            CREATE INDEX IF NOT EXISTS idx_flags
                ON cache_entries(flags);
            "#,
        )?;
        Ok(())
    }

    /// Insert a new cache entry.
    ///
    /// If an entry with the same hash already exists, it is replaced.
    pub fn insert(&self, entry: &CacheEntry) -> Result<(), CacheDbError> {
        let hash_hex = format!("{}", entry.hash);
        self.conn.execute(
            r#"
            INSERT OR REPLACE INTO cache_entries
                (hash_hex, size, created_at, last_access, access_count, ttl_seconds, flags)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            "#,
            params![
                hash_hex,
                entry.size as i64,
                entry.created_at as i64,
                entry.last_access as i64,
                entry.access_count as i64,
                entry.ttl_seconds.map(|t| t as i64),
                entry.flags as i64,
            ],
        )?;
        Ok(())
    }

    /// Get a cache entry by hash.
    ///
    /// Returns `None` if the entry doesn't exist.
    pub fn get(&self, hash: &ContentHash) -> Result<Option<CacheEntry>, CacheDbError> {
        let hash_hex = format!("{}", hash);
        let result = self
            .conn
            .query_row(
                r#"
                SELECT size, created_at, last_access, access_count, ttl_seconds, flags
                FROM cache_entries
                WHERE hash_hex = ?1
                "#,
                params![hash_hex],
                |row| {
                    Ok(CacheEntry {
                        hash: *hash,
                        size: row.get::<_, i64>(0)? as usize,
                        created_at: row.get::<_, i64>(1)? as u64,
                        last_access: row.get::<_, i64>(2)? as u64,
                        access_count: row.get::<_, i64>(3)? as u64,
                        ttl_seconds: row.get::<_, Option<i64>>(4)?.map(|t| t as u64),
                        flags: row.get::<_, i64>(5)? as u32,
                    })
                },
            )
            .optional()?;
        Ok(result)
    }

    /// Update the last_access timestamp and increment access_count.
    ///
    /// Returns `true` if the entry was found and updated.
    pub fn touch(&self, hash: &ContentHash) -> Result<bool, CacheDbError> {
        self.touch_at(hash, current_timestamp())
    }

    /// Update the last_access timestamp to a specific value and increment access_count.
    pub fn touch_at(&self, hash: &ContentHash, timestamp: u64) -> Result<bool, CacheDbError> {
        let hash_hex = format!("{}", hash);
        let rows = self.conn.execute(
            r#"
            UPDATE cache_entries
            SET last_access = ?1, access_count = access_count + 1
            WHERE hash_hex = ?2
            "#,
            params![timestamp as i64, hash_hex],
        )?;
        Ok(rows > 0)
    }

    /// Delete a cache entry by hash.
    ///
    /// Returns `true` if an entry was deleted.
    pub fn delete(&self, hash: &ContentHash) -> Result<bool, CacheDbError> {
        let hash_hex = format!("{}", hash);
        let rows = self.conn.execute(
            "DELETE FROM cache_entries WHERE hash_hex = ?1",
            params![hash_hex],
        )?;
        Ok(rows > 0)
    }

    /// Check if an entry exists and is expired.
    ///
    /// Returns `Some(true)` if expired, `Some(false)` if valid, `None` if not found.
    pub fn is_expired(&self, hash: &ContentHash) -> Result<Option<bool>, CacheDbError> {
        self.is_expired_at(hash, current_timestamp())
    }

    /// Check if an entry is expired at a specific timestamp.
    pub fn is_expired_at(
        &self,
        hash: &ContentHash,
        now: u64,
    ) -> Result<Option<bool>, CacheDbError> {
        match self.get(hash)? {
            Some(entry) => Ok(Some(entry.is_expired_at(now))),
            None => Ok(None),
        }
    }

    /// List all expired entries.
    pub fn list_expired(&self) -> Result<Vec<ContentHash>, CacheDbError> {
        self.list_expired_at(current_timestamp())
    }

    /// List all entries that are expired at a specific timestamp.
    pub fn list_expired_at(&self, now: u64) -> Result<Vec<ContentHash>, CacheDbError> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT hash_hex FROM cache_entries
            WHERE ttl_seconds IS NOT NULL
              AND (created_at + ttl_seconds) <= ?1
              AND (flags & 1) = 0
            "#,
        )?;

        let hashes = stmt
            .query_map(params![now as i64], |row| {
                let hex: String = row.get(0)?;
                Ok(hex)
            })?
            .filter_map(|r| r.ok())
            .filter_map(|hex| hex.parse::<ContentHash>().ok())
            .collect();

        Ok(hashes)
    }

    /// List LRU eviction candidates (least recently accessed, not pinned).
    ///
    /// Returns up to `limit` entries ordered by last_access ascending.
    pub fn list_lru_candidates(&self, limit: usize) -> Result<Vec<CacheEntry>, CacheDbError> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT hash_hex, size, created_at, last_access, access_count, ttl_seconds, flags
            FROM cache_entries
            WHERE (flags & 1) = 0
            ORDER BY last_access ASC
            LIMIT ?1
            "#,
        )?;

        let entries = stmt
            .query_map(params![limit as i64], |row| {
                let hex: String = row.get(0)?;
                let hash = hex.parse::<ContentHash>().map_err(|_| {
                    rusqlite::Error::FromSqlConversionFailure(
                        0,
                        rusqlite::types::Type::Text,
                        "invalid hash".into(),
                    )
                })?;
                Ok(CacheEntry {
                    hash,
                    size: row.get::<_, i64>(1)? as usize,
                    created_at: row.get::<_, i64>(2)? as u64,
                    last_access: row.get::<_, i64>(3)? as u64,
                    access_count: row.get::<_, i64>(4)? as u64,
                    ttl_seconds: row.get::<_, Option<i64>>(5)?.map(|t| t as u64),
                    flags: row.get::<_, i64>(6)? as u32,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(entries)
    }

    /// Count the total number of entries.
    pub fn count(&self) -> Result<usize, CacheDbError> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM cache_entries", [], |row| row.get(0))?;
        Ok(count as usize)
    }

    /// Get the total size of all cached entries.
    pub fn total_size(&self) -> Result<usize, CacheDbError> {
        let size: i64 = self.conn.query_row(
            "SELECT COALESCE(SUM(size), 0) FROM cache_entries",
            [],
            |row| row.get(0),
        )?;
        Ok(size as usize)
    }

    /// List all entries (for debugging/inspection).
    pub fn list_all(&self) -> Result<Vec<CacheEntry>, CacheDbError> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT hash_hex, size, created_at, last_access, access_count, ttl_seconds, flags
            FROM cache_entries
            ORDER BY created_at DESC
            "#,
        )?;

        let entries = stmt
            .query_map([], |row| {
                let hex: String = row.get(0)?;
                let hash = hex.parse::<ContentHash>().map_err(|_| {
                    rusqlite::Error::FromSqlConversionFailure(
                        0,
                        rusqlite::types::Type::Text,
                        "invalid hash".into(),
                    )
                })?;
                Ok(CacheEntry {
                    hash,
                    size: row.get::<_, i64>(1)? as usize,
                    created_at: row.get::<_, i64>(2)? as u64,
                    last_access: row.get::<_, i64>(3)? as u64,
                    access_count: row.get::<_, i64>(4)? as u64,
                    ttl_seconds: row.get::<_, Option<i64>>(5)?.map(|t| t as u64),
                    flags: row.get::<_, i64>(6)? as u32,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(entries)
    }

    /// Delete multiple entries by hash (batch operation).
    pub fn delete_batch(&self, hashes: &[ContentHash]) -> Result<usize, CacheDbError> {
        let mut deleted = 0;
        for hash in hashes {
            if self.delete(hash)? {
                deleted += 1;
            }
        }
        Ok(deleted)
    }

    /// Update entry flags.
    pub fn set_flags(&self, hash: &ContentHash, flags: u32) -> Result<bool, CacheDbError> {
        let hash_hex = format!("{}", hash);
        let rows = self.conn.execute(
            "UPDATE cache_entries SET flags = ?1 WHERE hash_hex = ?2",
            params![flags as i64, hash_hex],
        )?;
        Ok(rows > 0)
    }

    /// Pin an entry (set the PINNED flag).
    pub fn pin(&self, hash: &ContentHash) -> Result<bool, CacheDbError> {
        let hash_hex = format!("{}", hash);
        let rows = self.conn.execute(
            "UPDATE cache_entries SET flags = flags | 1 WHERE hash_hex = ?1",
            params![hash_hex],
        )?;
        Ok(rows > 0)
    }

    /// Unpin an entry (clear the PINNED flag).
    pub fn unpin(&self, hash: &ContentHash) -> Result<bool, CacheDbError> {
        let hash_hex = format!("{}", hash);
        let rows = self.conn.execute(
            "UPDATE cache_entries SET flags = flags & ~1 WHERE hash_hex = ?1",
            params![hash_hex],
        )?;
        Ok(rows > 0)
    }
}

// ---------------------------------------------------------------------------
// CacheDbError
// ---------------------------------------------------------------------------

/// Error type for cache database operations.
#[derive(Debug)]
pub enum CacheDbError {
    /// SQLite error.
    Sqlite(rusqlite::Error),
    /// I/O error.
    Io(io::Error),
    /// Entry not found.
    NotFound,
}

impl std::fmt::Display for CacheDbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sqlite(e) => write!(f, "SQLite error: {}", e),
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::NotFound => write!(f, "cache entry not found"),
        }
    }
}

impl std::error::Error for CacheDbError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Sqlite(e) => Some(e),
            Self::Io(e) => Some(e),
            Self::NotFound => None,
        }
    }
}

impl From<rusqlite::Error> for CacheDbError {
    fn from(e: rusqlite::Error) -> Self {
        Self::Sqlite(e)
    }
}

impl From<io::Error> for CacheDbError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

// ---------------------------------------------------------------------------
// EvictionPolicy
// ---------------------------------------------------------------------------

/// Configuration for cache eviction behavior.
#[derive(Debug, Clone)]
pub struct EvictionPolicy {
    /// Maximum number of entries in the cache.
    /// When exceeded, LRU eviction is triggered.
    pub max_entries: Option<usize>,
    /// Maximum total size of cached data in bytes.
    /// When exceeded, LRU eviction is triggered.
    pub max_size: Option<usize>,
    /// Whether to automatically evict expired entries on access.
    pub auto_evict_expired: bool,
}

impl Default for EvictionPolicy {
    fn default() -> Self {
        Self {
            max_entries: None,
            max_size: None,
            auto_evict_expired: true,
        }
    }
}

impl EvictionPolicy {
    /// Create a policy with entry limit.
    pub fn with_max_entries(max_entries: usize) -> Self {
        Self {
            max_entries: Some(max_entries),
            ..Default::default()
        }
    }

    /// Create a policy with size limit.
    pub fn with_max_size(max_size: usize) -> Self {
        Self {
            max_size: Some(max_size),
            ..Default::default()
        }
    }

    /// Create a policy with both limits.
    pub fn with_limits(max_entries: usize, max_size: usize) -> Self {
        Self {
            max_entries: Some(max_entries),
            max_size: Some(max_size),
            auto_evict_expired: true,
        }
    }
}

// ---------------------------------------------------------------------------
// TtlCache — unified cache with TTL and eviction
// ---------------------------------------------------------------------------

/// High-level cache combining [`FileBackend`] for data and [`CacheDb`] for metadata.
///
/// Provides per-entry TTL expiry, LRU eviction, and configurable limits.
///
/// # Example
///
/// ```ignore
/// let cache = TtlCache::new("/tmp/cache", "/tmp/cache.db")?;
///
/// // Store with 60-second TTL
/// let hash = cache.put(b"hello world", Some(60))?;
///
/// // Retrieve (returns None if expired)
/// let data = cache.get(&hash)?;
///
/// // Batch cleanup
/// let removed = cache.evict_expired()?;
/// ```
pub struct TtlCache {
    /// File backend for binary data.
    backend: FileBackend,
    /// SQLite database for metadata.
    db: CacheDb,
    /// Eviction policy.
    policy: EvictionPolicy,
}

impl TtlCache {
    /// Create a new TTL cache.
    ///
    /// # Arguments
    ///
    /// * `data_path` - Directory for content-addressed blob storage
    /// * `db_path` - Path to SQLite database file
    pub fn new<P1: AsRef<Path>, P2: AsRef<Path>>(
        data_path: P1,
        db_path: P2,
    ) -> Result<Self, CacheDbError> {
        Self::with_policy(data_path, db_path, EvictionPolicy::default())
    }

    /// Create a new TTL cache with a custom eviction policy.
    pub fn with_policy<P1: AsRef<Path>, P2: AsRef<Path>>(
        data_path: P1,
        db_path: P2,
        policy: EvictionPolicy,
    ) -> Result<Self, CacheDbError> {
        let backend = FileBackend::new(data_path)?;
        let db = CacheDb::open(db_path)?;
        Ok(Self {
            backend,
            db,
            policy,
        })
    }

    /// Create an in-memory cache for testing.
    pub fn in_memory<P: AsRef<Path>>(data_path: P) -> Result<Self, CacheDbError> {
        let backend = FileBackend::new(data_path)?;
        let db = CacheDb::in_memory()?;
        Ok(Self {
            backend,
            db,
            policy: EvictionPolicy::default(),
        })
    }

    /// Store data with an optional TTL.
    ///
    /// # Arguments
    ///
    /// * `data` - The data to cache
    /// * `ttl_seconds` - Optional TTL in seconds (None = never expires)
    ///
    /// # Returns
    ///
    /// The content hash of the stored data.
    pub fn put(&self, data: &[u8], ttl_seconds: Option<u64>) -> Result<ContentHash, CacheDbError> {
        self.put_with_flags(data, ttl_seconds, 0)
    }

    /// Store data with TTL and flags.
    pub fn put_with_flags(
        &self,
        data: &[u8],
        ttl_seconds: Option<u64>,
        flags: u32,
    ) -> Result<ContentHash, CacheDbError> {
        // Check if we need to evict before adding
        self.maybe_evict(data.len())?;

        // Store the data
        let hash = self.backend.put(data)?;
        let now = current_timestamp();

        // Create metadata entry
        let entry = CacheEntry {
            hash,
            size: data.len(),
            created_at: now,
            last_access: now,
            access_count: 1,
            ttl_seconds,
            flags,
        };

        self.db.insert(&entry)?;
        Ok(hash)
    }

    /// Retrieve data by hash.
    ///
    /// Returns `None` if the entry doesn't exist or has expired.
    /// Automatically evicts expired entries if `auto_evict_expired` is enabled.
    pub fn get(&self, hash: &ContentHash) -> Result<Option<Vec<u8>>, CacheDbError> {
        // Check metadata first
        let entry = match self.db.get(hash)? {
            Some(e) => e,
            None => return Ok(None),
        };

        // Check expiry
        if entry.is_expired() {
            if self.policy.auto_evict_expired {
                self.evict_entry(hash)?;
            }
            return Ok(None);
        }

        // Touch the entry (update access time)
        self.db.touch(hash)?;

        // Retrieve data from backend
        Ok(self.backend.get(hash)?)
    }

    /// Check if an entry exists and is valid (not expired).
    pub fn has(&self, hash: &ContentHash) -> Result<bool, CacheDbError> {
        match self.db.get(hash)? {
            Some(entry) => Ok(!entry.is_expired()),
            None => Ok(false),
        }
    }

    /// Delete a cache entry.
    pub fn delete(&self, hash: &ContentHash) -> Result<bool, CacheDbError> {
        self.evict_entry(hash)
    }

    /// Evict all expired entries.
    ///
    /// Returns the number of entries evicted.
    pub fn evict_expired(&self) -> Result<usize, CacheDbError> {
        let expired = self.db.list_expired()?;
        let mut evicted = 0;
        for hash in expired {
            if self.evict_entry(&hash)? {
                evicted += 1;
            }
        }
        Ok(evicted)
    }

    /// Get cache statistics.
    pub fn stats(&self) -> Result<CacheStats, CacheDbError> {
        Ok(CacheStats {
            entry_count: self.db.count()?,
            total_size: self.db.total_size()?,
            expired_count: self.db.list_expired()?.len(),
        })
    }

    /// Get the metadata entry for a hash (without touching).
    pub fn entry(&self, hash: &ContentHash) -> Result<Option<CacheEntry>, CacheDbError> {
        self.db.get(hash)
    }

    /// Pin an entry to prevent eviction.
    pub fn pin(&self, hash: &ContentHash) -> Result<bool, CacheDbError> {
        self.db.pin(hash)
    }

    /// Unpin an entry to allow eviction.
    pub fn unpin(&self, hash: &ContentHash) -> Result<bool, CacheDbError> {
        self.db.unpin(hash)
    }

    /// List all entries (for debugging).
    pub fn list_all(&self) -> Result<Vec<CacheEntry>, CacheDbError> {
        self.db.list_all()
    }

    /// Evict a single entry (delete from both backend and db).
    fn evict_entry(&self, hash: &ContentHash) -> Result<bool, CacheDbError> {
        let db_deleted = self.db.delete(hash)?;
        let backend_deleted = self.backend.delete(hash)?;
        Ok(db_deleted || backend_deleted)
    }

    /// Check eviction policy and evict if necessary.
    fn maybe_evict(&self, incoming_size: usize) -> Result<(), CacheDbError> {
        // First, evict any expired entries
        if self.policy.auto_evict_expired {
            self.evict_expired()?;
        }

        // Check entry count limit
        if let Some(max_entries) = self.policy.max_entries {
            while self.db.count()? >= max_entries {
                let candidates = self.db.list_lru_candidates(1)?;
                if let Some(entry) = candidates.first() {
                    self.evict_entry(&entry.hash)?;
                } else {
                    break;
                }
            }
        }

        // Check size limit
        if let Some(max_size) = self.policy.max_size {
            while self.db.total_size()? + incoming_size > max_size {
                let candidates = self.db.list_lru_candidates(1)?;
                if let Some(entry) = candidates.first() {
                    self.evict_entry(&entry.hash)?;
                } else {
                    break;
                }
            }
        }

        Ok(())
    }

    /// Get a reference to the underlying database.
    pub fn db(&self) -> &CacheDb {
        &self.db
    }

    /// Get a reference to the underlying file backend.
    pub fn backend(&self) -> &FileBackend {
        &self.backend
    }

    /// Get the eviction policy.
    pub fn policy(&self) -> &EvictionPolicy {
        &self.policy
    }
}

// ---------------------------------------------------------------------------
// CacheStats
// ---------------------------------------------------------------------------

/// Statistics about the cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheStats {
    /// Total number of entries.
    pub entry_count: usize,
    /// Total size of all entries in bytes.
    pub total_size: usize,
    /// Number of expired entries.
    pub expired_count: usize,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn setup_cache() -> (TtlCache, TempDir) {
        let tmp = TempDir::new().unwrap();
        let data_path = tmp.path().join("data");
        let db_path = tmp.path().join("cache.db");
        let cache = TtlCache::new(&data_path, &db_path).unwrap();
        (cache, tmp)
    }

    fn setup_db() -> CacheDb {
        CacheDb::in_memory().unwrap()
    }

    // ---- Test 1: Insert and retrieve entry ----

    #[test]
    fn test_insert_and_retrieve_entry() {
        let (cache, _tmp) = setup_cache();
        let data = b"hello world";
        let hash = cache.put(data, None).unwrap();

        let retrieved = cache.get(&hash).unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), data);
    }

    // ---- Test 2: TTL expiry works ----

    #[test]
    fn test_ttl_expiry_works() {
        let db = setup_db();
        let hash = ContentHash::from_bytes(b"test");
        let now = 1000;

        let entry = CacheEntry {
            hash,
            size: 100,
            created_at: now,
            last_access: now,
            access_count: 1,
            ttl_seconds: Some(60),
            flags: 0,
        };

        db.insert(&entry).unwrap();

        // Not expired at now + 30
        assert!(!db.is_expired_at(&hash, now + 30).unwrap().unwrap());

        // Expired at now + 60 (boundary)
        assert!(db.is_expired_at(&hash, now + 60).unwrap().unwrap());

        // Expired at now + 100
        assert!(db.is_expired_at(&hash, now + 100).unwrap().unwrap());
    }

    // ---- Test 3: LRU eviction correct ----

    #[test]
    fn test_lru_eviction_correct() {
        let db = setup_db();

        // Insert entries with different last_access times
        for i in 0..5 {
            let hash = ContentHash::from_bytes(&[i as u8]);
            let entry = CacheEntry {
                hash,
                size: 100,
                created_at: 1000,
                last_access: 1000 + i as u64 * 10, // 1000, 1010, 1020, 1030, 1040
                access_count: 1,
                ttl_seconds: None,
                flags: 0,
            };
            db.insert(&entry).unwrap();
        }

        // Get LRU candidates (should be ordered by last_access ascending)
        let candidates = db.list_lru_candidates(3).unwrap();
        assert_eq!(candidates.len(), 3);

        // First should be the oldest (last_access = 1000)
        assert_eq!(candidates[0].last_access, 1000);
        assert_eq!(candidates[1].last_access, 1010);
        assert_eq!(candidates[2].last_access, 1020);
    }

    // ---- Test 4: touch() updates last_access ----

    #[test]
    fn test_touch_updates_last_access() {
        let db = setup_db();
        let hash = ContentHash::from_bytes(b"test");

        let entry = CacheEntry {
            hash,
            size: 100,
            created_at: 1000,
            last_access: 1000,
            access_count: 1,
            ttl_seconds: None,
            flags: 0,
        };

        db.insert(&entry).unwrap();
        db.touch_at(&hash, 2000).unwrap();

        let updated = db.get(&hash).unwrap().unwrap();
        assert_eq!(updated.last_access, 2000);
    }

    // ---- Test 5: access_count increments ----

    #[test]
    fn test_access_count_increments() {
        let db = setup_db();
        let hash = ContentHash::from_bytes(b"test");

        let entry = CacheEntry {
            hash,
            size: 100,
            created_at: 1000,
            last_access: 1000,
            access_count: 1,
            ttl_seconds: None,
            flags: 0,
        };

        db.insert(&entry).unwrap();

        // Touch three times
        db.touch(&hash).unwrap();
        db.touch(&hash).unwrap();
        db.touch(&hash).unwrap();

        let updated = db.get(&hash).unwrap().unwrap();
        assert_eq!(updated.access_count, 4); // 1 initial + 3 touches
    }

    // ---- Test 6: Expired entries auto-evicted on get ----

    #[test]
    fn test_expired_entries_auto_evicted_on_get() {
        let tmp = TempDir::new().unwrap();
        let data_path = tmp.path().join("data");
        let db_path = tmp.path().join("cache.db");

        // Create cache and store entry with very short TTL
        let cache = TtlCache::with_policy(
            &data_path,
            &db_path,
            EvictionPolicy {
                auto_evict_expired: true,
                ..Default::default()
            },
        )
        .unwrap();

        // Insert with TTL of 0 (immediately expired)
        let data = b"test data";
        let hash = ContentHash::from_bytes(data);

        // Manually insert expired entry
        let entry = CacheEntry {
            hash,
            size: data.len(),
            created_at: 0, // Ancient timestamp
            last_access: 0,
            access_count: 1,
            ttl_seconds: Some(1), // Expired
            flags: 0,
        };
        cache.db.insert(&entry).unwrap();
        cache.backend.put(data).unwrap();

        // Try to get - should return None and evict
        let result = cache.get(&hash).unwrap();
        assert!(result.is_none());

        // Entry should be gone from database
        assert!(cache.db.get(&hash).unwrap().is_none());
    }

    // ---- Test 7: Batch eviction ----

    #[test]
    fn test_batch_eviction() {
        let (cache, _tmp) = setup_cache();

        // Insert some entries with expired TTLs
        for i in 0..5 {
            let data = format!("data_{}", i).into_bytes();
            let hash = ContentHash::from_bytes(&data);

            let entry = CacheEntry {
                hash,
                size: data.len(),
                created_at: 0, // Ancient
                last_access: 0,
                access_count: 1,
                ttl_seconds: Some(1), // Expired
                flags: 0,
            };
            cache.db.insert(&entry).unwrap();
            cache.backend.put(&data).unwrap();
        }

        // Insert some non-expired entries
        for i in 5..8 {
            let data = format!("data_{}", i).into_bytes();
            cache.put(&data, None).unwrap();
        }

        // Evict expired (already auto-evicted during put() calls due to auto_evict_expired: true)
        let evicted = cache.evict_expired().unwrap();
        assert_eq!(evicted, 0); // All 5 were evicted during the put() calls above

        // Only non-expired should remain
        assert_eq!(cache.db.count().unwrap(), 3);
    }

    // ---- Test 8: Empty cache edge cases ----

    #[test]
    fn test_empty_cache_edge_cases() {
        let (cache, _tmp) = setup_cache();

        // Get on empty cache
        let hash = ContentHash::from_bytes(b"nonexistent");
        assert!(cache.get(&hash).unwrap().is_none());

        // Has on empty cache
        assert!(!cache.has(&hash).unwrap());

        // Delete on empty cache
        assert!(!cache.delete(&hash).unwrap());

        // Evict on empty cache
        assert_eq!(cache.evict_expired().unwrap(), 0);

        // Stats on empty cache
        let stats = cache.stats().unwrap();
        assert_eq!(stats.entry_count, 0);
        assert_eq!(stats.total_size, 0);
        assert_eq!(stats.expired_count, 0);
    }

    // ---- Test 9: SQLite query performance (list_expired) ----

    #[test]
    fn test_sqlite_query_performance_list_expired() {
        let db = setup_db();
        let now = current_timestamp();

        // Insert many entries
        for i in 0u32..100 {
            let hash = ContentHash::from_bytes(&i.to_le_bytes());
            let entry = CacheEntry {
                hash,
                size: 100,
                created_at: now - 100, // Created 100 seconds ago
                last_access: now,
                access_count: 1,
                ttl_seconds: if i % 2 == 0 { Some(50) } else { None }, // Half expired
                flags: 0,
            };
            db.insert(&entry).unwrap();
        }

        let start = std::time::Instant::now();
        let expired = db.list_expired().unwrap();
        let elapsed = start.elapsed();

        // Should find 50 expired entries
        assert_eq!(expired.len(), 50);

        // Query should be fast (< 10ms for 100 entries, generous for CI)
        assert!(
            elapsed.as_millis() < 100,
            "Query took too long: {:?}",
            elapsed
        );
    }

    // ---- Test 10: SQLite query performance (list_lru_candidates) ----

    #[test]
    fn test_sqlite_query_performance_lru() {
        let db = setup_db();

        // Insert many entries with varied access times
        for i in 0u32..100 {
            let hash = ContentHash::from_bytes(&i.to_le_bytes());
            let entry = CacheEntry {
                hash,
                size: 100,
                created_at: 1000,
                last_access: 1000 + (i as u64) * 10,
                access_count: 1,
                ttl_seconds: None,
                flags: 0,
            };
            db.insert(&entry).unwrap();
        }

        let start = std::time::Instant::now();
        let candidates = db.list_lru_candidates(10).unwrap();
        let elapsed = start.elapsed();

        assert_eq!(candidates.len(), 10);

        // Query should be fast
        assert!(
            elapsed.as_millis() < 100,
            "Query took too long: {:?}",
            elapsed
        );
    }

    // ---- Test 11: Entry pinning prevents eviction ----

    #[test]
    fn test_entry_pinning_prevents_eviction() {
        let db = setup_db();
        let hash = ContentHash::from_bytes(b"important");

        let entry = CacheEntry {
            hash,
            size: 100,
            created_at: 0, // Ancient
            last_access: 0,
            access_count: 1,
            ttl_seconds: Some(1), // Would be expired
            flags: entry_flags::PINNED,
        };

        db.insert(&entry).unwrap();

        // Should not appear in expired list (pinned)
        let expired = db.list_expired().unwrap();
        assert!(expired.is_empty());

        // Should not appear in LRU candidates (pinned)
        let lru = db.list_lru_candidates(10).unwrap();
        assert!(lru.is_empty());
    }

    // ---- Test 12: Max entries eviction policy ----

    #[test]
    fn test_max_entries_eviction_policy() {
        let tmp = TempDir::new().unwrap();
        let data_path = tmp.path().join("data");
        let db_path = tmp.path().join("cache.db");

        let cache = TtlCache::with_policy(
            &data_path,
            &db_path,
            EvictionPolicy::with_max_entries(3),
        )
        .unwrap();

        // Insert 5 entries
        for i in 0..5 {
            let data = format!("entry_{}", i).into_bytes();
            cache.put(&data, None).unwrap();
            // Small delay to ensure different timestamps
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        // Should have evicted down to max (3)
        assert_eq!(cache.db.count().unwrap(), 3);
    }

    // ---- Test 13: Max size eviction policy ----

    #[test]
    fn test_max_size_eviction_policy() {
        let tmp = TempDir::new().unwrap();
        let data_path = tmp.path().join("data");
        let db_path = tmp.path().join("cache.db");

        // Max size of 200 bytes
        let cache = TtlCache::with_policy(
            &data_path,
            &db_path,
            EvictionPolicy::with_max_size(200),
        )
        .unwrap();

        // Insert entries of 100 bytes each
        for i in 0..5 {
            let data = vec![i as u8; 100];
            cache.put(&data, None).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        // Should have evicted to stay under 200 bytes
        let stats = cache.stats().unwrap();
        assert!(stats.total_size <= 200);
    }

    // ---- Test 14: No TTL means never expires ----

    #[test]
    fn test_no_ttl_never_expires() {
        let db = setup_db();
        let hash = ContentHash::from_bytes(b"permanent");

        let entry = CacheEntry {
            hash,
            size: 100,
            created_at: 0, // Very old
            last_access: 0,
            access_count: 1,
            ttl_seconds: None, // No TTL
            flags: 0,
        };

        db.insert(&entry).unwrap();

        // Should not be expired even with huge timestamp
        assert!(!db.is_expired_at(&hash, u64::MAX).unwrap().unwrap());

        // Should not appear in expired list
        let expired = db.list_expired().unwrap();
        assert!(expired.is_empty());
    }

    // ---- Test 15: CacheEntry is_expired_at method ----

    #[test]
    fn test_cache_entry_is_expired_at() {
        let entry_with_ttl = CacheEntry {
            hash: ContentHash::zero(),
            size: 0,
            created_at: 1000,
            last_access: 1000,
            access_count: 0,
            ttl_seconds: Some(100),
            flags: 0,
        };

        // Not expired before TTL
        assert!(!entry_with_ttl.is_expired_at(1050));
        assert!(!entry_with_ttl.is_expired_at(1099));

        // Expired at exactly TTL
        assert!(entry_with_ttl.is_expired_at(1100));

        // Expired after TTL
        assert!(entry_with_ttl.is_expired_at(2000));

        let entry_no_ttl = CacheEntry {
            hash: ContentHash::zero(),
            size: 0,
            created_at: 0,
            last_access: 0,
            access_count: 0,
            ttl_seconds: None,
            flags: 0,
        };

        // Never expired
        assert!(!entry_no_ttl.is_expired_at(u64::MAX));
    }

    // ---- Test 16: Delete removes from both db and backend ----

    #[test]
    fn test_delete_removes_from_both() {
        let (cache, _tmp) = setup_cache();
        let data = b"to be deleted";
        let hash = cache.put(data, None).unwrap();

        // Verify it exists
        assert!(cache.has(&hash).unwrap());
        assert!(cache.backend.has(&hash));

        // Delete
        assert!(cache.delete(&hash).unwrap());

        // Should be gone from both
        assert!(!cache.has(&hash).unwrap());
        assert!(!cache.backend.has(&hash));
    }

    // ---- Test 17: Database delete_batch ----

    #[test]
    fn test_database_delete_batch() {
        let db = setup_db();
        let mut hashes = Vec::new();

        for i in 0..5 {
            let hash = ContentHash::from_bytes(&[i]);
            let entry = CacheEntry {
                hash,
                size: 100,
                created_at: 1000,
                last_access: 1000,
                access_count: 1,
                ttl_seconds: None,
                flags: 0,
            };
            db.insert(&entry).unwrap();
            hashes.push(hash);
        }

        assert_eq!(db.count().unwrap(), 5);

        // Delete batch of 3
        let deleted = db.delete_batch(&hashes[0..3]).unwrap();
        assert_eq!(deleted, 3);
        assert_eq!(db.count().unwrap(), 2);
    }

    // ---- Test 18: Pin and unpin operations ----

    #[test]
    fn test_pin_unpin_operations() {
        let db = setup_db();
        let hash = ContentHash::from_bytes(b"pinnable");

        let entry = CacheEntry {
            hash,
            size: 100,
            created_at: 1000,
            last_access: 1000,
            access_count: 1,
            ttl_seconds: None,
            flags: 0,
        };

        db.insert(&entry).unwrap();

        // Initially not pinned
        let e = db.get(&hash).unwrap().unwrap();
        assert!(!e.is_pinned());

        // Pin it
        db.pin(&hash).unwrap();
        let e = db.get(&hash).unwrap().unwrap();
        assert!(e.is_pinned());

        // Unpin it
        db.unpin(&hash).unwrap();
        let e = db.get(&hash).unwrap().unwrap();
        assert!(!e.is_pinned());
    }

    // ---- Test 19: expires_at calculation ----

    #[test]
    fn test_expires_at_calculation() {
        let entry_with_ttl = CacheEntry {
            hash: ContentHash::zero(),
            size: 0,
            created_at: 1000,
            last_access: 1000,
            access_count: 0,
            ttl_seconds: Some(3600), // 1 hour
            flags: 0,
        };

        assert_eq!(entry_with_ttl.expires_at(), Some(4600));

        let entry_no_ttl = CacheEntry {
            hash: ContentHash::zero(),
            size: 0,
            created_at: 1000,
            last_access: 1000,
            access_count: 0,
            ttl_seconds: None,
            flags: 0,
        };

        assert_eq!(entry_no_ttl.expires_at(), None);
    }

    // ---- Test 20: CacheStats structure ----

    #[test]
    fn test_cache_stats() {
        let (cache, _tmp) = setup_cache();

        // Add some entries
        cache.put(b"hello", None).unwrap();
        cache.put(b"world", Some(3600)).unwrap();

        let stats = cache.stats().unwrap();
        assert_eq!(stats.entry_count, 2);
        assert_eq!(stats.total_size, 10); // 5 + 5
        assert_eq!(stats.expired_count, 0);
    }
}
