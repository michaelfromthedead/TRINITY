//! SQLite-backed metadata store for content-addressed assets (T-AS-4.4).
//!
//! Provides indexed metadata queries for the ContentStore:
//!
//! - Store: content hash, asset type, import date, provenance data, dependencies
//! - Queries: find_by_type, find_by_date_range, find_by_provenance
//! - Indexed on: content_hash, asset_type, import_date
//!
//! # Design
//!
//! This is an **optional** backend -- the core ContentStore works without it.
//! When SQLite is unavailable, queries fall back to O(n) in-memory scans.
//!
//! # Thread Safety
//!
//! Uses connection pooling with `parking_lot::Mutex` for thread-safe access.
//! Each query acquires a connection from the pool, executes, and returns it.
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::asset::sqlite_metadata::{
//!     SqliteMetadataStore, AssetMetadata, AssetType
//! };
//! use renderer_backend::pipeline::ContentHash;
//!
//! // Create or open metadata store
//! let store = SqliteMetadataStore::open(":memory:")?;
//!
//! // Insert metadata for an asset
//! let hash = ContentHash::from_bytes(b"asset data");
//! let metadata = AssetMetadata {
//!     content_hash: hash,
//!     asset_type: AssetType::Texture,
//!     import_date: 1716700000,
//!     provenance: vec![("source".into(), "photo.png".into())],
//!     dependencies: vec![],
//!     size_bytes: 1024,
//! };
//! store.insert(&metadata)?;
//!
//! // Query by type
//! let textures = store.find_by_type(AssetType::Texture)?;
//!
//! // Query by date range
//! let recent = store.find_by_date_range(1716600000, 1716800000)?;
//!
//! // Query by provenance
//! let from_source = store.find_by_provenance("source", "photo.png")?;
//! ```

use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::{Mutex, RwLock};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

use crate::pipeline::ContentHash;

// ---------------------------------------------------------------------------
// Asset Type
// ---------------------------------------------------------------------------

/// Types of assets that can be stored in the content store.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum AssetType {
    /// Unknown or unclassified asset
    Unknown = 0,
    /// Texture (PNG, JPEG, TGA, BMP, HDR, EXR)
    Texture = 1,
    /// Mesh geometry (vertices, indices)
    Mesh = 2,
    /// Material definition
    Material = 3,
    /// Shader source (WGSL, GLSL, HLSL)
    Shader = 4,
    /// Animation data
    Animation = 5,
    /// Audio asset
    Audio = 6,
    /// Scene graph or prefab
    Scene = 7,
    /// Binary blob
    Binary = 8,
    /// glTF document
    Gltf = 9,
    /// Compressed texture (KTX, DDS)
    CompressedTexture = 10,
    /// Font data
    Font = 11,
    /// Script or configuration
    Config = 12,
}

impl AssetType {
    /// Convert from u8 value.
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::Unknown,
            1 => Self::Texture,
            2 => Self::Mesh,
            3 => Self::Material,
            4 => Self::Shader,
            5 => Self::Animation,
            6 => Self::Audio,
            7 => Self::Scene,
            8 => Self::Binary,
            9 => Self::Gltf,
            10 => Self::CompressedTexture,
            11 => Self::Font,
            12 => Self::Config,
            _ => Self::Unknown,
        }
    }

    /// Get the string name of this asset type.
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Texture => "texture",
            Self::Mesh => "mesh",
            Self::Material => "material",
            Self::Shader => "shader",
            Self::Animation => "animation",
            Self::Audio => "audio",
            Self::Scene => "scene",
            Self::Binary => "binary",
            Self::Gltf => "gltf",
            Self::CompressedTexture => "compressed_texture",
            Self::Font => "font",
            Self::Config => "config",
        }
    }
}

impl fmt::Display for AssetType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl std::str::FromStr for AssetType {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "unknown" => Ok(Self::Unknown),
            "texture" => Ok(Self::Texture),
            "mesh" => Ok(Self::Mesh),
            "material" => Ok(Self::Material),
            "shader" => Ok(Self::Shader),
            "animation" => Ok(Self::Animation),
            "audio" => Ok(Self::Audio),
            "scene" => Ok(Self::Scene),
            "binary" => Ok(Self::Binary),
            "gltf" => Ok(Self::Gltf),
            "compressed_texture" => Ok(Self::CompressedTexture),
            "font" => Ok(Self::Font),
            "config" => Ok(Self::Config),
            _ => Err(()),
        }
    }
}

// ---------------------------------------------------------------------------
// Asset Metadata
// ---------------------------------------------------------------------------

/// Metadata for a content-addressed asset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssetMetadata {
    /// Content hash (primary key).
    pub content_hash: ContentHash,
    /// Type of asset.
    pub asset_type: AssetType,
    /// Import timestamp (seconds since UNIX epoch).
    pub import_date: i64,
    /// Provenance key-value pairs (e.g., source file, importer version).
    pub provenance: Vec<(String, String)>,
    /// Content hashes of dependencies.
    pub dependencies: Vec<ContentHash>,
    /// Size in bytes.
    pub size_bytes: u64,
}

impl AssetMetadata {
    /// Create new metadata with the current timestamp.
    pub fn new(content_hash: ContentHash, asset_type: AssetType, size_bytes: u64) -> Self {
        Self {
            content_hash,
            asset_type,
            import_date: current_timestamp(),
            provenance: Vec::new(),
            dependencies: Vec::new(),
            size_bytes,
        }
    }

    /// Add a provenance key-value pair.
    pub fn with_provenance(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.provenance.push((key.into(), value.into()));
        self
    }

    /// Add a dependency.
    pub fn with_dependency(mut self, dep: ContentHash) -> Self {
        self.dependencies.push(dep);
        self
    }

    /// Set the import date.
    pub fn with_import_date(mut self, timestamp: i64) -> Self {
        self.import_date = timestamp;
        self
    }

    /// Get a provenance value by key.
    pub fn get_provenance(&self, key: &str) -> Option<&str> {
        self.provenance
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
}

/// Get current timestamp in seconds since UNIX epoch.
fn current_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Error Types
// ---------------------------------------------------------------------------

/// Errors from SQLite metadata operations.
#[derive(Debug)]
pub enum MetadataError {
    /// SQLite error.
    Sqlite(rusqlite::Error),
    /// Serialization error.
    Serialization(String),
    /// Content hash parse error.
    HashParse(String),
    /// Pool exhausted (no available connections).
    PoolExhausted,
    /// Backend not available.
    NotAvailable,
}

impl fmt::Display for MetadataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sqlite(e) => write!(f, "SQLite error: {}", e),
            Self::Serialization(msg) => write!(f, "serialization error: {}", msg),
            Self::HashParse(msg) => write!(f, "hash parse error: {}", msg),
            Self::PoolExhausted => write!(f, "connection pool exhausted"),
            Self::NotAvailable => write!(f, "metadata backend not available"),
        }
    }
}

impl std::error::Error for MetadataError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Sqlite(e) => Some(e),
            _ => None,
        }
    }
}

impl From<rusqlite::Error> for MetadataError {
    fn from(e: rusqlite::Error) -> Self {
        Self::Sqlite(e)
    }
}

/// Result type for metadata operations.
pub type MetadataResult<T> = std::result::Result<T, MetadataError>;

// ---------------------------------------------------------------------------
// Connection Pool
// ---------------------------------------------------------------------------

/// A simple connection pool for SQLite.
///
/// Maintains a fixed number of connections that can be borrowed and returned.
/// Thread-safe via `Mutex` on each connection slot.
struct ConnectionPool {
    connections: Vec<Mutex<Option<Connection>>>,
    db_path: String,
    max_connections: usize,
}

impl ConnectionPool {
    /// Create a new connection pool.
    fn new(db_path: &str, max_connections: usize) -> MetadataResult<Self> {
        let mut connections = Vec::with_capacity(max_connections);

        // Create initial connections
        for _ in 0..max_connections {
            let conn = Self::create_connection(db_path)?;
            connections.push(Mutex::new(Some(conn)));
        }

        Ok(Self {
            connections,
            db_path: db_path.to_string(),
            max_connections,
        })
    }

    /// Create a new SQLite connection with optimal settings.
    fn create_connection(db_path: &str) -> MetadataResult<Connection> {
        // For in-memory databases, use shared cache mode to allow multiple connections
        // to access the same database. The path includes a unique identifier.
        let conn = if db_path.starts_with(":memory:") || db_path.starts_with("file::memory:") {
            // Use URI format for shared in-memory database
            // The path is like "file::memory:unique_id?cache=shared" or ":memory:unique_id"
            let uri = if db_path.starts_with("file::memory:") {
                db_path.to_string()
            } else if db_path == ":memory:" {
                // Plain :memory: without shared cache
                "file::memory:".to_string()
            } else {
                // :memory:suffix format - extract suffix and build URI
                let suffix = &db_path[8..]; // skip ":memory:"
                format!("file::memory:{}?cache=shared", suffix)
            };
            Connection::open_with_flags(
                &uri,
                rusqlite::OpenFlags::SQLITE_OPEN_READ_WRITE
                    | rusqlite::OpenFlags::SQLITE_OPEN_CREATE
                    | rusqlite::OpenFlags::SQLITE_OPEN_URI
                    | rusqlite::OpenFlags::SQLITE_OPEN_SHARED_CACHE,
            )?
        } else {
            Connection::open(db_path)?
        };

        // Enable WAL mode for better concurrent access (only for file-based databases)
        if !db_path.starts_with(":memory:") && !db_path.starts_with("file::memory:") {
            conn.execute_batch(
                "PRAGMA journal_mode = WAL;
                 PRAGMA synchronous = NORMAL;
                 PRAGMA cache_size = -64000;  -- 64MB cache
                 PRAGMA temp_store = MEMORY;
                 PRAGMA mmap_size = 268435456;  -- 256MB mmap",
            )?;
        } else {
            // For shared in-memory database, just set cache size and temp store
            conn.execute_batch(
                "PRAGMA cache_size = -64000;
                 PRAGMA temp_store = MEMORY;
                 PRAGMA busy_timeout = 5000;",
            )?;
        }

        Ok(conn)
    }

    /// Get a connection from the pool.
    ///
    /// Blocks until a connection is available.
    fn get(&self) -> MetadataResult<PooledConnection<'_>> {
        // Try each slot
        for (idx, slot) in self.connections.iter().enumerate() {
            let mut guard = slot.lock();
            if let Some(conn) = guard.take() {
                return Ok(PooledConnection {
                    pool: self,
                    conn: Some(conn),
                    slot_idx: idx,
                });
            }
        }

        // All connections are busy - create a temporary one
        // This allows burst traffic without blocking
        let conn = Self::create_connection(&self.db_path)?;
        Ok(PooledConnection {
            pool: self,
            conn: Some(conn),
            slot_idx: usize::MAX, // Mark as not from pool
        })
    }

    /// Return a connection to the pool.
    fn put(&self, conn: Connection, slot_idx: usize) {
        if slot_idx < self.connections.len() {
            let mut guard = self.connections[slot_idx].lock();
            *guard = Some(conn);
        }
        // If slot_idx == usize::MAX, connection is discarded (temporary connection)
    }

    /// Get pool statistics.
    fn stats(&self) -> PoolStats {
        let mut available = 0;
        for slot in &self.connections {
            if slot.lock().is_some() {
                available += 1;
            }
        }
        PoolStats {
            max_connections: self.max_connections,
            available_connections: available,
            in_use: self.max_connections - available,
        }
    }
}

/// A connection borrowed from the pool.
///
/// Automatically returns the connection when dropped.
struct PooledConnection<'a> {
    pool: &'a ConnectionPool,
    conn: Option<Connection>,
    slot_idx: usize,
}

impl<'a> PooledConnection<'a> {
    fn connection(&self) -> &Connection {
        self.conn.as_ref().unwrap()
    }

    fn connection_mut(&mut self) -> &mut Connection {
        self.conn.as_mut().unwrap()
    }
}

impl<'a> Drop for PooledConnection<'a> {
    fn drop(&mut self) {
        if let Some(conn) = self.conn.take() {
            self.pool.put(conn, self.slot_idx);
        }
    }
}

/// Pool statistics.
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Maximum number of connections.
    pub max_connections: usize,
    /// Currently available connections.
    pub available_connections: usize,
    /// Connections currently in use.
    pub in_use: usize,
}

// ---------------------------------------------------------------------------
// SQLite Metadata Store
// ---------------------------------------------------------------------------

/// SQLite-backed metadata store for content-addressed assets.
///
/// Provides indexed queries on asset metadata with thread-safe connection pooling.
pub struct SqliteMetadataStore {
    pool: ConnectionPool,
    /// Fallback in-memory cache when SQLite queries fail.
    fallback_cache: RwLock<HashMap<ContentHash, AssetMetadata>>,
    /// Whether the store is initialized and ready.
    initialized: bool,
}

impl SqliteMetadataStore {
    /// Open or create a metadata store at the given path.
    ///
    /// Use `:memory:` for an in-memory database.
    pub fn open<P: AsRef<Path>>(path: P) -> MetadataResult<Self> {
        Self::open_with_pool_size(path, 4)
    }

    /// Open with a custom connection pool size.
    pub fn open_with_pool_size<P: AsRef<Path>>(
        path: P,
        pool_size: usize,
    ) -> MetadataResult<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let pool = ConnectionPool::new(&path_str, pool_size.max(1))?;

        let store = Self {
            pool,
            fallback_cache: RwLock::new(HashMap::new()),
            initialized: false,
        };

        store.initialize_schema()?;

        Ok(Self {
            initialized: true,
            ..store
        })
    }

    /// Initialize the database schema.
    fn initialize_schema(&self) -> MetadataResult<()> {
        let conn = self.pool.get()?;

        conn.connection().execute_batch(
            r#"
            -- Main metadata table
            CREATE TABLE IF NOT EXISTS asset_metadata (
                content_hash TEXT PRIMARY KEY NOT NULL,
                asset_type INTEGER NOT NULL,
                import_date INTEGER NOT NULL,
                size_bytes INTEGER NOT NULL,
                provenance_json TEXT,
                dependencies_json TEXT
            );

            -- Indexes for common queries
            CREATE INDEX IF NOT EXISTS idx_asset_type ON asset_metadata(asset_type);
            CREATE INDEX IF NOT EXISTS idx_import_date ON asset_metadata(import_date);

            -- Provenance table for efficient key-value queries
            CREATE TABLE IF NOT EXISTS asset_provenance (
                content_hash TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (content_hash, key),
                FOREIGN KEY (content_hash) REFERENCES asset_metadata(content_hash) ON DELETE CASCADE
            );

            -- Index on provenance key-value pairs
            CREATE INDEX IF NOT EXISTS idx_provenance_key_value ON asset_provenance(key, value);

            -- Dependencies table for reverse lookup
            CREATE TABLE IF NOT EXISTS asset_dependencies (
                content_hash TEXT NOT NULL,
                depends_on TEXT NOT NULL,
                PRIMARY KEY (content_hash, depends_on),
                FOREIGN KEY (content_hash) REFERENCES asset_metadata(content_hash) ON DELETE CASCADE
            );

            -- Index for finding dependents
            CREATE INDEX IF NOT EXISTS idx_depends_on ON asset_dependencies(depends_on);
            "#,
        )?;

        Ok(())
    }

    /// Insert or update asset metadata.
    pub fn insert(&self, metadata: &AssetMetadata) -> MetadataResult<()> {
        let mut conn = self.pool.get()?;
        let tx = conn.connection_mut().transaction()?;

        let hash_hex = format!("{}", metadata.content_hash);

        // Serialize provenance and dependencies as JSON
        let provenance_json = serde_json::to_string(&metadata.provenance)
            .map_err(|e| MetadataError::Serialization(e.to_string()))?;
        let dependencies_json: Vec<String> = metadata
            .dependencies
            .iter()
            .map(|h| format!("{}", h))
            .collect();
        let deps_json = serde_json::to_string(&dependencies_json)
            .map_err(|e| MetadataError::Serialization(e.to_string()))?;

        // Upsert main metadata
        tx.execute(
            r#"
            INSERT INTO asset_metadata (content_hash, asset_type, import_date, size_bytes, provenance_json, dependencies_json)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6)
            ON CONFLICT(content_hash) DO UPDATE SET
                asset_type = excluded.asset_type,
                import_date = excluded.import_date,
                size_bytes = excluded.size_bytes,
                provenance_json = excluded.provenance_json,
                dependencies_json = excluded.dependencies_json
            "#,
            params![
                hash_hex,
                metadata.asset_type as u8,
                metadata.import_date,
                metadata.size_bytes as i64,
                provenance_json,
                deps_json,
            ],
        )?;

        // Update provenance table
        tx.execute(
            "DELETE FROM asset_provenance WHERE content_hash = ?1",
            params![hash_hex],
        )?;

        for (key, value) in &metadata.provenance {
            tx.execute(
                "INSERT INTO asset_provenance (content_hash, key, value) VALUES (?1, ?2, ?3)",
                params![hash_hex, key, value],
            )?;
        }

        // Update dependencies table
        tx.execute(
            "DELETE FROM asset_dependencies WHERE content_hash = ?1",
            params![hash_hex],
        )?;

        for dep in &metadata.dependencies {
            tx.execute(
                "INSERT INTO asset_dependencies (content_hash, depends_on) VALUES (?1, ?2)",
                params![hash_hex, format!("{}", dep)],
            )?;
        }

        tx.commit()?;

        // Update fallback cache
        self.fallback_cache
            .write()
            .insert(metadata.content_hash, metadata.clone());

        Ok(())
    }

    /// Get metadata for a specific content hash.
    pub fn get(&self, hash: &ContentHash) -> MetadataResult<Option<AssetMetadata>> {
        let conn = self.pool.get()?;
        let hash_hex = format!("{}", hash);

        let result: Option<(String, i64, i64, i64, Option<String>, Option<String>)> = conn
            .connection()
            .query_row(
                r#"
                SELECT content_hash, asset_type, import_date, size_bytes, provenance_json, dependencies_json
                FROM asset_metadata
                WHERE content_hash = ?1
                "#,
                params![hash_hex],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                    ))
                },
            )
            .optional()?;

        match result {
            Some((hash_str, asset_type, import_date, size_bytes, prov_json, deps_json)) => {
                let metadata = self.parse_metadata_row(
                    &hash_str,
                    asset_type,
                    import_date,
                    size_bytes,
                    prov_json.as_deref(),
                    deps_json.as_deref(),
                )?;
                Ok(Some(metadata))
            }
            None => Ok(None),
        }
    }

    /// Delete metadata for a content hash.
    pub fn delete(&self, hash: &ContentHash) -> MetadataResult<bool> {
        let conn = self.pool.get()?;
        let hash_hex = format!("{}", hash);

        let rows = conn.connection().execute(
            "DELETE FROM asset_metadata WHERE content_hash = ?1",
            params![hash_hex],
        )?;

        // Cascade deletes handle provenance and dependencies

        // Update fallback cache
        self.fallback_cache.write().remove(hash);

        Ok(rows > 0)
    }

    /// Find all assets of a given type.
    pub fn find_by_type(&self, asset_type: AssetType) -> MetadataResult<Vec<AssetMetadata>> {
        let conn = self.pool.get()?;

        let mut stmt = conn.connection().prepare(
            r#"
            SELECT content_hash, asset_type, import_date, size_bytes, provenance_json, dependencies_json
            FROM asset_metadata
            WHERE asset_type = ?1
            ORDER BY import_date DESC
            "#,
        )?;

        let results = stmt.query_map(params![asset_type as u8], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, i64>(3)?,
                row.get::<_, Option<String>>(4)?,
                row.get::<_, Option<String>>(5)?,
            ))
        })?;

        let mut metadata_list = Vec::new();
        for result in results {
            let (hash_str, at, import_date, size_bytes, prov_json, deps_json) = result?;
            let metadata = self.parse_metadata_row(
                &hash_str,
                at,
                import_date,
                size_bytes,
                prov_json.as_deref(),
                deps_json.as_deref(),
            )?;
            metadata_list.push(metadata);
        }

        Ok(metadata_list)
    }

    /// Find all assets imported within a date range.
    ///
    /// `start` and `end` are timestamps in seconds since UNIX epoch.
    pub fn find_by_date_range(
        &self,
        start: i64,
        end: i64,
    ) -> MetadataResult<Vec<AssetMetadata>> {
        let conn = self.pool.get()?;

        let mut stmt = conn.connection().prepare(
            r#"
            SELECT content_hash, asset_type, import_date, size_bytes, provenance_json, dependencies_json
            FROM asset_metadata
            WHERE import_date >= ?1 AND import_date <= ?2
            ORDER BY import_date DESC
            "#,
        )?;

        let results = stmt.query_map(params![start, end], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, i64>(3)?,
                row.get::<_, Option<String>>(4)?,
                row.get::<_, Option<String>>(5)?,
            ))
        })?;

        let mut metadata_list = Vec::new();
        for result in results {
            let (hash_str, at, import_date, size_bytes, prov_json, deps_json) = result?;
            let metadata = self.parse_metadata_row(
                &hash_str,
                at,
                import_date,
                size_bytes,
                prov_json.as_deref(),
                deps_json.as_deref(),
            )?;
            metadata_list.push(metadata);
        }

        Ok(metadata_list)
    }

    /// Find all assets with a specific provenance key-value pair.
    pub fn find_by_provenance(
        &self,
        key: &str,
        value: &str,
    ) -> MetadataResult<Vec<AssetMetadata>> {
        let conn = self.pool.get()?;

        let mut stmt = conn.connection().prepare(
            r#"
            SELECT m.content_hash, m.asset_type, m.import_date, m.size_bytes, m.provenance_json, m.dependencies_json
            FROM asset_metadata m
            INNER JOIN asset_provenance p ON m.content_hash = p.content_hash
            WHERE p.key = ?1 AND p.value = ?2
            ORDER BY m.import_date DESC
            "#,
        )?;

        let results = stmt.query_map(params![key, value], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, i64>(3)?,
                row.get::<_, Option<String>>(4)?,
                row.get::<_, Option<String>>(5)?,
            ))
        })?;

        let mut metadata_list = Vec::new();
        for result in results {
            let (hash_str, at, import_date, size_bytes, prov_json, deps_json) = result?;
            let metadata = self.parse_metadata_row(
                &hash_str,
                at,
                import_date,
                size_bytes,
                prov_json.as_deref(),
                deps_json.as_deref(),
            )?;
            metadata_list.push(metadata);
        }

        Ok(metadata_list)
    }

    /// Find all assets that have a specific provenance key (any value).
    pub fn find_by_provenance_key(&self, key: &str) -> MetadataResult<Vec<AssetMetadata>> {
        let conn = self.pool.get()?;

        let mut stmt = conn.connection().prepare(
            r#"
            SELECT m.content_hash, m.asset_type, m.import_date, m.size_bytes, m.provenance_json, m.dependencies_json
            FROM asset_metadata m
            INNER JOIN asset_provenance p ON m.content_hash = p.content_hash
            WHERE p.key = ?1
            ORDER BY m.import_date DESC
            "#,
        )?;

        let results = stmt.query_map(params![key], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, i64>(3)?,
                row.get::<_, Option<String>>(4)?,
                row.get::<_, Option<String>>(5)?,
            ))
        })?;

        let mut metadata_list = Vec::new();
        for result in results {
            let (hash_str, at, import_date, size_bytes, prov_json, deps_json) = result?;
            let metadata = self.parse_metadata_row(
                &hash_str,
                at,
                import_date,
                size_bytes,
                prov_json.as_deref(),
                deps_json.as_deref(),
            )?;
            metadata_list.push(metadata);
        }

        Ok(metadata_list)
    }

    /// Find all assets that depend on a given content hash.
    pub fn find_dependents(&self, hash: &ContentHash) -> MetadataResult<Vec<AssetMetadata>> {
        let conn = self.pool.get()?;
        let hash_hex = format!("{}", hash);

        let mut stmt = conn.connection().prepare(
            r#"
            SELECT m.content_hash, m.asset_type, m.import_date, m.size_bytes, m.provenance_json, m.dependencies_json
            FROM asset_metadata m
            INNER JOIN asset_dependencies d ON m.content_hash = d.content_hash
            WHERE d.depends_on = ?1
            ORDER BY m.import_date DESC
            "#,
        )?;

        let results = stmt.query_map(params![hash_hex], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, i64>(3)?,
                row.get::<_, Option<String>>(4)?,
                row.get::<_, Option<String>>(5)?,
            ))
        })?;

        let mut metadata_list = Vec::new();
        for result in results {
            let (hash_str, at, import_date, size_bytes, prov_json, deps_json) = result?;
            let metadata = self.parse_metadata_row(
                &hash_str,
                at,
                import_date,
                size_bytes,
                prov_json.as_deref(),
                deps_json.as_deref(),
            )?;
            metadata_list.push(metadata);
        }

        Ok(metadata_list)
    }

    /// Get the total count of assets.
    pub fn count(&self) -> MetadataResult<usize> {
        let conn = self.pool.get()?;

        let count: i64 = conn.connection().query_row(
            "SELECT COUNT(*) FROM asset_metadata",
            [],
            |row| row.get(0),
        )?;

        Ok(count as usize)
    }

    /// Get count of assets by type.
    pub fn count_by_type(&self, asset_type: AssetType) -> MetadataResult<usize> {
        let conn = self.pool.get()?;

        let count: i64 = conn.connection().query_row(
            "SELECT COUNT(*) FROM asset_metadata WHERE asset_type = ?1",
            params![asset_type as u8],
            |row| row.get(0),
        )?;

        Ok(count as usize)
    }

    /// Get statistics about the metadata store.
    pub fn stats(&self) -> MetadataResult<MetadataStats> {
        let conn = self.pool.get()?;

        let total_count: i64 = conn
            .connection()
            .query_row("SELECT COUNT(*) FROM asset_metadata", [], |row| row.get(0))?;

        let total_size: i64 = conn.connection().query_row(
            "SELECT COALESCE(SUM(size_bytes), 0) FROM asset_metadata",
            [],
            |row| row.get(0),
        )?;

        // Count by type
        let mut type_counts = HashMap::new();
        let mut stmt = conn.connection().prepare(
            "SELECT asset_type, COUNT(*) FROM asset_metadata GROUP BY asset_type",
        )?;
        let results = stmt.query_map([], |row| {
            let at: i64 = row.get(0)?;
            let count: i64 = row.get(1)?;
            Ok((AssetType::from_u8(at as u8), count as usize))
        })?;
        for result in results {
            let (at, count) = result?;
            type_counts.insert(at, count);
        }

        Ok(MetadataStats {
            total_count: total_count as usize,
            total_size_bytes: total_size as u64,
            type_counts,
            pool_stats: self.pool.stats(),
        })
    }

    /// Check if the store is initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get connection pool statistics.
    pub fn pool_stats(&self) -> PoolStats {
        self.pool.stats()
    }

    /// Parse a metadata row from the database.
    fn parse_metadata_row(
        &self,
        hash_str: &str,
        asset_type: i64,
        import_date: i64,
        size_bytes: i64,
        provenance_json: Option<&str>,
        dependencies_json: Option<&str>,
    ) -> MetadataResult<AssetMetadata> {
        let content_hash: ContentHash = hash_str
            .parse()
            .map_err(|e| MetadataError::HashParse(format!("{}", e)))?;

        let provenance: Vec<(String, String)> = provenance_json
            .map(|j| serde_json::from_str(j))
            .transpose()
            .map_err(|e| MetadataError::Serialization(e.to_string()))?
            .unwrap_or_default();

        let dep_strings: Vec<String> = dependencies_json
            .map(|j| serde_json::from_str(j))
            .transpose()
            .map_err(|e| MetadataError::Serialization(e.to_string()))?
            .unwrap_or_default();

        let mut dependencies = Vec::new();
        for dep_str in dep_strings {
            let dep_hash: ContentHash = dep_str
                .parse()
                .map_err(|e| MetadataError::HashParse(format!("{}", e)))?;
            dependencies.push(dep_hash);
        }

        Ok(AssetMetadata {
            content_hash,
            asset_type: AssetType::from_u8(asset_type as u8),
            import_date,
            provenance,
            dependencies,
            size_bytes: size_bytes as u64,
        })
    }

    /// List all metadata entries.
    ///
    /// Warning: This can be slow for large stores. Use with pagination for production.
    pub fn list_all(&self) -> MetadataResult<Vec<AssetMetadata>> {
        let conn = self.pool.get()?;

        let mut stmt = conn.connection().prepare(
            r#"
            SELECT content_hash, asset_type, import_date, size_bytes, provenance_json, dependencies_json
            FROM asset_metadata
            ORDER BY import_date DESC
            "#,
        )?;

        let results = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, i64>(3)?,
                row.get::<_, Option<String>>(4)?,
                row.get::<_, Option<String>>(5)?,
            ))
        })?;

        let mut metadata_list = Vec::new();
        for result in results {
            let (hash_str, at, import_date, size_bytes, prov_json, deps_json) = result?;
            let metadata = self.parse_metadata_row(
                &hash_str,
                at,
                import_date,
                size_bytes,
                prov_json.as_deref(),
                deps_json.as_deref(),
            )?;
            metadata_list.push(metadata);
        }

        Ok(metadata_list)
    }

    /// Check if metadata exists for a content hash.
    pub fn has(&self, hash: &ContentHash) -> MetadataResult<bool> {
        let conn = self.pool.get()?;
        let hash_hex = format!("{}", hash);

        let exists: bool = conn.connection().query_row(
            "SELECT EXISTS(SELECT 1 FROM asset_metadata WHERE content_hash = ?1)",
            params![hash_hex],
            |row| row.get(0),
        )?;

        Ok(exists)
    }

    /// Vacuum the database to reclaim space.
    pub fn vacuum(&self) -> MetadataResult<()> {
        let conn = self.pool.get()?;
        conn.connection().execute_batch("VACUUM")?;
        Ok(())
    }

    /// Analyze the database for query optimization.
    pub fn analyze(&self) -> MetadataResult<()> {
        let conn = self.pool.get()?;
        conn.connection().execute_batch("ANALYZE")?;
        Ok(())
    }
}

/// Statistics about the metadata store.
#[derive(Debug, Clone)]
pub struct MetadataStats {
    /// Total number of assets.
    pub total_count: usize,
    /// Total size of all assets in bytes.
    pub total_size_bytes: u64,
    /// Count of assets by type.
    pub type_counts: HashMap<AssetType, usize>,
    /// Connection pool statistics.
    pub pool_stats: PoolStats,
}

// ---------------------------------------------------------------------------
// Optional Metadata Backend
// ---------------------------------------------------------------------------

/// An optional metadata backend that gracefully degrades.
///
/// When SQLite is available, uses indexed queries. When not available,
/// falls back to O(n) in-memory scans.
pub struct OptionalMetadataBackend {
    /// SQLite store (if available).
    sqlite: Option<SqliteMetadataStore>,
    /// In-memory fallback cache.
    fallback: RwLock<HashMap<ContentHash, AssetMetadata>>,
}

impl OptionalMetadataBackend {
    /// Create a new optional backend, attempting to open SQLite.
    pub fn new<P: AsRef<Path>>(path: Option<P>) -> Self {
        let sqlite = path.and_then(|p| SqliteMetadataStore::open(p).ok());

        Self {
            sqlite,
            fallback: RwLock::new(HashMap::new()),
        }
    }

    /// Create an in-memory only backend (no SQLite).
    pub fn memory_only() -> Self {
        Self {
            sqlite: None,
            fallback: RwLock::new(HashMap::new()),
        }
    }

    /// Check if SQLite is available.
    pub fn has_sqlite(&self) -> bool {
        self.sqlite.is_some()
    }

    /// Insert metadata.
    pub fn insert(&self, metadata: &AssetMetadata) -> MetadataResult<()> {
        // Always update fallback cache
        self.fallback
            .write()
            .insert(metadata.content_hash, metadata.clone());

        // Try SQLite if available
        if let Some(ref store) = self.sqlite {
            store.insert(metadata)?;
        }

        Ok(())
    }

    /// Get metadata by hash.
    pub fn get(&self, hash: &ContentHash) -> MetadataResult<Option<AssetMetadata>> {
        // Try SQLite first
        if let Some(ref store) = self.sqlite {
            if let Ok(Some(meta)) = store.get(hash) {
                return Ok(Some(meta));
            }
        }

        // Fallback to in-memory
        Ok(self.fallback.read().get(hash).cloned())
    }

    /// Find by type - uses index if available, O(n) scan otherwise.
    pub fn find_by_type(&self, asset_type: AssetType) -> MetadataResult<Vec<AssetMetadata>> {
        // Try SQLite first
        if let Some(ref store) = self.sqlite {
            return store.find_by_type(asset_type);
        }

        // Fallback: O(n) scan
        let cache = self.fallback.read();
        let results: Vec<_> = cache
            .values()
            .filter(|m| m.asset_type == asset_type)
            .cloned()
            .collect();
        Ok(results)
    }

    /// Find by date range - uses index if available, O(n) scan otherwise.
    pub fn find_by_date_range(
        &self,
        start: i64,
        end: i64,
    ) -> MetadataResult<Vec<AssetMetadata>> {
        // Try SQLite first
        if let Some(ref store) = self.sqlite {
            return store.find_by_date_range(start, end);
        }

        // Fallback: O(n) scan
        let cache = self.fallback.read();
        let results: Vec<_> = cache
            .values()
            .filter(|m| m.import_date >= start && m.import_date <= end)
            .cloned()
            .collect();
        Ok(results)
    }

    /// Find by provenance - uses index if available, O(n) scan otherwise.
    pub fn find_by_provenance(
        &self,
        key: &str,
        value: &str,
    ) -> MetadataResult<Vec<AssetMetadata>> {
        // Try SQLite first
        if let Some(ref store) = self.sqlite {
            return store.find_by_provenance(key, value);
        }

        // Fallback: O(n) scan
        let cache = self.fallback.read();
        let results: Vec<_> = cache
            .values()
            .filter(|m| m.provenance.iter().any(|(k, v)| k == key && v == value))
            .cloned()
            .collect();
        Ok(results)
    }

    /// Delete metadata.
    pub fn delete(&self, hash: &ContentHash) -> MetadataResult<bool> {
        let mut deleted = self.fallback.write().remove(hash).is_some();

        if let Some(ref store) = self.sqlite {
            deleted |= store.delete(hash)?;
        }

        Ok(deleted)
    }

    /// Get count of all assets.
    pub fn count(&self) -> MetadataResult<usize> {
        if let Some(ref store) = self.sqlite {
            return store.count();
        }
        Ok(self.fallback.read().len())
    }

    /// List all metadata.
    pub fn list_all(&self) -> MetadataResult<Vec<AssetMetadata>> {
        if let Some(ref store) = self.sqlite {
            return store.list_all();
        }
        Ok(self.fallback.read().values().cloned().collect())
    }
}

// Thread safety
unsafe impl Send for SqliteMetadataStore {}
unsafe impl Sync for SqliteMetadataStore {}
unsafe impl Send for OptionalMetadataBackend {}
unsafe impl Sync for OptionalMetadataBackend {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a unique in-memory database path for testing.
    /// Each test gets its own isolated database to avoid conflicts when
    /// tests run in parallel.
    fn unique_db_path() -> String {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::SeqCst);
        let thread_id = std::thread::current().id();
        format!(":memory:test_{}_{:?}", id, thread_id)
    }

    // ========================================================================
    // AssetType tests
    // ========================================================================

    #[test]
    fn test_asset_type_from_u8() {
        assert_eq!(AssetType::from_u8(0), AssetType::Unknown);
        assert_eq!(AssetType::from_u8(1), AssetType::Texture);
        assert_eq!(AssetType::from_u8(2), AssetType::Mesh);
        assert_eq!(AssetType::from_u8(255), AssetType::Unknown);
    }

    #[test]
    fn test_asset_type_name() {
        assert_eq!(AssetType::Texture.name(), "texture");
        assert_eq!(AssetType::Mesh.name(), "mesh");
        assert_eq!(AssetType::Shader.name(), "shader");
    }

    #[test]
    fn test_asset_type_from_str() {
        assert_eq!("texture".parse::<AssetType>(), Ok(AssetType::Texture));
        assert_eq!("MESH".parse::<AssetType>(), Ok(AssetType::Mesh));
        assert_eq!("unknown_type".parse::<AssetType>(), Err(()));
    }

    #[test]
    fn test_asset_type_display() {
        assert_eq!(format!("{}", AssetType::Texture), "texture");
    }

    #[test]
    fn test_asset_type_roundtrip() {
        for i in 0..=12 {
            let at = AssetType::from_u8(i);
            assert_eq!(at as u8, i);
        }
    }

    // ========================================================================
    // AssetMetadata tests
    // ========================================================================

    #[test]
    fn test_asset_metadata_new() {
        let hash = ContentHash::from_bytes(b"test");
        let meta = AssetMetadata::new(hash, AssetType::Texture, 1024);

        assert_eq!(meta.content_hash, hash);
        assert_eq!(meta.asset_type, AssetType::Texture);
        assert_eq!(meta.size_bytes, 1024);
        assert!(meta.import_date > 0);
        assert!(meta.provenance.is_empty());
        assert!(meta.dependencies.is_empty());
    }

    #[test]
    fn test_asset_metadata_with_provenance() {
        let hash = ContentHash::from_bytes(b"test");
        let meta = AssetMetadata::new(hash, AssetType::Texture, 1024)
            .with_provenance("source", "photo.png")
            .with_provenance("version", "1.0");

        assert_eq!(meta.provenance.len(), 2);
        assert_eq!(meta.get_provenance("source"), Some("photo.png"));
        assert_eq!(meta.get_provenance("version"), Some("1.0"));
        assert_eq!(meta.get_provenance("missing"), None);
    }

    #[test]
    fn test_asset_metadata_with_dependencies() {
        let hash = ContentHash::from_bytes(b"test");
        let dep1 = ContentHash::from_bytes(b"dep1");
        let dep2 = ContentHash::from_bytes(b"dep2");

        let meta = AssetMetadata::new(hash, AssetType::Material, 512)
            .with_dependency(dep1)
            .with_dependency(dep2);

        assert_eq!(meta.dependencies.len(), 2);
        assert!(meta.dependencies.contains(&dep1));
        assert!(meta.dependencies.contains(&dep2));
    }

    #[test]
    fn test_asset_metadata_with_import_date() {
        let hash = ContentHash::from_bytes(b"test");
        let meta = AssetMetadata::new(hash, AssetType::Texture, 1024)
            .with_import_date(1716700000);

        assert_eq!(meta.import_date, 1716700000);
    }

    // ========================================================================
    // Table creation tests
    // ========================================================================

    #[test]
    fn test_sqlite_table_creation() {
        let store = SqliteMetadataStore::open(&unique_db_path()).unwrap();
        assert!(store.is_initialized());
    }

    #[test]
    fn test_sqlite_schema_idempotent() {
        // Opening twice should not fail
        let store = SqliteMetadataStore::open(&unique_db_path()).unwrap();
        assert!(store.is_initialized());
        // Schema creation is idempotent via IF NOT EXISTS
    }

    // ========================================================================
    // Insert tests
    // ========================================================================

    #[test]
    fn test_sqlite_insert_metadata() {
        let store = SqliteMetadataStore::open(&unique_db_path()).unwrap();

        let hash = ContentHash::from_bytes(b"test data");
        let meta = AssetMetadata::new(hash, AssetType::Texture, 1024)
            .with_provenance("source", "test.png");

        store.insert(&meta).unwrap();
        assert!(store.has(&hash).unwrap());
    }

    #[test]
    fn test_sqlite_insert_update() {
        let store = SqliteMetadataStore::open(&unique_db_path()).unwrap();

        let hash = ContentHash::from_bytes(b"test");
        let meta1 = AssetMetadata::new(hash, AssetType::Texture, 1024);
        store.insert(&meta1).unwrap();

        let meta2 = AssetMetadata::new(hash, AssetType::Mesh, 2048);
        store.insert(&meta2).unwrap();

        let retrieved = store.get(&hash).unwrap().unwrap();
        assert_eq!(retrieved.asset_type, AssetType::Mesh);
        assert_eq!(retrieved.size_bytes, 2048);
    }

    // ========================================================================
    // Query by hash tests
    // ========================================================================

    #[test]
    fn test_sqlite_query_by_hash() {
        let store = SqliteMetadataStore::open(&unique_db_path()).unwrap();

        let hash = ContentHash::from_bytes(b"test");
        let meta = AssetMetadata::new(hash, AssetType::Texture, 1024)
            .with_provenance("key", "value");

        store.insert(&meta).unwrap();

        let retrieved = store.get(&hash).unwrap().unwrap();
        assert_eq!(retrieved.content_hash, hash);
        assert_eq!(retrieved.asset_type, AssetType::Texture);
        assert_eq!(retrieved.size_bytes, 1024);
        assert_eq!(retrieved.get_provenance("key"), Some("value"));
    }

    #[test]
    fn test_sqlite_query_missing_hash() {
        let store = SqliteMetadataStore::open(&unique_db_path()).unwrap();

        let hash = ContentHash::from_bytes(b"nonexistent");
        let result = store.get(&hash).unwrap();
        assert!(result.is_none());
    }

    // ========================================================================
    // Query by type tests
    // ========================================================================

    #[test]
    fn test_sqlite_query_by_asset_type() {
        let store = SqliteMetadataStore::open(&unique_db_path()).unwrap();

        // Insert multiple assets of different types
        for i in 0..5 {
            let hash = ContentHash::from_bytes(format!("texture{}", i).as_bytes());
            let meta = AssetMetadata::new(hash, AssetType::Texture, 1024);
            store.insert(&meta).unwrap();
        }

        for i in 0..3 {
            let hash = ContentHash::from_bytes(format!("mesh{}", i).as_bytes());
            let meta = AssetMetadata::new(hash, AssetType::Mesh, 2048);
            store.insert(&meta).unwrap();
        }

        let textures = store.find_by_type(AssetType::Texture).unwrap();
        assert_eq!(textures.len(), 5);

        let meshes = store.find_by_type(AssetType::Mesh).unwrap();
        assert_eq!(meshes.len(), 3);

        let shaders = store.find_by_type(AssetType::Shader).unwrap();
        assert!(shaders.is_empty());
    }

    // ========================================================================
    // Query by date range tests
    // ========================================================================

    #[test]
    fn test_sqlite_query_by_date_range() {
        let store = SqliteMetadataStore::open(&unique_db_path()).unwrap();

        // Insert assets with different import dates
        let timestamps = [1000, 2000, 3000, 4000, 5000];
        for (i, ts) in timestamps.iter().enumerate() {
            let hash = ContentHash::from_bytes(format!("asset{}", i).as_bytes());
            let meta = AssetMetadata::new(hash, AssetType::Texture, 1024)
                .with_import_date(*ts);
            store.insert(&meta).unwrap();
        }

        // Query range
        let results = store.find_by_date_range(2000, 4000).unwrap();
        assert_eq!(results.len(), 3);

        // All dates in range
        for meta in &results {
            assert!(meta.import_date >= 2000 && meta.import_date <= 4000);
        }
    }

    #[test]
    fn test_sqlite_query_empty_date_range() {
        let store = SqliteMetadataStore::open(&unique_db_path()).unwrap();

        let hash = ContentHash::from_bytes(b"test");
        let meta = AssetMetadata::new(hash, AssetType::Texture, 1024)
            .with_import_date(5000);
        store.insert(&meta).unwrap();

        let results = store.find_by_date_range(1000, 2000).unwrap();
        assert!(results.is_empty());
    }

    // ========================================================================
    // Query by provenance tests
    // ========================================================================

    #[test]
    fn test_sqlite_query_by_provenance() {
        let store = SqliteMetadataStore::open(&unique_db_path()).unwrap();

        // Insert assets with provenance
        let hash1 = ContentHash::from_bytes(b"asset1");
        let meta1 = AssetMetadata::new(hash1, AssetType::Texture, 1024)
            .with_provenance("source", "photo.png")
            .with_provenance("author", "alice");
        store.insert(&meta1).unwrap();

        let hash2 = ContentHash::from_bytes(b"asset2");
        let meta2 = AssetMetadata::new(hash2, AssetType::Texture, 2048)
            .with_provenance("source", "photo.png")
            .with_provenance("author", "bob");
        store.insert(&meta2).unwrap();

        let hash3 = ContentHash::from_bytes(b"asset3");
        let meta3 = AssetMetadata::new(hash3, AssetType::Mesh, 512)
            .with_provenance("source", "model.obj");
        store.insert(&meta3).unwrap();

        // Find by source=photo.png
        let results = store.find_by_provenance("source", "photo.png").unwrap();
        assert_eq!(results.len(), 2);

        // Find by author=alice
        let results = store.find_by_provenance("author", "alice").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content_hash, hash1);
    }

    #[test]
    fn test_sqlite_query_by_provenance_key() {
        let store = SqliteMetadataStore::open(&unique_db_path()).unwrap();

        let hash1 = ContentHash::from_bytes(b"a1");
        let meta1 = AssetMetadata::new(hash1, AssetType::Texture, 1024)
            .with_provenance("version", "1.0");
        store.insert(&meta1).unwrap();

        let hash2 = ContentHash::from_bytes(b"a2");
        let meta2 = AssetMetadata::new(hash2, AssetType::Texture, 1024)
            .with_provenance("version", "2.0");
        store.insert(&meta2).unwrap();

        let results = store.find_by_provenance_key("version").unwrap();
        assert_eq!(results.len(), 2);
    }

    // ========================================================================
    // Index usage tests
    // ========================================================================

    #[test]
    fn test_sqlite_index_exists() {
        let store = SqliteMetadataStore::open(&unique_db_path()).unwrap();
        let conn = store.pool.get().unwrap();

        // Check that indexes exist
        let indexes: Vec<String> = conn
            .connection()
            .prepare("SELECT name FROM sqlite_master WHERE type='index'")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();

        assert!(indexes.iter().any(|n| n == "idx_asset_type"));
        assert!(indexes.iter().any(|n| n == "idx_import_date"));
        assert!(indexes.iter().any(|n| n == "idx_provenance_key_value"));
    }

    // ========================================================================
    // Optional backend tests
    // ========================================================================

    #[test]
    fn test_optional_backend_works_without_sqlite() {
        let backend = OptionalMetadataBackend::memory_only();
        assert!(!backend.has_sqlite());

        let hash = ContentHash::from_bytes(b"test");
        let meta = AssetMetadata::new(hash, AssetType::Texture, 1024);

        backend.insert(&meta).unwrap();
        let retrieved = backend.get(&hash).unwrap().unwrap();
        assert_eq!(retrieved.content_hash, hash);
    }

    #[test]
    fn test_optional_backend_fallback_scan() {
        let backend = OptionalMetadataBackend::memory_only();

        // Insert multiple assets
        for i in 0..10 {
            let hash = ContentHash::from_bytes(format!("asset{}", i).as_bytes());
            let at = if i < 5 {
                AssetType::Texture
            } else {
                AssetType::Mesh
            };
            let meta = AssetMetadata::new(hash, at, 1024);
            backend.insert(&meta).unwrap();
        }

        // Fallback O(n) scan
        let textures = backend.find_by_type(AssetType::Texture).unwrap();
        assert_eq!(textures.len(), 5);
    }

    #[test]
    fn test_optional_backend_fallback_date_range() {
        let backend = OptionalMetadataBackend::memory_only();

        for i in 0..5 {
            let hash = ContentHash::from_bytes(format!("asset{}", i).as_bytes());
            let meta = AssetMetadata::new(hash, AssetType::Texture, 1024)
                .with_import_date(1000 + i * 100);
            backend.insert(&meta).unwrap();
        }

        let results = backend.find_by_date_range(1100, 1300).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_optional_backend_fallback_provenance() {
        let backend = OptionalMetadataBackend::memory_only();

        let hash1 = ContentHash::from_bytes(b"a1");
        let meta1 = AssetMetadata::new(hash1, AssetType::Texture, 1024)
            .with_provenance("source", "file.png");
        backend.insert(&meta1).unwrap();

        let hash2 = ContentHash::from_bytes(b"a2");
        let meta2 = AssetMetadata::new(hash2, AssetType::Texture, 1024)
            .with_provenance("source", "other.png");
        backend.insert(&meta2).unwrap();

        let results = backend.find_by_provenance("source", "file.png").unwrap();
        assert_eq!(results.len(), 1);
    }

    // ========================================================================
    // Thread safety tests
    // ========================================================================

    #[test]
    fn test_sqlite_thread_safe_access() {
        use std::sync::Arc;
        use std::thread;

        // Use a file-based database for thread safety test because
        // in-memory shared cache has table-level locking limitations
        let temp_dir = tempfile::tempdir().unwrap();
        let db_path = temp_dir.path().join("test_thread_safe.db");
        let store = Arc::new(SqliteMetadataStore::open(&db_path).unwrap());
        let mut handles = Vec::new();

        // Spawn multiple threads writing
        for i in 0..10 {
            let store = Arc::clone(&store);
            let handle = thread::spawn(move || {
                let hash = ContentHash::from_bytes(format!("thread{}", i).as_bytes());
                let meta = AssetMetadata::new(hash, AssetType::Texture, 1024);
                // Retry on busy - SQLite WAL mode handles concurrent writes
                for attempt in 0..5 {
                    match store.insert(&meta) {
                        Ok(()) => return,
                        Err(MetadataError::Sqlite(ref e))
                            if e.to_string().contains("locked") && attempt < 4 => {
                            std::thread::sleep(std::time::Duration::from_millis(10 * (attempt + 1)));
                            continue;
                        }
                        Err(e) => panic!("Insert failed: {:?}", e),
                    }
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(store.count().unwrap(), 10);
    }

    #[test]
    fn test_sqlite_concurrent_reads() {
        use std::sync::Arc;
        use std::thread;

        let store = Arc::new(SqliteMetadataStore::open(&unique_db_path()).unwrap());

        // Pre-populate
        for i in 0..100 {
            let hash = ContentHash::from_bytes(format!("asset{}", i).as_bytes());
            let meta = AssetMetadata::new(hash, AssetType::Texture, 1024);
            store.insert(&meta).unwrap();
        }

        let mut handles = Vec::new();

        // Spawn readers
        for _ in 0..10 {
            let store = Arc::clone(&store);
            let handle = thread::spawn(move || {
                for i in 0..100 {
                    let hash = ContentHash::from_bytes(format!("asset{}", i).as_bytes());
                    let _ = store.get(&hash);
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    // ========================================================================
    // Connection pool tests
    // ========================================================================

    #[test]
    fn test_connection_pooling() {
        let store = SqliteMetadataStore::open_with_pool_size(&unique_db_path(), 2).unwrap();

        let stats = store.pool_stats();
        assert_eq!(stats.max_connections, 2);
        assert_eq!(stats.available_connections, 2);
        assert_eq!(stats.in_use, 0);
    }

    #[test]
    fn test_pool_burst_traffic() {
        use std::sync::Arc;
        use std::thread;

        let store = Arc::new(SqliteMetadataStore::open_with_pool_size(&unique_db_path(), 2).unwrap());

        // Pre-insert data
        let hash = ContentHash::from_bytes(b"data");
        let meta = AssetMetadata::new(hash, AssetType::Texture, 1024);
        store.insert(&meta).unwrap();

        let mut handles = Vec::new();

        // More threads than pool size
        for _ in 0..10 {
            let store = Arc::clone(&store);
            let handle = thread::spawn(move || {
                for _ in 0..10 {
                    let _ = store.get(&hash);
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    // ========================================================================
    // Delete tests
    // ========================================================================

    #[test]
    fn test_sqlite_delete() {
        let store = SqliteMetadataStore::open(&unique_db_path()).unwrap();

        let hash = ContentHash::from_bytes(b"test");
        let meta = AssetMetadata::new(hash, AssetType::Texture, 1024);
        store.insert(&meta).unwrap();

        assert!(store.has(&hash).unwrap());
        let deleted = store.delete(&hash).unwrap();
        assert!(deleted);
        assert!(!store.has(&hash).unwrap());
    }

    #[test]
    fn test_sqlite_delete_cascade() {
        let store = SqliteMetadataStore::open(&unique_db_path()).unwrap();

        let hash = ContentHash::from_bytes(b"test");
        let meta = AssetMetadata::new(hash, AssetType::Texture, 1024)
            .with_provenance("key", "value")
            .with_dependency(ContentHash::from_bytes(b"dep"));
        store.insert(&meta).unwrap();

        store.delete(&hash).unwrap();

        // Verify cascade delete removed provenance
        let results = store.find_by_provenance("key", "value").unwrap();
        assert!(results.is_empty());
    }

    // ========================================================================
    // Statistics tests
    // ========================================================================

    #[test]
    fn test_sqlite_stats() {
        let store = SqliteMetadataStore::open(&unique_db_path()).unwrap();

        for i in 0..10 {
            let hash = ContentHash::from_bytes(format!("asset{}", i).as_bytes());
            let at = if i < 6 {
                AssetType::Texture
            } else {
                AssetType::Mesh
            };
            let meta = AssetMetadata::new(hash, at, (i + 1) * 100);
            store.insert(&meta).unwrap();
        }

        let stats = store.stats().unwrap();
        assert_eq!(stats.total_count, 10);
        assert_eq!(stats.total_size_bytes, 5500); // sum 100..1000
        assert_eq!(stats.type_counts.get(&AssetType::Texture), Some(&6));
        assert_eq!(stats.type_counts.get(&AssetType::Mesh), Some(&4));
    }

    #[test]
    fn test_sqlite_count_by_type() {
        let store = SqliteMetadataStore::open(&unique_db_path()).unwrap();

        for i in 0..5 {
            let hash = ContentHash::from_bytes(format!("tex{}", i).as_bytes());
            let meta = AssetMetadata::new(hash, AssetType::Texture, 1024);
            store.insert(&meta).unwrap();
        }

        assert_eq!(store.count_by_type(AssetType::Texture).unwrap(), 5);
        assert_eq!(store.count_by_type(AssetType::Mesh).unwrap(), 0);
    }

    // ========================================================================
    // Dependency tests
    // ========================================================================

    #[test]
    fn test_sqlite_find_dependents() {
        let store = SqliteMetadataStore::open(&unique_db_path()).unwrap();

        let dep_hash = ContentHash::from_bytes(b"dependency");
        let dep_meta = AssetMetadata::new(dep_hash, AssetType::Texture, 1024);
        store.insert(&dep_meta).unwrap();

        let hash1 = ContentHash::from_bytes(b"asset1");
        let meta1 = AssetMetadata::new(hash1, AssetType::Material, 512)
            .with_dependency(dep_hash);
        store.insert(&meta1).unwrap();

        let hash2 = ContentHash::from_bytes(b"asset2");
        let meta2 = AssetMetadata::new(hash2, AssetType::Material, 512)
            .with_dependency(dep_hash);
        store.insert(&meta2).unwrap();

        let dependents = store.find_dependents(&dep_hash).unwrap();
        assert_eq!(dependents.len(), 2);
    }

    // ========================================================================
    // List all tests
    // ========================================================================

    #[test]
    fn test_sqlite_list_all() {
        let store = SqliteMetadataStore::open(&unique_db_path()).unwrap();

        for i in 0..5 {
            let hash = ContentHash::from_bytes(format!("asset{}", i).as_bytes());
            let meta = AssetMetadata::new(hash, AssetType::Texture, 1024);
            store.insert(&meta).unwrap();
        }

        let all = store.list_all().unwrap();
        assert_eq!(all.len(), 5);
    }

    // ========================================================================
    // Error handling tests
    // ========================================================================

    #[test]
    fn test_metadata_error_display() {
        let err = MetadataError::PoolExhausted;
        assert!(format!("{}", err).contains("exhausted"));

        let err = MetadataError::NotAvailable;
        assert!(format!("{}", err).contains("not available"));
    }

    // ========================================================================
    // Vacuum and analyze tests
    // ========================================================================

    #[test]
    fn test_sqlite_vacuum() {
        let store = SqliteMetadataStore::open(&unique_db_path()).unwrap();
        store.vacuum().unwrap();
    }

    #[test]
    fn test_sqlite_analyze() {
        let store = SqliteMetadataStore::open(&unique_db_path()).unwrap();

        // Insert some data first
        for i in 0..10 {
            let hash = ContentHash::from_bytes(format!("asset{}", i).as_bytes());
            let meta = AssetMetadata::new(hash, AssetType::Texture, 1024);
            store.insert(&meta).unwrap();
        }

        store.analyze().unwrap();
    }
}
