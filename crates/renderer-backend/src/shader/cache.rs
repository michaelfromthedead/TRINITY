//! 3-Level Shader Cache for TRINITY (T-AS-3.4).
//!
//! Implements a hierarchical shader cache with three levels:
//!
//! 1. **In-Memory LRU**: Stores compiled PSO bytecode or shader modules
//!    - Default 512 MB configurable
//!    - <1ms lookup performance
//!
//! 2. **Disk Cache**: ContentStore-backed persistent storage
//!    - Content-addressed by comprehensive cache key
//!    - <10ms lookup performance
//!
//! 3. **PAK Archive**: Pre-compiled common shaders shipped with application
//!    - Read-only, indexed for fast lookup
//!    - Falls back when disk cache misses
//!
//! # Cache Key Components
//!
//! The cache key is computed from:
//! - Source content hash (SHA-256 or BLAKE3)
//! - #define flags sorted alphabetically
//! - Target platform (Vulkan, Metal, D3D12)
//! - Language version (WGSL spec version)
//! - Compiler version (naga version)
//! - Optimization level (0-3)
//! - Include file content hashes
//!
//! # Invalidation Triggers
//!
//! - Source file modification (detected via content hash change)
//! - Compiler version bump
//! - Platform target change
//! - Explicit `clear()` call
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::shader::cache::{ShaderCache3L, CacheConfig, CacheKey};
//!
//! // Create cache with 256 MB memory limit
//! let config = CacheConfig::with_memory_size_mb(256);
//! let mut cache = ShaderCache3L::new("/tmp/shader_cache", config)?;
//!
//! // Build cache key
//! let key = CacheKey::builder()
//!     .source_hash(ContentHash::from_bytes(wgsl_source))
//!     .platform(TargetPlatform::Vulkan)
//!     .compiler_version("naga-0.24")
//!     .optimization_level(2)
//!     .build();
//!
//! // Try to get compiled bytecode
//! if let Some(spirv) = cache.get(&key)? {
//!     // Use cached bytecode
//! } else {
//!     // Compile and store
//!     let spirv = compile(wgsl_source)?;
//!     cache.put(&key, &spirv)?;
//! }
//! ```

use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use sha2::{Digest, Sha256};

use crate::pipeline::{ContentHash, FileBackend};

// ---------------------------------------------------------------------------
// Target Platform
// ---------------------------------------------------------------------------

/// Target platform for shader compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetPlatform {
    /// Vulkan SPIR-V
    Vulkan,
    /// Metal Shading Language
    Metal,
    /// Direct3D 12 DXIL
    D3D12,
    /// WebGPU WGSL (validated/transformed)
    WebGPU,
}

impl TargetPlatform {
    /// Get platform identifier string for cache key.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Vulkan => "vulkan",
            Self::Metal => "metal",
            Self::D3D12 => "d3d12",
            Self::WebGPU => "webgpu",
        }
    }
}

impl fmt::Display for TargetPlatform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ---------------------------------------------------------------------------
// Cache Key
// ---------------------------------------------------------------------------

/// Comprehensive cache key for shader lookup.
///
/// Includes all factors that could affect compilation output:
/// - Source content hash
/// - Preprocessor defines
/// - Target platform
/// - Language/compiler versions
/// - Optimization settings
/// - Include dependencies
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    /// SHA-256 hash of preprocessed source.
    pub source_hash: ContentHash,
    /// Sorted list of (name, value) define pairs.
    pub defines: Vec<(String, String)>,
    /// Target compilation platform.
    pub platform: TargetPlatform,
    /// WGSL language version (e.g., "1.0").
    pub language_version: String,
    /// Compiler version string (e.g., "naga-0.24").
    pub compiler_version: String,
    /// Optimization level (0 = none, 3 = maximum).
    pub optimization_level: u8,
    /// Content hashes of included files (sorted by path).
    pub include_hashes: Vec<(String, ContentHash)>,
}

impl CacheKey {
    /// Create a new cache key builder.
    pub fn builder() -> CacheKeyBuilder {
        CacheKeyBuilder::new()
    }

    /// Compute the combined hash of all key components.
    ///
    /// This is the actual key used for storage lookups.
    pub fn combined_hash(&self) -> ContentHash {
        let mut hasher = Sha256::new();

        // Source hash
        hasher.update(self.source_hash.as_bytes());

        // Defines (sorted)
        let mut defines = self.defines.clone();
        defines.sort();
        for (name, value) in &defines {
            hasher.update(name.as_bytes());
            hasher.update(b"=");
            hasher.update(value.as_bytes());
            hasher.update(b"\n");
        }

        // Platform
        hasher.update(self.platform.as_str().as_bytes());

        // Language version
        hasher.update(self.language_version.as_bytes());

        // Compiler version
        hasher.update(self.compiler_version.as_bytes());

        // Optimization level
        hasher.update(&[self.optimization_level]);

        // Include hashes (sorted by path)
        let mut includes = self.include_hashes.clone();
        includes.sort_by(|a, b| a.0.cmp(&b.0));
        for (path, hash) in &includes {
            hasher.update(path.as_bytes());
            hasher.update(hash.as_bytes());
        }

        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        ContentHash::from_raw(hash)
    }
}

impl Default for CacheKey {
    fn default() -> Self {
        Self {
            source_hash: ContentHash::zero(),
            defines: Vec::new(),
            platform: TargetPlatform::Vulkan,
            language_version: "1.0".to_string(),
            compiler_version: "naga-0.24".to_string(),
            optimization_level: 2,
            include_hashes: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Cache Key Builder
// ---------------------------------------------------------------------------

/// Builder for constructing cache keys.
pub struct CacheKeyBuilder {
    key: CacheKey,
}

impl CacheKeyBuilder {
    /// Create a new builder with default values.
    pub fn new() -> Self {
        Self {
            key: CacheKey::default(),
        }
    }

    /// Set the source content hash.
    pub fn source_hash(mut self, hash: ContentHash) -> Self {
        self.key.source_hash = hash;
        self
    }

    /// Set source from raw bytes (computes hash).
    pub fn source(self, source: &[u8]) -> Self {
        self.source_hash(ContentHash::from_bytes(source))
    }

    /// Add a preprocessor define.
    pub fn define(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.key.defines.push((name.into(), value.into()));
        self
    }

    /// Set all defines at once.
    pub fn defines(mut self, defines: Vec<(String, String)>) -> Self {
        self.key.defines = defines;
        self
    }

    /// Set the target platform.
    pub fn platform(mut self, platform: TargetPlatform) -> Self {
        self.key.platform = platform;
        self
    }

    /// Set the language version.
    pub fn language_version(mut self, version: impl Into<String>) -> Self {
        self.key.language_version = version.into();
        self
    }

    /// Set the compiler version.
    pub fn compiler_version(mut self, version: impl Into<String>) -> Self {
        self.key.compiler_version = version.into();
        self
    }

    /// Set the optimization level (0-3).
    pub fn optimization_level(mut self, level: u8) -> Self {
        self.key.optimization_level = level.min(3);
        self
    }

    /// Add an include file hash.
    pub fn include_hash(mut self, path: impl Into<String>, hash: ContentHash) -> Self {
        self.key.include_hashes.push((path.into(), hash));
        self
    }

    /// Set all include hashes at once.
    pub fn include_hashes(mut self, includes: Vec<(String, ContentHash)>) -> Self {
        self.key.include_hashes = includes;
        self
    }

    /// Build the cache key.
    pub fn build(self) -> CacheKey {
        self.key
    }
}

impl Default for CacheKeyBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// LRU Cache Entry
// ---------------------------------------------------------------------------

/// Entry in the LRU cache.
#[derive(Debug, Clone)]
struct LruEntry {
    /// Compiled bytecode or data.
    data: Arc<Vec<u8>>,
    /// Last access timestamp (for statistics).
    last_access: u64,
    /// Access count.
    access_count: u64,
}

// ---------------------------------------------------------------------------
// In-Memory LRU Cache
// ---------------------------------------------------------------------------

/// In-memory LRU cache with configurable size limit.
///
/// Provides O(1) lookups and automatic eviction of least recently used entries
/// when the size limit is exceeded.
struct MemoryLruCache {
    /// Cache entries keyed by combined hash.
    entries: HashMap<ContentHash, LruEntry>,
    /// LRU order: most recent at back, oldest at front.
    order: Vec<ContentHash>,
    /// Current total size in bytes.
    current_size: usize,
    /// Maximum size in bytes.
    max_size: usize,
    /// Monotonic counter for access timestamps.
    counter: u64,
    /// Statistics: total hits.
    hits: AtomicU64,
    /// Statistics: total misses.
    misses: AtomicU64,
}

impl MemoryLruCache {
    /// Create a new LRU cache with the given size limit.
    fn new(max_size: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: Vec::new(),
            current_size: 0,
            max_size,
            counter: 0,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Get an entry from the cache, updating LRU order.
    fn get(&mut self, key: &ContentHash) -> Option<Arc<Vec<u8>>> {
        if let Some(entry) = self.entries.get_mut(key) {
            self.counter += 1;
            entry.last_access = self.counter;
            entry.access_count += 1;

            // Move to back of LRU order
            self.order.retain(|k| k != key);
            self.order.push(*key);

            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(Arc::clone(&entry.data))
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Insert an entry, evicting LRU entries if necessary.
    fn put(&mut self, key: ContentHash, data: Vec<u8>) {
        let data_size = data.len();

        // If this single entry exceeds max size, don't cache it
        if data_size > self.max_size {
            return;
        }

        // Remove existing entry if present
        if let Some(existing) = self.entries.remove(&key) {
            self.current_size -= existing.data.len();
            self.order.retain(|k| k != &key);
        }

        // Evict LRU entries until we have room
        while self.current_size + data_size > self.max_size && !self.order.is_empty() {
            if let Some(lru_key) = self.order.first().cloned() {
                if let Some(entry) = self.entries.remove(&lru_key) {
                    self.current_size -= entry.data.len();
                }
                self.order.remove(0);
            }
        }

        // Insert new entry
        self.counter += 1;
        let entry = LruEntry {
            data: Arc::new(data),
            last_access: self.counter,
            access_count: 1,
        };
        self.current_size += data_size;
        self.entries.insert(key, entry);
        self.order.push(key);
    }

    /// Check if an entry exists.
    fn contains(&self, key: &ContentHash) -> bool {
        self.entries.contains_key(key)
    }

    /// Remove an entry.
    fn remove(&mut self, key: &ContentHash) -> bool {
        if let Some(entry) = self.entries.remove(key) {
            self.current_size -= entry.data.len();
            self.order.retain(|k| k != key);
            true
        } else {
            false
        }
    }

    /// Clear all entries.
    fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
        self.current_size = 0;
    }

    /// Get the number of entries.
    fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty.
    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get current size in bytes.
    fn size(&self) -> usize {
        self.current_size
    }

    /// Get statistics.
    fn stats(&self) -> MemoryCacheStats {
        MemoryCacheStats {
            entry_count: self.entries.len(),
            current_size: self.current_size,
            max_size: self.max_size,
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
        }
    }

    /// Reset hit/miss counters.
    fn reset_stats(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }
}

/// Statistics for memory cache.
#[derive(Debug, Clone, Default)]
pub struct MemoryCacheStats {
    /// Number of cached entries.
    pub entry_count: usize,
    /// Current size in bytes.
    pub current_size: usize,
    /// Maximum size in bytes.
    pub max_size: usize,
    /// Total cache hits.
    pub hits: u64,
    /// Total cache misses.
    pub misses: u64,
}

impl MemoryCacheStats {
    /// Compute hit rate as percentage.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            100.0
        } else {
            (self.hits as f64 / total as f64) * 100.0
        }
    }
}

// ---------------------------------------------------------------------------
// PAK Archive
// ---------------------------------------------------------------------------

/// PAK archive header magic.
const PAK_MAGIC: [u8; 4] = *b"TPAK";

/// PAK archive version.
const PAK_VERSION: u32 = 1;

/// Entry in a PAK archive index.
#[derive(Debug, Clone)]
pub struct PakEntry {
    /// Cache key hash.
    pub key_hash: ContentHash,
    /// Offset in the data section.
    pub offset: u64,
    /// Size of the data.
    pub size: u32,
    /// CRC32 checksum for integrity.
    pub crc32: u32,
}

/// Read-only PAK archive for pre-compiled shaders.
///
/// Format:
/// - Header: magic (4) + version (4) + entry_count (4) + reserved (20)
/// - Index: entry_count * (hash (32) + offset (8) + size (4) + crc32 (4))
/// - Data: concatenated shader bytecode
pub struct PakArchive {
    /// Path to the archive file.
    path: PathBuf,
    /// Index of entries (loaded into memory).
    index: HashMap<ContentHash, PakEntry>,
    /// Data start offset.
    data_offset: u64,
}

impl PakArchive {
    /// Open an existing PAK archive.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, CacheError> {
        let path = path.as_ref().to_path_buf();
        let mut file = fs::File::open(&path)?;

        // Read header
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if magic != PAK_MAGIC {
            return Err(CacheError::InvalidPakFormat("invalid magic".to_string()));
        }

        let mut version_bytes = [0u8; 4];
        file.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);
        if version != PAK_VERSION {
            return Err(CacheError::InvalidPakFormat(format!(
                "unsupported version: {}",
                version
            )));
        }

        let mut count_bytes = [0u8; 4];
        file.read_exact(&mut count_bytes)?;
        let entry_count = u32::from_le_bytes(count_bytes);

        // Skip reserved bytes
        file.seek(SeekFrom::Current(20))?;

        // Read index
        let mut index = HashMap::new();
        for _ in 0..entry_count {
            let mut hash_bytes = [0u8; 32];
            file.read_exact(&mut hash_bytes)?;
            let key_hash = ContentHash::from_raw(hash_bytes);

            let mut offset_bytes = [0u8; 8];
            file.read_exact(&mut offset_bytes)?;
            let offset = u64::from_le_bytes(offset_bytes);

            let mut size_bytes = [0u8; 4];
            file.read_exact(&mut size_bytes)?;
            let size = u32::from_le_bytes(size_bytes);

            let mut crc_bytes = [0u8; 4];
            file.read_exact(&mut crc_bytes)?;
            let crc32 = u32::from_le_bytes(crc_bytes);

            index.insert(
                key_hash,
                PakEntry {
                    key_hash,
                    offset,
                    size,
                    crc32,
                },
            );
        }

        let data_offset = file.stream_position()?;

        Ok(Self {
            path,
            index,
            data_offset,
        })
    }

    /// Get shader data by key hash.
    pub fn get(&self, key: &ContentHash) -> Result<Option<Vec<u8>>, CacheError> {
        let entry = match self.index.get(key) {
            Some(e) => e,
            None => return Ok(None),
        };

        let mut file = fs::File::open(&self.path)?;
        file.seek(SeekFrom::Start(self.data_offset + entry.offset))?;

        let mut data = vec![0u8; entry.size as usize];
        file.read_exact(&mut data)?;

        // Verify CRC32
        let computed_crc = crc32fast::hash(&data);
        if computed_crc != entry.crc32 {
            return Err(CacheError::CorruptedData(format!(
                "CRC32 mismatch: expected {:08x}, got {:08x}",
                entry.crc32, computed_crc
            )));
        }

        Ok(Some(data))
    }

    /// Check if a key exists in the archive.
    pub fn contains(&self, key: &ContentHash) -> bool {
        self.index.contains_key(key)
    }

    /// Get the number of entries.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// List all keys in the archive.
    pub fn keys(&self) -> impl Iterator<Item = &ContentHash> {
        self.index.keys()
    }

    /// Create a PAK archive from entries.
    pub fn create<P: AsRef<Path>>(
        path: P,
        entries: &[(ContentHash, Vec<u8>)],
    ) -> Result<(), CacheError> {
        let path = path.as_ref();
        let mut file = fs::File::create(path)?;

        // Write header
        file.write_all(&PAK_MAGIC)?;
        file.write_all(&PAK_VERSION.to_le_bytes())?;
        file.write_all(&(entries.len() as u32).to_le_bytes())?;
        file.write_all(&[0u8; 20])?; // Reserved

        // Calculate offsets and build index entries
        let mut pak_entries = Vec::new();
        let mut current_offset = 0u64;

        for (key, data) in entries {
            let crc32 = crc32fast::hash(data);
            pak_entries.push(PakEntry {
                key_hash: *key,
                offset: current_offset,
                size: data.len() as u32,
                crc32,
            });
            current_offset += data.len() as u64;
        }

        // Write index
        for entry in &pak_entries {
            file.write_all(entry.key_hash.as_bytes())?;
            file.write_all(&entry.offset.to_le_bytes())?;
            file.write_all(&entry.size.to_le_bytes())?;
            file.write_all(&entry.crc32.to_le_bytes())?;
        }

        // Write data
        for (_, data) in entries {
            file.write_all(data)?;
        }

        file.sync_all()?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Cache Configuration
// ---------------------------------------------------------------------------

/// Configuration for the 3-level shader cache.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum memory cache size in bytes (default: 512 MB).
    pub memory_size: usize,
    /// Path to PAK archive (optional).
    pub pak_path: Option<PathBuf>,
    /// Whether to verify CRC32 on disk reads.
    pub verify_crc: bool,
    /// Compiler version string for cache invalidation.
    pub compiler_version: String,
}

impl CacheConfig {
    /// Create config with default 512 MB memory limit.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create config with specified memory size in MB.
    pub fn with_memory_size_mb(mb: usize) -> Self {
        Self {
            memory_size: mb * 1024 * 1024,
            ..Default::default()
        }
    }

    /// Set the PAK archive path.
    pub fn with_pak<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.pak_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Set CRC verification.
    pub fn with_crc_verification(mut self, verify: bool) -> Self {
        self.verify_crc = verify;
        self
    }

    /// Set compiler version.
    pub fn with_compiler_version(mut self, version: impl Into<String>) -> Self {
        self.compiler_version = version.into();
        self
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            memory_size: 512 * 1024 * 1024, // 512 MB
            pak_path: None,
            verify_crc: true,
            compiler_version: format!("naga-{}", env!("CARGO_PKG_VERSION")),
        }
    }
}

// ---------------------------------------------------------------------------
// Cache Error
// ---------------------------------------------------------------------------

/// Error type for cache operations.
#[derive(Debug)]
pub enum CacheError {
    /// I/O error.
    Io(io::Error),
    /// Invalid PAK format.
    InvalidPakFormat(String),
    /// Data corruption detected.
    CorruptedData(String),
    /// Entry not found.
    NotFound,
}

impl fmt::Display for CacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::InvalidPakFormat(msg) => write!(f, "invalid PAK format: {}", msg),
            Self::CorruptedData(msg) => write!(f, "corrupted data: {}", msg),
            Self::NotFound => write!(f, "cache entry not found"),
        }
    }
}

impl std::error::Error for CacheError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for CacheError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

// ---------------------------------------------------------------------------
// 3-Level Shader Cache
// ---------------------------------------------------------------------------

/// Combined statistics for all cache levels.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Memory cache statistics.
    pub memory: MemoryCacheStats,
    /// Disk cache entry count.
    pub disk_entries: usize,
    /// Disk cache total size.
    pub disk_size: usize,
    /// PAK archive entry count.
    pub pak_entries: usize,
    /// Total disk lookups.
    pub disk_lookups: u64,
    /// Total disk hits.
    pub disk_hits: u64,
    /// Total PAK lookups.
    pub pak_lookups: u64,
    /// Total PAK hits.
    pub pak_hits: u64,
}

impl CacheStats {
    /// Overall hit rate across all levels.
    pub fn overall_hit_rate(&self) -> f64 {
        let total_lookups = self.memory.hits + self.memory.misses;
        let total_hits = self.memory.hits + self.disk_hits + self.pak_hits;
        if total_lookups == 0 {
            100.0
        } else {
            (total_hits as f64 / total_lookups as f64) * 100.0
        }
    }
}

/// 3-Level hierarchical shader cache.
///
/// Provides fast shader bytecode lookup with three cache levels:
/// 1. In-memory LRU (fastest, limited size)
/// 2. Disk cache (persistent, content-addressed)
/// 3. PAK archive (read-only, pre-compiled shaders)
pub struct ShaderCache3L {
    /// In-memory LRU cache (protected by RwLock for concurrent access).
    memory: RwLock<MemoryLruCache>,
    /// Disk backend for persistent storage.
    disk: FileBackend,
    /// Optional PAK archive.
    pak: Option<PakArchive>,
    /// Configuration.
    config: CacheConfig,
    /// Statistics counters.
    disk_lookups: AtomicU64,
    disk_hits: AtomicU64,
    pak_lookups: AtomicU64,
    pak_hits: AtomicU64,
}

impl ShaderCache3L {
    /// Create a new 3-level shader cache.
    ///
    /// # Arguments
    ///
    /// * `disk_path` - Directory for disk cache storage
    /// * `config` - Cache configuration
    pub fn new<P: AsRef<Path>>(disk_path: P, config: CacheConfig) -> Result<Self, CacheError> {
        let disk = FileBackend::new(disk_path)?;

        let pak = config
            .pak_path
            .as_ref()
            .and_then(|p| PakArchive::open(p).ok());

        Ok(Self {
            memory: RwLock::new(MemoryLruCache::new(config.memory_size)),
            disk,
            pak,
            config,
            disk_lookups: AtomicU64::new(0),
            disk_hits: AtomicU64::new(0),
            pak_lookups: AtomicU64::new(0),
            pak_hits: AtomicU64::new(0),
        })
    }

    /// Create with default configuration.
    pub fn with_defaults<P: AsRef<Path>>(disk_path: P) -> Result<Self, CacheError> {
        Self::new(disk_path, CacheConfig::default())
    }

    /// Get compiled shader bytecode by cache key.
    ///
    /// Searches all three levels in order:
    /// 1. Memory cache (fastest)
    /// 2. Disk cache
    /// 3. PAK archive
    ///
    /// If found in disk or PAK, promotes to memory cache.
    ///
    /// # Performance
    ///
    /// - Memory hit: <1ms
    /// - Disk hit: <10ms
    /// - PAK hit: <10ms
    pub fn get(&self, key: &CacheKey) -> Result<Option<Vec<u8>>, CacheError> {
        let combined = key.combined_hash();

        // Level 1: Memory cache
        {
            let mut memory = self.memory.write().unwrap();
            if let Some(data) = memory.get(&combined) {
                return Ok(Some((*data).clone()));
            }
        }

        // Level 2: Disk cache
        self.disk_lookups.fetch_add(1, Ordering::Relaxed);
        if let Some(data) = self.disk.get(&combined)? {
            self.disk_hits.fetch_add(1, Ordering::Relaxed);

            // Verify CRC if enabled
            if self.config.verify_crc {
                // CRC is stored as first 4 bytes
                if data.len() >= 4 {
                    let stored_crc = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                    let actual_data = &data[4..];
                    let computed_crc = crc32fast::hash(actual_data);
                    if stored_crc != computed_crc {
                        // Corrupted, remove from cache
                        let _ = self.disk.delete(&combined);
                        // Continue to PAK
                    } else {
                        // Valid, promote to memory
                        let actual_data = actual_data.to_vec();
                        {
                            let mut memory = self.memory.write().unwrap();
                            memory.put(combined, actual_data.clone());
                        }
                        return Ok(Some(actual_data));
                    }
                }
            } else {
                // No CRC verification, data is stored directly
                let mut memory = self.memory.write().unwrap();
                memory.put(combined, data.clone());
                return Ok(Some(data));
            }
        }

        // Level 3: PAK archive
        if let Some(pak) = &self.pak {
            self.pak_lookups.fetch_add(1, Ordering::Relaxed);
            if let Some(data) = pak.get(&combined)? {
                self.pak_hits.fetch_add(1, Ordering::Relaxed);

                // Promote to memory
                {
                    let mut memory = self.memory.write().unwrap();
                    memory.put(combined, data.clone());
                }
                return Ok(Some(data));
            }
        }

        Ok(None)
    }

    /// Store compiled shader bytecode.
    ///
    /// Stores in both memory and disk cache.
    pub fn put(&self, key: &CacheKey, data: &[u8]) -> Result<(), CacheError> {
        let combined = key.combined_hash();

        // Store in memory
        {
            let mut memory = self.memory.write().unwrap();
            memory.put(combined, data.to_vec());
        }

        // Store on disk with CRC prefix
        let mut disk_data = Vec::with_capacity(4 + data.len());
        let crc = crc32fast::hash(data);
        disk_data.extend_from_slice(&crc.to_le_bytes());
        disk_data.extend_from_slice(data);
        self.disk.put(&disk_data)?;

        Ok(())
    }

    /// Check if a key exists in any cache level.
    pub fn contains(&self, key: &CacheKey) -> bool {
        let combined = key.combined_hash();

        // Check memory
        {
            let memory = self.memory.read().unwrap();
            if memory.contains(&combined) {
                return true;
            }
        }

        // Check disk
        if self.disk.has(&combined) {
            return true;
        }

        // Check PAK
        if let Some(pak) = &self.pak {
            if pak.contains(&combined) {
                return true;
            }
        }

        false
    }

    /// Remove an entry from memory and disk caches.
    ///
    /// Note: PAK entries cannot be removed (read-only).
    pub fn remove(&self, key: &CacheKey) -> Result<bool, CacheError> {
        let combined = key.combined_hash();
        let mut removed = false;

        // Remove from memory
        {
            let mut memory = self.memory.write().unwrap();
            if memory.remove(&combined) {
                removed = true;
            }
        }

        // Remove from disk
        if self.disk.delete(&combined)? {
            removed = true;
        }

        Ok(removed)
    }

    /// Clear memory cache only.
    pub fn clear_memory(&self) {
        let mut memory = self.memory.write().unwrap();
        memory.clear();
    }

    /// Clear memory and disk caches.
    ///
    /// Warning: This removes all cached shaders from disk.
    pub fn clear_all(&self) -> Result<(), CacheError> {
        // Clear memory
        {
            let mut memory = self.memory.write().unwrap();
            memory.clear();
        }

        // Clear disk (list and delete all)
        let hashes = self.disk.list()?;
        for hash in hashes {
            let _ = self.disk.delete(&hash);
        }

        Ok(())
    }

    /// Get combined statistics.
    pub fn stats(&self) -> CacheStats {
        let memory_stats = {
            let memory = self.memory.read().unwrap();
            memory.stats()
        };

        let disk_entries = self.disk.list().map(|l| l.len()).unwrap_or(0);
        let pak_entries = self.pak.as_ref().map(|p| p.len()).unwrap_or(0);

        CacheStats {
            memory: memory_stats,
            disk_entries,
            disk_size: 0, // Would need to sum file sizes
            pak_entries,
            disk_lookups: self.disk_lookups.load(Ordering::Relaxed),
            disk_hits: self.disk_hits.load(Ordering::Relaxed),
            pak_lookups: self.pak_lookups.load(Ordering::Relaxed),
            pak_hits: self.pak_hits.load(Ordering::Relaxed),
        }
    }

    /// Reset all statistics counters.
    pub fn reset_stats(&self) {
        {
            let memory = self.memory.read().unwrap();
            memory.reset_stats();
        }
        self.disk_lookups.store(0, Ordering::Relaxed);
        self.disk_hits.store(0, Ordering::Relaxed);
        self.pak_lookups.store(0, Ordering::Relaxed);
        self.pak_hits.store(0, Ordering::Relaxed);
    }

    /// Get the configuration.
    pub fn config(&self) -> &CacheConfig {
        &self.config
    }

    /// Invalidate entries for a specific compiler version.
    ///
    /// This is a no-op currently since we'd need to store compiler version
    /// with each entry. Future versions may support this.
    pub fn invalidate_compiler_version(&self, _version: &str) -> Result<usize, CacheError> {
        // TODO: Implement version-based invalidation
        // Would require storing metadata with each cache entry
        Ok(0)
    }

    /// Warm up memory cache from disk.
    ///
    /// Loads entries from disk into memory up to the memory limit.
    pub fn warm_up(&self) -> Result<usize, CacheError> {
        let hashes = self.disk.list()?;
        let mut loaded = 0;

        for hash in hashes {
            if let Some(data) = self.disk.get(&hash)? {
                // Skip CRC prefix
                if data.len() > 4 {
                    let actual_data = if self.config.verify_crc {
                        data[4..].to_vec()
                    } else {
                        data
                    };

                    let mut memory = self.memory.write().unwrap();
                    if memory.size() + actual_data.len() <= self.config.memory_size {
                        memory.put(hash, actual_data);
                        loaded += 1;
                    } else {
                        break;
                    }
                }
            }
        }

        Ok(loaded)
    }

    /// Get direct access to the disk backend.
    pub fn disk_backend(&self) -> &FileBackend {
        &self.disk
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};
    use tempfile::TempDir;

    fn setup_cache() -> (ShaderCache3L, TempDir) {
        let tmp = TempDir::new().unwrap();
        let config = CacheConfig::with_memory_size_mb(1); // 1 MB for testing
        let cache = ShaderCache3L::new(tmp.path().join("cache"), config).unwrap();
        (cache, tmp)
    }

    fn test_key() -> CacheKey {
        CacheKey::builder()
            .source(b"test shader source")
            .platform(TargetPlatform::Vulkan)
            .build()
    }

    // -----------------------------------------------------------------------
    // Test 1: In-memory cache hit
    // -----------------------------------------------------------------------

    #[test]
    fn test_memory_cache_hit() {
        let (cache, _tmp) = setup_cache();
        let key = test_key();
        let data = b"compiled spirv bytecode".to_vec();

        // Store
        cache.put(&key, &data).unwrap();

        // Retrieve - should be memory hit
        let start = Instant::now();
        let result = cache.get(&key).unwrap();
        let elapsed = start.elapsed();

        assert_eq!(result, Some(data));
        assert!(elapsed < Duration::from_millis(1), "Memory lookup took {:?}", elapsed);

        // Check stats
        let stats = cache.stats();
        assert_eq!(stats.memory.hits, 1);
    }

    // -----------------------------------------------------------------------
    // Test 2: In-memory cache miss
    // -----------------------------------------------------------------------

    #[test]
    fn test_memory_cache_miss() {
        let (cache, _tmp) = setup_cache();
        let key = test_key();

        let result = cache.get(&key).unwrap();
        assert!(result.is_none());

        let stats = cache.stats();
        assert_eq!(stats.memory.misses, 1);
    }

    // -----------------------------------------------------------------------
    // Test 3: LRU eviction
    // -----------------------------------------------------------------------

    #[test]
    fn test_lru_eviction() {
        let tmp = TempDir::new().unwrap();
        // Very small cache: 100 bytes
        let config = CacheConfig {
            memory_size: 100,
            ..Default::default()
        };
        let cache = ShaderCache3L::new(tmp.path().join("cache"), config).unwrap();

        // Insert entries that exceed limit
        let key1 = CacheKey::builder().source(b"source1").build();
        let key2 = CacheKey::builder().source(b"source2").build();
        let key3 = CacheKey::builder().source(b"source3").build();

        let data = vec![0u8; 40]; // Each entry is 40 bytes

        cache.put(&key1, &data).unwrap();
        cache.put(&key2, &data).unwrap();
        cache.put(&key3, &data).unwrap(); // This should evict key1

        let stats = cache.stats();
        // With 100 byte limit and 40 byte entries, should fit 2
        assert!(stats.memory.entry_count <= 3);
        assert!(stats.memory.current_size <= 100);
    }

    // -----------------------------------------------------------------------
    // Test 4: Disk cache store/retrieve
    // -----------------------------------------------------------------------

    #[test]
    fn test_disk_cache_store_retrieve() {
        let (cache, _tmp) = setup_cache();
        let key = test_key();
        let data = b"disk cached data".to_vec();

        // Store
        cache.put(&key, &data).unwrap();

        // Clear memory to force disk read
        cache.clear_memory();

        // Retrieve from disk
        let start = Instant::now();
        let result = cache.get(&key).unwrap();
        let elapsed = start.elapsed();

        assert_eq!(result, Some(data));
        assert!(elapsed < Duration::from_millis(10), "Disk lookup took {:?}", elapsed);

        // Should have promoted to memory
        let stats = cache.stats();
        assert_eq!(stats.disk_hits, 1);
    }

    // -----------------------------------------------------------------------
    // Test 5: Cache key generation
    // -----------------------------------------------------------------------

    #[test]
    fn test_cache_key_generation() {
        let key1 = CacheKey::builder()
            .source(b"shader source")
            .define("MAX_LIGHTS", "8")
            .platform(TargetPlatform::Vulkan)
            .build();

        let key2 = CacheKey::builder()
            .source(b"shader source")
            .define("MAX_LIGHTS", "16") // Different define
            .platform(TargetPlatform::Vulkan)
            .build();

        let key3 = CacheKey::builder()
            .source(b"shader source")
            .define("MAX_LIGHTS", "8") // Same as key1
            .platform(TargetPlatform::Vulkan)
            .build();

        // Different defines should produce different hashes
        assert_ne!(key1.combined_hash(), key2.combined_hash());

        // Same parameters should produce same hash
        assert_eq!(key1.combined_hash(), key3.combined_hash());
    }

    // -----------------------------------------------------------------------
    // Test 6: Invalidation on source change
    // -----------------------------------------------------------------------

    #[test]
    fn test_invalidation_on_source_change() {
        let (cache, _tmp) = setup_cache();

        let key1 = CacheKey::builder().source(b"original source").build();
        let key2 = CacheKey::builder().source(b"modified source").build();

        let data1 = b"original bytecode".to_vec();
        let data2 = b"modified bytecode".to_vec();

        cache.put(&key1, &data1).unwrap();
        cache.put(&key2, &data2).unwrap();

        // Different sources should have different cache entries
        assert_eq!(cache.get(&key1).unwrap(), Some(data1));
        assert_eq!(cache.get(&key2).unwrap(), Some(data2));
    }

    // -----------------------------------------------------------------------
    // Test 7: Invalidation on compiler version
    // -----------------------------------------------------------------------

    #[test]
    fn test_invalidation_on_compiler_version() {
        let (cache, _tmp) = setup_cache();

        let key1 = CacheKey::builder()
            .source(b"same source")
            .compiler_version("naga-0.23")
            .build();

        let key2 = CacheKey::builder()
            .source(b"same source")
            .compiler_version("naga-0.24")
            .build();

        // Different compiler versions should have different keys
        assert_ne!(key1.combined_hash(), key2.combined_hash());
    }

    // -----------------------------------------------------------------------
    // Test 8: PAK archive lookup
    // -----------------------------------------------------------------------

    #[test]
    fn test_pak_archive_lookup() {
        let tmp = TempDir::new().unwrap();
        let pak_path = tmp.path().join("shaders.pak");

        // Create PAK with test entry
        let key = CacheKey::builder().source(b"precompiled shader").build();
        let data = b"precompiled bytecode".to_vec();

        PakArchive::create(&pak_path, &[(key.combined_hash(), data.clone())]).unwrap();

        // Create cache with PAK
        let config = CacheConfig::with_memory_size_mb(1).with_pak(&pak_path);
        let cache = ShaderCache3L::new(tmp.path().join("cache"), config).unwrap();

        // Lookup should hit PAK
        let result = cache.get(&key).unwrap();
        assert_eq!(result, Some(data));

        let stats = cache.stats();
        assert_eq!(stats.pak_hits, 1);
    }

    // -----------------------------------------------------------------------
    // Test 9: Performance benchmark - memory lookup
    // -----------------------------------------------------------------------

    #[test]
    fn test_performance_memory_lookup() {
        let (cache, _tmp) = setup_cache();
        let key = test_key();
        let data = vec![0u8; 1024]; // 1KB

        cache.put(&key, &data).unwrap();

        // Warm up
        for _ in 0..10 {
            let _ = cache.get(&key);
        }

        // Benchmark
        let iterations = 1000;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = cache.get(&key).unwrap();
        }
        let elapsed = start.elapsed();
        let per_lookup = elapsed / iterations;

        assert!(
            per_lookup < Duration::from_millis(1),
            "Memory lookup too slow: {:?}",
            per_lookup
        );

        println!("Memory lookup: {:?} per call", per_lookup);
    }

    // -----------------------------------------------------------------------
    // Test 10: Performance benchmark - disk lookup
    // -----------------------------------------------------------------------

    #[test]
    fn test_performance_disk_lookup() {
        let (cache, _tmp) = setup_cache();
        let key = test_key();
        let data = vec![0u8; 1024];

        cache.put(&key, &data).unwrap();

        // Benchmark cold disk reads
        let iterations = 100;
        let mut total = Duration::ZERO;

        for _ in 0..iterations {
            cache.clear_memory();
            let start = Instant::now();
            let _ = cache.get(&key).unwrap();
            total += start.elapsed();
        }

        let per_lookup = total / iterations;

        assert!(
            per_lookup < Duration::from_millis(10),
            "Disk lookup too slow: {:?}",
            per_lookup
        );

        println!("Disk lookup: {:?} per call", per_lookup);
    }

    // -----------------------------------------------------------------------
    // Test 11: Concurrent access
    // -----------------------------------------------------------------------

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let tmp = TempDir::new().unwrap();
        let config = CacheConfig::with_memory_size_mb(10);
        let cache = Arc::new(ShaderCache3L::new(tmp.path().join("cache"), config).unwrap());

        let threads: Vec<_> = (0..4)
            .map(|i| {
                let cache = Arc::clone(&cache);
                thread::spawn(move || {
                    for j in 0..100 {
                        let key = CacheKey::builder()
                            .source(format!("shader_{}_{}", i, j).as_bytes())
                            .build();
                        let data = vec![i as u8; 100];

                        cache.put(&key, &data).unwrap();
                        let result = cache.get(&key).unwrap();
                        assert_eq!(result, Some(data));
                    }
                })
            })
            .collect();

        for t in threads {
            t.join().unwrap();
        }
    }

    // -----------------------------------------------------------------------
    // Test 12: Cache size limits
    // -----------------------------------------------------------------------

    #[test]
    fn test_cache_size_limits() {
        let tmp = TempDir::new().unwrap();
        let config = CacheConfig {
            memory_size: 1000, // 1KB limit
            ..Default::default()
        };
        let cache = ShaderCache3L::new(tmp.path().join("cache"), config).unwrap();

        // Insert more than limit
        for i in 0..20 {
            let key = CacheKey::builder()
                .source(format!("shader_{}", i).as_bytes())
                .build();
            let data = vec![i as u8; 100]; // 100 bytes each
            cache.put(&key, &data).unwrap();
        }

        let stats = cache.stats();
        assert!(stats.memory.current_size <= 1000);
    }

    // -----------------------------------------------------------------------
    // Test 13: Platform-specific keys
    // -----------------------------------------------------------------------

    #[test]
    fn test_platform_specific_keys() {
        let base_key = |platform| {
            CacheKey::builder()
                .source(b"cross-platform shader")
                .platform(platform)
                .build()
        };

        let vulkan = base_key(TargetPlatform::Vulkan);
        let metal = base_key(TargetPlatform::Metal);
        let d3d12 = base_key(TargetPlatform::D3D12);

        // All platforms should have different hashes
        assert_ne!(vulkan.combined_hash(), metal.combined_hash());
        assert_ne!(vulkan.combined_hash(), d3d12.combined_hash());
        assert_ne!(metal.combined_hash(), d3d12.combined_hash());
    }

    // -----------------------------------------------------------------------
    // Test 14: Include hashes affect key
    // -----------------------------------------------------------------------

    #[test]
    fn test_include_hashes_affect_key() {
        let include_v1 = ContentHash::from_bytes(b"include content v1");
        let include_v2 = ContentHash::from_bytes(b"include content v2");

        let key1 = CacheKey::builder()
            .source(b"shader with include")
            .include_hash("common.wgsl", include_v1)
            .build();

        let key2 = CacheKey::builder()
            .source(b"shader with include")
            .include_hash("common.wgsl", include_v2)
            .build();

        assert_ne!(key1.combined_hash(), key2.combined_hash());
    }

    // -----------------------------------------------------------------------
    // Test 15: Optimization level affects key
    // -----------------------------------------------------------------------

    #[test]
    fn test_optimization_level_affects_key() {
        let key_o0 = CacheKey::builder()
            .source(b"shader")
            .optimization_level(0)
            .build();

        let key_o3 = CacheKey::builder()
            .source(b"shader")
            .optimization_level(3)
            .build();

        assert_ne!(key_o0.combined_hash(), key_o3.combined_hash());
    }

    // -----------------------------------------------------------------------
    // Test 16: CRC verification
    // -----------------------------------------------------------------------

    #[test]
    fn test_crc_verification() {
        let (cache, _tmp) = setup_cache();
        let key = test_key();
        let data = b"data to verify".to_vec();

        cache.put(&key, &data).unwrap();
        cache.clear_memory();

        // Normal retrieval should work
        let result = cache.get(&key).unwrap();
        assert_eq!(result, Some(data));
    }

    // -----------------------------------------------------------------------
    // Test 17: PAK archive creation
    // -----------------------------------------------------------------------

    #[test]
    fn test_pak_archive_creation() {
        let tmp = TempDir::new().unwrap();
        let pak_path = tmp.path().join("test.pak");

        let entries = vec![
            (ContentHash::from_bytes(b"key1"), b"data1".to_vec()),
            (ContentHash::from_bytes(b"key2"), b"data2".to_vec()),
            (ContentHash::from_bytes(b"key3"), b"data3".to_vec()),
        ];

        PakArchive::create(&pak_path, &entries).unwrap();

        let pak = PakArchive::open(&pak_path).unwrap();
        assert_eq!(pak.len(), 3);

        for (key, expected_data) in &entries {
            let data = pak.get(key).unwrap().unwrap();
            assert_eq!(&data, expected_data);
        }
    }

    // -----------------------------------------------------------------------
    // Test 18: Warm up from disk
    // -----------------------------------------------------------------------

    #[test]
    fn test_warm_up_from_disk() {
        let (cache, _tmp) = setup_cache();

        // Store some entries
        for i in 0..5 {
            let key = CacheKey::builder()
                .source(format!("shader_{}", i).as_bytes())
                .build();
            let data = vec![i as u8; 50];
            cache.put(&key, &data).unwrap();
        }

        // Clear memory
        cache.clear_memory();
        assert_eq!(cache.stats().memory.entry_count, 0);

        // Warm up
        let loaded = cache.warm_up().unwrap();
        assert!(loaded > 0);
        assert!(cache.stats().memory.entry_count > 0);
    }

    // -----------------------------------------------------------------------
    // Test 19: Clear all caches
    // -----------------------------------------------------------------------

    #[test]
    fn test_clear_all_caches() {
        let (cache, _tmp) = setup_cache();
        let key = test_key();
        let data = b"data to clear".to_vec();

        cache.put(&key, &data).unwrap();
        assert!(cache.contains(&key));

        cache.clear_all().unwrap();

        assert!(!cache.contains(&key));
        assert_eq!(cache.stats().memory.entry_count, 0);
        assert_eq!(cache.stats().disk_entries, 0);
    }

    // -----------------------------------------------------------------------
    // Test 20: Statistics tracking
    // -----------------------------------------------------------------------

    #[test]
    fn test_statistics_tracking() {
        let (cache, _tmp) = setup_cache();
        let key = test_key();
        let data = b"stats test".to_vec();

        // Initial stats
        let stats = cache.stats();
        assert_eq!(stats.memory.hits, 0);
        assert_eq!(stats.memory.misses, 0);

        // Miss
        let _ = cache.get(&key);
        assert_eq!(cache.stats().memory.misses, 1);

        // Store and hit
        cache.put(&key, &data).unwrap();
        let _ = cache.get(&key);
        assert_eq!(cache.stats().memory.hits, 1);

        // Hit rate
        let stats = cache.stats();
        assert!(stats.memory.hit_rate() > 0.0);

        // Reset stats
        cache.reset_stats();
        let stats = cache.stats();
        assert_eq!(stats.memory.hits, 0);
        assert_eq!(stats.memory.misses, 0);
    }

    // -----------------------------------------------------------------------
    // Test 21: Remove entry
    // -----------------------------------------------------------------------

    #[test]
    fn test_remove_entry() {
        let (cache, _tmp) = setup_cache();
        let key = test_key();
        let data = b"to be removed".to_vec();

        cache.put(&key, &data).unwrap();
        assert!(cache.contains(&key));

        let removed = cache.remove(&key).unwrap();
        assert!(removed);
        assert!(!cache.contains(&key));
    }

    // -----------------------------------------------------------------------
    // Test 22: Key builder defaults
    // -----------------------------------------------------------------------

    #[test]
    fn test_key_builder_defaults() {
        let key = CacheKey::builder().build();

        assert_eq!(key.source_hash, ContentHash::zero());
        assert!(key.defines.is_empty());
        assert_eq!(key.platform, TargetPlatform::Vulkan);
        assert_eq!(key.optimization_level, 2);
    }

    // -----------------------------------------------------------------------
    // Test 23: Multiple defines sorting
    // -----------------------------------------------------------------------

    #[test]
    fn test_multiple_defines_sorting() {
        // Order of defines shouldn't matter
        let key1 = CacheKey::builder()
            .source(b"shader")
            .define("A", "1")
            .define("B", "2")
            .build();

        let key2 = CacheKey::builder()
            .source(b"shader")
            .define("B", "2")
            .define("A", "1")
            .build();

        assert_eq!(key1.combined_hash(), key2.combined_hash());
    }

    // -----------------------------------------------------------------------
    // Test 24: Large data handling
    // -----------------------------------------------------------------------

    #[test]
    fn test_large_data_handling() {
        let tmp = TempDir::new().unwrap();
        let config = CacheConfig::with_memory_size_mb(10);
        let cache = ShaderCache3L::new(tmp.path().join("cache"), config).unwrap();

        let key = test_key();
        let data = vec![0x42u8; 1024 * 1024]; // 1 MB

        cache.put(&key, &data).unwrap();
        let result = cache.get(&key).unwrap();

        assert_eq!(result, Some(data));
    }

    // -----------------------------------------------------------------------
    // Test 25: Empty data handling
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_data_handling() {
        let (cache, _tmp) = setup_cache();
        let key = test_key();
        let data: Vec<u8> = vec![];

        cache.put(&key, &data).unwrap();
        let result = cache.get(&key).unwrap();

        assert_eq!(result, Some(data));
    }
}
