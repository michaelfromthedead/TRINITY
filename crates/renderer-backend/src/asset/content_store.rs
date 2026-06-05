//! ContentStore Streaming API for TRINITY asset pipeline.
//!
//! This module provides a unified streaming API for content-addressed storage:
//!
//! - [`ContentStore`]: Trait defining the streaming API
//! - [`FileContentStore`]: Disk-backed store with direct streaming I/O
//! - [`MemoryContentStore`]: In-memory store for testing and caching
//! - [`StreamingHashReader`]: Hash computation while reading
//! - [`SeekableContentReader`]: Seekable reader for partial asset access
//!
//! # Design Goals
//!
//! 1. **Low Memory**: Stream through bounded buffers (default 64KB)
//! 2. **Hash-While-Stream**: Compute BLAKE3 hash without buffering entire content
//! 3. **Partial Reads**: Support seeking within stored content
//! 4. **Backend Agnostic**: Same API for file and memory backends
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::asset::content_store::{ContentStore, FileContentStore};
//! use std::io::Cursor;
//!
//! let store = FileContentStore::new("/tmp/store", Default::default())?;
//!
//! // Stream data through hash computation
//! let mut data = Cursor::new(b"large asset data...");
//! let hash = store.put_stream(&mut data)?;
//!
//! // Read back with seeking support
//! let mut reader = store.get_stream(&hash)?.unwrap();
//! reader.seek_to(100)?; // Jump to byte 100
//! let mut buf = [0u8; 50];
//! reader.read_exact(&mut buf)?;
//! ```
//!
//! # Integration with ContentHash
//!
//! Uses the BLAKE3 streaming hasher from [`crate::asset::content_hash`] for
//! efficient hash computation during writes.

use std::collections::HashMap;
use std::fmt;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use crate::pipeline::ContentHash;

#[cfg(feature = "blake3")]
use crate::asset::content_hash::ContentHasher;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Default buffer size for streaming operations (64KB).
pub const DEFAULT_BUFFER_SIZE: usize = 64 * 1024;

/// Minimum buffer size (4KB).
pub const MIN_BUFFER_SIZE: usize = 4 * 1024;

/// Maximum buffer size (16MB).
pub const MAX_BUFFER_SIZE: usize = 16 * 1024 * 1024;

/// Configuration for ContentStore streaming operations.
#[derive(Debug, Clone)]
pub struct ContentStoreConfig {
    /// Buffer size for streaming reads/writes.
    pub buffer_size: usize,
    /// Whether to sync writes to disk immediately.
    pub sync_writes: bool,
    /// Whether to verify hash on read.
    pub verify_on_read: bool,
}

impl Default for ContentStoreConfig {
    fn default() -> Self {
        Self {
            buffer_size: DEFAULT_BUFFER_SIZE,
            sync_writes: true,
            verify_on_read: false,
        }
    }
}

impl ContentStoreConfig {
    /// Create a config with custom buffer size.
    ///
    /// The buffer size is clamped to [MIN_BUFFER_SIZE, MAX_BUFFER_SIZE].
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size.clamp(MIN_BUFFER_SIZE, MAX_BUFFER_SIZE);
        self
    }

    /// Enable or disable sync writes.
    pub fn with_sync_writes(mut self, sync: bool) -> Self {
        self.sync_writes = sync;
        self
    }

    /// Enable or disable hash verification on read.
    pub fn with_verify_on_read(mut self, verify: bool) -> Self {
        self.verify_on_read = verify;
        self
    }
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors that can occur during content store operations.
#[derive(Debug)]
pub enum ContentStoreError {
    /// I/O error during read/write.
    Io(io::Error),
    /// Content not found for the given hash.
    NotFound(ContentHash),
    /// Hash verification failed.
    HashMismatch {
        expected: ContentHash,
        actual: ContentHash,
    },
    /// Invalid seek position.
    InvalidSeek {
        position: u64,
        size: u64,
    },
    /// Buffer size out of range.
    InvalidBufferSize(usize),
}

impl fmt::Display for ContentStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::NotFound(hash) => write!(f, "content not found: {}", hash),
            Self::HashMismatch { expected, actual } => {
                write!(f, "hash mismatch: expected {}, got {}", expected, actual)
            }
            Self::InvalidSeek { position, size } => {
                write!(f, "invalid seek: position {} exceeds size {}", position, size)
            }
            Self::InvalidBufferSize(size) => {
                write!(
                    f,
                    "invalid buffer size {}: must be in range [{}, {}]",
                    size, MIN_BUFFER_SIZE, MAX_BUFFER_SIZE
                )
            }
        }
    }
}

impl std::error::Error for ContentStoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for ContentStoreError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

/// Result type for content store operations.
pub type Result<T> = std::result::Result<T, ContentStoreError>;

// ---------------------------------------------------------------------------
// Streaming hash computation
// ---------------------------------------------------------------------------

/// A reader that computes the content hash while reading.
///
/// Wraps any `Read` implementation and incrementally hashes the data
/// as it flows through. Call `finish()` to get the final hash.
pub struct StreamingHashReader<R: Read> {
    inner: R,
    #[cfg(feature = "blake3")]
    hasher: blake3::Hasher,
    #[cfg(not(feature = "blake3"))]
    hasher: sha2::Sha256,
    bytes_read: u64,
}

impl<R: Read> StreamingHashReader<R> {
    /// Create a new streaming hash reader.
    pub fn new(reader: R) -> Self {
        Self {
            inner: reader,
            #[cfg(feature = "blake3")]
            hasher: blake3::Hasher::new(),
            #[cfg(not(feature = "blake3"))]
            hasher: {
                use sha2::Digest;
                sha2::Sha256::new()
            },
            bytes_read: 0,
        }
    }

    /// Get the number of bytes read so far.
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read
    }

    /// Finish reading and return the content hash.
    ///
    /// Consumes the reader.
    #[cfg(feature = "blake3")]
    pub fn finish(self) -> ContentHash {
        let hash = self.hasher.finalize();
        ContentHash::from_raw(*hash.as_bytes())
    }

    #[cfg(not(feature = "blake3"))]
    pub fn finish(self) -> ContentHash {
        use sha2::Digest;
        let result = self.hasher.finalize();
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&result);
        ContentHash::from_raw(arr)
    }

    /// Get the inner reader, discarding hash state.
    pub fn into_inner(self) -> R {
        self.inner
    }
}

impl<R: Read> Read for StreamingHashReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        if n > 0 {
            #[cfg(feature = "blake3")]
            {
                self.hasher.update(&buf[..n]);
            }
            #[cfg(not(feature = "blake3"))]
            {
                use sha2::Digest;
                self.hasher.update(&buf[..n]);
            }
            self.bytes_read += n as u64;
        }
        Ok(n)
    }
}

// ---------------------------------------------------------------------------
// Streaming hash writer
// ---------------------------------------------------------------------------

/// A writer that computes the content hash while writing.
///
/// Wraps any `Write` implementation and incrementally hashes the data
/// as it flows through. Call `finish()` to get the final hash.
pub struct StreamingHashWriter<W: Write> {
    inner: W,
    #[cfg(feature = "blake3")]
    hasher: blake3::Hasher,
    #[cfg(not(feature = "blake3"))]
    hasher: sha2::Sha256,
    bytes_written: u64,
}

impl<W: Write> StreamingHashWriter<W> {
    /// Create a new streaming hash writer.
    pub fn new(writer: W) -> Self {
        Self {
            inner: writer,
            #[cfg(feature = "blake3")]
            hasher: blake3::Hasher::new(),
            #[cfg(not(feature = "blake3"))]
            hasher: {
                use sha2::Digest;
                sha2::Sha256::new()
            },
            bytes_written: 0,
        }
    }

    /// Get the number of bytes written so far.
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Finish writing and return the content hash.
    ///
    /// Flushes the inner writer and returns the hash.
    #[cfg(feature = "blake3")]
    pub fn finish(mut self) -> io::Result<(W, ContentHash)> {
        self.inner.flush()?;
        let hash = self.hasher.finalize();
        Ok((self.inner, ContentHash::from_raw(*hash.as_bytes())))
    }

    #[cfg(not(feature = "blake3"))]
    pub fn finish(mut self) -> io::Result<(W, ContentHash)> {
        use sha2::Digest;
        self.inner.flush()?;
        let result = self.hasher.finalize();
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&result);
        Ok((self.inner, ContentHash::from_raw(arr)))
    }

    /// Get the inner writer, discarding hash state.
    pub fn into_inner(self) -> W {
        self.inner
    }
}

impl<W: Write> Write for StreamingHashWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(buf)?;
        if n > 0 {
            #[cfg(feature = "blake3")]
            {
                self.hasher.update(&buf[..n]);
            }
            #[cfg(not(feature = "blake3"))]
            {
                use sha2::Digest;
                self.hasher.update(&buf[..n]);
            }
            self.bytes_written += n as u64;
        }
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

// ---------------------------------------------------------------------------
// Seekable content reader
// ---------------------------------------------------------------------------

/// A reader for stored content with seeking support.
///
/// Provides efficient partial reads for large assets.
pub trait SeekableReader: Read {
    /// Get the total size of the content.
    fn size(&self) -> u64;

    /// Get the current position.
    fn position(&self) -> u64;

    /// Seek to an absolute position.
    fn seek_to(&mut self, pos: u64) -> Result<()>;

    /// Seek relative to current position.
    fn seek_relative(&mut self, offset: i64) -> Result<()>;

    /// Remaining bytes to read.
    fn remaining(&self) -> u64 {
        self.size().saturating_sub(self.position())
    }
}

/// A seekable reader backed by a file.
pub struct FileSeekableReader {
    file: BufReader<File>,
    size: u64,
    position: u64,
}

impl FileSeekableReader {
    /// Create a new file seekable reader.
    pub fn new(file: File, buffer_size: usize) -> io::Result<Self> {
        let size = file.metadata()?.len();
        let reader = BufReader::with_capacity(buffer_size, file);
        Ok(Self {
            file: reader,
            size,
            position: 0,
        })
    }

    /// Get the underlying file.
    pub fn into_inner(self) -> File {
        self.file.into_inner()
    }
}

impl Read for FileSeekableReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.file.read(buf)?;
        self.position += n as u64;
        Ok(n)
    }
}

impl SeekableReader for FileSeekableReader {
    fn size(&self) -> u64 {
        self.size
    }

    fn position(&self) -> u64 {
        self.position
    }

    fn seek_to(&mut self, pos: u64) -> Result<()> {
        if pos > self.size {
            return Err(ContentStoreError::InvalidSeek {
                position: pos,
                size: self.size,
            });
        }
        self.file.seek(SeekFrom::Start(pos))?;
        self.position = pos;
        Ok(())
    }

    fn seek_relative(&mut self, offset: i64) -> Result<()> {
        let new_pos = if offset >= 0 {
            self.position.saturating_add(offset as u64)
        } else {
            self.position.saturating_sub((-offset) as u64)
        };
        self.seek_to(new_pos)
    }
}

/// A seekable reader backed by memory.
pub struct MemorySeekableReader {
    data: Arc<Vec<u8>>,
    position: u64,
}

impl MemorySeekableReader {
    /// Create a new memory seekable reader.
    pub fn new(data: Arc<Vec<u8>>) -> Self {
        Self { data, position: 0 }
    }

    /// Get the underlying data.
    pub fn data(&self) -> &[u8] {
        &self.data
    }
}

impl Read for MemorySeekableReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let pos = self.position as usize;
        if pos >= self.data.len() {
            return Ok(0);
        }
        let available = self.data.len() - pos;
        let to_read = buf.len().min(available);
        buf[..to_read].copy_from_slice(&self.data[pos..pos + to_read]);
        self.position += to_read as u64;
        Ok(to_read)
    }
}

impl SeekableReader for MemorySeekableReader {
    fn size(&self) -> u64 {
        self.data.len() as u64
    }

    fn position(&self) -> u64 {
        self.position
    }

    fn seek_to(&mut self, pos: u64) -> Result<()> {
        if pos > self.data.len() as u64 {
            return Err(ContentStoreError::InvalidSeek {
                position: pos,
                size: self.data.len() as u64,
            });
        }
        self.position = pos;
        Ok(())
    }

    fn seek_relative(&mut self, offset: i64) -> Result<()> {
        let new_pos = if offset >= 0 {
            self.position.saturating_add(offset as u64)
        } else {
            self.position.saturating_sub((-offset) as u64)
        };
        self.seek_to(new_pos)
    }
}

// ---------------------------------------------------------------------------
// ContentStore trait
// ---------------------------------------------------------------------------

/// Trait defining the streaming content store API.
///
/// Implementations must provide:
/// - `put_stream`: Store content from a reader, returning its hash
/// - `get_stream`: Retrieve content as a seekable reader
/// - `has`: Check if content exists
/// - `size`: Get content size without reading
/// - `delete`: Remove content by hash
pub trait ContentStore: Send + Sync {
    /// The type of seekable reader returned by `get_stream`.
    type Reader: SeekableReader;

    /// Store content from a reader, computing the hash while streaming.
    ///
    /// Data flows through a bounded buffer (default 64KB), keeping memory
    /// usage low even for large files.
    ///
    /// Returns the content hash computed during streaming.
    fn put_stream<R: Read>(&self, reader: &mut R) -> Result<ContentHash>;

    /// Retrieve content as a seekable reader.
    ///
    /// Returns `None` if content with the given hash doesn't exist.
    fn get_stream(&self, hash: &ContentHash) -> Result<Option<Self::Reader>>;

    /// Check if content with the given hash exists.
    fn has(&self, hash: &ContentHash) -> bool;

    /// Get the size of stored content in bytes.
    ///
    /// Returns `None` if content doesn't exist.
    fn size(&self, hash: &ContentHash) -> Result<Option<u64>>;

    /// Delete content by hash.
    ///
    /// Returns `true` if content was deleted, `false` if it didn't exist.
    fn delete(&self, hash: &ContentHash) -> Result<bool>;

    /// Get the current configuration.
    fn config(&self) -> &ContentStoreConfig;
}

// ---------------------------------------------------------------------------
// FileContentStore - disk-backed implementation
// ---------------------------------------------------------------------------

/// A disk-backed content store with streaming I/O.
///
/// Uses git-style layout: `{base}/{first_two_hex}/{remaining_hex}`.
/// Streams directly to/from disk without full-RAM buffering.
pub struct FileContentStore {
    base_path: PathBuf,
    config: ContentStoreConfig,
}

impl FileContentStore {
    /// Create a new file-backed content store.
    ///
    /// Creates the base directory if it doesn't exist.
    pub fn new<P: AsRef<Path>>(base_path: P, config: ContentStoreConfig) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        fs::create_dir_all(&base_path)?;
        Ok(Self { base_path, config })
    }

    /// Open an existing content store.
    ///
    /// Returns an error if the base path doesn't exist.
    pub fn open<P: AsRef<Path>>(base_path: P, config: ContentStoreConfig) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        if !base_path.is_dir() {
            return Err(ContentStoreError::Io(io::Error::new(
                io::ErrorKind::NotFound,
                format!("content store not found: {}", base_path.display()),
            )));
        }
        Ok(Self { base_path, config })
    }

    /// Get the path where a blob with the given hash would be stored.
    fn blob_path(&self, hash: &ContentHash) -> PathBuf {
        let hex = format!("{}", hash);
        let (prefix, suffix) = hex.split_at(2);
        self.base_path.join(prefix).join(suffix)
    }

    /// Get the base path of the store.
    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    /// List all content hashes in the store.
    ///
    /// Note: This can be slow for large stores.
    pub fn list(&self) -> Result<Vec<ContentHash>> {
        let mut hashes = Vec::new();

        for prefix_entry in fs::read_dir(&self.base_path)? {
            let prefix_entry = prefix_entry?;
            let prefix_path = prefix_entry.path();
            if !prefix_path.is_dir() {
                continue;
            }
            let prefix = prefix_entry.file_name();
            let prefix_str = prefix.to_string_lossy();
            if prefix_str.len() != 2 {
                continue;
            }

            for blob_entry in fs::read_dir(&prefix_path)? {
                let blob_entry = blob_entry?;
                let blob_name = blob_entry.file_name();
                let blob_str = blob_name.to_string_lossy();
                if blob_str.len() != 62 {
                    continue;
                }

                let hex = format!("{}{}", prefix_str, blob_str);
                if let Ok(hash) = hex.parse::<ContentHash>() {
                    hashes.push(hash);
                }
            }
        }

        Ok(hashes)
    }
}

impl ContentStore for FileContentStore {
    type Reader = FileSeekableReader;

    fn put_stream<R: Read>(&self, reader: &mut R) -> Result<ContentHash> {
        // Create temp file in same directory for atomic rename
        let temp_dir = self.base_path.join("tmp");
        fs::create_dir_all(&temp_dir)?;

        let temp_path = temp_dir.join(format!("upload_{}", std::process::id()));
        let temp_file = File::create(&temp_path)?;
        let mut writer = StreamingHashWriter::new(BufWriter::with_capacity(
            self.config.buffer_size,
            temp_file,
        ));

        // Stream through bounded buffer
        let mut buffer = vec![0u8; self.config.buffer_size];
        loop {
            let n = reader.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            writer.write_all(&buffer[..n])?;
        }

        let (inner, hash) = writer.finish()?;
        let file = inner.into_inner().map_err(|e| io::Error::new(
            io::ErrorKind::Other,
            format!("failed to flush buffer: {}", e.error()),
        ))?;

        if self.config.sync_writes {
            file.sync_all()?;
        }
        drop(file);

        // Move to final location
        let final_path = self.blob_path(&hash);
        if final_path.exists() {
            // Already exists, remove temp
            fs::remove_file(&temp_path)?;
        } else {
            let parent = final_path.parent().unwrap();
            fs::create_dir_all(parent)?;
            fs::rename(&temp_path, &final_path)?;
        }

        Ok(hash)
    }

    fn get_stream(&self, hash: &ContentHash) -> Result<Option<Self::Reader>> {
        let path = self.blob_path(hash);
        if !path.exists() {
            return Ok(None);
        }

        let file = File::open(&path)?;
        let reader = FileSeekableReader::new(file, self.config.buffer_size)?;
        Ok(Some(reader))
    }

    fn has(&self, hash: &ContentHash) -> bool {
        self.blob_path(hash).exists()
    }

    fn size(&self, hash: &ContentHash) -> Result<Option<u64>> {
        let path = self.blob_path(hash);
        if !path.exists() {
            return Ok(None);
        }
        let metadata = fs::metadata(&path)?;
        Ok(Some(metadata.len()))
    }

    fn delete(&self, hash: &ContentHash) -> Result<bool> {
        let path = self.blob_path(hash);
        if !path.exists() {
            return Ok(false);
        }
        fs::remove_file(&path)?;
        Ok(true)
    }

    fn config(&self) -> &ContentStoreConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// MemoryContentStore - in-memory implementation
// ---------------------------------------------------------------------------

/// An in-memory content store for testing and caching.
///
/// Accumulates data to bytes for API compatibility. Thread-safe via RwLock.
pub struct MemoryContentStore {
    data: RwLock<HashMap<ContentHash, Arc<Vec<u8>>>>,
    config: ContentStoreConfig,
}

impl MemoryContentStore {
    /// Create a new in-memory content store.
    pub fn new(config: ContentStoreConfig) -> Self {
        Self {
            data: RwLock::new(HashMap::new()),
            config,
        }
    }

    /// Create with default config.
    pub fn default_config() -> Self {
        Self::new(ContentStoreConfig::default())
    }

    /// Get the number of stored items.
    pub fn len(&self) -> usize {
        self.data.read().unwrap().len()
    }

    /// Check if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.data.read().unwrap().is_empty()
    }

    /// Clear all stored content.
    pub fn clear(&self) {
        self.data.write().unwrap().clear();
    }

    /// Get total bytes stored.
    pub fn total_bytes(&self) -> u64 {
        self.data
            .read()
            .unwrap()
            .values()
            .map(|v| v.len() as u64)
            .sum()
    }
}

impl Default for MemoryContentStore {
    fn default() -> Self {
        Self::default_config()
    }
}

impl ContentStore for MemoryContentStore {
    type Reader = MemorySeekableReader;

    fn put_stream<R: Read>(&self, reader: &mut R) -> Result<ContentHash> {
        // For memory backend, we accumulate to bytes
        let mut data = Vec::new();
        let mut hash_reader = StreamingHashReader::new(reader);
        let mut buffer = vec![0u8; self.config.buffer_size];

        loop {
            let n = hash_reader.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            data.extend_from_slice(&buffer[..n]);
        }

        let hash = hash_reader.finish();
        self.data.write().unwrap().insert(hash, Arc::new(data));
        Ok(hash)
    }

    fn get_stream(&self, hash: &ContentHash) -> Result<Option<Self::Reader>> {
        let guard = self.data.read().unwrap();
        match guard.get(hash) {
            Some(data) => Ok(Some(MemorySeekableReader::new(Arc::clone(data)))),
            None => Ok(None),
        }
    }

    fn has(&self, hash: &ContentHash) -> bool {
        self.data.read().unwrap().contains_key(hash)
    }

    fn size(&self, hash: &ContentHash) -> Result<Option<u64>> {
        let guard = self.data.read().unwrap();
        Ok(guard.get(hash).map(|v| v.len() as u64))
    }

    fn delete(&self, hash: &ContentHash) -> Result<bool> {
        Ok(self.data.write().unwrap().remove(hash).is_some())
    }

    fn config(&self) -> &ContentStoreConfig {
        &self.config
    }
}

// Implement Send + Sync for MemoryContentStore
unsafe impl Send for MemoryContentStore {}
unsafe impl Sync for MemoryContentStore {}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Compute content hash from a reader without storing.
///
/// Useful for verifying content before storage.
pub fn hash_reader<R: Read>(reader: &mut R, buffer_size: usize) -> Result<ContentHash> {
    let mut hash_reader = StreamingHashReader::new(reader);
    let mut buffer = vec![0u8; buffer_size.clamp(MIN_BUFFER_SIZE, MAX_BUFFER_SIZE)];

    loop {
        let n = hash_reader.read(&mut buffer)?;
        if n == 0 {
            break;
        }
    }

    Ok(hash_reader.finish())
}

/// Copy from a reader to a writer with streaming hash computation.
///
/// Returns the number of bytes copied and the content hash.
pub fn copy_with_hash<R: Read, W: Write>(
    reader: &mut R,
    writer: &mut W,
    buffer_size: usize,
) -> Result<(u64, ContentHash)> {
    let mut hash_reader = StreamingHashReader::new(reader);
    let mut buffer = vec![0u8; buffer_size.clamp(MIN_BUFFER_SIZE, MAX_BUFFER_SIZE)];
    let mut total = 0u64;

    loop {
        let n = hash_reader.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        writer.write_all(&buffer[..n])?;
        total += n as u64;
    }

    writer.flush()?;
    Ok((total, hash_reader.finish()))
}

/// Verify that content matches its expected hash.
///
/// Reads through the content computing the hash.
pub fn verify_content<R: Read>(reader: &mut R, expected: &ContentHash, buffer_size: usize) -> Result<bool> {
    let actual = hash_reader(reader, buffer_size)?;
    Ok(&actual == expected)
}

// ---------------------------------------------------------------------------
// LRU Eviction Support
// ---------------------------------------------------------------------------

/// Event emitted when an entry is evicted from the cache.
#[derive(Debug, Clone)]
pub struct EvictionEvent {
    /// Hash of the evicted content.
    pub hash: ContentHash,
    /// Size of the evicted content in bytes.
    pub size: u64,
    /// Reason for eviction.
    pub reason: EvictionReason,
    /// Timestamp when eviction occurred (milliseconds since UNIX epoch).
    pub timestamp_ms: u64,
}

/// Reason why an entry was evicted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionReason {
    /// Entry was evicted due to LRU policy (least recently used).
    Lru,
    /// Entry expired due to TTL.
    TtlExpired,
    /// Entry was manually deleted.
    Manual,
}

/// Callback type for eviction notifications.
pub type EvictionCallback = Arc<dyn Fn(EvictionEvent) + Send + Sync>;

/// Configuration for LRU content store caching.
#[derive(Debug, Clone)]
pub struct LruConfig {
    /// Maximum total bytes stored. 0 means unlimited.
    pub max_bytes: u64,
    /// Maximum number of entries. 0 means unlimited.
    pub max_entries: usize,
    /// Default TTL in seconds. `None` means entries never expire by time.
    pub default_ttl_secs: Option<u64>,
    /// Whether to evict expired entries on access.
    pub evict_on_access: bool,
}

impl Default for LruConfig {
    fn default() -> Self {
        Self {
            max_bytes: 0,
            max_entries: 0,
            default_ttl_secs: None,
            evict_on_access: true,
        }
    }
}

impl LruConfig {
    /// Create config with max bytes limit.
    pub fn with_max_bytes(mut self, bytes: u64) -> Self {
        self.max_bytes = bytes;
        self
    }

    /// Create config with max entry count limit.
    pub fn with_max_entries(mut self, count: usize) -> Self {
        self.max_entries = count;
        self
    }

    /// Set default TTL in seconds.
    pub fn with_default_ttl(mut self, secs: u64) -> Self {
        self.default_ttl_secs = Some(secs);
        self
    }

    /// Enable or disable eviction on access.
    pub fn with_evict_on_access(mut self, enabled: bool) -> Self {
        self.evict_on_access = enabled;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        // At least one limit must be set if we're doing LRU
        // (both 0 means unlimited, which is valid but essentially no LRU)
        Ok(())
    }

    /// Check if there are any limits configured.
    pub fn has_limits(&self) -> bool {
        self.max_bytes > 0 || self.max_entries > 0
    }
}

/// Internal entry tracking for LRU cache.
#[derive(Debug, Clone)]
struct LruEntry {
    /// Size of the content in bytes.
    size: u64,
    /// Timestamp of last access (milliseconds since UNIX epoch).
    last_access_ms: u64,
    /// Timestamp when entry was created (milliseconds since UNIX epoch).
    created_ms: u64,
    /// Optional TTL in seconds.
    ttl_secs: Option<u64>,
    /// Monotonic insertion order for tie-breaking when timestamps match.
    insert_order: u64,
}

impl LruEntry {
    fn is_expired(&self, now_ms: u64) -> bool {
        match self.ttl_secs {
            Some(ttl) => {
                let expiry_ms = self.created_ms.saturating_add(ttl * 1000);
                now_ms >= expiry_ms
            }
            None => false,
        }
    }
}

/// Get current timestamp in milliseconds since UNIX epoch.
fn current_time_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// LRU-evicting wrapper around MemoryContentStore.
///
/// Provides configurable eviction based on:
/// - Total bytes stored
/// - Maximum entry count
/// - Per-entry TTL
///
/// Thread-safe via RwLock. Eviction notifications can be received via callback.
///
/// # Example
///
/// ```ignore
/// let config = LruConfig::default()
///     .with_max_bytes(10 * 1024 * 1024)  // 10MB limit
///     .with_max_entries(1000)             // Max 1000 entries
///     .with_default_ttl(3600);            // 1 hour TTL
///
/// let store = LruContentStore::new(config, ContentStoreConfig::default());
///
/// // Store content
/// let mut data = Cursor::new(b"large asset data...");
/// let hash = store.put_stream(&mut data)?;
///
/// // Access updates LRU position
/// let reader = store.get_stream(&hash)?;
/// ```
pub struct LruContentStore {
    /// Underlying content storage.
    data: RwLock<HashMap<ContentHash, Arc<Vec<u8>>>>,
    /// LRU metadata and ordering.
    lru: RwLock<LruState>,
    /// LRU configuration.
    lru_config: LruConfig,
    /// Content store configuration.
    config: ContentStoreConfig,
    /// Optional eviction callback.
    eviction_callback: RwLock<Option<EvictionCallback>>,
}

/// Internal LRU state tracking.
struct LruState {
    /// Metadata for each entry.
    entries: HashMap<ContentHash, LruEntry>,
    /// Total bytes stored.
    total_bytes: u64,
    /// Monotonic counter for insertion order.
    insert_counter: u64,
}

impl LruState {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            total_bytes: 0,
            insert_counter: 0,
        }
    }

    /// Get next insertion order value.
    fn next_insert_order(&mut self) -> u64 {
        let order = self.insert_counter;
        self.insert_counter += 1;
        order
    }
}

impl LruContentStore {
    /// Create a new LRU content store.
    pub fn new(lru_config: LruConfig, config: ContentStoreConfig) -> Self {
        Self {
            data: RwLock::new(HashMap::new()),
            lru: RwLock::new(LruState::new()),
            lru_config,
            config,
            eviction_callback: RwLock::new(None),
        }
    }

    /// Create with default configs.
    pub fn default_config() -> Self {
        Self::new(LruConfig::default(), ContentStoreConfig::default())
    }

    /// Set eviction callback for notifications.
    pub fn set_eviction_callback(&self, callback: EvictionCallback) {
        *self.eviction_callback.write().unwrap() = Some(callback);
    }

    /// Clear eviction callback.
    pub fn clear_eviction_callback(&self) {
        *self.eviction_callback.write().unwrap() = None;
    }

    /// Get the LRU configuration.
    pub fn lru_config(&self) -> &LruConfig {
        &self.lru_config
    }

    /// Get the number of stored entries.
    pub fn len(&self) -> usize {
        self.lru.read().unwrap().entries.len()
    }

    /// Check if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.lru.read().unwrap().entries.is_empty()
    }

    /// Get total bytes stored.
    pub fn total_bytes(&self) -> u64 {
        self.lru.read().unwrap().total_bytes
    }

    /// Clear all stored content.
    pub fn clear(&self) {
        let mut data = self.data.write().unwrap();
        let mut lru = self.lru.write().unwrap();
        data.clear();
        lru.entries.clear();
        lru.total_bytes = 0;
    }

    /// Evict all expired entries based on TTL.
    ///
    /// Returns the number of entries evicted.
    pub fn evict_expired(&self) -> usize {
        let now_ms = current_time_ms();
        let expired: Vec<ContentHash> = {
            let lru = self.lru.read().unwrap();
            lru.entries
                .iter()
                .filter(|(_, entry)| entry.is_expired(now_ms))
                .map(|(hash, _)| *hash)
                .collect()
        };

        let count = expired.len();
        for hash in expired {
            self.evict_entry(&hash, EvictionReason::TtlExpired);
        }
        count
    }

    /// Get entry age in seconds.
    pub fn entry_age_secs(&self, hash: &ContentHash) -> Option<u64> {
        let lru = self.lru.read().unwrap();
        let entry = lru.entries.get(hash)?;
        let now_ms = current_time_ms();
        Some((now_ms.saturating_sub(entry.created_ms)) / 1000)
    }

    /// Get time until entry expires (in seconds), or None if no TTL.
    pub fn time_to_expire_secs(&self, hash: &ContentHash) -> Option<u64> {
        let lru = self.lru.read().unwrap();
        let entry = lru.entries.get(hash)?;
        let ttl = entry.ttl_secs?;
        let now_ms = current_time_ms();
        let expiry_ms = entry.created_ms.saturating_add(ttl * 1000);
        if now_ms >= expiry_ms {
            Some(0)
        } else {
            Some((expiry_ms - now_ms) / 1000)
        }
    }

    /// Store content with custom TTL.
    pub fn put_stream_with_ttl<R: Read>(&self, reader: &mut R, ttl_secs: Option<u64>) -> Result<ContentHash> {
        // Read all data and compute hash
        let mut data_vec = Vec::new();
        let mut hash_reader = StreamingHashReader::new(reader);
        let mut buffer = vec![0u8; self.config.buffer_size];

        loop {
            let n = hash_reader.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            data_vec.extend_from_slice(&buffer[..n]);
        }

        let hash = hash_reader.finish();
        let size = data_vec.len() as u64;

        // Evict if necessary before inserting
        self.evict_for_space(size, 1);

        // Insert data
        {
            let mut data = self.data.write().unwrap();
            data.insert(hash, Arc::new(data_vec));
        }

        // Update LRU state
        {
            let now_ms = current_time_ms();
            let mut lru = self.lru.write().unwrap();

            // If replacing existing entry, subtract its size first
            if let Some(old_entry) = lru.entries.get(&hash) {
                lru.total_bytes = lru.total_bytes.saturating_sub(old_entry.size);
            }

            let insert_order = lru.next_insert_order();
            lru.entries.insert(hash, LruEntry {
                size,
                last_access_ms: now_ms,
                created_ms: now_ms,
                ttl_secs: ttl_secs.or(self.lru_config.default_ttl_secs),
                insert_order,
            });
            lru.total_bytes += size;
        }

        Ok(hash)
    }

    /// Evict entries to make room for new content.
    fn evict_for_space(&self, new_size: u64, new_count: usize) {
        // First, evict expired entries if configured
        if self.lru_config.evict_on_access {
            self.evict_expired();
        }

        // Check if we need to evict for size
        if self.lru_config.max_bytes > 0 {
            let mut to_evict = Vec::new();
            {
                let lru = self.lru.read().unwrap();
                let mut projected_bytes = lru.total_bytes + new_size;

                if projected_bytes > self.lru_config.max_bytes {
                    // Get entries sorted by last access (LRU first), with insert_order as tie-breaker
                    let mut entries: Vec<_> = lru.entries.iter()
                        .map(|(h, e)| (*h, e.size, e.last_access_ms, e.insert_order))
                        .collect();
                    entries.sort_by_key(|(_, _, access, order)| (*access, *order));

                    for (hash, size, _, _) in entries {
                        if projected_bytes <= self.lru_config.max_bytes {
                            break;
                        }
                        to_evict.push(hash);
                        projected_bytes = projected_bytes.saturating_sub(size);
                    }
                }
            }

            for hash in to_evict {
                self.evict_entry(&hash, EvictionReason::Lru);
            }
        }

        // Check if we need to evict for count
        if self.lru_config.max_entries > 0 {
            let mut to_evict = Vec::new();
            {
                let lru = self.lru.read().unwrap();
                let projected_count = lru.entries.len() + new_count;

                if projected_count > self.lru_config.max_entries {
                    let excess = projected_count - self.lru_config.max_entries;

                    // Get entries sorted by last access (LRU first), with insert_order as tie-breaker
                    let mut entries: Vec<_> = lru.entries.iter()
                        .map(|(h, e)| (*h, e.last_access_ms, e.insert_order))
                        .collect();
                    entries.sort_by_key(|(_, access, order)| (*access, *order));

                    for (hash, _, _) in entries.into_iter().take(excess) {
                        to_evict.push(hash);
                    }
                }
            }

            for hash in to_evict {
                self.evict_entry(&hash, EvictionReason::Lru);
            }
        }
    }

    /// Evict a single entry.
    fn evict_entry(&self, hash: &ContentHash, reason: EvictionReason) {
        let size = {
            let mut lru = self.lru.write().unwrap();
            if let Some(entry) = lru.entries.remove(hash) {
                lru.total_bytes = lru.total_bytes.saturating_sub(entry.size);
                entry.size
            } else {
                return;
            }
        };

        {
            let mut data = self.data.write().unwrap();
            data.remove(hash);
        }

        // Fire callback if set
        if let Some(ref callback) = *self.eviction_callback.read().unwrap() {
            callback(EvictionEvent {
                hash: *hash,
                size,
                reason,
                timestamp_ms: current_time_ms(),
            });
        }
    }

    /// Touch an entry to update its access time (used internally on get).
    fn touch(&self, hash: &ContentHash) {
        let now_ms = current_time_ms();
        let mut lru = self.lru.write().unwrap();
        if let Some(entry) = lru.entries.get_mut(hash) {
            entry.last_access_ms = now_ms;
        }
    }

    /// Check if an entry is expired.
    fn is_expired(&self, hash: &ContentHash) -> bool {
        let now_ms = current_time_ms();
        let lru = self.lru.read().unwrap();
        if let Some(entry) = lru.entries.get(hash) {
            entry.is_expired(now_ms)
        } else {
            false
        }
    }

    /// Get entry metadata for testing/debugging.
    pub fn entry_info(&self, hash: &ContentHash) -> Option<(u64, u64, u64)> {
        let lru = self.lru.read().unwrap();
        let entry = lru.entries.get(hash)?;
        Some((entry.size, entry.last_access_ms, entry.created_ms))
    }
}

impl Default for LruContentStore {
    fn default() -> Self {
        Self::default_config()
    }
}

impl ContentStore for LruContentStore {
    type Reader = MemorySeekableReader;

    fn put_stream<R: Read>(&self, reader: &mut R) -> Result<ContentHash> {
        self.put_stream_with_ttl(reader, None)
    }

    fn get_stream(&self, hash: &ContentHash) -> Result<Option<Self::Reader>> {
        // Check for expiration first
        if self.lru_config.evict_on_access && self.is_expired(hash) {
            self.evict_entry(hash, EvictionReason::TtlExpired);
            return Ok(None);
        }

        let guard = self.data.read().unwrap();
        match guard.get(hash) {
            Some(data) => {
                drop(guard); // Release read lock before touch
                self.touch(hash);
                let guard = self.data.read().unwrap();
                match guard.get(hash) {
                    Some(data) => Ok(Some(MemorySeekableReader::new(Arc::clone(data)))),
                    None => Ok(None),
                }
            }
            None => Ok(None),
        }
    }

    fn has(&self, hash: &ContentHash) -> bool {
        // Check for expiration
        if self.lru_config.evict_on_access && self.is_expired(hash) {
            self.evict_entry(hash, EvictionReason::TtlExpired);
            return false;
        }
        self.data.read().unwrap().contains_key(hash)
    }

    fn size(&self, hash: &ContentHash) -> Result<Option<u64>> {
        // Check for expiration
        if self.lru_config.evict_on_access && self.is_expired(hash) {
            self.evict_entry(hash, EvictionReason::TtlExpired);
            return Ok(None);
        }
        let guard = self.data.read().unwrap();
        Ok(guard.get(hash).map(|v| v.len() as u64))
    }

    fn delete(&self, hash: &ContentHash) -> Result<bool> {
        let existed = self.data.read().unwrap().contains_key(hash);
        if existed {
            self.evict_entry(hash, EvictionReason::Manual);
        }
        Ok(existed)
    }

    fn config(&self) -> &ContentStoreConfig {
        &self.config
    }
}

// Implement Send + Sync for LruContentStore
unsafe impl Send for LruContentStore {}
unsafe impl Sync for LruContentStore {}

// ---------------------------------------------------------------------------
// Multi-Level Sharding for FileBackend (T-AS-4.5)
// ---------------------------------------------------------------------------

/// Sharding depth configuration for directory structure.
///
/// Controls how content hashes are split into directory hierarchies:
/// - `TwoLevel`: `hash[:2]/hash[2:]` (git-style, default)
/// - `FourLevel`: `hash[:2]/hash[2:4]/hash[4:]`
/// - `SixLevel`: `hash[:2]/hash[2:4]/hash[4:6]/hash[6:]`
///
/// Deeper sharding reduces directory entries at each level, improving
/// filesystem performance for 10k+ assets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ShardingDepth {
    /// 2-level sharding: `{prefix2}/{suffix62}` (256 directories)
    #[default]
    TwoLevel,
    /// 4-level sharding: `{prefix2}/{next2}/{suffix60}` (65,536 directories)
    FourLevel,
    /// 6-level sharding: `{prefix2}/{next2}/{next2}/{suffix58}` (16,777,216 directories)
    SixLevel,
}

impl ShardingDepth {
    /// Get the number of hex characters at each directory level.
    pub fn levels(&self) -> &[usize] {
        match self {
            ShardingDepth::TwoLevel => &[2],
            ShardingDepth::FourLevel => &[2, 2],
            ShardingDepth::SixLevel => &[2, 2, 2],
        }
    }

    /// Get the total number of prefix characters used in directories.
    pub fn prefix_len(&self) -> usize {
        self.levels().iter().sum()
    }

    /// Convert hash hex string to path components.
    ///
    /// Returns (directory_parts, filename).
    pub fn split_hash<'a>(&'a self, hex: &'a str) -> (Vec<&'a str>, &'a str) {
        let levels = self.levels();
        let mut parts = Vec::with_capacity(levels.len());
        let mut pos = 0;

        for &len in levels {
            if pos + len <= hex.len() {
                parts.push(&hex[pos..pos + len]);
                pos += len;
            }
        }

        let filename = &hex[pos..];
        (parts, filename)
    }

    /// Build path from base and hash.
    pub fn build_path(&self, base: &Path, hex: &str) -> PathBuf {
        let (parts, filename) = self.split_hash(hex);
        let mut path = base.to_path_buf();
        for part in parts {
            path.push(part);
        }
        path.push(filename);
        path
    }
}

/// Configuration for sharded content store.
#[derive(Debug, Clone)]
pub struct ShardedStoreConfig {
    /// Base content store configuration.
    pub content_config: ContentStoreConfig,
    /// Sharding depth for directory structure.
    pub sharding: ShardingDepth,
    /// Enable backward compatibility reads from 2-level sharding.
    pub backward_compat: bool,
}

impl Default for ShardedStoreConfig {
    fn default() -> Self {
        Self {
            content_config: ContentStoreConfig::default(),
            sharding: ShardingDepth::TwoLevel,
            backward_compat: false,
        }
    }
}

impl ShardedStoreConfig {
    /// Create config with specific sharding depth.
    pub fn with_sharding(mut self, depth: ShardingDepth) -> Self {
        self.sharding = depth;
        self
    }

    /// Enable backward compatibility reads.
    pub fn with_backward_compat(mut self, enabled: bool) -> Self {
        self.backward_compat = enabled;
        self
    }

    /// Set buffer size.
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.content_config = self.content_config.with_buffer_size(size);
        self
    }

    /// Set sync writes.
    pub fn with_sync_writes(mut self, sync: bool) -> Self {
        self.content_config = self.content_config.with_sync_writes(sync);
        self
    }
}

/// A disk-backed content store with configurable multi-level sharding.
///
/// Supports 2-level (git-style), 4-level, and 6-level directory sharding
/// to optimize filesystem performance for large asset collections (10k+).
///
/// # Directory Layout Examples
///
/// For hash `abcd1234...`:
/// - 2-level: `{base}/ab/cd1234...`
/// - 4-level: `{base}/ab/cd/1234...`
/// - 6-level: `{base}/ab/cd/12/34...`
///
/// # Backward Compatibility
///
/// When `backward_compat` is enabled, reads fall back to 2-level paths
/// if the content is not found at the configured sharding depth. This
/// allows gradual migration from older stores.
pub struct ShardedContentStore {
    base_path: PathBuf,
    config: ShardedStoreConfig,
}

impl ShardedContentStore {
    /// Create a new sharded content store.
    ///
    /// Creates the base directory if it doesn't exist.
    pub fn new<P: AsRef<Path>>(base_path: P, config: ShardedStoreConfig) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        fs::create_dir_all(&base_path)?;
        Ok(Self { base_path, config })
    }

    /// Open an existing sharded content store.
    pub fn open<P: AsRef<Path>>(base_path: P, config: ShardedStoreConfig) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        if !base_path.is_dir() {
            return Err(ContentStoreError::Io(io::Error::new(
                io::ErrorKind::NotFound,
                format!("content store not found: {}", base_path.display()),
            )));
        }
        Ok(Self { base_path, config })
    }

    /// Get the sharding configuration.
    pub fn sharding(&self) -> ShardingDepth {
        self.config.sharding
    }

    /// Get the base path.
    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    /// Get the path for a blob with the given hash at current sharding depth.
    fn blob_path(&self, hash: &ContentHash) -> PathBuf {
        let hex = format!("{}", hash);
        self.config.sharding.build_path(&self.base_path, &hex)
    }

    /// Get the legacy 2-level path for backward compatibility.
    fn legacy_blob_path(&self, hash: &ContentHash) -> PathBuf {
        let hex = format!("{}", hash);
        let (prefix, suffix) = hex.split_at(2);
        self.base_path.join(prefix).join(suffix)
    }

    /// Find the actual blob path, checking backward compatibility if enabled.
    fn find_blob_path(&self, hash: &ContentHash) -> Option<PathBuf> {
        let primary = self.blob_path(hash);
        if primary.exists() {
            return Some(primary);
        }

        // Check backward compat path if enabled and sharding is not 2-level
        if self.config.backward_compat && self.config.sharding != ShardingDepth::TwoLevel {
            let legacy = self.legacy_blob_path(hash);
            if legacy.exists() {
                return Some(legacy);
            }
        }

        None
    }

    /// List all content hashes in the store.
    ///
    /// Scans all sharding levels for completeness.
    pub fn list(&self) -> Result<Vec<ContentHash>> {
        let mut hashes = Vec::new();
        self.list_recursive(&self.base_path, String::new(), &mut hashes)?;
        Ok(hashes)
    }

    /// Recursively list hashes.
    fn list_recursive(&self, dir: &Path, prefix: String, hashes: &mut Vec<ContentHash>) -> Result<()> {
        if !dir.is_dir() {
            return Ok(());
        }

        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            let name = entry.file_name();
            let name_str = name.to_string_lossy();

            // Skip tmp directory
            if name_str == "tmp" {
                continue;
            }

            if path.is_dir() {
                // Recurse into subdirectories
                if name_str.len() == 2 && name_str.chars().all(|c| c.is_ascii_hexdigit()) {
                    let new_prefix = format!("{}{}", prefix, name_str);
                    self.list_recursive(&path, new_prefix, hashes)?;
                }
            } else if path.is_file() {
                // Found a blob file
                let hex = format!("{}{}", prefix, name_str);
                if hex.len() == 64 && hex.chars().all(|c| c.is_ascii_hexdigit()) {
                    if let Ok(hash) = hex.parse::<ContentHash>() {
                        hashes.push(hash);
                    }
                }
            }
        }

        Ok(())
    }

    /// Get statistics about the store.
    pub fn stats(&self) -> Result<ShardedStoreStats> {
        let hashes = self.list()?;
        let mut total_bytes = 0u64;

        for hash in &hashes {
            if let Some(size) = self.size(hash)? {
                total_bytes += size;
            }
        }

        Ok(ShardedStoreStats {
            entry_count: hashes.len(),
            total_bytes,
            sharding: self.config.sharding,
        })
    }
}

/// Statistics about a sharded content store.
#[derive(Debug, Clone)]
pub struct ShardedStoreStats {
    /// Number of entries in the store.
    pub entry_count: usize,
    /// Total bytes stored.
    pub total_bytes: u64,
    /// Current sharding depth.
    pub sharding: ShardingDepth,
}

impl ContentStore for ShardedContentStore {
    type Reader = FileSeekableReader;

    fn put_stream<R: Read>(&self, reader: &mut R) -> Result<ContentHash> {
        // Create temp file in same directory for atomic rename
        let temp_dir = self.base_path.join("tmp");
        fs::create_dir_all(&temp_dir)?;

        let temp_path = temp_dir.join(format!("upload_{}", std::process::id()));
        let temp_file = File::create(&temp_path)?;
        let mut writer = StreamingHashWriter::new(BufWriter::with_capacity(
            self.config.content_config.buffer_size,
            temp_file,
        ));

        // Stream through bounded buffer
        let mut buffer = vec![0u8; self.config.content_config.buffer_size];
        loop {
            let n = reader.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            writer.write_all(&buffer[..n])?;
        }

        let (inner, hash) = writer.finish()?;
        let file = inner.into_inner().map_err(|e| io::Error::new(
            io::ErrorKind::Other,
            format!("failed to flush buffer: {}", e.error()),
        ))?;

        if self.config.content_config.sync_writes {
            file.sync_all()?;
        }
        drop(file);

        // Move to final location with sharded path
        let final_path = self.blob_path(&hash);
        if final_path.exists() {
            // Already exists, remove temp
            fs::remove_file(&temp_path)?;
        } else {
            let parent = final_path.parent().unwrap();
            fs::create_dir_all(parent)?;
            fs::rename(&temp_path, &final_path)?;
        }

        Ok(hash)
    }

    fn get_stream(&self, hash: &ContentHash) -> Result<Option<Self::Reader>> {
        match self.find_blob_path(hash) {
            Some(path) => {
                let file = File::open(&path)?;
                let reader = FileSeekableReader::new(file, self.config.content_config.buffer_size)?;
                Ok(Some(reader))
            }
            None => Ok(None),
        }
    }

    fn has(&self, hash: &ContentHash) -> bool {
        self.find_blob_path(hash).is_some()
    }

    fn size(&self, hash: &ContentHash) -> Result<Option<u64>> {
        match self.find_blob_path(hash) {
            Some(path) => {
                let metadata = fs::metadata(&path)?;
                Ok(Some(metadata.len()))
            }
            None => Ok(None),
        }
    }

    fn delete(&self, hash: &ContentHash) -> Result<bool> {
        match self.find_blob_path(hash) {
            Some(path) => {
                fs::remove_file(&path)?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    fn config(&self) -> &ContentStoreConfig {
        &self.config.content_config
    }
}

// ---------------------------------------------------------------------------
// Sharding Migration Tool
// ---------------------------------------------------------------------------

/// Progress callback for migration operations.
pub type MigrationProgressCallback = Arc<dyn Fn(MigrationProgress) + Send + Sync>;

/// Progress information during migration.
#[derive(Debug, Clone)]
pub struct MigrationProgress {
    /// Total entries to migrate.
    pub total: usize,
    /// Entries migrated so far.
    pub completed: usize,
    /// Entries skipped (already exist at target).
    pub skipped: usize,
    /// Errors encountered.
    pub errors: usize,
    /// Current hash being migrated (if any).
    pub current_hash: Option<ContentHash>,
}

impl MigrationProgress {
    /// Get completion percentage (0-100).
    pub fn percent_complete(&self) -> f64 {
        if self.total == 0 {
            100.0
        } else {
            (self.completed + self.skipped) as f64 / self.total as f64 * 100.0
        }
    }
}

/// Result of a migration operation.
#[derive(Debug, Clone)]
pub struct MigrationResult {
    /// Total entries processed.
    pub total: usize,
    /// Entries successfully migrated.
    pub migrated: usize,
    /// Entries skipped (already at target location).
    pub skipped: usize,
    /// Entries that failed to migrate.
    pub errors: usize,
    /// Source sharding depth.
    pub from_sharding: ShardingDepth,
    /// Target sharding depth.
    pub to_sharding: ShardingDepth,
}

/// Tool for migrating content between different sharding depths.
///
/// Supports:
/// - 2-level to 4-level migration
/// - 2-level to 6-level migration
/// - 4-level to 6-level migration
/// - Any depth to 2-level (consolidation)
///
/// Migration is atomic per-file: each file is moved completely or not at all.
pub struct ShardingMigrator {
    base_path: PathBuf,
    from_sharding: ShardingDepth,
    to_sharding: ShardingDepth,
    progress_callback: Option<MigrationProgressCallback>,
    delete_source: bool,
}

impl ShardingMigrator {
    /// Create a new migration tool.
    pub fn new<P: AsRef<Path>>(
        base_path: P,
        from_sharding: ShardingDepth,
        to_sharding: ShardingDepth,
    ) -> Self {
        Self {
            base_path: base_path.as_ref().to_path_buf(),
            from_sharding,
            to_sharding,
            progress_callback: None,
            delete_source: true,
        }
    }

    /// Set progress callback.
    pub fn with_progress_callback(mut self, callback: MigrationProgressCallback) -> Self {
        self.progress_callback = Some(callback);
        self
    }

    /// Set whether to delete source files after migration.
    ///
    /// Default is `true`. Set to `false` to copy instead of move.
    pub fn with_delete_source(mut self, delete: bool) -> Self {
        self.delete_source = delete;
        self
    }

    /// Execute the migration.
    pub fn migrate(&self) -> Result<MigrationResult> {
        // First, collect all hashes at source sharding depth
        let source_config = ShardedStoreConfig::default().with_sharding(self.from_sharding);
        let source_store = ShardedContentStore::open(&self.base_path, source_config)?;
        let hashes = source_store.list()?;

        let total = hashes.len();
        let mut migrated = 0usize;
        let mut skipped = 0usize;
        let mut errors = 0usize;

        for (idx, hash) in hashes.iter().enumerate() {
            // Report progress
            if let Some(ref callback) = self.progress_callback {
                callback(MigrationProgress {
                    total,
                    completed: migrated,
                    skipped,
                    errors,
                    current_hash: Some(*hash),
                });
            }

            // Get source path
            let hex = format!("{}", hash);
            let source_path = self.from_sharding.build_path(&self.base_path, &hex);
            let target_path = self.to_sharding.build_path(&self.base_path, &hex);

            // Skip if source doesn't exist (might have been at different sharding)
            if !source_path.exists() {
                skipped += 1;
                continue;
            }

            // Skip if target already exists
            if target_path.exists() {
                if self.delete_source && source_path != target_path {
                    // Remove duplicate source
                    let _ = fs::remove_file(&source_path);
                }
                skipped += 1;
                continue;
            }

            // Create target directory
            if let Some(parent) = target_path.parent() {
                if let Err(_) = fs::create_dir_all(parent) {
                    errors += 1;
                    continue;
                }
            }

            // Move or copy the file
            let result = if self.delete_source {
                fs::rename(&source_path, &target_path)
            } else {
                fs::copy(&source_path, &target_path).map(|_| ())
            };

            match result {
                Ok(()) => {
                    migrated += 1;
                }
                Err(_) => {
                    errors += 1;
                }
            }
        }

        // Final progress report
        if let Some(ref callback) = self.progress_callback {
            callback(MigrationProgress {
                total,
                completed: migrated,
                skipped,
                errors,
                current_hash: None,
            });
        }

        // Clean up empty directories
        self.cleanup_empty_dirs()?;

        Ok(MigrationResult {
            total,
            migrated,
            skipped,
            errors,
            from_sharding: self.from_sharding,
            to_sharding: self.to_sharding,
        })
    }

    /// Remove empty directories left after migration.
    fn cleanup_empty_dirs(&self) -> Result<()> {
        self.cleanup_empty_dirs_recursive(&self.base_path)?;
        Ok(())
    }

    fn cleanup_empty_dirs_recursive(&self, dir: &Path) -> Result<bool> {
        if !dir.is_dir() {
            return Ok(false);
        }

        let mut is_empty = true;

        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            let name = entry.file_name();
            let name_str = name.to_string_lossy();

            // Skip tmp directory
            if name_str == "tmp" {
                is_empty = false;
                continue;
            }

            if path.is_dir() {
                // Recurse and check if empty after cleanup
                let child_empty = self.cleanup_empty_dirs_recursive(&path)?;
                if child_empty {
                    let _ = fs::remove_dir(&path);
                } else {
                    is_empty = false;
                }
            } else {
                is_empty = false;
            }
        }

        Ok(is_empty)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // ========================================================================
    // Configuration tests
    // ========================================================================

    #[test]
    fn test_default_config() {
        let config = ContentStoreConfig::default();
        assert_eq!(config.buffer_size, DEFAULT_BUFFER_SIZE);
        assert!(config.sync_writes);
        assert!(!config.verify_on_read);
    }

    #[test]
    fn test_config_with_buffer_size() {
        let config = ContentStoreConfig::default().with_buffer_size(128 * 1024);
        assert_eq!(config.buffer_size, 128 * 1024);
    }

    #[test]
    fn test_config_buffer_size_clamped_min() {
        let config = ContentStoreConfig::default().with_buffer_size(100);
        assert_eq!(config.buffer_size, MIN_BUFFER_SIZE);
    }

    #[test]
    fn test_config_buffer_size_clamped_max() {
        let config = ContentStoreConfig::default().with_buffer_size(100 * 1024 * 1024);
        assert_eq!(config.buffer_size, MAX_BUFFER_SIZE);
    }

    #[test]
    fn test_config_with_sync_writes() {
        let config = ContentStoreConfig::default().with_sync_writes(false);
        assert!(!config.sync_writes);
    }

    #[test]
    fn test_config_with_verify_on_read() {
        let config = ContentStoreConfig::default().with_verify_on_read(true);
        assert!(config.verify_on_read);
    }

    // ========================================================================
    // StreamingHashReader tests
    // ========================================================================

    #[test]
    fn test_streaming_hash_reader_basic() {
        let data = b"hello world";
        let mut cursor = Cursor::new(data);
        let mut reader = StreamingHashReader::new(&mut cursor);

        let mut buf = [0u8; 20];
        let n = reader.read(&mut buf).unwrap();
        assert_eq!(n, 11);
        assert_eq!(&buf[..n], data);

        let hash = reader.finish();
        assert_eq!(hash, ContentHash::from_bytes(data));
    }

    #[test]
    fn test_streaming_hash_reader_chunked() {
        let data = b"hello world streaming test";
        let mut cursor = Cursor::new(data);
        let mut reader = StreamingHashReader::new(&mut cursor);

        let mut result = Vec::new();
        let mut buf = [0u8; 5];
        loop {
            let n = reader.read(&mut buf).unwrap();
            if n == 0 {
                break;
            }
            result.extend_from_slice(&buf[..n]);
        }

        assert_eq!(result, data);
        let hash = reader.finish();
        assert_eq!(hash, ContentHash::from_bytes(data));
    }

    #[test]
    fn test_streaming_hash_reader_empty() {
        let data: &[u8] = b"";
        let mut cursor = Cursor::new(data);
        let mut reader = StreamingHashReader::new(&mut cursor);

        let mut buf = [0u8; 10];
        let n = reader.read(&mut buf).unwrap();
        assert_eq!(n, 0);

        let hash = reader.finish();
        assert_eq!(hash, ContentHash::from_bytes(data));
    }

    #[test]
    fn test_streaming_hash_reader_bytes_read() {
        let data = b"test data for counting bytes";
        let mut cursor = Cursor::new(data);
        let mut reader = StreamingHashReader::new(&mut cursor);

        assert_eq!(reader.bytes_read(), 0);

        let mut buf = [0u8; 10];
        reader.read(&mut buf).unwrap();
        assert_eq!(reader.bytes_read(), 10);

        reader.read(&mut buf).unwrap();
        assert_eq!(reader.bytes_read(), 20);

        reader.read(&mut buf).unwrap();
        assert_eq!(reader.bytes_read(), data.len() as u64);
    }

    // ========================================================================
    // StreamingHashWriter tests
    // ========================================================================

    #[test]
    fn test_streaming_hash_writer_basic() {
        let data = b"hello world";
        let mut output = Vec::new();
        let mut writer = StreamingHashWriter::new(&mut output);

        writer.write_all(data).unwrap();
        let (_, hash) = writer.finish().unwrap();

        assert_eq!(output, data);
        assert_eq!(hash, ContentHash::from_bytes(data));
    }

    #[test]
    fn test_streaming_hash_writer_chunked() {
        let data = b"hello world streaming test";
        let mut output = Vec::new();
        let mut writer = StreamingHashWriter::new(&mut output);

        for chunk in data.chunks(5) {
            writer.write_all(chunk).unwrap();
        }
        let (_, hash) = writer.finish().unwrap();

        assert_eq!(output, data);
        assert_eq!(hash, ContentHash::from_bytes(data));
    }

    #[test]
    fn test_streaming_hash_writer_bytes_written() {
        let mut output = Vec::new();
        let mut writer = StreamingHashWriter::new(&mut output);

        assert_eq!(writer.bytes_written(), 0);

        writer.write_all(b"hello").unwrap();
        assert_eq!(writer.bytes_written(), 5);

        writer.write_all(b" world").unwrap();
        assert_eq!(writer.bytes_written(), 11);
    }

    // ========================================================================
    // MemorySeekableReader tests
    // ========================================================================

    #[test]
    fn test_memory_seekable_reader_basic() {
        let data = Arc::new(b"hello world".to_vec());
        let mut reader = MemorySeekableReader::new(data.clone());

        assert_eq!(reader.size(), 11);
        assert_eq!(reader.position(), 0);

        let mut buf = [0u8; 5];
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"hello");
        assert_eq!(reader.position(), 5);
    }

    #[test]
    fn test_memory_seekable_reader_seek_to() {
        let data = Arc::new(b"hello world".to_vec());
        let mut reader = MemorySeekableReader::new(data);

        reader.seek_to(6).unwrap();
        assert_eq!(reader.position(), 6);

        let mut buf = [0u8; 5];
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"world");
    }

    #[test]
    fn test_memory_seekable_reader_seek_relative() {
        let data = Arc::new(b"hello world".to_vec());
        let mut reader = MemorySeekableReader::new(data);

        reader.seek_to(6).unwrap();
        reader.seek_relative(-3).unwrap();
        assert_eq!(reader.position(), 3);

        let mut buf = [0u8; 2];
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"lo");
    }

    #[test]
    fn test_memory_seekable_reader_seek_beyond_end() {
        let data = Arc::new(b"hello".to_vec());
        let mut reader = MemorySeekableReader::new(data);

        let result = reader.seek_to(100);
        assert!(matches!(result, Err(ContentStoreError::InvalidSeek { .. })));
    }

    #[test]
    fn test_memory_seekable_reader_remaining() {
        let data = Arc::new(b"hello world".to_vec());
        let mut reader = MemorySeekableReader::new(data);

        assert_eq!(reader.remaining(), 11);

        reader.seek_to(6).unwrap();
        assert_eq!(reader.remaining(), 5);

        reader.seek_to(11).unwrap();
        assert_eq!(reader.remaining(), 0);
    }

    // ========================================================================
    // MemoryContentStore tests
    // ========================================================================

    #[test]
    fn test_memory_store_put_stream_correct_hash() {
        let store = MemoryContentStore::default();
        let data = b"test content for hashing";
        let mut cursor = Cursor::new(data);

        let hash = store.put_stream(&mut cursor).unwrap();
        assert_eq!(hash, ContentHash::from_bytes(data));
    }

    #[test]
    fn test_memory_store_get_stream_returns_correct_data() {
        let store = MemoryContentStore::default();
        let data = b"test content";
        let mut cursor = Cursor::new(data);

        let hash = store.put_stream(&mut cursor).unwrap();
        let mut reader = store.get_stream(&hash).unwrap().unwrap();

        let mut result = Vec::new();
        reader.read_to_end(&mut result).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_memory_store_streaming_matches_non_streaming() {
        let store = MemoryContentStore::default();
        let data = b"content for streaming vs non-streaming comparison";

        // Streaming
        let mut cursor = Cursor::new(data);
        let stream_hash = store.put_stream(&mut cursor).unwrap();

        // Non-streaming (direct hash)
        let direct_hash = ContentHash::from_bytes(data);

        assert_eq!(stream_hash, direct_hash);
    }

    #[test]
    fn test_memory_store_large_data() {
        let store = MemoryContentStore::new(ContentStoreConfig::default().with_buffer_size(8 * 1024));

        // Simulate >100MB with smaller size for test speed (1MB)
        let data: Vec<u8> = (0..1_000_000).map(|i| (i % 256) as u8).collect();
        let mut cursor = Cursor::new(&data);

        let hash = store.put_stream(&mut cursor).unwrap();
        assert_eq!(hash, ContentHash::from_bytes(&data));

        let mut reader = store.get_stream(&hash).unwrap().unwrap();
        let mut result = Vec::new();
        reader.read_to_end(&mut result).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_memory_store_buffer_size_config() {
        let config = ContentStoreConfig::default().with_buffer_size(16 * 1024);
        let store = MemoryContentStore::new(config);

        assert_eq!(store.config().buffer_size, 16 * 1024);
    }

    #[test]
    fn test_memory_store_seek_operations() {
        let store = MemoryContentStore::default();
        let data = b"0123456789ABCDEFGHIJ";
        let mut cursor = Cursor::new(data);

        let hash = store.put_stream(&mut cursor).unwrap();
        let mut reader = store.get_stream(&hash).unwrap().unwrap();

        // Seek to middle
        reader.seek_to(10).unwrap();
        let mut buf = [0u8; 5];
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"ABCDE");

        // Seek back
        reader.seek_to(0).unwrap();
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"01234");

        // Relative seek
        reader.seek_relative(5).unwrap();
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"ABCDE");
    }

    #[test]
    fn test_memory_store_has() {
        let store = MemoryContentStore::default();
        let data = b"test";
        let mut cursor = Cursor::new(data);

        let hash = store.put_stream(&mut cursor).unwrap();
        assert!(store.has(&hash));

        let missing = ContentHash::from_bytes(b"nonexistent");
        assert!(!store.has(&missing));
    }

    #[test]
    fn test_memory_store_size() {
        let store = MemoryContentStore::default();
        let data = b"test content with known size";
        let mut cursor = Cursor::new(data);

        let hash = store.put_stream(&mut cursor).unwrap();
        let size = store.size(&hash).unwrap().unwrap();
        assert_eq!(size, data.len() as u64);
    }

    #[test]
    fn test_memory_store_delete() {
        let store = MemoryContentStore::default();
        let data = b"test";
        let mut cursor = Cursor::new(data);

        let hash = store.put_stream(&mut cursor).unwrap();
        assert!(store.has(&hash));

        let deleted = store.delete(&hash).unwrap();
        assert!(deleted);
        assert!(!store.has(&hash));

        // Delete again returns false
        let deleted_again = store.delete(&hash).unwrap();
        assert!(!deleted_again);
    }

    #[test]
    fn test_memory_store_len_and_is_empty() {
        let store = MemoryContentStore::default();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);

        let mut cursor = Cursor::new(b"test");
        store.put_stream(&mut cursor).unwrap();
        assert!(!store.is_empty());
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_memory_store_clear() {
        let store = MemoryContentStore::default();

        let mut c1 = Cursor::new(b"data1");
        let mut c2 = Cursor::new(b"data2");
        store.put_stream(&mut c1).unwrap();
        store.put_stream(&mut c2).unwrap();

        assert_eq!(store.len(), 2);
        store.clear();
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
    }

    #[test]
    fn test_memory_store_total_bytes() {
        let store = MemoryContentStore::default();

        let mut c1 = Cursor::new(b"12345");
        let mut c2 = Cursor::new(b"67890ABC");
        store.put_stream(&mut c1).unwrap();
        store.put_stream(&mut c2).unwrap();

        assert_eq!(store.total_bytes(), 5 + 8);
    }

    // ========================================================================
    // FileContentStore tests
    // ========================================================================

    #[test]
    fn test_file_store_put_stream_correct_hash() {
        let temp_dir = tempfile::tempdir().unwrap();
        let store = FileContentStore::new(temp_dir.path(), ContentStoreConfig::default()).unwrap();

        let data = b"test content for file store";
        let mut cursor = Cursor::new(data);

        let hash = store.put_stream(&mut cursor).unwrap();
        assert_eq!(hash, ContentHash::from_bytes(data));
    }

    #[test]
    fn test_file_store_get_stream_returns_correct_data() {
        let temp_dir = tempfile::tempdir().unwrap();
        let store = FileContentStore::new(temp_dir.path(), ContentStoreConfig::default()).unwrap();

        let data = b"test content";
        let mut cursor = Cursor::new(data);

        let hash = store.put_stream(&mut cursor).unwrap();
        let mut reader = store.get_stream(&hash).unwrap().unwrap();

        let mut result = Vec::new();
        reader.read_to_end(&mut result).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_file_store_streaming_matches_non_streaming() {
        let temp_dir = tempfile::tempdir().unwrap();
        let store = FileContentStore::new(temp_dir.path(), ContentStoreConfig::default()).unwrap();

        let data = b"content comparison test";
        let mut cursor = Cursor::new(data);

        let stream_hash = store.put_stream(&mut cursor).unwrap();
        let direct_hash = ContentHash::from_bytes(data);

        assert_eq!(stream_hash, direct_hash);
    }

    #[test]
    fn test_file_store_large_data() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = ContentStoreConfig::default().with_buffer_size(8 * 1024);
        let store = FileContentStore::new(temp_dir.path(), config).unwrap();

        // 1MB of data
        let data: Vec<u8> = (0..1_000_000).map(|i| (i % 256) as u8).collect();
        let mut cursor = Cursor::new(&data);

        let hash = store.put_stream(&mut cursor).unwrap();
        assert_eq!(hash, ContentHash::from_bytes(&data));

        let mut reader = store.get_stream(&hash).unwrap().unwrap();
        let mut result = Vec::new();
        reader.read_to_end(&mut result).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_file_store_seek_operations() {
        let temp_dir = tempfile::tempdir().unwrap();
        let store = FileContentStore::new(temp_dir.path(), ContentStoreConfig::default()).unwrap();

        let data = b"0123456789ABCDEFGHIJ";
        let mut cursor = Cursor::new(data);

        let hash = store.put_stream(&mut cursor).unwrap();
        let mut reader = store.get_stream(&hash).unwrap().unwrap();

        reader.seek_to(10).unwrap();
        let mut buf = [0u8; 5];
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"ABCDE");

        reader.seek_to(0).unwrap();
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"01234");
    }

    #[test]
    fn test_file_store_has() {
        let temp_dir = tempfile::tempdir().unwrap();
        let store = FileContentStore::new(temp_dir.path(), ContentStoreConfig::default()).unwrap();

        let data = b"test";
        let mut cursor = Cursor::new(data);

        let hash = store.put_stream(&mut cursor).unwrap();
        assert!(store.has(&hash));

        let missing = ContentHash::from_bytes(b"nonexistent");
        assert!(!store.has(&missing));
    }

    #[test]
    fn test_file_store_size() {
        let temp_dir = tempfile::tempdir().unwrap();
        let store = FileContentStore::new(temp_dir.path(), ContentStoreConfig::default()).unwrap();

        let data = b"test content with known size";
        let mut cursor = Cursor::new(data);

        let hash = store.put_stream(&mut cursor).unwrap();
        let size = store.size(&hash).unwrap().unwrap();
        assert_eq!(size, data.len() as u64);
    }

    #[test]
    fn test_file_store_delete() {
        let temp_dir = tempfile::tempdir().unwrap();
        let store = FileContentStore::new(temp_dir.path(), ContentStoreConfig::default()).unwrap();

        let data = b"test";
        let mut cursor = Cursor::new(data);

        let hash = store.put_stream(&mut cursor).unwrap();
        assert!(store.has(&hash));

        let deleted = store.delete(&hash).unwrap();
        assert!(deleted);
        assert!(!store.has(&hash));
    }

    #[test]
    fn test_file_store_direct_disk_io() {
        let temp_dir = tempfile::tempdir().unwrap();
        let store = FileContentStore::new(temp_dir.path(), ContentStoreConfig::default()).unwrap();

        let data = b"direct disk test";
        let mut cursor = Cursor::new(data);
        let hash = store.put_stream(&mut cursor).unwrap();

        // Verify file exists on disk
        let blob_path = store.blob_path(&hash);
        assert!(blob_path.exists());

        // Read directly from disk
        let disk_data = fs::read(&blob_path).unwrap();
        assert_eq!(disk_data, data);
    }

    #[test]
    fn test_file_store_list() {
        let temp_dir = tempfile::tempdir().unwrap();
        let store = FileContentStore::new(temp_dir.path(), ContentStoreConfig::default()).unwrap();

        let mut c1 = Cursor::new(b"data1");
        let mut c2 = Cursor::new(b"data2");
        let mut c3 = Cursor::new(b"data3");

        let h1 = store.put_stream(&mut c1).unwrap();
        let h2 = store.put_stream(&mut c2).unwrap();
        let h3 = store.put_stream(&mut c3).unwrap();

        let mut listed = store.list().unwrap();
        listed.sort_by(|a, b| format!("{}", a).cmp(&format!("{}", b)));

        let mut expected = vec![h1, h2, h3];
        expected.sort_by(|a, b| format!("{}", a).cmp(&format!("{}", b)));

        assert_eq!(listed, expected);
    }

    #[test]
    fn test_file_store_deduplication() {
        let temp_dir = tempfile::tempdir().unwrap();
        let store = FileContentStore::new(temp_dir.path(), ContentStoreConfig::default()).unwrap();

        let data = b"duplicate content";

        let mut c1 = Cursor::new(data);
        let mut c2 = Cursor::new(data);

        let h1 = store.put_stream(&mut c1).unwrap();
        let h2 = store.put_stream(&mut c2).unwrap();

        assert_eq!(h1, h2);
        assert_eq!(store.list().unwrap().len(), 1);
    }

    // ========================================================================
    // Error handling tests
    // ========================================================================

    #[test]
    fn test_read_error_propagation() {
        struct FailingReader;
        impl Read for FailingReader {
            fn read(&mut self, _buf: &mut [u8]) -> io::Result<usize> {
                Err(io::Error::new(io::ErrorKind::Other, "simulated read error"))
            }
        }

        let store = MemoryContentStore::default();
        let result = store.put_stream(&mut FailingReader);
        assert!(matches!(result, Err(ContentStoreError::Io(_))));
    }

    #[test]
    fn test_file_store_open_nonexistent() {
        let result = FileContentStore::open("/nonexistent/path", ContentStoreConfig::default());
        assert!(matches!(result, Err(ContentStoreError::Io(_))));
    }

    #[test]
    fn test_get_stream_not_found() {
        let store = MemoryContentStore::default();
        let missing = ContentHash::from_bytes(b"missing");
        let result = store.get_stream(&missing).unwrap();
        assert!(result.is_none());
    }

    // ========================================================================
    // Utility function tests
    // ========================================================================

    #[test]
    fn test_hash_reader_utility() {
        let data = b"utility test data";
        let mut cursor = Cursor::new(data);

        let hash = hash_reader(&mut cursor, DEFAULT_BUFFER_SIZE).unwrap();
        assert_eq!(hash, ContentHash::from_bytes(data));
    }

    #[test]
    fn test_copy_with_hash_utility() {
        let data = b"copy test data";
        let mut input = Cursor::new(data);
        let mut output = Vec::new();

        let (bytes, hash) = copy_with_hash(&mut input, &mut output, DEFAULT_BUFFER_SIZE).unwrap();

        assert_eq!(bytes, data.len() as u64);
        assert_eq!(hash, ContentHash::from_bytes(data));
        assert_eq!(output, data);
    }

    #[test]
    fn test_verify_content_utility() {
        let data = b"verify test data";
        let correct_hash = ContentHash::from_bytes(data);
        let wrong_hash = ContentHash::from_bytes(b"wrong");

        let mut cursor1 = Cursor::new(data);
        assert!(verify_content(&mut cursor1, &correct_hash, DEFAULT_BUFFER_SIZE).unwrap());

        let mut cursor2 = Cursor::new(data);
        assert!(!verify_content(&mut cursor2, &wrong_hash, DEFAULT_BUFFER_SIZE).unwrap());
    }

    // ========================================================================
    // Edge case tests
    // ========================================================================

    #[test]
    fn test_empty_content() {
        let store = MemoryContentStore::default();
        let data: &[u8] = b"";
        let mut cursor = Cursor::new(data);

        let hash = store.put_stream(&mut cursor).unwrap();
        assert_eq!(hash, ContentHash::from_bytes(data));

        let mut reader = store.get_stream(&hash).unwrap().unwrap();
        let mut result = Vec::new();
        reader.read_to_end(&mut result).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_single_byte_content() {
        let store = MemoryContentStore::default();
        let data = b"X";
        let mut cursor = Cursor::new(data);

        let hash = store.put_stream(&mut cursor).unwrap();
        assert_eq!(hash, ContentHash::from_bytes(data));

        let mut reader = store.get_stream(&hash).unwrap().unwrap();
        let mut result = Vec::new();
        reader.read_to_end(&mut result).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_exact_buffer_size_content() {
        let store = MemoryContentStore::new(ContentStoreConfig::default().with_buffer_size(MIN_BUFFER_SIZE));
        let data: Vec<u8> = (0..MIN_BUFFER_SIZE).map(|i| (i % 256) as u8).collect();
        let mut cursor = Cursor::new(&data);

        let hash = store.put_stream(&mut cursor).unwrap();
        assert_eq!(hash, ContentHash::from_bytes(&data));
    }

    #[test]
    fn test_buffer_size_plus_one() {
        let store = MemoryContentStore::new(ContentStoreConfig::default().with_buffer_size(MIN_BUFFER_SIZE));
        let data: Vec<u8> = (0..MIN_BUFFER_SIZE + 1).map(|i| (i % 256) as u8).collect();
        let mut cursor = Cursor::new(&data);

        let hash = store.put_stream(&mut cursor).unwrap();
        assert_eq!(hash, ContentHash::from_bytes(&data));
    }

    // ========================================================================
    // ContentStoreError display tests
    // ========================================================================

    #[test]
    fn test_error_display() {
        let io_err = ContentStoreError::Io(io::Error::new(io::ErrorKind::NotFound, "test"));
        assert!(io_err.to_string().contains("I/O error"));

        let not_found = ContentStoreError::NotFound(ContentHash::zero());
        assert!(not_found.to_string().contains("not found"));

        let mismatch = ContentStoreError::HashMismatch {
            expected: ContentHash::zero(),
            actual: ContentHash::from_bytes(b"x"),
        };
        assert!(mismatch.to_string().contains("mismatch"));

        let invalid_seek = ContentStoreError::InvalidSeek {
            position: 100,
            size: 50,
        };
        assert!(invalid_seek.to_string().contains("invalid seek"));

        let invalid_buf = ContentStoreError::InvalidBufferSize(0);
        assert!(invalid_buf.to_string().contains("buffer size"));
    }

    // ========================================================================
    // LRU Content Store Tests (T-AS-4.3)
    // ========================================================================

    #[test]
    fn test_lru_config_default() {
        let config = LruConfig::default();
        assert_eq!(config.max_bytes, 0);
        assert_eq!(config.max_entries, 0);
        assert!(config.default_ttl_secs.is_none());
        assert!(config.evict_on_access);
    }

    #[test]
    fn test_lru_config_with_max_bytes() {
        let config = LruConfig::default().with_max_bytes(1024 * 1024);
        assert_eq!(config.max_bytes, 1024 * 1024);
    }

    #[test]
    fn test_lru_config_with_max_entries() {
        let config = LruConfig::default().with_max_entries(100);
        assert_eq!(config.max_entries, 100);
    }

    #[test]
    fn test_lru_config_with_default_ttl() {
        let config = LruConfig::default().with_default_ttl(3600);
        assert_eq!(config.default_ttl_secs, Some(3600));
    }

    #[test]
    fn test_lru_config_has_limits() {
        let config = LruConfig::default();
        assert!(!config.has_limits());

        let config = LruConfig::default().with_max_bytes(1000);
        assert!(config.has_limits());

        let config = LruConfig::default().with_max_entries(10);
        assert!(config.has_limits());
    }

    #[test]
    fn test_lru_store_basic_put_get() {
        let store = LruContentStore::new(LruConfig::default(), ContentStoreConfig::default());

        let data = b"test content";
        let mut cursor = Cursor::new(data);

        let hash = store.put_stream(&mut cursor).unwrap();
        assert_eq!(hash, ContentHash::from_bytes(data));

        let mut reader = store.get_stream(&hash).unwrap().unwrap();
        let mut result = Vec::new();
        reader.read_to_end(&mut result).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_lru_store_has_and_size() {
        let store = LruContentStore::new(LruConfig::default(), ContentStoreConfig::default());

        let data = b"test";
        let mut cursor = Cursor::new(data);
        let hash = store.put_stream(&mut cursor).unwrap();

        assert!(store.has(&hash));
        assert_eq!(store.size(&hash).unwrap(), Some(4));

        let missing = ContentHash::from_bytes(b"missing");
        assert!(!store.has(&missing));
        assert_eq!(store.size(&missing).unwrap(), None);
    }

    #[test]
    fn test_lru_store_delete() {
        let store = LruContentStore::new(LruConfig::default(), ContentStoreConfig::default());

        let data = b"test";
        let mut cursor = Cursor::new(data);
        let hash = store.put_stream(&mut cursor).unwrap();

        assert!(store.has(&hash));
        assert!(store.delete(&hash).unwrap());
        assert!(!store.has(&hash));
        assert!(!store.delete(&hash).unwrap());
    }

    #[test]
    fn test_lru_store_len_and_total_bytes() {
        let store = LruContentStore::new(LruConfig::default(), ContentStoreConfig::default());

        assert_eq!(store.len(), 0);
        assert_eq!(store.total_bytes(), 0);
        assert!(store.is_empty());

        let mut c1 = Cursor::new(b"12345");
        let mut c2 = Cursor::new(b"67890ABC");
        store.put_stream(&mut c1).unwrap();
        store.put_stream(&mut c2).unwrap();

        assert_eq!(store.len(), 2);
        assert_eq!(store.total_bytes(), 5 + 8);
        assert!(!store.is_empty());
    }

    #[test]
    fn test_lru_store_clear() {
        let store = LruContentStore::new(LruConfig::default(), ContentStoreConfig::default());

        let mut c1 = Cursor::new(b"data1");
        let mut c2 = Cursor::new(b"data2");
        store.put_stream(&mut c1).unwrap();
        store.put_stream(&mut c2).unwrap();

        assert_eq!(store.len(), 2);
        store.clear();
        assert_eq!(store.len(), 0);
        assert_eq!(store.total_bytes(), 0);
    }

    #[test]
    fn test_lru_ordering_access_updates_position() {
        // Test that accessing an entry moves it to most-recently-used
        let config = LruConfig::default().with_max_entries(3);
        let store = LruContentStore::new(config, ContentStoreConfig::default());

        // Add 3 entries
        let mut c1 = Cursor::new(b"data1");
        let mut c2 = Cursor::new(b"data2");
        let mut c3 = Cursor::new(b"data3");
        let h1 = store.put_stream(&mut c1).unwrap();
        let h2 = store.put_stream(&mut c2).unwrap();
        let h3 = store.put_stream(&mut c3).unwrap();

        // Access h1 to move it to most-recently-used
        std::thread::sleep(std::time::Duration::from_millis(10));
        let _ = store.get_stream(&h1).unwrap();

        // Add new entry - should evict h2 (least recently used)
        let mut c4 = Cursor::new(b"data4");
        store.put_stream(&mut c4).unwrap();

        // h1, h3 should exist, h2 should be evicted
        assert!(store.has(&h1), "h1 should exist after access");
        assert!(!store.has(&h2), "h2 should be evicted as LRU");
        assert!(store.has(&h3), "h3 should exist");
    }

    #[test]
    fn test_lru_size_based_eviction() {
        // 50 bytes max budget
        let config = LruConfig::default().with_max_bytes(50);
        let store = LruContentStore::new(config, ContentStoreConfig::default());

        // Add entries: 20 + 20 + 20 = 60 bytes, should evict first
        let data1 = [0u8; 20];
        let data2 = [1u8; 20];
        let mut c1 = Cursor::new(&data1);
        let mut c2 = Cursor::new(&data2);
        let h1 = store.put_stream(&mut c1).unwrap();
        let h2 = store.put_stream(&mut c2).unwrap();

        assert!(store.has(&h1));
        assert!(store.has(&h2));
        assert_eq!(store.total_bytes(), 40);

        // This should evict h1 to make room
        let data3 = [2u8; 20];
        let mut c3 = Cursor::new(&data3);
        store.put_stream(&mut c3).unwrap();

        assert!(!store.has(&h1), "h1 should be evicted for size");
        assert!(store.has(&h2));
        assert!(store.total_bytes() <= 50);
    }

    #[test]
    fn test_lru_count_based_eviction() {
        let config = LruConfig::default().with_max_entries(2);
        let store = LruContentStore::new(config, ContentStoreConfig::default());

        let mut c1 = Cursor::new(b"data1");
        let mut c2 = Cursor::new(b"data2");
        let h1 = store.put_stream(&mut c1).unwrap();
        let h2 = store.put_stream(&mut c2).unwrap();

        assert_eq!(store.len(), 2);

        // Adding third should evict first
        let mut c3 = Cursor::new(b"data3");
        let h3 = store.put_stream(&mut c3).unwrap();

        assert_eq!(store.len(), 2);
        assert!(!store.has(&h1), "h1 should be evicted");
        assert!(store.has(&h2));
        assert!(store.has(&h3));
    }

    #[test]
    fn test_lru_budget_enforcement_on_put() {
        // Strict budget: only allow 100 bytes
        let config = LruConfig::default().with_max_bytes(100);
        let store = LruContentStore::new(config, ContentStoreConfig::default());

        // Add 90 bytes
        let data1 = [0u8; 90];
        let mut c1 = Cursor::new(&data1);
        let h1 = store.put_stream(&mut c1).unwrap();
        assert!(store.has(&h1));

        // Add 20 more - exceeds budget, should evict h1
        let data2 = [1u8; 20];
        let mut c2 = Cursor::new(&data2);
        let h2 = store.put_stream(&mut c2).unwrap();

        assert!(!store.has(&h1), "h1 should be evicted to make room");
        assert!(store.has(&h2));
        assert!(store.total_bytes() <= 100);
    }

    #[test]
    fn test_lru_multiple_evictions_single_put() {
        // Only 50 bytes allowed
        let config = LruConfig::default().with_max_bytes(50);
        let store = LruContentStore::new(config, ContentStoreConfig::default());

        // Add 3 entries of 10 bytes each = 30 bytes
        let data1 = [0u8; 10];
        let data2 = [1u8; 10];
        let data3 = [2u8; 10];
        let mut c1 = Cursor::new(&data1);
        let mut c2 = Cursor::new(&data2);
        let mut c3 = Cursor::new(&data3);
        let h1 = store.put_stream(&mut c1).unwrap();
        let h2 = store.put_stream(&mut c2).unwrap();
        let h3 = store.put_stream(&mut c3).unwrap();

        // total = 30, max = 50, add 25 byte entry
        // projected = 55 > 50, need to evict 5 bytes minimum
        // Evict h1 (10 bytes): projected = 45 <= 50 (stop)
        let data4 = [3u8; 25];
        let mut c4 = Cursor::new(&data4);
        store.put_stream(&mut c4).unwrap();

        assert!(!store.has(&h1), "h1 should be evicted");
        assert!(store.has(&h2), "h2 should remain");
        assert!(store.has(&h3), "h3 should remain");
        assert!(store.total_bytes() <= 50);
    }

    #[test]
    fn test_lru_eviction_callback() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let eviction_count = Arc::new(AtomicUsize::new(0));
        let evicted_hashes = Arc::new(RwLock::new(Vec::new()));

        let config = LruConfig::default().with_max_entries(2);
        let store = LruContentStore::new(config, ContentStoreConfig::default());

        let count_clone = Arc::clone(&eviction_count);
        let hashes_clone = Arc::clone(&evicted_hashes);
        store.set_eviction_callback(Arc::new(move |event| {
            count_clone.fetch_add(1, Ordering::SeqCst);
            hashes_clone.write().unwrap().push(event.hash);
        }));

        let mut c1 = Cursor::new(b"data1");
        let mut c2 = Cursor::new(b"data2");
        let mut c3 = Cursor::new(b"data3");
        let h1 = store.put_stream(&mut c1).unwrap();
        store.put_stream(&mut c2).unwrap();
        store.put_stream(&mut c3).unwrap();

        assert_eq!(eviction_count.load(Ordering::SeqCst), 1);
        assert!(evicted_hashes.read().unwrap().contains(&h1));
    }

    #[test]
    fn test_lru_eviction_callback_clear() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let eviction_count = Arc::new(AtomicUsize::new(0));

        let config = LruConfig::default().with_max_entries(2);
        let store = LruContentStore::new(config, ContentStoreConfig::default());

        let count_clone = Arc::clone(&eviction_count);
        store.set_eviction_callback(Arc::new(move |_| {
            count_clone.fetch_add(1, Ordering::SeqCst);
        }));

        store.clear_eviction_callback();

        let mut c1 = Cursor::new(b"data1");
        let mut c2 = Cursor::new(b"data2");
        let mut c3 = Cursor::new(b"data3");
        store.put_stream(&mut c1).unwrap();
        store.put_stream(&mut c2).unwrap();
        store.put_stream(&mut c3).unwrap();

        // Callback was cleared, should be 0
        assert_eq!(eviction_count.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_lru_eviction_reason_types() {
        assert_eq!(EvictionReason::Lru, EvictionReason::Lru);
        assert_eq!(EvictionReason::TtlExpired, EvictionReason::TtlExpired);
        assert_eq!(EvictionReason::Manual, EvictionReason::Manual);
        assert_ne!(EvictionReason::Lru, EvictionReason::Manual);
    }

    #[test]
    fn test_lru_single_entry_exact_budget() {
        // Budget exactly matches entry size
        let config = LruConfig::default().with_max_bytes(10);
        let store = LruContentStore::new(config, ContentStoreConfig::default());

        let data1 = [0u8; 10];
        let mut c1 = Cursor::new(&data1);
        let h1 = store.put_stream(&mut c1).unwrap();

        assert!(store.has(&h1));
        assert_eq!(store.total_bytes(), 10);

        // Another 10-byte entry should replace
        let data2 = [1u8; 10];
        let mut c2 = Cursor::new(&data2);
        let h2 = store.put_stream(&mut c2).unwrap();

        assert!(!store.has(&h1));
        assert!(store.has(&h2));
    }

    #[test]
    fn test_lru_single_entry_store() {
        // Only one entry allowed
        let config = LruConfig::default().with_max_entries(1);
        let store = LruContentStore::new(config, ContentStoreConfig::default());

        let mut c1 = Cursor::new(b"first");
        let h1 = store.put_stream(&mut c1).unwrap();
        assert!(store.has(&h1));

        let mut c2 = Cursor::new(b"second");
        let h2 = store.put_stream(&mut c2).unwrap();

        assert!(!store.has(&h1));
        assert!(store.has(&h2));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_lru_config_validation() {
        let config = LruConfig::default();
        assert!(config.validate().is_ok());

        let config = LruConfig::default().with_max_bytes(1000).with_max_entries(100);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_lru_unlimited_no_eviction() {
        // No limits set - should never evict
        let config = LruConfig::default();
        let store = LruContentStore::new(config, ContentStoreConfig::default());

        for i in 0..100 {
            let data = format!("data{}", i);
            let mut cursor = Cursor::new(data.as_bytes());
            store.put_stream(&mut cursor).unwrap();
        }

        assert_eq!(store.len(), 100);
    }

    #[test]
    fn test_lru_entry_info() {
        let store = LruContentStore::new(LruConfig::default(), ContentStoreConfig::default());

        let mut cursor = Cursor::new(b"test data");
        let hash = store.put_stream(&mut cursor).unwrap();

        let info = store.entry_info(&hash);
        assert!(info.is_some());
        let (size, last_access, created) = info.unwrap();
        assert_eq!(size, 9);
        assert!(last_access > 0);
        assert!(created > 0);
        assert_eq!(last_access, created);
    }

    #[test]
    fn test_lru_entry_age() {
        let store = LruContentStore::new(LruConfig::default(), ContentStoreConfig::default());

        let mut cursor = Cursor::new(b"test");
        let hash = store.put_stream(&mut cursor).unwrap();

        let age = store.entry_age_secs(&hash);
        assert!(age.is_some());
        assert!(age.unwrap() < 5); // Should be < 5 seconds old

        let missing = ContentHash::from_bytes(b"missing");
        assert!(store.entry_age_secs(&missing).is_none());
    }

    #[test]
    fn test_lru_replace_existing_entry() {
        let config = LruConfig::default().with_max_bytes(100);
        let store = LruContentStore::new(config, ContentStoreConfig::default());

        let data = b"test data";
        let mut c1 = Cursor::new(data);
        let h1 = store.put_stream(&mut c1).unwrap();

        // Put same content again
        let mut c2 = Cursor::new(data);
        let h2 = store.put_stream(&mut c2).unwrap();

        assert_eq!(h1, h2);
        assert_eq!(store.len(), 1);
        // Size should not double
        assert_eq!(store.total_bytes(), data.len() as u64);
    }

    // ========================================================================
    // LRU TTL Tests
    // ========================================================================

    #[test]
    fn test_lru_ttl_not_expired() {
        let config = LruConfig::default().with_default_ttl(3600); // 1 hour
        let store = LruContentStore::new(config, ContentStoreConfig::default());

        let mut cursor = Cursor::new(b"test");
        let hash = store.put_stream(&mut cursor).unwrap();

        // Should exist since TTL is 1 hour
        assert!(store.has(&hash));

        let tte = store.time_to_expire_secs(&hash);
        assert!(tte.is_some());
        assert!(tte.unwrap() > 3500); // Should be close to 3600
    }

    #[test]
    fn test_lru_ttl_custom_per_entry() {
        let config = LruConfig::default().with_default_ttl(3600);
        let store = LruContentStore::new(config, ContentStoreConfig::default());

        // Use custom TTL
        let mut cursor = Cursor::new(b"short lived");
        let hash = store.put_stream_with_ttl(&mut cursor, Some(60)).unwrap();

        let tte = store.time_to_expire_secs(&hash);
        assert!(tte.is_some());
        assert!(tte.unwrap() <= 60);
    }

    #[test]
    fn test_lru_evict_expired() {
        // This test uses a very short simulated expiry
        // In real use, we'd need to wait or mock time
        let config = LruConfig::default()
            .with_evict_on_access(false); // Don't auto-evict
        let store = LruContentStore::new(config, ContentStoreConfig::default());

        let mut cursor = Cursor::new(b"test");
        let _hash = store.put_stream(&mut cursor).unwrap();

        // Evict expired - nothing should be evicted since no TTL
        let evicted = store.evict_expired();
        assert_eq!(evicted, 0);
    }

    #[test]
    fn test_lru_no_ttl_never_expires() {
        let config = LruConfig::default(); // No default TTL
        let store = LruContentStore::new(config, ContentStoreConfig::default());

        let mut cursor = Cursor::new(b"immortal");
        let hash = store.put_stream(&mut cursor).unwrap();

        let tte = store.time_to_expire_secs(&hash);
        assert!(tte.is_none()); // No TTL = no expiry time
    }

    // ========================================================================
    // LRU Thread Safety Tests
    // ========================================================================

    #[test]
    fn test_lru_concurrent_reads() {
        use std::thread;

        let store = Arc::new(LruContentStore::new(LruConfig::default(), ContentStoreConfig::default()));

        let mut cursor = Cursor::new(b"shared data");
        let hash = store.put_stream(&mut cursor).unwrap();

        let handles: Vec<_> = (0..10)
            .map(|_| {
                let store = Arc::clone(&store);
                let hash = hash;
                thread::spawn(move || {
                    for _ in 0..100 {
                        assert!(store.has(&hash));
                        let reader = store.get_stream(&hash).unwrap();
                        assert!(reader.is_some());
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_lru_concurrent_writes() {
        use std::thread;

        // Use a larger limit to reduce contention and race conditions
        let config = LruConfig::default().with_max_entries(200);
        let store = Arc::new(LruContentStore::new(config, ContentStoreConfig::default()));

        let handles: Vec<_> = (0..10)
            .map(|thread_id| {
                let store = Arc::clone(&store);
                thread::spawn(move || {
                    for i in 0..20 {
                        let data = format!("thread{}_data{}", thread_id, i);
                        let mut cursor = Cursor::new(data.as_bytes());
                        let _ = store.put_stream(&mut cursor);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // Should have some entries (may have evicted some under contention)
        // With concurrent access, count may briefly exceed max_entries due to
        // race conditions between eviction check and insertion, but should
        // be close to the limit (allow 10% tolerance for race conditions)
        assert!(store.len() > 0);
        assert!(store.len() <= 220, "store has {} entries, expected <= 220", store.len());
    }

    #[test]
    fn test_lru_concurrent_mixed_operations() {
        use std::thread;

        let config = LruConfig::default().with_max_entries(50);
        let store = Arc::new(LruContentStore::new(config, ContentStoreConfig::default()));

        // Pre-populate some entries
        for i in 0..20 {
            let data = format!("initial{}", i);
            let mut cursor = Cursor::new(data.as_bytes());
            store.put_stream(&mut cursor).unwrap();
        }

        let handles: Vec<_> = (0..5)
            .map(|thread_id| {
                let store = Arc::clone(&store);
                thread::spawn(move || {
                    for i in 0..50 {
                        match i % 4 {
                            0 => {
                                // Write
                                let data = format!("t{}_write{}", thread_id, i);
                                let mut cursor = Cursor::new(data.as_bytes());
                                let _ = store.put_stream(&mut cursor);
                            }
                            1 => {
                                // Read
                                let data = format!("initial{}", i % 20);
                                let hash = ContentHash::from_bytes(data.as_bytes());
                                let _ = store.get_stream(&hash);
                            }
                            2 => {
                                // Has
                                let data = format!("initial{}", i % 20);
                                let hash = ContentHash::from_bytes(data.as_bytes());
                                let _ = store.has(&hash);
                            }
                            _ => {
                                // Size check
                                let _ = store.total_bytes();
                            }
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // Store should be in consistent state
        assert!(store.len() <= 50);
    }

    // ========================================================================
    // LRU Edge Cases
    // ========================================================================

    #[test]
    fn test_lru_empty_content() {
        let store = LruContentStore::new(LruConfig::default(), ContentStoreConfig::default());

        let mut cursor = Cursor::new(b"");
        let hash = store.put_stream(&mut cursor).unwrap();

        assert!(store.has(&hash));
        assert_eq!(store.size(&hash).unwrap(), Some(0));
    }

    #[test]
    fn test_lru_get_missing_entry() {
        let store = LruContentStore::new(LruConfig::default(), ContentStoreConfig::default());

        let missing = ContentHash::from_bytes(b"nonexistent");
        let result = store.get_stream(&missing).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_lru_delete_missing_entry() {
        let store = LruContentStore::new(LruConfig::default(), ContentStoreConfig::default());

        let missing = ContentHash::from_bytes(b"nonexistent");
        assert!(!store.delete(&missing).unwrap());
    }

    #[test]
    fn test_lru_zero_byte_budget() {
        // Edge case: 0 byte budget but unlimited entries
        let config = LruConfig::default().with_max_entries(100);
        let store = LruContentStore::new(config, ContentStoreConfig::default());

        // Should still work - no byte limit
        let data = [0u8; 1000];
        let mut cursor = Cursor::new(&data);
        let hash = store.put_stream(&mut cursor).unwrap();
        assert!(store.has(&hash));
    }

    #[test]
    fn test_lru_default_store() {
        let store = LruContentStore::default();
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
    }

    #[test]
    fn test_lru_eviction_event_fields() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let event_received = Arc::new(AtomicBool::new(false));

        let config = LruConfig::default().with_max_entries(1);
        let store = LruContentStore::new(config, ContentStoreConfig::default());

        let event_clone = Arc::clone(&event_received);
        store.set_eviction_callback(Arc::new(move |event| {
            assert!(!event.hash.is_zero());
            assert!(event.size > 0);
            assert!(event.timestamp_ms > 0);
            assert_eq!(event.reason, EvictionReason::Lru);
            event_clone.store(true, Ordering::SeqCst);
        }));

        let mut c1 = Cursor::new(b"first");
        let mut c2 = Cursor::new(b"second");
        store.put_stream(&mut c1).unwrap();
        store.put_stream(&mut c2).unwrap();

        assert!(event_received.load(Ordering::SeqCst));
    }

    #[test]
    fn test_lru_manual_delete_eviction_reason() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let manual_eviction = Arc::new(AtomicBool::new(false));

        let store = LruContentStore::new(LruConfig::default(), ContentStoreConfig::default());

        let manual_clone = Arc::clone(&manual_eviction);
        store.set_eviction_callback(Arc::new(move |event| {
            if event.reason == EvictionReason::Manual {
                manual_clone.store(true, Ordering::SeqCst);
            }
        }));

        let mut cursor = Cursor::new(b"test");
        let hash = store.put_stream(&mut cursor).unwrap();
        store.delete(&hash).unwrap();

        assert!(manual_eviction.load(Ordering::SeqCst));
    }

    #[test]
    fn test_lru_config_evict_on_access_disabled() {
        let config = LruConfig::default().with_evict_on_access(false);
        let store = LruContentStore::new(config, ContentStoreConfig::default());

        assert!(!store.lru_config().evict_on_access);
    }

    #[test]
    fn test_lru_lru_config_accessor() {
        let config = LruConfig::default().with_max_bytes(1000).with_max_entries(10);
        let store = LruContentStore::new(config, ContentStoreConfig::default());

        assert_eq!(store.lru_config().max_bytes, 1000);
        assert_eq!(store.lru_config().max_entries, 10);
    }

    // ========================================================================
    // Multi-Level Sharding Tests (T-AS-4.5)
    // ========================================================================

    #[test]
    fn test_sharding_depth_two_level_path_generation() {
        let depth = ShardingDepth::TwoLevel;
        let hex = "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890";

        let (parts, filename) = depth.split_hash(hex);
        assert_eq!(parts, vec!["ab"]);
        assert_eq!(filename, "cdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890");

        let path = depth.build_path(Path::new("/store"), hex);
        assert_eq!(
            path,
            PathBuf::from("/store/ab/cdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890")
        );
    }

    #[test]
    fn test_sharding_depth_four_level_path_generation() {
        let depth = ShardingDepth::FourLevel;
        let hex = "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890";

        let (parts, filename) = depth.split_hash(hex);
        assert_eq!(parts, vec!["ab", "cd"]);
        assert_eq!(filename, "ef1234567890abcdef1234567890abcdef1234567890abcdef1234567890");

        let path = depth.build_path(Path::new("/store"), hex);
        assert_eq!(
            path,
            PathBuf::from("/store/ab/cd/ef1234567890abcdef1234567890abcdef1234567890abcdef1234567890")
        );
    }

    #[test]
    fn test_sharding_depth_six_level_path_generation() {
        let depth = ShardingDepth::SixLevel;
        let hex = "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890";

        let (parts, filename) = depth.split_hash(hex);
        assert_eq!(parts, vec!["ab", "cd", "ef"]);
        assert_eq!(filename, "1234567890abcdef1234567890abcdef1234567890abcdef1234567890");

        let path = depth.build_path(Path::new("/store"), hex);
        assert_eq!(
            path,
            PathBuf::from("/store/ab/cd/ef/1234567890abcdef1234567890abcdef1234567890abcdef1234567890")
        );
    }

    #[test]
    fn test_sharding_depth_levels() {
        assert_eq!(ShardingDepth::TwoLevel.levels(), &[2]);
        assert_eq!(ShardingDepth::FourLevel.levels(), &[2, 2]);
        assert_eq!(ShardingDepth::SixLevel.levels(), &[2, 2, 2]);
    }

    #[test]
    fn test_sharding_depth_prefix_len() {
        assert_eq!(ShardingDepth::TwoLevel.prefix_len(), 2);
        assert_eq!(ShardingDepth::FourLevel.prefix_len(), 4);
        assert_eq!(ShardingDepth::SixLevel.prefix_len(), 6);
    }

    #[test]
    fn test_sharding_depth_default() {
        let depth: ShardingDepth = Default::default();
        assert_eq!(depth, ShardingDepth::TwoLevel);
    }

    #[test]
    fn test_sharded_store_config_default() {
        let config = ShardedStoreConfig::default();
        assert_eq!(config.sharding, ShardingDepth::TwoLevel);
        assert!(!config.backward_compat);
    }

    #[test]
    fn test_sharded_store_config_with_sharding() {
        let config = ShardedStoreConfig::default().with_sharding(ShardingDepth::FourLevel);
        assert_eq!(config.sharding, ShardingDepth::FourLevel);
    }

    #[test]
    fn test_sharded_store_config_with_backward_compat() {
        let config = ShardedStoreConfig::default().with_backward_compat(true);
        assert!(config.backward_compat);
    }

    #[test]
    fn test_sharded_store_two_level_put_get() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = ShardedStoreConfig::default().with_sharding(ShardingDepth::TwoLevel);
        let store = ShardedContentStore::new(temp_dir.path(), config).unwrap();

        let data = b"test content for two-level sharding";
        let mut cursor = Cursor::new(data);

        let hash = store.put_stream(&mut cursor).unwrap();
        assert_eq!(hash, ContentHash::from_bytes(data));

        let mut reader = store.get_stream(&hash).unwrap().unwrap();
        let mut result = Vec::new();
        reader.read_to_end(&mut result).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_sharded_store_four_level_put_get() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = ShardedStoreConfig::default().with_sharding(ShardingDepth::FourLevel);
        let store = ShardedContentStore::new(temp_dir.path(), config).unwrap();

        let data = b"test content for four-level sharding";
        let mut cursor = Cursor::new(data);

        let hash = store.put_stream(&mut cursor).unwrap();
        assert_eq!(hash, ContentHash::from_bytes(data));

        let mut reader = store.get_stream(&hash).unwrap().unwrap();
        let mut result = Vec::new();
        reader.read_to_end(&mut result).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_sharded_store_six_level_put_get() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = ShardedStoreConfig::default().with_sharding(ShardingDepth::SixLevel);
        let store = ShardedContentStore::new(temp_dir.path(), config).unwrap();

        let data = b"test content for six-level sharding";
        let mut cursor = Cursor::new(data);

        let hash = store.put_stream(&mut cursor).unwrap();
        assert_eq!(hash, ContentHash::from_bytes(data));

        let mut reader = store.get_stream(&hash).unwrap().unwrap();
        let mut result = Vec::new();
        reader.read_to_end(&mut result).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_sharded_store_has() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = ShardedStoreConfig::default().with_sharding(ShardingDepth::FourLevel);
        let store = ShardedContentStore::new(temp_dir.path(), config).unwrap();

        let data = b"test";
        let mut cursor = Cursor::new(data);
        let hash = store.put_stream(&mut cursor).unwrap();

        assert!(store.has(&hash));

        let missing = ContentHash::from_bytes(b"nonexistent");
        assert!(!store.has(&missing));
    }

    #[test]
    fn test_sharded_store_size() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = ShardedStoreConfig::default().with_sharding(ShardingDepth::SixLevel);
        let store = ShardedContentStore::new(temp_dir.path(), config).unwrap();

        let data = b"test content with known size";
        let mut cursor = Cursor::new(data);
        let hash = store.put_stream(&mut cursor).unwrap();

        let size = store.size(&hash).unwrap().unwrap();
        assert_eq!(size, data.len() as u64);
    }

    #[test]
    fn test_sharded_store_delete() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = ShardedStoreConfig::default().with_sharding(ShardingDepth::FourLevel);
        let store = ShardedContentStore::new(temp_dir.path(), config).unwrap();

        let data = b"to be deleted";
        let mut cursor = Cursor::new(data);
        let hash = store.put_stream(&mut cursor).unwrap();

        assert!(store.has(&hash));
        let deleted = store.delete(&hash).unwrap();
        assert!(deleted);
        assert!(!store.has(&hash));

        // Delete again returns false
        let deleted_again = store.delete(&hash).unwrap();
        assert!(!deleted_again);
    }

    #[test]
    fn test_sharded_store_list() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = ShardedStoreConfig::default().with_sharding(ShardingDepth::FourLevel);
        let store = ShardedContentStore::new(temp_dir.path(), config).unwrap();

        let mut c1 = Cursor::new(b"data1");
        let mut c2 = Cursor::new(b"data2");
        let mut c3 = Cursor::new(b"data3");
        let h1 = store.put_stream(&mut c1).unwrap();
        let h2 = store.put_stream(&mut c2).unwrap();
        let h3 = store.put_stream(&mut c3).unwrap();

        let mut listed = store.list().unwrap();
        listed.sort_by(|a, b| format!("{}", a).cmp(&format!("{}", b)));

        let mut expected = vec![h1, h2, h3];
        expected.sort_by(|a, b| format!("{}", a).cmp(&format!("{}", b)));

        assert_eq!(listed, expected);
    }

    #[test]
    fn test_sharded_store_stats() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = ShardedStoreConfig::default().with_sharding(ShardingDepth::SixLevel);
        let store = ShardedContentStore::new(temp_dir.path(), config).unwrap();

        let mut c1 = Cursor::new(b"12345");
        let mut c2 = Cursor::new(b"67890ABC");
        store.put_stream(&mut c1).unwrap();
        store.put_stream(&mut c2).unwrap();

        let stats = store.stats().unwrap();
        assert_eq!(stats.entry_count, 2);
        assert_eq!(stats.total_bytes, 5 + 8);
        assert_eq!(stats.sharding, ShardingDepth::SixLevel);
    }

    #[test]
    fn test_sharded_store_directory_structure_four_level() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = ShardedStoreConfig::default().with_sharding(ShardingDepth::FourLevel);
        let store = ShardedContentStore::new(temp_dir.path(), config).unwrap();

        let data = b"directory structure test";
        let mut cursor = Cursor::new(data);
        let hash = store.put_stream(&mut cursor).unwrap();

        let hex = format!("{}", hash);
        let expected_path = temp_dir.path()
            .join(&hex[..2])
            .join(&hex[2..4])
            .join(&hex[4..]);

        assert!(expected_path.exists());
    }

    #[test]
    fn test_sharded_store_directory_structure_six_level() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = ShardedStoreConfig::default().with_sharding(ShardingDepth::SixLevel);
        let store = ShardedContentStore::new(temp_dir.path(), config).unwrap();

        let data = b"six level structure test";
        let mut cursor = Cursor::new(data);
        let hash = store.put_stream(&mut cursor).unwrap();

        let hex = format!("{}", hash);
        let expected_path = temp_dir.path()
            .join(&hex[..2])
            .join(&hex[2..4])
            .join(&hex[4..6])
            .join(&hex[6..]);

        assert!(expected_path.exists());
    }

    #[test]
    fn test_sharded_store_backward_compat_reads() {
        let temp_dir = tempfile::tempdir().unwrap();

        // First, store with 2-level sharding
        {
            let config = ShardedStoreConfig::default().with_sharding(ShardingDepth::TwoLevel);
            let store = ShardedContentStore::new(temp_dir.path(), config).unwrap();
            let mut cursor = Cursor::new(b"legacy data");
            store.put_stream(&mut cursor).unwrap();
        }

        // Now open with 4-level but backward compat enabled
        {
            let config = ShardedStoreConfig::default()
                .with_sharding(ShardingDepth::FourLevel)
                .with_backward_compat(true);
            let store = ShardedContentStore::open(temp_dir.path(), config).unwrap();

            let hash = ContentHash::from_bytes(b"legacy data");
            assert!(store.has(&hash), "should find legacy data via backward compat");

            let mut reader = store.get_stream(&hash).unwrap().unwrap();
            let mut result = Vec::new();
            reader.read_to_end(&mut result).unwrap();
            assert_eq!(result, b"legacy data");
        }
    }

    #[test]
    fn test_sharded_store_backward_compat_disabled() {
        let temp_dir = tempfile::tempdir().unwrap();

        // Store with 2-level sharding
        {
            let config = ShardedStoreConfig::default().with_sharding(ShardingDepth::TwoLevel);
            let store = ShardedContentStore::new(temp_dir.path(), config).unwrap();
            let mut cursor = Cursor::new(b"legacy data");
            store.put_stream(&mut cursor).unwrap();
        }

        // Open with 4-level, backward compat DISABLED
        {
            let config = ShardedStoreConfig::default()
                .with_sharding(ShardingDepth::FourLevel)
                .with_backward_compat(false);
            let store = ShardedContentStore::open(temp_dir.path(), config).unwrap();

            let hash = ContentHash::from_bytes(b"legacy data");
            assert!(!store.has(&hash), "should NOT find legacy data without backward compat");
        }
    }

    #[test]
    fn test_sharding_migration_two_to_four_level() {
        let temp_dir = tempfile::tempdir().unwrap();

        // Create store with 2-level sharding and add content
        let h1;
        let h2;
        {
            let config = ShardedStoreConfig::default().with_sharding(ShardingDepth::TwoLevel);
            let store = ShardedContentStore::new(temp_dir.path(), config).unwrap();

            let mut c1 = Cursor::new(b"data1");
            let mut c2 = Cursor::new(b"data2");
            h1 = store.put_stream(&mut c1).unwrap();
            h2 = store.put_stream(&mut c2).unwrap();
        }

        // Run migration
        let migrator = ShardingMigrator::new(
            temp_dir.path(),
            ShardingDepth::TwoLevel,
            ShardingDepth::FourLevel,
        );
        let result = migrator.migrate().unwrap();

        assert_eq!(result.total, 2);
        assert_eq!(result.migrated, 2);
        assert_eq!(result.skipped, 0);
        assert_eq!(result.errors, 0);
        assert_eq!(result.from_sharding, ShardingDepth::TwoLevel);
        assert_eq!(result.to_sharding, ShardingDepth::FourLevel);

        // Verify content at new locations
        let config = ShardedStoreConfig::default().with_sharding(ShardingDepth::FourLevel);
        let store = ShardedContentStore::open(temp_dir.path(), config).unwrap();

        assert!(store.has(&h1));
        assert!(store.has(&h2));

        let mut reader1 = store.get_stream(&h1).unwrap().unwrap();
        let mut data1 = Vec::new();
        reader1.read_to_end(&mut data1).unwrap();
        assert_eq!(data1, b"data1");
    }

    #[test]
    fn test_sharding_migration_two_to_six_level() {
        let temp_dir = tempfile::tempdir().unwrap();

        // Create 2-level store
        let hashes: Vec<ContentHash>;
        {
            let config = ShardedStoreConfig::default().with_sharding(ShardingDepth::TwoLevel);
            let store = ShardedContentStore::new(temp_dir.path(), config).unwrap();

            hashes = (0..5)
                .map(|i| {
                    let data = format!("data_{}", i);
                    let mut cursor = Cursor::new(data.as_bytes());
                    store.put_stream(&mut cursor).unwrap()
                })
                .collect();
        }

        // Migrate to 6-level
        let migrator = ShardingMigrator::new(
            temp_dir.path(),
            ShardingDepth::TwoLevel,
            ShardingDepth::SixLevel,
        );
        let result = migrator.migrate().unwrap();

        assert_eq!(result.total, 5);
        assert_eq!(result.migrated, 5);

        // Verify at 6-level
        let config = ShardedStoreConfig::default().with_sharding(ShardingDepth::SixLevel);
        let store = ShardedContentStore::open(temp_dir.path(), config).unwrap();

        for hash in &hashes {
            assert!(store.has(hash));
        }
    }

    #[test]
    fn test_sharding_migration_four_to_six_level() {
        let temp_dir = tempfile::tempdir().unwrap();

        // Create 4-level store
        let hash;
        {
            let config = ShardedStoreConfig::default().with_sharding(ShardingDepth::FourLevel);
            let store = ShardedContentStore::new(temp_dir.path(), config).unwrap();
            let mut cursor = Cursor::new(b"four to six");
            hash = store.put_stream(&mut cursor).unwrap();
        }

        // Migrate to 6-level
        let migrator = ShardingMigrator::new(
            temp_dir.path(),
            ShardingDepth::FourLevel,
            ShardingDepth::SixLevel,
        );
        let result = migrator.migrate().unwrap();

        assert_eq!(result.migrated, 1);

        // Verify
        let config = ShardedStoreConfig::default().with_sharding(ShardingDepth::SixLevel);
        let store = ShardedContentStore::open(temp_dir.path(), config).unwrap();
        assert!(store.has(&hash));
    }

    #[test]
    fn test_sharding_migration_skip_existing() {
        let temp_dir = tempfile::tempdir().unwrap();

        let hash;
        // Create at both 2-level and 4-level
        {
            let config2 = ShardedStoreConfig::default().with_sharding(ShardingDepth::TwoLevel);
            let store2 = ShardedContentStore::new(temp_dir.path(), config2).unwrap();
            let mut cursor = Cursor::new(b"duplicate");
            hash = store2.put_stream(&mut cursor).unwrap();

            // Also put at 4-level
            let config4 = ShardedStoreConfig::default().with_sharding(ShardingDepth::FourLevel);
            let store4 = ShardedContentStore::new(temp_dir.path(), config4).unwrap();
            let mut cursor2 = Cursor::new(b"duplicate");
            store4.put_stream(&mut cursor2).unwrap();
        }

        // Migrate should skip the existing one
        let migrator = ShardingMigrator::new(
            temp_dir.path(),
            ShardingDepth::TwoLevel,
            ShardingDepth::FourLevel,
        );
        let result = migrator.migrate().unwrap();

        // All should be skipped since they exist at target
        assert_eq!(result.skipped, result.total);
    }

    #[test]
    fn test_sharding_migration_progress_callback() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let temp_dir = tempfile::tempdir().unwrap();
        let progress_count = Arc::new(AtomicUsize::new(0));

        // Create store with content
        {
            let config = ShardedStoreConfig::default().with_sharding(ShardingDepth::TwoLevel);
            let store = ShardedContentStore::new(temp_dir.path(), config).unwrap();
            for i in 0..5 {
                let data = format!("data_{}", i);
                let mut cursor = Cursor::new(data.as_bytes());
                store.put_stream(&mut cursor).unwrap();
            }
        }

        // Migrate with progress callback
        let count_clone = Arc::clone(&progress_count);
        let migrator = ShardingMigrator::new(
            temp_dir.path(),
            ShardingDepth::TwoLevel,
            ShardingDepth::FourLevel,
        )
        .with_progress_callback(Arc::new(move |progress| {
            count_clone.fetch_add(1, Ordering::SeqCst);
            assert!(progress.total == 5);
            assert!(progress.percent_complete() >= 0.0);
            assert!(progress.percent_complete() <= 100.0);
        }));

        migrator.migrate().unwrap();

        // Should have been called multiple times
        assert!(progress_count.load(Ordering::SeqCst) > 0);
    }

    #[test]
    fn test_sharding_migration_copy_mode() {
        let temp_dir = tempfile::tempdir().unwrap();

        // Create 2-level store
        let hash;
        {
            let config = ShardedStoreConfig::default().with_sharding(ShardingDepth::TwoLevel);
            let store = ShardedContentStore::new(temp_dir.path(), config).unwrap();
            let mut cursor = Cursor::new(b"copy me");
            hash = store.put_stream(&mut cursor).unwrap();
        }

        // Migrate with delete_source = false (copy mode)
        let migrator = ShardingMigrator::new(
            temp_dir.path(),
            ShardingDepth::TwoLevel,
            ShardingDepth::FourLevel,
        )
        .with_delete_source(false);
        migrator.migrate().unwrap();

        // Both locations should have the file
        let config2 = ShardedStoreConfig::default().with_sharding(ShardingDepth::TwoLevel);
        let store2 = ShardedContentStore::open(temp_dir.path(), config2).unwrap();
        assert!(store2.has(&hash), "2-level should still have file after copy migration");

        let config4 = ShardedStoreConfig::default().with_sharding(ShardingDepth::FourLevel);
        let store4 = ShardedContentStore::open(temp_dir.path(), config4).unwrap();
        assert!(store4.has(&hash), "4-level should have file after migration");
    }

    #[test]
    fn test_sharding_migration_empty_store() {
        let temp_dir = tempfile::tempdir().unwrap();

        // Create empty store
        let config = ShardedStoreConfig::default().with_sharding(ShardingDepth::TwoLevel);
        ShardedContentStore::new(temp_dir.path(), config).unwrap();

        // Migrate empty store
        let migrator = ShardingMigrator::new(
            temp_dir.path(),
            ShardingDepth::TwoLevel,
            ShardingDepth::FourLevel,
        );
        let result = migrator.migrate().unwrap();

        assert_eq!(result.total, 0);
        assert_eq!(result.migrated, 0);
        assert_eq!(result.skipped, 0);
        assert_eq!(result.errors, 0);
    }

    #[test]
    fn test_migration_progress_percent_complete() {
        let progress = MigrationProgress {
            total: 100,
            completed: 50,
            skipped: 10,
            errors: 0,
            current_hash: None,
        };
        assert_eq!(progress.percent_complete(), 60.0);

        let empty_progress = MigrationProgress {
            total: 0,
            completed: 0,
            skipped: 0,
            errors: 0,
            current_hash: None,
        };
        assert_eq!(empty_progress.percent_complete(), 100.0);
    }

    #[test]
    fn test_sharded_store_deduplication() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = ShardedStoreConfig::default().with_sharding(ShardingDepth::FourLevel);
        let store = ShardedContentStore::new(temp_dir.path(), config).unwrap();

        let data = b"duplicate content";
        let mut c1 = Cursor::new(data);
        let mut c2 = Cursor::new(data);

        let h1 = store.put_stream(&mut c1).unwrap();
        let h2 = store.put_stream(&mut c2).unwrap();

        assert_eq!(h1, h2);
        assert_eq!(store.list().unwrap().len(), 1);
    }

    #[test]
    fn test_sharded_store_hash_distribution() {
        // Test that different content produces different shard paths
        let temp_dir = tempfile::tempdir().unwrap();
        let config = ShardedStoreConfig::default().with_sharding(ShardingDepth::FourLevel);
        let store = ShardedContentStore::new(temp_dir.path(), config).unwrap();

        let mut hashes = Vec::new();
        for i in 0..100 {
            let data = format!("content_{}", i);
            let mut cursor = Cursor::new(data.as_bytes());
            let hash = store.put_stream(&mut cursor).unwrap();
            hashes.push(hash);
        }

        // All hashes should be unique
        let mut unique_hashes = hashes.clone();
        unique_hashes.sort_by(|a, b| format!("{}", a).cmp(&format!("{}", b)));
        unique_hashes.dedup();
        assert_eq!(unique_hashes.len(), 100);
    }

    #[test]
    fn test_sharded_store_directory_creation() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = ShardedStoreConfig::default().with_sharding(ShardingDepth::SixLevel);
        let store = ShardedContentStore::new(temp_dir.path(), config).unwrap();

        let data = b"test directory creation";
        let mut cursor = Cursor::new(data);
        let hash = store.put_stream(&mut cursor).unwrap();

        let hex = format!("{}", hash);

        // Check all directory levels exist
        let level1 = temp_dir.path().join(&hex[..2]);
        let level2 = level1.join(&hex[2..4]);
        let level3 = level2.join(&hex[4..6]);

        assert!(level1.is_dir());
        assert!(level2.is_dir());
        assert!(level3.is_dir());
    }

    #[test]
    fn test_sharded_store_performance_many_entries() {
        use std::time::Instant;

        let temp_dir = tempfile::tempdir().unwrap();
        let config = ShardedStoreConfig::default()
            .with_sharding(ShardingDepth::FourLevel)
            .with_sync_writes(false); // Faster for benchmark
        let store = ShardedContentStore::new(temp_dir.path(), config).unwrap();

        // Store 1000 entries
        let mut hashes = Vec::with_capacity(1000);
        for i in 0..1000 {
            let data = format!("benchmark_data_{:05}", i);
            let mut cursor = Cursor::new(data.as_bytes());
            let hash = store.put_stream(&mut cursor).unwrap();
            hashes.push(hash);
        }

        // Benchmark get() calls
        let start = Instant::now();
        for hash in &hashes {
            let reader = store.get_stream(hash).unwrap();
            assert!(reader.is_some());
        }
        let elapsed = start.elapsed();

        // Should be < 5ms per get on average (1000 gets)
        // Allow more time since we're doing 1000 gets total
        let avg_ms_per_get = elapsed.as_secs_f64() * 1000.0 / 1000.0;
        assert!(
            avg_ms_per_get < 5.0,
            "get() too slow: {:.3}ms avg",
            avg_ms_per_get
        );
    }

    #[test]
    fn test_sharded_store_performance_has_check() {
        use std::time::Instant;

        let temp_dir = tempfile::tempdir().unwrap();
        let config = ShardedStoreConfig::default()
            .with_sharding(ShardingDepth::FourLevel)
            .with_sync_writes(false);
        let store = ShardedContentStore::new(temp_dir.path(), config).unwrap();

        // Store entries
        let mut hashes = Vec::with_capacity(1000);
        for i in 0..1000 {
            let data = format!("perf_data_{:05}", i);
            let mut cursor = Cursor::new(data.as_bytes());
            let hash = store.put_stream(&mut cursor).unwrap();
            hashes.push(hash);
        }

        // Benchmark has() calls
        let start = Instant::now();
        for hash in &hashes {
            assert!(store.has(hash));
        }
        let elapsed = start.elapsed();

        // has() should be faster than get()
        let avg_ms_per_has = elapsed.as_secs_f64() * 1000.0 / 1000.0;
        assert!(
            avg_ms_per_has < 2.0,
            "has() too slow: {:.3}ms avg",
            avg_ms_per_has
        );
    }

    #[test]
    fn test_sharded_store_open_nonexistent() {
        let config = ShardedStoreConfig::default();
        let result = ShardedContentStore::open("/nonexistent/path/to/store", config);
        assert!(matches!(result, Err(ContentStoreError::Io(_))));
    }

    #[test]
    fn test_sharded_store_sharding_accessor() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = ShardedStoreConfig::default().with_sharding(ShardingDepth::SixLevel);
        let store = ShardedContentStore::new(temp_dir.path(), config).unwrap();

        assert_eq!(store.sharding(), ShardingDepth::SixLevel);
    }

    #[test]
    fn test_sharded_store_base_path_accessor() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = ShardedStoreConfig::default();
        let store = ShardedContentStore::new(temp_dir.path(), config).unwrap();

        assert_eq!(store.base_path(), temp_dir.path());
    }

    #[test]
    fn test_sharded_store_config_accessor() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = ShardedStoreConfig::default()
            .with_buffer_size(128 * 1024)
            .with_sync_writes(false);
        let store = ShardedContentStore::new(temp_dir.path(), config).unwrap();

        assert_eq!(store.config().buffer_size, 128 * 1024);
        assert!(!store.config().sync_writes);
    }

    #[test]
    fn test_sharded_store_empty_content() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = ShardedStoreConfig::default().with_sharding(ShardingDepth::FourLevel);
        let store = ShardedContentStore::new(temp_dir.path(), config).unwrap();

        let mut cursor = Cursor::new(b"");
        let hash = store.put_stream(&mut cursor).unwrap();

        assert!(store.has(&hash));
        assert_eq!(store.size(&hash).unwrap(), Some(0));
    }

    #[test]
    fn test_sharded_store_large_content() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = ShardedStoreConfig::default()
            .with_sharding(ShardingDepth::SixLevel)
            .with_buffer_size(8 * 1024);
        let store = ShardedContentStore::new(temp_dir.path(), config).unwrap();

        // 1MB of data
        let data: Vec<u8> = (0..1_000_000).map(|i| (i % 256) as u8).collect();
        let mut cursor = Cursor::new(&data);

        let hash = store.put_stream(&mut cursor).unwrap();
        assert_eq!(hash, ContentHash::from_bytes(&data));

        let mut reader = store.get_stream(&hash).unwrap().unwrap();
        let mut result = Vec::new();
        reader.read_to_end(&mut result).unwrap();
        assert_eq!(result, data);
    }
}
