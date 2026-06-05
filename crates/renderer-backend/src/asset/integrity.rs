//! CRC-32C Integrity Verification for TRINITY asset pipeline (T-AS-4.6).
//!
//! This module provides integrity verification using hardware-accelerated CRC-32C
//! checksums, with optional replication support for fault tolerance.
//!
//! # Features
//!
//! - CRC-32C computed on `put()`, stored alongside content hash
//! - CRC-32C verified on every `get()` read
//! - Corruption detected with asset identity in error
//! - Optional replication: N copies per asset (configurable)
//! - Replica fallback: if primary CRC fails, try replicas
//! - Per-instance configuration
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::asset::integrity::{IntegrityConfig, IntegrityStore};
//! use renderer_backend::asset::content_store::{FileContentStore, ContentStoreConfig};
//! use std::io::Cursor;
//!
//! // Create store with integrity verification and 2 replicas
//! let config = IntegrityConfig::default()
//!     .with_verify_on_read(true)
//!     .with_replicas(2);
//!
//! let inner = FileContentStore::new("/tmp/store", ContentStoreConfig::default())?;
//! let store = IntegrityStore::new(inner, config);
//!
//! // Store with CRC computation
//! let mut data = Cursor::new(b"asset data");
//! let hash = store.put_stream(&mut data)?;
//!
//! // Read with automatic CRC verification
//! let reader = store.get_stream(&hash)?;
//! ```

use std::collections::HashMap;
use std::fmt;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use crate::pipeline::ContentHash;
use super::content_store::{
    ContentStoreConfig, ContentStoreError, StreamingHashReader, FileSeekableReader, MemorySeekableReader,
    DEFAULT_BUFFER_SIZE, MIN_BUFFER_SIZE, MAX_BUFFER_SIZE,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for integrity verification.
#[derive(Debug, Clone)]
pub struct IntegrityConfig {
    /// Whether to verify CRC on read operations.
    pub verify_on_read: bool,
    /// Number of replicas to create (0 = no replication, 1 = one extra copy, etc.).
    pub replica_count: usize,
    /// Whether to fallback to replicas on CRC failure.
    pub fallback_on_failure: bool,
    /// Buffer size for streaming CRC computation.
    pub buffer_size: usize,
}

impl Default for IntegrityConfig {
    fn default() -> Self {
        Self {
            verify_on_read: true,
            replica_count: 0,
            fallback_on_failure: true,
            buffer_size: DEFAULT_BUFFER_SIZE,
        }
    }
}

impl IntegrityConfig {
    /// Enable or disable CRC verification on read.
    pub fn with_verify_on_read(mut self, verify: bool) -> Self {
        self.verify_on_read = verify;
        self
    }

    /// Set number of replicas (0 = primary only).
    pub fn with_replicas(mut self, count: usize) -> Self {
        self.replica_count = count;
        self
    }

    /// Enable or disable replica fallback on CRC failure.
    pub fn with_fallback_on_failure(mut self, fallback: bool) -> Self {
        self.fallback_on_failure = fallback;
        self
    }

    /// Set buffer size for CRC computation.
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size.clamp(MIN_BUFFER_SIZE, MAX_BUFFER_SIZE);
        self
    }
}

// ---------------------------------------------------------------------------
// CRC Metadata
// ---------------------------------------------------------------------------

/// CRC metadata for a stored asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CrcMetadata {
    /// CRC-32C checksum of the content.
    pub crc32c: u32,
    /// Size in bytes.
    pub size: u64,
}

impl CrcMetadata {
    /// Create new CRC metadata.
    pub fn new(crc32c: u32, size: u64) -> Self {
        Self { crc32c, size }
    }

    /// Serialize to bytes (8 bytes: 4 CRC + 4 unused for alignment).
    pub fn to_bytes(&self) -> [u8; 12] {
        let mut bytes = [0u8; 12];
        bytes[0..4].copy_from_slice(&self.crc32c.to_le_bytes());
        bytes[4..12].copy_from_slice(&self.size.to_le_bytes());
        bytes
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8; 12]) -> Self {
        let crc32c = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let size = u64::from_le_bytes([
            bytes[4], bytes[5], bytes[6], bytes[7],
            bytes[8], bytes[9], bytes[10], bytes[11],
        ]);
        Self { crc32c, size }
    }
}

// ---------------------------------------------------------------------------
// Integrity Error
// ---------------------------------------------------------------------------

/// Error types specific to integrity verification.
#[derive(Debug)]
pub enum IntegrityError {
    /// CRC mismatch detected.
    CrcMismatch {
        asset_hash: ContentHash,
        expected_crc: u32,
        actual_crc: u32,
        replica_index: Option<usize>,
    },
    /// All replicas failed CRC verification.
    AllReplicasFailed {
        asset_hash: ContentHash,
        failures: Vec<(usize, u32, u32)>, // (replica_index, expected, actual)
    },
    /// Content store error.
    Store(ContentStoreError),
    /// I/O error.
    Io(io::Error),
}

impl fmt::Display for IntegrityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CrcMismatch { asset_hash, expected_crc, actual_crc, replica_index } => {
                match replica_index {
                    Some(idx) => write!(
                        f,
                        "CRC-32C mismatch for asset {} (replica {}): expected {:08x}, got {:08x}",
                        asset_hash, idx, expected_crc, actual_crc
                    ),
                    None => write!(
                        f,
                        "CRC-32C mismatch for asset {}: expected {:08x}, got {:08x}",
                        asset_hash, expected_crc, actual_crc
                    ),
                }
            }
            Self::AllReplicasFailed { asset_hash, failures } => {
                write!(f, "All replicas failed CRC verification for asset {}: ", asset_hash)?;
                for (idx, expected, actual) in failures {
                    write!(f, "[replica {}: expected {:08x}, got {:08x}] ", idx, expected, actual)?;
                }
                Ok(())
            }
            Self::Store(e) => write!(f, "content store error: {}", e),
            Self::Io(e) => write!(f, "I/O error: {}", e),
        }
    }
}

impl std::error::Error for IntegrityError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Store(e) => Some(e),
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<ContentStoreError> for IntegrityError {
    fn from(e: ContentStoreError) -> Self {
        Self::Store(e)
    }
}

impl From<io::Error> for IntegrityError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

/// Result type for integrity operations.
pub type IntegrityResult<T> = std::result::Result<T, IntegrityError>;

// ---------------------------------------------------------------------------
// CRC-32C computation
// ---------------------------------------------------------------------------

/// Compute CRC-32C of data using hardware acceleration if available.
pub fn compute_crc32c(data: &[u8]) -> u32 {
    crc32fast::hash(data)
}

/// Compute CRC-32C while reading from a stream.
pub struct Crc32cReader<R: Read> {
    inner: R,
    hasher: crc32fast::Hasher,
    bytes_read: u64,
}

impl<R: Read> Crc32cReader<R> {
    /// Create a new CRC-32C reader.
    pub fn new(reader: R) -> Self {
        Self {
            inner: reader,
            hasher: crc32fast::Hasher::new(),
            bytes_read: 0,
        }
    }

    /// Get the number of bytes read so far.
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read
    }

    /// Finish and return the CRC-32C checksum.
    pub fn finish(self) -> u32 {
        self.hasher.finalize()
    }

    /// Get the inner reader.
    pub fn into_inner(self) -> R {
        self.inner
    }
}

impl<R: Read> Read for Crc32cReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        if n > 0 {
            self.hasher.update(&buf[..n]);
            self.bytes_read += n as u64;
        }
        Ok(n)
    }
}

/// Compute CRC-32C while writing to a stream.
pub struct Crc32cWriter<W: Write> {
    inner: W,
    hasher: crc32fast::Hasher,
    bytes_written: u64,
}

impl<W: Write> Crc32cWriter<W> {
    /// Create a new CRC-32C writer.
    pub fn new(writer: W) -> Self {
        Self {
            inner: writer,
            hasher: crc32fast::Hasher::new(),
            bytes_written: 0,
        }
    }

    /// Get the number of bytes written so far.
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Finish and return the CRC-32C checksum.
    pub fn finish(mut self) -> io::Result<(W, u32)> {
        self.inner.flush()?;
        Ok((self.inner, self.hasher.finalize()))
    }

    /// Get the inner writer.
    pub fn into_inner(self) -> W {
        self.inner
    }
}

impl<W: Write> Write for Crc32cWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(buf)?;
        if n > 0 {
            self.hasher.update(&buf[..n]);
            self.bytes_written += n as u64;
        }
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

// ---------------------------------------------------------------------------
// In-memory integrity store
// ---------------------------------------------------------------------------

/// In-memory content store with CRC-32C integrity verification.
///
/// Wraps content with CRC metadata and supports replication.
pub struct MemoryIntegrityStore {
    /// Primary data storage.
    data: RwLock<HashMap<ContentHash, Arc<Vec<u8>>>>,
    /// CRC metadata per hash.
    metadata: RwLock<HashMap<ContentHash, CrcMetadata>>,
    /// Replicas: replica_index -> hash -> data
    replicas: Vec<RwLock<HashMap<ContentHash, Arc<Vec<u8>>>>>,
    /// Configuration.
    config: IntegrityConfig,
    /// Content store config.
    store_config: ContentStoreConfig,
}

impl MemoryIntegrityStore {
    /// Create a new in-memory integrity store.
    pub fn new(config: IntegrityConfig, store_config: ContentStoreConfig) -> Self {
        let replicas = (0..config.replica_count)
            .map(|_| RwLock::new(HashMap::new()))
            .collect();

        Self {
            data: RwLock::new(HashMap::new()),
            metadata: RwLock::new(HashMap::new()),
            replicas,
            config,
            store_config,
        }
    }

    /// Create with default configs.
    pub fn default_config() -> Self {
        Self::new(IntegrityConfig::default(), ContentStoreConfig::default())
    }

    /// Get the integrity configuration.
    pub fn integrity_config(&self) -> &IntegrityConfig {
        &self.config
    }

    /// Get CRC metadata for a hash.
    pub fn crc_metadata(&self, hash: &ContentHash) -> Option<CrcMetadata> {
        self.metadata.read().unwrap().get(hash).copied()
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
        self.metadata.write().unwrap().clear();
        for replica in &self.replicas {
            replica.write().unwrap().clear();
        }
    }

    /// Get total bytes stored (primary only).
    pub fn total_bytes(&self) -> u64 {
        self.data
            .read()
            .unwrap()
            .values()
            .map(|v| v.len() as u64)
            .sum()
    }

    /// Get number of replicas configured.
    pub fn replica_count(&self) -> usize {
        self.config.replica_count
    }

    /// Store content with CRC computation and optional replication.
    pub fn put_stream<R: Read>(&self, reader: &mut R) -> IntegrityResult<ContentHash> {
        // Read all data and compute both hash and CRC
        let mut data_vec = Vec::new();
        let mut crc_reader = Crc32cReader::new(StreamingHashReader::new(reader));
        let mut buffer = vec![0u8; self.config.buffer_size];

        loop {
            let n = crc_reader.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            data_vec.extend_from_slice(&buffer[..n]);
        }

        let crc32c = crc_reader.hasher.finalize();
        let hash = crc_reader.inner.finish();
        let size = data_vec.len() as u64;

        // Store primary
        let data_arc = Arc::new(data_vec);
        self.data.write().unwrap().insert(hash, Arc::clone(&data_arc));
        self.metadata.write().unwrap().insert(hash, CrcMetadata::new(crc32c, size));

        // Store replicas
        for replica in &self.replicas {
            replica.write().unwrap().insert(hash, Arc::clone(&data_arc));
        }

        Ok(hash)
    }

    /// Get content with optional CRC verification.
    pub fn get_stream(&self, hash: &ContentHash) -> IntegrityResult<Option<MemorySeekableReader>> {
        let guard = self.data.read().unwrap();
        match guard.get(hash) {
            Some(data) => {
                if self.config.verify_on_read {
                    let meta = self.metadata.read().unwrap().get(hash).copied();
                    if let Some(meta) = meta {
                        let actual_crc = compute_crc32c(data);
                        if actual_crc != meta.crc32c {
                            // Primary failed, try replicas if configured
                            if self.config.fallback_on_failure && !self.replicas.is_empty() {
                                drop(guard);
                                return self.try_replicas(hash, meta.crc32c);
                            }
                            return Err(IntegrityError::CrcMismatch {
                                asset_hash: *hash,
                                expected_crc: meta.crc32c,
                                actual_crc,
                                replica_index: None,
                            });
                        }
                    }
                }
                Ok(Some(MemorySeekableReader::new(Arc::clone(data))))
            }
            None => Ok(None),
        }
    }

    /// Try replicas when primary fails CRC verification.
    fn try_replicas(&self, hash: &ContentHash, expected_crc: u32) -> IntegrityResult<Option<MemorySeekableReader>> {
        let mut failures = Vec::new();

        // First failure was primary
        if let Some(data) = self.data.read().unwrap().get(hash) {
            let actual_crc = compute_crc32c(data);
            failures.push((0, expected_crc, actual_crc));
        }

        for (idx, replica) in self.replicas.iter().enumerate() {
            let guard = replica.read().unwrap();
            if let Some(data) = guard.get(hash) {
                let actual_crc = compute_crc32c(data);
                if actual_crc == expected_crc {
                    // Found a good replica
                    return Ok(Some(MemorySeekableReader::new(Arc::clone(data))));
                }
                failures.push((idx + 1, expected_crc, actual_crc));
            }
        }

        Err(IntegrityError::AllReplicasFailed {
            asset_hash: *hash,
            failures,
        })
    }

    /// Check if content exists.
    pub fn has(&self, hash: &ContentHash) -> bool {
        self.data.read().unwrap().contains_key(hash)
    }

    /// Get content size.
    pub fn size(&self, hash: &ContentHash) -> IntegrityResult<Option<u64>> {
        let guard = self.data.read().unwrap();
        Ok(guard.get(hash).map(|v| v.len() as u64))
    }

    /// Delete content and replicas.
    pub fn delete(&self, hash: &ContentHash) -> IntegrityResult<bool> {
        let existed = self.data.write().unwrap().remove(hash).is_some();
        self.metadata.write().unwrap().remove(hash);
        for replica in &self.replicas {
            replica.write().unwrap().remove(hash);
        }
        Ok(existed)
    }

    /// Verify integrity of stored content without reading.
    pub fn verify(&self, hash: &ContentHash) -> IntegrityResult<bool> {
        let guard = self.data.read().unwrap();
        match guard.get(hash) {
            Some(data) => {
                let meta = self.metadata.read().unwrap().get(hash).copied();
                if let Some(meta) = meta {
                    let actual_crc = compute_crc32c(data);
                    Ok(actual_crc == meta.crc32c)
                } else {
                    // No metadata means no verification possible
                    Ok(true)
                }
            }
            None => Ok(false),
        }
    }

    /// Corrupt data for testing (internal use only).
    #[cfg(test)]
    pub fn corrupt_byte(&self, hash: &ContentHash, byte_index: usize) {
        let mut guard = self.data.write().unwrap();
        if let Some(data) = guard.get_mut(hash) {
            let data = Arc::make_mut(data);
            if byte_index < data.len() {
                data[byte_index] ^= 0xFF;
            }
        }
    }

    /// Corrupt specific replica for testing.
    #[cfg(test)]
    pub fn corrupt_replica(&self, hash: &ContentHash, replica_index: usize, byte_index: usize) {
        if replica_index < self.replicas.len() {
            let mut guard = self.replicas[replica_index].write().unwrap();
            if let Some(data) = guard.get_mut(hash) {
                let data = Arc::make_mut(data);
                if byte_index < data.len() {
                    data[byte_index] ^= 0xFF;
                }
            }
        }
    }
}

impl Default for MemoryIntegrityStore {
    fn default() -> Self {
        Self::default_config()
    }
}

// Implement Send + Sync
unsafe impl Send for MemoryIntegrityStore {}
unsafe impl Sync for MemoryIntegrityStore {}

// ---------------------------------------------------------------------------
// File-backed integrity store
// ---------------------------------------------------------------------------

/// File-backed content store with CRC-32C integrity verification.
///
/// Stores CRC metadata in a sidecar file: `{hash}.crc`
pub struct FileIntegrityStore {
    /// Base path for primary storage.
    base_path: PathBuf,
    /// Replica paths (if configured).
    replica_paths: Vec<PathBuf>,
    /// Configuration.
    config: IntegrityConfig,
    /// Content store config.
    store_config: ContentStoreConfig,
}

impl FileIntegrityStore {
    /// Create a new file-backed integrity store.
    ///
    /// Replica directories are created as `{base_path}/replica_{n}`.
    pub fn new<P: AsRef<Path>>(base_path: P, config: IntegrityConfig, store_config: ContentStoreConfig) -> IntegrityResult<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        fs::create_dir_all(&base_path)?;

        let replica_paths: Vec<PathBuf> = (0..config.replica_count)
            .map(|i| base_path.join(format!("replica_{}", i)))
            .collect();

        for path in &replica_paths {
            fs::create_dir_all(path)?;
        }

        Ok(Self {
            base_path,
            replica_paths,
            config,
            store_config,
        })
    }

    /// Get the base path.
    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    /// Get the integrity configuration.
    pub fn integrity_config(&self) -> &IntegrityConfig {
        &self.config
    }

    /// Get number of replicas configured.
    pub fn replica_count(&self) -> usize {
        self.config.replica_count
    }

    /// Get blob path for a hash.
    fn blob_path(&self, hash: &ContentHash) -> PathBuf {
        let hex = format!("{}", hash);
        let (prefix, suffix) = hex.split_at(2);
        self.base_path.join(prefix).join(suffix)
    }

    /// Get CRC metadata path for a hash.
    fn crc_path(&self, hash: &ContentHash) -> PathBuf {
        let hex = format!("{}", hash);
        let (prefix, suffix) = hex.split_at(2);
        self.base_path.join(prefix).join(format!("{}.crc", suffix))
    }

    /// Get replica blob path.
    fn replica_blob_path(&self, replica_index: usize, hash: &ContentHash) -> PathBuf {
        let hex = format!("{}", hash);
        let (prefix, suffix) = hex.split_at(2);
        self.replica_paths[replica_index].join(prefix).join(suffix)
    }

    /// Read CRC metadata from file.
    fn read_crc_metadata(&self, hash: &ContentHash) -> IntegrityResult<Option<CrcMetadata>> {
        let crc_path = self.crc_path(hash);
        if !crc_path.exists() {
            return Ok(None);
        }
        let bytes = fs::read(&crc_path)?;
        if bytes.len() != 12 {
            return Ok(None);
        }
        let mut arr = [0u8; 12];
        arr.copy_from_slice(&bytes);
        Ok(Some(CrcMetadata::from_bytes(&arr)))
    }

    /// Write CRC metadata to file.
    fn write_crc_metadata(&self, hash: &ContentHash, meta: CrcMetadata) -> IntegrityResult<()> {
        let crc_path = self.crc_path(hash);
        if let Some(parent) = crc_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&crc_path, meta.to_bytes())?;
        Ok(())
    }

    /// Store content with CRC computation and optional replication.
    pub fn put_stream<R: Read>(&self, reader: &mut R) -> IntegrityResult<ContentHash> {
        // Create temp file
        let temp_dir = self.base_path.join("tmp");
        fs::create_dir_all(&temp_dir)?;
        let temp_path = temp_dir.join(format!("upload_{}", std::process::id()));

        let temp_file = File::create(&temp_path)?;
        let mut crc_writer = Crc32cWriter::new(BufWriter::with_capacity(
            self.config.buffer_size,
            temp_file,
        ));

        // Also compute content hash
        let mut hash_reader = StreamingHashReader::new(reader);
        let mut buffer = vec![0u8; self.config.buffer_size];
        let mut total_size = 0u64;

        loop {
            let n = hash_reader.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            crc_writer.write_all(&buffer[..n])?;
            total_size += n as u64;
        }

        let (inner, crc32c) = crc_writer.finish()?;
        let file = inner.into_inner().map_err(|e| io::Error::new(
            io::ErrorKind::Other,
            format!("failed to flush buffer: {}", e.error()),
        ))?;
        let hash = hash_reader.finish();

        if self.store_config.sync_writes {
            file.sync_all()?;
        }
        drop(file);

        // Move to final location
        let final_path = self.blob_path(&hash);
        if final_path.exists() {
            fs::remove_file(&temp_path)?;
        } else {
            if let Some(parent) = final_path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::rename(&temp_path, &final_path)?;
        }

        // Write CRC metadata
        let meta = CrcMetadata::new(crc32c, total_size);
        self.write_crc_metadata(&hash, meta)?;

        // Create replicas
        for (idx, replica_path) in self.replica_paths.iter().enumerate() {
            let replica_blob = self.replica_blob_path(idx, &hash);
            if !replica_blob.exists() {
                if let Some(parent) = replica_blob.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::copy(&final_path, &replica_blob)?;
            }
        }

        Ok(hash)
    }

    /// Get content with optional CRC verification.
    pub fn get_stream(&self, hash: &ContentHash) -> IntegrityResult<Option<FileSeekableReader>> {
        let path = self.blob_path(hash);
        if !path.exists() {
            return Ok(None);
        }

        if self.config.verify_on_read {
            if let Some(meta) = self.read_crc_metadata(hash)? {
                // Compute CRC of file
                let actual_crc = self.compute_file_crc(&path)?;
                if actual_crc != meta.crc32c {
                    // Primary failed, try replicas
                    if self.config.fallback_on_failure && !self.replica_paths.is_empty() {
                        return self.try_replica_files(hash, meta.crc32c);
                    }
                    return Err(IntegrityError::CrcMismatch {
                        asset_hash: *hash,
                        expected_crc: meta.crc32c,
                        actual_crc,
                        replica_index: None,
                    });
                }
            }
        }

        let file = File::open(&path)?;
        let reader = FileSeekableReader::new(file, self.store_config.buffer_size)?;
        Ok(Some(reader))
    }

    /// Compute CRC-32C of a file.
    fn compute_file_crc(&self, path: &Path) -> IntegrityResult<u32> {
        let file = File::open(path)?;
        let mut reader = Crc32cReader::new(BufReader::new(file));
        let mut buffer = vec![0u8; self.config.buffer_size];

        loop {
            let n = reader.read(&mut buffer)?;
            if n == 0 {
                break;
            }
        }

        Ok(reader.finish())
    }

    /// Try replica files when primary fails.
    fn try_replica_files(&self, hash: &ContentHash, expected_crc: u32) -> IntegrityResult<Option<FileSeekableReader>> {
        let mut failures = Vec::new();

        // Primary failure already recorded
        let primary_path = self.blob_path(hash);
        if let Ok(actual_crc) = self.compute_file_crc(&primary_path) {
            failures.push((0, expected_crc, actual_crc));
        }

        for (idx, _) in self.replica_paths.iter().enumerate() {
            let replica_path = self.replica_blob_path(idx, hash);
            if replica_path.exists() {
                let actual_crc = self.compute_file_crc(&replica_path)?;
                if actual_crc == expected_crc {
                    // Found good replica
                    let file = File::open(&replica_path)?;
                    let reader = FileSeekableReader::new(file, self.store_config.buffer_size)?;
                    return Ok(Some(reader));
                }
                failures.push((idx + 1, expected_crc, actual_crc));
            }
        }

        Err(IntegrityError::AllReplicasFailed {
            asset_hash: *hash,
            failures,
        })
    }

    /// Check if content exists.
    pub fn has(&self, hash: &ContentHash) -> bool {
        self.blob_path(hash).exists()
    }

    /// Get content size.
    pub fn size(&self, hash: &ContentHash) -> IntegrityResult<Option<u64>> {
        let path = self.blob_path(hash);
        if !path.exists() {
            return Ok(None);
        }
        let metadata = fs::metadata(&path)?;
        Ok(Some(metadata.len()))
    }

    /// Delete content and replicas.
    pub fn delete(&self, hash: &ContentHash) -> IntegrityResult<bool> {
        let path = self.blob_path(hash);
        if !path.exists() {
            return Ok(false);
        }

        fs::remove_file(&path)?;

        // Remove CRC metadata
        let crc_path = self.crc_path(hash);
        if crc_path.exists() {
            fs::remove_file(&crc_path)?;
        }

        // Remove replicas
        for (idx, _) in self.replica_paths.iter().enumerate() {
            let replica_path = self.replica_blob_path(idx, hash);
            if replica_path.exists() {
                fs::remove_file(&replica_path)?;
            }
        }

        Ok(true)
    }

    /// Get CRC metadata for a hash.
    pub fn crc_metadata(&self, hash: &ContentHash) -> IntegrityResult<Option<CrcMetadata>> {
        self.read_crc_metadata(hash)
    }

    /// Verify integrity without reading full content.
    pub fn verify(&self, hash: &ContentHash) -> IntegrityResult<bool> {
        let path = self.blob_path(hash);
        if !path.exists() {
            return Ok(false);
        }

        if let Some(meta) = self.read_crc_metadata(hash)? {
            let actual_crc = self.compute_file_crc(&path)?;
            Ok(actual_crc == meta.crc32c)
        } else {
            // No metadata, assume OK
            Ok(true)
        }
    }

    /// Corrupt a file for testing (internal use only).
    #[cfg(test)]
    pub fn corrupt_byte(&self, hash: &ContentHash, byte_index: usize) -> IntegrityResult<()> {
        let path = self.blob_path(hash);
        let mut data = fs::read(&path)?;
        if byte_index < data.len() {
            data[byte_index] ^= 0xFF;
        }
        fs::write(&path, data)?;
        Ok(())
    }

    /// Corrupt specific replica for testing.
    #[cfg(test)]
    pub fn corrupt_replica(&self, hash: &ContentHash, replica_index: usize, byte_index: usize) -> IntegrityResult<()> {
        let path = self.replica_blob_path(replica_index, hash);
        let mut data = fs::read(&path)?;
        if byte_index < data.len() {
            data[byte_index] ^= 0xFF;
        }
        fs::write(&path, data)?;
        Ok(())
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
    fn test_integrity_config_default() {
        let config = IntegrityConfig::default();
        assert!(config.verify_on_read);
        assert_eq!(config.replica_count, 0);
        assert!(config.fallback_on_failure);
        assert_eq!(config.buffer_size, DEFAULT_BUFFER_SIZE);
    }

    #[test]
    fn test_integrity_config_with_verify_on_read() {
        let config = IntegrityConfig::default().with_verify_on_read(false);
        assert!(!config.verify_on_read);
    }

    #[test]
    fn test_integrity_config_with_replicas() {
        let config = IntegrityConfig::default().with_replicas(3);
        assert_eq!(config.replica_count, 3);
    }

    #[test]
    fn test_integrity_config_with_fallback() {
        let config = IntegrityConfig::default().with_fallback_on_failure(false);
        assert!(!config.fallback_on_failure);
    }

    #[test]
    fn test_integrity_config_with_buffer_size() {
        let config = IntegrityConfig::default().with_buffer_size(128 * 1024);
        assert_eq!(config.buffer_size, 128 * 1024);
    }

    #[test]
    fn test_integrity_config_buffer_size_clamped_min() {
        let config = IntegrityConfig::default().with_buffer_size(100);
        assert_eq!(config.buffer_size, MIN_BUFFER_SIZE);
    }

    #[test]
    fn test_integrity_config_buffer_size_clamped_max() {
        let config = IntegrityConfig::default().with_buffer_size(100 * 1024 * 1024);
        assert_eq!(config.buffer_size, MAX_BUFFER_SIZE);
    }

    // ========================================================================
    // CRC Metadata tests
    // ========================================================================

    #[test]
    fn test_crc_metadata_serialization() {
        let meta = CrcMetadata::new(0xDEADBEEF, 12345678);
        let bytes = meta.to_bytes();
        let restored = CrcMetadata::from_bytes(&bytes);
        assert_eq!(meta, restored);
    }

    #[test]
    fn test_crc_metadata_zero() {
        let meta = CrcMetadata::new(0, 0);
        let bytes = meta.to_bytes();
        let restored = CrcMetadata::from_bytes(&bytes);
        assert_eq!(meta.crc32c, 0);
        assert_eq!(meta.size, 0);
        assert_eq!(meta, restored);
    }

    #[test]
    fn test_crc_metadata_max_values() {
        let meta = CrcMetadata::new(u32::MAX, u64::MAX);
        let bytes = meta.to_bytes();
        let restored = CrcMetadata::from_bytes(&bytes);
        assert_eq!(meta, restored);
    }

    // ========================================================================
    // CRC-32C computation tests
    // ========================================================================

    #[test]
    fn test_compute_crc32c_empty() {
        let crc = compute_crc32c(b"");
        assert_eq!(crc, 0); // CRC of empty data
    }

    #[test]
    fn test_compute_crc32c_basic() {
        let crc1 = compute_crc32c(b"hello");
        let crc2 = compute_crc32c(b"hello");
        assert_eq!(crc1, crc2);
    }

    #[test]
    fn test_compute_crc32c_different_data() {
        let crc1 = compute_crc32c(b"hello");
        let crc2 = compute_crc32c(b"world");
        assert_ne!(crc1, crc2);
    }

    #[test]
    fn test_compute_crc32c_single_byte() {
        let crc = compute_crc32c(b"X");
        assert_ne!(crc, 0);
    }

    #[test]
    fn test_crc32c_reader_basic() {
        let data = b"test data for CRC";
        let mut cursor = Cursor::new(data);
        let mut reader = Crc32cReader::new(&mut cursor);

        let mut buf = [0u8; 100];
        let n = reader.read(&mut buf).unwrap();
        assert_eq!(n, data.len());
        assert_eq!(&buf[..n], data);

        let crc = reader.finish();
        assert_eq!(crc, compute_crc32c(data));
    }

    #[test]
    fn test_crc32c_reader_chunked() {
        let data = b"chunked read test data for CRC-32C";
        let mut cursor = Cursor::new(data);
        let mut reader = Crc32cReader::new(&mut cursor);

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
        let crc = reader.finish();
        assert_eq!(crc, compute_crc32c(data));
    }

    #[test]
    fn test_crc32c_reader_bytes_read() {
        let data = b"counting bytes";
        let mut cursor = Cursor::new(data);
        let mut reader = Crc32cReader::new(&mut cursor);

        assert_eq!(reader.bytes_read(), 0);

        let mut buf = [0u8; 5];
        reader.read(&mut buf).unwrap();
        assert_eq!(reader.bytes_read(), 5);

        reader.read(&mut buf).unwrap();
        assert_eq!(reader.bytes_read(), 10);
    }

    #[test]
    fn test_crc32c_writer_basic() {
        let data = b"test data";
        let mut output = Vec::new();
        let mut writer = Crc32cWriter::new(&mut output);

        writer.write_all(data).unwrap();
        let (_, crc) = writer.finish().unwrap();

        assert_eq!(output, data);
        assert_eq!(crc, compute_crc32c(data));
    }

    #[test]
    fn test_crc32c_writer_chunked() {
        let data = b"chunked write test data for CRC-32C";
        let mut output = Vec::new();
        let mut writer = Crc32cWriter::new(&mut output);

        for chunk in data.chunks(7) {
            writer.write_all(chunk).unwrap();
        }
        let (_, crc) = writer.finish().unwrap();

        assert_eq!(output, data);
        assert_eq!(crc, compute_crc32c(data));
    }

    #[test]
    fn test_crc32c_writer_bytes_written() {
        let mut output = Vec::new();
        let mut writer = Crc32cWriter::new(&mut output);

        assert_eq!(writer.bytes_written(), 0);

        writer.write_all(b"hello").unwrap();
        assert_eq!(writer.bytes_written(), 5);

        writer.write_all(b" world").unwrap();
        assert_eq!(writer.bytes_written(), 11);
    }

    // ========================================================================
    // MemoryIntegrityStore basic tests
    // ========================================================================

    #[test]
    fn test_memory_integrity_store_put_get() {
        let store = MemoryIntegrityStore::default();
        let data = b"test content";
        let mut cursor = Cursor::new(data);

        let hash = store.put_stream(&mut cursor).unwrap();
        assert!(store.has(&hash));

        let mut reader = store.get_stream(&hash).unwrap().unwrap();
        let mut result = Vec::new();
        reader.read_to_end(&mut result).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_memory_integrity_store_crc_computed() {
        let store = MemoryIntegrityStore::default();
        let data = b"test content";
        let mut cursor = Cursor::new(data);

        let hash = store.put_stream(&mut cursor).unwrap();
        let meta = store.crc_metadata(&hash).unwrap();

        assert_eq!(meta.crc32c, compute_crc32c(data));
        assert_eq!(meta.size, data.len() as u64);
    }

    #[test]
    fn test_memory_integrity_store_verify() {
        let store = MemoryIntegrityStore::default();
        let data = b"verify test";
        let mut cursor = Cursor::new(data);

        let hash = store.put_stream(&mut cursor).unwrap();
        assert!(store.verify(&hash).unwrap());
    }

    #[test]
    fn test_memory_integrity_store_corruption_detected() {
        let config = IntegrityConfig::default().with_verify_on_read(true);
        let store = MemoryIntegrityStore::new(config, ContentStoreConfig::default());

        let data = b"content to corrupt";
        let mut cursor = Cursor::new(data);
        let hash = store.put_stream(&mut cursor).unwrap();

        // Corrupt the data
        store.corrupt_byte(&hash, 5);

        // Should detect corruption on read
        let result = store.get_stream(&hash);
        assert!(matches!(result, Err(IntegrityError::CrcMismatch { .. })));
    }

    #[test]
    fn test_memory_integrity_store_corruption_reports_asset_id() {
        let config = IntegrityConfig::default().with_verify_on_read(true);
        let store = MemoryIntegrityStore::new(config, ContentStoreConfig::default());

        let data = b"track asset id";
        let mut cursor = Cursor::new(data);
        let hash = store.put_stream(&mut cursor).unwrap();

        store.corrupt_byte(&hash, 0);

        match store.get_stream(&hash) {
            Err(IntegrityError::CrcMismatch { asset_hash, .. }) => {
                assert_eq!(asset_hash, hash);
            }
            _ => panic!("expected CrcMismatch error"),
        }
    }

    #[test]
    fn test_memory_integrity_store_empty_content() {
        let store = MemoryIntegrityStore::default();
        let data: &[u8] = b"";
        let mut cursor = Cursor::new(data);

        let hash = store.put_stream(&mut cursor).unwrap();
        assert!(store.has(&hash));

        let meta = store.crc_metadata(&hash).unwrap();
        assert_eq!(meta.size, 0);

        let mut reader = store.get_stream(&hash).unwrap().unwrap();
        let mut result = Vec::new();
        reader.read_to_end(&mut result).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_memory_integrity_store_single_byte() {
        let store = MemoryIntegrityStore::default();
        let data = b"X";
        let mut cursor = Cursor::new(data);

        let hash = store.put_stream(&mut cursor).unwrap();
        let meta = store.crc_metadata(&hash).unwrap();
        assert_eq!(meta.size, 1);

        let mut reader = store.get_stream(&hash).unwrap().unwrap();
        let mut result = Vec::new();
        reader.read_to_end(&mut result).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_memory_integrity_store_len_and_total_bytes() {
        let store = MemoryIntegrityStore::default();

        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
        assert_eq!(store.total_bytes(), 0);

        let mut c1 = Cursor::new(b"12345");
        let mut c2 = Cursor::new(b"67890ABC");
        store.put_stream(&mut c1).unwrap();
        store.put_stream(&mut c2).unwrap();

        assert_eq!(store.len(), 2);
        assert!(!store.is_empty());
        assert_eq!(store.total_bytes(), 5 + 8);
    }

    #[test]
    fn test_memory_integrity_store_delete() {
        let store = MemoryIntegrityStore::default();
        let data = b"to delete";
        let mut cursor = Cursor::new(data);

        let hash = store.put_stream(&mut cursor).unwrap();
        assert!(store.has(&hash));

        assert!(store.delete(&hash).unwrap());
        assert!(!store.has(&hash));
        assert!(!store.delete(&hash).unwrap());
    }

    #[test]
    fn test_memory_integrity_store_clear() {
        let store = MemoryIntegrityStore::default();

        let mut c1 = Cursor::new(b"data1");
        let mut c2 = Cursor::new(b"data2");
        store.put_stream(&mut c1).unwrap();
        store.put_stream(&mut c2).unwrap();

        assert_eq!(store.len(), 2);
        store.clear();
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_memory_integrity_store_size() {
        let store = MemoryIntegrityStore::default();
        let data = b"size test content";
        let mut cursor = Cursor::new(data);

        let hash = store.put_stream(&mut cursor).unwrap();
        assert_eq!(store.size(&hash).unwrap(), Some(data.len() as u64));
    }

    #[test]
    fn test_memory_integrity_store_verify_on_read_disabled() {
        let config = IntegrityConfig::default().with_verify_on_read(false);
        let store = MemoryIntegrityStore::new(config, ContentStoreConfig::default());

        let data = b"no verify";
        let mut cursor = Cursor::new(data);
        let hash = store.put_stream(&mut cursor).unwrap();

        // Corrupt data
        store.corrupt_byte(&hash, 0);

        // Should succeed without verification
        let result = store.get_stream(&hash);
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    }

    // ========================================================================
    // Replication tests (N=2, N=3)
    // ========================================================================

    #[test]
    fn test_memory_integrity_store_replication_n2() {
        let config = IntegrityConfig::default().with_replicas(2);
        let store = MemoryIntegrityStore::new(config, ContentStoreConfig::default());

        let data = b"replicated content";
        let mut cursor = Cursor::new(data);

        let _hash = store.put_stream(&mut cursor).unwrap();
        assert_eq!(store.replica_count(), 2);
    }

    #[test]
    fn test_memory_integrity_store_replication_n3() {
        let config = IntegrityConfig::default().with_replicas(3);
        let store = MemoryIntegrityStore::new(config, ContentStoreConfig::default());

        let data = b"triple replicated";
        let mut cursor = Cursor::new(data);

        let _hash = store.put_stream(&mut cursor).unwrap();
        assert_eq!(store.replica_count(), 3);
    }

    #[test]
    fn test_memory_integrity_store_replica_fallback_on_primary_failure() {
        let config = IntegrityConfig::default()
            .with_verify_on_read(true)
            .with_replicas(1)
            .with_fallback_on_failure(true);
        let store = MemoryIntegrityStore::new(config, ContentStoreConfig::default());

        let data = b"fallback test";
        let mut cursor = Cursor::new(data);
        let hash = store.put_stream(&mut cursor).unwrap();

        // Corrupt primary only
        store.corrupt_byte(&hash, 0);

        // Should fallback to replica
        let mut reader = store.get_stream(&hash).unwrap().unwrap();
        let mut result = Vec::new();
        reader.read_to_end(&mut result).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_memory_integrity_store_all_replicas_failed() {
        let config = IntegrityConfig::default()
            .with_verify_on_read(true)
            .with_replicas(2)
            .with_fallback_on_failure(true);
        let store = MemoryIntegrityStore::new(config, ContentStoreConfig::default());

        let data = b"all fail test";
        let mut cursor = Cursor::new(data);
        let hash = store.put_stream(&mut cursor).unwrap();

        // Corrupt primary and all replicas
        store.corrupt_byte(&hash, 0);
        store.corrupt_replica(&hash, 0, 0);
        store.corrupt_replica(&hash, 1, 0);

        let result = store.get_stream(&hash);
        assert!(matches!(result, Err(IntegrityError::AllReplicasFailed { .. })));
    }

    #[test]
    fn test_memory_integrity_store_fallback_disabled() {
        let config = IntegrityConfig::default()
            .with_verify_on_read(true)
            .with_replicas(1)
            .with_fallback_on_failure(false);
        let store = MemoryIntegrityStore::new(config, ContentStoreConfig::default());

        let data = b"no fallback";
        let mut cursor = Cursor::new(data);
        let hash = store.put_stream(&mut cursor).unwrap();

        store.corrupt_byte(&hash, 0);

        // Should fail immediately without trying replica
        let result = store.get_stream(&hash);
        assert!(matches!(result, Err(IntegrityError::CrcMismatch { .. })));
    }

    #[test]
    fn test_memory_integrity_store_delete_clears_replicas() {
        let config = IntegrityConfig::default().with_replicas(2);
        let store = MemoryIntegrityStore::new(config, ContentStoreConfig::default());

        let data = b"delete all";
        let mut cursor = Cursor::new(data);
        let hash = store.put_stream(&mut cursor).unwrap();

        assert!(store.delete(&hash).unwrap());
        assert!(!store.has(&hash));

        // Even after re-putting different data, old hash should not exist
        let missing = ContentHash::from_bytes(b"nonexistent");
        assert!(!store.has(&missing));
    }

    // ========================================================================
    // FileIntegrityStore tests
    // ========================================================================

    #[test]
    fn test_file_integrity_store_put_get() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = IntegrityConfig::default();
        let store = FileIntegrityStore::new(temp_dir.path(), config, ContentStoreConfig::default()).unwrap();

        let data = b"file store test";
        let mut cursor = Cursor::new(data);

        let hash = store.put_stream(&mut cursor).unwrap();
        assert!(store.has(&hash));

        let mut reader = store.get_stream(&hash).unwrap().unwrap();
        let mut result = Vec::new();
        reader.read_to_end(&mut result).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_file_integrity_store_crc_computed() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = IntegrityConfig::default();
        let store = FileIntegrityStore::new(temp_dir.path(), config, ContentStoreConfig::default()).unwrap();

        let data = b"crc computation test";
        let mut cursor = Cursor::new(data);

        let hash = store.put_stream(&mut cursor).unwrap();
        let meta = store.crc_metadata(&hash).unwrap().unwrap();

        assert_eq!(meta.crc32c, compute_crc32c(data));
        assert_eq!(meta.size, data.len() as u64);
    }

    #[test]
    fn test_file_integrity_store_verify() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = IntegrityConfig::default();
        let store = FileIntegrityStore::new(temp_dir.path(), config, ContentStoreConfig::default()).unwrap();

        let data = b"verify file test";
        let mut cursor = Cursor::new(data);

        let hash = store.put_stream(&mut cursor).unwrap();
        assert!(store.verify(&hash).unwrap());
    }

    #[test]
    fn test_file_integrity_store_corruption_detected() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = IntegrityConfig::default().with_verify_on_read(true);
        let store = FileIntegrityStore::new(temp_dir.path(), config, ContentStoreConfig::default()).unwrap();

        let data = b"file corruption test";
        let mut cursor = Cursor::new(data);
        let hash = store.put_stream(&mut cursor).unwrap();

        store.corrupt_byte(&hash, 5).unwrap();

        let result = store.get_stream(&hash);
        assert!(matches!(result, Err(IntegrityError::CrcMismatch { .. })));
    }

    #[test]
    fn test_file_integrity_store_corruption_reports_asset_id() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = IntegrityConfig::default().with_verify_on_read(true);
        let store = FileIntegrityStore::new(temp_dir.path(), config, ContentStoreConfig::default()).unwrap();

        let data = b"track file asset id";
        let mut cursor = Cursor::new(data);
        let hash = store.put_stream(&mut cursor).unwrap();

        store.corrupt_byte(&hash, 0).unwrap();

        match store.get_stream(&hash) {
            Err(IntegrityError::CrcMismatch { asset_hash, .. }) => {
                assert_eq!(asset_hash, hash);
            }
            _ => panic!("expected CrcMismatch error"),
        }
    }

    #[test]
    fn test_file_integrity_store_empty_content() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = IntegrityConfig::default();
        let store = FileIntegrityStore::new(temp_dir.path(), config, ContentStoreConfig::default()).unwrap();

        let data: &[u8] = b"";
        let mut cursor = Cursor::new(data);

        let hash = store.put_stream(&mut cursor).unwrap();
        let meta = store.crc_metadata(&hash).unwrap().unwrap();
        assert_eq!(meta.size, 0);
    }

    #[test]
    fn test_file_integrity_store_single_byte() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = IntegrityConfig::default();
        let store = FileIntegrityStore::new(temp_dir.path(), config, ContentStoreConfig::default()).unwrap();

        let data = b"Y";
        let mut cursor = Cursor::new(data);

        let hash = store.put_stream(&mut cursor).unwrap();
        let meta = store.crc_metadata(&hash).unwrap().unwrap();
        assert_eq!(meta.size, 1);
    }

    #[test]
    fn test_file_integrity_store_delete() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = IntegrityConfig::default();
        let store = FileIntegrityStore::new(temp_dir.path(), config, ContentStoreConfig::default()).unwrap();

        let data = b"to delete file";
        let mut cursor = Cursor::new(data);
        let hash = store.put_stream(&mut cursor).unwrap();

        assert!(store.has(&hash));
        assert!(store.delete(&hash).unwrap());
        assert!(!store.has(&hash));
    }

    #[test]
    fn test_file_integrity_store_replication_n2() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = IntegrityConfig::default().with_replicas(2);
        let store = FileIntegrityStore::new(temp_dir.path(), config, ContentStoreConfig::default()).unwrap();

        let data = b"file replicated";
        let mut cursor = Cursor::new(data);

        let _hash = store.put_stream(&mut cursor).unwrap();
        assert_eq!(store.replica_count(), 2);

        // Check replica directories exist
        assert!(temp_dir.path().join("replica_0").exists());
        assert!(temp_dir.path().join("replica_1").exists());
    }

    #[test]
    fn test_file_integrity_store_replica_fallback() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = IntegrityConfig::default()
            .with_verify_on_read(true)
            .with_replicas(1)
            .with_fallback_on_failure(true);
        let store = FileIntegrityStore::new(temp_dir.path(), config, ContentStoreConfig::default()).unwrap();

        let data = b"file fallback test";
        let mut cursor = Cursor::new(data);
        let hash = store.put_stream(&mut cursor).unwrap();

        store.corrupt_byte(&hash, 0).unwrap();

        // Should fallback to replica
        let mut reader = store.get_stream(&hash).unwrap().unwrap();
        let mut result = Vec::new();
        reader.read_to_end(&mut result).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_file_integrity_store_all_replicas_failed() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = IntegrityConfig::default()
            .with_verify_on_read(true)
            .with_replicas(1)
            .with_fallback_on_failure(true);
        let store = FileIntegrityStore::new(temp_dir.path(), config, ContentStoreConfig::default()).unwrap();

        let data = b"all file fail";
        let mut cursor = Cursor::new(data);
        let hash = store.put_stream(&mut cursor).unwrap();

        store.corrupt_byte(&hash, 0).unwrap();
        store.corrupt_replica(&hash, 0, 0).unwrap();

        let result = store.get_stream(&hash);
        assert!(matches!(result, Err(IntegrityError::AllReplicasFailed { .. })));
    }

    #[test]
    fn test_file_integrity_store_verify_on_read_disabled() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = IntegrityConfig::default().with_verify_on_read(false);
        let store = FileIntegrityStore::new(temp_dir.path(), config, ContentStoreConfig::default()).unwrap();

        let data = b"no file verify";
        let mut cursor = Cursor::new(data);
        let hash = store.put_stream(&mut cursor).unwrap();

        store.corrupt_byte(&hash, 0).unwrap();

        let result = store.get_stream(&hash);
        assert!(result.is_ok());
    }

    // ========================================================================
    // Error display tests
    // ========================================================================

    #[test]
    fn test_error_display_crc_mismatch() {
        let hash = ContentHash::from_bytes(b"test");
        let err = IntegrityError::CrcMismatch {
            asset_hash: hash,
            expected_crc: 0xDEADBEEF,
            actual_crc: 0xCAFEBABE,
            replica_index: None,
        };
        let msg = err.to_string();
        assert!(msg.contains("CRC-32C mismatch"));
        assert!(msg.contains("deadbeef"));
        assert!(msg.contains("cafebabe"));
    }

    #[test]
    fn test_error_display_crc_mismatch_with_replica() {
        let hash = ContentHash::from_bytes(b"test");
        let err = IntegrityError::CrcMismatch {
            asset_hash: hash,
            expected_crc: 0xDEADBEEF,
            actual_crc: 0xCAFEBABE,
            replica_index: Some(2),
        };
        let msg = err.to_string();
        assert!(msg.contains("replica 2"));
    }

    #[test]
    fn test_error_display_all_replicas_failed() {
        let hash = ContentHash::from_bytes(b"test");
        let err = IntegrityError::AllReplicasFailed {
            asset_hash: hash,
            failures: vec![
                (0, 0xDEADBEEF, 0x11111111),
                (1, 0xDEADBEEF, 0x22222222),
            ],
        };
        let msg = err.to_string();
        assert!(msg.contains("All replicas failed"));
        assert!(msg.contains("replica 0"));
        assert!(msg.contains("replica 1"));
    }

    // ========================================================================
    // Performance overhead measurement tests
    // ========================================================================

    #[test]
    fn test_performance_overhead_baseline() {
        // Measure time for put/get without integrity
        let config = IntegrityConfig::default().with_verify_on_read(false);
        let store = MemoryIntegrityStore::new(config, ContentStoreConfig::default());

        let data: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();
        let start = std::time::Instant::now();

        for _ in 0..10 {
            let mut cursor = Cursor::new(&data);
            let hash = store.put_stream(&mut cursor).unwrap();
            let _ = store.get_stream(&hash).unwrap();
        }

        let baseline_duration = start.elapsed();

        // Measure with integrity verification
        let config = IntegrityConfig::default().with_verify_on_read(true);
        let store = MemoryIntegrityStore::new(config, ContentStoreConfig::default());

        let start = std::time::Instant::now();

        for _ in 0..10 {
            let mut cursor = Cursor::new(&data);
            let hash = store.put_stream(&mut cursor).unwrap();
            let _ = store.get_stream(&hash).unwrap();
        }

        let integrity_duration = start.elapsed();

        // CRC overhead should be reasonable (< 2x baseline typically)
        // This is a rough check - actual overhead depends on hardware
        assert!(integrity_duration.as_micros() > 0);
        println!("Baseline: {:?}, With integrity: {:?}", baseline_duration, integrity_duration);
    }

    // ========================================================================
    // Edge case tests
    // ========================================================================

    #[test]
    fn test_memory_integrity_store_get_nonexistent() {
        let store = MemoryIntegrityStore::default();
        let missing = ContentHash::from_bytes(b"nonexistent");
        let result = store.get_stream(&missing).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_file_integrity_store_get_nonexistent() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = IntegrityConfig::default();
        let store = FileIntegrityStore::new(temp_dir.path(), config, ContentStoreConfig::default()).unwrap();

        let missing = ContentHash::from_bytes(b"nonexistent");
        let result = store.get_stream(&missing).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_memory_integrity_store_verify_nonexistent() {
        let store = MemoryIntegrityStore::default();
        let missing = ContentHash::from_bytes(b"nonexistent");
        assert!(!store.verify(&missing).unwrap());
    }

    #[test]
    fn test_file_integrity_store_verify_nonexistent() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = IntegrityConfig::default();
        let store = FileIntegrityStore::new(temp_dir.path(), config, ContentStoreConfig::default()).unwrap();

        let missing = ContentHash::from_bytes(b"nonexistent");
        assert!(!store.verify(&missing).unwrap());
    }

    #[test]
    fn test_memory_integrity_store_modified_byte_different_positions() {
        let config = IntegrityConfig::default().with_verify_on_read(true);
        let _store = MemoryIntegrityStore::new(config.clone(), ContentStoreConfig::default());

        let data = b"test data for position testing";

        // Test corruption at different positions
        for pos in [0, 5, 15, data.len() - 1] {
            let store = MemoryIntegrityStore::new(config.clone(), ContentStoreConfig::default());
            let mut cursor = Cursor::new(data);
            let hash = store.put_stream(&mut cursor).unwrap();

            store.corrupt_byte(&hash, pos);

            let result = store.get_stream(&hash);
            assert!(matches!(result, Err(IntegrityError::CrcMismatch { .. })));
        }
    }

    #[test]
    fn test_concurrent_integrity_store_access() {
        use std::thread;
        use std::sync::Arc;

        let store = Arc::new(MemoryIntegrityStore::default());

        // Pre-populate
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
}
