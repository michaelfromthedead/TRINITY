//! Pipeline cache and management.
//!
//! Provides three abstractions for GPU pipeline lifecycle:
//!
//! - [`CachedPipeline`] -- a single compiled render pipeline with its
//!   bind-group layout and shader hash.
//! - [`ShaderCache`] -- deduplicates [`wgpu::ShaderModule`] allocations by
//!   keying on the SHA-256 hash of the WGSL source.
//! - [`PipelineTable`] -- a table of cached pipelines together with a shared
//!   shader cache and a convenience method for compiling new pipelines.
//!
//! # SHA-256 deduplication
//!
//! Every WGSL source string is hashed with SHA-256 **before** compilation.
//! If the same hash is encountered again the existing [`wgpu::ShaderModule`]
//! is returned, avoiding redundant GPU shader compilation.

use std::collections::HashMap;
use std::sync::Arc;

use sha2::{Digest, Sha256};

// ---------------------------------------------------------------------------
// ContentHash — SHA-256 content-addressed identifier
// ---------------------------------------------------------------------------

/// Hash algorithm selector for runtime algorithm choice.
///
/// By default, `ContentHash::from_bytes` uses compile-time feature selection:
/// - With `blake3` feature: uses BLAKE3
/// - Without `blake3` feature: uses SHA-256
///
/// For runtime algorithm selection, use `ContentHash::from_data_with_algo`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HashAlgorithm {
    /// SHA-256 (FIPS-180, 256-bit output)
    Sha256,
    /// BLAKE3 (faster, 256-bit output) - requires `blake3` feature
    #[cfg(feature = "blake3")]
    Blake3,
}

impl HashAlgorithm {
    /// Return the algorithm name as a static string.
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Sha256 => "sha256",
            #[cfg(feature = "blake3")]
            Self::Blake3 => "blake3",
        }
    }

    /// Get the default algorithm based on compile-time features.
    pub const fn default_algorithm() -> Self {
        #[cfg(feature = "blake3")]
        {
            Self::Blake3
        }
        #[cfg(not(feature = "blake3"))]
        {
            Self::Sha256
        }
    }
}

impl Default for HashAlgorithm {
    fn default() -> Self {
        Self::default_algorithm()
    }
}

/// A 32-byte SHA-256 hash used for content-addressed storage and deduplication.
///
/// `ContentHash` wraps a `[u8; 32]` array and provides:
/// - `Display`: lowercase hex encoding (64 characters)
/// - `Debug`: same as Display
/// - `FromStr`: parse from hex string
/// - `Hash`/`Eq`: for use as HashMap keys
///
/// # Example
///
/// ```ignore
/// let hash = ContentHash::from_bytes(b"hello world");
/// println!("{}", hash); // prints 64-char hex string
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContentHash([u8; 32]);

impl ContentHash {
    /// Compute the hash of the given bytes.
    ///
    /// Uses BLAKE3 when the `blake3` feature is enabled (faster), otherwise SHA-256.
    #[cfg(feature = "blake3")]
    pub fn from_bytes(data: &[u8]) -> Self {
        let hash = blake3::hash(data);
        Self(*hash.as_bytes())
    }

    /// Compute the SHA-256 hash of the given bytes.
    #[cfg(not(feature = "blake3"))]
    pub fn from_bytes(data: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        Self(hash)
    }

    /// Return the hash algorithm name.
    pub const fn algorithm() -> &'static str {
        #[cfg(feature = "blake3")]
        {
            "blake3"
        }
        #[cfg(not(feature = "blake3"))]
        {
            "sha256"
        }
    }

    /// Compute the hash of the given bytes using a specific algorithm.
    ///
    /// This allows runtime algorithm selection, unlike `from_bytes` which uses
    /// compile-time feature selection.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Always use SHA-256 regardless of features
    /// let sha_hash = ContentHash::from_data_with_algo(b"hello", HashAlgorithm::Sha256);
    ///
    /// // Use BLAKE3 (requires blake3 feature)
    /// #[cfg(feature = "blake3")]
    /// let blake_hash = ContentHash::from_data_with_algo(b"hello", HashAlgorithm::Blake3);
    /// ```
    pub fn from_data_with_algo(data: &[u8], algo: HashAlgorithm) -> Self {
        match algo {
            HashAlgorithm::Sha256 => {
                let mut hasher = Sha256::new();
                hasher.update(data);
                let result = hasher.finalize();
                let mut hash = [0u8; 32];
                hash.copy_from_slice(&result);
                Self(hash)
            }
            #[cfg(feature = "blake3")]
            HashAlgorithm::Blake3 => {
                let hash = blake3::hash(data);
                Self(*hash.as_bytes())
            }
        }
    }

    /// Create a ContentHash from a raw 32-byte array.
    pub const fn from_raw(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Get the raw 32-byte array.
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Get the raw 32-byte array (consuming self).
    pub const fn into_bytes(self) -> [u8; 32] {
        self.0
    }

    /// Create a zero hash (useful for testing or placeholder).
    pub const fn zero() -> Self {
        Self([0u8; 32])
    }

    /// Check if this is a zero hash.
    pub fn is_zero(&self) -> bool {
        self.0 == [0u8; 32]
    }
}

impl std::fmt::Display for ContentHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for byte in &self.0 {
            write!(f, "{:02x}", byte)?;
        }
        Ok(())
    }
}

impl std::fmt::Debug for ContentHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ContentHash({})", self)
    }
}

impl std::str::FromStr for ContentHash {
    type Err = ContentHashParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.len() != 64 {
            return Err(ContentHashParseError::InvalidLength(s.len()));
        }

        let mut bytes = [0u8; 32];
        for (i, chunk) in s.as_bytes().chunks(2).enumerate() {
            let hex_str = std::str::from_utf8(chunk)
                .map_err(|_| ContentHashParseError::InvalidHex)?;
            bytes[i] = u8::from_str_radix(hex_str, 16)
                .map_err(|_| ContentHashParseError::InvalidHex)?;
        }
        Ok(Self(bytes))
    }
}

/// Error type for parsing a `ContentHash` from a hex string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentHashParseError {
    /// The string was not exactly 64 characters.
    InvalidLength(usize),
    /// The string contained non-hexadecimal characters.
    InvalidHex,
}

impl std::fmt::Display for ContentHashParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidLength(len) => {
                write!(f, "ContentHash hex string must be 64 chars, got {}", len)
            }
            Self::InvalidHex => write!(f, "ContentHash hex string contains invalid characters"),
        }
    }
}

impl std::error::Error for ContentHashParseError {}

// ---------------------------------------------------------------------------
// FileBackend — git-style content-addressed file store
// ---------------------------------------------------------------------------

use std::fs;
use std::io::{self, Read as IoRead, Write as IoWrite};
use std::path::{Path, PathBuf};

/// A filesystem-backed content-addressed store using git-style layout.
///
/// Blobs are stored at `{base_path}/{first_two_hex}/{remaining_hex}`, e.g.:
/// - Hash `ba7816bf...` → `base_path/ba/7816bf...`
///
/// This spreads files across 256 subdirectories to avoid filesystem limits
/// on directory entries and improve lookup performance.
///
/// # Example
///
/// ```ignore
/// let store = FileBackend::new("/tmp/content_store")?;
/// let hash = store.put(b"hello world")?;
/// assert!(store.has(&hash));
/// let data = store.get(&hash)?.unwrap();
/// assert_eq!(data, b"hello world");
/// ```
pub struct FileBackend {
    /// Root directory for the content store.
    base_path: PathBuf,
}

impl FileBackend {
    /// Create a new FileBackend rooted at `base_path`.
    ///
    /// Creates the directory if it doesn't exist.
    pub fn new<P: AsRef<Path>>(base_path: P) -> io::Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        fs::create_dir_all(&base_path)?;
        Ok(Self { base_path })
    }

    /// Open an existing FileBackend. Returns an error if the path doesn't exist.
    pub fn open<P: AsRef<Path>>(base_path: P) -> io::Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        if !base_path.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("content store not found: {}", base_path.display()),
            ));
        }
        Ok(Self { base_path })
    }

    /// Get the path where a blob with the given hash would be stored.
    fn blob_path(&self, hash: &ContentHash) -> PathBuf {
        let hex = format!("{}", hash);
        let (prefix, suffix) = hex.split_at(2);
        self.base_path.join(prefix).join(suffix)
    }

    /// Store data and return its ContentHash.
    ///
    /// If the blob already exists (same hash), this is a no-op.
    pub fn put(&self, data: &[u8]) -> io::Result<ContentHash> {
        let hash = ContentHash::from_bytes(data);
        let path = self.blob_path(&hash);

        if path.exists() {
            return Ok(hash);
        }

        let parent = path.parent().unwrap();
        fs::create_dir_all(parent)?;

        let tmp_path = path.with_extension("tmp");
        {
            let mut file = fs::File::create(&tmp_path)?;
            file.write_all(data)?;
            file.sync_all()?;
        }

        fs::rename(&tmp_path, &path)?;
        Ok(hash)
    }

    /// Retrieve blob data by hash.
    ///
    /// Returns `Ok(None)` if no blob with that hash exists.
    pub fn get(&self, hash: &ContentHash) -> io::Result<Option<Vec<u8>>> {
        let path = self.blob_path(hash);
        if !path.exists() {
            return Ok(None);
        }

        let mut file = fs::File::open(&path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        Ok(Some(data))
    }

    /// Check if a blob with the given hash exists.
    pub fn has(&self, hash: &ContentHash) -> bool {
        self.blob_path(hash).exists()
    }

    /// Delete a blob by hash.
    ///
    /// Returns `Ok(true)` if the blob was deleted, `Ok(false)` if it didn't exist.
    pub fn delete(&self, hash: &ContentHash) -> io::Result<bool> {
        let path = self.blob_path(hash);
        if !path.exists() {
            return Ok(false);
        }
        fs::remove_file(&path)?;
        Ok(true)
    }

    /// Return the size of a blob in bytes, or `None` if it doesn't exist.
    pub fn size(&self, hash: &ContentHash) -> io::Result<Option<u64>> {
        let path = self.blob_path(hash);
        if !path.exists() {
            return Ok(None);
        }
        let metadata = fs::metadata(&path)?;
        Ok(Some(metadata.len()))
    }

    /// List all blob hashes in the store.
    ///
    /// Note: This can be slow for large stores.
    pub fn list(&self) -> io::Result<Vec<ContentHash>> {
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

    /// Store a tree structure (list of child hashes with names).
    ///
    /// The tree is serialized as newline-separated `"hash name"` pairs.
    pub fn tree_put(&self, entries: &[(ContentHash, String)]) -> io::Result<ContentHash> {
        let mut data = String::new();
        for (hash, name) in entries {
            data.push_str(&format!("{} {}\n", hash, name));
        }
        self.put(data.as_bytes())
    }

    /// Retrieve a tree structure by hash.
    ///
    /// Returns `Ok(None)` if the tree doesn't exist.
    pub fn tree_get(&self, hash: &ContentHash) -> io::Result<Option<Vec<(ContentHash, String)>>> {
        let data = match self.get(hash)? {
            Some(d) => d,
            None => return Ok(None),
        };

        let text = String::from_utf8(data).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("invalid UTF-8: {}", e))
        })?;

        let mut entries = Vec::new();
        for line in text.lines() {
            if line.is_empty() {
                continue;
            }
            let (hash_part, name) = line.split_once(' ').ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "malformed tree entry")
            })?;
            let hash: ContentHash = hash_part.parse().map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("invalid hash: {}", e))
            })?;
            entries.push((hash, name.to_string()));
        }

        Ok(Some(entries))
    }

    /// Get the base path of the store.
    pub fn base_path(&self) -> &Path {
        &self.base_path
    }
}

// ---------------------------------------------------------------------------
// StreamingStore — trait for streaming large content
// ---------------------------------------------------------------------------

/// Default chunk size for streaming: 256KB.
pub const DEFAULT_CHUNK_SIZE: usize = 256 * 1024;

/// Streaming store capabilities for large file handling.
///
/// The [`FileBackend`] implements streaming methods that chunk large files
/// into fixed-size blocks (default 256KB), each independently content-addressed:
///
/// - [`FileBackend::put_stream`] - Store large content from a reader
/// - [`FileBackend::put_stream_with_chunk_size`] - Store with custom chunk size
/// - [`FileBackend::get_stream`] - Retrieve as streaming reader
/// - [`FileBackend::get_manifest`] - Get chunk manifest metadata
///
/// # Benefits
///
/// - **Low memory**: Only one chunk (256KB) in memory during read/write
/// - **Deduplication**: Similar files share chunks automatically
/// - **Incremental sync**: Only changed chunks need transfer
///
/// # Example
///
/// ```ignore
/// use std::fs::File;
/// use std::io::BufReader;
///
/// let store = FileBackend::new("/tmp/store")?;
/// let mut file = BufReader::new(File::open("large_file.bin")?);
/// let manifest_hash = store.put_stream(&mut file)?;
///
/// // Later: read back with low memory
/// let mut reader = store.get_stream(&manifest_hash)?.unwrap();
/// std::io::copy(&mut reader, &mut output_file)?;
/// ```
///
/// # ContentTree Integration
///
/// Large files can be stored in a [`ContentTree`] using the [`TreeEntryType::Chunked`]
/// type, which indicates the hash points to a [`ChunkedContent`] manifest:
///
/// ```ignore
/// let manifest_hash = store.put_stream(&mut large_file)?;
/// let tree = ContentTree::from_entries(vec![
///     TreeEntry::chunked("large_asset.bin", manifest_hash),
/// ]);
/// ```

// ---------------------------------------------------------------------------
// ChunkedContent — streaming API for large content
// ---------------------------------------------------------------------------

/// Metadata for chunked content stored in the content store.
///
/// Large files are split into fixed-size chunks, each independently
/// content-addressed. The chunk list is stored as a manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkedContent {
    /// Total size of the original content in bytes.
    pub total_size: u64,
    /// Chunk size used (typically 256KB).
    pub chunk_size: usize,
    /// Ordered list of chunk hashes.
    pub chunks: Vec<ContentHash>,
}

impl ChunkedContent {
    /// Serialize the manifest to bytes.
    pub fn serialize(&self) -> Vec<u8> {
        let mut data = String::new();
        data.push_str(&format!("size:{}\n", self.total_size));
        data.push_str(&format!("chunk_size:{}\n", self.chunk_size));
        data.push_str("chunks:\n");
        for hash in &self.chunks {
            data.push_str(&format!("{}\n", hash));
        }
        data.into_bytes()
    }

    /// Deserialize a manifest from bytes.
    pub fn deserialize(data: &[u8]) -> io::Result<Self> {
        let text = String::from_utf8(data.to_vec()).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("invalid UTF-8: {}", e))
        })?;

        let mut total_size: Option<u64> = None;
        let mut chunk_size: Option<usize> = None;
        let mut chunks = Vec::new();
        let mut in_chunks = false;

        for line in text.lines() {
            if line.is_empty() {
                continue;
            }
            if line.starts_with("size:") {
                total_size = Some(line[5..].parse().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "invalid size")
                })?);
            } else if line.starts_with("chunk_size:") {
                chunk_size = Some(line[11..].parse().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "invalid chunk_size")
                })?);
            } else if line == "chunks:" {
                in_chunks = true;
            } else if in_chunks {
                let hash: ContentHash = line.parse().map_err(|e| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("invalid hash: {}", e))
                })?;
                chunks.push(hash);
            }
        }

        Ok(Self {
            total_size: total_size.ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "missing size")
            })?,
            chunk_size: chunk_size.ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "missing chunk_size")
            })?,
            chunks,
        })
    }

    /// Compute the manifest hash.
    pub fn hash(&self) -> ContentHash {
        ContentHash::from_bytes(&self.serialize())
    }
}

impl FileBackend {
    /// Store large content using chunked streaming.
    ///
    /// The content is split into fixed-size chunks (default 256KB), each
    /// independently stored. Returns the manifest hash.
    pub fn put_stream<R: io::Read>(&self, reader: &mut R) -> io::Result<ContentHash> {
        self.put_stream_with_chunk_size(reader, DEFAULT_CHUNK_SIZE)
    }

    /// Store large content with a custom chunk size.
    pub fn put_stream_with_chunk_size<R: io::Read>(
        &self,
        reader: &mut R,
        chunk_size: usize,
    ) -> io::Result<ContentHash> {
        let mut chunks = Vec::new();
        let mut total_size: u64 = 0;
        let mut buf = vec![0u8; chunk_size];

        loop {
            let mut filled = 0;
            while filled < chunk_size {
                match reader.read(&mut buf[filled..]) {
                    Ok(0) => break, // EOF
                    Ok(n) => filled += n,
                    Err(ref e) if e.kind() == io::ErrorKind::Interrupted => continue,
                    Err(e) => return Err(e),
                }
            }

            if filled == 0 {
                break;
            }

            let chunk_hash = self.put(&buf[..filled])?;
            chunks.push(chunk_hash);
            total_size += filled as u64;
        }

        let manifest = ChunkedContent {
            total_size,
            chunk_size,
            chunks,
        };

        self.put(&manifest.serialize())
    }

    /// Retrieve chunked content as a streaming reader.
    ///
    /// Returns `None` if the manifest doesn't exist.
    pub fn get_stream(&self, manifest_hash: &ContentHash) -> io::Result<Option<ChunkedReader<'_>>> {
        let manifest_data = match self.get(manifest_hash)? {
            Some(d) => d,
            None => return Ok(None),
        };

        let manifest = ChunkedContent::deserialize(&manifest_data)?;
        Ok(Some(ChunkedReader::new(self, manifest)))
    }

    /// Get the manifest for chunked content.
    pub fn get_manifest(&self, manifest_hash: &ContentHash) -> io::Result<Option<ChunkedContent>> {
        let data = match self.get(manifest_hash)? {
            Some(d) => d,
            None => return Ok(None),
        };
        Ok(Some(ChunkedContent::deserialize(&data)?))
    }
}

/// A streaming reader for chunked content.
///
/// Reads chunks on-demand, keeping memory usage low even for large files.
pub struct ChunkedReader<'a> {
    backend: &'a FileBackend,
    manifest: ChunkedContent,
    chunk_index: usize,
    chunk_buffer: Vec<u8>,
    buffer_offset: usize,
    bytes_read: u64,
}

impl<'a> ChunkedReader<'a> {
    fn new(backend: &'a FileBackend, manifest: ChunkedContent) -> Self {
        Self {
            backend,
            manifest,
            chunk_index: 0,
            chunk_buffer: Vec::new(),
            buffer_offset: 0,
            bytes_read: 0,
        }
    }

    /// Total size of the content.
    pub fn total_size(&self) -> u64 {
        self.manifest.total_size
    }

    /// Bytes read so far.
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read
    }

    /// Number of chunks.
    pub fn chunk_count(&self) -> usize {
        self.manifest.chunks.len()
    }

    fn load_next_chunk(&mut self) -> io::Result<bool> {
        if self.chunk_index >= self.manifest.chunks.len() {
            return Ok(false);
        }

        let hash = &self.manifest.chunks[self.chunk_index];
        self.chunk_buffer = self.backend.get(hash)?.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("missing chunk {}: {}", self.chunk_index, hash),
            )
        })?;
        self.buffer_offset = 0;
        self.chunk_index += 1;
        Ok(true)
    }
}

impl<'a> io::Read for ChunkedReader<'a> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }

        // Load next chunk if current is exhausted
        while self.buffer_offset >= self.chunk_buffer.len() {
            if !self.load_next_chunk()? {
                return Ok(0); // EOF
            }
        }

        let available = self.chunk_buffer.len() - self.buffer_offset;
        let to_copy = buf.len().min(available);
        buf[..to_copy]
            .copy_from_slice(&self.chunk_buffer[self.buffer_offset..self.buffer_offset + to_copy]);
        self.buffer_offset += to_copy;
        self.bytes_read += to_copy as u64;
        Ok(to_copy)
    }
}

// ---------------------------------------------------------------------------
// ContentStoreGC — garbage collection for content-addressed storage
// ---------------------------------------------------------------------------

use std::collections::HashSet;
use std::time::{Duration, Instant};

/// Configuration for content store garbage collection.
#[derive(Debug, Clone)]
pub struct GCConfig {
    /// Maximum time budget per GC run.
    pub time_budget: Duration,
    /// Whether to actually delete orphans (false = dry run).
    pub delete_orphans: bool,
}

impl Default for GCConfig {
    fn default() -> Self {
        Self {
            time_budget: Duration::from_millis(2),
            delete_orphans: true,
        }
    }
}

/// Result of a GC run.
#[derive(Debug, Clone)]
pub struct GCResult {
    /// Number of hashes marked as reachable.
    pub marked_count: usize,
    /// Number of orphan blobs found.
    pub orphan_count: usize,
    /// Number of orphan blobs deleted.
    pub deleted_count: usize,
    /// Whether GC completed within time budget.
    pub completed: bool,
    /// Time spent on this GC run.
    pub elapsed: Duration,
}

/// Mark-and-sweep garbage collector for content stores.
///
/// Usage:
/// 1. Create a `ContentStoreGC` with roots and backend
/// 2. Call `mark_from_roots()` to mark reachable content
/// 3. Call `sweep()` to delete unreachable content
pub struct ContentStoreGC<'a> {
    backend: &'a FileBackend,
    roots: Vec<ContentHash>,
    marked: HashSet<ContentHash>,
    config: GCConfig,
}

impl<'a> ContentStoreGC<'a> {
    /// Create a new GC with the given roots and backend.
    pub fn new(backend: &'a FileBackend, roots: Vec<ContentHash>) -> Self {
        Self {
            backend,
            roots,
            marked: HashSet::new(),
            config: GCConfig::default(),
        }
    }

    /// Create a new GC with custom configuration.
    pub fn with_config(backend: &'a FileBackend, roots: Vec<ContentHash>, config: GCConfig) -> Self {
        Self {
            backend,
            roots,
            marked: HashSet::new(),
            config,
        }
    }

    /// Mark phase: BFS from root set to find all reachable content.
    ///
    /// This traverses trees and marks all referenced blobs and subtrees.
    pub fn mark_from_roots(&mut self) -> io::Result<usize> {
        let start = Instant::now();
        let mut queue: Vec<ContentHash> = self.roots.clone();

        while let Some(hash) = queue.pop() {
            if start.elapsed() > self.config.time_budget {
                break;
            }

            if self.marked.contains(&hash) {
                continue;
            }
            self.marked.insert(hash);

            // Try to load as tree and mark children
            if let Some(data) = self.backend.get(&hash)? {
                // Try to parse as ContentTree
                if let Ok(tree) = ContentTree::deserialize(&data) {
                    for entry in tree.entries() {
                        if !self.marked.contains(&entry.hash) {
                            queue.push(entry.hash);
                        }
                    }
                }
                // Try to parse as ChunkedContent manifest
                if let Ok(manifest) = ChunkedContent::deserialize(&data) {
                    for chunk_hash in &manifest.chunks {
                        if !self.marked.contains(chunk_hash) {
                            queue.push(*chunk_hash);
                        }
                    }
                }
            }
        }

        Ok(self.marked.len())
    }

    /// Sweep phase: delete all blobs not in the marked set.
    ///
    /// Returns the number of blobs deleted.
    pub fn sweep(&self) -> io::Result<GCResult> {
        let start = Instant::now();
        let mut orphan_count = 0;
        let mut deleted_count = 0;

        let all_hashes = self.backend.list()?;

        for hash in all_hashes {
            if start.elapsed() > self.config.time_budget {
                return Ok(GCResult {
                    marked_count: self.marked.len(),
                    orphan_count,
                    deleted_count,
                    completed: false,
                    elapsed: start.elapsed(),
                });
            }

            if !self.marked.contains(&hash) {
                orphan_count += 1;
                if self.config.delete_orphans {
                    if self.backend.delete(&hash)? {
                        deleted_count += 1;
                    }
                }
            }
        }

        Ok(GCResult {
            marked_count: self.marked.len(),
            orphan_count,
            deleted_count,
            completed: true,
            elapsed: start.elapsed(),
        })
    }

    /// Run full GC (mark + sweep) in one call.
    pub fn run(&mut self) -> io::Result<GCResult> {
        self.mark_from_roots()?;
        self.sweep()
    }

    /// Get the set of marked hashes.
    pub fn marked(&self) -> &HashSet<ContentHash> {
        &self.marked
    }
}

// ---------------------------------------------------------------------------
// ContentTree — structural sharing via content-addressing
// ---------------------------------------------------------------------------

/// Entry type in a ContentTree (blob or subtree).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TreeEntryType {
    /// A blob (file content).
    Blob,
    /// A subtree (directory).
    Tree,
    /// A material definition.
    Material,
    /// A shader source file.
    Shader,
    /// A chunked large file (hash points to ChunkedContent manifest).
    Chunked,
}

/// An entry in a ContentTree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeEntry {
    /// Name of this entry.
    pub name: String,
    /// Content hash of the entry (blob data or serialized subtree).
    pub hash: ContentHash,
    /// Whether this is a blob or a subtree.
    pub entry_type: TreeEntryType,
}

impl TreeEntry {
    /// Create a blob entry.
    pub fn blob(name: impl Into<String>, hash: ContentHash) -> Self {
        Self {
            name: name.into(),
            hash,
            entry_type: TreeEntryType::Blob,
        }
    }

    /// Create a tree entry.
    pub fn tree(name: impl Into<String>, hash: ContentHash) -> Self {
        Self {
            name: name.into(),
            hash,
            entry_type: TreeEntryType::Tree,
        }
    }

    /// Create a material entry.
    pub fn material(name: impl Into<String>, hash: ContentHash) -> Self {
        Self {
            name: name.into(),
            hash,
            entry_type: TreeEntryType::Material,
        }
    }

    /// Create a shader entry.
    pub fn shader(name: impl Into<String>, hash: ContentHash) -> Self {
        Self {
            name: name.into(),
            hash,
            entry_type: TreeEntryType::Shader,
        }
    }

    /// Create a chunked file entry (for large files stored via streaming API).
    ///
    /// The hash points to a ChunkedContent manifest, not the raw file data.
    pub fn chunked(name: impl Into<String>, manifest_hash: ContentHash) -> Self {
        Self {
            name: name.into(),
            hash: manifest_hash,
            entry_type: TreeEntryType::Chunked,
        }
    }
}

/// A content-addressed tree structure with structural sharing.
///
/// Trees are immutable; modifications create new trees that share
/// unchanged subtrees with the original via content-addressing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContentTree {
    /// Sorted list of entries in this tree.
    entries: Vec<TreeEntry>,
    /// Cached hash of this tree (computed lazily).
    hash: Option<ContentHash>,
}

impl ContentTree {
    /// Create an empty tree.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            hash: None,
        }
    }

    /// Create a tree from entries.
    pub fn from_entries(mut entries: Vec<TreeEntry>) -> Self {
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        Self {
            entries,
            hash: None,
        }
    }

    /// Get the entries in this tree.
    pub fn entries(&self) -> &[TreeEntry] {
        &self.entries
    }

    /// Look up an entry by name.
    pub fn get(&self, name: &str) -> Option<&TreeEntry> {
        self.entries.iter().find(|e| e.name == name)
    }

    /// Insert or replace an entry, returning a new tree (structural sharing).
    pub fn with_entry(&self, entry: TreeEntry) -> Self {
        let mut entries = self.entries.clone();
        if let Some(pos) = entries.iter().position(|e| e.name == entry.name) {
            entries[pos] = entry;
        } else {
            entries.push(entry);
            entries.sort_by(|a, b| a.name.cmp(&b.name));
        }
        Self {
            entries,
            hash: None,
        }
    }

    /// Remove an entry by name, returning a new tree.
    pub fn without_entry(&self, name: &str) -> Self {
        let entries: Vec<_> = self.entries.iter()
            .filter(|e| e.name != name)
            .cloned()
            .collect();
        Self {
            entries,
            hash: None,
        }
    }

    /// Serialize this tree to bytes.
    pub fn serialize(&self) -> Vec<u8> {
        let mut data = String::new();
        for entry in &self.entries {
            let type_char = match entry.entry_type {
                TreeEntryType::Blob => 'b',
                TreeEntryType::Tree => 't',
                TreeEntryType::Material => 'm',
                TreeEntryType::Shader => 's',
                TreeEntryType::Chunked => 'c',
            };
            data.push_str(&format!("{} {} {}\n", type_char, entry.hash, entry.name));
        }
        data.into_bytes()
    }

    /// Deserialize a tree from bytes.
    pub fn deserialize(data: &[u8]) -> Result<Self, io::Error> {
        let text = String::from_utf8(data.to_vec()).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("invalid UTF-8: {}", e))
        })?;

        let mut entries = Vec::new();
        for line in text.lines() {
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.splitn(3, ' ').collect();
            if parts.len() != 3 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "malformed tree entry",
                ));
            }
            let entry_type = match parts[0] {
                "b" => TreeEntryType::Blob,
                "t" => TreeEntryType::Tree,
                "m" => TreeEntryType::Material,
                "s" => TreeEntryType::Shader,
                "c" => TreeEntryType::Chunked,
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unknown entry type: {}", parts[0]),
                    ))
                }
            };
            let hash: ContentHash = parts[1].parse().map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("invalid hash: {}", e))
            })?;
            entries.push(TreeEntry {
                name: parts[2].to_string(),
                hash,
                entry_type,
            });
        }

        Ok(Self::from_entries(entries))
    }

    /// Compute the content hash of this tree.
    pub fn compute_hash(&self) -> ContentHash {
        ContentHash::from_bytes(&self.serialize())
    }

    /// Get the cached hash or compute it.
    pub fn hash(&mut self) -> ContentHash {
        if let Some(h) = self.hash {
            return h;
        }
        let h = self.compute_hash();
        self.hash = Some(h);
        h
    }

    /// Store this tree in a FileBackend.
    pub fn store(&self, backend: &FileBackend) -> io::Result<ContentHash> {
        backend.put(&self.serialize())
    }

    /// Load a tree from a FileBackend.
    pub fn load(backend: &FileBackend, hash: &ContentHash) -> io::Result<Option<Self>> {
        match backend.get(hash)? {
            Some(data) => Ok(Some(Self::deserialize(&data)?)),
            None => Ok(None),
        }
    }

    /// Compute the diff between two trees.
    ///
    /// Returns a list of changes: (name, old_entry, new_entry).
    /// - `(name, Some(old), None)` = deleted
    /// - `(name, None, Some(new))` = added
    /// - `(name, Some(old), Some(new))` = modified (hashes differ)
    pub fn diff(&self, other: &ContentTree) -> Vec<TreeDiffEntry> {
        let mut diffs = Vec::new();

        // Build maps for O(1) lookup
        let self_map: std::collections::HashMap<&str, &TreeEntry> =
            self.entries.iter().map(|e| (e.name.as_str(), e)).collect();
        let other_map: std::collections::HashMap<&str, &TreeEntry> =
            other.entries.iter().map(|e| (e.name.as_str(), e)).collect();

        // Find deleted and modified entries
        for entry in &self.entries {
            match other_map.get(entry.name.as_str()) {
                None => diffs.push(TreeDiffEntry::Deleted(entry.clone())),
                Some(other_entry) => {
                    if entry.hash != other_entry.hash || entry.entry_type != other_entry.entry_type
                    {
                        diffs.push(TreeDiffEntry::Modified {
                            old: entry.clone(),
                            new: (*other_entry).clone(),
                        });
                    }
                }
            }
        }

        // Find added entries
        for entry in &other.entries {
            if !self_map.contains_key(entry.name.as_str()) {
                diffs.push(TreeDiffEntry::Added(entry.clone()));
            }
        }

        // Sort by name for deterministic output
        diffs.sort_by(|a, b| a.name().cmp(b.name()));
        diffs
    }

    /// Check if this tree is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return the number of entries in this tree.
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

impl Default for ContentTree {
    fn default() -> Self {
        Self::new()
    }
}

/// A single entry in a tree diff.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TreeDiffEntry {
    /// Entry was added.
    Added(TreeEntry),
    /// Entry was deleted.
    Deleted(TreeEntry),
    /// Entry was modified.
    Modified { old: TreeEntry, new: TreeEntry },
}

impl TreeDiffEntry {
    /// Get the name of the entry that changed.
    pub fn name(&self) -> &str {
        match self {
            Self::Added(e) => &e.name,
            Self::Deleted(e) => &e.name,
            Self::Modified { new, .. } => &new.name,
        }
    }
}

// ---------------------------------------------------------------------------
// ContentDiffer — diff/apply framework for content changes
// ---------------------------------------------------------------------------

/// Error type for diff/apply operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffError {
    /// The patch data is invalid or corrupted.
    InvalidPatch,
    /// Size mismatch during apply (e.g., old content doesn't match expected).
    SizeMismatch { expected: usize, actual: usize },
    /// I/O error during diff/apply.
    IoError(String),
    /// Delta type mismatch (e.g., applying BinaryPatch to tree).
    TypeMismatch { expected: &'static str, actual: &'static str },
}

impl std::fmt::Display for DiffError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPatch => write!(f, "invalid or corrupted patch data"),
            Self::SizeMismatch { expected, actual } => {
                write!(f, "size mismatch: expected {}, got {}", expected, actual)
            }
            Self::IoError(msg) => write!(f, "I/O error: {}", msg),
            Self::TypeMismatch { expected, actual } => {
                write!(f, "delta type mismatch: expected {}, got {}", expected, actual)
            }
        }
    }
}

impl std::error::Error for DiffError {}

/// A delta representing the difference between old and new content.
///
/// Different delta variants are optimized for different content types:
/// - `Full`: Complete replacement when no efficient diff is possible
/// - `BinaryPatch`: Compact patch for similar binary blobs
/// - `TreeDiff`: Structural changes in a ContentTree
/// - `ParameterPatch`: Sparse updates for uniform/shader parameters
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Delta {
    /// Full content replacement (no diff possible or beneficial).
    Full(Vec<u8>),
    /// Binary patch for similar binary blobs.
    /// Uses copy/insert instructions for compact representation.
    BinaryPatch {
        /// Serialized patch instructions.
        patch: Vec<u8>,
    },
    /// Tree diff for ContentTree changes.
    TreeDiff {
        /// List of tree entry changes.
        changes: Vec<TreeDiffEntry>,
    },
    /// Parameter patch for sparse updates (e.g., shader uniforms).
    /// Each tuple is (offset, new_bytes).
    ParameterPatch {
        /// Offset-value pairs for sparse updates.
        offsets: Vec<(usize, Vec<u8>)>,
    },
}

impl Delta {
    /// Returns the size of this delta in bytes (for metrics/heuristics).
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::Full(data) => data.len(),
            Self::BinaryPatch { patch } => patch.len(),
            Self::TreeDiff { changes } => {
                // Rough estimate: each change ~100 bytes
                changes.len() * 100
            }
            Self::ParameterPatch { offsets } => {
                offsets.iter().map(|(_, v)| 8 + v.len()).sum()
            }
        }
    }

    /// Returns true if this is an empty delta (no changes).
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Full(data) => data.is_empty(),
            Self::BinaryPatch { patch } => patch.is_empty(),
            Self::TreeDiff { changes } => changes.is_empty(),
            Self::ParameterPatch { offsets } => offsets.is_empty(),
        }
    }
}

/// Trait for computing and applying content diffs.
///
/// Implementations should be optimized for specific content types
/// (binary blobs, trees, parameter buffers, etc.).
pub trait ContentDiffer {
    /// Compute the difference between old and new content.
    ///
    /// Returns a `Delta` that, when applied to `old`, produces `new`.
    fn diff(&self, old: &[u8], new: &[u8]) -> Result<Delta, DiffError>;

    /// Apply a delta to old content to produce new content.
    ///
    /// The delta must have been produced by `diff()` on compatible content.
    fn apply(&self, old: &[u8], delta: &Delta) -> Result<Vec<u8>, DiffError>;
}

// ---------------------------------------------------------------------------
// BinaryDiffer — simple copy/insert patch format
// ---------------------------------------------------------------------------

/// Binary patch instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
enum PatchOp {
    /// Copy `len` bytes from old content at `offset`.
    Copy { offset: u32, len: u32 },
    /// Insert literal bytes.
    Insert { data: Vec<u8> },
}

/// A binary differ using a simple copy/insert patch format.
///
/// This implementation uses a rolling hash to find matching regions
/// between old and new content, then encodes the difference as a
/// sequence of copy (from old) and insert (literal) operations.
///
/// # Patch Format
///
/// The serialized patch is a sequence of instructions:
/// - `0x01 <offset:u32> <len:u32>` — copy from old
/// - `0x02 <len:u32> <data:bytes>` — insert literal
/// - `0x00` — end marker
///
/// All integers are little-endian.
#[derive(Debug, Clone, Default)]
pub struct BinaryDiffer {
    /// Minimum match length to emit a copy instruction (default: 8).
    pub min_match_len: usize,
    /// Block size for rolling hash (default: 16).
    pub block_size: usize,
}

impl BinaryDiffer {
    /// Create a new BinaryDiffer with default settings.
    pub fn new() -> Self {
        Self {
            min_match_len: 8,
            block_size: 16,
        }
    }

    /// Create a BinaryDiffer with custom settings.
    pub fn with_settings(min_match_len: usize, block_size: usize) -> Self {
        Self {
            min_match_len: min_match_len.max(4),
            block_size: block_size.max(4),
        }
    }

    /// Compute a simple rolling hash for a block.
    fn block_hash(data: &[u8]) -> u32 {
        let mut h: u32 = 0;
        for &b in data {
            h = h.wrapping_mul(31).wrapping_add(b as u32);
        }
        h
    }

    /// Build a hash table of block positions in the old content.
    fn build_hash_table(&self, old: &[u8]) -> HashMap<u32, Vec<usize>> {
        let mut table: HashMap<u32, Vec<usize>> = HashMap::new();
        if old.len() < self.block_size {
            return table;
        }
        for i in 0..=(old.len() - self.block_size) {
            let hash = Self::block_hash(&old[i..i + self.block_size]);
            table.entry(hash).or_default().push(i);
        }
        table
    }

    /// Find the longest match starting at `new_pos` in new content.
    fn find_match(
        &self,
        old: &[u8],
        new: &[u8],
        new_pos: usize,
        hash_table: &HashMap<u32, Vec<usize>>,
    ) -> Option<(usize, usize)> {
        if new_pos + self.block_size > new.len() {
            return None;
        }

        let hash = Self::block_hash(&new[new_pos..new_pos + self.block_size]);
        let candidates = hash_table.get(&hash)?;

        let mut best_offset = 0;
        let mut best_len = 0;

        for &old_pos in candidates {
            // Verify the block actually matches
            if old[old_pos..old_pos + self.block_size] != new[new_pos..new_pos + self.block_size] {
                continue;
            }

            // Extend the match forward
            let mut len = self.block_size;
            while old_pos + len < old.len()
                && new_pos + len < new.len()
                && old[old_pos + len] == new[new_pos + len]
            {
                len += 1;
            }

            if len > best_len {
                best_len = len;
                best_offset = old_pos;
            }
        }

        if best_len >= self.min_match_len {
            Some((best_offset, best_len))
        } else {
            None
        }
    }

    /// Serialize patch operations to bytes.
    fn serialize_ops(ops: &[PatchOp]) -> Vec<u8> {
        let mut out = Vec::new();
        for op in ops {
            match op {
                PatchOp::Copy { offset, len } => {
                    out.push(0x01);
                    out.extend_from_slice(&offset.to_le_bytes());
                    out.extend_from_slice(&len.to_le_bytes());
                }
                PatchOp::Insert { data } => {
                    out.push(0x02);
                    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
                    out.extend_from_slice(data);
                }
            }
        }
        out.push(0x00); // End marker
        out
    }

    /// Deserialize patch operations from bytes.
    fn deserialize_ops(data: &[u8]) -> Result<Vec<PatchOp>, DiffError> {
        let mut ops = Vec::new();
        let mut pos = 0;

        while pos < data.len() {
            let op_type = data[pos];
            pos += 1;

            match op_type {
                0x00 => break, // End marker
                0x01 => {
                    // Copy
                    if pos + 8 > data.len() {
                        return Err(DiffError::InvalidPatch);
                    }
                    let offset = u32::from_le_bytes([
                        data[pos],
                        data[pos + 1],
                        data[pos + 2],
                        data[pos + 3],
                    ]);
                    let len = u32::from_le_bytes([
                        data[pos + 4],
                        data[pos + 5],
                        data[pos + 6],
                        data[pos + 7],
                    ]);
                    pos += 8;
                    ops.push(PatchOp::Copy { offset, len });
                }
                0x02 => {
                    // Insert
                    if pos + 4 > data.len() {
                        return Err(DiffError::InvalidPatch);
                    }
                    let len = u32::from_le_bytes([
                        data[pos],
                        data[pos + 1],
                        data[pos + 2],
                        data[pos + 3],
                    ]) as usize;
                    pos += 4;
                    if pos + len > data.len() {
                        return Err(DiffError::InvalidPatch);
                    }
                    ops.push(PatchOp::Insert {
                        data: data[pos..pos + len].to_vec(),
                    });
                    pos += len;
                }
                _ => return Err(DiffError::InvalidPatch),
            }
        }

        Ok(ops)
    }
}

impl ContentDiffer for BinaryDiffer {
    fn diff(&self, old: &[u8], new: &[u8]) -> Result<Delta, DiffError> {
        // Special case: identical content - emit single Copy of everything
        if old == new {
            if old.is_empty() {
                // Both empty: no operations needed
                return Ok(Delta::BinaryPatch { patch: vec![0x00] });
            }
            // Copy all of old to produce new
            let ops = vec![PatchOp::Copy {
                offset: 0,
                len: old.len() as u32,
            }];
            return Ok(Delta::BinaryPatch {
                patch: Self::serialize_ops(&ops),
            });
        }

        // Special case: empty old content
        if old.is_empty() {
            let ops = vec![PatchOp::Insert { data: new.to_vec() }];
            return Ok(Delta::BinaryPatch {
                patch: Self::serialize_ops(&ops),
            });
        }

        // Build hash table for old content
        let hash_table = self.build_hash_table(old);
        let mut ops = Vec::new();
        let mut new_pos = 0;
        let mut pending_insert: Vec<u8> = Vec::new();

        while new_pos < new.len() {
            if let Some((old_offset, match_len)) = self.find_match(old, new, new_pos, &hash_table) {
                // Flush pending insert
                if !pending_insert.is_empty() {
                    ops.push(PatchOp::Insert {
                        data: std::mem::take(&mut pending_insert),
                    });
                }
                ops.push(PatchOp::Copy {
                    offset: old_offset as u32,
                    len: match_len as u32,
                });
                new_pos += match_len;
            } else {
                pending_insert.push(new[new_pos]);
                new_pos += 1;
            }
        }

        // Flush final pending insert
        if !pending_insert.is_empty() {
            ops.push(PatchOp::Insert { data: pending_insert });
        }

        let patch = Self::serialize_ops(&ops);

        // If patch is larger than full replacement, use Full delta
        if patch.len() >= new.len() {
            return Ok(Delta::Full(new.to_vec()));
        }

        Ok(Delta::BinaryPatch { patch })
    }

    fn apply(&self, old: &[u8], delta: &Delta) -> Result<Vec<u8>, DiffError> {
        match delta {
            Delta::Full(data) => Ok(data.clone()),
            Delta::BinaryPatch { patch } => {
                let ops = Self::deserialize_ops(patch)?;
                let mut result = Vec::new();

                for op in ops {
                    match op {
                        PatchOp::Copy { offset, len } => {
                            let start = offset as usize;
                            let end = start + len as usize;
                            if end > old.len() {
                                return Err(DiffError::SizeMismatch {
                                    expected: end,
                                    actual: old.len(),
                                });
                            }
                            result.extend_from_slice(&old[start..end]);
                        }
                        PatchOp::Insert { data } => {
                            result.extend_from_slice(&data);
                        }
                    }
                }

                Ok(result)
            }
            _ => Err(DiffError::TypeMismatch {
                expected: "BinaryPatch or Full",
                actual: match delta {
                    Delta::TreeDiff { .. } => "TreeDiff",
                    Delta::ParameterPatch { .. } => "ParameterPatch",
                    _ => "unknown",
                },
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// TreeDiffer — wrapper around ContentTree::diff()
// ---------------------------------------------------------------------------

/// A differ for ContentTree structures.
///
/// This wraps `ContentTree::diff()` and produces `Delta::TreeDiff` results.
/// The input bytes are expected to be serialized ContentTree data.
#[derive(Debug, Clone, Default)]
pub struct TreeDiffer;

impl TreeDiffer {
    /// Create a new TreeDiffer.
    pub fn new() -> Self {
        Self
    }

    /// Diff two ContentTree instances directly (without serialization).
    pub fn diff_trees(&self, old: &ContentTree, new: &ContentTree) -> Delta {
        let changes = old.diff(new);
        Delta::TreeDiff { changes }
    }

    /// Apply a TreeDiff to produce a new ContentTree.
    pub fn apply_to_tree(&self, old: &ContentTree, delta: &Delta) -> Result<ContentTree, DiffError> {
        match delta {
            Delta::TreeDiff { changes } => {
                let mut result = old.clone();
                for change in changes {
                    match change {
                        TreeDiffEntry::Added(entry) => {
                            result = result.with_entry(entry.clone());
                        }
                        TreeDiffEntry::Deleted(entry) => {
                            result = result.without_entry(&entry.name);
                        }
                        TreeDiffEntry::Modified { new, .. } => {
                            result = result.with_entry(new.clone());
                        }
                    }
                }
                Ok(result)
            }
            _ => Err(DiffError::TypeMismatch {
                expected: "TreeDiff",
                actual: match delta {
                    Delta::Full(_) => "Full",
                    Delta::BinaryPatch { .. } => "BinaryPatch",
                    Delta::ParameterPatch { .. } => "ParameterPatch",
                    _ => "unknown",
                },
            }),
        }
    }
}

impl ContentDiffer for TreeDiffer {
    fn diff(&self, old: &[u8], new: &[u8]) -> Result<Delta, DiffError> {
        let old_tree = ContentTree::deserialize(old)
            .map_err(|e| DiffError::IoError(e.to_string()))?;
        let new_tree = ContentTree::deserialize(new)
            .map_err(|e| DiffError::IoError(e.to_string()))?;

        Ok(self.diff_trees(&old_tree, &new_tree))
    }

    fn apply(&self, old: &[u8], delta: &Delta) -> Result<Vec<u8>, DiffError> {
        let old_tree = ContentTree::deserialize(old)
            .map_err(|e| DiffError::IoError(e.to_string()))?;
        let new_tree = self.apply_to_tree(&old_tree, delta)?;
        Ok(new_tree.serialize())
    }
}

// ---------------------------------------------------------------------------
// ProvenanceChain — version history with automatic pruning
// ---------------------------------------------------------------------------

/// A single entry in a provenance chain, representing one version of content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvenanceEntry {
    /// Content hash at this version.
    pub hash: ContentHash,
    /// When this version was created (Unix timestamp in seconds).
    pub timestamp: u64,
    /// Optional description/commit message.
    pub message: Option<String>,
    /// Previous entry hash (None for origin).
    pub parent: Option<ContentHash>,
}

impl ProvenanceEntry {
    /// Create a new origin entry (no parent).
    pub fn origin(hash: ContentHash, timestamp: u64, message: Option<String>) -> Self {
        Self {
            hash,
            timestamp,
            message,
            parent: None,
        }
    }

    /// Create a new entry with a parent.
    pub fn with_parent(
        hash: ContentHash,
        timestamp: u64,
        message: Option<String>,
        parent: ContentHash,
    ) -> Self {
        Self {
            hash,
            timestamp,
            message,
            parent: Some(parent),
        }
    }
}

/// Strategy for pruning provenance chains.
#[derive(Debug, Clone)]
pub enum PruningStrategy {
    /// Keep last N entries (default 10).
    KeepLastN(usize),
    /// Keep entries newer than max_age seconds.
    MaxAge(u64),
    /// Combined: keep last N AND anything newer than max_age.
    Combined {
        keep_last_n: usize,
        max_age_secs: u64,
    },
}

impl Default for PruningStrategy {
    fn default() -> Self {
        PruningStrategy::KeepLastN(10)
    }
}

/// A chain of provenance entries with automatic pruning.
///
/// The chain maintains version history for content, automatically pruning
/// old entries according to the configured strategy while always preserving:
/// - The origin (first) entry
/// - The current (last) entry
///
/// # Example
///
/// ```ignore
/// let origin = ProvenanceEntry::origin(hash1, 1000, Some("Initial".into()));
/// let mut chain = ProvenanceChain::with_origin(origin, PruningStrategy::KeepLastN(3));
///
/// chain.push(ProvenanceEntry::with_parent(hash2, 2000, None, hash1));
/// chain.push(ProvenanceEntry::with_parent(hash3, 3000, None, hash2));
/// // After exceeding the limit, middle entries are pruned
/// ```
pub struct ProvenanceChain {
    /// All entries in order (oldest first, newest last).
    entries: Vec<ProvenanceEntry>,
    /// Pruning strategy.
    strategy: PruningStrategy,
}

impl ProvenanceChain {
    /// Create a new empty provenance chain with the given strategy.
    pub fn new(strategy: PruningStrategy) -> Self {
        Self {
            entries: Vec::new(),
            strategy,
        }
    }

    /// Create a provenance chain initialized with an origin entry.
    pub fn with_origin(origin: ProvenanceEntry, strategy: PruningStrategy) -> Self {
        Self {
            entries: vec![origin],
            strategy,
        }
    }

    /// Add a new entry to the chain, automatically pruning if needed.
    ///
    /// Always keeps the first (origin) and last (current) entries.
    pub fn push(&mut self, entry: ProvenanceEntry) {
        self.entries.push(entry);
        self.prune();
    }

    /// Get the origin entry (first).
    pub fn origin(&self) -> Option<&ProvenanceEntry> {
        self.entries.first()
    }

    /// Get the current entry (last).
    pub fn current(&self) -> Option<&ProvenanceEntry> {
        self.entries.last()
    }

    /// Get all entries.
    pub fn entries(&self) -> &[ProvenanceEntry] {
        &self.entries
    }

    /// Get entry count.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the current pruning strategy.
    pub fn strategy(&self) -> &PruningStrategy {
        &self.strategy
    }

    /// Manually trigger pruning according to the configured strategy.
    ///
    /// Pruning rules:
    /// - Origin (first entry) is ALWAYS preserved
    /// - Current (last entry) is ALWAYS preserved
    /// - Middle entries are pruned according to the strategy
    fn prune(&mut self) {
        if self.entries.len() <= 2 {
            // Nothing to prune: origin and current must be preserved
            return;
        }

        match &self.strategy {
            PruningStrategy::KeepLastN(n) => self.prune_keep_last_n(*n),
            PruningStrategy::MaxAge(max_age) => self.prune_max_age(*max_age),
            PruningStrategy::Combined {
                keep_last_n,
                max_age_secs,
            } => self.prune_combined(*keep_last_n, *max_age_secs),
        }
    }

    /// Prune to keep at most N entries (including origin).
    ///
    /// Strategy: Keep first + last (N-1) entries.
    fn prune_keep_last_n(&mut self, n: usize) {
        let n = n.max(2); // Must keep at least origin and current

        if self.entries.len() <= n {
            return;
        }

        // Keep: first entry (origin) + last (n-1) entries
        let remove_count = self.entries.len() - n;

        // Remove entries from index 1 to (1 + remove_count)
        // This keeps index 0 (origin) and the last (n-1) entries
        self.entries.drain(1..1 + remove_count);
    }

    /// Prune entries older than max_age (except origin).
    fn prune_max_age(&mut self, max_age: u64) {
        if self.entries.len() <= 2 {
            return;
        }

        // Get current time from the most recent entry
        let now = self.entries.last().map(|e| e.timestamp).unwrap_or(0);
        let cutoff = now.saturating_sub(max_age);

        // Keep origin (index 0), keep current (last), filter middle
        let origin = self.entries.remove(0);
        let current = self.entries.pop();

        // Filter middle entries by age
        self.entries.retain(|e| e.timestamp >= cutoff);

        // Re-insert origin at front
        self.entries.insert(0, origin);

        // Re-add current at end
        if let Some(c) = current {
            self.entries.push(c);
        }
    }

    /// Combined pruning: keep entries that are either in last N OR newer than max_age.
    fn prune_combined(&mut self, keep_last_n: usize, max_age_secs: u64) {
        if self.entries.len() <= 2 {
            return;
        }

        let n = keep_last_n.max(2);
        let now = self.entries.last().map(|e| e.timestamp).unwrap_or(0);
        let cutoff = now.saturating_sub(max_age_secs);

        // Keep origin and current
        let origin = self.entries.remove(0);
        let current = self.entries.pop();

        // For middle entries, keep if:
        // 1. Within last (n-2) middle entries, OR
        // 2. Newer than cutoff time
        let middle_count = self.entries.len();
        let keep_last_middle = (n - 2).min(middle_count);

        let mut kept = Vec::new();

        for (i, entry) in self.entries.drain(..).enumerate() {
            let is_recent = entry.timestamp >= cutoff;
            let is_in_last_n = i >= middle_count.saturating_sub(keep_last_middle);
            if is_recent || is_in_last_n {
                kept.push(entry);
            }
        }

        self.entries = kept;
        self.entries.insert(0, origin);
        if let Some(c) = current {
            self.entries.push(c);
        }
    }
}

// ---------------------------------------------------------------------------
// DeltaSyncProtocol — client-server sync with checkpoint-based incremental transfer
// ---------------------------------------------------------------------------

/// A checkpoint representing sync state at a point in time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncCheckpoint {
    /// Unique checkpoint ID (content hash of state).
    pub id: ContentHash,
    /// Timestamp when checkpoint was created (Unix epoch seconds).
    pub timestamp: u64,
    /// Number of items in the checkpoint.
    pub item_count: usize,
}

impl SyncCheckpoint {
    /// Create a new checkpoint from items.
    pub fn from_items(items: &[SyncItem], timestamp: u64) -> Self {
        // Compute checkpoint ID from all item hashes
        let mut combined = Vec::new();
        for item in items {
            combined.extend_from_slice(item.hash.as_bytes());
        }
        Self {
            id: ContentHash::from_bytes(&combined),
            timestamp,
            item_count: items.len(),
        }
    }

    /// Create an empty checkpoint.
    pub fn empty(timestamp: u64) -> Self {
        Self {
            id: ContentHash::zero(),
            timestamp,
            item_count: 0,
        }
    }
}

/// A single item to sync.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncItem {
    /// Item path/identifier.
    pub path: String,
    /// Content hash.
    pub hash: ContentHash,
    /// Size in bytes.
    pub size: u64,
}

impl SyncItem {
    /// Create a new sync item.
    pub fn new(path: impl Into<String>, hash: ContentHash, size: u64) -> Self {
        Self {
            path: path.into(),
            hash,
            size,
        }
    }
}

/// A single operation in a sync batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SyncOperation {
    /// Add or update an item.
    Upsert {
        /// Path of the item.
        path: String,
        /// Delta to apply (or full content).
        delta: Delta,
        /// Target hash after applying delta.
        target_hash: ContentHash,
    },
    /// Remove an item.
    Remove {
        /// Path of the item to remove.
        path: String,
    },
}

/// A batch of deltas to apply.
#[derive(Debug, Clone)]
pub struct SyncBatch {
    /// Base checkpoint this batch applies to.
    pub base_checkpoint: ContentHash,
    /// Target checkpoint after applying batch.
    pub target_checkpoint: ContentHash,
    /// Ordered list of delta operations.
    pub operations: Vec<SyncOperation>,
    /// Protocol version used.
    pub protocol_version: u32,
}

impl SyncBatch {
    /// Create a new sync batch.
    pub fn new(
        base_checkpoint: ContentHash,
        target_checkpoint: ContentHash,
        operations: Vec<SyncOperation>,
        protocol_version: u32,
    ) -> Self {
        Self {
            base_checkpoint,
            target_checkpoint,
            operations,
            protocol_version,
        }
    }

    /// Create an empty batch (no changes).
    pub fn empty(checkpoint: ContentHash, protocol_version: u32) -> Self {
        Self {
            base_checkpoint: checkpoint.clone(),
            target_checkpoint: checkpoint,
            operations: Vec::new(),
            protocol_version,
        }
    }

    /// Check if this batch is empty (no operations).
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }
}

/// Error type for sync operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SyncError {
    /// Protocol version mismatch.
    VersionMismatch { expected: u32, got: u32 },
    /// Checkpoint mismatch (batch doesn't apply to current state).
    CheckpointMismatch,
    /// Missing content for delta computation.
    MissingContent(ContentHash),
    /// Diff/patch error.
    DiffError(DiffError),
    /// Compression/decompression error.
    CompressionError(String),
}

impl std::fmt::Display for SyncError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VersionMismatch { expected, got } => {
                write!(f, "version mismatch: expected {}, got {}", expected, got)
            }
            Self::CheckpointMismatch => write!(f, "checkpoint mismatch"),
            Self::MissingContent(hash) => write!(f, "missing content: {}", hash),
            Self::DiffError(e) => write!(f, "diff error: {}", e),
            Self::CompressionError(msg) => write!(f, "compression error: {}", msg),
        }
    }
}

impl std::error::Error for SyncError {}

impl From<DiffError> for SyncError {
    fn from(e: DiffError) -> Self {
        Self::DiffError(e)
    }
}

/// DeltaSync protocol for client-server synchronization.
///
/// Implements checkpoint-based incremental transfer with:
/// - Protocol version negotiation
/// - Delta compression using BinaryDiffer
/// - Batch operations with ordering
/// - Remove list for deleted assets
///
/// # Example
///
/// ```ignore
/// let protocol = DeltaSyncProtocol::new();
///
/// // Negotiate version with peer
/// let negotiated = protocol.negotiate(peer_version);
///
/// // Compute batch from local to remote state
/// let batch = protocol.compute_batch(&local_items, &remote_items, |hash| {
///     storage.get(hash)
/// })?;
///
/// // Compress for transfer
/// let compressed = protocol.compress_batch(&batch);
///
/// // On receiving end, decompress and apply
/// let batch = protocol.decompress_batch(&compressed)?;
/// let new_items = protocol.apply_batch(&batch, |hash| storage.get(hash), |data| {
///     storage.put(data)
/// })?;
/// ```
pub struct DeltaSyncProtocol {
    /// Protocol version (for negotiation).
    version: u32,
    /// Content differ for computing deltas.
    differ: BinaryDiffer,
    /// Compression enabled.
    compress: bool,
}

impl Default for DeltaSyncProtocol {
    fn default() -> Self {
        Self::new()
    }
}

impl DeltaSyncProtocol {
    /// Current protocol version.
    pub const CURRENT_VERSION: u32 = 1;

    /// Create a new DeltaSyncProtocol with default settings.
    pub fn new() -> Self {
        Self {
            version: Self::CURRENT_VERSION,
            differ: BinaryDiffer::new(),
            compress: true,
        }
    }

    /// Create a protocol instance with compression disabled.
    pub fn without_compression() -> Self {
        Self {
            version: Self::CURRENT_VERSION,
            differ: BinaryDiffer::new(),
            compress: false,
        }
    }

    /// Get the protocol version.
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Negotiate protocol version with peer.
    ///
    /// Returns the minimum of our version and peer version.
    pub fn negotiate(&self, peer_version: u32) -> u32 {
        self.version.min(peer_version)
    }

    /// Compute sync batch between two states.
    ///
    /// Compares local and remote item lists and generates delta operations
    /// to transform from local state to remote state.
    ///
    /// - Items in remote but not local: Upsert with full content
    /// - Items in both with different hash: Upsert with delta
    /// - Items in local but not remote: Remove
    pub fn compute_batch<F>(
        &self,
        local_items: &[SyncItem],
        remote_items: &[SyncItem],
        get_content: F,
    ) -> Result<SyncBatch, SyncError>
    where
        F: Fn(&ContentHash) -> Option<Vec<u8>>,
    {
        // Build lookup maps
        let local_map: HashMap<&str, &SyncItem> =
            local_items.iter().map(|i| (i.path.as_str(), i)).collect();
        let remote_map: HashMap<&str, &SyncItem> =
            remote_items.iter().map(|i| (i.path.as_str(), i)).collect();

        let mut operations = Vec::new();

        // Find additions and updates
        for remote_item in remote_items {
            match local_map.get(remote_item.path.as_str()) {
                None => {
                    // New item: send full content
                    let content = get_content(&remote_item.hash)
                        .ok_or_else(|| SyncError::MissingContent(remote_item.hash.clone()))?;
                    operations.push(SyncOperation::Upsert {
                        path: remote_item.path.clone(),
                        delta: Delta::Full(content),
                        target_hash: remote_item.hash.clone(),
                    });
                }
                Some(local_item) => {
                    if local_item.hash != remote_item.hash {
                        // Changed item: compute delta
                        let old_content = get_content(&local_item.hash)
                            .ok_or_else(|| SyncError::MissingContent(local_item.hash.clone()))?;
                        let new_content = get_content(&remote_item.hash)
                            .ok_or_else(|| SyncError::MissingContent(remote_item.hash.clone()))?;

                        let delta = self.differ.diff(&old_content, &new_content)?;

                        // If delta is larger than full content, use full
                        let delta = if delta.size_bytes() >= new_content.len() {
                            Delta::Full(new_content)
                        } else {
                            delta
                        };

                        operations.push(SyncOperation::Upsert {
                            path: remote_item.path.clone(),
                            delta,
                            target_hash: remote_item.hash.clone(),
                        });
                    }
                    // Same hash: no operation needed
                }
            }
        }

        // Find removals
        for local_item in local_items {
            if !remote_map.contains_key(local_item.path.as_str()) {
                operations.push(SyncOperation::Remove {
                    path: local_item.path.clone(),
                });
            }
        }

        // Sort operations for deterministic ordering (upserts before removes)
        operations.sort_by(|a, b| {
            use std::cmp::Ordering;
            match (a, b) {
                (SyncOperation::Upsert { path: p1, .. }, SyncOperation::Upsert { path: p2, .. }) => {
                    p1.cmp(p2)
                }
                (SyncOperation::Remove { path: p1 }, SyncOperation::Remove { path: p2 }) => {
                    p1.cmp(p2)
                }
                (SyncOperation::Upsert { .. }, SyncOperation::Remove { .. }) => Ordering::Less,
                (SyncOperation::Remove { .. }, SyncOperation::Upsert { .. }) => Ordering::Greater,
            }
        });

        let base_checkpoint = SyncCheckpoint::from_items(local_items, 0);
        let target_checkpoint = SyncCheckpoint::from_items(remote_items, 0);

        Ok(SyncBatch::new(
            base_checkpoint.id,
            target_checkpoint.id,
            operations,
            self.version,
        ))
    }

    /// Apply sync batch to local state.
    ///
    /// Processes operations in order and returns the new item list.
    pub fn apply_batch<F, G>(
        &self,
        batch: &SyncBatch,
        get_content: F,
        mut put_content: G,
    ) -> Result<Vec<SyncItem>, SyncError>
    where
        F: Fn(&ContentHash) -> Option<Vec<u8>>,
        G: FnMut(&[u8]) -> ContentHash,
    {
        // Verify protocol version compatibility
        if batch.protocol_version > self.version {
            return Err(SyncError::VersionMismatch {
                expected: self.version,
                got: batch.protocol_version,
            });
        }

        let mut result_items: HashMap<String, SyncItem> = HashMap::new();

        for op in &batch.operations {
            match op {
                SyncOperation::Upsert {
                    path,
                    delta,
                    target_hash,
                } => {
                    let new_content = match delta {
                        Delta::Full(data) => data.clone(),
                        _ => {
                            // Get existing content to apply patch
                            let existing_hash = result_items
                                .get(path)
                                .map(|i| &i.hash)
                                .ok_or_else(|| {
                                    SyncError::CompressionError(format!(
                                        "no existing content for path: {}",
                                        path
                                    ))
                                })?;

                            let existing_content = get_content(existing_hash).ok_or_else(|| {
                                SyncError::MissingContent(existing_hash.clone())
                            })?;

                            self.differ.apply(&existing_content, delta)?
                        }
                    };

                    let computed_hash = put_content(&new_content);

                    // Verify hash matches
                    if computed_hash != *target_hash {
                        return Err(SyncError::CheckpointMismatch);
                    }

                    result_items.insert(
                        path.clone(),
                        SyncItem::new(path.clone(), computed_hash, new_content.len() as u64),
                    );
                }
                SyncOperation::Remove { path } => {
                    result_items.remove(path);
                }
            }
        }

        // Convert to sorted list
        let mut items: Vec<SyncItem> = result_items.into_values().collect();
        items.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(items)
    }

    /// Compress batch for transfer using simple RLE compression.
    ///
    /// Format:
    /// - 4 bytes: magic "SYNC"
    /// - 4 bytes: protocol version (little-endian u32)
    /// - 32 bytes: base checkpoint
    /// - 32 bytes: target checkpoint
    /// - 4 bytes: operation count (little-endian u32)
    /// - For each operation:
    ///   - 1 byte: type (0x01 = Upsert, 0x02 = Remove)
    ///   - 4 bytes: path length
    ///   - N bytes: path
    ///   - For Upsert:
    ///     - 32 bytes: target hash
    ///     - 1 byte: delta type (0x00 = Full, 0x01 = BinaryPatch)
    ///     - 4 bytes: delta data length
    ///     - N bytes: delta data (RLE compressed if compress=true)
    pub fn compress_batch(&self, batch: &SyncBatch) -> Vec<u8> {
        let mut data = Vec::new();

        // Magic
        data.extend_from_slice(b"SYNC");

        // Protocol version
        data.extend_from_slice(&batch.protocol_version.to_le_bytes());

        // Checkpoints
        data.extend_from_slice(batch.base_checkpoint.as_bytes());
        data.extend_from_slice(batch.target_checkpoint.as_bytes());

        // Operation count
        data.extend_from_slice(&(batch.operations.len() as u32).to_le_bytes());

        for op in &batch.operations {
            match op {
                SyncOperation::Upsert {
                    path,
                    delta,
                    target_hash,
                } => {
                    data.push(0x01); // Upsert type

                    // Path
                    data.extend_from_slice(&(path.len() as u32).to_le_bytes());
                    data.extend_from_slice(path.as_bytes());

                    // Target hash
                    data.extend_from_slice(target_hash.as_bytes());

                    // Delta
                    let (delta_type, delta_data) = match delta {
                        Delta::Full(content) => (0x00u8, content.clone()),
                        Delta::BinaryPatch { patch } => (0x01u8, patch.clone()),
                        _ => (0x00u8, Vec::new()), // Fallback
                    };

                    data.push(delta_type);

                    // Compress delta data if enabled
                    let compressed_data = if self.compress {
                        Self::rle_compress(&delta_data)
                    } else {
                        delta_data
                    };

                    data.extend_from_slice(&(compressed_data.len() as u32).to_le_bytes());
                    data.extend_from_slice(&compressed_data);
                }
                SyncOperation::Remove { path } => {
                    data.push(0x02); // Remove type

                    // Path
                    data.extend_from_slice(&(path.len() as u32).to_le_bytes());
                    data.extend_from_slice(path.as_bytes());
                }
            }
        }

        data
    }

    /// Decompress batch from transfer format.
    pub fn decompress_batch(&self, data: &[u8]) -> Result<SyncBatch, SyncError> {
        if data.len() < 76 {
            // Minimum: magic(4) + version(4) + checkpoints(64) + count(4)
            return Err(SyncError::CompressionError("data too short".into()));
        }

        let mut pos = 0;

        // Magic
        if &data[pos..pos + 4] != b"SYNC" {
            return Err(SyncError::CompressionError("invalid magic".into()));
        }
        pos += 4;

        // Protocol version
        let protocol_version = u32::from_le_bytes([
            data[pos],
            data[pos + 1],
            data[pos + 2],
            data[pos + 3],
        ]);
        pos += 4;

        // Base checkpoint
        let mut base_bytes = [0u8; 32];
        base_bytes.copy_from_slice(&data[pos..pos + 32]);
        let base_checkpoint = ContentHash::from_raw(base_bytes);
        pos += 32;

        // Target checkpoint
        let mut target_bytes = [0u8; 32];
        target_bytes.copy_from_slice(&data[pos..pos + 32]);
        let target_checkpoint = ContentHash::from_raw(target_bytes);
        pos += 32;

        // Operation count
        let op_count = u32::from_le_bytes([
            data[pos],
            data[pos + 1],
            data[pos + 2],
            data[pos + 3],
        ]) as usize;
        pos += 4;

        let mut operations = Vec::with_capacity(op_count);

        for _ in 0..op_count {
            if pos >= data.len() {
                return Err(SyncError::CompressionError("unexpected end of data".into()));
            }

            let op_type = data[pos];
            pos += 1;

            // Path length
            if pos + 4 > data.len() {
                return Err(SyncError::CompressionError("truncated path length".into()));
            }
            let path_len = u32::from_le_bytes([
                data[pos],
                data[pos + 1],
                data[pos + 2],
                data[pos + 3],
            ]) as usize;
            pos += 4;

            // Path
            if pos + path_len > data.len() {
                return Err(SyncError::CompressionError("truncated path".into()));
            }
            let path = String::from_utf8(data[pos..pos + path_len].to_vec())
                .map_err(|e| SyncError::CompressionError(e.to_string()))?;
            pos += path_len;

            match op_type {
                0x01 => {
                    // Upsert
                    if pos + 32 > data.len() {
                        return Err(SyncError::CompressionError("truncated hash".into()));
                    }
                    let mut hash_bytes = [0u8; 32];
                    hash_bytes.copy_from_slice(&data[pos..pos + 32]);
                    let target_hash = ContentHash::from_raw(hash_bytes);
                    pos += 32;

                    if pos >= data.len() {
                        return Err(SyncError::CompressionError("truncated delta type".into()));
                    }
                    let delta_type = data[pos];
                    pos += 1;

                    if pos + 4 > data.len() {
                        return Err(SyncError::CompressionError("truncated delta length".into()));
                    }
                    let delta_len = u32::from_le_bytes([
                        data[pos],
                        data[pos + 1],
                        data[pos + 2],
                        data[pos + 3],
                    ]) as usize;
                    pos += 4;

                    if pos + delta_len > data.len() {
                        return Err(SyncError::CompressionError("truncated delta data".into()));
                    }
                    let compressed_data = &data[pos..pos + delta_len];
                    pos += delta_len;

                    // Decompress if needed
                    let delta_data = if self.compress {
                        Self::rle_decompress(compressed_data)
                            .map_err(|e| SyncError::CompressionError(e))?
                    } else {
                        compressed_data.to_vec()
                    };

                    let delta = match delta_type {
                        0x00 => Delta::Full(delta_data),
                        0x01 => Delta::BinaryPatch { patch: delta_data },
                        _ => {
                            return Err(SyncError::CompressionError(format!(
                                "unknown delta type: {}",
                                delta_type
                            )))
                        }
                    };

                    operations.push(SyncOperation::Upsert {
                        path,
                        delta,
                        target_hash,
                    });
                }
                0x02 => {
                    // Remove
                    operations.push(SyncOperation::Remove { path });
                }
                _ => {
                    return Err(SyncError::CompressionError(format!(
                        "unknown operation type: {}",
                        op_type
                    )))
                }
            }
        }

        Ok(SyncBatch::new(
            base_checkpoint,
            target_checkpoint,
            operations,
            protocol_version,
        ))
    }

    /// Simple RLE compression for repeated bytes.
    ///
    /// Format: For runs of 4+ same bytes: [0xFF, byte, count_hi, count_lo]
    /// For non-runs: literal bytes (0xFF is escaped as [0xFF, 0xFF, 0x00, 0x01])
    fn rle_compress(data: &[u8]) -> Vec<u8> {
        if data.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(data.len());
        let mut i = 0;

        while i < data.len() {
            let byte = data[i];
            let mut run_len = 1;

            // Count run length
            while i + run_len < data.len() && data[i + run_len] == byte && run_len < 65535 {
                run_len += 1;
            }

            if run_len >= 4 {
                // Encode run
                result.push(0xFF);
                result.push(byte);
                result.push((run_len >> 8) as u8);
                result.push((run_len & 0xFF) as u8);
            } else if byte == 0xFF {
                // Escape 0xFF
                for _ in 0..run_len {
                    result.push(0xFF);
                    result.push(0xFF);
                    result.push(0x00);
                    result.push(0x01);
                }
            } else {
                // Literal bytes
                for _ in 0..run_len {
                    result.push(byte);
                }
            }

            i += run_len;
        }

        result
    }

    /// Decompress RLE-compressed data.
    fn rle_decompress(data: &[u8]) -> Result<Vec<u8>, String> {
        let mut result = Vec::new();
        let mut i = 0;

        while i < data.len() {
            if data[i] == 0xFF {
                if i + 3 >= data.len() {
                    return Err("truncated RLE sequence".into());
                }

                let byte = data[i + 1];
                let count = ((data[i + 2] as usize) << 8) | (data[i + 3] as usize);

                for _ in 0..count {
                    result.push(byte);
                }
                i += 4;
            } else {
                result.push(data[i]);
                i += 1;
            }
        }

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// SHA-256 helper (deprecated, use ContentHash::from_bytes)
// ---------------------------------------------------------------------------

/// Compute the SHA-256 hash of `data` and return it as a `[u8; 32]`.
///
/// **Deprecated:** Use `ContentHash::from_bytes()` instead for type safety.
fn sha256(data: &[u8]) -> [u8; 32] {
    ContentHash::from_bytes(data).into_bytes()
}

// ---------------------------------------------------------------------------
// CachedPipeline
// ---------------------------------------------------------------------------

/// A fully compiled render pipeline together with its bind-group layout
/// and the SHA-256 hash of the WGSL source that produced it.
pub struct CachedPipeline {
    /// User-assigned pipeline identifier.
    pub id: u32,
    /// The compiled wgpu render pipeline.
    pub render_pipeline: wgpu::RenderPipeline,
    /// The bind-group layout used by this pipeline.
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// SHA-256 hash of the WGSL source (32 bytes).
    pub shader_hash: [u8; 32],
}

// ---------------------------------------------------------------------------
// ShaderCache
// ---------------------------------------------------------------------------

/// Deduplicates [`wgpu::ShaderModule`] allocations by keying on the
/// SHA-256 hash of the WGSL source.
///
/// Also maintains a map from source path to hash for file-path-based lookups.
pub struct ShaderCache {
    /// Compiled shader modules keyed by their SHA-256 hash (Arc-wrapped for sharing).
    pub modules: HashMap<[u8; 32], Arc<wgpu::ShaderModule>>,
    /// Maps source file paths to their SHA-256 hash.
    pub source_hashes: HashMap<String, [u8; 32]>,
}

impl ShaderCache {
    /// Create an empty shader cache.
    pub fn new() -> Self {
        Self {
            modules: HashMap::new(),
            source_hashes: HashMap::new(),
        }
    }

    /// Return a compiled shader module for `wgsl_source`.
    ///
    /// If a module with the same SHA-256 hash already exists in the cache it
    /// is returned without recompilation. Otherwise the source is compiled
    /// into a new [`wgpu::ShaderModule`] and stored in the cache.
    ///
    /// Returns the module (Arc-wrapped for sharing) together with its SHA-256 hash.
    pub fn get_or_compile(
        &mut self,
        device: &wgpu::Device,
        wgsl_source: &str,
    ) -> (Arc<wgpu::ShaderModule>, [u8; 32]) {
        let hash = sha256(wgsl_source.as_bytes());

        if let Some(module) = self.modules.get(&hash) {
            // Cheap Arc clone -- no GPU work.
            return (Arc::clone(module), hash);
        }

        let module = Arc::new(device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ShaderCache module"),
            source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
        }));

        self.modules.insert(hash, Arc::clone(&module));
        (module, hash)
    }

    /// Remove all cached modules and source-hash mappings, releasing the
    /// underlying GPU resources.
    pub fn clear(&mut self) {
        self.modules.clear();
        self.source_hashes.clear();
    }
}

impl Default for ShaderCache {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PipelineTable
// ---------------------------------------------------------------------------

/// Manages a collection of [`CachedPipeline`] entries together with a shared
/// [`ShaderCache`] for deduplicated shader compilation.
pub struct PipelineTable {
    /// Cached pipelines indexed by their numeric id.
    pub pipelines: HashMap<u32, CachedPipeline>,
    /// Shared shader cache for deduplicated compilation.
    pub shader_cache: ShaderCache,
}

impl PipelineTable {
    /// Create an empty pipeline table.
    pub fn new() -> Self {
        Self {
            pipelines: HashMap::new(),
            shader_cache: ShaderCache::new(),
        }
    }

    /// Alias for `new()` - creates an empty pipeline table.
    ///
    /// This is useful for hot-reload scenarios where we need a placeholder
    /// table before real pipelines are compiled.
    pub fn empty() -> Self {
        Self::new()
    }

    /// Insert a pre-built pipeline into the table, keyed by `id`.
    ///
    /// If a pipeline with the same `id` already exists it is silently
    /// replaced (dropping the old GPU resources).
    pub fn insert(&mut self, id: u32, pipeline: CachedPipeline) {
        self.pipelines.insert(id, pipeline);
    }

    /// Look up a cached pipeline by its numeric id.
    ///
    /// Returns `None` if no pipeline with that id exists.
    pub fn get(&self, id: u32) -> Option<&CachedPipeline> {
        self.pipelines.get(&id)
    }

    /// Remove a pipeline from the table.
    ///
    /// Returns `true` if the pipeline existed and was removed, `false` if
    /// no pipeline with that id was found.
    pub fn remove(&mut self, id: u32) -> bool {
        self.pipelines.remove(&id).is_some()
    }

    /// Number of cached pipelines currently in the table.
    pub fn len(&self) -> usize {
        self.pipelines.len()
    }

    /// Returns `true` if the table contains no pipelines.
    pub fn is_empty(&self) -> bool {
        self.pipelines.is_empty()
    }

    /// Compile a new render pipeline and insert it into the table.
    ///
    /// The WGSL source is deduplicated through the internal [`ShaderCache`].
    /// A default (empty) bind-group layout is used; callers that need custom
    /// layouts should construct a [`CachedPipeline`] manually and use
    /// [`insert`](Self::insert).
    ///
    /// # Errors
    ///
    /// Returns `Err(msg)` if shader compilation or pipeline creation fails.
    /// Note that wgpu may **panic** (via `wgpu::Device::create_shader_module`
    /// or `create_render_pipeline`) on invalid WGSL rather than returning an
    /// error. This method wraps those calls with `std::panic::catch_unwind`
    /// to convert panics into `Err` values.
    pub fn compile_pipeline<'a>(
        &mut self,
        device: &wgpu::Device,
        id: u32,
        wgsl_source: &str,
        vertex_entry: &'a str,
        fragment_entry: &'a str,
        vertex_layouts: &'a [wgpu::VertexBufferLayout<'a>],
        color_format: wgpu::TextureFormat,
    ) -> Result<u32, String> {
        // Compile (or fetch) the shader module.
        let (module, shader_hash) = self.shader_cache.get_or_compile(device, wgsl_source);

        // Create a default empty bind-group layout.
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(&format!("Pipeline {} BGL", id)),
                entries: &[],
            });

        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("Pipeline {} Layout", id)),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        // Catch panics from wgpu (e.g. invalid WGSL source).
        let render_pipeline = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(&format!("Pipeline {}", id)),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: vertex_entry,
                    buffers: vertex_layouts,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: fragment_entry,
                    targets: &[Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            })
        }))
        .map_err(|panic_payload| {
            let msg = panic_payload
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| panic_payload.downcast_ref::<String>().map(|s| s.as_str()))
                .unwrap_or("unknown wgpu panic");
            format!("pipeline compilation panicked: {msg}")
        })?;

        self.pipelines.insert(
            id,
            CachedPipeline {
                id,
                render_pipeline,
                bind_group_layout,
                shader_hash,
            },
        );

        Ok(id)
    }

    /// Create a PBR render pipeline for mesh rendering.
    ///
    /// This is the primary pipeline for PBR (Physically Based Rendering) mesh
    /// rendering. It expects vertex buffers with position, normal, UV, and
    /// optionally tangent attributes.
    ///
    /// # Bind Group Layouts
    ///
    /// The pipeline uses the following bind group layout (group 0):
    /// - Binding 0: Camera uniforms (view-projection matrix, camera position)
    /// - Binding 1: Material uniforms (base color, metallic, roughness, etc.)
    /// - Binding 2: Light uniforms (directional/point lights)
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `id` - Unique pipeline identifier.
    /// * `vertex_shader` - Compiled vertex shader module.
    /// * `fragment_shader` - Compiled fragment shader module.
    /// * `surface_format` - Output color attachment format.
    /// * `depth_format` - Depth attachment format (optional).
    /// * `sample_count` - MSAA sample count (1 = no MSAA).
    ///
    /// # Returns
    ///
    /// `Ok(id)` on success, `Err(msg)` if pipeline creation fails.
    pub fn create_pbr_pipeline(
        &mut self,
        device: &wgpu::Device,
        id: u32,
        vertex_shader: &wgpu::ShaderModule,
        fragment_shader: &wgpu::ShaderModule,
        surface_format: wgpu::TextureFormat,
        depth_format: Option<wgpu::TextureFormat>,
        sample_count: u32,
    ) -> Result<u32, String> {
        // Create bind group layout for PBR materials
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PBR Bind Group Layout"),
            entries: &[
                // Camera uniforms: view-projection matrix (64 bytes) + camera position (16 bytes)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(80), // mat4x4 + vec4
                    },
                    count: None,
                },
                // Material uniforms: base_color (16) + metallic_roughness (8) + emissive (16) + flags (16)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(64),
                    },
                    count: None,
                },
                // Light uniforms: direction (16) + color (16) + ambient (16)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(48),
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PBR Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Depth stencil state
        let depth_stencil = depth_format.map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });

        // Create render pipeline with panic catching
        let render_pipeline = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(&format!("PBR Pipeline {}", id)),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: vertex_shader,
                    entry_point: "vs_main",
                    buffers: &[pbr_vertex_buffer_layout()],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: fragment_shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil,
                multisample: wgpu::MultisampleState {
                    count: sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            })
        }))
        .map_err(|panic_payload| {
            let msg = panic_payload
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| panic_payload.downcast_ref::<String>().map(|s| s.as_str()))
                .unwrap_or("unknown wgpu panic");
            format!("PBR pipeline creation panicked: {msg}")
        })?;

        // Use a zero hash since we're using pre-compiled shader modules
        let shader_hash = [0u8; 32];

        self.pipelines.insert(
            id,
            CachedPipeline {
                id,
                render_pipeline,
                bind_group_layout,
                shader_hash,
            },
        );

        Ok(id)
    }
}

impl Default for PipelineTable {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PBR Vertex Types and Layouts
// ---------------------------------------------------------------------------

/// A PBR vertex with position, normal, and UV coordinates.
///
/// This is the standard vertex format for PBR mesh rendering:
/// - `position`: World-space position (vec3<f32>)
/// - `normal`: Surface normal (vec3<f32>)
/// - `uv`: Texture coordinates (vec2<f32>)
///
/// Total size: 32 bytes (12 + 12 + 8)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PbrVertex {
    /// World-space position.
    pub position: [f32; 3],
    /// Surface normal (should be normalized).
    pub normal: [f32; 3],
    /// Texture coordinates.
    pub uv: [f32; 2],
}

impl PbrVertex {
    /// Create a new PBR vertex.
    pub const fn new(position: [f32; 3], normal: [f32; 3], uv: [f32; 2]) -> Self {
        Self { position, normal, uv }
    }
}

/// A PBR vertex with tangent for normal mapping.
///
/// Extended vertex format including tangent vector for normal map calculations:
/// - `position`: World-space position (vec3<f32>)
/// - `normal`: Surface normal (vec3<f32>)
/// - `uv`: Texture coordinates (vec2<f32>)
/// - `tangent`: Tangent vector with handedness (vec4<f32>)
///
/// Total size: 48 bytes (12 + 12 + 8 + 16)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PbrVertexTangent {
    /// World-space position.
    pub position: [f32; 3],
    /// Surface normal (should be normalized).
    pub normal: [f32; 3],
    /// Texture coordinates.
    pub uv: [f32; 2],
    /// Tangent vector with handedness in w component.
    pub tangent: [f32; 4],
}

impl PbrVertexTangent {
    /// Create a new PBR vertex with tangent.
    pub const fn new(
        position: [f32; 3],
        normal: [f32; 3],
        uv: [f32; 2],
        tangent: [f32; 4],
    ) -> Self {
        Self { position, normal, uv, tangent }
    }
}

/// Returns the vertex buffer layout for [`PbrVertex`].
///
/// Layout:
/// - Location 0: position (vec3<f32>), offset 0
/// - Location 1: normal (vec3<f32>), offset 12
/// - Location 2: uv (vec2<f32>), offset 24
///
/// Array stride: 32 bytes
pub const fn pbr_vertex_buffer_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<PbrVertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            // position: vec3<f32>
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            // normal: vec3<f32>
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 12,
                shader_location: 1,
            },
            // uv: vec2<f32>
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 24,
                shader_location: 2,
            },
        ],
    }
}

/// Returns the vertex buffer layout for [`PbrVertexTangent`].
///
/// Layout:
/// - Location 0: position (vec3<f32>), offset 0
/// - Location 1: normal (vec3<f32>), offset 12
/// - Location 2: uv (vec2<f32>), offset 24
/// - Location 3: tangent (vec4<f32>), offset 32
///
/// Array stride: 48 bytes
pub const fn pbr_vertex_tangent_buffer_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<PbrVertexTangent>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            // position: vec3<f32>
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            // normal: vec3<f32>
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 12,
                shader_location: 1,
            },
            // uv: vec2<f32>
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 24,
                shader_location: 2,
            },
            // tangent: vec4<f32>
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 32,
                shader_location: 3,
            },
        ],
    }
}

/// Default PBR shader source for basic mesh rendering.
///
/// This shader provides a minimal PBR implementation with:
/// - Lambert diffuse
/// - GGX specular
/// - Single directional light
///
/// Use this as a starting point or for testing pipeline creation.
pub const PBR_SHADER_SRC: &str = r#"
// Camera uniforms
struct CameraUniforms {
    view_projection: mat4x4<f32>,
    camera_position: vec4<f32>,
}

// Material uniforms
struct MaterialUniforms {
    base_color: vec4<f32>,
    metallic_roughness: vec2<f32>,
    _padding1: vec2<f32>,
    emissive: vec4<f32>,
    flags: vec4<f32>,
}

// Light uniforms
struct LightUniforms {
    direction: vec4<f32>,
    color: vec4<f32>,
    ambient: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> material: MaterialUniforms;
@group(0) @binding(2) var<uniform> light: LightUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.clip_position = camera.view_projection * vec4<f32>(input.position, 1.0);
    output.world_position = input.position;
    output.world_normal = input.normal;
    output.uv = input.uv;
    return output;
}

// GGX/Trowbridge-Reitz NDF
fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let n_dot_h2 = n_dot_h * n_dot_h;
    let denom = n_dot_h2 * (a2 - 1.0) + 1.0;
    return a2 / (3.14159265 * denom * denom);
}

// Schlick-GGX geometry function
fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k);
}

// Smith's geometry function
fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let ggx_v = geometry_schlick_ggx(n_dot_v, roughness);
    let ggx_l = geometry_schlick_ggx(n_dot_l, roughness);
    return ggx_v * ggx_l;
}

// Fresnel-Schlick approximation
fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(input.world_normal);
    let v = normalize(camera.camera_position.xyz - input.world_position);
    let l = normalize(-light.direction.xyz);
    let h = normalize(v + l);

    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_v = max(dot(n, v), 0.0);
    let n_dot_h = max(dot(n, h), 0.0);
    let h_dot_v = max(dot(h, v), 0.0);

    let metallic = material.metallic_roughness.x;
    let roughness = material.metallic_roughness.y;
    let albedo = material.base_color.rgb;

    // Dielectric F0 = 0.04, metallic F0 = albedo
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    // Cook-Torrance BRDF
    let d = distribution_ggx(n_dot_h, roughness);
    let g = geometry_smith(n_dot_v, n_dot_l, roughness);
    let f = fresnel_schlick(h_dot_v, f0);

    let numerator = d * g * f;
    let denominator = 4.0 * n_dot_v * n_dot_l + 0.0001;
    let specular = numerator / denominator;

    // Energy conservation
    let k_s = f;
    let k_d = (vec3<f32>(1.0) - k_s) * (1.0 - metallic);

    // Outgoing radiance
    let lo = (k_d * albedo / 3.14159265 + specular) * light.color.rgb * n_dot_l;

    // Ambient + emissive
    let ambient = light.ambient.rgb * albedo;
    let emissive = material.emissive.rgb;

    let color = ambient + lo + emissive;

    // Simple tone mapping (Reinhard)
    let mapped = color / (color + vec3<f32>(1.0));

    return vec4<f32>(mapped, material.base_color.a);
}
"#;

// ---------------------------------------------------------------------------
// ShardedPipelineTable — concurrent pipeline cache with sharding
// ---------------------------------------------------------------------------

use parking_lot::RwLock;

/// Default number of shards for pipeline cache.
pub const DEFAULT_SHARD_COUNT: usize = 16;

/// A single shard containing a subset of cached pipelines.
pub struct PipelineShard {
    /// Pipelines in this shard.
    pipelines: HashMap<u32, CachedPipeline>,
}

impl PipelineShard {
    fn new() -> Self {
        Self {
            pipelines: HashMap::new(),
        }
    }
}

/// Thread-safe sharded pipeline cache for high-concurrency scenarios.
///
/// Pipelines are distributed across `N` shards based on their ID.
/// Each shard is protected by its own RwLock, allowing concurrent
/// reads and writes to different shards.
///
/// # NUMA Hints
///
/// When creating the table, you can specify `numa_node` to hint which
/// NUMA node this table is associated with. This is advisory only;
/// actual NUMA-aware allocation requires platform-specific code.
///
/// # Example
///
/// ```ignore
/// let table = ShardedPipelineTable::new(16);
/// table.insert(pipeline);
/// let p = table.get(42).unwrap();
/// ```
pub struct ShardedPipelineTable {
    /// Shards, each containing a subset of pipelines.
    shards: Vec<RwLock<PipelineShard>>,
    /// Number of shards (power of 2 for fast modulo).
    shard_count: usize,
    /// Shared shader cache (single instance, not sharded).
    shader_cache: RwLock<ShaderCache>,
    /// Advisory NUMA node hint.
    pub numa_node: Option<usize>,
}

impl ShardedPipelineTable {
    /// Create a new sharded table with the specified shard count.
    ///
    /// `shard_count` should be a power of 2 for optimal performance.
    pub fn new(shard_count: usize) -> Self {
        let shard_count = shard_count.max(1).next_power_of_two();
        let shards = (0..shard_count)
            .map(|_| RwLock::new(PipelineShard::new()))
            .collect();
        Self {
            shards,
            shard_count,
            shader_cache: RwLock::new(ShaderCache::new()),
            numa_node: None,
        }
    }

    /// Create a sharded table with a NUMA node hint.
    pub fn with_numa_node(shard_count: usize, numa_node: usize) -> Self {
        let mut table = Self::new(shard_count);
        table.numa_node = Some(numa_node);
        table
    }

    /// Compute the shard index for a pipeline ID.
    #[inline]
    fn shard_index(&self, id: u32) -> usize {
        (id as usize) & (self.shard_count - 1)
    }

    /// Insert a pre-built pipeline.
    pub fn insert(&self, pipeline: CachedPipeline) {
        let idx = self.shard_index(pipeline.id);
        let mut shard = self.shards[idx].write();
        shard.pipelines.insert(pipeline.id, pipeline);
    }

    /// Look up a pipeline by ID.
    ///
    /// Returns a clone of the pipeline data (excluding non-Clone wgpu types).
    pub fn contains(&self, id: u32) -> bool {
        let idx = self.shard_index(id);
        let shard = self.shards[idx].read();
        shard.pipelines.contains_key(&id)
    }

    /// Remove a pipeline by ID.
    ///
    /// Returns `true` if the pipeline was found and removed.
    pub fn remove(&self, id: u32) -> bool {
        let idx = self.shard_index(id);
        let mut shard = self.shards[idx].write();
        shard.pipelines.remove(&id).is_some()
    }

    /// Total number of cached pipelines across all shards.
    pub fn len(&self) -> usize {
        self.shards
            .iter()
            .map(|s| s.read().pipelines.len())
            .sum()
    }

    /// Returns `true` if no pipelines are cached.
    pub fn is_empty(&self) -> bool {
        self.shards.iter().all(|s| s.read().pipelines.is_empty())
    }

    /// Get the number of pipelines in a specific shard.
    pub fn shard_len(&self, shard_idx: usize) -> usize {
        if shard_idx >= self.shard_count {
            return 0;
        }
        self.shards[shard_idx].read().pipelines.len()
    }

    /// Get statistics about shard distribution.
    pub fn shard_stats(&self) -> ShardStats {
        let counts: Vec<usize> = self.shards.iter().map(|s| s.read().pipelines.len()).collect();
        let total: usize = counts.iter().sum();
        let min = counts.iter().copied().min().unwrap_or(0);
        let max = counts.iter().copied().max().unwrap_or(0);
        let avg = if self.shard_count > 0 {
            total as f64 / self.shard_count as f64
        } else {
            0.0
        };
        ShardStats {
            shard_count: self.shard_count,
            total_pipelines: total,
            min_shard_size: min,
            max_shard_size: max,
            avg_shard_size: avg,
        }
    }

    /// Execute a function with read access to a pipeline.
    ///
    /// The callback receives a reference to the pipeline if found.
    pub fn with_pipeline<F, R>(&self, id: u32, f: F) -> Option<R>
    where
        F: FnOnce(&CachedPipeline) -> R,
    {
        let idx = self.shard_index(id);
        let shard = self.shards[idx].read();
        shard.pipelines.get(&id).map(f)
    }

    /// Clear all pipelines from all shards.
    pub fn clear(&self) {
        for shard in &self.shards {
            shard.write().pipelines.clear();
        }
        self.shader_cache.write().clear();
    }

    /// Get or compile a shader module (thread-safe).
    pub fn get_or_compile_shader(
        &self,
        device: &wgpu::Device,
        wgsl_source: &str,
    ) -> (Arc<wgpu::ShaderModule>, [u8; 32]) {
        self.shader_cache.write().get_or_compile(device, wgsl_source)
    }
}

impl Default for ShardedPipelineTable {
    fn default() -> Self {
        Self::new(DEFAULT_SHARD_COUNT)
    }
}

/// Statistics about shard distribution in a ShardedPipelineTable.
#[derive(Debug, Clone)]
pub struct ShardStats {
    /// Number of shards.
    pub shard_count: usize,
    /// Total number of pipelines.
    pub total_pipelines: usize,
    /// Size of the smallest shard.
    pub min_shard_size: usize,
    /// Size of the largest shard.
    pub max_shard_size: usize,
    /// Average shard size.
    pub avg_shard_size: f64,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── SHA-256 ────────────────────────────────────────────────────────────

    #[test]
    fn test_sha256_same_input_same_hash() {
        let a = sha256(b"hello");
        let b = sha256(b"hello");
        assert_eq!(a, b);
    }

    #[test]
    fn test_sha256_different_input_different_hash() {
        let a = sha256(b"hello");
        let b = sha256(b"world");
        assert_ne!(a, b);
    }

    #[test]
    #[cfg(not(feature = "blake3"))]
    fn test_sha256_known_vector() {
        // Known SHA-256("abc") from FIPS-180 test vector.
        // Note: sha256() helper uses ContentHash::from_bytes(), which uses the
        // compile-time selected algorithm. This test only passes without blake3.
        let hash = sha256(b"abc");
        let expected: [u8; 32] = [
            0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
            0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
            0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
            0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad,
        ];
        assert_eq!(hash, expected);
    }

    #[test]
    #[cfg(not(feature = "blake3"))]
    fn test_sha256_empty_input() {
        // Note: sha256() helper uses ContentHash::from_bytes(), which uses the
        // compile-time selected algorithm. This test only passes without blake3.
        let hash = sha256(b"");
        let expected: [u8; 32] = [
            0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14,
            0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f, 0xb9, 0x24,
            0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c,
            0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52, 0xb8, 0x55,
        ];
        assert_eq!(hash, expected);
    }

    // ── ContentHash ────────────────────────────────────────────────────────

    #[test]
    fn test_content_hash_from_bytes() {
        let h1 = ContentHash::from_bytes(b"hello");
        let h2 = ContentHash::from_bytes(b"hello");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_content_hash_from_raw() {
        let bytes = [0xab; 32];
        let hash = ContentHash::from_raw(bytes);
        assert_eq!(hash.as_bytes(), &bytes);
        assert_eq!(hash.into_bytes(), bytes);
    }

    #[test]
    fn test_content_hash_display_hex() {
        let hash = ContentHash::from_bytes(b"abc");
        let hex = format!("{}", hash);
        assert_eq!(hex.len(), 64);
        // Algorithm-specific expected values
        #[cfg(not(feature = "blake3"))]
        assert_eq!(hex, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        #[cfg(feature = "blake3")]
        assert_eq!(hex, "6437b3ac38465133ffb63b75273a8db548c558465d79db03fd359c6cd5bd9d85");
    }

    #[test]
    fn test_content_hash_debug() {
        let hash = ContentHash::from_bytes(b"test");
        let debug = format!("{:?}", hash);
        assert!(debug.starts_with("ContentHash("));
        assert!(debug.ends_with(")"));
    }

    #[test]
    fn test_content_hash_from_str() {
        // Round-trip: hash -> hex string -> parse -> same hash
        let original = ContentHash::from_bytes(b"round trip test");
        let hex = format!("{}", original);
        let parsed: ContentHash = hex.parse().expect("should parse");
        assert_eq!(parsed, original);
    }

    #[test]
    #[cfg(not(feature = "blake3"))]
    fn test_content_hash_sha256_known_vector() {
        // Known SHA-256 test vector
        let hex = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        let parsed: ContentHash = hex.parse().expect("should parse");
        let expected = ContentHash::from_bytes(b"abc");
        assert_eq!(parsed, expected);
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_content_hash_blake3_known_vector() {
        // Known BLAKE3 test vector
        let hex = "6437b3ac38465133ffb63b75273a8db548c558465d79db03fd359c6cd5bd9d85";
        let parsed: ContentHash = hex.parse().expect("should parse");
        let expected = ContentHash::from_bytes(b"abc");
        assert_eq!(parsed, expected);
    }

    #[test]
    fn test_content_hash_from_str_invalid_length() {
        let result: Result<ContentHash, _> = "abc".parse();
        assert!(matches!(result, Err(ContentHashParseError::InvalidLength(3))));
    }

    #[test]
    fn test_content_hash_from_str_invalid_hex() {
        let result: Result<ContentHash, _> = "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz".parse();
        assert!(matches!(result, Err(ContentHashParseError::InvalidHex)));
    }

    #[test]
    fn test_content_hash_zero() {
        let z = ContentHash::zero();
        assert!(z.is_zero());
        assert_eq!(z.as_bytes(), &[0u8; 32]);
    }

    #[test]
    fn test_content_hash_hash_trait() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        let h1 = ContentHash::from_bytes(b"hello");
        let h2 = ContentHash::from_bytes(b"hello");
        let h3 = ContentHash::from_bytes(b"world");

        set.insert(h1);
        assert!(set.contains(&h2)); // same content, same hash
        assert!(!set.contains(&h3)); // different content
    }

    #[test]
    fn test_content_hash_algorithm() {
        let algo = ContentHash::algorithm();
        #[cfg(feature = "blake3")]
        assert_eq!(algo, "blake3");
        #[cfg(not(feature = "blake3"))]
        assert_eq!(algo, "sha256");
    }

    #[test]
    fn test_content_hash_deterministic() {
        // Hash should be deterministic regardless of algorithm
        let h1 = ContentHash::from_bytes(b"test data");
        let h2 = ContentHash::from_bytes(b"test data");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_content_hash_different_inputs() {
        // Different inputs should produce different hashes
        let h1 = ContentHash::from_bytes(b"input a");
        let h2 = ContentHash::from_bytes(b"input b");
        assert_ne!(h1, h2);
    }

    // ── HashAlgorithm & from_data_with_algo ────────────────────────────────

    #[test]
    fn test_hash_algorithm_default() {
        let algo = HashAlgorithm::default();
        #[cfg(feature = "blake3")]
        assert_eq!(algo, HashAlgorithm::Blake3);
        #[cfg(not(feature = "blake3"))]
        assert_eq!(algo, HashAlgorithm::Sha256);
    }

    #[test]
    fn test_hash_algorithm_name() {
        assert_eq!(HashAlgorithm::Sha256.name(), "sha256");
        #[cfg(feature = "blake3")]
        assert_eq!(HashAlgorithm::Blake3.name(), "blake3");
    }

    #[test]
    fn test_from_data_with_algo_sha256() {
        // SHA-256 should always be available
        let hash = ContentHash::from_data_with_algo(b"hello world", HashAlgorithm::Sha256);
        assert_ne!(hash.0, [0u8; 32]);
        // Known SHA-256 hash of "hello world"
        let expected_hex = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";
        assert_eq!(format!("{}", hash), expected_hex);
    }

    #[test]
    fn test_from_data_with_algo_sha256_deterministic() {
        let data = b"test data for sha256";
        let h1 = ContentHash::from_data_with_algo(data, HashAlgorithm::Sha256);
        let h2 = ContentHash::from_data_with_algo(data, HashAlgorithm::Sha256);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_from_data_with_algo_sha256_different_inputs() {
        let h1 = ContentHash::from_data_with_algo(b"input a", HashAlgorithm::Sha256);
        let h2 = ContentHash::from_data_with_algo(b"input b", HashAlgorithm::Sha256);
        assert_ne!(h1, h2);
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_from_data_with_algo_blake3() {
        let hash = ContentHash::from_data_with_algo(b"hello world", HashAlgorithm::Blake3);
        assert_ne!(hash.0, [0u8; 32]);
        // Known BLAKE3 hash of "hello world"
        let expected_hex = "d74981efa70a0c880b8d8c1985d075dbcbf679b99a5f9914e5aaf96b831a9e24";
        assert_eq!(format!("{}", hash), expected_hex);
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_from_data_with_algo_blake3_deterministic() {
        let data = b"test data for blake3";
        let h1 = ContentHash::from_data_with_algo(data, HashAlgorithm::Blake3);
        let h2 = ContentHash::from_data_with_algo(data, HashAlgorithm::Blake3);
        assert_eq!(h1, h2);
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_from_data_with_algo_blake3_differs_from_sha256() {
        let data = b"same input different algorithm";
        let sha = ContentHash::from_data_with_algo(data, HashAlgorithm::Sha256);
        let blake = ContentHash::from_data_with_algo(data, HashAlgorithm::Blake3);
        assert_ne!(sha, blake, "SHA-256 and BLAKE3 should produce different hashes for same input");
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_from_data_with_algo_blake3_empty_input() {
        let hash = ContentHash::from_data_with_algo(b"", HashAlgorithm::Blake3);
        assert_ne!(hash.0, [0u8; 32]);
    }

    #[test]
    fn test_from_data_with_algo_sha256_empty_input() {
        let hash = ContentHash::from_data_with_algo(b"", HashAlgorithm::Sha256);
        assert_ne!(hash.0, [0u8; 32]);
        // Known SHA-256 hash of empty string
        let expected_hex = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        assert_eq!(format!("{}", hash), expected_hex);
    }

    #[test]
    fn test_hash_algorithm_default_matches_from_bytes() {
        // from_bytes should use the same algorithm as HashAlgorithm::default()
        let data = b"consistent algorithm test";
        let from_bytes_hash = ContentHash::from_bytes(data);
        let with_algo_hash = ContentHash::from_data_with_algo(data, HashAlgorithm::default());
        assert_eq!(from_bytes_hash, with_algo_hash);
    }

    // ── FileBackend ────────────────────────────────────────────────────────

    fn create_temp_store() -> (tempfile::TempDir, FileBackend) {
        let dir = tempfile::tempdir().expect("create temp dir");
        let store = FileBackend::new(dir.path()).expect("create store");
        (dir, store)
    }

    #[test]
    fn test_file_backend_put_get_roundtrip() {
        let (_dir, store) = create_temp_store();
        let data = b"hello world";
        let hash = store.put(data).expect("put");
        let retrieved = store.get(&hash).expect("get").expect("should exist");
        assert_eq!(retrieved, data);
    }

    #[test]
    fn test_file_backend_has() {
        let (_dir, store) = create_temp_store();
        let data = b"test data";
        let hash = store.put(data).expect("put");
        assert!(store.has(&hash));

        let missing = ContentHash::from_bytes(b"nonexistent");
        assert!(!store.has(&missing));
    }

    #[test]
    fn test_file_backend_delete() {
        let (_dir, store) = create_temp_store();
        let data = b"to be deleted";
        let hash = store.put(data).expect("put");
        assert!(store.has(&hash));

        assert!(store.delete(&hash).expect("delete"));
        assert!(!store.has(&hash));

        // Deleting again returns false
        assert!(!store.delete(&hash).expect("delete again"));
    }

    #[test]
    fn test_file_backend_size() {
        let (_dir, store) = create_temp_store();
        let data = b"12345678901234567890"; // 20 bytes
        let hash = store.put(data).expect("put");

        let size = store.size(&hash).expect("size").expect("should exist");
        assert_eq!(size, 20);

        let missing = ContentHash::from_bytes(b"nonexistent");
        assert!(store.size(&missing).expect("size").is_none());
    }

    #[test]
    fn test_file_backend_dedup() {
        let (_dir, store) = create_temp_store();
        let data = b"same content";

        let hash1 = store.put(data).expect("put1");
        let hash2 = store.put(data).expect("put2");

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_file_backend_list() {
        let (_dir, store) = create_temp_store();

        let h1 = store.put(b"one").expect("put");
        let h2 = store.put(b"two").expect("put");
        let h3 = store.put(b"three").expect("put");

        let hashes = store.list().expect("list");
        assert_eq!(hashes.len(), 3);
        assert!(hashes.contains(&h1));
        assert!(hashes.contains(&h2));
        assert!(hashes.contains(&h3));
    }

    #[test]
    fn test_file_backend_tree_put_get() {
        let (_dir, store) = create_temp_store();

        let h1 = store.put(b"file1").expect("put");
        let h2 = store.put(b"file2").expect("put");

        let entries = vec![
            (h1, "foo.txt".to_string()),
            (h2, "bar.txt".to_string()),
        ];

        let tree_hash = store.tree_put(&entries).expect("tree_put");
        let retrieved = store.tree_get(&tree_hash).expect("tree_get").expect("should exist");

        assert_eq!(retrieved.len(), 2);
        assert_eq!(retrieved[0], (h1, "foo.txt".to_string()));
        assert_eq!(retrieved[1], (h2, "bar.txt".to_string()));
    }

    #[test]
    fn test_file_backend_open_nonexistent() {
        let result = FileBackend::open("/nonexistent/path/that/does/not/exist");
        assert!(result.is_err());
    }

    #[test]
    fn test_file_backend_git_style_layout() {
        let (dir, store) = create_temp_store();
        let data = b"abc";
        let hash = store.put(data).expect("put");

        // Hash of "abc" starts with "ba78..."
        let hex = format!("{}", hash);
        let (prefix, suffix) = hex.split_at(2);

        let expected_path = dir.path().join(prefix).join(suffix);
        assert!(expected_path.exists(), "blob should be at git-style path");
    }

    // ── ChunkedContent / Streaming API ─────────────────────────────────────

    #[test]
    fn test_chunked_content_serialize_deserialize() {
        let chunks = vec![
            ContentHash::from_bytes(b"chunk1"),
            ContentHash::from_bytes(b"chunk2"),
            ContentHash::from_bytes(b"chunk3"),
        ];
        let manifest = ChunkedContent {
            total_size: 768 * 1024,
            chunk_size: 256 * 1024,
            chunks,
        };

        let serialized = manifest.serialize();
        let deserialized = ChunkedContent::deserialize(&serialized).expect("deserialize");

        assert_eq!(manifest, deserialized);
    }

    #[test]
    fn test_put_stream_small_data() {
        let (_dir, store) = create_temp_store();
        let data = b"small data that fits in one chunk";

        let manifest_hash = store
            .put_stream(&mut &data[..])
            .expect("put_stream");

        let manifest = store
            .get_manifest(&manifest_hash)
            .expect("get_manifest")
            .expect("manifest exists");

        assert_eq!(manifest.total_size, data.len() as u64);
        assert_eq!(manifest.chunks.len(), 1);
    }

    #[test]
    fn test_put_stream_large_data() {
        let (_dir, store) = create_temp_store();

        // Create data that spans multiple chunks (use small chunk size for testing)
        let chunk_size = 1024; // 1KB chunks for testing
        let data: Vec<u8> = (0..3500).map(|i| (i % 256) as u8).collect(); // 3.5KB

        let manifest_hash = store
            .put_stream_with_chunk_size(&mut &data[..], chunk_size)
            .expect("put_stream");

        let manifest = store
            .get_manifest(&manifest_hash)
            .expect("get_manifest")
            .expect("manifest exists");

        assert_eq!(manifest.total_size, 3500);
        assert_eq!(manifest.chunk_size, chunk_size);
        assert_eq!(manifest.chunks.len(), 4); // 3 full chunks + 1 partial
    }

    #[test]
    fn test_get_stream_roundtrip() {
        let (_dir, store) = create_temp_store();

        let chunk_size = 1024;
        let data: Vec<u8> = (0..5000).map(|i| (i % 256) as u8).collect();

        let manifest_hash = store
            .put_stream_with_chunk_size(&mut &data[..], chunk_size)
            .expect("put_stream");

        let mut reader = store
            .get_stream(&manifest_hash)
            .expect("get_stream")
            .expect("reader exists");

        assert_eq!(reader.total_size(), 5000);
        assert_eq!(reader.chunk_count(), 5);

        let mut retrieved = Vec::new();
        std::io::Read::read_to_end(&mut reader, &mut retrieved).expect("read");

        assert_eq!(retrieved, data);
        assert_eq!(reader.bytes_read(), 5000);
    }

    #[test]
    fn test_get_stream_partial_reads() {
        let (_dir, store) = create_temp_store();

        let chunk_size = 100;
        let data: Vec<u8> = (0..350).map(|i| (i % 256) as u8).collect();

        let manifest_hash = store
            .put_stream_with_chunk_size(&mut &data[..], chunk_size)
            .expect("put_stream");

        let mut reader = store
            .get_stream(&manifest_hash)
            .expect("get_stream")
            .expect("reader exists");

        // Read in small increments
        let mut retrieved = Vec::new();
        let mut buf = [0u8; 37]; // Non-aligned read size
        loop {
            let n = std::io::Read::read(&mut reader, &mut buf).expect("read");
            if n == 0 {
                break;
            }
            retrieved.extend_from_slice(&buf[..n]);
        }

        assert_eq!(retrieved, data);
    }

    #[test]
    fn test_get_stream_missing_manifest() {
        let (_dir, store) = create_temp_store();

        let missing = ContentHash::from_bytes(b"nonexistent");
        let result = store.get_stream(&missing).expect("get_stream");

        assert!(result.is_none());
    }

    #[test]
    fn test_chunked_content_hash() {
        let manifest1 = ChunkedContent {
            total_size: 1000,
            chunk_size: 256,
            chunks: vec![ContentHash::from_bytes(b"a")],
        };
        let manifest2 = ChunkedContent {
            total_size: 1000,
            chunk_size: 256,
            chunks: vec![ContentHash::from_bytes(b"a")],
        };
        let manifest3 = ChunkedContent {
            total_size: 2000,
            chunk_size: 256,
            chunks: vec![ContentHash::from_bytes(b"a")],
        };

        assert_eq!(manifest1.hash(), manifest2.hash());
        assert_ne!(manifest1.hash(), manifest3.hash());
    }

    #[test]
    fn test_content_tree_chunked_entry() {
        let (_dir, store) = create_temp_store();

        // Create chunked content
        let chunk_size = 1024;
        let data: Vec<u8> = (0..3000).map(|i| (i % 256) as u8).collect();
        let manifest_hash = store
            .put_stream_with_chunk_size(&mut &data[..], chunk_size)
            .expect("put_stream");

        // Create tree with chunked entry
        let tree = ContentTree::from_entries(vec![
            TreeEntry::chunked("large_file.bin", manifest_hash),
            TreeEntry::blob("small.txt", ContentHash::from_bytes(b"small")),
        ]);

        // Store and load tree
        let tree_hash = tree.store(&store).expect("store tree");
        let loaded = ContentTree::load(&store, &tree_hash)
            .expect("load tree")
            .expect("tree should exist");

        // Verify chunked entry
        let entry = loaded.get("large_file.bin").expect("entry exists");
        assert_eq!(entry.entry_type, TreeEntryType::Chunked);
        assert_eq!(entry.hash, manifest_hash);

        // Verify we can read the chunked content via the manifest hash
        let mut reader = store.get_stream(&entry.hash).expect("get_stream").expect("exists");
        let mut retrieved = Vec::new();
        std::io::Read::read_to_end(&mut reader, &mut retrieved).expect("read");
        assert_eq!(retrieved, data);
    }

    #[test]
    fn test_streaming_memory_efficiency() {
        // Test that streaming uses bounded memory by processing large data
        // in chunks without loading everything at once.
        let (_dir, store) = create_temp_store();

        // Create 10MB of test data (enough to verify chunking, not too slow)
        let total_size = 10 * 1024 * 1024; // 10MB
        let chunk_size = 256 * 1024; // 256KB chunks

        // Generate data in a streaming fashion
        struct PatternReader {
            pos: usize,
            len: usize,
        }
        impl std::io::Read for PatternReader {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                if self.pos >= self.len {
                    return Ok(0);
                }
                let to_read = buf.len().min(self.len - self.pos);
                for (i, byte) in buf[..to_read].iter_mut().enumerate() {
                    *byte = ((self.pos + i) % 256) as u8;
                }
                self.pos += to_read;
                Ok(to_read)
            }
        }

        let mut reader = PatternReader { pos: 0, len: total_size };
        let manifest_hash = store
            .put_stream_with_chunk_size(&mut reader, chunk_size)
            .expect("put_stream");

        // Verify manifest
        let manifest = store.get_manifest(&manifest_hash).expect("get").expect("exists");
        assert_eq!(manifest.total_size, total_size as u64);
        assert_eq!(manifest.chunk_size, chunk_size);
        let expected_chunks = (total_size + chunk_size - 1) / chunk_size;
        assert_eq!(manifest.chunks.len(), expected_chunks);

        // Read back and verify - streaming reader should use bounded memory
        let mut stream_reader = store.get_stream(&manifest_hash).expect("get").expect("exists");
        let mut verify_pos = 0usize;
        let mut buf = [0u8; 8192]; // Small read buffer

        loop {
            let n = std::io::Read::read(&mut stream_reader, &mut buf).expect("read");
            if n == 0 {
                break;
            }
            // Verify data pattern
            for (i, &byte) in buf[..n].iter().enumerate() {
                assert_eq!(
                    byte,
                    ((verify_pos + i) % 256) as u8,
                    "mismatch at position {}",
                    verify_pos + i
                );
            }
            verify_pos += n;
        }

        assert_eq!(verify_pos, total_size);
        assert_eq!(stream_reader.bytes_read(), total_size as u64);
    }

    #[test]
    fn test_chunked_content_independent_chunks() {
        // Verify that each chunk is independently content-addressed
        let (_dir, store) = create_temp_store();

        let chunk_size = 100;

        // Create two files with shared prefix (first chunk identical)
        let data1: Vec<u8> = (0..250).map(|i| (i % 256) as u8).collect();
        let data2: Vec<u8> = (0..250).map(|i| if i < 100 { (i % 256) as u8 } else { 0 }).collect();

        let manifest1_hash = store
            .put_stream_with_chunk_size(&mut &data1[..], chunk_size)
            .expect("put_stream 1");
        let manifest2_hash = store
            .put_stream_with_chunk_size(&mut &data2[..], chunk_size)
            .expect("put_stream 2");

        let manifest1 = store.get_manifest(&manifest1_hash).expect("get").expect("exists");
        let manifest2 = store.get_manifest(&manifest2_hash).expect("get").expect("exists");

        // First chunks should be identical (same content)
        assert_eq!(manifest1.chunks[0], manifest2.chunks[0], "first chunk should be shared");

        // Subsequent chunks should differ
        assert_ne!(manifest1.chunks[1], manifest2.chunks[1], "second chunk should differ");

        // Manifests should differ
        assert_ne!(manifest1_hash, manifest2_hash);
    }

    #[test]
    fn test_empty_stream() {
        let (_dir, store) = create_temp_store();

        let data: &[u8] = &[];
        let manifest_hash = store.put_stream(&mut &data[..]).expect("put_stream");

        let manifest = store.get_manifest(&manifest_hash).expect("get").expect("exists");
        assert_eq!(manifest.total_size, 0);
        assert_eq!(manifest.chunks.len(), 0);

        // Read back empty content
        let mut reader = store.get_stream(&manifest_hash).expect("get").expect("exists");
        let mut result = Vec::new();
        std::io::Read::read_to_end(&mut reader, &mut result).expect("read");
        assert!(result.is_empty());
    }

    // ── ContentStoreGC ─────────────────────────────────────────────────────

    #[test]
    fn test_gc_marks_roots() {
        let (_dir, store) = create_temp_store();

        let h1 = store.put(b"root1").expect("put");
        let h2 = store.put(b"root2").expect("put");
        let _h3 = store.put(b"orphan").expect("put");

        let mut gc = ContentStoreGC::new(&store, vec![h1, h2]);
        let marked = gc.mark_from_roots().expect("mark");

        assert_eq!(marked, 2);
        assert!(gc.marked().contains(&h1));
        assert!(gc.marked().contains(&h2));
    }

    #[test]
    fn test_gc_marks_tree_children() {
        let (_dir, store) = create_temp_store();

        // Create blobs
        let blob1 = store.put(b"blob1").expect("put");
        let blob2 = store.put(b"blob2").expect("put");

        // Create tree referencing blobs
        let tree = ContentTree::from_entries(vec![
            TreeEntry::blob("a.txt", blob1),
            TreeEntry::blob("b.txt", blob2),
        ]);
        let tree_hash = tree.store(&store).expect("store tree");

        // Only mark tree as root, children should be discovered
        let mut gc = ContentStoreGC::new(&store, vec![tree_hash]);
        let marked = gc.mark_from_roots().expect("mark");

        assert_eq!(marked, 3); // tree + 2 blobs
        assert!(gc.marked().contains(&tree_hash));
        assert!(gc.marked().contains(&blob1));
        assert!(gc.marked().contains(&blob2));
    }

    #[test]
    fn test_gc_deletes_orphans() {
        let (_dir, store) = create_temp_store();

        let root = store.put(b"keep me").expect("put");
        let orphan = store.put(b"delete me").expect("put");

        assert!(store.has(&root));
        assert!(store.has(&orphan));

        let mut gc = ContentStoreGC::new(&store, vec![root]);
        let result = gc.run().expect("gc");

        assert_eq!(result.marked_count, 1);
        assert_eq!(result.orphan_count, 1);
        assert_eq!(result.deleted_count, 1);
        assert!(result.completed);

        assert!(store.has(&root));
        assert!(!store.has(&orphan));
    }

    #[test]
    fn test_gc_dry_run() {
        let (_dir, store) = create_temp_store();

        let root = store.put(b"keep me").expect("put");
        let orphan = store.put(b"would be deleted").expect("put");

        let config = GCConfig {
            delete_orphans: false,
            ..Default::default()
        };
        let mut gc = ContentStoreGC::with_config(&store, vec![root], config);
        let result = gc.run().expect("gc");

        assert_eq!(result.orphan_count, 1);
        assert_eq!(result.deleted_count, 0);

        // Orphan should still exist in dry run
        assert!(store.has(&orphan));
    }

    #[test]
    fn test_gc_marks_chunked_content() {
        let (_dir, store) = create_temp_store();

        // Create chunked content (multiple chunks + manifest)
        let chunk_size = 1024;
        let data: Vec<u8> = (0..3000).map(|i| (i % 256) as u8).collect();
        let manifest_hash = store
            .put_stream_with_chunk_size(&mut &data[..], chunk_size)
            .expect("put_stream");

        // Create orphan
        let orphan = store.put(b"orphan").expect("put");

        // Verify both exist before GC
        assert!(store.has(&manifest_hash));
        assert!(store.has(&orphan));

        let mut gc = ContentStoreGC::new(&store, vec![manifest_hash]);
        let result = gc.run().expect("gc");

        // Should mark manifest + chunks (at least 2)
        assert!(result.marked_count >= 2, "should mark manifest and chunks");
        // Should find and delete the orphan
        assert_eq!(result.orphan_count, 1);
        assert_eq!(result.deleted_count, 1);

        // Verify manifest still exists, orphan deleted
        assert!(store.has(&manifest_hash));
        assert!(!store.has(&orphan));

        // Verify we can still read the chunked content
        let mut reader = store.get_stream(&manifest_hash).expect("get").expect("exists");
        let mut retrieved = Vec::new();
        std::io::Read::read_to_end(&mut reader, &mut retrieved).expect("read");
        assert_eq!(retrieved, data);
    }

    #[test]
    fn test_gc_empty_store() {
        let (_dir, store) = create_temp_store();

        let mut gc = ContentStoreGC::new(&store, vec![]);
        let result = gc.run().expect("gc");

        assert_eq!(result.marked_count, 0);
        assert_eq!(result.orphan_count, 0);
        assert_eq!(result.deleted_count, 0);
        assert!(result.completed);
    }

    // ── ContentTree ────────────────────────────────────────────────────────

    #[test]
    fn test_content_tree_empty() {
        let tree = ContentTree::new();
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
    }

    #[test]
    fn test_content_tree_from_entries() {
        let h1 = ContentHash::from_bytes(b"file1");
        let h2 = ContentHash::from_bytes(b"file2");
        let entries = vec![
            TreeEntry::blob("b.txt", h2),
            TreeEntry::blob("a.txt", h1),
        ];
        let tree = ContentTree::from_entries(entries);
        assert_eq!(tree.len(), 2);
        // Should be sorted by name
        assert_eq!(tree.entries()[0].name, "a.txt");
        assert_eq!(tree.entries()[1].name, "b.txt");
    }

    #[test]
    fn test_content_tree_get() {
        let h1 = ContentHash::from_bytes(b"file1");
        let tree = ContentTree::from_entries(vec![TreeEntry::blob("test.txt", h1)]);
        assert!(tree.get("test.txt").is_some());
        assert!(tree.get("missing.txt").is_none());
    }

    #[test]
    fn test_content_tree_with_entry() {
        let h1 = ContentHash::from_bytes(b"file1");
        let h2 = ContentHash::from_bytes(b"file2");

        let tree1 = ContentTree::new();
        let tree2 = tree1.with_entry(TreeEntry::blob("a.txt", h1));
        let tree3 = tree2.with_entry(TreeEntry::blob("b.txt", h2));

        // Original tree unchanged
        assert!(tree1.is_empty());
        // New trees have entries
        assert_eq!(tree2.len(), 1);
        assert_eq!(tree3.len(), 2);
    }

    #[test]
    fn test_content_tree_with_entry_replace() {
        let h1 = ContentHash::from_bytes(b"v1");
        let h2 = ContentHash::from_bytes(b"v2");

        let tree1 = ContentTree::from_entries(vec![TreeEntry::blob("file.txt", h1)]);
        let tree2 = tree1.with_entry(TreeEntry::blob("file.txt", h2));

        assert_eq!(tree1.len(), 1);
        assert_eq!(tree2.len(), 1);
        assert_eq!(tree1.get("file.txt").unwrap().hash, h1);
        assert_eq!(tree2.get("file.txt").unwrap().hash, h2);
    }

    #[test]
    fn test_content_tree_without_entry() {
        let h1 = ContentHash::from_bytes(b"file1");
        let h2 = ContentHash::from_bytes(b"file2");

        let tree1 = ContentTree::from_entries(vec![
            TreeEntry::blob("a.txt", h1),
            TreeEntry::blob("b.txt", h2),
        ]);
        let tree2 = tree1.without_entry("a.txt");

        assert_eq!(tree1.len(), 2);
        assert_eq!(tree2.len(), 1);
        assert!(tree2.get("a.txt").is_none());
        assert!(tree2.get("b.txt").is_some());
    }

    #[test]
    fn test_content_tree_hash() {
        let h1 = ContentHash::from_bytes(b"file1");
        let mut tree1 = ContentTree::from_entries(vec![TreeEntry::blob("a.txt", h1)]);
        let mut tree2 = ContentTree::from_entries(vec![TreeEntry::blob("a.txt", h1)]);

        // Same entries → same hash
        assert_eq!(tree1.hash(), tree2.hash());

        // Different entries → different hash
        let h2 = ContentHash::from_bytes(b"file2");
        let mut tree3 = ContentTree::from_entries(vec![TreeEntry::blob("b.txt", h2)]);
        assert_ne!(tree1.hash(), tree3.hash());
    }

    #[test]
    fn test_content_tree_store_load() {
        let (_dir, store) = create_temp_store();
        let h1 = ContentHash::from_bytes(b"file1");
        let h2 = ContentHash::from_bytes(b"file2");

        let tree = ContentTree::from_entries(vec![
            TreeEntry::blob("a.txt", h1),
            TreeEntry::tree("subdir", h2),
        ]);

        let tree_hash = tree.store(&store).expect("store tree");
        let loaded = ContentTree::load(&store, &tree_hash)
            .expect("load tree")
            .expect("tree should exist");

        assert_eq!(tree, loaded);
    }

    #[test]
    fn test_content_tree_diff_added() {
        let h1 = ContentHash::from_bytes(b"file1");
        let tree1 = ContentTree::new();
        let tree2 = ContentTree::from_entries(vec![TreeEntry::blob("new.txt", h1)]);

        let diffs = tree1.diff(&tree2);
        assert_eq!(diffs.len(), 1);
        assert!(matches!(&diffs[0], TreeDiffEntry::Added(e) if e.name == "new.txt"));
    }

    #[test]
    fn test_content_tree_diff_deleted() {
        let h1 = ContentHash::from_bytes(b"file1");
        let tree1 = ContentTree::from_entries(vec![TreeEntry::blob("old.txt", h1)]);
        let tree2 = ContentTree::new();

        let diffs = tree1.diff(&tree2);
        assert_eq!(diffs.len(), 1);
        assert!(matches!(&diffs[0], TreeDiffEntry::Deleted(e) if e.name == "old.txt"));
    }

    #[test]
    fn test_content_tree_diff_modified() {
        let h1 = ContentHash::from_bytes(b"v1");
        let h2 = ContentHash::from_bytes(b"v2");
        let tree1 = ContentTree::from_entries(vec![TreeEntry::blob("file.txt", h1)]);
        let tree2 = ContentTree::from_entries(vec![TreeEntry::blob("file.txt", h2)]);

        let diffs = tree1.diff(&tree2);
        assert_eq!(diffs.len(), 1);
        assert!(matches!(&diffs[0], TreeDiffEntry::Modified { old, new }
            if old.hash == h1 && new.hash == h2));
    }

    #[test]
    fn test_content_tree_diff_no_changes() {
        let h1 = ContentHash::from_bytes(b"file1");
        let tree1 = ContentTree::from_entries(vec![TreeEntry::blob("a.txt", h1)]);
        let tree2 = ContentTree::from_entries(vec![TreeEntry::blob("a.txt", h1)]);

        let diffs = tree1.diff(&tree2);
        assert!(diffs.is_empty());
    }

    #[test]
    fn test_content_tree_structural_sharing() {
        let h1 = ContentHash::from_bytes(b"unchanged");
        let h2 = ContentHash::from_bytes(b"changed_v1");
        let h3 = ContentHash::from_bytes(b"changed_v2");

        let tree1 = ContentTree::from_entries(vec![
            TreeEntry::blob("unchanged.txt", h1),
            TreeEntry::blob("changed.txt", h2),
        ]);
        let tree2 = tree1.with_entry(TreeEntry::blob("changed.txt", h3));

        // The unchanged entry should have the same hash in both trees
        assert_eq!(
            tree1.get("unchanged.txt").unwrap().hash,
            tree2.get("unchanged.txt").unwrap().hash
        );
    }

    #[test]
    fn test_content_tree_material_and_shader_types() {
        let mat_hash = ContentHash::from_bytes(b"material_pbr");
        let shader_hash = ContentHash::from_bytes(b"shader_vertex");
        let blob_hash = ContentHash::from_bytes(b"texture_data");

        let tree = ContentTree::from_entries(vec![
            TreeEntry::material("pbr.mat", mat_hash),
            TreeEntry::shader("main.wgsl", shader_hash),
            TreeEntry::blob("albedo.png", blob_hash),
        ]);

        assert_eq!(tree.len(), 3);

        let mat = tree.get("pbr.mat").unwrap();
        assert_eq!(mat.entry_type, TreeEntryType::Material);
        assert_eq!(mat.hash, mat_hash);

        let shader = tree.get("main.wgsl").unwrap();
        assert_eq!(shader.entry_type, TreeEntryType::Shader);
        assert_eq!(shader.hash, shader_hash);

        let blob = tree.get("albedo.png").unwrap();
        assert_eq!(blob.entry_type, TreeEntryType::Blob);
    }

    #[test]
    fn test_content_tree_material_shader_roundtrip() {
        let (_dir, store) = create_temp_store();

        let mat_hash = ContentHash::from_bytes(b"material_data");
        let shader_hash = ContentHash::from_bytes(b"shader_source");

        let tree = ContentTree::from_entries(vec![
            TreeEntry::material("default.mat", mat_hash),
            TreeEntry::shader("pbr.wgsl", shader_hash),
        ]);

        // Store and reload
        let tree_hash = tree.store(&store).expect("store tree");
        let loaded = ContentTree::load(&store, &tree_hash)
            .expect("load tree")
            .expect("tree should exist");

        // Verify types are preserved
        assert_eq!(loaded.get("default.mat").unwrap().entry_type, TreeEntryType::Material);
        assert_eq!(loaded.get("pbr.wgsl").unwrap().entry_type, TreeEntryType::Shader);
        assert_eq!(tree, loaded);
    }

    #[test]
    fn test_content_tree_identical_children_same_hash() {
        // Two trees with identical children produce the same hash
        let h1 = ContentHash::from_bytes(b"file1");
        let h2 = ContentHash::from_bytes(b"file2");
        let h3 = ContentHash::from_bytes(b"shader");

        let mut tree1 = ContentTree::from_entries(vec![
            TreeEntry::blob("a.txt", h1),
            TreeEntry::blob("b.txt", h2),
            TreeEntry::shader("main.wgsl", h3),
        ]);

        let mut tree2 = ContentTree::from_entries(vec![
            TreeEntry::blob("a.txt", h1),
            TreeEntry::blob("b.txt", h2),
            TreeEntry::shader("main.wgsl", h3),
        ]);

        // Same entries -> same hash
        assert_eq!(tree1.hash(), tree2.hash());

        // Different order in constructor still produces same hash (entries are sorted)
        let mut tree3 = ContentTree::from_entries(vec![
            TreeEntry::shader("main.wgsl", h3),
            TreeEntry::blob("b.txt", h2),
            TreeEntry::blob("a.txt", h1),
        ]);

        assert_eq!(tree1.hash(), tree3.hash());
    }

    #[test]
    fn test_content_tree_structural_sharing_verified() {
        // Verify that shared subtrees have matching hashes
        let (_dir, store) = create_temp_store();

        let shared_blob = store.put(b"shared content").expect("put");
        let unique_blob1 = store.put(b"unique v1").expect("put");
        let unique_blob2 = store.put(b"unique v2").expect("put");

        // Two trees sharing the same child entry
        let tree1 = ContentTree::from_entries(vec![
            TreeEntry::blob("shared.txt", shared_blob),
            TreeEntry::blob("unique.txt", unique_blob1),
        ]);

        let tree2 = ContentTree::from_entries(vec![
            TreeEntry::blob("shared.txt", shared_blob),
            TreeEntry::blob("unique.txt", unique_blob2),
        ]);

        // Store both trees
        let hash1 = tree1.store(&store).expect("store");
        let hash2 = tree2.store(&store).expect("store");

        // Trees have different hashes (different content)
        assert_ne!(hash1, hash2);

        // But shared entry hash is the same in both
        assert_eq!(
            tree1.get("shared.txt").unwrap().hash,
            tree2.get("shared.txt").unwrap().hash
        );

        // The shared blob only exists once in the store (deduplication)
        let all_hashes = store.list().expect("list");
        let shared_count = all_hashes.iter().filter(|h| **h == shared_blob).count();
        assert_eq!(shared_count, 1, "shared blob should only be stored once");
    }

    // ── ProvenanceChain ────────────────────────────────────────────────────

    #[test]
    fn test_provenance_chain_new() {
        let chain = ProvenanceChain::new(PruningStrategy::default());
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
        assert!(chain.origin().is_none());
        assert!(chain.current().is_none());
    }

    #[test]
    fn test_provenance_chain_with_origin() {
        let hash = ContentHash::from_bytes(b"origin");
        let origin = ProvenanceEntry::origin(hash, 1000, Some("Initial commit".into()));
        let chain = ProvenanceChain::with_origin(origin.clone(), PruningStrategy::default());

        assert!(!chain.is_empty());
        assert_eq!(chain.len(), 1);
        assert_eq!(chain.origin(), Some(&origin));
        assert_eq!(chain.current(), Some(&origin)); // Origin is also current when only 1 entry
    }

    #[test]
    fn test_provenance_chain_push() {
        let h1 = ContentHash::from_bytes(b"v1");
        let h2 = ContentHash::from_bytes(b"v2");
        let h3 = ContentHash::from_bytes(b"v3");

        let origin = ProvenanceEntry::origin(h1, 1000, Some("v1".into()));
        let mut chain = ProvenanceChain::with_origin(origin.clone(), PruningStrategy::KeepLastN(10));

        let entry2 = ProvenanceEntry::with_parent(h2, 2000, Some("v2".into()), h1);
        let entry3 = ProvenanceEntry::with_parent(h3, 3000, Some("v3".into()), h2);

        chain.push(entry2.clone());
        assert_eq!(chain.len(), 2);
        assert_eq!(chain.origin().unwrap().hash, h1);
        assert_eq!(chain.current().unwrap().hash, h2);

        chain.push(entry3.clone());
        assert_eq!(chain.len(), 3);
        assert_eq!(chain.origin().unwrap().hash, h1);
        assert_eq!(chain.current().unwrap().hash, h3);
    }

    #[test]
    fn test_provenance_chain_prune_keep_last_n() {
        // KeepLastN(3) means: keep origin + last 2 entries
        let strategy = PruningStrategy::KeepLastN(3);

        let h1 = ContentHash::from_bytes(b"v1");
        let h2 = ContentHash::from_bytes(b"v2");
        let h3 = ContentHash::from_bytes(b"v3");
        let h4 = ContentHash::from_bytes(b"v4");
        let h5 = ContentHash::from_bytes(b"v5");

        let origin = ProvenanceEntry::origin(h1, 1000, None);
        let mut chain = ProvenanceChain::with_origin(origin, strategy);

        chain.push(ProvenanceEntry::with_parent(h2, 2000, None, h1));
        chain.push(ProvenanceEntry::with_parent(h3, 3000, None, h2));
        // At this point: [h1, h2, h3] - exactly 3 entries, no pruning yet
        assert_eq!(chain.len(), 3);

        chain.push(ProvenanceEntry::with_parent(h4, 4000, None, h3));
        // Now exceeds limit: should prune to [h1, h3, h4]
        assert_eq!(chain.len(), 3);
        assert_eq!(chain.origin().unwrap().hash, h1); // origin preserved
        assert_eq!(chain.current().unwrap().hash, h4); // current preserved
        // Middle should be h3 (last entry before current)
        assert_eq!(chain.entries()[1].hash, h3);

        chain.push(ProvenanceEntry::with_parent(h5, 5000, None, h4));
        // Should prune to [h1, h4, h5]
        assert_eq!(chain.len(), 3);
        assert_eq!(chain.origin().unwrap().hash, h1);
        assert_eq!(chain.current().unwrap().hash, h5);
        assert_eq!(chain.entries()[1].hash, h4);
    }

    #[test]
    fn test_provenance_chain_prune_preserves_origin() {
        // Even with aggressive pruning (KeepLastN(2)), origin is always preserved
        let strategy = PruningStrategy::KeepLastN(2);

        let h1 = ContentHash::from_bytes(b"origin");
        let h2 = ContentHash::from_bytes(b"v2");
        let h3 = ContentHash::from_bytes(b"v3");
        let h4 = ContentHash::from_bytes(b"v4");

        let origin = ProvenanceEntry::origin(h1, 1000, Some("origin".into()));
        let mut chain = ProvenanceChain::with_origin(origin, strategy);

        chain.push(ProvenanceEntry::with_parent(h2, 2000, None, h1));
        chain.push(ProvenanceEntry::with_parent(h3, 3000, None, h2));
        chain.push(ProvenanceEntry::with_parent(h4, 4000, None, h3));

        // Should always have exactly 2 entries: origin + current
        assert_eq!(chain.len(), 2);
        assert_eq!(chain.origin().unwrap().hash, h1);
        assert_eq!(chain.origin().unwrap().message, Some("origin".into()));
        assert_eq!(chain.current().unwrap().hash, h4);
    }

    #[test]
    fn test_provenance_chain_prune_preserves_current() {
        let strategy = PruningStrategy::KeepLastN(2);

        let h1 = ContentHash::from_bytes(b"v1");
        let h2 = ContentHash::from_bytes(b"v2");

        let origin = ProvenanceEntry::origin(h1, 1000, None);
        let mut chain = ProvenanceChain::with_origin(origin, strategy);

        let entry2 = ProvenanceEntry::with_parent(h2, 2000, Some("current".into()), h1);
        chain.push(entry2);

        assert_eq!(chain.len(), 2);
        assert_eq!(chain.current().unwrap().hash, h2);
        assert_eq!(chain.current().unwrap().message, Some("current".into()));
    }

    #[test]
    fn test_provenance_chain_max_age() {
        // MaxAge(100): keep entries newer than (now - 100) seconds
        let strategy = PruningStrategy::MaxAge(100);

        let h1 = ContentHash::from_bytes(b"origin");
        let h2 = ContentHash::from_bytes(b"old");
        let h3 = ContentHash::from_bytes(b"recent");
        let h4 = ContentHash::from_bytes(b"current");

        let origin = ProvenanceEntry::origin(h1, 1000, None);
        let mut chain = ProvenanceChain::with_origin(origin, strategy);

        // Add an old entry (timestamp 1050, more than 100 seconds before "now")
        chain.push(ProvenanceEntry::with_parent(h2, 1050, None, h1));
        // Add a recent entry (timestamp 1180, within 100 seconds of "now")
        chain.push(ProvenanceEntry::with_parent(h3, 1180, None, h2));
        // Add current entry (timestamp 1200 = "now")
        chain.push(ProvenanceEntry::with_parent(h4, 1200, None, h3));

        // Cutoff = 1200 - 100 = 1100
        // h2 (1050) is older than cutoff, should be pruned
        // h3 (1180) is newer than cutoff, should be kept
        // h1 (origin) is always kept
        // h4 (current) is always kept
        assert_eq!(chain.len(), 3);
        assert_eq!(chain.origin().unwrap().hash, h1);
        assert_eq!(chain.entries()[1].hash, h3);
        assert_eq!(chain.current().unwrap().hash, h4);
    }

    #[test]
    fn test_provenance_entry_origin() {
        let hash = ContentHash::from_bytes(b"test");
        let entry = ProvenanceEntry::origin(hash, 12345, Some("Initial".into()));

        assert_eq!(entry.hash, hash);
        assert_eq!(entry.timestamp, 12345);
        assert_eq!(entry.message, Some("Initial".into()));
        assert_eq!(entry.parent, None);
    }

    #[test]
    fn test_provenance_entry_with_parent() {
        let hash = ContentHash::from_bytes(b"child");
        let parent_hash = ContentHash::from_bytes(b"parent");
        let entry = ProvenanceEntry::with_parent(hash, 12345, None, parent_hash);

        assert_eq!(entry.hash, hash);
        assert_eq!(entry.timestamp, 12345);
        assert_eq!(entry.message, None);
        assert_eq!(entry.parent, Some(parent_hash));
    }

    #[test]
    fn test_pruning_strategy_default() {
        let strategy = PruningStrategy::default();
        assert!(matches!(strategy, PruningStrategy::KeepLastN(10)));
    }

    #[test]
    fn test_provenance_chain_empty_behavior() {
        // Empty chain should handle all operations gracefully
        let chain = ProvenanceChain::new(PruningStrategy::KeepLastN(3));

        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
        assert!(chain.origin().is_none());
        assert!(chain.current().is_none());
        assert!(chain.entries().is_empty());
    }

    #[test]
    fn test_provenance_chain_single_entry_no_pruning() {
        // Single entry (origin only) should never be pruned
        let h1 = ContentHash::from_bytes(b"origin");
        let origin = ProvenanceEntry::origin(h1, 1000, Some("genesis".into()));
        let chain = ProvenanceChain::with_origin(origin.clone(), PruningStrategy::KeepLastN(1));

        // Origin is both origin and current
        assert_eq!(chain.len(), 1);
        assert_eq!(chain.origin().unwrap().hash, h1);
        assert_eq!(chain.current().unwrap().hash, h1);
        assert!(chain.origin().unwrap().parent.is_none());
    }

    #[test]
    fn test_provenance_chain_two_entries_no_pruning() {
        // Two entry chain (origin + current) should never be pruned, even with KeepLastN(1)
        let h1 = ContentHash::from_bytes(b"origin");
        let h2 = ContentHash::from_bytes(b"current");

        let origin = ProvenanceEntry::origin(h1, 1000, None);
        let mut chain = ProvenanceChain::with_origin(origin, PruningStrategy::KeepLastN(1));

        chain.push(ProvenanceEntry::with_parent(h2, 2000, None, h1));

        // Both entries must be preserved (origin + current)
        assert_eq!(chain.len(), 2);
        assert_eq!(chain.origin().unwrap().hash, h1);
        assert_eq!(chain.current().unwrap().hash, h2);
    }

    #[test]
    fn test_provenance_chain_exactly_at_n_entries() {
        // Chain exactly at N entries should not trigger pruning
        let strategy = PruningStrategy::KeepLastN(4);

        let h1 = ContentHash::from_bytes(b"v1");
        let h2 = ContentHash::from_bytes(b"v2");
        let h3 = ContentHash::from_bytes(b"v3");
        let h4 = ContentHash::from_bytes(b"v4");

        let origin = ProvenanceEntry::origin(h1, 1000, None);
        let mut chain = ProvenanceChain::with_origin(origin, strategy);

        chain.push(ProvenanceEntry::with_parent(h2, 2000, None, h1));
        chain.push(ProvenanceEntry::with_parent(h3, 3000, None, h2));
        chain.push(ProvenanceEntry::with_parent(h4, 4000, None, h3));

        // Exactly 4 entries, KeepLastN(4), no pruning
        assert_eq!(chain.len(), 4);
        assert_eq!(chain.entries()[0].hash, h1);
        assert_eq!(chain.entries()[1].hash, h2);
        assert_eq!(chain.entries()[2].hash, h3);
        assert_eq!(chain.entries()[3].hash, h4);
    }

    #[test]
    fn test_provenance_chain_combined_strategy_basic() {
        // Combined: keep last 3 AND anything newer than 100 seconds
        let strategy = PruningStrategy::Combined {
            keep_last_n: 3,
            max_age_secs: 100,
        };

        let h1 = ContentHash::from_bytes(b"origin");
        let h2 = ContentHash::from_bytes(b"old_1");
        let h3 = ContentHash::from_bytes(b"old_2");
        let h4 = ContentHash::from_bytes(b"recent");
        let h5 = ContentHash::from_bytes(b"current");

        let origin = ProvenanceEntry::origin(h1, 1000, None);
        let mut chain = ProvenanceChain::with_origin(origin, strategy);

        // h2: old (1050), outside max_age window from current (1200)
        chain.push(ProvenanceEntry::with_parent(h2, 1050, None, h1));
        // h3: old (1080), outside max_age window
        chain.push(ProvenanceEntry::with_parent(h3, 1080, None, h2));
        // h4: recent (1150), within max_age window (1200 - 100 = 1100 cutoff)
        chain.push(ProvenanceEntry::with_parent(h4, 1150, None, h3));
        // h5: current (1200)
        chain.push(ProvenanceEntry::with_parent(h5, 1200, None, h4));

        // Combined strategy should keep:
        // - h1 (origin, always preserved)
        // - h4 (within last 3-2=1 middle entries AND recent)
        // - h5 (current, always preserved)
        // h2 and h3 are old AND not in last N-2 middle entries
        assert_eq!(chain.len(), 3);
        assert_eq!(chain.origin().unwrap().hash, h1);
        assert_eq!(chain.entries()[1].hash, h4);
        assert_eq!(chain.current().unwrap().hash, h5);
    }

    #[test]
    fn test_provenance_chain_combined_strategy_recent_trumps_n() {
        // Combined: keep last 2, but recent entries should also be kept
        let strategy = PruningStrategy::Combined {
            keep_last_n: 2,
            max_age_secs: 500,
        };

        let h1 = ContentHash::from_bytes(b"origin");
        let h2 = ContentHash::from_bytes(b"recent_1");
        let h3 = ContentHash::from_bytes(b"recent_2");
        let h4 = ContentHash::from_bytes(b"current");

        let origin = ProvenanceEntry::origin(h1, 1000, None);
        let mut chain = ProvenanceChain::with_origin(origin, strategy);

        // All entries are within 500 seconds of current (1300)
        chain.push(ProvenanceEntry::with_parent(h2, 1100, None, h1));
        chain.push(ProvenanceEntry::with_parent(h3, 1200, None, h2));
        chain.push(ProvenanceEntry::with_parent(h4, 1300, None, h3));

        // Cutoff = 1300 - 500 = 800
        // All middle entries (h2, h3) are newer than 800, so both kept
        // Even though KeepLastN(2) would only keep origin+current
        assert_eq!(chain.len(), 4);
        assert_eq!(chain.origin().unwrap().hash, h1);
        assert_eq!(chain.entries()[1].hash, h2);
        assert_eq!(chain.entries()[2].hash, h3);
        assert_eq!(chain.current().unwrap().hash, h4);
    }

    #[test]
    fn test_provenance_chain_max_age_all_old_except_origin_current() {
        // MaxAge with all middle entries older than cutoff
        let strategy = PruningStrategy::MaxAge(10);

        let h1 = ContentHash::from_bytes(b"origin");
        let h2 = ContentHash::from_bytes(b"very_old");
        let h3 = ContentHash::from_bytes(b"also_old");
        let h4 = ContentHash::from_bytes(b"current");

        let origin = ProvenanceEntry::origin(h1, 100, None);
        let mut chain = ProvenanceChain::with_origin(origin, strategy);

        // h2 and h3 are much older than cutoff (1000 - 10 = 990)
        chain.push(ProvenanceEntry::with_parent(h2, 200, None, h1));
        chain.push(ProvenanceEntry::with_parent(h3, 300, None, h2));
        chain.push(ProvenanceEntry::with_parent(h4, 1000, None, h3));

        // All middle entries pruned, only origin and current remain
        assert_eq!(chain.len(), 2);
        assert_eq!(chain.origin().unwrap().hash, h1);
        assert_eq!(chain.current().unwrap().hash, h4);
    }

    #[test]
    fn test_provenance_chain_keep_last_n_zero_or_one() {
        // KeepLastN with n=0 or n=1 should be treated as minimum 2
        let strategy = PruningStrategy::KeepLastN(0);

        let h1 = ContentHash::from_bytes(b"v1");
        let h2 = ContentHash::from_bytes(b"v2");
        let h3 = ContentHash::from_bytes(b"v3");

        let origin = ProvenanceEntry::origin(h1, 1000, None);
        let mut chain = ProvenanceChain::with_origin(origin, strategy);

        chain.push(ProvenanceEntry::with_parent(h2, 2000, None, h1));
        chain.push(ProvenanceEntry::with_parent(h3, 3000, None, h2));

        // Should keep at least 2 (origin + current)
        assert_eq!(chain.len(), 2);
        assert_eq!(chain.origin().unwrap().hash, h1);
        assert_eq!(chain.current().unwrap().hash, h3);
    }

    #[test]
    fn test_provenance_entry_parent_chain_integrity() {
        // Verify parent links are correctly maintained
        let h1 = ContentHash::from_bytes(b"a");
        let h2 = ContentHash::from_bytes(b"b");
        let h3 = ContentHash::from_bytes(b"c");

        let e1 = ProvenanceEntry::origin(h1, 100, None);
        let e2 = ProvenanceEntry::with_parent(h2, 200, None, h1);
        let e3 = ProvenanceEntry::with_parent(h3, 300, None, h2);

        // Verify chain of parent links
        assert!(e1.parent.is_none());
        assert_eq!(e2.parent, Some(h1));
        assert_eq!(e3.parent, Some(h2));
    }

    // ── ShardedPipelineTable ───────────────────────────────────────────────

    #[test]
    fn test_sharded_pipeline_table_new() {
        let table = ShardedPipelineTable::new(16);
        assert!(table.is_empty());
        assert_eq!(table.len(), 0);
        assert_eq!(table.shard_stats().shard_count, 16);
    }

    #[test]
    fn test_sharded_pipeline_table_shard_count_power_of_two() {
        // Non-power-of-2 should be rounded up
        let table = ShardedPipelineTable::new(5);
        assert_eq!(table.shard_stats().shard_count, 8);

        let table = ShardedPipelineTable::new(17);
        assert_eq!(table.shard_stats().shard_count, 32);
    }

    #[test]
    fn test_sharded_pipeline_table_shard_index() {
        let table = ShardedPipelineTable::new(16);

        // IDs should distribute across shards
        assert_eq!(table.shard_index(0), 0);
        assert_eq!(table.shard_index(1), 1);
        assert_eq!(table.shard_index(15), 15);
        assert_eq!(table.shard_index(16), 0); // wraps
        assert_eq!(table.shard_index(17), 1);
    }

    #[test]
    fn test_sharded_pipeline_table_with_numa_node() {
        let table = ShardedPipelineTable::with_numa_node(8, 2);
        assert_eq!(table.numa_node, Some(2));
    }

    #[test]
    fn test_sharded_pipeline_table_contains_remove() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let table = ShardedPipelineTable::new(4);

        // Create a minimal pipeline
        let src = r#"
            @vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0,0.0,0.0,1.0); }
            @fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0,0.0,0.0,1.0); }
        "#;
        let hash = sha256(src.as_bytes());
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("test"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("test"),
            entries: &[],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("test"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let rp = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("test"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: "vs",
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: "fs",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let pipeline = CachedPipeline {
            id: 42,
            render_pipeline: rp,
            bind_group_layout: bgl,
            shader_hash: hash,
        };

        assert!(!table.contains(42));
        table.insert(pipeline);
        assert!(table.contains(42));
        assert_eq!(table.len(), 1);

        assert!(table.remove(42));
        assert!(!table.contains(42));
        assert_eq!(table.len(), 0);
        assert!(!table.remove(42)); // already removed
    }

    #[test]
    fn test_sharded_pipeline_table_shard_stats() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let table = ShardedPipelineTable::new(4);

        // Add pipelines with IDs that will distribute across shards
        for id in [0u32, 1, 2, 3, 4, 5, 6, 7] {
            let src = format!(
                "@vertex fn vs{}() -> @builtin(position) vec4<f32> {{ return vec4<f32>(0.0,0.0,0.0,1.0); }}
                 @fragment fn fs{}() -> @location(0) vec4<f32> {{ return vec4<f32>(1.0,0.0,0.0,1.0); }}",
                id, id
            );
            let hash = sha256(src.as_bytes());
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("test{}", id)),
                source: wgpu::ShaderSource::Wgsl(src.as_str().into()),
            });
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[],
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });
            let rp = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: &format!("vs{}", id),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: &format!("fs{}", id),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

            table.insert(CachedPipeline {
                id,
                render_pipeline: rp,
                bind_group_layout: bgl,
                shader_hash: hash,
            });
        }

        let stats = table.shard_stats();
        assert_eq!(stats.total_pipelines, 8);
        assert_eq!(stats.shard_count, 4);
        // With 8 pipelines in 4 shards, each shard should have 2
        assert_eq!(stats.min_shard_size, 2);
        assert_eq!(stats.max_shard_size, 2);
        assert!((stats.avg_shard_size - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_sharded_pipeline_table_with_pipeline() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let table = ShardedPipelineTable::new(4);
        let src = r#"
            @vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0,0.0,0.0,1.0); }
            @fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0,0.0,0.0,1.0); }
        "#;
        let hash = sha256(src.as_bytes());
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("test"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let rp = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: "vs",
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: "fs",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        table.insert(CachedPipeline {
            id: 99,
            render_pipeline: rp,
            bind_group_layout: bgl,
            shader_hash: hash,
        });

        // with_pipeline should find it and allow read access
        let found_id = table.with_pipeline(99, |p| p.id);
        assert_eq!(found_id, Some(99));

        // Missing pipeline returns None
        let missing = table.with_pipeline(999, |p| p.id);
        assert!(missing.is_none());
    }

    #[test]
    fn test_sharded_pipeline_table_clear() {
        let table = ShardedPipelineTable::new(4);
        // Just verify clear doesn't panic on empty table
        table.clear();
        assert!(table.is_empty());
    }

    // ── ShaderCache ────────────────────────────────────────────────────────

    #[test]
    fn test_shader_cache_new_is_empty() {
        let cache = ShaderCache::new();
        assert!(cache.modules.is_empty());
        assert!(cache.source_hashes.is_empty());
    }

    #[test]
    fn test_shader_cache_clear() {
        let mut cache = ShaderCache::new();
        // Clear on an empty cache should not panic.
        cache.clear();
        assert!(cache.modules.is_empty());
        assert!(cache.source_hashes.is_empty());
    }

    #[test]
    fn test_shader_cache_get_or_compile_dedup() {
        // Requires a GPU device.
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        ));
        let Some(adapter) = adapter else {
            eprintln!("Skipping test_shader_cache_get_or_compile_dedup: no GPU adapter");
            return;
        };
        let (device, _queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("test device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .expect("device creation");

        let mut cache = ShaderCache::new();
        let src = r#"
            @vertex fn vs() -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }
            @fragment fn fs() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0, 0.0, 0.0, 1.0);
            }
        "#;

        let (module_a, hash_a) = cache.get_or_compile(&device, src);
        let (module_b, hash_b) = cache.get_or_compile(&device, src);

        // Same source -> same hash.
        assert_eq!(hash_a, hash_b);
        // Only one entry in the cache.
        assert_eq!(cache.modules.len(), 1);

        // Verify the modules compile (do not panic).
        let _ = module_a;
        let _ = module_b;
    }

    #[test]
    fn test_shader_cache_different_sources_different_hashes() {
        let cache = ShaderCache::new();
        let src_a = "one";
        let src_b = "two";

        let hash_a = sha256(src_a.as_bytes());
        let hash_b = sha256(src_b.as_bytes());

        assert_ne!(hash_a, hash_b);

        // This test doesn't need a device -- it verifies hashing only.
        let _ = cache;
    }

    // ── PipelineTable ──────────────────────────────────────────────────────

    #[test]
    fn test_pipeline_table_new_is_empty() {
        let table = PipelineTable::new();
        assert!(table.is_empty());
        assert_eq!(table.len(), 0);
        assert!(table.pipelines.is_empty());
    }

    /// Helper: obtain a (device, queue) pair, skipping the test if no GPU
    /// is available.
    fn create_test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        ));
        let adapter = adapter?;
        Some(
            pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("test device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            ))
            .expect("device creation"),
        )
    }

    #[test]
    fn test_pipeline_table_insert_get_remove_roundtrip() {
        // This test requires a GPU device.
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        // Create an empty bind-group layout (valid even without a shader).
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("test BGL"),
            entries: &[],
        });

        // Minimal valid WGSL shader.
        let src = r#"
            @vertex fn vs() -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }
            @fragment fn fs() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0, 0.0, 0.0, 1.0);
            }
        "#;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("test module"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let hash = sha256(src.as_bytes());

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("test layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let rp = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("test pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: "vs",
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: "fs",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        let pipeline = CachedPipeline {
            id: 42,
            render_pipeline: rp,
            bind_group_layout: bgl,
            shader_hash: hash,
        };

        let mut table = PipelineTable::new();
        assert!(table.is_empty());

        // Insert.
        table.insert(42, pipeline);
        assert_eq!(table.len(), 1);
        assert!(!table.is_empty());

        // Get.
        let fetched = table.get(42).expect("pipeline should exist");
        assert_eq!(fetched.id, 42);
        assert_eq!(fetched.shader_hash, sha256(src.as_bytes()));

        // Get nonexistent.
        assert!(table.get(99).is_none());

        // Remove.
        assert!(table.remove(42));
        assert_eq!(table.len(), 0);
        assert!(table.is_empty());

        // Remove again should return false.
        assert!(!table.remove(42));

        // Remove nonexistent.
        assert!(!table.remove(99));
    }

    #[test]
    fn test_pipeline_table_compile_pipeline() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        // Minimal valid WGSL with explicit entry points.
        let src = r#"
            @vertex fn vs_main() -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }
            @fragment fn fs_main() -> @location(0) vec4<f32> {
                return vec4<f32>(0.5, 0.6, 0.7, 1.0);
            }
        "#;

        let mut table = PipelineTable::new();

        let result = table.compile_pipeline(
            &device,
            1,
            src,
            "vs_main",
            "fs_main",
            &[], // no vertex buffers
            wgpu::TextureFormat::Rgba8Unorm,
        );

        assert!(result.is_ok(), "compile_pipeline should succeed: {:?}", result);
        assert_eq!(result.unwrap(), 1);
        assert_eq!(table.len(), 1);

        let pipeline = table.get(1).expect("pipeline 1 should exist");
        assert_eq!(pipeline.id, 1);
        assert_eq!(pipeline.shader_hash, sha256(src.as_bytes()));
    }

    #[test]
    fn test_pipeline_table_multiple_pipelines() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let src_a = r#"
            @vertex fn vs_a() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0,0.0,0.0,1.0); }
            @fragment fn fs_a() -> @location(0) vec4<f32> { return vec4<f32>(1.0,0.0,0.0,1.0); }
        "#;
        let src_b = r#"
            @vertex fn vs_b() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0,0.0,0.0,1.0); }
            @fragment fn fs_b() -> @location(0) vec4<f32> { return vec4<f32>(0.0,1.0,0.0,1.0); }
        "#;

        let mut table = PipelineTable::new();

        let r1 = table.compile_pipeline(
            &device, 10, src_a, "vs_a", "fs_a", &[], wgpu::TextureFormat::Rgba8Unorm,
        );
        let r2 = table.compile_pipeline(
            &device, 20, src_b, "vs_b", "fs_b", &[], wgpu::TextureFormat::Rgba8Unorm,
        );

        assert!(r1.is_ok());
        assert!(r2.is_ok());
        assert_eq!(table.len(), 2);

        // Pipelines have different shader hashes.
        let p10 = table.get(10).unwrap();
        let p20 = table.get(20).unwrap();
        assert_ne!(p10.shader_hash, p20.shader_hash);

        // Remove one.
        assert!(table.remove(10));
        assert_eq!(table.len(), 1);
        assert!(table.get(10).is_none());
        assert!(table.get(20).is_some());
    }

    #[test]
    fn test_pipeline_table_insert_overwrites() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let src = r#"
            @vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0,0.0,0.0,1.0); }
            @fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0,0.0,0.0,1.0); }
        "#;

        let mut table = PipelineTable::new();
        let _ = table.compile_pipeline(&device, 1, src, "vs", "fs", &[], wgpu::TextureFormat::Rgba8Unorm);
        assert_eq!(table.len(), 1);

        // Compile again with same id -- should overwrite without changing count.
        let _ = table.compile_pipeline(&device, 1, src, "vs", "fs", &[], wgpu::TextureFormat::Rgba8Unorm);
        assert_eq!(table.len(), 1);
        assert!(table.get(1).is_some());
    }

    #[test]
    fn test_pipeline_table_shared_shader_cache() {
        // Two pipelines with the same WGSL source should share a shader module.
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let src = r#"
            @vertex fn shared_vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0,0.0,0.0,1.0); }
            @fragment fn shared_fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0,0.0,0.0,1.0); }
        "#;

        let mut table = PipelineTable::new();

        let _ = table.compile_pipeline(&device, 1, src, "shared_vs", "shared_fs", &[], wgpu::TextureFormat::Rgba8Unorm);
        let _ = table.compile_pipeline(&device, 2, src, "shared_vs", "shared_fs", &[], wgpu::TextureFormat::Rgba8Unorm);

        // Only one shader module should be in the cache.
        assert_eq!(table.shader_cache.modules.len(), 1);

        // Both pipelines should have the same shader hash.
        assert_eq!(table.get(1).unwrap().shader_hash, table.get(2).unwrap().shader_hash);
    }

    // ── ContentDiffer / BinaryDiffer / TreeDiffer ──────────────────────────

    #[test]
    fn test_binary_differ_identical() {
        let differ = BinaryDiffer::new();
        let data = b"identical content here";

        let delta = differ.diff(data, data).expect("diff should succeed");

        // Identical inputs should produce a single Copy instruction
        match &delta {
            Delta::BinaryPatch { patch } => {
                // Copy instruction: 0x01 + offset(4) + len(4) + end(1) = 10 bytes
                assert_eq!(patch.len(), 10, "patch should be Copy + end marker");
                assert_eq!(patch[0], 0x01, "should start with Copy op");
            }
            _ => panic!("expected BinaryPatch, got {:?}", delta),
        }

        // Apply should return original
        let result = differ.apply(data, &delta).expect("apply should succeed");
        assert_eq!(result, data);
    }

    #[test]
    fn test_binary_differ_small_change() {
        let differ = BinaryDiffer::new();
        let old = b"hello world, this is a test of the differ";
        let mut new = old.to_vec();
        new[6] = b'W'; // "hello World, ..."

        let delta = differ.diff(old, &new).expect("diff should succeed");

        // Patch should be smaller than full content
        match &delta {
            Delta::BinaryPatch { patch } => {
                assert!(patch.len() < new.len(), "patch should be smaller than full content");
            }
            Delta::Full(_) => {
                // Small changes might produce Full if no good matches found
            }
            _ => panic!("expected BinaryPatch or Full, got {:?}", delta),
        }

        // Apply should produce new content
        let result = differ.apply(old, &delta).expect("apply should succeed");
        assert_eq!(result, new);
    }

    #[test]
    fn test_binary_differ_roundtrip() {
        let differ = BinaryDiffer::new();

        // Test with various content patterns
        let test_cases: Vec<(&[u8], &[u8])> = vec![
            (b"abc", b"def"),
            (b"", b"new content"),
            (b"old content", b""),
            (b"prefix_SAME_suffix", b"prefix_DIFF_suffix"),
            (
                b"repeated repeated repeated pattern",
                b"repeated modified repeated pattern",
            ),
        ];

        for (old, new) in test_cases {
            let delta = differ.diff(old, new).expect("diff should succeed");
            let result = differ.apply(old, &delta).expect("apply should succeed");
            assert_eq!(result, new, "roundtrip failed for old={:?}, new={:?}", old, new);
        }
    }

    #[test]
    fn test_binary_differ_compression_ratio() {
        let differ = BinaryDiffer::new();

        // Create 4KB of similar content with small differences
        let mut old = vec![0u8; 4096];
        for i in 0..4096 {
            old[i] = ((i * 7) % 256) as u8;
        }

        let mut new = old.clone();
        // Change ~10% of bytes scattered throughout
        for i in (0..4096).step_by(40) {
            new[i] = new[i].wrapping_add(1);
        }

        let delta = differ.diff(&old, &new).expect("diff should succeed");

        // Patch should be significantly smaller than full content
        let patch_size = delta.size_bytes();
        let full_size = new.len();

        // Target: patch < 50% of full size for similar inputs
        assert!(
            patch_size < full_size / 2,
            "patch size {} should be < 50% of full size {}, got {}%",
            patch_size,
            full_size,
            (patch_size * 100) / full_size
        );

        // Verify roundtrip
        let result = differ.apply(&old, &delta).expect("apply should succeed");
        assert_eq!(result, new);
    }

    #[test]
    fn test_binary_differ_empty_inputs() {
        let differ = BinaryDiffer::new();

        // Empty to empty
        let delta = differ.diff(b"", b"").expect("diff should succeed");
        let result = differ.apply(b"", &delta).expect("apply should succeed");
        assert!(result.is_empty());

        // Empty to content
        let delta = differ.diff(b"", b"new").expect("diff should succeed");
        let result = differ.apply(b"", &delta).expect("apply should succeed");
        assert_eq!(result, b"new");

        // Content to empty
        let delta = differ.diff(b"old", b"").expect("diff should succeed");
        let result = differ.apply(b"old", &delta).expect("apply should succeed");
        assert!(result.is_empty());
    }

    #[test]
    fn test_tree_differ_added() {
        let differ = TreeDiffer::new();

        let old_tree = ContentTree::new();
        let h1 = ContentHash::from_bytes(b"new_file");
        let new_tree = ContentTree::from_entries(vec![TreeEntry::blob("added.txt", h1)]);

        let delta = differ.diff_trees(&old_tree, &new_tree);

        match &delta {
            Delta::TreeDiff { changes } => {
                assert_eq!(changes.len(), 1);
                assert!(matches!(&changes[0], TreeDiffEntry::Added(e) if e.name == "added.txt"));
            }
            _ => panic!("expected TreeDiff, got {:?}", delta),
        }

        // Apply should produce new tree
        let result = differ.apply_to_tree(&old_tree, &delta).expect("apply should succeed");
        assert_eq!(result, new_tree);
    }

    #[test]
    fn test_tree_differ_removed() {
        let differ = TreeDiffer::new();

        let h1 = ContentHash::from_bytes(b"to_delete");
        let old_tree = ContentTree::from_entries(vec![TreeEntry::blob("deleted.txt", h1)]);
        let new_tree = ContentTree::new();

        let delta = differ.diff_trees(&old_tree, &new_tree);

        match &delta {
            Delta::TreeDiff { changes } => {
                assert_eq!(changes.len(), 1);
                assert!(matches!(&changes[0], TreeDiffEntry::Deleted(e) if e.name == "deleted.txt"));
            }
            _ => panic!("expected TreeDiff, got {:?}", delta),
        }

        // Apply should produce empty tree
        let result = differ.apply_to_tree(&old_tree, &delta).expect("apply should succeed");
        assert!(result.is_empty());
    }

    #[test]
    fn test_tree_differ_modified() {
        let differ = TreeDiffer::new();

        let h1 = ContentHash::from_bytes(b"version1");
        let h2 = ContentHash::from_bytes(b"version2");
        let old_tree = ContentTree::from_entries(vec![TreeEntry::blob("file.txt", h1)]);
        let new_tree = ContentTree::from_entries(vec![TreeEntry::blob("file.txt", h2)]);

        let delta = differ.diff_trees(&old_tree, &new_tree);

        match &delta {
            Delta::TreeDiff { changes } => {
                assert_eq!(changes.len(), 1);
                match &changes[0] {
                    TreeDiffEntry::Modified { old, new } => {
                        assert_eq!(old.hash, h1);
                        assert_eq!(new.hash, h2);
                    }
                    _ => panic!("expected Modified, got {:?}", changes[0]),
                }
            }
            _ => panic!("expected TreeDiff, got {:?}", delta),
        }

        // Apply should produce new tree
        let result = differ.apply_to_tree(&old_tree, &delta).expect("apply should succeed");
        assert_eq!(result.get("file.txt").unwrap().hash, h2);
    }

    #[test]
    fn test_tree_differ_serialized_roundtrip() {
        let differ = TreeDiffer::new();

        let h1 = ContentHash::from_bytes(b"old");
        let h2 = ContentHash::from_bytes(b"new");
        let old_tree = ContentTree::from_entries(vec![
            TreeEntry::blob("keep.txt", h1),
            TreeEntry::blob("change.txt", h1),
        ]);
        let new_tree = ContentTree::from_entries(vec![
            TreeEntry::blob("keep.txt", h1),
            TreeEntry::blob("change.txt", h2),
            TreeEntry::blob("added.txt", h2),
        ]);

        // Use the ContentDiffer trait methods with serialized data
        let old_bytes = old_tree.serialize();
        let new_bytes = new_tree.serialize();

        let delta = differ.diff(&old_bytes, &new_bytes).expect("diff should succeed");
        let result_bytes = differ.apply(&old_bytes, &delta).expect("apply should succeed");

        let result_tree = ContentTree::deserialize(&result_bytes).expect("deserialize should succeed");
        assert_eq!(result_tree, new_tree);
    }

    #[test]
    fn test_delta_size_bytes() {
        let full = Delta::Full(vec![1, 2, 3, 4, 5]);
        assert_eq!(full.size_bytes(), 5);

        let patch = Delta::BinaryPatch { patch: vec![0; 10] };
        assert_eq!(patch.size_bytes(), 10);

        let param = Delta::ParameterPatch {
            offsets: vec![(0, vec![1, 2]), (10, vec![3, 4, 5])],
        };
        // 8 + 2 + 8 + 3 = 21
        assert_eq!(param.size_bytes(), 21);
    }

    #[test]
    fn test_delta_is_empty() {
        assert!(Delta::Full(vec![]).is_empty());
        assert!(!Delta::Full(vec![1]).is_empty());

        assert!(Delta::BinaryPatch { patch: vec![] }.is_empty());
        assert!(!Delta::BinaryPatch { patch: vec![0] }.is_empty());

        assert!(Delta::TreeDiff { changes: vec![] }.is_empty());

        assert!(Delta::ParameterPatch { offsets: vec![] }.is_empty());
    }

    #[test]
    fn test_diff_error_display() {
        let err = DiffError::InvalidPatch;
        assert!(err.to_string().contains("invalid"));

        let err = DiffError::SizeMismatch { expected: 100, actual: 50 };
        assert!(err.to_string().contains("100"));
        assert!(err.to_string().contains("50"));

        let err = DiffError::IoError("test error".to_string());
        assert!(err.to_string().contains("test error"));

        let err = DiffError::TypeMismatch { expected: "A", actual: "B" };
        assert!(err.to_string().contains("A"));
        assert!(err.to_string().contains("B"));
    }

    #[test]
    fn test_binary_differ_invalid_patch() {
        let differ = BinaryDiffer::new();

        // Invalid op type
        let bad_patch = Delta::BinaryPatch { patch: vec![0xFF] };
        let result = differ.apply(b"old", &bad_patch);
        assert!(matches!(result, Err(DiffError::InvalidPatch)));

        // Truncated copy instruction
        let bad_patch = Delta::BinaryPatch { patch: vec![0x01, 0, 0] };
        let result = differ.apply(b"old", &bad_patch);
        assert!(matches!(result, Err(DiffError::InvalidPatch)));

        // Truncated insert instruction
        let bad_patch = Delta::BinaryPatch { patch: vec![0x02, 10, 0, 0, 0] };
        let result = differ.apply(b"old", &bad_patch);
        assert!(matches!(result, Err(DiffError::InvalidPatch)));
    }

    #[test]
    fn test_binary_differ_out_of_bounds_copy() {
        let differ = BinaryDiffer::new();

        // Construct a valid-looking patch that copies beyond old content bounds
        let mut patch = vec![0x01]; // Copy op
        patch.extend_from_slice(&100u32.to_le_bytes()); // offset = 100
        patch.extend_from_slice(&50u32.to_le_bytes());  // len = 50
        patch.push(0x00); // end marker

        let bad_delta = Delta::BinaryPatch { patch };
        let result = differ.apply(b"short", &bad_delta);
        assert!(matches!(result, Err(DiffError::SizeMismatch { .. })));
    }

    #[test]
    fn test_content_differ_type_mismatch() {
        let binary_differ = BinaryDiffer::new();
        let tree_differ = TreeDiffer::new();

        // Try to apply TreeDiff with BinaryDiffer
        let tree_delta = Delta::TreeDiff { changes: vec![] };
        let result = binary_differ.apply(b"old", &tree_delta);
        assert!(matches!(result, Err(DiffError::TypeMismatch { .. })));

        // Try to apply BinaryPatch with TreeDiffer
        let h1 = ContentHash::from_bytes(b"test");
        let tree = ContentTree::from_entries(vec![TreeEntry::blob("test.txt", h1)]);
        let binary_delta = Delta::BinaryPatch { patch: vec![0x00] };
        let result = tree_differ.apply_to_tree(&tree, &binary_delta);
        assert!(matches!(result, Err(DiffError::TypeMismatch { .. })));
    }

    #[test]
    fn test_binary_differ_large_insert() {
        let differ = BinaryDiffer::new();

        // When old is empty, everything becomes an insert
        let new_data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        let delta = differ.diff(b"", &new_data).expect("diff should succeed");

        let result = differ.apply(b"", &delta).expect("apply should succeed");
        assert_eq!(result, new_data);
    }

    #[test]
    fn test_binary_differ_with_custom_settings() {
        let differ = BinaryDiffer::with_settings(4, 8);

        let old = b"AAAABBBBCCCCDDDD";
        let new = b"AAAABBBBEEEE";

        let delta = differ.diff(old, new).expect("diff should succeed");
        let result = differ.apply(old, &delta).expect("apply should succeed");
        assert_eq!(result, new);
    }

    // ── Additional whitebox edge case tests for T-MAT-7.2 ──────────────────

    #[test]
    fn test_binary_differ_single_byte() {
        let differ = BinaryDiffer::new();

        // Single byte input
        let delta = differ.diff(b"A", b"B").expect("diff should succeed");
        let result = differ.apply(b"A", &delta).expect("apply should succeed");
        assert_eq!(result, b"B");

        // Single byte to empty
        let delta = differ.diff(b"X", b"").expect("diff should succeed");
        let result = differ.apply(b"X", &delta).expect("apply should succeed");
        assert!(result.is_empty());

        // Empty to single byte
        let delta = differ.diff(b"", b"Y").expect("diff should succeed");
        let result = differ.apply(b"", &delta).expect("apply should succeed");
        assert_eq!(result, b"Y");
    }

    #[test]
    fn test_binary_differ_smaller_than_block_size() {
        let differ = BinaryDiffer::new(); // default block_size = 16

        // Input smaller than block_size (16)
        let old = b"short";
        let new = b"shorter";

        let delta = differ.diff(old, new).expect("diff should succeed");
        let result = differ.apply(old, &delta).expect("apply should succeed");
        assert_eq!(result, new);

        // Edge case: exactly block_size - 1
        let old_15 = b"0123456789abcde"; // 15 bytes
        let new_15 = b"0123456789abcdX";

        let delta = differ.diff(old_15, new_15).expect("diff should succeed");
        let result = differ.apply(old_15, &delta).expect("apply should succeed");
        assert_eq!(result, new_15);
    }

    #[test]
    fn test_binary_differ_exactly_block_size() {
        let differ = BinaryDiffer::new(); // block_size = 16

        // Exactly 16 bytes
        let old = b"0123456789abcdef";
        let new = b"0123456789abcdeX";

        let delta = differ.diff(old, new).expect("diff should succeed");
        let result = differ.apply(old, &delta).expect("apply should succeed");
        assert_eq!(result, new);
    }

    #[test]
    fn test_binary_differ_all_different_bytes() {
        let differ = BinaryDiffer::new();

        // All bytes are different - should produce Full delta (no matches)
        let old: Vec<u8> = (0..100).map(|i| i as u8).collect();
        let new: Vec<u8> = (100..200).map(|i| i as u8).collect();

        let delta = differ.diff(&old, &new).expect("diff should succeed");

        // When no good matches found, might be BinaryPatch or Full
        let result = differ.apply(&old, &delta).expect("apply should succeed");
        assert_eq!(result, new);
    }

    #[test]
    fn test_binary_differ_hash_collision_handling() {
        let differ = BinaryDiffer::with_settings(4, 4);

        // Create data that could have hash collisions (repetitive patterns)
        let old = b"AAAAAAAAAAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBBBBBBBBB";
        let new = b"AAAAAAAAAAAAAAAAAAAAAAAA_MODIFIED_BBBBBBBBBBBBBBBBBB";

        let delta = differ.diff(old, new).expect("diff should succeed");
        let result = differ.apply(old, &delta).expect("apply should succeed");
        assert_eq!(result, new);
    }

    #[test]
    fn test_binary_differ_copy_instruction_boundary() {
        let differ = BinaryDiffer::new();

        // Test copy at offset 0 and at max offset
        let old: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        let mut new = old.clone();
        // Modify middle, keep start and end identical
        new[500] = 255;

        let delta = differ.diff(&old, &new).expect("diff should succeed");
        let result = differ.apply(&old, &delta).expect("apply should succeed");
        assert_eq!(result, new);
    }

    #[test]
    fn test_binary_differ_only_end_marker() {
        let differ = BinaryDiffer::new();

        // A patch with just an end marker should produce empty output
        let patch = Delta::BinaryPatch { patch: vec![0x00] };
        let result = differ.apply(b"anything", &patch).expect("apply should succeed");
        assert!(result.is_empty());
    }

    #[test]
    fn test_binary_differ_consecutive_inserts() {
        let differ = BinaryDiffer::new();

        // Completely new content smaller than block_size
        let old = b"AAAA";
        let new = b"BBBBCCCC";

        let delta = differ.diff(old, new).expect("diff should succeed");
        let result = differ.apply(old, &delta).expect("apply should succeed");
        assert_eq!(result, new);
    }

    #[test]
    fn test_tree_differ_empty_trees() {
        let differ = TreeDiffer::new();

        let empty1 = ContentTree::new();
        let empty2 = ContentTree::new();

        let delta = differ.diff_trees(&empty1, &empty2);
        assert!(delta.is_empty());

        let result = differ.apply_to_tree(&empty1, &delta).expect("apply should succeed");
        assert!(result.is_empty());
    }

    #[test]
    fn test_tree_differ_multiple_changes() {
        let differ = TreeDiffer::new();

        let h1 = ContentHash::from_bytes(b"file1");
        let h2 = ContentHash::from_bytes(b"file2");
        let h3 = ContentHash::from_bytes(b"file3");
        let h4 = ContentHash::from_bytes(b"file4");

        let old_tree = ContentTree::from_entries(vec![
            TreeEntry::blob("keep.txt", h1),
            TreeEntry::blob("delete.txt", h2),
            TreeEntry::blob("modify.txt", h3),
        ]);

        let new_tree = ContentTree::from_entries(vec![
            TreeEntry::blob("keep.txt", h1),
            TreeEntry::blob("modify.txt", h4),   // modified
            TreeEntry::blob("add.txt", h2),      // added
        ]);

        let delta = differ.diff_trees(&old_tree, &new_tree);

        match &delta {
            Delta::TreeDiff { changes } => {
                assert_eq!(changes.len(), 3); // delete + modify + add
            }
            _ => panic!("expected TreeDiff"),
        }

        let result = differ.apply_to_tree(&old_tree, &delta).expect("apply should succeed");
        assert_eq!(result, new_tree);
    }

    #[test]
    fn test_tree_differ_malformed_serialized_data() {
        let differ = TreeDiffer::new();

        // Invalid serialized data
        let bad_data = b"not a valid serialized tree";
        let result = differ.diff(bad_data, bad_data);

        // Should return IoError with deserialization message
        assert!(matches!(result, Err(DiffError::IoError(_))));
    }

    #[test]
    fn test_delta_tree_diff_size_estimation() {
        // TreeDiff uses rough estimate of 100 bytes per change
        let h1 = ContentHash::from_bytes(b"test");
        let changes = vec![
            TreeDiffEntry::Added(TreeEntry::blob("a.txt", h1)),
            TreeDiffEntry::Added(TreeEntry::blob("b.txt", h1)),
        ];

        let delta = Delta::TreeDiff { changes };
        // 2 changes * 100 bytes estimate = 200
        assert_eq!(delta.size_bytes(), 200);
    }

    #[test]
    fn test_binary_differ_insert_length_overflow_check() {
        let differ = BinaryDiffer::new();

        // Construct an insert instruction with length claiming more data than exists
        let mut patch = vec![0x02]; // Insert op
        patch.extend_from_slice(&100u32.to_le_bytes()); // len = 100
        patch.extend_from_slice(b"short"); // only 5 bytes of actual data
        patch.push(0x00); // end marker

        let bad_delta = Delta::BinaryPatch { patch };
        let result = differ.apply(b"old", &bad_delta);
        assert!(matches!(result, Err(DiffError::InvalidPatch)));
    }

    #[test]
    fn test_binary_differ_parameter_patch_rejected() {
        let differ = BinaryDiffer::new();

        // BinaryDiffer should reject ParameterPatch delta type
        let param_delta = Delta::ParameterPatch {
            offsets: vec![(0, vec![1, 2, 3])],
        };
        let result = differ.apply(b"old", &param_delta);
        assert!(matches!(result, Err(DiffError::TypeMismatch { .. })));
    }

    #[test]
    fn test_diff_error_trait_impls() {
        // Verify DiffError implements Error trait
        let err: Box<dyn std::error::Error> = Box::new(DiffError::InvalidPatch);
        assert!(!err.to_string().is_empty());

        // Verify PartialEq
        assert_eq!(DiffError::InvalidPatch, DiffError::InvalidPatch);
        assert_ne!(
            DiffError::InvalidPatch,
            DiffError::SizeMismatch { expected: 1, actual: 2 }
        );
    }

    #[test]
    fn test_binary_differ_min_match_len_boundary() {
        // Test with min_match_len = 4 (the minimum enforced)
        let differ = BinaryDiffer::with_settings(4, 4);

        // Create pattern where match is exactly min_match_len
        let old = b"XXXX____XXXX";
        let new = b"XXXX++++XXXX";

        let delta = differ.diff(old, new).expect("diff should succeed");
        let result = differ.apply(old, &delta).expect("apply should succeed");
        assert_eq!(result, new);
    }

    #[test]
    fn test_content_differ_with_large_matching_prefix() {
        let differ = BinaryDiffer::new();

        // Large shared prefix, small difference at end
        let prefix: Vec<u8> = (0..500).map(|i| (i % 256) as u8).collect();
        let mut old = prefix.clone();
        old.extend_from_slice(b"OLD_SUFFIX");
        let mut new = prefix;
        new.extend_from_slice(b"NEW_SUFFIX");

        let delta = differ.diff(&old, &new).expect("diff should succeed");

        // Patch should be significantly smaller than full content
        let patch_size = delta.size_bytes();
        assert!(
            patch_size < new.len() / 2,
            "patch size {} should be < half of content size {}",
            patch_size,
            new.len()
        );

        let result = differ.apply(&old, &delta).expect("apply should succeed");
        assert_eq!(result, new);
    }

    // ── DeltaSyncProtocol ──────────────────────────────────────────────────

    #[test]
    fn test_delta_sync_negotiate_version() {
        let protocol = DeltaSyncProtocol::new();

        // Negotiate with same version
        assert_eq!(protocol.negotiate(1), 1);

        // Negotiate with higher version (should use our version)
        assert_eq!(protocol.negotiate(5), 1);

        // Negotiate with lower version (should use their version)
        assert_eq!(protocol.negotiate(0), 0);
    }

    #[test]
    fn test_delta_sync_compute_batch_identical() {
        let protocol = DeltaSyncProtocol::new();

        let hash1 = ContentHash::from_bytes(b"content1");
        let hash2 = ContentHash::from_bytes(b"content2");

        let items = vec![
            SyncItem::new("file1.txt", hash1.clone(), 100),
            SyncItem::new("file2.txt", hash2.clone(), 200),
        ];

        // Content provider
        let get_content = |_hash: &ContentHash| -> Option<Vec<u8>> { Some(vec![1, 2, 3]) };

        // Same items on both sides -> empty batch
        let batch = protocol
            .compute_batch(&items, &items, get_content)
            .expect("compute should succeed");

        assert!(batch.is_empty());
        assert_eq!(batch.operations.len(), 0);
    }

    #[test]
    fn test_delta_sync_compute_batch_additions() {
        let protocol = DeltaSyncProtocol::new();

        let hash1 = ContentHash::from_bytes(b"content1");
        let hash2 = ContentHash::from_bytes(b"content2");
        let hash3 = ContentHash::from_bytes(b"content3");

        let local_items = vec![SyncItem::new("file1.txt", hash1.clone(), 100)];

        let remote_items = vec![
            SyncItem::new("file1.txt", hash1.clone(), 100),
            SyncItem::new("file2.txt", hash2.clone(), 200),
            SyncItem::new("file3.txt", hash3.clone(), 300),
        ];

        let content_map: HashMap<ContentHash, Vec<u8>> = [
            (hash1.clone(), b"content1".to_vec()),
            (hash2.clone(), b"content2".to_vec()),
            (hash3.clone(), b"content3".to_vec()),
        ]
        .into_iter()
        .collect();

        let get_content = |hash: &ContentHash| -> Option<Vec<u8>> { content_map.get(hash).cloned() };

        let batch = protocol
            .compute_batch(&local_items, &remote_items, get_content)
            .expect("compute should succeed");

        assert!(!batch.is_empty());

        // Should have 2 upserts for new files
        let upserts: Vec<_> = batch
            .operations
            .iter()
            .filter(|op| matches!(op, SyncOperation::Upsert { .. }))
            .collect();
        assert_eq!(upserts.len(), 2);
    }

    #[test]
    fn test_delta_sync_compute_batch_removals() {
        let protocol = DeltaSyncProtocol::new();

        let hash1 = ContentHash::from_bytes(b"content1");
        let hash2 = ContentHash::from_bytes(b"content2");
        let hash3 = ContentHash::from_bytes(b"content3");

        let local_items = vec![
            SyncItem::new("file1.txt", hash1.clone(), 100),
            SyncItem::new("file2.txt", hash2.clone(), 200),
            SyncItem::new("file3.txt", hash3.clone(), 300),
        ];

        let remote_items = vec![SyncItem::new("file1.txt", hash1.clone(), 100)];

        let get_content = |_hash: &ContentHash| -> Option<Vec<u8>> { Some(vec![1, 2, 3]) };

        let batch = protocol
            .compute_batch(&local_items, &remote_items, get_content)
            .expect("compute should succeed");

        // Should have 2 removes
        let removes: Vec<_> = batch
            .operations
            .iter()
            .filter(|op| matches!(op, SyncOperation::Remove { .. }))
            .collect();
        assert_eq!(removes.len(), 2);
    }

    #[test]
    fn test_delta_sync_compute_batch_updates() {
        let protocol = DeltaSyncProtocol::new();

        let hash1 = ContentHash::from_bytes(b"content1_old");
        let hash2 = ContentHash::from_bytes(b"content1_new");

        let local_items = vec![SyncItem::new("file1.txt", hash1.clone(), 100)];
        let remote_items = vec![SyncItem::new("file1.txt", hash2.clone(), 110)];

        let content_map: HashMap<ContentHash, Vec<u8>> = [
            (hash1.clone(), b"content1_old_data_here".to_vec()),
            (hash2.clone(), b"content1_new_data_here".to_vec()),
        ]
        .into_iter()
        .collect();

        let get_content = |hash: &ContentHash| -> Option<Vec<u8>> { content_map.get(hash).cloned() };

        let batch = protocol
            .compute_batch(&local_items, &remote_items, get_content)
            .expect("compute should succeed");

        // Should have 1 upsert for the updated file
        assert_eq!(batch.operations.len(), 1);
        assert!(matches!(&batch.operations[0], SyncOperation::Upsert { path, .. } if path == "file1.txt"));
    }

    #[test]
    fn test_delta_sync_apply_batch() {
        let protocol = DeltaSyncProtocol::new();

        let hash1 = ContentHash::from_bytes(b"new_content");
        let new_content = b"new_content".to_vec();

        let batch = SyncBatch::new(
            ContentHash::zero(),
            ContentHash::from_bytes(b"target"),
            vec![SyncOperation::Upsert {
                path: "file1.txt".into(),
                delta: Delta::Full(new_content.clone()),
                target_hash: hash1.clone(),
            }],
            1,
        );

        let get_content = |_hash: &ContentHash| -> Option<Vec<u8>> { None };
        let put_content = |data: &[u8]| -> ContentHash { ContentHash::from_bytes(data) };

        let result = protocol
            .apply_batch(&batch, get_content, put_content)
            .expect("apply should succeed");

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].path, "file1.txt");
        assert_eq!(result[0].hash, hash1);
    }

    #[test]
    fn test_delta_sync_compress_decompress() {
        let protocol = DeltaSyncProtocol::new();

        let hash1 = ContentHash::from_bytes(b"content1");
        let hash2 = ContentHash::from_bytes(b"content2");

        let batch = SyncBatch::new(
            ContentHash::from_bytes(b"base"),
            ContentHash::from_bytes(b"target"),
            vec![
                SyncOperation::Upsert {
                    path: "file1.txt".into(),
                    delta: Delta::Full(b"some content here".to_vec()),
                    target_hash: hash1,
                },
                SyncOperation::Remove {
                    path: "deleted.txt".into(),
                },
                SyncOperation::Upsert {
                    path: "file2.txt".into(),
                    delta: Delta::Full(b"more content".to_vec()),
                    target_hash: hash2,
                },
            ],
            1,
        );

        // Compress
        let compressed = protocol.compress_batch(&batch);
        assert!(!compressed.is_empty());

        // Decompress
        let decompressed = protocol
            .decompress_batch(&compressed)
            .expect("decompress should succeed");

        assert_eq!(decompressed.protocol_version, batch.protocol_version);
        assert_eq!(decompressed.base_checkpoint, batch.base_checkpoint);
        assert_eq!(decompressed.target_checkpoint, batch.target_checkpoint);
        assert_eq!(decompressed.operations.len(), batch.operations.len());
    }

    #[test]
    fn test_delta_sync_rle_compression_efficiency() {
        let protocol = DeltaSyncProtocol::new();

        // Data with many repeated bytes should compress well
        let repeated_data: Vec<u8> = vec![0xAA; 1000];
        let hash = ContentHash::from_bytes(&repeated_data);

        let batch = SyncBatch::new(
            ContentHash::zero(),
            ContentHash::from_bytes(b"target"),
            vec![SyncOperation::Upsert {
                path: "repeated.bin".into(),
                delta: Delta::Full(repeated_data.clone()),
                target_hash: hash,
            }],
            1,
        );

        let compressed = protocol.compress_batch(&batch);

        // Compressed should be much smaller than uncompressed
        // (1000 bytes of 0xAA compresses to ~4 bytes with RLE)
        assert!(
            compressed.len() < 200,
            "compressed size {} should be < 200 for 1000 repeated bytes",
            compressed.len()
        );

        // Verify roundtrip
        let decompressed = protocol
            .decompress_batch(&compressed)
            .expect("decompress should succeed");

        match &decompressed.operations[0] {
            SyncOperation::Upsert { delta: Delta::Full(data), .. } => {
                assert_eq!(data.len(), 1000);
                assert!(data.iter().all(|&b| b == 0xAA));
            }
            _ => panic!("expected Upsert with Full delta"),
        }
    }

    #[test]
    fn test_delta_sync_1000_items_performance() {
        use std::time::Instant;

        let protocol = DeltaSyncProtocol::new();

        // Generate 1000 items
        let items: Vec<SyncItem> = (0..1000)
            .map(|i| {
                let path = format!("assets/item_{:04}.bin", i);
                let content = format!("content_{}", i);
                let hash = ContentHash::from_bytes(content.as_bytes());
                SyncItem::new(path, hash, content.len() as u64)
            })
            .collect();

        // Content storage
        let content_store: HashMap<ContentHash, Vec<u8>> = items
            .iter()
            .map(|item| {
                let content = format!("content_data_for_{}", item.path);
                (item.hash.clone(), content.into_bytes())
            })
            .collect();

        let _get_content = |hash: &ContentHash| -> Option<Vec<u8>> { content_store.get(hash).cloned() };

        // Modify 10% of items (100 items)
        let mut modified_items = items.clone();
        for i in (0..1000).step_by(10) {
            let content = format!("modified_content_{}", i);
            modified_items[i].hash = ContentHash::from_bytes(content.as_bytes());
        }

        // Add modified content to store
        let mut full_store = content_store.clone();
        for i in (0..1000).step_by(10) {
            let content = format!("modified_content_data_for_{}", items[i].path);
            full_store.insert(modified_items[i].hash.clone(), content.into_bytes());
        }

        let get_content_full =
            |hash: &ContentHash| -> Option<Vec<u8>> { full_store.get(hash).cloned() };

        let start = Instant::now();

        // Compute batch
        let batch = protocol
            .compute_batch(&items, &modified_items, get_content_full)
            .expect("compute should succeed");

        // Compress
        let compressed = protocol.compress_batch(&batch);

        // Decompress
        let _decompressed = protocol
            .decompress_batch(&compressed)
            .expect("decompress should succeed");

        let elapsed = start.elapsed();

        // Should complete in under 5 seconds (much faster in practice)
        assert!(
            elapsed.as_secs() < 5,
            "sync took {:?}, should be < 5 seconds",
            elapsed
        );

        // Verify we detected the right number of changes
        assert_eq!(batch.operations.len(), 100, "should have 100 updates");
    }

    #[test]
    fn test_delta_sync_checkpoint_from_items() {
        let hash1 = ContentHash::from_bytes(b"content1");
        let hash2 = ContentHash::from_bytes(b"content2");

        let items = vec![
            SyncItem::new("file1.txt", hash1, 100),
            SyncItem::new("file2.txt", hash2, 200),
        ];

        let checkpoint = SyncCheckpoint::from_items(&items, 12345);

        assert_eq!(checkpoint.timestamp, 12345);
        assert_eq!(checkpoint.item_count, 2);
        assert!(!checkpoint.id.is_zero());
    }

    #[test]
    fn test_delta_sync_empty_checkpoint() {
        let checkpoint = SyncCheckpoint::empty(999);

        assert!(checkpoint.id.is_zero());
        assert_eq!(checkpoint.timestamp, 999);
        assert_eq!(checkpoint.item_count, 0);
    }

    #[test]
    fn test_delta_sync_version_mismatch_error() {
        let protocol = DeltaSyncProtocol::new();

        // Create batch with future version
        let batch = SyncBatch::new(
            ContentHash::zero(),
            ContentHash::zero(),
            vec![],
            999, // Future version
        );

        let get_content = |_: &ContentHash| -> Option<Vec<u8>> { None };
        let put_content = |data: &[u8]| -> ContentHash { ContentHash::from_bytes(data) };

        let result = protocol.apply_batch(&batch, get_content, put_content);

        assert!(matches!(
            result,
            Err(SyncError::VersionMismatch { expected: 1, got: 999 })
        ));
    }

    #[test]
    fn test_delta_sync_without_compression() {
        let protocol = DeltaSyncProtocol::without_compression();

        let batch = SyncBatch::new(
            ContentHash::zero(),
            ContentHash::from_bytes(b"target"),
            vec![SyncOperation::Upsert {
                path: "file.txt".into(),
                delta: Delta::Full(b"test content".to_vec()),
                target_hash: ContentHash::from_bytes(b"test content"),
            }],
            1,
        );

        let serialized = protocol.compress_batch(&batch);
        let deserialized = protocol
            .decompress_batch(&serialized)
            .expect("decompress should succeed");

        assert_eq!(deserialized.operations.len(), 1);
    }

    #[test]
    fn test_sync_error_display() {
        let err1 = SyncError::VersionMismatch { expected: 1, got: 2 };
        assert!(err1.to_string().contains("version mismatch"));

        let err2 = SyncError::CheckpointMismatch;
        assert!(err2.to_string().contains("checkpoint"));

        let err3 = SyncError::MissingContent(ContentHash::zero());
        assert!(err3.to_string().contains("missing content"));

        let err4 = SyncError::DiffError(DiffError::InvalidPatch);
        assert!(err4.to_string().contains("diff error"));

        let err5 = SyncError::CompressionError("test".into());
        assert!(err5.to_string().contains("compression error"));
    }

    #[test]
    fn test_sync_batch_empty() {
        let batch = SyncBatch::empty(ContentHash::from_bytes(b"checkpoint"), 1);

        assert!(batch.is_empty());
        assert_eq!(batch.base_checkpoint, batch.target_checkpoint);
    }

    #[test]
    fn test_delta_sync_mixed_operations_ordering() {
        let protocol = DeltaSyncProtocol::new();

        let hash_a = ContentHash::from_bytes(b"a");
        let hash_b = ContentHash::from_bytes(b"b");
        let hash_c = ContentHash::from_bytes(b"c");

        // Local has b, c; remote has a, c (different)
        // Expected: upsert a, upsert c, remove b
        let local_items = vec![
            SyncItem::new("b.txt", hash_b.clone(), 10),
            SyncItem::new("c.txt", hash_c.clone(), 10),
        ];

        let new_hash_c = ContentHash::from_bytes(b"c_new");
        let remote_items = vec![
            SyncItem::new("a.txt", hash_a.clone(), 10),
            SyncItem::new("c.txt", new_hash_c.clone(), 10),
        ];

        let content_map: HashMap<ContentHash, Vec<u8>> = [
            (hash_a.clone(), b"a_content".to_vec()),
            (hash_b.clone(), b"b_content".to_vec()),
            (hash_c.clone(), b"c_content".to_vec()),
            (new_hash_c.clone(), b"c_new_content".to_vec()),
        ]
        .into_iter()
        .collect();

        let get_content = |hash: &ContentHash| -> Option<Vec<u8>> { content_map.get(hash).cloned() };

        let batch = protocol
            .compute_batch(&local_items, &remote_items, get_content)
            .expect("compute should succeed");

        // Should have 3 operations: 2 upserts + 1 remove
        assert_eq!(batch.operations.len(), 3);

        // Upserts should come before removes
        assert!(matches!(&batch.operations[0], SyncOperation::Upsert { .. }));
        assert!(matches!(&batch.operations[1], SyncOperation::Upsert { .. }));
        assert!(matches!(&batch.operations[2], SyncOperation::Remove { path } if path == "b.txt"));
    }

    // ── DeltaSyncProtocol Edge Cases ───────────────────────────────────────

    #[test]
    fn test_delta_sync_empty_batch_handling() {
        let protocol = DeltaSyncProtocol::new();

        // Empty local and remote
        let empty_items: Vec<SyncItem> = vec![];
        let get_content = |_: &ContentHash| -> Option<Vec<u8>> { None };

        let batch = protocol
            .compute_batch(&empty_items, &empty_items, get_content)
            .expect("compute should succeed");

        assert!(batch.is_empty());
        assert_eq!(batch.operations.len(), 0);

        // Verify compress/decompress of empty batch
        let compressed = protocol.compress_batch(&batch);
        let decompressed = protocol
            .decompress_batch(&compressed)
            .expect("decompress should succeed");

        assert!(decompressed.is_empty());
    }

    #[test]
    fn test_delta_sync_version_0_negotiation() {
        let protocol = DeltaSyncProtocol::new();

        // Negotiate with version 0 (peer has ancient protocol)
        let negotiated = protocol.negotiate(0);
        assert_eq!(negotiated, 0);

        // Version 0 batch should still be processable (backward compat)
        let batch = SyncBatch::new(
            ContentHash::zero(),
            ContentHash::zero(),
            vec![],
            0, // Version 0
        );

        let get_content = |_: &ContentHash| -> Option<Vec<u8>> { None };
        let put_content = |data: &[u8]| -> ContentHash { ContentHash::from_bytes(data) };

        // Should succeed because batch version (0) <= our version (1)
        let result = protocol.apply_batch(&batch, get_content, put_content);
        assert!(result.is_ok());
    }

    #[test]
    fn test_delta_sync_compression_disabled_mode() {
        let protocol = DeltaSyncProtocol::without_compression();

        // Verify compression is disabled
        assert!(!protocol.compress);

        // Data that would normally compress well
        let repeated_data: Vec<u8> = vec![0xAA; 500];
        let hash = ContentHash::from_bytes(&repeated_data);

        let batch = SyncBatch::new(
            ContentHash::zero(),
            ContentHash::from_bytes(b"target"),
            vec![SyncOperation::Upsert {
                path: "file.bin".into(),
                delta: Delta::Full(repeated_data.clone()),
                target_hash: hash,
            }],
            1,
        );

        let serialized = protocol.compress_batch(&batch);

        // Without compression, the serialized data should contain the full 500 bytes
        // (plus overhead for headers)
        assert!(
            serialized.len() >= 500,
            "uncompressed should be >= 500 bytes, got {}",
            serialized.len()
        );

        // Verify roundtrip still works
        let decompressed = protocol
            .decompress_batch(&serialized)
            .expect("decompress should succeed");

        match &decompressed.operations[0] {
            SyncOperation::Upsert { delta: Delta::Full(data), .. } => {
                assert_eq!(data.len(), 500);
            }
            _ => panic!("expected Upsert with Full delta"),
        }
    }

    #[test]
    fn test_delta_sync_large_batch_10k_items() {
        use std::time::Instant;

        let protocol = DeltaSyncProtocol::new();

        // Generate 10,000 items
        let items: Vec<SyncItem> = (0..10_000)
            .map(|i| {
                let path = format!("assets/large_batch/item_{:05}.bin", i);
                let content = format!("content_{}", i);
                let hash = ContentHash::from_bytes(content.as_bytes());
                SyncItem::new(path, hash, content.len() as u64)
            })
            .collect();

        // Content storage for all items
        let content_store: HashMap<ContentHash, Vec<u8>> = items
            .iter()
            .map(|item| {
                let content = format!("data_for_{}", item.path);
                (item.hash.clone(), content.into_bytes())
            })
            .collect();

        // Simulate 5% modifications (500 items changed)
        let mut modified_items = items.clone();
        let mut modified_store = content_store.clone();

        for i in (0..10_000).step_by(20) {
            let new_content = format!("modified_content_{}", i);
            modified_items[i].hash = ContentHash::from_bytes(new_content.as_bytes());
            modified_store.insert(modified_items[i].hash.clone(), new_content.into_bytes());
        }

        let get_content =
            |hash: &ContentHash| -> Option<Vec<u8>> { modified_store.get(hash).cloned() };

        let start = Instant::now();

        // Compute batch
        let batch = protocol
            .compute_batch(&items, &modified_items, get_content)
            .expect("compute should succeed");

        // Compress and decompress
        let compressed = protocol.compress_batch(&batch);
        let decompressed = protocol
            .decompress_batch(&compressed)
            .expect("decompress should succeed");

        let elapsed = start.elapsed();

        // Performance: should complete in reasonable time
        assert!(
            elapsed.as_secs() < 10,
            "10K item sync took {:?}, should be < 10 seconds",
            elapsed
        );

        // Verify correct number of operations (500 modified = 500 upserts)
        assert_eq!(batch.operations.len(), 500);
        assert_eq!(decompressed.operations.len(), 500);
    }

    #[test]
    fn test_delta_sync_rle_edge_cases() {
        // Test RLE with 0xFF bytes (escape sequence)
        let data_with_ff: Vec<u8> = vec![0xFF, 0xFF, 0xFF, 0xFF, 0xFF]; // 5 0xFF bytes

        let compressed = DeltaSyncProtocol::rle_compress(&data_with_ff);
        let decompressed =
            DeltaSyncProtocol::rle_decompress(&compressed).expect("decompress should succeed");

        assert_eq!(decompressed, data_with_ff);

        // Test alternating bytes (worst case for RLE)
        let alternating: Vec<u8> = (0..100).map(|i| if i % 2 == 0 { 0xAA } else { 0xBB }).collect();

        let compressed_alt = DeltaSyncProtocol::rle_compress(&alternating);
        let decompressed_alt =
            DeltaSyncProtocol::rle_decompress(&compressed_alt).expect("decompress should succeed");

        assert_eq!(decompressed_alt, alternating);

        // Test exactly 4 repeated bytes (threshold for RLE encoding)
        let exactly_four: Vec<u8> = vec![0xCC; 4];
        let compressed_four = DeltaSyncProtocol::rle_compress(&exactly_four);
        let decompressed_four =
            DeltaSyncProtocol::rle_decompress(&compressed_four).expect("decompress should succeed");

        assert_eq!(decompressed_four, exactly_four);

        // Test 3 repeated bytes (should not trigger RLE, stays literal)
        let three_bytes: Vec<u8> = vec![0xDD; 3];
        let compressed_three = DeltaSyncProtocol::rle_compress(&three_bytes);

        // 3 literal bytes should stay as 3 bytes (not RLE encoded)
        assert_eq!(compressed_three.len(), 3);

        let decompressed_three =
            DeltaSyncProtocol::rle_decompress(&compressed_three).expect("decompress should succeed");
        assert_eq!(decompressed_three, three_bytes);
    }

    #[test]
    fn test_delta_sync_rle_empty_data() {
        let empty: Vec<u8> = vec![];

        let compressed = DeltaSyncProtocol::rle_compress(&empty);
        assert!(compressed.is_empty());

        let decompressed =
            DeltaSyncProtocol::rle_decompress(&compressed).expect("decompress should succeed");
        assert!(decompressed.is_empty());
    }

    #[test]
    fn test_delta_sync_rle_max_run_length() {
        // Test run length at maximum (65535 bytes)
        let max_run: Vec<u8> = vec![0xAA; 65535];

        let compressed = DeltaSyncProtocol::rle_compress(&max_run);
        let decompressed =
            DeltaSyncProtocol::rle_decompress(&compressed).expect("decompress should succeed");

        assert_eq!(decompressed.len(), 65535);
        assert!(decompressed.iter().all(|&b| b == 0xAA));

        // Test run exceeding max (should split into multiple RLE sequences)
        let over_max: Vec<u8> = vec![0xBB; 70000];

        let compressed_over = DeltaSyncProtocol::rle_compress(&over_max);
        let decompressed_over =
            DeltaSyncProtocol::rle_decompress(&compressed_over).expect("decompress should succeed");

        assert_eq!(decompressed_over.len(), 70000);
        assert!(decompressed_over.iter().all(|&b| b == 0xBB));
    }

    #[test]
    fn test_delta_sync_decompress_truncated_data() {
        let protocol = DeltaSyncProtocol::new();

        // Too short (less than minimum header)
        let too_short = vec![0x53, 0x59, 0x4E, 0x43]; // Just "SYNC"
        let result = protocol.decompress_batch(&too_short);
        assert!(matches!(result, Err(SyncError::CompressionError(_))));

        // Invalid magic
        let invalid_magic = vec![0x00; 100];
        let result = protocol.decompress_batch(&invalid_magic);
        assert!(matches!(result, Err(SyncError::CompressionError(_))));
    }

    #[test]
    fn test_delta_sync_apply_batch_remove_nonexistent() {
        let protocol = DeltaSyncProtocol::new();

        // Create batch with remove for non-existent item
        let batch = SyncBatch::new(
            ContentHash::zero(),
            ContentHash::from_bytes(b"target"),
            vec![SyncOperation::Remove {
                path: "nonexistent.txt".into(),
            }],
            1,
        );

        let get_content = |_: &ContentHash| -> Option<Vec<u8>> { None };
        let put_content = |data: &[u8]| -> ContentHash { ContentHash::from_bytes(data) };

        // Should succeed (removing non-existent is a no-op)
        let result = protocol
            .apply_batch(&batch, get_content, put_content)
            .expect("apply should succeed");

        assert!(result.is_empty());
    }

    #[test]
    fn test_delta_sync_incremental_transfers_only_changed() {
        let protocol = DeltaSyncProtocol::new();

        // Create 100 items
        let items: Vec<SyncItem> = (0..100)
            .map(|i| {
                let path = format!("file_{:02}.txt", i);
                let content = format!("content_{}", i);
                let hash = ContentHash::from_bytes(content.as_bytes());
                SyncItem::new(path, hash, content.len() as u64)
            })
            .collect();

        // Create content store
        let content_store: HashMap<ContentHash, Vec<u8>> = items
            .iter()
            .map(|item| {
                let content = format!("full_content_for_{}", item.path);
                (item.hash.clone(), content.into_bytes())
            })
            .collect();

        // Modify only 3 items
        let mut modified_items = items.clone();
        let mut modified_store = content_store.clone();

        for i in [10, 50, 90] {
            let new_content = format!("modified_content_{}", i);
            modified_items[i].hash = ContentHash::from_bytes(new_content.as_bytes());
            modified_store.insert(modified_items[i].hash.clone(), new_content.into_bytes());
        }

        let get_content =
            |hash: &ContentHash| -> Option<Vec<u8>> { modified_store.get(hash).cloned() };

        let batch = protocol
            .compute_batch(&items, &modified_items, get_content)
            .expect("compute should succeed");

        // Should have exactly 3 operations (only the changed items)
        assert_eq!(
            batch.operations.len(),
            3,
            "incremental sync should only transfer changed items"
        );

        // All should be upserts
        assert!(batch
            .operations
            .iter()
            .all(|op| matches!(op, SyncOperation::Upsert { .. })));
    }

    #[test]
    fn test_delta_sync_binary_patch_vs_full_content() {
        let protocol = DeltaSyncProtocol::new();

        // Create two similar contents (small diff)
        let old_content = b"Hello, this is a test document with some content that will be modified slightly.";
        let new_content = b"Hello, this is a test document with some content that has been modified slightly.";

        let old_hash = ContentHash::from_bytes(old_content);
        let new_hash = ContentHash::from_bytes(new_content);

        let local_items = vec![SyncItem::new("doc.txt", old_hash.clone(), old_content.len() as u64)];
        let remote_items = vec![SyncItem::new("doc.txt", new_hash.clone(), new_content.len() as u64)];

        let content_map: HashMap<ContentHash, Vec<u8>> = [
            (old_hash.clone(), old_content.to_vec()),
            (new_hash.clone(), new_content.to_vec()),
        ]
        .into_iter()
        .collect();

        let get_content = |hash: &ContentHash| -> Option<Vec<u8>> { content_map.get(hash).cloned() };

        let batch = protocol
            .compute_batch(&local_items, &remote_items, get_content)
            .expect("compute should succeed");

        assert_eq!(batch.operations.len(), 1);

        // The delta should be computed (either BinaryPatch or Full depending on efficiency)
        match &batch.operations[0] {
            SyncOperation::Upsert { delta, target_hash, .. } => {
                assert_eq!(target_hash, &new_hash);
                // Delta could be BinaryPatch or Full depending on diff efficiency
                let delta_size = delta.size_bytes();
                assert!(
                    delta_size > 0,
                    "delta should have content"
                );
            }
            _ => panic!("expected Upsert operation"),
        }
    }

    #[test]
    fn test_delta_sync_checkpoint_determinism() {
        let hash1 = ContentHash::from_bytes(b"file1");
        let hash2 = ContentHash::from_bytes(b"file2");

        let items_a = vec![
            SyncItem::new("a.txt", hash1.clone(), 100),
            SyncItem::new("b.txt", hash2.clone(), 200),
        ];

        let items_b = vec![
            SyncItem::new("a.txt", hash1.clone(), 100),
            SyncItem::new("b.txt", hash2.clone(), 200),
        ];

        let checkpoint_a = SyncCheckpoint::from_items(&items_a, 1000);
        let checkpoint_b = SyncCheckpoint::from_items(&items_b, 1000);

        // Same items should produce same checkpoint
        assert_eq!(checkpoint_a.id, checkpoint_b.id);
        assert_eq!(checkpoint_a.item_count, checkpoint_b.item_count);
    }
}

// ---------------------------------------------------------------------------
// Content Hashing and Diffing
// ---------------------------------------------------------------------------

/// SHA-256 based content hash for content-addressable storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ContentHash(pub [u8; 32]);

impl ContentHash {
    /// Creates a content hash from raw bytes.
    pub fn new(hash: [u8; 32]) -> Self {
        Self(hash)
    }

    /// Creates a content hash from a byte slice.
    pub fn from_bytes(data: &[u8]) -> Self {
        Self::compute(data)
    }

    /// Computes the content hash of the given data.
    pub fn compute(data: &[u8]) -> Self {
        Self(sha256(data))
    }

    /// Returns the hash as a hex string.
    pub fn to_hex(&self) -> String {
        self.0.iter().map(|b| format!("{:02x}", b)).collect()
    }

    /// Returns the raw hash bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Returns a zero hash (all bytes zero).
    pub fn zero() -> Self {
        Self([0u8; 32])
    }

    /// Returns true if this hash is all zeros.
    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|&b| b == 0)
    }
}

impl std::fmt::Display for ContentHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

/// A delta representing differences between content.
#[derive(Clone, Debug)]
pub enum Delta {
    /// Binary diff with compressed patch data.
    Binary {
        /// The compressed patch data.
        data: Vec<u8>,
    },
    /// Tree diff with structural changes.
    TreeDiff {
        /// List of tree changes.
        changes: Vec<TreeDiffEntry>,
    },
}

impl Delta {
    /// Creates a binary delta from patch data.
    pub fn binary(data: Vec<u8>) -> Self {
        Delta::Binary { data }
    }

    /// Creates a tree delta from changes.
    pub fn tree_diff(changes: Vec<TreeDiffEntry>) -> Self {
        Delta::TreeDiff { changes }
    }

    /// Returns the size of the delta in bytes.
    pub fn size(&self) -> usize {
        match self {
            Delta::Binary { data } => data.len(),
            Delta::TreeDiff { changes } => changes.len() * std::mem::size_of::<TreeDiffEntry>(),
        }
    }

    /// Alias for size() - returns the size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.size()
    }

    /// Returns the patch data for binary deltas.
    pub fn data(&self) -> Option<&[u8]> {
        match self {
            Delta::Binary { data } => Some(data),
            Delta::TreeDiff { .. } => None,
        }
    }
}

/// Binary diffing algorithm using simple copy/insert encoding.
#[derive(Clone, Debug, Default)]
pub struct BinaryDiffer;

impl BinaryDiffer {
    /// Creates a new binary differ.
    pub fn new() -> Self {
        Self
    }

    /// Computes a delta between old and new byte sequences.
    pub fn diff(&self, old: &[u8], new: &[u8]) -> Result<Delta, String> {
        // Simple delta encoding: store changed bytes with their positions
        let mut delta_data = Vec::new();

        // Find matching and differing regions
        let mut i = 0;
        while i < new.len() {
            if i < old.len() && old[i] == new[i] {
                // Skip matching bytes
                i += 1;
            } else {
                // Record position and new value
                delta_data.extend_from_slice(&(i as u32).to_le_bytes());
                delta_data.push(new.get(i).copied().unwrap_or(0));
                i += 1;
            }
        }

        Ok(Delta::Binary { data: delta_data })
    }

    /// Applies a delta to reconstruct the new data.
    pub fn apply(&self, old: &[u8], delta: &Delta) -> Result<Vec<u8>, String> {
        let delta_data = match delta {
            Delta::Binary { data } => data,
            Delta::TreeDiff { .. } => return Err("Cannot apply tree diff as binary".into()),
        };

        let mut result = old.to_vec();

        // Parse and apply delta operations
        let mut i = 0;
        while i + 4 < delta_data.len() {
            let pos = u32::from_le_bytes([
                delta_data[i], delta_data[i+1], delta_data[i+2], delta_data[i+3]
            ]) as usize;
            let value = delta_data[i + 4];

            if pos < result.len() {
                result[pos] = value;
            } else if pos == result.len() {
                result.push(value);
            }

            i += 5;
        }

        Ok(result)
    }
}

/// A tree entry for hierarchical content addressing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TreeEntry {
    /// Entry name/path.
    pub name: String,
    /// Content hash of this entry.
    pub hash: ContentHash,
    /// Whether this is a directory (contains children).
    pub is_dir: bool,
}

impl TreeEntry {
    /// Creates a blob (file) entry.
    pub fn blob(name: impl Into<String>, hash: ContentHash) -> Self {
        Self {
            name: name.into(),
            hash,
            is_dir: false,
        }
    }

    /// Creates a tree (directory) entry.
    pub fn tree(name: impl Into<String>, hash: ContentHash) -> Self {
        Self {
            name: name.into(),
            hash,
            is_dir: true,
        }
    }
}

/// A tree of content-addressed entries.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ContentTree {
    /// Root entries in this tree.
    entries_vec: Vec<TreeEntry>,
}

impl ContentTree {
    /// Creates an empty content tree.
    pub fn new() -> Self {
        Self { entries_vec: Vec::new() }
    }

    /// Creates a content tree from a list of entries.
    pub fn from_entries(entries: Vec<TreeEntry>) -> Self {
        Self { entries_vec: entries }
    }

    /// Adds an entry to the tree.
    pub fn add(&mut self, entry: TreeEntry) {
        self.entries_vec.push(entry);
    }

    /// Returns the entries as a vector reference.
    pub fn entries(&self) -> &Vec<TreeEntry> {
        &self.entries_vec
    }

    /// Returns the entries as a slice.
    pub fn as_slice(&self) -> &[TreeEntry] {
        &self.entries_vec
    }

    /// Serializes the tree to bytes (simple format).
    pub fn serialize(&self) -> Vec<u8> {
        let mut data = Vec::new();
        for entry in &self.entries_vec {
            data.extend_from_slice(entry.name.as_bytes());
            data.push(0); // null separator
            data.extend_from_slice(entry.hash.as_bytes());
            data.push(if entry.is_dir { 1 } else { 0 });
        }
        data
    }

    /// Deserializes a tree from bytes.
    pub fn deserialize(data: &[u8]) -> Result<Self, String> {
        let mut entries = Vec::new();
        let mut i = 0;

        while i < data.len() {
            // Read name until null
            let name_end = data[i..].iter().position(|&b| b == 0)
                .ok_or("Invalid format: missing name terminator")?;
            let name = String::from_utf8(data[i..i+name_end].to_vec())
                .map_err(|e| format!("Invalid name: {}", e))?;
            i += name_end + 1;

            // Read hash (32 bytes)
            if i + 33 > data.len() {
                return Err("Invalid format: truncated entry".into());
            }
            let mut hash = [0u8; 32];
            hash.copy_from_slice(&data[i..i+32]);
            i += 32;

            // Read is_dir flag
            let is_dir = data[i] != 0;
            i += 1;

            entries.push(TreeEntry {
                name,
                hash: ContentHash(hash),
                is_dir,
            });
        }

        Ok(Self::from_entries(entries))
    }

    /// Returns the number of entries.
    pub fn len(&self) -> usize {
        self.entries_vec.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.entries_vec.is_empty()
    }
}

/// Describes a difference in a tree structure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TreeDiffEntry {
    /// Entry was added.
    Added(TreeEntry),
    /// Entry was removed/deleted.
    Deleted(TreeEntry),
    /// Entry was modified.
    Modified { old: TreeEntry, new: TreeEntry },
}

impl TreeDiffEntry {
    /// Returns the name of the affected entry.
    pub fn name(&self) -> &str {
        match self {
            TreeDiffEntry::Added(e) => &e.name,
            TreeDiffEntry::Deleted(e) => &e.name,
            TreeDiffEntry::Modified { new, .. } => &new.name,
        }
    }

    /// Returns the entry (or new entry for modifications).
    pub fn entry(&self) -> &TreeEntry {
        match self {
            TreeDiffEntry::Added(e) => e,
            TreeDiffEntry::Deleted(e) => e,
            TreeDiffEntry::Modified { new, .. } => new,
        }
    }
}

/// Tree-based diffing for hierarchical content.
#[derive(Clone, Debug, Default)]
pub struct TreeDiffer;

impl TreeDiffer {
    /// Creates a new tree differ.
    pub fn new() -> Self {
        Self
    }

    /// Computes differences between two trees as a list.
    pub fn diff(&self, old: &ContentTree, new: &ContentTree) -> Vec<TreeDiffEntry> {
        self.diff_entries(old, new)
    }

    /// Computes differences between two trees and returns a Delta.
    pub fn diff_trees(&self, old: &ContentTree, new: &ContentTree) -> Delta {
        Delta::TreeDiff { changes: self.diff_entries(old, new) }
    }

    /// Computes differences between two trees (internal helper).
    fn diff_entries(&self, old: &ContentTree, new: &ContentTree) -> Vec<TreeDiffEntry> {
        use std::collections::HashMap;

        let old_map: HashMap<&str, &TreeEntry> = old.entries().iter()
            .map(|e| (e.name.as_str(), e))
            .collect();
        let new_map: HashMap<&str, &TreeEntry> = new.entries().iter()
            .map(|e| (e.name.as_str(), e))
            .collect();

        let mut diffs = Vec::new();

        // Find added and modified entries
        for entry in new.entries() {
            match old_map.get(entry.name.as_str()) {
                Some(old_entry) if old_entry.hash != entry.hash => {
                    diffs.push(TreeDiffEntry::Modified {
                        old: (*old_entry).clone(),
                        new: entry.clone(),
                    });
                }
                None => {
                    diffs.push(TreeDiffEntry::Added(entry.clone()));
                }
                _ => {}
            }
        }

        // Find removed entries
        for entry in old.entries() {
            if !new_map.contains_key(entry.name.as_str()) {
                diffs.push(TreeDiffEntry::Deleted(entry.clone()));
            }
        }

        diffs
    }

    /// Applies tree diff (Delta) to reconstruct a new tree.
    pub fn apply(&self, base: &ContentTree, delta: &Delta) -> Result<ContentTree, String> {
        self.apply_to_tree(base, delta)
    }

    /// Applies tree diff (Delta) to reconstruct a new tree.
    pub fn apply_to_tree(&self, base: &ContentTree, delta: &Delta) -> Result<ContentTree, String> {
        let diffs = match delta {
            Delta::TreeDiff { changes } => changes,
            Delta::Binary { .. } => return Err("Cannot apply binary delta to tree".into()),
        };

        use std::collections::HashMap;

        let mut result_map: HashMap<String, TreeEntry> = base.entries().iter()
            .map(|e| (e.name.clone(), e.clone()))
            .collect();

        for diff in diffs {
            match diff {
                TreeDiffEntry::Added(entry) => {
                    result_map.insert(entry.name.clone(), entry.clone());
                }
                TreeDiffEntry::Deleted(entry) => {
                    result_map.remove(&entry.name);
                }
                TreeDiffEntry::Modified { new, .. } => {
                    result_map.insert(new.name.clone(), new.clone());
                }
            }
        }

        Ok(ContentTree::from_entries(result_map.into_values().collect()))
    }
}

/// Trait for content diffing implementations.
pub trait ContentDiffer {
    /// Computes a delta between old and new content.
    fn diff(&self, old: &[u8], new: &[u8]) -> Result<Delta, String>;

    /// Applies a delta to reconstruct content.
    fn apply(&self, old: &[u8], delta: &Delta) -> Result<Vec<u8>, String>;
}

impl ContentDiffer for BinaryDiffer {
    fn diff(&self, old: &[u8], new: &[u8]) -> Result<Delta, String> {
        BinaryDiffer::diff(self, old, new)
    }

    fn apply(&self, old: &[u8], delta: &Delta) -> Result<Vec<u8>, String> {
        BinaryDiffer::apply(self, old, delta)
    }
}

// ---------------------------------------------------------------------------
// Content-Addressable File Storage
// ---------------------------------------------------------------------------

/// A chunk of content with streaming support.
#[derive(Clone, Debug)]
pub struct ChunkedContent {
    /// The content data.
    pub data: Vec<u8>,
    /// Whether this is the final chunk.
    pub is_final: bool,
}

impl ChunkedContent {
    /// Creates a new content chunk.
    pub fn new(data: Vec<u8>, is_final: bool) -> Self {
        Self { data, is_final }
    }

    /// Creates a complete (single-chunk) content.
    pub fn complete(data: Vec<u8>) -> Self {
        Self { data, is_final: true }
    }
}

impl std::io::Read for ChunkedContent {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let len = std::cmp::min(buf.len(), self.data.len());
        buf[..len].copy_from_slice(&self.data[..len]);
        self.data.drain(..len);
        Ok(len)
    }
}

/// File-based content-addressable storage backend.
///
/// Stores content using SHA-256 hash as the key, with a git-style
/// directory layout (ab/cdef...) for efficient filesystem access.
#[derive(Debug)]
pub struct FileBackend {
    /// Root directory for storage.
    root: std::path::PathBuf,
    /// Streaming threshold in bytes.
    streaming_threshold: usize,
}

impl FileBackend {
    /// Creates a new file backend at the given path.
    pub fn new(root: impl Into<std::path::PathBuf>) -> Self {
        Self {
            root: root.into(),
            streaming_threshold: 1024 * 1024, // 1MB default
        }
    }

    /// Sets the streaming threshold.
    pub fn with_streaming_threshold(mut self, threshold: usize) -> Self {
        self.streaming_threshold = threshold;
        self
    }

    /// Returns the path for a given hash.
    fn path_for_hash(&self, hash: &ContentHash) -> std::path::PathBuf {
        let hex = hash.to_hex();
        self.root.join(&hex[..2]).join(&hex[2..])
    }

    /// Stores content and returns its hash.
    pub fn put(&self, data: &[u8]) -> std::io::Result<ContentHash> {
        let hash = ContentHash::compute(data);
        let path = self.path_for_hash(&hash);

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        if !path.exists() {
            std::fs::write(&path, data)?;
        }

        Ok(hash)
    }

    /// Retrieves content by hash.
    pub fn get(&self, hash: &ContentHash) -> std::io::Result<Vec<u8>> {
        let path = self.path_for_hash(hash);
        std::fs::read(path)
    }

    /// Retrieves content as a chunked reader.
    pub fn get_chunked(&self, hash: &ContentHash) -> std::io::Result<ChunkedContent> {
        let data = self.get(hash)?;
        Ok(ChunkedContent::complete(data))
    }

    /// Checks if content exists.
    pub fn has(&self, hash: &ContentHash) -> bool {
        self.path_for_hash(hash).exists()
    }

    /// Deletes content by hash.
    pub fn delete(&self, hash: &ContentHash) -> std::io::Result<()> {
        let path = self.path_for_hash(hash);
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Provenance Chain — Content history tracking
// ---------------------------------------------------------------------------

/// Strategy for pruning old entries from a provenance chain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PruningStrategy {
    /// Keep all entries (no pruning).
    KeepAll,
    /// Keep only the last N entries (plus origin).
    KeepLastN(usize),
    /// Keep entries newer than the given timestamp.
    KeepNewerThan(u64),
    /// Keep entries with age less than the given duration in seconds.
    MaxAge(u64),
    /// Combined strategy: keep last N and within max age.
    Combined {
        keep_last_n: usize,
        max_age_secs: u64,
    },
}

impl Default for PruningStrategy {
    fn default() -> Self {
        Self::KeepAll
    }
}

/// A single entry in a provenance chain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProvenanceEntry {
    /// Content hash of this entry.
    pub hash: ContentHash,
    /// Timestamp when this entry was created.
    pub timestamp: u64,
    /// Optional human-readable message/description.
    pub message: Option<String>,
    /// Hash of the parent entry (None for origin).
    pub parent: Option<ContentHash>,
}

impl ProvenanceEntry {
    /// Creates an origin entry (no parent).
    pub fn origin(hash: ContentHash, timestamp: u64, message: Option<String>) -> Self {
        Self {
            hash,
            timestamp,
            message,
            parent: None,
        }
    }

    /// Creates an entry with a parent.
    pub fn with_parent(
        hash: ContentHash,
        timestamp: u64,
        message: Option<String>,
        parent: ContentHash,
    ) -> Self {
        Self {
            hash,
            timestamp,
            message,
            parent: Some(parent),
        }
    }

    /// Returns true if this is an origin entry.
    pub fn is_origin(&self) -> bool {
        self.parent.is_none()
    }
}

/// A chain of provenance entries tracking content history.
#[derive(Clone, Debug, Default)]
pub struct ProvenanceChain {
    entries: Vec<ProvenanceEntry>,
    strategy: PruningStrategy,
}

impl ProvenanceChain {
    /// Creates an empty provenance chain with the given strategy.
    pub fn new(strategy: PruningStrategy) -> Self {
        Self {
            entries: Vec::new(),
            strategy,
        }
    }

    /// Creates an empty provenance chain with KeepAll strategy.
    pub fn empty() -> Self {
        Self {
            entries: Vec::new(),
            strategy: PruningStrategy::KeepAll,
        }
    }

    /// Creates a chain with an origin entry and pruning strategy.
    pub fn with_origin(origin: ProvenanceEntry, strategy: PruningStrategy) -> Self {
        Self {
            entries: vec![origin],
            strategy,
        }
    }

    /// Pushes a new entry and applies pruning strategy.
    pub fn push(&mut self, entry: ProvenanceEntry) {
        self.entries.push(entry);
        self.apply_pruning();
    }

    /// Returns the origin entry if present.
    pub fn origin(&self) -> Option<&ProvenanceEntry> {
        self.entries.first()
    }

    /// Returns the current (most recent) entry.
    pub fn current(&self) -> Option<&ProvenanceEntry> {
        self.entries.last()
    }

    /// Returns the number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns an iterator over entries.
    pub fn iter(&self) -> impl Iterator<Item = &ProvenanceEntry> {
        self.entries.iter()
    }

    /// Returns the entries as a vector reference.
    pub fn entries(&self) -> &Vec<ProvenanceEntry> {
        &self.entries
    }

    /// Applies the pruning strategy to maintain size constraints.
    fn apply_pruning(&mut self) {
        match &self.strategy {
            PruningStrategy::KeepAll => {}
            PruningStrategy::KeepLastN(n) => {
                // Minimum of 2 entries (origin + current)
                let effective_n = (*n).max(2);
                if self.entries.len() > effective_n {
                    // Keep origin (first) and last (n-1) entries
                    let origin = self.entries.remove(0);
                    let keep_count = effective_n - 1;
                    if self.entries.len() > keep_count {
                        let start = self.entries.len() - keep_count;
                        self.entries = self.entries.drain(start..).collect();
                    }
                    self.entries.insert(0, origin);
                }
            }
            PruningStrategy::KeepNewerThan(timestamp) => {
                // Keep origin and entries newer than timestamp
                if let Some(origin) = self.entries.first().cloned() {
                    self.entries.retain(|e| e.timestamp >= *timestamp || e.is_origin());
                    if self.entries.is_empty() {
                        self.entries.push(origin);
                    }
                }
            }
            PruningStrategy::MaxAge(max_age) => {
                // Keep entries within max_age seconds of the newest
                if let Some(newest) = self.entries.last().map(|e| e.timestamp) {
                    let cutoff = newest.saturating_sub(*max_age);
                    if let Some(origin) = self.entries.first().cloned() {
                        self.entries.retain(|e| e.timestamp >= cutoff || e.is_origin());
                        if self.entries.is_empty() {
                            self.entries.push(origin);
                        }
                    }
                }
            }
            PruningStrategy::Combined { keep_last_n, max_age_secs } => {
                // Apply both constraints: keep last N AND within max age
                if let Some(newest) = self.entries.last().map(|e| e.timestamp) {
                    let cutoff = newest.saturating_sub(*max_age_secs);
                    // First filter by age
                    if let Some(origin) = self.entries.first().cloned() {
                        self.entries.retain(|e| e.timestamp >= cutoff || e.is_origin());
                        // Then apply keep_last_n
                        if self.entries.len() > *keep_last_n && *keep_last_n > 0 {
                            let origin = self.entries.remove(0);
                            let keep_count = keep_last_n - 1;
                            if self.entries.len() > keep_count {
                                let start = self.entries.len() - keep_count;
                                self.entries = self.entries.drain(start..).collect();
                            }
                            self.entries.insert(0, origin);
                        }
                        if self.entries.is_empty() {
                            self.entries.push(origin);
                        }
                    }
                }
            }
        }
    }

    /// Returns the chain as a slice.
    pub fn as_slice(&self) -> &[ProvenanceEntry] {
        &self.entries
    }
}
