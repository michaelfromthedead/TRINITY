//! DeltaSync: Incremental Asset Synchronization for TRINITY (T-AS-4.8).
//!
//! Provides O(differences) asset synchronization for distributed build systems:
//!
//! - **Remote Build Farm Sync**: Compute cooked asset diff against local cache,
//!   only transfer changed content hashes
//! - **Multi-Platform Cooking Sharing**: Identical intermediate data shared across
//!   platforms, only platform-specific output transmitted
//! - **Runtime Update Diffs**: Game clients receive content hash diffs, not full
//!   asset downloads
//! - **ContentDiffer Integration**: Produce O(differences) proofs for all use cases
//!
//! # Architecture
//!
//! ```text
//! DeltaSync
//!   |-- compute_manifest()     -> SyncManifest (local content hashes + version)
//!   |-- compute_delta()        -> DeltaProof (added/removed/modified)
//!   |-- prepare_upload()       -> Vec<TransferChunk> (chunked transfer)
//!   |-- apply_download()       -> Result<()> (receive and apply)
//!   |-- resolve_conflict()     -> ContentHash (conflict resolution)
//!   |
//!   Multi-Platform:
//!   |-- detect_platform_specific() -> bool
//!   |-- compute_shared_delta()     -> DeltaProof (shared content only)
//! ```
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::asset::delta_sync::{DeltaSync, DeltaSyncConfig, ConflictStrategy};
//!
//! // Set up delta sync with content store
//! let store = Arc::new(MemoryContentStore::default());
//! let config = DeltaSyncConfig::default();
//! let mut sync = DeltaSync::new(store, config);
//!
//! // Compute local manifest
//! let local_manifest = sync.compute_manifest();
//!
//! // Compute delta against remote
//! let delta = sync.compute_delta(&remote_manifest);
//!
//! // Prepare chunks for upload
//! let chunks = sync.prepare_upload(&delta)?;
//!
//! // Apply downloaded chunks
//! sync.apply_download(chunks)?;
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::io::{self, Read};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::asset::content_store::{ContentStore, SeekableReader};
use crate::asset::provenance::Platform;
use crate::pipeline::ContentHash;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Default chunk size for transfers (1MB).
pub const DEFAULT_CHUNK_SIZE: usize = 1024 * 1024;

/// Minimum chunk size (64KB).
pub const MIN_CHUNK_SIZE: usize = 64 * 1024;

/// Maximum chunk size (16MB).
pub const MAX_CHUNK_SIZE: usize = 16 * 1024 * 1024;

/// Default maximum concurrent transfers.
pub const DEFAULT_MAX_CONCURRENT: usize = 4;

/// Configuration for delta synchronization.
#[derive(Debug, Clone)]
pub struct DeltaSyncConfig {
    /// Transfer chunk size in bytes.
    pub chunk_size: usize,
    /// Maximum concurrent transfers.
    pub max_concurrent_transfers: usize,
    /// Conflict resolution strategy.
    pub conflict_strategy: ConflictStrategy,
    /// Whether to verify hashes after transfer.
    pub verify_transfers: bool,
    /// Whether to compress chunks during transfer.
    pub compress_chunks: bool,
}

impl Default for DeltaSyncConfig {
    fn default() -> Self {
        Self {
            chunk_size: DEFAULT_CHUNK_SIZE,
            max_concurrent_transfers: DEFAULT_MAX_CONCURRENT,
            conflict_strategy: ConflictStrategy::LatestWins,
            verify_transfers: true,
            compress_chunks: false,
        }
    }
}

impl DeltaSyncConfig {
    /// Create config with custom chunk size.
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size.clamp(MIN_CHUNK_SIZE, MAX_CHUNK_SIZE);
        self
    }

    /// Set maximum concurrent transfers.
    pub fn with_max_concurrent(mut self, count: usize) -> Self {
        self.max_concurrent_transfers = count.max(1);
        self
    }

    /// Set conflict resolution strategy.
    pub fn with_conflict_strategy(mut self, strategy: ConflictStrategy) -> Self {
        self.conflict_strategy = strategy;
        self
    }

    /// Enable or disable transfer verification.
    pub fn with_verify_transfers(mut self, verify: bool) -> Self {
        self.verify_transfers = verify;
        self
    }

    /// Enable or disable chunk compression.
    pub fn with_compress_chunks(mut self, compress: bool) -> Self {
        self.compress_chunks = compress;
        self
    }
}

/// Strategy for resolving sync conflicts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConflictStrategy {
    /// Latest timestamp wins.
    LatestWins,
    /// Use version vectors for distributed consistency.
    VersionVector,
    /// Mark for manual resolution.
    Manual,
    /// Local always wins (keep local version).
    LocalWins,
    /// Remote always wins (accept remote version).
    RemoteWins,
}

impl Default for ConflictStrategy {
    fn default() -> Self {
        Self::LatestWins
    }
}

// ---------------------------------------------------------------------------
// Error Types
// ---------------------------------------------------------------------------

/// Errors that can occur during delta sync operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeltaSyncError {
    /// Content not found in store.
    ContentNotFound(ContentHash),
    /// Hash verification failed.
    HashMismatch {
        expected: ContentHash,
        actual: ContentHash,
    },
    /// Transfer was interrupted.
    TransferInterrupted {
        hash: ContentHash,
        bytes_transferred: u64,
        total_bytes: u64,
    },
    /// Chunk index out of range.
    InvalidChunkIndex {
        index: usize,
        total_chunks: usize,
    },
    /// Conflict requires manual resolution.
    ConflictRequiresManual {
        local: ContentHash,
        remote: ContentHash,
    },
    /// Version vector conflict.
    VersionVectorConflict {
        local_version: u64,
        remote_version: u64,
    },
    /// I/O error.
    IoError(String),
    /// Invalid manifest format.
    InvalidManifest(String),
}

impl fmt::Display for DeltaSyncError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContentNotFound(hash) => write!(f, "content not found: {}", hash),
            Self::HashMismatch { expected, actual } => {
                write!(f, "hash mismatch: expected {}, got {}", expected, actual)
            }
            Self::TransferInterrupted {
                hash,
                bytes_transferred,
                total_bytes,
            } => {
                write!(
                    f,
                    "transfer interrupted for {}: {}/{} bytes",
                    hash, bytes_transferred, total_bytes
                )
            }
            Self::InvalidChunkIndex { index, total_chunks } => {
                write!(
                    f,
                    "invalid chunk index {}, total chunks {}",
                    index, total_chunks
                )
            }
            Self::ConflictRequiresManual { local, remote } => {
                write!(
                    f,
                    "conflict requires manual resolution: local={}, remote={}",
                    local, remote
                )
            }
            Self::VersionVectorConflict {
                local_version,
                remote_version,
            } => {
                write!(
                    f,
                    "version vector conflict: local={}, remote={}",
                    local_version, remote_version
                )
            }
            Self::IoError(msg) => write!(f, "I/O error: {}", msg),
            Self::InvalidManifest(msg) => write!(f, "invalid manifest: {}", msg),
        }
    }
}

impl std::error::Error for DeltaSyncError {}

impl From<io::Error> for DeltaSyncError {
    fn from(e: io::Error) -> Self {
        Self::IoError(e.to_string())
    }
}

/// Result type for delta sync operations.
pub type DeltaSyncResult<T> = Result<T, DeltaSyncError>;

// ---------------------------------------------------------------------------
// Sync Manifest
// ---------------------------------------------------------------------------

/// A manifest of content hashes for synchronization.
///
/// Contains the set of content hashes present locally along with
/// version information for conflict resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncManifest {
    /// Set of content hashes present in this manifest.
    pub local_hashes: HashSet<ContentHash>,
    /// Monotonic version number.
    pub version: u64,
    /// Timestamp of manifest creation (ms since UNIX epoch).
    pub timestamp: u64,
    /// Optional node ID for distributed environments.
    pub node_id: Option<String>,
    /// Platform-specific content markers.
    pub platform_markers: HashMap<ContentHash, Platform>,
}

impl SyncManifest {
    /// Create a new empty manifest.
    pub fn new() -> Self {
        Self {
            local_hashes: HashSet::new(),
            version: 0,
            timestamp: current_timestamp_ms(),
            node_id: None,
            platform_markers: HashMap::new(),
        }
    }

    /// Create manifest with a set of hashes.
    pub fn with_hashes(hashes: HashSet<ContentHash>) -> Self {
        Self {
            local_hashes: hashes,
            version: 0,
            timestamp: current_timestamp_ms(),
            node_id: None,
            platform_markers: HashMap::new(),
        }
    }

    /// Set the version number.
    pub fn with_version(mut self, version: u64) -> Self {
        self.version = version;
        self
    }

    /// Set the timestamp.
    pub fn with_timestamp(mut self, timestamp: u64) -> Self {
        self.timestamp = timestamp;
        self
    }

    /// Set the node ID.
    pub fn with_node_id(mut self, node_id: impl Into<String>) -> Self {
        self.node_id = Some(node_id.into());
        self
    }

    /// Add a hash to the manifest.
    pub fn add_hash(&mut self, hash: ContentHash) {
        self.local_hashes.insert(hash);
    }

    /// Remove a hash from the manifest.
    pub fn remove_hash(&mut self, hash: &ContentHash) -> bool {
        self.local_hashes.remove(hash)
    }

    /// Check if manifest contains a hash.
    pub fn contains(&self, hash: &ContentHash) -> bool {
        self.local_hashes.contains(hash)
    }

    /// Get the number of hashes in the manifest.
    pub fn len(&self) -> usize {
        self.local_hashes.len()
    }

    /// Check if manifest is empty.
    pub fn is_empty(&self) -> bool {
        self.local_hashes.is_empty()
    }

    /// Mark a hash as platform-specific.
    pub fn mark_platform_specific(&mut self, hash: ContentHash, platform: Platform) {
        self.platform_markers.insert(hash, platform);
    }

    /// Check if a hash is platform-specific.
    pub fn is_platform_specific(&self, hash: &ContentHash) -> bool {
        self.platform_markers.contains_key(hash)
    }

    /// Get the platform for a hash if it's platform-specific.
    pub fn get_platform(&self, hash: &ContentHash) -> Option<Platform> {
        self.platform_markers.get(hash).copied()
    }

    /// Get all hashes for a specific platform.
    pub fn hashes_for_platform(&self, platform: Platform) -> Vec<ContentHash> {
        self.platform_markers
            .iter()
            .filter(|(_, &p)| p == platform)
            .map(|(h, _)| *h)
            .collect()
    }

    /// Get all platform-independent (shared) hashes.
    pub fn shared_hashes(&self) -> Vec<ContentHash> {
        self.local_hashes
            .iter()
            .filter(|h| !self.platform_markers.contains_key(*h))
            .copied()
            .collect()
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut data = Vec::new();

        // Magic number and version
        data.extend_from_slice(b"SYNC");
        data.extend_from_slice(&1u32.to_le_bytes());

        // Version and timestamp
        data.extend_from_slice(&self.version.to_le_bytes());
        data.extend_from_slice(&self.timestamp.to_le_bytes());

        // Node ID
        match &self.node_id {
            Some(id) => {
                data.push(1);
                data.extend_from_slice(&(id.len() as u32).to_le_bytes());
                data.extend_from_slice(id.as_bytes());
            }
            None => {
                data.push(0);
            }
        }

        // Hash count and hashes
        data.extend_from_slice(&(self.local_hashes.len() as u32).to_le_bytes());
        for hash in &self.local_hashes {
            data.extend_from_slice(hash.as_bytes());
        }

        // Platform markers
        data.extend_from_slice(&(self.platform_markers.len() as u32).to_le_bytes());
        for (hash, platform) in &self.platform_markers {
            data.extend_from_slice(hash.as_bytes());
            data.push(platform.to_byte());
        }

        data
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> DeltaSyncResult<Self> {
        if data.len() < 8 {
            return Err(DeltaSyncError::InvalidManifest("data too short".into()));
        }

        // Check magic
        if &data[0..4] != b"SYNC" {
            return Err(DeltaSyncError::InvalidManifest("invalid magic".into()));
        }

        let format_version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        if format_version != 1 {
            return Err(DeltaSyncError::InvalidManifest(format!(
                "unsupported version: {}",
                format_version
            )));
        }

        let mut pos = 8;

        // Version and timestamp
        if pos + 16 > data.len() {
            return Err(DeltaSyncError::InvalidManifest("truncated header".into()));
        }
        let version = u64::from_le_bytes([
            data[pos],
            data[pos + 1],
            data[pos + 2],
            data[pos + 3],
            data[pos + 4],
            data[pos + 5],
            data[pos + 6],
            data[pos + 7],
        ]);
        pos += 8;

        let timestamp = u64::from_le_bytes([
            data[pos],
            data[pos + 1],
            data[pos + 2],
            data[pos + 3],
            data[pos + 4],
            data[pos + 5],
            data[pos + 6],
            data[pos + 7],
        ]);
        pos += 8;

        // Node ID
        if pos >= data.len() {
            return Err(DeltaSyncError::InvalidManifest("truncated node id flag".into()));
        }
        let has_node_id = data[pos] != 0;
        pos += 1;

        let node_id = if has_node_id {
            if pos + 4 > data.len() {
                return Err(DeltaSyncError::InvalidManifest("truncated node id length".into()));
            }
            let id_len = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                as usize;
            pos += 4;

            if pos + id_len > data.len() {
                return Err(DeltaSyncError::InvalidManifest("truncated node id".into()));
            }
            let id = String::from_utf8(data[pos..pos + id_len].to_vec())
                .map_err(|_| DeltaSyncError::InvalidManifest("invalid UTF-8 in node id".into()))?;
            pos += id_len;
            Some(id)
        } else {
            None
        };

        // Hashes
        if pos + 4 > data.len() {
            return Err(DeltaSyncError::InvalidManifest("truncated hash count".into()));
        }
        let hash_count =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;

        let mut local_hashes = HashSet::with_capacity(hash_count);
        for _ in 0..hash_count {
            if pos + 32 > data.len() {
                return Err(DeltaSyncError::InvalidManifest("truncated hash".into()));
            }
            let mut hash_bytes = [0u8; 32];
            hash_bytes.copy_from_slice(&data[pos..pos + 32]);
            local_hashes.insert(ContentHash::from_raw(hash_bytes));
            pos += 32;
        }

        // Platform markers
        let mut platform_markers = HashMap::new();
        if pos + 4 <= data.len() {
            let marker_count = u32::from_le_bytes([
                data[pos],
                data[pos + 1],
                data[pos + 2],
                data[pos + 3],
            ]) as usize;
            pos += 4;

            for _ in 0..marker_count {
                if pos + 33 > data.len() {
                    break;
                }
                let mut hash_bytes = [0u8; 32];
                hash_bytes.copy_from_slice(&data[pos..pos + 32]);
                let hash = ContentHash::from_raw(hash_bytes);
                pos += 32;

                let platform = Platform::from_byte(data[pos]);
                pos += 1;

                platform_markers.insert(hash, platform);
            }
        }

        Ok(Self {
            local_hashes,
            version,
            timestamp,
            node_id,
            platform_markers,
        })
    }
}

impl Default for SyncManifest {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Delta Proof
// ---------------------------------------------------------------------------

/// A proof of differences between two manifests.
///
/// Represents O(differences) data for efficient synchronization:
/// - Added: hashes present in new but not old
/// - Removed: hashes present in old but not new
/// - Modified: hashes that changed (old hash, new hash)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeltaProof {
    /// Content hashes that were added.
    pub added: Vec<ContentHash>,
    /// Content hashes that were removed.
    pub removed: Vec<ContentHash>,
    /// Content hashes that were modified (old, new).
    pub modified: Vec<(ContentHash, ContentHash)>,
    /// Count of unchanged content hashes.
    pub unchanged_count: usize,
    /// Source manifest version.
    pub source_version: u64,
    /// Target manifest version.
    pub target_version: u64,
}

impl DeltaProof {
    /// Create an empty delta proof.
    pub fn empty() -> Self {
        Self {
            added: Vec::new(),
            removed: Vec::new(),
            modified: Vec::new(),
            unchanged_count: 0,
            source_version: 0,
            target_version: 0,
        }
    }

    /// Check if the delta is empty (no changes).
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.modified.is_empty()
    }

    /// Get total number of changes.
    pub fn change_count(&self) -> usize {
        self.added.len() + self.removed.len() + self.modified.len()
    }

    /// Get total number of items (changed + unchanged).
    pub fn total_count(&self) -> usize {
        self.change_count() + self.unchanged_count
    }

    /// Get change ratio (0.0 = no changes, 1.0 = all changed).
    pub fn change_ratio(&self) -> f64 {
        let total = self.total_count();
        if total == 0 {
            0.0
        } else {
            self.change_count() as f64 / total as f64
        }
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut data = Vec::new();

        // Magic and version
        data.extend_from_slice(b"DELT");
        data.extend_from_slice(&1u32.to_le_bytes());

        // Versions
        data.extend_from_slice(&self.source_version.to_le_bytes());
        data.extend_from_slice(&self.target_version.to_le_bytes());

        // Unchanged count
        data.extend_from_slice(&(self.unchanged_count as u64).to_le_bytes());

        // Added
        data.extend_from_slice(&(self.added.len() as u32).to_le_bytes());
        for hash in &self.added {
            data.extend_from_slice(hash.as_bytes());
        }

        // Removed
        data.extend_from_slice(&(self.removed.len() as u32).to_le_bytes());
        for hash in &self.removed {
            data.extend_from_slice(hash.as_bytes());
        }

        // Modified
        data.extend_from_slice(&(self.modified.len() as u32).to_le_bytes());
        for (old, new) in &self.modified {
            data.extend_from_slice(old.as_bytes());
            data.extend_from_slice(new.as_bytes());
        }

        data
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> DeltaSyncResult<Self> {
        if data.len() < 8 {
            return Err(DeltaSyncError::InvalidManifest("delta too short".into()));
        }

        if &data[0..4] != b"DELT" {
            return Err(DeltaSyncError::InvalidManifest("invalid delta magic".into()));
        }

        let format_version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        if format_version != 1 {
            return Err(DeltaSyncError::InvalidManifest(format!(
                "unsupported delta version: {}",
                format_version
            )));
        }

        let mut pos = 8;

        // Versions
        if pos + 24 > data.len() {
            return Err(DeltaSyncError::InvalidManifest("truncated delta header".into()));
        }
        let source_version = u64::from_le_bytes([
            data[pos],
            data[pos + 1],
            data[pos + 2],
            data[pos + 3],
            data[pos + 4],
            data[pos + 5],
            data[pos + 6],
            data[pos + 7],
        ]);
        pos += 8;

        let target_version = u64::from_le_bytes([
            data[pos],
            data[pos + 1],
            data[pos + 2],
            data[pos + 3],
            data[pos + 4],
            data[pos + 5],
            data[pos + 6],
            data[pos + 7],
        ]);
        pos += 8;

        let unchanged_count = u64::from_le_bytes([
            data[pos],
            data[pos + 1],
            data[pos + 2],
            data[pos + 3],
            data[pos + 4],
            data[pos + 5],
            data[pos + 6],
            data[pos + 7],
        ]) as usize;
        pos += 8;

        // Added
        if pos + 4 > data.len() {
            return Err(DeltaSyncError::InvalidManifest("truncated added count".into()));
        }
        let added_count =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;

        let mut added = Vec::with_capacity(added_count);
        for _ in 0..added_count {
            if pos + 32 > data.len() {
                return Err(DeltaSyncError::InvalidManifest("truncated added hash".into()));
            }
            let mut hash_bytes = [0u8; 32];
            hash_bytes.copy_from_slice(&data[pos..pos + 32]);
            added.push(ContentHash::from_raw(hash_bytes));
            pos += 32;
        }

        // Removed
        if pos + 4 > data.len() {
            return Err(DeltaSyncError::InvalidManifest("truncated removed count".into()));
        }
        let removed_count =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;

        let mut removed = Vec::with_capacity(removed_count);
        for _ in 0..removed_count {
            if pos + 32 > data.len() {
                return Err(DeltaSyncError::InvalidManifest("truncated removed hash".into()));
            }
            let mut hash_bytes = [0u8; 32];
            hash_bytes.copy_from_slice(&data[pos..pos + 32]);
            removed.push(ContentHash::from_raw(hash_bytes));
            pos += 32;
        }

        // Modified
        if pos + 4 > data.len() {
            return Err(DeltaSyncError::InvalidManifest("truncated modified count".into()));
        }
        let modified_count =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;

        let mut modified = Vec::with_capacity(modified_count);
        for _ in 0..modified_count {
            if pos + 64 > data.len() {
                return Err(DeltaSyncError::InvalidManifest("truncated modified pair".into()));
            }
            let mut old_bytes = [0u8; 32];
            old_bytes.copy_from_slice(&data[pos..pos + 32]);
            pos += 32;

            let mut new_bytes = [0u8; 32];
            new_bytes.copy_from_slice(&data[pos..pos + 32]);
            pos += 32;

            modified.push((
                ContentHash::from_raw(old_bytes),
                ContentHash::from_raw(new_bytes),
            ));
        }

        Ok(Self {
            added,
            removed,
            modified,
            unchanged_count,
            source_version,
            target_version,
        })
    }
}

// ---------------------------------------------------------------------------
// Transfer Chunk
// ---------------------------------------------------------------------------

/// A chunk of data for network transfer.
///
/// Supports chunked upload/download with resume capability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferChunk {
    /// Content hash this chunk belongs to.
    pub content_hash: ContentHash,
    /// Chunk index (0-based).
    pub chunk_index: usize,
    /// Total number of chunks for this content.
    pub total_chunks: usize,
    /// Byte offset within the full content.
    pub offset: u64,
    /// Chunk data.
    pub data: Vec<u8>,
    /// Hash of this chunk (for verification).
    pub chunk_hash: ContentHash,
}

impl TransferChunk {
    /// Create a new transfer chunk.
    pub fn new(
        content_hash: ContentHash,
        chunk_index: usize,
        total_chunks: usize,
        offset: u64,
        data: Vec<u8>,
    ) -> Self {
        let chunk_hash = ContentHash::from_bytes(&data);
        Self {
            content_hash,
            chunk_index,
            total_chunks,
            offset,
            data,
            chunk_hash,
        }
    }

    /// Verify the chunk hash.
    pub fn verify(&self) -> bool {
        ContentHash::from_bytes(&self.data) == self.chunk_hash
    }

    /// Check if this is the first chunk.
    pub fn is_first(&self) -> bool {
        self.chunk_index == 0
    }

    /// Check if this is the last chunk.
    pub fn is_last(&self) -> bool {
        self.chunk_index + 1 == self.total_chunks
    }

    /// Get the data length.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if chunk is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut result = Vec::new();

        // Magic
        result.extend_from_slice(b"CHNK");

        // Content hash
        result.extend_from_slice(self.content_hash.as_bytes());

        // Chunk metadata
        result.extend_from_slice(&(self.chunk_index as u32).to_le_bytes());
        result.extend_from_slice(&(self.total_chunks as u32).to_le_bytes());
        result.extend_from_slice(&self.offset.to_le_bytes());

        // Chunk hash
        result.extend_from_slice(self.chunk_hash.as_bytes());

        // Data length and data
        result.extend_from_slice(&(self.data.len() as u32).to_le_bytes());
        result.extend_from_slice(&self.data);

        result
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> DeltaSyncResult<Self> {
        if data.len() < 4 || &data[0..4] != b"CHNK" {
            return Err(DeltaSyncError::InvalidManifest("invalid chunk magic".into()));
        }

        let mut pos = 4;

        // Content hash
        if pos + 32 > data.len() {
            return Err(DeltaSyncError::InvalidManifest("truncated chunk content hash".into()));
        }
        let mut content_bytes = [0u8; 32];
        content_bytes.copy_from_slice(&data[pos..pos + 32]);
        let content_hash = ContentHash::from_raw(content_bytes);
        pos += 32;

        // Metadata
        if pos + 16 > data.len() {
            return Err(DeltaSyncError::InvalidManifest("truncated chunk metadata".into()));
        }
        let chunk_index =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        let total_chunks =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        let offset = u64::from_le_bytes([
            data[pos],
            data[pos + 1],
            data[pos + 2],
            data[pos + 3],
            data[pos + 4],
            data[pos + 5],
            data[pos + 6],
            data[pos + 7],
        ]);
        pos += 8;

        // Chunk hash
        if pos + 32 > data.len() {
            return Err(DeltaSyncError::InvalidManifest("truncated chunk hash".into()));
        }
        let mut chunk_hash_bytes = [0u8; 32];
        chunk_hash_bytes.copy_from_slice(&data[pos..pos + 32]);
        let chunk_hash = ContentHash::from_raw(chunk_hash_bytes);
        pos += 32;

        // Data
        if pos + 4 > data.len() {
            return Err(DeltaSyncError::InvalidManifest("truncated chunk data length".into()));
        }
        let data_len =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;

        if pos + data_len > data.len() {
            return Err(DeltaSyncError::InvalidManifest("truncated chunk data".into()));
        }
        let chunk_data = data[pos..pos + data_len].to_vec();

        Ok(Self {
            content_hash,
            chunk_index,
            total_chunks,
            offset,
            data: chunk_data,
            chunk_hash,
        })
    }
}

// ---------------------------------------------------------------------------
// Transfer State (for resume support)
// ---------------------------------------------------------------------------

/// State of an in-progress transfer for resume support.
#[derive(Debug, Clone)]
pub struct TransferState {
    /// Content hash being transferred.
    pub content_hash: ContentHash,
    /// Total size in bytes.
    pub total_size: u64,
    /// Bytes transferred so far.
    pub bytes_transferred: u64,
    /// Chunks received (by index).
    pub chunks_received: HashSet<usize>,
    /// Total number of chunks.
    pub total_chunks: usize,
    /// Timestamp when transfer started.
    pub started_at: u64,
    /// Timestamp of last activity.
    pub last_activity: u64,
}

impl TransferState {
    /// Create a new transfer state.
    pub fn new(content_hash: ContentHash, total_size: u64, total_chunks: usize) -> Self {
        let now = current_timestamp_ms();
        Self {
            content_hash,
            total_size,
            bytes_transferred: 0,
            chunks_received: HashSet::new(),
            total_chunks,
            started_at: now,
            last_activity: now,
        }
    }

    /// Record that a chunk was received.
    pub fn record_chunk(&mut self, chunk_index: usize, chunk_size: u64) {
        self.chunks_received.insert(chunk_index);
        self.bytes_transferred += chunk_size;
        self.last_activity = current_timestamp_ms();
    }

    /// Check if transfer is complete.
    pub fn is_complete(&self) -> bool {
        self.chunks_received.len() == self.total_chunks
    }

    /// Get completion percentage.
    pub fn completion_percent(&self) -> f64 {
        if self.total_chunks == 0 {
            100.0
        } else {
            self.chunks_received.len() as f64 / self.total_chunks as f64 * 100.0
        }
    }

    /// Get missing chunk indices.
    pub fn missing_chunks(&self) -> Vec<usize> {
        (0..self.total_chunks)
            .filter(|i| !self.chunks_received.contains(i))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Version Vector (for distributed conflict resolution)
// ---------------------------------------------------------------------------

/// Version vector for distributed conflict detection.
///
/// Each node maintains a vector of version numbers, one per known node.
/// Used to detect concurrent modifications in distributed environments.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VersionVector {
    /// Version counts per node ID.
    versions: HashMap<String, u64>,
}

impl VersionVector {
    /// Create a new empty version vector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Increment the version for a node.
    pub fn increment(&mut self, node_id: &str) {
        let counter = self.versions.entry(node_id.to_string()).or_insert(0);
        *counter += 1;
    }

    /// Get the version for a node.
    pub fn get(&self, node_id: &str) -> u64 {
        self.versions.get(node_id).copied().unwrap_or(0)
    }

    /// Merge with another version vector (take max of each).
    pub fn merge(&mut self, other: &VersionVector) {
        for (node, &version) in &other.versions {
            let current = self.versions.entry(node.clone()).or_insert(0);
            *current = (*current).max(version);
        }
    }

    /// Check if this vector happens-before another.
    ///
    /// Returns `true` if all versions in `self` are <= corresponding versions in `other`,
    /// and at least one is strictly less.
    pub fn happens_before(&self, other: &VersionVector) -> bool {
        let mut all_lte = true;
        let mut any_lt = false;

        for (node, &version) in &self.versions {
            let other_version = other.get(node);
            if version > other_version {
                all_lte = false;
                break;
            }
            if version < other_version {
                any_lt = true;
            }
        }

        // Check versions in other that aren't in self
        for (node, &version) in &other.versions {
            if !self.versions.contains_key(node) && version > 0 {
                any_lt = true;
            }
        }

        all_lte && any_lt
    }

    /// Check if vectors are concurrent (neither happens-before the other).
    pub fn concurrent_with(&self, other: &VersionVector) -> bool {
        !self.happens_before(other) && !other.happens_before(self) && self != other
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut data = Vec::new();

        data.extend_from_slice(&(self.versions.len() as u32).to_le_bytes());
        for (node, &version) in &self.versions {
            data.extend_from_slice(&(node.len() as u32).to_le_bytes());
            data.extend_from_slice(node.as_bytes());
            data.extend_from_slice(&version.to_le_bytes());
        }

        data
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> DeltaSyncResult<Self> {
        if data.len() < 4 {
            return Err(DeltaSyncError::InvalidManifest("version vector too short".into()));
        }

        let count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let mut pos = 4;
        let mut versions = HashMap::with_capacity(count);

        for _ in 0..count {
            if pos + 4 > data.len() {
                return Err(DeltaSyncError::InvalidManifest("truncated version vector".into()));
            }
            let node_len =
                u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                    as usize;
            pos += 4;

            if pos + node_len + 8 > data.len() {
                return Err(DeltaSyncError::InvalidManifest("truncated version entry".into()));
            }

            let node = String::from_utf8(data[pos..pos + node_len].to_vec())
                .map_err(|_| DeltaSyncError::InvalidManifest("invalid UTF-8 in node id".into()))?;
            pos += node_len;

            let version = u64::from_le_bytes([
                data[pos],
                data[pos + 1],
                data[pos + 2],
                data[pos + 3],
                data[pos + 4],
                data[pos + 5],
                data[pos + 6],
                data[pos + 7],
            ]);
            pos += 8;

            versions.insert(node, version);
        }

        Ok(Self { versions })
    }
}

// ---------------------------------------------------------------------------
// DeltaSync Core
// ---------------------------------------------------------------------------

/// Core delta synchronization engine.
///
/// Provides incremental asset synchronization with O(differences) complexity.
pub struct DeltaSync<S: ContentStore> {
    /// Configuration.
    config: DeltaSyncConfig,
    /// Local content store.
    local_store: Arc<S>,
    /// Current manifest.
    manifest: SyncManifest,
    /// In-progress transfers.
    transfers: HashMap<ContentHash, TransferState>,
    /// Version vector for this node.
    version_vector: VersionVector,
    /// Local node ID.
    node_id: String,
}

impl<S: ContentStore> DeltaSync<S> {
    /// Create a new delta sync instance.
    pub fn new(store: Arc<S>, config: DeltaSyncConfig) -> Self {
        let node_id = format!("node_{}", current_timestamp_ms() % 100000);
        Self {
            config,
            local_store: store,
            manifest: SyncManifest::new(),
            transfers: HashMap::new(),
            version_vector: VersionVector::new(),
            node_id,
        }
    }

    /// Create with a specific node ID.
    pub fn with_node_id(store: Arc<S>, config: DeltaSyncConfig, node_id: impl Into<String>) -> Self {
        let node_id = node_id.into();
        let mut manifest = SyncManifest::new();
        manifest.node_id = Some(node_id.clone());
        Self {
            config,
            local_store: store,
            manifest,
            transfers: HashMap::new(),
            version_vector: VersionVector::new(),
            node_id,
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &DeltaSyncConfig {
        &self.config
    }

    /// Get the local store.
    pub fn local_store(&self) -> &Arc<S> {
        &self.local_store
    }

    /// Get the current manifest.
    pub fn manifest(&self) -> &SyncManifest {
        &self.manifest
    }

    /// Get the version vector.
    pub fn version_vector(&self) -> &VersionVector {
        &self.version_vector
    }

    /// Compute the current sync manifest from the local store.
    pub fn compute_manifest(&mut self) -> SyncManifest {
        // In a real implementation, this would scan the store
        // For now, return the current manifest with updated timestamp
        self.manifest.timestamp = current_timestamp_ms();
        self.manifest.version += 1;
        self.version_vector.increment(&self.node_id);
        self.manifest.clone()
    }

    /// Add content to the local manifest.
    pub fn add_content(&mut self, hash: ContentHash) {
        self.manifest.add_hash(hash);
        self.manifest.version += 1;
        self.version_vector.increment(&self.node_id);
    }

    /// Remove content from the local manifest.
    pub fn remove_content(&mut self, hash: &ContentHash) {
        self.manifest.remove_hash(hash);
        self.manifest.version += 1;
        self.version_vector.increment(&self.node_id);
    }

    /// Compute the delta between local and remote manifests.
    ///
    /// Returns an O(differences) proof of what changed.
    pub fn compute_delta(&self, remote: &SyncManifest) -> DeltaProof {
        let local_set = &self.manifest.local_hashes;
        let remote_set = &remote.local_hashes;

        // Added: in remote but not local
        let added: Vec<_> = remote_set.difference(local_set).copied().collect();

        // Removed: in local but not remote
        let removed: Vec<_> = local_set.difference(remote_set).copied().collect();

        // Unchanged: in both
        let unchanged_count = local_set.intersection(remote_set).count();

        // Modified: for content-addressed storage, we track this separately
        // In practice, "modified" means the same logical asset has different content
        // This would require an asset ID -> content hash mapping
        let modified = Vec::new();

        DeltaProof {
            added,
            removed,
            modified,
            unchanged_count,
            source_version: self.manifest.version,
            target_version: remote.version,
        }
    }

    /// Prepare chunks for uploading content based on a delta.
    ///
    /// Returns chunks for all added/modified content.
    pub fn prepare_upload(&self, delta: &DeltaProof) -> DeltaSyncResult<Vec<TransferChunk>> {
        let mut chunks = Vec::new();

        for hash in &delta.added {
            let content_chunks = self.chunk_content(hash)?;
            chunks.extend(content_chunks);
        }

        for (_, new_hash) in &delta.modified {
            let content_chunks = self.chunk_content(new_hash)?;
            chunks.extend(content_chunks);
        }

        Ok(chunks)
    }

    /// Chunk content from the local store.
    fn chunk_content(&self, hash: &ContentHash) -> DeltaSyncResult<Vec<TransferChunk>> {
        let reader = self
            .local_store
            .get_stream(hash)
            .map_err(|e| DeltaSyncError::IoError(e.to_string()))?
            .ok_or_else(|| DeltaSyncError::ContentNotFound(*hash))?;

        let size = reader.size();
        let chunk_size = self.config.chunk_size;
        let total_chunks = ((size as usize + chunk_size - 1) / chunk_size).max(1);

        let mut chunks = Vec::with_capacity(total_chunks);
        let mut offset = 0u64;
        let mut chunk_index = 0;

        // Read the full content (in a real impl, we'd stream)
        let mut all_data = Vec::new();
        let mut buf_reader = reader;
        let mut buf = [0u8; 8192];
        loop {
            let n = buf_reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            all_data.extend_from_slice(&buf[..n]);
        }

        // Split into chunks
        for chunk_data in all_data.chunks(chunk_size) {
            let chunk = TransferChunk::new(
                *hash,
                chunk_index,
                total_chunks,
                offset,
                chunk_data.to_vec(),
            );
            chunks.push(chunk);
            offset += chunk_data.len() as u64;
            chunk_index += 1;
        }

        // Handle empty content
        if chunks.is_empty() {
            chunks.push(TransferChunk::new(*hash, 0, 1, 0, Vec::new()));
        }

        Ok(chunks)
    }

    /// Apply downloaded chunks to the local store.
    pub fn apply_download(&mut self, chunks: Vec<TransferChunk>) -> DeltaSyncResult<Vec<ContentHash>> {
        // Group chunks by content hash
        let mut by_content: HashMap<ContentHash, Vec<TransferChunk>> = HashMap::new();
        for chunk in chunks {
            by_content
                .entry(chunk.content_hash)
                .or_default()
                .push(chunk);
        }

        let mut completed = Vec::new();

        for (content_hash, mut content_chunks) in by_content {
            // Verify all chunks received
            content_chunks.sort_by_key(|c| c.chunk_index);

            if content_chunks.is_empty() {
                continue;
            }

            let total_chunks = content_chunks[0].total_chunks;
            if content_chunks.len() != total_chunks {
                return Err(DeltaSyncError::TransferInterrupted {
                    hash: content_hash,
                    bytes_transferred: content_chunks.iter().map(|c| c.data.len() as u64).sum(),
                    total_bytes: 0, // Unknown
                });
            }

            // Verify chunk indices
            for (i, chunk) in content_chunks.iter().enumerate() {
                if chunk.chunk_index != i {
                    return Err(DeltaSyncError::InvalidChunkIndex {
                        index: chunk.chunk_index,
                        total_chunks,
                    });
                }

                if self.config.verify_transfers && !chunk.verify() {
                    return Err(DeltaSyncError::HashMismatch {
                        expected: chunk.chunk_hash,
                        actual: ContentHash::from_bytes(&chunk.data),
                    });
                }
            }

            // Reassemble content
            let mut full_data = Vec::new();
            for chunk in &content_chunks {
                full_data.extend_from_slice(&chunk.data);
            }

            // Verify full content hash
            let computed_hash = ContentHash::from_bytes(&full_data);
            if computed_hash != content_hash {
                return Err(DeltaSyncError::HashMismatch {
                    expected: content_hash,
                    actual: computed_hash,
                });
            }

            // Store content
            let mut cursor = std::io::Cursor::new(&full_data);
            self.local_store
                .put_stream(&mut cursor)
                .map_err(|e| DeltaSyncError::IoError(e.to_string()))?;

            // Update manifest
            self.manifest.add_hash(content_hash);
            completed.push(content_hash);
        }

        if !completed.is_empty() {
            self.manifest.version += 1;
            self.version_vector.increment(&self.node_id);
        }

        Ok(completed)
    }

    /// Resolve a conflict between local and remote content hashes.
    pub fn resolve_conflict(
        &self,
        local: &ContentHash,
        remote: &ContentHash,
        local_timestamp: u64,
        remote_timestamp: u64,
    ) -> DeltaSyncResult<ContentHash> {
        match self.config.conflict_strategy {
            ConflictStrategy::LatestWins => {
                if local_timestamp >= remote_timestamp {
                    Ok(*local)
                } else {
                    Ok(*remote)
                }
            }
            ConflictStrategy::LocalWins => Ok(*local),
            ConflictStrategy::RemoteWins => Ok(*remote),
            ConflictStrategy::Manual => Err(DeltaSyncError::ConflictRequiresManual {
                local: *local,
                remote: *remote,
            }),
            ConflictStrategy::VersionVector => {
                // Would need version vectors for both to make this decision
                // Default to latest timestamp if no version info
                if local_timestamp >= remote_timestamp {
                    Ok(*local)
                } else {
                    Ok(*remote)
                }
            }
        }
    }

    /// Check if content is platform-specific.
    pub fn detect_platform_specific(&self, hash: &ContentHash) -> bool {
        self.manifest.is_platform_specific(hash)
    }

    /// Mark content as platform-specific.
    pub fn mark_platform_specific(&mut self, hash: ContentHash, platform: Platform) {
        self.manifest.mark_platform_specific(hash, platform);
    }

    /// Compute delta for shared (non-platform-specific) content only.
    pub fn compute_shared_delta(&self, _platforms: &[Platform], remote: &SyncManifest) -> DeltaProof {
        // Get shared hashes from local
        let local_shared: HashSet<_> = self.manifest.shared_hashes().into_iter().collect();

        // Get shared hashes from remote
        let remote_shared: HashSet<_> = remote.shared_hashes().into_iter().collect();

        // Compute delta on shared content only
        let added: Vec<_> = remote_shared.difference(&local_shared).copied().collect();
        let removed: Vec<_> = local_shared.difference(&remote_shared).copied().collect();
        let unchanged_count = local_shared.intersection(&remote_shared).count();

        DeltaProof {
            added,
            removed,
            modified: Vec::new(),
            unchanged_count,
            source_version: self.manifest.version,
            target_version: remote.version,
        }
    }

    /// Start a new transfer (for resume support).
    pub fn start_transfer(
        &mut self,
        content_hash: ContentHash,
        total_size: u64,
        total_chunks: usize,
    ) -> &TransferState {
        let state = TransferState::new(content_hash, total_size, total_chunks);
        self.transfers.insert(content_hash, state);
        self.transfers.get(&content_hash).unwrap()
    }

    /// Record a received chunk.
    pub fn record_chunk_received(
        &mut self,
        content_hash: &ContentHash,
        chunk_index: usize,
        chunk_size: u64,
    ) -> Option<&TransferState> {
        if let Some(state) = self.transfers.get_mut(content_hash) {
            state.record_chunk(chunk_index, chunk_size);
            Some(state)
        } else {
            None
        }
    }

    /// Get transfer state.
    pub fn get_transfer_state(&self, content_hash: &ContentHash) -> Option<&TransferState> {
        self.transfers.get(content_hash)
    }

    /// Check if a transfer is complete.
    pub fn is_transfer_complete(&self, content_hash: &ContentHash) -> bool {
        self.transfers
            .get(content_hash)
            .map(|s| s.is_complete())
            .unwrap_or(false)
    }

    /// Get missing chunks for a transfer.
    pub fn get_missing_chunks(&self, content_hash: &ContentHash) -> Option<Vec<usize>> {
        self.transfers.get(content_hash).map(|s| s.missing_chunks())
    }

    /// Clear completed transfers.
    pub fn clear_completed_transfers(&mut self) {
        self.transfers.retain(|_, state| !state.is_complete());
    }
}

// ---------------------------------------------------------------------------
// Utility Functions
// ---------------------------------------------------------------------------

/// Get current timestamp in milliseconds since UNIX epoch.
fn current_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Compute delta between two hash sets efficiently.
pub fn compute_set_delta(
    local: &HashSet<ContentHash>,
    remote: &HashSet<ContentHash>,
) -> (Vec<ContentHash>, Vec<ContentHash>, usize) {
    let added: Vec<_> = remote.difference(local).copied().collect();
    let removed: Vec<_> = local.difference(remote).copied().collect();
    let unchanged = local.intersection(remote).count();
    (added, removed, unchanged)
}

/// Estimate transfer size for a delta.
pub fn estimate_transfer_size(delta: &DeltaProof, avg_content_size: u64) -> u64 {
    (delta.added.len() as u64 + delta.modified.len() as u64) * avg_content_size
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::content_store::MemoryContentStore;
    use std::io::Cursor;

    // ========================================================================
    // Configuration tests
    // ========================================================================

    #[test]
    fn test_default_config() {
        let config = DeltaSyncConfig::default();
        assert_eq!(config.chunk_size, DEFAULT_CHUNK_SIZE);
        assert_eq!(config.max_concurrent_transfers, DEFAULT_MAX_CONCURRENT);
        assert_eq!(config.conflict_strategy, ConflictStrategy::LatestWins);
        assert!(config.verify_transfers);
        assert!(!config.compress_chunks);
    }

    #[test]
    fn test_config_with_chunk_size() {
        let config = DeltaSyncConfig::default().with_chunk_size(512 * 1024);
        assert_eq!(config.chunk_size, 512 * 1024);
    }

    #[test]
    fn test_config_chunk_size_clamped_min() {
        let config = DeltaSyncConfig::default().with_chunk_size(1024);
        assert_eq!(config.chunk_size, MIN_CHUNK_SIZE);
    }

    #[test]
    fn test_config_chunk_size_clamped_max() {
        let config = DeltaSyncConfig::default().with_chunk_size(100 * 1024 * 1024);
        assert_eq!(config.chunk_size, MAX_CHUNK_SIZE);
    }

    #[test]
    fn test_config_with_conflict_strategy() {
        let config = DeltaSyncConfig::default().with_conflict_strategy(ConflictStrategy::Manual);
        assert_eq!(config.conflict_strategy, ConflictStrategy::Manual);
    }

    // ========================================================================
    // SyncManifest tests
    // ========================================================================

    #[test]
    fn test_manifest_new() {
        let manifest = SyncManifest::new();
        assert!(manifest.is_empty());
        assert_eq!(manifest.version, 0);
        assert!(manifest.timestamp > 0);
    }

    #[test]
    fn test_manifest_with_hashes() {
        let hash1 = ContentHash::from_bytes(b"content1");
        let hash2 = ContentHash::from_bytes(b"content2");
        let mut hashes = HashSet::new();
        hashes.insert(hash1);
        hashes.insert(hash2);

        let manifest = SyncManifest::with_hashes(hashes);
        assert_eq!(manifest.len(), 2);
        assert!(manifest.contains(&hash1));
        assert!(manifest.contains(&hash2));
    }

    #[test]
    fn test_manifest_add_remove() {
        let mut manifest = SyncManifest::new();
        let hash = ContentHash::from_bytes(b"content");

        manifest.add_hash(hash);
        assert!(manifest.contains(&hash));
        assert_eq!(manifest.len(), 1);

        let removed = manifest.remove_hash(&hash);
        assert!(removed);
        assert!(!manifest.contains(&hash));
        assert!(manifest.is_empty());
    }

    #[test]
    fn test_manifest_platform_markers() {
        let mut manifest = SyncManifest::new();
        let hash = ContentHash::from_bytes(b"windows_texture");

        manifest.add_hash(hash);
        manifest.mark_platform_specific(hash, Platform::Windows);

        assert!(manifest.is_platform_specific(&hash));
        assert_eq!(manifest.get_platform(&hash), Some(Platform::Windows));

        let windows_hashes = manifest.hashes_for_platform(Platform::Windows);
        assert_eq!(windows_hashes.len(), 1);
        assert_eq!(windows_hashes[0], hash);

        // Non-platform-specific hashes
        let shared_hash = ContentHash::from_bytes(b"shared");
        manifest.add_hash(shared_hash);

        let shared = manifest.shared_hashes();
        assert_eq!(shared.len(), 1);
        assert_eq!(shared[0], shared_hash);
    }

    #[test]
    fn test_manifest_serialization() {
        let mut manifest = SyncManifest::new()
            .with_version(42)
            .with_node_id("test_node");

        let hash1 = ContentHash::from_bytes(b"content1");
        let hash2 = ContentHash::from_bytes(b"content2");
        manifest.add_hash(hash1);
        manifest.add_hash(hash2);
        manifest.mark_platform_specific(hash1, Platform::Linux);

        let bytes = manifest.to_bytes();
        let restored = SyncManifest::from_bytes(&bytes).unwrap();

        assert_eq!(restored.version, 42);
        assert_eq!(restored.node_id, Some("test_node".to_string()));
        assert_eq!(restored.len(), 2);
        assert!(restored.contains(&hash1));
        assert!(restored.contains(&hash2));
        assert_eq!(restored.get_platform(&hash1), Some(Platform::Linux));
    }

    // ========================================================================
    // DeltaProof tests
    // ========================================================================

    #[test]
    fn test_delta_proof_empty() {
        let delta = DeltaProof::empty();
        assert!(delta.is_empty());
        assert_eq!(delta.change_count(), 0);
    }

    #[test]
    fn test_delta_proof_changes() {
        let delta = DeltaProof {
            added: vec![
                ContentHash::from_bytes(b"new1"),
                ContentHash::from_bytes(b"new2"),
            ],
            removed: vec![ContentHash::from_bytes(b"old1")],
            modified: vec![(
                ContentHash::from_bytes(b"old_ver"),
                ContentHash::from_bytes(b"new_ver"),
            )],
            unchanged_count: 10,
            source_version: 1,
            target_version: 2,
        };

        assert!(!delta.is_empty());
        assert_eq!(delta.change_count(), 4);
        assert_eq!(delta.total_count(), 14);
        assert!((delta.change_ratio() - 4.0 / 14.0).abs() < 0.001);
    }

    #[test]
    fn test_delta_proof_serialization() {
        let delta = DeltaProof {
            added: vec![ContentHash::from_bytes(b"added")],
            removed: vec![ContentHash::from_bytes(b"removed")],
            modified: vec![(
                ContentHash::from_bytes(b"old"),
                ContentHash::from_bytes(b"new"),
            )],
            unchanged_count: 5,
            source_version: 10,
            target_version: 20,
        };

        let bytes = delta.to_bytes();
        let restored = DeltaProof::from_bytes(&bytes).unwrap();

        assert_eq!(delta.added, restored.added);
        assert_eq!(delta.removed, restored.removed);
        assert_eq!(delta.modified, restored.modified);
        assert_eq!(delta.unchanged_count, restored.unchanged_count);
        assert_eq!(delta.source_version, restored.source_version);
        assert_eq!(delta.target_version, restored.target_version);
    }

    // ========================================================================
    // TransferChunk tests
    // ========================================================================

    #[test]
    fn test_transfer_chunk_new() {
        let content_hash = ContentHash::from_bytes(b"full_content");
        let data = b"chunk data here".to_vec();

        let chunk = TransferChunk::new(content_hash, 0, 3, 0, data.clone());

        assert_eq!(chunk.content_hash, content_hash);
        assert_eq!(chunk.chunk_index, 0);
        assert_eq!(chunk.total_chunks, 3);
        assert_eq!(chunk.offset, 0);
        assert_eq!(chunk.data, data);
        assert!(chunk.verify());
        assert!(chunk.is_first());
        assert!(!chunk.is_last());
    }

    #[test]
    fn test_transfer_chunk_last() {
        let content_hash = ContentHash::from_bytes(b"content");
        let chunk = TransferChunk::new(content_hash, 2, 3, 200, b"last".to_vec());

        assert!(!chunk.is_first());
        assert!(chunk.is_last());
    }

    #[test]
    fn test_transfer_chunk_verify() {
        let content_hash = ContentHash::from_bytes(b"content");
        let mut chunk = TransferChunk::new(content_hash, 0, 1, 0, b"original".to_vec());

        assert!(chunk.verify());

        // Corrupt the data
        chunk.data = b"corrupted".to_vec();
        assert!(!chunk.verify());
    }

    #[test]
    fn test_transfer_chunk_serialization() {
        let content_hash = ContentHash::from_bytes(b"content");
        let chunk = TransferChunk::new(content_hash, 1, 5, 1024, b"chunk data".to_vec());

        let bytes = chunk.to_bytes();
        let restored = TransferChunk::from_bytes(&bytes).unwrap();

        assert_eq!(chunk.content_hash, restored.content_hash);
        assert_eq!(chunk.chunk_index, restored.chunk_index);
        assert_eq!(chunk.total_chunks, restored.total_chunks);
        assert_eq!(chunk.offset, restored.offset);
        assert_eq!(chunk.data, restored.data);
        assert_eq!(chunk.chunk_hash, restored.chunk_hash);
    }

    // ========================================================================
    // TransferState tests
    // ========================================================================

    #[test]
    fn test_transfer_state_new() {
        let hash = ContentHash::from_bytes(b"content");
        let state = TransferState::new(hash, 1000, 10);

        assert_eq!(state.content_hash, hash);
        assert_eq!(state.total_size, 1000);
        assert_eq!(state.bytes_transferred, 0);
        assert_eq!(state.total_chunks, 10);
        assert!(!state.is_complete());
        assert_eq!(state.missing_chunks().len(), 10);
    }

    #[test]
    fn test_transfer_state_record_chunk() {
        let hash = ContentHash::from_bytes(b"content");
        let mut state = TransferState::new(hash, 300, 3);

        state.record_chunk(0, 100);
        assert_eq!(state.bytes_transferred, 100);
        assert_eq!(state.chunks_received.len(), 1);
        assert!(!state.is_complete());

        state.record_chunk(1, 100);
        state.record_chunk(2, 100);
        assert!(state.is_complete());
        assert_eq!(state.completion_percent(), 100.0);
    }

    #[test]
    fn test_transfer_state_missing_chunks() {
        let hash = ContentHash::from_bytes(b"content");
        let mut state = TransferState::new(hash, 500, 5);

        state.record_chunk(0, 100);
        state.record_chunk(2, 100);
        state.record_chunk(4, 100);

        let missing = state.missing_chunks();
        assert_eq!(missing, vec![1, 3]);
    }

    // ========================================================================
    // VersionVector tests
    // ========================================================================

    #[test]
    fn test_version_vector_new() {
        let vv = VersionVector::new();
        assert_eq!(vv.get("node1"), 0);
    }

    #[test]
    fn test_version_vector_increment() {
        let mut vv = VersionVector::new();
        vv.increment("node1");
        assert_eq!(vv.get("node1"), 1);

        vv.increment("node1");
        assert_eq!(vv.get("node1"), 2);

        vv.increment("node2");
        assert_eq!(vv.get("node2"), 1);
    }

    #[test]
    fn test_version_vector_merge() {
        let mut vv1 = VersionVector::new();
        vv1.increment("node1");
        vv1.increment("node1");

        let mut vv2 = VersionVector::new();
        vv2.increment("node1");
        vv2.increment("node2");
        vv2.increment("node2");

        vv1.merge(&vv2);
        assert_eq!(vv1.get("node1"), 2);
        assert_eq!(vv1.get("node2"), 2);
    }

    #[test]
    fn test_version_vector_happens_before() {
        let mut vv1 = VersionVector::new();
        vv1.increment("node1");

        let mut vv2 = VersionVector::new();
        vv2.increment("node1");
        vv2.increment("node1");

        assert!(vv1.happens_before(&vv2));
        assert!(!vv2.happens_before(&vv1));
    }

    #[test]
    fn test_version_vector_concurrent() {
        let mut vv1 = VersionVector::new();
        vv1.increment("node1");

        let mut vv2 = VersionVector::new();
        vv2.increment("node2");

        assert!(vv1.concurrent_with(&vv2));
        assert!(vv2.concurrent_with(&vv1));
    }

    #[test]
    fn test_version_vector_serialization() {
        let mut vv = VersionVector::new();
        vv.increment("node1");
        vv.increment("node1");
        vv.increment("node2");

        let bytes = vv.to_bytes();
        let restored = VersionVector::from_bytes(&bytes).unwrap();

        assert_eq!(restored.get("node1"), 2);
        assert_eq!(restored.get("node2"), 1);
    }

    // ========================================================================
    // DeltaSync core tests
    // ========================================================================

    #[test]
    fn test_delta_sync_new() {
        let store = Arc::new(MemoryContentStore::default());
        let config = DeltaSyncConfig::default();
        let sync = DeltaSync::new(store, config);

        assert!(sync.manifest().is_empty());
    }

    #[test]
    fn test_delta_sync_add_content() {
        let store = Arc::new(MemoryContentStore::default());
        let config = DeltaSyncConfig::default();
        let mut sync = DeltaSync::new(store, config);

        let hash = ContentHash::from_bytes(b"content");
        sync.add_content(hash);

        assert!(sync.manifest().contains(&hash));
        assert!(sync.manifest().version > 0);
    }

    #[test]
    fn test_delta_sync_compute_delta() {
        let store = Arc::new(MemoryContentStore::default());
        let config = DeltaSyncConfig::default();
        let mut sync = DeltaSync::new(store, config);

        // Local has hash1 and hash2
        let hash1 = ContentHash::from_bytes(b"content1");
        let hash2 = ContentHash::from_bytes(b"content2");
        sync.add_content(hash1);
        sync.add_content(hash2);

        // Remote has hash2 and hash3
        let hash3 = ContentHash::from_bytes(b"content3");
        let mut remote_hashes = HashSet::new();
        remote_hashes.insert(hash2);
        remote_hashes.insert(hash3);
        let remote = SyncManifest::with_hashes(remote_hashes);

        let delta = sync.compute_delta(&remote);

        // hash3 is added (in remote, not local)
        assert_eq!(delta.added, vec![hash3]);
        // hash1 is removed (in local, not remote)
        assert_eq!(delta.removed, vec![hash1]);
        // hash2 is unchanged
        assert_eq!(delta.unchanged_count, 1);
    }

    #[test]
    fn test_delta_sync_platform_specific() {
        let store = Arc::new(MemoryContentStore::default());
        let config = DeltaSyncConfig::default();
        let mut sync = DeltaSync::new(store, config);

        let hash = ContentHash::from_bytes(b"texture");
        sync.add_content(hash);
        sync.mark_platform_specific(hash, Platform::Android);

        assert!(sync.detect_platform_specific(&hash));
    }

    #[test]
    fn test_delta_sync_resolve_conflict_latest_wins() {
        let store = Arc::new(MemoryContentStore::default());
        let config = DeltaSyncConfig::default().with_conflict_strategy(ConflictStrategy::LatestWins);
        let sync = DeltaSync::new(store, config);

        let local = ContentHash::from_bytes(b"local");
        let remote = ContentHash::from_bytes(b"remote");

        // Local is newer
        let result = sync.resolve_conflict(&local, &remote, 2000, 1000).unwrap();
        assert_eq!(result, local);

        // Remote is newer
        let result = sync.resolve_conflict(&local, &remote, 1000, 2000).unwrap();
        assert_eq!(result, remote);
    }

    #[test]
    fn test_delta_sync_resolve_conflict_local_wins() {
        let store = Arc::new(MemoryContentStore::default());
        let config = DeltaSyncConfig::default().with_conflict_strategy(ConflictStrategy::LocalWins);
        let sync = DeltaSync::new(store, config);

        let local = ContentHash::from_bytes(b"local");
        let remote = ContentHash::from_bytes(b"remote");

        let result = sync.resolve_conflict(&local, &remote, 1000, 2000).unwrap();
        assert_eq!(result, local);
    }

    #[test]
    fn test_delta_sync_resolve_conflict_manual() {
        let store = Arc::new(MemoryContentStore::default());
        let config = DeltaSyncConfig::default().with_conflict_strategy(ConflictStrategy::Manual);
        let sync = DeltaSync::new(store, config);

        let local = ContentHash::from_bytes(b"local");
        let remote = ContentHash::from_bytes(b"remote");

        let result = sync.resolve_conflict(&local, &remote, 1000, 2000);
        assert!(matches!(
            result,
            Err(DeltaSyncError::ConflictRequiresManual { .. })
        ));
    }

    // ========================================================================
    // Transfer flow tests
    // ========================================================================

    #[test]
    fn test_delta_sync_transfer_flow() {
        let store = Arc::new(MemoryContentStore::default());
        let config = DeltaSyncConfig::default().with_chunk_size(MIN_CHUNK_SIZE);
        let mut sync = DeltaSync::new(store.clone(), config);

        // Store some content
        let data = b"test content for chunking";
        let mut cursor = Cursor::new(data);
        let hash = store.put_stream(&mut cursor).unwrap();
        sync.add_content(hash);

        // Compute manifest
        let manifest = sync.compute_manifest();
        assert!(manifest.contains(&hash));

        // Create delta against empty remote
        let remote = SyncManifest::new();
        let delta = sync.compute_delta(&remote);

        // Remote doesn't have our content, so it's "removed" from their perspective
        // (they need to receive it)
        assert!(delta.removed.contains(&hash));
    }

    #[test]
    fn test_delta_sync_chunk_content() {
        let store = Arc::new(MemoryContentStore::default());
        let config = DeltaSyncConfig::default().with_chunk_size(MIN_CHUNK_SIZE);
        let sync = DeltaSync::new(store.clone(), config);

        // Store content larger than chunk size
        let data = vec![0u8; MIN_CHUNK_SIZE * 2 + 100];
        let mut cursor = Cursor::new(&data);
        let hash = store.put_stream(&mut cursor).unwrap();

        let chunks = sync.chunk_content(&hash).unwrap();

        assert_eq!(chunks.len(), 3);
        assert!(chunks[0].is_first());
        assert!(chunks[2].is_last());

        // Verify reassembly
        let mut reassembled = Vec::new();
        for chunk in &chunks {
            reassembled.extend_from_slice(&chunk.data);
        }
        assert_eq!(reassembled, data);
    }

    #[test]
    fn test_delta_sync_apply_download() {
        let store = Arc::new(MemoryContentStore::default());
        let config = DeltaSyncConfig::default();
        let mut sync = DeltaSync::new(store.clone(), config);

        // Create chunks for some content
        let data = b"content to download";
        let content_hash = ContentHash::from_bytes(data);

        let chunk = TransferChunk::new(content_hash, 0, 1, 0, data.to_vec());

        // Apply download
        let completed = sync.apply_download(vec![chunk]).unwrap();

        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0], content_hash);
        assert!(sync.manifest().contains(&content_hash));
        assert!(store.has(&content_hash));
    }

    #[test]
    fn test_delta_sync_apply_download_multi_chunk() {
        let store = Arc::new(MemoryContentStore::default());
        let config = DeltaSyncConfig::default();
        let mut sync = DeltaSync::new(store.clone(), config);

        let data = b"this is a longer piece of content for multi-chunk";
        let content_hash = ContentHash::from_bytes(data);

        // Split into chunks
        let chunk1 = TransferChunk::new(content_hash, 0, 2, 0, data[..25].to_vec());
        let chunk2 = TransferChunk::new(content_hash, 1, 2, 25, data[25..].to_vec());

        let completed = sync.apply_download(vec![chunk1, chunk2]).unwrap();

        assert_eq!(completed.len(), 1);
        assert!(store.has(&content_hash));
    }

    #[test]
    fn test_delta_sync_transfer_state_tracking() {
        let store = Arc::new(MemoryContentStore::default());
        let config = DeltaSyncConfig::default();
        let mut sync = DeltaSync::new(store, config);

        let hash = ContentHash::from_bytes(b"content");
        sync.start_transfer(hash, 1000, 10);

        assert!(!sync.is_transfer_complete(&hash));

        for i in 0..10 {
            sync.record_chunk_received(&hash, i, 100);
        }

        assert!(sync.is_transfer_complete(&hash));

        sync.clear_completed_transfers();
        assert!(sync.get_transfer_state(&hash).is_none());
    }

    // ========================================================================
    // Shared delta tests
    // ========================================================================

    #[test]
    fn test_delta_sync_compute_shared_delta() {
        let store = Arc::new(MemoryContentStore::default());
        let config = DeltaSyncConfig::default();
        let mut sync = DeltaSync::new(store, config);

        // Add shared and platform-specific content
        let shared_hash = ContentHash::from_bytes(b"shared");
        let windows_hash = ContentHash::from_bytes(b"windows");
        let linux_hash = ContentHash::from_bytes(b"linux");

        sync.add_content(shared_hash);
        sync.add_content(windows_hash);
        sync.add_content(linux_hash);
        sync.mark_platform_specific(windows_hash, Platform::Windows);
        sync.mark_platform_specific(linux_hash, Platform::Linux);

        // Remote has only new_shared
        let new_shared = ContentHash::from_bytes(b"new_shared");
        let mut remote_hashes = HashSet::new();
        remote_hashes.insert(new_shared);
        let remote = SyncManifest::with_hashes(remote_hashes);

        let delta = sync.compute_shared_delta(&[Platform::Windows, Platform::Linux], &remote);

        // Only shared content should be in the delta
        assert_eq!(delta.added, vec![new_shared]);
        assert_eq!(delta.removed, vec![shared_hash]);
    }

    // ========================================================================
    // Utility function tests
    // ========================================================================

    #[test]
    fn test_compute_set_delta() {
        let mut local = HashSet::new();
        local.insert(ContentHash::from_bytes(b"a"));
        local.insert(ContentHash::from_bytes(b"b"));

        let mut remote = HashSet::new();
        remote.insert(ContentHash::from_bytes(b"b"));
        remote.insert(ContentHash::from_bytes(b"c"));

        let (added, removed, unchanged) = compute_set_delta(&local, &remote);

        assert_eq!(added, vec![ContentHash::from_bytes(b"c")]);
        assert_eq!(removed, vec![ContentHash::from_bytes(b"a")]);
        assert_eq!(unchanged, 1);
    }

    #[test]
    fn test_estimate_transfer_size() {
        let delta = DeltaProof {
            added: vec![
                ContentHash::from_bytes(b"a"),
                ContentHash::from_bytes(b"b"),
            ],
            removed: vec![],
            modified: vec![(
                ContentHash::from_bytes(b"old"),
                ContentHash::from_bytes(b"new"),
            )],
            unchanged_count: 10,
            source_version: 1,
            target_version: 2,
        };

        let size = estimate_transfer_size(&delta, 1000);
        assert_eq!(size, 3000); // 2 added + 1 modified = 3 * 1000
    }

    // ========================================================================
    // Error handling tests
    // ========================================================================

    #[test]
    fn test_apply_download_hash_mismatch() {
        let store = Arc::new(MemoryContentStore::default());
        let config = DeltaSyncConfig::default();
        let mut sync = DeltaSync::new(store, config);

        let content_hash = ContentHash::from_bytes(b"expected");
        let wrong_data = b"wrong content";

        let chunk = TransferChunk::new(content_hash, 0, 1, 0, wrong_data.to_vec());

        let result = sync.apply_download(vec![chunk]);
        assert!(matches!(result, Err(DeltaSyncError::HashMismatch { .. })));
    }

    #[test]
    fn test_apply_download_missing_chunks() {
        let store = Arc::new(MemoryContentStore::default());
        let config = DeltaSyncConfig::default();
        let mut sync = DeltaSync::new(store, config);

        let content_hash = ContentHash::from_bytes(b"content");

        // Only provide chunk 0 of 3
        let chunk = TransferChunk::new(content_hash, 0, 3, 0, b"partial".to_vec());

        let result = sync.apply_download(vec![chunk]);
        assert!(matches!(
            result,
            Err(DeltaSyncError::TransferInterrupted { .. })
        ));
    }

    #[test]
    fn test_chunk_content_not_found() {
        let store = Arc::new(MemoryContentStore::default());
        let config = DeltaSyncConfig::default();
        let sync = DeltaSync::new(store, config);

        let missing_hash = ContentHash::from_bytes(b"missing");
        let result = sync.chunk_content(&missing_hash);

        assert!(matches!(result, Err(DeltaSyncError::ContentNotFound(_))));
    }

    // ========================================================================
    // Serialization edge cases
    // ========================================================================

    #[test]
    fn test_manifest_serialization_empty() {
        let manifest = SyncManifest::new();
        let bytes = manifest.to_bytes();
        let restored = SyncManifest::from_bytes(&bytes).unwrap();
        assert!(restored.is_empty());
    }

    #[test]
    fn test_manifest_invalid_magic() {
        let data = b"XXXX1234567890";
        let result = SyncManifest::from_bytes(data);
        assert!(result.is_err());
    }

    #[test]
    fn test_delta_proof_serialization_empty() {
        let delta = DeltaProof::empty();
        let bytes = delta.to_bytes();
        let restored = DeltaProof::from_bytes(&bytes).unwrap();
        assert!(restored.is_empty());
    }

    #[test]
    fn test_transfer_chunk_serialization_empty_data() {
        let content_hash = ContentHash::from_bytes(b"empty");
        let chunk = TransferChunk::new(content_hash, 0, 1, 0, Vec::new());

        let bytes = chunk.to_bytes();
        let restored = TransferChunk::from_bytes(&bytes).unwrap();

        assert!(restored.is_empty());
        assert!(restored.verify());
    }
}
