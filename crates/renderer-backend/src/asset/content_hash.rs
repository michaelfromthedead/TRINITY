//! BLAKE3 content hashing for the TRINITY asset pipeline.
//!
//! This module provides a comprehensive content hashing API with:
//! - BLAKE3 SIMD-parallel hashing (10x+ faster than SHA-256)
//! - Streaming incremental hashing for large files
//! - Keyed hashing mode for authenticated content
//! - Extendable output (256, 512 bits configurable)
//! - Dual-hash mode: store both SHA-256 and BLAKE3 during migration
//! - Hash function recorded in asset manifest
//!
//! # Performance Target
//!
//! Less than 5ms per 1MB of data on modern hardware.
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::asset::content_hash::{ContentHasher, HashOutput};
//!
//! // Basic BLAKE3 hashing
//! let hash = ContentHasher::blake3(b"hello world");
//!
//! // Streaming for large files
//! let mut hasher = ContentHasher::new_blake3_streaming();
//! hasher.update(b"chunk 1");
//! hasher.update(b"chunk 2");
//! let hash = hasher.finalize();
//!
//! // Keyed hashing for authenticated content
//! let key = [0u8; 32];
//! let mac = ContentHasher::blake3_keyed(&key, b"authenticated data");
//!
//! // Dual-hash mode for migration
//! let dual = ContentHasher::dual_hash(b"migrating content");
//! println!("BLAKE3: {}", dual.blake3);
//! println!("SHA256: {}", dual.sha256);
//! ```

use std::fmt;
use std::io::{self, Read};
use std::str::FromStr;

use sha2::{Digest as Sha2Digest, Sha256};

// Re-export from pipeline for backward compatibility
pub use crate::pipeline::{ContentHash, ContentHashParseError, HashAlgorithm};

// ---------------------------------------------------------------------------
// HashOutput — configurable output length
// ---------------------------------------------------------------------------

/// Output length configuration for BLAKE3 hashes.
///
/// BLAKE3 supports extendable output (XOF), allowing any output length.
/// Common sizes are 256 bits (32 bytes) and 512 bits (64 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HashOutputLength {
    /// 256-bit (32-byte) output — default, compatible with ContentHash
    Bits256,
    /// 512-bit (64-byte) output — extra collision resistance
    Bits512,
    /// Custom output length in bytes (must be > 0)
    Custom(usize),
}

impl HashOutputLength {
    /// Get the output length in bytes.
    pub const fn bytes(&self) -> usize {
        match self {
            Self::Bits256 => 32,
            Self::Bits512 => 64,
            Self::Custom(n) => *n,
        }
    }
}

impl Default for HashOutputLength {
    fn default() -> Self {
        Self::Bits256
    }
}

// ---------------------------------------------------------------------------
// ExtendedHash — variable-length hash output
// ---------------------------------------------------------------------------

/// A hash output with variable length (256, 512 bits, or custom).
///
/// Unlike [`ContentHash`] which is always 32 bytes, `ExtendedHash` can
/// hold any output length supported by BLAKE3's XOF mode.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ExtendedHash {
    /// The hash bytes.
    bytes: Vec<u8>,
    /// The algorithm that produced this hash.
    algorithm: HashAlgorithm,
}

impl ExtendedHash {
    /// Create an ExtendedHash from raw bytes.
    pub fn from_bytes(bytes: Vec<u8>, algorithm: HashAlgorithm) -> Self {
        Self { bytes, algorithm }
    }

    /// Get the hash bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Get the hash algorithm.
    pub fn algorithm(&self) -> HashAlgorithm {
        self.algorithm
    }

    /// Get the length in bytes.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Check if the hash is empty (should never happen in practice).
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Convert to a 32-byte ContentHash (truncates if longer).
    ///
    /// Returns `None` if the hash is shorter than 32 bytes.
    pub fn to_content_hash(&self) -> Option<ContentHash> {
        if self.bytes.len() < 32 {
            return None;
        }
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&self.bytes[..32]);
        Some(ContentHash::from_raw(arr))
    }
}

impl fmt::Display for ExtendedHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in &self.bytes {
            write!(f, "{:02x}", byte)?;
        }
        Ok(())
    }
}

impl fmt::Debug for ExtendedHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ExtendedHash({}, {} bytes)", self, self.bytes.len())
    }
}

// ---------------------------------------------------------------------------
// DualHash — BLAKE3 + SHA-256 for migration
// ---------------------------------------------------------------------------

/// Dual hash result containing both BLAKE3 and SHA-256 hashes.
///
/// Used during migration to store both hash types, allowing verification
/// with either algorithm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DualHash {
    /// BLAKE3 hash (primary).
    pub blake3: ContentHash,
    /// SHA-256 hash (legacy).
    pub sha256: ContentHash,
}

impl DualHash {
    /// Create a new DualHash from both hash values.
    pub const fn new(blake3: ContentHash, sha256: ContentHash) -> Self {
        Self { blake3, sha256 }
    }

    /// Get the primary (BLAKE3) hash.
    pub const fn primary(&self) -> &ContentHash {
        &self.blake3
    }

    /// Get the legacy (SHA-256) hash.
    pub const fn legacy(&self) -> &ContentHash {
        &self.sha256
    }

    /// Verify that data matches either hash.
    pub fn verify(&self, data: &[u8]) -> bool {
        self.verify_blake3(data) || self.verify_sha256(data)
    }

    /// Verify that data matches the BLAKE3 hash.
    #[cfg(feature = "blake3")]
    pub fn verify_blake3(&self, data: &[u8]) -> bool {
        let hash = blake3::hash(data);
        ContentHash::from_raw(*hash.as_bytes()) == self.blake3
    }

    #[cfg(not(feature = "blake3"))]
    pub fn verify_blake3(&self, _data: &[u8]) -> bool {
        false
    }

    /// Verify that data matches the SHA-256 hash.
    pub fn verify_sha256(&self, data: &[u8]) -> bool {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let result = hasher.finalize();
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&result);
        ContentHash::from_raw(arr) == self.sha256
    }
}

// ---------------------------------------------------------------------------
// ContentHasher — main hashing API
// ---------------------------------------------------------------------------

/// Content hasher with support for multiple algorithms and streaming.
///
/// Provides both one-shot and streaming hashing APIs.
pub struct ContentHasher {
    state: HasherState,
}

enum HasherState {
    #[cfg(feature = "blake3")]
    Blake3(blake3::Hasher),
    #[cfg(feature = "blake3")]
    Blake3Keyed(blake3::Hasher),
    Sha256(Sha256),
    #[cfg(feature = "blake3")]
    Dual {
        blake3: blake3::Hasher,
        sha256: Sha256,
    },
    #[cfg(not(feature = "blake3"))]
    Dual {
        sha256: Sha256,
    },
}

impl ContentHasher {
    // -----------------------------------------------------------------------
    // One-shot hashing
    // -----------------------------------------------------------------------

    /// Compute BLAKE3 hash of data (one-shot).
    ///
    /// Requires the `blake3` feature.
    #[cfg(feature = "blake3")]
    pub fn blake3(data: &[u8]) -> ContentHash {
        let hash = blake3::hash(data);
        ContentHash::from_raw(*hash.as_bytes())
    }

    /// Compute BLAKE3 hash with extended output length.
    #[cfg(feature = "blake3")]
    pub fn blake3_extended(data: &[u8], output_len: HashOutputLength) -> ExtendedHash {
        let mut hasher = blake3::Hasher::new();
        hasher.update(data);
        let mut output = vec![0u8; output_len.bytes()];
        hasher.finalize_xof().fill(&mut output);
        ExtendedHash::from_bytes(output, HashAlgorithm::Blake3)
    }

    /// Compute BLAKE3 keyed hash (MAC) of data.
    ///
    /// The key must be exactly 32 bytes. This provides authenticated
    /// hashing for content integrity verification.
    #[cfg(feature = "blake3")]
    pub fn blake3_keyed(key: &[u8; 32], data: &[u8]) -> ContentHash {
        let mut hasher = blake3::Hasher::new_keyed(key);
        let hash = hasher.update(data).finalize();
        ContentHash::from_raw(*hash.as_bytes())
    }

    /// Compute BLAKE3 keyed hash with extended output.
    #[cfg(feature = "blake3")]
    pub fn blake3_keyed_extended(
        key: &[u8; 32],
        data: &[u8],
        output_len: HashOutputLength,
    ) -> ExtendedHash {
        let mut hasher = blake3::Hasher::new_keyed(key);
        hasher.update(data);
        let mut output = vec![0u8; output_len.bytes()];
        hasher.finalize_xof().fill(&mut output);
        ExtendedHash::from_bytes(output, HashAlgorithm::Blake3)
    }

    /// Compute SHA-256 hash of data (one-shot).
    pub fn sha256(data: &[u8]) -> ContentHash {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let result = hasher.finalize();
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&result);
        ContentHash::from_raw(arr)
    }

    /// Compute both BLAKE3 and SHA-256 hashes (dual-hash mode).
    ///
    /// Used during migration to maintain compatibility with both algorithms.
    #[cfg(feature = "blake3")]
    pub fn dual_hash(data: &[u8]) -> DualHash {
        let blake3 = Self::blake3(data);
        let sha256 = Self::sha256(data);
        DualHash::new(blake3, sha256)
    }

    #[cfg(not(feature = "blake3"))]
    pub fn dual_hash(data: &[u8]) -> DualHash {
        let sha256 = Self::sha256(data);
        // Without blake3 feature, use zero hash for blake3 field
        DualHash::new(ContentHash::zero(), sha256)
    }

    // -----------------------------------------------------------------------
    // Streaming hashing
    // -----------------------------------------------------------------------

    /// Create a new streaming BLAKE3 hasher.
    #[cfg(feature = "blake3")]
    pub fn new_blake3_streaming() -> Self {
        Self {
            state: HasherState::Blake3(blake3::Hasher::new()),
        }
    }

    /// Create a new streaming BLAKE3 keyed hasher.
    #[cfg(feature = "blake3")]
    pub fn new_blake3_keyed_streaming(key: &[u8; 32]) -> Self {
        Self {
            state: HasherState::Blake3Keyed(blake3::Hasher::new_keyed(key)),
        }
    }

    /// Create a new streaming SHA-256 hasher.
    pub fn new_sha256_streaming() -> Self {
        Self {
            state: HasherState::Sha256(Sha256::new()),
        }
    }

    /// Create a new streaming dual hasher (BLAKE3 + SHA-256).
    #[cfg(feature = "blake3")]
    pub fn new_dual_streaming() -> Self {
        Self {
            state: HasherState::Dual {
                blake3: blake3::Hasher::new(),
                sha256: Sha256::new(),
            },
        }
    }

    #[cfg(not(feature = "blake3"))]
    pub fn new_dual_streaming() -> Self {
        Self {
            state: HasherState::Dual {
                sha256: Sha256::new(),
            },
        }
    }

    /// Update the hasher with more data.
    pub fn update(&mut self, data: &[u8]) {
        match &mut self.state {
            #[cfg(feature = "blake3")]
            HasherState::Blake3(h) => {
                h.update(data);
            }
            #[cfg(feature = "blake3")]
            HasherState::Blake3Keyed(h) => {
                h.update(data);
            }
            HasherState::Sha256(h) => {
                h.update(data);
            }
            #[cfg(feature = "blake3")]
            HasherState::Dual { blake3, sha256 } => {
                blake3.update(data);
                sha256.update(data);
            }
            #[cfg(not(feature = "blake3"))]
            HasherState::Dual { sha256 } => {
                sha256.update(data);
            }
        }
    }

    /// Finalize the hash and return the result.
    pub fn finalize(self) -> ContentHash {
        match self.state {
            #[cfg(feature = "blake3")]
            HasherState::Blake3(h) => {
                let hash = h.finalize();
                ContentHash::from_raw(*hash.as_bytes())
            }
            #[cfg(feature = "blake3")]
            HasherState::Blake3Keyed(h) => {
                let hash = h.finalize();
                ContentHash::from_raw(*hash.as_bytes())
            }
            HasherState::Sha256(h) => {
                let result = h.finalize();
                let mut arr = [0u8; 32];
                arr.copy_from_slice(&result);
                ContentHash::from_raw(arr)
            }
            #[cfg(feature = "blake3")]
            HasherState::Dual { blake3, .. } => {
                let hash = blake3.finalize();
                ContentHash::from_raw(*hash.as_bytes())
            }
            #[cfg(not(feature = "blake3"))]
            HasherState::Dual { sha256 } => {
                let result = sha256.finalize();
                let mut arr = [0u8; 32];
                arr.copy_from_slice(&result);
                ContentHash::from_raw(arr)
            }
        }
    }

    /// Finalize and return extended output (BLAKE3 only).
    #[cfg(feature = "blake3")]
    pub fn finalize_extended(self, output_len: HashOutputLength) -> ExtendedHash {
        match self.state {
            HasherState::Blake3(h) | HasherState::Blake3Keyed(h) => {
                let mut output = vec![0u8; output_len.bytes()];
                h.finalize_xof().fill(&mut output);
                ExtendedHash::from_bytes(output, HashAlgorithm::Blake3)
            }
            HasherState::Sha256(h) => {
                let result = h.finalize();
                let mut arr = vec![0u8; 32];
                arr.copy_from_slice(&result);
                ExtendedHash::from_bytes(arr, HashAlgorithm::Sha256)
            }
            HasherState::Dual { blake3, .. } => {
                let mut output = vec![0u8; output_len.bytes()];
                blake3.finalize_xof().fill(&mut output);
                ExtendedHash::from_bytes(output, HashAlgorithm::Blake3)
            }
        }
    }

    /// Finalize and return both hashes (dual mode only).
    #[cfg(feature = "blake3")]
    pub fn finalize_dual(self) -> DualHash {
        match self.state {
            HasherState::Dual { blake3, sha256 } => {
                let b3_hash = blake3.finalize();
                let sha_result = sha256.finalize();
                let mut sha_arr = [0u8; 32];
                sha_arr.copy_from_slice(&sha_result);
                DualHash::new(
                    ContentHash::from_raw(*b3_hash.as_bytes()),
                    ContentHash::from_raw(sha_arr),
                )
            }
            HasherState::Blake3(h) | HasherState::Blake3Keyed(h) => {
                let hash = h.finalize();
                DualHash::new(ContentHash::from_raw(*hash.as_bytes()), ContentHash::zero())
            }
            HasherState::Sha256(h) => {
                let result = h.finalize();
                let mut arr = [0u8; 32];
                arr.copy_from_slice(&result);
                DualHash::new(ContentHash::zero(), ContentHash::from_raw(arr))
            }
        }
    }

    #[cfg(not(feature = "blake3"))]
    pub fn finalize_dual(self) -> DualHash {
        match self.state {
            HasherState::Dual { sha256 } | HasherState::Sha256(sha256) => {
                let result = sha256.finalize();
                let mut arr = [0u8; 32];
                arr.copy_from_slice(&result);
                DualHash::new(ContentHash::zero(), ContentHash::from_raw(arr))
            }
        }
    }

    // -----------------------------------------------------------------------
    // Stream hashing from reader
    // -----------------------------------------------------------------------

    /// Hash content from a reader using BLAKE3.
    ///
    /// Reads in 64KB chunks for memory efficiency.
    #[cfg(feature = "blake3")]
    pub fn hash_reader<R: Read>(reader: &mut R) -> io::Result<ContentHash> {
        let mut hasher = blake3::Hasher::new();
        let mut buffer = [0u8; 65536]; // 64KB buffer
        loop {
            let n = reader.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            hasher.update(&buffer[..n]);
        }
        let hash = hasher.finalize();
        Ok(ContentHash::from_raw(*hash.as_bytes()))
    }

    /// Hash content from a reader using SHA-256.
    pub fn hash_reader_sha256<R: Read>(reader: &mut R) -> io::Result<ContentHash> {
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 65536];
        loop {
            let n = reader.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            hasher.update(&buffer[..n]);
        }
        let result = hasher.finalize();
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&result);
        Ok(ContentHash::from_raw(arr))
    }

    /// Hash content from a reader using both BLAKE3 and SHA-256.
    #[cfg(feature = "blake3")]
    pub fn hash_reader_dual<R: Read>(reader: &mut R) -> io::Result<DualHash> {
        let mut blake3_hasher = blake3::Hasher::new();
        let mut sha256_hasher = Sha256::new();
        let mut buffer = [0u8; 65536];
        loop {
            let n = reader.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            blake3_hasher.update(&buffer[..n]);
            sha256_hasher.update(&buffer[..n]);
        }
        let blake3 = blake3_hasher.finalize();
        let sha256_result = sha256_hasher.finalize();
        let mut sha_arr = [0u8; 32];
        sha_arr.copy_from_slice(&sha256_result);
        Ok(DualHash::new(
            ContentHash::from_raw(*blake3.as_bytes()),
            ContentHash::from_raw(sha_arr),
        ))
    }
}

// ---------------------------------------------------------------------------
// ContentHashWrapper — multi-algorithm hash with metadata
// ---------------------------------------------------------------------------

/// A content hash with associated algorithm metadata.
///
/// This wrapper records which hash function was used, enabling the asset
/// pipeline to support multiple algorithms during migration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContentHashWrapper {
    /// The hash value (always 32 bytes for compatibility).
    hash: ContentHash,
    /// The algorithm used to produce this hash.
    algorithm: HashAlgorithm,
}

impl ContentHashWrapper {
    /// Create a new wrapper from a hash and algorithm.
    pub const fn new(hash: ContentHash, algorithm: HashAlgorithm) -> Self {
        Self { hash, algorithm }
    }

    /// Compute a hash wrapper from data using the default algorithm.
    pub fn from_data(data: &[u8]) -> Self {
        Self {
            hash: ContentHash::from_bytes(data),
            algorithm: HashAlgorithm::default(),
        }
    }

    /// Compute a hash wrapper from data using a specific algorithm.
    pub fn from_data_with_algo(data: &[u8], algorithm: HashAlgorithm) -> Self {
        Self {
            hash: ContentHash::from_data_with_algo(data, algorithm),
            algorithm,
        }
    }

    /// Get the hash value.
    pub const fn hash(&self) -> &ContentHash {
        &self.hash
    }

    /// Get the algorithm.
    pub const fn algorithm(&self) -> HashAlgorithm {
        self.algorithm
    }

    /// Verify that data matches this hash.
    pub fn verify(&self, data: &[u8]) -> bool {
        let computed = ContentHash::from_data_with_algo(data, self.algorithm);
        computed == self.hash
    }
}

impl fmt::Display for ContentHashWrapper {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.algorithm.name(), self.hash)
    }
}

impl FromStr for ContentHashWrapper {
    type Err = ContentHashWrapperParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Some((algo_str, hash_str)) = s.split_once(':') {
            let algorithm = match algo_str {
                "sha256" => HashAlgorithm::Sha256,
                #[cfg(feature = "blake3")]
                "blake3" => HashAlgorithm::Blake3,
                _ => return Err(ContentHashWrapperParseError::UnknownAlgorithm),
            };
            let hash: ContentHash = hash_str.parse().map_err(|_| ContentHashWrapperParseError::InvalidHash)?;
            Ok(Self { hash, algorithm })
        } else {
            // No algorithm prefix — assume default
            let hash: ContentHash = s.parse().map_err(|_| ContentHashWrapperParseError::InvalidHash)?;
            Ok(Self {
                hash,
                algorithm: HashAlgorithm::default(),
            })
        }
    }
}

/// Error parsing a ContentHashWrapper from a string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentHashWrapperParseError {
    /// Unknown hash algorithm prefix.
    UnknownAlgorithm,
    /// Invalid hash hex string.
    InvalidHash,
}

impl fmt::Display for ContentHashWrapperParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownAlgorithm => write!(f, "unknown hash algorithm"),
            Self::InvalidHash => write!(f, "invalid hash hex string"),
        }
    }
}

impl std::error::Error for ContentHashWrapperParseError {}

// ---------------------------------------------------------------------------
// Serialization support
// ---------------------------------------------------------------------------

#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[cfg(feature = "serde")]
impl Serialize for ExtendedHash {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let hex: String = self.bytes.iter().map(|b| format!("{:02x}", b)).collect();
        serializer.serialize_str(&hex)
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for ExtendedHash {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let hex = String::deserialize(deserializer)?;
        let bytes = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16))
            .collect::<Result<Vec<u8>, _>>()
            .map_err(serde::de::Error::custom)?;
        Ok(Self {
            bytes,
            algorithm: HashAlgorithm::default(),
        })
    }
}

#[cfg(feature = "serde")]
impl Serialize for DualHash {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("DualHash", 2)?;
        s.serialize_field("blake3", &format!("{}", self.blake3))?;
        s.serialize_field("sha256", &format!("{}", self.sha256))?;
        s.end()
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for DualHash {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct DualHashHelper {
            blake3: String,
            sha256: String,
        }
        let helper = DualHashHelper::deserialize(deserializer)?;
        let blake3: ContentHash = helper.blake3.parse().map_err(serde::de::Error::custom)?;
        let sha256: ContentHash = helper.sha256.parse().map_err(serde::de::Error::custom)?;
        Ok(Self { blake3, sha256 })
    }
}

#[cfg(feature = "serde")]
impl Serialize for ContentHashWrapper {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for ContentHashWrapper {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

// ---------------------------------------------------------------------------
// Asset manifest integration
// ---------------------------------------------------------------------------

/// Asset manifest entry with hash algorithm metadata.
///
/// Records the hash algorithm used for each asset, enabling the pipeline
/// to support multiple algorithms during migration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssetManifestEntry {
    /// Asset path relative to the asset root.
    pub path: String,
    /// Content hash with algorithm metadata.
    pub hash: ContentHashWrapper,
    /// Size in bytes.
    pub size: u64,
    /// Optional legacy SHA-256 hash (for dual-hash migration).
    pub legacy_sha256: Option<ContentHash>,
}

impl AssetManifestEntry {
    /// Create a new manifest entry.
    pub fn new(path: String, hash: ContentHashWrapper, size: u64) -> Self {
        Self {
            path,
            hash,
            size,
            legacy_sha256: None,
        }
    }

    /// Create a manifest entry with dual hashes.
    pub fn with_dual_hash(path: String, dual: DualHash, size: u64) -> Self {
        Self {
            path,
            #[cfg(feature = "blake3")]
            hash: ContentHashWrapper::new(dual.blake3, HashAlgorithm::Blake3),
            #[cfg(not(feature = "blake3"))]
            hash: ContentHashWrapper::new(dual.sha256, HashAlgorithm::Sha256),
            size,
            legacy_sha256: Some(dual.sha256),
        }
    }

    /// Verify asset content against stored hashes.
    ///
    /// Returns `true` if the content matches either the primary or legacy hash.
    pub fn verify(&self, data: &[u8]) -> bool {
        if self.hash.verify(data) {
            return true;
        }
        if let Some(legacy) = &self.legacy_sha256 {
            let computed = ContentHasher::sha256(data);
            return computed == *legacy;
        }
        false
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // ---- Basic BLAKE3 hash computation ----

    #[test]
    #[cfg(feature = "blake3")]
    fn test_blake3_basic_hash() {
        let data = b"hello world";
        let hash = ContentHasher::blake3(data);

        // BLAKE3 is deterministic
        let hash2 = ContentHasher::blake3(data);
        assert_eq!(hash, hash2);

        // Different data produces different hash
        let hash3 = ContentHasher::blake3(b"different data");
        assert_ne!(hash, hash3);
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_blake3_empty_data() {
        let hash = ContentHasher::blake3(b"");
        let hash2 = ContentHasher::blake3(b"");
        assert_eq!(hash, hash2);
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_blake3_large_data() {
        let data: Vec<u8> = (0..1_000_000).map(|i| (i % 256) as u8).collect();
        let hash = ContentHasher::blake3(&data);
        let hash2 = ContentHasher::blake3(&data);
        assert_eq!(hash, hash2);
    }

    // ---- Streaming hash matches full-buffer hash ----

    #[test]
    #[cfg(feature = "blake3")]
    fn test_streaming_matches_full_buffer() {
        let data = b"hello world streaming test data";

        // One-shot hash
        let hash_oneshot = ContentHasher::blake3(data);

        // Streaming hash - single update
        let mut hasher = ContentHasher::new_blake3_streaming();
        hasher.update(data);
        let hash_streaming_single = hasher.finalize();

        assert_eq!(hash_oneshot, hash_streaming_single);

        // Streaming hash - chunked updates
        let mut hasher = ContentHasher::new_blake3_streaming();
        hasher.update(b"hello ");
        hasher.update(b"world ");
        hasher.update(b"streaming ");
        hasher.update(b"test ");
        hasher.update(b"data");
        let hash_streaming_chunked = hasher.finalize();

        assert_eq!(hash_oneshot, hash_streaming_chunked);
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_streaming_large_chunks() {
        let data: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();

        let hash_oneshot = ContentHasher::blake3(&data);

        let mut hasher = ContentHasher::new_blake3_streaming();
        for chunk in data.chunks(8192) {
            hasher.update(chunk);
        }
        let hash_streaming = hasher.finalize();

        assert_eq!(hash_oneshot, hash_streaming);
    }

    // ---- Keyed hashing mode ----

    #[test]
    #[cfg(feature = "blake3")]
    fn test_keyed_hashing() {
        let key = [0x42u8; 32];
        let data = b"authenticated content";

        let mac = ContentHasher::blake3_keyed(&key, data);
        let mac2 = ContentHasher::blake3_keyed(&key, data);
        assert_eq!(mac, mac2);

        // Different key produces different MAC
        let key2 = [0x43u8; 32];
        let mac3 = ContentHasher::blake3_keyed(&key2, data);
        assert_ne!(mac, mac3);

        // Different data produces different MAC
        let mac4 = ContentHasher::blake3_keyed(&key, b"different content");
        assert_ne!(mac, mac4);
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_keyed_hashing_streaming() {
        let key = [0x42u8; 32];
        let data = b"authenticated content";

        let mac_oneshot = ContentHasher::blake3_keyed(&key, data);

        let mut hasher = ContentHasher::new_blake3_keyed_streaming(&key);
        hasher.update(data);
        let mac_streaming = hasher.finalize();

        assert_eq!(mac_oneshot, mac_streaming);
    }

    // ---- 256-bit and 512-bit output ----

    #[test]
    #[cfg(feature = "blake3")]
    fn test_256_bit_output() {
        let data = b"test data";
        let hash = ContentHasher::blake3_extended(data, HashOutputLength::Bits256);
        assert_eq!(hash.len(), 32);

        // Should match regular blake3 hash
        let regular = ContentHasher::blake3(data);
        assert_eq!(&hash.as_bytes()[..32], regular.as_bytes());
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_512_bit_output() {
        let data = b"test data";
        let hash = ContentHasher::blake3_extended(data, HashOutputLength::Bits512);
        assert_eq!(hash.len(), 64);

        // First 32 bytes should match regular hash
        let regular = ContentHasher::blake3(data);
        assert_eq!(&hash.as_bytes()[..32], regular.as_bytes());
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_custom_output_length() {
        let data = b"test data";
        let hash = ContentHasher::blake3_extended(data, HashOutputLength::Custom(48));
        assert_eq!(hash.len(), 48);
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_extended_to_content_hash() {
        let data = b"test data";
        let extended = ContentHasher::blake3_extended(data, HashOutputLength::Bits512);
        let content_hash = extended.to_content_hash().unwrap();

        let regular = ContentHasher::blake3(data);
        assert_eq!(content_hash, regular);
    }

    // ---- Dual-hash mode (BLAKE3 + SHA-256) ----

    #[test]
    #[cfg(feature = "blake3")]
    fn test_dual_hash() {
        let data = b"dual hash test";
        let dual = ContentHasher::dual_hash(data);

        // Verify both hashes are computed correctly
        assert_eq!(dual.blake3, ContentHasher::blake3(data));
        assert_eq!(dual.sha256, ContentHasher::sha256(data));

        // Hashes should be different
        assert_ne!(dual.blake3, dual.sha256);
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_dual_hash_streaming() {
        let data = b"dual hash streaming test";

        let dual_oneshot = ContentHasher::dual_hash(data);

        let mut hasher = ContentHasher::new_dual_streaming();
        hasher.update(data);
        let dual_streaming = hasher.finalize_dual();

        assert_eq!(dual_oneshot.blake3, dual_streaming.blake3);
        assert_eq!(dual_oneshot.sha256, dual_streaming.sha256);
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_dual_hash_verify() {
        let data = b"verification test";
        let dual = ContentHasher::dual_hash(data);

        assert!(dual.verify(data));
        assert!(dual.verify_blake3(data));
        assert!(dual.verify_sha256(data));

        assert!(!dual.verify(b"wrong data"));
    }

    // ---- Hash equality comparison ----

    #[test]
    fn test_hash_equality() {
        let data = b"equality test";
        let hash1 = ContentHasher::sha256(data);
        let hash2 = ContentHasher::sha256(data);
        let hash3 = ContentHasher::sha256(b"different");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_content_hash_zero() {
        let zero = ContentHash::zero();
        assert!(zero.is_zero());
        assert_eq!(zero.as_bytes(), &[0u8; 32]);

        let nonzero = ContentHasher::sha256(b"data");
        assert!(!nonzero.is_zero());
    }

    // ---- Performance benchmark (<5ms/MB) ----

    #[test]
    #[cfg(feature = "blake3")]
    fn test_performance_1mb() {
        let data: Vec<u8> = (0..1_000_000).map(|i| (i % 256) as u8).collect();

        let start = std::time::Instant::now();
        let _hash = ContentHasher::blake3(&data);
        let elapsed = start.elapsed();

        // Should complete in under 5ms for 1MB
        assert!(
            elapsed.as_millis() < 5,
            "BLAKE3 took {}ms for 1MB, expected <5ms",
            elapsed.as_millis()
        );
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_performance_streaming() {
        let data: Vec<u8> = (0..1_000_000).map(|i| (i % 256) as u8).collect();

        let start = std::time::Instant::now();
        let mut hasher = ContentHasher::new_blake3_streaming();
        for chunk in data.chunks(65536) {
            hasher.update(chunk);
        }
        let _hash = hasher.finalize();
        let elapsed = start.elapsed();

        // Streaming should also be under 5ms for 1MB
        assert!(
            elapsed.as_millis() < 5,
            "Streaming BLAKE3 took {}ms for 1MB, expected <5ms",
            elapsed.as_millis()
        );
    }

    // ---- ContentHash serialization/deserialization ----

    #[test]
    fn test_content_hash_display_parse() {
        let hash = ContentHasher::sha256(b"test");
        let hex = format!("{}", hash);

        assert_eq!(hex.len(), 64);

        let parsed: ContentHash = hex.parse().unwrap();
        assert_eq!(parsed, hash);
    }

    #[test]
    fn test_content_hash_parse_invalid() {
        // Too short
        let result: Result<ContentHash, _> = "abc".parse();
        assert!(result.is_err());

        // Too long
        let long = "a".repeat(100);
        let result: Result<ContentHash, _> = long.parse();
        assert!(result.is_err());

        // Invalid hex
        let result: Result<ContentHash, _> = "zzzz".repeat(16).parse();
        assert!(result.is_err());
    }

    // ---- ContentHashWrapper ----

    #[test]
    fn test_content_hash_wrapper() {
        let data = b"wrapper test";
        let wrapper = ContentHashWrapper::from_data(data);

        assert!(wrapper.verify(data));
        assert!(!wrapper.verify(b"wrong"));
    }

    #[test]
    fn test_content_hash_wrapper_display_parse() {
        let data = b"wrapper parse test";
        let wrapper = ContentHashWrapper::from_data_with_algo(data, HashAlgorithm::Sha256);

        let s = wrapper.to_string();
        assert!(s.starts_with("sha256:"));

        let parsed: ContentHashWrapper = s.parse().unwrap();
        assert_eq!(parsed, wrapper);
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_content_hash_wrapper_blake3() {
        let data = b"blake3 wrapper test";
        let wrapper = ContentHashWrapper::from_data_with_algo(data, HashAlgorithm::Blake3);

        let s = wrapper.to_string();
        assert!(s.starts_with("blake3:"));

        let parsed: ContentHashWrapper = s.parse().unwrap();
        assert_eq!(parsed, wrapper);
        assert!(parsed.verify(data));
    }

    // ---- Migration path tests ----

    #[test]
    #[cfg(feature = "blake3")]
    fn test_migration_dual_hash_entry() {
        let data = b"migration test asset";
        let dual = ContentHasher::dual_hash(data);

        let entry = AssetManifestEntry::with_dual_hash(
            "assets/texture.png".to_string(),
            dual.clone(),
            data.len() as u64,
        );

        // Should verify with either hash
        assert!(entry.verify(data));

        // Check both hashes are stored
        assert!(entry.legacy_sha256.is_some());
    }

    #[test]
    fn test_asset_manifest_entry() {
        let data = b"manifest entry test";
        let wrapper = ContentHashWrapper::from_data(data);

        let entry = AssetManifestEntry::new(
            "models/mesh.bin".to_string(),
            wrapper,
            data.len() as u64,
        );

        assert!(entry.verify(data));
        assert_eq!(entry.path, "models/mesh.bin");
        assert_eq!(entry.size, data.len() as u64);
    }

    // ---- Reader hashing ----

    #[test]
    #[cfg(feature = "blake3")]
    fn test_hash_reader() {
        let data = b"reader hashing test data";
        let mut cursor = Cursor::new(data);

        let hash_reader = ContentHasher::hash_reader(&mut cursor).unwrap();
        let hash_direct = ContentHasher::blake3(data);

        assert_eq!(hash_reader, hash_direct);
    }

    #[test]
    fn test_hash_reader_sha256() {
        let data = b"sha256 reader test";
        let mut cursor = Cursor::new(data);

        let hash_reader = ContentHasher::hash_reader_sha256(&mut cursor).unwrap();
        let hash_direct = ContentHasher::sha256(data);

        assert_eq!(hash_reader, hash_direct);
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_hash_reader_dual() {
        let data = b"dual reader test";
        let mut cursor = Cursor::new(data);

        let dual_reader = ContentHasher::hash_reader_dual(&mut cursor).unwrap();
        let dual_direct = ContentHasher::dual_hash(data);

        assert_eq!(dual_reader.blake3, dual_direct.blake3);
        assert_eq!(dual_reader.sha256, dual_direct.sha256);
    }

    // ---- Extended hash tests ----

    #[test]
    #[cfg(feature = "blake3")]
    fn test_extended_hash_display() {
        let data = b"extended display test";
        let hash = ContentHasher::blake3_extended(data, HashOutputLength::Bits512);

        let hex = hash.to_string();
        assert_eq!(hex.len(), 128); // 64 bytes = 128 hex chars
    }

    #[test]
    #[cfg(feature = "blake3")]
    fn test_extended_hash_keyed() {
        let key = [0x55u8; 32];
        let data = b"keyed extended test";

        let hash = ContentHasher::blake3_keyed_extended(&key, data, HashOutputLength::Bits512);
        assert_eq!(hash.len(), 64);

        // Should differ from unkeyed
        let unkeyed = ContentHasher::blake3_extended(data, HashOutputLength::Bits512);
        assert_ne!(hash.as_bytes(), unkeyed.as_bytes());
    }

    // ---- Algorithm selection ----

    #[test]
    fn test_algorithm_name() {
        assert_eq!(HashAlgorithm::Sha256.name(), "sha256");
        #[cfg(feature = "blake3")]
        assert_eq!(HashAlgorithm::Blake3.name(), "blake3");
    }

    #[test]
    fn test_default_algorithm() {
        let default = HashAlgorithm::default();
        #[cfg(feature = "blake3")]
        assert_eq!(default, HashAlgorithm::Blake3);
        #[cfg(not(feature = "blake3"))]
        assert_eq!(default, HashAlgorithm::Sha256);
    }

    // ---- Dual hash primary/legacy accessors ----

    #[test]
    #[cfg(feature = "blake3")]
    fn test_dual_hash_accessors() {
        let data = b"accessor test";
        let dual = ContentHasher::dual_hash(data);

        assert_eq!(dual.primary(), &dual.blake3);
        assert_eq!(dual.legacy(), &dual.sha256);
    }

    // ---- Hash output length ----

    #[test]
    fn test_hash_output_length() {
        assert_eq!(HashOutputLength::Bits256.bytes(), 32);
        assert_eq!(HashOutputLength::Bits512.bytes(), 64);
        assert_eq!(HashOutputLength::Custom(100).bytes(), 100);
    }

    // ---- SHA-256 streaming ----

    #[test]
    fn test_sha256_streaming() {
        let data = b"sha256 streaming test";

        let hash_oneshot = ContentHasher::sha256(data);

        let mut hasher = ContentHasher::new_sha256_streaming();
        hasher.update(data);
        let hash_streaming = hasher.finalize();

        assert_eq!(hash_oneshot, hash_streaming);
    }

    #[test]
    fn test_sha256_streaming_chunked() {
        let data = b"sha256 chunked streaming test with more data";

        let hash_oneshot = ContentHasher::sha256(data);

        let mut hasher = ContentHasher::new_sha256_streaming();
        for chunk in data.chunks(10) {
            hasher.update(chunk);
        }
        let hash_streaming = hasher.finalize();

        assert_eq!(hash_oneshot, hash_streaming);
    }

    // ---- Edge cases ----

    #[test]
    fn test_empty_data_all_algorithms() {
        let empty = b"";

        let sha256 = ContentHasher::sha256(empty);
        assert!(!sha256.is_zero());

        #[cfg(feature = "blake3")]
        {
            let blake3 = ContentHasher::blake3(empty);
            assert!(!blake3.is_zero());
            assert_ne!(sha256, blake3);
        }
    }

    #[test]
    fn test_single_byte() {
        let data = b"X";

        let sha256 = ContentHasher::sha256(data);
        assert!(!sha256.is_zero());

        let sha256_2 = ContentHasher::sha256(data);
        assert_eq!(sha256, sha256_2);
    }

    // ---- Wrapper parse errors ----

    #[test]
    fn test_wrapper_parse_unknown_algorithm() {
        let result: Result<ContentHashWrapper, _> = "unknown:abcd".repeat(16).as_str().parse();
        assert!(matches!(result, Err(ContentHashWrapperParseError::UnknownAlgorithm)));
    }

    #[test]
    fn test_wrapper_parse_invalid_hash() {
        let result: Result<ContentHashWrapper, _> = "sha256:notahash".parse();
        assert!(matches!(result, Err(ContentHashWrapperParseError::InvalidHash)));
    }
}
