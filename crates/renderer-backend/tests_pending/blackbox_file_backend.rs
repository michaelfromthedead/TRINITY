//! Blackbox tests for FileBackend content-addressable storage.
//!
//! Tests the FileBackend implementation for:
//! - Put/get round-trip for small and large data
//! - Duplicate put returns same hash
//! - has() returns correct status
//! - Concurrent reads
//! - Streaming threshold behavior
//! - Git-style directory layout (ab/cdef...)

use renderer_backend::pipeline::{ChunkedContent, ContentHash, FileBackend};
use std::io::Read;
use std::thread;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

fn create_temp_backend() -> (TempDir, FileBackend) {
    let dir = TempDir::new().expect("create temp dir");
    let backend = FileBackend::new(dir.path()).expect("create backend");
    (dir, backend)
}

fn generate_data(size: usize) -> Vec<u8> {
    (0..size).map(|i| (i % 256) as u8).collect()
}

// ===========================================================================
// Test 1: Basic put/get round-trip for small data
// ===========================================================================
#[test]
fn test_put_get_roundtrip_small() {
    let (_dir, backend) = create_temp_backend();

    let data = b"hello world";
    let hash = backend.put(data).expect("put should succeed");

    let retrieved = backend.get(&hash).expect("get should succeed");
    assert_eq!(retrieved, Some(data.to_vec()));
}

// ===========================================================================
// Test 2: Put/get round-trip for large data (above streaming threshold)
// ===========================================================================
#[test]
fn test_put_get_roundtrip_large() {
    let (_dir, backend) = create_temp_backend();

    // 2MB data - above the typical streaming threshold
    let data = generate_data(2 * 1024 * 1024);
    let hash = backend.put(&data).expect("put should succeed");

    let retrieved = backend.get(&hash).expect("get should succeed");
    assert_eq!(retrieved, Some(data));
}

// ===========================================================================
// Test 3: Duplicate put returns same hash
// ===========================================================================
#[test]
fn test_duplicate_put_returns_same_hash() {
    let (_dir, backend) = create_temp_backend();

    let data = b"duplicate content test";

    let hash1 = backend.put(data).expect("first put");
    let hash2 = backend.put(data).expect("second put");

    assert_eq!(hash1, hash2, "same content should produce same hash");

    // Verify content is still correct
    let retrieved = backend.get(&hash1).expect("get");
    assert_eq!(retrieved, Some(data.to_vec()));
}

// ===========================================================================
// Test 4: has() returns correct status
// ===========================================================================
#[test]
fn test_has_returns_correct_status() {
    let (_dir, backend) = create_temp_backend();

    let data = b"test content for has()";
    let hash = backend.put(data).expect("put");

    // Stored content should exist
    assert!(backend.has(&hash), "stored content should exist");

    // Non-existent content should not exist
    let fake_hash = ContentHash::from_bytes(b"nonexistent content");
    assert!(!backend.has(&fake_hash), "non-existent content should not exist");
}

// ===========================================================================
// Test 5: Concurrent reads succeed without conflicts
// ===========================================================================
#[test]
fn test_concurrent_reads() {
    let (dir, backend) = create_temp_backend();

    // Store some data
    let data = b"concurrent read test data that needs to be read by multiple threads";
    let hash = backend.put(data).expect("put");

    // Create multiple reader threads
    let backend_path = dir.path().to_path_buf();
    let hash_clone = hash;
    let expected_data = data.to_vec();

    let handles: Vec<_> = (0..10)
        .map(|_| {
            let path = backend_path.clone();
            let h = hash_clone;
            let expected = expected_data.clone();

            thread::spawn(move || {
                let backend = FileBackend::open(&path).expect("open backend");
                for _ in 0..100 {
                    let result = backend.get(&h).expect("get should succeed");
                    assert_eq!(result, Some(expected.clone()));
                }
            })
        })
        .collect();

    // Wait for all threads to complete
    for handle in handles {
        handle.join().expect("thread should complete");
    }
}

// ===========================================================================
// Test 6: Directory structure matches git-style layout (ab/cdef...)
// ===========================================================================
#[test]
fn test_git_style_directory_layout() {
    let (dir, backend) = create_temp_backend();

    let data = b"verify directory structure";
    let hash = backend.put(data).expect("put");

    // Hash should be hex string
    let hex = format!("{}", hash);
    assert_eq!(hex.len(), 64, "SHA-256 hash should be 64 hex chars");

    // Directory structure should be root/ab/cdef...
    let prefix = &hex[..2];
    let suffix = &hex[2..];

    let expected_path = dir.path().join(prefix).join(suffix);
    assert!(
        expected_path.exists(),
        "blob should exist at git-style path: {}",
        expected_path.display()
    );

    // Verify the directory hierarchy
    let prefix_dir = dir.path().join(prefix);
    assert!(prefix_dir.is_dir(), "prefix directory should exist");
}

// ===========================================================================
// Test 7: Empty data handling
// ===========================================================================
#[test]
fn test_empty_data() {
    let (_dir, backend) = create_temp_backend();

    let empty: &[u8] = b"";
    let hash = backend.put(empty).expect("put empty");

    let retrieved = backend.get(&hash).expect("get");
    assert_eq!(retrieved, Some(vec![]));

    assert!(backend.has(&hash));
}

// ===========================================================================
// Test 8: Get non-existent content returns None
// ===========================================================================
#[test]
fn test_get_nonexistent_returns_none() {
    let (_dir, backend) = create_temp_backend();

    let fake_hash = ContentHash::from_bytes(b"this content does not exist");
    let result = backend.get(&fake_hash).expect("get should not error");

    assert_eq!(result, None);
}

// ===========================================================================
// Test 9: Delete removes content
// ===========================================================================
#[test]
fn test_delete_removes_content() {
    let (_dir, backend) = create_temp_backend();

    let data = b"content to delete";
    let hash = backend.put(data).expect("put");

    assert!(backend.has(&hash));

    let deleted = backend.delete(&hash).expect("delete");
    assert!(deleted, "delete should return true for existing content");

    assert!(!backend.has(&hash), "content should not exist after delete");

    // Delete again should return false
    let deleted_again = backend.delete(&hash).expect("delete again");
    assert!(!deleted_again, "deleting non-existent should return false");
}

// ===========================================================================
// Test 10: Size returns correct value
// ===========================================================================
#[test]
fn test_size_returns_correct_value() {
    let (_dir, backend) = create_temp_backend();

    let data = b"content with known size";
    let hash = backend.put(data).expect("put");

    let size = backend.size(&hash).expect("size");
    assert_eq!(size, Some(data.len() as u64));

    // Non-existent should return None
    let fake_hash = ContentHash::from_bytes(b"nonexistent");
    let size = backend.size(&fake_hash).expect("size of nonexistent");
    assert_eq!(size, None);
}

// ===========================================================================
// Test 11: List returns all stored hashes
// ===========================================================================
#[test]
fn test_list_returns_all_hashes() {
    let (_dir, backend) = create_temp_backend();

    let data1 = b"first item";
    let data2 = b"second item";
    let data3 = b"third item";

    let hash1 = backend.put(data1).expect("put 1");
    let hash2 = backend.put(data2).expect("put 2");
    let hash3 = backend.put(data3).expect("put 3");

    let list = backend.list().expect("list");

    assert_eq!(list.len(), 3);
    assert!(list.contains(&hash1));
    assert!(list.contains(&hash2));
    assert!(list.contains(&hash3));
}

// ===========================================================================
// Test 12: Streaming write/read for large content
// ===========================================================================
#[test]
fn test_streaming_write_read() {
    let (_dir, backend) = create_temp_backend();

    // Create large data that would benefit from streaming
    let data: Vec<u8> = generate_data(512 * 1024); // 512KB

    // Use streaming write
    let mut cursor = std::io::Cursor::new(&data);
    let manifest_hash = backend.put_stream(&mut cursor).expect("put_stream");

    // Use streaming read
    let mut reader = backend
        .get_stream(&manifest_hash)
        .expect("get_stream")
        .expect("manifest should exist");

    let mut retrieved = Vec::new();
    reader.read_to_end(&mut retrieved).expect("read to end");

    assert_eq!(retrieved, data);
    assert_eq!(reader.total_size(), data.len() as u64);
}

// ===========================================================================
// Test 13: Chunked content manifest
// ===========================================================================
#[test]
fn test_chunked_content_manifest() {
    let (_dir, backend) = create_temp_backend();

    // Create data that will span multiple chunks
    let chunk_size = 64 * 1024; // 64KB chunks for testing
    let data: Vec<u8> = generate_data(chunk_size * 3 + 1000); // 3 full chunks + partial

    let mut cursor = std::io::Cursor::new(&data);
    let manifest_hash = backend
        .put_stream_with_chunk_size(&mut cursor, chunk_size)
        .expect("put_stream");

    // Get the manifest
    let manifest = backend
        .get_manifest(&manifest_hash)
        .expect("get_manifest")
        .expect("manifest should exist");

    assert_eq!(manifest.total_size, data.len() as u64);
    assert_eq!(manifest.chunk_size, chunk_size);
    assert_eq!(manifest.chunks.len(), 4); // 3 full + 1 partial

    // Verify all chunks exist
    for chunk_hash in &manifest.chunks {
        assert!(backend.has(chunk_hash), "chunk should exist");
    }
}

// ===========================================================================
// Test 14: Tree storage and retrieval
// ===========================================================================
#[test]
fn test_tree_put_get() {
    let (_dir, backend) = create_temp_backend();

    // Store some blobs
    let blob1 = backend.put(b"blob 1 content").expect("put blob1");
    let blob2 = backend.put(b"blob 2 content").expect("put blob2");
    let blob3 = backend.put(b"blob 3 content").expect("put blob3");

    // Store a tree referencing the blobs
    let entries = vec![
        (blob1, "file1.txt".to_string()),
        (blob2, "file2.txt".to_string()),
        (blob3, "subdir/file3.txt".to_string()),
    ];

    let tree_hash = backend.tree_put(&entries).expect("tree_put");

    // Retrieve the tree
    let retrieved = backend.tree_get(&tree_hash).expect("tree_get");
    let retrieved = retrieved.expect("tree should exist");

    assert_eq!(retrieved.len(), 3);
    assert_eq!(retrieved[0], (blob1, "file1.txt".to_string()));
    assert_eq!(retrieved[1], (blob2, "file2.txt".to_string()));
    assert_eq!(retrieved[2], (blob3, "subdir/file3.txt".to_string()));
}

// ===========================================================================
// Test 15: Concurrent writes with same content
// ===========================================================================
#[test]
fn test_concurrent_writes_same_content() {
    let (dir, _backend) = create_temp_backend();
    let backend_path = dir.path().to_path_buf();

    let data = b"same content written by multiple threads";
    let expected_hash = ContentHash::from_bytes(data);

    let handles: Vec<_> = (0..10)
        .map(|_| {
            let path = backend_path.clone();
            let d = data.to_vec();
            let expected = expected_hash;

            thread::spawn(move || {
                let backend = FileBackend::new(&path).expect("create backend");
                for _ in 0..50 {
                    let hash = backend.put(&d).expect("put");
                    assert_eq!(hash, expected);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("thread should complete");
    }

    // Verify final state
    let backend = FileBackend::open(dir.path()).expect("open");
    assert!(backend.has(&expected_hash));
    assert_eq!(backend.get(&expected_hash).expect("get"), Some(data.to_vec()));
}

// ===========================================================================
// Test 16: Open non-existent directory fails
// ===========================================================================
#[test]
fn test_open_nonexistent_fails() {
    let result = FileBackend::open("/nonexistent/path/to/store");
    assert!(result.is_err());
}

// ===========================================================================
// Test 17: New creates directory if missing
// ===========================================================================
#[test]
fn test_new_creates_directory() {
    let temp = TempDir::new().expect("create temp");
    let store_path = temp.path().join("nested").join("content").join("store");

    assert!(!store_path.exists());

    let backend = FileBackend::new(&store_path).expect("create nested backend");
    assert!(store_path.exists());

    // Verify it works
    let hash = backend.put(b"test").expect("put");
    assert!(backend.has(&hash));
}

// ===========================================================================
// Test 18: Base path accessor
// ===========================================================================
#[test]
fn test_base_path_accessor() {
    let (dir, backend) = create_temp_backend();
    assert_eq!(backend.base_path(), dir.path());
}

// ===========================================================================
// Test 19: Binary data with all byte values
// ===========================================================================
#[test]
fn test_binary_data_all_byte_values() {
    let (_dir, backend) = create_temp_backend();

    // All 256 byte values
    let data: Vec<u8> = (0..=255).collect();
    let hash = backend.put(&data).expect("put");

    let retrieved = backend.get(&hash).expect("get");
    assert_eq!(retrieved, Some(data));
}

// ===========================================================================
// Test 20: Large binary file stress test
// ===========================================================================
#[test]
fn test_large_binary_stress() {
    let (_dir, backend) = create_temp_backend();

    // 5MB of random-ish data
    let data: Vec<u8> = (0..5 * 1024 * 1024)
        .map(|i| ((i * 17 + 31) % 256) as u8)
        .collect();

    let hash = backend.put(&data).expect("put");

    // Verify hash is deterministic
    let hash2 = ContentHash::from_bytes(&data);
    assert_eq!(hash, hash2);

    // Retrieve and verify
    let retrieved = backend.get(&hash).expect("get");
    assert_eq!(retrieved.as_ref().map(|v| v.len()), Some(data.len()));
    assert_eq!(retrieved, Some(data));
}

// ===========================================================================
// Test 21: Streaming reader tracks bytes read
// ===========================================================================
#[test]
fn test_streaming_reader_tracks_bytes() {
    let (_dir, backend) = create_temp_backend();

    let data: Vec<u8> = generate_data(200 * 1024); // 200KB
    let mut cursor = std::io::Cursor::new(&data);
    let manifest_hash = backend.put_stream(&mut cursor).expect("put_stream");

    let mut reader = backend
        .get_stream(&manifest_hash)
        .expect("get_stream")
        .expect("manifest exists");

    assert_eq!(reader.bytes_read(), 0);

    let mut buf = vec![0u8; 50 * 1024]; // Read 50KB at a time
    let n = reader.read(&mut buf).expect("read");
    assert!(n > 0);
    assert_eq!(reader.bytes_read(), n as u64);

    // Read rest
    let mut rest = Vec::new();
    reader.read_to_end(&mut rest).expect("read rest");
    assert_eq!(reader.bytes_read(), data.len() as u64);
}

// ===========================================================================
// Test 22: ChunkedContent serialization roundtrip
// ===========================================================================
#[test]
fn test_chunked_content_serialization() {
    let hash1 = ContentHash::from_bytes(b"chunk1");
    let hash2 = ContentHash::from_bytes(b"chunk2");
    let hash3 = ContentHash::from_bytes(b"chunk3");

    let original = ChunkedContent {
        total_size: 1000000,
        chunk_size: 256 * 1024,
        chunks: vec![hash1, hash2, hash3],
    };

    let serialized = original.serialize();
    let deserialized = ChunkedContent::deserialize(&serialized).expect("deserialize");

    assert_eq!(deserialized.total_size, original.total_size);
    assert_eq!(deserialized.chunk_size, original.chunk_size);
    assert_eq!(deserialized.chunks, original.chunks);
}

// ===========================================================================
// Test 23: ContentHash display and parse roundtrip
// ===========================================================================
#[test]
fn test_content_hash_display_parse() {
    let data = b"test data for hash";
    let hash = ContentHash::from_bytes(data);

    let hex_str = format!("{}", hash);
    assert_eq!(hex_str.len(), 64);

    let parsed: ContentHash = hex_str.parse().expect("parse");
    assert_eq!(parsed, hash);
}

// ===========================================================================
// Test 24: ContentHash zero and is_zero
// ===========================================================================
#[test]
fn test_content_hash_zero() {
    let zero = ContentHash::zero();
    assert!(zero.is_zero());

    let non_zero = ContentHash::from_bytes(b"not zero");
    assert!(!non_zero.is_zero());
}

// ===========================================================================
// Test 25: Multiple streaming operations
// ===========================================================================
#[test]
fn test_multiple_streaming_operations() {
    let (_dir, backend) = create_temp_backend();

    let data1: Vec<u8> = generate_data(300 * 1024);
    let data2: Vec<u8> = generate_data(400 * 1024);
    let data3: Vec<u8> = generate_data(500 * 1024);

    let hash1 = backend.put_stream(&mut std::io::Cursor::new(&data1)).expect("put1");
    let hash2 = backend.put_stream(&mut std::io::Cursor::new(&data2)).expect("put2");
    let hash3 = backend.put_stream(&mut std::io::Cursor::new(&data3)).expect("put3");

    // All should be retrievable
    for (hash, expected) in [(hash1, &data1), (hash2, &data2), (hash3, &data3)] {
        let mut reader = backend
            .get_stream(&hash)
            .expect("get_stream")
            .expect("exists");

        let mut retrieved = Vec::new();
        reader.read_to_end(&mut retrieved).expect("read");
        assert_eq!(&retrieved, expected);
    }
}
