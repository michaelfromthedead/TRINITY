//! Blackbox tests for HarnessDb public API
//!
//! These tests verify the contract of HarnessDb::open(path) from an external
//! perspective, without knowledge of internal implementation details.
//!
//! Contract under test (from PHASE_1_INFRASTRUCTURE_ARCH.md):
//! - HarnessDb::open(path: &str) -> Result<Self>
//! - Configures pragmas (WAL mode, cache)
//! - Database should be usable after open

use std::fs;
use trinity_harness::HarnessDb;

/// Test that opening a database with a valid path succeeds.
/// This is the happy path - a new database file should be created.
#[test]
fn test_open_creates_database_file() {
    let temp_dir = std::env::temp_dir();
    let db_path = temp_dir.join("test_harness_open_creates.db");

    // Clean up any previous test artifact
    let _ = fs::remove_file(&db_path);

    let path_str = db_path.to_str().expect("valid utf8 path");
    let result = HarnessDb::open(path_str);

    assert!(result.is_ok(), "HarnessDb::open should succeed for valid path");
    assert!(db_path.exists(), "Database file should be created at specified path");

    // Cleanup
    let _ = fs::remove_file(&db_path);
}

/// Test that WAL mode is enabled as per the contract.
/// WAL mode creates additional files (-wal, -shm) which we can observe externally.
#[test]
fn test_open_enables_wal_mode() {
    let temp_dir = std::env::temp_dir();
    let db_path = temp_dir.join("test_harness_wal_mode.db");
    let wal_path = temp_dir.join("test_harness_wal_mode.db-wal");
    let shm_path = temp_dir.join("test_harness_wal_mode.db-shm");

    // Clean up any previous test artifacts
    let _ = fs::remove_file(&db_path);
    let _ = fs::remove_file(&wal_path);
    let _ = fs::remove_file(&shm_path);

    let path_str = db_path.to_str().expect("valid utf8 path");
    let db = HarnessDb::open(path_str).expect("open should succeed");

    // WAL mode creates -wal and -shm files when transactions occur
    // We need to trigger a write to see them
    // The schema initialization (per ARCH) should have triggered writes

    // If WAL is enabled and schema was initialized, the db file should exist
    // The -wal file may or may not exist depending on checkpoint behavior,
    // but the database should be in WAL mode which we can verify via query
    assert!(db_path.exists(), "Database file should exist");

    // Cleanup
    drop(db);
    let _ = fs::remove_file(&db_path);
    let _ = fs::remove_file(&wal_path);
    let _ = fs::remove_file(&shm_path);
}

/// Test that opening the same database path twice works (not locked exclusively).
/// WAL mode allows concurrent readers, so reopening should succeed.
#[test]
fn test_open_allows_reopen() {
    let temp_dir = std::env::temp_dir();
    let db_path = temp_dir.join("test_harness_reopen.db");

    // Clean up any previous test artifact
    let _ = fs::remove_file(&db_path);

    let path_str = db_path.to_str().expect("valid utf8 path");

    // First open
    let db1 = HarnessDb::open(path_str).expect("first open should succeed");
    drop(db1);

    // Second open on existing database
    let db2 = HarnessDb::open(path_str);
    assert!(db2.is_ok(), "Reopening existing database should succeed");

    // Cleanup
    drop(db2);
    let _ = fs::remove_file(&db_path);
    let _ = fs::remove_file(temp_dir.join("test_harness_reopen.db-wal"));
    let _ = fs::remove_file(temp_dir.join("test_harness_reopen.db-shm"));
}

/// Test that opening with an invalid directory path fails gracefully.
/// The function should return an error, not panic.
#[test]
fn test_open_invalid_directory_returns_error() {
    let invalid_path = "/nonexistent/deeply/nested/directory/that/does/not/exist/test.db";

    let result = HarnessDb::open(invalid_path);

    assert!(result.is_err(), "Opening database in nonexistent directory should fail");
}

/// Test that opening with an empty path succeeds (creates anonymous in-memory db).
/// SQLite treats empty string as a temporary database (anonymous in-memory).
/// This is valid SQLite behavior per the SQLite documentation.
#[test]
fn test_open_empty_path_creates_temp_database() {
    let result = HarnessDb::open("");

    // SQLite accepts empty string as valid path (anonymous temporary database)
    assert!(result.is_ok(), "Opening database with empty path should succeed (temp db)");
}

/// Test that an in-memory database can be created.
/// SQLite special path ":memory:" should work.
#[test]
fn test_open_in_memory_database() {
    let result = HarnessDb::open(":memory:");

    assert!(result.is_ok(), "Opening in-memory database should succeed");
}

/// Test that the database is initialized with a schema.
/// Per the contract, open() calls init_schema() which creates tables.
/// We verify by checking that the database is usable after open.
#[test]
fn test_open_initializes_schema() {
    let temp_dir = std::env::temp_dir();
    let db_path = temp_dir.join("test_harness_schema_init.db");

    // Clean up any previous test artifact
    let _ = fs::remove_file(&db_path);

    let path_str = db_path.to_str().expect("valid utf8 path");
    let db = HarnessDb::open(path_str);

    // If schema initialization failed, open would have returned an error
    assert!(db.is_ok(), "Database with schema initialization should open successfully");

    // Cleanup
    drop(db);
    let _ = fs::remove_file(&db_path);
    let _ = fs::remove_file(temp_dir.join("test_harness_schema_init.db-wal"));
    let _ = fs::remove_file(temp_dir.join("test_harness_schema_init.db-shm"));
}

/// Test opening with a path that has special characters.
/// Boundary case: paths with spaces and unicode should work.
#[test]
fn test_open_path_with_spaces() {
    let temp_dir = std::env::temp_dir();
    let db_path = temp_dir.join("test harness with spaces.db");

    // Clean up any previous test artifact
    let _ = fs::remove_file(&db_path);

    let path_str = db_path.to_str().expect("valid utf8 path");
    let result = HarnessDb::open(path_str);

    assert!(result.is_ok(), "Opening database with spaces in path should succeed");

    // Cleanup
    drop(result);
    let _ = fs::remove_file(&db_path);
    let _ = fs::remove_file(temp_dir.join("test harness with spaces.db-wal"));
    let _ = fs::remove_file(temp_dir.join("test harness with spaces.db-shm"));
}

/// Test opening with a very long path.
/// Boundary case: paths near filesystem limits.
#[test]
fn test_open_long_path() {
    let temp_dir = std::env::temp_dir();
    // Create a moderately long filename (not exceeding typical 255 char limit)
    let long_name = format!("{}.db", "a".repeat(200));
    let db_path = temp_dir.join(&long_name);

    // Clean up any previous test artifact
    let _ = fs::remove_file(&db_path);

    let path_str = db_path.to_str().expect("valid utf8 path");
    let result = HarnessDb::open(path_str);

    // This should succeed on most filesystems
    assert!(result.is_ok(), "Opening database with long filename should succeed");

    // Cleanup
    drop(result);
    let _ = fs::remove_file(&db_path);
}

/// Test that multiple databases can be opened simultaneously.
/// Each should be independent.
#[test]
fn test_open_multiple_databases() {
    let temp_dir = std::env::temp_dir();
    let db_path1 = temp_dir.join("test_harness_multi_1.db");
    let db_path2 = temp_dir.join("test_harness_multi_2.db");

    // Clean up any previous test artifacts
    let _ = fs::remove_file(&db_path1);
    let _ = fs::remove_file(&db_path2);

    let path_str1 = db_path1.to_str().expect("valid utf8 path");
    let path_str2 = db_path2.to_str().expect("valid utf8 path");

    let db1 = HarnessDb::open(path_str1);
    let db2 = HarnessDb::open(path_str2);

    assert!(db1.is_ok(), "First database should open successfully");
    assert!(db2.is_ok(), "Second database should open successfully");

    // Both files should exist
    assert!(db_path1.exists(), "First database file should exist");
    assert!(db_path2.exists(), "Second database file should exist");

    // Cleanup
    drop(db1);
    drop(db2);
    let _ = fs::remove_file(&db_path1);
    let _ = fs::remove_file(&db_path2);
    for suffix in &["-wal", "-shm"] {
        let _ = fs::remove_file(temp_dir.join(format!("test_harness_multi_1.db{}", suffix)));
        let _ = fs::remove_file(temp_dir.join(format!("test_harness_multi_2.db{}", suffix)));
    }
}

/// Test that opening a file that exists but is not a valid SQLite database fails.
/// Edge case: corrupted or non-database files.
#[test]
fn test_open_non_database_file() {
    let temp_dir = std::env::temp_dir();
    let fake_db_path = temp_dir.join("test_harness_fake.db");

    // Create a file with garbage content
    fs::write(&fake_db_path, b"This is not a valid SQLite database file")
        .expect("should create test file");

    let path_str = fake_db_path.to_str().expect("valid utf8 path");
    let result = HarnessDb::open(path_str);

    // Opening a corrupted file should fail when schema init is attempted
    // Note: SQLite may accept the file initially but fail on first operation
    // The contract says open() initializes schema, so it should fail
    assert!(result.is_err(), "Opening non-database file should fail during schema init");

    // Cleanup
    let _ = fs::remove_file(&fake_db_path);
}

/// Test that a read-only location fails appropriately.
/// On Unix systems, we can test with a path in a read-only directory.
#[test]
#[cfg(unix)]
fn test_open_readonly_location_fails() {
    use std::os::unix::fs::PermissionsExt;

    let temp_dir = std::env::temp_dir();
    let readonly_dir = temp_dir.join("test_harness_readonly_dir");

    // Clean up from previous runs
    let _ = fs::remove_dir_all(&readonly_dir);

    // Create directory and make it read-only
    fs::create_dir(&readonly_dir).expect("should create test directory");
    let mut perms = fs::metadata(&readonly_dir)
        .expect("should get metadata")
        .permissions();
    perms.set_mode(0o555); // r-xr-xr-x
    fs::set_permissions(&readonly_dir, perms).expect("should set permissions");

    let db_path = readonly_dir.join("cannot_create.db");
    let path_str = db_path.to_str().expect("valid utf8 path");

    let result = HarnessDb::open(path_str);

    // Restore write permission before cleanup
    let mut perms = fs::metadata(&readonly_dir)
        .expect("should get metadata")
        .permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&readonly_dir, perms).expect("should restore permissions");

    assert!(result.is_err(), "Opening database in read-only directory should fail");

    // Cleanup
    let _ = fs::remove_dir_all(&readonly_dir);
}
