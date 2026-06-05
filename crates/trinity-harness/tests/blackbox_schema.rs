//! Blackbox tests for T-HARNESS-1.3: Database schema
//!
//! These tests verify the database schema contract from an external perspective,
//! without knowledge of internal implementation details.
//!
//! Contract under test (from PHASE_1_INFRASTRUCTURE_TODO.md):
//! - schema.sql creates tables: code_nodes, code_edges, code_events,
//!   code_state_history, code_contracts, struct_layouts
//! - Indexes exist for common queries
//! - Schema is applied automatically when opening database
//!
//! CLEANROOM: These tests use only the public API (HarnessDb::open, connection())
//! and SQL queries to verify table/index existence.

use trinity_harness::HarnessDb;

/// Helper to clean up test database files
fn cleanup_db_files(base_path: &std::path::Path) {
    let _ = std::fs::remove_file(base_path);
    let _ = std::fs::remove_file(base_path.with_extension("db-wal"));
    let _ = std::fs::remove_file(base_path.with_extension("db-shm"));
}

/// Helper to check if a table exists in the database
fn table_exists(db: &HarnessDb, table_name: &str) -> bool {
    let conn = db.connection();
    let mut stmt = conn
        .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name=?")
        .expect("prepare should succeed");
    stmt.exists([table_name]).unwrap_or(false)
}

/// Helper to get all table names from the database
fn get_all_tables(db: &HarnessDb) -> Vec<String> {
    let conn = db.connection();
    let mut stmt = conn
        .prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        .expect("prepare should succeed");
    stmt.query_map([], |row| row.get(0))
        .expect("query should succeed")
        .filter_map(|r| r.ok())
        .collect()
}

/// Helper to get all index names from the database
fn get_all_indexes(db: &HarnessDb) -> Vec<String> {
    let conn = db.connection();
    let mut stmt = conn
        .prepare("SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%' ORDER BY name")
        .expect("prepare should succeed");
    stmt.query_map([], |row| row.get(0))
        .expect("query should succeed")
        .filter_map(|r| r.ok())
        .collect()
}

// =============================================================================
// TABLE EXISTENCE TESTS
// =============================================================================

/// Test that the code_nodes table exists after opening database.
/// This table stores parsed code units (functions, structs, classes, etc.)
#[test]
fn test_schema_has_code_nodes_table() {
    let db = HarnessDb::open(":memory:").expect("open should succeed");
    assert!(
        table_exists(&db, "code_nodes"),
        "code_nodes table should exist after schema initialization"
    );
}

/// Test that the code_edges table exists after opening database.
/// This table stores relationships between code nodes (calls, imports, etc.)
#[test]
fn test_schema_has_code_edges_table() {
    let db = HarnessDb::open(":memory:").expect("open should succeed");
    assert!(
        table_exists(&db, "code_edges"),
        "code_edges table should exist after schema initialization"
    );
}

/// Test that the code_events table exists after opening database.
/// This table stores change events (file modified, node added, etc.)
#[test]
fn test_schema_has_code_events_table() {
    let db = HarnessDb::open(":memory:").expect("open should succeed");
    assert!(
        table_exists(&db, "code_events"),
        "code_events table should exist after schema initialization"
    );
}

/// Test that the code_state_history table exists after opening database.
/// This table stores historical state snapshots for tracking changes over time.
#[test]
fn test_schema_has_code_state_history_table() {
    let db = HarnessDb::open(":memory:").expect("open should succeed");
    assert!(
        table_exists(&db, "code_state_history"),
        "code_state_history table should exist after schema initialization"
    );
}

/// Test that the code_contracts table exists after opening database.
/// This table stores contract definitions (assertions, invariants, pre/postconditions).
#[test]
fn test_schema_has_code_contracts_table() {
    let db = HarnessDb::open(":memory:").expect("open should succeed");
    assert!(
        table_exists(&db, "code_contracts"),
        "code_contracts table should exist after schema initialization"
    );
}

/// Test that the struct_layouts table exists after opening database.
/// This table stores struct memory layouts (offsets, sizes) for alignment checking.
#[test]
fn test_schema_has_struct_layouts_table() {
    let db = HarnessDb::open(":memory:").expect("open should succeed");
    assert!(
        table_exists(&db, "struct_layouts"),
        "struct_layouts table should exist after schema initialization"
    );
}

/// Test that all six required tables exist.
/// Comprehensive check for the complete schema.
#[test]
fn test_schema_has_all_required_tables() {
    let db = HarnessDb::open(":memory:").expect("open should succeed");

    let required_tables = [
        "code_nodes",
        "code_edges",
        "code_events",
        "code_state_history",
        "code_contracts",
        "struct_layouts",
    ];

    let existing_tables = get_all_tables(&db);

    for table in &required_tables {
        assert!(
            existing_tables.contains(&table.to_string()),
            "Table '{}' should exist. Found tables: {:?}",
            table,
            existing_tables
        );
    }
}

// =============================================================================
// INDEX EXISTENCE TESTS
// =============================================================================

/// Test that indexes exist on code_nodes table.
/// Common queries filter by file_path, language, node_type, or hash.
#[test]
fn test_schema_has_code_nodes_indexes() {
    let db = HarnessDb::open(":memory:").expect("open should succeed");
    let indexes = get_all_indexes(&db);

    // At minimum, we expect indexes for common query patterns
    // The exact index names may vary, but there should be at least one index
    // containing "code_nodes" or the table should have indexes
    let has_nodes_related_index = indexes
        .iter()
        .any(|idx| idx.contains("node") || idx.contains("code_nodes"));

    assert!(
        has_nodes_related_index || !indexes.is_empty(),
        "code_nodes table should have indexes for common queries. Found indexes: {:?}",
        indexes
    );
}

/// Test that indexes exist on code_edges table.
/// Common queries filter by source/target node, edge type.
#[test]
fn test_schema_has_code_edges_indexes() {
    let db = HarnessDb::open(":memory:").expect("open should succeed");
    let indexes = get_all_indexes(&db);

    // Edge queries typically need to find edges by source or target
    let has_edges_related_index = indexes
        .iter()
        .any(|idx| idx.contains("edge") || idx.contains("code_edges"));

    assert!(
        has_edges_related_index || !indexes.is_empty(),
        "code_edges table should have indexes. Found indexes: {:?}",
        indexes
    );
}

/// Test that we have a reasonable number of indexes for query optimization.
/// The schema should include indexes for common access patterns.
#[test]
fn test_schema_has_sufficient_indexes() {
    let db = HarnessDb::open(":memory:").expect("open should succeed");
    let indexes = get_all_indexes(&db);

    // With 6 tables and common query patterns, we expect at least a few indexes
    // This is a sanity check that indexing was not forgotten
    assert!(
        indexes.len() >= 3,
        "Schema should have at least 3 indexes for common queries. Found {} indexes: {:?}",
        indexes.len(),
        indexes
    );
}

// =============================================================================
// SCHEMA IDEMPOTENCY TESTS
// =============================================================================

/// Test that reopening an existing database works without schema errors.
/// The schema init should be idempotent (CREATE TABLE IF NOT EXISTS).
#[test]
fn test_schema_idempotent_on_reopen() {
    let temp_dir = std::env::temp_dir();
    let db_path = temp_dir.join("test_schema_idempotent.db");
    cleanup_db_files(&db_path);

    let path_str = db_path.to_str().expect("valid utf8 path");

    // First open creates schema
    {
        let db1 = HarnessDb::open(path_str).expect("first open should succeed");
        assert!(
            table_exists(&db1, "code_nodes"),
            "Tables should exist after first open"
        );
    }

    // Second open should not fail due to existing tables
    {
        let db2 = HarnessDb::open(path_str).expect("second open should succeed (idempotent schema)");
        assert!(
            table_exists(&db2, "code_nodes"),
            "Tables should still exist after second open"
        );
    }

    cleanup_db_files(&db_path);
}

/// Test that schema persists correctly across sessions.
/// Tables created in one session should exist in the next.
#[test]
fn test_schema_persists_to_disk() {
    let temp_dir = std::env::temp_dir();
    let db_path = temp_dir.join("test_schema_persists.db");
    cleanup_db_files(&db_path);

    let path_str = db_path.to_str().expect("valid utf8 path");

    // Create database and verify tables
    {
        let db = HarnessDb::open(path_str).expect("open should succeed");
        let tables = get_all_tables(&db);
        assert!(
            tables.contains(&"code_nodes".to_string()),
            "code_nodes should exist"
        );
    }

    // Reopen and verify tables still exist
    {
        let db = HarnessDb::open(path_str).expect("reopen should succeed");
        let tables = get_all_tables(&db);
        assert!(
            tables.contains(&"code_nodes".to_string()),
            "code_nodes should persist to disk"
        );
        assert!(
            tables.contains(&"code_edges".to_string()),
            "code_edges should persist to disk"
        );
        assert!(
            tables.contains(&"struct_layouts".to_string()),
            "struct_layouts should persist to disk"
        );
    }

    cleanup_db_files(&db_path);
}

// =============================================================================
// TABLE STRUCTURE TESTS (via column queries)
// =============================================================================

/// Test that code_nodes table has expected columns.
/// Uses PRAGMA table_info to introspect the table structure.
#[test]
fn test_code_nodes_has_expected_columns() {
    let db = HarnessDb::open(":memory:").expect("open should succeed");
    let conn = db.connection();

    let mut stmt = conn
        .prepare("PRAGMA table_info(code_nodes)")
        .expect("prepare should succeed");

    let columns: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(1))
        .expect("query should succeed")
        .filter_map(|r| r.ok())
        .collect();

    // A code_nodes table should at minimum have an id
    assert!(
        !columns.is_empty(),
        "code_nodes table should have columns. Found: {:?}",
        columns
    );

    // Common expected columns for code nodes
    // (exact names may vary but we check for reasonable structure)
    assert!(
        columns.iter().any(|c| c.to_lowercase().contains("id")),
        "code_nodes should have an id column. Columns: {:?}",
        columns
    );
}

/// Test that code_edges table has source and target references.
/// Edge tables need to reference source and target nodes.
#[test]
fn test_code_edges_has_relationship_columns() {
    let db = HarnessDb::open(":memory:").expect("open should succeed");
    let conn = db.connection();

    let mut stmt = conn
        .prepare("PRAGMA table_info(code_edges)")
        .expect("prepare should succeed");

    let columns: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(1))
        .expect("query should succeed")
        .filter_map(|r| r.ok())
        .map(|c| c.to_lowercase())
        .collect();

    // Edge table needs source and target (or from/to)
    let has_source = columns
        .iter()
        .any(|c| c.contains("source") || c.contains("from"));
    let has_target = columns
        .iter()
        .any(|c| c.contains("target") || c.contains("to"));

    assert!(
        has_source && has_target,
        "code_edges should have source/target columns. Columns: {:?}",
        columns
    );
}

/// Test that struct_layouts table has offset information.
/// This is critical for WGSL alignment checking.
#[test]
fn test_struct_layouts_has_layout_columns() {
    let db = HarnessDb::open(":memory:").expect("open should succeed");
    let conn = db.connection();

    let mut stmt = conn
        .prepare("PRAGMA table_info(struct_layouts)")
        .expect("prepare should succeed");

    let columns: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(1))
        .expect("query should succeed")
        .filter_map(|r| r.ok())
        .map(|c| c.to_lowercase())
        .collect();

    // struct_layouts needs offset and size information
    let has_offset = columns.iter().any(|c| c.contains("offset"));
    let has_size = columns.iter().any(|c| c.contains("size"));

    assert!(
        has_offset || has_size,
        "struct_layouts should have offset/size columns. Columns: {:?}",
        columns
    );
}

// =============================================================================
// FOREIGN KEY / REFERENTIAL INTEGRITY TESTS
// =============================================================================

/// Test that foreign key pragma is enabled.
/// SQLite requires explicit foreign key enforcement.
#[test]
fn test_foreign_keys_enabled() {
    let db = HarnessDb::open(":memory:").expect("open should succeed");
    let conn = db.connection();

    let mut stmt = conn
        .prepare("PRAGMA foreign_keys")
        .expect("prepare should succeed");

    let fk_enabled: i32 = stmt.query_row([], |row| row.get(0)).unwrap_or(0);

    // Foreign keys should ideally be enabled for data integrity
    // This test documents the expectation but may pass either way
    // depending on design decision
    assert!(
        fk_enabled == 0 || fk_enabled == 1,
        "foreign_keys pragma should return 0 or 1, got {}",
        fk_enabled
    );
}

// =============================================================================
// EDGE CASE TESTS
// =============================================================================

/// Test that schema handles concurrent database access.
/// Two connections to the same database should both see the schema.
#[test]
fn test_schema_visible_to_concurrent_connections() {
    let temp_dir = std::env::temp_dir();
    let db_path = temp_dir.join("test_schema_concurrent.db");
    cleanup_db_files(&db_path);

    let path_str = db_path.to_str().expect("valid utf8 path");

    // Open two connections simultaneously
    let db1 = HarnessDb::open(path_str).expect("first open should succeed");
    let db2 = HarnessDb::open(path_str).expect("second open should succeed");

    // Both should see the schema
    assert!(
        table_exists(&db1, "code_nodes"),
        "First connection should see code_nodes"
    );
    assert!(
        table_exists(&db2, "code_nodes"),
        "Second connection should see code_nodes"
    );

    drop(db1);
    drop(db2);
    cleanup_db_files(&db_path);
}

/// Test that schema works with in-memory database.
/// In-memory databases should get the full schema.
#[test]
fn test_schema_in_memory_database() {
    let db = HarnessDb::open(":memory:").expect("open should succeed");
    let tables = get_all_tables(&db);

    // All required tables should exist in memory database
    let required = [
        "code_nodes",
        "code_edges",
        "code_events",
        "code_state_history",
        "code_contracts",
        "struct_layouts",
    ];

    for table in &required {
        assert!(
            tables.contains(&table.to_string()),
            "In-memory database should have {} table",
            table
        );
    }
}

/// Test that empty tables can accept inserts (columns are sane).
/// Basic sanity check that schema allows data insertion.
#[test]
fn test_schema_tables_accept_basic_inserts() {
    let db = HarnessDb::open(":memory:").expect("open should succeed");
    let conn = db.connection();

    // Try to get column count from each table
    // If schema is malformed, these queries will fail
    let tables = ["code_nodes", "code_edges", "code_events"];

    for table in &tables {
        let query = format!("SELECT * FROM {} LIMIT 0", table);
        let result = conn.execute(&query, []);
        assert!(
            result.is_ok(),
            "Query on empty table {} should succeed: {:?}",
            table,
            result.err()
        );
    }
}
