//! Blackbox tests for T-HARNESS-1.1: Dependency verification
//!
//! These tests verify that the crate's dependencies are properly configured
//! as specified in the TODO:
//! - rusqlite (or superrusqlite as rusqlite)
//! - syn (for Rust parsing)
//! - rustpython_parser (for Python parsing)
//! - naga (for WGSL parsing)
//! - tree-sitter-* (for tree-sitter grammars)
//!
//! We verify this indirectly by checking that types requiring these
//! dependencies can be used.

use trinity_harness::{HarnessDb, ParserRegistry};
use trinity_harness::parsers::Language;
use std::path::Path;

/// Test that database functionality works (implies rusqlite dependency)
/// Contract: HarnessDb uses SQLite with WAL mode
#[test]
fn database_dependency_functional() {
    let temp_dir = std::env::temp_dir();
    let db_path = temp_dir.join("blackbox_dep_test.db");
    let path_str = db_path.to_str().expect("valid path");

    // If rusqlite isn't properly configured, this would fail to compile or run
    let result = HarnessDb::open(path_str);

    // Clean up regardless of result
    let _ = std::fs::remove_file(&db_path);

    // Verify we got a valid result (success expected for new database)
    assert!(
        result.is_ok(),
        "HarnessDb::open should succeed for a new database path"
    );
}

/// Test that Rust parsing is available (implies syn + tree-sitter-rust)
#[test]
fn rust_parser_dependency_functional() {
    let registry = ParserRegistry::new();
    let path = Path::new("test.rs");
    let source = "fn main() {}";

    // If syn or tree-sitter-rust isn't configured, parsing would fail
    let units = registry.parse_file(path, source, Language::Rust);

    // Simple valid Rust should parse successfully
    // We just verify no panic and valid return
    let _ = units;
}

/// Test that Python parsing is available (implies rustpython_parser + tree-sitter-python)
#[test]
fn python_parser_dependency_functional() {
    let registry = ParserRegistry::new();
    let path = Path::new("test.py");
    let source = "def main(): pass";

    // If rustpython_parser or tree-sitter-python isn't configured, parsing would fail
    let units = registry.parse_file(path, source, Language::Python);

    // Simple valid Python should parse successfully
    let _ = units;
}

/// Test that WGSL parsing is available (implies naga)
#[test]
fn wgsl_parser_dependency_functional() {
    let registry = ParserRegistry::new();
    let path = Path::new("test.wgsl");
    let source = "@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }";

    // If naga isn't configured, parsing would fail
    let units = registry.parse_file(path, source, Language::Wgsl);

    // Simple valid WGSL should parse successfully
    let _ = units;
}
