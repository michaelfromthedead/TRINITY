//! Blackbox tests for T-HARNESS-1.1: Crate skeleton verification
//!
//! These tests verify that the trinity-harness crate exposes the public API
//! as defined in PHASE_1_INFRASTRUCTURE_ARCH.md. Tests are written from
//! an external user's perspective without knowledge of implementation details.

use trinity_harness::{HarnessDb, ParserRegistry};
use trinity_harness::parsers::{CodeUnit, Language};
use std::path::Path;

/// Test that HarnessDb type exists and is publicly accessible
#[test]
fn harness_db_type_exists() {
    fn _assert_type<T>() {}
    _assert_type::<HarnessDb>();
}

/// Test that HarnessDb::open() method exists with expected signature
/// Contract: HarnessDb::open(path: &str) -> Result<Self>
#[test]
fn harness_db_open_method_exists() {
    // Verify the open method exists by creating a temporary database
    let temp_dir = std::env::temp_dir();
    let db_path = temp_dir.join("blackbox_test_harness.db");
    let path_str = db_path.to_str().expect("valid path");

    // The contract states open() takes a &str path and returns Result<Self>
    let result = HarnessDb::open(path_str);

    // We only care that the method exists and returns a Result
    // Success or failure depends on implementation, but Result must be returned
    assert!(result.is_ok() || result.is_err(), "open() must return a Result");

    // Clean up
    let _ = std::fs::remove_file(&db_path);
}

/// Test that ParserRegistry type exists and is publicly accessible
#[test]
fn parser_registry_type_exists() {
    fn _assert_type<T>() {}
    _assert_type::<ParserRegistry>();
}

/// Test that CodeUnit type exists (unified enum spanning all languages)
#[test]
fn code_unit_type_exists() {
    fn _assert_type<T>() {}
    _assert_type::<CodeUnit>();
}

/// Test that Language enum exists
#[test]
fn language_enum_exists() {
    fn _assert_type<T>() {}
    _assert_type::<Language>();
}

/// Test that ParserRegistry can be constructed
/// Contract: ParserRegistry contains rust, python, wgsl parsers
#[test]
fn parser_registry_can_be_constructed() {
    // The contract shows ParserRegistry has a constructor or Default
    // Try Default if available, otherwise new()
    let _registry = ParserRegistry::new();
}

/// Test that ParserRegistry::parse_file method exists with expected signature
/// Contract: parse_file(&self, path: &Path, source: &str, lang: Language) -> Vec<CodeUnit>
#[test]
fn parser_registry_parse_file_exists() {
    let registry = ParserRegistry::new();
    let path = Path::new("test.rs");
    let source = "";
    let lang = Language::Rust;

    // The contract states parse_file returns Vec<CodeUnit>
    let result: Vec<CodeUnit> = registry.parse_file(path, source, lang);

    // Empty source should return empty or some valid Vec
    // We only verify the method exists and returns the correct type
    let _ = result;
}

/// Test that Language enum has Rust variant
#[test]
fn language_enum_has_rust_variant() {
    let _lang = Language::Rust;
}

/// Test that Language enum has Python variant
#[test]
fn language_enum_has_python_variant() {
    let _lang = Language::Python;
}

/// Test that Language enum has Wgsl variant
#[test]
fn language_enum_has_wgsl_variant() {
    let _lang = Language::Wgsl;
}
