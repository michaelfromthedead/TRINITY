//! Whitebox tests for trinity-harness parsers module.
//!
//! WHITEBOX coverage plan:
//!   - ParserRegistry::new() initializes all parsers
//!   - ParserRegistry::default() delegates to new()
//!   - detect_language() for each extension (.rs, .py, .wgsl, unknown, no extension)
//!   - RustParser: function, struct, impl trait, module, empty, invalid syntax
//!   - PythonParser: function, async function, class, empty, invalid syntax
//!   - WgslParser: struct, function, entry point, empty, invalid syntax
//!   - Language and UnitType enum variants

use std::path::Path;
use trinity_harness::ParserRegistry;

// Re-export internal types for whitebox testing via the crate's public interface
// Since parsers module is public, we can access its types

#[test]
fn test_parser_registry_new_succeeds() {
    // ParserRegistry::new() should succeed
    let _registry = ParserRegistry::new();
}

#[test]
fn test_parser_registry_default_succeeds() {
    // Default impl should work
    let _registry = ParserRegistry::default();
}

#[test]
fn test_detect_language_rust() {
    let path = Path::new("src/main.rs");
    let lang = ParserRegistry::detect_language(path);
    assert!(lang.is_some(), "should detect .rs as Rust");
}

#[test]
fn test_detect_language_python() {
    let path = Path::new("script.py");
    let lang = ParserRegistry::detect_language(path);
    assert!(lang.is_some(), "should detect .py as Python");
}

#[test]
fn test_detect_language_wgsl() {
    let path = Path::new("shader.wgsl");
    let lang = ParserRegistry::detect_language(path);
    assert!(lang.is_some(), "should detect .wgsl as WGSL");
}

#[test]
fn test_detect_language_unknown_extension() {
    let path = Path::new("file.txt");
    let lang = ParserRegistry::detect_language(path);
    assert!(lang.is_none(), "should return None for unknown extension");
}

#[test]
fn test_detect_language_no_extension() {
    let path = Path::new("Makefile");
    let lang = ParserRegistry::detect_language(path);
    assert!(lang.is_none(), "should return None for no extension");
}

#[test]
fn test_detect_language_hidden_file() {
    let path = Path::new(".gitignore");
    let lang = ParserRegistry::detect_language(path);
    assert!(lang.is_none(), "should return None for hidden file");
}
