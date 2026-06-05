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

// ============================================================================
// T-HARNESS-1.7: Unified CodeUnit Tests
// ============================================================================

use trinity_harness::{CodeUnit, Language, UnitType};

// ----------------------------------------------------------------------------
// Language Field Correctness Tests
// ----------------------------------------------------------------------------

#[test]
fn test_rust_parser_sets_language_rust() {
    let registry = ParserRegistry::new();
    let source = r#"
fn foo() {}
struct Bar { x: i32 }
enum Qux { A, B }
"#;
    let units = registry.parse(source, Language::Rust);
    assert!(!units.is_empty(), "should parse Rust code units");
    for unit in &units {
        assert_eq!(
            unit.language,
            Language::Rust,
            "Rust parser should set language to Rust for unit '{}'",
            unit.name
        );
    }
}

#[test]
fn test_python_parser_sets_language_python() {
    let registry = ParserRegistry::new();
    let source = r#"
def foo():
    pass

class Bar:
    pass

async def baz():
    pass
"#;
    let units = registry.parse(source, Language::Python);
    assert!(!units.is_empty(), "should parse Python code units");
    for unit in &units {
        assert_eq!(
            unit.language,
            Language::Python,
            "Python parser should set language to Python for unit '{}'",
            unit.name
        );
    }
}

#[test]
fn test_wgsl_parser_sets_language_wgsl() {
    let registry = ParserRegistry::new();
    let source = r#"
struct Uniforms {
    mvp: mat4x4<f32>,
}

fn helper() -> f32 {
    return 1.0;
}

@vertex
fn vs_main() -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
"#;
    let units = registry.parse(source, Language::Wgsl);
    assert!(!units.is_empty(), "should parse WGSL code units");
    for unit in &units {
        assert_eq!(
            unit.language,
            Language::Wgsl,
            "WGSL parser should set language to WGSL for unit '{}'",
            unit.name
        );
    }
}

// ----------------------------------------------------------------------------
// parse_file_auto() Auto-Detection Tests
// ----------------------------------------------------------------------------

#[test]
fn test_parse_file_auto_rust() {
    let registry = ParserRegistry::new();
    let path = Path::new("src/main.rs");
    let source = "fn main() {}";

    let result = registry.parse_file_auto(path, source);
    assert!(result.is_some(), "should auto-detect .rs as Rust");

    let units = result.unwrap();
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "main");
    assert_eq!(units[0].language, Language::Rust);
}

#[test]
fn test_parse_file_auto_python() {
    let registry = ParserRegistry::new();
    let path = Path::new("script.py");
    let source = "def hello():\n    pass";

    let result = registry.parse_file_auto(path, source);
    assert!(result.is_some(), "should auto-detect .py as Python");

    let units = result.unwrap();
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "hello");
    assert_eq!(units[0].language, Language::Python);
}

#[test]
fn test_parse_file_auto_wgsl() {
    let registry = ParserRegistry::new();
    let path = Path::new("shaders/pbr.wgsl");
    let source = "struct Light { intensity: f32 }";

    let result = registry.parse_file_auto(path, source);
    assert!(result.is_some(), "should auto-detect .wgsl as WGSL");

    let units = result.unwrap();
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "Light");
    assert_eq!(units[0].language, Language::Wgsl);
}

#[test]
fn test_parse_file_auto_unknown_extension_returns_none() {
    let registry = ParserRegistry::new();
    let path = Path::new("README.md");
    let source = "# Hello World";

    let result = registry.parse_file_auto(path, source);
    assert!(result.is_none(), "should return None for unknown extension");
}

#[test]
fn test_parse_file_auto_no_extension_returns_none() {
    let registry = ParserRegistry::new();
    let path = Path::new("Makefile");
    let source = "all: build";

    let result = registry.parse_file_auto(path, source);
    assert!(result.is_none(), "should return None for no extension");
}

#[test]
fn test_parse_file_auto_deeply_nested_path() {
    let registry = ParserRegistry::new();
    let path = Path::new("a/b/c/d/e/f/module.py");
    let source = "class DeepClass:\n    pass";

    let result = registry.parse_file_auto(path, source);
    assert!(result.is_some(), "should detect language from deeply nested path");

    let units = result.unwrap();
    assert_eq!(units[0].language, Language::Python);
}

// ----------------------------------------------------------------------------
// detect_language() Extension Tests
// ----------------------------------------------------------------------------

#[test]
fn test_detect_language_returns_correct_enum_rust() {
    let path = Path::new("lib.rs");
    let lang = ParserRegistry::detect_language(path);
    assert_eq!(lang, Some(Language::Rust));
}

#[test]
fn test_detect_language_returns_correct_enum_python() {
    let path = Path::new("app.py");
    let lang = ParserRegistry::detect_language(path);
    assert_eq!(lang, Some(Language::Python));
}

#[test]
fn test_detect_language_returns_correct_enum_wgsl() {
    let path = Path::new("compute.wgsl");
    let lang = ParserRegistry::detect_language(path);
    assert_eq!(lang, Some(Language::Wgsl));
}

#[test]
fn test_detect_language_case_sensitive() {
    // Extensions should be case-sensitive (standard behavior)
    let path_upper_rs = Path::new("file.RS");
    let path_upper_py = Path::new("file.PY");
    let path_upper_wgsl = Path::new("file.WGSL");

    // These should NOT be detected as valid (case mismatch)
    assert!(
        ParserRegistry::detect_language(path_upper_rs).is_none(),
        ".RS should not be detected (case-sensitive)"
    );
    assert!(
        ParserRegistry::detect_language(path_upper_py).is_none(),
        ".PY should not be detected (case-sensitive)"
    );
    assert!(
        ParserRegistry::detect_language(path_upper_wgsl).is_none(),
        ".WGSL should not be detected (case-sensitive)"
    );
}

#[test]
fn test_detect_language_similar_extensions_not_matched() {
    // Ensure partial matches don't work
    let paths = [
        Path::new("file.rsx"),     // Similar to .rs
        Path::new("file.pyw"),     // Python Windows variant
        Path::new("file.glsl"),    // Different shader language
        Path::new("file.hlsl"),    // DirectX shader
        Path::new("file.rust"),    // Wrong extension
        Path::new("file.python"),  // Wrong extension
    ];

    for path in paths {
        assert!(
            ParserRegistry::detect_language(path).is_none(),
            "Should not detect {:?} as valid language",
            path
        );
    }
}

// ----------------------------------------------------------------------------
// Cross-Language Consistency Tests
// ----------------------------------------------------------------------------

/// Helper to verify all CodeUnit fields are populated
fn verify_code_unit_populated(unit: &CodeUnit, context: &str) {
    assert!(!unit.name.is_empty(), "{}: name should not be empty", context);
    assert!(unit.start_line > 0, "{}: start_line should be > 0", context);
    assert!(
        unit.end_line >= unit.start_line,
        "{}: end_line ({}) should be >= start_line ({})",
        context,
        unit.end_line,
        unit.start_line
    );
    // full_hash should be non-zero for actual code
    let zero_hash = [0u8; 32];
    assert_ne!(
        unit.hashes.full_hash, zero_hash,
        "{}: full_hash should not be all zeros",
        context
    );
}

#[test]
fn test_cross_language_function_consistency() {
    let registry = ParserRegistry::new();

    // Rust function
    let rust_src = "fn compute(x: i32) -> i32 { x * 2 }";
    let rust_units = registry.parse(rust_src, Language::Rust);
    assert_eq!(rust_units.len(), 1);
    verify_code_unit_populated(&rust_units[0], "Rust function");
    assert_eq!(rust_units[0].unit_type, UnitType::Function);

    // Python function
    let python_src = "def compute(x):\n    return x * 2";
    let python_units = registry.parse(python_src, Language::Python);
    assert_eq!(python_units.len(), 1);
    verify_code_unit_populated(&python_units[0], "Python function");
    assert_eq!(python_units[0].unit_type, UnitType::Function);

    // WGSL function
    let wgsl_src = "fn compute(x: f32) -> f32 { return x * 2.0; }";
    let wgsl_units = registry.parse(wgsl_src, Language::Wgsl);
    assert_eq!(wgsl_units.len(), 1);
    verify_code_unit_populated(&wgsl_units[0], "WGSL function");
    assert_eq!(wgsl_units[0].unit_type, UnitType::Function);
}

#[test]
fn test_cross_language_struct_consistency() {
    let registry = ParserRegistry::new();

    // Rust struct
    let rust_src = "struct Point { x: f32, y: f32 }";
    let rust_units = registry.parse(rust_src, Language::Rust);
    assert_eq!(rust_units.len(), 1);
    verify_code_unit_populated(&rust_units[0], "Rust struct");
    assert_eq!(rust_units[0].unit_type, UnitType::Struct);
    assert_eq!(rust_units[0].name, "Point");

    // WGSL struct (Python doesn't have native structs)
    let wgsl_src = "struct Point { x: f32, y: f32 }";
    let wgsl_units = registry.parse(wgsl_src, Language::Wgsl);
    assert_eq!(wgsl_units.len(), 1);
    verify_code_unit_populated(&wgsl_units[0], "WGSL struct");
    assert_eq!(wgsl_units[0].unit_type, UnitType::Struct);
    assert_eq!(wgsl_units[0].name, "Point");
}

#[test]
fn test_cross_language_class_type_mapping() {
    let registry = ParserRegistry::new();

    // Python class -> UnitType::Class
    let python_src = "class MyClass:\n    def __init__(self):\n        pass";
    let python_units = registry.parse(python_src, Language::Python);
    assert_eq!(python_units.len(), 1);
    verify_code_unit_populated(&python_units[0], "Python class");
    assert_eq!(python_units[0].unit_type, UnitType::Class);
    assert_eq!(python_units[0].name, "MyClass");
}

#[test]
fn test_rust_specific_unit_types() {
    let registry = ParserRegistry::new();

    let rust_src = r#"
mod inner {}
enum Status { Active, Inactive }
trait Drawable { fn draw(&self); }
impl Drawable for Point { fn draw(&self) {} }
struct Point;
"#;
    let units = registry.parse(rust_src, Language::Rust);

    // Verify all Rust-specific types are present
    let unit_types: Vec<UnitType> = units.iter().map(|u| u.unit_type).collect();
    assert!(unit_types.contains(&UnitType::Module), "should have Module");
    assert!(unit_types.contains(&UnitType::Enum), "should have Enum");
    assert!(unit_types.contains(&UnitType::Trait), "should have Trait");
    assert!(unit_types.contains(&UnitType::Impl), "should have Impl");
    assert!(unit_types.contains(&UnitType::Struct), "should have Struct");

    // Verify all have correct language
    for unit in &units {
        assert_eq!(unit.language, Language::Rust);
        verify_code_unit_populated(unit, &format!("Rust {:?}", unit.unit_type));
    }
}

#[test]
fn test_python_async_function_type() {
    let registry = ParserRegistry::new();

    let python_src = "async def fetch_data():\n    pass";
    let units = registry.parse(python_src, Language::Python);

    assert_eq!(units.len(), 1);
    // Async functions should still be UnitType::Function
    assert_eq!(units[0].unit_type, UnitType::Function);
    assert_eq!(units[0].name, "fetch_data");
    assert_eq!(units[0].language, Language::Python);
}

#[test]
fn test_wgsl_entry_point_is_function_type() {
    let registry = ParserRegistry::new();

    let wgsl_src = r#"
@vertex
fn vs_main() -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}

@compute @workgroup_size(64)
fn cs_main() {}
"#;
    let units = registry.parse(wgsl_src, Language::Wgsl);

    // All entry points should be UnitType::Function
    assert!(units.len() >= 3, "should have at least 3 entry points");
    for unit in &units {
        assert_eq!(unit.unit_type, UnitType::Function);
        assert_eq!(unit.language, Language::Wgsl);
        verify_code_unit_populated(unit, &format!("WGSL entry point '{}'", unit.name));
    }
}

// ----------------------------------------------------------------------------
// Hash Consistency Tests
// ----------------------------------------------------------------------------

#[test]
fn test_same_code_produces_same_hash() {
    let registry = ParserRegistry::new();

    let source = "fn identical() { let x = 42; }";
    let units1 = registry.parse(source, Language::Rust);
    let units2 = registry.parse(source, Language::Rust);

    assert_eq!(units1.len(), 1);
    assert_eq!(units2.len(), 1);
    assert_eq!(
        units1[0].hashes.full_hash, units2[0].hashes.full_hash,
        "same code should produce same full_hash"
    );
    assert_eq!(
        units1[0].hashes.signature_hash, units2[0].hashes.signature_hash,
        "same code should produce same signature_hash"
    );
}

#[test]
fn test_different_code_produces_different_hash() {
    let registry = ParserRegistry::new();

    let source1 = "fn foo() { let x = 1; }";
    let source2 = "fn foo() { let x = 2; }";

    let units1 = registry.parse(source1, Language::Rust);
    let units2 = registry.parse(source2, Language::Rust);

    assert_eq!(units1.len(), 1);
    assert_eq!(units2.len(), 1);
    assert_ne!(
        units1[0].hashes.full_hash, units2[0].hashes.full_hash,
        "different code should produce different full_hash"
    );
    assert_ne!(
        units1[0].hashes.body_hash, units2[0].hashes.body_hash,
        "different body should produce different body_hash"
    );
    // Signature should be the same (same function name/params)
    assert_eq!(
        units1[0].hashes.signature_hash, units2[0].hashes.signature_hash,
        "same signature should produce same signature_hash"
    );
}

#[test]
fn test_struct_layout_hash_changes_with_fields() {
    let registry = ParserRegistry::new();

    let source1 = "struct Foo { a: i32 }";
    let source2 = "struct Foo { a: i32, b: i32 }";

    let units1 = registry.parse(source1, Language::Rust);
    let units2 = registry.parse(source2, Language::Rust);

    assert_eq!(units1.len(), 1);
    assert_eq!(units2.len(), 1);
    assert_ne!(
        units1[0].hashes.layout_hash, units2[0].hashes.layout_hash,
        "different struct fields should produce different layout_hash"
    );
}

// ----------------------------------------------------------------------------
// Empty/Invalid Input Tests
// ----------------------------------------------------------------------------

#[test]
fn test_empty_source_returns_empty_vec() {
    let registry = ParserRegistry::new();

    assert!(registry.parse("", Language::Rust).is_empty());
    assert!(registry.parse("", Language::Python).is_empty());
    assert!(registry.parse("", Language::Wgsl).is_empty());
}

#[test]
fn test_whitespace_only_source() {
    let registry = ParserRegistry::new();

    let whitespace = "   \n\t\n   ";
    assert!(registry.parse(whitespace, Language::Rust).is_empty());
    assert!(registry.parse(whitespace, Language::Python).is_empty());
    assert!(registry.parse(whitespace, Language::Wgsl).is_empty());
}

#[test]
fn test_comments_only_source() {
    let registry = ParserRegistry::new();

    let rust_comments = "// This is a comment\n/* block */";
    let python_comments = "# This is a comment";
    let wgsl_comments = "// WGSL comment\n/* block */";

    assert!(registry.parse(rust_comments, Language::Rust).is_empty());
    assert!(registry.parse(python_comments, Language::Python).is_empty());
    assert!(registry.parse(wgsl_comments, Language::Wgsl).is_empty());
}

#[test]
fn test_invalid_syntax_returns_empty_gracefully() {
    let registry = ParserRegistry::new();

    // Invalid Rust
    let invalid_rust = "fn missing_brace(";
    let rust_units = registry.parse(invalid_rust, Language::Rust);
    // Should not panic, should return empty or partial
    assert!(rust_units.is_empty() || rust_units.iter().all(|u| u.language == Language::Rust));

    // Invalid Python
    let invalid_python = "def broken(:\n    pass";
    let python_units = registry.parse(invalid_python, Language::Python);
    assert!(
        python_units.is_empty()
            || python_units.iter().all(|u| u.language == Language::Python)
    );

    // Invalid WGSL
    let invalid_wgsl = "struct Broken { x: }";
    let wgsl_units = registry.parse(invalid_wgsl, Language::Wgsl);
    assert!(wgsl_units.is_empty() || wgsl_units.iter().all(|u| u.language == Language::Wgsl));
}

// ----------------------------------------------------------------------------
// parse_file() vs parse() Equivalence
// ----------------------------------------------------------------------------

#[test]
fn test_parse_file_equivalent_to_parse() {
    let registry = ParserRegistry::new();
    let source = "fn test_func() {}";
    let path = Path::new("dummy.rs");

    let from_parse = registry.parse(source, Language::Rust);
    let from_parse_file = registry.parse_file(path, source, Language::Rust);

    assert_eq!(from_parse.len(), from_parse_file.len());
    for (a, b) in from_parse.iter().zip(from_parse_file.iter()) {
        assert_eq!(a.name, b.name);
        assert_eq!(a.unit_type, b.unit_type);
        assert_eq!(a.language, b.language);
        assert_eq!(a.hashes.full_hash, b.hashes.full_hash);
    }
}

// ----------------------------------------------------------------------------
// Language Enum Tests
// ----------------------------------------------------------------------------

#[test]
fn test_language_enum_debug_format() {
    // Ensure Debug is implemented correctly
    let rust = Language::Rust;
    let python = Language::Python;
    let wgsl = Language::Wgsl;

    assert_eq!(format!("{:?}", rust), "Rust");
    assert_eq!(format!("{:?}", python), "Python");
    assert_eq!(format!("{:?}", wgsl), "Wgsl");
}

#[test]
fn test_language_enum_clone_and_copy() {
    let lang = Language::Rust;
    let cloned = lang.clone();
    let copied = lang;

    assert_eq!(lang, cloned);
    assert_eq!(lang, copied);
}

#[test]
fn test_language_enum_eq() {
    assert_eq!(Language::Rust, Language::Rust);
    assert_eq!(Language::Python, Language::Python);
    assert_eq!(Language::Wgsl, Language::Wgsl);

    assert_ne!(Language::Rust, Language::Python);
    assert_ne!(Language::Python, Language::Wgsl);
    assert_ne!(Language::Wgsl, Language::Rust);
}

// ----------------------------------------------------------------------------
// UnitType Enum Tests
// ----------------------------------------------------------------------------

#[test]
fn test_unit_type_enum_variants_exist() {
    // Verify all expected variants exist
    let _function = UnitType::Function;
    let _struct_ = UnitType::Struct;
    let _enum_ = UnitType::Enum;
    let _class = UnitType::Class;
    let _method = UnitType::Method;
    let _module = UnitType::Module;
    let _impl_ = UnitType::Impl;
    let _trait_ = UnitType::Trait;
}

#[test]
fn test_unit_type_enum_debug_format() {
    assert_eq!(format!("{:?}", UnitType::Function), "Function");
    assert_eq!(format!("{:?}", UnitType::Struct), "Struct");
    assert_eq!(format!("{:?}", UnitType::Enum), "Enum");
    assert_eq!(format!("{:?}", UnitType::Class), "Class");
    assert_eq!(format!("{:?}", UnitType::Method), "Method");
    assert_eq!(format!("{:?}", UnitType::Module), "Module");
    assert_eq!(format!("{:?}", UnitType::Impl), "Impl");
    assert_eq!(format!("{:?}", UnitType::Trait), "Trait");
}

#[test]
fn test_unit_type_clone_and_copy() {
    let ut = UnitType::Function;
    let cloned = ut.clone();
    let copied = ut;

    assert_eq!(ut, cloned);
    assert_eq!(ut, copied);
}
