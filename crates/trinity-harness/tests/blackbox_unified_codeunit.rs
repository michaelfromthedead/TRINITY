//! Blackbox tests for Unified CodeUnit and ParserRegistry.
//!
//! CLEANROOM: Tests are written based on the public contract only.
//! Implementation files (parsers/*.rs) were NOT read.
//!
//! Test coverage plan:
//!   - T1: ParserRegistry routes correctly to Rust parser
//!   - T2: ParserRegistry routes correctly to Python parser
//!   - T3: ParserRegistry routes correctly to WGSL parser
//!   - T4: Language auto-detection from file extension
//!   - T5: CodeUnit structure consistency across languages
//!   - T6: Hash computation present for all languages
//!   - T7: Line number tracking for all languages
//!   - T8: UnitType consistency across languages
//!   - T9: Mixed language parsing in sequence
//!   - T10: Edge cases (unknown extension, empty files)
//!   - T11: Cross-language structural equivalence
//!   - T12: Registry reuse (multiple parses)
//!   - T13: Default trait implementation
//!   - T14: Real-world multi-language project patterns
//!   - T15: CodeUnit field completeness validation

use std::path::Path;
use trinity_harness::{ContentHashes, Language, ParserRegistry, UnitType};

// ============================================================================
// T1: ParserRegistry Routes Correctly to Rust Parser
// ============================================================================

#[test]
fn blackbox_unified_registry_routes_to_rust() {
    let registry = ParserRegistry::new();
    let source = "fn rust_function() {}";
    let units = registry.parse_file(Path::new("test.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "rust_function");
    assert_eq!(units[0].language, Language::Rust);
    assert_eq!(units[0].unit_type, UnitType::Function);
}

#[test]
fn blackbox_unified_registry_routes_rust_struct() {
    let registry = ParserRegistry::new();
    let source = "struct RustStruct { field: i32 }";
    let units = registry.parse_file(Path::new("lib.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "RustStruct");
    assert_eq!(units[0].language, Language::Rust);
    assert_eq!(units[0].unit_type, UnitType::Struct);
}

#[test]
fn blackbox_unified_registry_routes_rust_enum() {
    let registry = ParserRegistry::new();
    let source = "enum RustEnum { A, B, C }";
    let units = registry.parse_file(Path::new("types.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "RustEnum");
    assert_eq!(units[0].language, Language::Rust);
    assert_eq!(units[0].unit_type, UnitType::Enum);
}

// ============================================================================
// T2: ParserRegistry Routes Correctly to Python Parser
// ============================================================================

#[test]
fn blackbox_unified_registry_routes_to_python() {
    let registry = ParserRegistry::new();
    let source = "def python_function():\n    pass";
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "python_function");
    assert_eq!(units[0].language, Language::Python);
    assert_eq!(units[0].unit_type, UnitType::Function);
}

#[test]
fn blackbox_unified_registry_routes_python_class() {
    let registry = ParserRegistry::new();
    let source = "class PythonClass:\n    pass";
    let units = registry.parse_file(Path::new("module.py"), source, Language::Python);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "PythonClass");
    assert_eq!(units[0].language, Language::Python);
    assert_eq!(units[0].unit_type, UnitType::Class);
}

#[test]
fn blackbox_unified_registry_routes_python_method() {
    let registry = ParserRegistry::new();
    let source = r#"
class Container:
    def method(self):
        pass
"#;
    let units = registry.parse_file(Path::new("utils.py"), source, Language::Python);

    let classes: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Class).collect();
    let methods: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Method).collect();

    // Class should always be found
    assert_eq!(classes.len(), 1);
    assert_eq!(classes[0].name, "Container");
    assert_eq!(classes[0].language, Language::Python);

    // Methods may or may not be extracted separately depending on implementation
    // If methods are extracted, verify they have the right type
    for method in &methods {
        assert_eq!(method.unit_type, UnitType::Method);
        assert_eq!(method.language, Language::Python);
    }
}

// ============================================================================
// T3: ParserRegistry Routes Correctly to WGSL Parser
// ============================================================================

#[test]
fn blackbox_unified_registry_routes_to_wgsl() {
    let registry = ParserRegistry::new();
    let source = "fn wgsl_function() {}";
    let units = registry.parse_file(Path::new("shader.wgsl"), source, Language::Wgsl);

    let functions: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Function).collect();
    assert!(!functions.is_empty());
    assert!(functions.iter().any(|f| f.name == "wgsl_function"));
    assert!(functions.iter().all(|f| f.language == Language::Wgsl));
}

#[test]
fn blackbox_unified_registry_routes_wgsl_struct() {
    let registry = ParserRegistry::new();
    let source = "struct WgslStruct { value: f32, }";
    let units = registry.parse_file(Path::new("types.wgsl"), source, Language::Wgsl);

    let structs: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Struct).collect();
    assert_eq!(structs.len(), 1);
    assert_eq!(structs[0].name, "WgslStruct");
    assert_eq!(structs[0].language, Language::Wgsl);
}

#[test]
fn blackbox_unified_registry_routes_wgsl_entry_point() {
    let registry = ParserRegistry::new();
    let source = r#"
@vertex
fn vertex_main() -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
"#;
    let units = registry.parse_file(Path::new("render.wgsl"), source, Language::Wgsl);

    let functions: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Function).collect();
    assert!(!functions.is_empty());
    assert!(functions.iter().any(|f| f.name == "vertex_main"));
}

// ============================================================================
// T4: Language Auto-Detection from File Extension
// ============================================================================

#[test]
fn blackbox_unified_autodetect_rust_extension() {
    let registry = ParserRegistry::new();
    let source = "fn auto_detected() {}";
    let units = registry.parse_file_auto(Path::new("auto.rs"), source);

    assert!(units.is_some());
    let units = units.unwrap();
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].language, Language::Rust);
}

#[test]
fn blackbox_unified_autodetect_python_extension() {
    let registry = ParserRegistry::new();
    let source = "def auto_detected():\n    pass";
    let units = registry.parse_file_auto(Path::new("auto.py"), source);

    assert!(units.is_some());
    let units = units.unwrap();
    assert_eq!(units.len(), 1);
    assert_eq!(units[0].language, Language::Python);
}

#[test]
fn blackbox_unified_autodetect_wgsl_extension() {
    let registry = ParserRegistry::new();
    let source = "fn auto_detected() {}";
    let units = registry.parse_file_auto(Path::new("auto.wgsl"), source);

    assert!(units.is_some());
    let units = units.unwrap();
    let functions: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Function).collect();
    assert!(!functions.is_empty());
    assert!(functions.iter().all(|f| f.language == Language::Wgsl));
}

#[test]
fn blackbox_unified_autodetect_unknown_extension() {
    let registry = ParserRegistry::new();
    let source = "some content";
    let units = registry.parse_file_auto(Path::new("file.txt"), source);

    assert!(units.is_none());
}

#[test]
fn blackbox_unified_autodetect_no_extension() {
    let registry = ParserRegistry::new();
    let source = "fn no_ext() {}";
    let units = registry.parse_file_auto(Path::new("Makefile"), source);

    assert!(units.is_none());
}

#[test]
fn blackbox_unified_detect_language_static() {
    // Test the static detect_language function
    assert_eq!(ParserRegistry::detect_language(Path::new("test.rs")), Some(Language::Rust));
    assert_eq!(ParserRegistry::detect_language(Path::new("test.py")), Some(Language::Python));
    assert_eq!(ParserRegistry::detect_language(Path::new("test.wgsl")), Some(Language::Wgsl));
    assert_eq!(ParserRegistry::detect_language(Path::new("test.cpp")), None);
    assert_eq!(ParserRegistry::detect_language(Path::new("test")), None);
}

// ============================================================================
// T5: CodeUnit Structure Consistency Across Languages
// ============================================================================

#[test]
fn blackbox_unified_codeunit_has_name() {
    let registry = ParserRegistry::new();

    // Rust
    let rust_units = registry.parse_file(Path::new("t.rs"), "fn foo() {}", Language::Rust);
    assert!(!rust_units[0].name.is_empty());

    // Python
    let py_units = registry.parse_file(Path::new("t.py"), "def bar():\n    pass", Language::Python);
    assert!(!py_units[0].name.is_empty());

    // WGSL
    let wgsl_units = registry.parse_file(Path::new("t.wgsl"), "fn baz() {}", Language::Wgsl);
    let funcs: Vec<_> = wgsl_units.iter().filter(|u| u.unit_type == UnitType::Function).collect();
    assert!(!funcs.is_empty());
    assert!(!funcs[0].name.is_empty());
}

#[test]
fn blackbox_unified_codeunit_has_unit_type() {
    let registry = ParserRegistry::new();

    // Functions from all languages should have UnitType::Function
    let rust_fn = registry.parse_file(Path::new("t.rs"), "fn x() {}", Language::Rust);
    assert_eq!(rust_fn[0].unit_type, UnitType::Function);

    let py_fn = registry.parse_file(Path::new("t.py"), "def y():\n    pass", Language::Python);
    assert_eq!(py_fn[0].unit_type, UnitType::Function);

    let wgsl_fn = registry.parse_file(Path::new("t.wgsl"), "fn z() {}", Language::Wgsl);
    let funcs: Vec<_> = wgsl_fn.iter().filter(|u| u.unit_type == UnitType::Function).collect();
    assert!(!funcs.is_empty());
}

#[test]
fn blackbox_unified_codeunit_has_language() {
    let registry = ParserRegistry::new();

    let rust_units = registry.parse_file(Path::new("t.rs"), "fn r() {}", Language::Rust);
    assert_eq!(rust_units[0].language, Language::Rust);

    let py_units = registry.parse_file(Path::new("t.py"), "def p():\n    pass", Language::Python);
    assert_eq!(py_units[0].language, Language::Python);

    let wgsl_units = registry.parse_file(Path::new("t.wgsl"), "fn w() {}", Language::Wgsl);
    assert!(wgsl_units.iter().all(|u| u.language == Language::Wgsl));
}

// ============================================================================
// T6: Hash Computation Present for All Languages
// ============================================================================

#[test]
fn blackbox_unified_rust_has_hashes() {
    let registry = ParserRegistry::new();
    let source = "fn hashed_fn() { let x = 1; }";
    let units = registry.parse_file(Path::new("t.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1);
    // Hashes should be computed (not all zeros for non-trivial content)
    let hashes = &units[0].hashes;
    // At least one hash should be non-zero for meaningful content
    let full_is_zero = hashes.full_hash.iter().all(|&b| b == 0);
    // Full hash should be computed
    assert!(!full_is_zero, "full_hash should be computed for Rust function");
}

#[test]
fn blackbox_unified_python_has_hashes() {
    let registry = ParserRegistry::new();
    let source = "def hashed_fn():\n    x = 1\n    return x";
    let units = registry.parse_file(Path::new("t.py"), source, Language::Python);

    assert_eq!(units.len(), 1);
    let hashes = &units[0].hashes;
    let full_is_zero = hashes.full_hash.iter().all(|&b| b == 0);
    assert!(!full_is_zero, "full_hash should be computed for Python function");
}

#[test]
fn blackbox_unified_wgsl_has_hashes() {
    let registry = ParserRegistry::new();
    let source = "fn hashed_fn() { var x: f32 = 1.0; }";
    let units = registry.parse_file(Path::new("t.wgsl"), source, Language::Wgsl);

    let funcs: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Function).collect();
    assert!(!funcs.is_empty());

    let hashes = &funcs[0].hashes;
    let full_is_zero = hashes.full_hash.iter().all(|&b| b == 0);
    assert!(!full_is_zero, "full_hash should be computed for WGSL function");
}

#[test]
fn blackbox_unified_different_content_different_hash() {
    let registry = ParserRegistry::new();

    let source1 = "fn version_one() { let x = 1; }";
    let source2 = "fn version_one() { let x = 2; }";

    let units1 = registry.parse_file(Path::new("t.rs"), source1, Language::Rust);
    let units2 = registry.parse_file(Path::new("t.rs"), source2, Language::Rust);

    // Different body should produce different full_hash
    assert_ne!(
        units1[0].hashes.full_hash,
        units2[0].hashes.full_hash,
        "Different function bodies should have different full_hash"
    );
}

#[test]
fn blackbox_unified_same_content_same_hash() {
    let registry = ParserRegistry::new();
    let source = "fn deterministic() { let v = 42; }";

    let units1 = registry.parse_file(Path::new("a.rs"), source, Language::Rust);
    let units2 = registry.parse_file(Path::new("b.rs"), source, Language::Rust);

    // Same content should produce same hash regardless of filename
    assert_eq!(
        units1[0].hashes.full_hash,
        units2[0].hashes.full_hash,
        "Same content should produce same full_hash"
    );
}

// ============================================================================
// T7: Line Number Tracking for All Languages
// ============================================================================

#[test]
fn blackbox_unified_rust_line_numbers() {
    let registry = ParserRegistry::new();
    let source = r#"
fn first() {}

fn second() {}

fn third() {}
"#;
    let units = registry.parse_file(Path::new("t.rs"), source, Language::Rust);

    assert_eq!(units.len(), 3);
    // Line numbers should be in ascending order
    assert!(units[0].start_line < units[1].start_line);
    assert!(units[1].start_line < units[2].start_line);
    // Each unit should have valid line range
    for unit in &units {
        assert!(unit.start_line <= unit.end_line);
    }
}

#[test]
fn blackbox_unified_python_line_numbers() {
    let registry = ParserRegistry::new();
    let source = r#"
def first():
    pass

def second():
    pass

def third():
    pass
"#;
    let units = registry.parse_file(Path::new("t.py"), source, Language::Python);

    assert_eq!(units.len(), 3);
    assert!(units[0].start_line < units[1].start_line);
    assert!(units[1].start_line < units[2].start_line);
    for unit in &units {
        assert!(unit.start_line <= unit.end_line);
    }
}

#[test]
fn blackbox_unified_wgsl_line_numbers() {
    let registry = ParserRegistry::new();
    let source = r#"
fn first() {}

fn second() {}

fn third() {}
"#;
    let units = registry.parse_file(Path::new("t.wgsl"), source, Language::Wgsl);

    let funcs: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Function).collect();
    assert_eq!(funcs.len(), 3);

    for i in 0..funcs.len() - 1 {
        assert!(funcs[i].start_line < funcs[i + 1].start_line);
    }
    for func in &funcs {
        assert!(func.start_line <= func.end_line);
    }
}

#[test]
fn blackbox_unified_single_line_unit() {
    let registry = ParserRegistry::new();
    let source = "fn single_line() {}";
    let units = registry.parse_file(Path::new("t.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1);
    // Single line function should have start_line == end_line
    assert_eq!(units[0].start_line, units[0].end_line);
}

// ============================================================================
// T8: UnitType Consistency Across Languages
// ============================================================================

#[test]
fn blackbox_unified_function_type_consistent() {
    let registry = ParserRegistry::new();

    let rust = registry.parse_file(Path::new("t.rs"), "fn f() {}", Language::Rust);
    let python = registry.parse_file(Path::new("t.py"), "def f():\n    pass", Language::Python);
    let wgsl = registry.parse_file(Path::new("t.wgsl"), "fn f() {}", Language::Wgsl);

    // All should use the same UnitType::Function
    assert_eq!(rust[0].unit_type, UnitType::Function);
    assert_eq!(python[0].unit_type, UnitType::Function);

    let wgsl_funcs: Vec<_> = wgsl.iter().filter(|u| u.unit_type == UnitType::Function).collect();
    assert!(!wgsl_funcs.is_empty());
}

#[test]
fn blackbox_unified_struct_type_consistent() {
    let registry = ParserRegistry::new();

    let rust = registry.parse_file(Path::new("t.rs"), "struct S { x: i32 }", Language::Rust);
    let wgsl = registry.parse_file(Path::new("t.wgsl"), "struct S { x: f32, }", Language::Wgsl);

    // Both should use UnitType::Struct
    assert_eq!(rust[0].unit_type, UnitType::Struct);

    let wgsl_structs: Vec<_> = wgsl.iter().filter(|u| u.unit_type == UnitType::Struct).collect();
    assert_eq!(wgsl_structs.len(), 1);
    assert_eq!(wgsl_structs[0].unit_type, UnitType::Struct);
}

#[test]
fn blackbox_unified_class_for_python_only() {
    let registry = ParserRegistry::new();

    let python = registry.parse_file(Path::new("t.py"), "class C:\n    pass", Language::Python);

    assert_eq!(python.len(), 1);
    assert_eq!(python[0].unit_type, UnitType::Class);
    // Class is Python-specific, Rust and WGSL use Struct
}

// ============================================================================
// T9: Mixed Language Parsing in Sequence
// ============================================================================

#[test]
fn blackbox_unified_sequential_multi_language() {
    let registry = ParserRegistry::new();

    // Parse different languages in sequence with same registry
    let rust = registry.parse_file(Path::new("a.rs"), "fn rust_fn() {}", Language::Rust);
    let python = registry.parse_file(Path::new("b.py"), "def py_fn():\n    pass", Language::Python);
    let wgsl = registry.parse_file(Path::new("c.wgsl"), "fn wgsl_fn() {}", Language::Wgsl);
    let rust2 = registry.parse_file(Path::new("d.rs"), "fn another_rust() {}", Language::Rust);

    // All should parse correctly
    assert_eq!(rust[0].name, "rust_fn");
    assert_eq!(rust[0].language, Language::Rust);

    assert_eq!(python[0].name, "py_fn");
    assert_eq!(python[0].language, Language::Python);

    let wgsl_funcs: Vec<_> = wgsl.iter().filter(|u| u.unit_type == UnitType::Function).collect();
    assert!(wgsl_funcs.iter().any(|f| f.name == "wgsl_fn"));

    assert_eq!(rust2[0].name, "another_rust");
    assert_eq!(rust2[0].language, Language::Rust);
}

#[test]
fn blackbox_unified_interleaved_parsing() {
    let registry = ParserRegistry::new();

    // Interleave parsing of different languages
    for i in 0..5 {
        let rust = registry.parse_file(
            Path::new("t.rs"),
            &format!("fn rust_{i}() {{}}"),
            Language::Rust,
        );
        let python = registry.parse_file(
            Path::new("t.py"),
            &format!("def python_{i}():\n    pass"),
            Language::Python,
        );

        assert_eq!(rust[0].name, format!("rust_{i}"));
        assert_eq!(python[0].name, format!("python_{i}"));
    }
}

// ============================================================================
// T10: Edge Cases (Unknown Extension, Empty Files)
// ============================================================================

#[test]
fn blackbox_unified_empty_rust_file() {
    let registry = ParserRegistry::new();
    let units = registry.parse_file(Path::new("empty.rs"), "", Language::Rust);
    assert!(units.is_empty());
}

#[test]
fn blackbox_unified_empty_python_file() {
    let registry = ParserRegistry::new();
    let units = registry.parse_file(Path::new("empty.py"), "", Language::Python);
    assert!(units.is_empty());
}

#[test]
fn blackbox_unified_empty_wgsl_file() {
    let registry = ParserRegistry::new();
    let units = registry.parse_file(Path::new("empty.wgsl"), "", Language::Wgsl);
    assert!(units.is_empty());
}

#[test]
fn blackbox_unified_whitespace_only() {
    let registry = ParserRegistry::new();

    let rust = registry.parse_file(Path::new("t.rs"), "   \n\n\t\t  \n", Language::Rust);
    let python = registry.parse_file(Path::new("t.py"), "   \n\n\t\t  \n", Language::Python);
    let wgsl = registry.parse_file(Path::new("t.wgsl"), "   \n\n\t\t  \n", Language::Wgsl);

    assert!(rust.is_empty());
    assert!(python.is_empty());
    assert!(wgsl.is_empty());
}

#[test]
fn blackbox_unified_comments_only_rust() {
    let registry = ParserRegistry::new();
    let source = r#"
// A comment
/* Block comment */
/// Doc comment
"#;
    let units = registry.parse_file(Path::new("t.rs"), source, Language::Rust);
    assert!(units.is_empty());
}

#[test]
fn blackbox_unified_comments_only_python() {
    let registry = ParserRegistry::new();
    let source = r#"
# A comment
# Another comment
"""
Docstring without a function
"""
"#;
    let units = registry.parse_file(Path::new("t.py"), source, Language::Python);
    assert!(units.is_empty());
}

#[test]
fn blackbox_unified_comments_only_wgsl() {
    let registry = ParserRegistry::new();
    let source = r#"
// A comment
/* Block comment */
"#;
    let units = registry.parse_file(Path::new("t.wgsl"), source, Language::Wgsl);
    assert!(units.is_empty());
}

// ============================================================================
// T11: Cross-Language Structural Equivalence
// ============================================================================

#[test]
fn blackbox_unified_equivalent_functions_same_fields() {
    let registry = ParserRegistry::new();

    // Parse equivalent functions in different languages
    let rust = registry.parse_file(Path::new("t.rs"), "fn compute() {}", Language::Rust);
    let python = registry.parse_file(
        Path::new("t.py"),
        "def compute():\n    pass",
        Language::Python,
    );
    let wgsl = registry.parse_file(Path::new("t.wgsl"), "fn compute() {}", Language::Wgsl);

    // All should have the same name
    assert_eq!(rust[0].name, "compute");
    assert_eq!(python[0].name, "compute");

    let wgsl_funcs: Vec<_> = wgsl.iter().filter(|u| u.name == "compute").collect();
    assert!(!wgsl_funcs.is_empty());

    // All should be functions
    assert_eq!(rust[0].unit_type, UnitType::Function);
    assert_eq!(python[0].unit_type, UnitType::Function);
    assert_eq!(wgsl_funcs[0].unit_type, UnitType::Function);

    // All should have valid line numbers
    assert!(rust[0].start_line <= rust[0].end_line);
    assert!(python[0].start_line <= python[0].end_line);
    assert!(wgsl_funcs[0].start_line <= wgsl_funcs[0].end_line);

    // Languages should be different
    assert_eq!(rust[0].language, Language::Rust);
    assert_eq!(python[0].language, Language::Python);
    assert_eq!(wgsl_funcs[0].language, Language::Wgsl);
}

#[test]
fn blackbox_unified_struct_equivalence() {
    let registry = ParserRegistry::new();

    // Rust struct
    let rust = registry.parse_file(
        Path::new("t.rs"),
        "struct Vector { x: f32, y: f32 }",
        Language::Rust,
    );

    // WGSL struct (similar concept)
    let wgsl = registry.parse_file(
        Path::new("t.wgsl"),
        "struct Vector { x: f32, y: f32, }",
        Language::Wgsl,
    );

    // Both should be structs with same name
    assert_eq!(rust[0].unit_type, UnitType::Struct);
    assert_eq!(rust[0].name, "Vector");

    let wgsl_structs: Vec<_> = wgsl.iter().filter(|u| u.unit_type == UnitType::Struct).collect();
    assert_eq!(wgsl_structs[0].name, "Vector");
}

// ============================================================================
// T12: Registry Reuse (Multiple Parses)
// ============================================================================

#[test]
fn blackbox_unified_registry_reuse_many_parses() {
    let registry = ParserRegistry::new();

    // Parse many files with the same registry
    for i in 0..100 {
        let source = format!("fn func_{i}() {{}}");
        let units = registry.parse_file(Path::new("t.rs"), &source, Language::Rust);
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].name, format!("func_{i}"));
    }
}

#[test]
fn blackbox_unified_registry_large_file() {
    let registry = ParserRegistry::new();

    // Generate a large file with many functions
    let mut source = String::new();
    for i in 0..50 {
        source.push_str(&format!("fn func_{i}() {{}}\n\n"));
    }

    let units = registry.parse_file(Path::new("large.rs"), &source, Language::Rust);
    assert_eq!(units.len(), 50);
}

#[test]
fn blackbox_unified_registry_concurrent_safe() {
    // Test that multiple parses don't interfere with each other's results
    let registry = ParserRegistry::new();

    let source_a = "fn alpha() {}";
    let source_b = "fn beta() {}";

    let units_a = registry.parse_file(Path::new("a.rs"), source_a, Language::Rust);
    let units_b = registry.parse_file(Path::new("b.rs"), source_b, Language::Rust);

    // Results should be independent
    assert_eq!(units_a.len(), 1);
    assert_eq!(units_b.len(), 1);
    assert_eq!(units_a[0].name, "alpha");
    assert_eq!(units_b[0].name, "beta");
}

// ============================================================================
// T13: Default Trait Implementation
// ============================================================================

#[test]
fn blackbox_unified_registry_default() {
    // ParserRegistry should implement Default
    let registry = ParserRegistry::default();
    let units = registry.parse_file(Path::new("t.rs"), "fn test() {}", Language::Rust);
    assert_eq!(units.len(), 1);
}

#[test]
fn blackbox_unified_content_hashes_default() {
    // ContentHashes should implement Default
    let hashes = ContentHashes::default();
    // Default should have all zeros
    assert!(hashes.full_hash.iter().all(|&b| b == 0));
    assert!(hashes.signature_hash.iter().all(|&b| b == 0));
    assert!(hashes.body_hash.iter().all(|&b| b == 0));
    assert!(hashes.layout_hash.iter().all(|&b| b == 0));
}

// ============================================================================
// T14: Real-World Multi-Language Project Patterns
// ============================================================================

#[test]
fn blackbox_unified_rust_module_system() {
    let registry = ParserRegistry::new();
    let source = r#"
mod internal {
    pub fn helper() -> i32 { 42 }
}

pub struct Config {
    pub name: String,
    pub value: i32,
}

impl Config {
    pub fn new(name: &str, value: i32) -> Self {
        Config {
            name: name.to_string(),
            value,
        }
    }
}

pub fn process(config: &Config) -> i32 {
    internal::helper() + config.value
}
"#;
    let units = registry.parse_file(Path::new("lib.rs"), source, Language::Rust);

    // Should extract module, struct, impl, and functions
    let modules: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Module).collect();
    let structs: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Struct).collect();
    let impls: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Impl).collect();
    let funcs: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Function).collect();

    assert_eq!(modules.len(), 1);
    assert_eq!(structs.len(), 1);
    assert_eq!(impls.len(), 1);
    assert!(funcs.len() >= 1); // At least the top-level function
}

#[test]
fn blackbox_unified_python_class_hierarchy() {
    let registry = ParserRegistry::new();
    let source = r#"
class Base:
    def __init__(self):
        self.value = 0

    def compute(self):
        return self.value

class Derived(Base):
    def __init__(self, extra):
        super().__init__()
        self.extra = extra

    def compute(self):
        return super().compute() + self.extra

def factory(use_derived: bool):
    if use_derived:
        return Derived(10)
    return Base()
"#;
    let units = registry.parse_file(Path::new("classes.py"), source, Language::Python);

    let classes: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Class).collect();
    let funcs: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Function).collect();
    let methods: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Method).collect();

    // Classes should always be extracted
    assert_eq!(classes.len(), 2);
    let class_names: Vec<_> = classes.iter().map(|c| c.name.as_str()).collect();
    assert!(class_names.contains(&"Base"));
    assert!(class_names.contains(&"Derived"));

    // Top-level factory function should be extracted
    assert!(funcs.len() >= 1);
    assert!(funcs.iter().any(|f| f.name == "factory"));

    // Methods may or may not be extracted separately
    // If they are extracted, verify correctness
    for method in &methods {
        assert_eq!(method.unit_type, UnitType::Method);
        assert_eq!(method.language, Language::Python);
    }
}

#[test]
fn blackbox_unified_wgsl_render_pipeline() {
    let registry = ParserRegistry::new();
    let source = r#"
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
}

struct Uniforms {
    model: mat4x4<f32>,
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.projection * uniforms.view * uniforms.model * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
"#;
    let units = registry.parse_file(Path::new("render.wgsl"), source, Language::Wgsl);

    let structs: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Struct).collect();
    let funcs: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Function).collect();

    // Should extract 3 structs and 2 functions (entry points)
    assert_eq!(structs.len(), 3);
    assert_eq!(funcs.len(), 2);

    let struct_names: Vec<_> = structs.iter().map(|s| s.name.as_str()).collect();
    assert!(struct_names.contains(&"VertexInput"));
    assert!(struct_names.contains(&"VertexOutput"));
    assert!(struct_names.contains(&"Uniforms"));

    let func_names: Vec<_> = funcs.iter().map(|f| f.name.as_str()).collect();
    assert!(func_names.contains(&"vs_main"));
    assert!(func_names.contains(&"fs_main"));
}

// ============================================================================
// T15: CodeUnit Field Completeness Validation
// ============================================================================

#[test]
fn blackbox_unified_codeunit_all_fields_populated() {
    let registry = ParserRegistry::new();

    let source = r#"
fn complete_function() {
    let x = 1;
    let y = 2;
    x + y
}
"#;
    let units = registry.parse_file(Path::new("complete.rs"), source, Language::Rust);

    assert_eq!(units.len(), 1);
    let unit = &units[0];

    // All fields should be populated
    assert!(!unit.name.is_empty(), "name should be populated");
    assert_eq!(unit.unit_type, UnitType::Function, "unit_type should be Function");
    assert_eq!(unit.language, Language::Rust, "language should be Rust");
    assert!(unit.start_line > 0, "start_line should be positive");
    assert!(unit.end_line >= unit.start_line, "end_line should be >= start_line");

    // Hashes should be computed
    let hashes = &unit.hashes;
    assert!(
        !hashes.full_hash.iter().all(|&b| b == 0),
        "full_hash should be non-zero"
    );
}

#[test]
fn blackbox_unified_struct_layout_hash() {
    let registry = ParserRegistry::new();

    // Two structs with different layouts should have different layout hashes
    let source1 = "struct Ordered { a: i32, b: i32 }";
    let source2 = "struct Ordered { b: i32, a: i32 }";

    let units1 = registry.parse_file(Path::new("t.rs"), source1, Language::Rust);
    let units2 = registry.parse_file(Path::new("t.rs"), source2, Language::Rust);

    // Different field order should produce different layout hash
    assert_ne!(
        units1[0].hashes.layout_hash,
        units2[0].hashes.layout_hash,
        "Different struct layouts should have different layout_hash"
    );
}

#[test]
fn blackbox_unified_function_signature_hash() {
    let registry = ParserRegistry::new();

    // Same signature, different body
    let source1 = "fn sig(x: i32) -> i32 { x + 1 }";
    let source2 = "fn sig(x: i32) -> i32 { x * 2 }";

    let units1 = registry.parse_file(Path::new("t.rs"), source1, Language::Rust);
    let units2 = registry.parse_file(Path::new("t.rs"), source2, Language::Rust);

    // Same signature should produce same signature_hash
    assert_eq!(
        units1[0].hashes.signature_hash,
        units2[0].hashes.signature_hash,
        "Same function signature should have same signature_hash"
    );

    // Different body should produce different body_hash
    assert_ne!(
        units1[0].hashes.body_hash,
        units2[0].hashes.body_hash,
        "Different function bodies should have different body_hash"
    );
}

// ============================================================================
// Additional Edge Cases
// ============================================================================

#[test]
fn blackbox_unified_unicode_identifiers() {
    let registry = ParserRegistry::new();

    // Python allows unicode identifiers
    let py_source = "def calcular_área():\n    pass";
    let units = registry.parse_file(Path::new("t.py"), py_source, Language::Python);
    assert_eq!(units.len(), 1);
    assert!(units[0].name.contains("área") || units[0].name.contains("calcular"));
}

#[test]
fn blackbox_unified_deeply_nested() {
    let registry = ParserRegistry::new();

    let source = r#"
mod level1 {
    mod level2 {
        mod level3 {
            fn deeply_nested() {}
        }
    }
}
"#;
    let units = registry.parse_file(Path::new("t.rs"), source, Language::Rust);

    // At minimum, the outer module should be extracted
    // Nested modules may or may not be extracted depending on implementation
    let modules: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Module).collect();
    assert!(modules.len() >= 1, "should find at least outer module");
    assert!(modules.iter().any(|m| m.name == "level1"));
}

#[test]
fn blackbox_unified_parse_vs_parse_file() {
    let registry = ParserRegistry::new();
    let source = "fn same_result() {}";

    // parse and parse_file should produce the same result
    let via_parse = registry.parse(source, Language::Rust);
    let via_parse_file = registry.parse_file(Path::new("t.rs"), source, Language::Rust);

    assert_eq!(via_parse.len(), via_parse_file.len());
    assert_eq!(via_parse[0].name, via_parse_file[0].name);
    assert_eq!(via_parse[0].unit_type, via_parse_file[0].unit_type);
    assert_eq!(via_parse[0].language, via_parse_file[0].language);
}

#[test]
fn blackbox_unified_language_enum_debug() {
    // Language should implement Debug for logging
    let rust = Language::Rust;
    let python = Language::Python;
    let wgsl = Language::Wgsl;

    let debug_rust = format!("{:?}", rust);
    let debug_python = format!("{:?}", python);
    let debug_wgsl = format!("{:?}", wgsl);

    assert!(debug_rust.contains("Rust"));
    assert!(debug_python.contains("Python"));
    assert!(debug_wgsl.contains("Wgsl"));
}

#[test]
fn blackbox_unified_unit_type_enum_debug() {
    // UnitType should implement Debug
    let func = UnitType::Function;
    let class = UnitType::Class;

    assert!(format!("{:?}", func).contains("Function"));
    assert!(format!("{:?}", class).contains("Class"));
}

#[test]
fn blackbox_unified_codeunit_clone() {
    let registry = ParserRegistry::new();
    let units = registry.parse_file(Path::new("t.rs"), "fn cloneable() {}", Language::Rust);

    // CodeUnit should be cloneable
    let cloned = units[0].clone();
    assert_eq!(cloned.name, units[0].name);
    assert_eq!(cloned.unit_type, units[0].unit_type);
    assert_eq!(cloned.language, units[0].language);
    assert_eq!(cloned.hashes.full_hash, units[0].hashes.full_hash);
}

#[test]
fn blackbox_unified_content_hashes_clone() {
    let registry = ParserRegistry::new();
    let units = registry.parse_file(Path::new("t.rs"), "fn hashable() {}", Language::Rust);

    // ContentHashes should be cloneable
    let cloned_hashes = units[0].hashes.clone();
    assert_eq!(cloned_hashes.full_hash, units[0].hashes.full_hash);
}
