//! Whitebox tests for PythonParser.
//!
//! WHITEBOX coverage plan:
//!   - Path A: empty input -> empty Vec
//!   - Path B: function definition -> CodeUnit with Function type
//!   - Path C: async function definition -> CodeUnit with Function type
//!   - Path D: class definition -> CodeUnit with Class type
//!   - Path E: invalid syntax -> empty Vec
//!   - Path F: multiple items -> multiple CodeUnits
//!   - Path G: nested items (class with methods) - only top-level extracted
//!   - Path H: Hash computation (full_hash, signature_hash, body_hash, layout_hash)
//!   - Path I: Line number extraction accuracy
//!   - Path J: Hash consistency (same input -> same hash)
//!   - Path K: Class layout hash (method names)
//!   - Path L: Function with parameters (signature hash)
//!   - Path M: Async function hashes include "async" prefix
//!   - Path N: Class with bases (signature hash)
//!   - Path O: Edge case - boundary offsets

use std::path::Path;
use trinity_harness::parsers::{Language, ParserRegistry, PythonParser, UnitType};

#[test]
fn test_python_parse_empty_input() {
    // Path A: empty input returns empty Vec
    let registry = ParserRegistry::new();
    let units = registry.parse_file(Path::new("test.py"), "", Language::Python);
    assert!(units.is_empty(), "empty input should produce no units");
}

#[test]
fn test_python_parse_function() {
    // Path B: function definition
    let registry = ParserRegistry::new();
    let source = r#"
def hello():
    print("Hello")
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    assert_eq!(units.len(), 1, "should find one function");
    assert_eq!(units[0].name, "hello");
    assert_eq!(units[0].unit_type, UnitType::Function);
    assert_eq!(units[0].language, Language::Python);
}

#[test]
fn test_python_parse_async_function() {
    // Path C: async function definition
    let registry = ParserRegistry::new();
    let source = r#"
async def fetch_data():
    await something()
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    assert_eq!(units.len(), 1, "should find one async function");
    assert_eq!(units[0].name, "fetch_data");
    assert_eq!(units[0].unit_type, UnitType::Function);
}

#[test]
fn test_python_parse_class() {
    // Path D: class definition
    let registry = ParserRegistry::new();
    let source = r#"
class MyClass:
    pass
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    assert_eq!(units.len(), 1, "should find one class");
    assert_eq!(units[0].name, "MyClass");
    assert_eq!(units[0].unit_type, UnitType::Class);
}

#[test]
fn test_python_parse_invalid_syntax() {
    // Path E: invalid syntax returns empty Vec
    let registry = ParserRegistry::new();
    let source = r#"
def broken(
    # missing body
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);
    assert!(units.is_empty(), "invalid syntax should produce no units");
}

#[test]
fn test_python_parse_multiple_items() {
    // Path F: multiple items
    let registry = ParserRegistry::new();
    let source = r#"
def func1():
    pass

def func2():
    pass

class ClassA:
    pass

class ClassB:
    pass

async def async_func():
    pass
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    assert_eq!(units.len(), 5, "should find all 5 items");

    let functions = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Function)
        .count();
    let classes = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Class)
        .count();

    assert_eq!(functions, 3, "should find 3 functions (including async)");
    assert_eq!(classes, 2, "should find 2 classes");
}

#[test]
fn test_python_parse_nested_class_methods_not_extracted() {
    // Path G: nested items - only top-level extracted
    let registry = ParserRegistry::new();
    let source = r#"
class MyClass:
    def method(self):
        pass

    async def async_method(self):
        pass
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    // Implementation only extracts top-level statements
    assert_eq!(units.len(), 1, "should only find the class, not methods");
    assert_eq!(units[0].name, "MyClass");
    assert_eq!(units[0].unit_type, UnitType::Class);
}

#[test]
fn test_python_parse_ignores_other_statements() {
    // Import, assignment, etc. should be ignored
    let registry = ParserRegistry::new();
    let source = r#"
import os
from sys import path

x = 10
y: int = 20

if True:
    pass
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);
    assert!(
        units.is_empty(),
        "should ignore imports, assignments, and control flow"
    );
}

#[test]
fn test_python_parse_line_numbers() {
    // Verify line numbers are captured (even if as byte offsets)
    let registry = ParserRegistry::new();
    let source = "def foo():\n    pass\n";
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    assert_eq!(units.len(), 1);
    // start_line and end_line are byte offsets in the current impl
    // Just verify they're set to non-zero values for non-trivial input
    assert!(
        units[0].start_line < units[0].end_line || units[0].end_line > 0,
        "line numbers should be populated"
    );
}

#[test]
fn test_python_parser_default_impl() {
    // PythonParser::default() should work
    let _parser = PythonParser::default();
}

#[test]
fn test_python_parse_decorated_function() {
    // Decorated functions should still be extracted
    let registry = ParserRegistry::new();
    let source = r#"
@decorator
def decorated():
    pass

@decorator1
@decorator2
def multi_decorated():
    pass
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    assert_eq!(units.len(), 2, "should find both decorated functions");
    let names: Vec<_> = units.iter().map(|u| u.name.as_str()).collect();
    assert!(names.contains(&"decorated"));
    assert!(names.contains(&"multi_decorated"));
}

// ============================================================================
// Path H: Hash computation tests
// ============================================================================

#[test]
fn test_python_full_hash_computed() {
    // Path H: full_hash is computed (non-zero)
    let registry = ParserRegistry::new();
    let source = r#"
def greet(name):
    return f"Hello, {name}!"
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    assert_eq!(units.len(), 1);
    let hashes = &units[0].hashes;

    // full_hash should be non-zero (blake3 hash of full text)
    assert!(
        hashes.full_hash.iter().any(|&b| b != 0),
        "full_hash should be non-zero"
    );
}

#[test]
fn test_python_signature_hash_computed() {
    // Path H: signature_hash is computed for function
    let registry = ParserRegistry::new();
    let source = r#"
def add(a, b):
    return a + b
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    assert_eq!(units.len(), 1);
    let hashes = &units[0].hashes;

    // signature_hash should be non-zero
    assert!(
        hashes.signature_hash.iter().any(|&b| b != 0),
        "signature_hash should be non-zero"
    );
}

#[test]
fn test_python_body_hash_computed() {
    // Path H: body_hash is computed for function
    let registry = ParserRegistry::new();
    let source = r#"
def multiply(x, y):
    result = x * y
    return result
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    assert_eq!(units.len(), 1);
    let hashes = &units[0].hashes;

    // body_hash should be non-zero
    assert!(
        hashes.body_hash.iter().any(|&b| b != 0),
        "body_hash should be non-zero"
    );
}

#[test]
fn test_python_function_layout_hash_is_zero() {
    // Path H: layout_hash is zero for functions (not applicable)
    let registry = ParserRegistry::new();
    let source = r#"
def foo():
    pass
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    assert_eq!(units.len(), 1);
    let hashes = &units[0].hashes;

    // layout_hash should be all zeros for functions
    assert!(
        hashes.layout_hash.iter().all(|&b| b == 0),
        "layout_hash should be zero for functions"
    );
}

// ============================================================================
// Path I: Line number extraction accuracy
// ============================================================================

#[test]
fn test_python_line_numbers_accurate() {
    // Path I: verify line numbers are 1-indexed and accurate
    let registry = ParserRegistry::new();
    // Line 1: def foo():
    // Line 2:     pass
    let source = "def foo():\n    pass\n";
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    assert_eq!(units.len(), 1);
    // start_line should be 1 (first line)
    assert_eq!(units[0].start_line, 1, "start_line should be 1");
    // end_line should be 2 (pass is on line 2)
    assert_eq!(units[0].end_line, 2, "end_line should be 2");
}

#[test]
fn test_python_multiline_function_line_numbers() {
    // Path I: multiline function line numbers
    let registry = ParserRegistry::new();
    let source = r#"def complex():
    x = 1
    y = 2
    z = 3
    return x + y + z
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].start_line, 1, "start_line should be 1");
    assert_eq!(units[0].end_line, 5, "end_line should be 5");
}

#[test]
fn test_python_function_after_blank_lines() {
    // Path I: function after blank lines
    let registry = ParserRegistry::new();
    let source = "\n\n\ndef foo():\n    pass\n";
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    assert_eq!(units.len(), 1);
    // Function starts on line 4 (after 3 blank lines)
    assert_eq!(units[0].start_line, 4, "start_line should be 4");
    assert_eq!(units[0].end_line, 5, "end_line should be 5");
}

// ============================================================================
// Path J: Hash consistency tests
// ============================================================================

#[test]
fn test_python_hash_consistency_same_input() {
    // Path J: same input should produce same hashes
    let registry = ParserRegistry::new();
    let source = r#"
def consistent():
    return 42
"#;
    let units1 = registry.parse_file(Path::new("test.py"), source, Language::Python);
    let units2 = registry.parse_file(Path::new("test.py"), source, Language::Python);

    assert_eq!(units1.len(), 1);
    assert_eq!(units2.len(), 1);

    // All hashes should be identical
    assert_eq!(
        units1[0].hashes.full_hash, units2[0].hashes.full_hash,
        "full_hash should be consistent"
    );
    assert_eq!(
        units1[0].hashes.signature_hash, units2[0].hashes.signature_hash,
        "signature_hash should be consistent"
    );
    assert_eq!(
        units1[0].hashes.body_hash, units2[0].hashes.body_hash,
        "body_hash should be consistent"
    );
}

#[test]
fn test_python_hash_differs_with_body_change() {
    // Path J: changing body should change body_hash but not signature_hash
    let registry = ParserRegistry::new();

    let source1 = "def foo(x):\n    return x\n";
    let source2 = "def foo(x):\n    return x + 1\n";

    let units1 = registry.parse_file(Path::new("test.py"), source1, Language::Python);
    let units2 = registry.parse_file(Path::new("test.py"), source2, Language::Python);

    assert_eq!(units1.len(), 1);
    assert_eq!(units2.len(), 1);

    // full_hash should differ
    assert_ne!(
        units1[0].hashes.full_hash, units2[0].hashes.full_hash,
        "full_hash should differ when body changes"
    );

    // body_hash should differ
    assert_ne!(
        units1[0].hashes.body_hash, units2[0].hashes.body_hash,
        "body_hash should differ when body changes"
    );

    // signature_hash should be the same (same signature)
    assert_eq!(
        units1[0].hashes.signature_hash, units2[0].hashes.signature_hash,
        "signature_hash should be same when only body changes"
    );
}

#[test]
fn test_python_hash_differs_with_signature_change() {
    // Path J: changing signature should change signature_hash
    let registry = ParserRegistry::new();

    let source1 = "def foo(x):\n    return x\n";
    let source2 = "def foo(x, y):\n    return x\n";

    let units1 = registry.parse_file(Path::new("test.py"), source1, Language::Python);
    let units2 = registry.parse_file(Path::new("test.py"), source2, Language::Python);

    assert_eq!(units1.len(), 1);
    assert_eq!(units2.len(), 1);

    // signature_hash should differ
    assert_ne!(
        units1[0].hashes.signature_hash, units2[0].hashes.signature_hash,
        "signature_hash should differ when parameters change"
    );
}

// ============================================================================
// Path K: Class layout hash tests
// ============================================================================

#[test]
fn test_python_class_layout_hash_computed() {
    // Path K: class layout_hash is computed from method names
    let registry = ParserRegistry::new();
    let source = r#"
class MyClass:
    def method_a(self):
        pass

    def method_b(self):
        pass
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].unit_type, UnitType::Class);

    // layout_hash should be non-zero for classes
    assert!(
        units[0].hashes.layout_hash.iter().any(|&b| b != 0),
        "layout_hash should be non-zero for classes with methods"
    );
}

#[test]
fn test_python_class_layout_hash_differs_with_method_order() {
    // Path K: different method order should produce different layout_hash
    let registry = ParserRegistry::new();

    let source1 = r#"
class MyClass:
    def alpha(self):
        pass

    def beta(self):
        pass
"#;
    let source2 = r#"
class MyClass:
    def beta(self):
        pass

    def alpha(self):
        pass
"#;

    let units1 = registry.parse_file(Path::new("test.py"), source1, Language::Python);
    let units2 = registry.parse_file(Path::new("test.py"), source2, Language::Python);

    assert_eq!(units1.len(), 1);
    assert_eq!(units2.len(), 1);

    // layout_hash should differ (method order changed)
    assert_ne!(
        units1[0].hashes.layout_hash, units2[0].hashes.layout_hash,
        "layout_hash should differ when method order changes"
    );
}

#[test]
fn test_python_class_layout_includes_async_methods() {
    // Path K: async methods should be included in layout
    let registry = ParserRegistry::new();
    let source = r#"
class AsyncClass:
    async def fetch(self):
        pass

    def process(self):
        pass
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    assert_eq!(units.len(), 1);

    // layout_hash should be non-zero
    assert!(
        units[0].hashes.layout_hash.iter().any(|&b| b != 0),
        "layout_hash should include async methods"
    );
}

// ============================================================================
// Path L: Function with parameters (signature hash)
// ============================================================================

#[test]
fn test_python_function_signature_with_params() {
    // Path L: signature_hash includes parameter names
    let registry = ParserRegistry::new();

    let source1 = "def func(a, b):\n    pass\n";
    let source2 = "def func(x, y):\n    pass\n";

    let units1 = registry.parse_file(Path::new("test.py"), source1, Language::Python);
    let units2 = registry.parse_file(Path::new("test.py"), source2, Language::Python);

    assert_eq!(units1.len(), 1);
    assert_eq!(units2.len(), 1);

    // signature_hash should differ (different param names)
    assert_ne!(
        units1[0].hashes.signature_hash, units2[0].hashes.signature_hash,
        "signature_hash should differ when param names change"
    );
}

#[test]
fn test_python_function_signature_with_varargs() {
    // Path L: *args and **kwargs in signature
    let registry = ParserRegistry::new();
    let source = "def variadic(*args, **kwargs):\n    pass\n";

    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    assert_eq!(units.len(), 1);
    assert!(
        units[0].hashes.signature_hash.iter().any(|&b| b != 0),
        "signature_hash should be computed for variadic functions"
    );
}

#[test]
fn test_python_function_signature_with_return_type() {
    // Path L: return type annotation affects signature
    let registry = ParserRegistry::new();

    let source1 = "def typed():\n    pass\n";
    let source2 = "def typed() -> int:\n    pass\n";

    let units1 = registry.parse_file(Path::new("test.py"), source1, Language::Python);
    let units2 = registry.parse_file(Path::new("test.py"), source2, Language::Python);

    assert_eq!(units1.len(), 1);
    assert_eq!(units2.len(), 1);

    // signature_hash should differ (return type added)
    assert_ne!(
        units1[0].hashes.signature_hash, units2[0].hashes.signature_hash,
        "signature_hash should differ when return type annotation changes"
    );
}

// ============================================================================
// Path M: Async function hashes include "async" prefix
// ============================================================================

#[test]
fn test_python_async_function_signature_differs_from_sync() {
    // Path M: async def signature_hash differs from sync def
    let registry = ParserRegistry::new();

    let source_sync = "def handler():\n    pass\n";
    let source_async = "async def handler():\n    pass\n";

    let units_sync = registry.parse_file(Path::new("test.py"), source_sync, Language::Python);
    let units_async = registry.parse_file(Path::new("test.py"), source_async, Language::Python);

    assert_eq!(units_sync.len(), 1);
    assert_eq!(units_async.len(), 1);

    // signature_hash should differ due to "async" prefix
    assert_ne!(
        units_sync[0].hashes.signature_hash, units_async[0].hashes.signature_hash,
        "async function signature_hash should differ from sync"
    );
}

#[test]
fn test_python_async_function_hashes_computed() {
    // Path M: all hashes computed for async functions
    let registry = ParserRegistry::new();
    let source = r#"
async def fetch_data(url):
    response = await http_get(url)
    return response
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    assert_eq!(units.len(), 1);
    let hashes = &units[0].hashes;

    assert!(
        hashes.full_hash.iter().any(|&b| b != 0),
        "async function full_hash should be computed"
    );
    assert!(
        hashes.signature_hash.iter().any(|&b| b != 0),
        "async function signature_hash should be computed"
    );
    assert!(
        hashes.body_hash.iter().any(|&b| b != 0),
        "async function body_hash should be computed"
    );
}

// ============================================================================
// Path N: Class with bases (signature hash)
// ============================================================================

#[test]
fn test_python_class_signature_with_bases() {
    // Path N: class signature includes base classes
    let registry = ParserRegistry::new();

    let source1 = "class Child:\n    pass\n";
    let source2 = "class Child(Parent):\n    pass\n";

    let units1 = registry.parse_file(Path::new("test.py"), source1, Language::Python);
    let units2 = registry.parse_file(Path::new("test.py"), source2, Language::Python);

    assert_eq!(units1.len(), 1);
    assert_eq!(units2.len(), 1);

    // signature_hash should differ (base class added)
    assert_ne!(
        units1[0].hashes.signature_hash, units2[0].hashes.signature_hash,
        "class signature_hash should differ when bases change"
    );
}

#[test]
fn test_python_class_multiple_bases() {
    // Path N: multiple base classes
    let registry = ParserRegistry::new();
    let source = "class Multi(Base1, Base2, Mixin):\n    pass\n";

    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    assert_eq!(units.len(), 1);
    assert!(
        units[0].hashes.signature_hash.iter().any(|&b| b != 0),
        "signature_hash should be computed for class with multiple bases"
    );
}

// ============================================================================
// Path O: Edge cases - boundary offsets
// ============================================================================

#[test]
fn test_python_single_line_function() {
    // Path O: single line function (lambda-style def)
    let registry = ParserRegistry::new();
    let source = "def f(): pass";

    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].start_line, 1);
    assert_eq!(units[0].end_line, 1);
}

#[test]
fn test_python_empty_function_body() {
    // Path O: function with only docstring (empty effective body)
    let registry = ParserRegistry::new();
    let source = r#"
def documented():
    """This function does nothing."""
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    assert_eq!(units.len(), 1);
    // body_hash should still be computed
    assert!(
        units[0].hashes.body_hash.iter().any(|&b| b != 0),
        "body_hash should be computed even for docstring-only body"
    );
}

#[test]
fn test_python_class_with_only_pass() {
    // Path O: minimal class
    let registry = ParserRegistry::new();
    let source = "class Empty:\n    pass\n";

    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    assert_eq!(units.len(), 1);
    // body_hash for class with just pass
    assert!(
        units[0].hashes.body_hash.iter().any(|&b| b != 0),
        "body_hash should be computed for class with pass"
    );
}

#[test]
fn test_python_unicode_function_name() {
    // Path O: unicode in function names
    let registry = ParserRegistry::new();
    let source = "def calculate_\u{03C0}():\n    return 3.14159\n";

    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    assert_eq!(units.len(), 1);
    assert!(
        units[0].name.contains('\u{03C0}'),
        "should handle unicode in function name"
    );
}

#[test]
fn test_python_deeply_indented_code() {
    // Path O: deeply indented code
    let registry = ParserRegistry::new();
    let source = r#"
def nested():
    if True:
        if True:
            if True:
                return "deep"
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    assert_eq!(units.len(), 1);
    // Should still compute proper hashes
    assert!(
        units[0].hashes.body_hash.iter().any(|&b| b != 0),
        "body_hash should handle deeply indented code"
    );
}

#[test]
fn test_python_class_with_annotated_attribute() {
    // Path O: class with annotated attributes in layout
    let registry = ParserRegistry::new();
    let source = r#"
class DataClass:
    name: str
    value: int

    def get_name(self):
        return self.name
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    assert_eq!(units.len(), 1);
    // layout_hash should include attributes
    assert!(
        units[0].hashes.layout_hash.iter().any(|&b| b != 0),
        "layout_hash should include annotated attributes"
    );
}
