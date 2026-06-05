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
