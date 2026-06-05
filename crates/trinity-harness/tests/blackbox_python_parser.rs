//! Blackbox tests for PythonParser via ParserRegistry.
//!
//! CLEANROOM: Tests are written based on the public contract only.
//! The implementation file (python.rs) was NOT read.
//!
//! Test coverage plan:
//!   - T1: Basic function extraction
//!   - T2: Class extraction
//!   - T3: Method extraction (within classes)
//!   - T4: Import extraction
//!   - T5: Multiple items in single file
//!   - T6: Nested structures
//!   - T7: Hash computation validation
//!   - T8: Line number accuracy
//!   - T9: Edge cases (empty, invalid syntax)
//!   - T10: Real-world code patterns
//!   - T11: Decorators
//!   - T12: Async functions
//!   - T13: Lambda and special constructs
//!   - T14: Complex inheritance
//!   - T15: Docstrings and annotations

use std::path::Path;
use trinity_harness::{Language, ParserRegistry, UnitType};

// ============================================================================
// T1: Basic Function Extraction
// ============================================================================

#[test]
fn blackbox_python_simple_function() {
    let registry = ParserRegistry::new();
    let source = "def simple():\n    pass";
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "simple");
    assert_eq!(units[0].unit_type, UnitType::Function);
    assert_eq!(units[0].language, Language::Python);
}

#[test]
fn blackbox_python_function_with_params() {
    let registry = ParserRegistry::new();
    let source = r#"
def add(x, y):
    return x + y
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "add");
    assert_eq!(units[0].unit_type, UnitType::Function);
}

#[test]
fn blackbox_python_function_with_typed_params() {
    let registry = ParserRegistry::new();
    let source = r#"
def typed_add(x: int, y: int) -> int:
    return x + y
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "typed_add");
    assert_eq!(units[0].unit_type, UnitType::Function);
}

#[test]
fn blackbox_python_function_default_args() {
    let registry = ParserRegistry::new();
    let source = r#"
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "greet");
    assert_eq!(units[0].unit_type, UnitType::Function);
}

#[test]
fn blackbox_python_function_kwargs() {
    let registry = ParserRegistry::new();
    let source = r#"
def variadic(*args, **kwargs):
    pass
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "variadic");
    assert_eq!(units[0].unit_type, UnitType::Function);
}

// ============================================================================
// T2: Class Extraction
// ============================================================================

#[test]
fn blackbox_python_simple_class() {
    let registry = ParserRegistry::new();
    let source = r#"
class Simple:
    pass
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    let classes: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Class).collect();
    assert_eq!(classes.len(), 1);
    assert_eq!(classes[0].name, "Simple");
    assert_eq!(classes[0].language, Language::Python);
}

#[test]
fn blackbox_python_class_with_inheritance() {
    let registry = ParserRegistry::new();
    let source = r#"
class Parent:
    pass

class Child(Parent):
    pass
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    let classes: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Class).collect();
    assert_eq!(classes.len(), 2);

    let names: Vec<_> = classes.iter().map(|c| c.name.as_str()).collect();
    assert!(names.contains(&"Parent"));
    assert!(names.contains(&"Child"));
}

#[test]
fn blackbox_python_class_multiple_inheritance() {
    let registry = ParserRegistry::new();
    let source = r#"
class A:
    pass

class B:
    pass

class C(A, B):
    pass
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    let classes: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Class).collect();
    assert_eq!(classes.len(), 3);
    assert!(classes.iter().any(|c| c.name == "C"));
}

#[test]
fn blackbox_python_class_with_metaclass() {
    let registry = ParserRegistry::new();
    let source = r#"
class Meta(type):
    pass

class MyClass(metaclass=Meta):
    pass
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    let classes: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Class).collect();
    assert_eq!(classes.len(), 2);
}

// ============================================================================
// T3: Method Extraction
// ============================================================================

#[test]
fn blackbox_python_class_with_methods() {
    let registry = ParserRegistry::new();
    let source = r#"
class Counter:
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1

    def get_value(self):
        return self.value
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    let classes: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Class).collect();
    let methods: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Method).collect();

    assert_eq!(classes.len(), 1);
    assert_eq!(classes[0].name, "Counter");

    // Methods may or may not be extracted separately depending on implementation
    // The key is that the class itself is extracted correctly
    // If methods are extracted, verify they have the right type
    for method in &methods {
        assert_eq!(method.unit_type, UnitType::Method);
        assert_eq!(method.language, Language::Python);
    }
}

#[test]
fn blackbox_python_static_method() {
    let registry = ParserRegistry::new();
    let source = r#"
class Utils:
    @staticmethod
    def helper(x):
        return x * 2
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    // Class should be found
    assert!(units.iter().any(|u| u.name == "Utils" && u.unit_type == UnitType::Class));
}

#[test]
fn blackbox_python_class_method() {
    let registry = ParserRegistry::new();
    let source = r#"
class Factory:
    @classmethod
    def create(cls):
        return cls()
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    assert!(units.iter().any(|u| u.name == "Factory" && u.unit_type == UnitType::Class));
}

#[test]
fn blackbox_python_property() {
    let registry = ParserRegistry::new();
    let source = r#"
class Person:
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    assert!(units.iter().any(|u| u.name == "Person" && u.unit_type == UnitType::Class));
}

// ============================================================================
// T4: Import Extraction
// ============================================================================

#[test]
fn blackbox_python_simple_import() {
    let registry = ParserRegistry::new();
    let source = "import os";
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    // Imports may be tracked as Module units or may be ignored
    // Check that parser handles it gracefully
    let _ = units; // Parser should not panic
}

#[test]
fn blackbox_python_from_import() {
    let registry = ParserRegistry::new();
    let source = "from typing import List, Dict, Optional";
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    // Parser should handle from imports gracefully
    let _ = units;
}

#[test]
fn blackbox_python_import_alias() {
    let registry = ParserRegistry::new();
    let source = r#"
import numpy as np
from collections import defaultdict as dd
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    // Parser should handle import aliases
    let _ = units;
}

#[test]
fn blackbox_python_relative_import() {
    let registry = ParserRegistry::new();
    let source = r#"
from . import module
from .. import parent_module
from .sibling import func
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    let _ = units;
}

// ============================================================================
// T5: Multiple Items in Single File
// ============================================================================

#[test]
fn blackbox_python_mixed_items() {
    let registry = ParserRegistry::new();
    let source = r#"
import os

def standalone_function():
    pass

class DataProcessor:
    def process(self, data):
        return data

def another_function(x):
    return x * 2

class AnotherClass:
    pass
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    let functions: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Function).collect();
    let classes: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Class).collect();

    assert_eq!(functions.len(), 2, "should find 2 top-level functions");
    assert_eq!(classes.len(), 2, "should find 2 classes");
}

#[test]
fn blackbox_python_functions_and_classes_interleaved() {
    let registry = ParserRegistry::new();
    let source = r#"
def func1():
    pass

class A:
    pass

def func2():
    pass

class B:
    pass

def func3():
    pass
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    let functions = units.iter().filter(|u| u.unit_type == UnitType::Function).count();
    let classes = units.iter().filter(|u| u.unit_type == UnitType::Class).count();

    assert_eq!(functions, 3);
    assert_eq!(classes, 2);
}

// ============================================================================
// T6: Nested Structures
// ============================================================================

#[test]
fn blackbox_python_nested_function() {
    let registry = ParserRegistry::new();
    let source = r#"
def outer():
    def inner():
        return "inner"
    return inner()
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    // At minimum, outer function should be found
    assert!(units.iter().any(|u| u.name == "outer" && u.unit_type == UnitType::Function));
}

#[test]
fn blackbox_python_nested_class() {
    let registry = ParserRegistry::new();
    let source = r#"
class Outer:
    class Inner:
        pass
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    // At minimum, Outer should be found
    assert!(units.iter().any(|u| u.name == "Outer" && u.unit_type == UnitType::Class));
}

#[test]
fn blackbox_python_closure() {
    let registry = ParserRegistry::new();
    let source = r#"
def make_adder(n):
    def adder(x):
        return x + n
    return adder
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    assert!(units.iter().any(|u| u.name == "make_adder" && u.unit_type == UnitType::Function));
}

// ============================================================================
// T7: Hash Computation Validation
// ============================================================================

#[test]
fn blackbox_python_hash_populated() {
    let registry = ParserRegistry::new();
    let source = r#"
def compute(x):
    return x * 2
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    assert_eq!(units.len(), 1);

    // Full hash should be non-zero (not default)
    let hashes = &units[0].hashes;
    assert_ne!(hashes.full_hash, [0u8; 32], "full_hash should be computed");
}

#[test]
fn blackbox_python_different_functions_different_hashes() {
    let registry = ParserRegistry::new();

    let source1 = "def foo():\n    x = 1";
    let source2 = "def bar():\n    y = 2";

    let units1 = registry.parse_file(Path::new("a.py"), source1, Language::Python);
    let units2 = registry.parse_file(Path::new("b.py"), source2, Language::Python);

    assert_eq!(units1.len(), 1);
    assert_eq!(units2.len(), 1);

    // Different function content should produce different hashes
    assert_ne!(
        units1[0].hashes.full_hash,
        units2[0].hashes.full_hash,
        "different functions should have different full hashes"
    );
}

#[test]
fn blackbox_python_same_function_same_hash() {
    let registry = ParserRegistry::new();

    let source = "def identical():\n    print('hello')";

    let units1 = registry.parse_file(Path::new("a.py"), source, Language::Python);
    let units2 = registry.parse_file(Path::new("b.py"), source, Language::Python);

    assert_eq!(units1.len(), 1);
    assert_eq!(units2.len(), 1);

    // Same function content should produce same hash
    assert_eq!(
        units1[0].hashes.full_hash,
        units2[0].hashes.full_hash,
        "identical functions should have same full hash"
    );
}

// ============================================================================
// T8: Line Number Accuracy
// ============================================================================

#[test]
fn blackbox_python_line_numbers_start_line() {
    let registry = ParserRegistry::new();
    let source = r#"
def first():
    pass

def second():
    pass
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    let functions: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Function).collect();
    assert_eq!(functions.len(), 2);

    let first = functions.iter().find(|u| u.name == "first").unwrap();
    let second = functions.iter().find(|u| u.name == "second").unwrap();

    // First function should start early in file
    assert!(first.start_line >= 1, "first should start early in file");
    // Second function should start after first
    assert!(second.start_line > first.start_line, "second should start after first");
}

#[test]
fn blackbox_python_line_numbers_end_after_start() {
    let registry = ParserRegistry::new();
    let source = r#"
def multiline():
    a = 1
    b = 2
    c = a + b
    print(c)
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    assert_eq!(units.len(), 1);
    assert!(
        units[0].end_line >= units[0].start_line,
        "end_line should be >= start_line"
    );
}

#[test]
fn blackbox_python_class_span() {
    let registry = ParserRegistry::new();
    let source = r#"
class BigClass:
    def method1(self):
        pass

    def method2(self):
        pass

    def method3(self):
        pass
"#;
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    let classes: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Class).collect();
    assert_eq!(classes.len(), 1);

    // Class should span multiple lines
    let class_span = classes[0].end_line - classes[0].start_line;
    assert!(class_span >= 5, "class should span multiple lines");
}

// ============================================================================
// T9: Edge Cases
// ============================================================================

#[test]
fn blackbox_python_empty_source() {
    let registry = ParserRegistry::new();
    let units = registry.parse_file(Path::new("empty.py"), "", Language::Python);
    assert!(units.is_empty(), "empty source should produce no units");
}

#[test]
fn blackbox_python_whitespace_only() {
    let registry = ParserRegistry::new();
    let units = registry.parse_file(Path::new("whitespace.py"), "   \n\n\t\t\n   ", Language::Python);
    assert!(units.is_empty(), "whitespace-only source should produce no units");
}

#[test]
fn blackbox_python_comments_only() {
    let registry = ParserRegistry::new();
    let source = r#"
# This is a comment
# Another comment
"""
Docstring not assigned to anything
"""
"#;
    let units = registry.parse_file(Path::new("comments.py"), source, Language::Python);
    assert!(units.is_empty(), "comments-only source should produce no units");
}

#[test]
fn blackbox_python_invalid_syntax() {
    let registry = ParserRegistry::new();
    let source = "def broken(\n    x = ";
    let units = registry.parse_file(Path::new("broken.py"), source, Language::Python);
    // Invalid syntax should either return empty or gracefully handle
    // The exact behavior depends on implementation, but it should not panic
    let _ = units;
}

#[test]
fn blackbox_python_just_expressions() {
    let registry = ParserRegistry::new();
    let source = r#"
1 + 2
"hello"
[1, 2, 3]
"#;
    let units = registry.parse_file(Path::new("expr.py"), source, Language::Python);
    // Expressions should not produce code units
    assert!(units.is_empty(), "plain expressions should not produce units");
}

#[test]
fn blackbox_python_variable_assignments() {
    let registry = ParserRegistry::new();
    let source = r#"
x = 10
y = "hello"
z = [1, 2, 3]
"#;
    let units = registry.parse_file(Path::new("vars.py"), source, Language::Python);
    // Variable assignments are not functions/classes/methods
    assert!(units.is_empty(), "variable assignments should not produce units");
}

// ============================================================================
// T10: Real-World Code Patterns
// ============================================================================

#[test]
fn blackbox_python_dataclass() {
    let registry = ParserRegistry::new();
    let source = r#"
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

    def distance_from_origin(self) -> float:
        return (self.x ** 2 + self.y ** 2) ** 0.5
"#;
    let units = registry.parse_file(Path::new("dataclass.py"), source, Language::Python);

    assert!(units.iter().any(|u| u.name == "Point" && u.unit_type == UnitType::Class));
}

#[test]
fn blackbox_python_context_manager() {
    let registry = ParserRegistry::new();
    let source = r#"
class FileHandler:
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        self.file = open(self.filename)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
"#;
    let units = registry.parse_file(Path::new("ctx.py"), source, Language::Python);

    assert!(units.iter().any(|u| u.name == "FileHandler" && u.unit_type == UnitType::Class));
}

#[test]
fn blackbox_python_exception_class() {
    let registry = ParserRegistry::new();
    let source = r#"
class CustomError(Exception):
    def __init__(self, message, code):
        super().__init__(message)
        self.code = code
"#;
    let units = registry.parse_file(Path::new("error.py"), source, Language::Python);

    assert!(units.iter().any(|u| u.name == "CustomError" && u.unit_type == UnitType::Class));
}

#[test]
fn blackbox_python_abstract_class() {
    let registry = ParserRegistry::new();
    let source = r#"
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass

    @abstractmethod
    def perimeter(self) -> float:
        pass
"#;
    let units = registry.parse_file(Path::new("abstract.py"), source, Language::Python);

    assert!(units.iter().any(|u| u.name == "Shape" && u.unit_type == UnitType::Class));
}

// ============================================================================
// T11: Decorators
// ============================================================================

#[test]
fn blackbox_python_single_decorator() {
    let registry = ParserRegistry::new();
    let source = r#"
@decorator
def decorated():
    pass
"#;
    let units = registry.parse_file(Path::new("deco.py"), source, Language::Python);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "decorated");
    assert_eq!(units[0].unit_type, UnitType::Function);
}

#[test]
fn blackbox_python_multiple_decorators() {
    let registry = ParserRegistry::new();
    let source = r#"
@decorator1
@decorator2
@decorator3
def multi_decorated():
    pass
"#;
    let units = registry.parse_file(Path::new("multi_deco.py"), source, Language::Python);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "multi_decorated");
}

#[test]
fn blackbox_python_decorator_with_args() {
    let registry = ParserRegistry::new();
    let source = r#"
@decorator(arg1, arg2=value)
def with_args():
    pass
"#;
    let units = registry.parse_file(Path::new("deco_args.py"), source, Language::Python);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "with_args");
}

#[test]
fn blackbox_python_class_decorator() {
    let registry = ParserRegistry::new();
    let source = r#"
@register
@dataclass
class DecoratedClass:
    value: int
"#;
    let units = registry.parse_file(Path::new("class_deco.py"), source, Language::Python);

    assert!(units.iter().any(|u| u.name == "DecoratedClass" && u.unit_type == UnitType::Class));
}

// ============================================================================
// T12: Async Functions
// ============================================================================

#[test]
fn blackbox_python_async_function() {
    let registry = ParserRegistry::new();
    let source = r#"
async def fetch_data():
    return "data"
"#;
    let units = registry.parse_file(Path::new("async.py"), source, Language::Python);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "fetch_data");
    assert_eq!(units[0].unit_type, UnitType::Function);
}

#[test]
fn blackbox_python_async_method() {
    let registry = ParserRegistry::new();
    let source = r#"
class Client:
    async def request(self):
        return []
"#;
    let units = registry.parse_file(Path::new("async_method.py"), source, Language::Python);

    assert!(units.iter().any(|u| u.name == "Client" && u.unit_type == UnitType::Class));
}

#[test]
fn blackbox_python_async_context_manager() {
    let registry = ParserRegistry::new();
    let source = r#"
class AsyncConnection:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass
"#;
    let units = registry.parse_file(Path::new("async_ctx.py"), source, Language::Python);

    assert!(units.iter().any(|u| u.name == "AsyncConnection" && u.unit_type == UnitType::Class));
}

#[test]
fn blackbox_python_async_generator() {
    let registry = ParserRegistry::new();
    let source = r#"
async def async_range(n):
    for i in range(n):
        yield i
"#;
    let units = registry.parse_file(Path::new("async_gen.py"), source, Language::Python);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "async_range");
}

// ============================================================================
// T13: Lambda and Special Constructs
// ============================================================================

#[test]
fn blackbox_python_lambda_not_extracted() {
    let registry = ParserRegistry::new();
    let source = r#"
square = lambda x: x ** 2
"#;
    let units = registry.parse_file(Path::new("lambda.py"), source, Language::Python);

    // Lambdas assigned to variables are not top-level function definitions
    // They should not be extracted as Function units
    let functions = units.iter().filter(|u| u.unit_type == UnitType::Function).count();
    assert_eq!(functions, 0, "lambdas should not be extracted as functions");
}

#[test]
fn blackbox_python_comprehension() {
    let registry = ParserRegistry::new();
    let source = r#"
squares = [x**2 for x in range(10)]
evens = {x for x in range(10) if x % 2 == 0}
mapping = {x: x**2 for x in range(10)}
"#;
    let units = registry.parse_file(Path::new("comp.py"), source, Language::Python);

    // Comprehensions are not functions/classes
    assert!(units.is_empty());
}

// ============================================================================
// T14: Complex Inheritance
// ============================================================================

#[test]
fn blackbox_python_diamond_inheritance() {
    let registry = ParserRegistry::new();
    let source = r#"
class A:
    def method(self):
        pass

class B(A):
    pass

class C(A):
    pass

class D(B, C):
    pass
"#;
    let units = registry.parse_file(Path::new("diamond.py"), source, Language::Python);

    let classes: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Class).collect();
    assert_eq!(classes.len(), 4);
}

#[test]
fn blackbox_python_mixin_pattern() {
    let registry = ParserRegistry::new();
    let source = r#"
class LoggingMixin:
    def log(self, message):
        print(message)

class SerializeMixin:
    def to_json(self):
        return "{}"

class MyClass(LoggingMixin, SerializeMixin):
    pass
"#;
    let units = registry.parse_file(Path::new("mixin.py"), source, Language::Python);

    let classes: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Class).collect();
    assert_eq!(classes.len(), 3);
}

// ============================================================================
// T15: Docstrings and Annotations
// ============================================================================

#[test]
fn blackbox_python_function_docstring() {
    let registry = ParserRegistry::new();
    let source = r#"
def documented():
    """This is a docstring.

    It spans multiple lines.
    """
    pass
"#;
    let units = registry.parse_file(Path::new("docstring.py"), source, Language::Python);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "documented");
}

#[test]
fn blackbox_python_class_docstring() {
    let registry = ParserRegistry::new();
    let source = r#"
class Documented:
    """A class with documentation.

    Attributes:
        value: The value
    """

    def __init__(self):
        self.value = 0
"#;
    let units = registry.parse_file(Path::new("class_doc.py"), source, Language::Python);

    assert!(units.iter().any(|u| u.name == "Documented" && u.unit_type == UnitType::Class));
}

#[test]
fn blackbox_python_type_annotations() {
    let registry = ParserRegistry::new();
    let source = r#"
from typing import List, Dict, Optional

def process(items: List[int], config: Dict[str, str]) -> Optional[int]:
    if items:
        return items[0]
    return None
"#;
    let units = registry.parse_file(Path::new("typed.py"), source, Language::Python);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "process");
}

#[test]
fn blackbox_python_variable_annotations() {
    let registry = ParserRegistry::new();
    let source = r#"
class Config:
    debug: bool = False
    port: int = 8080
    host: str = "localhost"
"#;
    let units = registry.parse_file(Path::new("annotated.py"), source, Language::Python);

    assert!(units.iter().any(|u| u.name == "Config" && u.unit_type == UnitType::Class));
}

// ============================================================================
// Language Detection (via ParserRegistry)
// ============================================================================

#[test]
fn blackbox_python_language_detection() {
    assert_eq!(
        ParserRegistry::detect_language(Path::new("test.py")),
        Some(Language::Python)
    );
    assert_eq!(
        ParserRegistry::detect_language(Path::new("/path/to/module.py")),
        Some(Language::Python)
    );
    assert_eq!(
        ParserRegistry::detect_language(Path::new("__init__.py")),
        Some(Language::Python)
    );
}

#[test]
fn blackbox_python_language_returned_correct() {
    let registry = ParserRegistry::new();
    let source = "def test():\n    pass";
    let units = registry.parse_file(Path::new("test.py"), source, Language::Python);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].language, Language::Python);
}

// ============================================================================
// ParserRegistry API Tests
// ============================================================================

#[test]
fn blackbox_python_registry_default() {
    // ParserRegistry should implement Default
    let registry = ParserRegistry::default();
    let units = registry.parse_file(Path::new("test.py"), "def x():\n    pass", Language::Python);
    assert_eq!(units.len(), 1);
}

#[test]
fn blackbox_python_parser_reusable() {
    // Same registry should be usable for multiple parse calls
    let registry = ParserRegistry::new();

    let units1 = registry.parse_file(Path::new("a.py"), "def a():\n    pass", Language::Python);
    let units2 = registry.parse_file(Path::new("b.py"), "def b():\n    pass", Language::Python);
    let units3 = registry.parse_file(Path::new("c.py"), "def c():\n    pass", Language::Python);

    assert_eq!(units1.len(), 1);
    assert_eq!(units2.len(), 1);
    assert_eq!(units3.len(), 1);

    assert_eq!(units1[0].name, "a");
    assert_eq!(units2[0].name, "b");
    assert_eq!(units3[0].name, "c");
}

// ============================================================================
// Comprehensive Real-World File
// ============================================================================

#[test]
fn blackbox_python_realistic_module() {
    let registry = ParserRegistry::new();
    let source = r#"
"""Module documentation."""

from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration holder."""

    def __init__(self, name: str, options: Optional[dict] = None):
        self.name = name
        self.options = options or {}

    def get(self, key: str) -> Optional[str]:
        return self.options.get(key)

    def set(self, key: str, value: str) -> None:
        self.options[key] = value


class ConfigError(Exception):
    """Configuration error."""
    pass


def load_config(path: str) -> Config:
    """Load configuration from file."""
    return Config(path)


def validate_config(config: Config) -> List[str]:
    """Validate configuration and return errors."""
    errors = []
    if not config.name:
        errors.append("Name is required")
    return errors


async def async_load_config(path: str) -> Config:
    """Asynchronously load configuration."""
    return Config(path)


class _InternalHelper:
    """Private helper class."""

    @staticmethod
    def helper():
        pass
"#;
    let units = registry.parse_file(Path::new("realistic.py"), source, Language::Python);

    // Verify we find the expected items
    let classes: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Class).collect();
    let functions: Vec<_> = units.iter().filter(|u| u.unit_type == UnitType::Function).collect();

    // Should find Config, ConfigError, _InternalHelper classes
    assert!(classes.len() >= 3, "should find at least 3 classes");
    assert!(classes.iter().any(|c| c.name == "Config"), "should find Config class");
    assert!(classes.iter().any(|c| c.name == "ConfigError"), "should find ConfigError class");
    assert!(classes.iter().any(|c| c.name == "_InternalHelper"), "should find _InternalHelper class");

    // Should find load_config, validate_config, async_load_config functions
    assert!(functions.len() >= 3, "should find at least 3 functions");
    assert!(functions.iter().any(|f| f.name == "load_config"), "should find load_config function");
    assert!(functions.iter().any(|f| f.name == "validate_config"), "should find validate_config function");
    assert!(functions.iter().any(|f| f.name == "async_load_config"), "should find async_load_config function");
}

// ============================================================================
// Special Python Constructs
// ============================================================================

#[test]
fn blackbox_python_dunder_methods() {
    let registry = ParserRegistry::new();
    let source = r#"
class Container:
    def __init__(self):
        self.items = []

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def __setitem__(self, index, value):
        self.items[index] = value

    def __iter__(self):
        return iter(self.items)

    def __repr__(self):
        return f"Container({self.items})"
"#;
    let units = registry.parse_file(Path::new("dunder.py"), source, Language::Python);

    assert!(units.iter().any(|u| u.name == "Container" && u.unit_type == UnitType::Class));
}

#[test]
fn blackbox_python_slots() {
    let registry = ParserRegistry::new();
    let source = r#"
class Optimized:
    __slots__ = ['x', 'y']

    def __init__(self, x, y):
        self.x = x
        self.y = y
"#;
    let units = registry.parse_file(Path::new("slots.py"), source, Language::Python);

    assert!(units.iter().any(|u| u.name == "Optimized" && u.unit_type == UnitType::Class));
}

#[test]
fn blackbox_python_enum_class() {
    let registry = ParserRegistry::new();
    let source = r#"
from enum import Enum, auto

class Status(Enum):
    PENDING = auto()
    ACTIVE = auto()
    COMPLETED = auto()
"#;
    let units = registry.parse_file(Path::new("enum.py"), source, Language::Python);

    assert!(units.iter().any(|u| u.name == "Status" && u.unit_type == UnitType::Class));
}

#[test]
fn blackbox_python_protocol() {
    let registry = ParserRegistry::new();
    let source = r#"
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> None:
        ...
"#;
    let units = registry.parse_file(Path::new("protocol.py"), source, Language::Python);

    assert!(units.iter().any(|u| u.name == "Drawable" && u.unit_type == UnitType::Class));
}

// ============================================================================
// Unicode and Special Characters
// ============================================================================

#[test]
fn blackbox_python_unicode_identifiers() {
    let registry = ParserRegistry::new();
    let source = r#"
def calculate_π():
    return 3.14159

class Naïve:
    pass
"#;
    let units = registry.parse_file(Path::new("unicode.py"), source, Language::Python);

    // Parser should handle unicode identifiers
    let _ = units;
}

#[test]
fn blackbox_python_unicode_strings() {
    let registry = ParserRegistry::new();
    let source = r#"
def greet():
    return "Hello, 世界! 🌍"
"#;
    let units = registry.parse_file(Path::new("unicode_str.py"), source, Language::Python);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "greet");
}

// ============================================================================
// Python 3.10+ Features (Pattern Matching, etc.)
// ============================================================================

#[test]
fn blackbox_python_match_statement() {
    let registry = ParserRegistry::new();
    let source = r#"
def process_command(command):
    match command:
        case "start":
            return 1
        case "stop":
            return 0
        case _:
            return -1
"#;
    let units = registry.parse_file(Path::new("match.py"), source, Language::Python);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "process_command");
}

#[test]
fn blackbox_python_walrus_operator() {
    let registry = ParserRegistry::new();
    let source = r#"
def process_with_walrus(data):
    if (n := len(data)) > 10:
        return n
    return 0
"#;
    let units = registry.parse_file(Path::new("walrus.py"), source, Language::Python);

    assert_eq!(units.len(), 1);
    assert_eq!(units[0].name, "process_with_walrus");
}
