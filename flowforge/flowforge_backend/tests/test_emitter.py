"""Tests for emitter module.

This module tests Python source code emission from AST,
including formatting and validation helpers.
"""

from __future__ import annotations

import ast
import sys
import pytest

from ..codegen.emitter import (
    emit_python,
    emit_python_minimal,
    emit_class,
    validate_syntax,
    roundtrip_ast,
    _add_blank_lines,
    _format_with_black,
)


class TestEmitPython:
    """Tests for emit_python function."""

    def test_basic_module(self):
        """Test emitting basic module."""
        module = ast.parse("x = 1")
        source = emit_python(module, format_with_black=False)

        assert "x = 1" in source

    def test_includes_header_by_default(self):
        """Test that header comment is included by default."""
        module = ast.parse("x = 1")
        source = emit_python(module, format_with_black=False)

        assert "FlowForge" in source

    def test_no_header_when_disabled(self):
        """Test that header is excluded when disabled."""
        module = ast.parse("x = 1")
        source = emit_python(module, format_with_black=False, add_header=False)

        assert "FlowForge" not in source

    def test_trailing_newline(self):
        """Test that source ends with newline."""
        module = ast.parse("x = 1")
        source = emit_python(module, format_with_black=False)

        assert source.endswith("\n")

    def test_class_definition(self):
        """Test emitting class definition."""
        module = ast.parse("""
class Player:
    name: str
    health: int = 100
""")
        source = emit_python(module, format_with_black=False)

        assert "class Player" in source
        assert "name" in source
        assert "health" in source

    def test_function_definition(self):
        """Test emitting function definition."""
        module = ast.parse("""
def greet(name: str) -> str:
    return f"Hello, {name}"
""")
        source = emit_python(module, format_with_black=False)

        assert "def greet" in source
        assert "name: str" in source

    def test_decorated_class(self):
        """Test emitting decorated class."""
        module = ast.parse("""
@dataclass
class Position:
    x: float
    y: float
""")
        source = emit_python(module, format_with_black=False)

        assert "@dataclass" in source
        assert "class Position" in source

    def test_imports_preserved(self):
        """Test that imports are preserved."""
        module = ast.parse("""
from typing import List, Optional
import os
""")
        source = emit_python(module, format_with_black=False, add_header=False)

        assert "from typing import" in source
        assert "import os" in source

    def test_complex_module(self):
        """Test emitting complex module."""
        module = ast.parse("""
from typing import Optional

class Entity:
    id: int
    name: str = ""

class Player(Entity):
    health: int = 100
    position: Optional[tuple] = None
""")
        source = emit_python(module, format_with_black=False)

        assert "class Entity" in source
        assert "class Player" in source

    def test_fix_missing_locations_called(self):
        """Test that fix_missing_locations is called."""
        # Create module without line numbers
        module = ast.Module(body=[
            ast.Assign(
                targets=[ast.Name(id='x', ctx=ast.Store())],
                value=ast.Constant(value=1),
            )
        ], type_ignores=[])

        # Should not raise - fix_missing_locations should be called
        source = emit_python(module, format_with_black=False, add_header=False)
        assert "x = 1" in source


class TestEmitPythonMinimal:
    """Tests for emit_python_minimal function."""

    def test_minimal_output(self):
        """Test minimal output without formatting or header."""
        module = ast.parse("x = 1")
        source = emit_python_minimal(module)

        assert "x = 1" in source
        # Should not have FlowForge header
        assert "FlowForge" not in source

    def test_trailing_newline(self):
        """Test that minimal output has trailing newline."""
        module = ast.parse("x = 1")
        source = emit_python_minimal(module)

        assert source.endswith("\n")

    def test_preserves_code(self):
        """Test that code is preserved correctly."""
        original = """
class Foo:
    x: int = 0

    def bar(self) -> int:
        return self.x
"""
        module = ast.parse(original)
        source = emit_python_minimal(module)

        assert "class Foo" in source
        assert "def bar" in source


class TestEmitClass:
    """Tests for emit_class function."""

    def test_single_class(self):
        """Test emitting single class."""
        class_def = ast.parse("""
class Player:
    health: int = 100
""").body[0]

        source = emit_class(class_def)

        assert "class Player" in source
        assert "health" in source

    def test_decorated_class(self):
        """Test emitting decorated class."""
        class_def = ast.parse("""
@component
class Position:
    x: float
    y: float
""").body[0]

        source = emit_class(class_def)

        assert "@component" in source
        assert "class Position" in source

    def test_without_imports(self):
        """Test class without imports."""
        class_def = ast.parse("class Empty: pass").body[0]
        source = emit_class(class_def, include_imports=False)

        assert "from trinity" not in source
        assert "from __future__" not in source

    def test_with_imports(self):
        """Test class with imports."""
        class_def = ast.parse("class Empty: pass").body[0]
        source = emit_class(class_def, include_imports=True)

        assert "from __future__ import annotations" in source
        assert "from trinity import" in source

    def test_class_with_methods(self):
        """Test class with methods."""
        class_def = ast.parse("""
class Calculator:
    def add(self, a: int, b: int) -> int:
        return a + b
""").body[0]

        source = emit_class(class_def)

        assert "class Calculator" in source
        assert "def add" in source


class TestValidateSyntax:
    """Tests for validate_syntax function."""

    def test_valid_code(self):
        """Test valid code returns True."""
        is_valid, error = validate_syntax("x = 1")

        assert is_valid is True
        assert error is None

    def test_valid_multiline(self):
        """Test valid multiline code."""
        source = """
def foo():
    return 42
"""
        is_valid, error = validate_syntax(source)

        assert is_valid is True
        assert error is None

    def test_invalid_code(self):
        """Test invalid code returns False with error."""
        is_valid, error = validate_syntax("def foo(")

        assert is_valid is False
        assert error is not None

    def test_error_message_content(self):
        """Test that error message contains useful info."""
        is_valid, error = validate_syntax("if True\n    pass")

        assert is_valid is False
        assert error is not None
        assert len(error) > 0

    def test_empty_code_valid(self):
        """Test that empty code is valid."""
        is_valid, error = validate_syntax("")

        assert is_valid is True
        assert error is None


class TestRoundtripAst:
    """Tests for roundtrip_ast function."""

    def test_simple_roundtrip(self):
        """Test simple code roundtrip."""
        original = ast.parse("x = 1")
        result = roundtrip_ast(original)

        assert isinstance(result, ast.Module)

    def test_class_roundtrip(self):
        """Test class definition roundtrip."""
        original = ast.parse("""
class Player:
    name: str
    health: int = 100
""")
        result = roundtrip_ast(original)

        assert isinstance(result, ast.Module)
        # Should have the class definition
        assert any(isinstance(s, ast.ClassDef) and s.name == "Player" for s in result.body)

    def test_decorated_class_roundtrip(self):
        """Test decorated class roundtrip."""
        original = ast.parse("""
@component
class Position:
    x: float = 0.0
    y: float = 0.0
""")
        result = roundtrip_ast(original)

        class_def = next(s for s in result.body if isinstance(s, ast.ClassDef))
        assert class_def.name == "Position"
        assert len(class_def.decorator_list) == 1

    def test_function_roundtrip(self):
        """Test function definition roundtrip."""
        original = ast.parse("""
def greet(name: str) -> str:
    return f"Hello, {name}"
""")
        result = roundtrip_ast(original)

        func_def = next(s for s in result.body if isinstance(s, ast.FunctionDef))
        assert func_def.name == "greet"

    def test_imports_roundtrip(self):
        """Test imports roundtrip."""
        original = ast.parse("""
from typing import List, Optional
import os
""")
        result = roundtrip_ast(original)

        # Should have both import types
        has_from_import = any(isinstance(s, ast.ImportFrom) for s in result.body)
        has_import = any(isinstance(s, ast.Import) for s in result.body)

        assert has_from_import
        assert has_import


class TestAddBlankLines:
    """Tests for _add_blank_lines helper function."""

    def test_no_classes(self):
        """Test source without classes."""
        source = "x = 1\ny = 2"
        result = _add_blank_lines(source)

        # Should be mostly unchanged
        assert "x = 1" in result
        assert "y = 2" in result

    def test_single_class_at_start(self):
        """Test single class at start of file."""
        source = "class Foo:\n    pass"
        result = _add_blank_lines(source)

        assert "class Foo" in result

    def test_multiple_classes(self):
        """Test multiple classes get blank lines between."""
        source = "class Foo:\n    pass\nclass Bar:\n    pass"
        result = _add_blank_lines(source)

        # Should have blank lines between classes
        lines = result.split("\n")
        # Find class lines
        class_indices = [i for i, line in enumerate(lines) if line.strip().startswith("class ")]
        assert len(class_indices) == 2

        # Check there are blank lines before second class
        second_class_idx = class_indices[1]
        # At least one blank line before
        has_blank_before = any(
            not lines[i].strip() for i in range(class_indices[0] + 1, second_class_idx)
        )
        assert has_blank_before

    def test_decorator_not_separated_from_class(self):
        """Test decorator stays with class."""
        source = "@dataclass\nclass Foo:\n    pass"
        result = _add_blank_lines(source)

        lines = result.split("\n")
        # Find decorator and class lines
        for i, line in enumerate(lines):
            if line.strip() == "@dataclass":
                # Next non-empty line should be class
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines):
                    assert lines[j].strip().startswith("class ")


class TestFormatWithBlack:
    """Tests for _format_with_black helper function."""

    def test_returns_source_if_black_unavailable(self):
        """Test that source is returned unchanged if black unavailable."""
        source = "x=1"
        result = _format_with_black(source)

        # Either returns formatted (if black available) or original
        assert "x" in result
        assert "1" in result

    def test_invalid_syntax_returns_original(self):
        """Test that invalid syntax returns original source."""
        source = "def foo("
        result = _format_with_black(source)

        # Should return original since it can't be formatted
        assert result == source

    def test_line_length_parameter(self):
        """Test that line_length parameter is accepted."""
        source = "x = 1"
        result = _format_with_black(source, line_length=120)

        # Should work without error
        assert "x" in result


class TestEmitterEdgeCases:
    """Tests for edge cases in emitter."""

    def test_empty_module(self):
        """Test emitting empty module."""
        module = ast.Module(body=[], type_ignores=[])
        source = emit_python(module, format_with_black=False, add_header=False)

        # Should produce valid (empty or minimal) output
        assert source.endswith("\n")

    def test_complex_type_annotations(self):
        """Test emitting complex type annotations."""
        module = ast.parse("""
class Container:
    items: List[Dict[str, Optional[int]]]
    callback: Callable[[int], str]
""")
        source = emit_python(module, format_with_black=False, add_header=False)

        assert "List[Dict[str, Optional[int]]]" in source
        assert "Callable[[int], str]" in source

    def test_multiline_string(self):
        """Test emitting multiline string."""
        module = ast.parse('x = """multi\nline\nstring"""')
        source = emit_python(module, format_with_black=False, add_header=False)

        assert "multi" in source

    def test_nested_classes(self):
        """Test emitting nested classes."""
        module = ast.parse("""
class Outer:
    class Inner:
        value: int = 0
""")
        source = emit_python(module, format_with_black=False, add_header=False)

        assert "class Outer" in source
        assert "class Inner" in source

    def test_async_function(self):
        """Test emitting async function."""
        module = ast.parse("""
async def fetch_data():
    await something()
""")
        source = emit_python(module, format_with_black=False, add_header=False)

        assert "async def fetch_data" in source

    def test_generic_class(self):
        """Test emitting generic class."""
        module = ast.parse("""
class Stack(Generic[T]):
    items: List[T]
""")
        source = emit_python(module, format_with_black=False, add_header=False)

        assert "Generic[T]" in source


class TestRoundtripPreservation:
    """Tests to verify roundtrip preserves code semantics."""

    def test_default_values_preserved(self):
        """Test that default values are preserved in roundtrip."""
        original = ast.parse("""
class Config:
    debug: bool = False
    timeout: int = 30
    name: str = "default"
""")
        result = roundtrip_ast(original)
        source = ast.unparse(result)

        assert "False" in source
        assert "30" in source
        assert "default" in source

    def test_inheritance_preserved(self):
        """Test that inheritance is preserved in roundtrip."""
        original = ast.parse("""
class Child(Parent, Mixin):
    pass
""")
        result = roundtrip_ast(original)
        source = ast.unparse(result)

        assert "Parent" in source
        assert "Mixin" in source

    def test_method_signatures_preserved(self):
        """Test that method signatures are preserved."""
        original = ast.parse("""
class Calculator:
    def add(self, a: int, b: int = 0) -> int:
        return a + b
""")
        result = roundtrip_ast(original)
        source = ast.unparse(result)

        assert "a: int" in source
        # ast.unparse may format as "b: int=0" without space
        assert "b: int" in source and "= 0" in source or "b: int=0" in source
        assert "-> int" in source
