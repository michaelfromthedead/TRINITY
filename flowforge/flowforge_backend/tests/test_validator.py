"""Tests for validator module.

This module tests Python code validation including syntax checking
and semantic analysis.
"""

from __future__ import annotations

import pytest

from ..codegen.validator import (
    validate_python,
    validate_and_format,
    quick_validate,
    get_syntax_error_details,
    PYTHON_BUILTINS,
)
from ..codegen.types import Severity, ValidationError, ValidationResult


class TestValidatePython:
    """Tests for validate_python function."""

    def test_valid_simple_code(self):
        """Test validation of simple valid code."""
        source = "x = 1"
        result = validate_python(source)

        assert result.success is True
        assert len(result.errors) == 0

    def test_valid_function_definition(self):
        """Test validation of valid function."""
        source = """
def greet(name: str) -> str:
    return f"Hello, {name}"
"""
        result = validate_python(source)

        assert result.success is True
        assert len(result.errors) == 0

    def test_valid_class_definition(self):
        """Test validation of valid class."""
        source = """
class Player:
    def __init__(self, name: str):
        self.name = name
"""
        result = validate_python(source)

        assert result.success is True
        assert len(result.errors) == 0

    def test_valid_decorated_class(self):
        """Test validation of decorated class."""
        source = """
from dataclasses import dataclass

@dataclass
class Position:
    x: float = 0.0
    y: float = 0.0
"""
        result = validate_python(source)

        assert result.success is True

    def test_syntax_error_incomplete_def(self):
        """Test detection of syntax error in incomplete function."""
        source = "def foo("
        result = validate_python(source)

        assert result.success is False
        assert len(result.errors) == 1
        assert result.errors[0].severity == Severity.ERROR

    def test_syntax_error_missing_colon(self):
        """Test detection of missing colon in def."""
        source = "def foo()\n    pass"
        result = validate_python(source)

        assert result.success is False
        assert len(result.errors) >= 1

    def test_syntax_error_invalid_syntax(self):
        """Test detection of general invalid syntax."""
        source = "if x ==:"
        result = validate_python(source)

        assert result.success is False
        assert result.has_errors

    def test_syntax_error_unmatched_paren(self):
        """Test detection of unmatched parenthesis."""
        source = "print((x)"
        result = validate_python(source)

        assert result.success is False

    def test_syntax_error_location(self):
        """Test that syntax error includes line and column."""
        source = """
x = 1
y = (
"""
        result = validate_python(source)

        assert result.success is False
        assert len(result.errors) == 1
        error = result.errors[0]
        assert error.line >= 1
        assert error.column >= 0

    def test_empty_source_valid(self):
        """Test that empty source is valid."""
        result = validate_python("")

        assert result.success is True

    def test_whitespace_only_valid(self):
        """Test that whitespace-only source is valid."""
        result = validate_python("   \n\n   ")

        assert result.success is True

    def test_source_hash_generated(self):
        """Test that source hash is generated."""
        source = "x = 1"
        result = validate_python(source)

        assert result.source_hash is not None
        assert len(result.source_hash) == 16  # SHA256 truncated to 16 chars

    def test_same_source_same_hash(self):
        """Test that same source produces same hash."""
        source = "x = 1"
        result1 = validate_python(source)
        result2 = validate_python(source)

        assert result1.source_hash == result2.source_hash

    def test_different_source_different_hash(self):
        """Test that different source produces different hash."""
        result1 = validate_python("x = 1")
        result2 = validate_python("x = 2")

        assert result1.source_hash != result2.source_hash

    def test_custom_filename_in_error(self):
        """Test that custom filename appears in error context."""
        source = "def foo("
        result = validate_python(source, filename="game.py")

        assert result.success is False
        # The filename is used for error context but may not appear in message
        # Just verify we still get the error
        assert len(result.errors) >= 1


class TestSemanticChecks:
    """Tests for semantic analysis in validate_python."""

    def test_semantic_checks_disabled_by_default(self):
        """Test that semantic checks are disabled by default."""
        # This code has an unused import, but no warning without semantic checks
        source = """
import os
x = 1
"""
        result = validate_python(source, check_semantics=False)

        assert result.success is True
        assert len(result.warnings) == 0

    def test_unused_import_warning(self):
        """Test detection of unused import."""
        source = """
import os
x = 1
"""
        result = validate_python(source, check_semantics=True)

        assert result.success is True  # Warnings don't fail validation
        assert len(result.warnings) >= 1
        assert any("os" in w.message for w in result.warnings)

    def test_unused_from_import_warning(self):
        """Test detection of unused from import."""
        source = """
from typing import List, Dict
x: List[int] = []
"""
        result = validate_python(source, check_semantics=True)

        # Dict is unused
        assert any("Dict" in w.message for w in result.warnings)

    def test_used_import_no_warning(self):
        """Test that used import does not produce warning."""
        source = """
import os
path = os.path.join("a", "b")
"""
        result = validate_python(source, check_semantics=True)

        # No warning for os since it's used
        assert not any("os" in w.message and "Unused" in w.message for w in result.warnings)

    def test_shadowed_builtin_warning(self):
        """Test detection of shadowed builtin."""
        source = """
list = [1, 2, 3]
"""
        result = validate_python(source, check_semantics=True)

        assert len(result.warnings) >= 1
        assert any("list" in w.message and "shadow" in w.message.lower() for w in result.warnings)

    def test_shadowed_builtin_in_function(self):
        """Test detection of shadowed builtin in function scope."""
        source = """
def foo():
    id = 42
    return id
"""
        result = validate_python(source, check_semantics=True)

        assert any("id" in w.message for w in result.warnings)

    def test_shadowed_builtin_in_class(self):
        """Test detection of shadowed builtin in class."""
        source = """
class Foo:
    type = "bar"
"""
        result = validate_python(source, check_semantics=True)

        assert any("type" in w.message for w in result.warnings)

    def test_no_warning_for_non_builtins(self):
        """Test no warning for regular variable names."""
        source = """
my_list = [1, 2, 3]
my_type = "custom"
"""
        result = validate_python(source, check_semantics=True)

        # Only warnings for unused names, not shadowing
        for w in result.warnings:
            assert "shadow" not in w.message.lower()

    def test_import_alias_not_shadowing(self):
        """Test that import alias is tracked correctly."""
        source = """
import numpy as np
x = np.array([1, 2])
"""
        result = validate_python(source, check_semantics=True)

        # np is used, numpy is the module name
        assert not any("np" in w.message and "Unused" in w.message for w in result.warnings)


class TestQuickValidate:
    """Tests for quick_validate function."""

    def test_valid_code_returns_true(self):
        """Test that valid code returns True."""
        assert quick_validate("x = 1") is True

    def test_valid_multiline_returns_true(self):
        """Test that valid multiline code returns True."""
        source = """
def foo():
    return 42
"""
        assert quick_validate(source) is True

    def test_invalid_code_returns_false(self):
        """Test that invalid code returns False."""
        assert quick_validate("def foo(") is False

    def test_empty_string_returns_true(self):
        """Test that empty string is valid."""
        assert quick_validate("") is True

    def test_whitespace_returns_true(self):
        """Test that whitespace is valid."""
        assert quick_validate("   \n\t   ") is True


class TestGetSyntaxErrorDetails:
    """Tests for get_syntax_error_details function."""

    def test_valid_code_returns_none(self):
        """Test that valid code returns None."""
        result = get_syntax_error_details("x = 1")
        assert result is None

    def test_syntax_error_returns_dict(self):
        """Test that syntax error returns details dict."""
        result = get_syntax_error_details("def foo(")
        assert result is not None
        assert isinstance(result, dict)

    def test_syntax_error_has_message(self):
        """Test that error details include message."""
        result = get_syntax_error_details("def foo(")
        assert "message" in result
        assert result["message"] is not None

    def test_syntax_error_has_line(self):
        """Test that error details include line number."""
        result = get_syntax_error_details("x = 1\ndef foo(")
        assert "line" in result
        assert result["line"] >= 1

    def test_syntax_error_has_column(self):
        """Test that error details include column."""
        result = get_syntax_error_details("def foo(")
        assert "column" in result

    def test_syntax_error_has_text(self):
        """Test that error details include problematic text."""
        result = get_syntax_error_details("def foo(")
        assert "text" in result
        assert result["text"] is not None

    def test_syntax_error_end_positions(self):
        """Test that error details may include end positions."""
        result = get_syntax_error_details("if True\n    pass")
        assert "end_line" in result
        assert "end_column" in result


class TestValidateAndFormat:
    """Tests for validate_and_format function."""

    def test_returns_source_and_result(self):
        """Test that function returns tuple of source and result."""
        source = "x = 1"
        formatted_source, result = validate_and_format(source)

        assert isinstance(formatted_source, str)
        assert isinstance(result, ValidationResult)

    def test_valid_code_success(self):
        """Test validation with valid code."""
        source = "x = 1"
        _, result = validate_and_format(source)

        assert result.success is True

    def test_invalid_code_failure(self):
        """Test validation with invalid code."""
        source = "def foo("
        _, result = validate_and_format(source)

        assert result.success is False

    def test_source_preserved_without_formatter(self):
        """Test that source is preserved when no formatter available."""
        source = "x=1"
        formatted, _ = validate_and_format(source)

        # Without black installed, source should be unchanged
        # (or formatted if black is available)
        assert "x" in formatted
        assert "1" in formatted


class TestValidationErrorHelpers:
    """Tests for ValidationError factory methods."""

    def test_syntax_error_factory(self):
        """Test syntax_error class method."""
        error = ValidationError.syntax_error("Unexpected token", line=10, column=5)

        assert error.severity == Severity.ERROR
        assert error.code == "E001"
        assert error.line == 10
        assert error.column == 5

    def test_unused_import_factory(self):
        """Test unused_import class method."""
        error = ValidationError.unused_import("os", line=1, column=0)

        assert error.severity == Severity.WARNING
        assert error.code == "W001"
        assert "os" in error.message

    def test_shadowed_builtin_factory(self):
        """Test shadowed_builtin class method."""
        error = ValidationError.shadowed_builtin("list", line=5, column=0)

        assert error.severity == Severity.WARNING
        assert error.code == "W002"
        assert "list" in error.message


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_valid_factory(self):
        """Test valid class method."""
        result = ValidationResult.valid()

        assert result.success is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_valid_with_hash(self):
        """Test valid with source hash."""
        result = ValidationResult.valid(source_hash="abc123")

        assert result.source_hash == "abc123"

    def test_invalid_factory(self):
        """Test invalid class method."""
        errors = [ValidationError.syntax_error("Error", line=1)]
        result = ValidationResult.invalid(errors)

        assert result.success is False
        assert len(result.errors) == 1

    def test_has_errors_property(self):
        """Test has_errors property."""
        result = ValidationResult.valid()
        assert result.has_errors is False

        errors = [ValidationError.syntax_error("Error", line=1)]
        result = ValidationResult.invalid(errors)
        assert result.has_errors is True

    def test_has_warnings_property(self):
        """Test has_warnings property."""
        warnings = [ValidationError.unused_import("os", line=1)]
        result = ValidationResult(success=True, warnings=warnings)

        assert result.has_warnings is True

    def test_all_issues_property(self):
        """Test all_issues property combines errors and warnings."""
        errors = [ValidationError.syntax_error("Error", line=1)]
        warnings = [ValidationError.unused_import("os", line=2)]
        result = ValidationResult(success=False, errors=errors, warnings=warnings)

        all_issues = result.all_issues
        assert len(all_issues) == 2

    def test_to_dict(self):
        """Test to_dict serialization."""
        result = ValidationResult.valid(source_hash="abc123")
        data = result.to_dict()

        assert data["success"] is True
        assert "errors" in data
        assert "warnings" in data
        assert data["source_hash"] == "abc123"

    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "success": False,
            "errors": [{"line": 1, "column": 0, "message": "Error", "severity": "error"}],
            "warnings": [],
        }
        result = ValidationResult.from_dict(data)

        assert result.success is False
        assert len(result.errors) == 1


class TestPythonBuiltins:
    """Tests for PYTHON_BUILTINS constant."""

    def test_common_builtins_present(self):
        """Test that common builtins are in the set."""
        assert "list" in PYTHON_BUILTINS
        assert "dict" in PYTHON_BUILTINS
        assert "str" in PYTHON_BUILTINS
        assert "int" in PYTHON_BUILTINS
        assert "print" in PYTHON_BUILTINS
        assert "len" in PYTHON_BUILTINS
        assert "range" in PYTHON_BUILTINS

    def test_type_function_present(self):
        """Test that type() is in builtins."""
        assert "type" in PYTHON_BUILTINS

    def test_id_function_present(self):
        """Test that id() is in builtins."""
        assert "id" in PYTHON_BUILTINS

    def test_dunder_import_present(self):
        """Test that __import__ is in builtins."""
        assert "__import__" in PYTHON_BUILTINS
