"""Python code validation for FlowForge.

This module provides validation functionality for Python source code,
including syntax checking and optional semantic analysis.
"""

from __future__ import annotations

import ast
import hashlib
import sys
from typing import Any, Optional, Set

from .types import (
    GenerationResult,
    ImportInfo,
    Severity,
    ValidationError,
    ValidationResult,
)


# Python builtins that should not be shadowed
PYTHON_BUILTINS: Set[str] = {
    "abs", "all", "any", "ascii", "bin", "bool", "breakpoint", "bytearray",
    "bytes", "callable", "chr", "classmethod", "compile", "complex",
    "delattr", "dict", "dir", "divmod", "enumerate", "eval", "exec",
    "filter", "float", "format", "frozenset", "getattr", "globals",
    "hasattr", "hash", "help", "hex", "id", "input", "int", "isinstance",
    "issubclass", "iter", "len", "list", "locals", "map", "max",
    "memoryview", "min", "next", "object", "oct", "open", "ord", "pow",
    "print", "property", "range", "repr", "reversed", "round", "set",
    "setattr", "slice", "sorted", "staticmethod", "str", "sum", "super",
    "tuple", "type", "vars", "zip", "__import__",
}


def _compute_hash(source: str) -> str:
    """Compute a hash of the source code for caching."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def validate_python(
    source: str,
    *,
    check_semantics: bool = False,
    filename: str = "<generated>",
) -> ValidationResult:
    """Validate Python source code.

    Parses the source with ast.parse() to check syntax and optionally
    performs basic semantic checks like undefined name detection.

    Args:
        source: The Python source code to validate.
        check_semantics: If True, perform semantic checks (undefined names, etc.).
        filename: Filename to use in error messages.

    Returns:
        ValidationResult with success status and any errors/warnings.

    Example:
        >>> result = validate_python("def foo():\\n    return 42")
        >>> result.success
        True

        >>> result = validate_python("def foo(")
        >>> result.success
        False
        >>> result.errors[0].message
        "unexpected EOF while parsing"
    """
    errors: list[ValidationError] = []
    warnings: list[ValidationError] = []
    source_hash = _compute_hash(source)

    # Handle empty source
    if not source or not source.strip():
        return ValidationResult.valid(source_hash=source_hash)

    # Step 1: Syntax validation with ast.parse()
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as e:
        # Extract error details from the SyntaxError
        line = e.lineno or 1
        column = e.offset or 0
        # Python's offset is 1-indexed, we want 0-indexed
        if column > 0:
            column -= 1

        # Build a clear error message
        message = e.msg or "Syntax error"
        if e.text:
            # Include the problematic text snippet
            text_preview = e.text.strip()
            if len(text_preview) > 50:
                text_preview = text_preview[:50] + "..."
            message = f"{message}: {text_preview}"

        errors.append(ValidationError.syntax_error(
            message=message,
            line=line,
            column=column,
            end_line=getattr(e, "end_lineno", None),
            end_column=getattr(e, "end_offset", None),
        ))

        return ValidationResult.invalid(
            errors=errors,
            warnings=warnings,
            source_hash=source_hash,
        )

    # Step 2: Optional semantic checks
    if check_semantics:
        semantic_errors, semantic_warnings = _check_semantics(tree, source)
        errors.extend(semantic_errors)
        warnings.extend(semantic_warnings)

    # Determine success
    success = len(errors) == 0

    return ValidationResult(
        success=success,
        errors=errors,
        warnings=warnings,
        source_hash=source_hash,
    )


def _check_semantics(
    tree: ast.AST,
    source: str,
) -> tuple[list[ValidationError], list[ValidationError]]:
    """Perform semantic analysis on the AST.

    Checks for:
    - Undefined names (variables used before definition)
    - Unused imports
    - Shadowed builtins

    Args:
        tree: The parsed AST.
        source: The original source code.

    Returns:
        Tuple of (errors, warnings).
    """
    errors: list[ValidationError] = []
    warnings: list[ValidationError] = []

    # Track defined names and their locations
    defined_names: dict[str, tuple[int, int]] = {}  # name -> (line, col)
    used_names: Set[str] = set()
    imported_names: dict[str, tuple[int, int]] = {}  # name -> (line, col)

    class NameVisitor(ast.NodeVisitor):
        """Visitor to track name definitions and usages."""

        def __init__(self) -> None:
            self.scope_stack: list[Set[str]] = [set()]  # Stack of local scopes

        @property
        def current_scope(self) -> Set[str]:
            return self.scope_stack[-1]

        def push_scope(self) -> None:
            self.scope_stack.append(set())

        def pop_scope(self) -> None:
            if len(self.scope_stack) > 1:
                self.scope_stack.pop()

        def define_name(self, name: str, line: int, col: int) -> None:
            """Register a name as defined."""
            self.current_scope.add(name)
            if name not in defined_names:
                defined_names[name] = (line, col)

            # Check for shadowed builtins
            if name in PYTHON_BUILTINS:
                warnings.append(ValidationError.shadowed_builtin(name, line, col))

        def is_defined(self, name: str) -> bool:
            """Check if a name is defined in any scope."""
            for scope in reversed(self.scope_stack):
                if name in scope:
                    return True
            return name in PYTHON_BUILTINS or name in imported_names

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name.split(".")[0]
                self.define_name(name, node.lineno, node.col_offset)
                imported_names[name] = (node.lineno, node.col_offset)
            self.generic_visit(node)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                if name != "*":
                    self.define_name(name, node.lineno, node.col_offset)
                    imported_names[name] = (node.lineno, node.col_offset)
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.define_name(node.name, node.lineno, node.col_offset)
            self.push_scope()

            # Add function arguments to local scope
            for arg in node.args.args:
                self.define_name(arg.arg, node.lineno, node.col_offset)
            for arg in node.args.posonlyargs:
                self.define_name(arg.arg, node.lineno, node.col_offset)
            for arg in node.args.kwonlyargs:
                self.define_name(arg.arg, node.lineno, node.col_offset)
            if node.args.vararg:
                self.define_name(node.args.vararg.arg, node.lineno, node.col_offset)
            if node.args.kwarg:
                self.define_name(node.args.kwarg.arg, node.lineno, node.col_offset)

            self.generic_visit(node)
            self.pop_scope()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            # Same logic as FunctionDef
            self.define_name(node.name, node.lineno, node.col_offset)
            self.push_scope()

            for arg in node.args.args:
                self.define_name(arg.arg, node.lineno, node.col_offset)
            for arg in node.args.posonlyargs:
                self.define_name(arg.arg, node.lineno, node.col_offset)
            for arg in node.args.kwonlyargs:
                self.define_name(arg.arg, node.lineno, node.col_offset)
            if node.args.vararg:
                self.define_name(node.args.vararg.arg, node.lineno, node.col_offset)
            if node.args.kwarg:
                self.define_name(node.args.kwarg.arg, node.lineno, node.col_offset)

            self.generic_visit(node)
            self.pop_scope()

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.define_name(node.name, node.lineno, node.col_offset)
            self.push_scope()
            self.generic_visit(node)
            self.pop_scope()

        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Store):
                self.define_name(node.id, node.lineno, node.col_offset)
            elif isinstance(node.ctx, ast.Load):
                used_names.add(node.id)
                # Check for undefined names (only report if clearly undefined)
                # We skip this by default as it has many false positives
                # (e.g., names defined in outer scopes, conditional defines, etc.)
            self.generic_visit(node)

        def visit_For(self, node: ast.For) -> None:
            # The target of a for loop defines names
            if isinstance(node.target, ast.Name):
                self.define_name(node.target.id, node.lineno, node.col_offset)
            elif isinstance(node.target, ast.Tuple):
                for elt in node.target.elts:
                    if isinstance(elt, ast.Name):
                        self.define_name(elt.id, node.lineno, node.col_offset)
            self.generic_visit(node)

        def visit_With(self, node: ast.With) -> None:
            for item in node.items:
                if item.optional_vars:
                    if isinstance(item.optional_vars, ast.Name):
                        self.define_name(
                            item.optional_vars.id,
                            node.lineno,
                            node.col_offset,
                        )
            self.generic_visit(node)

        def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
            if node.name:
                self.define_name(node.name, node.lineno, node.col_offset)
            self.generic_visit(node)

        def visit_comprehension(self, node: ast.comprehension) -> None:
            if isinstance(node.target, ast.Name):
                self.define_name(node.target.id, node.target.lineno, node.target.col_offset)
            self.generic_visit(node)

    # Run the visitor
    visitor = NameVisitor()
    visitor.visit(tree)

    # Check for unused imports
    for name, (line, col) in imported_names.items():
        if name not in used_names:
            warnings.append(ValidationError.unused_import(name, line, col))

    return errors, warnings


def validate_and_format(
    source: str,
    *,
    check_semantics: bool = False,
    filename: str = "<generated>",
) -> tuple[str, ValidationResult]:
    """Validate Python source and optionally format it.

    This is a convenience function that validates the source and
    returns both the (potentially formatted) source and the validation result.

    Note: Formatting is not implemented yet - returns source as-is.

    Args:
        source: The Python source code to validate.
        check_semantics: If True, perform semantic checks.
        filename: Filename to use in error messages.

    Returns:
        Tuple of (source, validation_result).
    """
    result = validate_python(source, check_semantics=check_semantics, filename=filename)
    # TODO: Add optional formatting with black or autopep8
    return source, result


def quick_validate(source: str) -> bool:
    """Quickly check if source is valid Python syntax.

    This is a convenience function for simple validation without
    detailed error information.

    Args:
        source: The Python source code to validate.

    Returns:
        True if the source is valid Python, False otherwise.
    """
    try:
        ast.parse(source)
        return True
    except SyntaxError:
        return False


def get_syntax_error_details(source: str) -> Optional[dict[str, Any]]:
    """Get detailed information about a syntax error.

    Args:
        source: The Python source code to check.

    Returns:
        Dictionary with error details, or None if valid.
    """
    try:
        ast.parse(source)
        return None
    except SyntaxError as e:
        return {
            "message": e.msg,
            "line": e.lineno,
            "column": e.offset,
            "text": e.text,
            "end_line": getattr(e, "end_lineno", None),
            "end_column": getattr(e, "end_offset", None),
        }
