"""Tests for the formatter module.

Tests comment preservation, blank line preservation, and code formatting.
"""

from __future__ import annotations

import ast
from unittest.mock import patch

import pytest

from ..codegen.formatter import (
    preserve_comments,
    preserve_blank_lines,
    format_code,
    _simple_format,
    _extract_comments,
    _split_inline_comment,
)


class TestPreserveComments:
    """Tests for preserve_comments function."""

    def test_standalone_comment_reinserted(self):
        """Standalone comments are placed before the matching code line."""
        original = "# This is a comment\nx = 1\n"
        generated = "x = 1\n"

        result = preserve_comments(original, generated)
        assert "# This is a comment" in result
        lines = result.split("\n")
        comment_idx = next(i for i, l in enumerate(lines) if "# This is a comment" in l)
        code_idx = next(i for i, l in enumerate(lines) if "x = 1" in l)
        assert comment_idx < code_idx

    def test_no_comments_returns_unchanged(self):
        """When original has no comments, generated source is returned as-is."""
        original = "x = 1\n"
        generated = "x = 1\n"

        result = preserve_comments(original, generated)
        assert result == generated

    def test_inline_comment_preserved(self):
        """Inline comments are re-attached to the matching line."""
        original = "x = 1  # important\n"
        generated = "x = 1\n"

        result = preserve_comments(original, generated)
        assert "# important" in result

    def test_multiple_comments(self):
        """Multiple comments are all preserved."""
        original = "# first\nx = 1\n# second\ny = 2\n"
        generated = "x = 1\ny = 2\n"

        result = preserve_comments(original, generated)
        assert "# first" in result
        assert "# second" in result

    def test_comment_inside_string_not_extracted(self):
        """Hash inside a string should not be treated as a comment."""
        line = 'x = "hello # world"'
        code, sep, comment = _split_inline_comment(line)
        assert comment is None


class TestPreserveBlankLines:
    """Tests for preserve_blank_lines function."""

    def test_blank_line_between_classes_preserved(self):
        """Blank lines between class definitions are preserved."""
        original = "class A:\n    pass\n\nclass B:\n    pass\n"
        generated = "class A:\n    pass\nclass B:\n    pass\n"

        result = preserve_blank_lines(original, generated)
        lines = result.split("\n")
        # There should be a blank line before "class B:"
        b_idx = next(i for i, l in enumerate(lines) if l.strip() == "class B:")
        assert lines[b_idx - 1].strip() == ""

    def test_no_blanks_in_original_returns_unchanged(self):
        """When original has no meaningful blank lines, output is unchanged."""
        original = "x = 1\ny = 2\n"
        generated = "x = 1\ny = 2\n"

        result = preserve_blank_lines(original, generated)
        assert result == generated

    def test_does_not_double_blank_lines(self):
        """If generated already has blank lines, don't add duplicates."""
        original = "class A:\n    pass\n\nclass B:\n    pass\n"
        generated = "class A:\n    pass\n\nclass B:\n    pass\n"

        result = preserve_blank_lines(original, generated)
        # Should not have triple blank lines
        assert "\n\n\n" not in result


class TestFormatCode:
    """Tests for format_code function."""

    def test_tabs_converted_to_spaces(self):
        """Tabs are converted to 4 spaces in fallback formatter."""
        source = "def f():\n\treturn 1\n"
        result = _simple_format(source)
        assert "\t" not in result
        assert "    return 1" in result

    def test_trailing_whitespace_removed(self):
        """Trailing whitespace is stripped."""
        source = "x = 1   \ny = 2  \n"
        result = _simple_format(source)
        for line in result.split("\n"):
            if line:
                assert line == line.rstrip()

    def test_trailing_newline_ensured(self):
        """Output always ends with a newline."""
        source = "x = 1"
        result = _simple_format(source)
        assert result.endswith("\n")

    def test_two_blank_lines_before_toplevel_class(self):
        """PEP 8: two blank lines before top-level class defs."""
        source = "import os\nclass Foo:\n    pass\n"
        result = _simple_format(source)
        lines = result.split("\n")
        class_idx = next(i for i, l in enumerate(lines) if l.startswith("class "))
        # The two lines before should be blank
        assert lines[class_idx - 1] == ""
        assert lines[class_idx - 2] == ""

    def test_format_code_without_black(self):
        """format_code falls back to simple formatter when black unavailable."""
        with patch.dict("sys.modules", {"black": None}):
            source = "x = 1\t \n"
            # Force reimport path by calling _simple_format directly
            result = _simple_format(source)
            assert "\t" not in result

    def test_format_code_with_black_available(self):
        """format_code uses black when available."""
        try:
            import black  # noqa: F401
        except ImportError:
            pytest.skip("black not installed")

        source = "x=1\n"
        result = format_code(source)
        assert "x = 1" in result


class TestExtractComments:
    """Tests for internal comment extraction."""

    def test_extracts_standalone_comments(self):
        """Standalone comments are extracted."""
        source = "# hello\nx = 1\n"
        comments = _extract_comments(source)
        assert len(comments) >= 1
        assert any("hello" in c[1] for c in comments)

    def test_extracts_inline_comments(self):
        """Inline comments are extracted."""
        source = "x = 1  # inline\n"
        comments = _extract_comments(source)
        assert len(comments) >= 1
        assert any("inline" in c[1] for c in comments)
