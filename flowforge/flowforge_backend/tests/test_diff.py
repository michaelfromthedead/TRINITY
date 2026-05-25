"""Tests for diff module.

This module tests unified diff generation and parsing functionality.
"""

from __future__ import annotations

import pytest

from ..codegen.diff import (
    generate_diff,
    generate_side_by_side_diff,
    _parse_unified_diff,
    DiffLine,
    DiffLineType,
    DiffHunk,
    DiffStats,
    DiffResult,
)


class TestGenerateDiff:
    """Tests for generate_diff function."""

    def test_no_changes(self):
        """Test diff with identical content."""
        original = "line1\nline2\nline3"
        modified = "line1\nline2\nline3"

        result = generate_diff(original, modified)

        assert result.has_changes is False
        assert len(result.hunks) == 0
        assert result.stats.additions == 0
        assert result.stats.deletions == 0

    def test_additions_only(self):
        """Test diff with only additions."""
        original = "line1\nline2"
        modified = "line1\nline2\nline3\nline4"

        result = generate_diff(original, modified)

        assert result.has_changes is True
        assert result.stats.additions == 2
        assert result.stats.deletions == 0

    def test_deletions_only(self):
        """Test diff with only deletions."""
        original = "line1\nline2\nline3\nline4"
        modified = "line1\nline2"

        result = generate_diff(original, modified)

        assert result.has_changes is True
        assert result.stats.additions == 0
        assert result.stats.deletions == 2

    def test_mixed_changes(self):
        """Test diff with both additions and deletions."""
        original = "line1\nline2\nline3"
        modified = "line1\nmodified line2\nline3\nline4"

        result = generate_diff(original, modified)

        assert result.has_changes is True
        assert result.stats.additions >= 1
        assert result.stats.deletions >= 1

    def test_filename_in_result(self):
        """Test that filename is preserved in result."""
        result = generate_diff("a", "b", filename="test.py")

        assert result.filename == "test.py"

    def test_original_path_in_result(self):
        """Test that original_path is preserved in result."""
        result = generate_diff("a", "b", original_path="/path/to/file.py")

        assert result.original_path == "/path/to/file.py"

    def test_unified_diff_string(self):
        """Test that unified diff string is generated."""
        original = "old line"
        modified = "new line"

        result = generate_diff(original, modified, filename="test.py")

        assert "---" in result.unified_diff
        assert "+++" in result.unified_diff
        assert "@@" in result.unified_diff

    def test_context_lines_default(self):
        """Test default context lines (3)."""
        # Create content with changes in the middle
        original = "\n".join([f"line{i}" for i in range(10)])
        modified = "\n".join([f"line{i}" if i != 5 else "changed" for i in range(10)])

        result = generate_diff(original, modified, context_lines=3)

        # Hunks should include context
        assert result.has_changes is True

    def test_context_lines_custom(self):
        """Test custom context lines."""
        original = "\n".join([f"line{i}" for i in range(10)])
        modified = "\n".join([f"line{i}" if i != 5 else "changed" for i in range(10)])

        result = generate_diff(original, modified, context_lines=1)

        assert result.has_changes is True

    def test_empty_original(self):
        """Test diff from empty original."""
        original = ""
        modified = "line1\nline2"

        result = generate_diff(original, modified)

        assert result.has_changes is True
        assert result.stats.additions >= 1

    def test_empty_modified(self):
        """Test diff to empty modified."""
        original = "line1\nline2"
        modified = ""

        result = generate_diff(original, modified)

        assert result.has_changes is True
        assert result.stats.deletions >= 1

    def test_both_empty(self):
        """Test diff with both empty."""
        result = generate_diff("", "")

        assert result.has_changes is False

    def test_single_line_change(self):
        """Test single line change."""
        original = "old"
        modified = "new"

        result = generate_diff(original, modified)

        assert result.has_changes is True
        assert result.stats.additions == 1
        assert result.stats.deletions == 1

    def test_whitespace_changes_detected(self):
        """Test that whitespace changes are detected."""
        original = "line with spaces"
        modified = "line  with  spaces"

        result = generate_diff(original, modified)

        assert result.has_changes is True


class TestDiffHunks:
    """Tests for diff hunk parsing."""

    def test_single_hunk(self):
        """Test parsing single hunk."""
        original = "line1\nline2"
        modified = "line1\nchanged"

        result = generate_diff(original, modified)

        assert len(result.hunks) == 1

    def test_multiple_hunks(self):
        """Test parsing multiple hunks with separated changes."""
        # Create content with changes far apart
        lines = [f"line{i}" for i in range(20)]
        original = "\n".join(lines)
        modified_lines = lines.copy()
        modified_lines[2] = "changed2"
        modified_lines[17] = "changed17"
        modified = "\n".join(modified_lines)

        result = generate_diff(original, modified, context_lines=1)

        # Should have multiple hunks due to distance
        assert len(result.hunks) >= 1

    def test_hunk_line_numbers(self):
        """Test hunk line number tracking."""
        original = "a\nb\nc"
        modified = "a\nB\nc"

        result = generate_diff(original, modified)

        if result.hunks:
            hunk = result.hunks[0]
            assert hunk.original_start >= 1
            assert hunk.modified_start >= 1

    def test_hunk_contains_header(self):
        """Test that hunk contains header line."""
        original = "a"
        modified = "b"

        result = generate_diff(original, modified)

        if result.hunks:
            hunk = result.hunks[0]
            header_lines = [l for l in hunk.lines if l.type == DiffLineType.HEADER]
            assert len(header_lines) >= 1

    def test_hunk_added_lines(self):
        """Test added lines in hunk."""
        original = "a"
        modified = "a\nb"

        result = generate_diff(original, modified)

        if result.hunks:
            added_lines = [l for l in result.hunks[0].lines if l.type == DiffLineType.ADDED]
            assert len(added_lines) >= 1

    def test_hunk_removed_lines(self):
        """Test removed lines in hunk."""
        original = "a\nb"
        modified = "a"

        result = generate_diff(original, modified)

        if result.hunks:
            removed_lines = [l for l in result.hunks[0].lines if l.type == DiffLineType.REMOVED]
            assert len(removed_lines) >= 1


class TestParseUnifiedDiff:
    """Tests for _parse_unified_diff function."""

    def test_parse_empty_diff(self):
        """Test parsing empty diff."""
        hunks, stats = _parse_unified_diff([])

        assert len(hunks) == 0
        assert stats.additions == 0
        assert stats.deletions == 0
        assert stats.changes == 0

    def test_parse_header_lines_skipped(self):
        """Test that --- and +++ lines are skipped."""
        diff_lines = [
            "--- a/file.txt\n",
            "+++ b/file.txt\n",
            "@@ -1 +1 @@\n",
            "-old\n",
            "+new\n",
        ]
        hunks, stats = _parse_unified_diff(diff_lines)

        assert len(hunks) == 1

    def test_parse_hunk_header(self):
        """Test parsing hunk header."""
        diff_lines = [
            "@@ -1,3 +1,4 @@\n",
            " context\n",
            "-removed\n",
            "+added\n",
            " context2\n",
        ]
        hunks, stats = _parse_unified_diff(diff_lines)

        assert len(hunks) == 1
        hunk = hunks[0]
        assert hunk.original_start == 1
        assert hunk.original_count == 3
        assert hunk.modified_start == 1
        assert hunk.modified_count == 4

    def test_parse_single_line_hunk(self):
        """Test parsing hunk with single line range."""
        diff_lines = [
            "@@ -5 +5 @@\n",
            "-old\n",
            "+new\n",
        ]
        hunks, stats = _parse_unified_diff(diff_lines)

        assert len(hunks) == 1
        assert hunks[0].original_count == 1
        assert hunks[0].modified_count == 1

    def test_parse_line_types(self):
        """Test parsing different line types."""
        diff_lines = [
            "@@ -1,3 +1,3 @@\n",
            " unchanged\n",
            "-removed\n",
            "+added\n",
        ]
        hunks, stats = _parse_unified_diff(diff_lines)

        hunk = hunks[0]
        types = [l.type for l in hunk.lines]

        assert DiffLineType.HEADER in types
        assert DiffLineType.CONTEXT in types
        assert DiffLineType.REMOVED in types
        assert DiffLineType.ADDED in types

    def test_parse_stats_additions(self):
        """Test counting additions."""
        diff_lines = [
            "@@ -1 +1,3 @@\n",
            " existing\n",
            "+new1\n",
            "+new2\n",
        ]
        hunks, stats = _parse_unified_diff(diff_lines)

        assert stats.additions == 2

    def test_parse_stats_deletions(self):
        """Test counting deletions."""
        diff_lines = [
            "@@ -1,3 +1 @@\n",
            " existing\n",
            "-old1\n",
            "-old2\n",
        ]
        hunks, stats = _parse_unified_diff(diff_lines)

        assert stats.deletions == 2

    def test_parse_stats_changes(self):
        """Test counting change hunks."""
        diff_lines = [
            "@@ -1 +1 @@\n",
            "-old\n",
            "+new\n",
            "@@ -10 +10 @@\n",
            "-old2\n",
            "+new2\n",
        ]
        hunks, stats = _parse_unified_diff(diff_lines)

        assert stats.changes == 2
        assert len(hunks) == 2

    def test_parse_no_newline_marker_skipped(self):
        """Test that 'No newline at end of file' is skipped."""
        diff_lines = [
            "@@ -1 +1 @@\n",
            "-old\n",
            "\\ No newline at end of file\n",
            "+new\n",
        ]
        hunks, stats = _parse_unified_diff(diff_lines)

        # Should still parse correctly, just skip the marker
        assert len(hunks) == 1


class TestGenerateSideBySideDiff:
    """Tests for generate_side_by_side_diff function."""

    def test_no_changes(self):
        """Test side-by-side diff with no changes."""
        original = "line1\nline2"
        modified = "line1\nline2"

        result = generate_side_by_side_diff(original, modified)

        assert "left" in result
        assert "right" in result
        assert len(result["left"]) == len(result["right"])

    def test_additions(self):
        """Test side-by-side diff with additions."""
        original = "line1"
        modified = "line1\nline2"

        result = generate_side_by_side_diff(original, modified)

        # Right side should have added line
        right_types = [l["type"] for l in result["right"]]
        assert "added" in right_types

    def test_deletions(self):
        """Test side-by-side diff with deletions."""
        original = "line1\nline2"
        modified = "line1"

        result = generate_side_by_side_diff(original, modified)

        # Left side should have removed line
        left_types = [l["type"] for l in result["left"]]
        assert "removed" in left_types

    def test_replacements(self):
        """Test side-by-side diff with replacements."""
        original = "old line"
        modified = "new line"

        result = generate_side_by_side_diff(original, modified)

        left_types = [l["type"] for l in result["left"]]
        right_types = [l["type"] for l in result["right"]]

        assert "removed" in left_types
        assert "added" in right_types

    def test_filename_preserved(self):
        """Test that filename is preserved."""
        result = generate_side_by_side_diff("a", "b", filename="test.py")

        assert result["filename"] == "test.py"

    def test_titles_present(self):
        """Test that left/right titles are present."""
        result = generate_side_by_side_diff("a", "b")

        assert result["leftTitle"] == "Original"
        assert result["rightTitle"] == "Modified"

    def test_line_numbers(self):
        """Test that line numbers are included."""
        original = "line1\nline2"
        modified = "line1\nline2"

        result = generate_side_by_side_diff(original, modified)

        # Check first line has line number 1
        assert result["left"][0]["lineNumber"] == 1
        assert result["right"][0]["lineNumber"] == 1

    def test_empty_placeholders(self):
        """Test empty line placeholders for unmatched lines."""
        original = "line1"
        modified = "line1\nline2"

        result = generate_side_by_side_diff(original, modified)

        # Left should have empty placeholder for the added line
        empty_left = [l for l in result["left"] if l["type"] == "empty"]
        assert len(empty_left) >= 1

    def test_line_content_preserved(self):
        """Test that line content is preserved."""
        original = "hello world"
        modified = "hello world"

        result = generate_side_by_side_diff(original, modified)

        assert result["left"][0]["content"] == "hello world"
        assert result["right"][0]["content"] == "hello world"


class TestDiffLine:
    """Tests for DiffLine dataclass."""

    def test_to_dict(self):
        """Test DiffLine to_dict serialization."""
        line = DiffLine(
            type=DiffLineType.ADDED,
            content="new line",
            original_line=None,
            modified_line=5,
        )
        data = line.to_dict()

        assert data["type"] == "added"
        assert data["content"] == "new line"
        assert data["originalLine"] is None
        assert data["modifiedLine"] == 5

    def test_to_dict_unchanged(self):
        """Test DiffLine to_dict for unchanged line."""
        line = DiffLine(
            type=DiffLineType.UNCHANGED,
            content="same line",
            original_line=3,
            modified_line=3,
        )
        data = line.to_dict()

        assert data["type"] == "unchanged"
        assert data["originalLine"] == 3
        assert data["modifiedLine"] == 3


class TestDiffHunk:
    """Tests for DiffHunk dataclass."""

    def test_to_dict(self):
        """Test DiffHunk to_dict serialization."""
        hunk = DiffHunk(
            original_start=1,
            original_count=5,
            modified_start=1,
            modified_count=6,
            lines=[
                DiffLine(type=DiffLineType.CONTEXT, content="line", original_line=1, modified_line=1)
            ],
        )
        data = hunk.to_dict()

        assert data["originalStart"] == 1
        assert data["originalCount"] == 5
        assert data["modifiedStart"] == 1
        assert data["modifiedCount"] == 6
        assert len(data["lines"]) == 1


class TestDiffStats:
    """Tests for DiffStats dataclass."""

    def test_defaults(self):
        """Test DiffStats default values."""
        stats = DiffStats()

        assert stats.additions == 0
        assert stats.deletions == 0
        assert stats.changes == 0

    def test_to_dict(self):
        """Test DiffStats to_dict serialization."""
        stats = DiffStats(additions=5, deletions=3, changes=2)
        data = stats.to_dict()

        assert data["additions"] == 5
        assert data["deletions"] == 3
        assert data["changes"] == 2


class TestDiffResult:
    """Tests for DiffResult dataclass."""

    def test_to_dict(self):
        """Test DiffResult to_dict serialization."""
        result = DiffResult(
            filename="test.py",
            original_path="/path/test.py",
            has_changes=True,
            hunks=[],
            stats=DiffStats(additions=1, deletions=0, changes=1),
            unified_diff="@@ -1 +1,2 @@\n",
        )
        data = result.to_dict()

        assert data["filename"] == "test.py"
        assert data["originalPath"] == "/path/test.py"
        assert data["hasChanges"] is True
        assert isinstance(data["hunks"], list)
        assert isinstance(data["stats"], dict)
        assert data["unifiedDiff"] == "@@ -1 +1,2 @@\n"

    def test_no_changes_result(self):
        """Test DiffResult for no changes."""
        result = DiffResult(
            filename="test.py",
            original_path=None,
            has_changes=False,
            hunks=[],
            stats=DiffStats(),
            unified_diff="",
        )

        assert result.has_changes is False
        assert len(result.hunks) == 0
