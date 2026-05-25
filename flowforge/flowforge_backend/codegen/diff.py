"""Diff Generation Module for FlowForge Backend.

Provides unified diff generation between original and modified source code
with structured output for UI rendering.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class DiffLineType(str, Enum):
    """Type of a diff line."""
    ADDED = "added"
    REMOVED = "removed"
    UNCHANGED = "unchanged"
    CONTEXT = "context"
    HEADER = "header"


@dataclass
class DiffLine:
    """Represents a single line in a diff.

    Attributes:
        type: The type of change (added, removed, unchanged, context, header)
        content: The actual line content
        original_line: Line number in the original file (None for added lines)
        modified_line: Line number in the modified file (None for removed lines)
    """
    type: DiffLineType
    content: str
    original_line: Optional[int] = None
    modified_line: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.type.value,
            "content": self.content,
            "originalLine": self.original_line,
            "modifiedLine": self.modified_line,
        }


@dataclass
class DiffHunk:
    """Represents a contiguous block of changes in a diff.

    Attributes:
        original_start: Starting line number in original file
        original_count: Number of lines from original file
        modified_start: Starting line number in modified file
        modified_count: Number of lines in modified file
        lines: List of DiffLine objects in this hunk
    """
    original_start: int
    original_count: int
    modified_start: int
    modified_count: int
    lines: List[DiffLine] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "originalStart": self.original_start,
            "originalCount": self.original_count,
            "modifiedStart": self.modified_start,
            "modifiedCount": self.modified_count,
            "lines": [line.to_dict() for line in self.lines],
        }


@dataclass
class DiffStats:
    """Statistics about the diff.

    Attributes:
        additions: Number of lines added
        deletions: Number of lines removed
        changes: Number of hunks (change blocks)
    """
    additions: int = 0
    deletions: int = 0
    changes: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "additions": self.additions,
            "deletions": self.deletions,
            "changes": self.changes,
        }


@dataclass
class DiffResult:
    """Complete diff result with structured data.

    Attributes:
        filename: Name of the file being diffed
        original_path: Path to original file (if any)
        has_changes: Whether there are any changes
        hunks: List of DiffHunk objects
        stats: DiffStats with summary statistics
        unified_diff: Raw unified diff string
    """
    filename: str
    original_path: Optional[str]
    has_changes: bool
    hunks: List[DiffHunk]
    stats: DiffStats
    unified_diff: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "filename": self.filename,
            "originalPath": self.original_path,
            "hasChanges": self.has_changes,
            "hunks": [hunk.to_dict() for hunk in self.hunks],
            "stats": self.stats.to_dict(),
            "unifiedDiff": self.unified_diff,
        }


def generate_diff(
    original: str,
    modified: str,
    filename: str = "",
    original_path: Optional[str] = None,
    context_lines: int = 3,
) -> DiffResult:
    """Generate unified diff between original and modified source.

    Args:
        original: Original source code string
        modified: Modified source code string
        filename: Name of the file for display purposes
        original_path: Path to the original file (optional)
        context_lines: Number of context lines around changes (default: 3)

    Returns:
        DiffResult with structured diff data
    """
    # Split into lines, preserving line endings for accurate diff
    original_lines = original.splitlines(keepends=True)
    modified_lines = modified.splitlines(keepends=True)

    # Handle files without trailing newlines
    if original_lines and not original_lines[-1].endswith('\n'):
        original_lines[-1] += '\n'
    if modified_lines and not modified_lines[-1].endswith('\n'):
        modified_lines[-1] += '\n'

    # Generate unified diff
    diff_generator = difflib.unified_diff(
        original_lines,
        modified_lines,
        fromfile=f"a/{filename}" if filename else "a/original",
        tofile=f"b/{filename}" if filename else "b/modified",
        n=context_lines,
    )

    unified_diff_lines = list(diff_generator)
    unified_diff = ''.join(unified_diff_lines)

    # Parse the diff into structured format
    hunks, stats = _parse_unified_diff(unified_diff_lines)

    has_changes = len(hunks) > 0

    return DiffResult(
        filename=filename,
        original_path=original_path,
        has_changes=has_changes,
        hunks=hunks,
        stats=stats,
        unified_diff=unified_diff,
    )


def _parse_unified_diff(diff_lines: List[str]) -> tuple[List[DiffHunk], DiffStats]:
    """Parse unified diff lines into structured hunks.

    Args:
        diff_lines: Lines from difflib.unified_diff

    Returns:
        Tuple of (list of DiffHunk, DiffStats)
    """
    hunks: List[DiffHunk] = []
    stats = DiffStats()

    current_hunk: Optional[DiffHunk] = None
    original_line = 0
    modified_line = 0

    for line in diff_lines:
        # Skip file headers (---, +++)
        if line.startswith('---') or line.startswith('+++'):
            continue

        # Parse hunk header: @@ -start,count +start,count @@
        if line.startswith('@@'):
            if current_hunk:
                hunks.append(current_hunk)
                stats.changes += 1

            # Parse the hunk header
            parts = line.split('@@')
            if len(parts) >= 2:
                ranges = parts[1].strip().split()

                # Parse original range (-start,count)
                orig_range = ranges[0][1:] if ranges else "1,0"
                if ',' in orig_range:
                    orig_start, orig_count = orig_range.split(',')
                else:
                    orig_start, orig_count = orig_range, "1"

                # Parse modified range (+start,count)
                mod_range = ranges[1][1:] if len(ranges) > 1 else "1,0"
                if ',' in mod_range:
                    mod_start, mod_count = mod_range.split(',')
                else:
                    mod_start, mod_count = mod_range, "1"

                current_hunk = DiffHunk(
                    original_start=int(orig_start),
                    original_count=int(orig_count),
                    modified_start=int(mod_start),
                    modified_count=int(mod_count),
                )
                original_line = int(orig_start)
                modified_line = int(mod_start)

                # Add header line
                current_hunk.lines.append(DiffLine(
                    type=DiffLineType.HEADER,
                    content=line.rstrip('\n'),
                    original_line=None,
                    modified_line=None,
                ))
            continue

        if current_hunk is None:
            continue

        # Parse content lines
        content = line[1:].rstrip('\n') if len(line) > 1 else ""

        if line.startswith('+'):
            # Added line
            current_hunk.lines.append(DiffLine(
                type=DiffLineType.ADDED,
                content=content,
                original_line=None,
                modified_line=modified_line,
            ))
            modified_line += 1
            stats.additions += 1

        elif line.startswith('-'):
            # Removed line
            current_hunk.lines.append(DiffLine(
                type=DiffLineType.REMOVED,
                content=content,
                original_line=original_line,
                modified_line=None,
            ))
            original_line += 1
            stats.deletions += 1

        elif line.startswith(' '):
            # Context/unchanged line
            current_hunk.lines.append(DiffLine(
                type=DiffLineType.CONTEXT,
                content=content,
                original_line=original_line,
                modified_line=modified_line,
            ))
            original_line += 1
            modified_line += 1

        elif line.startswith('\\'):
            # "\ No newline at end of file" - skip
            continue

    # Add the last hunk
    if current_hunk:
        hunks.append(current_hunk)
        stats.changes += 1

    return hunks, stats


def generate_side_by_side_diff(
    original: str,
    modified: str,
    filename: str = "",
) -> dict:
    """Generate a side-by-side diff view.

    Args:
        original: Original source code string
        modified: Modified source code string
        filename: Name of the file for display purposes

    Returns:
        Dictionary with side-by-side diff data
    """
    original_lines = original.splitlines()
    modified_lines = modified.splitlines()

    # Use SequenceMatcher for more granular comparison
    matcher = difflib.SequenceMatcher(None, original_lines, modified_lines)

    left_lines: List[dict] = []
    right_lines: List[dict] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for i, j in zip(range(i1, i2), range(j1, j2)):
                left_lines.append({
                    "lineNumber": i + 1,
                    "content": original_lines[i],
                    "type": "unchanged",
                })
                right_lines.append({
                    "lineNumber": j + 1,
                    "content": modified_lines[j],
                    "type": "unchanged",
                })

        elif tag == 'replace':
            # Handle replacement - show both sides
            max_lines = max(i2 - i1, j2 - j1)
            for k in range(max_lines):
                if i1 + k < i2:
                    left_lines.append({
                        "lineNumber": i1 + k + 1,
                        "content": original_lines[i1 + k],
                        "type": "removed",
                    })
                else:
                    left_lines.append({
                        "lineNumber": None,
                        "content": "",
                        "type": "empty",
                    })

                if j1 + k < j2:
                    right_lines.append({
                        "lineNumber": j1 + k + 1,
                        "content": modified_lines[j1 + k],
                        "type": "added",
                    })
                else:
                    right_lines.append({
                        "lineNumber": None,
                        "content": "",
                        "type": "empty",
                    })

        elif tag == 'delete':
            for i in range(i1, i2):
                left_lines.append({
                    "lineNumber": i + 1,
                    "content": original_lines[i],
                    "type": "removed",
                })
                right_lines.append({
                    "lineNumber": None,
                    "content": "",
                    "type": "empty",
                })

        elif tag == 'insert':
            for j in range(j1, j2):
                left_lines.append({
                    "lineNumber": None,
                    "content": "",
                    "type": "empty",
                })
                right_lines.append({
                    "lineNumber": j + 1,
                    "content": modified_lines[j],
                    "type": "added",
                })

    return {
        "filename": filename,
        "left": left_lines,
        "right": right_lines,
        "leftTitle": "Original",
        "rightTitle": "Modified",
    }
