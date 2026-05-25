"""File operations for VCS integration.

Provides file status tracking, revert operations, and diff viewing
with support for multiple VCS providers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import difflib
import os
import re
import time

from .vcs_integration import (
    VCSProvider,
    VCSType,
    FileStatus,
    VCSError,
)


@dataclass
class FileStatusInfo:
    """Detailed status information for a file."""
    path: str
    status: FileStatus
    staged: bool = False
    working_tree_status: Optional[FileStatus] = None
    staged_status: Optional[FileStatus] = None
    original_path: Optional[str] = None  # For renames
    size: int = 0
    modified_time: float = 0.0
    permissions: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_binary(self) -> bool:
        """Check if the file is likely binary."""
        binary_extensions = {
            ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".tiff",
            ".mp3", ".wav", ".ogg", ".flac", ".aac",
            ".mp4", ".avi", ".mov", ".mkv", ".webm",
            ".zip", ".tar", ".gz", ".7z", ".rar",
            ".exe", ".dll", ".so", ".dylib",
            ".pdf", ".doc", ".docx", ".xls", ".xlsx",
            ".fbx", ".obj", ".blend", ".max", ".mb",
            ".psd", ".ai", ".svg",
            ".ttf", ".otf", ".woff", ".woff2",  # Fonts
        }
        ext = os.path.splitext(self.path)[1].lower()
        return ext in binary_extensions


class DiffLineType(Enum):
    """Type of line in a diff."""
    CONTEXT = auto()
    ADDITION = auto()
    DELETION = auto()
    HEADER = auto()
    HUNK_HEADER = auto()


@dataclass
class DiffLine:
    """A single line in a diff."""
    line_type: DiffLineType
    content: str
    old_line_number: Optional[int] = None
    new_line_number: Optional[int] = None

    def format(self, show_line_numbers: bool = True) -> str:
        """Format the line for display."""
        prefix = {
            DiffLineType.CONTEXT: " ",
            DiffLineType.ADDITION: "+",
            DiffLineType.DELETION: "-",
            DiffLineType.HEADER: "",
            DiffLineType.HUNK_HEADER: "@",
        }.get(self.line_type, " ")

        if show_line_numbers:
            old = str(self.old_line_number) if self.old_line_number else ""
            new = str(self.new_line_number) if self.new_line_number else ""
            return f"{old:>4} {new:>4} {prefix}{self.content}"

        return f"{prefix}{self.content}"


@dataclass
class DiffHunk:
    """A hunk (section) of changes in a diff."""
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    header: str = ""
    lines: List[DiffLine] = field(default_factory=list)

    @property
    def additions(self) -> int:
        """Count of added lines."""
        return sum(1 for line in self.lines if line.line_type == DiffLineType.ADDITION)

    @property
    def deletions(self) -> int:
        """Count of deleted lines."""
        return sum(1 for line in self.lines if line.line_type == DiffLineType.DELETION)


@dataclass
class FileDiff:
    """Diff information for a single file."""
    old_path: Optional[str]
    new_path: Optional[str]
    status: FileStatus
    hunks: List[DiffHunk] = field(default_factory=list)
    is_binary: bool = False
    old_mode: str = ""
    new_mode: str = ""
    similarity: int = 100  # For renames

    @property
    def path(self) -> str:
        """Get the effective path."""
        return self.new_path or self.old_path or ""

    @property
    def additions(self) -> int:
        """Total additions in all hunks."""
        return sum(h.additions for h in self.hunks)

    @property
    def deletions(self) -> int:
        """Total deletions in all hunks."""
        return sum(h.deletions for h in self.hunks)


class FileOperations:
    """File operations for VCS."""

    def __init__(self, provider: VCSProvider):
        self._provider = provider

    def get_file_status(self, path: str) -> FileStatusInfo:
        """Get detailed status for a file."""
        status = self._provider.get_file_status(path)
        full_path = os.path.join(self._provider.root_path, path)

        info = FileStatusInfo(path=path, status=status)

        if os.path.exists(full_path):
            stat = os.stat(full_path)
            info.size = stat.st_size
            info.modified_time = stat.st_mtime
            info.permissions = oct(stat.st_mode)[-3:]

        return info

    def get_all_status(self) -> List[FileStatusInfo]:
        """Get status for all modified files."""
        modified = self._provider.get_modified_files()
        return [self.get_file_status(path) for path, _ in modified]

    def revert_file(self, path: str) -> bool:
        """Revert a single file to its last committed state."""
        return self._provider.revert([path])

    def revert_files(self, paths: List[str]) -> bool:
        """Revert multiple files."""
        return self._provider.revert(paths)

    def revert_all(self) -> bool:
        """Revert all modified files."""
        modified = self._provider.get_modified_files()
        paths = [path for path, _ in modified if _ != FileStatus.UNTRACKED]
        if not paths:
            return True
        return self._provider.revert(paths)

    def stage_file(self, path: str) -> bool:
        """Stage a file for commit."""
        return self._provider.add([path])

    def stage_files(self, paths: List[str]) -> bool:
        """Stage multiple files."""
        return self._provider.add(paths)

    def unstage_file(self, path: str) -> bool:
        """Unstage a file."""
        # This is VCS-specific; for Git, use reset
        if self._provider.vcs_type == VCSType.GIT:
            import subprocess
            result = subprocess.run(
                ["git", "reset", "HEAD", "--", path],
                cwd=self._provider.root_path,
                capture_output=True,
            )
            return result.returncode == 0
        return False

    def discard_untracked(self, paths: Optional[List[str]] = None) -> List[str]:
        """Discard untracked files."""
        return self._provider.clean(directories=True, force=True, dry_run=False)

    def get_file_history(self, path: str, count: int = 10) -> List[Any]:
        """Get commit history for a file."""
        return self._provider.get_commits(count=count, path=path)


class DiffViewer:
    """View and parse diffs."""

    def __init__(self, provider: VCSProvider):
        self._provider = provider

    def get_diff(
        self,
        path: Optional[str] = None,
        staged: bool = False,
        commit1: Optional[str] = None,
        commit2: Optional[str] = None
    ) -> str:
        """Get raw diff output."""
        return self._provider.diff(path, staged, commit1, commit2)

    def parse_diff(self, diff_text: str) -> List[FileDiff]:
        """Parse a unified diff into structured format."""
        file_diffs = []
        current_diff: Optional[FileDiff] = None
        current_hunk: Optional[DiffHunk] = None
        old_line = 0
        new_line = 0

        for line in diff_text.split("\n"):
            # File header
            if line.startswith("diff --"):
                if current_diff:
                    file_diffs.append(current_diff)
                current_diff = FileDiff(old_path=None, new_path=None, status=FileStatus.MODIFIED)
                current_hunk = None

            elif line.startswith("---"):
                if current_diff:
                    path = line[4:].strip()
                    if path.startswith("a/"):
                        path = path[2:]
                    current_diff.old_path = path if path != "/dev/null" else None

            elif line.startswith("+++"):
                if current_diff:
                    path = line[4:].strip()
                    if path.startswith("b/"):
                        path = path[2:]
                    current_diff.new_path = path if path != "/dev/null" else None

                    # Determine status
                    if current_diff.old_path is None:
                        current_diff.status = FileStatus.ADDED
                    elif current_diff.new_path is None:
                        current_diff.status = FileStatus.DELETED

            elif line.startswith("@@"):
                # Hunk header
                match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)", line)
                if match and current_diff:
                    old_start = int(match.group(1))
                    old_count = int(match.group(2)) if match.group(2) else 1
                    new_start = int(match.group(3))
                    new_count = int(match.group(4)) if match.group(4) else 1
                    header = match.group(5).strip()

                    current_hunk = DiffHunk(
                        old_start=old_start,
                        old_count=old_count,
                        new_start=new_start,
                        new_count=new_count,
                        header=header,
                    )
                    current_diff.hunks.append(current_hunk)
                    old_line = old_start
                    new_line = new_start

            elif line.startswith("Binary files"):
                if current_diff:
                    current_diff.is_binary = True

            elif current_hunk is not None:
                # Diff line
                if line.startswith("+"):
                    current_hunk.lines.append(DiffLine(
                        line_type=DiffLineType.ADDITION,
                        content=line[1:],
                        new_line_number=new_line,
                    ))
                    new_line += 1
                elif line.startswith("-"):
                    current_hunk.lines.append(DiffLine(
                        line_type=DiffLineType.DELETION,
                        content=line[1:],
                        old_line_number=old_line,
                    ))
                    old_line += 1
                elif line.startswith(" ") or line == "":
                    current_hunk.lines.append(DiffLine(
                        line_type=DiffLineType.CONTEXT,
                        content=line[1:] if line else "",
                        old_line_number=old_line,
                        new_line_number=new_line,
                    ))
                    old_line += 1
                    new_line += 1

        if current_diff:
            file_diffs.append(current_diff)

        return file_diffs

    def get_file_diff(
        self,
        path: str,
        staged: bool = False,
        commit1: Optional[str] = None,
        commit2: Optional[str] = None
    ) -> Optional[FileDiff]:
        """Get parsed diff for a specific file."""
        diff_text = self.get_diff(path, staged, commit1, commit2)
        diffs = self.parse_diff(diff_text)
        return diffs[0] if diffs else None

    def compare_files(self, file1: str, file2: str) -> FileDiff:
        """Compare two files directly."""
        content1 = ""
        content2 = ""

        if os.path.exists(file1):
            with open(file1, "r") as f:
                content1 = f.read()

        if os.path.exists(file2):
            with open(file2, "r") as f:
                content2 = f.read()

        return self.compare_content(content1, content2, file1, file2)

    def compare_content(
        self,
        content1: str,
        content2: str,
        name1: str = "a",
        name2: str = "b"
    ) -> FileDiff:
        """Compare two content strings."""
        lines1 = content1.splitlines(keepends=True)
        lines2 = content2.splitlines(keepends=True)

        diff = list(difflib.unified_diff(lines1, lines2, name1, name2))

        file_diff = FileDiff(old_path=name1, new_path=name2, status=FileStatus.MODIFIED)
        current_hunk: Optional[DiffHunk] = None
        old_line = 0
        new_line = 0

        for line in diff:
            if line.startswith("@@"):
                match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
                if match:
                    current_hunk = DiffHunk(
                        old_start=int(match.group(1)),
                        old_count=int(match.group(2)) if match.group(2) else 1,
                        new_start=int(match.group(3)),
                        new_count=int(match.group(4)) if match.group(4) else 1,
                    )
                    file_diff.hunks.append(current_hunk)
                    old_line = current_hunk.old_start
                    new_line = current_hunk.new_start

            elif current_hunk and not line.startswith("---") and not line.startswith("+++"):
                content = line[1:].rstrip("\n")
                if line.startswith("+"):
                    current_hunk.lines.append(DiffLine(
                        line_type=DiffLineType.ADDITION,
                        content=content,
                        new_line_number=new_line,
                    ))
                    new_line += 1
                elif line.startswith("-"):
                    current_hunk.lines.append(DiffLine(
                        line_type=DiffLineType.DELETION,
                        content=content,
                        old_line_number=old_line,
                    ))
                    old_line += 1
                else:
                    current_hunk.lines.append(DiffLine(
                        line_type=DiffLineType.CONTEXT,
                        content=content,
                        old_line_number=old_line,
                        new_line_number=new_line,
                    ))
                    old_line += 1
                    new_line += 1

        return file_diff

    def format_diff(
        self,
        file_diff: FileDiff,
        context_lines: int = 3,
        show_line_numbers: bool = True,
        colorize: bool = False
    ) -> str:
        """Format a diff for display."""
        lines = []

        # Header
        if file_diff.old_path:
            lines.append(f"--- {file_diff.old_path}")
        if file_diff.new_path:
            lines.append(f"+++ {file_diff.new_path}")

        if file_diff.is_binary:
            lines.append("Binary files differ")
            return "\n".join(lines)

        # Hunks
        for hunk in file_diff.hunks:
            header = f"@@ -{hunk.old_start},{hunk.old_count} +{hunk.new_start},{hunk.new_count} @@"
            if hunk.header:
                header += f" {hunk.header}"
            lines.append(header)

            for diff_line in hunk.lines:
                formatted = diff_line.format(show_line_numbers)
                if colorize:
                    if diff_line.line_type == DiffLineType.ADDITION:
                        formatted = f"\033[32m{formatted}\033[0m"
                    elif diff_line.line_type == DiffLineType.DELETION:
                        formatted = f"\033[31m{formatted}\033[0m"
                lines.append(formatted)

        return "\n".join(lines)

    def get_stats(self, file_diff: FileDiff) -> Dict[str, int]:
        """Get statistics for a diff."""
        return {
            "additions": file_diff.additions,
            "deletions": file_diff.deletions,
            "hunks": len(file_diff.hunks),
            "lines_changed": file_diff.additions + file_diff.deletions,
        }
