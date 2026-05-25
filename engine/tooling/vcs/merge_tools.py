"""Merge conflict resolution with 3-way merge support.

Provides tools for detecting, analyzing, and resolving merge conflicts
with support for various merge strategies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import difflib
import os
import re

from .vcs_integration import VCSProvider, VCSError


class MergeStrategy(Enum):
    """Merge resolution strategy."""
    OURS = auto()           # Keep our version
    THEIRS = auto()         # Keep their version
    UNION = auto()          # Include both (for text)
    MANUAL = auto()         # Requires manual resolution
    AUTO = auto()           # Automatic resolution where possible
    RECURSIVE = auto()      # Recursive 3-way merge


class MergeResult(Enum):
    """Result of a merge operation."""
    SUCCESS = auto()
    CONFLICT = auto()
    NO_CHANGES = auto()
    ERROR = auto()


@dataclass
class ConflictRegion:
    """A region of conflict in a file."""
    start_line: int
    end_line: int
    ours_content: List[str]
    theirs_content: List[str]
    base_content: Optional[List[str]] = None
    resolved_content: Optional[List[str]] = None
    resolved: bool = False

    @property
    def line_count(self) -> int:
        """Total lines in the conflict region."""
        return self.end_line - self.start_line + 1


@dataclass
class ConflictInfo:
    """Information about conflicts in a file."""
    path: str
    regions: List[ConflictRegion] = field(default_factory=list)
    is_binary: bool = False
    ours_branch: str = "HEAD"
    theirs_branch: str = ""

    @property
    def conflict_count(self) -> int:
        """Number of conflict regions."""
        return len(self.regions)

    @property
    def unresolved_count(self) -> int:
        """Number of unresolved conflicts."""
        return sum(1 for r in self.regions if not r.resolved)

    @property
    def is_resolved(self) -> bool:
        """Check if all conflicts are resolved."""
        return all(r.resolved for r in self.regions)


class ThreeWayMerge:
    """3-way merge algorithm implementation."""

    def __init__(self):
        pass

    def merge(
        self,
        base: List[str],
        ours: List[str],
        theirs: List[str]
    ) -> Tuple[List[str], List[ConflictRegion]]:
        """Perform a 3-way merge."""
        # Get diffs from base to each version
        diff_ours = list(difflib.unified_diff(base, ours, lineterm=""))
        diff_theirs = list(difflib.unified_diff(base, theirs, lineterm=""))

        # Use SequenceMatcher for detailed comparison
        matcher_ours = difflib.SequenceMatcher(None, base, ours)
        matcher_theirs = difflib.SequenceMatcher(None, base, theirs)

        result = []
        conflicts = []
        base_idx = 0

        # Get matching blocks
        blocks_ours = matcher_ours.get_matching_blocks()
        blocks_theirs = matcher_theirs.get_matching_blocks()

        # Simple merge: try to interleave changes
        while base_idx < len(base):
            # Check if either version modified this region
            ours_changed = self._is_region_changed(base_idx, base, ours, matcher_ours)
            theirs_changed = self._is_region_changed(base_idx, base, theirs, matcher_theirs)

            if not ours_changed and not theirs_changed:
                # No changes, keep base
                result.append(base[base_idx])
                base_idx += 1
            elif ours_changed and not theirs_changed:
                # Only ours changed, take ours
                ours_lines = self._get_changed_lines(base_idx, base, ours, matcher_ours)
                result.extend(ours_lines)
                base_idx += 1
            elif theirs_changed and not ours_changed:
                # Only theirs changed, take theirs
                theirs_lines = self._get_changed_lines(base_idx, base, theirs, matcher_theirs)
                result.extend(theirs_lines)
                base_idx += 1
            else:
                # Both changed - potential conflict
                ours_lines = self._get_changed_lines(base_idx, base, ours, matcher_ours)
                theirs_lines = self._get_changed_lines(base_idx, base, theirs, matcher_theirs)

                if ours_lines == theirs_lines:
                    # Same change, no conflict
                    result.extend(ours_lines)
                else:
                    # Conflict
                    conflict = ConflictRegion(
                        start_line=len(result),
                        end_line=len(result) + max(len(ours_lines), len(theirs_lines)) - 1,
                        ours_content=ours_lines,
                        theirs_content=theirs_lines,
                        base_content=[base[base_idx]] if base_idx < len(base) else [],
                    )
                    conflicts.append(conflict)

                    # Add conflict markers
                    result.append("<<<<<<< ours")
                    result.extend(ours_lines)
                    result.append("=======")
                    result.extend(theirs_lines)
                    result.append(">>>>>>> theirs")

                base_idx += 1

        # Handle any remaining lines
        ours_remaining = ours[len(base):] if len(ours) > len(base) else []
        theirs_remaining = theirs[len(base):] if len(theirs) > len(base) else []

        if ours_remaining and theirs_remaining:
            if ours_remaining == theirs_remaining:
                result.extend(ours_remaining)
            else:
                conflict = ConflictRegion(
                    start_line=len(result),
                    end_line=len(result) + max(len(ours_remaining), len(theirs_remaining)) - 1,
                    ours_content=ours_remaining,
                    theirs_content=theirs_remaining,
                )
                conflicts.append(conflict)
                result.append("<<<<<<< ours")
                result.extend(ours_remaining)
                result.append("=======")
                result.extend(theirs_remaining)
                result.append(">>>>>>> theirs")
        elif ours_remaining:
            result.extend(ours_remaining)
        elif theirs_remaining:
            result.extend(theirs_remaining)

        return result, conflicts

    def _is_region_changed(
        self,
        base_idx: int,
        base: List[str],
        other: List[str],
        matcher: difflib.SequenceMatcher
    ) -> bool:
        """Check if a region was changed."""
        for block in matcher.get_matching_blocks():
            if block.a <= base_idx < block.a + block.size:
                return False
        return True

    def _get_changed_lines(
        self,
        base_idx: int,
        base: List[str],
        other: List[str],
        matcher: difflib.SequenceMatcher
    ) -> List[str]:
        """Get the changed lines for a region."""
        # Find corresponding lines in other
        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if i1 <= base_idx < i2:
                if op == "replace" or op == "insert":
                    return list(other[j1:j2])
                elif op == "delete":
                    return []
                elif op == "equal":
                    return [other[j1 + (base_idx - i1)]]

        return []


class MergeResolver:
    """Resolve merge conflicts."""

    def __init__(self, provider: VCSProvider):
        self._provider = provider
        self._merge = ThreeWayMerge()

    def get_conflicts(self) -> List[ConflictInfo]:
        """Get all files with conflicts."""
        from .vcs_integration import FileStatus

        conflicts = []
        modified_files = self._provider.get_modified_files()

        for path, status in modified_files:
            if status == FileStatus.CONFLICTED:
                conflict_info = self.analyze_conflict(path)
                if conflict_info:
                    conflicts.append(conflict_info)

        return conflicts

    def analyze_conflict(self, path: str) -> Optional[ConflictInfo]:
        """Analyze conflicts in a file."""
        full_path = os.path.join(self._provider.root_path, path)
        if not os.path.exists(full_path):
            return None

        try:
            with open(full_path, "rb") as f:
                # Check if binary
                chunk = f.read(8192)
                if b"\x00" in chunk:
                    return ConflictInfo(path=path, is_binary=True)
        except Exception:
            return None

        # Read file content
        try:
            with open(full_path, "r") as f:
                content = f.read()
        except Exception:
            return ConflictInfo(path=path, is_binary=True)

        # Parse conflict markers
        regions = self._parse_conflict_markers(content)

        return ConflictInfo(
            path=path,
            regions=regions,
        )

    def _parse_conflict_markers(self, content: str) -> List[ConflictRegion]:
        """Parse Git-style conflict markers."""
        regions = []
        lines = content.split("\n")

        i = 0
        while i < len(lines):
            if lines[i].startswith("<<<<<<<"):
                # Found start of conflict
                ours_start = i + 1
                ours_content = []
                theirs_content = []
                base_content = None

                # Find separator and end
                j = ours_start
                found_separator = False
                found_base = False

                while j < len(lines):
                    if lines[j].startswith("|||||||"):
                        # Base content (diff3 style)
                        found_base = True
                        base_content = []
                        j += 1
                        while j < len(lines) and not lines[j].startswith("======="):
                            base_content.append(lines[j])
                            j += 1
                    elif lines[j].startswith("======="):
                        found_separator = True
                        j += 1
                        while j < len(lines) and not lines[j].startswith(">>>>>>>"):
                            theirs_content.append(lines[j])
                            j += 1
                        break
                    else:
                        ours_content.append(lines[j])
                        j += 1

                if found_separator:
                    region = ConflictRegion(
                        start_line=i,
                        end_line=j,
                        ours_content=ours_content,
                        theirs_content=theirs_content,
                        base_content=base_content,
                    )
                    regions.append(region)
                    i = j + 1
                else:
                    i += 1
            else:
                i += 1

        return regions

    def resolve_conflict(
        self,
        path: str,
        strategy: MergeStrategy,
        region_index: Optional[int] = None
    ) -> bool:
        """Resolve conflicts in a file."""
        conflict_info = self.analyze_conflict(path)
        if not conflict_info or conflict_info.is_binary:
            return False

        full_path = os.path.join(self._provider.root_path, path)
        with open(full_path, "r") as f:
            content = f.read()

        lines = content.split("\n")
        regions = conflict_info.regions

        if region_index is not None:
            # Resolve specific region
            if region_index >= len(regions):
                return False
            regions = [regions[region_index]]

        # Process regions in reverse order to maintain line numbers
        for region in reversed(regions):
            resolved_lines = self._resolve_region(region, strategy)
            region.resolved_content = resolved_lines
            region.resolved = True

            # Replace conflict in file
            lines = (
                lines[:region.start_line] +
                resolved_lines +
                lines[region.end_line + 1:]
            )

        # Write resolved content
        with open(full_path, "w") as f:
            f.write("\n".join(lines))

        return True

    def _resolve_region(
        self,
        region: ConflictRegion,
        strategy: MergeStrategy
    ) -> List[str]:
        """Resolve a single conflict region."""
        if strategy == MergeStrategy.OURS:
            return region.ours_content
        elif strategy == MergeStrategy.THEIRS:
            return region.theirs_content
        elif strategy == MergeStrategy.UNION:
            # Combine both, removing duplicates
            result = list(region.ours_content)
            for line in region.theirs_content:
                if line not in result:
                    result.append(line)
            return result
        elif strategy == MergeStrategy.AUTO:
            # Try automatic resolution
            if region.base_content is not None:
                merged, conflicts = self._merge.merge(
                    region.base_content,
                    region.ours_content,
                    region.theirs_content
                )
                if not conflicts:
                    return merged
            # Fall back to ours if auto fails
            return region.ours_content
        else:
            # Manual - return markers intact
            return (
                ["<<<<<<< ours"] +
                region.ours_content +
                ["======="] +
                region.theirs_content +
                [">>>>>>> theirs"]
            )

    def resolve_with_content(
        self,
        path: str,
        region_index: int,
        content: List[str]
    ) -> bool:
        """Resolve a conflict region with specific content."""
        conflict_info = self.analyze_conflict(path)
        if not conflict_info or region_index >= len(conflict_info.regions):
            return False

        full_path = os.path.join(self._provider.root_path, path)
        with open(full_path, "r") as f:
            file_content = f.read()

        lines = file_content.split("\n")
        region = conflict_info.regions[region_index]

        # Replace the conflict region
        lines = (
            lines[:region.start_line] +
            content +
            lines[region.end_line + 1:]
        )

        with open(full_path, "w") as f:
            f.write("\n".join(lines))

        return True

    def mark_resolved(self, path: str) -> bool:
        """Mark a file as resolved."""
        return self._provider.add([path])

    def abort_merge(self) -> bool:
        """Abort the current merge."""
        if self._provider.vcs_type.name == "GIT":
            import subprocess
            result = subprocess.run(
                ["git", "merge", "--abort"],
                cwd=self._provider.root_path,
                capture_output=True,
            )
            return result.returncode == 0
        return False

    def get_common_ancestor(self, branch1: str, branch2: str) -> Optional[str]:
        """Get the common ancestor of two branches."""
        try:
            return self._provider.get_merge_base(branch1, branch2)
        except (VCSError, NotImplementedError):
            return None

    def preview_merge(
        self,
        source_branch: str,
        target_branch: Optional[str] = None
    ) -> Dict[str, Any]:
        """Preview the result of a merge without performing it."""
        current_branch = self._provider.get_current_branch()
        target = target_branch or (current_branch.name if current_branch else None)

        if not target:
            return {"error": "No target branch"}

        # Get the merge base
        base = self.get_common_ancestor(source_branch, target)

        # Get file changes
        source_diff = self._provider.diff(commit1=base, commit2=source_branch) if base else ""
        target_diff = self._provider.diff(commit1=base, commit2=target) if base else ""

        # Estimate conflicts
        potential_conflicts = self._estimate_conflicts(source_diff, target_diff)

        return {
            "source_branch": source_branch,
            "target_branch": target,
            "merge_base": base,
            "potential_conflicts": potential_conflicts,
        }

    def _estimate_conflicts(self, diff1: str, diff2: str) -> List[str]:
        """Estimate files that might have conflicts."""
        # Extract file paths from diffs
        files1 = set(re.findall(r"^\+\+\+ b/(.+)$", diff1, re.MULTILINE))
        files2 = set(re.findall(r"^\+\+\+ b/(.+)$", diff2, re.MULTILINE))

        # Files modified in both diffs might conflict
        return list(files1 & files2)
