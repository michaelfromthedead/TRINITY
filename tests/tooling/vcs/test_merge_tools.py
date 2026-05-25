"""Tests for merge conflict resolution tools."""
import pytest
import os
import subprocess
from engine.tooling.vcs.merge_tools import (
    MergeStrategy,
    MergeResult,
    ConflictRegion,
    ConflictInfo,
    ThreeWayMerge,
    MergeResolver,
)
from engine.tooling.vcs.git_provider import GitProvider


@pytest.fixture
def git_repo(tmp_path):
    """Create a temporary Git repository."""
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    subprocess.run(["git", "init"], cwd=str(repo_path), capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=str(repo_path), capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=str(repo_path), capture_output=True
    )

    (repo_path / "file.txt").write_text("line 1\nline 2\nline 3\n")
    subprocess.run(["git", "add", "."], cwd=str(repo_path), capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=str(repo_path), capture_output=True
    )

    yield str(repo_path)


class TestMergeStrategy:
    """Tests for MergeStrategy enum."""

    def test_all_strategies_exist(self):
        """Test all merge strategies exist."""
        assert MergeStrategy.OURS
        assert MergeStrategy.THEIRS
        assert MergeStrategy.UNION
        assert MergeStrategy.MANUAL
        assert MergeStrategy.AUTO
        assert MergeStrategy.RECURSIVE


class TestMergeResult:
    """Tests for MergeResult enum."""

    def test_all_results_exist(self):
        """Test all merge results exist."""
        assert MergeResult.SUCCESS
        assert MergeResult.CONFLICT
        assert MergeResult.NO_CHANGES
        assert MergeResult.ERROR


class TestConflictRegion:
    """Tests for ConflictRegion dataclass."""

    def test_region_creation(self):
        """Test creating conflict region."""
        region = ConflictRegion(
            start_line=10,
            end_line=20,
            ours_content=["our line 1", "our line 2"],
            theirs_content=["their line 1"],
        )
        assert region.start_line == 10
        assert len(region.ours_content) == 2
        assert len(region.theirs_content) == 1

    def test_line_count(self):
        """Test line count calculation."""
        region = ConflictRegion(
            start_line=5,
            end_line=15,
            ours_content=[],
            theirs_content=[],
        )
        assert region.line_count == 11

    def test_resolved_state(self):
        """Test resolved state."""
        region = ConflictRegion(
            start_line=1,
            end_line=5,
            ours_content=["a"],
            theirs_content=["b"],
        )
        assert region.resolved is False

        region.resolved_content = ["merged"]
        region.resolved = True
        assert region.resolved is True


class TestConflictInfo:
    """Tests for ConflictInfo dataclass."""

    def test_conflict_info_creation(self):
        """Test creating conflict info."""
        info = ConflictInfo(
            path="src/main.cpp",
            regions=[
                ConflictRegion(1, 5, ["a"], ["b"]),
                ConflictRegion(10, 15, ["c"], ["d"]),
            ],
        )
        assert info.path == "src/main.cpp"
        assert info.conflict_count == 2

    def test_unresolved_count(self):
        """Test unresolved count."""
        regions = [
            ConflictRegion(1, 5, ["a"], ["b"]),
            ConflictRegion(10, 15, ["c"], ["d"]),
        ]
        regions[0].resolved = True

        info = ConflictInfo(path="file.txt", regions=regions)
        assert info.unresolved_count == 1

    def test_is_resolved(self):
        """Test is_resolved property."""
        regions = [
            ConflictRegion(1, 5, ["a"], ["b"]),
        ]

        info = ConflictInfo(path="file.txt", regions=regions)
        assert info.is_resolved is False

        regions[0].resolved = True
        assert info.is_resolved is True

    def test_binary_conflict(self):
        """Test binary file conflict."""
        info = ConflictInfo(
            path="image.png",
            is_binary=True,
        )
        assert info.is_binary is True
        assert info.conflict_count == 0


class TestThreeWayMerge:
    """Tests for ThreeWayMerge."""

    def test_no_conflict_merge(self):
        """Test merge with no conflicts."""
        merger = ThreeWayMerge()

        base = ["line 1", "line 2", "line 3"]
        ours = ["line 1", "modified ours", "line 3"]
        theirs = ["line 1", "line 2", "line 3", "new line"]

        result, conflicts = merger.merge(base, ours, theirs)

        # Should have no conflicts
        assert len(conflicts) == 0

    def test_conflict_merge(self):
        """Test merge with conflicts."""
        merger = ThreeWayMerge()

        base = ["line 1", "line 2", "line 3"]
        ours = ["line 1", "ours change", "line 3"]
        theirs = ["line 1", "theirs change", "line 3"]

        result, conflicts = merger.merge(base, ours, theirs)

        # Should have conflict on line 2
        assert len(conflicts) >= 1

    def test_same_change_no_conflict(self):
        """Test same change in both branches."""
        merger = ThreeWayMerge()

        base = ["line 1", "line 2"]
        ours = ["line 1", "same change"]
        theirs = ["line 1", "same change"]

        result, conflicts = merger.merge(base, ours, theirs)

        assert len(conflicts) == 0
        assert "same change" in result


class TestMergeResolver:
    """Tests for MergeResolver."""

    def test_resolver_creation(self, git_repo):
        """Test creating merge resolver."""
        provider = GitProvider(git_repo)
        resolver = MergeResolver(provider)
        assert resolver is not None

    def test_parse_conflict_markers(self, git_repo):
        """Test parsing conflict markers in file."""
        conflict_content = """line 1
<<<<<<< ours
our version
=======
their version
>>>>>>> theirs
line 3
"""
        conflict_file = os.path.join(git_repo, "conflict.txt")
        with open(conflict_file, "w") as f:
            f.write(conflict_content)

        provider = GitProvider(git_repo)
        resolver = MergeResolver(provider)
        info = resolver.analyze_conflict("conflict.txt")

        assert info is not None
        assert len(info.regions) == 1
        assert "our version" in info.regions[0].ours_content
        assert "their version" in info.regions[0].theirs_content

    def test_resolve_ours(self, git_repo):
        """Test resolving with ours strategy."""
        conflict_content = """line 1
<<<<<<< ours
keep this
=======
discard this
>>>>>>> theirs
line 3
"""
        conflict_file = os.path.join(git_repo, "conflict.txt")
        with open(conflict_file, "w") as f:
            f.write(conflict_content)

        provider = GitProvider(git_repo)
        resolver = MergeResolver(provider)
        result = resolver.resolve_conflict("conflict.txt", MergeStrategy.OURS)

        assert result is True

        with open(conflict_file, "r") as f:
            resolved = f.read()
        assert "keep this" in resolved
        assert "discard this" not in resolved
        assert "<<<<<<<" not in resolved

    def test_resolve_theirs(self, git_repo):
        """Test resolving with theirs strategy."""
        conflict_content = """start
<<<<<<< ours
discard
=======
keep
>>>>>>> theirs
end
"""
        conflict_file = os.path.join(git_repo, "conflict.txt")
        with open(conflict_file, "w") as f:
            f.write(conflict_content)

        provider = GitProvider(git_repo)
        resolver = MergeResolver(provider)
        result = resolver.resolve_conflict("conflict.txt", MergeStrategy.THEIRS)

        assert result is True

        with open(conflict_file, "r") as f:
            resolved = f.read()
        assert "keep" in resolved
        assert "discard" not in resolved

    def test_resolve_union(self, git_repo):
        """Test resolving with union strategy."""
        conflict_content = """start
<<<<<<< ours
line a
=======
line b
>>>>>>> theirs
end
"""
        conflict_file = os.path.join(git_repo, "conflict.txt")
        with open(conflict_file, "w") as f:
            f.write(conflict_content)

        provider = GitProvider(git_repo)
        resolver = MergeResolver(provider)
        result = resolver.resolve_conflict("conflict.txt", MergeStrategy.UNION)

        assert result is True

        with open(conflict_file, "r") as f:
            resolved = f.read()
        assert "line a" in resolved
        assert "line b" in resolved

    def test_resolve_with_content(self, git_repo):
        """Test resolving with custom content."""
        conflict_content = """before
<<<<<<< ours
old
=======
also old
>>>>>>> theirs
after
"""
        conflict_file = os.path.join(git_repo, "conflict.txt")
        with open(conflict_file, "w") as f:
            f.write(conflict_content)

        provider = GitProvider(git_repo)
        resolver = MergeResolver(provider)
        result = resolver.resolve_with_content(
            "conflict.txt",
            region_index=0,
            content=["completely new content"]
        )

        assert result is True

        with open(conflict_file, "r") as f:
            resolved = f.read()
        assert "completely new content" in resolved
        assert "old" not in resolved

    def test_get_conflicts_empty(self, git_repo):
        """Test getting conflicts when none exist."""
        provider = GitProvider(git_repo)
        resolver = MergeResolver(provider)
        conflicts = resolver.get_conflicts()

        assert len(conflicts) == 0

    def test_mark_resolved(self, git_repo):
        """Test marking file as resolved."""
        # Create and modify file
        test_file = os.path.join(git_repo, "resolved.txt")
        with open(test_file, "w") as f:
            f.write("resolved content")
        subprocess.run(["git", "add", "resolved.txt"], cwd=git_repo)

        provider = GitProvider(git_repo)
        resolver = MergeResolver(provider)
        result = resolver.mark_resolved("resolved.txt")

        assert result is True

    def test_preview_merge(self, git_repo):
        """Test merge preview."""
        # Create a branch with changes
        subprocess.run(["git", "checkout", "-b", "feature"], cwd=git_repo)
        with open(os.path.join(git_repo, "feature.txt"), "w") as f:
            f.write("feature content")
        subprocess.run(["git", "add", "."], cwd=git_repo)
        subprocess.run(["git", "commit", "-m", "Feature"], cwd=git_repo)
        subprocess.run(["git", "checkout", "-"], cwd=git_repo)

        provider = GitProvider(git_repo)
        resolver = MergeResolver(provider)
        preview = resolver.preview_merge("feature")

        assert "source_branch" in preview
        assert preview["source_branch"] == "feature"


class TestMergeEdgeCases:
    """Tests for edge cases in merge operations."""

    def test_multiple_conflicts(self, git_repo):
        """Test file with multiple conflict regions."""
        conflict_content = """part 1
<<<<<<< ours
conflict 1 ours
=======
conflict 1 theirs
>>>>>>> theirs
middle
<<<<<<< ours
conflict 2 ours
=======
conflict 2 theirs
>>>>>>> theirs
part 3
"""
        conflict_file = os.path.join(git_repo, "multi.txt")
        with open(conflict_file, "w") as f:
            f.write(conflict_content)

        provider = GitProvider(git_repo)
        resolver = MergeResolver(provider)
        info = resolver.analyze_conflict("multi.txt")

        assert info is not None
        assert len(info.regions) == 2

    def test_empty_conflict_side(self, git_repo):
        """Test conflict with empty side."""
        conflict_content = """start
<<<<<<< ours
=======
only theirs
>>>>>>> theirs
end
"""
        conflict_file = os.path.join(git_repo, "empty.txt")
        with open(conflict_file, "w") as f:
            f.write(conflict_content)

        provider = GitProvider(git_repo)
        resolver = MergeResolver(provider)
        info = resolver.analyze_conflict("empty.txt")

        assert len(info.regions) == 1
        assert len(info.regions[0].ours_content) == 0
        assert len(info.regions[0].theirs_content) == 1
