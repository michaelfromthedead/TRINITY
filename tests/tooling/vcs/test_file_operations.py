"""Tests for VCS file operations."""
import pytest
import os
import subprocess
import tempfile
from unittest.mock import MagicMock, patch
from engine.tooling.vcs.file_operations import (
    FileStatusInfo,
    DiffLine,
    DiffLineType,
    DiffHunk,
    FileDiff,
    FileOperations,
    DiffViewer,
)
from engine.tooling.vcs.vcs_integration import FileStatus
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

    (repo_path / "README.md").write_text("# Test Repo\n")
    subprocess.run(["git", "add", "."], cwd=str(repo_path), capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=str(repo_path), capture_output=True
    )

    yield str(repo_path)


class TestFileStatusInfo:
    """Tests for FileStatusInfo dataclass."""

    def test_status_info_creation(self):
        """Test creating file status info."""
        info = FileStatusInfo(
            path="src/main.cpp",
            status=FileStatus.MODIFIED,
            size=1024,
        )
        assert info.path == "src/main.cpp"
        assert info.status == FileStatus.MODIFIED

    def test_is_binary_by_extension(self):
        """Test binary detection by extension."""
        binary_info = FileStatusInfo(
            path="textures/diffuse.png",
            status=FileStatus.MODIFIED,
        )
        text_info = FileStatusInfo(
            path="src/main.cpp",
            status=FileStatus.MODIFIED,
        )

        assert binary_info.is_binary is True
        assert text_info.is_binary is False

    def test_is_binary_various_formats(self):
        """Test binary detection for various formats."""
        binary_paths = [
            "model.fbx", "audio.wav", "video.mp4",
            "archive.zip", "document.pdf", "font.ttf"
        ]

        for path in binary_paths:
            info = FileStatusInfo(path=path, status=FileStatus.ADDED)
            assert info.is_binary is True, f"{path} should be detected as binary"


class TestDiffLine:
    """Tests for DiffLine dataclass."""

    def test_context_line(self):
        """Test context line."""
        line = DiffLine(
            line_type=DiffLineType.CONTEXT,
            content="unchanged code",
            old_line_number=10,
            new_line_number=10,
        )
        assert line.line_type == DiffLineType.CONTEXT

    def test_addition_line(self):
        """Test addition line."""
        line = DiffLine(
            line_type=DiffLineType.ADDITION,
            content="new code",
            new_line_number=15,
        )
        assert line.old_line_number is None
        assert line.new_line_number == 15

    def test_format_with_numbers(self):
        """Test formatting with line numbers."""
        line = DiffLine(
            line_type=DiffLineType.ADDITION,
            content="new line",
            new_line_number=42,
        )
        formatted = line.format(show_line_numbers=True)
        assert "42" in formatted
        assert "+new line" in formatted

    def test_format_without_numbers(self):
        """Test formatting without line numbers."""
        line = DiffLine(
            line_type=DiffLineType.DELETION,
            content="old line",
        )
        formatted = line.format(show_line_numbers=False)
        assert "-old line" in formatted


class TestDiffHunk:
    """Tests for DiffHunk dataclass."""

    def test_hunk_creation(self):
        """Test creating diff hunk."""
        hunk = DiffHunk(
            old_start=10,
            old_count=5,
            new_start=10,
            new_count=7,
        )
        assert hunk.old_start == 10
        assert hunk.new_count == 7

    def test_additions_count(self):
        """Test counting additions."""
        hunk = DiffHunk(
            old_start=1, old_count=3, new_start=1, new_count=5,
            lines=[
                DiffLine(DiffLineType.CONTEXT, "line1", 1, 1),
                DiffLine(DiffLineType.ADDITION, "new1", None, 2),
                DiffLine(DiffLineType.ADDITION, "new2", None, 3),
                DiffLine(DiffLineType.CONTEXT, "line2", 2, 4),
                DiffLine(DiffLineType.DELETION, "old1", 3, None),
            ]
        )
        assert hunk.additions == 2
        assert hunk.deletions == 1


class TestFileDiff:
    """Tests for FileDiff dataclass."""

    def test_file_diff_creation(self):
        """Test creating file diff."""
        diff = FileDiff(
            old_path="old_name.txt",
            new_path="new_name.txt",
            status=FileStatus.RENAMED,
            similarity=90,
        )
        assert diff.path == "new_name.txt"

    def test_file_diff_stats(self):
        """Test file diff statistics."""
        hunk1 = DiffHunk(1, 2, 1, 3, lines=[
            DiffLine(DiffLineType.ADDITION, "a"),
            DiffLine(DiffLineType.ADDITION, "b"),
        ])
        hunk2 = DiffHunk(10, 2, 10, 1, lines=[
            DiffLine(DiffLineType.DELETION, "c"),
        ])

        diff = FileDiff(
            old_path="file.txt",
            new_path="file.txt",
            status=FileStatus.MODIFIED,
            hunks=[hunk1, hunk2],
        )

        assert diff.additions == 2
        assert diff.deletions == 1


class TestFileOperations:
    """Tests for FileOperations."""

    def test_get_file_status(self, git_repo):
        """Test getting file status."""
        provider = GitProvider(git_repo)
        ops = FileOperations(provider)

        # Modify file
        with open(os.path.join(git_repo, "README.md"), "a") as f:
            f.write("Modified\n")

        info = ops.get_file_status("README.md")
        assert info.status == FileStatus.MODIFIED
        assert info.path == "README.md"

    def test_get_all_status(self, git_repo):
        """Test getting all file statuses."""
        # Create modifications
        with open(os.path.join(git_repo, "README.md"), "a") as f:
            f.write("Modified\n")
        with open(os.path.join(git_repo, "new.txt"), "w") as f:
            f.write("New\n")

        provider = GitProvider(git_repo)
        ops = FileOperations(provider)
        statuses = ops.get_all_status()

        assert len(statuses) == 2

    def test_revert_file(self, git_repo):
        """Test reverting a file."""
        # Modify file
        with open(os.path.join(git_repo, "README.md"), "w") as f:
            f.write("Different content\n")

        provider = GitProvider(git_repo)
        ops = FileOperations(provider)
        result = ops.revert_file("README.md")

        assert result is True
        with open(os.path.join(git_repo, "README.md"), "r") as f:
            content = f.read()
        assert "# Test Repo" in content

    def test_stage_file(self, git_repo):
        """Test staging a file."""
        with open(os.path.join(git_repo, "to_stage.txt"), "w") as f:
            f.write("Content\n")

        provider = GitProvider(git_repo)
        ops = FileOperations(provider)
        result = ops.stage_file("to_stage.txt")

        assert result is True

    def test_unstage_file(self, git_repo):
        """Test unstaging a file."""
        with open(os.path.join(git_repo, "staged.txt"), "w") as f:
            f.write("Content\n")
        subprocess.run(["git", "add", "staged.txt"], cwd=git_repo)

        provider = GitProvider(git_repo)
        ops = FileOperations(provider)
        result = ops.unstage_file("staged.txt")

        assert result is True


class TestDiffViewer:
    """Tests for DiffViewer."""

    def test_get_diff(self, git_repo):
        """Test getting raw diff."""
        with open(os.path.join(git_repo, "README.md"), "a") as f:
            f.write("New line\n")

        provider = GitProvider(git_repo)
        viewer = DiffViewer(provider)
        diff = viewer.get_diff()

        assert "New line" in diff

    def test_parse_diff(self, git_repo):
        """Test parsing unified diff."""
        diff_text = """diff --git a/file.txt b/file.txt
--- a/file.txt
+++ b/file.txt
@@ -1,3 +1,4 @@
 line 1
+added line
 line 2
 line 3
"""
        provider = GitProvider(git_repo)
        viewer = DiffViewer(provider)
        diffs = viewer.parse_diff(diff_text)

        assert len(diffs) == 1
        assert diffs[0].path == "file.txt"
        assert len(diffs[0].hunks) == 1
        assert diffs[0].hunks[0].additions == 1

    def test_compare_content(self, git_repo):
        """Test comparing content strings."""
        content1 = "line 1\nline 2\nline 3\n"
        content2 = "line 1\nmodified line\nline 3\n"

        provider = GitProvider(git_repo)
        viewer = DiffViewer(provider)
        diff = viewer.compare_content(content1, content2, "old.txt", "new.txt")

        assert diff.old_path == "old.txt"
        assert diff.new_path == "new.txt"
        assert diff.deletions >= 1
        assert diff.additions >= 1

    def test_compare_files(self, git_repo):
        """Test comparing two files."""
        file1 = os.path.join(git_repo, "file1.txt")
        file2 = os.path.join(git_repo, "file2.txt")

        with open(file1, "w") as f:
            f.write("original\n")
        with open(file2, "w") as f:
            f.write("modified\n")

        provider = GitProvider(git_repo)
        viewer = DiffViewer(provider)
        diff = viewer.compare_files(file1, file2)

        assert diff is not None

    def test_format_diff(self, git_repo):
        """Test formatting a diff."""
        hunk = DiffHunk(
            old_start=1, old_count=2, new_start=1, new_count=3,
            lines=[
                DiffLine(DiffLineType.CONTEXT, "unchanged", 1, 1),
                DiffLine(DiffLineType.ADDITION, "added", None, 2),
                DiffLine(DiffLineType.CONTEXT, "unchanged2", 2, 3),
            ]
        )
        file_diff = FileDiff(
            old_path="test.txt",
            new_path="test.txt",
            status=FileStatus.MODIFIED,
            hunks=[hunk],
        )

        provider = GitProvider(git_repo)
        viewer = DiffViewer(provider)
        formatted = viewer.format_diff(file_diff)

        assert "test.txt" in formatted
        assert "+added" in formatted

    def test_get_stats(self, git_repo):
        """Test getting diff statistics."""
        hunk = DiffHunk(
            old_start=1, old_count=3, new_start=1, new_count=4,
            lines=[
                DiffLine(DiffLineType.ADDITION, "a"),
                DiffLine(DiffLineType.ADDITION, "b"),
                DiffLine(DiffLineType.DELETION, "c"),
            ]
        )
        file_diff = FileDiff(
            old_path="f.txt", new_path="f.txt",
            status=FileStatus.MODIFIED, hunks=[hunk]
        )

        provider = GitProvider(git_repo)
        viewer = DiffViewer(provider)
        stats = viewer.get_stats(file_diff)

        assert stats["additions"] == 2
        assert stats["deletions"] == 1
        assert stats["hunks"] == 1
