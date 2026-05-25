"""Tests for VCS abstract interface."""
import pytest
from engine.tooling.vcs.vcs_integration import (
    VCSType,
    VCSStatus,
    FileStatus,
    ChangeType,
    ChangeInfo,
    Commit,
    Branch,
    Tag,
    Remote,
    VCSError,
    VCSNotFoundError,
    VCSConflictError,
    VCSProvider,
    VCSProviderRegistry,
)


class TestVCSType:
    """Tests for VCSType enum."""

    def test_all_types_exist(self):
        """Test all VCS types exist."""
        assert VCSType.GIT
        assert VCSType.PERFORCE
        assert VCSType.SVN
        assert VCSType.PLASTIC_SCM
        assert VCSType.MERCURIAL


class TestVCSStatus:
    """Tests for VCSStatus enum."""

    def test_all_statuses_exist(self):
        """Test all statuses exist."""
        assert VCSStatus.CLEAN
        assert VCSStatus.MODIFIED
        assert VCSStatus.STAGED
        assert VCSStatus.CONFLICTED
        assert VCSStatus.DETACHED
        assert VCSStatus.UNKNOWN


class TestFileStatus:
    """Tests for FileStatus enum."""

    def test_all_statuses_exist(self):
        """Test all file statuses exist."""
        assert FileStatus.UNTRACKED
        assert FileStatus.ADDED
        assert FileStatus.MODIFIED
        assert FileStatus.DELETED
        assert FileStatus.RENAMED
        assert FileStatus.COPIED
        assert FileStatus.CONFLICTED
        assert FileStatus.IGNORED
        assert FileStatus.UNCHANGED


class TestChangeType:
    """Tests for ChangeType enum."""

    def test_all_types_exist(self):
        """Test all change types exist."""
        assert ChangeType.ADD
        assert ChangeType.MODIFY
        assert ChangeType.DELETE
        assert ChangeType.RENAME
        assert ChangeType.COPY
        assert ChangeType.TYPE_CHANGE


class TestChangeInfo:
    """Tests for ChangeInfo dataclass."""

    def test_change_info_creation(self):
        """Test creating change info."""
        change = ChangeInfo(
            path="src/main.cpp",
            change_type=ChangeType.MODIFY,
            additions=10,
            deletions=5,
        )
        assert change.path == "src/main.cpp"
        assert change.additions == 10

    def test_rename_with_old_path(self):
        """Test rename change with old path."""
        change = ChangeInfo(
            path="src/new_name.cpp",
            change_type=ChangeType.RENAME,
            old_path="src/old_name.cpp",
        )
        assert change.old_path == "src/old_name.cpp"


class TestCommit:
    """Tests for Commit dataclass."""

    def test_commit_creation(self):
        """Test creating commit."""
        commit = Commit(
            id="abc123def456",
            short_id="abc123d",
            message="Add feature\n\nDetailed description",
            author="Test User",
            author_email="test@example.com",
            timestamp=1234567890.0,
        )
        assert commit.id == "abc123def456"
        assert commit.author == "Test User"

    def test_commit_title(self):
        """Test getting commit title."""
        commit = Commit(
            id="abc",
            short_id="abc",
            message="First line\n\nSecond line",
            author="Test",
            author_email="test@test.com",
            timestamp=0,
        )
        assert commit.title == "First line"

    def test_is_merge_commit(self):
        """Test identifying merge commits."""
        regular = Commit(
            id="abc",
            short_id="abc",
            message="Regular commit",
            author="Test",
            author_email="test@test.com",
            timestamp=0,
            parent_ids=["parent1"],
        )
        merge = Commit(
            id="xyz",
            short_id="xyz",
            message="Merge commit",
            author="Test",
            author_email="test@test.com",
            timestamp=0,
            parent_ids=["parent1", "parent2"],
        )

        assert regular.is_merge is False
        assert merge.is_merge is True


class TestBranch:
    """Tests for Branch dataclass."""

    def test_branch_creation(self):
        """Test creating branch."""
        branch = Branch(
            name="feature/test",
            commit_id="abc123",
            is_current=True,
        )
        assert branch.name == "feature/test"
        assert branch.is_current is True

    def test_branch_with_upstream(self):
        """Test branch with upstream info."""
        branch = Branch(
            name="main",
            commit_id="abc123",
            upstream="origin/main",
            ahead=2,
            behind=1,
        )
        assert branch.upstream == "origin/main"
        assert branch.ahead == 2
        assert branch.behind == 1


class TestTag:
    """Tests for Tag dataclass."""

    def test_tag_creation(self):
        """Test creating tag."""
        tag = Tag(
            name="v1.0.0",
            commit_id="abc123",
            message="Release version 1.0.0",
            is_annotated=True,
        )
        assert tag.name == "v1.0.0"
        assert tag.is_annotated is True

    def test_lightweight_tag(self):
        """Test lightweight tag."""
        tag = Tag(
            name="temp-tag",
            commit_id="abc123",
        )
        assert tag.is_annotated is False


class TestRemote:
    """Tests for Remote dataclass."""

    def test_remote_creation(self):
        """Test creating remote."""
        remote = Remote(
            name="origin",
            fetch_url="https://github.com/user/repo.git",
            push_url="git@github.com:user/repo.git",
        )
        assert remote.name == "origin"
        assert "github.com" in remote.fetch_url


class TestVCSErrors:
    """Tests for VCS error classes."""

    def test_vcs_error(self):
        """Test VCSError."""
        error = VCSError("Something went wrong")
        assert str(error) == "Something went wrong"

    def test_vcs_not_found_error(self):
        """Test VCSNotFoundError."""
        error = VCSNotFoundError("Repository not found")
        assert "Repository" in str(error)

    def test_vcs_conflict_error(self):
        """Test VCSConflictError with files."""
        error = VCSConflictError(
            "Merge conflict",
            conflicted_files=["file1.txt", "file2.txt"]
        )
        assert len(error.conflicted_files) == 2


class TestVCSProviderRegistry:
    """Tests for VCSProviderRegistry."""

    def test_detect_vcs_none(self, tmp_path):
        """Test detecting no VCS in empty directory."""
        result = VCSProviderRegistry.detect_vcs(str(tmp_path))
        assert result is None

    def test_detect_git(self, tmp_path):
        """Test detecting Git repository."""
        (tmp_path / ".git").mkdir()
        result = VCSProviderRegistry.detect_vcs(str(tmp_path))
        assert result == VCSType.GIT

    def test_detect_svn(self, tmp_path):
        """Test detecting SVN repository."""
        (tmp_path / ".svn").mkdir()
        result = VCSProviderRegistry.detect_vcs(str(tmp_path))
        assert result == VCSType.SVN

    def test_auto_provider_no_vcs(self, tmp_path):
        """Test auto_provider returns None for non-VCS directory."""
        provider = VCSProviderRegistry.auto_provider(str(tmp_path))
        assert provider is None
