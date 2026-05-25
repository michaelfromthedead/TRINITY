"""Tests for Git provider implementation."""
import pytest
import os
import subprocess
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from engine.tooling.vcs.git_provider import (
    GitProvider,
    GitConfig,
    GitStash,
    GitDiffOptions,
)
from engine.tooling.vcs.vcs_integration import (
    VCSType,
    VCSStatus,
    FileStatus,
    VCSError,
    VCSNotFoundError,
)


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

    # Create initial commit
    (repo_path / "README.md").write_text("# Test Repo")
    subprocess.run(["git", "add", "."], cwd=str(repo_path), capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=str(repo_path), capture_output=True
    )

    yield str(repo_path)


class TestGitConfig:
    """Tests for GitConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = GitConfig()
        assert config.default_branch == "main"
        assert config.auto_crlf == "input"

    def test_custom_config(self):
        """Test custom configuration."""
        config = GitConfig(
            user_name="Test User",
            user_email="test@test.com",
        )
        assert config.user_name == "Test User"


class TestGitStash:
    """Tests for GitStash dataclass."""

    def test_stash_creation(self):
        """Test creating stash object."""
        stash = GitStash(
            index=0,
            message="WIP: feature",
            branch="main",
            commit_id="abc123",
        )
        assert stash.index == 0
        assert stash.message == "WIP: feature"


class TestGitDiffOptions:
    """Tests for GitDiffOptions dataclass."""

    def test_default_options(self):
        """Test default diff options."""
        options = GitDiffOptions()
        assert options.context_lines == 3
        assert options.ignore_whitespace is False


class TestGitProvider:
    """Tests for GitProvider."""

    def test_provider_creation(self, git_repo):
        """Test creating Git provider."""
        provider = GitProvider(git_repo)
        assert provider.vcs_type == VCSType.GIT

    def test_is_valid_repository(self, git_repo):
        """Test repository validation."""
        provider = GitProvider(git_repo)
        assert provider.is_valid_repository() is True

    def test_is_not_valid_repository(self, tmp_path):
        """Test non-repository validation."""
        provider = GitProvider(str(tmp_path))
        assert provider.is_valid_repository() is False

    def test_root_path(self, git_repo):
        """Test getting root path."""
        provider = GitProvider(git_repo)
        assert provider.root_path == git_repo

    def test_get_status_clean(self, git_repo):
        """Test getting clean status."""
        provider = GitProvider(git_repo)
        status = provider.get_status()
        assert status == VCSStatus.CLEAN

    def test_get_status_modified(self, git_repo):
        """Test getting modified status."""
        # Modify a file
        with open(os.path.join(git_repo, "README.md"), "a") as f:
            f.write("\nModified")

        provider = GitProvider(git_repo)
        status = provider.get_status()
        assert status == VCSStatus.MODIFIED

    def test_get_status_staged(self, git_repo):
        """Test getting staged status."""
        # Create and stage a file
        with open(os.path.join(git_repo, "new_file.txt"), "w") as f:
            f.write("New content")
        subprocess.run(["git", "add", "new_file.txt"], cwd=git_repo)

        provider = GitProvider(git_repo)
        status = provider.get_status()
        assert status == VCSStatus.STAGED

    def test_get_file_status_unchanged(self, git_repo):
        """Test file status for unchanged file."""
        provider = GitProvider(git_repo)
        status = provider.get_file_status("README.md")
        assert status == FileStatus.UNCHANGED

    def test_get_file_status_modified(self, git_repo):
        """Test file status for modified file."""
        with open(os.path.join(git_repo, "README.md"), "a") as f:
            f.write("\nModified")

        provider = GitProvider(git_repo)
        status = provider.get_file_status("README.md")
        assert status == FileStatus.MODIFIED

    def test_get_file_status_untracked(self, git_repo):
        """Test file status for untracked file."""
        with open(os.path.join(git_repo, "untracked.txt"), "w") as f:
            f.write("Untracked")

        provider = GitProvider(git_repo)
        status = provider.get_file_status("untracked.txt")
        assert status == FileStatus.UNTRACKED

    def test_get_modified_files(self, git_repo):
        """Test getting list of modified files."""
        # Create untracked and modify existing
        with open(os.path.join(git_repo, "new.txt"), "w") as f:
            f.write("New")
        with open(os.path.join(git_repo, "README.md"), "a") as f:
            f.write("\nModified")

        provider = GitProvider(git_repo)
        modified = provider.get_modified_files()

        assert len(modified) == 2
        paths = [m[0] for m in modified]
        assert "new.txt" in paths
        assert "README.md" in paths

    def test_commit(self, git_repo):
        """Test creating a commit."""
        with open(os.path.join(git_repo, "new_file.txt"), "w") as f:
            f.write("Content")

        provider = GitProvider(git_repo)
        provider.add(["new_file.txt"])
        commit = provider.commit("Add new file")

        assert "Add new file" in commit.message
        assert commit.id is not None

    def test_get_commit(self, git_repo):
        """Test getting a specific commit."""
        provider = GitProvider(git_repo)
        commit = provider.get_commit("HEAD")

        assert commit.id is not None
        assert commit.message == "Initial commit"

    def test_get_commits(self, git_repo):
        """Test getting commit history."""
        # Add another commit
        with open(os.path.join(git_repo, "file.txt"), "w") as f:
            f.write("Content")
        subprocess.run(["git", "add", "."], cwd=git_repo)
        subprocess.run(["git", "commit", "-m", "Second commit"], cwd=git_repo)

        provider = GitProvider(git_repo)
        commits = provider.get_commits(count=10)

        assert len(commits) >= 2

    def test_get_current_branch(self, git_repo):
        """Test getting current branch."""
        provider = GitProvider(git_repo)
        branch = provider.get_current_branch()

        assert branch is not None
        # Could be main or master depending on git config
        assert branch.is_current is True

    def test_get_branches(self, git_repo):
        """Test getting all branches."""
        # Create another branch
        subprocess.run(["git", "branch", "feature"], cwd=git_repo)

        provider = GitProvider(git_repo)
        branches = provider.get_branches()

        assert len(branches) >= 2
        names = [b.name for b in branches]
        assert "feature" in names

    def test_create_branch(self, git_repo):
        """Test creating a branch."""
        provider = GitProvider(git_repo)
        branch = provider.create_branch("new-feature")

        assert branch.name == "new-feature"

    def test_delete_branch(self, git_repo):
        """Test deleting a branch."""
        subprocess.run(["git", "branch", "to-delete"], cwd=git_repo)

        provider = GitProvider(git_repo)
        result = provider.delete_branch("to-delete")

        assert result is True

    def test_checkout(self, git_repo):
        """Test checking out a branch."""
        subprocess.run(["git", "branch", "feature"], cwd=git_repo)

        provider = GitProvider(git_repo)
        result = provider.checkout("feature")

        assert result is True
        assert provider.get_current_branch().name == "feature"

    def test_diff(self, git_repo):
        """Test getting diff."""
        with open(os.path.join(git_repo, "README.md"), "a") as f:
            f.write("\nNew line")

        provider = GitProvider(git_repo)
        diff = provider.diff()

        assert "New line" in diff

    def test_get_tags(self, git_repo):
        """Test getting tags."""
        subprocess.run(
            ["git", "tag", "-a", "v1.0", "-m", "Version 1.0"],
            cwd=git_repo
        )

        provider = GitProvider(git_repo)
        tags = provider.get_tags()

        assert len(tags) >= 1
        assert any(t.name == "v1.0" for t in tags)

    def test_create_tag(self, git_repo):
        """Test creating a tag."""
        provider = GitProvider(git_repo)
        tag = provider.create_tag("v2.0", "Version 2.0")

        assert tag.name == "v2.0"

    def test_add_files(self, git_repo):
        """Test adding files."""
        with open(os.path.join(git_repo, "to_add.txt"), "w") as f:
            f.write("Content")

        provider = GitProvider(git_repo)
        result = provider.add(["to_add.txt"])

        assert result is True

    def test_revert_files(self, git_repo):
        """Test reverting files."""
        with open(os.path.join(git_repo, "README.md"), "a") as f:
            f.write("\nUnwanted change")

        provider = GitProvider(git_repo)
        result = provider.revert(["README.md"])

        assert result is True

        # Verify file is reverted
        with open(os.path.join(git_repo, "README.md"), "r") as f:
            content = f.read()
        assert "Unwanted change" not in content

    def test_resolve_ref(self, git_repo):
        """Test resolving ref to commit."""
        provider = GitProvider(git_repo)
        commit_id = provider.resolve_ref("HEAD")

        assert len(commit_id) == 40  # Full SHA

    def test_is_ancestor(self, git_repo):
        """Test checking ancestor relationship."""
        # Get initial commit
        result = subprocess.run(
            ["git", "rev-list", "--max-parents=0", "HEAD"],
            cwd=git_repo, capture_output=True, text=True
        )
        initial = result.stdout.strip()

        provider = GitProvider(git_repo)
        assert provider.is_ancestor(initial, "HEAD") is True


class TestGitProviderSpecificOperations:
    """Tests for Git-specific operations."""

    def test_stash(self, git_repo):
        """Test stashing changes."""
        with open(os.path.join(git_repo, "README.md"), "a") as f:
            f.write("\nModified")

        provider = GitProvider(git_repo)
        stash = provider.stash("WIP")

        assert stash.index == 0
        assert provider.get_status() == VCSStatus.CLEAN

    def test_stash_list(self, git_repo):
        """Test listing stashes."""
        with open(os.path.join(git_repo, "README.md"), "a") as f:
            f.write("\nModified")
        subprocess.run(["git", "stash", "push", "-m", "Test stash"], cwd=git_repo)

        provider = GitProvider(git_repo)
        stashes = provider.stash_list()

        assert len(stashes) >= 1

    def test_get_config(self, git_repo):
        """Test getting config value."""
        provider = GitProvider(git_repo)
        email = provider.get_config("user.email")

        assert email == "test@test.com"

    def test_set_config(self, git_repo):
        """Test setting config value."""
        provider = GitProvider(git_repo)
        result = provider.set_config("test.key", "test-value")

        assert result is True
        assert provider.get_config("test.key") == "test-value"
