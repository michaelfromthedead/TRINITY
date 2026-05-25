"""Tests for Perforce provider implementation."""
import pytest
import os
from unittest.mock import patch, MagicMock
from engine.tooling.vcs.perforce_provider import (
    PerforceProvider,
    P4Config,
    P4Changelist,
    P4ClientSpec,
)
from engine.tooling.vcs.vcs_integration import (
    VCSType,
    VCSStatus,
    FileStatus,
    VCSError,
)


class TestP4Config:
    """Tests for P4Config dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = P4Config()
        assert config.port == ""
        assert config.user == ""
        assert config.charset == "utf8"

    def test_custom_config(self):
        """Test custom configuration."""
        config = P4Config(
            port="ssl:perforce.example.com:1666",
            user="testuser",
            client="test-client",
        )
        assert config.port == "ssl:perforce.example.com:1666"
        assert config.user == "testuser"


class TestP4Changelist:
    """Tests for P4Changelist dataclass."""

    def test_changelist_creation(self):
        """Test creating changelist."""
        cl = P4Changelist(
            number=12345,
            status="pending",
            description="Fix bug",
            user="testuser",
            client="test-client",
        )
        assert cl.number == 12345
        assert cl.status == "pending"

    def test_changelist_with_files(self):
        """Test changelist with files."""
        cl = P4Changelist(
            number=100,
            status="submitted",
            description="Changes",
            user="user",
            client="client",
            files=["//depot/main.cpp", "//depot/util.cpp"],
        )
        assert len(cl.files) == 2


class TestP4ClientSpec:
    """Tests for P4ClientSpec dataclass."""

    def test_client_spec_creation(self):
        """Test creating client spec."""
        spec = P4ClientSpec(
            client="my-client",
            owner="testuser",
            host="workstation",
            root="/workspace",
            view=[
                ("//depot/...", "//my-client/...")
            ],
        )
        assert spec.client == "my-client"
        assert len(spec.view) == 1


class TestPerforceProvider:
    """Tests for PerforceProvider."""

    @pytest.fixture
    def mock_p4(self):
        """Mock p4 command execution."""
        with patch.object(PerforceProvider, '_run_p4') as mock:
            yield mock

    def test_provider_creation(self, tmp_path):
        """Test creating Perforce provider."""
        config = P4Config(
            port="localhost:1666",
            user="testuser",
            client="test-client",
        )
        provider = PerforceProvider(str(tmp_path), config)
        assert provider.vcs_type == VCSType.PERFORCE

    def test_vcs_type(self, tmp_path):
        """Test VCS type is Perforce."""
        provider = PerforceProvider(str(tmp_path))
        assert provider.vcs_type == VCSType.PERFORCE

    def test_load_p4config_file(self, tmp_path):
        """Test loading .p4config file."""
        p4config = tmp_path / ".p4config"
        p4config.write_text("P4PORT=ssl:server:1666\nP4USER=configuser\n")

        provider = PerforceProvider(str(tmp_path))
        assert provider._config.port == "ssl:server:1666"
        assert provider._config.user == "configuser"

    @patch.object(PerforceProvider, '_run_p4')
    def test_get_status_clean(self, mock_run, tmp_path):
        """Test getting clean status."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=""
        )

        # Simulate valid connection
        provider = PerforceProvider(str(tmp_path))
        provider._root = str(tmp_path)
        status = provider.get_status()

        assert status == VCSStatus.CLEAN

    @patch.object(PerforceProvider, '_run_p4')
    def test_get_status_modified(self, mock_run, tmp_path):
        """Test getting modified status."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="//depot/file.txt#1 - edit default change"
        )

        provider = PerforceProvider(str(tmp_path))
        provider._root = str(tmp_path)
        status = provider.get_status()

        assert status == VCSStatus.MODIFIED

    @patch.object(PerforceProvider, '_run_p4')
    def test_get_file_status_unchanged(self, mock_run, tmp_path):
        """Test file status for synced file."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="Client root: " + str(tmp_path)),  # info (from __init__)
            MagicMock(returncode=1, stdout=""),  # opened
            MagicMock(returncode=0, stdout="... depotFile //depot/file.txt"),  # fstat
        ]

        provider = PerforceProvider(str(tmp_path))
        status = provider.get_file_status("file.txt")

        assert status == FileStatus.UNCHANGED

    @patch.object(PerforceProvider, '_run_p4')
    def test_get_file_status_edited(self, mock_run, tmp_path):
        """Test file status for edited file."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="//depot/file.txt#1 - edit change 12345"
        )

        provider = PerforceProvider(str(tmp_path))
        provider._root = str(tmp_path)
        status = provider.get_file_status("file.txt")

        assert status == FileStatus.MODIFIED

    @patch.object(PerforceProvider, '_run_p4')
    def test_get_file_status_added(self, mock_run, tmp_path):
        """Test file status for added file."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="//depot/new.txt#1 - add change 12345"
        )

        provider = PerforceProvider(str(tmp_path))
        provider._root = str(tmp_path)
        status = provider.get_file_status("new.txt")

        assert status == FileStatus.ADDED

    @patch.object(PerforceProvider, '_run_p4')
    def test_sync(self, mock_run, tmp_path):
        """Test sync operation."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")

        provider = PerforceProvider(str(tmp_path))
        provider._root = str(tmp_path)
        result = provider.sync()

        assert result is True
        mock_run.assert_called()

    @patch.object(PerforceProvider, '_run_p4')
    def test_edit(self, mock_run, tmp_path):
        """Test edit operation."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")

        provider = PerforceProvider(str(tmp_path))
        provider._root = str(tmp_path)
        result = provider.edit(["file.txt"])

        assert result is True

    @patch.object(PerforceProvider, '_run_p4')
    def test_add(self, mock_run, tmp_path):
        """Test add operation."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")

        provider = PerforceProvider(str(tmp_path))
        provider._root = str(tmp_path)
        result = provider.add(["new_file.txt"])

        assert result is True

    @patch.object(PerforceProvider, '_run_p4')
    def test_revert(self, mock_run, tmp_path):
        """Test revert operation."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")

        provider = PerforceProvider(str(tmp_path))
        provider._root = str(tmp_path)
        result = provider.revert(["file.txt"])

        assert result is True

    @patch.object(PerforceProvider, '_run_p4')
    def test_shelve(self, mock_run, tmp_path):
        """Test shelve operation."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Change 12345 files shelved."
        )

        provider = PerforceProvider(str(tmp_path))
        provider._root = str(tmp_path)
        cl_number = provider.shelve()

        assert cl_number == 12345

    @patch.object(PerforceProvider, '_run_p4')
    def test_unshelve(self, mock_run, tmp_path):
        """Test unshelve operation."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")

        provider = PerforceProvider(str(tmp_path))
        provider._root = str(tmp_path)
        result = provider.unshelve(12345)

        assert result is True

    @patch.object(PerforceProvider, '_run_p4')
    def test_get_changelists(self, mock_run, tmp_path):
        """Test getting changelists."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Change 100 on 2024/01/01 by user@client 'Fix bug'\n"
                   "Change 99 on 2024/01/01 by user@client 'Add feature'\n"
        )

        provider = PerforceProvider(str(tmp_path))
        provider._root = str(tmp_path)
        provider._config.user = "user"
        changelists = provider.get_changelists("pending")

        assert len(changelists) == 2
        assert changelists[0].number == 100

    @patch.object(PerforceProvider, '_run_p4')
    def test_get_tags_labels(self, mock_run, tmp_path):
        """Test getting labels as tags."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Label release-1.0 2024/01/01 'Release 1.0'\n"
        )

        provider = PerforceProvider(str(tmp_path))
        provider._root = str(tmp_path)
        tags = provider.get_tags()

        assert len(tags) >= 1

    @patch.object(PerforceProvider, '_run_p4')
    def test_create_tag_label(self, mock_run, tmp_path):
        """Test creating label as tag."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")

        provider = PerforceProvider(str(tmp_path))
        provider._root = str(tmp_path)
        tag = provider.create_tag("v1.0", "Version 1.0")

        assert tag.name == "v1.0"

    @patch.object(PerforceProvider, '_run_p4')
    def test_get_remotes(self, mock_run, tmp_path):
        """Test getting remotes (P4PORT)."""
        config = P4Config(port="ssl:server:1666")
        provider = PerforceProvider(str(tmp_path), config)
        provider._root = str(tmp_path)

        remotes = provider.get_remotes()
        assert len(remotes) == 1
        assert "server" in remotes[0].fetch_url

    def test_is_ancestor(self, tmp_path):
        """Test is_ancestor with changelist numbers."""
        provider = PerforceProvider(str(tmp_path))
        provider._root = str(tmp_path)

        assert provider.is_ancestor("100", "200") is True
        assert provider.is_ancestor("200", "100") is False

    @patch.object(PerforceProvider, '_run_p4')
    def test_diff(self, mock_run, tmp_path):
        """Test diff operation."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="==== //depot/file.txt#1 ====\n- old\n+ new"
        )

        provider = PerforceProvider(str(tmp_path))
        provider._root = str(tmp_path)
        diff = provider.diff("file.txt")

        assert "old" in diff
        assert "new" in diff
