"""Abstract VCS interface for provider independence.

Provides a unified interface for version control operations that can be
implemented by different VCS providers (Git, Perforce, SVN, etc.).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import time


class VCSType(Enum):
    """Supported VCS types."""
    GIT = auto()
    PERFORCE = auto()
    SVN = auto()
    PLASTIC_SCM = auto()
    MERCURIAL = auto()


class VCSStatus(Enum):
    """VCS repository status."""
    CLEAN = auto()
    MODIFIED = auto()
    STAGED = auto()
    CONFLICTED = auto()
    DETACHED = auto()
    UNKNOWN = auto()


class FileStatus(Enum):
    """Status of a file in the VCS."""
    UNTRACKED = auto()
    ADDED = auto()
    MODIFIED = auto()
    DELETED = auto()
    RENAMED = auto()
    COPIED = auto()
    CONFLICTED = auto()
    IGNORED = auto()
    UNCHANGED = auto()


class ChangeType(Enum):
    """Type of change in a commit."""
    ADD = auto()
    MODIFY = auto()
    DELETE = auto()
    RENAME = auto()
    COPY = auto()
    TYPE_CHANGE = auto()


class VCSError(Exception):
    """Base exception for VCS errors."""
    pass


class VCSNotFoundError(VCSError):
    """Raised when VCS repository is not found."""
    pass


class VCSConflictError(VCSError):
    """Raised when a conflict is detected."""
    def __init__(self, message: str, conflicted_files: Optional[List[str]] = None):
        super().__init__(message)
        self.conflicted_files = conflicted_files or []


@dataclass
class ChangeInfo:
    """Information about a file change."""
    path: str
    change_type: ChangeType
    old_path: Optional[str] = None
    additions: int = 0
    deletions: int = 0
    binary: bool = False


@dataclass
class Commit:
    """Represents a VCS commit."""
    id: str
    short_id: str
    message: str
    author: str
    author_email: str
    timestamp: float
    parent_ids: List[str] = field(default_factory=list)
    changes: List[ChangeInfo] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_merge(self) -> bool:
        """Check if this is a merge commit."""
        return len(self.parent_ids) > 1

    @property
    def title(self) -> str:
        """Get the first line of the commit message."""
        return self.message.split("\n")[0]


@dataclass
class Branch:
    """Represents a VCS branch."""
    name: str
    commit_id: str
    is_current: bool = False
    is_remote: bool = False
    upstream: Optional[str] = None
    ahead: int = 0
    behind: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Tag:
    """Represents a VCS tag."""
    name: str
    commit_id: str
    message: str = ""
    tagger: str = ""
    timestamp: float = 0.0
    is_annotated: bool = False


@dataclass
class Remote:
    """Represents a remote repository."""
    name: str
    fetch_url: str
    push_url: str
    branches: List[str] = field(default_factory=list)


class VCSProvider(ABC):
    """Abstract base class for VCS providers."""

    @property
    @abstractmethod
    def vcs_type(self) -> VCSType:
        """Get the VCS type."""
        pass

    @property
    @abstractmethod
    def root_path(self) -> str:
        """Get the repository root path."""
        pass

    @abstractmethod
    def is_valid_repository(self) -> bool:
        """Check if the current path is a valid repository."""
        pass

    # Status operations
    @abstractmethod
    def get_status(self) -> VCSStatus:
        """Get the overall repository status."""
        pass

    @abstractmethod
    def get_file_status(self, path: str) -> FileStatus:
        """Get the status of a specific file."""
        pass

    @abstractmethod
    def get_modified_files(self) -> List[Tuple[str, FileStatus]]:
        """Get list of modified files with their status."""
        pass

    # Commit operations
    @abstractmethod
    def commit(self, message: str, files: Optional[List[str]] = None, amend: bool = False) -> Commit:
        """Create a new commit."""
        pass

    @abstractmethod
    def get_commit(self, commit_id: str) -> Commit:
        """Get a specific commit."""
        pass

    @abstractmethod
    def get_commits(
        self,
        count: int = 10,
        branch: Optional[str] = None,
        since: Optional[float] = None,
        until: Optional[float] = None,
        author: Optional[str] = None,
        path: Optional[str] = None
    ) -> List[Commit]:
        """Get commit history."""
        pass

    # Branch operations
    @abstractmethod
    def get_current_branch(self) -> Optional[Branch]:
        """Get the current branch."""
        pass

    @abstractmethod
    def get_branches(self, include_remote: bool = False) -> List[Branch]:
        """Get list of branches."""
        pass

    @abstractmethod
    def create_branch(self, name: str, start_point: Optional[str] = None) -> Branch:
        """Create a new branch."""
        pass

    @abstractmethod
    def delete_branch(self, name: str, force: bool = False) -> bool:
        """Delete a branch."""
        pass

    @abstractmethod
    def checkout(self, ref: str, create: bool = False) -> bool:
        """Checkout a branch or commit."""
        pass

    # Merge operations
    @abstractmethod
    def merge(self, branch: str, message: Optional[str] = None) -> Commit:
        """Merge a branch into current branch."""
        pass

    @abstractmethod
    def get_merge_base(self, ref1: str, ref2: str) -> str:
        """Get the merge base of two refs."""
        pass

    # Diff operations
    @abstractmethod
    def diff(
        self,
        path: Optional[str] = None,
        staged: bool = False,
        commit1: Optional[str] = None,
        commit2: Optional[str] = None
    ) -> str:
        """Get diff output."""
        pass

    @abstractmethod
    def blame(self, path: str, start_line: int = 0, end_line: int = 0) -> List[Tuple[str, str, str]]:
        """Get blame information for a file."""
        pass

    # Tag operations
    @abstractmethod
    def get_tags(self) -> List[Tag]:
        """Get list of tags."""
        pass

    @abstractmethod
    def create_tag(self, name: str, message: str = "", commit: Optional[str] = None) -> Tag:
        """Create a new tag."""
        pass

    @abstractmethod
    def delete_tag(self, name: str) -> bool:
        """Delete a tag."""
        pass

    # Remote operations
    @abstractmethod
    def get_remotes(self) -> List[Remote]:
        """Get list of remotes."""
        pass

    @abstractmethod
    def fetch(self, remote: str = "origin", prune: bool = False) -> bool:
        """Fetch from remote."""
        pass

    @abstractmethod
    def pull(self, remote: str = "origin", branch: Optional[str] = None) -> bool:
        """Pull from remote."""
        pass

    @abstractmethod
    def push(
        self,
        remote: str = "origin",
        branch: Optional[str] = None,
        force: bool = False,
        tags: bool = False
    ) -> bool:
        """Push to remote."""
        pass

    # File operations
    @abstractmethod
    def add(self, paths: List[str]) -> bool:
        """Add files to staging."""
        pass

    @abstractmethod
    def remove(self, paths: List[str], force: bool = False, cached: bool = False) -> bool:
        """Remove files."""
        pass

    @abstractmethod
    def revert(self, paths: List[str]) -> bool:
        """Revert file changes."""
        pass

    @abstractmethod
    def clean(self, directories: bool = False, force: bool = False, dry_run: bool = True) -> List[str]:
        """Clean untracked files."""
        pass

    # Utility operations
    @abstractmethod
    def resolve_ref(self, ref: str) -> str:
        """Resolve a reference to a commit ID."""
        pass

    @abstractmethod
    def is_ancestor(self, commit1: str, commit2: str) -> bool:
        """Check if commit1 is an ancestor of commit2."""
        pass


class VCSProviderRegistry:
    """Registry for VCS providers."""

    _providers: Dict[VCSType, type] = {}

    @classmethod
    def register(cls, vcs_type: VCSType, provider_class: type) -> None:
        """Register a provider class."""
        cls._providers[vcs_type] = provider_class

    @classmethod
    def get_provider(cls, vcs_type: VCSType, path: str, **kwargs) -> VCSProvider:
        """Get a provider instance."""
        provider_class = cls._providers.get(vcs_type)
        if not provider_class:
            raise ValueError(f"No provider registered for {vcs_type}")
        return provider_class(path, **kwargs)

    @classmethod
    def detect_vcs(cls, path: str) -> Optional[VCSType]:
        """Detect the VCS type for a path."""
        import os

        # Check for common VCS directories
        if os.path.exists(os.path.join(path, ".git")):
            return VCSType.GIT
        elif os.path.exists(os.path.join(path, ".svn")):
            return VCSType.SVN
        elif os.path.exists(os.path.join(path, ".plastic")):
            return VCSType.PLASTIC_SCM
        elif os.path.exists(os.path.join(path, ".hg")):
            return VCSType.MERCURIAL

        # Check for Perforce by looking for workspace file or env var
        if os.environ.get("P4CLIENT") or os.path.exists(os.path.join(path, ".p4config")):
            return VCSType.PERFORCE

        return None

    @classmethod
    def auto_provider(cls, path: str, **kwargs) -> Optional[VCSProvider]:
        """Auto-detect and create a provider for the path."""
        vcs_type = cls.detect_vcs(path)
        if vcs_type and vcs_type in cls._providers:
            return cls.get_provider(vcs_type, path, **kwargs)
        return None
