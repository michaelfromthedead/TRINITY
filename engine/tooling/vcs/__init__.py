"""Version Control System integration for AI Game Engine.

This module provides VCS integration with support for multiple providers
including Git, Perforce, SVN, and Plastic SCM.

Components:
- VCSProvider: Abstract interface for VCS operations
- GitProvider: Git integration with full repository operations
- PerforceProvider: Perforce/Helix Core integration
- FileOperations: File status, revert, and diff viewing
- MergeTools: Merge conflict resolution with 3-way merge
- LockManager: File locking for binary assets
"""
from __future__ import annotations

from .vcs_integration import (
    VCSProvider,
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
)
from .git_provider import (
    GitProvider,
    GitConfig,
    GitStash,
    GitDiffOptions,
)
from .perforce_provider import (
    PerforceProvider,
    P4Config,
    P4Changelist,
    P4ClientSpec,
)
from .file_operations import (
    FileStatusInfo,
    DiffLine,
    DiffHunk,
    FileDiff,
    FileOperations,
    DiffViewer,
)
from .merge_tools import (
    MergeStrategy,
    MergeResult,
    ConflictInfo,
    ConflictRegion,
    ThreeWayMerge,
    MergeResolver,
)
from .lock_manager import (
    LockType,
    LockInfo,
    LockState,
    LockManager,
    BinaryFileLockManager,
)

__all__ = [
    # VCS Integration
    "VCSProvider",
    "VCSType",
    "VCSStatus",
    "FileStatus",
    "ChangeType",
    "ChangeInfo",
    "Commit",
    "Branch",
    "Tag",
    "Remote",
    "VCSError",
    "VCSNotFoundError",
    "VCSConflictError",
    # Git Provider
    "GitProvider",
    "GitConfig",
    "GitStash",
    "GitDiffOptions",
    # Perforce Provider
    "PerforceProvider",
    "P4Config",
    "P4Changelist",
    "P4ClientSpec",
    # File Operations
    "FileStatusInfo",
    "DiffLine",
    "DiffHunk",
    "FileDiff",
    "FileOperations",
    "DiffViewer",
    # Merge Tools
    "MergeStrategy",
    "MergeResult",
    "ConflictInfo",
    "ConflictRegion",
    "ThreeWayMerge",
    "MergeResolver",
    # Lock Manager
    "LockType",
    "LockInfo",
    "LockState",
    "LockManager",
    "BinaryFileLockManager",
]
