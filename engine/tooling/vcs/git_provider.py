"""Git provider implementation.

Provides Git operations including commit, branch, merge, diff, and blame.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import os
import re
import subprocess
import time

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
    VCSProviderRegistry,
)


@dataclass
class GitConfig:
    """Git configuration."""
    user_name: str = ""
    user_email: str = ""
    default_branch: str = "main"
    auto_crlf: str = "input"
    core_autocrlf: bool = False
    push_default: str = "current"


@dataclass
class GitStash:
    """Represents a Git stash entry."""
    index: int
    message: str
    branch: str
    commit_id: str
    timestamp: float = 0.0


@dataclass
class GitDiffOptions:
    """Options for diff operations."""
    context_lines: int = 3
    ignore_whitespace: bool = False
    ignore_blank_lines: bool = False
    word_diff: bool = False
    stat_only: bool = False
    name_only: bool = False


class GitProvider(VCSProvider):
    """Git VCS provider implementation."""

    def __init__(self, path: str, git_executable: str = "git"):
        self._path = os.path.abspath(path)
        self._git = git_executable
        self._root: Optional[str] = None

        # Find repository root
        self._find_root()

    def _find_root(self) -> None:
        """Find the repository root."""
        try:
            result = self._run_git(["rev-parse", "--show-toplevel"], check=False)
            if result.returncode == 0:
                self._root = result.stdout.strip()
        except Exception:
            self._root = None

    def _run_git(
        self,
        args: List[str],
        cwd: Optional[str] = None,
        check: bool = True,
        input_data: Optional[str] = None
    ) -> subprocess.CompletedProcess:
        """Run a git command."""
        cmd = [self._git] + args
        cwd = cwd or self._root or self._path

        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            input=input_data,
        )

        if check and result.returncode != 0:
            raise VCSError(f"Git command failed: {' '.join(cmd)}\n{result.stderr}")

        return result

    @property
    def vcs_type(self) -> VCSType:
        return VCSType.GIT

    @property
    def root_path(self) -> str:
        if not self._root:
            raise VCSNotFoundError("Not a Git repository")
        return self._root

    def is_valid_repository(self) -> bool:
        return self._root is not None

    def get_status(self) -> VCSStatus:
        if not self.is_valid_repository():
            return VCSStatus.UNKNOWN

        result = self._run_git(["status", "--porcelain", "-b"], check=False)
        if result.returncode != 0:
            return VCSStatus.UNKNOWN

        lines = result.stdout.strip().split("\n")
        if not lines:
            return VCSStatus.CLEAN

        # Check branch line for detached state
        if lines[0].startswith("## HEAD (no branch)"):
            return VCSStatus.DETACHED

        # Check for conflicts
        for line in lines[1:]:
            if line and line[0] in ("U", "A") and line[1] in ("U", "A"):
                return VCSStatus.CONFLICTED

        # Check for staged vs modified
        has_staged = False
        has_modified = False
        for line in lines[1:]:
            if line:
                if line[0] != " " and line[0] != "?":
                    has_staged = True
                if line[1] != " " and line[1] != "?":
                    has_modified = True

        if has_staged:
            return VCSStatus.STAGED
        if has_modified:
            return VCSStatus.MODIFIED

        return VCSStatus.CLEAN

    def get_file_status(self, path: str) -> FileStatus:
        result = self._run_git(["status", "--porcelain", "--", path], check=False)
        if result.returncode != 0 or not result.stdout.strip():
            return FileStatus.UNCHANGED

        line = result.stdout.strip()
        if not line:
            return FileStatus.UNCHANGED

        xy = line[:2]
        return self._parse_status_code(xy)

    def _parse_status_code(self, xy: str) -> FileStatus:
        """Parse Git status code to FileStatus."""
        x, y = xy[0], xy[1]

        if x == "?" or y == "?":
            return FileStatus.UNTRACKED
        if x == "!" or y == "!":
            return FileStatus.IGNORED
        if "U" in xy or (x == "A" and y == "A") or (x == "D" and y == "D"):
            return FileStatus.CONFLICTED
        if x == "A" or y == "A":
            return FileStatus.ADDED
        if x == "D" or y == "D":
            return FileStatus.DELETED
        if x == "R" or y == "R":
            return FileStatus.RENAMED
        if x == "C" or y == "C":
            return FileStatus.COPIED
        if x == "M" or y == "M":
            return FileStatus.MODIFIED

        return FileStatus.UNCHANGED

    def get_modified_files(self) -> List[Tuple[str, FileStatus]]:
        result = self._run_git(["status", "--porcelain"], check=False)
        if result.returncode != 0:
            return []

        files = []
        # Don't strip the output - leading spaces are significant in porcelain format
        for line in result.stdout.rstrip("\n").split("\n"):
            if not line:
                continue

            xy = line[:2]
            path = line[3:].split(" -> ")[-1]  # Handle renames
            status = self._parse_status_code(xy)
            files.append((path, status))

        return files

    def commit(self, message: str, files: Optional[List[str]] = None, amend: bool = False) -> Commit:
        # Stage files if specified
        if files:
            self._run_git(["add", "--"] + files)

        # Build commit command
        cmd = ["commit", "-m", message]
        if amend:
            cmd.append("--amend")

        self._run_git(cmd)

        # Return the new commit
        return self.get_commit("HEAD")

    def get_commit(self, commit_id: str) -> Commit:
        # Get commit info
        format_str = "%H%n%h%n%s%n%an%n%ae%n%at%n%P"
        result = self._run_git(["show", "-s", f"--format={format_str}", commit_id])

        lines = result.stdout.strip().split("\n")
        if len(lines) < 6:
            raise VCSError(f"Invalid commit: {commit_id}")

        # Get full message
        msg_result = self._run_git(["log", "-1", "--format=%B", commit_id])

        # Get changes
        changes = self._get_commit_changes(commit_id)

        # Get tags pointing to this commit
        tag_result = self._run_git(["tag", "--points-at", lines[0]], check=False)
        tags = tag_result.stdout.strip().split("\n") if tag_result.stdout.strip() else []

        # Parent IDs may not exist for initial commit (only 6 lines)
        parent_ids = []
        if len(lines) > 6 and lines[6]:
            parent_ids = lines[6].split()

        return Commit(
            id=lines[0],
            short_id=lines[1],
            message=msg_result.stdout.strip(),
            author=lines[3],
            author_email=lines[4],
            timestamp=float(lines[5]),
            parent_ids=parent_ids,
            changes=changes,
            tags=tags,
        )

    def _get_commit_changes(self, commit_id: str) -> List[ChangeInfo]:
        """Get changes in a commit."""
        result = self._run_git([
            "diff-tree", "--no-commit-id", "--name-status", "-r", "--root", commit_id
        ], check=False)

        changes = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue

            status = parts[0][0]
            path = parts[-1]
            old_path = parts[1] if len(parts) > 2 else None

            change_type_map = {
                "A": ChangeType.ADD,
                "M": ChangeType.MODIFY,
                "D": ChangeType.DELETE,
                "R": ChangeType.RENAME,
                "C": ChangeType.COPY,
                "T": ChangeType.TYPE_CHANGE,
            }

            changes.append(ChangeInfo(
                path=path,
                change_type=change_type_map.get(status, ChangeType.MODIFY),
                old_path=old_path,
            ))

        return changes

    def get_commits(
        self,
        count: int = 10,
        branch: Optional[str] = None,
        since: Optional[float] = None,
        until: Optional[float] = None,
        author: Optional[str] = None,
        path: Optional[str] = None
    ) -> List[Commit]:
        cmd = ["log", f"-{count}", "--format=%H"]

        if branch:
            cmd.append(branch)
        if since:
            cmd.append(f"--since={int(since)}")
        if until:
            cmd.append(f"--until={int(until)}")
        if author:
            cmd.append(f"--author={author}")
        if path:
            cmd.extend(["--", path])

        result = self._run_git(cmd, check=False)
        if result.returncode != 0:
            return []

        commits = []
        for commit_id in result.stdout.strip().split("\n"):
            if commit_id:
                try:
                    commits.append(self.get_commit(commit_id))
                except VCSError:
                    pass

        return commits

    def get_current_branch(self) -> Optional[Branch]:
        result = self._run_git(["branch", "--show-current"], check=False)
        if result.returncode != 0 or not result.stdout.strip():
            return None

        name = result.stdout.strip()
        commit_result = self._run_git(["rev-parse", name], check=False)
        commit_id = commit_result.stdout.strip() if commit_result.returncode == 0 else ""

        # Get upstream info
        upstream_result = self._run_git([
            "rev-parse", "--abbrev-ref", f"{name}@{{upstream}}"
        ], check=False)
        upstream = upstream_result.stdout.strip() if upstream_result.returncode == 0 else None

        # Get ahead/behind
        ahead, behind = 0, 0
        if upstream:
            ab_result = self._run_git([
                "rev-list", "--left-right", "--count", f"{name}...{upstream}"
            ], check=False)
            if ab_result.returncode == 0:
                parts = ab_result.stdout.strip().split()
                if len(parts) == 2:
                    ahead, behind = int(parts[0]), int(parts[1])

        return Branch(
            name=name,
            commit_id=commit_id,
            is_current=True,
            upstream=upstream,
            ahead=ahead,
            behind=behind,
        )

    def get_branches(self, include_remote: bool = False) -> List[Branch]:
        cmd = ["branch", "--format=%(refname:short)%09%(objectname)%09%(upstream:short)"]
        if include_remote:
            cmd.append("-a")

        result = self._run_git(cmd, check=False)
        if result.returncode != 0:
            return []

        branches = []
        current_result = self._run_git(["branch", "--show-current"], check=False)
        current_name = current_result.stdout.strip()

        for line in result.stdout.strip().split("\n"):
            if not line:
                continue

            parts = line.split("\t")
            name = parts[0]
            commit_id = parts[1] if len(parts) > 1 else ""
            upstream = parts[2] if len(parts) > 2 and parts[2] else None

            is_remote = name.startswith("remotes/") or "/" in name
            if is_remote:
                name = name.replace("remotes/", "")

            branches.append(Branch(
                name=name,
                commit_id=commit_id,
                is_current=(name == current_name),
                is_remote=is_remote,
                upstream=upstream,
            ))

        return branches

    def create_branch(self, name: str, start_point: Optional[str] = None) -> Branch:
        cmd = ["branch", name]
        if start_point:
            cmd.append(start_point)

        self._run_git(cmd)

        # Get commit ID
        commit_result = self._run_git(["rev-parse", name])
        return Branch(
            name=name,
            commit_id=commit_result.stdout.strip(),
        )

    def delete_branch(self, name: str, force: bool = False) -> bool:
        cmd = ["branch", "-D" if force else "-d", name]
        result = self._run_git(cmd, check=False)
        return result.returncode == 0

    def checkout(self, ref: str, create: bool = False) -> bool:
        cmd = ["checkout"]
        if create:
            cmd.append("-b")
        cmd.append(ref)

        result = self._run_git(cmd, check=False)
        return result.returncode == 0

    def merge(self, branch: str, message: Optional[str] = None) -> Commit:
        cmd = ["merge", branch]
        if message:
            cmd.extend(["-m", message])

        result = self._run_git(cmd, check=False)
        if result.returncode != 0:
            # Check for conflicts
            status_result = self._run_git(["status", "--porcelain"], check=False)
            conflicts = []
            for line in status_result.stdout.strip().split("\n"):
                if line and ("UU" in line[:2] or "AA" in line[:2] or "DD" in line[:2]):
                    conflicts.append(line[3:])

            if conflicts:
                raise VCSConflictError("Merge conflict", conflicts)
            raise VCSError(f"Merge failed: {result.stderr}")

        return self.get_commit("HEAD")

    def get_merge_base(self, ref1: str, ref2: str) -> str:
        result = self._run_git(["merge-base", ref1, ref2])
        return result.stdout.strip()

    def diff(
        self,
        path: Optional[str] = None,
        staged: bool = False,
        commit1: Optional[str] = None,
        commit2: Optional[str] = None
    ) -> str:
        cmd = ["diff"]

        if staged:
            cmd.append("--cached")

        if commit1:
            cmd.append(commit1)
        if commit2:
            cmd.append(commit2)

        if path:
            cmd.extend(["--", path])

        result = self._run_git(cmd, check=False)
        return result.stdout

    def blame(self, path: str, start_line: int = 0, end_line: int = 0) -> List[Tuple[str, str, str]]:
        cmd = ["blame", "--porcelain"]

        if start_line > 0 and end_line > 0:
            cmd.extend(["-L", f"{start_line},{end_line}"])

        cmd.append(path)

        result = self._run_git(cmd, check=False)
        if result.returncode != 0:
            return []

        blame_info = []
        lines = result.stdout.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]
            if not line or line.startswith("\t"):
                i += 1
                continue

            parts = line.split()
            if len(parts) >= 3:
                commit_id = parts[0]

                # Find author and content
                author = ""
                content = ""
                while i < len(lines):
                    i += 1
                    if i >= len(lines):
                        break
                    if lines[i].startswith("author "):
                        author = lines[i][7:]
                    elif lines[i].startswith("\t"):
                        content = lines[i][1:]
                        break

                blame_info.append((commit_id[:8], author, content))
            else:
                i += 1

        return blame_info

    def get_tags(self) -> List[Tag]:
        result = self._run_git([
            "tag", "-l", "--format=%(refname:short)%09%(objectname)%09%(*objectname)"
        ], check=False)

        if result.returncode != 0:
            return []

        tags = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue

            parts = line.split("\t")
            name = parts[0]
            tag_obj = parts[1] if len(parts) > 1 else ""
            commit_obj = parts[2] if len(parts) > 2 and parts[2] else tag_obj

            # Check if annotated
            is_annotated = commit_obj != tag_obj

            tags.append(Tag(
                name=name,
                commit_id=commit_obj,
                is_annotated=is_annotated,
            ))

        return tags

    def create_tag(self, name: str, message: str = "", commit: Optional[str] = None) -> Tag:
        cmd = ["tag"]

        if message:
            cmd.extend(["-a", "-m", message])

        cmd.append(name)

        if commit:
            cmd.append(commit)

        self._run_git(cmd)

        # Get tag info
        commit_result = self._run_git(["rev-parse", f"{name}^{{commit}}"])

        return Tag(
            name=name,
            commit_id=commit_result.stdout.strip(),
            message=message,
            is_annotated=bool(message),
        )

    def delete_tag(self, name: str) -> bool:
        result = self._run_git(["tag", "-d", name], check=False)
        return result.returncode == 0

    def get_remotes(self) -> List[Remote]:
        result = self._run_git(["remote", "-v"], check=False)
        if result.returncode != 0:
            return []

        remotes_dict: Dict[str, Remote] = {}

        for line in result.stdout.strip().split("\n"):
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            name = parts[0]
            url = parts[1]
            is_push = "(push)" in line

            if name not in remotes_dict:
                remotes_dict[name] = Remote(name=name, fetch_url="", push_url="")

            if is_push:
                remotes_dict[name].push_url = url
            else:
                remotes_dict[name].fetch_url = url

        return list(remotes_dict.values())

    def fetch(self, remote: str = "origin", prune: bool = False) -> bool:
        cmd = ["fetch", remote]
        if prune:
            cmd.append("--prune")

        result = self._run_git(cmd, check=False)
        return result.returncode == 0

    def pull(self, remote: str = "origin", branch: Optional[str] = None) -> bool:
        cmd = ["pull", remote]
        if branch:
            cmd.append(branch)

        result = self._run_git(cmd, check=False)
        return result.returncode == 0

    def push(
        self,
        remote: str = "origin",
        branch: Optional[str] = None,
        force: bool = False,
        tags: bool = False
    ) -> bool:
        cmd = ["push", remote]

        if branch:
            cmd.append(branch)
        if force:
            cmd.append("--force")
        if tags:
            cmd.append("--tags")

        result = self._run_git(cmd, check=False)
        return result.returncode == 0

    def add(self, paths: List[str]) -> bool:
        if not paths:
            return True
        result = self._run_git(["add", "--"] + paths, check=False)
        return result.returncode == 0

    def remove(self, paths: List[str], force: bool = False, cached: bool = False) -> bool:
        if not paths:
            return True

        cmd = ["rm"]
        if force:
            cmd.append("-f")
        if cached:
            cmd.append("--cached")
        cmd.extend(["--"] + paths)

        result = self._run_git(cmd, check=False)
        return result.returncode == 0

    def revert(self, paths: List[str]) -> bool:
        if not paths:
            return True

        result = self._run_git(["checkout", "--"] + paths, check=False)
        return result.returncode == 0

    def clean(self, directories: bool = False, force: bool = False, dry_run: bool = True) -> List[str]:
        cmd = ["clean"]
        if directories:
            cmd.append("-d")
        if force:
            cmd.append("-f")
        if dry_run:
            cmd.append("-n")

        result = self._run_git(cmd, check=False)
        if result.returncode != 0:
            return []

        cleaned = []
        for line in result.stdout.strip().split("\n"):
            if line.startswith("Would remove ") or line.startswith("Removing "):
                cleaned.append(line.split()[-1])

        return cleaned

    def resolve_ref(self, ref: str) -> str:
        result = self._run_git(["rev-parse", ref])
        return result.stdout.strip()

    def is_ancestor(self, commit1: str, commit2: str) -> bool:
        result = self._run_git(["merge-base", "--is-ancestor", commit1, commit2], check=False)
        return result.returncode == 0

    # Git-specific operations
    def stash(self, message: str = "") -> GitStash:
        """Create a stash."""
        cmd = ["stash", "push"]
        if message:
            cmd.extend(["-m", message])

        self._run_git(cmd)

        # Get stash info
        result = self._run_git(["stash", "list", "-1", "--format=%H %s"])
        parts = result.stdout.strip().split(" ", 1)

        branch = self.get_current_branch()
        return GitStash(
            index=0,
            message=message or parts[1] if len(parts) > 1 else "",
            branch=branch.name if branch else "",
            commit_id=parts[0] if parts else "",
        )

    def stash_pop(self, index: int = 0) -> bool:
        """Pop a stash."""
        result = self._run_git(["stash", "pop", f"stash@{{{index}}}"], check=False)
        return result.returncode == 0

    def stash_list(self) -> List[GitStash]:
        """List all stashes."""
        result = self._run_git(["stash", "list", "--format=%H %gd %s"], check=False)
        if result.returncode != 0:
            return []

        stashes = []
        for i, line in enumerate(result.stdout.strip().split("\n")):
            if not line:
                continue

            parts = line.split(" ", 2)
            stashes.append(GitStash(
                index=i,
                message=parts[2] if len(parts) > 2 else "",
                branch="",
                commit_id=parts[0] if parts else "",
            ))

        return stashes

    def get_config(self, key: str, scope: str = "local") -> Optional[str]:
        """Get a git config value."""
        cmd = ["config"]
        if scope == "global":
            cmd.append("--global")
        elif scope == "system":
            cmd.append("--system")

        cmd.extend(["--get", key])

        result = self._run_git(cmd, check=False)
        if result.returncode == 0:
            return result.stdout.strip()
        return None

    def set_config(self, key: str, value: str, scope: str = "local") -> bool:
        """Set a git config value."""
        cmd = ["config"]
        if scope == "global":
            cmd.append("--global")
        elif scope == "system":
            cmd.append("--system")

        cmd.extend([key, value])

        result = self._run_git(cmd, check=False)
        return result.returncode == 0


# Register the provider
VCSProviderRegistry.register(VCSType.GIT, GitProvider)
