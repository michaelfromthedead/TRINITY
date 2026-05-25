"""Perforce/Helix Core provider implementation.

Provides Perforce operations including checkout, submit, sync, and shelve.
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
class P4Config:
    """Perforce configuration."""
    port: str = ""
    user: str = ""
    client: str = ""
    password: str = ""
    charset: str = "utf8"
    host: str = ""


@dataclass
class P4Changelist:
    """Represents a Perforce changelist."""
    number: int
    status: str  # pending, submitted, shelved
    description: str
    user: str
    client: str
    timestamp: float = 0.0
    files: List[str] = field(default_factory=list)
    jobs: List[str] = field(default_factory=list)


@dataclass
class P4ClientSpec:
    """Perforce client specification."""
    client: str
    owner: str
    host: str
    root: str
    options: List[str] = field(default_factory=list)
    view: List[Tuple[str, str]] = field(default_factory=list)


class PerforceProvider(VCSProvider):
    """Perforce/Helix Core VCS provider implementation."""

    def __init__(self, path: str, config: Optional[P4Config] = None):
        self._path = os.path.abspath(path)
        self._config = config or P4Config()
        self._root: Optional[str] = None
        self._client_spec: Optional[P4ClientSpec] = None

        # Initialize connection
        self._init_connection()

    def _init_connection(self) -> None:
        """Initialize Perforce connection."""
        # Load config from environment or .p4config
        if not self._config.port:
            self._config.port = os.environ.get("P4PORT", "")
        if not self._config.user:
            self._config.user = os.environ.get("P4USER", "")
        if not self._config.client:
            self._config.client = os.environ.get("P4CLIENT", "")

        # Check for .p4config file
        p4config_path = os.path.join(self._path, ".p4config")
        if os.path.exists(p4config_path):
            self._load_p4config(p4config_path)

        # Try to get client root
        try:
            result = self._run_p4(["info"], check=False)
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if line.startswith("Client root:"):
                        self._root = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass

    def _load_p4config(self, path: str) -> None:
        """Load .p4config file."""
        try:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip().upper()
                        value = value.strip()

                        if key == "P4PORT":
                            self._config.port = value
                        elif key == "P4USER":
                            self._config.user = value
                        elif key == "P4CLIENT":
                            self._config.client = value
        except Exception:
            pass

    def _run_p4(
        self,
        args: List[str],
        cwd: Optional[str] = None,
        check: bool = True,
        input_data: Optional[str] = None
    ) -> subprocess.CompletedProcess:
        """Run a p4 command."""
        cmd = ["p4"]

        # Add connection options
        if self._config.port:
            cmd.extend(["-p", self._config.port])
        if self._config.user:
            cmd.extend(["-u", self._config.user])
        if self._config.client:
            cmd.extend(["-c", self._config.client])

        cmd.extend(args)
        cwd = cwd or self._root or self._path

        env = os.environ.copy()
        if self._config.password:
            env["P4PASSWD"] = self._config.password
        if self._config.charset:
            env["P4CHARSET"] = self._config.charset

        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            input=input_data,
            env=env,
        )

        if check and result.returncode != 0:
            raise VCSError(f"P4 command failed: {' '.join(cmd)}\n{result.stderr}")

        return result

    @property
    def vcs_type(self) -> VCSType:
        return VCSType.PERFORCE

    @property
    def root_path(self) -> str:
        if not self._root:
            raise VCSNotFoundError("Not in a Perforce workspace")
        return self._root

    def is_valid_repository(self) -> bool:
        return self._root is not None

    def get_status(self) -> VCSStatus:
        if not self.is_valid_repository():
            return VCSStatus.UNKNOWN

        # Check for opened files
        result = self._run_p4(["opened", "-s"], check=False)
        if result.returncode != 0:
            return VCSStatus.UNKNOWN

        if not result.stdout.strip():
            return VCSStatus.CLEAN

        # Check for conflicts
        if "can't" in result.stdout.lower() or "conflict" in result.stdout.lower():
            return VCSStatus.CONFLICTED

        return VCSStatus.MODIFIED

    def get_file_status(self, path: str) -> FileStatus:
        # Check if file is opened
        result = self._run_p4(["opened", path], check=False)
        if result.returncode != 0 or not result.stdout.strip():
            # Check if file is in depot
            fstat_result = self._run_p4(["fstat", path], check=False)
            if fstat_result.returncode != 0:
                return FileStatus.UNTRACKED
            return FileStatus.UNCHANGED

        output = result.stdout.lower()
        if "add" in output:
            return FileStatus.ADDED
        elif "delete" in output:
            return FileStatus.DELETED
        elif "edit" in output:
            return FileStatus.MODIFIED
        elif "move/add" in output or "move/delete" in output:
            return FileStatus.RENAMED

        return FileStatus.MODIFIED

    def get_modified_files(self) -> List[Tuple[str, FileStatus]]:
        result = self._run_p4(["opened"], check=False)
        if result.returncode != 0:
            return []

        files = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue

            # Parse: //depot/path#rev - action change (type)
            match = re.match(r"^(.+?)#\d+ - (\w+)", line)
            if match:
                depot_path = match.group(1)
                action = match.group(2).lower()

                status = FileStatus.MODIFIED
                if action == "add":
                    status = FileStatus.ADDED
                elif action == "delete":
                    status = FileStatus.DELETED
                elif "move" in action:
                    status = FileStatus.RENAMED

                # Convert depot path to local path
                local_path = self._depot_to_local(depot_path)
                if local_path:
                    files.append((local_path, status))

        return files

    def _depot_to_local(self, depot_path: str) -> Optional[str]:
        """Convert depot path to local path."""
        result = self._run_p4(["where", depot_path], check=False)
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split()
            if len(parts) >= 3:
                return parts[2]
        return None

    def _local_to_depot(self, local_path: str) -> Optional[str]:
        """Convert local path to depot path."""
        result = self._run_p4(["where", local_path], check=False)
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split()
            if parts:
                return parts[0]
        return None

    def commit(self, message: str, files: Optional[List[str]] = None, amend: bool = False) -> Commit:
        # In Perforce, commit = submit
        # First, create or update a changelist
        if files:
            # Move files to default changelist or specified one
            self._run_p4(["reopen", "-c", "default"] + files)

        # Get default changelist description
        change_spec = f"Change: new\n\nDescription:\n\t{message}\n"

        # Submit
        result = self._run_p4(["submit", "-d", message], check=False)
        if result.returncode != 0:
            raise VCSError(f"Submit failed: {result.stderr}")

        # Parse submitted changelist number
        match = re.search(r"Change (\d+) submitted", result.stdout)
        if match:
            cl_number = int(match.group(1))
            return self._changelist_to_commit(cl_number)

        raise VCSError("Could not determine submitted changelist number")

    def _changelist_to_commit(self, cl_number: int) -> Commit:
        """Convert a Perforce changelist to a Commit object."""
        result = self._run_p4(["describe", "-s", str(cl_number)])

        lines = result.stdout.split("\n")
        description = ""
        user = ""
        timestamp = 0.0
        changes = []

        in_description = False
        for line in lines:
            if line.startswith("Change"):
                parts = line.split()
                if len(parts) >= 5:
                    user = parts[3]
            elif line.startswith("Date"):
                # Parse date
                date_str = line.split(":", 1)[1].strip() if ":" in line else ""
                # Simplified timestamp
                timestamp = time.time()
            elif line.strip() == "" and not in_description:
                in_description = True
            elif in_description and not line.startswith("Affected files"):
                description += line.strip() + "\n"
            elif line.startswith("..."):
                # File entry
                match = re.match(r"^\.\.\. (.+?)#(\d+) (\w+)", line)
                if match:
                    change_type = ChangeType.MODIFY
                    action = match.group(3).lower()
                    if action == "add":
                        change_type = ChangeType.ADD
                    elif action == "delete":
                        change_type = ChangeType.DELETE

                    changes.append(ChangeInfo(
                        path=match.group(1),
                        change_type=change_type,
                    ))

        return Commit(
            id=str(cl_number),
            short_id=str(cl_number),
            message=description.strip(),
            author=user,
            author_email=f"{user}@perforce",
            timestamp=timestamp,
            changes=changes,
        )

    def get_commit(self, commit_id: str) -> Commit:
        try:
            cl_number = int(commit_id)
        except ValueError:
            raise VCSError(f"Invalid changelist number: {commit_id}")

        return self._changelist_to_commit(cl_number)

    def get_commits(
        self,
        count: int = 10,
        branch: Optional[str] = None,
        since: Optional[float] = None,
        until: Optional[float] = None,
        author: Optional[str] = None,
        path: Optional[str] = None
    ) -> List[Commit]:
        cmd = ["changes", "-s", "submitted", f"-m{count}"]

        if author:
            cmd.extend(["-u", author])
        if path:
            cmd.append(path)
        else:
            cmd.append("//...")

        result = self._run_p4(cmd, check=False)
        if result.returncode != 0:
            return []

        commits = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue

            match = re.match(r"Change (\d+)", line)
            if match:
                try:
                    commits.append(self._changelist_to_commit(int(match.group(1))))
                except VCSError:
                    pass

        return commits

    def get_current_branch(self) -> Optional[Branch]:
        # Perforce doesn't have branches in the same way Git does
        # Return the stream if using streams, or None otherwise
        result = self._run_p4(["info"], check=False)
        if result.returncode != 0:
            return None

        for line in result.stdout.split("\n"):
            if line.startswith("Client stream:"):
                stream = line.split(":", 1)[1].strip()
                return Branch(
                    name=stream,
                    commit_id="",
                    is_current=True,
                )

        return None

    def get_branches(self, include_remote: bool = False) -> List[Branch]:
        # Get streams if using streams depot
        result = self._run_p4(["streams", "-F", "Type=mainline | Type=development | Type=release"], check=False)
        if result.returncode != 0:
            return []

        branches = []
        for line in result.stdout.strip().split("\n"):
            if line.startswith("Stream"):
                parts = line.split()
                if len(parts) >= 2:
                    branches.append(Branch(
                        name=parts[1],
                        commit_id="",
                    ))

        return branches

    def create_branch(self, name: str, start_point: Optional[str] = None) -> Branch:
        # In Perforce, this would create a stream or use branching
        raise NotImplementedError("Branch creation requires stream setup in Perforce")

    def delete_branch(self, name: str, force: bool = False) -> bool:
        raise NotImplementedError("Branch deletion requires admin privileges in Perforce")

    def checkout(self, ref: str, create: bool = False) -> bool:
        # In Perforce, checkout means opening files for edit
        # Switching streams requires p4 switch
        result = self._run_p4(["switch", ref], check=False)
        return result.returncode == 0

    def merge(self, branch: str, message: Optional[str] = None) -> Commit:
        # Perforce merge is done via p4 merge or p4 integrate
        result = self._run_p4(["merge", "-b", branch], check=False)
        if result.returncode != 0:
            raise VCSError(f"Merge failed: {result.stderr}")

        # Auto-resolve where possible
        self._run_p4(["resolve", "-am"], check=False)

        # Check for remaining conflicts
        resolve_result = self._run_p4(["resolve", "-n"], check=False)
        if resolve_result.stdout.strip():
            raise VCSConflictError("Merge conflicts require manual resolution")

        return self.commit(message or f"Merge {branch}")

    def get_merge_base(self, ref1: str, ref2: str) -> str:
        # Perforce doesn't have merge-base concept
        raise NotImplementedError("Merge base not directly supported in Perforce")

    def diff(
        self,
        path: Optional[str] = None,
        staged: bool = False,
        commit1: Optional[str] = None,
        commit2: Optional[str] = None
    ) -> str:
        cmd = ["diff"]

        if commit1 and commit2:
            cmd.append(f"{path or '//...'}@{commit1},@{commit2}")
        elif path:
            cmd.append(path)
        else:
            # Diff all opened files
            cmd.append("//...")

        result = self._run_p4(cmd, check=False)
        return result.stdout

    def blame(self, path: str, start_line: int = 0, end_line: int = 0) -> List[Tuple[str, str, str]]:
        result = self._run_p4(["annotate", "-c", path], check=False)
        if result.returncode != 0:
            return []

        blame_info = []
        for line in result.stdout.split("\n"):
            if not line:
                continue

            # Format: changelist: content
            if ":" in line:
                cl, content = line.split(":", 1)
                blame_info.append((cl.strip(), "", content))

        return blame_info

    def get_tags(self) -> List[Tag]:
        # Perforce uses labels instead of tags
        result = self._run_p4(["labels"], check=False)
        if result.returncode != 0:
            return []

        tags = []
        for line in result.stdout.strip().split("\n"):
            if line.startswith("Label"):
                parts = line.split()
                if len(parts) >= 2:
                    tags.append(Tag(
                        name=parts[1],
                        commit_id="",
                    ))

        return tags

    def create_tag(self, name: str, message: str = "", commit: Optional[str] = None) -> Tag:
        # Create a label
        label_spec = f"Label: {name}\n\nDescription:\n\t{message}\n\nView:\n\t//...\n"

        self._run_p4(["label", "-i"], input_data=label_spec)

        if commit:
            self._run_p4(["labelsync", "-l", name, f"@{commit}"])

        return Tag(name=name, commit_id=commit or "", message=message)

    def delete_tag(self, name: str) -> bool:
        result = self._run_p4(["label", "-d", name], check=False)
        return result.returncode == 0

    def get_remotes(self) -> List[Remote]:
        # Perforce server is configured via P4PORT
        return [Remote(
            name="origin",
            fetch_url=self._config.port,
            push_url=self._config.port,
        )]

    def fetch(self, remote: str = "origin", prune: bool = False) -> bool:
        # In Perforce, this would be sync
        return True

    def pull(self, remote: str = "origin", branch: Optional[str] = None) -> bool:
        # Sync to latest
        result = self._run_p4(["sync"], check=False)
        return result.returncode == 0

    def push(
        self,
        remote: str = "origin",
        branch: Optional[str] = None,
        force: bool = False,
        tags: bool = False
    ) -> bool:
        # In Perforce, submit is the equivalent of push
        # This is handled by commit()
        return True

    def add(self, paths: List[str]) -> bool:
        if not paths:
            return True
        result = self._run_p4(["add"] + paths, check=False)
        return result.returncode == 0

    def remove(self, paths: List[str], force: bool = False, cached: bool = False) -> bool:
        if not paths:
            return True
        result = self._run_p4(["delete"] + paths, check=False)
        return result.returncode == 0

    def revert(self, paths: List[str]) -> bool:
        if not paths:
            return True
        result = self._run_p4(["revert"] + paths, check=False)
        return result.returncode == 0

    def clean(self, directories: bool = False, force: bool = False, dry_run: bool = True) -> List[str]:
        # Perforce doesn't have clean in the same way
        # Use reconcile to find files that need to be cleaned
        cmd = ["reconcile", "-n"]
        result = self._run_p4(cmd, check=False)

        files = []
        for line in result.stdout.strip().split("\n"):
            if "delete" in line.lower():
                parts = line.split()
                if parts:
                    files.append(parts[0])

        return files

    def resolve_ref(self, ref: str) -> str:
        # In Perforce, refs are changelist numbers
        return ref

    def is_ancestor(self, commit1: str, commit2: str) -> bool:
        # Check if commit1 is earlier than commit2
        try:
            return int(commit1) < int(commit2)
        except ValueError:
            return False

    # Perforce-specific operations
    def sync(self, path: Optional[str] = None, revision: Optional[str] = None) -> bool:
        """Sync files from depot."""
        cmd = ["sync"]
        target = path or "//..."

        if revision:
            target += f"@{revision}"

        cmd.append(target)
        result = self._run_p4(cmd, check=False)
        return result.returncode == 0

    def edit(self, paths: List[str]) -> bool:
        """Open files for edit."""
        if not paths:
            return True
        result = self._run_p4(["edit"] + paths, check=False)
        return result.returncode == 0

    def shelve(self, changelist: Optional[int] = None, message: str = "") -> int:
        """Shelve files."""
        cmd = ["shelve"]
        if changelist:
            cmd.extend(["-c", str(changelist)])

        result = self._run_p4(cmd)

        match = re.search(r"Change (\d+)", result.stdout)
        if match:
            return int(match.group(1))

        raise VCSError("Could not determine shelved changelist number")

    def unshelve(self, changelist: int, into_changelist: Optional[int] = None) -> bool:
        """Unshelve files."""
        cmd = ["unshelve", "-s", str(changelist)]
        if into_changelist:
            cmd.extend(["-c", str(into_changelist)])

        result = self._run_p4(cmd, check=False)
        return result.returncode == 0

    def get_changelists(self, status: str = "pending") -> List[P4Changelist]:
        """Get changelists."""
        cmd = ["changes", "-s", status, "-u", self._config.user]
        result = self._run_p4(cmd, check=False)

        if result.returncode != 0:
            return []

        changelists = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue

            match = re.match(r"Change (\d+) on .+ by (.+)@(.+) '(.+)'", line)
            if match:
                changelists.append(P4Changelist(
                    number=int(match.group(1)),
                    status=status,
                    description=match.group(4),
                    user=match.group(2),
                    client=match.group(3),
                ))

        return changelists

    def get_client_spec(self) -> Optional[P4ClientSpec]:
        """Get the current client specification."""
        result = self._run_p4(["client", "-o"], check=False)
        if result.returncode != 0:
            return None

        spec = P4ClientSpec(client="", owner="", host="", root="")
        view = []

        for line in result.stdout.split("\n"):
            if line.startswith("Client:"):
                spec.client = line.split(":", 1)[1].strip()
            elif line.startswith("Owner:"):
                spec.owner = line.split(":", 1)[1].strip()
            elif line.startswith("Host:"):
                spec.host = line.split(":", 1)[1].strip()
            elif line.startswith("Root:"):
                spec.root = line.split(":", 1)[1].strip()
            elif line.startswith("\t//"):
                # View mapping
                parts = line.strip().split()
                if len(parts) >= 2:
                    view.append((parts[0], parts[1]))

        spec.view = view
        return spec


# Register the provider
VCSProviderRegistry.register(VCSType.PERFORCE, PerforceProvider)
