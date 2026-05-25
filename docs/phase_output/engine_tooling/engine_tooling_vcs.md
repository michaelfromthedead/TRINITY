# Investigation: engine/tooling/vcs/

**Classification: REAL**

## Summary

The `engine/tooling/vcs/` module provides a fully functional, production-ready version control system abstraction layer with 3,437 lines of implementation code. This is genuine working code with complete Git and Perforce provider implementations, file locking for binary assets, 3-way merge conflict resolution, and comprehensive file operations.

## Files Analyzed

| File | Lines | Classification | Description |
|------|-------|----------------|-------------|
| `git_provider.py` | 777 | REAL | Full Git VCS provider implementation |
| `perforce_provider.py` | 694 | REAL | Full Perforce/Helix Core provider |
| `lock_manager.py` | 529 | REAL | Binary file locking system |
| `merge_tools.py` | 495 | REAL | 3-way merge conflict resolution |
| `file_operations.py` | 452 | REAL | File status tracking and diff parsing |
| `vcs_integration.py` | 378 | REAL | Abstract VCS provider interface |
| `__init__.py` | 112 | REAL | Module exports |

**Total: 3,437 lines**

---

## Classification Evidence: REAL

### 1. Complete Subprocess Integration

Both providers execute actual shell commands via `subprocess.run()`:

```python
# git_provider.py lines 91-106
def _run_git(
    self,
    args: List[str],
    cwd: Optional[str] = None,
    check: bool = True,
    input_data: Optional[str] = None
) -> subprocess.CompletedProcess:
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
```

### 2. Sophisticated Git Output Parsing

Porcelain format parsing with proper status codes:

```python
# git_provider.py lines 171-192
def _parse_status_code(self, xy: str) -> FileStatus:
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
```

### 3. Comprehensive Blame Parsing

Full blame output parsing with author tracking:

```python
# git_provider.py lines 481-523
def blame(self, path: str, start_line: int = 0, end_line: int = 0) -> List[Tuple[str, str, str]]:
    cmd = ["blame", "--porcelain"]
    if start_line > 0 and end_line > 0:
        cmd.extend(["-L", f"{start_line},{end_line}"])
    cmd.append(path)
    # ... parses commit_id, author, and content from porcelain output
```

### 4. Perforce-Specific Operations

Real Perforce command handling with changelists, shelving, and client specs:

```python
# perforce_provider.py lines 617-638
def shelve(self, changelist: Optional[int] = None, message: str = "") -> int:
    cmd = ["shelve"]
    if changelist:
        cmd.extend(["-c", str(changelist)])
    result = self._run_p4(cmd)
    match = re.search(r"Change (\d+)", result.stdout)
    if match:
        return int(match.group(1))
    raise VCSError("Could not determine shelved changelist number")
```

### 5. 3-Way Merge Algorithm

Actual merge algorithm using `difflib.SequenceMatcher`:

```python
# merge_tools.py lines 84-154
def merge(
    self,
    base: List[str],
    ours: List[str],
    theirs: List[str]
) -> Tuple[List[str], List[ConflictRegion]]:
    matcher_ours = difflib.SequenceMatcher(None, base, ours)
    matcher_theirs = difflib.SequenceMatcher(None, base, theirs)
    # ... full merge logic with conflict detection
```

### 6. Git LFS Integration

Lock manager integrates with Git LFS:

```python
# lock_manager.py lines 360-388
def _git_lfs_lock(self, path: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "lfs", "lock", path],
            cwd=self._provider.root_path,
            capture_output=True,
        )
        return result.returncode == 0
    except Exception:
        return False
```

### 7. Unified Diff Parsing

Complete unified diff format parsing:

```python
# file_operations.py lines 228-318
def parse_diff(self, diff_text: str) -> List[FileDiff]:
    # Parses @@ headers, +/- lines, file headers
    # Tracks old/new line numbers
    # Handles binary files
    # Builds structured FileDiff objects
```

---

## Architecture

### Provider Pattern

```
VCSProvider (Abstract Base)
    |
    +-- GitProvider (subprocess: git)
    |
    +-- PerforceProvider (subprocess: p4)
```

### VCSProviderRegistry

Auto-detection and factory pattern:

```python
@classmethod
def detect_vcs(cls, path: str) -> Optional[VCSType]:
    if os.path.exists(os.path.join(path, ".git")):
        return VCSType.GIT
    elif os.path.exists(os.path.join(path, ".svn")):
        return VCSType.SVN
    # ... checks for .plastic, .hg, P4CLIENT env var
```

### Component Relationships

```
VCSProvider
    |
    +-- FileOperations (status, staging, revert)
    |
    +-- DiffViewer (diff parsing, formatting)
    |
    +-- MergeResolver (conflict detection/resolution)
    |
    +-- LockManager (binary file locking)
```

---

## Capabilities

### Git Provider
- Repository detection and root finding
- Status (porcelain format parsing)
- Commits with message formatting
- Branches (create, delete, checkout, upstream tracking)
- Merges with conflict detection
- Diffs (staged, unstaged, between commits)
- Blame with line ranges
- Tags (lightweight and annotated)
- Remotes (fetch, pull, push)
- Stash operations
- Config get/set

### Perforce Provider
- Connection initialization from env vars and .p4config
- Changelist management (pending, submitted, shelved)
- Sync and submit operations
- Stream-based branching
- Label operations (tag equivalent)
- Client spec parsing
- Depot-to-local path conversion

### Lock Manager
- Exclusive and shared lock types
- Expiration support
- Binary file auto-detection (65+ extensions)
- Lock transfer and break operations
- Batch locking
- Git LFS integration
- JSON persistence

### Merge Tools
- 3-way merge with conflict markers
- Strategy-based resolution (ours, theirs, union, auto)
- Conflict marker parsing (Git and diff3 style)
- Merge preview
- Ancestor detection

---

## Data Structures

### Commit
```python
@dataclass
class Commit:
    id: str
    short_id: str
    message: str
    author: str
    author_email: str
    timestamp: float
    parent_ids: List[str]
    changes: List[ChangeInfo]
    tags: List[str]
```

### LockInfo
```python
@dataclass
class LockInfo:
    path: str
    lock_type: LockType  # EXCLUSIVE, SHARED, INTENT
    state: LockState     # LOCKED, UNLOCKED, PENDING, BREAKING, STOLEN
    owner: str
    expires: Optional[float]
    machine: str
    branch: str
```

### ConflictRegion
```python
@dataclass
class ConflictRegion:
    start_line: int
    end_line: int
    ours_content: List[str]
    theirs_content: List[str]
    base_content: Optional[List[str]]  # diff3 style
    resolved_content: Optional[List[str]]
    resolved: bool
```

---

## Binary File Extensions Supported

Lock manager recognizes 65+ binary extensions:
- **Textures**: png, jpg, jpeg, gif, bmp, tga, tiff, exr, hdr, dds, psd, ai, svg
- **3D Models**: fbx, obj, blend, max, mb, ma, 3ds, dae, gltf, glb
- **Audio**: wav, mp3, ogg, flac, aiff, aac, wma
- **Video**: mp4, avi, mov, mkv, webm, wmv
- **Archives**: zip, tar, gz, 7z, rar
- **Documents**: pdf, doc, docx, xls, xlsx, ppt, pptx
- **Fonts**: ttf, otf, woff, woff2
- **Binaries**: exe, dll, so, dylib, a, lib
- **Game Engine**: uasset, umap (Unreal Engine)

---

## Integration Points

### VCS Type Enum
```python
class VCSType(Enum):
    GIT = auto()
    PERFORCE = auto()
    SVN = auto()         # Registered, not implemented
    PLASTIC_SCM = auto() # Registered, not implemented
    MERCURIAL = auto()   # Registered, not implemented
```

### Module Exports (32 symbols)
- VCS Core: VCSProvider, VCSType, VCSStatus, FileStatus, ChangeType, ChangeInfo, Commit, Branch, Tag, Remote
- Errors: VCSError, VCSNotFoundError, VCSConflictError
- Git: GitProvider, GitConfig, GitStash, GitDiffOptions
- Perforce: PerforceProvider, P4Config, P4Changelist, P4ClientSpec
- File Ops: FileStatusInfo, DiffLine, DiffHunk, FileDiff, FileOperations, DiffViewer
- Merge: MergeStrategy, MergeResult, ConflictInfo, ConflictRegion, ThreeWayMerge, MergeResolver
- Locking: LockType, LockInfo, LockState, LockManager, BinaryFileLockManager

---

## Quality Indicators

| Indicator | Assessment |
|-----------|------------|
| Real subprocess calls | Yes - `subprocess.run()` throughout |
| Error handling | Comprehensive - custom exceptions |
| Edge case handling | Yes - binary files, conflicts, detached HEAD |
| Documentation | Docstrings on all public methods |
| Type annotations | Complete throughout |
| Data validation | Input validation present |
| Resource cleanup | Handled via context managers |

---

## Gaps / Incomplete Areas

1. **SVN/PlasticSCM/Mercurial**: Registered in enum but no provider implementations
2. **Network operations**: No retry logic or timeout handling for remote operations
3. **Large file handling**: No chunked reading for very large files
4. **Concurrent access**: Lock file has thread lock but no cross-process locking

---

## Conclusion

This is a production-quality VCS abstraction layer. The code is fully functional with real subprocess integration, proper output parsing, and comprehensive error handling. The dual Git/Perforce support makes it suitable for both indie and enterprise game development workflows.
