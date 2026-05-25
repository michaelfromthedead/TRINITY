# PHASE 5 ARCHITECTURE: OS Subsystem Enhancements

## Phase Overview

Phase 5 addresses gaps in the OS abstraction layer identified during investigation. The subsystem is largely complete (2,067 lines) but has platform limitations and missing features that should be addressed.

## Current State (from Investigation)

| Component | Status | Lines | Notes |
|-----------|--------|-------|-------|
| file_system.py | REAL | 384 | Result pattern, async I/O, mmap |
| threading.py | REAL | 301 | Full threading primitives |
| file_watcher.py | REAL | 275 | Polling-based watcher |
| virtual_memory.py | REAL (limited) | 231 | No mprotect/madvise |
| atomics.py | REAL | 228 | Lock-based atomics |
| dynamic_library.py | REAL | 185 | ctypes-based loading |
| timing.py | REAL | 174 | Nanosecond precision |
| system_info.py | REAL | 168 | Linux-centric |

**Key Gaps:**
1. mprotect/madvise not available through Python mmap
2. System info fallbacks return 0 on non-Linux
3. Lock-based atomics (Python GIL limitation)
4. No Windows/macOS equivalents for /proc parsing

## Architectural Decisions

### ADR-P5-001: Virtual Memory Enhancement Strategy

**Status:** Proposed

**Context:**
Python's mmap module doesn't expose mprotect or madvise. Options:
1. ctypes to libc (Linux/macOS)
2. ctypes to kernel32 (Windows)
3. Accept limitation, document as such

**Decision:**
Add ctypes-based mprotect/madvise for platforms that support it:

```python
# Linux/macOS
import ctypes

_libc = ctypes.CDLL("libc.so.6", use_errno=True)  # or "libSystem.B.dylib" on macOS

def mprotect(addr: int, length: int, prot: int) -> int:
    return _libc.mprotect(addr, length, prot)

def madvise(addr: int, length: int, advice: int) -> int:
    return _libc.madvise(addr, length, advice)

# Windows
_kernel32 = ctypes.WinDLL("kernel32", use_errno=True)

def virtual_protect(addr: int, size: int, new_protect: int) -> tuple[bool, int]:
    old_protect = ctypes.c_ulong()
    success = _kernel32.VirtualProtect(addr, size, new_protect, ctypes.byref(old_protect))
    return bool(success), old_protect.value
```

**Consequences:**
- Full memory protection control on supported platforms
- Graceful no-op on unsupported platforms
- ctypes dependency (already used in dynamic_library.py)

### ADR-P5-002: Cross-Platform System Info

**Status:** Proposed

**Context:**
System info currently reads /proc/meminfo and /sys/devices. Needs Windows and macOS equivalents.

**Decision:**
Platform-specific implementations with fallback chain:

```python
class SystemInfo:
    @staticmethod
    def memory_info() -> MemoryInfo:
        system = platform.system()
        if system == "Linux":
            return _linux_memory_info()
        elif system == "Darwin":
            return _macos_memory_info()
        elif system == "Windows":
            return _windows_memory_info()
        return MemoryInfo(0, 0, 0, 0.0)

def _windows_memory_info() -> MemoryInfo:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            # ...
        ]
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return MemoryInfo(
        total=status.ullTotalPhys,
        available=status.ullAvailPhys,
        used=status.ullTotalPhys - status.ullAvailPhys,
        percent=(status.ullTotalPhys - status.ullAvailPhys) / status.ullTotalPhys * 100
    )

def _macos_memory_info() -> MemoryInfo:
    # Use vm_stat or sysctl
    ...
```

**Consequences:**
- Accurate memory info on all desktop platforms
- Platform detection at runtime
- Fallback to zeros if detection fails

### ADR-P5-003: Lock-Free Atomics via Shared Memory

**Status:** Deferred

**Context:**
Python GIL makes true lock-free atomics impossible in pure Python. Options:
1. Accept lock-based atomics (current)
2. Use multiprocessing.Value with ctypes
3. Use Rust extension (via renderer-backend)

**Decision:**
Keep lock-based atomics in Python layer. For hot paths requiring true lock-free operations, use the Rust renderer-backend.

**Consequences:**
- Python atomics are correct but not optimal
- Performance-critical code uses Rust
- Clear boundary between Python (coordination) and Rust (performance)

### ADR-P5-004: File Watcher Enhancement

**Status:** Proposed

**Context:**
Polling-based file watcher works but is CPU-intensive for large directories. Options:
1. Keep polling (current)
2. Use inotify (Linux), FSEvents (macOS), ReadDirectoryChangesW (Windows)
3. Use watchdog library

**Decision:**
Add optional watchdog backend for efficient file watching:

```python
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    class WatchdogFileWatcher(FileWatcher):
        def __init__(self):
            self._observer = Observer()
            # ...

    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False

def create_file_watcher(use_watchdog: bool = True) -> FileWatcher:
    if use_watchdog and HAS_WATCHDOG:
        return WatchdogFileWatcher()
    return PollingFileWatcher()
```

**Consequences:**
- Efficient watching when watchdog available
- Fallback to polling
- Optional dependency

### ADR-P5-005: Thread Priority Portability

**Status:** Proposed

**Context:**
Thread priority currently only works on Linux via os.sched_setaffinity. Need Windows/macOS.

**Decision:**
Add platform-specific priority setting:

```python
def _set_thread_priority(priority: ThreadPriority) -> bool:
    system = platform.system()
    if system == "Windows":
        return _set_windows_thread_priority(priority)
    elif system == "Darwin":
        return _set_macos_thread_priority(priority)
    elif system == "Linux":
        return _set_linux_thread_priority(priority)
    return False

def _set_windows_thread_priority(priority: ThreadPriority) -> bool:
    import ctypes
    thread = ctypes.windll.kernel32.GetCurrentThread()
    priority_map = {
        ThreadPriority.LOW: -1,       # THREAD_PRIORITY_BELOW_NORMAL
        ThreadPriority.NORMAL: 0,     # THREAD_PRIORITY_NORMAL
        ThreadPriority.HIGH: 1,       # THREAD_PRIORITY_ABOVE_NORMAL
        ThreadPriority.REALTIME: 15,  # THREAD_PRIORITY_TIME_CRITICAL
    }
    return bool(ctypes.windll.kernel32.SetThreadPriority(thread, priority_map[priority]))
```

**Consequences:**
- Thread priority works on all desktop platforms
- Best-effort (may require elevated privileges)
- Graceful failure returns False

## Component Diagram

```
engine/platform/os/
    |
    +-- file_system.py       # File I/O (unchanged)
    +-- threading.py         # Threading primitives (enhanced priority)
    +-- file_watcher.py      # File watching (optional watchdog)
    +-- virtual_memory.py    # Memory management (add mprotect/madvise)
    +-- atomics.py           # Atomics (unchanged, document limitation)
    +-- dynamic_library.py   # Library loading (unchanged)
    +-- timing.py            # Timing (unchanged)
    +-- system_info.py       # System info (cross-platform)
    |
    +-- _platform/           # NEW: Platform-specific helpers
            |
            +-- __init__.py
            +-- linux.py     # Linux /proc, sysfs
            +-- windows.py   # Windows ctypes
            +-- darwin.py    # macOS ctypes
```

## File Changes Required

### New Files

| File | Purpose |
|------|---------|
| engine/platform/os/_platform/__init__.py | Platform detection, dispatcher |
| engine/platform/os/_platform/linux.py | Linux-specific (current code moved here) |
| engine/platform/os/_platform/windows.py | Windows ctypes implementations |
| engine/platform/os/_platform/darwin.py | macOS ctypes implementations |

### Modified Files

| File | Changes |
|------|---------|
| virtual_memory.py | Add mprotect/madvise via platform module |
| system_info.py | Use platform module for cross-platform |
| threading.py | Add cross-platform priority |
| file_watcher.py | Optional watchdog backend |

## Dependencies

### Optional Python Packages

| Package | Version | Purpose |
|---------|---------|---------|
| watchdog | >=3.0.0 | Efficient file watching |

### Native Libraries (via ctypes)

| Library | Platform | Purpose |
|---------|----------|---------|
| libc.so.6 | Linux | mprotect, madvise |
| libSystem.B.dylib | macOS | mprotect, madvise |
| kernel32.dll | Windows | VirtualProtect, GlobalMemoryStatusEx |

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| ctypes signatures wrong | Test on each platform, match MSDN/man pages |
| Elevated privileges required | Document, graceful failure |
| watchdog dependency optional | Fallback to polling |
| Platform detection edge cases | Use platform.system() consistently |

## Phase Exit Criteria

1. mprotect/madvise work on Linux and macOS
2. VirtualProtect works on Windows
3. MemoryInfo accurate on all desktop platforms
4. Thread priority settable on all platforms
5. File watcher uses watchdog when available
6. All existing tests pass
7. New platform-specific tests added
