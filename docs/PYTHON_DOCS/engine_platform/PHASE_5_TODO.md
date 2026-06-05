# PHASE 5 TODO: OS Subsystem Enhancements

## Summary

Address gaps in OS abstraction layer: cross-platform system info, memory protection, thread priority.

**Estimated Effort:** 10-14 hours
**Dependencies:** Phase 1 complete
**Blocking:** None (OS is independent)

---

## Tasks

### T-P5-001: Create Platform Detection Module

**Priority:** P0 (Blocking)
**Estimate:** 1 hour

Create `engine/platform/os/_platform/__init__.py`:

```python
import platform

SYSTEM = platform.system()
IS_LINUX = SYSTEM == "Linux"
IS_WINDOWS = SYSTEM == "Windows"
IS_MACOS = SYSTEM == "Darwin"

def get_platform_module():
    if IS_LINUX:
        from . import linux
        return linux
    elif IS_WINDOWS:
        from . import windows
        return windows
    elif IS_MACOS:
        from . import darwin
        return darwin
    return None
```

**Acceptance Criteria:**
- [ ] Correct detection on each platform
- [ ] Returns appropriate module
- [ ] Graceful None on unknown platform

---

### T-P5-002: Create Linux Platform Module

**Priority:** P0 (Blocking)
**Estimate:** 1.5 hours

Create `engine/platform/os/_platform/linux.py`:

```python
import ctypes

_libc = ctypes.CDLL("libc.so.6", use_errno=True)

# Memory protection
PROT_NONE = 0x0
PROT_READ = 0x1
PROT_WRITE = 0x2
PROT_EXEC = 0x4

def mprotect(addr: int, length: int, prot: int) -> bool:
    return _libc.mprotect(addr, length, prot) == 0

# Memory advice
MADV_NORMAL = 0
MADV_RANDOM = 1
MADV_SEQUENTIAL = 2
MADV_WILLNEED = 3
MADV_DONTNEED = 4

def madvise(addr: int, length: int, advice: int) -> bool:
    return _libc.madvise(addr, length, advice) == 0

# Memory info (existing code moved here)
def memory_info() -> tuple[int, int, int, float]:
    # Parse /proc/meminfo
    ...

# CPU info (existing code moved here)
def cpu_count_physical() -> int:
    # Parse /sys/devices/system/cpu/present
    ...
```

**Acceptance Criteria:**
- [ ] mprotect works
- [ ] madvise works
- [ ] memory_info returns accurate values
- [ ] cpu_count_physical works

---

### T-P5-003: Create Windows Platform Module

**Priority:** P0 (Blocking)
**Estimate:** 2 hours

Create `engine/platform/os/_platform/windows.py`:

```python
import ctypes
from ctypes import wintypes

kernel32 = ctypes.WinDLL("kernel32", use_errno=True)

# Memory protection constants
PAGE_NOACCESS = 0x01
PAGE_READONLY = 0x02
PAGE_READWRITE = 0x04
PAGE_EXECUTE = 0x10
PAGE_EXECUTE_READ = 0x20
PAGE_EXECUTE_READWRITE = 0x40

def virtual_protect(addr: int, size: int, new_protect: int) -> tuple[bool, int]:
    old_protect = wintypes.DWORD()
    success = kernel32.VirtualProtect(addr, size, new_protect, ctypes.byref(old_protect))
    return bool(success), old_protect.value

# Memory info
class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]

def memory_info() -> tuple[int, int, int, float]:
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)
    kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    total = status.ullTotalPhys
    avail = status.ullAvailPhys
    used = total - avail
    percent = (used / total * 100) if total > 0 else 0.0
    return (total, avail, used, percent)
```

**Acceptance Criteria:**
- [ ] VirtualProtect works
- [ ] memory_info returns accurate values
- [ ] Works on Windows 10/11

---

### T-P5-004: Create macOS Platform Module

**Priority:** P0 (Blocking)
**Estimate:** 2 hours

Create `engine/platform/os/_platform/darwin.py`:

```python
import ctypes
import subprocess

_libc = ctypes.CDLL("libSystem.B.dylib", use_errno=True)

# Memory protection (same as Linux)
PROT_NONE = 0x0
PROT_READ = 0x1
PROT_WRITE = 0x2
PROT_EXEC = 0x4

def mprotect(addr: int, length: int, prot: int) -> bool:
    return _libc.mprotect(addr, length, prot) == 0

def madvise(addr: int, length: int, advice: int) -> bool:
    return _libc.madvise(addr, length, advice) == 0

# Memory info via sysctl
def memory_info() -> tuple[int, int, int, float]:
    # Use sysctl -n hw.memsize for total
    # Use vm_stat for available/used
    total = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip())
    # Parse vm_stat output for page counts
    ...
```

**Acceptance Criteria:**
- [ ] mprotect works
- [ ] madvise works
- [ ] memory_info returns accurate values
- [ ] Works on macOS 12+

---

### T-P5-005: Integrate mprotect/madvise into VirtualMemory

**Priority:** P0 (Blocking)
**Estimate:** 1.5 hours

Modify `engine/platform/os/virtual_memory.py`:

```python
from ._platform import get_platform_module

class VirtualMemory:
    def protect(self, addr: int, size: int, flags: ProtectionFlags) -> bool:
        plat = get_platform_module()
        if plat is None:
            return False

        if hasattr(plat, 'mprotect'):
            prot = self._flags_to_prot(flags)
            return plat.mprotect(addr, size, prot)
        elif hasattr(plat, 'virtual_protect'):
            protect = self._flags_to_windows_protect(flags)
            success, _ = plat.virtual_protect(addr, size, protect)
            return success
        return False
```

**Acceptance Criteria:**
- [ ] protect() works on Linux/macOS/Windows
- [ ] decommit() uses madvise on Linux/macOS
- [ ] Fallback to no-op on unsupported platforms
- [ ] Unit tests pass

---

### T-P5-006: Integrate Cross-Platform SystemInfo

**Priority:** P0 (Blocking)
**Estimate:** 1 hour

Modify `engine/platform/os/system_info.py`:

```python
from ._platform import get_platform_module, IS_LINUX, IS_WINDOWS, IS_MACOS

class SystemInfo:
    @staticmethod
    def memory_info() -> MemoryInfo:
        plat = get_platform_module()
        if plat and hasattr(plat, 'memory_info'):
            total, avail, used, percent = plat.memory_info()
            return MemoryInfo(total, avail, used, percent)
        return MemoryInfo(0, 0, 0, 0.0)

    @staticmethod
    def cpu_count_physical() -> int:
        plat = get_platform_module()
        if plat and hasattr(plat, 'cpu_count_physical'):
            return plat.cpu_count_physical()
        # Fallback
        return max(1, os.cpu_count() // 2)
```

**Acceptance Criteria:**
- [ ] memory_info() accurate on all platforms
- [ ] cpu_count_physical() accurate on Linux
- [ ] Graceful fallback on other platforms

---

### T-P5-007: Add Cross-Platform Thread Priority

**Priority:** P1 (Important)
**Estimate:** 1.5 hours

Modify `engine/platform/os/threading.py`:

```python
from ._platform import get_platform_module, IS_LINUX, IS_WINDOWS, IS_MACOS

def _set_thread_priority(priority: ThreadPriority) -> bool:
    if IS_WINDOWS:
        import ctypes
        thread = ctypes.windll.kernel32.GetCurrentThread()
        priority_map = {
            ThreadPriority.LOW: -1,
            ThreadPriority.NORMAL: 0,
            ThreadPriority.HIGH: 1,
            ThreadPriority.REALTIME: 15,
        }
        return bool(ctypes.windll.kernel32.SetThreadPriority(thread, priority_map[priority]))
    elif IS_LINUX:
        import os
        # Use nice or sched_setscheduler
        nice_map = {
            ThreadPriority.LOW: 10,
            ThreadPriority.NORMAL: 0,
            ThreadPriority.HIGH: -10,
            ThreadPriority.REALTIME: -20,
        }
        try:
            os.nice(nice_map[priority])
            return True
        except PermissionError:
            return False
    elif IS_MACOS:
        # pthread_setschedparam via ctypes
        ...
    return False
```

**Acceptance Criteria:**
- [ ] Priority setting works on Windows
- [ ] Priority setting works on Linux (may need sudo for HIGH/REALTIME)
- [ ] Priority setting works on macOS
- [ ] Returns False on failure

---

### T-P5-008: Add Optional Watchdog Backend

**Priority:** P2 (Nice to have)
**Estimate:** 2 hours

Modify `engine/platform/os/file_watcher.py`:

```python
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False

class WatchdogFileWatcher(FileWatcher):
    """Efficient file watcher using watchdog library."""

    def __init__(self):
        self._observer = Observer()
        self._handlers: dict[str, FileSystemEventHandler] = {}
        # ...

def create_file_watcher(prefer_watchdog: bool = True) -> FileWatcher:
    if prefer_watchdog and HAS_WATCHDOG:
        return WatchdogFileWatcher()
    return PollingFileWatcher()  # Rename current FileWatcher
```

**Acceptance Criteria:**
- [ ] Watchdog backend works when available
- [ ] Polling backend still works
- [ ] Factory function selects appropriately
- [ ] Same API for both backends

---

### T-P5-009: Write Platform Module Tests

**Priority:** P0 (Blocking)
**Estimate:** 1.5 hours

Create `tests/platform/os/test_platform.py`:

```python
import pytest
from engine.platform.os._platform import IS_LINUX, IS_WINDOWS, IS_MACOS, get_platform_module

def test_platform_detected():
    assert IS_LINUX or IS_WINDOWS or IS_MACOS

def test_platform_module_available():
    assert get_platform_module() is not None

@pytest.mark.skipif(not IS_LINUX, reason="Linux only")
def test_linux_memory_info():
    from engine.platform.os._platform import linux
    total, avail, used, percent = linux.memory_info()
    assert total > 0
    assert avail >= 0
    assert used >= 0

# Similar tests for Windows and macOS
```

**Acceptance Criteria:**
- [ ] Platform detection tested
- [ ] Memory info tested per platform
- [ ] Tests skip appropriately

---

### T-P5-010: Write mprotect/madvise Tests

**Priority:** P1 (Important)
**Estimate:** 1 hour

Create `tests/platform/os/test_virtual_memory_enhanced.py`:

```python
import mmap
import pytest
from engine.platform.os import VirtualMemory, ProtectionFlags

def test_protect_read_write():
    vm = VirtualMemory()
    # Allocate anonymous mmap
    m = mmap.mmap(-1, 4096)
    addr = ctypes.addressof(ctypes.c_char.from_buffer(m))

    # Make read-only
    success = vm.protect(addr, 4096, ProtectionFlags.READ)
    assert success or pytest.skip("mprotect not supported")

    # Writing should fail (SIGSEGV) - can't easily test this

    # Restore read-write
    success = vm.protect(addr, 4096, ProtectionFlags.READ | ProtectionFlags.WRITE)
    assert success

    m.close()
```

**Acceptance Criteria:**
- [ ] protect() tested on supported platforms
- [ ] Tests skip gracefully on unsupported platforms
- [ ] No crashes

---

## Task Dependency Graph

```
T-P5-001 (Platform Detection)
    |
    +-- T-P5-002 (Linux Module)
    |       |
    |       +-- T-P5-005 (VirtualMemory Integration)
    |       +-- T-P5-006 (SystemInfo Integration)
    |
    +-- T-P5-003 (Windows Module)
    |       |
    |       +-- T-P5-005
    |       +-- T-P5-006
    |
    +-- T-P5-004 (macOS Module)
            |
            +-- T-P5-005
            +-- T-P5-006

T-P5-007 (Thread Priority) -- after T-P5-001
T-P5-008 (Watchdog) -- independent
T-P5-009 (Platform Tests) -- after T-P5-002, T-P5-003, T-P5-004
T-P5-010 (VirtualMemory Tests) -- after T-P5-005
```

## Verification Commands

```bash
# Verify platform detection
uv run python -c "from engine.platform.os._platform import SYSTEM, IS_LINUX, IS_WINDOWS, IS_MACOS; print(f'{SYSTEM=}, {IS_LINUX=}, {IS_WINDOWS=}, {IS_MACOS=}')"

# Verify memory info
uv run python -c "from engine.platform.os import SystemInfo; print(SystemInfo.memory_info())"

# Run OS tests
uv run pytest tests/platform/os/ -v

# Test mprotect (Linux/macOS)
uv run python -c "
import mmap
import ctypes
from engine.platform.os import VirtualMemory, ProtectionFlags
m = mmap.mmap(-1, 4096)
vm = VirtualMemory()
print('Protect:', vm.protect(ctypes.addressof(ctypes.c_char.from_buffer(m)), 4096, ProtectionFlags.READ))
m.close()
"
```

## Completion Checklist

- [ ] T-P5-001: Platform detection module created
- [ ] T-P5-002: Linux platform module created
- [ ] T-P5-003: Windows platform module created
- [ ] T-P5-004: macOS platform module created
- [ ] T-P5-005: VirtualMemory uses platform modules
- [ ] T-P5-006: SystemInfo uses platform modules
- [ ] T-P5-007: Thread priority cross-platform
- [ ] T-P5-008: Watchdog backend added
- [ ] T-P5-009: Platform tests pass
- [ ] T-P5-010: VirtualMemory tests pass
