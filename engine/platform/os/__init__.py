"""
Platform OS abstraction layer.

Provides cross-platform interfaces for:
- File system operations (sync/async, memory-mapped)
- Threading primitives (threads, locks, semaphores)
- Atomic operations (lock-free primitives)
- Virtual memory management
- Dynamic library loading
- System information
- High-resolution timing
- File watching
"""

from .file_system import (
    FileSystem,
    FileHandle,
    FileMode,
    MapMode,
    MappedFile,
    Result,
)

from .threading import (
    Thread,
    ThreadConfig,
    ThreadPriority,
    Mutex,
    RWLock,
    Semaphore,
    CondVar,
    Barrier,
    ThreadLocalStorage,
)

from .atomics import (
    AtomicInt,
    AtomicFloat,
    AtomicBool,
    AtomicRef,
)

from .virtual_memory import (
    VirtualMemory,
    ProtectionFlags,
    MemoryStats,
    page_size,
)

from .dynamic_library import (
    DynamicLibrary,
    LibraryHandle,
)

from .system_info import (
    SystemInfo,
    CPUInfo,
    MemoryInfo,
)

from .timing import (
    Timer,
    Stopwatch,
)

from .file_watcher import (
    FileWatcher,
    FileEvent,
    FileEventData,
)

__all__ = [
    # File system
    'FileSystem',
    'FileHandle',
    'FileMode',
    'MapMode',
    'MappedFile',
    'Result',

    # Threading
    'Thread',
    'ThreadConfig',
    'ThreadPriority',
    'Mutex',
    'RWLock',
    'Semaphore',
    'CondVar',
    'Barrier',
    'ThreadLocalStorage',

    # Atomics
    'AtomicInt',
    'AtomicFloat',
    'AtomicBool',
    'AtomicRef',

    # Virtual memory
    'VirtualMemory',
    'ProtectionFlags',
    'MemoryStats',
    'page_size',

    # Dynamic library
    'DynamicLibrary',
    'LibraryHandle',

    # System info
    'SystemInfo',
    'CPUInfo',
    'MemoryInfo',

    # Timing
    'Timer',
    'Stopwatch',

    # File watching
    'FileWatcher',
    'FileEvent',
    'FileEventData',
]
