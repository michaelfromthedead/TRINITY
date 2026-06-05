# PHASE 1 ARCHITECTURE: Platform Backend Registry and Constants

## Phase Overview

Phase 1 establishes the foundation for all platform subsystems: the backend registry pattern and centralized constants. This phase is prerequisite to all other phases.

## Architectural Decisions

### ADR-P1-001: Backend Registry Pattern

**Status:** Confirmed (already implemented in audio/)

**Context:**
All platform subsystems need pluggable backends. The audio subsystem already implements a registry pattern in `backends/__init__.py`.

**Decision:**
Standardize the backend registry pattern across all subsystems:

```python
class BackendRegistry(Generic[T]):
    """Generic backend registry for platform subsystems."""

    def __init__(self):
        self._backends: dict[str, type[T]] = {}
        self._default: str | None = None

    def register(self, name: str, backend_cls: type[T], set_default: bool = False) -> None:
        self._backends[name] = backend_cls
        if set_default:
            self._default = name

    def get(self, name: str) -> type[T] | None:
        return self._backends.get(name)

    def create(self, name: str | None = None, *args, **kwargs) -> T:
        name = name or self._default
        if name is None:
            raise ValueError("No default backend set")
        cls = self._backends.get(name)
        if cls is None:
            raise ValueError(f"Unknown backend: {name}")
        return cls(*args, **kwargs)

    def list(self) -> list[str]:
        return list(self._backends.keys())
```

**Consequences:**
- All subsystems can be tested with null/headless backends
- Platform-specific code registers itself at import time
- Runtime backend selection via environment variables or config

### ADR-P1-002: Constants Centralization

**Status:** Confirmed (engine/platform/constants.py exists)

**Context:**
Magic numbers scattered across files cause maintenance burden and inconsistency.

**Decision:**
All platform layer constants live in `engine/platform/constants.py`:

| Category | Constants |
|----------|-----------|
| Audio | Sample rates, buffer sizes, channel counts, spatial distances |
| GPU | Handle ranges (buffer, texture, sampler, shader, pipeline) |
| Input | Deadzone defaults, max touches |
| Timing | Ticks per second, nanoseconds per millisecond |
| Display | Standard resolutions, refresh rates, HDR luminance |
| VRR | Default min/max Hz |
| OS | Hyperthreading ratio |

**Consequences:**
- Single source of truth for defaults
- Easy to tune for different platforms
- Self-documenting via constant names

### ADR-P1-003: Handle Allocation Scheme

**Status:** Confirmed (already implemented in rhi/)

**Context:**
Resources need unique handles for tracking across Python/Rust boundary.

**Decision:**
Non-overlapping handle ranges per resource type:

```python
BUFFER_HANDLE_START = 1
TEXTURE_HANDLE_START = 1000
SAMPLER_HANDLE_START = 2000
SHADER_HANDLE_START = 3000
PIPELINE_HANDLE_START = 4000
GPU_ADDRESS_START = 0x100000000  # 4GB offset
```

Each resource type has a class-level `_next_handle` counter protected by a lock:

```python
class NullBuffer:
    _handle_counter = BUFFER_HANDLE_START
    _handle_lock = threading.Lock()

    @classmethod
    def _next_handle(cls) -> int:
        with cls._handle_lock:
            handle = cls._handle_counter
            cls._handle_counter += 1
            return handle
```

**Consequences:**
- Handles are globally unique within a type
- Thread-safe allocation
- Debug-friendly (handle range tells you resource type)

### ADR-P1-004: Thread Safety Model

**Status:** Confirmed (consistent across all subsystems)

**Context:**
Platform code is called from multiple threads (audio callback, render thread, main thread).

**Decision:**
Three-tier thread safety model:

1. **Immutable Data:** Dataclasses with frozen=True where possible
2. **Protected Mutation:** threading.Lock for mutable shared state
3. **Thread-Local:** ThreadLocalStorage for per-thread caches

Locking discipline:
- Acquire locks in consistent order to prevent deadlock
- Release locks before calling callbacks (avoid lock inversion)
- Use timeout-based try_lock for non-critical paths

**Consequences:**
- No data races in platform code
- Deadlock avoidance via discipline
- Performance acceptable for platform operations

## Component Diagram

```
engine/platform/
    |
    +-- constants.py                 # All magic numbers
    |
    +-- registry.py                  # Generic BackendRegistry[T]
    |
    +-- audio/
    |       +-- backends/
    |       |       +-- __init__.py  # Audio backend registry
    |       |       +-- null_backend.py
    |       +-- ...
    |
    +-- gpu/
    |       +-- backends/            # (to be added)
    |       |       +-- __init__.py
    |       |       +-- null_backend.py
    |       +-- ...
    |
    +-- input/
    |       +-- backends/            # (to be added)
    |       +-- ...
    |
    +-- os/
    |       # No backends (uses Python stdlib)
    |
    +-- rhi/
    |       # Backends are external (Rust crate)
    |
    +-- services/
    |       +-- backends/            # (to be added for mobile)
    |       +-- ...
    |
    +-- window/
            +-- backends/            # (to be added)
            +-- ...
```

## Data Flow

### Backend Selection at Startup

```
Application Start
       |
       v
detect_platform()  --> PlatformInfo
       |
       v
For each subsystem:
    |
    +-- Check environment variable (e.g., TRINITY_AUDIO_BACKEND)
    |
    +-- Fall back to platform default
    |
    +-- Fall back to "null" backend
       |
       v
Backend instantiated and assigned to subsystem singleton
```

### Cross-Subsystem Dependencies

```
constants.py <-- All subsystems import constants
    |
    v
registry.py  <-- Subsystems with pluggable backends
    |
    +-- audio/backends/
    +-- window/backends/
    +-- input/backends/
    +-- services/backends/ (mobile)
```

## File Changes Required

### New Files

| File | Purpose |
|------|---------|
| engine/platform/registry.py | Generic BackendRegistry[T] class |

### Modified Files

| File | Changes |
|------|---------|
| engine/platform/constants.py | Add any missing constants discovered in investigation |
| engine/platform/__init__.py | Export BackendRegistry |

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Constants changes break existing code | All constant changes are additive |
| Generic registry too complex | Keep it simple; subsystems can extend if needed |
| Thread safety overhead | Only lock where mutation occurs |

## Phase Exit Criteria

1. `engine/platform/registry.py` exists with BackendRegistry[T]
2. All constants from investigation docs present in constants.py
3. Unit tests for BackendRegistry (register, get, create, list)
4. Existing audio backend registry migrated to generic version (or documented why not)
