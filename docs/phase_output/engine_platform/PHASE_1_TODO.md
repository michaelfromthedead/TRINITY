# PHASE 1 TODO: Platform Backend Registry and Constants

## Summary

Establish foundational infrastructure for all platform subsystems.

**Estimated Effort:** 4-6 hours
**Dependencies:** None
**Blocking:** All subsequent phases

---

## Tasks

### T-P1-001: Create Generic Backend Registry

**Priority:** P0 (Blocking)
**Estimate:** 2 hours

Create `engine/platform/registry.py` with a generic backend registry:

```python
from typing import Generic, TypeVar

T = TypeVar('T')

class BackendRegistry(Generic[T]):
    def __init__(self): ...
    def register(self, name: str, backend_cls: type[T], set_default: bool = False) -> None: ...
    def get(self, name: str) -> type[T] | None: ...
    def create(self, name: str | None = None, *args, **kwargs) -> T: ...
    def list(self) -> list[str]: ...
    def default(self) -> str | None: ...
```

**Acceptance Criteria:**
- [ ] Generic type parameter properly propagated
- [ ] Thread-safe registration (lock around _backends dict mutation)
- [ ] Raises ValueError for unknown backend or missing default
- [ ] Unit tests in tests/platform/test_registry.py

---

### T-P1-002: Audit and Complete Constants

**Priority:** P0 (Blocking)
**Estimate:** 1 hour

Review `engine/platform/constants.py` against all investigation docs. Add any missing constants.

**Constants to verify exist:**

| Constant | Value | Source |
|----------|-------|--------|
| DEFAULT_AUDIO_SAMPLE_RATE | 48000 | audio investigation |
| FALLBACK_AUDIO_SAMPLE_RATE | 44100 | audio investigation |
| DEFAULT_AUDIO_CHANNELS | 2 | audio investigation |
| DEFAULT_AUDIO_BUFFER_SIZE | 1024 | audio investigation |
| AUDIO_THREAD_SLEEP_FACTOR | 0.95 | audio investigation |
| SPATIAL_DEFAULT_MIN_DISTANCE | 1.0 | audio investigation |
| SPATIAL_DEFAULT_MAX_DISTANCE | 100.0 | audio investigation |
| SPATIAL_DEFAULT_CONE_ANGLE | 360.0 | audio investigation |
| BUFFER_HANDLE_START | 1 | rhi investigation |
| TEXTURE_HANDLE_START | 1000 | rhi investigation |
| SAMPLER_HANDLE_START | 2000 | rhi investigation |
| SHADER_HANDLE_START | 3000 | rhi investigation |
| PIPELINE_HANDLE_START | 4000 | rhi investigation |
| GPU_ADDRESS_START | 0x100000000 | rhi investigation |
| DEFAULT_GAMEPAD_DEADZONE | (varies) | input investigation |
| MAX_TOUCH_POINTS | 10 | input investigation |
| TICKS_PER_SECOND | (varies) | os investigation |
| NANOS_PER_MILLI | 1000000 | os investigation |
| HYPERTHREADING_RATIO | 2 | os investigation |
| STANDARD_RESOLUTIONS | list | window investigation |
| STANDARD_REFRESH_RATES | list | window investigation |
| HDR_DEFAULT_MIN_LUMINANCE | (varies) | window investigation |
| HDR_DEFAULT_MAX_LUMINANCE | (varies) | window investigation |
| VRR_DEFAULT_MIN_HZ | (varies) | window investigation |
| VRR_DEFAULT_MAX_HZ | (varies) | window investigation |

**Acceptance Criteria:**
- [ ] All constants from investigation docs present
- [ ] Each constant has a docstring explaining its purpose
- [ ] Constants organized by subsystem with section comments

---

### T-P1-003: Export BackendRegistry from Platform Package

**Priority:** P0 (Blocking)
**Estimate:** 15 minutes

Update `engine/platform/__init__.py` to export BackendRegistry.

**Acceptance Criteria:**
- [ ] BackendRegistry in `__all__` list
- [ ] Import works: `from engine.platform import BackendRegistry`

---

### T-P1-004: Write Unit Tests for BackendRegistry

**Priority:** P0 (Blocking)
**Estimate:** 1 hour

Create `tests/platform/test_registry.py`:

```python
def test_register_and_get(): ...
def test_create_with_default(): ...
def test_create_with_name(): ...
def test_create_unknown_raises(): ...
def test_create_no_default_raises(): ...
def test_list_backends(): ...
def test_thread_safe_registration(): ...
```

**Acceptance Criteria:**
- [ ] All methods have at least one test
- [ ] Thread safety test uses concurrent.futures.ThreadPoolExecutor
- [ ] Tests pass with `uv run pytest tests/platform/test_registry.py`

---

### T-P1-005: Document Backend Selection Convention

**Priority:** P1 (Important)
**Estimate:** 30 minutes

Add docstrings to BackendRegistry explaining:
1. Environment variable convention (e.g., TRINITY_AUDIO_BACKEND)
2. Platform default selection order
3. Fallback to null backend

**Acceptance Criteria:**
- [ ] Module docstring explains the registry pattern
- [ ] Each method has a docstring
- [ ] Example usage in module docstring

---

### T-P1-006: Evaluate Audio Backend Migration

**Priority:** P2 (Nice to have)
**Estimate:** 1 hour

The audio subsystem already has `BackendRegistry` in `audio/backends/__init__.py`. Evaluate whether to:
- Migrate to generic version
- Keep audio-specific version (if specialized)
- Use composition (generic registry holds audio-specific instances)

**Acceptance Criteria:**
- [ ] Decision documented in this file or PHASE_1_ARCH.md
- [ ] If migrating, audio subsystem still works
- [ ] If not migrating, reason documented

---

## Task Dependency Graph

```
T-P1-001 (Create Registry)
    |
    +-- T-P1-004 (Unit Tests)
    |
    +-- T-P1-003 (Export)
    |
    +-- T-P1-005 (Document)
    |
    +-- T-P1-006 (Evaluate Audio)

T-P1-002 (Audit Constants)
    |
    (Independent, can run in parallel)
```

## Verification Commands

```bash
# Verify constants file syntax
uv run python -m py_compile engine/platform/constants.py

# Verify registry file syntax
uv run python -m py_compile engine/platform/registry.py

# Run unit tests
uv run pytest tests/platform/test_registry.py -v

# Verify import works
uv run python -c "from engine.platform import BackendRegistry; print('OK')"
```

## Completion Checklist

- [ ] T-P1-001: Generic BackendRegistry created
- [ ] T-P1-002: All constants audited and present
- [ ] T-P1-003: BackendRegistry exported from package
- [ ] T-P1-004: Unit tests pass
- [ ] T-P1-005: Documentation complete
- [ ] T-P1-006: Audio migration decision made
