# PHASE A Sprint: Resource & Platform

**Sprint Start:** 2026-05-23
**Sprint End:** 2026-05-30
**Duration:** 7 days
**Status:** NOT STARTED

---

## Sprint Goal

Complete the three PARTIAL resource/platform modules to bring TRINITY Python from 97% to ~99% completion.

---

## Sprint Backlog

| # | Task | Days | Status | Assignee |
|---|------|------|--------|----------|
| 1 | Resource Streaming | 2 | ⏳ TODO | — |
| 2 | Resource Build Pipeline | 2 | ⏳ TODO | — |
| 3 | Platform Services | 3 | ⏳ TODO | — |

---

## Task 1: Resource Streaming

**Path:** `engine/resource/streaming/`
**Effort:** 2 days
**Dependencies:** None

### Current State Analysis

```bash
# Run before starting
uv run python -c "from engine.resource.streaming import *; print('OK')"
wc -l engine/resource/streaming/*.py
```

### Implementation Checklist

#### Day 1
- [ ] Read existing `stream_manager.py`
- [ ] Document current API surface
- [ ] Add `StreamPriority` enum:
  ```python
  class StreamPriority(IntEnum):
      CRITICAL = 0    # Blocking gameplay
      HIGH = 1        # Visible on screen
      NORMAL = 2      # Nearby but not visible
      LOW = 3         # Predictive prefetch
      BACKGROUND = 4  # Opportunistic
  ```
- [ ] Replace list with `heapq` priority queue
- [ ] Add `max_bytes_per_frame` throttling

#### Day 2
- [ ] Add `cancel_stream(stream_id)` API
- [ ] Add `on_memory_pressure(callback)` hook
- [ ] Add streaming metrics:
  - Bytes streamed this frame
  - Queue depth
  - Average latency
- [ ] Write test: `tests/test_resource/test_streaming.py`

### Verification

```bash
uv run python -m pytest tests/test_resource/test_streaming.py -v
```

### Definition of Done
- [ ] Priority queue sorts correctly
- [ ] Throttling prevents frame spikes
- [ ] Cancellation is immediate
- [ ] Test passes

---

## Task 2: Resource Build Pipeline

**Path:** `engine/resource/build/`
**Effort:** 2 days
**Dependencies:** None (parallel with Task 1)

### Current State Analysis

```bash
uv run python -c "from engine.resource.build import *; print('OK')"
wc -l engine/resource/build/*.py
```

### Implementation Checklist

#### Day 3
- [ ] Read existing `dependency_tracker.py`
- [ ] Implement Kahn's algorithm for topological sort
- [ ] Add SHA256 file hash cache (JSON file)
- [ ] Implement dirty detection:
  ```python
  def is_dirty(asset_path: Path) -> bool:
      current_hash = hash_file(asset_path)
      cached_hash = self.cache.get(asset_path)
      return current_hash != cached_hash
  ```

#### Day 4
- [ ] Add `ThreadPoolExecutor` for parallel builds
- [ ] Implement progress callback:
  ```python
  def build(on_progress: Callable[[int, int, str], None]):
      # on_progress(current, total, asset_name)
  ```
- [ ] Output build manifest (JSON)
- [ ] Write test: `tests/test_resource/test_build.py`

### Verification

```bash
uv run python -m pytest tests/test_resource/test_build.py -v
```

### Definition of Done
- [ ] Incremental build skips unchanged files
- [ ] Parallel build uses N cores
- [ ] Progress callback fires correctly
- [ ] Test passes

---

## Task 3: Platform Services

**Path:** `engine/platform/services/`
**Effort:** 3 days
**Dependencies:** None (parallel with Tasks 1-2)

### Current State Analysis

```bash
uv run python -c "from engine.platform.services import *; print('OK')"
ls -la engine/platform/services/
```

### Implementation Checklist

#### Day 5 — Null + Clipboard
- [ ] Create `NullServiceProvider`:
  ```python
  class NullServiceProvider(ServiceProvider):
      def get_clipboard(self) -> ClipboardService:
          return NullClipboardService()
      # ... all return Null implementations
  ```
- [ ] Create `LinuxClipboardService`:
  ```python
  class LinuxClipboardService(ClipboardService):
      def copy(self, text: str) -> None:
          subprocess.run(['xclip', '-selection', 'clipboard'], 
                        input=text.encode(), check=True)
      def paste(self) -> str:
          result = subprocess.run(['xclip', '-selection', 'clipboard', '-o'],
                                 capture_output=True, text=True)
          return result.stdout
  ```
- [ ] Create `LinuxFileDialogService` (zenity)

#### Day 6 — Notifications + Factory
- [ ] Create `LinuxNotificationService` (notify-send)
- [ ] Create `ServiceProviderFactory`:
  ```python
  def get_provider() -> ServiceProvider:
      if sys.platform == 'linux':
          return LinuxServiceProvider()
      elif sys.platform == 'darwin':
          return MacOSServiceProvider()  # stub
      elif sys.platform == 'win32':
          return WindowsServiceProvider()  # stub
      return NullServiceProvider()
  ```
- [ ] Test on Linux

#### Day 7 — Stubs + Tests
- [ ] Stub `MacOSServiceProvider` (raises NotImplementedError)
- [ ] Stub `WindowsServiceProvider` (raises NotImplementedError)
- [ ] Write `tests/test_platform/test_services.py`
- [ ] Documentation in docstrings

### Verification

```bash
uv run python -m pytest tests/test_platform/test_services.py -v
```

### Definition of Done
- [ ] Linux services work
- [ ] Factory auto-detects platform
- [ ] Null fallback works on unknown OS
- [ ] Test passes

---

## Daily Standup Template

```
Date: YYYY-MM-DD
Yesterday: [what was completed]
Today: [what will be done]
Blockers: [any blockers]
```

---

## Sprint Completion Checklist

- [ ] All three tasks marked DONE
- [ ] All tests pass: `uv run python -m pytest tests/test_resource tests/test_platform -v`
- [ ] No new syntax errors: `uv run python -m py_compile engine/resource/**/*.py engine/platform/**/*.py`
- [ ] Update REMAINING_WORK_ROADMAP.md status
- [ ] Update INVESTIGATION_STATE.json (mark PARTIAL → COMPLETE)

---

## Post-Sprint

After completing Phase A:
1. Update docs to mark these modules as REAL
2. Plan Phase B (GPU & Visual)
3. Consider deleting empty scaffolding

---

*Created: 2026-05-22*
*Sprint Tracking Document for PHASE A*
