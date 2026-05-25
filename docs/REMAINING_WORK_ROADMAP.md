# TRINITY — Remaining Work Roadmap

**Created:** 2026-05-22
**Status:** Active
**Scope:** Python GRANDPHASE1 completion (not GRANDPHASE2 Rust bridge)

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Python LOC | ~545,000 |
| Complete | ~530,000 (97%) |
| Remaining | ~15,000 (3%) |
| Estimated Effort | 2-3 weeks |

---

## Phase 1: Critical Path (Week 1)

**Goal:** Complete items that block other work or affect core functionality.

### 1.1 Platform Services (`engine/platform/services`)

**Status:** Interface only, no concrete implementations
**Blocks:** Platform-specific features (notifications, clipboard, file dialogs)
**Effort:** 3-4 days

```
engine/platform/services/
├── __init__.py          # Has abstract ServiceProvider
├── clipboard.py         # Abstract ClipboardService
├── file_dialog.py       # Abstract FileDialogService
├── notifications.py     # Abstract NotificationService
└── config.py            # Service configuration
```

**Tasks:**
- [ ] Implement `NullServiceProvider` (no-op fallback)
- [ ] Implement `LinuxServiceProvider` (xdg-open, xclip, notify-send)
- [ ] Implement `WindowsServiceProvider` (win32 APIs)
- [ ] Implement `MacOSServiceProvider` (pbcopy, osascript)
- [ ] Service factory with auto-detection

**Dependencies:** None
**Priority:** HIGH (editors need file dialogs)

---

### 1.2 Resource Streaming (`engine/resource/streaming`)

**Status:** Basic implementation, missing priority queue optimization
**Blocks:** Large world streaming, asset hot-loading
**Effort:** 2 days

**Tasks:**
- [ ] Add bandwidth throttling
- [ ] Implement stream priority decay
- [ ] Add stream cancellation on camera teleport
- [ ] Memory pressure callbacks

**Dependencies:** None
**Priority:** HIGH (affects runtime performance)

---

### 1.3 Resource Build Pipeline (`engine/resource/build`)

**Status:** Partial, missing incremental rebuild
**Blocks:** Asset iteration workflow
**Effort:** 2 days

**Tasks:**
- [ ] Complete dependency graph walker
- [ ] Add parallel build jobs
- [ ] Implement build cache invalidation
- [ ] Add progress reporting callbacks

**Dependencies:** None
**Priority:** MEDIUM (dev workflow, not runtime)

---

## Phase 2: GPU Backends (Week 2)

**Goal:** Complete GPU-accelerated paths for performance-critical systems.

### 2.1 GPU Cloth Simulation (`engine/simulation/cloth/gpu_cloth.py`)

**Status:** CPU simulation works, GPU path is stub
**Blocks:** Large cloth simulations (capes, flags, curtains)
**Effort:** 3-4 days

**Current State:**
```python
# gpu_cloth.py:338
"WARNING: This stub does not simulate. Use ClothSimulation for CPU simulation."
```

**Tasks:**
- [ ] Define compute shader interface (match WGSL in GRANDPHASE2)
- [ ] Implement `GPUClothSimulation.step()` dispatch
- [ ] Buffer upload/download for particle positions
- [ ] Fallback detection (no GPU → use CPU)

**Dependencies:** GRANDPHASE2 wgpu integration (can stub with CPU for now)
**Priority:** MEDIUM (CPU path works, GPU is optimization)

---

### 2.2 Post-Processing Stubs (`engine/rendering/postprocess`)

**Status:** 8,861 lines, most effects work, some stubs
**Blocks:** Full visual pipeline
**Effort:** 2-3 days

**Stubbed Effects (from investigation):**
- [ ] `SSR` (Screen-Space Reflections) — raymarching incomplete
- [ ] `VolumetricLighting` — raymarch scatter stub
- [ ] `TemporalAA` — history buffer management

**Tasks:**
- [ ] Complete SSR raymarching with hierarchical tracing
- [ ] Implement volumetric scatter integration
- [ ] Add TAA jitter patterns and history rejection

**Dependencies:** Frame graph must be functional
**Priority:** MEDIUM (visual quality, not functionality)

---

## Phase 3: XR Runtime (Week 2-3)

**Goal:** Make VR/AR actually runnable.

### 3.1 XR Platform (`engine/xr/platform`)

**Status:** Abstract interfaces, no concrete runtime
**Blocks:** All VR/AR functionality
**Effort:** 4-5 days

**Tasks:**
- [ ] `OpenXRRuntime` — Real OpenXR loader integration
- [ ] `MockXRRuntime` — Testing without headset
- [ ] Session lifecycle (create, begin, end, destroy)
- [ ] View configuration (stereo, mono, quad)
- [ ] Input action bindings

**Dependencies:** OpenXR SDK headers (C/Rust, called from Python)
**Priority:** LOW (unless VR is immediate goal)

---

## Phase 4: Empty Scaffolding Decision

**Goal:** Decide fate of 17 empty directories.

### Option A: Delete (Recommended)

These directories have 0 lines and the functionality exists elsewhere:

| Directory | Recommendation | Reason |
|-----------|----------------|--------|
| `engine/common/*` | DELETE | Types live in actual modules |
| `engine/determinism/*` | DELETE | Networking handles replay |
| `engine/engine/*` | DELETE | Core handles bootstrap |
| `engine/integration/*` | DELETE | Decorators handle binding |

**Action:** `rm -rf engine/common engine/determinism engine/engine engine/integration`

### Option B: Implement (If Needed)

If these ARE needed for future architecture:

| Directory | What It Would Do | Effort |
|-----------|------------------|--------|
| `engine/determinism/` | Lock-step networking | 2 weeks |
| `engine/integration/flowforge` | Visual scripting bridge | 1 week |
| `engine/integration/mods` | Mod loading system | 1 week |

**Recommendation:** Delete now. Recreate if needed. Empty scaffolding is tech debt.

---

## Development Flow — PHASE A ACTIVE

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE A: Resource & Platform (CURRENT SPRINT)             │
│                                                             │
│  Step 1: Resource Streaming      ← START HERE              │
│          └── 2 days                                         │
│                                                             │
│  Step 2: Resource Build Pipeline                            │
│          └── 2 days                                         │
│                                                             │
│  Step 3: Platform Services                                  │
│          └── 3 days                                         │
│                                                             │
│  Total: ~7 days                                             │
└─────────────────────────────────────────────────────────────┘

PHASE B: GPU & Visual (Next Sprint)
├── Post-processing stubs (2-3 days)
└── GPU Cloth interface (3-4 days)

PHASE C: XR (Deferred)
└── XR Platform runtime (5 days)

ANYTIME: Delete empty scaffolding (1 hour)
```

---

## PHASE A: Detailed Execution Plan

### Step 1: Resource Streaming (Days 1-2)

**Directory:** `engine/resource/streaming/`
**Current:** Basic implementation, missing optimization
**Goal:** Production-ready streaming with priority management

#### Day 1 Checklist
- [ ] Audit `stream_manager.py` current state
- [ ] Add `StreamPriority` enum (CRITICAL, HIGH, NORMAL, LOW, BACKGROUND)
- [ ] Implement priority queue with heapq
- [ ] Add bandwidth throttling (`max_bytes_per_frame`)

#### Day 2 Checklist
- [ ] Add stream cancellation API
- [ ] Implement memory pressure callbacks
- [ ] Add metrics/telemetry hooks
- [ ] Write integration test

#### Acceptance Criteria
```
✓ Can stream 100 assets with correct priority ordering
✓ Cancellation works mid-stream
✓ Memory pressure triggers eviction callback
```

---

### Step 2: Resource Build Pipeline (Days 3-4)

**Directory:** `engine/resource/build/`
**Current:** Partial, missing incremental rebuild
**Goal:** Fast incremental builds with dependency tracking

#### Day 3 Checklist
- [ ] Audit current implementation
- [ ] Complete `DependencyGraph.walk()` with topological sort
- [ ] Add file hash caching (SHA256)
- [ ] Implement dirty detection

#### Day 4 Checklist
- [ ] Add parallel build jobs (ThreadPoolExecutor)
- [ ] Implement progress callbacks
- [ ] Add build manifest output
- [ ] Write integration test

#### Acceptance Criteria
```
✓ Incremental rebuild skips unchanged assets
✓ Parallel builds use all CPU cores
✓ Progress is reportable to UI
```

---

### Step 3: Platform Services (Days 5-7)

**Directory:** `engine/platform/services/`
**Current:** Interface only, no implementations
**Goal:** Cross-platform service implementations

#### Day 5 Checklist
- [ ] Implement `NullServiceProvider` (no-op fallback)
- [ ] Implement `ClipboardService` for Linux (xclip/wl-copy)
- [ ] Implement `FileDialogService` for Linux (zenity/kdialog)

#### Day 6 Checklist
- [ ] Implement `NotificationService` for Linux (notify-send)
- [ ] Add service auto-detection factory
- [ ] Test all services on Linux

#### Day 7 Checklist
- [ ] Stub Windows/macOS providers (defer real impl)
- [ ] Write documentation
- [ ] Integration test suite

#### Acceptance Criteria
```
✓ File dialog opens on Linux
✓ Clipboard copy/paste works
✓ Graceful fallback to Null provider on unknown OS
```

---

## Success Criteria

### Definition of Done

- [ ] All PARTIAL directories upgraded to REAL
- [ ] Zero empty scaffolding directories remain
- [ ] `uv run python -m py_compile` passes all files
- [ ] Core systems can initialize without stubbed warnings
- [ ] One integration test per completed module

### Metrics

| Metric | Current | Target |
|--------|---------|--------|
| REAL directories | 117 | 125 |
| PARTIAL directories | 8 | 0 |
| EMPTY directories | 17 | 0 |
| Python completion | 97% | 100% |

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| GPU cloth requires GRANDPHASE2 | Blocks full completion | Keep CPU fallback, mark as "GRANDPHASE2 dependent" |
| OpenXR requires native code | Blocks XR completion | Use Mock runtime for testing, defer real OpenXR |
| Platform services vary by OS | Fragmented testing | Implement Null provider first, add platforms incrementally |

---

## Quick Start

```bash
# View current status
grep -c "PARTIAL\|EMPTY" docs/investigation/INVESTIGATION_TODO.md

# Start with streaming (no dependencies)
code engine/resource/streaming/

# Run syntax check after changes
uv run python -m py_compile engine/resource/streaming/*.py

# Delete empty scaffolding when ready
rm -rf engine/common engine/determinism engine/engine engine/integration
```

---

## Appendix: File Locations

| Item | Path | Lines |
|------|------|-------|
| Platform Services | `engine/platform/services/` | ~500 |
| Resource Streaming | `engine/resource/streaming/` | ~800 |
| Resource Build | `engine/resource/build/` | ~600 |
| GPU Cloth | `engine/simulation/cloth/gpu_cloth.py` | ~400 |
| Post-process Stubs | `engine/rendering/postprocess/` | ~2000 |
| XR Platform | `engine/xr/platform/` | ~800 |

---

*This roadmap covers Python GRANDPHASE1 completion only.*
*GRANDPHASE2 (Rust bridge) is a separate roadmap in `docs/gap_sets/`.*
