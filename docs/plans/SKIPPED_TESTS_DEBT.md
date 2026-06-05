# Skipped Tests Technical Debt

**Created:** 2026-05-24
**Status:** ACTIVE DEBT
**Total Skipped Files:** 24

This document tracks all test files that are skipped due to missing implementations. These are NOT swept under the rug — they represent planned features that need implementation.

---

## Summary

| Category | Files | Root Cause |
|----------|-------|------------|
| Missing module: `trinity.omega` | 1 | GRANDPHASE2 Rust scheduler |
| Missing module: `flowforge_backend` | 2 | FlowForge Project 2 |
| Missing classes: demoscene AST | 9 | MaterialNode, etc. |
| Missing classes: gameplay/quest | 3 | DialogueChoice, etc. |
| Missing classes: resource/build | 1 | HashCache |
| Missing classes: animation_tools | 4 | BlendSpaceSample, etc. |
| Pre-existing: xr/avatars | 4 | Import error handling |

---

## Category 1: Missing Module — trinity.omega

### File: tests/core/ecs/test_ecs_comprehensive.py

**Skip Type:** `pytest.importorskip("trinity.omega")`

**What's Missing:**
```python
from trinity.omega.scheduler import Phase, Scheduler, create_default_scheduler
```

**Root Cause:** `trinity.omega` is the planned Python binding to the Omega Rust RHI. It doesn't exist yet — it's GRANDPHASE2 work.

**To Fix:**
1. Implement `omega/` Rust crate with Python bindings (PyO3)
2. Create `trinity/omega/` Python module as wrapper
3. Expose `Phase`, `Scheduler`, `create_default_scheduler`

**Estimated Effort:** 1-2 weeks (after Rust backend is working)

---

## Category 2: Missing Module — flowforge_backend

### Files:
- tests/flowforge/test_trinity_nodes.py
- tests/flowforge/test_trinity_nodes_whitebox.py

**Skip Type:** `pytest.importorskip("flowforge_backend")`

**What's Missing:**
```python
from flowforge_backend.ast_parser.trinity_nodes import (
    TrinityNodePosition, TrinitySourceLocation, TrinityFieldData,
    TrinityParameterData, TrinityMethodData, TrinityComponentData,
    TrinitySystemData, TrinityResourceData, TrinityEventData,
    TrinityAssetData, TrinityStateData, TrinityProtocolData,
    TrinityDecoratorData, TrinityGraphData,
)
```

**Root Cause:** FlowForge Project 2 (AST Parser) is not started. See `flowforge/README.md`:
```
PROJECT 2: AST Parser       ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0%
```

**To Fix:**
1. Create `flowforge_backend/` Python package
2. Implement `ast_parser/trinity_nodes.py` with Trinity-aware graph nodes
3. Implement `ast_parser/graph_types.py` with base graph types

**Estimated Effort:** 1 week

---

## Category 3: Missing Classes — demoscene AST

### Files (9 total):
- tests/rendering/demoscene/test_ast_nodes.py
- tests/rendering/demoscene/test_ast_nodes_contract.py
- tests/rendering/demoscene/test_ast_nodes_whitebox.py
- tests/rendering/demoscene/test_domain_codegen_blackbox.py
- tests/rendering/demoscene/test_domain_codegen_whitebox.py
- tests/rendering/demoscene/test_material_codegen_blackbox.py
- tests/rendering/demoscene/test_material_codegen_whitebox.py
- tests/rendering/demoscene/test_scene_codegen_blackbox.py
- tests/rendering/demoscene/test_scene_codegen_whitebox.py

**Skip Type:** `pytest.skip("Demoscene AST not fully implemented", allow_module_level=True)`

**What's Missing in `engine/rendering/demoscene/ast_nodes.py`:**
```python
class MaterialNode(ExprNode):
    """Material definition node for demoscene renderer."""
    ...

# Also potentially missing from tests:
# - Additional SDF primitive types
# - Material property nodes
# - Scene composition nodes
```

**Root Cause:** The demoscene renderer AST was partially implemented. `MaterialNode` and related material system nodes were planned but not written.

**To Fix:**
1. Add `MaterialNode` class to `ast_nodes.py`
2. Add material-related nodes (color, roughness, metallic, etc.)
3. Update `__init__.py` exports
4. Update `wgsl_codegen.py` to handle material nodes

**Estimated Effort:** 2-3 days

**Detailed Missing Imports:**
```python
# From test_ast_nodes.py:
MaterialNode  # NOT IN ast_nodes.py

# From test_wgsl_codegen_*.py:
# All the above plus material codegen support
```

---

## Category 4: Missing Classes — gameplay/quest

### Files:
- tests/gameplay/quest/test_dialogue.py
- tests/gameplay/quest/test_journal.py
- tests/gameplay/quest/test_tracker.py

**Skip Type:** `pytest.skip("Dialogue/Journal/Tracker API not fully implemented", allow_module_level=True)`

**What's Missing:**

### test_dialogue.py expects:
```python
from engine.gameplay.quest.dialogue import (
    DialogueGraph,      # EXISTS
    DialogueNode,       # EXISTS
    DialogueChoice,     # MISSING (module has "Choice")
    DialogueSession,    # MISSING
    DialogueContext,    # MISSING
    DialogueSpeaker,    # MISSING
    TextNode,           # EXISTS
    ChoiceNode,         # EXISTS
    BranchNode,         # EXISTS
    EventNode,          # EXISTS
    RandomNode,         # EXISTS
    EntryNode,          # EXISTS
    ExitNode,           # EXISTS
)
```

### test_journal.py expects:
```python
from engine.gameplay.quest.journal import (
    QuestJournal,       # EXISTS
    JournalEntry,       # EXISTS
    JournalCategory,    # EXISTS
    JournalFilter,      # EXISTS
    JournalPage,        # MISSING
    JournalSortOrder,   # MISSING
    JournalView,        # MISSING
)
```

### test_tracker.py expects:
```python
from engine.gameplay.quest.tracker import (
    QuestTracker,       # EXISTS
    QuestProgress,      # MISSING
    ObjectiveProgress,  # MISSING
    TrackerConfig,      # MISSING
    QuestTrackerEvent,  # MISSING (module has "QuestEvent")
    QuestTrackerListener,  # MISSING
)
```

**Root Cause:** Tests were written for a planned API that diverged from implementation. Either:
- Rename existing classes to match tests, OR
- Update tests to match actual API

**To Fix (Option A — match test API):**
1. Add `DialogueChoice = Choice` alias or rename class
2. Add `DialogueSession`, `DialogueContext`, `DialogueSpeaker` classes
3. Add `JournalPage`, `JournalSortOrder`, `JournalView` classes
4. Add `QuestProgress`, `ObjectiveProgress`, `TrackerConfig`, etc.

**To Fix (Option B — match implementation):**
1. Rewrite tests to use actual class names
2. Remove tests for features that don't exist

**Estimated Effort:** 1-2 days (Option A) or 4 hours (Option B)

---

## Category 5: Missing Class — resource/build

### File: tests/test_resource/test_build.py

**Skip Type:** `pytest.skip("HashCache not yet implemented", allow_module_level=True)`

**What's Missing:**
```python
from engine.resource.build import (
    BuildDependencyTracker,  # EXISTS
    DistributedBuildCoordinator,  # EXISTS
    HashCache,  # MISSING
    JobState,  # EXISTS
)
```

**Root Cause:** `HashCache` is part of the incremental rebuild feature mentioned in `docs/REMAINING_WORK_ROADMAP.md`:
```
- [ ] Add file hash caching (SHA256)
- [ ] Implement dirty detection
```

**To Fix:**
1. Create `HashCache` class in `engine/resource/build/` (new file or in dependency_tracker.py)
2. Implement `get(path, mtime)` and `put(path, mtime, hash)` methods
3. Export from `__init__.py`

**Estimated Effort:** 2-4 hours

**HashCache Interface (from test):**
```python
class HashCache:
    def get(self, path: str, mtime: float) -> Optional[str]: ...
    def put(self, path: str, mtime: float, hash: str) -> None: ...
```

---

## Category 6: Missing Classes — animation_tools

### Files:
- tests/tooling/animation_tools/test_anim_graph_editor.py
- tests/tooling/animation_tools/test_notifies_editor.py
- tests/tooling/animation_tools/test_pose_editor.py
- tests/tooling/animation_tools/test_preview_scene.py

**Skip Type:** `pytest.skip("... API mismatch", allow_module_level=True)`

**What's Missing:**

### test_anim_graph_editor.py:
```python
# Test expects:      Module has:
BlendSpaceSample    BlendSample
# (naming mismatch)
```

### test_notifies_editor.py:
```python
# Test expects:      Module has:
FootstepType        # MISSING (only FootstepNotify exists)
```

### test_pose_editor.py:
```python
# Test expects:      Module has:
PoseSnapshot        # MISSING
```

### test_preview_scene.py:
```python
# Test expects:      Module has:
GroundType          # MISSING
LightingPreset      # MISSING
PreviewCamera       # MISSING (has PreviewViewport)
```

**Root Cause:** API drift between test expectations and implementation.

**To Fix:**
1. Add missing classes OR add aliases
2. `BlendSpaceSample = BlendSample` alias
3. Add `FootstepType` enum
4. Add `PoseSnapshot` dataclass
5. Add `GroundType`, `LightingPreset` enums
6. Add `PreviewCamera` class or alias to `PreviewViewport`

**Estimated Effort:** 4-6 hours

---

## Category 7: Pre-existing — xr/avatars

### Files:
- tests/xr/avatars/test_avatar.py
- tests/xr/avatars/test_calibration.py
- tests/xr/avatars/test_hand_animator.py
- tests/xr/avatars/test_ik_solver.py

**Skip Type:** `pytest.skip(f"XR module has unrelated import errors: {e}", allow_module_level=True)`

**Note:** These were pre-existing skips with try/except import handling. They skip if ANY import error occurs in the XR module chain. Currently the imports work, so these tests SHOULD run.

**Status:** May need investigation — the try/except pattern might be overly broad.

---

## Priority Order for Fixes

### HIGH Priority (blocks significant functionality):
1. **HashCache** (resource/build) — 2-4 hours — blocks incremental builds
2. **Quest API** (gameplay/quest) — 1-2 days — blocks quest system tests

### MEDIUM Priority (nice to have):
3. **Animation tools classes** — 4-6 hours — editor-only, not runtime
4. **Demoscene MaterialNode** — 2-3 days — demo/visualization feature

### LOW Priority (future work):
5. **trinity.omega** — 1-2 weeks — depends on GRANDPHASE2 Rust work
6. **flowforge_backend** — 1 week — depends on FlowForge Project 2

---

## Tracking

| Date | Action | Files Affected |
|------|--------|----------------|
| 2026-05-24 | Initial skip markers added | 24 files |
| | | |

---

## How to Unskip

When implementing missing features:

1. Implement the missing class/module
2. Remove the `pytest.skip()` or `pytest.importorskip()` line
3. Run the tests: `uv run pytest tests/path/to/test_file.py -v`
4. Fix any test failures
5. Remove entry from this document
6. Add entry to Tracking table

---

*This document is the source of truth for skipped tests. Do not add skips without documenting here.*
