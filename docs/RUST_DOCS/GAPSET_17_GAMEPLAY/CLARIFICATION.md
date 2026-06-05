# GAPSET_17_GAMEPLAY — Architectural Clarification & Divergence Analysis

## 1. Systematic PHASE_N_TODO.md Inaccuracy

### Finding
The PHASE_N_TODO.md document marks Phases 2, 3, 4, 5, 7, 8, 9, and 10 as entirely pending `[ ]` (all 118 tasks across 8 phases). In reality, **109 of those 118 tasks (92.4%) have complete, verified implementations** in the codebase.

### Root Cause
The document appears to have been generated as a **forward-looking roadmap** rather than a **retrospective audit**. It lists "what should exist" without checking "what already exists." The Phase 1 tasks were marked `[x]` because they were part of an earlier verification pass; the remaining phases were never audited against the actual codebase.

### Impact
- Codebase is significantly more mature than the plan document suggests
- Remaining work is focused on 9 absent tasks (MCTS Phase 6, Influence Maps) and 6 partial tasks
- No structural rewrite needed — the architecture is implemented and functional

### Recommendation
Maintain PHASE_N_TODO.md as a live audit document. Update it whenever new modules are implemented. Consider generating task lists from actual source code structure rather than from planning documents alone.

---

## 2. Phase 6 MCTS — Not Implemented

### Finding
All 8 tasks in Phase 6 (Monte Carlo Tree Search solver) are absent. No MCTS module, tree node, UCB1 selection, expansion, simulation, backpropagation, budget management, combat integration, or fallback chain exists anywhere in `engine/gameplay/`.

### Assessment
This is a genuine implementation gap. MCTS represents the third tier of the AI architecture (Behavior Trees → Utility AI / GOAP → MCTS). The existing AI system uses BT for reactive behaviors, Utility AI for scored decision-making, and GOAP for goal-oriented planning — all of which work without MCTS.

### Recommendation
Implement MCTS as a standalone module at `engine/gameplay/ai/mcts/` with:
1. MCTSNode with visit/score/children
2. Standard UCB1 selection, expansion, simulation, backpropagation
3. Configurable iteration budget (time or count-based)
4. Domain adapter for combat scenarios
5. Fallback chain that degrades gracefully through GOAP → Utility → BT

Priority: Medium (MCTS is additive — existing AI tiers cover most gameplay needs)

---

## 3. T-GP-1.6 @spawner Decorator — Partial Implementation

### Finding
The task specifies `@spawner(prefab, pool_size, spawn_rate, max_alive)` as a decorator with object pooling parameters. The codebase has:
- `spawn_prefab()` and `register_prefab()` as regular functions in `entity/prefab.py`
- Deferred destroy functionality in LifecycleManager
- No `@spawner` decorator with the specified parameter signature

### Assessment
Spawn capability exists but not as a decorator with pooling semantics. The `@prefab` decorator handles prefab registration, and `spawn_prefab()` handles instantiation. Object pooling as described (pool_size recycle, spawn_rate regulation, max_alive enforcement) is not implemented.

### Recommendation
Create `@spawner` decorator in `entity/prefab.py` that wraps the existing `spawn_prefab()` function with pooling logic. Pooled entities can leverage LifecycleManager's deferred destroy.

---

## 4. T-GP-3.19 Influence Maps — Constants Only

### Finding
`ai/constants.py` contains configuration constants for influence maps:
- `INFLUENCE_MAP_CELL_SIZE`
- `INFLUENCE_MAX_VALUE`, `INFLUENCE_MIN_VALUE`
- `INFLUENCE_DECAY_RATE`
- `INFLUENCE_PROPAGATION_RATE`
- `INFLUENCE_MAX_PROPAGATION_DISTANCE`

However, **no implementation class exists** — no grid, no value propagation, no layer system, no `sample()` method.

### Assessment
The constants suggest influence maps were planned and partially scaffolded but never implemented. This is the only genuine gap in the AI Tier 1 phase.

### Recommendation
Implement `InfluenceMap` class at `engine/gameplay/ai/influence.py`:
- 2D/3D grid overlay with configurable cell size
- Multi-layer support: threat, resource, interest
- Value propagation with configurable decay rate
- `sample(layer, position)` query API
- Integration with AI perception system

---

## 5. ai/__init__.py — Bus-Factor File

### Finding
`engine/gameplay/ai/__init__.py` is **1185 lines** containing BT, Utility AI, GOAP, Perception, Knowledge, and CombatAI code combined in a single file. This coexists alongside separate refactored files (`behavior_tree.py`, `utility.py`, `goal.py`, `perception.py`, `blackboard.py`).

### Assessment
This is a significant maintainability risk. The combined file appears to be either:
- An older version predating the refactored separate files
- A convenience re-export/registry file that grew organically
- A migration artifact where new refactored files were created alongside the legacy combined file

### Impact
- High cognitive load for new developers
- Merge conflict risk when multiple developers work on different AI systems
- Testing difficulty — the combined file cannot be unit-tested in isolation
- Import confusion — two versions of the same code may conflict

### Recommendation
1. Audit the combined `ai/__init__.py` against each separate file
2. Identify any unique functionality not present in refactored files
3. Move unique code to appropriate separate files
4. Convert `ai/__init__.py` to a clean re-export module
5. Add deprecation warnings for any duplicated exports

---

## 6. T-GP-10.10/10.11/10.12 — Game Mode Subclasses

### Finding
`combat/game_mode.py` (655 lines) implements a GameMode base class with:
- Match lifecycle (Lobby → Countdown → Playing → Match End → Results)
- WinConditionType (7 types)
- ScoringEventType (13 types)
- Spawn logic, rule hooks, round management, overtime
- Configurable time/score limits

However, no specific mode subclasses (Deathmatch, Team Deathmatch, CTF, KOTH, Battle Royale) were verified as separate classes.

### Assessment
The GameMode framework is implemented and functional. Specific mode configurations can be created by instantiating GameMode with different parameters (WinConditionType, ScoringEventType). Dedicated subclasses would provide cleaner encapsulation of mode-specific rules.

---

## 7. Camera Blending System — Implemented But Not in PHASE_N_TODO.md

### Finding
The camera system includes a complete blending framework (`camera/blending.py`) with:
- 12 blend types (Cut, Linear, EaseIn, EaseOut, EaseInOut, Acceleration, Deceleration, Exponential, Smooth, Spring, Elastic, Bounce)
- CameraBlend with pause/resume/reverse/skip
- BlendStack for nested transitions
- SplitScreenLayout with 7 layouts
- CameraDirector for coordinated camera management

This is a significant feature not captured in any PHASE_N_TODO.md task.

### Recommendation
Consider adding a task for camera blending in the next revision of the phase plan.

---

## 8. Dialogue Condition/Effect/Variable Architecture — Well-Designed

### Finding
The dialogue system has a well-separated architecture:
- `dialogue.py` (1453 lines): Graph structure, node types, traversal
- `dialogue_conditions.py` (1078 lines): Condition checking system
- `dialogue_effects.py` (1486 lines): Game effect execution from dialogue
- `dialogue_variables.py` (942 lines): Variable scoping and resolution (local/global/quest-linked/world state)

This separation of concerns is exemplary and should serve as a pattern for other gameplay subsystems.

---

## 9. Cross-Reference: Shared Infrastructure via GAPSET_3_BRIDGE

The following shared infrastructure from GAPSET_3_BRIDGE is used across GAPSET_17_GAMEPLAY:

| Infrastructure | Usage in Gameplay |
|---------------|-------------------|
| ComponentStore | Entity component storage and lookup |
| wgpu Renderer | Camera view/projection matrices, debug visualization |
| WGSL Shaders | Post-processing effects, camera effects |
| omega math types | Vec3, AABB, Ray for transform, collision, NavMesh |
| Foundation EventLog | AI decisions, entity lifecycle, quest state logging |
| EngineMeta/MetaStack | ActorMeta, ControllerMeta, StateMeta registration |
| @component/@decorator | ECS component declaration and composition |
| TrackedDescriptor | Attribute change tracking, health monitoring |
| ResourceMeta | Input action/axis resource registration |
