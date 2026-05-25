# GAPSET_14_ANIMATION -- Clarification Document

## Key Discoveries from RDC Investigation

### 1. TODO Was Written as Greenfield Plan, Not Reality

The PHASE_N_TODO.md header claims "Implementation Status: 0% complete, 68 tasks pending" but the codebase already contains a mature Python implementation of ~85-95% of the described functionality. This suggests the TODO was written before the main implementation effort, or is tracking a *different deliverable* (e.g., Rust/WGSL/Foundation integration) while the Python algorithmic layer was built in parallel.

**Action**: The TODO should be split into two tiers or the Python completion status should be recorded.

### 2. T-AN-5.7 and T-AN-9.3 Are Consolidated

The TODO describes two separate systems:
- **T-AN-5.7**: `animation_state_system.py` -- simulation-side state machine evaluator
- **T-AN-9.3**: `animation_graph_system.py` -- presentation-side playback system

The actual codebase has a single `systems/animation_graph_system.py` that handles both roles in 409 lines. It evaluates state machines, samples animations, manages transitions, and outputs poses. The TODO's two-phase architecture (simulation + presentation) could be an optimization target but is not how the current code works.

### 3. Foundation Integration Status

The codebase uses its own custom decorator/metaclass system:
- `@animation_data` in skeleton.py -- NOT Foundation's `@asset`
- `@ik_goal`, `@ik_chain` in ik_goal.py -- NOT Foundation's EventMeta/AssetMeta
- `@state_machine` in state_machine.py -- custom metaclass
- `@blend_tree` in blend_tree.py -- custom decorator

The systems use `engine.core.ecs.World/Entity` for ECS integration but do NOT use Foundation's `@system(phase=...)` annotation pattern. Systems are plain classes with `update()` methods.

### 4. Config is Decoupled from Foundation ResourceMeta

The `config.py` module provides 10 frozen dataclass configuration types (AnimationSystemConfig, IKConfig, ProceduralConfig, etc.) with thorough parameter documentation. However, these are plain Python dataclass instances, not registered as Foundation `@resource`. This means hot-reloading and Foundation Inspector integration are not wired.

### 5. Inertialization Exists in Motion Matching Only

The TODO's T-AN-2.2 (pose blending) lists inertialization as an acceptance criterion for the blending module. The actual inertialization implementation exists in `motionmatching/transition.py` as part of the motion matching pipeline. The general `blending.py` module handles lerp/slerp/additive/multiply blends but does not independently implement inertialization. This is architecturally sound -- inertialization is primarily a motion matching concern.

### 6. No Cinematics Module Exists

Phases 9.1 and 9.2 (cutscene playback, camera tracks) have no corresponding `cinematics/` submodule. This is the only entirely missing functional module.

### 7. No Test Infrastructure

Zero test files exist for any animation module. This is the largest quality gap -- 39,827 lines of complex mathematical code with no automated verification.

### 8. Rust and WGSL Layers Are Entirely Missing

The planned Rust backend (`crates/animation/`) and WGSL compute shaders (`shaders/skinning/`, `shaders/crowd/`) do not exist. The Python code prepares data structures for GPU consumption (e.g., `GPUSkinningData`, `prepare_gpu_skinning_data`, `AnimationTexture`) but no GPU-side shader consumes them.

### 9. Revised Effort Estimate

The original estimate of 12-24 weeks assumes greenfield implementation. For the actual remaining work:

| Remaining Work | Effort | Depends On |
|---------------|--------|------------|
| Rust backend (SIMD, parallel) | 4-6 weeks | Omega math types |
| WGSL shaders (skinning + crowd) | 2-3 weeks | Rust backend |
| Foundation integration | 1-2 weeks | S15/S16 foundations |
| Tests (unit + integration) | 2-3 weeks | All above |
| Cinematics module | 2-3 weeks | All animation systems |
| **Total remaining** | **~12-17 weeks** | |

### 10. Architecture Documentation Exists

`ANIMATION_CONTEXT.md` (848 lines) provides comprehensive architecture documentation covering all 11 submodules, decorators, metaclasses, Foundation integration points, evaluation pipeline, budget/LOD system, and canonical usage examples. This document is accurate and can serve as the primary reference for new developers.
