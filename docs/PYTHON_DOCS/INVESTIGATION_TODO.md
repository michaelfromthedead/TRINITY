# TRINITY Archaeological Investigation — Work Queue

**Created:** 2026-05-22
**Completed:** 2026-05-22
**Purpose:** Track deep-dive investigation of every directory
**Output:** One MD file per directory in `docs/investigation/`
**Final Report:** `GRAND_SYNTHESIS.md`

## Status Key
- `[ ]` Not started
- `[~]` In progress
- `[x]` Complete
- `[0]` Empty/Zero-line (confirmed scaffolding)

---

## Priority 1: Core Infrastructure — COMPLETE

### engine/core/ (Heart of the Engine)
- [x] `engine/core/ecs` — Entity-Component-System — REAL
- [x] `engine/core/math` — Math types — REAL
- [x] `engine/core/memory` — Allocators — REAL
- [x] `engine/core/scheduler` — Task scheduling — REAL
- [x] `engine/core/session` — Session management — REAL
- [x] `engine/core/tasks` — Task system — REAL

### trinity/ (Metaprogramming)
- [x] `trinity/decorators` — ~275 decorators — REAL (~18,920 lines)
- [x] `trinity/decorators/builtin_stacks` — Composite stacks — REAL
- [x] `trinity/descriptors` — Attribute access — REAL (~3,900 lines)
- [x] `trinity/metaclasses` — Class creation — REAL (~3,541 lines)
- [x] `trinity/tools` — Introspection utilities — REAL (~256 lines)

### foundation/ (Runtime)
- [x] `foundation` — Registry, Tracker, EventLog, Mirror, etc. — REAL (~6,700 lines)
- [x] `foundation/shelllang` — Interactive shell — REAL (~1,787 lines)

---

## Priority 2: Platform & Rendering — COMPLETE

### engine/platform/ (Hardware Abstraction)
- [x] `engine/platform/rhi` — RHI — REAL (1,818 lines, abstract interfaces)
- [x] `engine/platform/audio` — Audio device — REAL
- [x] `engine/platform/audio/backends` — Audio backends — REAL
- [x] `engine/platform/gpu` — GPU utilities — STUB (98 lines)
- [x] `engine/platform/input` — Input devices — REAL
- [x] `engine/platform/os` — OS abstraction — REAL
- [x] `engine/platform/services` — Platform services — PARTIAL
- [x] `engine/platform/window` — Window management — REAL

### engine/rendering/ (Graphics Pipeline)
- [x] `engine/rendering/framegraph` — Frame graph — REAL (~3,524 lines)
- [x] `engine/rendering/gpu_driven` — GPU culling — REAL (~4,859 lines)
- [x] `engine/rendering/materials` — Material system — REAL (~5,976 lines)
- [x] `engine/rendering/lighting` — Lighting — REAL (~4,470 lines)
- [x] `engine/rendering/particles` — Particle system — REAL (~5,982 lines)
- [x] `engine/rendering/postprocess` — Post-processing — PARTIAL (~8,861 lines)
- [x] `engine/rendering/demoscene` — SDF/raymarching — REAL (~1,130 lines)

---

## Priority 3: Zero-Line Mysteries — CONFIRMED EMPTY

### engine/common/
- [0] `engine/common/constants` — Empty scaffolding
- [0] `engine/common/types` — Empty scaffolding
- [0] `engine/common/utils` — Empty scaffolding

### engine/determinism/
- [0] `engine/determinism/core` — Empty scaffolding
- [0] `engine/determinism/network` — Empty scaffolding
- [0] `engine/determinism/replay` — Empty scaffolding
- [0] `engine/determinism/snapshot` — Empty scaffolding

### engine/engine/
- [0] `engine/engine/bootstrap` — Empty scaffolding
- [0] `engine/engine/scheduler` — Empty scaffolding
- [0] `engine/engine/session` — Empty scaffolding
- [0] `engine/engine/world` — Empty scaffolding

### engine/integration/
- [0] `engine/integration/decorator_binding` — Empty scaffolding
- [0] `engine/integration/descriptor_chain` — Empty scaffolding
- [0] `engine/integration/flowforge` — Empty scaffolding
- [0] `engine/integration/foundation_sync` — Empty scaffolding
- [0] `engine/integration/mods` — Empty scaffolding
- [0] `engine/integration/shelllang` — Empty scaffolding

---

## Priority 4: Animation System — COMPLETE (ALL REAL)

- [x] `engine/animation/crowds` — REAL (~2,237 lines) — RVO avoidance, GPU textures
- [x] `engine/animation/facial` — REAL (~5,233 lines) — FACS, lip sync
- [x] `engine/animation/graph` — REAL (~5,057 lines) — Quaternion SLERP, blend trees
- [x] `engine/animation/ik` — REAL (~4,776 lines) — FABRIK, CCD, Jacobian
- [x] `engine/animation/motionmatching` — REAL (~6,451 lines) — KD-tree search
- [x] `engine/animation/procedural` — REAL (~4,744 lines) — Verlet, spring bone
- [x] `engine/animation/skeletal` — REAL (~7,398 lines) — Dual quaternion skinning
- [x] `engine/animation/systems` — REAL (~3,225 lines) — ECS integration

---

## Priority 5: Audio System — COMPLETE (ALL REAL)

- [x] `engine/audio/adaptive` — REAL (~5,606 lines) — Beat-grid quantization
- [x] `engine/audio/core` — REAL (~4,994 lines) — Voice stealing, Doppler
- [x] `engine/audio/dialogue` — REAL (~5,433 lines) — VO streaming, subtitles
- [x] `engine/audio/dsp` — REAL (~6,761 lines) — Biquad, Freeverb
- [x] `engine/audio/mixing` — REAL (~5,020 lines) — HDR audio, sidechain
- [x] `engine/audio/spatial` — REAL (~4,880 lines) — HRTF, VBAP, Ambisonics

---

## Priority 6: Gameplay Systems — COMPLETE (ALL REAL)

- [x] `engine/gameplay/abilities` — REAL (~3,136 lines) — GAS-style effects
- [x] `engine/gameplay/ai` — REAL (~4,523 lines) — BT, GOAP, utility AI
- [x] `engine/gameplay/camera` — REAL (~6,724 lines) — Rails, collision
- [x] `engine/gameplay/combat` — REAL (~6,343 lines) — Hitbox, killstreaks
- [x] `engine/gameplay/combat/modes` — REAL (~385 lines) — Deathmatch
- [x] `engine/gameplay/components` — REAL (~3,462 lines) — Stats, movement
- [x] `engine/gameplay/economy` — REAL (~4,217 lines) — Pity systems, loot
- [x] `engine/gameplay/entity` — REAL (~4,418 lines) — UE5-style actors
- [x] `engine/gameplay/input` — REAL (~4,064 lines) — Response curves
- [x] `engine/gameplay/nav` — REAL (~6,493 lines) — A*/JPS/Theta*/HPA*
- [x] `engine/gameplay/quest` — REAL (~7,762 lines) — Dialogue graph

---

## Priority 7: Simulation (Physics) — COMPLETE (MOSTLY REAL)

- [x] `engine/simulation/character` — REAL (~4,614 lines) — Active ragdoll
- [x] `engine/simulation/cloth` — PARTIAL (~3,345 lines) — PBD; gpu_cloth interface only
- [x] `engine/simulation/collision` — REAL (~5,349 lines) — GJK, EPA, SAP/BVH
- [x] `engine/simulation/components` — REAL (~3,406 lines) — Physics wrappers
- [x] `engine/simulation/constraints` — REAL (~3,311 lines) — D6 joints
- [x] `engine/simulation/destruction` — REAL (~4,869 lines) — Voronoi fracture
- [x] `engine/simulation/fluid` — REAL (~3,504 lines) — SPH, FLIP/PIC, PBF
- [x] `engine/simulation/hair` — REAL (~2,600 lines) — Position-Based Dynamics
- [x] `engine/simulation/physics` — REAL (~5,805 lines) — Rigid body, sleeping
- [x] `engine/simulation/softbody` — REAL (~3,546 lines) — FEM, Neo-Hookean
- [x] `engine/simulation/solver` — REAL (~3,987 lines) — SI, TGS, XPBD
- [x] `engine/simulation/vehicles` — REAL (~4,681 lines) — Pacejka tires

---

## Priority 8: Tooling (Largest) — COMPLETE (ALL REAL)

- [x] `engine/tooling/animation_tools` — REAL (~9,157 lines)
- [x] `engine/tooling/assettools` — REAL (~7,523 lines)
- [x] `engine/tooling/automation` — REAL (~3,981 lines)
- [x] `engine/tooling/build` — REAL (~4,158 lines)
- [x] `engine/tooling/console` — REAL (~2,694 lines)
- [x] `engine/tooling/crash` — REAL (~2,757 lines)
- [x] `engine/tooling/debug` — REAL (~6,931 lines)
- [x] `engine/tooling/editor` — REAL (~5,919 lines)
- [x] `engine/tooling/hotreload` — REAL (~2,729 lines)
- [x] `engine/tooling/leveleditor` — REAL (~8,041 lines)
- [x] `engine/tooling/localization` — REAL (~3,423 lines)
- [x] `engine/tooling/logging` — REAL (~2,759 lines)
- [x] `engine/tooling/material_editor` — REAL (~6,705 lines)
- [x] `engine/tooling/profiling` — REAL (~6,479 lines)
- [x] `engine/tooling/replay` — REAL (~6,550 lines)
- [x] `engine/tooling/terrain` — REAL (~4,344 lines)
- [x] `engine/tooling/testing` — REAL (~3,812 lines)
- [x] `engine/tooling/undo` — REAL (~2,473 lines)
- [x] `engine/tooling/vcs` — REAL (~3,437 lines)
- [x] `engine/tooling/visual_scripting` — REAL (~7,711 lines)

---

## Priority 9: UI System — COMPLETE (ALL REAL)

- [x] `engine/ui/accessibility` — REAL (~3,543 lines) — WCAG 2.1
- [x] `engine/ui/animation` — REAL (~4,507 lines) — 22 easing functions
- [x] `engine/ui/binding` — REAL (~3,793 lines) — Two-way binding
- [x] `engine/ui/framework` — REAL (~4,106 lines) — W3C events
- [x] `engine/ui/layout` — REAL (~4,394 lines) — Grid, Flexbox
- [x] `engine/ui/screens` — REAL (~2,745 lines) — Screen stack
- [x] `engine/ui/styling` — REAL (~3,491 lines) — 12 blend modes
- [x] `engine/ui/text` — REAL (~4,361 lines) — Unicode, SDF fonts
- [x] `engine/ui/widgets` — REAL (~461 lines) — Index
- [x] `engine/ui/widgets/display` — REAL (~3,360 lines)
- [x] `engine/ui/widgets/game` — REAL (~4,199 lines)
- [x] `engine/ui/widgets/input` — REAL (~4,742 lines)
- [x] `engine/ui/widgets/primitives` — REAL (~2,181 lines)

---

## Priority 10: World & Environment — COMPLETE (ALL REAL)

- [x] `engine/world/environment` — REAL
- [x] `engine/world/foliage` — REAL
- [x] `engine/world/hlod` — REAL
- [x] `engine/world/partition` — REAL
- [x] `engine/world/pcg` — REAL
- [x] `engine/world/queries` — REAL
- [x] `engine/world/terrain` — REAL

---

## Priority 11: XR (VR/AR) — COMPLETE (MOSTLY REAL)

- [x] `engine/xr/avatars` — REAL (~3,152 lines)
- [x] `engine/xr/input` — REAL (~4,757 lines)
- [x] `engine/xr/interaction` — REAL (~3,700 lines)
- [x] `engine/xr/locomotion` — REAL (~3,310 lines)
- [x] `engine/xr/platform` — PARTIAL
- [x] `engine/xr/rendering` — REAL (~2,600 lines)
- [x] `engine/xr/runtime` — REAL
- [x] `engine/xr/spatial` — REAL (~4,873 lines)
- [x] `engine/xr/ui` — REAL (~2,782 lines)
- [x] `engine/xr/utils` — REAL (~391 lines)

---

## Priority 12: Networking — COMPLETE (ALL REAL)

- [x] `engine/networking/lag_compensation` — REAL
- [x] `engine/networking/prediction` — REAL
- [x] `engine/networking/replication` — REAL
- [x] `engine/networking/rpc` — REAL
- [x] `engine/networking/security` — REAL
- [x] `engine/networking/serialization` — REAL
- [x] `engine/networking/social` — REAL
- [x] `engine/networking/transport` — REAL

---

## Priority 13: Debug & Resource — COMPLETE (MOSTLY REAL)

### engine/debug/
- [x] `engine/debug/console` — REAL
- [x] `engine/debug/crash` — REAL
- [x] `engine/debug/logging` — REAL
- [x] `engine/debug/profiling` — REAL
- [x] `engine/debug/replay` — REAL
- [x] `engine/debug/testing` — REAL
- [x] `engine/debug/tools` — REAL
- [x] `engine/debug/visual` — REAL

### engine/resource/
- [x] `engine/resource/asset` — REAL
- [x] `engine/resource/build` — PARTIAL
- [x] `engine/resource/memory` — REAL
- [x] `engine/resource/streaming` — PARTIAL
- [x] `engine/resource/types` — REAL
- [x] `engine/resource/virtualization` — REAL

---

## Summary

| Priority | Category | Directories | Status |
|----------|----------|-------------|--------|
| 1 | Core Infrastructure | 13 | ✅ COMPLETE (ALL REAL) |
| 2 | Platform & Rendering | 15 | ✅ COMPLETE (12 REAL, 3 PARTIAL) |
| 3 | Zero-Line Mysteries | 17 | ✅ CONFIRMED EMPTY |
| 4 | Animation | 8 | ✅ COMPLETE (ALL REAL) |
| 5 | Audio | 6 | ✅ COMPLETE (ALL REAL) |
| 6 | Gameplay | 11 | ✅ COMPLETE (ALL REAL) |
| 7 | Simulation | 12 | ✅ COMPLETE (11 REAL, 1 PARTIAL) |
| 8 | Tooling | 20 | ✅ COMPLETE (ALL REAL) |
| 9 | UI | 13 | ✅ COMPLETE (ALL REAL) |
| 10 | World | 7 | ✅ COMPLETE (ALL REAL) |
| 11 | XR | 10 | ✅ COMPLETE (9 REAL, 1 PARTIAL) |
| 12 | Networking | 8 | ✅ COMPLETE (ALL REAL) |
| 13 | Debug & Resource | 14 | ✅ COMPLETE (12 REAL, 2 PARTIAL) |
| **TOTAL** | | **142** | **✅ ALL INVESTIGATED** |

---

## Final Classification

| Classification | Count | Lines |
|----------------|-------|-------|
| REAL | 117 | ~570,000 |
| PARTIAL | 8 | ~15,000 |
| EMPTY/STUB | 17 | ~5,000 |
| **TOTAL** | **142** | **~600,000** |

---

*Investigation complete: 2026-05-22*
*See GRAND_SYNTHESIS.md for full analysis*
