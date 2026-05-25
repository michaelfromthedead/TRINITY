# INVENTORY: engine_gameplay_abilities_ai_camera

**Created**: 2026-05-23
**Subsystem**: Gameplay Abilities, AI, and Camera
**Total Source Lines**: ~18,643 (aggregated from all sources)

---

## Source Documents (Temporal Order)

| Order | Document | Date | Lines | Summary |
|-------|----------|------|-------|---------|
| 1 | `engine_gameplay_abilities.md` | 2026-05-22 | ~145 | Deep investigation of abilities subsystem (3,136 lines examined): attributes, effects, targeting, tags |
| 2 | `engine_gameplay_ai.md` | 2026-05-22 | ~185 | Deep investigation of AI subsystem (4,523 lines examined): behavior trees, GOAP, utility AI, blackboard, perception |
| 3 | `engine_gameplay_camera.md` | 2026-05-22 | ~100 | Deep investigation of camera subsystem (7,060 lines examined): 8 controllers, collision, effects, rails, blending |
| 4 | `engine_gameplay_abilities_ai_camera.md` | 2026-05-22 | ~143 | Executive summary consolidating all three subsystems (14,383 lines reported) |

---

## Detection Rationale

All documents dated 2026-05-22, ordered by specificity (detailed investigations first, consolidated summary last). Documents 1-3 provide deep per-subsystem analysis; Document 4 provides cross-subsystem synthesis.

---

## Files in Each Subsystem (per source docs)

### abilities/ (3,136 lines)
- `attributes.py` (592 lines) - Attribute system with modifiers
- `effects.py` (829 lines) - Effect system (Instant, Duration, Infinite, Periodic)
- `targeting.py` (823 lines) - Targeting modes and area shapes
- `tags.py` (575 lines) - Hierarchical gameplay tags
- `constants.py` (322 lines) - Constants and enums

### ai/ (4,523 lines)
- `__init__.py` (1,184 lines) - BT, Utility, GOAP, Perception, Combat AI
- `behavior_tree.py` (948 lines) - 14 BT node types
- `blackboard.py` (496 lines) - Key-value store with observers
- `goap.py` (727 lines) - A* GOAP planner
- `utility_ai.py` (711 lines) - Response curves and considerations
- `constants.py` (457 lines) - 90+ constants, 9 enums

### camera/ (7,060 lines)
- `__init__.py` (330 lines) - Exports and documentation
- `constants.py` (607 lines) - 150+ camera constants
- `controller.py` (1,660 lines) - 8 camera controllers
- `collision.py` (709 lines) - Sphere-cast collision
- `effects.py` (1,317 lines) - Shake, DOF, motion blur, vignette
- `blending.py` (1,091 lines) - 12 blend curves, split-screen
- `rails.py` (1,346 lines) - Spline rails, dolly, crane

---

## Classification

**ALL REAL** - No stubs found. All files contain production-ready implementations with complete algorithms.
