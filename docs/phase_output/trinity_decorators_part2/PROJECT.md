# PROJECT: Trinity Decorators Part 2 - Ops-Based Decorator Framework

## Source Document
`docs/investigation/trinity_decorators_part2.md`

## Scope

Archaeological investigation of 39 decorator files in `trinity/decorators/` directory, totaling approximately 6,900 lines of code. This analysis covers the complete decorator framework spanning game engine functionality from assets to networking to gameplay systems.

## Classification

**100% REAL IMPLEMENTATIONS** - All 39 files contain fully implemented decorator systems using the ops-based architecture. Zero stub files, zero `pass` placeholders, zero `NotImplementedError` raises.

## Goals

1. Document the consistent architecture pattern used across all decorator files
2. Catalog the ~110 decorators organized by functional domain
3. Understand the ops-based Step system (TAG, REGISTER, HOOK, TRACK, VALIDATE, DESCRIBE)
4. Document the validation quality patterns and error message conventions
5. Identify special cases: stacks.py (composition), introspection.py (query API), debug_safety.py (mixed patterns)
6. Map tier distribution across files (Tier 7-51)
7. Document config dataclass patterns used in several files

## Constraints

- Analysis limited to existing decorator implementations (no new functionality)
- Must preserve the make_decorator pattern as canonical
- Validation must maintain specific, actionable error messages
- Registry registration patterns must remain consistent

## File Coverage

| Category | Files | Line Range |
|----------|-------|------------|
| Assets & Resources | assets.py (354), lod_streaming.py (292), prefabs.py (133) | 122-354 |
| Gameplay Systems | gameplay.py (349), game_ai.py (247), state_machine.py (196) | 196-349 |
| World Building | world_building.py (339), spatial.py (165), procedural.py (179) | 165-339 |
| Networking | network_extended.py (331), rpc.py (118) | 118-331 |
| Animation & IK | ik_procedural.py (288), animation.py (160), cinematics.py (155) | 155-288 |
| Debug & Safety | debug_safety.py (278), debug_extended.py (212), debug_cheat.py (191) | 191-278 |
| Persistence | save_system.py (236), transactions.py (137), replay.py (171) | 137-236 |
| Social & Economy | social.py (233), economy.py (228), achievements.py (197), analytics.py (195) | 195-233 |
| Localization & UI | localization.py (227), ui.py (165), accessibility.py (120) | 120-227 |
| Time & Error Handling | time.py (224), error_handling.py (216) | 216-224 |
| Build & Security | build_deploy.py (216), security.py (196), platform_specifics.py (111) | 111-216 |
| Narrative & Input | narrative.py (192), input.py (159) | 159-192 |
| Core Infrastructure | lifecycle.py (189), composition.py (138), stacks.py (122), introspection.py (99) | 99-189 |

## Acceptance Criteria

1. All 39 files documented with decorator counts and line totals
2. Architecture pattern (6-part structure) fully captured
3. All 6 Op types documented with usage patterns
4. Stack validation anti-patterns cataloged
5. Introspection API functions documented
6. Config dataclass patterns identified
7. Tier distribution mapped (Tier 7-51)
8. Decorator count by category verified (~110 total)
