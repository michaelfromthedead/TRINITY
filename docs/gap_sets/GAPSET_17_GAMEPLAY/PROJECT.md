# GAPSET_17_GAMEPLAY — Project Overview

## Scope

The Gameplay gapset covers all runtime gameplay systems for the Trinity Engine: entity framework, input, camera, AI (3 tiers), navigation, abilities, inventory/economy, quests/dialogue, and combat/game modes. The codebase is implemented in Python under `engine/gameplay/`.

## Goal

Provide a complete, production-ready gameplay layer supporting:
- UE5-inspired Actor/Component entity model with lifecycle management
- Multi-device input system with action/axis mapping and context stacks
- 8-mode camera system with collision, effects, rails, and blending
- 3-tier AI architecture (BT/Utility → GOAP → MCTS [planned])
- Full navigation pipeline (NavMesh generation, 5 pathfinding algorithms, steering, avoidance)
- GAS-style ability system with attributes, effects, tags, and cooldowns
- Inventory, equipment, loot, crafting, and economy systems
- Quest and dialogue systems with graph-based narrative flow
- Combat system with damage, death, scoring, and game mode framework

## Current State

| Phase | Description | Tasks | REAL [x] | PARTIAL [~] | ABSENT [-] | Status |
|-------|-------------|-------|---------|-------------|------------|--------|
| 1 | Entity Framework & Core | 12 | 11 | 1 | 0 | COMPLETE |
| 2 | Input & Camera | 14 | 14 | 0 | 0 | COMPLETE |
| 3 | AI Tier 1 (BT + Utility + Perception) | 23 | 20 | 2 | 1 | COMPLETE (gap: Influence Maps) |
| 4 | AI Tier 2 (GOAP) | 7 | 7 | 0 | 0 | COMPLETE |
| 5 | Navigation | 19 | 19 | 0 | 0 | COMPLETE |
| 6 | AI Tier 3 (MCTS) | 8 | 0 | 0 | 8 | NOT STARTED |
| 7 | Ability System (GAS-style) | 11 | 11 | 0 | 0 | COMPLETE |
| 8 | Inventory & Economy | 14 | 14 | 0 | 0 | COMPLETE |
| 9 | Quests & Dialogue | 10 | 10 | 0 | 0 | COMPLETE |
| 10 | Combat & Game Modes | 12 | 9 | 3 | 0 | COMPLETE (partial: CTF/KOTH/BR) |
| **Total** | | **130** | **115** | **6** | **9** | |

## Key Metrics

- **Total source files investigated**: 30+ across 13 module directories
- **Total tasks**: 130 across 10 phases
- **Implemented (REAL)**: 115 (88.5%)
- **Partially implemented**: 6 (4.6%)
- **Not implemented**: 9 (6.9%)
- **SLOC in gameplay directory**: ~20,000+ lines of Python

## Architecture

```
gameplay/
├── entity/        — Actor/Component model, lifecycle, controllers, prefabs, movement
├── input/         — Device management, action/axis mapping, context stack
├── camera/        — 8 camera modes, collision, effects, rails, blending
├── fsm/           — FSM, HFSM, pushdown automaton
├── ai/            — BT, Utility AI, GOAP, Perception, Knowledge, CombatAI
├── nav/           — NavMesh, pathfinding, steering, avoidance, nav links, smart objects
├── abilities/     — GAS-style ability system, effects, attributes, tags, targeting
├── economy/       — Inventory, items, equipment, loot, crafting, trading
├── quest/         — Quest definitions, objectives, journal, tracker, dialogue
└── combat/        — Health, damage, death, scoring, game modes
```

## Dependencies

- **Foundation systems**: EngineMeta, ComponentMeta, @component, @decorator, TrackedDescriptor, EventLog, ResourceMeta, StateMeta
- **Math**: Vec3, AABB, Ray from omega core
- **Physics/Rendering**: Physics raycast integration, wgpu renderer (via GAPSET_3_BRIDGE shared infrastructure)
- **Audio**: Audio system for sound perception (T-GP-3.14)

## Next Steps

1. **Implement MCTS** (Phase 6) — 8 tasks, full solver implementation from tree node through fallback chain
2. **Implement Influence Maps** (T-GP-3.19) — spatial grid with propagation and multiple layers
3. **Create @spawner decorator** (T-GP-1.6) — pool_size/spawn_rate/max_alive parameters
4. **Implement game mode subclasses** (T-GP-10.10/10.11/10.12) — CTF, KOTH, Battle Royale
5. **Refactor ai/__init__.py** — decompose 1185-line combined file into separate modules
6. **Verify and wire** @ai_debug decorator and AI EventLog integration (T-GP-3.22/3.23)
