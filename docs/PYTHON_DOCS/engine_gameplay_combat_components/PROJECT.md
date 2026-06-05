# PROJECT: engine_gameplay_combat_components

## Source Investigation
- **Document**: docs/investigation/engine_gameplay_combat_components.md
- **Date**: 2026-05-22
- **Classification**: REAL IMPLEMENTATION
- **Total Lines Analyzed**: ~9,805 (6,343 combat + 3,462 components)

---

## Scope

This project covers the gameplay systems in two modules:

### engine/gameplay/combat (~6,343 lines)
Seven production-ready combat system files:
1. **scoring.py** (1,187 lines) - Kill/death/assist attribution, killstreaks, multi-kills, leaderboards
2. **hitbox.py** (1,029 lines) - Hitbox/hurtbox collision, AABB intersection, priority resolution
3. **health.py** (982 lines) - Health component with shields, invulnerability, regeneration
4. **spawn_manager.py** (818 lines) - Spawn point selection with 5 strategies
5. **death.py** (758 lines) - Death state machine, respawn queue, cleanup handlers
6. **teams.py** (749 lines) - Team/faction system, IFF checks, friendly fire
7. **modes/deathmatch.py** (342 lines) - Game mode with win conditions, streak bonuses

### engine/gameplay/components (~3,462 lines)
Five production-ready ECS-style components:
1. **stats.py** (756 lines) - Stat/attribute system with modifier stacking
2. **movement.py** (683 lines) - Character movement with jump mechanics
3. **health.py** (658 lines) - Health component with damage types, resistances
4. **team.py** (623 lines) - Team component with IFF tag system
5. **transform.py** (571 lines) - Spatial transform with parent-child hierarchy

---

## Goals

1. **Maintain Production Quality**: All code is production-ready with comprehensive docstrings, type hints, error handling, and serialization
2. **Preserve Integration Points**: Both modules integrate via Trinity descriptors, entity IDs, event callbacks, and serialization patterns
3. **Document Architecture Decisions**: Capture the design patterns used (ECS, event systems, dirty tracking)
4. **Ensure Test Coverage**: Validate all algorithms and edge cases

---

## Constraints

1. **Engine Integration Required**: Combat and components depend on:
   - `engine.gameplay.combat.constants` - Shared constants
   - `engine.gameplay.combat.damage` - DamageInfo type
   - `engine.gameplay.combat.game_mode` - Base class for modes
   - `trinity.descriptors` - Dirty tracking
   - `engine.core.math.vec/quat/mat` - Math primitives
   - `engine.gameplay.components.constants` - Component constants

2. **Cross-Module Consistency**: Both modules must maintain compatible:
   - Entity ID conventions (string/int)
   - Callback patterns (`_on_*` lists)
   - Serialization format (`to_dict()`/`from_dict()`)

3. **No Breaking Changes**: These are production systems in active use

---

## Acceptance Criteria

### Phase 1: Combat Systems
- [ ] scoring.py algorithms validated (multi-kill, killstreak, assist attribution)
- [ ] hitbox.py collision detection verified (AABB, priority, counter-hit)
- [ ] health.py shield/invulnerability logic confirmed
- [ ] spawn_manager.py selection strategies tested
- [ ] death.py state machine transitions verified
- [ ] teams.py IFF checks and relationships validated
- [ ] deathmatch.py win conditions and bonuses confirmed

### Phase 2: Component Systems
- [ ] stats.py modifier stacking order verified (OVERRIDE > FLAT > PERCENT_BASE > MULTIPLY > PERCENT_TOTAL)
- [ ] movement.py jump mechanics tested (coyote time, jump buffering)
- [ ] health.py damage type resistance system validated
- [ ] team.py IFF tag bitflag system confirmed
- [ ] transform.py hierarchy dirty propagation verified

### Phase 3: Integration
- [ ] Cross-module entity ID consistency verified
- [ ] Event callback patterns consistent across modules
- [ ] Serialization round-trip tested for all components
- [ ] Trinity descriptor integration confirmed

---

## Code Metrics Summary

| File | Classes | Methods | Lines | Complexity |
|------|---------|---------|-------|------------|
| scoring.py | 7 | 45+ | 1,187 | High |
| hitbox.py | 8 | 55+ | 1,029 | High |
| health.py (combat) | 4 | 35+ | 982 | Medium |
| spawn_manager.py | 4 | 40+ | 818 | High |
| death.py | 6 | 30+ | 758 | Medium |
| teams.py | 5 | 35+ | 749 | Medium |
| deathmatch.py | 2 | 15+ | 342 | Low |
| stats.py | 4 | 40+ | 756 | Medium |
| movement.py | 4 | 45+ | 683 | Medium |
| health.py (components) | 4 | 35+ | 658 | Medium |
| team.py | 5 | 40+ | 623 | Medium |
| transform.py | 3 | 40+ | 571 | Medium |
| **TOTAL** | **56** | **455+** | **9,156** | - |
