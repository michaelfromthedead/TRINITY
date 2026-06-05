# PHASE 2 ARCHITECTURE: Gameplay & World Systems

## Scope

Gameplay and world-building decorators:
- `gameplay.py` (349 lines) - Tier 33-40
- `world_building.py` (339 lines)
- `game_ai.py` (247 lines)
- `state_machine.py` (196 lines)
- `procedural.py` (179 lines)
- `spatial.py` (165 lines)

## Architecture Pattern

All files follow the 6-part decorator pattern with domain-specific `VALID_*` constants.

## Component: Gameplay Decorators

**File**: `trinity/decorators/gameplay.py` (349 lines)

**Decorators**: `@ability`, `@buff`, `@gameplay_tag`, `@spawner`, `@interactable`, `@quest`

**Architecture**:
- Full validation chains for each decorator
- Uses VALID_STACKING for buff modes: `{"replace", "stack", "refresh", "unique"}`
- Ability cost validation (dict[str, float])
- Cooldown validation (>= 0)

**Example Pattern**:
```python
def _validate_ability(cost={}, cooldown=0.0, tags=set(), blocked_by=set(), **_):
    if cooldown < 0:
        raise ValueError(f"@ability: cooldown must be >= 0, got {cooldown}")

def _ability_steps(params):
    return [
        Step(Op.TAG, {"key": "ability", "value": True}),
        Step(Op.TAG, {"key": "ability_cost", "value": dict(cost)}),
        Step(Op.TAG, {"key": "ability_cooldown", "value": cooldown}),
        Step(Op.TAG, {"key": "ability_tags", "value": set(tags)}),
        Step(Op.TAG, {"key": "ability_blocked_by", "value": set(blocked_by)}),
        Step(Op.REGISTER, {"registry": "gameplay"}),
    ]
```

## Component: World Building Decorators

**File**: `trinity/decorators/world_building.py` (339 lines)

**Decorators**: `@foliage_type`, `@procedural_placement`, `@level_instance`, `@water_body`, `@navmesh_modifier`, `@trigger_volume`

**Architecture**:
- Domain-specific VALID_* constants for each decorator
- Foliage types: vegetation classifications
- Water body types: ocean, lake, river, pond
- Navmesh modifiers: walkable, obstacle, dynamic

**Key Constants**:
```python
VALID_FOLIAGE_TYPES = frozenset({"tree", "bush", "grass", "flower", "rock"})
VALID_WATER_TYPES = frozenset({"ocean", "lake", "river", "pond", "stream"})
VALID_NAVMESH_MODIFIERS = frozenset({"walkable", "obstacle", "dynamic", "expensive"})
```

## Component: Game AI Decorators

**File**: `trinity/decorators/game_ai.py` (247 lines)

**Decorators**: `@behavior_tree`, `@utility_ai`, `@blackboard`, `@ai_debug`, `@perception`

**Architecture**:
- Full ops chain for AI systems
- Behavior tree node registration
- Utility AI scoring systems
- Blackboard for shared AI state
- Perception for sensory systems

**Pattern**:
```python
def _behavior_tree_steps(params):
    return [
        Step(Op.TAG, {"key": "behavior_tree", "value": params["root"]}),
        Step(Op.TAG, {"key": "bt_tick_rate", "value": params.get("tick_rate", 0.1)}),
        Step(Op.REGISTER, {"registry": "ai"}),
    ]
```

## Component: State Machine Decorators

**File**: `trinity/decorators/state_machine.py` (196 lines)

**Decorators**: `@state_machine`, `@on_enter`, `@on_exit`

**Architecture**:
- State validation against defined states set
- Transition validation
- Entry/exit hooks for state changes

**Validation**:
```python
def _validate_state_machine(states=[], initial=None, **_):
    states_set = set(states)
    if initial not in states_set:
        raise ValueError(
            f"@state_machine: initial state '{initial}' is not in states {sorted(states_set)}"
        )
```

## Component: Procedural Decorators

**File**: `trinity/decorators/procedural.py` (179 lines)

**Decorators**: `@seeded`, `@procedural`, `@constraint`

**Architecture**:
- Seed source validation
- Deterministic generation markers
- Constraint-based generation

**Valid Seed Sources**:
```python
VALID_SEED_SOURCES = frozenset({"world", "local", "instance", "fixed"})
```

## Component: Spatial Decorators

**File**: `trinity/decorators/spatial.py` (165 lines)

**Decorators**: `@spatial`, `@partitioned`

**Architecture**:
- Spatial structure validation
- Partitioning for spatial queries

**Valid Structures**:
```python
VALID_SPATIAL_STRUCTURES = frozenset({"octree", "bvh", "grid", "kdtree"})
```

## Op Types Used

| Op | Purpose | Files |
|----|---------|-------|
| `Op.TAG` | Store gameplay/world metadata | All |
| `Op.REGISTER` | Register in "gameplay", "world", "ai" | All |
| `Op.HOOK` | State enter/exit callbacks | state_machine.py |
| `Op.VALIDATE` | Runtime constraint checking | procedural.py |

## Dependencies

- gameplay.py depends on lifecycle (for spawn/despawn)
- world_building.py depends on spatial (for spatial queries)
- game_ai.py depends on state_machine
- procedural.py is independent
- spatial.py is independent

## Key Decisions

1. **VALID_* constants**: Each domain defines its own valid options as frozensets
2. **Error messages**: All validators produce actionable messages with valid options listed
3. **Registry separation**: gameplay, world, ai registries keep domains isolated
4. **State validation**: Initial state must be in states set (compile-time check)
