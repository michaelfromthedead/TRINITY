# PHASE 3 ARCHITECTURE: Integration

## Scope

This phase covers the integration points between combat and components modules, and their integration with the broader Trinity engine.

---

## Architecture Decisions

### ADR-040: Entity ID Convention

**Context**: Both modules reference entities by ID.

**Decision**: Support both string and int entity IDs.

**Usage**:
- Combat systems accept `int` for performance (network sync, leaderboards)
- Components accept `int` for consistency with ECS pattern
- String IDs used in serialization for human readability

**Rationale**: Flexibility for different use cases without breaking compatibility.

---

### ADR-041: Event Callback Pattern

**Context**: Both modules need to notify external systems of changes.

**Decision**: Consistent callback list pattern with exception handling.

**Pattern**:
```python
class Component:
    def __init__(self):
        self._on_change_callbacks: List[Callable] = []
    
    def register_callback(self, callback: Callable) -> None:
        self._on_change_callbacks.append(callback)
    
    def _emit_change(self, event: Any) -> None:
        for callback in self._on_change_callbacks:
            try:
                callback(event)
            except Exception:
                pass  # Log but don't cascade
```

**Rationale**: Uniform pattern across all modules, fault isolation.

---

### ADR-042: Serialization Contract

**Context**: Both modules need save/load and network sync.

**Decision**: All classes implement `to_dict()` and `from_dict()` methods.

**Contract**:
```python
class Component:
    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable dict."""
        ...
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Component":
        """Reconstruct from dict."""
        ...
```

**Requirements**:
- Output must be JSON-serializable
- Round-trip: `from_dict(to_dict(x)) == x`
- Version tolerance: unknown keys ignored
- Defaults for missing keys

**Rationale**: Enables save/load, network sync, debugging, modding.

---

### ADR-043: Trinity Descriptor Integration

**Context**: Components use Trinity descriptors for dirty tracking.

**Decision**: Use `TrackedDescriptor` for all mutable properties.

**Pattern**:
```python
from trinity.descriptors import TrackedDescriptor

class MovementComponent:
    velocity = TrackedDescriptor()  # Marks dirty on write
    position = TrackedDescriptor()
    
    @property
    def is_dirty(self) -> bool:
        return self._dirty_flags != 0
```

**Benefits**:
- Automatic dirty tracking without manual flags
- Integration with Trinity's sync system
- Consistent pattern across components

**Rationale**: Leverages engine infrastructure, reduces boilerplate.

---

### ADR-044: Dual Health Systems

**Context**: `health.py` exists in both combat and components.

**Decision**: Keep both, as they serve different purposes.

**combat/health.py**:
- Focus: Multiplayer competitive gameplay
- Features: Shield stacking, invulnerability sources, combat state
- Use case: FPS, battle royale

**components/health.py**:
- Focus: RPG damage calculation
- Features: Damage types, resistances, armor
- Use case: Action RPG, survival

**Integration**:
- Entity can have both components
- Combat health for PvP mechanics
- Components health for PvE/RPG mechanics
- Or use one exclusively based on game type

**Rationale**: Different games have fundamentally different health models.

---

### ADR-045: Dual Team Systems

**Context**: `team.py` (components) and `teams.py` (combat) overlap.

**Decision**: Component is entity data, system is manager.

**team.py (components)**:
- ECS component attached to entity
- Stores: team_id, IFF tags, secondary memberships
- Responsibility: Entity's team identity

**teams.py (combat)**:
- System managing all teams
- Stores: Team definitions, relationship matrix
- Responsibility: Team-level queries and rules

**Pattern**:
```python
# Component holds entity's team data
entity.team_component.primary_team_id = 1

# System handles relationships
team_system = TeamSystem.instance()
iff_result = team_system.check_iff(entity_a.id, entity_b.id)
```

**Rationale**: ECS pattern: components are data, systems are behavior.

---

### ADR-046: Math Library Integration

**Context**: Transform and hitbox need 3D math primitives.

**Decision**: Use engine math library from `engine.core.math`.

**Types**:
- `Vec3`: 3D vector (position, velocity, scale)
- `Quat`: Quaternion (rotation)
- `Mat4`: 4x4 matrix (transform)
- `BoundingBox`: AABB for collision

**Import Path**:
```python
from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.mat import Mat4
```

**Rationale**: Consistent math types, potential SIMD optimization.

---

### ADR-047: Constants Organization

**Context**: Magic numbers should be configurable.

**Decision**: Per-module constants files.

**Files**:
- `engine/gameplay/combat/constants.py`
- `engine/gameplay/components/constants.py`

**Pattern**:
```python
# constants.py
MAX_RESISTANCE_CAP = 0.9  # 90% max damage reduction
COYOTE_TIME_DEFAULT = 0.1  # seconds
MULTI_KILL_WINDOW = 3.0  # seconds
```

**Usage**:
```python
from engine.gameplay.components.constants import MAX_RESISTANCE_CAP

final_damage *= (1.0 - min(resistance, MAX_RESISTANCE_CAP))
```

**Rationale**: Tuning without code changes, clear documentation of limits.

---

### ADR-048: Cross-Module Dependencies

**Context**: Combat depends on damage types, game mode base class.

**Decision**: Define shared types in combat module.

**combat/damage.py**:
```python
@dataclass
class DamageInfo:
    amount: float
    damage_type: DamageType
    source_id: Optional[int]
    is_critical: bool
    is_headshot: bool
```

**combat/game_mode.py**:
```python
class GameMode(ABC):
    @abstractmethod
    def on_player_killed(self, ...): ...
    
    @abstractmethod
    def check_win_condition(self) -> Optional[WinResult]: ...
```

**Rationale**: Combat module owns competitive gameplay types.

---

## Integration Matrix

| Source | Target | Integration Type |
|--------|--------|-----------------|
| scoring.py | death.py | Events (kill/death) |
| hitbox.py | health.py | Damage application |
| hitbox.py | teams.py | IFF filtering |
| death.py | spawn_manager.py | Respawn requests |
| death.py | scoring.py | Death attribution |
| spawn_manager.py | teams.py | Team spawn filtering |
| deathmatch.py | scoring.py | Leaderboards |
| stats.py | health.py | Base health stat |
| movement.py | transform.py | Position updates |
| team.py | teams.py | Entity team data |

---

## Data Flow Diagrams

### Kill Event Flow
```
Hitbox detects collision
    ↓
teams.py: check_iff() → is this a valid target?
    ↓
health.py: take_damage() → apply damage
    ↓
death.py: trigger_death() → if health <= 0
    ↓
scoring.py: on_kill() → attribute kill, check streaks
    ↓
spawn_manager.py: queue_respawn() → after delay
```

### Movement Input Flow
```
Input system: movement input
    ↓
movement.py: process_input() → calculate velocity
    ↓
transform.py: update position
    ↓
Children transforms: dirty propagation
```

### Stat Modification Flow
```
Equipment change / buff applied
    ↓
stats.py: add_modifier() → invalidate cache
    ↓
stats.py: get_value() → recompute with modifiers
    ↓
health.py: on_max_health_changed() → adjust current health
```

---

## Serialization Schema

### Entity Snapshot
```json
{
  "entity_id": 12345,
  "components": {
    "transform": {
      "position": [0.0, 1.0, 0.0],
      "rotation": [0.0, 0.0, 0.0, 1.0],
      "scale": [1.0, 1.0, 1.0],
      "parent_id": null
    },
    "health": {
      "current": 80.0,
      "max": 100.0,
      "shield": 20.0,
      "resistances": {"FIRE": 0.5}
    },
    "stats": {
      "base_values": {"strength": 10, "dexterity": 8},
      "modifiers": [
        {"stat": "strength", "type": "FLAT", "value": 5, "source": "sword"}
      ]
    },
    "movement": {
      "velocity": [0.0, 0.0, 0.0],
      "mode": "WALKING",
      "is_grounded": true
    },
    "team": {
      "primary_team_id": 1,
      "secondary_teams": [],
      "iff_tags": 65
    }
  }
}
```

---

## Testing Strategy

### Unit Tests
- Each component in isolation
- Mock dependencies (descriptors, math)
- Verify algorithms match spec

### Integration Tests
- Combat flow: hitbox → health → death → scoring
- Component flow: stats → health, movement → transform
- Cross-module: team component ↔ team system

### System Tests
- Full deathmatch game simulation
- Serialization round-trip
- Performance benchmarks (1000 entities)
