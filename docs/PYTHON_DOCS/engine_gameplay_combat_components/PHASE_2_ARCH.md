# PHASE 2 ARCHITECTURE: Component Systems

## Scope

This phase covers the five ECS-style component files in `engine/gameplay/components/`:
- stats.py
- movement.py
- health.py
- team.py
- transform.py

---

## Architecture Decisions

### ADR-020: ECS Component Pattern

**Context**: Need composable gameplay building blocks attached to entities.

**Decision**: Use data-focused components with minimal behavior, managed by systems.

**Component Characteristics**:
- Single responsibility (stats, movement, health, etc.)
- Dirty tracking via Trinity descriptors
- Serialization for save/load and networking
- Callback hooks for external systems

**Rationale**: Decouples data from behavior, enables optimization (batch processing), supports networking.

---

### ADR-021: Stat Modifier Stacking Order

**Context**: Multiple modifiers (equipment, buffs, debuffs) affect stats.

**Decision**: Fixed stacking order: OVERRIDE > FLAT > PERCENT_BASE > MULTIPLY > PERCENT_TOTAL

**Algorithm** (stats.py lines 114-156):
```python
def _compute_value(self) -> None:
    # 1. Check for override first
    override_mods = [m for m in self.modifiers if m.modifier_type == ModifierType.OVERRIDE]
    if override_mods:
        override_mods.sort(key=lambda m: m.priority, reverse=True)
        self._cached_value = override_mods[0].get_total_value()
        return
    
    result = self.base_value
    
    # 2. FLAT: Add to base
    for mod in self.modifiers:
        if mod.modifier_type == ModifierType.FLAT:
            result += mod.get_total_value()
    
    # 3. PERCENT_BASE: Multiply base only
    base_percent = sum(m.get_total_value() for m in self.modifiers 
                       if m.modifier_type == ModifierType.PERCENT_BASE)
    result += self.base_value * base_percent
    
    # 4. MULTIPLY: Stack multiplicatively
    for mod in self.modifiers:
        if mod.modifier_type == ModifierType.MULTIPLY:
            result *= mod.get_total_value()
    
    # 5. PERCENT_TOTAL: Multiply final
    total_percent = sum(m.get_total_value() for m in self.modifiers 
                        if m.modifier_type == ModifierType.PERCENT_TOTAL)
    result *= (1.0 + total_percent)
    
    self._cached_value = result
```

**Rationale**: Industry-standard RPG formula. Predictable stacking prevents balance issues.

---

### ADR-022: Modifier Cache Invalidation

**Context**: Stat computation is expensive with many modifiers.

**Decision**: Cache computed value, invalidate on any modifier change.

**Invalidation Triggers** (lines 175-182):
- Modifier added/removed
- Modifier value changed
- Base value changed
- Timed modifier expired

**Rationale**: Modifier changes are rare (equipment change, buff cast), reads are frequent (every frame).

---

### ADR-023: Timed Modifier Expiration

**Context**: Buffs/debuffs have durations.

**Decision**: Store end timestamp, cleanup on next access.

**Algorithm** (stats.py lines 574-606):
```python
def _cleanup_expired_modifiers(self, current_time: float):
    expired = [m for m in self.modifiers if m.end_time and m.end_time <= current_time]
    for mod in expired:
        self.remove_modifier(mod)
```

**Rationale**: Lazy cleanup avoids per-frame iteration over all entities.

---

### ADR-024: Derived Stat Computation

**Context**: Some stats depend on others (e.g., attack_speed from dexterity).

**Decision**: Define dependencies, recompute when dependencies change.

**Implementation** (stats.py lines 352-368):
```python
class DerivedStat:
    dependencies: List[str]  # ["dexterity", "agility"]
    formula: Callable[[Dict[str, float]], float]
    
    def compute(self, stats: Dict[str, float]) -> float:
        return self.formula(stats)
```

**Rationale**: Explicit dependencies enable targeted invalidation.

---

### ADR-025: Movement Mode System

**Context**: Characters have different movement capabilities (walk, run, swim, fly).

**Decision**: Per-mode settings with mode switching.

**Modes** (movement.py lines 124-135):
- WALKING: Ground movement, jump enabled
- RUNNING: Faster ground, more air control
- SWIMMING: Buoyancy, stamina drain
- FLYING: Full 3D movement, no gravity

**Per-Mode Settings**:
```python
@dataclass
class MovementSettings:
    max_speed: float
    acceleration: float
    deceleration: float
    can_jump: bool
    air_control: float
    gravity_scale: float
```

**Rationale**: Different modes have fundamentally different physics.

---

### ADR-026: Jump Mechanics (Coyote Time + Jump Buffering)

**Context**: Precise platforming feels unfair without input forgiveness.

**Decision**: Implement both coyote time and jump buffering.

**Coyote Time**: Allow jump briefly after leaving ground.
```python
def can_use_coyote_time(self, current_time: float) -> bool:
    return (current_time - self._last_grounded_time) <= self.coyote_time_window
```

**Jump Buffering**: Remember jump input, execute when possible.
```python
def request_jump(self, current_time: float = 0.0) -> bool:
    self._jump_requested = True
    self._jump_request_time = current_time
    # Execute immediately or buffer for later
```

**Rationale**: Industry-standard platformer feel. Configurable for different game styles.

---

### ADR-027: Air Control Factor

**Context**: How much control does player have while airborne?

**Decision**: Configurable air_control multiplier (0.0 to 1.0).

**Implementation** (movement.py line 535):
```python
effective_accel = self.current_settings.acceleration * (
    1.0 if self._is_grounded else self.current_settings.air_control
)
```

**Rationale**: Platformers want high air control, realistic games want low.

---

### ADR-028: Damage Type System

**Context**: Different damage types (physical, fire, etc.) should have resistances.

**Decision**: Enum-based damage types with per-type resistance map.

**Types** (health.py):
- PHYSICAL
- FIRE
- ICE
- LIGHTNING
- POISON
- MAGIC
- TRUE (ignores resistance and armor)

**Resistance Calculation** (health.py lines 227-303):
```python
if damage_type != DamageType.TRUE:
    resistance = self._resistances.get(damage_type, 0.0)
    final_damage *= (1.0 - min(resistance, MAX_RESISTANCE_CAP))
```

**Rationale**: Standard ARPG elemental system. TRUE damage prevents full immunity stacking.

---

### ADR-029: Damage/Heal History

**Context**: Need to track recent damage for assist attribution and combat logging.

**Decision**: Ring buffer of recent events with configurable size.

**Implementation** (health.py lines 551-570):
```python
@dataclass
class DamageEvent:
    amount: float
    damage_type: DamageType
    source_id: Optional[int]
    timestamp: float

# Store last N events
self._damage_history: deque = deque(maxlen=50)
```

**Rationale**: Fixed memory, fast append, supports assist window queries.

---

### ADR-030: IFF Tag Bitflags

**Context**: Entity can have multiple identity tags (PLAYER, FRIEND, AI, etc.).

**Decision**: IntFlag enum for bitwise operations.

**Implementation** (team.py lines 34-45):
```python
class IFFResponse(IntFlag):
    NONE = 0
    FRIEND = 1 << 0
    FOE = 1 << 1
    UNKNOWN = 1 << 2
    CIVILIAN = 1 << 3
    OBJECTIVE = 1 << 4
    HAZARD = 1 << 5
    PLAYER = 1 << 6
    AI = 1 << 7
```

**Usage**:
```python
# Check multiple tags
if entity.iff_tags & (IFFResponse.PLAYER | IFFResponse.FRIEND):
    # Friendly player
```

**Rationale**: Efficient multi-tag queries without multiple checks.

---

### ADR-031: Team Registry Singleton

**Context**: Need global team/faction definitions shared by all entities.

**Decision**: Singleton registry with faction hierarchy.

**Implementation** (team.py lines 102-276):
```python
class TeamRegistry:
    _instance: Optional["TeamRegistry"] = None
    
    @classmethod
    def instance(cls) -> "TeamRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
```

**Contents**:
- Faction definitions (name, color, parent faction)
- Team definitions (subset of faction)
- Relationship matrix

**Rationale**: Avoids passing registry reference through all code. Single source of truth.

---

### ADR-032: Secondary Team Memberships

**Context**: Entity may belong to multiple teams (main team + temporary squad).

**Decision**: Primary team plus list of secondary memberships.

**Implementation** (team.py lines 392-419):
```python
@dataclass
class TeamComponent:
    primary_team_id: int
    secondary_teams: List[int] = field(default_factory=list)
    
    def is_member_of(self, team_id: int) -> bool:
        return team_id == self.primary_team_id or team_id in self.secondary_teams
```

**Rationale**: Supports complex scenarios (temporary alliances, escort missions).

---

### ADR-033: Hierarchical Transform

**Context**: Child transforms should follow parent transforms.

**Decision**: Parent-child tree with weak references to prevent cycles.

**Implementation** (transform.py lines 130-220):
```python
class TransformComponent:
    _parent: Optional[weakref.ref["TransformComponent"]] = None
    _children: List[weakref.ref["TransformComponent"]] = []
    
    @property
    def parent(self) -> Optional["TransformComponent"]:
        return self._parent() if self._parent else None
```

**Rationale**: Weak references allow garbage collection, prevent reference cycles.

---

### ADR-034: World Matrix Caching

**Context**: World matrix computation is expensive for deep hierarchies.

**Decision**: Lazy evaluation with dirty propagation.

**Algorithm** (transform.py lines 278-287):
```python
@property
def world_matrix(self) -> Mat4:
    if self._world_matrix_dirty or self._world_matrix_cache is None:
        if self.parent is not None:
            self._world_matrix_cache = self.parent.world_matrix @ self.local_matrix
        else:
            self._world_matrix_cache = self.local_matrix
        self._world_matrix_dirty = False
    return self._world_matrix_cache
```

**Dirty Propagation** (lines 460-469):
```python
def _mark_dirty(self):
    self._world_matrix_dirty = True
    for child_ref in self._children:
        child = child_ref()
        if child:
            child._mark_dirty()
```

**Rationale**: Only recompute when needed, propagate invalidation down tree.

---

### ADR-035: Look-At Rotation

**Context**: Common operation: rotate to face a target point.

**Decision**: Compute rotation quaternion from forward vector.

**Algorithm** (transform.py lines 384-435):
```python
def look_at(self, target: Vec3, up: Vec3 = Vec3.UP) -> None:
    direction = (target - self.position).normalized()
    # Compute rotation from forward to direction
    rotation = Quat.from_look_rotation(direction, up)
    self.rotation = rotation
```

**Rationale**: Standard 3D graphics operation, avoid manual matrix math.

---

### ADR-036: Coordinate Space Transformations

**Context**: Need to convert between local, world, and self coordinate spaces.

**Decision**: Explicit TransformSpace enum with conversion methods.

**Spaces**:
- LOCAL: Relative to parent
- WORLD: Absolute position
- SELF: Object's own coordinate system

**Conversions** (transform.py lines 336-383):
```python
def transform_point(self, point: Vec3, from_space: TransformSpace, to_space: TransformSpace) -> Vec3:
    # Convert to world, then to target space
    world_point = self._to_world(point, from_space)
    return self._from_world(world_point, to_space)
```

**Rationale**: Explicit about coordinate spaces prevents bugs.

---

## Integration Points

### Components -> Trinity

- `TrackedDescriptor` from `trinity.descriptors` for dirty tracking
- Math primitives from `engine.core.math` (Vec3, Quat, Mat4)
- Constants from `engine.gameplay.components.constants`

### Components -> Combat

- `health.py` in components is RPG-focused (damage types, resistances)
- `health.py` in combat is multiplayer-focused (shields, invulnerability)
- Both can coexist on same entity for different purposes

---

## File Dependencies

```
stats.py
    └── trinity.descriptors (TrackedDescriptor)

movement.py
    └── engine.core.math.vec (Vec3)
    └── engine.gameplay.components.constants

health.py (components)
    └── trinity.descriptors (TrackedDescriptor)
    └── engine.gameplay.components.constants

team.py
    └── (standalone with IntFlag)

transform.py
    └── engine.core.math.vec (Vec3)
    └── engine.core.math.quat (Quat)
    └── engine.core.math.mat (Mat4)
```
