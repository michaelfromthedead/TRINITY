# Investigation: engine/gameplay/components

## Summary
The gameplay components package is a fully implemented ECS component system with five production-quality components (Health, Movement, Transform, Team, Stats) plus a centralized constants module. Each component uses Trinity's TrackedDescriptor system for dirty tracking, includes comprehensive callbacks, serialization support, and proper integration patterns. This is a substantial, well-architected implementation with approximately 3,000 lines of real game logic.

## Files
| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `__init__.py` | 93 | COMPLETE | Exports all public APIs cleanly |
| `constants.py` | 172 | COMPLETE | Centralized magic numbers, well-organized |
| `health.py` | 659 | COMPLETE | Full damage system with resistances, shields, regen |
| `movement.py` | 684 | COMPLETE | 10 movement modes, jump buffering, coyote time |
| `transform.py` | 572 | COMPLETE | Hierarchical transforms with matrix caching |
| `team.py` | 624 | COMPLETE | Faction system, IFF tags, relationship queries |
| `stats.py` | 757 | COMPLETE | Full RPG attribute system with modifiers |

## Component Types
- **HealthComponent**: Current/max health, damage types (7 types), resistances, armor, shields, regeneration, invulnerability, death/revival callbacks
- **MovementComponent**: Velocity, 10 movement modes (walking, running, sprinting, crouching, swimming, flying, falling, climbing, sliding, custom), jump mechanics with coyote time and jump buffering, ground detection, input handling
- **TransformComponent**: Position/rotation/scale, parent-child hierarchy with weak references, world/local matrix caching, coordinate space transforms, look-at, snapshot interpolation
- **TeamComponent**: Team/faction membership, IFF (Identification Friend or Foe) system with 8 tags, relationship queries (ally/enemy/neutral), secondary team memberships, custom relation overrides
- **StatsComponent**: Dynamic stat registration, 5 modifier types (flat, percent_base, percent_total, override, multiply), modifier stacking/duration, derived stats, cache invalidation

## Implementation
- Real component definitions? **yes** - Each component is a full class with __slots__, properties, methods
- Real data structures? **yes** - Extensive use of dataclasses, enums, typed dicts, callbacks
- Real ECS integration? **yes** - Uses TrackedDescriptor from trinity.descriptors for dirty tracking and network sync

## Verdict
**REAL IMPLEMENTATION**

This is production-quality game engine code with:
- Sophisticated damage calculation with resistance caps and armor penetration
- Complete character controller movement with multiple modes
- Full hierarchical transform system with matrix caching
- Faction/team system suitable for RTS or competitive games
- RPG-style stat system with modifier stacking and prioritization

## Evidence

### Health damage calculation (health.py:266-302):
```python
# Calculate final damage
final_damage = amount * self._damage_multiplier

# Apply resistance
if not ignore_resistance and damage_type != DamageType.TRUE:
    resistance = self._resistances.get(damage_type, 0.0)
    final_damage *= (1.0 - min(resistance, HealthConstants.MAX_RESISTANCE_CAP))

# Apply armor
if not ignore_armor and damage_type != DamageType.TRUE:
    final_damage = max(0, final_damage - self._armor)

# Absorb with shield first
if self._shield > 0 and final_damage > 0:
    shield_absorbed = min(self._shield, final_damage)
    self._shield -= shield_absorbed
    final_damage -= shield_absorbed
```

### Movement mode settings (movement.py:124-135):
```python
DEFAULT_MODE_SETTINGS: Dict[MovementMode, MovementSettings] = {
    MovementMode.WALKING: MovementSettings(max_speed=4.0, acceleration=15.0),
    MovementMode.RUNNING: MovementSettings(max_speed=7.0, acceleration=20.0),
    MovementMode.SPRINTING: MovementSettings(max_speed=10.0, acceleration=25.0, turn_rate=180.0),
    MovementMode.CROUCHING: MovementSettings(max_speed=2.0, acceleration=10.0, height_scale=0.5, can_jump=False),
    MovementMode.SWIMMING: MovementSettings(max_speed=3.0, acceleration=8.0, gravity_scale=0.1, jump_velocity=4.0),
    MovementMode.FLYING: MovementSettings(max_speed=8.0, acceleration=12.0, gravity_scale=0.0, air_control=1.0),
    MovementMode.FALLING: MovementSettings(max_speed=50.0, acceleration=0.0, air_control=0.2),
    MovementMode.CLIMBING: MovementSettings(max_speed=2.0, acceleration=10.0, gravity_scale=0.0),
    MovementMode.SLIDING: MovementSettings(max_speed=12.0, acceleration=5.0, deceleration=3.0),
    MovementMode.CUSTOM: MovementSettings(),
}
```

### Stat modifier computation (stats.py:114-156):
```python
def _compute_value(self) -> None:
    # Check for override first
    override_mods = [m for m in self.modifiers if m.modifier_type == ModifierType.OVERRIDE]
    if override_mods:
        override_mods.sort(key=lambda m: m.priority, reverse=True)
        self._cached_value = max(self.min_value, min(self.max_value, override_mods[0].get_total_value()))
        return

    # Start with base value
    result = self.base_value

    # Apply FLAT modifiers
    for mod in self.modifiers:
        if mod.modifier_type == ModifierType.FLAT:
            result += mod.get_total_value()

    # Apply PERCENT_BASE modifiers (multiplicative)
    percent_base = 1.0
    for mod in self.modifiers:
        if mod.modifier_type == ModifierType.PERCENT_BASE:
            percent_base += mod.get_total_value() / 100.0
    result = self.base_value * percent_base + (result - self.base_value)

    # Apply MULTIPLY modifiers
    for mod in self.modifiers:
        if mod.modifier_type == ModifierType.MULTIPLY:
            result *= mod.get_total_value()

    # Apply PERCENT_TOTAL modifiers
    for mod in self.modifiers:
        if mod.modifier_type == ModifierType.PERCENT_TOTAL:
            result *= (1.0 + mod.get_total_value() / 100.0)
```

### TrackedDescriptor integration (health.py:88-106):
```python
# Tracked descriptor for current health with clamping
current_health = TrackedDescriptor(
    field_type=float,
    use_bitmask=True,
    field_offset=0,
)

max_health = TrackedDescriptor(
    field_type=float,
    use_bitmask=True,
    field_offset=1,
)

regen_rate = TrackedDescriptor(
    field_type=float,
    use_bitmask=True,
    field_offset=2,
)
```
