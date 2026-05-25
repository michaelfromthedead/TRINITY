# PHASE 1 ARCHITECTURE: Abilities Subsystem

**Phase**: 1 of 3
**Subsystem**: engine/gameplay/abilities
**Lines**: 3,136
**Status**: REAL IMPLEMENTATION

---

## 1. Overview

The abilities subsystem implements a GAS-style (Gameplay Ability System) architecture for managing character attributes, gameplay effects, targeting, and gameplay tags.

---

## 2. Module Structure

```
engine/gameplay/abilities/
    __init__.py          # Package marker
    attributes.py        # 592 lines - Attribute system
    effects.py           # 829 lines - Effect system
    targeting.py         # 823 lines - Targeting system
    tags.py              # 575 lines - Gameplay tags
    constants.py         # 322 lines - Configuration
```

---

## 3. Component Architecture

### 3.1 Attribute System (attributes.py)

```
Attribute
    |-- base_value: float
    |-- current_value: float (computed)
    |-- min_value / max_value: float
    |-- _modifiers: List[AttributeModifier]
    |-- _dirty: bool (caching)

AttributeModifier
    |-- source: str
    |-- magnitude: float
    |-- operation: ModifierOperation (ADD, MULTIPLY, OVERRIDE)
    |-- order: int

AttributeSet
    |-- _attributes: Dict[str, Attribute]
    |-- _derived: Dict[str, DerivedAttribute]
    |-- dependency tracking

DerivedAttribute
    |-- formula: Callable
    |-- dependencies: List[str]
```

**Algorithm: Modifier Order of Operations**
1. ADD_BASE: Sum all additive modifiers to base
2. MULTIPLY_BASE: Apply multiplicative modifiers to result
3. ADD_BONUS: Add bonus modifiers
4. MULTIPLY_BONUS: Apply bonus multipliers
5. OVERRIDE: If present, replace with override value
6. Clamp to min/max bounds

### 3.2 Effect System (effects.py)

```
GameplayEffect (abstract)
    |-- tags: GameplayTagContainer (required/blocked)
    |-- apply() / remove()

InstantEffect
    |-- One-shot attribute change

DurationEffect
    |-- duration: float
    |-- remaining_time: float
    |-- tick()

InfiniteEffect
    |-- Until explicitly removed

PeriodicEffect
    |-- tick_rate: float
    |-- time_since_tick: float
    |-- _execute_tick()

EffectContainer
    |-- _active_effects: List[GameplayEffect]
    |-- add_effect() / remove_effect()
    |-- tick()
```

**Algorithm: Periodic Tick**
```python
while time_since_tick >= tick_rate:
    time_since_tick -= tick_rate
    _execute_tick(attributes)
```

### 3.3 Targeting System (targeting.py)

```
TargetingSystem (abstract)
    |-- max_targets: int
    |-- range: float
    |-- target_filter: TargetFilter

SelfTargeting      -> Self only
ActorTargeting     -> Single actor
PointTargeting     -> World position
AreaTargeting      -> AOE with shape
ConfirmationTargeting -> Wrapper requiring confirm

AreaShape (enum)
    |-- CIRCLE
    |-- CONE
    |-- RECTANGLE
    |-- LINE
    |-- CAPSULE
```

**Geometry Algorithms**:
- Circle: `distance_squared <= radius_squared`
- Cone: `dot(direction, to_target) >= cos(half_angle)`
- Rectangle: Axis projection bounds check
- Line: Point-to-segment distance
- Capsule: Swept sphere (line + radius)

### 3.4 Gameplay Tags (tags.py)

```
GameplayTag
    |-- name: str (e.g., "ability.offensive.fire")
    |-- parts: List[str] (cached split)
    |-- matches(pattern) with wildcards

GameplayTagContainer
    |-- _tags: Set[GameplayTag]
    |-- has_tag() / has_any() / has_all()

GameplayTagQuery
    |-- all_of: List[GameplayTag]
    |-- any_of: List[GameplayTag]
    |-- none_of: List[GameplayTag]

GameplayTagRegistry
    |-- LRU cache for tag lookups
```

---

## 4. Integration Points

| Component | Trinity Integration |
|-----------|---------------------|
| Attribute | ComponentMeta registers, TrackedDescriptor on current_value |
| Effect | ComponentMeta registers, EventMeta for effect events |
| TargetFilter | Function-based, no metaclass needed |
| GameplayTag | Pure data, no metaclass needed |

---

## 5. Dependencies

```python
from engine.core.math.vec import Vec3
from engine.gameplay.abilities.constants import *
from engine.gameplay.entity import Actor
```

---

## 6. Design Decisions

### 6.1 Why Modifier Order of Operations?
Industry standard (D&D, Diablo, WoW). Predictable stacking behavior. Easy to debug.

### 6.2 Why Four Effect Types?
Covers all common gameplay patterns:
- Instant: Damage, heal
- Duration: Buffs, debuffs
- Infinite: Passives, equipment
- Periodic: DOT, HOT

### 6.3 Why Hierarchical Tags?
Enables both specific and broad matching. `ability.*` matches all abilities. `ability.offensive.fire` matches exactly one.
