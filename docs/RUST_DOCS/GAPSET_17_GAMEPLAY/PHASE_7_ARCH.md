# Phase 7: Ability System (GAS-Style) — Architecture

## Overview

Gameplay Ability System inspired by Unreal Engine's GAS. Provides ability definition, activation flow, gameplay effects, attribute system, targeting, tags, cooldowns, and cost management.

## Component Breakdown

### Ability Activation (`abilities/ability.py`)

```
4-Phase State Machine
├── ACTIVATE → checks can-activate (cooldown, cost, tags, blocked_by)
├── COMMIT → pays costs, starts cooldown
├── EXECUTE → runs effects, applies modifications
└── END → cleanup, event emission

@ability(cost, cooldown, tags, blocked_by) decorator
AbilitySystem (UPDATE phase, order 3)
├── Tick: process activations, tick effects, update cooldowns
└── Event emission per phase
```

### Gameplay Effects (`abilities/effects.py`)

```
4 Effect Types
├── Instant — apply once immediately
├── Duration — apply for N seconds, then remove
├── Infinite — apply until explicitly removed
└── Periodic — tick every N seconds

EffectContainer
├── Multiple modifiers per effect
├── Tag-based filtering (blocked_by_tags, application_tags)
└── Factory functions: instant_damage, damage_over_time, stat_buff, stat_debuff

@buff(duration, stacking, max_stacks, tick_rate) decorator
├── Stacking: none, duration, intensity, independent
└── Time management for duration/periodic effects
```

### Attribute System (`abilities/attributes.py`)

```
6-Step Recalculation Pipeline
├── 1. ADD_BASE — flat base value
├── 2. MULTIPLY_BASE — percentage of base
├── 3. ADD_BONUS — flat bonus
├── 4. MULTIPLY_BONUS — percentage of bonus
├── 5. OVERRIDE — set absolute value
└── 6. CLAMP — enforce min/max

DerivedAttribute
├── Formula: function of other attributes
├── Dependency tracking (dirty cache)
└── Auto-recalculate on dependency change

TrackedDescriptor integration
├── Change reporting to Foundation Tracker
├── UI binding hooks
└── Network replication hooks

create_standard_attributes() — factory for common attributes
```

### Targeting (`abilities/targeting.py`)

```
5 Targeting Modes
├── Self — caster only
├── Actor — raycast/overlap for target entity
├── Point — world position
├── Area — radius query
└── Confirmation — player must confirm (UI reticle)
```

### Gameplay Tags (`abilities/tags.py`)

```
Hierarchical Tag System
├── Structure: ability.offensive.fireball
├── Parent matches children (unless explicitly excluded)
├── Queries: has_tag(tag), matches(tag_query), blocked_by(tags)
└── @gameplay_tag(hierarchy) decorator
    ├── Tags inherited from decorator stacks
    └── Serializable
```

### Cooldown / Cost Management

```
Cooldown
├── Per-ability tracking
├── Persistent across activations
└── CooldownSystem (tick-based reduction)

Costs (multiple resource types)
├── Mana, stamina, health, resources
├── Validation in ACTIVATE phase
└── Payment in COMMIT phase
```

## Key Files

| File | Purpose |
|------|---------|
| `abilities/ability.py` | Ability definition, 4-phase activation, AbilitySystem |
| `abilities/effects.py` | 4 effect types, stacking, EffectContainer |
| `abilities/attributes.py` | 6-step recalculation, DerivedAttribute, factory |
| `abilities/targeting.py` | 5 targeting modes |
| `abilities/tags.py` | Hierarchical tags, @gameplay_tag decorator |

## Dependencies

- Phase 1 entity framework (Actor, ComponentStore)
- Foundation: TrackedDescriptor, EventLog
- Phase 2 (Input) — for input-triggered abilities
