# PHASE 2 TODO: Component Systems

## Overview

Validate and test the five ECS-style component files in `engine/gameplay/components/`.

---

## T-2.1: Stats Component Validation

**File**: `engine/gameplay/components/stats.py`

### Tasks

- [ ] **T-2.1.1**: Test modifier stacking order (lines 114-156)
  - OVERRIDE takes precedence
  - FLAT adds to base
  - PERCENT_BASE multiplies base only
  - MULTIPLY stacks multiplicatively
  - PERCENT_TOTAL multiplies final
  - **Acceptance**: Stacking order matches spec

- [ ] **T-2.1.2**: Test cache invalidation (lines 175-182)
  - Cache invalidated on modifier add
  - Cache invalidated on modifier remove
  - Cache invalidated on base change
  - **Acceptance**: Computed value always correct

- [ ] **T-2.1.3**: Test timed modifier expiration (lines 574-606)
  - Modifier expires at end_time
  - Expired modifier removed on next access
  - Cache invalidated after expiration
  - **Acceptance**: Timed modifiers expire correctly

- [ ] **T-2.1.4**: Test derived stat computation (lines 352-368)
  - Derived stat recomputes when dependency changes
  - Formula receives correct input values
  - Circular dependencies handled/prevented
  - **Acceptance**: Derived stats update correctly

- [ ] **T-2.1.5**: Test modifier priority (OVERRIDE case)
  - Multiple overrides: highest priority wins
  - Override blocks all other modifier types
  - **Acceptance**: Priority system correct

### Acceptance Criteria
- All modifier types compute correctly
- Cache always reflects current state
- Serialization round-trip preserves modifiers
- Events emitted on value change

---

## T-2.2: Movement Component Validation

**File**: `engine/gameplay/components/movement.py`

### Tasks

- [ ] **T-2.2.1**: Test movement mode switching (lines 124-135)
  - WALKING mode settings applied
  - RUNNING mode settings applied
  - SWIMMING mode settings applied
  - FLYING mode settings applied
  - **Acceptance**: Per-mode settings correct

- [ ] **T-2.2.2**: Test coyote time (lines 386-408)
  - Jump allowed briefly after leaving ground
  - Window configurable
  - Timer resets on landing
  - **Acceptance**: Coyote time works within window

- [ ] **T-2.2.3**: Test jump buffering (lines 386-408)
  - Jump request stored
  - Jump executes when possible
  - Buffer clears after window
  - **Acceptance**: Buffered jumps execute

- [ ] **T-2.2.4**: Test velocity acceleration/deceleration (lines 510-568)
  - Acceleration applies per frame
  - Deceleration stops movement
  - Max speed capped
  - **Acceptance**: Velocity curves correct

- [ ] **T-2.2.5**: Test air control factor (line 535)
  - Grounded: full control
  - Airborne: reduced by air_control multiplier
  - **Acceptance**: Air control scales correctly

- [ ] **T-2.2.6**: Test jump count (multi-jump)
  - jumps_remaining decrements on jump
  - Resets on landing
  - Max jumps configurable
  - **Acceptance**: Multi-jump system works

### Acceptance Criteria
- Mode transitions emit events
- Input handling correct for all modes
- Serialization preserves velocity state
- Integration with physics system verified

---

## T-2.3: Health Component Validation (Components)

**File**: `engine/gameplay/components/health.py`

### Tasks

- [ ] **T-2.3.1**: Test damage type resistance (lines 227-303)
  - Each damage type has configurable resistance
  - Resistance caps at MAX_RESISTANCE_CAP
  - TRUE damage ignores resistance
  - **Acceptance**: Resistance reduces damage correctly

- [ ] **T-2.3.2**: Test armor calculation (lines 227-303)
  - Armor subtracts from damage after resistance
  - TRUE damage ignores armor
  - Minimum damage of 0 (no negative)
  - **Acceptance**: Armor reduces damage correctly

- [ ] **T-2.3.3**: Test shield absorption (lines 279-283)
  - Shield absorbs before health
  - Shield depletes correctly
  - Multiple shields not implemented (differs from combat/health.py)
  - **Acceptance**: Shield mechanic works

- [ ] **T-2.3.4**: Test damage/heal history (lines 551-570)
  - Events stored in ring buffer
  - Buffer size respected
  - Timestamp recorded
  - Source ID recorded
  - **Acceptance**: History queryable

- [ ] **T-2.3.5**: Test invulnerability timer (lines 500-525)
  - Damage blocked during invulnerability
  - Timer expires correctly
  - Multiple invulnerability sources (if supported)
  - **Acceptance**: Invulnerability works

### Acceptance Criteria
- All damage types behave correctly
- History enables assist attribution
- Serialization preserves current health
- Events emitted for all damage/heal

---

## T-2.4: Team Component Validation

**File**: `engine/gameplay/components/team.py`

### Tasks

- [ ] **T-2.4.1**: Test IFF tag system (lines 34-45, 425-461)
  - IntFlag bitwise operations work
  - Multiple tags combinable
  - Tag query efficient
  - **Acceptance**: IFF tags behave as IntFlag

- [ ] **T-2.4.2**: Test team registry (lines 102-276)
  - Singleton pattern works
  - Faction hierarchy respected
  - Teams registered correctly
  - **Acceptance**: Registry is single source of truth

- [ ] **T-2.4.3**: Test relationship queries (lines 466-511)
  - Ally relationship returns FRIEND
  - Enemy relationship returns FOE
  - Neutral relationship returns UNKNOWN
  - **Acceptance**: Relationships query correctly

- [ ] **T-2.4.4**: Test secondary team memberships (lines 392-419)
  - Primary team assignment
  - Secondary teams addable
  - is_member_of checks both
  - **Acceptance**: Multi-team membership works

- [ ] **T-2.4.5**: Test faction hierarchy
  - Child factions inherit parent relationships
  - Override at faction level
  - **Acceptance**: Hierarchy respected

### Acceptance Criteria
- IFFResponse bitflags combine correctly
- Registry serializes/deserializes
- Relationship matrix symmetric (if required)
- Team changes emit events

---

## T-2.5: Transform Component Validation

**File**: `engine/gameplay/components/transform.py`

### Tasks

- [ ] **T-2.5.1**: Test parent-child hierarchy (lines 130-220)
  - Parent assignment works
  - Weak references don't prevent GC
  - Cycle detection/prevention
  - **Acceptance**: Hierarchy maintains correctly

- [ ] **T-2.5.2**: Test world matrix caching (lines 278-287)
  - Cache computed on first access
  - Cache reused on subsequent access
  - Cache invalidated on change
  - **Acceptance**: Cache always correct

- [ ] **T-2.5.3**: Test dirty propagation (lines 460-469)
  - Parent change marks children dirty
  - Deep hierarchy propagates
  - Only descendants affected
  - **Acceptance**: Dirty propagation correct

- [ ] **T-2.5.4**: Test look-at rotation (lines 384-435)
  - Rotation faces target
  - Up vector respected
  - Edge case: target at self position
  - **Acceptance**: Look-at computes correct rotation

- [ ] **T-2.5.5**: Test coordinate space transformations (lines 336-383)
  - LOCAL to WORLD correct
  - WORLD to LOCAL correct
  - SELF space correct
  - **Acceptance**: Space conversions accurate

- [ ] **T-2.5.6**: Test TransformSnapshot (immutable)
  - Snapshot captures current state
  - Original transform changes don't affect snapshot
  - Interpolation between snapshots
  - **Acceptance**: Snapshots immutable

### Acceptance Criteria
- Matrix math matches reference implementation
- Hierarchy operations O(depth) not O(total nodes)
- Serialization preserves full hierarchy
- Integration with physics verified

---

## Test Infrastructure

### Required Fixtures

- Mock Trinity descriptor system
- Math library test utilities
- Time provider for timed modifiers
- Entity factory for component testing

### Coverage Targets

| File | Target Coverage |
|------|----------------|
| stats.py | 85% |
| movement.py | 85% |
| health.py | 85% |
| team.py | 80% |
| transform.py | 90% |

---

## Integration Notes

### Stats <-> Health Integration

Both components track entity vitality:
- Stats provides base health value via stat system
- Health component uses stats for max health
- Modifiers from stats affect health pool

### Movement <-> Transform Integration

- Movement updates transform position
- Transform provides world position for movement calculations
- Coordinate space conversions for input handling

### Team (Components) vs Teams (Combat)

- `team.py` (components): ECS component with IFF tags
- `teams.py` (combat): System managing team relationships
- Both can coexist: component for entity data, system for queries
