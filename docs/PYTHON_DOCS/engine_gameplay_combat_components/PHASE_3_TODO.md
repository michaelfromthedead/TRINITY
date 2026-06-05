# PHASE 3 TODO: Integration

## Overview

Validate integration between combat and components modules, and with the broader Trinity engine.

---

## T-3.1: Entity ID Consistency

### Tasks

- [ ] **T-3.1.1**: Verify int entity IDs work across all combat systems
  - scoring.py accepts int IDs
  - hitbox.py accepts int IDs
  - health.py accepts int IDs
  - spawn_manager.py accepts int IDs
  - death.py accepts int IDs
  - teams.py accepts int IDs
  - **Acceptance**: All combat systems interoperate with int IDs

- [ ] **T-3.1.2**: Verify int entity IDs work across all components
  - stats.py uses int IDs
  - movement.py uses int IDs
  - health.py uses int IDs
  - team.py uses int IDs
  - transform.py uses int IDs
  - **Acceptance**: All components interoperate with int IDs

- [ ] **T-3.1.3**: Verify cross-module ID passing
  - Combat system receives ID from component
  - Component receives ID from combat system
  - **Acceptance**: IDs pass correctly between modules

### Acceptance Criteria
- No type errors when passing IDs between modules
- ID lookups work in both modules
- Serialization preserves ID type

---

## T-3.2: Event Callback Consistency

### Tasks

- [ ] **T-3.2.1**: Verify callback pattern in combat systems
  - All systems use `_on_*_callbacks` lists
  - All systems wrap callbacks in try/except
  - All systems support register/unregister
  - **Acceptance**: Callback pattern consistent

- [ ] **T-3.2.2**: Verify callback pattern in components
  - All components use `_on_*_callbacks` lists
  - All components wrap callbacks in try/except
  - All components support register/unregister
  - **Acceptance**: Callback pattern consistent

- [ ] **T-3.2.3**: Test cross-module callbacks
  - Combat system callback triggers component update
  - Component callback triggers combat system update
  - **Acceptance**: Callbacks work across modules

- [ ] **T-3.2.4**: Test callback exception isolation
  - Bad callback doesn't break system
  - Other callbacks still execute
  - Error logged (not swallowed silently)
  - **Acceptance**: Fault isolation works

### Acceptance Criteria
- Uniform callback pattern across codebase
- No callback cascade failures
- Callbacks are garbage-collection safe

---

## T-3.3: Serialization Consistency

### Tasks

- [ ] **T-3.3.1**: Test combat system serialization
  - scoring.py to_dict/from_dict
  - hitbox.py to_dict/from_dict
  - health.py (combat) to_dict/from_dict
  - spawn_manager.py to_dict/from_dict
  - death.py to_dict/from_dict
  - teams.py to_dict/from_dict
  - deathmatch.py to_dict/from_dict
  - **Acceptance**: All combat systems serialize correctly

- [ ] **T-3.3.2**: Test component serialization
  - stats.py to_dict/from_dict
  - movement.py to_dict/from_dict
  - health.py (components) to_dict/from_dict
  - team.py to_dict/from_dict
  - transform.py to_dict/from_dict
  - **Acceptance**: All components serialize correctly

- [ ] **T-3.3.3**: Test round-trip consistency
  - `from_dict(to_dict(x)) == x` for all types
  - No data loss during serialization
  - **Acceptance**: Perfect round-trip

- [ ] **T-3.3.4**: Test version tolerance
  - Unknown keys ignored
  - Missing keys use defaults
  - Old format loads in new code
  - **Acceptance**: Forward/backward compatible

- [ ] **T-3.3.5**: Test JSON compliance
  - All output JSON-serializable
  - No custom types in output
  - **Acceptance**: json.dumps() succeeds on all to_dict() output

### Acceptance Criteria
- All classes have to_dict/from_dict
- Round-trip preserves all data
- JSON-serializable output

---

## T-3.4: Trinity Descriptor Integration

### Tasks

- [ ] **T-3.4.1**: Verify TrackedDescriptor usage
  - MovementComponent uses TrackedDescriptor
  - HealthComponent uses TrackedDescriptor (if applicable)
  - StatsComponent uses TrackedDescriptor (if applicable)
  - **Acceptance**: Dirty tracking works

- [ ] **T-3.4.2**: Test dirty flag propagation
  - Property write sets dirty flag
  - Multiple writes don't stack
  - Clear dirty resets flag
  - **Acceptance**: Dirty tracking correct

- [ ] **T-3.4.3**: Test descriptor with serialization
  - Deserialization doesn't trigger dirty
  - Or: dirty cleared after load
  - **Acceptance**: Load doesn't cause spurious syncs

### Acceptance Criteria
- Components integrate with Trinity sync system
- Dirty flags accurate
- No performance regression from descriptors

---

## T-3.5: Dual Health System Integration

### Tasks

- [ ] **T-3.5.1**: Test combat health in isolation
  - Shield stacking works
  - Invulnerability sources work
  - Combat state tracking works
  - **Acceptance**: Combat health standalone

- [ ] **T-3.5.2**: Test components health in isolation
  - Damage types work
  - Resistances work
  - Armor works
  - **Acceptance**: Components health standalone

- [ ] **T-3.5.3**: Test both health systems on same entity
  - Both can be attached
  - Both receive damage events
  - Clear responsibility split
  - **Acceptance**: Systems coexist

- [ ] **T-3.5.4**: Document usage guidelines
  - When to use combat health
  - When to use components health
  - When to use both
  - **Acceptance**: Clear guidance

### Acceptance Criteria
- Both health systems work independently
- Both can coexist on entity
- Clear documentation on when to use each

---

## T-3.6: Dual Team System Integration

### Tasks

- [ ] **T-3.6.1**: Test team component standalone
  - Primary team assignment
  - Secondary teams
  - IFF tags
  - **Acceptance**: Component works standalone

- [ ] **T-3.6.2**: Test team system standalone
  - Team definitions
  - Relationship matrix
  - IFF queries
  - **Acceptance**: System works standalone

- [ ] **T-3.6.3**: Test component + system integration
  - Component reads team_id
  - System queries entity via component
  - Relationship lookups use system
  - **Acceptance**: Integration works

- [ ] **T-3.6.4**: Test relationship symmetry
  - A enemy of B implies B enemy of A
  - Or: asymmetric relationships supported
  - **Acceptance**: Relationship model documented

### Acceptance Criteria
- Component-system integration verified
- Clear data ownership (component = entity data, system = rules)
- Serialization works for both

---

## T-3.7: Full Combat Flow Integration

### Tasks

- [ ] **T-3.7.1**: Test hitbox -> health -> death flow
  - Hitbox detects collision
  - Health takes damage
  - Death triggers at 0 health
  - **Acceptance**: Flow completes

- [ ] **T-3.7.2**: Test death -> scoring flow
  - Death event received by scoring
  - Kill attributed correctly
  - Assist attributed correctly
  - **Acceptance**: Attribution correct

- [ ] **T-3.7.3**: Test death -> respawn flow
  - Death adds to respawn queue
  - Respawn delay respected
  - Spawn point selected
  - **Acceptance**: Respawn completes

- [ ] **T-3.7.4**: Test IFF integration in combat
  - Hitbox checks IFF before damage
  - Friendly fire multiplier applied
  - Team filtering in spawn
  - **Acceptance**: IFF integrated

- [ ] **T-3.7.5**: Test full deathmatch round
  - Multiple players
  - Kills, deaths, assists
  - Leaderboard accurate
  - Win condition triggers
  - **Acceptance**: Full round works

### Acceptance Criteria
- End-to-end combat works
- All systems communicate correctly
- No dropped events

---

## T-3.8: Movement -> Transform Integration

### Tasks

- [ ] **T-3.8.1**: Test position updates
  - Movement velocity updates position
  - Transform position reflects movement
  - **Acceptance**: Position synced

- [ ] **T-3.8.2**: Test coordinate spaces
  - Movement input in world space
  - Transform stores in local space (if parented)
  - Conversions correct
  - **Acceptance**: Spaces handled

- [ ] **T-3.8.3**: Test hierarchy movement
  - Child follows parent
  - Child independent movement works
  - World position correct
  - **Acceptance**: Hierarchy works

### Acceptance Criteria
- Movement updates transform correctly
- Coordinate space conversions accurate
- Hierarchy respected

---

## T-3.9: Stats -> Health Integration

### Tasks

- [ ] **T-3.9.1**: Test max health from stats
  - Base health stat exists
  - Health component reads max from stats
  - Modifiers affect max health
  - **Acceptance**: Max health from stats

- [ ] **T-3.9.2**: Test health scaling
  - Current health scales with max
  - Or: current capped at max
  - Design decision documented
  - **Acceptance**: Scaling behavior correct

- [ ] **T-3.9.3**: Test modifier on health
  - Add health modifier
  - Max health updates
  - Current health behavior defined
  - **Acceptance**: Modifiers work

### Acceptance Criteria
- Stats drive health values
- Modifier changes reflected in health
- Edge cases handled (current > max)

---

## Test Infrastructure

### Integration Test Fixtures

- Full entity factory (all components attached)
- Combat scenario builder (teams, players, positions)
- Event recorder for verification
- Time controller for deterministic tests

### Performance Benchmarks

| Scenario | Target |
|----------|--------|
| 100 entities, full combat frame | < 1ms |
| 1000 entities, transform hierarchy | < 5ms |
| Serialization of 100 entities | < 10ms |
| Stat recomputation cascade | < 0.1ms |

### Coverage Targets

| Integration Area | Target Coverage |
|-----------------|----------------|
| Combat flow | 80% |
| Component flow | 80% |
| Cross-module | 70% |
| Serialization | 90% |
