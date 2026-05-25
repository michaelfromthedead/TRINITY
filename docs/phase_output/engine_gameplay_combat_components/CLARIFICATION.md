# CLARIFICATION: engine_gameplay_combat_components

## Philosophical Framing

The `engine/gameplay/combat` and `engine/gameplay/components` modules represent the gameplay layer of the Trinity engine. These are not engine-level abstractions but domain-specific implementations for game mechanics.

---

## Design Rationale

### Why Two Modules?

**Combat** contains systems for multiplayer competitive gameplay:
- Scoring, hitboxes, death, spawning, teams, game modes
- Designed for networked scenarios where attribution matters
- Focus on fairness, anti-cheat considerations, spectator data

**Components** contains ECS-style building blocks:
- Stats, movement, health, team, transform
- Designed as composable units attached to entities
- Focus on modularity, dirty tracking, serialization

This separation allows:
1. Single-player games to use components without combat systems
2. Different game types to share component infrastructure
3. Clear ownership boundaries for team development

---

## Key Design Decisions

### 1. Event-Driven Architecture

Both modules use callback lists (`_on_damage`, `_on_death`, etc.) rather than direct coupling:

```python
# Registration pattern
component._on_damage_callbacks.append(my_handler)

# Emission pattern  
for callback in self._on_damage_callbacks:
    try:
        callback(event)
    except:
        pass  # Prevent cascade failures
```

**Rationale**: Allows game logic to react to engine events without modifying engine code. The try/except ensures one bad callback doesn't break the system.

### 2. Dirty Tracking via Descriptors

Components use `TrackedDescriptor` from Trinity for change detection:

```python
class MovementComponent:
    velocity = TrackedDescriptor()  # Marks component dirty on write
```

**Rationale**: Enables optimization (skip unchanged components), networking (only replicate dirty data), and undo systems.

### 3. Modifier Stacking Order

Stats use a precise modifier order: OVERRIDE > FLAT > PERCENT_BASE > MULTIPLY > PERCENT_TOTAL

```python
# Order matters:
# 1. Override takes precedence (buffs that set exact value)
# 2. Flat bonuses add to base (+10 strength)
# 3. Percent base multiplies base only (+20% of base)
# 4. Multiply stacks multiplicatively (*1.5)
# 5. Percent total multiplies final result (+10% of total)
```

**Rationale**: This is the standard ARPG/MMORPG formula. Allows predictable stacking of equipment bonuses, buffs, debuffs.

### 4. IFF Tag Bitflags

Team identification uses IntFlag for efficient multi-tag queries:

```python
class IFFResponse(IntFlag):
    FRIEND = 1 << 0
    FOE = 1 << 1
    CIVILIAN = 1 << 3
    PLAYER = 1 << 6
```

**Rationale**: A single entity can be both PLAYER and FRIEND. Bitwise AND allows efficient filtering (e.g., "all hostile players").

### 5. Hierarchical Transform Caching

Transform uses lazy evaluation with dirty propagation:

```python
@property
def world_matrix(self) -> Mat4:
    if self._world_matrix_dirty:
        if self.parent:
            self._world_matrix_cache = self.parent.world_matrix @ self.local_matrix
        else:
            self._world_matrix_cache = self.local_matrix
        self._world_matrix_dirty = False
    return self._world_matrix_cache
```

**Rationale**: Avoids recomputing matrices every frame. Children inherit parent dirty state.

---

## Integration Philosophy

### Entity IDs as Glue

Both modules use entity IDs (string or int) as the cross-referencing mechanism:

- `ScoringSystem.get_player_stats(entity_id)`
- `HealthComponent.take_damage(amount, source_id=killer_id)`
- `TeamSystem.get_team_id(entity_id)`

This avoids direct object references, enabling serialization and network replication.

### Serialization as Contract

All components implement `to_dict()`/`from_dict()`:

```python
def to_dict(self) -> dict:
    return {
        "current_health": self._current,
        "max_health": self._max,
        "shields": [s.to_dict() for s in self._shields],
    }
```

**Rationale**: Enables save/load, network sync, and debugging. The dict format is the "wire protocol" between systems.

---

## Complexity Justification

### High-Complexity Files

**scoring.py (1,187 lines, High)**: Multi-kill detection, killstreak tracking, assist attribution, and leaderboard sorting are inherently complex. Splitting would lose cohesion around "score a kill" operation.

**hitbox.py (1,029 lines, High)**: Collision detection, priority resolution, and multi-hit prevention form a single coherent system. The complexity is in the algorithms, not the structure.

**spawn_manager.py (818 lines, High)**: Five different spawn selection strategies (random, sequential, distance, safe, weighted) with configurable rules justify the line count.

### Medium-Complexity Files

Most files are 600-800 lines with medium complexity. This is appropriate for production game systems that need:
- Full error handling
- Edge case coverage
- Documentation
- Serialization
- Event emission

---

## Future Considerations

1. **Network Sync**: These systems are designed for networked games. Server-authoritative patterns are assumed but not enforced.

2. **Modding Support**: Serialization via `to_dict()` enables data-driven modding. Game designers can tweak values without code changes.

3. **Performance Profiling**: Caching patterns (world matrix, stat computation) assume profiles will reveal optimization needs.

4. **Testing Strategy**: Event-driven design enables mock-based testing. Inject callbacks to verify behavior without full game setup.
