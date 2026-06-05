# PHASE 1 ARCHITECTURE: Combat Systems

## Scope

This phase covers the seven combat system files in `engine/gameplay/combat/`:
- scoring.py
- hitbox.py
- health.py
- spawn_manager.py
- death.py
- teams.py
- modes/deathmatch.py

---

## Architecture Decisions

### ADR-001: Scoring System Structure

**Context**: Need to track kills, deaths, assists, streaks, and multi-kills with attribution.

**Decision**: Use a central `ScoringSystem` class that manages `PlayerStats` objects.

**Key Classes**:
- `PlayerStats` - 28 tracked statistics per player
- `TeamStats` - Aggregated team-level stats
- `ScoringSystem` - Main coordinator with leaderboard generation
- `ScoreEvent` - Immutable event record
- `LeaderboardEntry` - Display-ready sorted data

**Rationale**: Centralized tracking enables consistent attribution and prevents double-counting.

---

### ADR-002: Multi-Kill Detection Algorithm

**Context**: Detect double, triple, quad, penta kills within a time window.

**Decision**: Track last kill time per player, increment counter if within window, reset on timeout.

**Algorithm** (lines 650-676):
```python
if time_since_last <= self._config.multi_kill_window:
    killer_stats._multi_kill_count += 1
    multi = killer_stats._multi_kill_count
    if multi == 2:
        killer_stats.double_kills += 1
    elif multi == 3:
        killer_stats.triple_kills += 1
    # ... up to penta
```

**Rationale**: Simple state machine, O(1) per kill, no complex event correlation.

---

### ADR-003: Assist Attribution

**Context**: Players who damaged a victim should receive assists.

**Decision**: Track damage contributors with timestamps, attribute assist if damage exceeds threshold within time window.

**Algorithm** (lines 774-797):
- Maintain damage log: `{victim_id: [(attacker_id, damage, time), ...]}`
- On death, filter by `time_since_damage <= assist_window`
- Sum damage per attacker
- Grant assist if `total_damage >= assist_threshold`

**Rationale**: Prevents kill-stealing complaints, rewards teamwork.

---

### ADR-004: Hitbox/Hurtbox Separation

**Context**: Need collision detection between attacks (hitboxes) and vulnerable areas (hurtboxes).

**Decision**: Separate data structures with different responsibilities.

**Hitbox Properties**:
- Position, size (BoundingBox)
- Damage, priority, zone (head/body/limb)
- Multi-hit prevention via victim set
- Active/inactive state

**Hurtbox Properties**:
- Position, size (BoundingBox)
- Armor value, intangibility flag
- Counter-state for counter-hit detection
- Zone multiplier

**Rationale**: Allows independent tuning of offensive and defensive properties.

---

### ADR-005: AABB Collision Test

**Context**: Need fast intersection test for 3D boxes.

**Decision**: Standard AABB separating axis test.

**Algorithm** (lines 146-155):
```python
def intersects(self, other: "BoundingBox") -> bool:
    return (
        self.min_point.x <= other.max_point.x and
        self.max_point.x >= other.min_point.x and
        self.min_point.y <= other.max_point.y and
        self.max_point.y >= other.min_point.y and
        self.min_point.z <= other.max_point.z and
        self.max_point.z >= other.min_point.z
    )
```

**Rationale**: O(1), no square roots, sufficient for gameplay (not physics).

---

### ADR-006: Collision Priority Resolution

**Context**: When multiple hitboxes hit simultaneously, which takes precedence?

**Decision**: Higher priority wins, ties broken by damage amount.

**Algorithm** (lines 783-792):
- Gather all overlapping hitbox-hurtbox pairs
- Sort by priority descending, then damage descending
- Apply first hit, mark victim in hitbox's hit-set
- Skip subsequent hits from same hitbox

**Rationale**: Prevents multi-hit frame exploits, predictable combat.

---

### ADR-007: Shield Absorption System

**Context**: Shields should absorb damage before health.

**Decision**: Priority-ordered shield stack with typed absorption.

**Algorithm** (health.py lines 699-729):
```python
def _apply_shields(self, damage: float, damage_type: Optional[Any] = None) -> Tuple[float, float]:
    self._cleanup_expired_shields()
    remaining = damage
    total_absorbed = 0.0
    for shield in self._shields[:]:  # Copy to allow removal
        if remaining <= 0:
            break
        remaining, absorbed = shield.absorb(remaining, damage_type)
        total_absorbed += absorbed
        if shield.amount <= 0:
            self._shields.remove(shield)
    return (remaining, total_absorbed)
```

**Rationale**: Multiple shields can stack (e.g., personal + team buff).

---

### ADR-008: Spawn Selection Strategies

**Context**: Different game modes need different spawn logic.

**Decision**: Five configurable strategies:
1. **Random** - Uniform random from valid points
2. **Sequential** - Round-robin through points
3. **Distance-based** - Maximize distance from enemies
4. **Safe** - Multi-factor scoring (distance, line-of-sight, recent deaths)
5. **Priority-weighted** - Weighted random by point priority

**Implementation** (spawn_manager.py):
- `SpawnRule` enum selects strategy
- Each strategy returns sorted candidates
- Fallback chain if primary fails

**Rationale**: One mode wants action (close spawns), another wants fairness (far spawns).

---

### ADR-009: Death State Machine

**Context**: Death involves animation, cleanup, respawn - not instantaneous.

**Decision**: Three-state machine: DYING -> DEAD -> RESPAWNING

**States**:
- `DYING` - Death animation playing, entity still exists
- `DEAD` - Animation complete, pending cleanup
- `RESPAWNING` - In respawn queue, waiting for spawn point

**Transitions** (death.py lines 300-370):
- `trigger_death()` -> DYING
- `transition_to_dead()` -> DEAD (adds to cleanup queue)
- `request_respawn()` -> RESPAWNING (adds to respawn queue)
- `spawn_complete()` -> Removed from system

**Rationale**: Allows death cams, loot drops, delayed respawn waves.

---

### ADR-010: Team Relationship Matrix

**Context**: Need to query if two teams are allies, enemies, or neutral.

**Decision**: Bidirectional relationship map with default fallback.

**Implementation** (teams.py):
```python
# Stored as: relationships[(team_a, team_b)] = RelationType
# Query both directions for bidirectional relationships
```

**Relationship Types**:
- `ALLY` - No damage, shared spawns
- `ENEMY` - Full damage
- `NEUTRAL` - Configurable damage, no assists

**Rationale**: Supports complex scenarios (temporary alliances, civilians).

---

### ADR-011: IFF (Identify Friend/Foe) Query

**Context**: Combat code needs fast friend/foe determination.

**Decision**: Single query method returning complete relationship info.

**Return Type** (teams.py lines 510-553):
```python
@dataclass
class IFFResult:
    is_same_team: bool
    relationship: RelationType
    friendly_fire_multiplier: float
    can_damage: bool
    grants_assist: bool
```

**Rationale**: Single call, no repeated lookups, all info needed for damage calc.

---

### ADR-012: Game Mode Base Class

**Context**: Multiple game modes share common infrastructure.

**Decision**: Abstract `GameMode` base class with hooks.

**Hooks**:
- `on_player_killed()` - Score attribution
- `check_win_condition()` - Game end check
- `on_round_start()`/`on_round_end()` - Round lifecycle
- `get_leaderboard()` - Mode-specific sorting

**Deathmatch Implementation** (modes/deathmatch.py):
- Score limit win condition
- Time limit win condition
- Killstreak bonus progression
- Multi-kill bonus tiers

**Rationale**: New modes inherit infrastructure, override specific behavior.

---

## Integration Points

### Combat -> Components

- `HealthComponent` from components is distinct from combat's health system
- Combat's `health.py` focuses on shields/invulnerability in multiplayer context
- Components' `health.py` focuses on damage types/resistances in RPG context

### Combat Internal

- `scoring.py` receives events from `death.py`
- `hitbox.py` generates damage events for `health.py`
- `spawn_manager.py` receives requests from `death.py`
- `teams.py` provides IFF for `hitbox.py` and `scoring.py`

---

## File Dependencies

```
deathmatch.py
    └── game_mode.py (base class)
            └── scoring.py (leaderboards)

hitbox.py
    └── teams.py (IFF checks)

health.py (combat)
    └── (standalone)

death.py
    └── spawn_manager.py (respawn)
    └── scoring.py (death attribution)

spawn_manager.py
    └── teams.py (team-based spawns)

teams.py
    └── (standalone)

scoring.py
    └── (standalone)
```
