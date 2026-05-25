# PHASE 1 TODO: Combat Systems

## Overview

Validate and test the seven combat system files in `engine/gameplay/combat/`.

---

## T-1.1: Scoring System Validation

**File**: `engine/gameplay/combat/scoring.py`

### Tasks

- [ ] **T-1.1.1**: Test multi-kill detection (lines 650-676)
  - Double kill within window
  - Triple kill within window
  - Counter reset after window expires
  - **Acceptance**: All multi-kill tiers correctly attributed

- [ ] **T-1.1.2**: Test killstreak detection (lines 638-647)
  - Streak increment on kill
  - Streak reset on death
  - Configurable thresholds
  - **Acceptance**: Streaks match expected values

- [ ] **T-1.1.3**: Test assist attribution (lines 774-797)
  - Damage within time window grants assist
  - Damage below threshold does not grant assist
  - Multiple assisters on single kill
  - **Acceptance**: Assists attributed correctly

- [ ] **T-1.1.4**: Test leaderboard sorting (lines 940-998)
  - Sort by kills (primary)
  - Sort by deaths (secondary, ascending)
  - Sort by score (tertiary)
  - **Acceptance**: Leaderboard order matches expectations

### Acceptance Criteria
- All PlayerStats fields update correctly
- TeamStats aggregate individual stats
- ScoreEvents emitted for all actions
- Serialization round-trip preserves data

---

## T-1.2: Hitbox System Validation

**File**: `engine/gameplay/combat/hitbox.py`

### Tasks

- [ ] **T-1.2.1**: Test AABB intersection (lines 146-155)
  - Overlapping boxes return true
  - Non-overlapping boxes return false
  - Edge-touching boxes (boundary case)
  - **Acceptance**: Intersection test correct for all cases

- [ ] **T-1.2.2**: Test collision priority (lines 783-792)
  - Higher priority hitbox wins
  - Equal priority: higher damage wins
  - Multi-hit prevention via victim set
  - **Acceptance**: Only highest priority hit applies

- [ ] **T-1.2.3**: Test counter-hit detection (lines 754-757)
  - Hurtbox in counter-state takes bonus damage
  - Damage multiplier applied correctly
  - **Acceptance**: Counter-hit damage matches expected

- [ ] **T-1.2.4**: Test super armor absorption (lines 368-378)
  - Armor absorbs damage up to threshold
  - Armor-piercing damage bypasses
  - Armor regeneration
  - **Acceptance**: Armor behavior matches design

- [ ] **T-1.2.5**: Test parry/block window (lines 729-747)
  - Perfect parry within frame window
  - Block reduces damage
  - Parry window expiration
  - **Acceptance**: Defensive mechanics work correctly

### Acceptance Criteria
- All hitbox-hurtbox collisions detected
- Priority system resolves conflicts
- Zone multipliers applied (head > body > limb)
- Events emitted for all hits

---

## T-1.3: Health System Validation

**File**: `engine/gameplay/combat/health.py`

### Tasks

- [ ] **T-1.3.1**: Test shield absorption (lines 699-729)
  - Shield absorbs before health
  - Multiple shields stack by priority
  - Expired shields cleaned up
  - **Acceptance**: Shield logic correct

- [ ] **T-1.3.2**: Test invulnerability (lines 569-630)
  - Multiple invulnerability sources tracked
  - Duration-based expiration
  - Damage blocked during invulnerability
  - **Acceptance**: Invulnerability prevents damage

- [ ] **T-1.3.3**: Test health regeneration (lines 530-555)
  - Regen starts after delay
  - Damage resets regen delay
  - Regen rate configurable
  - **Acceptance**: Regen behavior matches config

- [ ] **T-1.3.4**: Test combat state tracking (lines 311-319)
  - Enter combat on damage
  - Exit combat after timeout
  - Combat state affects regen
  - **Acceptance**: Combat state transitions correct

### Acceptance Criteria
- Shield absorption priority correct
- Invulnerability blocks all damage types
- Regeneration respects combat state
- All events emitted (on_damage, on_heal, on_death)

---

## T-1.4: Spawn Manager Validation

**File**: `engine/gameplay/combat/spawn_manager.py`

### Tasks

- [ ] **T-1.4.1**: Test random spawn selection (lines 455-477)
  - Uniform distribution over valid points
  - Team filtering works
  - **Acceptance**: All valid points can be selected

- [ ] **T-1.4.2**: Test sequential spawn selection (lines 479-498)
  - Round-robin through points
  - Index wraps correctly
  - **Acceptance**: Points selected in order

- [ ] **T-1.4.3**: Test distance-based selection (lines 523-575)
  - Spawn furthest from enemies
  - Minimum distance threshold respected
  - **Acceptance**: Selected point maximizes distance

- [ ] **T-1.4.4**: Test safe spawn scoring (lines 577-615)
  - Multi-factor scoring (distance, LOS, recent deaths)
  - Weights configurable
  - **Acceptance**: Safest point selected

- [ ] **T-1.4.5**: Test respawn queue (lines 693-730)
  - Queue respects delay timing
  - Priority ordering works
  - **Acceptance**: Respawns occur at correct times

### Acceptance Criteria
- All five selection strategies work
- Fallback chain if primary fails
- Cooldown prevents immediate reuse
- Team-based spawn filtering correct

---

## T-1.5: Death System Validation

**File**: `engine/gameplay/combat/death.py`

### Tasks

- [ ] **T-1.5.1**: Test state machine transitions (lines 300-370)
  - ALIVE -> DYING on trigger_death
  - DYING -> DEAD on transition_to_dead
  - DEAD -> RESPAWNING on request_respawn
  - Invalid transitions rejected
  - **Acceptance**: State machine correct

- [ ] **T-1.5.2**: Test death info tracking (lines 88-113)
  - Killer ID recorded
  - Weapon/method recorded
  - Headshot flag set
  - Overkill damage tracked
  - **Acceptance**: Death info complete

- [ ] **T-1.5.3**: Test cleanup handler registration (lines 596-618)
  - Handlers called in order
  - Handler exceptions don't cascade
  - **Acceptance**: All handlers invoked

- [ ] **T-1.5.4**: Test respawn timing (lines 376-439)
  - Respawn delay respected
  - Wave-based respawn works
  - Instant respawn option
  - **Acceptance**: Timing matches config

### Acceptance Criteria
- State transitions emit events
- Death info serializes correctly
- Cleanup handlers called on DEAD state
- Respawn queue integrates with spawn manager

---

## T-1.6: Team System Validation

**File**: `engine/gameplay/combat/teams.py`

### Tasks

- [ ] **T-1.6.1**: Test IFF checks (lines 510-553)
  - Same team returns FRIEND
  - Enemy team returns FOE
  - Neutral returns NEUTRAL
  - **Acceptance**: IFF results correct

- [ ] **T-1.6.2**: Test friendly fire multiplier (lines 528-532)
  - FF multiplier applied to same-team damage
  - Default FF from config used if no team setting
  - **Acceptance**: FF damage scaled correctly

- [ ] **T-1.6.3**: Test bidirectional relationships (lines 434-480)
  - Set A->B also sets B->A
  - Query from either direction works
  - **Acceptance**: Relationships symmetric

- [ ] **T-1.6.4**: Test auto-balance assignment (lines 681-722)
  - New players assigned to smallest team
  - Max team size respected
  - **Acceptance**: Teams stay balanced

### Acceptance Criteria
- IFFResult contains all needed info
- Team membership changes emit events
- Relationship matrix serializes correctly
- Auto-balance maintains fairness

---

## T-1.7: Deathmatch Mode Validation

**File**: `engine/gameplay/combat/modes/deathmatch.py`

### Tasks

- [ ] **T-1.7.1**: Test win condition checking (lines 249-255)
  - Score limit triggers win
  - Time limit triggers win
  - Correct winner determined
  - **Acceptance**: Game ends correctly

- [ ] **T-1.7.2**: Test killstreak bonuses (lines 127-134)
  - Bonus points at configured thresholds
  - Streak reset on death
  - **Acceptance**: Bonus points awarded

- [ ] **T-1.7.3**: Test multi-kill bonuses (lines 136-142)
  - Bonus tiers match config
  - Counter resets on window expiry
  - **Acceptance**: Multi-kill bonuses correct

- [ ] **T-1.7.4**: Test kill attribution (lines 161-247)
  - Killer gets full points
  - Assists get partial points
  - Suicide/fall damage handled
  - **Acceptance**: Points attributed correctly

### Acceptance Criteria
- GameMode base class integration works
- Leaderboard sorted by mode-specific criteria
- Round lifecycle hooks called
- Serialization preserves mode state

---

## Test Infrastructure

### Required Fixtures

- Mock entity system for ID generation
- Mock event bus for emission verification
- Time provider for deterministic timing
- Damage info factory for test cases

### Coverage Targets

| File | Target Coverage |
|------|----------------|
| scoring.py | 85% |
| hitbox.py | 90% |
| health.py | 85% |
| spawn_manager.py | 80% |
| death.py | 85% |
| teams.py | 80% |
| deathmatch.py | 85% |
