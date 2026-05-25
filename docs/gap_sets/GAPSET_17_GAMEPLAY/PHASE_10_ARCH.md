# Phase 10: Combat & Game Modes — Architecture

## Overview

Complete combat system: health/damage/death pipeline, damage modifiers, scoring, and game mode framework with match lifecycle.

## Component Breakdown

### Health System (`combat/health.py`)

```
HealthComponent
├── current (RangeDescriptor: 0-max)
├── max_hp
├── regen_rate (per-second)
├── is_invulnerable flag
├── shields: ShieldInfo list with priority + damage type filtering
├── InvulnerabilityInfo (configurable duration/pierce)
├── TrackedDescriptor on current for change tracking
└── HealthPool manager

Operations
├── damage(amount, type, source) — returns actual damage dealt
├── heal(amount)
├── add_shield / remove_shield
├── set_invulnerable(duration, can_pierce)
└── Events: on_damage_taken, on_healed, on_shield_destroyed, on_invulnerability_end
```

### Damage System (`combat/damage.py`)

```
DamageSystem
├── Reads PendingDamage component each tick
├── Armor formula: damage_reduction = armor / (armor + constant)
├── 5+ damage types: physical, fire, ice, lightning, poison, holy, dark
├── Hitbox multipliers: head (2.0x), chest (1.0x), limbs (0.7x), back (1.5x)
├── ResistanceProfile (flat + percentage resistances)
├── DamageModifier chain (configurable modifiers)
└── Calculates DPS and EHP

Damage Formula Pipeline
├── Base damage
├── × hitbox multiplier
├── × (1 - resistance) per damage type
├── × vulnerability multiplier
├── - shield absorption
└── → health reduction
```

### Death System (`combat/death.py`)

```
DeathSystem — State Machine
├── DYING → DEAD → RESPAWNING
├── Configurable dying duration
├── DeathInfo: death_state, timestamp, death_cause, was_headshot, was_critical, overkill_damage
├── RespawnRequest queue
│   ├── Configurable delay
│   ├── Configurable respawn health/invulnerability
│   └── Respawn provider binding
├── Cleanup handler registration
├── Event emission: on_death, on_respawn, on_state_changed
└── DeathSubject / CleanupHandler / RespawnProvider protocols
```

### Team / Faction

```
TeamComponent (shared with T-GP-3.21)
├── team_id, faction string
├── IFF: is_enemy, is_ally, is_neutral
├── Configurable friendly fire
└── Runtime team change (with restrictions)
```

### Scoring System (`combat/scoring.py`)

```
ScoreEventType (16 types)
├── KILL, DEATH, ASSIST, HEADSHOT
├── FIRST_BLOOD, REVENGE, KILLSTREAK, KILLSTREAK_ENDED
├── MULTI_KILL
├── OBJECTIVE_CAPTURE, OBJECTIVE_DEFEND, OBJECTIVE_PROGRESS
├── BONUS, PENALTY, TEAM_BONUS
└── ROUND_WIN, MATCH_WIN

PlayerStats
├── 30+ fields: score, kills, deaths, assists, damage_dealt/taken
├── kd_ratio, kda_ratio, total_multi_kills
├── damage tracking with ASSIST_TIME_WINDOW
└── record_damage_dealt / get_assist_damage / clear_damage_tracking

ScoringSystem
├── record_kill: full chain
│   ├── base points
│   ├── first blood bonus
│   ├── headshot bonus
│   ├── revenge bonus
│   ├── killstreak bonus (KILLSTREAK_THRESHOLDS)
│   ├── multi-kill detection (MULTI_KILL_WINDOW, MULTI_KILL_NAMES)
│   └── assist calculation
├── record_death, record_assist
├── record_objective_capture, record_objective_defend
├── get_leaderboard (9 sort keys)
├── get_team_leaderboard
└── Event history (configurable max size)

TeamStats: score/kills/deaths/assists/objectives/rounds_won/members

Killstreak Thresholds & Multi-Kill Names
├── 3=KILLING_SPREE, 5=RAMPAGE, 7=DOMINATING, 10=UNSTOPPABLE, etc.
├── 2=Double Kill, 3=Triple Kill, 4=Quadra Kill, 5=Penta Kill
```

### Game Mode Framework (`combat/game_mode.py`)

```
GameMode — Match Lifecycle
├── Lobby → Countdown → Playing → Match End → Results
├── Spawn logic (spawn points, respawn timer)
├── Rule hooks: can_player_respawn, is_match_over, get_winner
├── WinConditionType (7 types): KILLS, SCORE, TIME, FLAG_CAPTURE, HILL_CONTROL, LAST_ALIVE, OBJECTIVE
├── ScoringEventType (13 types)
├── Round management, overtime
├── Configurable time/score limits
└── Base framework for specific modes

Specific Modes
├── Deathmatch / Team Deathmatch (via WinConditionType configuration)
├── CTF — NOT VERIFIED as separate subclass
├── KOTH — NOT VERIFIED as separate subclass
└── BR — NOT VERIFIED as separate subclass
```

### Execution Order

```
Combat Systems Registration
├── DamageSystem (order 7)
├── DeathSystem (order 8)
├── CleanupSystem (order 9)
└── Relative to other gameplay systems (Input order 1, AI order 2, Abilities order 3)
```

## Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `combat/health.py` | — | HealthComponent, shields, invulnerability |
| `combat/damage.py` | — | DamageSystem, armor formula, modifiers |
| `combat/death.py` | 759 | DeathSystem state machine, respawn |
| `combat/scoring.py` | 1188 | ScoringSystem, PlayerStats, leaderboards |
| `combat/game_mode.py` | 655 | GameMode, match lifecycle, win conditions |

## Dependencies

- Phase 1 entity framework (Actor, ComponentStore, lifecycle)
- Phase 2 (Input) — for player controls during gameplay
- Phase 7 (Abilities) — for combat abilities
- Phase 8 (Inventory) — for weapon/armor equipment in combat
