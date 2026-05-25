# Investigation: engine/networking/lag_compensation/

## Classification: REAL (Fully Implemented)

All four files in this module contain complete, production-ready implementations with no stub markers, TODO placeholders, or NotImplementedError patterns.

---

## Module Overview

The lag compensation system provides server-side hit validation by rewinding world state to match what clients were seeing at the time they fired. This is a standard technique for multiplayer FPS games to handle network latency fairly.

**Total Lines:** 1,483 across 4 files

---

## File Analysis

### 1. hitbox_history.py (519 lines) — REAL

**Purpose:** Historical hitbox tracking for efficient hit detection without full world state rewind.

**Key Classes:**

| Class | Lines | Description |
|-------|-------|-------------|
| `Bounds` | 31-127 | AABB with intersection/containment tests, translation, center/extents helpers |
| `HitboxSnapshot` | 129-168 | Entity hitbox state at a timestamp (position, bounds, tick, active flag) |
| `EntityHitboxHistory` | 172-202 | Per-entity deque-based snapshot buffer with timestamp lookup |
| `HitboxHistory` | 205-519 | Central manager for all entity hitboxes with frame caching |

**Core Methods:**

- `record()` — Stores snapshot with frame cache for tick-based lookup
- `get_hitbox_at_time()` / `get_hitbox_at_tick()` — Point lookups
- `get_all_hitboxes_at_time()` / `get_all_hitboxes_at_tick()` — Batch queries
- `get_interpolated_hitbox()` — Linear interpolation between surrounding frames
- `set_tick()` — Advances current tick/timestamp

**Implementation Quality:**
- Uses `deque(maxlen=N)` for automatic history pruning
- Frame cache with bounded size (`HITBOX_CACHE_MULTIPLIER`)
- Position interpolation for sub-frame accuracy
- Bounds interpolation intentionally skipped (bounds assumed stable)

---

### 2. rewind_manager.py (489 lines) — REAL

**Purpose:** Full world state history for server-side rollback during hit validation.

**Key Classes:**

| Class | Lines | Description |
|-------|-------|-------------|
| `EntityState` | 37-63 | Position, rotation, velocity, custom data with deep copy |
| `WorldState` | 67-107 | Dictionary of all entity states at a single tick |
| `HistoryFrame` | 110-133 | Tick + timestamp + WorldState bundle |
| `RewindManager` | 135-489 | History buffer with rewind/restore API |

**Core Methods:**

- `record_frame()` — Deep-copies world state into history deque
- `get_frame_at_time()` / `get_frame_at_tick()` — History lookup
- `get_interpolated_frame()` — Full entity interpolation between frames
- `rewind_to()` — Marks manager as rewound, returns historical state
- `restore_to_current()` — Returns to live state, clears rewind flag
- `can_rewind_to()` — Bounds check with small buffer tolerance

**Implementation Quality:**
- Enforces single-rewind safety (RuntimeError if already rewound)
- Deep copies prevent mutation of historical data
- `_lerp_vector()` for position/velocity interpolation
- Rotation interpolation explicitly skipped (comment: "for simplicity")

**Configuration Integration:**
- Uses `DEFAULT_MAX_REWIND_TIME_MS`, `DEFAULT_TICK_RATE`
- `calculate_max_history_frames()` computes buffer size from config

---

### 3. view_time.py (438 lines) — REAL

**Purpose:** Calculate when a client was perceiving the world based on RTT and interpolation delay.

**Key Classes:**

| Class | Lines | Description |
|-------|-------|-------------|
| `RTTSample` | 35-42 | RTT measurement with timestamp |
| `ViewTimeConfig` | 45-63 | Max lag compensation, interpolation delay, jitter buffer, min samples |
| `ViewTimeCalculator` | 88-345 | Per-client RTT tracking with statistical smoothing |
| `LagCompensationValidator` | 348-438 | Anti-cheat validation of client view time claims |

**Core Functions/Methods:**

- `calculate_client_view_time()` — Simple formula: `server_time - RTT/2 - interpolation_delay`
- `add_rtt_sample()` — Updates running stats (avg, variance, min, max)
- `get_interpolated_view_time()` — Jitter-compensated view time with configurable buffer
- `get_conservative_view_time()` — Uses min RTT (defender-favored)
- `get_liberal_view_time()` — Uses max RTT + jitter buffer (shooter-favored)
- `get_view_time_range()` — Returns (conservative, liberal) tuple for uncertainty window

**Anti-Cheat Features (LagCompensationValidator):**
- `register_client()` — Creates per-client calculator
- `validate_view_time_claim()` — Checks claimed view time against expected range, records violations
- `is_suspicious()` — Returns true if violations exceed threshold

**Implementation Quality:**
- Standard deviation-based jitter calculation
- Clamping to `max_lag_compensation_ms`
- Violation tracking per-client for anti-cheat flagging
- Uses config constants (`JITTER_STANDARD_DEVIATIONS`, `DEFAULT_SUSPICIOUS_THRESHOLD`)

---

### 4. __init__.py (37 lines) — REAL

**Purpose:** Public API exports.

**Exports:**
- `RewindManager`, `WorldState`, `HistoryFrame` (rewind subsystem)
- `HitboxHistory`, `HitboxSnapshot`, `Bounds` (hitbox subsystem)
- `ViewTimeCalculator`, `calculate_client_view_time` (view time subsystem)

---

## Architecture Summary

```
                    ┌─────────────────────────────────────────────────┐
                    │              Client fires weapon                │
                    └────────────────────┬────────────────────────────┘
                                         │
                    ┌────────────────────▼────────────────────────────┐
                    │  ViewTimeCalculator.get_interpolated_view_time  │
                    │  (RTT/2 + interpolation_delay + jitter_buffer)  │
                    └────────────────────┬────────────────────────────┘
                                         │
              ┌──────────────────────────┴──────────────────────────┐
              │                                                     │
┌─────────────▼─────────────┐                   ┌───────────────────▼───────────────────┐
│   RewindManager           │                   │   HitboxHistory                       │
│   .rewind_to(view_time)   │                   │   .get_interpolated_hitbox(view_time) │
│   → full world state      │                   │   → efficient hitbox-only lookup      │
└─────────────┬─────────────┘                   └───────────────────┬───────────────────┘
              │                                                     │
              └───────────────────────┬─────────────────────────────┘
                                      │
                    ┌─────────────────▼─────────────────────────────┐
                    │  Perform hit detection at historical state    │
                    └─────────────────┬─────────────────────────────┘
                                      │
                    ┌─────────────────▼─────────────────────────────┐
                    │  RewindManager.restore_to_current()           │
                    │  Apply damage/effects in live state           │
                    └───────────────────────────────────────────────┘
```

---

## Key Patterns Identified

### 1. Circular Buffer History
Both `HitboxHistory` and `RewindManager` use `deque(maxlen=N)` for automatic pruning:
```python
self._history: deque[HistoryFrame] = deque(maxlen=self._max_frames)
```

### 2. Deep Copy for Immutable History
`WorldState.copy()` and `EntityState.copy()` ensure recorded frames are not mutated by game simulation:
```python
frame = HistoryFrame(
    tick=tick,
    timestamp=world_state.timestamp,
    world_state=world_state.copy(),  # Deep copy
)
```

### 3. Interpolation Between Frames
Sub-tick accuracy via linear interpolation:
```python
t = (timestamp - before.timestamp) / duration
position = (
    before.position[0] + (after.position[0] - before.position[0]) * t,
    ...
)
```

### 4. Anti-Cheat Validation Pattern
```python
def validate_view_time_claim(self, client_id, claimed_view_time, server_time):
    expected = calculator.get_interpolated_view_time(server_time)
    if abs(claimed_view_time - expected) > threshold:
        self._violation_counts[client_id] += 1
        return False, expected  # Correct to server-calculated value
```

---

## Configuration Dependencies

All modules import from `engine.networking.config`:
- `DEFAULT_HITBOX_HISTORY_FRAMES`
- `DEFAULT_TICK_RATE`
- `HITBOX_CACHE_MULTIPLIER`
- `DEFAULT_MAX_REWIND_TIME_MS`
- `DEFAULT_MAX_LAG_COMPENSATION_MS`
- `DEFAULT_CLIENT_INTERPOLATION_DELAY_MS`
- `DEFAULT_JITTER_BUFFER_MS`
- `DEFAULT_MIN_RTT_SAMPLES`
- `DEFAULT_RTT_HISTORY_SIZE`
- `JITTER_STANDARD_DEVIATIONS`
- `DEFAULT_MAX_VIEW_TIME_DEVIATION_MS`
- `DEFAULT_SUSPICIOUS_THRESHOLD`
- `calculate_max_history_frames()`

---

## Integration Points

| System | Integration |
|--------|-------------|
| **Server tick loop** | Calls `record_frame()` / `hitbox.record()` each tick |
| **Hit detection** | Queries `rewind_to()` or `get_interpolated_hitbox()` |
| **Network layer** | Feeds RTT samples to `ViewTimeCalculator` |
| **Anti-cheat** | Uses `LagCompensationValidator` to flag suspicious clients |

---

## Gaps / Future Work

1. **Rotation interpolation** — Explicitly skipped in both managers ("for simplicity"). Quaternion slerp would improve accuracy for fast-rotating entities.

2. **No extrapolation** — If client view time is slightly ahead of newest frame, it clamps to newest. Could extrapolate using velocity.

3. **No per-bone hitboxes** — Current `Bounds` is a single AABB. Skeletal hitboxes would require pose history.

4. **No multi-hit reconciliation** — Each shot is validated independently. Burst weapons may benefit from batched validation.

---

## Test Coverage Needs

- Unit tests for `Bounds` intersection/containment
- Unit tests for interpolation accuracy
- Integration tests for rewind-validate-restore cycle
- Edge cases: empty history, single frame, timestamp at exact frame boundary
- Anti-cheat tests: violation accumulation, threshold triggering

---

## Summary

The lag compensation module is fully implemented with clean separation of concerns:
- **HitboxHistory** for lightweight per-entity tracking
- **RewindManager** for full world state rollback
- **ViewTimeCalculator** for client perception timing
- **LagCompensationValidator** for anti-cheat enforcement

No stubs, placeholders, or incomplete implementations detected. Ready for integration and testing.
