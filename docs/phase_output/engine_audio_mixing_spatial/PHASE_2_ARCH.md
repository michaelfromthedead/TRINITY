# PHASE 2 ARCH: Audio Mixing Dynamics

**RDC Phase Architecture**
**Phase**: Ducking, Sidechain, HDR, and Snapshots

---

## Phase Overview

Add dynamic mixing capabilities: ducking for dialogue/music priority, sidechain compression for pumping effects, HDR audio for dynamic range management, and mix snapshots for state transitions.

---

## Components

### 2.1 DuckingManager

**Purpose**: Reduce volume of target buses when trigger conditions are met.

**Duck Types**:
- `DIALOGUE` - Duck music/sfx when dialogue plays
- `EVENT` - Duck ambient when event plays (explosion, etc.)
- `FOCUS` - Duck everything except focus source

**Envelope FSM**:
```
IDLE -> ATTACKING -> HOLDING -> RELEASING -> IDLE
```

**State per Duck Instance**:
- `trigger_threshold_db: float` - Level to trigger ducking
- `attack_ms: float` - Time to reach target reduction
- `hold_ms: float` - Time to hold at full reduction
- `release_ms: float` - Time to return to normal
- `reduction_db: float` - Amount to reduce target
- `current_state: DuckState` - FSM state
- `envelope_value: float` - Current reduction (0.0 to 1.0)

**Algorithm**:
1. Monitor trigger bus level
2. If level > threshold and state == IDLE: transition to ATTACKING
3. In ATTACKING: ramp envelope_value toward 1.0 over attack_ms
4. In HOLDING: maintain envelope at 1.0 for hold_ms
5. In RELEASING: ramp envelope_value toward 0.0 over release_ms
6. Apply `reduction_db * envelope_value` to target bus

### 2.2 SidechainManager

**Purpose**: Apply compressor to one signal controlled by another (key) signal.

**Compressor Parameters**:
- `threshold_db: float` - Level above which compression starts
- `ratio: float` - Compression ratio (e.g., 4:1)
- `knee_db: float` - Soft knee width
- `attack_ms: float` - Envelope attack time
- `release_ms: float` - Envelope release time
- `makeup_gain_db: float` - Post-compression gain

**Soft Knee Algorithm**:
```python
knee_start = threshold - knee / 2
knee_end = threshold + knee / 2
if input_db <= knee_start:
    reduction = 0.0
elif input_db >= knee_end:
    overshoot = input_db - threshold
    reduction = overshoot * (1 - 1/ratio)
else:
    knee_factor = (input_db - knee_start) / knee
    effective_ratio = 1 + (ratio - 1) * knee_factor
    # Gradual compression onset
```

### 2.3 HDRAudioManager

**Purpose**: Manage dynamic range through a sliding loudness window.

**Concept**: Instead of hard-limiting, maintain a "window" of audible levels that slides based on content. Quiet scenes have a lower window; loud scenes have a higher window.

**Parameters**:
- `window_db: float` - Size of audible window (e.g., 40dB)
- `floor_db: float` - Output floor (e.g., -60dB)
- `ceiling_db: float` - Output ceiling (e.g., 0dB)
- `adaptation_speed: float` - How fast window slides

**Priority System**:
- Each source has a priority (VO > SFX > Music > Ambient)
- Window position weighted by priority of active sources
- Protected sources (VO, UI) bypass HDR entirely

**Level Mapping**:
```python
def map_level(input_db: float) -> float:
    if input_db <= window_floor:
        return MIN_VOLUME_DB  # Below window
    if input_db >= window_ceiling:
        return ceiling_db  # Above window
    # Linear mapping within window
    position = (input_db - window_floor) / window_db
    return floor_db + position * (ceiling_db - floor_db)
```

### 2.4 SnapshotManager

**Purpose**: Capture and recall mix state for different game contexts.

**Preset Snapshots**:
- `default` - Normal gameplay
- `combat` - Boost SFX, duck music
- `stealth` - Reduce ambient, focus footsteps
- `menu` - Music only, UI at full
- `cutscene` - VO priority, reduced SFX

**Snapshot Data**:
- Per-bus: volume, mute, solo, filter settings
- Per-send: gain values
- Global: master volume

**Blending**:
- Multiple snapshots can be active with weights
- `BusSnapshot.interpolate(a, b, t)` - Weighted blend
- Priority-based conflict resolution

---

## Integration with Phase 1

```
8-Stage Pipeline (from Phase 1)
    |
    +-- Stage 6: Ducking adjustments (DuckingManager)
    |
    +-- Stage 7: HDR + Sidechain (HDRAudioManager, SidechainManager)
```

---

## Thread Safety

1. Ducking envelope updates must be atomic
2. Sidechain key signal analysis cached per tick
3. Snapshot transitions schedule parameter interpolation
4. All managers use `threading.RLock()`

---

## Success Criteria

1. Dialogue triggers music ducking with smooth envelope
2. Sidechain creates audible pumping effect
3. HDR compresses dynamic range without audible artifacts
4. Snapshot transitions blend smoothly over time
