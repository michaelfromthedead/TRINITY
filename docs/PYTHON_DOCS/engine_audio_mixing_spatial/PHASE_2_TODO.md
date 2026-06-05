# PHASE 2 TODO: Audio Mixing Dynamics

**RDC Phase Task Breakdown**
**Phase**: Ducking, Sidechain, HDR, and Snapshots

---

## Task 2.1: DuckingManager Implementation

**File**: `engine/audio/mixing/ducking.py`
**Estimated Lines**: ~675

### Subtasks

- [ ] Define `DuckType` enum (DIALOGUE, EVENT, FOCUS)
- [ ] Define `DuckState` enum (IDLE, ATTACKING, HOLDING, RELEASING)
- [ ] Define `DuckConfig` dataclass with timing and reduction params
- [ ] Define `DuckInstance` class with FSM state and envelope
- [ ] Implement `DuckingManager.__init__` with instance registry
- [ ] Implement `create_duck()` with trigger/target bus binding
- [ ] Implement `remove_duck()` cleanup
- [ ] Implement `_update_envelope()` per-tick FSM advancement
- [ ] Implement `_check_trigger()` level threshold detection
- [ ] Implement `get_reduction_for_bus()` query for Mixer stage 6
- [ ] Implement `notify_source_start()` / `notify_source_stop()` callbacks
- [ ] Add preset duck configurations (VO ducks music, etc.)

### Acceptance Criteria

- Envelope follows attack/hold/release timing accurately
- Level threshold comparison uses dB scale
- Multiple ducks on same target combine multiplicatively
- FSM state transitions logged for debugging

---

## Task 2.2: SidechainManager Implementation

**File**: `engine/audio/mixing/sidechain.py`
**Estimated Lines**: ~500

### Subtasks

- [ ] Define `SidechainConfig` dataclass with compressor params
- [ ] Define `SidechainInstance` class with envelope state
- [ ] Implement `SidechainManager.__init__` with instance registry
- [ ] Implement `create_sidechain()` binding key bus to target bus
- [ ] Implement `remove_sidechain()` cleanup
- [ ] Implement `_calculate_gain_reduction()` with soft knee:
  - [ ] Below knee_start: no reduction
  - [ ] Above knee_end: full ratio compression
  - [ ] In knee: gradual onset
- [ ] Implement `_update_envelope()` with attack/release smoothing
- [ ] Implement `process_sidechain()` called from Mixer stage 7
- [ ] Implement makeup gain application
- [ ] Add lookahead buffer (optional)

### Acceptance Criteria

- Soft knee produces smooth compression onset
- Gain reduction formula: `overshoot * (1 - 1/ratio)`
- Attack/release envelope prevents pumping artifacts
- Makeup gain compensates for volume loss

---

## Task 2.3: HDRAudioManager Implementation

**File**: `engine/audio/mixing/hdr_audio.py`
**Estimated Lines**: ~550

### Subtasks

- [ ] Define `HDRConfig` dataclass with window params
- [ ] Define `MixWindow` class with floor/ceiling tracking
- [ ] Implement `HDRAudioManager.__init__` with source priority map
- [ ] Implement `register_source()` / `unregister_source()` with priority
- [ ] Implement `_calculate_loudness()` weighted by priority
- [ ] Implement `_adapt_window()` sliding window adjustment
- [ ] Implement `map_level()` input -> output through window
- [ ] Implement `process_source()` for individual source HDR
- [ ] Implement `get_window_position()` debug query
- [ ] Add protected source bypass (VO, UI)
- [ ] Add minimum window floor to prevent silence

### Acceptance Criteria

- Window slides smoothly based on loudness
- High-priority sources pull window toward their level
- Protected sources always map to their natural level
- No audible pumping from window adaptation

---

## Task 2.4: SnapshotManager Implementation

**File**: `engine/audio/mixing/mix_snapshot.py`
**Estimated Lines**: ~675

### Subtasks

- [ ] Define `BusSnapshot` dataclass with bus state
- [ ] Define `MixSnapshot` class with all bus snapshots
- [ ] Implement `SnapshotManager.__init__` with snapshot registry
- [ ] Implement `capture()` create snapshot from current state
- [ ] Implement `recall()` apply snapshot immediately
- [ ] Implement `transition()` blend to snapshot over time
- [ ] Implement `_interpolate_snapshots()` weighted blend
- [ ] Implement `set_active_weight()` for multi-snapshot blending
- [ ] Implement `_resolve_conflicts()` priority-based resolution
- [ ] Add preset snapshots: default, combat, stealth, menu, cutscene
- [ ] Implement `export()` / `import()` for snapshot persistence

### Acceptance Criteria

- Capture preserves all bus and send state
- Recall restores state atomically
- Transitions interpolate smoothly over specified duration
- Multiple active snapshots blend by weight

---

## Task 2.5: Sidechain Bridge

**File**: `engine/audio/mixing/sidechain_bridge.py`
**Estimated Lines**: ~300

### Subtasks

- [ ] Implement bridge between Mixer and SidechainManager
- [ ] Handle key signal routing
- [ ] Manage sidechain processing order
- [ ] Add debug visualization hooks

### Acceptance Criteria

- Key signal available before target processing
- Multiple sidechains per target combine correctly

---

## Dependencies

- Phase 1: Mixer, MixBus, BusRouter
- NumPy for envelope calculations

---

## Verification

1. Ducking test: play dialogue, verify music ducks
2. Sidechain test: kick drums pump bass
3. HDR test: quiet scene followed by explosion, verify smooth adaptation
4. Snapshot test: transition from gameplay to menu, verify blend
