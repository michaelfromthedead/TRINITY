# PHASE 1 TODO: Adaptive Music System

**Phase**: 1 of 3
**Status**: COMPLETE (minor polish items remain)

---

## Completed Tasks

- [x] VerticalRemixer implementation with intensity levels
- [x] HorizontalSequencer with 4 branching types
- [x] MusicStateManager with 10 states and priorities
- [x] MusicClock/BeatGrid with quantization
- [x] LayeredMusicPlayer with 8 stem types
- [x] MusicCallbackManager with 5ms precision target
- [x] TransitionManager with 6 transition types
- [x] StingerManager with beat/bar alignment
- [x] Fade curves (linear, equal_power, s_curve, exponential, logarithmic)
- [x] Thread safety with locks
- [x] Configuration centralization in config.py

---

## Remaining Tasks

### T-AUDIO-1.1: Track End Integration

**Priority**: Medium
**Location**: `music_state.py:574-585`
**Description**: Wire track end detection to MusicStateManager for automatic state progression.

**Acceptance Criteria**:
- [ ] MusicStateManager receives track-end notifications from player
- [ ] Automatic state transitions based on track completion
- [ ] Loop/no-loop configuration per state

**Effort**: Small (1-2 hours)

---

### T-AUDIO-1.2: Stem Group Persistence

**Priority**: Low
**Location**: `music_stem.py`
**Description**: Save/load stem group configurations for editor tooling.

**Acceptance Criteria**:
- [ ] Stem groups serializable to JSON
- [ ] Load stem group presets from files
- [ ] Editor integration for group management

**Effort**: Small (2-3 hours)

---

### T-AUDIO-1.3: BPM Change Support

**Priority**: Low
**Location**: `music_timing.py`
**Description**: Support tempo changes mid-track (rallentando, accelerando).

**Acceptance Criteria**:
- [ ] BPM can change at runtime
- [ ] Beat callbacks adjust to new tempo
- [ ] Smooth tempo interpolation option

**Effort**: Medium (4-6 hours)

---

### T-AUDIO-1.4: Rule-Based Branching DSL

**Priority**: Low
**Location**: `adaptive_music.py`
**Description**: Formalize rule syntax for horizontal branching decisions.

**Acceptance Criteria**:
- [ ] Document rule syntax
- [ ] Validate rules at load time
- [ ] Support compound conditions (AND, OR)

**Effort**: Medium (4-6 hours)

---

## Verification Tasks

### V-AUDIO-1.1: Callback Precision Validation

**Description**: Verify 5ms beat callback precision under load.

**Steps**:
1. Create test with 120 BPM (500ms beats)
2. Register beat callback, log timestamps
3. Run 1000 beats, measure deviation
4. Pass if 95th percentile < 10ms deviation

---

### V-AUDIO-1.2: Intensity Smoothing Test

**Description**: Verify intensity changes are smoothed, not jumpy.

**Steps**:
1. Set intensity to 0.0
2. Jump to 1.0
3. Sample intensity value each frame
4. Pass if no frame-to-frame delta exceeds smoothing rate

---

### V-AUDIO-1.3: State Priority Test

**Description**: Verify higher priority states override lower.

**Steps**:
1. Enter exploration state
2. Request combat state
3. Verify combat active (higher priority)
4. Request exploration state
5. Verify combat still active (combat > exploration)
6. Exit combat explicitly
7. Verify exploration becomes active

---

## Dependencies

| Task | Depends On | Blocks |
|------|------------|--------|
| T-AUDIO-1.1 | Phase 2 AudioEngine | Nothing |
| T-AUDIO-1.2 | Nothing | Editor integration |
| T-AUDIO-1.3 | Nothing | Nothing |
| T-AUDIO-1.4 | Nothing | Complex adaptive scores |
