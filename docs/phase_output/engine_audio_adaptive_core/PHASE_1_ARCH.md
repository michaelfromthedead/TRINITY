# PHASE 1 ARCHITECTURE: Adaptive Music System

**Phase**: 1 of 3
**Status**: COMPLETE
**Lines**: ~5,606
**Classification**: REAL IMPLEMENTATION

---

## 1. Overview

Phase 1 delivers the complete adaptive music system with vertical layering and horizontal sequencing. This phase is architecturally complete and functionally operational.

---

## 2. Components

### 2.1 AdaptiveMusicSystem (adaptive_music.py)

**Purpose**: Top-level coordinator combining vertical and horizontal systems.

**Dependencies**:
- MusicClock (music_timing.py)
- LayeredMusicPlayer (music_stem.py)
- MusicCallbackManager (music_callback.py)
- MusicStateManager (music_state.py)
- TransitionManager (music_transition.py)
- StingerManager (stinger.py)

**Key Methods**:
- `update(delta_time)`: Main tick, processes all subsystems
- `set_intensity(value)`: Controls vertical remixer
- `trigger_state_change(state)`: Drives horizontal sequencer

### 2.2 VerticalRemixer (adaptive_music.py)

**Purpose**: Controls stem volumes based on gameplay intensity (0.0-1.0).

**Data Structures**:
- `IntensityLevel`: Dataclass with level_id, threshold, layer volume dict
- `_levels`: List of intensity levels, sorted by threshold
- `_current_intensity`: Smoothed intensity value

**Algorithm**:
```python
intensity += diff * smoothing * delta * rate
for level in levels:
    if intensity >= level.threshold:
        active_level = level
for stem, target_volume in active_level.layers.items():
    stem.fade_to(target_volume, fade_time)
```

### 2.3 HorizontalSequencer (adaptive_music.py)

**Purpose**: Section-based music branching.

**Branching Types**:
- SEQUENTIAL: Play sections in order
- RANDOM: Random selection from valid transitions
- WEIGHTED: Weighted random based on section weights
- RULE_BASED: Evaluate transition rules

**Data Structures**:
- `MusicSection`: Dataclass with section_id, duration, transitions, weights, rules
- `_sections`: Dict of section_id to MusicSection
- `_current_section`: Active section
- `_loop_count`: Current loop iteration

### 2.4 MusicStateManager (music_state.py)

**Purpose**: Gameplay state machine driving music selection.

**States** (10 total):
- exploration, combat, stealth, victory, defeat
- boss, menu, cutscene, ambient, tension

**Priority System**: Higher priority states override lower:
- boss (highest) > combat > stealth > exploration (lowest)

**Data Structures**:
- `MusicState`: Dataclass with state_id, priority, min_duration, transitions
- `_states`: Dict of state_id to MusicState
- `_history`: Deque of previous states
- `_parameters`: Dict of gameplay parameters (danger, intensity, area)

### 2.5 MusicClock/BeatGrid (music_timing.py)

**Purpose**: BPM-locked timing for musical synchronization.

**Key Calculations**:
- beat_duration_ms = 60000 / bpm
- bar_duration_ms = beat_duration_ms * beats_per_bar
- time_to_beat(ms) = ms / beat_duration_ms
- beat_to_bar(beat) = beat / beats_per_bar

**Quantization**: Snap to beat, bar, or subdivision boundaries.

### 2.6 LayeredMusicPlayer (music_stem.py)

**Purpose**: Multi-stem playback with individual fade control.

**Stem Types** (8 default):
- drums, bass, melody, pads, strings, percussion, vocals, fx

**Features**:
- Individual volume per stem
- Solo/mute per stem
- Stem groups for batch control
- Fade curves: linear, equal_power, s_curve, exponential, logarithmic

### 2.7 MusicCallbackManager (music_callback.py)

**Purpose**: Event notification for beat/bar/marker events.

**Callback Types**:
- BEAT: Fires on each beat
- BAR: Fires on each bar
- MARKER: Fires at specific positions
- TRACK_END: Fires when track ends
- LOOP: Fires at loop point
- SYNC_POINT: Fires at named sync positions

**Precision Target**: 5ms

### 2.8 TransitionManager (music_transition.py)

**Purpose**: Smooth transitions between music pieces.

**Transition Types**:
- CROSSFADE: Simple volume blend
- BEAT_SYNC: Start on next beat
- BAR_SYNC: Start on next bar
- STINGER: Play stinger during transition
- IMMEDIATE: No wait
- EXIT_CUE: Wait for exit marker

### 2.9 StingerManager (stinger.py)

**Purpose**: Short musical impacts for transitions and events.

**Stinger Types**:
- IMPACT: Accent on hit
- TRANSITION: Bridge between sections
- ACCENT: Emphasis
- TAIL: Fadeout/ending

**Alignment**: Beat or bar aligned for musical coherence.

---

## 3. Data Flow

```
Gameplay Events
      |
      v
MusicStateManager.change_state()
      |
      v
AdaptiveMusicSystem.update()
      |
      +---> VerticalRemixer.update() ---> LayeredMusicPlayer.set_stem_volume()
      |
      +---> HorizontalSequencer.update() ---> TransitionManager.start_transition()
      |
      +---> MusicClock.update() ---> MusicCallbackManager.fire_callbacks()
      |
      v
StingerManager.update()
```

---

## 4. Configuration

### 4.1 Constants (adaptive/config.py)

| Constant | Value | Purpose |
|----------|-------|---------|
| BPM_MIN | 30 | Minimum valid BPM |
| BPM_MAX | 300 | Maximum valid BPM |
| FADE_MIN_DURATION | 0.1 | Minimum fade time (seconds) |
| CALLBACK_PRECISION_MS | 5 | Beat callback target precision |
| INTENSITY_SMOOTHING | 0.1 | Intensity change rate |
| MAX_STEMS | 8 | Maximum stems per track |
| STATE_HISTORY_DEPTH | 5 | States to remember |

---

## 5. Thread Safety

All shared state protected by threading locks:
- `_lock` on MusicStateManager
- `_lock` on LayeredMusicPlayer
- `_lock` on MusicCallbackManager

Update methods acquire locks before mutations.

---

## 6. Integration Points

### 6.1 Input (from game)
- `set_intensity(float)`: Gameplay intensity 0.0-1.0
- `set_parameter(name, value)`: Named gameplay parameters
- `change_state(state_id)`: State transition request

### 6.2 Output (to audio core)
- Calls to play/stop/fade sources
- Volume adjustments per stem
- Playback position queries

---

## 7. Known Limitations

1. **Track-end integration incomplete**: music_state.py:574-585 has `pass` for track end detection
2. **Single listener only**: No split-screen music considerations
3. **No MIDI support**: All music is audio stems, not sequenced
