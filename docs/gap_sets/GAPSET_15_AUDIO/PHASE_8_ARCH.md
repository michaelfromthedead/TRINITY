# Phase 8 Architecture: Adaptive Music Engine

## Purpose
Dynamic music system with beat-synced timing, state-driven transitions, vertical layering of stems, horizontal resequencing of segments, stingers for one-shot accents, and full callback infrastructure.

## Current Implementation
**10/14 tasks complete [x], 2 partial [~], 2 missing [-].**

### Music Core (`adaptive/music_timing.py`) [x]
- `BeatGrid`: BPM 30-300, `TimeSignature` validation (beat_unit must be power of 2)
- `quantize_to_beat()`/`quantize_to_bar()`, `next_beat()`/`next_bar()` calculations
- Subdivision support (1/4, 1/8, 1/16, 1/32)
- `MusicClock`: start/stop/pause/seek, `time_until_next_beat()`, BPM ramping
- `SyncPointManager`: register sync points at beat/bar positions

### State Management (`adaptive/music_state.py`) [x]
- `MusicStateManager`: 10 predefined states with priority stack
  - Priority: MENU=0, AMBIENT=1, EXPLORATION=2, CUTSCENE=3, STEALTH=4, TENSION=5, COMBAT=6, BOSS=7, VICTORY/ DEFEAT=8
- `push_state()`/`pop_state()` with priority-based ordering
- `StateChangeRule`: valid_from/valid_to transition validation
- Parameter triggers: danger parameter -> combat/tension automatic transition
- Factory functions: `create_exploration_state()`, `create_combat_state()`, etc.

### Horizontal Resequencing (`adaptive/adaptive_music.py`) [x]
- `HorizontalSequencer`: segment selection modes (random/weighted/sequential/rule-based)
- State-to-segment mapping via `register_state_segments(state_id, segments)`
- Beat-synced segment boundaries for seamless transitions

### Vertical Remixing (`adaptive/music_stem.py`) [x]
- `LayeredMusicPlayer`: stem groups, layer indices (0-7)
- `FadeCurve`: linear, equal_power, s_curve, exponential
- `SoloGroup`: exclusive solo/mute per group
- `MixGroup`: group master volume
- `activate_stems_by_intensity(intensity)`: maps [0,1] to layer activation
- `MusicStem`: StemState (INACTIVE/ACTIVE/FADING_IN/FADING_OUT/MUTED)

### Transitions (`adaptive/music_transition.py`) [x]
- 6 transition types: INSTANT, CROSSFADE, BEAT_MATCH, STINGER_LEAD, ABRUPT_STOP, INTRO_TO_LOOP
- `TransitionRequest`: priority queue for pending transitions
- `MusicTransition`: fade progress curves with source/destination volumes
- `TransitionManager`: beat/bar quantization, overlap management

### Stingers (`adaptive/stinger.py`) [x]
- `StingerInfo`: duration validation (0.1-5.0s), type validation
- `Stinger`: scheduled/play/stop/fade lifecycle
- `StingerManager`: register/unregister, `play_random_stinger(type, tag)`, beat-aligned triggering

### Callbacks (`adaptive/music_callback.py`) [x]
- `MusicCallbackManager`: register by type (beat/bar/marker/track_end/loop/state_change)
- Priority ordering, filter functions for conditional firing
- `BeatScheduler`: schedule callbacks N bars/beats in future

### Player (`adaptive/music_player.py`) [x]
- `TrackInfo`: duration_ms, bpm, time_signature, loop/intro/outro regions
- `Playlist`: linear/loop/shuffle/adaptive modes, shuffle order generation
- `MusicPlayer`: play/stop/pause/seek, crossfade between tracks

### Orchestrator (`adaptive/adaptive_music.py`) [x]
- `AdaptiveMusicSystem`: integrates all subsystems
- `set_parameter(name, value, immediate)`: bind game params to music behavior
- `trigger_combat()`/`trigger_exploration()`/`trigger_stealth()` convenience methods
- `start_update_loop(interval_ms=16)`: 60fps update loop

### Decorators [x]
- `@music_stem(group, layer, sync_to_beat)` in `audio_extended.py`: validates group/layer/sync_to_beat
- `@music_transition(from_state, to_state, type, duration_beats)` in `audio_extended.py`: validates transition params
- Both register with `Tier.AUDIO_EXTENDED`

### Architecture
```
Music Update Loop (60fps):
  AdaptiveMusicSystem.update(delta):
    1. MusicClock.tick() -> advance beat/bar position
    2. Check pending transitions (quantized to beat boundary)
    3. Update stem volumes (fade progress, intensity changes)
    4. Fire beat/bar callbacks
    5. Evaluate parameter triggers for state changes
    
State Management:
  push_state(combat):
    - Evaluate transition rules: current -> combat
    - Queue TransitionRequest with type (e.g., CROSSFADE)
    - TransitionManager executes at next beat/bar boundary
    - StateManager updates active state stack
    
Vertical Remixing:
  set_intensity(0.7):
    - LayeredMusicPlayer activates stems up to layer 0.7
    - Lower layers already active, new layer fades in
    - FadeCurve determines ramp shape
    
Horizontal Resequencing:
  state=combat -> HorizontalSequencer picks segment:
    - Random: any combat segment
    - Weighted: weighted toward recently unused
    - Rule-based: game state conditions
```

### Missing (2 tasks)
| Task | Component | Gap |
|------|-----------|-----|
| T-AU-8.11 | `@adaptive_audio` composite | Composite decorator stack not implemented |
| T-AU-8.13 | Session persistence | Music state/intensity not persisted |

### Partial (2 tasks)
| Task | Gap |
|------|-----|
| T-AU-8.3 | `@state`/StateMeta | MusicStateManager exists as Python class, not Foundation decorator |
| T-AU-8.14 | `@audio_snapshot`<->music | Snapshot decorator exists, not wired to AdaptiveMusicSystem |
