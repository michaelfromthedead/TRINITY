# Phase 3 Architecture: Sound Playback Engine

## Purpose
High-level playback controls: AudioSource component, AudioClip asset loading, command dispatch to audio thread, sound cue selection and variation, AudioListener 3D positioning, and decorator integration.

## Current Implementation
**7/14 tasks complete [x], 1 partial [~], 6 missing [-].**

### AudioSource (`core/audio_source.py`, 704 lines) [x]
- `AudioSource` dataclass: volume (0-1), pitch (0.1-4.0), pan (-1 to 1), muted
- `PlaybackMode`: ONCE, LOOP, PING_PONG
- Fade: fade_in_time/fade_out_time, fade progress tracking
- 3D: position, velocity, spread, priority
- `SourceType`: ONE_SHOT, LOOPING, STREAMING
- Callbacks: on_start, on_stop, on_finish, on_pause, on_resume, custom event callbacks
- `AudioSourcePool`: acquire/release with pool recycling
- Playback state machine with source-level lifecycles

### AudioClip (`core/audio_clip.py`, 587 lines) [x]
- PCM data buffer, sample_rate, channels, duration, format
- Metadata: name, tags, description
- Reference counting (retain/release)
- Loop points: loop_start, loop_end, loop_crossfade
- `get_samples(position, num_samples)`: returns byte data

### Sound Cues (`core/sound_cue.py`, 622 lines) [x]
- `SoundCue` / `SoundCueManager`: variation container
- `SoundEntry`: clip reference, weight
- `CueVariation`: pitch_randomization, volume_randomization, start_offset_randomization
- 5 selection modes: SIMPLE, RANDOM, SEQUENCE, SWITCH, SHUFFLE
- Variation constants: `PITCH_VARIATION_RANGE=0.1`, `VOLUME_VARIATION_DB=3.0`

### AudioEngine (`core/audio_engine.py`, 824 lines) [x]
- `AudioEngine`: threading model (game thread -> audio thread)
- Command types: PlayCommand, StopCommand, PauseCommand, ResumeCommand, SetVolumeCommand, etc.
- `update(dt)`: per-frame processing of voice state, commands, spatial updates, fade progress
- `play_source()`, `stop_source()`, `pause_source()`, `resume_source()`

### AudioListener (`core/audio_listener.py`, 506 lines) [x]
- `AudioListener`: position, velocity, forward, up (all Vector3)
- Damping, speed_of_sound configuration
- `AudioListenerManager`: manage multiple listeners
- `Vector3`: normalize, dot, cross, distance, lerp

### Decorator [x]
- `@sound(bank, preload)` in `trinity/decorators/audio.py`: validates bank (required), preload (optional), generates TAG+REGISTER ops, registered with Tier.AUDIO

### Architecture
```
Playback Pipeline:
  1. Game code calls AudioEngine.play_sound(sound_id, position, ...)
  2. Engine resolves AudioClip from sound bank
  3. Acquires voice from VoiceManager (or virtualizes)
  4. Pushes PlayCommand to command queue (queue.Queue)
  5. Audio thread pops command, starts playback
  6. Per-frame: AudioEngine.update() processes active voices

Format Loading Path:
  WAV -> Python wave module -> PCM f32
  OGG/FLAC/MP3/Opus -> [decoder binding needed]

Sound Cue Resolution:
  SoundCue (SIMPLE) -> always plays same clip
  SoundCue (RANDOM) -> random weighted selection
  SoundCue (SEQUENCE) -> next in order
  SoundCue (SWITCH) -> game-state dependent
  SoundCue (SHUFFLE) -> shuffle without repeats
```

### Missing (6 tasks)
| Task | Component | Priority |
|------|-----------|----------|
| T-AU-3.5 | WAV loader (dedicated) | Low (Python wave works) |
| T-AU-3.6 | OGG/Vorbis loader | High |
| T-AU-3.7 | FLAC loader | High |
| T-AU-3.8 | MP3 loader | High |
| T-AU-3.9 | Opus loader | High |
| T-AU-3.3 | `@tracked`/TrackedDescriptor | Medium (Foundation dep) |

### Partial (1 task)
| Task | Status | Gap |
|------|--------|-----|
| T-AU-3.10 | Command queue exists | Uses `queue.Queue`, not lock-free SPSC |
| T-AU-3.11 | AudioEngine.update() exists | Not wired as Foundation `@system` |
