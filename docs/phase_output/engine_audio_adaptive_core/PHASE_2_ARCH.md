# PHASE 2 ARCHITECTURE: Audio Core Engine

**Phase**: 2 of 3
**Status**: PARTIAL (backend stubbed)
**Lines**: ~4,994
**Classification**: PARTIAL IMPLEMENTATION

---

## 1. Overview

Phase 2 delivers the core audio engine with voice management, memory pooling, 3D spatial audio, and format handling. The architecture is complete; the platform backend binding is stubbed.

---

## 2. Components

### 2.1 AudioEngine (audio_engine.py)

**Purpose**: Central coordinator for all audio operations.

**Threading Model**:
| Thread | Tick Rate | Responsibility |
|--------|-----------|----------------|
| Game | Variable | Send commands, manage high-level state |
| Audio | 200Hz (5ms) | Process commands, mix audio |
| Stream | 100Hz (10ms) | File I/O for streaming |
| Decode | As needed | Compressed format decoding |

**Command Queue**:
- Type-safe command classes: PlayCommand, StopCommand, PauseCommand, FadeCommand, etc.
- Thread-safe queue between game and audio threads
- Ensures game thread never directly mutates audio state

### 2.2 VoiceManager (voice_manager.py)

**Purpose**: Allocate and manage voice resources.

**Voice Limits**:
| Category | Limit |
|----------|-------|
| Total | 64 |
| SFX | 48 |
| Music | 8 |
| VO | 8 |
| Ambient | 16 |

**Stealing Strategies**:
- OLDEST: Steal longest-running voice
- QUIETEST: Steal softest voice
- FARTHEST: Steal most distant voice
- LOWEST_PRIORITY: Steal least important voice

**Data Structures**:
- `_active_voices`: Dict of voice_id to VoiceState
- `_virtual_voices`: Dict of voice_id to VirtualVoice
- `_category_counts`: Dict of category to active count

### 2.3 VirtualVoiceTracker (virtual_voice.py)

**Purpose**: Track virtualized sounds for later promotion.

**Urgency Scoring**:
- Distance to listener (closer = higher urgency)
- Priority level
- Time since virtualization
- Current volume (if playing)

**Promotion**: When real voices free up, highest-urgency virtual voices promote.

### 2.4 AudioMemoryManager (memory_manager.py)

**Purpose**: Pool and manage audio memory.

**Memory Pools**:
| Pool | Size | Purpose |
|------|------|---------|
| RESIDENT | 64MB | Always-loaded sounds |
| STREAMING | 32MB | Ring buffers for large files |
| TEMPORARY | 16MB | Short-lived allocations |

**Per-Category Budgets**:
| Category | Budget |
|----------|--------|
| SFX | 128MB |
| Music | 64MB |
| VO | 64MB |
| Ambient | 32MB |
| Total | 256MB |

**Eviction Algorithm**: LRU with priority weighting
```python
candidates = [block for block in blocks if not block.pinned]
candidates.sort(key=lambda b: (b.priority, b.last_access))
for block in candidates:
    if freed >= needed: break
    freed += block.size
```

### 2.5 StreamBuffer (memory_manager.py)

**Purpose**: Circular buffer for streaming audio.

**Data Structures**:
- `_buffer`: bytearray of fixed size
- `_read_pos`: Read pointer position
- `_write_pos`: Write pointer position
- `_low_watermark`: Trigger prefetch threshold
- `_high_watermark`: Stop prefetch threshold

**Wrap-around**: Handled via modulo arithmetic.

### 2.6 AudioSource (audio_source.py)

**Purpose**: Individual playable sound instance.

**Properties**:
- volume, pitch, pan
- position (3D Vector3)
- velocity (for Doppler)
- loop_start, loop_end
- fade_state (in, out, sustain)

**Pool**: AudioSourcePool provides object reuse (32 initial, 128 max).

### 2.7 AudioClip (audio_clip.py)

**Purpose**: Loaded audio data with metadata.

**Format Detection**:
| Format | Magic | Notes |
|--------|-------|-------|
| WAV | RIFF | Full header parsing |
| OGG | OggS | Simplified (default sample rate) |
| FLAC | fLaC | STREAMINFO parsing |
| MP3 | ID3/sync | Tag or sync word detection |

**AudioClipManager**: Caches clips, reference counts for sharing.

### 2.8 AudioListener (audio_listener.py)

**Purpose**: 3D listener position and orientation.

**3D Calculations**:
- **Distance**: `sqrt((sx-lx)^2 + (sy-ly)^2 + (sz-lz)^2)`
- **Attenuation**: `min_dist / (min_dist + rolloff * (dist - min_dist))`
- **Pan**: Based on angle to listener forward vector
- **Doppler**: `(speed - listener_v) / (speed - source_v)` with 343 m/s speed of sound

**AudioListenerManager**: Multi-listener support for split-screen.

### 2.9 SoundCue (sound_cue.py)

**Purpose**: Variation containers for sound playback.

**Play Modes**:
- RANDOM: Random selection from pool
- SEQUENCE: Play in order
- SHUFFLE: Randomized non-repeating order
- SWITCH: Parameter-driven selection

**Variation**:
- Pitch: `base * (1 + random(-range, +range))`
- Volume: `base * 10^(random_db / 20)`
- Start offset: ratio of duration
- Repeat avoidance: configurable history depth

**SoundCueBuilder**: Fluent API for cue construction.
**SoundCueManager**: Registry and instance tracking.

### 2.10 VoicePriorityBridge (voice_priority_bridge.py)

**Purpose**: Bridge between @voice_priority decorator and VoiceManager.

**Integration**: Allows decorators to specify voice priority without direct VoiceManager coupling.

---

## 3. Data Flow

```
Game Code: engine.play(clip, ...)
        |
        v
CommandQueue.enqueue(PlayCommand)
        |
        v (Audio Thread)
AudioEngine._process_commands()
        |
        v
VoiceManager.allocate_voice()
        |
        +---> Voice available: return allocation
        |
        +---> Voice unavailable: attempt steal
        |            |
        |            v
        |     _steal_voice() selects victim
        |            |
        |            v
        |     victim -> VirtualVoiceTracker
        |
        v
AudioSource configured from pool
        |
        v
3D calculations: attenuation, pan, Doppler
        |
        v
[STUB: _fill_stream_buffers / _process_audio output]
```

---

## 4. Configuration

### 4.1 Constants (core/config.py)

| Constant | Value | Purpose |
|----------|-------|---------|
| AUDIO_THREAD_TICK_MS | 5 | Audio thread tick rate |
| STREAM_THREAD_TICK_MS | 10 | Stream thread tick rate |
| MAX_VOICES | 64 | Total voice limit |
| MAX_INSTANCES_PER_SOUND | 4 | Per-sound instance cap |
| MEMORY_TOTAL | 256MB | Total audio memory |
| SPEED_OF_SOUND | 343 | m/s at 20C |
| DOPPLER_SCALE | 1.0 | Doppler effect intensity |

---

## 5. Thread Safety

All shared state protected:
- `_lock` on VoiceManager
- `_lock` on AudioMemoryManager
- `_lock` on AudioSourcePool
- Command queue is thread-safe by design

---

## 6. Integration Points

### 6.1 Input (from game/adaptive)
- `play(clip, ...)`: Start playback
- `stop(source)`: Stop playback
- `fade(source, target, duration)`: Volume fade
- `set_listener_position(x, y, z)`: Update listener

### 6.2 Output (to platform)
- **STUBBED**: `_fill_stream_buffers()` - needs platform binding
- **STUBBED**: `_process_audio()` - needs audio output backend

---

## 7. Known Limitations

1. **Backend stubbed**: `_fill_stream_buffers` is `pass` - no actual audio output
2. **OGG parsing simplified**: Uses default sample rate
3. **Duplicate constants**: VIRTUAL_VOICE_* defined 3 times
4. **No DSP chain**: Effects handled in separate module (engine/audio/dsp)
