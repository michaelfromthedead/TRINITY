# Audio Layer — Implementation Context

> Everything needed to implement `engine/audio/`. No other document required.
> 
> **Architecture spec:** `DIAGRAMS/ARCHITECTURE_AUDIO.md`
> **Integration spec:** `docs/GAME_ENGINE_INTEGRATION.md` §4.9, §8
> **Trinity spec:** `docs/TRINITY_LATEST.md`
> **TODO checklist:** `docs/GAME_ENGINE_INTEGRATION_TODO.md` §8

---

## 1. Architecture Summary

The audio layer sits parallel to the Resource layer in the engine stack. It handles:
- Sound playback (one-shot, looping, streaming)
- Spatial audio (3D positioning, HRTF, attenuation)
- Acoustic simulation (reverb, occlusion, propagation)
- Mix bus hierarchy (master → submixes → voices)
- Adaptive music (horizontal re-sequencing, vertical remixing)
- DSP effects (filters, dynamics, time-based, distortion, pitch)
- Voice management (priority, virtualization, stealing)
- Dialogue systems (VO queuing, contextual barks)

### Audio Pipeline
```
Source → Decode → Process (DSP) → Spatialize → Mix → Output
```

### Threading Model
- **Game Thread**: commands, state changes, parameter updates
- **Audio Thread**: mixing, DSP processing, voice management
- **Stream Thread**: async audio file streaming
- **Decode Thread(s)**: compressed format decoding

### Memory Model
- **Resident pool**: frequently-used short sounds (UI, footsteps)
- **Streaming pool**: long sounds loaded in chunks (music, ambient)
- **Temporary pool**: one-shot decode buffers

---

## 2. Trinity Decorators for Audio

### 2.1 Core Audio Decorators (Tier 14: AUDIO)

These are in `trinity/decorators/audio.py`:

#### @sound
```python
@sound(bank: str, preload: bool = False)
```
Marks a class as a sound component with a sound bank reference.
- `bank` (str, required): Sound bank identifier
- `preload` (bool, default False): Whether to preload into resident memory
- **Steps**: TAG(sound=True), TAG(sound_bank), TAG(sound_preload), REGISTER(audio)
- **Use**: Any entity that emits sound

#### @audio_bus
```python
@audio_bus(name: str, volume: float = 1.0, effects: Optional[list[str]] = None)
```
Configures an audio bus with routing and effects.
- `name` (str, required): Bus name (e.g., "sfx", "music", "dialogue")
- `volume` (float, 0.0-1.0): Default volume
- `effects` (list[str], optional): Effect chain names
- **Steps**: TAG(audio_bus=True), TAG(bus_name), TAG(bus_volume), TAG(bus_effects), REGISTER(audio)
- **Use**: Mix bus definitions (registered as Resources via ResourceMeta)

#### @spatial_audio
```python
@spatial_audio(falloff: str = "inverse", max_distance: float = 100.0)
```
Enables 3D spatial positioning for a sound source.
- `falloff` ("inverse" | "linear" | "exponential"): Distance attenuation curve
- `max_distance` (float, >0): Maximum audible distance (culled beyond)
- **Steps**: TAG(spatial_audio=True), TAG(audio_falloff), TAG(audio_max_distance), REGISTER(audio)
- **Use**: Any 3D-positioned sound source

### 2.2 Extended Audio Decorators (Tier 49: AUDIO_EXTENDED)

These are in `trinity/decorators/audio_extended.py`:

#### @dsp_node
```python
@dsp_node(inputs: int = 1, outputs: int = 1, latency_samples: int = 0)
```
Defines a DSP processing node in the audio graph.
- `inputs` (int, >0): Number of input channels
- `outputs` (int, >0): Number of output channels
- `latency_samples` (int, >=0): Processing latency in samples
- **Steps**: TAG(dsp_node), TAG(dsp_inputs), TAG(dsp_outputs), TAG(dsp_latency_samples), REGISTER(audio_extended)

#### @voice_priority
```python
@voice_priority(priority: int = 0, virtualize: bool = True, steal_oldest: bool = True)
```
Voice management with priority-based allocation and stealing.
- `priority` (int): Higher = more important, less likely stolen
- `virtualize` (bool): Continue tracking position when inaudible
- `steal_oldest` (bool): Steal oldest voice when limit reached
- **Steps**: TAG(voice_priority), TAG values, REGISTER(audio_extended)

#### @occlusion
```python
@occlusion(method: Literal["raycast", "propagation", "baked"], max_occlusion: float = 1.0)
```
Sound occlusion simulation between source and listener.
- `method`: "raycast" (real-time rays), "propagation" (path tracing), "baked" (precomputed)
- `max_occlusion` (float, 0-1): Maximum occlusion amount (1 = fully occluded)
- **Steps**: TAG(occlusion), TAG(occlusion_method), TAG(occlusion_max), REGISTER(audio_extended)

#### @reverb_zone
```python
@reverb_zone(preset: str, fade_distance: float = 5.0)
```
Defines a reverb environment volume.
- `preset` (str, required): Reverb preset name (e.g., "cathedral", "cave", "bathroom")
- `fade_distance` (float, >0): Distance to blend in/out at zone boundary
- **Steps**: TAG(reverb_zone), TAG(reverb_preset), TAG(reverb_fade_distance), REGISTER(audio_extended)

#### @music_stem
```python
@music_stem(group: str, layer: int = 0, sync_to_beat: bool = True)
```
Defines an adaptive music stem (one layer of a multi-layer composition).
- `group` (str, required): Music group this stem belongs to
- `layer` (int, >=0): Layer index (0 = base, higher = overlays)
- `sync_to_beat` (bool): Whether to quantize start/stop to musical beat
- **Steps**: TAG(music_stem), TAG(music_stem_group), TAG(music_stem_layer), TAG(music_stem_sync_to_beat), REGISTER(audio_extended)

#### @music_transition
```python
@music_transition(
    from_state: str, to_state: str,
    type: Literal["immediate", "next_beat", "next_bar", "crossfade"] = "immediate",
    duration_beats: float = 0.0
)
```
Defines a transition rule between music states.
- `from_state` / `to_state` (str, required): State names
- `type`: When/how to transition
- `duration_beats` (float, >=0): Crossfade duration in beats
- **Steps**: TAG(music_transition), TAG values, REGISTER(audio_extended)

#### @audio_snapshot
```python
@audio_snapshot(bus_overrides: dict[str, float], crossfade_time: float = 0.5)
```
A mixer snapshot that overrides bus volumes (e.g., "combat mix", "cutscene mix").
- `bus_overrides` (dict, required): {bus_name: volume} overrides
- `crossfade_time` (float, >=0): Time to blend to this snapshot
- **Steps**: TAG(audio_snapshot), TAG values, REGISTER(audio_extended)

#### @sidechain
```python
@sidechain(source_bus: str, attack: float = 0.01, release: float = 0.1, ratio: float = 4.0)
```
Sidechain compression (duck one bus based on another's level).
- `source_bus` (str, required): The bus that triggers ducking
- `attack` (float, >0): Attack time in seconds
- `release` (float, >0): Release time in seconds
- `ratio` (float, >=1): Compression ratio
- **Steps**: TAG(sidechain), TAG values, REGISTER(audio_extended)

### 2.3 Composite Stack

From `trinity/decorators/builtin_stacks/audio_stacks.py`:

#### @adaptive_audio (parameterized stack)
```python
@adaptive_audio(crossfade_time: float = 0.5, stem_group: str = "music")
```
Expands to:
1. `@music_stem(group=stem_group)`
2. `@music_transition(from_state="explore", to_state="combat", type="crossfade")`
3. `@audio_snapshot(bus_overrides={"master": 0.8}, crossfade_time=crossfade_time)`
4. `@serializable(format="binary")`

Use for: Complete adaptive music system with stems, state transitions, mixing snapshots, and save/load.

---

## 3. Metaclasses Relevant to Audio

### ComponentMeta
Audio sources, listeners, and spatial audio components are **Components**.
```python
@component
@sound(bank="sfx")
@spatial_audio(falloff="inverse", max_distance=50.0)
class GunfireSound(Component):
    volume: float = 1.0
    pitch: float = 1.0
    playing: bool = False
```
ComponentMeta will:
1. Generate `_component_id`
2. Process fields (volume, pitch, playing)
3. Install descriptors (StorageDescriptor for each field)
4. Register with Foundation Registry
5. Record `_metaclass_steps`

### ResourceMeta
Mix buses, audio settings, and the audio engine itself are **Resources** (global singletons).
```python
@resource
@audio_bus(name="master", volume=1.0)
class MasterBus(Resource):
    volume: float = 1.0
    muted: bool = False
```
ResourceMeta will:
1. Generate `_resource_id`
2. Enforce singleton semantics
3. Register with Foundation Registry

### EventMeta
Audio events (sound finished, beat hit, transition complete) are **Events**.
```python
@event
class SoundFinished(Event):
    entity_id: int
    sound_bank: str
    
@event
class BeatHit(Event):
    bar: int
    beat: int
    bpm: float
```
EventMeta will:
1. Generate `_event_id`
2. Build payload schema from annotations
3. Register with Foundation Registry

### StateMeta
Music states (explore, combat, stealth, menu) are **States**.
```python
@state
class MusicState(State):
    _valid_transitions = {
        "explore": ["combat", "stealth", "menu"],
        "combat": ["explore", "victory", "defeat"],
        "stealth": ["explore", "combat"],
    }
```
StateMeta will:
1. Generate `_state_id`
2. Validate transition table
3. Register with Foundation Registry

### AssetMeta
Audio clips and sound banks are **Assets**.
```python
@asset(extensions=[".wav", ".ogg", ".mp3"])
class AudioClip(Asset):
    sample_rate: int = 44100
    channels: int = 2
    duration: float = 0.0
    compressed: bool = False
```
AssetMeta will:
1. Generate `_asset_id`
2. Record supported extensions
3. Register loader with Foundation Registry

---

## 4. Descriptors Relevant to Audio

### TrackedDescriptor
Audio parameters that need change detection (volume, pitch, playing state) should be tracked so the audio thread knows what changed each frame.
```python
@component
@tracked
@sound(bank="sfx")
class AudioSource(Component):
    volume: Annotated[float, Tracked(), Range(0.0, 1.0)] = 1.0
    pitch: Annotated[float, Tracked(), Range(0.1, 4.0)] = 1.0
    playing: Annotated[bool, Tracked()] = False
```
- `TrackedDescriptor.post_set()` -> `tracker.mark_dirty(obj, field, old, new)`
- Audio system queries `tracker.dirty_fields(source)` each frame
- Only dirty parameters are pushed to audio thread

### ValidatedDescriptor / RangeDescriptor
Audio parameters have strict ranges:
- Volume: 0.0-1.0
- Pitch: 0.1-4.0
- Max distance: >0
- Crossfade time: >=0

### ObservableDescriptor
UI and debug tools observe audio state changes:
- `ObservableDescriptor.on_change` fires callbacks when audio parameters change
- Inspector displays live audio state

### NetworkedDescriptor
In multiplayer, some audio state may replicate:
```python
@component
@tracked
@networked(authority="server")
@sound(bank="sfx")
class NetworkedAudioSource(Component):
    sound_id: Annotated[int, Tracked(), Networked()] = 0
    playing: Annotated[bool, Tracked(), Networked()] = False
    # Volume/pitch are client-side only (not networked)
    volume: Annotated[float, Tracked(), Range(0.0, 1.0)] = 1.0
```

### SerializableDescriptor
Audio bus configuration and music state save/load:
- Mix bus volumes save to session
- Current music state saves to session
- `SerializableDescriptor` provides schema for Foundation Serializer

### TransientDescriptor
Runtime-only audio state that should NOT be serialized:
- Currently playing voice handles
- DSP buffer state
- Decode cursor positions
```python
@component
@sound(bank="sfx")
class AudioSource(Component):
    volume: float = 1.0
    _voice_handle: Annotated[int, Transient()] = -1  # Not saved
    _decode_pos: Annotated[int, Transient()] = 0      # Not saved
```

### ProfiledDescriptor
For audio performance monitoring:
- Track how often audio parameters are read/written
- Identify hot audio components

---

## 5. Foundation Integration Points

### 5.1 Registry
- All audio Components registered via ComponentMeta -> Foundation Registry
- All audio Resources (buses, engine config) registered via ResourceMeta
- All audio Events registered via EventMeta
- All audio Assets registered via AssetMeta
- All music States registered via StateMeta
- **Query at startup**: `registry.subclasses(Component)` filtered by `_sound` tag to find all sound components

### 5.2 Tracker
- Audio system reads `tracker.all_dirty()` each frame to find changed audio components
- Pushes dirty parameter updates to audio thread
- Used by: volume changes, pitch changes, play/stop commands
- `tracker.on_change(AudioSource, callback)` for type-level subscriptions

### 5.3 EventLog
- Record audio events: play, stop, fade, transition
- Record music state changes with causal chains
- `@traced` on audio system update methods for profiling
- Used by: replay system (re-trigger sounds at correct frames), debugging

### 5.4 Mirror
- `mirror(audio_source)` returns field info for Inspector display
- Schema hash for audio asset versioning
- Used by: editor audio preview, runtime inspection

### 5.5 Bridge / ShellLang
- `world.query(has=AudioSource, where=lambda s: s.playing)` -- find playing sounds
- `entity.audio_source.volume = 0.5` -- live audio parameter tweaking via Shell
- `AIInterface.execute({"op": "set", "entity": 42, "component": "AudioSource", "field": "volume", "value": 0.3})`

### 5.6 Session
- Audio bus volumes persist in Session
- Current music state persists in Session
- Playing sounds do NOT persist (transient)

---

## 6. Architecture Spec Details

### 6.1 Sound Sources
**Source Types:**
- One-shot: play once, release voice when done
- Looping: play continuously until stopped
- Streaming: load and decode in chunks (for music/ambient)

**Sound Cues (containers):**
- Simple: single sound
- Random: pick from pool (no-repeat, weighted)
- Sequence: play in order
- Switch: select based on game state parameter

**Variation:**
- Pitch randomization (+/- semitones)
- Volume randomization (+/- dB)
- Start offset randomization

### 6.2 Voice Management
- Global voice limit (e.g., 64 simultaneous voices)
- Per-category limits (e.g., max 8 footstep sounds)
- Per-sound limits (e.g., max 3 gunfire from same weapon)
- Priority-based stealing: lowest priority voice stolen first
- Virtual voices: continue tracking position/time when inaudible, resume when audible

### 6.3 Spatial Audio
**Positioning:**
- Point source (most common)
- Area source (rectangular, e.g., waterfall)
- Line source (e.g., river)
- Volume source (e.g., rain inside a room)

**Attenuation:**
- Linear: `volume = 1 - (distance / max_distance)`
- Logarithmic: `volume = 1 / (1 + distance)`
- Inverse: `volume = min_distance / distance`
- Custom curve: artist-defined spline

**Spatialization:**
- Stereo panning (simple, fast)
- HRTF binaural (headphones, VR)
- VBAP (multi-speaker arrays)
- Ambisonics (360 degree capture/playback)

**Speaker Configs:**
- Stereo (2.0)
- Quadraphonic (4.0)
- Surround (5.1, 7.1)
- Object-based (Dolby Atmos, etc.)

### 6.4 Acoustic Simulation
**Reverb:**
- Algorithmic: CPU-efficient, parameterized (RT60, room size, damping, diffusion, mix)
- Convolution: realistic, uses impulse response recordings
- Hybrid: algorithmic early reflections + convolution tail

**Occlusion/Obstruction:**
- Raycast: single ray source to listener
- Multi-ray: multiple rays for more accuracy
- Volume query: check if geometry blocks path
- Precomputed/baked: offline calculation stored per zone

**Propagation:**
- Direct path
- Early reflections (1st-3rd order)
- Diffraction (around corners)
- Transmission (through walls, with material-based frequency filtering)

**Materials:**
- Absorption coefficient (per frequency band)
- Reflection coefficient
- Transmission coefficient

### 6.5 Mix Bus Hierarchy
```
Master Bus
+-- SFX Bus
|   +-- Weapons Bus
|   +-- Footsteps Bus
|   +-- Environment Bus
|   +-- UI Bus
+-- Music Bus
|   +-- Stems Bus
|   +-- Stingers Bus
+-- Dialogue Bus
|   +-- Player VO Bus
|   +-- NPC VO Bus
+-- Ambient Bus
```

Each bus has: volume, pitch, mute, solo, low-pass, high-pass, effect chain.

**Mix Snapshots:** Named presets of all bus volumes (e.g., "combat", "cutscene", "underwater"). Blend between snapshots with crossfade time.

**HDR Audio:** Virtual dB scale where all sounds exist in a wide range, with an "audible window" that shifts dynamically. Loud events (explosion) push the window up, quieting soft sounds.

### 6.6 Adaptive Music
**Horizontal re-sequencing:** Change which section plays next based on game state.
**Vertical remixing:** Layer stems in/out based on intensity.
**Musical timing:** BPM, time signature, beat/bar sync, quantized transitions.
**Stingers:** One-shot musical accents triggered by game events.

### 6.7 DSP Effects
**Filters:** Low-pass, High-pass, Band-pass, Notch, Shelf, Parametric EQ
**Dynamics:** Compressor, Limiter, Expander, Gate (threshold, ratio, attack, release, knee)
**Time-based:** Delay (with feedback), Chorus, Flanger, Phaser
**Distortion:** Hard clip, Soft clip, Waveshaping, Bitcrush, Foldback
**Pitch/Time:** Simple pitch shift, Formant preservation, Granular, Phase vocoder

### 6.8 Dialogue
**VO Management:** Priority queue, interruption rules, overlap handling, localization support.
**Contextual:** Barks (short reactive lines), Conversations (multi-line exchanges), Ambient VO, Narration.
**Line Selection:** Random, Sequential, Cooldown, Conditional (based on game state).

### 6.9 Platform Audio APIs
- Windows: WASAPI, XAudio2
- macOS/iOS: Core Audio
- Linux: ALSA, PulseAudio
- PS5: Tempest 3D
- Xbox: XAudio
- Switch: NN Audio
- Android: AAudio/Oboe

---

## 7. TODO Checklist

From `GAME_ENGINE_INTEGRATION_TODO.md` section 8:

### 7.1 Audio Core
- [ ] Implement (or integrate) audio backend (platform audio APIs)
- [ ] Implement audio source/listener system
- [ ] Implement audio clip playback (one-shot, looping, streaming)
- [ ] Wire `@sound` decorator to audio source configuration
- [ ] Wire `@spatial_audio` decorator to 3D spatialization
- [ ] Register audio sources/listeners as Components via ComponentMeta

### 7.2 Mix Bus
- [ ] Implement mix bus hierarchy (master to submixes to voices)
- [ ] Implement volume, pitch, low-pass, high-pass per bus
- [ ] Wire `@audio_bus` decorator to bus assignment and routing
- [ ] Register mix buses as Resources via ResourceMeta

### 7.3 Spatial & Acoustic
- [ ] Implement HRTF-based 3D audio
- [ ] Implement reverb zones and acoustic simulation
- [ ] Implement occlusion/obstruction
- [ ] Implement distance attenuation curves

### 7.4 Adaptive Music
- [ ] Implement adaptive music system (layers, transitions, stingers)
- [ ] Implement music state machine
- [ ] Wire StateMeta to music state registration

### 7.5 DSP Effects
- [ ] Implement DSP effect chain (reverb, delay, EQ, compressor, chorus)
- [ ] Implement real-time parameter modulation

---

## 8. Directory Structure

```
engine/audio/
    __init__.py
    AUDIO_CONTEXT.md              <-- This file
    core/
        __init__.py
        audio_engine.py           # Main audio engine (Resource, uses EngineMeta)
        audio_source.py           # AudioSource component
        audio_listener.py         # AudioListener component
        audio_clip.py             # AudioClip asset type
        voice_manager.py          # Voice allocation, priority, stealing
        sound_cue.py              # Sound cue containers (random, sequence, switch)
    mixing/
        __init__.py
        mix_bus.py                # MixBus resource
        mix_snapshot.py           # MixSnapshot (audio_snapshot decorator)
        hdr_audio.py              # HDR audio with audible window
        sidechain.py              # Sidechain compression
    spatial/
        __init__.py
        spatialization.py         # Panning, HRTF, VBAP, Ambisonics
        attenuation.py            # Distance curves (linear, log, inverse, custom)
        occlusion.py              # Occlusion/obstruction (raycast, propagation, baked)
        reverb.py                 # Reverb zones (algorithmic, convolution, hybrid)
        propagation.py            # Sound propagation (reflections, diffraction)
    adaptive/
        __init__.py
        music_system.py           # Adaptive music engine
        music_stem.py             # Stem management
        music_state.py            # Music state machine (StateMeta)
        music_transition.py       # Transition rules
        stinger.py                # Musical stingers
    dsp/
        __init__.py
        dsp_graph.py              # DSP processing graph
        filters.py                # Low-pass, high-pass, band-pass, EQ
        dynamics.py               # Compressor, limiter, gate
        time_effects.py           # Delay, chorus, flanger, phaser
        distortion.py             # Clip, waveshape, bitcrush
        pitch.py                  # Pitch shift, formant, granular
    systems/
        __init__.py
        audio_update_system.py    # Main audio system (@system, phase="audio")
        spatial_update_system.py  # Spatial audio update
        music_update_system.py    # Music state machine update
        voice_cleanup_system.py   # Voice recycling
```

---

## 9. Canonical Usage Examples

### Basic Sound Source
```python
@component
@tracked
@sound(bank="weapons")
@spatial_audio(falloff="inverse", max_distance=80.0)
@voice_priority(priority=5, virtualize=True)
class GunfireSound(Component):
    volume: Annotated[float, Tracked(), Range(0.0, 1.0)] = 0.8
    pitch: Annotated[float, Tracked(), Range(0.5, 2.0)] = 1.0
    playing: Annotated[bool, Tracked()] = False
    _voice_handle: Annotated[int, Transient()] = -1
```

### Mix Bus Resource
```python
@resource
@audio_bus(name="sfx", volume=0.9, effects=["compressor", "eq"])
class SFXBus(Resource):
    volume: Annotated[float, Tracked(), Range(0.0, 1.0)] = 0.9
    muted: bool = False
```

### Reverb Zone
```python
@component
@reverb_zone(preset="cathedral", fade_distance=8.0)
class CathedralReverb(Component):
    wet_mix: Annotated[float, Range(0.0, 1.0)] = 0.6
```

### Adaptive Music
```python
@resource
@adaptive_audio(crossfade_time=1.0, stem_group="battle_music")
class BattleMusic(Resource):
    intensity: Annotated[float, Tracked(), Range(0.0, 1.0)] = 0.0
    
@state
class MusicState(State):
    _valid_transitions = {
        "explore": ["combat", "stealth"],
        "combat": ["explore", "victory"],
        "stealth": ["explore", "combat"],
    }
```

### Audio Update System
```python
@system(phase="audio")
@traced
class AudioUpdateSystem(System):
    def update(self, dt: float):
        # 1. Collect dirty audio sources from Tracker
        for source in tracker.all_dirty():
            if hasattr(source, '_sound'):
                dirty = tracker.dirty_fields(source)
                if 'playing' in dirty:
                    if source.playing:
                        self._play(source)
                    else:
                        self._stop(source)
                if 'volume' in dirty or 'pitch' in dirty:
                    self._update_params(source)
        
        # 2. Update spatial positions
        # 3. Process voice management
        # 4. Flush commands to audio thread
```

### Audio Event
```python
@event
class SoundFinished(Event):
    entity_id: int
    bank: str
    cue_name: str
```

### Audio Asset
```python
@asset(extensions=[".wav", ".ogg", ".mp3", ".opus"])
class AudioClip(Asset):
    sample_rate: int = 44100
    channels: int = 2
    duration: float = 0.0
    compressed: bool = False
    stream: bool = False
```

---

## 10. Key Integration Patterns

### Pattern: Game Thread to Audio Thread
The game thread writes to tracked components. The audio system reads dirty flags each frame and pushes commands to the audio thread via a lock-free command queue. The audio thread never reads game state directly.

```
Game Thread                    Audio Thread
    |                              |
    +-- source.volume = 0.5        |
    |   (TrackedDescriptor)        |
    |   (tracker.mark_dirty)       |
    |                              |
    +-- AudioUpdateSystem.update() |
    |   reads tracker.all_dirty()  |
    |   pushes SetVolume(id, 0.5)  |
    |   to command queue ----------+-- processes SetVolume
    |                              |   applies to voice
    |                              |   mixes output
```

### Pattern: Foundation EventLog for Replay
Audio events are recorded in EventLog so the replay system can re-trigger sounds at the correct frame. Only trigger commands are logged (not continuous DSP state).

### Pattern: Mix Bus as Resource
Mix buses are global singletons (ResourceMeta), meaning there is exactly one SFXBus, one MusicBus, etc. Their volumes are tracked, so changes propagate to the audio thread and persist in Session.

### Pattern: Music State Machine
Music states use StateMeta with validated transitions. The music system queries the current state each frame and drives stem volume/playback accordingly. State changes fire Events for EventLog.

---

## 11. Decorator Quick Reference

| Decorator | Tier | Metaclass | Registry | Key Params |
|-----------|------|-----------|----------|------------|
| @sound | 14 | ComponentMeta | audio | bank, preload |
| @audio_bus | 14 | ResourceMeta | audio | name, volume, effects |
| @spatial_audio | 14 | ComponentMeta | audio | falloff, max_distance |
| @dsp_node | 49 | ComponentMeta | audio_extended | inputs, outputs, latency_samples |
| @voice_priority | 49 | ComponentMeta | audio_extended | priority, virtualize, steal_oldest |
| @occlusion | 49 | ComponentMeta | audio_extended | method, max_occlusion |
| @reverb_zone | 49 | ComponentMeta | audio_extended | preset, fade_distance |
| @music_stem | 49 | ResourceMeta | audio_extended | group, layer, sync_to_beat |
| @music_transition | 49 | ResourceMeta | audio_extended | from_state, to_state, type, duration_beats |
| @audio_snapshot | 49 | ResourceMeta | audio_extended | bus_overrides, crossfade_time |
| @sidechain | 49 | ResourceMeta | audio_extended | source_bus, attack, release, ratio |
| @adaptive_audio | stack | ResourceMeta | audio_extended | crossfade_time, stem_group |

## 12. Descriptor Quick Reference

| Descriptor | Use in Audio | Foundation System |
|-----------|-------------|------------------|
| TrackedDescriptor | Volume, pitch, playing state - dirty flags | Tracker |
| ValidatedDescriptor | Volume 0-1, pitch bounds, distance >0 | (internal) |
| RangeDescriptor | Numeric parameter clamping | (internal) |
| ObservableDescriptor | UI/inspector callbacks on param change | Tracker |
| NetworkedDescriptor | Replicate sound_id, playing for multiplayer | Tracker |
| SerializableDescriptor | Bus config, music state save/load | Mirror |
| TransientDescriptor | Voice handles, decode positions (NOT saved) | Mirror |
| ProfiledDescriptor | Audio perf monitoring | EventLog |
