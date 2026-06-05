# Phase 2 Architecture: Mixer Graph and Voice Management

## Purpose
Central mixing infrastructure: hierarchical bus tree, voice pool with priority-based allocation, virtual voice system, sidechain processing, ducking, HDR audio, and snapshot-based mix management.

## Current Implementation
**17/18 tasks complete [x], 1 partial [~].** The `mixing/` subsystem (10 files, ~5,495 lines) was entirely missed by the original TODO.

### Mix Bus Hierarchy (`mixing/mix_bus.py`, 763 lines) [x]
- `MixBus` with `BusType` enum (MASTER, CATEGORY, SUB, AUX)
- `FilterState`: low-pass, high-pass with configurable Q
- `BusState`: volume (linear + dB), pitch, mute, solo, effects chain
- Parent-child hierarchy via `add_child()`/`remove_child()`
- `create_default_hierarchy()`: Master -> SFX/Music/VO/Ambient/UI + sub-buses
- Thread-safe with lock for bus state access

### Bus Routing (`mixing/bus_routing.py`, 491 lines) [x]
- `BusRouter`: voice-to-bus assignment, routing map
- `AuxSend`: configurable send level, pre/post fader
- `DirectOutput`: direct routing bypassing main mix
- `RoutingMode`: SEND, PARALLEL, REPLACE

### Mix Snapshots (`mixing/mix_snapshot.py`, 675 lines) [x]
- `MixSnapshot`: named preset with per-bus volume/pitch/mute/solo/filter snapshots
- `BusSnapshot`: individual bus state capture
- `SnapshotManager`: priority-based layering, `apply(snapshot, blend)`
- Interpolation curves: linear, s-curve, exponential, logarithmic
- Blend-in/blend-out state machine

### HDR Audio (`mixing/hdr_audio.py`, 548 lines) [x]
- `HDRAudioManager`: virtual dB scale (VDB)
- `MixWindow`: dynamic audible window
- `HDRPriority`: CRITICAL, HIGH, NORMAL, LOW
- Adaptation speed, ceiling/floor configuration
- Priority-based importance calculation, loudness analysis
- `update()` per-frame window shifting

### Ducking (`mixing/ducking.py`, 674 lines) [x]
- `DuckingManager`: multi-source ducking control
- `DuckType`: DIALOGUE, EVENT, FOCUS
- Per-bus ducking with configurable depth/attack/release
- Ducking bus assignments

### Sidechain (`mixing/sidechain.py`, 499 lines) [x]
- `SidechainCompressor`: `KeySource` selection, envelope follower, configurable threshold/ratio/attack/release/knee
- `SidechainManager`: multiple compressor instances with bus routing
- `update()` per-frame gain reduction computation

### Mixer (`mixing/mixer.py`, 1101 lines) [x]
- `Mixer`: central coordinator integrating all mixing subsystems
- `MixerConfig`: sample rate, buffer size, thread model
- `update(dt)`: per-frame processing of buses, ducking, sidechain, HDR, snapshots
- Thread-safe with `RLock`

### Voice Management (`core/voice_manager.py`, 657 lines) [x]
- `Voice` dataclass: state, source, bus, priority, fade state
- Heap-based free/active voice tracking
- `VoiceStealStrategy`: OLDEST, QUIETEST, LOWEST_PRIORITY
- `steal_voice()`: min-heap ordering, `VOICE_STEAL_FADE_MS` for smooth fade-out
- `CATEGORY_VOICE_LIMITS`, `MAX_INSTANCES_PER_SOUND`

### Virtual Voices (`core/virtual_voice.py`, 293 lines) [x]
- `VirtualVoiceManager`: `make_virtual()`/`make_real()`
- `VIRTUAL_VOICE_MAX_TIME_SECONDS` configurable timeout
- Urgency-based prioritization, timeout-based release
- Virtual voice tracks position/time for seamless resume

### Decorators [x]
- `@audio_bus` in `trinity/decorators/audio.py`: validates name/volume/effects, TAG+REGISTER ops
- `@audio_snapshot` in `trinity/decorators/audio_extended.py`: validates params, TAG+REGISTER
- `@voice_priority` in `trinity/decorators/audio_extended.py`: priority/virtualize/steal params
- `@sidechain` in `trinity/decorators/audio_extended.py`: source_bus/attack/release/ratio params
- `VoicePriorityBridge` in `core/voice_priority_bridge.py` (224 lines): bridges decorator to VM

### Architecture
```
Bus Hierarchy:
  Master
  +-- SFX (48 voices)
  |   +-- Weapons
  |   +-- Footsteps
  |   +-- Impacts
  +-- Music (8 voices)
  +-- VO (8 voices)
  +-- Ambient (16 voices)
  +-- UI (16 voices)

Voice Pipeline:
  acquire voice -> assign to bus -> priority sort -> 
  if full: steal lowest priority / virtualize ->
  per-frame: fade in/out stolen -> update volume -> accumulator -> bus effects

Snapshot System:
  Snapshot A --50% blend--> Snapshot B --30%--> Snapshot C
  Each snapshot stores per-bus: volume, mute, solo, filter, effects
  Blending interpolates all parameters with configurable curve
```

### Missing (1 task)
| Task | Component | Gap |
|------|-----------|-----|
| T-AU-2.18 | Session persistence | SnapshotManager stores/recalls but not wired to Foundation Session |
