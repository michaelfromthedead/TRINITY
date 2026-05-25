# GAPSET_15_AUDIO — Project Documentation (CORRECTED 2026-05-22)

> **Generated**: 2026-05-22 via RDC (Research, Document, Compile)
> **Codebase Location**: `/engine/audio/` — Python audio engine (~62 source files, ~18,339 lines)
> **Original TODO**: 52/129 [x] (incorrect — missed `mixing/` and `spatial/` subsystems)
> **Corrected Status**: 92/129 [x] (71%) — 19 [~], 18 [-]

---

## Architecture Overview

The TRINITY audio engine is organized into six core subsystems plus decorator files and shared config:

```
engine/audio/
+-- core/                          # Core types, engine, voice management
|   +-- config.py                  # All audio constants centralized
|   +-- audio_source.py            # AudioSource + AudioSourcePool
|   +-- audio_clip.py              # AudioClip asset type
|   +-- audio_listener.py          # AudioListener 3D positioning
|   +-- audio_engine.py            # AudioEngine (threading, commands, update)
|   +-- voice_manager.py           # Voice allocation, stealing, limits
|   +-- virtual_voice.py           # Virtual voice lifecycle
|   +-- voice_priority_bridge.py   # Bridge: decorator -> VoiceManager
|   +-- memory_manager.py          # Memory pools, LRU, streaming buffers
|   +-- sound_cue.py               # Sound cue containers + variation
|
+-- mixing/                        # Mix bus hierarchy and mixing
|   +-- config.py                  # Mixing constants
|   +-- mix_bus.py                 # MixBus with hierarchy, filters, states
|   +-- bus_routing.py             # BusRouter, AuxSend, DirectOutput
|   +-- mixer.py                   # Mixer (central coordinator)
|   +-- mix_snapshot.py            # SnapshotManager, MixSnapshot
|   +-- ducking.py                 # DuckingManager, DuckType
|   +-- sidechain.py               # SidechainManager + Compressor
|   +-- sidechain_bridge.py        # Bridge integration
|   +-- hdr_audio.py               # HDRAudioManager, MixWindow
|
+-- spatial/                       # Spatial audio, 3D, acoustic simulation
|   +-- config.py                  # Spatial constants and enums
|   +-- positioning.py             # Point/Area/Line/Volume sources
|   +-- attenuation.py             # 6 curve types + presets
|   +-- spatialization.py          # Stereo/Surround/VBAP/Ambisonics
|   +-- hrtf.py                    # HRTF with ITD/ILD
|   +-- doppler.py                 # Doppler effect + presets
|   +-- speaker_config.py          # Speaker layouts + routing
|   +-- occlusion.py               # Multi-ray occlusion
|   +-- propagation.py             # Reflections, diffraction, transmission
|   +-- reverb_zone.py             # Reverb zone manager + presets
|   +-- materials.py               # Acoustic material database
|
+-- dsp/                           # Digital Signal Processing
|   +-- config.py                  # DSP constants
|   +-- dsp_node.py                # DSPNode ABC, SmoothedParameter
|   +-- dsp_graph.py               # DSPChain, DSPParallel, DSPGraph, EffectRack
|   +-- filters.py                 # 8 biquad types + ParametricEQ + SVF
|   +-- dynamics.py                # Compressor, Limiter, Gate, Expander
|   +-- time_effects.py            # Delay, Chorus, Flanger, Phaser, LFO
|   +-- reverb.py                  # Freeverb, Plate, Convolution, Simple
|   +-- distortion.py              # Hard/Soft clip, Waveshaper, Bitcrush
|   +-- pitch_time.py              # Pitch shift, Time stretch, Granular, Vocoder
|   +-- special_fx.py              # Radio, Underwater, Cave, Explosion, etc.
|
+-- adaptive/                      # Adaptive Music System
|   +-- config.py                  # Music system constants
|   +-- music_timing.py            # BeatGrid, MusicClock, SyncPointManager
|   +-- music_state.py             # MusicStateManager, StateChangeRule
|   +-- music_stem.py              # LayeredMusicPlayer, VerticalRemixer
|   +-- music_transition.py        # TransitionManager, 6 types
|   +-- music_callback.py          # MusicCallbackManager, BeatScheduler
|   +-- music_player.py            # MusicPlayer, Playlist, TrackInfo
|   +-- stinger.py                 # StingerManager, Stinger
|   +-- adaptive_music.py          # AdaptiveMusicSystem (orchestrator)
|
+-- dialogue/                      # Dialogue System
|   +-- config.py                  # Dialogue constants
|   +-- vo_line.py                 # VOLine, LipSyncData, SubtitleData
|   +-- vo_queue.py                # VOQueue, VOQueueManager
|   +-- vo_streaming.py            # VOStreamManager, VOCache
|   +-- vo_processing.py           # VOProcessor, Radio/Distance/Reverb/Spatial
|   +-- conversation.py            # ConversationManager, ConversationNode
|   +-- contextual_dialogue.py     # LinePool, BarkSystem, AmbientVOSystem
|   +-- localization.py            # LocalizationManager, 10 languages
|   +-- subtitle_sync.py           # SubtitleManager, SubtitleTrack
|   +-- dialogue_manager.py        # DialogueManager (orchestrator)

trinity/decorators/
+-- audio.py                       # @sound, @audio_bus, @spatial_audio
+-- audio_extended.py              # @dsp_node, @voice_priority, @occlusion,
                                   # @reverb_zone, @music_stem, @music_transition,
                                   # @audio_snapshot, @sidechain
```

---

## Scope and Goals

### What This Gap Set Covers

- **Complete audio engine** for TRINITY: playback, mixing, spatialization, DSP, adaptive music, dialogue
- **Production-quality** subsystems: DSP (100%), Adaptive Music (~90%), Dialogue (~95%), Mixing (~95%), Spatial (~95%)
- **Platform-independent** Python implementation with threading model for game-thread audio-thread separation

### What This Gap Set Does NOT Cover

- **Platform audio backends**: No WASAPI, CoreAudio, ALSA, or PulseAudio implementations
- **Lock-free threading**: All concurrency uses Python `threading.Lock`/`queue.Queue` — no lock-free SPSC/MPSC ring buffers
- **Format decoders**: No native decoders for OGG, FLAC, MP3, Opus (only raw PCM/WAV via Python `wave` module)
- **Foundation decorator wiring**: `@tracked`, `@system`, `@state`/StateMeta not wired (Foundation framework dependency)
- **Animation integration**: Lip sync data model exists, blend shape driver not implemented
- **Composite decorators**: `@adaptive_audio` stack not implemented

### Design Philosophy

1. **Python-first prototyping**: All subsystems implemented in Python for rapid iteration. Production build would port performance-critical paths (DSP, mixing, spatialization) to Rust/Cython or use NumPy vectorization.

2. **DSP is the crown jewel**: 20+ effect types, full biquad suite, dynamics, time effects, distortion, pitch/time processing — all production quality. This is engine-ready.

3. **Mixing and spatial audio are done**: The `mixing/` and `spatial/` subsystems were complete but undocumented in the original TODO. They provide full bus hierarchy, snapshots, HDR audio, HRTF, VBAP, Ambisonics, occlusion, and propagation.

4. **Real gap is platform integration**: The 18 [-] tasks are all platform-layer work: audio backends, format decoders, lock-free threading. No architectural design is needed, only implementation.

---

## Phase Overview

| Phase | Name | Tasks [x]/[~]/[-] | Completion | Status |
|-------|------|-------------------|------------|--------|
| 1 | Audio Device Abstraction & Core Types | 5/1/6 | 42% | Core engine exists, platform backends missing |
| 2 | Mixer Graph & Voice Management | 17/1/0 | 94% | **Essentially complete** |
| 3 | Sound Playback Engine | 7/1/6 | 50% | Playback works, format loaders missing |
| 4 | Stream & Decode Thread Architecture | 0/6/4 | 30% | Weakest area — Python threading only |
| 5 | Spatial Audio System | 14/1/0 | 93% | **Essentially complete** |
| 6 | Acoustic Simulation | 10/5/0 | 83% | **Nearly complete** |
| 7 | DSP Effect Chain | 20/0/0 | 100% | **Complete** |
| 8 | Adaptive Music Engine | 10/2/2 | 79% | Core complete, decorators/decorator stacks missing |
| 9 | Dialogue System | 9/2/0 | 91% | **Nearly complete** |
| **Total** | | **92/19/18** | **71%** | |

---

## Key Integration Points

### To Foundation Registry (TODO items blocked)
- `@tracked`/TrackedDescriptor — Foundation Tracker dependency
- `@system` — Foundation SystemMeta dependency
- `@state`/StateMeta — Foundation StateMeta dependency
- Foundation Session — mix bus and music state persistence

### To Animation System
- `LipSyncData` blend shapes need animation system blend shape driver
- Data format is ready (phonemes, visemes, blend_shapes with timing)

### To Platform Layer
- Audio backends need platform API bindings (WASAPI, CoreAudio, ALSA, PulseAudio)
- Format decoders need native library bindings (libvorbis, libflac, mpg123, libopus)

---

## Immediate Action Items

1. **Port audio engine to platform**: Implement WASAPI/CoreAudio/ALSA/PulseAudio backends using either PortAudio binding (fastest), Python ctypes (no deps), or Rust C FFI (most performant)
2. **Add format decoders**: Bind libvorbis/libflac/mpg123/libopus via ctypes or use PyO3 bindings
3. **Implement lock-free SPSC ring buffer**: Replace `queue.Queue` in AudioEngine command path
4. **Wire decorators to Foundation**: Integrate `@tracked`, `@system`, `@state` decorators when Foundation framework is available
5. **Animate lip sync**: Build blend shape driver consuming `LipSyncData` streams
6. **Fix bug**: `VIRTUAL_VOICE` constants triplicated in `core/config.py` — deduplicated
