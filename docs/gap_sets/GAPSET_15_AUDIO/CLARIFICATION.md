# GAPSET_15_AUDIO — Clarification Questions (UPDATED 2026-05-22)

> **Generated**: 2026-05-22 during RDC
> **Context**: Codebase is Python; original TODO was written for Rust/SDL. Several questions are now resolved by discovered code.
> **Updated**: Added resolved/status indicators.

---

## RESOLVED — Ambiguities Answered by Codebase Discovery

### 1. Language Mismatch: Python vs Rust (RESOLVED)

The entire `engine/audio/` codebase is Python (~18,339 lines). The TODO described Rust traits. However, **decorator files** `trinity/decorators/audio.py` and `audio_extended.py` (679 lines) are fully implemented using the ops-based `make_decorator` system, meaning the Python codebase is the real implementation.

**Resolution**: The codebase is Python. The audio engine will remain Python until a port to Rust (C FFI or PyO3) is explicitly needed. The `@sound`, `@audio_bus`, `@spatial_audio`, `@dsp_node`, `@voice_priority`, `@occlusion`, `@reverb_zone`, `@music_stem`, `@music_transition`, `@audio_snapshot`, and `@sidechain` decorators are ALL implemented and registered with the decorator registry.

### 2. Foundation Framework Dependency (PARTIALLY RESOLVED)

Tasks reference Foundation ComponentMeta, StateMeta, ResourceMeta, SystemMeta, etc. The decorator files implement the `@sound`, `@audio_bus`, `@spatial_audio`, etc. decorators using `make_decorator` with TAG and REGISTER ops. These register with `Tier.AUDIO` and `Tier.AUDIO_EXTENDED` in the decorator registry at `trinity/decorators/registry.py`.

**Resolution**: The audio-specific decorators (11 total) are implemented and registered. However:
- `@tracked`/TrackedDescriptor — NOT found. Foundation Tracker system may not exist yet.
- `@system`/SystemMeta — NOT found. Audio systems are plain Python classes with `update()` methods.
- `@state`/StateMeta — NOT found. Music states are Python dataclasses with manual validation.
- `@adaptive_audio` composite stack — NOT implemented.

These 3 decorators and 1 composite stack remain blocked until Foundation is available or the architecture is changed.

### 3. Rust math.rs / memory.rs Dependencies (RESOLVED)

**Resolution**: The Python codebase has its own `Vector3` in `core/audio_listener.py` with full normalize/dot/cross/distance/lerp. The spatial subsystem uses `engine.core.math.vec.Vec3` (from omega math layer). No Foundation math dependency is needed.

For lock-free structures: The codebase uses Python `threading.Lock`/`RLock`/`queue.Queue` throughout. No lock-free ring buffers exist. This is the biggest performance gap.

---

## RESOLVED — Design Questions Answered by Codebase

### 4. Platform Backend Strategy (UNRESOLVED — design decision needed)

Tasks T-AU-1.5 through T-AU-1.8. Still need a decision:
- (a) Rust crates with C FFI to Python?
- (b) Python ctypes bindings to platform APIs?
- (c) Via an existing cross-platform library (e.g., `sounddevice`, `PortAudio`)?

### 5. Format Decoder Architecture (UNRESOLVED — design decision needed)

Tasks T-AU-3.5 through T-AU-3.9. Still need a decision:
- (a) Python bindings to native decoders (libvorbis, libflac, mpg123, libopus)?
- (b) Pure-Rust decoders via Symphonia?
- (c) System-provided codecs (Windows Media Foundation, Android MediaCodec)?

### 6. Dialogue Bank Format (PARTIALLY RESOLVED)

**Resolution**: `AudioBank` and `LocalizedAsset` dataclasses exist in `dialogue/localization.py`. `VOLine.to_dict()`/`from_dict()` serialize to/from dict. No standardized manifest format. The question is whether to:
- (a) Design a new schema
- (b) Follow FMOD/Wwise bank format
- (c) Use the existing `to_dict()`/`from_dict()` serialization

---

## RESOLVED — Missing Details Answered by Codebase

### 7. Third-Party Reverb IRs (RESOLVED)

**Resolution**: `ConvolutionReverb` in `dsp/reverb.py` supports IR loading. `REVERB_PRESETS` (Room/Hall/Church/Plate/Spring/Cave/Arena/Outdoors) are algorithmic presets for `SimpleReverb`, not convolution IRs. IR files are not bundled. Standard IR packs (OpenAIR, EchoThief) can be used when available.

### 8. HRTF Dataset (PARTIALLY RESOLVED)

**Resolution**: `HRTFSpatializer` in `spatial/hrtf.py` uses analytic HRTF based on: ITD via Woodworth's spherical head model, ILD via frequency-dependent gain. `HRTFProfile` can accept measured data but no SOFA file loader exists. Options: CIPIC (35 subjects), MIT Kemar, SADIE II, or continue with analytic model.

### 9. Acoustic Material Presets (RESOLVED)

**Resolution**: 12 material types are already defined in `spatial/materials.py`:
- CONCRETE, WOOD, GLASS, CARPET, CURTAIN, BRICK, METAL, WATER, PLASTER, ACOUSTIC_TILE, SOIL, GRASS
- Each has absorption/reflection/scattering/transmission coefficients across 6 frequency bands (125Hz, 250Hz, 500Hz, 1kHz, 2kHz, 4kHz)
- MaterialDatabase with get/create_custom operations

### 10. Beat Clock Jitter Tolerance (RESOLVED)

**Resolution**: `MusicClock` in `adaptive/music_timing.py` uses Python `time.monotonic()` with beat quantization. At 120 BPM (500ms beat), Python scheduling jitter is ~1-16ms depending on system load. The quantization ensures transitions land on exact beat/bar boundaries regardless of jitter. Frame-accurate (16ms at 60fps) is sufficient for game audio.

---

## NEW — Questions Arising from RDC

### 11. `mixing/` and `spatial/` Integration

The `mixing/` subsystem (5495 lines, 10 files) and `spatial/` subsystem (6641 lines, 11 files) were not referenced in the original TODO. They are fully implemented and integrated with each other (Mixer imports spatial types, BusRouter integrates with MixBus). However:

- **Question**: Are there unused files or dead code in these subsystems that should be pruned? The original RDC did not verify every function call chain.

### 12. NumPy Dependency

Many mixing and spatial files import `numpy` (e.g., `import numpy as np` for array operations, FFT in reverb).

- **Question**: Is NumPy an acceptable runtime dependency, or should DSP/spatial operations be ported to pure Python or Rust? For production, NumPy is fast but adds a dependency.

### 13. Test Coverage

Tests exist only for `tests/engine/audio/core/` (virtual voice, voice priority bridge) and `tests/engine/audio/mixing/` (sidechain bridge). No tests for DSP, spatial, adaptive music, or dialogue subsystems.

- **Question**: Prioritize writing tests for DSP (most critical for correctness) or spatial (most complex)?

### 14. Decoration Pipeline Status

The `@dsp_node` decorator is fully implemented in `audio_extended.py` and registers with `Tier.AUDIO_EXTENDED`. But the original TODO marked T-AU-7.2 as "Wire `@dsp_node`" with a [~] status.

- **Question**: What is the expected behavior of `@dsp_node` that isn't yet working? The existing implementation validates params and registers the class. Is the missing piece the actual pipeline that processes `_dsp_node`-tagged classes and connects them to `DSPChain`/`DSPGraph`?

---

## Updated Summary of Blocked Tasks

| Tasks | Blocked By | Resolution Needed |
|-------|-----------|-------------------|
| T-AU-1.5-1.8 | No platform backends | Architecture decision (Q4) |
| T-AU-1.10-1.11 | No lock-free ring buffers | Implement SPSC/MPSC in Python or Rust |
| T-AU-3.3, 3.11, 8.3 | Foundation @tracked/@system/@state | Find Foundation or redesign without |
| T-AU-3.5-3.9 | No format decoders | Architecture decision (Q5) |
| T-AU-5.7 | No SOFA HRTF loader | Dataset + loader decision (Q8) |
| T-AU-8.11 | No composite decorator stack | Implement @adaptive_audio |
| T-AU-8.13 | No music state persistence | Wire to Session |
| T-AU-9.9 | Animation system integration | Blend shape driver spec |
