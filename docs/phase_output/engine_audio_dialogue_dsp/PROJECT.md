# PROJECT: engine_audio_dialogue_dsp

**Scope, Goals, and Constraints**
**Generated:** 2026-05-23

---

## 1. Project Identity

**Name:** Audio Dialogue and DSP Subsystems
**Domain:** Game Audio Processing
**Location:** `engine/audio/dialogue/` and `engine/audio/dsp/`
**Total Lines:** ~12,194 Python
**Status:** REAL IMPLEMENTATION (production-quality)

---

## 2. Scope

### 2.1 In Scope

#### Dialogue Subsystem
- Voice-over playback with priority queue scheduling
- Branching conversation management with state machines
- Contextual dialogue (barks, ambient VO) with cooldowns
- Multi-language localization with fallback chains
- Subtitle synchronization with display management
- Spatial audio processing for VO (3D pan, distance attenuation)

#### DSP Subsystem
- Filter implementations (biquad, SVF, parametric EQ)
- Dynamics processing (compressor, limiter, gate, expander)
- Time-based effects (delay, chorus, flanger, phaser)
- Algorithmic reverb (Freeverb, plate)
- Convolution reverb with FFT
- Distortion algorithms (tube, tape, bitcrusher)
- Pitch/time manipulation via granular synthesis
- Game-specific effects (radio, underwater, explosion)

### 2.2 Out of Scope

- Audio file decoding/encoding (assumed handled by resource layer)
- Platform-specific audio API bindings (Vulkan Audio, WASAPI, CoreAudio)
- Hardware DSP offloading
- MIDI/instrument synthesis
- Music composition/sequencing
- Procedural sound generation (beyond granular synthesis)

---

## 3. Goals

### 3.1 Primary Goals

| Goal | Description | Status |
|------|-------------|--------|
| Production Audio DSP | Real-time audio processing with correct mathematics | ACHIEVED |
| Thread-Safe Processing | Concurrent parameter updates without glitches | ACHIEVED |
| Block Processing | Efficient batch operations with NumPy | ACHIEVED |
| Modular Architecture | Composable DSP graph for effect chains | ACHIEVED |
| Game Dialogue System | Complete VO pipeline from queue to playback | ACHIEVED |
| Localization Support | Multi-language audio with fallback | ACHIEVED |

### 3.2 Quality Goals

| Goal | Metric | Status |
|------|--------|--------|
| No Stubs | Zero `NotImplementedError`, zero `pass` in processing | ACHIEVED |
| Correct Math | Algorithms match published DSP literature | ACHIEVED |
| SIMD-Ready | 32-byte aligned buffers for AVX | ACHIEVED |
| Low Latency | Block-based processing without per-sample allocation | ACHIEVED |

---

## 4. Constraints

### 4.1 Technical Constraints

| Constraint | Impact |
|------------|--------|
| Python 3.13 target | Must use uv runtime, not system Python 3.14 |
| NumPy dependency | Block processing relies on NumPy arrays |
| No GPU DSP | All processing is CPU-bound (no compute shaders) |
| Threading model | RLock-based synchronization, not lock-free |

### 4.2 Architectural Constraints

| Constraint | Rationale |
|------------|-----------|
| DSPNode base class | All processors inherit common interface |
| SmoothedParameter | Parameter changes must avoid zipper noise |
| Sample rate awareness | Coefficients recalculate on sample rate change |
| State management | Each processor maintains per-channel state |

### 4.3 Integration Constraints

| Constraint | Dependency |
|------------|------------|
| Audio source | Requires `engine/audio/core/audio_source.py` |
| Configuration | Requires `engine/audio/*/config.py` |
| VO Line | Requires `engine/audio/dialogue/vo_line.py` data structure |

---

## 5. Architecture Principles

### 5.1 Separation of Concerns

- **Dialogue** manages what to play and when
- **DSP** manages how audio is processed
- **Core** (not in this scope) manages audio device I/O

### 5.2 Composition Over Inheritance

- DSPChain composes processors in series
- DSPParallel composes processors in parallel
- DSPGraph allows arbitrary routing

### 5.3 Immutable Processing

- Input buffers are not modified
- Output buffers are newly allocated or pre-allocated
- State is encapsulated per processor instance

---

## 6. Success Criteria

| Criterion | Evidence |
|-----------|----------|
| All files are REAL | No stubs found in any of 22 files |
| Algorithms are correct | Bilinear transform matches literature, Freeverb delays match original |
| Thread-safe | RLock used throughout managers |
| Block-efficient | NumPy operations, pre-allocated buffers |
| Complete dialogue pipeline | Queue -> Stream -> Process -> Output with localization |
| Complete DSP suite | Filters, dynamics, time, reverb, distortion, pitch |
