# Archaeological Investigation: engine/audio/dialogue + engine/audio/dsp

**Investigation Date**: 2026-05-22  
**Investigator**: Research Agent (Opus 4.5)  
**Total Lines**: ~12,194 (dialogue: 5,433 + dsp: 6,761)

---

## Executive Summary

**CLASSIFICATION: REAL IMPLEMENTATION**

Both `engine/audio/dialogue` and `engine/audio/dsp` subdirectories contain **genuine, production-quality audio DSP implementations**. These are NOT stubs. The code demonstrates:

1. **Mathematically correct DSP algorithms** (biquad filters, compressors, reverb, granular synthesis)
2. **Proper sample-by-sample and block-based processing** with state management
3. **Real coefficient calculations** using established audio engineering formulas
4. **Thread-safe implementations** with proper locking patterns
5. **NumPy-based efficient buffer processing** with SIMD alignment considerations

---

## Classification Summary

### engine/audio/dialogue (5,433 lines) - **REAL**

| File | Lines | Classification | Evidence |
|------|-------|----------------|----------|
| contextual_dialogue.py | 774 | REAL | Full bark/ambient system with cooldown, weighted selection |
| vo_processing.py | 728 | REAL | Radio effect, distance filtering, spatial audio calculations |
| conversation.py | 725 | REAL | Branching dialogue nodes, conversation state machine |
| dialogue_manager.py | 710 | REAL | Central orchestrator integrating all dialogue subsystems |
| vo_streaming.py | 707 | REAL | LRU cache, streaming, preloading with memory budgeting |
| subtitle_sync.py | 636 | REAL | Timed cues, fade states, reading speed calculations |
| localization.py | 580 | REAL | Audio banks, language switching, fallback chains |
| vo_queue.py | 573 | REAL | Priority heap queue with interrupt handling |

### engine/audio/dsp (6,761 lines) - **REAL**

| File | Lines | Classification | Evidence |
|------|-------|----------------|----------|
| dynamics.py | 1350 | REAL | Compressor, limiter, gate, expander, sidechain with proper math |
| filters.py | 972 | REAL | Biquad filters, SVF, parametric EQ with coefficient calculation |
| time_effects.py | 971 | REAL | Delay, chorus, flanger, phaser, vibrato with interpolation |
| reverb.py | 855 | REAL | Freeverb (Schroeder), plate reverb, convolution reverb |
| dsp_graph.py | 760 | REAL | Effect chains, parallel routing, topological sort |
| special_fx.py | 740 | REAL | Radio, underwater, explosion tinnitus, game-specific effects |
| pitch_time.py | 579 | REAL | Granular pitch shifting, time stretching with overlap-add |

---

## Key Algorithms Found

### Dynamics Processing (dynamics.py)

**Envelope Follower** - Real attack/release coefficient calculation:
```python
attack_samples = ms_to_samples(self._attack_ms, sr)
self._attack_coeff = math.exp(-1.0 / attack_samples)
```

**Compressor with Soft Knee** - Proper gain curve calculation:
```python
# Soft knee interpolation in dB domain
half_knee = knee / 2.0
knee_start = threshold - half_knee
knee_end = threshold + half_knee
if input_db < knee_start:
    return 0.0
elif input_db > knee_end:
    return (threshold - input_db) * (1.0 - 1.0 / ratio)
else:
    x = input_db - knee_start
    return (1.0 / ratio - 1.0) * (x * x) / (2.0 * knee)
```

**Lookahead Limiter** - Peak detection with delay compensation implemented.

**Sidechain Compressor** - Full implementation with external key signal routing.

### Filters (filters.py)

**Biquad Filter** - Complete bilinear transform implementation for all filter types:
```python
# Example: Low-pass coefficient calculation
omega = 2.0 * math.pi * freq / sr
sin_omega = math.sin(omega)
cos_omega = math.cos(omega)
alpha = sin_omega / (2.0 * q)

b0 = (1.0 - cos_omega) / 2.0
b1 = 1.0 - cos_omega
b2 = (1.0 - cos_omega) / 2.0
a0 = 1.0 + alpha
a1 = -2.0 * cos_omega
a2 = 1.0 - alpha
```

**State Variable Filter** - Numerically stable implementation with simultaneous LP/HP/BP/notch outputs.

**Parametric EQ** - Multi-band cascaded biquad with frequency response calculation.

### Reverb (reverb.py)

**Freeverb (Schroeder)** - 8 parallel comb filters + 4 series allpass:
- Comb filter delays: [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617]
- Allpass delays: [556, 441, 341, 225]
- Stereo spread: 23 samples

**Plate Reverb** - Diffusion network with tank delays and damping.

**Convolution Reverb** - FFT-based overlap-add implementation:
```python
input_fft = np.fft.rfft(input_padded)
output_fft = input_fft * self._ir_fft
conv_output = np.fft.irfft(output_fft)
# Overlap-add to output buffer
```

### Time Effects (time_effects.py)

**Delay Line** - Cubic (Hermite) interpolation for fractional delays:
```python
# Hermite interpolation
c0 = y1
c1 = 0.5 * (y2 - y0)
c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3
c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2)
return ((c3 * frac + c2) * frac + c1) * frac + c0
```

**LFO** - Multiple waveforms (sine, triangle, square, saw, random) with phase accumulation.

**Chorus** - Multi-voice modulated delay with phase-spread LFOs.

**Phaser** - Cascaded allpass filters with LFO-modulated center frequency.

### Pitch/Time Manipulation (pitch_time.py)

**Granular Pitch Shifter** - Hann window, overlap-add, resampling within grains:
```python
def _apply_hann_window(self, grain: np.ndarray) -> np.ndarray:
    n = len(grain)
    window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / n))
    return grain * window

def _resample_grain(self, grain: np.ndarray, ratio: float) -> np.ndarray:
    # Linear interpolation resampling
```

**Time Stretcher** - Independent duration control with grain synchronization.

### Dialogue System (dialogue/ directory)

**Priority Queue** - Heap-based with timeout expiration:
```python
@dataclass(order=True)
class QueueEntry:
    sort_key: tuple[int, float] = field(compare=True)  # (neg_priority, time)
    line: VOLine = field(compare=False)
```

**Conversation State Machine** - Full branching dialogue with callbacks:
- INACTIVE -> STARTING -> ACTIVE -> WAITING (for input) -> COMPLETED
- Branch point handling with choice selection

**VO Streaming** - LRU cache with:
- Hit/miss tracking
- Eviction threshold
- Preload queue management
- Memory budgeting

**Spatial Audio** - 3D positioning with pan calculation:
```python
def calculate_pan(self, listener_position, listener_forward) -> float:
    source_angle = math.atan2(dz, dx)
    relative_angle = source_angle - forward_angle
    pan = math.sin(relative_angle)
    return max(-1.0, min(1.0, pan)) * self.blend
```

---

## Implementation Quality Markers

### Real DSP Indicators

1. **Coefficient Calculations** - All filters use proper bilinear transform formulas
2. **State Persistence** - Per-channel state arrays (z1, z2 for biquads)
3. **Sample Rate Handling** - Coefficient recalculation on sample rate change
4. **Block Processing** - Efficient NumPy operations for batch processing
5. **Thread Safety** - RLock usage in managers
6. **SIMD Alignment** - Buffer allocation with alignment considerations

### Code Architecture

- **DSPNode Base Class** - Common interface with process_sample/process_block
- **Smoothed Parameters** - Parameter interpolation to avoid zipper noise
- **Reset Handling** - Proper state clearing on sample rate changes
- **Latency Tracking** - Lookahead latency reported for compensation

---

## Dependencies

### Internal
- `dsp_node.py` - Base DSP node class (not read but referenced)
- `config.py` - Constants and utility functions
- `vo_line.py` - Voice-over line data structure
- `distortion.py` - Distortion effect (used by special_fx.py)

### External
- `numpy` - Array processing and FFT
- `math` - Trigonometric functions
- `threading` - Synchronization primitives
- `dataclasses` - Data structures
- `enum` - Type enumerations
- `uuid` - Unique identifiers

---

## No Stub Indicators Found

The following stub patterns were NOT present:

- No `pass` statements in processing methods
- No `raise NotImplementedError`
- No `# TODO: implement` comments
- No placeholder return values
- No empty class bodies
- No mock/fake data generation

---

## Conclusion

The `engine/audio/dialogue` and `engine/audio/dsp` modules represent a **complete, production-ready audio processing stack**. The DSP implementations follow established audio engineering practices with mathematically correct algorithms. The dialogue system provides full game audio dialogue management including localization, subtitles, and contextual voice-over.

This is some of the most sophisticated real code found in the TRINITY engine codebase, demonstrating genuine domain expertise in audio DSP and game dialogue systems.
