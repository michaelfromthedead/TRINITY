# PHASE 2 ARCH: DSP Subsystem

**Architecture Specification**
**Phase:** 2 of 2
**Scope:** engine/audio/dsp/

---

## 1. Phase Overview

Phase 2 covers the **DSP Subsystem** - the complete digital signal processing stack for audio effects. This phase provides processing capabilities used by Phase 1 (Dialogue) and other audio systems.

**Total Implementation:** 6,761 lines across 11 files

---

## 2. Module Architecture

```
engine/audio/dsp/
+-- __init__.py (249)         # Public API exports
+-- config.py (~450)          # DSP constants and utilities
+-- dsp_node.py (493)         # Base class and SmoothedParameter
+-- dsp_graph.py (~800)       # Effect chains and routing
+-- filters.py (973)          # Biquad, SVF, EQ, DC blocking
+-- dynamics.py (1,351)       # Compressor, limiter, gate, expander
+-- time_effects.py (972)     # Delay, chorus, flanger, phaser
+-- reverb.py (856)           # Freeverb, plate, convolution
+-- distortion.py (455)       # Tube, tape, bitcrusher, waveshaper
+-- pitch_time.py (580)       # Granular pitch shift, time stretch
+-- special_fx.py (741)       # Radio, underwater, explosion effects
```

---

## 3. Core Infrastructure

### 3.1 DSPNode (dsp_node.py)

**Purpose:** Abstract base class for all DSP processors.

**Interface:**
```python
class DSPNode(ABC):
    def process_sample(self, sample: float) -> float: ...
    def process_block(self, input_buffer: np.ndarray) -> np.ndarray: ...
    def reset(self) -> None: ...
    def set_sample_rate(self, sample_rate: int) -> None: ...
    
    @property
    def bypass(self) -> bool: ...
    @property
    def wet_dry(self) -> float: ...  # 0.0 = dry, 1.0 = wet
```

**Lifecycle:**
1. Construct with sample_rate
2. Set parameters (triggers coefficient calculation)
3. process_block() or process_sample() repeatedly
4. reset() when needed (clears state)
5. set_sample_rate() if rate changes (recalculates coefficients)

### 3.2 SmoothedParameter (dsp_node.py)

**Purpose:** Click-free parameter automation.

**Algorithm:** Exponential smoothing
```python
def advance(self) -> float:
    self._current_value += self._coefficient * (self._target_value - self._current_value)
    return self._current_value
```

**Configuration:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| smoothing_ms | 10.0 | Time to reach 63% of target |
| sample_rate | 44100 | For coefficient calculation |

### 3.3 DSPGraph (dsp_graph.py)

**Purpose:** Composable effect routing.

**Components:**

#### DSPChain
- Series processing: A -> B -> C
- Pre-allocated intermediate buffers
- No allocation during processing

#### DSPParallel
- Parallel processing: [A, B, C] -> sum
- Optional per-branch gain

#### DSPGraph
- Arbitrary routing via NodeConnection
- Topological sort for processing order
- Feedback detection and handling

#### EffectRack
- Insert slots (series)
- Send slots (parallel mix)
- Wet/dry per slot

---

## 4. Filter Specifications

### 4.1 BiquadFilter (filters.py)

**Topology:** Direct Form II Transposed

**Filter Types:**
| Type | Transfer Function |
|------|-------------------|
| lowpass | LP2 with resonance |
| highpass | HP2 with resonance |
| bandpass | BP2 with Q |
| notch | Notch with Q |
| allpass | AP2 with Q |
| peak | Parametric peak/dip |
| low_shelf | Low frequency shelf |
| high_shelf | High frequency shelf |

**Coefficient Calculation:**
```python
omega = 2.0 * math.pi * freq / sr
sin_omega = math.sin(omega)
cos_omega = math.cos(omega)
alpha = sin_omega / (2.0 * q)

# Lowpass example:
b0 = (1.0 - cos_omega) / 2.0
b1 = 1.0 - cos_omega
b2 = (1.0 - cos_omega) / 2.0
a0 = 1.0 + alpha
a1 = -2.0 * cos_omega
a2 = 1.0 - alpha
```

**State:** Per-channel z1, z2 delay elements

### 4.2 StateVariableFilter (filters.py)

**Advantage:** Numerically stable, simultaneous outputs

**Outputs:** lowpass, highpass, bandpass, notch

**Parameters:** frequency, resonance (0-1)

### 4.3 ParametricEQ (filters.py)

**Architecture:** Cascaded biquad bands

**Default:** 4 bands

**Per-Band:** frequency, Q, gain_db

### 4.4 DCBlocker (filters.py)

**Purpose:** Remove DC offset

**Frequency:** 20Hz highpass

---

## 5. Dynamics Specifications

### 5.1 EnvelopeFollower (dynamics.py)

**Detection Modes:** RMS, Peak

**Coefficient Formula:**
```python
attack_samples = ms_to_samples(attack_ms, sr)
attack_coeff = math.exp(-1.0 / attack_samples)
```

### 5.2 Compressor (dynamics.py)

**Parameters:**
| Parameter | Range | Description |
|-----------|-------|-------------|
| threshold | -60 to 0 dB | Compression start point |
| ratio | 1:1 to 100:1 | Compression amount |
| attack | 0.1 to 200 ms | Attack time |
| release | 10 to 2000 ms | Release time |
| knee | 0 to 20 dB | Soft knee width |
| makeup | -20 to +20 dB | Output gain |

**Soft Knee Algorithm:**
```python
half_knee = knee / 2.0
knee_start = threshold - half_knee
knee_end = threshold + half_knee

if input_db < knee_start:
    gain_reduction = 0.0
elif input_db > knee_end:
    gain_reduction = (threshold - input_db) * (1.0 - 1.0 / ratio)
else:
    x = input_db - knee_start
    gain_reduction = (1.0 / ratio - 1.0) * (x * x) / (2.0 * knee)
```

### 5.3 Limiter (dynamics.py)

**Type:** Brickwall with lookahead

**Parameters:** threshold, release, lookahead_ms

**Lookahead:** Delay line for peak anticipation

### 5.4 Gate (dynamics.py)

**Parameters:** threshold, attack, hold, release, range

**Range:** dB reduction when closed (e.g., -80 dB)

### 5.5 MultibandCompressor (dynamics.py)

**Crossover:** Linkwitz-Riley filters (2/3/4-way)

**Per-Band:** Full compressor controls

---

## 6. Time Effects Specifications

### 6.1 LFO (time_effects.py)

**Waveforms:**
- sine: sin(2 * pi * phase)
- triangle: 4 * |phase - 0.5| - 1
- square: sign(sin(2 * pi * phase))
- saw: 2 * phase - 1
- random: sample-and-hold noise

### 6.2 DelayLine (time_effects.py)

**Implementation:** Circular buffer

**Interpolation:**
| Type | Quality | CPU |
|------|---------|-----|
| none | Low (aliasing) | Lowest |
| linear | Medium | Low |
| cubic (Hermite) | High | Medium |

**Hermite Formula:**
```python
c0 = y1
c1 = 0.5 * (y2 - y0)
c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3
c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2)
output = ((c3 * frac + c2) * frac + c1) * frac + c0
```

### 6.3 Delay (time_effects.py)

**Modes:** Normal, Ping-pong (stereo)

**Parameters:** delay_ms, feedback, wet_dry

### 6.4 Chorus (time_effects.py)

**Voices:** 2-8 (default: 3)

**Per-Voice:** Phase offset, depth variation

### 6.5 Flanger (time_effects.py)

**Delay Range:** 0.1-10 ms

**Through-Zero:** Optional polarity inversion at minimum

### 6.6 Phaser (time_effects.py)

**Stages:** 4, 8, or 12 allpass filters

**Sweep:** LFO-modulated center frequency

---

## 7. Reverb Specifications

### 7.1 Freeverb (reverb.py)

**Topology:** Schroeder (8 parallel combs + 4 series allpass)

**Comb Delays (@ 44.1kHz):**
| Index | Delay (samples) |
|-------|-----------------|
| 0 | 1116 |
| 1 | 1188 |
| 2 | 1277 |
| 3 | 1356 |
| 4 | 1422 |
| 5 | 1491 |
| 6 | 1557 |
| 7 | 1617 |

**Allpass Delays:**
| Index | Delay (samples) |
|-------|-----------------|
| 0 | 556 |
| 1 | 441 |
| 2 | 341 |
| 3 | 225 |

**Stereo Spread:** 23 samples offset

**Parameters:** room_size, damping, wet, dry, width

### 7.2 PlateReverb (reverb.py)

**Diffusers:** 4-6 input allpass filters

**Tank:** Modulated delays with cross-coupling

**Parameters:** decay, damping, diffusion, size

### 7.3 ConvolutionReverb (reverb.py)

**Algorithm:** FFT-based overlap-add

**Implementation:**
```python
input_fft = np.fft.rfft(input_padded)
output_fft = input_fft * self._ir_fft
conv_output = np.fft.irfft(output_fft)
```

**Partitioning:** For long IRs, use partitioned convolution

---

## 8. Distortion Specifications

### 8.1 Transfer Functions

| Type | Formula | Character |
|------|---------|-----------|
| HardClipper | clip(x, -1, 1) | Harsh, digital |
| SoftClipper | x / (1 + |x|) | Smooth |
| TanhClip | tanh(drive * x) | Warm, tube-like |
| TubeSaturator | asymmetric exp | Even+odd harmonics |
| TapeSaturator | soft comp + rolloff | Vintage, warm |

### 8.2 Bitcrusher (distortion.py)

**Parameters:**
- bit_depth: 1-16 bits
- sample_rate_reduction: 1-100x

---

## 9. Pitch/Time Specifications

### 9.1 PitchShifter (pitch_time.py)

**Algorithm:** Granular synthesis

**Grain Processing:**
1. Extract grain from input
2. Apply Hann window
3. Resample grain by pitch ratio
4. Overlap-add to output

**Hann Window:**
```python
window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / n))
```

**Parameters:** pitch_ratio, grain_size_ms, overlap

### 9.2 TimeStretcher (pitch_time.py)

**Algorithm:** WSOLA (Waveform Similarity Overlap-Add)

**Parameters:** time_ratio, grain_size_ms

---

## 10. Special Effects Specifications

### 10.1 RadioEffect (special_fx.py)

**Chain:** Bandpass (300-3400Hz) -> Distortion -> Noise

**Parameters:** quality (affects bandwidth), noise_level

### 10.2 UnderwaterEffect (special_fx.py)

**Chain:** Lowpass (variable) -> Resonance boost

**Parameters:** depth (affects cutoff), bubble_level

### 10.3 ExplosionEffect (special_fx.py)

**Behavior:**
1. Initial muffling (lowpass)
2. Tinnitus tone (high sine)
3. Gradual recovery over time

**Parameters:** intensity, recovery_time_ms

---

## 11. Memory Layout

### 11.1 Buffer Alignment

- 32-byte alignment for AVX SIMD
- NumPy arrays with proper dtype (float64 for precision)

### 11.2 Pre-allocation

- DSPChain pre-allocates intermediate buffers at construction
- No allocation during process_block()

### 11.3 State Arrays

- Per-channel state (z1, z2 for biquads)
- Circular buffers for delays
- FFT buffers for convolution
