# Investigation: engine/audio/dsp

## Summary
The DSP subsystem is a fully realized, production-quality audio processing implementation with complete sample-by-sample and block-based processing. It includes real biquad filter implementations with proper bilinear transform mathematics, Freeverb-style algorithmic reverb with comb and allpass filters, time-based effects (delay, chorus, flanger, phaser), dynamics processors (compressor, limiter, gate, expander), distortion algorithms, pitch/time manipulation via granular synthesis, and game-specific special effects. All processing uses numpy for efficient block operations with SIMD-aligned buffers.

## Files
| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `__init__.py` | 249 | REAL | Comprehensive exports, all modules imported |
| `config.py` | ~450 | REAL | DSP constants: sample rates, filter params, dynamics thresholds |
| `dsp_node.py` | 493 | REAL | Base DSPNode class with sample/block processing, SmoothedParameter |
| `dsp_graph.py` | ~800 | REAL | DSPChain, DSPParallel, DSPGraph, EffectRack |
| `filters.py` | 973 | REAL | BiquadFilter, StateVariableFilter, ParametricEQ, OnePoleFilter |
| `dynamics.py` | 1351 | REAL | Compressor, Limiter, Gate, Expander, SidechainCompressor |
| `time_effects.py` | 972 | REAL | DelayLine, Delay, Chorus, Flanger, Phaser, Vibrato, LFO |
| `reverb.py` | 856 | REAL | Freeverb, PlateReverb, ConvolutionReverb, CombFilter |
| `distortion.py` | 455 | REAL | Hard/soft clipping, tube, tape, bitcrusher, waveshaper |
| `pitch_time.py` | 580 | REAL | PitchShifter, TimeStretcher via granular synthesis |
| `special_fx.py` | 741 | REAL | RadioEffect, UnderwaterEffect, ExplosionEffect, etc. |

## DSP Components

### Filters (filters.py)
- **BiquadFilter**: Full implementation of Direct Form II Transposed with proper bilinear transform for LP, HP, BP, notch, allpass, peak, low/high shelf
- **BiquadCoefficients**: Coefficient calculation using omega, cos/sin, alpha
- **StateVariableFilter**: Numerically stable SVF with simultaneous LP/HP/BP/notch outputs
- **ParametricEQ**: Multi-band cascaded EQ (default 4 bands)
- **OnePoleFilter**: First-order filter for smoothing/DC blocking
- **DCBlocker**: High-pass at 20Hz for DC offset removal

### Dynamics (dynamics.py)
- **EnvelopeFollower**: RMS/Peak detection with attack/release coefficients
- **Compressor**: Soft knee, makeup gain, stereo linking, ratio 1:1 to 100:1
- **Limiter**: Brickwall with lookahead, peak detection
- **Gate**: Hold time, range control, smooth open/close
- **Expander**: Downward expansion for noise reduction
- **MultibandCompressor**: Crossover filters + per-band compression
- **SidechainCompressor**: External key signal input

### Time Effects (time_effects.py)
- **LFO**: Sine, triangle, square, saw waveforms with phase control
- **DelayLine**: Circular buffer with linear/cubic interpolation
- **Delay**: Ping-pong mode, tempo sync capable
- **MultiTapDelay**: Multiple read taps with individual gains
- **Chorus**: Multi-voice LFO-modulated delay
- **Flanger**: Short modulated delay with feedback
- **Phaser**: Cascaded allpass filters with LFO sweep
- **Vibrato**: Pitch modulation via delay modulation

### Reverb (reverb.py)
- **CombFilter**: Feedback comb with damping (one-pole LP)
- **AllPassFilterReverb**: Schroeder allpass for diffusion
- **Freeverb**: 8 parallel combs + 4 series allpass, stereo spread
- **PlateReverb**: Input diffusers, tank delays, damping filters
- **ConvolutionReverb**: FFT-based overlap-add convolution with IR
- **SimpleReverb**: Lightweight 4-comb reverb

### Distortion (distortion.py)
- **HardClipper**: Digital clipping at +/-1
- **SoftClipper**: Smooth saturation curve
- **TanhClip**: Hyperbolic tangent saturation
- **TubeSaturator**: Asymmetric exp-based tube emulation
- **TapeSaturator**: Soft compression + even harmonics
- **Bitcrusher**: Bit depth + sample rate reduction
- **Waveshaper**: Table-based transfer function
- **Foldback**: Wave folding distortion

### Pitch/Time (pitch_time.py)
- **PitchShifter**: Granular synthesis with Hann windowing, overlap-add
- **TimeStretcher**: WSOLA-style time stretching
- **PitchTimeProcessor**: Combined pitch + time manipulation
- **SimplePitchShifter**: Resampling-based (changes duration)

### Special FX (special_fx.py)
- **RadioEffect**: Band-pass + distortion + noise/crackle
- **UnderwaterEffect**: Low-pass with resonance + bubble sounds
- **SlowMotionEffect**: Low-pass + delay/reverb tail
- **ExplosionEffect**: Tinnitus tone + muffled recovery over time
- **MuffledEffect**: Low-pass + gain reduction
- **PhoneEffect**: 300-3400Hz band-pass + compression
- **MegaphoneEffect**: Band-pass + soft clipping
- **CaveEffect**: Dual delay lines + low-pass

### DSP Infrastructure (dsp_node.py, dsp_graph.py)
- **DSPNode**: Abstract base with process_sample(), process_block(), bypass, state
- **SmoothedParameter**: Thread-safe exponential smoothing for click-free changes
- **DSPChain**: Series node processing
- **DSPParallel**: Parallel processing with summing
- **DSPGraph**: Arbitrary routing via NodeConnection
- **EffectRack**: Insert/send effect management

## Implementation

### Real filters (LP/HP/BP)? **YES**
The BiquadFilter class implements proper biquad coefficient calculation using the bilinear transform. The `_calculate_coefficients()` method computes omega, sin/cos omega, and alpha from frequency and Q, then derives b0/b1/b2/a0/a1/a2 for each filter type (lowpass, highpass, bandpass, notch, allpass, peak, shelving). The difference equation `y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]` is implemented using Direct Form II Transposed for numerical stability. The StateVariableFilter provides an alternative topology that allows smooth modulation.

### Real effects (reverb/delay)? **YES**
- **Reverb**: Freeverb implements the Schroeder reverb topology with 8 parallel comb filters (delays: 1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617 samples at 44.1kHz) followed by 4 series allpass filters (556, 441, 341, 225). Each comb has a one-pole damping filter. Stereo is achieved via sample offset (REVERB_STEREO_SPREAD=23).
- **Delay**: Full implementation with circular buffer, linear/cubic interpolation for fractional delays, ping-pong mode, multi-tap support.
- **Chorus/Flanger/Phaser**: Proper LFO modulation of delay times with interpolated reads.
- **ConvolutionReverb**: FFT-based overlap-add using numpy.fft.rfft/irfft.

### Real-time processing? **YES**
- Block-based processing with configurable BLOCK_SIZE (default 512)
- SIMD-aligned buffer allocation (32-byte alignment for AVX)
- SmoothedParameter class prevents zipper noise via exponential interpolation
- Thread-safe parameter updates via threading.RLock
- Per-sample and per-block processing modes
- DSPChain avoids memory allocation during processing (pre-allocated intermediate buffers)

## Verdict
**REAL IMPLEMENTATION**

This is a complete, production-quality DSP subsystem with mathematically correct filter implementations, industry-standard reverb algorithms (Freeverb), and comprehensive effect coverage. The code quality is high with proper abstraction (DSPNode base class), efficient block processing, thread safety, and SIMD-friendly memory layout.

## Evidence

### BiquadFilter coefficient calculation (lines 170-249 of filters.py):
```python
def _calculate_coefficients(self) -> None:
    # Pre-warp frequency for bilinear transform
    omega = 2.0 * math.pi * freq / sr
    sin_omega = math.sin(omega)
    cos_omega = math.cos(omega)
    alpha = sin_omega / (2.0 * q)

    if self._filter_type == FilterType.LOWPASS:
        b0 = (1.0 - cos_omega) / 2.0
        b1 = 1.0 - cos_omega
        b2 = (1.0 - cos_omega) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_omega
        a2 = 1.0 - alpha
```

### Freeverb comb filter processing (lines 112-125 of reverb.py):
```python
def process(self, input_sample: float) -> float:
    output = self._buffer[self._buffer_index]
    # One-pole low-pass filter for damping
    self._filter_state = output * (1.0 - self._damping) + self._filter_state * self._damping
    # Store input + filtered feedback
    self._buffer[self._buffer_index] = input_sample + self._filter_state * self._feedback
    self._buffer_index = (self._buffer_index + 1) % self._delay_samples
    return output
```

### SmoothedParameter for click-free automation (lines 57-127 of dsp_node.py):
```python
def advance(self) -> float:
    """Advance smoothing by one sample and return current value."""
    self._current_value += self._coefficient * (self._target_value - self._current_value)
    return self._current_value
```

### Granular pitch shifting (lines 155-174 of pitch_time.py):
```python
def _apply_hann_window(self, grain: np.ndarray) -> np.ndarray:
    n = len(grain)
    window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / n))
    return grain * window

def _resample_grain(self, grain: np.ndarray, ratio: float) -> np.ndarray:
    output_length = len(grain)
    output = np.zeros(output_length, dtype=np.float64)
    for i in range(output_length):
        position = i * ratio
        index = int(position) % len(grain)
        frac = position - int(position)
        next_index = (index + 1) % len(grain)
        output[i] = grain[index] * (1.0 - frac) + grain[next_index] * frac
    return output
```
