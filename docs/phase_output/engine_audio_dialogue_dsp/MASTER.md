# MASTER: engine_audio_dialogue_dsp

**RDC Consolidated Knowledge Document**
**Generated:** 2026-05-23
**Total Source Lines:** ~14,187 lines of production Python code

---

## 1. Executive Classification

**STATUS: REAL IMPLEMENTATION**

The `engine/audio/dialogue` and `engine/audio/dsp` subsystems contain **genuine, production-quality audio implementations**. This is NOT stub code. Evidence includes:

- Mathematically correct DSP algorithms (bilinear transform, Schroeder reverb topology)
- Proper sample-by-sample and block-based processing with state management
- Real coefficient calculations using established audio engineering formulas
- Thread-safe implementations with RLock synchronization
- NumPy-based efficient buffer processing with SIMD alignment considerations

---

## 2. Architecture Overview

### 2.1 Subsystem Boundaries

```
engine/audio/
+-- dialogue/              # Voice-over and conversation management (6,267 lines)
|   +-- Playback           # VOLine, VOQueue, VOStreamManager
|   +-- Conversations      # State machine, branching, choices
|   +-- Contextual         # Barks, ambient VO, line pools
|   +-- Localization       # Multi-language, audio banks, fallbacks
|   +-- Subtitles          # Timing, tracks, display management
|   +-- Processing         # Radio effect, spatial audio, reverb
|
+-- dsp/                   # Digital Signal Processing (~7,920 lines)
    +-- Infrastructure     # DSPNode, SmoothedParameter, DSPGraph
    +-- Filters            # Biquad, SVF, ParametricEQ, DCBlocker
    +-- Dynamics           # Compressor, Limiter, Gate, Expander
    +-- Time Effects       # Delay, Chorus, Flanger, Phaser
    +-- Reverb             # Freeverb, PlateReverb, ConvolutionReverb
    +-- Distortion         # Tube, Tape, Bitcrusher, Waveshaper
    +-- Pitch/Time         # Granular synthesis, WSOLA
    +-- Special FX         # Radio, Underwater, Explosion, Phone
```

### 2.2 Integration Points

| Integration | From | To | Mechanism |
|------------|------|-----|-----------|
| VO Processing | dialogue/vo_processing.py | dsp/filters.py | Bandpass for radio effect |
| Environment Reverb | dialogue/vo_processing.py | dsp/reverb.py | ReverbSettings presets |
| Effect Chains | dialogue/dialogue_manager.py | dsp/dsp_graph.py | DSPChain routing |

---

## 3. Dialogue Subsystem (6,267 lines)

### 3.1 Voice Playback System

#### VOLine (vo_line.py, 341 lines)
- Complete VO line representation
- Fields: audio_asset, text, speaker, duration, priority, conditions
- Playback state tracking (PENDING, PLAYING, COMPLETED, INTERRUPTED)
- Lip sync data attachment
- Subtitle data binding

#### VOQueue (vo_queue.py, 573 lines)
- Heap-based priority queue
- Timeout expiration handling
- Interrupt support (higher priority can interrupt lower)
- Maximum simultaneous VO (configurable, default 2)
- Statistics: total_played, total_interrupted, total_expired

```python
@dataclass(order=True)
class QueueEntry:
    sort_key: tuple[int, float] = field(compare=True)  # (neg_priority, time)
    line: VOLine = field(compare=False)
```

#### VOStreamManager (vo_streaming.py, 707 lines)
- LRU cache with configurable size
- Hit/miss tracking with eviction threshold
- Preload queue management
- Memory budgeting
- Stream states: IDLE, BUFFERING, STREAMING, COMPLETE, ERROR

#### VOProcessor (vo_processing.py, 728 lines)
- **Radio Effect**: Bandpass filter (300-3400Hz) + distortion + noise/crackle
- **Distance Filtering**: Low-pass based on distance (attenuation curve)
- **Environment Reverb Presets**:
  - outdoor: room_size=0.2, damping=0.8, decay_time=0.5s
  - cave: room_size=0.9, damping=0.2, decay_time=3.0s
  - church: room_size=0.8, damping=0.3, decay_time=4.0s
- **3D Spatialization**: Pan calculation from listener/source positions

```python
def calculate_pan(self, listener_position, listener_forward) -> float:
    source_angle = math.atan2(dz, dx)
    relative_angle = source_angle - forward_angle
    pan = math.sin(relative_angle)
    return max(-1.0, min(1.0, pan)) * self.blend
```

### 3.2 Conversation Management

#### Conversation (conversation.py, 725 lines)
- Full dialogue trees with ConversationNode
- State machine: INACTIVE -> STARTING -> ACTIVE -> WAITING -> COMPLETED
- Branch points with player choices
- Condition evaluation for conditional branches
- Participant tracking (speakers involved)
- on_enter/on_exit callbacks per node

#### ConversationManager (dialogue_manager.py, 710 lines)
- Multi-conversation support (up to 4 active simultaneously)
- Pause/resume functionality
- Auto-advance option
- Integration of all dialogue subsystems

### 3.3 Contextual Dialogue

#### BarkSystem (contextual_dialogue.py, 774 lines)
- Short reaction barks (reload, enemy_spotted, low_health)
- Cooldown tracking per line, per speaker, per category
- Selection modes: random, sequential, weighted, shuffle, conditional

#### AmbientVOSystem
- Zone-based ambient VO
- Interval randomization (min/max delay between plays)
- Trigger conditions (time of day, weather, player state)

#### LinePool
- Collection of VO lines for contextual selection
- Selection algorithms:
  - **random**: Uniform random
  - **sequential**: Round-robin
  - **weighted**: Probability weights
  - **shuffle**: Randomized order, no repeats until exhausted
  - **conditional**: Based on game state evaluation

#### CooldownTracker
- Per-line cooldown (specific line cannot repeat)
- Per-speaker cooldown (same character cannot talk)
- Per-category cooldown (same type of bark cannot repeat)

### 3.4 Localization Support

#### LocalizationManager (localization.py, 580 lines)
- 10 supported languages: en, es, fr, de, it, ja, ko, zh, pt, ru
- Fallback chains (e.g., es-MX -> es -> en)
- Audio bank loading/unloading by language
- Asset variants per language

#### LocalizedAsset
- Per-language audio paths
- Per-language durations (may differ)
- Per-language subtitles
- Per-language lip sync data

#### AudioBank
- Collections of localized assets
- Organized by language and category
- Hot-swap on language change

### 3.5 Subtitle Synchronization

#### SubtitleManager (subtitle_sync.py, 636 lines)
- Display management with fade animations
- Multi-speaker support (different colors/positions)
- Reading speed calculation for automatic timing

#### SubtitleTrack
- Timed cue points synchronized to audio playback
- Start/end times per subtitle segment
- Character timing for typewriter effects

#### SubtitleStyle
- Font configuration
- Color (text, background, outline, shadow)
- Position (top, center, bottom)
- Animation (fade, slide, pop)

---

## 4. DSP Subsystem (~7,920 lines)

### 4.1 DSP Infrastructure

#### DSPNode (dsp_node.py, 493 lines)
- Abstract base class for all DSP processors
- Methods: process_sample(), process_block()
- Bypass mode with wet/dry mix
- State management (reset, clear)
- Sample rate change handling

#### SmoothedParameter
- Thread-safe exponential smoothing
- Prevents zipper noise on parameter changes
- Configurable smoothing coefficient
- advance() method returns interpolated value

```python
def advance(self) -> float:
    self._current_value += self._coefficient * (self._target_value - self._current_value)
    return self._current_value
```

#### DSPGraph (dsp_graph.py, ~800 lines)
- **DSPChain**: Series node processing
- **DSPParallel**: Parallel processing with summing
- **DSPGraph**: Arbitrary routing via NodeConnection
- **EffectRack**: Insert/send effect management
- Topological sort for processing order
- Pre-allocated intermediate buffers (no allocation during processing)

### 4.2 Filters (filters.py, 973 lines)

#### BiquadFilter
- Direct Form II Transposed implementation
- Bilinear transform for analog-to-digital conversion
- Filter types: lowpass, highpass, bandpass, notch, allpass, peak, low_shelf, high_shelf
- Per-channel state arrays (z1, z2)
- Sample rate change triggers coefficient recalculation

```python
# Coefficient calculation (lowpass example)
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

#### StateVariableFilter
- Numerically stable topology
- Simultaneous outputs: lowpass, highpass, bandpass, notch
- Smooth modulation-friendly
- Oversampling option for stability at high frequencies

#### ParametricEQ
- Multi-band cascaded biquad (default 4 bands)
- Per-band: frequency, Q, gain
- Frequency response calculation
- Bypass per band

#### OnePoleFilter
- First-order filter for smoothing/DC blocking
- Configurable coefficient

#### DCBlocker
- High-pass at 20Hz for DC offset removal
- Uses OnePoleFilter internally

### 4.3 Dynamics (dynamics.py, 1,351 lines)

#### EnvelopeFollower
- Detection modes: RMS, Peak
- Attack/release coefficient calculation:
```python
attack_samples = ms_to_samples(self._attack_ms, sr)
self._attack_coeff = math.exp(-1.0 / attack_samples)
```

#### Compressor
- Threshold (dB)
- Ratio (1:1 to 100:1, infinity for limiting)
- Attack/release (ms)
- Knee (hard/soft with interpolation)
- Makeup gain (manual or auto)
- Stereo linking

**Soft Knee Implementation:**
```python
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

#### Limiter
- Brickwall limiting
- Lookahead with delay compensation
- Peak detection
- Release time
- Output ceiling

#### Gate
- Threshold (dB)
- Attack/hold/release (ms)
- Range (dB reduction when closed)
- Smooth open/close transitions

#### Expander
- Downward expansion for noise reduction
- Ratio, threshold, attack/release
- Range limiting

#### MultibandCompressor
- Crossover filters (2-way, 3-way, 4-way)
- Per-band compression settings
- Band solo/mute
- Output summing

#### SidechainCompressor
- External key signal input
- Key filter (highpass for de-essing, lowpass for ducking)
- Full compressor controls

### 4.4 Time Effects (time_effects.py, 972 lines)

#### LFO
- Waveforms: sine, triangle, square, saw, random
- Phase accumulation
- Sync to external clock
- Rate in Hz

#### DelayLine
- Circular buffer implementation
- Interpolation: linear, cubic (Hermite)
- Maximum delay in samples

**Hermite Interpolation:**
```python
c0 = y1
c1 = 0.5 * (y2 - y0)
c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3
c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2)
return ((c3 * frac + c2) * frac + c1) * frac + c0
```

#### Delay
- Delay time (ms or samples)
- Feedback (0.0 to 1.0)
- Wet/dry mix
- Ping-pong mode (stereo bounce)
- Tempo sync capable

#### MultiTapDelay
- Multiple read taps
- Per-tap gain and pan
- Per-tap feedback option

#### Chorus
- Multi-voice (2-8 voices typical)
- LFO-modulated delay per voice
- Phase spread between voices
- Depth, rate controls

#### Flanger
- Short modulated delay (0.1-10ms)
- High feedback for resonance
- Through-zero option
- Depth, rate, feedback controls

#### Phaser
- Cascaded allpass filters (4, 8, or 12 stages)
- LFO-modulated center frequency
- Feedback path
- Stages, rate, depth, feedback controls

#### Vibrato
- Pitch modulation via delay modulation
- No mix with dry signal
- Rate, depth controls

### 4.5 Reverb (reverb.py, 856 lines)

#### CombFilter
- Feedback comb implementation
- One-pole low-pass damping filter
- Delay time, feedback, damping controls

```python
def process(self, input_sample: float) -> float:
    output = self._buffer[self._buffer_index]
    self._filter_state = output * (1.0 - self._damping) + self._filter_state * self._damping
    self._buffer[self._buffer_index] = input_sample + self._filter_state * self._feedback
    self._buffer_index = (self._buffer_index + 1) % self._delay_samples
    return output
```

#### AllPassFilterReverb
- Schroeder allpass for diffusion
- Feedback coefficient
- Delay samples

#### Freeverb
- 8 parallel comb filters + 4 series allpass (Schroeder topology)
- Comb delays: [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617] samples @ 44.1kHz
- Allpass delays: [556, 441, 341, 225] samples
- Stereo spread: 23 samples offset
- Room size, damping, wet/dry, width controls

#### PlateReverb
- Input diffusers (4-6 allpass filters)
- Tank delays (modulated)
- Cross-coupling between channels
- Damping filters
- Decay time, damping, diffusion controls

#### ConvolutionReverb
- FFT-based overlap-add implementation
- Impulse response loading
- Partitioned convolution for long IRs
- Wet/dry mix

```python
input_fft = np.fft.rfft(input_padded)
output_fft = input_fft * self._ir_fft
conv_output = np.fft.irfft(output_fft)
# Overlap-add to output buffer
```

#### SimpleReverb
- Lightweight 4-comb reverb
- Minimal CPU for background ambience

### 4.6 Distortion (distortion.py, 455 lines)

#### HardClipper
- Digital clipping at +/-1 (or configurable threshold)
- Alias-prone at high gains

#### SoftClipper
- Smooth saturation curve (tanh-like)
- Less aliasing than hard clipping

#### TanhClip
- Hyperbolic tangent saturation
- Drive parameter scales input

#### TubeSaturator
- Asymmetric exponential curve (tube emulation)
- Even and odd harmonics
- Warmth parameter

#### TapeSaturator
- Soft compression characteristic
- Bias control for even harmonics
- High-frequency rolloff

#### Bitcrusher
- Bit depth reduction (1-16 bits)
- Sample rate reduction
- Pre/post filter options

#### Waveshaper
- Table-based transfer function
- Cubic interpolation between table points
- Customizable curve

#### Foldback
- Wave folding distortion
- Threshold parameter
- Creates complex harmonics

### 4.7 Pitch/Time (pitch_time.py, 580 lines)

#### PitchShifter
- Granular synthesis approach
- Hann windowing
- Overlap-add reconstruction
- Resampling within grains

```python
def _apply_hann_window(self, grain: np.ndarray) -> np.ndarray:
    n = len(grain)
    window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / n))
    return grain * window

def _resample_grain(self, grain: np.ndarray, ratio: float) -> np.ndarray:
    # Linear interpolation resampling
```

#### TimeStretcher
- WSOLA-style (Waveform Similarity Overlap-Add)
- Grain synchronization to transients
- Independent duration control (preserves pitch)

#### PitchTimeProcessor
- Combined pitch shift + time stretch
- Formant preservation option

#### SimplePitchShifter
- Resampling-based (changes duration proportionally)
- Lower latency than granular

### 4.8 Special FX (special_fx.py, 741 lines)

#### RadioEffect
- Bandpass filter (300-3400Hz, telephone band)
- Soft distortion
- White noise/crackle overlay
- AM modulation option

#### UnderwaterEffect
- Low-pass filter (500-1000Hz)
- Resonance boost
- Optional bubble sounds
- Depth parameter controls cutoff

#### SlowMotionEffect
- Low-pass sweep
- Delay/reverb tail increase
- Pitch drop (optional)
- Transition time parameter

#### ExplosionEffect
- Initial muffling (low-pass)
- Tinnitus tone (high-frequency sine)
- Gradual recovery over time
- Intensity, recovery_time parameters

#### MuffledEffect
- Low-pass filter
- Gain reduction
- Used for obstacles, walls

#### PhoneEffect
- Bandpass (300-3400Hz)
- Compression (telephone dynamics)
- Minimal processing

#### MegaphoneEffect
- Bandpass (wider than phone)
- Soft clipping
- Slight distortion

#### CaveEffect
- Dual delay lines (cross-feedback)
- Low-pass damping
- Simulates hard surface reflections

---

## 5. Technical Implementation Quality

### 5.1 Real DSP Indicators

| Indicator | Evidence |
|-----------|----------|
| Coefficient Calculations | All filters use proper bilinear transform formulas |
| State Persistence | Per-channel state arrays (z1, z2 for biquads) |
| Sample Rate Handling | Coefficient recalculation on sample rate change |
| Block Processing | Efficient NumPy operations for batch processing |
| Thread Safety | RLock usage in managers |
| SIMD Alignment | Buffer allocation with 32-byte alignment for AVX |

### 5.2 Absent Stub Patterns

The following stub patterns were NOT present:
- No `pass` statements in processing methods
- No `raise NotImplementedError`
- No `# TODO: implement` comments
- No placeholder return values
- No empty class bodies
- No mock/fake data generation

### 5.3 Dependencies

**Internal:**
- dsp_node.py - Base DSP node class
- config.py - Constants and utility functions
- vo_line.py - Voice-over line data structure
- distortion.py - Distortion effect (used by special_fx.py)

**External:**
- numpy - Array processing and FFT
- math - Trigonometric functions
- threading - Synchronization primitives
- dataclasses - Data structures
- enum - Type enumerations
- uuid - Unique identifiers

---

## 6. Key Algorithms Reference

### 6.1 DSP Algorithms

| Algorithm | Location | Description |
|-----------|----------|-------------|
| Bilinear Transform | filters.py | Analog-to-digital filter conversion |
| Direct Form II Transposed | filters.py | Numerically stable biquad implementation |
| Schroeder Reverb | reverb.py | 8 combs + 4 allpass topology |
| FFT Convolution | reverb.py | Overlap-add for impulse response |
| Hermite Interpolation | time_effects.py | Cubic interpolation for fractional delay |
| Granular Synthesis | pitch_time.py | Hann-windowed overlap-add pitch shift |
| Envelope Following | dynamics.py | RMS/Peak with attack/release coefficients |
| Soft Knee Compression | dynamics.py | Quadratic interpolation in dB domain |

### 6.2 Dialogue Algorithms

| Algorithm | Location | Description |
|-----------|----------|-------------|
| Priority Heap Queue | vo_queue.py | Heap-based VO scheduling with interrupt |
| LRU Cache | vo_streaming.py | Least-recently-used eviction for audio cache |
| State Machine | conversation.py | Branching dialogue with conditions |
| Cooldown Tracking | contextual_dialogue.py | Per-line, per-speaker, per-category |
| 3D Pan Calculation | vo_processing.py | Listener-relative angle to stereo pan |
