# PHASE 4 TODO: Advanced Spatialization

**RDC Phase Task Breakdown**
**Phase**: HRTF, VBAP, Ambisonics, and Surround

---

## Task 4.1: Speaker Configuration

**File**: `engine/audio/spatial/speaker_config.py`
**Estimated Lines**: ~515

### Subtasks

- [ ] Define `Speaker` dataclass with name, azimuth, elevation, distance
- [ ] Define `SpeakerLayout` class with speaker list and pairs
- [ ] Implement preset layouts:
  - [ ] STEREO (2 speakers)
  - [ ] QUAD (4 speakers)
  - [ ] SURROUND_5_1 (6 channels)
  - [ ] SURROUND_7_1 (8 channels)
  - [ ] ATMOS_7_1_4 (12 channels)
- [ ] Implement `compute_vbap_pairs()` for 2D layouts
- [ ] Implement `get_channel_index()` name -> index lookup
- [ ] Implement `validate_layout()` sanity checks

### Acceptance Criteria

- All standard layouts match industry specifications
- VBAP pairs cover full 360 degrees
- Layout validation catches malformed configs

---

## Task 4.2: HRTF Implementation

**File**: `engine/audio/spatial/hrtf.py`
**Estimated Lines**: ~546

### Subtasks

- [ ] Define `HRTFConfig` with head radius, filter length
- [ ] Define `HRTFProcessingState` with delay buffers and filter state
- [ ] Implement `calculate_itd()` using Woodworth formula:
  - [ ] `ITD = (r/c) * (theta + sin(theta))`
  - [ ] Convert to samples at sample rate
- [ ] Implement `calculate_ild()`:
  - [ ] `ILD = ILD_MAX_DB * sin(azimuth) * cos(elevation)`
  - [ ] Apply frequency-dependent scaling
- [ ] Implement `generate_synthetic_filters()`:
  - [ ] Low-pass for head shadow
  - [ ] Pinna coloration approximation
- [ ] Implement `HRTFSpatializer.process()`:
  - [ ] Apply ITD via delay lines
  - [ ] Apply ILD via gain difference
  - [ ] Convolve with HRTF filters
  - [ ] Interpolate between positions smoothly
- [ ] Implement `HRTFSpatializer.reset_state()`

### Acceptance Criteria

- ITD range matches human hearing (~700us max)
- ILD follows expected pattern (louder on near ear)
- No clicks when source moves rapidly
- Filter convolution efficient via overlap-add

---

## Task 4.3: VBAP Implementation

**File**: `engine/audio/spatial/spatialization.py` (partial)
**Estimated Lines**: ~200

### Subtasks

- [ ] Implement `VBAPSpatializer.__init__` with speaker layout
- [ ] Implement `_find_speaker_pair()` sector lookup
- [ ] Implement `_solve_2d_vbap()`:
  - [ ] Build 2x2 matrix from speaker vectors
  - [ ] Calculate determinant
  - [ ] Solve for gains
  - [ ] Handle degenerate cases (colinear speakers)
- [ ] Implement `_normalize_gains()` constant-power scaling
- [ ] Implement `VBAPSpatializer.spatialize()`:
  - [ ] Convert direction to Cartesian
  - [ ] Find speaker pair
  - [ ] Solve for gains
  - [ ] Return per-speaker gains
- [ ] Add fallback to nearest speaker if out of range

### Acceptance Criteria

- Gains sum to constant power
- Source between two speakers activates only those two
- Source on speaker activates only that speaker
- Smooth movement across speaker boundaries

---

## Task 4.4: Ambisonics Implementation

**File**: `engine/audio/spatial/spatialization.py` (partial)
**Estimated Lines**: ~150

### Subtasks

- [ ] Implement `AmbisonicsSpatializer.__init__` with order setting
- [ ] Implement `encode()` B-format coefficients:
  - [ ] W = gain / sqrt(2)
  - [ ] Y = gain * sin(az) * cos(el)
  - [ ] Z = gain * sin(el)
  - [ ] X = gain * cos(az) * cos(el)
- [ ] Implement `generate_decoder_matrix()` from speaker layout
- [ ] Implement `decode()` B-format to speaker feeds
- [ ] Implement `set_spread()` reduce directional components
- [ ] Implement `AmbisonicsSpatializer.spatialize()`

### Acceptance Criteria

- B-format encoding matches specification
- Decode -> encode round-trip preserves direction
- Spread = 0 is fully directional
- Spread = 1 is fully diffuse (W only)

---

## Task 4.5: Surround Panner Implementation

**File**: `engine/audio/spatial/spatialization.py` (partial)
**Estimated Lines**: ~180

### Subtasks

- [ ] Implement `SurroundPanner.__init__` with layout
- [ ] Implement channel routing tables for 5.1, 7.1
- [ ] Implement `_map_to_speakers()` azimuth -> channel gains
- [ ] Implement `_apply_lfe_crossover()` route low freq to LFE
- [ ] Implement `SurroundPanner.spatialize()`
- [ ] Handle center channel fold-down for front sources
- [ ] Handle rear speaker mapping

### Acceptance Criteria

- Front center sources route to C channel
- Side sources route to L/R appropriately
- Rear sources route to Ls/Rs (or Lrs/Rrs in 7.1)
- LFE receives low frequencies from all sources

---

## Task 4.6: Doppler Processor

**File**: `engine/audio/spatial/doppler.py`
**Estimated Lines**: ~356

### Subtasks

- [ ] Define `DopplerConfig` with max shift, smoothing
- [ ] Implement `DopplerProcessor.__init__` with per-source state
- [ ] Implement `calculate_pitch_shift()`:
  - [ ] Classical Doppler: `f' = f * c / (c - v_source)`
  - [ ] Handle approaching vs receding
- [ ] Implement `_smooth_pitch()` prevent sudden jumps
- [ ] Implement `DopplerProcessor.process()` apply pitch shift
- [ ] Add clamp for maximum pitch shift

### Acceptance Criteria

- Approaching sources pitch up
- Receding sources pitch down
- Smooth transitions when velocity changes
- Maximum shift clamped to prevent artifacts

---

## Dependencies

- Phase 3: Positioning, Attenuation, Listener
- NumPy for matrix operations
- Math for trigonometry

---

## Verification

1. HRTF test: source circles listener, verify movement perception
2. VBAP test: source at each speaker, verify single speaker active
3. Ambisonics test: encode/decode, verify direction preserved
4. Surround test: source pans around, verify correct channel routing
5. Doppler test: source approaches/recedes, verify pitch shift
