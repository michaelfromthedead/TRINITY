# MASTER: Audio Mixing and Spatial Subsystems

**RDC Consolidated Knowledge Base**
**Last Updated**: 2026-05-23

---

## 1. Subsystem Overview

### 1.1 Classification

Both `engine/audio/mixing` and `engine/audio/spatial` are **fully implemented, production-ready subsystems** with complete algorithms, thread-safe designs, and professional audio engineering principles.

### 1.2 Scale

| Metric | Mixing | Spatial | Total |
|--------|--------|---------|-------|
| Files | 7-10 | 12 | 19-22 |
| Lines | ~5,020 | ~6,656 | ~11,676 |
| Classes | 15+ | 25+ | 40+ |
| Algorithms | 8+ | 12+ | 20+ |
| Stubs | 0 | 0 | 0 |

---

## 2. Audio Mixing Subsystem

### 2.1 Files

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 219 | Module exports |
| `mixer.py` | 1,101-1,102 | Central 8-stage tick pipeline |
| `mix_bus.py` | 763-764 | Hierarchical bus tree, DSP chain |
| `bus_routing.py` | 491-492 | Aux sends, direct outputs |
| `ducking.py` | 674-675 | Envelope followers |
| `sidechain.py` | 499-500 | Compressor with soft knee |
| `hdr_audio.py` | 548-549 | Sliding audibility window |
| `mix_snapshot.py` | 675-676 | State capture/restore |
| `config.py` | 257 | Audio constants |
| `sidechain_bridge.py` | ~300 | Sidechain integration |

### 2.2 Core Components

#### MixBus
- Hierarchical bus with volume (dB/linear), pitch, mute/solo
- Filters (low-pass, high-pass) with Q factor
- DSP chain integration for effects
- Thread-safe accumulation buffers

#### BusRouter
- Pre-fader and post-fader aux sends
- Direct outputs
- Routing state management

#### DuckingManager
- Dialogue/event/focus ducking types
- Four-state FSM: IDLE -> ATTACKING -> HOLDING -> RELEASING
- Attack/hold/release timing in milliseconds
- Level-triggered activation via threshold_db

#### SidechainManager
- Compressor instances with threshold/ratio/knee/makeup gain
- Soft-knee implementation with gradual compression onset
- Gain reduction formula: `overshoot * (1 - 1/ratio)`
- Envelope following with separate attack/release rates

#### HDRAudioManager
- Priority-weighted loudness analysis
- Sliding mix window with configurable adaptation speed
- Protected sources (VO, UI) bypass HDR compression
- `map_level()`: floor -> ceiling linear mapping

#### SnapshotManager
- Mix state capture, blending transitions
- Multi-snapshot active layer support
- Weighted interpolation using `BusSnapshot.interpolate()`
- Priority-based conflict resolution
- Preset snapshots: default, combat, stealth, menu, cutscene

### 2.3 8-Stage Tick Pipeline

```
Stage 1: Compute DFS post-order processing (leaf-to-root)
Stage 2: Clear bus accumulation buffers
Stage 2b: Generate source impulses for routed sources
Stage 3: Bottom-up bus processing
Stage 4: PRE_FADER aux sends (tap raw audio)
Stage 5: Process through bus (volume + effects + filters)
Stage 5b: POST_FADER aux sends (tap processed audio)
Stage 6: Apply ducking volume adjustments
Stage 7: HDR gain adjustment + sidechain compression
Stage 8: Hard clip to [-1.0, 1.0]
```

### 2.4 Default Bus Hierarchy

```
master
+-- sfx (footsteps, weapons, impacts)
+-- music (combat, exploration)
+-- vo (dialogue, barks)
+-- ambient
+-- ui
```

---

## 3. Audio Spatial Subsystem

### 3.1 Files

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 415 | Module exports |
| `config.py` | 532 | Physics constants |
| `positioning.py` | 656 | Source types, multi-listener |
| `attenuation.py` | 520 | Distance curves |
| `spatialization.py` | 682 | Panning algorithms |
| `hrtf.py` | 546 | Binaural processing |
| `doppler.py` | 356 | Pitch shift |
| `speaker_config.py` | 515 | Speaker layouts |
| `reverb_zone.py` | 490 | RT60 zones |
| `occlusion.py` | 572 | Multi-ray detection |
| `propagation.py` | 791 | Reflection/diffraction paths |
| `materials.py` | 581 | Acoustic absorption |

### 3.2 Core Components

#### ListenerManager
- Multi-listener support (up to 4 for split-screen)
- Position, forward, up, velocity tracking
- World-to-local transform computation

#### Source Types

| Type | Description | Methods |
|------|-------------|---------|
| PointSource | Single location | get_closest_point(), get_distance(), get_direction(), is_in_range() |
| AreaSource | 2D rectangular region with normal | Same interface |
| LineSource | Path between two points | Same interface |
| VolumeSource | 3D box region with interior detection | Same interface |

#### AttenuationCurve

| Model | Formula |
|-------|---------|
| LINEAR | `1 - rolloff * normalized_distance` |
| LOGARITHMIC | `1 / (1 + rolloff * log2(distance/min))` |
| INVERSE | `min / (min + rolloff * (distance - min))` |
| INVERSE_SQUARED | `(min / distance)^2` (physically accurate) |
| CUSTOM | Designer-defined points with smoothstep interpolation |

#### ConeAttenuation
- Directional sound with inner/outer angles
- Inner cone: full volume
- Outer cone: attenuated

#### Spatializers

| Spatializer | Description |
|-------------|-------------|
| StereoPanner | Basic left/right panning |
| SurroundPanner | 5.1/7.1 channel routing |
| VBAPSpatializer | Vector Base Amplitude Panning |
| AmbisonicsSpatializer | First-order B-format encoding |
| HRTFSpatializer | Binaural with ITD/ILD |

#### VBAP Algorithm
- Speaker pair selection based on source direction
- 2D VBAP with Cartesian coordinate system
- Gain normalization: `g1*v1 + g2*v2 = source_dir` solved via determinant

#### Ambisonics (First-Order)
- B-format encoding: W (omni), Y, Z, X (ACN ordering)
- Decoder matrix generation from speaker angles
- Spread parameter reduces directional components

#### HRTF Implementation
- **Woodworth's ITD Formula**: `ITD = (r/c) * (theta + sin(theta))`
- **ILD Model**: `ILD_MAX_DB * sin(azimuth) * cos(elevation)`
- Synthetic filter generation with head shadowing simulation
- HRTFProcessingState: delay buffers, convolution state, interpolation

#### DopplerProcessor
- Classical Doppler formula with velocity-based pitch shift
- Smoothing for gradual transitions

#### OcclusionDetector
- Multi-ray geometry queries (up to OCCLUSION_MAX_RAYS)
- Ray spread pattern around direct path
- Material transmission factor integration
- Low-pass frequency scaling: `max_freq - (max_freq - min_freq) * t^2`

#### PropagationCalculator
- Direct path with occlusion check
- Image source method for reflections (up to 4 bounces)
- Uniform Theory of Diffraction (UTD) for edge diffraction
- Energy-weighted dominant direction calculation
- Propagation cache with position tolerance

#### ReverbZoneManager
- Volume triggers
- Zone blending with smoothstep fade
- RT60 presets

### 3.3 Material Database

15-16 preset materials with 6-band absorption coefficients (125Hz to 4kHz):

| Material | Category |
|----------|----------|
| CONCRETE | Hard surface |
| BRICK | Hard surface |
| WOOD | Medium surface |
| GLASS | Reflective |
| METAL | Reflective |
| CARPET | Absorptive |
| FABRIC | Absorptive |
| TILE | Hard surface |
| DRYWALL | Medium surface |
| GRASS | Outdoor |
| GRAVEL | Outdoor |
| WATER | Special |
| SNOW | Outdoor |
| ACOUSTIC_FOAM | Absorptive |
| ACOUSTIC_TILE | Absorptive |

Each material includes: absorption per band, reflection, transmission, scattering, density.

### 3.4 RT60 Calculation

- **Sabine Equation**: `RT60 = 0.161 * V / A`
- **Eyring Equation**: `RT60 = 0.161 * V / (-S * ln(1 - alpha))`
- NRC (Noise Reduction Coefficient) calculation

---

## 4. Integration Architecture

```
                    +-----------------+
                    |     Mixer       |
                    +-----------------+
                           |
       +-------------------+-------------------+
       |                   |                   |
+------+------+    +-------+-------+   +-------+-------+
| BusRouter   |    | DuckingManager|   |   HDRAudio    |
| (Aux/Direct)|    | (Envelope)    |   | (MixWindow)   |
+-------------+    +---------------+   +---------------+
       |                   |
+------+------+    +-------+-------+
| MixBus Tree |    | Sidechain     |
| (Hierarchy) |    | (Compressor)  |
+-------------+    +---------------+

                    +-----------------+
                    |   Spatial       |
                    +-----------------+
                           |
       +-------------------+-------------------+
       |                   |                   |
+------+------+    +-------+-------+   +-------+-------+
|Propagation  |    |Spatialization |   |  Occlusion   |
|(Reflections)|    |(Panning/HRTF) |   | (Multi-ray)  |
+-------------+    +---------------+   +---------------+
       |                   |                   |
+------+------+    +-------+-------+   +-------+-------+
| Materials   |    |  Attenuation  |   |  Positioning |
| (Acoustic)  |    | (Distance)    |   | (Sources)    |
+-------------+    +---------------+   +---------------+
```

---

## 5. Shared Dependencies

- **NumPy**: Audio buffer operations throughout
- **threading.RLock**: Thread-safe operations
- **DSP chain**: `../dsp/dsp_graph`
- **Filters**: `../dsp/filters` (LowPassFilter, HighPassFilter)
- **Configuration**: Local `config.py` modules

---

## 6. Configuration Constants

### Mixing Constants
- `DUCK_ATTACK_MS`, `DUCK_HOLD_MS`, `DUCK_RELEASE_MS`
- `HDR_WINDOW_DB`, `HDR_ADAPTATION_SPEED`
- `SIDECHAIN_THRESHOLD_DB`, `SIDECHAIN_RATIO`, `SIDECHAIN_KNEE_DB`

### Spatial Constants
- `SPEED_OF_SOUND` (343 m/s)
- `HEAD_RADIUS` (8.75 cm)
- `MAX_ITD_SAMPLES`
- `OCCLUSION_MAX_RAYS`
- `HRTF_FILTER_LENGTH`
- `VBAP_MAX_SPEAKERS`

---

## 7. Thread Safety

### Mixing Subsystem
- `mixer.py`: `self._lock = threading.RLock()` with context managers
- `mix_bus.py`: Separate `_lock` and `_acc_lock` for hierarchy vs audio data
- Callbacks copied before notification to avoid lock-during-callback

### Spatial Subsystem
- `OcclusionProcessor`: Maintains per-source state dictionary
- `PropagationCache`: Source-keyed cache with position tolerance
- Stateless calculations allow concurrent processing

---

## 8. Error Handling Patterns

- Division-by-zero guards: `if length < 0.0001`
- Value clamping: `max(0.0, min(1.0, x))`
- Silent exception handling for callback errors
- Graceful fallbacks (e.g., VBAP falls back to StereoPanner if < 2 speakers)

---

## 9. Factory Functions

### Mixing
- `create_mixer(**config)`
- `create_bus(name, parent=None, **config)`
- `create_snapshot(name, **config)`

### Spatial
- `create_spatializer(method, layout, **kwargs)`
- `create_attenuation(model, min_distance, max_distance, rolloff, **kwargs)`
- `create_source(source_type, position, **kwargs)`

---

## 10. Mathematical Correctness Evidence

1. **Woodworth ITD formula** matches published acoustic research
2. **Sabine/Eyring RT60** equations match standard acoustics textbooks
3. **Constant-power panning** using `cos(angle)` / `sin(angle)` is industry standard
4. **dB <-> linear conversions**: `10^(dB/20)` and `20*log10(linear)` are correct
5. **B-format Ambisonics** uses correct ACN ordering (W, Y, Z, X)
