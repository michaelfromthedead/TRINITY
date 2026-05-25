# PEDAGOGY: Audio Mixing and Spatial Concept Evolution

**RDC Archaeological Record**
**Append-Only Log of Concept Evolution**

---

## Pass 1: engine_audio_mixing.md

### Concept: 8-Stage Tick Pipeline
- **Prior Value**: (none - new concept)
- **New Value**: Complete 8-stage pipeline with DFS ordering, bus processing, aux sends, ducking, HDR, and hard clipping
- **Reason**: First encounter in source docs; establishes the core mixing architecture

### Concept: Sidechain Compression
- **Prior Value**: (none - new concept)
- **New Value**: Soft-knee compressor with gain reduction formula `overshoot * (1 - 1/ratio)`, attack/release envelopes
- **Reason**: First detailed specification of compression algorithm

### Concept: HDR Audio
- **Prior Value**: (none - new concept)
- **New Value**: Priority-weighted loudness analysis with sliding mix window, protected sources (VO, UI)
- **Reason**: First encounter; unique dynamic range management approach

### Concept: Mix Snapshots
- **Prior Value**: (none - new concept)
- **New Value**: Weighted interpolation, multi-snapshot layers, preset snapshots (default, combat, stealth, menu, cutscene)
- **Reason**: First specification of state capture/restore system

### Concept: Default Bus Hierarchy
- **Prior Value**: (none - new concept)
- **New Value**: master -> sfx, music, vo, ambient, ui
- **Reason**: First specification of standard hierarchy

### Concept: Thread Safety (Mixing)
- **Prior Value**: (none - new concept)
- **New Value**: RLock with separate locks for hierarchy vs audio data, callbacks copied before notification
- **Reason**: First documentation of thread-safety patterns

---

## Pass 2: engine_audio_spatial.md

### Concept: HRTF Implementation
- **Prior Value**: (none - new concept)
- **New Value**: Woodworth ITD formula `ITD = (r/c) * (theta + sin(theta))`, ILD model with azimuth/elevation
- **Reason**: First detailed HRTF algorithm specification

### Concept: Attenuation Models
- **Prior Value**: (none - new concept)
- **New Value**: 6 models - LINEAR, LOGARITHMIC, INVERSE, INVERSE_SQUARED, CUSTOM, NONE
- **Reason**: First comprehensive attenuation documentation

### Concept: Spatialization Methods
- **Prior Value**: (none - new concept)
- **New Value**: Stereo, Surround, VBAP, Ambisonics (first-order B-format)
- **Reason**: First specification of all supported methods

### Concept: Source Types
- **Prior Value**: (none - new concept)
- **New Value**: PointSource, AreaSource, LineSource, VolumeSource with common interface
- **Reason**: First documentation of source type hierarchy

### Concept: Multi-Ray Occlusion
- **Prior Value**: (none - new concept)
- **New Value**: Configurable ray count, spread pattern, material transmission integration
- **Reason**: First occlusion algorithm specification

### Concept: RT60 Calculation
- **Prior Value**: (none - new concept)
- **New Value**: Sabine `RT60 = 0.161 * V / A` and Eyring equations
- **Reason**: First reverb time calculation documentation

### Concept: Material Database
- **Prior Value**: (none - new concept)
- **New Value**: 16 preset materials with 6-band absorption coefficients (125Hz to 4kHz)
- **Reason**: First material system specification

---

## Pass 3: engine_audio_mixing_spatial.md

### Concept: File Count (Mixing)
- **Prior Value**: 7-10 files
- **New Value**: 7 files (consolidated count from combined investigation)
- **Reason**: Combined document provides authoritative file inventory

### Concept: Total Lines
- **Prior Value**: Mixing ~5,020, Spatial ~6,656
- **New Value**: Mixing ~5,020, Spatial ~4,880, Total ~9,900
- **Reason**: Combined document reconciles line counts; spatial count differs slightly between standalone and combined docs

### Concept: Integration Architecture
- **Prior Value**: (none - new concept)
- **New Value**: Detailed ASCII diagram showing Mixer coordinating BusRouter/DuckingManager/HDRAudio, and Spatial coordinating Propagation/Spatialization/Occlusion
- **Reason**: Combined document provides integration view not present in standalone investigations

### Concept: Algorithm Count
- **Prior Value**: Mixing 8+, Spatial 12+
- **New Value**: Consolidated total of 20+ algorithms
- **Reason**: Combined document provides aggregate count

### Concept: Material Count
- **Prior Value**: 16 presets
- **New Value**: 15 presets (slight discrepancy between standalone and combined docs)
- **Reason**: Combined document lists 15 specific materials

### Concept: Propagation Algorithm
- **Prior Value**: Mentioned but not detailed in standalone spatial doc
- **New Value**: Image source method (up to 4 bounces), UTD diffraction, energy-weighted dominant direction, propagation cache
- **Reason**: Combined document provides fuller algorithm specification

### Concept: Shared Dependencies
- **Prior Value**: Individual module dependencies
- **New Value**: Unified view: NumPy for audio buffers, threading.RLock, DSP chain, filters from `../dsp/`
- **Reason**: Combined document synthesizes common dependencies

---

## Summary

| Pass | Source Doc | New Concepts | Updated Concepts |
|------|------------|--------------|------------------|
| 1 | engine_audio_mixing.md | 6 | 0 |
| 2 | engine_audio_spatial.md | 7 | 0 |
| 3 | engine_audio_mixing_spatial.md | 3 | 4 |
| **Total** | | **16** | **4** |
