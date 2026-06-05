# Phase 5 Architecture: Spatial Audio System

## Purpose
Full 3D audio spatialization pipeline: attenuation curves, source positioning (point/area/line/volume), spatialization methods (panning/HRTF/VBAP/ambisonics), occlusion, doppler, and per-frame spatial update.

## Current Implementation
**14/15 tasks complete [x], 1 partial [~].** The `spatial/` subsystem (11 files, ~6,641 lines) was entirely missed by the original TODO.

### Attenuation (`spatial/attenuation.py`, 520 lines) [x]
- 6 curve types: `LinearAttenuation`, `LogarithmicAttenuation`, `InverseAttenuation`, `InverseSquaredAttenuation`, `NoAttenuation`, `CustomCurveAttenuation` (spline with `CurvePoint` list)
- `ConeAttenuation`: directional sources with inner/outer cone angles
- `AttenuationVolume`: combined curve sets
- Presets via `get_attenuation_preset()`

### Positioning (`spatial/positioning.py`, 656 lines) [x]
- `SpatialSource` base class with state machine (ACTIVE/PAUSED/CULLED)
- `PointSource`: position/direction/distance/elevation/azimuth calculation
- `AreaSource`: rectangular bounds with random point selection for closest-point
- `LineSource`: line segment with nearest-point projection
- `VolumeSource`: 3D volume with random interior point
- `ListenerManager`: multiple listeners with `ListenerState` (position/forward/up/velocity)
- `create_source()` factory function

### Spatialization (`spatial/spatialization.py`, 682 lines) [x]
- `StereoPanner`: equal-power panning (sin/cos law), configurable `PANNING_LAW_DB`
- `SurroundPanner`: multi-channel speaker output with `SurroundMode`
- `VBAPSpatializer`: speaker triplet selection, gain computation, `VBAP_MAX_SPEAKERS` config
- `AmbisonicsSpatializer`: ambisonic encoding/decoding, configurable order, speaker layout routing
- `NoSpatializer`: passthrough
- `SpatializationParams`: direction (azimuth/elevation/distance) from source-to-listener vector
- `SpatializationResult`: per-channel gains vector
- `ChannelGains`: indexed channel gain access

### HRTF (`spatial/hrtf.py`, 546 lines) [x]
- `HRTFSpatializer`: full binaural spatialization via `process_hrtf_block()`
- `calculate_itd()`: Woodworth's spherical head model (ITD = (head_radius/c) * (arcsin(cos(elevation)*sin(azimuth)) + cos(elevation)*sin(azimuth)))
- `calculate_ild()`: frequency-dependent level differences based on azimuth/elevation
- `HRTFProfile`: filter coefficients, can accept measured data
- `create_default_hrtf_profile()`: generates analytic HRTF coefficients

### Doppler (`spatial/doppler.py`, 356 lines) [x]
- `DopplerProcessor`: `calculate_doppler_shift(speed_of_sound, source_vel, listener_vel)`
- `DopplerConfig`: `MAX_DOPPLER_SHIFT=10.0`, `DOPPLER_SMOOTHING_TIME=0.05`
- 6 presets: Realistic, Exaggerated, Cinematic, Subtle, Maximum, Off

### Occlusion (`spatial/occlusion.py`, 572 lines) [x]
- `OcclusionType`: NONE, PARTIAL, FULL, OBSTRUCTION
- `OcclusionDetector`: configurable ray count (1-64), `RaycastHit`, material-aware processing
- `OcclusionProcessor`: multi-ray fractional occlusion, interpolated values
- `OCCLUSION_MAX_RAYS=64`, `OCCLUSION_DEFAULT_FILTER_FREQUENCY_Hz=4000`
- `OcclusionMethod`: RAYCAST, PROPAGATION, BAKED
- `@occlusion(method, max_occlusion)` decorator in `audio_extended.py`

### Propagation (`spatial/propagation.py`, 791 lines) [x]
- `PropagationCalculator`: reflection/diffraction/transmission paths
- `ReflectionSurface`: absorption coefficient, image source method, `MAX_REFLECTION_ORDER=3`
- `DiffractionEdge`: Huygens-Fresnel approximation, `DIFFRACTION_ANGLE_THRESHOLD`, `UTD_PATH_DECAY_FACTOR`
- `PathType`: DIRECT, REFLECTION, DIFFRACTION, TRANSMISSION
- `PropagationCache`: LRU cached propagation results with configurable size
- Material-aware transmission loss via `get_transmission_loss_db()`

### Reverb Zones (`spatial/reverb_zone.py`, 490 lines) [x]
- `ReverbZone`: position, size, shape (box/sphere), rotation, priority
- `ReverbZoneState`: active/inactive with distance-based fade
- `ReverbZoneManager`: zone registration, detection by listener position, priority-based selection
- `@reverb_zone(preset, fade_distance)` decorator in `audio_extended.py`

### Speaker Configs (`spatial/speaker_config.py`, 515 lines) [x]
- `SpeakerConfiguration`: layout name, speaker positions, channel mapping
- `SpeakerPosition`: azimuth, elevation, radius
- `ChannelRouter`: input-to-output channel mapping matrix
- `VirtualSpeaker`: phantom speaker generation
- Support: stereo, quad, surround 5.1/7.1, Atmos, custom layouts

### Materials (`spatial/materials.py`, 581 lines) [x]
- `MaterialType` enum: CONCRETE, WOOD, GLASS, CARPET, CURTAIN, BRICK, METAL, WATER, PLASTER, ACOUSTIC_TILE, SOIL, GRASS
- `AcousticMaterial`: absorption/reflection/scattering/transmission per frequency band
- 6 frequency bands: 125Hz, 250Hz, 500Hz, 1kHz, 2kHz, 4kHz
- `MaterialDatabase`: `get_material()`, `create_custom_material()`, `calculate_absorption_area()`

### Decorators [x]
- `@spatial_audio(falloff, max_distance)` in `trinity/decorators/audio.py`: validated, TAG+REGISTER
- `@occlusion(method, max_occlusion)` in `trinity/decorators/audio_extended.py`: validated
- `@reverb_zone(preset, fade_distance)` in `trinity/decorators/audio_extended.py`: validated

### Architecture
```
Per-Frame Spatial Pipeline:
  1. ListenerManager updates listener position/velocity
  2. For each spatial source:
     a. Calculate direction (azimuth/elevation/distance from listener)
     b. Apply attenuation curve (distance-based)
     c. Apply occlusion (raycast -> frequency-band attenuation)
     d. Apply doppler shift (source/listener velocity differential)
     e. Spatialize: HRTF (headphones) or VBAP (speakers) or Ambisonics
     f. Return SpatializationResult with per-channel gains + pan

Spatialization Method Selection:
  Headphones -> HRTF (analytic or measured SOFA)
  Stereo -> StereoPanner (equal-power sin/cos)
  Multi-speaker -> VBAP (2.0, 5.1, 7.1) or Surround
  Ambisonics -> FOA/HOA encoding -> decode to speaker layout

Occlusion Pipeline:
  Physics raycast(s): number of rays = distance * OCCLUSION_RAY_DENSITY
  Fraction blocked -> filter frequency cutoff
  Material-aware: absorbed frequencies differ by material type
  Propagation: reflections/diffraction/transmission computed separately
```

### Partial (1 task)
| Task | Gap |
|------|-----|
| T-AU-5.10 | HOA ambisonics N=3 configured but higher-order decoding not fully validated |
