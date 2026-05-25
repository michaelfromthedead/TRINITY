# Phase 6 Architecture: Acoustic Simulation

## Purpose
Environmental acoustics: algorithmic reverb (Schroeder/convolution), early reflections, diffraction, transmission through materials, acoustic material database, and reverb zone integration with the mixer.

## Current Implementation
**10/15 tasks complete [x], 5 partial [~], 0 missing.**

### Algorithmic Reverb (`dsp/reverb.py`) [x]
- `CombFilter`: delay line + feedback + LPF damping, configurable delay length/feedback/damping
- `AllPassFilterReverb`: all-pass delay structure for diffusion
- `Freeverb`: Schroeder-Moorer topology (8 comb + 4 all-pass), stereo spread=23 samples
- Configurable: room size, damping, wet/dry, width, freeze mode
- `PlateReverb`: plate reverb algorithm, different tonal character
- `SimpleReverb`: basic configurable reverb
- `REVERB_PRESETS`: ROOM, HALL, CHURCH, PLATE, SPRING, CAVE, ARENA, OUTDOORS

### Convolution Reverb (`dsp/reverb.py`) [x]
- `ConvolutionReverb`: IR loading from file, partitioned convolution via FFT
- Configurable wet/dry mix, IR duration limit
- Can load external IR files (OpenAIR, EchoThief formats)

### Reverb Zones (`spatial/reverb_zone.py`, 490 lines) [x]
- `ReverbZone`: position (Vec3), size (Vec3), shape (BOX/SPHERE), rotation (Quaternion), priority (0-100)
- `ReverbZoneState`: ACTIVE/INACTIVE with distance-based fade progress
- `ReverbParameters`: room_size, damping, wet/dry, width, freeze, diffusion, low_cut, high_cut
- `REVERB_PRESET_PARAMS`: maps preset names to ReverbParameters
- `ReverbZoneManager`: `register_zone()`, `unregister_zone()`, `get_zone_at(listener_pos)`, priority-based selection
- `@reverb_zone(preset, fade_distance)` decorator in `audio_extended.py`

### Propagation (`spatial/propagation.py`, 791 lines) [x]
- `PropagationCalculator`: full path computation pipeline
- `ReflectionSurface`: absorption coefficient per frequency band, image source method
- `MAX_REFLECTION_ORDER=3` (1st-3rd order reflections)
- `DiffractionEdge`: edge detection, `DIFFRACTION_ANGLE_THRESHOLD`, `UTD_PATH_DECAY_FACTOR`
- Transmission: `PathType.TRANSMISSION`, `TRANSMISSION_MAX_THICKNESS`/`MIN_THICKNESS`
- Material-aware: `get_transmission_loss_db()` uses material coefficients
- `PropagationCache`: LRU caching with configurable max_size

### Acoustic Materials (`spatial/materials.py`, 581 lines) [x]
- `MaterialType` enum: 12 types (CONCRETE, WOOD, GLASS, CARPET, CURTAIN, BRICK, METAL, WATER, PLASTER, ACOUSTIC_TILE, SOIL, GRASS)
- `AcousticMaterial`: absorption/reflection/scattering/transmission coefficients
- 6 frequency bands: 125Hz, 250Hz, 500Hz, 1kHz, 2kHz, 4kHz
- `MATERIAL_PRESETS`: complete coefficient data for all 12 types
- `MaterialDatabase`: `get_material()`, `get_material_names()`, `create_custom_material()`
- Helper: `calculate_absorption_area(surface_count, absorption_coeff)`

### Architecture
```
Reverb Pipeline:
  Dry voice signal -> mix blend (based on zone/wet-dry) -> Wet path
  Wet path options:
    Freeverb (Schroeder): 8 comb stages -> 4 all-pass -> stereo output
    Convolution: FFT partitioned -> IR convolution -> wet/dry mix
    Simple: basic delay + feedback + damping

Reverb Zone System:
  Zone has: position, size, shape, preset name, fade_distance
  Listener position queries zone manager -> get active zone(s)
  Distance from zone center -> fade progress -> blended reverb params

Propagation Pipeline:
  Source position -> geometry query ->
    Direct path (line-of-sight check)
    Reflections (1st-3rd order image source)
    Diffraction (edge-based UTD approximation)
    Transmission (through-wall with material loss)
  Cached in PropagationCache per source-listener pair

Acoustic Material Model per frequency band:
  Material = { absorption: [125Hz, 250Hz, 500Hz, 1kHz, 2kHz, 4kHz],
               reflection: [...],
               scattering: [...],
               transmission: [...] }
  Concrete: high reflection, low absorption (hard surface)
  Carpet:  low reflection, high absorption (soft surface)
  Water:   unique reflection/transmission characteristics
```

### Partial (5 tasks)
| Task | Component | Gap |
|------|-----------|-----|
| T-AU-6.1 | FDN reverb | Freeverb uses Schroeder topology, not true FDN with feedback matrix |
| T-AU-6.6 | Real-time IR swap | ConvolutionReverb supports IR reloading, crossfade not implemented |
| T-AU-6.7 | Hybrid reverb | Early reflections + convolution tail available as separate components, no combined processor |
| T-AU-6.13 | Dry/wet distance blend | SimpleReverb has configurable mix, not wired to zone distance blending |
| T-AU-6.15 | Baked occlusion | `OcclusionMethod.BAKED` defined, precomputed zone-pair table not implemented |
