# Investigation: engine/audio/spatial

## Summary

The spatial audio subsystem is a **REAL IMPLEMENTATION** with 6,656 lines of production-quality code implementing industry-standard algorithms. It provides comprehensive 3D positioning, HRTF binaural processing with Woodworth ITD formula, VBAP/Ambisonics spatialization, multi-ray occlusion detection, simplified UTD diffraction, and RT60-based reverb zones with Sabine/Eyring equations.

## Files

| File | Lines | Status | Notes |
|------|-------|--------|-------|
| __init__.py | 415 | REAL | Comprehensive exports, all modules connected |
| config.py | 532 | REAL | Physics constants (343 m/s sound, head radius 8.75cm) |
| positioning.py | 656 | REAL | Point/Area/Line/Volume sources, multi-listener |
| attenuation.py | 520 | REAL | Linear/Log/Inverse/InverseSquared/Custom curves |
| spatialization.py | 682 | REAL | Stereo panning, VBAP, Ambisonics (B-format) |
| hrtf.py | 546 | REAL | ITD/ILD, Woodworth formula, filter convolution |
| doppler.py | 356 | REAL | Classical Doppler formula with smoothing |
| speaker_config.py | 515 | REAL | 5.1/7.1/Atmos layouts, channel routing |
| reverb_zone.py | 490 | REAL | RT60 presets, zone blending, smoothstep fade |
| occlusion.py | 572 | REAL | Multi-ray detection, low-pass response |
| propagation.py | 791 | REAL | Image source reflections, UTD diffraction |
| materials.py | 581 | REAL | 6-band absorption, NRC, Sabine/Eyring RT60 |

## Spatial Audio Components

- **ListenerManager**: Multi-listener support (up to 4 for split-screen)
- **ListenerState**: Position, forward, up, velocity, world-to-local transform
- **SpatialSource**: Abstract base with Point/Area/Line/Volume implementations
- **AttenuationCurve**: 6 models (Linear, Log, Inverse, InverseSquared, Custom, None)
- **ConeAttenuation**: Directional sound with inner/outer angles
- **Spatializer**: Base class with StereoPanner, SurroundPanner, VBAPSpatializer, AmbisonicsSpatializer
- **HRTFSpatializer**: Binaural with ITD/ILD, synthetic filter generation
- **DopplerProcessor**: Velocity-based pitch shift with smoothing
- **OcclusionDetector**: Multi-ray geometry queries
- **PropagationCalculator**: Direct/reflection/diffraction paths
- **ReverbZoneManager**: Volume triggers, zone blending, presets
- **MaterialDatabase**: 16 preset materials, frequency-dependent absorption

## Implementation

- Real 3D positioning? **YES** - Full Vec3 math, distance calculations, world-to-listener transforms
- Real HRTF? **YES** - Woodworth ITD formula, ILD based on azimuth/elevation, delay buffer convolution
- Real reverb/occlusion? **YES** - RT60 with Sabine equation, multi-ray occlusion, low-pass filtering
- Real distance attenuation? **YES** - Inverse square law, logarithmic, custom curves with rolloff

## Verdict

**REAL IMPLEMENTATION**

This is production-quality spatial audio code implementing:
1. Industry-standard HRTF with Woodworth's spherical head model for ITD
2. VBAP and first-order Ambisonics (B-format W,X,Y,Z)
3. Physically accurate inverse-square attenuation
4. Material-aware propagation with frequency-dependent absorption (6 octave bands)
5. Simplified UTD diffraction around edges
6. Multi-ray occlusion with transmission coefficients
7. Sabine and Eyring equations for RT60 estimation

## Evidence

### Woodworth ITD Formula (hrtf.py:62-93)
```python
def calculate_itd(azimuth: float, head_radius: float = HEAD_RADIUS, sample_rate: int = HRTF_SAMPLE_RATE) -> int:
    """Uses Woodworth's formula for spherical head model:
    ITD = (r/c) * (theta + sin(theta))
    """
    theta = math.radians(az_clamped)
    itd_seconds = (head_radius / speed_of_sound) * (theta + math.sin(theta))
    itd_samples = int(round(itd_seconds * sample_rate))
```

### Inverse Square Law (attenuation.py:199-226)
```python
class InverseSquaredAttenuation(AttenuationCurve):
    """Physically accurate for point sources: volume = (min_distance / distance)^2"""
    def calculate(self, distance: float) -> float:
        ratio = self._min_distance / distance
        attenuation = ratio * ratio * self._rolloff
```

### Ambisonics B-Format Encoding (spatialization.py:465-487)
```python
def encode(self, params: SpatializationParams) -> List[float]:
    """First-order B-format (ACN ordering: W, Y, Z, X)"""
    w = params.gain / math.sqrt(2)
    y = params.gain * math.sin(az_rad) * math.cos(el_rad)
    z = params.gain * math.sin(el_rad)
    x = params.gain * math.cos(az_rad) * math.cos(el_rad)
```

### Sabine RT60 Equation (materials.py:440-472)
```python
def calculate_room_rt60(self, volume: float, surface_areas: Dict[str, float]) -> float:
    """RT60 = 0.161 * V / A (Sabine equation)"""
    rt60 = 0.161 * volume / total_absorption
```

### Multi-Ray Occlusion (occlusion.py:162-244)
```python
def detect(self, source_pos: Vec3, listener_pos: Vec3, settings) -> OcclusionResult:
    ray_origins = self._generate_ray_origins(source_pos, listener_pos, num_rays, spread)
    for origin in ray_origins:
        hit = self._raycast_func(origin, direction)
        if hit is not None and hit.hit and hit.distance < distance:
            blocked += 1
            total_transmission += hit.transmission
    effective_occlusion = occlusion_factor * (1.0 - avg_transmission)
```
