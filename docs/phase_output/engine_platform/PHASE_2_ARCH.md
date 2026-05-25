# PHASE 2 ARCHITECTURE: Audio Platform Backends

## Phase Overview

Phase 2 completes the audio subsystem by implementing platform-specific backends (WASAPI, CoreAudio, ALSA) and completing spatial audio features (Doppler effect, cone attenuation).

## Current State (from Investigation)

| Component | Status | Lines |
|-----------|--------|-------|
| AudioDevice abstraction | REAL | 445 |
| NullAudioBackend | REAL | 142 |
| SpatialAudioEngine | REAL | 272 |
| BackendRegistry | REAL | 70 |
| Standalone null_backend.py | REAL | 225 |

**Missing:**
- Platform backends (WASAPI, CoreAudio, ALSA)
- Doppler effect implementation
- Cone attenuation implementation
- Reverb DSP pipeline

## Architectural Decisions

### ADR-P2-001: Platform Backend Strategy

**Status:** Proposed

**Context:**
Audio backends require native library access. Options:
1. ctypes/cffi to native APIs
2. sounddevice (PortAudio wrapper)
3. miniaudio-py
4. Custom native extensions

**Decision:**
Use sounddevice (PortAudio) as the primary cross-platform backend:

```python
import sounddevice as sd

class PortAudioBackend(AudioBackend):
    def enumerate_devices(self) -> list[AudioDeviceInfo]:
        return [
            AudioDeviceInfo(
                id=str(d["index"]),
                name=d["name"],
                device_type=AudioDeviceType.PLAYBACK if d["max_output_channels"] > 0 else AudioDeviceType.CAPTURE,
                sample_rates=[int(d["default_samplerate"])],
                channels=d["max_output_channels"] or d["max_input_channels"],
                is_default=d["index"] == sd.default.device[0] or d["index"] == sd.default.device[1]
            )
            for d in sd.query_devices()
        ]
```

**Consequences:**
- PortAudio handles WASAPI/CoreAudio/ALSA/JACK automatically
- Single dependency, well-maintained
- Callback model matches our abstraction
- Optional: Add raw backends for advanced features (low latency, spatial APIs)

### ADR-P2-002: Spatial Audio Platform Integration

**Status:** Proposed

**Context:**
Platform spatial APIs (Windows Sonic, Tempest 3D, Apple Spatial) provide hardware-accelerated HRTF. Our SpatialAudioEngine does software spatialization.

**Decision:**
Two-tier spatial audio:

1. **Software Tier:** SpatialAudioEngine (current implementation)
   - Always available
   - Cross-platform
   - Good for stereo output

2. **Platform Tier:** PlatformSpatialAudio (new)
   - Detects and uses platform APIs
   - Falls back to software tier
   - Required for true 3D audio (headphones, surround)

```python
class PlatformSpatialAudio:
    @staticmethod
    def current_api() -> SpatialAudioAPI:
        if _windows_sonic_available():
            return SpatialAudioAPI.WINDOWS_SONIC
        if _tempest_available():
            return SpatialAudioAPI.TEMPEST_3D
        if _apple_spatial_available():
            return SpatialAudioAPI.APPLE_SPATIAL
        return SpatialAudioAPI.NONE
```

**Consequences:**
- Existing code keeps working (software fallback)
- Platform features available when present
- Clear upgrade path

### ADR-P2-003: Doppler Effect Implementation

**Status:** Proposed

**Context:**
SpatialSource and SpatialListener have velocity fields but no Doppler implementation.

**Decision:**
Implement Doppler as pitch shift in SpatialAudioEngine:

```python
SPEED_OF_SOUND = 343.0  # m/s

def calculate_doppler_pitch(self, source: SpatialSource, listener: SpatialListener) -> float:
    # Relative velocity along listener-source axis
    to_source = source.position.sub(listener.position).normalize()
    listener_velocity_along = listener.velocity.dot(to_source)
    source_velocity_along = source.velocity.dot(to_source)

    # Doppler formula: f' = f * (v + vl) / (v + vs)
    # v = speed of sound, vl = listener velocity toward source, vs = source velocity toward listener
    denominator = SPEED_OF_SOUND - source_velocity_along
    if denominator <= 0:
        return 2.0  # Clamp at 2x pitch for approaching sources
    return (SPEED_OF_SOUND + listener_velocity_along) / denominator
```

**Consequences:**
- Pitch-shift-based Doppler (standard approach)
- Clamp to prevent singularity at speed of sound
- Requires pitch-shifting DSP in audio pipeline

### ADR-P2-004: Cone Attenuation Implementation

**Status:** Proposed

**Context:**
SpatialSource has inner_cone_angle and outer_cone_angle fields, unused in current attenuation calculation.

**Decision:**
Extend calculate_attenuation to include cone:

```python
def calculate_cone_attenuation(self, source: SpatialSource, listener: SpatialListener) -> float:
    to_listener = listener.position.sub(source.position).normalize()
    angle = math.acos(source.direction.dot(to_listener)) * (180.0 / math.pi)

    if angle < source.inner_cone_angle / 2:
        return 1.0  # Inside inner cone
    elif angle > source.outer_cone_angle / 2:
        return source.outer_cone_volume  # Outside outer cone
    else:
        # Linear interpolation between inner and outer
        t = (angle - source.inner_cone_angle / 2) / (source.outer_cone_angle / 2 - source.inner_cone_angle / 2)
        return 1.0 - t * (1.0 - source.outer_cone_volume)
```

**Consequences:**
- Directional sources work correctly
- Requires source.direction field (add if missing)
- outer_cone_volume controls sound leakage

## Component Diagram

```
engine/platform/audio/
    |
    +-- audio_device.py          # AudioDevice, AudioBackend (ABC)
    |
    +-- spatial.py               # SpatialAudioEngine (enhanced with Doppler/cone)
    |
    +-- platform_spatial.py      # NEW: Platform spatial API detection
    |
    +-- backends/
            |
            +-- __init__.py      # BackendRegistry
            +-- null_backend.py  # NullAudioBackend
            +-- portaudio.py     # NEW: PortAudioBackend (sounddevice)
            +-- wasapi.py        # FUTURE: Low-latency Windows
            +-- coreaudio.py     # FUTURE: Low-latency macOS
            +-- alsa.py          # FUTURE: Low-latency Linux
```

## Data Flow

### Audio Playback Path

```
SpatialAudioEngine
       |
       +-- calculate_attenuation(source, listener)
       +-- calculate_doppler_pitch(source, listener)
       +-- calculate_cone_attenuation(source, listener)
       |
       v
Apply to source audio buffer (gain, pan, pitch)
       |
       v
Mix sources to output buffer
       |
       v
AudioStream.callback(input, frames) -> output
       |
       v
PortAudioBackend submits to hardware
```

### Platform Spatial Path (when available)

```
PlatformSpatialAudio.current_api()
       |
       +-- WINDOWS_SONIC: Use ISpatialAudioClient
       +-- TEMPEST_3D: Use PS5 Audio SDK
       +-- APPLE_SPATIAL: Use AVAudioEnvironmentNode
       +-- NONE: Fall back to SpatialAudioEngine
```

## File Changes Required

### New Files

| File | Purpose |
|------|---------|
| engine/platform/audio/backends/portaudio.py | PortAudio backend via sounddevice |
| engine/platform/audio/platform_spatial.py | Platform spatial API detection |

### Modified Files

| File | Changes |
|------|---------|
| engine/platform/audio/spatial.py | Add Doppler, cone attenuation, direction field |
| engine/platform/audio/__init__.py | Export new classes |
| engine/platform/constants.py | Add SPEED_OF_SOUND, cone defaults |

## Dependencies

### Python Packages

| Package | Version | Purpose |
|---------|---------|---------|
| sounddevice | >=0.4.6 | PortAudio wrapper |
| numpy | (existing) | Audio buffer handling |

### Native Libraries

| Library | Platforms | Notes |
|---------|-----------|-------|
| PortAudio | All | Installed with sounddevice wheel |

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| sounddevice not available | Fall back to null backend |
| Platform spatial APIs undocumented | Start with software tier, add platform tier later |
| Doppler pitch shift artifacts | Use high-quality resampler (e.g., scipy.signal.resample) |
| Cone math edge cases | Clamp angles, handle zero direction vector |

## Phase Exit Criteria

1. PortAudioBackend passes all AudioBackend tests
2. Doppler effect works with moving sources
3. Cone attenuation works with directional sources
4. Platform spatial detection returns correct API per platform
5. All existing tests pass
6. No performance regression in audio callback
