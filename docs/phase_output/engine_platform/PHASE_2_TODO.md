# PHASE 2 TODO: Audio Platform Backends

## Summary

Complete audio subsystem with platform backends and spatial audio enhancements.

**Estimated Effort:** 12-16 hours
**Dependencies:** Phase 1 complete
**Blocking:** None (audio is independent)

---

## Tasks

### T-P2-001: Create PortAudio Backend

**Priority:** P0 (Blocking)
**Estimate:** 4 hours

Create `engine/platform/audio/backends/portaudio.py`:

```python
class PortAudioBackend(AudioBackend):
    def enumerate_devices(self) -> list[AudioDeviceInfo]: ...
    def default_device(self, device_type: AudioDeviceType) -> AudioDeviceInfo | None: ...
    def open_stream(self, device_id: str, ...) -> AudioStream: ...
    def close_stream(self, stream: AudioStream) -> None: ...
```

**Acceptance Criteria:**
- [ ] Device enumeration returns real hardware devices
- [ ] Stream creation succeeds with default device
- [ ] Callback receives audio data at expected rate
- [ ] Close stream stops callback thread
- [ ] Falls back to null backend if sounddevice unavailable

---

### T-P2-002: Register PortAudio Backend

**Priority:** P0 (Blocking)
**Estimate:** 30 minutes

Update `engine/platform/audio/backends/__init__.py`:

```python
try:
    from .portaudio import PortAudioBackend
    register_backend("portaudio", PortAudioBackend, set_default=True)
except ImportError:
    pass  # sounddevice not available

# Null backend remains fallback
register_backend("null", NullAudioBackend, set_default=False)
```

**Acceptance Criteria:**
- [ ] PortAudio is default when available
- [ ] Null backend available as fallback
- [ ] ImportError handled gracefully

---

### T-P2-003: Implement Doppler Effect

**Priority:** P1 (Important)
**Estimate:** 3 hours

Add to `engine/platform/audio/spatial.py`:

```python
SPEED_OF_SOUND = 343.0  # Add to constants.py

def calculate_doppler_pitch(self, source: SpatialSource, listener: SpatialListener) -> float:
    ...
```

**Acceptance Criteria:**
- [ ] Approaching sources have higher pitch (ratio > 1.0)
- [ ] Receding sources have lower pitch (ratio < 1.0)
- [ ] Stationary sources have normal pitch (ratio = 1.0)
- [ ] Supersonic sources clamped (no singularity)
- [ ] Unit tests cover all cases

---

### T-P2-004: Implement Cone Attenuation

**Priority:** P1 (Important)
**Estimate:** 2 hours

Add to `engine/platform/audio/spatial.py`:

```python
# Add to SpatialSource dataclass:
direction: Vec3 = field(default_factory=lambda: Vec3(0, 0, -1))
outer_cone_volume: float = 0.0

def calculate_cone_attenuation(self, source: SpatialSource, listener: SpatialListener) -> float:
    ...
```

**Acceptance Criteria:**
- [ ] Listener inside inner cone: full volume
- [ ] Listener outside outer cone: outer_cone_volume
- [ ] Listener between cones: interpolated
- [ ] Zero direction vector handled (default to omnidirectional)
- [ ] Unit tests cover all cases

---

### T-P2-005: Integrate Doppler and Cone into Attenuation

**Priority:** P1 (Important)
**Estimate:** 1 hour

Modify `calculate_attenuation` to return composite result:

```python
@dataclass
class SpatialResult:
    gain: float      # Distance + cone attenuation
    pan_left: float
    pan_right: float
    pitch: float     # Doppler pitch ratio

def calculate_spatial(self, source: SpatialSource, listener: SpatialListener) -> SpatialResult:
    ...
```

**Acceptance Criteria:**
- [ ] SpatialResult combines all effects
- [ ] Backward compatible (old API still works)
- [ ] Performance acceptable (< 1us per source)

---

### T-P2-006: Create Platform Spatial Detection

**Priority:** P2 (Nice to have)
**Estimate:** 2 hours

Create `engine/platform/audio/platform_spatial.py`:

```python
class PlatformSpatialAudio:
    @staticmethod
    def current_api() -> SpatialAudioAPI: ...

    @staticmethod
    def is_available(api: SpatialAudioAPI) -> bool: ...
```

**Acceptance Criteria:**
- [ ] Windows: Detects Windows Sonic if available
- [ ] macOS: Detects Apple Spatial if available
- [ ] Other: Returns NONE
- [ ] Detection is fast (< 10ms)

---

### T-P2-007: Add Audio Constants

**Priority:** P0 (Blocking)
**Estimate:** 15 minutes

Add to `engine/platform/constants.py`:

```python
# Audio - Spatial
SPEED_OF_SOUND = 343.0  # m/s at 20C
DOPPLER_MAX_PITCH = 4.0  # Clamp for supersonic
DOPPLER_MIN_PITCH = 0.25

# Audio - Cone defaults
CONE_DEFAULT_INNER_ANGLE = 360.0  # Omnidirectional
CONE_DEFAULT_OUTER_ANGLE = 360.0
CONE_DEFAULT_OUTER_VOLUME = 0.0
```

**Acceptance Criteria:**
- [ ] Constants have docstrings
- [ ] Used by spatial.py

---

### T-P2-008: Write PortAudio Backend Tests

**Priority:** P0 (Blocking)
**Estimate:** 2 hours

Create `tests/platform/audio/test_portaudio.py`:

```python
@pytest.mark.skipif(not SOUNDDEVICE_AVAILABLE, reason="sounddevice not installed")
class TestPortAudioBackend:
    def test_enumerate_devices(self): ...
    def test_default_device(self): ...
    def test_open_close_stream(self): ...
    def test_callback_invoked(self): ...
```

**Acceptance Criteria:**
- [ ] Tests skip gracefully if sounddevice unavailable
- [ ] All AudioBackend contract tests pass
- [ ] CI can run without audio hardware (skip or mock)

---

### T-P2-009: Write Spatial Enhancement Tests

**Priority:** P1 (Important)
**Estimate:** 1.5 hours

Update `tests/platform/audio/test_spatial.py`:

```python
def test_doppler_approaching_source(): ...
def test_doppler_receding_source(): ...
def test_doppler_stationary_source(): ...
def test_cone_inside_inner(): ...
def test_cone_outside_outer(): ...
def test_cone_between(): ...
def test_spatial_result_combined(): ...
```

**Acceptance Criteria:**
- [ ] All new methods have tests
- [ ] Edge cases covered (zero velocity, zero direction)
- [ ] Tests run without audio hardware

---

## Task Dependency Graph

```
T-P2-007 (Constants)
    |
    +-- T-P2-003 (Doppler)
    |       |
    |       +-- T-P2-005 (Integration)
    |               |
    |               +-- T-P2-009 (Spatial Tests)
    |
    +-- T-P2-004 (Cone)
            |
            +-- T-P2-005 (Integration)

T-P2-001 (PortAudio Backend)
    |
    +-- T-P2-002 (Register)
    |
    +-- T-P2-008 (Backend Tests)

T-P2-006 (Platform Spatial) -- independent
```

## Verification Commands

```bash
# Install dependencies
uv pip install sounddevice

# Verify imports
uv run python -c "from engine.platform.audio.backends.portaudio import PortAudioBackend"

# Run audio tests
uv run pytest tests/platform/audio/ -v

# List audio devices (manual verification)
uv run python -c "import sounddevice; print(sounddevice.query_devices())"
```

## Completion Checklist

- [ ] T-P2-001: PortAudio backend created
- [ ] T-P2-002: Backend registered
- [ ] T-P2-003: Doppler effect implemented
- [ ] T-P2-004: Cone attenuation implemented
- [ ] T-P2-005: Spatial integration complete
- [ ] T-P2-006: Platform spatial detection created
- [ ] T-P2-007: Constants added
- [ ] T-P2-008: Backend tests pass
- [ ] T-P2-009: Spatial tests pass
