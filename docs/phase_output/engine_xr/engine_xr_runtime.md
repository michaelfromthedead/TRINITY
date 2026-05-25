# Investigation: engine/xr/runtime

## Summary

The XR runtime module provides a comprehensive abstraction layer for VR/AR with full API designs for OpenXR and WebXR backends. However, all implementations are **simulations** - they return hardcoded poses, simulated timing, and mock device data. No actual native OpenXR SDK bindings, ctypes FFI, or browser JavaScript interop exists.

## Files

| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `__init__.py` | 129 | Complete | Well-organized exports, good documentation |
| `xr_runtime.py` | 721 | **STUB** | Abstract base + _MockRuntime, no real SDK calls |
| `openxr.py` | 664 | **STUB** | OpenXR API design only, comments say "simulated" |
| `webxr.py` | 698 | **STUB** | WebXR API design only, no browser interop |
| `session.py` | 541 | Real Logic | State machine with hooks - actual implementation |
| `capabilities.py` | 380 | Real Logic | Feature detection dataclasses - actual implementation |

**Total**: ~3,133 lines in runtime subdirectory

## XR Components

- **XRRuntime (ABC)**: Abstract base class defining unified XR API
- **OpenXRRuntime**: OpenXR 1.0+ backend (simulated)
- **WebXRRuntime**: WebXR Device API backend (simulated)
- **_MockRuntime**: Testing/fallback runtime
- **XRSession**: Session lifecycle state machine with enter/exit hooks
- **XRCapabilities**: Device feature detection (26 XRFeature types)
- **Pose**: 6DOF pose representation (position, orientation, velocities)
- **ViewInfo**: Per-eye rendering parameters (FOV, clip planes)

## XR Implementation

| Question | Answer | Evidence |
|----------|--------|----------|
| Real OpenXR? | **NO** | Line 107-109: "This implementation simulates OpenXR behavior. In production, it would use the actual OpenXR SDK via ctypes or a Python binding." |
| Real headset API? | **NO** | All `get_head_pose()` methods return static hardcoded positions (0.0, 1.6/1.7, 0.0) |
| Real controller tracking? | **NO** | `get_controller_pose()` returns fixed positions (-0.3, 1.0, -0.3) |
| Real frame sync? | **NO** | `wait_frame()` uses `time.sleep(1.0 / refresh_rate)` instead of xrWaitFrame |
| Just abstractions? | **YES** | The API surface is complete but all methods return simulated data |

## Verdict

**PARTIAL - API DESIGN COMPLETE, NATIVE BINDINGS MISSING**

The XR runtime has:
- Complete API design following OpenXR/WebXR specifications
- Proper state machine for session lifecycle
- Comprehensive capability detection system
- Event system with callbacks
- Good architecture for graceful degradation

But critically lacks:
- Any ctypes/cffi bindings to OpenXR SDK
- Any WASM/JS interop for WebXR
- Any real HMD communication
- Any real tracking data

## Evidence

### OpenXR "Real" Implementation Note (openxr.py:107-109)
```python
class OpenXRRuntime(XRRuntime):
    """OpenXR runtime implementation.
    ...
    Note:
        This implementation simulates OpenXR behavior for development.
        In production, it would use the actual OpenXR SDK via ctypes or
        a Python binding.
    """
```

### Simulated Tracking (openxr.py:349-356)
```python
def get_head_pose(self, predicted_time: float = 0.0) -> Pose:
    # In real implementation, this calls xrLocateSpace
    # with the VIEW reference space

    # Simulated tracking - return a reasonable default pose
    return Pose(
        position=(0.0, 1.7, 0.0),  # Approximate standing eye height
        orientation=(0.0, 0.0, 0.0, 1.0),  # Looking forward
        ...
    )
```

### Simulated Frame Timing (openxr.py:252-286)
```python
def wait_frame(self) -> bool:
    # In real implementation, this calls xrWaitFrame
    # which blocks until the compositor is ready

    # Simulate frame timing
    current_time = time.perf_counter()
    if self._last_frame_time > 0:
        frame_time = current_time - self._last_frame_time
        target_frame_time = 1.0 / self._state.display_refresh_rate

        if frame_time < target_frame_time:
            time.sleep(target_frame_time - frame_time)  # <-- Software timing, not compositor sync
```

### WebXR No Browser Interop (webxr.py:115-119)
```python
class WebXRRuntime(XRRuntime):
    """WebXR runtime implementation.
    ...
    Note:
        This implementation simulates WebXR behavior for development.
        In production within a browser, it would use the WebXR Device API
        through JavaScript interop.
    """
```

### Good Architecture Pattern (session.py state machine)
```python
_VALID_TRANSITIONS: Dict[XRSessionState, frozenset[XRSessionState]] = {
    XRSessionState.IDLE: frozenset({XRSessionState.READY, XRSessionState.ERROR}),
    XRSessionState.READY: frozenset({XRSessionState.RUNNING, XRSessionState.STOPPING, ...}),
    XRSessionState.RUNNING: frozenset({XRSessionState.PAUSED, XRSessionState.STOPPING, ...}),
    ...
}
```

## Recommendation

To make this real:
1. Add `pyopenxr` or ctypes bindings to OpenXR loader library
2. Implement native xrCreateInstance, xrWaitFrame, xrLocateSpace calls
3. For WebXR: requires running in browser context (PyScript/Pyodide) with JS interop
4. Or: use Rust backend (renderer-backend crate) to implement OpenXR and expose via PyO3
