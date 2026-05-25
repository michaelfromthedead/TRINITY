# CLARIFICATION: Engine Platform Layer Design Rationale

## Philosophical Framing

### The Platform Layer as a Contract

The engine/platform/ layer exists to answer a fundamental question: *How does game engine code interact with hardware without knowing which hardware it runs on?*

The answer is not "abstraction for abstraction's sake" but rather:

1. **Testability** - Every subsystem can run in CI without hardware
2. **Portability** - Desktop, mobile, console, web share the same API
3. **Extensibility** - New backends plug in without changing client code
4. **Reliability** - Null backends expose API contract violations immediately

### Why Null Backends are REAL Implementations

The investigation revealed that "null" and "headless" implementations are not stubs. They are fully functional implementations that:

- Track all state changes correctly
- Generate all events the API promises
- Validate all inputs and outputs
- Serve as the reference implementation for the API contract

A true stub would be:
```python
def is_available(self) -> bool:
    return False  # Not implemented
```

A real null backend is:
```python
def is_available(self) -> bool:
    return True  # Available as null backend

def create_buffer(self, desc: BufferDesc) -> Buffer:
    handle = self._next_handle()  # Track allocation
    self._buffers[handle] = NullBuffer(handle, desc)  # Store state
    return self._buffers[handle]  # Return real object
```

The null backend IS the implementation. Platform backends (WASAPI, Vulkan, etc.) are specializations.

## Design Rationale

### Audio Subsystem

**Why callback-based streaming?**

Audio hardware operates on fixed timing constraints. Push-based APIs (write data when ready) cause glitches when the application is slow. Pull-based APIs (hardware requests data) let the driver maintain timing.

The AudioStream callback model:
```python
def callback(input_buffer: np.ndarray, frames: int) -> np.ndarray:
    # Fill output buffer based on input
```

This matches WASAPI (IAudioRenderClient), CoreAudio (AudioUnit), and ALSA (snd_pcm_writei) mental models.

**Why spatial audio is separate?**

Positional audio involves math (distance attenuation, panning, HRTF) that is API-agnostic. The SpatialAudioEngine can run on top of ANY audio backend. Platform-specific spatial APIs (Windows Sonic, Tempest 3D) are optimizations, not requirements.

### Input Subsystem

**Why event injection, not hardware polling?**

Platform input APIs vary wildly:
- Win32: GetAsyncKeyState, Raw Input, DirectInput, XInput
- Linux: evdev, libinput
- macOS: IOKit, NSEvent
- Consoles: Proprietary HID protocols

By making the input manager event-driven, platform code becomes a thin event translator. The input system itself is 100% testable.

**Why frame-based pressed/released detection?**

Games need edge detection: "was jump pressed this frame?" not "is jump down?"

Maintaining previous/current state per frame is the standard solution:
```python
def is_key_pressed(self, key: KeyCode) -> bool:
    return key in self._current_keys and key not in self._previous_keys
```

This matches Unity (Input.GetKeyDown), Unreal (IsInputKeyPressed), and Godot (is_action_just_pressed).

### RHI Subsystem

**Why separate ABC from Null?**

The ABC defines WHAT the API promises. The Null implementation proves the promise is fulfillable. Concrete backends then have a working reference to compare against.

```
ABC (Contract) --> Null (Reference) --> Vulkan/D3D12/Metal (Production)
```

**Why so many enums?**

Modern GPU APIs have combinatorial state spaces. Enums make invalid states unrepresentable:
```python
class BufferUsage(Flag):
    VERTEX = auto()
    INDEX = auto()
    CONSTANT = auto()
    # ...

# Valid: BufferUsage.VERTEX | BufferUsage.INDEX
# Invalid: BufferUsage(-1) -- type error
```

### OS Subsystem

**Why Result pattern for file I/O?**

Exceptions are expensive and can be uncaught. Result types force error handling:
```python
result = fs.read_file("/path")
if result.is_ok():
    data = result.unwrap()
else:
    error = result.error
```

This matches Rust's `std::io::Result` and is increasingly common in Python (returns, result).

**Why lock-based atomics?**

Python's GIL means true lock-free operations require C extensions. Lock-based atomics are:
1. Correct (no data races)
2. Portable (pure Python)
3. Good enough (contention is rare in practice)

For hot paths, use the Rust renderer-backend.

### Window Subsystem

**Why headless-first design?**

Game engines run in many contexts:
- Developer machines (need windows)
- CI pipelines (no display server)
- Dedicated servers (no GPU at all)
- Mobile (OS owns the window)

Starting with headless means the API is designed for testability. Native backends add platform-specific behavior without changing the contract.

### GPU Low-Latency Subsystem

**Why is this a stub?**

NVIDIA Reflex and AMD Anti-Lag require:
1. Vendor libraries (NVAPI, AGS)
2. Driver detection and capability queries
3. Tight coupling with the render loop

These cannot be implemented in pure Python without native extensions. The stub exists to:
1. Define the API shape
2. Allow code to call low-latency APIs without runtime errors
3. Mark the integration point for future native code

### Services Subsystem

**Why does permissions auto-grant?**

Desktop apps rarely need runtime permissions. Mobile apps do. Rather than:
- Crash when permissions aren't implemented
- Silently fail
- Require platform code for all platforms

The stub grants all permissions. Platform code can override with real dialogs. Desktop deploys with the stub and everything works.

## Anti-Patterns Avoided

### Avoided: Platform #ifdefs in Core Code

```python
# BAD
if sys.platform == "win32":
    # Windows-specific code
elif sys.platform == "darwin":
    # macOS-specific code
```

Instead: Backend registry selects implementation at runtime.

### Avoided: Hardware Polling in Tests

```python
# BAD
def test_gamepad():
    gamepad = Gamepad.get_first()  # Fails if no gamepad connected
```

Instead: Event injection allows deterministic testing.

### Avoided: Blocking I/O Everywhere

```python
# BAD
data = file.read()  # Blocks entire thread
```

Instead: Async I/O via asyncio for file operations that might block.

### Avoided: Magic Numbers Scattered

```python
# BAD
buffer_size = 1024  # Where did this come from?
```

Instead: All defaults in engine/platform/constants.py with documentation.

## Integration Philosophy

### Python Layer vs Rust Layer

The platform layer is the Python API. It defines contracts and provides testable implementations.

The Rust layer (crates/renderer-backend/) provides performance-critical operations:
- Actual GPU command recording
- Shader compilation
- Memory management

The boundary:
```
Python: "I want a buffer with these properties" (RHI call)
Rust: "I'll allocate that buffer on the GPU" (wgpu call)
```

PyO3 bridges the boundary. Python defines what; Rust does how.

### Event Flow

```
Platform Backend (Win32, X11, etc.)
       |
       v
Event Translation (platform-specific)
       |
       v
InputManager.queue_event() (platform-agnostic)
       |
       v
InputDevice.update() (per-device processing)
       |
       v
Game Code (is_key_pressed, get_axis, etc.)
```

Each arrow is a clean abstraction boundary.
