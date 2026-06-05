# PHASE 4 ARCHITECTURE: Window Native Backends

## Phase Overview

Phase 4 adds native window backends to the existing headless window subsystem. The headless backend (899 lines) provides the complete API; this phase adds platform-specific implementations for real windowing.

## Current State (from Investigation)

| Component | Status | Lines |
|-----------|--------|-------|
| Window | REAL (Headless) | 308 |
| Display | REAL (Headless) | 165 |
| HDR | REAL (Headless) | 145 |
| VRR | REAL (Headless) | 110 |
| Cursor | REAL (Headless) | 108 |

**Design:** The headless backend is fully functional for CI/testing. Native backends will implement the same API using platform windowing systems.

## Architectural Decisions

### ADR-P4-001: Backend Strategy

**Status:** Proposed

**Context:**
Window backends require native windowing:
- Win32: CreateWindow, HWND, WndProc
- X11: XCreateWindow, Display, Window
- Wayland: wl_surface, xdg_toplevel
- Cocoa: NSWindow, NSView

Options:
1. Raw platform APIs via ctypes
2. GLFW via pyglfw
3. SDL2 via pysdl2 (reuse from input)
4. Custom native extension

**Decision:**
Use SDL2 as primary window backend (shares dependency with input):

```python
class SDL2WindowBackend(WindowBackend):
    def create_window(self, config: WindowConfig) -> Window:
        flags = sdl2.SDL_WINDOW_RESIZABLE if config.resizable else 0
        sdl_window = sdl2.SDL_CreateWindow(
            config.title.encode(),
            config.x, config.y,
            config.width, config.height,
            flags
        )
        return SDL2Window(sdl_window, config)
```

Also add GLFW as alternative for contexts requiring specific OpenGL features.

**Consequences:**
- Shared SDL2 dependency with input
- Cross-platform with single codebase
- GLFW option for Vulkan/OpenGL contexts
- Native handle extraction for RHI integration

### ADR-P4-002: Window Backend Interface

**Status:** Proposed

**Context:**
Need abstraction over window creation and display enumeration.

**Decision:**
Define WindowBackend ABC:

```python
class WindowBackend(ABC):
    @abstractmethod
    def initialize(self) -> bool: ...

    @abstractmethod
    def shutdown(self) -> None: ...

    @abstractmethod
    def create_window(self, config: WindowConfig) -> Window: ...

    @abstractmethod
    def destroy_window(self, window: Window) -> None: ...

    @abstractmethod
    def poll_events(self) -> list[WindowEvent]: ...

    @abstractmethod
    def enumerate_displays(self) -> list[DisplayInfo]: ...

    @abstractmethod
    def get_display_modes(self, display_index: int) -> list[DisplayMode]: ...

    @abstractmethod
    def get_native_handle(self, window: Window) -> int: ...
```

**Consequences:**
- Consistent API across backends
- Native handle enables RHI swapchain creation
- Event polling separate from input (window resize, focus, etc.)

### ADR-P4-003: HDR and VRR Integration

**Status:** Proposed

**Context:**
HDR and VRR require platform-specific queries:
- HDR: DXGI output info (Windows), ColorSync (macOS)
- VRR: DXGI adaptive sync, Xrandr VRR

**Decision:**
Add capability query methods to WindowBackend:

```python
class WindowBackend(ABC):
    @abstractmethod
    def query_hdr_support(self, display_index: int) -> HDRCapabilities: ...

    @abstractmethod
    def query_vrr_support(self, display_index: int) -> VRRCapabilities: ...

    @abstractmethod
    def set_hdr_mode(self, window: Window, enabled: bool, metadata: HDRMetadata | None) -> bool: ...

    @abstractmethod
    def set_vrr_mode(self, window: Window, enabled: bool) -> bool: ...
```

SDL2 has limited HDR/VRR support; for full features, may need platform-specific code.

**Consequences:**
- Graceful degradation on platforms without HDR/VRR
- Full support requires platform-specific backends
- SDL2 backend returns conservative capabilities

### ADR-P4-004: Native Handle Extraction

**Status:** Proposed

**Context:**
RHI needs native window handle (HWND, Window, NSWindow) for swapchain creation.

**Decision:**
Expose native handle via backend:

```python
class SDL2WindowBackend(WindowBackend):
    def get_native_handle(self, window: Window) -> int:
        info = sdl2.SDL_SysWMinfo()
        sdl2.SDL_VERSION(info.version)
        sdl2.SDL_GetWindowWMInfo(window._sdl_window, ctypes.byref(info))

        if info.subsystem == sdl2.SDL_SYSWM_WINDOWS:
            return info.info.win.window  # HWND
        elif info.subsystem == sdl2.SDL_SYSWM_X11:
            return info.info.x11.window  # Window
        elif info.subsystem == sdl2.SDL_SYSWM_COCOA:
            return info.info.cocoa.window  # NSWindow*
        return 0
```

**Consequences:**
- RHI can create swapchains on real windows
- Platform-specific swapchain code uses handle
- Headless backend returns 0 (no native handle)

### ADR-P4-005: Event Dispatch Integration

**Status:** Proposed

**Context:**
Window events (resize, close, focus) and input events (key, mouse) both come from SDL2. Need coordinated dispatch.

**Decision:**
Single SDL2 event loop dispatches to both subsystems:

```python
class SDL2Backend:
    """Combined window + input backend."""

    def poll_all(self) -> tuple[list[WindowEvent], list[InputEvent]]:
        window_events = []
        input_events = []

        for event in sdl2.ext.get_events():
            if event.type in (sdl2.SDL_WINDOWEVENT, ...):
                window_events.append(self._translate_window_event(event))
            elif event.type in (sdl2.SDL_KEYDOWN, sdl2.SDL_MOUSEMOTION, ...):
                input_events.append(self._translate_input_event(event))

        return window_events, input_events
```

**Consequences:**
- Single SDL2 initialization
- Coordinated shutdown
- Window and input share event loop
- Optional: split backends for testing

## Component Diagram

```
engine/platform/window/
    |
    +-- window.py           # Window, WindowConfig, WindowEvent
    +-- display.py          # Display, DisplayMode, DisplayInfo
    +-- hdr.py              # HDRCapabilities, DisplayHDR
    +-- vrr.py              # VRRCapabilities, VariableRefresh
    +-- cursor.py           # CursorManager, CursorType
    |
    +-- window_backend.py   # NEW: WindowBackend ABC
    |
    +-- backends/
            |
            +-- __init__.py      # Backend registry
            +-- headless.py      # Existing headless (moved)
            +-- sdl2.py          # NEW: SDL2 backend
            +-- glfw.py          # FUTURE: GLFW backend
```

## Data Flow

### Window Creation Flow

```
Application
    |
    +-- WindowBackend.create_window(config)
            |
            +-- SDL2: SDL_CreateWindow()
            +-- Native handle stored
            |
            v
    Window object returned
            |
            +-- RHI.create_swapchain(window.native_handle)
```

### Event Flow

```
SDL2Backend.poll_all()
       |
       +-- Window events (resize, close, focus)
       |       |
       |       v
       |   WindowManager.dispatch(events)
       |
       +-- Input events (key, mouse, gamepad)
               |
               v
           InputManager.queue_event(events)
```

## File Changes Required

### New Files

| File | Purpose |
|------|---------|
| engine/platform/window/window_backend.py | WindowBackend ABC |
| engine/platform/window/backends/__init__.py | Backend registry |
| engine/platform/window/backends/headless.py | Existing headless (refactored) |
| engine/platform/window/backends/sdl2.py | SDL2 window backend |

### Modified Files

| File | Changes |
|------|---------|
| engine/platform/window/window.py | Window accepts backend, exposes native_handle |
| engine/platform/window/display.py | Display uses backend for enumeration |
| engine/platform/window/hdr.py | HDR queries through backend |
| engine/platform/window/vrr.py | VRR queries through backend |
| engine/platform/window/__init__.py | Export new classes |

## Dependencies

Shared with Phase 3 (Input):
- pysdl2 >=0.9.16
- pysdl2-dll >=2.28.0 (Windows)

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| SDL2 limited HDR support | Document limitations, future platform backends |
| VRR requires driver support | Detect and degrade gracefully |
| Native handle varies by platform | SDL2 SysWMinfo provides abstraction |
| Window/input event coordination | Shared backend class handles both |

## Phase Exit Criteria

1. SDL2WindowBackend creates visible windows
2. Window resize/close events dispatched correctly
3. Native handle extraction works on Windows/Linux/macOS
4. Display enumeration returns real displays
5. Headless backend still works for testing
6. Integration with RHI demonstrated (swapchain creation)
