# PHASE 4 TODO: Window Native Backends

## Summary

Add native window backends (SDL2) to the existing headless window subsystem.

**Estimated Effort:** 14-18 hours
**Dependencies:** Phase 1, Phase 3 (shared SDL2)
**Blocking:** RHI swapchain integration

---

## Tasks

### T-P4-001: Create WindowBackend ABC

**Priority:** P0 (Blocking)
**Estimate:** 1.5 hours

Create `engine/platform/window/window_backend.py`:

```python
from abc import ABC, abstractmethod

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
    @abstractmethod
    def query_hdr_support(self, display_index: int) -> HDRCapabilities: ...
    @abstractmethod
    def query_vrr_support(self, display_index: int) -> VRRCapabilities: ...
```

**Acceptance Criteria:**
- [ ] All required methods defined
- [ ] Type hints complete
- [ ] Docstrings explain each method

---

### T-P4-002: Refactor Headless to Backend

**Priority:** P0 (Blocking)
**Estimate:** 2 hours

Move headless implementation to `engine/platform/window/backends/headless.py`:

```python
class HeadlessWindowBackend(WindowBackend):
    # Move existing Window, Display logic here
    def create_window(self, config: WindowConfig) -> Window:
        # Existing headless window creation
        ...
```

**Acceptance Criteria:**
- [ ] All headless functionality preserved
- [ ] Existing tests pass without modification
- [ ] Window class uses backend internally

---

### T-P4-003: Create Backend Registry

**Priority:** P0 (Blocking)
**Estimate:** 30 minutes

Create `engine/platform/window/backends/__init__.py`:

```python
from engine.platform.registry import BackendRegistry
from .headless import HeadlessWindowBackend

_registry = BackendRegistry[WindowBackend]()
_registry.register("headless", HeadlessWindowBackend, set_default=True)

def get_backend(name: str | None = None) -> WindowBackend:
    return _registry.create(name)
```

**Acceptance Criteria:**
- [ ] Uses generic BackendRegistry
- [ ] Headless is default
- [ ] get_backend() works

---

### T-P4-004: Create SDL2 Window Backend

**Priority:** P0 (Blocking)
**Estimate:** 5 hours

Create `engine/platform/window/backends/sdl2.py`:

```python
class SDL2WindowBackend(WindowBackend):
    def initialize(self) -> bool:
        return sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO) == 0

    def create_window(self, config: WindowConfig) -> Window:
        flags = sdl2.SDL_WINDOW_SHOWN
        if config.resizable:
            flags |= sdl2.SDL_WINDOW_RESIZABLE
        if config.fullscreen:
            flags |= sdl2.SDL_WINDOW_FULLSCREEN_DESKTOP

        sdl_window = sdl2.SDL_CreateWindow(
            config.title.encode(),
            sdl2.SDL_WINDOWPOS_CENTERED if config.x < 0 else config.x,
            sdl2.SDL_WINDOWPOS_CENTERED if config.y < 0 else config.y,
            config.width,
            config.height,
            flags
        )
        return SDL2Window(sdl_window, config)
```

**Acceptance Criteria:**
- [ ] Window visible on screen
- [ ] Fullscreen works (borderless and exclusive)
- [ ] Resize events generated
- [ ] Close button generates close event
- [ ] Focus events work

---

### T-P4-005: Implement Native Handle Extraction

**Priority:** P0 (Blocking)
**Estimate:** 2 hours

Add to `backends/sdl2.py`:

```python
def get_native_handle(self, window: Window) -> int:
    if not isinstance(window, SDL2Window):
        return 0

    info = sdl2.SDL_SysWMinfo()
    sdl2.SDL_VERSION(info.version)
    if sdl2.SDL_GetWindowWMInfo(window._sdl_window, ctypes.byref(info)) != sdl2.SDL_TRUE:
        return 0

    if info.subsystem == sdl2.SDL_SYSWM_WINDOWS:
        return int(info.info.win.window)
    elif info.subsystem == sdl2.SDL_SYSWM_X11:
        return int(info.info.x11.window)
    elif info.subsystem == sdl2.SDL_SYSWM_COCOA:
        return int(info.info.cocoa.window)
    elif info.subsystem == sdl2.SDL_SYSWM_WAYLAND:
        return int(info.info.wl.surface)
    return 0
```

**Acceptance Criteria:**
- [ ] Returns valid HWND on Windows
- [ ] Returns valid Window on X11
- [ ] Returns valid NSWindow* on macOS
- [ ] Returns valid wl_surface* on Wayland
- [ ] Returns 0 on failure

---

### T-P4-006: Implement Display Enumeration

**Priority:** P0 (Blocking)
**Estimate:** 1.5 hours

Add to `backends/sdl2.py`:

```python
def enumerate_displays(self) -> list[DisplayInfo]:
    displays = []
    num_displays = sdl2.SDL_GetNumVideoDisplays()
    for i in range(num_displays):
        name = sdl2.SDL_GetDisplayName(i).decode()
        bounds = sdl2.SDL_Rect()
        sdl2.SDL_GetDisplayBounds(i, ctypes.byref(bounds))
        dpi = ctypes.c_float()
        sdl2.SDL_GetDisplayDPI(i, None, ctypes.byref(dpi), None)

        displays.append(DisplayInfo(
            name=name,
            bounds=Rect(bounds.x, bounds.y, bounds.w, bounds.h),
            work_area=self._get_work_area(i),
            dpi_scale=dpi.value / 96.0 if dpi.value > 0 else 1.0,
            is_primary=(i == 0)
        ))
    return displays
```

**Acceptance Criteria:**
- [ ] All connected displays returned
- [ ] Bounds correct for each display
- [ ] DPI scale accurate
- [ ] Primary display identified

---

### T-P4-007: Implement Display Mode Enumeration

**Priority:** P1 (Important)
**Estimate:** 1 hour

Add to `backends/sdl2.py`:

```python
def get_display_modes(self, display_index: int) -> list[DisplayMode]:
    modes = []
    num_modes = sdl2.SDL_GetNumDisplayModes(display_index)
    for i in range(num_modes):
        mode = sdl2.SDL_DisplayMode()
        sdl2.SDL_GetDisplayMode(display_index, i, ctypes.byref(mode))
        modes.append(DisplayMode(
            width=mode.w,
            height=mode.h,
            refresh_rate=mode.refresh_rate,
            format=self._translate_format(mode.format)
        ))
    return modes
```

**Acceptance Criteria:**
- [ ] All available modes returned
- [ ] Refresh rates accurate
- [ ] No duplicate modes

---

### T-P4-008: Implement Window Event Translation

**Priority:** P0 (Blocking)
**Estimate:** 2 hours

Add to `backends/sdl2.py`:

```python
def poll_events(self) -> list[WindowEvent]:
    events = []
    for event in sdl2.ext.get_events():
        if event.type == sdl2.SDL_WINDOWEVENT:
            window_id = event.window.windowID
            window = self._windows.get(window_id)
            if window is None:
                continue

            if event.window.event == sdl2.SDL_WINDOWEVENT_CLOSE:
                events.append(WindowEvent(WindowEventType.CLOSE, {}))
            elif event.window.event == sdl2.SDL_WINDOWEVENT_RESIZED:
                events.append(WindowEvent(WindowEventType.RESIZE, {
                    "width": event.window.data1,
                    "height": event.window.data2
                }))
            # ... focus, minimize, maximize, move
    return events
```

**Acceptance Criteria:**
- [ ] CLOSE event on window close button
- [ ] RESIZE event on resize
- [ ] FOCUS_GAINED/LOST on focus change
- [ ] MINIMIZED/MAXIMIZED on state change
- [ ] MOVE event on window drag

---

### T-P4-009: Register SDL2 Backend

**Priority:** P0 (Blocking)
**Estimate:** 30 minutes

Update `backends/__init__.py`:

```python
try:
    from .sdl2 import SDL2WindowBackend
    _registry.register("sdl2", SDL2WindowBackend, set_default=True)
except ImportError:
    pass  # SDL2 not available
```

**Acceptance Criteria:**
- [ ] SDL2 becomes default when available
- [ ] Headless remains available
- [ ] ImportError handled gracefully

---

### T-P4-010: Integrate Window with Backend

**Priority:** P0 (Blocking)
**Estimate:** 1.5 hours

Modify `engine/platform/window/window.py`:

```python
class Window:
    def __init__(self, backend: WindowBackend, config: WindowConfig):
        self._backend = backend
        self._config = config
        # Backend creates actual window

    @property
    def native_handle(self) -> int:
        return self._backend.get_native_handle(self)

    # Factory method
    @classmethod
    def create(cls, config: WindowConfig, backend: WindowBackend | None = None) -> "Window":
        if backend is None:
            from .backends import get_backend
            backend = get_backend()
        return backend.create_window(config)
```

**Acceptance Criteria:**
- [ ] Window.create() uses default backend
- [ ] native_handle property works
- [ ] Backward compatible

---

### T-P4-011: Integrate Display with Backend

**Priority:** P1 (Important)
**Estimate:** 1 hour

Modify `engine/platform/window/display.py`:

```python
class Display:
    def __init__(self, backend: WindowBackend | None = None):
        if backend is None:
            from .backends import get_backend
            backend = get_backend()
        self._backend = backend

    def enumerate(self) -> list[DisplayInfo]:
        return self._backend.enumerate_displays()

    def get_modes(self, display_index: int = 0) -> list[DisplayMode]:
        return self._backend.get_display_modes(display_index)
```

**Acceptance Criteria:**
- [ ] Display uses backend for enumeration
- [ ] Headless still returns simulated displays
- [ ] SDL2 returns real displays

---

### T-P4-012: Write Backend Tests

**Priority:** P0 (Blocking)
**Estimate:** 2 hours

Create `tests/platform/window/test_backends.py`:

```python
class TestHeadlessWindowBackend:
    def test_create_destroy_window(self): ...
    def test_enumerate_displays(self): ...
    def test_native_handle_is_zero(self): ...

@pytest.mark.skipif(not SDL2_AVAILABLE, reason="SDL2 not installed")
class TestSDL2WindowBackend:
    def test_initialize_shutdown(self): ...
    def test_create_destroy_window(self): ...
    def test_native_handle_nonzero(self): ...
    def test_enumerate_displays_returns_at_least_one(self): ...
```

**Acceptance Criteria:**
- [ ] Headless backend fully tested
- [ ] SDL2 backend tested (where display available)
- [ ] Tests skip gracefully in CI

---

## Task Dependency Graph

```
T-P4-001 (WindowBackend ABC)
    |
    +-- T-P4-002 (Refactor Headless)
    |       |
    |       +-- T-P4-003 (Registry)
    |               |
    |               +-- T-P4-009 (Register SDL2)
    |               |
    |               +-- T-P4-010 (Window Integration)
    |               |
    |               +-- T-P4-011 (Display Integration)
    |
    +-- T-P4-004 (SDL2 Backend)
            |
            +-- T-P4-005 (Native Handle)
            |
            +-- T-P4-006 (Display Enum)
            |
            +-- T-P4-007 (Mode Enum)
            |
            +-- T-P4-008 (Event Translation)
            |
            +-- T-P4-009 (Register)

T-P4-012 (Tests) -- after all others
```

## Verification Commands

```bash
# Verify imports
uv run python -c "from engine.platform.window.backends.sdl2 import SDL2WindowBackend"

# Run window tests
uv run pytest tests/platform/window/ -v

# Manual test (requires display)
uv run python -c "
from engine.platform.window import Window, WindowConfig
from engine.platform.window.backends import get_backend

backend = get_backend('sdl2')
backend.initialize()
window = backend.create_window(WindowConfig(title='Test', width=800, height=600))
print(f'Native handle: {backend.get_native_handle(window)}')
import time; time.sleep(2)
backend.destroy_window(window)
backend.shutdown()
"
```

## Completion Checklist

- [ ] T-P4-001: WindowBackend ABC created
- [ ] T-P4-002: Headless refactored to backend
- [ ] T-P4-003: Backend registry created
- [ ] T-P4-004: SDL2 backend created
- [ ] T-P4-005: Native handle extraction works
- [ ] T-P4-006: Display enumeration works
- [ ] T-P4-007: Mode enumeration works
- [ ] T-P4-008: Event translation works
- [ ] T-P4-009: SDL2 registered
- [ ] T-P4-010: Window integrated
- [ ] T-P4-011: Display integrated
- [ ] T-P4-012: Tests pass
