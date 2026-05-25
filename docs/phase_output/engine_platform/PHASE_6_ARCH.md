# PHASE 6 ARCHITECTURE: RHI Concrete Backend Integration

## Phase Overview

Phase 6 connects the Python RHI abstraction layer to the Rust renderer-backend crate. The RHI layer (1,818 lines) provides complete ABCs and Null implementations; this phase bridges to real GPU operations.

## Architectural Decisions

### ADR-P6-002: RHI Backend Wrapper

**Status:** Proposed

**Context:**
Python RHI expects ABCs. Rust module exposes concrete classes. Need adapter.

**Decision:**
Create wrapper classes that implement RHI ABCs using Rust backend:

```python
# engine/platform/rhi/backends/wgpu.py

from renderer_backend import WgpuDevice as _WgpuDevice

class WgpuDevice(Device):
    def __init__(self, rust_device: _WgpuDevice):
        self._rust = rust_device

    def create_buffer(self, desc: BufferDesc) -> Buffer:
        rust_buffer = self._rust.create_buffer(desc.size, desc.usage.value, desc.memory_type.value)
        return WgpuBuffer(rust_buffer, desc)

class WgpuBuffer(Buffer):
    def __init__(self, rust_buffer, desc: BufferDesc):
        self._rust = rust_buffer
        self._desc = desc

    @property
    def handle(self) -> int:
        return self._rust.handle()

    @property
    def desc(self) -> BufferDesc:
        return self._desc

    def destroy(self) -> None:
        self._rust.destroy()
```

**Consequences:**
- Clean separation: Python defines contracts, Rust provides performance
- Wrappers are thin (delegation only)
- Easy to test (Null backend vs Wgpu backend)

### ADR-P6-003: Swapchain Window Integration

**Status:** Proposed

**Context:**
Swapchain creation requires native window handle. Window backend provides this (Phase 4).

**Decision:**
Swapchain creation flow:

```python
# Application code
from engine.platform.window import Window, WindowConfig
from engine.platform.rhi import Device

# 1. Create window
window = Window.create(WindowConfig(title="Game", width=1920, height=1080))
native_handle = window.native_handle

# 2. Create device
device = Device.create()

# 3. Create swapchain with window handle
swapchain = device.create_swapchain(SwapchainDesc(
    window_handle=native_handle,
    width=1920,
    height=1080,
    format=Format.RGBA8_SRGB,
    present_mode=PresentMode.VSYNC
))
```

Rust backend handles platform-specific surface creation:

```rust
impl WgpuDevice {
    fn create_swapchain(&self, desc: SwapchainDesc) -> PyResult<WgpuSwapchain> {
        // Platform-specific surface creation
        #[cfg(target_os = "windows")]
        let surface = unsafe { self.instance.create_surface_from_hwnd(desc.window_handle) };

        #[cfg(target_os = "linux")]
        let surface = unsafe { self.instance.create_surface_from_xlib(desc.window_handle) };

        // Configure swapchain
        let config = wgpu::SurfaceConfiguration { ... };
        surface.configure(&self.device, &config);

        Ok(WgpuSwapchain { surface, config })
    }
}
```

**Consequences:**
- Window handle is platform-agnostic integer
- Rust handles platform-specific surface creation
- Works with SDL2 window backend (provides native handle)

### ADR-P6-004: Command Recording Strategy

**Status:** Proposed

**Context:**
CommandList records commands. Need to bridge to wgpu command encoder.

**Decision:**
Rust CommandEncoder wrapped by Python CommandList:

```python
class WgpuCommandList(CommandList):
    def __init__(self, rust_encoder):
        self._rust = rust_encoder
        self._recording = False

    def begin(self) -> None:
        self._rust.begin()
        self._recording = True

    def end(self) -> None:
        self._rust.finish()
        self._recording = False

    def draw(self, vertex_count: int, instance_count: int, first_vertex: int, first_instance: int) -> None:
        self._rust.draw(vertex_count, instance_count, first_vertex, first_instance)

    def dispatch(self, x: int, y: int, z: int) -> None:
        self._rust.dispatch(x, y, z)
```

**Consequences:**
- Commands recorded on Rust side (no Python overhead in hot loop)
- Python CommandList is thin wrapper
- Command validation can happen in Rust

### ADR-P6-005: Resource Ownership Model

**Status:** Proposed

**Context:**
GPU resources (buffers, textures) have complex lifetimes. Need clear ownership.

**Decision:**
Python owns handle, Rust owns GPU resource. Destroy explicitly:

```python
buffer = device.create_buffer(desc)  # Rust allocates GPU memory
# ... use buffer ...
buffer.destroy()  # Rust releases GPU memory, Python handle invalid
```

Do NOT rely on Python __del__ for GPU cleanup (GC timing unpredictable).

**Consequences:**
- Explicit lifecycle management
- No memory leaks from GC delays
- destroy() required before device shutdown

## Data Flow

## Build Integration

## Phase Exit Criteria

1. renderer_backend Python module imports successfully
2. WgpuDevice creates buffers and textures
3. WgpuSwapchain presents to SDL2 window
4. Basic render pass (clear screen) works
5. Performance acceptable (< 1ms overhead per frame)
6. Null backend tests still pass
7. Documentation for building renderer-backend
