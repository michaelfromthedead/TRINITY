# PHASE 6 TODO: RHI Concrete Backend Integration

## Tasks

### T-P6-003: Implement Buffer Operations

**Priority:** P0 (Blocking)
**Estimate:** 2 hours

In Rust:

```rust
#[pyclass]
pub struct WgpuBuffer {
    buffer: wgpu::Buffer,
    handle: u64,
}

#[pymethods]
impl WgpuBuffer {
    fn handle(&self) -> u64 { self.handle }
    fn size(&self) -> u64 { self.buffer.size() }
    fn destroy(&mut self) { self.buffer.destroy(); }
    fn map_read(&self) -> PyResult<Vec<u8>> { ... }
    fn write(&self, data: &[u8], offset: u64) -> PyResult<()> { ... }
}
```

**Acceptance Criteria:**
- [ ] create_buffer works
- [ ] map_read returns buffer contents
- [ ] write updates buffer
- [ ] destroy releases GPU memory

---

### T-P6-004: Implement Texture Operations

**Priority:** P0 (Blocking)
**Estimate:** 2 hours

In Rust:

```rust
#[pyclass]
pub struct WgpuTexture {
    texture: wgpu::Texture,
    handle: u64,
}

#[pymethods]
impl WgpuTexture {
    fn handle(&self) -> u64 { self.handle }
    fn width(&self) -> u32 { self.texture.width() }
    fn height(&self) -> u32 { self.texture.height() }
    fn create_view(&self) -> PyResult<WgpuTextureView> { ... }
    fn destroy(&mut self) { self.texture.destroy(); }
}
```

**Acceptance Criteria:**
- [ ] create_texture works for 2D textures
- [ ] create_view returns usable view
- [ ] Multiple formats supported (RGBA8, BGRA8, RGBA16F)
- [ ] destroy releases GPU memory

---

### T-P6-005: Implement Swapchain

**Priority:** P0 (Blocking)
**Estimate:** 3 hours

In Rust:

```rust
#[pyclass]
pub struct WgpuSwapchain {
    surface: wgpu::Surface,
    config: wgpu::SurfaceConfiguration,
}

#[pymethods]
impl WgpuSwapchain {
    fn get_current_texture(&self) -> PyResult<WgpuTextureView> {
        let frame = self.surface.get_current_texture()?;
        Ok(WgpuTextureView { view: frame.texture.create_view(&Default::default()) })
    }
    fn present(&self) -> PyResult<()> { ... }
    fn resize(&mut self, width: u32, height: u32) { ... }
}
```

**Acceptance Criteria:**
- [ ] create_swapchain from window handle
- [ ] get_current_texture returns renderable view
- [ ] present swaps buffers
- [ ] resize handles window resize

---

### T-P6-006: Implement Command Encoder

**Priority:** P0 (Blocking)
**Estimate:** 3 hours

In Rust:

```rust
#[pyclass]
pub struct WgpuCommandEncoder {
    encoder: Option<wgpu::CommandEncoder>,
    device: Arc<wgpu::Device>,
}

#[pymethods]
impl WgpuCommandEncoder {
    fn begin(&mut self) {
        self.encoder = Some(self.device.create_command_encoder(&Default::default()));
    }
    fn begin_render_pass(&mut self, color_view: &WgpuTextureView) -> PyResult<WgpuRenderPass> { ... }
    fn copy_buffer_to_buffer(&mut self, src: &WgpuBuffer, dst: &WgpuBuffer, size: u64) { ... }
    fn finish(&mut self) -> PyResult<WgpuCommandBuffer> { ... }
}
```

**Acceptance Criteria:**
- [ ] begin/finish lifecycle works
- [ ] begin_render_pass creates pass
- [ ] copy operations work
- [ ] Commands recorded correctly

---

### T-P6-007: Implement Render Pass

**Priority:** P0 (Blocking)
**Estimate:** 2 hours

In Rust:

```rust
#[pyclass]
pub struct WgpuRenderPass { ... }

#[pymethods]
impl WgpuRenderPass {
    fn set_pipeline(&mut self, pipeline: &WgpuPipeline) { ... }
    fn set_vertex_buffer(&mut self, slot: u32, buffer: &WgpuBuffer) { ... }
    fn set_index_buffer(&mut self, buffer: &WgpuBuffer, format: u32) { ... }
    fn draw(&mut self, vertices: u32, instances: u32, first_vertex: u32, first_instance: u32) { ... }
    fn draw_indexed(&mut self, indices: u32, instances: u32, first_index: u32, base_vertex: i32, first_instance: u32) { ... }
    fn end(&mut self) { ... }
}
```

**Acceptance Criteria:**
- [ ] set_pipeline binds pipeline
- [ ] set_vertex_buffer/set_index_buffer work
- [ ] draw/draw_indexed record commands
- [ ] end finalizes pass

---

### T-P6-008: Create Python Wrappers

**Priority:** P0 (Blocking)
**Estimate:** 3 hours

Create `engine/platform/rhi/backends/wgpu.py`:

```python
from renderer_backend import (
    WgpuAdapter as _WgpuAdapter,
    WgpuDevice as _WgpuDevice,
    # ...
)
from ..device import Adapter, Device, AdapterInfo
from ..resources import Buffer, Texture, BufferDesc, TextureDesc

class WgpuAdapter(Adapter):
    def __init__(self, rust_adapter: _WgpuAdapter):
        self._rust = rust_adapter

    @staticmethod
    def enumerate() -> list["WgpuAdapter"]:
        return [WgpuAdapter(a) for a in _WgpuAdapter.enumerate()]

    def info(self) -> AdapterInfo:
        rust_info = self._rust.info()
        return AdapterInfo(
            name=rust_info.name,
            vendor_id=rust_info.vendor_id,
            device_id=rust_info.device_id,
            vram_mb=rust_info.vram_mb
        )

class WgpuDevice(Device):
    # ... wrapper methods
```

**Acceptance Criteria:**
- [ ] All RHI ABCs have wgpu implementations
- [ ] Wrappers delegate to Rust
- [ ] Type conversions handled

---

### T-P6-009: Register wgpu Backend

**Priority:** P0 (Blocking)
**Estimate:** 30 minutes

Create/update `engine/platform/rhi/backends/__init__.py`:

```python
try:
    from .wgpu import WgpuAdapter, WgpuDevice
    _registry.register("wgpu", WgpuDevice, set_default=True)
    WGPU_AVAILABLE = True
except ImportError:
    WGPU_AVAILABLE = False
```

**Acceptance Criteria:**
- [ ] wgpu backend registered when available
- [ ] Null backend remains default fallback
- [ ] WGPU_AVAILABLE flag for testing

---

### T-P6-011: Write Integration Tests

**Priority:** P0 (Blocking)
**Estimate:** 3 hours

Create `tests/platform/rhi/test_wgpu.py`:

```python
import pytest
from engine.platform.rhi.backends import WGPU_AVAILABLE

@pytest.mark.skipif(not WGPU_AVAILABLE, reason="wgpu not available")
class TestWgpuBackend:
    def test_enumerate_adapters(self):
        from engine.platform.rhi.backends.wgpu import WgpuAdapter
        adapters = WgpuAdapter.enumerate()
        assert len(adapters) > 0

    def test_create_device(self):
        adapters = WgpuAdapter.enumerate()
        device = adapters[0].create_device()
        assert device is not None

    def test_create_buffer(self):
        # ...

    def test_render_pass_clear(self):
        # Create window, swapchain, clear to color, present
        # ...
```

**Acceptance Criteria:**
- [ ] Adapter enumeration tested
- [ ] Device creation tested
- [ ] Buffer creation tested
- [ ] Basic render (clear screen) tested
- [ ] Tests skip without GPU

---

### T-P6-012: Document Build Process

**Priority:** P1 (Important)
**Estimate:** 1 hour

Update documentation with:
- Prerequisites (Rust, maturin, GPU drivers)
- Build commands
- Troubleshooting common issues
- CI configuration

**Acceptance Criteria:**
- [ ] Build steps documented
- [ ] Prerequisites listed
- [ ] Troubleshooting section
- [ ] CI config example

---

## Task Dependency Graph

```
T-P6-001 (Audit Rust Crate)
    |
    +-- T-P6-002 (Define Interface)
            |
            +-- T-P6-003 (Buffer Ops)
            +-- T-P6-004 (Texture Ops)
            +-- T-P6-005 (Swapchain)
            +-- T-P6-006 (Command Encoder)
            +-- T-P6-007 (Render Pass)
                    |
                    +-- T-P6-008 (Python Wrappers)
                            |
                            +-- T-P6-009 (Register Backend)
                                    |
                                    +-- T-P6-011 (Integration Tests)

T-P6-010 (Build/Package) -- parallel with T-P6-003 through T-P6-007
T-P6-012 (Documentation) -- after T-P6-010
```

## Completion Checklist

- [ ] T-P6-001: Rust crate audited
- [ ] T-P6-002: Python interface defined
- [ ] T-P6-003: Buffer operations work
- [ ] T-P6-004: Texture operations work
- [ ] T-P6-005: Swapchain works
- [ ] T-P6-006: Command encoder works
- [ ] T-P6-007: Render pass works
- [ ] T-P6-008: Python wrappers created
- [ ] T-P6-009: Backend registered
- [ ] T-P6-010: Build process works
- [ ] T-P6-011: Integration tests pass
- [ ] T-P6-012: Build documented
