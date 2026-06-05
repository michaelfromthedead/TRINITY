# RUST BACKLOG — Translocated from Python SDLC TODO/ARCH files
Tasks referencing Rust crates, .rs files, pyo3, or cargo.
Preserved for future implementation. Removed from Python-focused workflow files.

---

## engine_animation_crowds_facial/PHASE_1_TODO.md

### T1.4 GPU Backend Integration (BRIDGE)

**Priority:** CRITICAL
**Estimate:** 4 hours
**File:** `crates/renderer-backend/src/bridge.rs` (new)

**Acceptance Criteria:**
- [ ] Rust function receives packed instance buffer
- [ ] Texture atlas uploaded to GPU
- [ ] Draw call executes with correct instance count
- [ ] Animation texture sampling in vertex shader

**Notes:**
- This is the Python-Rust bridge gap identified in GAPSET_3
- Requires `pyo3` bindings for buffer passing
- Vertex shader must sample animation texture

---

---

## engine_audio_adaptive_core/PHASE_3_ARCH.md

### 3.1 Option A: PyO3/Rust Backend

**Description**: Implement audio output in Rust, call from Python via PyO3.

**Advantages**:
- High performance mixing
- Low latency
- Aligns with TRINITY's Rust backend direction

**Architecture**:
```
Python AudioEngine
        |
        v
PyO3 Bindings
        |
        v
Rust AudioBackend
        |
        v
Platform API (WASAPI/CoreAudio/ALSA/PulseAudio)
```

**Effort**: Large (multi-week)


## 4. Recommended Approach

**Recommendation**: Option D (miniaudio) for initial integration, with Option A (PyO3/Rust) as long-term target.

**Rationale**:
1. miniaudio provides immediate playback with minimal effort
2. Rust backend can replace miniaudio later without changing Python API
3. Command queue pattern isolates game code from backend choice

---

### 8.4 Phase 3d: Rust Backend (Future)

1. Implement Rust AudioBackend
2. PyO3 bindings
3. Replace miniaudio
4. Performance comparison

**Duration**: 2-4 weeks


---

## engine_audio_adaptive_core/PHASE_3_TODO.md

### T-AUDIO-3.12: PyO3 Bindings

**Priority**: Low (Future)
**Description**: Create PyO3 bindings for Rust backend.

**Acceptance Criteria**:
- [ ] Python can call Rust AudioBackend
- [ ] Zero-copy buffer passing where possible
- [ ] Drop-in replacement for MiniaudioBackend

**Effort**: Medium (1 week)

---

---

## engine_debug_resource/PHASE_2_TODO.md

### Task GPU-INT-002: Integrate GPUProfiler with renderer

**Priority:** HIGH  
**Estimate:** 2-3 hours  
**Dependencies:** GPU-INT-001, Renderer frame graph

**Description:**
Wire GPUProfiler into the renderer's frame graph execution so passes are automatically profiled.

**Acceptance Criteria:**
- [ ] Renderer passes command encoder to profiler
- [ ] begin_pass/end_pass called for each render pass
- [ ] Frame timing automatically resolved after command buffer submission
- [ ] Profiler results available via get_average_pass_times()

**Files to Modify:**
- `crates/renderer-backend/src/frame_graph/mod.rs` (or Python equivalent)
- `engine/debug/profiling/gpu.py`

---

---

## engine_platform/PHASE_6_ARCH.md

## Current State (from Investigation)

| Component | Status | Lines | Notes |
|-----------|--------|-------|-------|
| device.py | REAL | 272 | Adapter, Device ABCs + Null |
| resources.py | REAL | 309 | Buffer, Texture, Sampler |
| pipeline.py | REAL | 268 | Shader, PipelineState |
| commands.py | REAL | 300 | CommandList, Queue |
| sync.py | REAL | 126 | Fence with threading |
| swapchain.py | REAL | 139 | Swapchain ABC |
| binding.py | REAL | 95 | DescriptorHeap |
| raytracing.py | REAL | 105 | AccelerationStructure |
| mesh_shaders.py | REAL | 32 | MeshPipelineDesc |

**Integration Point:** `crates/renderer-backend/` provides wgpu-based GPU operations via PyO3.


### ADR-P6-001: Python-Rust Bridge Strategy

**Status:** Proposed

**Context:**
The Rust crate `renderer-backend` exists and likely uses wgpu. Need to bridge Python RHI to Rust.

Options:
1. PyO3 bindings (Rust exposes Python module)
2. cffi to C ABI (Rust exports C functions)
3. IPC/RPC (separate processes)

**Decision:**
Use PyO3 bindings. The Rust crate should expose a `renderer_backend` Python module:

```rust
// In crates/renderer-backend/src/lib.rs
use pyo3::prelude::*;

#[pyclass]
struct WgpuDevice {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

#[pymethods]
impl WgpuDevice {
    fn create_buffer(&self, desc: BufferDesc) -> PyResult<WgpuBuffer> {
        // ...
    }
}

#[pymodule]
fn renderer_backend(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<WgpuDevice>()?;
    m.add_class::<WgpuBuffer>()?;
    // ...
    Ok(())
}
```

**Consequences:**
- Direct Python-Rust calls (no serialization overhead)
- Type-safe via PyO3 derive macros
- Build complexity (maturin or setuptools-rust)
- Platform-specific wheel building


## Component Diagram

```
Python Layer                    Rust Layer
==============                  ==========

engine/platform/rhi/            crates/renderer-backend/
    |                               |
    +-- device.py (ABC)             +-- lib.rs (PyO3 module)
    +-- resources.py (ABC)          +-- device.rs (WgpuDevice)
    +-- commands.py (ABC)           +-- resources.rs (WgpuBuffer, etc.)
    +-- ...                         +-- commands.rs (WgpuCommandEncoder)
    |                               +-- swapchain.rs
    +-- backends/                   +-- ...
        |
        +-- null.py (existing)
        +-- wgpu.py (NEW, wrappers)
            |
            +-- imports renderer_backend
```


### Buffer Creation

```
Python: device.create_buffer(BufferDesc)
    |
    v
WgpuDevice.create_buffer(desc)
    |
    v
    [PyO3 boundary]
    |
    v
Rust: WgpuDevice::create_buffer(&self, desc)
    |
    v
wgpu::Device::create_buffer(&descriptor)
    |
    v
GPU: Buffer allocated
    |
    v
Rust: WgpuBuffer { handle, buffer }
    |
    v
    [PyO3 boundary]
    |
    v
Python: WgpuBuffer(rust_buffer, desc)
```

### Command Submission

```
Python:
    cmd = device.create_command_list()
    cmd.begin()
    cmd.set_pipeline(pipeline)
    cmd.draw(3, 1, 0, 0)
    cmd.end()
    queue.submit([cmd])
    |
    v
    [PyO3 boundary - each call]
    |
    v
Rust:
    encoder.begin_render_pass(...)
    pass.set_pipeline(pipeline)
    pass.draw(3, 1, 0, 0)
    pass.end()
    queue.submit([encoder.finish()])
    |
    v
GPU: Commands executed
```


### Rust Crate Configuration

```toml
# crates/renderer-backend/Cargo.toml
[package]
name = "renderer-backend"

[lib]
crate-type = ["cdylib"]  # For PyO3 module

[dependencies]
pyo3 = { version = "0.20", features = ["extension-module"] }
wgpu = "0.19"

[features]
default = []
vulkan = ["wgpu/vulkan-portability"]
dx12 = []
metal = []
```

### Python Build

```toml
# pyproject.toml (using maturin)
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "renderer-backend"
```

```bash
# Build and install
cd crates/renderer-backend
maturin develop --release
```

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| PyO3 version mismatch | Pin pyo3 version, test in CI |
| wgpu API changes | Vendor wgpu version, update carefully |
| Platform-specific build | CI matrix for Windows/Linux/macOS |
| GPU driver issues | Require recent drivers, document |


---

## engine_platform/PHASE_6_TODO.md

## Summary

Bridge Python RHI abstraction to Rust renderer-backend (wgpu) via PyO3.

**Estimated Effort:** 20-30 hours
**Dependencies:** Phase 1, Phase 4 (window native handles)
**Blocking:** All GPU rendering

---

### T-P6-001: Audit Existing Rust Crate

**Priority:** P0 (Blocking)
**Estimate:** 2 hours

Review `crates/renderer-backend/`:
- Check current PyO3 integration status
- Identify existing exports
- Document gaps vs RHI ABC requirements

**Acceptance Criteria:**
- [ ] Inventory of current Rust exports
- [ ] Gap analysis vs Python RHI ABCs
- [ ] Build instructions verified

---

### T-P6-002: Define Python Interface in Rust

**Priority:** P0 (Blocking)
**Estimate:** 4 hours

Add/update PyO3 bindings in `crates/renderer-backend/src/`:

```rust
#[pyclass]
pub struct WgpuAdapter { ... }

#[pymethods]
impl WgpuAdapter {
    #[staticmethod]
    fn enumerate() -> PyResult<Vec<WgpuAdapter>> { ... }
    fn info(&self) -> PyResult<AdapterInfo> { ... }
    fn create_device(&self) -> PyResult<WgpuDevice> { ... }
}

#[pyclass]
pub struct WgpuDevice { ... }

#[pymethods]
impl WgpuDevice {
    fn create_buffer(&self, size: u64, usage: u32, memory_type: u32) -> PyResult<WgpuBuffer> { ... }
    fn create_texture(&self, desc: TextureDesc) -> PyResult<WgpuTexture> { ... }
    fn create_swapchain(&self, handle: u64, width: u32, height: u32) -> PyResult<WgpuSwapchain> { ... }
    fn create_command_encoder(&self) -> PyResult<WgpuCommandEncoder> { ... }
    fn get_queue(&self) -> PyResult<WgpuQueue> { ... }
}
```

**Acceptance Criteria:**
- [ ] WgpuAdapter with enumerate(), info(), create_device()
- [ ] WgpuDevice with resource creation methods
- [ ] WgpuQueue with submit()
- [ ] All methods have PyResult error handling

---

### T-P6-010: Build and Package Rust Module

**Priority:** P0 (Blocking)
**Estimate:** 2 hours

Set up maturin build:

```toml
# crates/renderer-backend/pyproject.toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "renderer_backend"
version = "0.1.0"
```

```bash
# Build commands
cd crates/renderer-backend
maturin develop --release
```

**Acceptance Criteria:**
- [ ] `maturin develop` succeeds
- [ ] `import renderer_backend` works
- [ ] Module available in venv

---

## Verification Commands

```bash
# Build Rust module
cd crates/renderer-backend && maturin develop --release

# Verify import
uv run python -c "import renderer_backend; print(renderer_backend)"

# Run RHI tests
uv run pytest tests/platform/rhi/ -v

# Manual GPU test (requires display + GPU)
uv run python -c "
from engine.platform.rhi.backends.wgpu import WgpuAdapter
adapters = WgpuAdapter.enumerate()
print(f'Found {len(adapters)} GPU(s)')
for a in adapters:
    print(f'  {a.info().name}')
"
```


---

## engine_rendering_framegraph/PHASE_1_ARCH.md

## Integration Points

- `crates/renderer-backend/src/bridge.rs` — Rust side must call Python context methods via PyO3
- `engine/rendering/rhi/wgpu_context.py` — production implementation (Phase 3)
- `tests/framegraph/mock_context.py` — test implementation (Phase 2)


---

## engine_rendering_framegraph/PHASE_5_ARCH.md

### ADR-FG-017: JSON as IR Transport

**Decision**: Use JSON for Python->Rust IR transport, not raw PyO3 bindings.

**Rationale**:
- JSON is debuggable (human-readable, can log to file)
- Schema validation possible at boundary
- Decouples Python object layout from Rust struct layout
- Avoids complex lifetime management in PyO3 bindings

**Consequences**:
- Serialization cost (~1ms per frame graph)
- Rust side must parse JSON (serde_json)
- Schema must be kept in sync manually (or codegen)

### ADR-FG-018: IR Types Match Exactly

**Decision**: Python `serialize()` output MUST match Rust `IrPass`/`IrResource` exactly.

```rust
// crates/renderer-backend/src/ir.rs
#[derive(Deserialize)]
pub struct IrPass {
    pub name: String,
    pub pass_type: PassType,
    pub reads: Vec<IrResourceAccess>,
    pub writes: Vec<IrResourceAccess>,
    // ...
}
```

**Rationale**:
- Any mismatch causes silent data loss or parse errors
- Rust `#[serde(deny_unknown_fields)]` catches extra Python fields
- Python-side tests validate against Rust schema

**Consequences**:
- Python changes require Rust schema review
- Schema version field recommended for future evolution
- Tests deserialize Python output into Rust types


### ADR-FG-020: Pass Callbacks Not Serialized

**Decision**: Pass execute callbacks stay in Python; Rust schedules, Python executes.

**Rationale**:
- Callbacks are Python functions (can't serialize to Rust)
- Rust compiles the graph (ordering, barriers)
- Python executes passes via callback invocation

**Consequences**:
- Execution model: Rust->Python callback per pass
- PyO3 overhead per callback (acceptable for frame-level granularity)
- GPU commands still go through wgpu (Rust or Python bindings)


## Component Diagram

```
+---------------------+
|  FrameGraph         |
|  serialize()        |
+----------+----------+
           |
           | JSON string
           v
+----------+----------+
|  PyO3 boundary      |
|  (pyo3::types::Str) |
+----------+----------+
           |
           | &str
           v
+----------+----------+
|  serde_json::from   |
|  -> IrFrameGraph    |
+----------+----------+
           |
           | typed IR
           v
+----------+----------+
|  Rust execution     |
|  (command encoding) |
+---------------------+
```

## Files Affected

- `engine/rendering/framegraph/frame_graph.py` — `serialize()` method refinements
- `crates/renderer-backend/src/ir.rs` — ensure IR types match
- `crates/renderer-backend/src/bridge.rs` — PyO3 entry point for serialization
- `tests/framegraph/test_serialization.py` — round-trip validation tests

## Integration Points

- `serialize()` returns `str` (JSON)
- Rust `FrameGraphBridge::from_json(json: &str) -> Result<IrFrameGraph>`
- PyO3 `#[pyfunction]` wraps Rust entry point


---

## engine_rendering_framegraph/PHASE_5_TODO.md

## T-FG-5.5: Verify Rust IrFrameGraph Matches

**File**: `crates/renderer-backend/src/ir.rs`

**Tasks**:
- [ ] Add `#[serde(deny_unknown_fields)]` to all IR structs
- [ ] Ensure IrPass has: name, pass_type, reads, writes, execution_order, queue
- [ ] Ensure IrResource has: name, type, format, dimensions, allocation
- [ ] Ensure IrSyncPoint has: signal_queue, wait_queue, fence_value

**Acceptance Criteria**:
- Unknown Python fields cause parse error (not silent ignore)
- All expected fields are present
- Field types match (string/int/nested struct)

---

## T-FG-5.6: Create Round-Trip Test Suite

**File**: `tests/framegraph/test_serialization.py` (new)

**Tasks**:
- [ ] Create test building a complete frame graph
- [ ] Serialize to JSON
- [ ] Call Rust `from_json()` via PyO3 binding
- [ ] Verify no errors and data matches
- [ ] Test edge cases: empty graph, many passes, large resources

**Acceptance Criteria**:
- Round-trip test passes
- Edge cases covered
- Failures identify which field mismatched

---

## T-FG-5.7: PyO3 Bridge Entry Point

**File**: `crates/renderer-backend/src/bridge.rs`

**Tasks**:
- [ ] Add `#[pyfunction] fn load_frame_graph(json: &str) -> PyResult<()>`
- [ ] Parse JSON into IrFrameGraph
- [ ] Return Python-friendly error on parse failure
- [ ] Export function in module

**Acceptance Criteria**:
- Python can call `renderer_backend.load_frame_graph(json_str)`
- Parse errors include line/column and field name
- Success returns None (data stored in Rust for execution)

---

---

## engine_rendering_materials/PHASE_1_ARCH.md

## Integration Points

- `engine/rendering/gpu_driven/texture_table.py` - Texture binding slots
- `engine/rendering/frame_graph` - Render pass material binding
- `crates/renderer-backend` - Uniform buffer upload


---

## engine_rendering_materials/PHASE_2_ARCH.md

## Integration Points

- `engine/rendering/gpu_driven/texture_table.py` - Bindless texture slots
- `engine/rendering/materials/material_graph.py` - OutputNode PBR slots
- `crates/renderer-backend/src/gpu_driven/texture_table.rs` - Rust binding


---

## engine_rendering_materials/PHASE_6_ARCH.md

## Integration Points

- `engine/rendering/materials/material_graph.py` - GLSL source generation
- `crates/renderer-backend` - GPU bytecode upload
- `engine/rendering/frame_graph` - Per-pass shader selection


---

## engine_ui_widgets/PHASE_4_ARCH.md

## Dependencies

- Phase 1-3 complete (layout, focus, input)
- Rust renderer backend with PyO3 bindings
- Widget `get_geometry()` methods
- Widget dirty-tracking system


---

## engine_ui_widgets/PHASE_4_TODO.md

## Task 4.10: Rust FFI Bridge

**File**: `crates/renderer-backend/src/ui_bridge.rs`

**Description**: Rust side of geometry submission.

**Acceptance Criteria**:
- [ ] `submit_ui_frame(buffer: &[u8])` PyO3 function
- [ ] `deserialize_geometries(buffer) -> Vec<UIGeometry>` parses Python data
- [ ] Geometry struct matches Python UIGeometry
- [ ] Validation of buffer format
- [ ] Queues geometries for next render pass

**Evidence of Completion**: Python `submit_ui_frame(packed)` queues geometries in Rust renderer.

---

---

## engine_world/PHASE_4_ARCH.md

## Phase Overview

Phase 4 prepares the world subsystem for selective migration to Rust via PyO3 bindings. The goal is NOT to rewrite everything in Rust, but to identify performance-critical hot paths and establish the bridge infrastructure.


### ADR-W4-002: Bridge Architecture

**Context**: PyO3 provides multiple binding styles. Choice affects maintainability and performance.

**Decision**: Use function-level bindings, not class-level:
```rust
#[pyfunction]
fn perlin_noise_batch(positions: Vec<(f64, f64)>, seed: u64) -> Vec<f64>
```

**Rationale**:
- Functions have less overhead than method dispatch
- Python side retains all state management
- Easier to test Rust and Python independently
- Incremental migration possible

### ADR-W4-003: Data Transfer Format

**Context**: Crossing Python-Rust boundary has overhead. Large data transfers should minimize conversions.

**Decision**: Use typed arrays for bulk data:
- Input: `Vec<(f64, f64)>` for positions (PyO3 converts from list of tuples)
- Output: `Vec<f64>` for results (PyO3 converts to list)
- Avoid custom Python classes in bridge signatures

**Future**: Consider `numpy` arrays via `rust-numpy` if benchmarks show list conversion is bottleneck.

### ADR-W4-004: Error Handling

**Context**: Rust errors must be translated to Python exceptions.

**Decision**: Define custom exception types:
```rust
#[pyclass]
struct WorldBridgeError {
    message: String,
}

impl From<WorldBridgeError> for PyErr {
    fn from(err: WorldBridgeError) -> PyErr {
        pyo3::exceptions::PyRuntimeError::new_err(err.message)
    }
}
```


## Bridge Module Structure

```
crates/world-bridge/
├── Cargo.toml
├── src/
│   ├── lib.rs           # PyO3 module definition
│   ├── noise.rs         # Noise generation functions
│   ├── hlod.rs          # Mesh simplification
│   ├── pathfinding.rs   # A* implementation
│   └── culling.rs       # Frustum culling
└── tests/
    └── property_tests.rs

engine/world/
├── _bridge.pyi          # Type stubs for IDE support
└── bridge.py            # Python wrapper with fallback
```


## Success Criteria

- [ ] Bridge candidates identified from Phase 3 benchmarks
- [ ] Rust crate compiles with PyO3
- [ ] At least one function bridged (noise or pathfinding)
- [ ] Property tests pass (Rust == Python results)
- [ ] Performance benchmark shows speedup
- [ ] Crossover threshold documented
- [ ] Python fallback works when bridge unavailable


---

## engine_world/PHASE_4_TODO.md

### T-W4-002: Rust Crate Setup
**Description**: Create world-bridge Rust crate with PyO3
**Files**: `crates/world-bridge/`
**Dependencies**: T-W4-001
**Acceptance Criteria**:
- [ ] Create `Cargo.toml` with PyO3 dependency
- [ ] Create `lib.rs` with module definition
- [ ] Add to workspace `Cargo.toml`
- [ ] Verify `cargo build` succeeds
- [ ] Verify `maturin develop` installs Python module

### T-W4-003: Noise Bridge Implementation
**Description**: Implement Perlin noise batch sampling in Rust
**Files**: `crates/world-bridge/src/noise.rs`
**Dependencies**: T-W4-002
**Acceptance Criteria**:
- [ ] Implement `perlin_noise_batch` function
- [ ] Match Python implementation exactly (same permutation table, fade function)
- [ ] Expose via PyO3 `#[pyfunction]`
- [ ] Unit test against known outputs
- [ ] Property test against Python implementation

### T-W4-004: Simplex Noise Bridge
**Description**: Implement Simplex noise batch sampling in Rust
**Files**: `crates/world-bridge/src/noise.rs`
**Dependencies**: T-W4-003
**Acceptance Criteria**:
- [ ] Implement `simplex_noise_batch` function
- [ ] Match Python skewing factors exactly
- [ ] Property test against Python implementation
- [ ] Benchmark shows >10x speedup

### T-W4-005: Worley Noise Bridge
**Description**: Implement Worley noise batch sampling in Rust
**Files**: `crates/world-bridge/src/noise.rs`
**Dependencies**: T-W4-003
**Acceptance Criteria**:
- [ ] Implement `worley_noise_batch` function
- [ ] Support all distance metrics (Euclidean, Manhattan, Chebyshev)
- [ ] Property test against Python implementation
- [ ] Benchmark shows >10x speedup


### T-W4-008: QEM Mesh Simplification Bridge
**Description**: Implement QEM mesh simplification in Rust
**Files**: `crates/world-bridge/src/hlod.rs`
**Dependencies**: T-W4-002
**Acceptance Criteria**:
- [ ] Implement `simplify_mesh_qem` function
- [ ] Input: vertices, indices, target_ratio
- [ ] Output: simplified vertices, indices
- [ ] Property test: output triangle count = input * ratio (approximately)
- [ ] Property test: output bounds similar to input bounds
- [ ] Benchmark shows significant speedup

### T-W4-009: A* Pathfinding Bridge
**Description**: Implement A* pathfinding in Rust
**Files**: `crates/world-bridge/src/pathfinding.rs`
**Dependencies**: T-W4-002
**Acceptance Criteria**:
- [ ] Implement `astar_pathfind` function
- [ ] Input: start, goal, walkable grid, cell_size
- [ ] Output: path points
- [ ] Property test: path start == input start, path end == input goal
- [ ] Property test: all path points on walkable cells
- [ ] Benchmark shows speedup for large grids

### T-W4-010: Frustum Culling Bridge
**Description**: Implement batch frustum culling in Rust
**Files**: `crates/world-bridge/src/culling.rs`
**Dependencies**: T-W4-002
**Acceptance Criteria**:
- [ ] Implement `frustum_cull_aabbs` function
- [ ] Input: 6 frustum planes, list of AABBs
- [ ] Output: list of booleans (visible/not)
- [ ] Property test: visible AABBs intersect frustum
- [ ] Property test: invisible AABBs outside frustum
- [ ] Benchmark shows speedup for large instance counts

### T-W4-011: Bridge Error Handling
**Description**: Implement error handling for bridge functions
**Files**: `crates/world-bridge/src/lib.rs`
**Dependencies**: T-W4-002
**Acceptance Criteria**:
- [ ] Define `WorldBridgeError` type
- [ ] Implement `From<WorldBridgeError> for PyErr`
- [ ] Bridge functions return proper errors on invalid input
- [ ] Python side receives appropriate exceptions
- [ ] Test error handling paths

### T-W4-012: Property Test Suite
**Description**: Create comprehensive property tests for bridge
**Files**: `crates/world-bridge/tests/property_tests.rs`
**Dependencies**: T-W4-003 through T-W4-010
**Acceptance Criteria**:
- [ ] Add `quickcheck` dependency
- [ ] Property tests for all bridged functions
- [ ] Tests verify Rust == Python output
- [ ] Floating point comparison with tolerance (1e-10)
- [ ] CI runs property tests

### T-W4-013: Bridge Documentation
**Description**: Document bridge architecture and usage
**Files**: `crates/world-bridge/README.md`, docstrings
**Dependencies**: All bridge tasks complete
**Acceptance Criteria**:
- [ ] README explains bridge architecture
- [ ] README documents crossover thresholds
- [ ] All functions have Rust doc comments
- [ ] Python stubs have docstrings
- [ ] Performance comparison documented


---

