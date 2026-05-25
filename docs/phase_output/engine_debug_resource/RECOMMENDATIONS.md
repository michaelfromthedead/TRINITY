# RECOMMENDATIONS.md - engine/debug/ and engine/resource/ Subsystems

## Rust Bridge Requirements

### HIGH Priority

| Requirement | Source Component | Target | Rationale |
|-------------|------------------|--------|-----------|
| Asset Handle FFI | AssetManager, AssetHandle | renderer-backend | GPU resources referenced by Python must resolve to Rust indices |
| Streaming Queue Export | StreamManager | async loader | Rust-side async IO should consume Python-prioritized requests |
| Virtual Texture Feedback | VirtualTextureSystem | texture_table.rs | GPU feedback buffer must inform Python page allocation |
| Page Table State | PageTable | gpu_driven | Rust renderer needs page-to-physical mapping for sampling |

### MEDIUM Priority

| Requirement | Source Component | Target | Rationale |
|-------------|------------------|--------|-----------|
| Profiling Data Export | CPUProfiler, GPUProfiler | overlay renderer | Real-time profiling overlay should render via wgpu |
| Debug Draw Batching | DebugDraw | immediate renderer | Debug primitives must reach GPU each frame |
| CVar Change Events | CVarRegistry | bridge config | Graphics CVars (vsync, resolution) must propagate to Rust |
| Budget Pressure Alerts | BudgetManager | memory pool | Rust can adjust streaming if Python reports high pressure |

### LOW Priority

| Requirement | Source Component | Target | Rationale |
|-------------|------------------|--------|-----------|
| Crash Context Export | CrashHandler | crash reporter | Native crash reports should include Python context |
| Log Sink Bridge | Logger | tracing crate | Unified logging across Python/Rust |
| Test Harness Hooks | TestRunner | integration tests | Rust tests may need Python fixture setup |

---

## Integration Strategy

### Phase 1: Data Types (Week 1-2)

Define shared types in `bridge.rs`:

```rust
// Asset reference crossing the FFI boundary
#[repr(C)]
pub struct BridgedAssetHandle {
    pub index: u32,
    pub generation: u16,
    pub asset_type: u16,  // enum discriminant
}

// Streaming request from Python
#[repr(C)]
pub struct StreamRequest {
    pub asset_id: u64,
    pub priority: u8,
    pub stream_type: u8,
    pub bytes_total: u32,
}

// Virtual texture page mapping
#[repr(C)]
pub struct PageMapping {
    pub page_x: u16,
    pub page_y: u16,
    pub mip_level: u8,
    pub physical_x: u16,
    pub physical_y: u16,
}
```

### Phase 2: Read Path (Week 2-3)

Python -> Rust data flow:

1. **Asset Handles**: Python `AssetManager.load()` returns handle; Rust queries handle state via FFI
2. **Stream Queue**: Python `StreamManager.update()` exports pending requests array
3. **Page Table**: Python `PageTable.get_resident_pages()` exports mappings array

### Phase 3: Write Path (Week 3-4)

Rust -> Python callbacks:

1. **Stream Complete**: Rust async loader notifies Python via callback
2. **GPU Feedback**: Rust samples feedback buffer, calls Python page request
3. **Memory Pressure**: Rust reports VRAM usage, Python adjusts budgets

### Phase 4: Debug Tools (Week 4-5)

1. **Debug Draw**: Batch Python primitives, submit via wgpu immediate mode
2. **Profiling Overlay**: Render Python timing data as GPU text/charts
3. **CVar Sync**: Python CVar changes trigger Rust config reload

---

## Testing Strategy

### Unit Tests (Python Side)

```python
# test_asset_bridge.py
def test_handle_roundtrip():
    """Handle created in Python validates in Rust"""
    mgr = AssetManager()
    handle = mgr.load("test.png", TextureAsset)
    # FFI call to Rust validation
    assert rust_validate_handle(handle.index, handle.generation)

def test_stream_request_encoding():
    """StreamRequest serializes correctly for FFI"""
    req = StreamRequest(asset_id="tex_001", priority=StreamPriority.HIGH)
    encoded = encode_stream_request(req)
    assert len(encoded) == 16  # Fixed struct size
```

### Integration Tests (Rust Side)

```rust
#[test]
fn test_python_asset_resolution() {
    let py = Python::acquire_gil();
    let handle = py.call("asset_manager.load", ("mesh.obj",));
    let rust_id = bridge.resolve_handle(handle);
    assert!(rust_id.is_some());
}

#[test]
fn test_page_table_consistency() {
    // Python allocates pages
    py.call("vt_system.request_page", (0, 0, 0));
    py.call("vt_system.update", ());
    
    // Rust reads mappings
    let mappings = bridge.get_page_mappings();
    assert!(!mappings.is_empty());
}
```

### Stress Tests

```python
def test_streaming_throughput():
    """Measure stream request throughput"""
    mgr = StreamManager()
    start = time.perf_counter()
    for i in range(10000):
        mgr.request_stream(f"asset_{i}", StreamPriority.NORMAL)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.1  # 100k requests/sec target
```

---

## Risk Assessment

### HIGH Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| Handle lifetime mismatch | Use-after-free, corruption | Generation checks at FFI boundary |
| GIL contention in streaming | Frame stalls | Release GIL during Rust calls |
| Page table race conditions | Visual corruption | Single-writer (Python), atomic reads |

### MEDIUM Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| Memory budget disagreement | OOM on one side | Unified budget source-of-truth |
| Profiling overhead | False performance data | Separate debug/release profilers |
| CVar sync timing | One-frame-off settings | Apply CVars at frame boundary |

### LOW Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| Debug draw Z-fighting | Visual artifacts | Depth bias configuration |
| Log message ordering | Confusing logs | Timestamp synchronization |
| Test fixture timing | Flaky tests | Explicit synchronization points |

---

## Recommended Implementation Order

1. **BridgedAssetHandle** struct + validation FFI (enables basic integration)
2. **StreamRequest** export + Rust consumer (enables async loading)
3. **PageMapping** export (enables virtual texturing)
4. **Debug draw batch export** (enables visual debugging)
5. **Profiling data export** (enables performance monitoring)
6. **CVar sync mechanism** (enables runtime configuration)
7. **Crash context export** (enables unified crash reporting)
