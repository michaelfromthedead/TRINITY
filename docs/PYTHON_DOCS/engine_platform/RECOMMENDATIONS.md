# RECOMMENDATIONS: engine_platform

## Rust Bridge Requirements

### High Priority

| Component | Rust Requirement | Justification |
|-----------|------------------|---------------|
| RHI Device | PyO3 wrapper for wgpu/ash | GPU operations need native performance |
| RHI Resources | Buffer/Texture bindings | Memory management in Rust |
| RHI Commands | CommandEncoder bindings | GPU command submission |
| Audio Backends | cpal or rodio integration | Low-latency audio requires native code |

### Medium Priority

| Component | Rust Requirement | Justification |
|-----------|------------------|---------------|
| File System | async-std/tokio file ops | High-throughput asset loading |
| Virtual Memory | mmap crate | Large texture streaming |
| Threading | rayon integration | Parallel job system |
| Low Latency | NVIDIA/AMD SDK bindings | Frame latency reduction |

### Low Priority

| Component | Rust Requirement | Justification |
|-----------|------------------|---------------|
| File Watcher | notify crate | Native OS event APIs |
| System Info | sysinfo crate | More accurate metrics |
| Dynamic Library | libloading | Plugin system |

## Integration Strategy

### Phase 1: Core RHI Bridge
1. Create `renderer-backend/src/python.rs` module with PyO3 bindings
2. Expose wgpu Device/Queue/Buffer/Texture via Python classes
3. Maintain Python abstract interfaces - Rust becomes another backend
4. NullDevice continues to work for tests

```rust
#[pyclass]
struct WgpuDevice {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

#[pymethods]
impl WgpuDevice {
    fn create_buffer(&self, desc: BufferDesc) -> PyResult<WgpuBuffer> { ... }
    fn create_texture(&self, desc: TextureDesc) -> PyResult<WgpuTexture> { ... }
}
```

### Phase 2: Audio Bridge
1. Use cpal for cross-platform audio I/O
2. Expose stream callbacks via Python
3. Keep NullAudioBackend for CI

### Phase 3: OS Primitives
1. Replace file I/O with Rust async where needed
2. Expose Rust threading primitives for job system
3. Native file watching with notify crate

## Testing Strategy

### Unit Tests (Python)
```python
# Null backend tests remain in Python
def test_buffer_creation():
    device = NullDevice.create(adapter, config)
    buffer = device.create_buffer(BufferDesc(size=1024, usage=BufferUsage.VERTEX))
    assert buffer.is_valid()
    assert buffer.desc.size == 1024
```

### Integration Tests (Rust + Python)
```python
# Test Rust backend through Python API
@pytest.mark.skipif(not has_gpu(), reason="No GPU")
def test_wgpu_buffer_creation():
    device = WgpuDevice.create(adapter)
    buffer = device.create_buffer(BufferDesc(size=1024, usage=BufferUsage.VERTEX))
    assert buffer.is_valid()
```

### Blackbox Tests
1. Create resources through Python API
2. Verify via Rust FFI that GPU state is correct
3. Test swapchain present cycle end-to-end

### Whitebox Tests
1. Direct Rust unit tests for wgpu wrapper
2. Memory leak detection via GPU memory tracking
3. Thread safety tests with concurrent access

## Risk Assessment

### High Risk
| Risk | Mitigation |
|------|------------|
| PyO3 GIL contention in audio callbacks | Use `allow_threads` for callback dispatch |
| GPU resource lifetime mismatch | Arc/weak patterns with explicit destroy() |
| Platform-specific bugs in native backends | Extensive CI matrix (Windows/Linux/macOS) |

### Medium Risk
| Risk | Mitigation |
|------|------------|
| wgpu version compatibility | Pin wgpu version, test upgrades |
| Audio underruns in Python callbacks | Rust-side ring buffer with Python polling |
| File watcher event flooding | Debounce/coalesce in Rust layer |

### Low Risk
| Risk | Mitigation |
|------|------------|
| NullDevice behavior drift | Continuous integration of null vs native tests |
| API surface changes | Python abstract interfaces are stable contracts |
| Performance regression | Benchmark suite comparing null vs native |

## Migration Path

1. **Week 1-2**: PyO3 skeleton for RHI
2. **Week 3-4**: Buffer/Texture creation and mapping
3. **Week 5-6**: Command encoding and submission
4. **Week 7-8**: Swapchain and presentation
5. **Week 9-10**: Audio backend with cpal
6. **Week 11-12**: Integration testing and stabilization

## Success Criteria

- [ ] All existing Python tests pass with Rust backend
- [ ] GPU frame time < 16ms at 1080p
- [ ] Audio latency < 20ms
- [ ] Zero memory leaks in 24hr stress test
- [ ] CI passes on Windows/Linux/macOS
