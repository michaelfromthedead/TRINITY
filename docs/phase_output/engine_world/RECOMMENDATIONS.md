# RECOMMENDATIONS: engine_world

## Rust Bridge Requirements

### High Priority

| Component | Current State | Rust Acceleration Benefit | Estimated Speedup |
|-----------|---------------|---------------------------|-------------------|
| **Heightfield.get_height_at()** | Python bilinear interp | Called per-character per-frame | 50-100x |
| **MeshSimplifier.simplify()** | Python QEM edge collapse | HLOD generation bottleneck | 20-50x |
| **PerlinNoise.sample()** | Python gradient calc | World gen hot path | 100-500x |
| **Quadtree.select_lod()** | Python tree traversal | Per-frame view culling | 10-30x |
| **SpatialQuerySystem.execute_raycast()** | Python list iteration | Per-frame line-of-sight | 20-50x |

### Medium Priority

| Component | Current State | Rust Acceleration Benefit | Estimated Speedup |
|-----------|---------------|---------------------------|-------------------|
| **Heightfield.compress()** | Python zlib wrapper | Streaming I/O bound | 2-5x |
| **SimplexNoise.sample()** | Python skewing math | Alternative noise type | 100-500x |
| **WorldStreaming.update()** | Python priority queue | Per-frame cell management | 5-10x |
| **TerrainQuadtree construction** | Python recursive build | Level load time | 10-20x |
| **MeshMerger.merge_meshes()** | Python vertex dedup | HLOD generation | 10-30x |

### Low Priority

| Component | Current State | Rust Acceleration Benefit | Estimated Speedup |
|-----------|---------------|---------------------------|-------------------|
| **NoiseMap.generate()** | Python bulk sampling | Bake-time only | 100x (but rare) |
| **ImpostorGenerator** | CPU rasterization | Should be GPU instead | N/A |
| **FoliageManager** | Python instance management | Per-frame culling | 5-10x |

## Integration Strategy

### Phase 1: Core Math Acceleration (Week 1-2)

1. **Create `rust_world` module in renderer-backend crate**
   ```rust
   // crates/renderer-backend/src/world/mod.rs
   pub mod heightfield;
   pub mod noise;
   pub mod quadtree;
   ```

2. **Expose via PyO3 bindings**
   ```python
   # trinity/descriptors/rust_world.py
   from renderer_backend import RustHeightfield, RustPerlinNoise
   ```

3. **Wrap existing Python classes to delegate hot paths**
   ```python
   class Heightfield:
       def __init__(self, config):
           self._rust_impl = RustHeightfield(config.resolution, config.scale)
       
       def get_height_at(self, x, z):
           return self._rust_impl.sample(x, z)  # Delegate to Rust
   ```

### Phase 2: Spatial Structures (Week 3-4)

1. **Port Quadtree to Rust with LOD selection**
2. **Implement Rust BVH for spatial queries**
3. **Add batch query API for multiple rays**

### Phase 3: HLOD Pipeline (Week 5-6)

1. **Port MeshSimplifier with parallel edge collapse**
2. **Port MeshMerger with spatial hashing for vertex dedup**
3. **Add incremental simplification for runtime updates**

## Testing Strategy

### Unit Tests (Per Component)

```python
# tests/blackbox_heightfield_rust.py
def test_heightfield_bilinear_matches_python():
    py_hf = Heightfield(HeightfieldConfig(resolution=65))
    rust_hf = RustHeightfield(65, 1.0)
    
    # Set identical data
    for z in range(65):
        for x in range(65):
            h = math.sin(x * 0.1) * math.cos(z * 0.1)
            py_hf.set_height_at(x, z, h)
            rust_hf.set_height_at(x, z, h)
    
    # Query at random points
    for _ in range(1000):
        qx, qz = random.uniform(0, 64), random.uniform(0, 64)
        assert abs(py_hf.get_height_at(qx, qz) - rust_hf.sample(qx, qz)) < 1e-5

def test_perlin_noise_determinism():
    rust_noise = RustPerlinNoise(seed=12345)
    values = [rust_noise.sample(i * 0.1, 0.0) for i in range(100)]
    
    # Same seed should produce identical values
    rust_noise_2 = RustPerlinNoise(seed=12345)
    values_2 = [rust_noise_2.sample(i * 0.1, 0.0) for i in range(100)]
    
    assert values == values_2
```

### Integration Tests (Cross-System)

```python
# tests/integration_terrain_streaming.py
def test_terrain_streams_with_rust_heightfield():
    """Verify streaming system works with Rust-backed heightfield."""
    grid = WorldGrid(cell_size=256.0)
    streaming = WorldStreaming(grid)
    
    # Create terrain with Rust heightfield
    terrain = TerrainActor(use_rust_heightfield=True)
    
    # Simulate player movement
    for frame in range(100):
        player_pos = Vec3(frame * 10.0, 0, 0)
        streaming.update(0.016)  # 60fps
        
        # Verify terrain queries work
        height = terrain.get_height_at(player_pos.x, player_pos.z)
        assert height is not None
```

### Performance Benchmarks

```python
# benches/bench_heightfield.py
def bench_heightfield_query(benchmark):
    hf = Heightfield(HeightfieldConfig(resolution=4097))  # 4K terrain
    hf.fill(0.0)
    
    def query_grid():
        for z in range(0, 4096, 8):
            for x in range(0, 4096, 8):
                _ = hf.get_height_at(x + 0.5, z + 0.5)
    
    benchmark(query_grid)

# Expected results:
# Python: ~500ms for 512x512 queries
# Rust:   ~5ms for 512x512 queries
```

## Risk Assessment

### High Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| **API compatibility breaks** | Python code stops working | Maintain identical Python API, delegate to Rust internally |
| **Floating point divergence** | Visual popping at boundaries | Use consistent rounding modes, add tolerance tests |
| **Memory layout mismatch** | Data corruption | Use explicit byte-order in serialization |

### Medium Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| **GIL contention** | Rust speedup negated | Release GIL during long operations via `py.allow_threads()` |
| **Panic propagation** | Python crashes | Convert Rust panics to Python exceptions |
| **Build complexity** | CI/CD breaks | Add manylinux wheel builds, test on all platforms |

### Low Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Algorithm differences** | Slightly different output | Document intentional improvements, version API |
| **Missing features** | Fall back to Python | Hybrid approach: Rust for hot paths, Python for cold |

## Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Heightfield query latency | 50us | 0.5us | `timeit` micro-benchmark |
| Perlin noise sample rate | 100K/s | 50M/s | `criterion` benchmark |
| HLOD simplification time | 30s/mesh | 0.5s/mesh | Wall clock on reference mesh |
| Frame time (world update) | 5ms | 0.5ms | GPU profiler |
| Memory overhead (Rust wrapper) | 0% | <5% | `tracemalloc` delta |
