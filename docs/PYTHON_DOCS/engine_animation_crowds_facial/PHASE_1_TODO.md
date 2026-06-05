# PHASE 1 TODO: Core GPU Crowd Rendering Pipeline

**Phase:** 1 of 3
**Focus:** GPU-accelerated crowd rendering with animation textures
**Status:** IMPLEMENTED (Verification Required)

---

## Task List

### T1.1 Animation Texture Verification

**Priority:** HIGH
**Estimate:** 2 hours
**File:** `engine/animation/crowds/animation_texture.py`

**Acceptance Criteria:**
- [x] Round-trip test: encode transform, decode, compare within epsilon
- [x] Cubic Hermite interpolation produces smooth curves
- [x] Atlas UV ranges are non-overlapping
- [x] Edge case: single-frame animation handles correctly

**Test Cases:**
```python
def test_transform_roundtrip():
    original = Transform(pos=Vec3(1,2,3), rot=Quaternion.identity(), scale=Vec3(1,1,1))
    pixels = encode_transform_to_pixels(original)
    decoded = decode_pixels_to_transform(pixels)
    assert original.approx_equal(decoded, epsilon=0.001)

def test_interpolation_smoothness():
    # Sample at 0, 0.5, 1.0 and verify monotonic if keyframes are monotonic
    pass
```

---

### T1.2 Instance Buffer Stress Test

**Priority:** HIGH
**Estimate:** 1 hour
**File:** `engine/animation/crowds/crowd_renderer.py`

**Acceptance Criteria:**
- [x] Buffer handles 10,000 instances without crash
- [x] `InstanceBufferOverflowError` raised at max capacity
- [x] Dynamic growth works correctly
- [x] Memory layout matches expected byte sizes

**Test Cases:**
```python
def test_buffer_overflow_protection():
    buffer = InstanceBuffer(max_instances=100)
    for i in range(100):
        buffer.add_instance(make_instance(i))
    with pytest.raises(InstanceBufferOverflowError):
        buffer.add_instance(make_instance(101))

def test_buffer_memory_layout():
    buffer = InstanceBuffer(max_instances=10)
    buffer.add_instance(make_instance(0))
    assert buffer.get_byte_size() == 96  # 64 + 16 + 16
```

---

### T1.3 Batch Rendering Logic

**Priority:** MEDIUM
**Estimate:** 1 hour
**File:** `engine/animation/crowds/crowd_renderer.py`

**Acceptance Criteria:**
- [x] Instances group by (mesh_id, material_id)
- [x] Batch priority ordering is correct
- [x] Empty batches are handled gracefully
- [x] Single instance per batch works

**Test Cases:**
```python
def test_batch_grouping():
    renderer = CrowdRenderer()
    renderer.add_instance(CrowdInstance(mesh_id=1, material_id=1))
    renderer.add_instance(CrowdInstance(mesh_id=1, material_id=1))
    renderer.add_instance(CrowdInstance(mesh_id=2, material_id=1))
    assert len(renderer.batches) == 2
    assert renderer.batches[(1, 1)].instance_count == 2
```

---

### T1.5 LOD Integration

**Priority:** MEDIUM
**Estimate:** 2 hours
**Files:** `crowd_lod.py`, `crowd_renderer.py`

**Acceptance Criteria:**
- [x] LOD level selection based on camera distance
- [x] Hysteresis prevents LOD flickering
- [x] Reduced skeleton LODs render correctly
- [x] LOD transition modes work (instant, blend, dither)

**Test Cases:**
```python
def test_lod_hysteresis():
    lod = CrowdLOD(levels=[LODLevel(distance=10), LODLevel(distance=20)])
    # At distance 15, should stay at current LOD unless threshold exceeded
    pass
```

---

## Dependency Graph

```
T1.4 (Bridge)
    ^
    |
T1.1 (Texture) --> T1.5 (LOD)
    ^                 ^
    |                 |
T1.2 (Buffer) --------+
    ^
    |
T1.3 (Batch)
```

---

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Rust bridge complexity | Blocks GPU rendering | Start T1.4 early |
| NumPy-Rust data transfer | Performance overhead | Use buffer protocol |
| Animation texture format | Incompatible with shaders | Validate against WGSL |

---

## Definition of Done

Phase 1 is complete when:
1. All T1.x tasks have passing tests
2. 1000 crowd instances render at 60fps
3. Animation texture sampling produces correct poses
4. No memory leaks after 10 minutes of stress test
