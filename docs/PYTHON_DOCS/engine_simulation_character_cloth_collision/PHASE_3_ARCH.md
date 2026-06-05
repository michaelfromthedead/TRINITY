# PHASE 3 ARCHITECTURE: Test Coverage Expansion

## Objective

Add comprehensive test coverage for edge cases and degenerate inputs across all physics simulation modules.

## Current State

The investigation identified production-quality implementations but noted the need for edge case testing. Specific areas of concern:

1. **GJK/EPA**: Degenerate simplex cases (coplanar points, near-collinear edges)
2. **Cloth Simulation**: Zero-area triangles, collapsed edges
3. **Broadphase**: Algorithm selection benchmarking
4. **Character Controller**: Edge case state transitions

## Architecture Decisions

### ADR-TEST-001: Property-Based Testing for Math Invariants

**Decision**: Use hypothesis library for property-based tests on mathematical operations.

**Rationale**:
- Generates edge cases automatically
- Tests mathematical invariants (associativity, identity, inverse)
- Catches subtle numerical issues

**Example Properties**:
- `dot(a, b) == dot(b, a)` (commutativity)
- `(q1 * q2) * q3 == q1 * (q2 * q3)` (associativity)
- `q * q.conjugate() == identity` (within epsilon)

### ADR-TEST-002: Fuzz Testing for Collision Detection

**Decision**: Use fuzzing to generate random collision scenarios.

**Rationale**:
- Collision code must handle arbitrary geometry
- Manually constructed tests miss subtle cases
- Fuzz testing finds numerical instabilities

**Coverage Targets**:
- GJK with degenerate Minkowski differences
- EPA with coplanar polytope faces
- SAT with parallel face normals

### ADR-TEST-003: Benchmark Suite for Broadphase Selection

**Decision**: Create automated benchmark suite comparing all four broadphase algorithms.

**Metrics Collected**:
- Insert time (single object)
- Batch insert time (N objects)
- Query time (AABB, ray)
- Pair finding time
- Memory usage

**Scene Types**:
- Uniform grid (best for spatial hash)
- Random scattered (varies)
- Clustered groups (best for octree/BVH)
- Mostly static (best for SAP)

### ADR-TEST-004: Visual Regression for Cloth Simulation

**Decision**: Compare cloth simulation output against golden reference frames.

**Rationale**:
- Cloth behavior changes are visually subtle
- Numerical drift can be hard to detect
- Reference frames capture expected behavior

**Implementation**:
- Serialize particle positions to JSON
- Compare against golden files with epsilon tolerance
- Flag visual regressions for manual review

## Test Categories

### Unit Tests

| Module | Focus Areas |
|--------|-------------|
| math | Vector operations, quaternion math, transform composition |
| broadphase | Insert/remove/update correctness, query accuracy |
| narrowphase | Contact point generation, penetration depth |
| cloth | Constraint satisfaction, particle behavior |
| character | State transitions, collision response |

### Integration Tests

| Test | Components |
|------|------------|
| Character-Collision | Character controller + broadphase + narrowphase |
| Cloth-Collision | Cloth particles + collision shapes |
| Physics-Animation | Ragdoll + animation blend + PD control |

### Stress Tests

| Test | Description |
|------|-------------|
| 10K Particles | Cloth simulation stability at scale |
| 1K Colliders | Broadphase scaling behavior |
| 100 Characters | Character controller performance |

## Edge Case Inventory

### GJK Degenerate Cases

1. **Collinear support points**: Simplex degenerates to line
2. **Coplanar support points**: Simplex degenerates to triangle
3. **Near-touching shapes**: Numerical precision at contact
4. **Concentric shapes**: Origin inside Minkowski difference

### EPA Edge Cases

1. **Flat polytope faces**: Degenerate face normals
2. **Very small penetration**: Near-zero depth
3. **Multiple equal-depth faces**: Ambiguous contact normal
4. **Origin on polytope edge**: Edge case contact point

### Cloth Degenerate Cases

1. **Zero-area triangle**: Collapsed cloth region
2. **Inverted triangle**: Negative area (wind direction flip)
3. **Zero rest length edge**: Collapsed constraint
4. **All particles pinned**: No simulation degrees of freedom
5. **Single free particle**: Minimal system

### Character Edge Cases

1. **Zero-length movement**: No-op optimization
2. **Vertical collision**: Ceiling hits
3. **Multiple simultaneous contacts**: Resolution ordering
4. **Slope exactly at limit**: Boundary condition
5. **Moving platform detachment**: Velocity inheritance

## Test Infrastructure

```
tests/
  engine/
    simulation/
      collision/
        test_broadphase_correctness.py
        test_broadphase_benchmark.py
        test_narrowphase_gjk.py
        test_narrowphase_epa.py
        test_ccd_edge_cases.py
      cloth/
        test_cloth_constraints.py
        test_cloth_collision.py
        test_cloth_degenerate.py
        test_cloth_golden.py
      character/
        test_movement_modes.py
        test_collision_response.py
        test_ragdoll_activation.py
    math/
      test_vector_properties.py
      test_quaternion_properties.py
      test_transform_properties.py
```

## Coverage Targets

| Module | Current | Target |
|--------|---------|--------|
| collision | Unknown | 90% |
| cloth | Unknown | 85% |
| character | Unknown | 85% |
| math | Unknown | 95% |

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Test brittleness | Medium | Use epsilon comparisons, not exact equality |
| Slow test suite | Low | Categorize tests (fast/slow), parallel execution |
| False positives | Medium | Tune epsilon tolerances carefully |
| Missing edge cases | High | Property-based testing, community review |
