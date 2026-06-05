# PHASE 3 TODO: Test Coverage Expansion

## Objective

Add comprehensive test coverage for edge cases, degenerate inputs, and performance benchmarks across all physics simulation modules.

---

## Task 1: GJK Degenerate Simplex Tests

**File**: `tests/engine/simulation/collision/test_narrowphase_gjk.py`

**Description**: Test GJK algorithm with degenerate simplex configurations.

**Test Cases**:
- [ ] Collinear support points (simplex degenerates to line)
- [ ] Coplanar support points (simplex degenerates to triangle)
- [ ] Overlapping shapes (origin inside Minkowski difference)
- [ ] Near-touching shapes (numerical precision at epsilon distance)
- [ ] Identical shapes (zero-size Minkowski difference)
- [ ] Very thin shapes (near-degenerate geometry)

**Acceptance Criteria**:
- [ ] All tests pass without exceptions
- [ ] GJK returns correct intersection status
- [ ] Distance computation stable for near-touching cases
- [ ] No infinite loops on degenerate input

---

## Task 2: EPA Edge Case Tests

**File**: `tests/engine/simulation/collision/test_narrowphase_epa.py`

**Description**: Test EPA algorithm with edge case polytope configurations.

**Test Cases**:
- [ ] Flat polytope faces (degenerate normal)
- [ ] Very small penetration depth (near-zero)
- [ ] Multiple faces with equal depth (ambiguous contact)
- [ ] Origin on polytope edge (edge-case contact point)
- [ ] Origin on polytope vertex (corner contact)
- [ ] Highly tessellated polytope (many faces)

**Acceptance Criteria**:
- [ ] EPA returns valid penetration depth and normal
- [ ] Contact normal is unit length
- [ ] No division by zero errors
- [ ] Graceful handling of numerical edge cases

---

## Task 3: Cloth Zero-Area Triangle Tests

**File**: `tests/engine/simulation/cloth/test_cloth_degenerate.py`

**Description**: Test cloth simulation with degenerate mesh configurations.

**Test Cases**:
- [ ] Zero-area triangle (collapsed vertices)
- [ ] Inverted triangle (negative area, wind flip)
- [ ] Zero rest length edge (collapsed constraint)
- [ ] All particles pinned (no degrees of freedom)
- [ ] Single free particle (minimal system)
- [ ] Extremely high particle density
- [ ] Extremely low particle density

**Acceptance Criteria**:
- [ ] Simulation does not crash on degenerate input
- [ ] Zero-area triangles handled gracefully (ignored or removed)
- [ ] Wind force handles inverted triangles correctly
- [ ] Pinned-only mesh returns unchanged positions

---

## Task 4: Broadphase Algorithm Benchmark Suite

**File**: `tests/engine/simulation/collision/test_broadphase_benchmark.py`

**Description**: Benchmark all four broadphase algorithms for performance comparison.

**Benchmarks**:
- [ ] Single object insert (SAP, BVH, SpatialHash, Octree)
- [ ] Batch insert (100, 1K, 10K objects)
- [ ] AABB query (sparse vs dense scenes)
- [ ] Ray query (single ray, multiple rays)
- [ ] Pair finding (all overlapping pairs)
- [ ] Object update (position change)
- [ ] Object removal

**Scene Configurations**:
- [ ] Uniform grid (evenly spaced objects)
- [ ] Random scattered (uniform random positions)
- [ ] Clustered groups (Gaussian clusters)
- [ ] Mostly static (5% moving objects)

**Acceptance Criteria**:
- [ ] Results written to `tests/benchmarks/broadphase_results.json`
- [ ] Each algorithm identified with optimal use case
- [ ] Memory usage tracked alongside time
- [ ] Reproducible results (fixed random seed)

---

## Task 5: CCD Edge Case Tests

**File**: `tests/engine/simulation/collision/test_ccd_edge_cases.py`

**Description**: Test continuous collision detection with edge case motions.

**Test Cases**:
- [ ] Zero motion (stationary objects)
- [ ] Parallel motion (objects moving in same direction)
- [ ] Opposite motion (head-on collision)
- [ ] Grazing contact (near-miss)
- [ ] Tunneling scenario (fast object through thin wall)
- [ ] Rotating object collision
- [ ] Object coming to rest exactly on surface

**Acceptance Criteria**:
- [ ] TOI computed accurately within tolerance
- [ ] No tunneling through thin geometry
- [ ] Speculative contacts generated for near-miss
- [ ] Numerical stability at high velocities

---

## Task 6: Character State Transition Tests

**File**: `tests/engine/simulation/character/test_movement_modes.py`

**Description**: Test all movement mode state transitions.

**Test Cases**:
- [ ] All 14 movement modes can be entered
- [ ] All valid transitions (per MovementTransition rules)
- [ ] Invalid transitions rejected
- [ ] Stamina-gated transitions (SPRINTING requires stamina)
- [ ] Mode-specific physics parameters applied
- [ ] CUSTOM mode handling

**Acceptance Criteria**:
- [ ] State machine never enters invalid state
- [ ] Transition requirements enforced
- [ ] Stamina consumption accurate
- [ ] Parameter changes take effect immediately

---

## Task 7: Ragdoll Activation Tests

**File**: `tests/engine/simulation/character/test_ragdoll_activation.py`

**Description**: Test ragdoll activation and deactivation scenarios.

**Test Cases**:
- [ ] Clean activation from animation pose
- [ ] Clean deactivation to animation pose
- [ ] Activation during movement
- [ ] Partial ragdoll (upper body only)
- [ ] Active ragdoll with PD control
- [ ] Balance recovery (STEP, STUMBLE, FALL, BRACE)

**Acceptance Criteria**:
- [ ] No pose discontinuities on activation
- [ ] Motors properly disabled/enabled
- [ ] Joint limits respected
- [ ] PD torques computed correctly

---

## Task 8: Collision Response Tests

**File**: `tests/engine/simulation/character/test_collision_response.py`

**Description**: Test character controller collision response.

**Test Cases**:
- [ ] Ground snap behavior
- [ ] Slope limiting at max angle
- [ ] Step climbing up to max height
- [ ] Ceiling hits (upward collision)
- [ ] Multi-contact resolution (corner)
- [ ] Moving platform attachment
- [ ] Moving platform detachment

**Acceptance Criteria**:
- [ ] Move-and-slide iterations terminate
- [ ] Ground detection accurate
- [ ] Slope limiting prevents climbing too-steep surfaces
- [ ] Platform velocity inherited correctly

---

## Task 9: Contact Manifold Tests

**File**: `tests/engine/simulation/collision/test_contact_manifold.py`

**Description**: Test contact manifold management.

**Test Cases**:
- [ ] Single contact point
- [ ] 4-point manifold (max capacity)
- [ ] Contact point reduction (hull area maximization)
- [ ] Warm starting from previous frame
- [ ] Contact persistence across frames
- [ ] Touch state transitions (BEGIN, PERSIST, END)

**Acceptance Criteria**:
- [ ] Manifold never exceeds 4 points
- [ ] Point reduction preserves convex hull
- [ ] Warm starting reduces impulse oscillation
- [ ] State transitions fire correctly

---

## Task 10: Property-Based Math Tests

**File**: `tests/engine/math/test_vector_properties.py`, `tests/engine/math/test_quaternion_properties.py`

**Description**: Use hypothesis for property-based testing of math primitives.

**Properties to Test**:
- [ ] `dot(a, b) == dot(b, a)` (commutativity)
- [ ] `cross(a, b) == -cross(b, a)` (anti-commutativity)
- [ ] `a.normalized().length() == 1` (unit length)
- [ ] `(q1 * q2) * q3 == q1 * (q2 * q3)` (associativity)
- [ ] `q * q.inverse() == identity` (within epsilon)
- [ ] `slerp(q, q, t) == q` for any t
- [ ] `transform(t.inverse(t), p) == p` (round trip)

**Acceptance Criteria**:
- [ ] hypothesis generates 1000+ test cases per property
- [ ] All properties hold within floating-point epsilon
- [ ] Edge cases (zero vectors, identity quaternions) handled

---

## Dependencies

- Phase 2 math module unification (for clean test structure)
- hypothesis library for property-based tests
- pytest-benchmark for performance tests

## Estimated Effort

| Task | Complexity | Estimate |
|------|------------|----------|
| Task 1: GJK Tests | Medium | 3 hours |
| Task 2: EPA Tests | Medium | 3 hours |
| Task 3: Cloth Degenerate | Medium | 2 hours |
| Task 4: Broadphase Benchmark | High | 6 hours |
| Task 5: CCD Tests | Medium | 3 hours |
| Task 6: State Transitions | Low | 2 hours |
| Task 7: Ragdoll Tests | Medium | 3 hours |
| Task 8: Collision Response | Medium | 3 hours |
| Task 9: Manifold Tests | Medium | 2 hours |
| Task 10: Property Tests | Medium | 3 hours |
| **Total** | | **30 hours** |
