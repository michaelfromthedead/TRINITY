# PHASE 1 TODO: Test Infrastructure and Verification

---

## T-1.1: Create Test Base Class

**File**: `tests/simulation/physics_test_base.py`

**Tasks**:
- [ ] Create `PhysicsTestCase` extending `unittest.TestCase`
- [ ] Implement `assertAlmostEqualVec3(actual, expected, places=6)`
- [ ] Implement `assertAlmostEqualMat3(actual, expected, places=6)`
- [ ] Implement `assertAlmostEqualQuat(actual, expected, places=6)` with sign ambiguity handling
- [ ] Implement `assertPositiveDefinite(mat3)` for inertia tensor validation

**Acceptance Criteria**:
- [ ] All helper methods handle edge cases (zero vector, identity matrix, unit quaternion)
- [ ] Clear failure messages showing expected vs actual with numerical diff
- [ ] No external dependencies

---

## T-1.2: Test Inertia Tensors (collision_shapes.py)

**File**: `tests/simulation/physics/test_collision_shapes.py`

**Tasks**:
- [ ] Test sphere inertia: `I = (2/5) * m * r^2` (diagonal, isotropic)
- [ ] Test box inertia: `Ixx = (1/12) * m * (sy^2 + sz^2)` etc.
- [ ] Test capsule inertia: cylinder + hemispheres + parallel axis theorem
- [ ] Test cylinder inertia: `Ixx = (1/12) * m * (3r^2 + h^2)`, `Izz = (1/2) * m * r^2`
- [ ] Test cone inertia: `Ixx = (3/20) * m * r^2 + (3/80) * m * h^2`
- [ ] Test compound shape inertia: parallel axis theorem for children
- [ ] Verify all tensors are positive definite

**Reference Values**:
- Sphere (m=10, r=2): I_diag = 16.0 each
- Box (m=12, sx=3, sy=4, sz=5): Ixx=41, Iyy=34, Izz=25
- Capsule (m=8, r=1, h=4): verify via numerical integration

**Acceptance Criteria**:
- [ ] All 10 shape types have inertia tests
- [ ] Relative error < 1e-6 for analytical shapes
- [ ] Positive definiteness verified for all

---

## T-1.3: Test Quaternion Operations (jacobian.py)

**File**: `tests/simulation/solver/test_jacobian.py`

**Tasks**:
- [ ] Test quaternion multiplication: `q1 * q2` matches Shoemake formula
- [ ] Test quaternion conjugate: `q.conjugate() = (w, -x, -y, -z)`
- [ ] Test quaternion rotation: `q.rotate(v)` matches `q * v * q^-1`
- [ ] Test quaternion from axis-angle: roundtrip with rotation
- [ ] Test quaternion slerp: midpoint, endpoints, t=0 and t=1
- [ ] Test quaternion normalization: result has unit magnitude

**Reference Values**:
- 90 deg rotation around Z: q=(0.707, 0, 0, 0.707), v=(1,0,0) -> (0,1,0)
- Identity rotation: q=(1,0,0,0), any v -> v

**Acceptance Criteria**:
- [ ] All quaternion operations pass with tolerance 1e-6
- [ ] Edge cases: zero rotation, 180-degree rotation, very small angles

---

## T-1.4: Test Vec3 and Mat3 Operations (jacobian.py)

**File**: `tests/simulation/solver/test_jacobian.py` (extend)

**Tasks**:
- [ ] Test Vec3 dot product
- [ ] Test Vec3 cross product (right-hand rule)
- [ ] Test Vec3 normalization
- [ ] Test Mat3 multiplication
- [ ] Test Mat3 transpose
- [ ] Test Mat3 inverse (and pseudo-inverse for singular)
- [ ] Test Mat3 determinant

**Acceptance Criteria**:
- [ ] Cross product: (1,0,0) x (0,1,0) = (0,0,1)
- [ ] Matrix inverse: A * A^-1 = I within tolerance
- [ ] Determinant matches cofactor expansion

---

## T-1.5: Test Sequential Impulse Solver (constraint_solver.py)

**File**: `tests/simulation/solver/test_constraint_solver.py`

**Tasks**:
- [ ] Test Jacobian computation for contact constraint
- [ ] Test effective mass computation: `K = J * M^-1 * J^T`
- [ ] Test impulse computation: `lambda = -K^-1 * (Jv + bias)`
- [ ] Test warm starting: impulse accumulation and clamping
- [ ] Test single body falling under gravity (no constraints)
- [ ] Test two bodies connected by distance constraint

**Acceptance Criteria**:
- [ ] Single body reaches expected velocity after N steps
- [ ] Distance constraint maintains distance within solver tolerance
- [ ] Energy does not increase (conservation or dissipation only)

---

## T-1.6: Test TGS Solver (tgs_solver.py)

**File**: `tests/simulation/solver/test_tgs_solver.py`

**Tasks**:
- [ ] Test regularization term: `gamma = compliance / dt`
- [ ] Test mass scaling for extreme ratios (1000:1)
- [ ] Test split impulse separating position/velocity correction
- [ ] Test substep integration consistency

**Acceptance Criteria**:
- [ ] Extreme mass ratio does not blow up
- [ ] Split impulse produces less jitter than combined
- [ ] Substep results converge as substeps increase

---

## T-1.7: Test XPBD Solver (xpbd_solver.py)

**File**: `tests/simulation/solver/test_xpbd_solver.py`

**Tasks**:
- [ ] Test compliance parameter effect: higher compliance = softer
- [ ] Test delta lambda formula: `(-C - alpha * lambda) / (w + alpha)`
- [ ] Test distance constraint in XPBD
- [ ] Test collision constraint in XPBD

**Acceptance Criteria**:
- [ ] Zero compliance matches stiff constraint
- [ ] High compliance allows visible stretch
- [ ] Lagrange multipliers accumulate correctly

---

## T-1.8: Test Island Manager (island_manager.py)

**File**: `tests/simulation/solver/test_island_manager.py`

**Tasks**:
- [ ] Test Union-Find: two bodies with contact form one island
- [ ] Test path compression: repeated find returns same root
- [ ] Test rank union: tree stays balanced
- [ ] Test sleep propagation: one body wakes entire island
- [ ] Test island isolation: disconnected bodies form separate islands

**Acceptance Criteria**:
- [ ] N bodies with N-1 contacts form 1 island
- [ ] N bodies with no contacts form N islands
- [ ] Wake signal propagates to all island members

---

## T-1.9: Test Physics World Simulation Step (physics_world.py)

**File**: `tests/simulation/physics/test_physics_world.py`

**Tasks**:
- [ ] Test empty world step (no bodies, no crash)
- [ ] Test single body gravity integration
- [ ] Test two colliding spheres
- [ ] Test collision callback invocation
- [ ] Test step order (callbacks fire after positions finalized)

**Acceptance Criteria**:
- [ ] Body falls 0.5 * g * t^2 in freefall
- [ ] Colliding bodies separate (no interpenetration)
- [ ] Callbacks receive correct collision data

---

## T-1.10: Test Ray Queries (queries.py)

**File**: `tests/simulation/physics/test_queries.py`

**Tasks**:
- [ ] Test ray-sphere intersection (hit center, hit edge, miss)
- [ ] Test ray-box intersection (slab method, all axes)
- [ ] Test ray-capsule intersection
- [ ] Test collision layer filtering (hit/ignore layers)
- [ ] Test closest hit vs all hits mode

**Acceptance Criteria**:
- [ ] Ray origin inside sphere returns immediate hit
- [ ] Ray parallel to box face misses
- [ ] Filtered layers are not hit

---

## T-1.11: Test Sleep System (sleeping.py)

**File**: `tests/simulation/physics/test_sleeping.py`

**Tasks**:
- [ ] Test sleep threshold: body sleeps after T seconds of low velocity
- [ ] Test wake on contact
- [ ] Test island wake propagation
- [ ] Test manual sleep/wake API

**Acceptance Criteria**:
- [ ] Body with |v| < threshold for T seconds sleeps
- [ ] Any contact immediately wakes
- [ ] Sleeping body does not integrate

---

## T-1.12: Establish Baseline Benchmarks

**File**: `tests/simulation/benchmarks.py`

**Tasks**:
- [ ] Benchmark broadphase for N = 100, 500, 1000 bodies
- [ ] Benchmark narrowphase contact generation
- [ ] Benchmark SI solver for 10 iterations
- [ ] Benchmark TGS solver for 10 iterations
- [ ] Benchmark XPBD solver for 10 iterations
- [ ] Benchmark full physics step

**Output Format**:
```
benchmark_broadphase_100:  mean=0.5ms, std=0.02ms
benchmark_broadphase_500:  mean=8.3ms, std=0.15ms
benchmark_broadphase_1000: mean=32.1ms, std=0.42ms
```

**Acceptance Criteria**:
- [ ] Benchmarks are reproducible (std < 10% of mean)
- [ ] Results saved to `tests/simulation/baseline_benchmarks.json`
- [ ] O(n^2) broadphase scaling confirmed
