# PHASE 1 ARCHITECTURE: Test Infrastructure and Verification

---

## Phase Overview

Establish comprehensive test coverage for the existing real implementation before any modifications. Verify mathematical correctness against textbook references. Create baseline performance benchmarks.

---

## Architectural Decisions

### AD-1.1: Test Framework Selection

**Decision**: Use Python unittest with no external dependencies.

**Rationale**: The physics modules have zero external dependencies. Tests should follow the same constraint to remain embeddable in the engine runtime.

**Alternatives Considered**:
- pytest: Better ergonomics, but external dependency
- doctest: Too limited for numerical testing
- hypothesis: Property-based testing, but external dependency

### AD-1.2: Numerical Test Tolerance

**Decision**: Use relative tolerance of 1e-6 for mathematical correctness tests.

**Rationale**: 
- Single-precision float has ~7 significant digits
- 1e-6 catches egregious errors while allowing implementation flexibility
- Matches the EPSILON used in production code

**Formula**: `abs(actual - expected) <= max(1e-6 * abs(expected), 1e-9)`

### AD-1.3: Test Organization

**Decision**: Mirror source structure under `tests/simulation/`.

```
tests/
  simulation/
    physics/
      test_collision_shapes.py
      test_rigid_body.py
      test_physics_world.py
      test_queries.py
      test_sleeping.py
      test_physics_material.py
    solver/
      test_jacobian.py
      test_constraint_solver.py
      test_tgs_solver.py
      test_xpbd_solver.py
      test_island_manager.py
```

**Rationale**: 1:1 mapping makes it clear what tests what. No hunting.

### AD-1.4: Reference Data Source

**Decision**: Hand-verified reference values from textbooks and Wolfram Alpha.

**Sources**:
- Goldstein "Classical Mechanics" for inertia tensors
- Shoemake "Quaternions" SIGGRAPH 1985 for rotation
- Catto GDC slides for SI math

**Rationale**: Independent verification, not testing implementation against itself.

---

## Component Boundaries

### Test Categories

| Category | What It Tests | Pass Criteria |
|----------|---------------|---------------|
| Unit | Single function/method | Output matches reference |
| Integration | Component interaction | No exceptions, plausible output |
| Property | Invariants (energy conservation, etc.) | Invariant holds across random inputs |
| Regression | Known bugs | Bug stays fixed |

### Coverage Targets

| Module | Target | Priority |
|--------|--------|----------|
| `jacobian.py` | 95% | HIGH - mathematical foundation |
| `collision_shapes.py` | 90% | HIGH - inertia tensors critical |
| `rigid_body.py` | 85% | HIGH - state management critical |
| `constraint_solver.py` | 80% | MEDIUM - algorithm correctness |
| `physics_world.py` | 70% | MEDIUM - integration focus |

---

## Interfaces

### Test Base Class

```python
class PhysicsTestCase(unittest.TestCase):
    """Base class for physics tests with numerical helpers."""
    
    def assertAlmostEqualVec3(self, actual, expected, places=6):
        """Assert Vec3 equality within tolerance."""
        
    def assertAlmostEqualMat3(self, actual, expected, places=6):
        """Assert Mat3 equality within tolerance."""
        
    def assertAlmostEqualQuat(self, actual, expected, places=6):
        """Assert quaternion equality (handles sign ambiguity)."""
        
    def assertPositiveDefinite(self, mat3):
        """Assert matrix is positive definite (valid inertia tensor)."""
```

### Benchmark Interface

```python
@dataclass
class BenchmarkResult:
    name: str
    iterations: int
    total_time_s: float
    mean_time_us: float
    std_dev_us: float
    
def benchmark(fn, iterations=1000, warmup=100) -> BenchmarkResult:
    """Run function N times, return statistics."""
```

---

## Data Flow

```
Reference Values (textbooks, Wolfram Alpha)
    |
    v
test_*.py files (hardcoded expected values)
    |
    v
PhysicsTestCase.assert* methods
    |
    v
Pass/Fail report with numerical diff on failure
```

---

## Dependencies

### Internal

- All modules under `engine/simulation/physics/`
- All modules under `engine/simulation/solver/`

### External

- None (unittest is stdlib)

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Tests pass but math is wrong | HIGH | Independent reference values, not circular |
| Flaky tests from floating-point | MEDIUM | Appropriate tolerance, deterministic inputs |
| Tests too slow | LOW | Separate benchmark suite, unit tests stay fast |
