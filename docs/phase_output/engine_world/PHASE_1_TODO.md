# ENGINE_WORLD - Phase 1 TODO: Foundation Validation

## Task List

### T-W1-001: Constants Audit - Terrain
**Description**: Audit all terrain module files for magic numbers not in `constants.py`
**Files**: `engine/world/terrain/*.py` (9 files, 6,563 lines)
**Acceptance Criteria**:
- [ ] No numeric literals in implementation code except 0, 1, 2 (indices)
- [ ] All constants in `constants.py` have docstrings
- [ ] Constants grouped by functional area (heightfield, lod, materials, sculpting, features)

### T-W1-002: Constants Audit - Environment
**Description**: Audit environment module for magic numbers
**Files**: `engine/world/environment/*.py` (7 files, 5,063 lines)
**Acceptance Criteria**:
- [ ] All Rayleigh/Mie scattering coefficients in constants
- [ ] All weather transition timings in constants
- [ ] All time-of-day keyframe values in constants

### T-W1-003: Constants Audit - Foliage
**Description**: Audit foliage module for magic numbers
**Files**: `engine/world/foliage/*.py` (6 files, 3,341 lines)
**Acceptance Criteria**:
- [ ] LOD distance thresholds in constants
- [ ] Grass chunk sizes in constants
- [ ] Wind animation parameters in constants

### T-W1-004: Constants Audit - HLOD
**Description**: Audit HLOD module for magic numbers
**Files**: `engine/world/hlod/*.py` (5 files, 3,725 lines)
**Acceptance Criteria**:
- [ ] QEM simplification thresholds in constants
- [ ] Impostor atlas dimensions in constants
- [ ] Transition dither pattern in constants (or document why inline)

### T-W1-005: Constants Audit - Partition
**Description**: Audit partition module for magic numbers
**Files**: `engine/world/partition/*.py` (6 files, 2,569 lines)
**Acceptance Criteria**:
- [ ] Cell dimensions in constants
- [ ] Streaming budget defaults in constants
- [ ] Priority calculation factors in constants

### T-W1-006: Constants Audit - PCG
**Description**: Audit PCG module for magic numbers
**Files**: `engine/world/pcg/*.py` (6 files, 4,232 lines)
**Acceptance Criteria**:
- [ ] Noise algorithm parameters (octaves, persistence) in constants
- [ ] Poisson disk sampling parameters in constants
- [ ] PRNG multipliers/increments in constants

### T-W1-007: Constants Audit - Queries
**Description**: Audit queries module for magic numbers
**Files**: `engine/world/queries/*.py` (5 files, 3,530 lines)
**Acceptance Criteria**:
- [ ] Epsilon values in constants
- [ ] Default query limits in constants
- [ ] Binary search iteration counts in constants

---

### T-W1-010: Type Hint Completion - Terrain
**Description**: Add type hints to all functions in terrain module
**Acceptance Criteria**:
- [ ] All function parameters typed
- [ ] All return types specified
- [ ] `mypy --strict` passes with no errors

### T-W1-011: Type Hint Completion - Environment
**Description**: Add type hints to all functions in environment module
**Acceptance Criteria**:
- [ ] Same as T-W1-010

### T-W1-012: Type Hint Completion - Foliage
**Description**: Add type hints to all functions in foliage module
**Acceptance Criteria**:
- [ ] Same as T-W1-010

### T-W1-013: Type Hint Completion - HLOD
**Description**: Add type hints to all functions in HLOD module
**Acceptance Criteria**:
- [ ] Same as T-W1-010

### T-W1-014: Type Hint Completion - Partition
**Description**: Add type hints to all functions in partition module
**Acceptance Criteria**:
- [ ] Same as T-W1-010

### T-W1-015: Type Hint Completion - PCG
**Description**: Add type hints to all functions in PCG module
**Acceptance Criteria**:
- [ ] Same as T-W1-010

### T-W1-016: Type Hint Completion - Queries
**Description**: Add type hints to all functions in queries module
**Acceptance Criteria**:
- [ ] Same as T-W1-010

---

### T-W1-020: Protocol Decoration
**Description**: Add `@runtime_checkable` to all Protocol classes
**Files**: All modules with Protocol definitions
**Acceptance Criteria**:
- [ ] `TerrainInterface` is runtime_checkable
- [ ] `SpatialIndex` is runtime_checkable
- [ ] `NavMesh` is runtime_checkable
- [ ] `TerrainSystem` is runtime_checkable
- [ ] All other Protocol classes decorated

---

### T-W1-030: Limitation Documentation - HLOD
**Description**: Document known limitations in HLOD impostor generation
**Context**: Investigation noted impostor capture uses "simplified CPU rasterization" marked as "demonstration"
**Acceptance Criteria**:
- [ ] HLOD module docstring updated with limitations section
- [ ] GPU capture path documented as future work
- [ ] Technical debt ticket created if appropriate

### T-W1-031: Limitation Documentation - Queries
**Description**: Document known limitations in sweep/capsule queries
**Context**: Investigation noted capsule treated as sphere, sweep may miss thin geometry
**Acceptance Criteria**:
- [ ] Query module docstring updated with limitations section
- [ ] True capsule intersection documented as future work
- [ ] Continuous collision detection documented as future work

### T-W1-032: Limitation Documentation - Navigation
**Description**: Document known limitations in pathfinding
**Context**: Investigation noted simple FIFO cache, no hierarchical pathfinding
**Acceptance Criteria**:
- [ ] Navigation module docstring updated with limitations section
- [ ] LRU cache documented as potential improvement
- [ ] HPA* documented as future work for large maps

---

### T-W1-040: Unit Test Creation - Terrain
**Description**: Create unit tests for terrain module
**File**: `tests/unit/world/test_terrain.py`
**Acceptance Criteria**:
- [ ] Heightfield construction and interpolation tested
- [ ] LOD selection tested at various distances
- [ ] Weight map paint/normalize tested
- [ ] Erosion tool produces valid output
- [ ] Spline evaluation tested

### T-W1-041: Unit Test Creation - Environment
**Description**: Create unit tests for environment module
**File**: `tests/unit/world/test_environment.py`
**Acceptance Criteria**:
- [ ] Weather state machine transitions tested
- [ ] Time-of-day sun position tested against known values
- [ ] Sky color computation tested
- [ ] Volume manager lookup tested

### T-W1-042: Unit Test Creation - Foliage
**Description**: Create unit tests for foliage module
**File**: `tests/unit/world/test_foliage.py`
**Acceptance Criteria**:
- [ ] Grass generation produces instances
- [ ] Frustum culling works correctly
- [ ] LOD distance selection tested
- [ ] Terrain-aware placement filters correctly

### T-W1-043: Unit Test Creation - HLOD
**Description**: Create unit tests for HLOD module
**File**: `tests/unit/world/test_hlod.py`
**Acceptance Criteria**:
- [ ] QEM simplification reduces triangle count
- [ ] Cluster hierarchy builds correctly
- [ ] Screen-space error calculation tested
- [ ] Transition manager updates states

### T-W1-044: Unit Test Creation - Partition
**Description**: Create unit tests for partition module
**File**: `tests/unit/world/test_partition.py`
**Acceptance Criteria**:
- [ ] Cell state machine transitions tested
- [ ] Grid coordinate conversion tested
- [ ] Streaming priority calculation tested
- [ ] Memory budget tracking tested

### T-W1-045: Unit Test Creation - PCG
**Description**: Create unit tests for PCG module
**File**: `tests/unit/world/test_pcg.py`
**Acceptance Criteria**:
- [ ] Noise determinism tested (same seed = same output)
- [ ] Poisson disk spacing verified
- [ ] Placement rules filter correctly
- [ ] Seed hierarchy produces unique values

### T-W1-046: Unit Test Creation - Queries
**Description**: Create unit tests for queries module
**File**: `tests/unit/world/test_queries.py`
**Acceptance Criteria**:
- [ ] Raycast returns correct hit position
- [ ] Terrain raycast binary search converges
- [ ] A* pathfinding finds valid path
- [ ] Path caching returns cached results

---

## Summary

| Category | Tasks | Estimated Effort |
|----------|-------|------------------|
| Constants Audit | 7 | Low (audit + extract) |
| Type Hints | 7 | Medium (add annotations) |
| Protocol Decoration | 1 | Low (add decorators) |
| Limitation Documentation | 3 | Low (update docstrings) |
| Unit Tests | 7 | High (write tests) |
| **Total** | **25** | |
