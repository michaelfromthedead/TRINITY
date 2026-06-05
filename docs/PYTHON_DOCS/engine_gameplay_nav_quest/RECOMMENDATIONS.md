# RECOMMENDATIONS.md — engine/gameplay/nav + engine/gameplay/quest

## Rust Bridge Requirements

### High Priority

| Component | Rationale | Estimated Gain |
|-----------|-----------|----------------|
| `Vector3` operations | Used in every nav calculation; dot, cross, normalize called millions of times per frame | 10-50x speedup |
| ORCA `_solve_linear_program` | Per-agent per-frame; hot loop for multi-agent games | 5-20x speedup |
| A* open set operations | Heap push/pop dominates pathfinding cost | 3-10x speedup |
| NavMesh polygon containment | `contains_point_2d` called for every query | 5-15x speedup |

### Medium Priority

| Component | Rationale | Estimated Gain |
|-----------|-----------|----------------|
| HPA* cluster building | Preprocessing step; amortizes cost over many queries | 10x preprocessing |
| NavMesh generation pipeline | Voxelization, contour tracing are CPU-intensive | 20x build time |
| Steering force combination | Weighted sum of multiple behaviors per frame | 2-5x speedup |
| RVO velocity sampling | N^2 agent comparisons | 5-10x speedup |

### Low Priority

| Component | Rationale | Estimated Gain |
|-----------|-----------|----------------|
| Dialogue condition evaluation | Runs once per dialogue advance; not a hot path | Minimal |
| Effect execution | Runs once per dialogue node; not performance-critical | Minimal |
| Objective state transitions | Event-driven, infrequent | Minimal |
| Quest flow traversal | Player-paced; no performance concern | Minimal |

---

## Integration Strategy

### Phase 1: Vector3 Bridge (Foundation)
1. Create Rust `Vector3` struct with SIMD-optimized operations
2. Expose via PyO3 to Python
3. Replace Python `Vector3` dataclass with Rust wrapper
4. Benchmark: Target 10x improvement in pathfinding micro-benchmarks

### Phase 2: Pathfinding Core
1. Move `PathNode` heap operations to Rust
2. Implement A* loop in Rust, call Python for heuristics (hybrid)
3. Full Rust A* with Python callback for custom costs
4. Benchmark: Target 1000+ path queries per frame

### Phase 3: Collision Avoidance
1. Rust implementation of `HalfPlane.contains`
2. Rust linear program solver (2D simplex or GJK)
3. Full ORCA loop in Rust
4. Benchmark: Target 100+ agents at 60 FPS

### Phase 4: NavMesh Generation (Offline Tool)
1. Rust voxelization with rayon parallelization
2. Rust region building and contour tracing
3. Python orchestration, Rust computation
4. Benchmark: Target 1M triangle meshes in seconds

---

## Testing Strategy

### Unit Tests (Whitebox)

| Test | File | Purpose |
|------|------|---------|
| `test_vector3_operations` | navmesh.py | Verify Vector3 math correctness |
| `test_a_star_basic` | pathfinding.py | Simple path on small graph |
| `test_heuristics` | pathfinding.py | Each heuristic returns sensible values |
| `test_orca_constraint` | avoidance.py | Half-plane computation |
| `test_condition_operators` | dialogue_conditions.py | AND/OR/NOT/XOR behavior |
| `test_effect_rollback` | dialogue_effects.py | Partial failure rolls back |

### Integration Tests (Blackbox)

| Test | Purpose |
|------|---------|
| JPS vs A* Path Equivalence | JPS finds same-cost paths as A* |
| Theta* Smoothness | Theta* paths have fewer waypoints |
| ORCA No Collision | 50 agents navigate without overlap |
| Dialogue Full Conversation | Complete dialogue tree traversal |
| Quest Sequential Flow | Complete objectives in order |
| Quest Parallel Flow | Complete objectives in any order |

### Stress Tests

| Test | Parameters |
|------|------------|
| Large NavMesh Query | 10K polygons, 1K queries/frame |
| Multi-Agent ORCA | 200 agents in confined space |
| Deep Dialogue Graph | 100+ nodes, 20+ branches |
| Complex Quest Flow | 50 objectives, nested flows |

### Fuzz Tests

| Test | Target |
|------|--------|
| Condition Combinations | Random AND/OR/NOT trees |
| PathRequest Parameters | Edge case heuristic weights |
| NavLink Traversal | Random agent flags vs requirements |

---

## Risk Assessment

### High Risk

| Risk | Mitigation |
|------|------------|
| Vector3 bridge breaks existing code | Ensure 100% API compatibility via Python wrapper |
| ORCA linear solver edge cases | Port test suite with Rust implementation |
| NavMesh generation regression | Validate against known-good meshes |

### Medium Risk

| Risk | Mitigation |
|------|------------|
| HPA* cluster quality | Compare paths to non-hierarchical A* |
| Dialogue effect ordering | Document and test effect priorities |
| Objective callback timing | Clear documentation on event-driven model |

### Low Risk

| Risk | Mitigation |
|------|------------|
| Steering behavior tuning | Expose all parameters; defaults from constants.py |
| Quest flow complexity | FlowBuilder provides safe construction |
| Localization fallback | DefaultLocalizationProvider returns keys gracefully |

---

## Action Items

### Immediate (This Sprint)
1. Create benchmarking harness for pathfinding
2. Profile ORCA under 50+ agent load
3. Add blackbox tests for JPS correctness

### Short-Term (Next 2 Sprints)
1. Prototype Vector3 Rust bridge
2. Add integration tests for dialogue system
3. Document condition DSL usage

### Medium-Term (Next Quarter)
1. Full A* Rust implementation
2. ORCA Rust implementation
3. NavMesh debug visualization tool

### Long-Term (Roadmap)
1. Dynamic NavMesh obstacle carving
2. Navigation volume for 3D movement
3. Procedural quest generation framework

---

## Dependencies

### External Crates (for Rust Bridge)
- `pyo3` — Python bindings
- `rayon` — Parallel iteration
- `nalgebra` or `glam` — Vector math (or custom SIMD)
- `ahash` — Fast hashing for PathNode

### Python Dependencies
- None beyond standard library (dataclasses, typing, abc, math, heapq)

### Internal Dependencies
- `engine/gameplay/nav/constants.py` — Must be kept in sync with Rust constants
- `engine/gameplay/quest/constants.py` — Enum values must match
