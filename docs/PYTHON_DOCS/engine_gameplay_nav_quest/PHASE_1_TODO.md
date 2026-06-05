# PHASE 1 TODO: Navigation Core

**Scope**: NavMesh generation and pathfinding algorithms  
**Files**: `navmesh.py`, `pathfinding.py`

---

## T-NAV-1.1: Verify NavMesh Build Pipeline

**Priority**: P0  
**Estimate**: 2 hours

### Description
Verify the voxelization -> region building -> contour tracing -> polygon generation pipeline produces valid NavMesh output.

### Tasks
- [ ] Review `NavMeshBuilder.voxelize()` for correct span management
- [ ] Review `NavMeshBuilder.build_regions()` flood-fill implementation
- [ ] Review `NavMeshBuilder._graham_scan()` convex hull correctness
- [ ] Verify polygon adjacency graph is correctly computed
- [ ] Check serialization/deserialization round-trip

### Acceptance Criteria
- NavMesh builds from sample geometry without errors
- Generated polygons are convex (all cross products same sign)
- Adjacency graph reflects shared edges
- Serialized NavMesh deserializes to equivalent structure

---

## T-NAV-1.2: Validate A* Implementation

**Priority**: P0  
**Estimate**: 1 hour

### Description
Verify the A* pathfinding algorithm finds optimal paths on NavMesh.

### Tasks
- [ ] Review heap-based open set management
- [ ] Verify heuristic admissibility (never overestimates)
- [ ] Check path reconstruction from came_from map
- [ ] Test on known graphs with known optimal paths

### Acceptance Criteria
- A* finds shortest path on test graphs
- Heuristics do not cause suboptimal paths
- Algorithm terminates on unreachable goals
- Path cost equals sum of edge weights

---

## T-NAV-1.3: Validate JPS Implementation

**Priority**: P1  
**Estimate**: 1.5 hours

### Description
Verify Jump Point Search correctly identifies forced neighbors and jump points.

### Tasks
- [ ] Review `_jump()` recursive implementation
- [ ] Verify forced neighbor detection for cardinal directions
- [ ] Verify forced neighbor detection for diagonal directions
- [ ] Compare JPS paths against A* paths for equivalence

### Acceptance Criteria
- JPS finds paths equivalent to A* on uniform grids
- Jump point identification matches academic definition
- No false negatives (missed jump points)
- Performance improvement over A* measurable

---

## T-NAV-1.4: Validate Theta* Implementation

**Priority**: P1  
**Estimate**: 1 hour

### Description
Verify Theta* produces any-angle paths with line-of-sight optimization.

### Tasks
- [ ] Review line-of-sight check implementation
- [ ] Verify parent pointer updates when LOS exists
- [ ] Compare path smoothness against A* with post-processing
- [ ] Test corner cases (near obstacles, narrow passages)

### Acceptance Criteria
- Theta* paths have fewer waypoints than A* paths
- Line-of-sight checks are geometrically correct
- Paths do not pass through obstacles
- Path length <= A* path length (never longer)

---

## T-NAV-1.5: Validate HPA* Implementation

**Priority**: P2  
**Estimate**: 2 hours

### Description
Verify Hierarchical Path-A* correctly decomposes large maps into clusters.

### Tasks
- [ ] Review cluster generation algorithm
- [ ] Verify inter-cluster edge computation
- [ ] Review abstract path to concrete path refinement
- [ ] Test on maps with 10K+ nodes

### Acceptance Criteria
- Clusters are spatially coherent
- Abstract paths refine to valid concrete paths
- Speedup measurable on large maps vs. flat A*
- Path quality within acceptable bound of optimal

---

## T-NAV-1.6: Verify Path Simplification

**Priority**: P1  
**Estimate**: 1 hour

### Description
Verify Ramer-Douglas-Peucker and funnel algorithm implementations.

### Tasks
- [ ] Review RDP epsilon parameter handling
- [ ] Verify funnel algorithm left/right portal tracking
- [ ] Test simplification preserves path validity
- [ ] Measure simplification ratios on sample paths

### Acceptance Criteria
- Simplified paths do not cross obstacles
- Vertex count reduction is significant (> 50% typical)
- Funnel algorithm produces tight string-pulling paths
- RDP respects epsilon tolerance

---

## T-NAV-1.7: Document Heuristic Selection

**Priority**: P2  
**Estimate**: 0.5 hours

### Description
Document when to use each heuristic function.

### Tasks
- [ ] Create decision matrix for heuristic selection
- [ ] Document performance characteristics
- [ ] Add usage examples to code comments

### Acceptance Criteria
- Clear guidance on heuristic selection
- Performance data supports recommendations
- Code comments updated

---

## Summary

| Task | Priority | Estimate | Status |
|------|----------|----------|--------|
| T-NAV-1.1 | P0 | 2h | Pending |
| T-NAV-1.2 | P0 | 1h | Pending |
| T-NAV-1.3 | P1 | 1.5h | Pending |
| T-NAV-1.4 | P1 | 1h | Pending |
| T-NAV-1.5 | P2 | 2h | Pending |
| T-NAV-1.6 | P1 | 1h | Pending |
| T-NAV-1.7 | P2 | 0.5h | Pending |

**Total Estimate**: 9 hours
