# PHASE 1 ARCHITECTURE: Navigation Core

**Scope**: NavMesh generation and pathfinding algorithms  
**Files**: `navmesh.py`, `pathfinding.py`  
**Lines**: ~2,556

---

## Architecture Overview

Phase 1 covers the foundational navigation infrastructure: converting geometry to walkable surfaces and finding paths across them.

```
                    +------------------+
                    |   Raw Geometry   |
                    +--------+---------+
                             |
                    +--------v---------+
                    |   Voxelization   |
                    |  (Span Management)|
                    +--------+---------+
                             |
                    +--------v---------+
                    |  Region Building |
                    |   (Flood-Fill)   |
                    +--------+---------+
                             |
                    +--------v---------+
                    | Contour Tracing  |
                    +--------+---------+
                             |
                    +--------v---------+
                    |  Polygon Gen     |
                    |  (Graham Scan)   |
                    +--------+---------+
                             |
                    +--------v---------+
                    |     NavMesh      |
                    +--------+---------+
                             |
         +-------------------+-------------------+
         |                   |                   |
   +-----v-----+       +-----v-----+       +-----v-----+
   |    A*     |       |    JPS    |       |  Theta*   |
   +-----------+       +-----------+       +-----------+
```

---

## Component Architecture

### NavMesh Builder (`navmesh.py`)

```
NavMeshBuilder
├── voxelize(geometry) -> VoxelField
│   ├── create_voxel_grid()
│   └── insert_spans()
├── build_regions(voxels) -> RegionSet
│   └── flood_fill()
├── trace_contours(regions) -> ContourSet
│   └── walk_boundary()
└── generate_polygons(contours) -> NavMesh
    └── _graham_scan()  # Convex hull per contour
```

**Key Data Structures**:
- `VoxelField`: 3D grid of walkable spans
- `RegionSet`: Connected components of walkable areas
- `ContourSet`: Boundary edges of regions
- `NavMesh`: Final polygon mesh with adjacency

### Pathfinding Engine (`pathfinding.py`)

```
PathfindingEngine
├── find_path(start, goal, algorithm) -> Path
├── _astar(start, goal) -> Path
│   └── heap-based open set
├── _jps(start, goal) -> Path
│   └── _jump()  # Recursive jump point detection
├── _theta_star(start, goal) -> Path
│   └── line_of_sight()  # Any-angle optimization
└── _hpa_star(start, goal) -> Path
    └── cluster decomposition
```

**Algorithm Selection**:
| Input | Algorithm |
|-------|-----------|
| NavMesh | A* (weighted graph) |
| Uniform grid | JPS (optimized A*) |
| Open terrain | Theta* (any-angle) |
| Large map | HPA* (hierarchical) |

---

## Data Flow

### NavMesh Generation Flow

```
Input: Triangle soup (vertices, indices)
  |
  v
Voxelize: Project triangles onto voxel grid
  |
  v
Span Management: Track walkable height ranges per column
  |
  v
Flood Fill: Identify connected walkable regions
  |
  v
Contour Tracing: Extract boundary edges
  |
  v
Polygon Generation: Graham scan for convex hull
  |
  v
Output: NavMesh (polygons + adjacency graph)
```

### Pathfinding Flow

```
Input: Start point, Goal point, NavMesh
  |
  v
Point Location: Find containing polygon for start/goal
  |
  v
Graph Search: A*/JPS/Theta*/HPA* on polygon graph
  |
  v
Path Extraction: Reconstruct node sequence
  |
  v
Path Smoothing: Funnel algorithm or Ramer-Douglas-Peucker
  |
  v
Output: Waypoint sequence
```

---

## Architectural Decisions

### ADR-NAV-001: Voxel-Based NavMesh Generation

**Context**: NavMesh must handle arbitrary geometry (overhangs, multi-level, slopes).

**Decision**: Use voxelization pipeline (Recast-style) rather than polygon projection.

**Rationale**:
- Handles complex geometry uniformly
- Proven approach (Recast/Detour, Unity, Unreal)
- Span management handles vertical complexity

**Consequences**:
- Higher memory during build (voxel grid)
- Build time proportional to volume, not surface area
- Resolution affects quality vs. performance

### ADR-NAV-002: Graham Scan for Convex Polygons

**Context**: NavMesh polygons must be convex for funnel algorithm.

**Decision**: Use Graham scan to generate convex hulls from contours.

**Rationale**:
- O(n log n) complexity
- Simple, well-understood algorithm
- Guarantees convexity

**Consequences**:
- May lose concave detail (acceptable for navigation)
- All polygons guaranteed convex

### ADR-NAV-003: Multiple Pathfinding Algorithms

**Context**: Different scenarios benefit from different algorithms.

**Decision**: Support A*, JPS, Theta*, HPA* with runtime selection.

**Rationale**:
- A*: General-purpose, any graph
- JPS: 10-100x faster on uniform grids
- Theta*: Smoother paths without post-processing
- HPA*: Scales to very large maps

**Consequences**:
- More code to maintain
- Must select appropriate algorithm per use case
- Shared data structures (PathNode, Path)

### ADR-NAV-004: Heap-Based Open Set

**Context**: A* open set must support efficient min-extraction and membership test.

**Decision**: Use Python heapq with lazy deletion.

**Rationale**:
- O(log n) insert and extract-min
- Built-in, no dependencies
- Lazy deletion handles decrease-key

**Consequences**:
- Extra memory for stale entries
- Must check validity on extraction

---

## Heuristics

### Pathfinding Heuristics

| Heuristic | Formula | Use Case |
|-----------|---------|----------|
| Manhattan | \|dx\| + \|dz\| | 4-directional grid |
| Euclidean | sqrt(dx^2 + dz^2) | Open terrain |
| Octile | max(\|dx\|, \|dz\|) + 0.41 * min(\|dx\|, \|dz\|) | 8-directional grid |
| Chebyshev | max(\|dx\|, \|dz\|) | Diagonal cost = straight cost |
| Zero | 0 | Dijkstra (no heuristic) |

---

## Performance Considerations

### NavMesh Build

- **Voxel resolution**: Higher = better quality, slower build
- **Region minimum size**: Filter small regions to reduce polygon count
- **Contour simplification**: Reduce vertex count on boundaries

### Pathfinding

- **Precompute polygon adjacency**: O(1) neighbor lookup
- **Spatial indexing**: Accelerate point location
- **Path caching**: Reuse paths for common routes
- **Hierarchical search**: HPA* for maps > 10K polygons
