# PHASE 1 ARCHITECTURE: Destruction System

**Scope**: ~4,869 lines across 6 files  
**Classification**: REAL (Production-Ready)

---

## System Overview

The destruction system implements mesh fracturing and debris management for real-time destructible environments. It separates concerns into three domains:

1. **Fracture Generation** - Geometry modification algorithms
2. **Support Analysis** - Structural integrity graph computation
3. **Debris Management** - Post-fracture physics object lifecycle

---

## Component Architecture

### 1. Fracture Algorithms

```
destruction_system.py (Coordinator)
       |
       +---> fracture_voronoi.py  (Voronoi cell-based)
       |
       +---> fracture_radial.py   (Impact-centered)
       |
       +---> fracture_slice.py    (Planar cutting)
```

**Algorithm Selection**: The destruction_system.py coordinator selects fracture algorithm based on:
- Material type (stone/concrete -> Voronoi, glass -> Radial, wood -> Slice)
- Impact parameters (energy, direction, contact area)
- Performance budget

### 2. Support Graph

```
support_graph.py
       |
       +---> Dijkstra stress propagation
       |
       +---> Connected component detection
       |
       +---> Anchor management
```

**Purpose**: Determines which fragments remain attached vs fall after fracture.

### 3. Debris Management

```
debris.py
       |
       +---> Object pool (pre-allocated fragments)
       |
       +---> LOD system (FULL/REDUCED/SIMPLE/PARTICLE)
       |
       +---> Merge system (consolidate small nearby debris)
       |
       +---> Sleep detection (freeze stationary debris)
```

---

## Key Algorithms

### Voronoi Fracture (fracture_voronoi.py)

**Algorithm Flow**:
1. Generate Voronoi sites (random or impact-centered)
2. Compute Voronoi cell bisector planes
3. For each mesh triangle:
   - Clip to each cell using half-space intersection
   - Apply Sutherland-Hodgman 3D polygon clipping
   - Filter degenerate triangles (area < threshold)
4. Generate output meshes per cell

**Sutherland-Hodgman 3D Extension**:
```
For each plane:
    For each edge (curr, next):
        if curr inside AND next inside:
            output next
        elif curr inside AND next outside:
            output intersection
        elif curr outside AND next inside:
            output intersection, next
        // else: both outside, output nothing
```

**Numerical Guards**:
- Division by zero check when edge lies on plane
- Clamped interpolation parameter t to [0, 1]
- Degenerate triangle filtering with configurable area threshold

### Radial Fracture (fracture_radial.py)

**Algorithm Flow**:
1. Compute impact center and direction
2. Generate radial slices (configurable count, typically 8-16)
3. Generate concentric rings (quadratic spacing for realism)
4. Apply slices to create pie-shaped fragments
5. Optionally add spider-web pattern (secondary radial cracks)

**Quadratic Ring Spacing**:
```
ring_radius[i] = impact_radius * (i / num_rings)^2
```
This creates smaller fragments near impact center (higher energy density).

### Slice Fracture (fracture_slice.py)

**Algorithm Flow**:
1. Define cutting plane(s)
2. Classify vertices (above/below plane)
3. Compute edge-plane intersections
4. Generate cap surface using ear clipping triangulation
5. Output two mesh fragments per slice

**Cap Generation**:
- Collect intersection points forming polygon boundary
- Triangulate using ear clipping (O(n^2) but robust)
- Assign proper winding order for correct normals

### Support Graph Analysis (support_graph.py)

**Purpose**: Determine structural connectivity after fracture.

**Algorithm**:
1. Build graph: nodes = fragments, edges = contact points
2. Mark anchor nodes (floor/wall connections)
3. Run Dijkstra from all anchors simultaneously
4. Compute stress along paths (configurable decay function)
5. When stress exceeds threshold, break edge
6. Find connected components to identify falling groups

**Stress Decay Options**:
- Linear: stress = stress_0 - decay_rate * distance
- Exponential: stress = stress_0 * exp(-decay_rate * distance)
- Custom per-material curves

### Debris Management (debris.py)

**Object Pool**:
- Pre-allocate N debris slots at initialization
- Track free list (indices of available slots)
- On fracture: pop from free list, configure fragment
- On cleanup: return to free list (no deallocation)

**LOD Levels**:
| Level | Description | Physics | Rendering |
|-------|-------------|---------|-----------|
| FULL | Full simulation | All collisions | Full mesh |
| REDUCED | Simplified | Major collisions only | LOD mesh |
| SIMPLE | Minimal | Gravity only | Billboard |
| PARTICLE | Substitute | None | Particle effect |

**LOD Selection**:
```
distance = camera_distance(debris)
if distance < threshold_full:
    level = FULL
elif distance < threshold_reduced:
    level = REDUCED
elif distance < threshold_simple:
    level = SIMPLE
else:
    level = PARTICLE
```

**Debris Merging**:
When many small debris accumulate:
1. Spatial hash to find nearby debris
2. If total mass < merge_threshold AND count > merge_count:
   - Remove individual debris
   - Spawn single combined debris at centroid

**Sleep Detection**:
Debris transitions to sleep state when:
- Linear velocity < sleep_linear_threshold for N frames
- Angular velocity < sleep_angular_threshold for N frames
- Not in contact with awake objects

---

## Data Flow

```
Impact Event
    |
    v
destruction_system.py
    |
    +---> Select fracture algorithm (material, energy)
    |
    v
fracture_*.py
    |
    +---> Generate fragment meshes
    |
    v
support_graph.py
    |
    +---> Compute connectivity, stress paths
    |
    +---> Identify falling groups
    |
    v
debris.py
    |
    +---> Acquire from pool
    |
    +---> Configure physics/rendering
    |
    +---> LOD assignment
    |
    v
Frame Update Loop
    |
    +---> Update physics (per LOD)
    |
    +---> Sleep detection
    |
    +---> Merge check
    |
    +---> Return to pool when done
```

---

## Configuration Points

All tunable parameters externalized to `config.py`:

| Parameter | Domain | Purpose |
|-----------|--------|---------|
| `degenerate_area_threshold` | Voronoi | Filter tiny triangles |
| `voronoi_site_count` | Voronoi | Fragment count control |
| `radial_slice_count` | Radial | Wedge count |
| `radial_ring_count` | Radial | Concentric ring count |
| `stress_decay_rate` | Support | Collapse propagation speed |
| `debris_pool_size` | Debris | Pre-allocation count |
| `lod_thresholds` | Debris | Distance breakpoints |
| `merge_threshold_mass` | Debris | Consolidation trigger |
| `sleep_velocity_thresholds` | Debris | Freeze detection |

---

## Integration Points

### Input Interfaces
- `PhysicsBodyProtocol`: Position, velocity, apply_force
- `MeshProtocol`: Vertices, triangles, UVs

### Output Interfaces
- `DebrisCallback`: On debris created/destroyed/merged
- `FractureCallback`: On fracture completed

### External Dependencies
- numpy: Vector math
- Spatial hash: Neighbor queries for merging
