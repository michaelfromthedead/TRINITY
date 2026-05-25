# Phase 5: Navigation System — Architecture

## Overview

Complete navigation pipeline: NavMesh generation (voxelization → mesh), 5 pathfinding algorithms, 12+ steering behaviors, 3 local avoidance modes, navigation links, and smart objects.

## Component Breakdown

### NavMesh Generation (`nav/nav_mesh.py`)

```
Recast-Compatible Pipeline
├── 1. Voxelization (walkable surface rasterization)
│   └── Agent radius, height, step_height, max_slope
├── 2. Region Building (contiguous walkable regions)
│   └── Watershed/monotone partitioning
└── 3. Contour → Mesh (polyline → triangulation)
    └── Edge classification: walkable/obstacle

3 Build Modes
├── Static (pre-built, compile-time, immutable)
├── Dynamic (runtime obstacle carving, polygon re-connection)
└── Hybrid (tiled with streaming)

Tiled NavMesh
├── Dirty tracking
├── Proximity-based tile loading/unloading
└── Seamless transitions
```

### Pathfinding (`nav/pathfinding.py`)

```
5 Algorithms
├── A* (NavMesh polygon graph, Euclidean heuristic)
├── JPS (grid, forced neighbors, straight/diagonal pruning)
├── Theta* (grid, line-of-sight grandparent check, any-angle)
├── HPA* (hierarchical: clusters → abstract → intra-cluster)
└── Dijkstra (uniform cost)

5 Heuristics: Euclidean, Manhattan, Chebyshev, Octile, Diagonal

PathResult / PathRequest types
├── Path smoothing: Funnel algorithm, Ramer-Douglas-Peucker simplification, Chaikin
├── Path corridor (left/right edges for formation)
└── Partial paths (when full path not found)
```

### Steering (`nav/steering.py`)

```
SteeringAgent
├── position, velocity, heading, side
├── mass, max_speed, max_force
├── local_to_world / world_to_local transforms
└── WanderState (persistent per agent)

12+ Behaviors
├── Seek (force toward target)
├── Flee (force away from target)
├── Arrive (slow/stop radii, deceleration)
├── Pursue (predicted intercept point)
├── Evade (flee from predicted position)
├── Wander (random jitter with direction bias)
├── Separation (inverse distance weighting)
├── Alignment (match neighbor velocity)
├── Cohesion (move toward group center)
├── Flocking (weighted sum of sep + align + cohesion)
├── Obstacle Avoidance (detection box, lateral + braking force)
└── Wall Following (3 feelers, line intersection)
└── Path Following (prediction + waypoint tracking)

SteeringManager
├── Weighted sum calculation
└── Priority-based (dithering)
```

### Local Avoidance (`nav/avoidance.py`)

```
3 Modes + Unified Interface
├── RVO (Reciprocal Velocity Obstacles)
│   ├── VelocityObstacle cone computation
│   ├── Velocity sampling grid (RVO_VELOCITY_SAMPLES)
│   └── Collision detection with leg calculations
├── ORCA (Optimal Reciprocal Collision Avoidance)
│   ├── Half-plane linear programming
│   ├── Truncated VO with cut-off circle
│   ├── ORCA constraint computation with priority
│   └── Iterative constraint projection
└── ForceBasedAvoidance
    ├── Quadratic falloff repulsion
    └── Combined agent + obstacle forces

AvoidanceSystem
├── Unified interface: NONE, RVO, ORCA, FORCE_BASED
├── AvoidanceAgent, AvoidanceObstacle
└── AvoidanceResult
```

### Navigation Links (`nav/nav_links.py`)

```
NavLink Types
├── JUMP (parabolic arc interpolation)
├── DROP (accelerating fall)
├── CLIMB (linear interpolation)
├── TELEPORT (instant)
└── CUSTOM (user-defined)

Specialized Links
├── DoorLink (open/close/lock/unlock, auto-close timer, animation)
├── LadderLink (rung positions, climb speed, dismount height)

NavLinkManager
├── Spatial indexing
├── Type-specific factory methods
├── begin_traversal / update_traversal / cancel_traversal
├── find_links_at_position, find_links_by_type
└── Link validation, cost adjustment
```

### Smart Objects (`nav/smart_objects.py`)

```
12 Categories
├── GENERIC, COVER, DOOR, BUTTON, TERMINAL
├── SEAT, BED, WORKSTATION, VEHICLE
├── CONTAINER, CONVERSATION

SmartObject
├── Slots: AVAILABLE / RESERVED / OCCUPIED / DISABLED
├── Interaction parameters (range, duration, animation)
├── World position calculation
└── CoverPoint (tactical: cover type, stance, exposed directions, is_safe_from)

SmartObjectManager
├── 3D cell grid spatial indexing
├── Category indexing
├── Radius queries, nearest/filtered search
├── Reservation/queue system with timeout
└── find_cover_from_threat
```

## Data Flow

```
Path Request
  └─→ PathfindingAlgorithm.find_path(start, end)
       └─→ PathResult
            ├─→ PathSmoothing.funnel() → smooth path
            └─→ PathCorridor → corridor edges
                 └─→ SteeringAgent
                      ├─→ SteeringManager (weighted sum)
                      │    └─→ Desired velocity
                      ├─→ AvoidanceSystem (RVO/ORCA/Force)
                      │    └─→ Corrected velocity
                      └─→ MovementComponent → position update
```

## Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `nav/nav_mesh.py` | — | NavMesh pipeline, 3 build modes, tiled |
| `nav/pathfinding.py` | — | A*, JPS, Theta*, HPA*, Dijkstra, smoothing |
| `nav/steering.py` | 944 | 12 steering behaviors, SteeringAgent |
| `nav/avoidance.py` | 1009 | RVO, ORCA, ForceBased, AvoidanceSystem |
| `nav/nav_links.py` | 825 | NavLink types, DoorLink, LadderLink |
| `nav/smart_objects.py` | 822 | Smart objects, cover points, reservations |

## Dependencies

- Foundation math: Vec3, AABB, Ray
- Phase 1 entity framework (movement component)
- Phase 3 AI (AI movement integration)
