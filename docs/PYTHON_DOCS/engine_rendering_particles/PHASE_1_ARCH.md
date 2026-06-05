# PHASE 1 ARCHITECTURE: CPU Particle System Validation

## Overview
Validate the production-ready CPU particle simulation including all 17 modules, pool management, and physics correctness.

## Component Architecture

### ParticleEmitter
- Full lifecycle management: spawn, update, death
- Prewarm support for pre-simulated initial state
- Budget integration for category-based limits
- Module pipeline: spawn modules -> update modules -> render modules

### ParticlePool
- Ring buffer with free list for O(1) allocation
- Reverse lookup dictionary for O(1) deallocation
- Compaction support for cache-friendly iteration
- Alive/dead state tracking via index sets

### ParticleBudget
- Category-based limits: ambient, gameplay, critical
- Priority-based allocation when budget exceeded
- Per-emitter and global budget tracking

## Module Categories

### Spawn Modules (3)
| Module | Description |
|--------|-------------|
| ShapeEmitter | Point, sphere, box, cone, circle, edge sampling |
| BurstEmitter | Instant count with repeat interval |
| RateEmitter | Particles/second with accumulator |

### Force Modules (7)
| Module | Description |
|--------|-------------|
| GravityModule | Constant acceleration vector |
| WindModule | Directional force + turbulence |
| TurbulenceModule | Position-based pseudo-noise |
| VortexModule | Tangential swirl + radial pull |
| AttractionModule | Point attraction with falloff |
| VectorFieldModule | 3D force volume (data stubbed) |
| CollisionModule | Ground plane + spatial hash neighbor collision |

### Attribute Modules (5)
| Module | Description |
|--------|-------------|
| SizeOverLifeModule | Easing curves (linear, ease_in, ease_out) |
| ColorOverLifeModule | Lerp or gradient sampling |
| RotationModule | Random initial + angular velocity |
| LifetimeModule | Random lifetime range |
| VelocityModule | Initial velocity with spread cone |

### Render Modules (2)
| Module | Description |
|--------|-------------|
| BillboardRenderer | View/velocity alignment, velocity stretch |
| MeshParticleRenderer | Instanced mesh preparation |

## Data Structures

### Particle State
```
position: Vec3
velocity: Vec3
acceleration: Vec3
color: Vec4
size: float
rotation: float
angular_velocity: float
age: float
lifetime: float
state: ParticleState (ALIVE | DEAD)
```

### Spatial Hash Grid
```
cell_size: float
grid: dict[tuple[int,int,int], list[Particle]]
```
Cell coordinates: `(int(x // cell_size), int(y // cell_size), int(z // cell_size))`

## Key Algorithms

### Volume-Corrected Sphere Sampling
```
theta = uniform(0, 2*pi)
phi = acos(uniform(-1, 1))
direction = (sin(phi)*cos(theta), sin(phi)*sin(theta), cos(phi))
r = radius * (uniform(0,1) ^ (1/3))  # Cube root for volume uniformity
position = direction * r
```

### Spatial Hash Neighbor Query
3x3x3 neighborhood search around particle cell:
```
for dx in [-1, 0, 1]:
  for dy in [-1, 0, 1]:
    for dz in [-1, 0, 1]:
      check cell (cx+dx, cy+dy, cz+dz)
```

## Decisions

### ADR-PART-001: Reverse Lookup for O(1) Deallocation
- **Context**: Need fast particle death without linear search
- **Decision**: Maintain `dict[id(particle), index]` mapping
- **Consequence**: 8 bytes overhead per particle, but O(1) deallocation

### ADR-PART-002: Module Pipeline Stages
- **Context**: Modules must execute in correct order
- **Decision**: Explicit SPAWN/UPDATE/RENDER stages with ordered lists
- **Consequence**: Clear execution model, easy to add modules

### ADR-PART-003: Collision via Spatial Hashing
- **Context**: O(n^2) collision checks prohibitive for high counts
- **Decision**: Spatial hash grid with 3x3x3 neighborhood
- **Consequence**: O(n) average case, bounded neighbor count
