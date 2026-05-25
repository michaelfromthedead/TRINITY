# PHASE 1 TODO: CPU Particle System Validation

## Tasks

### 1.1 Validate ParticlePool O(1) Operations
**File**: `particle_system.py` (lines 270-316)
**Acceptance**: 
- Allocation returns particle in O(1) when pool has capacity
- Deallocation returns particle to free list in O(1)
- Reverse lookup dictionary correctly maps particle id to index
- Pool compaction maintains valid index mappings

### 1.2 Validate Sphere Sampling Volume Distribution
**File**: `particle_modules.py` (lines 238-255)
**Acceptance**:
- Cube-root correction `(random ** (1/3))` applied for volume sampling
- Surface emission uses radius directly without correction
- Direction vector normalized correctly
- Theta/phi angles produce uniform spherical distribution

### 1.3 Validate Spatial Hash Collision Detection
**File**: `particle_modules.py` (lines 666-697)
**Acceptance**:
- Cell coordinates computed correctly via floor division
- 3x3x3 neighborhood query returns all adjacent cells
- Particles correctly inserted/removed from grid on position change
- Ground plane collision applies bounce coefficient and friction

### 1.4 Validate GravityModule
**File**: `particle_modules.py` (lines 423-427)
**Acceptance**:
- Acceleration accumulated (not replaced)
- Gravity vector applied correctly each frame
- Works with other force modules simultaneously

### 1.5 Validate WindModule
**File**: `particle_modules.py`
**Acceptance**:
- Direction + magnitude applied as force
- Turbulence adds position-based variation
- Combined with other forces correctly

### 1.6 Validate TurbulenceModule
**File**: `particle_modules.py`
**Acceptance**:
- Pseudo-noise computed from position
- Force varies spatially but deterministically
- Strength parameter scales effect correctly

### 1.7 Validate VortexModule
**File**: `particle_modules.py`
**Acceptance**:
- Tangential force creates swirl effect
- Radial force pulls toward/away from axis
- Combined effect produces spiral motion

### 1.8 Validate AttractionModule
**File**: `particle_modules.py`
**Acceptance**:
- Point attractor with configurable falloff
- Force magnitude decreases with distance
- Attraction center configurable

### 1.9 Validate SizeOverLifeModule
**File**: `particle_modules.py`
**Acceptance**:
- Linear easing interpolates correctly
- Ease-in starts slow, ends fast
- Ease-out starts fast, ends slow
- Size at age=0 equals start size, age=lifetime equals end size

### 1.10 Validate ColorOverLifeModule
**File**: `particle_modules.py`
**Acceptance**:
- Lerp mode interpolates between two colors
- Gradient mode samples from color stops
- Alpha channel interpolated correctly

### 1.11 Validate ShapeEmitter Sampling
**File**: `particle_modules.py`
**Acceptance**:
- Point: all particles at origin
- Sphere: volume-corrected distribution
- Box: uniform within bounds
- Cone: direction within cone angle
- Circle: 2D disk sampling
- Edge: linear along line segment

### 1.12 Validate BurstEmitter Timing
**File**: `particle_modules.py`
**Acceptance**:
- Spawns exact count on trigger
- Repeat interval respected if configured
- Works with emitter prewarm

### 1.13 Validate RateEmitter Accumulation
**File**: `particle_modules.py`
**Acceptance**:
- Accumulator tracks fractional particles
- Spawns when accumulator >= 1.0
- Rate per second correctly converted to per-frame

### 1.14 Validate ParticleBudget Categories
**File**: `particle_system.py`
**Acceptance**:
- Ambient particles lowest priority
- Gameplay particles medium priority
- Critical particles always allocated if possible
- Global budget respected across emitters

### 1.15 Validate Emitter Prewarm
**File**: `particle_system.py`
**Acceptance**:
- Prewarm simulates N seconds before first frame
- Particles in correct lifecycle state after prewarm
- Pool state consistent after prewarm

### 1.16 Validate VectorFieldModule (Stub)
**File**: `particle_modules.py`
**Acceptance**:
- Architecture in place for 3D force volume
- Data loading correctly stubbed (returns zero force)
- Does not crash when field data unavailable

### 1.17 Validate BillboardRenderer
**File**: `particle_modules.py`
**Acceptance**:
- View alignment computes correct facing direction
- Velocity alignment orients along movement
- Velocity stretch scales billboard correctly

### 1.18 Validate MeshParticleRenderer
**File**: `particle_modules.py`
**Acceptance**:
- Instance data prepared correctly for batch rendering
- Transform matrices computed from particle state
- Culling data generated for frustum tests
