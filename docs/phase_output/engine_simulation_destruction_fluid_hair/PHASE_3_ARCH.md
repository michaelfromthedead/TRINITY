# PHASE 3 ARCHITECTURE: Hair System

**Scope**: ~2,600 lines across 4 files  
**Classification**: REAL (Production-Ready)

---

## System Overview

The hair system implements strand-based hair simulation using Position-Based Dynamics (PBD). It provides physically plausible hair behavior for real-time applications through constraint-based solving rather than force-based integration.

---

## Component Architecture

```
hair/
  |
  +-- hair_simulation.py (Main simulation loop)
  |     |
  |     +-- Follow-The-Leader (FTL) constraint solving
  |     +-- Verlet integration
  |     +-- Inertia transfer from head motion
  |     +-- Guide hair management
  |
  +-- hair_collision.py (Collision detection/response)
  |     |
  |     +-- Point-capsule collision
  |     +-- Point-sphere collision
  |     +-- SDF collision
  |     +-- Density-field self-collision
  |
  +-- hair_constraints.py (Constraint types)
  |     |
  |     +-- Length constraint (inextensibility)
  |     +-- Global shape matching
  |     +-- Local shape constraint (angle preservation)
  |     +-- Rodrigues rotation formula
  |
  +-- hair_lod.py (Level-of-detail)
        |
        +-- Distance-based LOD selection
        +-- Guide hair selection
        +-- Interpolation for rendering
        +-- Shell rendering preparation
```

---

## Core Algorithms

### Follow-The-Leader (FTL) Constraint Solving

**Purpose**: Efficiently enforce length constraints along hair strands.

**Algorithm**:
```
For each strand:
    # Fix root to head
    positions[0] = head_attachment_point
    
    # Propagate constraints root-to-tip
    For i in 1..N:
        direction = normalize(positions[i] - positions[i-1])
        positions[i] = positions[i-1] + direction * rest_length
```

**Why FTL over Global Solve**:
- O(n) per strand vs O(n^3) for global Gauss-Seidel
- Parallelizable across strands
- Guaranteed length preservation in single pass
- Correct energy propagation (root dominates tip)

### Verlet Integration

**Purpose**: Time integration with implicit velocity.

```
# Store old position for velocity extraction
old_position = position

# Semi-implicit position update
position = position + velocity * dt + acceleration * dt^2

# Solve constraints (FTL, collision, shape)
solve_constraints()

# Extract velocity from position change
velocity = (position - old_position) / dt
```

**Benefits**:
- Second-order accurate
- Implicit velocity extraction (no separate velocity solve)
- Natural damping from constraint projection

### Inertia Transfer

**Purpose**: Transfer head motion to hair roots for realistic following behavior.

```
# Compute head motion in local space
delta_position = head_position - prev_head_position
delta_rotation = head_rotation * inverse(prev_head_rotation)

# Apply to root particle with decay
For i in 0..N:
    decay = inertia_decay ^ i
    positions[i] += delta_position * decay
    positions[i] = rotate_around(root, delta_rotation, positions[i]) * decay
```

### Guide Hair Interpolation

**Purpose**: Reduce simulation cost by only simulating guide hairs.

**Approach**:
1. Select subset of hairs as "guides" (e.g., 1 in 8)
2. Simulate only guide hairs
3. Interpolate render hairs from nearest guides

**Interpolation**:
```
For each render hair:
    # Find N nearest guide hairs
    guides = find_nearest_guides(root_position, N)
    
    # Inverse-distance weighted interpolation
    weights = [1 / distance(root, guide.root) for guide in guides]
    weights = normalize(weights)
    
    For each segment:
        position = sum(weight * guide.position for weight, guide in zip(weights, guides))
```

---

## Collision System

### Point-Capsule Collision

**Purpose**: Hair collision with limbs, body parts.

```
# Capsule defined by (point_a, point_b, radius)

# Find closest point on capsule axis
t = dot(point - capsule_a, axis) / length_sq(axis)
t = clamp(t, 0, 1)
closest = capsule_a + t * axis

# Check penetration
direction = point - closest
distance = length(direction)

if distance < radius:
    # Push out with friction
    penetration = radius - distance
    normal = direction / distance
    point = point + normal * penetration
    velocity = apply_friction(velocity, normal, friction_coeff)
```

### Point-Sphere Collision

**Purpose**: Hair collision with head, spherical obstacles.

```
# Sphere defined by (center, radius)

direction = point - center
distance = length(direction)

if distance < radius:
    # Push to surface
    normal = direction / distance
    point = center + normal * radius
    velocity = apply_friction(velocity, normal, friction_coeff)
```

### SDF Collision

**Purpose**: Hair collision with arbitrary geometry via Signed Distance Field.

```
# SDF: function returning signed distance and gradient

distance, gradient = sdf.evaluate(point)

if distance < 0:
    # Push to surface
    point = point - gradient * distance
    velocity = apply_friction(velocity, gradient, friction_coeff)
```

### Density-Field Self-Collision

**Purpose**: Prevent hair-hair interpenetration without O(n^2) tests.

```
# Build density field from all hair particles
density_grid = build_density_grid(all_particles)

For each particle:
    density = sample_density_grid(position)
    gradient = sample_density_gradient(position)
    
    if density > threshold:
        # Push away from high-density regions
        push = gradient * (density - threshold) * stiffness
        position = position + push
```

**Why Density Field**:
- O(n) to build grid
- O(1) lookup per particle
- Approximate but fast
- No pair enumeration

---

## Constraint Types

### Length Constraint (Inextensibility)

**Purpose**: Prevent hair segments from stretching.

```
# Edge from particle i to j with rest length L

direction = positions[j] - positions[i]
current_length = length(direction)
direction = direction / current_length

# Move both particles to satisfy constraint
correction = (current_length - L) / 2
positions[i] = positions[i] + direction * correction * weight_i
positions[j] = positions[j] - direction * correction * weight_j
```

### Global Shape Matching

**Purpose**: Restore hair to rest pose over time.

```
# Compute centroid of current and rest configuration
centroid_current = mean(positions)
centroid_rest = mean(rest_positions)

# Compute optimal rotation (polar decomposition)
A = sum(outer(positions[i] - centroid_current, rest_positions[i] - centroid_rest))
R = polar_decomposition_rotation(A)

# Blend toward matched configuration
For each particle:
    target = R @ (rest_positions[i] - centroid_rest) + centroid_current
    positions[i] = lerp(positions[i], target, blend_factor)
```

### Local Shape Constraint (Angle Preservation)

**Purpose**: Maintain curvature/styling of hair.

**Rodrigues Rotation Formula**:
```
# Given current edge direction and target angle, compute rotation

# Rotation axis (perpendicular to both edges)
axis = normalize(cross(current_edge, previous_edge))

# Rotation angle (difference from rest angle)
correction_angle = rest_angle - current_angle

# Rodrigues formula
cos_a = cos(correction_angle)
sin_a = sin(correction_angle)
rotated = (
    edge * cos_a
    + cross(axis, edge) * sin_a
    + axis * dot(axis, edge) * (1 - cos_a)
)
```

**Why Rodrigues over Quaternions**:
- Small angle corrections (Rodrigues is exact and stable)
- Axis-angle representation matches constraint definition
- No quaternion normalization overhead

---

## LOD System

### LOD Levels

| Level | Simulation | Rendering | Use Case |
|-------|------------|-----------|----------|
| HIGH | All strands | All strands | Close-up |
| MEDIUM | Guide hairs | Interpolated | Medium distance |
| LOW | Reduced guides | Interpolated | Far distance |
| SHELL | None | Shell textures | Very far |

### Distance Thresholds with Hysteresis

```
# Level-up threshold > level-down threshold to prevent thrashing

if current_level == HIGH:
    if distance > threshold_high_to_medium:
        level = MEDIUM
elif current_level == MEDIUM:
    if distance < threshold_medium_to_high:
        level = HIGH
    elif distance > threshold_medium_to_low:
        level = LOW
# ... etc
```

**Hysteresis Gap**: Typically 10-20% between up/down thresholds.

### Guide Hair Selection

Selection criteria for guide hairs:
1. **Uniform distribution**: Select every N-th hair
2. **Importance sampling**: Prioritize silhouette hairs
3. **Clustering**: K-means on root positions, select cluster centers

### Shell Rendering Data

For SHELL level, prepare:
1. Shell mesh layers at increasing distances from head
2. Texture with hair density, direction, color
3. Alpha transparency based on strand density

---

## Data Flow

```
Head Transform Update
    |
    v
Inertia Transfer (root particles)
    |
    v
External Forces (gravity, wind)
    |
    v
Verlet Integration (position update)
    |
    v
Constraint Solving:
    +-- FTL length constraints (per strand)
    +-- Shape constraints (global/local)
    +-- Collision constraints (capsule, sphere, SDF)
    +-- Self-collision (density field)
    |
    v
Velocity Extraction
    |
    v
LOD Selection (per strand group)
    |
    v
Guide Hair Interpolation (if LOD < HIGH)
    |
    v
Render Data Output
```

---

## Configuration Points

| Parameter | Purpose |
|-----------|---------|
| `strand_count` | Total hair strands |
| `segments_per_strand` | Particles per strand |
| `guide_ratio` | Fraction simulated at MEDIUM LOD |
| `inertia_decay` | Root-to-tip inertia falloff |
| `length_stiffness` | Length constraint strength |
| `shape_stiffness` | Shape matching strength |
| `friction_coefficient` | Collision friction |
| `self_collision_stiffness` | Density field push strength |
| `lod_thresholds` | Distance breakpoints for LOD |
| `hysteresis_margin` | LOD transition gap |

---

## Integration Points

### Input Interfaces
- `HeadTransformProvider`: Head position/rotation each frame
- `ColliderSet`: Capsules, spheres, SDFs for collision

### Output Interfaces
- `StrandRenderData`: Positions, tangents, UVs for rendering
- `ShellRenderData`: Shell mesh and textures for distant LOD
