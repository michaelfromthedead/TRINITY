# PHASE 5 ARCH: Environment and Propagation

**RDC Phase Architecture**
**Phase**: Occlusion, Propagation, Reverb Zones, and Materials

---

## Phase Overview

Implement environmental audio effects: occlusion when geometry blocks sound, propagation paths including reflections and diffraction, reverb zones for room acoustics, and material properties for realistic absorption.

---

## Components

### 5.1 OcclusionDetector

**Purpose**: Detect geometry blocking direct path between source and listener.

**Multi-Ray Approach**:
- Cast multiple rays from source to listener
- Rays spread in cone pattern around direct path
- Count blocked vs. open rays
- Apply weighted occlusion factor

**Algorithm**:
```python
def detect(source_pos, listener_pos, num_rays, spread):
    rays = generate_ray_origins(source_pos, listener_pos, num_rays, spread)
    blocked = 0
    total_transmission = 0.0
    
    for ray in rays:
        hit = raycast(ray.origin, ray.direction)
        if hit and hit.distance < total_distance:
            blocked += 1
            total_transmission += hit.material.transmission
    
    occlusion_factor = blocked / num_rays
    avg_transmission = total_transmission / blocked if blocked > 0 else 1.0
    effective_occlusion = occlusion_factor * (1.0 - avg_transmission)
    return effective_occlusion
```

**Occlusion Effects**:
- Volume reduction proportional to occlusion
- Low-pass filter: `max_freq - (max_freq - min_freq) * occlusion^2`
- Higher frequencies blocked more than lower

**Parameters**:
- `OCCLUSION_MAX_RAYS = 8`
- `OCCLUSION_SPREAD = 0.1` (radians)
- `OCCLUSION_MIN_FREQ = 200` (Hz)
- `OCCLUSION_MAX_FREQ = 4000` (Hz)

### 5.2 PropagationCalculator

**Purpose**: Find sound paths including reflections and diffraction.

**Path Types**:
1. **Direct**: Straight line (may be occluded)
2. **Reflected**: Bounces off surfaces (image source method)
3. **Diffracted**: Bends around edges (UTD)

#### Image Source Method (Reflections)

For each reflective surface:
1. Mirror source position across surface
2. Check if image source has line-of-sight to listener
3. Trace path: source -> reflection point -> listener
4. Calculate path length and reflection loss

**Implementation**:
- Simplified: up to 4 bounces
- Recursively mirror source across surfaces
- Validate each path segment

#### Uniform Theory of Diffraction (UTD)

For edges (wall corners, pillars):
1. Find shortest path over edge
2. Calculate diffraction coefficient based on angle
3. Apply frequency-dependent attenuation

**Propagation Cache**:
- Cache paths per source
- Invalidate on source/listener movement > tolerance
- Re-use paths within position tolerance

**Output**: `PropagationResult`
- `paths: List[PropagationPath]` - All valid paths
- `dominant_direction: Vec3` - Energy-weighted direction
- `total_energy: float` - Sum of path energies

### 5.3 ReverbZoneManager

**Purpose**: Assign reverb parameters based on listener position.

**ReverbZone**:
- `volume: AABB` - Trigger region
- `rt60: float` - Reverb time (seconds)
- `wet_mix: float` - Reverb blend (0-1)
- `early_delay: float` - Pre-delay (ms)
- `diffusion: float` - Reverb diffusion
- `priority: int` - Conflict resolution

**Zone Blending**:
- Listener can be in multiple overlapping zones
- Blend parameters by priority and distance to zone center
- Smoothstep fade at zone boundaries

**Presets**:
| Preset | RT60 | Diffusion | Use |
|--------|------|-----------|-----|
| SMALL_ROOM | 0.4s | 0.7 | Closets, bathrooms |
| MEDIUM_ROOM | 0.8s | 0.8 | Offices, bedrooms |
| LARGE_ROOM | 1.5s | 0.85 | Halls, gyms |
| HALLWAY | 2.0s | 0.6 | Corridors |
| CATHEDRAL | 4.0s | 0.95 | Large reverberant spaces |
| OUTDOOR | 0.1s | 0.3 | Minimal reverb |

### 5.4 MaterialDatabase

**Purpose**: Store acoustic properties for geometry materials.

**Material Properties**:
- `absorption: List[float]` - 6-band coefficients (125Hz to 4kHz)
- `reflection: float` - Specular reflection coefficient
- `transmission: float` - Sound passes through (for occlusion)
- `scattering: float` - Diffuse reflection
- `density: float` - Material density (kg/m3)

**6-Band Frequencies**: 125Hz, 250Hz, 500Hz, 1kHz, 2kHz, 4kHz

**Preset Materials** (15):
| Material | Absorption (avg) | Transmission | Use |
|----------|-----------------|--------------|-----|
| CONCRETE | 0.02 | 0.01 | Walls, floors |
| BRICK | 0.03 | 0.01 | Walls |
| WOOD | 0.10 | 0.05 | Floors, furniture |
| GLASS | 0.04 | 0.02 | Windows |
| METAL | 0.01 | 0.00 | Machinery |
| CARPET | 0.40 | 0.10 | Floors |
| FABRIC | 0.50 | 0.20 | Curtains, furniture |
| TILE | 0.02 | 0.01 | Bathrooms |
| DRYWALL | 0.08 | 0.08 | Interior walls |
| GRASS | 0.30 | 1.00 | Exterior |
| GRAVEL | 0.50 | 0.80 | Exterior paths |
| WATER | 0.02 | 0.50 | Pools, lakes |
| SNOW | 0.70 | 0.90 | Winter exterior |
| ACOUSTIC_FOAM | 0.90 | 0.05 | Studios |
| ACOUSTIC_TILE | 0.80 | 0.10 | Ceilings |

**RT60 Calculation**:

Sabine Equation:
```
RT60 = 0.161 * V / A
```
Where V = volume (m3), A = total absorption (sum of surface_area * absorption)

Eyring Equation (for high absorption):
```
RT60 = 0.161 * V / (-S * ln(1 - alpha))
```
Where S = total surface area, alpha = average absorption coefficient

**NRC Calculation**:
```
NRC = (alpha_250 + alpha_500 + alpha_1000 + alpha_2000) / 4
```

---

## Data Flow

```
Source Position
      |
      +----------------+
      |                |
      v                v
Propagation       Occlusion
Calculator        Detector
      |                |
      v                v
Multiple         Volume +
Paths            Low-pass
      |                |
      +-------+--------+
              |
              v
        Reverb Zone
        Manager
              |
              v
        Apply Reverb
        Parameters
```

---

## Integration with Geometry System

**Required Callbacks**:
- `raycast(origin, direction, max_distance)` -> HitResult
- `get_surfaces_in_sphere(center, radius)` -> List[Surface]
- `get_edges_in_sphere(center, radius)` -> List[Edge]

**HitResult**:
- `hit: bool`
- `distance: float`
- `position: Vec3`
- `normal: Vec3`
- `material: Material`
- `transmission: float`

---

## Success Criteria

1. Occlusion produces audible muffling behind walls
2. Reflections create realistic spatial richness
3. Reverb zones blend smoothly at boundaries
4. RT60 matches measured room acoustics
5. Material absorption affects reflection energy
