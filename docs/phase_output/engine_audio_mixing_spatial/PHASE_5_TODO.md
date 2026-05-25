# PHASE 5 TODO: Environment and Propagation

**RDC Phase Task Breakdown**
**Phase**: Occlusion, Propagation, Reverb Zones, and Materials

---

## Task 5.1: Material Database Implementation

**File**: `engine/audio/spatial/materials.py`
**Estimated Lines**: ~581

### Subtasks

- [ ] Define `AcousticMaterial` dataclass with 6-band absorption
- [ ] Implement `MaterialDatabase.__init__` with preset registry
- [ ] Add 15 preset materials with published absorption coefficients:
  - [ ] CONCRETE, BRICK, WOOD, GLASS, METAL
  - [ ] CARPET, FABRIC, TILE, DRYWALL
  - [ ] GRASS, GRAVEL, WATER, SNOW
  - [ ] ACOUSTIC_FOAM, ACOUSTIC_TILE
- [ ] Implement `get_material()` lookup by name
- [ ] Implement `register_material()` for custom materials
- [ ] Implement `calculate_nrc()`:
  - [ ] `NRC = (a_250 + a_500 + a_1000 + a_2000) / 4`
- [ ] Implement `calculate_sabine_rt60()`:
  - [ ] `RT60 = 0.161 * V / A`
- [ ] Implement `calculate_eyring_rt60()`:
  - [ ] `RT60 = 0.161 * V / (-S * ln(1 - alpha))`
- [ ] Implement `interpolate_absorption()` for arbitrary frequency

### Acceptance Criteria

- Absorption coefficients match published acoustic data
- NRC calculation matches standard formula
- Both Sabine and Eyring RT60 available
- Custom materials can override presets

---

## Task 5.2: Occlusion Detector Implementation

**File**: `engine/audio/spatial/occlusion.py`
**Estimated Lines**: ~572

### Subtasks

- [ ] Define `OcclusionSettings` with ray count, spread, freq range
- [ ] Define `OcclusionResult` with factor, low_pass_freq, avg_transmission
- [ ] Implement `OcclusionDetector.__init__` with raycast callback
- [ ] Implement `_generate_ray_origins()`:
  - [ ] Direct ray at center
  - [ ] Spread rays in cone pattern
  - [ ] Use spherical distribution
- [ ] Implement `detect()`:
  - [ ] Cast all rays
  - [ ] Count blocked vs open
  - [ ] Accumulate transmission factors
  - [ ] Calculate effective occlusion
- [ ] Implement `_calculate_low_pass_freq()`:
  - [ ] `freq = max_freq - (max_freq - min_freq) * occlusion^2`
- [ ] Implement per-source state caching
- [ ] Add early-out when direct path is clear

### Acceptance Criteria

- Multi-ray spread improves accuracy over single ray
- Low-pass frequency scales with occlusion
- Material transmission affects final occlusion
- Efficient caching reduces raycast load

---

## Task 5.3: Propagation Calculator Implementation

**File**: `engine/audio/spatial/propagation.py`
**Estimated Lines**: ~791

### Subtasks

- [ ] Define `PropagationPath` with type, length, energy, direction
- [ ] Define `PropagationResult` with paths list, dominant direction
- [ ] Implement `PropagationCalculator.__init__` with geometry callbacks
- [ ] Implement `calculate()` main entry point
- [ ] Implement `_calculate_direct_path()`:
  - [ ] Check occlusion
  - [ ] Calculate energy loss
- [ ] Implement `_calculate_reflections()` (image source):
  - [ ] Get nearby surfaces
  - [ ] Mirror source across each surface
  - [ ] Validate path visibility
  - [ ] Recurse for multiple bounces (max 4)
  - [ ] Apply reflection coefficient per bounce
- [ ] Implement `_calculate_diffraction()` (UTD):
  - [ ] Find nearby edges
  - [ ] Calculate shortest path over each edge
  - [ ] Apply diffraction coefficient
- [ ] Implement `_merge_paths()`:
  - [ ] Combine paths with similar directions
  - [ ] Calculate energy-weighted dominant direction
- [ ] Implement `PropagationCache`:
  - [ ] Key by source ID
  - [ ] Invalidate on position change > tolerance
  - [ ] Store calculated paths

### Acceptance Criteria

- Direct path calculated correctly
- Reflections produce image sources at correct positions
- Diffraction paths route around edges
- Cache reduces recalculation frequency

---

## Task 5.4: Reverb Zone Manager Implementation

**File**: `engine/audio/spatial/reverb_zone.py`
**Estimated Lines**: ~490

### Subtasks

- [ ] Define `ReverbPreset` enum (SMALL_ROOM, CATHEDRAL, etc.)
- [ ] Define `ReverbZone` with volume, RT60, wet_mix, early_delay
- [ ] Implement `ReverbZoneManager.__init__` with zone registry
- [ ] Implement `create_zone()` with AABB bounds
- [ ] Implement `destroy_zone()` cleanup
- [ ] Implement `get_active_zones()` for listener position
- [ ] Implement `_calculate_blend()`:
  - [ ] Check priority
  - [ ] Calculate distance to zone center
  - [ ] Apply smoothstep fade at boundaries
- [ ] Implement `get_reverb_params()`:
  - [ ] Blend parameters from active zones
  - [ ] Return unified reverb settings
- [ ] Add preset zone configurations:
  - [ ] SMALL_ROOM: RT60=0.4, diffusion=0.7
  - [ ] MEDIUM_ROOM: RT60=0.8, diffusion=0.8
  - [ ] LARGE_ROOM: RT60=1.5, diffusion=0.85
  - [ ] HALLWAY: RT60=2.0, diffusion=0.6
  - [ ] CATHEDRAL: RT60=4.0, diffusion=0.95
  - [ ] OUTDOOR: RT60=0.1, diffusion=0.3

### Acceptance Criteria

- Zones trigger when listener enters AABB
- Multiple zones blend by priority and distance
- Smoothstep prevents sudden parameter changes
- Presets match expected acoustic characteristics

---

## Task 5.5: Integration and Module Init

**File**: `engine/audio/spatial/__init__.py`
**Estimated Lines**: ~415

### Subtasks

- [ ] Import and export all Phase 5 classes
- [ ] Implement `create_propagation_calculator()` factory
- [ ] Implement `create_occlusion_detector()` factory
- [ ] Implement `create_reverb_zone()` factory
- [ ] Document integration with geometry system

### Acceptance Criteria

- All public classes exported
- Factory functions documented with examples
- Geometry callback requirements documented

---

## Dependencies

- Phase 3 and 4: Positioning, Spatialization
- Geometry system for raycasting
- NumPy for calculations

---

## Verification

1. Occlusion test: source behind wall, verify muffling
2. Reflection test: room with walls, verify audible reflections
3. Diffraction test: source around corner, verify bending
4. Reverb zone test: walk through zones, verify smooth transitions
5. Material test: different materials, verify absorption differences
6. RT60 test: measure decay time, compare to calculated value
