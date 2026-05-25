# PHASE 3 TODO: Spatial Audio Core

**RDC Phase Task Breakdown**
**Phase**: Positioning, Attenuation, and Listener Management

---

## Task 3.1: Configuration Module

**File**: `engine/audio/spatial/config.py`
**Estimated Lines**: ~530

### Subtasks

- [ ] Define physics constants (SPEED_OF_SOUND, HEAD_RADIUS)
- [ ] Define spatial constants (MAX_DISTANCE, MAX_LISTENERS)
- [ ] Define attenuation defaults
- [ ] Define panning constants
- [ ] Implement utility functions (angle normalization, etc.)

### Acceptance Criteria

- All constants documented with units
- Constants match published acoustic values

---

## Task 3.2: ListenerManager Implementation

**File**: `engine/audio/spatial/positioning.py` (partial)
**Estimated Lines**: ~250 (of 656 total)

### Subtasks

- [ ] Define `ListenerState` dataclass
- [ ] Implement `ListenerManager.__init__` with listener slots
- [ ] Implement `create_listener()` returning listener ID
- [ ] Implement `destroy_listener()` cleanup
- [ ] Implement `set_position()` / `get_position()`
- [ ] Implement `set_orientation()` with forward/up vectors
- [ ] Implement `set_velocity()` for Doppler
- [ ] Implement `world_to_listener_space()` matrix transform
- [ ] Implement `get_direction_to()` source direction
- [ ] Implement `get_distance_to()` source distance
- [ ] Implement `get_active_listeners()` query

### Acceptance Criteria

- Up to 4 listeners supported
- Orientation normalized automatically
- World-to-local transform computed correctly

---

## Task 3.3: Source Types Implementation

**File**: `engine/audio/spatial/positioning.py` (partial)
**Estimated Lines**: ~400 (of 656 total)

### Subtasks

- [ ] Define `SpatialSource` abstract base class
- [ ] Implement `PointSource`:
  - [ ] `get_closest_point()` returns position
  - [ ] `get_distance()` euclidean distance
  - [ ] `get_direction()` normalized vector
  - [ ] `is_in_range()` distance check
- [ ] Implement `AreaSource`:
  - [ ] Store center, dimensions, normal
  - [ ] `get_closest_point()` clamp to rectangle
  - [ ] Handle oriented rectangles
- [ ] Implement `LineSource`:
  - [ ] Store start, end points
  - [ ] `get_closest_point()` project onto segment
- [ ] Implement `VolumeSource`:
  - [ ] Store center, dimensions
  - [ ] `get_closest_point()` clamp to box
  - [ ] `get_distance()` return 0 if inside
- [ ] Implement `create_source()` factory function

### Acceptance Criteria

- All source types implement common interface
- Interior detection works for VolumeSource
- Closest point calculations geometrically correct

---

## Task 3.4: AttenuationCurve Implementation

**File**: `engine/audio/spatial/attenuation.py`
**Estimated Lines**: ~520

### Subtasks

- [ ] Define `AttenuationModel` enum (LINEAR, LOGARITHMIC, etc.)
- [ ] Define `AttenuationCurve` abstract base
- [ ] Implement `LinearAttenuation`:
  - [ ] Formula: `1 - rolloff * normalized_distance`
- [ ] Implement `LogarithmicAttenuation`:
  - [ ] Formula: `1 / (1 + rolloff * log2(d / min))`
- [ ] Implement `InverseAttenuation`:
  - [ ] Formula: `min / (min + rolloff * (d - min))`
- [ ] Implement `InverseSquaredAttenuation`:
  - [ ] Formula: `(min / d)^2`
- [ ] Implement `CustomAttenuation`:
  - [ ] Designer-defined curve points
  - [ ] Smoothstep interpolation between points
- [ ] Implement `NoneAttenuation`:
  - [ ] Always returns 1.0
- [ ] Implement `create_attenuation()` factory function
- [ ] All curves handle edge cases (d < min, d > max)

### Acceptance Criteria

- All curves return 1.0 at min_distance
- All curves return 0.0 at max_distance
- Inverse squared matches physical point source behavior
- Custom curve interpolates smoothly

---

## Task 3.5: ConeAttenuation Implementation

**File**: `engine/audio/spatial/attenuation.py` (extension)
**Estimated Lines**: ~100

### Subtasks

- [ ] Define `ConeAttenuation` class
- [ ] Implement `calculate()` with inner/outer cone logic
- [ ] Handle edge cases (source facing away from listener)
- [ ] Add integration with distance attenuation (multiplicative)

### Acceptance Criteria

- Inner cone: full volume
- Outer cone: outer_volume
- Smooth interpolation between cones

---

## Task 3.6: StereoPanner Implementation

**File**: `engine/audio/spatial/spatialization.py` (partial)
**Estimated Lines**: ~150

### Subtasks

- [ ] Define `Spatializer` abstract base class
- [ ] Implement `StereoPanner`:
  - [ ] Constant-power panning: `cos(angle)`, `sin(angle)`
  - [ ] Convert azimuth to pan value
- [ ] Return `SpatializationResult` with left/right gains

### Acceptance Criteria

- Center (0 pan) produces equal L/R
- Hard left (-1) produces L=1, R=0
- Hard right (+1) produces L=0, R=1
- Constant power: `L^2 + R^2 = 1`

---

## Dependencies

- NumPy for vector math
- Math library for trig functions

---

## Verification

1. Listener test: set position, verify direction/distance to source
2. Source type test: each type returns correct closest point
3. Attenuation test: verify curve shapes match formulas
4. Panning test: verify constant-power law
