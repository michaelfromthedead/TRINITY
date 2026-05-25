# PHASE 3 ARCH: Spatial Audio Core

**RDC Phase Architecture**
**Phase**: Positioning, Attenuation, and Listener Management

---

## Phase Overview

Establish the foundational spatial audio infrastructure: listener state, source types, distance attenuation, and basic stereo panning.

---

## Components

### 3.1 ListenerManager

**Purpose**: Track listener position, orientation, and velocity for spatial calculations.

**Multi-Listener Support**: Up to 4 listeners for split-screen.

**ListenerState**:
- `position: Vec3` - World position
- `forward: Vec3` - Look direction (normalized)
- `up: Vec3` - Up vector (normalized)
- `velocity: Vec3` - Movement velocity for Doppler
- `world_to_local: Mat4` - Transform matrix
- `listener_id: int` - Unique identifier

**Operations**:
- `set_position()` / `set_orientation()` / `set_velocity()`
- `get_direction_to()` - Direction from listener to source
- `get_distance_to()` - Distance from listener to source
- `world_to_listener_space()` - Transform world pos to local

### 3.2 Source Types

**Abstract Base**: `SpatialSource` with common interface.

**Common Interface**:
- `get_closest_point(listener_pos)` - Nearest point on source
- `get_distance(listener_pos)` - Distance to listener
- `get_direction(listener_pos)` - Direction toward listener
- `is_in_range(listener_pos, max_range)` - Range check

#### PointSource
- Single 3D position
- Simplest source type

#### AreaSource
- 2D rectangular region with normal
- Sound emanates from closest point on rectangle
- Used for large flat surfaces (walls, windows)

#### LineSource
- Path between two points
- Sound emanates from closest point on line segment
- Used for rivers, roads, wires

#### VolumeSource
- 3D box region
- Interior detection: inside volume = 0 distance
- Used for rooms, ambient zones

### 3.3 AttenuationCurve

**Purpose**: Calculate volume falloff with distance.

**Models**:

| Model | Formula | Use Case |
|-------|---------|----------|
| LINEAR | `1 - rolloff * (d - min) / (max - min)` | Predictable falloff |
| LOGARITHMIC | `1 / (1 + rolloff * log2(d / min))` | Natural perception |
| INVERSE | `min / (min + rolloff * (d - min))` | Smooth falloff |
| INVERSE_SQUARED | `(min / d)^2` | Physically accurate |
| CUSTOM | Designer curve with smoothstep | Full control |
| NONE | Always 1.0 | No distance falloff |

**Parameters**:
- `min_distance: float` - Distance at full volume
- `max_distance: float` - Distance at zero volume (cull)
- `rolloff: float` - Curve steepness

### 3.4 ConeAttenuation

**Purpose**: Directional sound with inner/outer cone.

**Parameters**:
- `inner_angle: float` - Angle of full volume cone
- `outer_angle: float` - Angle of full attenuation
- `outer_volume: float` - Volume at outer cone edge

**Algorithm**:
1. Calculate angle between source forward and direction to listener
2. If angle < inner_angle/2: full volume
3. If angle > outer_angle/2: outer_volume
4. Otherwise: interpolate between 1.0 and outer_volume

### 3.5 Basic Stereo Panner

**Purpose**: Simple left/right panning for stereo output.

**Constant-Power Panning**:
```python
# pan: -1.0 (left) to 1.0 (right)
angle = (pan + 1) * PI / 4  # 0 to PI/2
left_gain = cos(angle)
right_gain = sin(angle)
```

---

## Data Flow

```
Source Position
      |
      v
+-------------+
| Listener    |---> world_to_listener_space()
+-------------+
      |
      v
+-------------+
| Attenuation |---> distance -> volume
+-------------+
      |
      v
+-------------+
| Cone        |---> direction -> volume modifier
+-------------+
      |
      v
+-------------+
| Panner      |---> direction -> left/right gains
+-------------+
      |
      v
Output (stereo)
```

---

## Configuration

From `engine/audio/spatial/config.py`:
- `SPEED_OF_SOUND = 343.0` (m/s at 20C)
- `DEFAULT_MIN_DISTANCE = 1.0`
- `DEFAULT_MAX_DISTANCE = 100.0`
- `DEFAULT_ROLLOFF = 1.0`
- `MAX_LISTENERS = 4`

---

## Thread Safety

- Listener state updates are atomic (single writer, multiple readers)
- Source positions cached per frame
- Attenuation calculations are stateless (inherently thread-safe)

---

## Success Criteria

1. Listener position updates reflect in spatial calculations
2. All source types return correct closest point
3. Distance attenuation curves produce expected falloff
4. Stereo panning follows constant-power law
5. Multi-listener support functional
