# PHASE 2 ARCH: Runtime Animation Systems

**RDC Workflow Output**
**Generated:** 2026-05-23
**Phase:** 2 of 3

---

## Phase Overview

Phase 2 implements the runtime systems that consume the Phase 1 infrastructure. This phase focuses on per-frame updates, state management, and smooth transitions.

---

## 1. Motion Matching Controller

### 1.1 State Machine

```
ControllerState
├── IDLE          # Character stationary, idle animation
├── MOVING        # Character locomoting, motion matching active
├── TRANSITIONING # Blending between motion segments
└── STOPPED       # Controller paused
```

### 1.2 Controller Architecture

```
MotionMatchingController
├── database: MotionDatabase
├── search: MotionSearch
├── blender: InertializationBlender
├── context: MotionContext
├── state: ControllerState
└── config: ControllerTimingConfig
```

### 1.3 Update Loop

Per-frame sequence:
1. Update desired trajectory from input
2. Check if transition is active; if so, update blender
3. Evaluate search cost improvement threshold
4. If search triggered, find best match
5. If new match found, initiate transition
6. Advance playback frame
7. Return current pose

### 1.4 Trajectory Prediction

```
DesiredTrajectory
├── positions: List[Vec3]     # Future world positions
├── facings: List[Vec2]       # Future facing directions
├── time_points: List[float]  # Prediction horizons
└── speed: float              # Current movement speed
```

Input sources:
- Gamepad: analog stick direction + magnitude
- Keyboard: WASD → 8-direction quantized
- Velocity: current physics velocity extrapolation

---

## 2. Inertialization Transition System

### 2.1 Offset Representation

```
BoneOffset
├── position_offset: Vec3
├── position_velocity: Vec3
├── rotation_offset: Quaternion
└── rotation_velocity: Vec3
```

### 2.2 Spring Decay

Critical damped spring decay formula:
```
new_offset = (offset + velocity * dt) * exp(-decay * dt)
new_velocity = velocity * exp(-decay * dt)
```

Parameters:
- `decay`: Controls how fast offset diminishes (typical: 2.0-5.0)
- Higher decay = faster transition = more jarring
- Lower decay = smoother transition = more sluggish

### 2.3 Transition States

```
MotionTransition
├── source_clip: int
├── source_frame: float
├── target_clip: int
├── target_frame: float
├── elapsed_time: float
├── duration: float
└── blend_mode: BlendMode
```

---

## 3. Foot Sliding Correction

### 3.1 Contact Detection

Foot is grounded when:
- Foot bone height < ground_threshold
- Foot bone velocity < velocity_threshold
- FootContact feature > contact_threshold (from motion matching)

### 3.2 Correction Algorithm

When foot is grounded:
1. Store world-space foot position at contact start
2. Each frame: compute delta between locked position and animation position
3. Apply inverse delta to root to maintain foot in place
4. Blend correction out as contact ends

### 3.3 Hysteresis

- `contact_on_threshold`: 0.7 (must exceed to enter contact)
- `contact_off_threshold`: 0.3 (must drop below to exit contact)
- Prevents flickering at threshold boundary

---

## 4. Procedural Locomotion

### 4.1 Gait Configuration

```
GaitConfig
├── cycle_duration: float      # Full cycle time (seconds)
├── stance_phase_ratio: float  # Portion spent on ground
├── swing_height: float        # Max foot lift height
├── stride_length: float       # Distance covered per cycle
├── body_bob_amplitude: float  # Vertical body motion
├── body_sway_amplitude: float # Lateral body motion
└── leg_order: List[int]       # Phase offsets per leg
```

### 4.2 Supported Gaits

| Gait | Legs | Phase Pattern |
|------|------|---------------|
| Biped Walk | 2 | 180° offset |
| Biped Run | 2 | 180° offset, shorter stance |
| Quadruped Trot | 4 | Diagonal pairs in sync |
| Quadruped Gallop | 4 | Sequential bound |

### 4.3 Foot Trajectory

**Stance Phase:** Linear slide along ground
**Swing Phase:** Parabolic arc with configurable height

```
t = (phase - stance_end) / swing_duration
height = swing_height * 4 * t * (1 - t)  # Parabola peak at midpoint
```

---

## 5. Look-At Controller

### 5.1 Joint Chain

```
LookAtChain
├── eye_left: JointConfig
├── eye_right: JointConfig
├── head: JointConfig
├── neck: JointConfig
└── upper_spine: JointConfig  # Optional
```

### 5.2 Angle Limits

| Joint | Yaw Range | Pitch Range |
|-------|-----------|-------------|
| Eye | +/- 30° | +/- 20° |
| Head | +/- 60° | +/- 30° |
| Neck | +/- 45° | +/- 15° |

### 5.3 Update Algorithm

1. Compute direction to target in local space
2. Clamp to joint angle limits
3. Apply rotation offset
4. Add saccade offset (eyes only)

### 5.4 Saccade Generation

```
SaccadeGenerator
├── next_saccade_time: float
├── current_offset: Vec2
├── target_offset: Vec2
└── offset_velocity: Vec2
```

Behavior:
- Random interval between saccades: 0.1-3.0 seconds
- Random target offset: up to 3° from center
- High-speed transition: 500°/s
- Smooth ease-out when reaching target

---

## 6. Breathing Controller

### 6.1 Cycle Phases

| Phase | Chest Motion |
|-------|--------------|
| Inhale | Expand forward/up |
| Hold | Peak expansion |
| Exhale | Contract back/down |
| Rest | Minimal motion |

### 6.2 Exertion Levels

| Level | Breaths/Min | Chest Amplitude |
|-------|-------------|-----------------|
| Resting | 12-16 | Low |
| Light | 16-20 | Medium-Low |
| Moderate | 20-25 | Medium |
| Heavy | 25-35 | Medium-High |
| Extreme | 35-50 | High |

### 6.3 Affected Bones

- Chest (primary expansion)
- Upper spine (secondary)
- Shoulders (raise on inhale)
- Clavicles (optional)

---

## 7. Idle Detection

### 7.1 Hysteresis State Machine

```
IdleDetector
├── state: IDLE | MOVING
├── idle_enter_threshold: float  # Speed below this = maybe idle
├── idle_exit_threshold: float   # Speed above this = definitely moving
├── idle_enter_time: float       # Must stay below threshold for this long
└── time_below_threshold: float  # Current accumulated time
```

### 7.2 Benefits

- Prevents rapid IDLE↔MOVING flickering
- Allows motion matching to settle before switching to idle
- Configurable timing per character archetype

---

## 8. Annotation Runtime

### 8.1 Tag Queries

Fast tag lookup via pre-built indices:
```python
database.get_entries_with_tag("walk")  # O(1) lookup
database.get_entries_with_tags(["walk", "forward"])  # Intersection
```

### 8.2 Cost Modifiers

Per-entry cost modifiers for authorial control:
- Preferred motion: 0.5x cost
- Fallback motion: 2.0x cost
- Disabled motion: infinity cost

---

## 9. Performance Considerations

### 9.1 Search Amortization

- Full search every N frames (configurable: 3-5)
- Quick cost evaluation every frame (compare current vs best candidate)
- Transition only if improvement exceeds threshold

### 9.2 Feature Caching

- Current pose features cached, updated only when pose changes
- Trajectory features recomputed each frame (input dependent)

### 9.3 Vectorized Cost Computation

- NumPy broadcast for parallel feature comparison
- float64 accumulation for numerical stability
- float32 storage for memory efficiency
