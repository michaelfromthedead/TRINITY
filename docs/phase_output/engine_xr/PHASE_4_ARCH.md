# PHASE 4 ARCHITECTURE: Locomotion and Comfort

## Phase Overview

Phase 4 implements the movement systems that allow users to traverse virtual spaces while minimizing motion sickness. This phase covers teleportation, smooth locomotion, climbing, and comfort features (vignette, snap turn, comfort presets). Locomotion design directly impacts user comfort and session duration.

## Architectural Decisions

### ADR-XR-030: Multi-Mode Locomotion Strategy

**Context**: Different users have different comfort tolerances; no single locomotion mode works for everyone.

**Decision**: Implement three locomotion modes that can be combined:
1. **Teleport**: Instant or fade-based point-to-point movement
2. **Smooth**: Continuous thumbstick-based movement
3. **Climbing**: Grab-based movement on climbable surfaces

**Consequences**:
- Users can choose their comfort level
- Hybrid locomotion possible (teleport + smooth)
- Each mode has dedicated tuning parameters

### ADR-XR-031: Teleport Arc Physics

**Context**: Teleport visualization needs predictable, natural-feeling arc trajectory.

**Decision**: Use projectile motion physics:
```
y(t) = y0 + v0y*t - 0.5*g*t^2
x(t) = x0 + v0x*t
```

With configurable parameters:
- `arc_velocity`: Initial velocity magnitude
- `arc_gravity`: Gravity strength (can differ from world gravity)
- `max_distance`: Arc cutoff distance

**Consequences**:
- Arc feels natural (matches thrown objects)
- Designer control over arc shape
- Ground intersection via linear interpolation

### ADR-XR-032: Smooth Locomotion Input Processing

**Context**: Raw thumbstick input feels wrong due to drift, curves, and reference frames.

**Decision**: Implement three-stage input processing:
1. **Deadzone**: Eliminate center drift (typically 0.1-0.2)
2. **Curve**: Apply power function for precision at low values
3. **Reference Frame**: Select head, left hand, or right hand for direction

**Consequences**:
- Precise control at low speeds
- Natural direction mapping to user intent
- Configurable per-user preference

### ADR-XR-033: Climbing State Machine

**Context**: Climbing involves complex state (grabbing, climbing, mantling, falling).

**Decision**: Implement explicit state machine:
- **IDLE**: Not touching climbable surface
- **GRABBING**: Hand on surface, not moving
- **CLIMBING**: Moving via hand pull
- **MANTLING**: Pulling up over ledge
- **FALLING**: Released grip, in air

**Consequences**:
- Clear transitions between states
- Mantle animation has explicit trigger
- Stamina system ties into states

### ADR-XR-034: Comfort Vignette Strategy

**Context**: Peripheral vision is most sensitive to motion sickness triggers.

**Decision**: Implement dynamic vignette that:
- Activates on linear velocity above threshold
- Activates on angular velocity above threshold
- Fades in/out smoothly to avoid jarring transitions
- Provides shader parameters for rendering

**Consequences**:
- Reduces peripheral motion cues during locomotion
- Smooth transitions prevent vignette itself causing discomfort
- Configurable per-user

### ADR-XR-035: Comfort Preset System

**Context**: Users need simple presets rather than tuning 20 individual parameters.

**Decision**: Implement five named presets:
| Preset | Snap Turn | Vignette | Tunneling | Speed |
|--------|-----------|----------|-----------|-------|
| veteran | Off | Off | Disabled | 1.0 |
| intermediate | 45 deg | 0.4 | Mild | 0.9 |
| comfortable | 30 deg | 0.6 | Moderate | 0.7 |
| maximum | 30 deg | 0.8 | Strong | 0.5 |
| seated | 45 deg | 0.5 | Mild | 0.8 |

**Consequences**:
- One-click comfort configuration
- Presets cover common user profiles
- Custom preset for power users

### ADR-XR-036: Locomotion Provider Pattern

**Context**: Locomotion modes need runtime integration with physics and collision.

**Decision**: Each locomotion type has a Provider class:
- Holds reference to locomotion component
- Exposes integration interface
- Delegates to component methods

**Consequences**:
- Clean separation of data (component) and behavior (provider)
- Runtime can interact with any locomotion type uniformly
- Testing can mock providers

## Component Specifications

### Teleport Locomotion

```
TeleportLocomotion (Component)
├── Configuration
│   ├── max_distance: float (1-50m, default 10m)
│   ├── arc_gravity: float (default -9.8)
│   ├── arc_velocity: float (default 10 m/s)
│   ├── style: TeleportStyle (INSTANT, FADE, DASH, BLINK)
│   ├── fade_duration: float (0-0.5s)
│   └── snap_rotation: Optional[float] (e.g., 30 degrees)
├── State Machine
│   ├── state: TeleportState (IDLE, AIMING, VALIDATING, TRANSITIONING, COOLDOWN)
│   └── cooldown_time: float
├── Methods
│   ├── begin_aim(hand: XRHand) -> None
│   ├── update_aim(aim_direction: Vec3) -> TeleportTarget
│   ├── execute_teleport() -> bool
│   └── cancel_aim() -> None
└── Arc Calculation
    ├── calculate_arc(origin, direction) -> List[ArcSegment]
    └── find_landing_point(arc) -> Optional[Vec3]

TeleportArcCalculator
├── calculate_arc() -> List[Vec3]
│   # Projectile motion: p(t) = p0 + v0*t + 0.5*a*t^2
│   # Samples at configurable interval
├── find_landing_point() -> Optional[Vec3]
│   # Linear interpolation between samples
│   # Returns first ground intersection
└── classify_segments() -> List[ArcSegmentType]
    # VALID: ground below, reachable
    # INVALID: no ground, blocked
    # OUT_OF_RANGE: beyond max_distance

TeleportTarget
├── position: Vec3
├── normal: Vec3
├── rotation: Quat (optional snap rotation)
└── area: Optional[Bounds] (for area targets)

@xr_teleport_area Decorator
└── Marks valid teleport destinations
```

### Smooth Locomotion

```
SmoothLocomotion (Component)
├── Configuration
│   ├── speed: float (m/s)
│   ├── sprint_multiplier: float
│   ├── backward_speed_ratio: float (default 0.7)
│   ├── strafe_behavior: StrafeBehavior (NORMAL, REDUCED, DISABLED)
│   ├── direction_reference: DirectionReference (HEAD, LEFT_HAND, RIGHT_HAND)
│   └── dead_zone: float (0-1)
├── Turn Settings
│   ├── turn_type: TurnType (SNAP, SMOOTH, DISABLED)
│   ├── snap_angle: float (degrees)
│   ├── snap_cooldown: float (seconds)
│   └── smooth_turn_speed: float (degrees/second)
├── State
│   ├── state: MovementState (IDLE, MOVING, TURNING, DISABLED)
│   ├── is_grounded: bool
│   └── gravity_velocity: Vec3
├── Methods
│   ├── calculate_movement(input: Vec2, delta_time) -> MovementResult
│   ├── calculate_turn(input: float, delta_time) -> float
│   ├── set_grounded(grounded: bool) -> None
│   └── get_vignette_intensity() -> float
└── Arm Swing Mode
    └── calculate_arm_swing_movement(hand_velocities) -> Vec3

MovementResult
├── velocity: Vec3
├── rotation_delta: float
└── vignette_intensity: float

TurnSettings
├── type: TurnType
├── snap_angle: float
├── snap_cooldown: float
├── smooth_speed: float
└── dead_zone: float

@xr_locomotion Decorator
└── Configures locomotion type and speed on classes
```

### Climbing Locomotion

```
ClimbingLocomotion (Component)
├── Configuration
│   ├── max_reach: float (arm length)
│   ├── grip_threshold: float (0-1, trigger/grip amount)
│   ├── stamina_max: float
│   ├── stamina_drain_rate: float (per second)
│   ├── stamina_recovery_rate: float (per second, grounded)
│   └── mantle_threshold: float (height above ledge)
├── State
│   ├── state: ClimbingState (IDLE, GRABBING, CLIMBING, MANTLING, FALLING)
│   ├── left_hand_state: GrabHandState
│   ├── right_hand_state: GrabHandState
│   ├── stamina: float
│   └── grabbed_climbable: Optional[ClimbableVolume]
├── Methods
│   ├── try_grab(hand, grip_amount) -> bool
│   ├── release_grab(hand) -> None
│   ├── calculate_climbing_movement() -> Vec3
│   ├── start_mantle() -> bool
│   └── update(delta_time) -> None
├── Callbacks
│   ├── on_grab(hand, grab_point)
│   ├── on_release(hand)
│   ├── on_stamina_empty()
│   ├── on_mantle_start()
│   └── on_mantle_complete()
└── Physics Integration
    ├── climbing_gravity: float (reduced, e.g., -2.0)
    └── falling_gravity: float (full, e.g., -9.8)

ClimbableVolume
├── bounds: AABB or ConvexHull
├── climbable_type: ClimbableType (SURFACE, LADDER, ROPE, LEDGE, HOLDS)
├── grab_points: List[GrabPoint] (discrete or auto-generated)
├── grip_strength_required: float
├── stamina_drain_multiplier: float
├── is_point_inside(point) -> bool
└── get_nearest_grab_point(position) -> Optional[GrabPoint]

GrabPoint
├── position: Vec3
├── normal: Vec3
├── grip_type: str
├── radius: float
└── is_available: bool

@xr_climbable Decorator
└── Marks surfaces as climbable with configuration
```

### Comfort System

```
ComfortVignette
├── Configuration
│   ├── enabled: bool
│   ├── shape: VignetteShape (CIRCULAR, ELLIPTICAL, RECTANGULAR)
│   ├── color: Color (default black)
│   ├── inner_radius: float (0-1, clear area)
│   ├── outer_radius: float (0-1, full darkness)
│   ├── feather: float (transition smoothness)
│   ├── velocity_threshold: float (m/s)
│   └── angular_threshold: float (deg/s)
├── State
│   ├── current_intensity: float (0-1)
│   ├── target_intensity: float (0-1)
│   └── fade_speed: float
├── Methods
│   ├── update(linear_velocity, angular_velocity, delta_time)
│   ├── set_manual_intensity(intensity) -> None
│   └── get_shader_params() -> VignetteParams
└── VignetteParams (for shader)
    ├── intensity: float
    ├── inner_radius: float
    ├── outer_radius: float
    ├── feather: float
    └── color: Color

XRComfortSettings (Global Resource)
├── Preset
│   ├── current_preset: ComfortLevel
│   └── apply_preset(preset: str) -> None
├── Snap Turn
│   ├── snap_turn_enabled: bool
│   ├── snap_turn_angle: float
│   └── snap_turn_on_button: bool
├── Vignette
│   ├── vignette_enabled: bool
│   ├── vignette_intensity: float
│   ├── vignette_on_turn: bool
│   └── vignette_on_move: bool
├── Teleport
│   ├── teleport_fade_enabled: bool
│   └── teleport_fade_duration: float
├── Stability
│   ├── stable_horizon_enabled: bool
│   ├── lock_roll: bool
│   └── lock_pitch: bool
├── Seated Mode
│   ├── seated_mode_enabled: bool
│   └── seated_height_offset: float
├── Metrics
│   ├── cumulative_rotation: float
│   ├── cumulative_velocity: float
│   └── time_in_motion: float
└── Persistence
    ├── save(path) -> bool
    └── load(path) -> bool

ComfortPreset
├── name: str
├── settings: Dict[str, Any]
└── Built-in Presets
    ├── veteran: All comfort off, full speed
    ├── intermediate: Snap 45, vignette 0.4
    ├── comfortable: Snap 30, vignette 0.6, horizon lock
    ├── maximum: All comfort on, reduced speed
    └── seated: Snap 45, seated offset

ComfortManager (Runtime Service)
├── vignette: ComfortVignette
├── settings: XRComfortSettings
├── update(velocities, delta_time) -> None
└── apply_preset(preset_name) -> None

@xr_comfort Decorator
└── Configures comfort features on classes
```

### Locomotion Provider (Integration Layer)

```
LocomotionProvider (Abstract)
├── get_movement_velocity() -> Vec3
├── get_rotation_delta() -> float
├── is_grounded() -> bool
├── can_move() -> bool
└── set_enabled(enabled: bool) -> None

TeleportLocomotionProvider(LocomotionProvider)
├── teleport: TeleportLocomotion
├── begin_aim() / cancel_aim() / execute_teleport()
└── get_pending_teleport() -> Optional[TeleportTarget]

SmoothLocomotionProvider(LocomotionProvider)
├── smooth: SmoothLocomotion
├── process_input(thumbstick, turn_input, delta_time)
└── set_grounded(grounded: bool)

ClimbingLocomotionProvider(LocomotionProvider)
├── climbing: ClimbingLocomotion
├── try_grab() / release_grab()
└── get_climb_state() -> ClimbingState
```

## Integration Points

### Dependencies (Incoming)
- Phase 2: Controller input (thumbstick, trigger, grip)
- Physics System: Ground detection, collision
- `engine.xr.config`: XR_CONFIG for defaults

### Dependents (Outgoing)
- Character Controller: Consumes velocity/rotation
- Renderer: Vignette shader parameters
- UI: Comfort settings panel

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Controller Input                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │Thumbstick│ │ Trigger  │ │   Grip   │ │  Button  │       │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘       │
└───────┼────────────┼────────────┼────────────┼──────────────┘
        │            │            │            │
        ▼            ▼            ▼            ▼
┌──────────────────────────────────────────────────────────────┐
│                    Locomotion System                          │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐    │
│  │   Teleport    │  │    Smooth     │  │   Climbing    │    │
│  │  (arc + aim)  │  │ (thumbstick)  │  │ (grab + pull) │    │
│  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘    │
│          │                  │                  │             │
│          └────────────┬─────┴──────────────────┘             │
│                       │                                       │
│                       ▼                                       │
│          ┌────────────────────────┐                          │
│          │   Comfort Manager      │                          │
│          │   (vignette, presets)  │                          │
│          └────────────────────────┘                          │
└──────────────────────────────────────────────────────────────┘
        │                       │
        ▼                       ▼
┌──────────────┐        ┌──────────────┐
│  Character   │        │   Renderer   │
│  Controller  │        │  (vignette)  │
└──────────────┘        └──────────────┘
```

## Performance Requirements

| Component | Update Rate | CPU Budget |
|-----------|-------------|------------|
| Input Processing | 90 Hz | <0.1ms |
| Arc Calculation | 90 Hz | <0.2ms |
| Movement Calculation | 90 Hz | <0.1ms |
| Climbing State | 90 Hz | <0.1ms |
| Comfort Vignette | 90 Hz | <0.05ms |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Motion sickness complaints | High | High | Default to comfortable preset, clear warnings |
| Teleport through walls | Medium | Medium | Collision validation before execute |
| Climbing stamina confusion | Medium | Low | Clear stamina indicator, audio feedback |
| Vignette too aggressive | Medium | Medium | Per-user intensity configuration |
