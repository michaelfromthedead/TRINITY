# XR Locomotion System Investigation

**Path:** `/home/user/dev/USER/PROJECTS_VOID/TRINITY/engine/xr/locomotion/`
**Date:** 2026-05-22
**Total Lines:** 3,310

## File Inventory

| File | Lines | Classification | Status |
|------|-------|----------------|--------|
| `__init__.py` | 371 | REAL | Complete module aggregator with factory functions |
| `comfort.py` | 789 | REAL | Full implementation with algorithms |
| `climbing.py` | 772 | REAL | Full implementation with state machine |
| `teleport.py` | 723 | REAL | Full implementation with physics |
| `smooth.py` | 655 | REAL | Full implementation with input processing |

**Overall Classification: REAL** - All files contain working algorithms, state machines, and computational logic. No stubs or NotImplementedError patterns found.

## Module Architecture

### `__init__.py` (371 lines)

**Purpose:** Module aggregator and public API surface.

**Key Components:**
- `LocomotionProvider` - Abstract base class integrating teleport, smooth, climbing, and comfort systems
- Factory functions: `create_teleport_locomotion()`, `create_smooth_locomotion()`, `create_climbing_locomotion()`, `create_comfort_settings()`
- Complete re-exports from all submodules (68 symbols in `__all__`)

**Notable Pattern:** Provider class holds optional references to each locomotion type, allowing mixed locomotion strategies.

---

### `teleport.py` (723 lines)

**Purpose:** Teleportation-based movement with arc visualization.

**Enums:**
- `TeleportStyle`: INSTANT, FADE, DASH, BLINK
- `TeleportState`: IDLE, AIMING, VALIDATING, TRANSITIONING, COOLDOWN
- `ArcSegmentType`: VALID, INVALID, OUT_OF_RANGE

**Key Classes:**

1. **`TeleportArcCalculator`** - Parabolic trajectory physics
   - `calculate_arc()` - Projectile motion with configurable gravity/velocity
   - `find_landing_point()` - Ground intersection via linear interpolation
   - Uses quadratic formula for flight time calculation

2. **`TeleportLocomotion`** - Main component (dataclass)
   - State machine: IDLE -> AIMING -> TRANSITIONING -> COOLDOWN
   - Configurable: max_distance (1-50m), arc_gravity (-9.8), fade_duration (0-0.5s)
   - `begin_aim()`, `update_aim()`, `execute_teleport()`, `cancel_aim()`
   - Snap rotation support with configurable angle

3. **`TeleportTarget`** - Destination marker component
   - Surface normal, landing offset, rotation hint
   - Area bounds support for region-based targets

4. **`TeleportLocomotionProvider`** - Runtime integration interface

**Decorator:** `@xr_teleport_area` - Marks valid teleport destinations

**Algorithm Quality:** Real physics implementation with proper interpolation and ground detection.

---

### `smooth.py` (655 lines)

**Purpose:** Continuous movement with comfort features.

**Enums:**
- `MovementMode`: THUMBSTICK, ARM_SWING, HEAD_DIRECTED, HAND_DIRECTED
- `TurnType`: SNAP, SMOOTH, DISABLED
- `MovementState`: IDLE, MOVING, TURNING, DISABLED
- `StrafeBehavior`: NORMAL, REDUCED, DISABLED

**Key Classes:**

1. **`TurnSettings`** - Turn behavior configuration
   - Snap turn: configurable angle (15-90 deg), cooldown
   - Smooth turn: degrees/second calculation
   - Dead zone handling

2. **`SmoothLocomotion`** - Main component
   - Input processing with dead zone and curve (power function)
   - Velocity calculation: forward, strafe, with sprint multiplier
   - Direction reference: head, left_hand, right_hand
   - Gravity support when not grounded
   - Vignette intensity calculation from linear and angular velocity

3. **`SmoothLocomotionProvider`** - Runtime integration

**Decorator:** `@xr_locomotion` - Configures locomotion type and speed

**Features:**
- `calculate_movement()` - Full velocity + rotation delta calculation
- `calculate_arm_swing_movement()` - Arm swing detection from hand velocities
- Backward speed reduction (0.7x default)
- Strafe behavior options

**Algorithm Quality:** Complete input processing pipeline with proper dead zones, curves, and vignette triggering.

---

### `climbing.py` (772 lines)

**Purpose:** Climbing-based locomotion for VR.

**Enums:**
- `ClimbingState`: IDLE, GRABBING, CLIMBING, MANTLING, FALLING
- `GrabHandState`: FREE, GRABBING, RELEASING
- `ClimbableType`: SURFACE, LADDER, ROPE, LEDGE, HOLDS
- `MantleType`: PULL_UP, VAULT, CLIMB_OVER

**Key Classes:**

1. **`GrabPoint`** - Discrete grab location
   - Position, normal, grip_type, radius, availability

2. **`ClimbableVolume`** - Climbable surface marker
   - Bounds checking: `is_point_inside()`
   - Nearest grab point search with distance calculation
   - Grip strength requirements
   - Stamina drain rate configuration
   - Auto-generated grab points option

3. **`ClimbingLocomotion`** - Main component
   - Dual hand grab state tracking
   - `try_grab()` - Grip threshold, climbable detection, grab point selection
   - `release_grab()` - Hand release with state update
   - `calculate_climbing_movement()` - Inverse hand velocity for player movement
   - Stamina system: drain while climbing, recovery when grounded
   - Mantle support with progress tracking

4. **`ClimbingLocomotionProvider`** - Runtime integration

**Decorator:** `@xr_climbable` - Marks climbable surfaces

**Features:**
- Movement = inverse of hand movement while grabbing
- Average velocity when both hands grabbing
- Reduced gravity while climbing (-2.0 vs -9.8 falling)
- Haptic feedback flags
- Callback system: on_grab, on_release, on_stamina_empty, on_mantle_start/complete

**Algorithm Quality:** Complete state machine with proper physics and stamina mechanics.

---

### `comfort.py` (789 lines)

**Purpose:** Motion sickness mitigation features.

**Enums:**
- `ComfortLevel`: NONE, LOW, MEDIUM, HIGH, CUSTOM
- `VignetteShape`: CIRCULAR, ELLIPTICAL, RECTANGULAR
- `VignetteTrigger`: VELOCITY, ROTATION, BOTH, MANUAL
- `TunnelingMode`: DISABLED, MILD, MODERATE, STRONG
- `PlayMode`: STANDING, SEATED, ROOMSCALE

**Key Classes:**

1. **`ComfortVignette`** - Dynamic vignette effect
   - Velocity and angular velocity thresholds
   - Adaptive intensity scaling
   - Smooth fade in/out with configurable speeds
   - `get_shader_params()` for rendering
   - Shape, inner/outer radius, feather, color

2. **`XRComfortSettings`** - Global preferences resource
   - `apply_preset()` - Applies NONE/LOW/MEDIUM/HIGH configurations
   - Snap turn: enabled, angle (15-90 deg)
   - Vignette: enabled, intensity, on_turn, on_move flags
   - Teleport fade settings
   - Tunneling mode
   - Stable horizon (roll/pitch lock)
   - Seated mode with height offset
   - Movement speed scale
   - Metrics tracking: cumulative rotation/velocity, time in motion

3. **`ComfortPreset`** - Pre-configured preset definition
   - 5 built-in presets: veteran, intermediate, comfortable, maximum, seated

4. **`ComfortManager`** - System coordinator
   - `update()` - Updates vignette based on velocity/angular velocity
   - `apply_preset()` - Applies named preset

**Decorator:** `@xr_comfort` - Configures comfort features

**Presets:**
| Preset | Snap Turn | Vignette | Tunneling | Horizon | Speed Scale |
|--------|-----------|----------|-----------|---------|-------------|
| veteran | Off | Off | Disabled | Off | 1.0 |
| intermediate | On (45) | 0.4 | Mild | Off | 0.9 |
| comfortable | On (30) | 0.6 | Moderate | On | 0.7 |
| maximum | On (30) | 0.8 | Strong | On | 0.5 |
| seated | On (45) | 0.5 | Mild | On | 0.8 |

**Algorithm Quality:** Complete comfort system with proper vignette calculations and preset management.

---

## Cross-Cutting Patterns

### Type Annotations
All components use `Annotated` types with custom markers:
- `Tracked` - Property change tracking
- `Range(min, max)` - Value constraints
- `Observable` - Event emission
- `Transient` - Non-serialized state
- `Immutable` - Read-only after init

### Decorator Integration
Each module registers decorators with the central registry:
- `@xr_teleport_area`, `@xr_locomotion`, `@xr_climbable`, `@xr_comfort`
- All use `make_decorator()` from `trinity.decorators.ops`
- Steps: TAG -> REGISTER pattern

### Configuration
References `engine.xr.config.XR_CONFIG` for defaults:
- `TELEPORT_MAX_DISTANCE_M`
- `TELEPORT_ARC_GRAVITY`
- `TELEPORT_ARC_VELOCITY`
- `SNAP_TURN_ANGLE_DEGREES`
- `VIGNETTE_INTENSITY_DEFAULT`

### Provider Pattern
Each locomotion type has a Provider class:
- Takes locomotion component as constructor argument
- Provides runtime integration interface
- Delegates to component methods

---

## Code Quality Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| **Completeness** | Excellent | No stubs, all methods implemented |
| **Algorithm Quality** | Excellent | Real physics, proper state machines |
| **Type Safety** | Excellent | Full type annotations with constraints |
| **Documentation** | Good | Docstrings present, some methods sparse |
| **Test Coverage** | Unknown | No test files in this directory |
| **Error Handling** | Adequate | Boundary checks, but minimal exceptions |

### Strengths
1. Real physics calculations (projectile motion, gravity, velocity)
2. Comprehensive state machines with proper transitions
3. Flexible configuration via decorators and presets
4. Clean provider abstraction for runtime integration
5. Consistent annotation patterns across modules

### Areas for Improvement
1. No explicit unit tests visible in module
2. Climbable detection relies on external system (noted as "done externally")
3. Ground collision is external - `set_grounded()` must be called by physics system
4. No explicit error handling for invalid configurations

---

## Dependencies

**Internal:**
- `trinity.decorators.ops` - Decorator framework
- `trinity.decorators.registry` - Decorator registration
- `engine.xr.config` - XR configuration constants
- `engine.xr.utils.markers` - Type annotation markers

**External:**
- `math` (standard library)
- `dataclasses`
- `enum`
- `typing`

---

## Summary

The XR locomotion module is a **production-quality implementation** providing three locomotion modes (teleport, smooth, climbing) with comprehensive comfort features. All code is functional with real algorithms - no stubs detected. The architecture follows a consistent pattern of dataclass components, enum-based state machines, and provider interfaces for runtime integration. The comfort system is particularly well-developed with multiple presets and adaptive vignette effects.
