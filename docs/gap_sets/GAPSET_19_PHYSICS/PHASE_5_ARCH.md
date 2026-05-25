# Phase 5: Specialized -- Architecture

## Status: 5 [x] 0 [~] 0 [-]

## Module: `engine/simulation/`

### Overview
Phase 5 provides specialized physics systems: vehicles (wheeled, aircraft, watercraft), ragdoll physics, and character controllers. All 5 tasks are fully implemented as Python reference code.

---

### T-PHY-5.1: Wheeled Vehicle Physics

**Status**: [x] Complete.
**Location**: `engine/simulation/vehicles/` (shared with T-PHY-5.2/5.3)

**Current Implementation**:
- `wheeled_vehicle.py` (762 lines): Raycast wheel model, suspension spring/damper
- `tire_model.py` (831 lines): Pacejka tire model (Magic Formula)
- `suspension.py` (546 lines): Independent/dependent suspension geometries
- `drivetrain.py` (982 lines): Engine torque curve, clutch, gearbox ratios, differential
- `vehicle_system.py` (685 lines): Vehicle system orchestrator
- 4 drive layouts: FWD, RWD, AWD, 4WD
- Steering geometry: Ackermann
- Per-wheel brake torque
- `vehicle_component.py` (530 lines): ECS component wrapper

**Gap**: No Rust backend. Uses float math (Pacejka formula is float-only).

### T-PHY-5.2: Aircraft Physics

**Status**: [x] Complete.
**Location**: `engine/simulation/vehicles/aircraft.py` (757 lines)

**Current Implementation**:
- `AircraftSimulation`: Four forces model (lift, drag, thrust, weight)
- Lift/drag coefficient curves (angle of attack lookup)
- Control surfaces: aileron, elevator, rudder, flaps
- Stall model (critical angle of attack detection)
- Landing gear with compression

### T-PHY-5.3: Watercraft Physics

**Status**: [x] Complete.
**Location**: `engine/simulation/vehicles/watercraft.py` (664 lines)

**Current Implementation**:
- `WatercraftSimulation`: Buoyancy evaluation at sample points (Archimedes principle)
- Hydrodynamic drag (velocity-dependent)
- Propeller thrust model
- Sail force model (wind-dependent)
- Water jet propulsion option

### T-PHY-5.4: Ragdoll Physics

**Status**: [x] Complete.
**Location**: `engine/simulation/character/ragdoll.py` (776 lines), `active_ragdoll.py` (678 lines)

**Current Implementation**:
- `Ragdoll`: Passive ragdoll with joint limits + motor constraints
- `ActiveRagdoll`: PD controller per joint tracking target pose
- Balance controller: center-of-mass tracking with feedback
- Transition from animation to ragdoll (blend-in)
- Transition from ragdoll to animation (get-up via target pose blending)

### T-PHY-5.5: Character Controllers

**Status**: [x] Complete.
**Location**: `engine/simulation/character/character_controller.py` (879 lines)

**Current Implementation**:
- `CharacterController`: Kinematic, dynamic, and hybrid controller modes
- Capsule-based simplified collision + movement
- `ground_detection.py` (618 lines): Swept check ground detection, ground state management
- `slope_handling.py` (651 lines): Slope limits, stair navigation
- `platform_handling.py` (553 lines): Moving platform attachment with relative motion
- `movement_modes.py` (692 lines): Ground, air, swim, climb movement modes
- `character_interaction.py` (896 lines): Push, pull, carry interactions
- `physics_animation_blend.py` (693 lines): Transition blending between physics and animation
- `character_component.py` (542 lines): ECS component wrapper

---

## Key Design Decisions

- **Vehicle physics as standalone systems**: Unlike core physics where components interact through the solver, vehicle systems are somewhat self-contained. Wheel forces are computed from tire models and applied as external forces to the chassis rigid body. This enables efficient LOD (simple force at distance, full drivetrain up close).
- **Pacejka tire model**: The Pacejka "Magic Formula" is the industry standard for tire simulation. The Python implementation provides the full mathematical model with configurable parameters for different tire types.
- **Ragdoll duality**: Both passive (joint-limited) and active (PD-controlled) ragdolls are implemented. Active ragdolls enable gameplay-relevant features like getting up from ragdoll state and partial activation.
- **Character controller modes**: Three modes (kinematic, dynamic, hybrid) provide the full spectrum from "push rigid bodies" to "full physics simulation" for character movement, matching game engine standards.
- **Comprehensive vehicle coverage**: 5 vehicle types (wheeled, aircraft, watercraft, tracked, hover) plus detailed drivetrain and suspension models provide simulation-grade vehicle physics suitable for both games and technical simulations.
