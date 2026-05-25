# Engine Debug Tools Investigation

**Date:** 2026-05-22
**Module:** `engine/debug/tools/`
**Total Lines:** 3,964

## Executive Summary

All 7 files in `engine/debug/tools/` are **REAL implementations** with complete, production-ready code. These are fully functional debug tools providing comprehensive runtime debugging capabilities for game development. Security considerations are properly handled with shipping build restrictions.

## Classification Summary

| File | Lines | Classification | Completeness |
|------|-------|----------------|--------------|
| `__init__.py` | 45 | REAL | 100% - Clean module exports |
| `cheats.py` | 786 | REAL | 100% - Full cheat system |
| `network_debug.py` | 726 | REAL | 100% - Network simulation |
| `debug_camera.py` | 659 | REAL | 100% - Multi-mode camera |
| `physics_debug.py` | 656 | REAL | 100% - Physics visualization |
| `ai_debug.py` | 618 | REAL | 100% - AI debugging |
| `time_control.py` | 474 | REAL | 100% - Time manipulation |

---

## Module: `__init__.py` (45 lines)

**Classification:** REAL - Clean public API

Exports all debug tools with proper `__all__` definition:
- `CheatManager`, `CheatCommand`, `CheatFlags`
- `TimeController`, `TimeState`
- `DebugCamera`, `DebugCameraMode`
- `AIDebugger`, `AIDebugState`, `PerceptionVisual`
- `PhysicsDebugger`, `PhysicsDebugState`, `BodyInspection`
- `NetworkDebugger`, `NetworkStats`, `NetworkSimulation`

---

## Module: `cheats.py` (786 lines)

**Classification:** REAL - Complete cheat management system

### Components

1. **CheatConfig** (dataclass)
   - Build-type restrictions (`allow_in_shipping`, `allow_in_debug`)
   - Multiplayer restrictions (`allow_in_multiplayer`)
   - Logging configuration

2. **Build Detection Functions**
   - `is_shipping_build()` - Checks `GAME_BUILD_TYPE`, `NDEBUG`, `SHIPPING` env vars
   - `is_debug_build()` - Checks `DEBUG`, `_DEBUG` env vars
   - `cheats_allowed()` - Master gate for cheat availability
   - `require_debug_build()` - Decorator for guarding functions

3. **CheatFlags** (Flag enum)
   - `REQUIRES_DEBUG_BUILD`, `REQUIRES_DEVELOPER`, `REQUIRES_ADMIN`
   - `PERSISTENT` (survives level changes)
   - `REPLICATED` (syncs across network)
   - `LOGGED`, `DISABLED_IN_MULTIPLAYER`

4. **CheatCommand** (dataclass)
   - Handler function binding
   - Parameter definitions with types
   - Alias support

5. **CheatState** (dataclass)
   - `god_mode`, `fly_mode`, `ghost_mode`, `invisible`
   - `infinite_ammo`, `speed_multiplier`, `damage_multiplier`

6. **CheatManager** (main class)
   - Command registration/unregistration
   - Per-entity state tracking
   - Callback system for cheat events

### Built-in Cheats
| Command | Aliases | Description |
|---------|---------|-------------|
| `god` | `godmode`, `invincible` | Toggle invulnerability |
| `fly` | `noclip`, `flymode` | Pass through geometry |
| `ghost` | `nocollision`, `ghostmode` | Disable collision |
| `teleport` | `tp`, `goto` | Move to X Y Z location |
| `spawn` | `create`, `summon` | Create entity at location |
| `kill` | `destroy`, `remove` | Destroy target entity |
| `sethealth` | `hp`, `health` | Set health value |
| `give` | `additem`, `giveitem` | Add item to inventory |
| `infiniteammo` | `unlimitedammo`, `ammo` | Toggle infinite ammo |

### Singleton Access
- `get_cheat_manager()` - Global instance
- `reset_cheat_manager()` - Testing support

---

## Module: `network_debug.py` (726 lines)

**Classification:** REAL - Complete network debugging system

### Components

1. **PacketDirection** (enum)
   - `INBOUND`, `OUTBOUND`

2. **PacketLog** (dataclass)
   - Timestamp, direction, size, channel, packet type
   - Source/destination addresses
   - Latency applied, dropped status, metadata

3. **NetworkStats** (dataclass)
   - Bandwidth metrics: `bytes_sent`, `bytes_received`, `bytes_per_second_*`
   - Packet counts: `packets_sent`, `packets_received`, `packets_dropped`
   - Latency: `current_latency_ms`, `average_latency_ms`, `min/max_latency_ms`, `jitter_ms`
   - RTT: `rtt_ms`, `rtt_average_ms`
   - Quality: `connection_quality` (0.0-1.0), `packet_loss_percent`
   - Simulation state: `simulated_latency_ms`, `simulated_packet_loss`, etc.

4. **NetworkDebugConfig** (dataclass)
   - Buffer limits: `max_recent_packets=10000`, `max_packet_log=1000`
   - Sample buffers: `max_latency_samples=100`, `max_rtt_samples=100`
   - Quality thresholds: latency, jitter, packet loss coefficients
   - Penalty weights for quality calculation

5. **NetworkSimulation** (dataclass)
   - `latency_ms`, `latency_variance_ms`
   - `packet_loss_percent`, `packet_loss_burst_chance`, `packet_loss_burst_length`
   - `jitter_ms`, `bandwidth_limit_kbps`
   - `duplicate_chance`, `reorder_chance`, `corruption_chance`

6. **NetworkDebugger** (main class)
   - Latency/jitter/bandwidth simulation
   - Packet loss with burst mode
   - Packet logging and filtering
   - Bandwidth throttling with sliding window
   - Connection quality scoring
   - Statistics collection and callbacks

### Console Commands
| Command | Description |
|---------|-------------|
| `net.latency <ms>` | Set simulated latency |
| `net.loss <percent>` | Set packet loss |
| `net.jitter <ms>` | Set jitter |
| `net.bandwidth <kbps>` | Set bandwidth limit |
| `net.stats` | Show network statistics |
| `net.reset` | Reset simulation and stats |

---

## Module: `debug_camera.py` (659 lines)

**Classification:** REAL - Multi-mode debug camera

### Components

1. **DebugCameraMode** (enum)
   - `FREE` - WASD movement, mouse look
   - `ORBIT` - Orbit around target point
   - `FOLLOW` - Follow entity with offset
   - `CYCLE` - Cycle through entity list

2. **CameraConfig** (dataclass)
   - Movement: `move_speed=10.0`, `fast_speed_multiplier=3.0`, `slow_speed_multiplier=0.2`
   - Rotation: `rotate_speed=0.3`, `orbit_speed=1.0`
   - Orbit: `min_orbit_distance=1.0`, `max_orbit_distance=100.0`, `default_orbit_distance=10.0`
   - Follow: `follow_distance=5.0`, `follow_height=2.0`, `follow_smoothing=5.0`
   - Pitch limits: `min_pitch=-89.0`, `max_pitch=89.0`
   - Smoothing: `position_smoothing=10.0`, `rotation_smoothing=15.0`

3. **CameraTransform** (dataclass)
   - Position tuple, rotation tuple (pitch, yaw, roll)

4. **DebugCamera** (main class)
   - Mode switching with initialization
   - Target entity tracking
   - Entity list cycling (next/previous)
   - Movement with direction and speed modifiers
   - Rotation with pitch clamping and yaw normalization
   - Zoom control (orbit mode)
   - View matrix computation
   - Teleport and look-at functions
   - Smooth interpolation with `_lerp` and `_lerp_angle`
   - Mode change callbacks

### Camera Math
- Forward/right/up vector calculation from pitch/yaw
- View matrix construction (4x4)
- Angle wrap-around handling for smooth rotation

---

## Module: `physics_debug.py` (656 lines)

**Classification:** REAL - Physics visualization and control

### Components

1. **PhysicsDebugState** (enum)
   - `RUNNING`, `PAUSED`, `STEPPING`

2. **PhysicsVisualization** (Flag enum)
   - `COLLISION_SHAPES`, `CONTACT_POINTS`, `VELOCITIES`
   - `ANGULAR_VELOCITIES`, `CONSTRAINTS`, `JOINTS`
   - `RAYCASTS`, `CENTER_OF_MASS`, `BOUNDING_BOXES`
   - `SLEEP_STATE`, `FORCES`, `ALL`

3. **PhysicsVisualizationConfig** (dataclass)
   - Collision: color (green), wireframe toggle
   - Contact: color (red), normal length
   - Velocity: color (blue), scale factor
   - Angular velocity: color (magenta)
   - Constraints: color (yellow)
   - Center of mass: size
   - Sleep state: sleeping/active colors
   - Buffer limits: `max_contacts=1000`, `max_raycast_history=100`

4. **ContactPoint** (dataclass)
   - Position, normal, penetration depth
   - Body A/B references, impulse

5. **BodyInspection** (dataclass)
   - Entity reference, body type
   - Mass, inertia tensor
   - Position, rotation (quaternion)
   - Linear/angular velocity and damping
   - Friction, restitution
   - Sleep state, sensor flag
   - Collision group/mask
   - Shape/constraint/contact counts

6. **PhysicsDebugger** (main class)
   - Per-visualization toggle functions
   - Per-entity visualization selection
   - Physics pause/resume/step
   - Force application with visualization
   - Contact point recording
   - Raycast history tracking
   - Body inspection with attribute extraction
   - State change callbacks

### Console Commands
| Command | Description |
|---------|-------------|
| `physics.pause` | Pause physics |
| `physics.resume` | Resume physics |
| `physics.step [count]` | Step physics frames |
| `show.collision [0/1]` | Toggle collision shapes |
| `show.velocities [0/1]` | Toggle velocity vectors |

---

## Module: `ai_debug.py` (618 lines)

**Classification:** REAL - AI debugging with behavior tree support

### Components

1. **AIDebugState** (enum)
   - `RUNNING`, `PAUSED`, `STEPPING`

2. **PerceptionType** (enum)
   - `SIGHT`, `HEARING`, `TOUCH`, `SMELL`, `CUSTOM`

3. **AIDebugConfig** (dataclass)
   - Default ranges: `default_sight_range=20.0`, `default_hearing_range=15.0`
   - Default angles: `default_sight_angle=90.0`, `default_hearing_angle=360.0`
   - Colors: sight (yellow), hearing (cyan), custom (orange)

4. **PerceptionVisual** (dataclass)
   - Perception type, RGBA color
   - Range, angle
   - Show flags: range, direction, detected

5. **BTNodeDebugInfo** (dataclass)
   - Name, node type, status
   - Active flag, depth level
   - Execution count, last result
   - Recursive children list

6. **BlackboardDebugInfo** (dataclass)
   - Key, value, namespace
   - Value type, timestamp, TTL

7. **AIDebugger** (main class)
   - **Perception debugging:**
     - Per-entity perception visualization
     - Show all perception toggle
     - Custom perception config per entity
   - **Behavior tree debugging:**
     - Per-entity BT visualization
     - Recursive node info extraction
     - Handles composite/decorator nodes
   - **Blackboard debugging:**
     - Per-entity blackboard display
     - Namespace-aware key parsing
     - TTL support
   - **AI control:**
     - Pause/resume/step
     - State override per entity
     - Tick gating via `should_tick()`

### Console Commands
| Command | Description |
|---------|-------------|
| `ai.debug [entity]` | Show AI debug help |
| `ai.pause` | Pause all AI |
| `ai.resume` | Resume AI |
| `ai.step` | Step one AI tick |

---

## Module: `time_control.py` (474 lines)

**Classification:** REAL - Complete time manipulation system

### Components

1. **TimeState** (enum)
   - `NORMAL`, `PAUSED`, `SLOW_MOTION`, `FAST_FORWARD`, `FRAME_STEP`

2. **TimeControlConfig** (dataclass)
   - Limits: `min_time_scale=0.01`, `max_time_scale=10.0`
   - Defaults: `default_slow_motion=0.25`, `default_fast_forward=2.0`
   - Presets: `preset_super_slow=0.1`, `preset_slow=0.25`, `preset_half=0.5`
   - `preset_normal=1.0`, `preset_fast=2.0`, `preset_super_fast=4.0`

3. **TimeController** (main class)
   - Pause/resume with duration tracking
   - Time scale with clamping
   - Frame stepping while paused
   - Slow motion / fast forward shortcuts
   - Preset application
   - Automatic state derivation from scale
   - Callback system: pause, scale, state

### Preset Properties
| Property | Value | Description |
|----------|-------|-------------|
| `PRESET_SUPER_SLOW` | 0.1 | 10% speed |
| `PRESET_SLOW` | 0.25 | 25% speed |
| `PRESET_HALF` | 0.5 | 50% speed |
| `PRESET_NORMAL` | 1.0 | Real-time |
| `PRESET_FAST` | 2.0 | 200% speed |
| `PRESET_SUPER_FAST` | 4.0 | 400% speed |

### Console Commands
| Command | Description |
|---------|-------------|
| `pause` | Pause game |
| `resume` | Resume game |
| `slomo <scale>` | Set time scale |
| `step [count]` | Step frames while paused |

### Helper Function
- `register_time_commands(controller, console)` - Auto-register commands

---

## Security Architecture

All debug tools implement consistent security patterns:

1. **Build Type Detection**
   - Environment variables: `GAME_BUILD_TYPE`, `SHIPPING`, `NDEBUG`, `DEBUG`
   - Integration with `engine.tooling.build.build_config` when available

2. **Configuration-Based Restrictions**
   - Each tool has `allow_in_shipping: bool = False` in its config
   - Multiplayer restrictions for cheats

3. **Runtime Guards**
   - `_check_build_allowed()` method in each debugger
   - `enabled` setter refuses enable in disallowed builds
   - `require_debug_build()` decorator for function-level protection

4. **Graceful Degradation**
   - Tools log warnings but don't crash when disabled
   - Return safe defaults when blocked

---

## Integration Points

### Console System
All tools provide `cmd_*` methods ready for console registration.

### Callback System
All tools implement observer pattern:
- `add_*_callback(callback)` - Register callback
- `remove_*_callback(callback)` - Unregister callback
- `_notify_*_callbacks(...)` - Internal notification

### Singleton Pattern
All tools provide:
- `get_*()` - Global instance accessor
- Optional `reset_*()` - For testing

### Type Hints
- Full TYPE_CHECKING imports for engine types
- Proper Optional/Tuple/List/Dict usage
- No runtime type checking overhead

---

## Quality Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| Completeness | 10/10 | All features fully implemented |
| Documentation | 9/10 | Comprehensive docstrings |
| Type Safety | 9/10 | Full type hints throughout |
| Security | 10/10 | Proper shipping build guards |
| Configurability | 10/10 | All magic numbers in config dataclasses |
| Testability | 9/10 | Singleton reset for testing |
| Code Quality | 9/10 | Clean separation of concerns |

---

## Recommendations

1. **Integration Testing**: These tools would benefit from integration tests with mock game systems.

2. **Visual Rendering**: The visualization components (physics shapes, perception cones) define data structures but actual rendering would be handled elsewhere. Ensure renderer integration exists.

3. **Performance Profiling**: For the network debugger's bandwidth calculations and physics contact tracking, verify buffer sizes are appropriate for target platforms.

4. **Documentation**: Consider adding usage examples for console command integration.
