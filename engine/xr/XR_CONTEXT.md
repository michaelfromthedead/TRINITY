# XR Layer Implementation Context

Complete context for implementing `engine/xr/` using the Trinity Pattern + Foundation runtime.

**Implementation Status:**
| Layer | Status | Notes |
|-------|--------|-------|
| Python | ✅ COMPLETE | VR/AR, hand tracking, haptics |
| Rust | ✅ 80% | GAPSET_18_UI_XR — Core XR wired |
| Wired | ✅ Yes | XR runtime functional |

*See `docs/STATUS.md` for current progress.*

---

## Table of Contents

- [Overview](#overview)
- [Part I: Trinity Recommendations for XR](#part-i-trinity-recommendations-for-xr)
  - [Metaclasses](#metaclasses)
  - [Descriptors](#descriptors)
  - [Decorators](#decorators)
- [Part II: Decorator Stacks for XR](#part-ii-decorator-stacks-for-xr)
- [Part III: XR-Specific Patterns](#part-iii-xr-specific-patterns)
- [Part IV: Engine/XR Directory Structure](#part-iv-enginexr-directory-structure)
- [Part V: Implementation Checklist](#part-v-implementation-checklist)

---

## Overview

The XR Layer (`engine/xr/`) provides comprehensive support for virtual reality (VR), augmented reality (AR), and mixed reality (MR) experiences. It abstracts hardware differences across platforms while enabling immersive, comfortable, and interactive XR applications.

### Architecture Reference

From `DIAGRAMS/ARCHITECTURE_XR.md`:

```
XR LAYER
+-- XR Architecture (VR/AR/MR modes, runtimes, abstraction layer)
+-- Head-Mounted Display (head tracking, display system, optics, audio)
+-- Controller Input (motion controllers, hand tracking, haptics)
+-- XR Rendering (stereo, foveated, reprojection, compositor)
+-- Eye Tracking (gaze, pupil, fixation, calibration)
+-- Spatial Understanding (mapping, planes, anchors, scene understanding)
+-- XR Interaction (direct, indirect, grab, locomotion, UI)
+-- AR Features (image tracking, light estimation, passthrough)
+-- XR Comfort (motion sickness mitigation, ergonomics, settings)
+-- XR Avatars (body IK, hand animation, face tracking, social)
+-- XR Platform Integration (devices, services, guardian)
```

### Core Principles

| Principle | Description |
|-----------|-------------|
| **Presence** | Feel like you're there |
| **Comfort** | Minimize motion sickness |
| **Immersion** | Believable world |
| **Interaction** | Natural input |

### Integration Point

From `GAME_ENGINE_INTEGRATION.md`:

> XR systems use `TrackedDescriptor` for pose data with change detection. Controller input uses `@input_action` and `@input_axis` for binding. Hand tracking uses `InterpolatedDescriptor` for smooth joint positions. Avatar IK uses `ComputedDescriptor` for derived bone transforms. Multiplayer XR uses `@networked` with spatial relevance for avatar sync. All XR data flows through Foundation's Registry for device lookup and Tracker for pose updates.

---

## Part I: Trinity Recommendations for XR

### Metaclasses

**Use `ComponentMeta` for XR entities.** HMD, controllers, hands, tracked objects, and anchors are ECS components. This gives you:
- Unique IDs for fast lookup
- Field processing with type annotations
- Automatic descriptor installation
- Foundation Registry integration

**Use `ResourceMeta` for XR global state.** XR runtime state, comfort settings, and platform capabilities are singleton resources:
- Single instance per XR session
- Global access pattern
- Automatic lifecycle

**Use `SystemMeta` for XR systems.** Tracking system, rendering system, interaction system are ECS systems:
- Update order specification
- Query-based entity access
- Dependency injection

**Use `StateMeta` for XR state machines.** Controller state (idle/tracking/lost), interaction state (grab/release), locomotion state:
- Valid transition validation
- Enter/exit hooks
- State history for rollback

**Use `EventMeta` for XR events.** Controller button presses, hand gestures, spatial anchor changes:
- Type-safe event payloads
- Automatic EventLog integration
- Replay support

### Descriptors

#### Core Descriptors for XR

| Descriptor | XR Purpose | Example |
|-----------|-----------|---------|
| `TrackedDescriptor` | Pose change detection | HMD/controller position triggers render update |
| `InterpolatedDescriptor` | Smooth pose data | Hand joint positions interpolated between frames |
| `PredictedDescriptor` | Pose prediction | HMD pose predicted for next frame |
| `ValidatedDescriptor` | Input validation | Button values clamped 0-1, angles clamped |
| `RangeDescriptor` | Numeric constraints | IPD range, comfort settings limits |
| `ComputedDescriptor` | Derived values | Avatar bone transforms from IK |
| `TransientDescriptor` | Non-serializable state | Cached render data, haptic state |
| `ObservableDescriptor` | Change notifications | Tracking state -> UI updates |
| `NetworkedDescriptor` | Multiplayer sync | Avatar pose replication |
| `ImmutableDescriptor` | Read-only config | Device capabilities, display specs |
| `TransformDescriptor` | Value transformation | Coordinate space conversions |
| `BatchedDescriptor` | Bulk updates | Hand joint batch updates |
| `AtomicDescriptor` | Thread-safe updates | Pose data from tracking thread |
| `ExpiringDescriptor` | Timeout values | Tracking confidence decay |

#### Descriptor Chains for XR Fields

**HMD Pose (tracked + predicted):**
```
PredictedDescriptor -> TrackedDescriptor -> AtomicDescriptor -> StorageDescriptor
```
Pose is predicted, tracked for dirty detection, atomic for thread safety.

**Controller Button (validated + tracked):**
```
TrackedDescriptor -> ValidatedDescriptor -> RangeDescriptor -> StorageDescriptor
```
Button value tracked, validated, clamped to 0-1 range.

**Hand Joint Position (interpolated + batched):**
```
InterpolatedDescriptor -> BatchedDescriptor -> TrackedDescriptor -> StorageDescriptor
```
Joints interpolated smoothly, batched for performance, tracked for updates.

**Avatar Bone Transform (computed):**
```
ComputedDescriptor
```
Read-only, computed from IK solver each frame.

**Multiplayer Avatar Pose (networked + interpolated):**
```
NetworkedDescriptor -> InterpolatedDescriptor -> TrackedDescriptor -> StorageDescriptor
```
Network-replicated, interpolated for smoothness, tracked for dirty detection.

**Spatial Anchor (immutable config + tracked state):**
```
TrackedDescriptor -> StorageDescriptor  // For anchor state
ImmutableDescriptor -> StorageDescriptor  // For anchor ID
```

**Tracking Confidence (expiring):**
```
ExpiringDescriptor -> TrackedDescriptor -> RangeDescriptor -> StorageDescriptor
```
Confidence decays over time, tracked, clamped 0-1.

#### Annotated Field Syntax (Preferred)

```python
from typing import Annotated
from trinity.descriptors import (
    Tracked, Predicted, Interpolated, Validated, Range, Computed,
    Transient, Observable, Networked, Immutable, Transform, Batched, Atomic, Expiring
)

@component
class HMDPose:
    """Head-mounted display pose data."""
    
    # Position (predicted + tracked + atomic for thread safety)
    position: Annotated[tuple[float, float, float], Predicted, Tracked, Atomic] = (0.0, 0.0, 0.0)
    
    # Orientation as quaternion (predicted + tracked)
    orientation: Annotated[tuple[float, float, float, float], Predicted, Tracked, Atomic] = (0.0, 0.0, 0.0, 1.0)
    
    # Velocity for prediction
    linear_velocity: Annotated[tuple[float, float, float], Tracked, Atomic] = (0.0, 0.0, 0.0)
    angular_velocity: Annotated[tuple[float, float, float], Tracked, Atomic] = (0.0, 0.0, 0.0)
    
    # Tracking state
    tracking_state: Annotated[str, Tracked, Observable] = "unknown"  # unknown, tracking, lost
    confidence: Annotated[float, Tracked, Range(0, 1), Expiring(ttl=0.5)] = 0.0
    
    # Transient render data
    _view_matrix_left: Annotated[Any, Transient] = None
    _view_matrix_right: Annotated[Any, Transient] = None

@component
class ControllerState:
    """XR controller input state."""
    
    # Identity
    hand: Annotated[str, Immutable] = "left"  # left, right
    
    # Pose (predicted for low latency)
    grip_position: Annotated[tuple, Predicted, Tracked] = (0.0, 0.0, 0.0)
    grip_orientation: Annotated[tuple, Predicted, Tracked] = (0.0, 0.0, 0.0, 1.0)
    aim_position: Annotated[tuple, Predicted, Tracked] = (0.0, 0.0, 0.0)
    aim_orientation: Annotated[tuple, Predicted, Tracked] = (0.0, 0.0, 0.0, 1.0)
    
    # Buttons (tracked + validated + range clamped)
    trigger: Annotated[float, Tracked, Validated, Range(0, 1)] = 0.0
    grip: Annotated[float, Tracked, Validated, Range(0, 1)] = 0.0
    primary_button: Annotated[bool, Tracked] = False
    secondary_button: Annotated[bool, Tracked] = False
    thumbstick: Annotated[tuple[float, float], Tracked, Validated] = (0.0, 0.0)
    thumbstick_click: Annotated[bool, Tracked] = False
    
    # Tracking
    is_tracked: Annotated[bool, Tracked, Observable] = False
    
    # Haptics (transient, not replicated)
    _haptic_amplitude: Annotated[float, Transient] = 0.0
    _haptic_duration: Annotated[float, Transient] = 0.0

@component
class HandTrackingData:
    """Hand tracking joint data."""
    
    hand: Annotated[str, Immutable] = "left"
    
    # Joint positions (26 joints, batched + interpolated)
    joint_positions: Annotated[list[tuple], Batched, Interpolated, Tracked] = field(default_factory=list)
    joint_orientations: Annotated[list[tuple], Batched, Interpolated, Tracked] = field(default_factory=list)
    joint_radii: Annotated[list[float], Batched, Tracked] = field(default_factory=list)
    
    # Tracking state
    is_tracked: Annotated[bool, Tracked, Observable] = False
    confidence: Annotated[float, Tracked, Range(0, 1)] = 0.0
    
    # Gesture recognition
    current_gesture: Annotated[str, Tracked, Observable] = "none"  # pinch, point, fist, open, etc.
    gesture_confidence: Annotated[float, Tracked, Range(0, 1)] = 0.0
    
    # Pinch state
    pinch_strength: Annotated[float, Tracked, Range(0, 1)] = 0.0
    pinch_position: Annotated[tuple, Tracked] = (0.0, 0.0, 0.0)

@component
class EyeTrackingData:
    """Eye tracking gaze data."""
    
    # Combined gaze (both eyes)
    gaze_origin: Annotated[tuple, Tracked] = (0.0, 0.0, 0.0)
    gaze_direction: Annotated[tuple, Tracked] = (0.0, 0.0, -1.0)
    
    # Per-eye data
    left_pupil_position: Annotated[tuple, Tracked] = (0.0, 0.0)
    right_pupil_position: Annotated[tuple, Tracked] = (0.0, 0.0)
    left_pupil_diameter: Annotated[float, Tracked] = 0.0
    right_pupil_diameter: Annotated[float, Tracked] = 0.0
    left_openness: Annotated[float, Tracked, Range(0, 1)] = 1.0
    right_openness: Annotated[float, Tracked, Range(0, 1)] = 1.0
    
    # Fixation/saccade detection
    is_fixating: Annotated[bool, Tracked] = False
    fixation_point: Annotated[tuple, Tracked] = (0.0, 0.0, 0.0)
    
    # Tracking state
    is_calibrated: Annotated[bool, Tracked] = False
    confidence: Annotated[float, Tracked, Range(0, 1)] = 0.0

@resource
class XRRuntimeState:
    """Global XR runtime state."""
    
    # Runtime info (immutable after init)
    runtime_name: Annotated[str, Immutable] = ""  # OpenXR, SteamVR, etc.
    runtime_version: Annotated[str, Immutable] = ""
    
    # Session state (tracked for UI updates)
    session_state: Annotated[str, Tracked, Observable] = "idle"  # idle, ready, running, stopping
    
    # Display specs (immutable)
    display_refresh_rate: Annotated[float, Immutable] = 90.0
    display_resolution: Annotated[tuple[int, int], Immutable] = (1920, 1080)
    field_of_view: Annotated[tuple[float, float], Immutable] = (90.0, 90.0)
    
    # Current settings
    render_scale: Annotated[float, Tracked, Range(0.5, 2.0)] = 1.0
    foveated_level: Annotated[int, Tracked, Range(0, 4)] = 0
    
    # Capabilities (immutable flags)
    supports_hand_tracking: Annotated[bool, Immutable] = False
    supports_eye_tracking: Annotated[bool, Immutable] = False
    supports_passthrough: Annotated[bool, Immutable] = False

@resource
class XRComfortSettings:
    """User comfort preferences."""
    
    # Locomotion settings
    snap_turn_enabled: Annotated[bool, Tracked] = True
    snap_turn_angle: Annotated[float, Tracked, Range(15, 90)] = 45.0
    smooth_turn_speed: Annotated[float, Tracked, Range(30, 180)] = 90.0
    
    # Vignette settings
    vignette_enabled: Annotated[bool, Tracked] = True
    vignette_intensity: Annotated[float, Tracked, Range(0, 1)] = 0.5
    
    # Teleport settings
    teleport_fade_enabled: Annotated[bool, Tracked] = True
    teleport_fade_duration: Annotated[float, Tracked, Range(0, 0.5)] = 0.1
    
    # Seated mode
    seated_mode: Annotated[bool, Tracked] = False
    seated_height_offset: Annotated[float, Tracked] = 0.0
```

### Decorators

#### Existing Decorators for XR

From `trinity/decorators/input.py`:

| Decorator | Purpose | Parameters | Steps |
|-----------|---------|------------|-------|
| `@input_action` | Register input action | `name`, `default_bindings` | `TAG(input_action), TAG(action_name), TAG(action_bindings), REGISTER(input)` |
| `@input_axis` | Register input axis | `name`, `positive`, `negative` | `TAG(input_axis), TAG(axis_name), TAG(axis_positive), TAG(axis_negative), REGISTER(input)` |

From `trinity/decorators/spatial.py`:

| Decorator | Purpose | Parameters | Steps |
|-----------|---------|------------|-------|
| `@spatial` | Spatial indexing | `structure`, `cell_size` | `TAG(spatial), TAG(spatial_structure), TAG(spatial_cell_size), REGISTER(spatial)` |
| `@partitioned` | Spatial partitioning | `dimensions`, `max_entities` | `TAG(partitioned), TAG(partition_dimensions), TAG(max_entities), REGISTER(spatial)` |

From `trinity/decorators/animation.py`:

| Decorator | Purpose | Parameters | Steps |
|-----------|---------|------------|-------|
| `@tween` | Tween animation | `property`, `duration`, `easing` | `TAG(tween), TAG(tween_property), TAG(tween_duration), TAG(tween_easing), REGISTER(animation)` |
| `@blend_tree` | Animation blend | `parameter`, `clips` | `TAG(blend_tree), TAG(blend_parameter), TAG(blend_clips), REGISTER(animation)` |

From `trinity/decorators/data_flow.py`:

| Decorator | Purpose | Parameters | Steps |
|-----------|---------|------------|-------|
| `@serializable` | Serialization | `format`, `version` | `TAG(serializable), TAG(serializable_config), REGISTER(data_flow)` |
| `@networked` | Network replication | `relevance`, `authority`, `priority`, `interpolated`, `predicted` | `TAG(networked), TAG(networked_config), REGISTER(data_flow)` |
| `@snapshot` | State history | `history_frames` | `TAG(snapshot), TAG(snapshot_config), REGISTER(data_flow)` |

From `trinity/decorators/lifecycle.py`:

| Decorator | Purpose | Steps |
|-----------|---------|-------|
| `@on_add` | Component added hook | `TAG(on_add), HOOK(on_add), REGISTER(lifecycle)` |
| `@on_remove` | Component removed hook | `TAG(on_remove), HOOK(on_remove), REGISTER(lifecycle)` |
| `@on_change` | Component changed hook | `TAG(on_change), HOOK(on_change), REGISTER(lifecycle)` |

#### New XR Decorators to Create

| Decorator | Purpose | Steps |
|-----------|---------|-------|
| `@xr_tracked` | Mark as XR tracked object | `TAG(xr_tracked), TAG(tracking_type), TAG(tracking_space), REGISTER(xr)` |
| `@xr_controller` | Controller configuration | `TAG(xr_controller), TAG(hand), TAG(controller_type), REGISTER(xr)` |
| `@xr_hand` | Hand tracking config | `TAG(xr_hand), TAG(hand), TAG(gesture_recognition), REGISTER(xr)` |
| `@xr_interactable` | Interactable object | `TAG(xr_interactable), TAG(interaction_types), TAG(grab_points), REGISTER(xr)` |
| `@xr_grabbable` | Grabbable object | `TAG(xr_grabbable), TAG(grab_type), TAG(attach_transform), REGISTER(xr)` |
| `@xr_socket` | Snap socket | `TAG(xr_socket), TAG(socket_type), TAG(accepted_tags), REGISTER(xr)` |
| `@xr_teleport_area` | Teleport target | `TAG(xr_teleport_area), TAG(teleport_type), REGISTER(xr)` |
| `@xr_locomotion` | Locomotion config | `TAG(xr_locomotion), TAG(locomotion_type), TAG(speed), REGISTER(xr)` |
| `@xr_haptic` | Haptic feedback | `TAG(xr_haptic), TAG(amplitude), TAG(duration), TAG(frequency), REGISTER(xr)` |
| `@spatial_anchor` | AR anchor | `TAG(spatial_anchor), TAG(anchor_type), TAG(persistent), REGISTER(xr)` |
| `@ar_trackable` | AR tracked image/object | `TAG(ar_trackable), TAG(trackable_type), TAG(reference_id), REGISTER(xr)` |
| `@passthrough_layer` | Passthrough config | `TAG(passthrough_layer), TAG(blend_mode), TAG(opacity), REGISTER(xr)` |
| `@xr_avatar` | Avatar configuration | `TAG(xr_avatar), TAG(ik_enabled), TAG(network_sync), REGISTER(xr)` |
| `@xr_ik_target` | IK target point | `TAG(xr_ik_target), TAG(target_type), TAG(bone_chain), REGISTER(xr)` |
| `@foveated_region` | Foveation config | `TAG(foveated_region), TAG(region_type), TAG(quality_level), REGISTER(xr)` |
| `@xr_ui_panel` | XR UI configuration | `TAG(xr_ui_panel), TAG(panel_type), TAG(interaction_mode), REGISTER(xr)` |
| `@xr_comfort` | Comfort feature | `TAG(xr_comfort), TAG(comfort_type), TAG(settings), REGISTER(xr)` |

---

## Part II: Decorator Stacks for XR

### Recommended New Stacks

Create in `trinity/decorators/builtin_stacks/xr.py`:

```python
@parameterized_stack
def tracked_xr_device(
    tracking_type: str = "6dof",
    prediction_enabled: bool = True,
    pool_size: int = 8,
) -> Stack:
    """XR tracked device (HMD, controller)."""
    return stack(
        xr_tracked(tracking_type=tracking_type, tracking_space="stage"),
        track_changes,
        pooled(initial_size=pool_size),
        component,
    )

@parameterized_stack
def xr_controller_component(
    hand: str = "left",
    haptics_enabled: bool = True,
) -> Stack:
    """XR motion controller."""
    return stack(
        xr_controller(hand=hand, controller_type="motion"),
        xr_tracked(tracking_type="6dof", tracking_space="stage"),
        xr_haptic(amplitude=1.0, duration=0.1, frequency=200.0) if haptics_enabled else _noop,
        track_changes,
        component,
    )

@parameterized_stack
def hand_tracking_component(
    hand: str = "left",
    gesture_recognition: bool = True,
) -> Stack:
    """Hand tracking with gesture recognition."""
    return stack(
        xr_hand(hand=hand, gesture_recognition=gesture_recognition),
        xr_tracked(tracking_type="hand", tracking_space="stage"),
        track_changes,
        component,
    )

@parameterized_stack
def xr_interactable_object(
    grab_enabled: bool = True,
    physics_enabled: bool = True,
    highlight_on_hover: bool = True,
) -> Stack:
    """Object that can be interacted with in XR."""
    return stack(
        xr_interactable(
            interaction_types=["hover", "select", "grab"] if grab_enabled else ["hover", "select"],
            grab_points=[],
        ),
        xr_grabbable(grab_type="physics" if physics_enabled else "kinematic") if grab_enabled else _noop,
        track_changes,
        component,
    )

@parameterized_stack
def xr_grabbable_object(
    grab_type: str = "physics",
    two_handed: bool = False,
    attach_to_hand: bool = True,
) -> Stack:
    """Object that can be grabbed and held."""
    return stack(
        xr_interactable(interaction_types=["hover", "select", "grab"], grab_points=[]),
        xr_grabbable(grab_type=grab_type, attach_transform=(0, 0, 0)),
        track_changes,
        serializable(format="binary"),
        component,
    )

@parameterized_stack
def xr_ui_element(
    panel_type: str = "world",
    interaction_mode: str = "ray",
    curved: bool = False,
) -> Stack:
    """XR UI panel element."""
    return stack(
        xr_ui_panel(panel_type=panel_type, interaction_mode=interaction_mode),
        xr_interactable(interaction_types=["hover", "select"], grab_points=[]),
        track_changes,
        component,
    )

@parameterized_stack
def teleport_destination(
    teleport_type: str = "instant",
    valid_surface: bool = True,
) -> Stack:
    """Valid teleportation target."""
    return stack(
        xr_teleport_area(teleport_type=teleport_type),
        spatial(structure="grid", cell_size=1.0),
        track_changes,
        component,
    )

@parameterized_stack
def ar_spatial_anchor(
    persistent: bool = True,
    cloud_enabled: bool = False,
) -> Stack:
    """AR spatial anchor for world-locked content."""
    return stack(
        spatial_anchor(anchor_type="local" if not cloud_enabled else "cloud", persistent=persistent),
        xr_tracked(tracking_type="anchor", tracking_space="unbounded"),
        track_changes,
        serializable(format="binary") if persistent else _noop,
        component,
    )

@parameterized_stack
def ar_image_target(
    reference_image: str = "",
    tracking_mode: str = "continuous",
) -> Stack:
    """AR image tracking target."""
    return stack(
        ar_trackable(trackable_type="image", reference_id=reference_image),
        xr_tracked(tracking_type="image", tracking_space="unbounded"),
        track_changes,
        component,
    )

@parameterized_stack
def xr_avatar_component(
    ik_enabled: bool = True,
    network_sync: bool = True,
    face_tracking: bool = False,
) -> Stack:
    """XR avatar with IK and optional networking."""
    return stack(
        xr_avatar(ik_enabled=ik_enabled, network_sync=network_sync),
        networked(
            relevance="spatial", 
            authority="owner", 
            interpolated="hermite",
            predicted=True,
        ) if network_sync else _noop,
        blend_tree(parameter="movement", clips=["idle", "walk", "run"]),
        track_changes,
        component,
    )

@parameterized_stack
def xr_avatar_hand(
    hand: str = "left",
    physics_enabled: bool = True,
) -> Stack:
    """XR avatar hand with finger animation."""
    return stack(
        xr_hand(hand=hand, gesture_recognition=True),
        xr_ik_target(target_type="hand", bone_chain=["shoulder", "elbow", "wrist"]),
        blend_tree(parameter="grip", clips=["open", "point", "fist", "pinch"]),
        track_changes,
        component,
    )

@parameterized_stack
def multiplayer_xr_avatar(
    pool_size: int = 16,
    interpolation: str = "hermite",
) -> Stack:
    """Networked XR avatar for multiplayer."""
    from trinity.decorators.builtin_stacks.network import predicted_entity
    
    return (
        xr_avatar_component(ik_enabled=True, network_sync=True)
        + predicted_entity(history_frames=30)
    )

@parameterized_stack
def xr_comfort_locomotion(
    locomotion_type: str = "teleport",
    vignette: bool = True,
) -> Stack:
    """Comfort-aware locomotion system."""
    return stack(
        xr_locomotion(locomotion_type=locomotion_type, speed=3.0),
        xr_comfort(comfort_type="locomotion", settings={"vignette": vignette}),
        track_changes,
        component,
    )
```

### Composite Stacks

```python
@parameterized_stack
def full_xr_player(
    hand_tracking: bool = True,
    eye_tracking: bool = False,
    multiplayer: bool = False,
) -> Stack:
    """Complete XR player rig with HMD, controllers, and optional features."""
    return stack(
        tracked_xr_device(tracking_type="6dof"),
        xr_avatar_component(ik_enabled=True, network_sync=multiplayer),
        xr_comfort_locomotion(locomotion_type="teleport", vignette=True),
        component,
    )

@parameterized_stack
def xr_weapon(
    two_handed: bool = False,
    haptic_feedback: bool = True,
) -> Stack:
    """XR weapon with grab and haptics."""
    return stack(
        xr_grabbable_object(grab_type="physics", two_handed=two_handed),
        xr_haptic(amplitude=0.8, duration=0.05, frequency=250.0) if haptic_feedback else _noop,
        serializable(format="binary"),
        component,
    )

@parameterized_stack
def xr_tool(
    tool_type: str = "generic",
    attach_to_socket: bool = True,
) -> Stack:
    """XR tool that attaches to hand or belt."""
    return stack(
        xr_grabbable_object(grab_type="kinematic", attach_to_hand=True),
        xr_socket(socket_type="tool", accepted_tags=["hand", "belt"]) if attach_to_socket else _noop,
        track_changes,
        component,
    )

@parameterized_stack
def ar_furniture_placement(
    persistent: bool = True,
) -> Stack:
    """AR furniture for room decoration apps."""
    return stack(
        ar_spatial_anchor(persistent=persistent),
        xr_interactable_object(grab_enabled=True, physics_enabled=False),
        serializable(format="json"),
        component,
    )
```

---

## Part III: XR-Specific Patterns

### HMD Tracking Pattern

```python
@xr_tracked(tracking_type="6dof", tracking_space="stage")
@component
class HeadMountedDisplay:
    """Head-mounted display tracking."""
    
    # Identity
    device_id: Annotated[str, Immutable] = ""
    
    # Pose (predicted + tracked)
    position: Annotated[tuple, Predicted, Tracked, Atomic] = (0.0, 0.0, 0.0)
    orientation: Annotated[tuple, Predicted, Tracked, Atomic] = (0.0, 0.0, 0.0, 1.0)
    
    # Velocity for prediction
    linear_velocity: Annotated[tuple, Tracked] = (0.0, 0.0, 0.0)
    angular_velocity: Annotated[tuple, Tracked] = (0.0, 0.0, 0.0)
    
    # Tracking quality
    tracking_state: Annotated[str, Tracked, Observable] = "unknown"
    confidence: Annotated[float, Tracked, Range(0, 1)] = 0.0
    
    # Computed view matrices
    @computed
    def left_view_matrix(self) -> Any:
        """Compute left eye view matrix from pose."""
        return compute_view_matrix(self.position, self.orientation, ipd=-0.032)
    
    @computed
    def right_view_matrix(self) -> Any:
        """Compute right eye view matrix from pose."""
        return compute_view_matrix(self.position, self.orientation, ipd=0.032)

# Tracking state machine
@state_machine(
    initial="initializing",
    states={"initializing", "tracking", "limited", "lost", "disabled"},
    transitions={
        "initializing": {"tracking", "lost", "disabled"},
        "tracking": {"limited", "lost", "disabled"},
        "limited": {"tracking", "lost", "disabled"},
        "lost": {"tracking", "limited", "disabled"},
        "disabled": {"initializing"},
    }
)
class HMDTrackingState:
    """HMD tracking state machine."""
    
    @on_enter(state="lost")
    def on_tracking_lost(self):
        # Show recenter UI, pause gameplay
        pass
    
    @on_enter(state="tracking")
    def on_tracking_restored(self):
        # Resume normal operation
        pass
```

### Controller Input Pattern

```python
@xr_controller(hand="left", controller_type="motion")
@xr_tracked(tracking_type="6dof", tracking_space="stage")
@component
class XRController:
    """XR motion controller with input and haptics."""
    
    # Identity
    hand: Annotated[str, Immutable] = "left"
    
    # Poses
    grip_position: Annotated[tuple, Predicted, Tracked] = (0.0, 0.0, 0.0)
    grip_orientation: Annotated[tuple, Predicted, Tracked] = (0.0, 0.0, 0.0, 1.0)
    aim_position: Annotated[tuple, Predicted, Tracked] = (0.0, 0.0, 0.0)
    aim_orientation: Annotated[tuple, Predicted, Tracked] = (0.0, 0.0, 0.0, 1.0)
    
    # Analog inputs
    trigger: Annotated[float, Tracked, Range(0, 1)] = 0.0
    grip: Annotated[float, Tracked, Range(0, 1)] = 0.0
    thumbstick_x: Annotated[float, Tracked, Range(-1, 1)] = 0.0
    thumbstick_y: Annotated[float, Tracked, Range(-1, 1)] = 0.0
    
    # Digital inputs
    trigger_pressed: Annotated[bool, Tracked] = False
    grip_pressed: Annotated[bool, Tracked] = False
    primary_pressed: Annotated[bool, Tracked] = False  # A/X
    secondary_pressed: Annotated[bool, Tracked] = False  # B/Y
    thumbstick_pressed: Annotated[bool, Tracked] = False
    menu_pressed: Annotated[bool, Tracked] = False
    
    # Touch sensing
    trigger_touched: Annotated[bool, Tracked] = False
    thumbstick_touched: Annotated[bool, Tracked] = False
    thumbrest_touched: Annotated[bool, Tracked] = False
    
    # Tracking
    is_tracked: Annotated[bool, Tracked, Observable] = False
    
    # Haptics (write-only, transient)
    def play_haptic(self, amplitude: float, duration: float, frequency: float = 200.0):
        """Trigger haptic feedback."""
        pass

# Input action bindings
@input_action(name="xr_grab", default_bindings=["xr_left_grip", "xr_right_grip"])
def on_grab(controller: XRController, pressed: bool):
    pass

@input_action(name="xr_trigger", default_bindings=["xr_left_trigger", "xr_right_trigger"])
def on_trigger(controller: XRController, value: float):
    pass

@input_axis(name="xr_move", positive=["xr_left_thumbstick_up"], negative=["xr_left_thumbstick_down"])
def on_move(value: float):
    pass
```

### Hand Tracking Pattern

```python
# Hand joint enum
class HandJoint:
    WRIST = 0
    THUMB_METACARPAL = 1
    THUMB_PROXIMAL = 2
    THUMB_DISTAL = 3
    THUMB_TIP = 4
    INDEX_METACARPAL = 5
    INDEX_PROXIMAL = 6
    INDEX_INTERMEDIATE = 7
    INDEX_DISTAL = 8
    INDEX_TIP = 9
    # ... (26 total joints)

@xr_hand(hand="left", gesture_recognition=True)
@xr_tracked(tracking_type="hand", tracking_space="stage")
@component
class HandTracking:
    """Hand tracking with 26 joints and gesture recognition."""
    
    hand: Annotated[str, Immutable] = "left"
    
    # Joint data (batched for efficient updates)
    joint_positions: Annotated[list[tuple], Batched, Interpolated, Tracked] = field(
        default_factory=lambda: [(0.0, 0.0, 0.0)] * 26
    )
    joint_orientations: Annotated[list[tuple], Batched, Interpolated, Tracked] = field(
        default_factory=lambda: [(0.0, 0.0, 0.0, 1.0)] * 26
    )
    joint_radii: Annotated[list[float], Batched, Tracked] = field(
        default_factory=lambda: [0.01] * 26
    )
    
    # Tracking state
    is_tracked: Annotated[bool, Tracked, Observable] = False
    confidence: Annotated[float, Tracked, Range(0, 1)] = 0.0
    
    # Gesture recognition
    current_gesture: Annotated[str, Tracked, Observable] = "none"
    gesture_confidence: Annotated[float, Tracked, Range(0, 1)] = 0.0
    
    # Pinch detection
    pinch_strength: Annotated[float, Tracked, Range(0, 1)] = 0.0
    
    @computed
    def pinch_position(self) -> tuple:
        """Midpoint between thumb and index tips."""
        thumb_tip = self.joint_positions[HandJoint.THUMB_TIP]
        index_tip = self.joint_positions[HandJoint.INDEX_TIP]
        return (
            (thumb_tip[0] + index_tip[0]) / 2,
            (thumb_tip[1] + index_tip[1]) / 2,
            (thumb_tip[2] + index_tip[2]) / 2,
        )
    
    @computed
    def is_pinching(self) -> bool:
        return self.pinch_strength > 0.8

# Gesture events
@event
class GestureEvent:
    hand: str
    gesture: str
    confidence: float
    timestamp: float
```

### XR Interaction Pattern

```python
@xr_interactable(interaction_types=["hover", "select", "grab"], grab_points=[])
@component
class XRInteractable:
    """Base interactable object in XR."""
    
    # State
    is_hovered: Annotated[bool, Tracked, Observable] = False
    is_selected: Annotated[bool, Tracked, Observable] = False
    is_grabbed: Annotated[bool, Tracked, Observable] = False
    
    # Interacting controller/hand
    hover_interactor: Annotated[Optional[int], Tracked] = None  # Entity ID
    select_interactor: Annotated[Optional[int], Tracked] = None
    grab_interactor: Annotated[Optional[int], Tracked] = None
    
    # Grab configuration
    grab_points: Annotated[list[tuple], Tracked] = field(default_factory=list)
    
    # Callbacks
    @on_change(component="is_hovered")
    def on_hover_changed(self, old: bool, new: bool):
        pass
    
    @on_change(component="is_grabbed")
    def on_grab_changed(self, old: bool, new: bool):
        pass

@xr_grabbable(grab_type="physics", attach_transform=(0, 0, 0))
@component
class XRGrabbable(XRInteractable):
    """Object that can be grabbed and manipulated."""
    
    # Grab type
    grab_type: Annotated[str, Tracked] = "physics"  # physics, kinematic, custom
    
    # Attachment
    attach_transform: Annotated[tuple, Tracked] = (0.0, 0.0, 0.0)
    attach_rotation: Annotated[tuple, Tracked] = (0.0, 0.0, 0.0, 1.0)
    
    # Physics state when grabbed
    _was_kinematic: Annotated[bool, Transient] = False
    _original_parent: Annotated[Any, Transient] = None
    
    # Two-handed grab support
    allow_two_handed: Annotated[bool, Tracked] = False
    secondary_grab_point: Annotated[tuple, Tracked] = None

@xr_socket(socket_type="holster", accepted_tags=["weapon"])
@component
class XRSocket:
    """Snap socket for attaching objects."""
    
    socket_type: Annotated[str, Tracked] = "generic"
    accepted_tags: Annotated[list[str], Tracked] = field(default_factory=list)
    
    # State
    is_occupied: Annotated[bool, Tracked, Observable] = False
    attached_entity: Annotated[Optional[int], Tracked] = None
    
    # Snap behavior
    snap_distance: Annotated[float, Tracked] = 0.1
    snap_on_release: Annotated[bool, Tracked] = True
```

### Locomotion Pattern

```python
@xr_locomotion(locomotion_type="teleport", speed=3.0)
@xr_comfort(comfort_type="locomotion", settings={})
@component
class TeleportLocomotion:
    """Teleport-based locomotion with comfort features."""
    
    # State
    is_aiming: Annotated[bool, Tracked] = False
    aim_valid: Annotated[bool, Tracked] = False
    
    # Target
    target_position: Annotated[tuple, Tracked] = (0.0, 0.0, 0.0)
    target_rotation: Annotated[float, Tracked] = 0.0
    
    # Arc visualization
    arc_points: Annotated[list[tuple], Tracked] = field(default_factory=list)
    
    # Settings
    max_distance: Annotated[float, Tracked] = 10.0
    arc_gravity: Annotated[float, Tracked] = -9.8
    
    # Comfort
    fade_on_teleport: Annotated[bool, Tracked] = True
    fade_duration: Annotated[float, Tracked, Range(0, 0.5)] = 0.1

@xr_locomotion(locomotion_type="smooth", speed=3.0)
@xr_comfort(comfort_type="locomotion", settings={"vignette": True})
@component
class SmoothLocomotion:
    """Smooth locomotion with vignette comfort."""
    
    # Movement
    move_speed: Annotated[float, Tracked] = 3.0
    strafe_speed: Annotated[float, Tracked] = 2.0
    
    # Turning
    turn_type: Annotated[str, Tracked] = "snap"  # snap, smooth
    snap_angle: Annotated[float, Tracked, Range(15, 90)] = 45.0
    smooth_turn_speed: Annotated[float, Tracked] = 90.0
    
    # Comfort vignette
    vignette_enabled: Annotated[bool, Tracked] = True
    vignette_intensity: Annotated[float, Tracked, Range(0, 1)] = 0.5
    vignette_angular_velocity_threshold: Annotated[float, Tracked] = 30.0
```

### XR Avatar Pattern

```python
@xr_avatar(ik_enabled=True, network_sync=True)
@networked(relevance="spatial", authority="owner", interpolated="hermite", predicted=True)
@component
class XRAvatar:
    """Full-body XR avatar with IK."""
    
    # IK targets (from HMD and controllers)
    head_target: Annotated[tuple, Tracked] = (0.0, 1.7, 0.0)
    head_rotation: Annotated[tuple, Tracked] = (0.0, 0.0, 0.0, 1.0)
    left_hand_target: Annotated[tuple, Tracked] = (0.0, 0.0, 0.0)
    left_hand_rotation: Annotated[tuple, Tracked] = (0.0, 0.0, 0.0, 1.0)
    right_hand_target: Annotated[tuple, Tracked] = (0.0, 0.0, 0.0)
    right_hand_rotation: Annotated[tuple, Tracked] = (0.0, 0.0, 0.0, 1.0)
    
    # Calibration
    player_height: Annotated[float, Tracked] = 1.7
    arm_span: Annotated[float, Tracked] = 1.7
    
    # Body estimation (computed from head/hands)
    estimated_torso_position: Annotated[tuple, Computed] = (0.0, 0.0, 0.0)
    estimated_torso_rotation: Annotated[tuple, Computed] = (0.0, 0.0, 0.0, 1.0)
    
    # Visibility
    visible_to_self: Annotated[bool, Tracked] = False  # Usually false for self
    visible_to_others: Annotated[bool, Tracked] = True

@xr_ik_target(target_type="hand", bone_chain=["shoulder", "elbow", "wrist"])
@component
class AvatarHand:
    """Avatar hand with finger animation."""
    
    hand: Annotated[str, Immutable] = "left"
    
    # From controller or hand tracking
    target_position: Annotated[tuple, Tracked] = (0.0, 0.0, 0.0)
    target_rotation: Annotated[tuple, Tracked] = (0.0, 0.0, 0.0, 1.0)
    
    # Finger curl (0 = open, 1 = closed)
    thumb_curl: Annotated[float, Tracked, Range(0, 1)] = 0.0
    index_curl: Annotated[float, Tracked, Range(0, 1)] = 0.0
    middle_curl: Annotated[float, Tracked, Range(0, 1)] = 0.0
    ring_curl: Annotated[float, Tracked, Range(0, 1)] = 0.0
    pinky_curl: Annotated[float, Tracked, Range(0, 1)] = 0.0
    
    # Display mode
    display_mode: Annotated[str, Tracked] = "hand"  # hand, controller, tool
    held_tool_id: Annotated[Optional[int], Tracked] = None
```

### Spatial Understanding Pattern (AR)

```python
@spatial_anchor(anchor_type="local", persistent=True)
@xr_tracked(tracking_type="anchor", tracking_space="unbounded")
@component
class SpatialAnchor:
    """AR spatial anchor for world-locked content."""
    
    # Identity
    anchor_id: Annotated[str, Immutable] = ""
    
    # Pose
    position: Annotated[tuple, Tracked] = (0.0, 0.0, 0.0)
    orientation: Annotated[tuple, Tracked] = (0.0, 0.0, 0.0, 1.0)
    
    # State
    tracking_state: Annotated[str, Tracked, Observable] = "unknown"
    
    # Persistence
    is_persistent: Annotated[bool, Tracked] = True
    
    # Cloud anchor (for multi-user)
    cloud_anchor_id: Annotated[Optional[str], Tracked] = None
    is_resolved: Annotated[bool, Tracked] = False

@component
class PlaneDetection:
    """Detected AR plane (floor, wall, table)."""
    
    plane_id: Annotated[str, Immutable] = ""
    
    # Classification
    plane_type: Annotated[str, Tracked] = "unknown"  # floor, ceiling, wall, table, seat
    
    # Geometry
    center: Annotated[tuple, Tracked] = (0.0, 0.0, 0.0)
    normal: Annotated[tuple, Tracked] = (0.0, 1.0, 0.0)
    bounds: Annotated[list[tuple], Tracked] = field(default_factory=list)  # Polygon vertices
    
    # Size
    width: Annotated[float, Tracked] = 0.0
    height: Annotated[float, Tracked] = 0.0
    
    # Tracking
    is_tracked: Annotated[bool, Tracked, Observable] = False

@ar_trackable(trackable_type="image", reference_id="")
@component
class ImageTarget:
    """AR image tracking target."""
    
    # Reference
    reference_image_id: Annotated[str, Immutable] = ""
    physical_size: Annotated[tuple[float, float], Immutable] = (0.1, 0.1)
    
    # Tracking
    is_tracked: Annotated[bool, Tracked, Observable] = False
    
    # Pose
    position: Annotated[tuple, Tracked] = (0.0, 0.0, 0.0)
    orientation: Annotated[tuple, Tracked] = (0.0, 0.0, 0.0, 1.0)
    
    # Extents in world space
    tracked_size: Annotated[tuple[float, float], Tracked] = (0.0, 0.0)
```

### XR UI Pattern

```python
@xr_ui_panel(panel_type="world", interaction_mode="ray")
@xr_interactable(interaction_types=["hover", "select"], grab_points=[])
@component
class XRUIPanel:
    """World-space XR UI panel."""
    
    # Layout
    width: Annotated[float, Tracked] = 1.0  # Meters
    height: Annotated[float, Tracked] = 0.75
    pixels_per_meter: Annotated[float, Tracked] = 1000.0
    
    # Type
    panel_type: Annotated[str, Tracked] = "world"  # world, head_locked, hand_attached, wrist
    
    # Interaction
    interaction_mode: Annotated[str, Tracked] = "ray"  # ray, poke, gaze
    
    # Curved display (for menus)
    is_curved: Annotated[bool, Tracked] = False
    curve_radius: Annotated[float, Tracked] = 2.0
    
    # Visibility
    face_camera: Annotated[bool, Tracked] = False
    billboard: Annotated[bool, Tracked] = False

@component
class XRButton(XRUIPanel):
    """XR-interactable button."""
    
    label: Annotated[str, Tracked] = ""
    
    # State
    is_hovered: Annotated[bool, Tracked, Observable] = False
    is_pressed: Annotated[bool, Tracked, Observable] = False
    
    # Haptic feedback on press
    haptic_on_press: Annotated[bool, Tracked] = True
    
    # Poke depth for physical buttons
    press_depth: Annotated[float, Tracked] = 0.02  # Meters
    current_depth: Annotated[float, Tracked] = 0.0
```

### XR Rendering Pattern

```python
@resource
class XRRenderSettings:
    """XR rendering configuration."""
    
    # Resolution
    render_scale: Annotated[float, Tracked, Range(0.5, 2.0)] = 1.0
    
    # Stereo mode
    stereo_mode: Annotated[str, Tracked] = "multiview"  # multiview, instanced, sequential
    
    # Foveated rendering
    foveated_enabled: Annotated[bool, Tracked] = False
    foveated_level: Annotated[int, Tracked, Range(0, 4)] = 2
    foveated_dynamic: Annotated[bool, Tracked] = False  # Requires eye tracking
    
    # Reprojection
    reprojection_mode: Annotated[str, Tracked] = "atw"  # none, atw, asw
    
    # Hidden area mask
    hidden_area_mask_enabled: Annotated[bool, Tracked] = True
    
    # Performance targets
    target_framerate: Annotated[int, Tracked] = 90  # 72, 90, 120
    
@foveated_region(region_type="peripheral", quality_level=2)
@component
class FoveatedRenderRegion:
    """Foveated rendering region configuration."""
    
    region_type: Annotated[str, Tracked] = "fovea"  # fovea, parafoveal, peripheral
    
    # Quality level (0 = full, 4 = lowest)
    quality_level: Annotated[int, Tracked, Range(0, 4)] = 0
    
    # Size (as fraction of FOV)
    inner_radius: Annotated[float, Tracked, Range(0, 1)] = 0.0
    outer_radius: Annotated[float, Tracked, Range(0, 1)] = 0.2
```

---

## Part IV: Engine/XR Directory Structure

```
engine/xr/
+-- __init__.py              # Public API exports
+-- runtime/
|   +-- __init__.py
|   +-- xr_runtime.py        # XR runtime abstraction
|   +-- openxr.py            # OpenXR backend
|   +-- webxr.py             # WebXR backend
|   +-- capabilities.py      # Device capability queries
|   +-- session.py           # XR session management
+-- input/
|   +-- __init__.py
|   +-- hmd.py               # HMD tracking component
|   +-- controller.py        # Controller input component
|   +-- hand_tracking.py     # Hand tracking component
|   +-- eye_tracking.py      # Eye tracking component
|   +-- bindings.py          # Input action bindings
|   +-- haptics.py           # Haptic feedback system
+-- rendering/
|   +-- __init__.py
|   +-- stereo.py            # Stereo rendering
|   +-- foveated.py          # Foveated rendering
|   +-- reprojection.py      # ATW/ASW reprojection
|   +-- compositor.py        # Compositor layer management
|   +-- hidden_area.py       # Hidden area mesh optimization
+-- interaction/
|   +-- __init__.py
|   +-- interactable.py      # XR interactable component
|   +-- grabbable.py         # Grabbable objects
|   +-- socket.py            # Snap socket system
|   +-- ray_interactor.py    # Ray-based interaction
|   +-- direct_interactor.py # Direct/poke interaction
|   +-- gaze_interactor.py   # Gaze-based interaction
+-- spatial/
|   +-- __init__.py
|   +-- anchor.py            # Spatial anchor component
|   +-- plane_detection.py   # Plane detection
|   +-- mesh_mapping.py      # Spatial mesh mapping
|   +-- scene_understanding.py # Scene semantic understanding
|   +-- image_tracking.py    # AR image tracking
|   +-- object_tracking.py   # AR 3D object tracking
+-- avatars/
|   +-- __init__.py
|   +-- avatar.py            # XR avatar component
|   +-- ik_solver.py         # IK solver for body estimation
|   +-- hand_animator.py     # Hand/finger animation
|   +-- face_tracking.py     # Face/expression tracking
|   +-- calibration.py       # Avatar calibration
+-- locomotion/
|   +-- __init__.py
|   +-- teleport.py          # Teleport locomotion
|   +-- smooth.py            # Smooth locomotion
|   +-- climbing.py          # Climbing locomotion
|   +-- comfort.py           # Comfort settings/vignette
+-- ui/
|   +-- __init__.py
|   +-- panel.py             # XR UI panel
|   +-- button.py            # XR button
|   +-- slider.py            # XR slider
|   +-- keyboard.py          # Virtual keyboard
|   +-- wrist_ui.py          # Wrist-attached UI
```

---

## Part V: Implementation Checklist

### Phase 1: XR Runtime Foundation

- [ ] `runtime/xr_runtime.py` - XR runtime abstraction layer
- [ ] `runtime/openxr.py` - OpenXR backend implementation
- [ ] `runtime/capabilities.py` - Device capability detection
- [ ] `runtime/session.py` - XR session state machine

### Phase 2: Input Tracking

- [ ] `input/hmd.py` - HMD pose tracking with prediction
- [ ] `input/controller.py` - Controller input with all buttons/axes
- [ ] `input/bindings.py` - Input action binding system
- [ ] `input/haptics.py` - Haptic feedback system

### Phase 3: Hand & Eye Tracking

- [ ] `input/hand_tracking.py` - 26-joint hand tracking
- [ ] `input/eye_tracking.py` - Gaze tracking and fixation
- [ ] Gesture recognition system

### Phase 4: XR Rendering

- [ ] `rendering/stereo.py` - Stereo rendering (multiview/instanced)
- [ ] `rendering/foveated.py` - Fixed and dynamic foveated rendering
- [ ] `rendering/reprojection.py` - ATW/ASW implementation
- [ ] `rendering/compositor.py` - Compositor layers
- [ ] `rendering/hidden_area.py` - Hidden area mesh optimization

### Phase 5: XR Interaction

- [ ] `interaction/interactable.py` - Base interactable component
- [ ] `interaction/grabbable.py` - Grab mechanics (physics/kinematic)
- [ ] `interaction/socket.py` - Snap socket system
- [ ] `interaction/ray_interactor.py` - Ray casting interaction
- [ ] `interaction/direct_interactor.py` - Direct/poke interaction

### Phase 6: Spatial Understanding (AR)

- [ ] `spatial/anchor.py` - Spatial anchor with persistence
- [ ] `spatial/plane_detection.py` - Floor/wall/table detection
- [ ] `spatial/mesh_mapping.py` - Real-time mesh generation
- [ ] `spatial/image_tracking.py` - Image target tracking

### Phase 7: Avatars

- [ ] `avatars/avatar.py` - Full-body avatar with IK
- [ ] `avatars/ik_solver.py` - FABRIK/CCD IK solver
- [ ] `avatars/hand_animator.py` - Finger pose animation
- [ ] `avatars/calibration.py` - Height/arm span calibration

### Phase 8: Locomotion

- [ ] `locomotion/teleport.py` - Teleport with arc visualization
- [ ] `locomotion/smooth.py` - Smooth movement with snap/smooth turn
- [ ] `locomotion/comfort.py` - Vignette and comfort features

### Phase 9: XR UI

- [ ] `ui/panel.py` - World-space UI panels
- [ ] `ui/button.py` - XR buttons with haptics
- [ ] `ui/wrist_ui.py` - Wrist-attached quick menu

### Phase 10: Integration

- [ ] Wire TrackedDescriptor to pose updates
- [ ] Wire PredictedDescriptor to ATW system
- [ ] Wire InterpolatedDescriptor to hand joint smoothing
- [ ] Wire NetworkedDescriptor to multiplayer avatar sync
- [ ] Wire Foundation Tracker for XR state changes
- [ ] Wire Foundation EventLog for input events

---

## Quick Reference

### Descriptor Choice Guide

| Need | Descriptor | Example |
|------|-----------|---------|
| Pose change detection | `Tracked` | HMD position triggers render |
| Smooth joint animation | `Interpolated` | Hand joints between frames |
| Low-latency pose | `Predicted` | HMD for ATW correction |
| Button value limits | `Range` | Trigger 0-1, thumbstick -1 to 1 |
| Derived IK values | `Computed` | Avatar bone from targets |
| Thread-safe pose | `Atomic` | Tracking thread updates |
| Timed confidence | `Expiring` | Tracking confidence decay |
| Multiplayer sync | `Networked` | Avatar pose replication |
| Bulk joint updates | `Batched` | 26 hand joints at once |
| Read-only specs | `Immutable` | Device capabilities |
| Runtime cache | `Transient` | View matrices, haptic state |

### Decorator Choice Guide

| Need | Decorator | Module |
|------|-----------|--------|
| Mark tracked device | `@xr_tracked` | (new) |
| Controller config | `@xr_controller` | (new) |
| Hand tracking | `@xr_hand` | (new) |
| Interactable object | `@xr_interactable` | (new) |
| Grabbable object | `@xr_grabbable` | (new) |
| Snap socket | `@xr_socket` | (new) |
| Teleport area | `@xr_teleport_area` | (new) |
| Locomotion config | `@xr_locomotion` | (new) |
| Haptic feedback | `@xr_haptic` | (new) |
| Spatial anchor | `@spatial_anchor` | (new) |
| AR trackable | `@ar_trackable` | (new) |
| Avatar config | `@xr_avatar` | (new) |
| XR UI panel | `@xr_ui_panel` | (new) |
| Input action | `@input_action` | `input.py` |
| Input axis | `@input_axis` | `input.py` |
| Animation blend | `@blend_tree` | `animation.py` |
| Network sync | `@networked` | `data_flow.py` |
| State history | `@snapshot` | `data_flow.py` |

### Stack Choice Guide

| Scenario | Stack | Purpose |
|----------|-------|---------|
| HMD/controller | `tracked_xr_device()` | 6DOF tracking |
| Motion controller | `xr_controller_component()` | Full controller input |
| Hand tracking | `hand_tracking_component()` | Gesture recognition |
| Grabbable object | `xr_grabbable_object()` | Physics/kinematic grab |
| XR UI | `xr_ui_element()` | World-space UI |
| Teleport target | `teleport_destination()` | Valid teleport area |
| AR anchor | `ar_spatial_anchor()` | World-locked content |
| Image marker | `ar_image_target()` | Image tracking |
| Local avatar | `xr_avatar_component()` | IK-driven body |
| Network avatar | `multiplayer_xr_avatar()` | Synced avatar |
| Comfort move | `xr_comfort_locomotion()` | Safe locomotion |

### Foundation Integration Points

| System | XR Use |
|--------|--------|
| Registry | Device/controller lookup, interactable query |
| Tracker | Pose dirty flags, input state changes |
| EventLog | Button presses, gesture events, tracking state |
| Mirror | XR state inspection for debugging |
| Bridge | ShellLang access to XR for testing |

### Performance Targets

| Metric | Target |
|--------|--------|
| Pose-to-Photon Latency | <20ms |
| Frame Time @ 90Hz | <11.1ms |
| Frame Time @ 120Hz | <8.3ms |
| Tracking Update Rate | >250Hz |
| Hand Joint Update Rate | >30Hz |
| Input Polling Rate | >1000Hz |

---

## References

- `docs/specs/TRINITY_LATEST.md` - Full Trinity Pattern specification
- `docs/GAME_ENGINE_INTEGRATION.md` - Trinity <-> Foundation integration
- `docs/GAME_ENGINE_INTEGRATION_TODO.md` - Section 13 (XR Layer)
- `DIAGRAMS/ARCHITECTURE_XR.md` - XR layer architecture
- `trinity/decorators/input.py` - Input action decorators
- `trinity/decorators/spatial.py` - Spatial indexing decorators
- `trinity/decorators/animation.py` - Animation decorators
- `trinity/decorators/data_flow.py` - Networking decorators
- `trinity/decorators/lifecycle.py` - Lifecycle hook decorators
- `trinity/decorators/builtin_stacks/network.py` - Network stacks
- `trinity/descriptors/` - All descriptor implementations
