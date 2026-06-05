# Phase X4: XR Interaction — Architecture

**Tasks:** T-XR-4.1 through T-XR-4.7 (7 tasks)
**Effort:** 18-25 days
**Status:** ✅ COMPLETE (per PROJECT.md verification)

---

## 1. Overview

Phase X4 implements XR interaction systems: interactable objects, grab mechanics, snap sockets, and multiple interactor types (ray, direct/poke, gaze).

---

## 2. Interactable Base (`interaction/interactable.py`)

### XRInteractable Component
```python
class XRInteractable:
    # State (observable for UI binding)
    hover_state: TrackedDescriptor[bool]
    select_state: TrackedDescriptor[bool]
    grab_state: TrackedDescriptor[bool]
    
    # Current interactor
    interactor: TrackedDescriptor[XRInteractor?]
    
    # Configuration
    interaction_types: list[InteractionType]  # ray, direct, gaze
```

### State Machine
```
IDLE → HOVERED → SELECTED → GRABBED → RELEASED → IDLE
```

`@on_change` hooks for each state transition.

---

## 3. Grab Mechanics (`interaction/grabbable.py`)

### XRGrabbable Component
Extends XRInteractable.

### Grab Types
| Type | Description |
|------|-------------|
| Physics | Joint constraint to hand |
| Kinematic | Parents to hand transform |
| Attach | Custom attach point/rotation |

### Two-Handed Grab
Secondary grab point enables two-handed object manipulation.

### State Machine
```
IDLE → HOVERED → SELECTED → GRABBED → RELEASED
                              ↓
                       TWO_HAND_GRABBED
```

---

## 4. Snap Sockets (`interaction/socket.py`)

### XRSocket Component
```python
class XRSocket:
    socket_type: str
    accepted_tags: list[str]
    snap_distance: float  # meters
    
    is_occupied: TrackedDescriptor[bool]
    attached_entity: TrackedDescriptor[Entity?]
```

### Snap Behavior
- Auto-snap on release within proximity
- Visual preview when hovering near socket
- Example: holster, belt slot, weapon rack

---

## 5. Interactor Types

### Ray Interactor (`interaction/ray_interactor.py`)
```
Controller Aim Pose
    ↓
Parabolic Arc (gravity -9.8)
    ↓
Hit Detection
    ↓
Hover/Select/Grab dispatch
```

- Visual arc with bend indicator
- Configurable max distance (10m default)
- Valid/invalid visual feedback

### Direct Interactor (`interaction/direct_interactor.py`)
- Index fingertip proximity check
- Physical button press with depth tracking
- Haptic feedback on full press

### Gaze Interactor (`interaction/gaze_interactor.py`)
- Eye tracking gaze ray
- Dwell time selection (0.5s default)
- Progress indicator during dwell
- Disabled by default (least-preferred)

---

## 6. Decorators

| Decorator | Configuration |
|-----------|---------------|
| `@xr_interactable` | interaction_types, grab_points |
| `@xr_grabbable` | grab_type, attach_transform |
| `@xr_socket` | socket_type, accepted_tags |
| `@xr_haptic` | amplitude, duration, frequency |

---

## 7. Dependencies

- Phase X1: XR Runtime
- Phase X2: Controller input, hand tracking, eye tracking
- Phase X5: Spatial for hit testing
