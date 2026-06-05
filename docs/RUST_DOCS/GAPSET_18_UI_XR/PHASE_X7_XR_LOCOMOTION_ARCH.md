# Phase X7: XR Locomotion and Comfort — Architecture

**Tasks:** T-XR-7.1 through T-XR-7.5 (5 tasks)
**Effort:** 14-19 days
**Status:** ✅ COMPLETE (per PROJECT.md verification)

---

## 1. Overview

Phase X7 implements locomotion systems (teleport, smooth, climbing) and comfort features (vignette, seated mode, snap turn).

---

## 2. Teleport Locomotion (`locomotion/teleport.py`)

### Parabolic Arc Projection
```
position(t) = start + velocity*t + 0.5*gravity*t²
gravity = (0, -9.8, 0)  # configurable
```

### Arc Visualization
- Curved line with bend indicator
- Color: green (valid) / red (invalid)
- Max distance: 10m default

### Landing Validation
| Check | Requirement |
|-------|-------------|
| Surface normal | Within 45° of up |
| Clear space | 2m height clearance |
| Teleport area | Tagged with `@xr_teleport_area` |

### Transition
Fade to black (0.1s default), move, fade in.

---

## 3. Smooth Locomotion (`locomotion/smooth.py`)

### Movement
- Head-relative direction
- Speed: 3 m/s default
- Strafe: thumbstick X axis

### Rotation
| Mode | Input | Default |
|------|-------|---------|
| Snap Turn | thumbstick X threshold | 45° per snap |
| Smooth Turn | thumbstick X analog | 90°/s |

`@xr_locomotion` decorator configures mode and speed.

---

## 4. Climbing Locomotion (`locomotion/climbing.py`)

### Grab-to-Climb
1. Grab climbable surface with both hands
2. Pull/push to move body
3. Release mechanism:
   - Release both hands → fall
   - Release one + press button → controlled release

### Climbable Surfaces
Tagged via `@xr_interactable(interaction_types=["climb"])`.

---

## 5. Comfort Settings (`locomotion/comfort.py`)

### XRComfortSettings Resource
```python
class XRComfortSettings:
    vignette_intensity: RangeDescriptor[float]  # 0-1
    snap_turn_enabled: TrackedDescriptor[bool]
    snap_turn_angle: RangeDescriptor[float]  # 15-90°
    smooth_turn_speed: RangeDescriptor[float]  # 30-180°/s
    seated_mode: TrackedDescriptor[bool]
    seated_height_offset: RangeDescriptor[float]  # meters
```

### Vignette
Darkens periphery proportionally to motion.

| Motion | Vignette |
|--------|----------|
| Rest | 0% |
| Walking | 30% |
| Fast turn (30+°/s) | 100% |

Smooth transition between states.

---

## 6. Decorators

| Decorator | Configuration |
|-----------|---------------|
| `@xr_teleport_area` | teleport_type (instant/fade) |
| `@xr_locomotion` | locomotion_type, speed |
| `@xr_comfort` | comfort_type, settings |

---

## 7. Dependencies

- Phase X1: XR Runtime
- Phase X2: Controller input
- Phase X4: Grab mechanics for climbing
