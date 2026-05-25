# Phase 2: Input & Camera Systems — Architecture

## Overview

Four-layer input system (Device → Raw Processing → Action/Axis Mapping → Context Stack) and 8-mode camera system with collision, effects, rails, and blending.

## Component Breakdown

### Input System (`input/`)

```
Layer 1 — Device Manager
├── Device types: Keyboard, Mouse, Gamepad, Touch, Motion, XR
├── Hot-plug events
└── Raw input event emission

Layer 2 — Raw Input Processing
├── DeadZoneProcessor (3 types: axial, radial, custom)
├── ResponseCurveProcessor (5 types: linear, exponential, etc.)
├── SmoothingProcessor (4 types: moving average, etc.)
├── InvertProcessor
└── InputModifierChain (configurable pipeline)

Layer 3 — Action/Axis Mapping
├── @input_action(name, default_bindings)
│   └── Trigger types: Pressed, Released, Hold, Tap, DoubleTap, Combo
├── @input_axis(name, positive, negative)
│   └── Combined value [-1, 1]
├── Input buffer (500ms configurable window)
└── Combo detection

Layer 4 — Context Stack
├── Context priorities
├── Push/pop lifecycle
├── Passthrough flag
└── Contexts: OnFoot, InVehicle, Menu, Dialogue, Cutscene

Runtime Rebinding
├── Binding mutation at runtime
├── Serialization to disk
└── Load on session start
```

### Camera System (`camera/`)

```
CameraBase
├── Active mode enum, target, offset, FOV
└── 8 Camera Controllers
    ├── FirstPerson (character head socket)
    ├── ThirdPerson (spring arm orbit)
    ├── Orbit (rotate around pivot)
    ├── Follow (track with lag)
    ├── Free (WASD + mouse)
    ├── Cinematic (keyframes/spline path)
    ├── TopDown
    └── Isometric

CameraCollision
├── 5 response modes: PULL_IN, PUSH_OUT, FADE, CLIP, BLEND
├── Sphere cast with multi-probe rays
├── Soft collision interpolation
├── OcclusionDetector
└── TransparencyManager

CameraEffects
├── Shake: 7 types (Perlin, Sine, Random, Directional, Explosion, Impact, Continuous)
├── FOVEffect: punch, zoom, mod stack
├── TiltEffect: auto-level
├── DOFEffect: auto-focus
├── MotionBlur: velocity-based
└── VignetteEffect: damage/low-health presets

CameraRails
├── 4 spline types: Linear, Catmull-Rom, Bezier, Hermite
├── RailFollower: 4 loop modes (ONCE, LOOP, PING_PONG, CLAMP)
├── Arc-length parameterization
├── TriggerVolume (enter/exit/stay callbacks)
├── BlendRegion (axis-aligned blend weights)
├── Dolly (linear track)
├── Crane (vertical arc)
└── TriggerVolumeManager

CameraBlending
├── 12 blend types: Cut, Linear, EaseIn/Out, Smooth, Spring, Elastic, Bounce, etc.
├── CameraBlend: pause/resume/reverse/skip
├── BlendStack
├── SplitScreenLayout (7 layouts)
└── CameraDirector
```

## Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `input/devices.py` | — | DeviceManager with 6 device types, hot-plug |
| `input/processing.py` | — | Raw input processing pipeline |
| `input/actions.py` | — | @input_action, @input_axis, triggers |
| `input/context.py` | — | Input context stack |
| `input/bindings.py` | — | Runtime rebinding, serialization |
| `camera/camera.py` | — | Camera component, base |
| `camera/modes.py` | — | 8 camera controllers |
| `camera/collision.py` | 709 | Camera collision, occlusion, transparency |
| `camera/effects.py` | — | Shake, FOV, tilt, DOF, motion blur, vignette |
| `camera/rails.py` | 1346 | Camera rails, triggers, dolly, crane |
| `camera/blending.py` | — | 12 blend types, CameraDirector |
