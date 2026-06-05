# SUMMARY.md - engine_xr

## Metrics

| Metric | Value |
|--------|-------|
| **Total Lines** | 32,426 |
| **Total Files** | 60 |
| **Classification** | REAL |
| **Subsystems** | 10 |
| **Public Classes** | 85+ |
| **Decorators** | 3 (xr_interactable, xr_grabbable, xr_teleport_area) |

## File Distribution by Subsystem

| Subsystem | Files | Lines (approx) | Status |
|-----------|-------|----------------|--------|
| runtime/ | 5 | 3,100 | REAL |
| input/ | 6 | 4,200 | REAL |
| rendering/ | 6 | 3,800 | REAL |
| interaction/ | 6 | 3,600 | REAL |
| spatial/ | 6 | 4,700 | REAL |
| avatars/ | 6 | 3,500 | REAL |
| locomotion/ | 5 | 3,200 | REAL |
| ui/ | 6 | 2,700 | REAL |
| platform/ | 4 | 1,800 | REAL |
| utils/ | 4 | 400 | REAL |
| Root (config.py, __init__.py) | 2 | 700 | REAL |
| XR_CONTEXT.md | 1 | 1,700 | REAL |

## Algorithm Inventory

| Algorithm | File | Lines | Status | Complexity |
|-----------|------|-------|--------|------------|
| FABRIK IK Solver | avatars/ik_solver.py | 232-317 | REAL | O(n*k) iterations |
| CCD IK Solver | avatars/ik_solver.py | 319-417 | REAL | O(n*k) iterations |
| TwoBone Analytical IK | avatars/ik_solver.py | 419-588 | REAL | O(1) closed-form |
| Fixed Foveated Rendering | rendering/foveated.py | 175-296 | REAL | O(w*h) per frame |
| Dynamic Foveated Rendering | rendering/foveated.py | 298-454 | REAL | O(w*h) per frame |
| Contrast-Adaptive Foveation | rendering/foveated.py | 456-633 | REAL | O(w*h) per frame |
| Teleport Arc Trajectory | locomotion/teleport.py | 104-256 | REAL | O(n) projectile sim |
| Plane Detection Raycast | spatial/plane_detection.py | 675-719 | REAL | O(p) planes |
| Plane Contains Point | spatial/plane_detection.py | 67-94 | REAL | O(v) ray casting |
| Shoelace Area Formula | spatial/plane_detection.py | 96-115 | REAL | O(v) vertices |
| Gesture Recognition Pipeline | input/hand_tracking.py | 402-471 | REAL | O(g) gestures |
| Pinch Detection | input/hand_tracking.py | 525-542 | REAL | O(1) distance |
| Point Gesture Detection | input/hand_tracking.py | 544-572 | REAL | O(1) finger curls |
| Fist Gesture Detection | input/hand_tracking.py | 574-591 | REAL | O(1) finger curls |
| Finger Curl Calculation | input/hand_tracking.py | 267-316 | REAL | O(1) per finger |
| Throw Velocity Calculation | interaction/grabbable.py | 543-587 | REAL | O(s) samples |
| Two-Handed Transform | interaction/grabbable.py | 623-657 | REAL | O(1) interpolation |
| Look Rotation from Forward | avatars/ik_solver.py | 532-588 | REAL | O(1) quaternion |
| Gaze Smoothing (EMA) | rendering/foveated.py | 327-349 | REAL | O(1) exponential |

## Key Data Structures

| Structure | File | Purpose |
|-----------|------|---------|
| Pose | runtime/xr_runtime.py | 6DOF pose with velocities |
| ViewInfo | runtime/xr_runtime.py | Per-eye rendering parameters |
| IKChain | avatars/ik_solver.py | Joint chain for IK solving |
| IKJoint | avatars/ik_solver.py | Joint with limits and axes |
| HandTrackingData | input/hand_tracking.py | 26-joint hand skeleton |
| GestureResult | input/hand_tracking.py | Gesture detection output |
| DetectedPlane | spatial/plane_detection.py | AR plane with bounds |
| PlaneGeometry | spatial/plane_detection.py | Plane center/normal/extent |
| PlaneBounds | spatial/plane_detection.py | Boundary polygon |
| FoveationConfig | rendering/foveated.py | Foveation parameters |
| GazePoint | rendering/foveated.py | Eye gaze in NDC |
| TeleportLocomotion | locomotion/teleport.py | Teleport state machine |
| ArcPoint | locomotion/teleport.py | Point on teleport arc |
| GrabState | interaction/grabbable.py | Active grab information |
| ThrowData | interaction/grabbable.py | Throw physics data |
| XRConfig | config.py | All XR configuration |

## Dependencies

| Dependency | Usage |
|------------|-------|
| engine.core.math.vec | Vec2, Vec3 vector math |
| engine.core.math.quat | Quat quaternion operations |
| engine.core.math.transform | Transform, RigidTransform |
| engine.core.constants | MATH_EPSILON |
| trinity.decorators.ops | Op, Step, make_decorator |
| trinity.decorators.registry | DecoratorSpec, Tier, registry |
| engine.xr.config | XR_CONFIG singleton |
| engine.xr.utils.shading | VRS rate helpers |
| engine.xr.utils.math_utils | rotation_from_axes |
| engine.xr.utils.markers | Tracked, Range, Observable |
