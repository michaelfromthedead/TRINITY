# PHASE 3 TODO: Camera Subsystem

**Phase**: 3 of 3
**Subsystem**: engine/gameplay/camera
**Status**: Investigation Complete

---

## 1. Verification Tasks

### 1.1 Camera Controllers
- [ ] **T-CAM-1.1**: Test FirstPersonController head bob calculation
- [ ] **T-CAM-1.2**: Test ThirdPersonController boom arm interpolation
- [ ] **T-CAM-1.3**: Verify camera lag formula: `1 - exp(-speed * dt)`
- [ ] **T-CAM-1.4**: Test ThirdPersonController pitch limits enforcement
- [ ] **T-CAM-1.5**: Test OrbitController zoom and rotation
- [ ] **T-CAM-1.6**: Test FollowController lead prediction
- [ ] **T-CAM-1.7**: Test FreeController WASD movement
- [ ] **T-CAM-1.8**: Test CinematicController keyframe interpolation
- [ ] **T-CAM-1.9**: Test TopDownController pan bounds
- [ ] **T-CAM-1.10**: Test IsometricController 45-degree snap rotation

### 1.2 Collision
- [ ] **T-CAM-2.1**: Test sphere cast with 9 rays
- [ ] **T-CAM-2.2**: Test PULL_IN response mode
- [ ] **T-CAM-2.3**: Test PUSH_OUT response mode
- [ ] **T-CAM-2.4**: Test FADE response mode
- [ ] **T-CAM-2.5**: Test CLIP response mode
- [ ] **T-CAM-2.6**: Test BLEND response mode
- [ ] **T-CAM-2.7**: Test OcclusionDetector fade states
- [ ] **T-CAM-2.8**: Test hysteresis prevents flicker
- [ ] **T-CAM-2.9**: Test TransparencyManager mark/restore

### 1.3 Camera Effects
- [ ] **T-CAM-3.1**: Test PerlinShake octave layering
- [ ] **T-CAM-3.2**: Test SineShake oscillation
- [ ] **T-CAM-3.3**: Test RandomShake displacement
- [ ] **T-CAM-3.4**: Test DirectionalShake axis constraint
- [ ] **T-CAM-3.5**: Test ExplosionShake radial falloff
- [ ] **T-CAM-3.6**: Test ImpactShake impulse decay
- [ ] **T-CAM-3.7**: Test ContinuousShake persistence
- [ ] **T-CAM-3.8**: Test FOVEffect modifier stack
- [ ] **T-CAM-3.9**: Test TiltEffect Dutch angle
- [ ] **T-CAM-3.10**: Test DOFEffect auto-focus
- [ ] **T-CAM-3.11**: Test MotionBlurEffect velocity tracking
- [ ] **T-CAM-3.12**: Test VignetteEffect radius/intensity

### 1.4 Blending
- [ ] **T-CAM-4.1**: Test LINEAR blend curve
- [ ] **T-CAM-4.2**: Test EASE_IN blend curve
- [ ] **T-CAM-4.3**: Test EASE_OUT blend curve
- [ ] **T-CAM-4.4**: Test EASE_IN_OUT blend curve
- [ ] **T-CAM-4.5**: Test CUBIC blend curve
- [ ] **T-CAM-4.6**: Test EXPONENTIAL blend curve
- [ ] **T-CAM-4.7**: Test ELASTIC blend curve with overshoot
- [ ] **T-CAM-4.8**: Test BOUNCE blend curve piecewise segments
- [ ] **T-CAM-4.9**: Test BlendStack concurrent blends
- [ ] **T-CAM-4.10**: Test ViewportSplit HORIZONTAL_2 layout
- [ ] **T-CAM-4.11**: Test ViewportSplit VERTICAL_2 layout
- [ ] **T-CAM-4.12**: Test ViewportSplit QUAD layout
- [ ] **T-CAM-4.13**: Test ViewportSplit PIP layout
- [ ] **T-CAM-4.14**: Test CameraPriority selection
- [ ] **T-CAM-4.15**: Test CameraDirector cut_to/blend_to

### 1.5 Rails
- [ ] **T-CAM-5.1**: Test LINEAR spline interpolation
- [ ] **T-CAM-5.2**: Test CATMULL_ROM spline with tension
- [ ] **T-CAM-5.3**: Test BEZIER cubic interpolation
- [ ] **T-CAM-5.4**: Test HERMITE spline with tangents
- [ ] **T-CAM-5.5**: Test arc-length parameterization uniformity
- [ ] **T-CAM-5.6**: Test RailFollower LOOP mode
- [ ] **T-CAM-5.7**: Test RailFollower PING_PONG mode
- [ ] **T-CAM-5.8**: Test TriggerVolume enter/exit callbacks
- [ ] **T-CAM-5.9**: Test BlendRegion transition
- [ ] **T-CAM-5.10**: Test Dolly look-at tracking
- [ ] **T-CAM-5.11**: Test Crane arm angle/length

---

## 2. Integration Tasks

### 2.1 Trinity Pattern Integration
- [ ] **T-CAM-6.1**: Register CameraController with ComponentMeta
- [ ] **T-CAM-6.2**: Install TrackedDescriptor on position/rotation
- [ ] **T-CAM-6.3**: Register CameraEffect with ComponentMeta
- [ ] **T-CAM-6.4**: Create BlendStartEvent with EventMeta
- [ ] **T-CAM-6.5**: Create BlendEndEvent with EventMeta
- [ ] **T-CAM-6.6**: Register CameraRail with AssetMeta

### 2.2 Foundation Integration
- [ ] **T-CAM-7.1**: Connect camera state changes to Tracker
- [ ] **T-CAM-7.2**: Connect blend events to EventLog
- [ ] **T-CAM-7.3**: Register camera types with Registry

### 2.3 Physics Integration
- [ ] **T-CAM-8.1**: Wire sphere cast to PhysicsWorld
- [ ] **T-CAM-8.2**: Support collision layer masks
- [ ] **T-CAM-8.3**: Handle dynamic obstacles

---

## 3. Future Enhancements (Out of Scope)

### 3.1 VR/XR Support
- Stereoscopic rendering
- Head tracking integration
- IPD adjustment

### 3.2 Editor Tooling
- Visual rail editor
- Keyframe timeline
- Live camera preview
- Effect parameter tweaker

### 3.3 Network Replication
- Camera state sync
- Spectator mode
- Replay camera

---

## 4. Acceptance Criteria

| Task Group | Criteria |
|------------|----------|
| Controllers | All 8 controllers update correctly |
| Collision | All 5 response modes work, no flicker |
| Effects | All 7 shake types + 5 effects verified |
| Blending | All 12 curves correct, stack works |
| Rails | All 4 spline types correct, arc-length uniform |
| Integration | Trinity metaclasses/descriptors wired |
