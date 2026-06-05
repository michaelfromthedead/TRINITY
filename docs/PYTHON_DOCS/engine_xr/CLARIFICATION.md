# ENGINE XR CLARIFICATION

## Philosophical Framing

### What is Engine XR?

Engine XR is TRINITY's abstraction layer for immersive reality experiences. It does not merely wrap hardware APIs; it establishes a **presence contract** between the user's physical body and the virtual world. Every subsystem—from tracking to rendering to UI—exists to maintain this contract with minimal latency, maximum fidelity, and zero discomfort.

### The Presence Contract

Presence requires:
1. **Spatial Consistency**: The virtual world must respond to physical movement within 11ms
2. **Proprioceptive Alignment**: Avatars must match the user's body schema
3. **Perceptual Comfort**: Visual motion must not trigger vestibular mismatch
4. **Interaction Fidelity**: Virtual objects must feel physically responsive

These requirements cascade through every design decision.

## Design Rationale

### Why Three IK Solvers?

The avatar system implements FABRIK, CCD, and TwoBone because each excels at different constraints:

- **TwoBone**: Analytical solution for arms/legs. O(1) time, exact answer. Used when joint count is exactly two.
- **FABRIK**: Iterative forward/backward reaching. Handles long chains gracefully. Best for spine, tentacles, procedural creatures.
- **CCD**: Cyclic coordinate descent. Respects joint angle limits. Used when constraints matter more than naturalness.

A single solver cannot serve all body parts. The factory pattern (`create_solver()`) selects the appropriate algorithm.

### Why 52 Blend Shapes?

Face tracking uses ARKit's 52-blend-shape standard because:
1. It is the industry-adopted superset covering all major facial action units
2. Cross-platform tools (Unreal, Unity, Blender) support this exact set
3. Future Meta/Varjo face tracking maps to these shapes

The alternative—bone-driven faces—works but requires custom rigging per avatar.

### Why Three Stereo Rendering Methods?

- **Multi-View (OVR_multiview2)**: Single draw call renders both eyes. 40% fewer draw calls. Requires GPU support.
- **Instanced**: Uses gl_InstanceID for eye selection. Fallback when multi-view unsupported.
- **Sequential**: Traditional two-pass. Works everywhere. 2x draw call overhead.

The factory selects the best available method at runtime.

### Why Simulation Runtimes?

The OpenXR and WebXR runtimes return simulated poses because:
1. Native SDK bindings require platform-specific build toolchains
2. Development can proceed without physical hardware
3. The API contract is established for real implementation

This is intentional architecture, not incomplete code. The comment "This implementation simulates OpenXR behavior" is documentation, not a TODO.

### Why Guardian Math Without SDK?

Guardian boundary geometry (shoelace area, point-in-polygon, proximity calculation) is implemented because:
1. These algorithms are SDK-agnostic
2. Unit testing verifies correctness without hardware
3. SDK integration only needs to provide vertex data

The math is real; only the data source is stubbed.

### Why Memory-Efficient Markers?

Type annotation markers (`Tracked`, `Range`, `Observable`, `Transient`, `Immutable`) enable:
1. Descriptor-based change detection without runtime overhead
2. Serialization filtering (transient fields excluded)
3. UI binding (observable triggers render)
4. Validation (range clamping)

This is TRINITY's reactive component model applied to XR.

## Architectural Decisions

### Component vs Manager Pattern

Each XR subsystem follows the pattern:
- **Dataclass Component**: Pure data with optional methods (`XRAvatar`, `DetectedPlane`, `SpatialMesh`)
- **Manager Class**: Lifecycle, queries, and coordination (`AnchorManager`, `PlaneDetector`, `HandTracker`)

Components are ECS-compatible. Managers are singleton services.

### Provider Abstraction

Locomotion, haptics, and social services use Provider classes because:
1. Runtime swapping of implementations (teleport vs smooth)
2. Dependency injection for testing
3. Platform-specific behavior encapsulation

### Network State Pattern

All multiplayer-ready components implement:
- `get_network_state() -> dict`
- `apply_network_state(state: dict)`

Bandwidth optimization (non-zero blend shapes only) is explicit.

### Callback Registration

Events use explicit callback registration rather than inheritance because:
1. Multiple listeners per event
2. Runtime listener addition/removal
3. No diamond inheritance problems

## Platform Strategy

### Device Priority

The detection order (Quest > Vision Pro > PSVR2 > SteamVR > OpenXR) reflects:
1. Market share (Quest dominates standalone)
2. API specificity (native SDK > generic OpenXR for features)
3. Fallback safety (OpenXR is universal)

### SDK Integration Path

When implementing real SDK bindings:
1. **OpenXR**: Use `pyopenxr` or ctypes to `openxr_loader.dll`
2. **SteamVR**: Use `openvr` Python package
3. **Meta Quest**: Requires Android NDK bridge
4. **visionOS**: Requires Swift/Objective-C interop
5. **PSVR2**: Requires devkit and NDA

The abstraction is designed for this eventual integration.

## Performance Considerations

### Frame Time Budget

At 90Hz, the frame time budget is 11.11ms:
- **Tracking**: <1ms (prediction compensates for latency)
- **Culling**: <1ms (hidden area mesh, frustum cull)
- **Rendering**: 6-8ms (foveated rendering reduces this)
- **Reprojection**: 2ms (ATW/ASW handles drops)
- **Compositor**: <1ms (layer blending)

### Memory Optimization

All dataclasses use `slots=True` to:
1. Reduce per-instance memory overhead
2. Improve attribute access speed
3. Prevent accidental attribute creation

### Caching Strategy

- VRS images cached per-frame (foveated rendering)
- Motion vectors cached for ASW
- Pose history uses bounded deque (reprojection)

## Comfort Philosophy

### Motion Sickness is Not Subjective

The comfort system treats motion sickness as a measurable physiological response to vestibular-visual mismatch. The presets (veteran, intermediate, comfortable, maximum) represent empirically-determined thresholds, not user preferences.

### Vignette is Not Decoration

The comfort vignette is a perceptual intervention that:
1. Reduces peripheral motion cues during locomotion
2. Provides a stable visual reference (the dark edges)
3. Triggers automatically from velocity/angular velocity

Users who "don't need it" still benefit from it during sustained play.

### Snap Turn is Accessibility

Snap turn is not a comfort preference; it is the only rotation method that does not trigger vestibular response. Smooth turn is an opt-in for users who have developed VR legs.

## Social XR Design

### Avatars are Identity

The avatar system is not rendering code; it is identity infrastructure:
1. Calibration captures the user's body proportions
2. Face tracking conveys emotional state
3. Hand gestures enable nonverbal communication
4. Personal space enforces physical boundaries

### Voice is Spatial

Voice chat in XR must be positional audio:
- `VoiceChannel.spatial_audio = True`
- `VoiceChannel.distance_falloff` controls attenuation
- `VoiceChannel.max_distance` defines cutoff

Non-spatial voice breaks presence.

## Future Considerations

### Mixed Reality

The rendering and spatial modules support MR, but passthrough integration requires:
1. Camera feed access (platform-specific)
2. Depth estimation for occlusion
3. Lighting estimation for virtual object integration

### Body Tracking

The avatar system estimates feet position procedurally. Full-body tracking would require:
1. Tracker puck support (SteamVR)
2. ML-based body estimation (Quest)
3. IK chain extension to spine and feet

### Hand-to-World Physics

Current hand tracking is visual only. Physics integration would require:
1. Collision volumes per finger bone
2. Force feedback (not available on current hardware)
3. Physics solver integration
