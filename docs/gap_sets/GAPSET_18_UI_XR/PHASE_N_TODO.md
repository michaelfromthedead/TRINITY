# GAPSET_18_UI_XR -- Task Checklist

> **Namespace convention**: UI tasks use `T-UX-{PHASE}.{N}`, XR tasks use `T-XR-{PHASE}.{N}`.
> **Status**: All tasks are `[ ]` not started.
> **Total tasks**: 68 (UI: 27, XR: 41)
> **Effort estimate**: UI: 9-14 weeks, XR: 22-31 weeks (can run in parallel)

---

## UI Tasks

### Phase U1: UI Foundation

**Dependencies**: Foundation runtime (Registry, Tracker, EventLog, Mirror), Trinity ComponentMeta, Trinity descriptor implementations (TrackedDescriptor, ImmutableDescriptor, ComputedDescriptor, TransientDescriptor).

- [ ] **T-UX-1.1** -- Implement `framework/widget.py`: Base Widget component with TrackedDescriptor fields
  - Acceptance: Widget class exists with id (Immutable), parent/children (Tracked), local_x/y, width/height (Tracked), visible/enabled (Tracked), global_x/y (Computed). Parent-child relationship test passes.
  - Dependencies: ComponentMeta, TrackedDescriptor, ComputedDescriptor, ImmutableDescriptor, TransientDescriptor
  - Effort: 3-4 days

- [ ] **T-UX-1.2** -- Implement `framework/events.py`: UI event types with EventMeta
  - Acceptance: ClickEvent, KeyEvent, FocusEvent, HoverEvent, DragEvent exist as EventMeta classes. Events integrate with Foundation EventLog. Test: dispatch event and verify EventLog capture.
  - Dependencies: EventMeta, Foundation EventLog
  - Effort: 2-3 days

- [ ] **T-UX-1.3** -- Implement `framework/coordinate.py`: Coordinate transforms with anchor support
  - Acceptance: Coordinate system supports local-to-global transform via parent chain. Anchor points (top-left, center, stretch) correctly offset position. World/screen coordinate conversion works.
  - Dependencies: Phase U1 framework/widget.py
  - Effort: 2-3 days

- [ ] **T-UX-1.4** -- Implement `framework/focus.py`: Focus management system
  - Acceptance: Focus manager tracks focused widget, supports tab-order navigation, dispatches FocusEvent on focus change, handles keyboard arrow/tab navigation. Widget can be marked @focusable.
  - Dependencies: Phase U1 framework/widget.py, Phase U1 framework/events.py, @focusable decorator
  - Effort: 2-3 days

- [ ] **T-UX-1.5** -- Create `@focusable`, `@ui_layer`, and `@anchor` decorators
  - Acceptance: Decorators exist in trinity/decorators/ui.py or new decorator files. Each follows TAG + REGISTER protocol. Test: apply to widget and verify TAGs are set.
  - Dependencies: Trinity decorator infrastructure
  - Effort: 2-3 days

**Phase U1 total**: 5 tasks, ~12-16 days

---

### Phase U2: UI Widgets

**Dependencies**: Phase U1 (Widget base, events, coordinate, focus). Foundation runtime.

- [ ] **T-UX-2.1** -- Implement `framework/container.py`: Container widget with children management
  - Acceptance: Container extends Widget with children list management. Add/remove children updates parent reference. `@layout` decorator configures direction/gap/padding.
  - Dependencies: Phase U1 framework/widget.py, @layout decorator
  - Effort: 2-3 days

- [ ] **T-UX-2.2** -- Implement primitive widgets: `widgets/primitives/image.py`, `text.py`, `border.py`, `spacer.py`
  - Acceptance: Image displays loaded texture. Text renders with font selection. Border renders filled rectangle with configurable corner radius. Spacer occupies empty space.
  - Dependencies: Phase U1 framework/widget.py
  - Effort: 3-4 days

- [ ] **T-UX-2.3** -- Implement input widgets: `widgets/input/button.py`, `checkbox.py`, `slider.py`
  - Acceptance: Button responds to click with hover/press visual states. Checkbox toggles checked state with animated transition. Slider draggable thumb with RangeDescriptor value clamping.
  - Dependencies: Phase U2 T-UX-2.2, @input_action decorator
  - Effort: 3-4 days

- [ ] **T-UX-2.4** -- Implement input widgets: `widgets/input/text_input.py`, `dropdown.py`
  - Acceptance: TextInput captures keyboard input with cursor, selection, IME composition support. Dropdown shows/hides option list with keyboard/pointer selection.
  - Dependencies: Phase U2 T-UX-2.2, @input_action decorator, Phase U4 localization (for IME)
  - Effort: 4-5 days

- [ ] **T-UX-2.5** -- Implement display widgets: `widgets/display/label.py`, `progress_bar.py`, `icon.py`
  - Acceptance: Label renders text with alignment. ProgressBar shows percentage fill with color gradient. Icon renders scalable icon from sprite atlas.
  - Dependencies: Phase U2 T-UX-2.2
  - Effort: 2-3 days

- [ ] **T-UX-2.6** -- Implement game widgets: `widgets/game/health_bar.py`, `minimap.py`, `inventory_slot.py`, `damage_numbers.py`, `tooltip.py`
  - Acceptance: HealthBar shows resource fill with animated damage flash. Minimap renders top-down world view. InventorySlot supports drag-and-drop. DamageNumbers float and fade. Tooltip appears on hover with delay.
  - Dependencies: Phase U2 T-UX-2.2, T-UX-2.3, @draggable/@droppable/@tooltip decorators
  - Effort: 5-7 days

- [ ] **T-UX-2.7** -- Create `@draggable`, `@droppable`, `@scrollable`, `@tooltip`, `@responsive` decorators
  - Acceptance: All 5 decorators implemented with TAG + HOOK + REGISTER protocol. Draggable fires on_drag_start/on_drag_end hooks. Droppable fires on_drop hook. Scrollable supports scroll direction. Tooltip configures text/delay. Responsive supports breakpoint rules.
  - Dependencies: Trinity decorator infrastructure
  - Effort: 3-4 days

- [ ] **T-UX-2.8** -- Implement decorator stacks in `trinity/decorators/builtin_stacks/ui.py`
  - Acceptance: Stacks exist for interactive_widget, data_bound_widget, game_hud_element, screen, accessible_widget, inventory_slot, localized_text_widget. Each composes multiple decorators correctly.
  - Dependencies: All UI decorators (T-UX-1.5, T-UX-2.7)
  - Effort: 2-3 days

**Phase U2 total**: 8 tasks, ~24-33 days

---

### Phase U3: UI Styling and Themes

**Dependencies**: Phase U1 (Widget base). Trinity ValidatedDescriptor, RangeDescriptor, ChoiceDescriptor.

- [ ] **T-UX-3.1** -- Implement `styling/style.py`: Style properties component
  - Acceptance: StyleProperties component with ValidatedDescriptor chains for colors (hex/named validation), opacity (Range 0-1), border, corner radius (Range 0-100), font properties (size Range 1-200, weight Choice). Style inheritance through widget tree works.
  - Dependencies: Phase U1 framework/widget.py, ValidatedDescriptor, RangeDescriptor, ChoiceDescriptor
  - Effort: 3-4 days

- [ ] **T-UX-3.2** -- Implement `styling/theme.py`: Theme system
  - Acceptance: Light, dark, and high-contrast theme sets defined. Theme switching updates all widget style properties via Tracker subscriptions. Theme resources are collections of named style values. Inheritance and override resolution work.
  - Dependencies: Phase U3 T-UX-3.1, Foundation Tracker
  - Effort: 3-4 days

- [ ] **T-UX-3.3** -- Implement `styling/brush.py` and `styling/color.py`: Brush types and color utilities
  - Acceptance: SolidBrush, LinearGradientBrush, RadialGradientBrush, ImageBrush implement common Brush interface. Color module validates hex (#RRGGBB, #RRGGBBAA), named colors (CSS-named), and rgba() values. sRGB/linear conversion works.
  - Dependencies: None (utility modules)
  - Effort: 2-3 days

**Phase U3 total**: 3 tasks, ~8-11 days

---

### Phase U4: UI Data Binding and Screen Management

**Dependencies**: Phase U2 (Widgets), Phase U3 (Styling). Trinity ObservableDescriptor, BoundDescriptor.

- [ ] **T-UX-4.1** -- Implement `binding/binding.py`: Data binding system
  - Acceptance: One-way binding via ObservableDescriptor propagates model changes to widget. Two-way binding via BoundDescriptor propagates widget changes back to model. One-time binding for static values. Multiple widgets can bind to same model field.
  - Dependencies: Phase U1 framework/widget.py, ObservableDescriptor, BoundDescriptor, Foundation Tracker
  - Effort: 4-5 days

- [ ] **T-UX-4.2** -- Implement `binding/converter.py` and `binding/validation.py`: Value converters and input validation
  - Acceptance: Converters for float-to-percentage-string, float-to-color, int-to-string implemented. Converters are registered by type pair. Validation runs before model update and rejects invalid values with error callback.
  - Dependencies: Phase U4 T-UX-4.1, ValidatedDescriptor
  - Effort: 2-3 days

- [ ] **T-UX-4.3** -- Implement `screens/screen.py` and `screens/screen_stack.py`: Screen management
  - Acceptance: Screen base class with enter/exit lifecycle hooks. Screen stack supports push, pop, and replace operations. @state_machine decorator defines valid screen transitions. Invalid navigation raises error.
  - Dependencies: Phase U1 framework/widget.py, @state_machine, @on_enter, @on_exit decorators
  - Effort: 3-4 days

- [ ] **T-UX-4.4** -- Implement `screens/transitions.py`: Screen transitions
  - Acceptance: Fade, slide (left/right/up/down), and zoom transitions between screens. Configurable duration and easing function via @tween decorator. Transition animation test with visual verification.
  - Dependencies: Phase U4 T-UX-4.3, @tween decorator
  - Effort: 2-3 days

**Phase U4 total**: 4 tasks, ~11-15 days

---

### Phase U5: UI Integration

**Dependencies**: All UI phases U1-U4. Foundation runtime (Tracker, EventLog, Mirror).

- [ ] **T-UX-5.1** -- Wire TrackedDescriptor changes to layout invalidation
  - Acceptance: Size/position changes on any widget automatically trigger parent layout recalculation. Dirty flag propagation is bounded and does not cause infinite loops. Test: change width, verify children repositioned within frame.
  - Dependencies: Phase U1 framework/widget.py, Phase U2 framework/container.py, Foundation Tracker
  - Effort: 2-3 days

- [ ] **T-UX-5.2** -- Wire ObservableDescriptor to UI re-render
  - Acceptance: Model component update (e.g., PlayerStats.health change) propagates through ObservableDescriptor to bound Widget field update. Widget field change triggers re-render through Tracked descriptor chain.
  - Dependencies: Phase U4 T-UX-4.1, Foundation Tracker
  - Effort: 2-3 days

- [ ] **T-UX-5.3** -- Wire Foundation Tracker for undo/redo in editor
  - Acceptance: Widget state changes are recorded by Tracker in frame-grouped batches. Undo restores previous state. Redo reapplies. Works for property changes, widget add/remove, layout changes.
  - Dependencies: Foundation Tracker, Phase U1-U4
  - Effort: 2-3 days

- [ ] **T-UX-5.4** -- Wire Foundation Mirror for UI inspector
  - Acceptance: Mirror system can inspect widget tree, all widget fields (with descriptor metadata), binding state, and style property values. Inspector output is human-readable.
  - Dependencies: Foundation Mirror, Phase U1-U4
  - Effort: 1-2 days

- [ ] **T-UX-5.5** -- Wire Foundation EventLog for UI input events
  - Acceptance: All UI input events (click, key, focus, hover, drag) are recorded in EventLog. Events include widget ID, timestamp, event type, and payload. EventLog replay reproduces input sequence.
  - Dependencies: Foundation EventLog, Phase U1 framework/events.py
  - Effort: 1-2 days

**Phase U5 total**: 5 tasks, ~8-13 days

---

### UI Total: 25 tasks, 9-14 weeks

---

## XR Tasks

### Phase X1: XR Runtime Foundation

**Dependencies**: Foundation runtime (Registry, Tracker, EventLog, Mirror), Trinity ResourceMeta, Trinity SystemMeta, ComponentMeta. OpenXR SDK (external dependency).

- [ ] **T-XR-1.1** -- Implement `runtime/xr_runtime.py`: XR runtime abstraction layer
  - Acceptance: XRRuntimeState resource with ImmutableDescriptor fields for runtime info, TrackedDescriptor + ObservableDescriptor session state, device capabilities, display specs. Runtime abstraction interface defines backends contract.
  - Dependencies: ResourceMeta, ImmutableDescriptor, TrackedDescriptor, ObservableDescriptor, RangeDescriptor
  - Effort: 3-4 days

- [ ] **T-XR-1.2** -- Implement `runtime/openxr.py`: OpenXR backend
  - Acceptance: OpenXR 1.0+ backend implementing runtime abstraction. Instance creation, system selection, session lifecycle (idle/ready/running/stopping), reference space enumeration (view/stage/local). Supports HMD pose polling at 250Hz+.
  - Dependencies: Phase X1 T-XR-1.1, OpenXR SDK
  - Effort: 5-7 days

- [ ] **T-XR-1.3** -- Implement `runtime/capabilities.py`: Device capability detection
  - Acceptance: Queries device capabilities: supports_hand_tracking, supports_eye_tracking, supports_passthrough, supports_spatial_mesh, max_controllers, display_refresh_rate, field_of_view. Capabilities cached ImmutableDescriptor after init.
  - Dependencies: Phase X1 T-XR-1.2
  - Effort: 2-3 days

- [ ] **T-XR-1.4** -- Implement `runtime/session.py`: XR session state machine with StateMeta
  - Acceptance: Session states (idle/ready/running/stopping) with valid transitions. @state_machine configuration. @on_enter/@on_exit hooks for each state. Session state changes fire ObservableDescriptor to UI and game systems.
  - Dependencies: Phase X1 T-XR-1.1, StateMeta, @state_machine, @on_enter, @on_exit
  - Effort: 2-3 days

**Phase X1 total**: 4 tasks, ~12-17 days

---

### Phase X2: XR Input Tracking

**Dependencies**: Phase X1 (XR Runtime). Trinity PredictedDescriptor, InterpolatedDescriptor, AtomicDescriptor, BatchedDescriptor, ExpiringDescriptor. Phase X1 T-XR-1.2 (OpenXR backend for device poll).

- [ ] **T-XR-2.1** -- Implement `input/hmd.py`: HMD pose tracking with prediction
  - Acceptance: HMDPose component with PredictedDescriptor -> TrackedDescriptor -> AtomicDescriptor position/orientation. Linear/angular velocity for prediction. Tracking state machine (unknown/tracking/limited/lost/disabled). Per-eye view matrices as ComputedDescriptor. Tracking confidence with ExpiringDescriptor (0.5s TTL).
  - Dependencies: Phase X1 T-XR-1.2, PredictedDescriptor, TrackedDescriptor, AtomicDescriptor, ComputedDescriptor, ExpiringDescriptor, StateMeta
  - Effort: 4-5 days

- [ ] **T-XR-2.2** -- Implement `input/controller.py`: Controller input component
  - Acceptance: XRController component with Immutable hand ID, PredictedDescriptor + TrackedDescriptor grip/aim poses, Tracked -> Validated -> Range analog inputs (trigger 0-1, grip 0-1, thumbstick -1 to 1), bool digital inputs (buttons, clicks), touch sensing. Haptic feedback as write-only TransientDescriptor.
  - Dependencies: Phase X1 T-XR-1.2, PredictedDescriptor, TrackedDescriptor, ValidatedDescriptor, RangeDescriptor, ImmutableDescriptor, TransientDescriptor
  - Effort: 4-5 days

- [ ] **T-XR-2.3** -- Implement `input/bindings.py`: Input action binding system
  - Acceptance: Input actions bound to controller inputs via @input_action and @input_axis decorators. XR-specific actions: xr_grab, xr_trigger, xr_move, xr_turn, xr_teleport, xr_menu. Default bindings for left/right controllers. Input action dispatch through Foundation EventLog.
  - Dependencies: Phase X2 T-XR-2.2, @input_action, @input_axis, Foundation EventLog
  - Effort: 2-3 days

- [ ] **T-XR-2.4** -- Implement `input/haptics.py`: Haptic feedback system
  - Acceptance: Haptic system reads TransientDescriptor amplitude/duration/frequency from controllers and sends to OpenXR haptic feedback. Supports pulse and continuous vibration modes. Configurable amplitude envelope.
  - Dependencies: Phase X1 T-XR-1.2, Phase X2 T-XR-2.2
  - Effort: 2-3 days

- [ ] **T-XR-2.5** -- Implement `input/hand_tracking.py`: 26-joint hand tracking
  - Acceptance: HandTracking component with InterpolatedDescriptor -> BatchedDescriptor -> TrackedDescriptor joint positions/orientations/radii. Gesture recognition (pinch, point, fist, open, thumbs up). Pinch detection with strength 0-1. Joint update rate >30Hz with smooth interpolation between samples.
  - Dependencies: Phase X2 T-XR-2.2, InterpolatedDescriptor, BatchedDescriptor, TrackedDescriptor
  - Effort: 4-5 days

- [ ] **T-XR-2.6** -- Implement `input/eye_tracking.py`: Eye tracking and gaze system
  - Acceptance: EyeTrackingData component with gaze origin/direction, per-eye pupil data (position, diameter, openness). Fixation detection with point/duration. Saccade detection. Calibration state machine. Confidence tracking.
  - Dependencies: Phase X1 T-XR-1.2, TrackedDescriptor, RangeDescriptor
  - Effort: 3-4 days

- [ ] **T-XR-2.7** -- Create `@xr_tracked`, `@xr_controller`, `@xr_hand` decorators
  - Acceptance: Decorators follow TAG + REGISTER protocol. @xr_tracked marks tracking_type and tracking_space. @xr_controller configures hand and controller_type. @xr_hand configures gesture_recognition.
  - Dependencies: Trinity decorator infrastructure
  - Effort: 2-3 days

**Phase X2 total**: 7 tasks, ~21-28 days

---

### Phase X3: XR Rendering

**Dependencies**: Phase X1 (XR Runtime). S14 wgpu RHI backend. S1 Frame Graph backend. S15 Rust math library. Phase X2 T-XR-2.1 (HMD pose for view matrices).

- [ ] **T-XR-3.1** -- Implement `rendering/stereo.py`: Stereo rendering pipeline
  - Acceptance: Per-eye view matrices computed from HMD pose + IPD via Rust math library. Multiview rendering as primary mode (VK_KHR_multiview). Instanced fallback. Sequential last resort. Render scale configuration (0.5x-2.0x). Per-eye projection matrices from FOV. Frame time within 11.1ms @90Hz.
  - Dependencies: S14 wgpu RHI, S1 Frame Graph, S15 Rust math, Phase X2 T-XR-2.1
  - Effort: 5-7 days

- [ ] **T-XR-3.2** -- Implement `rendering/foveated.py`: Foveated rendering
  - Acceptance: Fixed foveated rendering with configurable level 0-4. Dynamic foveated rendering with eye-tracking gaze follow (requires T-XR-2.6). Foveation regions: fovea (full), parafoveal (reduced), peripheral (lowest). Configurable inner/outer radius per region.
  - Dependencies: Phase X3 T-XR-3.1, Phase X2 T-XR-2.6 (for dynamic), @foveated_region decorator
  - Effort: 4-5 days

- [ ] **T-XR-3.3** -- Implement `rendering/reprojection.py`: ATW/ASW reprojection
  - Acceptance: ATW warps last rendered frame to latest predicted HMD pose. ASW generates motion vectors and interpolates new frame. Reprojection mode configurable: none/atw/asw. Runs as post-compositor pass. Pose-to-photon latency <20ms target.
  - Dependencies: Phase X3 T-XR-3.1, Phase X2 T-XR-2.1 (predicted pose), S14 wgpu RHI
  - Effort: 5-7 days

- [ ] **T-XR-3.4** -- Implement `rendering/compositor.py`: Compositor layer management
  - Acceptance: Multi-layer compositor with scene layer, UI overlay layer(s), passthrough camera layer for AR. Layer ordering and blending. Each layer has independent resolution and refresh rate. Quad layer for UI panels.
  - Dependencies: Phase X3 T-XR-3.1, S14 wgpu RHI
  - Effort: 3-4 days

- [ ] **T-XR-3.5** -- Implement `rendering/hidden_area.py`: Hidden area mesh optimization
  - Acceptance: Hidden area mesh loaded from device profile. Mesh culls pixels not visible through lens. GPU time saved proportional to hidden area (typically 15-25% of pixels). Per-eye mesh with TrackedDescriptor for updates.
  - Dependencies: Phase X3 T-XR-3.1, device profile data
  - Effort: 2-3 days

- [ ] **T-XR-3.6** -- Create `@foveated_region` decorator
  - Acceptance: @foveated_region decorator configures region_type (fovea/parafoveal/peripheral), quality_level (0-4), and inner/outer radius. TAG + REGISTER protocol.
  - Dependencies: Trinity decorator infrastructure
  - Effort: 1 day

**Phase X3 total**: 6 tasks, ~19-26 days

---

### Phase X4: XR Interaction

**Dependencies**: Phase X1 (XR Runtime), Phase X2 (Input tracking), Phase X5 (Spatial for hit testing).

- [ ] **T-XR-4.1** -- Implement `interaction/interactable.py`: Base interactable component
  - Acceptance: XRInteractable component with TrackedDescriptor + ObservableDescriptor for hover/select/grab state. Interactor reference tracking. @on_change hooks for state transitions. Configurable interaction_types list.
  - Dependencies: Phase X1 T-XR-1.1, @xr_interactable decorator, @on_change
  - Effort: 2-3 days

- [ ] **T-XR-4.2** -- Implement `interaction/grabbable.py`: Grab mechanics
  - Acceptance: XRGrabbable extends XRInteractable. Physics grab connects via joint. Kinematic grab parents to hand. Attach transform/rotation configuration. Two-handed grab with secondary grab point. Grab state machine (idle/hovered/selected/grabbed/released).
  - Dependencies: Phase X4 T-XR-4.1, @xr_grabbable decorator
  - Effort: 4-5 days

- [ ] **T-XR-4.3** -- Implement `interaction/socket.py`: Snap socket system
  - Acceptance: XRSocket with socket_type, accepted_tags, is_occupied state. Snap detection within snap_distance. Auto-snap on release within proximity. Attached_entity tracking. Holster/belt socket scenario works.
  - Dependencies: Phase X4 T-XR-4.2, @xr_socket decorator
  - Effort: 2-3 days

- [ ] **T-XR-4.4** -- Implement `interaction/ray_interactor.py`: Ray-based interaction
  - Acceptance: Parabolic arc ray from controller aim pose. Hit detection against interactable objects. Visual arc rendering with bend. Configurable arc length (default 10m). Hover/select/grab events dispatched to interactable components.
  - Dependencies: Phase X4 T-XR-4.1, Phase X2 T-XR-2.2, Phase X5 spatial for spatial queries
  - Effort: 3-4 days

- [ ] **T-XR-4.5** -- Implement `interaction/direct_interactor.py`: Direct/poke interaction
  - Acceptance: Hand joint proximity check against interactable colliders. Index fingertip as primary touch point. Physical button press with press_depth tracking. Haptic feedback on full press.
  - Dependencies: Phase X4 T-XR-4.1, Phase X2 T-XR-2.5 (hand tracking)
  - Effort: 3-4 days

- [ ] **T-XR-4.6** -- Implement `interaction/gaze_interactor.py`: Gaze-based interaction
  - Acceptance: Eye tracking gaze ray for selection. Configurable dwell time (default 0.5s). Progress indicator during dwell. Gaze as least-preferred interaction, disabled by default.
  - Dependencies: Phase X4 T-XR-4.1, Phase X2 T-XR-2.6 (eye tracking)
  - Effort: 2-3 days

- [ ] **T-XR-4.7** -- Create `@xr_interactable`, `@xr_grabbable`, `@xr_socket`, `@xr_haptic` decorators
  - Acceptance: All 4 decorators follow TAG + HOOK + REGISTER protocol. @xr_interactable configures interaction_types and grab_points. @xr_grabbable configures grab_type and attach_transform. @xr_socket configures socket_type and accepted_tags. @xr_haptic configures amplitude/duration/frequency.
  - Dependencies: Trinity decorator infrastructure
  - Effort: 2-3 days

**Phase X4 total**: 7 tasks, ~18-25 days

---

### Phase X5: XR Spatial Understanding and AR

**Dependencies**: Phase X1 (XR Runtime). OpenXR spatial extensions.

- [ ] **T-XR-5.1** -- Implement `spatial/anchor.py`: Spatial anchor system
  - Acceptance: SpatialAnchor component with ImmutableDescriptor anchor_id, TrackedDescriptor position/orientation (runtime-refined). Persistence support (local and cloud anchor UUIDs). Tracking state management (unknown/tracking/lost). Anchor resolve for cloud anchors.
  - Dependencies: Phase X1 T-XR-1.2, @spatial_anchor decorator
  - Effort: 3-4 days

- [ ] **T-XR-5.2** -- Implement `spatial/plane_detection.py`: Plane detection
  - Acceptance: PlaneDetection component with classification (floor/ceiling/wall/table/seat), center, normal, polygon bounds, width/height. Tracked for runtime updates. is_tracked ObservableDescriptor for state changes. Filtering by type.
  - Dependencies: Phase X1 T-XR-1.2 (OpenXR plane detection extension), TrackedDescriptor
  - Effort: 3-4 days

- [ ] **T-XR-5.3** -- Implement `spatial/mesh_mapping.py`: Spatial mesh generation
  - Acceptance: Real-time environment mesh with incremental vertex updates. Per-vertex confidence weighting. Mesh stored for physics queries and occlusion rendering. Configurable update rate (default 1Hz). Mesh LOD levels for performance.
  - Dependencies: Phase X1 T-XR-1.2 (OpenXR mesh detection extension)
  - Effort: 4-5 days

- [ ] **T-XR-5.4** -- Implement `spatial/image_tracking.py` and `object_tracking.py`: AR tracking
  - Acceptance: ImageTarget component with reference_image_id, physical_size, is_tracked state, tracked position/orientation, tracked_size. Object tracking for 3D reference objects. Tracking mode: one-shot or continuous.
  - Dependencies: Phase X1 T-XR-1.2, @ar_trackable decorator
  - Effort: 3-4 days

- [ ] **T-XR-5.5** -- Create `@spatial_anchor` and `@ar_trackable` decorators
  - Acceptance: @spatial_anchor configures anchor_type (local/cloud) and persistent. @ar_trackable configures trackable_type (image/object) and reference_id. TAG + REGISTER protocol.
  - Dependencies: Trinity decorator infrastructure
  - Effort: 1-2 days

**Phase X5 total**: 5 tasks, ~14-19 days

---

### Phase X6: XR Avatars

**Dependencies**: Phase X1 (XR Runtime), Phase X2 (Input tracking for IK targets), Phase X4 (Interaction for hand pose).

- [ ] **T-XR-6.1** -- Implement `avatars/avatar.py`: Full-body avatar component
  - Acceptance: XRAvatar component with IK targets from HMD (head) and controllers/hands. ComputedDescriptor for torso estimation. Visibility flags for self/others. Calibration data (height, arm_span). Avatar skeleton updates at 90Hz for body.
  - Dependencies: Phase X1 T-XR-1.1, @xr_avatar decorator, ComputedDescriptor
  - Effort: 3-4 days

- [ ] **T-XR-6.2** -- Implement `avatars/ik_solver.py`: FABRIK/CCD IK solver
  - Acceptance: FABRIK solver for full-body IK (head, arms, spine). Converges in 3-5 iterations. Bone constraint support (joint angle limits). CCD fallback for fingers. Solve time <0.1ms per chain at 90Hz. Chain definitions for upper body (head + arms), fingers.
  - Dependencies: Phase X6 T-XR-6.1, Rust math library (S15) for vector/quaternion operations
  - Effort: 5-7 days

- [ ] **T-XR-6.3** -- Implement `avatars/hand_animator.py`: Hand/finger animation
  - Acceptance: Finger curl values (thumb through pinky, 0-1 range) driven by controller grip/trigger or hand tracking joint data. Blend tree for grip postures (open/point/fist/pinch). Hand display modes: hand, controller, tool. Held tool display at hand.
  - Dependencies: Phase X6 T-XR-6.2, Phase X2 T-XR-2.5 (hand tracking), @blend_tree decorator
  - Effort: 3-4 days

- [ ] **T-XR-6.4** -- Implement `avatars/face_tracking.py`: Face/expression tracking
  - Acceptance: Face tracking from eye tracking cameras. Blend shape weights for FACS-compatible expressions (jaw open, brow raise, smile, etc.). Configurable blend shape count per avatar. Update rate 30Hz.
  - Dependencies: Phase X2 T-XR-2.6 (eye tracking), Phase X6 T-XR-6.1
  - Effort: 3-4 days

- [ ] **T-XR-6.5** -- Implement `avatars/calibration.py`: Avatar calibration system
  - Acceptance: T-pose/A-pose calibration at session start. Measures height and arm span. Applies scale to IK skeleton proportions. Calibration persistence across sessions. Manual recalibration trigger. Calibration accuracy within 1cm.
  - Dependencies: Phase X6 T-XR-6.1, Foundation Serializer (for persistence)
  - Effort: 2-3 days

- [ ] **T-XR-6.6** -- Create `@xr_avatar`, `@xr_ik_target` decorators
  - Acceptance: @xr_avatar configures ik_enabled and network_sync. @xr_ik_target configures target_type and bone_chain. TAG + REGISTER protocol.
  - Dependencies: Trinity decorator infrastructure
  - Effort: 1-2 days

**Phase X6 total**: 6 tasks, ~17-24 days

---

### Phase X7: XR Locomotion and Comfort

**Dependencies**: Phase X1 (XR Runtime), Phase X2 (Controller input for movement), Phase X4 (Interaction for grab-to-climb).

- [ ] **T-XR-7.1** -- Implement `locomotion/teleport.py`: Teleport locomotion
  - Acceptance: Parabolic arc projection from controller aim. Arc visualization with configurable gravity (default -9.8). Landing point validation: surface normal, clear space height. Fade on teleport (configurable duration, default 0.1s). Configurable max distance (default 10m). Aim valid/invalid visual feedback.
  - Dependencies: Phase X2 T-XR-2.2, @xr_teleport_area decorator, @xr_locomotion decorator
  - Effort: 4-5 days

- [ ] **T-XR-7.2** -- Implement `locomotion/smooth.py`: Smooth locomotion
  - Acceptance: Head-relative smooth movement with configurable speed (default 3 m/s). Strafe support. Snap turn with configurable angle (15-90, default 45). Smooth turn with configurable speed (30-180 deg/s, default 90). Movement from thumbstick input.
  - Dependencies: Phase X2 T-XR-2.2 (controller thumbstick), @xr_locomotion decorator
  - Effort: 3-4 days

- [ ] **T-XR-7.3** -- Implement `locomotion/climbing.py`: Climbing locomotion
  - Acceptance: Grab-to-climb on climbable surfaces. Two-hand grab enters climbing mode. Movement follows hand pull/push. Release mechanism (release both hands or release one + press button). Climbable surfaces tagged via @xr_interactable configuration.
  - Dependencies: Phase X4 T-XR-4.2 (grab mechanics), Phase X6 T-XR-6.1 (avatar arm IK), @xr_locomotion decorator
  - Effort: 3-4 days

- [ ] **T-XR-7.4** -- Implement `locomotion/comfort.py`: Comfort settings and vignette
  - Acceptance: Vignette darkens periphery proportionally to linear/angular velocity. Fully transparent at rest, full intensity at 30+ deg/s angular velocity. Configurable intensity (default 0-1 0.5). Seated mode with configurable height offset. Snap turn enabled/disabled toggle. All comfort settings as TrackedDescriptor in XRComfortSettings resource.
  - Dependencies: Phase X1 T-XR-1.1 (XRComfortSettings resource), @xr_comfort decorator
  - Effort: 2-3 days

- [ ] **T-XR-7.5** -- Create `@xr_teleport_area`, `@xr_locomotion`, `@xr_comfort` decorators
  - Acceptance: @xr_teleport_area configures teleport_type (instant/fade). @xr_locomotion configures locomotion_type (teleport/smooth/climb) and speed. @xr_comfort configures comfort_type and settings. TAG + REGISTER protocol.
  - Dependencies: Trinity decorator infrastructure
  - Effort: 2-3 days

**Phase X7 total**: 5 tasks, ~14-19 days

---

### Phase X8: XR UI

**Dependencies**: Phase U1 (UI Widget base), Phase X1 (XR Runtime), Phase X4 (Interaction for ray/poke on panels).

- [ ] **T-XR-8.1** -- Implement XR UI panel rendering: `ui/panel.py`
  - Acceptance: XRUIPanel renders UI widget tree to render target texture. Quad display in 3D scene. Panel types: world-space, head-locked, hand-attached, wrist. Configurable width/height (meters), pixels_per_meter. Curved panel support with configurable radius (default 2m). Billboard option.
  - Dependencies: Phase U1 framework/widget.py (for widget tree), Phase X3 T-XR-3.1 (for render target), @xr_ui_panel decorator
  - Effort: 4-5 days

- [ ] **T-XR-8.2** -- Implement XR button: `ui/button.py`
  - Acceptance: XRButton with label, hover/press state, press depth for physical button feel. Haptic feedback on press. Interaction through ray, poke, or gaze. @on_change hooks for state transitions.
  - Dependencies: Phase X8 T-XR-8.1, Phase X4 interaction system, @xr_haptic
  - Effort: 2-3 days

- [ ] **T-XR-8.3** -- Implement XR slider and keyboard: `ui/slider.py`, `ui/keyboard.py`
  - Acceptance: XRSlider with draggable thumb, range value, snap points. VirtualKeyboard with key press events, text input compositor integration, multi-layout support (QWERTY, numeric, symbols).
  - Dependencies: Phase X8 T-XR-8.1, Phase X4 interaction system
  - Effort: 3-4 days

- [ ] **T-XR-8.4** -- Implement `ui/wrist_ui.py`: Wrist-attached UI
  - Acceptance: WristUI panel attached to virtual forearm. Visible on wrist turn toward face. Quick action buttons, notifications, mini-map display. Configurable layout per application.
  - Dependencies: Phase X8 T-XR-8.1, Phase X6 T-XR-6.1 (avatar for arm transform)
  - Effort: 2-3 days

- [ ] **T-XR-8.5** -- Create `@xr_ui_panel` and `@passthrough_layer` decorators
  - Acceptance: @xr_ui_panel configures panel_type (world/head_locked/hand_attached/wrist) and interaction_mode (ray/poke/gaze). @passthrough_layer configures blend_mode and opacity for AR. TAG + REGISTER protocol.
  - Dependencies: Trinity decorator infrastructure
  - Effort: 1-2 days

- [ ] **T-XR-8.6** -- Implement decorator stacks in `trinity/decorators/builtin_stacks/xr.py`
  - Acceptance: Stacks for tracked_xr_device, xr_controller_component, hand_tracking_component, xr_interactable_object, xr_grabbable_object, xr_ui_element, teleport_destination, ar_spatial_anchor, xr_avatar_component, multiplayer_xr_avatar, xr_comfort_locomotion, full_xr_player, xr_weapon, xr_tool, ar_furniture_placement.
  - Dependencies: All XR decorators (T-XR-2.7, T-XR-3.6, T-XR-4.7, T-XR-5.5, T-XR-6.6, T-XR-7.5, T-XR-8.5)
  - Effort: 3-4 days

**Phase X8 total**: 6 tasks, ~15-21 days

---

## Summary

| Namespace | Phases | Tasks | Effort (weeks) |
|-----------|--------|-------|----------------|
| T-UX (UI) | U1-U5 | 25 | 9-14 |
| T-XR (XR) | X1-X8 | 41 | 22-31 |
| **Total** | **13 phases** | **66 tasks** | **--** |

**Parallel execution plan**: UI (U1-U5) and XR (X1-X8) are architecturally independent and can be implemented concurrently, reducing wall-clock time to approximately 22-31 weeks limited by the larger XR subsystem.

**Shared dependency blocking**: XR Phase X3 (stereo rendering) depends on S14 (wgpu RHI), S1 (frame graph), and S15 (Rust math library), which are part of gap sets for rendering and core systems. UI does not depend on wgpu directly but requires Foundation runtime primitives.

**Decorator count**: 8 new UI decorators (T-UX-1.5, T-UX-2.7) + 17 new XR decorators (T-XR-2.7, T-XR-3.6, T-XR-4.7, T-XR-5.5, T-XR-6.6, T-XR-7.5, T-XR-8.5) = 25 total new decorators.
