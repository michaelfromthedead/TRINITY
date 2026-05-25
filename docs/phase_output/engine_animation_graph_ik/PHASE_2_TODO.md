# PHASE 2 TODO: State Machines and Blend Trees

**Phase**: 2 of 4
**Focus**: Animation Control Structures

---

## Tasks

### T-AG-2.1: AnimationState Class

**Priority**: P0 (Critical)
**Estimate**: 2 hours
**Dependencies**: Phase 1 complete

**Acceptance Criteria:**
- [ ] AnimationState dataclass with clip/graph reference
- [ ] Motion handling (loop, once, ping-pong)
- [ ] Speed multiplier support
- [ ] State entry/exit callbacks
- [ ] Current time tracking

---

### T-AG-2.2: TransitionCondition System

**Priority**: P0 (Critical)
**Estimate**: 3 hours
**Dependencies**: T-AG-2.1

**Acceptance Criteria:**
- [ ] ConditionOperator enum (8 operators)
- [ ] TransitionCondition dataclass
- [ ] evaluate(context) method
- [ ] Support for trigger parameters (one-shot)
- [ ] Support for exit_time conditions
- [ ] Parameter type checking

---

### T-AG-2.3: Blend Curves (6 types)

**Priority**: P1 (High)
**Estimate**: 2 hours
**Dependencies**: None

**Acceptance Criteria:**
- [ ] BlendCurve enum
- [ ] evaluate_curve(curve, t) function
- [ ] LINEAR: t
- [ ] EASE_IN: t^2
- [ ] EASE_OUT: 1 - (1-t)^2
- [ ] EASE_IN_OUT: Hermite interpolation
- [ ] SMOOTH_STEP: 3t^2 - 2t^3
- [ ] SMOOTHER_STEP: t^3 * (t * (t * 6 - 15) + 10)

---

### T-AG-2.4: StateTransition Class

**Priority**: P0 (Critical)
**Estimate**: 2 hours
**Dependencies**: T-AG-2.2, T-AG-2.3

**Acceptance Criteria:**
- [ ] StateTransition dataclass
- [ ] Source and target state references
- [ ] Condition list (all must pass)
- [ ] Duration (fixed or percentage)
- [ ] Blend curve selection
- [ ] Priority for multiple valid transitions
- [ ] Interruption mode (can/cannot interrupt)

---

### T-AG-2.5: StateMachine Core

**Priority**: P0 (Critical)
**Estimate**: 4 hours
**Dependencies**: T-AG-2.4

**Acceptance Criteria:**
- [ ] StateMachine extends AnimationNode
- [ ] State registry
- [ ] Transition registry
- [ ] Any-state transitions
- [ ] Current state tracking
- [ ] evaluate(context) with transition checking
- [ ] force_state(name) for debugging

---

### T-AG-2.6: Transition Blending

**Priority**: P0 (Critical)
**Estimate**: 3 hours
**Dependencies**: T-AG-2.5

**Acceptance Criteria:**
- [ ] TransitionData for in-progress transitions
- [ ] Pose blending during transition
- [ ] Curve application to blend weight
- [ ] Transition completion detection
- [ ] Sync mode handling (none, normalized, proportional)

---

### T-AG-2.7: StateMachineBuilder

**Priority**: P1 (High)
**Estimate**: 2 hours
**Dependencies**: T-AG-2.5

**Acceptance Criteria:**
- [ ] Fluent builder API
- [ ] add_state(name, source)
- [ ] add_transition(source, target, condition, ...)
- [ ] add_any_state_transition(target, condition)
- [ ] set_initial(state_name)
- [ ] build() with validation
- [ ] Clear error messages for invalid configurations

---

### T-AG-2.8: BlendTree1D

**Priority**: P0 (Critical)
**Estimate**: 3 hours
**Dependencies**: Phase 1 complete

**Acceptance Criteria:**
- [ ] BlendTree1D extends AnimationNode
- [ ] BlendSample1D dataclass
- [ ] Parameter-based sample selection
- [ ] Linear interpolation between samples
- [ ] Gradient bands option for smooth transitions
- [ ] Handle edge cases (before first, after last)

---

### T-AG-2.9: BlendTree2D Base

**Priority**: P0 (Critical)
**Estimate**: 3 hours
**Dependencies**: T-AG-2.8

**Acceptance Criteria:**
- [ ] BlendTree2D extends AnimationNode
- [ ] BlendSample2D dataclass
- [ ] BlendMode2D enum (4 modes)
- [ ] Dual parameter support
- [ ] Mode-specific evaluation dispatch

---

### T-AG-2.10: Cartesian Blend Mode

**Priority**: P1 (High)
**Estimate**: 2 hours
**Dependencies**: T-AG-2.9

**Acceptance Criteria:**
- [ ] Grid-based sample placement
- [ ] Bilinear interpolation
- [ ] Handle non-rectangular sample sets
- [ ] Weight normalization

---

### T-AG-2.11: Polar Blend Mode

**Priority**: P1 (High)
**Estimate**: 2 hours
**Dependencies**: T-AG-2.9

**Acceptance Criteria:**
- [ ] Direction/magnitude interpretation
- [ ] Angular interpolation (handle 360 wrap)
- [ ] Radial weight calculation
- [ ] Center sample handling

---

### T-AG-2.12: Delaunay Triangulation

**Priority**: P1 (High)
**Estimate**: 4 hours
**Dependencies**: T-AG-2.9

**Acceptance Criteria:**
- [ ] Bowyer-Watson algorithm
- [ ] Super-triangle creation
- [ ] Circumcircle test
- [ ] Incremental point insertion
- [ ] Polygon boundary finding
- [ ] Super-triangle removal
- [ ] Cache triangulation for performance

---

### T-AG-2.13: Barycentric Interpolation

**Priority**: P1 (High)
**Estimate**: 2 hours
**Dependencies**: T-AG-2.12

**Acceptance Criteria:**
- [ ] get_barycentric(point, triangle) function
- [ ] Proper dot product computation
- [ ] Numerical stability (clamping)
- [ ] Handle degenerate triangles
- [ ] Point-in-triangle test

---

### T-AG-2.14: Freeform Blend Modes

**Priority**: P1 (High)
**Estimate**: 3 hours
**Dependencies**: T-AG-2.12, T-AG-2.13

**Acceptance Criteria:**
- [ ] FREEFORM_DIRECTIONAL mode
- [ ] FREEFORM_CARTESIAN mode (Delaunay)
- [ ] Triangle lookup for point
- [ ] Barycentric weight calculation
- [ ] Smooth blending at triangle edges

---

### T-AG-2.15: Blend Nodes (10 types)

**Priority**: P0 (Critical)
**Estimate**: 4 hours
**Dependencies**: Phase 1 complete

**Acceptance Criteria:**
- [ ] ClipNode with loop modes
- [ ] BlendNode for binary blending
- [ ] AdditiveNode for additive blending
- [ ] LayerNode with mask support
- [ ] MirrorNode for left/right mirroring
- [ ] TimeScaleNode for speed control
- [ ] PoseCacheNode for pose reuse
- [ ] SelectNode for conditional selection
- [ ] LoopNode for loop control
- [ ] SubGraphNode for nested graphs

---

### T-AG-2.16: LayerStack System

**Priority**: P1 (High)
**Estimate**: 3 hours
**Dependencies**: T-AG-2.15

**Acceptance Criteria:**
- [ ] LayerStack class
- [ ] AnimationLayer dataclass
- [ ] LayerBlendMode enum (OVERRIDE, ADDITIVE, MULTIPLY)
- [ ] Layer weight support
- [ ] Bone mask integration
- [ ] evaluate(context) with layer composition

---

### T-AG-2.17: LayerStackBuilder

**Priority**: P2 (Medium)
**Estimate**: 2 hours
**Dependencies**: T-AG-2.16

**Acceptance Criteria:**
- [ ] Fluent builder API
- [ ] set_base(source)
- [ ] add_layer(source, mask, mode, weight)
- [ ] build() with validation
- [ ] Clear error messages

---

## Task Summary

| Task ID | Description | Priority | Est. Hours | Dependencies |
|---------|-------------|----------|------------|--------------|
| T-AG-2.1 | AnimationState | P0 | 2 | Phase 1 |
| T-AG-2.2 | TransitionCondition | P0 | 3 | T-AG-2.1 |
| T-AG-2.3 | Blend Curves | P1 | 2 | None |
| T-AG-2.4 | StateTransition | P0 | 2 | T-AG-2.2, T-AG-2.3 |
| T-AG-2.5 | StateMachine Core | P0 | 4 | T-AG-2.4 |
| T-AG-2.6 | Transition Blending | P0 | 3 | T-AG-2.5 |
| T-AG-2.7 | StateMachineBuilder | P1 | 2 | T-AG-2.5 |
| T-AG-2.8 | BlendTree1D | P0 | 3 | Phase 1 |
| T-AG-2.9 | BlendTree2D Base | P0 | 3 | T-AG-2.8 |
| T-AG-2.10 | Cartesian Mode | P1 | 2 | T-AG-2.9 |
| T-AG-2.11 | Polar Mode | P1 | 2 | T-AG-2.9 |
| T-AG-2.12 | Delaunay Triangulation | P1 | 4 | T-AG-2.9 |
| T-AG-2.13 | Barycentric Interpolation | P1 | 2 | T-AG-2.12 |
| T-AG-2.14 | Freeform Modes | P1 | 3 | T-AG-2.12, T-AG-2.13 |
| T-AG-2.15 | Blend Nodes | P0 | 4 | Phase 1 |
| T-AG-2.16 | LayerStack | P1 | 3 | T-AG-2.15 |
| T-AG-2.17 | LayerStackBuilder | P2 | 2 | T-AG-2.16 |

**Total Estimate**: 46 hours

---

## Verification Checklist

After Phase 2 completion:

- [ ] State machine transitions correctly
- [ ] All 6 blend curves work
- [ ] 1D blend tree interpolates
- [ ] 2D blend tree all 4 modes work
- [ ] Delaunay triangulation correct
- [ ] Barycentric coordinates correct
- [ ] All 10 blend node types work
- [ ] Layers compose correctly
- [ ] Builders produce valid objects
- [ ] All tests pass
