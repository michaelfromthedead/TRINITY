# PHASE 1 TODO: Core Animation Graph

**Phase**: 1 of 4
**Focus**: Animation Graph Foundation

---

## Tasks

### T-AG-1.1: AnimationNode Base Class

**Priority**: P0 (Critical)
**Estimate**: 4 hours
**Dependencies**: None

**Acceptance Criteria:**
- [x] AnimationNode abstract base class implemented
- [x] GraphNodeMeta metaclass with auto-registration
- [x] Node registry accessible via GraphNodeMeta.registry
- [x] evaluate(context) method signature defined
- [x] Input/output slot system implemented
- [x] Node naming and identification

**Implementation Notes:**
- Use EngineMeta as metaclass base for Trinity compatibility
- Registry should support lookup by name and by type
- Slots should be typed (pose, float, bool, etc.)

---

### T-AG-1.2: Transform and Pose Data Structures

**Priority**: P0 (Critical)
**Estimate**: 3 hours
**Dependencies**: T-AG-1.1

**Acceptance Criteria:**
- [x] Transform dataclass with position, rotation, scale
- [x] Pose dataclass with bone transforms dict
- [x] Transform.blend(other, t) with SLERP for rotation
- [x] Pose.blend(other, t) for full pose blending
- [x] Transform.compose(other) for hierarchical transforms
- [x] Numerical stability checks (epsilon, clamping)

**Implementation Notes:**
- Quaternion SLERP must handle dot < 0 case (shorter path)
- SLERP_DOT_THRESHOLD fallback to linear interpolation
- Pose blend must handle missing bones gracefully

---

### T-AG-1.3: Skeleton and Bone Hierarchy

**Priority**: P0 (Critical)
**Estimate**: 3 hours
**Dependencies**: T-AG-1.2

**Acceptance Criteria:**
- [x] Skeleton class with root bone and bone dictionary
- [x] Bone class with parent/children relationships
- [x] Bind pose storage per bone
- [x] get_bone(name) lookup
- [x] get_chain(start, end) for IK integration
- [x] Skeleton validation (single root, no orphans)

**Implementation Notes:**
- Bone lookup should be O(1) via dictionary
- get_chain returns ordered list from start to end
- Support for multiple root bones (rare but valid)

---

### T-AG-1.4: Bone Mask System

**Priority**: P1 (High)
**Estimate**: 2 hours
**Dependencies**: T-AG-1.3

**Acceptance Criteria:**
- [x] BoneMask class with per-bone weights
- [x] apply(pose) method to mask a pose
- [x] combine(other) for mask composition
- [x] BoneMaskPresets enum/class with common masks
- [x] UPPER_BODY, LOWER_BODY, LEFT_ARM, RIGHT_ARM, LEFT_LEG, RIGHT_LEG
- [x] GRADIENT preset generator with falloff

**Implementation Notes:**
- Weights should be clamped to [0, 1]
- Missing bones in mask default to 0 or 1 based on mode
- Gradient falloff can be linear or exponential

---

### T-AG-1.5: AnimationGraph Container

**Priority**: P0 (Critical)
**Estimate**: 4 hours
**Dependencies**: T-AG-1.1, T-AG-1.2

**Acceptance Criteria:**
- [x] AnimationGraph class as DAG container
- [x] add_node(node) method
- [x] remove_node(name) method
- [x] connect(source, target, slot) method
- [x] disconnect(source, target) method
- [x] evaluate(context) method with topological traversal
- [x] Cycle detection before evaluation

**Implementation Notes:**
- Use topological sort for evaluation order
- Cycle detection with three-color DFS
- Support subgraph nesting

---

### T-AG-1.6: GraphContext

**Priority**: P0 (Critical)
**Estimate**: 2 hours
**Dependencies**: T-AG-1.5

**Acceptance Criteria:**
- [x] GraphContext dataclass with evaluation state
- [x] dt (delta time) parameter
- [x] skeleton reference
- [x] parameters dictionary access
- [x] sync_group reference (optional)
- [x] Current time/tick tracking

**Implementation Notes:**
- Context should be lightweight (passed by reference)
- Parameters should support get with default
- Consider context pooling for performance

**Implementation Status:** COMPLETE (2026-05-31)
- Full GraphContext dataclass in `engine/animation/graph/animation_graph.py` (lines 725-830)
- ContextPool for performance optimization (lines 838-926)
- Helper methods: get_parameter(), get_parameter_float/int/bool(), advance_time(), with_depth()
- Additional features: normalized_time, layer_weight, bone_masks, evaluation_depth
- 142 tests passing (contract + whitebox)

---

### T-AG-1.7: Configuration Module

**Priority**: P1 (High)
**Estimate**: 1 hour
**Dependencies**: None (can be done in parallel)

**Acceptance Criteria:**
- [x] config.py with all tuning constants
- [x] QuaternionConfig class
- [x] GraphConfig class
- [x] BlendConfig class
- [x] Documentation for each constant

**Implementation Notes:**
- Use dataclass or NamedTuple for immutability
- Consider environment variable overrides for testing
- Document units (radians, seconds, etc.)

---

### T-AG-1.8: Cycle Detection Algorithm

**Priority**: P1 (High)
**Estimate**: 2 hours
**Dependencies**: T-AG-1.5

**Acceptance Criteria:**
- [x] detect_cycles(graph) function
- [x] Three-color DFS implementation
- [x] Returns list of cycle descriptions
- [x] Configurable (can be disabled for performance)
- [x] Clear error messages with node names

**Implementation Notes:**
- WHITE/GRAY/BLACK coloring scheme
- GRAY node visited twice = cycle
- Report all cycles, not just first

---

## Task Summary

| Task ID | Description | Priority | Est. Hours | Dependencies |
|---------|-------------|----------|------------|--------------|
| T-AG-1.1 | AnimationNode Base | P0 | 4 | None |
| T-AG-1.2 | Transform/Pose | P0 | 3 | T-AG-1.1 |
| T-AG-1.3 | Skeleton/Bone | P0 | 3 | T-AG-1.2 |
| T-AG-1.4 | Bone Masks | P1 | 2 | T-AG-1.3 |
| T-AG-1.5 | AnimationGraph | P0 | 4 | T-AG-1.1, T-AG-1.2 |
| T-AG-1.6 | GraphContext | P0 | 2 | T-AG-1.5 |
| T-AG-1.7 | Configuration | P1 | 1 | None |
| T-AG-1.8 | Cycle Detection | P1 | 2 | T-AG-1.5 |

**Total Estimate**: 21 hours

---

## Verification Checklist

After Phase 1 completion:

- [ ] AnimationGraph can be instantiated
- [ ] Nodes can be added and connected
- [ ] Cycle detection prevents invalid graphs
- [ ] Poses can be created and blended
- [ ] Skeleton hierarchy traversal works
- [ ] Bone masks apply correctly
- [ ] All tests pass
- [ ] config.py constants documented
