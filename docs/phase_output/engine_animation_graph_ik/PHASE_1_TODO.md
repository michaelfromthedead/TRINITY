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
- [ ] AnimationNode abstract base class implemented
- [ ] GraphNodeMeta metaclass with auto-registration
- [ ] Node registry accessible via GraphNodeMeta.registry
- [ ] evaluate(context) method signature defined
- [ ] Input/output slot system implemented
- [ ] Node naming and identification

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
- [ ] Transform dataclass with position, rotation, scale
- [ ] Pose dataclass with bone transforms dict
- [ ] Transform.blend(other, t) with SLERP for rotation
- [ ] Pose.blend(other, t) for full pose blending
- [ ] Transform.compose(other) for hierarchical transforms
- [ ] Numerical stability checks (epsilon, clamping)

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
- [ ] Skeleton class with root bone and bone dictionary
- [ ] Bone class with parent/children relationships
- [ ] Bind pose storage per bone
- [ ] get_bone(name) lookup
- [ ] get_chain(start, end) for IK integration
- [ ] Skeleton validation (single root, no orphans)

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
- [ ] BoneMask class with per-bone weights
- [ ] apply(pose) method to mask a pose
- [ ] combine(other) for mask composition
- [ ] BoneMaskPresets enum/class with common masks
- [ ] UPPER_BODY, LOWER_BODY, LEFT_ARM, RIGHT_ARM, LEFT_LEG, RIGHT_LEG
- [ ] GRADIENT preset generator with falloff

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
- [ ] AnimationGraph class as DAG container
- [ ] add_node(node) method
- [ ] remove_node(name) method
- [ ] connect(source, target, slot) method
- [ ] disconnect(source, target) method
- [ ] evaluate(context) method with topological traversal
- [ ] Cycle detection before evaluation

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
- [ ] GraphContext dataclass with evaluation state
- [ ] dt (delta time) parameter
- [ ] skeleton reference
- [ ] parameters dictionary access
- [ ] sync_group reference (optional)
- [ ] Current time/tick tracking

**Implementation Notes:**
- Context should be lightweight (passed by reference)
- Parameters should support get with default
- Consider context pooling for performance

---

### T-AG-1.7: Configuration Module

**Priority**: P1 (High)
**Estimate**: 1 hour
**Dependencies**: None (can be done in parallel)

**Acceptance Criteria:**
- [ ] config.py with all tuning constants
- [ ] QuaternionConfig class
- [ ] GraphConfig class
- [ ] BlendConfig class
- [ ] Documentation for each constant

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
- [ ] detect_cycles(graph) function
- [ ] Three-color DFS implementation
- [ ] Returns list of cycle descriptions
- [ ] Configurable (can be disabled for performance)
- [ ] Clear error messages with node names

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
