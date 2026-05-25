# PHASE 2 TODO: Crowd Behavior and LOD Systems

**Phase:** 2 of 3
**Focus:** Agent behavior simulation and LOD management
**Status:** IMPLEMENTED (Verification Required)

---

## Task List

### T2.1 Avoidance Algorithm Verification

**Priority:** HIGH
**Estimate:** 2 hours
**File:** `engine/animation/crowds/crowd_behavior.py`

**Acceptance Criteria:**
- [ ] Agents do not collide at avoidance_radius distance
- [ ] Coincident agents separate (not stuck)
- [ ] Priority weighting works (high priority agents push more)
- [ ] MIN_DISTANCE_EPSILON prevents division by zero

**Test Cases:**
```python
def test_coincident_agents_separate():
    agent_a = CrowdAgent(position=Vec3(0, 0, 0))
    agent_b = CrowdAgent(position=Vec3(0, 0, 0))  # Same position
    simulator.step(dt=0.016)
    assert (agent_a.position - agent_b.position).length() > 0

def test_priority_avoidance():
    low_priority = CrowdAgent(priority=1, position=Vec3(0, 0, 0))
    high_priority = CrowdAgent(priority=10, position=Vec3(1, 0, 0))
    # Move toward each other
    simulator.step(dt=0.016)
    # Low priority should move more
    assert low_priority.velocity.length() > high_priority.velocity.length()
```

---

### T2.2 Behavior State Machine

**Priority:** HIGH
**Estimate:** 1 hour
**File:** `engine/animation/crowds/crowd_behavior.py`

**Acceptance Criteria:**
- [ ] All state transitions are valid
- [ ] Invalid transitions are rejected
- [ ] State-specific behaviors activate correctly
- [ ] Animation clips change with state

**Test Cases:**
```python
def test_valid_state_transitions():
    agent = CrowdAgent(state=AgentState.IDLE)
    agent.transition_to(AgentState.WALKING)  # Valid
    assert agent.state == AgentState.WALKING

def test_invalid_state_transition():
    agent = CrowdAgent(state=AgentState.IDLE)
    with pytest.raises(InvalidTransitionError):
        agent.transition_to(AgentState.FORMATION)  # Invalid: need WALKING first
```

---

### T2.3 LOD Level Selection

**Priority:** MEDIUM
**Estimate:** 1 hour
**File:** `engine/animation/crowds/crowd_lod.py`

**Acceptance Criteria:**
- [ ] Correct LOD level selected for distance ranges
- [ ] Hysteresis prevents rapid switching
- [ ] Boundary conditions handled correctly
- [ ] Very far agents use lowest LOD

**Test Cases:**
```python
def test_lod_distance_selection():
    lod = CrowdLOD(levels=[
        LODLevel(distance=10, max_bone_count=50),
        LODLevel(distance=20, max_bone_count=30),
        LODLevel(distance=50, max_bone_count=10),
    ])
    assert lod.get_lod_level(distance=5) == 0   # Closest
    assert lod.get_lod_level(distance=15) == 1  # Middle
    assert lod.get_lod_level(distance=100) == 2 # Farthest

def test_lod_hysteresis():
    lod = CrowdLOD(hysteresis=2.0)
    # At distance 11 with current LOD 0 (threshold 10), should stay at 0
    assert lod.get_lod_level(distance=11, current=0) == 0
    # At distance 13, should switch
    assert lod.get_lod_level(distance=13, current=0) == 1
```

---

### T2.4 Skeleton Reduction

**Priority:** MEDIUM
**Estimate:** 2 hours
**File:** `engine/animation/crowds/crowd_lod.py`

**Acceptance Criteria:**
- [ ] Important bones are kept (root, spine, head)
- [ ] Unimportant bones are culled (fingers, twist)
- [ ] Reduced skeleton is valid (no orphan bones)
- [ ] Bone count matches target

**Test Cases:**
```python
def test_skeleton_reduction_priorities():
    skeleton = Skeleton(bones=["root", "spine", "head", "finger_01", "twist_arm"])
    reduced = create_reduced_skeleton(skeleton, max_bones=3)
    assert "root" in reduced.bones
    assert "spine" in reduced.bones
    assert "head" in reduced.bones
    assert "finger_01" not in reduced.bones
    assert "twist_arm" not in reduced.bones
```

---

### T2.5 Behavior Configuration

**Priority:** LOW
**Estimate:** 1 hour
**File:** `engine/animation/config.py`

**Acceptance Criteria:**
- [ ] All behavior parameters are configurable
- [ ] Config changes take effect without restart
- [ ] Invalid config values are rejected
- [ ] Default values are sensible

**Test Cases:**
```python
def test_config_validation():
    with pytest.raises(ValueError):
        CROWD_BEHAVIOR_CONFIG.avoidance_radius = -1  # Invalid

def test_config_defaults():
    assert CROWD_BEHAVIOR_CONFIG.MIN_DISTANCE_EPSILON > 0
    assert CROWD_BEHAVIOR_CONFIG.AVOIDANCE_PRIORITY_MULTIPLIER >= 1.0
```

---

### T2.6 Spatial Partitioning (ENHANCEMENT)

**Priority:** LOW (Future)
**Estimate:** 4 hours
**File:** `engine/animation/crowds/crowd_behavior.py` (new code)

**Acceptance Criteria:**
- [ ] Spatial hash reduces neighbor query from O(n) to O(1)
- [ ] Grid cell size is configurable
- [ ] Dynamic agent movement updates grid correctly
- [ ] Performance improvement measurable

**Notes:**
- This is an identified gap, not currently implemented
- Would enable 10K+ agents at interactive rates
- Can be added without breaking existing API

---

## Dependency Graph

```
T2.5 (Config)
    |
    v
T2.1 (Avoidance) --> T2.6 (Spatial - Future)
    |
    v
T2.2 (State Machine)
    
T2.3 (LOD Selection) --> T2.4 (Skeleton Reduction)
```

---

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| O(n) neighbor queries | Poor scaling | T2.6 (spatial hash) |
| Complex state machine | State bugs | Explicit transition table |
| Skeleton reduction artifacts | Visual glitches | Conservative importance |

---

## Definition of Done

Phase 2 is complete when:
1. All T2.x tasks (except T2.6) have passing tests
2. 500 agents navigate without collisions
3. LOD transitions are visually smooth
4. No state machine deadlocks after stress test
