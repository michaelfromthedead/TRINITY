# PHASE 2 ARCH: Crowd Behavior and LOD Systems

**Phase:** 2 of 3
**Focus:** Agent behavior simulation and LOD management
**Status:** IMPLEMENTED (Investigation confirms REAL)

---

## Phase Overview

Phase 2 adds intelligent behavior to crowd agents and implements the LOD system for performance scaling. Agents move, avoid, and interact while the LOD system optimizes rendering based on camera distance.

---

## Architecture Components

### 2.1 Crowd Agent Model

**Module:** `engine/animation/crowds/crowd_behavior.py`
**Lines:** 711

```
CrowdAgent
    +-- id: int
    +-- position: Vec3
    +-- velocity: Vec3
    +-- facing: Vec3
    +-- state: AgentState
    +-- priority: int
    +-- group_id: Optional[int]
    +-- animation_clip: str
    +-- animation_speed: float
```

**State Machine:**
```
AgentState (Enum):
    IDLE       -> WALKING, WAITING, FLEEING
    WALKING    -> IDLE, WAITING, FLEEING, FORMATION
    WAITING    -> IDLE, WALKING, FLEEING
    FLEEING    -> IDLE, WALKING
    FORMATION  -> IDLE, WALKING, FLEEING
```

### 2.2 Behavior System

**Module:** `engine/animation/crowds/crowd_behavior.py`

```
CrowdBehavior (Abstract Base)
    +-- update(agent, context, dt) -> BehaviorResult

BehaviorContext
    +-- get_nearby_agents(agent, radius) -> list[CrowdAgent]
    +-- get_nearby_obstacles(agent, radius) -> list[Obstacle]
    +-- get_threat_source() -> Optional[Vec3]
```

**Concrete Behaviors:**

| Behavior | Purpose | Key Parameters |
|----------|---------|----------------|
| `IdleBehavior` | Standing in place | animation_variation |
| `WalkingBehavior` | Movement with steering | target, avoidance_radius |
| `WaitingBehavior` | Queue-like waiting | fidget_chance |
| `FleeingBehavior` | Panic escape | threat_source, safe_distance |
| `FormationBehavior` | Leader following | leader, formation_offset |

### 2.3 Avoidance Algorithm

**Implementation:** Lines 304-345

```
_calculate_avoidance(agent, context) -> Vec3:
    avoidance = Vec3.zero()
    
    for other in context.get_nearby_agents(agent, avoidance_radius):
        distance = (agent.position - other.position).length()
        
        if distance < MIN_DISTANCE_EPSILON:
            # Coincident: push random direction
            avoidance += random_direction() * strength
        else:
            # Distance-weighted avoidance
            strength = (1 - distance/avoidance_radius) * base_strength
            
            # Priority weighting
            if other.priority > agent.priority:
                strength *= PRIORITY_MULTIPLIER
            
            avoidance += direction_away * strength
    
    # Similar for obstacles...
    return avoidance
```

**Parameters (from config):**
- `MIN_DISTANCE_EPSILON`: 0.01 (prevents division by zero)
- `AVOIDANCE_PRIORITY_MULTIPLIER`: 1.5 (higher priority agents push more)
- `avoidance_radius`: configurable per behavior

### 2.4 LOD System

**Module:** `engine/animation/crowds/crowd_lod.py`
**Lines:** 497

```
LODLevel
    +-- distance_threshold: float
    +-- max_bone_count: int
    +-- update_rate: int (Hz)
    +-- cast_shadows: bool
    
CrowdLOD
    +-- levels: list[LODLevel]
    +-- hysteresis: float
    +-- get_lod_level(distance, current_level) -> int
    +-- create_reduced_skeleton(skeleton, bone_count) -> Skeleton
```

**Skeleton Reduction Algorithm:**

```
_calculate_bone_importance(name, index, skeleton) -> float:
    score = 0.5  # base
    
    # Anatomical importance
    if "root" in name:      score += 0.5
    elif "spine" in name:   score += 0.4
    elif "head" in name:    score += 0.35
    elif "arm" in name:     score += 0.3
    elif "leg" in name:     score += 0.3
    elif "hand" in name:    score += 0.2
    elif "finger" in name:  score += 0.1
    
    # Penalties
    if "twist" in name:     score -= 0.2
    score -= depth * 0.02   # Hierarchy depth penalty
    
    return clamp(score, 0, 1)
```

### 2.5 Crowd Simulator

**Module:** `engine/animation/crowds/crowd_behavior.py`

```
CrowdSimulator
    +-- agents: dict[int, CrowdAgent]
    +-- behaviors: dict[AgentState, CrowdBehavior]
    +-- update(dt):
        for agent in agents:
            behavior = behaviors[agent.state]
            context = build_context(agent)
            result = behavior.update(agent, context, dt)
            apply_result(agent, result)
```

---

## Dependencies

### Internal
- Phase 1: `CrowdInstance` for rendering
- `engine/animation/config.py` - `CROWD_BEHAVIOR_CONFIG`, `CROWD_LOD_CONFIG`

### External
- None (pure Python simulation)

---

## Interfaces

### Input Interface
```python
class BehaviorContext:
    def get_nearby_agents(agent: CrowdAgent, radius: float) -> list[CrowdAgent]
    def get_nearby_obstacles(agent: CrowdAgent, radius: float) -> list[Obstacle]
    def get_threat_source() -> Optional[Vec3]
    def get_target_position(agent: CrowdAgent) -> Optional[Vec3]
```

### Output Interface
```python
class BehaviorResult:
    new_velocity: Vec3
    new_facing: Vec3
    new_state: Optional[AgentState]
    animation_clip: Optional[str]
    animation_speed: Optional[float]
```

---

## Quality Attributes

### Performance
- O(n) neighbor queries (known limitation)
- LOD reduces bone count for distant agents
- Update rate scaling per LOD level

### Correctness
- Edge case handling for coincident agents
- Hysteresis prevents LOD flickering
- Priority-based avoidance asymmetry

### Extensibility
- Pluggable behavior system
- Configurable LOD levels
- Custom bone importance functions

---

## Verification Criteria

| Criterion | Verification Method |
|-----------|-------------------|
| Agents avoid each other | Visual inspection + collision count |
| LOD levels transition smoothly | Distance sweep test |
| Skeleton reduction preserves important bones | Importance ranking test |
| Hysteresis prevents flickering | Rapid distance oscillation test |
