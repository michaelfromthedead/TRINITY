# PHASE 2 ARCHITECTURE: AI Subsystem

**Phase**: 2 of 3
**Subsystem**: engine/gameplay/ai
**Lines**: 4,523
**Status**: REAL IMPLEMENTATION

---

## 1. Overview

The AI subsystem provides a complete toolkit for game AI: behavior trees, GOAP planning, utility AI, perception, blackboard, and combat behaviors. Pathfinding is located separately in `engine/gameplay/nav/pathfinding.py`.

---

## 2. Module Structure

```
engine/gameplay/ai/
    __init__.py          # 1,184 lines - BT, Perception, Combat
    behavior_tree.py     # 948 lines - Full BT implementation
    blackboard.py        # 496 lines - Key-value knowledge store
    goap.py              # 727 lines - A* GOAP planner
    utility_ai.py        # 711 lines - Response curves
    constants.py         # 457 lines - 90+ constants
```

---

## 3. Component Architecture

### 3.1 Behavior Tree (behavior_tree.py + __init__.py)

```
BTNode (abstract)
    |-- _status: BTStatus (RUNNING, SUCCESS, FAILURE)
    |-- tick(context) -> BTStatus
    |-- reset() / abort()

CompositeNode
    |-- _children: List[BTNode]
    +-- SequenceNode (AND: all must succeed)
    +-- SelectorNode (OR: first success wins)
    +-- ParallelNode (policies: REQUIRE_ALL, REQUIRE_ONE, REQUIRE_MAJORITY)

DecoratorNode
    |-- _child: BTNode
    +-- InvertDecorator (flip SUCCESS/FAILURE)
    +-- RepeatDecorator (loop N times or forever)
    +-- TimeoutDecorator (fail after duration)
    +-- CooldownDecorator (block for duration after run)
    +-- RetryDecorator (retry N times on failure)
    +-- ForceSuccessDecorator
    +-- ForceFailureDecorator

LeafNode
    +-- ActionNode (execute gameplay action)
    +-- ConditionNode (check condition)
    +-- BlackboardConditionNode (check blackboard key)
    +-- WaitNode (wait for duration)
    +-- SetBlackboardNode (write to blackboard)

BTContext
    |-- delta_time: float
    |-- blackboard: Blackboard
    |-- entity: Entity
    |-- depth: int (max 100)
    |-- debug_trace: List[str]
```

**Tick Algorithm (Sequence)**:
```python
for child in children:
    status = child.tick(context)
    if status == RUNNING: return RUNNING
    if status == FAILURE: return FAILURE
return SUCCESS
```

### 3.2 GOAP (goap.py)

```
WorldState
    |-- _state: Dict[str, Any]
    |-- to_hashable() -> tuple
    |-- satisfies(conditions) -> bool

Goal
    |-- name: str
    |-- conditions: Dict[str, Any]
    |-- priority: float
    |-- is_satisfied(state) -> bool

GOAPAction
    |-- name: str
    |-- preconditions: Dict[str, Any]
    |-- effects: Dict[str, Any]
    |-- cost: float
    |-- procedural_check: Optional[Callable]

PlanNode (internal)
    |-- state: WorldState
    |-- action: Optional[GOAPAction]
    |-- parent: Optional[PlanNode]
    |-- g_cost / h_cost / f_cost

GOAPPlanner
    |-- _actions: List[GOAPAction]
    |-- _plan_cache: Dict (100 plans, 5s TTL)
    |-- plan(current_state, goal) -> List[GOAPAction]

GOAPAgent
    |-- _planner: GOAPPlanner
    |-- _current_plan: List[GOAPAction]
    |-- _current_action_index: int
    |-- update() with replan-on-failure
```

**A* Algorithm**:
1. Initialize open set with start state (g=0)
2. Pop lowest f_cost node
3. If goal satisfied, reconstruct plan
4. For each applicable action:
   - Compute new state by applying effects
   - Calculate g_cost = parent.g + action.cost
   - Calculate h_cost = unsatisfied conditions count
   - Add to open set if not in closed set

### 3.3 Utility AI (utility_ai.py)

```
ResponseCurve (enum)
    |-- LINEAR, QUADRATIC, EXPONENTIAL, LOGISTIC
    |-- SINE, INVERSE, STEP, SMOOTHSTEP, CUSTOM

Consideration (abstract)
    |-- name: str
    |-- curve: ResponseCurve
    |-- score(context) -> float (0.0 - 1.0)
    +-- BlackboardConsideration
    +-- FunctionConsideration
    +-- DistanceConsideration
    +-- HealthConsideration

UtilityAction
    |-- name: str
    |-- base_score: float
    |-- _considerations: List[Consideration]
    |-- calculate_score(context) -> ActionScore

UtilityAI
    |-- _actions: List[UtilityAction]
    |-- _current_action: Optional[UtilityAction]
    |-- momentum: float (default 0.1)
    |-- select_action(context) -> UtilityAction
```

**Scoring with Compensation Factor**:
```python
for consideration in considerations:
    score = consideration.score(context)
    if score <= EPSILON: return 0.0  # Early out
    modification = (1 - score) * (1 - 1/n)
    total_score += score + modification * score
total_score /= n
```

### 3.4 Perception (__init__.py)

```
Stimulus
    |-- source: Entity
    |-- sense_type: SenseType (SIGHT, HEARING, DAMAGE, SQUAD, TOUCH, SMELL)
    |-- position: Vec3
    |-- strength: float
    |-- timestamp: float
    |-- age: float

PerceptionSystem
    |-- _stimuli: List[Stimulus]
    |-- _known_targets: Dict[int, Stimulus]
    |-- _decay_rate: float
    |-- KNOWN_TARGET_PERSISTENCE_MULTIPLIER = 3.0
    |-- add_stimulus() / update() / get_known_targets()
```

**Decay Algorithm**:
```python
for stimulus in stimuli:
    stimulus.age += delta_time
stimuli = [s for s in stimuli if s.age < decay_rate]
# Known targets persist 3x longer
```

### 3.5 Blackboard (blackboard.py)

```
BlackboardKey
    |-- name: str (hierarchical, e.g., "combat.target")
    |-- namespace: Optional[str]

BlackboardEntry
    |-- value: Any
    |-- timestamp: float
    |-- ttl: Optional[float]

Blackboard
    |-- _entries: Dict[str, BlackboardEntry]
    |-- _observers: Dict[str, List[Callable]]
    |-- get() / set() / has()
    |-- observe() / notify()
    |-- cleanup_expired()

BlackboardScope
    |-- Focused view of subset of keys

TypedBlackboardKey[T]
    |-- Type-safe access wrapper
```

### 3.6 Combat AI (__init__.py)

```
CombatBehavior (enum)
    |-- ATTACK, DEFEND, FLANK, RETREAT, SUPPORT
    |-- COVER, SUPPRESS, ADVANCE, HOLD_POSITION

ThreatAssessment
    |-- target: Entity
    |-- threat_level: float
    |-- distance: float
    |-- visibility: float

TargetPriority (enum)
    |-- NEAREST, WEAKEST, STRONGEST, HIGHEST_THREAT

CombatAI
    |-- _threats: List[ThreatAssessment]
    |-- _target_priority: TargetPriority
    |-- health_retreat_threshold: float (default 0.25)
    |-- select_behavior() / select_target()
```

---

## 4. Integration Points

| Component | Trinity Integration |
|-----------|---------------------|
| BTNode | ComponentMeta for serialization |
| GOAPAgent | SystemMeta for tick scheduling |
| Blackboard | TrackedDescriptor on entries |
| Perception | EventMeta for PerceptionEvent |
| CombatAI | ComponentMeta for entity attachment |

---

## 5. Dependencies

```python
from engine.core.math.vec import Vec3
from engine.gameplay.ai.constants import *
from engine.gameplay.entity import Entity
import heapq  # For A* priority queue
```

---

## 6. Design Decisions

### 6.1 Why Three AI Systems?
Each solves different problems:
- BT: Reactive, sequential logic
- GOAP: Emergent, goal-driven planning
- Utility: Smooth, priority-based selection

### 6.2 Why Blackboard?
Decouples knowledge from behavior. Multiple systems can read/write without direct coupling. Observers enable reactive updates.

### 6.3 Why Compensation Factor in Utility?
Prevents score collapse when many considerations are present. Geometric mean alone would make high-consideration actions always low-scoring.
