# Investigation: engine/gameplay/ai

## Summary
The AI subsystem is a **REAL IMPLEMENTATION** with 4,523 lines of production-ready code. It provides complete behavior trees (Sequence, Selector, Parallel, decorators), utility AI with response curves and considerations, GOAP with A* planning, blackboard knowledge sharing, perception systems with stimuli and decay, and combat AI. This is not a stub - it is a fully functional AI framework ready for game use.

Note: Pathfinding (A*, JPS, Theta*, HPA*) is located in `engine/gameplay/nav/pathfinding.py` (959 lines) rather than the ai/ directory.

## Files
| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `__init__.py` | 1,184 | REAL | Full BT, Utility, GOAP, Perception, Combat AI |
| `behavior_tree.py` | 948 | REAL | Complete BT with 14 node types, context, debugging |
| `blackboard.py` | 496 | REAL | Key-value store with observers, TTL, scopes, typed keys |
| `constants.py` | 457 | REAL | 90+ typed constants, 9 enums, no magic numbers |
| `goap.py` | 727 | REAL | A* planner, plan caching, GOAPAgent lifecycle |
| `utility_ai.py` | 711 | REAL | 8 response curves, considerations, momentum |
| **Total** | **4,523** | | |

## AI Components

### Behavior Tree (behavior_tree.py + __init__.py)
- **Composite Nodes**: Sequence, Selector, Parallel (with policies: REQUIRE_ALL, REQUIRE_ONE, REQUIRE_MAJORITY)
- **Decorator Nodes**: Invert, Repeat, Timeout, Cooldown, Retry, ForceSuccess, ForceFailure
- **Leaf Nodes**: Action, Condition, BlackboardCondition, Wait, SetBlackboard
- **Context System**: BTContext with delta_time, blackboard, entity, debug tracing
- **Depth Limiting**: BT_MAX_DEPTH = 100 prevents infinite recursion

### Utility AI (utility_ai.py)
- **Response Curves**: Linear, Quadratic, Exponential, Logistic, Sine, Inverse, Step, Smoothstep, Custom
- **Considerations**: Abstract base, Blackboard, Function, Distance, Health (5 types)
- **Scoring**: Compensation factor prevents score collapse with multiple considerations
- **Momentum**: Prevents action thrashing (default 0.1)
- **Debug**: get_debug_info() for visualization

### GOAP (goap.py)
- **WorldState**: Immutable dict-based state representation with hashing
- **Actions**: Preconditions, effects, costs, procedural checks
- **Planner**: A* search with heuristic weighting, iteration limits
- **Caching**: Plan cache with TTL (100 plans, 5s TTL)
- **Agent**: Full lifecycle with replan-on-failure

### Perception (__init__.py)
- **Stimuli**: Source, sense type, position, strength, timestamp, age
- **Senses**: SIGHT, HEARING, DAMAGE, SQUAD, TOUCH, SMELL
- **Memory**: Known targets persist 3x longer than raw stimuli
- **Decay**: Configurable decay rate per stimulus

### Combat AI (__init__.py)
- **Behaviors**: ATTACK, DEFEND, FLANK, RETREAT, SUPPORT, COVER, SUPPRESS, ADVANCE, HOLD_POSITION
- **Threat System**: ThreatAssessment with threat_level, distance, visibility
- **Target Priority**: NEAREST, WEAKEST, STRONGEST, HIGHEST_THREAT, etc.
- **Health Retreat**: Configurable threshold (default 25%)

### Blackboard (blackboard.py)
- **Keys**: Namespaced, hierarchical (e.g., "combat.target")
- **Observers**: Pattern matching, one-shot, max 100 per key
- **TTL**: Automatic expiration with cleanup
- **Scopes**: BlackboardScope for focused access
- **Typed**: TypedBlackboardKey[T] for type-safe access

## AI Implementation

- Real behavior trees? **YES** - Full implementation with 14 node types, proper tick/reset/abort lifecycle
- Real pathfinding (A*)? **YES** - Located in nav/pathfinding.py with A*, JPS, Theta*, HPA*
- Real utility AI? **YES** - 8 response curves, 5 consideration types, compensation factor
- Real perception? **YES** - Multi-sense system with stimuli aging and memory
- Real GOAP? **YES** - A* planner with caching, GOAPAgent with replan-on-failure

## Verdict
**REAL IMPLEMENTATION**

This is a production-grade AI framework with:
- Zero stubs or placeholder code
- Proper error handling throughout
- Configurable constants (no magic numbers)
- Debug/trace support
- Memory management (TTL, decay, cleanup)
- Industry-standard algorithms (behavior trees, utility AI, GOAP)

## Evidence

### Behavior Tree Tick (behavior_tree.py:173-201)
```python
def tick(self, context: BTContext) -> BTStatus:
    if context.depth > BT_MAX_DEPTH:
        self._status = BTStatus.FAILURE
        return self._status

    if context.abort_requested:
        self.abort()
        return self._status

    child_context = context.child_context()

    while self._current_index < len(self._children):
        child = self._children[self._current_index]
        status = child.tick(child_context)
        context.log_trace(child, status)

        if status == BTStatus.RUNNING:
            self._status = BTStatus.RUNNING
            return self._status
        elif status == BTStatus.FAILURE:
            self._current_index = 0
            self._status = BTStatus.FAILURE
            return self._status

        self._current_index += 1

    self._current_index = 0
    self._status = BTStatus.SUCCESS
    return self._status
```

### GOAP A* Planning (goap.py:456-506)
```python
while open_set and iterations < self.max_iterations:
    iterations += 1

    # Get node with lowest f_cost
    current = heapq.heappop(open_set)
    current_state_hash = current.state.to_hashable()

    # Skip if already visited
    if current_state_hash in closed_set:
        continue

    closed_set.add(current_state_hash)

    # Check if goal is satisfied
    if goal.is_satisfied(current.state):
        plan = self._reconstruct_plan(current, goal, current_state)
        # Cache the plan
        if use_cache and len(self._plan_cache) < GOAP_PLAN_CACHE_SIZE:
            cache_key = (current_state.to_hashable(), goal.name)
            self._plan_cache[cache_key] = plan
        return plan
```

### Utility AI Score Compensation (utility_ai.py:374-400)
```python
def calculate_score(self, context: ConsiderationContext) -> ActionScore:
    if not self._considerations:
        return ActionScore(action=self, score=self.base_score)

    total_score = self.base_score
    consideration_scores: Dict[str, float] = {}

    for consideration in self._considerations:
        score = consideration.score(context)
        consideration_scores[consideration.name] = score

        if score <= UTILITY_SCORE_EPSILON:
            return ActionScore(action=self, score=0.0, consideration_scores=consideration_scores)

        # Apply compensation factor
        modification = (1 - score) * (1 - (1 / len(self._considerations)))
        total_score += score + modification * score

    total_score /= len(self._considerations)
    return ActionScore(action=self, score=total_score, consideration_scores=consideration_scores)
```

### Perception Decay (__init__.py:949-967)
```python
def update(self, delta_time: float) -> None:
    # Age stimuli
    for stimulus in self._stimuli:
        stimulus.age += delta_time

    # Remove old stimuli
    self._stimuli = [s for s in self._stimuli if s.age < self._decay_rate]

    # Age known targets
    expired_targets = []
    for actor_id, stimulus in self._known_targets.items():
        stimulus.age += delta_time
        # Known targets persist longer in memory
        if stimulus.age >= self._decay_rate * self.KNOWN_TARGET_PERSISTENCE_MULTIPLIER:
            expired_targets.append(actor_id)

    for actor_id in expired_targets:
        del self._known_targets[actor_id]
```
