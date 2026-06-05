# PHASE 2 TODO: AI Subsystem

**Phase**: 2 of 3
**Subsystem**: engine/gameplay/ai
**Status**: Investigation Complete

---

## 1. Verification Tasks

### 1.1 Behavior Tree
- [ ] **T-AI-1.1**: Verify SequenceNode returns FAILURE on first child failure
- [ ] **T-AI-1.2**: Verify SelectorNode returns SUCCESS on first child success
- [ ] **T-AI-1.3**: Test ParallelNode REQUIRE_ALL policy
- [ ] **T-AI-1.4**: Test ParallelNode REQUIRE_ONE policy
- [ ] **T-AI-1.5**: Test ParallelNode REQUIRE_MAJORITY policy
- [ ] **T-AI-1.6**: Verify depth limit (100) prevents infinite recursion
- [ ] **T-AI-1.7**: Test InvertDecorator flips SUCCESS/FAILURE
- [ ] **T-AI-1.8**: Test RepeatDecorator with finite count
- [ ] **T-AI-1.9**: Test TimeoutDecorator expires after duration
- [ ] **T-AI-1.10**: Test CooldownDecorator blocks during cooldown
- [ ] **T-AI-1.11**: Test RetryDecorator retries on failure
- [ ] **T-AI-1.12**: Verify debug trace captures node execution

### 1.2 GOAP
- [ ] **T-AI-2.1**: Test WorldState hashing for closed set
- [ ] **T-AI-2.2**: Test Goal satisfaction check
- [ ] **T-AI-2.3**: Test GOAPAction precondition checking
- [ ] **T-AI-2.4**: Test A* planner finds optimal plan
- [ ] **T-AI-2.5**: Test A* respects iteration limit
- [ ] **T-AI-2.6**: Test plan caching (100 plans, 5s TTL)
- [ ] **T-AI-2.7**: Test GOAPAgent replan-on-failure behavior
- [ ] **T-AI-2.8**: Test procedural precondition callbacks

### 1.3 Utility AI
- [ ] **T-AI-3.1**: Test Linear response curve
- [ ] **T-AI-3.2**: Test Quadratic response curve
- [ ] **T-AI-3.3**: Test Exponential response curve
- [ ] **T-AI-3.4**: Test Logistic response curve: `1/(1+e^(-slope*x))`
- [ ] **T-AI-3.5**: Test Sine response curve
- [ ] **T-AI-3.6**: Test Inverse response curve
- [ ] **T-AI-3.7**: Test Step response curve
- [ ] **T-AI-3.8**: Test Smoothstep response curve: `x^2*(3-2x)`
- [ ] **T-AI-3.9**: Test compensation factor calculation
- [ ] **T-AI-3.10**: Test momentum prevents action thrashing
- [ ] **T-AI-3.11**: Test early-out on zero score

### 1.4 Perception
- [ ] **T-AI-4.1**: Test stimulus aging over time
- [ ] **T-AI-4.2**: Test stimulus decay and removal
- [ ] **T-AI-4.3**: Test known target persistence (3x multiplier)
- [ ] **T-AI-4.4**: Test all 6 sense types
- [ ] **T-AI-4.5**: Test stimulus strength filtering

### 1.5 Blackboard
- [ ] **T-AI-5.1**: Test namespaced key access
- [ ] **T-AI-5.2**: Test observer notification on change
- [ ] **T-AI-5.3**: Test pattern-based observer matching
- [ ] **T-AI-5.4**: Test TTL expiration and cleanup
- [ ] **T-AI-5.5**: Test BlackboardScope focused access
- [ ] **T-AI-5.6**: Test TypedBlackboardKey type safety
- [ ] **T-AI-5.7**: Test max observer limit (100 per key)

### 1.6 Combat AI
- [ ] **T-AI-6.1**: Test all 9 combat behaviors selection
- [ ] **T-AI-6.2**: Test threat assessment calculation
- [ ] **T-AI-6.3**: Test target priority modes (4 types)
- [ ] **T-AI-6.4**: Test health retreat threshold trigger

---

## 2. Integration Tasks

### 2.1 Trinity Pattern Integration
- [ ] **T-AI-7.1**: Register BTNode hierarchy with ComponentMeta
- [ ] **T-AI-7.2**: Register AISystem with SystemMeta
- [ ] **T-AI-7.3**: Install TrackedDescriptor on Blackboard entries
- [ ] **T-AI-7.4**: Create PerceptionEvent with EventMeta
- [ ] **T-AI-7.5**: Create ThreatEvent with EventMeta

### 2.2 Foundation Integration
- [ ] **T-AI-8.1**: Connect blackboard changes to EventLog
- [ ] **T-AI-8.2**: Register AI component types with Registry
- [ ] **T-AI-8.3**: Wire perception events to Tracker

---

## 3. Future Enhancements (Out of Scope)

### 3.1 Pathfinding Integration
- Connect to nav/pathfinding.py
- Path request/callback system
- Dynamic obstacle avoidance

### 3.2 Squad AI
- Formation maintenance
- Coordinated behaviors
- Communication protocols

### 3.3 Editor Tooling
- Visual behavior tree editor
- GOAP action graph
- Utility AI curve visualizer
- Live blackboard inspector

---

## 4. Acceptance Criteria

| Task Group | Criteria |
|------------|----------|
| Behavior Tree | All 14 node types tested |
| GOAP | A* finds optimal plans, caching works |
| Utility AI | All 8 curves correct, compensation verified |
| Perception | Decay and persistence work correctly |
| Blackboard | Observers fire, TTL expires |
| Combat AI | All behaviors select correctly |
