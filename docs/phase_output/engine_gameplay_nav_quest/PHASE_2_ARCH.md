# PHASE 2 ARCHITECTURE: Collision Avoidance and Steering

**Scope**: RVO/ORCA collision avoidance and steering behaviors  
**Files**: `avoidance.py`, `steering.py`  
**Lines**: ~1,951

---

## Architecture Overview

Phase 2 covers local movement control: avoiding collisions with other agents and producing smooth, believable motion through steering behaviors.

```
                    +------------------+
                    |  Desired Velocity |
                    | (from pathfinding)|
                    +--------+---------+
                             |
         +-------------------+-------------------+
         |                                       |
   +-----v-----+                          +------v------+
   | Collision |                          |  Steering   |
   | Avoidance |                          |  Behaviors  |
   +-----------+                          +-------------+
         |                                       |
   +-----v-----+                          +------v------+
   |   RVO     |                          |    Seek     |
   |  (sample) |                          |    Flee     |
   +-----------+                          |   Arrive    |
         |                                |   Pursue    |
   +-----v-----+                          |   Evade     |
   |   ORCA    |                          |   Wander    |
   |  (solve)  |                          |  Flocking   |
   +-----------+                          +-------------+
         |                                       |
         +-------------------+-------------------+
                             |
                    +--------v---------+
                    |  Final Velocity  |
                    +------------------+
```

---

## Component Architecture

### Collision Avoidance (`avoidance.py`)

```
CollisionAvoidanceEngine
├── compute_safe_velocity(agent, neighbors) -> Velocity
├── _rvo_velocity(agent, neighbors) -> Velocity
│   ├── _compute_obstacle_cone()
│   └── _sample_velocities()
├── _orca_velocity(agent, neighbors) -> Velocity
│   ├── _compute_orca_line()  # Half-plane constraint
│   └── _linear_program()      # Solve constraints
└── _force_avoidance(agent, neighbors) -> Force
    └── Fallback for degenerate cases
```

**Key Data Structures**:
- `VelocityObstacle`: Cone in velocity space representing collision region
- `OrcaLine`: Half-plane constraint (point + direction)
- `Agent`: Position, velocity, radius, preferred velocity

### Steering Behaviors (`steering.py`)

```
SteeringEngine
├── Basic Behaviors
│   ├── seek(target) -> Force
│   ├── flee(target) -> Force
│   ├── arrive(target, decel_radius) -> Force
│   ├── pursue(target_agent) -> Force
│   └── evade(target_agent) -> Force
├── Wander
│   └── wander(wander_radius, wander_distance) -> Force
├── Flocking
│   ├── separation(neighbors) -> Force
│   ├── alignment(neighbors) -> Force
│   └── cohesion(neighbors) -> Force
├── Environmental
│   ├── obstacle_avoidance(obstacles) -> Force
│   ├── wall_following(walls) -> Force
│   └── path_following(path) -> Force
└── Combination
    ├── weighted_sum(behaviors, weights) -> Force
    └── prioritized_dithering(behaviors) -> Force
```

---

## Algorithm Details

### RVO (Reciprocal Velocity Obstacles)

```
Input: Agent position, velocity, radius; Neighbor positions, velocities, radii
  |
  v
Compute Obstacle Cones: For each neighbor, compute velocity-space cone
  |
  v
Sample Velocities: Generate candidate velocities around preferred
  |
  v
Evaluate Candidates: Score by proximity to preferred + collision-free
  |
  v
Output: Best collision-free velocity
```

**Velocity Obstacle Cone**:
- Apex at agent's position in velocity space
- Sides tangent to Minkowski sum of agent + neighbor radii
- Shifted by neighbor's velocity (reciprocal)

### ORCA (Optimal Reciprocal Collision Avoidance)

```
Input: Agent position, velocity, radius; Neighbor positions, velocities, radii
  |
  v
Compute Half-Planes: For each neighbor, compute ORCA line
  |
  v
Solve Linear Program: Find velocity closest to preferred satisfying all constraints
  |
  v
Output: Optimal collision-free velocity
```

**ORCA Line Computation**:
- Compute relative position and velocity
- Find closest point on velocity obstacle boundary
- Half-plane passes through this point
- Normal points away from collision region

### Steering Behaviors

**Seek**: Steer toward target
```
desired = normalize(target - position) * max_speed
steering = desired - velocity
```

**Arrive**: Seek with deceleration
```
distance = length(target - position)
if distance < decel_radius:
    desired_speed = max_speed * (distance / decel_radius)
else:
    desired_speed = max_speed
```

**Pursue/Evade**: Seek/flee predicted position
```
prediction_time = distance / (speed + target_speed)
predicted_position = target_position + target_velocity * prediction_time
```

**Wander**: Jittered circle on sphere
```
wander_target += random_offset
wander_target = normalize(wander_target) * wander_radius
world_target = position + forward * wander_distance + wander_target
```

**Flocking**:
- Separation: Steer away from neighbors
- Alignment: Match average neighbor heading
- Cohesion: Steer toward center of neighbors

---

## Architectural Decisions

### ADR-NAV-005: Layered Avoidance (ORCA Primary, RVO Fallback)

**Context**: ORCA is optimal but can fail on degenerate cases.

**Decision**: Use ORCA as primary, fall back to RVO, then force-based.

**Rationale**:
- ORCA provides provably optimal solutions
- RVO handles cases ORCA cannot
- Force-based is always solvable (last resort)

**Consequences**:
- Multiple algorithms to maintain
- Need detection of ORCA failure
- Graceful degradation path

### ADR-NAV-006: Linear Program Solver for ORCA

**Context**: ORCA reduces to linear program: minimize ||v - v_pref|| subject to half-plane constraints.

**Decision**: Implement incremental 2D linear program solver.

**Rationale**:
- 2D LP is simple (intersect half-planes incrementally)
- No external solver dependency
- O(n) for n constraints (expected case)

**Consequences**:
- Must handle infeasible constraints (no solution)
- Degenerate cases require special handling

### ADR-NAV-007: Classic Reynolds Steering

**Context**: Agent locomotion needs smooth, believable movement.

**Decision**: Implement Craig Reynolds' steering behaviors.

**Rationale**:
- Proven approach since 1987
- Simple, composable, predictable
- Industry standard

**Consequences**:
- Must tune parameters (max force, max speed, radii)
- Combination requires weight tuning

### ADR-NAV-008: Prioritized Dithering for Behavior Combination

**Context**: Multiple steering behaviors must combine sensibly.

**Decision**: Support weighted sum and prioritized dithering.

**Rationale**:
- Weighted sum: Simple, tunable
- Prioritized dithering: Handles emergencies (collision avoidance first)

**Consequences**:
- Prioritized dithering may drop low-priority behaviors
- Weighted sum may produce conflicting forces

---

## Performance Considerations

### Collision Avoidance

- **Neighbor query**: Use spatial hash or quadtree for O(1) average
- **ORCA line count**: Limit neighbors considered (nearest N)
- **Velocity sampling**: Reduce samples for distant/slow agents
- **Cache constraints**: Reuse half-planes if positions unchanged

### Steering Behaviors

- **Flocking neighbors**: Use spatial index, limit range
- **Obstacle queries**: Broad-phase culling before ray casts
- **Path following**: Precompute path segments, binary search current
- **Force limits**: Clamp early to avoid expensive operations

---

## Integration Points

### Avoidance -> Pathfinding

- Collision avoidance operates in local space
- Pathfinding provides waypoints in world space
- Integration: Avoidance modifies velocity toward next waypoint

### Steering -> Animation

- Steering forces translate to velocity changes
- Animation system consumes velocity for locomotion
- Turn rate limits affect steering responsiveness

### Steering -> Physics

- Steering forces integrate with physics simulation
- Mass affects force-to-acceleration
- Friction affects velocity damping
