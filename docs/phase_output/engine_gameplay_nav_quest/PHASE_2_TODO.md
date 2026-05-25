# PHASE 2 TODO: Collision Avoidance and Steering

**Scope**: RVO/ORCA collision avoidance and steering behaviors  
**Files**: `avoidance.py`, `steering.py`

---

## T-NAV-2.1: Verify RVO Velocity Obstacle Computation

**Priority**: P0  
**Estimate**: 1.5 hours

### Description
Verify RVO correctly computes velocity obstacle cones for neighboring agents.

### Tasks
- [ ] Review `_compute_obstacle_cone()` geometry
- [ ] Verify cone apex position (relative velocity space)
- [ ] Verify tangent line computation (Minkowski sum radii)
- [ ] Test with known agent configurations

### Acceptance Criteria
- Velocity obstacle cones correctly represent collision regions
- Cones are symmetric for reciprocal pairs
- Edge cases (overlapping agents, zero velocity) handled

---

## T-NAV-2.2: Verify RVO Velocity Sampling

**Priority**: P0  
**Estimate**: 1 hour

### Description
Verify RVO velocity sampling produces collision-free velocities.

### Tasks
- [ ] Review `_sample_velocities()` distribution
- [ ] Verify candidate evaluation against obstacle cones
- [ ] Check velocity selection criterion (distance to preferred)
- [ ] Test sampling density vs. quality tradeoff

### Acceptance Criteria
- Sampled velocities cover velocity space adequately
- Selected velocity is collision-free
- Selected velocity is near preferred when possible
- Performance acceptable (< 1ms for 10 neighbors)

---

## T-NAV-2.3: Verify ORCA Half-Plane Computation

**Priority**: P0  
**Estimate**: 1.5 hours

### Description
Verify ORCA correctly computes half-plane constraints from velocity obstacles.

### Tasks
- [ ] Review `_compute_orca_line()` geometry
- [ ] Verify relative position/velocity handling
- [ ] Verify half-plane normal direction (away from collision)
- [ ] Test constraint placement for various configurations

### Acceptance Criteria
- Half-plane correctly separates safe/unsafe velocities
- Constraints are consistent between agents (reciprocal)
- Edge cases (head-on, parallel, overtake) handled

---

## T-NAV-2.4: Verify ORCA Linear Program Solver

**Priority**: P0  
**Estimate**: 1.5 hours

### Description
Verify the 2D linear program solver finds optimal velocities.

### Tasks
- [ ] Review `_linear_program()` implementation
- [ ] Verify incremental constraint intersection
- [ ] Test feasibility detection (no solution case)
- [ ] Compare solutions against known optimal

### Acceptance Criteria
- LP solver finds velocity closest to preferred
- All half-plane constraints satisfied
- Infeasible cases detected and handled
- Solution is geometrically correct

---

## T-NAV-2.5: Verify Steering Behavior Implementations

**Priority**: P1  
**Estimate**: 2 hours

### Description
Verify all steering behaviors produce correct forces.

### Tasks
- [ ] Review seek/flee force direction
- [ ] Review arrive deceleration curve
- [ ] Review pursue/evade prediction
- [ ] Review wander circle jitter
- [ ] Review flocking (separation, alignment, cohesion)

### Acceptance Criteria
- Seek produces force toward target
- Flee produces force away from target
- Arrive decelerates smoothly to stop
- Pursue intercepts moving target
- Evade escapes moving pursuer
- Wander produces organic random motion
- Flocking produces emergent group behavior

---

## T-NAV-2.6: Verify Environmental Behaviors

**Priority**: P1  
**Estimate**: 1 hour

### Description
Verify obstacle avoidance, wall following, and path following behaviors.

### Tasks
- [ ] Review obstacle ray casting approach
- [ ] Review wall following sensor placement
- [ ] Review path following segment selection
- [ ] Test behaviors against sample environments

### Acceptance Criteria
- Obstacle avoidance steers away from obstacles
- Wall following maintains consistent distance
- Path following tracks waypoints smoothly
- Transitions between segments are smooth

---

## T-NAV-2.7: Verify Behavior Combination Methods

**Priority**: P1  
**Estimate**: 1 hour

### Description
Verify weighted sum and prioritized dithering combination methods.

### Tasks
- [ ] Review weighted sum normalization
- [ ] Review prioritized dithering budget distribution
- [ ] Test with conflicting behaviors
- [ ] Verify force magnitude limits

### Acceptance Criteria
- Weighted sum respects weights
- Prioritized dithering respects priorities
- Combined force does not exceed max
- Conflicting behaviors resolve sensibly

---

## T-NAV-2.8: Test Avoidance Fallback Chain

**Priority**: P2  
**Estimate**: 1 hour

### Description
Verify fallback from ORCA to RVO to force-based avoidance.

### Tasks
- [ ] Create ORCA-infeasible scenario
- [ ] Verify RVO fallback activates
- [ ] Create RVO-degenerate scenario
- [ ] Verify force-based fallback activates

### Acceptance Criteria
- ORCA failure detected correctly
- RVO provides valid backup
- Force-based always produces result
- Transitions are smooth (no discontinuity)

---

## Summary

| Task | Priority | Estimate | Status |
|------|----------|----------|--------|
| T-NAV-2.1 | P0 | 1.5h | Pending |
| T-NAV-2.2 | P0 | 1h | Pending |
| T-NAV-2.3 | P0 | 1.5h | Pending |
| T-NAV-2.4 | P0 | 1.5h | Pending |
| T-NAV-2.5 | P1 | 2h | Pending |
| T-NAV-2.6 | P1 | 1h | Pending |
| T-NAV-2.7 | P1 | 1h | Pending |
| T-NAV-2.8 | P2 | 1h | Pending |

**Total Estimate**: 10.5 hours
