# PHASE 3 TODO: Off-Mesh Navigation and Module Integration

**Scope**: Off-mesh links (jumps, ladders, doors), SmartObjects, module aggregation  
**Files**: `nav_links.py`, `__init__.py`

---

## T-NAV-3.1: Verify NavLink Base Implementation

**Priority**: P0  
**Estimate**: 1 hour

### Description
Verify the base NavLink class correctly stores and interpolates link positions.

### Tasks
- [ ] Review NavLink position storage
- [ ] Verify linear interpolation for basic links
- [ ] Test bidirectional flag handling
- [ ] Verify cost calculation

### Acceptance Criteria
- Links store start/end positions correctly
- Linear interpolation returns correct positions at t=0, 0.5, 1
- Bidirectional links are traversable both ways
- Cost reflects link difficulty

---

## T-NAV-3.2: Verify JumpLink Parabolic Interpolation

**Priority**: P0  
**Estimate**: 1 hour

### Description
Verify JumpLink produces correct parabolic trajectories.

### Tasks
- [ ] Review `_parabolic_interpolate()` formula
- [ ] Verify apex height is reached at t=0.5
- [ ] Test various jump distances and heights
- [ ] Verify start and end positions exact at t=0 and t=1

### Acceptance Criteria
- Trajectory follows parabolic arc
- Apex position correct for configured height
- Interpolation is smooth (no discontinuities)
- Works for horizontal, upward, and downward jumps

---

## T-NAV-3.3: Verify DoorLink State Machine

**Priority**: P0  
**Estimate**: 1.5 hours

### Description
Verify DoorLink state machine handles all transitions correctly.

### Tasks
- [ ] Review state enumeration (Open, Closed, Locked, Unlocked)
- [ ] Verify open/close transitions
- [ ] Verify lock/unlock transitions
- [ ] Test auto-close timer
- [ ] Verify invalid transition handling

### Acceptance Criteria
- All valid state transitions work
- Invalid transitions rejected with error
- Auto-close timer triggers correctly
- State persists through serialization

---

## T-NAV-3.4: Verify LadderLink Implementation

**Priority**: P1  
**Estimate**: 1 hour

### Description
Verify LadderLink correctly handles climb movement with rung positions.

### Tasks
- [ ] Review rung position calculation
- [ ] Verify climb speed affects traversal time
- [ ] Test interpolation between rungs
- [ ] Verify top/bottom entry handling

### Acceptance Criteria
- Interpolation visits each rung position
- Climb speed scales traversal duration
- Works for ladders of various heights
- Entry from top and bottom both work

---

## T-NAV-3.5: Verify Spatial Index for Links

**Priority**: P1  
**Estimate**: 1 hour

### Description
Verify spatial indexing enables efficient link queries.

### Tasks
- [ ] Review spatial index structure (grid or tree)
- [ ] Verify insert/remove operations
- [ ] Test point queries (links near position)
- [ ] Test bounds queries (links along path)

### Acceptance Criteria
- Point queries return links within radius
- Bounds queries return links in bounding box
- Query performance O(1) or O(log n) average
- Index stays consistent after inserts/removes

---

## T-NAV-3.6: Verify SmartObject Slot Reservation

**Priority**: P0  
**Estimate**: 1.5 hours

### Description
Verify SmartObject slot reservation prevents conflicts.

### Tasks
- [ ] Review slot data structure
- [ ] Verify `reserve_slot()` returns available slot
- [ ] Verify `reserve_slot()` returns None when full
- [ ] Test `release_slot()` frees correctly
- [ ] Test concurrent reservation attempts

### Acceptance Criteria
- Only one agent can reserve a slot
- Reserved slot not returned to other agents
- Release makes slot available again
- Availability query reflects current state

---

## T-NAV-3.7: Verify Unified Pathfinding Integration

**Priority**: P0  
**Estimate**: 1.5 hours

### Description
Verify the unified pathfinding entry point correctly combines NavMesh paths with off-mesh links.

### Tasks
- [ ] Review `unified_find_path()` implementation
- [ ] Verify link insertion into paths
- [ ] Test paths requiring link traversal
- [ ] Verify path simplification with links

### Acceptance Criteria
- Paths cross links when optimal
- Link segments included in output path
- Path is continuous (no gaps)
- Simplification does not remove necessary waypoints

---

## T-NAV-3.8: Verify A* Integration in Module

**Priority**: P1  
**Estimate**: 1 hour

### Description
Verify the module-level A* search properly integrates with NavMesh and links.

### Tasks
- [ ] Review NavigationManager A* wiring
- [ ] Verify polygon neighbors include link endpoints
- [ ] Test path finding with and without links
- [ ] Verify cost comparison (polygon vs. link)

### Acceptance Criteria
- A* considers both polygons and links
- Cheaper routes selected regardless of type
- Unreachable goals handled correctly
- Performance acceptable with many links

---

## T-NAV-3.9: Verify Serialization Round-Trip

**Priority**: P2  
**Estimate**: 1 hour

### Description
Verify all navigation state serializes and deserializes correctly.

### Tasks
- [ ] Test NavMesh serialization
- [ ] Test link state serialization (door states)
- [ ] Test SmartObject reservation serialization
- [ ] Verify loaded state matches saved state

### Acceptance Criteria
- All data survives round-trip
- Door states preserved
- Reservations optionally cleared on load (game design choice)
- No data corruption or loss

---

## Summary

| Task | Priority | Estimate | Status |
|------|----------|----------|--------|
| T-NAV-3.1 | P0 | 1h | Pending |
| T-NAV-3.2 | P0 | 1h | Pending |
| T-NAV-3.3 | P0 | 1.5h | Pending |
| T-NAV-3.4 | P1 | 1h | Pending |
| T-NAV-3.5 | P1 | 1h | Pending |
| T-NAV-3.6 | P0 | 1.5h | Pending |
| T-NAV-3.7 | P0 | 1.5h | Pending |
| T-NAV-3.8 | P1 | 1h | Pending |
| T-NAV-3.9 | P2 | 1h | Pending |

**Total Estimate**: 10.5 hours
