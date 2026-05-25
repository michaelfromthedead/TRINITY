# PHASE 3 ARCHITECTURE: Off-Mesh Navigation and Module Integration

**Scope**: Off-mesh links (jumps, ladders, doors), SmartObjects, module aggregation  
**Files**: `nav_links.py`, `__init__.py`  
**Lines**: ~1,986

---

## Architecture Overview

Phase 3 covers special navigation cases that cannot be represented by NavMesh polygons, plus the module-level integration that ties all navigation components together.

```
                    +------------------+
                    |     NavMesh      |
                    +--------+---------+
                             |
         +-------------------+-------------------+
         |                   |                   |
   +-----v-----+       +-----v-----+       +-----v-----+
   |  NavLinks |       | SmartObjects|      |   Module  |
   +-----------+       +-----------+       | Integration|
         |                   |             +-----------+
   +-----v-----+       +-----v-----+             |
   | JumpLinks |       |   Slots   |       +-----v-----+
   | DoorLinks |       | Reservations|     |A* Search  |
   |LadderLinks|       +-----------+       |  (unified)|
   +-----------+                           +-----------+
```

---

## Component Architecture

### Off-Mesh Links (`nav_links.py`)

```
NavLinkManager
├── NavLink (base)
│   ├── start_position, end_position
│   ├── cost, bidirectional
│   └── interpolate(t) -> Position
├── JumpLink
│   ├── apex_height
│   └── _parabolic_interpolate(t) -> Position
├── DoorLink
│   ├── State: Open, Closed, Locked, Unlocked
│   ├── auto_close_time
│   └── interact() -> StateTransition
├── LadderLink
│   ├── climb_speed
│   ├── rung_positions
│   └── interpolate(t) -> Position (per-rung)
└── SpatialIndex
    ├── insert(link, bounds)
    ├── query_point(position, radius) -> Links
    └── query_path(start, end) -> Links
```

**State Machine for Doors**:
```
           ┌─────────────┐
           │   Locked    │
           └──────┬──────┘
                  │ unlock()
           ┌──────v──────┐
           │   Closed    │◄──────────┐
           └──────┬──────┘           │
                  │ open()           │ auto_close
           ┌──────v──────┐           │
           │    Open     │───────────┘
           └─────────────┘
```

### SmartObjects and Module Integration (`__init__.py`)

```
NavigationModule
├── SmartObject
│   ├── slots: List[Slot]
│   ├── reserve_slot(agent) -> Slot | None
│   ├── release_slot(agent)
│   └── is_available() -> bool
├── Slot
│   ├── position, rotation
│   ├── reserved_by: Agent | None
│   └── interaction_data
├── NavigationManager
│   ├── navmesh: NavMesh
│   ├── pathfinder: PathfindingEngine
│   ├── avoidance: CollisionAvoidanceEngine
│   ├── steering: SteeringEngine
│   ├── links: NavLinkManager
│   └── smart_objects: SmartObjectManager
└── unified_find_path(start, goal) -> Path
    ├── Check NavMesh path
    ├── Insert off-mesh links
    └── Apply path simplification
```

---

## Algorithm Details

### Parabolic Jump Interpolation

```
Input: Start position (p0), End position (p1), Apex height (h), Progress t [0,1]
  |
  v
Compute Apex: apex = (p0 + p1) / 2 + (0, h, 0)
  |
  v
Split at t=0.5:
  - t < 0.5: lerp(p0, apex, t * 2)
  - t >= 0.5: lerp(apex, p1, (t - 0.5) * 2)
  |
  v
Output: Interpolated position on parabolic arc
```

**Alternative (True Parabola)**:
```python
def parabolic(t):
    x = lerp(p0.x, p1.x, t)
    z = lerp(p0.z, p1.z, t)
    y = p0.y + (p1.y - p0.y) * t + 4 * h * t * (1 - t)
    return Vector3(x, y, z)
```

### Door State Transitions

```
lock():   Open -> Closed -> Locked
unlock(): Locked -> Closed
open():   Closed -> Open
close():  Open -> Closed

auto_close: Open --(timer)--> Closed
```

**Interaction Requirements**:
- `open()`: Requires Closed state, agent proximity
- `close()`: Requires Open state, agent proximity
- `lock()`: Requires key item, Closed state
- `unlock()`: Requires key item, Locked state

### Slot Reservation

```
Input: Agent requesting interaction with SmartObject
  |
  v
Query Available Slots: Filter slots where reserved_by is None
  |
  v
Select Best Slot: Choose nearest to agent position
  |
  v
Reserve: Set slot.reserved_by = agent
  |
  v
Output: Reserved slot or None if all occupied
```

**Reservation Lifecycle**:
1. Agent approaches SmartObject
2. Agent calls `reserve_slot()` -> gets Slot
3. Agent navigates to slot.position
4. Agent performs interaction
5. Agent calls `release_slot()` -> slot freed

### Unified Pathfinding

```
Input: Start position, Goal position
  |
  v
NavMesh Path: Find polygon-to-polygon path
  |
  v
Insert Links: For each link along path, insert link segment
  |
  v
Simplify: Funnel algorithm / RDP
  |
  v
Output: Waypoint sequence including link traversals
```

---

## Architectural Decisions

### ADR-NAV-009: NavLinks as First-Class Graph Edges

**Context**: Off-mesh connections must integrate with pathfinding.

**Decision**: NavLinks are edges in the navigation graph, costed like polygon edges.

**Rationale**:
- Unified A* traverses both polygons and links
- Link cost affects route selection
- No special-case pathfinding code

**Consequences**:
- Links must be discoverable from polygon queries
- Link cost must be comparable to polygon traversal cost

### ADR-NAV-010: Parabolic Interpolation for Jumps

**Context**: Jump links need realistic trajectories.

**Decision**: Use parabolic interpolation with configurable apex height.

**Rationale**:
- Parabolas match projectile motion
- Apex height is intuitive for designers
- Simple formula, no physics simulation needed

**Consequences**:
- Must handle variable jump distances
- Apex height may need per-agent adjustment

### ADR-NAV-011: State Machine for Door Links

**Context**: Doors have multiple states and complex transitions.

**Decision**: Implement explicit state machine with transition rules.

**Rationale**:
- States are clearly defined
- Transitions are explicit and auditable
- Auto-close timer integrates naturally

**Consequences**:
- Must handle concurrent access (multiple agents)
- State persistence for save/load

### ADR-NAV-012: SmartObject Slot Reservation

**Context**: Multiple agents may want to interact with same object.

**Decision**: Slots with reservation system prevents conflicts.

**Rationale**:
- Slots are explicit interaction positions
- Reservation prevents overlap
- Release mechanism handles cleanup

**Consequences**:
- Must handle agent death/disconnect (orphaned reservations)
- Timeout may be needed for stuck agents

### ADR-NAV-013: Spatial Index for Link Queries

**Context**: Path must efficiently find relevant links.

**Decision**: Use spatial index (grid or tree) for link lookup.

**Rationale**:
- O(1) average lookup vs. O(n) linear scan
- Bounds queries for path segments
- Point queries for agent proximity

**Consequences**:
- Must update index when links change
- Memory overhead for index structure

---

## Performance Considerations

### Link Queries

- **Spatial index**: Grid for uniform distributions, quadtree for clustered
- **Query bounds**: Use path bounding box for relevant links
- **Link caching**: Cache links per polygon for frequent queries

### SmartObject Management

- **Active tracking**: Only process objects near agents
- **Reservation cleanup**: Timeout stale reservations
- **Slot proximity**: Use spatial index for nearest slot queries

### Unified Pathfinding

- **Link insertion**: Lazy evaluation (only check links along path)
- **Path caching**: Cache paths for common start/goal pairs
- **Incremental updates**: Re-plan only affected path segments

---

## Integration Points

### Links -> Pathfinding

- Links become edges in navigation graph
- Path includes link traversal segments
- Link cost affects route selection

### Links -> Animation

- `interpolate(t)` provides position for animation playback
- Door state triggers animation state changes
- Ladder rungs may trigger step animations

### SmartObjects -> AI

- AI queries available slots before approaching
- Reservation ensures exclusive access
- Slot position/rotation informs AI positioning

### Module -> Game Systems

- Unified `find_path()` is entry point for all navigation
- Event hooks for link state changes
- Save/load serializes link states and reservations
