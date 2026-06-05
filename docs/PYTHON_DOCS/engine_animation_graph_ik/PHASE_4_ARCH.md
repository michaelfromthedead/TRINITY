# PHASE 4 ARCHITECTURE: Full Body IK and Integration

**Phase**: 4 of 4
**Focus**: Full Body IK, Foot Placement, Synchronization, Integration
**Subsystem**: engine/animation/ik (fullbody.py, foot_placement.py) + engine/animation/graph (sync.py)

---

## 1. Phase Scope

Implement high-level IK systems and integration:
- Full body IK with balance maintenance
- Foot placement for terrain adaptation
- Animation synchronization
- Graph/IK integration

---

## 2. Full Body IK Architecture

### 2.1 fullbody.py (~768 lines)

**Purpose**: Multi-effector full body IK with COM balance

```
FullBodyIK
    |
    +-- skeleton: Skeleton
    +-- chains: Dict[str, IKChain]
    +-- pelvis: BoneRef
    +-- spine: List[BoneRef]
    +-- com_calculator: COMCalculator
    |
    +-- solve(goals: List[IKGoal]) -> FullBodyResult
```

**Key Classes:**

| Class | Purpose |
|-------|---------|
| FullBodyIK | Multi-chain coordinator |
| IKChain | Single chain (arm, leg, spine) |
| COMCalculator | Center of mass computation |
| BalanceController | Support polygon checking |
| LookAtSolver | Head/eye tracking |

### 2.2 Balance Maintenance

**Point-in-Polygon Test (Ray Casting):**
```python
def point_in_polygon(point: Vec3, polygon: List[Vec3]) -> bool:
    """
    Ray casting algorithm for support polygon check.
    Cast ray from point, count intersections.
    """
    x, z = point.x, point.z  # Project to ground plane
    inside = False
    
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, zi = polygon[i].x, polygon[i].z
        xj, zj = polygon[j].x, polygon[j].z
        
        if ((zi > z) != (zj > z)) and \
           (x < (xj - xi) * (z - zi) / (zj - zi) + xi):
            inside = not inside
        j = i
    
    return inside
```

**Closest Point on Polygon Edge:**
```python
def closest_point_on_polygon(point: Vec3, polygon: List[Vec3]) -> Vec3:
    """
    Find closest point on polygon boundary for COM correction.
    """
    min_dist = float('inf')
    closest = point
    
    for i in range(len(polygon)):
        j = (i + 1) % len(polygon)
        edge_closest = closest_point_on_segment(point, polygon[i], polygon[j])
        dist = (point - edge_closest).length()
        
        if dist < min_dist:
            min_dist = dist
            closest = edge_closest
    
    return closest
```

### 2.3 LookAtSolver

```python
class LookAtSolver:
    """Head/eye tracking with spine distribution."""
    
    head_bone: BoneRef
    spine_bones: List[BoneRef]
    distribution: List[float]  # How much each spine bone rotates
    
    def solve(self, target: Vec3, weight: float = 1.0) -> List[Quat]:
        """
        Distribute rotation across spine and head.
        Head takes most, spine distributes rest.
        """
        total_angle = self._calculate_required_rotation(target)
        
        rotations = []
        remaining = total_angle * weight
        
        for bone, dist_weight in zip(self.spine_bones, self.distribution):
            bone_angle = remaining * dist_weight
            rotations.append(Quat.from_axis_angle(self._up_axis, bone_angle))
            remaining -= bone_angle
        
        # Head takes the rest
        rotations.append(Quat.from_axis_angle(self._up_axis, remaining))
        
        return rotations
```

---

## 3. Foot Placement Architecture

### 3.1 foot_placement.py (~737 lines)

**Purpose**: Terrain-adaptive foot IK

```
FootPlacement
    |
    +-- left_leg_ik: TwoBoneIK
    +-- right_leg_ik: TwoBoneIK
    +-- raycast: Callable[[Vec3, Vec3], RaycastHit]
    +-- pelvis: BoneRef
    +-- max_pelvis_drop: float
    |
    +-- update(skeleton_pose: Pose, dt: float) -> FootPlacementResult
    
FootPlacementAnimated
    |
    +-- (inherits FootPlacement)
    +-- animation_curves: Dict[str, AnimationCurve]
    
MultiLegFootPlacement
    |
    +-- legs: List[LegConfig]  # For spiders, centaurs, etc.
```

### 3.2 Raycast Interface

```python
@dataclass
class RaycastHit:
    hit: bool
    position: Vec3
    normal: Vec3
    distance: float

RaycastCallback = Callable[[Vec3, Vec3], RaycastHit]
# (origin, direction) -> hit result
```

### 3.3 Pelvis Adjustment

```python
def _calculate_pelvis_offset(self, pelvis_pos, left_target, right_target):
    """
    Calculate required pelvis height offset.
    Move pelvis down so both feet can reach targets.
    """
    SAFETY_MARGIN = 0.9  # Don't extend fully
    
    left_reach = right_reach = 0
    
    if left_target:
        to_target = left_target - pelvis_pos
        leg_reach = self._left_leg_ik.max_reach * SAFETY_MARGIN
        left_reach = max(0, to_target.length() - leg_reach)
    
    if right_target:
        to_target = right_target - pelvis_pos
        leg_reach = self._right_leg_ik.max_reach * SAFETY_MARGIN
        right_reach = max(0, to_target.length() - leg_reach)
    
    # Take worst case
    required_drop = max(left_reach, right_reach)
    return min(required_drop, self.max_pelvis_drop)
```

### 3.4 Multi-Leg Support

```python
@dataclass
class LegConfig:
    hip_bone: BoneRef
    knee_bone: BoneRef
    foot_bone: BoneRef
    toe_bone: Optional[BoneRef]
    
    leg_ik: TwoBoneIK
    ray_offset: Vec3  # Offset from hip for raycast

class MultiLegFootPlacement:
    """N-legged characters (spiders, centaurs, etc.)"""
    
    legs: List[LegConfig]
    body_bone: BoneRef
    
    def update(self, pose: Pose, dt: float) -> MultiLegResult:
        # Find all foot targets
        targets = []
        for leg in self.legs:
            hit = self._raycast_for_leg(leg)
            targets.append(hit.position if hit.hit else None)
        
        # Calculate body adjustment
        body_offset = self._calculate_body_offset(targets)
        
        # Solve each leg
        leg_results = []
        for leg, target in zip(self.legs, targets):
            if target:
                result = leg.leg_ik.solve(target)
                leg_results.append(result)
        
        return MultiLegResult(body_offset, leg_results)
```

---

## 4. Animation Synchronization

### 4.1 sync.py (~672 lines)

**Purpose**: Animation synchronization across clips

```
SyncGroup
    |
    +-- entries: List[SyncEntry]
    +-- mode: SyncMode
    +-- leader: Optional[SyncEntry]
    |
    +-- update(dt: float)
    +-- get_synchronized_time(entry: SyncEntry) -> float
```

**Sync Modes:**
```python
class SyncMode(Enum):
    NONE = "none"           # No synchronization
    NORMALIZED = "normalized"  # Match normalized time [0,1]
    PHASE = "phase"         # Match phase via markers
    LEADER = "leader"       # All follow leader
    WEIGHTED = "weighted"   # Weighted average of times
```

### 4.2 Sync Markers

```python
@dataclass
class SyncMarker:
    name: str
    normalized_time: float  # Position in [0,1]
    
class MarkerTrack:
    markers: List[SyncMarker]
    
    def get_nearest_marker(self, time: float, name: str = None) -> SyncMarker

class EventSynchronizer:
    """Cross-animation event coordination."""
    
    def sync_event(self, event_name: str) -> float:
        """Get synchronized time for event across all entries."""
```

### 4.3 Phase Synchronization

```python
def _sync_phase(self, dt: float):
    """
    Sync followers to leader via markers.
    """
    leader = self.get_leader()
    leader.advance(dt)
    
    if leader.marker_track:
        leader_marker = leader.marker_track.get_nearest_marker(leader.normalized_time)
    
    for entry in self.entries:
        if entry.is_leader:
            continue
        
        if leader_marker and entry.marker_track:
            # Find corresponding marker in follower
            follower_marker = entry.marker_track.get_nearest_marker(
                entry.normalized_time, leader_marker.name)
            
            # Calculate target time based on marker offset
            offset = leader.normalized_time - leader_marker.normalized_time
            target_time = follower_marker.normalized_time + offset
            entry.normalized_time = target_time % 1.0
```

---

## 5. Integration Architecture

### 5.1 Graph + IK Integration

```python
class IKLayer(AnimationLayer):
    """Animation layer that applies IK after graph evaluation."""
    
    ik_solver: Union[TwoBoneIK, FABRIKChain, FullBodyIK]
    goals: List[IKGoal]
    
    def apply(self, pose: Pose, context: GraphContext) -> Pose:
        # Update goals from context (e.g., look-at target)
        updated_goals = self._update_goals_from_context(context)
        
        # Solve IK
        ik_result = self.ik_solver.solve(updated_goals)
        
        # Apply IK result to pose
        return self._apply_ik_to_pose(pose, ik_result)
```

### 5.2 ECS Integration

```python
@component
class FullBodyIKController:
    ik_system: FullBodyIK
    look_at_target: Optional[Vec3]
    foot_placement: FootPlacement
    
@component
class AnimationGraphController:
    graph: AnimationGraph
    ik_layers: List[IKLayer]

@system(phase="animation_late")
class FullBodyIKSystem:
    def update(self, entity, animation, ik_controller):
        # Get base pose from animation graph
        base_pose = animation.current_pose
        
        # Apply foot placement
        foot_result = ik_controller.foot_placement.update(base_pose)
        
        # Apply full body IK
        goals = self._gather_goals(ik_controller)
        ik_result = ik_controller.ik_system.solve(goals)
        
        # Combine results
        animation.current_pose = self._combine(base_pose, foot_result, ik_result)
```

---

## 6. Dependencies

### 6.1 Internal (Phase 1-3)

| Dependency | From |
|------------|------|
| AnimationNode | Phase 1 |
| Pose, Transform | Phase 1 |
| AnimationLayer | Phase 2 |
| TwoBoneIK | Phase 3 |
| FABRIKChain | Phase 3 |
| JacobianIK | Phase 3 |
| IKGoal | Phase 3 |

### 6.2 External

| Dependency | Usage |
|------------|-------|
| Physics raycast | Foot placement |
| ECS system | Integration |
