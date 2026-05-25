# PHASE 2 ARCHITECTURE: Broadphase Acceleration

---

## Phase Overview

Replace O(n^2) broadphase collision detection with a spatial acceleration structure. The investigation identified BVH and octree as enumerated options. This phase implements BVH as the primary structure.

---

## Architectural Decisions

### AD-2.1: BVH Over Octree

**Decision**: Implement Bounding Volume Hierarchy (BVH) as the primary broadphase.

**Rationale**:
- BVH adapts to object distribution (no wasted space)
- Better for dynamic scenes with varying density
- Well-studied incremental update algorithms exist
- Matches investigation's enum options

**Alternatives Considered**:
- Octree: Fixed structure, wastes space in non-uniform distributions
- Grid: Requires known bounds, poor for large scale differences
- Sweep-and-Prune: Good for mostly static scenes, complex for general use

### AD-2.2: Top-Down SAH Construction

**Decision**: Use Surface Area Heuristic (SAH) for BVH construction.

**Rationale**:
- SAH minimizes expected traversal cost
- Industry standard for ray tracing, proven for collision
- One-time cost at construction, amortized over many queries

**Construction Algorithm**:
1. Compute centroid bounds of all objects
2. Find axis with largest centroid spread
3. For each candidate split point: cost = SA(left) * N(left) + SA(right) * N(right)
4. Choose split with minimum cost
5. Recurse until leaf (1-4 objects per leaf)

### AD-2.3: AABB Bounding Volume

**Decision**: Use axis-aligned bounding boxes (AABB) for BVH nodes.

**Rationale**:
- Fast overlap test (6 comparisons)
- Fast union operation
- Existing collision_shapes already compute AABBs
- Sufficient for broadphase (narrowphase handles tight bounds)

### AD-2.4: Incremental Update Strategy

**Decision**: Mark-and-refit for moving objects, full rebuild threshold.

**Algorithm**:
- Object moves: recompute leaf AABB, refit ancestors
- Refit cost exceeds threshold (e.g., 30% of nodes): full rebuild
- Rebuild is O(n log n), acceptable at threshold

**Rationale**: Avoids rebuild every frame while preventing degenerate trees.

---

## Component Boundaries

### New Components

| Component | Responsibility |
|-----------|----------------|
| `BVHNode` | Tree node: AABB, children, leaf data |
| `BVH` | Tree container: build, query, update |
| `SAHBuilder` | Construction algorithm |
| `BVHBroadphase` | Adapter implementing broadphase interface |

### Modified Components

| Component | Change |
|-----------|--------|
| `physics_world.py` | Use BVHBroadphase instead of O(n^2) loop |

---

## Interfaces

### BVHNode

```python
@dataclass
class BVHNode:
    aabb: AABB
    left: Optional['BVHNode']  # None for leaf
    right: Optional['BVHNode']  # None for leaf
    objects: Optional[List[int]]  # Leaf only: object indices
    
    @property
    def is_leaf(self) -> bool:
        return self.objects is not None
```

### BVH

```python
class BVH:
    def __init__(self, objects: List[AABB], max_leaf_size: int = 4):
        """Build BVH from list of AABBs."""
        
    def query_pairs(self) -> List[Tuple[int, int]]:
        """Return all overlapping pairs."""
        
    def query_aabb(self, aabb: AABB) -> List[int]:
        """Return all objects overlapping query AABB."""
        
    def update(self, index: int, new_aabb: AABB) -> None:
        """Update single object's AABB, refit tree."""
        
    def rebuild_if_needed(self) -> bool:
        """Rebuild tree if quality threshold exceeded."""
```

### BVHBroadphase (adapter)

```python
class BVHBroadphase:
    """Implements physics_world broadphase interface using BVH."""
    
    def __init__(self, world: 'PhysicsWorld'):
        self._bvh: Optional[BVH] = None
        
    def find_pairs(self, bodies: List['RigidBody']) -> List[Tuple[int, int]]:
        """Return candidate collision pairs."""
```

---

## Data Flow

```
physics_world.step()
    |
    v
BVHBroadphase.find_pairs(bodies)
    |
    +-- Build/update BVH from body AABBs
    |
    +-- BVH.query_pairs()
    |       |
    |       +-- Stack-based tree traversal
    |       |
    |       +-- Node-node AABB overlap test
    |       |
    |       +-- Collect leaf-leaf overlaps
    |
    v
Return candidate pairs to narrowphase
```

---

## Algorithmic Details

### SAH Cost Function

```
cost(split) = SA(left) * N(left) + SA(right) * N(right)
```

Where:
- SA(node) = surface area of node's AABB
- N(node) = number of objects in node

### Tree Traversal for Pairs

```
stack = [(root, root)]
while stack:
    nodeA, nodeB = stack.pop()
    if not overlap(nodeA.aabb, nodeB.aabb):
        continue
    if nodeA.is_leaf and nodeB.is_leaf:
        yield from cartesian_product(nodeA.objects, nodeB.objects)
    elif nodeA.is_leaf:
        stack.extend([(nodeA, nodeB.left), (nodeA, nodeB.right)])
    elif nodeB.is_leaf:
        stack.extend([(nodeA.left, nodeB), (nodeA.right, nodeB)])
    else:
        # Both internal: push all four child pairs
        ...
```

### Refit Algorithm

```
def refit(node):
    if node.is_leaf:
        node.aabb = union(object.aabb for object in node.objects)
    else:
        refit(node.left)
        refit(node.right)
        node.aabb = union(node.left.aabb, node.right.aabb)
```

---

## Performance Targets

| Metric | O(n^2) Baseline | BVH Target | Improvement |
|--------|-----------------|------------|-------------|
| 100 bodies | 0.5ms | 0.1ms | 5x |
| 500 bodies | 8.3ms | 0.4ms | 20x |
| 1000 bodies | 32.1ms | 0.8ms | 40x |

(Targets assume uniform distribution. Clustered scenes will show less improvement.)

---

## Dependencies

### Internal

- `engine/simulation/physics/collision_shapes.py` (AABB computation)
- `engine/simulation/physics/physics_world.py` (integration point)

### External

- None

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Tree degenerates over time | MEDIUM | Rebuild threshold triggers full SAH rebuild |
| SAH construction slow for large N | LOW | One-time cost, cache between frames |
| Clustered objects defeat BVH | MEDIUM | Fall back to O(n^2) for small N (< 16) |
