# PHASE 1 ARCHITECTURE: Core Animation Graph

**Phase**: 1 of 4
**Focus**: Animation Graph Foundation
**Subsystem**: engine/animation/graph (core modules)

---

## 1. Phase Scope

Establish the foundational animation graph architecture including:
- Animation node base class and metaclass
- Pose and transform data structures
- Skeleton and bone hierarchy
- Bone mask system
- Graph context for evaluation

---

## 2. Module Architecture

### 2.1 animation_graph.py (~1040 lines)

**Purpose**: Core DAG container and base infrastructure

```
AnimationGraph
    |
    +-- nodes: Dict[str, AnimationNode]
    +-- connections: List[Connection]
    +-- parameters: Dict[str, Parameter]
    +-- subgraphs: Dict[str, AnimationGraph]
    |
    +-- evaluate(context: GraphContext) -> Pose
    +-- add_node(node: AnimationNode)
    +-- connect(source, target, slot)
```

**Key Classes:**

| Class | Purpose | Lines (approx) |
|-------|---------|----------------|
| AnimationGraph | DAG container | 200 |
| AnimationNode | Base node with metaclass | 150 |
| GraphNodeMeta | Auto-registration metaclass | 50 |
| GraphContext | Evaluation context | 100 |
| Connection | Node connections | 50 |
| Parameter | Typed graph parameters | 80 |

### 2.2 Pose and Transform

**Transform Structure:**
```python
@dataclass
class Transform:
    position: Vec3
    rotation: Quat  # Quaternion
    scale: Vec3
    
    def blend(self, other: Transform, t: float) -> Transform
    def compose(self, other: Transform) -> Transform
```

**Pose Structure:**
```python
@dataclass
class Pose:
    transforms: Dict[str, Transform]  # Bone name -> transform
    skeleton: Skeleton
    
    def blend(self, other: Pose, t: float) -> Pose
    def apply_mask(self, mask: BoneMask) -> Pose
```

### 2.3 Skeleton and Bones

```
Skeleton
    |
    +-- root: Bone
    +-- bones: Dict[str, Bone]
    +-- bone_count: int
    |
    +-- get_bone(name: str) -> Bone
    +-- get_chain(start, end) -> List[Bone]
    
Bone
    |
    +-- name: str
    +-- parent: Optional[Bone]
    +-- children: List[Bone]
    +-- bind_pose: Transform
```

### 2.4 Bone Masks

```
BoneMask
    |
    +-- weights: Dict[str, float]  # Bone name -> weight [0,1]
    +-- mode: BlendMode
    |
    +-- apply(pose: Pose) -> Pose
    +-- combine(other: BoneMask) -> BoneMask
    
BoneMaskPresets:
    - UPPER_BODY
    - LOWER_BODY
    - LEFT_ARM
    - RIGHT_ARM
    - LEFT_LEG
    - RIGHT_LEG
    - GRADIENT(start_bone, falloff)
```

---

## 3. Key Algorithms

### 3.1 Quaternion SLERP

```python
def _slerp(q1, q2, t) -> Quat:
    dot = q1.dot(q2)
    
    # Take shorter path
    if dot < 0:
        q2 = -q2
        dot = -dot
    
    # Linear interpolation for near-parallel
    if dot > SLERP_DOT_THRESHOLD:
        return (q1 + (q2 - q1) * t).normalized()
    
    # Spherical interpolation
    theta_0 = acos(clamp(dot, -1, 1))
    theta = theta_0 * t
    
    s0 = cos(theta) - dot * sin(theta) / sin(theta_0)
    s1 = sin(theta) / sin(theta_0)
    
    return (q1 * s0 + q2 * s1).normalized()
```

### 3.2 DAG Cycle Detection

```python
def detect_cycles(graph: AnimationGraph) -> List[str]:
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in graph.nodes}
    cycles = []
    
    def dfs(node):
        color[node] = GRAY
        for neighbor in graph.get_successors(node):
            if color[neighbor] == GRAY:
                cycles.append(f"Cycle detected: {node} -> {neighbor}")
            elif color[neighbor] == WHITE:
                dfs(neighbor)
        color[node] = BLACK
    
    for node in graph.nodes:
        if color[node] == WHITE:
            dfs(node)
    
    return cycles
```

---

## 4. Configuration

### 4.1 config.py Constants

```python
class QuaternionConfig:
    SLERP_DOT_THRESHOLD = 0.9995  # Fallback to lerp threshold
    NORMALIZATION_EPSILON = 1e-6

class GraphConfig:
    MAX_EVALUATION_DEPTH = 100    # Prevent infinite recursion
    CYCLE_DETECTION_ENABLED = True
    
class BlendConfig:
    WEIGHT_EPSILON = 0.001        # Minimum blend weight
    NORMALIZE_WEIGHTS = True
```

---

## 5. Dependencies

### 5.1 External

| Dependency | Usage |
|------------|-------|
| engine.core.math.vec.Vec3 | Position, scale vectors |
| engine.core.math.quat.Quat | Rotation quaternions |
| engine.core.math.transform.Transform | Optional: may use internal |
| engine.core.constants.MATH_EPSILON | Numerical stability |

### 5.2 Internal (None for Phase 1)

Phase 1 establishes the foundation; no internal dependencies.

---

## 6. Integration Points

### 6.1 ECS Integration

```python
@component
class AnimationGraphController:
    graph: AnimationGraph
    context: GraphContext
    current_pose: Pose

@system(phase="animation")
class AnimationGraphSystem:
    def update(self, entity, controller):
        controller.current_pose = controller.graph.evaluate(controller.context)
```

### 6.2 Output to Renderer

```python
def export_bone_matrices(pose: Pose) -> List[Matrix4x4]:
    """Convert pose to bone matrices for GPU skinning."""
    matrices = []
    for bone_name, transform in pose.transforms.items():
        matrices.append(transform.to_matrix())
    return matrices
```

---

## 7. Validation Rules

| Rule | Check |
|------|-------|
| No cycles in graph | detect_cycles() returns empty |
| All connections valid | Source and target nodes exist |
| Parameters typed | Each parameter has explicit type |
| Skeleton valid | Root bone exists, no orphans |
| Bone mask weights | All weights in [0, 1] range |
