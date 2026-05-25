# PHASE 2 ARCHITECTURE: State Machines and Blend Trees

**Phase**: 2 of 4
**Focus**: Animation Control Structures
**Subsystem**: engine/animation/graph (state_machine.py, blend_tree.py, blend_node.py)

---

## 1. Phase Scope

Implement animation control structures:
- State machine with conditions and transitions
- 1D/2D/Direct blend trees
- Blend nodes for animation composition
- Layer system for multi-track blending

---

## 2. Module Architecture

### 2.1 state_machine.py (~829 lines)

**Purpose**: Full finite state machine for animation control

```
StateMachine (AnimationNode)
    |
    +-- states: Dict[str, AnimationState]
    +-- transitions: List[StateTransition]
    +-- any_state_transitions: List[StateTransition]
    +-- current_state: AnimationState
    +-- transition_in_progress: Optional[TransitionData]
    |
    +-- evaluate(context) -> Pose
    +-- add_state(state)
    +-- add_transition(transition)
    +-- force_state(state_name)
```

**Key Classes:**

| Class | Purpose |
|-------|---------|
| StateMachine | FSM container, AnimationNode subclass |
| AnimationState | Single state with clip/graph reference |
| StateTransition | Transition between states |
| TransitionCondition | Condition for transition trigger |
| TransitionData | In-progress transition state |
| BlendCurve | Easing function for transitions |

### 2.2 Transition Conditions

```python
class ConditionOperator(Enum):
    EQUALS = "=="
    NOT_EQUALS = "!="
    GREATER = ">"
    GREATER_EQUAL = ">="
    LESS = "<"
    LESS_EQUAL = "<="
    TRIGGER = "trigger"  # One-shot, auto-reset
    EXIT_TIME = "exit_time"  # Based on animation completion

@dataclass
class TransitionCondition:
    parameter: str
    operator: ConditionOperator
    value: Any
    
    def evaluate(self, context: GraphContext) -> bool
```

### 2.3 Blend Curves (6 types)

```python
class BlendCurve(Enum):
    LINEAR = "linear"
    EASE_IN = "ease_in"          # t^2
    EASE_OUT = "ease_out"        # 1 - (1-t)^2
    EASE_IN_OUT = "ease_in_out"  # Hermite
    SMOOTH_STEP = "smooth_step"  # 3t^2 - 2t^3
    SMOOTHER_STEP = "smoother_step"  # 6t^5 - 15t^4 + 10t^3

def evaluate_curve(curve: BlendCurve, t: float) -> float:
    if curve == BlendCurve.SMOOTHER_STEP:
        return t * t * t * (t * (t * 6 - 15) + 10)
    # ... other curves
```

---

## 3. Blend Tree Architecture

### 3.1 blend_tree.py (~849 lines)

**Purpose**: Parameter-driven animation blending

```
BlendTree (AnimationNode)
    |
    +-- BlendTree1D: Single parameter blending
    +-- BlendTree2D: Two parameter blending (4 modes)
    +-- BlendTreeDirect: Explicit weight blending
```

### 3.2 BlendTree1D

```python
@dataclass
class BlendSample1D:
    threshold: float
    clip: AnimationClip
    
class BlendTree1D(AnimationNode):
    parameter: str
    samples: List[BlendSample1D]
    gradient_bands: bool  # Smooth transitions between samples
    
    def evaluate(self, context) -> Pose:
        value = context.get_parameter(self.parameter)
        # Find surrounding samples, compute weights, blend
```

### 3.3 BlendTree2D (4 modes)

```python
class BlendMode2D(Enum):
    CARTESIAN = "cartesian"           # Regular X/Y grid
    POLAR = "polar"                   # Direction/magnitude
    FREEFORM_DIRECTIONAL = "freeform_directional"
    FREEFORM_CARTESIAN = "freeform_cartesian"  # Delaunay

@dataclass
class BlendSample2D:
    position: Tuple[float, float]
    clip: AnimationClip
    
class BlendTree2D(AnimationNode):
    parameter_x: str
    parameter_y: str
    mode: BlendMode2D
    samples: List[BlendSample2D]
    triangulation: List[Triangle]  # For Delaunay modes
```

### 3.4 Delaunay Triangulation (Bowyer-Watson)

```python
def triangulate(points: List[Tuple[float, float]]) -> List[Triangle]:
    """Bowyer-Watson algorithm for Delaunay triangulation."""
    # 1. Create super-triangle containing all points
    triangles = [create_super_triangle(points)]
    
    # 2. Add points incrementally
    for point in points:
        bad_triangles = [t for t in triangles 
                        if point_in_circumcircle(point, t)]
        
        # Find boundary polygon
        polygon = find_polygon_boundary(bad_triangles)
        
        # Remove bad triangles
        triangles = [t for t in triangles if t not in bad_triangles]
        
        # Create new triangles from point to polygon edges
        for edge in polygon:
            triangles.append(Triangle(point, edge[0], edge[1]))
    
    # 3. Remove triangles connected to super-triangle
    return [t for t in triangles if not touches_super_triangle(t)]
```

### 3.5 Barycentric Interpolation

```python
def get_barycentric(point, triangle) -> Tuple[float, float, float]:
    """Compute barycentric coordinates for point in triangle."""
    v0, v1, v2 = triangle.vertices
    
    # Compute vectors
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = point - v0
    
    # Compute dot products
    dot00 = dot(v0v1, v0v1)
    dot01 = dot(v0v1, v0v2)
    dot02 = dot(v0v1, v0p)
    dot11 = dot(v0v2, v0v2)
    dot12 = dot(v0v2, v0p)
    
    # Compute barycentric coordinates
    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    w = 1.0 - u - v
    
    return (max(0, w), max(0, v), max(0, u))
```

---

## 4. Blend Nodes

### 4.1 blend_node.py (~776 lines)

**Node Types (10):**

| Node | Purpose |
|------|---------|
| ClipNode | Single animation clip playback |
| BlendNode | Binary blend between two inputs |
| AdditiveNode | Additive animation on base pose |
| LayerNode | Masked layer blend |
| MirrorNode | Left/right animation mirroring |
| TimeScaleNode | Playback speed modification |
| PoseCacheNode | Cache pose for reuse |
| SelectNode | Switch between inputs |
| LoopNode | Loop control for clips |
| SubGraphNode | Reference to nested graph |

### 4.2 ClipNode

```python
class ClipNode(AnimationNode):
    clip: AnimationClip
    loop_mode: LoopMode  # ONCE, LOOP, PING_PONG
    time_scale: float
    
    current_time: float
    
    def evaluate(self, context) -> Pose:
        # Advance time
        self.current_time += context.dt * self.time_scale
        
        # Handle loop modes
        if self.loop_mode == LoopMode.LOOP:
            self.current_time %= self.clip.duration
        
        # Sample clip at current time
        return self.clip.sample(self.current_time)
```

---

## 5. Layer System

### 5.1 layer.py (~552 lines)

```
LayerStack
    |
    +-- layers: List[AnimationLayer]
    +-- base_pose: Pose
    |
    +-- evaluate(context) -> Pose
    
AnimationLayer
    |
    +-- source: AnimationNode
    +-- mask: Optional[BoneMask]
    +-- blend_mode: LayerBlendMode
    +-- weight: float
```

**Blend Modes:**

| Mode | Operation |
|------|-----------|
| OVERRIDE | Replace base with layer |
| ADDITIVE | Add layer delta to base |
| MULTIPLY | Multiply transforms |

---

## 6. Builder Patterns

### 6.1 StateMachineBuilder

```python
state_machine = (StateMachineBuilder()
    .add_state("idle", idle_clip)
    .add_state("walk", walk_clip)
    .add_state("run", run_clip)
    .add_transition("idle", "walk", 
        condition=param_greater("speed", 0.1),
        duration=0.2,
        curve=BlendCurve.SMOOTH_STEP)
    .add_any_state_transition("death",
        condition=trigger("die"))
    .set_initial("idle")
    .build())
```

### 6.2 LayerStackBuilder

```python
layer_stack = (LayerStackBuilder()
    .set_base(locomotion_graph)
    .add_layer(upper_body_override, 
        mask=BoneMaskPresets.UPPER_BODY,
        weight=1.0)
    .add_layer(facial_additive,
        mask=BoneMaskPresets.HEAD,
        mode=LayerBlendMode.ADDITIVE)
    .build())
```

---

## 7. Dependencies

### 7.1 Internal (Phase 1)

| Dependency | From |
|------------|------|
| AnimationNode | animation_graph.py |
| Pose, Transform | animation_graph.py |
| BoneMask | animation_graph.py |
| GraphContext | animation_graph.py |

### 7.2 External

| Dependency | Usage |
|------------|-------|
| AnimationClip | Animation data (external) |
| math.acos, math.sin | Blend curve evaluation |
