# MASTER: engine_gameplay_abilities_ai_camera

**Version**: 1.0
**Last Updated**: 2026-05-23
**Total Lines**: ~18,643

---

## 1. Executive Summary

Three gameplay subsystems forming the high-level game logic layer of the Trinity engine. All are production-ready with no stubs:

| Subsystem | Lines | Status | Core Capability |
|-----------|-------|--------|-----------------|
| Abilities | 3,136 | REAL | GAS-style effects, attributes, targeting, tags |
| AI | 4,523 | REAL | BT, GOAP, Utility AI, Perception, Combat |
| Camera | 7,060 | REAL | 8 modes, collision, effects, rails, blending |

---

## 2. Abilities Subsystem

### 2.1 Attribute System

**File**: `attributes.py` (592 lines)

**Components**:
- `Attribute`: Base/current values, modifiers, bounds, dirty caching
- `AttributeModifier`: Operations (ADD, MULTIPLY, OVERRIDE, STACKING)
- `AttributeSet`: Collection with dependency tracking
- `DerivedAttribute`: Formula-based computed attributes

**Modifier Order of Operations**:
1. ADD_BASE
2. MULTIPLY_BASE
3. ADD_BONUS
4. MULTIPLY_BONUS
5. OVERRIDE
6. Clamp to bounds

### 2.2 Effect System

**File**: `effects.py` (829 lines)

**Effect Types**:
| Type | Behavior | Use Case |
|------|----------|----------|
| InstantEffect | One-shot attribute change | Damage, heal |
| DurationEffect | Time-limited with tick | Buffs, debuffs |
| InfiniteEffect | Until explicitly removed | Passives |
| PeriodicEffect | DOT/HOT with tick rate | Poison, regen |

**Effect Lifecycle**: APPLY -> TICK (if duration > 0) -> REMOVE

**EffectContainer**: Manages active effects on an entity with stacking rules.

### 2.3 Targeting System

**File**: `targeting.py` (823 lines)

**Targeting Modes**:
| Mode | Description |
|------|-------------|
| SelfTargeting | Self-only |
| ActorTargeting | Single actor selection |
| PointTargeting | World position |
| AreaTargeting | AOE with shape |
| ConfirmationTargeting | Wrapper requiring confirm |

**Area Shapes**:
- Circle: Distance-squared check
- Cone: Dot product angle check
- Rectangle: Axis projection
- Line: Point-to-segment projection
- Capsule: Swept sphere

### 2.4 Gameplay Tags

**File**: `tags.py` (575 lines)

**Components**:
- `GameplayTag`: Hierarchical (e.g., `ability.offensive.fire`)
- `GameplayTagContainer`: Collection with matching/filtering
- `GameplayTagQuery`: Complex queries (all_of, any_of, none_of)
- Registry with LRU caching

**Matching Features**:
- Exact match
- Single wildcard: `ability.*.fire`
- Trailing wildcard: `ability.*`
- Ancestor traversal

---

## 3. AI Subsystem

### 3.1 Behavior Tree

**Files**: `behavior_tree.py` (948 lines), `__init__.py` (partial)

**Node Types (14 total)**:

| Category | Nodes |
|----------|-------|
| Composite | Sequence, Selector, Parallel |
| Decorator | Invert, Repeat, Timeout, Cooldown, Retry, ForceSuccess, ForceFailure |
| Leaf | Action, Condition, BlackboardCondition, Wait, SetBlackboard |

**Parallel Policies**:
- REQUIRE_ALL: All children must succeed
- REQUIRE_ONE: One child success is enough
- REQUIRE_MAJORITY: >50% must succeed

**Context System**: BTContext with delta_time, blackboard, entity, debug tracing. Depth limit of 100.

### 3.2 GOAP (Goal-Oriented Action Planning)

**File**: `goap.py` (727 lines)

**Components**:
- `WorldState`: Immutable dict-based with hashing
- `Goal`: Target conditions with priority
- `GOAPAction`: Preconditions, effects, costs, procedural checks
- `GOAPPlanner`: A* search with heuristic (unsatisfied condition count)
- `GOAPAgent`: Full lifecycle with replan-on-failure

**Plan Caching**: 100 plans, 5s TTL.

### 3.3 Utility AI

**File**: `utility_ai.py` (711 lines)

**Response Curves (8 types)**:
| Curve | Formula |
|-------|---------|
| Linear | `y = x` |
| Quadratic | `y = x^2` |
| Exponential | `y = e^(kx)` |
| Logistic | `y = 1 / (1 + e^(-slope*x))` |
| Sine | `y = sin(x * pi/2)` |
| Inverse | `y = 1 - x` |
| Step | `y = x > threshold ? 1 : 0` |
| Smoothstep | `y = x^2 * (3 - 2x)` |

**Consideration Types**:
- Blackboard consideration
- Function consideration
- Distance consideration
- Health consideration

**Scoring**: Geometric mean with compensation factor `(1 - score) * (1 - 1/n)`. Momentum (default 0.1) prevents action thrashing.

### 3.4 Perception

**Source**: `__init__.py`

**Stimulus Properties**:
- Source, sense type, position, strength, timestamp, age

**Sense Types**: SIGHT, HEARING, DAMAGE, SQUAD, TOUCH, SMELL

**Memory**: Known targets persist 3x longer than raw stimuli. Configurable decay rate.

### 3.5 Blackboard

**File**: `blackboard.py` (496 lines)

**Features**:
- Namespaced keys (e.g., `combat.target`)
- Observer pattern with pattern matching
- TTL with automatic cleanup
- Scopes for focused access
- TypedBlackboardKey[T] for type safety

### 3.6 Combat AI

**Source**: `__init__.py`

**Behaviors**: ATTACK, DEFEND, FLANK, RETREAT, SUPPORT, COVER, SUPPRESS, ADVANCE, HOLD_POSITION

**Threat System**:
- ThreatAssessment with threat_level, distance, visibility
- Target priorities: NEAREST, WEAKEST, STRONGEST, HIGHEST_THREAT

**Health Retreat Threshold**: Default 25%

---

## 4. Camera Subsystem

### 4.1 Camera Controllers

**File**: `controller.py` (1,660 lines)

**8 Controllers**:
| Controller | Features |
|------------|----------|
| FirstPerson | Head bob, FOV |
| ThirdPerson | Boom arm with lag, pitch limits |
| Orbit | Zoom/pitch limits, auto-rotate |
| Follow | Lead prediction, offset |
| Free | WASD movement |
| Cinematic | Keyframe timeline |
| TopDown | Pan limits, zoom |
| Isometric | 45-degree snap rotation |

**Camera Lag Formula**: `lag_factor = 1.0 - exp(-lag_speed * dt)`

### 4.2 Collision

**File**: `collision.py` (709 lines)

**Response Modes**:
| Mode | Behavior |
|------|----------|
| Pull-in | Move camera closer |
| Push-out | Move camera away |
| Fade | Fade occluding object |
| Clip | Clip near plane |
| Blend | Interpolate position |

**Sphere Cast**: 9 rays (center + 8 offsets along perpendicular axes)

**OcclusionDetector**: Fade states with hysteresis

### 4.3 Camera Effects

**File**: `effects.py` (1,317 lines)

**Shake Types (7)**:
- Perlin: Octave noise layering
- Sine: Sinusoidal oscillation
- Random: Random displacement
- Directional: Along specific axis
- Explosion: Radial falloff
- Impact: Impulse decay
- Continuous: Persistent shake

**Other Effects**:
- FOVEffect: Modifier stack for punch/zoom
- TiltEffect: Dutch angle
- DOFEffect: Auto-focus circle of confusion
- MotionBlur: Velocity tracking
- VignetteEffect: Edge darkening

**Perlin Shake**: Octaves with persistence decay

### 4.4 Blending

**File**: `blending.py` (1,091 lines)

**Blend Curves (12)**:
Linear, Ease-in, Ease-out, Ease-in-out, Cubic, Exponential, Elastic (overshoot), Bounce, Custom

**Elastic Formula**: `pow(2, 10*(t-1)) * sin((t-s)*2pi/p)`

**Bounce**: Piecewise quadratic segments

**BlendStack**: Concurrent overlapping blends

**Split-Screen Layouts (7)**: Single, Horizontal 2-way, Vertical 2-way, Quad, Triple, PIP

**CameraPriority**: Priority-based camera selection

### 4.5 Rails

**File**: `rails.py` (1,346 lines)

**Spline Types**:
| Type | Description |
|------|-------------|
| Linear | Straight segments |
| Catmull-Rom | Tension-adjusted, 4 control points |
| Bezier | Cubic Bezier |
| Hermite | h00/h10/h01/h11 basis with tangents |

**Arc-Length Parameterization**: Binary search for uniform t mapping

**Tools**:
- RailFollower with loop modes
- TriggerVolume/BlendRegion
- Dolly helper
- Crane helper

---

## 5. Architecture Patterns

| Pattern | Where Used |
|---------|------------|
| Strategy | Targeting modes, spline types, collision responses |
| Observer | Blackboard change notifications, camera state callbacks |
| Composite | Behavior tree nodes |
| State Machine | Effect lifecycle, blend progress |
| Factory | `create_aoe()`, `instant_damage()`, `stat_buff()` |
| Builder | `GOAPPlanner.add_action()` chaining |

---

## 6. Dependencies

```
engine.core.math.vec.Vec3
engine.core.math.quat.Quat
engine.core.math.mat.Mat4
engine.simulation.physics.PhysicsWorld
engine.gameplay.components.transform.TransformComponent
engine.gameplay.abilities.constants
engine.gameplay.camera.constants
engine.gameplay.constants (AI)
engine.gameplay.entity.Actor
```

---

## 7. Integration Points with Trinity

| Trinity Layer | Integration |
|---------------|-------------|
| Metaclasses | ComponentMeta registers Attribute, Effect, CameraController |
| Descriptors | TrackedDescriptor on dirty fields (attributes, camera state) |
| Decorators | @component, @system applied to gameplay classes |
| Foundation | Registry, Tracker, EventLog for change notifications |

---

## 8. Quality Indicators

**Present in All Files**:
1. TYPE_CHECKING imports
2. `@dataclass(slots=True)`
3. Abstract base classes with `@abstractmethod`
4. Comprehensive `__all__` exports
5. Separate constants modules
6. Property decorators with caching
7. Full type annotations
8. Docstrings with Args/Returns
