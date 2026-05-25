# PHASE 2 ARCHITECTURE: Animation Module

## Scope

5 files, ~4,233 lines in `engine/ui/animation/`

| File | Lines | Purpose |
|------|-------|---------|
| animator.py | 928 | State machine animation controller |
| tween.py | 899 | Property tweening with interpolation |
| keyframe.py | 890 | Keyframe animation tracks |
| triggers.py | 885 | Animation trigger system |
| easing.py | 631 | Easing function library |

---

## Component Architecture

### Animator (animator.py)

```
Animator (state machine controller)
    |
    +-- AnimatorState[]
    |       +-- animation: Animation
    |       +-- transitions: AnimationTransition[]
    |
    +-- AnimationTransition
    |       +-- source: AnimatorState
    |       +-- target: AnimatorState
    |       +-- condition: Callable[[], bool]
    |       +-- duration: float
    |
    +-- AnimationLayer[]
    |       +-- blend_mode: BlendMode
    |       +-- weight: float
    |       +-- priority: int
    |
    +-- AnimationState (runtime)
            +-- current_state: AnimatorState
            +-- progress: float
            +-- is_transitioning: bool
```

**Blend Modes**:
- OVERRIDE: Replace lower layer values entirely
- ADDITIVE: Add to lower layer values (v = lower + upper * weight)
- MULTIPLY: Scale lower layer values (v = lower * upper * weight)
- AVERAGE: Weighted average (v = (lower * (1-w) + upper * w))

---

### Tween (tween.py)

```
Tween[T] (generic property interpolator)
    |
    +-- start_value: T
    +-- end_value: T
    +-- duration: float
    +-- easing: EasingFunction
    +-- delay: float
    +-- repeat: int (0 = once, -1 = infinite)
    +-- yoyo: bool
    |
    +-- callbacks
            +-- on_start: Callable
            +-- on_update: Callable[[T], None]
            +-- on_complete: Callable

TweenSequence
    +-- tweens: List[Tween] (sequential execution)

TweenGroup
    +-- tweens: List[Tween] (parallel execution)

TweenManager (singleton)
    +-- active_tweens: Set[Tween]
    +-- update(dt) → batch update all
```

**Type Support**:
- Numeric (int, float): Linear interpolation
- Tuple: Element-wise interpolation
- Dict: Key-wise interpolation (recursive)

---

### Keyframe Animation (keyframe.py)

```
KeyframeAnimation
    |
    +-- KeyframeTrack[]
    |       +-- property: str
    |       +-- keyframes: List[Keyframe]
    |
    +-- Keyframe
    |       +-- time: float (normalized 0-1 or seconds)
    |       +-- value: Any
    |       +-- easing: EasingFunction
    |
    +-- loop_mode: LoopMode
            +-- ONCE: Play once and stop
            +-- LOOP: Repeat from start
            +-- PING_PONG: Forward then reverse

KeyframeAnimationManager (singleton)
    +-- active_animations: Set[KeyframeAnimation]
```

**Sampling Algorithm**:
1. Find keyframes surrounding current time
2. Calculate normalized time between them
3. Apply easing function
4. Interpolate values

---

### Triggers (triggers.py)

```
TriggerBase (abstract)
    |
    +-- StateTrigger
    |       +-- widget: Widget
    |       +-- state: WidgetState (HOVER, PRESSED, FOCUSED, etc.)
    |
    +-- EventTrigger
    |       +-- event_type: str
    |       +-- source: EventEmitter
    |
    +-- PropertyTrigger
    |       +-- source: Observable
    |       +-- property: str
    |       +-- condition: Callable[[Any], bool]
    |
    +-- DataTrigger
    |       +-- binding: Binding
    |       +-- value: Any (trigger when equals)
    |
    +-- MultiTrigger
            +-- triggers: List[TriggerBase]
            +-- operator: LogicOp (AND, OR, XOR, NAND, NOR)
```

**Logic Operators**:
- AND: All triggers must be true
- OR: At least one trigger must be true
- XOR: Exactly one trigger must be true
- NAND: Not all triggers are true
- NOR: No triggers are true

---

### Easing Functions (easing.py)

```
EasingFunction = Callable[[float], float]

Standard Easings (30+):
    Linear
    
    Quad In/Out/InOut
    Cubic In/Out/InOut
    Quart In/Out/InOut
    Quint In/Out/InOut
    
    Sine In/Out/InOut
    Expo In/Out/InOut
    Circ In/Out/InOut
    
    Elastic In/Out/InOut
    Back In/Out/InOut
    Bounce In/Out/InOut

CubicBezier
    +-- control points (x1, y1, x2, y2)
    +-- Newton-Raphson iteration for x → y mapping
```

**Newton-Raphson Details**:
- Max 8 iterations
- Tolerance 1e-6
- Handles edge cases (dx near zero)

---

## Data Flow

```
Game Loop
    |
    v
TweenManager.update(dt)
    |
    +-- for each active Tween:
    |       +-- advance time
    |       +-- calculate progress (with repeat/yoyo)
    |       +-- apply easing
    |       +-- interpolate value
    |       +-- call on_update callback
    |       +-- check completion
    |
    v
KeyframeAnimationManager.update(dt)
    |
    +-- for each active KeyframeAnimation:
    |       +-- advance time
    |       +-- handle loop mode
    |       +-- for each KeyframeTrack:
    |               +-- sample value at time
    |               +-- apply to target property
    |
    v
Animator.update(dt) (for each instance)
    |
    +-- evaluate transition conditions
    +-- advance current state
    +-- blend layers
    +-- output final values
```

---

## Integration Points

| From | To | Purpose |
|------|----|---------| 
| MotionManager | TweenManager | Duration multipliers |
| MotionManager | Animator | should_animate() check |
| Trigger | Animation | Start/stop animations |
| Binding | DataTrigger | Property value changes |
| Widget | StateTrigger | State changes (hover, etc.) |

---

## Design Decisions

### D1: Generic Tween[T]

**Decision**: Tween is generic over the value type.

**Rationale**: Supports numeric, tuple, dict without type-specific subclasses. Interpolation logic handles type detection at runtime.

**Trade-off**: Slight runtime overhead vs cleaner API.

### D2: State Machine Over Timeline

**Decision**: Animator uses state machine, not single timeline.

**Rationale**: Game animations often need branching (idle → walk → run, can interrupt). State machines model this naturally.

### D3: Layer Priority System

**Decision**: Layers have explicit priority and blend mode.

**Rationale**: Complex animations need additive layers (breathing + walking) and override layers (hit reaction overrides idle).

### D4: Newton-Raphson for Bezier

**Decision**: Use iterative solver, not lookup table.

**Rationale**: Accuracy at all curve shapes. Lookup tables need interpolation anyway and use more memory.

**Guard**: 8 iteration limit prevents infinite loops.

### D5: Trigger Composition

**Decision**: MultiTrigger with logic operators.

**Rationale**: Complex conditions without custom trigger classes. "Hover AND NOT disabled" expressed declaratively.

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Many active tweens | Frame time spike | Batch updates, object pooling |
| Newton-Raphson divergence | Wrong easing values | Iteration limit, fallback to linear |
| Layer blending artifacts | Visual glitches | Test edge cases, clamp values |
| Trigger evaluation cost | Per-frame overhead | Cache trigger results, dirty flag |
