# CLARIFICATION: UI Subsystem Design Philosophy

## Architectural Rationale

### Why Self-Contained UI Framework?

The engine implements its own UI framework rather than wrapping an existing library (Qt, wxWidgets, GTK) for several reasons:

1. **Game Engine Integration** — UI must integrate with the render pipeline, not fight it
2. **Predictable Performance** — No hidden allocations or GC pauses from framework internals
3. **Accessibility First** — WCAG compliance built-in, not bolted-on
4. **Python/Rust Hybrid** — Clean FFI boundary requires controlled types

---

## Module Relationships

```
framework (foundation)
    |
    +-- binding (MVVM data flow)
    |       |
    |       +-- animation (property animation)
    |               |
    |               +-- accessibility (user preferences)
```

Each layer builds on the one below. Framework provides the widget tree and event system. Binding connects data models to widgets. Animation drives visual changes. Accessibility modifies behavior based on user needs.

---

## Key Design Decisions

### 1. W3C Event Model

**Decision**: Implement full W3C event dispatch (capture, target, bubble phases).

**Rationale**: Industry standard, predictable behavior, familiar to web developers. Enables event delegation and complex interaction patterns.

**Alternative Rejected**: Simple callback registration would be simpler but lacks the flexibility for complex UIs.

### 2. WCAG 2.1 Contrast Algorithm

**Decision**: Use exact sRGB linearization formula from WCAG specification.

**Rationale**: Legal compliance and actual accessibility. Approximate formulas produce incorrect results at edge cases.

**Critical Implementation Detail**:
```python
# The 0.03928 threshold and 12.92 divisor are NOT arbitrary
# They come from the sRGB specification for gamma correction
if value <= 0.03928:
    return value / 12.92
return ((value + 0.055) / 1.055) ** 2.4
```

### 3. Brettel Colorblind Simulation

**Decision**: Use Brettel, Vienot, Mollon (1997) algorithm with LMS color space transformation.

**Rationale**: More accurate than older Viénot (1999) simplified matrices. Handles partial colorblindness (anomalous trichromacy) correctly.

**Note**: The transformation matrices are empirically derived from human vision research and must not be modified.

### 4. Newton-Raphson for Bezier Curves

**Decision**: Use Newton-Raphson iteration to solve x → y mapping for cubic bezier easing.

**Rationale**: Closed-form solution doesn't exist. Newton-Raphson converges quickly (typically 3-4 iterations) for well-behaved curves.

**Guard Rails**: 8 iteration limit and 1e-6 tolerance prevent infinite loops on degenerate curves.

### 5. State Machine Animation

**Decision**: Layer-based state machine similar to Unity's Animator Controller.

**Rationale**: Familiar to game developers, handles complex character animation, supports blending between states.

**Layer Blend Modes**:
- OVERRIDE: Replace lower layer values entirely
- ADDITIVE: Add to lower layer values (good for procedural animation)
- MULTIPLY: Scale lower layer values (good for damage effects)
- AVERAGE: Weighted average (good for smooth transitions)

### 6. MVVM Binding with Property Paths

**Decision**: Support dot-notation paths with indexer access (`user.addresses[0].city`).

**Rationale**: Enables binding to complex nested data structures without flattening the model.

**Implementation Challenge**: Weak references prevent memory leaks but require careful lifetime management.

### 7. Virtualized Lists

**Decision**: Widget recycling with visible-range calculation.

**Rationale**: Rendering 10K items directly is impossible. Virtualization only creates widgets for visible items plus small buffer.

**Critical Formula**:
```python
first_visible = scroll_offset // item_height
last_visible = first_visible + (viewport_height // item_height) + 2
# The +2 provides buffer for smooth scrolling
```

---

## Accessibility Philosophy

### Not Optional

Accessibility is not a feature flag or premium tier. Every widget must:

1. Announce its role and state to screen readers
2. Be keyboard navigable
3. Meet WCAG 2.5.5 touch target minimums
4. Respect reduced motion preferences
5. Work in high contrast modes

### Reduced Motion Hierarchy

```
NO_PREFERENCE → All animations run normally
REDUCE        → Essential animations only (feedback, state changes)
               Decorative animations disabled (parallax, auto-play)
NONE          → No animations at all (instant transitions)
```

Essential animations provide necessary feedback. Decorative animations are visual enhancement only.

---

## Performance Considerations

### Event Dispatch

The W3C model requires walking the widget tree twice (capture down, bubble up). For deep hierarchies (100+ levels), this could become a bottleneck.

**Mitigation**: Most events don't need capture phase. The `is_stopped` check short-circuits traversal when possible.

### Binding Updates

Property path navigation involves reflection (getattr, indexing). Frequent updates to deeply nested properties could be slow.

**Mitigation**: Cache the resolved path segments. Only re-resolve when the source object changes.

### Animation Updates

All active tweens and keyframe animations update every frame.

**Mitigation**: TweenManager batches updates. Completed animations are removed from the active list immediately.

---

## Integration Points

### With Render Pipeline

- `Widget.render()` produces draw commands, not pixels
- Draw commands go to the frame graph for batching
- GPU-accelerated where possible (transforms, clipping)

### With Input System

- Raw input events (from engine/input) are converted to UIEvents
- Event dispatch routes to correct widget
- Focus system determines keyboard target

### With Audio System

- Screen reader announcements may trigger TTS
- UI sounds (click, hover) integrate with audio mixer

---

## Testing Strategy

### Unit Tests

- Each easing function against known values
- Contrast ratio calculation against WCAG examples
- Transform composition against matrix multiplication

### Integration Tests

- Event dispatch with complex widget hierarchies
- Binding with nested observables
- Animation state machine transitions

### Visual Tests

- Colorblind simulation against reference images
- High contrast theme rendering
- Layout container alignment
