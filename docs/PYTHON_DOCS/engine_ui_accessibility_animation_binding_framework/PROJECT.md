# PROJECT: engine/ui — Accessibility, Animation, Binding, Framework

## Scope

Integration and verification of 19 Python files (~15,949 lines) across four UI subsystems:

| Module | Files | Lines | Responsibility |
|--------|-------|-------|----------------|
| accessibility | 5 | ~3,449 | WCAG compliance, screen reader, keyboard nav, scaling, motion |
| animation | 5 | ~4,233 | State machines, tweening, keyframes, triggers, easing |
| binding | 4 | ~3,526 | MVVM data binding, validation, converters, observables |
| framework | 5 | ~3,924 | Widget base, coordinates, focus, containers, events |

All files classified as REAL with production-quality implementations.

---

## Goals

1. **Verify WCAG 2.1 AA Compliance** — Validate contrast ratios, touch targets, reduced motion, screen reader integration
2. **Validate Animation Architecture** — Confirm state machine, tween, keyframe, and trigger systems integrate correctly
3. **Ensure MVVM Binding Correctness** — Test property paths, two-way binding, validation, converters
4. **Confirm Framework Foundation** — Verify widget hierarchy, event dispatch (W3C model), focus management

---

## Constraints

- Python 3.13 target (not 3.14)
- No external UI framework dependencies (self-contained system)
- WCAG 2.1 AA minimum, AAA where possible
- Performance: virtualized lists must handle 10K+ items
- Animation: 60 FPS minimum on supported hardware

---

## Acceptance Criteria

### Phase 1: Accessibility
- [ ] WCAG 2.1 contrast ratio calculation matches spec exactly
- [ ] Brettel colorblind simulation produces correct transformations
- [ ] Touch targets meet 44x44 CSS pixel minimum (WCAG 2.5.5)
- [ ] Screen reader announcements queue correctly (polite vs assertive)
- [ ] Reduced motion preference respects system settings

### Phase 2: Animation
- [ ] State machine transitions respect conditions and callbacks
- [ ] Layer blending modes (override, additive, multiply) work correctly
- [ ] Tween interpolation handles numeric, tuple, dict types
- [ ] Keyframe animation supports all loop modes (ONCE, LOOP, PING_PONG)
- [ ] All 30+ easing functions match mathematical specifications
- [ ] CubicBezier Newton-Raphson converges within 8 iterations

### Phase 3: Binding
- [ ] Property paths navigate nested objects with indexers (`items[0].name`)
- [ ] TWO_WAY binding propagates changes bidirectionally
- [ ] Async validation supports cancellation
- [ ] Chained converters compose correctly (convert and convert_back)
- [ ] ObservableList virtualization recycles widgets correctly

### Phase 4: Framework
- [ ] Widget hit testing traverses children in correct order (reverse z-order)
- [ ] W3C event dispatch follows capture → target → bubble phases
- [ ] Focus trapping contains tab navigation within modal boundaries
- [ ] Transform2D composition produces correct affine matrices
- [ ] Container layouts (HBox, VBox, Stack) respect alignment and spacing

---

## Dependencies

- No external dependencies beyond Python stdlib
- Internal dependencies: framework → binding → animation → accessibility

---

## Risks

| Risk | Mitigation |
|------|------------|
| Colorblind matrices inaccurate | Validate against reference implementations |
| Newton-Raphson divergence | Ensure bounds checking and iteration limits |
| Event dispatch performance | Profile with deep widget hierarchies |
| Virtualization edge cases | Test scroll boundaries and rapid scrolling |
