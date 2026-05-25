# SUMMARY: engine/ui/accessibility, animation, binding, framework

## Metrics

| Metric | Value |
|--------|-------|
| Total Lines | 15,949 |
| Classification | REAL |
| Files | 23 (19 implementation + 4 __init__.py) |
| Modules | 4 |

### Per-Module Breakdown

| Module | Files | Lines | Status |
|--------|-------|-------|--------|
| accessibility | 6 | 3,543 | REAL |
| animation | 6 | 4,507 | REAL |
| binding | 5 | 3,793 | REAL |
| framework | 6 | 4,106 | REAL |

## Algorithm Inventory

| Algorithm | File | Lines | Status |
|-----------|------|-------|--------|
| WCAG Contrast Ratio (sRGB) | high_contrast.py | 131-148 | REAL |
| Brettel Colorblind Matrices | high_contrast.py | 68-92, 501-535 | REAL |
| Tab Order Sorting | keyboard_nav.py | 261-275 | REAL |
| Spatial Navigation (nearest widget) | keyboard_nav.py | 778-829 | REAL |
| Grid Navigation | keyboard_nav.py | 713-758 | REAL |
| Linear Easing | easing.py | 74-76 | REAL |
| Quad In/Out/InOut | easing.py | 84-98 | REAL |
| Cubic In/Out/InOut | easing.py | 106-122 | REAL |
| Quart In/Out/InOut | easing.py | 130-146 | REAL |
| Quint In/Out/InOut | easing.py | 154-170 | REAL |
| Sine In/Out/InOut | easing.py | 178-190 | REAL |
| Expo In/Out/InOut | easing.py | 198-220 | REAL |
| Circ In/Out/InOut | easing.py | 228-244 | REAL |
| Elastic In/Out/InOut | easing.py | 256-286 | REAL |
| Back In/Out/InOut | easing.py | 297-315 | REAL |
| Bounce Out/In/InOut | easing.py | 336-360 | REAL |
| CubicBezier Newton-Raphson | easing.py | 369-433 | REAL |
| Tween Value Interpolation | tween.py | 394-420 | REAL |
| Tween Sequence Update | tween.py | 549-582 | REAL |
| Tween Group Update | tween.py | 668-699 | REAL |
| Keyframe Sample | keyframe.py | ~273-282 | REAL |
| Multi-Trigger Logic | triggers.py | ~306-314 | REAL |
| State Machine Layer Blending | animator.py | ~209-219 | REAL |
| Property Path Parsing | binding.py | 100-137 | REAL |
| Property Path Get/Set | binding.py | 162-222 | REAL |
| Two-Way Binding Update | binding.py | 651-705 | REAL |
| Multi-Binding Combine | binding.py | 969-1014 | REAL |
| Async Validation | validation.py | ~405-408 | REAL |
| Chained Converter | converter.py | ~435-446 | REAL |
| Observable List Virtualization | observable.py | ~470-482 | REAL |
| Widget Hit Test | widget.py | 738-766 | REAL |
| Widget Hierarchy Traversal | widget.py | 646-654 | REAL |
| Dirty Tracking | widget.py | 660-686 | REAL |
| W3C Event Dispatch | events.py | 589-643 | REAL |
| Transform Composition | coordinate.py | ~543-551 | REAL |
| Focus Trap Management | focus.py | ~573-582 | REAL |
| Container Layout | container.py | ~607-614 | REAL |

## File-Level Summary

| File | Lines | Purpose |
|------|-------|---------|
| accessibility/high_contrast.py | 608 | WCAG contrast, colorblind simulation |
| accessibility/keyboard_nav.py | 839 | Tab order, spatial nav, shortcuts |
| accessibility/motion.py | 664 | Reduced motion preferences |
| accessibility/scale.py | 696 | DPI awareness, touch targets |
| accessibility/screen_reader.py | 642 | ARIA roles, live regions |
| animation/animator.py | 928 | State machine, layers, transitions |
| animation/easing.py | 631 | 30+ easing functions, CubicBezier |
| animation/keyframe.py | 890 | Keyframe tracks, loop modes |
| animation/triggers.py | 885 | State/event/property/data triggers |
| animation/tween.py | 899 | Property tweening, sequences, groups |
| binding/binding.py | 1281 | MVVM binding core, PropertyPath |
| binding/converter.py | 752 | Value converters, chained, cached |
| binding/observable.py | 643 | Observable collections, virtualization |
| binding/validation.py | 850 | Validators, async, composite |
| framework/container.py | 708 | HBox, VBox, Stack, ScrollContainer |
| framework/coordinate.py | 770 | Point, Size, Rect, Transform2D |
| framework/events.py | 662 | UIEvent, MouseEvent, KeyboardEvent |
| framework/focus.py | 753 | FocusManager, focus traps |
| framework/widget.py | 1031 | Widget base class, hierarchy, lifecycle |
