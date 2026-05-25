# PHASE 4 TODO: Framework Module

## Summary

Verify and test the 5 framework files (~3,924 lines) for correct widget hierarchy, coordinate transforms, focus management, layouts, and W3C event dispatch.

---

## T1: Widget Hierarchy

**File**: `engine/ui/framework/widget.py`

### T1.1: Test Parent-Child Relationships
- [ ] add_child() sets child.parent
- [ ] remove_child() clears child.parent
- [ ] Child appears in parent.children
- [ ] Removed child not in parent.children

**Acceptance**: Parent-child relationships consistent.

### T1.2: Test Depth-First Traversal
- [ ] Root visited first
- [ ] Children visited before siblings
- [ ] Order: A → A1 → A1a → A1b → A2 → B

**Acceptance**: Depth-first order correct.

### T1.3: Test Breadth-First Traversal
- [ ] Root visited first
- [ ] All siblings visited before children
- [ ] Order: A, B → A1, A2, B1 → A1a, A1b

**Acceptance**: Breadth-first order correct.

### T1.4: Test Lifecycle Hooks
- [ ] on_mount() called when added to tree
- [ ] on_unmount() called when removed
- [ ] on_update(dt) called each frame
- [ ] on_render() called when visible

**Acceptance**: Lifecycle hooks fire correctly.

### T1.5: Test Dirty Tracking
- [ ] Property change marks widget dirty
- [ ] Parent notified of dirty child
- [ ] Render clears dirty flag

**Acceptance**: Dirty tracking works.

---

## T2: Coordinate System

**File**: `engine/ui/framework/coordinate.py`

### T2.1: Test Point Operations
- [ ] Point(1,2) + Point(3,4) = Point(4,6)
- [ ] Point(4,3) - Point(1,1) = Point(3,2)
- [ ] Point(2,3).scale(2) = Point(4,6)
- [ ] Point(3,4).length() = 5.0
- [ ] Point(3,4).normalize() = Point(0.6, 0.8)

**Acceptance**: Point math is correct.

### T2.2: Test Rect Operations
- [ ] Rect contains point inside
- [ ] Rect does not contain point outside
- [ ] Two rects intersect correctly
- [ ] Union of rects is bounding box
- [ ] Intersection of rects is overlap

**Acceptance**: Rect operations correct.

### T2.3: Test Transform Identity
- [ ] Identity transform returns same point
- [ ] Identity * Identity = Identity

**Acceptance**: Identity transform works.

### T2.4: Test Transform Translate
- [ ] translate(10, 20) moves point by (10, 20)
- [ ] translate(a) then translate(b) = translate(a+b)

**Acceptance**: Translation works.

### T2.5: Test Transform Rotate
- [ ] rotate(90°) rotates point 90° counterclockwise
- [ ] rotate(180°) = scale(-1, -1)
- [ ] rotate(360°) = identity (within tolerance)

**Acceptance**: Rotation works.

### T2.6: Test Transform Scale
- [ ] scale(2, 2) doubles coordinates
- [ ] scale(1, -1) flips vertically
- [ ] scale(0.5, 0.5) halves coordinates

**Acceptance**: Scaling works.

### T2.7: Test Transform Composition
- [ ] A.compose(B) applies B then A
- [ ] translate.compose(rotate) ≠ rotate.compose(translate)
- [ ] Matrix multiplication is correct

**Acceptance**: Composition correct.

### T2.8: Test Transform Invert
- [ ] A.compose(A.invert()) = identity
- [ ] Invert of translate(-dx, -dy)
- [ ] Invert of scale(1/sx, 1/sy)

**Acceptance**: Inversion correct.

### T2.9: Test Coordinate Conversion
- [ ] local_to_global() applies ancestor transforms
- [ ] global_to_local() reverses transforms
- [ ] Round-trip: global_to_local(local_to_global(p)) = p

**Acceptance**: Coordinate conversion works.

---

## T3: Focus Management

**File**: `engine/ui/framework/focus.py`

### T3.1: Test Set Focus
- [ ] set_focus(widget) sets _focused
- [ ] Previous widget receives focus_out event
- [ ] New widget receives focus_in event

**Acceptance**: Focus change works.

### T3.2: Test Clear Focus
- [ ] clear_focus() sets _focused to None
- [ ] Previous widget receives focus_out event

**Acceptance**: Focus clearing works.

### T3.3: Test Focus Next (Tab)
- [ ] Moves to next focusable in order
- [ ] Skips non-focusable widgets
- [ ] Wraps from last to first

**Acceptance**: Tab navigation works.

### T3.4: Test Focus Previous (Shift+Tab)
- [ ] Moves to previous focusable in order
- [ ] Skips non-focusable widgets
- [ ] Wraps from first to last

**Acceptance**: Shift+Tab navigation works.

### T3.5: Test Focus History
- [ ] Focus changes recorded in history
- [ ] restore_focus() returns to previous
- [ ] History limited in size

**Acceptance**: Focus history works.

### T3.6: Test Focus Group
- [ ] Navigation confined to group
- [ ] Direction respected (H/V/both)
- [ ] Wrap setting honored

**Acceptance**: Focus groups work.

### T3.7: Test Focus Trap
- [ ] trap_focus(container) limits Tab to container
- [ ] First focusable in container gets focus
- [ ] Tab wraps within container
- [ ] release_trap() restores previous focus

**Acceptance**: Focus trapping works.

### T3.8: Test Nested Focus Traps
- [ ] Traps stack correctly
- [ ] Inner trap takes precedence
- [ ] Release inner, outer becomes active

**Acceptance**: Nested traps work.

---

## T4: Container Layouts

**File**: `engine/ui/framework/container.py`

### T4.1: Test HBox Basic Layout
- [ ] Children laid out horizontally
- [ ] Spacing between children
- [ ] Padding at edges

**Acceptance**: Basic HBox works.

### T4.2: Test HBox Main Axis Alignment
- [ ] START: children at left
- [ ] CENTER: children centered
- [ ] END: children at right
- [ ] SPACE_BETWEEN: equal space between
- [ ] SPACE_AROUND: equal space around

**Acceptance**: Main axis alignment works.

### T4.3: Test HBox Cross Axis Alignment
- [ ] START: children at top
- [ ] CENTER: children centered vertically
- [ ] END: children at bottom
- [ ] STRETCH: children fill height

**Acceptance**: Cross axis alignment works.

### T4.4: Test VBox Layout
- [ ] Same as HBox but vertical
- [ ] Main axis is vertical
- [ ] Cross axis is horizontal

**Acceptance**: VBox works (symmetric to HBox).

### T4.5: Test Stack Layout
- [ ] Children overlap at same position
- [ ] All children at (padding.left, padding.top)
- [ ] Last child on top

**Acceptance**: Stack layout works.

### T4.6: Test ScrollContainer
- [ ] Content can exceed viewport
- [ ] scroll_x, scroll_y offset content
- [ ] Content clipped to viewport
- [ ] Scroll bounds enforced

**Acceptance**: Scroll container works.

### T4.7: Test Flexible Sizing
- [ ] flex: 1 children share space equally
- [ ] flex: 2 gets twice the space of flex: 1
- [ ] flex: 0 gets natural size

**Acceptance**: Flexible sizing works.

---

## T5: Event System

**File**: `engine/ui/framework/events.py`

### T5.1: Test Event Creation
- [ ] MouseEvent has x, y, button
- [ ] KeyboardEvent has key, code, modifiers
- [ ] FocusEvent has related_target
- [ ] All have type, target, timestamp

**Acceptance**: Event properties correct.

### T5.2: Test Capture Phase
- [ ] Handlers with capture=True called
- [ ] Called from root toward target
- [ ] Target not included in capture phase
- [ ] stop_propagation() stops traversal

**Acceptance**: Capture phase correct.

### T5.3: Test Target Phase
- [ ] Both capture and bubble handlers called
- [ ] stop_immediate_propagation() stops at current handler
- [ ] phase == TARGET

**Acceptance**: Target phase correct.

### T5.4: Test Bubble Phase
- [ ] Handlers with capture=False called
- [ ] Called from target toward root
- [ ] Target not included in bubble phase
- [ ] Only if event.bubbles is True

**Acceptance**: Bubble phase correct.

### T5.5: Test stop_propagation()
- [ ] Stops dispatch to next widget
- [ ] Current widget's handlers still run
- [ ] Works in all phases

**Acceptance**: stop_propagation() works.

### T5.6: Test stop_immediate_propagation()
- [ ] Stops dispatch immediately
- [ ] Current handler's remaining code runs
- [ ] Next handler on same widget skipped

**Acceptance**: stop_immediate_propagation() works.

### T5.7: Test prevent_default()
- [ ] Sets is_default_prevented = True
- [ ] dispatch() returns False
- [ ] Default action should not occur

**Acceptance**: prevent_default() works.

### T5.8: Test Non-Bubbling Events
- [ ] Focus events don't bubble
- [ ] Only capture and target phases
- [ ] event.bubbles = False

**Acceptance**: Non-bubbling events work.

---

## T6: Hit Testing

**File**: `engine/ui/framework/widget.py`

### T6.1: Test Point Inside Widget
- [ ] Point inside bounds returns widget
- [ ] Considers widget position and size
- [ ] Respects is_interactive flag

**Acceptance**: Basic hit test works.

### T6.2: Test Point Outside Widget
- [ ] Point outside bounds returns None
- [ ] Edge cases handled correctly

**Acceptance**: Miss detection works.

### T6.3: Test Child Hit Testing
- [ ] Children checked in reverse order
- [ ] First match returned (topmost)
- [ ] Point converted to local coordinates

**Acceptance**: Child hit testing works.

### T6.4: Test Overlapping Children
- [ ] Later child (higher z-order) wins
- [ ] Even if earlier child is larger

**Acceptance**: Z-order respected.

### T6.5: Test Non-Interactive Widgets
- [ ] is_interactive=False skips widget
- [ ] Children still checked
- [ ] Parent may receive event

**Acceptance**: Non-interactive handling works.

### T6.6: Test Transformed Widgets
- [ ] Hit test respects transform
- [ ] Rotated widget hit correctly
- [ ] Scaled widget hit correctly

**Acceptance**: Transform-aware hit testing.

---

## T7: Integration Tests

### T7.1: Mouse Click Flow
- [ ] Raw input → MouseEvent
- [ ] Hit test finds target
- [ ] Capture → Target → Bubble
- [ ] Handler updates widget state

**Acceptance**: Full click flow works.

### T7.2: Keyboard Focus Flow
- [ ] Tab key → FocusManager.focus_next()
- [ ] Focus event dispatched
- [ ] Keyboard events route to focused widget

**Acceptance**: Full keyboard flow works.

### T7.3: Layout After Property Change
- [ ] Property change marks dirty
- [ ] Layout pass triggered
- [ ] Children repositioned correctly

**Acceptance**: Reactive layout works.

### T7.4: Modal Dialog Focus Trap
- [ ] Open modal → trap_focus()
- [ ] Tab confined to modal
- [ ] Close modal → release_trap()
- [ ] Focus restored to previous

**Acceptance**: Modal focus flow works.

---

## Completion Criteria

All tasks T1-T7 marked complete with tests passing.
