# PHASE 3 TODO: Input Dispatch System

## Prerequisites
- Phase 2 (Focus Management) complete
- Widgets have `contains_point(x, y)` method
- Widget tree traversal available

---

## Task 3.1: Input Event Classes

**File**: `engine/ui/input/events.py`

**Description**: Create event classes for input dispatch.

**Acceptance Criteria**:
- [ ] `Modifiers` dataclass: shift, ctrl, alt, meta booleans
- [ ] `MouseEvent` class: x, y, button, modifiers, stop_propagation(), prevent_default()
- [ ] `KeyEvent` class: key, modifiers, stop_propagation(), prevent_default()
- [ ] `DragEvent` class: payload, source_widget, x, y
- [ ] Events are immutable except for propagation flags

**Evidence of Completion**: `event.stop_propagation()` prevents further handlers.

---

## Task 3.2: Hit Testing

**File**: `engine/ui/input/hit_test.py`

**Description**: Implement hit testing for mouse events.

**Acceptance Criteria**:
- [ ] `hit_test(root, x, y) -> Widget | None` traverses tree back-to-front
- [ ] Returns topmost widget where `contains_point(x, y)` is True
- [ ] Respects widget visibility (invisible widgets not hit)
- [ ] Respects widget enabled state (disabled widgets may or may not be hit - configurable)
- [ ] Handles overlapping widgets correctly

**Evidence of Completion**: Click at (100, 100) returns correct topmost widget.

---

## Task 3.3: Input Router Core

**File**: `engine/ui/input/router.py`

**Description**: Create central input router.

**Acceptance Criteria**:
- [ ] `InputRouter` class with root widget reference
- [ ] `set_root(widget)` sets the widget tree root
- [ ] `set_focus_coordinator(coordinator)` integrates with Phase 2
- [ ] Maintains `_hover_widget` for enter/leave events
- [ ] Single router instance per application

**Evidence of Completion**: Router created with root widget and focus coordinator.

---

## Task 3.4: Mouse Event Dispatch

**File**: `engine/ui/input/router.py`

**Description**: Implement mouse event dispatch.

**Acceptance Criteria**:
- [ ] `dispatch_mouse_down(event)` hit tests and dispatches
- [ ] `dispatch_mouse_up(event)` hit tests and dispatches
- [ ] `dispatch_mouse_move(event)` tracks hover, dispatches enter/leave
- [ ] Events propagate through capture and bubble phases
- [ ] `stop_propagation()` halts propagation

**Evidence of Completion**: Mouse down on button triggers `button.handle_mouse_down()`.

---

## Task 3.5: Event Propagation Path

**File**: `engine/ui/input/router.py`

**Description**: Compute propagation path for events.

**Acceptance Criteria**:
- [ ] `_get_propagation_path(target) -> list[Widget]` returns root-to-target path
- [ ] Capture phase iterates path forward (root to target)
- [ ] Bubble phase iterates path backward (target to root)
- [ ] Widgets can stop propagation at any point

**Evidence of Completion**: Click on nested widget fires handlers on all ancestors.

---

## Task 3.6: Hover Tracking

**File**: `engine/ui/input/router.py`

**Description**: Track hover state and dispatch enter/leave.

**Acceptance Criteria**:
- [ ] `_hover_widget` tracks widget under cursor
- [ ] `dispatch_mouse_move()` detects hover changes
- [ ] `handle_mouse_enter()` called on new hover widget
- [ ] `handle_mouse_leave()` called on previous hover widget
- [ ] Nested enter/leave handled correctly

**Evidence of Completion**: Moving cursor from button A to button B triggers leave on A, enter on B.

---

## Task 3.7: Keyboard Event Dispatch

**File**: `engine/ui/input/router.py`

**Description**: Implement keyboard event dispatch.

**Acceptance Criteria**:
- [ ] `dispatch_key_down(event)` routes to focused widget
- [ ] `dispatch_key_up(event)` routes to focused widget
- [ ] Tab and Shift+Tab intercepted for focus navigation
- [ ] Escape key triggers focus trap pop if active
- [ ] No keyboard events if no focused widget

**Evidence of Completion**: Key press on focused text input triggers handler.

---

## Task 3.8: Drag State Management

**File**: `engine/ui/input/drag.py`

**Description**: Implement drag state tracking.

**Acceptance Criteria**:
- [ ] `DragState` class: source_widget, payload, start_x, start_y
- [ ] `start_drag(widget, payload, x, y)` initiates drag
- [ ] `get_drag_state() -> DragState | None` returns current drag
- [ ] `end_drag()` clears drag state
- [ ] Drag threshold before drag starts (prevent accidental drags)

**Evidence of Completion**: Start drag on inventory slot creates drag state with item payload.

---

## Task 3.9: Drag Event Dispatch

**File**: `engine/ui/input/router.py`

**Description**: Integrate drag events with router.

**Acceptance Criteria**:
- [ ] `dispatch_mouse_move()` calls `handle_drag_over()` on hover widget during drag
- [ ] `dispatch_mouse_up()` calls `handle_drop()` on hover widget during drag
- [ ] Drop cancelled if released outside valid target
- [ ] Source widget notified of drag cancel
- [ ] Drag visual feedback coordinated

**Evidence of Completion**: Drag item from slot A to slot B triggers drop handler on B.

---

## Task 3.10: Widget Drag Handlers

**File**: `engine/ui/widgets/base.py`

**Description**: Add drag handler methods to widget base.

**Acceptance Criteria**:
- [ ] `handle_drag_over(drag_state, event) -> bool` for drag hover
- [ ] `handle_drop(drag_state, event) -> bool` for drop acceptance
- [ ] `handle_drag_cancel()` for drag cancel notification
- [ ] Default implementations return False (no drag support)
- [ ] InventorySlot overrides with item drag logic

**Evidence of Completion**: InventorySlot accepts drop of compatible item type.

---

## Task 3.11: Input Module Exports

**File**: `engine/ui/input/__init__.py`

**Description**: Export input classes for public API.

**Acceptance Criteria**:
- [ ] Exports: InputRouter, MouseEvent, KeyEvent, DragEvent, Modifiers
- [ ] Exports: hit_test (if needed externally)
- [ ] No internal implementation details exposed

**Evidence of Completion**: `from engine.ui.input import InputRouter` works.

---

## Summary

| Task | Effort | Priority |
|------|--------|----------|
| 3.1 Event Classes | Medium | P0 |
| 3.2 Hit Testing | Medium | P0 |
| 3.3 Router Core | Medium | P0 |
| 3.4 Mouse Dispatch | Large | P0 |
| 3.5 Propagation Path | Medium | P0 |
| 3.6 Hover Tracking | Medium | P1 |
| 3.7 Keyboard Dispatch | Medium | P0 |
| 3.8 Drag State | Medium | P1 |
| 3.9 Drag Dispatch | Medium | P1 |
| 3.10 Widget Drag Handlers | Small | P1 |
| 3.11 Module Exports | Small | P0 |

**Total Tasks**: 11
**Critical Path**: 3.1 -> 3.2 -> 3.3 -> 3.4 -> 3.5 -> 3.7 -> 3.11
