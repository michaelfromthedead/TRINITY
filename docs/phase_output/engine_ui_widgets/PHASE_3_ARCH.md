# PHASE 3 ARCHITECTURE: Input Dispatch System

## Problem Statement

Currently, the application must manually iterate widgets and call their input handlers. There is no central input router, no event propagation (capture/bubble), and no coordination for cross-widget interactions like drag-and-drop.

Current pattern:
```python
for widget in widgets:
    if widget.handle_mouse_down(x, y):
        break
```

## Architectural Decision

### Input Router

Central router receives all input events and dispatches to appropriate widgets:

```python
class InputRouter:
    _root: Widget
    _focus_coordinator: FocusCoordinator
    _dragging: DragState | None
    _hover_widget: Widget | None
```

### Event Flow

```
Platform Input
     |
     v
InputRouter
     |
     +-- Mouse events --> hit test --> target widget
     |
     +-- Keyboard events --> focused widget
     |
     +-- Drag events --> drag coordinator
```

### Hit Testing

Mouse events require determining which widget is under the cursor:
```python
def hit_test(self, x: float, y: float) -> Widget | None:
    # Traverse widget tree back-to-front (render order)
    # Return topmost widget that contains_point(x, y)
```

### Event Propagation

Events propagate in two phases:
1. **Capture**: Root to target (top-down)
2. **Bubble**: Target to root (bottom-up)

```python
class InputEvent:
    x: float
    y: float
    modifiers: Modifiers
    _stop_propagation: bool = False
    _prevent_default: bool = False

def dispatch_mouse_down(self, event: InputEvent) -> None:
    target = self.hit_test(event.x, event.y)
    path = self._get_propagation_path(target)
    
    # Capture phase
    for widget in path:
        if event._stop_propagation:
            break
        widget.handle_mouse_down_capture(event)
    
    # Bubble phase
    for widget in reversed(path):
        if event._stop_propagation:
            break
        widget.handle_mouse_down(event)
```

### Keyboard Routing

Keyboard events route to focused widget:
```python
def dispatch_key_down(self, event: KeyEvent) -> None:
    # Tab/Shift+Tab handled by focus coordinator
    if event.key == Key.TAB:
        if event.shift:
            self._focus_coordinator.focus_previous()
        else:
            self._focus_coordinator.focus_next()
        return
    
    # Other keys go to focused widget
    focused = self._focus_coordinator.get_focused()
    if focused:
        focused.handle_key_down(event)
```

### Drag Coordination

Drag operations that span widgets:
```python
class DragState:
    source_widget: Widget
    payload: Any
    start_x: float
    start_y: float

def start_drag(self, widget: Widget, payload: Any, x: float, y: float) -> None:
    self._dragging = DragState(widget, payload, x, y)

def dispatch_mouse_move(self, event: InputEvent) -> None:
    if self._dragging:
        target = self.hit_test(event.x, event.y)
        if target and target != self._dragging.source_widget:
            target.handle_drag_over(self._dragging, event)

def dispatch_mouse_up(self, event: InputEvent) -> None:
    if self._dragging:
        target = self.hit_test(event.x, event.y)
        if target:
            target.handle_drop(self._dragging, event)
        self._dragging = None
```

## Component Diagram

```
+------------+     +------------------+
|  Platform  | --> |   InputRouter    |
+------------+     +------------------+
                          |
         +----------------+----------------+
         |                |                |
         v                v                v
   +-----------+   +-------------+   +----------+
   | Hit Test  |   |Focus Coord  |   |Drag State|
   +-----------+   +-------------+   +----------+
         |                |
         v                v
   +---------+      +---------+
   | Widget  |      | Widget  |
   +---------+      +---------+
```

## Event Types

| Event | Dispatch Target | Propagation |
|-------|-----------------|-------------|
| mouse_down | Hit test result | Capture + Bubble |
| mouse_up | Hit test result | Capture + Bubble |
| mouse_move | Hit test result | Bubble only |
| mouse_enter | Widget gaining hover | None |
| mouse_leave | Widget losing hover | None |
| key_down | Focused widget | Bubble |
| key_up | Focused widget | Bubble |
| drag_over | Widget under cursor | None |
| drop | Widget under cursor | None |

## Dependencies

- Phase 2 Focus Coordinator
- Widget `contains_point()` method
- Widget tree structure for propagation path

## Risks

1. **Performance**: Hit testing on every mouse move may be expensive. Mitigation: cache hover widget, only re-test on move.

2. **Event ordering**: Simultaneous events (click + focus) may have order dependencies. Mitigation: define strict event order.

3. **Drag compatibility**: Existing widgets may have custom drag handling. Mitigation: dragHandler returns bool to claim event.
