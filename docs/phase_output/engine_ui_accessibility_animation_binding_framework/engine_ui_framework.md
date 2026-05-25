# Investigation: engine/ui/framework

## Summary
The UI framework is a fully-implemented, production-quality widget system with ~4,100 lines of code across 5 core modules. It provides a complete hierarchical widget architecture with flexbox-style layout, W3C-compliant event bubbling/capture, comprehensive focus management with trapping and groups, and coordinate system transforms. This is genuine, well-architected framework code ready for integration with a renderer.

## Files
| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `__init__.py` | 182 | Complete | Clean public API with 40+ exports |
| `widget.py` | 1031 | Complete | Base widget with hierarchy, dirty tracking, lifecycle hooks |
| `container.py` | 708 | Complete | HBox, VBox, Stack, ScrollContainer with full layout |
| `events.py` | 662 | Complete | UIEvent, MouseEvent, KeyboardEvent, DragEvent, EventDispatcher |
| `focus.py` | 753 | Complete | FocusManager singleton, groups, traps, tab navigation |
| `coordinate.py` | 770 | Complete | Point, Size, Rect, Margins, Transform2D, CoordinateConverter |
| `config.py` | 51 | Complete | Viewport, focus, layout defaults |

**Total: 4,157 lines**

## Framework Components

### Widget System (`widget.py`)
- `Widget` base class with unique ID generation
- `TrackedDescriptor` for property change tracking and dirty flags
- `WidgetStyle` (background, border, corner radius, opacity, padding)
- `LayoutConstraints` (min/max/preferred sizes)
- Hierarchical parent/child management with circular reference detection
- Z-index sorting, bring-to-front/send-to-back
- Lifecycle hooks: `on_mount`, `on_unmount`, `on_update`, `on_render`
- Hit testing with `hit_test()` and `hit_test_all()`
- Coordinate conversion: local-to-global, global-to-local

### Layout Engine (`container.py`)
- `Container` with automatic child positioning
- `LayoutConfig` with direction, main/cross alignment, padding, spacing
- `LayoutDirection.HORIZONTAL` / `VERTICAL`
- Flexbox-style alignments: START, CENTER, END, STRETCH, SPACE_BETWEEN, SPACE_AROUND, SPACE_EVENLY
- `HBox` and `VBox` convenience containers
- `Stack` for z-order overlapping widgets
- `ScrollContainer` with viewport, scroll offsets, scroll-to-child

### Event System (`events.py`)
- W3C event model with CAPTURE, TARGET, BUBBLE phases
- `UIEvent` base with stop propagation, prevent default
- `MouseEvent` with position, buttons, modifiers, click count, scroll delta
- `KeyboardEvent` with key, keyCode, char, modifiers, repeat
- `FocusEvent` with related target
- `DragEvent` with data, data_type, source
- `EventDispatcher.dispatch()` implements full capture/bubble traversal

### Focus Management (`focus.py`)
- `FocusManager` singleton with focus history
- `FocusGroup` for logical widget groupings with tab order
- `FocusTrap` for modal/dialog focus containment
- `FocusDirection`: NEXT, PREVIOUS, UP, DOWN, LEFT, RIGHT
- Spatial navigation with distance calculation
- Keyboard handling for Tab, Shift+Tab, arrows, Escape
- Widget lifecycle notifications (removed, visibility/enabled changed)

### Coordinate System (`coordinate.py`)
- `Point` with arithmetic, lerp, distance
- `Size` with area, aspect ratio, validation
- `Rect` with edges, corners, intersection, union, expand/contract
- `Margins` for padding/spacing
- `Transform2D` with position, rotation, scale and composition
- `Anchor` enum (9 positions from TOP_LEFT to BOTTOM_RIGHT)
- `CoordinateConverter` for pixel/normalized/viewport/parent conversions
- DPI scaling support

## Implementation
- Real widget system? **YES** - Full hierarchical widget tree with lifecycle, dirty tracking, event dispatch
- Real layout engine? **YES** - Complete flexbox-style layout with alignment, spacing, wrapping support
- Real UI rendering? **PARTIAL** - Framework provides `on_render(context)` hook but no actual renderer; expects external GPU/canvas integration

## Verdict
**REAL IMPLEMENTATION**

This is a complete, production-quality UI framework comparable to Qt's widget system or browser DOM. The only missing piece is the actual rendering backend - the framework correctly delegates this to an external renderer via the `on_render(context)` hook pattern. All core UI concerns (layout, events, focus, coordinates) are fully implemented.

## Evidence

### Widget Hierarchy Management
```python
def add_child(self, child: "Widget") -> "Widget":
    if child._parent is not None:
        raise ValueError(f"Widget '{child.name}' already has a parent")
    if child is self:
        raise ValueError("Cannot add widget as its own child")
    # Check for circular reference
    ancestor = self._parent
    while ancestor is not None:
        if ancestor is child:
            raise ValueError("Cannot add ancestor as child (circular reference)")
        ancestor = ancestor._parent
    child._parent = self
    self._children.append(child)
    self._sort_children_by_z()
    if self._is_mounted:
        child._mount()
    self._mark_dirty("children")
    return child
```

### Flexbox Layout Algorithm
```python
def _layout_horizontal(self, children, x, y, width, height):
    total_width = sum(c.width for c in children)
    total_spacing = spacing * max(0, len(children) - 1)
    remaining = width - total_width - total_spacing
    
    if config.main_alignment == Alignment.SPACE_BETWEEN:
        if len(children) > 1:
            gap = (width - total_width) / (len(children) - 1)
    elif config.main_alignment == Alignment.SPACE_EVENLY:
        even = remaining / (len(children) + 1)
        current_x = x + even
        gap = spacing + even
```

### W3C Event Dispatch
```python
@staticmethod
def dispatch(event: UIEvent, target: "Widget") -> bool:
    event.target = target
    path: list["Widget"] = []
    current = target
    while current is not None:
        path.insert(0, current)
        current = current.parent
    
    # Capture phase (root to target)
    event.phase = EventPhase.CAPTURE
    for widget in path[:-1]:
        if event.is_stopped:
            break
        event.current_target = widget
        widget._dispatch_to_handlers(event, capture=True)
    
    # Target phase
    if not event.is_stopped:
        event.phase = EventPhase.TARGET
        target._dispatch_to_handlers(event, capture=True)
        target._dispatch_to_handlers(event, capture=False)
    
    # Bubble phase (target to root)
    if event.bubbles and not event.is_stopped:
        event.phase = EventPhase.BUBBLE
        for widget in reversed(path[:-1]):
            widget._dispatch_to_handlers(event, capture=False)
```

### Focus Trap for Modals
```python
def push_trap(self, container: "Widget", initial_focus=None, restore_on_exit=True):
    trap = FocusTrap(
        container=container,
        previous_focus=self._focused,
        restore_on_exit=restore_on_exit,
    )
    self._traps.append(trap)
    if initial_focus and self._can_focus(initial_focus):
        self.set_focus(initial_focus)
    elif trap.group:
        first = trap.group.get_first()
        if first:
            self.set_focus(first)
    return trap
```
