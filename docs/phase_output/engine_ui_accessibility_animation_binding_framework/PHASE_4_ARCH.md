# PHASE 4 ARCHITECTURE: Framework Module

## Scope

5 files, ~3,924 lines in `engine/ui/framework/`

| File | Lines | Purpose |
|------|-------|---------|
| widget.py | 1031 | Base widget class for UI hierarchy |
| coordinate.py | 770 | Coordinate system utilities |
| focus.py | 753 | Focus management system |
| container.py | 708 | Container widgets for layout |
| events.py | 662 | Event system following W3C model |

---

## Component Architecture

### Widget (widget.py)

```
Widget (base class)
    |
    +-- Hierarchy
    |       +-- parent: Widget | None
    |       +-- children: List[Widget]
    |       +-- traverse_depth_first()
    |       +-- traverse_breadth_first()
    |
    +-- Geometry
    |       +-- x, y: float (local position)
    |       +-- width, height: float
    |       +-- constraints: LayoutConstraints
    |       +-- transform: Transform2D
    |
    +-- State
    |       +-- is_visible: bool
    |       +-- is_enabled: bool
    |       +-- is_focusable: bool
    |       +-- is_interactive: bool
    |
    +-- Style
    |       +-- style: WidgetStyle
    |       +-- dirty: bool (needs redraw)
    |
    +-- Lifecycle
    |       +-- on_mount()
    |       +-- on_unmount()
    |       +-- on_update(dt)
    |       +-- on_render(context)
    |
    +-- Event Handlers
            +-- _handlers: Dict[str, List[Callable]]
            +-- on(event_type, handler)
            +-- off(event_type, handler)

TrackedDescriptor
    +-- Property descriptor with dirty tracking
    +-- Set marks widget as dirty

LayoutConstraints
    +-- min_width, max_width
    +-- min_height, max_height
    +-- preferred_width, preferred_height
```

---

### Coordinate System (coordinate.py)

```
Point
    +-- x, y: float
    +-- Operations: add, sub, scale, dot, length, normalize

Size
    +-- width, height: float
    +-- Operations: add, scale, contains

Rect
    +-- x, y, width, height: float
    +-- Properties: left, right, top, bottom, center
    +-- Operations: contains_point, intersects, union, intersection

Margins
    +-- top, right, bottom, left: float
    +-- Properties: horizontal, vertical
    +-- inset(rect) → smaller rect

Transform2D
    +-- 2D affine transformation matrix [a, b, c, d, tx, ty]
    |
    +-- identity() → no transformation
    +-- translate(dx, dy)
    +-- rotate(angle)
    +-- scale(sx, sy)
    +-- compose(other) → matrix multiplication
    +-- invert() → inverse matrix
    +-- transform_point(point) → transformed point

CoordinateConverter
    +-- local_to_global(widget, point)
    +-- global_to_local(widget, point)
    +-- widget_to_widget(source, target, point)
```

**Affine Matrix Layout**:
```
| a  c  tx |
| b  d  ty |
| 0  0  1  |

point' = [a*x + c*y + tx, b*x + d*y + ty]
```

---

### Focus System (focus.py)

```
FocusManager (singleton)
    |
    +-- _focused: Widget | None
    +-- _focus_history: List[Widget]
    +-- _trap_stack: List[FocusTrap]
    |
    +-- set_focus(widget)
    +-- clear_focus()
    +-- focus_next() → Tab
    +-- focus_previous() → Shift+Tab
    +-- restore_focus()

FocusGroup
    +-- widgets: List[Widget]
    +-- direction: HORIZONTAL | VERTICAL | BOTH
    +-- wrap: bool

FocusTrap
    +-- container: Widget
    +-- previous_focus: Widget | None
    +-- focusables: List[Widget]
    |
    +-- Tab navigation confined to container
    +-- Used for modals and dialogs
```

**Focus Order Calculation**:
1. Find all focusable descendants
2. Sort by tab_index (if specified)
3. Fall back to document order (depth-first)
4. Handle wrap-around at boundaries

---

### Container Layouts (container.py)

```
Container (base)
    +-- children: List[Widget]
    +-- padding: Margins
    +-- _layout_children() [abstract]

HBox (horizontal layout)
    +-- spacing: float
    +-- main_axis_align: START | CENTER | END | SPACE_BETWEEN | SPACE_AROUND
    +-- cross_axis_align: START | CENTER | END | STRETCH

VBox (vertical layout)
    +-- Same properties as HBox, rotated 90°

Stack (z-order stacking)
    +-- Children overlap at same position
    +-- Last child is on top (highest z-order)

ScrollContainer
    +-- content: Widget
    +-- scroll_x, scroll_y: float
    +-- viewport clipping
    +-- scroll bars (optional)

LayoutConfig
    +-- flex: float (flexible sizing)
    +-- alignment: Alignment (per-child override)
```

**HBox Layout Algorithm**:
```python
x = padding.left
for child in children:
    child.x = x
    child.y = calculate_cross_position(child)
    x += child.width + spacing
```

**VBox Layout Algorithm**:
```python
y = padding.top
for child in children:
    child.y = y
    child.x = calculate_cross_position(child)
    y += child.height + spacing
```

---

### Event System (events.py)

```
UIEvent (base)
    +-- type: str
    +-- target: Widget
    +-- current_target: Widget
    +-- phase: CAPTURE | TARGET | BUBBLE
    +-- timestamp: float
    +-- is_stopped: bool
    +-- is_stopped_immediate: bool
    +-- is_default_prevented: bool
    |
    +-- stop_propagation()
    +-- stop_immediate_propagation()
    +-- prevent_default()

MouseEvent extends UIEvent
    +-- x, y: float (local coordinates)
    +-- global_x, global_y: float
    +-- button: LEFT | MIDDLE | RIGHT
    +-- buttons: int (bit flags)
    +-- modifiers: shift, ctrl, alt, meta

KeyboardEvent extends UIEvent
    +-- key: str (logical key name)
    +-- code: str (physical key code)
    +-- modifiers: shift, ctrl, alt, meta
    +-- is_repeat: bool

FocusEvent extends UIEvent
    +-- related_target: Widget | None

DragEvent extends UIEvent
    +-- data_transfer: DataTransfer
    +-- effect: NONE | COPY | MOVE | LINK

EventDispatcher
    +-- dispatch(event, target) → bool
```

**W3C Event Dispatch Algorithm**:
```
1. Build path from root to target (ancestors)

2. CAPTURE PHASE (root → target, excluding target):
   for each widget in path[:-1]:
       event.phase = CAPTURE
       widget._dispatch_to_handlers(event, capture=True)
       if event.is_stopped: break

3. TARGET PHASE:
   if not event.is_stopped:
       event.phase = TARGET
       target._dispatch_to_handlers(event, capture=True)
       if not event.is_stopped_immediate:
           target._dispatch_to_handlers(event, capture=False)

4. BUBBLE PHASE (target → root, excluding target):
   if event.bubbles and not event.is_stopped:
       for each widget in reversed(path[:-1]):
           event.phase = BUBBLE
           widget._dispatch_to_handlers(event, capture=False)
           if event.is_stopped: break

5. Return: not event.is_default_prevented
```

---

## Data Flow

### Input to Widget

```
Raw Input Event (from engine/input)
    |
    v
Convert to UIEvent
    |
    v
Hit Test → find target Widget
    |
    v
EventDispatcher.dispatch(event, target)
    |
    v
W3C event phases execute
    |
    v
Handlers may:
    +-- Update state
    +-- Start animation
    +-- Change focus
    +-- Modify data (binding)
```

### Layout Update

```
Widget property changes
    |
    v
TrackedDescriptor marks dirty
    |
    v
Layout pass requested
    |
    v
Root container._layout_children()
    |
    v
Recursive layout calculation
    |
    v
Render pass with new positions
```

---

## Integration Points

| From | To | Purpose |
|------|----|---------| 
| engine/input | EventDispatcher | Raw input conversion |
| FocusManager | KeyboardNavigator | Tab navigation |
| Widget | Binding | Property notification |
| Container | Transform2D | Child positioning |
| ScrollContainer | VirtualizedListView | Scrolling coordination |

---

## Design Decisions

### D1: W3C Event Model

**Decision**: Full W3C event dispatch with capture, target, and bubble phases.

**Rationale**: Industry standard, familiar to web developers, enables event delegation and complex interaction patterns.

### D2: Hit Testing in Z-Order

**Decision**: Children iterated in reverse order for hit testing.

**Rationale**: Last child rendered on top should receive events first.

### D3: Affine Transforms

**Decision**: Full 2D affine transformation matrices.

**Rationale**: Supports arbitrary rotation, scaling, skewing. Essential for animation effects.

### D4: Focus Trapping

**Decision**: Modal dialogs trap focus within their bounds.

**Rationale**: Accessibility requirement. Users shouldn't Tab to hidden elements behind a modal.

### D5: Layout Constraints

**Decision**: Min/max/preferred constraints for each dimension.

**Rationale**: Flexible layouts that adapt to content while respecting boundaries.

### D6: Dirty Tracking

**Decision**: TrackedDescriptor marks widget dirty on property change.

**Rationale**: Only re-render widgets that actually changed. Essential for performance.

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Deep hierarchy event dispatch | Performance | Short-circuit on is_stopped |
| Complex transform chains | Floating point drift | Recompute from root periodically |
| Focus trap escape | Accessibility failure | Always provide escape mechanism |
| Layout thrashing | Frame drops | Batch layout updates, single pass |
| Hit test performance | Slow input response | Spatial partitioning for large UIs |
