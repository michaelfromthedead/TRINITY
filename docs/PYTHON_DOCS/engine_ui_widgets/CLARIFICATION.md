# CLARIFICATION: engine/ui/widgets

## Philosophical Framing

### The Geometry-First Architecture

The widget system follows a deliberate architectural pattern: widgets are geometry producers, not renderers. Each widget class manages its own state machine, handles input events, and computes geometry data (rectangles, UV coordinates, positions) that an external renderer consumes. This separation is intentional and correct for integration with the Rust renderer backend.

This philosophy means:
- Widgets do not own GPU resources
- Widgets do not issue draw calls
- Widgets are pure state + geometry computation
- The renderer pulls geometry from widgets on demand

### Event-Driven State Machines

Every interactive widget is a complete state machine with:
- Visual states (normal, hovered, pressed, focused, disabled)
- Input handlers that transition between states
- Callbacks emitted on state transitions
- Dirty flags that signal render invalidation

Example from Button:
```python
def handle_mouse_down(self, x: float, y: float, ...) -> bool:
    if not self._enabled or not self.contains_point(x, y):
        return False
    self._is_pressed = True
    self._update_visual_state()
    self._dirty = True
    self._emit_press(True)
    return True
```

### Composition Over Inheritance

Widgets are composed from primitives (Text, Image, Border, Spacer) rather than inheriting from complex base classes. A HealthBar contains Text primitives for values, Image primitives for bars, and computes its own damage/shield logic.

## Design Rationale

### Why No Layout Engine?

The current implementation uses absolute positioning exclusively. Each widget has explicit (x, y, width, height) bounds. This was likely the fastest path to functional widgets but creates maintenance burden:
- Every widget placement requires manual coordinate calculation
- Screen resize requires recalculating all positions
- Responsive UI is impossible without significant manual work

The missing layout engine (flex, grid) is the highest priority gap.

### Why No Focus Coordinator?

Widgets have focus-related state (`_is_focused`) but no system to coordinate focus:
- No tracking of which widget is currently focused
- No Tab/Shift+Tab navigation
- No focus trapping for modals
- No focus restoration after dialog close

Each widget handles focus internally but cannot participate in a focus chain.

### Why No Input Router?

Currently, the application must manually call input handlers on widgets:
```python
# Manual dispatch (current pattern)
for widget in widgets:
    if widget.handle_mouse_down(x, y):
        break
```

A proper input router would:
- Receive all input events centrally
- Determine the target widget (by position or focus)
- Dispatch with proper event propagation (capture, bubble)
- Handle drag coordination across widget boundaries

### Accessibility as First-Class

The widgets include accessibility support:
```python
def get_accessible_text(self) -> str:
    return f"Progress: {self.percentage:.0f}%"

def get_accessible_role(self) -> str:
    return "progressbar"
```

This is correct and must be preserved. The renderer integration should expose an accessibility tree to the platform.

## Integration Points

### Rust Renderer Backend

Widgets produce geometry data through methods like:
- `get_fill_rect()` - rectangle to fill
- `get_thumb_position()` - slider thumb location
- `get_uv_coords()` - texture coordinates

The renderer integration phase must define the protocol for:
1. How the Rust renderer queries widget geometry
2. How dirty flags trigger geometry updates
3. How animation frames synchronize with render loop

### Coordinate Systems

Widgets use local coordinates relative to their bounds. The Minimap demonstrates coordinate transformation:
```python
def world_to_map(self, world_x: float, world_y: float) -> tuple[float, float]:
    # Transform world coords to map-local coords
    # Handles zoom, rotation, centering
```

Layout and input systems must respect these coordinate spaces.

### Serialization

Widgets support `to_dict()`/`from_dict()` for persistence. Any new systems (layout, focus) must integrate with this serialization.

## Key Decisions

1. **Layout is additive**: Layout containers wrap existing widgets; widgets retain their bounds but can delegate positioning to layout.

2. **Focus is opt-in**: Widgets declare focusability; the coordinator manages the chain.

3. **Input router is transparent**: Existing `handle_mouse_*` methods remain; router calls them.

4. **Renderer integration is data-driven**: Python side produces geometry dictionaries; Rust side consumes them.
