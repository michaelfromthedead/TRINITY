# UI Layer Implementation Context

Complete context for implementing `engine/ui/` using the Trinity Pattern + Foundation runtime.

---

## Table of Contents

- [Overview](#overview)
- [Part I: Trinity Recommendations for UI](#part-i-trinity-recommendations-for-ui)
  - [Metaclasses](#metaclasses)
  - [Descriptors](#descriptors)
  - [Decorators](#decorators)
- [Part II: Decorator Stacks for UI](#part-ii-decorator-stacks-for-ui)
- [Part III: UI-Specific Patterns](#part-iii-ui-specific-patterns)
- [Part IV: Engine/UI Directory Structure](#part-iv-engineui-directory-structure)
- [Part V: Implementation Checklist](#part-v-implementation-checklist)

---

## Overview

The UI Layer (`engine/ui/`) implements all user interface functionality following the Trinity Pattern:

1. **Definition-Time (Trinity):** Metaclasses, descriptors, and decorators define widget structure
2. **Runtime (Foundation):** Registry, Tracker, Mirror, and EventLog provide observation and change tracking
3. **Engine Layer:** Actual widget implementations, layout algorithms, rendering, and input handling

### Architecture Reference

From `DIAGRAMS/ARCHITECTURE_UI.md`:

```
UI LAYER
+-- UI Architecture (modes, coordinate systems, layers)
+-- Widget System (primitives, common, game-specific)
+-- Layout System (flex, grid, responsive)
+-- Input Handling (mouse, keyboard, touch, gamepad)
+-- Styling System (themes, states, brushes)
+-- UI Animation (tweens, transitions)
+-- Data Binding (one-way, two-way, list virtualization)
+-- Text & Localization (fonts, RTL, IME)
+-- UI Rendering (batching, clipping, atlas)
+-- Accessibility (screen reader, high contrast)
+-- Screen Management (stack, transitions)
```

### Integration Point

From `GAME_ENGINE_INTEGRATION.md` Section 4.11:

> UI widgets are Components with data binding through `TrackedDescriptor` -> `on_change` subscriptions. When a game model value changes, the Tracker fires the UI subscription, which re-renders the widget. Style properties use `ValidatedDescriptor` to ensure valid CSS-like values. Screen management (push/pop/replace) uses `StateMeta` for screen state machines.

---

## Part I: Trinity Recommendations for UI

### Metaclasses

**Use `ComponentMeta` for all UI widgets.** Widgets are ECS components. This gives you:
- Unique IDs for fast lookup
- Field processing with type annotations
- Automatic descriptor installation
- Foundation Registry integration (instance tracking)

**Use `StateMeta` for screen state machines.** Screen navigation (push/pop/replace) is a state machine. Use StateMeta for:
- Valid transition validation
- Enter/exit hooks
- Machine-scoped registration

**Do NOT create a new `WidgetMeta`.** The existing ComponentMeta handles everything widgets need.

### Descriptors

#### Core Descriptors for UI

| Descriptor | UI Purpose | Example |
|-----------|-----------|---------|
| `TrackedDescriptor` | Dirty flags for re-render | Widget position/size changes trigger layout |
| `ObservableDescriptor` | Data binding callbacks | Model changes -> widget updates |
| `ValidatedDescriptor` | Style property validation | Ensure opacity in [0,1], colors valid |
| `RangeDescriptor` | Numeric constraints | Slider value, progress bar percent |
| `ComputedDescriptor` | Derived layout values | Computed width from children |
| `TransientDescriptor` | Non-serializable state | Cached render data, hover state |
| `BoundDescriptor` | Two-way binding | Input field <-> model value |
| `ImmutableDescriptor` | Read-only after init | Widget ID, parent reference |

#### Descriptor Chains for UI Fields

**Position/Size Fields:**
```
TrackedDescriptor -> StorageDescriptor
```
Changes trigger layout recalculation.

**Style Properties (validated):**
```
TrackedDescriptor -> ValidatedDescriptor -> StorageDescriptor
```
Example: `opacity: Annotated[float, Tracked, Range(0, 1)] = 1.0`

**Data-Bound Fields:**
```
ObservableDescriptor -> TrackedDescriptor -> StorageDescriptor
```
Model changes fire observer callbacks that update the widget.

**Computed Layout:**
```
ComputedDescriptor
```
Read-only, computed on access (e.g., `computed_width`).

#### Annotated Field Syntax (Preferred)

```python
from typing import Annotated
from trinity.descriptors import Tracked, Range, Transient, Computed

@component
class Button(Widget):
    # Position/size (tracked for layout)
    x: Annotated[float, Tracked] = 0.0
    y: Annotated[float, Tracked] = 0.0
    width: Annotated[float, Tracked] = 100.0
    height: Annotated[float, Tracked] = 40.0
    
    # Style properties (tracked + validated)
    opacity: Annotated[float, Tracked, Range(0, 1)] = 1.0
    corner_radius: Annotated[float, Tracked, Range(0, 100)] = 4.0
    
    # State (tracked for visual updates)
    is_hovered: Annotated[bool, Tracked] = False
    is_pressed: Annotated[bool, Tracked] = False
    is_disabled: Annotated[bool, Tracked] = False
    
    # Transient (not serialized)
    _cached_mesh: Annotated[Any, Transient] = None
    _dirty_layout: Annotated[bool, Transient] = True
```

### Decorators

#### Existing UI Decorators

From `trinity/decorators/ui.py`:

| Decorator | Purpose | Steps |
|-----------|---------|-------|
| `@widget` | Mark class as UI widget | `TAG(widget=True), TAG(widget_style), REGISTER(ui)` |
| `@layout` | Configure layout direction/gap/padding | `TAG(layout=True), TAG(layout_direction), TAG(layout_gap), TAG(layout_padding), REGISTER(ui)` |

#### Related Decorators for UI

From other decorator modules:

| Decorator | Module | Purpose |
|-----------|--------|---------|
| `@accessible` | `accessibility.py` | Screen reader support, ARIA roles |
| `@state_machine` | `state_machine.py` | Screen navigation state machine |
| `@on_enter`, `@on_exit` | `state_machine.py` | Screen transition hooks |
| `@localized` | `localization.py` | Text translation markers |
| `@rtl_aware` | `localization.py` | Right-to-left layout support |
| `@text_overflow` | `localization.py` | Truncate/shrink/scroll/wrap |
| `@tween` | `animation.py` | UI animation configuration |
| `@on_change` | `lifecycle.py` | Field change hooks |
| `@input_action` | `input.py` | Input action bindings |
| `@serializable` | `data_flow.py` | Save/load widget state |

#### New UI Decorators to Create

| Decorator | Purpose | Steps |
|-----------|---------|-------|
| `@focusable` | Mark widget as focusable | `TAG(focusable=True), TAG(tab_order)` |
| `@draggable` | Enable drag interactions | `TAG(draggable=True), HOOK(on_drag_start), HOOK(on_drag_end)` |
| `@droppable` | Enable drop targets | `TAG(droppable=True), HOOK(on_drop)` |
| `@scrollable` | Enable scroll behavior | `TAG(scrollable=True), TAG(scroll_direction)` |
| `@tooltip` | Attach tooltip configuration | `TAG(tooltip_text), TAG(tooltip_delay)` |
| `@ui_layer` | Assign to UI layer | `TAG(ui_layer), REGISTER(ui)` |
| `@anchor` | Configure anchor point | `TAG(anchor_x), TAG(anchor_y), TAG(pivot_x), TAG(pivot_y)` |
| `@responsive` | Breakpoint-based adaptations | `TAG(breakpoints), TAG(responsive_rules)` |

---

## Part II: Decorator Stacks for UI

### Recommended New Stacks

Create in `trinity/decorators/builtin_stacks/ui.py`:

```python
@parameterized_stack
def interactive_widget(
    focusable: bool = True,
    tab_order: int = 0,
) -> Stack:
    """Widget that responds to user input."""
    return stack(
        widget(),
        focusable(order=tab_order) if focusable else _noop,
        accessible(role="button"),
        track_changes,
        component,
    )

@parameterized_stack
def data_bound_widget(
    model_field: str = "",
    two_way: bool = True,
) -> Stack:
    """Widget with automatic model data binding."""
    return stack(
        widget(),
        track_changes,
        # binding configured via model_field
        component,
    )

@parameterized_stack
def game_hud_element(
    layer: str = "hud",
    persist: bool = False,
) -> Stack:
    """HUD element for gameplay overlay."""
    return stack(
        widget(),
        ui_layer(layer=layer),
        track_changes,
        serializable(format="binary") if persist else _noop,
        component,
    )

@parameterized_stack
def screen(
    initial: str = "",
    states: set = None,
    transitions: dict = None,
) -> Stack:
    """Full-screen UI with state machine navigation."""
    return stack(
        widget(),
        state_machine(initial=initial, states=states or set(), transitions=transitions or {}),
        accessible(role="dialog"),
        component,
    )

@parameterized_stack
def accessible_widget(
    role: str = "button",
    screen_reader: str = None,
) -> Stack:
    """Widget with full accessibility support."""
    return stack(
        widget(),
        accessible(role=role, screen_reader=screen_reader),
        focusable(),
        rtl_aware(),
        track_changes,
        component,
    )
```

### Composite Stacks

```python
@parameterized_stack
def inventory_slot(
    size: int = 64,
) -> Stack:
    """Inventory slot with drag-and-drop support."""
    return stack(
        interactive_widget(focusable=True),
        draggable(),
        droppable(),
        tooltip(delay=0.5),
    )

@parameterized_stack
def localized_text_widget(
    key: str = "",
    overflow: str = "truncate",
) -> Stack:
    """Text widget with localization support."""
    return stack(
        widget(),
        localized(key=key) if key else _noop,
        text_overflow(strategy=overflow),
        rtl_aware(),
        accessible(role="text"),
        component,
    )
```

---

## Part III: UI-Specific Patterns

### Widget Hierarchy Pattern

Widgets form a tree. Parent-child relationships are tracked via descriptors.

```python
@component
class Widget:
    # Identity (immutable after creation)
    id: Annotated[int, Immutable] = 0
    
    # Hierarchy
    parent: Annotated[Optional["Widget"], Tracked] = None
    children: Annotated[list["Widget"], Tracked] = field(default_factory=list)
    
    # Transform
    local_x: Annotated[float, Tracked] = 0.0
    local_y: Annotated[float, Tracked] = 0.0
    width: Annotated[float, Tracked] = 0.0
    height: Annotated[float, Tracked] = 0.0
    
    # Visibility
    visible: Annotated[bool, Tracked] = True
    enabled: Annotated[bool, Tracked] = True
    
    # Computed (read-only)
    @computed
    def global_x(self) -> float:
        return self.local_x + (self.parent.global_x if self.parent else 0)
    
    @computed
    def global_y(self) -> float:
        return self.local_y + (self.parent.global_y if self.parent else 0)
```

### Data Binding Pattern

Use `ObservableDescriptor` + `BoundDescriptor` for reactive UI.

```python
# Model (game state)
@component
class PlayerStats:
    health: Annotated[float, Tracked, Observable, Range(0, 100)] = 100.0
    max_health: Annotated[float, Tracked] = 100.0

# Widget (bound to model)
@component
class HealthBar(Widget):
    # Bound to PlayerStats.health
    current_value: Annotated[float, Tracked, Bound] = 0.0
    max_value: Annotated[float, Tracked] = 100.0
    
    @computed
    def fill_percent(self) -> float:
        return self.current_value / self.max_value if self.max_value > 0 else 0

# Binding setup (in Foundation)
def bind_health_bar(health_bar: HealthBar, player: PlayerStats):
    add_observer(player, "health", lambda obj, field, old, new: 
        setattr(health_bar, "current_value", new))
```

### Screen Stack Pattern

Use `StateMeta` for screen navigation.

```python
@state_machine(
    initial="main_menu",
    states={"main_menu", "gameplay", "pause", "inventory", "settings"},
    transitions={
        "main_menu": {"gameplay", "settings"},
        "gameplay": {"pause", "inventory"},
        "pause": {"gameplay", "main_menu"},
        "inventory": {"gameplay"},
        "settings": {"main_menu"},
    }
)
class ScreenManager:
    _stack: list[str] = field(default_factory=list)
    
    @on_enter(state="pause")
    def on_pause_enter(self):
        # Pause game logic
        pass
    
    @on_exit(state="pause")
    def on_pause_exit(self):
        # Resume game logic
        pass
```

### Layout System Pattern

Layout containers use `@layout` decorator with tracked children.

```python
@layout(direction="vertical", gap=8, padding=16)
@component
class VBox(Widget):
    children: Annotated[list[Widget], Tracked] = field(default_factory=list)
    
    def calculate_layout(self):
        """Recalculate child positions based on layout config."""
        y_offset = self._layout_padding
        for child in self.children:
            child.local_x = self._layout_padding
            child.local_y = y_offset
            y_offset += child.height + self._layout_gap
```

### Styling Pattern

Style properties use `ValidatedDescriptor` with theme support.

```python
@component
class StyleProperties:
    # Colors (validated as hex or named)
    background_color: Annotated[str, Tracked, Validated(is_color)] = "#FFFFFF"
    foreground_color: Annotated[str, Tracked, Validated(is_color)] = "#000000"
    border_color: Annotated[str, Tracked, Validated(is_color)] = "#CCCCCC"
    
    # Numeric (validated ranges)
    border_width: Annotated[float, Tracked, Range(0, 100)] = 0.0
    corner_radius: Annotated[float, Tracked, Range(0, 100)] = 0.0
    opacity: Annotated[float, Tracked, Range(0, 1)] = 1.0
    
    # Font
    font_family: Annotated[str, Tracked] = "default"
    font_size: Annotated[float, Tracked, Range(1, 200)] = 14.0
    font_weight: Annotated[str, Tracked, Choice(["normal", "bold", "light"])] = "normal"
```

### Input Handling Pattern

Input events flow through Foundation's EventLog.

```python
@event
class UIEvent:
    widget_id: int
    timestamp: float

@event
class ClickEvent(UIEvent):
    button: int  # 0=left, 1=right, 2=middle
    x: float
    y: float

@event
class KeyEvent(UIEvent):
    key: str
    modifiers: int  # Shift=1, Ctrl=2, Alt=4

@event
class FocusEvent(UIEvent):
    focused: bool
```

### Accessibility Pattern

Use `@accessible` decorator with proper roles.

```python
@accessible(role="button", screen_reader="Submit form button")
@widget()
@component
class SubmitButton(Widget):
    label: Annotated[str, Tracked, Localized(key="submit_btn")] = "Submit"
    
    # Accessibility metadata
    _aria_label: str = "Submit"
    _aria_disabled: bool = False
```

---

## Part IV: Engine/UI Directory Structure

```
engine/ui/
+-- __init__.py              # Public API exports
+-- framework/
|   +-- __init__.py
|   +-- widget.py            # Base Widget component
|   +-- container.py         # Container widget base
|   +-- events.py            # UI event types
|   +-- focus.py             # Focus management system
|   +-- coordinate.py        # Coordinate systems and transforms
+-- widgets/
|   +-- __init__.py
|   +-- primitives/
|   |   +-- image.py         # Image widget
|   |   +-- text.py          # Text block widget
|   |   +-- border.py        # Border/rectangle widget
|   |   +-- spacer.py        # Empty space widget
|   +-- input/
|   |   +-- button.py        # Button widget
|   |   +-- checkbox.py      # Checkbox widget
|   |   +-- slider.py        # Slider widget
|   |   +-- text_input.py    # Text input widget
|   |   +-- dropdown.py      # Dropdown/select widget
|   +-- display/
|   |   +-- label.py         # Label widget
|   |   +-- progress_bar.py  # Progress bar widget
|   |   +-- icon.py          # Icon widget
|   +-- game/
|   |   +-- health_bar.py    # Health/resource bar
|   |   +-- minimap.py       # Minimap widget
|   |   +-- inventory_slot.py# Inventory slot
|   |   +-- damage_numbers.py# Floating damage text
|   |   +-- tooltip.py       # Tooltip widget
+-- styling/
|   +-- __init__.py
|   +-- style.py             # Style properties component
|   +-- theme.py             # Theme system
|   +-- brush.py             # Brush types (solid, gradient, image)
|   +-- color.py             # Color utilities
+-- binding/
|   +-- __init__.py
|   +-- binding.py           # Data binding system
|   +-- converter.py         # Value converters
|   +-- validation.py        # Binding validation
+-- screens/
|   +-- __init__.py
|   +-- screen.py            # Base screen class
|   +-- screen_stack.py      # Screen stack manager
|   +-- transitions.py       # Screen transition effects
+-- layout/
|   +-- __init__.py          # (if needed for complex layouts)
```

---

## Part V: Implementation Checklist

### Phase 1: Foundation

- [ ] `framework/widget.py` - Base Widget component with TrackedDescriptor fields
- [ ] `framework/events.py` - UI event types (Click, Key, Focus, Hover, Drag)
- [ ] `framework/coordinate.py` - Coordinate systems, anchor points, transforms
- [ ] `framework/focus.py` - Focus management, tab order, navigation

### Phase 2: Primitive Widgets

- [ ] `widgets/primitives/image.py` - Image display
- [ ] `widgets/primitives/text.py` - Text block (with localization support)
- [ ] `widgets/primitives/border.py` - Rectangle/border
- [ ] `widgets/primitives/spacer.py` - Empty space

### Phase 3: Input Widgets

- [ ] `widgets/input/button.py` - Clickable button
- [ ] `widgets/input/checkbox.py` - Boolean toggle
- [ ] `widgets/input/slider.py` - Range value (with RangeDescriptor)
- [ ] `widgets/input/text_input.py` - Text input field
- [ ] `widgets/input/dropdown.py` - Dropdown select

### Phase 4: Styling

- [ ] `styling/style.py` - Style properties component
- [ ] `styling/theme.py` - Theme system (light/dark/high contrast)
- [ ] `styling/brush.py` - Solid, gradient, image brushes
- [ ] `styling/color.py` - Color utilities and validation

### Phase 5: Data Binding

- [ ] `binding/binding.py` - One-way, two-way, one-time binding
- [ ] `binding/converter.py` - Value converters for binding
- [ ] `binding/validation.py` - Input validation

### Phase 6: Screen Management

- [ ] `screens/screen.py` - Base screen class
- [ ] `screens/screen_stack.py` - Push/pop/replace operations
- [ ] `screens/transitions.py` - Fade, slide, zoom transitions

### Phase 7: Game Widgets

- [ ] `widgets/game/health_bar.py` - Health/mana/stamina bars
- [ ] `widgets/game/minimap.py` - World overview
- [ ] `widgets/game/inventory_slot.py` - Drag-drop item slot
- [ ] `widgets/game/damage_numbers.py` - Floating combat text
- [ ] `widgets/game/tooltip.py` - Contextual tooltips

### Phase 8: Integration

- [ ] Wire TrackedDescriptor changes to layout invalidation
- [ ] Wire ObservableDescriptor to UI re-render
- [ ] Wire Foundation Tracker for undo/redo in editor
- [ ] Wire Foundation Mirror for UI inspector

---

## Quick Reference

### Descriptor Choice Guide

| Need | Descriptor | Example |
|------|-----------|---------|
| Trigger re-render on change | `Tracked` | Position, size, visibility |
| Validate style values | `Validated`, `Range`, `Choice` | Opacity, colors, font size |
| Computed layout values | `Computed` | Global position, fill percent |
| Data binding callback | `Observable` | Model -> widget sync |
| Two-way binding | `Bound` | Input field <-> model |
| Non-serializable | `Transient` | Cached meshes, hover state |
| Read-only after init | `Immutable` | Widget ID |

### Decorator Choice Guide

| Need | Decorator | Module |
|------|-----------|--------|
| Mark as widget | `@widget` | `ui.py` |
| Layout container | `@layout` | `ui.py` |
| Accessibility | `@accessible` | `accessibility.py` |
| Screen navigation | `@state_machine` | `state_machine.py` |
| Screen hooks | `@on_enter`, `@on_exit` | `state_machine.py` |
| Translation | `@localized` | `localization.py` |
| RTL support | `@rtl_aware` | `localization.py` |
| Text overflow | `@text_overflow` | `localization.py` |
| Animation | `@tween` | `animation.py` |
| Change detection | `@track_changes` | `debug_safety.py` |
| Input binding | `@input_action` | `input.py` |
| Serialization | `@serializable` | `data_flow.py` |

### Foundation Integration Points

| System | UI Use |
|--------|--------|
| Registry | Widget type lookup, instance tracking |
| Tracker | Dirty flags, change subscriptions, undo/redo |
| EventLog | Input events, screen transitions, debugging |
| Mirror | Widget introspection in editor |
| Bridge | ShellLang access to UI for debugging |

---

## References

- `docs/TRINITY_LATEST.md` - Full Trinity Pattern specification
- `docs/GAME_ENGINE_INTEGRATION.md` - Trinity <-> Foundation integration
- `docs/GAME_ENGINE_INTEGRATION_TODO.md` - Section 10 (UI Layer)
- `DIAGRAMS/ARCHITECTURE_UI.md` - UI layer architecture
- `trinity/decorators/ui.py` - Existing UI decorators
- `trinity/decorators/accessibility.py` - Accessibility decorators
- `trinity/decorators/localization.py` - Localization decorators
- `trinity/decorators/state_machine.py` - State machine decorators
- `trinity/decorators/animation.py` - Animation decorators
- `trinity/descriptors/` - All descriptor implementations
