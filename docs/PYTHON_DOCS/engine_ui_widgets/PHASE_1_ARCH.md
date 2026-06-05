# PHASE 1 ARCHITECTURE: Layout Engine

## Problem Statement

Widgets currently use absolute positioning only. Each widget has explicit (x, y, width, height) bounds but no system to compute these bounds automatically from layout rules. This makes responsive UI impossible and requires manual coordinate calculation for every widget placement.

## Architectural Decision

### Layout Container Abstraction

Introduce layout containers that wrap widgets and compute their bounds:

```
LayoutContainer
  - FlexLayout (flexbox-style)
  - GridLayout (CSS grid-style)
  - StackLayout (vertical/horizontal stacks)
```

Widgets retain their existing bounds properties. Layout containers set these bounds when layout is computed.

### Layout Hierarchy

```
FlexLayout (root)
  +-- GridLayout (sidebar)
  |     +-- Button
  |     +-- Button
  +-- FlexLayout (content)
        +-- Label
        +-- TextInput
```

Each container computes child bounds, then recursively triggers child layout.

### Key Interfaces

```python
class LayoutContainer:
    def add_child(self, widget, constraints=None) -> None: ...
    def remove_child(self, widget) -> None: ...
    def compute_layout(self) -> None: ...
    def mark_layout_dirty(self) -> None: ...
```

```python
class FlexLayout(LayoutContainer):
    direction: FlexDirection  # ROW, COLUMN
    justify: JustifyContent   # START, CENTER, END, SPACE_BETWEEN, SPACE_AROUND
    align: AlignItems         # START, CENTER, END, STRETCH
    gap: float               # spacing between children
```

```python
class GridLayout(LayoutContainer):
    columns: list[GridTrack]  # fixed, fractional, or auto
    rows: list[GridTrack]
    gap: tuple[float, float]  # row_gap, column_gap
```

### Integration with Existing Widgets

Widgets do not change. Layout containers call:
```python
widget.x = computed_x
widget.y = computed_y
widget.width = computed_width
widget.height = computed_height
```

After setting bounds, containers set the widget's dirty flag to trigger re-render.

### Layout Invalidation

Changes that trigger layout recompute:
- Child added/removed
- Child size hint changed
- Container bounds changed
- Container properties changed (gap, direction, etc.)

Dirty propagation:
1. Widget marks layout dirty on parent
2. Parent marks dirty on its parent (up to root)
3. On next frame, root computes layout down

### Size Hints

Widgets can provide size hints for layout:
```python
def get_min_size(self) -> tuple[float, float]: ...
def get_preferred_size(self) -> tuple[float, float]: ...
def get_max_size(self) -> tuple[float, float]: ...
```

Default implementation returns current (width, height). Widgets can override.

## Component Diagram

```
+----------------+
|  LayoutRoot    |  <- root container, sized to window
+----------------+
        |
+-------v--------+
|  FlexLayout    |  <- computes child bounds via flexbox
+----------------+
     |       |
+----v--+ +--v----+
| Label | | Button|  <- widgets receive computed bounds
+-------+ +-------+
```

## Dependencies

- Existing widget bounds properties (x, y, width, height)
- Existing dirty-tracking system
- No external layout library (custom implementation)

## Risks

1. **Performance**: Deep layout hierarchies may cause cascading recomputes. Mitigation: dirty flag prevents unnecessary recomputes.

2. **Compatibility**: Widgets using absolute positioning must coexist with layout-managed widgets. Mitigation: layout is opt-in; widgets without a layout parent use manual bounds.

3. **Complexity**: Flexbox and grid algorithms are non-trivial. Mitigation: start with simplified subset (no wrapping, fixed tracks only).
