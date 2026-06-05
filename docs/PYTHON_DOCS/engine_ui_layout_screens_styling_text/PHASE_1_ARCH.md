# PHASE 1 ARCHITECTURE: Layout Module

---

## Overview

The Layout Module provides CSS-compatible layout algorithms for positioning UI elements within containers. It comprises 6 files (~4,394 lines) implementing Grid, Flexbox, HBox, VBox, Responsive, and Canvas layouts.

---

## Component Architecture

### Grid Layout (grid.py — 909 lines)

**Purpose**: CSS Grid layout implementation with track sizing and item placement.

**Classes**:
- `TrackSize` — Defines column/row sizing (fixed, auto, fr, minmax)
- `GridItem` — An item placed in the grid with span support
- `GridLayout` — Container managing grid items

**Key Algorithms**:

1. **Track Sizing Algorithm** (`_calculate_track_sizes()`):
   - Resolve fixed sizes first
   - Calculate min-content and max-content for auto tracks
   - Distribute remaining space to `fr` units proportionally

2. **`fr` Unit Resolution** (`resolve_fr_units()`):
   - Sum all `fr` values to get total flex factor
   - Divide available space by total flex
   - Assign space to each `fr` track proportionally

3. **Item Placement** (`_place_item()`):
   - Handle row/column spans
   - Resolve auto-placement for items without explicit positions
   - Calculate final bounds from track positions

**Data Flow**:
```
GridLayout.layout(available_size)
    -> _calculate_track_sizes(columns, available_width)
    -> _calculate_track_sizes(rows, available_height)
    -> resolve_fr_units(tracks, remaining_space)
    -> _place_item(item, column_positions, row_positions)
    -> item.bounds = calculated_rect
```

---

### Flexbox Layout (flex.py — 887 lines)

**Purpose**: CSS Flexbox implementation with main/cross axis alignment.

**Classes**:
- `FlexItem` — Item with flex grow/shrink/basis properties
- `FlexLayout` — Container managing flex items
- `FlexLine` — A single line of items (for wrapping)

**Key Algorithms**:

1. **Free Space Distribution** (`_distribute_free_space()`):
   - Calculate total flex grow/shrink factors
   - Distribute positive space via grow factors
   - Distribute negative space via shrink factors (clamped to min-size)
   - Track frozen items that hit min/max constraints

2. **Multi-Line Wrapping** (`_wrap_lines()`):
   - Accumulate items until line overflows
   - Create new `FlexLine` at wrap point
   - Handle `flex-wrap: wrap-reverse`

3. **Cross Axis Alignment** (`_align_cross_axis()`):
   - `align-items`: stretch, flex-start, flex-end, center, baseline
   - `align-content`: distribution across multiple lines
   - Handle stretched items filling cross axis

**Justification Modes**:
- `flex-start`, `flex-end`, `center`
- `space-between`, `space-around`, `space-evenly`

---

### HBox Layout (hbox.py — 676 lines)

**Purpose**: Simplified horizontal box layout (flexbox subset).

**Classes**:
- `HBoxItem` — Child with flex, alignment, spacing properties
- `HBox` — Horizontal container

**Key Methods**:
- `_calculate_sizes()` — Flex distribution along X axis
- `_position_children()` — Apply spacing and alignment

**Simplifications over Flexbox**:
- Fixed main axis (horizontal)
- No wrapping
- Simplified API for common cases

---

### VBox Layout (vbox.py — 654 lines)

**Purpose**: Vertical box layout, symmetric to HBox.

**Classes**:
- `VBoxItem` — Child with flex, alignment, spacing properties
- `VBox` — Vertical container

**Architecture Note**: Shares implementation pattern with HBox but swaps axes. Could potentially share a base class `LinearBox` with axis parameter.

---

### Responsive Layout (responsive.py — 652 lines)

**Purpose**: Responsive design system for adaptive layouts.

**Classes**:
- `Breakpoint` — Width threshold with name (mobile, tablet, desktop, etc.)
- `SafeAreaInsets` — Notch/cutout safe areas (top, bottom, left, right)
- `ResponsiveContainer` — Container that adapts to viewport size
- `ResponsiveValue` — Value that changes per breakpoint

**Key Methods**:
- `_find_active_breakpoint()` — Determine current breakpoint from viewport width
- `get_current_value()` — Resolve `ResponsiveValue` to concrete value

**Breakpoint Strategy**:
```
0-599px     -> mobile
600-899px   -> tablet
900-1199px  -> desktop
1200px+     -> widescreen
```

**Container Query Support**: Containers can respond to their own size, not just viewport.

---

### Canvas Layout (canvas.py — 616 lines)

**Purpose**: Absolute positioning with anchor points and z-ordering.

**Classes**:
- `CanvasItem` — Positioned item with anchor, pivot, z-index
- `CanvasLayout` — Container for absolutely positioned items

**Key Methods**:
- `_apply_anchor()` — Resolve anchor point (0-1 range to parent bounds)
- `_apply_pivot()` — Offset by pivot point (rotation/scale origin)
- `_sort_by_z_index()` — Order items for rendering
- `hit_test()` — Find item at point with z-order traversal

**Anchor System**:
```
(0, 0) = top-left
(0.5, 0.5) = center
(1, 1) = bottom-right
```

---

## Module Dependencies

```
grid.py      --(standalone)
flex.py      --(standalone)
hbox.py      --(may use flex internally)
vbox.py      --(may use flex internally)
responsive.py --(standalone)
canvas.py    --(standalone)
```

All layout modules depend on a common `Rect` or bounds type (likely from a core geometry module).

---

## Integration Points

1. **Widget System** — Widgets use layouts to position children
2. **Styling System** — Layout properties (margin, padding) come from styles
3. **Screen System** — Screens contain layouts for their content

---

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Grid track sizing | O(rows * cols) | Single pass for fixed, one for fr |
| Flexbox distribution | O(n) per pass | May iterate for frozen items |
| Hit testing | O(n) | Could use spatial index |
| Z-order sort | O(n log n) | Sorted once per layout pass |

---

## Design Decisions

1. **CSS Compatibility** — Grid and Flexbox mirror CSS semantics for familiarity
2. **Simplified Variants** — HBox/VBox for simpler use cases
3. **Responsive First** — Breakpoint system built-in, not bolted on
4. **Anchor-Based Canvas** — Game-UI-friendly absolute positioning
