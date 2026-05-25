# PHASE 1 TODO: Layout Engine

## Prerequisites
- Existing widget system with bounds properties (x, y, width, height)
- Dirty-tracking system functional

---

## Task 1.1: Layout Container Base Class

**File**: `engine/ui/layout/container.py`

**Description**: Create abstract base class for layout containers.

**Acceptance Criteria**:
- [ ] `LayoutContainer` class with children list
- [ ] `add_child(widget, constraints)` method
- [ ] `remove_child(widget)` method
- [ ] `compute_layout()` abstract method
- [ ] `mark_layout_dirty()` method with upward propagation
- [ ] `_layout_dirty` flag to prevent redundant recomputes

**Evidence of Completion**: Unit test showing add/remove child and dirty propagation.

---

## Task 1.2: Size Hint Protocol

**File**: `engine/ui/widgets/base.py` (or appropriate base)

**Description**: Add size hint methods to widget base.

**Acceptance Criteria**:
- [ ] `get_min_size() -> tuple[float, float]` returns (0, 0) by default
- [ ] `get_preferred_size() -> tuple[float, float]` returns (width, height) by default
- [ ] `get_max_size() -> tuple[float, float]` returns (inf, inf) by default
- [ ] Widgets can override to provide specific hints
- [ ] Text widgets return hints based on text content

**Evidence of Completion**: Label widget returns preferred size based on text length.

---

## Task 1.3: FlexLayout Implementation

**File**: `engine/ui/layout/flex.py`

**Description**: Implement flexbox-style layout container.

**Acceptance Criteria**:
- [ ] `FlexDirection` enum: ROW, COLUMN
- [ ] `JustifyContent` enum: START, CENTER, END, SPACE_BETWEEN, SPACE_AROUND
- [ ] `AlignItems` enum: START, CENTER, END, STRETCH
- [ ] `gap` property for spacing between children
- [ ] `compute_layout()` distributes space according to rules
- [ ] Children bounds set after compute
- [ ] Children dirty flags set after bounds change

**Evidence of Completion**: Test with 3 buttons in ROW direction, SPACE_BETWEEN justify, produces correct x positions.

---

## Task 1.4: GridLayout Implementation

**File**: `engine/ui/layout/grid.py`

**Description**: Implement grid-style layout container.

**Acceptance Criteria**:
- [ ] `GridTrack` class: fixed size, fractional (fr), or auto
- [ ] `columns` and `rows` properties as list of GridTrack
- [ ] `gap` property as (row_gap, column_gap) tuple
- [ ] Children can span multiple cells via constraints
- [ ] `compute_layout()` resolves track sizes and places children
- [ ] Auto-placement for children without explicit row/column

**Evidence of Completion**: 2x2 grid with one item spanning 2 columns produces correct bounds.

---

## Task 1.5: StackLayout Implementation

**File**: `engine/ui/layout/stack.py`

**Description**: Simplified vertical/horizontal stack layout.

**Acceptance Criteria**:
- [ ] `StackDirection` enum: VERTICAL, HORIZONTAL
- [ ] `gap` property for spacing
- [ ] `alignment` property: START, CENTER, END
- [ ] Children stacked in order with gap between
- [ ] Cross-axis alignment applied

**Evidence of Completion**: Vertical stack of 3 labels with CENTER alignment produces centered x positions.

---

## Task 1.6: Layout Root Integration

**File**: `engine/ui/layout/root.py`

**Description**: Root layout container that sizes to window.

**Acceptance Criteria**:
- [ ] `LayoutRoot` class wraps top-level container
- [ ] `set_window_size(width, height)` triggers layout recompute
- [ ] Integrates with application resize events
- [ ] Single entry point for layout computation per frame

**Evidence of Completion**: Window resize triggers layout recompute, all child widgets updated.

---

## Task 1.7: Absolute Positioning Escape Hatch

**File**: `engine/ui/layout/container.py`

**Description**: Allow widgets to opt out of layout positioning.

**Acceptance Criteria**:
- [ ] `absolute` constraint flag skips layout positioning
- [ ] Widget with absolute constraint retains manual bounds
- [ ] Absolute widgets still part of children list (for input/render order)
- [ ] Layout recompute does not modify absolute widget bounds

**Evidence of Completion**: Widget with absolute=True in FlexLayout retains manual (x, y).

---

## Task 1.8: Layout Serialization

**File**: `engine/ui/layout/serialization.py`

**Description**: Add to_dict/from_dict for layout containers.

**Acceptance Criteria**:
- [ ] `FlexLayout.to_dict()` includes direction, justify, align, gap, children
- [ ] `GridLayout.to_dict()` includes columns, rows, gap, children placements
- [ ] `from_dict()` reconstructs layout hierarchy
- [ ] Integrates with existing widget serialization

**Evidence of Completion**: Round-trip serialization of FlexLayout with 3 children preserves structure.

---

## Task 1.9: Layout Module Exports

**File**: `engine/ui/layout/__init__.py`

**Description**: Export layout classes for public API.

**Acceptance Criteria**:
- [ ] Exports: LayoutContainer, FlexLayout, GridLayout, StackLayout, LayoutRoot
- [ ] Exports: FlexDirection, JustifyContent, AlignItems, GridTrack, StackDirection
- [ ] No internal implementation details exposed

**Evidence of Completion**: `from engine.ui.layout import FlexLayout` works.

---

## Summary

| Task | Effort | Priority |
|------|--------|----------|
| 1.1 Container Base | Medium | P0 |
| 1.2 Size Hints | Small | P0 |
| 1.3 FlexLayout | Large | P0 |
| 1.4 GridLayout | Large | P1 |
| 1.5 StackLayout | Small | P1 |
| 1.6 Layout Root | Medium | P0 |
| 1.7 Absolute Escape | Small | P1 |
| 1.8 Serialization | Medium | P2 |
| 1.9 Module Exports | Small | P0 |

**Total Tasks**: 9
**Critical Path**: 1.1 -> 1.2 -> 1.3 -> 1.6 -> 1.9
