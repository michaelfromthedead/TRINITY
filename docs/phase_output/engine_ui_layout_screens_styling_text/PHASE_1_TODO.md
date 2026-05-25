# PHASE 1 TODO: Layout Module

---

## Grid Layout Tasks

### T-1.1: Verify Grid Track Sizing Algorithm

**File**: `engine/ui/layout/grid.py`

**Description**: Ensure `_calculate_track_sizes()` correctly handles all track size types.

**Acceptance Criteria**:
- [ ] Fixed pixel sizes resolve correctly
- [ ] `auto` tracks size to content (min-content or max-content based on context)
- [ ] `minmax(min, max)` clamps correctly
- [ ] `fr` units distribute remaining space proportionally
- [ ] Mixed track types work together (e.g., `100px 1fr auto 2fr`)

**Test Cases**:
- Grid with 3 columns: `100px 1fr 2fr` in 400px container -> 100px, 100px, 200px
- Grid with minmax: `minmax(50px, 1fr) 1fr` in 200px -> 100px, 100px (min not hit)
- Grid with auto: `auto 1fr` where content is 80px in 200px -> 80px, 120px

---

### T-1.2: Verify Grid Item Placement with Spans

**File**: `engine/ui/layout/grid.py`

**Description**: Ensure items spanning multiple rows/columns place correctly.

**Acceptance Criteria**:
- [ ] `column_span` extends item across multiple columns
- [ ] `row_span` extends item across multiple rows
- [ ] Combined row+column spans work
- [ ] Auto-placement respects existing spans
- [ ] Gap is not applied within spanned area

---

### T-1.3: Verify Gap Handling

**File**: `engine/ui/layout/grid.py`

**Description**: Ensure `row_gap` and `column_gap` apply correctly.

**Acceptance Criteria**:
- [ ] Gap appears between tracks, not at edges
- [ ] Gap does not affect track size calculation (space is subtracted first)
- [ ] Spanning items account for gaps within their span

---

## Flexbox Layout Tasks

### T-1.4: Verify Flex Grow Distribution

**File**: `engine/ui/layout/flex.py`

**Description**: Ensure positive free space distributes per grow factors.

**Acceptance Criteria**:
- [ ] Items with `flex_grow: 0` do not grow
- [ ] Items with equal grow factors share space equally
- [ ] Items with different grow factors share proportionally (1:2 ratio)
- [ ] Items clamped by `max_width`/`max_height` are frozen

---

### T-1.5: Verify Flex Shrink Distribution

**File**: `engine/ui/layout/flex.py`

**Description**: Ensure negative free space distributes per shrink factors.

**Acceptance Criteria**:
- [ ] Items with `flex_shrink: 0` do not shrink
- [ ] Shrink is proportional to both shrink factor AND base size
- [ ] Items clamped by `min_width`/`min_height` are frozen
- [ ] Remaining shrink redistributes to non-frozen items

---

### T-1.6: Verify Multi-Line Wrapping

**File**: `engine/ui/layout/flex.py`

**Description**: Ensure `flex-wrap: wrap` creates correct lines.

**Acceptance Criteria**:
- [ ] Items wrap to new line when main axis overflows
- [ ] `wrap-reverse` reverses line order
- [ ] `align-content` distributes lines on cross axis
- [ ] Each line independently calculates flex distribution

---

### T-1.7: Verify Justify Content Modes

**File**: `engine/ui/layout/flex.py`

**Description**: Ensure all 6 justify modes work correctly.

**Acceptance Criteria**:
- [ ] `flex-start` — Items packed at start
- [ ] `flex-end` — Items packed at end
- [ ] `center` — Items centered
- [ ] `space-between` — First at start, last at end, equal space between
- [ ] `space-around` — Equal space around each item (half at edges)
- [ ] `space-evenly` — Equal space between all items and edges

---

### T-1.8: Verify Align Items Modes

**File**: `engine/ui/layout/flex.py`

**Description**: Ensure cross-axis alignment works correctly.

**Acceptance Criteria**:
- [ ] `stretch` — Items stretch to fill cross axis
- [ ] `flex-start` — Items aligned to cross start
- [ ] `flex-end` — Items aligned to cross end
- [ ] `center` — Items centered on cross axis
- [ ] `baseline` — Items aligned by text baseline

---

## HBox/VBox Layout Tasks

### T-1.9: Verify HBox Flex Distribution

**File**: `engine/ui/layout/hbox.py`

**Description**: Ensure HBox distributes space correctly along horizontal axis.

**Acceptance Criteria**:
- [ ] Flex items grow/shrink horizontally
- [ ] Spacing between items applied correctly
- [ ] Alignment on vertical axis works

---

### T-1.10: Verify VBox Flex Distribution

**File**: `engine/ui/layout/vbox.py`

**Description**: Ensure VBox distributes space correctly along vertical axis.

**Acceptance Criteria**:
- [ ] Flex items grow/shrink vertically
- [ ] Spacing between items applied correctly
- [ ] Alignment on horizontal axis works

---

## Responsive Layout Tasks

### T-1.11: Verify Breakpoint Matching

**File**: `engine/ui/layout/responsive.py`

**Description**: Ensure `_find_active_breakpoint()` returns correct breakpoint.

**Acceptance Criteria**:
- [ ] Width < mobile threshold returns mobile
- [ ] Width between mobile and tablet returns tablet
- [ ] Width between tablet and desktop returns desktop
- [ ] Width >= widescreen threshold returns widescreen
- [ ] Exact boundary values handled correctly (no off-by-one)

---

### T-1.12: Verify Responsive Value Resolution

**File**: `engine/ui/layout/responsive.py`

**Description**: Ensure `ResponsiveValue.get_current_value()` returns correct value.

**Acceptance Criteria**:
- [ ] Returns value for current breakpoint
- [ ] Falls back to smaller breakpoint if current not defined
- [ ] Interpolates between values if configured
- [ ] Works with any value type (number, string, style)

---

### T-1.13: Verify Safe Area Insets

**File**: `engine/ui/layout/responsive.py`

**Description**: Ensure notched device safe areas are respected.

**Acceptance Criteria**:
- [ ] `SafeAreaInsets` provides top, bottom, left, right values
- [ ] Container queries insets from platform
- [ ] Content is inset correctly (not under notch)

---

## Canvas Layout Tasks

### T-1.14: Verify Anchor Point Calculation

**File**: `engine/ui/layout/canvas.py`

**Description**: Ensure `_apply_anchor()` positions items correctly.

**Acceptance Criteria**:
- [ ] Anchor (0, 0) positions at top-left of parent
- [ ] Anchor (0.5, 0.5) positions at center of parent
- [ ] Anchor (1, 1) positions at bottom-right of parent
- [ ] Fractional anchors interpolate correctly

---

### T-1.15: Verify Pivot Point Calculation

**File**: `engine/ui/layout/canvas.py`

**Description**: Ensure `_apply_pivot()` offsets items from anchor.

**Acceptance Criteria**:
- [ ] Pivot (0, 0) puts item's top-left at anchor
- [ ] Pivot (0.5, 0.5) centers item on anchor
- [ ] Pivot (1, 1) puts item's bottom-right at anchor

---

### T-1.16: Verify Z-Order Sorting

**File**: `engine/ui/layout/canvas.py`

**Description**: Ensure `_sort_by_z_index()` orders items correctly.

**Acceptance Criteria**:
- [ ] Higher z-index items render on top
- [ ] Equal z-index preserves insertion order
- [ ] Negative z-index items render below default (0)

---

### T-1.17: Verify Hit Testing

**File**: `engine/ui/layout/canvas.py`

**Description**: Ensure `hit_test()` returns correct item at point.

**Acceptance Criteria**:
- [ ] Returns topmost item (highest z-index) at point
- [ ] Returns `None` if no item at point
- [ ] Respects item bounds correctly
- [ ] Traverses in z-order (front to back)
