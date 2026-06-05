# PHASE 4 TODO: Renderer Integration

## Prerequisites
- Phases 1-3 complete (layout, focus, input)
- Rust renderer backend operational
- Widget dirty-tracking functional

---

## Task 4.1: UIGeometry Data Class

**File**: `engine/ui/render/geometry.py`

**Description**: Define geometry data class for renderer consumption.

**Acceptance Criteria**:
- [ ] `UIGeometry` dataclass with: rect, uv, color, texture_id, nine_slice, z_order
- [ ] `NineSlice` dataclass with: left, right, top, bottom margins
- [ ] Geometry is immutable after creation
- [ ] Default values for optional fields (uv=None, texture_id=None)
- [ ] Type hints for all fields

**Evidence of Completion**: `UIGeometry(rect=(0,0,100,50), color=(1,1,1,1), z_order=0)` creates valid geometry.

---

## Task 4.2: Widget get_geometry Protocol

**File**: `engine/ui/widgets/base.py`

**Description**: Add geometry extraction method to widget base.

**Acceptance Criteria**:
- [ ] `get_geometry() -> list[UIGeometry]` method signature
- [ ] Default implementation returns empty list
- [ ] Interactive widgets return background + content geometries
- [ ] Text widgets return glyph geometries with UV coords
- [ ] Image widgets return texture geometries

**Evidence of Completion**: `button.get_geometry()` returns list with background rect and text glyphs.

---

## Task 4.3: Button Geometry Implementation

**File**: `engine/ui/widgets/input/button.py`

**Description**: Implement get_geometry for Button widget.

**Acceptance Criteria**:
- [ ] Returns background rect geometry with current visual state color
- [ ] Returns text geometry with font atlas texture
- [ ] Returns border geometry if styled
- [ ] Z-order correct for layering (background < border < text)
- [ ] Nine-slice if button uses nine-slice background

**Evidence of Completion**: Button in hovered state returns correct hovered color in geometry.

---

## Task 4.4: Text Geometry Implementation

**File**: `engine/ui/widgets/primitives/text.py`

**Description**: Implement get_geometry for Text widget.

**Acceptance Criteria**:
- [ ] Returns one geometry per glyph (or batched glyph run)
- [ ] UV coordinates from font atlas
- [ ] Color from text style
- [ ] Position computed from layout/alignment
- [ ] Rich text returns multiple styled runs

**Evidence of Completion**: "Hello" text returns geometry with correct UV for each glyph.

---

## Task 4.5: Image Geometry Implementation

**File**: `engine/ui/widgets/primitives/image.py`

**Description**: Implement get_geometry for Image widget.

**Acceptance Criteria**:
- [ ] Returns single geometry with texture_id
- [ ] UV coordinates for texture region
- [ ] Scale mode affects rect size
- [ ] Nine-slice supported for scalable images
- [ ] Tint color applied

**Evidence of Completion**: Nine-slice image returns geometry with nine_slice margins populated.

---

## Task 4.6: Game Widget Geometries

**Files**: `engine/ui/widgets/game/*.py`

**Description**: Implement get_geometry for game widgets.

**Acceptance Criteria**:
- [ ] HealthBar returns bar background, fill, damage preview, shield geometries
- [ ] Minimap returns map background, visible region, markers
- [ ] InventorySlot returns slot background, item icon, rarity border, count text
- [ ] Tooltip returns background, border, content geometries
- [ ] DamageNumbers returns floating text geometries with positions

**Evidence of Completion**: HealthBar at 50% returns fill rect at half width.

---

## Task 4.7: Geometry Collector

**File**: `engine/ui/render/collector.py`

**Description**: Collect geometry from widget tree.

**Acceptance Criteria**:
- [ ] `GeometryCollector` class
- [ ] `collect(root) -> list[UIGeometry]` traverses widget tree
- [ ] Only collects from dirty widgets (optimization)
- [ ] Clears dirty flag after collection
- [ ] Respects visibility (invisible widgets skipped)

**Evidence of Completion**: Tree with 2 dirty widgets out of 10, returns geometry from only those 2.

---

## Task 4.8: Geometry Batching

**File**: `engine/ui/render/batch.py`

**Description**: Batch and sort geometries for efficient rendering.

**Acceptance Criteria**:
- [ ] `batch_geometries(geometries) -> list[RenderBatch]`
- [ ] Sort by z_order for correct layering
- [ ] Group by texture_id to minimize texture swaps
- [ ] Merge adjacent rects with same texture/color where possible
- [ ] Return batches ready for single draw call each

**Evidence of Completion**: 100 button geometries with same texture batched into 1 RenderBatch.

---

## Task 4.9: Geometry Serialization

**File**: `engine/ui/render/serialize.py`

**Description**: Serialize geometries for FFI transfer.

**Acceptance Criteria**:
- [ ] `pack_geometries(geometries) -> bytes` creates compact binary format
- [ ] Format includes header with count, version
- [ ] Each geometry packed with fixed-size fields
- [ ] Handles None values for optional fields
- [ ] Endianness explicit (little-endian)

**Evidence of Completion**: 10 geometries pack to predictable byte size.

---

## Task 4.11: Animation Controller

**File**: `engine/ui/animation/controller.py`

**Description**: Centralize animation updates for render sync.

**Acceptance Criteria**:
- [ ] `AnimationController` manages active animations
- [ ] `update(delta_time)` steps all animations
- [ ] Sets `_dirty = True` on animated widgets
- [ ] Called by render loop before geometry collection
- [ ] Removes completed animations

**Evidence of Completion**: Fading button animation updates opacity and sets dirty each frame.

---

## Task 4.12: Accessibility Tree Builder

**File**: `engine/ui/accessibility/tree.py`

**Description**: Build accessibility tree for platform integration.

**Acceptance Criteria**:
- [ ] `AccessNode` dataclass: role, name, bounds, children, widget_id
- [ ] `build_accessibility_tree(root) -> AccessNode` traverses widgets
- [ ] Uses `get_accessible_role()` and `get_accessible_text()` from widgets
- [ ] Filters non-accessible widgets (decorative elements)
- [ ] Tree exposed to Rust renderer for platform API

**Evidence of Completion**: Tree built from ProgressBar returns role="progressbar", name="Progress: 50%".

---

## Task 4.13: Render Module Exports

**File**: `engine/ui/render/__init__.py`

**Description**: Export render classes for public API.

**Acceptance Criteria**:
- [ ] Exports: UIGeometry, NineSlice, GeometryCollector, batch_geometries, pack_geometries
- [ ] Exports: AnimationController (if in render module)
- [ ] No internal implementation details exposed

**Evidence of Completion**: `from engine.ui.render import GeometryCollector` works.

---

## Summary

| Task | Effort | Priority |
|------|--------|----------|
| 4.1 UIGeometry Class | Small | P0 |
| 4.2 get_geometry Protocol | Small | P0 |
| 4.3 Button Geometry | Medium | P0 |
| 4.4 Text Geometry | Large | P0 |
| 4.5 Image Geometry | Medium | P0 |
| 4.6 Game Widget Geometries | Large | P1 |
| 4.7 Geometry Collector | Medium | P0 |
| 4.8 Geometry Batching | Medium | P1 |
| 4.9 Geometry Serialization | Medium | P0 |
| 4.10 Rust FFI Bridge | Large | P0 |
| 4.11 Animation Controller | Medium | P1 |
| 4.12 Accessibility Tree | Medium | P2 |
| 4.13 Module Exports | Small | P0 |

**Total Tasks**: 13
**Critical Path**: 4.1 -> 4.2 -> 4.3/4.4/4.5 -> 4.7 -> 4.9 -> 4.10 -> 4.13
