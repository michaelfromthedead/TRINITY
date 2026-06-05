# PHASE 4 ARCHITECTURE: Renderer Integration

## Problem Statement

Widgets prepare geometry data (rectangles, UV coordinates, positions) but there is no actual GPU rendering code. The widgets produce data through methods like `get_fill_rect()` and `get_thumb_position()`, but no system consumes this data and renders it via the Rust backend.

## Architectural Decision

### Geometry Protocol

Define a standardized geometry format that widgets produce and the Rust renderer consumes:

```python
@dataclass
class UIGeometry:
    rect: tuple[float, float, float, float]  # x, y, width, height
    uv: tuple[float, float, float, float] | None  # u0, v0, u1, v1
    color: tuple[float, float, float, float]  # r, g, b, a
    texture_id: int | None
    nine_slice: NineSlice | None
    z_order: int
```

### Geometry Collector

Central collector gathers geometry from all dirty widgets:

```python
class GeometryCollector:
    def collect(self, root: Widget) -> list[UIGeometry]:
        geometries = []
        for widget in self._traverse_dirty(root):
            geometries.extend(widget.get_geometry())
            widget._dirty = False
        return geometries
```

### Widget Geometry Methods

Widgets implement a standard interface:
```python
def get_geometry(self) -> list[UIGeometry]:
    # Button example
    return [
        UIGeometry(
            rect=self.get_fill_rect(),
            color=self._current_background_color,
            texture_id=None,
            z_order=self._z_order
        ),
        UIGeometry(
            rect=self._text.get_bounds(),
            color=self._current_text_color,
            texture_id=self._font_atlas_id,
            uv=self._text.get_uv(),
            z_order=self._z_order + 1
        )
    ]
```

### Rust FFI Bridge

Python geometry data crosses to Rust via FFI:

```python
# Python side
def submit_frame(geometries: list[UIGeometry]) -> None:
    buffer = pack_geometries(geometries)
    rust_renderer.submit_ui_frame(buffer)
```

```rust
// Rust side
#[pyfunction]
fn submit_ui_frame(buffer: &[u8]) {
    let geometries = deserialize_geometries(buffer);
    renderer.queue_ui_draw(geometries);
}
```

### Batch Optimization

Geometries are batched for efficient rendering:
1. Sort by texture_id to minimize texture swaps
2. Sort by z_order for correct layering
3. Merge adjacent rects with same texture/color
4. Use instanced rendering for repeated elements

### Animation Synchronization

Widget animations must sync with render loop:
```python
class AnimationController:
    def update(self, delta_time: float) -> None:
        for animation in self._active_animations:
            animation.step(delta_time)
            animation.target_widget._dirty = True
```

Render loop calls `animation_controller.update(dt)` before geometry collection.

## Component Diagram

```
+----------------+     +-----------------+     +----------------+
|    Widgets     | --> | GeometryCollect | --> |  Rust Renderer |
| (dirty-track)  |     |  (batch/sort)   |     |  (GPU submit)  |
+----------------+     +-----------------+     +----------------+
        ^                                              |
        |                                              |
+-------+--------+                              +------v------+
| AnimController |                              |   Display   |
| (sync timing)  |                              +-------------+
+----------------+
```

## Accessibility Tree

Platform accessibility requires exposing widget tree:
```python
class AccessibilityTree:
    def build(self, root: Widget) -> AccessNode:
        return AccessNode(
            role=root.get_accessible_role(),
            name=root.get_accessible_text(),
            bounds=root.get_bounds(),
            children=[self.build(c) for c in root.children]
        )
```

Rust renderer exposes this tree to platform accessibility APIs (MSAA, UI Automation, ATK).

## Risks

1. **FFI overhead**: Frequent Python-to-Rust calls may cause latency. Mitigation: batch entire frame geometry in one call.

2. **Data format mismatch**: Python and Rust must agree on geometry format. Mitigation: versioned schema, validation.

3. **Animation jank**: Timing mismatch between Python animation and Rust render. Mitigation: Rust-side interpolation for smooth animation.

4. **Memory pressure**: Large UI may produce many geometries. Mitigation: dirty-tracking limits updates, geometry pooling.
