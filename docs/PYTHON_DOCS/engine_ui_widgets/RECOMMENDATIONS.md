# RECOMMENDATIONS: engine/ui/widgets

## Rust Bridge Requirements

### High Priority

| Requirement | Description | Affected Files |
|-------------|-------------|----------------|
| **Text Rendering Protocol** | Define geometry format for text (glyphs, positions, colors) that widgets produce and Rust renderer consumes | text.py, label.py, button.py, text_input.py |
| **Rect Batch Format** | Standardize filled/bordered rectangle commands for efficient GPU batching | All widgets (bounds property) |
| **Nine-Slice Protocol** | Define UV coordinates and slice dimensions format for nine-slice image rendering | primitives/image.py |
| **Circular Progress Shader** | Arc rendering for circular progress bars requires shader support | display/progress_bar.py |

### Medium Priority

| Requirement | Description | Affected Files |
|-------------|-------------|----------------|
| **Icon Atlas Format** | Define atlas JSON/binary format matching IconAtlasManager | display/icon.py |
| **Texture Handle Protocol** | How Python references textures loaded by Rust | primitives/image.py, game/minimap.py |
| **Animation Frame Callback** | Rust-side animation tick that calls Python update(dt) | All animated widgets |

### Low Priority

| Requirement | Description | Affected Files |
|-------------|-------------|----------------|
| **Clipping/Scissor Rect** | For scroll views and dropdown lists (not yet implemented) | Future: ScrollView |
| **Z-order Protocol** | Tooltip layering above other UI | game/tooltip.py |

## Integration Strategy

### Phase 1: Define Render Command Protocol (Python-side)
```python
# Proposed interface in engine/ui/rendering/commands.py
@dataclass
class UIRenderCommand:
    """Base for all UI render commands."""
    z_index: int = 0

@dataclass
class FillRectCommand(UIRenderCommand):
    x: float
    y: float
    width: float
    height: float
    color: str  # hex RGBA
    corner_radius: float = 0.0

@dataclass  
class DrawTextCommand(UIRenderCommand):
    x: float
    y: float
    text: str
    font_size: float
    color: str
    alignment: str = "left"
```

### Phase 2: Widget Command Generation
Add `get_render_commands() -> list[UIRenderCommand]` method to each widget. This makes the implicit geometry output explicit and testable.

### Phase 3: Rust Consumer
In renderer-backend, add a UI pass that:
1. Receives serialized commands via PyO3
2. Batches rectangles by color/texture
3. Renders text via glyph atlas
4. Applies clipping rects

### Phase 4: Animation Loop Integration
```python
# engine/ui/animation.py
class UIAnimationLoop:
    def __init__(self, root_widget):
        self.root = root_widget
        
    def tick(self, dt: float):
        """Called from Rust frame callback."""
        self._update_recursive(self.root, dt)
```

## Testing Strategy

### Unit Tests Required

| Test Area | Priority | Files to Test |
|-----------|----------|---------------|
| State machine transitions | HIGH | button.py, checkbox.py |
| Coordinate transforms | HIGH | minimap.py (world_to_map, map_to_world) |
| Text selection | HIGH | text_input.py |
| Serialization round-trip | MEDIUM | All widgets with to_dict/from_dict |
| Event subscription/unsubscribe | MEDIUM | All widgets with callbacks |
| Bounds checking | MEDIUM | All widgets (negative dimensions) |
| Accessibility output | LOW | label.py, progress_bar.py |

### Integration Tests Required

| Test Area | Priority | Description |
|-----------|----------|-------------|
| Render command generation | HIGH | Verify widgets produce correct commands |
| Input event flow | HIGH | Mouse/keyboard through widget hierarchy |
| Focus navigation | MEDIUM | Tab order through focusable widgets |
| Dirty tracking | MEDIUM | Only dirty widgets produce commands |

### Property-Based Tests (Hypothesis)

| Property | Widgets |
|----------|---------|
| `map_to_world(world_to_map(x,y)) == (x,y)` | minimap.py |
| `from_dict(to_dict(widget)).state == widget.state` | All serializable |
| `selection.normalized.start <= selection.normalized.end` | text_input.py |

## Risk Assessment

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Layout system complexity | HIGH | HIGH | Start with simple flexbox subset |
| Text rendering performance | MEDIUM | HIGH | Font atlas caching in Rust |
| Animation timing jitter | MEDIUM | MEDIUM | Fixed timestep with interpolation |
| Focus order edge cases | LOW | LOW | Test with complex widget trees |

### Integration Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| PyO3 GIL contention | MEDIUM | HIGH | Batch commands, minimize crossings |
| Serialization versioning | MEDIUM | MEDIUM | Add version field to to_dict |
| Clipboard platform differences | LOW | MEDIUM | Abstract via Protocol |

### Schedule Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Layout engine scope creep | HIGH | HIGH | Define MVP feature set upfront |
| Testing coverage gaps | MEDIUM | MEDIUM | Add tests before new features |
| Documentation lag | HIGH | LOW | Auto-generate from docstrings |

## Recommended Implementation Order

1. **Week 1-2**: Render command protocol and widget command generation
2. **Week 2-3**: Rust UI pass consuming commands
3. **Week 3-4**: Focus manager and input router
4. **Week 4-6**: Layout engine (flexbox subset)
5. **Week 6-7**: Animation loop integration
6. **Week 7-8**: Testing and documentation

## Dependencies on Other Subsystems

| Subsystem | Dependency Type | Status |
|-----------|-----------------|--------|
| renderer-backend | Consumer of geometry | EXISTS |
| engine/ui/layout | Provider of positioning | NOT IMPLEMENTED |
| engine/ui/rendering | Render command protocol | NOT IMPLEMENTED |
| engine/input | Input event source | UNKNOWN |
| trinity/descriptors | Not needed for widgets | N/A |
