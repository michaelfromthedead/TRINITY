# RECOMMENDATIONS: engine/ui (layout, screens, styling, text)

**Date**: 2026-05-22

---

## Rust Bridge Requirements

### High Priority

| Component | Rationale | Effort | Impact |
|-----------|-----------|--------|--------|
| Text Shaping | Complex scripts (Arabic, Devanagari) need harfbuzz; Rust bindings more robust than Python ctypes | Medium | High |
| Font Atlas Generation | GPU compute via renderer-backend would be 10-100x faster for large character sets | Medium | Medium |

### Medium Priority

| Component | Rationale | Effort | Impact |
|-----------|-----------|--------|--------|
| Color Space Conversion | Wide-gamut (Display P3, Rec.2020) needs precision; Rust numeric stability better | Low | Medium |
| Layout Dirty Propagation | For UI trees >1000 widgets, Rust could accelerate dirty flag propagation | High | Low |

### Low Priority

| Component | Rationale | Effort | Impact |
|-----------|-----------|--------|--------|
| Easing Functions | Already pure math; Rust would add FFI overhead without benefit | Low | None |
| Screen Stack | Logic is simple; Python is sufficient | Low | None |
| Theme System | Declarative config; no computation to optimize | Low | None |

---

## Integration Strategy

### Phase 1: Keep Python UI (No Changes)

The UI subsystem is production-ready and performant. Do not introduce Rust dependencies unless:
1. Performance profiling identifies a bottleneck
2. A specific feature requires native code (e.g., harfbuzz for complex scripts)

### Phase 2: Text Rendering Bridge (If Needed)

If CJK/RTL text performance is insufficient:
1. Create `renderer-backend/src/text_shaping.rs` with harfbuzz-rs bindings
2. Expose via PyO3: `shape_text(text: str, font: str, size: f32) -> Vec<GlyphInfo>`
3. Replace `TextShaper._shape_run()` with Rust call

### Phase 3: GPU Font Atlas (If Needed)

If font atlas generation is slow:
1. Add compute shader to renderer-backend for distance field generation
2. Python calls Rust, which submits GPU work
3. Atlas texture returned as GPU handle

---

## Testing Strategy

### Unit Tests Required

| Module | Test Coverage Needed |
|--------|---------------------|
| Layout | Grid track sizing, flex distribution, responsive breakpoints |
| Screens | Screen stack push/pop, LRU cache eviction, transition timing |
| Styling | Blend modes, contrast ratio, theme inheritance |
| Text | Plural rules, parameter substitution, RTL detection |

### Integration Tests Required

| Scenario | Description |
|----------|-------------|
| Layout + Styling | Grid with themed children, style cascade |
| Screens + Transitions | Push/pop with fade/slide, modal overlays |
| Text + Localization | Rich text with i18n, plural forms |
| Full Pipeline | Screen with layout, themed widgets, localized text |

### Performance Tests Required

| Benchmark | Target |
|-----------|--------|
| Layout 100 widgets | < 1ms |
| Layout 1000 widgets | < 10ms |
| Theme switch | < 5ms |
| Font atlas 256 glyphs | < 100ms |

---

## Risk Assessment

### Low Risk
- **Layout algorithms**: Well-tested patterns (CSS Grid/Flex), predictable behavior
- **Screen navigation**: Simple stack operations, clear lifecycle
- **Theme system**: Declarative, no complex logic

### Medium Risk
- **Text shaping**: Complex scripts may expose edge cases not covered by current implementation
- **IME handling**: Platform-specific behavior varies; testing across OSes needed
- **Font loading**: Relies on file I/O; error handling for missing fonts

### High Risk
- **None identified**: The subsystem is mature and follows established patterns

---

## Action Items

1. **Verify Tests Exist**: Check `tests/` for UI module coverage
2. **Profile Text Rendering**: Measure `TextRenderer` performance with CJK text
3. **Document Widget Authoring**: Create guide for custom widget development
4. **Consider Async File I/O**: For font/string loading, add async variants

---

## Conclusion

The engine/ui subsystem is a solid GRANDPHASE1 implementation requiring no immediate Rust bridging. The Python code is clean, well-structured, and performant for typical game UI workloads. Rust integration should only be pursued if specific performance or feature requirements emerge.
