# RECOMMENDATIONS: engine/rendering/demoscene

## Rust Bridge Requirements

### High Priority: None

This subsystem is complete Python code with no blocking dependencies on Rust. No high-priority bridging required.

### Medium Priority

| Requirement | Rationale | Effort |
|-------------|-----------|--------|
| **SceneGraph PyO3 Type** | Enable zero-copy sharing with renderer-backend | 2-3 days |
| **AST Validation in Rust** | Catch malformed scenes before WGSL generation | 1-2 days |
| **Type Registry Integration** | Register AST node types for cross-language validation | 1 day |

### Low Priority

| Requirement | Rationale | Effort |
|-------------|-----------|--------|
| **Rust WGSL Emitter** | Compile-time perf improvement (not runtime-critical) | 1 week |
| **Scene Serialization** | Binary format for faster scene loading | 2-3 days |
| **GPU Resource Handles** | Direct buffer/texture references from Python | 3-4 days |

## Integration Strategy

### Phase 1: Keep Python Frontend (Recommended for GRANDPHASE2)

```
Python DSL (lambdas, dicts)
         |
         v
    AstBuilder.walk()
         |
         v
    SceneGraph (frozen dataclasses)
         |
         v
    WgslCodeGen.emit()
         |
         v
    WGSL String -> renderer-backend
```

**Action**: No changes required. Pass generated WGSL to Rust renderer.

### Phase 2: Optional Rust Backend (Future Enhancement)

```
Python DSL
         |
         v
    AstBuilder.walk() -> SceneGraph
         |
         v
    PyO3 boundary (zero-copy)
         |
         v
    Rust AST validation
         |
         v
    Rust WGSL emission (naga or custom)
         |
         v
    renderer-backend pipeline
```

**When**: Only if WGSL generation becomes a bottleneck (unlikely for typical scene sizes).

### Integration with Existing Crates

| Crate | Integration Point | Notes |
|-------|-------------------|-------|
| renderer-backend | WGSL consumer | Accept generated shader strings |
| type_registry | Optional | Could validate AST node types |
| frame_graph | Optional | Scene compilation as a graph node |
| component_store | Not applicable | Demoscene is stateless generation |

## Testing Strategy

### Unit Tests (Python)

| Test Category | Coverage Focus |
|---------------|----------------|
| AST Node Construction | All 17+ node types instantiate correctly |
| Tree Traversal | walk(), children(), pretty() methods |
| Lambda Introspection | Parse various lambda patterns |
| WGSL Emission | Generated code matches expected templates |

### Integration Tests

| Test Category | Coverage Focus |
|---------------|----------------|
| End-to-End Pipeline | Python DSL -> AST -> WGSL -> validation |
| Material System | PBR properties propagate to shader struct |
| CSG Operations | Union/intersection/subtraction combinations |
| Domain Operations | All 7 operations with compensation |

### Validation Tests

| Test Category | Coverage Focus |
|---------------|----------------|
| WGSL Syntax | Generated code parses with naga |
| Math Correctness | SDF functions produce correct distances |
| Compensation Factors | Non-isometric ops have correct adjustments |

### Recommended Test Framework

```python
# Example test structure
class TestWgslCodegen:
    def test_sphere_emission(self):
        scene = SceneGraph(primitives=(SphereNode(radius=1.0),))
        wgsl = WgslCodeGen(scene).emit()
        assert "fn sdSphere" in wgsl
        assert "length(p) - r" in wgsl

    def test_kifs_compensation(self):
        scene = SceneGraph(
            primitives=(SphereNode(radius=1.0),),
            pipeline=(KifsNode(folds=6),)
        )
        wgsl = WgslCodeGen(scene).emit()
        assert "domain_kifs_compensation" in wgsl
```

## Risk Assessment

### Low Risk

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| WGSL syntax errors | Low | Medium | Validate with naga before shader compilation |
| Lambda introspection failures | Low | Low | Fallback to dict-based scene definition |
| Performance bottleneck | Very Low | Low | Generation is compile-time, not runtime |

### No High Risks Identified

The demoscene subsystem is:
- Self-contained (no external dependencies beyond Python stdlib)
- Well-architected (clean separation of concerns)
- Mathematically correct (follows established conventions)
- Feature-complete (all primitives, operations, materials implemented)

### Recommendations Summary

1. **No immediate action required** - Subsystem is production-ready
2. **Keep Python frontend** - Excellent UX for scene authoring
3. **Pass WGSL to Rust** - Simple string handoff, no complex bridging
4. **Optional Rust backend** - Only pursue if compile-time perf becomes an issue
5. **Add validation tests** - Ensure generated WGSL parses correctly

## GRANDPHASE2 Priority

**Priority**: LOW

This subsystem is complete and does not block other work. Focus GRANDPHASE2 efforts on incomplete subsystems with actual gaps or stub implementations.
