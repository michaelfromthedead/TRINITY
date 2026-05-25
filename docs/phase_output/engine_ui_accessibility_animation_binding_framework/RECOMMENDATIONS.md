# RECOMMENDATIONS: engine/ui/accessibility, animation, binding, framework

## Rust Bridge Requirements

### High Priority
None. This subsystem is Python-native and does not require Rust bridging.

### Medium Priority
| Component | Rationale | Effort |
|-----------|-----------|--------|
| Easing Functions | Pure math, called frequently in animation loops | Low |
| WCAG Contrast Ratio | Called during theme application, hot path for accessibility checks | Low |
| Color Transform Matrices | Matrix-vector multiplication for colorblind simulation | Low |

### Low Priority
| Component | Rationale | Effort |
|-----------|-----------|--------|
| Property Path Parsing | Only parsed once per binding, cached | Very Low |
| Widget Hit Testing | Recursive tree traversal, Python is adequate | Medium |

## Integration Strategy

### Phase 1: Standalone Use (Current State)
The UI subsystem is fully functional as pure Python. No integration work required for basic use.

### Phase 2: Platform Integration
1. **Input Layer**: Connect to SDL2/GLFW for keyboard/mouse events
   - Create MouseEvent/KeyboardEvent from platform events
   - Route through EventDispatcher to widget tree

2. **System Accessibility**:
   - Windows: SystemParametersInfo for high contrast detection
   - macOS: NSWorkspace.accessibilityDisplayShouldIncreaseContrast
   - Linux: GTK settings / GNOME accessibility

3. **Screen Reader Bridges**:
   - Windows: UI Automation API
   - macOS: Accessibility API (VoiceOver)
   - Linux: AT-SPI (Orca)

### Phase 3: Renderer Integration
1. **Graphics Backend**: Implement on_render(context) for widgets
   - Vulkan: Vertex buffer generation for UI primitives
   - OpenGL: Immediate mode or VBO rendering
   - Software: Direct pixel manipulation

2. **Shared Types with Rust Renderer** (if GAPSET_3_BRIDGE proceeds):
   - Define Point, Size, Rect in Rust omega crate
   - Export via PyO3 for Python use
   - Ensures zero-copy between Python UI and Rust renderer

## Testing Strategy

### Unit Tests
| Area | Test Type | Coverage Target |
|------|-----------|-----------------|
| Easing Functions | Property-based (hypothesis) | All functions return [0,1] for input [0,1] |
| Contrast Ratio | WCAG test vectors | Known color pairs from spec |
| Colorblind Simulation | Reference images | Compare against ImageMagick filters |
| Property Paths | Parametric | Nested, indexed, edge cases |
| Event Dispatch | State machine | All phase combinations |

### Integration Tests
| Area | Test Type | Coverage Target |
|------|-----------|-----------------|
| Animation Timing | Mock clock | Verify progress at specific times |
| Data Binding | Observable model | Source/target sync correctness |
| Focus Management | Widget tree | Tab order traversal |
| Hit Testing | Geometric | Overlapping/clipped widgets |

### Accessibility Compliance Tests
| Standard | Automated Check |
|----------|-----------------|
| WCAG 2.1 AA Contrast | All text/background combos >= 4.5:1 |
| WCAG 2.5.5 Touch Targets | All interactive widgets >= 44x44px |
| ARIA Roles | All widgets have appropriate roles |
| Keyboard Accessibility | All functions reachable via keyboard |

## Risk Assessment

### Low Risk
| Risk | Mitigation |
|------|------------|
| Python Performance | UI is I/O-bound; benchmarks show adequate speed |
| Memory Leaks | Weak references in bindings; explicit cleanup in unmount |
| Circular Dependencies | Clean import structure; no circular imports detected |

### Medium Risk
| Risk | Mitigation |
|------|------------|
| Platform API Differences | Abstract platform layer; per-OS implementations |
| Screen Reader Compatibility | Test with NVDA, JAWS, VoiceOver, Orca |
| Animation Timing Drift | Use monotonic clock; frame-independent updates |

### Not Applicable
| Risk | Why Not Applicable |
|------|-------------------|
| Thread Safety | UI runs on main thread only |
| Network Dependencies | No network calls in UI subsystem |
| Data Persistence | UI is stateless; state lives in model layer |

## Implementation Priorities

1. **Immediate**: Add platform input integration (SDL2 event loop)
2. **Short-term**: Implement platform accessibility detection hooks
3. **Medium-term**: Connect to graphics renderer for actual rendering
4. **Long-term**: Consider Rust acceleration if animation profiling shows need

## Dependencies for Full Functionality

### Required
- Python 3.11+ (match expressions, type annotations)
- asyncio (async validation)
- dataclasses, enum, typing (type system)

### Optional (for platform integration)
- pysdl2 or glfw (input events)
- pywin32 (Windows accessibility APIs)
- pyobjc (macOS accessibility APIs)
- pgi/PyGObject (Linux accessibility APIs)

### Optional (for Rust acceleration)
- PyO3 bindings to omega crate
- maturin for building Python wheels
