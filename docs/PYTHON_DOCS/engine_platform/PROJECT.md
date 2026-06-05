# PROJECT: Engine Platform Layer

## Scope

The engine/platform/ layer provides cross-platform abstractions for hardware and OS services. This project covers seven subsystems discovered in the investigation phase:

| Subsystem | Status | Lines | Classification |
|-----------|--------|-------|----------------|
| audio/ | Production-Ready | 1,266 | REAL |
| gpu/ | Skeleton | 98 | STUB |
| input/ | Production-Ready | 1,698 | REAL |
| os/ | Production-Ready | 2,067 | REAL (Linux-optimized) |
| rhi/ | Abstract Layer | 1,818 | REAL (Null Backend) |
| services/ | Partial | 265 | MIXED |
| window/ | Headless Backend | 899 | REAL |

Total: 8,111 lines across 7 subsystems (44 files)

## Goals

### Primary Goals

1. Complete the platform abstraction layer for cross-platform game engine deployment
2. Provide testable null/headless backends for CI/CD environments
3. Enable backend extensibility for platform-specific implementations
4. Maintain consistent API patterns across all subsystems

### Secondary Goals

1. Integrate with Rust renderer-backend crate for GPU operations
2. Support mobile platforms (iOS, Android) where applicable
3. Support console platforms (PS5, Xbox, Switch) via future SDK integration

## Constraints

### Technical Constraints

1. **Python 3.13 Required** - Project targets statically-linked Python 3.13 interpreter
2. **No Native Extensions** - All subsystems use pure Python (ctypes for dynamic library loading)
3. **Linux-Optimized** - OS subsystem relies on /proc and sysfs; fallbacks for other platforms
4. **Python GIL** - Atomics are lock-based, not true lock-free
5. **Python mmap** - No mprotect/madvise exposure; memory protection limited

### Architectural Constraints

1. **Backend Registry Pattern** - All subsystems must support pluggable backends
2. **Null Backend First** - Every subsystem must have a testable null/headless implementation
3. **ABC + Concrete** - Abstract base classes define contracts; concrete classes implement
4. **Thread Safety** - All shared state must be protected (locks, thread-local storage)
5. **Constants Centralization** - All magic numbers in engine/platform/constants.py

### Integration Constraints

1. **RHI Layer** - GPU operations go through rhi/ ABCs; concrete backends external
2. **Rust Renderer** - crates/renderer-backend/ provides real GPU implementation
3. **Event-Driven Input** - Input subsystem receives events; does not poll hardware directly

## Acceptance Criteria

### Phase Completion Criteria

Each phase is complete when:

1. All source files pass `python -m py_compile` without errors
2. All public APIs have type annotations
3. Unit tests exist and pass for all new functionality
4. No regressions in existing functionality
5. Thread safety verified where applicable
6. Documentation updated for API changes

### Subsystem-Specific Criteria

| Subsystem | Ready When |
|-----------|------------|
| audio/ | Platform backends (WASAPI, CoreAudio, ALSA) registered |
| gpu/ | Low-latency features connect to actual GPU APIs |
| input/ | Platform event sources (SDL, GLFW) integrated |
| os/ | mprotect/madvise available via ctypes on supported platforms |
| rhi/ | At least one concrete backend (wgpu) passes conformance |
| services/ | Permissions work on target mobile platforms |
| window/ | At least one native backend (Win32, X11, Wayland, Cocoa) functional |

### Quality Gates

1. **No Stub Markers** - No remaining `# Stub implementation` comments in production code
2. **No Hardcoded Returns** - No `return False` or `return None` placeholders in capability queries
3. **Event System Coverage** - All device types generate proper events
4. **Error Handling** - All I/O operations use Result pattern or exceptions with context

## Out of Scope

1. **Console SDK Integration** - PS5, Xbox, Switch require vendor NDAs
2. **Mobile Native UI** - iOS/Android UI uses native frameworks, not this layer
3. **Shader Compilation** - Handled by Rust renderer-backend, not Python RHI
4. **Audio DSP** - Reverb, Doppler, convolution belong in audio core, not platform layer
