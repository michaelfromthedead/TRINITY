# PROJECT: Trinity Decorators System

## Source
`docs/investigation/trinity_decorators_part1.md`

## Classification
REAL code - 100% implemented, no stubs detected across 20 files (~10,900 lines)

## Scope

The Trinity decorator system implements a Chomsky grammar-based declarative configuration layer for a game engine. The system:

1. Defines 7 primitive operations (TAG, HOOK, REGISTER, DESCRIBE, TRACK, VALIDATE, INTERCEPT)
2. Composes these into ~124 domain-specific decorators
3. Organizes decorators across 54 tiers (0-53)
4. Provides thread-safe registry management
5. Integrates with ECS metaclass system

## Goals

1. Maintain production-quality decorator composition system
2. Ensure WGSL-compliant GPU struct layout algorithms
3. Provide thread-safe memory management decorators
4. Support async scheduling with fixed timestep physics
5. Enable serialization/network state management
6. Integrate with ComponentMeta for ECS registration

## Constraints

- Standard library only for core dependencies
- Thread safety required for registry and atomic operations
- WGSL alignment rules must be strictly followed
- All decorator parameters must be validated with descriptive errors
- Flyweight pattern requires registry-based ID management

## Acceptance Criteria

### Core System
- [ ] 7 primitive Ops implemented in `ops.py`
- [ ] `make_decorator()` factory functional
- [ ] 54-tier system in `registry.py`
- [ ] Thread-safe singleton registry
- [ ] Attribute attachment via `base.py`

### GPU Decorators
- [ ] WGSL struct layout computation correct
- [ ] wgpu usage flags resolve per WebGPU spec
- [ ] MSAA validation (powers of 2: 1, 2, 4, 8, 16)

### Memory Decorators
- [ ] Flyweight registry with auto-incrementing IDs
- [ ] Atomic operations with RLock
- [ ] Pool allocation support
- [ ] CoW semantics

### Scheduling Decorators
- [ ] Fixed timestep with Hz/delta calculation
- [ ] Async coroutine detection
- [ ] Chain linking for dependent systems

### Data Flow
- [ ] Serialize/deserialize with version tracking
- [ ] Snapshot history ring buffer
- [ ] Network config support

### ECS Integration
- [ ] ComponentMeta integration
- [ ] Query extraction from type hints
- [ ] System registration

## Files Examined (20)

| File | Lines | Decorators |
|------|-------|------------|
| gpu.py | 886 | 8 |
| __init__.py | 857 | N/A |
| registry.py | 704 | N/A |
| compilation.py | 654 | 7 |
| base.py | 644 | N/A |
| dev.py | 643 | 9 |
| memory.py | 640 | 12 |
| ops.py | 588 | N/A |
| ecs_core.py | 580 | 9 |
| bridges_caching.py | 516 | 9 |
| destruction.py | 477 | 6 |
| audio_extended.py | 461 | 8 |
| data_flow.py | 455 | 4 |
| scheduling.py | 445 | 12 |
| rendering.py | 443 | 6 |
| modding.py | 421 | 9 |
| ai_generation.py | 411 | 7 |
| physics_sim.py | 380 | 7 |
| crafting.py | 379 | 5 |
| particles_vfx.py | 360 | 6 |
