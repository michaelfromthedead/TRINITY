# PHASE 5 ARCHITECTURE: Domain-Specific Decorators

## Phase Scope

Domain decorators: `compilation.py` (Tier 0), `bridges_caching.py` (Tier 53), `destruction.py` (Tier 43), `audio_extended.py` (Tier 49), `rendering.py` (Tier 42), `modding.py` (Tier 30), `ai_generation.py` (Tier 9), `physics_sim.py` (Tier 46), `crafting.py` (Tier 52), `particles_vfx.py` (Tier 45)

## Architecture Decisions

### ADR-DEC-017: FFI Binding Decorators

**Context**: Native code integration requires platform detection and binding generation.

**Decision**: `compilation.py` provides:
- `@native` - marks native function binding
- `@ffi` - FFI configuration (calling convention, DLL)
- `@target(platform)` - platform-specific compilation
- Unavailability stubs for missing platforms

**Consequences**:
- Platform detection at import time
- Stub functions raise `NotImplementedError` on unsupported platforms
- FFI metadata drives code generation

### ADR-DEC-018: Retry with Exponential Backoff

**Context**: Network/IO operations need retry logic.

**Decision**: `@retry` with backoff parameters:
```python
@retry(max_attempts=3, base_delay_ms=100, max_delay_ms=5000, backoff=2.0)
```

Validation:
- `base_delay_ms <= max_delay_ms`
- `backoff >= 1.0`

**Consequences**:
- Automatic retry on exception
- Exponential backoff prevents thundering herd
- Configurable per-decorator

### ADR-DEC-019: Destructible Physics Configuration

**Context**: Objects need destruction physics configuration.

**Decision**: `destruction.py` provides:
- `@destructible` - marks object as breakable
- `@fracture_pattern` - fracture mesh generation
- `@joint_physics` - joint/constraint configuration

**Consequences**:
- Physics system knows destruction parameters
- Fracture meshes can be pre-computed
- Joint breaking thresholds configurable

### ADR-DEC-020: DSP Node Chain

**Context**: Audio processing needs node-based DSP configuration.

**Decision**: `audio_extended.py` provides:
- `@dsp_node` - marks audio processing node
- `@voice_priority` - voice stealing priority
- `@sidechain` - sidechain compression source

**Consequences**:
- Audio graph construction from decorators
- Voice management based on priority
- Sidechain routing explicit

## Component Diagram

```
+-----------------+
| compilation.py  |  @native, @ffi, @target, platform stubs
+-----------------+

+-------------------+
| bridges_caching.py|  @retry, @throttle, @cached, @lazy, @observable
+-------------------+

+-----------------+
| destruction.py  |  @destructible, @fracture_pattern, @joint_physics
+-----------------+

+------------------+
| audio_extended.py|  @dsp_node, @voice_priority, @sidechain
+------------------+

+-----------------+
|  rendering.py   |  @gi_contribution, @shadow_caster, @reflection_probe
+-----------------+

+-----------------+
|   modding.py    |  @mod_metadata, @mod_dependency, @moddable
+-----------------+

+-----------------+
| ai_generation.py|  @example, @pattern_category, @complexity
+-----------------+

+-----------------+
|  physics_sim.py |  @solver, @ccd_mode, @buoyancy, @wind
+-----------------+

+-----------------+
|   crafting.py   |  @recipe, @ingredient, @loot_table
+-----------------+

+-----------------+
| particles_vfx.py|  @particle_system, @gpu_compute, @trail_renderer
+-----------------+
```

## Decorator Count by Module

| Module | Decorators | Tier |
|--------|------------|------|
| compilation.py | 7 | 0 |
| bridges_caching.py | 9 | 53 |
| destruction.py | 6 | 43 |
| audio_extended.py | 8 | 49 |
| rendering.py | 6 | 42 |
| modding.py | 9 | 30 |
| ai_generation.py | 7 | 9 |
| physics_sim.py | 7 | 46 |
| crafting.py | 5 | 52 |
| particles_vfx.py | 6 | 45 |

## Validation Patterns by Domain

### Retry/Caching
```python
def _validate_retry_params(**kwargs):
    if base_delay_ms > max_delay_ms:
        raise ValueError("base_delay_ms must be <= max_delay_ms")
    if backoff < 1.0:
        raise ValueError("backoff must be >= 1.0")
```

### Physics
```python
def _validate_solver(solver_type: str, **_):
    if solver_type not in ("pbd", "xpbd", "impulse"):
        raise ValueError("solver_type must be pbd, xpbd, or impulse")
```

### Rendering
```python
def _validate_gi_contribution(mode: str, **_):
    if mode not in ("baked", "realtime", "mixed"):
        raise ValueError("GI mode must be baked, realtime, or mixed")
```

## Cross-Domain Dependencies

```
rendering.py ──> gpu.py (shader references)
particles_vfx.py ──> gpu.py (compute shaders)
audio_extended.py ──> scheduling.py (audio thread)
physics_sim.py ──> scheduling.py (fixed timestep)
destruction.py ──> physics_sim.py (physics config)
```
