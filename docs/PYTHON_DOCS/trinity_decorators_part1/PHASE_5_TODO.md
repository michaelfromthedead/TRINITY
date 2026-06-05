# PHASE 5 TODO: Domain-Specific Decorators

## Summary
Implement all domain-specific decorators across 10 modules (~70 decorators total).

---

## T-DEC-5.1: Implement Compilation Decorators (7)

**File**: `trinity/decorators/compilation.py`

**Task**: FFI and platform decorators.

**Acceptance Criteria**:
- [ ] @native - marks native function binding
- [ ] @ffi(calling_conv="cdecl", dll="lib.so") - FFI config
- [ ] @target("windows", "linux", "macos") - platform-specific
- [ ] @unavailable_stub - generates NotImplementedError stub
- [ ] @inline - hint for inlining
- [ ] @no_gil - GIL release hint
- [ ] @vectorize - SIMD hint
- [ ] Platform detection at import time

---

## T-DEC-5.2: Implement Bridges/Caching Decorators (9)

**File**: `trinity/decorators/bridges_caching.py`

**Task**: Retry, caching, and observable decorators.

**Acceptance Criteria**:
- [ ] @retry(max_attempts, base_delay_ms, max_delay_ms, backoff)
- [ ] @throttle(calls_per_second) - rate limiting
- [ ] @cached(ttl_seconds) - result caching
- [ ] @lazy - deferred initialization
- [ ] @memoize - argument-based caching
- [ ] @observable - observer pattern
- [ ] @debounce(delay_ms) - call debouncing
- [ ] @circuit_breaker(threshold, reset_ms) - failure circuit
- [ ] @timeout(ms) - execution timeout
- [ ] Validation: base_delay_ms <= max_delay_ms

---

## T-DEC-5.3: Implement Destruction Decorators (6)

**File**: `trinity/decorators/destruction.py`

**Task**: Destructible object configuration.

**Acceptance Criteria**:
- [ ] @destructible(health, damage_threshold) - marks breakable
- [ ] @fracture_pattern(mode="voronoi", pieces=10) - fracture config
- [ ] @joint_physics(break_force, break_torque) - joint breaking
- [ ] @debris(lifetime, physics_enabled) - debris config
- [ ] @damage_type(types=["physical", "explosive"]) - damage filtering
- [ ] @destruction_event(callback) - destruction callback

---

## T-DEC-5.4: Implement Audio Extended Decorators (8)

**File**: `trinity/decorators/audio_extended.py`

**Task**: DSP and audio processing decorators.

**Acceptance Criteria**:
- [ ] @dsp_node(inputs, outputs) - DSP processing node
- [ ] @voice_priority(priority, steal_mode) - voice management
- [ ] @sidechain(source, ratio, threshold) - sidechain compression
- [ ] @spatial_audio(mode="hrtf", radius) - 3D audio
- [ ] @reverb_zone(preset, mix) - reverb configuration
- [ ] @audio_mixer(channels, routing) - mixer routing
- [ ] @audio_effect(type, params) - effect chain
- [ ] @ducking(source, amount, attack, release) - audio ducking

---

## T-DEC-5.5: Implement Rendering Decorators (6)

**File**: `trinity/decorators/rendering.py`

**Task**: Rendering pipeline configuration.

**Acceptance Criteria**:
- [ ] @gi_contribution(mode="baked"|"realtime"|"mixed")
- [ ] @shadow_caster(type, resolution, distance)
- [ ] @reflection_probe(resolution, refresh_mode)
- [ ] @lod_group(distances=[10, 50, 100]) - LOD configuration
- [ ] @occlusion_culling(mode, bounding_volume)
- [ ] @render_layer(layer_mask, sorting_order)

---

## T-DEC-5.6: Implement Modding Decorators (9)

**File**: `trinity/decorators/modding.py`

**Task**: Mod system integration.

**Acceptance Criteria**:
- [ ] @mod_metadata(name, version, author, description)
- [ ] @mod_dependency(mod_id, version_range)
- [ ] @moddable(fields=[]) - marks moddable fields
- [ ] @mod_hook(event) - mod event hook
- [ ] @mod_asset(type, path) - asset registration
- [ ] @mod_config(schema) - config definition
- [ ] @mod_api(version) - API version
- [ ] @mod_permission(permissions=[]) - permission requirements
- [ ] @incompatible_with(mod_ids=[]) - incompatibility declaration
- [ ] Version validation (semver)

---

## T-DEC-5.7: Implement AI Generation Decorators (7)

**File**: `trinity/decorators/ai_generation.py`

**Task**: AI/ML training data decorators.

**Acceptance Criteria**:
- [ ] @example(category, weight) - training example
- [ ] @pattern_category(categories=[]) - pattern tagging
- [ ] @complexity(level, reasoning) - complexity annotation
- [ ] @feature_vector(dimensions) - feature extraction
- [ ] @label(labels=[]) - classification labels
- [ ] @augmentable(transforms=[]) - data augmentation
- [ ] @embedding(model, dimensions) - embedding configuration

---

## T-DEC-5.8: Implement Physics Sim Decorators (7)

**File**: `trinity/decorators/physics_sim.py`

**Task**: Physics simulation configuration.

**Acceptance Criteria**:
- [ ] @solver(type="pbd"|"xpbd"|"impulse") - solver selection
- [ ] @ccd_mode(mode="discrete"|"swept"|"speculative") - collision detection
- [ ] @buoyancy(density, drag, angular_drag) - fluid physics
- [ ] @wind(force, turbulence, direction) - wind physics
- [ ] @soft_body(stiffness, damping) - soft body config
- [ ] @cloth(bend_stiffness, stretch_stiffness) - cloth simulation
- [ ] @ragdoll(joint_limits, muscle_strength) - ragdoll config
- [ ] Validation: solver type in allowed set

---

## T-DEC-5.9: Implement Crafting Decorators (5)

**File**: `trinity/decorators/crafting.py`

**Task**: Crafting system decorators.

**Acceptance Criteria**:
- [ ] @recipe(ingredients, outputs, time) - crafting recipe
- [ ] @ingredient(type, quantity, quality_range) - ingredient config
- [ ] @loot_table(entries, weights, rolls) - loot generation
- [ ] @quality_tier(tier, stat_modifiers) - item quality
- [ ] @craftable(station, skill_required) - crafting requirements

---

## T-DEC-5.10: Implement Particles/VFX Decorators (6)

**File**: `trinity/decorators/particles_vfx.py`

**Task**: Particle system configuration.

**Acceptance Criteria**:
- [ ] @particle_system(max_particles, emission_rate)
- [ ] @gpu_compute(shader, dispatch_size) - GPU particles
- [ ] @trail_renderer(width, time, color_gradient) - trails
- [ ] @billboard(mode="camera"|"velocity"|"world")
- [ ] @particle_collision(bounce, lifetime_loss)
- [ ] @vfx_event(trigger, prefab) - VFX event triggers

---

## Dependencies

```
PHASE 1-4 ──> T-DEC-5.1 (Tier 0, no deps)
          ──> T-DEC-5.7 (Tier 9, minimal deps)
          ──> T-DEC-5.6 (Tier 30)
          ──> T-DEC-5.5 (Tier 42, depends on gpu.py)
          ──> T-DEC-5.3 (Tier 43, depends on physics_sim)
          ──> T-DEC-5.10 (Tier 45, depends on gpu.py)
          ──> T-DEC-5.8 (Tier 46, depends on scheduling)
          ──> T-DEC-5.4 (Tier 49, depends on scheduling)
          ──> T-DEC-5.9 (Tier 52)
          ──> T-DEC-5.2 (Tier 53, top-level)
```

## Estimated Effort

| Task | Decorators | Lines | Complexity |
|------|------------|-------|------------|
| T-DEC-5.1 | 7 | ~150 | Medium |
| T-DEC-5.2 | 9 | ~200 | Medium |
| T-DEC-5.3 | 6 | ~120 | Low |
| T-DEC-5.4 | 8 | ~160 | Medium |
| T-DEC-5.5 | 6 | ~120 | Low |
| T-DEC-5.6 | 9 | ~180 | Medium |
| T-DEC-5.7 | 7 | ~140 | Low |
| T-DEC-5.8 | 7 | ~140 | Medium |
| T-DEC-5.9 | 5 | ~100 | Low |
| T-DEC-5.10 | 6 | ~120 | Medium |

**Total Phase 5**: ~70 decorators, ~1430 lines
