# PHASE 5 ARCHITECTURE: GPU Particle Compute Passes

> **Phase**: 5/7 | **Status**: [x] 88% (3 implemented, 1 partial)
> **Tasks**: T-GPU-5.1 through T-GPU-5.4 (4 tasks)
> **Gaps**: S9-G1, S9-G2, S9-G3, S9-G4

---

## Files Implemented

| File | Lines | Role |
|------|-------|------|
| `crates/.../shaders/particles.wgsl` | 272 | Full particle compute pipeline |
| `crates/.../src/particles.rs` | 431 | Rust pass factory functions |
| `engine/rendering/particles/gpu_particles.py` | 776 | Python GPU particle orchestrator |
| `engine/rendering/particles/particle_system.py` | 855 | CPU particle system |

## Reality by Task

### T-GPU-5.1: Particle spawn [x] IMPLEMENTED
`particles.wgsl` `particle_spawn()` (lines 88-151):
```
for each thread gid.x < spawn_count:
    old_count = atomicAdd(&alive_count[0], 1)
    if old_count >= emitter.max_particles:
        atomicSub → rollback
        return
    idx = old_count
    // Random position within emitter sphere (uniform volume distribution)
    // Random velocity (hemisphere up, random speed)
    // Random lifetime, size
    particles[idx] = Particle(emitter_position + offset, 0, velocity, lifetime, size, ...)
```

- Atomic counter for pool allocation
- Hash-based PRNG (hash11/hash31)
- RGBA color NOT initialized (Particle struct has no color field)
- `particles.rs` `create_particle_spawn_pass()` creates IrPass for frame graph

### T-GPU-5.2: Particle update [x] IMPLEMENTED (alternative approach)
`particles.wgsl` `particle_update()` (lines 157-188):
```
for each thread gid.x < alive_count:
    age += delta_time
    if age >= lifetime: mark dead (lifetime=0), increment dead_count
    velocity += gravity * dt
    velocity *= (1 - drag * dt)
    position += velocity * dt
    size *= (1 - life_ratio * 0.5)
```

- **Deviation from spec**: Uses single storage buffer (read-modify-write), not ping-pong SoA buffers
- Gravity, drag, age, size-over-life implemented
- No wind, turbulence, vortex, or attraction forces
- No color-over-life

### T-GPU-5.3: Particle compact [~] SIMPLIFIED ALTERNATIVE
`particles.wgsl` `particle_compact()` (lines 235-265):
```
for each thread gid.x < alive_count:
    if particle[gid.x].lifetime <= 0:
        last_idx = alive_count - 1
        if gid.x < last_idx and particle[last_idx].lifetime > 0:
            swap particle[gid.x] with particle[last_idx]
            particle[last_idx].lifetime = 0
        atomicSub(&alive_count[0], 1)
```

- **Deviation from spec**: Swap-based compaction, NOT prefix-sum + scatter
- No indirect draw count update (rendering uses vertex_buffer fill, not indirect draw)
- Race condition documented: "one-frame visual glitch at worst"
- Has `particle_reset_dead_count()` helper shader

### T-GPU-5.4: Particle sort [ - ] NOT IMPLEMENTED
- No depth computation, no radix sort, no indirection array for translucent sort order
- Particle struct has no depth field
- All particles render in arbitrary order (pool allocation order)

## Complete Particle Pipeline Architecture

```
Each frame (4 compute dispatches):

  1. particle_spawn (1 workgroup → spawn_count threads)
     ├── atomic alloc from pool
     └── init position, velocity, lifetime, size

  2. particle_update (1 workgroup → alive_count threads)
     ├── age += dt
     ├── velocity += gravity * dt
     ├── velocity *= (1 - drag * dt)
     ├── position += velocity * dt
     ├── size shrink over life
     └── mark dead if age >= lifetime

  3. particle_render (1 workgroup → alive_count threads)
     ├── compute camera-facing quad vertices
     └── write 6 vertices to vertex_buffer per alive particle

  4. particle_compact (1 workgroup → alive_count threads)
     ├── swap dead particles to end
     └── atomic decrement alive count

  5. particle_reset_dead_count (single thread)
     └── reset dead counter to 0
```

## Particle Struct (GPU)

```wgsl
struct Particle {
    position: vec3<f32>,  // 12 bytes
    age: f32,              // 4 bytes
    velocity: vec3<f32>,  // 12 bytes
    lifetime: f32,         // 4 bytes
    size: f32,             // 4 bytes
    _pad0: f32, _pad1: f32, _pad2: f32,  // 12 bytes padding
}; // Total: 48 bytes, 12 floats (PARTICLE_STRIDE_FLOATS = 12)
```

## Rust Pass Factories (particles.rs)

All four pass types created:
| Function | Name | Type | Access |
|----------|------|------|--------|
| `create_particle_spawn_pass` | particle_spawn | Compute | write particle_buffer |
| `create_particle_update_pass` | particle_update | Compute | read + write particle_buffer |
| `create_particle_render_pass` | particle_render | Graphics | read particle_buffer, write output |
| `create_particle_compact_pass` | particle_compact | Compute | read + write particle_buffer |

Each returns an `IrPass` with tags (e.g., `["particle", "spawn"]`), proper AccessSet, and DispatchSource/InstanceSource.
