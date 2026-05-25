# Investigation: engine/rendering/particles

**Lines**: 5,982 (6 main files) + 160 (constants.py) = 6,142 total
**Classification**: REAL (with GPU dispatch stubs awaiting RHI)

## Summary
The particle system is a **REAL IMPLEMENTATION** with fully functional CPU simulation, comprehensive physics modules (gravity, wind, turbulence, vortex, collision with spatial hashing), modular spawn/update/render architecture, trail rendering with Catmull-Rom splines, and deferred decal system with G-Buffer support. GPU compute particle simulation has complete architecture but GPU dispatch calls are placeholders awaiting RHI integration.

## Files
| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `particle_modules.py` | 1,016 | REAL | 17 modules: spawn, forces, collision, rendering |
| `vfx_graph.py` | 995 | REAL | Node-based graph compilation to emitter |
| `decal_system.py` | 946 | REAL | Deferred decals, G-Buffer, atlas packing |
| `particle_system.py` | 850 | REAL | Full particle emitter, pool, budget management |
| `gpu_particles.py` | 844 | PARTIAL | Architecture complete, GPU dispatch stubbed |
| `trail_renderer.py` | 815 | REAL | Full mesh generation, Catmull-Rom, caps |
| `constants.py` | 160 | REAL | Centralized config constants |

## Particle Components
- **ParticleEmitter**: Full lifecycle (spawn/update/death), prewarm, budget, modules
- **ParticlePool**: O(1) ring buffer with free list, compaction
- **ParticleBudget**: Category-based limits (ambient/gameplay/critical)
- **GPUParticleSystem**: SoA buffers, attributes, simulator, renderer (GPU dispatch stubbed)
- **VFXGraph**: Node-based authoring, compiles to ParticleEmitter
- **TrailRenderer**: Ring buffer, Catmull-Rom tangents, ribbon mesh, caps
- **DecalSystem**: Box projection, deferred G-Buffer, atlas packing, sorting

## Spawn Modules
- `ShapeEmitter`: Point, sphere, box, cone, circle, edge (real sampling math)
- `BurstEmitter`: Instant count with repeat interval
- `RateEmitter`: Particles/second with accumulator

## Force/Update Modules
- `GravityModule`: Constant acceleration
- `WindModule`: Direction + turbulence
- `TurbulenceModule`: Position-based pseudo-noise
- `VortexModule`: Tangential swirl + radial pull
- `AttractionModule`: Point attraction with falloff
- `VectorFieldModule`: 3D force volume (data loading stubbed)
- `CollisionModule`: Ground plane collision with bounce/friction, spatial hashing

## Attribute Modules
- `SizeOverLifeModule`: Easing curves (linear, ease_in/out)
- `ColorOverLifeModule`: Lerp or gradient sampling
- `RotationModule`: Random initial rotation + angular velocity
- `LifetimeModule`: Random lifetime range
- `VelocityModule`: Initial velocity with spread

## Render Modules
- `BillboardRenderer`: View/velocity alignment, velocity stretch
- `MeshParticleRenderer`: Instanced mesh prep

## Implementation
- Real particle simulation? **YES** - CPU physics fully implemented
- Real GPU compute? **NO** - Architecture present, dispatch methods are placeholders
- Real VFX graph? **YES** - Compiles to working ParticleEmitter
- GPU execution? **STUBBED** - Comments indicate where GL calls would go

## Verdict
**PARTIAL IMPLEMENTATION** - CPU particle system is production-ready. GPU compute is architecture-only with stubbed execution.

## Evidence

### Real CPU Physics (particle_modules.py:423-427)
```python
def apply_to_particle(self, particle: Particle, dt: float) -> None:
    particle.acceleration = particle.acceleration + self._gravity
```

### Real Pool Management (particle_system.py:286-302)
```python
def allocate(self) -> Optional[Particle]:
    if not self._free_indices:
        return None
    index = self._free_indices.pop()
    particle = self._particles[index]
    particle.reset()
    particle.state = ParticleState.ALIVE
    self._alive_indices.add(index)
    self._alive_count += 1
    return particle
```

### Stubbed GPU Dispatch (gpu_particles.py:452-472)
```python
def update(self, dt: float) -> None:
    if self._buffer.alive_count == 0:
        return
    dispatch_size = self._calculate_dispatch_size(self._buffer.alive_count)
    # In actual implementation, this would:
    # 1. Bind update compute shader
    # 2. Set uniforms (dt, gravity, wind, etc.)
    # 3. Bind attribute buffers
    # 4. Dispatch compute shader
    # 5. Memory barrier
    # Placeholder for simulation logic
    pass
```

### Real Trail Mesh Generation (trail_renderer.py:498-556)
```python
def _generate_mesh(self) -> None:
    # ...calculates tangents via Catmull-Rom...
    # ...computes perpendiculars for ribbon width...
    for i, point in enumerate(points):
        # Calculate ribbon edge positions
        half_width = point.width * 0.5
        left_pos = point.position + point.right * half_width
        right_pos = point.position - point.right * half_width
        # Generates actual vertex data with UVs and colors
```

### Real Collision with Spatial Hashing (particle_modules.py:666-697)
```python
def _get_cell(self, position: Vec3) -> tuple[int, int, int]:
    return (
        int(position.x // self._cell_size),
        int(position.y // self._cell_size),
        int(position.z // self._cell_size),
    )

def get_nearby_particles(self, particle: Particle) -> list[Particle]:
    cell = self._get_cell(particle.position)
    nearby = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                # 3x3x3 neighborhood check
```

### Real Sphere Sampling with Volume Correction (particle_modules.py:238-255)
```python
def _sample_sphere(self) -> Tuple[Vec3, Vec3]:
    theta = random.uniform(0, 2 * math.pi)
    phi = math.acos(random.uniform(-1, 1))
    dir_x = math.sin(phi) * math.cos(theta)
    dir_y = math.sin(phi) * math.sin(theta)
    dir_z = math.cos(phi)
    direction = Vec3(dir_x, dir_y, dir_z)
    if self._emit_from_surface:
        r = self._radius
    else:
        r = self._radius * (random.random() ** (1 / 3))  # Correct volume sampling
    position = direction * r
    return position, direction
```

### Real VFX Graph Compilation (vfx_graph.py:830-870)
```python
def compile(self) -> ParticleEmitter:
    if not self._dirty and self._compiled_emitter:
        return self._compiled_emitter
    self._categorize_modules()
    # Find emitter configuration
    emitter_config = EmitterConfig()
    for module in self._spawn_modules:
        if isinstance(module, VFXEmitterModule):
            emitter_config = module.to_emitter_config()
            break
    emitter = ParticleEmitter(config=emitter_config)
    # Add spawn/update/render modules
    for module in self._spawn_modules:
        pm = module.to_particle_module()
        if pm and pm.stage == ModuleStage.SPAWN:
            emitter.add_spawn_module(pm)
```

### Real Decal Atlas Shelf Packing (decal_system.py:568-632)
```python
def add_texture(self, texture_id: str, texture_width: int, texture_height: int):
    padded_width = texture_width + self._padding * 2
    padded_height = texture_height + self._padding * 2
    # Try to fit on current shelf
    if self._current_shelf_x + padded_width <= self._width:
        if self._current_shelf_y + padded_height <= self._height:
            # Fits on current shelf
            x = self._current_shelf_x + self._padding
            y = self._current_shelf_y + self._padding
            # ...allocate region...
    # Start new shelf if needed
    new_shelf_y = self._current_shelf_y + self._current_shelf_height
```

### O(1) Pool Deallocation with Reverse Lookup (particle_system.py:270-316)
```python
self._particle_to_index: dict[int, int] = {
    id(p): i for i, p in enumerate(self._particles)
}

def deallocate(self, particle: Particle) -> None:
    """Return a particle to the pool for reuse. O(1) operation."""
    particle_id = id(particle)
    if particle_id not in self._particle_to_index:
        return
    index = self._particle_to_index[particle_id]
    if index in self._alive_indices:
        self._alive_indices.remove(index)
        self._free_indices.append(index)
        particle.state = ParticleState.DEAD
```
