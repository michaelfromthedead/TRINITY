# SUMMARY: engine/rendering/particles

## Metrics

| Metric | Value |
|--------|-------|
| Total Lines | 6,142 |
| Files | 7 |
| Classification | PARTIAL (CPU: REAL, GPU: STUBBED) |
| Modules | 17 particle modules |
| VFX Nodes | Graph-based compilation |
| Production Readiness | CPU: YES, GPU: NO |

## File Breakdown

| File | Lines | Status |
|------|-------|--------|
| particle_modules.py | 1,016 | REAL |
| vfx_graph.py | 995 | REAL |
| decal_system.py | 946 | REAL |
| particle_system.py | 850 | REAL |
| gpu_particles.py | 844 | PARTIAL |
| trail_renderer.py | 815 | REAL |
| constants.py | 160 | REAL |

## Algorithm Inventory

| Algorithm | Status | Evidence |
|-----------|--------|----------|
| Sphere Volume Sampling | COMPLETE | Cube-root radius correction for uniform distribution |
| Cone Sampling | COMPLETE | Correct angular distribution within cone angle |
| Box/Circle/Edge Sampling | COMPLETE | Standard random sampling |
| Gravity Force | COMPLETE | Constant acceleration integration |
| Wind Force | COMPLETE | Direction + turbulence noise |
| Turbulence Force | COMPLETE | Position-based pseudo-noise perturbation |
| Vortex Force | COMPLETE | Tangential swirl + radial pull |
| Attraction Force | COMPLETE | Point attraction with inverse-square falloff |
| Ground Collision | COMPLETE | Plane intersection + bounce/friction response |
| Spatial Hashing | COMPLETE | Cell-based O(1) neighbor lookup |
| Size Over Life | COMPLETE | Easing curves (linear, ease_in, ease_out) |
| Color Over Life | COMPLETE | Lerp or gradient texture sampling |
| Billboard Alignment | COMPLETE | View-aligned or velocity-aligned modes |
| Velocity Stretch | COMPLETE | Speed-proportional billboard stretching |
| Catmull-Rom Splines | COMPLETE | Smooth tangent calculation for trails |
| Trail Ribbon Mesh | COMPLETE | Per-segment vertices with width/UV |
| Trail Caps | COMPLETE | Start/end cap geometry |
| O(1) Pool Allocation | COMPLETE | Free list + reverse lookup dict |
| Pool Compaction | COMPLETE | Alive particle defragmentation |
| VFX Graph Compilation | COMPLETE | Node categorization to emitter modules |
| Shelf Atlas Packing | COMPLETE | Row-based texture allocation |
| Decal Box Projection | COMPLETE | Screen-space G-Buffer projection |
| GPU Particle Dispatch | STUBBED | Architecture only, no actual dispatch |

## Evidence Snippets

### Correct Volume Sampling (particle_modules.py:244)
```python
r = self._radius * (random.random() ** (1 / 3))  # Correct volume sampling
```

### O(1) Pool Management (particle_system.py:286-302)
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

### Spatial Hashing (particle_modules.py:666-697)
```python
def _get_cell(self, position: Vec3) -> tuple[int, int, int]:
    return (
        int(position.x // self._cell_size),
        int(position.y // self._cell_size),
        int(position.z // self._cell_size),
    )
```

### Trail Mesh Generation (trail_renderer.py:498-556)
```python
for i, point in enumerate(points):
    half_width = point.width * 0.5
    left_pos = point.position + point.right * half_width
    right_pos = point.position - point.right * half_width
```

### VFX Graph Compilation (vfx_graph.py:830-870)
```python
def compile(self) -> ParticleEmitter:
    if not self._dirty and self._compiled_emitter:
        return self._compiled_emitter
    self._categorize_modules()
    emitter = ParticleEmitter(config=emitter_config)
    for module in self._spawn_modules:
        pm = module.to_particle_module()
        if pm and pm.stage == ModuleStage.SPAWN:
            emitter.add_spawn_module(pm)
```

### GPU Dispatch Stub (gpu_particles.py:452-472)
```python
def update(self, dt: float) -> None:
    dispatch_size = self._calculate_dispatch_size(self._buffer.alive_count)
    # In actual implementation, this would:
    # 1. Bind update compute shader
    # 2. Set uniforms (dt, gravity, wind, etc.)
    # 3. Bind attribute buffers
    # 4. Dispatch compute shader
    # 5. Memory barrier
    pass  # Placeholder
```

### Atlas Shelf Packing (decal_system.py:568-632)
```python
if self._current_shelf_x + padded_width <= self._width:
    if self._current_shelf_y + padded_height <= self._height:
        x = self._current_shelf_x + self._padding
        y = self._current_shelf_y + self._padding
```
