# CLARIFICATION: Lighting Subsystem Design Philosophy

## Why CPU-First Architecture?

The lighting subsystem was implemented CPU-first as a **reference implementation**. This approach:

1. **Validates mathematics** before committing to GPU shader code
2. **Enables unit testing** of complex algorithms (CSM splits, SH evaluation, DDGI octahedral encoding)
3. **Provides simulation mode** for headless testing and baking
4. **Documents intent** - Python code serves as executable specification for WGSL shaders

The CPU code is not redundant; it remains the authoritative definition of behavior.

## Partial Implementation Classification

The investigation classified this as "PARTIAL IMPLEMENTATION" rather than "STUB" because:

- **All math is real** - CSM logarithmic splits, PCSS penumbra estimation, SH basis functions
- **Data structures are complete** - Shadow atlases pack, froxel grids slice, probe grids interpolate
- **Only GPU execution is missing** - A well-defined surface to fill

This is fundamentally different from a stub that says `# TODO: implement lighting`.

## Shadow Technique Selection

### Cascade Shadow Maps (CSM)
Chosen for directional lights because:
- Industry standard for large outdoor scenes
- Logarithmic split scheme balances near/far quality
- Stabilization (texel snapping) prevents shimmer
- Cascade count is runtime-configurable (1-4)

### Shadow Filtering Hierarchy
Multiple techniques exist because each has tradeoffs:

| Technique | Cost | Quality | Use Case |
|-----------|------|---------|----------|
| PCF | Low | Hard edges | Distant shadows, performance mode |
| PCSS | Medium | Variable penumbra | Area light simulation |
| VSM | Medium | Smooth, light bleeds | Fast soft shadows |
| ESM | Low | Smooth, artifacts | Mobile/low-end |
| Contact | Variable | Screen-space detail | Ground contact only |

The system allows per-light technique selection based on importance and GPU budget.

## Clustered Lighting Design

### Why Froxels?
3D frustum-aligned voxels provide:
- Efficient GPU traversal (single texture lookup per fragment)
- Natural LOD (far froxels cover more area)
- Predictable memory footprint (fixed grid size)

### Exponential Depth Slicing
The slice formula:
```
depth = near * pow(far/near, (slice + 0.5) / num_slices)
```

This allocates more resolution to near slices where precision matters, matching perspective projection's depth distribution.

## Global Illumination Strategy

### SH Probes (Baked)
L2 spherical harmonics (27 coefficients per probe) encode low-frequency irradiance:
- Fast evaluation (9 multiplies per color channel)
- Compact representation (108 bytes per probe RGB)
- Supports trilinear interpolation between probes

Limitations: Only captures diffuse lighting, requires rebaking for dynamic lights.

### DDGI (Dynamic)
Dynamic Diffuse Global Illumination provides:
- Real-time indirect lighting updates
- Ray-traced visibility for correct occlusion
- Octahedral encoding for efficient GPU storage

The implementation uses:
- **Chebyshev visibility test** for soft indirect shadows
- **Fibonacci spiral** for ray distribution
- **Scrolling grids** for efficient updates in large worlds

### Reflection Probes
Parallax box correction solves the "infinite environment" problem:
- Probes store finite-volume captures
- Correction vector moves sample direction to box intersection
- Blend factor handles probe overlap

## Integration Philosophy

### CPU Keeps Authority
The CPU implementation should remain the **source of truth** for:
- Configuration (cascade distances, probe positions, atlas layout)
- Culling decisions (which lights affect which froxels)
- Update scheduling (DDGI probe priorities)

The GPU executes but does not decide.

### Shader Generation Over Shader Writing
Rather than hand-writing WGSL, the system should:
1. Read CPU-side configuration
2. Generate shaders with correct constants baked in
3. Handle permutations (filter type, light count, GI mode) via preprocessor

This prevents drift between CPU config and GPU code.

## What "GPU Integration" Means

For each component, integration requires:

| Component | GPU Resource | Shader Code | Upload Path |
|-----------|--------------|-------------|-------------|
| Shadow Map | Depth texture | Sample + compare | Render pass output |
| Shadow Filter | None (shader only) | PCF/PCSS/VSM/ESM | N/A |
| Light List | Storage buffer | Iterate lights | CPU cull -> upload |
| SH Probe | 3D texture or buffer | Evaluate SH | Bake -> upload |
| DDGI | 2D texture array | Octahedral lookup | RT pass output |
| Reflection | Cubemap array | Corrected sample | Capture -> upload |

Each row is independent work with clear inputs and outputs.
