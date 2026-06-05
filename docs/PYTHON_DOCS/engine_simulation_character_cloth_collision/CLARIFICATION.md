# CLARIFICATION: engine/simulation Philosophy and Design Rationale

## Architectural Philosophy

### Pure Python Implementation

The simulation subsystem deliberately implements physics algorithms in pure Python without external engine dependencies. This decision enables:

1. **Full Source Visibility**: Every algorithm step is inspectable and modifiable
2. **Python 3.13 Static Linking**: No native library complications for embedded interpreter
3. **Educational Clarity**: Industry-standard algorithms (GJK, EPA, PBD) are documented in-code
4. **Gradual GPU Migration**: CPU implementations serve as reference for GPU shader ports

### Industry-Standard Algorithms

The codebase implements algorithms from established physics engine literature:

- **GJK/EPA**: From Bullet Physics, used by PhysX, Box2D
- **Sweep and Prune**: Classic broadphase from ODE, Havok
- **Surface Area Heuristic BVH**: Used by Embree, OptiX, modern ray tracers
- **Position-Based Dynamics**: Muller et al., used by NVIDIA Flex, Unreal Chaos

This alignment ensures predictable behavior and enables knowledge transfer from existing resources.

## Design Rationale

### Character Physics

**Why 14 Movement Modes?**

Modern character controllers require nuanced state handling. The enum-based state machine with explicit transition rules prevents illegal state combinations (e.g., SWIMMING while PRONE) and enables stamina-gated transitions (SPRINTING requires stamina > 0).

**Why Active Ragdoll with PD Controllers?**

Proportional-Derivative control provides physics-based animation blending. The quaternion error calculation (`q_error = q_target * q_current.conjugate()`) enables smooth transitions between animation and physics-driven poses. Balance strategies (STEP, STUMBLE, FALL, BRACE) provide natural-looking recovery behaviors.

### Cloth Simulation

**Why Position-Based Dynamics?**

PBD is unconditionally stable (no exploding simulations), visually plausible, and maps well to GPU parallel execution. The constraint projection approach (predict, project, update) naturally handles multiple constraint types without complex solver coupling.

**Why Multiple Constraint Types?**

- **Distance**: Basic edge length preservation
- **Bending**: Dihedral angle control prevents flat/crumpled appearance
- **Shear**: Diagonal resistance for woven fabric behavior
- **Anchor**: Fixed attachment points
- **Tether**: Maximum stretch limit for stability
- **Long-Range**: Prevents excessive global stretching

**Why GPU Cloth as PARTIAL STUB?**

The buffer definitions and shader templates provide the interface contract without coupling to a specific GPU backend. This allows:
- Renderer-backend team to complete wgpu integration
- Shader code to be validated against CPU reference
- Clear boundary between physics logic and GPU plumbing

### Collision Detection

**Why Four Broadphase Algorithms?**

Different scene configurations favor different algorithms:
- **SAP**: Best for mostly-static scenes with incremental updates
- **BVH**: Best for ray queries and scenes with varied object sizes
- **Spatial Hash**: Best for uniform-size objects (particles, cloth)
- **Octree**: Best for sparse scenes with hierarchical clustering

Runtime selection or benchmarking enables game-specific optimization.

**Why Separate Narrowphase Functions?**

Specialized sphere/capsule tests are significantly faster than general GJK. The dispatcher (`collide_shapes()`) selects the optimal algorithm based on shape types, falling back to GJK/EPA only when necessary.

**Why 32 Collision Layers?**

32 bits fit in a single integer, enabling efficient bitwise filtering. Predefined layers (PLAYER, NPC, ENEMY, VEHICLE, etc.) cover common game patterns while 16 CUSTOM layers allow project-specific extensions.

## Duplication Acknowledgment

### Vec3 Duplication

`Vec3` is defined in both `broadphase.py` and `character_controller.py`. This is a known technical debt item. The duplication exists because:

1. Both modules were developed independently
2. No shared math module existed at development time
3. Extraction requires coordinated refactor across all consumers

The PHASE_2 TODO addresses this explicitly.

## GPU Cloth Gap Analysis

### What Exists

- `GPUBuffer` and `GPUClothBuffers` memory layout definitions
- `GPUComputePipeline` abstract interface
- GLSL shader templates for integration, constraint projection, velocity update

### What Is Missing

- Actual wgpu/Vulkan backend binding
- Shader compilation and dispatch logic
- GPU-CPU synchronization for collision results
- Buffer upload/download for particle positions

### Why This Matters

CPU cloth simulation works but does not scale to dense meshes (10K+ particles). GPU implementation enables real-time cloth for character clothing, capes, flags, and soft-body props.

## Classification Methodology

| Classification | Evidence Required |
|----------------|-------------------|
| REAL | Working algorithms, edge case handling, no placeholder returns |
| PARTIAL STUB | Interface exists, some logic present, key functionality no-op |
| STUB | Type signatures only, raises NotImplementedError |

The `gpu_cloth.py` PARTIAL STUB classification reflects that buffer definitions are usable but `step()` does nothing. The explicit warning string documents this intentionally.
