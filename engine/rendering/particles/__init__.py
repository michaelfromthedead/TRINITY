"""
Particles & VFX Rendering Subsystem.

Provides GPU-accelerated particle systems, VFX graphs, trails, and decals
for creating visual effects in the game engine.

Subsystems:
    particle_system - Core emitter lifecycle and budget management
    gpu_particles - GPU compute shader particle simulation
    particle_modules - Modular spawn/update/render behaviors
    vfx_graph - Node-based VFX authoring
    trail_renderer - Trail/ribbon rendering
    decal_system - Projected decals

Usage:
    from engine.rendering.particles import (
        ParticleEmitter,
        EmitterConfig,
        GPUParticleSystem,
        VFXGraph,
        TrailRenderer,
        DecalSystem,
    )

    # Create a simple particle emitter
    config = EmitterConfig(max_particles=10000, simulation="gpu")
    emitter = ParticleEmitter(config=config)
    emitter.add_spawn_module(RateEmitter(rate=500))
    emitter.add_update_module(GravityModule())
    emitter.start()

    # Create a VFX graph
    graph = VFXGraph("Fire")
    graph.add_module(VFXEmitterModule(max_particles=5000))
    graph.add_module(VFXSpawnRateModule(rate=200))
    graph.add_module(VFXGravityModule())
    emitter = graph.compile()

    # Create trails
    trail = TrailRenderer(TrailConfig(width=0.1, fade_time=2.0))
    trail.update(dt, position=player_position)

    # Create decals
    decal_system = DecalSystem()
    decal = decal_system.spawn(hit_position, texture_id="bullet_hole")
"""

from __future__ import annotations

# Centralized constants
from engine.rendering.particles.constants import (
    ParticleConstants,
    PARTICLE_CONSTANTS,
)

# Core particle system
from engine.rendering.particles.particle_system import (
    # Enums
    SimulationMode,
    EmitterState,
    ParticleState,
    # Data structures
    Vec3,
    Vec4,
    Particle,
    # Configuration
    EmitterConfig,
    # Core classes
    ParticlePool,
    ParticleBudget,
    BudgetAllocation,
    ParticleEmitter,
    ParticleSystemManager,
)

# GPU particles
from engine.rendering.particles.gpu_particles import (
    # Enums
    AttributeType,
    BufferUsage,
    DrawMode,
    # Constants
    ATTRIBUTE_SIZES,
    STANDARD_ATTRIBUTES,
    # Attributes
    GPUParticleAttribute,
    GPUParticleAttributes,
    # Buffer
    BufferAllocation,
    GPUParticleBuffer,
    # Simulator
    ComputeShaderConfig,
    GPUParticleSimulator,
    # Renderer
    RenderState,
    GPUParticleRenderer,
    # Config
    GPUParticleConfig,
    # Combined System
    GPUParticleSystem,
)

# Particle modules
from engine.rendering.particles.particle_modules import (
    # Enums
    ModuleStage,
    EmitterShape,
    CollisionMode,
    BlendMode,
    # Config
    ModuleConfig,
    # Base class
    ParticleModule,
    # Spawn modules
    ShapeEmitter,
    BurstEmitter,
    RateEmitter,
    # Force modules
    GravityModule,
    WindModule,
    TurbulenceModule,
    VortexModule,
    AttractionModule,
    VectorFieldModule,
    CollisionModule,
    # Attribute modules
    SizeOverLifeModule,
    ColorOverLifeModule,
    RotationModule,
    LifetimeModule,
    VelocityModule,
    # Render modules
    BillboardRenderer,
    MeshParticleRenderer,
)

# VFX graph
from engine.rendering.particles.vfx_graph import (
    # Enums
    VFXContextType,
    VFXParameterType,
    VFXEventTrigger,
    VFXNodeType,
    # Context
    VFXContext,
    # Parameter
    VFXParameter,
    # Event
    VFXEventConfig,
    VFXEvent,
    # Modules
    VFXModule,
    VFXConnection,
    VFXEmitterModule,
    VFXSpawnRateModule,
    VFXBurstModule,
    VFXGravityModule,
    VFXSizeOverLifeModule,
    VFXColorOverLifeModule,
    VFXEventModule,
    # Graph
    VFXGraph,
)

# Trail renderer
from engine.rendering.particles.trail_renderer import (
    # Enums
    TextureMode,
    TrailAlignment,
    TrailCapStyle,
    # Configuration
    TrailConfig,
    # Data structures
    TrailPoint,
    TrailBuffer,
    TrailVertex,
    TrailMesh,
    # Renderer
    TrailRenderer,
    TrailManager,
)

# Decal system
from engine.rendering.particles.decal_system import (
    # Enums
    DecalChannel,
    DecalBlendMode,
    DecalProjection,
    DecalSortMode,
    # Configuration
    DecalConfig,
    # Data structures
    DecalVolume,
    AtlasRegion,
    # Decal types
    Decal,
    DeferredDecal,
    # Atlas
    DecalAtlas,
    # Sorting
    DecalSorting,
    # System
    DecalSystem,
)

# Deterministic emitter (T-CC-2.1: Fixed32 particle system)
from engine.rendering.particles.deterministic_emitter import (
    # Fixed32 types
    Fixed32Vec3,
    # Particle
    DeterministicParticle,
    # Pool
    DeterministicParticlePool,
    # Module base
    DeterministicSpawnModule,
    # Spawn modules
    DeterministicEmitterShape,
    DeterministicShapeEmitter,
    DeterministicRateEmitter,
    DeterministicLifetimeModule,
    DeterministicVelocityModule,
    DeterministicSizeModule,
    # Update modules
    DeterministicGravityModule,
    # Emitter
    DeterministicEmitter,
)


__all__ = [
    # =========================================================================
    # CONSTANTS
    # =========================================================================
    "ParticleConstants",
    "PARTICLE_CONSTANTS",
    # =========================================================================
    # PARTICLE SYSTEM
    # =========================================================================
    # Enums
    "SimulationMode",
    "EmitterState",
    "ParticleState",
    # Data structures
    "Vec3",
    "Vec4",
    "Particle",
    # Configuration
    "EmitterConfig",
    # Core classes
    "ParticlePool",
    "ParticleBudget",
    "BudgetAllocation",
    "ParticleEmitter",
    "ParticleSystemManager",
    # =========================================================================
    # GPU PARTICLES
    # =========================================================================
    # Enums
    "AttributeType",
    "BufferUsage",
    "DrawMode",
    # Constants
    "ATTRIBUTE_SIZES",
    "STANDARD_ATTRIBUTES",
    # Attributes
    "GPUParticleAttribute",
    "GPUParticleAttributes",
    # Buffer
    "BufferAllocation",
    "GPUParticleBuffer",
    # Simulator
    "ComputeShaderConfig",
    "GPUParticleSimulator",
    # Renderer
    "RenderState",
    "GPUParticleRenderer",
    # Config
    "GPUParticleConfig",
    # Combined System
    "GPUParticleSystem",
    # =========================================================================
    # PARTICLE MODULES
    # =========================================================================
    # Enums
    "ModuleStage",
    "EmitterShape",
    "CollisionMode",
    "BlendMode",
    # Config
    "ModuleConfig",
    # Base class
    "ParticleModule",
    # Spawn modules
    "ShapeEmitter",
    "BurstEmitter",
    "RateEmitter",
    # Force modules
    "GravityModule",
    "WindModule",
    "TurbulenceModule",
    "VortexModule",
    "AttractionModule",
    "VectorFieldModule",
    "CollisionModule",
    # Attribute modules
    "SizeOverLifeModule",
    "ColorOverLifeModule",
    "RotationModule",
    "LifetimeModule",
    "VelocityModule",
    # Render modules
    "BillboardRenderer",
    "MeshParticleRenderer",
    # =========================================================================
    # VFX GRAPH
    # =========================================================================
    # Enums
    "VFXContextType",
    "VFXParameterType",
    "VFXEventTrigger",
    "VFXNodeType",
    # Context
    "VFXContext",
    # Parameter
    "VFXParameter",
    # Event
    "VFXEventConfig",
    "VFXEvent",
    # Modules
    "VFXModule",
    "VFXConnection",
    "VFXEmitterModule",
    "VFXSpawnRateModule",
    "VFXBurstModule",
    "VFXGravityModule",
    "VFXSizeOverLifeModule",
    "VFXColorOverLifeModule",
    "VFXEventModule",
    # Graph
    "VFXGraph",
    # =========================================================================
    # TRAIL RENDERER
    # =========================================================================
    # Enums
    "TextureMode",
    "TrailAlignment",
    "TrailCapStyle",
    # Configuration
    "TrailConfig",
    # Data structures
    "TrailPoint",
    "TrailBuffer",
    "TrailVertex",
    "TrailMesh",
    # Renderer
    "TrailRenderer",
    "TrailManager",
    # =========================================================================
    # DECAL SYSTEM
    # =========================================================================
    # Enums
    "DecalChannel",
    "DecalBlendMode",
    "DecalProjection",
    "DecalSortMode",
    # Configuration
    "DecalConfig",
    # Data structures
    "DecalVolume",
    "AtlasRegion",
    # Decal types
    "Decal",
    "DeferredDecal",
    # Atlas
    "DecalAtlas",
    # Sorting
    "DecalSorting",
    # System
    "DecalSystem",
    # =========================================================================
    # DETERMINISTIC EMITTER (T-CC-2.1: Fixed32 particle system)
    # =========================================================================
    # Fixed32 types
    "Fixed32Vec3",
    # Particle
    "DeterministicParticle",
    # Pool
    "DeterministicParticlePool",
    # Module base
    "DeterministicSpawnModule",
    # Spawn modules
    "DeterministicEmitterShape",
    "DeterministicShapeEmitter",
    "DeterministicRateEmitter",
    "DeterministicLifetimeModule",
    "DeterministicVelocityModule",
    "DeterministicSizeModule",
    # Update modules
    "DeterministicGravityModule",
    # Emitter
    "DeterministicEmitter",
]
