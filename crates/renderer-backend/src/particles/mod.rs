//! GPU Particle System for TRINITY Engine.
//!
//! Provides high-performance GPU-driven particle simulation including:
//! - `spawn`: GPU particle spawning from emitter configuration (T-GPU-5.1)
//! - `update`: GPU particle physics and aging (T-GPU-5.2)
//! - `compact`: GPU particle stream compaction (T-GPU-5.3)
//! - `sort`: GPU particle depth sorting for alpha blending (T-GPU-5.4)
//! - `billboard`: Billboard particle rendering (T-GPU-6.1)
//! - `mesh_particle`: 3D mesh particle rendering (T-GPU-6.2)
//! - `trail`: Trail/ribbon rendering with CPU ring buffer (T-GPU-6.3)
//! - `frame_graph`: Frame graph pass builders for particle pipeline
//!
//! # Architecture
//!
//! The particle system uses a five-phase pipeline:
//!
//! 1. **Spawn** - Initialize new particles from emitter configuration
//! 2. **Update** - Integrate velocity, apply forces, age particles
//! 3. **Compact** - Remove dead particles, defragment buffer
//! 4. **Sort** - Depth-sort particles for correct alpha blending
//! 5. **Render** - Generate camera-facing quads for rendering
//!
//! Each phase is implemented as a compute shader with corresponding
//! Rust resources and pipeline management.
//!
//! # Data Layout
//!
//! Particles use a Structure-of-Arrays (SoA) friendly layout optimized
//! for GPU cache access patterns. The `Particle` struct is 64 bytes:
//!
//! ```text
//! struct Particle {
//!     position: vec3<f32>,     // 12 bytes
//!     age: f32,                // 4 bytes
//!     velocity: vec3<f32>,     // 12 bytes
//!     lifetime: f32,           // 4 bytes
//!     color: vec4<f32>,        // 16 bytes
//!     size: f32,               // 4 bytes
//!     rotation: f32,           // 4 bytes
//!     rotation_speed: f32,     // 4 bytes
//!     flags: u32,              // 4 bytes
//! }                            // Total: 64 bytes
//! ```
//!
//! # Random Number Generation
//!
//! Uses PCG (Permuted Congruential Generator) for high-quality,
//! deterministic randomness suitable for visual effects.
//!
//! # Depth Sorting
//!
//! For correct transparency rendering, particles must be rendered back-to-front.
//! The `sort` module provides:
//! - GPU-computed depth keys from particle positions
//! - Integration with `GpuRadixSort` for efficient sorting
//! - Indirection array output (particles stay in place, indices are sorted)
//!
//! # Usage Example
//!
//! ```ignore
//! use renderer_backend::particles::{
//!     EmitterConfig, Particle, ParticleSpawnParams,
//!     ParticleSpawnPipeline, ParticleSpawnResources,
//!     ParticleSortPipeline, ParticleSortResources, ParticleSortParams,
//! };
//!
//! // Create pipelines
//! let spawn_pipeline = ParticleSpawnPipeline::new(&device);
//! let sort_pipeline = ParticleSortPipeline::new(&device, 65536);
//!
//! // Create resources
//! let spawn_resources = ParticleSpawnResources::new(
//!     &device, 65536, &spawn_pipeline.bind_group_layout,
//! );
//! let sort_resources = ParticleSortResources::new(
//!     &device, 65536, &spawn_resources.particle_buffer,
//!     &sort_pipeline.bind_group_layout,
//! );
//!
//! // Configure emitter
//! let emitter = EmitterConfig::fountain_emitter([0.0, 0.0, 0.0]);
//! spawn_resources.update_emitter(&queue, &emitter);
//!
//! // Each frame: spawn, update, compact, then sort
//! spawn_pipeline.dispatch(&mut encoder, &spawn_resources, spawn_count);
//! // ... update pass ...
//! // ... compact pass ...
//!
//! // Sort for alpha blending
//! let sort_params = ParticleSortParams::new(
//!     particle_count, camera_position, camera_forward,
//! );
//! sort_pipeline.sort(&device, &queue, &mut encoder,
//!     &sort_resources, &sort_params, particle_count);
//!
//! // Render using sorted indices
//! render_pass.set_index_buffer(sort_resources.indices_buffer.slice(..), ...);
//! ```

pub mod billboard;
pub mod compact;
pub mod frame_graph;
pub mod mesh_particle;
pub mod sort;
pub mod spawn;
pub mod trail;
pub mod update;

// Re-export frame_graph module public items (original particles.rs content)
pub use frame_graph::{
    create_particle_compact_pass,
    create_particle_render_pass,
    create_particle_spawn_pass,
    create_particle_update_pass,
    ParticleEmitter,
};

// Re-export spawn module public items
pub use spawn::{
    // Structs
    EmitterConfig,
    EmitterConfigBuilder,
    Particle,
    ParticleSpawnParams,
    ParticleSpawnPipeline,
    ParticleSpawnResources,

    // Constants
    DEFAULT_MAX_PARTICLES,
    EMITTER_CONFIG_SIZE,
    PARTICLE_FLAG_ALIVE,
    PARTICLE_SIZE,
    SPAWN_PARAMS_SIZE,
    WORKGROUP_SIZE,

    // CPU reference functions
    cpu_pcg_hash,
    cpu_random_float,
    cpu_random_in_sphere,
    cpu_random_on_sphere,
    cpu_random_range,
    cpu_spawn_particles,
};

// Re-export update module public items (T-GPU-5.2)
pub use update::{
    // Constants
    WORKGROUP_SIZE as UPDATE_WORKGROUP_SIZE,
    MAX_PARTICLES,
    PARTICLE_SIZE as UPDATE_PARTICLE_SIZE,
    FLAG_ALIVE,
    FORCE_FLAG_GRAVITY,
    FORCE_FLAG_WIND,
    FORCE_FLAG_TURBULENCE,
    FORCE_FLAG_VORTEX,
    FORCE_FLAG_ATTRACTION,

    // Structs
    ParticleUpdateParams,
    ForceConfig,
    ColorParams,
    ParticleUpdateResources,
    ParticleUpdatePipeline,
    UpdateMode,

    // Note: Particle struct is re-exported from spawn module (same layout)

    // CPU reference functions
    simplex_noise_3d,
    cpu_apply_gravity,
    cpu_apply_drag,
    cpu_apply_wind,
    cpu_interpolate_color,
    cpu_interpolate_size,
    cpu_particle_update,
};

// Re-export compact module public items (T-GPU-5.3)
pub use compact::{
    // Structs
    DrawIndirectArgs,
    ParticleCompactParams,
    ParticleCompactPipeline,
    ParticleCompactResources,

    // Constants
    MAX_BLOCKS_SIMPLE,
    MAX_PARTICLES_SIMPLE,
    SINGLE_PASS_MAX,

    // CPU reference functions
    cpu_compact_particles,
    cpu_prefix_sum,
    extract_alive_flags,
};

// Re-export sort module public items (T-GPU-5.4)
pub use sort::{
    // Structs
    ParticleSortParams,
    ParticleSortPipeline,
    ParticleSortResources,

    // Enums
    SortMode,

    // Constants
    DEAD_PARTICLE_KEY,
    MIN_GPU_PARTICLES,
    SORT_PARAMS_SIZE,
    WORKGROUP_SIZE as SORT_WORKGROUP_SIZE,

    // CPU reference functions
    cpu_compute_depth_key,
    cpu_compute_distance_key,
    cpu_radix_sort_particles,
    cpu_sort_particles,
    verify_back_to_front,
};

// Re-export billboard module public items (T-GPU-6.1)
pub use billboard::{
    // Structs
    BillboardParams,
    BillboardPipeline,
    BillboardResources,

    // Enums
    AlignmentMode,
    BlendMode,

    // Constants
    BILLBOARD_PARAMS_SIZE,
    DEFAULT_TEXTURE_SIZE,
    VERTICES_PER_PARTICLE,

    // CPU reference functions
    cpu_calculate_lifetime_alpha,
    cpu_calculate_uv,
    cpu_generate_billboard_quad,
};

// Re-export mesh_particle module public items (T-GPU-6.2)
pub use mesh_particle::{
    // Structs
    MeshParticleParams,
    MeshParticlePipeline,
    MeshParticleResources,

    // Constants
    MESH_PARTICLE_PARAMS_SIZE,
    ROTATION_MODE_VELOCITY_ALIGNED,
    ROTATION_MODE_Y_UP,
    SCALE_MODE_FROM_SIZE,
    SCALE_MODE_UNIFORM,

    // CPU reference functions
    cpu_build_particle_transform,
    cpu_transform_normal,
    cpu_transform_vertex,
};

// Re-export trail module public items (T-GPU-6.3)
pub use trail::{
    // Structs
    TrailBuffer,
    TrailConfig,
    TrailParams,
    TrailPipeline,
    TrailPoint,
    TrailResources,

    // Enums
    CapStyle,
    UvMode,

    // Constants
    DEFAULT_FADE_TIME,
    DEFAULT_MAX_TRAIL_POINTS,
    DEFAULT_MIN_POINT_DISTANCE,
    DEFAULT_TRAIL_WIDTH,
    TRAIL_PARAMS_SIZE,
    TRAIL_POINT_SIZE,
};
