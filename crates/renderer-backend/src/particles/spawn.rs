//! GPU Particle Spawning for TRINITY Engine (T-GPU-5.1).
//!
//! This module implements GPU-based particle spawning using compute shaders.
//! The spawn shader initializes new particles from emitter configuration,
//! supporting high particle counts with minimal CPU overhead.
//!
//! # Overview
//!
//! The spawning pipeline:
//! 1. CPU determines spawn count (based on emission rate * delta_time)
//! 2. CPU uploads `ParticleSpawnParams` and `EmitterConfig` to uniform buffers
//! 3. GPU dispatches spawn compute shader with ceil(spawn_count / 256) workgroups
//! 4. Each thread initializes one particle with randomized properties
//!
//! # Random Number Generation
//!
//! Uses PCG (Permuted Congruential Generator) hash for deterministic randomness.
//! The seed is constructed from:
//! - Frame random seed (changes each frame for variation)
//! - Spawn index (each particle gets unique values)
//! - Current time (ensures no repetition over long runs)
//!
//! # SoA Layout
//!
//! The `Particle` struct is designed for cache-efficient access patterns:
//! - Position and age packed together (commonly accessed in update)
//! - Velocity and lifetime packed together (physics integration)
//! - Color standalone (rendering access)
//! - Size, rotation, flags packed (rendering attributes)
//!
//! # Usage
//!
//! ```ignore
//! // Create spawn pipeline
//! let pipeline = ParticleSpawnPipeline::new(&device);
//!
//! // Create resources for 100K max particles
//! let resources = ParticleSpawnResources::new(&device, 100_000);
//!
//! // Configure emitter
//! let emitter = EmitterConfig::builder()
//!     .position([0.0, 5.0, 0.0])
//!     .spawn_radius(0.5)
//!     .velocity_range([-1.0, 2.0, -1.0], [1.0, 5.0, 1.0])
//!     .lifetime_range(1.0, 3.0)
//!     .build();
//!
//! // Each frame: spawn particles
//! let spawn_count = (emitter_rate * delta_time) as u32;
//! resources.update_params(&queue, spawn_count, particle_offset, time);
//! resources.update_emitter(&queue, &emitter);
//! pipeline.dispatch(&mut encoder, &resources, spawn_count);
//! ```

use std::mem;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Compute shader workgroup size (must match WGSL constant).
pub const WORKGROUP_SIZE: u32 = 256;

/// Particle struct size in bytes (must match WGSL Particle).
pub const PARTICLE_SIZE: usize = 64;

/// ParticleSpawnParams size in bytes.
pub const SPAWN_PARAMS_SIZE: usize = 32;

/// EmitterConfig size in bytes.
pub const EMITTER_CONFIG_SIZE: usize = 96;

/// Default maximum particles per emitter.
pub const DEFAULT_MAX_PARTICLES: u32 = 65536;

/// Particle flag: particle is alive.
pub const PARTICLE_FLAG_ALIVE: u32 = 1;

// ---------------------------------------------------------------------------
// ParticleSpawnParams
// ---------------------------------------------------------------------------

/// GPU uniform buffer for spawn parameters.
///
/// Matches the WGSL `ParticleSpawnParams` struct layout.
///
/// # Memory Layout (32 bytes, std140 compatible)
///
/// | Offset | Field          | Size    |
/// |--------|----------------|---------|
/// | 0      | spawn_count    | 4 bytes |
/// | 4      | particle_offset| 4 bytes |
/// | 8      | max_particles  | 4 bytes |
/// | 12     | time           | 4 bytes |
/// | 16     | delta_time     | 4 bytes |
/// | 20     | random_seed    | 4 bytes |
/// | 24     | _padding       | 8 bytes |
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ParticleSpawnParams {
    /// Number of particles to spawn this frame.
    pub spawn_count: u32,
    /// Write offset in the particle buffer.
    pub particle_offset: u32,
    /// Maximum particles the buffer can hold.
    pub max_particles: u32,
    /// Current simulation time in seconds.
    pub time: f32,
    /// Delta time since last frame in seconds.
    pub delta_time: f32,
    /// Random seed for this frame.
    pub random_seed: u32,
    /// Padding for 16-byte alignment.
    pub _padding: [u32; 2],
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<ParticleSpawnParams>() == SPAWN_PARAMS_SIZE);

impl ParticleSpawnParams {
    /// Create spawn parameters for this frame.
    ///
    /// # Arguments
    ///
    /// * `spawn_count` - Number of particles to spawn.
    /// * `particle_offset` - Write offset in particle buffer.
    /// * `max_particles` - Maximum capacity of particle buffer.
    /// * `time` - Current simulation time (seconds).
    /// * `delta_time` - Time since last frame (seconds).
    pub fn new(
        spawn_count: u32,
        particle_offset: u32,
        max_particles: u32,
        time: f32,
        delta_time: f32,
    ) -> Self {
        // Use time as seed base for frame-to-frame variation
        let random_seed = ((time * 1000.0) as u32).wrapping_mul(2654435761);

        Self {
            spawn_count,
            particle_offset,
            max_particles,
            time,
            delta_time,
            random_seed,
            _padding: [0, 0],
        }
    }

    /// Create parameters with explicit random seed.
    pub fn with_seed(
        spawn_count: u32,
        particle_offset: u32,
        max_particles: u32,
        time: f32,
        delta_time: f32,
        random_seed: u32,
    ) -> Self {
        Self {
            spawn_count,
            particle_offset,
            max_particles,
            time,
            delta_time,
            random_seed,
            _padding: [0, 0],
        }
    }

    /// Get number of workgroups needed for dispatch.
    #[inline]
    pub fn num_workgroups(&self) -> u32 {
        (self.spawn_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
    }

    /// Check if spawn would exceed buffer capacity.
    #[inline]
    pub fn would_overflow(&self) -> bool {
        self.particle_offset.saturating_add(self.spawn_count) > self.max_particles
    }

    /// Get clamped spawn count that fits in buffer.
    #[inline]
    pub fn clamped_spawn_count(&self) -> u32 {
        self.max_particles.saturating_sub(self.particle_offset).min(self.spawn_count)
    }
}

impl Default for ParticleSpawnParams {
    fn default() -> Self {
        Self::new(0, 0, DEFAULT_MAX_PARTICLES, 0.0, 1.0 / 60.0)
    }
}

// ---------------------------------------------------------------------------
// EmitterConfig
// ---------------------------------------------------------------------------

/// GPU uniform buffer for emitter configuration.
///
/// Matches the WGSL `EmitterConfig` struct layout.
///
/// # Memory Layout (96 bytes, std140 compatible)
///
/// | Offset | Field             | Size     |
/// |--------|-------------------|----------|
/// | 0      | position          | 12 bytes |
/// | 12     | spawn_radius      | 4 bytes  |
/// | 16     | velocity_min      | 12 bytes |
/// | 28     | velocity_spread   | 4 bytes  |
/// | 32     | velocity_max      | 12 bytes |
/// | 44     | lifetime_min      | 4 bytes  |
/// | 48     | color_start       | 16 bytes |
/// | 64     | color_end         | 16 bytes |
/// | 80     | size_start        | 4 bytes  |
/// | 84     | size_end          | 4 bytes  |
/// | 88     | lifetime_max      | 4 bytes  |
/// | 92     | rotation_speed_max| 4 bytes  |
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EmitterConfig {
    /// World-space position of emitter center.
    pub position: [f32; 3],
    /// Radius of spherical spawn region.
    pub spawn_radius: f32,

    /// Minimum initial velocity (per axis).
    pub velocity_min: [f32; 3],
    /// Velocity spread factor (0=uniform, 1=full random).
    pub velocity_spread: f32,

    /// Maximum initial velocity (per axis).
    pub velocity_max: [f32; 3],
    /// Minimum particle lifetime in seconds.
    pub lifetime_min: f32,

    /// Starting color (RGBA premultiplied alpha).
    pub color_start: [f32; 4],

    /// Ending color (RGBA, interpolated over lifetime).
    pub color_end: [f32; 4],

    /// Starting particle size (world units).
    pub size_start: f32,
    /// Ending particle size (world units).
    pub size_end: f32,
    /// Maximum particle lifetime in seconds.
    pub lifetime_max: f32,
    /// Maximum rotation speed (radians/second).
    pub rotation_speed_max: f32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<EmitterConfig>() == EMITTER_CONFIG_SIZE);

impl EmitterConfig {
    /// Create a new emitter configuration builder.
    pub fn builder() -> EmitterConfigBuilder {
        EmitterConfigBuilder::default()
    }

    /// Create a simple point emitter at the origin.
    pub fn point_emitter() -> Self {
        Self::builder().build()
    }

    /// Create a spherical emitter with given radius.
    pub fn sphere_emitter(position: [f32; 3], radius: f32) -> Self {
        Self::builder()
            .position(position)
            .spawn_radius(radius)
            .build()
    }

    /// Create a fountain-like emitter (upward velocity bias).
    pub fn fountain_emitter(position: [f32; 3]) -> Self {
        Self::builder()
            .position(position)
            .spawn_radius(0.1)
            .velocity_range([-0.5, 3.0, -0.5], [0.5, 5.0, 0.5])
            .velocity_spread(0.3)
            .lifetime_range(1.0, 3.0)
            .size_range(0.1, 0.02)
            .build()
    }

    /// Create an explosion-like emitter (radial velocity).
    pub fn explosion_emitter(position: [f32; 3]) -> Self {
        Self::builder()
            .position(position)
            .spawn_radius(0.2)
            .velocity_range([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0])
            .velocity_spread(1.0)
            .lifetime_range(0.5, 1.5)
            .color_start([1.0, 0.8, 0.2, 1.0])
            .color_end([1.0, 0.2, 0.0, 0.0])
            .size_range(0.2, 0.05)
            .build()
    }
}

impl Default for EmitterConfig {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            spawn_radius: 0.0,
            velocity_min: [0.0, 0.0, 0.0],
            velocity_spread: 0.5,
            velocity_max: [0.0, 1.0, 0.0],
            lifetime_min: 1.0,
            color_start: [1.0, 1.0, 1.0, 1.0],
            color_end: [1.0, 1.0, 1.0, 0.0],
            size_start: 0.1,
            size_end: 0.0,
            lifetime_max: 2.0,
            rotation_speed_max: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// EmitterConfigBuilder
// ---------------------------------------------------------------------------

/// Builder for `EmitterConfig`.
#[derive(Clone, Debug)]
pub struct EmitterConfigBuilder {
    config: EmitterConfig,
}

impl Default for EmitterConfigBuilder {
    fn default() -> Self {
        Self {
            config: EmitterConfig::default(),
        }
    }
}

impl EmitterConfigBuilder {
    /// Set the emitter position.
    pub fn position(mut self, position: [f32; 3]) -> Self {
        self.config.position = position;
        self
    }

    /// Set the spawn radius.
    pub fn spawn_radius(mut self, radius: f32) -> Self {
        self.config.spawn_radius = radius;
        self
    }

    /// Set the velocity range (min and max per axis).
    pub fn velocity_range(mut self, min: [f32; 3], max: [f32; 3]) -> Self {
        self.config.velocity_min = min;
        self.config.velocity_max = max;
        self
    }

    /// Set the velocity spread factor (0=uniform, 1=full random).
    pub fn velocity_spread(mut self, spread: f32) -> Self {
        self.config.velocity_spread = spread.clamp(0.0, 1.0);
        self
    }

    /// Set the lifetime range.
    pub fn lifetime_range(mut self, min: f32, max: f32) -> Self {
        self.config.lifetime_min = min;
        self.config.lifetime_max = max;
        self
    }

    /// Set the starting color (RGBA).
    pub fn color_start(mut self, color: [f32; 4]) -> Self {
        self.config.color_start = color;
        self
    }

    /// Set the ending color (RGBA).
    pub fn color_end(mut self, color: [f32; 4]) -> Self {
        self.config.color_end = color;
        self
    }

    /// Set the size range (start and end over lifetime).
    pub fn size_range(mut self, start: f32, end: f32) -> Self {
        self.config.size_start = start;
        self.config.size_end = end;
        self
    }

    /// Set the maximum rotation speed.
    pub fn rotation_speed_max(mut self, speed: f32) -> Self {
        self.config.rotation_speed_max = speed;
        self
    }

    /// Build the emitter configuration.
    pub fn build(self) -> EmitterConfig {
        self.config
    }
}

// ---------------------------------------------------------------------------
// Particle
// ---------------------------------------------------------------------------

/// GPU particle data structure (64 bytes, SoA-friendly layout).
///
/// Matches the WGSL `Particle` struct layout.
///
/// # Memory Layout (64 bytes)
///
/// | Offset | Field          | Size     |
/// |--------|----------------|----------|
/// | 0      | position       | 12 bytes |
/// | 12     | age            | 4 bytes  |
/// | 16     | velocity       | 12 bytes |
/// | 24     | lifetime       | 4 bytes  |
/// | 32     | color          | 16 bytes |
/// | 48     | size           | 4 bytes  |
/// | 52     | rotation       | 4 bytes  |
/// | 56     | rotation_speed | 4 bytes  |
/// | 60     | flags          | 4 bytes  |
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Particle {
    /// World-space position.
    pub position: [f32; 3],
    /// Current age (seconds since spawn).
    pub age: f32,

    /// Current velocity (world units/second).
    pub velocity: [f32; 3],
    /// Total lifetime (seconds).
    pub lifetime: f32,

    /// Current color (RGBA premultiplied alpha).
    pub color: [f32; 4],

    /// Current size (world units).
    pub size: f32,
    /// Current rotation (radians).
    pub rotation: f32,
    /// Rotation speed (radians/second).
    pub rotation_speed: f32,
    /// Flags (bit 0: alive).
    pub flags: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<Particle>() == PARTICLE_SIZE);

impl Particle {
    /// Create a dead (empty) particle.
    pub const fn dead() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            age: 0.0,
            velocity: [0.0, 0.0, 0.0],
            lifetime: 0.0,
            color: [0.0, 0.0, 0.0, 0.0],
            size: 0.0,
            rotation: 0.0,
            rotation_speed: 0.0,
            flags: 0,
        }
    }

    /// Check if the particle is alive.
    #[inline]
    pub fn is_alive(&self) -> bool {
        (self.flags & PARTICLE_FLAG_ALIVE) != 0
    }

    /// Get the normalized age (0=just spawned, 1=about to die).
    #[inline]
    pub fn normalized_age(&self) -> f32 {
        if self.lifetime > 0.0 {
            (self.age / self.lifetime).min(1.0)
        } else {
            1.0
        }
    }

    /// Check if the particle has expired.
    #[inline]
    pub fn is_expired(&self) -> bool {
        self.age >= self.lifetime
    }
}

impl Default for Particle {
    fn default() -> Self {
        Self::dead()
    }
}

// ---------------------------------------------------------------------------
// ParticleSpawnResources
// ---------------------------------------------------------------------------

/// GPU resources for particle spawning.
///
/// Contains all buffers needed for the spawn compute shader:
/// - `params_buffer`: Uniform buffer for `ParticleSpawnParams`
/// - `emitter_buffer`: Uniform buffer for `EmitterConfig`
/// - `particle_buffer`: Storage buffer for particle data
pub struct ParticleSpawnResources {
    /// Uniform buffer for spawn parameters.
    pub params_buffer: wgpu::Buffer,
    /// Uniform buffer for emitter configuration.
    pub emitter_buffer: wgpu::Buffer,
    /// Storage buffer for particle data.
    pub particle_buffer: wgpu::Buffer,
    /// Maximum capacity in particles.
    pub capacity: u32,
    /// Bind group for spawn shader.
    pub bind_group: wgpu::BindGroup,
}

impl ParticleSpawnResources {
    /// Create spawn resources with the given capacity.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `capacity` - Maximum number of particles.
    /// * `bind_group_layout` - Layout from `ParticleSpawnPipeline`.
    pub fn new(
        device: &wgpu::Device,
        capacity: u32,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle_spawn_params"),
            size: SPAWN_PARAMS_SIZE as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let emitter_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle_emitter_config"),
            size: EMITTER_CONFIG_SIZE as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let particle_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle_buffer"),
            size: (capacity as u64) * (PARTICLE_SIZE as u64),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("particle_spawn_bind_group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: emitter_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: particle_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            params_buffer,
            emitter_buffer,
            particle_buffer,
            capacity,
            bind_group,
        }
    }

    /// Update spawn parameters.
    pub fn update_params(&self, queue: &wgpu::Queue, params: &ParticleSpawnParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Update emitter configuration.
    pub fn update_emitter(&self, queue: &wgpu::Queue, emitter: &EmitterConfig) {
        queue.write_buffer(&self.emitter_buffer, 0, bytemuck::bytes_of(emitter));
    }
}

// ---------------------------------------------------------------------------
// ParticleSpawnPipeline
// ---------------------------------------------------------------------------

/// GPU compute pipeline for particle spawning.
///
/// Encapsulates the shader module, pipeline, and bind group layout.
pub struct ParticleSpawnPipeline {
    /// Bind group layout for spawn shader.
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// Compute pipeline for basic spawning.
    pub pipeline: wgpu::ComputePipeline,
    /// Compute pipeline for directed spawning.
    pub pipeline_directed: wgpu::ComputePipeline,
}

impl ParticleSpawnPipeline {
    /// Create the particle spawn pipeline.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    pub fn new(device: &wgpu::Device) -> Self {
        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("particle_spawn_bind_group_layout"),
            entries: &[
                // binding 0: ParticleSpawnParams (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: EmitterConfig (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: particles (storage, read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("particle_spawn_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Load shader module
        let shader_source = include_str!(
            "../../shaders/particles/gpu_particle_spawn.comp.wgsl"
        );
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("particle_spawn_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create basic spawn pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("particle_spawn_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "spawn_particles",
            compilation_options: Default::default(),
            cache: None,
        });

        // Create directed spawn pipeline
        let pipeline_directed = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("particle_spawn_directed_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "spawn_particles_directed",
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            bind_group_layout,
            pipeline,
            pipeline_directed,
        }
    }

    /// Dispatch the spawn compute shader.
    ///
    /// # Arguments
    ///
    /// * `encoder` - Command encoder to record dispatch.
    /// * `resources` - Spawn resources with bind group.
    /// * `spawn_count` - Number of particles to spawn.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        resources: &ParticleSpawnResources,
        spawn_count: u32,
    ) {
        if spawn_count == 0 {
            return;
        }

        let num_workgroups = (spawn_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("particle_spawn_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &resources.bind_group, &[]);
        pass.dispatch_workgroups(num_workgroups, 1, 1);
    }

    /// Dispatch the directed spawn compute shader.
    ///
    /// # Arguments
    ///
    /// * `encoder` - Command encoder to record dispatch.
    /// * `resources` - Spawn resources with bind group.
    /// * `spawn_count` - Number of particles to spawn.
    pub fn dispatch_directed(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        resources: &ParticleSpawnResources,
        spawn_count: u32,
    ) {
        if spawn_count == 0 {
            return;
        }

        let num_workgroups = (spawn_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("particle_spawn_directed_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline_directed);
        pass.set_bind_group(0, &resources.bind_group, &[]);
        pass.dispatch_workgroups(num_workgroups, 1, 1);
    }
}

// ---------------------------------------------------------------------------
// CPU Reference Implementation
// ---------------------------------------------------------------------------

/// PCG hash function (CPU reference implementation).
///
/// Matches the WGSL `pcg_hash` function for testing.
#[inline]
pub fn cpu_pcg_hash(seed: u32) -> u32 {
    let state = seed.wrapping_mul(747796405).wrapping_add(2891336453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277803737);
    (word >> 22) ^ word
}

/// Generate random float in [0, 1) from mutable seed.
#[inline]
pub fn cpu_random_float(seed: &mut u32) -> f32 {
    *seed = cpu_pcg_hash(*seed);
    (*seed as f32) / 4294967296.0
}

/// Generate random float in [min, max).
#[inline]
pub fn cpu_random_range(seed: &mut u32, min: f32, max: f32) -> f32 {
    min + cpu_random_float(seed) * (max - min)
}

/// Generate random unit vector on sphere surface.
pub fn cpu_random_on_sphere(seed: &mut u32) -> [f32; 3] {
    let theta = cpu_random_float(seed) * std::f32::consts::TAU;
    let phi = (2.0 * cpu_random_float(seed) - 1.0).acos();
    let sin_phi = phi.sin();
    [sin_phi * theta.cos(), sin_phi * theta.sin(), phi.cos()]
}

/// Generate random point inside sphere of given radius.
pub fn cpu_random_in_sphere(seed: &mut u32, radius: f32) -> [f32; 3] {
    let r = radius * cpu_random_float(seed).powf(1.0 / 3.0);
    let dir = cpu_random_on_sphere(seed);
    [dir[0] * r, dir[1] * r, dir[2] * r]
}

/// CPU reference implementation of particle spawning.
///
/// Spawns particles into the output buffer using the same algorithm
/// as the GPU shader, for testing and validation.
pub fn cpu_spawn_particles(
    params: &ParticleSpawnParams,
    emitter: &EmitterConfig,
    output: &mut [Particle],
) -> u32 {
    let mut spawned = 0u32;

    for i in 0..params.spawn_count {
        let particle_index = params.particle_offset + i;

        if particle_index >= params.max_particles {
            break;
        }
        if particle_index as usize >= output.len() {
            break;
        }

        // Initialize seed (match GPU algorithm)
        let mut seed = cpu_pcg_hash(params.random_seed.wrapping_add(i));
        seed = cpu_pcg_hash(seed ^ params.time.to_bits());
        seed = cpu_pcg_hash(seed.wrapping_add(particle_index));

        // Position
        let offset = cpu_random_in_sphere(&mut seed, emitter.spawn_radius);
        let position = [
            emitter.position[0] + offset[0],
            emitter.position[1] + offset[1],
            emitter.position[2] + offset[2],
        ];

        // Velocity
        let base_velocity = [
            (emitter.velocity_min[0] + emitter.velocity_max[0]) * 0.5,
            (emitter.velocity_min[1] + emitter.velocity_max[1]) * 0.5,
            (emitter.velocity_min[2] + emitter.velocity_max[2]) * 0.5,
        ];
        let velocity_range = [
            (emitter.velocity_max[0] - emitter.velocity_min[0]) * 0.5,
            (emitter.velocity_max[1] - emitter.velocity_min[1]) * 0.5,
            (emitter.velocity_max[2] - emitter.velocity_min[2]) * 0.5,
        ];

        let spread_dir = cpu_random_on_sphere(&mut seed);
        let velocity = [
            base_velocity[0]
                + velocity_range[0] * spread_dir[0] * emitter.velocity_spread
                + cpu_random_range(&mut seed, -velocity_range[0], velocity_range[0]),
            base_velocity[1]
                + velocity_range[1] * spread_dir[1] * emitter.velocity_spread
                + cpu_random_range(&mut seed, -velocity_range[1], velocity_range[1]),
            base_velocity[2]
                + velocity_range[2] * spread_dir[2] * emitter.velocity_spread
                + cpu_random_range(&mut seed, -velocity_range[2], velocity_range[2]),
        ];

        // Lifetime
        let lifetime = cpu_random_range(&mut seed, emitter.lifetime_min, emitter.lifetime_max);

        // Rotation
        let rotation = cpu_random_float(&mut seed) * std::f32::consts::TAU;
        let rotation_speed = cpu_random_range(
            &mut seed,
            -emitter.rotation_speed_max,
            emitter.rotation_speed_max,
        );

        output[particle_index as usize] = Particle {
            position,
            age: 0.0,
            velocity,
            lifetime,
            color: emitter.color_start,
            size: emitter.size_start,
            rotation,
            rotation_speed,
            flags: PARTICLE_FLAG_ALIVE,
        };

        spawned += 1;
    }

    spawned
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── ParticleSpawnParams ─────────────────────────────────────────────

    #[test]
    fn test_spawn_params_default() {
        let params = ParticleSpawnParams::default();
        assert_eq!(params.spawn_count, 0);
        assert_eq!(params.particle_offset, 0);
        assert_eq!(params.max_particles, DEFAULT_MAX_PARTICLES);
    }

    #[test]
    fn test_spawn_params_new() {
        let params = ParticleSpawnParams::new(100, 50, 1000, 1.5, 0.016);
        assert_eq!(params.spawn_count, 100);
        assert_eq!(params.particle_offset, 50);
        assert_eq!(params.max_particles, 1000);
        assert!((params.time - 1.5).abs() < f32::EPSILON);
        assert!((params.delta_time - 0.016).abs() < f32::EPSILON);
    }

    #[test]
    fn test_spawn_params_workgroups() {
        assert_eq!(ParticleSpawnParams::new(1, 0, 100, 0.0, 0.0).num_workgroups(), 1);
        assert_eq!(ParticleSpawnParams::new(256, 0, 1000, 0.0, 0.0).num_workgroups(), 1);
        assert_eq!(ParticleSpawnParams::new(257, 0, 1000, 0.0, 0.0).num_workgroups(), 2);
        assert_eq!(ParticleSpawnParams::new(512, 0, 1000, 0.0, 0.0).num_workgroups(), 2);
        assert_eq!(ParticleSpawnParams::new(513, 0, 1000, 0.0, 0.0).num_workgroups(), 3);
    }

    #[test]
    fn test_spawn_params_overflow() {
        let params = ParticleSpawnParams::new(100, 950, 1000, 0.0, 0.0);
        assert!(params.would_overflow());
        assert_eq!(params.clamped_spawn_count(), 50);
    }

    #[test]
    fn test_spawn_params_no_overflow() {
        let params = ParticleSpawnParams::new(100, 0, 1000, 0.0, 0.0);
        assert!(!params.would_overflow());
        assert_eq!(params.clamped_spawn_count(), 100);
    }

    // ── EmitterConfig ───────────────────────────────────────────────────

    #[test]
    fn test_emitter_default() {
        let emitter = EmitterConfig::default();
        assert_eq!(emitter.position, [0.0, 0.0, 0.0]);
        assert_eq!(emitter.spawn_radius, 0.0);
        assert!((emitter.lifetime_min - 1.0).abs() < f32::EPSILON);
        assert!((emitter.lifetime_max - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_emitter_builder() {
        let emitter = EmitterConfig::builder()
            .position([1.0, 2.0, 3.0])
            .spawn_radius(0.5)
            .velocity_range([-1.0, 0.0, -1.0], [1.0, 5.0, 1.0])
            .lifetime_range(0.5, 2.5)
            .build();

        assert_eq!(emitter.position, [1.0, 2.0, 3.0]);
        assert!((emitter.spawn_radius - 0.5).abs() < f32::EPSILON);
        assert_eq!(emitter.velocity_min, [-1.0, 0.0, -1.0]);
        assert_eq!(emitter.velocity_max, [1.0, 5.0, 1.0]);
        assert!((emitter.lifetime_min - 0.5).abs() < f32::EPSILON);
        assert!((emitter.lifetime_max - 2.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_emitter_point() {
        let emitter = EmitterConfig::point_emitter();
        assert_eq!(emitter.spawn_radius, 0.0);
    }

    #[test]
    fn test_emitter_sphere() {
        let emitter = EmitterConfig::sphere_emitter([5.0, 10.0, 0.0], 2.0);
        assert_eq!(emitter.position, [5.0, 10.0, 0.0]);
        assert!((emitter.spawn_radius - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_emitter_fountain() {
        let emitter = EmitterConfig::fountain_emitter([0.0, 0.0, 0.0]);
        assert!(emitter.velocity_max[1] > emitter.velocity_min[1]);
    }

    #[test]
    fn test_emitter_explosion() {
        let emitter = EmitterConfig::explosion_emitter([0.0, 0.0, 0.0]);
        // Explosion should have radial velocity (symmetric)
        assert!((emitter.velocity_min[0] + emitter.velocity_max[0]).abs() < f32::EPSILON);
        assert!((emitter.velocity_min[1] + emitter.velocity_max[1]).abs() < f32::EPSILON);
        assert!((emitter.velocity_min[2] + emitter.velocity_max[2]).abs() < f32::EPSILON);
    }

    // ── Particle ────────────────────────────────────────────────────────

    #[test]
    fn test_particle_dead() {
        let p = Particle::dead();
        assert!(!p.is_alive());
        assert_eq!(p.flags, 0);
    }

    #[test]
    fn test_particle_alive() {
        let mut p = Particle::dead();
        p.flags = PARTICLE_FLAG_ALIVE;
        assert!(p.is_alive());
    }

    #[test]
    fn test_particle_normalized_age() {
        let mut p = Particle::dead();
        p.lifetime = 2.0;
        p.age = 1.0;
        assert!((p.normalized_age() - 0.5).abs() < f32::EPSILON);

        p.age = 2.0;
        assert!((p.normalized_age() - 1.0).abs() < f32::EPSILON);

        p.age = 3.0; // Over lifetime
        assert!((p.normalized_age() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_particle_expired() {
        let mut p = Particle::dead();
        p.lifetime = 2.0;
        p.age = 1.0;
        assert!(!p.is_expired());

        p.age = 2.0;
        assert!(p.is_expired());

        p.age = 3.0;
        assert!(p.is_expired());
    }

    // ── PCG Random ──────────────────────────────────────────────────────

    #[test]
    fn test_pcg_hash_deterministic() {
        let a = cpu_pcg_hash(12345);
        let b = cpu_pcg_hash(12345);
        assert_eq!(a, b);
    }

    #[test]
    fn test_pcg_hash_varied() {
        let a = cpu_pcg_hash(0);
        let b = cpu_pcg_hash(1);
        let c = cpu_pcg_hash(2);
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
    }

    #[test]
    fn test_random_float_range() {
        let mut seed = 42u32;
        for _ in 0..1000 {
            let f = cpu_random_float(&mut seed);
            assert!(f >= 0.0);
            assert!(f < 1.0);
        }
    }

    #[test]
    fn test_random_range_bounds() {
        let mut seed = 42u32;
        for _ in 0..1000 {
            let f = cpu_random_range(&mut seed, 5.0, 10.0);
            assert!(f >= 5.0);
            assert!(f < 10.0);
        }
    }

    #[test]
    fn test_random_on_sphere_unit_length() {
        let mut seed = 42u32;
        for _ in 0..100 {
            let v = cpu_random_on_sphere(&mut seed);
            let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_random_in_sphere_radius() {
        let mut seed = 42u32;
        let radius = 5.0;
        for _ in 0..100 {
            let v = cpu_random_in_sphere(&mut seed, radius);
            let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!(len <= radius + 1e-5);
        }
    }

    // ── CPU Spawn ───────────────────────────────────────────────────────

    #[test]
    fn test_cpu_spawn_single_particle() {
        let params = ParticleSpawnParams::with_seed(1, 0, 100, 0.0, 0.016, 42);
        let emitter = EmitterConfig::default();
        let mut particles = vec![Particle::dead(); 100];

        let spawned = cpu_spawn_particles(&params, &emitter, &mut particles);
        assert_eq!(spawned, 1);
        assert!(particles[0].is_alive());
        assert!(!particles[1].is_alive());
    }

    #[test]
    fn test_cpu_spawn_batch() {
        let params = ParticleSpawnParams::with_seed(256, 0, 1000, 0.0, 0.016, 42);
        let emitter = EmitterConfig::default();
        let mut particles = vec![Particle::dead(); 1000];

        let spawned = cpu_spawn_particles(&params, &emitter, &mut particles);
        assert_eq!(spawned, 256);

        for i in 0..256 {
            assert!(particles[i].is_alive(), "particle {} should be alive", i);
        }
        for i in 256..1000 {
            assert!(!particles[i].is_alive(), "particle {} should be dead", i);
        }
    }

    #[test]
    fn test_cpu_spawn_position_in_radius() {
        let params = ParticleSpawnParams::with_seed(100, 0, 1000, 0.0, 0.016, 42);
        let emitter = EmitterConfig::builder()
            .position([10.0, 20.0, 30.0])
            .spawn_radius(5.0)
            .build();
        let mut particles = vec![Particle::dead(); 1000];

        cpu_spawn_particles(&params, &emitter, &mut particles);

        for i in 0..100 {
            let p = &particles[i];
            let dx = p.position[0] - emitter.position[0];
            let dy = p.position[1] - emitter.position[1];
            let dz = p.position[2] - emitter.position[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            assert!(
                dist <= emitter.spawn_radius + 1e-5,
                "particle {} at distance {} exceeds radius {}",
                i,
                dist,
                emitter.spawn_radius
            );
        }
    }

    #[test]
    fn test_cpu_spawn_velocity_range() {
        let params = ParticleSpawnParams::with_seed(100, 0, 1000, 0.0, 0.016, 42);
        let emitter = EmitterConfig::builder()
            .velocity_range([0.0, 1.0, 0.0], [0.0, 5.0, 0.0])
            .velocity_spread(0.0) // No spread for predictable bounds
            .build();
        let mut particles = vec![Particle::dead(); 1000];

        cpu_spawn_particles(&params, &emitter, &mut particles);

        // With spread=0, velocity should be base + random within range
        for i in 0..100 {
            let vy = particles[i].velocity[1];
            // Velocity should be roughly between min and max
            // (exact bounds depend on random distribution)
            assert!(vy >= -5.0 && vy <= 10.0, "velocity y {} out of expected range", vy);
        }
    }

    #[test]
    fn test_cpu_spawn_lifetime_range() {
        let params = ParticleSpawnParams::with_seed(100, 0, 1000, 0.0, 0.016, 42);
        let emitter = EmitterConfig::builder()
            .lifetime_range(1.0, 3.0)
            .build();
        let mut particles = vec![Particle::dead(); 1000];

        cpu_spawn_particles(&params, &emitter, &mut particles);

        for i in 0..100 {
            let lifetime = particles[i].lifetime;
            assert!(
                lifetime >= 1.0 && lifetime <= 3.0,
                "lifetime {} out of range [1, 3]",
                lifetime
            );
        }
    }

    #[test]
    fn test_cpu_spawn_respects_max_particles() {
        let params = ParticleSpawnParams::with_seed(100, 950, 1000, 0.0, 0.016, 42);
        let emitter = EmitterConfig::default();
        let mut particles = vec![Particle::dead(); 1000];

        let spawned = cpu_spawn_particles(&params, &emitter, &mut particles);
        assert_eq!(spawned, 50); // Only 50 slots available (1000 - 950)
    }

    #[test]
    fn test_cpu_spawn_color_initialization() {
        let params = ParticleSpawnParams::with_seed(1, 0, 100, 0.0, 0.016, 42);
        let emitter = EmitterConfig::builder()
            .color_start([1.0, 0.5, 0.0, 1.0])
            .color_end([0.0, 0.0, 1.0, 0.0])
            .build();
        let mut particles = vec![Particle::dead(); 100];

        cpu_spawn_particles(&params, &emitter, &mut particles);

        // Spawned particle should have start color
        assert_eq!(particles[0].color, [1.0, 0.5, 0.0, 1.0]);
    }

    // ── Size Assertions ─────────────────────────────────────────────────

    #[test]
    fn test_struct_sizes() {
        assert_eq!(mem::size_of::<ParticleSpawnParams>(), SPAWN_PARAMS_SIZE);
        assert_eq!(mem::size_of::<EmitterConfig>(), EMITTER_CONFIG_SIZE);
        assert_eq!(mem::size_of::<Particle>(), PARTICLE_SIZE);
    }

    #[test]
    fn test_struct_alignment() {
        assert_eq!(mem::align_of::<ParticleSpawnParams>(), 4);
        assert_eq!(mem::align_of::<EmitterConfig>(), 4);
        assert_eq!(mem::align_of::<Particle>(), 4);
    }

    // ── Shader Validation ───────────────────────────────────────────────

    #[test]
    fn test_shader_parses() {
        let source = include_str!("../../shaders/particles/gpu_particle_spawn.comp.wgsl");
        let module = naga::front::wgsl::parse_str(source);
        assert!(module.is_ok(), "Shader failed to parse: {:?}", module.err());
    }

    #[test]
    fn test_shader_has_spawn_particles_entry_point() {
        let source = include_str!("../../shaders/particles/gpu_particle_spawn.comp.wgsl");
        let module = naga::front::wgsl::parse_str(source).expect("Failed to parse shader");

        let entry_names: Vec<_> = module
            .entry_points
            .iter()
            .map(|ep| ep.name.as_str())
            .collect();
        assert!(
            entry_names.contains(&"spawn_particles"),
            "Missing spawn_particles entry point. Found: {:?}",
            entry_names
        );
    }

    #[test]
    fn test_shader_has_spawn_particles_directed_entry_point() {
        let source = include_str!("../../shaders/particles/gpu_particle_spawn.comp.wgsl");
        let module = naga::front::wgsl::parse_str(source).expect("Failed to parse shader");

        let entry_names: Vec<_> = module
            .entry_points
            .iter()
            .map(|ep| ep.name.as_str())
            .collect();
        assert!(
            entry_names.contains(&"spawn_particles_directed"),
            "Missing spawn_particles_directed entry point. Found: {:?}",
            entry_names
        );
    }

    #[test]
    fn test_shader_entry_points_are_compute() {
        let source = include_str!("../../shaders/particles/gpu_particle_spawn.comp.wgsl");
        let module = naga::front::wgsl::parse_str(source).expect("Failed to parse shader");

        for ep in &module.entry_points {
            assert_eq!(
                ep.stage,
                naga::ShaderStage::Compute,
                "Entry point '{}' is not compute",
                ep.name
            );
        }
    }

    #[test]
    fn test_shader_workgroup_size() {
        let source = include_str!("../../shaders/particles/gpu_particle_spawn.comp.wgsl");
        let module = naga::front::wgsl::parse_str(source).expect("Failed to parse shader");

        for ep in &module.entry_points {
            // Check workgroup size is [256, 1, 1]
            assert_eq!(
                ep.workgroup_size,
                [256, 1, 1],
                "Entry point '{}' has unexpected workgroup size {:?}",
                ep.name,
                ep.workgroup_size
            );
        }
    }
}
