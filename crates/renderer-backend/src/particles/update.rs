//! GPU Particle Update for TRINITY Engine (T-GPU-5.2).
//!
//! This module provides GPU-based particle physics simulation supporting:
//! - Multiple force types: gravity, wind, turbulence, vortex, attraction
//! - Ping-pong buffer pattern for stable updates
//! - Lifetime-based color and size interpolation
//! - Dead particle marking for subsequent compaction
//!
//! # Overview
//!
//! The particle update pipeline runs each frame:
//! 1. Read particles from input buffer (ping)
//! 2. Apply physics forces to velocity
//! 3. Integrate position using semi-implicit Euler
//! 4. Advance age and mark dead particles
//! 5. Interpolate visual properties (color, size)
//! 6. Write to output buffer (pong)
//! 7. Swap buffers for next frame
//!
//! # Performance
//!
//! - Work complexity: O(n) where n = particle count
//! - Target: < 0.1ms for 100K particles
//! - Workgroup size: 256 (optimal GPU occupancy)
//!
//! # Usage
//!
//! ```ignore
//! // Create pipeline and resources
//! let pipeline = ParticleUpdatePipeline::new(&device);
//! let resources = ParticleUpdateResources::new(&device, 100_000);
//!
//! // Configure forces
//! let forces = ForceConfig {
//!     gravity: [0.0, -9.8, 0.0],
//!     gravity_strength: 1.0,
//!     drag: 0.1,
//!     force_flags: FORCE_FLAG_GRAVITY | FORCE_FLAG_DRAG,
//!     ..Default::default()
//! };
//!
//! // Each frame: update particles
//! resources.update_forces(&queue, &forces);
//! pipeline.dispatch(&mut encoder, &resources, particle_count, delta_time);
//! resources.swap_buffers();
//! ```

use std::mem;
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Compute shader workgroup size (must match WGSL constant).
pub const WORKGROUP_SIZE: u32 = 256;

/// Maximum particles supported by a single update dispatch.
pub const MAX_PARTICLES: u32 = 1_000_000;

/// Size of a single particle in bytes (64 bytes).
pub const PARTICLE_SIZE: usize = 64;

/// Force flag: enable gravity.
pub const FORCE_FLAG_GRAVITY: u32 = 1 << 0;

/// Force flag: enable wind.
pub const FORCE_FLAG_WIND: u32 = 1 << 1;

/// Force flag: enable turbulence.
pub const FORCE_FLAG_TURBULENCE: u32 = 1 << 2;

/// Force flag: enable vortex.
pub const FORCE_FLAG_VORTEX: u32 = 1 << 3;

/// Force flag: enable attraction.
pub const FORCE_FLAG_ATTRACTION: u32 = 1 << 4;

/// Flag bit indicating particle is alive.
pub const FLAG_ALIVE: u32 = 1;

// ---------------------------------------------------------------------------
// ParticleUpdateParams
// ---------------------------------------------------------------------------

/// GPU uniform buffer for particle update parameters.
///
/// Matches the WGSL `ParticleUpdateParams` struct layout.
///
/// # Memory Layout
///
/// 16 bytes total, std140/std430 compatible:
///
/// | Offset | Field         | Size    |
/// |--------|---------------|---------|
/// | 0      | num_particles | 4 bytes |
/// | 4      | delta_time    | 4 bytes |
/// | 8      | time          | 4 bytes |
/// | 12     | _padding      | 4 bytes |
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ParticleUpdateParams {
    /// Number of particles to process.
    pub num_particles: u32,
    /// Delta time since last frame (seconds).
    pub delta_time: f32,
    /// Current simulation time (seconds).
    pub time: f32,
    /// Padding for 16-byte alignment.
    pub _padding: f32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<ParticleUpdateParams>() == 16);

impl ParticleUpdateParams {
    /// Create parameters for the given update.
    pub fn new(num_particles: u32, delta_time: f32, time: f32) -> Self {
        Self {
            num_particles,
            delta_time,
            time,
            _padding: 0.0,
        }
    }

    /// Get the number of workgroups needed for dispatch.
    #[inline]
    pub fn num_workgroups(&self) -> u32 {
        (self.num_particles + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
    }
}

impl Default for ParticleUpdateParams {
    fn default() -> Self {
        Self::new(0, 1.0 / 60.0, 0.0)
    }
}

// ---------------------------------------------------------------------------
// ForceConfig
// ---------------------------------------------------------------------------

/// Force configuration for particle simulation.
///
/// Matches the WGSL `ForceConfig` struct layout.
///
/// # Memory Layout
///
/// 112 bytes total:
///
/// | Offset | Field               | Size     |
/// |--------|---------------------|----------|
/// | 0      | gravity             | 12 bytes |
/// | 12     | gravity_strength    | 4 bytes  |
/// | 16     | wind                | 12 bytes |
/// | 28     | wind_strength       | 4 bytes  |
/// | 32     | turbulence_frequency| 4 bytes  |
/// | 36     | turbulence_amplitude| 4 bytes  |
/// | 40     | _pad0               | 8 bytes  |
/// | 48     | vortex_center       | 12 bytes |
/// | 60     | vortex_strength     | 4 bytes  |
/// | 64     | vortex_axis         | 12 bytes |
/// | 76     | vortex_radius       | 4 bytes  |
/// | 80     | attraction_point    | 12 bytes |
/// | 92     | attraction_strength | 4 bytes  |
/// | 96     | drag                | 4 bytes  |
/// | 100    | force_flags         | 4 bytes  |
/// | 104    | _pad1               | 8 bytes  |
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ForceConfig {
    // Gravity
    /// Gravity direction (normalized).
    pub gravity: [f32; 3],
    /// Gravity strength (m/s^2, typically 9.8).
    pub gravity_strength: f32,

    // Wind
    /// Wind direction (normalized).
    pub wind: [f32; 3],
    /// Wind strength (m/s^2).
    pub wind_strength: f32,

    // Turbulence
    /// Turbulence noise frequency.
    pub turbulence_frequency: f32,
    /// Turbulence force amplitude.
    pub turbulence_amplitude: f32,
    /// Padding.
    pub _pad0: [f32; 2],

    // Vortex
    /// Vortex center position.
    pub vortex_center: [f32; 3],
    /// Vortex rotation strength.
    pub vortex_strength: f32,
    /// Vortex rotation axis.
    pub vortex_axis: [f32; 3],
    /// Vortex falloff radius.
    pub vortex_radius: f32,

    // Attraction
    /// Attraction point position.
    pub attraction_point: [f32; 3],
    /// Attraction strength (negative = repel).
    pub attraction_strength: f32,

    // Drag
    /// Drag coefficient (0-1).
    pub drag: f32,
    /// Force enable flags.
    pub force_flags: u32,
    /// Padding.
    pub _pad1: [f32; 2],
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<ForceConfig>() == 112);

impl ForceConfig {
    /// Create a gravity-only configuration.
    pub fn gravity_only(gravity: [f32; 3], strength: f32, drag: f32) -> Self {
        Self {
            gravity,
            gravity_strength: strength,
            drag,
            force_flags: FORCE_FLAG_GRAVITY,
            ..Default::default()
        }
    }

    /// Create configuration with gravity and wind.
    pub fn gravity_wind(
        gravity: [f32; 3],
        gravity_strength: f32,
        wind: [f32; 3],
        wind_strength: f32,
        drag: f32,
    ) -> Self {
        Self {
            gravity,
            gravity_strength,
            wind,
            wind_strength,
            drag,
            force_flags: FORCE_FLAG_GRAVITY | FORCE_FLAG_WIND,
            ..Default::default()
        }
    }

    /// Enable gravity force.
    pub fn with_gravity(mut self, gravity: [f32; 3], strength: f32) -> Self {
        self.gravity = gravity;
        self.gravity_strength = strength;
        self.force_flags |= FORCE_FLAG_GRAVITY;
        self
    }

    /// Enable wind force.
    pub fn with_wind(mut self, wind: [f32; 3], strength: f32) -> Self {
        self.wind = wind;
        self.wind_strength = strength;
        self.force_flags |= FORCE_FLAG_WIND;
        self
    }

    /// Enable turbulence force.
    pub fn with_turbulence(mut self, frequency: f32, amplitude: f32) -> Self {
        self.turbulence_frequency = frequency;
        self.turbulence_amplitude = amplitude;
        self.force_flags |= FORCE_FLAG_TURBULENCE;
        self
    }

    /// Enable vortex force.
    pub fn with_vortex(
        mut self,
        center: [f32; 3],
        axis: [f32; 3],
        strength: f32,
        radius: f32,
    ) -> Self {
        self.vortex_center = center;
        self.vortex_axis = axis;
        self.vortex_strength = strength;
        self.vortex_radius = radius;
        self.force_flags |= FORCE_FLAG_VORTEX;
        self
    }

    /// Enable attraction force.
    pub fn with_attraction(mut self, point: [f32; 3], strength: f32) -> Self {
        self.attraction_point = point;
        self.attraction_strength = strength;
        self.force_flags |= FORCE_FLAG_ATTRACTION;
        self
    }

    /// Set drag coefficient.
    pub fn with_drag(mut self, drag: f32) -> Self {
        self.drag = drag;
        self
    }
}

impl Default for ForceConfig {
    fn default() -> Self {
        Self {
            gravity: [0.0, -1.0, 0.0],
            gravity_strength: 9.8,
            wind: [1.0, 0.0, 0.0],
            wind_strength: 0.0,
            turbulence_frequency: 1.0,
            turbulence_amplitude: 0.0,
            _pad0: [0.0; 2],
            vortex_center: [0.0; 3],
            vortex_strength: 0.0,
            vortex_axis: [0.0, 1.0, 0.0],
            vortex_radius: 1.0,
            attraction_point: [0.0; 3],
            attraction_strength: 0.0,
            drag: 0.0,
            force_flags: 0,
            _pad1: [0.0; 2],
        }
    }
}

// ---------------------------------------------------------------------------
// ColorParams
// ---------------------------------------------------------------------------

/// Color and size interpolation parameters.
///
/// Matches the WGSL `ColorParams` struct layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ColorParams {
    /// Starting color at age = 0.
    pub color_start: [f32; 4],
    /// Ending color at age = lifetime.
    pub color_end: [f32; 4],
    /// Starting size at age = 0.
    pub size_start: f32,
    /// Ending size at age = lifetime.
    pub size_end: f32,
    /// Padding.
    pub _pad: [f32; 2],
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<ColorParams>() == 48);

impl ColorParams {
    /// Create color parameters.
    pub fn new(
        color_start: [f32; 4],
        color_end: [f32; 4],
        size_start: f32,
        size_end: f32,
    ) -> Self {
        Self {
            color_start,
            color_end,
            size_start,
            size_end,
            _pad: [0.0; 2],
        }
    }

    /// Create fade-out animation (alpha goes to 0).
    pub fn fade_out(color: [f32; 4], size_start: f32, size_end: f32) -> Self {
        let color_end = [color[0], color[1], color[2], 0.0];
        Self::new(color, color_end, size_start, size_end)
    }

    /// Create shrink animation.
    pub fn shrink(color: [f32; 4], size_start: f32) -> Self {
        Self::new(color, color, size_start, 0.0)
    }
}

impl Default for ColorParams {
    fn default() -> Self {
        Self {
            color_start: [1.0, 1.0, 1.0, 1.0],
            color_end: [1.0, 1.0, 1.0, 0.0],
            size_start: 1.0,
            size_end: 0.5,
            _pad: [0.0; 2],
        }
    }
}

// ---------------------------------------------------------------------------
// Particle
// ---------------------------------------------------------------------------

/// GPU particle data (matches WGSL Particle struct, 64 bytes).
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
    /// Create a new alive particle.
    pub fn new(
        position: [f32; 3],
        velocity: [f32; 3],
        lifetime: f32,
        color: [f32; 4],
        size: f32,
    ) -> Self {
        Self {
            position,
            age: 0.0,
            velocity,
            lifetime,
            color,
            size,
            rotation: 0.0,
            rotation_speed: 0.0,
            flags: FLAG_ALIVE,
        }
    }

    /// Create a dead particle (placeholder).
    pub fn dead() -> Self {
        Self {
            position: [0.0; 3],
            age: 0.0,
            velocity: [0.0; 3],
            lifetime: 0.0,
            color: [0.0; 4],
            size: 0.0,
            rotation: 0.0,
            rotation_speed: 0.0,
            flags: 0,
        }
    }

    /// Check if particle is alive.
    #[inline]
    pub fn is_alive(&self) -> bool {
        (self.flags & FLAG_ALIVE) != 0
    }

    /// Mark particle as dead.
    pub fn kill(&mut self) {
        self.flags &= !FLAG_ALIVE;
    }
}

impl Default for Particle {
    fn default() -> Self {
        Self::dead()
    }
}

// ---------------------------------------------------------------------------
// ParticleUpdateResources
// ---------------------------------------------------------------------------

/// GPU resources for particle update.
///
/// Uses ping-pong buffers for stable updates:
/// - Frame N: Read from particles_a, write to particles_b
/// - Frame N+1: Read from particles_b, write to particles_a
pub struct ParticleUpdateResources {
    /// Uniform buffer for update parameters.
    pub params_buffer: wgpu::Buffer,
    /// Uniform buffer for force configuration.
    pub forces_buffer: wgpu::Buffer,
    /// Uniform buffer for color parameters.
    pub color_params_buffer: wgpu::Buffer,
    /// Particle buffer A (ping).
    pub particles_a: wgpu::Buffer,
    /// Particle buffer B (pong).
    pub particles_b: wgpu::Buffer,
    /// Alive flags buffer (for compaction).
    pub alive_flags: wgpu::Buffer,
    /// Current buffer index (0 = read A/write B, 1 = read B/write A).
    buffer_index: u32,
    /// Maximum capacity in particles.
    pub capacity: u32,
}

impl ParticleUpdateResources {
    /// Create particle update resources for the given capacity.
    pub fn new(device: &wgpu::Device, capacity: u32) -> Self {
        let particle_buffer_size = (capacity as u64) * (PARTICLE_SIZE as u64);
        let alive_flags_size = (capacity as u64) * 4; // u32 per particle

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle_update_params"),
            size: mem::size_of::<ParticleUpdateParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let forces_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle_update_forces"),
            size: mem::size_of::<ForceConfig>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let color_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle_color_params"),
            size: mem::size_of::<ColorParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let particles_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particles_a"),
            size: particle_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let particles_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particles_b"),
            size: particle_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let alive_flags = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle_alive_flags"),
            size: alive_flags_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Self {
            params_buffer,
            forces_buffer,
            color_params_buffer,
            particles_a,
            particles_b,
            alive_flags,
            buffer_index: 0,
            capacity,
        }
    }

    /// Get the current input buffer (for reading).
    pub fn input_buffer(&self) -> &wgpu::Buffer {
        if self.buffer_index == 0 {
            &self.particles_a
        } else {
            &self.particles_b
        }
    }

    /// Get the current output buffer (for writing).
    pub fn output_buffer(&self) -> &wgpu::Buffer {
        if self.buffer_index == 0 {
            &self.particles_b
        } else {
            &self.particles_a
        }
    }

    /// Swap ping-pong buffers. Call after each update dispatch.
    pub fn swap_buffers(&mut self) {
        self.buffer_index = 1 - self.buffer_index;
    }

    /// Update the parameters buffer.
    pub fn update_params(&self, queue: &wgpu::Queue, params: &ParticleUpdateParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Update the forces buffer.
    pub fn update_forces(&self, queue: &wgpu::Queue, forces: &ForceConfig) {
        queue.write_buffer(&self.forces_buffer, 0, bytemuck::bytes_of(forces));
    }

    /// Update the color parameters buffer.
    pub fn update_color_params(&self, queue: &wgpu::Queue, color_params: &ColorParams) {
        queue.write_buffer(&self.color_params_buffer, 0, bytemuck::bytes_of(color_params));
    }

    /// Upload particles to the current input buffer.
    pub fn upload_particles(&self, queue: &wgpu::Queue, particles: &[Particle]) {
        queue.write_buffer(self.input_buffer(), 0, bytemuck::cast_slice(particles));
    }
}

// ---------------------------------------------------------------------------
// ParticleUpdatePipeline
// ---------------------------------------------------------------------------

/// Update mode determining which shader variant to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpdateMode {
    /// Full update with color/size interpolation.
    Full,
    /// Physics only (no color/size interpolation).
    PhysicsOnly,
    /// Simple gravity-only update (fastest).
    Simple,
}

/// GPU compute pipeline for particle updates.
pub struct ParticleUpdatePipeline {
    /// Full update pipeline.
    full_pipeline: wgpu::ComputePipeline,
    /// Physics-only pipeline.
    physics_only_pipeline: wgpu::ComputePipeline,
    /// Simple gravity-only pipeline.
    simple_pipeline: wgpu::ComputePipeline,
    /// Bind group layout.
    bind_group_layout: wgpu::BindGroupLayout,
}

impl ParticleUpdatePipeline {
    /// Create a new particle update pipeline.
    pub fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("particle_update_bind_group_layout"),
            entries: &[
                // Binding 0: params uniform
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
                // Binding 1: forces uniform
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
                // Binding 2: particles_in (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 3: particles_out (read-write storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 4: alive_flags (read-write storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 5: color_params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("particle_update_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_particle_update_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/particles/gpu_particle_update.comp.wgsl").into(),
            ),
        });

        let full_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("particle_update_full"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "update_particles",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let physics_only_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("particle_update_physics_only"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "update_particles_physics_only",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let simple_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("particle_update_simple"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "update_particles_simple",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            full_pipeline,
            physics_only_pipeline,
            simple_pipeline,
            bind_group_layout,
        }
    }

    /// Create a bind group for the given resources.
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        resources: &ParticleUpdateResources,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("particle_update_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.forces_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.input_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources.output_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: resources.alive_flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: resources.color_params_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Dispatch particle update compute shader.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        num_particles: u32,
        mode: UpdateMode,
    ) {
        let num_workgroups = (num_particles + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        let pipeline = match mode {
            UpdateMode::Full => &self.full_pipeline,
            UpdateMode::PhysicsOnly => &self.physics_only_pipeline,
            UpdateMode::Simple => &self.simple_pipeline,
        };

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("particle_update_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(num_workgroups, 1, 1);
    }
}

// ---------------------------------------------------------------------------
// CPU Reference Implementation
// ---------------------------------------------------------------------------

/// 3D simplex noise (CPU reference implementation).
///
/// Returns value in [-1, 1].
pub fn simplex_noise_3d(x: f32, y: f32, z: f32) -> f32 {
    // Skewing and unskewing factors for 3D
    const F3: f32 = 1.0 / 3.0;
    const G3: f32 = 1.0 / 6.0;

    // Skew input space to determine simplex cell
    let s = (x + y + z) * F3;
    let i = (x + s).floor();
    let j = (y + s).floor();
    let k = (z + s).floor();

    let t = (i + j + k) * G3;
    let x0 = x - (i - t);
    let y0 = y - (j - t);
    let z0 = z - (k - t);

    // Determine which simplex we're in
    let (i1, j1, k1, i2, j2, k2);
    if x0 >= y0 {
        if y0 >= z0 {
            i1 = 1;
            j1 = 0;
            k1 = 0;
            i2 = 1;
            j2 = 1;
            k2 = 0;
        } else if x0 >= z0 {
            i1 = 1;
            j1 = 0;
            k1 = 0;
            i2 = 1;
            j2 = 0;
            k2 = 1;
        } else {
            i1 = 0;
            j1 = 0;
            k1 = 1;
            i2 = 1;
            j2 = 0;
            k2 = 1;
        }
    } else if y0 < z0 {
        i1 = 0;
        j1 = 0;
        k1 = 1;
        i2 = 0;
        j2 = 1;
        k2 = 1;
    } else if x0 < z0 {
        i1 = 0;
        j1 = 1;
        k1 = 0;
        i2 = 0;
        j2 = 1;
        k2 = 1;
    } else {
        i1 = 0;
        j1 = 1;
        k1 = 0;
        i2 = 1;
        j2 = 1;
        k2 = 0;
    }

    // Offsets for corners
    let x1 = x0 - i1 as f32 + G3;
    let y1 = y0 - j1 as f32 + G3;
    let z1 = z0 - k1 as f32 + G3;
    let x2 = x0 - i2 as f32 + 2.0 * G3;
    let y2 = y0 - j2 as f32 + 2.0 * G3;
    let z2 = z0 - k2 as f32 + 2.0 * G3;
    let x3 = x0 - 1.0 + 3.0 * G3;
    let y3 = y0 - 1.0 + 3.0 * G3;
    let z3 = z0 - 1.0 + 3.0 * G3;

    // Hash coordinates
    let ii = (i as i32 & 255) as usize;
    let jj = (j as i32 & 255) as usize;
    let kk = (k as i32 & 255) as usize;

    // Gradient table (simplified)
    fn grad(hash: i32, x: f32, y: f32, z: f32) -> f32 {
        let h = hash & 15;
        let u = if h < 8 { x } else { y };
        let v = if h < 4 {
            y
        } else if h == 12 || h == 14 {
            x
        } else {
            z
        };
        (if (h & 1) == 0 { u } else { -u }) + (if (h & 2) == 0 { v } else { -v })
    }

    fn perm(i: usize) -> i32 {
        // Simple permutation based on hash
        let table: [i32; 16] = [1, 5, 3, 7, 2, 6, 0, 4, 9, 13, 11, 15, 10, 14, 8, 12];
        table[i & 15]
    }

    let gi0 = perm((ii + perm((jj + perm(kk) as usize) & 255) as usize) & 255);
    let gi1 =
        perm((ii + i1 + perm((jj + j1 + perm((kk + k1) & 255) as usize) & 255) as usize) & 255);
    let gi2 =
        perm((ii + i2 + perm((jj + j2 + perm((kk + k2) & 255) as usize) & 255) as usize) & 255);
    let gi3 = perm((ii + 1 + perm((jj + 1 + perm((kk + 1) & 255) as usize) & 255) as usize) & 255);

    // Calculate contributions from corners
    fn contribution(t: f32, gi: i32, x: f32, y: f32, z: f32) -> f32 {
        if t < 0.0 {
            0.0
        } else {
            let t2 = t * t;
            t2 * t2 * grad(gi, x, y, z)
        }
    }

    let n0 = contribution(0.6 - x0 * x0 - y0 * y0 - z0 * z0, gi0, x0, y0, z0);
    let n1 = contribution(0.6 - x1 * x1 - y1 * y1 - z1 * z1, gi1, x1, y1, z1);
    let n2 = contribution(0.6 - x2 * x2 - y2 * y2 - z2 * z2, gi2, x2, y2, z2);
    let n3 = contribution(0.6 - x3 * x3 - y3 * y3 - z3 * z3, gi3, x3, y3, z3);

    // Scale to [-1, 1]
    32.0 * (n0 + n1 + n2 + n3)
}

/// Apply gravity force to velocity (CPU reference).
pub fn cpu_apply_gravity(velocity: [f32; 3], gravity: [f32; 3], strength: f32, dt: f32) -> [f32; 3] {
    [
        velocity[0] + gravity[0] * strength * dt,
        velocity[1] + gravity[1] * strength * dt,
        velocity[2] + gravity[2] * strength * dt,
    ]
}

/// Apply drag to velocity (CPU reference).
pub fn cpu_apply_drag(velocity: [f32; 3], drag: f32, dt: f32) -> [f32; 3] {
    let factor = (1.0 - drag * dt).max(0.0);
    [velocity[0] * factor, velocity[1] * factor, velocity[2] * factor]
}

/// Apply wind force to velocity (CPU reference).
pub fn cpu_apply_wind(
    velocity: [f32; 3],
    wind: [f32; 3],
    strength: f32,
    position: [f32; 3],
    time: f32,
    dt: f32,
) -> [f32; 3] {
    let noise = simplex_noise_3d(position[0] * 0.1 + time * 0.5, position[1] * 0.1, position[2] * 0.1);
    let wind_strength = strength * (1.0 + noise * 0.3);
    [
        velocity[0] + wind[0] * wind_strength * dt,
        velocity[1] + wind[1] * wind_strength * dt,
        velocity[2] + wind[2] * wind_strength * dt,
    ]
}

/// Interpolate color based on age/lifetime (CPU reference).
pub fn cpu_interpolate_color(
    age: f32,
    lifetime: f32,
    start: [f32; 4],
    end: [f32; 4],
) -> [f32; 4] {
    let t = (age / lifetime.max(0.001)).clamp(0.0, 1.0);
    [
        start[0] + (end[0] - start[0]) * t,
        start[1] + (end[1] - start[1]) * t,
        start[2] + (end[2] - start[2]) * t,
        start[3] + (end[3] - start[3]) * t,
    ]
}

/// Interpolate size based on age/lifetime (CPU reference).
pub fn cpu_interpolate_size(age: f32, lifetime: f32, start: f32, end: f32) -> f32 {
    let t = (age / lifetime.max(0.001)).clamp(0.0, 1.0);
    start + (end - start) * t
}

/// CPU reference implementation for particle update.
///
/// Updates a slice of particles in-place using the same algorithm as the GPU shader.
pub fn cpu_particle_update(
    particles: &mut [Particle],
    forces: &ForceConfig,
    color_params: &ColorParams,
    delta_time: f32,
    time: f32,
    alive_flags: &mut [u32],
) {
    for (i, particle) in particles.iter_mut().enumerate() {
        // Check if already dead
        if (particle.flags & FLAG_ALIVE) == 0 {
            if i < alive_flags.len() {
                alive_flags[i] = 0;
            }
            continue;
        }

        // Advance age
        particle.age += delta_time;

        // Check if particle should die
        if particle.age >= particle.lifetime {
            particle.flags &= !FLAG_ALIVE;
            if i < alive_flags.len() {
                alive_flags[i] = 0;
            }
            continue;
        }

        // Apply gravity
        if (forces.force_flags & FORCE_FLAG_GRAVITY) != 0 {
            particle.velocity = cpu_apply_gravity(
                particle.velocity,
                forces.gravity,
                forces.gravity_strength,
                delta_time,
            );
        }

        // Apply wind
        if (forces.force_flags & FORCE_FLAG_WIND) != 0 {
            particle.velocity = cpu_apply_wind(
                particle.velocity,
                forces.wind,
                forces.wind_strength,
                particle.position,
                time,
                delta_time,
            );
        }

        // Apply drag
        particle.velocity = cpu_apply_drag(particle.velocity, forces.drag, delta_time);

        // Integrate position
        particle.position[0] += particle.velocity[0] * delta_time;
        particle.position[1] += particle.velocity[1] * delta_time;
        particle.position[2] += particle.velocity[2] * delta_time;

        // Interpolate color
        particle.color = cpu_interpolate_color(
            particle.age,
            particle.lifetime,
            color_params.color_start,
            color_params.color_end,
        );

        // Interpolate size
        particle.size = cpu_interpolate_size(
            particle.age,
            particle.lifetime,
            color_params.size_start,
            color_params.size_end,
        );

        // Advance rotation
        particle.rotation += particle.rotation_speed * delta_time;
        particle.rotation = particle.rotation % (2.0 * PI);

        // Mark as alive
        if i < alive_flags.len() {
            alive_flags[i] = 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── ParticleUpdateParams ────────────────────────────────────────────────

    #[test]
    fn test_update_params_size() {
        assert_eq!(mem::size_of::<ParticleUpdateParams>(), 16);
    }

    #[test]
    fn test_update_params_new() {
        let params = ParticleUpdateParams::new(1000, 0.016, 1.5);
        assert_eq!(params.num_particles, 1000);
        assert!((params.delta_time - 0.016).abs() < f32::EPSILON);
        assert!((params.time - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_update_params_workgroups() {
        let params = ParticleUpdateParams::new(1000, 0.016, 0.0);
        assert_eq!(params.num_workgroups(), 4); // ceil(1000/256) = 4

        let params2 = ParticleUpdateParams::new(256, 0.016, 0.0);
        assert_eq!(params2.num_workgroups(), 1);

        let params3 = ParticleUpdateParams::new(257, 0.016, 0.0);
        assert_eq!(params3.num_workgroups(), 2);
    }

    // ── ForceConfig ─────────────────────────────────────────────────────────

    #[test]
    fn test_force_config_size() {
        assert_eq!(mem::size_of::<ForceConfig>(), 112);
    }

    #[test]
    fn test_force_config_gravity_only() {
        let config = ForceConfig::gravity_only([0.0, -1.0, 0.0], 9.8, 0.1);
        assert_eq!(config.force_flags, FORCE_FLAG_GRAVITY);
        assert!((config.gravity_strength - 9.8).abs() < f32::EPSILON);
        assert!((config.drag - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn test_force_config_builder() {
        let config = ForceConfig::default()
            .with_gravity([0.0, -1.0, 0.0], 9.8)
            .with_wind([1.0, 0.0, 0.0], 5.0)
            .with_drag(0.2);

        assert_eq!(config.force_flags, FORCE_FLAG_GRAVITY | FORCE_FLAG_WIND);
        assert!((config.gravity_strength - 9.8).abs() < f32::EPSILON);
        assert!((config.wind_strength - 5.0).abs() < f32::EPSILON);
        assert!((config.drag - 0.2).abs() < f32::EPSILON);
    }

    #[test]
    fn test_force_config_vortex() {
        let config = ForceConfig::default()
            .with_vortex([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 10.0, 5.0);

        assert_eq!(config.force_flags, FORCE_FLAG_VORTEX);
        assert_eq!(config.vortex_center, [0.0, 0.0, 0.0]);
        assert_eq!(config.vortex_axis, [0.0, 1.0, 0.0]);
        assert!((config.vortex_strength - 10.0).abs() < f32::EPSILON);
        assert!((config.vortex_radius - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_force_config_attraction() {
        let config = ForceConfig::default()
            .with_attraction([5.0, 5.0, 5.0], -2.0);

        assert_eq!(config.force_flags, FORCE_FLAG_ATTRACTION);
        assert_eq!(config.attraction_point, [5.0, 5.0, 5.0]);
        assert!((config.attraction_strength - (-2.0)).abs() < f32::EPSILON);
    }

    // ── ColorParams ─────────────────────────────────────────────────────────

    #[test]
    fn test_color_params_size() {
        assert_eq!(mem::size_of::<ColorParams>(), 48);
    }

    #[test]
    fn test_color_params_fade_out() {
        let params = ColorParams::fade_out([1.0, 0.5, 0.0, 1.0], 2.0, 0.5);
        assert_eq!(params.color_start, [1.0, 0.5, 0.0, 1.0]);
        assert_eq!(params.color_end[3], 0.0); // Alpha fades to 0
        assert!((params.size_start - 2.0).abs() < f32::EPSILON);
        assert!((params.size_end - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_color_params_shrink() {
        let params = ColorParams::shrink([1.0, 1.0, 1.0, 1.0], 3.0);
        assert!((params.size_start - 3.0).abs() < f32::EPSILON);
        assert!((params.size_end - 0.0).abs() < f32::EPSILON);
    }

    // ── Particle ────────────────────────────────────────────────────────────

    #[test]
    fn test_particle_size() {
        assert_eq!(mem::size_of::<Particle>(), PARTICLE_SIZE);
        assert_eq!(mem::size_of::<Particle>(), 64);
    }

    #[test]
    fn test_particle_new() {
        let p = Particle::new(
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 0.0],
            5.0,
            [1.0, 0.0, 0.0, 1.0],
            0.5,
        );

        assert_eq!(p.position, [1.0, 2.0, 3.0]);
        assert_eq!(p.velocity, [0.0, 1.0, 0.0]);
        assert!((p.age - 0.0).abs() < f32::EPSILON);
        assert!((p.lifetime - 5.0).abs() < f32::EPSILON);
        assert!(p.is_alive());
    }

    #[test]
    fn test_particle_dead() {
        let p = Particle::dead();
        assert!(!p.is_alive());
        assert_eq!(p.flags, 0);
    }

    #[test]
    fn test_particle_kill() {
        let mut p = Particle::new([0.0; 3], [0.0; 3], 1.0, [1.0; 4], 1.0);
        assert!(p.is_alive());
        p.kill();
        assert!(!p.is_alive());
    }

    // ── CPU Apply Gravity ───────────────────────────────────────────────────

    #[test]
    fn test_cpu_apply_gravity() {
        let velocity = [0.0, 0.0, 0.0];
        let gravity = [0.0, -1.0, 0.0];
        let result = cpu_apply_gravity(velocity, gravity, 9.8, 1.0);

        assert!((result[0] - 0.0).abs() < f32::EPSILON);
        assert!((result[1] - (-9.8)).abs() < f32::EPSILON);
        assert!((result[2] - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cpu_apply_gravity_partial() {
        let velocity = [1.0, 2.0, 3.0];
        let gravity = [0.0, -1.0, 0.0];
        let result = cpu_apply_gravity(velocity, gravity, 9.8, 0.5);

        assert!((result[0] - 1.0).abs() < f32::EPSILON);
        assert!((result[1] - (2.0 - 4.9)).abs() < 0.01);
        assert!((result[2] - 3.0).abs() < f32::EPSILON);
    }

    // ── CPU Apply Drag ──────────────────────────────────────────────────────

    #[test]
    fn test_cpu_apply_drag() {
        let velocity = [10.0, 10.0, 10.0];
        let result = cpu_apply_drag(velocity, 0.5, 1.0);

        // factor = 1 - 0.5 * 1.0 = 0.5
        assert!((result[0] - 5.0).abs() < f32::EPSILON);
        assert!((result[1] - 5.0).abs() < f32::EPSILON);
        assert!((result[2] - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cpu_apply_drag_high() {
        let velocity = [10.0, 10.0, 10.0];
        let result = cpu_apply_drag(velocity, 2.0, 1.0);

        // factor = max(1 - 2.0 * 1.0, 0) = 0
        assert!((result[0] - 0.0).abs() < f32::EPSILON);
        assert!((result[1] - 0.0).abs() < f32::EPSILON);
        assert!((result[2] - 0.0).abs() < f32::EPSILON);
    }

    // ── CPU Interpolate Color ───────────────────────────────────────────────

    #[test]
    fn test_cpu_interpolate_color_start() {
        let start = [1.0, 0.0, 0.0, 1.0];
        let end = [0.0, 1.0, 0.0, 0.0];
        let result = cpu_interpolate_color(0.0, 1.0, start, end);

        assert!((result[0] - 1.0).abs() < f32::EPSILON);
        assert!((result[1] - 0.0).abs() < f32::EPSILON);
        assert!((result[2] - 0.0).abs() < f32::EPSILON);
        assert!((result[3] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cpu_interpolate_color_end() {
        let start = [1.0, 0.0, 0.0, 1.0];
        let end = [0.0, 1.0, 0.0, 0.0];
        let result = cpu_interpolate_color(1.0, 1.0, start, end);

        assert!((result[0] - 0.0).abs() < f32::EPSILON);
        assert!((result[1] - 1.0).abs() < f32::EPSILON);
        assert!((result[2] - 0.0).abs() < f32::EPSILON);
        assert!((result[3] - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cpu_interpolate_color_mid() {
        let start = [1.0, 0.0, 0.0, 1.0];
        let end = [0.0, 1.0, 0.0, 0.0];
        let result = cpu_interpolate_color(0.5, 1.0, start, end);

        assert!((result[0] - 0.5).abs() < f32::EPSILON);
        assert!((result[1] - 0.5).abs() < f32::EPSILON);
        assert!((result[2] - 0.0).abs() < f32::EPSILON);
        assert!((result[3] - 0.5).abs() < f32::EPSILON);
    }

    // ── CPU Interpolate Size ────────────────────────────────────────────────

    #[test]
    fn test_cpu_interpolate_size_start() {
        let result = cpu_interpolate_size(0.0, 1.0, 2.0, 0.5);
        assert!((result - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cpu_interpolate_size_end() {
        let result = cpu_interpolate_size(1.0, 1.0, 2.0, 0.5);
        assert!((result - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cpu_interpolate_size_mid() {
        let result = cpu_interpolate_size(0.5, 1.0, 2.0, 0.0);
        assert!((result - 1.0).abs() < f32::EPSILON);
    }

    // ── CPU Particle Update ─────────────────────────────────────────────────

    #[test]
    fn test_cpu_particle_update_position_integration() {
        let mut particles = vec![Particle::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            10.0,
            [1.0; 4],
            1.0,
        )];
        let mut alive_flags = vec![1u32; 1];
        let forces = ForceConfig::default();
        let color_params = ColorParams::default();

        cpu_particle_update(&mut particles, &forces, &color_params, 1.0, 0.0, &mut alive_flags);

        // Position should move by velocity * dt
        assert!((particles[0].position[0] - 1.0).abs() < f32::EPSILON);
        assert!(particles[0].is_alive());
        assert_eq!(alive_flags[0], 1);
    }

    #[test]
    fn test_cpu_particle_update_gravity() {
        let mut particles = vec![Particle::new(
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 0.0],
            10.0,
            [1.0; 4],
            1.0,
        )];
        let mut alive_flags = vec![1u32; 1];
        let forces = ForceConfig::gravity_only([0.0, -1.0, 0.0], 10.0, 0.0);
        let color_params = ColorParams::default();

        cpu_particle_update(&mut particles, &forces, &color_params, 1.0, 0.0, &mut alive_flags);

        // Velocity should have gravity applied
        assert!((particles[0].velocity[1] - (-10.0)).abs() < f32::EPSILON);
        // Position should be updated
        assert!((particles[0].position[1] - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cpu_particle_update_age_advancement() {
        let mut particles = vec![Particle::new(
            [0.0; 3],
            [0.0; 3],
            5.0,
            [1.0; 4],
            1.0,
        )];
        let mut alive_flags = vec![1u32; 1];
        let forces = ForceConfig::default();
        let color_params = ColorParams::default();

        cpu_particle_update(&mut particles, &forces, &color_params, 2.0, 0.0, &mut alive_flags);

        assert!((particles[0].age - 2.0).abs() < f32::EPSILON);
        assert!(particles[0].is_alive());
    }

    #[test]
    fn test_cpu_particle_update_dead_marking() {
        let mut particles = vec![Particle::new(
            [0.0; 3],
            [0.0; 3],
            1.0, // Short lifetime
            [1.0; 4],
            1.0,
        )];
        let mut alive_flags = vec![1u32; 1];
        let forces = ForceConfig::default();
        let color_params = ColorParams::default();

        // First update: age goes to 1.0, equals lifetime
        cpu_particle_update(&mut particles, &forces, &color_params, 1.0, 0.0, &mut alive_flags);

        assert!(!particles[0].is_alive());
        assert_eq!(alive_flags[0], 0);
    }

    #[test]
    fn test_cpu_particle_update_ping_pong_pattern() {
        // Simulate ping-pong by using two arrays
        let mut particles_a = vec![Particle::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            10.0,
            [1.0; 4],
            1.0,
        )];
        let mut particles_b = vec![Particle::dead()];
        let mut alive_flags = vec![1u32; 1];
        let forces = ForceConfig::default();
        let color_params = ColorParams::default();

        // Frame 1: Update A -> B
        particles_b[0] = particles_a[0];
        cpu_particle_update(&mut particles_b, &forces, &color_params, 0.1, 0.0, &mut alive_flags);
        assert!((particles_b[0].position[0] - 0.1).abs() < f32::EPSILON);

        // Frame 2: Update B -> A
        particles_a[0] = particles_b[0];
        cpu_particle_update(&mut particles_a, &forces, &color_params, 0.1, 0.1, &mut alive_flags);
        assert!((particles_a[0].position[0] - 0.2).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cpu_particle_update_color_interpolation() {
        let mut particles = vec![Particle::new(
            [0.0; 3],
            [0.0; 3],
            2.0,
            [1.0, 0.0, 0.0, 1.0],
            1.0,
        )];
        let mut alive_flags = vec![1u32; 1];
        let forces = ForceConfig::default();
        let color_params = ColorParams::new(
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            2.0,
            0.0,
        );

        // Advance to halfway through lifetime
        cpu_particle_update(&mut particles, &forces, &color_params, 1.0, 0.0, &mut alive_flags);

        // Color should be interpolated 50%
        assert!((particles[0].color[0] - 0.5).abs() < f32::EPSILON);
        assert!((particles[0].color[1] - 0.5).abs() < f32::EPSILON);
        assert!((particles[0].color[3] - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cpu_particle_update_size_interpolation() {
        let mut particles = vec![Particle::new(
            [0.0; 3],
            [0.0; 3],
            2.0,
            [1.0; 4],
            2.0, // Start size
        )];
        let mut alive_flags = vec![1u32; 1];
        let forces = ForceConfig::default();
        let color_params = ColorParams::new([1.0; 4], [1.0; 4], 2.0, 0.0);

        // Advance to halfway
        cpu_particle_update(&mut particles, &forces, &color_params, 1.0, 0.0, &mut alive_flags);

        assert!((particles[0].size - 1.0).abs() < f32::EPSILON);
    }

    // ── Simplex Noise ───────────────────────────────────────────────────────

    #[test]
    fn test_simplex_noise_range() {
        // Simplex noise should be in [-1, 1]
        for i in 0..100 {
            let x = (i as f32) * 0.1;
            let y = (i as f32) * 0.13;
            let z = (i as f32) * 0.17;
            let n = simplex_noise_3d(x, y, z);
            assert!(n >= -1.0 && n <= 1.0, "noise out of range: {}", n);
        }
    }

    #[test]
    fn test_simplex_noise_continuity() {
        // Nearby points should have similar values
        let n1 = simplex_noise_3d(0.0, 0.0, 0.0);
        let n2 = simplex_noise_3d(0.001, 0.0, 0.0);
        assert!((n1 - n2).abs() < 0.1, "noise not continuous");
    }

    // ── Update Mode ─────────────────────────────────────────────────────────

    #[test]
    fn test_update_mode_variants() {
        assert_ne!(UpdateMode::Full, UpdateMode::PhysicsOnly);
        assert_ne!(UpdateMode::Full, UpdateMode::Simple);
        assert_ne!(UpdateMode::PhysicsOnly, UpdateMode::Simple);
    }

    // ── Edge Cases ──────────────────────────────────────────────────────────

    #[test]
    fn test_cpu_particle_update_zero_delta_time() {
        let mut particles = vec![Particle::new(
            [1.0, 2.0, 3.0],
            [1.0, 0.0, 0.0],
            10.0,
            [1.0; 4],
            1.0,
        )];
        let mut alive_flags = vec![1u32; 1];
        let forces = ForceConfig::gravity_only([0.0, -1.0, 0.0], 9.8, 0.0);
        let color_params = ColorParams::default();

        let pos_before = particles[0].position;
        cpu_particle_update(&mut particles, &forces, &color_params, 0.0, 0.0, &mut alive_flags);

        // Position should not change with zero delta time
        assert_eq!(particles[0].position, pos_before);
    }

    #[test]
    fn test_cpu_particle_update_already_dead() {
        let mut particles = vec![Particle::dead()];
        let mut alive_flags = vec![0u32; 1];
        let forces = ForceConfig::default();
        let color_params = ColorParams::default();

        cpu_particle_update(&mut particles, &forces, &color_params, 1.0, 0.0, &mut alive_flags);

        assert!(!particles[0].is_alive());
        assert_eq!(alive_flags[0], 0);
    }

    #[test]
    fn test_cpu_particle_update_empty_slice() {
        let mut particles: Vec<Particle> = vec![];
        let mut alive_flags: Vec<u32> = vec![];
        let forces = ForceConfig::default();
        let color_params = ColorParams::default();

        // Should not panic with empty input
        cpu_particle_update(&mut particles, &forces, &color_params, 1.0, 0.0, &mut alive_flags);
    }

    #[test]
    fn test_cpu_particle_update_drag_only() {
        let mut particles = vec![Particle::new(
            [0.0; 3],
            [10.0, 10.0, 10.0],
            10.0,
            [1.0; 4],
            1.0,
        )];
        let mut alive_flags = vec![1u32; 1];
        let forces = ForceConfig::default().with_drag(0.5);
        let color_params = ColorParams::default();

        cpu_particle_update(&mut particles, &forces, &color_params, 1.0, 0.0, &mut alive_flags);

        // Velocity should be damped by drag
        assert!((particles[0].velocity[0] - 5.0).abs() < f32::EPSILON);
        assert!((particles[0].velocity[1] - 5.0).abs() < f32::EPSILON);
        assert!((particles[0].velocity[2] - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cpu_particle_update_zero_lifetime() {
        // Zero lifetime should immediately kill the particle
        let mut particles = vec![Particle::new(
            [0.0; 3],
            [0.0; 3],
            0.0, // Zero lifetime
            [1.0; 4],
            1.0,
        )];
        let mut alive_flags = vec![1u32; 1];
        let forces = ForceConfig::default();
        let color_params = ColorParams::default();

        cpu_particle_update(&mut particles, &forces, &color_params, 0.001, 0.0, &mut alive_flags);

        assert!(!particles[0].is_alive());
        assert_eq!(alive_flags[0], 0);
    }

    // ── Shader Validation ───────────────────────────────────────────────────

    #[test]
    fn test_shader_parses() {
        let source = include_str!("../../shaders/particles/gpu_particle_update.comp.wgsl");
        let module = naga::front::wgsl::parse_str(source);
        assert!(module.is_ok(), "Failed to parse shader: {:?}", module.err());
    }

    #[test]
    fn test_shader_has_update_particles_entry_point() {
        let source = include_str!("../../shaders/particles/gpu_particle_update.comp.wgsl");
        let module = naga::front::wgsl::parse_str(source).expect("Failed to parse shader");

        let entry_points: Vec<_> = module
            .entry_points
            .iter()
            .map(|ep| ep.name.as_str())
            .collect();

        assert!(
            entry_points.contains(&"update_particles"),
            "Shader missing update_particles entry point. Found: {:?}",
            entry_points
        );
    }

    #[test]
    fn test_shader_has_update_particles_physics_only_entry_point() {
        let source = include_str!("../../shaders/particles/gpu_particle_update.comp.wgsl");
        let module = naga::front::wgsl::parse_str(source).expect("Failed to parse shader");

        let entry_points: Vec<_> = module
            .entry_points
            .iter()
            .map(|ep| ep.name.as_str())
            .collect();

        assert!(
            entry_points.contains(&"update_particles_physics_only"),
            "Shader missing update_particles_physics_only entry point. Found: {:?}",
            entry_points
        );
    }

    #[test]
    fn test_shader_has_update_particles_simple_entry_point() {
        let source = include_str!("../../shaders/particles/gpu_particle_update.comp.wgsl");
        let module = naga::front::wgsl::parse_str(source).expect("Failed to parse shader");

        let entry_points: Vec<_> = module
            .entry_points
            .iter()
            .map(|ep| ep.name.as_str())
            .collect();

        assert!(
            entry_points.contains(&"update_particles_simple"),
            "Shader missing update_particles_simple entry point. Found: {:?}",
            entry_points
        );
    }

    #[test]
    fn test_shader_entry_points_are_compute() {
        let source = include_str!("../../shaders/particles/gpu_particle_update.comp.wgsl");
        let module = naga::front::wgsl::parse_str(source).expect("Failed to parse shader");

        for entry_point in &module.entry_points {
            assert_eq!(
                entry_point.stage,
                naga::ShaderStage::Compute,
                "Entry point {} is not a compute shader",
                entry_point.name
            );
        }
    }
}
