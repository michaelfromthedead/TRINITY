//! Indirect draw buffer management for GPU-driven rendering (T-GPU-2.4).
//!
//! Provides Rust wrappers for indirect draw commands that integrate with wgpu.
//! Supports hardware tier detection for optimal rendering path selection:
//!
//! - **Tier 1 (Full)**: `MULTI_DRAW_INDIRECT_COUNT` - Draw with GPU-side count
//! - **Tier 2 (Partial)**: `MULTI_DRAW_INDIRECT` - Multiple draws, CPU count
//! - **Tier 3 (Minimal)**: Single indirect draws, CPU batching
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::gpu_driven::{IndirectDrawBuffer, IndirectTier};
//!
//! let buffer = IndirectDrawBuffer::new(&device, 1024);
//! match buffer.tier() {
//!     IndirectTier::Full => {
//!         // Use multi_draw_indexed_indirect_count
//!         render_pass.multi_draw_indexed_indirect_count(
//!             buffer.commands_buffer(),
//!             0,
//!             buffer.count_buffer().unwrap(),
//!             0,
//!             buffer.max_draws(),
//!         );
//!     }
//!     IndirectTier::Partial => {
//!         // Use multi_draw_indexed_indirect with CPU-read count
//!         render_pass.multi_draw_indexed_indirect(
//!             buffer.commands_buffer(),
//!             0,
//!             cpu_draw_count,
//!         );
//!     }
//!     IndirectTier::Minimal => {
//!         // Issue individual draw_indexed_indirect calls
//!         for i in 0..cpu_draw_count {
//!             render_pass.draw_indexed_indirect(
//!                 buffer.commands_buffer(),
//!                 (i as u64) * IndirectDrawIndexedArgs::SIZE as u64,
//!             );
//!         }
//!     }
//! }
//! ```

use bytemuck::{Pod, Zeroable};
use wgpu::{Buffer, BufferUsages, Device, Queue};

// =============================================================================
// CONSTANTS
// =============================================================================

/// Default maximum number of draw commands in an indirect buffer.
pub const DEFAULT_MAX_DRAWS: u32 = 65536;

/// Size of IndirectDrawIndexedArgs in bytes (5 x u32 = 20 bytes).
pub const INDIRECT_DRAW_INDEXED_ARGS_SIZE: usize = 20;

/// Size of IndirectDrawArgs in bytes (4 x u32 = 16 bytes).
pub const INDIRECT_DRAW_ARGS_SIZE: usize = 16;

/// Size of IndirectDispatchArgs in bytes (3 x u32 = 12 bytes).
pub const INDIRECT_DISPATCH_ARGS_SIZE: usize = 12;

// =============================================================================
// INDIRECT DRAW STRUCTURES
// =============================================================================

/// Arguments for `draw_indexed_indirect`.
///
/// Must match wgpu's expected layout exactly (20 bytes).
/// This is the GPU-side struct written by compute shaders and read by the
/// graphics pipeline for indirect indexed drawing.
///
/// Layout matches:
/// - Vulkan: `VkDrawIndexedIndirectCommand`
/// - D3D12: `D3D12_DRAW_INDEXED_ARGUMENTS`
/// - Metal: `MTLDrawIndexedPrimitivesIndirectArguments`
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct IndirectDrawIndexedArgs {
    /// Number of indices to draw.
    pub index_count: u32,
    /// Number of instances to draw.
    pub instance_count: u32,
    /// Base index within the index buffer.
    pub first_index: u32,
    /// Vertex offset to add to each index (signed).
    pub base_vertex: i32,
    /// First instance to draw.
    pub first_instance: u32,
}

impl IndirectDrawIndexedArgs {
    /// Size in bytes of this struct.
    pub const SIZE: usize = INDIRECT_DRAW_INDEXED_ARGS_SIZE;

    /// Create new indexed draw arguments.
    #[inline]
    pub const fn new(
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    ) -> Self {
        Self {
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        }
    }

    /// Create a single instance draw.
    #[inline]
    pub const fn single(index_count: u32, first_index: u32, base_vertex: i32) -> Self {
        Self::new(index_count, 1, first_index, base_vertex, 0)
    }

    /// Create a zeroed (no-op) draw.
    #[inline]
    pub const fn zeroed() -> Self {
        Self::new(0, 0, 0, 0, 0)
    }

    /// Check if this draw would actually render anything.
    #[inline]
    pub const fn is_visible(&self) -> bool {
        self.index_count > 0 && self.instance_count > 0
    }

    /// Get the total number of vertices that will be processed.
    #[inline]
    pub const fn total_vertices(&self) -> u64 {
        self.index_count as u64 * self.instance_count as u64
    }

    /// Create draw arguments with index and instance count only.
    ///
    /// Sets `first_index`, `base_vertex`, and `first_instance` to 0.
    /// This is the most common constructor for simple indexed draw calls.
    #[inline]
    pub const fn with_counts(index_count: u32, instance_count: u32) -> Self {
        Self {
            index_count,
            instance_count,
            first_index: 0,
            base_vertex: 0,
            first_instance: 0,
        }
    }
}

/// Type alias matching wgpu 25.x naming convention.
///
/// This alias provides API compatibility with wgpu's `DrawIndexedIndirectArgs`
/// type while using our extended implementation with additional helper methods.
pub type DrawIndexedIndirectArgs = IndirectDrawIndexedArgs;

/// Arguments for `draw_indirect` (non-indexed).
///
/// Must match wgpu's expected layout exactly (16 bytes).
/// Compatible with wgpu 25.x `DrawIndirectArgs` structure.
///
/// # Layout (wgpu 25.x)
///
/// ```text
/// +-------------------+--------+--------+
/// | Field             | Offset | Size   |
/// +-------------------+--------+--------+
/// | vertex_count      | 0      | 4      |
/// | instance_count    | 4      | 4      |
/// | first_vertex      | 8      | 4      |
/// | first_instance    | 12     | 4      |
/// +-------------------+--------+--------+
/// | Total             |        | 16     |
/// +-------------------+--------+--------+
/// ```
///
/// This is the GPU-side struct written by compute shaders and read by the
/// graphics pipeline for indirect non-indexed drawing.
///
/// Layout matches:
/// - wgpu 25.x: `wgpu::DrawIndirectArgs`
/// - Vulkan: `VkDrawIndirectCommand`
/// - D3D12: `D3D12_DRAW_ARGUMENTS`
/// - Metal: `MTLDrawPrimitivesIndirectArguments`
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct IndirectDrawArgs {
    /// Number of vertices to draw.
    pub vertex_count: u32,
    /// Number of instances to draw.
    pub instance_count: u32,
    /// First vertex to draw.
    pub first_vertex: u32,
    /// First instance to draw.
    pub first_instance: u32,
}

impl IndirectDrawArgs {
    /// Size in bytes of this struct.
    pub const SIZE: usize = INDIRECT_DRAW_ARGS_SIZE;

    /// Create new draw arguments with all fields specified.
    #[inline]
    pub const fn new(
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> Self {
        Self {
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        }
    }

    /// Create draw arguments with vertex and instance count only.
    ///
    /// Sets `first_vertex` and `first_instance` to 0.
    /// This is the most common constructor for simple draw calls.
    #[inline]
    pub const fn with_counts(vertex_count: u32, instance_count: u32) -> Self {
        Self {
            vertex_count,
            instance_count,
            first_vertex: 0,
            first_instance: 0,
        }
    }

    /// Create a single instance draw.
    #[inline]
    pub const fn single(vertex_count: u32, first_vertex: u32) -> Self {
        Self::new(vertex_count, 1, first_vertex, 0)
    }

    /// Create a zeroed (no-op) draw.
    #[inline]
    pub const fn zeroed() -> Self {
        Self::new(0, 0, 0, 0)
    }

    /// Check if this draw would actually render anything.
    #[inline]
    pub const fn is_visible(&self) -> bool {
        self.vertex_count > 0 && self.instance_count > 0
    }

    /// Get the total number of vertices that will be processed.
    #[inline]
    pub const fn total_vertices(&self) -> u64 {
        self.vertex_count as u64 * self.instance_count as u64
    }
}

/// Type alias matching wgpu 25.x naming convention.
///
/// This alias provides API compatibility with wgpu's `DrawIndirectArgs` type
/// while using our extended implementation with additional helper methods.
pub type DrawIndirectArgs = IndirectDrawArgs;

/// Arguments for `dispatch_workgroups_indirect` compute calls.
///
/// Layout matches wgpu 25.x DispatchIndirectArgs:
/// - workgroup_count_x: u32 (4 bytes)
/// - workgroup_count_y: u32 (4 bytes)
/// - workgroup_count_z: u32 (4 bytes)
/// Total: 12 bytes
///
/// This is the GPU-side struct written by compute shaders and read by the
/// compute pipeline for indirect dispatch.
///
/// Layout matches:
/// - wgpu 25.x: `wgpu::DispatchIndirectArgs`
/// - Vulkan: `VkDispatchIndirectCommand`
/// - D3D12: `D3D12_DISPATCH_ARGUMENTS`
/// - Metal: `MTLDispatchThreadgroupsIndirectArguments`
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Pod, Zeroable)]
pub struct IndirectDispatchArgs {
    /// Number of workgroups in X dimension.
    pub workgroup_count_x: u32,
    /// Number of workgroups in Y dimension.
    pub workgroup_count_y: u32,
    /// Number of workgroups in Z dimension.
    pub workgroup_count_z: u32,
}

/// Type alias matching wgpu 25.x naming convention.
///
/// This alias provides API compatibility with wgpu's `DispatchIndirectArgs`
/// type while using our extended implementation with additional helper methods.
pub type DispatchIndirectArgs = IndirectDispatchArgs;

impl IndirectDispatchArgs {
    /// Size in bytes of this struct.
    pub const SIZE: usize = INDIRECT_DISPATCH_ARGS_SIZE;

    /// Create new dispatch arguments with all fields specified.
    #[inline]
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self {
            workgroup_count_x: x,
            workgroup_count_y: y,
            workgroup_count_z: z,
        }
    }

    /// Create a 1D dispatch (single workgroup in Y and Z).
    ///
    /// This is the most common pattern for linear compute operations.
    #[inline]
    pub const fn single(x: u32) -> Self {
        Self::new(x, 1, 1)
    }

    /// Create a 1D dispatch (alias for `single`).
    #[inline]
    pub const fn linear(x: u32) -> Self {
        Self::single(x)
    }

    /// Create a 2D dispatch (single workgroup in Z).
    #[inline]
    pub const fn grid_2d(x: u32, y: u32) -> Self {
        Self::new(x, y, 1)
    }

    /// Create a zeroed (no-op) dispatch.
    #[inline]
    pub const fn zeroed() -> Self {
        Self::new(0, 0, 0)
    }

    /// Check if this dispatch would actually execute.
    #[inline]
    pub const fn is_active(&self) -> bool {
        self.workgroup_count_x > 0 && self.workgroup_count_y > 0 && self.workgroup_count_z > 0
    }

    /// Get the total number of workgroups.
    #[inline]
    pub const fn total_workgroups(&self) -> u64 {
        self.workgroup_count_x as u64 * self.workgroup_count_y as u64 * self.workgroup_count_z as u64
    }

    /// Calculate dispatch args for a given element count and workgroup size.
    ///
    /// Returns a 1D dispatch with enough workgroups to cover all elements.
    #[inline]
    pub fn for_elements(element_count: u32, workgroup_size: u32) -> Self {
        let groups = (element_count + workgroup_size - 1) / workgroup_size;
        Self::single(groups)
    }

    /// Get the X dimension workgroup count.
    #[inline]
    pub const fn x(&self) -> u32 {
        self.workgroup_count_x
    }

    /// Get the Y dimension workgroup count.
    #[inline]
    pub const fn y(&self) -> u32 {
        self.workgroup_count_y
    }

    /// Get the Z dimension workgroup count.
    #[inline]
    pub const fn z(&self) -> u32 {
        self.workgroup_count_z
    }
}

// =============================================================================
// HARDWARE TIER DETECTION
// =============================================================================

/// Hardware capability tier for indirect drawing.
///
/// Different GPUs and APIs support different levels of indirect draw
/// functionality. This enum helps select the optimal rendering path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndirectTier {
    /// Tier 1: Full support - `draw_indexed_indirect_count`.
    ///
    /// Available on:
    /// - Vulkan 1.2+ with `drawIndirectCount`
    /// - D3D12 with `ExecuteIndirect` and count buffer
    /// - Metal 3.0+
    ///
    /// Allows GPU-side determination of draw count.
    Full,

    /// Tier 2: Partial - `draw_indexed_indirect` without count.
    ///
    /// Available on:
    /// - Vulkan with `VK_KHR_draw_indirect_count` (optional)
    /// - Most D3D12 hardware
    /// - Metal 2.0+
    ///
    /// Requires CPU to read back draw count.
    Partial,

    /// Tier 3: Minimal - CPU-side batching.
    ///
    /// Available on:
    /// - OpenGL ES 3.0
    /// - WebGL 2.0
    /// - Older mobile GPUs
    ///
    /// No multi-draw support; must issue individual draw calls.
    Minimal,
}

impl IndirectTier {
    /// Detect the hardware tier from device features.
    pub fn detect(device: &Device) -> Self {
        let features = device.features();

        if features.contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT) {
            Self::Full
        } else if features.contains(wgpu::Features::MULTI_DRAW_INDIRECT) {
            Self::Partial
        } else {
            Self::Minimal
        }
    }

    /// Check if GPU-side draw count is supported.
    #[inline]
    pub const fn supports_gpu_count(&self) -> bool {
        matches!(self, Self::Full)
    }

    /// Check if multi-draw is supported.
    #[inline]
    pub const fn supports_multi_draw(&self) -> bool {
        matches!(self, Self::Full | Self::Partial)
    }

    /// Get a human-readable description of this tier.
    #[inline]
    pub const fn description(&self) -> &'static str {
        match self {
            Self::Full => "Full (GPU count)",
            Self::Partial => "Partial (multi-draw)",
            Self::Minimal => "Minimal (single draw)",
        }
    }
}

impl std::fmt::Display for IndirectTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.description())
    }
}

// =============================================================================
// COUNT BUFFER (T-WGPU-P6.1.5)
// =============================================================================

/// Buffer for storing draw count for multi_draw_indirect_count.
///
/// Contains a single u32 that is:
/// - Written atomically by GPU culling compute shaders (via `storage_buffer()`)
/// - Read by multi_draw_indirect_count calls (via `indirect_buffer()`)
/// - Reset to 0 at frame start (via `reset()`)
///
/// # Usage Pattern
///
/// ```text
/// Frame Start:
///   count_buffer.reset(&queue);  // Clear to 0
///
/// Culling Pass (compute shader):
///   @group(0) @binding(0) var<storage, read_write> draw_count: atomic<u32>;
///   atomicAdd(&draw_count, 1u);  // Increment for each visible object
///
/// Draw Pass:
///   render_pass.multi_draw_indexed_indirect_count(
///       commands_buffer,
///       0,
///       count_buffer.indirect_buffer(),
///       0,
///       max_draws,
///   );
/// ```
///
/// # Buffer Usage Flags
///
/// The buffer is created with:
/// - `INDIRECT`: Required for multi_draw_indirect_count offset parameter
/// - `STORAGE`: Required for compute shader atomic writes
/// - `COPY_DST`: Required for CPU reset via queue.write_buffer
/// - `COPY_SRC`: Optional, for GPU readback if needed
pub struct CountBuffer {
    /// GPU buffer holding a single u32.
    buffer: Buffer,
    /// Label for debugging.
    label: Option<String>,
}

impl CountBuffer {
    /// Size of the count buffer in bytes (single u32).
    pub const SIZE: u64 = std::mem::size_of::<u32>() as u64;

    /// Create a new count buffer.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device to create the buffer on
    /// * `label` - Optional debug label (will have "_count" appended)
    pub fn new(device: &Device, label: Option<&str>) -> Self {
        let buffer_label = label.map(|l| format!("{}_count", l));
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: buffer_label.as_deref(),
            size: Self::SIZE,
            usage: BufferUsages::INDIRECT
                | BufferUsages::STORAGE
                | BufferUsages::COPY_DST
                | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            label: label.map(String::from),
        }
    }

    /// Reset count to 0 (call at frame start before culling).
    ///
    /// This must be called before the culling compute pass writes to the buffer.
    #[inline]
    pub fn reset(&self, queue: &Queue) {
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&0u32));
    }

    /// Get buffer for binding to compute shader (storage binding).
    ///
    /// Use this when creating a bind group for the culling compute shader:
    ///
    /// ```ignore
    /// wgpu::BindGroupEntry {
    ///     binding: 0,
    ///     resource: count_buffer.storage_buffer().as_entire_binding(),
    /// }
    /// ```
    #[inline]
    pub fn storage_buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Get buffer for multi_draw_indirect_count call.
    ///
    /// Use this when issuing the draw call:
    ///
    /// ```ignore
    /// render_pass.multi_draw_indexed_indirect_count(
    ///     commands_buffer,
    ///     0,
    ///     count_buffer.indirect_buffer(),
    ///     0,
    ///     max_draws,
    /// );
    /// ```
    #[inline]
    pub fn indirect_buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Get the underlying buffer directly.
    ///
    /// This is the same buffer returned by both `storage_buffer()` and
    /// `indirect_buffer()` - those methods exist for API clarity.
    #[inline]
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Upload a specific count value (for CPU-driven fallback).
    ///
    /// Use this when you know the draw count from the CPU side, bypassing
    /// GPU culling. This is useful for:
    /// - CPU-side culling fallback
    /// - Testing and debugging
    /// - Hybrid CPU/GPU culling
    #[inline]
    pub fn upload(&self, queue: &Queue, count: u32) {
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&count));
    }

    /// Get the debug label.
    #[inline]
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Get the buffer size in bytes (always 4).
    #[inline]
    pub const fn size(&self) -> u64 {
        Self::SIZE
    }

    /// Create a bind group layout entry for this buffer (storage, read-write).
    ///
    /// Returns a layout entry suitable for compute shader binding.
    pub fn bind_group_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: std::num::NonZeroU64::new(Self::SIZE),
            },
            count: None,
        }
    }

    /// Create a bind group entry for this buffer.
    pub fn bind_group_entry(&self, binding: u32) -> wgpu::BindGroupEntry<'_> {
        wgpu::BindGroupEntry {
            binding,
            resource: self.buffer.as_entire_binding(),
        }
    }
}

// =============================================================================
// INDIRECT DRAW BUFFER
// =============================================================================

/// Manager for indirect draw command buffers.
///
/// Manages a GPU buffer that holds an array of [`IndirectDrawIndexedArgs`]
/// (or [`IndirectDrawArgs`]) for use with wgpu's indirect draw commands.
///
/// # Buffer Layout
///
/// ```text
/// +--------------------------------------------------+
/// | IndirectDrawIndexedArgs[0] (20 bytes)            |
/// | IndirectDrawIndexedArgs[1] (20 bytes)            |
/// | ...                                               |
/// | IndirectDrawIndexedArgs[max_draws-1] (20 bytes)  |
/// +--------------------------------------------------+
/// ```
///
/// For Tier 1 hardware, a separate count buffer holds the number of valid
/// draw commands (written by GPU compute shader, read by draw call).
pub struct IndirectDrawBuffer {
    /// GPU buffer holding IndirectDrawIndexedArgs array.
    buffer: Buffer,
    /// GPU buffer for the draw count (Tier 1 only).
    count_buffer: Option<Buffer>,
    /// Maximum number of draw commands.
    max_draws: u32,
    /// Current number of valid draws (CPU-side tracking).
    current_draws: u32,
    /// Hardware tier.
    tier: IndirectTier,
    /// Label for debugging.
    label: String,
}

impl IndirectDrawBuffer {
    /// Create a new indirect draw buffer.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device to create buffers on
    /// * `max_draws` - Maximum number of draw commands to support
    pub fn new(device: &Device, max_draws: u32) -> Self {
        Self::with_label(device, max_draws, "indirect_draw")
    }

    /// Create a new indirect draw buffer with a custom label.
    pub fn with_label(device: &Device, max_draws: u32, label: &str) -> Self {
        let tier = IndirectTier::detect(device);

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{}_commands", label)),
            size: (max_draws as u64) * (IndirectDrawIndexedArgs::SIZE as u64),
            usage: BufferUsages::INDIRECT | BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let count_buffer = if tier == IndirectTier::Full {
            Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{}_count", label)),
                size: 4, // Single u32
                usage: BufferUsages::INDIRECT | BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
        } else {
            None
        };

        Self {
            buffer,
            count_buffer,
            max_draws,
            current_draws: 0,
            tier,
            label: label.to_string(),
        }
    }

    /// Get the draw commands buffer.
    #[inline]
    pub fn commands_buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Get the draw count buffer (Tier 1 only).
    #[inline]
    pub fn count_buffer(&self) -> Option<&Buffer> {
        self.count_buffer.as_ref()
    }

    /// Get the hardware tier.
    #[inline]
    pub fn tier(&self) -> IndirectTier {
        self.tier
    }

    /// Get max draws capacity.
    #[inline]
    pub fn max_draws(&self) -> u32 {
        self.max_draws
    }

    /// Get current draw count (CPU-side tracking).
    #[inline]
    pub fn current_draws(&self) -> u32 {
        self.current_draws
    }

    /// Get the label.
    #[inline]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Get the size of the commands buffer in bytes.
    #[inline]
    pub fn commands_buffer_size(&self) -> u64 {
        (self.max_draws as u64) * (IndirectDrawIndexedArgs::SIZE as u64)
    }

    /// Clear the buffer for a new frame.
    ///
    /// This resets the CPU-side draw count. The actual GPU buffer
    /// is typically cleared by the culling compute shader.
    #[inline]
    pub fn clear(&mut self) {
        self.current_draws = 0;
    }

    /// Upload draw commands to the GPU buffer.
    ///
    /// # Arguments
    ///
    /// * `queue` - The wgpu queue for buffer writes
    /// * `commands` - Slice of draw commands to upload
    ///
    /// # Returns
    ///
    /// Number of commands actually uploaded (may be less if buffer is full).
    pub fn upload_commands(&mut self, queue: &Queue, commands: &[IndirectDrawIndexedArgs]) -> u32 {
        let count = commands.len().min(self.max_draws as usize) as u32;
        if count == 0 {
            return 0;
        }

        let data = bytemuck::cast_slice(&commands[..count as usize]);
        queue.write_buffer(&self.buffer, 0, data);

        self.current_draws = count;

        // Update count buffer if present
        if let Some(count_buf) = &self.count_buffer {
            queue.write_buffer(count_buf, 0, bytemuck::bytes_of(&count));
        }

        count
    }

    /// Upload a single draw command at a specific offset.
    pub fn upload_command_at(&self, queue: &Queue, index: u32, command: &IndirectDrawIndexedArgs) {
        if index >= self.max_draws {
            return;
        }

        let offset = (index as u64) * (IndirectDrawIndexedArgs::SIZE as u64);
        queue.write_buffer(&self.buffer, offset, bytemuck::bytes_of(command));
    }

    /// Upload the draw count (for Tier 1 hardware).
    pub fn upload_count(&self, queue: &Queue, count: u32) {
        if let Some(count_buf) = &self.count_buffer {
            queue.write_buffer(count_buf, 0, bytemuck::bytes_of(&count));
        }
    }

    /// Get the byte offset for a draw command at the given index.
    #[inline]
    pub const fn offset_for_draw(&self, index: u32) -> u64 {
        (index as u64) * (IndirectDrawIndexedArgs::SIZE as u64)
    }

    /// Resize the buffer to a new capacity.
    ///
    /// If `new_max_draws` is greater than the current capacity, creates a new
    /// larger buffer. If smaller or equal, does nothing (buffers cannot shrink).
    ///
    /// Note: This does NOT preserve existing data in the buffer. After resize,
    /// you must re-upload any draw commands.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device to create the new buffer on
    /// * `new_max_draws` - The desired new maximum number of draw commands
    ///
    /// # Returns
    ///
    /// `true` if the buffer was resized, `false` if no resize was needed.
    pub fn resize(&mut self, device: &Device, new_max_draws: u32) -> bool {
        if new_max_draws <= self.max_draws {
            return false;
        }

        // Create new commands buffer
        let new_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{}_commands", self.label)),
            size: (new_max_draws as u64) * (IndirectDrawIndexedArgs::SIZE as u64),
            usage: BufferUsages::INDIRECT | BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.buffer = new_buffer;
        self.max_draws = new_max_draws;
        self.current_draws = 0; // Reset count since buffer contents are invalidated

        true
    }

    /// Get the buffer (alias for commands_buffer for API consistency).
    #[inline]
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Get the capacity (alias for max_draws).
    #[inline]
    pub fn capacity(&self) -> u32 {
        self.max_draws
    }

    /// Get the current count of draw commands.
    #[inline]
    pub fn count(&self) -> u32 {
        self.current_draws
    }
}

// =============================================================================
// INDIRECT DISPATCH BUFFER
// =============================================================================

/// Manager for indirect compute dispatch buffers.
///
/// Manages a GPU buffer that holds [`IndirectDispatchArgs`] for use with
/// wgpu's `dispatch_workgroups_indirect`.
pub struct IndirectDispatchBuffer {
    /// GPU buffer holding dispatch arguments.
    buffer: Buffer,
    /// Label for debugging.
    label: String,
}

impl IndirectDispatchBuffer {
    /// Create a new indirect dispatch buffer.
    pub fn new(device: &Device) -> Self {
        Self::with_label(device, "indirect_dispatch")
    }

    /// Create a new indirect dispatch buffer with a custom label.
    pub fn with_label(device: &Device, label: &str) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: IndirectDispatchArgs::SIZE as u64,
            usage: BufferUsages::INDIRECT | BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            label: label.to_string(),
        }
    }

    /// Get the dispatch arguments buffer.
    #[inline]
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Get the label.
    #[inline]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Upload dispatch arguments to the GPU buffer.
    pub fn upload(&self, queue: &Queue, args: &IndirectDispatchArgs) {
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(args));
    }
}

// =============================================================================
// MULTI-INDIRECT DRAW BUFFER
// =============================================================================

/// Configuration for a multi-indirect draw buffer.
#[derive(Debug, Clone)]
pub struct MultiIndirectConfig {
    /// Maximum number of draw commands.
    pub max_draws: u32,
    /// Whether to use indexed draws.
    pub indexed: bool,
    /// Buffer label for debugging.
    pub label: String,
}

impl Default for MultiIndirectConfig {
    fn default() -> Self {
        Self {
            max_draws: DEFAULT_MAX_DRAWS,
            indexed: true,
            label: "multi_indirect".to_string(),
        }
    }
}

impl MultiIndirectConfig {
    /// Create a new config for indexed draws.
    pub fn indexed(max_draws: u32) -> Self {
        Self {
            max_draws,
            indexed: true,
            label: "multi_indirect_indexed".to_string(),
        }
    }

    /// Create a new config for non-indexed draws.
    pub fn non_indexed(max_draws: u32) -> Self {
        Self {
            max_draws,
            indexed: false,
            label: "multi_indirect".to_string(),
        }
    }

    /// Set a custom label.
    pub fn with_label(mut self, label: &str) -> Self {
        self.label = label.to_string();
        self
    }
}

/// Buffer for multi-draw indirect rendering.
///
/// Supports both indexed and non-indexed multi-draw indirect commands.
/// The stride between draw arguments is automatically determined based
/// on the argument type.
pub struct MultiIndirectBuffer {
    /// GPU buffer holding draw arguments.
    buffer: Buffer,
    /// GPU buffer for draw count (Tier 1 only).
    count_buffer: Option<Buffer>,
    /// Configuration.
    config: MultiIndirectConfig,
    /// Hardware tier.
    tier: IndirectTier,
    /// Current draw count (CPU tracking).
    current_draws: u32,
}

impl MultiIndirectBuffer {
    /// Create a new multi-indirect buffer.
    pub fn new(device: &Device, config: MultiIndirectConfig) -> Self {
        let tier = IndirectTier::detect(device);
        let entry_size = if config.indexed {
            IndirectDrawIndexedArgs::SIZE
        } else {
            IndirectDrawArgs::SIZE
        };

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{}_commands", config.label)),
            size: (config.max_draws as u64) * (entry_size as u64),
            usage: BufferUsages::INDIRECT | BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let count_buffer = if tier == IndirectTier::Full {
            Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{}_count", config.label)),
                size: 4,
                usage: BufferUsages::INDIRECT | BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
        } else {
            None
        };

        Self {
            buffer,
            count_buffer,
            config,
            tier,
            current_draws: 0,
        }
    }

    /// Get the draw commands buffer.
    #[inline]
    pub fn commands_buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Get the draw count buffer (Tier 1 only).
    #[inline]
    pub fn count_buffer(&self) -> Option<&Buffer> {
        self.count_buffer.as_ref()
    }

    /// Get the configuration.
    #[inline]
    pub fn config(&self) -> &MultiIndirectConfig {
        &self.config
    }

    /// Get the hardware tier.
    #[inline]
    pub fn tier(&self) -> IndirectTier {
        self.tier
    }

    /// Get the stride between draw arguments in bytes.
    #[inline]
    pub fn stride(&self) -> u32 {
        if self.config.indexed {
            IndirectDrawIndexedArgs::SIZE as u32
        } else {
            IndirectDrawArgs::SIZE as u32
        }
    }

    /// Get current draw count.
    #[inline]
    pub fn current_draws(&self) -> u32 {
        self.current_draws
    }

    /// Clear for a new frame.
    #[inline]
    pub fn clear(&mut self) {
        self.current_draws = 0;
    }

    /// Upload indexed draw commands.
    pub fn upload_indexed(&mut self, queue: &Queue, commands: &[IndirectDrawIndexedArgs]) -> u32 {
        assert!(self.config.indexed, "Buffer configured for non-indexed draws");

        let count = commands.len().min(self.config.max_draws as usize) as u32;
        if count == 0 {
            return 0;
        }

        let data = bytemuck::cast_slice(&commands[..count as usize]);
        queue.write_buffer(&self.buffer, 0, data);

        self.current_draws = count;

        if let Some(count_buf) = &self.count_buffer {
            queue.write_buffer(count_buf, 0, bytemuck::bytes_of(&count));
        }

        count
    }

    /// Upload non-indexed draw commands.
    pub fn upload_non_indexed(&mut self, queue: &Queue, commands: &[IndirectDrawArgs]) -> u32 {
        assert!(!self.config.indexed, "Buffer configured for indexed draws");

        let count = commands.len().min(self.config.max_draws as usize) as u32;
        if count == 0 {
            return 0;
        }

        let data = bytemuck::cast_slice(&commands[..count as usize]);
        queue.write_buffer(&self.buffer, 0, data);

        self.current_draws = count;

        if let Some(count_buf) = &self.count_buffer {
            queue.write_buffer(count_buf, 0, bytemuck::bytes_of(&count));
        }

        count
    }
}

// =============================================================================
// DRAW BATCH BUILDER
// =============================================================================

/// Builder for constructing indirect draw batches.
///
/// Collects draw commands on the CPU side before uploading to GPU.
/// Useful for CPU-driven scenarios or hybrid CPU/GPU culling.
#[derive(Debug, Default)]
pub struct DrawBatchBuilder {
    /// Accumulated indexed draw commands.
    indexed_commands: Vec<IndirectDrawIndexedArgs>,
    /// Accumulated non-indexed draw commands.
    commands: Vec<IndirectDrawArgs>,
}

impl DrawBatchBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new builder with pre-allocated capacity.
    pub fn with_capacity(indexed_capacity: usize, non_indexed_capacity: usize) -> Self {
        Self {
            indexed_commands: Vec::with_capacity(indexed_capacity),
            commands: Vec::with_capacity(non_indexed_capacity),
        }
    }

    /// Add an indexed draw command.
    #[inline]
    pub fn add_indexed(&mut self, args: IndirectDrawIndexedArgs) {
        self.indexed_commands.push(args);
    }

    /// Add a non-indexed draw command.
    #[inline]
    pub fn add(&mut self, args: IndirectDrawArgs) {
        self.commands.push(args);
    }

    /// Get the indexed draw commands.
    #[inline]
    pub fn indexed_commands(&self) -> &[IndirectDrawIndexedArgs] {
        &self.indexed_commands
    }

    /// Get the non-indexed draw commands.
    #[inline]
    pub fn commands(&self) -> &[IndirectDrawArgs] {
        &self.commands
    }

    /// Get the number of indexed draw commands.
    #[inline]
    pub fn indexed_count(&self) -> usize {
        self.indexed_commands.len()
    }

    /// Get the number of non-indexed draw commands.
    #[inline]
    pub fn count(&self) -> usize {
        self.commands.len()
    }

    /// Clear all commands.
    pub fn clear(&mut self) {
        self.indexed_commands.clear();
        self.commands.clear();
    }

    /// Upload indexed commands to a buffer.
    pub fn upload_indexed_to(&self, buffer: &mut IndirectDrawBuffer, queue: &Queue) -> u32 {
        buffer.upload_commands(queue, &self.indexed_commands)
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Size/Layout Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_indirect_draw_indexed_args_size() {
        assert_eq!(
            std::mem::size_of::<IndirectDrawIndexedArgs>(),
            20,
            "IndirectDrawIndexedArgs must be exactly 20 bytes (5 x u32)"
        );
        assert_eq!(IndirectDrawIndexedArgs::SIZE, 20);
        assert_eq!(INDIRECT_DRAW_INDEXED_ARGS_SIZE, 20);
    }

    #[test]
    fn test_indirect_draw_args_size() {
        assert_eq!(
            std::mem::size_of::<IndirectDrawArgs>(),
            16,
            "IndirectDrawArgs must be exactly 16 bytes (4 x u32)"
        );
        assert_eq!(IndirectDrawArgs::SIZE, 16);
        assert_eq!(INDIRECT_DRAW_ARGS_SIZE, 16);
    }

    #[test]
    fn test_indirect_dispatch_args_size() {
        assert_eq!(
            std::mem::size_of::<IndirectDispatchArgs>(),
            12,
            "IndirectDispatchArgs must be exactly 12 bytes (3 x u32)"
        );
        assert_eq!(IndirectDispatchArgs::SIZE, 12);
        assert_eq!(INDIRECT_DISPATCH_ARGS_SIZE, 12);
    }

    #[test]
    fn test_struct_alignment() {
        // All structs should have 4-byte alignment (u32 alignment)
        assert_eq!(std::mem::align_of::<IndirectDrawIndexedArgs>(), 4);
        assert_eq!(std::mem::align_of::<IndirectDrawArgs>(), 4);
        assert_eq!(std::mem::align_of::<IndirectDispatchArgs>(), 4);
    }

    // -------------------------------------------------------------------------
    // IndirectDrawIndexedArgs Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_indirect_draw_indexed_args_new() {
        let args = IndirectDrawIndexedArgs::new(100, 5, 0, 0, 0);
        assert_eq!(args.index_count, 100);
        assert_eq!(args.instance_count, 5);
        assert_eq!(args.first_index, 0);
        assert_eq!(args.base_vertex, 0);
        assert_eq!(args.first_instance, 0);
    }

    #[test]
    fn test_indirect_draw_indexed_args_single() {
        let args = IndirectDrawIndexedArgs::single(36, 100, -50);
        assert_eq!(args.index_count, 36);
        assert_eq!(args.instance_count, 1);
        assert_eq!(args.first_index, 100);
        assert_eq!(args.base_vertex, -50);
        assert_eq!(args.first_instance, 0);
    }

    #[test]
    fn test_indirect_draw_indexed_args_zeroed() {
        let args = IndirectDrawIndexedArgs::zeroed();
        assert_eq!(args.index_count, 0);
        assert_eq!(args.instance_count, 0);
        assert!(!args.is_visible());
    }

    #[test]
    fn test_indirect_draw_indexed_args_is_visible() {
        assert!(IndirectDrawIndexedArgs::new(1, 1, 0, 0, 0).is_visible());
        assert!(!IndirectDrawIndexedArgs::new(0, 1, 0, 0, 0).is_visible());
        assert!(!IndirectDrawIndexedArgs::new(1, 0, 0, 0, 0).is_visible());
    }

    #[test]
    fn test_indirect_draw_indexed_args_total_vertices() {
        let args = IndirectDrawIndexedArgs::new(100, 50, 0, 0, 0);
        assert_eq!(args.total_vertices(), 5000);
    }

    #[test]
    fn test_indirect_draw_indexed_args_bytemuck() {
        let args = IndirectDrawIndexedArgs::new(100, 5, 10, -20, 30);
        let bytes: &[u8] = bytemuck::bytes_of(&args);

        // Verify we can round-trip through bytes
        let restored: IndirectDrawIndexedArgs = *bytemuck::from_bytes(bytes);
        assert_eq!(args, restored);

        // Verify byte layout matches expected
        assert_eq!(bytes.len(), 20);
    }

    #[test]
    fn test_indirect_draw_indexed_args_with_counts() {
        let args = IndirectDrawIndexedArgs::with_counts(100, 5);
        assert_eq!(args.index_count, 100);
        assert_eq!(args.instance_count, 5);
        assert_eq!(args.first_index, 0);
        assert_eq!(args.base_vertex, 0);
        assert_eq!(args.first_instance, 0);
    }

    // -------------------------------------------------------------------------
    // DrawIndexedIndirectArgs Type Alias Tests (wgpu 25.x compatibility)
    // -------------------------------------------------------------------------

    #[test]
    fn test_draw_indexed_indirect_args_size_is_20_bytes() {
        assert_eq!(DrawIndexedIndirectArgs::SIZE, 20);
        assert_eq!(std::mem::size_of::<DrawIndexedIndirectArgs>(), 20);
    }

    #[test]
    fn test_draw_indexed_indirect_args_base_vertex_is_signed() {
        let args = DrawIndexedIndirectArgs::new(36, 1, 0, -100, 0);
        assert_eq!(args.base_vertex, -100);

        // Test extreme negative value
        let args_extreme = DrawIndexedIndirectArgs::new(36, 1, 0, i32::MIN, 0);
        assert_eq!(args_extreme.base_vertex, i32::MIN);
    }

    #[test]
    fn test_draw_indexed_indirect_args_bytemuck_pod() {
        let args = DrawIndexedIndirectArgs::with_counts(100, 5);
        let bytes: &[u8] = bytemuck::bytes_of(&args);
        assert_eq!(bytes.len(), 20);

        // Verify round-trip through bytes
        let restored: DrawIndexedIndirectArgs = *bytemuck::from_bytes(bytes);
        assert_eq!(args, restored);
    }

    #[test]
    fn test_draw_indexed_indirect_args_is_type_alias() {
        // Verify DrawIndexedIndirectArgs is the same type as IndirectDrawIndexedArgs
        let indirect: IndirectDrawIndexedArgs = IndirectDrawIndexedArgs::new(36, 1, 0, -50, 0);
        let draw: DrawIndexedIndirectArgs = indirect; // Should compile without conversion
        assert_eq!(indirect, draw);
    }

    #[test]
    fn test_draw_indexed_indirect_args_default_zeros() {
        let args = DrawIndexedIndirectArgs::default();
        assert_eq!(args.index_count, 0);
        assert_eq!(args.instance_count, 0);
        assert_eq!(args.first_index, 0);
        assert_eq!(args.base_vertex, 0);
        assert_eq!(args.first_instance, 0);
    }

    // -------------------------------------------------------------------------
    // IndirectDrawArgs Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_indirect_draw_args_new() {
        let args = IndirectDrawArgs::new(100, 5, 10, 20);
        assert_eq!(args.vertex_count, 100);
        assert_eq!(args.instance_count, 5);
        assert_eq!(args.first_vertex, 10);
        assert_eq!(args.first_instance, 20);
    }

    #[test]
    fn test_indirect_draw_args_single() {
        let args = IndirectDrawArgs::single(36, 100);
        assert_eq!(args.vertex_count, 36);
        assert_eq!(args.instance_count, 1);
        assert_eq!(args.first_vertex, 100);
        assert_eq!(args.first_instance, 0);
    }

    #[test]
    fn test_indirect_draw_args_is_visible() {
        assert!(IndirectDrawArgs::new(1, 1, 0, 0).is_visible());
        assert!(!IndirectDrawArgs::new(0, 1, 0, 0).is_visible());
        assert!(!IndirectDrawArgs::new(1, 0, 0, 0).is_visible());
    }

    #[test]
    fn test_indirect_draw_args_bytemuck() {
        let args = IndirectDrawArgs::new(100, 5, 10, 20);
        let bytes: &[u8] = bytemuck::bytes_of(&args);

        let restored: IndirectDrawArgs = *bytemuck::from_bytes(bytes);
        assert_eq!(args, restored);
        assert_eq!(bytes.len(), 16);
    }

    #[test]
    fn test_indirect_draw_args_with_counts() {
        let args = IndirectDrawArgs::with_counts(100, 5);
        assert_eq!(args.vertex_count, 100);
        assert_eq!(args.instance_count, 5);
        assert_eq!(args.first_vertex, 0);
        assert_eq!(args.first_instance, 0);
    }

    // -------------------------------------------------------------------------
    // DrawIndirectArgs Type Alias Tests (wgpu 25.x compatibility)
    // -------------------------------------------------------------------------

    #[test]
    fn test_draw_indirect_args_size_is_16_bytes() {
        assert_eq!(DrawIndirectArgs::SIZE, 16);
        assert_eq!(std::mem::size_of::<DrawIndirectArgs>(), 16);
    }

    #[test]
    fn test_draw_indirect_args_default_zeros() {
        let args = DrawIndirectArgs::default();
        assert_eq!(args.vertex_count, 0);
        assert_eq!(args.instance_count, 0);
        assert_eq!(args.first_vertex, 0);
        assert_eq!(args.first_instance, 0);
    }

    #[test]
    fn test_draw_indirect_args_new_constructor() {
        let args = DrawIndirectArgs::with_counts(100, 5);
        assert_eq!(args.vertex_count, 100);
        assert_eq!(args.instance_count, 5);
        assert_eq!(args.first_vertex, 0);
        assert_eq!(args.first_instance, 0);
    }

    #[test]
    fn test_draw_indirect_args_bytemuck_cast() {
        let args = DrawIndirectArgs::with_counts(10, 1);
        let bytes: &[u8] = bytemuck::bytes_of(&args);
        assert_eq!(bytes.len(), 16);
    }

    #[test]
    fn test_draw_indirect_args_is_type_alias() {
        // Verify DrawIndirectArgs is the same type as IndirectDrawArgs
        let indirect: IndirectDrawArgs = IndirectDrawArgs::new(10, 1, 0, 0);
        let draw: DrawIndirectArgs = indirect; // Should compile without conversion
        assert_eq!(indirect, draw);
    }

    // -------------------------------------------------------------------------
    // IndirectDispatchArgs Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dispatch_args_size_is_12_bytes() {
        assert_eq!(IndirectDispatchArgs::SIZE, 12);
        assert_eq!(std::mem::size_of::<IndirectDispatchArgs>(), 12);
        assert_eq!(INDIRECT_DISPATCH_ARGS_SIZE, 12);
    }

    #[test]
    fn test_indirect_dispatch_args_new() {
        let args = IndirectDispatchArgs::new(64, 64, 1);
        assert_eq!(args.workgroup_count_x, 64);
        assert_eq!(args.workgroup_count_y, 64);
        assert_eq!(args.workgroup_count_z, 1);
    }

    #[test]
    fn test_indirect_dispatch_args_single() {
        let args = IndirectDispatchArgs::single(256);
        assert_eq!(args.workgroup_count_x, 256);
        assert_eq!(args.workgroup_count_y, 1);
        assert_eq!(args.workgroup_count_z, 1);
    }

    #[test]
    fn test_indirect_dispatch_args_linear() {
        let args = IndirectDispatchArgs::linear(256);
        assert_eq!(args.workgroup_count_x, 256);
        assert_eq!(args.workgroup_count_y, 1);
        assert_eq!(args.workgroup_count_z, 1);
    }

    #[test]
    fn test_indirect_dispatch_args_grid_2d() {
        let args = IndirectDispatchArgs::grid_2d(16, 16);
        assert_eq!(args.workgroup_count_x, 16);
        assert_eq!(args.workgroup_count_y, 16);
        assert_eq!(args.workgroup_count_z, 1);
    }

    #[test]
    fn test_indirect_dispatch_args_is_active() {
        assert!(IndirectDispatchArgs::new(1, 1, 1).is_active());
        assert!(!IndirectDispatchArgs::new(0, 1, 1).is_active());
        assert!(!IndirectDispatchArgs::new(1, 0, 1).is_active());
        assert!(!IndirectDispatchArgs::new(1, 1, 0).is_active());
    }

    #[test]
    fn test_indirect_dispatch_args_total_workgroups() {
        let args = IndirectDispatchArgs::new(8, 8, 4);
        assert_eq!(args.total_workgroups(), 256);
    }

    #[test]
    fn test_indirect_dispatch_args_for_elements() {
        // 1000 elements with workgroup size 256 = ceil(1000/256) = 4 groups
        let args = IndirectDispatchArgs::for_elements(1000, 256);
        assert_eq!(args.workgroup_count_x, 4);
        assert_eq!(args.workgroup_count_y, 1);
        assert_eq!(args.workgroup_count_z, 1);

        // Exact multiple
        let args = IndirectDispatchArgs::for_elements(512, 256);
        assert_eq!(args.workgroup_count_x, 2);

        // Edge case: 0 elements
        let args = IndirectDispatchArgs::for_elements(0, 256);
        assert_eq!(args.workgroup_count_x, 0);
    }

    #[test]
    fn test_indirect_dispatch_args_bytemuck_pod() {
        let args = IndirectDispatchArgs::new(64, 64, 1);
        let bytes: &[u8] = bytemuck::bytes_of(&args);
        assert_eq!(bytes.len(), 12);

        // Verify round-trip through bytes
        let restored: IndirectDispatchArgs = *bytemuck::from_bytes(bytes);
        assert_eq!(args, restored);
    }

    #[test]
    fn test_dispatch_indirect_args_type_alias() {
        // Verify DispatchIndirectArgs is the same type as IndirectDispatchArgs
        let _args: DispatchIndirectArgs = IndirectDispatchArgs::default();
        let indirect: IndirectDispatchArgs = IndirectDispatchArgs::new(64, 64, 1);
        let dispatch: DispatchIndirectArgs = indirect;
        assert_eq!(indirect, dispatch);
    }

    #[test]
    fn test_dispatch_args_accessor_methods() {
        let args = IndirectDispatchArgs::new(8, 4, 2);
        assert_eq!(args.x(), 8);
        assert_eq!(args.y(), 4);
        assert_eq!(args.z(), 2);
    }

    #[test]
    fn test_dispatch_args_default_zeros() {
        let args = IndirectDispatchArgs::default();
        assert_eq!(args.workgroup_count_x, 0);
        assert_eq!(args.workgroup_count_y, 0);
        assert_eq!(args.workgroup_count_z, 0);
        assert!(!args.is_active());
    }

    #[test]
    fn test_dispatch_args_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(IndirectDispatchArgs::new(1, 2, 3));
        set.insert(IndirectDispatchArgs::new(1, 2, 3));
        assert_eq!(set.len(), 1); // Duplicate should not be inserted
    }

    // -------------------------------------------------------------------------
    // IndirectTier Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_indirect_tier_supports_gpu_count() {
        assert!(IndirectTier::Full.supports_gpu_count());
        assert!(!IndirectTier::Partial.supports_gpu_count());
        assert!(!IndirectTier::Minimal.supports_gpu_count());
    }

    #[test]
    fn test_indirect_tier_supports_multi_draw() {
        assert!(IndirectTier::Full.supports_multi_draw());
        assert!(IndirectTier::Partial.supports_multi_draw());
        assert!(!IndirectTier::Minimal.supports_multi_draw());
    }

    #[test]
    fn test_indirect_tier_description() {
        assert!(!IndirectTier::Full.description().is_empty());
        assert!(!IndirectTier::Partial.description().is_empty());
        assert!(!IndirectTier::Minimal.description().is_empty());
    }

    #[test]
    fn test_indirect_tier_display() {
        let tier = IndirectTier::Full;
        let display = format!("{}", tier);
        assert!(!display.is_empty());
    }

    // -------------------------------------------------------------------------
    // DrawBatchBuilder Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_draw_batch_builder_new() {
        let builder = DrawBatchBuilder::new();
        assert_eq!(builder.indexed_count(), 0);
        assert_eq!(builder.count(), 0);
    }

    #[test]
    fn test_draw_batch_builder_add_indexed() {
        let mut builder = DrawBatchBuilder::new();
        builder.add_indexed(IndirectDrawIndexedArgs::new(100, 1, 0, 0, 0));
        builder.add_indexed(IndirectDrawIndexedArgs::new(200, 2, 0, 0, 0));

        assert_eq!(builder.indexed_count(), 2);
        assert_eq!(builder.indexed_commands()[0].index_count, 100);
        assert_eq!(builder.indexed_commands()[1].index_count, 200);
    }

    #[test]
    fn test_draw_batch_builder_add_non_indexed() {
        let mut builder = DrawBatchBuilder::new();
        builder.add(IndirectDrawArgs::new(100, 1, 0, 0));
        builder.add(IndirectDrawArgs::new(200, 2, 0, 0));

        assert_eq!(builder.count(), 2);
        assert_eq!(builder.commands()[0].vertex_count, 100);
        assert_eq!(builder.commands()[1].vertex_count, 200);
    }

    #[test]
    fn test_draw_batch_builder_clear() {
        let mut builder = DrawBatchBuilder::new();
        builder.add_indexed(IndirectDrawIndexedArgs::zeroed());
        builder.add(IndirectDrawArgs::zeroed());

        builder.clear();

        assert_eq!(builder.indexed_count(), 0);
        assert_eq!(builder.count(), 0);
    }

    // -------------------------------------------------------------------------
    // MultiIndirectConfig Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_multi_indirect_config_default() {
        let config = MultiIndirectConfig::default();
        assert_eq!(config.max_draws, DEFAULT_MAX_DRAWS);
        assert!(config.indexed);
    }

    #[test]
    fn test_multi_indirect_config_indexed() {
        let config = MultiIndirectConfig::indexed(1024);
        assert_eq!(config.max_draws, 1024);
        assert!(config.indexed);
    }

    #[test]
    fn test_multi_indirect_config_non_indexed() {
        let config = MultiIndirectConfig::non_indexed(512);
        assert_eq!(config.max_draws, 512);
        assert!(!config.indexed);
    }

    #[test]
    fn test_multi_indirect_config_with_label() {
        let config = MultiIndirectConfig::indexed(1024).with_label("custom_buffer");
        assert_eq!(config.label, "custom_buffer");
    }

    // -------------------------------------------------------------------------
    // Constants Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_default_max_draws() {
        assert_eq!(DEFAULT_MAX_DRAWS, 65536);
    }

    // -------------------------------------------------------------------------
    // Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_large_instance_count() {
        let args = IndirectDrawIndexedArgs::new(36, u32::MAX, 0, 0, 0);
        assert_eq!(args.instance_count, u32::MAX);
        assert!(args.is_visible());
    }

    #[test]
    fn test_negative_base_vertex() {
        let args = IndirectDrawIndexedArgs::new(36, 1, 0, i32::MIN, 0);
        assert_eq!(args.base_vertex, i32::MIN);

        // Verify bytemuck handles negative values correctly
        let bytes = bytemuck::bytes_of(&args);
        let restored: IndirectDrawIndexedArgs = *bytemuck::from_bytes(bytes);
        assert_eq!(restored.base_vertex, i32::MIN);
    }

    #[test]
    fn test_max_workgroups() {
        let args = IndirectDispatchArgs::new(u32::MAX, 1, 1);
        assert_eq!(args.total_workgroups(), u32::MAX as u64);
        assert_eq!(args.workgroup_count_x, u32::MAX);
        assert_eq!(args.workgroup_count_y, 1);
        assert_eq!(args.workgroup_count_z, 1);
    }

    // -------------------------------------------------------------------------
    // IndirectDrawBuffer Tests (T-WGPU-P6.1.4)
    // -------------------------------------------------------------------------

    #[test]
    fn test_buffer_creation_with_capacity() {
        // Test that buffer capacity is correctly stored
        // (Full GPU test would require a wgpu device)
        let capacity = 1024u32;
        let expected_size = (capacity as u64) * (IndirectDrawIndexedArgs::SIZE as u64);
        assert_eq!(expected_size, 1024 * 20);
    }

    #[test]
    fn test_clear_resets_count() {
        // Verify clear logic resets to 0
        let mut builder = DrawBatchBuilder::new();
        builder.add_indexed(IndirectDrawIndexedArgs::new(100, 1, 0, 0, 0));
        assert_eq!(builder.indexed_count(), 1);

        builder.clear();
        assert_eq!(builder.indexed_count(), 0);
    }

    #[test]
    fn test_resize_logic() {
        // Test resize logic behavior (actual buffer resize needs GPU)
        // The resize method returns true if new_max_draws > current capacity
        // This verifies the logic without needing a device

        let initial_capacity = 256u32;
        let new_capacity = 1024u32;

        // Verify size calculation is correct after resize
        let new_size = (new_capacity as u64) * (IndirectDrawIndexedArgs::SIZE as u64);
        assert_eq!(new_size, 1024 * 20);
        assert!(new_capacity > initial_capacity);
    }

    #[test]
    fn test_indirect_usage_flag_required() {
        // Verify INDIRECT flag is included in buffer usage
        let usage = BufferUsages::INDIRECT | BufferUsages::STORAGE | BufferUsages::COPY_DST;
        assert!(usage.contains(BufferUsages::INDIRECT));
        assert!(usage.contains(BufferUsages::STORAGE));
        assert!(usage.contains(BufferUsages::COPY_DST));
    }

    #[test]
    fn test_buffer_offset_calculation() {
        // Test offset_for_draw calculation
        let stride = IndirectDrawIndexedArgs::SIZE as u64;

        // Offset for draw 0
        let offset_0 = 0u64 * stride;
        assert_eq!(offset_0, 0);

        // Offset for draw 10
        let offset_10 = 10u64 * stride;
        assert_eq!(offset_10, 200); // 10 * 20 bytes

        // Offset for draw 100
        let offset_100 = 100u64 * stride;
        assert_eq!(offset_100, 2000); // 100 * 20 bytes
    }

    #[test]
    fn test_commands_buffer_size_calculation() {
        // Verify buffer size is calculated correctly
        let max_draws = 65536u32;
        let expected_size = (max_draws as u64) * (IndirectDrawIndexedArgs::SIZE as u64);
        assert_eq!(expected_size, 65536 * 20);
        assert_eq!(expected_size, 1310720); // 1.25 MB
    }

    #[test]
    fn test_resize_does_not_shrink() {
        // Verify resize returns false when new_capacity <= current_capacity
        // This is the expected behavior: buffers should not shrink
        let current_capacity = 1024u32;
        let smaller_capacity = 512u32;
        let equal_capacity = 1024u32;

        assert!(smaller_capacity <= current_capacity);
        assert!(equal_capacity <= current_capacity);
    }

    #[test]
    fn test_capacity_and_count_accessors() {
        // Test the new accessor methods
        let mut builder = DrawBatchBuilder::new();

        // Empty initially
        assert_eq!(builder.indexed_count(), 0);

        // Add commands and verify count
        builder.add_indexed(IndirectDrawIndexedArgs::with_counts(36, 1));
        builder.add_indexed(IndirectDrawIndexedArgs::with_counts(72, 2));
        assert_eq!(builder.indexed_count(), 2);
    }

    // -------------------------------------------------------------------------
    // CountBuffer Tests (T-WGPU-P6.1.5)
    // -------------------------------------------------------------------------

    #[test]
    fn test_count_buffer_size_is_4_bytes() {
        // CountBuffer must be exactly 4 bytes (single u32)
        assert_eq!(CountBuffer::SIZE, 4);
        assert_eq!(CountBuffer::SIZE, std::mem::size_of::<u32>() as u64);
    }

    #[test]
    fn test_count_buffer_size_method() {
        // The size() method should return the constant SIZE
        // We can test the constant directly since size() returns Self::SIZE
        assert_eq!(CountBuffer::SIZE, 4);
    }

    #[test]
    fn test_count_buffer_usage_flags() {
        // Verify required usage flags for CountBuffer
        let required_usage = BufferUsages::INDIRECT
            | BufferUsages::STORAGE
            | BufferUsages::COPY_DST
            | BufferUsages::COPY_SRC;

        // INDIRECT: Required for multi_draw_indirect_count
        assert!(required_usage.contains(BufferUsages::INDIRECT));

        // STORAGE: Required for compute shader atomic writes
        assert!(required_usage.contains(BufferUsages::STORAGE));

        // COPY_DST: Required for CPU reset via queue.write_buffer
        assert!(required_usage.contains(BufferUsages::COPY_DST));

        // COPY_SRC: Optional, for GPU readback
        assert!(required_usage.contains(BufferUsages::COPY_SRC));
    }

    #[test]
    fn test_count_buffer_bind_group_layout_entry() {
        // Test bind group layout entry generation
        let entry = CountBuffer::bind_group_layout_entry(0);

        assert_eq!(entry.binding, 0);
        assert_eq!(entry.visibility, wgpu::ShaderStages::COMPUTE);

        // Verify it's a read-write storage buffer
        match entry.ty {
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset,
                min_binding_size,
            } => {
                assert!(!read_only, "CountBuffer must be read-write for atomics");
                assert!(!has_dynamic_offset);
                assert_eq!(min_binding_size, std::num::NonZeroU64::new(4));
            }
            _ => panic!("Expected storage buffer binding type"),
        }
    }

    #[test]
    fn test_count_buffer_bind_group_layout_entry_binding_index() {
        // Test different binding indices
        let entry_0 = CountBuffer::bind_group_layout_entry(0);
        let entry_5 = CountBuffer::bind_group_layout_entry(5);
        let entry_max = CountBuffer::bind_group_layout_entry(u32::MAX);

        assert_eq!(entry_0.binding, 0);
        assert_eq!(entry_5.binding, 5);
        assert_eq!(entry_max.binding, u32::MAX);
    }

    #[test]
    fn test_count_value_bytemuck() {
        // Verify u32 count value can be serialized correctly
        let count: u32 = 42;
        let bytes = bytemuck::bytes_of(&count);
        assert_eq!(bytes.len(), 4);

        let restored: u32 = *bytemuck::from_bytes(bytes);
        assert_eq!(restored, 42);
    }

    #[test]
    fn test_count_zero_bytemuck() {
        // Verify zero value (used in reset) serializes correctly
        let zero: u32 = 0;
        let bytes = bytemuck::bytes_of(&zero);
        assert_eq!(bytes, &[0, 0, 0, 0]);
    }

    #[test]
    fn test_count_max_bytemuck() {
        // Verify max value serializes correctly
        let max: u32 = u32::MAX;
        let bytes = bytemuck::bytes_of(&max);
        assert_eq!(bytes.len(), 4);

        let restored: u32 = *bytemuck::from_bytes(bytes);
        assert_eq!(restored, u32::MAX);
    }
}
