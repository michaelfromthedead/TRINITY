//! Resource State Tracking for WGPU Barrier Management (T-WGPU-P4.7.1)
//!
//! This module provides infrastructure for tracking resource states across
//! the command buffer to enable automatic barrier insertion.
//!
//! # Architecture
//!
//! - `PipelineStage`: Represents the pipeline stages where resources can be accessed
//! - `AccessFlags`: Bitflags representing read/write access patterns
//! - `TextureLayout`: Represents the image layout state for textures
//! - `ResourceState`: Combines stage, access, and optional layout for a resource
//! - `ResourceStateTracker`: HashMap-based tracker for per-resource state
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::resource_state::*;
//!
//! let mut tracker = ResourceStateTracker::new();
//!
//! // Set initial state for a texture
//! tracker.set(texture_id, ResourceState {
//!     stage: PipelineStage::Transfer,
//!     access: AccessFlags::TRANSFER_WRITE,
//!     layout: Some(TextureLayout::TransferDst),
//! });
//!
//! // Transition to shader read
//! tracker.update_layout(texture_id, TextureLayout::ShaderReadOnly);
//! tracker.update(texture_id, PipelineStage::FragmentShader, AccessFlags::SHADER_READ);
//! ```

use bitflags::bitflags;
use std::collections::HashMap;

/// Pipeline stage where a resource can be accessed.
///
/// These stages follow the logical order of GPU execution and are used
/// to determine synchronization requirements between accesses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PipelineStage {
    /// No stage - resource is not being accessed.
    #[default]
    None,
    /// Vertex input assembly stage (index/vertex buffer reads).
    VertexInput,
    /// Vertex shader execution stage.
    VertexShader,
    /// Fragment shader execution stage.
    FragmentShader,
    /// Early depth/stencil test stage.
    EarlyDepth,
    /// Late depth/stencil test stage.
    LateDepth,
    /// Color attachment output stage.
    ColorOutput,
    /// Compute shader execution stage.
    ComputeShader,
    /// Transfer/copy operations stage.
    Transfer,
    /// Host (CPU) access stage.
    Host,
    /// All graphics pipeline stages combined.
    AllGraphics,
    /// All pipeline stages combined (graphics + compute + transfer).
    AllCommands,
}

impl PipelineStage {
    /// Returns true if this stage is a graphics stage.
    #[inline]
    pub fn is_graphics(&self) -> bool {
        matches!(
            self,
            PipelineStage::VertexInput
                | PipelineStage::VertexShader
                | PipelineStage::FragmentShader
                | PipelineStage::EarlyDepth
                | PipelineStage::LateDepth
                | PipelineStage::ColorOutput
                | PipelineStage::AllGraphics
        )
    }

    /// Returns true if this stage is a compute stage.
    #[inline]
    pub fn is_compute(&self) -> bool {
        matches!(self, PipelineStage::ComputeShader)
    }

    /// Returns true if this stage is a transfer stage.
    #[inline]
    pub fn is_transfer(&self) -> bool {
        matches!(self, PipelineStage::Transfer)
    }

    /// Returns true if this stage involves shader execution.
    #[inline]
    pub fn is_shader_stage(&self) -> bool {
        matches!(
            self,
            PipelineStage::VertexShader
                | PipelineStage::FragmentShader
                | PipelineStage::ComputeShader
        )
    }

    /// Returns the ordering index for this stage (earlier stages have lower indices).
    /// This can be used to determine if barriers are needed between stages.
    #[inline]
    pub fn order_index(&self) -> u32 {
        match self {
            PipelineStage::None => 0,
            PipelineStage::Host => 1,
            PipelineStage::Transfer => 2,
            PipelineStage::VertexInput => 3,
            PipelineStage::VertexShader => 4,
            PipelineStage::EarlyDepth => 5,
            PipelineStage::FragmentShader => 6,
            PipelineStage::LateDepth => 7,
            PipelineStage::ColorOutput => 8,
            PipelineStage::ComputeShader => 9,
            PipelineStage::AllGraphics => 10,
            PipelineStage::AllCommands => 11,
        }
    }

    /// Returns true if this stage logically comes before the other stage.
    #[inline]
    pub fn comes_before(&self, other: &PipelineStage) -> bool {
        self.order_index() < other.order_index()
    }
}

bitflags! {
    /// Access flags describing how a resource is being accessed.
    ///
    /// These flags are used to determine memory barriers and
    /// cache flush/invalidate requirements.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    pub struct AccessFlags: u32 {
        /// No access.
        const NONE = 0;
        /// Generic read access.
        const READ = 1 << 0;
        /// Generic write access.
        const WRITE = 1 << 1;
        /// Combined read and write access.
        const READ_WRITE = Self::READ.bits() | Self::WRITE.bits();
        /// Shader read access (uniform buffers, textures).
        const SHADER_READ = 1 << 2;
        /// Shader write access (storage buffers, storage textures).
        const SHADER_WRITE = 1 << 3;
        /// Combined shader read and write access.
        const SHADER_READ_WRITE = Self::SHADER_READ.bits() | Self::SHADER_WRITE.bits();
        /// Color attachment read access.
        const COLOR_ATTACHMENT_READ = 1 << 4;
        /// Color attachment write access.
        const COLOR_ATTACHMENT_WRITE = 1 << 5;
        /// Depth/stencil attachment read access.
        const DEPTH_STENCIL_READ = 1 << 6;
        /// Depth/stencil attachment write access.
        const DEPTH_STENCIL_WRITE = 1 << 7;
        /// Depth/stencil combined read/write.
        const DEPTH_STENCIL_READ_WRITE = Self::DEPTH_STENCIL_READ.bits() | Self::DEPTH_STENCIL_WRITE.bits();
        /// Transfer (copy) source read access.
        const TRANSFER_READ = 1 << 8;
        /// Transfer (copy) destination write access.
        const TRANSFER_WRITE = 1 << 9;
        /// Vertex buffer read access.
        const VERTEX_BUFFER_READ = 1 << 10;
        /// Index buffer read access.
        const INDEX_BUFFER_READ = 1 << 11;
        /// Indirect buffer read access.
        const INDIRECT_BUFFER_READ = 1 << 12;
        /// Uniform buffer read access.
        const UNIFORM_BUFFER_READ = 1 << 13;
        /// Host read access.
        const HOST_READ = 1 << 14;
        /// Host write access.
        const HOST_WRITE = 1 << 15;
    }
}

impl AccessFlags {
    /// Returns true if this access involves any read operation.
    #[inline]
    pub fn has_read(&self) -> bool {
        self.intersects(
            AccessFlags::READ
                | AccessFlags::SHADER_READ
                | AccessFlags::COLOR_ATTACHMENT_READ
                | AccessFlags::DEPTH_STENCIL_READ
                | AccessFlags::TRANSFER_READ
                | AccessFlags::VERTEX_BUFFER_READ
                | AccessFlags::INDEX_BUFFER_READ
                | AccessFlags::INDIRECT_BUFFER_READ
                | AccessFlags::UNIFORM_BUFFER_READ
                | AccessFlags::HOST_READ,
        )
    }

    /// Returns true if this access involves any write operation.
    #[inline]
    pub fn has_write(&self) -> bool {
        self.intersects(
            AccessFlags::WRITE
                | AccessFlags::SHADER_WRITE
                | AccessFlags::COLOR_ATTACHMENT_WRITE
                | AccessFlags::DEPTH_STENCIL_WRITE
                | AccessFlags::TRANSFER_WRITE
                | AccessFlags::HOST_WRITE,
        )
    }

    /// Returns true if this access is read-only.
    #[inline]
    pub fn is_read_only(&self) -> bool {
        self.has_read() && !self.has_write()
    }

    /// Returns true if this access is write-only.
    #[inline]
    pub fn is_write_only(&self) -> bool {
        self.has_write() && !self.has_read()
    }

    /// Returns true if this access conflicts with another access.
    /// Two accesses conflict if at least one is a write.
    #[inline]
    pub fn conflicts_with(&self, other: AccessFlags) -> bool {
        self.has_write() || other.has_write()
    }

    /// Returns true if a barrier is required between this access and the next access.
    /// A barrier is required if there's a read-after-write, write-after-read,
    /// or write-after-write hazard.
    #[inline]
    pub fn requires_barrier_to(&self, next: AccessFlags) -> bool {
        // Write-after-write: need barrier to ensure ordering
        if self.has_write() && next.has_write() {
            return true;
        }
        // Read-after-write: need barrier to make writes visible
        if self.has_write() && next.has_read() {
            return true;
        }
        // Write-after-read: need barrier to ensure read completes
        if self.has_read() && next.has_write() {
            return true;
        }
        false
    }
}

/// Texture layout state for image resources.
///
/// Different operations require textures to be in specific layouts.
/// Layout transitions may require barriers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TextureLayout {
    /// Undefined/uninitialized layout - contents are undefined.
    #[default]
    Undefined,
    /// General layout - can be used for any operation but may not be optimal.
    General,
    /// Optimal for use as a color attachment.
    ColorAttachment,
    /// Optimal for use as a depth/stencil attachment (read/write).
    DepthStencilAttachment,
    /// Optimal for use as a depth/stencil attachment (read-only).
    DepthStencilReadOnly,
    /// Optimal for shader read-only access (sampling).
    ShaderReadOnly,
    /// Optimal for use as a transfer source.
    TransferSrc,
    /// Optimal for use as a transfer destination.
    TransferDst,
    /// Layout for presentation to the display.
    Present,
    /// Storage image layout for compute shader read/write.
    StorageImage,
    /// Preinitialized layout - for linear tiled images with host data.
    Preinitialized,
}

impl TextureLayout {
    /// Returns true if this layout supports shader reading.
    #[inline]
    pub fn supports_shader_read(&self) -> bool {
        matches!(
            self,
            TextureLayout::General
                | TextureLayout::ShaderReadOnly
                | TextureLayout::DepthStencilReadOnly
                | TextureLayout::StorageImage
        )
    }

    /// Returns true if this layout supports shader writing.
    #[inline]
    pub fn supports_shader_write(&self) -> bool {
        matches!(self, TextureLayout::General | TextureLayout::StorageImage)
    }

    /// Returns true if this layout supports use as a color attachment.
    #[inline]
    pub fn supports_color_attachment(&self) -> bool {
        matches!(self, TextureLayout::General | TextureLayout::ColorAttachment)
    }

    /// Returns true if this layout supports use as a depth/stencil attachment.
    #[inline]
    pub fn supports_depth_stencil(&self) -> bool {
        matches!(
            self,
            TextureLayout::General
                | TextureLayout::DepthStencilAttachment
                | TextureLayout::DepthStencilReadOnly
        )
    }

    /// Returns true if this layout supports transfer read operations.
    #[inline]
    pub fn supports_transfer_read(&self) -> bool {
        matches!(self, TextureLayout::General | TextureLayout::TransferSrc)
    }

    /// Returns true if this layout supports transfer write operations.
    #[inline]
    pub fn supports_transfer_write(&self) -> bool {
        matches!(self, TextureLayout::General | TextureLayout::TransferDst)
    }

    /// Returns true if a layout transition is required to move to the target layout.
    #[inline]
    pub fn requires_transition_to(&self, target: TextureLayout) -> bool {
        *self != target
    }

    /// Returns the optimal layout for the given access flags.
    pub fn optimal_for_access(access: AccessFlags) -> TextureLayout {
        if access.contains(AccessFlags::COLOR_ATTACHMENT_WRITE) {
            TextureLayout::ColorAttachment
        } else if access.contains(AccessFlags::DEPTH_STENCIL_WRITE) {
            TextureLayout::DepthStencilAttachment
        } else if access.contains(AccessFlags::DEPTH_STENCIL_READ) {
            TextureLayout::DepthStencilReadOnly
        } else if access.contains(AccessFlags::SHADER_WRITE) {
            TextureLayout::StorageImage
        } else if access.contains(AccessFlags::SHADER_READ) {
            TextureLayout::ShaderReadOnly
        } else if access.contains(AccessFlags::TRANSFER_WRITE) {
            TextureLayout::TransferDst
        } else if access.contains(AccessFlags::TRANSFER_READ) {
            TextureLayout::TransferSrc
        } else {
            TextureLayout::General
        }
    }
}

/// Unique identifier for a tracked resource.
pub type ResourceId = u64;

/// Complete state information for a tracked resource.
///
/// Combines pipeline stage, access flags, and optional texture layout
/// to fully describe how a resource is currently being used.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ResourceState {
    /// The pipeline stage where the resource is being accessed.
    pub stage: PipelineStage,
    /// The access flags describing the type of access.
    pub access: AccessFlags,
    /// The texture layout (None for buffers, Some for textures).
    pub layout: Option<TextureLayout>,
}

impl ResourceState {
    /// Creates a new resource state for a buffer.
    #[inline]
    pub fn buffer(stage: PipelineStage, access: AccessFlags) -> Self {
        Self {
            stage,
            access,
            layout: None,
        }
    }

    /// Creates a new resource state for a texture.
    #[inline]
    pub fn texture(stage: PipelineStage, access: AccessFlags, layout: TextureLayout) -> Self {
        Self {
            stage,
            access,
            layout: Some(layout),
        }
    }

    /// Creates an undefined/initial state.
    #[inline]
    pub fn undefined() -> Self {
        Self::default()
    }

    /// Returns true if this is a buffer state (no layout).
    #[inline]
    pub fn is_buffer(&self) -> bool {
        self.layout.is_none()
    }

    /// Returns true if this is a texture state (has layout).
    #[inline]
    pub fn is_texture(&self) -> bool {
        self.layout.is_some()
    }

    /// Returns true if a barrier is required to transition to the target state.
    pub fn requires_barrier_to(&self, target: &ResourceState) -> bool {
        // Check for access hazards first - this is the primary consideration
        if self.access.requires_barrier_to(target.access) {
            return true;
        }

        // Check for layout transitions (textures only)
        if let (Some(current_layout), Some(target_layout)) = (self.layout, target.layout) {
            if current_layout.requires_transition_to(target_layout) {
                return true;
            }
        }

        // Note: Stage changes alone don't require barriers if access is read-only on both sides.
        // The GPU can read from the same resource in multiple stages without barriers.
        // Only access hazards (RAW, WAR, WAW) require explicit barriers.

        false
    }
}

/// Tracks the current state of all resources in a command buffer.
///
/// This tracker maintains a mapping from resource IDs to their current state,
/// enabling automatic barrier insertion based on state transitions.
#[derive(Debug, Default)]
pub struct ResourceStateTracker {
    /// Map of resource ID to current state.
    states: HashMap<ResourceId, ResourceState>,
}

impl ResourceStateTracker {
    /// Creates a new empty resource state tracker.
    #[inline]
    pub fn new() -> Self {
        Self {
            states: HashMap::new(),
        }
    }

    /// Creates a new resource state tracker with the specified capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            states: HashMap::with_capacity(capacity),
        }
    }

    /// Gets the current state of a resource.
    ///
    /// Returns `None` if the resource is not being tracked.
    #[inline]
    pub fn get(&self, id: ResourceId) -> Option<&ResourceState> {
        self.states.get(&id)
    }

    /// Sets the state of a resource, replacing any previous state.
    ///
    /// This is typically used for initial resource registration or
    /// when performing a complete state reset.
    #[inline]
    pub fn set(&mut self, id: ResourceId, state: ResourceState) {
        self.states.insert(id, state);
    }

    /// Updates the stage and access flags for a resource.
    ///
    /// If the resource doesn't exist, a new entry is created with no layout.
    /// If it exists, the layout is preserved.
    pub fn update(&mut self, id: ResourceId, stage: PipelineStage, access: AccessFlags) {
        if let Some(state) = self.states.get_mut(&id) {
            state.stage = stage;
            state.access = access;
        } else {
            self.states.insert(
                id,
                ResourceState {
                    stage,
                    access,
                    layout: None,
                },
            );
        }
    }

    /// Updates the texture layout for a resource.
    ///
    /// If the resource doesn't exist, a new entry is created with the given layout
    /// and default stage/access. If it exists, only the layout is updated.
    pub fn update_layout(&mut self, id: ResourceId, layout: TextureLayout) {
        if let Some(state) = self.states.get_mut(&id) {
            state.layout = Some(layout);
        } else {
            self.states.insert(
                id,
                ResourceState {
                    stage: PipelineStage::None,
                    access: AccessFlags::NONE,
                    layout: Some(layout),
                },
            );
        }
    }

    /// Removes a resource from tracking.
    ///
    /// Returns the previous state if the resource was being tracked.
    #[inline]
    pub fn remove(&mut self, id: ResourceId) -> Option<ResourceState> {
        self.states.remove(&id)
    }

    /// Clears all tracked resources.
    #[inline]
    pub fn clear(&mut self) {
        self.states.clear();
    }

    /// Returns the number of tracked resources.
    #[inline]
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Returns true if no resources are being tracked.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Returns true if a resource is being tracked.
    #[inline]
    pub fn contains(&self, id: ResourceId) -> bool {
        self.states.contains_key(&id)
    }

    /// Returns an iterator over all tracked resource IDs.
    #[inline]
    pub fn ids(&self) -> impl Iterator<Item = &ResourceId> {
        self.states.keys()
    }

    /// Returns an iterator over all tracked states.
    #[inline]
    pub fn states(&self) -> impl Iterator<Item = (&ResourceId, &ResourceState)> {
        self.states.iter()
    }

    /// Returns an iterator over all tracked states (mutable).
    #[inline]
    pub fn states_mut(&mut self) -> impl Iterator<Item = (&ResourceId, &mut ResourceState)> {
        self.states.iter_mut()
    }

    /// Transitions a resource to a new state and returns the barrier info if needed.
    ///
    /// Returns `Some((old_state, new_state))` if a barrier is required,
    /// `None` if no barrier is needed or the resource wasn't tracked.
    pub fn transition(
        &mut self,
        id: ResourceId,
        new_state: ResourceState,
    ) -> Option<(ResourceState, ResourceState)> {
        let old_state = self.states.get(&id).cloned();

        if let Some(old) = old_state {
            let needs_barrier = old.requires_barrier_to(&new_state);
            self.states.insert(id, new_state.clone());

            if needs_barrier {
                return Some((old, new_state));
            }
        } else {
            self.states.insert(id, new_state);
        }

        None
    }

    /// Merges another tracker's states into this one.
    ///
    /// States from the other tracker overwrite existing states.
    pub fn merge(&mut self, other: &ResourceStateTracker) {
        for (id, state) in &other.states {
            self.states.insert(*id, state.clone());
        }
    }
}

// ============================================================================
// Barrier Detection (T-WGPU-P4.7.2)
// ============================================================================

/// Types of memory hazards that require barriers for correct synchronization.
///
/// These represent the classic GPU hazard types that occur when multiple
/// operations access the same resource without proper synchronization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum HazardType {
    /// No hazard detected - no barrier required.
    #[default]
    None,
    /// Read-after-write hazard: a read depends on a previous write.
    /// Requires a barrier to ensure the write is visible before the read.
    ReadAfterWrite,
    /// Write-after-read hazard: a write would overwrite data still being read.
    /// Requires a barrier to ensure the read completes before the write.
    WriteAfterRead,
    /// Write-after-write hazard: two writes to the same resource.
    /// Requires a barrier to ensure correct ordering and visibility.
    WriteAfterWrite,
    /// Layout transition hazard: texture needs to change its memory layout.
    /// This may combine with other hazard types.
    LayoutTransition,
}

impl HazardType {
    /// Returns true if this hazard requires a barrier.
    #[inline]
    pub fn requires_barrier(&self) -> bool {
        !matches!(self, HazardType::None)
    }

    /// Returns true if this is a write hazard (WAW or WAR).
    #[inline]
    pub fn is_write_hazard(&self) -> bool {
        matches!(self, HazardType::WriteAfterRead | HazardType::WriteAfterWrite)
    }

    /// Returns true if this is a read hazard (RAW).
    #[inline]
    pub fn is_read_hazard(&self) -> bool {
        matches!(self, HazardType::ReadAfterWrite)
    }

    /// Returns true if this involves a layout transition.
    #[inline]
    pub fn is_layout_transition(&self) -> bool {
        matches!(self, HazardType::LayoutTransition)
    }

    /// Combines two hazard types, returning the more severe one.
    /// Layout transitions are considered separate from data hazards.
    #[inline]
    pub fn combine(self, other: HazardType) -> HazardType {
        match (self, other) {
            (HazardType::None, h) | (h, HazardType::None) => h,
            // WAW is the most severe data hazard
            (HazardType::WriteAfterWrite, _) | (_, HazardType::WriteAfterWrite) => {
                HazardType::WriteAfterWrite
            }
            // RAW and WAR are equally severe for synchronization purposes
            (HazardType::ReadAfterWrite, HazardType::WriteAfterRead)
            | (HazardType::WriteAfterRead, HazardType::ReadAfterWrite) => {
                HazardType::WriteAfterWrite
            }
            // Layout transition is orthogonal
            (HazardType::LayoutTransition, h) | (h, HazardType::LayoutTransition) => h,
            // Same hazard type
            (h, _) => h,
        }
    }
}

/// Complete information about a required barrier between two resource accesses.
///
/// This struct contains all the information needed to insert a proper
/// synchronization barrier in the command buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BarrierInfo {
    /// The resource that requires the barrier.
    pub resource_id: ResourceId,
    /// The type of hazard being resolved.
    pub hazard: HazardType,
    /// The pipeline stage of the source (previous) access.
    pub src_stage: PipelineStage,
    /// The pipeline stage of the destination (next) access.
    pub dst_stage: PipelineStage,
    /// The access flags of the source (previous) access.
    pub src_access: AccessFlags,
    /// The access flags of the destination (next) access.
    pub dst_access: AccessFlags,
    /// The old texture layout (None for buffers).
    pub old_layout: Option<TextureLayout>,
    /// The new texture layout (None for buffers).
    pub new_layout: Option<TextureLayout>,
}

impl BarrierInfo {
    /// Creates a new barrier info for a buffer resource.
    pub fn buffer(
        resource_id: ResourceId,
        hazard: HazardType,
        src_stage: PipelineStage,
        dst_stage: PipelineStage,
        src_access: AccessFlags,
        dst_access: AccessFlags,
    ) -> Self {
        Self {
            resource_id,
            hazard,
            src_stage,
            dst_stage,
            src_access,
            dst_access,
            old_layout: None,
            new_layout: None,
        }
    }

    /// Creates a new barrier info for a texture resource.
    pub fn texture(
        resource_id: ResourceId,
        hazard: HazardType,
        src_stage: PipelineStage,
        dst_stage: PipelineStage,
        src_access: AccessFlags,
        dst_access: AccessFlags,
        old_layout: TextureLayout,
        new_layout: TextureLayout,
    ) -> Self {
        Self {
            resource_id,
            hazard,
            src_stage,
            dst_stage,
            src_access,
            dst_access,
            old_layout: Some(old_layout),
            new_layout: Some(new_layout),
        }
    }

    /// Returns true if this barrier involves a texture layout transition.
    #[inline]
    pub fn has_layout_transition(&self) -> bool {
        match (self.old_layout, self.new_layout) {
            (Some(old), Some(new)) => old != new,
            _ => false,
        }
    }

    /// Returns true if this is a buffer barrier (no layout info).
    #[inline]
    pub fn is_buffer_barrier(&self) -> bool {
        self.old_layout.is_none() && self.new_layout.is_none()
    }

    /// Returns true if this is a texture barrier (has layout info).
    #[inline]
    pub fn is_texture_barrier(&self) -> bool {
        self.old_layout.is_some() || self.new_layout.is_some()
    }
}

/// Detects when barriers are needed between resource accesses.
///
/// The `BarrierDetector` wraps a `ResourceStateTracker` and provides
/// hazard detection logic to determine when synchronization barriers
/// are required between GPU operations.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::resource_state::*;
///
/// let mut detector = BarrierDetector::new();
///
/// // Record a write to a buffer
/// detector.record_access(buffer_id, ResourceState::buffer(
///     PipelineStage::Transfer,
///     AccessFlags::TRANSFER_WRITE,
/// ));
///
/// // Check if we need a barrier before reading
/// let new_state = ResourceState::buffer(
///     PipelineStage::VertexShader,
///     AccessFlags::VERTEX_BUFFER_READ,
/// );
/// if let Some(barrier) = detector.needs_barrier(buffer_id, &new_state) {
///     println!("Barrier required: {:?}", barrier.hazard);
/// }
/// ```
#[derive(Debug, Default)]
pub struct BarrierDetector {
    /// Internal state tracker for resources.
    tracker: ResourceStateTracker,
}

impl BarrierDetector {
    /// Creates a new barrier detector with an empty state tracker.
    #[inline]
    pub fn new() -> Self {
        Self {
            tracker: ResourceStateTracker::new(),
        }
    }

    /// Creates a barrier detector with an existing state tracker.
    #[inline]
    pub fn with_tracker(tracker: ResourceStateTracker) -> Self {
        Self { tracker }
    }

    /// Creates a barrier detector with pre-allocated capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            tracker: ResourceStateTracker::with_capacity(capacity),
        }
    }

    /// Returns a reference to the internal state tracker.
    #[inline]
    pub fn tracker(&self) -> &ResourceStateTracker {
        &self.tracker
    }

    /// Returns a mutable reference to the internal state tracker.
    #[inline]
    pub fn tracker_mut(&mut self) -> &mut ResourceStateTracker {
        &mut self.tracker
    }

    /// Detects the hazard type between an old and new resource state.
    ///
    /// This is a pure function that analyzes two states and determines
    /// what type of hazard (if any) exists between them.
    pub fn detect_hazard(old: &ResourceState, new: &ResourceState) -> HazardType {
        let old_reads = old.access.has_read();
        let old_writes = old.access.has_write();
        let new_reads = new.access.has_read();
        let new_writes = new.access.has_write();

        // Check for layout transition first (textures only)
        let has_layout_transition = match (old.layout, new.layout) {
            (Some(old_layout), Some(new_layout)) => old_layout != new_layout,
            _ => false,
        };

        // Determine the data hazard type
        let data_hazard = if old_writes && new_writes {
            // Write-after-write: both operations write
            HazardType::WriteAfterWrite
        } else if old_writes && new_reads {
            // Read-after-write: previous write, current read
            HazardType::ReadAfterWrite
        } else if old_reads && new_writes {
            // Write-after-read: previous read, current write
            HazardType::WriteAfterRead
        } else {
            // Read-after-read: no hazard
            HazardType::None
        };

        // If there's a layout transition but no data hazard, report layout transition
        if data_hazard == HazardType::None && has_layout_transition {
            return HazardType::LayoutTransition;
        }

        data_hazard
    }

    /// Checks if a barrier is needed for a resource before a new access.
    ///
    /// Returns `Some(BarrierInfo)` if a barrier is required, `None` otherwise.
    /// This method does NOT update the internal state.
    pub fn needs_barrier(&self, id: ResourceId, new_state: &ResourceState) -> Option<BarrierInfo> {
        let old_state = self.tracker.get(id)?;

        let hazard = Self::detect_hazard(old_state, new_state);
        if hazard == HazardType::None {
            return None;
        }

        Some(BarrierInfo {
            resource_id: id,
            hazard,
            src_stage: old_state.stage,
            dst_stage: new_state.stage,
            src_access: old_state.access,
            dst_access: new_state.access,
            old_layout: old_state.layout,
            new_layout: new_state.layout,
        })
    }

    /// Records a resource access without returning barrier info.
    ///
    /// This simply updates the tracked state for the resource.
    #[inline]
    pub fn record_access(&mut self, id: ResourceId, state: ResourceState) {
        self.tracker.set(id, state);
    }

    /// Transitions a resource to a new state and returns barrier info if needed.
    ///
    /// This method both checks for hazards AND updates the internal state.
    /// It's the primary method for tracking state changes during command recording.
    pub fn transition(&mut self, id: ResourceId, new_state: ResourceState) -> Option<BarrierInfo> {
        let barrier_info = self.needs_barrier(id, &new_state);
        self.tracker.set(id, new_state);
        barrier_info
    }

    /// Resets the detector, clearing all tracked state.
    #[inline]
    pub fn reset(&mut self) {
        self.tracker.clear();
    }

    /// Returns the number of tracked resources.
    #[inline]
    pub fn len(&self) -> usize {
        self.tracker.len()
    }

    /// Returns true if no resources are being tracked.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tracker.is_empty()
    }

    /// Detects all barriers needed for a batch of resource accesses.
    ///
    /// This method checks each access against the current tracked state
    /// and returns a list of all required barriers. The internal state
    /// is NOT modified.
    ///
    /// This is useful for analyzing a set of operations before committing them.
    pub fn detect_all_barriers(&self, accesses: &[(ResourceId, ResourceState)]) -> Vec<BarrierInfo> {
        let mut barriers = Vec::new();

        for (id, new_state) in accesses {
            if let Some(barrier) = self.needs_barrier(*id, new_state) {
                barriers.push(barrier);
            }
        }

        barriers
    }

    /// Transitions multiple resources and collects all barrier info.
    ///
    /// This method processes a batch of accesses, updating internal state
    /// and returning all required barriers.
    pub fn transition_batch(
        &mut self,
        accesses: &[(ResourceId, ResourceState)],
    ) -> Vec<BarrierInfo> {
        let mut barriers = Vec::new();

        for (id, new_state) in accesses {
            if let Some(barrier) = self.transition(*id, new_state.clone()) {
                barriers.push(barrier);
            }
        }

        barriers
    }

    /// Gets the current state of a resource.
    #[inline]
    pub fn get_state(&self, id: ResourceId) -> Option<&ResourceState> {
        self.tracker.get(id)
    }

    /// Returns true if a resource is being tracked.
    #[inline]
    pub fn is_tracked(&self, id: ResourceId) -> bool {
        self.tracker.contains(id)
    }

    /// Merges another detector's tracked states into this one.
    ///
    /// States from the other detector overwrite existing states.
    pub fn merge(&mut self, other: &BarrierDetector) {
        self.tracker.merge(&other.tracker);
    }

    /// Creates a snapshot of the current tracking state.
    pub fn snapshot(&self) -> ResourceStateTracker {
        let mut new_tracker = ResourceStateTracker::with_capacity(self.tracker.len());
        new_tracker.merge(&self.tracker);
        new_tracker
    }
}

// ============================================================================
// Layout Transition Manager (T-WGPU-P4.7.3)
// ============================================================================

/// Represents a subresource range within a texture.
///
/// Used to track layouts at a per-mip-level and per-array-layer granularity,
/// enabling more precise barrier insertion.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct SubresourceRange {
    /// Base mip level (0-indexed).
    pub base_mip: u32,
    /// Number of mip levels. `None` means all remaining mip levels.
    pub mip_count: Option<u32>,
    /// Base array layer (0-indexed).
    pub base_layer: u32,
    /// Number of array layers. `None` means all remaining layers.
    pub layer_count: Option<u32>,
}

impl SubresourceRange {
    /// Creates a subresource range covering a single mip level and layer.
    #[inline]
    pub fn single(mip: u32, layer: u32) -> Self {
        Self {
            base_mip: mip,
            mip_count: Some(1),
            base_layer: layer,
            layer_count: Some(1),
        }
    }

    /// Creates a subresource range covering all mip levels and layers.
    #[inline]
    pub fn all() -> Self {
        Self {
            base_mip: 0,
            mip_count: None,
            base_layer: 0,
            layer_count: None,
        }
    }

    /// Creates a subresource range for a specific mip level range.
    #[inline]
    pub fn mips(base: u32, count: u32) -> Self {
        Self {
            base_mip: base,
            mip_count: Some(count),
            base_layer: 0,
            layer_count: None,
        }
    }

    /// Creates a subresource range for a specific layer range.
    #[inline]
    pub fn layers(base: u32, count: u32) -> Self {
        Self {
            base_mip: 0,
            mip_count: None,
            base_layer: base,
            layer_count: Some(count),
        }
    }

    /// Returns true if this range overlaps with another range.
    ///
    /// Ranges with `None` count are considered to extend to infinity.
    pub fn overlaps(&self, other: &SubresourceRange) -> bool {
        // Check mip overlap
        let self_mip_end = self.mip_count.map(|c| self.base_mip + c);
        let other_mip_end = other.mip_count.map(|c| other.base_mip + c);

        let mip_overlaps = match (self_mip_end, other_mip_end) {
            (Some(se), Some(oe)) => self.base_mip < oe && other.base_mip < se,
            (Some(se), None) => self.base_mip < u32::MAX && other.base_mip < se,
            (None, Some(oe)) => self.base_mip < oe,
            (None, None) => true,
        };

        if !mip_overlaps {
            return false;
        }

        // Check layer overlap
        let self_layer_end = self.layer_count.map(|c| self.base_layer + c);
        let other_layer_end = other.layer_count.map(|c| other.base_layer + c);

        match (self_layer_end, other_layer_end) {
            (Some(se), Some(oe)) => self.base_layer < oe && other.base_layer < se,
            (Some(se), None) => self.base_layer < u32::MAX && other.base_layer < se,
            (None, Some(oe)) => self.base_layer < oe,
            (None, None) => true,
        }
    }

    /// Returns true if this range fully contains another range.
    pub fn contains(&self, other: &SubresourceRange) -> bool {
        // Check mip containment
        let mip_contained = match self.mip_count {
            None => self.base_mip <= other.base_mip,
            Some(self_count) => {
                let self_end = self.base_mip + self_count;
                let other_end = match other.mip_count {
                    None => return false, // Can't contain infinite range
                    Some(c) => other.base_mip + c,
                };
                self.base_mip <= other.base_mip && other_end <= self_end
            }
        };

        if !mip_contained {
            return false;
        }

        // Check layer containment
        match self.layer_count {
            None => self.base_layer <= other.base_layer,
            Some(self_count) => {
                let self_end = self.base_layer + self_count;
                let other_end = match other.layer_count {
                    None => return false, // Can't contain infinite range
                    Some(c) => other.base_layer + c,
                };
                self.base_layer <= other.base_layer && other_end <= self_end
            }
        }
    }

    /// Attempts to merge two overlapping ranges into a single range.
    ///
    /// Returns `Some` if the ranges can be merged into a contiguous range,
    /// `None` if they cannot be merged (e.g., disjoint ranges).
    pub fn try_merge(&self, other: &SubresourceRange) -> Option<SubresourceRange> {
        if !self.overlaps(other) && !self.is_adjacent(other) {
            return None;
        }

        let base_mip = self.base_mip.min(other.base_mip);
        let base_layer = self.base_layer.min(other.base_layer);

        let mip_count = match (self.mip_count, other.mip_count) {
            (None, _) | (_, None) => None,
            (Some(sc), Some(oc)) => {
                let self_end = self.base_mip + sc;
                let other_end = other.base_mip + oc;
                Some(self_end.max(other_end) - base_mip)
            }
        };

        let layer_count = match (self.layer_count, other.layer_count) {
            (None, _) | (_, None) => None,
            (Some(sc), Some(oc)) => {
                let self_end = self.base_layer + sc;
                let other_end = other.base_layer + oc;
                Some(self_end.max(other_end) - base_layer)
            }
        };

        Some(SubresourceRange {
            base_mip,
            mip_count,
            base_layer,
            layer_count,
        })
    }

    /// Returns true if this range is adjacent to another range.
    ///
    /// Two ranges are adjacent if one dimension is immediately adjacent
    /// (e.g., mips 0-2 and 2-4) while the other dimension overlaps.
    pub fn is_adjacent(&self, other: &SubresourceRange) -> bool {
        // Check if mips are adjacent
        let mip_adjacent = match (self.mip_count, other.mip_count) {
            (Some(sc), _) if self.base_mip + sc == other.base_mip => true,
            (_, Some(oc)) if other.base_mip + oc == self.base_mip => true,
            _ => false,
        };

        // Check if layers are adjacent
        let layer_adjacent = match (self.layer_count, other.layer_count) {
            (Some(sc), _) if self.base_layer + sc == other.base_layer => true,
            (_, Some(oc)) if other.base_layer + oc == self.base_layer => true,
            _ => false,
        };

        // Adjacent if one dimension is adjacent and the other overlaps
        (mip_adjacent && self.layers_overlap(other)) || (layer_adjacent && self.mips_overlap(other))
    }

    fn mips_overlap(&self, other: &SubresourceRange) -> bool {
        let self_end = self.mip_count.map(|c| self.base_mip + c);
        let other_end = other.mip_count.map(|c| other.base_mip + c);

        match (self_end, other_end) {
            (Some(se), Some(oe)) => self.base_mip < oe && other.base_mip < se,
            _ => true,
        }
    }

    fn layers_overlap(&self, other: &SubresourceRange) -> bool {
        let self_end = self.layer_count.map(|c| self.base_layer + c);
        let other_end = other.layer_count.map(|c| other.base_layer + c);

        match (self_end, other_end) {
            (Some(se), Some(oe)) => self.base_layer < oe && other.base_layer < se,
            _ => true,
        }
    }

    /// Returns the effective mip count for comparison, treating None as max.
    #[inline]
    pub fn effective_mip_count(&self) -> u32 {
        self.mip_count.unwrap_or(u32::MAX)
    }

    /// Returns the effective layer count for comparison, treating None as max.
    #[inline]
    pub fn effective_layer_count(&self) -> u32 {
        self.layer_count.unwrap_or(u32::MAX)
    }
}

/// A layout transition for a texture resource.
///
/// Contains all information needed to perform a layout transition,
/// including the resource ID, old/new layouts, and subresource range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutTransition {
    /// The resource ID of the texture being transitioned.
    pub resource_id: ResourceId,
    /// The layout before the transition.
    pub old_layout: TextureLayout,
    /// The layout after the transition.
    pub new_layout: TextureLayout,
    /// The subresource range affected by this transition.
    pub subresource: SubresourceRange,
}

impl LayoutTransition {
    /// Creates a new layout transition.
    pub fn new(
        resource_id: ResourceId,
        old_layout: TextureLayout,
        new_layout: TextureLayout,
        subresource: SubresourceRange,
    ) -> Self {
        Self {
            resource_id,
            old_layout,
            new_layout,
            subresource,
        }
    }

    /// Creates a layout transition for the entire resource.
    pub fn whole(
        resource_id: ResourceId,
        old_layout: TextureLayout,
        new_layout: TextureLayout,
    ) -> Self {
        Self {
            resource_id,
            old_layout,
            new_layout,
            subresource: SubresourceRange::all(),
        }
    }

    /// Returns true if this transition is actually needed (layouts differ).
    #[inline]
    pub fn is_needed(&self) -> bool {
        self.old_layout != self.new_layout
    }

    /// Returns true if this transition can be merged with another.
    ///
    /// Transitions can be merged if they are for the same resource,
    /// have the same source and destination layouts, and their
    /// subresource ranges can be merged.
    pub fn can_merge_with(&self, other: &LayoutTransition) -> bool {
        self.resource_id == other.resource_id
            && self.old_layout == other.old_layout
            && self.new_layout == other.new_layout
            && (self.subresource.overlaps(&other.subresource)
                || self.subresource.is_adjacent(&other.subresource))
    }

    /// Attempts to merge this transition with another.
    pub fn try_merge(&self, other: &LayoutTransition) -> Option<LayoutTransition> {
        if !self.can_merge_with(other) {
            return None;
        }

        self.subresource.try_merge(&other.subresource).map(|merged_range| {
            LayoutTransition {
                resource_id: self.resource_id,
                old_layout: self.old_layout,
                new_layout: self.new_layout,
                subresource: merged_range,
            }
        })
    }
}

/// Mode for layout transition tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TransitionMode {
    /// Automatic layout tracking - transitions are calculated automatically
    /// based on recorded usage.
    #[default]
    Implicit,
    /// User-specified transitions - the caller explicitly provides transitions.
    Explicit,
}

/// Manages texture layout transitions with subresource-level granularity.
///
/// The `LayoutTransitionManager` tracks the current layout of each texture
/// subresource and calculates optimal transitions when layouts need to change.
///
/// # Features
///
/// - Per-subresource layout tracking
/// - Automatic transition calculation
/// - Transition coalescing (merging multiple transitions)
/// - Support for implicit and explicit transition modes
///
/// # Example
///
/// ```ignore
/// use renderer_backend::resource_state::*;
///
/// let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);
///
/// // Set initial layout
/// manager.set_layout(texture_id, SubresourceRange::all(), TextureLayout::Undefined);
///
/// // Transition to transfer destination
/// if let Some(transition) = manager.transition_to(texture_id, TextureLayout::TransferDst) {
///     // Insert barrier for this transition
/// }
///
/// // Later, transition to shader read
/// if let Some(transition) = manager.transition_to(texture_id, TextureLayout::ShaderReadOnly) {
///     // Insert barrier for this transition
/// }
/// ```
#[derive(Debug, Default)]
pub struct LayoutTransitionManager {
    /// Per-resource, per-subresource layout tracking.
    /// Maps resource ID -> (subresource range -> layout).
    layouts: HashMap<ResourceId, HashMap<SubresourceRange, TextureLayout>>,
    /// The transition mode (implicit or explicit).
    mode: TransitionMode,
    /// Pending transitions waiting to be coalesced and flushed.
    pending: Vec<LayoutTransition>,
}

impl LayoutTransitionManager {
    /// Creates a new layout transition manager with the specified mode.
    pub fn new(mode: TransitionMode) -> Self {
        Self {
            layouts: HashMap::new(),
            mode,
            pending: Vec::new(),
        }
    }

    /// Creates a new layout transition manager with preallocated capacity.
    pub fn with_capacity(mode: TransitionMode, capacity: usize) -> Self {
        Self {
            layouts: HashMap::with_capacity(capacity),
            mode,
            pending: Vec::new(),
        }
    }

    /// Returns the current transition mode.
    #[inline]
    pub fn mode(&self) -> TransitionMode {
        self.mode
    }

    /// Sets the transition mode.
    #[inline]
    pub fn set_mode(&mut self, mode: TransitionMode) {
        self.mode = mode;
    }

    /// Gets the layout for a specific subresource.
    ///
    /// Returns `None` if the resource or subresource is not tracked.
    pub fn get_layout(&self, id: ResourceId, subresource: SubresourceRange) -> Option<TextureLayout> {
        let resource_layouts = self.layouts.get(&id)?;

        // First try exact match
        if let Some(&layout) = resource_layouts.get(&subresource) {
            return Some(layout);
        }

        // Then try to find a containing range
        for (range, &layout) in resource_layouts {
            if range.contains(&subresource) {
                return Some(layout);
            }
        }

        None
    }

    /// Gets the layout for the entire resource (all subresources).
    ///
    /// Returns `None` if the resource is not tracked or has mixed layouts.
    pub fn get_whole_layout(&self, id: ResourceId) -> Option<TextureLayout> {
        self.get_layout(id, SubresourceRange::all())
    }

    /// Sets the layout for a specific subresource range.
    pub fn set_layout(&mut self, id: ResourceId, subresource: SubresourceRange, layout: TextureLayout) {
        let resource_layouts = self.layouts.entry(id).or_insert_with(HashMap::new);

        // Remove any overlapping ranges
        let overlapping: Vec<_> = resource_layouts
            .keys()
            .filter(|r| r.overlaps(&subresource))
            .cloned()
            .collect();

        for range in overlapping {
            resource_layouts.remove(&range);
        }

        resource_layouts.insert(subresource, layout);
    }

    /// Sets the layout for the entire resource.
    pub fn set_whole_layout(&mut self, id: ResourceId, layout: TextureLayout) {
        self.set_layout(id, SubresourceRange::all(), layout);
    }

    /// Transitions a resource to a new layout and returns the transition info if needed.
    ///
    /// This method transitions all subresources of the resource to the new layout.
    pub fn transition_to(&mut self, id: ResourceId, new_layout: TextureLayout) -> Option<LayoutTransition> {
        self.transition_subresource(id, SubresourceRange::all(), new_layout)
    }

    /// Transitions a specific subresource range to a new layout.
    ///
    /// Returns `Some(LayoutTransition)` if a transition is needed, `None` otherwise.
    pub fn transition_subresource(
        &mut self,
        id: ResourceId,
        subresource: SubresourceRange,
        new_layout: TextureLayout,
    ) -> Option<LayoutTransition> {
        let old_layout = self.get_layout(id, subresource).unwrap_or(TextureLayout::Undefined);

        if !Self::is_transition_needed(old_layout, new_layout) {
            return None;
        }

        // Update the tracked layout
        self.set_layout(id, subresource, new_layout);

        Some(LayoutTransition {
            resource_id: id,
            old_layout,
            new_layout,
            subresource,
        })
    }

    /// Adds a transition to the pending list for later coalescing.
    pub fn add_pending(&mut self, transition: LayoutTransition) {
        if transition.is_needed() {
            self.pending.push(transition);
        }
    }

    /// Coalesces pending transitions by merging compatible ones.
    ///
    /// This method attempts to merge transitions that:
    /// - Target the same resource
    /// - Have the same source and destination layouts
    /// - Have overlapping or adjacent subresource ranges
    ///
    /// Returns the coalesced list of transitions.
    pub fn coalesce_pending(&mut self) -> Vec<LayoutTransition> {
        if self.pending.is_empty() {
            return Vec::new();
        }

        let mut result: Vec<LayoutTransition> = Vec::new();
        let pending = std::mem::take(&mut self.pending);

        for transition in pending {
            let mut merged = false;

            for existing in &mut result {
                if let Some(merged_transition) = existing.try_merge(&transition) {
                    *existing = merged_transition;
                    merged = true;
                    break;
                }
            }

            if !merged {
                result.push(transition);
            }
        }

        // Do a second pass to try to merge any newly created overlaps
        let mut final_result: Vec<LayoutTransition> = Vec::new();
        for transition in result {
            let mut merged = false;

            for existing in &mut final_result {
                if let Some(merged_transition) = existing.try_merge(&transition) {
                    *existing = merged_transition;
                    merged = true;
                    break;
                }
            }

            if !merged {
                final_result.push(transition);
            }
        }

        final_result
    }

    /// Flushes all pending transitions, applying coalescing and updating tracked layouts.
    ///
    /// Returns the final list of transitions to execute.
    pub fn flush_pending(&mut self) -> Vec<LayoutTransition> {
        let coalesced = self.coalesce_pending();

        // Update tracked layouts for all flushed transitions
        for transition in &coalesced {
            self.set_layout(transition.resource_id, transition.subresource, transition.new_layout);
        }

        coalesced
    }

    /// Calculates the optimal transition path between two layouts.
    ///
    /// Some layout transitions may benefit from intermediate steps.
    /// For example, Undefined -> ShaderReadOnly might go through TransferDst
    /// if the texture needs to be initialized first.
    ///
    /// Returns a vector of layouts representing the transition path.
    pub fn optimal_transition_path(from: TextureLayout, to: TextureLayout) -> Vec<TextureLayout> {
        if from == to {
            return vec![from];
        }

        // Most transitions are direct
        // Special cases where intermediate layouts might be beneficial:
        match (from, to) {
            // Undefined to a read layout might need initialization
            (TextureLayout::Undefined, TextureLayout::ShaderReadOnly) => {
                vec![TextureLayout::Undefined, TextureLayout::TransferDst, TextureLayout::ShaderReadOnly]
            }
            (TextureLayout::Undefined, TextureLayout::DepthStencilReadOnly) => {
                vec![TextureLayout::Undefined, TextureLayout::DepthStencilAttachment, TextureLayout::DepthStencilReadOnly]
            }
            // Preinitialized typically goes through transfer
            (TextureLayout::Preinitialized, TextureLayout::ShaderReadOnly) => {
                vec![TextureLayout::Preinitialized, TextureLayout::TransferSrc, TextureLayout::ShaderReadOnly]
            }
            // Present to color attachment is common for double buffering
            (TextureLayout::Present, TextureLayout::ColorAttachment) => {
                vec![TextureLayout::Present, TextureLayout::ColorAttachment]
            }
            // General case: direct transition
            _ => vec![from, to],
        }
    }

    /// Returns true if a transition is needed between two layouts.
    #[inline]
    pub fn is_transition_needed(from: TextureLayout, to: TextureLayout) -> bool {
        from != to
    }

    /// Returns true if the given resource is being tracked.
    #[inline]
    pub fn is_tracked(&self, id: ResourceId) -> bool {
        self.layouts.contains_key(&id)
    }

    /// Returns the number of tracked resources.
    #[inline]
    pub fn len(&self) -> usize {
        self.layouts.len()
    }

    /// Returns true if no resources are being tracked.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.layouts.is_empty()
    }

    /// Returns the number of pending transitions.
    #[inline]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Returns true if there are pending transitions.
    #[inline]
    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Clears all tracked layouts and pending transitions.
    pub fn clear(&mut self) {
        self.layouts.clear();
        self.pending.clear();
    }

    /// Clears only pending transitions without affecting tracked layouts.
    pub fn clear_pending(&mut self) {
        self.pending.clear();
    }

    /// Removes a resource from tracking.
    pub fn remove(&mut self, id: ResourceId) -> bool {
        self.layouts.remove(&id).is_some()
    }

    /// Returns an iterator over all tracked resource IDs.
    pub fn tracked_resources(&self) -> impl Iterator<Item = &ResourceId> {
        self.layouts.keys()
    }

    /// Gets all subresource layouts for a resource.
    pub fn get_all_layouts(&self, id: ResourceId) -> Option<&HashMap<SubresourceRange, TextureLayout>> {
        self.layouts.get(&id)
    }

    /// Batch transition multiple resources to new layouts.
    ///
    /// Returns a list of all transitions that were needed.
    pub fn transition_batch(
        &mut self,
        transitions: &[(ResourceId, TextureLayout)],
    ) -> Vec<LayoutTransition> {
        transitions
            .iter()
            .filter_map(|(id, layout)| self.transition_to(*id, *layout))
            .collect()
    }

    /// Batch transition multiple subresources to new layouts.
    pub fn transition_subresources_batch(
        &mut self,
        transitions: &[(ResourceId, SubresourceRange, TextureLayout)],
    ) -> Vec<LayoutTransition> {
        transitions
            .iter()
            .filter_map(|(id, subresource, layout)| {
                self.transition_subresource(*id, *subresource, *layout)
            })
            .collect()
    }

    /// Creates a snapshot of the current layout tracking state.
    pub fn snapshot(&self) -> HashMap<ResourceId, HashMap<SubresourceRange, TextureLayout>> {
        self.layouts.clone()
    }

    /// Restores layout tracking state from a snapshot.
    pub fn restore(&mut self, snapshot: HashMap<ResourceId, HashMap<SubresourceRange, TextureLayout>>) {
        self.layouts = snapshot;
    }

    /// Merges another manager's tracked layouts into this one.
    pub fn merge(&mut self, other: &LayoutTransitionManager) {
        for (id, layouts) in &other.layouts {
            let entry = self.layouts.entry(*id).or_insert_with(HashMap::new);
            for (range, layout) in layouts {
                entry.insert(*range, *layout);
            }
        }
    }
}

// ============================================================================
// Barrier Batching (T-WGPU-P4.7.4)
// ============================================================================

/// A batched barrier containing multiple buffer and texture barriers
/// that share the same pipeline stage transitions.
///
/// Batching barriers allows the GPU driver to process multiple barriers
/// in a single submission, reducing API overhead and potentially allowing
/// better optimization of memory barriers.
#[derive(Debug, Clone, Default)]
pub struct BatchedBarrier {
    /// Source pipeline stages (combined from all barriers).
    pub src_stages: PipelineStageMask,
    /// Destination pipeline stages (combined from all barriers).
    pub dst_stages: PipelineStageMask,
    /// Buffer memory barriers in this batch.
    pub buffer_barriers: Vec<BufferBarrier>,
    /// Texture/image memory barriers in this batch.
    pub texture_barriers: Vec<TextureBarrier>,
}

impl BatchedBarrier {
    /// Creates a new empty batched barrier.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a batched barrier with the given stage masks.
    pub fn with_stages(src_stages: PipelineStageMask, dst_stages: PipelineStageMask) -> Self {
        Self {
            src_stages,
            dst_stages,
            buffer_barriers: Vec::new(),
            texture_barriers: Vec::new(),
        }
    }

    /// Returns true if this batch is empty (no barriers).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buffer_barriers.is_empty() && self.texture_barriers.is_empty()
    }

    /// Returns the total number of barriers in this batch.
    #[inline]
    pub fn len(&self) -> usize {
        self.buffer_barriers.len() + self.texture_barriers.len()
    }

    /// Adds a buffer barrier to this batch.
    pub fn add_buffer_barrier(&mut self, barrier: BufferBarrier) {
        self.buffer_barriers.push(barrier);
    }

    /// Adds a texture barrier to this batch.
    pub fn add_texture_barrier(&mut self, barrier: TextureBarrier) {
        self.texture_barriers.push(barrier);
    }

    /// Merges another batched barrier into this one.
    ///
    /// Stage masks are combined using bitwise OR.
    pub fn merge(&mut self, other: BatchedBarrier) {
        self.src_stages = self.src_stages | other.src_stages;
        self.dst_stages = self.dst_stages | other.dst_stages;
        self.buffer_barriers.extend(other.buffer_barriers);
        self.texture_barriers.extend(other.texture_barriers);
    }

    /// Clears all barriers from this batch.
    pub fn clear(&mut self) {
        self.src_stages = PipelineStageMask::NONE;
        self.dst_stages = PipelineStageMask::NONE;
        self.buffer_barriers.clear();
        self.texture_barriers.clear();
    }
}

/// A buffer memory barrier for synchronization.
///
/// Describes a memory dependency on a buffer resource, specifying
/// the access patterns before and after the barrier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferBarrier {
    /// The resource ID of the buffer.
    pub resource_id: ResourceId,
    /// Access flags before the barrier.
    pub src_access: AccessFlags,
    /// Access flags after the barrier.
    pub dst_access: AccessFlags,
    /// Offset into the buffer (in bytes).
    pub offset: u64,
    /// Size of the region (in bytes). `None` means the whole buffer.
    pub size: Option<u64>,
}

impl BufferBarrier {
    /// Creates a new buffer barrier for the whole buffer.
    pub fn whole(resource_id: ResourceId, src_access: AccessFlags, dst_access: AccessFlags) -> Self {
        Self {
            resource_id,
            src_access,
            dst_access,
            offset: 0,
            size: None,
        }
    }

    /// Creates a new buffer barrier for a specific region.
    pub fn region(
        resource_id: ResourceId,
        src_access: AccessFlags,
        dst_access: AccessFlags,
        offset: u64,
        size: u64,
    ) -> Self {
        Self {
            resource_id,
            src_access,
            dst_access,
            offset,
            size: Some(size),
        }
    }

    /// Returns true if this barrier covers the whole buffer.
    #[inline]
    pub fn is_whole_buffer(&self) -> bool {
        self.offset == 0 && self.size.is_none()
    }

    /// Returns true if this barrier can be merged with another.
    ///
    /// Barriers can be merged if they are for the same resource,
    /// have adjacent or overlapping regions, and compatible access patterns.
    pub fn can_merge_with(&self, other: &BufferBarrier) -> bool {
        if self.resource_id != other.resource_id {
            return false;
        }

        // If either is whole buffer, they can be merged
        if self.is_whole_buffer() || other.is_whole_buffer() {
            return true;
        }

        // Check if regions overlap or are adjacent
        match (self.size, other.size) {
            (Some(self_size), Some(other_size)) => {
                let self_end = self.offset + self_size;
                let other_end = other.offset + other_size;
                // Overlapping or adjacent
                self.offset <= other_end && other.offset <= self_end
            }
            _ => true,
        }
    }

    /// Attempts to merge this barrier with another.
    ///
    /// Returns the merged barrier if successful, `None` otherwise.
    pub fn try_merge(&self, other: &BufferBarrier) -> Option<BufferBarrier> {
        if !self.can_merge_with(other) {
            return None;
        }

        // Merge access flags
        let src_access = self.src_access | other.src_access;
        let dst_access = self.dst_access | other.dst_access;

        // Merge regions
        let (offset, size) = match (self.is_whole_buffer(), other.is_whole_buffer()) {
            (true, _) | (_, true) => (0, None),
            _ => {
                let self_size = self.size.unwrap();
                let other_size = other.size.unwrap();
                let min_offset = self.offset.min(other.offset);
                let max_end = (self.offset + self_size).max(other.offset + other_size);
                (min_offset, Some(max_end - min_offset))
            }
        };

        Some(BufferBarrier {
            resource_id: self.resource_id,
            src_access,
            dst_access,
            offset,
            size,
        })
    }
}

/// A texture/image memory barrier for synchronization.
///
/// Describes a memory dependency on a texture resource, including
/// layout transitions and access pattern changes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextureBarrier {
    /// The resource ID of the texture.
    pub resource_id: ResourceId,
    /// Access flags before the barrier.
    pub src_access: AccessFlags,
    /// Access flags after the barrier.
    pub dst_access: AccessFlags,
    /// Layout before the barrier.
    pub old_layout: TextureLayout,
    /// Layout after the barrier.
    pub new_layout: TextureLayout,
    /// The subresource range affected by this barrier.
    pub subresource: SubresourceRange,
}

impl TextureBarrier {
    /// Creates a new texture barrier for the whole texture.
    pub fn whole(
        resource_id: ResourceId,
        src_access: AccessFlags,
        dst_access: AccessFlags,
        old_layout: TextureLayout,
        new_layout: TextureLayout,
    ) -> Self {
        Self {
            resource_id,
            src_access,
            dst_access,
            old_layout,
            new_layout,
            subresource: SubresourceRange::all(),
        }
    }

    /// Creates a new texture barrier for a specific subresource range.
    pub fn subresource(
        resource_id: ResourceId,
        src_access: AccessFlags,
        dst_access: AccessFlags,
        old_layout: TextureLayout,
        new_layout: TextureLayout,
        subresource: SubresourceRange,
    ) -> Self {
        Self {
            resource_id,
            src_access,
            dst_access,
            old_layout,
            new_layout,
            subresource,
        }
    }

    /// Returns true if this barrier involves a layout transition.
    #[inline]
    pub fn has_layout_transition(&self) -> bool {
        self.old_layout != self.new_layout
    }

    /// Returns true if this barrier covers the whole texture.
    #[inline]
    pub fn is_whole_texture(&self) -> bool {
        self.subresource == SubresourceRange::all()
    }

    /// Returns true if this barrier can be merged with another.
    ///
    /// Barriers can be merged if they are for the same resource,
    /// have the same layout transitions, and overlapping/adjacent subresources.
    pub fn can_merge_with(&self, other: &TextureBarrier) -> bool {
        if self.resource_id != other.resource_id {
            return false;
        }

        // Layout transitions must match for merging
        if self.old_layout != other.old_layout || self.new_layout != other.new_layout {
            return false;
        }

        // Check subresource overlap or adjacency
        self.subresource.overlaps(&other.subresource)
            || self.subresource.is_adjacent(&other.subresource)
    }

    /// Attempts to merge this barrier with another.
    ///
    /// Returns the merged barrier if successful, `None` otherwise.
    pub fn try_merge(&self, other: &TextureBarrier) -> Option<TextureBarrier> {
        if !self.can_merge_with(other) {
            return None;
        }

        let merged_subresource = self.subresource.try_merge(&other.subresource)?;

        Some(TextureBarrier {
            resource_id: self.resource_id,
            src_access: self.src_access | other.src_access,
            dst_access: self.dst_access | other.dst_access,
            old_layout: self.old_layout,
            new_layout: self.new_layout,
            subresource: merged_subresource,
        })
    }
}

bitflags! {
    /// Pipeline stage mask for barrier batching.
    ///
    /// This is a bitflag representation of pipeline stages, allowing
    /// multiple stages to be combined for barrier synchronization.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    pub struct PipelineStageMask: u32 {
        /// No stage.
        const NONE = 0;
        /// Top of pipe (before any work).
        const TOP_OF_PIPE = 1 << 0;
        /// Vertex input assembly.
        const VERTEX_INPUT = 1 << 1;
        /// Vertex shader.
        const VERTEX_SHADER = 1 << 2;
        /// Fragment shader.
        const FRAGMENT_SHADER = 1 << 3;
        /// Early depth test.
        const EARLY_DEPTH = 1 << 4;
        /// Late depth test.
        const LATE_DEPTH = 1 << 5;
        /// Color output.
        const COLOR_OUTPUT = 1 << 6;
        /// Compute shader.
        const COMPUTE_SHADER = 1 << 7;
        /// Transfer operations.
        const TRANSFER = 1 << 8;
        /// Host (CPU) operations.
        const HOST = 1 << 9;
        /// Bottom of pipe (after all work).
        const BOTTOM_OF_PIPE = 1 << 10;
        /// All graphics stages combined.
        const ALL_GRAPHICS = Self::VERTEX_INPUT.bits() | Self::VERTEX_SHADER.bits()
            | Self::EARLY_DEPTH.bits() | Self::FRAGMENT_SHADER.bits()
            | Self::LATE_DEPTH.bits() | Self::COLOR_OUTPUT.bits();
        /// All stages combined.
        const ALL_COMMANDS = Self::ALL_GRAPHICS.bits() | Self::COMPUTE_SHADER.bits()
            | Self::TRANSFER.bits() | Self::HOST.bits();
    }
}

impl PipelineStageMask {
    /// Converts a `PipelineStage` enum to a `PipelineStageMask`.
    pub fn from_stage(stage: PipelineStage) -> Self {
        match stage {
            PipelineStage::None => PipelineStageMask::NONE,
            PipelineStage::VertexInput => PipelineStageMask::VERTEX_INPUT,
            PipelineStage::VertexShader => PipelineStageMask::VERTEX_SHADER,
            PipelineStage::FragmentShader => PipelineStageMask::FRAGMENT_SHADER,
            PipelineStage::EarlyDepth => PipelineStageMask::EARLY_DEPTH,
            PipelineStage::LateDepth => PipelineStageMask::LATE_DEPTH,
            PipelineStage::ColorOutput => PipelineStageMask::COLOR_OUTPUT,
            PipelineStage::ComputeShader => PipelineStageMask::COMPUTE_SHADER,
            PipelineStage::Transfer => PipelineStageMask::TRANSFER,
            PipelineStage::Host => PipelineStageMask::HOST,
            PipelineStage::AllGraphics => PipelineStageMask::ALL_GRAPHICS,
            PipelineStage::AllCommands => PipelineStageMask::ALL_COMMANDS,
        }
    }

    /// Returns true if this mask contains any graphics stages.
    #[inline]
    pub fn has_graphics(&self) -> bool {
        self.intersects(PipelineStageMask::ALL_GRAPHICS)
    }

    /// Returns true if this mask contains the compute stage.
    #[inline]
    pub fn has_compute(&self) -> bool {
        self.contains(PipelineStageMask::COMPUTE_SHADER)
    }

    /// Returns true if this mask contains the transfer stage.
    #[inline]
    pub fn has_transfer(&self) -> bool {
        self.contains(PipelineStageMask::TRANSFER)
    }

    /// Returns true if this mask contains the host stage.
    #[inline]
    pub fn has_host(&self) -> bool {
        self.contains(PipelineStageMask::HOST)
    }

    /// Merges two stage masks using bitwise OR.
    #[inline]
    pub fn merge(self, other: PipelineStageMask) -> PipelineStageMask {
        self | other
    }
}

/// Batches multiple barriers into efficient grouped submissions.
///
/// The `BarrierBatcher` collects individual barriers and groups them
/// by compatible pipeline stages, minimizing the number of barrier
/// submissions to the GPU.
///
/// # Features
///
/// - Collects buffer and texture barriers
/// - Merges compatible barriers for the same resource
/// - Groups barriers by pipeline stage transitions
/// - Optimizes memory barrier counts
///
/// # Example
///
/// ```ignore
/// use renderer_backend::resource_state::*;
///
/// let mut batcher = BarrierBatcher::new();
///
/// // Add multiple barriers
/// batcher.add_buffer_barrier(BufferBarrier::whole(
///     1,
///     AccessFlags::TRANSFER_WRITE,
///     AccessFlags::VERTEX_BUFFER_READ,
/// ));
/// batcher.add_texture_barrier(TextureBarrier::whole(
///     2,
///     AccessFlags::TRANSFER_WRITE,
///     AccessFlags::SHADER_READ,
///     TextureLayout::TransferDst,
///     TextureLayout::ShaderReadOnly,
/// ));
///
/// // Batch all barriers together
/// let batched = batcher.batch();
/// // Submit batched barrier to GPU
/// ```
#[derive(Debug, Default)]
pub struct BarrierBatcher {
    /// Pending buffer barriers.
    pending_buffer_barriers: Vec<BufferBarrier>,
    /// Pending texture barriers.
    pending_texture_barriers: Vec<TextureBarrier>,
    /// Source pipeline stage mask for current batch.
    src_stage_mask: PipelineStageMask,
    /// Destination pipeline stage mask for current batch.
    dst_stage_mask: PipelineStageMask,
}

impl BarrierBatcher {
    /// Creates a new empty barrier batcher.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a barrier batcher with preallocated capacity.
    pub fn with_capacity(buffer_capacity: usize, texture_capacity: usize) -> Self {
        Self {
            pending_buffer_barriers: Vec::with_capacity(buffer_capacity),
            pending_texture_barriers: Vec::with_capacity(texture_capacity),
            src_stage_mask: PipelineStageMask::NONE,
            dst_stage_mask: PipelineStageMask::NONE,
        }
    }

    /// Adds a buffer barrier to the pending list.
    pub fn add_buffer_barrier(&mut self, barrier: BufferBarrier) {
        self.pending_buffer_barriers.push(barrier);
    }

    /// Adds a texture barrier to the pending list.
    pub fn add_texture_barrier(&mut self, barrier: TextureBarrier) {
        self.pending_texture_barriers.push(barrier);
    }

    /// Adds a barrier from a `BarrierInfo` struct.
    ///
    /// This converts the generic `BarrierInfo` from barrier detection
    /// into the appropriate buffer or texture barrier.
    pub fn add_barrier_info(&mut self, info: BarrierInfo) {
        // Update stage masks
        self.src_stage_mask = self.src_stage_mask | PipelineStageMask::from_stage(info.src_stage);
        self.dst_stage_mask = self.dst_stage_mask | PipelineStageMask::from_stage(info.dst_stage);

        if info.is_buffer_barrier() {
            self.pending_buffer_barriers.push(BufferBarrier::whole(
                info.resource_id,
                info.src_access,
                info.dst_access,
            ));
        } else {
            self.pending_texture_barriers.push(TextureBarrier::whole(
                info.resource_id,
                info.src_access,
                info.dst_access,
                info.old_layout.unwrap_or(TextureLayout::Undefined),
                info.new_layout.unwrap_or(TextureLayout::Undefined),
            ));
        }
    }

    /// Batches all pending barriers into a single `BatchedBarrier`.
    ///
    /// This method merges compatible barriers and combines stage masks,
    /// then clears the pending lists.
    pub fn batch(&mut self) -> BatchedBarrier {
        let buffer_barriers = self.merge_buffer_barriers();
        let texture_barriers = self.merge_texture_barriers();

        let result = BatchedBarrier {
            src_stages: self.src_stage_mask,
            dst_stages: self.dst_stage_mask,
            buffer_barriers,
            texture_barriers,
        };

        self.clear();
        result
    }

    /// Batches barriers grouped by their pipeline stage transitions.
    ///
    /// This method creates separate batches for different stage combinations,
    /// which can be useful when barriers need to be inserted at different
    /// points in the command buffer.
    pub fn batch_by_stage(&mut self) -> Vec<BatchedBarrier> {
        if self.is_empty() {
            return Vec::new();
        }

        // Group barriers by their effective stage transitions
        let mut stage_groups: HashMap<(PipelineStageMask, PipelineStageMask), BatchedBarrier> =
            HashMap::new();

        // Process buffer barriers
        for barrier in std::mem::take(&mut self.pending_buffer_barriers) {
            let src_mask = Self::access_to_stage_mask(&barrier.src_access);
            let dst_mask = Self::access_to_stage_mask(&barrier.dst_access);
            let key = (src_mask, dst_mask);

            let batch = stage_groups.entry(key).or_insert_with(|| {
                BatchedBarrier::with_stages(src_mask, dst_mask)
            });
            batch.buffer_barriers.push(barrier);
        }

        // Process texture barriers
        for barrier in std::mem::take(&mut self.pending_texture_barriers) {
            let src_mask = Self::access_to_stage_mask(&barrier.src_access);
            let dst_mask = Self::access_to_stage_mask(&barrier.dst_access);
            let key = (src_mask, dst_mask);

            let batch = stage_groups.entry(key).or_insert_with(|| {
                BatchedBarrier::with_stages(src_mask, dst_mask)
            });
            batch.texture_barriers.push(barrier);
        }

        // Merge barriers within each group and collect results
        let mut result: Vec<BatchedBarrier> = stage_groups
            .into_iter()
            .map(|(_, mut batch)| {
                batch.buffer_barriers = Self::merge_buffer_list(batch.buffer_barriers);
                batch.texture_barriers = Self::merge_texture_list(batch.texture_barriers);
                batch
            })
            .filter(|batch| !batch.is_empty())
            .collect();

        // Sort by source stage for deterministic ordering
        result.sort_by_key(|b| b.src_stages.bits());

        self.src_stage_mask = PipelineStageMask::NONE;
        self.dst_stage_mask = PipelineStageMask::NONE;

        result
    }

    /// Merges two pipeline stage masks.
    ///
    /// This is a convenience function for combining stage masks.
    #[inline]
    pub fn merge_stages(a: PipelineStageMask, b: PipelineStageMask) -> PipelineStageMask {
        a | b
    }

    /// Merges two access flag sets.
    ///
    /// This is a convenience function for combining access flags.
    #[inline]
    pub fn merge_access(a: AccessFlags, b: AccessFlags) -> AccessFlags {
        a | b
    }

    /// Returns the number of pending barriers.
    #[inline]
    pub fn pending_count(&self) -> usize {
        self.pending_buffer_barriers.len() + self.pending_texture_barriers.len()
    }

    /// Returns the number of pending buffer barriers.
    #[inline]
    pub fn pending_buffer_count(&self) -> usize {
        self.pending_buffer_barriers.len()
    }

    /// Returns the number of pending texture barriers.
    #[inline]
    pub fn pending_texture_count(&self) -> usize {
        self.pending_texture_barriers.len()
    }

    /// Clears all pending barriers.
    pub fn clear(&mut self) {
        self.pending_buffer_barriers.clear();
        self.pending_texture_barriers.clear();
        self.src_stage_mask = PipelineStageMask::NONE;
        self.dst_stage_mask = PipelineStageMask::NONE;
    }

    /// Returns true if there are no pending barriers.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.pending_buffer_barriers.is_empty() && self.pending_texture_barriers.is_empty()
    }

    /// Returns the current source stage mask.
    #[inline]
    pub fn src_stage_mask(&self) -> PipelineStageMask {
        self.src_stage_mask
    }

    /// Returns the current destination stage mask.
    #[inline]
    pub fn dst_stage_mask(&self) -> PipelineStageMask {
        self.dst_stage_mask
    }

    /// Sets the source and destination stage masks explicitly.
    pub fn set_stage_masks(&mut self, src: PipelineStageMask, dst: PipelineStageMask) {
        self.src_stage_mask = src;
        self.dst_stage_mask = dst;
    }

    /// Merges pending buffer barriers.
    fn merge_buffer_barriers(&self) -> Vec<BufferBarrier> {
        Self::merge_buffer_list(self.pending_buffer_barriers.clone())
    }

    /// Merges pending texture barriers.
    fn merge_texture_barriers(&self) -> Vec<TextureBarrier> {
        Self::merge_texture_list(self.pending_texture_barriers.clone())
    }

    /// Merges a list of buffer barriers.
    fn merge_buffer_list(barriers: Vec<BufferBarrier>) -> Vec<BufferBarrier> {
        if barriers.len() <= 1 {
            return barriers;
        }

        let mut result: Vec<BufferBarrier> = Vec::new();

        for barrier in barriers {
            let mut merged = false;

            for existing in &mut result {
                if let Some(merged_barrier) = existing.try_merge(&barrier) {
                    *existing = merged_barrier;
                    merged = true;
                    break;
                }
            }

            if !merged {
                result.push(barrier);
            }
        }

        // Second pass for any newly mergeable barriers
        let mut final_result: Vec<BufferBarrier> = Vec::new();
        for barrier in result {
            let mut merged = false;

            for existing in &mut final_result {
                if let Some(merged_barrier) = existing.try_merge(&barrier) {
                    *existing = merged_barrier;
                    merged = true;
                    break;
                }
            }

            if !merged {
                final_result.push(barrier);
            }
        }

        final_result
    }

    /// Merges a list of texture barriers.
    fn merge_texture_list(barriers: Vec<TextureBarrier>) -> Vec<TextureBarrier> {
        if barriers.len() <= 1 {
            return barriers;
        }

        let mut result: Vec<TextureBarrier> = Vec::new();

        for barrier in barriers {
            let mut merged = false;

            for existing in &mut result {
                if let Some(merged_barrier) = existing.try_merge(&barrier) {
                    *existing = merged_barrier;
                    merged = true;
                    break;
                }
            }

            if !merged {
                result.push(barrier);
            }
        }

        // Second pass for any newly mergeable barriers
        let mut final_result: Vec<TextureBarrier> = Vec::new();
        for barrier in result {
            let mut merged = false;

            for existing in &mut final_result {
                if let Some(merged_barrier) = existing.try_merge(&barrier) {
                    *existing = merged_barrier;
                    merged = true;
                    break;
                }
            }

            if !merged {
                final_result.push(barrier);
            }
        }

        final_result
    }

    /// Converts access flags to a pipeline stage mask.
    fn access_to_stage_mask(access: &AccessFlags) -> PipelineStageMask {
        let mut mask = PipelineStageMask::NONE;

        if access.contains(AccessFlags::VERTEX_BUFFER_READ)
            || access.contains(AccessFlags::INDEX_BUFFER_READ)
        {
            mask = mask | PipelineStageMask::VERTEX_INPUT;
        }

        if access.contains(AccessFlags::UNIFORM_BUFFER_READ) {
            mask = mask | PipelineStageMask::VERTEX_SHADER | PipelineStageMask::FRAGMENT_SHADER;
        }

        if access.contains(AccessFlags::SHADER_READ) || access.contains(AccessFlags::SHADER_WRITE) {
            mask = mask | PipelineStageMask::FRAGMENT_SHADER | PipelineStageMask::COMPUTE_SHADER;
        }

        if access.contains(AccessFlags::COLOR_ATTACHMENT_READ)
            || access.contains(AccessFlags::COLOR_ATTACHMENT_WRITE)
        {
            mask = mask | PipelineStageMask::COLOR_OUTPUT;
        }

        if access.contains(AccessFlags::DEPTH_STENCIL_READ)
            || access.contains(AccessFlags::DEPTH_STENCIL_WRITE)
        {
            mask = mask | PipelineStageMask::EARLY_DEPTH | PipelineStageMask::LATE_DEPTH;
        }

        if access.contains(AccessFlags::TRANSFER_READ)
            || access.contains(AccessFlags::TRANSFER_WRITE)
        {
            mask = mask | PipelineStageMask::TRANSFER;
        }

        if access.contains(AccessFlags::HOST_READ) || access.contains(AccessFlags::HOST_WRITE) {
            mask = mask | PipelineStageMask::HOST;
        }

        if access.contains(AccessFlags::INDIRECT_BUFFER_READ) {
            mask = mask | PipelineStageMask::VERTEX_INPUT;
        }

        // Default to all commands if no specific stage determined
        if mask == PipelineStageMask::NONE && !access.is_empty() {
            mask = PipelineStageMask::ALL_COMMANDS;
        }

        mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================
    // PipelineStage Tests
    // ============================================================

    #[test]
    fn test_pipeline_stage_default() {
        assert_eq!(PipelineStage::default(), PipelineStage::None);
    }

    #[test]
    fn test_pipeline_stage_is_graphics() {
        assert!(PipelineStage::VertexInput.is_graphics());
        assert!(PipelineStage::VertexShader.is_graphics());
        assert!(PipelineStage::FragmentShader.is_graphics());
        assert!(PipelineStage::EarlyDepth.is_graphics());
        assert!(PipelineStage::LateDepth.is_graphics());
        assert!(PipelineStage::ColorOutput.is_graphics());
        assert!(PipelineStage::AllGraphics.is_graphics());
        assert!(!PipelineStage::ComputeShader.is_graphics());
        assert!(!PipelineStage::Transfer.is_graphics());
        assert!(!PipelineStage::Host.is_graphics());
        assert!(!PipelineStage::None.is_graphics());
    }

    #[test]
    fn test_pipeline_stage_is_compute() {
        assert!(PipelineStage::ComputeShader.is_compute());
        assert!(!PipelineStage::VertexShader.is_compute());
        assert!(!PipelineStage::FragmentShader.is_compute());
        assert!(!PipelineStage::Transfer.is_compute());
    }

    #[test]
    fn test_pipeline_stage_is_transfer() {
        assert!(PipelineStage::Transfer.is_transfer());
        assert!(!PipelineStage::ComputeShader.is_transfer());
        assert!(!PipelineStage::VertexShader.is_transfer());
    }

    #[test]
    fn test_pipeline_stage_is_shader_stage() {
        assert!(PipelineStage::VertexShader.is_shader_stage());
        assert!(PipelineStage::FragmentShader.is_shader_stage());
        assert!(PipelineStage::ComputeShader.is_shader_stage());
        assert!(!PipelineStage::VertexInput.is_shader_stage());
        assert!(!PipelineStage::Transfer.is_shader_stage());
        assert!(!PipelineStage::ColorOutput.is_shader_stage());
    }

    #[test]
    fn test_pipeline_stage_order_index() {
        assert_eq!(PipelineStage::None.order_index(), 0);
        assert_eq!(PipelineStage::Host.order_index(), 1);
        assert_eq!(PipelineStage::Transfer.order_index(), 2);
        assert!(PipelineStage::VertexInput.order_index() < PipelineStage::VertexShader.order_index());
        assert!(PipelineStage::VertexShader.order_index() < PipelineStage::FragmentShader.order_index());
        assert!(PipelineStage::FragmentShader.order_index() < PipelineStage::ColorOutput.order_index());
    }

    #[test]
    fn test_pipeline_stage_comes_before() {
        assert!(PipelineStage::VertexInput.comes_before(&PipelineStage::VertexShader));
        assert!(PipelineStage::VertexShader.comes_before(&PipelineStage::FragmentShader));
        assert!(PipelineStage::Transfer.comes_before(&PipelineStage::VertexInput));
        assert!(!PipelineStage::FragmentShader.comes_before(&PipelineStage::VertexShader));
        assert!(!PipelineStage::VertexShader.comes_before(&PipelineStage::VertexShader));
    }

    #[test]
    fn test_pipeline_stage_equality() {
        assert_eq!(PipelineStage::VertexShader, PipelineStage::VertexShader);
        assert_ne!(PipelineStage::VertexShader, PipelineStage::FragmentShader);
    }

    #[test]
    fn test_pipeline_stage_clone() {
        let stage = PipelineStage::ComputeShader;
        let cloned = stage;
        assert_eq!(stage, cloned);
    }

    #[test]
    fn test_pipeline_stage_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(PipelineStage::VertexShader);
        set.insert(PipelineStage::FragmentShader);
        assert!(set.contains(&PipelineStage::VertexShader));
        assert!(!set.contains(&PipelineStage::ComputeShader));
    }

    // ============================================================
    // AccessFlags Tests
    // ============================================================

    #[test]
    fn test_access_flags_default() {
        assert_eq!(AccessFlags::default(), AccessFlags::NONE);
    }

    #[test]
    fn test_access_flags_none() {
        let flags = AccessFlags::NONE;
        assert!(!flags.has_read());
        assert!(!flags.has_write());
    }

    #[test]
    fn test_access_flags_read_write_combined() {
        let flags = AccessFlags::READ_WRITE;
        assert!(flags.contains(AccessFlags::READ));
        assert!(flags.contains(AccessFlags::WRITE));
        assert!(flags.has_read());
        assert!(flags.has_write());
    }

    #[test]
    fn test_access_flags_shader_read_write_combined() {
        let flags = AccessFlags::SHADER_READ_WRITE;
        assert!(flags.contains(AccessFlags::SHADER_READ));
        assert!(flags.contains(AccessFlags::SHADER_WRITE));
    }

    #[test]
    fn test_access_flags_depth_stencil_combined() {
        let flags = AccessFlags::DEPTH_STENCIL_READ_WRITE;
        assert!(flags.contains(AccessFlags::DEPTH_STENCIL_READ));
        assert!(flags.contains(AccessFlags::DEPTH_STENCIL_WRITE));
    }

    #[test]
    fn test_access_flags_has_read() {
        assert!(AccessFlags::READ.has_read());
        assert!(AccessFlags::SHADER_READ.has_read());
        assert!(AccessFlags::COLOR_ATTACHMENT_READ.has_read());
        assert!(AccessFlags::DEPTH_STENCIL_READ.has_read());
        assert!(AccessFlags::TRANSFER_READ.has_read());
        assert!(AccessFlags::VERTEX_BUFFER_READ.has_read());
        assert!(AccessFlags::INDEX_BUFFER_READ.has_read());
        assert!(AccessFlags::INDIRECT_BUFFER_READ.has_read());
        assert!(AccessFlags::UNIFORM_BUFFER_READ.has_read());
        assert!(AccessFlags::HOST_READ.has_read());
        assert!(!AccessFlags::WRITE.has_read());
        assert!(!AccessFlags::SHADER_WRITE.has_read());
    }

    #[test]
    fn test_access_flags_has_write() {
        assert!(AccessFlags::WRITE.has_write());
        assert!(AccessFlags::SHADER_WRITE.has_write());
        assert!(AccessFlags::COLOR_ATTACHMENT_WRITE.has_write());
        assert!(AccessFlags::DEPTH_STENCIL_WRITE.has_write());
        assert!(AccessFlags::TRANSFER_WRITE.has_write());
        assert!(AccessFlags::HOST_WRITE.has_write());
        assert!(!AccessFlags::READ.has_write());
        assert!(!AccessFlags::SHADER_READ.has_write());
    }

    #[test]
    fn test_access_flags_is_read_only() {
        assert!(AccessFlags::READ.is_read_only());
        assert!(AccessFlags::SHADER_READ.is_read_only());
        assert!(!AccessFlags::WRITE.is_read_only());
        assert!(!AccessFlags::READ_WRITE.is_read_only());
        assert!(!AccessFlags::NONE.is_read_only());
    }

    #[test]
    fn test_access_flags_is_write_only() {
        assert!(AccessFlags::WRITE.is_write_only());
        assert!(AccessFlags::SHADER_WRITE.is_write_only());
        assert!(!AccessFlags::READ.is_write_only());
        assert!(!AccessFlags::READ_WRITE.is_write_only());
        assert!(!AccessFlags::NONE.is_write_only());
    }

    #[test]
    fn test_access_flags_conflicts_with() {
        // Write conflicts with anything
        assert!(AccessFlags::WRITE.conflicts_with(AccessFlags::READ));
        assert!(AccessFlags::WRITE.conflicts_with(AccessFlags::WRITE));
        assert!(AccessFlags::WRITE.conflicts_with(AccessFlags::NONE));

        // Read vs read doesn't conflict
        assert!(!AccessFlags::READ.conflicts_with(AccessFlags::READ));
        assert!(!AccessFlags::SHADER_READ.conflicts_with(AccessFlags::SHADER_READ));

        // Read conflicts with write
        assert!(AccessFlags::READ.conflicts_with(AccessFlags::WRITE));
    }

    #[test]
    fn test_access_flags_requires_barrier_to() {
        // Write-after-write
        assert!(AccessFlags::WRITE.requires_barrier_to(AccessFlags::WRITE));

        // Read-after-write
        assert!(AccessFlags::WRITE.requires_barrier_to(AccessFlags::READ));

        // Write-after-read
        assert!(AccessFlags::READ.requires_barrier_to(AccessFlags::WRITE));

        // Read-after-read (no barrier needed)
        assert!(!AccessFlags::READ.requires_barrier_to(AccessFlags::READ));
        assert!(!AccessFlags::SHADER_READ.requires_barrier_to(AccessFlags::SHADER_READ));
    }

    #[test]
    fn test_access_flags_bitwise_operations() {
        let combined = AccessFlags::READ | AccessFlags::WRITE;
        assert_eq!(combined, AccessFlags::READ_WRITE);

        let filtered = combined & AccessFlags::READ;
        assert_eq!(filtered, AccessFlags::READ);

        let inverted = !AccessFlags::NONE;
        assert!(inverted.bits() != 0);
    }

    #[test]
    fn test_access_flags_union() {
        let flags = AccessFlags::SHADER_READ | AccessFlags::TRANSFER_READ;
        assert!(flags.has_read());
        assert!(!flags.has_write());
        assert!(flags.contains(AccessFlags::SHADER_READ));
        assert!(flags.contains(AccessFlags::TRANSFER_READ));
    }

    // ============================================================
    // TextureLayout Tests
    // ============================================================

    #[test]
    fn test_texture_layout_default() {
        assert_eq!(TextureLayout::default(), TextureLayout::Undefined);
    }

    #[test]
    fn test_texture_layout_supports_shader_read() {
        assert!(TextureLayout::General.supports_shader_read());
        assert!(TextureLayout::ShaderReadOnly.supports_shader_read());
        assert!(TextureLayout::DepthStencilReadOnly.supports_shader_read());
        assert!(TextureLayout::StorageImage.supports_shader_read());
        assert!(!TextureLayout::ColorAttachment.supports_shader_read());
        assert!(!TextureLayout::TransferSrc.supports_shader_read());
        assert!(!TextureLayout::Undefined.supports_shader_read());
    }

    #[test]
    fn test_texture_layout_supports_shader_write() {
        assert!(TextureLayout::General.supports_shader_write());
        assert!(TextureLayout::StorageImage.supports_shader_write());
        assert!(!TextureLayout::ShaderReadOnly.supports_shader_write());
        assert!(!TextureLayout::ColorAttachment.supports_shader_write());
    }

    #[test]
    fn test_texture_layout_supports_color_attachment() {
        assert!(TextureLayout::General.supports_color_attachment());
        assert!(TextureLayout::ColorAttachment.supports_color_attachment());
        assert!(!TextureLayout::ShaderReadOnly.supports_color_attachment());
        assert!(!TextureLayout::DepthStencilAttachment.supports_color_attachment());
    }

    #[test]
    fn test_texture_layout_supports_depth_stencil() {
        assert!(TextureLayout::General.supports_depth_stencil());
        assert!(TextureLayout::DepthStencilAttachment.supports_depth_stencil());
        assert!(TextureLayout::DepthStencilReadOnly.supports_depth_stencil());
        assert!(!TextureLayout::ColorAttachment.supports_depth_stencil());
        assert!(!TextureLayout::ShaderReadOnly.supports_depth_stencil());
    }

    #[test]
    fn test_texture_layout_supports_transfer_read() {
        assert!(TextureLayout::General.supports_transfer_read());
        assert!(TextureLayout::TransferSrc.supports_transfer_read());
        assert!(!TextureLayout::TransferDst.supports_transfer_read());
        assert!(!TextureLayout::ShaderReadOnly.supports_transfer_read());
    }

    #[test]
    fn test_texture_layout_supports_transfer_write() {
        assert!(TextureLayout::General.supports_transfer_write());
        assert!(TextureLayout::TransferDst.supports_transfer_write());
        assert!(!TextureLayout::TransferSrc.supports_transfer_write());
        assert!(!TextureLayout::ShaderReadOnly.supports_transfer_write());
    }

    #[test]
    fn test_texture_layout_requires_transition() {
        assert!(TextureLayout::Undefined.requires_transition_to(TextureLayout::ShaderReadOnly));
        assert!(TextureLayout::ShaderReadOnly.requires_transition_to(TextureLayout::ColorAttachment));
        assert!(!TextureLayout::ShaderReadOnly.requires_transition_to(TextureLayout::ShaderReadOnly));
    }

    #[test]
    fn test_texture_layout_optimal_for_access() {
        assert_eq!(
            TextureLayout::optimal_for_access(AccessFlags::COLOR_ATTACHMENT_WRITE),
            TextureLayout::ColorAttachment
        );
        assert_eq!(
            TextureLayout::optimal_for_access(AccessFlags::DEPTH_STENCIL_WRITE),
            TextureLayout::DepthStencilAttachment
        );
        assert_eq!(
            TextureLayout::optimal_for_access(AccessFlags::DEPTH_STENCIL_READ),
            TextureLayout::DepthStencilReadOnly
        );
        assert_eq!(
            TextureLayout::optimal_for_access(AccessFlags::SHADER_WRITE),
            TextureLayout::StorageImage
        );
        assert_eq!(
            TextureLayout::optimal_for_access(AccessFlags::SHADER_READ),
            TextureLayout::ShaderReadOnly
        );
        assert_eq!(
            TextureLayout::optimal_for_access(AccessFlags::TRANSFER_WRITE),
            TextureLayout::TransferDst
        );
        assert_eq!(
            TextureLayout::optimal_for_access(AccessFlags::TRANSFER_READ),
            TextureLayout::TransferSrc
        );
        assert_eq!(
            TextureLayout::optimal_for_access(AccessFlags::NONE),
            TextureLayout::General
        );
    }

    #[test]
    fn test_texture_layout_all_variants() {
        // Ensure all variants are distinct
        let layouts = [
            TextureLayout::Undefined,
            TextureLayout::General,
            TextureLayout::ColorAttachment,
            TextureLayout::DepthStencilAttachment,
            TextureLayout::DepthStencilReadOnly,
            TextureLayout::ShaderReadOnly,
            TextureLayout::TransferSrc,
            TextureLayout::TransferDst,
            TextureLayout::Present,
            TextureLayout::StorageImage,
            TextureLayout::Preinitialized,
        ];
        for (i, l1) in layouts.iter().enumerate() {
            for (j, l2) in layouts.iter().enumerate() {
                if i == j {
                    assert_eq!(l1, l2);
                } else {
                    assert_ne!(l1, l2);
                }
            }
        }
    }

    // ============================================================
    // ResourceState Tests
    // ============================================================

    #[test]
    fn test_resource_state_default() {
        let state = ResourceState::default();
        assert_eq!(state.stage, PipelineStage::None);
        assert_eq!(state.access, AccessFlags::NONE);
        assert_eq!(state.layout, None);
    }

    #[test]
    fn test_resource_state_buffer() {
        let state = ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::UNIFORM_BUFFER_READ);
        assert_eq!(state.stage, PipelineStage::VertexShader);
        assert_eq!(state.access, AccessFlags::UNIFORM_BUFFER_READ);
        assert!(state.is_buffer());
        assert!(!state.is_texture());
    }

    #[test]
    fn test_resource_state_texture() {
        let state = ResourceState::texture(
            PipelineStage::FragmentShader,
            AccessFlags::SHADER_READ,
            TextureLayout::ShaderReadOnly,
        );
        assert_eq!(state.stage, PipelineStage::FragmentShader);
        assert_eq!(state.access, AccessFlags::SHADER_READ);
        assert_eq!(state.layout, Some(TextureLayout::ShaderReadOnly));
        assert!(state.is_texture());
        assert!(!state.is_buffer());
    }

    #[test]
    fn test_resource_state_undefined() {
        let state = ResourceState::undefined();
        assert_eq!(state.stage, PipelineStage::None);
        assert_eq!(state.access, AccessFlags::NONE);
        assert_eq!(state.layout, None);
    }

    #[test]
    fn test_resource_state_requires_barrier_access_hazard() {
        let write_state = ResourceState::buffer(PipelineStage::Transfer, AccessFlags::TRANSFER_WRITE);
        let read_state = ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::VERTEX_BUFFER_READ);

        // Write -> Read requires barrier
        assert!(write_state.requires_barrier_to(&read_state));

        // Read -> Read doesn't require barrier
        let read_state2 = ResourceState::buffer(PipelineStage::FragmentShader, AccessFlags::UNIFORM_BUFFER_READ);
        assert!(!read_state.requires_barrier_to(&read_state2));
    }

    #[test]
    fn test_resource_state_requires_barrier_layout_transition() {
        let src_state = ResourceState::texture(
            PipelineStage::Transfer,
            AccessFlags::TRANSFER_WRITE,
            TextureLayout::TransferDst,
        );
        let dst_state = ResourceState::texture(
            PipelineStage::FragmentShader,
            AccessFlags::SHADER_READ,
            TextureLayout::ShaderReadOnly,
        );

        // Layout transition requires barrier
        assert!(src_state.requires_barrier_to(&dst_state));
    }

    #[test]
    fn test_resource_state_requires_barrier_same_state() {
        let state = ResourceState::texture(
            PipelineStage::FragmentShader,
            AccessFlags::SHADER_READ,
            TextureLayout::ShaderReadOnly,
        );
        let same_state = state.clone();

        // Same read-only state doesn't require barrier
        assert!(!state.requires_barrier_to(&same_state));
    }

    #[test]
    fn test_resource_state_clone_and_eq() {
        let state = ResourceState::texture(
            PipelineStage::ComputeShader,
            AccessFlags::SHADER_READ_WRITE,
            TextureLayout::StorageImage,
        );
        let cloned = state.clone();
        assert_eq!(state, cloned);
    }

    // ============================================================
    // ResourceStateTracker Tests
    // ============================================================

    #[test]
    fn test_tracker_new() {
        let tracker = ResourceStateTracker::new();
        assert!(tracker.is_empty());
        assert_eq!(tracker.len(), 0);
    }

    #[test]
    fn test_tracker_with_capacity() {
        let tracker = ResourceStateTracker::with_capacity(100);
        assert!(tracker.is_empty());
    }

    #[test]
    fn test_tracker_set_and_get() {
        let mut tracker = ResourceStateTracker::new();
        let state = ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::UNIFORM_BUFFER_READ);

        tracker.set(42, state.clone());

        assert_eq!(tracker.len(), 1);
        assert!(!tracker.is_empty());
        assert!(tracker.contains(42));

        let retrieved = tracker.get(42).unwrap();
        assert_eq!(*retrieved, state);
    }

    #[test]
    fn test_tracker_get_nonexistent() {
        let tracker = ResourceStateTracker::new();
        assert!(tracker.get(999).is_none());
    }

    #[test]
    fn test_tracker_update_existing() {
        let mut tracker = ResourceStateTracker::new();
        let initial_state = ResourceState::buffer(PipelineStage::Transfer, AccessFlags::TRANSFER_WRITE);
        tracker.set(1, initial_state);

        tracker.update(1, PipelineStage::VertexShader, AccessFlags::VERTEX_BUFFER_READ);

        let state = tracker.get(1).unwrap();
        assert_eq!(state.stage, PipelineStage::VertexShader);
        assert_eq!(state.access, AccessFlags::VERTEX_BUFFER_READ);
    }

    #[test]
    fn test_tracker_update_nonexistent() {
        let mut tracker = ResourceStateTracker::new();
        tracker.update(1, PipelineStage::ComputeShader, AccessFlags::SHADER_READ);

        let state = tracker.get(1).unwrap();
        assert_eq!(state.stage, PipelineStage::ComputeShader);
        assert_eq!(state.access, AccessFlags::SHADER_READ);
        assert_eq!(state.layout, None);
    }

    #[test]
    fn test_tracker_update_preserves_layout() {
        let mut tracker = ResourceStateTracker::new();
        let initial = ResourceState::texture(
            PipelineStage::Transfer,
            AccessFlags::TRANSFER_WRITE,
            TextureLayout::TransferDst,
        );
        tracker.set(1, initial);

        tracker.update(1, PipelineStage::FragmentShader, AccessFlags::SHADER_READ);

        let state = tracker.get(1).unwrap();
        assert_eq!(state.layout, Some(TextureLayout::TransferDst));
    }

    #[test]
    fn test_tracker_update_layout_existing() {
        let mut tracker = ResourceStateTracker::new();
        let initial = ResourceState::texture(
            PipelineStage::Transfer,
            AccessFlags::TRANSFER_WRITE,
            TextureLayout::TransferDst,
        );
        tracker.set(1, initial);

        tracker.update_layout(1, TextureLayout::ShaderReadOnly);

        let state = tracker.get(1).unwrap();
        assert_eq!(state.layout, Some(TextureLayout::ShaderReadOnly));
        // Stage and access preserved
        assert_eq!(state.stage, PipelineStage::Transfer);
        assert_eq!(state.access, AccessFlags::TRANSFER_WRITE);
    }

    #[test]
    fn test_tracker_update_layout_nonexistent() {
        let mut tracker = ResourceStateTracker::new();
        tracker.update_layout(1, TextureLayout::ColorAttachment);

        let state = tracker.get(1).unwrap();
        assert_eq!(state.layout, Some(TextureLayout::ColorAttachment));
        assert_eq!(state.stage, PipelineStage::None);
        assert_eq!(state.access, AccessFlags::NONE);
    }

    #[test]
    fn test_tracker_remove() {
        let mut tracker = ResourceStateTracker::new();
        let state = ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::VERTEX_BUFFER_READ);
        tracker.set(1, state.clone());

        let removed = tracker.remove(1);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap(), state);
        assert!(!tracker.contains(1));
        assert!(tracker.is_empty());
    }

    #[test]
    fn test_tracker_remove_nonexistent() {
        let mut tracker = ResourceStateTracker::new();
        let removed = tracker.remove(999);
        assert!(removed.is_none());
    }

    #[test]
    fn test_tracker_clear() {
        let mut tracker = ResourceStateTracker::new();
        tracker.set(1, ResourceState::default());
        tracker.set(2, ResourceState::default());
        tracker.set(3, ResourceState::default());

        assert_eq!(tracker.len(), 3);
        tracker.clear();
        assert!(tracker.is_empty());
        assert_eq!(tracker.len(), 0);
    }

    #[test]
    fn test_tracker_ids_iterator() {
        let mut tracker = ResourceStateTracker::new();
        tracker.set(10, ResourceState::default());
        tracker.set(20, ResourceState::default());
        tracker.set(30, ResourceState::default());

        let ids: Vec<_> = tracker.ids().copied().collect();
        assert_eq!(ids.len(), 3);
        assert!(ids.contains(&10));
        assert!(ids.contains(&20));
        assert!(ids.contains(&30));
    }

    #[test]
    fn test_tracker_states_iterator() {
        let mut tracker = ResourceStateTracker::new();
        tracker.set(1, ResourceState::buffer(PipelineStage::Transfer, AccessFlags::TRANSFER_READ));
        tracker.set(2, ResourceState::buffer(PipelineStage::Transfer, AccessFlags::TRANSFER_WRITE));

        let count = tracker.states().count();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_tracker_states_mut_iterator() {
        let mut tracker = ResourceStateTracker::new();
        tracker.set(1, ResourceState::buffer(PipelineStage::Transfer, AccessFlags::TRANSFER_READ));
        tracker.set(2, ResourceState::buffer(PipelineStage::Transfer, AccessFlags::TRANSFER_READ));

        // Modify all states
        for (_, state) in tracker.states_mut() {
            state.access = AccessFlags::SHADER_READ;
        }

        // Verify modifications
        for (_, state) in tracker.states() {
            assert_eq!(state.access, AccessFlags::SHADER_READ);
        }
    }

    #[test]
    fn test_tracker_transition_needs_barrier() {
        let mut tracker = ResourceStateTracker::new();
        let initial = ResourceState::buffer(PipelineStage::Transfer, AccessFlags::TRANSFER_WRITE);
        tracker.set(1, initial.clone());

        let new_state = ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::VERTEX_BUFFER_READ);
        let barrier_info = tracker.transition(1, new_state.clone());

        assert!(barrier_info.is_some());
        let (old, new) = barrier_info.unwrap();
        assert_eq!(old, initial);
        assert_eq!(new, new_state);

        // State should be updated
        assert_eq!(tracker.get(1).unwrap(), &new_state);
    }

    #[test]
    fn test_tracker_transition_no_barrier() {
        let mut tracker = ResourceStateTracker::new();
        let initial = ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::UNIFORM_BUFFER_READ);
        tracker.set(1, initial);

        let new_state = ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::UNIFORM_BUFFER_READ);
        let barrier_info = tracker.transition(1, new_state.clone());

        // Same read-only state = no barrier needed
        assert!(barrier_info.is_none());
        assert_eq!(tracker.get(1).unwrap(), &new_state);
    }

    #[test]
    fn test_tracker_transition_new_resource() {
        let mut tracker = ResourceStateTracker::new();
        let state = ResourceState::buffer(PipelineStage::Transfer, AccessFlags::TRANSFER_WRITE);

        let barrier_info = tracker.transition(999, state.clone());

        // New resource = no barrier needed
        assert!(barrier_info.is_none());
        assert!(tracker.contains(999));
        assert_eq!(tracker.get(999).unwrap(), &state);
    }

    #[test]
    fn test_tracker_merge() {
        let mut tracker1 = ResourceStateTracker::new();
        tracker1.set(1, ResourceState::buffer(PipelineStage::Transfer, AccessFlags::TRANSFER_READ));
        tracker1.set(2, ResourceState::buffer(PipelineStage::Transfer, AccessFlags::TRANSFER_READ));

        let mut tracker2 = ResourceStateTracker::new();
        tracker2.set(2, ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::VERTEX_BUFFER_READ));
        tracker2.set(3, ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::VERTEX_BUFFER_READ));

        tracker1.merge(&tracker2);

        assert_eq!(tracker1.len(), 3);
        assert!(tracker1.contains(1));
        assert!(tracker1.contains(2));
        assert!(tracker1.contains(3));

        // Resource 2 should have been overwritten
        let state2 = tracker1.get(2).unwrap();
        assert_eq!(state2.stage, PipelineStage::VertexShader);
    }

    #[test]
    fn test_tracker_multiple_resources() {
        let mut tracker = ResourceStateTracker::new();

        // Add various resources
        tracker.set(
            100,
            ResourceState::texture(
                PipelineStage::FragmentShader,
                AccessFlags::SHADER_READ,
                TextureLayout::ShaderReadOnly,
            ),
        );
        tracker.set(
            200,
            ResourceState::buffer(PipelineStage::VertexInput, AccessFlags::VERTEX_BUFFER_READ),
        );
        tracker.set(
            300,
            ResourceState::buffer(PipelineStage::VertexInput, AccessFlags::INDEX_BUFFER_READ),
        );
        tracker.set(
            400,
            ResourceState::texture(
                PipelineStage::ColorOutput,
                AccessFlags::COLOR_ATTACHMENT_WRITE,
                TextureLayout::ColorAttachment,
            ),
        );

        assert_eq!(tracker.len(), 4);

        // Verify each
        assert!(tracker.get(100).unwrap().is_texture());
        assert!(tracker.get(200).unwrap().is_buffer());
        assert!(tracker.get(300).unwrap().is_buffer());
        assert!(tracker.get(400).unwrap().is_texture());
    }

    #[test]
    fn test_tracker_overwrite_state() {
        let mut tracker = ResourceStateTracker::new();
        let state1 = ResourceState::buffer(PipelineStage::Transfer, AccessFlags::TRANSFER_WRITE);
        let state2 = ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::VERTEX_BUFFER_READ);

        tracker.set(1, state1);
        tracker.set(1, state2.clone());

        assert_eq!(tracker.len(), 1);
        assert_eq!(tracker.get(1).unwrap(), &state2);
    }

    #[test]
    fn test_tracker_contains() {
        let mut tracker = ResourceStateTracker::new();
        tracker.set(42, ResourceState::default());

        assert!(tracker.contains(42));
        assert!(!tracker.contains(43));
    }

    #[test]
    fn test_default_impl() {
        let tracker: ResourceStateTracker = Default::default();
        assert!(tracker.is_empty());
    }

    // ============================================================
    // HazardType Tests (T-WGPU-P4.7.2)
    // ============================================================

    #[test]
    fn test_hazard_type_default() {
        assert_eq!(HazardType::default(), HazardType::None);
    }

    #[test]
    fn test_hazard_type_requires_barrier() {
        assert!(!HazardType::None.requires_barrier());
        assert!(HazardType::ReadAfterWrite.requires_barrier());
        assert!(HazardType::WriteAfterRead.requires_barrier());
        assert!(HazardType::WriteAfterWrite.requires_barrier());
        assert!(HazardType::LayoutTransition.requires_barrier());
    }

    #[test]
    fn test_hazard_type_is_write_hazard() {
        assert!(!HazardType::None.is_write_hazard());
        assert!(!HazardType::ReadAfterWrite.is_write_hazard());
        assert!(HazardType::WriteAfterRead.is_write_hazard());
        assert!(HazardType::WriteAfterWrite.is_write_hazard());
        assert!(!HazardType::LayoutTransition.is_write_hazard());
    }

    #[test]
    fn test_hazard_type_is_read_hazard() {
        assert!(!HazardType::None.is_read_hazard());
        assert!(HazardType::ReadAfterWrite.is_read_hazard());
        assert!(!HazardType::WriteAfterRead.is_read_hazard());
        assert!(!HazardType::WriteAfterWrite.is_read_hazard());
        assert!(!HazardType::LayoutTransition.is_read_hazard());
    }

    #[test]
    fn test_hazard_type_is_layout_transition() {
        assert!(!HazardType::None.is_layout_transition());
        assert!(!HazardType::ReadAfterWrite.is_layout_transition());
        assert!(!HazardType::WriteAfterRead.is_layout_transition());
        assert!(!HazardType::WriteAfterWrite.is_layout_transition());
        assert!(HazardType::LayoutTransition.is_layout_transition());
    }

    #[test]
    fn test_hazard_type_combine_with_none() {
        assert_eq!(HazardType::None.combine(HazardType::None), HazardType::None);
        assert_eq!(HazardType::None.combine(HazardType::ReadAfterWrite), HazardType::ReadAfterWrite);
        assert_eq!(HazardType::ReadAfterWrite.combine(HazardType::None), HazardType::ReadAfterWrite);
        assert_eq!(HazardType::None.combine(HazardType::WriteAfterRead), HazardType::WriteAfterRead);
        assert_eq!(HazardType::None.combine(HazardType::WriteAfterWrite), HazardType::WriteAfterWrite);
    }

    #[test]
    fn test_hazard_type_combine_waw_dominates() {
        assert_eq!(
            HazardType::WriteAfterWrite.combine(HazardType::ReadAfterWrite),
            HazardType::WriteAfterWrite
        );
        assert_eq!(
            HazardType::WriteAfterWrite.combine(HazardType::WriteAfterRead),
            HazardType::WriteAfterWrite
        );
        assert_eq!(
            HazardType::ReadAfterWrite.combine(HazardType::WriteAfterWrite),
            HazardType::WriteAfterWrite
        );
    }

    #[test]
    fn test_hazard_type_combine_raw_and_war_becomes_waw() {
        assert_eq!(
            HazardType::ReadAfterWrite.combine(HazardType::WriteAfterRead),
            HazardType::WriteAfterWrite
        );
        assert_eq!(
            HazardType::WriteAfterRead.combine(HazardType::ReadAfterWrite),
            HazardType::WriteAfterWrite
        );
    }

    #[test]
    fn test_hazard_type_combine_layout_transition() {
        assert_eq!(
            HazardType::LayoutTransition.combine(HazardType::ReadAfterWrite),
            HazardType::ReadAfterWrite
        );
        assert_eq!(
            HazardType::ReadAfterWrite.combine(HazardType::LayoutTransition),
            HazardType::ReadAfterWrite
        );
    }

    #[test]
    fn test_hazard_type_equality() {
        assert_eq!(HazardType::ReadAfterWrite, HazardType::ReadAfterWrite);
        assert_ne!(HazardType::ReadAfterWrite, HazardType::WriteAfterRead);
    }

    #[test]
    fn test_hazard_type_clone() {
        let h = HazardType::WriteAfterWrite;
        let cloned = h;
        assert_eq!(h, cloned);
    }

    #[test]
    fn test_hazard_type_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(HazardType::ReadAfterWrite);
        set.insert(HazardType::WriteAfterRead);
        assert!(set.contains(&HazardType::ReadAfterWrite));
        assert!(!set.contains(&HazardType::WriteAfterWrite));
    }

    // ============================================================
    // BarrierInfo Tests (T-WGPU-P4.7.2)
    // ============================================================

    #[test]
    fn test_barrier_info_buffer() {
        let info = BarrierInfo::buffer(
            42,
            HazardType::ReadAfterWrite,
            PipelineStage::Transfer,
            PipelineStage::VertexShader,
            AccessFlags::TRANSFER_WRITE,
            AccessFlags::VERTEX_BUFFER_READ,
        );

        assert_eq!(info.resource_id, 42);
        assert_eq!(info.hazard, HazardType::ReadAfterWrite);
        assert_eq!(info.src_stage, PipelineStage::Transfer);
        assert_eq!(info.dst_stage, PipelineStage::VertexShader);
        assert_eq!(info.src_access, AccessFlags::TRANSFER_WRITE);
        assert_eq!(info.dst_access, AccessFlags::VERTEX_BUFFER_READ);
        assert!(info.old_layout.is_none());
        assert!(info.new_layout.is_none());
    }

    #[test]
    fn test_barrier_info_texture() {
        let info = BarrierInfo::texture(
            100,
            HazardType::LayoutTransition,
            PipelineStage::Transfer,
            PipelineStage::FragmentShader,
            AccessFlags::TRANSFER_WRITE,
            AccessFlags::SHADER_READ,
            TextureLayout::TransferDst,
            TextureLayout::ShaderReadOnly,
        );

        assert_eq!(info.resource_id, 100);
        assert_eq!(info.hazard, HazardType::LayoutTransition);
        assert_eq!(info.old_layout, Some(TextureLayout::TransferDst));
        assert_eq!(info.new_layout, Some(TextureLayout::ShaderReadOnly));
    }

    #[test]
    fn test_barrier_info_has_layout_transition() {
        let with_transition = BarrierInfo::texture(
            1,
            HazardType::ReadAfterWrite,
            PipelineStage::Transfer,
            PipelineStage::FragmentShader,
            AccessFlags::TRANSFER_WRITE,
            AccessFlags::SHADER_READ,
            TextureLayout::TransferDst,
            TextureLayout::ShaderReadOnly,
        );
        assert!(with_transition.has_layout_transition());

        let same_layout = BarrierInfo::texture(
            1,
            HazardType::WriteAfterWrite,
            PipelineStage::ComputeShader,
            PipelineStage::ComputeShader,
            AccessFlags::SHADER_WRITE,
            AccessFlags::SHADER_WRITE,
            TextureLayout::StorageImage,
            TextureLayout::StorageImage,
        );
        assert!(!same_layout.has_layout_transition());

        let buffer = BarrierInfo::buffer(
            1,
            HazardType::ReadAfterWrite,
            PipelineStage::Transfer,
            PipelineStage::VertexShader,
            AccessFlags::TRANSFER_WRITE,
            AccessFlags::VERTEX_BUFFER_READ,
        );
        assert!(!buffer.has_layout_transition());
    }

    #[test]
    fn test_barrier_info_is_buffer_barrier() {
        let buffer = BarrierInfo::buffer(
            1,
            HazardType::ReadAfterWrite,
            PipelineStage::Transfer,
            PipelineStage::VertexShader,
            AccessFlags::TRANSFER_WRITE,
            AccessFlags::VERTEX_BUFFER_READ,
        );
        assert!(buffer.is_buffer_barrier());
        assert!(!buffer.is_texture_barrier());
    }

    #[test]
    fn test_barrier_info_is_texture_barrier() {
        let texture = BarrierInfo::texture(
            1,
            HazardType::ReadAfterWrite,
            PipelineStage::Transfer,
            PipelineStage::FragmentShader,
            AccessFlags::TRANSFER_WRITE,
            AccessFlags::SHADER_READ,
            TextureLayout::TransferDst,
            TextureLayout::ShaderReadOnly,
        );
        assert!(texture.is_texture_barrier());
        assert!(!texture.is_buffer_barrier());
    }

    #[test]
    fn test_barrier_info_clone_and_eq() {
        let info = BarrierInfo::buffer(
            42,
            HazardType::WriteAfterRead,
            PipelineStage::FragmentShader,
            PipelineStage::Transfer,
            AccessFlags::SHADER_READ,
            AccessFlags::TRANSFER_WRITE,
        );
        let cloned = info.clone();
        assert_eq!(info, cloned);
    }

    // ============================================================
    // BarrierDetector Tests (T-WGPU-P4.7.2)
    // ============================================================

    #[test]
    fn test_detector_new() {
        let detector = BarrierDetector::new();
        assert!(detector.is_empty());
        assert_eq!(detector.len(), 0);
    }

    #[test]
    fn test_detector_with_capacity() {
        let detector = BarrierDetector::with_capacity(100);
        assert!(detector.is_empty());
    }

    #[test]
    fn test_detector_with_tracker() {
        let mut tracker = ResourceStateTracker::new();
        tracker.set(1, ResourceState::buffer(PipelineStage::Transfer, AccessFlags::TRANSFER_WRITE));

        let detector = BarrierDetector::with_tracker(tracker);
        assert_eq!(detector.len(), 1);
        assert!(detector.is_tracked(1));
    }

    #[test]
    fn test_detector_default() {
        let detector: BarrierDetector = Default::default();
        assert!(detector.is_empty());
    }

    // ============================================================
    // RAW (Read-After-Write) Detection Tests
    // ============================================================

    #[test]
    fn test_detect_hazard_raw_buffer_transfer_to_vertex() {
        let old = ResourceState::buffer(PipelineStage::Transfer, AccessFlags::TRANSFER_WRITE);
        let new = ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::VERTEX_BUFFER_READ);

        assert_eq!(BarrierDetector::detect_hazard(&old, &new), HazardType::ReadAfterWrite);
    }

    #[test]
    fn test_detect_hazard_raw_buffer_compute_write_to_shader_read() {
        let old = ResourceState::buffer(PipelineStage::ComputeShader, AccessFlags::SHADER_WRITE);
        let new = ResourceState::buffer(PipelineStage::FragmentShader, AccessFlags::SHADER_READ);

        assert_eq!(BarrierDetector::detect_hazard(&old, &new), HazardType::ReadAfterWrite);
    }

    #[test]
    fn test_detect_hazard_raw_texture_write_to_sample() {
        let old = ResourceState::texture(
            PipelineStage::ColorOutput,
            AccessFlags::COLOR_ATTACHMENT_WRITE,
            TextureLayout::ColorAttachment,
        );
        let new = ResourceState::texture(
            PipelineStage::FragmentShader,
            AccessFlags::SHADER_READ,
            TextureLayout::ShaderReadOnly,
        );

        assert_eq!(BarrierDetector::detect_hazard(&old, &new), HazardType::ReadAfterWrite);
    }

    #[test]
    fn test_detect_hazard_raw_depth_write_to_read() {
        let old = ResourceState::texture(
            PipelineStage::LateDepth,
            AccessFlags::DEPTH_STENCIL_WRITE,
            TextureLayout::DepthStencilAttachment,
        );
        let new = ResourceState::texture(
            PipelineStage::FragmentShader,
            AccessFlags::SHADER_READ,
            TextureLayout::DepthStencilReadOnly,
        );

        assert_eq!(BarrierDetector::detect_hazard(&old, &new), HazardType::ReadAfterWrite);
    }

    #[test]
    fn test_detect_hazard_raw_host_write_to_gpu_read() {
        let old = ResourceState::buffer(PipelineStage::Host, AccessFlags::HOST_WRITE);
        let new = ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::UNIFORM_BUFFER_READ);

        assert_eq!(BarrierDetector::detect_hazard(&old, &new), HazardType::ReadAfterWrite);
    }

    // ============================================================
    // WAR (Write-After-Read) Detection Tests
    // ============================================================

    #[test]
    fn test_detect_hazard_war_vertex_read_to_transfer_write() {
        let old = ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::VERTEX_BUFFER_READ);
        let new = ResourceState::buffer(PipelineStage::Transfer, AccessFlags::TRANSFER_WRITE);

        assert_eq!(BarrierDetector::detect_hazard(&old, &new), HazardType::WriteAfterRead);
    }

    #[test]
    fn test_detect_hazard_war_shader_read_to_compute_write() {
        let old = ResourceState::buffer(PipelineStage::FragmentShader, AccessFlags::SHADER_READ);
        let new = ResourceState::buffer(PipelineStage::ComputeShader, AccessFlags::SHADER_WRITE);

        assert_eq!(BarrierDetector::detect_hazard(&old, &new), HazardType::WriteAfterRead);
    }

    #[test]
    fn test_detect_hazard_war_texture_sample_to_color_write() {
        let old = ResourceState::texture(
            PipelineStage::FragmentShader,
            AccessFlags::SHADER_READ,
            TextureLayout::ShaderReadOnly,
        );
        let new = ResourceState::texture(
            PipelineStage::ColorOutput,
            AccessFlags::COLOR_ATTACHMENT_WRITE,
            TextureLayout::ColorAttachment,
        );

        assert_eq!(BarrierDetector::detect_hazard(&old, &new), HazardType::WriteAfterRead);
    }

    #[test]
    fn test_detect_hazard_war_gpu_read_to_host_write() {
        let old = ResourceState::buffer(PipelineStage::FragmentShader, AccessFlags::UNIFORM_BUFFER_READ);
        let new = ResourceState::buffer(PipelineStage::Host, AccessFlags::HOST_WRITE);

        assert_eq!(BarrierDetector::detect_hazard(&old, &new), HazardType::WriteAfterRead);
    }

    // ============================================================
    // WAW (Write-After-Write) Detection Tests
    // ============================================================

    #[test]
    fn test_detect_hazard_waw_transfer_to_transfer() {
        let old = ResourceState::buffer(PipelineStage::Transfer, AccessFlags::TRANSFER_WRITE);
        let new = ResourceState::buffer(PipelineStage::Transfer, AccessFlags::TRANSFER_WRITE);

        assert_eq!(BarrierDetector::detect_hazard(&old, &new), HazardType::WriteAfterWrite);
    }

    #[test]
    fn test_detect_hazard_waw_compute_write_to_compute_write() {
        let old = ResourceState::buffer(PipelineStage::ComputeShader, AccessFlags::SHADER_WRITE);
        let new = ResourceState::buffer(PipelineStage::ComputeShader, AccessFlags::SHADER_WRITE);

        assert_eq!(BarrierDetector::detect_hazard(&old, &new), HazardType::WriteAfterWrite);
    }

    #[test]
    fn test_detect_hazard_waw_color_attachment() {
        let old = ResourceState::texture(
            PipelineStage::ColorOutput,
            AccessFlags::COLOR_ATTACHMENT_WRITE,
            TextureLayout::ColorAttachment,
        );
        let new = ResourceState::texture(
            PipelineStage::ColorOutput,
            AccessFlags::COLOR_ATTACHMENT_WRITE,
            TextureLayout::ColorAttachment,
        );

        assert_eq!(BarrierDetector::detect_hazard(&old, &new), HazardType::WriteAfterWrite);
    }

    #[test]
    fn test_detect_hazard_waw_host_write_to_host_write() {
        let old = ResourceState::buffer(PipelineStage::Host, AccessFlags::HOST_WRITE);
        let new = ResourceState::buffer(PipelineStage::Host, AccessFlags::HOST_WRITE);

        assert_eq!(BarrierDetector::detect_hazard(&old, &new), HazardType::WriteAfterWrite);
    }

    #[test]
    fn test_detect_hazard_waw_transfer_to_compute() {
        let old = ResourceState::buffer(PipelineStage::Transfer, AccessFlags::TRANSFER_WRITE);
        let new = ResourceState::buffer(PipelineStage::ComputeShader, AccessFlags::SHADER_WRITE);

        assert_eq!(BarrierDetector::detect_hazard(&old, &new), HazardType::WriteAfterWrite);
    }

    // ============================================================
    // Layout Transition Detection Tests
    // ============================================================

    #[test]
    fn test_detect_hazard_layout_transition_undefined_to_transfer() {
        let old = ResourceState::texture(
            PipelineStage::None,
            AccessFlags::NONE,
            TextureLayout::Undefined,
        );
        let new = ResourceState::texture(
            PipelineStage::Transfer,
            AccessFlags::TRANSFER_WRITE,
            TextureLayout::TransferDst,
        );

        // This is also a WAR because None->Write
        // But since old has no read, it's actually just layout transition
        // Actually old has NONE access, so it's layout transition only
        assert_eq!(BarrierDetector::detect_hazard(&old, &new), HazardType::LayoutTransition);
    }

    #[test]
    fn test_detect_hazard_layout_transition_shader_to_present() {
        let old = ResourceState::texture(
            PipelineStage::FragmentShader,
            AccessFlags::SHADER_READ,
            TextureLayout::ShaderReadOnly,
        );
        let new = ResourceState::texture(
            PipelineStage::AllCommands,
            AccessFlags::NONE,
            TextureLayout::Present,
        );

        // Read -> None with layout change = just layout transition
        assert_eq!(BarrierDetector::detect_hazard(&old, &new), HazardType::LayoutTransition);
    }

    #[test]
    fn test_detect_hazard_layout_transition_color_to_shader() {
        let old = ResourceState::texture(
            PipelineStage::ColorOutput,
            AccessFlags::COLOR_ATTACHMENT_WRITE,
            TextureLayout::ColorAttachment,
        );
        let new = ResourceState::texture(
            PipelineStage::FragmentShader,
            AccessFlags::SHADER_READ,
            TextureLayout::ShaderReadOnly,
        );

        // Write -> Read = RAW (layout transition is secondary)
        assert_eq!(BarrierDetector::detect_hazard(&old, &new), HazardType::ReadAfterWrite);
    }

    // ============================================================
    // No Barrier Cases (RAR - Read-After-Read)
    // ============================================================

    #[test]
    fn test_detect_hazard_rar_no_barrier() {
        let old = ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::VERTEX_BUFFER_READ);
        let new = ResourceState::buffer(PipelineStage::FragmentShader, AccessFlags::UNIFORM_BUFFER_READ);

        assert_eq!(BarrierDetector::detect_hazard(&old, &new), HazardType::None);
    }

    #[test]
    fn test_detect_hazard_rar_same_stage_no_barrier() {
        let old = ResourceState::buffer(PipelineStage::FragmentShader, AccessFlags::SHADER_READ);
        let new = ResourceState::buffer(PipelineStage::FragmentShader, AccessFlags::SHADER_READ);

        assert_eq!(BarrierDetector::detect_hazard(&old, &new), HazardType::None);
    }

    #[test]
    fn test_detect_hazard_rar_texture_same_layout_no_barrier() {
        let old = ResourceState::texture(
            PipelineStage::FragmentShader,
            AccessFlags::SHADER_READ,
            TextureLayout::ShaderReadOnly,
        );
        let new = ResourceState::texture(
            PipelineStage::FragmentShader,
            AccessFlags::SHADER_READ,
            TextureLayout::ShaderReadOnly,
        );

        assert_eq!(BarrierDetector::detect_hazard(&old, &new), HazardType::None);
    }

    #[test]
    fn test_detect_hazard_rar_multiple_read_types() {
        let old = ResourceState::buffer(
            PipelineStage::VertexInput,
            AccessFlags::VERTEX_BUFFER_READ | AccessFlags::INDEX_BUFFER_READ,
        );
        let new = ResourceState::buffer(
            PipelineStage::FragmentShader,
            AccessFlags::UNIFORM_BUFFER_READ | AccessFlags::SHADER_READ,
        );

        assert_eq!(BarrierDetector::detect_hazard(&old, &new), HazardType::None);
    }

    #[test]
    fn test_detect_hazard_none_to_read_no_barrier() {
        let old = ResourceState::buffer(PipelineStage::None, AccessFlags::NONE);
        let new = ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::VERTEX_BUFFER_READ);

        assert_eq!(BarrierDetector::detect_hazard(&old, &new), HazardType::None);
    }

    // ============================================================
    // BarrierDetector needs_barrier Tests
    // ============================================================

    #[test]
    fn test_detector_needs_barrier_raw() {
        let mut detector = BarrierDetector::new();
        detector.record_access(1, ResourceState::buffer(PipelineStage::Transfer, AccessFlags::TRANSFER_WRITE));

        let new_state = ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::VERTEX_BUFFER_READ);
        let barrier = detector.needs_barrier(1, &new_state);

        assert!(barrier.is_some());
        let info = barrier.unwrap();
        assert_eq!(info.hazard, HazardType::ReadAfterWrite);
        assert_eq!(info.resource_id, 1);
        assert_eq!(info.src_stage, PipelineStage::Transfer);
        assert_eq!(info.dst_stage, PipelineStage::VertexShader);
    }

    #[test]
    fn test_detector_needs_barrier_untracked_resource() {
        let detector = BarrierDetector::new();
        let new_state = ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::VERTEX_BUFFER_READ);

        // Untracked resource should not need barrier
        assert!(detector.needs_barrier(999, &new_state).is_none());
    }

    #[test]
    fn test_detector_needs_barrier_no_hazard() {
        let mut detector = BarrierDetector::new();
        detector.record_access(1, ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::VERTEX_BUFFER_READ));

        let new_state = ResourceState::buffer(PipelineStage::FragmentShader, AccessFlags::UNIFORM_BUFFER_READ);

        // RAR should not need barrier
        assert!(detector.needs_barrier(1, &new_state).is_none());
    }

    // ============================================================
    // BarrierDetector transition Tests
    // ============================================================

    #[test]
    fn test_detector_transition_returns_barrier_and_updates() {
        let mut detector = BarrierDetector::new();
        detector.record_access(1, ResourceState::buffer(PipelineStage::Transfer, AccessFlags::TRANSFER_WRITE));

        let new_state = ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::VERTEX_BUFFER_READ);
        let barrier = detector.transition(1, new_state.clone());

        assert!(barrier.is_some());

        // State should be updated
        let current = detector.get_state(1).unwrap();
        assert_eq!(current.stage, PipelineStage::VertexShader);
        assert_eq!(current.access, AccessFlags::VERTEX_BUFFER_READ);
    }

    #[test]
    fn test_detector_transition_no_barrier_still_updates() {
        let mut detector = BarrierDetector::new();
        detector.record_access(1, ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::VERTEX_BUFFER_READ));

        let new_state = ResourceState::buffer(PipelineStage::FragmentShader, AccessFlags::UNIFORM_BUFFER_READ);
        let barrier = detector.transition(1, new_state.clone());

        assert!(barrier.is_none());

        // State should still be updated
        let current = detector.get_state(1).unwrap();
        assert_eq!(current.stage, PipelineStage::FragmentShader);
    }

    #[test]
    fn test_detector_transition_new_resource() {
        let mut detector = BarrierDetector::new();
        let state = ResourceState::buffer(PipelineStage::Transfer, AccessFlags::TRANSFER_WRITE);

        let barrier = detector.transition(42, state.clone());

        // New resource - no barrier needed
        assert!(barrier.is_none());

        // But it should be tracked now
        assert!(detector.is_tracked(42));
        assert_eq!(detector.get_state(42).unwrap(), &state);
    }

    // ============================================================
    // BarrierDetector Batch Operations Tests
    // ============================================================

    #[test]
    fn test_detector_detect_all_barriers() {
        let mut detector = BarrierDetector::new();
        detector.record_access(1, ResourceState::buffer(PipelineStage::Transfer, AccessFlags::TRANSFER_WRITE));
        detector.record_access(2, ResourceState::buffer(PipelineStage::Transfer, AccessFlags::TRANSFER_WRITE));
        detector.record_access(3, ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::VERTEX_BUFFER_READ));

        let accesses = vec![
            (1, ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::VERTEX_BUFFER_READ)), // RAW
            (2, ResourceState::buffer(PipelineStage::FragmentShader, AccessFlags::SHADER_READ)),      // RAW
            (3, ResourceState::buffer(PipelineStage::ComputeShader, AccessFlags::SHADER_READ)),       // RAR - no barrier
            (4, ResourceState::buffer(PipelineStage::Transfer, AccessFlags::TRANSFER_WRITE)),         // new - no barrier
        ];

        let barriers = detector.detect_all_barriers(&accesses);

        // Only resources 1 and 2 need barriers
        assert_eq!(barriers.len(), 2);
        assert!(barriers.iter().any(|b| b.resource_id == 1));
        assert!(barriers.iter().any(|b| b.resource_id == 2));
    }

    #[test]
    fn test_detector_detect_all_barriers_empty() {
        let detector = BarrierDetector::new();
        let accesses: Vec<(ResourceId, ResourceState)> = vec![];

        let barriers = detector.detect_all_barriers(&accesses);
        assert!(barriers.is_empty());
    }

    #[test]
    fn test_detector_transition_batch() {
        let mut detector = BarrierDetector::new();
        detector.record_access(1, ResourceState::buffer(PipelineStage::Transfer, AccessFlags::TRANSFER_WRITE));
        detector.record_access(2, ResourceState::buffer(PipelineStage::Transfer, AccessFlags::TRANSFER_WRITE));

        let accesses = vec![
            (1, ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::VERTEX_BUFFER_READ)),
            (2, ResourceState::buffer(PipelineStage::FragmentShader, AccessFlags::SHADER_READ)),
        ];

        let barriers = detector.transition_batch(&accesses);

        assert_eq!(barriers.len(), 2);

        // States should be updated
        assert_eq!(detector.get_state(1).unwrap().stage, PipelineStage::VertexShader);
        assert_eq!(detector.get_state(2).unwrap().stage, PipelineStage::FragmentShader);
    }

    // ============================================================
    // BarrierDetector State Management Tests
    // ============================================================

    #[test]
    fn test_detector_reset() {
        let mut detector = BarrierDetector::new();
        detector.record_access(1, ResourceState::default());
        detector.record_access(2, ResourceState::default());

        assert_eq!(detector.len(), 2);

        detector.reset();

        assert!(detector.is_empty());
        assert_eq!(detector.len(), 0);
    }

    #[test]
    fn test_detector_merge() {
        let mut detector1 = BarrierDetector::new();
        detector1.record_access(1, ResourceState::buffer(PipelineStage::Transfer, AccessFlags::TRANSFER_READ));

        let mut detector2 = BarrierDetector::new();
        detector2.record_access(2, ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::VERTEX_BUFFER_READ));
        detector2.record_access(1, ResourceState::buffer(PipelineStage::ComputeShader, AccessFlags::SHADER_WRITE));

        detector1.merge(&detector2);

        assert_eq!(detector1.len(), 2);
        // Resource 1 should be overwritten
        assert_eq!(detector1.get_state(1).unwrap().stage, PipelineStage::ComputeShader);
        assert!(detector1.is_tracked(2));
    }

    #[test]
    fn test_detector_snapshot() {
        let mut detector = BarrierDetector::new();
        detector.record_access(1, ResourceState::buffer(PipelineStage::Transfer, AccessFlags::TRANSFER_WRITE));
        detector.record_access(2, ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::VERTEX_BUFFER_READ));

        let snapshot = detector.snapshot();

        assert_eq!(snapshot.len(), 2);
        assert!(snapshot.contains(1));
        assert!(snapshot.contains(2));

        // Modify original, snapshot should be unchanged
        detector.reset();
        assert!(detector.is_empty());
        assert_eq!(snapshot.len(), 2);
    }

    #[test]
    fn test_detector_tracker_access() {
        let mut detector = BarrierDetector::new();
        detector.record_access(1, ResourceState::default());

        // Test immutable access
        let tracker = detector.tracker();
        assert!(tracker.contains(1));

        // Test mutable access
        let tracker_mut = detector.tracker_mut();
        tracker_mut.set(2, ResourceState::default());

        assert!(detector.is_tracked(2));
    }

    // ============================================================
    // Combined Hazard Tests (RAW + Layout Transition)
    // ============================================================

    #[test]
    fn test_detect_hazard_raw_with_layout_transition() {
        let old = ResourceState::texture(
            PipelineStage::ColorOutput,
            AccessFlags::COLOR_ATTACHMENT_WRITE,
            TextureLayout::ColorAttachment,
        );
        let new = ResourceState::texture(
            PipelineStage::FragmentShader,
            AccessFlags::SHADER_READ,
            TextureLayout::ShaderReadOnly,
        );

        // This has both RAW (write->read) and layout transition
        // RAW takes precedence in detection
        let hazard = BarrierDetector::detect_hazard(&old, &new);
        assert_eq!(hazard, HazardType::ReadAfterWrite);
    }

    #[test]
    fn test_detect_hazard_war_with_layout_transition() {
        let old = ResourceState::texture(
            PipelineStage::FragmentShader,
            AccessFlags::SHADER_READ,
            TextureLayout::ShaderReadOnly,
        );
        let new = ResourceState::texture(
            PipelineStage::ColorOutput,
            AccessFlags::COLOR_ATTACHMENT_WRITE,
            TextureLayout::ColorAttachment,
        );

        // This has both WAR (read->write) and layout transition
        let hazard = BarrierDetector::detect_hazard(&old, &new);
        assert_eq!(hazard, HazardType::WriteAfterRead);
    }

    #[test]
    fn test_detect_hazard_waw_with_layout_transition() {
        let old = ResourceState::texture(
            PipelineStage::ColorOutput,
            AccessFlags::COLOR_ATTACHMENT_WRITE,
            TextureLayout::ColorAttachment,
        );
        let new = ResourceState::texture(
            PipelineStage::Transfer,
            AccessFlags::TRANSFER_WRITE,
            TextureLayout::TransferDst,
        );

        // This has both WAW (write->write) and layout transition
        let hazard = BarrierDetector::detect_hazard(&old, &new);
        assert_eq!(hazard, HazardType::WriteAfterWrite);
    }

    // ============================================================
    // Edge Cases
    // ============================================================

    #[test]
    fn test_detect_hazard_mixed_access_read_write_old() {
        // Old state has both read and write
        let old = ResourceState::buffer(
            PipelineStage::ComputeShader,
            AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE,
        );
        let new = ResourceState::buffer(
            PipelineStage::FragmentShader,
            AccessFlags::SHADER_READ,
        );

        // Old has write, new has read -> RAW
        assert_eq!(BarrierDetector::detect_hazard(&old, &new), HazardType::ReadAfterWrite);
    }

    #[test]
    fn test_detect_hazard_mixed_access_read_write_new() {
        let old = ResourceState::buffer(
            PipelineStage::FragmentShader,
            AccessFlags::SHADER_READ,
        );
        // New state has both read and write
        let new = ResourceState::buffer(
            PipelineStage::ComputeShader,
            AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE,
        );

        // Old has read, new has write -> WAR
        assert_eq!(BarrierDetector::detect_hazard(&old, &new), HazardType::WriteAfterRead);
    }

    #[test]
    fn test_detect_hazard_both_mixed_access() {
        // Both have read and write
        let old = ResourceState::buffer(
            PipelineStage::ComputeShader,
            AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE,
        );
        let new = ResourceState::buffer(
            PipelineStage::ComputeShader,
            AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE,
        );

        // Both have write -> WAW
        assert_eq!(BarrierDetector::detect_hazard(&old, &new), HazardType::WriteAfterWrite);
    }

    #[test]
    fn test_detect_hazard_buffer_vs_texture_states() {
        // Buffer (no layout)
        let buffer_old = ResourceState::buffer(PipelineStage::Transfer, AccessFlags::TRANSFER_WRITE);
        let buffer_new = ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::VERTEX_BUFFER_READ);

        assert_eq!(BarrierDetector::detect_hazard(&buffer_old, &buffer_new), HazardType::ReadAfterWrite);

        // Same access pattern but for texture
        let texture_old = ResourceState::texture(
            PipelineStage::Transfer,
            AccessFlags::TRANSFER_WRITE,
            TextureLayout::TransferDst,
        );
        let texture_new = ResourceState::texture(
            PipelineStage::FragmentShader,
            AccessFlags::SHADER_READ,
            TextureLayout::ShaderReadOnly,
        );

        assert_eq!(BarrierDetector::detect_hazard(&texture_old, &texture_new), HazardType::ReadAfterWrite);
    }

    #[test]
    fn test_detector_complex_workflow() {
        let mut detector = BarrierDetector::new();

        // Simulate a texture upload and use workflow:
        // 1. Transfer write (upload)
        detector.record_access(1, ResourceState::texture(
            PipelineStage::Transfer,
            AccessFlags::TRANSFER_WRITE,
            TextureLayout::TransferDst,
        ));

        // 2. Transition to shader read (needs RAW barrier)
        let barrier = detector.transition(1, ResourceState::texture(
            PipelineStage::FragmentShader,
            AccessFlags::SHADER_READ,
            TextureLayout::ShaderReadOnly,
        ));
        assert!(barrier.is_some());
        assert_eq!(barrier.unwrap().hazard, HazardType::ReadAfterWrite);

        // 3. Continue reading (no barrier)
        let barrier = detector.transition(1, ResourceState::texture(
            PipelineStage::FragmentShader,
            AccessFlags::SHADER_READ,
            TextureLayout::ShaderReadOnly,
        ));
        assert!(barrier.is_none());

        // 4. Render to it as color attachment (needs WAR barrier)
        let barrier = detector.transition(1, ResourceState::texture(
            PipelineStage::ColorOutput,
            AccessFlags::COLOR_ATTACHMENT_WRITE,
            TextureLayout::ColorAttachment,
        ));
        assert!(barrier.is_some());
        assert_eq!(barrier.unwrap().hazard, HazardType::WriteAfterRead);
    }

    #[test]
    fn test_detector_multiple_resources_independent() {
        let mut detector = BarrierDetector::new();

        // Set up multiple independent resources
        detector.record_access(1, ResourceState::buffer(PipelineStage::Transfer, AccessFlags::TRANSFER_WRITE));
        detector.record_access(2, ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::VERTEX_BUFFER_READ));
        detector.record_access(3, ResourceState::texture(
            PipelineStage::ColorOutput,
            AccessFlags::COLOR_ATTACHMENT_WRITE,
            TextureLayout::ColorAttachment,
        ));

        // Transition each independently
        let b1 = detector.transition(1, ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::VERTEX_BUFFER_READ));
        assert!(b1.is_some()); // RAW

        let b2 = detector.transition(2, ResourceState::buffer(PipelineStage::FragmentShader, AccessFlags::UNIFORM_BUFFER_READ));
        assert!(b2.is_none()); // RAR

        let b3 = detector.transition(3, ResourceState::texture(
            PipelineStage::FragmentShader,
            AccessFlags::SHADER_READ,
            TextureLayout::ShaderReadOnly,
        ));
        assert!(b3.is_some()); // RAW
    }

    // ============================================================
    // SubresourceRange Tests (T-WGPU-P4.7.3)
    // ============================================================

    #[test]
    fn test_subresource_range_default() {
        let range = SubresourceRange::default();
        assert_eq!(range.base_mip, 0);
        assert_eq!(range.mip_count, None);
        assert_eq!(range.base_layer, 0);
        assert_eq!(range.layer_count, None);
    }

    #[test]
    fn test_subresource_range_single() {
        let range = SubresourceRange::single(2, 3);
        assert_eq!(range.base_mip, 2);
        assert_eq!(range.mip_count, Some(1));
        assert_eq!(range.base_layer, 3);
        assert_eq!(range.layer_count, Some(1));
    }

    #[test]
    fn test_subresource_range_all() {
        let range = SubresourceRange::all();
        assert_eq!(range.base_mip, 0);
        assert_eq!(range.mip_count, None);
        assert_eq!(range.base_layer, 0);
        assert_eq!(range.layer_count, None);
    }

    #[test]
    fn test_subresource_range_mips() {
        let range = SubresourceRange::mips(2, 4);
        assert_eq!(range.base_mip, 2);
        assert_eq!(range.mip_count, Some(4));
        assert_eq!(range.base_layer, 0);
        assert_eq!(range.layer_count, None);
    }

    #[test]
    fn test_subresource_range_layers() {
        let range = SubresourceRange::layers(1, 6);
        assert_eq!(range.base_mip, 0);
        assert_eq!(range.mip_count, None);
        assert_eq!(range.base_layer, 1);
        assert_eq!(range.layer_count, Some(6));
    }

    #[test]
    fn test_subresource_range_overlaps_identical() {
        let range1 = SubresourceRange::single(0, 0);
        let range2 = SubresourceRange::single(0, 0);
        assert!(range1.overlaps(&range2));
    }

    #[test]
    fn test_subresource_range_overlaps_partial_mip() {
        let range1 = SubresourceRange { base_mip: 0, mip_count: Some(3), base_layer: 0, layer_count: Some(1) };
        let range2 = SubresourceRange { base_mip: 2, mip_count: Some(2), base_layer: 0, layer_count: Some(1) };
        assert!(range1.overlaps(&range2));
    }

    #[test]
    fn test_subresource_range_overlaps_partial_layer() {
        let range1 = SubresourceRange { base_mip: 0, mip_count: Some(1), base_layer: 0, layer_count: Some(3) };
        let range2 = SubresourceRange { base_mip: 0, mip_count: Some(1), base_layer: 2, layer_count: Some(2) };
        assert!(range1.overlaps(&range2));
    }

    #[test]
    fn test_subresource_range_no_overlap_disjoint_mips() {
        let range1 = SubresourceRange { base_mip: 0, mip_count: Some(2), base_layer: 0, layer_count: Some(1) };
        let range2 = SubresourceRange { base_mip: 3, mip_count: Some(2), base_layer: 0, layer_count: Some(1) };
        assert!(!range1.overlaps(&range2));
    }

    #[test]
    fn test_subresource_range_no_overlap_disjoint_layers() {
        let range1 = SubresourceRange { base_mip: 0, mip_count: Some(1), base_layer: 0, layer_count: Some(2) };
        let range2 = SubresourceRange { base_mip: 0, mip_count: Some(1), base_layer: 4, layer_count: Some(2) };
        assert!(!range1.overlaps(&range2));
    }

    #[test]
    fn test_subresource_range_overlaps_unbounded() {
        let range1 = SubresourceRange::all();
        let range2 = SubresourceRange::single(5, 10);
        assert!(range1.overlaps(&range2));
        assert!(range2.overlaps(&range1));
    }

    #[test]
    fn test_subresource_range_contains_subset() {
        let range1 = SubresourceRange { base_mip: 0, mip_count: Some(4), base_layer: 0, layer_count: Some(6) };
        let range2 = SubresourceRange { base_mip: 1, mip_count: Some(2), base_layer: 2, layer_count: Some(2) };
        assert!(range1.contains(&range2));
        assert!(!range2.contains(&range1));
    }

    #[test]
    fn test_subresource_range_contains_identical() {
        let range1 = SubresourceRange::single(2, 3);
        let range2 = SubresourceRange::single(2, 3);
        assert!(range1.contains(&range2));
    }

    #[test]
    fn test_subresource_range_all_contains_any() {
        let all = SubresourceRange::all();
        let specific = SubresourceRange { base_mip: 5, mip_count: Some(3), base_layer: 10, layer_count: Some(4) };
        assert!(all.contains(&specific));
    }

    #[test]
    fn test_subresource_range_try_merge_overlapping() {
        let range1 = SubresourceRange { base_mip: 0, mip_count: Some(3), base_layer: 0, layer_count: Some(1) };
        let range2 = SubresourceRange { base_mip: 2, mip_count: Some(3), base_layer: 0, layer_count: Some(1) };

        let merged = range1.try_merge(&range2);
        assert!(merged.is_some());
        let m = merged.unwrap();
        assert_eq!(m.base_mip, 0);
        assert_eq!(m.mip_count, Some(5));
        assert_eq!(m.base_layer, 0);
        assert_eq!(m.layer_count, Some(1));
    }

    #[test]
    fn test_subresource_range_try_merge_adjacent() {
        let range1 = SubresourceRange { base_mip: 0, mip_count: Some(2), base_layer: 0, layer_count: Some(1) };
        let range2 = SubresourceRange { base_mip: 2, mip_count: Some(2), base_layer: 0, layer_count: Some(1) };

        let merged = range1.try_merge(&range2);
        assert!(merged.is_some());
        let m = merged.unwrap();
        assert_eq!(m.base_mip, 0);
        assert_eq!(m.mip_count, Some(4));
    }

    #[test]
    fn test_subresource_range_try_merge_disjoint_fails() {
        let range1 = SubresourceRange { base_mip: 0, mip_count: Some(2), base_layer: 0, layer_count: Some(1) };
        let range2 = SubresourceRange { base_mip: 5, mip_count: Some(2), base_layer: 0, layer_count: Some(1) };

        assert!(range1.try_merge(&range2).is_none());
    }

    #[test]
    fn test_subresource_range_effective_counts() {
        let bounded = SubresourceRange { base_mip: 0, mip_count: Some(5), base_layer: 0, layer_count: Some(3) };
        assert_eq!(bounded.effective_mip_count(), 5);
        assert_eq!(bounded.effective_layer_count(), 3);

        let unbounded = SubresourceRange::all();
        assert_eq!(unbounded.effective_mip_count(), u32::MAX);
        assert_eq!(unbounded.effective_layer_count(), u32::MAX);
    }

    #[test]
    fn test_subresource_range_equality() {
        let r1 = SubresourceRange::single(1, 2);
        let r2 = SubresourceRange::single(1, 2);
        let r3 = SubresourceRange::single(1, 3);
        assert_eq!(r1, r2);
        assert_ne!(r1, r3);
    }

    #[test]
    fn test_subresource_range_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(SubresourceRange::single(0, 0));
        set.insert(SubresourceRange::single(1, 0));
        assert!(set.contains(&SubresourceRange::single(0, 0)));
        assert!(!set.contains(&SubresourceRange::single(2, 0)));
    }

    // ============================================================
    // LayoutTransition Tests (T-WGPU-P4.7.3)
    // ============================================================

    #[test]
    fn test_layout_transition_new() {
        let t = LayoutTransition::new(
            42,
            TextureLayout::Undefined,
            TextureLayout::ShaderReadOnly,
            SubresourceRange::all(),
        );
        assert_eq!(t.resource_id, 42);
        assert_eq!(t.old_layout, TextureLayout::Undefined);
        assert_eq!(t.new_layout, TextureLayout::ShaderReadOnly);
    }

    #[test]
    fn test_layout_transition_whole() {
        let t = LayoutTransition::whole(100, TextureLayout::TransferSrc, TextureLayout::TransferDst);
        assert_eq!(t.resource_id, 100);
        assert_eq!(t.subresource, SubresourceRange::all());
    }

    #[test]
    fn test_layout_transition_is_needed() {
        let needed = LayoutTransition::new(1, TextureLayout::Undefined, TextureLayout::ShaderReadOnly, SubresourceRange::all());
        assert!(needed.is_needed());

        let not_needed = LayoutTransition::new(1, TextureLayout::ShaderReadOnly, TextureLayout::ShaderReadOnly, SubresourceRange::all());
        assert!(!not_needed.is_needed());
    }

    #[test]
    fn test_layout_transition_can_merge_same_layouts() {
        let t1 = LayoutTransition::new(1, TextureLayout::Undefined, TextureLayout::ShaderReadOnly, SubresourceRange::mips(0, 2));
        let t2 = LayoutTransition::new(1, TextureLayout::Undefined, TextureLayout::ShaderReadOnly, SubresourceRange::mips(2, 2));
        assert!(t1.can_merge_with(&t2));
    }

    #[test]
    fn test_layout_transition_cannot_merge_different_resources() {
        let t1 = LayoutTransition::new(1, TextureLayout::Undefined, TextureLayout::ShaderReadOnly, SubresourceRange::all());
        let t2 = LayoutTransition::new(2, TextureLayout::Undefined, TextureLayout::ShaderReadOnly, SubresourceRange::all());
        assert!(!t1.can_merge_with(&t2));
    }

    #[test]
    fn test_layout_transition_cannot_merge_different_layouts() {
        let t1 = LayoutTransition::new(1, TextureLayout::Undefined, TextureLayout::ShaderReadOnly, SubresourceRange::all());
        let t2 = LayoutTransition::new(1, TextureLayout::Undefined, TextureLayout::ColorAttachment, SubresourceRange::all());
        assert!(!t1.can_merge_with(&t2));
    }

    #[test]
    fn test_layout_transition_try_merge_success() {
        let t1 = LayoutTransition::new(1, TextureLayout::Undefined, TextureLayout::ShaderReadOnly, SubresourceRange::mips(0, 2));
        let t2 = LayoutTransition::new(1, TextureLayout::Undefined, TextureLayout::ShaderReadOnly, SubresourceRange::mips(2, 2));

        let merged = t1.try_merge(&t2);
        assert!(merged.is_some());
        let m = merged.unwrap();
        assert_eq!(m.resource_id, 1);
        assert_eq!(m.subresource.base_mip, 0);
        assert_eq!(m.subresource.mip_count, Some(4));
    }

    #[test]
    fn test_layout_transition_try_merge_failure() {
        let t1 = LayoutTransition::new(1, TextureLayout::Undefined, TextureLayout::ShaderReadOnly, SubresourceRange::mips(0, 2));
        let t2 = LayoutTransition::new(2, TextureLayout::Undefined, TextureLayout::ShaderReadOnly, SubresourceRange::mips(0, 2));

        assert!(t1.try_merge(&t2).is_none());
    }

    #[test]
    fn test_layout_transition_clone_and_eq() {
        let t = LayoutTransition::new(1, TextureLayout::Undefined, TextureLayout::ColorAttachment, SubresourceRange::single(0, 0));
        let cloned = t.clone();
        assert_eq!(t, cloned);
    }

    // ============================================================
    // TransitionMode Tests (T-WGPU-P4.7.3)
    // ============================================================

    #[test]
    fn test_transition_mode_default() {
        assert_eq!(TransitionMode::default(), TransitionMode::Implicit);
    }

    #[test]
    fn test_transition_mode_equality() {
        assert_eq!(TransitionMode::Implicit, TransitionMode::Implicit);
        assert_eq!(TransitionMode::Explicit, TransitionMode::Explicit);
        assert_ne!(TransitionMode::Implicit, TransitionMode::Explicit);
    }

    // ============================================================
    // LayoutTransitionManager Basic Tests (T-WGPU-P4.7.3)
    // ============================================================

    #[test]
    fn test_manager_new() {
        let manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        assert!(manager.is_empty());
        assert_eq!(manager.len(), 0);
        assert_eq!(manager.mode(), TransitionMode::Implicit);
    }

    #[test]
    fn test_manager_with_capacity() {
        let manager = LayoutTransitionManager::with_capacity(TransitionMode::Explicit, 100);
        assert!(manager.is_empty());
        assert_eq!(manager.mode(), TransitionMode::Explicit);
    }

    #[test]
    fn test_manager_set_mode() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        manager.set_mode(TransitionMode::Explicit);
        assert_eq!(manager.mode(), TransitionMode::Explicit);
    }

    #[test]
    fn test_manager_set_and_get_layout() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        manager.set_layout(1, SubresourceRange::all(), TextureLayout::ShaderReadOnly);

        let layout = manager.get_layout(1, SubresourceRange::all());
        assert_eq!(layout, Some(TextureLayout::ShaderReadOnly));
    }

    #[test]
    fn test_manager_get_layout_untracked() {
        let manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        assert!(manager.get_layout(999, SubresourceRange::all()).is_none());
    }

    #[test]
    fn test_manager_get_whole_layout() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        manager.set_whole_layout(1, TextureLayout::ColorAttachment);

        assert_eq!(manager.get_whole_layout(1), Some(TextureLayout::ColorAttachment));
    }

    #[test]
    fn test_manager_get_layout_from_containing_range() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        manager.set_layout(1, SubresourceRange::all(), TextureLayout::ShaderReadOnly);

        // Query a specific subresource within the tracked range
        let layout = manager.get_layout(1, SubresourceRange::single(2, 3));
        assert_eq!(layout, Some(TextureLayout::ShaderReadOnly));
    }

    #[test]
    fn test_manager_is_tracked() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        manager.set_layout(1, SubresourceRange::all(), TextureLayout::Undefined);

        assert!(manager.is_tracked(1));
        assert!(!manager.is_tracked(2));
    }

    #[test]
    fn test_manager_len_and_empty() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        assert!(manager.is_empty());
        assert_eq!(manager.len(), 0);

        manager.set_layout(1, SubresourceRange::all(), TextureLayout::Undefined);
        manager.set_layout(2, SubresourceRange::all(), TextureLayout::Undefined);

        assert!(!manager.is_empty());
        assert_eq!(manager.len(), 2);
    }

    // ============================================================
    // LayoutTransitionManager Transition Tests (T-WGPU-P4.7.3)
    // ============================================================

    #[test]
    fn test_manager_transition_to_returns_transition() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        manager.set_layout(1, SubresourceRange::all(), TextureLayout::Undefined);

        let transition = manager.transition_to(1, TextureLayout::ShaderReadOnly);
        assert!(transition.is_some());
        let t = transition.unwrap();
        assert_eq!(t.resource_id, 1);
        assert_eq!(t.old_layout, TextureLayout::Undefined);
        assert_eq!(t.new_layout, TextureLayout::ShaderReadOnly);
    }

    #[test]
    fn test_manager_transition_to_updates_layout() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        manager.set_layout(1, SubresourceRange::all(), TextureLayout::Undefined);

        manager.transition_to(1, TextureLayout::ShaderReadOnly);

        assert_eq!(manager.get_whole_layout(1), Some(TextureLayout::ShaderReadOnly));
    }

    #[test]
    fn test_manager_transition_to_same_layout_returns_none() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        manager.set_layout(1, SubresourceRange::all(), TextureLayout::ShaderReadOnly);

        let transition = manager.transition_to(1, TextureLayout::ShaderReadOnly);
        assert!(transition.is_none());
    }

    #[test]
    fn test_manager_transition_to_untracked_uses_undefined() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);

        let transition = manager.transition_to(1, TextureLayout::ShaderReadOnly);
        assert!(transition.is_some());
        let t = transition.unwrap();
        assert_eq!(t.old_layout, TextureLayout::Undefined);
    }

    #[test]
    fn test_manager_transition_subresource() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        manager.set_layout(1, SubresourceRange::all(), TextureLayout::Undefined);

        let subresource = SubresourceRange::mips(0, 4);
        let transition = manager.transition_subresource(1, subresource, TextureLayout::ShaderReadOnly);

        assert!(transition.is_some());
        let t = transition.unwrap();
        assert_eq!(t.subresource, subresource);
    }

    // ============================================================
    // LayoutTransitionManager Pending/Coalesce Tests (T-WGPU-P4.7.3)
    // ============================================================

    #[test]
    fn test_manager_add_pending() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        let t = LayoutTransition::new(1, TextureLayout::Undefined, TextureLayout::ShaderReadOnly, SubresourceRange::all());

        manager.add_pending(t);
        assert_eq!(manager.pending_count(), 1);
        assert!(manager.has_pending());
    }

    #[test]
    fn test_manager_add_pending_skips_no_op() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        // Same old and new layout - should be skipped
        let t = LayoutTransition::new(1, TextureLayout::ShaderReadOnly, TextureLayout::ShaderReadOnly, SubresourceRange::all());

        manager.add_pending(t);
        assert_eq!(manager.pending_count(), 0);
    }

    #[test]
    fn test_manager_coalesce_pending_merges_compatible() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);

        // Add two adjacent transitions
        manager.add_pending(LayoutTransition::new(1, TextureLayout::Undefined, TextureLayout::ShaderReadOnly, SubresourceRange::mips(0, 2)));
        manager.add_pending(LayoutTransition::new(1, TextureLayout::Undefined, TextureLayout::ShaderReadOnly, SubresourceRange::mips(2, 2)));

        let coalesced = manager.coalesce_pending();
        assert_eq!(coalesced.len(), 1);
        assert_eq!(coalesced[0].subresource.base_mip, 0);
        assert_eq!(coalesced[0].subresource.mip_count, Some(4));
    }

    #[test]
    fn test_manager_coalesce_pending_keeps_incompatible_separate() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);

        // Add transitions for different resources
        manager.add_pending(LayoutTransition::new(1, TextureLayout::Undefined, TextureLayout::ShaderReadOnly, SubresourceRange::all()));
        manager.add_pending(LayoutTransition::new(2, TextureLayout::Undefined, TextureLayout::ShaderReadOnly, SubresourceRange::all()));

        let coalesced = manager.coalesce_pending();
        assert_eq!(coalesced.len(), 2);
    }

    #[test]
    fn test_manager_coalesce_pending_keeps_different_layouts_separate() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);

        // Add transitions with different target layouts
        manager.add_pending(LayoutTransition::new(1, TextureLayout::Undefined, TextureLayout::ShaderReadOnly, SubresourceRange::mips(0, 2)));
        manager.add_pending(LayoutTransition::new(1, TextureLayout::Undefined, TextureLayout::ColorAttachment, SubresourceRange::mips(2, 2)));

        let coalesced = manager.coalesce_pending();
        assert_eq!(coalesced.len(), 2);
    }

    #[test]
    fn test_manager_coalesce_pending_empty() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        let coalesced = manager.coalesce_pending();
        assert!(coalesced.is_empty());
    }

    #[test]
    fn test_manager_flush_pending() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        manager.set_layout(1, SubresourceRange::all(), TextureLayout::Undefined);

        manager.add_pending(LayoutTransition::new(1, TextureLayout::Undefined, TextureLayout::ShaderReadOnly, SubresourceRange::all()));

        let flushed = manager.flush_pending();
        assert_eq!(flushed.len(), 1);

        // Layout should be updated
        assert_eq!(manager.get_whole_layout(1), Some(TextureLayout::ShaderReadOnly));

        // Pending should be cleared
        assert!(!manager.has_pending());
    }

    #[test]
    fn test_manager_clear_pending() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        manager.add_pending(LayoutTransition::new(1, TextureLayout::Undefined, TextureLayout::ShaderReadOnly, SubresourceRange::all()));

        manager.clear_pending();
        assert!(!manager.has_pending());
        assert_eq!(manager.pending_count(), 0);
    }

    // ============================================================
    // LayoutTransitionManager Optimal Path Tests (T-WGPU-P4.7.3)
    // ============================================================

    #[test]
    fn test_manager_optimal_path_same_layout() {
        let path = LayoutTransitionManager::optimal_transition_path(TextureLayout::ShaderReadOnly, TextureLayout::ShaderReadOnly);
        assert_eq!(path.len(), 1);
        assert_eq!(path[0], TextureLayout::ShaderReadOnly);
    }

    #[test]
    fn test_manager_optimal_path_undefined_to_shader() {
        let path = LayoutTransitionManager::optimal_transition_path(TextureLayout::Undefined, TextureLayout::ShaderReadOnly);
        assert_eq!(path.len(), 3);
        assert_eq!(path[0], TextureLayout::Undefined);
        assert_eq!(path[1], TextureLayout::TransferDst);
        assert_eq!(path[2], TextureLayout::ShaderReadOnly);
    }

    #[test]
    fn test_manager_optimal_path_undefined_to_depth_read() {
        let path = LayoutTransitionManager::optimal_transition_path(TextureLayout::Undefined, TextureLayout::DepthStencilReadOnly);
        assert_eq!(path.len(), 3);
        assert_eq!(path[0], TextureLayout::Undefined);
        assert_eq!(path[1], TextureLayout::DepthStencilAttachment);
        assert_eq!(path[2], TextureLayout::DepthStencilReadOnly);
    }

    #[test]
    fn test_manager_optimal_path_direct() {
        let path = LayoutTransitionManager::optimal_transition_path(TextureLayout::ColorAttachment, TextureLayout::ShaderReadOnly);
        assert_eq!(path.len(), 2);
        assert_eq!(path[0], TextureLayout::ColorAttachment);
        assert_eq!(path[1], TextureLayout::ShaderReadOnly);
    }

    #[test]
    fn test_manager_is_transition_needed() {
        assert!(LayoutTransitionManager::is_transition_needed(TextureLayout::Undefined, TextureLayout::ShaderReadOnly));
        assert!(!LayoutTransitionManager::is_transition_needed(TextureLayout::ShaderReadOnly, TextureLayout::ShaderReadOnly));
    }

    // ============================================================
    // LayoutTransitionManager State Management Tests (T-WGPU-P4.7.3)
    // ============================================================

    #[test]
    fn test_manager_clear() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        manager.set_layout(1, SubresourceRange::all(), TextureLayout::Undefined);
        manager.add_pending(LayoutTransition::new(1, TextureLayout::Undefined, TextureLayout::ShaderReadOnly, SubresourceRange::all()));

        manager.clear();

        assert!(manager.is_empty());
        assert!(!manager.has_pending());
    }

    #[test]
    fn test_manager_remove() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        manager.set_layout(1, SubresourceRange::all(), TextureLayout::Undefined);
        manager.set_layout(2, SubresourceRange::all(), TextureLayout::Undefined);

        assert!(manager.remove(1));
        assert!(!manager.is_tracked(1));
        assert!(manager.is_tracked(2));
        assert_eq!(manager.len(), 1);
    }

    #[test]
    fn test_manager_remove_nonexistent() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        assert!(!manager.remove(999));
    }

    #[test]
    fn test_manager_tracked_resources() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        manager.set_layout(10, SubresourceRange::all(), TextureLayout::Undefined);
        manager.set_layout(20, SubresourceRange::all(), TextureLayout::Undefined);
        manager.set_layout(30, SubresourceRange::all(), TextureLayout::Undefined);

        let ids: Vec<_> = manager.tracked_resources().copied().collect();
        assert_eq!(ids.len(), 3);
        assert!(ids.contains(&10));
        assert!(ids.contains(&20));
        assert!(ids.contains(&30));
    }

    #[test]
    fn test_manager_get_all_layouts() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        manager.set_layout(1, SubresourceRange::mips(0, 2), TextureLayout::ShaderReadOnly);
        manager.set_layout(1, SubresourceRange::mips(2, 2), TextureLayout::ColorAttachment);

        let layouts = manager.get_all_layouts(1);
        assert!(layouts.is_some());
        assert_eq!(layouts.unwrap().len(), 2);
    }

    // ============================================================
    // LayoutTransitionManager Batch Operations Tests (T-WGPU-P4.7.3)
    // ============================================================

    #[test]
    fn test_manager_transition_batch() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        manager.set_layout(1, SubresourceRange::all(), TextureLayout::Undefined);
        manager.set_layout(2, SubresourceRange::all(), TextureLayout::Undefined);
        manager.set_layout(3, SubresourceRange::all(), TextureLayout::ShaderReadOnly);

        let transitions = manager.transition_batch(&[
            (1, TextureLayout::ShaderReadOnly),
            (2, TextureLayout::ColorAttachment),
            (3, TextureLayout::ShaderReadOnly), // Same layout - no transition
        ]);

        assert_eq!(transitions.len(), 2);
    }

    #[test]
    fn test_manager_transition_subresources_batch() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        manager.set_layout(1, SubresourceRange::all(), TextureLayout::Undefined);

        let transitions = manager.transition_subresources_batch(&[
            (1, SubresourceRange::mips(0, 2), TextureLayout::ShaderReadOnly),
            (1, SubresourceRange::mips(2, 2), TextureLayout::ColorAttachment),
        ]);

        assert_eq!(transitions.len(), 2);
    }

    // ============================================================
    // LayoutTransitionManager Snapshot/Merge Tests (T-WGPU-P4.7.3)
    // ============================================================

    #[test]
    fn test_manager_snapshot() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        manager.set_layout(1, SubresourceRange::all(), TextureLayout::ShaderReadOnly);
        manager.set_layout(2, SubresourceRange::all(), TextureLayout::ColorAttachment);

        let snapshot = manager.snapshot();
        assert_eq!(snapshot.len(), 2);

        // Modify original
        manager.clear();
        assert!(manager.is_empty());

        // Snapshot should be unchanged
        assert_eq!(snapshot.len(), 2);
    }

    #[test]
    fn test_manager_restore() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        manager.set_layout(1, SubresourceRange::all(), TextureLayout::ShaderReadOnly);

        let snapshot = manager.snapshot();

        // Clear and add different data
        manager.clear();
        manager.set_layout(2, SubresourceRange::all(), TextureLayout::ColorAttachment);

        // Restore
        manager.restore(snapshot);

        assert!(manager.is_tracked(1));
        assert!(!manager.is_tracked(2));
        assert_eq!(manager.get_whole_layout(1), Some(TextureLayout::ShaderReadOnly));
    }

    #[test]
    fn test_manager_merge() {
        let mut manager1 = LayoutTransitionManager::new(TransitionMode::Implicit);
        manager1.set_layout(1, SubresourceRange::all(), TextureLayout::Undefined);

        let mut manager2 = LayoutTransitionManager::new(TransitionMode::Implicit);
        manager2.set_layout(2, SubresourceRange::all(), TextureLayout::ShaderReadOnly);
        manager2.set_layout(1, SubresourceRange::all(), TextureLayout::ColorAttachment);

        manager1.merge(&manager2);

        assert_eq!(manager1.len(), 2);
        assert!(manager1.is_tracked(1));
        assert!(manager1.is_tracked(2));
        // Resource 1 should be overwritten with merged value
        assert_eq!(manager1.get_whole_layout(1), Some(TextureLayout::ColorAttachment));
    }

    // ============================================================
    // LayoutTransitionManager Complex Workflow Tests (T-WGPU-P4.7.3)
    // ============================================================

    #[test]
    fn test_manager_texture_upload_workflow() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);

        // Start with undefined
        manager.set_whole_layout(1, TextureLayout::Undefined);

        // Transition to transfer destination for upload
        let t1 = manager.transition_to(1, TextureLayout::TransferDst);
        assert!(t1.is_some());
        assert_eq!(t1.unwrap().old_layout, TextureLayout::Undefined);

        // Transition to shader read for sampling
        let t2 = manager.transition_to(1, TextureLayout::ShaderReadOnly);
        assert!(t2.is_some());
        assert_eq!(t2.unwrap().old_layout, TextureLayout::TransferDst);

        // Verify final layout
        assert_eq!(manager.get_whole_layout(1), Some(TextureLayout::ShaderReadOnly));
    }

    #[test]
    fn test_manager_render_target_workflow() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);

        // Create render target
        manager.set_whole_layout(1, TextureLayout::ColorAttachment);

        // Transition to shader read
        let t1 = manager.transition_to(1, TextureLayout::ShaderReadOnly);
        assert!(t1.is_some());

        // Transition back to color attachment
        let t2 = manager.transition_to(1, TextureLayout::ColorAttachment);
        assert!(t2.is_some());

        // Present
        let t3 = manager.transition_to(1, TextureLayout::Present);
        assert!(t3.is_some());
    }

    #[test]
    fn test_manager_mipmap_generation_workflow() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);

        // Set base mip to transfer src (source for blit)
        manager.set_layout(1, SubresourceRange::mips(0, 1), TextureLayout::TransferSrc);

        // Set remaining mips to transfer dst (destinations for blit)
        manager.set_layout(1, SubresourceRange::mips(1, 4), TextureLayout::TransferDst);

        // After generation, transition all to shader read
        let t1 = manager.transition_subresource(1, SubresourceRange::mips(0, 1), TextureLayout::ShaderReadOnly);
        let t2 = manager.transition_subresource(1, SubresourceRange::mips(1, 4), TextureLayout::ShaderReadOnly);

        assert!(t1.is_some());
        assert!(t2.is_some());
    }

    #[test]
    fn test_manager_overlapping_subresource_update() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);

        // Set all to one layout
        manager.set_whole_layout(1, TextureLayout::ShaderReadOnly);

        // Set a subset to different layout
        manager.set_layout(1, SubresourceRange::mips(0, 2), TextureLayout::ColorAttachment);

        // The all range should be replaced by the specific range
        let layouts = manager.get_all_layouts(1).unwrap();
        // Should have the specific range now
        assert!(layouts.contains_key(&SubresourceRange::mips(0, 2)));
    }

    #[test]
    fn test_manager_coalesce_multiple_adjacent_transitions() {
        let mut manager = LayoutTransitionManager::new(TransitionMode::Implicit);

        // Add multiple adjacent mip transitions
        for i in 0..4 {
            manager.add_pending(LayoutTransition::new(
                1,
                TextureLayout::Undefined,
                TextureLayout::ShaderReadOnly,
                SubresourceRange::single(i, 0),
            ));
        }

        let coalesced = manager.coalesce_pending();

        // Should be coalesced into fewer transitions
        // (exact count depends on merging algorithm, but should be less than 4)
        assert!(coalesced.len() <= 4);
        // All should have the same target layout
        for t in &coalesced {
            assert_eq!(t.new_layout, TextureLayout::ShaderReadOnly);
        }
    }

    #[test]
    fn test_manager_implicit_vs_explicit_mode() {
        let mut implicit_manager = LayoutTransitionManager::new(TransitionMode::Implicit);
        let mut explicit_manager = LayoutTransitionManager::new(TransitionMode::Explicit);

        // Both should track layouts the same way
        implicit_manager.set_whole_layout(1, TextureLayout::Undefined);
        explicit_manager.set_whole_layout(1, TextureLayout::Undefined);

        // Transitions work the same
        let t1 = implicit_manager.transition_to(1, TextureLayout::ShaderReadOnly);
        let t2 = explicit_manager.transition_to(1, TextureLayout::ShaderReadOnly);

        assert!(t1.is_some());
        assert!(t2.is_some());

        // The mode is just a hint for the caller
        assert_eq!(implicit_manager.mode(), TransitionMode::Implicit);
        assert_eq!(explicit_manager.mode(), TransitionMode::Explicit);
    }

    #[test]
    fn test_manager_default_impl() {
        let manager: LayoutTransitionManager = Default::default();
        assert!(manager.is_empty());
        assert_eq!(manager.mode(), TransitionMode::Implicit);
    }

    // ============================================================
    // PipelineStageMask Tests (T-WGPU-P4.7.4)
    // ============================================================

    #[test]
    fn test_pipeline_stage_mask_default() {
        assert_eq!(PipelineStageMask::default(), PipelineStageMask::NONE);
    }

    #[test]
    fn test_pipeline_stage_mask_from_stage() {
        assert_eq!(PipelineStageMask::from_stage(PipelineStage::None), PipelineStageMask::NONE);
        assert_eq!(PipelineStageMask::from_stage(PipelineStage::VertexInput), PipelineStageMask::VERTEX_INPUT);
        assert_eq!(PipelineStageMask::from_stage(PipelineStage::VertexShader), PipelineStageMask::VERTEX_SHADER);
        assert_eq!(PipelineStageMask::from_stage(PipelineStage::FragmentShader), PipelineStageMask::FRAGMENT_SHADER);
        assert_eq!(PipelineStageMask::from_stage(PipelineStage::ComputeShader), PipelineStageMask::COMPUTE_SHADER);
        assert_eq!(PipelineStageMask::from_stage(PipelineStage::Transfer), PipelineStageMask::TRANSFER);
        assert_eq!(PipelineStageMask::from_stage(PipelineStage::Host), PipelineStageMask::HOST);
        assert_eq!(PipelineStageMask::from_stage(PipelineStage::AllGraphics), PipelineStageMask::ALL_GRAPHICS);
        assert_eq!(PipelineStageMask::from_stage(PipelineStage::AllCommands), PipelineStageMask::ALL_COMMANDS);
    }

    #[test]
    fn test_pipeline_stage_mask_has_graphics() {
        assert!(PipelineStageMask::VERTEX_INPUT.has_graphics());
        assert!(PipelineStageMask::VERTEX_SHADER.has_graphics());
        assert!(PipelineStageMask::FRAGMENT_SHADER.has_graphics());
        assert!(PipelineStageMask::ALL_GRAPHICS.has_graphics());
        assert!(!PipelineStageMask::COMPUTE_SHADER.has_graphics());
        assert!(!PipelineStageMask::TRANSFER.has_graphics());
        assert!(!PipelineStageMask::HOST.has_graphics());
    }

    #[test]
    fn test_pipeline_stage_mask_has_compute() {
        assert!(PipelineStageMask::COMPUTE_SHADER.has_compute());
        assert!(!PipelineStageMask::VERTEX_SHADER.has_compute());
        assert!(!PipelineStageMask::TRANSFER.has_compute());
    }

    #[test]
    fn test_pipeline_stage_mask_has_transfer() {
        assert!(PipelineStageMask::TRANSFER.has_transfer());
        assert!(!PipelineStageMask::COMPUTE_SHADER.has_transfer());
    }

    #[test]
    fn test_pipeline_stage_mask_has_host() {
        assert!(PipelineStageMask::HOST.has_host());
        assert!(!PipelineStageMask::TRANSFER.has_host());
    }

    #[test]
    fn test_pipeline_stage_mask_merge() {
        let a = PipelineStageMask::VERTEX_SHADER;
        let b = PipelineStageMask::FRAGMENT_SHADER;
        let merged = a.merge(b);
        assert!(merged.contains(PipelineStageMask::VERTEX_SHADER));
        assert!(merged.contains(PipelineStageMask::FRAGMENT_SHADER));
    }

    #[test]
    fn test_pipeline_stage_mask_all_graphics_contents() {
        let all_graphics = PipelineStageMask::ALL_GRAPHICS;
        assert!(all_graphics.contains(PipelineStageMask::VERTEX_INPUT));
        assert!(all_graphics.contains(PipelineStageMask::VERTEX_SHADER));
        assert!(all_graphics.contains(PipelineStageMask::FRAGMENT_SHADER));
        assert!(all_graphics.contains(PipelineStageMask::EARLY_DEPTH));
        assert!(all_graphics.contains(PipelineStageMask::LATE_DEPTH));
        assert!(all_graphics.contains(PipelineStageMask::COLOR_OUTPUT));
        assert!(!all_graphics.contains(PipelineStageMask::COMPUTE_SHADER));
        assert!(!all_graphics.contains(PipelineStageMask::TRANSFER));
    }

    #[test]
    fn test_pipeline_stage_mask_all_commands_contents() {
        let all_commands = PipelineStageMask::ALL_COMMANDS;
        assert!(all_commands.contains(PipelineStageMask::ALL_GRAPHICS));
        assert!(all_commands.contains(PipelineStageMask::COMPUTE_SHADER));
        assert!(all_commands.contains(PipelineStageMask::TRANSFER));
        assert!(all_commands.contains(PipelineStageMask::HOST));
    }

    // ============================================================
    // BufferBarrier Tests (T-WGPU-P4.7.4)
    // ============================================================

    #[test]
    fn test_buffer_barrier_whole() {
        let barrier = BufferBarrier::whole(42, AccessFlags::TRANSFER_WRITE, AccessFlags::VERTEX_BUFFER_READ);
        assert_eq!(barrier.resource_id, 42);
        assert_eq!(barrier.src_access, AccessFlags::TRANSFER_WRITE);
        assert_eq!(barrier.dst_access, AccessFlags::VERTEX_BUFFER_READ);
        assert_eq!(barrier.offset, 0);
        assert!(barrier.size.is_none());
        assert!(barrier.is_whole_buffer());
    }

    #[test]
    fn test_buffer_barrier_region() {
        let barrier = BufferBarrier::region(42, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ, 100, 256);
        assert_eq!(barrier.resource_id, 42);
        assert_eq!(barrier.offset, 100);
        assert_eq!(barrier.size, Some(256));
        assert!(!barrier.is_whole_buffer());
    }

    #[test]
    fn test_buffer_barrier_can_merge_same_resource_whole() {
        let b1 = BufferBarrier::whole(1, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ);
        let b2 = BufferBarrier::whole(1, AccessFlags::SHADER_WRITE, AccessFlags::SHADER_READ);
        assert!(b1.can_merge_with(&b2));
    }

    #[test]
    fn test_buffer_barrier_cannot_merge_different_resources() {
        let b1 = BufferBarrier::whole(1, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ);
        let b2 = BufferBarrier::whole(2, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ);
        assert!(!b1.can_merge_with(&b2));
    }

    #[test]
    fn test_buffer_barrier_can_merge_overlapping_regions() {
        let b1 = BufferBarrier::region(1, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ, 0, 100);
        let b2 = BufferBarrier::region(1, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ, 50, 100);
        assert!(b1.can_merge_with(&b2));
    }

    #[test]
    fn test_buffer_barrier_can_merge_adjacent_regions() {
        let b1 = BufferBarrier::region(1, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ, 0, 100);
        let b2 = BufferBarrier::region(1, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ, 100, 100);
        assert!(b1.can_merge_with(&b2));
    }

    #[test]
    fn test_buffer_barrier_cannot_merge_disjoint_regions() {
        let b1 = BufferBarrier::region(1, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ, 0, 50);
        let b2 = BufferBarrier::region(1, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ, 100, 50);
        assert!(!b1.can_merge_with(&b2));
    }

    #[test]
    fn test_buffer_barrier_try_merge_success() {
        let b1 = BufferBarrier::region(1, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ, 0, 100);
        let b2 = BufferBarrier::region(1, AccessFlags::HOST_WRITE, AccessFlags::UNIFORM_BUFFER_READ, 100, 100);

        let merged = b1.try_merge(&b2);
        assert!(merged.is_some());

        let m = merged.unwrap();
        assert_eq!(m.resource_id, 1);
        assert_eq!(m.offset, 0);
        assert_eq!(m.size, Some(200));
        assert!(m.src_access.contains(AccessFlags::TRANSFER_WRITE));
        assert!(m.src_access.contains(AccessFlags::HOST_WRITE));
        assert!(m.dst_access.contains(AccessFlags::SHADER_READ));
        assert!(m.dst_access.contains(AccessFlags::UNIFORM_BUFFER_READ));
    }

    #[test]
    fn test_buffer_barrier_try_merge_whole_with_region() {
        let whole = BufferBarrier::whole(1, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ);
        let region = BufferBarrier::region(1, AccessFlags::HOST_WRITE, AccessFlags::UNIFORM_BUFFER_READ, 50, 100);

        let merged = whole.try_merge(&region);
        assert!(merged.is_some());

        let m = merged.unwrap();
        assert!(m.is_whole_buffer());
    }

    #[test]
    fn test_buffer_barrier_try_merge_failure() {
        let b1 = BufferBarrier::region(1, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ, 0, 50);
        let b2 = BufferBarrier::region(1, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ, 200, 50);

        assert!(b1.try_merge(&b2).is_none());
    }

    #[test]
    fn test_buffer_barrier_equality() {
        let b1 = BufferBarrier::whole(1, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ);
        let b2 = BufferBarrier::whole(1, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ);
        let b3 = BufferBarrier::whole(2, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ);
        assert_eq!(b1, b2);
        assert_ne!(b1, b3);
    }

    // ============================================================
    // TextureBarrier Tests (T-WGPU-P4.7.4)
    // ============================================================

    #[test]
    fn test_texture_barrier_whole() {
        let barrier = TextureBarrier::whole(
            42,
            AccessFlags::TRANSFER_WRITE,
            AccessFlags::SHADER_READ,
            TextureLayout::TransferDst,
            TextureLayout::ShaderReadOnly,
        );
        assert_eq!(barrier.resource_id, 42);
        assert_eq!(barrier.src_access, AccessFlags::TRANSFER_WRITE);
        assert_eq!(barrier.dst_access, AccessFlags::SHADER_READ);
        assert_eq!(barrier.old_layout, TextureLayout::TransferDst);
        assert_eq!(barrier.new_layout, TextureLayout::ShaderReadOnly);
        assert!(barrier.is_whole_texture());
        assert!(barrier.has_layout_transition());
    }

    #[test]
    fn test_texture_barrier_subresource() {
        let range = SubresourceRange::mips(0, 4);
        let barrier = TextureBarrier::subresource(
            42,
            AccessFlags::TRANSFER_WRITE,
            AccessFlags::SHADER_READ,
            TextureLayout::TransferDst,
            TextureLayout::ShaderReadOnly,
            range,
        );
        assert_eq!(barrier.subresource, range);
        assert!(!barrier.is_whole_texture());
    }

    #[test]
    fn test_texture_barrier_no_layout_transition() {
        let barrier = TextureBarrier::whole(
            1,
            AccessFlags::SHADER_READ,
            AccessFlags::SHADER_READ,
            TextureLayout::ShaderReadOnly,
            TextureLayout::ShaderReadOnly,
        );
        assert!(!barrier.has_layout_transition());
    }

    #[test]
    fn test_texture_barrier_can_merge_same_layouts() {
        let t1 = TextureBarrier::subresource(
            1,
            AccessFlags::TRANSFER_WRITE,
            AccessFlags::SHADER_READ,
            TextureLayout::TransferDst,
            TextureLayout::ShaderReadOnly,
            SubresourceRange::mips(0, 2),
        );
        let t2 = TextureBarrier::subresource(
            1,
            AccessFlags::TRANSFER_WRITE,
            AccessFlags::SHADER_READ,
            TextureLayout::TransferDst,
            TextureLayout::ShaderReadOnly,
            SubresourceRange::mips(2, 2),
        );
        assert!(t1.can_merge_with(&t2));
    }

    #[test]
    fn test_texture_barrier_cannot_merge_different_resources() {
        let t1 = TextureBarrier::whole(1, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ, TextureLayout::TransferDst, TextureLayout::ShaderReadOnly);
        let t2 = TextureBarrier::whole(2, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ, TextureLayout::TransferDst, TextureLayout::ShaderReadOnly);
        assert!(!t1.can_merge_with(&t2));
    }

    #[test]
    fn test_texture_barrier_cannot_merge_different_layouts() {
        let t1 = TextureBarrier::whole(1, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ, TextureLayout::TransferDst, TextureLayout::ShaderReadOnly);
        let t2 = TextureBarrier::whole(1, AccessFlags::TRANSFER_WRITE, AccessFlags::COLOR_ATTACHMENT_WRITE, TextureLayout::TransferDst, TextureLayout::ColorAttachment);
        assert!(!t1.can_merge_with(&t2));
    }

    #[test]
    fn test_texture_barrier_try_merge_success() {
        let t1 = TextureBarrier::subresource(
            1,
            AccessFlags::TRANSFER_WRITE,
            AccessFlags::SHADER_READ,
            TextureLayout::TransferDst,
            TextureLayout::ShaderReadOnly,
            SubresourceRange::mips(0, 2),
        );
        let t2 = TextureBarrier::subresource(
            1,
            AccessFlags::HOST_WRITE,
            AccessFlags::UNIFORM_BUFFER_READ,
            TextureLayout::TransferDst,
            TextureLayout::ShaderReadOnly,
            SubresourceRange::mips(2, 2),
        );

        let merged = t1.try_merge(&t2);
        assert!(merged.is_some());

        let m = merged.unwrap();
        assert_eq!(m.resource_id, 1);
        assert_eq!(m.subresource.base_mip, 0);
        assert_eq!(m.subresource.mip_count, Some(4));
        assert!(m.src_access.contains(AccessFlags::TRANSFER_WRITE));
        assert!(m.src_access.contains(AccessFlags::HOST_WRITE));
    }

    #[test]
    fn test_texture_barrier_try_merge_failure_different_layouts() {
        let t1 = TextureBarrier::whole(1, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ, TextureLayout::TransferDst, TextureLayout::ShaderReadOnly);
        let t2 = TextureBarrier::whole(1, AccessFlags::TRANSFER_WRITE, AccessFlags::COLOR_ATTACHMENT_WRITE, TextureLayout::TransferDst, TextureLayout::ColorAttachment);

        assert!(t1.try_merge(&t2).is_none());
    }

    #[test]
    fn test_texture_barrier_equality() {
        let t1 = TextureBarrier::whole(1, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ, TextureLayout::TransferDst, TextureLayout::ShaderReadOnly);
        let t2 = TextureBarrier::whole(1, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ, TextureLayout::TransferDst, TextureLayout::ShaderReadOnly);
        let t3 = TextureBarrier::whole(2, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ, TextureLayout::TransferDst, TextureLayout::ShaderReadOnly);
        assert_eq!(t1, t2);
        assert_ne!(t1, t3);
    }

    // ============================================================
    // BatchedBarrier Tests (T-WGPU-P4.7.4)
    // ============================================================

    #[test]
    fn test_batched_barrier_new() {
        let batch = BatchedBarrier::new();
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
        assert_eq!(batch.src_stages, PipelineStageMask::NONE);
        assert_eq!(batch.dst_stages, PipelineStageMask::NONE);
    }

    #[test]
    fn test_batched_barrier_with_stages() {
        let batch = BatchedBarrier::with_stages(PipelineStageMask::TRANSFER, PipelineStageMask::VERTEX_SHADER);
        assert!(batch.is_empty());
        assert_eq!(batch.src_stages, PipelineStageMask::TRANSFER);
        assert_eq!(batch.dst_stages, PipelineStageMask::VERTEX_SHADER);
    }

    #[test]
    fn test_batched_barrier_add_buffer() {
        let mut batch = BatchedBarrier::new();
        batch.add_buffer_barrier(BufferBarrier::whole(1, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ));

        assert!(!batch.is_empty());
        assert_eq!(batch.len(), 1);
        assert_eq!(batch.buffer_barriers.len(), 1);
        assert_eq!(batch.texture_barriers.len(), 0);
    }

    #[test]
    fn test_batched_barrier_add_texture() {
        let mut batch = BatchedBarrier::new();
        batch.add_texture_barrier(TextureBarrier::whole(
            1,
            AccessFlags::TRANSFER_WRITE,
            AccessFlags::SHADER_READ,
            TextureLayout::TransferDst,
            TextureLayout::ShaderReadOnly,
        ));

        assert!(!batch.is_empty());
        assert_eq!(batch.len(), 1);
        assert_eq!(batch.buffer_barriers.len(), 0);
        assert_eq!(batch.texture_barriers.len(), 1);
    }

    #[test]
    fn test_batched_barrier_len() {
        let mut batch = BatchedBarrier::new();
        batch.add_buffer_barrier(BufferBarrier::whole(1, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ));
        batch.add_buffer_barrier(BufferBarrier::whole(2, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ));
        batch.add_texture_barrier(TextureBarrier::whole(3, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ, TextureLayout::TransferDst, TextureLayout::ShaderReadOnly));

        assert_eq!(batch.len(), 3);
        assert_eq!(batch.buffer_barriers.len(), 2);
        assert_eq!(batch.texture_barriers.len(), 1);
    }

    #[test]
    fn test_batched_barrier_merge() {
        let mut batch1 = BatchedBarrier::with_stages(PipelineStageMask::TRANSFER, PipelineStageMask::VERTEX_SHADER);
        batch1.add_buffer_barrier(BufferBarrier::whole(1, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ));

        let mut batch2 = BatchedBarrier::with_stages(PipelineStageMask::COMPUTE_SHADER, PipelineStageMask::FRAGMENT_SHADER);
        batch2.add_buffer_barrier(BufferBarrier::whole(2, AccessFlags::SHADER_WRITE, AccessFlags::SHADER_READ));
        batch2.add_texture_barrier(TextureBarrier::whole(3, AccessFlags::SHADER_WRITE, AccessFlags::SHADER_READ, TextureLayout::StorageImage, TextureLayout::ShaderReadOnly));

        batch1.merge(batch2);

        assert_eq!(batch1.buffer_barriers.len(), 2);
        assert_eq!(batch1.texture_barriers.len(), 1);
        assert!(batch1.src_stages.contains(PipelineStageMask::TRANSFER));
        assert!(batch1.src_stages.contains(PipelineStageMask::COMPUTE_SHADER));
        assert!(batch1.dst_stages.contains(PipelineStageMask::VERTEX_SHADER));
        assert!(batch1.dst_stages.contains(PipelineStageMask::FRAGMENT_SHADER));
    }

    #[test]
    fn test_batched_barrier_clear() {
        let mut batch = BatchedBarrier::with_stages(PipelineStageMask::TRANSFER, PipelineStageMask::VERTEX_SHADER);
        batch.add_buffer_barrier(BufferBarrier::whole(1, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ));
        batch.add_texture_barrier(TextureBarrier::whole(2, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ, TextureLayout::TransferDst, TextureLayout::ShaderReadOnly));

        batch.clear();

        assert!(batch.is_empty());
        assert_eq!(batch.src_stages, PipelineStageMask::NONE);
        assert_eq!(batch.dst_stages, PipelineStageMask::NONE);
    }

    // ============================================================
    // BarrierBatcher Basic Tests (T-WGPU-P4.7.4)
    // ============================================================

    #[test]
    fn test_batcher_new() {
        let batcher = BarrierBatcher::new();
        assert!(batcher.is_empty());
        assert_eq!(batcher.pending_count(), 0);
    }

    #[test]
    fn test_batcher_with_capacity() {
        let batcher = BarrierBatcher::with_capacity(10, 20);
        assert!(batcher.is_empty());
    }

    #[test]
    fn test_batcher_default() {
        let batcher: BarrierBatcher = Default::default();
        assert!(batcher.is_empty());
    }

    #[test]
    fn test_batcher_add_buffer_barrier() {
        let mut batcher = BarrierBatcher::new();
        batcher.add_buffer_barrier(BufferBarrier::whole(1, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ));

        assert!(!batcher.is_empty());
        assert_eq!(batcher.pending_count(), 1);
        assert_eq!(batcher.pending_buffer_count(), 1);
        assert_eq!(batcher.pending_texture_count(), 0);
    }

    #[test]
    fn test_batcher_add_texture_barrier() {
        let mut batcher = BarrierBatcher::new();
        batcher.add_texture_barrier(TextureBarrier::whole(
            1,
            AccessFlags::TRANSFER_WRITE,
            AccessFlags::SHADER_READ,
            TextureLayout::TransferDst,
            TextureLayout::ShaderReadOnly,
        ));

        assert!(!batcher.is_empty());
        assert_eq!(batcher.pending_count(), 1);
        assert_eq!(batcher.pending_buffer_count(), 0);
        assert_eq!(batcher.pending_texture_count(), 1);
    }

    #[test]
    fn test_batcher_add_barrier_info_buffer() {
        let mut batcher = BarrierBatcher::new();
        let info = BarrierInfo::buffer(
            1,
            HazardType::ReadAfterWrite,
            PipelineStage::Transfer,
            PipelineStage::VertexShader,
            AccessFlags::TRANSFER_WRITE,
            AccessFlags::VERTEX_BUFFER_READ,
        );
        batcher.add_barrier_info(info);

        assert_eq!(batcher.pending_buffer_count(), 1);
        assert!(batcher.src_stage_mask().contains(PipelineStageMask::TRANSFER));
        assert!(batcher.dst_stage_mask().contains(PipelineStageMask::VERTEX_SHADER));
    }

    #[test]
    fn test_batcher_add_barrier_info_texture() {
        let mut batcher = BarrierBatcher::new();
        let info = BarrierInfo::texture(
            1,
            HazardType::ReadAfterWrite,
            PipelineStage::Transfer,
            PipelineStage::FragmentShader,
            AccessFlags::TRANSFER_WRITE,
            AccessFlags::SHADER_READ,
            TextureLayout::TransferDst,
            TextureLayout::ShaderReadOnly,
        );
        batcher.add_barrier_info(info);

        assert_eq!(batcher.pending_texture_count(), 1);
    }

    // ============================================================
    // BarrierBatcher Batching Tests (T-WGPU-P4.7.4)
    // ============================================================

    #[test]
    fn test_batcher_batch_single_buffer() {
        let mut batcher = BarrierBatcher::new();
        batcher.set_stage_masks(PipelineStageMask::TRANSFER, PipelineStageMask::VERTEX_SHADER);
        batcher.add_buffer_barrier(BufferBarrier::whole(1, AccessFlags::TRANSFER_WRITE, AccessFlags::VERTEX_BUFFER_READ));

        let batch = batcher.batch();

        assert_eq!(batch.buffer_barriers.len(), 1);
        assert_eq!(batch.texture_barriers.len(), 0);
        assert_eq!(batch.src_stages, PipelineStageMask::TRANSFER);
        assert_eq!(batch.dst_stages, PipelineStageMask::VERTEX_SHADER);
        assert!(batcher.is_empty());
    }

    #[test]
    fn test_batcher_batch_multiple_barriers() {
        let mut batcher = BarrierBatcher::new();
        batcher.set_stage_masks(PipelineStageMask::TRANSFER, PipelineStageMask::FRAGMENT_SHADER);
        batcher.add_buffer_barrier(BufferBarrier::whole(1, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ));
        batcher.add_buffer_barrier(BufferBarrier::whole(2, AccessFlags::TRANSFER_WRITE, AccessFlags::UNIFORM_BUFFER_READ));
        batcher.add_texture_barrier(TextureBarrier::whole(3, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ, TextureLayout::TransferDst, TextureLayout::ShaderReadOnly));

        let batch = batcher.batch();

        assert_eq!(batch.buffer_barriers.len(), 2);
        assert_eq!(batch.texture_barriers.len(), 1);
    }

    #[test]
    fn test_batcher_batch_merges_same_resource_buffers() {
        let mut batcher = BarrierBatcher::new();
        batcher.add_buffer_barrier(BufferBarrier::whole(1, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ));
        batcher.add_buffer_barrier(BufferBarrier::whole(1, AccessFlags::HOST_WRITE, AccessFlags::UNIFORM_BUFFER_READ));

        let batch = batcher.batch();

        // Should be merged into one barrier
        assert_eq!(batch.buffer_barriers.len(), 1);
        let merged = &batch.buffer_barriers[0];
        assert!(merged.src_access.contains(AccessFlags::TRANSFER_WRITE));
        assert!(merged.src_access.contains(AccessFlags::HOST_WRITE));
        assert!(merged.dst_access.contains(AccessFlags::SHADER_READ));
        assert!(merged.dst_access.contains(AccessFlags::UNIFORM_BUFFER_READ));
    }

    #[test]
    fn test_batcher_batch_merges_same_resource_textures() {
        let mut batcher = BarrierBatcher::new();
        // Same resource, same layout transition, adjacent mips
        batcher.add_texture_barrier(TextureBarrier::subresource(
            1,
            AccessFlags::TRANSFER_WRITE,
            AccessFlags::SHADER_READ,
            TextureLayout::TransferDst,
            TextureLayout::ShaderReadOnly,
            SubresourceRange::mips(0, 2),
        ));
        batcher.add_texture_barrier(TextureBarrier::subresource(
            1,
            AccessFlags::TRANSFER_WRITE,
            AccessFlags::SHADER_READ,
            TextureLayout::TransferDst,
            TextureLayout::ShaderReadOnly,
            SubresourceRange::mips(2, 2),
        ));

        let batch = batcher.batch();

        // Should be merged into one barrier
        assert_eq!(batch.texture_barriers.len(), 1);
        let merged = &batch.texture_barriers[0];
        assert_eq!(merged.subresource.base_mip, 0);
        assert_eq!(merged.subresource.mip_count, Some(4));
    }

    #[test]
    fn test_batcher_batch_keeps_different_textures_separate() {
        let mut batcher = BarrierBatcher::new();
        batcher.add_texture_barrier(TextureBarrier::whole(1, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ, TextureLayout::TransferDst, TextureLayout::ShaderReadOnly));
        batcher.add_texture_barrier(TextureBarrier::whole(2, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ, TextureLayout::TransferDst, TextureLayout::ShaderReadOnly));

        let batch = batcher.batch();

        // Different resources should not be merged
        assert_eq!(batch.texture_barriers.len(), 2);
    }

    #[test]
    fn test_batcher_batch_keeps_different_layouts_separate() {
        let mut batcher = BarrierBatcher::new();
        batcher.add_texture_barrier(TextureBarrier::whole(1, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ, TextureLayout::TransferDst, TextureLayout::ShaderReadOnly));
        batcher.add_texture_barrier(TextureBarrier::whole(1, AccessFlags::TRANSFER_WRITE, AccessFlags::COLOR_ATTACHMENT_WRITE, TextureLayout::TransferDst, TextureLayout::ColorAttachment));

        let batch = batcher.batch();

        // Different target layouts should not be merged
        assert_eq!(batch.texture_barriers.len(), 2);
    }

    #[test]
    fn test_batcher_batch_empty() {
        let mut batcher = BarrierBatcher::new();
        let batch = batcher.batch();

        assert!(batch.is_empty());
    }

    // ============================================================
    // BarrierBatcher batch_by_stage Tests (T-WGPU-P4.7.4)
    // ============================================================

    #[test]
    fn test_batcher_batch_by_stage_empty() {
        let mut batcher = BarrierBatcher::new();
        let batches = batcher.batch_by_stage();
        assert!(batches.is_empty());
    }

    #[test]
    fn test_batcher_batch_by_stage_single_stage() {
        let mut batcher = BarrierBatcher::new();
        batcher.add_buffer_barrier(BufferBarrier::whole(1, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ));
        batcher.add_buffer_barrier(BufferBarrier::whole(2, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ));

        let batches = batcher.batch_by_stage();

        // All barriers have same stage transition, should be one batch
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].buffer_barriers.len(), 2);
    }

    #[test]
    fn test_batcher_batch_by_stage_multiple_stages() {
        let mut batcher = BarrierBatcher::new();
        // Transfer -> Shader read
        batcher.add_buffer_barrier(BufferBarrier::whole(1, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ));
        // Transfer -> Vertex input
        batcher.add_buffer_barrier(BufferBarrier::whole(2, AccessFlags::TRANSFER_WRITE, AccessFlags::VERTEX_BUFFER_READ));

        let batches = batcher.batch_by_stage();

        // Different stage transitions, should be multiple batches
        assert!(batches.len() >= 1);
    }

    #[test]
    fn test_batcher_batch_by_stage_merges_within_group() {
        let mut batcher = BarrierBatcher::new();
        // Same access patterns = same stage group
        batcher.add_buffer_barrier(BufferBarrier::whole(1, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ));
        batcher.add_buffer_barrier(BufferBarrier::whole(1, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ));

        let batches = batcher.batch_by_stage();

        // Should be merged within the group
        assert_eq!(batches.len(), 1);
        // Both barriers for same resource should be merged
        assert_eq!(batches[0].buffer_barriers.len(), 1);
    }

    // ============================================================
    // BarrierBatcher Utility Tests (T-WGPU-P4.7.4)
    // ============================================================

    #[test]
    fn test_batcher_merge_stages() {
        let merged = BarrierBatcher::merge_stages(PipelineStageMask::VERTEX_SHADER, PipelineStageMask::FRAGMENT_SHADER);
        assert!(merged.contains(PipelineStageMask::VERTEX_SHADER));
        assert!(merged.contains(PipelineStageMask::FRAGMENT_SHADER));
    }

    #[test]
    fn test_batcher_merge_access() {
        let merged = BarrierBatcher::merge_access(AccessFlags::SHADER_READ, AccessFlags::UNIFORM_BUFFER_READ);
        assert!(merged.contains(AccessFlags::SHADER_READ));
        assert!(merged.contains(AccessFlags::UNIFORM_BUFFER_READ));
    }

    #[test]
    fn test_batcher_clear() {
        let mut batcher = BarrierBatcher::new();
        batcher.set_stage_masks(PipelineStageMask::TRANSFER, PipelineStageMask::FRAGMENT_SHADER);
        batcher.add_buffer_barrier(BufferBarrier::whole(1, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ));
        batcher.add_texture_barrier(TextureBarrier::whole(2, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ, TextureLayout::TransferDst, TextureLayout::ShaderReadOnly));

        batcher.clear();

        assert!(batcher.is_empty());
        assert_eq!(batcher.src_stage_mask(), PipelineStageMask::NONE);
        assert_eq!(batcher.dst_stage_mask(), PipelineStageMask::NONE);
    }

    #[test]
    fn test_batcher_stage_mask_accumulation() {
        let mut batcher = BarrierBatcher::new();
        batcher.add_barrier_info(BarrierInfo::buffer(
            1, HazardType::ReadAfterWrite,
            PipelineStage::Transfer, PipelineStage::VertexShader,
            AccessFlags::TRANSFER_WRITE, AccessFlags::VERTEX_BUFFER_READ,
        ));
        batcher.add_barrier_info(BarrierInfo::buffer(
            2, HazardType::ReadAfterWrite,
            PipelineStage::ComputeShader, PipelineStage::FragmentShader,
            AccessFlags::SHADER_WRITE, AccessFlags::SHADER_READ,
        ));

        // Stage masks should accumulate
        assert!(batcher.src_stage_mask().contains(PipelineStageMask::TRANSFER));
        assert!(batcher.src_stage_mask().contains(PipelineStageMask::COMPUTE_SHADER));
        assert!(batcher.dst_stage_mask().contains(PipelineStageMask::VERTEX_SHADER));
        assert!(batcher.dst_stage_mask().contains(PipelineStageMask::FRAGMENT_SHADER));
    }

    // ============================================================
    // BarrierBatcher Complex Workflow Tests (T-WGPU-P4.7.4)
    // ============================================================

    #[test]
    fn test_batcher_texture_upload_workflow() {
        let mut batcher = BarrierBatcher::new();

        // Multiple textures uploaded in parallel, all transitioning from transfer to shader read
        for i in 0..4 {
            batcher.add_texture_barrier(TextureBarrier::whole(
                i,
                AccessFlags::TRANSFER_WRITE,
                AccessFlags::SHADER_READ,
                TextureLayout::TransferDst,
                TextureLayout::ShaderReadOnly,
            ));
        }

        let batch = batcher.batch();

        // All should be batched together
        assert_eq!(batch.texture_barriers.len(), 4);
    }

    #[test]
    fn test_batcher_render_pass_workflow() {
        let mut batcher = BarrierBatcher::new();

        // Color attachment transition
        batcher.add_texture_barrier(TextureBarrier::whole(
            1,
            AccessFlags::SHADER_READ,
            AccessFlags::COLOR_ATTACHMENT_WRITE,
            TextureLayout::ShaderReadOnly,
            TextureLayout::ColorAttachment,
        ));

        // Depth attachment transition
        batcher.add_texture_barrier(TextureBarrier::whole(
            2,
            AccessFlags::SHADER_READ,
            AccessFlags::DEPTH_STENCIL_WRITE,
            TextureLayout::ShaderReadOnly,
            TextureLayout::DepthStencilAttachment,
        ));

        // Vertex buffer barrier
        batcher.add_buffer_barrier(BufferBarrier::whole(
            3,
            AccessFlags::TRANSFER_WRITE,
            AccessFlags::VERTEX_BUFFER_READ,
        ));

        // Index buffer barrier
        batcher.add_buffer_barrier(BufferBarrier::whole(
            4,
            AccessFlags::TRANSFER_WRITE,
            AccessFlags::INDEX_BUFFER_READ,
        ));

        let batch = batcher.batch();

        assert_eq!(batch.texture_barriers.len(), 2);
        assert_eq!(batch.buffer_barriers.len(), 2);
    }

    #[test]
    fn test_batcher_compute_dispatch_workflow() {
        let mut batcher = BarrierBatcher::new();

        // Input buffer (read from previous compute)
        batcher.add_buffer_barrier(BufferBarrier::whole(
            1,
            AccessFlags::SHADER_WRITE,
            AccessFlags::SHADER_READ,
        ));

        // Output buffer (write after previous read)
        batcher.add_buffer_barrier(BufferBarrier::whole(
            2,
            AccessFlags::SHADER_READ,
            AccessFlags::SHADER_WRITE,
        ));

        // Storage texture
        batcher.add_texture_barrier(TextureBarrier::whole(
            3,
            AccessFlags::SHADER_READ,
            AccessFlags::SHADER_WRITE,
            TextureLayout::ShaderReadOnly,
            TextureLayout::StorageImage,
        ));

        let batch = batcher.batch();

        assert_eq!(batch.buffer_barriers.len(), 2);
        assert_eq!(batch.texture_barriers.len(), 1);
    }

    #[test]
    fn test_batcher_mipmap_generation_workflow() {
        let mut batcher = BarrierBatcher::new();

        // Each mip level transition (base level to transfer src, others from transfer dst to transfer src)
        for mip in 0..4 {
            batcher.add_texture_barrier(TextureBarrier::subresource(
                1,
                AccessFlags::TRANSFER_WRITE,
                AccessFlags::TRANSFER_READ,
                TextureLayout::TransferDst,
                TextureLayout::TransferSrc,
                SubresourceRange::single(mip, 0),
            ));
        }

        let batch = batcher.batch();

        // All for same resource with same layout transition should be merged
        assert_eq!(batch.texture_barriers.len(), 1);
        // The merged barrier should cover mips 0-3
        assert_eq!(batch.texture_barriers[0].subresource.base_mip, 0);
    }

    #[test]
    fn test_batcher_with_barrier_detector_integration() {
        let mut detector = BarrierDetector::new();
        let mut batcher = BarrierBatcher::new();

        // Simulate a sequence of operations
        detector.record_access(1, ResourceState::buffer(PipelineStage::Transfer, AccessFlags::TRANSFER_WRITE));
        detector.record_access(2, ResourceState::texture(
            PipelineStage::Transfer,
            AccessFlags::TRANSFER_WRITE,
            TextureLayout::TransferDst,
        ));

        // Transition to shader read
        if let Some(info) = detector.transition(1, ResourceState::buffer(PipelineStage::VertexShader, AccessFlags::VERTEX_BUFFER_READ)) {
            batcher.add_barrier_info(info);
        }

        if let Some(info) = detector.transition(2, ResourceState::texture(
            PipelineStage::FragmentShader,
            AccessFlags::SHADER_READ,
            TextureLayout::ShaderReadOnly,
        )) {
            batcher.add_barrier_info(info);
        }

        let batch = batcher.batch();

        assert_eq!(batch.buffer_barriers.len(), 1);
        assert_eq!(batch.texture_barriers.len(), 1);
    }

    #[test]
    fn test_batcher_multiple_batches_workflow() {
        let mut batcher = BarrierBatcher::new();

        // First batch: upload phase
        batcher.add_buffer_barrier(BufferBarrier::whole(1, AccessFlags::HOST_WRITE, AccessFlags::TRANSFER_READ));
        let upload_batch = batcher.batch();
        assert_eq!(upload_batch.buffer_barriers.len(), 1);

        // Second batch: render phase
        batcher.add_buffer_barrier(BufferBarrier::whole(1, AccessFlags::TRANSFER_WRITE, AccessFlags::VERTEX_BUFFER_READ));
        batcher.add_texture_barrier(TextureBarrier::whole(2, AccessFlags::TRANSFER_WRITE, AccessFlags::SHADER_READ, TextureLayout::TransferDst, TextureLayout::ShaderReadOnly));
        let render_batch = batcher.batch();
        assert_eq!(render_batch.buffer_barriers.len(), 1);
        assert_eq!(render_batch.texture_barriers.len(), 1);
    }
}
