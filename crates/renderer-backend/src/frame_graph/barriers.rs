//! Barrier Resolution for Frame Graph Pass Synchronization (T-WGPU-P7.5.9)
//!
//! This module provides the barrier resolver that computes optimal synchronization
//! barriers between frame graph passes. It tracks resource states and generates
//! the minimal set of barriers required for correct GPU execution.
//!
//! # Architecture
//!
//! The barrier resolver operates in several phases:
//! 1. **State Tracking**: Maintains per-resource state as passes are processed
//! 2. **Transition Detection**: Identifies when resource states change between passes
//! 3. **Barrier Generation**: Creates appropriate barrier types (texture/buffer/memory)
//! 4. **Batch Formation**: Groups barriers by pipeline stages for efficient submission
//!
//! # Usage
//!
//! ```rust,ignore
//! use renderer_backend::frame_graph::barriers::*;
//!
//! let mut resolver = BarrierResolver::new();
//!
//! // Process frame graph
//! let barrier_batches = resolver.resolve(&frame_graph);
//!
//! // Execute with barriers
//! for (pass_id, batch) in barrier_batches {
//!     // Insert barriers before pass
//!     command_encoder.insert_barriers(&batch.barriers);
//!     // Execute pass
//!     execute_pass(pass_id);
//! }
//! ```

use std::collections::HashMap;
use std::fmt;

// Import from parent module
use super::{PassIndex, ResourceHandle, ResourceState};

// ---------------------------------------------------------------------------
// SubresourceRange
// ---------------------------------------------------------------------------

/// Defines a range of subresources within a texture.
///
/// Used to specify which mip levels and array layers are affected by a barrier.
/// This enables fine-grained synchronization for large textures with many
/// mip levels or array layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SubresourceRange {
    /// Base mip level (0 = highest resolution).
    pub base_mip: u32,
    /// Number of mip levels affected (0 = remaining mips from base).
    pub mip_count: u32,
    /// Base array layer (0 = first layer).
    pub base_layer: u32,
    /// Number of array layers affected (0 = remaining layers from base).
    pub layer_count: u32,
}

impl SubresourceRange {
    /// Creates a subresource range covering all mips and layers.
    pub const fn all() -> Self {
        Self {
            base_mip: 0,
            mip_count: 0, // 0 = all remaining
            base_layer: 0,
            layer_count: 0, // 0 = all remaining
        }
    }

    /// Creates a subresource range for a single mip level.
    pub const fn single_mip(mip: u32) -> Self {
        Self {
            base_mip: mip,
            mip_count: 1,
            base_layer: 0,
            layer_count: 0,
        }
    }

    /// Creates a subresource range for a single array layer.
    pub const fn single_layer(layer: u32) -> Self {
        Self {
            base_mip: 0,
            mip_count: 0,
            base_layer: layer,
            layer_count: 1,
        }
    }

    /// Creates a subresource range for a specific mip and layer.
    pub const fn single(mip: u32, layer: u32) -> Self {
        Self {
            base_mip: mip,
            mip_count: 1,
            base_layer: layer,
            layer_count: 1,
        }
    }

    /// Creates a subresource range with explicit bounds.
    pub const fn new(base_mip: u32, mip_count: u32, base_layer: u32, layer_count: u32) -> Self {
        Self {
            base_mip,
            mip_count,
            base_layer,
            layer_count,
        }
    }

    /// Returns true if this range covers all subresources.
    #[inline]
    pub const fn is_all(&self) -> bool {
        self.base_mip == 0
            && self.mip_count == 0
            && self.base_layer == 0
            && self.layer_count == 0
    }

    /// Returns true if this range overlaps with another range.
    pub fn overlaps(&self, other: &SubresourceRange) -> bool {
        // Handle "all remaining" cases
        let self_mip_end = if self.mip_count == 0 {
            u32::MAX
        } else {
            self.base_mip + self.mip_count
        };
        let other_mip_end = if other.mip_count == 0 {
            u32::MAX
        } else {
            other.base_mip + other.mip_count
        };
        let self_layer_end = if self.layer_count == 0 {
            u32::MAX
        } else {
            self.base_layer + self.layer_count
        };
        let other_layer_end = if other.layer_count == 0 {
            u32::MAX
        } else {
            other.base_layer + other.layer_count
        };

        // Check for non-overlap
        let mip_overlap =
            self.base_mip < other_mip_end && other.base_mip < self_mip_end;
        let layer_overlap =
            self.base_layer < other_layer_end && other.base_layer < self_layer_end;

        mip_overlap && layer_overlap
    }
}

impl Default for SubresourceRange {
    fn default() -> Self {
        Self::all()
    }
}

impl fmt::Display for SubresourceRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_all() {
            write!(f, "SubresourceRange::all()")
        } else {
            write!(
                f,
                "SubresourceRange(mips={}..{}, layers={}..{})",
                self.base_mip,
                if self.mip_count == 0 {
                    "all".to_string()
                } else {
                    (self.base_mip + self.mip_count).to_string()
                },
                self.base_layer,
                if self.layer_count == 0 {
                    "all".to_string()
                } else {
                    (self.base_layer + self.layer_count).to_string()
                },
            )
        }
    }
}

// ---------------------------------------------------------------------------
// PipelineStageFlags (bitflags for barrier scheduling)
// ---------------------------------------------------------------------------

bitflags::bitflags! {
    /// Pipeline stage flags for barrier scheduling.
    ///
    /// These flags indicate which pipeline stages are involved in a barrier
    /// transition. Multiple stages can be combined to express complex
    /// synchronization requirements.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use renderer_backend::frame_graph::barriers::PipelineStageFlags;
    ///
    /// let stages = PipelineStageFlags::VERTEX_SHADER | PipelineStageFlags::FRAGMENT_SHADER;
    /// assert!(stages.contains(PipelineStageFlags::VERTEX_SHADER));
    /// ```
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct PipelineStageFlags: u32 {
        /// Top of pipe (before any commands execute).
        const TOP = 1 << 0;
        /// Draw indirect command processing.
        const DRAW_INDIRECT = 1 << 1;
        /// Vertex input assembly (index/vertex buffer reads).
        const VERTEX_INPUT = 1 << 2;
        /// Vertex shader execution.
        const VERTEX_SHADER = 1 << 3;
        /// Fragment shader execution.
        const FRAGMENT_SHADER = 1 << 4;
        /// Early depth/stencil tests.
        const EARLY_FRAGMENT = 1 << 5;
        /// Late depth/stencil tests.
        const LATE_FRAGMENT = 1 << 6;
        /// Color attachment output.
        const COLOR_OUTPUT = 1 << 7;
        /// Compute shader execution.
        const COMPUTE_SHADER = 1 << 8;
        /// Transfer/copy operations.
        const TRANSFER = 1 << 9;
        /// Bottom of pipe (after all commands complete).
        const BOTTOM = 1 << 10;
        /// All graphics pipeline stages.
        const ALL_GRAPHICS = 1 << 11;
        /// All commands (graphics + compute + transfer).
        const ALL_COMMANDS = 1 << 12;
        /// Ray tracing shader stages.
        const RAY_TRACING = 1 << 13;
        /// Host (CPU) access.
        const HOST = 1 << 14;
    }
}

impl Default for PipelineStageFlags {
    fn default() -> Self {
        Self::empty()
    }
}

impl fmt::Display for PipelineStageFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return write!(f, "NONE");
        }

        let mut parts: Vec<&str> = Vec::new();
        if self.contains(Self::TOP) {
            parts.push("TOP");
        }
        if self.contains(Self::DRAW_INDIRECT) {
            parts.push("DRAW_INDIRECT");
        }
        if self.contains(Self::VERTEX_INPUT) {
            parts.push("VERTEX_INPUT");
        }
        if self.contains(Self::VERTEX_SHADER) {
            parts.push("VERTEX_SHADER");
        }
        if self.contains(Self::FRAGMENT_SHADER) {
            parts.push("FRAGMENT_SHADER");
        }
        if self.contains(Self::EARLY_FRAGMENT) {
            parts.push("EARLY_FRAGMENT");
        }
        if self.contains(Self::LATE_FRAGMENT) {
            parts.push("LATE_FRAGMENT");
        }
        if self.contains(Self::COLOR_OUTPUT) {
            parts.push("COLOR_OUTPUT");
        }
        if self.contains(Self::COMPUTE_SHADER) {
            parts.push("COMPUTE_SHADER");
        }
        if self.contains(Self::TRANSFER) {
            parts.push("TRANSFER");
        }
        if self.contains(Self::BOTTOM) {
            parts.push("BOTTOM");
        }
        if self.contains(Self::ALL_GRAPHICS) {
            parts.push("ALL_GRAPHICS");
        }
        if self.contains(Self::ALL_COMMANDS) {
            parts.push("ALL_COMMANDS");
        }
        if self.contains(Self::RAY_TRACING) {
            parts.push("RAY_TRACING");
        }
        if self.contains(Self::HOST) {
            parts.push("HOST");
        }

        write!(f, "{}", parts.join(" | "))
    }
}

impl PipelineStageFlags {
    /// Returns the pipeline stages for a given resource state.
    ///
    /// Maps resource states to the pipeline stages where they are typically accessed.
    pub fn from_resource_state(state: ResourceState) -> Self {
        match state {
            ResourceState::Uninitialized => Self::TOP,
            ResourceState::VertexBuffer | ResourceState::IndexBuffer => Self::VERTEX_INPUT,
            ResourceState::IndirectArgument => Self::DRAW_INDIRECT,
            ResourceState::ColorAttachment => Self::COLOR_OUTPUT,
            ResourceState::DepthStencilAttachment => {
                Self::EARLY_FRAGMENT | Self::LATE_FRAGMENT
            }
            ResourceState::DepthStencilReadOnly => Self::EARLY_FRAGMENT | Self::FRAGMENT_SHADER,
            ResourceState::ShaderRead => {
                Self::VERTEX_SHADER | Self::FRAGMENT_SHADER | Self::COMPUTE_SHADER
            }
            ResourceState::ShaderReadWrite => {
                Self::FRAGMENT_SHADER | Self::COMPUTE_SHADER
            }
            ResourceState::TransferSrc | ResourceState::TransferDst => Self::TRANSFER,
            ResourceState::AccelerationStructure => Self::RAY_TRACING,
            ResourceState::Present => Self::BOTTOM,
        }
    }
}

// ---------------------------------------------------------------------------
// BarrierType
// ---------------------------------------------------------------------------

/// Describes a GPU synchronization barrier.
///
/// Barriers ensure that GPU operations complete in the correct order and that
/// memory is visible to subsequent operations. This enum captures the different
/// types of barriers needed for textures, buffers, and general synchronization.
#[derive(Clone, Debug, PartialEq)]
pub enum BarrierType {
    /// Texture state transition barrier.
    ///
    /// Transitions a texture between states (e.g., render target to shader read).
    /// May involve layout changes and cache operations.
    Texture {
        /// The texture resource being transitioned.
        resource: ResourceHandle,
        /// State the texture is transitioning from.
        old_state: ResourceState,
        /// State the texture is transitioning to.
        new_state: ResourceState,
        /// Subresource range affected by this barrier.
        subresource: SubresourceRange,
    },
    /// Buffer state transition barrier.
    ///
    /// Transitions a buffer between states (e.g., write to read).
    /// Ensures proper cache coherency for buffer accesses.
    Buffer {
        /// The buffer resource being transitioned.
        resource: ResourceHandle,
        /// State the buffer is transitioning from.
        old_state: ResourceState,
        /// State the buffer is transitioning to.
        new_state: ResourceState,
        /// Byte offset of the affected range (0 = start of buffer).
        offset: u64,
        /// Size of the affected range (0 = entire buffer from offset).
        size: u64,
    },
    /// Full memory barrier.
    ///
    /// Ensures all previous writes are visible to all subsequent reads.
    /// More expensive than targeted barriers but simpler to use.
    Memory,
    /// Execution-only barrier.
    ///
    /// Ensures all previous commands complete before subsequent commands begin.
    /// Does not involve memory synchronization.
    Execution,
}

impl BarrierType {
    /// Creates a texture barrier with full subresource range.
    pub fn texture_full(
        resource: ResourceHandle,
        old_state: ResourceState,
        new_state: ResourceState,
    ) -> Self {
        Self::Texture {
            resource,
            old_state,
            new_state,
            subresource: SubresourceRange::all(),
        }
    }

    /// Creates a buffer barrier for the entire buffer.
    pub fn buffer_full(
        resource: ResourceHandle,
        old_state: ResourceState,
        new_state: ResourceState,
    ) -> Self {
        Self::Buffer {
            resource,
            old_state,
            new_state,
            offset: 0,
            size: 0, // 0 = entire buffer
        }
    }

    /// Returns the resource handle if this is a resource barrier.
    pub fn resource(&self) -> Option<ResourceHandle> {
        match self {
            BarrierType::Texture { resource, .. } | BarrierType::Buffer { resource, .. } => {
                Some(*resource)
            }
            BarrierType::Memory | BarrierType::Execution => None,
        }
    }

    /// Returns true if this barrier involves a state change.
    pub fn has_state_change(&self) -> bool {
        match self {
            BarrierType::Texture {
                old_state,
                new_state,
                ..
            }
            | BarrierType::Buffer {
                old_state,
                new_state,
                ..
            } => old_state != new_state,
            BarrierType::Memory | BarrierType::Execution => false,
        }
    }

    /// Returns true if this is a texture barrier.
    #[inline]
    pub fn is_texture(&self) -> bool {
        matches!(self, BarrierType::Texture { .. })
    }

    /// Returns true if this is a buffer barrier.
    #[inline]
    pub fn is_buffer(&self) -> bool {
        matches!(self, BarrierType::Buffer { .. })
    }

    /// Returns true if this is a memory barrier.
    #[inline]
    pub fn is_memory(&self) -> bool {
        matches!(self, BarrierType::Memory)
    }

    /// Returns true if this is an execution barrier.
    #[inline]
    pub fn is_execution(&self) -> bool {
        matches!(self, BarrierType::Execution)
    }
}

impl fmt::Display for BarrierType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BarrierType::Texture {
                resource,
                old_state,
                new_state,
                subresource,
            } => {
                write!(
                    f,
                    "Texture({}, {} -> {}, {})",
                    resource, old_state, new_state, subresource
                )
            }
            BarrierType::Buffer {
                resource,
                old_state,
                new_state,
                offset,
                size,
            } => {
                write!(
                    f,
                    "Buffer({}, {} -> {}, offset={}, size={})",
                    resource, old_state, new_state, offset, size
                )
            }
            BarrierType::Memory => write!(f, "Memory"),
            BarrierType::Execution => write!(f, "Execution"),
        }
    }
}

// ---------------------------------------------------------------------------
// BarrierBatch
// ---------------------------------------------------------------------------

/// A batch of barriers to be submitted together.
///
/// Groups multiple barriers that can be submitted in a single API call,
/// along with the pipeline stages involved in the synchronization.
#[derive(Clone, Debug, Default)]
pub struct BarrierBatch {
    /// The barriers to submit.
    pub barriers: Vec<BarrierType>,
    /// Pipeline stages that must complete before the barrier.
    pub source_stage: PipelineStageFlags,
    /// Pipeline stages that wait for the barrier.
    pub dest_stage: PipelineStageFlags,
}

impl BarrierBatch {
    /// Creates an empty barrier batch.
    pub fn new() -> Self {
        Self {
            barriers: Vec::new(),
            source_stage: PipelineStageFlags::empty(),
            dest_stage: PipelineStageFlags::empty(),
        }
    }

    /// Creates a barrier batch with the given stages.
    pub fn with_stages(source_stage: PipelineStageFlags, dest_stage: PipelineStageFlags) -> Self {
        Self {
            barriers: Vec::new(),
            source_stage,
            dest_stage,
        }
    }

    /// Adds a barrier to this batch.
    pub fn add(&mut self, barrier: BarrierType) {
        self.barriers.push(barrier);
    }

    /// Adds a barrier and updates stages based on state transition.
    pub fn add_with_stages(&mut self, barrier: BarrierType) {
        match &barrier {
            BarrierType::Texture {
                old_state,
                new_state,
                ..
            }
            | BarrierType::Buffer {
                old_state,
                new_state,
                ..
            } => {
                self.source_stage |= PipelineStageFlags::from_resource_state(*old_state);
                self.dest_stage |= PipelineStageFlags::from_resource_state(*new_state);
            }
            BarrierType::Memory => {
                self.source_stage |= PipelineStageFlags::ALL_COMMANDS;
                self.dest_stage |= PipelineStageFlags::ALL_COMMANDS;
            }
            BarrierType::Execution => {
                self.source_stage |= PipelineStageFlags::BOTTOM;
                self.dest_stage |= PipelineStageFlags::TOP;
            }
        }
        self.barriers.push(barrier);
    }

    /// Returns true if this batch is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.barriers.is_empty()
    }

    /// Returns the number of barriers in this batch.
    #[inline]
    pub fn len(&self) -> usize {
        self.barriers.len()
    }

    /// Merges another batch into this one.
    pub fn merge(&mut self, other: BarrierBatch) {
        self.barriers.extend(other.barriers);
        self.source_stage |= other.source_stage;
        self.dest_stage |= other.dest_stage;
    }

    /// Returns the number of texture barriers.
    pub fn texture_count(&self) -> usize {
        self.barriers.iter().filter(|b| b.is_texture()).count()
    }

    /// Returns the number of buffer barriers.
    pub fn buffer_count(&self) -> usize {
        self.barriers.iter().filter(|b| b.is_buffer()).count()
    }
}

impl fmt::Display for BarrierBatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BarrierBatch({} barriers, src={}, dst={})",
            self.barriers.len(),
            self.source_stage,
            self.dest_stage
        )
    }
}

// ---------------------------------------------------------------------------
// BarrierResolver
// ---------------------------------------------------------------------------

/// Resolves resource state transitions into synchronization barriers.
///
/// The barrier resolver tracks the current state of each resource and generates
/// the appropriate barriers when a pass requires a different state. It ensures
/// optimal barrier placement by:
///
/// 1. Tracking per-resource state across the frame
/// 2. Detecting state transitions at pass boundaries
/// 3. Generating minimal barriers for correct synchronization
/// 4. Grouping barriers for efficient submission
///
/// # Example
///
/// ```rust,ignore
/// use renderer_backend::frame_graph::barriers::*;
/// use renderer_backend::frame_graph::graph::FrameGraph;
///
/// let mut resolver = BarrierResolver::new();
///
/// // Resolve barriers for a frame graph
/// let batches = resolver.resolve(&graph);
///
/// for (pass_id, batch) in batches {
///     println!("Pass {:?}: {} barriers", pass_id, batch.len());
/// }
///
/// // Reset for next frame
/// resolver.reset();
/// ```
#[derive(Clone, Debug, Default)]
pub struct BarrierResolver {
    /// Current state of each resource.
    resource_states: HashMap<ResourceHandle, ResourceState>,
}

impl BarrierResolver {
    /// Creates a new barrier resolver with empty state.
    pub fn new() -> Self {
        Self {
            resource_states: HashMap::new(),
        }
    }

    /// Creates a barrier resolver with pre-initialized resource states.
    pub fn with_states(states: HashMap<ResourceHandle, ResourceState>) -> Self {
        Self {
            resource_states: states,
        }
    }

    /// Returns the current state of a resource.
    ///
    /// Returns `ResourceState::Uninitialized` if the resource hasn't been tracked yet.
    pub fn get_state(&self, resource: ResourceHandle) -> ResourceState {
        self.resource_states
            .get(&resource)
            .copied()
            .unwrap_or(ResourceState::Uninitialized)
    }

    /// Sets the state of a resource without generating a barrier.
    ///
    /// Use this to initialize resource states before processing passes.
    pub fn set_state(&mut self, resource: ResourceHandle, state: ResourceState) {
        self.resource_states.insert(resource, state);
    }

    /// Transitions a resource to a new state and returns any required barrier.
    ///
    /// If the resource is already in the requested state, returns `None`.
    /// Otherwise, updates the tracked state and returns a barrier for the transition.
    ///
    /// # Arguments
    ///
    /// * `resource` - The resource to transition.
    /// * `new_state` - The desired new state.
    ///
    /// # Returns
    ///
    /// `Some(BarrierType)` if a transition is needed, `None` otherwise.
    pub fn transition(
        &mut self,
        resource: ResourceHandle,
        new_state: ResourceState,
    ) -> Option<BarrierType> {
        let old_state = self.get_state(resource);

        if old_state == new_state {
            return None;
        }

        self.resource_states.insert(resource, new_state);

        // Determine barrier type based on resource (default to texture)
        // In practice, the caller should use transition_texture or transition_buffer
        Some(BarrierType::texture_full(resource, old_state, new_state))
    }

    /// Transitions a texture resource with subresource range.
    pub fn transition_texture(
        &mut self,
        resource: ResourceHandle,
        new_state: ResourceState,
        subresource: SubresourceRange,
    ) -> Option<BarrierType> {
        let old_state = self.get_state(resource);

        if old_state == new_state {
            return None;
        }

        self.resource_states.insert(resource, new_state);

        Some(BarrierType::Texture {
            resource,
            old_state,
            new_state,
            subresource,
        })
    }

    /// Transitions a buffer resource with byte range.
    pub fn transition_buffer(
        &mut self,
        resource: ResourceHandle,
        new_state: ResourceState,
        offset: u64,
        size: u64,
    ) -> Option<BarrierType> {
        let old_state = self.get_state(resource);

        if old_state == new_state {
            return None;
        }

        self.resource_states.insert(resource, new_state);

        Some(BarrierType::Buffer {
            resource,
            old_state,
            new_state,
            offset,
            size,
        })
    }

    /// Resolves all barriers for a frame graph.
    ///
    /// Processes passes in topological order and generates barriers for each
    /// resource state transition. Returns a list of (PassIndex, BarrierBatch)
    /// pairs indicating barriers to insert before each pass.
    ///
    /// # Arguments
    ///
    /// * `passes` - Ordered list of passes with their resource accesses.
    /// * `pass_resources` - Function that returns (resource, required_state) pairs for a pass.
    ///
    /// # Returns
    ///
    /// Vector of (PassIndex, BarrierBatch) pairs for barrier insertion.
    pub fn resolve_passes<F>(
        &mut self,
        passes: &[PassIndex],
        mut pass_resources: F,
    ) -> Vec<(PassIndex, BarrierBatch)>
    where
        F: FnMut(PassIndex) -> Vec<(ResourceHandle, ResourceState, bool)>, // (resource, state, is_texture)
    {
        let mut result = Vec::with_capacity(passes.len());

        for &pass_idx in passes {
            let mut batch = BarrierBatch::new();
            let resources = pass_resources(pass_idx);

            for (resource, required_state, is_texture) in resources {
                let barrier = if is_texture {
                    self.transition_texture(resource, required_state, SubresourceRange::all())
                } else {
                    self.transition_buffer(resource, required_state, 0, 0)
                };

                if let Some(b) = barrier {
                    batch.add_with_stages(b);
                }
            }

            if !batch.is_empty() {
                result.push((pass_idx, batch));
            }
        }

        result
    }

    /// Resets all tracked resource states.
    ///
    /// Call this at the start of each frame to clear the previous frame's state.
    pub fn reset(&mut self) {
        self.resource_states.clear();
    }

    /// Returns the number of tracked resources.
    #[inline]
    pub fn resource_count(&self) -> usize {
        self.resource_states.len()
    }

    /// Returns an iterator over all tracked resources and their states.
    pub fn iter(&self) -> impl Iterator<Item = (&ResourceHandle, &ResourceState)> {
        self.resource_states.iter()
    }

    /// Returns true if any resources are being tracked.
    #[inline]
    pub fn is_tracking(&self) -> bool {
        !self.resource_states.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- SubresourceRange tests --

    #[test]
    fn test_subresource_range_all() {
        let range = SubresourceRange::all();
        assert!(range.is_all());
        assert_eq!(range.base_mip, 0);
        assert_eq!(range.mip_count, 0);
        assert_eq!(range.base_layer, 0);
        assert_eq!(range.layer_count, 0);
    }

    #[test]
    fn test_subresource_range_single_mip() {
        let range = SubresourceRange::single_mip(3);
        assert!(!range.is_all());
        assert_eq!(range.base_mip, 3);
        assert_eq!(range.mip_count, 1);
        assert_eq!(range.base_layer, 0);
        assert_eq!(range.layer_count, 0);
    }

    #[test]
    fn test_subresource_range_single_layer() {
        let range = SubresourceRange::single_layer(5);
        assert!(!range.is_all());
        assert_eq!(range.base_mip, 0);
        assert_eq!(range.mip_count, 0);
        assert_eq!(range.base_layer, 5);
        assert_eq!(range.layer_count, 1);
    }

    #[test]
    fn test_subresource_range_single() {
        let range = SubresourceRange::single(2, 4);
        assert!(!range.is_all());
        assert_eq!(range.base_mip, 2);
        assert_eq!(range.mip_count, 1);
        assert_eq!(range.base_layer, 4);
        assert_eq!(range.layer_count, 1);
    }

    #[test]
    fn test_subresource_range_overlaps_same() {
        let r1 = SubresourceRange::new(0, 4, 0, 6);
        let r2 = SubresourceRange::new(0, 4, 0, 6);
        assert!(r1.overlaps(&r2));
    }

    #[test]
    fn test_subresource_range_overlaps_partial() {
        let r1 = SubresourceRange::new(0, 4, 0, 4);
        let r2 = SubresourceRange::new(2, 4, 2, 4);
        assert!(r1.overlaps(&r2));
    }

    #[test]
    fn test_subresource_range_no_overlap_mip() {
        let r1 = SubresourceRange::new(0, 2, 0, 4);
        let r2 = SubresourceRange::new(3, 2, 0, 4);
        assert!(!r1.overlaps(&r2));
    }

    #[test]
    fn test_subresource_range_no_overlap_layer() {
        let r1 = SubresourceRange::new(0, 4, 0, 2);
        let r2 = SubresourceRange::new(0, 4, 3, 2);
        assert!(!r1.overlaps(&r2));
    }

    #[test]
    fn test_subresource_range_display() {
        let all = SubresourceRange::all();
        assert!(format!("{}", all).contains("all"));

        let specific = SubresourceRange::new(1, 3, 2, 4);
        let display = format!("{}", specific);
        assert!(display.contains("mips=1..4"));
        assert!(display.contains("layers=2..6"));
    }

    // -- PipelineStageFlags tests --

    #[test]
    fn test_pipeline_stage_flags_empty() {
        let flags = PipelineStageFlags::empty();
        assert!(flags.is_empty());
        assert!(!flags.contains(PipelineStageFlags::TOP));
    }

    #[test]
    fn test_pipeline_stage_flags_single() {
        let flags = PipelineStageFlags::VERTEX_SHADER;
        assert!(flags.contains(PipelineStageFlags::VERTEX_SHADER));
        assert!(!flags.contains(PipelineStageFlags::FRAGMENT_SHADER));
    }

    #[test]
    fn test_pipeline_stage_flags_combined() {
        let flags = PipelineStageFlags::VERTEX_SHADER | PipelineStageFlags::FRAGMENT_SHADER;
        assert!(flags.contains(PipelineStageFlags::VERTEX_SHADER));
        assert!(flags.contains(PipelineStageFlags::FRAGMENT_SHADER));
        assert!(!flags.contains(PipelineStageFlags::COMPUTE_SHADER));
    }

    #[test]
    fn test_pipeline_stage_flags_all_graphics() {
        let flags = PipelineStageFlags::ALL_GRAPHICS;
        assert!(flags.contains(PipelineStageFlags::ALL_GRAPHICS));
        assert!(!flags.contains(PipelineStageFlags::COMPUTE_SHADER));
    }

    #[test]
    fn test_pipeline_stage_flags_from_resource_state() {
        assert!(PipelineStageFlags::from_resource_state(ResourceState::ColorAttachment)
            .contains(PipelineStageFlags::COLOR_OUTPUT));
        assert!(PipelineStageFlags::from_resource_state(ResourceState::ShaderRead)
            .contains(PipelineStageFlags::FRAGMENT_SHADER));
        assert!(PipelineStageFlags::from_resource_state(ResourceState::TransferSrc)
            .contains(PipelineStageFlags::TRANSFER));
        assert!(PipelineStageFlags::from_resource_state(ResourceState::AccelerationStructure)
            .contains(PipelineStageFlags::RAY_TRACING));
    }

    #[test]
    fn test_pipeline_stage_flags_display() {
        let flags = PipelineStageFlags::VERTEX_SHADER | PipelineStageFlags::FRAGMENT_SHADER;
        let display = format!("{}", flags);
        assert!(display.contains("VERTEX_SHADER"));
        assert!(display.contains("FRAGMENT_SHADER"));
    }

    // -- BarrierType tests --

    #[test]
    fn test_barrier_type_texture_full() {
        let barrier = BarrierType::texture_full(
            ResourceHandle(1),
            ResourceState::ShaderRead,
            ResourceState::ColorAttachment,
        );

        assert!(barrier.is_texture());
        assert!(!barrier.is_buffer());
        assert!(barrier.has_state_change());
        assert_eq!(barrier.resource(), Some(ResourceHandle(1)));
    }

    #[test]
    fn test_barrier_type_buffer_full() {
        let barrier = BarrierType::buffer_full(
            ResourceHandle(2),
            ResourceState::TransferDst,
            ResourceState::ShaderRead,
        );

        assert!(barrier.is_buffer());
        assert!(!barrier.is_texture());
        assert!(barrier.has_state_change());
        assert_eq!(barrier.resource(), Some(ResourceHandle(2)));
    }

    #[test]
    fn test_barrier_type_memory() {
        let barrier = BarrierType::Memory;
        assert!(barrier.is_memory());
        assert!(!barrier.has_state_change());
        assert_eq!(barrier.resource(), None);
    }

    #[test]
    fn test_barrier_type_execution() {
        let barrier = BarrierType::Execution;
        assert!(barrier.is_execution());
        assert!(!barrier.has_state_change());
        assert_eq!(barrier.resource(), None);
    }

    #[test]
    fn test_barrier_type_no_state_change() {
        let barrier = BarrierType::texture_full(
            ResourceHandle(1),
            ResourceState::ShaderRead,
            ResourceState::ShaderRead,
        );
        assert!(!barrier.has_state_change());
    }

    #[test]
    fn test_barrier_type_display() {
        let texture = BarrierType::texture_full(
            ResourceHandle(1),
            ResourceState::ShaderRead,
            ResourceState::ColorAttachment,
        );
        let display = format!("{}", texture);
        assert!(display.contains("Texture"));
        assert!(display.contains("ShaderRead"));
        assert!(display.contains("ColorAttachment"));

        let buffer = BarrierType::Buffer {
            resource: ResourceHandle(2),
            old_state: ResourceState::TransferDst,
            new_state: ResourceState::ShaderRead,
            offset: 64,
            size: 256,
        };
        let display = format!("{}", buffer);
        assert!(display.contains("Buffer"));
        assert!(display.contains("offset=64"));
        assert!(display.contains("size=256"));
    }

    // -- BarrierBatch tests --

    #[test]
    fn test_barrier_batch_new() {
        let batch = BarrierBatch::new();
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
    }

    #[test]
    fn test_barrier_batch_add() {
        let mut batch = BarrierBatch::new();
        batch.add(BarrierType::Memory);
        assert!(!batch.is_empty());
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_barrier_batch_add_with_stages() {
        let mut batch = BarrierBatch::new();
        batch.add_with_stages(BarrierType::texture_full(
            ResourceHandle(1),
            ResourceState::ColorAttachment,
            ResourceState::ShaderRead,
        ));

        assert!(batch.source_stage.contains(PipelineStageFlags::COLOR_OUTPUT));
        assert!(batch.dest_stage.contains(PipelineStageFlags::FRAGMENT_SHADER));
    }

    #[test]
    fn test_barrier_batch_merge() {
        let mut batch1 = BarrierBatch::with_stages(
            PipelineStageFlags::VERTEX_SHADER,
            PipelineStageFlags::FRAGMENT_SHADER,
        );
        batch1.add(BarrierType::Memory);

        let mut batch2 = BarrierBatch::with_stages(
            PipelineStageFlags::COMPUTE_SHADER,
            PipelineStageFlags::TRANSFER,
        );
        batch2.add(BarrierType::Execution);

        batch1.merge(batch2);

        assert_eq!(batch1.len(), 2);
        assert!(batch1.source_stage.contains(PipelineStageFlags::VERTEX_SHADER));
        assert!(batch1.source_stage.contains(PipelineStageFlags::COMPUTE_SHADER));
        assert!(batch1.dest_stage.contains(PipelineStageFlags::FRAGMENT_SHADER));
        assert!(batch1.dest_stage.contains(PipelineStageFlags::TRANSFER));
    }

    #[test]
    fn test_barrier_batch_counts() {
        let mut batch = BarrierBatch::new();
        batch.add(BarrierType::texture_full(
            ResourceHandle(1),
            ResourceState::ShaderRead,
            ResourceState::ColorAttachment,
        ));
        batch.add(BarrierType::buffer_full(
            ResourceHandle(2),
            ResourceState::TransferDst,
            ResourceState::ShaderRead,
        ));
        batch.add(BarrierType::Memory);

        assert_eq!(batch.texture_count(), 1);
        assert_eq!(batch.buffer_count(), 1);
        assert_eq!(batch.len(), 3);
    }

    // -- BarrierResolver tests --

    #[test]
    fn test_barrier_resolver_new() {
        let resolver = BarrierResolver::new();
        assert!(!resolver.is_tracking());
        assert_eq!(resolver.resource_count(), 0);
    }

    #[test]
    fn test_barrier_resolver_get_state_uninitialized() {
        let resolver = BarrierResolver::new();
        assert_eq!(
            resolver.get_state(ResourceHandle(1)),
            ResourceState::Uninitialized
        );
    }

    #[test]
    fn test_barrier_resolver_set_state() {
        let mut resolver = BarrierResolver::new();
        resolver.set_state(ResourceHandle(1), ResourceState::ShaderRead);
        assert_eq!(
            resolver.get_state(ResourceHandle(1)),
            ResourceState::ShaderRead
        );
    }

    #[test]
    fn test_barrier_resolver_transition() {
        let mut resolver = BarrierResolver::new();
        resolver.set_state(ResourceHandle(1), ResourceState::ShaderRead);

        let barrier = resolver.transition(ResourceHandle(1), ResourceState::ColorAttachment);
        assert!(barrier.is_some());

        let b = barrier.unwrap();
        assert!(b.is_texture());
        assert_eq!(resolver.get_state(ResourceHandle(1)), ResourceState::ColorAttachment);
    }

    #[test]
    fn test_barrier_resolver_transition_no_change() {
        let mut resolver = BarrierResolver::new();
        resolver.set_state(ResourceHandle(1), ResourceState::ShaderRead);

        let barrier = resolver.transition(ResourceHandle(1), ResourceState::ShaderRead);
        assert!(barrier.is_none());
    }

    #[test]
    fn test_barrier_resolver_transition_texture() {
        let mut resolver = BarrierResolver::new();
        resolver.set_state(ResourceHandle(1), ResourceState::ColorAttachment);

        let barrier = resolver.transition_texture(
            ResourceHandle(1),
            ResourceState::ShaderRead,
            SubresourceRange::single_mip(0),
        );

        assert!(barrier.is_some());
        if let Some(BarrierType::Texture { subresource, .. }) = barrier {
            assert_eq!(subresource.base_mip, 0);
            assert_eq!(subresource.mip_count, 1);
        } else {
            panic!("Expected texture barrier");
        }
    }

    #[test]
    fn test_barrier_resolver_transition_buffer() {
        let mut resolver = BarrierResolver::new();
        resolver.set_state(ResourceHandle(2), ResourceState::TransferDst);

        let barrier = resolver.transition_buffer(
            ResourceHandle(2),
            ResourceState::ShaderRead,
            64,
            256,
        );

        assert!(barrier.is_some());
        if let Some(BarrierType::Buffer { offset, size, .. }) = barrier {
            assert_eq!(offset, 64);
            assert_eq!(size, 256);
        } else {
            panic!("Expected buffer barrier");
        }
    }

    #[test]
    fn test_barrier_resolver_reset() {
        let mut resolver = BarrierResolver::new();
        resolver.set_state(ResourceHandle(1), ResourceState::ShaderRead);
        resolver.set_state(ResourceHandle(2), ResourceState::ColorAttachment);
        assert_eq!(resolver.resource_count(), 2);

        resolver.reset();
        assert_eq!(resolver.resource_count(), 0);
        assert!(!resolver.is_tracking());
    }

    #[test]
    fn test_barrier_resolver_resolve_passes() {
        let mut resolver = BarrierResolver::new();

        // Pre-initialize some resources
        resolver.set_state(ResourceHandle(1), ResourceState::Uninitialized);
        resolver.set_state(ResourceHandle(2), ResourceState::Uninitialized);

        let passes = vec![PassIndex(0), PassIndex(1), PassIndex(2)];

        let results = resolver.resolve_passes(&passes, |pass_idx| {
            match pass_idx.0 {
                0 => vec![
                    (ResourceHandle(1), ResourceState::ColorAttachment, true),
                ],
                1 => vec![
                    (ResourceHandle(1), ResourceState::ShaderRead, true),
                    (ResourceHandle(2), ResourceState::TransferDst, false),
                ],
                2 => vec![
                    (ResourceHandle(2), ResourceState::ShaderRead, false),
                ],
                _ => vec![],
            }
        });

        // All three passes should have barriers
        assert_eq!(results.len(), 3);

        // Pass 0: Initialize resource 1 to ColorAttachment
        assert_eq!(results[0].0, PassIndex(0));
        assert_eq!(results[0].1.len(), 1);

        // Pass 1: Transition resource 1 to ShaderRead, resource 2 to TransferDst
        assert_eq!(results[1].0, PassIndex(1));
        assert_eq!(results[1].1.len(), 2);

        // Pass 2: Transition resource 2 to ShaderRead
        assert_eq!(results[2].0, PassIndex(2));
        assert_eq!(results[2].1.len(), 1);
    }

    #[test]
    fn test_barrier_resolver_iter() {
        let mut resolver = BarrierResolver::new();
        resolver.set_state(ResourceHandle(1), ResourceState::ShaderRead);
        resolver.set_state(ResourceHandle(2), ResourceState::ColorAttachment);

        let states: Vec<_> = resolver.iter().collect();
        assert_eq!(states.len(), 2);
    }

    #[test]
    fn test_barrier_resolver_with_states() {
        let mut initial = HashMap::new();
        initial.insert(ResourceHandle(1), ResourceState::ShaderRead);
        initial.insert(ResourceHandle(2), ResourceState::ColorAttachment);

        let resolver = BarrierResolver::with_states(initial);
        assert_eq!(resolver.resource_count(), 2);
        assert_eq!(resolver.get_state(ResourceHandle(1)), ResourceState::ShaderRead);
        assert_eq!(resolver.get_state(ResourceHandle(2)), ResourceState::ColorAttachment);
    }
}
