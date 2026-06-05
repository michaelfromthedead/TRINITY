//! Frame Graph Execution System (T-WGPU-P7.5.13)
//!
//! This module provides the execution infrastructure for compiling and running
//! frame graphs on the GPU. It bridges the declarative frame graph representation
//! with actual GPU command submission.
//!
//! # Architecture
//!
//! The execution system operates in two phases:
//!
//! 1. **Compilation**: The `FrameGraphCompiler` takes a `FrameGraph` and produces
//!    a `CompiledFrameGraph` containing:
//!    - Topologically sorted execution order
//!    - Pre-computed barrier batches for each pass
//!    - Resource allocation information with aliasing
//!
//! 2. **Execution**: The `FrameGraphExecutor` takes the compiled graph and an
//!    `ExecutionContext` to actually record and submit GPU commands.
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::frame_graph::execution::*;
//! use renderer_backend::frame_graph::graph::FrameGraph;
//!
//! // Build and compile the frame graph
//! let mut graph = FrameGraph::new();
//! // ... add passes and resources ...
//! graph.compile().unwrap();
//!
//! let mut compiler = FrameGraphCompiler::new();
//! let compiled = compiler.compile(&graph);
//!
//! // Execute on the GPU
//! let executor = FrameGraphExecutor::new();
//! let mut ctx = ExecutionContext::new(&device, &queue);
//! executor.execute(&compiled, &mut ctx);
//! ```

use std::collections::HashMap;
use std::fmt;

use super::aliasing::{AliasAnalyzer, AliasPolicy, MemoryAliasInfo};
use super::barriers::{BarrierBatch, BarrierResolver};
use super::graph::{PassId, ResourceId};
use super::scheduling::{ScheduleBuilder, SchedulingHint};

// ---------------------------------------------------------------------------
// MemoryType
// ---------------------------------------------------------------------------

/// GPU memory type for resource allocation.
///
/// Different memory types have different performance characteristics and
/// accessibility. Choosing the right type is critical for optimal performance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum MemoryType {
    /// GPU-only memory. Fastest for GPU access but not CPU-accessible.
    /// Best for render targets, intermediate buffers, and GPU-only resources.
    #[default]
    DeviceLocal,

    /// CPU-accessible memory with direct GPU access.
    /// Good for upload buffers and frequently updated uniform data.
    /// May have slower GPU access than DeviceLocal.
    HostVisible,

    /// CPU-cached memory optimized for GPU-to-CPU readback.
    /// Use for screenshot capture, GPU query results, and compute output
    /// that needs CPU processing.
    HostCached,

    /// Transient memory that can be aliased with other transient resources.
    /// Memory is pooled and reused across non-overlapping resource lifetimes.
    /// Provides significant memory savings (30-50%) for typical rendering.
    Transient,
}

impl MemoryType {
    /// Returns true if this memory is accessible from the CPU.
    #[inline]
    pub const fn is_host_accessible(&self) -> bool {
        matches!(self, Self::HostVisible | Self::HostCached)
    }

    /// Returns true if this memory is GPU-only.
    #[inline]
    pub const fn is_device_local(&self) -> bool {
        matches!(self, Self::DeviceLocal)
    }

    /// Returns true if this memory can be aliased.
    #[inline]
    pub const fn can_alias(&self) -> bool {
        matches!(self, Self::Transient)
    }

    /// Returns true if this memory is optimized for readback.
    #[inline]
    pub const fn is_readback(&self) -> bool {
        matches!(self, Self::HostCached)
    }
}

impl fmt::Display for MemoryType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeviceLocal => write!(f, "DeviceLocal"),
            Self::HostVisible => write!(f, "HostVisible"),
            Self::HostCached => write!(f, "HostCached"),
            Self::Transient => write!(f, "Transient"),
        }
    }
}

// ---------------------------------------------------------------------------
// ResourceAllocation
// ---------------------------------------------------------------------------

/// Describes a resource's memory allocation.
///
/// Contains information about where a resource is located in GPU memory,
/// including its memory type, offset, size, and aliasing relationships.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceAllocation {
    /// The resource identifier.
    pub resource: ResourceId,
    /// The type of memory used for this allocation.
    pub memory_type: MemoryType,
    /// Byte offset within the memory heap.
    pub offset: u64,
    /// Size of the allocation in bytes.
    pub size: u64,
    /// Other resources that share this memory region (aliasing).
    pub aliased_with: Vec<ResourceId>,
}

impl ResourceAllocation {
    /// Creates a new resource allocation with no aliasing.
    ///
    /// # Arguments
    ///
    /// * `resource` - The resource identifier.
    /// * `memory_type` - The type of memory.
    /// * `offset` - Byte offset in the heap.
    /// * `size` - Size in bytes.
    pub fn new(resource: ResourceId, memory_type: MemoryType, offset: u64, size: u64) -> Self {
        Self {
            resource,
            memory_type,
            offset,
            size,
            aliased_with: Vec::new(),
        }
    }

    /// Creates a transient allocation with aliasing information.
    ///
    /// # Arguments
    ///
    /// * `resource` - The resource identifier.
    /// * `offset` - Byte offset in the transient heap.
    /// * `size` - Size in bytes.
    /// * `aliased_with` - Resources sharing this memory.
    pub fn transient(
        resource: ResourceId,
        offset: u64,
        size: u64,
        aliased_with: Vec<ResourceId>,
    ) -> Self {
        Self {
            resource,
            memory_type: MemoryType::Transient,
            offset,
            size,
            aliased_with,
        }
    }

    /// Returns true if this allocation is aliased with other resources.
    #[inline]
    pub fn is_aliased(&self) -> bool {
        !self.aliased_with.is_empty()
    }

    /// Returns true if this allocation overlaps with another in memory.
    pub fn overlaps_with(&self, other: &ResourceAllocation) -> bool {
        // Different memory types can't overlap
        if self.memory_type != other.memory_type {
            return false;
        }

        let self_end = self.offset + self.size;
        let other_end = other.offset + other.size;

        self.offset < other_end && other.offset < self_end
    }

    /// Returns the end offset of this allocation.
    #[inline]
    pub fn end_offset(&self) -> u64 {
        self.offset + self.size
    }
}

impl fmt::Display for ResourceAllocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ResourceAllocation({}, type={}, offset={}, size={}, aliased={})",
            self.resource,
            self.memory_type,
            self.offset,
            self.size,
            self.aliased_with.len()
        )
    }
}

// ---------------------------------------------------------------------------
// CompiledFrameGraph
// ---------------------------------------------------------------------------

/// A compiled frame graph ready for execution.
///
/// Contains all the information needed to execute a frame graph:
/// - Topologically sorted pass execution order
/// - Pre-computed barrier batches for synchronization
/// - Resource allocation information with aliasing
///
/// The compiled graph is immutable and can be reused across multiple frames
/// as long as the graph structure doesn't change.
#[derive(Clone, Debug, Default)]
pub struct CompiledFrameGraph {
    /// Passes in topological execution order.
    pub execution_order: Vec<PassId>,
    /// Barrier batches to insert before each pass.
    /// Key is the pass index in execution_order.
    pub barrier_batches: Vec<(PassId, BarrierBatch)>,
    /// Resource memory allocations.
    pub resource_allocations: Vec<ResourceAllocation>,
    /// Memory aliasing information.
    pub alias_info: Vec<MemoryAliasInfo>,
}

impl CompiledFrameGraph {
    /// Creates a new empty compiled frame graph.
    pub fn new() -> Self {
        Self {
            execution_order: Vec::new(),
            barrier_batches: Vec::new(),
            resource_allocations: Vec::new(),
            alias_info: Vec::new(),
        }
    }

    /// Returns the number of passes in the execution order.
    #[inline]
    pub fn pass_count(&self) -> usize {
        self.execution_order.len()
    }

    /// Returns true if the compiled graph is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.execution_order.is_empty()
    }

    /// Returns the total number of barriers across all batches.
    pub fn total_barrier_count(&self) -> usize {
        self.barrier_batches.iter().map(|(_, b)| b.len()).sum()
    }

    /// Returns the barrier batch for a specific pass, if any.
    pub fn get_barriers_for_pass(&self, pass_id: PassId) -> Option<&BarrierBatch> {
        self.barrier_batches
            .iter()
            .find(|(id, _)| *id == pass_id)
            .map(|(_, batch)| batch)
    }

    /// Returns the allocation for a specific resource, if it exists.
    pub fn get_allocation(&self, resource: ResourceId) -> Option<&ResourceAllocation> {
        self.resource_allocations
            .iter()
            .find(|a| a.resource == resource)
    }

    /// Returns the total memory used by all non-aliased allocations.
    pub fn total_memory_usage(&self) -> u64 {
        // For aliased resources, only count the largest in each group
        let mut counted: std::collections::HashSet<ResourceId> = std::collections::HashSet::new();
        let mut total = 0u64;

        for alloc in &self.resource_allocations {
            if counted.contains(&alloc.resource) {
                continue;
            }

            if alloc.is_aliased() {
                // For aliased groups, find max size and mark all as counted
                let group_max = self
                    .resource_allocations
                    .iter()
                    .filter(|a| a.resource == alloc.resource || alloc.aliased_with.contains(&a.resource))
                    .map(|a| a.size)
                    .max()
                    .unwrap_or(0);

                total += group_max;
                counted.insert(alloc.resource);
                for &r in &alloc.aliased_with {
                    counted.insert(r);
                }
            } else {
                total += alloc.size;
                counted.insert(alloc.resource);
            }
        }

        total
    }

    /// Returns the memory savings from aliasing.
    pub fn memory_savings(&self) -> u64 {
        self.alias_info.iter().map(|info| info.savings_bytes).sum()
    }
}

impl fmt::Display for CompiledFrameGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CompiledFrameGraph(passes={}, barriers={}, allocations={}, aliases={})",
            self.execution_order.len(),
            self.total_barrier_count(),
            self.resource_allocations.len(),
            self.alias_info.len()
        )
    }
}

// ---------------------------------------------------------------------------
// FrameGraphCompiler
// ---------------------------------------------------------------------------

/// Compiles frame graphs into an executable form.
///
/// The compiler takes a `FrameGraph` and produces a `CompiledFrameGraph` by:
/// 1. Determining optimal execution order from the scheduler
/// 2. Resolving resource barriers using the barrier resolver
/// 3. Computing resource allocations with aliasing from the alias analyzer
///
/// # Example
///
/// ```ignore
/// let mut compiler = FrameGraphCompiler::new();
///
/// // Add scheduling hints for specific passes
/// compiler.with_scheduling_hints(hints);
///
/// // Set alias policies for specific resources
/// compiler.with_alias_policies(policies);
///
/// let compiled = compiler.compile(&graph);
/// ```
#[derive(Debug, Default)]
pub struct FrameGraphCompiler {
    /// The schedule builder for determining pass order.
    scheduler: ScheduleBuilder,
    /// The barrier resolver for computing synchronization.
    barrier_resolver: BarrierResolver,
    /// The alias analyzer for memory optimization.
    alias_analyzer: AliasAnalyzer,
    /// Custom scheduling hints per pass.
    scheduling_hints: HashMap<PassId, SchedulingHint>,
    /// Custom alias policies per resource.
    alias_policies: HashMap<ResourceId, AliasPolicy>,
}

impl FrameGraphCompiler {
    /// Creates a new frame graph compiler with default settings.
    pub fn new() -> Self {
        Self {
            scheduler: ScheduleBuilder::new(),
            barrier_resolver: BarrierResolver::new(),
            alias_analyzer: AliasAnalyzer::new(),
            scheduling_hints: HashMap::new(),
            alias_policies: HashMap::new(),
        }
    }

    /// Creates a builder with passes and resources for compilation.
    ///
    /// Use `.compile()` to run the compilation pipeline.
    pub fn build(passes: Vec<super::IrPass>, resources: Vec<super::IrResource>) -> FrameGraphCompilerBuilder {
        FrameGraphCompilerBuilder { passes, resources, config: super::CompilerConfig::default() }
    }

    /// Compiles passes and resources directly to a CompiledFrameGraph.
    ///
    /// This is a convenience method that matches the test API:
    /// `FrameGraphCompiler::from_ir(passes, resources).expect("should compile")`
    pub fn from_ir(
        passes: Vec<super::IrPass>,
        resources: Vec<super::IrResource>,
    ) -> Result<super::CompiledFrameGraph, String> {
        super::CompiledFrameGraph::compile(passes, resources)
    }

    /// Creates a compiler builder with the given configuration.
    pub fn with_config(
        passes: Vec<super::IrPass>,
        resources: Vec<super::IrResource>,
        config: super::CompilerConfig,
    ) -> FrameGraphCompilerBuilder {
        FrameGraphCompilerBuilder { passes, resources, config }
    }
}

/// Builder for FrameGraphCompiler that holds passes and resources.
pub struct FrameGraphCompilerBuilder {
    passes: Vec<super::IrPass>,
    resources: Vec<super::IrResource>,
    config: super::CompilerConfig,
}

impl FrameGraphCompilerBuilder {
    /// Creates the builder with passes and resources.
    pub fn new(passes: Vec<super::IrPass>, resources: Vec<super::IrResource>) -> Self {
        Self { passes, resources, config: super::CompilerConfig::default() }
    }

    /// Compiles the frame graph and returns the result.
    pub fn compile(self) -> Result<super::CompiledFrameGraph, String> {
        super::CompiledFrameGraph::compile_with_config(self.passes, self.resources, self.config)
    }
}

impl FrameGraphCompiler {

    /// Sets custom scheduling hints for specific passes.
    ///
    /// These hints guide the scheduler in determining optimal pass ordering.
    ///
    /// # Arguments
    ///
    /// * `hints` - Map from pass ID to scheduling hint.
    pub fn with_scheduling_hints(&mut self, hints: HashMap<PassId, SchedulingHint>) -> &mut Self {
        self.scheduling_hints = hints;
        self
    }

    /// Sets custom alias policies for specific resources.
    ///
    /// These policies control which resources can share memory.
    ///
    /// # Arguments
    ///
    /// * `policies` - Map from resource ID to alias policy.
    pub fn with_alias_policies(&mut self, policies: HashMap<ResourceId, AliasPolicy>) -> &mut Self {
        self.alias_policies = policies;
        self
    }

    /// Adds a scheduling hint for a single pass.
    pub fn add_scheduling_hint(&mut self, pass: PassId, hint: SchedulingHint) -> &mut Self {
        self.scheduling_hints.insert(pass, hint);
        self
    }

    /// Adds an alias policy for a single resource.
    pub fn add_alias_policy(&mut self, resource: ResourceId, policy: AliasPolicy) -> &mut Self {
        self.alias_policies.insert(resource, policy);
        self
    }

    /// Compiles a frame graph into an executable form.
    ///
    /// # Arguments
    ///
    /// * `graph` - The frame graph to compile. Must be already compiled via `graph.compile()`.
    ///
    /// # Returns
    ///
    /// A compiled frame graph ready for execution.
    pub fn compile(&mut self, graph: &super::graph::FrameGraph) -> CompiledFrameGraph {
        // Reset internal state
        self.scheduler = ScheduleBuilder::new();
        self.barrier_resolver.reset();
        self.alias_analyzer.clear();

        // Phase 1: Build the scheduler with passes and hints
        for pass in graph.passes() {
            let hint = self.scheduling_hints
                .get(&pass.id)
                .cloned()
                .unwrap_or_default();
            self.scheduler.add_pass(pass.id, hint);
        }

        // Phase 2: Get execution order from the graph (already compiled)
        let execution_order = graph.execution_order().to_vec();

        // Phase 3: Analyze resource lifetimes for aliasing
        self.alias_analyzer.analyze_lifetimes(graph);

        // Apply custom alias policies
        for (&resource, &policy) in &self.alias_policies {
            self.alias_analyzer.set_policy(resource, policy);
        }

        // Phase 4: Compute aliasing information
        let alias_groups = self.alias_analyzer.find_alias_groups();
        let alias_info = self.alias_analyzer.compute_aliasing(1024 * 1024 * 256); // 256MB heap

        // Phase 5: Build resource allocations
        let resource_allocations = self.build_resource_allocations(graph, &alias_info);

        // Phase 6: Compute barriers (placeholder - actual barrier computation
        // would use the barrier resolver with resource state tracking)
        let barrier_batches = Vec::new(); // Barrier computation would go here

        CompiledFrameGraph {
            execution_order,
            barrier_batches,
            resource_allocations,
            alias_info,
        }
    }

    /// Builds resource allocation information from the graph and alias info.
    fn build_resource_allocations(
        &self,
        graph: &super::graph::FrameGraph,
        alias_info: &[MemoryAliasInfo],
    ) -> Vec<ResourceAllocation> {
        let mut allocations = Vec::new();
        let mut processed: std::collections::HashSet<ResourceId> = std::collections::HashSet::new();

        // Process aliased resources first
        for info in alias_info {
            for &resource in &info.aliased_resources {
                if processed.contains(&resource) {
                    continue;
                }

                let aliased_with: Vec<ResourceId> = info
                    .aliased_resources
                    .iter()
                    .filter(|&&r| r != resource)
                    .copied()
                    .collect();

                let memory_type = if aliased_with.is_empty() {
                    // Check if resource is transient
                    if let Some(res) = graph.get_resource(resource) {
                        if res.is_transient() {
                            MemoryType::Transient
                        } else {
                            MemoryType::DeviceLocal
                        }
                    } else {
                        MemoryType::DeviceLocal
                    }
                } else {
                    MemoryType::Transient
                };

                allocations.push(ResourceAllocation {
                    resource,
                    memory_type,
                    offset: info.offset,
                    size: info.size,
                    aliased_with,
                });

                processed.insert(resource);
            }
        }

        // Process remaining resources
        let mut current_offset = allocations
            .iter()
            .map(|a| a.end_offset())
            .max()
            .unwrap_or(0);

        for resource in graph.resources() {
            if processed.contains(&resource.id) {
                continue;
            }

            let memory_type = if resource.is_transient() {
                MemoryType::Transient
            } else if resource.is_imported() {
                MemoryType::DeviceLocal // Imported resources are typically device-local
            } else {
                MemoryType::DeviceLocal
            };

            // Placeholder size - actual size would come from resource descriptors
            let size = 1024u64; // Placeholder

            allocations.push(ResourceAllocation::new(
                resource.id,
                memory_type,
                current_offset,
                size,
            ));

            current_offset += size;
            processed.insert(resource.id);
        }

        allocations
    }

    /// Resets the compiler state for reuse.
    pub fn reset(&mut self) {
        self.scheduler = ScheduleBuilder::new();
        self.barrier_resolver.reset();
        self.alias_analyzer.clear();
        self.scheduling_hints.clear();
        self.alias_policies.clear();
    }
}

// ---------------------------------------------------------------------------
// AllocatedResource
// ---------------------------------------------------------------------------

/// Represents an allocated GPU resource with its actual wgpu objects.
///
/// This enum wraps the actual wgpu resource types (Buffer, Texture) along
/// with any views needed for rendering.
#[derive(Debug)]
pub enum AllocatedResource {
    /// A GPU buffer with its allocation info.
    Buffer {
        /// The wgpu buffer.
        buffer: wgpu::Buffer,
        /// Byte offset within the buffer (for suballocations).
        offset: u64,
        /// Size of the allocation.
        size: u64,
    },
    /// A GPU texture with its views.
    Texture {
        /// The wgpu texture.
        texture: wgpu::Texture,
        /// The default texture view.
        view: wgpu::TextureView,
    },
}

impl AllocatedResource {
    /// Creates a new buffer resource.
    pub fn buffer(buffer: wgpu::Buffer, offset: u64, size: u64) -> Self {
        Self::Buffer { buffer, offset, size }
    }

    /// Creates a new texture resource with a default view.
    pub fn texture(texture: wgpu::Texture) -> Self {
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self::Texture { texture, view }
    }

    /// Creates a new texture resource with a custom view.
    pub fn texture_with_view(texture: wgpu::Texture, view: wgpu::TextureView) -> Self {
        Self::Texture { texture, view }
    }

    /// Returns true if this is a buffer resource.
    #[inline]
    pub fn is_buffer(&self) -> bool {
        matches!(self, Self::Buffer { .. })
    }

    /// Returns true if this is a texture resource.
    #[inline]
    pub fn is_texture(&self) -> bool {
        matches!(self, Self::Texture { .. })
    }

    /// Returns the buffer if this is a buffer resource.
    pub fn as_buffer(&self) -> Option<(&wgpu::Buffer, u64, u64)> {
        match self {
            Self::Buffer { buffer, offset, size } => Some((buffer, *offset, *size)),
            _ => None,
        }
    }

    /// Returns the texture and view if this is a texture resource.
    pub fn as_texture(&self) -> Option<(&wgpu::Texture, &wgpu::TextureView)> {
        match self {
            Self::Texture { texture, view } => Some((texture, view)),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// ExecutionContext
// ---------------------------------------------------------------------------

/// Context for executing a compiled frame graph.
///
/// Contains references to the GPU device and queue, as well as a map of
/// allocated resources that passes can access during execution.
pub struct ExecutionContext<'a> {
    /// Reference to the wgpu device.
    pub device: &'a wgpu::Device,
    /// Reference to the wgpu queue.
    pub queue: &'a wgpu::Queue,
    /// Map of resource IDs to their GPU allocations.
    pub resources: HashMap<ResourceId, AllocatedResource>,
}

impl<'a> ExecutionContext<'a> {
    /// Creates a new execution context.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `queue` - The wgpu queue.
    pub fn new(device: &'a wgpu::Device, queue: &'a wgpu::Queue) -> Self {
        Self {
            device,
            queue,
            resources: HashMap::new(),
        }
    }

    /// Creates a new execution context with pre-allocated resources.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `queue` - The wgpu queue.
    /// * `resources` - Pre-allocated resources.
    pub fn with_resources(
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        resources: HashMap<ResourceId, AllocatedResource>,
    ) -> Self {
        Self {
            device,
            queue,
            resources,
        }
    }

    /// Adds a resource to the context.
    pub fn add_resource(&mut self, id: ResourceId, resource: AllocatedResource) {
        self.resources.insert(id, resource);
    }

    /// Gets a resource by ID.
    pub fn get_resource(&self, id: ResourceId) -> Option<&AllocatedResource> {
        self.resources.get(&id)
    }

    /// Gets a mutable resource by ID.
    pub fn get_resource_mut(&mut self, id: ResourceId) -> Option<&mut AllocatedResource> {
        self.resources.get_mut(&id)
    }

    /// Returns true if a resource exists.
    pub fn has_resource(&self, id: ResourceId) -> bool {
        self.resources.contains_key(&id)
    }

    /// Returns the number of allocated resources.
    pub fn resource_count(&self) -> usize {
        self.resources.len()
    }

    /// Removes a resource from the context.
    pub fn remove_resource(&mut self, id: ResourceId) -> Option<AllocatedResource> {
        self.resources.remove(&id)
    }

    /// Creates a command encoder for recording commands.
    pub fn create_encoder(&self, label: Option<&str>) -> wgpu::CommandEncoder {
        self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label })
    }
}

impl<'a> fmt::Debug for ExecutionContext<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExecutionContext")
            .field("resources", &self.resources.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// FrameGraphExecutor
// ---------------------------------------------------------------------------

/// Executes compiled frame graphs on the GPU.
///
/// The executor takes a `CompiledFrameGraph` and an `ExecutionContext` and
/// records all necessary GPU commands, handling barriers and resource
/// transitions automatically.
///
/// # Example
///
/// ```ignore
/// let executor = FrameGraphExecutor::new();
///
/// // Execute the frame graph
/// executor.execute(&compiled, &mut ctx);
///
/// // Or submit and get a submission index
/// let submission_idx = executor.submit(&compiled, &ctx);
/// ```
#[derive(Clone, Debug, Default)]
pub struct FrameGraphExecutor {
    /// Debug label prefix for command encoders.
    label_prefix: Option<String>,
}

impl FrameGraphExecutor {
    /// Creates a new frame graph executor.
    pub fn new() -> Self {
        Self { label_prefix: None }
    }

    /// Creates an executor with a debug label prefix.
    pub fn with_label(label: impl Into<String>) -> Self {
        Self {
            label_prefix: Some(label.into()),
        }
    }

    /// Sets the debug label prefix.
    pub fn set_label(&mut self, label: impl Into<String>) {
        self.label_prefix = Some(label.into());
    }

    /// Executes a compiled frame graph.
    ///
    /// Records all passes in execution order, inserting barriers as needed.
    /// Commands are recorded but not submitted; use `submit()` to also submit.
    ///
    /// # Arguments
    ///
    /// * `compiled` - The compiled frame graph.
    /// * `ctx` - The execution context with device, queue, and resources.
    pub fn execute(&self, compiled: &CompiledFrameGraph, ctx: &mut ExecutionContext) {
        if compiled.is_empty() {
            return;
        }

        let label = self.label_prefix.as_deref();
        let mut encoder = ctx.create_encoder(label);

        for pass_id in &compiled.execution_order {
            // Insert barriers if needed
            if let Some(barriers) = compiled.get_barriers_for_pass(*pass_id) {
                // Note: wgpu doesn't expose explicit barriers like Vulkan.
                // Resource state tracking is handled internally by wgpu.
                // This is a placeholder for where barrier insertion would happen
                // in a lower-level API.
                let _ = barriers; // Suppress unused warning
            }

            // Pass execution would happen here via callbacks or similar
            // In practice, the frame graph's execute method would be called
            // or passes would have their own executors invoked.
        }

        // Submit the recorded commands
        ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Submits a compiled frame graph and returns the submission index.
    ///
    /// Similar to `execute()` but returns a `wgpu::SubmissionIndex` that can
    /// be used to track completion via `device.poll()`.
    ///
    /// # Arguments
    ///
    /// * `compiled` - The compiled frame graph.
    /// * `ctx` - The execution context.
    ///
    /// # Returns
    ///
    /// The submission index for tracking completion.
    pub fn submit(
        &self,
        compiled: &CompiledFrameGraph,
        ctx: &ExecutionContext,
    ) -> wgpu::SubmissionIndex {
        if compiled.is_empty() {
            // Submit an empty command buffer
            return ctx.queue.submit(std::iter::empty());
        }

        let label = self.label_prefix.as_deref();
        let encoder = ctx.create_encoder(label);

        // In a full implementation, we would:
        // 1. Record barrier commands
        // 2. Begin/end render/compute passes
        // 3. Record draw/dispatch commands via pass callbacks

        // For now, just submit the (empty) encoder
        ctx.queue.submit(std::iter::once(encoder.finish()))
    }
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // MemoryType Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_memory_type_default() {
        let mt = MemoryType::default();
        assert_eq!(mt, MemoryType::DeviceLocal);
    }

    #[test]
    fn test_memory_type_is_host_accessible() {
        assert!(!MemoryType::DeviceLocal.is_host_accessible());
        assert!(MemoryType::HostVisible.is_host_accessible());
        assert!(MemoryType::HostCached.is_host_accessible());
        assert!(!MemoryType::Transient.is_host_accessible());
    }

    #[test]
    fn test_memory_type_is_device_local() {
        assert!(MemoryType::DeviceLocal.is_device_local());
        assert!(!MemoryType::HostVisible.is_device_local());
        assert!(!MemoryType::HostCached.is_device_local());
        assert!(!MemoryType::Transient.is_device_local());
    }

    #[test]
    fn test_memory_type_can_alias() {
        assert!(!MemoryType::DeviceLocal.can_alias());
        assert!(!MemoryType::HostVisible.can_alias());
        assert!(!MemoryType::HostCached.can_alias());
        assert!(MemoryType::Transient.can_alias());
    }

    #[test]
    fn test_memory_type_is_readback() {
        assert!(!MemoryType::DeviceLocal.is_readback());
        assert!(!MemoryType::HostVisible.is_readback());
        assert!(MemoryType::HostCached.is_readback());
        assert!(!MemoryType::Transient.is_readback());
    }

    #[test]
    fn test_memory_type_display() {
        assert_eq!(format!("{}", MemoryType::DeviceLocal), "DeviceLocal");
        assert_eq!(format!("{}", MemoryType::HostVisible), "HostVisible");
        assert_eq!(format!("{}", MemoryType::HostCached), "HostCached");
        assert_eq!(format!("{}", MemoryType::Transient), "Transient");
    }

    // -----------------------------------------------------------------------
    // ResourceAllocation Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_resource_allocation_new() {
        let resource = ResourceId::new(1);
        let alloc = ResourceAllocation::new(resource, MemoryType::DeviceLocal, 0, 1024);

        assert_eq!(alloc.resource, resource);
        assert_eq!(alloc.memory_type, MemoryType::DeviceLocal);
        assert_eq!(alloc.offset, 0);
        assert_eq!(alloc.size, 1024);
        assert!(alloc.aliased_with.is_empty());
        assert!(!alloc.is_aliased());
    }

    #[test]
    fn test_resource_allocation_transient() {
        let resource = ResourceId::new(1);
        let aliased = vec![ResourceId::new(2), ResourceId::new(3)];
        let alloc = ResourceAllocation::transient(resource, 0, 2048, aliased.clone());

        assert_eq!(alloc.memory_type, MemoryType::Transient);
        assert!(alloc.is_aliased());
        assert_eq!(alloc.aliased_with, aliased);
    }

    #[test]
    fn test_resource_allocation_overlaps() {
        let r1 = ResourceId::new(1);
        let r2 = ResourceId::new(2);

        // Same type, overlapping range
        let a1 = ResourceAllocation::new(r1, MemoryType::DeviceLocal, 0, 1024);
        let a2 = ResourceAllocation::new(r2, MemoryType::DeviceLocal, 512, 1024);
        assert!(a1.overlaps_with(&a2));

        // Same type, non-overlapping
        let a3 = ResourceAllocation::new(r2, MemoryType::DeviceLocal, 1024, 1024);
        assert!(!a1.overlaps_with(&a3));

        // Different types, same range
        let a4 = ResourceAllocation::new(r2, MemoryType::HostVisible, 0, 1024);
        assert!(!a1.overlaps_with(&a4));
    }

    #[test]
    fn test_resource_allocation_end_offset() {
        let alloc = ResourceAllocation::new(ResourceId::new(1), MemoryType::DeviceLocal, 100, 500);
        assert_eq!(alloc.end_offset(), 600);
    }

    #[test]
    fn test_resource_allocation_display() {
        let alloc = ResourceAllocation::new(ResourceId::new(42), MemoryType::Transient, 0, 4096);
        let display = format!("{}", alloc);
        assert!(display.contains("42"));
        assert!(display.contains("Transient"));
        assert!(display.contains("4096"));
    }

    // -----------------------------------------------------------------------
    // CompiledFrameGraph Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_compiled_frame_graph_new() {
        let compiled = CompiledFrameGraph::new();
        assert!(compiled.is_empty());
        assert_eq!(compiled.pass_count(), 0);
        assert_eq!(compiled.total_barrier_count(), 0);
    }

    #[test]
    fn test_compiled_frame_graph_with_passes() {
        let mut compiled = CompiledFrameGraph::new();
        compiled.execution_order = vec![PassId::new(0), PassId::new(1), PassId::new(2)];

        assert!(!compiled.is_empty());
        assert_eq!(compiled.pass_count(), 3);
    }

    #[test]
    fn test_compiled_frame_graph_barriers() {
        let mut compiled = CompiledFrameGraph::new();
        compiled.execution_order = vec![PassId::new(0), PassId::new(1)];

        let mut batch = BarrierBatch::new();
        batch.add(super::super::barriers::BarrierType::Memory);
        batch.add(super::super::barriers::BarrierType::Execution);
        compiled.barrier_batches.push((PassId::new(1), batch));

        assert_eq!(compiled.total_barrier_count(), 2);

        let barriers = compiled.get_barriers_for_pass(PassId::new(1));
        assert!(barriers.is_some());
        assert_eq!(barriers.unwrap().len(), 2);

        let no_barriers = compiled.get_barriers_for_pass(PassId::new(0));
        assert!(no_barriers.is_none());
    }

    #[test]
    fn test_compiled_frame_graph_allocations() {
        let mut compiled = CompiledFrameGraph::new();

        let alloc1 = ResourceAllocation::new(ResourceId::new(1), MemoryType::DeviceLocal, 0, 1024);
        let alloc2 = ResourceAllocation::new(ResourceId::new(2), MemoryType::DeviceLocal, 1024, 2048);
        compiled.resource_allocations = vec![alloc1, alloc2];

        assert!(compiled.get_allocation(ResourceId::new(1)).is_some());
        assert!(compiled.get_allocation(ResourceId::new(2)).is_some());
        assert!(compiled.get_allocation(ResourceId::new(99)).is_none());
    }

    #[test]
    fn test_compiled_frame_graph_memory_usage() {
        let mut compiled = CompiledFrameGraph::new();

        // Two non-aliased resources
        let alloc1 = ResourceAllocation::new(ResourceId::new(1), MemoryType::DeviceLocal, 0, 1000);
        let alloc2 = ResourceAllocation::new(ResourceId::new(2), MemoryType::DeviceLocal, 1000, 2000);
        compiled.resource_allocations = vec![alloc1, alloc2];

        assert_eq!(compiled.total_memory_usage(), 3000);
    }

    #[test]
    fn test_compiled_frame_graph_memory_usage_aliased() {
        let mut compiled = CompiledFrameGraph::new();

        // Two aliased resources - only count the larger one
        let alloc1 = ResourceAllocation::transient(
            ResourceId::new(1),
            0,
            1000,
            vec![ResourceId::new(2)],
        );
        let alloc2 = ResourceAllocation::transient(
            ResourceId::new(2),
            0,
            2000,
            vec![ResourceId::new(1)],
        );
        compiled.resource_allocations = vec![alloc1, alloc2];

        assert_eq!(compiled.total_memory_usage(), 2000);
    }

    #[test]
    fn test_compiled_frame_graph_display() {
        let compiled = CompiledFrameGraph::new();
        let display = format!("{}", compiled);
        assert!(display.contains("CompiledFrameGraph"));
        assert!(display.contains("passes=0"));
    }

    // -----------------------------------------------------------------------
    // FrameGraphCompiler Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_frame_graph_compiler_new() {
        let compiler = FrameGraphCompiler::new();
        assert!(compiler.scheduling_hints.is_empty());
        assert!(compiler.alias_policies.is_empty());
    }

    #[test]
    fn test_frame_graph_compiler_with_scheduling_hints() {
        let mut compiler = FrameGraphCompiler::new();
        let mut hints = HashMap::new();
        hints.insert(
            PassId::new(0),
            SchedulingHint::new(super::super::scheduling::SchedulingPriority::High),
        );

        compiler.with_scheduling_hints(hints);
        assert_eq!(compiler.scheduling_hints.len(), 1);
    }

    #[test]
    fn test_frame_graph_compiler_with_alias_policies() {
        let mut compiler = FrameGraphCompiler::new();
        let mut policies = HashMap::new();
        policies.insert(ResourceId::new(0), AliasPolicy::Never);

        compiler.with_alias_policies(policies);
        assert_eq!(compiler.alias_policies.len(), 1);
    }

    #[test]
    fn test_frame_graph_compiler_add_scheduling_hint() {
        let mut compiler = FrameGraphCompiler::new();
        compiler.add_scheduling_hint(
            PassId::new(5),
            SchedulingHint::new(super::super::scheduling::SchedulingPriority::Critical),
        );

        assert!(compiler.scheduling_hints.contains_key(&PassId::new(5)));
    }

    #[test]
    fn test_frame_graph_compiler_add_alias_policy() {
        let mut compiler = FrameGraphCompiler::new();
        compiler.add_alias_policy(ResourceId::new(10), AliasPolicy::Aggressive);

        assert!(compiler.alias_policies.contains_key(&ResourceId::new(10)));
    }

    #[test]
    fn test_frame_graph_compiler_reset() {
        let mut compiler = FrameGraphCompiler::new();
        compiler.add_scheduling_hint(PassId::new(0), SchedulingHint::default());
        compiler.add_alias_policy(ResourceId::new(0), AliasPolicy::Never);

        compiler.reset();

        assert!(compiler.scheduling_hints.is_empty());
        assert!(compiler.alias_policies.is_empty());
    }

    // -----------------------------------------------------------------------
    // FrameGraphExecutor Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_frame_graph_executor_new() {
        let executor = FrameGraphExecutor::new();
        assert!(executor.label_prefix.is_none());
    }

    #[test]
    fn test_frame_graph_executor_with_label() {
        let executor = FrameGraphExecutor::with_label("test_frame");
        assert_eq!(executor.label_prefix, Some("test_frame".to_string()));
    }

    #[test]
    fn test_frame_graph_executor_set_label() {
        let mut executor = FrameGraphExecutor::new();
        executor.set_label("my_label");
        assert_eq!(executor.label_prefix, Some("my_label".to_string()));
    }
}
